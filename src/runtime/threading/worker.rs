// =========================================================================
// Worker Pool and Thread Management
// Fixed-size thread pool with per-worker work-stealing deques
// Implements Michael-Scott MPMC queue for injectors
// =========================================================================

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use super::deque::WorkStealingDeque;
use super::epoch::{Guard, Participant};
use super::numa::{NumaTopology, num_cores};
use super::percpu_deque::PerCpuDeque;
use super::rseq::register_rseq;

/// Number of worker threads (one per CPU core)
fn num_workers() -> usize {
    num_cores()
}

/// Michael-Scott MPMC queue node
struct MpmcNode {
    data: *mut (),
    next: AtomicPtr<MpmcNode>,
}

/// Michael-Scott MPMC queue for injector
struct MpmcQueue {
    head: AtomicPtr<MpmcNode>,
    tail: AtomicPtr<MpmcNode>,
}

impl MpmcQueue {
    fn new() -> Self {
        let dummy = Box::into_raw(Box::new(MpmcNode {
            data: std::ptr::null_mut(),
            next: AtomicPtr::new(std::ptr::null_mut()),
        }));
        
        Self {
            head: AtomicPtr::new(dummy),
            tail: AtomicPtr::new(dummy),
        }
    }

    fn push(&self, task: *mut ()) {
        let node = Box::into_raw(Box::new(MpmcNode {
            data: task,
            next: AtomicPtr::new(std::ptr::null_mut()),
        }));

        loop {
            let tail = self.tail.load(Ordering::Acquire);
            let tail_node = unsafe { &*tail };
            let next = tail_node.next.load(Ordering::Acquire);

            if next.is_null() {
                if tail_node.next.compare_exchange_weak(
                    std::ptr::null_mut(),
                    node,
                    Ordering::Release,
                    Ordering::Acquire,
                ).is_ok() {
                    let _ = self.tail.compare_exchange_weak(
                        tail,
                        node,
                        Ordering::Release,
                        Ordering::Acquire,
                    );
                    return;
                }
            } else {
                let _ = self.tail.compare_exchange_weak(
                    tail,
                    next,
                    Ordering::Release,
                    Ordering::Acquire,
                );
            }
        }
    }

    fn try_pop(&self) -> Option<*mut ()> {
        loop {
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);
            let head_node = unsafe { &*head };
            let next = head_node.next.load(Ordering::Acquire);

            if head == tail {
                if next.is_null() {
                    return None;
                }
                let _ = self.tail.compare_exchange_weak(
                    tail,
                    next,
                    Ordering::Release,
                    Ordering::Acquire,
                );
            } else {
                let data = unsafe { (*next).data };
                if self.head.compare_exchange_weak(
                    head,
                    next,
                    Ordering::Release,
                    Ordering::Acquire,
                ).is_ok() {
                    // Free the dummy node
                    let _ = unsafe { Box::from_raw(head) };
                    return Some(data);
                }
            }
        }
    }
}

/// Notification mechanism for waking workers using futex-like semantics
#[allow(dead_code)]
struct Notify {
    flag: AtomicBool,
}

#[allow(dead_code)]
impl Notify {
    fn new() -> Self {
        Self {
            flag: AtomicBool::new(false),
        }
    }

    fn notify_one(&self) {
        self.flag.store(true, Ordering::Release);
    }

    fn notify_all(&self) {
        self.flag.store(true, Ordering::Release);
    }

    fn wait(&self) {
        while !self.flag.swap(false, Ordering::Acquire) {
            thread::sleep(Duration::from_micros(100));
        }
    }
}

/// Global injector queue for external task submission
struct Injector {
    queue: MpmcQueue,
    notify: Arc<Notify>,
}

impl Injector {
    fn new() -> Self {
        Self {
            queue: MpmcQueue::new(),
            notify: Arc::new(Notify::new()),
        }
    }

    fn push(&self, task: *mut ()) {
        self.queue.push(task);
        self.notify.notify_one();
    }

    fn try_pop(&self) -> Option<*mut ()> {
        self.queue.try_pop()
    }
}

/// Per-worker state with 128-byte alignment to prevent false sharing
#[repr(C, align(128))]
pub struct Worker {
    /// Worker ID
    id: usize,
    _pad1: [u8; 64],
    /// Local work-stealing deque (fallback)
    deque: WorkStealingDeque,
    _pad2: [u8; 64],
    /// Per-CPU deque (wait-fast with rseq)
    percpu_deque: Arc<PerCpuDeque>,
    _pad3: [u8; 64],
    /// Reference to global injector
    injector: Arc<Injector>,
    /// Reference to all workers (for stealing) — late-initialized via OnceLock
    workers: Arc<std::sync::OnceLock<Arc<Vec<Arc<Worker>>>>>,
    /// Epoch participant
    participant: Arc<Participant>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// NUMA node ID (if NUMA is enabled)
    numa_node: Option<usize>,
    /// Statistics
    tasks_executed: AtomicUsize,
    steals_performed: AtomicUsize,
}

impl Worker {
    /// Create a new worker
    fn new(
        id: usize,
        injector: Arc<Injector>,
        workers: Arc<std::sync::OnceLock<Arc<Vec<Arc<Worker>>>>>,
        participant: Arc<Participant>,
        shutdown: Arc<AtomicBool>,
        numa_node: Option<usize>,
        percpu_deque: Arc<PerCpuDeque>,
    ) -> Self {
        Self {
            id,
            _pad1: [0; 64],
            deque: WorkStealingDeque::new(participant.clone()),
            _pad2: [0; 64],
            percpu_deque,
            _pad3: [0; 64],
            injector,
            workers,
            participant,
            shutdown,
            numa_node,
            tasks_executed: AtomicUsize::new(0),
            steals_performed: AtomicUsize::new(0),
        }
    }

    /// Main worker loop with exponential backoff for idle periods
    fn run(&self) {
        // Register rseq for this thread
        let _ = register_rseq();
        
        let mut idle_iterations = 0;
        
        while !self.shutdown.load(Ordering::Acquire) {
            let guard = self.participant.pin();
            
            // Try to pop from per-CPU deque first (wait-free with rseq)
            if let Some(task) = self.percpu_deque.pop(&guard) {
                self.execute_task(task);
                idle_iterations = 0;
                continue;
            }
            
            // Try to pop from local deque (fallback)
            if let Some(task) = self.deque.pop(&guard) {
                self.execute_task(task);
                idle_iterations = 0;
                continue;
            }
            
            // Try to steal from injector
            if let Some(task) = self.injector.try_pop() {
                self.execute_task(task);
                idle_iterations = 0;
                continue;
            }
            
            // Try to steal from other workers
            if let Some(task) = self.steal(&guard) {
                self.execute_task(task);
                idle_iterations = 0;
                continue;
            }
            
            // No work found, exponential backoff
            idle_iterations += 1;
            let sleep_us = if idle_iterations < 10 {
                1
            } else if idle_iterations < 100 {
                10
            } else if idle_iterations < 1000 {
                100
            } else {
                1000
            };
            thread::sleep(Duration::from_micros(sleep_us));
        }
    }

    /// Steal work from other workers with NUMA-aware priority
    fn steal(&self, guard: &Guard) -> Option<*mut ()> {
        let workers = match self.workers.get() {
            Some(w) => w,
            None => return None,
        };
        let num_workers = workers.len();
        
        // NUMA-aware stealing: prefer same-node workers first
        if let Some(my_node) = self.numa_node {
            // Try same-node workers first
            for offset in 1..num_workers {
                let target_id = (self.id + offset) % num_workers;
                let target = &workers[target_id];
                
                if target.numa_node == Some(my_node) {
                    // Try per-CPU deque first
                    for task in target.percpu_deque.steal_half(target_id, guard) {
                        if !task.is_null() {
                            self.steals_performed.fetch_add(1, Ordering::Relaxed);
                            return Some(task);
                        }
                    }
                    // Fallback to regular deque
                    for task in target.deque.steal_half(guard) {
                        if !task.is_null() {
                            self.steals_performed.fetch_add(1, Ordering::Relaxed);
                            return Some(task);
                        }
                    }
                }
            }
            
            // Then try cross-node workers
            for offset in 1..num_workers {
                let target_id = (self.id + offset) % num_workers;
                let target = &workers[target_id];
                
                if target.numa_node != Some(my_node) {
                    for task in target.percpu_deque.steal_half(target_id, guard) {
                        if !task.is_null() {
                            self.steals_performed.fetch_add(1, Ordering::Relaxed);
                            return Some(task);
                        }
                    }
                    for task in target.deque.steal_half(guard) {
                        if !task.is_null() {
                            self.steals_performed.fetch_add(1, Ordering::Relaxed);
                            return Some(task);
                        }
                    }
                }
            }
        } else {
            // No NUMA info, steal from any worker
            for offset in 1..num_workers {
                let target_id = (self.id + offset) % num_workers;
                let target = &workers[target_id];
                
                for task in target.percpu_deque.steal_half(target_id, guard) {
                    if !task.is_null() {
                        self.steals_performed.fetch_add(1, Ordering::Relaxed);
                        return Some(task);
                    }
                }
                for task in target.deque.steal_half(guard) {
                    if !task.is_null() {
                        self.steals_performed.fetch_add(1, Ordering::Relaxed);
                        return Some(task);
                    }
                }
            }
        }
        
        None
    }

    /// Execute a task
    fn execute_task(&self, task: *mut ()) {
        self.tasks_executed.fetch_add(1, Ordering::Relaxed);
        unsafe {
            // Task is a Box<Box<dyn FnOnce()>> passed through *mut ().
            // Double-boxing is needed because Box<dyn FnOnce()> is a fat pointer
            // and cannot be stored in a single *mut ().
            let func: Box<Box<dyn FnOnce()>> = Box::from_raw(task as *mut Box<dyn FnOnce()>);
            (*func)();
        }
    }

    /// Push a task to this worker's deque
    pub fn push(&self, task: *mut ()) {
        let guard = self.participant.pin();
        // Try per-CPU deque first (wait-free with rseq)
        self.percpu_deque.push(task, &guard);
    }

    /// Get worker statistics
    pub fn stats(&self) -> WorkerStats {
        WorkerStats {
            id: self.id,
            tasks_executed: self.tasks_executed.load(Ordering::Relaxed),
            steals_performed: self.steals_performed.load(Ordering::Relaxed),
            deque_size: self.deque.len(),
        }
    }
}

/// Worker statistics
#[derive(Debug, Clone)]
pub struct WorkerStats {
    pub id: usize,
    pub tasks_executed: usize,
    pub steals_performed: usize,
    pub deque_size: usize,
}

/// Thread pool for managing workers
#[allow(dead_code)]
pub struct ThreadPool {
    workers: Arc<Vec<Arc<Worker>>>,
    injector: Arc<Injector>,
    shutdown: Arc<AtomicBool>,
    participant: Arc<Participant>,
}

impl ThreadPool {
    /// Create a new thread pool
    pub fn new() -> Self {
        let num_workers = num_workers();
        let participant = Arc::new(Participant::new());
        let injector = Arc::new(Injector::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        
        // Create per-CPU deque
        let percpu_deque = Arc::new(PerCpuDeque::new(participant.clone()));
        
        // Detect NUMA topology
        let topology = NumaTopology::detect();
        
        let mut workers = Vec::with_capacity(num_workers);
        
        for id in 0..num_workers {
            // Determine NUMA node for this worker
            let numa_node = topology.get_node_for_cpu(id).map(|n| n.id);
            
            let worker = Arc::new(Worker::new(
                id,
                injector.clone(),
                // Placeholder — will be set after all workers are created
                Arc::new(std::sync::OnceLock::new()),
                participant.clone(),
                shutdown.clone(),
                numa_node,
                percpu_deque.clone(),
            ));
            workers.push(worker);
        }
        
        // Build the shared worker list and initialize each worker's OnceLock
        let workers_arc = Arc::new(workers);
        for worker in workers_arc.iter() {
            let _ = worker.workers.set(workers_arc.clone());
        }
        
        // Spawn worker threads with CPU affinity
        for worker in workers_arc.iter() {
            let worker_clone = worker.clone();
            let _cpu_id = worker.id;
            
            thread::spawn(move || {
                // Set CPU affinity if NUMA is enabled
                #[cfg(feature = "numa")]
                let _ = set_thread_affinity(_cpu_id);
                
                worker_clone.run();
            });
        }
        
        Self {
            workers: workers_arc,
            injector,
            shutdown,
            participant,
        }
    }

    /// Submit a task to the global injector
    pub fn submit(&self, task: *mut ()) {
        self.injector.push(task);
    }

    /// Get the number of workers
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Get statistics for all workers
    pub fn stats(&self) -> Vec<WorkerStats> {
        self.workers.iter().map(|w| w.stats()).collect()
    }

    /// Shutdown the thread pool
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }
}

impl Default for ThreadPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_pool_creation() {
        let pool = ThreadPool::new();
        assert!(pool.num_workers() > 0);
    }

    #[test]
    fn test_num_workers() {
        let n = num_workers();
        assert!(n > 0);
    }

    #[test]
    fn test_worker_stats() {
        let pool = ThreadPool::new();
        let stats = pool.stats();
        assert_eq!(stats.len(), pool.num_workers());
    }
}
