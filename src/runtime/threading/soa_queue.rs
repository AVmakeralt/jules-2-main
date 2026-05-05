// =========================================================================
// Structure-of-Arrays (SoA) Task Queues
// Improves cache efficiency by storing task fields in separate arrays
// Enables SIMD priority scanning and batch operations
// Includes software prefetching and cache warming
// =========================================================================

use std::sync::atomic::{AtomicUsize, Ordering};
use std::ptr;
use std::cell::UnsafeCell;

/// Cache line size
const CACHE_LINE_SIZE: usize = 64;

/// Task function pointer type
type TaskFn = fn(*mut ());

/// SoA task queue
/// Stores task fields in separate contiguous arrays for better cache efficiency
pub struct SoaTaskQueue {
    /// Function pointers (separate array)
    functions: UnsafeCell<Vec<TaskFn>>,
    /// Data pointers (separate array)
    data: UnsafeCell<Vec<*mut ()>>,
    /// Priorities (separate array, for SIMD scanning)
    priorities: UnsafeCell<Vec<u8>>,
    /// Flags (separate array)
    flags: UnsafeCell<Vec<u8>>,
    /// Task types (separate array)
    task_types: UnsafeCell<Vec<u8>>,
    /// Head index
    head: AtomicUsize,
    /// Tail index
    tail: AtomicUsize,
    /// Capacity
    capacity: usize,
    /// Mask for modulo operation
    mask: usize,
}

// SAFETY: SoaTaskQueue uses atomic head/tail indices for synchronization and
// UnsafeCell for interior mutability of the data arrays. Access to the arrays
// is coordinated through the atomic indices, making it safe to share across threads.
unsafe impl Sync for SoaTaskQueue {}

impl SoaTaskQueue {
    /// Create a new SoA task queue
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        
        Self {
            functions: UnsafeCell::new(vec![|_| {}; capacity]),
            data: UnsafeCell::new(vec![ptr::null_mut(); capacity]),
            priorities: UnsafeCell::new(vec![255; capacity]),
            flags: UnsafeCell::new(vec![0; capacity]),
            task_types: UnsafeCell::new(vec![0; capacity]),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            capacity,
            mask: capacity - 1,
        }
    }
    
    /// Push a task to the queue
    pub fn push(&self, func: TaskFn, data: *mut (), priority: u8, task_type: u8) -> bool {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        
        if tail.wrapping_sub(head) >= self.capacity {
            return false; // Queue full
        }
        
        let idx = tail & self.mask;
        
        // Write to separate arrays using UnsafeCell for interior mutability
        // SAFETY: We're the only producer (single-producer), and we've verified there's space
        unsafe {
            let functions = (*self.functions.get()).as_mut_ptr();
            let data_arr = (*self.data.get()).as_mut_ptr();
            let priorities = (*self.priorities.get()).as_mut_ptr();
            let flags = (*self.flags.get()).as_mut_ptr();
            let task_types = (*self.task_types.get()).as_mut_ptr();
            
            *functions.add(idx) = func;
            *data_arr.add(idx) = data;
            *priorities.add(idx) = priority;
            *flags.add(idx) = 0;
            *task_types.add(idx) = task_type;
        }
        
        std::sync::atomic::fence(Ordering::Release);
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        
        true
    }
    
    /// Pop the highest-priority task
    /// Uses SIMD-like scanning of priority array
    pub fn pop_highest_priority(&self) -> Option<(TaskFn, *mut (), u8, u8)> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        
        if head >= tail {
            return None;
        }
        
        // Scan priority array for highest priority (lowest value)
        let mut best_idx = None;
        let mut best_priority = 255u8;
        
        let priorities = unsafe { &*self.priorities.get() };
        let functions = unsafe { &*self.functions.get() };
        let data = unsafe { &*self.data.get() };
        let task_types = unsafe { &*self.task_types.get() };
        
        for i in 0..(tail.wrapping_sub(head)) {
            let idx = (head.wrapping_add(i)) & self.mask;
            let p = priorities[idx];
            
            if p < best_priority {
                best_priority = p;
                best_idx = Some(idx);
                
                if p == 0 {
                    break; // Can't get better than 0
                }
            }
        }
        
        if let Some(idx) = best_idx {
            let func = functions[idx];
            let d = data[idx];
            let tt = task_types[idx];

            // Compact the queue after popping: shift remaining elements
            // after the popped position down by one and decrement tail.
            // This prevents consumed slots from leaking capacity.
            let count = tail.wrapping_sub(head);
            let head_pos = head & self.mask;

            if idx == head_pos {
                // Popped the head element — simply advance head
                self.head.store(head.wrapping_add(1), Ordering::Release);
            } else {
                // Popped from the middle or tail — shift elements after
                // idx down by one to fill the gap.
                let idx_logical = (idx.wrapping_sub(head_pos)) & self.mask;
                let remaining = count.saturating_sub(idx_logical).saturating_sub(1);

                // SAFETY: Single consumer pattern with atomic synchronization
                unsafe {
                    let functions_ptr = (*self.functions.get()).as_mut_ptr();
                    let data_arr = (*self.data.get()).as_mut_ptr();
                    let priorities_ptr = (*self.priorities.get()).as_mut_ptr();
                    let flags_ptr = (*self.flags.get()).as_mut_ptr();
                    let task_types_ptr = (*self.task_types.get()).as_mut_ptr();

                    for i in 0..remaining {
                        let src = (idx.wrapping_add(i).wrapping_add(1)) & self.mask;
                        let dst = (idx.wrapping_add(i)) & self.mask;
                        *functions_ptr.add(dst) = *functions_ptr.add(src);
                        *data_arr.add(dst) = *data_arr.add(src);
                        *priorities_ptr.add(dst) = *priorities_ptr.add(src);
                        *flags_ptr.add(dst) = *flags_ptr.add(src);
                        *task_types_ptr.add(dst) = *task_types_ptr.add(src);
                    }

                    // Clear the last slot (now vacant after shift)
                    let last = (idx.wrapping_add(remaining)) & self.mask;
                    *priorities_ptr.add(last) = 255;
                    *data_arr.add(last) = ptr::null_mut();
                }

                // Decrement tail to reflect the removed element
                self.tail.store(tail.wrapping_sub(1), Ordering::Release);
            }

            Some((func, d, best_priority, tt))
        } else {
            None
        }
    }
    
    /// Pop a task from the front (FIFO)
    pub fn pop_front(&self) -> Option<(TaskFn, *mut (), u8, u8)> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        
        if head >= tail {
            return None;
        }
        
        let idx = head & self.mask;
        let functions = unsafe { &*self.functions.get() };
        let data_arr = unsafe { &*self.data.get() };
        let priorities = unsafe { &*self.priorities.get() };
        let task_types = unsafe { &*self.task_types.get() };
        let func = functions[idx];
        let data = data_arr[idx];
        let priority = priorities[idx];
        let task_type = task_types[idx];
        
        self.head.store(head.wrapping_add(1), Ordering::Release);
        
        Some((func, data, priority, task_type))
    }
    
    /// Get the number of tasks in the queue
    pub fn len(&self) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }
    
    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get the capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Software prefetch instructions
pub fn prefetch_t0(addr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::asm!("prefetcht0 byte ptr [{0}]", in(reg) addr, options(nostack));
    }
}

pub fn prefetch_t1(addr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::asm!("prefetcht1 byte ptr [{0}]", in(reg) addr, options(nostack));
    }
}

pub fn prefetch_t2(addr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::asm!("prefetcht2 byte ptr [{0}]", in(reg) addr, options(nostack));
    }
}

pub fn prefetch_w(addr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::asm!("prefetchw byte ptr [{0}]", in(reg) addr, options(nostack));
    }
}

/// Cache warming for task data
pub fn warm_task_cache(task_data: *const u8, size: usize) {
    // Prefetch in cache lines
    for offset in (0..size).step_by(CACHE_LINE_SIZE) {
        prefetch_t0(unsafe { task_data.add(offset) });
    }
}

/// Cache warming for function code
pub fn warm_function_cache(func: TaskFn) {
    let func_ptr = func as *const ();
    prefetch_t0(func_ptr as *const u8);
    
    // Prefetch next few cache lines for instruction cache
    for offset in 1..4 {
        prefetch_t0(unsafe { (func_ptr as *const u8).add(offset * CACHE_LINE_SIZE) });
    }
}

/// Data-to-worker affinity map
/// Tracks which worker last accessed each data address
pub struct DataAffinityMap {
    /// Map from data address to worker ID
    map: std::collections::HashMap<usize, usize>,
    /// Maximum number of workers
    num_workers: usize,
}

impl DataAffinityMap {
    /// Create a new data affinity map
    pub fn new(num_workers: usize) -> Self {
        Self {
            map: std::collections::HashMap::new(),
            num_workers,
        }
    }
    
    /// Record that a worker accessed data at an address
    pub fn record_access(&mut self, addr: usize, worker_id: usize) {
        if worker_id < self.num_workers {
            self.map.insert(addr, worker_id);
        }
    }
    
    /// Get the preferred worker for a data address
    pub fn get_preferred_worker(&self, addr: usize) -> Option<usize> {
        self.map.get(&addr).copied()
    }
    
    /// Clear the map
    pub fn clear(&mut self) {
        self.map.clear();
    }
}

/// SoA scheduler with cache warming
#[allow(dead_code)]
pub struct SoaScheduler {
    /// Ready queue (SoA layout)
    ready_queue: SoaTaskQueue,
    /// Data affinity map
    affinity_map: DataAffinityMap,
    /// Number of workers
    num_workers: usize,
    /// Enable cache warming
    enable_cache_warming: bool,
}

impl SoaScheduler {
    /// Create a new SoA scheduler
    pub fn new(capacity: usize, num_workers: usize) -> Self {
        Self {
            ready_queue: SoaTaskQueue::new(capacity),
            affinity_map: DataAffinityMap::new(num_workers),
            num_workers,
            enable_cache_warming: true,
        }
    }
    
    /// Enable or disable cache warming
    pub fn set_cache_warming(&mut self, enabled: bool) {
        self.enable_cache_warming = enabled;
    }
    
    /// Schedule a task
    pub fn schedule(&mut self, func: TaskFn, data: *mut (), priority: u8, task_type: u8, data_addr: usize) {
        // Record data affinity
        self.affinity_map.record_access(data_addr, 0); // Will be updated by scheduler
        
        // Push to ready queue
        self.ready_queue.push(func, data, priority, task_type);
        
        // Warm cache if enabled
        if self.enable_cache_warming {
            warm_task_cache(data as *const u8, 64); // Assume 64 bytes
            warm_function_cache(func);
        }
    }
    
    /// Get the next task for a specific worker
    pub fn get_task_for_worker(&mut self, worker_id: usize) -> Option<(TaskFn, *mut (), u8, u8)> {
        // Try to get a task with data affinity for this worker
        let task = self.ready_queue.pop_highest_priority();
        
        if let Some((func, data, priority, task_type)) = task {
            // Update affinity
            self.affinity_map.record_access(data as usize, worker_id);
            Some((func, data, priority, task_type))
        } else {
            None
        }
    }
    
    /// Get the ready queue reference
    pub fn ready_queue(&self) -> &SoaTaskQueue {
        &self.ready_queue
    }
    
    /// Get the affinity map reference
    pub fn affinity_map(&self) -> &DataAffinityMap {
        &self.affinity_map
    }
    
    /// Submit a task to the SoA scheduler
    pub fn submit_task(&self, task: *mut ()) -> bool {
        self.ready_queue.push(|_| {}, task, 128, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soa_queue_creation() {
        let queue = SoaTaskQueue::new(64);
        assert_eq!(queue.capacity(), 64);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_soa_queue_push_pop() {
        let queue = SoaTaskQueue::new(64);
        
        let task: TaskFn = |_| {};
        assert!(queue.push(task, ptr::null_mut(), 10, 0));
        
        let popped = queue.pop_front();
        assert!(popped.is_some());
    }

    #[test]
    fn test_soa_queue_priority() {
        let queue = SoaTaskQueue::new(64);
        
        let task1: TaskFn = |_| {};
        let task2: TaskFn = |_| {};
        
        queue.push(task1, ptr::null_mut(), 10, 0);
        queue.push(task2, ptr::null_mut(), 5, 0);
        
        let popped = queue.pop_highest_priority();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().2, 5); // Should get priority 5 first
    }

    #[test]
    fn test_data_affinity_map() {
        let mut map = DataAffinityMap::new(4);
        map.record_access(0x1000, 2);
        
        let preferred = map.get_preferred_worker(0x1000);
        assert_eq!(preferred, Some(2));
    }

    #[test]
    fn test_soa_scheduler() {
        let mut scheduler = SoaScheduler::new(64, 4);
        
        let task: TaskFn = |_| {};
        scheduler.schedule(task, ptr::null_mut(), 10, 0, 0x1000);
        
        let task = scheduler.get_task_for_worker(0);
        assert!(task.is_some());
    }
}
