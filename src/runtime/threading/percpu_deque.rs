// =========================================================================
// Per-CPU Wait-Free Deque using rseq
// Eliminates CAS operations from the hot path for local push/pop
// Falls back to CAS-based stealing for cross-CPU operations
// =========================================================================

use std::sync::atomic::{AtomicUsize, AtomicPtr, Ordering};
use std::ptr;
use std::alloc::{Layout, alloc, dealloc};

use super::rseq::{rseq_begin, rseq_end};
use super::epoch::{Guard, Participant};

/// Cache line size for alignment
const CACHE_LINE_SIZE: usize = 128;

/// Per-CPU deque capacity (must be power of 2)
const PER_CPU_CAPACITY: usize = 256;

/// Task pointer type
type TaskPtr = *mut ();

/// Per-CPU deque entry
#[repr(C, align(64))]
struct PerCpuDequeEntry {
    /// Task pointer
    task: AtomicPtr<()>,
    /// Sequence number for validation
    seq: AtomicUsize,
}

/// Per-CPU deque buffer
struct PerCpuDequeBuffer {
    /// Array of entries
    entries: *mut PerCpuDequeEntry,
    /// Capacity (must be power of 2)
    capacity: usize,
    /// Mask for modulo operation (capacity - 1)
    mask: usize,
}

impl PerCpuDequeBuffer {
    fn new(capacity: usize) -> Self {
        let layout = Layout::array::<PerCpuDequeEntry>(capacity).unwrap();
        let entries = unsafe { alloc(layout) };
        
        if entries.is_null() {
            panic!("Failed to allocate per-CPU deque buffer");
        }
        
        // Initialize entries
        for i in 0..capacity {
            unsafe {
                let entry = (entries as *mut PerCpuDequeEntry).add(i);
                (*entry).task.store(ptr::null_mut(), Ordering::Relaxed);
                (*entry).seq.store(i, Ordering::Relaxed);
            }
        }
        
        Self {
            entries: unsafe { alloc(layout) as *mut PerCpuDequeEntry },
            capacity,
            mask: capacity - 1,
        }
    }
}

impl Drop for PerCpuDequeBuffer {
    fn drop(&mut self) {
        if !self.entries.is_null() {
            let layout = Layout::array::<PerCpuDequeEntry>(self.capacity).unwrap();
            unsafe { dealloc(self.entries as *mut u8, layout) };
        }
    }
}

/// Per-CPU deque state (one per CPU)
#[repr(C, align(128))]
struct PerCpuDequeState {
    /// Bottom index (owner only)
    bottom: AtomicUsize,
    _pad1: [u8; CACHE_LINE_SIZE - std::mem::size_of::<AtomicUsize>()],
    /// Top index (owner + thieves)
    top: AtomicUsize,
    _pad2: [u8; CACHE_LINE_SIZE - std::mem::size_of::<AtomicUsize>()],
    /// Buffer pointer
    buffer: AtomicPtr<PerCpuDequeBuffer>,
}

impl PerCpuDequeState {
    fn new() -> Self {
        let buffer = Box::into_raw(Box::new(PerCpuDequeBuffer::new(PER_CPU_CAPACITY)));
        
        Self {
            bottom: AtomicUsize::new(0),
            _pad1: [0; CACHE_LINE_SIZE - std::mem::size_of::<AtomicUsize>()],
            top: AtomicUsize::new(0),
            _pad2: [0; CACHE_LINE_SIZE - std::mem::size_of::<AtomicUsize>()],
            buffer: AtomicPtr::new(buffer),
        }
    }
}

impl Drop for PerCpuDequeState {
    fn drop(&mut self) {
        let buffer_ptr = self.buffer.load(Ordering::Acquire);
        if !buffer_ptr.is_null() {
            unsafe {
                let _ = Box::from_raw(buffer_ptr);
            }
        }
    }
}

/// Per-CPU deque manager
/// Manages per-CPU deques with rseq-based wait-free local operations
#[allow(dead_code)]
pub struct PerCpuDeque {
    /// Array of per-CPU deque states
    deques: Vec<PerCpuDequeState>,
    /// Number of CPUs
    num_cpus: usize,
    /// Epoch participant for safe reclamation
    participant: std::sync::Arc<Participant>,
}

impl PerCpuDeque {
    /// Create a new per-CPU deque
    pub fn new(participant: std::sync::Arc<Participant>) -> Self {
        let num_cpus = num_cpus::get();
        let mut deques = Vec::with_capacity(num_cpus);
        
        for _ in 0..num_cpus {
            deques.push(PerCpuDequeState::new());
        }
        
        Self {
            deques,
            num_cpus,
            participant,
        }
    }
    
    /// Push a task to the current CPU's deque (wait-free with rseq)
    pub fn push(&self, task: TaskPtr, _guard: &Guard) {
        if let Some(cpu_id) = rseq_begin() {
            if cpu_id < self.num_cpus {
                // Wait-free path: just increment bottom and write
                let deque = &self.deques[cpu_id];
                let buffer = unsafe { &*deque.buffer.load(Ordering::Acquire) };
                
                let bottom = deque.bottom.load(Ordering::Acquire);
                let idx = bottom & buffer.mask;
                
                unsafe {
                    let entry = buffer.entries.add(idx);
                    (*entry).task.store(task, Ordering::Release);
                    (*entry).seq.fetch_add(1, Ordering::Release);
                }
                
                deque.bottom.store(bottom.wrapping_add(1), Ordering::Release);
                rseq_end();
                return;
            }
            rseq_end();
        }
        
        // Fallback: use CPU 0 with atomic operations
        self.push_fallback(task, _guard);
    }
    
    /// Push fallback path (uses atomic operations)
    fn push_fallback(&self, task: TaskPtr, _guard: &Guard) {
        let deque = &self.deques[0];
        let buffer = unsafe { &*deque.buffer.load(Ordering::Acquire) };
        
        let bottom = deque.bottom.fetch_add(1, Ordering::AcqRel);
        let idx = bottom & buffer.mask;
        
        unsafe {
            let entry = buffer.entries.add(idx);
            (*entry).task.store(task, Ordering::Release);
            (*entry).seq.fetch_add(1, Ordering::Release);
        }
    }
    
    /// Pop a task from the current CPU's deque (wait-free with rseq)
    pub fn pop(&self, _guard: &Guard) -> Option<TaskPtr> {
        if let Some(cpu_id) = rseq_begin() {
            if cpu_id < self.num_cpus {
                // Wait-free path: just decrement bottom and read
                let deque = &self.deques[cpu_id];
                let buffer = unsafe { &*deque.buffer.load(Ordering::Acquire) };
                
                let bottom = deque.bottom.load(Ordering::Acquire);
                if bottom == 0 {
                    rseq_end();
                    return None;
                }
                
                let bottom = bottom.wrapping_sub(1);
                deque.bottom.store(bottom, Ordering::Release);
                
                std::sync::atomic::fence(Ordering::SeqCst);
                
                let top = deque.top.load(Ordering::Acquire);
                
                if bottom < top {
                    // Deque is empty, restore bottom
                    deque.bottom.store(top, Ordering::Release);
                    rseq_end();
                    return None;
                }
                
                let idx = bottom & buffer.mask;
                let task = unsafe {
                    let entry = buffer.entries.add(idx);
                    let task_ptr = (*entry).task.load(Ordering::Acquire);
                    (*entry).task.store(ptr::null_mut(), Ordering::Release);
                    task_ptr
                };
                
                if bottom > top {
                    rseq_end();
                    return if task.is_null() { None } else { Some(task) };
                }
                
                // Last element, need CAS
                if deque.top.compare_exchange(top, top.wrapping_add(1), Ordering::SeqCst, Ordering::Acquire).is_ok() {
                    deque.bottom.store(top.wrapping_add(1), Ordering::Release);
                    rseq_end();
                    return if task.is_null() { None } else { Some(task) };
                } else {
                    deque.bottom.store(top.wrapping_add(1), Ordering::Release);
                    rseq_end();
                    return None;
                }
            }
            rseq_end();
        }
        
        // Fallback: use CPU 0 with atomic operations
        self.pop_fallback(_guard)
    }
    
    /// Pop fallback path (uses atomic operations)
    fn pop_fallback(&self, _guard: &Guard) -> Option<TaskPtr> {
        let deque = &self.deques[0];
        let buffer = unsafe { &*deque.buffer.load(Ordering::Acquire) };
        
        let bottom = deque.bottom.load(Ordering::Acquire);
        if bottom == 0 {
            return None;
        }
        
        let bottom = bottom.wrapping_sub(1);
        deque.bottom.store(bottom, Ordering::Release);
        
        std::sync::atomic::fence(Ordering::SeqCst);
        
        let top = deque.top.load(Ordering::Acquire);
        
        if bottom < top {
            deque.bottom.store(top, Ordering::Release);
            return None;
        }
        
        let idx = bottom & buffer.mask;
        let task = unsafe {
            let entry = buffer.entries.add(idx);
            let task_ptr = (*entry).task.load(Ordering::Acquire);
            (*entry).task.store(ptr::null_mut(), Ordering::Release);
            task_ptr
        };
        
        if bottom > top {
            return if task.is_null() { None } else { Some(task) };
        }
        
        if deque.top.compare_exchange(top, top.wrapping_add(1), Ordering::SeqCst, Ordering::Acquire).is_ok() {
            deque.bottom.store(top.wrapping_add(1), Ordering::Release);
            return if task.is_null() { None } else { Some(task) };
        } else {
            deque.bottom.store(top.wrapping_add(1), Ordering::Release);
            return None;
        }
    }
    
    /// Steal from a specific CPU's deque (uses CAS)
    pub fn steal(&self, cpu_id: usize, _guard: &Guard) -> Option<TaskPtr> {
        if cpu_id >= self.num_cpus {
            return None;
        }
        
        let deque = &self.deques[cpu_id];
        let buffer = unsafe { &*deque.buffer.load(Ordering::Acquire) };
        
        let top = deque.top.load(Ordering::Acquire);
        std::sync::atomic::fence(Ordering::Acquire);
        
        let bottom = deque.bottom.load(Ordering::Acquire);
        
        if top >= bottom {
            return None;
        }
        
        let idx = top & buffer.mask;
        let task = unsafe {
            let entry = buffer.entries.add(idx);
            (*entry).task.load(Ordering::Acquire)
        };
        
        if deque.top.compare_exchange(top, top.wrapping_add(1), Ordering::SeqCst, Ordering::Acquire).is_ok() {
            if !task.is_null() {
                unsafe {
                    let entry = buffer.entries.add(idx);
                    (*entry).task.store(ptr::null_mut(), Ordering::Release);
                }
            }
            Some(task)
        } else {
            None
        }
    }
    
    /// Steal half the tasks from a specific CPU's deque
    pub fn steal_half(&self, cpu_id: usize, _guard: &Guard) -> Vec<TaskPtr> {
        if cpu_id >= self.num_cpus {
            return Vec::new();
        }
        
        let deque = &self.deques[cpu_id];
        let buffer = unsafe { &*deque.buffer.load(Ordering::Acquire) };
        
        let top = deque.top.load(Ordering::Acquire);
        std::sync::atomic::fence(Ordering::Acquire);
        
        let bottom = deque.bottom.load(Ordering::Acquire);
        
        if top >= bottom {
            return Vec::new();
        }
        
        let size = bottom.wrapping_sub(top);
        let half = size / 2;
        let new_top = top.wrapping_add(half);
        
        if deque.top.compare_exchange(top, new_top, Ordering::SeqCst, Ordering::Acquire).is_ok() {
            let mut tasks = Vec::with_capacity(half);
            for i in 0..half {
                let idx = (top.wrapping_add(i)) & buffer.mask;
                let task = unsafe {
                    let entry = buffer.entries.add(idx);
                    let task_ptr = (*entry).task.load(Ordering::Acquire);
                    (*entry).task.store(ptr::null_mut(), Ordering::Release);
                    task_ptr
                };
                if !task.is_null() {
                    tasks.push(task);
                }
            }
            tasks
        } else {
            Vec::new()
        }
    }
    
    /// Get the size of a specific CPU's deque
    pub fn len(&self, cpu_id: usize) -> usize {
        if cpu_id >= self.num_cpus {
            return 0;
        }
        
        let deque = &self.deques[cpu_id];
        let bottom = deque.bottom.load(Ordering::Acquire);
        let top = deque.top.load(Ordering::Acquire);
        bottom.wrapping_sub(top)
    }
    
    /// Check if a specific CPU's deque is empty
    pub fn is_empty(&self, cpu_id: usize) -> bool {
        self.len(cpu_id) == 0
    }
    
    /// Get the number of CPUs
    pub fn num_cpus(&self) -> usize {
        self.num_cpus
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_per_cpu_deque_creation() {
        let participant = Arc::new(Participant::new());
        let deque = PerCpuDeque::new(participant);
        assert!(deque.num_cpus() > 0);
    }

    #[test]
    fn test_push_pop() {
        let participant = Arc::new(Participant::new());
        let deque = PerCpuDeque::new(participant.clone());
        let guard = participant.pin();
        
        let task = Box::into_raw(Box::new(42u32)) as TaskPtr;
        deque.push(task, &guard);
        
        let popped = deque.pop(&guard);
        assert!(!popped.unwrap().is_null());
    }

    #[test]
    fn test_empty_deque() {
        let participant = Arc::new(Participant::new());
        let deque = PerCpuDeque::new(participant.clone());
        let guard = participant.pin();
        
        assert!(deque.pop(&guard).is_none());
    }

    #[test]
    fn test_steal() {
        let participant = Arc::new(Participant::new());
        let deque = PerCpuDeque::new(participant.clone());
        let guard = participant.pin();
        
        let task = Box::into_raw(Box::new(42u32)) as TaskPtr;
        deque.push(task, &guard);
        
        let stolen = deque.steal(0, &guard);
        assert!(!stolen.unwrap().is_null());
    }

    #[test]
    fn test_steal_half() {
        let participant = Arc::new(Participant::new());
        let deque = PerCpuDeque::new(participant.clone());
        let guard = participant.pin();
        
        for i in 0..10 {
            let task = Box::into_raw(Box::new(i)) as TaskPtr;
            deque.push(task, &guard);
        }
        
        let stolen = deque.steal_half(0, &guard);
        assert!(stolen.len() > 0);
    }
}
