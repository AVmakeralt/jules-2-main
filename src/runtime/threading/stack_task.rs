// =========================================================================
// Stack-Based Task Descriptors
// Zero-allocation tasks that live on the stack or in pre-allocated slabs
// Eliminates heap allocation overhead for task spawning
// =========================================================================

use std::sync::atomic::{AtomicUsize, Ordering};
use std::ptr;

/// Task function pointer type
pub type TaskFn = fn(*mut ());

/// Stack-allocated task descriptor (64 bytes, cache-line aligned)
#[repr(C, align(64))]
pub struct StackTask {
    /// Task function pointer
    pub func: TaskFn,
    /// Task data pointer
    pub data: *mut (),
    /// Task priority (0 = highest)
    pub priority: u8,
    /// Task flags
    pub flags: u8,
    /// Task type (compute, io, etc.)
    pub task_type: u8,
    /// Padding
    _pad: [u8; 64 - std::mem::size_of::<TaskFn>() - std::mem::size_of::<*mut ()>() - 3],
}

impl StackTask {
    /// Create a new stack task
    pub fn new(func: TaskFn, data: *mut (), priority: u8, task_type: u8) -> Self {
        Self {
            func,
            data,
            priority,
            flags: 0,
            task_type,
            _pad: [0; 64 - std::mem::size_of::<TaskFn>() - std::mem::size_of::<*mut ()>() - 3],
        }
    }
    
    /// Execute the task
    pub unsafe fn execute(&self) {
        (self.func)(self.data);
    }
    
    /// Check if the task is high priority
    pub fn is_high_priority(&self) -> bool {
        self.priority < 128
    }
    
    /// Get the task type
    pub fn get_type(&self) -> u8 {
        self.task_type
    }
}

impl Default for StackTask {
    fn default() -> Self {
        Self {
            func: |_| {},
            data: ptr::null_mut(),
            priority: 255,
            flags: 0,
            task_type: 0,
            _pad: [0; 64 - std::mem::size_of::<TaskFn>() - std::mem::size_of::<*mut ()>() - 3],
        }
    }
}

/// Task types
pub const TASK_TYPE_COMPUTE: u8 = 0;
pub const TASK_TYPE_IO: u8 = 1;
pub const TASK_TYPE_GPU: u8 = 2;
pub const TASK_TYPE_SYNC: u8 = 3;

/// Task flags
pub const TASK_FLAG_STEALABLE: u8 = 1 << 0;
pub const TASK_FLAG_PINNED: u8 = 1 << 1;
pub const TASK_FLAG_BATCHED: u8 = 1 << 2;

/// Task batch for zero-allocation batch spawning
#[repr(C, align(64))]
pub struct TaskBatch {
    /// Array of tasks
    pub tasks: [StackTask; 16],
    /// Number of valid tasks in the batch
    pub count: AtomicUsize,
    /// Padding
    _pad: [u8; 64 - std::mem::size_of::<AtomicUsize>()],
}

impl TaskBatch {
    /// Create a new empty task batch
    pub fn new() -> Self {
        Self {
            tasks: Default::default(),
            count: AtomicUsize::new(0),
            _pad: [0; 64 - std::mem::size_of::<AtomicUsize>()],
        }
    }
    
    /// Add a task to the batch
    pub fn add(&mut self, task: StackTask) -> bool {
        let idx = self.count.fetch_add(1, Ordering::AcqRel);
        if idx < 16 {
            self.tasks[idx] = task;
            true
        } else {
            self.count.fetch_sub(1, Ordering::AcqRel);
            false
        }
    }
    
    /// Get the number of tasks in the batch
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Acquire).min(16)
    }
    
    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Clear the batch
    pub fn clear(&mut self) {
        self.count.store(0, Ordering::Release);
    }
    
    /// Get a task from the batch
    pub fn get(&self, idx: usize) -> Option<&StackTask> {
        if idx < self.len() {
            Some(&self.tasks[idx])
        } else {
            None
        }
    }
}

impl Default for TaskBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-thread task cache for zero-allocation task spawning
/// Pre-allocates a pool of task descriptors that can be reused
pub struct TaskCache {
    /// Pool of available task descriptors
    pool: Vec<StackTask>,
    /// Free list indices
    free_list: Vec<usize>,
    /// Capacity
    capacity: usize,
}

impl TaskCache {
    /// Create a new task cache
    pub fn new(capacity: usize) -> Self {
        let mut pool = Vec::with_capacity(capacity);
        let mut free_list = Vec::with_capacity(capacity);
        
        for i in 0..capacity {
            pool.push(StackTask::default());
            free_list.push(i);
        }
        
        Self {
            pool,
            free_list,
            capacity,
        }
    }
    
    /// Allocate a task descriptor from the cache
    pub fn allocate(&mut self) -> Option<&mut StackTask> {
        if let Some(idx) = self.free_list.pop() {
            Some(&mut self.pool[idx])
        } else {
            None
        }
    }
    
    /// Deallocate a task descriptor back to the cache
    pub fn deallocate(&mut self, task: &StackTask) {
        // Find the index of the task in the pool
        if let Some(idx) = self.pool.iter().position(|t| ptr::eq(t, task)) {
            self.free_list.push(idx);
        }
    }
    
    /// Get the number of available task descriptors
    pub fn available(&self) -> usize {
        self.free_list.len()
    }
    
    /// Get the total capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Default for TaskCache {
    fn default() -> Self {
        Self::new(256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_task_creation() {
        let task = StackTask::new(|_| {}, std::ptr::null_mut(), 0, TASK_TYPE_COMPUTE);
        assert_eq!(task.priority, 0);
        assert_eq!(task.task_type, TASK_TYPE_COMPUTE);
    }

    #[test]
    fn test_task_batch() {
        let mut batch = TaskBatch::new();
        let task = StackTask::new(|_| {}, std::ptr::null_mut(), 0, TASK_TYPE_COMPUTE);
        
        assert!(batch.add(task));
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_task_batch_full() {
        let mut batch = TaskBatch::new();
        
        for _ in 0..16 {
            let task = StackTask::new(|_| {}, std::ptr::null_mut(), 0, TASK_TYPE_COMPUTE);
            assert!(batch.add(task));
        }
        
        let task = StackTask::new(|_| {}, std::ptr::null_mut(), 0, TASK_TYPE_COMPUTE);
        assert!(!batch.add(task));
    }

    #[test]
    fn test_task_cache() {
        let mut cache = TaskCache::new(10);

        assert_eq!(cache.available(), 10);

        // Allocate a task - the returned &mut StackTask borrows cache mutably.
        // Convert to raw pointer to release the borrow so we can call other methods.
        let task_ptr = cache.allocate().map(|t| t as *const StackTask);
        assert!(task_ptr.is_some());

        assert_eq!(cache.available(), 9);

        // Deallocate using the raw pointer (safe because the task is still in cache.pool)
        if let Some(ptr) = task_ptr {
            cache.deallocate(unsafe { &*ptr });
        }

        assert_eq!(cache.available(), 10);
    }
}
