// =========================================================================
// Chase-Lev Work-Stealing Deque
// Lock-free single-producer, multi-consumer deque
// Implements corrected Le et al. algorithm for weak memory models
// =========================================================================

use std::sync::atomic::{AtomicUsize, AtomicPtr, Ordering};
use std::sync::Arc;
use std::ptr;

use super::epoch::{Guard, Participant};

/// Cache line size for alignment (128 bytes to account for Intel spatial prefetcher)
const CACHE_LINE_SIZE: usize = 128;

/// Initial buffer capacity (must be power of 2)
const INITIAL_CAPACITY: usize = 64;

/// Task pointer type
type TaskPtr = *mut ();

/// Chase-Lev work-stealing deque with 128-byte alignment
#[repr(C, align(128))]
pub struct WorkStealingDeque {
    /// Bottom index (owner only) - padded to prevent false sharing
    bottom: AtomicUsize,
    _pad1: [u8; CACHE_LINE_SIZE - std::mem::size_of::<AtomicUsize>()],
    /// Top index (owner + thieves) - padded to prevent false sharing
    top: AtomicUsize,
    _pad2: [u8; CACHE_LINE_SIZE - std::mem::size_of::<AtomicUsize>()],
    /// Buffer pointer
    buffer: AtomicPtr<Buffer>,
    /// Epoch participant for safe reclamation
    participant: Arc<Participant>,
}

/// Circular buffer for task storage
struct Buffer {
    /// Array of task pointers
    data: std::cell::UnsafeCell<Vec<TaskPtr>>,
    /// Capacity (must be power of 2)
    capacity: usize,
    /// Epoch when this buffer was retired
    retired_epoch: u64,
}

// SAFETY: Buffer is only accessed by the owner thread for push/pop,
// and thieves only read through atomic operations with proper synchronization.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("capacity", &self.capacity)
            .field("retired_epoch", &self.retired_epoch)
            .finish()
    }
}

impl WorkStealingDeque {
    /// Create a new empty deque
    pub fn new(participant: Arc<Participant>) -> Self {
        let buffer = Buffer::new(INITIAL_CAPACITY);
        let buffer_ptr = Box::into_raw(Box::new(buffer));
        
        Self {
            bottom: AtomicUsize::new(0),
            _pad1: [0; CACHE_LINE_SIZE - std::mem::size_of::<AtomicUsize>()],
            top: AtomicUsize::new(0),
            _pad2: [0; CACHE_LINE_SIZE - std::mem::size_of::<AtomicUsize>()],
            buffer: AtomicPtr::new(buffer_ptr),
            participant,
        }
    }

    /// Push a task to the bottom (owner only)
    /// This is the fast path - no CAS required, just atomic stores
    pub fn push(&self, task: TaskPtr, guard: &Guard) {
        let bottom = self.bottom.load(Ordering::Acquire);
        let top = self.top.load(Ordering::Acquire);
        let buffer = self.load_buffer();
        
        let size = bottom.wrapping_sub(top);
        
        if size >= buffer.capacity {
            // Need to grow buffer
            self.grow(bottom, top, buffer, guard);
        }
        
        let buffer = self.load_buffer();
        let idx = bottom & (buffer.capacity - 1);
        
        buffer.data_mut()[idx] = task;
        
        std::sync::atomic::fence(Ordering::Release);
        self.bottom.store(bottom.wrapping_add(1), Ordering::Release);
    }

    /// Pop a task from the bottom (owner only)
    /// Implements the corrected Le et al. algorithm
    pub fn pop(&self, _guard: &Guard) -> Option<TaskPtr> {
        let buffer = self.load_buffer();
        let bottom = self.bottom.load(Ordering::Acquire);
        
        if bottom == 0 {
            return None;
        }
        
        let bottom = bottom.wrapping_sub(1);
        self.bottom.store(bottom, Ordering::Release);
        
        std::sync::atomic::fence(Ordering::SeqCst);
        
        let top = self.top.load(Ordering::Acquire);
        
        if bottom < top {
            // Deque is empty, restore bottom
            self.bottom.store(top, Ordering::Release);
            return None;
        }
        
        let idx = bottom & (buffer.capacity - 1);
        let task = buffer.data()[idx];
        
        if bottom > top {
            // Successfully popped
            buffer.data_mut()[idx] = ptr::null_mut();
            return Some(task);
        }
        
        // Last element, try to steal it ourselves
        // SeqCst is critical here for weak memory models (ARM/AArch64)
        if self.top.compare_exchange(top, top.wrapping_add(1), Ordering::SeqCst, Ordering::Acquire).is_ok() {
            // Successfully stole the last element
            self.bottom.store(top.wrapping_add(1), Ordering::Release);
            buffer.data_mut()[idx] = ptr::null_mut();
            Some(task)
        } else {
            // Someone else stole it
            self.bottom.store(top.wrapping_add(1), Ordering::Release);
            None
        }
    }

    /// Steal a task from the top (thief only)
    /// SeqCst on CAS is critical for correctness on weak memory models
    pub fn steal(&self, _guard: &Guard) -> Option<TaskPtr> {
        let buffer = self.load_buffer();
        let top = self.top.load(Ordering::Acquire);
        
        std::sync::atomic::fence(Ordering::Acquire);
        
        let bottom = self.bottom.load(Ordering::Acquire);
        
        if top >= bottom {
            return None;
        }
        
        let idx = top & (buffer.capacity - 1);
        let task = buffer.data()[idx];
        
        // Try to claim the task with SeqCst ordering
        if self.top.compare_exchange(top, top.wrapping_add(1), Ordering::SeqCst, Ordering::Acquire).is_ok() {
            Some(task)
        } else {
            None
        }
    }

    /// Steal half the tasks (batch stealing)
    /// Amortizes cache invalidation cost across multiple tasks
    pub fn steal_half(&self, _guard: &Guard) -> Vec<TaskPtr> {
        let buffer = self.load_buffer();
        let top = self.top.load(Ordering::Acquire);
        
        std::sync::atomic::fence(Ordering::Acquire);
        
        let bottom = self.bottom.load(Ordering::Acquire);
        
        if top >= bottom {
            return Vec::new();
        }
        
        let size = bottom.wrapping_sub(top);
        let half = size / 2;
        let new_top = top.wrapping_add(half);
        
        // Try to claim half the tasks with single CAS
        if self.top.compare_exchange(top, new_top, Ordering::SeqCst, Ordering::Acquire).is_ok() {
            let mut tasks = Vec::with_capacity(half);
            for i in 0..half {
                let idx = (top.wrapping_add(i)) & (buffer.capacity - 1);
                let task = buffer.data()[idx];
                if !task.is_null() {
                    tasks.push(task);
                    buffer.data_mut()[idx] = ptr::null_mut();
                }
            }
            tasks
        } else {
            Vec::new()
        }
    }

    /// Get the current size of the deque
    pub fn len(&self) -> usize {
        let bottom = self.bottom.load(Ordering::Acquire);
        let top = self.top.load(Ordering::Acquire);
        bottom.wrapping_sub(top)
    }

    /// Check if the deque is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Load the current buffer with Acquire ordering
    fn load_buffer(&self) -> &Buffer {
        unsafe { &*self.buffer.load(Ordering::Acquire) }
    }

    /// Grow the buffer with epoch-based reclamation
    /// This is the critical path for safe memory reclamation
    fn grow(&self, bottom: usize, top: usize, old_buffer: &Buffer, guard: &Guard) {
        let new_capacity = old_buffer.capacity * 2;
        let new_buffer = Buffer::new(new_capacity);
        
        // Copy elements from old buffer to new buffer
        let size = bottom.wrapping_sub(top);
        for i in 0..size {
            let old_idx = (top.wrapping_add(i)) & (old_buffer.capacity - 1);
            let new_idx = (top.wrapping_add(i)) & (new_capacity - 1);
            new_buffer.data_mut()[new_idx] = old_buffer.data()[old_idx];
        }
        
        let new_buffer_ptr = Box::into_raw(Box::new(new_buffer));
        
        // Swap buffers atomically with AcqRel ordering
        let old_ptr = self.buffer.swap(new_buffer_ptr, Ordering::AcqRel);
        
        // Retire old buffer for epoch-based reclamation
        // This ensures no thief is still reading the old buffer
        let old_buffer = unsafe { Box::from_raw(old_ptr) };
        guard.defer(Box::new(old_buffer) as Box<dyn Send + std::fmt::Debug>);
    }
}

impl Buffer {
    fn new(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity);
        data.resize(capacity, ptr::null_mut());
        
        Self {
            data: std::cell::UnsafeCell::new(data),
            capacity,
            retired_epoch: 0,
        }
    }
    
    /// Get a reference to the data vector
    fn data(&self) -> &Vec<TaskPtr> {
        // SAFETY: Access is synchronized through atomic operations on bottom/top
        unsafe { &*self.data.get() }
    }
    
    /// Get a mutable reference to the data vector
    fn data_mut(&self) -> &mut Vec<TaskPtr> {
        // SAFETY: Access is synchronized through atomic operations on bottom/top
        unsafe { &mut *self.data.get() }
    }
}

impl Drop for WorkStealingDeque {
    fn drop(&mut self) {
        let buffer_ptr = self.buffer.load(Ordering::Acquire);
        if !buffer_ptr.is_null() {
            unsafe {
                let _buffer = Box::from_raw(buffer_ptr);
            }
        }
    }
}

unsafe impl Send for WorkStealingDeque {}
unsafe impl Sync for WorkStealingDeque {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_push_pop() {
        let participant = Arc::new(Participant::new());
        let deque = WorkStealingDeque::new(participant.clone());
        let guard = participant.pin();
        
        let task = Box::into_raw(Box::new(42u32)) as TaskPtr;
        deque.push(task, &guard);
        
        let popped = deque.pop(&guard);
        assert!(!popped.unwrap().is_null());
    }

    #[test]
    fn test_empty_deque() {
        let participant = Arc::new(Participant::new());
        let deque = WorkStealingDeque::new(participant.clone());
        let guard = participant.pin();
        
        assert!(deque.pop(&guard).is_none());
        assert!(deque.steal(&guard).is_none());
    }

    #[test]
    fn test_steal() {
        let participant = Arc::new(Participant::new());
        let deque = WorkStealingDeque::new(participant.clone());
        let guard = participant.pin();
        
        let task = Box::into_raw(Box::new(42u32)) as TaskPtr;
        deque.push(task, &guard);
        
        let stolen = deque.steal(&guard);
        assert!(!stolen.unwrap().is_null());
    }

    #[test]
    fn test_buffer_growth() {
        let participant = Arc::new(Participant::new());
        let deque = WorkStealingDeque::new(participant.clone());
        let guard = participant.pin();
        
        // Push more than initial capacity to trigger growth
        for i in 0..100 {
            let task = Box::into_raw(Box::new(i)) as TaskPtr;
            deque.push(task, &guard);
        }
        
        assert!(deque.len() > 0);
    }

    #[test]
    fn test_steal_half() {
        let participant = Arc::new(Participant::new());
        let deque = WorkStealingDeque::new(participant.clone());
        let guard = participant.pin();
        
        // Push multiple tasks
        for i in 0..10 {
            let task = Box::into_raw(Box::new(i)) as TaskPtr;
            deque.push(task, &guard);
        }
        
        let stolen = deque.steal_half(&guard);
        assert!(stolen.len() > 0);
    }
}
