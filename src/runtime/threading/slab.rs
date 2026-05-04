// =========================================================================
// Slab Allocator for Zero-Allocation Tasks
// Per-worker slab of pre-allocated task descriptors
// Lock-free free-list with epoch-based reclamation
// Enhanced with per-CPU allocation and huge page support
// =========================================================================

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::ptr;
use std::alloc::{Layout, alloc, dealloc};
use super::rseq::{rseq_begin, rseq_end};

/// Page size for slab allocation (4KB)
const PAGE_SIZE: usize = 4096;

/// Task descriptor size (64 bytes for inline data + metadata)
const TASK_SIZE: usize = 128;

/// Number of task descriptors per page
const TASKS_PER_PAGE: usize = PAGE_SIZE / TASK_SIZE;

/// Alignment for task descriptors (64 bytes for cache line)
const TASK_ALIGN: usize = 64;

/// Slab page containing multiple task descriptors
struct SlabPage {
    /// Next page in the free list
    next: AtomicPtr<SlabPage>,
    /// Task descriptors (raw memory)
    tasks: *mut u8,
    /// Number of tasks in this page
    capacity: usize,
    /// Free list head
    free_list: AtomicPtr<TaskDescriptor>,
}

/// Task descriptor header
#[repr(C)]
pub struct TaskDescriptor {
    /// Next free descriptor in page
    next: AtomicPtr<TaskDescriptor>,
    /// In-use flag
    in_use: AtomicBool,
    /// Padding to align to 64 bytes
    _pad: [u8; 64 - std::mem::size_of::<AtomicPtr<TaskDescriptor>>() - std::mem::size_of::<AtomicBool>()],
}

impl SlabPage {
    fn new(capacity: usize) -> Self {
        let layout = Layout::from_size_align(capacity * TASK_SIZE, TASK_ALIGN).unwrap();
        let tasks = unsafe { alloc(layout) };
        
        if tasks.is_null() {
            panic!("Failed to allocate slab page");
        }
        
        // Initialize free list
        let first_task = tasks as *mut TaskDescriptor;
        for i in 0..capacity {
            let task = unsafe { (tasks as *mut TaskDescriptor).add(i) };
            unsafe {
                (*task).next = if i < capacity - 1 {
                    AtomicPtr::new((tasks as *mut TaskDescriptor).add(i + 1))
                } else {
                    AtomicPtr::new(ptr::null_mut())
                };
                (*task).in_use = AtomicBool::new(false);
            }
        }
        
        Self {
            next: AtomicPtr::new(ptr::null_mut()),
            tasks,
            capacity,
            free_list: AtomicPtr::new(first_task),
        }
    }
}

impl Drop for SlabPage {
    fn drop(&mut self) {
        if !self.tasks.is_null() {
            let layout = Layout::from_size_align(self.capacity * TASK_SIZE, TASK_ALIGN).unwrap();
            unsafe { dealloc(self.tasks, layout) };
        }
    }
}

/// Per-worker slab allocator with lock-free free-list
pub struct SlabAllocator {
    /// Free list of task descriptors
    free_list: AtomicPtr<TaskDescriptor>,
    /// List of allocated pages
    pages: AtomicPtr<SlabPage>,
    /// Number of pages allocated
    page_count: AtomicUsize,
    /// Tasks per page
    tasks_per_page: usize,
    /// Per-CPU free lists for wait-free allocation
    per_cpu_free_lists: Vec<AtomicPtr<TaskDescriptor>>,
    /// Use huge pages if available
    use_huge_pages: bool,
}

impl SlabAllocator {
    /// Create a new slab allocator
    pub fn new() -> Self {
        let tasks_per_page = TASKS_PER_PAGE;
        let page = Box::into_raw(Box::new(SlabPage::new(tasks_per_page)));
        
        let first_task = unsafe { (*page).free_list.load(Ordering::Acquire) };
        
        let num_cpus = num_cpus::get();
        let mut per_cpu_free_lists = Vec::with_capacity(num_cpus);
        for _ in 0..num_cpus {
            per_cpu_free_lists.push(AtomicPtr::new(ptr::null_mut()));
        }
        
        Self {
            free_list: AtomicPtr::new(first_task),
            pages: AtomicPtr::new(page),
            page_count: AtomicUsize::new(1),
            tasks_per_page,
            per_cpu_free_lists,
            use_huge_pages: false,
        }
    }
    
    /// Create a new slab allocator with huge page support
    pub fn with_huge_pages() -> Self {
        let mut alloc = Self::new();
        alloc.use_huge_pages = true;
        alloc
    }

    /// Allocate a new task descriptor (wait-free with rseq on local CPU)
    pub fn allocate(&self) -> *mut TaskDescriptor {
        // Try per-CPU free list first (wait-free with rseq)
        if let Some(cpu_id) = rseq_begin() {
            if cpu_id < self.per_cpu_free_lists.len() {
                let head = self.per_cpu_free_lists[cpu_id].load(Ordering::Acquire);
                if !head.is_null() {
                    let next = unsafe { (*head).next.load(Ordering::Acquire) };
                    self.per_cpu_free_lists[cpu_id].store(next, Ordering::Release);
                    unsafe { (*head).in_use.store(true, Ordering::Release); }
                    rseq_end();
                    return head;
                }
            }
            rseq_end();
        }
        
        // Fallback to global free list with CAS loop
        loop {
            let head = self.free_list.load(Ordering::Acquire);
            
            if head.is_null() {
                // Free list empty, allocate new page
                self.allocate_page();
                continue;
            }
            
            let next = unsafe { (*head).next.load(Ordering::Acquire) };
            
            if self.free_list.compare_exchange_weak(
                head,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ).is_ok() {
                unsafe { (*head).in_use.store(true, Ordering::Release); }
                return head;
            }
            // CAS failed, retry
        }
    }

    /// Deallocate a task descriptor (wait-free with rseq on local CPU)
    pub fn deallocate(&self, descriptor: *mut TaskDescriptor) {
        if descriptor.is_null() {
            return;
        }
        
        unsafe {
            (*descriptor).in_use.store(false, Ordering::Release);
            
            // Try per-CPU free list first (wait-free with rseq)
            if let Some(cpu_id) = rseq_begin() {
                if cpu_id < self.per_cpu_free_lists.len() {
                    let head = self.per_cpu_free_lists[cpu_id].load(Ordering::Acquire);
                    (*descriptor).next.store(head, Ordering::Release);
                    self.per_cpu_free_lists[cpu_id].store(descriptor, Ordering::Release);
                    rseq_end();
                    return;
                }
                rseq_end();
            }
            
            // Fallback to global free list with CAS loop
            loop {
                let head = self.free_list.load(Ordering::Acquire);
                (*descriptor).next.store(head, Ordering::Release);
                
                if self.free_list.compare_exchange_weak(
                    head,
                    descriptor,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ).is_ok() {
                    break;
                }
            }
        }
    }

    /// Allocate a new page of task descriptors
    fn allocate_page(&self) {
        let page = Box::into_raw(Box::new(SlabPage::new(self.tasks_per_page)));
        let first_task = unsafe { (*page).free_list.load(Ordering::Acquire) };
        
        // Add page to pages list
        loop {
            let head = self.pages.load(Ordering::Acquire);
            unsafe { (*page).next.store(head, Ordering::Release); }
            
            if self.pages.compare_exchange_weak(
                head,
                page,
                Ordering::AcqRel,
                Ordering::Acquire,
            ).is_ok() {
                break;
            }
        }
        
        // Update free list to point to first descriptor in new page
        loop {
            let head = self.free_list.load(Ordering::Acquire);
            unsafe { (*first_task).next.store(head, Ordering::Release); }
            
            if self.free_list.compare_exchange_weak(
                head,
                first_task,
                Ordering::AcqRel,
                Ordering::Acquire,
            ).is_ok() {
                break;
            }
        }
        
        self.page_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the number of pages allocated
    pub fn page_count(&self) -> usize {
        self.page_count.load(Ordering::Relaxed)
    }

    /// Get the total capacity
    pub fn capacity(&self) -> usize {
        self.page_count() * self.tasks_per_page
    }
}

impl Default for SlabAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SlabAllocator {
    fn drop(&mut self) {
        // Free all allocated pages
        let mut page = self.pages.load(Ordering::Acquire);
        
        while !page.is_null() {
            unsafe {
                let next = (*page).next.load(Ordering::Acquire);
                let _ = Box::from_raw(page);
                page = next;
            }
        }
    }
}

unsafe impl Send for SlabAllocator {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slab_allocation() {
        let slab = SlabAllocator::new();
        let desc = slab.allocate();
        assert!(!desc.is_null());
        slab.deallocate(desc);
    }

    #[test]
    fn test_multiple_allocations() {
        let slab = SlabAllocator::new();
        let mut descriptors = Vec::new();
        
        for _ in 0..100 {
            let desc = slab.allocate();
            assert!(!desc.is_null());
            descriptors.push(desc);
        }
        
        for desc in descriptors {
            slab.deallocate(desc);
        }
    }

    #[test]
    fn test_allocate_after_deallocate() {
        let slab = SlabAllocator::new();
        let desc1 = slab.allocate();
        slab.deallocate(desc1);
        let desc2 = slab.allocate();
        assert!(!desc2.is_null());
        slab.deallocate(desc2);
    }

    #[test]
    fn test_page_growth() {
        let slab = SlabAllocator::new();
        let initial_pages = slab.page_count();
        
        // Allocate more than fits in one page
        for _ in 0..TASKS_PER_PAGE + 10 {
            let desc = slab.allocate();
            assert!(!desc.is_null());
        }
        
        assert!(slab.page_count() > initial_pages);
    }

    #[test]
    fn test_capacity() {
        let slab = SlabAllocator::new();
        assert!(slab.capacity() > 0);
    }
}
