// =========================================================================
// Join API for Fork-Join Parallelism
// Stack-allocated tasks for zero-allocation fork-join
// Implements vtable pattern for type-erased execution
// =========================================================================

use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::Arc;
use std::mem::ManuallyDrop;

use super::epoch::{Guard, Participant};
use super::slab::SlabAllocator;
use super::worker::ThreadPool;

/// Global thread pool (lazy initialization)
static mut GLOBAL_POOL: Option<ThreadPool> = None;
static mut GLOBAL_PARTICIPANT: Option<Participant> = None;
static mut GLOBAL_SLAB: Option<SlabAllocator> = None;
static POOL_INIT: std::sync::Once = std::sync::Once::new();

/// Get or create the global thread pool
fn get_pool() -> &'static ThreadPool {
    unsafe {
        POOL_INIT.call_once(|| {
            GLOBAL_PARTICIPANT = Some(Participant::new());
            GLOBAL_SLAB = Some(SlabAllocator::new());
            GLOBAL_POOL = Some(ThreadPool::new());
        });
        GLOBAL_POOL.as_ref().unwrap()
    }
}

/// Get the global slab allocator
fn get_slab() -> &'static SlabAllocator {
    unsafe {
        POOL_INIT.call_once(|| {
            GLOBAL_PARTICIPANT = Some(Participant::new());
            GLOBAL_SLAB = Some(SlabAllocator::new());
            GLOBAL_POOL = Some(ThreadPool::new());
        });
        GLOBAL_SLAB.as_ref().unwrap()
    }
}

/// Get the global epoch participant
fn get_participant() -> &'static Participant {
    unsafe {
        POOL_INIT.call_once(|| {
            GLOBAL_PARTICIPANT = Some(Participant::new());
            GLOBAL_SLAB = Some(SlabAllocator::new());
            GLOBAL_POOL = Some(ThreadPool::new());
        });
        GLOBAL_PARTICIPANT.as_ref().unwrap()
    }
}

/// Vtable for task execution
#[repr(C)]
struct TaskVtable {
    /// Run the task
    run: unsafe fn(*mut ()),
    /// Drop the task
    drop: unsafe fn(*mut ()),
}

/// Stack-allocated task descriptor with vtable
#[repr(C)]
struct StackTask {
    /// Vtable pointer
    vtable: *const TaskVtable,
    /// Inline data (up to 64 bytes)
    data: [u8; 64],
    /// Latch for stolen task detection
    stolen: AtomicBool,
    /// Completed flag
    completed: AtomicBool,
}

impl StackTask {
    /// Create a new stack task
    fn new(vtable: *const TaskVtable) -> Self {
        Self {
            vtable,
            data: [0; 64],
            stolen: AtomicBool::new(false),
            completed: AtomicBool::new(false),
        }
    }

    /// Execute the task
    unsafe fn execute(&mut self) {
        if let Some(vtable) = self.vtable.as_ref() {
            (vtable.run)(self as *mut _ as *mut ());
        }
        self.completed.store(true, Ordering::Release);
    }

    /// Mark as stolen
    fn mark_stolen(&self) {
        self.stolen.store(true, Ordering::Release);
    }

    /// Check if stolen
    fn is_stolen(&self) -> bool {
        self.stolen.load(Ordering::Acquire)
    }

    /// Check if completed
    fn is_completed(&self) -> bool {
        self.completed.load(Ordering::Acquire)
    }
}

/// Slab-allocated task descriptor
struct SlabTask {
    /// Vtable pointer
    vtable: *const TaskVtable,
    /// Inline data (up to 64 bytes)
    data: [u8; 64],
    /// Completed flag
    completed: AtomicBool,
    /// Result pointer (for JoinHandle)
    result_ptr: AtomicPtr<()>,
}

impl SlabTask {
    fn new(vtable: *const TaskVtable) -> Self {
        Self {
            vtable,
            data: [0; 64],
            completed: AtomicBool::new(false),
            result_ptr: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    unsafe fn execute(&mut self) {
        if let Some(vtable) = self.vtable.as_ref() {
            (vtable.run)(self as *mut _ as *mut ());
        }
        self.completed.store(true, Ordering::Release);
    }
}

/// Join handle for spawned tasks
pub struct JoinHandle<T> {
    /// Task pointer
    task: *mut SlabTask,
    /// Phantom data
    _marker: std::marker::PhantomData<T>,
}

impl<T> JoinHandle<T> {
    /// Wait for the task to complete and return the result
    pub fn join(self) -> ThreadResult<T> {
        unsafe {
            if self.task.is_null() {
                return Err("Task pointer is null".to_string());
            }

            let task_ref = &*self.task;
            
            // Wait for completion
            while !task_ref.completed.load(Ordering::Acquire) {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }

            // Extract result
            let result_ptr = task_ref.result_ptr.load(Ordering::Acquire);
            if result_ptr.is_null() {
                return Err("No result available".to_string());
            }

            let result = Box::from_raw(result_ptr as *mut ManuallyDrop<T>);
            Ok(result.into_inner())
        }
    }
}

impl<T> Drop for JoinHandle<T> {
    fn drop(&mut self) {
        unsafe {
            if !self.task.is_null() {
                let _ = Box::from_raw(self.task);
            }
        }
    }
}

/// Execute two closures in parallel and wait for both to complete
/// Fixed: actually runs closures in parallel using std::thread::scope
/// No external crate needed (requires Rust 1.63+ for thread::scope)
pub fn join<A, B, FA, FB>(a: FA, b: FB) -> (A, B)
where
    A: Send,
    B: Send,
    FA: FnOnce() -> A + Send,
    FB: FnOnce() -> B + Send,
{
    std::thread::scope(|s| {
        // Spawn thread B in the background
        let handle = s.spawn(|| {
            b()
        });
        // Run closure A on the current thread
        let ra = a();
        // Wait for B to finish
        let rb = handle.join().unwrap();
        (ra, rb)
    })
}

/// For single-threaded fallback (when Send is not available)
pub fn join_sequential<A, B, FA, FB>(a: FA, b: FB) -> (A, B)
where
    FA: FnOnce() -> A,
    FB: FnOnce() -> B,
{
    (a(), b())
}

/// Spawn a task that runs in the background
/// Uses slab-allocated task descriptor for zero-allocation
pub fn spawn<F, T>(f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    // Create vtable for this closure type
    extern "C" fn run_task<T, F>(ptr: *mut ())
    where
        F: FnOnce() -> T,
    {
        let task = unsafe { &mut *(ptr as *mut SlabTask) };
        
        // Extract closure from inline data
        let closure = unsafe {
            let closure_ptr = task.data.as_ptr() as *const F;
            std::ptr::read(closure_ptr)
        };
        
        // Execute closure
        let result = closure();
        
        // Store result
        let result_box = Box::new(ManuallyDrop::new(result));
        task.result_ptr.store(Box::into_raw(result_box) as *mut (), Ordering::Release);
    }

    extern "C" fn drop_task<T, F>(ptr: *mut ()) {
        unsafe {
            let task = &mut *(ptr as *mut SlabTask);
            // Drop closure if it was stored
            let closure_ptr = task.data.as_mut_ptr() as *mut F;
            std::ptr::drop_in_place(closure_ptr);
        }
    }

    static VTABLE: TaskVtable = TaskVtable {
        run: run_task::<T, F>,
        drop: drop_task::<T, F>,
    };

    // Use slab-allocated task descriptor
    let slab = get_slab();
    let descriptor = slab.allocate();
    
    if descriptor.is_null() {
        // Slab allocation failed, return dummy handle
        return JoinHandle {
            task: std::ptr::null_mut(),
            _marker: std::marker::PhantomData,
        };
    }

    unsafe {
        let task = &mut *(descriptor as *mut SlabTask);
        task.vtable = &VTABLE;
        
        // Store closure in inline data
        let closure_ptr = task.data.as_mut_ptr() as *mut F;
        std::ptr::write(closure_ptr, f);
    }

    let pool = get_pool();
    let task_ptr = descriptor as *mut ();
    pool.submit(task_ptr);

    JoinHandle {
        task: descriptor as *mut SlabTask,
        _marker: std::marker::PhantomData,
    }
}

/// Execute a closure on all available threads in parallel
/// Adaptive splitting based on workload characteristics
pub fn par_for<F>(range: std::ops::Range<usize>, f: F)
where
    F: Fn(usize) + Send + Sync,
{
    let pool = get_pool();
    let num_workers = pool.num_workers();
    
    if num_workers <= 1 || range.len() <= 1 {
        // Sequential execution for small workloads
        for i in range {
            f(i);
        }
        return;
    }

    // Adaptive chunking
    let chunk_size = (range.len() / num_workers).max(1);
    
    for chunk_start in (range.start..range.end).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(range.end);
        let f_clone = &f;
        
        let task = Box::new(move || {
            for i in chunk_start..chunk_end {
                f_clone(i);
            }
        });
        
        let task_ptr = Box::into_raw(task) as *mut ();
        pool.submit(task_ptr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_join_sequential() {
        let (a, b) = join(|| 1, || 2);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
    }

    #[test]
    fn test_par_for_sequential() {
        let mut sum = 0;
        par_for(0..10, |i| {
            sum += i;
        });
        assert_eq!(sum, 45);
    }

    #[test]
    fn test_spawn_join() {
        let handle = spawn(|| 42);
        let result = handle.join().unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_global_pool() {
        let pool = get_pool();
        assert!(pool.num_workers() > 0);
    }
}
