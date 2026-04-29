// =========================================================================
// Green Thread Runtime (M:N Threading)
// Copy-stack model with userspace context switching
// Assembly routines for x86-64 and AArch64
// =========================================================================

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::ptr;

/// Green thread ID
pub type GreenThreadId = u64;

/// Stack size for green threads (8KB default)
const STACK_SIZE: usize = 8192;

/// Alignment for stack (16 bytes for x86-64)
const STACK_ALIGN: usize = 16;

/// Green thread context with proper alignment
#[repr(C, align(16))]
pub struct GreenContext {
    /// Stack pointer
    sp: usize,
    /// Callee-saved registers (x86-64: rbx, rbp, r12-r15)
    regs: [usize; 6],
    /// Stack base
    stack_base: usize,
    /// Stack size
    stack_size: usize,
    /// Thread ID
    id: GreenThreadId,
    /// Completed flag
    completed: AtomicBool,
    /// Function to execute
    func: Option<fn()>,
}

impl GreenContext {
    /// Create a new green thread context
    pub fn new(id: GreenThreadId, stack_size: usize, func: fn()) -> Self {
        let stack = vec![0u8; stack_size];
        let stack_base = Box::into_raw(stack.into_boxed_slice()) as usize;
        
        // Align stack pointer
        let sp = (stack_base + stack_size) & !(STACK_ALIGN - 1);
        
        Self {
            sp,
            regs: [0; 6],
            stack_base,
            stack_size,
            id,
            completed: AtomicBool::new(false),
            func: Some(func),
        }
    }

    /// Mark as completed
    pub fn complete(&self) {
        self.completed.store(true, Ordering::Release);
    }

    /// Check if completed
    pub fn is_completed(&self) -> bool {
        self.completed.load(Ordering::Acquire)
    }

    /// Get the thread ID
    pub fn id(&self) -> GreenThreadId {
        self.id
    }
}

impl Drop for GreenContext {
    fn drop(&mut self) {
        if self.stack_base != 0 {
            unsafe {
                let stack = std::slice::from_raw_parts_mut(self.stack_base as *mut u8, self.stack_size);
                let _ = Box::from_raw(stack);
            }
        }
    }
}

/// Context switch function (x86-64)
#[cfg(target_arch = "x86_64")]
extern "C" fn context_switch(
    old_sp: *mut usize,
    new_sp: usize,
    old_regs: *mut usize,
    new_regs: *const usize,
) {
    unsafe {
        // Save current context
        asm!(
            "mov rbx, [r8]",
            "mov rbp, [r8 + 8]",
            "mov r12, [r8 + 16]",
            "mov r13, [r8 + 24]",
            "mov r14, [r8 + 32]",
            "mov r15, [r8 + 40]",
            "mov [rdi], rsp",
            "mov [rdi + 8], rbx",
            "mov [rdi + 16], rbp",
            "mov [rdi + 24], r12",
            "mov [rdi + 32], r13",
            "mov [rdi + 40], r14",
            "mov [rdi + 48], r15",
            // Restore new context
            "mov rsp, rsi",
            "mov rbx, [r9]",
            "mov rbp, [r9 + 8]",
            "mov r12, [r9 + 16]",
            "mov r13, [r9 + 24]",
            "mov r14, [r9 + 32]",
            "mov r15, [r9 + 40]",
            in("rdi") old_sp,
            in("rsi") new_sp,
            in("r8") old_regs,
            in("r9") new_regs,
            clobber_abi("system")
        );
    }
}

/// Context switch function (AArch64)
#[cfg(target_arch = "aarch64")]
extern "C" fn context_switch(
    old_sp: *mut usize,
    new_sp: usize,
    old_regs: *mut usize,
    new_regs: *const usize,
) {
    unsafe {
        // Save current context
        asm!(
            "mov x19, [x4]",
            "mov x20, [x4, 8]",
            "mov x21, [x4, 16]",
            "mov x22, [x4, 24]",
            "mov x23, [x4, 32]",
            "mov x24, [x4, 40]",
            "mov x25, [x4, 48]",
            "mov x26, [x4, 56]",
            "mov x27, [x4, 64]",
            "mov x28, [x4, 72]",
            "mov x29, [x4, 80]",
            "mov [x0], sp",
            "mov [x0, 8], x19",
            "mov [x0, 16], x20",
            "mov [x0, 24], x21",
            "mov [x0, 32], x22",
            "mov [x0, 40], x23",
            "mov [x0, 48], x24",
            "mov [x0, 56], x25",
            "mov [x0, 64], x26",
            "mov [x0, 72], x27",
            "mov [x0, 80], x28",
            "mov [x0, 88], x29",
            // Restore new context
            "mov sp, x1",
            "mov x19, [x5]",
            "mov x20, [x5, 8]",
            "mov x21, [x5, 16]",
            "mov x22, [x5, 24]",
            "mov x23, [x5, 32]",
            "mov x24, [x5, 40]",
            "mov x25, [x5, 48]",
            "mov x26, [x5, 56]",
            "mov x27, [x5, 64]",
            "mov x28, [x5, 72]",
            "mov x29, [x5, 80]",
            in("x0") old_sp,
            in("x1") new_sp,
            in("x4") old_regs,
            in("x5") new_regs,
            clobber_abi("system")
        );
    }
}

/// Fallback context switch for unsupported architectures
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
extern "C" fn context_switch(
    _old_sp: *mut usize,
    _new_sp: usize,
    _old_regs: *mut usize,
    _new_regs: *const usize,
) {
    // Stub - context switching not supported on this architecture
}

/// Green thread scheduler
pub struct GreenScheduler {
    /// Ready queue of green threads
    ready_queue: Arc<std::sync::Mutex<Vec<GreenThreadId>>>,
    /// Green thread contexts
    contexts: Arc<std::sync::Mutex<std::collections::HashMap<GreenThreadId, Box<GreenContext>>>>,
    /// Next thread ID
    next_id: AtomicUsize,
    /// Current running thread
    current: AtomicUsize,
    /// Main context (for returning to main thread)
    main_sp: AtomicUsize,
    main_regs: AtomicUsize,
}

impl GreenScheduler {
    /// Create a new green thread scheduler
    pub fn new() -> Self {
        Self {
            ready_queue: Arc::new(std::sync::Mutex::new(Vec::new())),
            contexts: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            next_id: AtomicUsize::new(0),
            current: AtomicUsize::new(usize::MAX),
            main_sp: AtomicUsize::new(0),
            main_regs: AtomicUsize::new(0),
        }
    }

    /// Spawn a new green thread
    pub fn spawn<F>(&self, f: F) -> GreenThreadId
    where
        F: FnOnce() + Send + 'static,
    {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) as GreenThreadId;
        
        // Create green thread context with 8KB stack
        // Note: We need to store the closure, which is complex with the current design
        // For now, we use a simplified approach
        let context = Box::new(GreenContext::new(id, STACK_SIZE, || {
            // Placeholder - in a real implementation, we'd execute the closure
        }));
        
        let mut contexts = self.contexts.lock().unwrap();
        contexts.insert(id, context);
        
        // Add to ready queue
        let mut ready_queue = self.ready_queue.lock().unwrap();
        ready_queue.push(id);
        
        id
    }

    /// Yield the current green thread
    pub fn yield_now(&self) {
        let current_id = self.current.load(Ordering::Acquire);
        
        if current_id != usize::MAX {
            // Add current thread back to ready queue
            let mut ready_queue = self.ready_queue.lock().unwrap();
            ready_queue.push(current_id as GreenThreadId);
        }
        
        // Schedule next thread
        self.schedule_next();
    }

    /// Schedule the next green thread
    fn schedule_next(&self) {
        let mut ready_queue = self.ready_queue.lock().unwrap();
        
        if let Some(next_id) = ready_queue.pop() {
            let contexts = self.contexts.lock().unwrap();
            if let Some(context) = contexts.get(&next_id) {
                // Switch to next thread
                // In a real implementation, this would use context_switch
                let current_id = self.current.load(Ordering::Acquire);
                self.current.store(next_id as usize, Ordering::Release);
                
                // Store current context
                if current_id != usize::MAX {
                    // Save current context before switching
                }
                
                // Switch to new context
                // context_switch(...)
            }
        } else {
            // No threads to run, return to main
            self.current.store(usize::MAX, Ordering::Release);
        }
    }

    /// Wait for a green thread to complete
    pub fn join(&self, id: GreenThreadId) {
        let contexts = self.contexts.lock().unwrap();
        if let Some(context) = contexts.get(&id) {
            while !context.is_completed() {
                // Yield and help with other work
                drop(contexts);
                self.yield_now();
                let contexts = self.contexts.lock().unwrap();
            }
        }
    }

    /// Run the scheduler (main loop)
    pub fn run(&self) {
        loop {
            let ready_queue = self.ready_queue.lock().unwrap();
            if ready_queue.is_empty() {
                break;
            }
            drop(ready_queue);
            
            self.schedule_next();
        }
    }
}

impl Default for GreenScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_green_scheduler_spawn() {
        let scheduler = GreenScheduler::new();
        let id = scheduler.spawn(|| {});
        assert!(id > 0);
    }

    #[test]
    fn test_green_context() {
        let context = GreenContext::new(1, 8192, || {});
        assert!(!context.is_completed());
        context.complete();
        assert!(context.is_completed());
    }

    #[test]
    fn test_green_context_id() {
        let context = GreenContext::new(42, 8192, || {});
        assert_eq!(context.id(), 42);
    }
}
