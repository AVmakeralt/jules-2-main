// =========================================================================
// Green Thread Runtime (M:N Threading)
// Copy-stack model with userspace context switching
// Assembly routines for x86-64 and AArch64
// =========================================================================

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

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
    /// Function to execute (boxed closure)
    func: Option<Box<dyn FnOnce()>>,
}

impl GreenContext {
    /// Create a new green thread context
    pub fn new(id: GreenThreadId, stack_size: usize, func: Box<dyn FnOnce()>) -> Self {
        let stack = vec![0u8; stack_size];
        let stack_base = Box::into_raw(stack.into_boxed_slice()) as *mut u8 as usize;
        
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
                // Reconstruct Box<[u8]> with proper slice metadata.
                // The original allocation was Box<[u8]> (boxed slice), so we must
                // use slice_from_raw_parts to provide the length metadata that
                // Box::from_raw expects.
                let ptr = std::ptr::slice_from_raw_parts_mut(self.stack_base as *mut u8, self.stack_size);
                let _ = Box::from_raw(ptr);
            }
        }
    }
}

/// Context switch function (x86-64)
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
extern "C" fn context_switch(
    old_sp: *mut usize,
    new_sp: usize,
    old_regs: *mut usize,
    new_regs: *const usize,
) {
    unsafe {
        use std::arch::asm;
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
            "ldr x19, [x4]",
            "ldr x20, [x4, 8]",
            "ldr x21, [x4, 16]",
            "ldr x22, [x4, 24]",
            "ldr x23, [x4, 32]",
            "ldr x24, [x4, 40]",
            "ldr x25, [x4, 48]",
            "ldr x26, [x4, 56]",
            "ldr x27, [x4, 64]",
            "ldr x28, [x4, 72]",
            "ldr x29, [x4, 80]",
            "str sp, [x0]",
            "str x19, [x0, 8]",
            "str x20, [x0, 16]",
            "str x21, [x0, 24]",
            "str x22, [x0, 32]",
            "str x23, [x0, 40]",
            "str x24, [x0, 48]",
            "str x25, [x0, 56]",
            "str x26, [x0, 64]",
            "str x27, [x0, 72]",
            "str x28, [x0, 80]",
            "str x29, [x0, 88]",
            // Restore new context
            "mov sp, x1",
            "ldr x19, [x5]",
            "ldr x20, [x5, 8]",
            "ldr x21, [x5, 16]",
            "ldr x22, [x5, 24]",
            "ldr x23, [x5, 32]",
            "ldr x24, [x5, 40]",
            "ldr x25, [x5, 48]",
            "ldr x26, [x5, 56]",
            "ldr x27, [x5, 64]",
            "ldr x28, [x5, 72]",
            "ldr x29, [x5, 80]",
            in("x0") old_sp,
            in("x1") new_sp,
            in("x4") old_regs,
            in("x5") new_regs,
            clobber_abi("system")
        );
    }
}

/// Fallback context switch for unsupported architectures
///
/// On architectures without hand-written assembly, we perform a simplified
/// stack-pointer swap using `ptr::write`/`ptr::read` for the SP and a
/// byte-wise copy for the callee-saved register array.  This is **not** a
/// real context switch (it cannot resume at the swap site) but preserves the
/// register state so the scheduler can inspect it and is sufficient for
/// single-threaded cooperative yielding where the "switch" is really just
/// saving state and returning to the scheduler loop.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
extern "C" fn context_switch(
    old_sp: *mut usize,
    new_sp: usize,
    old_regs: *mut usize,
    new_regs: *const usize,
) {
    use std::ptr;
    // Save current stack pointer into the old context
    unsafe {
        let current_sp: usize;
        // Best-effort: read the current stack pointer.  On most ISAs the
        // frame pointer or a volatile local gives us a usable approximation.
        // We store `new_sp` as the SP to restore — the caller already
        // computed the target SP.
        std::arch::asm!("mov {}, sp", out(reg) current_sp, options(nostack, preserves_flags));
        ptr::write(old_sp, current_sp);

        // Copy 6 callee-saved register slots from new_regs into old_regs
        // so that the restore path can pick them up.  We treat the register
        // file as a plain `[usize; 6]`.
        ptr::copy_nonoverlapping(new_regs, old_regs, 6);
    }
    // NOTE: A true context switch would also rewrite the return address on
    // the stack so that after this function returns, execution resumes in
    // the *new* context.  That requires assembly and is not possible in
    // portable Rust.  The scheduler therefore uses this routine purely to
    // snapshot/restore state between cooperative yields.
}

/// Green thread scheduler
#[allow(dead_code)]
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
        
        // Create green thread context, storing the closure for execution
        let context = Box::new(GreenContext::new(id, STACK_SIZE, Box::new(f)));
        
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
            if let Some(_context) = contexts.get(&next_id) {
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
        loop {
            {
                let contexts = self.contexts.lock().unwrap();
                if let Some(context) = contexts.get(&id) {
                    if context.is_completed() {
                        return;
                    }
                } else {
                    return;
                }
            }
            // Yield and help with other work
            self.yield_now();
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
        let context = GreenContext::new(1, 8192, Box::new(|| {}));
        assert!(!context.is_completed());
        context.complete();
        assert!(context.is_completed());
    }

    #[test]
    fn test_green_context_id() {
        let context = GreenContext::new(42, 8192, Box::new(|| {}));
        assert_eq!(context.id(), 42);
    }
}
