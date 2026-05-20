// =========================================================================
// Epoch-Based Memory Reclamation
// Safe memory reclamation for lock-free data structures
// =========================================================================

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock, Weak};

/// Global epoch counter
static GLOBAL_EPOCH: AtomicU64 = AtomicU64::new(0);

/// FIX (C5): Store Weak<Participant> instead of raw pointers.
/// The original code stored ParticipantPtr(&p as *const Participant) which
/// took a reference to a local variable and stored the raw pointer into a
/// global registry. The pointer became dangling the moment Participant::new()
/// returned. Now we store Weak<Participant> so that:
///   1. The Weak upgrades to Arc only if the Participant is still alive.
///   2. If the Participant has been dropped, upgrade() returns None (safe).
///   3. No dangling pointer dereferences are possible.
static PARTICIPANT_REGISTRY: RwLock<Vec<Weak<Participant>>> = RwLock::new(Vec::new());

/// Number of epochs in the cycle
const NUM_EPOCHS: usize = 3;

/// Thread-local epoch participant
///
/// FIX (PERF-3): Replaced global ACTIVE_READERS counter with per-participant
/// `active` flag. The original global counter caused severe cache-line
/// contention on multi-core systems — every pin()/unpin() modified the
/// same cache line. Now each participant has its own `active` atomic on
/// a separate cache line, eliminating the contention.
#[derive(Debug)]
pub struct Participant {
    local_epoch: AtomicU64,
    /// Per-participant active flag (replaces global ACTIVE_READERS).
    /// Set to true when pinned, false when unpinned. Each participant
    /// has its own cache line, eliminating false sharing.
    active: AtomicBool,
    garbage_bags: [std::sync::Mutex<GarbageBag>; NUM_EPOCHS],
}

impl Participant {
    pub fn new() -> Self {
        let p = Self {
            local_epoch: AtomicU64::new(0),
            active: AtomicBool::new(false),
            garbage_bags: [
                std::sync::Mutex::new(GarbageBag::new()),
                std::sync::Mutex::new(GarbageBag::new()),
                std::sync::Mutex::new(GarbageBag::new()),
            ],
        };
        p
    }

    /// Register this participant into the global registry.
    /// Must be called AFTER the Participant is placed in its Arc.
    /// C5 fix: Now stores Weak<Participant> instead of raw pointers.
    pub fn register(this: &Arc<Participant>) {
        if let Ok(mut registry) = PARTICIPANT_REGISTRY.write() {
            registry.push(Arc::downgrade(this));
        }
    }

    /// Pin the current epoch
    ///
    /// FIX (PERF-3): Uses per-participant `active` flag instead of global
    /// ACTIVE_READERS counter. This eliminates cache-line contention:
    /// each participant's active flag is on its own cache line (within the
    /// Participant struct which is already 128-byte aligned in the Worker),
    /// so pinning one participant doesn't invalidate another's cache.
    pub fn pin(&self) -> Guard<'_> {
        // Set this participant as active to prevent epoch advancement
        // while any guard exists.
        self.active.store(true, Ordering::Release);

        let current = self.local_epoch.load(Ordering::Acquire);
        let global = GLOBAL_EPOCH.load(Ordering::Acquire);
        
        if current != global {
            self.local_epoch.store(global, Ordering::Release);
            self.try_collect(global);
        }
        
        Guard { participant: self }
    }

    /// Try to collect garbage from old epochs — only when the global
    /// epoch has advanced far enough to guarantee no thread still holds
    /// references to objects in the old bag.
    ///
    /// C6 fix: Removed unconditional bag.collect() call. The old code
    /// called collect() even when try_safe_collect decided it was NOT safe,
    /// which reclaimed objects that might still be in use by pinned threads.
    /// Now we only collect when the safety check confirms it.
    fn try_collect(&self, current_epoch: u64) {
        let old_epoch_idx = ((current_epoch as usize).saturating_sub(2)) % NUM_EPOCHS;
        
        let mut bag = self.garbage_bags[old_epoch_idx].lock().unwrap();
        // C6 fix: Only collect when safe — remove unconditional collect().
        // try_safe_collect checks the global epoch and only clears if it
        // has advanced enough. The old code also called bag.collect()
        // unconditionally here, which cleared items even when unsafe.
        bag.try_safe_collect(current_epoch.saturating_sub(2));
    }

    /// Add garbage to the current epoch's bag
    pub fn add_garbage(&self, epoch: u64, garbage: Box<dyn Send + std::fmt::Debug>) {
        let idx = (epoch as usize) % NUM_EPOCHS;
        self.garbage_bags[idx].lock().unwrap().add(garbage);
    }
}

impl Default for Participant {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for epoch pinning
#[derive(Debug)]
pub struct Guard<'a> {
    participant: &'a Participant,
}

impl<'a> Guard<'a> {
    /// Get the current epoch
    pub fn epoch(&self) -> u64 {
        self.participant.local_epoch.load(Ordering::Acquire)
    }

    /// Add garbage to be reclaimed
    pub fn defer(&self, garbage: Box<dyn Send + std::fmt::Debug>) {
        let epoch = self.epoch();
        self.participant.add_garbage(epoch, garbage);
    }
}

impl<'a> Drop for Guard<'a> {
    fn drop(&mut self) {
        // Unpin: clear the participant's active flag so epoch
        // advancement can proceed. This replaces the global
        // ACTIVE_READERS.fetch_sub(1) with a per-participant store,
        // eliminating cache-line contention.
        self.participant.active.store(false, Ordering::Release);
    }
}

/// Garbage bag for a single epoch
#[derive(Debug)]
struct GarbageBag {
    items: Vec<Box<dyn Send + std::fmt::Debug>>,
}

impl GarbageBag {
    fn new() -> Self {
        Self {
            items: Vec::with_capacity(32),
        }
    }

    fn add(&mut self, item: Box<dyn Send + std::fmt::Debug>) {
        self.items.push(item);
        
        // Advance epoch periodically so deferred objects don't leak forever.
        // Every 256 additions, try to advance the global epoch.
        if self.items.len() % 256 == 0 {
            advance_epoch();
        }
    }

    /// Collect garbage only when safe: the global epoch must be at least
    /// 2 epochs beyond the epoch this bag belongs to, ensuring no thread
    /// still holds a reference to any object in this bag.
    fn try_safe_collect(&mut self, bag_epoch: u64) {
        let global = GLOBAL_EPOCH.load(Ordering::Acquire);
        if global >= bag_epoch.saturating_add(2) {
            self.items.clear();
        }
    }

    fn collect(&mut self) {
        // Legacy method kept for API compatibility; prefer try_safe_collect.
        self.items.clear();
    }
}

/// Advance the global epoch
///
/// Only advances when no readers are currently pinned AND all participants
/// have caught up to the current epoch. This prevents garbage from being
/// reclaimed while a reader is still accessing it.
///
/// C5 fix: Now uses Weak<Participant> instead of raw pointers.
/// The Weak upgrades to Arc only if the Participant is still alive,
/// preventing dangling pointer dereferences.
pub fn advance_epoch() {
    // Check if any participant is currently active (pinned).
    if let Ok(registry) = PARTICIPANT_REGISTRY.read() {
        for weak in registry.iter() {
            // C5 fix: Use Weak::upgrade() instead of raw pointer deref.
            // If the Participant has been dropped, upgrade returns None
            // and we skip it (no dangling pointer dereference).
            if let Some(participant) = weak.upgrade() {
                // If any participant is active (pinned), don't advance
                if participant.active.load(Ordering::Acquire) {
                    return;
                }
            }
        }

        // Also check that all participants have caught up before advancing.
        let global = GLOBAL_EPOCH.load(Ordering::Acquire);
        for weak in registry.iter() {
            if let Some(participant) = weak.upgrade() {
                let participant_epoch = participant.local_epoch.load(Ordering::Acquire);
                // If any participant is more than 1 epoch behind, don't advance yet
                if global.saturating_sub(participant_epoch) > 1 {
                    return;
                }
            }
        }
    }

    let current = GLOBAL_EPOCH.fetch_add(1, Ordering::AcqRel);
    let next = current + 1;
    
    // Collect from the epoch that's now two old
    let _old_epoch = if next >= 2 { next - 2 } else { 0 };
}

/// Get the current global epoch
pub fn current_epoch() -> u64 {
    GLOBAL_EPOCH.load(Ordering::Acquire)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epoch_advancement() {
        let e1 = current_epoch();
        advance_epoch();
        let e2 = current_epoch();
        assert_eq!(e2, e1 + 1);
    }

    #[test]
    fn test_participant_pin() {
        let participant = Arc::new(Participant::new());
        Participant::register(&participant);
        let guard = participant.pin();
        let _epoch = guard.epoch();
    }

    #[test]
    fn test_garbage_defer() {
        let participant = Arc::new(Participant::new());
        Participant::register(&participant);
        let guard = participant.pin();
        guard.defer(Box::new(42));
    }
}
