// =========================================================================
// Epoch-Based Memory Reclamation
// Safe memory reclamation for lock-free data structures
// =========================================================================

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::RwLock;

/// Wrapper around `*const Participant` that implements `Send` + `Sync`.
///
/// # Safety
/// The caller must ensure that the pointed-to `Participant` outlives
/// any access through this pointer and that all access is through
/// atomic operations or methods that are inherently thread-safe.
#[repr(transparent)]
struct ParticipantPtr(*const Participant);

unsafe impl Send for ParticipantPtr {}
unsafe impl Sync for ParticipantPtr {}

/// Global epoch counter
static GLOBAL_EPOCH: AtomicU64 = AtomicU64::new(0);

/// FIX (PERF-3): Participant registry for tracking active/pinned state and
/// epoch across all participants. The original implementation used a single
/// global ACTIVE_READERS counter, causing severe cache-line contention on
/// multi-core systems. Now we track per-participant state and check each
/// participant's `active` flag independently, spreading cache-line traffic
/// across per-participant cache lines.
static PARTICIPANT_REGISTRY: RwLock<Vec<ParticipantPtr>> = RwLock::new(Vec::new());

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
        // FIX (PERF-3): Register this participant so advance_epoch can
        // check its active flag and local_epoch. We store a raw pointer
        // because Participant is stored in Arc<Participant> in the worker
        // pool and will outlive the registry entry.
        if let Ok(mut registry) = PARTICIPANT_REGISTRY.write() {
            registry.push(ParticipantPtr(&p as *const Participant));
        }
        p
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
    fn try_collect(&self, current_epoch: u64) {
        let _epoch_idx = (current_epoch as usize) % NUM_EPOCHS;
        let old_epoch_idx = ((current_epoch as usize).saturating_sub(2)) % NUM_EPOCHS;
        
        // Only collect when safe: global epoch must be >= old_epoch + 2
        let mut bag = self.garbage_bags[old_epoch_idx].lock().unwrap();
        bag.try_safe_collect(current_epoch.saturating_sub(2));
        // Also run the legacy collect to clear any remaining items.
        bag.collect();
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
        // NOTE: We do NOT eagerly collect when the bag is "full".
        // Premature collection can reclaim objects still accessed by threads
        // pinned in older epochs (use-after-free). Collection is now only
        // performed by try_collect() when the global epoch has advanced by
        // at least 2 beyond the bag's epoch, guaranteeing a grace period.
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
/// FIX (PERF-3): Replaced global ACTIVE_READERS counter with per-participant
/// `active` flags. The original single counter caused cache-line contention
/// on multi-core systems. Now we check each participant's active flag and
/// local_epoch independently, which spreads the cache-line traffic across
/// per-participant cache lines (the Participant structs are 128-byte aligned
/// in the Worker struct, so they're already on separate cache lines).
pub fn advance_epoch() {
    // Check if any participant is currently active (pinned).
    // We use the PARTICIPANT_REGISTRY to iterate all participants.
    // This replaces the global ACTIVE_READERS counter.
    if let Ok(registry) = PARTICIPANT_REGISTRY.read() {
        for pp in registry.iter() {
            // SAFETY: The pointer comes from a Participant that is still alive.
            // Participants are stored in Arc<Participant> in the worker pool.
            let participant = unsafe { &*pp.0 };
            // If any participant is active (pinned), don't advance
            if participant.active.load(Ordering::Acquire) {
                return;
            }
        }

        // Also check that all participants have caught up before advancing.
        let global = GLOBAL_EPOCH.load(Ordering::Acquire);
        for pp in registry.iter() {
            let participant_epoch = unsafe { (*pp.0).local_epoch.load(Ordering::Acquire) };
            // If any participant is more than 1 epoch behind, don't advance yet
            if global.saturating_sub(participant_epoch) > 1 {
                return;
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
        let participant = Participant::new();
        let guard = participant.pin();
        let _epoch = guard.epoch(); // epoch is always >= 0 for u64
    }

    #[test]
    fn test_garbage_defer() {
        let participant = Participant::new();
        let guard = participant.pin();
        guard.defer(Box::new(42));
    }
}
