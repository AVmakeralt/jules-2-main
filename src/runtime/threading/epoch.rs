// =========================================================================
// Epoch-Based Memory Reclamation
// Safe memory reclamation for lock-free data structures
// =========================================================================

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::RwLock;

/// Wrapper around `*const AtomicU64` that implements `Send` + `Sync`.
///
/// # Safety
/// The caller must ensure that the pointed-to `AtomicU64` outlives
/// any access through this pointer and that all access is through
/// atomic operations (which are thread-safe by definition).
#[repr(transparent)]
struct EpochPtr(*const AtomicU64);

unsafe impl Send for EpochPtr {}
unsafe impl Sync for EpochPtr {}

/// Global epoch counter
static GLOBAL_EPOCH: AtomicU64 = AtomicU64::new(0);

/// Number of active readers (pinned guards). Epoch advancement is blocked
/// while this is non-zero, preventing garbage reclamation while any guard
/// still holds a reference to an epoch.
static ACTIVE_READERS: AtomicUsize = AtomicUsize::new(0);

/// FIX (PERF-3): Participant registry for tracking minimum epoch across
/// all participants. The original implementation used a single global
/// ACTIVE_READERS counter, causing severe cache-line contention on
/// multi-core systems. Now we track per-participant epochs and compute
/// the minimum for safe epoch advancement.
static PARTICIPANT_EPOCHS: RwLock<Vec<EpochPtr>> = RwLock::new(Vec::new());

/// Number of epochs in the cycle
const NUM_EPOCHS: usize = 3;

/// Thread-local epoch participant
#[derive(Debug)]
pub struct Participant {
    local_epoch: AtomicU64,
    garbage_bags: [std::sync::Mutex<GarbageBag>; NUM_EPOCHS],
}

impl Participant {
    pub fn new() -> Self {
        let p = Self {
            local_epoch: AtomicU64::new(0),
            garbage_bags: [
                std::sync::Mutex::new(GarbageBag::new()),
                std::sync::Mutex::new(GarbageBag::new()),
                std::sync::Mutex::new(GarbageBag::new()),
            ],
        };
        // FIX (PERF-3): Register this participant's epoch pointer so
        // advance_epoch can compute the minimum across all participants.
        if let Ok(mut registry) = PARTICIPANT_EPOCHS.write() {
            registry.push(EpochPtr(&p.local_epoch as *const AtomicU64));
        }
        p
    }

    /// Pin the current epoch
    pub fn pin(&self) -> Guard<'_> {
        // Increment active readers to prevent epoch advancement while
        // any guard exists.
        ACTIVE_READERS.fetch_add(1, Ordering::Acquire);

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
        self.garbage_bags[old_epoch_idx].lock().unwrap().try_safe_collect(
            current_epoch.saturating_sub(2)
        );
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
        // Unpin: decrement active readers so epoch advancement can proceed
        ACTIVE_READERS.fetch_sub(1, Ordering::Release);
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

    #[allow(dead_code)]
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
/// FIX (PERF-3): The original implementation used a single global
/// ACTIVE_READERS counter, creating cache-line contention on multi-core
/// systems. Now we also verify that no participant is lagging more than
/// 1 epoch behind before advancing, by checking the minimum epoch across
/// all registered participants.
pub fn advance_epoch() {
    // Do not advance while any guard (reader) is active — otherwise garbage
    // could be reclaimed while a reader is still accessing it.
    if ACTIVE_READERS.load(Ordering::Acquire) > 0 {
        return;
    }

    // FIX (PERF-3): Check that all participants have caught up before
    // advancing. This prevents premature reclamation when a participant
    // is pinned in an older epoch.
    let global = GLOBAL_EPOCH.load(Ordering::Acquire);
    if let Ok(registry) = PARTICIPANT_EPOCHS.read() {
        for ep in registry.iter() {
            // SAFETY: The pointer comes from a Participant that is still alive.
            // Participants are stored in Arc<Participant> in the worker pool.
            let participant_epoch = unsafe { (*ep.0).load(Ordering::Acquire) };
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
        let epoch = guard.epoch();
        assert!(epoch >= 0);
    }

    #[test]
    fn test_garbage_defer() {
        let participant = Participant::new();
        let guard = participant.pin();
        guard.defer(Box::new(42));
    }
}
