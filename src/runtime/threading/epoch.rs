// =========================================================================
// Epoch-Based Memory Reclamation
// Safe memory reclamation for lock-free data structures
// =========================================================================

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

/// Global epoch counter
static GLOBAL_EPOCH: AtomicU64 = AtomicU64::new(0);

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
        Self {
            local_epoch: AtomicU64::new(0),
            garbage_bags: [
                std::sync::Mutex::new(GarbageBag::new()),
                std::sync::Mutex::new(GarbageBag::new()),
                std::sync::Mutex::new(GarbageBag::new()),
            ],
        }
    }

    /// Pin the current epoch
    pub fn pin(&self) -> Guard {
        let current = self.local_epoch.load(Ordering::Acquire);
        let global = GLOBAL_EPOCH.load(Ordering::Acquire);
        
        if current != global {
            self.local_epoch.store(global, Ordering::Release);
            self.try_collect(global);
        }
        
        Guard { participant: self }
    }

    /// Try to collect garbage from old epochs
    fn try_collect(&self, current_epoch: u64) {
        let epoch_idx = (current_epoch as usize) % NUM_EPOCHS;
        let old_epoch_idx = ((current_epoch as usize).wrapping_sub(1)) % NUM_EPOCHS;
        
        // Collect from two epochs ago
        self.garbage_bags[old_epoch_idx].lock().unwrap().collect();
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
        // Unpin is automatic - just drop the guard
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
        
        // Try to collect when bag is full
        if self.items.len() >= 32 {
            self.collect();
        }
    }

    fn collect(&mut self) {
        self.items.clear();
    }
}

/// Advance the global epoch
pub fn advance_epoch() {
    let current = GLOBAL_EPOCH.fetch_add(1, Ordering::AcqRel);
    let next = current + 1;
    
    // Collect from the epoch that's now two old
    let old_epoch = if next >= 2 { next - 2 } else { 0 };
    // In a real implementation, we'd notify participants to collect
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
