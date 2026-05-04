// =========================================================================
// Zero-Copy Pipeline Architecture
// LMAX Disruptor pattern at scale for ultra-fast inter-thread messaging
// Ring buffer with sequence numbers for lock-free communication
// Ownership transfer without data movement
// =========================================================================

use std::sync::atomic::{AtomicU64, AtomicPtr, Ordering};
use std::sync::Arc;
use std::ptr;

/// Cache line size for alignment
const CACHE_LINE_SIZE: usize = 64;

/// Ring buffer capacity (must be power of 2)
const RING_CAPACITY: usize = 1024;

/// Sequence number type
#[allow(dead_code)]
type Sequence = u64;

/// Ring buffer entry
#[repr(C, align(64))]
pub struct RingEntry<T> {
    /// Sequence number
    sequence: AtomicU64,
    /// Data pointer (zero-copy ownership transfer)
    data: AtomicPtr<T>,
    /// Padding to prevent false sharing
    _pad: [u8; CACHE_LINE_SIZE - std::mem::size_of::<AtomicU64>() - std::mem::size_of::<usize>() * 2],
}

impl<T> RingEntry<T> {
    fn new() -> Self {
        Self {
            sequence: AtomicU64::new(0),
            data: AtomicPtr::new(ptr::null_mut()),
            _pad: [0; CACHE_LINE_SIZE - std::mem::size_of::<AtomicU64>() - std::mem::size_of::<usize>() * 2],
        }
    }
}

/// Disruptor-style ring buffer
pub struct DisruptorRing<T> {
    /// Ring buffer entries
    entries: Vec<RingEntry<T>>,
    /// Capacity (must be power of 2)
    capacity: usize,
    /// Mask for modulo operation
    mask: usize,
    /// Producer sequence
    producer_sequence: AtomicU64,
    /// Consumer sequence
    consumer_sequence: AtomicU64,
}

impl<T> DisruptorRing<T> {
    /// Create a new disruptor ring
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        let mut entries: Vec<RingEntry<T>> = Vec::with_capacity(capacity);
        
        for _ in 0..capacity {
            entries.push(RingEntry::new());
        }
        
        Self {
            entries,
            capacity,
            mask: capacity - 1,
            producer_sequence: AtomicU64::new(0),
            consumer_sequence: AtomicU64::new(0),
        }
    }
    
    /// Claim a slot in the ring (producer)
    pub fn claim(&self) -> Option<usize> {
        let current = self.producer_sequence.load(Ordering::Acquire);
        let next = current.wrapping_add(1);
        let consumer = self.consumer_sequence.load(Ordering::Acquire);
        
        // Check if ring is full
        if next.wrapping_sub(consumer) > self.capacity as u64 {
            return None;
        }
        
        let idx = (current as usize) & self.mask;
        
        // Wait for the slot to be available
        let entry = &self.entries[idx];
        let expected_sequence = current;
        
        if entry.sequence.load(Ordering::Acquire) != expected_sequence {
            return None;
        }
        
        // Claim the slot
        if self.producer_sequence.compare_exchange_weak(
            current,
            next,
            Ordering::AcqRel,
            Ordering::Acquire,
        ).is_ok() {
            Some(idx)
        } else {
            None
        }
    }
    
    /// Publish data to a claimed slot (producer)
    pub fn publish(&self, idx: usize, data: *mut T) {
        let entry = &self.entries[idx];
        entry.data.store(data, Ordering::Release);
        
        // Update sequence to publish
        let current = entry.sequence.load(Ordering::Acquire);
        entry.sequence.store(current.wrapping_add(1), Ordering::Release);
    }
    
    /// Try to claim and publish in one operation (producer)
    pub fn try_publish(&self, data: *mut T) -> bool {
        if let Some(idx) = self.claim() {
            self.publish(idx, data);
            true
        } else {
            false
        }
    }
    
    /// Get the next available slot (consumer)
    pub fn next(&self) -> Option<usize> {
        let current = self.consumer_sequence.load(Ordering::Acquire);
        let idx = (current as usize) & self.mask;
        
        let entry = &self.entries[idx];
        let expected_sequence = current.wrapping_add(1);
        
        if entry.sequence.load(Ordering::Acquire) == expected_sequence {
            Some(idx)
        } else {
            None
        }
    }
    
    /// Consume data from a slot (consumer)
    pub fn consume(&self, idx: usize) -> *mut T {
        let entry = &self.entries[idx];
        let data = entry.data.load(Ordering::Acquire);
        
        // Update consumer sequence
        let current = self.consumer_sequence.load(Ordering::Acquire);
        self.consumer_sequence.store(current.wrapping_add(1), Ordering::Release);
        
        // Mark slot as available for producer
        entry.sequence.store(current.wrapping_add(self.capacity as u64), Ordering::Release);
        
        data
    }
    
    /// Try to consume in one operation (consumer)
    pub fn try_consume(&self) -> Option<*mut T> {
        if let Some(idx) = self.next() {
            Some(self.consume(idx))
        } else {
            None
        }
    }
    
    /// Get the producer sequence
    pub fn producer_sequence(&self) -> u64 {
        self.producer_sequence.load(Ordering::Acquire)
    }
    
    /// Get the consumer sequence
    pub fn consumer_sequence(&self) -> u64 {
        self.consumer_sequence.load(Ordering::Acquire)
    }
    
    /// Get the available slots
    pub fn available(&self) -> usize {
        let producer = self.producer_sequence.load(Ordering::Acquire);
        let consumer = self.consumer_sequence.load(Ordering::Acquire);
        (producer.wrapping_sub(consumer)) as usize
    }

    /// Check if the ring is empty
    pub fn is_empty(&self) -> bool {
        self.available() == 0
    }

    /// Check if the ring is full
    pub fn is_full(&self) -> bool {
        self.available() >= self.capacity
    }
}

impl<T> Default for DisruptorRing<T> {
    fn default() -> Self {
        Self::new(RING_CAPACITY)
    }
}

/// Per-worker disruptor ring for zero-copy messaging
pub struct WorkerDisruptor<T> {
    /// Ring buffer
    ring: DisruptorRing<T>,
    /// Worker ID
    worker_id: usize,
}

impl<T> WorkerDisruptor<T> {
    /// Create a new worker disruptor
    pub fn new(worker_id: usize, capacity: usize) -> Self {
        Self {
            ring: DisruptorRing::<T>::new(capacity),
            worker_id,
        }
    }
    
    /// Send data to this worker (zero-copy)
    pub fn send(&self, data: *mut T) -> bool {
        self.ring.try_publish(data)
    }
    
    /// Receive data (zero-copy)
    pub fn receive(&self) -> Option<*mut T> {
        self.ring.try_consume()
    }
    
    /// Get the ring
    pub fn ring(&self) -> &DisruptorRing<T> {
        &self.ring
    }
    
    /// Get the worker ID
    pub fn worker_id(&self) -> usize {
        self.worker_id
    }
}

/// Zero-copy messaging system
pub struct ZeroCopyMessaging<T> {
    /// Per-worker disruptor rings
    worker_rings: Vec<Arc<WorkerDisruptor<T>>>,
    /// Number of workers
    num_workers: usize,
}

impl<T> ZeroCopyMessaging<T> {
    /// Create a new zero-copy messaging system
    pub fn new(num_workers: usize, ring_capacity: usize) -> Self {
        let mut worker_rings = Vec::with_capacity(num_workers);
        
        for worker_id in 0..num_workers {
            worker_rings.push(Arc::new(WorkerDisruptor::<T>::new(worker_id, ring_capacity)));
        }
        
        Self {
            worker_rings,
            num_workers,
        }
    }
    
    /// Send data to a specific worker (zero-copy)
    pub fn send_to_worker(&self, worker_id: usize, data: *mut T) -> bool {
        if worker_id >= self.num_workers {
            return false;
        }
        
        self.worker_rings[worker_id].send(data)
    }
    
    /// Receive data for a specific worker (zero-copy)
    pub fn receive_for_worker(&self, worker_id: usize) -> Option<*mut T> {
        if worker_id >= self.num_workers {
            return None;
        }
        
        self.worker_rings[worker_id].receive()
    }
    
    /// Get a worker's disruptor ring
    pub fn worker_ring(&self, worker_id: usize) -> Option<&Arc<WorkerDisruptor<T>>> {
        self.worker_rings.get(worker_id)
    }
    
    /// Get the number of workers
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
}

/// Ownership transfer wrapper
/// Ensures that data is not used after transfer
pub struct OwnedData<T> {
    /// Data pointer
    data: *mut T,
    /// Ownership flag
    owned: bool,
}

impl<T> OwnedData<T> {
    /// Create new owned data
    pub fn new(data: *mut T) -> Self {
        Self {
            data,
            owned: true,
        }
    }
    
    /// Take ownership (consume)
    pub fn take(mut self) -> *mut T {
        self.owned = false;
        self.data
    }
    
    /// Get the data pointer (without taking ownership)
    pub fn get(&self) -> *mut T {
        self.data
    }
    
    /// Check if still owned
    pub fn is_owned(&self) -> bool {
        self.owned
    }
}

impl<T> Drop for OwnedData<T> {
    fn drop(&mut self) {
        if self.owned && !self.data.is_null() {
            // Deallocate the data since we own it
            unsafe {
                let _ = Box::from_raw(self.data);
            }
            self.owned = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disruptor_ring_creation() {
        let ring: DisruptorRing<i32> = DisruptorRing::new(64);
        assert_eq!(ring.capacity, 64);
        assert!(ring.is_empty());
    }

    #[test]
    fn test_disruptor_publish_consume() {
        let ring: DisruptorRing<i32> = DisruptorRing::new(64);
        
        let data = Box::into_raw(Box::new(42));
        assert!(ring.try_publish(data));
        
        let consumed = ring.try_consume();
        assert!(consumed.is_some());
        assert!(!consumed.unwrap().is_null());
    }

    #[test]
    fn test_worker_disruptor() {
        let worker = WorkerDisruptor::<i32>::new(0, 64);
        
        let data = Box::into_raw(Box::new(42));
        assert!(worker.send(data));
        
        let received = worker.receive();
        assert!(received.is_some());
    }

    #[test]
    fn test_zero_copy_messaging() {
        let messaging = ZeroCopyMessaging::<i32>::new(4, 64);
        
        let data = Box::into_raw(Box::new(42));
        assert!(messaging.send_to_worker(0, data));
        
        let received = messaging.receive_for_worker(0);
        assert!(received.is_some());
    }

    #[test]
    fn test_owned_data() {
        let data = Box::into_raw(Box::new(42));
        let owned = OwnedData::new(data);
        
        assert!(owned.is_owned());
        let taken = owned.take();
        // After take(), owned is moved; verify the taken pointer is valid
        assert!(!taken.is_null());
    }
}
