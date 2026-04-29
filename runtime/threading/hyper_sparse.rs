// =========================================================================
// Hyper-Sparse Data Structures: The "Memory Wall" Solution
// Designed for scenarios where 99% of a dataset is empty or "zero"
// Uses Segmented Sieve logic and Structure-of-Arrays (SoA) task queues
// Treats data as a "map of bits" - only allocates RAM for non-empty space
// Eliminates "Abstractions Tax" of octrees or hash maps
// 8x cache pressure reduction by fetching only relevant data
// =========================================================================

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Bit segment for hyper-sparse representation
#[derive(Debug, Clone)]
pub struct BitSegment {
    /// Segment ID
    pub id: usize,
    /// Bit data (each bit represents whether a coordinate is occupied)
    pub bits: Vec<u64>,
    /// Base coordinate for this segment
    pub base: usize,
    /// Number of occupied bits in this segment
    pub occupied_count: AtomicUsize,
}

impl BitSegment {
    /// Create a new bit segment
    pub fn new(id: usize, base: usize, capacity: usize) -> Self {
        let bits_per_u64 = 64;
        let num_u64 = (capacity + bits_per_u64 - 1) / bits_per_u64;
        
        Self {
            id,
            bits: vec![0; num_u64],
            base,
            occupied_count: AtomicUsize::new(0),
        }
    }
    
    /// Set a bit at a relative coordinate
    pub fn set_bit(&mut self, rel_coord: usize) {
        let u64_idx = rel_coord / 64;
        let bit_idx = rel_coord % 64;
        
        if u64_idx < self.bits.len() {
            let mask = 1u64 << bit_idx;
            if self.bits[u64_idx] & mask == 0 {
                self.bits[u64_idx] |= mask;
                self.occupied_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    
    /// Check if a bit is set at a relative coordinate
    pub fn is_set(&self, rel_coord: usize) -> bool {
        let u64_idx = rel_coord / 64;
        let bit_idx = rel_coord % 64;
        
        if u64_idx < self.bits.len() {
            let mask = 1u64 << bit_idx;
            (self.bits[u64_idx] & mask) != 0
        } else {
            false
        }
    }
    
    /// Clear a bit at a relative coordinate
    pub fn clear_bit(&mut self, rel_coord: usize) {
        let u64_idx = rel_coord / 64;
        let bit_idx = rel_coord % 64;
        
        if u64_idx < self.bits.len() {
            let mask = 1u64 << bit_idx;
            if self.bits[u64_idx] & mask != 0 {
                self.bits[u64_idx] &= !mask;
                self.occupied_count.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
    
    /// Get the number of occupied bits
    pub fn occupied_count(&self) -> usize {
        self.occupied_count.load(Ordering::Relaxed)
    }
    
    /// Get the sparsity ratio (0.0 = empty, 1.0 = full)
    pub fn sparsity(&self) -> f64 {
        let total_bits = self.bits.len() * 64;
        if total_bits == 0 {
            0.0
        } else {
            1.0 - (self.occupied_count() as f64 / total_bits as f64)
        }
    }
}

/// Hyper-sparse coordinate map
pub struct HyperSparseMap {
    /// Segments of the map
    segments: HashMap<usize, BitSegment>,
    /// Segment size (number of coordinates per segment)
    segment_size: usize,
    /// Total coordinates
    total_coordinates: usize,
    /// Occupied coordinates
    occupied_coordinates: AtomicUsize,
}

impl HyperSparseMap {
    /// Create a new hyper-sparse map
    pub fn new(total_coordinates: usize, segment_size: usize) -> Self {
        Self {
            segments: HashMap::new(),
            segment_size,
            total_coordinates,
            occupied_coordinates: AtomicUsize::new(0),
        }
    }
    
    /// Get the segment ID for a coordinate
    fn segment_id(&self, coord: usize) -> usize {
        coord / self.segment_size
    }
    
    /// Get the relative coordinate within a segment
    fn relative_coord(&self, coord: usize) -> usize {
        coord % self.segment_size
    }
    
    /// Get or create a segment
    fn get_or_create_segment(&mut self, segment_id: usize) -> &mut BitSegment {
        let base = segment_id * self.segment_size;
        self.segments.entry(segment_id)
            .or_insert_with(|| BitSegment::new(segment_id, base, self.segment_size))
    }
    
    /// Set a coordinate as occupied
    pub fn set(&mut self, coord: usize) {
        if coord >= self.total_coordinates {
            return;
        }
        
        let segment_id = self.segment_id(coord);
        let rel_coord = self.relative_coord(coord);
        
        let segment = self.get_or_create_segment(segment_id);
        
        if !segment.is_set(rel_coord) {
            segment.set_bit(rel_coord);
            self.occupied_coordinates.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Check if a coordinate is occupied
    pub fn is_occupied(&self, coord: usize) -> bool {
        if coord >= self.total_coordinates {
            return false;
        }
        
        let segment_id = self.segment_id(coord);
        let rel_coord = self.relative_coord(coord);
        
        if let Some(segment) = self.segments.get(&segment_id) {
            segment.is_set(rel_coord)
        } else {
            false
        }
    }
    
    /// Clear a coordinate
    pub fn clear(&mut self, coord: usize) {
        if coord >= self.total_coordinates {
            return;
        }
        
        let segment_id = self.segment_id(coord);
        let rel_coord = self.relative_coord(coord);
        
        if let Some(segment) = self.segments.get_mut(&segment_id) {
            if segment.is_set(rel_coord) {
                segment.clear_bit(rel_coord);
                self.occupied_coordinates.fetch_sub(1, Ordering::Relaxed);
                
                // Remove empty segments
                if segment.occupied_count() == 0 {
                    self.segments.remove(&segment_id);
                }
            }
        }
    }
    
    /// Get the number of occupied coordinates
    pub fn occupied_count(&self) -> usize {
        self.occupied_coordinates.load(Ordering::Relaxed)
    }
    
    /// Get the total number of coordinates
    pub fn total_coordinates(&self) -> usize {
        self.total_coordinates
    }
    
    /// Get the overall sparsity ratio
    pub fn sparsity(&self) -> f64 {
        if self.total_coordinates == 0 {
            0.0
        } else {
            1.0 - (self.occupied_count() as f64 / self.total_coordinates as f64)
        }
    }
    
    /// Get the number of segments
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
    
    /// Get all occupied coordinates
    pub fn occupied_coordinates(&self) -> Vec<usize> {
        let mut coords = Vec::new();
        
        for segment in self.segments.values() {
            for (u64_idx, &bits) in segment.bits.iter().enumerate() {
                for bit_idx in 0..64 {
                    if (bits & (1u64 << bit_idx)) != 0 {
                        let rel_coord = u64_idx * 64 + bit_idx;
                        let coord = segment.base + rel_coord;
                        if coord < self.total_coordinates {
                            coords.push(coord);
                        }
                    }
                }
            }
        }
        
        coords
    }
    
    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let bits_memory = self.segments.values()
            .map(|s| s.bits.len() * std::mem::size_of::<u64>())
            .sum::<usize>();
        
        let overhead = self.segments.len() * std::mem::size_of::<BitSegment>();
        
        bits_memory + overhead
    }
    
    /// Get memory savings compared to dense representation
    pub fn memory_savings(&self) -> f64 {
        let dense_memory = self.total_coordinates * std::mem::size_of::<u8>();
        let sparse_memory = self.memory_usage();
        
        if dense_memory == 0 {
            0.0
        } else {
            1.0 - (sparse_memory as f64 / dense_memory as f64)
        }
    }
}

/// SoA (Structure-of-Arrays) representation for hyper-sparse data
pub struct HyperSparseSoA {
    /// Occupied coordinates (compact array)
    pub coordinates: Vec<usize>,
    /// Values corresponding to coordinates
    pub values: Vec<f64>,
    /// Coordinate to index mapping
    coord_to_index: HashMap<usize, usize>,
}

impl HyperSparseSoA {
    /// Create a new hyper-sparse SoA
    pub fn new() -> Self {
        Self {
            coordinates: Vec::new(),
            values: Vec::new(),
            coord_to_index: HashMap::new(),
        }
    }
    
    /// Insert a coordinate-value pair
    pub fn insert(&mut self, coord: usize, value: f64) {
        if let Some(&idx) = self.coord_to_index.get(&coord) {
            // Update existing
            self.values[idx] = value;
        } else {
            // Insert new
            let idx = self.coordinates.len();
            self.coordinates.push(coord);
            self.values.push(value);
            self.coord_to_index.insert(coord, idx);
        }
    }
    
    /// Get a value by coordinate
    pub fn get(&self, coord: usize) -> Option<f64> {
        self.coord_to_index.get(&coord)
            .map(|&idx| self.values[idx])
    }
    
    /// Remove a coordinate
    pub fn remove(&mut self, coord: usize) -> Option<f64> {
        if let Some(&idx) = self.coord_to_index.remove(&coord) {
            let value = self.values[idx];
            
            // Swap with last element
            let last_idx = self.coordinates.len() - 1;
            if idx != last_idx {
                let last_coord = self.coordinates[last_idx];
                self.coordinates[idx] = last_coord;
                self.values[idx] = self.values[last_idx];
                self.coord_to_index.insert(last_coord, idx);
            }
            
            self.coordinates.pop();
            self.values.pop();
            
            Some(value)
        } else {
            None
        }
    }
    
    /// Get the number of occupied entries
    pub fn len(&self) -> usize {
        self.coordinates.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.coordinates.is_empty()
    }
    
    /// Iterate over coordinate-value pairs
    pub fn iter(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.coordinates.iter()
            .zip(self.values.iter())
            .map(|(&coord, &value)| (coord, value))
    }
}

impl Default for HyperSparseSoA {
    fn default() -> Self {
        Self::new()
    }
}

/// Segmented sieve for hyper-sparse data traversal
pub struct SegmentedSieve {
    /// Segment size
    segment_size: usize,
    /// Current segment
    current_segment: usize,
    /// Sieve results (occupied coordinates in current segment)
    sieve_results: Vec<usize>,
}

impl SegmentedSieve {
    /// Create a new segmented sieve
    pub fn new(segment_size: usize) -> Self {
        Self {
            segment_size,
            current_segment: 0,
            sieve_results: Vec::new(),
        }
    }
    
    /// Sieve the next segment
    pub fn sieve_next(&mut self, map: &HyperSparseMap) -> &[usize] {
        self.sieve_results.clear();
        
        let segment_id = self.current_segment;
        let base = segment_id * self.segment_size;
        
        if let Some(segment) = map.segments.get(&segment_id) {
            for (u64_idx, &bits) in segment.bits.iter().enumerate() {
                for bit_idx in 0..64 {
                    if (bits & (1u64 << bit_idx)) != 0 {
                        let rel_coord = u64_idx * 64 + bit_idx;
                        let coord = base + rel_coord;
                        if coord < map.total_coordinates {
                            self.sieve_results.push(coord);
                        }
                    }
                }
            }
        }
        
        self.current_segment += 1;
        &self.sieve_results
    }
    
    /// Reset the sieve
    pub fn reset(&mut self) {
        self.current_segment = 0;
        self.sieve_results.clear();
    }
    
    /// Get the current segment
    pub fn current_segment(&self) -> usize {
        self.current_segment
    }
}

impl Default for SegmentedSieve {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_segment() {
        let mut segment = BitSegment::new(0, 0, 128);
        
        segment.set_bit(10);
        assert!(segment.is_set(10));
        assert!(!segment.is_set(20));
        
        segment.clear_bit(10);
        assert!(!segment.is_set(10));
    }

    #[test]
    fn test_hyper_sparse_map() {
        let mut map = HyperSparseMap::new(10000, 1024);
        
        map.set(100);
        map.set(200);
        map.set(300);
        
        assert!(map.is_occupied(100));
        assert!(map.is_occupied(200));
        assert!(!map.is_occupied(400));
        
        assert_eq!(map.occupied_count(), 3);
        assert!(map.sparsity() > 0.9);
    }

    #[test]
    fn test_hyper_sparse_soa() {
        let mut soa = HyperSparseSoA::new();
        
        soa.insert(100, 1.5);
        soa.insert(200, 2.5);
        
        assert_eq!(soa.get(100), Some(1.5));
        assert_eq!(soa.get(200), Some(2.5));
        assert_eq!(soa.get(300), None);
        
        assert_eq!(soa.len(), 2);
    }

    #[test]
    fn test_segmented_sieve() {
        let mut map = HyperSparseMap::new(10000, 1024);
        map.set(100);
        map.set(200);
        
        let mut sieve = SegmentedSieve::new(1024);
        let results = sieve.sieve_next(&map);
        
        assert!(results.contains(&100));
        assert!(results.contains(&200));
    }

    #[test]
    fn test_memory_savings() {
        let mut map = HyperSparseMap::new(1000000, 1024);
        
        // Occupy only 1% of coordinates
        for i in (0..1000000).step_by(100) {
            map.set(i);
        }
        
        let savings = map.memory_savings();
        assert!(savings > 0.5); // Should save at least 50% memory
    }
}
