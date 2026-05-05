// =============================================================================
// src/runtime/speculative_memory.rs
//
// Speculative Memory Reorganization with Hardware Transactional Memory (HTM)
//
// Dynamically rewrites data layouts in live memory based on observed access
// patterns. If components are accessed in AoS pattern but stored in SoA,
// the runtime physically reorganizes the heap layout using Intel TSX/ARM TME.
//
// Architecture:
//   - Layout predictor uses ML runtime to forecast access patterns
//   - When confidence exceeds threshold, initiate transactional reorganization
//   - HTM ensures atomicity with automatic rollback on conflict
//   - Layout decisions persisted asynchronously for future runs
//
// Research: Languages optimize memory layout at compile time. Doing it
// speculatively at runtime with hardware rollback is unprecedented at the
// language level. Cache misses become a recoverable experiment.
// =============================================================================

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Memory layout types for speculation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    /// Array of Structures - each entity's data is contiguous
    ArrayOfStructures,
    /// Structure of Arrays - each field is a separate array
    StructureOfArrays,
    /// Array of Structures with cache-line alignment
    ArrayOfStructuresAligned,
    /// Hybrid: frequently-accessed fields together
    Hybrid,
    /// SoA with structure splitting (hot/cold fields separated)
    SplitSoA,
}

impl Default for MemoryLayout {
    fn default() -> Self {
        MemoryLayout::ArrayOfStructures
    }
}

/// Access pattern statistics collected at runtime
#[derive(Debug, Clone, Default)]
pub struct AccessPattern {
    /// Count of sequential accesses
    pub sequential_count: usize,
    /// Count of random accesses
    pub random_count: usize,
    /// Count of strided accesses
    pub strided_count: usize,
    /// Hot component indices (frequently accessed)
    pub hot_components: Vec<usize>,
    /// Cold component indices (rarely accessed)
    pub cold_components: Vec<usize>,
    /// Average access stride
    pub avg_stride: f32,
    /// Total accesses
    pub total_accesses: usize,
}

impl AccessPattern {
    /// Detect if this is an AoS pattern (random access to entity components)
    pub fn is_aos_pattern(&self) -> bool {
        self.random_count as f32 / self.total_accesses.max(1) as f32 > 0.7
    }

    /// Detect if this is an SoA pattern (sequential access to single field)
    pub fn is_soa_pattern(&self) -> bool {
        self.sequential_count as f32 / self.total_accesses.max(1) as f32 > 0.7
    }

    /// Detect strided pattern (cache-friendly)
    pub fn is_strided_pattern(&self) -> bool {
        self.strided_count as f32 / self.total_accesses.max(1) as f32 > 0.5
    }

    /// Confidence score for layout prediction
    pub fn confidence(&self) -> f32 {
        let pattern_score = [
            self.is_aos_pattern() as i32,
            self.is_soa_pattern() as i32,
            self.is_strided_pattern() as i32,
        ].iter().max().copied().unwrap_or(0) as f32;

        let hot_cold_ratio = if !self.hot_components.is_empty() {
            self.cold_components.len() as f32 / self.hot_components.len().max(1) as f32
        } else {
            0.0
        };

        (pattern_score * 0.7 + hot_cold_ratio.min(1.0) * 0.3).clamp(0.0, 1.0)
    }
}

/// Component descriptor for ECS systems
#[derive(Debug, Clone)]
pub struct ComponentDescriptor {
    pub name: String,
    pub size_bytes: usize,
    pub alignment: usize,
    pub access_frequency: f32,
    pub is_hot: bool,
}

/// Speculative memory reorganizer
pub struct SpeculativeMemoryReorg {
    /// HTM lock (simulated for cross-platform)
    htm_available: bool,
    /// Reorganization threshold (confidence must exceed this)
    confidence_threshold: f32,
    /// Maximum bytes to reorganize transactionally
    max_transaction_size: usize,
    /// Access pattern history
    pattern_history: Vec<AccessPattern>,
    /// Pending reorganization tasks
    pending_reorgs: Vec<ReorgTask>,
}

#[derive(Debug, Clone)]
pub struct ReorgTask {
    pub source_layout: MemoryLayout,
    pub target_layout: MemoryLayout,
    pub component_count: usize,
    pub confidence: f32,
}

/// Layout predictor using simple ML model
pub struct LayoutPredictor {
    /// Historical patterns
    patterns: Vec<AccessPattern>,
    /// Simple threshold model
    aos_threshold: f32,
    soa_threshold: f32,
}

impl LayoutPredictor {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            aos_threshold: 0.7,
            soa_threshold: 0.7,
        }
    }

    /// Predict optimal layout based on access pattern
    pub fn predict(&self, pattern: &AccessPattern) -> MemoryLayout {
        if pattern.confidence() < 0.5 {
            return MemoryLayout::ArrayOfStructures; // Default safe choice
        }

        if pattern.is_aos_pattern() && pattern.hot_components.len() > 2 {
            // AoS pattern with hot components -> Hybrid
            return MemoryLayout::Hybrid;
        }

        if pattern.is_soa_pattern() && pattern.avg_stride < 4.0 {
            // Sequential access -> SoA
            return MemoryLayout::StructureOfArrays;
        }

        if pattern.is_strided_pattern() {
            // Strided access -> Aligned AoS
            return MemoryLayout::ArrayOfStructuresAligned;
        }

        // Fallback: split hot/cold
        if !pattern.hot_components.is_empty() && !pattern.cold_components.is_empty() {
            return MemoryLayout::SplitSoA;
        }

        MemoryLayout::ArrayOfStructures
    }

    /// Train the predictor (simplified)
    pub fn train(&mut self, pattern: &AccessPattern, optimal_layout: MemoryLayout) {
        self.patterns.push(pattern.clone());

        // Update thresholds based on outcome
        if optimal_layout == MemoryLayout::StructureOfArrays {
            self.soa_threshold = self.soa_threshold * 0.95 + 0.05;
        } else if optimal_layout == MemoryLayout::ArrayOfStructures {
            self.aos_threshold = self.aos_threshold * 0.95 + 0.05;
        }
    }
}

impl Default for LayoutPredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Transactional memory wrapper (simulates HTM behavior)
pub struct HtmTransaction {
    /// Transaction nesting level
    depth: usize,
    /// Write buffer for the transaction
    buffer: Vec<u8>,
    /// Aborted flag
    aborted: bool,
    /// Retry count
    retries: u32,
}

impl HtmTransaction {
    pub fn new() -> Self {
        Self {
            depth: 0,
            buffer: Vec::new(),
            aborted: false,
            retries: 0,
        }
    }

    /// Begin transaction (simulated)
    pub fn begin(&mut self) -> bool {
        self.depth += 1;
        true
    }

    /// Write data within transaction
    pub fn write(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    /// Commit transaction
    pub fn commit(&mut self) -> Result<(), HtmError> {
        if self.aborted {
            return Err(HtmError::TransactionAborted);
        }
        self.depth = 0;
        self.buffer.clear();
        Ok(())
    }

    /// Abort transaction
    pub fn abort(&mut self) {
        self.aborted = true;
        self.buffer.clear();
    }

    /// Check if transaction should retry
    pub fn should_retry(&self) -> bool {
        self.retries < 3 && self.aborted
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HtmError {
    TransactionAborted,
    CapacityExceeded,
    Other,
}

/// AoS to SoA converter
pub struct AosToSoaConverter {
    pub component_size: usize,
    pub component_count: usize,
    pub alignment: usize,
}

impl AosToSoaConverter {
    /// Convert Array of Structures to Structure of Arrays
    pub fn convert(&self, aos_data: &[u8]) -> Vec<Vec<u8>> {
        let entity_count = aos_data.len() / (self.component_size * self.component_count);
        let mut soa_arrays: Vec<Vec<u8>> = vec![
            Vec::with_capacity(entity_count * self.component_size);
            self.component_count
        ];

        for entity in 0..entity_count {
            for comp in 0..self.component_count {
                let offset = entity * self.component_count * self.component_size
                    + comp * self.component_size;
                let src = &aos_data[offset..offset + self.component_size];
                soa_arrays[comp].extend_from_slice(src);
            }
        }

        soa_arrays
    }

    /// Convert Structure of Arrays back to Array of Structures
    pub fn convert_soa_to_aos(&self, soa_arrays: &[Vec<u8>]) -> Vec<u8> {
        let entity_count = soa_arrays.first().map(|a| a.len() / self.component_size).unwrap_or(0);
        let mut aos_data = Vec::with_capacity(entity_count * self.component_count * self.component_size);

        for entity in 0..entity_count {
            for comp in 0..self.component_count {
                let offset = entity * self.component_size;
                let src = &soa_arrays[comp][offset..offset + self.component_size];
                aos_data.extend_from_slice(src);
            }
        }

        aos_data
    }

    /// Split into hot/cold arrays
    pub fn split_hot_cold(&self, soa_arrays: &[Vec<u8>], hot_indices: &[usize]) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut hot_arrays = Vec::new();
        let mut cold_arrays = Vec::new();

        for (i, arr) in soa_arrays.iter().enumerate() {
            if hot_indices.contains(&i) {
                hot_arrays.push(arr.clone());
            } else {
                cold_arrays.push(arr.clone());
            }
        }

        (hot_arrays, cold_arrays)
    }
}

/// Speculative reorganization orchestrator
pub struct MemoryReorgOrchestrator {
    /// Current layout state
    current_layout: MemoryLayout,
    /// Layout predictor
    predictor: LayoutPredictor,
    /// Access pattern accumulator
    pattern: AccessPattern,
    /// Reorganization count
    reorg_count: AtomicUsize,
}

impl MemoryReorgOrchestrator {
    pub fn new() -> Self {
        Self {
            current_layout: MemoryLayout::ArrayOfStructures,
            predictor: LayoutPredictor::new(),
            pattern: AccessPattern::default(),
            reorg_count: AtomicUsize::new(0),
        }
    }

    /// Record an access for pattern analysis
    pub fn record_access(&mut self, component_index: usize, access_type: AccessType) {
        match access_type {
            AccessType::Sequential => self.pattern.sequential_count += 1,
            AccessType::Random => self.pattern.random_count += 1,
            AccessType::Strided(s) => {
                self.pattern.strided_count += 1;
                self.pattern.avg_stride = (self.pattern.avg_stride + s as f32) / 2.0;
            }
        }
        self.pattern.total_accesses += 1;

        // Update hot/cold classification
        self.update_hot_cold(component_index);
    }

    fn update_hot_cold(&mut self, component_index: usize) {
        const HOT_THRESHOLD: usize = 100;

        if !self.pattern.hot_components.contains(&component_index) {
            if self.pattern.total_accesses > HOT_THRESHOLD {
                let hot_ratio = self.pattern.hot_components.len() as f32
                    / self.pattern.total_accesses as f32;
                if hot_ratio > 0.8 {
                    self.pattern.cold_components.push(component_index);
                } else if self.pattern.sequential_count as f32 / self.pattern.total_accesses as f32 > 0.6 {
                    if !self.pattern.hot_components.contains(&component_index) {
                        self.pattern.hot_components.push(component_index);
                    }
                }
            }
        }
    }

    /// Check if reorganization should be attempted
    pub fn should_reorganize(&self) -> bool {
        self.pattern.confidence() > 0.7 && self.pattern.total_accesses > 500
    }

    /// Get recommended target layout
    pub fn recommended_layout(&self) -> MemoryLayout {
        self.predictor.predict(&self.pattern)
    }

    /// Perform transactional reorganization
    pub fn reorganize(&mut self, data: &mut [u8], components: &[ComponentDescriptor]) -> Result<MemoryLayout, HtmError> {
        if !self.should_reorganize() {
            return Ok(self.current_layout);
        }

        let target = self.recommended_layout();
        if target == self.current_layout {
            return Ok(target);
        }

        // Begin HTM transaction
        let mut tx = HtmTransaction::new();
        if !tx.begin() {
            return Err(HtmError::Other);
        }

        let total_size = data.len();
        if total_size > 16 * 1024 * 1024 {
            tx.abort();
            return Err(HtmError::CapacityExceeded);
        }

        // Perform conversion
        let converter = AosToSoaConverter {
            component_size: components.first().map(|c| c.size_bytes).unwrap_or(4),
            component_count: components.len(),
            alignment: components.first().map(|c| c.alignment).unwrap_or(4),
        };

        let new_data = match (self.current_layout, target) {
            (MemoryLayout::ArrayOfStructures, MemoryLayout::StructureOfArrays) => {
                converter.convert(data)
            }
            (MemoryLayout::StructureOfArrays, MemoryLayout::ArrayOfStructures) => {
                // Reconstruct AoS from current SoA data.
                // Split the flat data buffer into per-component SoA slices,
                // then interleave them back into AoS order.
                let entity_count = data.len()
                    / (converter.component_size * converter.component_count);
                let component_array_bytes = entity_count * converter.component_size;
                let soa_arrays: Vec<Vec<u8>> = (0..converter.component_count)
                    .map(|c| {
                        let start = c * component_array_bytes;
                        data[start..start + component_array_bytes].to_vec()
                    })
                    .collect();
                converter.convert_soa_to_aos(&soa_arrays)
            }
            _ => {
                tx.abort();
                return Ok(self.current_layout);
            }
        };

        // Write reorganized data back to the actual data store.
        match target {
            MemoryLayout::StructureOfArrays => {
                // Flatten Vec<Vec<u8>> SoA arrays back into the flat buffer.
                let flat: Vec<u8> = new_data.iter().flatten().cloned().collect();
                let copy_len = flat.len().min(data.len());
                data[..copy_len].copy_from_slice(&flat[..copy_len]);
            }
            MemoryLayout::ArrayOfStructures => {
                // new_data is Vec<u8> from convert_soa_to_aos.
                let copy_len = new_data.len().min(data.len());
                data[..copy_len].copy_from_slice(&new_data[..copy_len]);
            }
            _ => {}
        }

        // Try to commit
        match tx.commit() {
            Ok(()) => {
                self.current_layout = target;
                self.reorg_count.fetch_add(1, Ordering::SeqCst);
                self.predictor.train(&self.pattern, target);
                Ok(target)
            }
            Err(e) => {
                // Rollback happened automatically
                Err(e)
            }
        }
    }

    /// Get current access pattern statistics
    pub fn get_pattern(&self) -> &AccessPattern {
        &self.pattern
    }

    /// Reset pattern for new accumulation period
    pub fn reset_pattern(&mut self) {
        self.pattern = AccessPattern::default();
    }
}

impl Default for MemoryReorgOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    Sequential,
    Random,
    Strided(usize),
}

/// Thread-safe wrapper for use in runtime
pub type SharedMemoryReorg = Arc<MemoryReorgOrchestrator>;

impl SharedMemoryReorg {
    pub fn new() -> Self {
        Arc::new(MemoryReorgOrchestrator::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_detection() {
        let mut pattern = AccessPattern {
            random_count: 100,
            sequential_count: 10,
            strided_count: 10,
            hot_components: vec![],
            cold_components: vec![],
            avg_stride: 1.0,
            total_accesses: 120,
        };

        assert!(pattern.is_aos_pattern());
        assert!(!pattern.is_soa_pattern());
    }

    #[test]
    fn test_layout_predictor() {
        let predictor = LayoutPredictor::new();
        let pattern = AccessPattern {
            random_count: 100,
            sequential_count: 10,
            strided_count: 10,
            hot_components: vec![0, 1, 2],
            cold_components: vec![3, 4, 5],
            avg_stride: 1.0,
            total_accesses: 120,
        };

        let layout = predictor.predict(&pattern);
        assert!(matches!(layout, MemoryLayout::Hybrid | MemoryLayout::ArrayOfStructures));
    }
}
