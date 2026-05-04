// =============================================================================
// Auto-SoA Detection and Hot-Swap
//
// This module automatically detects when Structure of Arrays (SoA) layout
// would be beneficial and hot-swaps from Array of Structures (AoS) to SoA
// at runtime based on access patterns.
//
// Features:
// - Access pattern analysis (field-wise vs structure-wise)
// - AoS ↔ SoA hot-swapping
// - Cache-line utilization optimization
// - Automatic chunking for large arrays
// =============================================================================

use std::collections::HashMap;

/// Memory layout strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutStrategy {
    /// Array of Structures (AoS) - default
    AoS,
    /// Structure of Arrays (SoA) - field-separated
    SoA,
    /// Hybrid - chunked SoA
    ChunkedSoA,
}

/// Field access pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldAccessPattern {
    /// Field accessed independently (good for SoA)
    FieldWise,
    /// All fields accessed together (good for AoS)
    StructureWise,
    /// Mixed pattern
    Mixed,
}

/// Structure metadata
#[derive(Debug, Clone)]
pub struct StructureMetadata {
    /// Structure name
    pub name: String,
    /// Field names and types
    pub fields: Vec<(String, String)>,
    /// Field sizes in bytes
    pub field_sizes: Vec<usize>,
    /// Total structure size
    pub total_size: usize,
}

impl StructureMetadata {
    pub fn new(name: String, fields: Vec<(String, String)>, field_sizes: Vec<usize>) -> Self {
        let total_size = field_sizes.iter().sum();
        Self {
            name,
            fields,
            field_sizes,
            total_size,
        }
    }

    /// Get the index of a field by name
    pub fn field_index(&self, field_name: &str) -> Option<usize> {
        self.fields.iter().position(|(name, _)| name == field_name)
    }
}

/// Array access statistics
#[derive(Debug, Clone)]
pub struct AccessStats {
    /// Number of field-wise accesses
    pub field_wise_accesses: u64,
    /// Number of structure-wise accesses
    pub structure_wise_accesses: u64,
    /// Field access counts (field index -> count)
    pub field_access_counts: Vec<u64>,
    /// Total accesses
    pub total_accesses: u64,
}

impl AccessStats {
    pub fn new(num_fields: usize) -> Self {
        Self {
            field_wise_accesses: 0,
            structure_wise_accesses: 0,
            field_access_counts: vec![0; num_fields],
            total_accesses: 0,
        }
    }

    /// Record a field-wise access
    pub fn record_field_access(&mut self, field_index: usize) {
        self.field_wise_accesses += 1;
        if field_index < self.field_access_counts.len() {
            self.field_access_counts[field_index] += 1;
        }
        self.total_accesses += 1;
    }

    /// Record a structure-wise access
    pub fn record_structure_access(&mut self) {
        self.structure_wise_accesses += 1;
        self.total_accesses += 1;
    }

    /// Get the access pattern
    pub fn pattern(&self) -> FieldAccessPattern {
        if self.total_accesses == 0 {
            return FieldAccessPattern::Mixed;
        }

        let field_ratio = self.field_wise_accesses as f64 / self.total_accesses as f64;
        let struct_ratio = self.structure_wise_accesses as f64 / self.total_accesses as f64;

        if field_ratio > 0.7 {
            FieldAccessPattern::FieldWise
        } else if struct_ratio > 0.7 {
            FieldAccessPattern::StructureWise
        } else {
            FieldAccessPattern::Mixed
        }
    }

    /// Get the most accessed field
    pub fn most_accessed_field(&self) -> Option<usize> {
        self.field_access_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(index, _)| index)
    }
}

/// Array metadata
#[derive(Debug, Clone)]
pub struct ArrayMetadata {
    /// Array name
    pub name: String,
    /// Element structure metadata
    pub structure: StructureMetadata,
    /// Number of elements
    pub num_elements: usize,
    /// Current layout strategy
    pub layout: LayoutStrategy,
    /// Access statistics
    pub access_stats: AccessStats,
    /// Whether this array is hot (frequently accessed)
    pub is_hot: bool,
    /// Cache miss rate (if available)
    pub cache_miss_rate: f64,
}

impl ArrayMetadata {
    pub fn new(name: String, structure: StructureMetadata, num_elements: usize) -> Self {
        Self {
            name,
            structure: structure.clone(),
            num_elements,
            layout: LayoutStrategy::AoS,
            access_stats: AccessStats::new(structure.fields.len()),
            is_hot: false,
            cache_miss_rate: 0.0,
        }
    }

    /// Update hot status based on access frequency
    pub fn update_hot_status(&mut self, threshold: u64) {
        self.is_hot = self.access_stats.total_accesses >= threshold;
    }

    /// Determine optimal layout based on access pattern
    pub fn determine_optimal_layout(&self) -> LayoutStrategy {
        let pattern = self.access_stats.pattern();

        match pattern {
            FieldAccessPattern::FieldWise => {
                // Field-wise access: SoA is better
                if self.num_elements > 1000 {
                    LayoutStrategy::ChunkedSoA
                } else {
                    LayoutStrategy::SoA
                }
            }
            FieldAccessPattern::StructureWise => {
                // Structure-wise access: AoS is better
                LayoutStrategy::AoS
            }
            FieldAccessPattern::Mixed => {
                // Mixed: use current layout or SoA if large
                if self.num_elements > 10000 {
                    LayoutStrategy::ChunkedSoA
                } else {
                    self.layout
                }
            }
        }
    }

    /// Check if layout should be swapped
    pub fn should_swap_layout(&self) -> bool {
        let optimal = self.determine_optimal_layout();
        optimal != self.layout && self.is_hot
    }

    /// Estimate speedup from layout swap
    pub fn estimate_swap_speedup(&self) -> f64 {
        let optimal = self.determine_optimal_layout();
        
        if optimal == self.layout {
            1.0
        } else {
            match (self.layout, optimal) {
                (LayoutStrategy::AoS, LayoutStrategy::SoA) => {
                    // AoS → SoA: depends on field-wise access ratio
                    let field_ratio = self.access_stats.field_wise_accesses as f64 / self.access_stats.total_accesses as f64;
                    1.0 + field_ratio * 2.0 // Up to 3x speedup
                }
                (LayoutStrategy::SoA, LayoutStrategy::AoS) => {
                    // SoA → AoS: depends on structure-wise access ratio
                    let struct_ratio = self.access_stats.structure_wise_accesses as f64 / self.access_stats.total_accesses as f64;
                    1.0 + struct_ratio * 1.5 // Up to 2.5x speedup
                }
                (LayoutStrategy::AoS, LayoutStrategy::ChunkedSoA) => {
                    // AoS → ChunkedSoA: better for large arrays
                    1.5
                }
                (LayoutStrategy::ChunkedSoA, LayoutStrategy::AoS) => {
                    // ChunkedSoA → AoS: if structure-wise
                    1.3
                }
                _ => 1.0,
            }
        }
    }
}

/// SoA optimizer
pub struct SoaOptimizer {
    /// All arrays being tracked
    arrays: HashMap<String, ArrayMetadata>,
    /// Hot threshold (accesses before considered hot)
    hot_threshold: u64,
    /// Minimum array size for SoA consideration
    min_size_for_soa: usize,
    /// Whether auto-swap is enabled
    auto_swap_enabled: bool,
}

impl SoaOptimizer {
    pub fn new() -> Self {
        Self {
            arrays: HashMap::new(),
            hot_threshold: 1000,
            min_size_for_soa: 100,
            auto_swap_enabled: true,
        }
    }

    /// Register an array for monitoring
    pub fn register_array(&mut self, metadata: ArrayMetadata) {
        let name = metadata.name.clone();
        self.arrays.insert(name, metadata);
    }

    /// Record a field access
    pub fn record_field_access(&mut self, array_name: &str, field_name: &str) {
        if let Some(array) = self.arrays.get_mut(array_name) {
            if let Some(field_index) = array.structure.field_index(field_name) {
                array.access_stats.record_field_access(field_index);
                array.update_hot_status(self.hot_threshold);
            }
        }
    }

    /// Record a structure access
    pub fn record_structure_access(&mut self, array_name: &str) {
        if let Some(array) = self.arrays.get_mut(array_name) {
            array.access_stats.record_structure_access();
            array.update_hot_status(self.hot_threshold);
        }
    }

    /// Analyze all arrays and perform layout swaps if beneficial
    pub fn analyze_and_swap(&mut self) -> Vec<LayoutSwap> {
        let mut swaps = Vec::new();

        for (name, array) in self.arrays.iter_mut() {
            if array.should_swap_layout() && self.auto_swap_enabled {
                let old_layout = array.layout;
                let new_layout = array.determine_optimal_layout();
                let speedup = array.estimate_swap_speedup();

                array.layout = new_layout;
                swaps.push(LayoutSwap {
                    array_name: name.clone(),
                    old_layout,
                    new_layout,
                    speedup,
                });
            }
        }

        swaps
    }

    /// Get the current layout of an array
    pub fn get_layout(&self, array_name: &str) -> Option<LayoutStrategy> {
        self.arrays.get(array_name).map(|a| a.layout)
    }

    /// Get optimization suggestions
    pub fn get_suggestions(&self) -> Vec<SoaSuggestion> {
        self.arrays
            .values()
            .filter_map(|array| {
                if array.num_elements < self.min_size_for_soa {
                    return None;
                }

                let optimal = array.determine_optimal_layout();
                if optimal == array.layout {
                    return None;
                }

                let speedup = array.estimate_swap_speedup();
                if speedup < 1.2 {
                    return None; // Not worth it
                }

                Some(SoaSuggestion {
                    array_name: array.name.clone(),
                    current_layout: array.layout,
                    suggested_layout: optimal,
                    reason: self.suggestion_reason(array, optimal),
                    expected_speedup: speedup,
                })
            })
            .collect()
    }

    fn suggestion_reason(&self, array: &ArrayMetadata, optimal: LayoutStrategy) -> String {
        let pattern = array.access_stats.pattern();
        
        match (pattern, optimal) {
            (FieldAccessPattern::FieldWise, LayoutStrategy::SoA) => {
                format!(
                    "Field-wise access pattern ({}% field accesses). SoA layout improves cache locality.",
                    (array.access_stats.field_wise_accesses as f64 / array.access_stats.total_accesses as f64 * 100.0) as u32
                )
            }
            (FieldAccessPattern::FieldWise, LayoutStrategy::ChunkedSoA) => {
                format!(
                    "Large array ({}) with field-wise access. Chunked SoA balances cache locality and memory overhead.",
                    array.num_elements
                )
            }
            (FieldAccessPattern::StructureWise, LayoutStrategy::AoS) => {
                "Structure-wise access pattern. AoS layout is more efficient.".to_string()
            }
            _ => {
                format!(
                    "Mixed access pattern. {} layout provides better balance.",
                    match optimal {
                        LayoutStrategy::AoS => "AoS",
                        LayoutStrategy::SoA => "SoA",
                        LayoutStrategy::ChunkedSoA => "ChunkedSoA",
                    }
                )
            }
        }
    }

    /// Enable or disable auto-swap
    pub fn set_auto_swap(&mut self, enabled: bool) {
        self.auto_swap_enabled = enabled;
    }

    /// Get statistics
    pub fn stats(&self) -> SoaOptimizerStats {
        let total = self.arrays.len();
        let aos = self.arrays.values().filter(|a| a.layout == LayoutStrategy::AoS).count();
        let soa = self.arrays.values().filter(|a| a.layout == LayoutStrategy::SoA).count();
        let chunked = self.arrays.values().filter(|a| a.layout == LayoutStrategy::ChunkedSoA).count();
        let hot = self.arrays.values().filter(|a| a.is_hot).count();

        SoaOptimizerStats {
            total_arrays: total,
            aos,
            soa,
            chunked_soa: chunked,
            hot,
        }
    }
}

/// Layout swap operation
#[derive(Debug, Clone)]
pub struct LayoutSwap {
    pub array_name: String,
    pub old_layout: LayoutStrategy,
    pub new_layout: LayoutStrategy,
    pub speedup: f64,
}

/// SoA optimization suggestion
#[derive(Debug, Clone)]
pub struct SoaSuggestion {
    pub array_name: String,
    pub current_layout: LayoutStrategy,
    pub suggested_layout: LayoutStrategy,
    pub reason: String,
    pub expected_speedup: f64,
}

/// SoA optimizer statistics
#[derive(Debug)]
pub struct SoaOptimizerStats {
    pub total_arrays: usize,
    pub aos: usize,
    pub soa: usize,
    pub chunked_soa: usize,
    pub hot: usize,
}

/// Cache-line utilization analyzer
pub struct CacheLineAnalyzer {
    /// Cache line size in bytes
    cache_line_size: usize,
}

impl CacheLineAnalyzer {
    pub fn new() -> Self {
        Self {
            cache_line_size: 64,
        }
    }

    /// Calculate cache-line utilization for AoS layout
    pub fn aos_utilization(&self, structure: &StructureMetadata, num_elements: usize) -> f64 {
        let bytes_per_element = structure.total_size;
        let elements_per_line = self.cache_line_size / bytes_per_element;
        
        if elements_per_line == 0 {
            1.0 // One element spans multiple lines
        } else {
            let used_bytes = (num_elements * bytes_per_element) % self.cache_line_size;
            used_bytes as f64 / self.cache_line_size as f64
        }
    }

    /// Calculate cache-line utilization for SoA layout
    pub fn soa_utilization(&self, field_size: usize, num_elements: usize) -> f64 {
        let elements_per_line = self.cache_line_size / field_size;
        
        if elements_per_line == 0 {
            1.0
        } else {
            let used_bytes = (num_elements * field_size) % self.cache_line_size;
            used_bytes as f64 / self.cache_line_size as f64
        }
    }

    /// Estimate cache miss reduction from SoA
    pub fn estimate_miss_reduction(&self, structure: &StructureMetadata, num_elements: usize, field_index: usize) -> f64 {
        let aos_util = self.aos_utilization(structure, num_elements);
        let soa_util = self.soa_utilization(structure.field_sizes[field_index], num_elements);
        
        if aos_util > 0.0 {
            soa_util / aos_util
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structure_metadata() {
        let structure = StructureMetadata::new(
            "Point".to_string(),
            vec![("x".to_string(), "f64".to_string()), ("y".to_string(), "f64".to_string())],
            vec![8, 8],
        );
        
        assert_eq!(structure.total_size, 16);
        assert_eq!(structure.field_index("x"), Some(0));
    }

    #[test]
    fn test_access_stats() {
        let mut stats = AccessStats::new(2);
        stats.record_field_access(0);
        stats.record_field_access(1);
        stats.record_structure_access();
        
        assert_eq!(stats.total_accesses, 3);
        assert_eq!(stats.field_wise_accesses, 2);
        assert_eq!(stats.structure_wise_accesses, 1);
    }

    #[test]
    fn test_array_metadata() {
        let structure = StructureMetadata::new(
            "Point".to_string(),
            vec![("x".to_string(), "f64".to_string())],
            vec![8],
        );
        let mut array = ArrayMetadata::new("points".to_string(), structure, 1000);
        
        array.access_stats.record_field_access(0);
        array.access_stats.record_field_access(0);
        array.update_hot_status(1);
        
        assert!(array.is_hot);
    }

    #[test]
    fn test_soa_optimizer() {
        let mut optimizer = SoaOptimizer::new();
        
        let structure = StructureMetadata::new(
            "Point".to_string(),
            vec![("x".to_string(), "f64".to_string())],
            vec![8],
        );
        let array = ArrayMetadata::new("points".to_string(), structure, 1000);
        
        optimizer.register_array(array);
        optimizer.record_field_access("points", "x");
        
        let layout = optimizer.get_layout("points");
        assert!(layout.is_some());
    }

    #[test]
    fn test_cache_line_analyzer() {
        let analyzer = CacheLineAnalyzer::new();
        
        let structure = StructureMetadata::new(
            "Point".to_string(),
            vec![("x".to_string(), "f64".to_string())],
            vec![8],
        );
        
        let util = analyzer.aos_utilization(&structure, 100);
        assert!(util > 0.0 && util <= 1.0);
    }
}
