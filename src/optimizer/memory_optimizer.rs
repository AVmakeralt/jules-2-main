// =============================================================================
// Auto-Apply NUMA/Huge Pages in Compiler
//
// This module automatically applies NUMA-aware allocation and huge pages
// based on data access patterns detected during compilation.
//
// Features:
// - NUMA topology detection and analysis
// - Hot data identification for huge pages
// - Automatic NUMA node assignment
// - Cache-line alignment optimization
// =============================================================================

use std::collections::HashMap;

/// Memory allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Standard allocation (no optimization)
    Standard,
    /// NUMA-aware allocation (allocate on local node)
    NumaLocal,
    /// NUMA-interleaved (spread across nodes)
    NumaInterleaved,
    /// Huge pages (2MB or 1GB)
    HugePages,
    /// NUMA + Huge pages combined
    NumaHuge,
}

/// Memory access pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Sequential access (good for prefetching)
    Sequential,
    /// Random access (needs NUMA awareness)
    Random,
    /// Strided access (may benefit from huge pages)
    Strided,
    /// Unknown pattern
    Unknown,
}

/// Data region metadata
#[derive(Debug, Clone)]
pub struct DataRegion {
    /// Region name/identifier
    pub name: String,
    /// Size in bytes
    pub size: usize,
    /// Access pattern
    pub pattern: AccessPattern,
    /// Access frequency (accesses per second)
    pub frequency: f64,
    /// Whether this is hot data (frequently accessed)
    pub is_hot: bool,
    /// Preferred NUMA node (if known)
    pub preferred_numa_node: Option<usize>,
    /// Current allocation strategy
    pub strategy: AllocationStrategy,
}

impl DataRegion {
    /// Create a new data region
    pub fn new(name: String, size: usize) -> Self {
        Self {
            name,
            size,
            pattern: AccessPattern::Unknown,
            frequency: 0.0,
            is_hot: false,
            preferred_numa_node: None,
            strategy: AllocationStrategy::Standard,
        }
    }

    /// Update access pattern
    pub fn set_pattern(&mut self, pattern: AccessPattern) {
        self.pattern = pattern;
    }

    /// Update access frequency
    pub fn set_frequency(&mut self, frequency: f64) {
        self.frequency = frequency;
        self.is_hot = frequency > 1000.0; // Hot if >1000 accesses/sec
    }

    /// Set preferred NUMA node
    pub fn set_numa_node(&mut self, node: usize) {
        self.preferred_numa_node = Some(node);
    }

    /// Determine optimal allocation strategy
    pub fn determine_strategy(&mut self, num_numa_nodes: usize) {
        self.strategy = if self.is_hot && self.size >= 2 * 1024 * 1024 {
            // Hot and large: use huge pages
            if let Some(_node) = self.preferred_numa_node {
                AllocationStrategy::NumaHuge
            } else {
                AllocationStrategy::HugePages
            }
        } else if self.pattern == AccessPattern::Random && num_numa_nodes > 1 {
            // Random access on multi-NUMA: use local allocation
            if let Some(_node) = self.preferred_numa_node {
                AllocationStrategy::NumaLocal
            } else {
                AllocationStrategy::Standard
            }
        } else if self.size >= 1024 * 1024 && num_numa_nodes > 1 {
            // Large data on multi-NUMA: interleave
            AllocationStrategy::NumaInterleaved
        } else {
            AllocationStrategy::Standard
        };
    }
}

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub num_nodes: usize,
    /// CPUs per node
    pub cpus_per_node: Vec<usize>,
    /// Memory per node (in bytes)
    pub memory_per_node: Vec<usize>,
    /// Distance matrix (node i to node j)
    pub distance_matrix: Vec<Vec<u32>>,
}

impl NumaTopology {
    /// Detect the NUMA topology from the OS
    pub fn detect() -> Self {
        // Try Linux sysfs first
        if let Some(topo) = Self::detect_linux_sysfs() {
            return topo;
        }
        // Fallback: reasonable heuristic
        Self::detect_fallback()
    }

    /// Detect from Linux /sys/devices/system/node/
    fn detect_linux_sysfs() -> Option<Self> {
        // Read /sys/devices/system/node/online to get node count
        let online = std::fs::read_to_string("/sys/devices/system/node/online")
            .ok()?;

        // Parse range like "0-3" or "0,2-4"
        let max_node = Self::parse_range_max(&online)?;
        let num_nodes = max_node + 1;

        let mut cpus_per_node = Vec::with_capacity(num_nodes);
        let mut memory_per_node = Vec::with_capacity(num_nodes);

        for i in 0..num_nodes {
            // Read /sys/devices/system/node/node{i}/cpulist
            let cpulist = std::fs::read_to_string(
                format!("/sys/devices/system/node/node{}/cpulist", i)
            ).unwrap_or_default();
            let cpu_count = Self::parse_cpulist_count(&cpulist).max(1);
            cpus_per_node.push(cpu_count);

            // Read /sys/devices/system/node/node{i}/meminfo
            // Look for "MemTotal:   XXXX kB"
            let meminfo = std::fs::read_to_string(
                format!("/sys/devices/system/node/node{}/meminfo", i)
            ).unwrap_or_default();
            let mem_bytes = Self::parse_meminfo_kb(&meminfo)
                .unwrap_or(16 * 1024 * 1024 * 1024);
            memory_per_node.push(mem_bytes);
        }

        let distance_matrix = Self::read_distance_matrix(num_nodes);

        Some(Self {
            num_nodes,
            cpus_per_node,
            memory_per_node,
            distance_matrix,
        })
    }

    /// Parse a range string like "0-3" or "0,2-4" and return the maximum value
    fn parse_range_max(s: &str) -> Option<usize> {
        let s = s.trim();
        let mut max_val = 0usize;
        for part in s.split(',') {
            let part = part.trim();
            if let Some(dash) = part.find('-') {
                let high: usize = part[dash+1..].parse().ok()?;
                max_val = max_val.max(high);
            } else {
                let val: usize = part.parse().ok()?;
                max_val = max_val.max(val);
            }
        }
        Some(max_val)
    }

    /// Parse a cpulist like "0-7,16-23" and count CPUs
    fn parse_cpulist_count(s: &str) -> usize {
        let s = s.trim();
        let mut count = 0usize;
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() { continue; }
            if let Some(dash) = part.find('-') {
                let lo: usize = part[..dash].parse().unwrap_or(0);
                let hi: usize = part[dash+1..].parse().unwrap_or(lo);
                count += hi.saturating_sub(lo) + 1;
            } else {
                count += 1;
            }
        }
        count
    }

    /// Parse meminfo to get total memory in bytes
    fn parse_meminfo_kb(s: &str) -> Option<usize> {
        for line in s.lines() {
            if line.contains("MemTotal") {
                // Format: "Node 0 MemTotal:   16384000 kB"
                let kb: usize = line.split_whitespace()
                    .filter_map(|w| w.parse::<usize>().ok())
                    .next()?;
                return Some(kb * 1024);
            }
        }
        None
    }

    /// Read NUMA distance matrix from sysfs
    fn read_distance_matrix(num_nodes: usize) -> Vec<Vec<u32>> {
        let mut matrix = Self::default_distance_matrix(num_nodes);
        for i in 0..num_nodes {
            if let Ok(data) = std::fs::read_to_string(
                format!("/sys/devices/system/node/node{}/distance", i)
            ) {
                let distances: Vec<u32> = data.split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                for (j, &d) in distances.iter().enumerate() {
                    if j < num_nodes {
                        matrix[i][j] = d;
                    }
                }
            }
        }
        matrix
    }

    /// Fallback when sysfs is not available
    fn detect_fallback() -> Self {
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // Heuristic: assume 2 NUMA nodes for >= 16 CPUs, 1 otherwise
        let num_nodes = if num_cpus >= 16 { 2 } else { 1 };
        let cpus_per_node = if num_nodes > 1 {
            vec![num_cpus / 2, num_cpus - num_cpus / 2]
        } else {
            vec![num_cpus]
        };
        let memory_per_node = vec![16 * 1024 * 1024 * 1024; num_nodes];
        let distance_matrix = Self::default_distance_matrix(num_nodes);

        Self {
            num_nodes,
            cpus_per_node,
            memory_per_node,
            distance_matrix,
        }
    }

    /// Create a default distance matrix (all nodes are local)
    fn default_distance_matrix(num_nodes: usize) -> Vec<Vec<u32>> {
        (0..num_nodes)
            .map(|i| {
                (0..num_nodes)
                    .map(|j| if i == j { 10 } else { 20 })
                    .collect()
            })
            .collect()
    }

    /// Get the distance between two nodes
    pub fn distance(&self, from: usize, to: usize) -> u32 {
        if from < self.num_nodes && to < self.num_nodes {
            self.distance_matrix[from][to]
        } else {
            20 // Default remote distance
        }
    }

    /// Find the closest node to a given CPU
    pub fn closest_node_to_cpu(&self, cpu_id: usize) -> usize {
        if self.num_nodes == 1 {
            return 0;
        }
        cpu_id % self.num_nodes
    }
}

/// Memory optimizer
pub struct MemoryOptimizer {
    /// NUMA topology
    numa_topology: NumaTopology,
    /// Data regions
    regions: HashMap<String, DataRegion>,
    /// Huge page size (in bytes)
    huge_page_size: usize,
    /// Whether huge pages are available
    #[allow(dead_code)]
    huge_pages_available: bool,
}

impl MemoryOptimizer {
    pub fn new() -> Self {
        let numa_topology = NumaTopology::detect();
        let huge_pages_available = Self::detect_huge_pages();

        Self {
            numa_topology,
            regions: HashMap::new(),
            huge_page_size: 2 * 1024 * 1024, // 2MB default
            huge_pages_available,
        }
    }

    /// Detect if huge pages are available
    fn detect_huge_pages() -> bool {
        #[cfg(target_os = "linux")]
        {
            // Check /proc/meminfo for HugePages_Total
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                return meminfo.contains("HugePages_Total:");
            }
        }
        false
    }

    /// Register a data region
    pub fn register_region(&mut self, region: DataRegion) {
        let name = region.name.clone();
        self.regions.insert(name, region);
    }

    /// Analyze and optimize all regions
    pub fn optimize_all(&mut self) {
        for region in self.regions.values_mut() {
            region.determine_strategy(self.numa_topology.num_nodes);
        }
    }

    /// Get the allocation strategy for a region
    pub fn get_strategy(&self, name: &str) -> Option<AllocationStrategy> {
        self.regions.get(name).map(|r| r.strategy)
    }

    /// Get optimization suggestions
    pub fn get_suggestions(&self) -> Vec<OptimizationSuggestion> {
        self.regions
            .values()
            .filter_map(|region| {
                if region.strategy == AllocationStrategy::Standard {
                    None
                } else {
                    Some(OptimizationSuggestion {
                        region: region.name.clone(),
                        strategy: region.strategy,
                        reason: self.suggestion_reason(region),
                        expected_benefit: self.estimate_benefit(region),
                    })
                }
            })
            .collect()
    }

    /// Generate a reason for the optimization
    fn suggestion_reason(&self, region: &DataRegion) -> String {
        match region.strategy {
            AllocationStrategy::NumaLocal => {
                format!(
                    "Frequent random access pattern on NUMA system. Allocate on node {}.",
                    region.preferred_numa_node.unwrap_or(0)
                )
            }
            AllocationStrategy::NumaInterleaved => {
                format!(
                    "Large data region ({} MB) on NUMA system. Interleave across {} nodes.",
                    region.size / (1024 * 1024),
                    self.numa_topology.num_nodes
                )
            }
            AllocationStrategy::HugePages => {
                format!(
                    "Hot data region ({} MB). Use {} huge pages to reduce TLB misses.",
                    region.size / (1024 * 1024),
                    self.huge_page_size / (1024 * 1024)
                )
            }
            AllocationStrategy::NumaHuge => {
                format!(
                    "Hot large data on NUMA system. Use huge pages on local node {}.",
                    region.preferred_numa_node.unwrap_or(0)
                )
            }
            AllocationStrategy::Standard => {
                "No optimization needed".to_string()
            }
        }
    }

    /// Estimate the benefit of an optimization
    fn estimate_benefit(&self, region: &DataRegion) -> f64 {
        match region.strategy {
            AllocationStrategy::NumaLocal => {
                // NUMA local access is ~2x faster than remote
                2.0
            }
            AllocationStrategy::NumaInterleaved => {
                // Interleaving provides ~1.3x bandwidth improvement
                1.3
            }
            AllocationStrategy::HugePages => {
                // Huge pages reduce TLB misses by ~10x for large datasets
                let tlb_reduction = if region.size > 100 * 1024 * 1024 { 10.0 } else { 3.0 };
                tlb_reduction
            }
            AllocationStrategy::NumaHuge => {
                // Combined benefit
                2.0 * 3.0
            }
            AllocationStrategy::Standard => 1.0,
        }
    }

    /// Get the NUMA topology
    pub fn numa_topology(&self) -> &NumaTopology {
        &self.numa_topology
    }

    /// Get statistics
    pub fn stats(&self) -> MemoryOptimizerStats {
        let total = self.regions.len();
        let numa_local = self.regions.values().filter(|r| r.strategy == AllocationStrategy::NumaLocal).count();
        let numa_interleaved = self.regions.values().filter(|r| r.strategy == AllocationStrategy::NumaInterleaved).count();
        let huge_pages = self.regions.values().filter(|r| r.strategy == AllocationStrategy::HugePages).count();
        let numa_huge = self.regions.values().filter(|r| r.strategy == AllocationStrategy::NumaHuge).count();
        let standard = self.regions.values().filter(|r| r.strategy == AllocationStrategy::Standard).count();

        MemoryOptimizerStats {
            total_regions: total,
            numa_local,
            numa_interleaved,
            huge_pages,
            numa_huge,
            standard,
        }
    }
}

/// Optimization suggestion
#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    /// Region name
    pub region: String,
    /// Suggested strategy
    pub strategy: AllocationStrategy,
    /// Reason for the suggestion
    pub reason: String,
    /// Expected speedup multiplier
    pub expected_benefit: f64,
}

/// Memory optimizer statistics
#[derive(Debug)]
pub struct MemoryOptimizerStats {
    pub total_regions: usize,
    pub numa_local: usize,
    pub numa_interleaved: usize,
    pub huge_pages: usize,
    pub numa_huge: usize,
    pub standard: usize,
}

/// Cache-line alignment optimizer
pub struct AlignmentOptimizer {
    /// Cache line size (in bytes)
    cache_line_size: usize,
}

impl AlignmentOptimizer {
    pub fn new() -> Self {
        Self {
            cache_line_size: 64, // Standard x86-64 cache line
        }
    }

    /// Calculate aligned size
    pub fn aligned_size(&self, size: usize) -> usize {
        ((size + self.cache_line_size - 1) / self.cache_line_size) * self.cache_line_size
    }

    /// Check if an address is aligned
    pub fn is_aligned(&self, address: usize) -> bool {
        address % self.cache_line_size == 0
    }

    /// Get alignment padding needed
    pub fn padding_needed(&self, address: usize) -> usize {
        (self.cache_line_size - (address % self.cache_line_size)) % self.cache_line_size
    }

    /// Suggest alignment for a data structure
    pub fn suggest_alignment(&self, size: usize, access_pattern: AccessPattern) -> AlignmentSuggestion {
        let alignment = if access_pattern == AccessPattern::Sequential && size >= 1024 {
            // Large sequential data: align to cache line
            self.cache_line_size
        } else if access_pattern == AccessPattern::Random && size >= 64 {
            // Random access: align to avoid false sharing
            self.cache_line_size
        } else {
            // Small data: natural alignment
            size.next_power_of_two().min(64)
        };

        AlignmentSuggestion {
            alignment,
            reason: self.alignment_reason(size, access_pattern, alignment),
        }
    }

    fn alignment_reason(&self, _size: usize, pattern: AccessPattern, alignment: usize) -> String {
        if alignment == self.cache_line_size {
            format!(
                "Cache-line alignment ({} bytes) prevents false sharing and improves prefetching for {} access",
                self.cache_line_size,
                match pattern {
                    AccessPattern::Sequential => "sequential",
                    AccessPattern::Random => "random",
                    AccessPattern::Strided => "strided",
                    AccessPattern::Unknown => "unknown",
                }
            )
        } else {
            format!("Natural alignment ({} bytes) for efficient access", alignment)
        }
    }
}

/// Alignment suggestion
#[derive(Debug, Clone)]
pub struct AlignmentSuggestion {
    /// Suggested alignment in bytes
    pub alignment: usize,
    /// Reason for the suggestion
    pub reason: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_region() {
        let mut region = DataRegion::new("test".to_string(), 1024);
        region.set_pattern(AccessPattern::Sequential);
        region.set_frequency(2000.0);
        
        assert!(region.is_hot);
        assert_eq!(region.pattern, AccessPattern::Sequential);
    }

    #[test]
    fn test_numa_topology() {
        let topology = NumaTopology::detect();
        assert!(topology.num_nodes > 0);
        
        let distance = topology.distance(0, 0);
        assert_eq!(distance, 10);
    }

    #[test]
    fn test_memory_optimizer() {
        let mut optimizer = MemoryOptimizer::new();
        
        let mut region = DataRegion::new("hot_data".to_string(), 4 * 1024 * 1024);
        region.set_pattern(AccessPattern::Random);
        region.set_frequency(5000.0);
        region.set_numa_node(0);
        
        optimizer.register_region(region);
        optimizer.optimize_all();
        
        let strategy = optimizer.get_strategy("hot_data");
        assert!(strategy.is_some());
    }

    #[test]
    fn test_alignment_optimizer() {
        let optimizer = AlignmentOptimizer::new();
        
        assert_eq!(optimizer.aligned_size(100), 128);
        assert!(optimizer.is_aligned(64));
        assert!(!optimizer.is_aligned(63));
        
        let suggestion = optimizer.suggest_alignment(1024, AccessPattern::Sequential);
        assert_eq!(suggestion.alignment, 64);
    }
}
