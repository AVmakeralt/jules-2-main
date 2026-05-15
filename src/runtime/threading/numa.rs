// =========================================================================
// NUMA Topology Detection and Management
// Per-NUMA-node injector queues and locality-aware scheduling
// Parses /sys/devices/system/node/ on Linux for NUMA topology
// =========================================================================

#[cfg(feature = "numa")]
use nix::sched::{CpuSet, sched_setaffinity};

/// NUMA node information
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// Node ID
    pub id: usize,
    /// CPU cores belonging to this node
    pub cpus: Vec<usize>,
    /// Distance to other nodes (optional)
    pub distances: Vec<usize>,
}

/// NUMA topology
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// All NUMA nodes
    pub nodes: Vec<NumaNode>,
    /// Total number of CPU cores
    pub total_cores: usize,
    /// Number of NUMA nodes
    pub num_nodes: usize,
    /// Issue #81: Cached CPU→NUMA-node-index mapping for O(1) lookup.
    /// `cpu_to_node[cpu_id]` returns the index into `nodes` for that CPU,
    /// or `CPU_NOT_FOUND` if the CPU is not present in any node.
    /// Built once during `detect()` / `detect_from_sysfs()` to avoid the
    /// O(N×M) linear scan that the old `get_node_for_cpu()` performed
    /// (iterating all nodes and calling `cpus.contains(&cpu)` on each).
    cpu_to_node: Vec<usize>,
}

/// Sentinel value in `cpu_to_node` indicating the CPU is not in any node.
const CPU_NOT_FOUND: usize = usize::MAX;

/// Build the cpu_to_node cache from the list of NUMA nodes.
fn build_cpu_to_node_cache(nodes: &[NumaNode]) -> Vec<usize> {
    // Find the maximum CPU ID to size the cache vector.
    let max_cpu = nodes.iter()
        .flat_map(|n| n.cpus.iter().copied())
        .max()
        .unwrap_or(0);
    let mut cache = vec![CPU_NOT_FOUND; max_cpu + 1];
    for (node_idx, node) in nodes.iter().enumerate() {
        for &cpu in &node.cpus {
            if cpu < cache.len() {
                cache[cpu] = node_idx;
            }
        }
    }
    cache
}

impl NumaTopology {
    /// Detect NUMA topology by parsing /sys/devices/system/node/
    pub fn detect() -> Self {
        #[cfg(feature = "numa")]
        {
            Self::detect_from_sysfs()
        }
        
        #[cfg(not(feature = "numa"))]
        {
            // Fallback: single NUMA node with all cores
            let num_cores = num_cpus::get();
            let cpus: Vec<usize> = (0..num_cores).collect();
            
            let nodes = vec![NumaNode { 
                id: 0, 
                cpus,
                distances: vec![10],
            }];
            let cpu_to_node = build_cpu_to_node_cache(&nodes);
            
            Self {
                nodes,
                total_cores: num_cores,
                num_nodes: 1,
                cpu_to_node,
            }
        }
    }

    #[cfg(feature = "numa")]
    #[cfg(target_os = "linux")]
    fn detect_from_sysfs() -> Self {
        use std::fs;
        use std::path::Path;
        
        let mut nodes = Vec::new();
        let mut total_cores = 0;
        
        // Try to parse /sys/devices/system/node/
        let nodes_path = Path::new("/sys/devices/system/node");
        
        if nodes_path.exists() {
            // Iterate over node directories (node0, node1, etc.)
            if let Ok(entries) = fs::read_dir(nodes_path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let name = entry.file_name().to_string_lossy();
                    
                    if name.starts_with("node") && path.is_dir() {
                        // Parse node ID
                        if let Ok(node_id) = name[4..].parse::<usize>() {
                            // Parse CPUs for this node
                            let cpus = Self::parse_node_cpus(&path);
                            total_cores += cpus.len();
                            
                            // Parse distances (optional)
                            let distances = Self::parse_node_distances(&path);
                            
                            nodes.push(NumaNode {
                                id: node_id,
                                cpus,
                                distances,
                            });
                        }
                    }
                }
            }
        }
        
        // If no NUMA nodes detected, fall back to single node
        if nodes.is_empty() {
            let num_cores = num_cpus::get();
            let cpus: Vec<usize> = (0..num_cores).collect();
            
            let nodes = vec![NumaNode { 
                id: 0, 
                cpus,
                distances: vec![10],
            }];
            let cpu_to_node = build_cpu_to_node_cache(&nodes);
            
            return Self {
                nodes,
                total_cores: num_cores,
                num_nodes: 1,
                cpu_to_node,
            };
        }
        
        let num_nodes = nodes.len();
        let cpu_to_node = build_cpu_to_node_cache(&nodes);
        
        Self {
            nodes,
            total_cores,
            num_nodes,
            cpu_to_node,
        }
    }

    #[cfg(feature = "numa")]
    #[cfg(not(target_os = "linux"))]
    fn detect_from_sysfs() -> Self {
        // Non-Linux systems: fall back to single NUMA node
        let num_cores = num_cpus::get();
        let cpus: Vec<usize> = (0..num_cores).collect();
        
        let nodes = vec![NumaNode { 
            id: 0, 
            cpus,
            distances: vec![10],
        }];
        let cpu_to_node = build_cpu_to_node_cache(&nodes);
        
        Self {
            nodes,
            total_cores: num_cores,
            num_nodes: 1,
            cpu_to_node,
        }
    }

    #[cfg(feature = "numa")]
    #[cfg(target_os = "linux")]
    fn parse_node_cpus(node_path: &std::path::Path) -> Vec<usize> {
        use std::fs;
        
        let cpulist_path = node_path.join("cpulist");
        let mut cpus = Vec::new();
        
        if let Ok(content) = fs::read_to_string(&cpulist_path) {
            // Parse CPU list format: "0-7,16-23" or "0,1,2,3"
            for part in content.trim().split(',') {
                if part.contains('-') {
                    // Range format: "0-7"
                    let range: Vec<&str> = part.split('-').collect();
                    if range.len() == 2 {
                        if let (Ok(start), Ok(end)) = (range[0].parse::<usize>(), range[1].parse::<usize>()) {
                            for cpu in start..=end {
                                cpus.push(cpu);
                            }
                        }
                    }
                } else {
                    // Single CPU format: "0"
                    if let Ok(cpu) = part.parse::<usize>() {
                        cpus.push(cpu);
                    }
                }
            }
        }
        
        cpus.sort();
        cpus.dedup();
        cpus
    }

    #[cfg(feature = "numa")]
    #[cfg(target_os = "linux")]
    fn parse_node_distances(node_path: &std::path::Path) -> Vec<usize> {
        use std::fs;
        
        let distance_path = node_path.join("distance");
        let mut distances = Vec::new();
        
        if let Ok(content) = fs::read_to_string(&distance_path) {
            // Parse distance format: "10 20 30 40"
            for part in content.trim().split_whitespace() {
                if let Ok(d) = part.parse::<usize>() {
                    distances.push(d);
                }
            }
        }
        
        if distances.is_empty() {
            distances.push(10); // Default distance
        }
        
        distances
    }

    /// Get the NUMA node for a given CPU core.
    ///
    /// Issue #81: Previously this performed an O(N×M) linear scan
    /// (iterating all nodes and calling `cpus.contains(&cpu)` which
    /// is O(M) per node). Now uses a pre-built `cpu_to_node` Vec for
    /// O(1) lookup: index by CPU ID, get the node index directly.
    #[inline]
    pub fn get_node_for_cpu(&self, cpu: usize) -> Option<&NumaNode> {
        let node_idx = *self.cpu_to_node.get(cpu).unwrap_or(&CPU_NOT_FOUND);
        if node_idx == CPU_NOT_FOUND {
            None
        } else {
            self.nodes.get(node_idx)
        }
    }

    /// Get all CPUs in a given NUMA node
    pub fn get_cpus_in_node(&self, node_id: usize) -> Vec<usize> {
        self.nodes
            .get(node_id)
            .map(|node| node.cpus.clone())
            .unwrap_or_default()
    }

    /// Get the distance between two NUMA nodes
    pub fn get_distance(&self, node_a: usize, node_b: usize) -> usize {
        if let Some(node) = self.nodes.get(node_a) {
            if node_b < node.distances.len() {
                return node.distances[node_b];
            }
        }
        20 // Default distance for unknown nodes
    }

    /// Check if the system is NUMA-aware
    pub fn is_numa(&self) -> bool {
        self.num_nodes > 1
    }
}

impl Default for NumaTopology {
    fn default() -> Self {
        Self::detect()
    }
}

/// Get the number of CPU cores
pub fn num_cores() -> usize {
    num_cpus::get()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_detection() {
        let topology = NumaTopology::detect();
        assert!(topology.total_cores > 0);
        assert!(!topology.nodes.is_empty());
    }

    #[test]
    fn test_num_cores() {
        let n = num_cores();
        assert!(n > 0);
    }

    #[test]
    fn test_get_node_for_cpu() {
        let topology = NumaTopology::detect();
        if topology.total_cores > 0 {
            let node = topology.get_node_for_cpu(0);
            assert!(node.is_some());
        }
    }

    #[test]
    fn test_get_cpus_in_node() {
        let topology = NumaTopology::detect();
        let cpus = topology.get_cpus_in_node(0);
        assert!(!cpus.is_empty() || topology.num_nodes == 0);
    }

    #[test]
    fn test_get_distance() {
        let topology = NumaTopology::detect();
        let distance = topology.get_distance(0, 0);
        assert!(distance > 0);
    }

    #[test]
    fn test_is_numa() {
        let topology = NumaTopology::detect();
        let is_numa = topology.is_numa();
        // Should work regardless of NUMA or UMA
        let _ = is_numa;
    }

    #[test]
    fn test_cpu_to_node_cache_o1_lookup() {
        // Issue #81: Verify that get_node_for_cpu returns correct results
        // using the O(1) cached lookup instead of linear scan.
        let topology = NumaTopology::detect();
        // Every CPU listed in a node should be found by get_node_for_cpu
        for node in &topology.nodes {
            for &cpu in &node.cpus {
                let found = topology.get_node_for_cpu(cpu);
                assert!(found.is_some(), "CPU {} not found in cache", cpu);
                assert_eq!(found.unwrap().id, node.id,
                    "CPU {} mapped to wrong node (expected {}, got {})",
                    cpu, node.id, found.unwrap().id);
            }
        }
        // A CPU ID beyond the max should return None
        let beyond = topology.cpu_to_node.len() + 100;
        assert!(topology.get_node_for_cpu(beyond).is_none());
    }
}
