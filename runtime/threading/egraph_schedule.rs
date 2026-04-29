// =========================================================================
// AOT Scheduling with E-Graph Extraction
// Compile-time extraction of optimal task graphs from e-graphs
// Precomputed schedules for near-zero runtime overhead
// Static scheduling with known execution patterns
// =========================================================================

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Task node in the e-graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EGraphNode {
    /// Node ID
    pub id: usize,
    /// Operation type
    pub op: String,
    /// Dependencies (node IDs)
    pub dependencies: Vec<usize>,
    /// Estimated cost (cycles)
    pub cost: u64,
    /// Can be parallelized
    pub parallelizable: bool,
}

/// E-graph representation of a computation
#[derive(Debug, Clone)]
pub struct EGraph {
    /// Nodes in the graph
    pub nodes: Vec<EGraphNode>,
    /// Entry nodes (no dependencies)
    pub entry_nodes: Vec<usize>,
    /// Exit nodes (no dependents)
    pub exit_nodes: Vec<usize>,
}

impl EGraph {
    /// Create a new e-graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            entry_nodes: Vec::new(),
            exit_nodes: Vec::new(),
        }
    }
    
    /// Add a node to the e-graph
    pub fn add_node(&mut self, node: EGraphNode) {
        let id = node.id;
        self.nodes.push(node);
        
        // Update entry/exit nodes
        if node.dependencies.is_empty() {
            self.entry_nodes.push(id);
        }
    }
    
    /// Build the e-graph from nodes
    pub fn build(&mut self) {
        // Find exit nodes (nodes with no dependents)
        let mut has_dependents: HashSet<usize> = HashSet::new();
        
        for node in &self.nodes {
            for &dep in &node.dependencies {
                has_dependents.insert(dep);
            }
        }
        
        self.exit_nodes = self.nodes.iter()
            .filter(|n| !has_dependents.contains(&n.id))
            .map(|n| n.id)
            .collect();
    }
    
    /// Get a node by ID
    pub fn get_node(&self, id: usize) -> Option<&EGraphNode> {
        self.nodes.iter().find(|n| n.id == id)
    }
    
    /// Get the number of nodes
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Precomputed schedule for a task graph
#[derive(Debug, Clone)]
pub struct PrecomputedSchedule {
    /// Execution order (node IDs)
    pub execution_order: Vec<usize>,
    /// Parallel groups (each group can execute in parallel)
    pub parallel_groups: Vec<Vec<usize>>,
    /// Total estimated cost
    pub total_cost: u64,
    /// Critical path length
    pub critical_path: u64,
}

impl PrecomputedSchedule {
    /// Create a new precomputed schedule
    pub fn new() -> Self {
        Self {
            execution_order: Vec::new(),
            parallel_groups: Vec::new(),
            total_cost: 0,
            critical_path: 0,
        }
    }
    
    /// Add a parallel group
    pub fn add_parallel_group(&mut self, group: Vec<usize>) {
        self.parallel_groups.push(group);
    }
    
    /// Set the execution order
    pub fn set_execution_order(&mut self, order: Vec<usize>) {
        self.execution_order = order;
    }
    
    /// Set the total cost
    pub fn set_total_cost(&mut self, cost: u64) {
        self.total_cost = cost;
    }
    
    /// Set the critical path
    pub fn set_critical_path(&mut self, path: u64) {
        self.critical_path = path;
    }
}

impl Default for PrecomputedSchedule {
    fn default() -> Self {
        Self::new()
    }
}

/// E-graph extractor for schedule extraction
pub struct EGraphExtractor {
    /// E-graph to extract from
    egraph: EGraph,
    /// Number of workers available
    num_workers: usize,
}

impl EGraphExtractor {
    /// Create a new e-graph extractor
    pub fn new(egraph: EGraph, num_workers: usize) -> Self {
        Self {
            egraph,
            num_workers,
        }
    }
    
    /// Extract an optimal schedule from the e-graph
    pub fn extract_schedule(&self) -> PrecomputedSchedule {
        let mut schedule = PrecomputedSchedule::new();
        
        // Topological sort for execution order
        let execution_order = self.topological_sort();
        schedule.set_execution_order(execution_order.clone());
        
        // Group by parallelism
        let parallel_groups = self.compute_parallel_groups(&execution_order);
        for group in parallel_groups {
            schedule.add_parallel_group(group);
        }
        
        // Compute costs
        let (total_cost, critical_path) = self.compute_costs(&execution_order);
        schedule.set_total_cost(total_cost);
        schedule.set_critical_path(critical_path);
        
        schedule
    }
    
    /// Topological sort of the e-graph
    fn topological_sort(&self) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        
        for &entry_id in &self.egraph.entry_nodes {
            self.dfs_visit(entry_id, &mut visited, &mut order);
        }
        
        order.reverse();
        order
    }
    
    /// DFS visit for topological sort
    fn dfs_visit(&self, node_id: usize, visited: &mut HashSet<usize>, order: &mut Vec<usize>) {
        if visited.contains(&node_id) {
            return;
        }
        
        visited.insert(node_id);
        
        if let Some(node) = self.egraph.get_node(node_id) {
            for &dep in &node.dependencies {
                self.dfs_visit(dep, visited, order);
            }
        }
        
        order.push(node_id);
    }
    
    /// Compute parallel groups from execution order
    fn compute_parallel_groups(&self, execution_order: &[usize]) -> Vec<Vec<usize>> {
        let mut groups = Vec::new();
        let mut current_group = Vec::new();
        let mut in_current_group = HashSet::new();
        
        for &node_id in execution_order {
            if let Some(node) = self.egraph.get_node(node_id) {
                // Check if all dependencies are in previous groups
                let deps_ready = node.dependencies.iter()
                    .all(|dep| in_current_group.contains(dep) || groups.iter().flatten().any(|id| id == dep));
                
                if deps_ready && current_group.len() < self.num_workers {
                    current_group.push(node_id);
                    in_current_group.insert(node_id);
                } else {
                    if !current_group.is_empty() {
                        groups.push(current_group);
                        current_group = Vec::new();
                        in_current_group = HashSet::new();
                    }
                    current_group.push(node_id);
                    in_current_group.insert(node_id);
                }
            }
        }
        
        if !current_group.is_empty() {
            groups.push(current_group);
        }
        
        groups
    }
    
    /// Compute total cost and critical path
    fn compute_costs(&self, execution_order: &[usize]) -> (u64, u64) {
        let mut total_cost = 0u64;
        let mut node_costs: HashMap<usize, u64> = HashMap::new();
        
        for &node_id in execution_order {
            if let Some(node) = self.egraph.get_node(node_id) {
                let dep_cost = node.dependencies.iter()
                    .map(|dep| *node_costs.get(dep).unwrap_or(&0))
                    .max()
                    .unwrap_or(0);
                
                let node_cost = dep_cost + node.cost;
                node_costs.insert(node_id, node_cost);
                total_cost += node.cost;
            }
        }
        
        let critical_path = node_costs.values().copied().max().unwrap_or(0);
        
        (total_cost, critical_path)
    }
}

/// AOT scheduler with precomputed schedules
pub struct AotScheduler {
    /// Precomputed schedules for different e-graphs
    schedules: HashMap<String, PrecomputedSchedule>,
    /// Current active schedule
    active_schedule: Option<String>,
}

impl AotScheduler {
    /// Create a new AOT scheduler
    pub fn new() -> Self {
        Self {
            schedules: HashMap::new(),
            active_schedule: None,
        }
    }
    
    /// Add a precomputed schedule
    pub fn add_schedule(&mut self, key: String, schedule: PrecomputedSchedule) {
        self.schedules.insert(key, schedule);
    }
    
    /// Activate a schedule
    pub fn activate_schedule(&mut self, key: &str) -> Result<(), String> {
        if self.schedules.contains_key(key) {
            self.active_schedule = Some(key.to_string());
            Ok(())
        } else {
            Err("Schedule not found".to_string())
        }
    }
    
    /// Get the active schedule
    pub fn active_schedule(&self) -> Option<&PrecomputedSchedule> {
        self.active_schedule.as_ref()
            .and_then(|key| self.schedules.get(key))
    }
    
    /// Get a schedule by key
    pub fn get_schedule(&self, key: &str) -> Option<&PrecomputedSchedule> {
        self.schedules.get(key)
    }
    
    /// Extract and add a schedule from an e-graph
    pub fn extract_and_add(&mut self, key: String, egraph: EGraph, num_workers: usize) {
        let extractor = EGraphExtractor::new(egraph, num_workers);
        let schedule = extractor.extract_schedule();
        self.add_schedule(key, schedule);
    }
    
    /// Get the next task to execute from the active schedule
    pub fn next_task(&self) -> Option<usize> {
        if let Some(schedule) = self.active_schedule() {
            if !schedule.execution_order.is_empty() {
                Some(schedule.execution_order[0])
            } else {
                None
            }
        } else {
            None
        }
    }
    
    /// Get the next parallel group
    pub fn next_parallel_group(&self) -> Option<&Vec<usize>> {
        if let Some(schedule) = self.active_schedule() {
            if !schedule.parallel_groups.is_empty() {
                Some(&schedule.parallel_groups[0])
            } else {
                None
            }
        } else {
            None
        }
    }
    
    /// Mark a task as completed
    pub fn mark_completed(&mut self, task_id: usize) {
        if let Some(key) = self.active_schedule.clone() {
            if let Some(schedule) = self.schedules.get_mut(&key) {
                schedule.execution_order.retain(|&id| id != task_id);
                
                // Also remove from parallel groups
                for group in &mut schedule.parallel_groups {
                    group.retain(|&id| id != task_id);
                }
                
                // Remove empty groups
                schedule.parallel_groups.retain(|g| !g.is_empty());
            }
        }
    }
    
    /// Check if the schedule is complete
    pub fn is_complete(&self) -> bool {
        if let Some(schedule) = self.active_schedule() {
            schedule.execution_order.is_empty()
        } else {
            true
        }
    }
}

impl Default for AotScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// E-graph builder for constructing e-graphs from code
pub struct EGraphBuilder {
    /// E-graph being built
    egraph: EGraph,
    /// Next node ID
    next_id: usize,
}

impl EGraphBuilder {
    /// Create a new e-graph builder
    pub fn new() -> Self {
        Self {
            egraph: EGraph::new(),
            next_id: 0,
        }
    }
    
    /// Add a node to the e-graph
    pub fn add_node(&mut self, op: String, dependencies: Vec<usize>, cost: u64, parallelizable: bool) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        
        let node = EGraphNode {
            id,
            op,
            dependencies,
            cost,
            parallelizable,
        };
        
        self.egraph.add_node(node);
        id
    }
    
    /// Build the e-graph
    pub fn build(mut self) -> EGraph {
        self.egraph.build();
        self.egraph
    }
    
    /// Get the current e-graph
    pub fn egraph(&self) -> &EGraph {
        &self.egraph
    }
}

impl Default for EGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_egraph_creation() {
        let mut egraph = EGraph::new();
        egraph.add_node(EGraphNode {
            id: 0,
            op: "add".to_string(),
            dependencies: vec![],
            cost: 10,
            parallelizable: true,
        });
        egraph.build();
        
        assert_eq!(egraph.len(), 1);
        assert_eq!(egraph.entry_nodes.len(), 1);
    }

    #[test]
    fn test_egraph_extractor() {
        let mut egraph = EGraph::new();
        egraph.add_node(EGraphNode {
            id: 0,
            op: "add".to_string(),
            dependencies: vec![],
            cost: 10,
            parallelizable: true,
        });
        egraph.add_node(EGraphNode {
            id: 1,
            op: "mul".to_string(),
            dependencies: vec![0],
            cost: 20,
            parallelizable: true,
        });
        egraph.build();
        
        let extractor = EGraphExtractor::new(egraph, 2);
        let schedule = extractor.extract_schedule();
        
        assert!(!schedule.execution_order.is_empty());
    }

    #[test]
    fn test_aot_scheduler() {
        let mut scheduler = AotScheduler::new();
        
        let mut egraph = EGraph::new();
        egraph.add_node(EGraphNode {
            id: 0,
            op: "add".to_string(),
            dependencies: vec![],
            cost: 10,
            parallelizable: true,
        });
        egraph.build();
        
        scheduler.extract_and_add("test".to_string(), egraph, 2);
        scheduler.activate_schedule("test").unwrap();
        
        let task = scheduler.next_task();
        assert!(task.is_some());
    }

    #[test]
    fn test_egraph_builder() {
        let mut builder = EGraphBuilder::new();
        let id1 = builder.add_node("add".to_string(), vec![], 10, true);
        let id2 = builder.add_node("mul".to_string(), vec![id1], 20, true);
        
        let egraph = builder.build();
        assert_eq!(egraph.len(), 2);
    }
}
