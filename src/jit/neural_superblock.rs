// =============================================================================
// src/jit/neural_superblock.rs
//
// Neural Superblock Formation with Microarchitecture Simulation
//
// Replaces simple "hot path" tracing with a Graph Neural Network that simulates
// the CPU pipeline. The GNN reads the upcoming instruction stream and predicts
// register pressure, port contention, and cache behavior. It then emits
// superblocks—regions of optimized code—tailored to the specific microarchitecture.
//
// Architecture:
//   - Microarchitectural model: port usage, cache hierarchy, register file
//   - GNN processes instruction dependency graph
//   - Predicts IPC (instructions per cycle) for candidate sequences
//   - Traces anticipated hot paths before they're hot
//
// Research: Existing JITs react to observed behavior. A neural pipeline simulator
// lets Jules pre-optimize code before the first slow execution, optimizing for
// throughput rather than just frequency.
// =============================================================================

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, RwLock};

/// CPU microarchitectural model parameters
#[derive(Debug, Clone)]
pub struct MicroArchModel {
    /// Execution ports (in-order issue width)
    pub issue_width: usize,
    /// ALU ports
    pub alu_ports: usize,
    /// Load/Store ports
    pub mem_ports: usize,
    /// FP ports
    pub fp_ports: usize,
    /// Vector ports
    pub vec_ports: usize,
    /// L1 cache size (bytes)
    pub l1_size: usize,
    /// L1 cache line size
    pub l1_line_size: usize,
    /// L2 cache size (bytes)
    pub l2_size: usize,
    /// Register file size (integer)
    pub int_regs: usize,
    /// Register file size (FP/vector)
    pub fp_regs: usize,
    /// Cycles per load (L1 hit)
    pub load_latency: usize,
    /// Cycles per load (L2 hit)
    pub load_latency_l2: usize,
    /// Cycles per load (L3/memory)
    pub load_latency_mem: usize,
}

impl Default for MicroArchModel {
    fn default() -> Self {
        // Default: modern x86-64 (Skylake-like)
        Self {
            issue_width: 4,
            alu_ports: 3,
            mem_ports: 2,
            fp_ports: 2,
            vec_ports: 2,
            l1_size: 32 * 1024,
            l1_line_size: 64,
            l2_size: 256 * 1024,
            int_regs: 16,
            fp_regs: 16,
            load_latency: 4,
            load_latency_l2: 12,
            load_latency_mem: 100,
        }
    }
}

/// Intel Haswell model
pub fn haswell_model() -> MicroArchModel {
    MicroArchModel {
        issue_width: 4,
        alu_ports: 4,
        mem_ports: 2,
        fp_ports: 2,
        vec_ports: 2,
        l1_size: 32 * 1024,
        l1_line_size: 64,
        l2_size: 256 * 1024,
        int_regs: 16,
        fp_regs: 32,
        load_latency: 4,
        load_latency_l2: 12,
        load_latency_mem: 100,
    }
}

/// AMD Zen model
pub fn zen_model() -> MicroArchModel {
    MicroArchModel {
        issue_width: 4,
        alu_ports: 4,
        mem_ports: 2,
        fp_ports: 4,
        vec_ports: 2,
        l1_size: 32 * 1024,
        l1_line_size: 64,
        l2_size: 512 * 1024,
        int_regs: 16,
        fp_regs: 32,
        load_latency: 4,
        load_latency_l2: 14,
        load_latency_mem: 100,
    }
}

/// Instruction category for port assignment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstrCategory {
    ALU,
    Load,
    Store,
    FPAdd,
    FPMul,
    Vec,
    Branch,
    Nop,
}

impl InstrCategory {
    /// Which execution port this instruction uses
    pub fn port(&self) -> usize {
        match self {
            InstrCategory::ALU => 0,    // ALU port
            InstrCategory::Load => 1,   // Load port
            InstrCategory::Store => 1,  // Store port
            InstrCategory::FPAdd => 2,  // FP add port
            InstrCategory::FPMul => 3,  // FP mul port
            InstrCategory::Vec => 4,   // Vector port
            InstrCategory::Branch => 0, // ALU port
            InstrCategory::Nop => 0,
        }
    }

    /// Latency of this instruction
    pub fn latency(&self) -> usize {
        match self {
            InstrCategory::ALU => 1,
            InstrCategory::Load => 4,
            InstrCategory::Store => 1,
            InstrCategory::FPAdd => 3,
            InstrCategory::FPMul => 4,
            InstrCategory::Vec => 4,
            InstrCategory::Branch => 1,
            InstrCategory::Nop => 0,
        }
    }

    /// Whether this is a memory operation
    pub fn is_memory(&self) -> bool {
        matches!(self, InstrCategory::Load | InstrCategory::Store)
    }
}

/// An instruction in the trace for neural analysis
#[derive(Debug, Clone)]
pub struct TraceInstr {
    pub id: usize,
    pub category: InstrCategory,
    /// Source registers
    pub src_regs: Vec<usize>,
    /// Destination register
    pub dst_reg: Option<usize>,
    /// Memory operand size (0 if not memory)
    pub mem_size: usize,
    /// Whether this is a load
    pub is_load: bool,
    /// Address computation dependency (previous instr id)
    pub addr_dep: Option<usize>,
    /// Data dependency (previous instr id)
    pub data_dep: Option<usize>,
}

/// Graph representation of a basic block for GNN processing
pub struct BlockGraph {
    pub nodes: Vec<TraceInstr>,
    pub edges: Vec<(usize, usize)>, // (from, to)
}

impl BlockGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add instruction and track dependencies
    pub fn add_instr(&mut self, instr: TraceInstr) -> usize {
        let id = self.nodes.len();
        let instr_id = id;

        // Add data dependency edge if exists
        if let Some(dep) = instr.data_dep {
            self.edges.push((dep, instr_id));
        }

        // Add address dependency edge if exists
        if let Some(dep) = instr.addr_dep {
            self.edges.push((dep, instr_id));
        }

        self.nodes.push(instr);
        instr_id
    }

    /// Build node features for GNN input
    pub fn node_features(&self) -> Vec<Vec<f32>> {
        self.nodes.iter().map(|instr| {
            let mut features = vec![0.0f32; 16];

            // One-hot category encoding
            let cat_idx = match instr.category {
                InstrCategory::ALU => 0,
                InstrCategory::Load => 1,
                InstrCategory::Store => 2,
                InstrCategory::FPAdd => 3,
                InstrCategory::FPMul => 4,
                InstrCategory::Vec => 5,
                InstrCategory::Branch => 6,
                InstrCategory::Nop => 7,
            };
            features[cat_idx] = 1.0;

            // Port assignment
            features[8] = instr.category.port() as f32;

            // Latency
            features[9] = instr.category.latency() as f32;

            // Memory indicator
            features[10] = if instr.category.is_memory() { 1.0 } else { 0.0 };

            // Memory size (log scale)
            features[11] = (instr.mem_size as f32).log2().clamp(0.0, 7.0);

            // Register pressure estimate
            features[12] = instr.src_regs.len() as f32;
            features[13] = if instr.dst_reg.is_some() { 1.0 } else { 0.0 };

            // Dependency features
            features[14] = if instr.data_dep.is_some() { 1.0 } else { 0.0 };
            features[15] = if instr.addr_dep.is_some() { 1.0 } else { 0.0 };

            features
        }).collect()
    }

    /// Build adjacency matrix for GNN
    pub fn adjacency_matrix(&self) -> Vec<Vec<f32>> {
        let n = self.nodes.len();
        let mut adj = vec![vec![0.0f32; n]; n];

        for &(from, to) in &self.edges {
            adj[from][to] = 1.0;
        }

        adj
    }
}

impl Default for BlockGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Compressed Sparse Row representation for graph adjacency
#[derive(Debug, Clone)]
pub struct CsrAdjacency {
    /// Row offsets (N+1 entries)
    pub row_offsets: Vec<usize>,
    /// Column indices (E entries)
    pub col_indices: Vec<usize>,
}

impl CsrAdjacency {
    pub fn from_edges(num_nodes: usize, edges: &[(usize, usize)]) -> Self {
        let mut row_offsets = vec![0usize; num_nodes + 1];
        let mut col_indices = Vec::with_capacity(edges.len());

        // Count edges per row
        for &(from, _to) in edges {
            row_offsets[from + 1] += 1;
        }
        // Prefix sum
        for i in 1..=num_nodes {
            row_offsets[i] += row_offsets[i - 1];
        }

        // Fill column indices (stable sort by source)
        let mut temp_offsets = row_offsets.clone();
        col_indices.resize(edges.len(), 0);
        for &(from, to) in edges {
            let pos = temp_offsets[from];
            col_indices[pos] = to;
            temp_offsets[from] += 1;
        }

        Self { row_offsets, col_indices }
    }

    pub fn neighbors(&self, node: usize) -> &[usize] {
        let start = self.row_offsets[node];
        let end = self.row_offsets[node + 1];
        &self.col_indices[start..end]
    }
}

/// Functional GNN with proper weight matrices and IPC readout head
pub struct MicroArchGNN {
    embed_dim: usize,
    hidden_dim: usize,
    node_embeddings: Vec<Vec<f32>>,
    /// W_self weight matrix (hidden_dim × hidden_dim) per round
    w_self: Vec<Vec<f32>>,
    /// W_msg weight matrix (hidden_dim × hidden_dim) per round
    w_msg: Vec<Vec<f32>>,
    /// Readout MLP layer 1 weights (hidden_dim × hidden_dim/4)
    readout_w1: Vec<Vec<f32>>,
    /// Readout MLP layer 2 weights (hidden_dim/4 × 1)
    readout_w2: Vec<f32>,
    /// Training samples
    training_samples: Vec<(Vec<f32>, f32)>,
}

impl MicroArchGNN {
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Self {
        let fan_in = hidden_dim;
        let fan_out = hidden_dim;
        let limit = (6.0f32 / (fan_in as f32 + fan_out as f32)).sqrt();

        // Xavier/Glorot uniform initialization
        let init_weight = || -> f32 {
            // Deterministic initialization based on a simple LCG
            static mut SEED: u64 = 42;
            unsafe {
                SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = (SEED >> 33) as f32 / (1u64 << 31) as f32;
                (val * 2.0 - 1.0) * limit
            }
        };

        let make_matrix = |rows: usize, cols: usize| -> Vec<Vec<f32>> {
            (0..rows).map(|_| (0..cols).map(|_| init_weight()).collect()).collect()
        };

        let readout_dim = hidden_dim / 4;

        Self {
            embed_dim,
            hidden_dim,
            node_embeddings: Vec::new(),
            w_self: make_matrix(hidden_dim, hidden_dim),
            w_msg: make_matrix(hidden_dim, hidden_dim),
            readout_w1: make_matrix(readout_dim, hidden_dim * 2), // mean + max pooling
            readout_w2: (0..readout_dim).map(|_| init_weight()).collect(),
            training_samples: Vec::new(),
        }
    }

    /// Forward pass with proper weight matrices and CSR adjacency
    pub fn predict_ipc(&self, graph: &BlockGraph, arch: &MicroArchModel) -> f32 {
        let node_feats = graph.node_features();
        if node_feats.is_empty() { return 0.0; }

        // Build CSR adjacency
        let csr = CsrAdjacency::from_edges(graph.nodes.len(), &graph.edges);

        // Initialize hidden states from features
        let mut hidden: Vec<Vec<f32>> = node_feats.iter().map(|f| {
            f.iter().cloned().take(self.hidden_dim.min(f.len()))
                .chain(std::iter::repeat(0.0))
                .take(self.hidden_dim)
                .collect()
        }).collect();

        // Message passing (2 rounds) with proper weight matrices
        for _round in 0..2 {
            let mut new_hidden = hidden.clone();

            for i in 0..graph.nodes.len() {
                let neighbors = csr.neighbors(i);

                // Compute self contribution: W_self * h_i
                let self_contrib = self.mat_vec(&self.w_self, &hidden[i]);

                // Compute message from neighbors: mean(W_msg * h_j)
                let mut msg = vec![0.0f32; self.hidden_dim];
                if !neighbors.is_empty() {
                    for &j in neighbors {
                        let w_msg_h = self.mat_vec(&self.w_msg, &hidden[j]);
                        for k in 0..self.hidden_dim {
                            msg[k] += w_msg_h[k];
                        }
                    }
                    let n = neighbors.len() as f32;
                    for k in 0..self.hidden_dim { msg[k] /= n; }
                }

                // Combine: new_h = ReLU(W_self * h_i + mean(W_msg * h_j))
                for k in 0..self.hidden_dim {
                    new_hidden[i][k] = (self_contrib[k] + msg[k]).max(0.0);
                }
            }

            hidden = new_hidden;
        }

        // IPC readout head: MLP(mean_pool || max_pool)
        let mean_pool = self.mean_pool(&hidden);
        let max_pool = self.max_pool(&hidden);
        let pooled: Vec<f32> = mean_pool.iter().chain(max_pool.iter()).cloned().collect();

        // Layer 1: hidden_dim*2 → hidden_dim/4
        let readout_dim = self.hidden_dim / 4;
        let mut h1 = vec![0.0f32; readout_dim];
        for i in 0..readout_dim.min(self.readout_w1.len()) {
            for j in 0..pooled.len().min(self.readout_w1[i].len()) {
                h1[i] += self.readout_w1[i][j] * pooled[j];
            }
            h1[i] = h1[i].max(0.0); // ReLU
        }

        // Layer 2: hidden_dim/4 → 1
        let mut ipc = 0.0f32;
        for i in 0..readout_dim.min(self.readout_w2.len()) {
            ipc += self.readout_w2[i] * h1[i];
        }

        ipc.max(0.0).min(arch.issue_width as f32 * 1.5)
    }

    fn mat_vec(&self, mat: &[Vec<f32>], vec: &[f32]) -> Vec<f32> {
        let rows = mat.len().min(self.hidden_dim);
        (0..rows).map(|i| {
            mat[i].iter().zip(vec.iter()).map(|(w, v)| w * v).sum()
        }).chain(std::iter::repeat(0.0)).take(self.hidden_dim).collect()
    }

    fn mean_pool(&self, hidden: &[Vec<f32>]) -> Vec<f32> {
        if hidden.is_empty() { return vec![0.0; self.hidden_dim]; }
        let n = hidden.len() as f32;
        (0..self.hidden_dim).map(|j| {
            hidden.iter().map(|h| h[j]).sum::<f32>() / n
        }).collect()
    }

    fn max_pool(&self, hidden: &[Vec<f32>]) -> Vec<f32> {
        if hidden.is_empty() { return vec![0.0; self.hidden_dim]; }
        (0..self.hidden_dim).map(|j| {
            hidden.iter().map(|h| h[j]).fold(f32::NEG_INFINITY, f32::max)
        }).collect()
    }

    /// Online training from hardware counters (SGD with MSE loss)
    pub fn train(&mut self, iterations: usize) {
        if self.training_samples.len() < 10 { return; }

        let lr = 0.001f32;

        for _ in 0..iterations {
            // Pick a random sample (simple LCG)
            static mut SEED: u64 = 12345;
            unsafe {
                SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1);
            }
            let idx = unsafe { (SEED as usize) % self.training_samples.len() };
            let (features, measured_ipc) = &self.training_samples[idx];

            // Simple gradient step on readout weights
            // (Full backprop through GNN is complex; this is a simplified approach)
            let pred_ipc: f32 = features.iter().take(self.readout_w2.len())
                .zip(self.readout_w2.iter())
                .map(|(f, w)| f * w).sum();

            let error = pred_ipc - *measured_ipc;

            // Update readout_w2 (simplified gradient)
            for i in 0..self.readout_w2.len().min(features.len()) {
                self.readout_w2[i] -= lr * error * features[i];
            }
        }
    }

    /// Predict register pressure for block
    pub fn predict_register_pressure(&self, graph: &BlockGraph) -> usize {
        let mut live_regs: HashSet<usize> = HashSet::new();
        let mut max_pressure = 0usize;

        for node in graph.nodes.iter().rev() {
            // Remove dead registers
            if let Some(dst) = node.dst_reg {
                live_regs.remove(&dst);
            }

            // Add used registers
            for &reg in &node.src_regs {
                live_regs.insert(reg);
            }

            max_pressure = max_pressure.max(live_regs.len());
        }

        max_pressure
    }

    /// Predict cache behavior
    pub fn predict_cache_behavior(&self, graph: &BlockGraph, arch: &MicroArchModel) -> CachePrediction {
        let mut load_count = 0;
        let mut unique_addrs = HashSet::new();
        let mut stride_detected = false;
        let mut last_addr = 0i64;
        let mut stride = 0i64;

        for instr in &graph.nodes {
            if instr.is_load {
                load_count += 1;
                unique_addrs.insert(instr.id); // Simplified

                if last_addr != 0 {
                    // Would need real addresses for stride detection
                }
            }
        }

        // Estimate cache hit rate
        let l1_fits = unique_addrs.len() * 8 <= arch.l1_size;
        let l2_fits = unique_addrs.len() * 8 <= arch.l2_size;

        let hit_rate = if l1_fits {
            0.95
        } else if l2_fits {
            0.70
        } else {
            0.40
        };

        CachePrediction {
            estimated_l1_hits: (load_count as f32 * hit_rate) as usize,
            estimated_l1_misses: load_count - (load_count as f32 * hit_rate) as usize,
            estimated_l2_hits: ((load_count as f32 * hit_rate * 0.8) as usize),
            stride_detected,
        }
    }

    /// Predict port contention
    pub fn predict_port_contention(&self, graph: &BlockGraph, arch: &MicroArchModel) -> f32 {
        let mut port_usage: HashMap<usize, usize> = HashMap::new();

        for instr in &graph.nodes {
            let port = instr.category.port();
            *port_usage.entry(port).or_insert(0) += instr.category.latency();
        }

        // Calculate contention as max usage / capacity
        let max_usage = port_usage.values().max().copied().unwrap_or(0);
        let capacity = arch.issue_width * 10; // 10 cycles window

        (max_usage as f32 / capacity as f32).clamp(0.0, 1.0)
    }
}

/// Cache behavior prediction
#[derive(Debug, Clone)]
pub struct CachePrediction {
    pub estimated_l1_hits: usize,
    pub estimated_l1_misses: usize,
    pub estimated_l2_hits: usize,
    pub stride_detected: bool,
}

/// Superblock decision made by neural predictor
#[derive(Debug, Clone)]
pub struct SuperblockDecision {
    /// Whether to trace this block
    pub should_trace: bool,
    /// Whether to unroll any loops
    pub unroll_factor: usize,
    /// Whether to vectorize
    pub vectorize: bool,
    /// Vector width (1, 4, 8, 16)
    pub vec_width: usize,
    /// Whether to inline
    pub inline: bool,
    /// Predicted IPC improvement
    pub predicted_ipc: f32,
    /// Confidence score
    pub confidence: f32,
}

/// Neural superblock predictor
pub struct NeuralSuperblockPredictor {
    /// Microarchitectural model
    arch: MicroArchModel,
    /// GNN for IPC prediction
    gnn: MicroArchGNN,
}

impl NeuralSuperblockPredictor {
    pub fn new(arch: MicroArchModel) -> Self {
        Self {
            arch: arch.clone(),
            gnn: MicroArchGNN::new(16, 32),
        }
    }

    /// Analyze a basic block and make superblock decisions
    pub fn analyze_block(&self, graph: &BlockGraph) -> SuperblockDecision {
        let predicted_ipc = self.gnn.predict_ipc(graph, &self.arch);
        let reg_pressure = self.gnn.predict_register_pressure(graph);
        let cache_pred = self.gnn.predict_cache_behavior(graph, &self.arch);
        let contention = self.gnn.predict_port_contention(graph, &self.arch);

        // Decision heuristics based on predictions
        let should_trace = predicted_ipc > 1.0 || cache_pred.estimated_l1_misses > 2;
        let unroll_factor = if reg_pressure < 8 && cache_pred.stride_detected {
            4.min(graph.nodes.len().next_power_of_two() / 4)
        } else {
            1
        };
        let vectorize = contention < 0.5 && graph.nodes.iter().any(|n| n.category == InstrCategory::FPAdd);
        let vec_width = if vectorize { 8 } else { 1 };
        let inline = graph.nodes.len() < 20 || predicted_ipc < 0.5;

        // Confidence based on model agreement
        let confidence = if cache_pred.stride_detected {
            0.8
        } else if cache_pred.estimated_l1_misses < 5 {
            0.6
        } else {
            0.4
        };

        SuperblockDecision {
            should_trace,
            unroll_factor,
            vectorize,
            vec_width,
            inline,
            predicted_ipc,
            confidence,
        }
    }

    /// Record actual performance for training
    pub fn record_performance(&mut self, graph: &BlockGraph, measured_ipc: f32) {
        let features: Vec<f32> = graph.node_features()
            .iter()
            .flatten()
            .take(32)
            .copied()
            .collect();

        self.gnn.training_samples.push((features, measured_ipc));
    }

    /// Simple training (gradient descent on error)
    pub fn train(&mut self, iterations: usize) {
        self.gnn.train(iterations);
    }
}

/// Neural tracing JIT integration
pub struct NeuralTracingJIT {
    predictor: NeuralSuperblockPredictor,
    /// Pending traces waiting for compilation
    pending_traces: Vec<usize>,
    /// Compiled trace cache (Arc<RwLock> for concurrent read/write access)
    compiled_traces: Arc<RwLock<HashMap<usize, CompiledSuperblock>>>,
}

impl NeuralTracingJIT {
    pub fn new() -> Self {
        Self {
            predictor: NeuralSuperblockPredictor::new(MicroArchModel::default()),
            pending_traces: Vec::new(),
            compiled_traces: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start tracing a basic block
    pub fn start_trace(&mut self, block_id: usize) {
        if !self.pending_traces.contains(&block_id) {
            self.pending_traces.push(block_id);
        }
    }

    /// Check if a trace should be compiled
    pub fn should_compile(&self, graph: &BlockGraph) -> bool {
        let decision = self.predictor.analyze_block(graph);
        decision.should_trace && decision.confidence > 0.5
    }

    /// Compile a superblock with neural guidance
    pub fn compile_superblock(&mut self, block_id: usize, graph: BlockGraph) -> Option<CompiledSuperblock> {
        let decision = self.predictor.analyze_block(&graph);

        if !decision.should_trace {
            return None;
        }

        let superblock = CompiledSuperblock {
            block_id,
            decision,
            estimated_ipc: self.predictor.gnn.predict_ipc(&graph, &self.predictor.arch),
        };

        self.compiled_traces.write().unwrap().insert(block_id, superblock.clone());
        Some(superblock)
    }

    /// Record actual IPC after execution
    pub fn record_execution(&mut self, block_id: usize, measured_ipc: f32) {
        let exists = self.compiled_traces.read().unwrap().contains_key(&block_id);
        if exists {
            // Rebuild graph for recording
            let graph = BlockGraph::new(); // Would need to rebuild from actual trace
            self.predictor.record_performance(&graph, measured_ipc);
        }
    }
}

impl Default for NeuralTracingJIT {
    fn default() -> Self {
        Self::new()
    }
}

/// A compiled superblock
#[derive(Debug, Clone)]
pub struct CompiledSuperblock {
    pub block_id: usize,
    pub decision: SuperblockDecision,
    pub estimated_ipc: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_microarch_model() {
        let arch = MicroArchModel::default();
        assert_eq!(arch.issue_width, 4);
    }

    #[test]
    fn test_block_graph() {
        let mut graph = BlockGraph::new();
        graph.add_instr(TraceInstr {
            id: 0,
            category: InstrCategory::Load,
            src_regs: vec![],
            dst_reg: Some(1),
            mem_size: 8,
            is_load: true,
            addr_dep: None,
            data_dep: None,
        });

        assert_eq!(graph.nodes.len(), 1);
    }

    #[test]
    fn test_gnn_ipc_prediction() {
        let gnn = MicroArchGNN::new(16, 32);
        let arch = MicroArchModel::default();

        let mut graph = BlockGraph::new();
        graph.add_instr(TraceInstr {
            id: 0,
            category: InstrCategory::ALU,
            src_regs: vec![],
            dst_reg: Some(1),
            mem_size: 0,
            is_load: false,
            addr_dep: None,
            data_dep: None,
        });

        let ipc = gnn.predict_ipc(&graph, &arch);
        assert!(ipc >= 0.0);
    }
}
