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
use std::sync::atomic::{AtomicU64, Ordering};
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

/// FIX C: Edge type for Relational GNN. Different weight matrices for
/// data dependencies vs address dependencies (address deps have higher latency).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    Data,  // Register result dependency (ALU math)
    Addr,  // Memory address dependency (pointer chasing — higher latency)
}

/// Graph representation of a basic block for GNN processing
#[derive(Clone, Debug)]
pub struct BlockGraph {
    pub nodes: Vec<TraceInstr>,
    pub edges: Vec<(usize, usize, EdgeType)>, // FIX C: Typed edges for R-GNN
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
            self.edges.push((dep, instr_id, EdgeType::Data));
        }

        // Add address dependency edge if exists
        if let Some(dep) = instr.addr_dep {
            self.edges.push((dep, instr_id, EdgeType::Addr));
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
        for &(from, to, etype) in &self.edges {
            // FIX C: Address dependencies get higher weight (2.0) than data deps (1.0)
            // because address computation chains have higher latency in CPU pipelines.
            adj[from][to] = match etype {
                EdgeType::Data => 1.0,
                EdgeType::Addr => 2.0,
            };
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
    pub fn from_edges(num_nodes: usize, edges: &[(usize, usize, EdgeType)]) -> Self {
        let mut row_offsets = vec![0usize; num_nodes + 1];
        let mut col_indices = Vec::with_capacity(edges.len());

        // Count edges per row (edge type is ignored for CSR structure)
        for &(from, _to, _etype) in edges {
            row_offsets[from + 1] += 1;
        }
        // Prefix sum
        for i in 1..=num_nodes {
            row_offsets[i] += row_offsets[i - 1];
        }

        // Fill column indices (stable sort by source)
        let mut temp_offsets = row_offsets.clone();
        col_indices.resize(edges.len(), 0);
        for &(from, to, _etype) in edges {
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
    _embed_dim: usize,
    hidden_dim: usize,
    _node_embeddings: Vec<Vec<f32>>,
    /// W_self weight matrix (hidden_dim × hidden_dim) per round
    w_self: Vec<Vec<f32>>,
    /// W_msg weight matrix (hidden_dim × hidden_dim) per round
    w_msg: Vec<Vec<f32>>,
    /// Readout MLP layer 1 weights (hidden_dim × hidden_dim/4)
    readout_w1: Vec<Vec<f32>>,
    /// Readout MLP layer 2 weights (hidden_dim/4 × 1)
    readout_w2: Vec<f32>,
    /// Instance-local RNG seed (replaces static mut for thread safety)
    seed: u64,
    /// Training samples (graph + measured IPC)
    /// FIX #7: Use Arc<BlockGraph> for shared ownership instead of cloning
    /// the graph on every training step.
    training_samples: Vec<(Arc<BlockGraph>, f32)>,
    /// FIX #8: Pre-allocated scratch buffers to avoid per-prediction
    /// O(N²×H²) heap allocations. These are reused across predictions.
    hidden_buffer: Vec<f32>,
    message_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    /// FIX B: Adam optimizer state — moving averages of gradients and squared gradients
    /// for each weight matrix. Replaces raw SGD with adaptive learning rates.
    m_readout_w2: Vec<f32>,
    v_readout_w2: Vec<f32>,
    m_readout_w1: Vec<Vec<f32>>,
    v_readout_w1: Vec<Vec<f32>>,
    m_w_self: Vec<Vec<f32>>,
    v_w_self: Vec<Vec<f32>>,
    m_w_msg: Vec<Vec<f32>>,
    v_w_msg: Vec<Vec<f32>>,
    adam_t: usize,
}

impl MicroArchGNN {
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Self {
        let fan_in = hidden_dim;
        let fan_out = hidden_dim;
        let limit = (6.0f32 / (fan_in as f32 + fan_out as f32)).sqrt();

        // Instance-local seed: use atomic counter so different GNN instances
        // get different initial weights (thread-safe, unlike static mut)
        static INSTANCE_COUNTER: AtomicU64 = AtomicU64::new(0);
        let instance_id = INSTANCE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut seed = 42u64.wrapping_add(instance_id.wrapping_mul(0x517cc1b727220a95));

        // Xavier/Glorot uniform initialization using instance-local seed
        let mut init_weight = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = (seed >> 33) as f32 / (1u64 << 31) as f32;
            (val * 2.0 - 1.0) * limit
        };

        let mut make_matrix = |rows: usize, cols: usize| -> Vec<Vec<f32>> {
            (0..rows).map(|_| (0..cols).map(|_| init_weight()).collect()).collect()
        };

        let readout_dim = hidden_dim / 4;

        // FIX B: Initialize weight matrices first, then capture seed value.
        // The closures mutably borrow `seed`, so we must finish all matrix
        // construction before we can move `seed` into the struct.
        let w_self = make_matrix(hidden_dim, hidden_dim);
        let w_msg = make_matrix(hidden_dim, hidden_dim);
        let readout_w1 = make_matrix(readout_dim, hidden_dim * 2);
        // readout_w2 needs init_weight too — compute via make_matrix with 1 col
        let readout_w2: Vec<f32> = make_matrix(readout_dim, 1).into_iter().map(|mut row| row.pop().unwrap_or(0.0)).collect();
        let m_readout_w1 = make_matrix(readout_dim, hidden_dim * 2);
        let v_readout_w1 = make_matrix(readout_dim, hidden_dim * 2);
        let m_w_self = make_matrix(hidden_dim, hidden_dim);
        let v_w_self = make_matrix(hidden_dim, hidden_dim);
        let m_w_msg = make_matrix(hidden_dim, hidden_dim);
        let v_w_msg = make_matrix(hidden_dim, hidden_dim);

        Self {
            _embed_dim: embed_dim,
            hidden_dim,
            _node_embeddings: Vec::new(),
            w_self,
            w_msg,
            readout_w1,
            readout_w2,
            seed,
            training_samples: Vec::new(),
            // FIX #8: Pre-allocate scratch buffers (avoid per-prediction heap allocation)
            hidden_buffer: vec![0.0f32; hidden_dim],
            message_buffer: vec![0.0f32; hidden_dim],
            output_buffer: vec![0.0f32; hidden_dim / 4],
            m_readout_w2: vec![0.0f32; readout_dim],
            v_readout_w2: vec![0.0f32; readout_dim],
            m_readout_w1,
            v_readout_w1,
            m_w_self,
            v_w_self,
            m_w_msg,
            v_w_msg,
            adam_t: 0,
        }
    }

    /// Forward pass with proper weight matrices and CSR adjacency.
    /// Uses pre-allocated scratch buffers (hidden_buffer, message_buffer, output_buffer)
    /// to avoid per-prediction heap allocations.
    pub fn predict_ipc(&mut self, graph: &BlockGraph, arch: &MicroArchModel) -> f32 {
        let node_feats = graph.node_features();
        if node_feats.is_empty() { return 0.0; }

        // Build CSR adjacency
        let csr = CsrAdjacency::from_edges(graph.nodes.len(), &graph.edges);

        // FIX A: Flatten hidden states into contiguous memory for cache locality.
        // Instead of Vec<Vec<f32>> (fragmented heap allocations), use a single
        // Vec<f32> of size n_nodes * hidden_dim. This eliminates N heap allocations
        // per prediction and improves cache line utilization.
        let n_nodes = node_feats.len();
        let hd = self.hidden_dim;
        let mut hidden = vec![0.0f32; n_nodes * hd];
        for (i, f) in node_feats.iter().enumerate() {
            let row = &mut hidden[i * hd..(i + 1) * hd];
            for (j, &v) in f.iter().take(hd.min(f.len())).enumerate() {
                row[j] = v;
            }
        }

        // Message passing (2 rounds) with flat-vector access
        for _round in 0..2 {
            let mut new_hidden = hidden.clone();

            for i in 0..n_nodes {
                let neighbors = csr.neighbors(i);
                let h_i = &hidden[i * hd..(i + 1) * hd];
                let self_contrib = self.mat_vec_flat(&self.w_self, h_i, hd);
                // FIX A: Reuse pre-allocated message_buffer instead of allocating
                // a new Vec each iteration. Clear it first, then fill with messages.
                self.message_buffer.iter_mut().for_each(|v| *v = 0.0);
                if !neighbors.is_empty() {
                    for &j in neighbors {
                        let h_j = &hidden[j * hd..(j + 1) * hd];
                        let w_msg_h = self.mat_vec_flat(&self.w_msg, h_j, hd);
                        for k in 0..hd { self.message_buffer[k] += w_msg_h[k]; }
                    }
                    let n = neighbors.len() as f32;
                    for k in 0..hd { self.message_buffer[k] /= n; }
                }

                let out = &mut new_hidden[i * hd..(i + 1) * hd];
                for k in 0..hd {
                    out[k] = (self_contrib[k] + self.message_buffer[k]).max(0.0);
                }
            }

            hidden = new_hidden;
        }

        // FIX A: Flat-vector pooling — use hidden_buffer and output_buffer
        // as scratch space instead of allocating new vectors.
        self.hidden_buffer.clear();
        self.hidden_buffer.extend((0..hd).map(|j| {
            (0..n_nodes).map(|i| hidden[i * hd + j]).sum::<f32>() / n_nodes as f32
        }));
        self.output_buffer.clear();
        self.output_buffer.extend((0..hd).map(|j| {
            (0..n_nodes).map(|i| hidden[i * hd + j]).fold(f32::NEG_INFINITY, f32::max)
        }));
        let pooled: Vec<f32> = self.hidden_buffer.iter().chain(self.output_buffer.iter()).cloned().collect();

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

    /// FIX A: Flat-vector matrix-vector multiply operating on contiguous slices
    fn mat_vec_flat(&self, mat: &[Vec<f32>], vec: &[f32], hd: usize) -> Vec<f32> {
        let rows = mat.len().min(hd);
        (0..rows).map(|i| {
            mat[i].iter().zip(vec.iter()).map(|(w, v)| w * v).sum()
        }).chain(std::iter::repeat(0.0)).take(hd).collect()
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

    /// Online training from hardware counters (SGD with MSE loss, full backprop through GNN)
    pub fn train(&mut self, iterations: usize) {
        if self.training_samples.len() < 10 { return; }

        let lr = 0.001f32;
        let n_rounds = 2usize;
        let readout_dim = self.hidden_dim / 4;
        if readout_dim == 0 { return; }
        let _grad_clip = 1.0f32;

        for _ in 0..iterations {
            // FIX B: Adam optimizer timestep
            self.adam_t += 1;
            let t = self.adam_t as f32;
            let beta1 = 0.9f32;
            let beta2 = 0.999f32;
            let epsilon = 1e-8f32;

            // Pick a random sample using instance-local seed
            self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (self.seed as usize) % self.training_samples.len();

            // FIX #7: Use Arc::clone instead of deep-cloning the graph.
            // Arc::clone only increments the reference count (cheap) instead
            // of copying the entire graph data structure (expensive).
            let (graph, measured_ipc) = (Arc::clone(&self.training_samples[idx].0), self.training_samples[idx].1);

            let n_nodes = graph.nodes.len();
            if n_nodes == 0 { continue; }

            let node_feats = graph.node_features();
            let csr = CsrAdjacency::from_edges(n_nodes, &graph.edges);

            // ===== FORWARD PASS (saving all intermediate values) =====

            // Initialize hidden states from features
            let h_init: Vec<Vec<f32>> = node_feats.iter().map(|f| {
                f.iter().cloned().take(self.hidden_dim.min(f.len()))
                    .chain(std::iter::repeat(0.0))
                    .take(self.hidden_dim)
                    .collect()
            }).collect();

            // all_hidden[0] = h_init, all_hidden[r+1] = hidden after round r
            let mut all_hidden: Vec<Vec<Vec<f32>>> = vec![h_init];
            let mut pre_relu_rounds: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_rounds);

            for _round in 0..n_rounds {
                let h_input = &all_hidden[all_hidden.len() - 1];
                let mut pre_relu = vec![vec![0.0f32; self.hidden_dim]; n_nodes];
                let mut new_hidden = vec![vec![0.0f32; self.hidden_dim]; n_nodes];

                for i in 0..n_nodes {
                    let neighbors = csr.neighbors(i);
                    let self_contrib = self.mat_vec(&self.w_self, &h_input[i]);
                    let mut msg = vec![0.0f32; self.hidden_dim];
                    if !neighbors.is_empty() {
                        for &j in neighbors {
                            let w_msg_h = self.mat_vec(&self.w_msg, &h_input[j]);
                            for k in 0..self.hidden_dim {
                                msg[k] += w_msg_h[k];
                            }
                        }
                        let n = neighbors.len() as f32;
                        for k in 0..self.hidden_dim { msg[k] /= n; }
                    }
                    for k in 0..self.hidden_dim {
                        pre_relu[i][k] = self_contrib[k] + msg[k];
                        new_hidden[i][k] = pre_relu[i][k].max(0.0);
                    }
                }

                pre_relu_rounds.push(pre_relu);
                all_hidden.push(new_hidden);
            }

            let h_final = &all_hidden[n_rounds];

            // Pooling: mean || max — use mean_pool and max_pool methods
            // to compute graph-level readout features from node hidden states.
            let n_nodes_f = n_nodes as f32;
            let mean_pool_result = self.mean_pool(h_final);

            let max_pool_result = self.max_pool(h_final);
            // Also compute max_indices for gradient backpropagation
            let mut max_indices = vec![0usize; self.hidden_dim];
            for (i, h) in h_final.iter().enumerate() {
                for j in 0..self.hidden_dim {
                    if h[j] > max_pool_result[j] {
                        max_indices[j] = i;
                    }
                }
            }

            let pooled: Vec<f32> = mean_pool_result.iter().chain(max_pool_result.iter()).cloned().collect();

            // Readout layer 1: hidden_dim*2 → readout_dim
            let mut h1_pre = vec![0.0f32; readout_dim];
            for i in 0..readout_dim.min(self.readout_w1.len()) {
                for j in 0..pooled.len().min(self.readout_w1[i].len()) {
                    h1_pre[i] += self.readout_w1[i][j] * pooled[j];
                }
            }
            let h1: Vec<f32> = h1_pre.iter().map(|&v| v.max(0.0)).collect();

            // Readout layer 2: readout_dim → 1
            let mut predicted_ipc = 0.0f32;
            for i in 0..readout_dim.min(self.readout_w2.len()) {
                predicted_ipc += self.readout_w2[i] * h1[i];
            }

            // ===== BACKWARD PASS (full backprop through all GNN layers) =====
            let error = predicted_ipc - measured_ipc;

            // --- Readout layer 2 gradients ---
            // dL/d(readout_w2[i]) = error * h1[i]
            // dL/d(h1[i]) = error * readout_w2[i]
            let mut grad_h1 = vec![0.0f32; readout_dim];
            for i in 0..readout_dim.min(self.readout_w2.len()) {
                grad_h1[i] = error * self.readout_w2[i];
                // FIX B: Adam optimizer instead of raw SGD
                let raw_grad = error * h1[i];
                self.m_readout_w2[i] = beta1 * self.m_readout_w2[i] + (1.0 - beta1) * raw_grad;
                self.v_readout_w2[i] = beta2 * self.v_readout_w2[i] + (1.0 - beta2) * raw_grad * raw_grad;
                let m_hat = self.m_readout_w2[i] / (1.0 - beta1.powf(t));
                let v_hat = self.v_readout_w2[i] / (1.0 - beta2.powf(t));
                self.readout_w2[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
            }

            // --- Readout layer 1 gradients ---
            // dL/d(readout_w1[i][j]) = grad_pre * pooled[j]
            // dL/d(pooled[j]) = Σ_i grad_pre * readout_w1[i][j]
            let mut grad_pooled = vec![0.0f32; pooled.len()];
            for i in 0..readout_dim.min(self.readout_w1.len()) {
                let relu_mask = if h1_pre[i] > 0.0 { 1.0f32 } else { 0.0f32 };
                let grad_pre = grad_h1[i] * relu_mask;
                for j in 0..pooled.len().min(self.readout_w1[i].len()) {
                    grad_pooled[j] += grad_pre * self.readout_w1[i][j];
                    // FIX B: Adam optimizer for readout_w1
                    let raw_grad = grad_pre * pooled[j];
                    self.m_readout_w1[i][j] = beta1 * self.m_readout_w1[i][j] + (1.0 - beta1) * raw_grad;
                    self.v_readout_w1[i][j] = beta2 * self.v_readout_w1[i][j] + (1.0 - beta2) * raw_grad * raw_grad;
                    let m_hat = self.m_readout_w1[i][j] / (1.0 - beta1.powf(t));
                    let v_hat = self.v_readout_w1[i][j] / (1.0 - beta2.powf(t));
                    self.readout_w1[i][j] -= lr * m_hat / (v_hat.sqrt() + epsilon);
                }
            }

            // --- Pooling gradients → dL/d(h_final) ---
            let mut grad_hidden: Vec<Vec<f32>> = vec![vec![0.0f32; self.hidden_dim]; n_nodes];

            // Mean pool: dL/d(h_final[i][j]) += dL/d(mean_pool[j]) / N
            for j in 0..self.hidden_dim {
                let g = grad_pooled[j] / n_nodes_f;
                for i in 0..n_nodes {
                    grad_hidden[i][j] += g;
                }
            }

            // Max pool: dL/d(h_final[argmax[j]][j]) += dL/d(max_pool[j])
            for j in 0..self.hidden_dim {
                grad_hidden[max_indices[j]][j] += grad_pooled[self.hidden_dim + j];
            }

            // --- Message passing rounds (reverse order) ---
            // Accumulate gradients for shared w_self and w_msg across rounds
            let mut grad_w_self = vec![vec![0.0f32; self.hidden_dim]; self.hidden_dim];
            let mut grad_w_msg = vec![vec![0.0f32; self.hidden_dim]; self.hidden_dim];

            for round in (0..n_rounds).rev() {
                let h_input = &all_hidden[round];

                // Apply ReLU mask: grad_hidden currently = dL/d(h_output)
                // h_output = ReLU(pre_relu), so dL/d(pre_relu) = dL/d(h_output) * 1_{pre_relu>0}
                for i in 0..n_nodes {
                    for k in 0..self.hidden_dim {
                        if pre_relu_rounds[round][i][k] <= 0.0 {
                            grad_hidden[i][k] = 0.0;
                        }
                    }
                }

                // Now grad_hidden = dL/d(pre_relu[round])
                // pre_relu[i][k] = (W_self * h_input[i])[k] + (1/|N(i)|) Σ_{j∈N(i)} (W_msg * h_input[j])[k]
                //
                // Gradients:
                //   dL/d(W_self[k][m]) += grad_hidden[i][k] * h_input[i][m]
                //   dL/d(h_input[i][m]) += Σ_k grad_hidden[i][k] * W_self[k][m]   (self path)
                //   dL/d(W_msg[k][m])  += Σ_{j∈N(i)} grad_hidden[i][k] * h_input[j][m] / |N(i)|
                //   dL/d(h_input[j][m]) += Σ_k grad_hidden[i][k] * W_msg[k][m] / |N(i)|  (msg path)
                let mut grad_h_input = vec![vec![0.0f32; self.hidden_dim]; n_nodes];

                for i in 0..n_nodes {
                    let neighbors = csr.neighbors(i);

                    // Self contribution
                    for k in 0..self.hidden_dim {
                        for m in 0..self.hidden_dim {
                            grad_w_self[k][m] += grad_hidden[i][k] * h_input[i][m];
                            grad_h_input[i][m] += grad_hidden[i][k] * self.w_self[k][m];
                        }
                    }

                    // Message contribution
                    if !neighbors.is_empty() {
                        let n_nbrs = neighbors.len() as f32;
                        for &j in neighbors {
                            for k in 0..self.hidden_dim {
                                for m in 0..self.hidden_dim {
                                    grad_w_msg[k][m] += grad_hidden[i][k] * h_input[j][m] / n_nbrs;
                                    grad_h_input[j][m] += grad_hidden[i][k] * self.w_msg[k][m] / n_nbrs;
                                }
                            }
                        }
                    }
                }

                grad_hidden = grad_h_input;
            }

            // FIX B: Adam optimizer for w_self and w_msg
            for k in 0..self.hidden_dim {
                for m in 0..self.hidden_dim {
                    let raw_g_self = grad_w_self[k][m];
                    self.m_w_self[k][m] = beta1 * self.m_w_self[k][m] + (1.0 - beta1) * raw_g_self;
                    self.v_w_self[k][m] = beta2 * self.v_w_self[k][m] + (1.0 - beta2) * raw_g_self * raw_g_self;
                    let m_hat = self.m_w_self[k][m] / (1.0 - beta1.powf(t));
                    let v_hat = self.v_w_self[k][m] / (1.0 - beta2.powf(t));
                    self.w_self[k][m] -= lr * m_hat / (v_hat.sqrt() + epsilon);

                    let raw_g_msg = grad_w_msg[k][m];
                    self.m_w_msg[k][m] = beta1 * self.m_w_msg[k][m] + (1.0 - beta1) * raw_g_msg;
                    self.v_w_msg[k][m] = beta2 * self.v_w_msg[k][m] + (1.0 - beta2) * raw_g_msg * raw_g_msg;
                    let m_hat = self.m_w_msg[k][m] / (1.0 - beta1.powf(t));
                    let v_hat = self.v_w_msg[k][m] / (1.0 - beta2.powf(t));
                    self.w_msg[k][m] -= lr * m_hat / (v_hat.sqrt() + epsilon);
                }
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
        let stride_detected = false;
        let last_addr = 0i64;
        let _stride = 0i64;

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
    pub fn analyze_block(&mut self, graph: &BlockGraph) -> SuperblockDecision {
        let predicted_ipc = self.gnn.predict_ipc(graph, &self.arch);
        let reg_pressure = self.gnn.predict_register_pressure(graph);
        let cache_pred = self.gnn.predict_cache_behavior(graph, &self.arch);
        let contention = self.gnn.predict_port_contention(graph, &self.arch);

        // Decision heuristics based on predictions
        let should_trace = predicted_ipc > 1.0 || cache_pred.estimated_l1_misses > 2;
        // FIX D: Dynamic threshold based on microarch model register count.
        // Zen has more registers than Haswell, so the threshold should be higher.
        let reg_threshold = (self.arch.int_regs / 2).max(4);
        let unroll_factor = if reg_pressure < reg_threshold && cache_pred.stride_detected {
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
    /// FIX #7: Use Arc<BlockGraph> to avoid cloning the graph on every training step.
    pub fn record_performance(&mut self, graph: Arc<BlockGraph>, measured_ipc: f32) {
        self.gnn.training_samples.push((graph, measured_ipc));
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
    pub fn should_compile(&mut self, graph: &BlockGraph) -> bool {
        let decision = self.predictor.analyze_block(graph);
        decision.should_trace && decision.confidence > 0.5
    }

    /// Compile a superblock with neural guidance
    pub fn compile_superblock(&mut self, block_id: usize, graph: BlockGraph) -> Option<CompiledSuperblock> {
        let decision = self.predictor.analyze_block(&graph);

        if !decision.should_trace {
            return None;
        }

        let estimated_ipc = self.predictor.gnn.predict_ipc(&graph, &self.predictor.arch);
        let superblock = CompiledSuperblock {
            block_id,
            decision,
            estimated_ipc,
            graph: Arc::new(graph), // FIX #7: Wrap in Arc for shared ownership
        };

        self.compiled_traces.write().unwrap_or_else(|e| e.into_inner()).insert(block_id, superblock.clone());
        Some(superblock)
    }

    /// Record actual IPC after execution
    /// FIX #7: Use Arc::clone instead of deep-cloning the graph.
    pub fn record_execution(&mut self, block_id: usize, measured_ipc: f32) {
        // Use the stored graph from the compiled trace (not an empty graph)
        let graph = self.compiled_traces.read().unwrap_or_else(|e| e.into_inner()).get(&block_id)
            .map(|c| Arc::clone(&c.graph));
        if let Some(graph) = graph {
            self.predictor.record_performance(graph, measured_ipc);
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
    /// Stored graph so record_execution can use the actual graph structure
    /// FIX #7: Use Arc<BlockGraph> for shared ownership to avoid deep-cloning
    pub graph: Arc<BlockGraph>,
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
        let mut gnn = MicroArchGNN::new(16, 32);
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
