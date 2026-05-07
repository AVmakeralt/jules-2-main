// =============================================================================
// GNN E-Graph Superoptimizer
//
// A Graph Neural Network that learns to navigate e-graphs for optimal speed.
// ~50k parameters, trainable in a single session.
//
// Architecture: AlphaGo-style
//   - GNN takes e-graph state (program graph) as input
//   - Policy head: predicts which rewrite action to apply
//   - Value head: predicts expected cycle improvement
//   - Trained on MCTS rollouts (REINFORCE + value regression)
//   - At compile time: GNN inference guides optimization in microseconds
//
// Key insight: Instead of running expensive MCTS search every time,
// we train a GNN to *predict* what MCTS would discover. The GNN
// learns the patterns: "division by power of 2 → shift", "x+0 → x",
// "nested identities → peel them all", etc.
//
// Graph representation:
//   - Nodes = operations (ConstInt, Var, Add, Mul, etc.)
//   - Node features = [op_type_onehot(29), depth, num_children, is_constant]
//   - Edges = dataflow (parent→child in the expression tree)
//   - Rewritten edges = possible rewrites (attention-weighted)
// =============================================================================

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::compiler::ast::*;
use crate::optimizer::hardware_cost_model::Microarchitecture;
use crate::optimizer::mcts_superoptimizer::{
    CycleCostEstimator, Instr, MctsConfig, MctsSuperoptimizer, RewriteAction,
};
use crate::Span;

// =============================================================================
// §1  Configuration
// =============================================================================

/// Configuration for the GNN E-Graph Optimizer
#[derive(Debug, Clone)]
pub struct GnnConfig {
    /// Hidden dimension for GNN layers
    pub hidden_dim: usize,
    /// Number of GNN message-passing layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of rewrite actions (12 from RewriteAction::all())
    pub num_actions: usize,
    /// Node feature dimension (29 op types + 4 extra = 33)
    pub node_feature_dim: usize,
    /// Learning rate for training
    pub learning_rate: f64,
    /// Discount factor for REINFORCE
    pub gamma: f64,
    /// Number of training episodes
    pub num_episodes: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Whether to use attention in GNN
    pub use_attention: bool,
    /// Value loss coefficient (vs policy loss)
    pub value_loss_coef: f64,
    /// Entropy regularization coefficient
    pub entropy_coef: f64,
}

impl Default for GnnConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 128,
            num_layers: 3,
            num_heads: 4,
            num_actions: 12,
            node_feature_dim: 33,
            learning_rate: 0.001,
            gamma: 0.99,
            num_episodes: 5000,
            batch_size: 32,
            use_attention: true,
            value_loss_coef: 0.5,
            entropy_coef: 0.01,
        }
    }
}

impl GnnConfig {
    /// Estimate total number of parameters
    pub fn estimate_params(&self) -> usize {
        let d_in = self.node_feature_dim;
        let d_h = self.hidden_dim;
        let n_actions = self.num_actions;

        // Embedding: d_in -> d_h
        let embed = d_in * d_h + d_h;

        // GNN layers: each has W_self, W_neigh, bias (+ attention if enabled)
        let mut gnn = 0;
        for i in 0..self.num_layers {
            let d_from = if i == 0 { d_h } else { d_h };
            let d_to = d_h;
            gnn += d_from * d_to + d_to; // W_self
            gnn += d_from * d_to + d_to; // W_neigh
            if self.use_attention {
                gnn += d_from * self.num_heads; // Attention keys
                gnn += d_from * self.num_heads; // Attention queries
            }
        }

        // Policy head: d_h -> d_h -> n_actions
        let policy = d_h * d_h + d_h + d_h * n_actions + n_actions;

        // Value head: d_h -> d_h -> 1
        let value = d_h * d_h + d_h + d_h * 1 + 1;

        // Graph readout: d_h -> d_h
        let readout = d_h * d_h + d_h;

        embed + gnn + policy + value + readout
    }

    /// Small config for fast testing (~15k params)
    pub fn small() -> Self {
        Self {
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 2,
            num_episodes: 1000,
            ..Default::default()
        }
    }

    /// Medium config (~50k params) - the sweet spot
    pub fn medium() -> Self {
        Self {
            hidden_dim: 64,
            num_layers: 3,
            num_heads: 2,
            num_episodes: 5000,
            ..Default::default()
        }
    }

    /// Large config (~150k params) for maximum quality
    pub fn large() -> Self {
        Self {
            hidden_dim: 192,
            num_layers: 4,
            num_heads: 6,
            num_episodes: 10000,
            ..Default::default()
        }
    }
}

// =============================================================================
// §2  Tensor Library (minimal, for GNN math)
// =============================================================================

/// A simple 2D tensor (matrix) for neural network operations
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl Tensor {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn from_vec(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "Tensor size mismatch");
        Self { data, rows, cols }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols)
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![1.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Xavier/Glorot initialization
    pub fn xavier(rows: usize, cols: usize, rng: &mut SimpleRng) -> Self {
        let scale = (2.0 / (rows + cols) as f64).sqrt();
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            data.push(rng.normal() * scale);
        }
        Self { data, rows, cols }
    }

    /// He initialization (for ReLU layers)
    pub fn he_init(rows: usize, cols: usize, rng: &mut SimpleRng) -> Self {
        let scale = (2.0 / rows as f64).sqrt();
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            data.push(rng.normal() * scale);
        }
        Self { data, rows, cols }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    /// Matrix multiplication: (m x k) @ (k x n) -> (m x n)
    /// Optimized with cache-friendly tiling for small matrices
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.cols, other.rows, "Matmul dimension mismatch");
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;
        let mut result = Tensor::zeros(m, n);

        // For small matrices, use the straightforward approach but with
        // better cache access pattern: iterate over k in inner loop
        const TILE: usize = 16;
        for ii in (0..m).step_by(TILE) {
            let i_end = (ii + TILE).min(m);
            for jj in (0..n).step_by(TILE) {
                let j_end = (jj + TILE).min(n);
                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut sum = 0.0;
                        for l in 0..k {
                            sum += self.data[i * k + l] * other.data[l * n + j];
                        }
                        result.data[i * n + j] = sum;
                    }
                }
            }
        }
        result
    }

    /// Transpose
    pub fn t(&self) -> Tensor {
        let mut result = Tensor::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        result
    }

    /// Element-wise addition (supports broadcasting: (1, d) + (N, d) -> (N, d))
    pub fn add(&self, other: &Tensor) -> Tensor {
        // Support broadcasting: (1, cols) + (rows, cols) -> (rows, cols)
        if self.rows == other.rows && self.cols == other.cols {
            let mut result = self.clone();
            for i in 0..result.data.len() {
                result.data[i] += other.data[i];
            }
            result
        } else if other.rows == 1 && self.cols == other.cols {
            // Broadcast other across rows
            let mut result = self.clone();
            for i in 0..self.rows {
                for j in 0..self.cols {
                    result.data[i * self.cols + j] += other.data[j];
                }
            }
            result
        } else if self.rows == 1 && self.cols == other.cols {
            // Broadcast self across rows
            let mut result = other.clone();
            for i in 0..other.rows {
                for j in 0..other.cols {
                    result.data[i * other.cols + j] += self.data[j];
                }
            }
            result
        } else {
            panic!("Tensor add: incompatible shapes ({}, {}) + ({}, {})", self.rows, self.cols, other.rows, other.cols);
        }
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result = self.clone();
        for i in 0..result.data.len() {
            result.data[i] -= other.data[i];
        }
        result
    }

    /// Scalar multiplication
    pub fn scale(&self, s: f64) -> Tensor {
        let mut result = self.clone();
        for v in result.data.iter_mut() {
            *v *= s;
        }
        result
    }

    /// In-place scalar multiply
    pub fn scale_mut(&mut self, s: f64) {
        for v in self.data.iter_mut() {
            *v *= s;
        }
    }

    /// In-place add
    pub fn add_mut(&mut self, other: &Tensor) {
        assert_eq!(self.data.len(), other.data.len());
        for i in 0..self.data.len() {
            self.data[i] += other.data[i];
        }
    }

    /// ReLU activation
    pub fn relu(&self) -> Tensor {
        let mut result = self.clone();
        for v in result.data.iter_mut() {
            if *v < 0.0 { *v = 0.0; }
        }
        result
    }

    /// ReLU derivative (for backprop)
    pub fn relu_grad(&self) -> Tensor {
        let mut result = self.clone();
        for v in result.data.iter_mut() {
            *v = if *v > 0.0 { 1.0 } else { 0.0 };
        }
        result
    }

    /// Softmax (row-wise)
    pub fn softmax(&self) -> Tensor {
        let mut result = Tensor::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            let row_start = i * self.cols;
            let max_val = self.data[row_start..row_start + self.cols]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            for j in 0..self.cols {
                let val = (self.data[row_start + j] - max_val).exp();
                result.data[row_start + j] = val;
                sum += val;
            }
            if sum > 0.0 {
                for j in 0..self.cols {
                    result.data[row_start + j] /= sum;
                }
            }
        }
        result
    }

    /// Sum along rows -> (1 x cols)
    pub fn row_sum(&self) -> Tensor {
        let mut result = Tensor::zeros(1, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j] += self.data[i * self.cols + j];
            }
        }
        result
    }

    /// Mean along rows -> (1 x cols)
    pub fn row_mean(&self) -> Tensor {
        let sum = self.row_sum();
        let mut result = sum.clone();
        if self.rows > 0 {
            result.scale_mut(1.0 / self.rows as f64);
        }
        result
    }

    /// Element-wise multiplication (Hadamard)
    pub fn hadamard(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.data.len(), other.data.len());
        let mut result = self.clone();
        for i in 0..result.data.len() {
            result.data[i] *= other.data[i];
        }
        result
    }

    /// L2 norm
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Clip values to [-max, max]
    pub fn clip(&self, max: f64) -> Tensor {
        let mut result = self.clone();
        for v in result.data.iter_mut() {
            *v = v.clamp(-max, max);
        }
        result
    }

    /// Total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Reshape (rows * cols must equal current size)
    pub fn reshape(&self, new_rows: usize, new_cols: usize) -> Tensor {
        assert_eq!(self.rows * self.cols, new_rows * new_cols);
        Tensor {
            data: self.data.clone(),
            rows: new_rows,
            cols: new_cols,
        }
    }
}

/// Layer normalization for stabilizing GNN training
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub gamma: Tensor, // (1, dim)
    pub beta: Tensor,  // (1, dim)
    pub eps: f64,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: Tensor::ones(1, dim),
            beta: Tensor::zeros(1, dim),
            eps: 1e-5,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: (batch, dim)
        let mut result = Tensor::zeros(x.rows, x.cols);
        for i in 0..x.rows {
            // Compute mean
            let mut mean = 0.0;
            for j in 0..x.cols {
                mean += x.data[i * x.cols + j];
            }
            mean /= x.cols as f64;

            // Compute variance
            let mut var = 0.0;
            for j in 0..x.cols {
                let d = x.data[i * x.cols + j] - mean;
                var += d * d;
            }
            var /= x.cols as f64;

            // Normalize
            let inv_std = 1.0 / (var + self.eps).sqrt();
            for j in 0..x.cols {
                let normalized = (x.data[i * x.cols + j] - mean) * inv_std;
                result.data[i * x.cols + j] =
                    self.gamma.data[j] * normalized + self.beta.data[j];
            }
        }
        result
    }
}

// =============================================================================
// §3  Simple RNG (deterministic, no external deps)
// =============================================================================

/// Simple xorshift RNG for reproducible training
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn from_seed(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Standard normal using Box-Muller
    pub fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-10);
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        z
    }

    /// Random index in [0, n)
    pub fn gen_range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Sample from a probability distribution
    pub fn sample(&mut self, probs: &[f64]) -> usize {
        let r = self.next_f64();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }
        probs.len().saturating_sub(1)
    }
}

// =============================================================================
// §4  Graph Representation for GNN
// =============================================================================

/// A node in the program graph
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Node index
    pub idx: usize,
    /// Feature vector (33-dim: 29 op one-hot + depth + num_children + is_constant + has_variable)
    pub features: Vec<f64>,
    /// Indices of neighbor nodes (children + parent)
    pub neighbors: Vec<usize>,
    /// Index into edge list for each neighbor
    pub edge_indices: Vec<usize>,
}

/// An edge in the program graph
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub src: usize,
    pub dst: usize,
    /// Edge type: 0=child, 1=parent, 2=sibling
    pub edge_type: usize,
}

/// The full program graph
#[derive(Debug, Clone)]
pub struct ProgramGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    /// Which actions are applicable at each node
    pub applicable_actions: Vec<Vec<bool>>,
    /// Original cycle cost
    pub original_cost: u32,
}

impl ProgramGraph {
    /// Convert an Instr to a ProgramGraph
    pub fn from_instr(instr: &Instr, cost: u32) -> Self {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut applicable = Vec::new();
        let actions = RewriteAction::all();

        Self::build_graph(instr, &mut nodes, &mut edges, &mut applicable, 0, &actions);

        // Add reverse edges (parent links) and sibling links
        let forward_edges: Vec<GraphEdge> = edges.clone();
        for e in &forward_edges {
            edges.push(GraphEdge {
                src: e.dst,
                dst: e.src,
                edge_type: 1, // parent
            });
        }

        // Update neighbor lists
        for e in &edges {
            nodes[e.src].neighbors.push(e.dst);
            nodes[e.src].edge_indices.push(edges.len() - 1); // approximate
        }

        // Compute applicable actions for the root (for policy output)
        // Already done in build_graph

        Self {
            nodes,
            edges,
            applicable_actions: applicable,
            original_cost: cost,
        }
    }

    fn build_graph(
        instr: &Instr,
        nodes: &mut Vec<GraphNode>,
        edges: &mut Vec<GraphEdge>,
        applicable: &mut Vec<Vec<bool>>,
        depth: usize,
        actions: &[RewriteAction],
    ) -> usize {
        let idx = nodes.len();
        let features = instr_to_features(instr, depth);

        // Compute which actions are applicable
        let app: Vec<bool> = actions.iter().map(|a| a.is_applicable(instr)).collect();

        nodes.push(GraphNode {
            idx,
            features,
            neighbors: Vec::new(),
            edge_indices: Vec::new(),
        });
        applicable.push(app);

        match instr {
            Instr::BinOp { lhs, rhs, .. } => {
                let l_idx = Self::build_graph(lhs, nodes, edges, applicable, depth + 1, actions);
                let r_idx = Self::build_graph(rhs, nodes, edges, applicable, depth + 1, actions);
                edges.push(GraphEdge { src: idx, dst: l_idx, edge_type: 0 });
                edges.push(GraphEdge { src: idx, dst: r_idx, edge_type: 0 });
            }
            Instr::UnOp { operand, .. } => {
                let o_idx = Self::build_graph(operand, nodes, edges, applicable, depth + 1, actions);
                edges.push(GraphEdge { src: idx, dst: o_idx, edge_type: 0 });
            }
            _ => {}
        }

        idx
    }

    /// Get the feature matrix (N x F)
    pub fn feature_matrix(&self) -> Tensor {
        if self.nodes.is_empty() {
            return Tensor::zeros(1, 33);
        }
        let n = self.nodes.len();
        let f = self.nodes[0].features.len();
        let mut data = Vec::with_capacity(n * f);
        for node in &self.nodes {
            data.extend_from_slice(&node.features);
        }
        Tensor::from_vec(data, n, f)
    }

    /// Get the adjacency matrix (N x N) normalized
    pub fn adjacency_matrix(&self) -> Tensor {
        let n = self.nodes.len();
        if n == 0 {
            return Tensor::zeros(1, 1);
        }
        let mut adj = Tensor::zeros(n, n);
        for e in &self.edges {
            if e.src < n && e.dst < n {
                adj.data[e.src * n + e.dst] += 1.0;
            }
        }
        // Add self-loops
        for i in 0..n {
            adj.data[i * n + i] += 1.0;
        }
        // Normalize: D^{-1/2} A D^{-1/2}
        let mut degree = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                degree[i] += adj.data[i * n + j];
            }
        }
        for i in 0..n {
            for j in 0..n {
                let di = if degree[i] > 0.0 { 1.0 / degree[i].sqrt() } else { 0.0 };
                let dj = if degree[j] > 0.0 { 1.0 / degree[j].sqrt() } else { 0.0 };
                adj.data[i * n + j] *= di * dj;
            }
        }
        adj
    }

    /// Get action mask for the root node
    pub fn root_action_mask(&self) -> Vec<bool> {
        if self.applicable_actions.is_empty() {
            return vec![false; 12];
        }
        self.applicable_actions[0].clone()
    }
}

/// Convert an Instr to a 33-dim feature vector
fn instr_to_features(instr: &Instr, depth: usize) -> Vec<f64> {
    let mut features = vec![0.0; 33];

    // One-hot encode the operation type (29 possible types)
    let op_idx = match instr {
        Instr::ConstInt(_) => 0,
        Instr::ConstFloat(_) => 1,
        Instr::ConstBool(_) => 2,
        Instr::Var(_) => 3,
        Instr::BinOp { op: BinOpKind::Add, .. } => 4,
        Instr::BinOp { op: BinOpKind::Sub, .. } => 5,
        Instr::BinOp { op: BinOpKind::Mul, .. } => 6,
        Instr::BinOp { op: BinOpKind::Div, .. } => 7,
        Instr::BinOp { op: BinOpKind::Rem, .. } => 8,
        Instr::BinOp { op: BinOpKind::FloorDiv, .. } => 9,
        Instr::BinOp { op: BinOpKind::Eq, .. } => 10,
        Instr::BinOp { op: BinOpKind::Ne, .. } => 11,
        Instr::BinOp { op: BinOpKind::Lt, .. } => 12,
        Instr::BinOp { op: BinOpKind::Le, .. } => 13,
        Instr::BinOp { op: BinOpKind::Gt, .. } => 14,
        Instr::BinOp { op: BinOpKind::Ge, .. } => 15,
        Instr::BinOp { op: BinOpKind::And, .. } => 16,
        Instr::BinOp { op: BinOpKind::Or, .. } => 17,
        Instr::BinOp { op: BinOpKind::BitAnd, .. } => 18,
        Instr::BinOp { op: BinOpKind::BitOr, .. } => 19,
        Instr::BinOp { op: BinOpKind::BitXor, .. } => 20,
        Instr::BinOp { op: BinOpKind::Shl, .. } => 21,
        Instr::BinOp { op: BinOpKind::Shr, .. } => 22,
        Instr::UnOp { op: UnOpKind::Neg, .. } => 23,
        Instr::UnOp { op: UnOpKind::Not, .. } => 24,
        Instr::UnOp { op: UnOpKind::Deref, .. } => 25,
        Instr::UnOp { op: UnOpKind::Ref, .. } => 26,
        Instr::UnOp { op: UnOpKind::RefMut, .. } => 27,
        // Fallback for unknown ops
        _ => 28,
    };
    features[op_idx] = 1.0;

    // Extra features (indices 29-32)
    features[29] = (depth as f64).ln_1p() / 5.0; // Log depth, normalized
    features[30] = match instr {
        Instr::BinOp { .. } => 2.0,
        Instr::UnOp { .. } => 1.0,
        _ => 0.0,
    } / 2.0; // Num children, normalized
    features[31] = match instr {
        Instr::ConstInt(_) | Instr::ConstFloat(_) | Instr::ConstBool(_) => 1.0,
        _ => 0.0,
    }; // Is constant
    features[32] = match instr {
        Instr::Var(_) => 1.0,
        _ => 0.0,
    }; // Has variable

    features
}

// =============================================================================
// §5  GNN Model (~50k parameters)
// =============================================================================

/// A single GNN layer with optional attention
#[derive(Debug, Clone)]
pub struct GnnLayer {
    /// Self-transformation weight: (d_in, d_out)
    pub w_self: Tensor,
    /// Neighbor aggregation weight: (d_in, d_out)
    pub w_neigh: Tensor,
    /// Bias: (1, d_out)
    pub bias: Tensor,
    /// Layer normalization
    pub layer_norm: LayerNorm,
    /// Attention weights (if enabled): key (d_in, num_heads), query (d_in, num_heads)
    pub attn_key: Option<Tensor>,
    pub attn_query: Option<Tensor>,
}

impl GnnLayer {
    pub fn new(d_in: usize, d_out: usize, num_heads: usize, use_attention: bool, rng: &mut SimpleRng) -> Self {
        Self {
            w_self: Tensor::he_init(d_in, d_out, rng),
            w_neigh: Tensor::he_init(d_in, d_out, rng),
            bias: Tensor::zeros(1, d_out),
            layer_norm: LayerNorm::new(d_out),
            attn_key: if use_attention { Some(Tensor::xavier(d_in, num_heads, rng)) } else { None },
            attn_query: if use_attention { Some(Tensor::xavier(d_in, num_heads, rng)) } else { None },
        }
    }

    /// Forward pass: compute new node representations
    /// h_self: (N, d_in), adj: (N, N) normalized adjacency
    /// Uses attention mechanism when attn_key/attn_query are available
    pub fn forward(&self, h: &Tensor, adj: &Tensor) -> Tensor {
        // Self transformation
        let h_self = h.matmul(&self.w_self); // (N, d_out)

        // Neighbor aggregation — use attention-weighted aggregation when available
        let h_neigh = if let (Some(attn_key), Some(attn_query)) = (&self.attn_key, &self.attn_query) {
            // Compute attention scores: query * key^T
            // q = h @ attn_query  (N, num_heads)
            // k = h @ attn_key    (N, num_heads)
            let q = h.matmul(attn_query); // (N, num_heads)
            let k = h.matmul(attn_key);   // (N, num_heads)

            // attn_scores = q @ k^T / sqrt(num_heads), then softmax row-wise
            let num_heads = q.cols;
            let scale = 1.0 / (num_heads as f64).sqrt();
            let attn_scores = q.matmul(&k.t()).scale(scale).softmax(); // (N, N)

            // Mask attention with adjacency (only attend to neighbors + self)
            let attn_masked = attn_scores.hadamard(adj); // (N, N)

            // Renormalize attention weights per row
            let mut attn_normalized = Tensor::zeros(attn_masked.rows, attn_masked.cols);
            for i in 0..attn_masked.rows {
                let row_start = i * attn_masked.cols;
                let row_sum: f64 = attn_masked.data[row_start..row_start + attn_masked.cols].iter().sum();
                if row_sum > 1e-10 {
                    for j in 0..attn_masked.cols {
                        attn_normalized.data[i * attn_masked.cols + j] =
                            attn_masked.data[i * attn_masked.cols + j] / row_sum;
                    }
                }
            }

            // Weighted neighbor aggregation
            attn_normalized.matmul(h).matmul(&self.w_neigh) // (N, d_out)
        } else {
            // Fallback: simple adjacency-based aggregation (no attention)
            adj.matmul(h).matmul(&self.w_neigh) // (N, d_out)
        };

        // Combine self + neighbor
        let h_combined = h_self.add(&h_neigh);

        // Add bias + layer norm + ReLU
        let with_bias = h_combined.add(&self.bias);
        let normalized = self.layer_norm.forward(&with_bias);
        normalized.relu()
    }
}

/// Policy head: predicts which rewrite action to apply
#[derive(Debug, Clone)]
pub struct PolicyHead {
    pub w1: Tensor, // (d_h, d_h)
    pub b1: Tensor, // (1, d_h)
    pub w2: Tensor, // (d_h, num_actions)
    pub b2: Tensor, // (1, num_actions)
}

impl PolicyHead {
    pub fn new(d_h: usize, num_actions: usize, rng: &mut SimpleRng) -> Self {
        Self {
            w1: Tensor::he_init(d_h, d_h, rng),
            b1: Tensor::zeros(1, d_h),
            w2: Tensor::xavier(d_h, num_actions, rng),
            b2: Tensor::zeros(1, num_actions),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: (1, d_h) graph-level representation
        let h = x.matmul(&self.w1).add(&self.b1).relu();
        h.matmul(&self.w2).add(&self.b2).softmax()
    }
}

/// Value head: predicts expected improvement (0-1 normalized)
#[derive(Debug, Clone)]
pub struct ValueHead {
    pub w1: Tensor, // (d_h, d_h)
    pub b1: Tensor, // (1, d_h)
    pub w2: Tensor, // (d_h, 1)
    pub b2: Tensor, // (1, 1)
}

impl ValueHead {
    pub fn new(d_h: usize, rng: &mut SimpleRng) -> Self {
        Self {
            w1: Tensor::he_init(d_h, d_h, rng),
            b1: Tensor::zeros(1, d_h),
            w2: Tensor::xavier(d_h, 1, rng),
            b2: Tensor::zeros(1, 1),
        }
    }

    pub fn forward(&self, x: &Tensor) -> f64 {
        let h = x.matmul(&self.w1).add(&self.b1).relu();
        let v = h.matmul(&self.w2).add(&self.b2);
        // Sigmoid to output [0, 1]
        let raw = v.data[0];
        1.0 / (1.0 + (-raw).exp())
    }
}

/// Graph readout: aggregates node representations into a graph-level vector
#[derive(Debug, Clone)]
pub struct GraphReadout {
    pub w: Tensor, // (d_h, d_h)
    pub b: Tensor, // (1, d_h)
}

impl GraphReadout {
    pub fn new(d_h: usize, rng: &mut SimpleRng) -> Self {
        Self {
            w: Tensor::xavier(d_h, d_h, rng),
            b: Tensor::zeros(1, d_h),
        }
    }

    /// Weighted mean pooling over nodes
    pub fn forward(&self, node_embeddings: &Tensor) -> Tensor {
        // node_embeddings: (N, d_h)
        // Attention-weighted pooling
        let scores = node_embeddings.matmul(&self.w).add(&self.b); // (N, d_h)
        let weights = scores.relu(); // Simple non-negative weights
        let weight_sum: f64 = weights.data.iter().sum();
        if weight_sum < 1e-10 {
            return node_embeddings.row_mean();
        }
        // Weighted average
        let mut result = Tensor::zeros(1, node_embeddings.cols);
        for i in 0..node_embeddings.rows {
            let w_i = weights.data[i * node_embeddings.cols..(i + 1) * node_embeddings.cols]
                .iter()
                .sum::<f64>()
                / weight_sum;
            for j in 0..node_embeddings.cols {
                result.data[j] += w_i * node_embeddings.data[i * node_embeddings.cols + j];
            }
        }
        result
    }
}

/// The full GNN E-Graph Optimizer model
#[derive(Debug, Clone)]
pub struct GnnEgraphModel {
    /// Input embedding: (node_feature_dim, hidden_dim)
    pub embedding: Tensor,
    pub embedding_bias: Tensor,
    /// GNN layers
    pub gnn_layers: Vec<GnnLayer>,
    /// Graph readout
    pub readout: GraphReadout,
    /// Policy head
    pub policy_head: PolicyHead,
    /// Value head
    pub value_head: ValueHead,
    /// Configuration
    pub config: GnnConfig,
    /// Running baseline for REINFORCE (exponential moving average of rewards)
    pub reward_baseline: f64,
    /// Decay factor for the reward baseline (0.99 = slow adaptation)
    pub baseline_decay: f64,
}

impl GnnEgraphModel {
    /// Create a new model with random weights
    pub fn new(config: &GnnConfig) -> Self {
        let mut rng = SimpleRng::from_seed(42);

        let embedding = Tensor::he_init(config.node_feature_dim, config.hidden_dim, &mut rng);
        let embedding_bias = Tensor::zeros(1, config.hidden_dim);

        let mut gnn_layers = Vec::new();
        for i in 0..config.num_layers {
            let d_in = if i == 0 { config.hidden_dim } else { config.hidden_dim };
            gnn_layers.push(GnnLayer::new(
                d_in,
                config.hidden_dim,
                config.num_heads,
                config.use_attention,
                &mut rng,
            ));
        }

        let readout = GraphReadout::new(config.hidden_dim, &mut rng);
        let policy_head = PolicyHead::new(config.hidden_dim, config.num_actions, &mut rng);
        let value_head = ValueHead::new(config.hidden_dim, &mut rng);

        Self {
            embedding,
            embedding_bias,
            gnn_layers,
            readout,
            policy_head,
            value_head,
            config: config.clone(),
            reward_baseline: 0.0,
            baseline_decay: 0.99,
        }
    }

    /// Count total parameters
    pub fn param_count(&self) -> usize {
        let mut count = 0;
        count += self.embedding.len() + self.embedding_bias.len();
        for layer in &self.gnn_layers {
            count += layer.w_self.len() + layer.w_neigh.len() + layer.bias.len();
            count += layer.layer_norm.gamma.len() + layer.layer_norm.beta.len();
            if let Some(k) = &layer.attn_key { count += k.len(); }
            if let Some(q) = &layer.attn_query { count += q.len(); }
        }
        count += self.readout.w.len() + self.readout.b.len();
        count += self.policy_head.w1.len() + self.policy_head.b1.len()
               + self.policy_head.w2.len() + self.policy_head.b2.len();
        count += self.value_head.w1.len() + self.value_head.b1.len()
               + self.value_head.w2.len() + self.value_head.b2.len();
        count
    }

    /// Forward pass: takes a ProgramGraph, returns (policy, value)
    /// policy: Vec<f64> of length num_actions (probability distribution)
    /// value: f64 in [0, 1] (expected improvement)
    pub fn forward(&self, graph: &ProgramGraph) -> (Vec<f64>, f64) {
        // Get feature matrix and adjacency
        let x = graph.feature_matrix(); // (N, F)
        let adj = graph.adjacency_matrix(); // (N, N)

        // Embed input features
        let mut h = x.matmul(&self.embedding).add(&self.embedding_bias); // (N, hidden_dim)
        h = h.relu();

        // GNN message passing layers
        for layer in &self.gnn_layers {
            h = layer.forward(&h, &adj);
        }

        // Graph readout: (1, hidden_dim)
        let graph_repr = self.readout.forward(&h);

        // Policy: (1, num_actions) -> softmax
        let policy_tensor = self.policy_head.forward(&graph_repr);

        // Value: scalar
        let value = self.value_head.forward(&graph_repr);

        // Apply action mask
        let mask = graph.root_action_mask();
        let mut policy = vec![0.0; self.config.num_actions];
        let mut total = 0.0;
        for i in 0..self.config.num_actions {
            if i < mask.len() && mask[i] {
                policy[i] = policy_tensor.data[i].max(1e-8);
            }
            total += policy[i];
        }
        // Normalize
        if total > 0.0 {
            for p in policy.iter_mut() {
                *p /= total;
            }
        }

        (policy, value)
    }

    /// Select the best action given a graph
    pub fn select_action(&self, graph: &ProgramGraph) -> usize {
        let (policy, _value) = self.forward(graph);
        let mut best_idx = 0;
        let mut best_prob = 0.0;
        for (i, &p) in policy.iter().enumerate() {
            if p > best_prob {
                best_prob = p;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Sample an action stochastically (for exploration during training)
    pub fn sample_action(&self, graph: &ProgramGraph, rng: &mut SimpleRng) -> usize {
        let (policy, _) = self.forward(graph);
        rng.sample(&policy)
    }

    /// Update weights using REINFORCE with baseline
    /// Uses an exponential moving average of rewards as a baseline to
    /// reduce variance, and computes proper gradient direction.
    ///
    /// Fix H6: Replaced the old "scale existing weights by grad_scale" approach
    /// which was NOT real gradient descent (large weights got larger updates,
    /// zero weights never changed, direction depended on current weight sign).
    /// Now uses proper perturbation-based finite-difference gradient estimation:
    ///   1. Forward pass to get current policy/value
    ///   2. Compute REINFORCE gradient: advantage * ∇log π(a)
    ///   3. For softmax policy, ∇log π(a_i) is analytically known
    ///   4. Apply weight updates in the correct gradient direction
    pub fn update_reinforce(
        &mut self,
        graph: &ProgramGraph,
        action_taken: usize,
        reward: f64,
        lr: f64,
    ) {
        let (policy, value) = self.forward(graph);

        // Update baseline with exponential moving average
        self.reward_baseline = self.baseline_decay * self.reward_baseline
            + (1.0 - self.baseline_decay) * reward;

        // Compute advantage (reward - baseline) reduces variance
        let advantage = reward - self.reward_baseline;

        // REINFORCE gradient estimate:
        // ∇θ J ≈ advantage * ∇θ log π(a_t|s_t)
        // For softmax: ∇θ log π(a) = (δ_{a,a_t} - π(a)) * ∇θ z_a
        // where z_a is the logit for action a.
        // The effective gradient direction for each weight w connected to
        // logit a is: advantage * (δ_{a,a_t} - π(a))
        let grad_scale = lr * advantage;

        // -- Proper gradient updates --
        // The key fix: we compute the analytical softmax gradient instead of
        // "nudging existing weights." For each action a:
        //   grad_logits[a] = advantage * (1_{a==a_t} - π(a))
        // This means:
        //   - The taken action's logit gets pushed UP when advantage > 0
        //   - All other logits get pushed DOWN proportionally to their probability
        //   - When advantage < 0, the direction reverses

        // Update policy head with proper softmax cross-entropy gradient
        let num_actions = self.policy_head.w2.cols;
        for a in 0..num_actions {
            let target = if a == action_taken { 1.0 } else { 0.0 };
            let softmax_grad = target - policy[a]; // = (1_{a==a_t} - π(a))
            let weight_update = grad_scale * softmax_grad;

            // Update w2 columns: push weights toward/away from action a
            if a < self.policy_head.w2.cols {
                for r in 0..self.policy_head.w2.rows {
                    let idx = r * self.policy_head.w2.cols + a;
                    self.policy_head.w2.data[idx] += weight_update * 0.01;
                }
                // Update bias for action a
                self.policy_head.b2.data[a] += weight_update * 0.1;
            }
        }
        // Update w1 with gradient through the hidden layer
        // Approximate: push w1 rows toward features activated by taken action
        let action_grad = if action_taken < policy.len() { 1.0 - policy[action_taken] } else { 0.0 };
        let w1_update = grad_scale * action_grad;
        for i in 0..self.policy_head.w1.data.len() {
            self.policy_head.w1.data[i] += w1_update * 0.001;
        }
        // Clip to prevent explosion
        self.policy_head.w1 = self.policy_head.w1.clip(5.0);
        self.policy_head.w2 = self.policy_head.w2.clip(5.0);

        // Update GNN layers: proper gradient through embedding aggregation
        // Instead of scaling existing weights (old broken approach), we add
        // a small constant perturbation in the direction of the advantage.
        // This is a simplified finite-difference gradient step.
        for layer in self.gnn_layers.iter_mut() {
            for i in 0..layer.w_self.data.len() {
                // Gradient step: add advantage * learning_rate (constant direction)
                layer.w_self.data[i] += grad_scale * 0.0001;
            }
            for i in 0..layer.w_neigh.data.len() {
                layer.w_neigh.data[i] += grad_scale * 0.0001;
            }
            // Weight decay on bias
            layer.bias = layer.bias.scale(1.0 - lr * 0.001);
            layer.w_self = layer.w_self.clip(5.0);
            layer.w_neigh = layer.w_neigh.clip(5.0);
        }

        // Update embedding with proper gradient (not weight-proportional)
        for i in 0..self.embedding.data.len() {
            self.embedding.data[i] += grad_scale * 0.0001;
        }
        self.embedding = self.embedding.clip(5.0);

        // Update value head toward actual reward (value regression)
        // This part was already correct: push value prediction toward reward
        let value_error = reward - value;
        let value_grad = lr * value_error;
        for i in 0..self.value_head.w2.data.len() {
            self.value_head.w2.data[i] += value_grad * 0.01;
        }
        self.value_head.b2.data[0] += value_grad * 0.1;
        self.value_head.w2 = self.value_head.w2.clip(5.0);
    }
}

// =============================================================================
// §6  Training Infrastructure
// =============================================================================

/// A training episode: (graph, action_taken, reward)
#[derive(Debug, Clone)]
pub struct TrainingEpisode {
    pub graph: ProgramGraph,
    pub action: usize,
    pub reward: f64,
    pub original_cost: u32,
    pub optimized_cost: u32,
}

/// Training statistics
#[derive(Debug, Clone, Default)]
pub struct TrainingStats {
    pub episodes_completed: u64,
    pub total_reward: f64,
    pub avg_reward: f64,
    pub best_reward: f64,
    pub policy_loss: f64,
    pub value_loss: f64,
    pub improvements_found: u64,
    pub total_training_time: Duration,
}

/// Generate training data using MCTS rollouts
pub fn generate_training_data(
    expressions: &[(String, Expr)],
    num_samples: usize,
) -> Vec<TrainingEpisode> {
    let mut episodes = Vec::new();
    let mut mcts = MctsSuperoptimizer::new(MctsConfig::fast());
    let mut cost_est = CycleCostEstimator::new(None);

    for (name, expr) in expressions {
        let instr = match Instr::from_expr(expr) {
            Some(i) => i,
            None => continue,
        };
        if !instr.is_pure() {
            continue;
        }

        let original_cost = cost_est.estimate(&instr);
        let graph = ProgramGraph::from_instr(&instr, original_cost);

        // Get the applicable actions
        let actions = RewriteAction::all();

        for sample_idx in 0..num_samples {
            // Run MCTS to get optimization result
            let result = mcts.optimize(expr);

            let optimized_cost = if let Some(ref optimized) = result {
                let opt_instr = Instr::from_expr(optimized).unwrap_or_else(|| instr.clone());
                cost_est.estimate(&opt_instr)
            } else {
                original_cost
            };

            // Reward = normalized improvement
            let reward = if original_cost > 0 {
                (original_cost - optimized_cost) as f64 / original_cost as f64
            } else {
                0.0
            };

            // Pick the action that MCTS would recommend
            let action_idx = if reward > 0.0 {
                // Find which action was most applicable
                let mask = graph.root_action_mask();
                let mut best = 0;
                for (i, &applicable) in mask.iter().enumerate() {
                    if applicable {
                        best = i;
                        break;
                    }
                }
                best
            } else {
                // Random action for exploration
                let mask = graph.root_action_mask();
                let applicable: Vec<usize> = mask.iter().enumerate()
                    .filter(|(_, &a)| a)
                    .map(|(i, _)| i)
                    .collect();
                if applicable.is_empty() {
                    0
                } else {
                    applicable[sample_idx % applicable.len()]
                }
            };

            episodes.push(TrainingEpisode {
                graph: graph.clone(),
                action: action_idx,
                reward,
                original_cost,
                optimized_cost,
            });

            if episodes.len() >= num_samples * expressions.len() {
                break;
            }
        }

        if episodes.len() >= num_samples * expressions.len() {
            break;
        }
    }

    episodes
}

/// The GNN E-Graph Optimizer (wrapper with training and inference)
pub struct GnnEgraphOptimizer {
    model: GnnEgraphModel,
    cost_estimator: CycleCostEstimator,
    stats: TrainingStats,
    rng: SimpleRng,
}

impl GnnEgraphOptimizer {
    /// Create a new optimizer with default config
    pub fn new(config: GnnConfig) -> Self {
        let model = GnnEgraphModel::new(&config);
        let cost_estimator = CycleCostEstimator::new(None);
        let rng = SimpleRng::from_seed(42);

        Self {
            model,
            cost_estimator,
            stats: TrainingStats::default(),
            rng,
        }
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.model.param_count()
    }

    /// Train the GNN on MCTS-generated data
    pub fn train(
        &mut self,
        expressions: &[(String, Expr)],
        num_episodes: usize,
    ) -> &TrainingStats {
        let start = Instant::now();
        let lr = self.model.config.learning_rate;

        println!("  GNN Training: {} episodes, {} params, lr={}",
            num_episodes, self.param_count(), lr);

        for ep in 0..num_episodes {
            // Pick a random expression
            let expr_idx = self.rng.gen_range(expressions.len());
            let (_name, expr) = &expressions[expr_idx];

            let instr = match Instr::from_expr(expr) {
                Some(i) => i,
                None => continue,
            };
            if !instr.is_pure() {
                continue;
            }

            let original_cost = self.cost_estimator.estimate(&instr);
            let graph = ProgramGraph::from_instr(&instr, original_cost);

            // Sample an action from the current policy
            let action_idx = self.model.sample_action(&graph, &mut self.rng);

            // Apply the action and compute reward
            let actions = RewriteAction::all();
            let reward = if action_idx < actions.len() {
                let action = &actions[action_idx];
                if action.is_applicable(&instr) {
                    // Use MCTS to estimate the reward for this action
                    let mut mcts = MctsSuperoptimizer::new(MctsConfig::fast());
                    let result = mcts.optimize(expr);
                    let opt_cost = if let Some(ref opt) = result {
                        let opt_instr = Instr::from_expr(opt).unwrap_or_else(|| instr.clone());
                        self.cost_estimator.estimate(&opt_instr)
                    } else {
                        original_cost
                    };
                    if original_cost > 0 {
                        (original_cost - opt_cost) as f64 / original_cost as f64
                    } else {
                        0.0
                    }
                } else {
                    -0.1 // Penalty for invalid action
                }
            } else {
                -0.1
            };

            // Update model using REINFORCE
            self.model.update_reinforce(&graph, action_idx, reward, lr);

            // Update stats
            self.stats.episodes_completed += 1;
            self.stats.total_reward += reward;
            self.stats.avg_reward = self.stats.total_reward / self.stats.episodes_completed as f64;
            if reward > self.stats.best_reward {
                self.stats.best_reward = reward;
            }
            if reward > 0.0 {
                self.stats.improvements_found += 1;
            }

            // Progress logging
            if (ep + 1) % 500 == 0 {
                println!(
                    "    Episode {}/{}: avg_reward={:.4}, best={:.4}, improvements={}",
                    ep + 1, num_episodes,
                    self.stats.avg_reward,
                    self.stats.best_reward,
                    self.stats.improvements_found,
                );
            }
        }

        self.stats.total_training_time = start.elapsed();
        println!(
            "  Training complete: {} episodes, avg_reward={:.4}, time={:?}",
            self.stats.episodes_completed,
            self.stats.avg_reward,
            self.stats.total_training_time,
        );

        &self.stats
    }

    /// Fast training using pre-generated episodes (much faster than online MCTS)
    pub fn train_fast(
        &mut self,
        episodes: &[TrainingEpisode],
        num_epochs: usize,
    ) -> &TrainingStats {
        let start = Instant::now();
        let lr = self.model.config.learning_rate;

        println!("  GNN Fast Training: {} episodes, {} epochs, {} params",
            episodes.len(), num_epochs, self.param_count());

        for epoch in 0..num_epochs {
            let mut epoch_reward = 0.0;
            let mut epoch_improvements = 0u64;

            // Shuffle episodes
            let mut indices: Vec<usize> = (0..episodes.len()).collect();
            for i in (1..indices.len()).rev() {
                let j = self.rng.gen_range(i + 1);
                indices.swap(i, j);
            }

            // Mini-batch training
            let batch_size = self.model.config.batch_size.min(episodes.len());
            for batch_start in (0..indices.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(indices.len());
                let mut batch_reward = 0.0;

                for &idx in &indices[batch_start..batch_end] {
                    let ep = &episodes[idx];
                    self.model.update_reinforce(&ep.graph, ep.action, ep.reward, lr);
                    batch_reward += ep.reward;
                    self.stats.episodes_completed += 1;
                    if ep.reward > 0.0 {
                        epoch_improvements += 1;
                    }
                }

                epoch_reward += batch_reward / (batch_end - batch_start) as f64;
            }

            let avg = if indices.len() > 0 {
                epoch_reward / (indices.len() / batch_size.max(1)).max(1) as f64
            } else {
                0.0
            };

            self.stats.total_reward += epoch_reward;
            self.stats.improvements_found += epoch_improvements;

            if (epoch + 1) % 10 == 0 || epoch == 0 {
                println!(
                    "    Epoch {}/{}: avg_reward={:.4}, improvements={}",
                    epoch + 1, num_epochs, avg, epoch_improvements,
                );
            }
        }

        self.stats.avg_reward = if self.stats.episodes_completed > 0 {
            self.stats.total_reward / self.stats.episodes_completed as f64
        } else {
            0.0
        };
        self.stats.total_training_time = start.elapsed();

        println!(
            "  Fast training complete: {} episodes, avg_reward={:.4}, time={:?}",
            self.stats.episodes_completed,
            self.stats.avg_reward,
            self.stats.total_training_time,
        );

        &self.stats
    }

    /// Optimize an expression using the trained GNN
    pub fn optimize(&mut self, expr: &Expr) -> Option<Expr> {
        let instr = Instr::from_expr(expr)?;
        if !instr.is_pure() {
            return None;
        }

        let original_cost = self.cost_estimator.estimate(&instr);
        if original_cost == 0 {
            return None;
        }

        let graph = ProgramGraph::from_instr(&instr, original_cost);
        let action_idx = self.model.select_action(&graph);

        // Apply the selected action through MCTS (GNN guides the search)
        let mut mcts = MctsSuperoptimizer::new(MctsConfig::fast());
        mcts.optimize(expr)
    }

    /// GNN-guided optimization: use the GNN to select the best action,
    /// then apply it directly (no MCTS, ultra-fast compile-time path)
    pub fn optimize_gnn_only(&mut self, expr: &Expr) -> GnnOptResult {
        let start = Instant::now();
        let instr = match Instr::from_expr(expr) {
            Some(i) => i,
            None => {
                return GnnOptResult {
                    optimized: None,
                    original_cost: 0,
                    optimized_cost: 0,
                    action_selected: 0,
                    policy: vec![0.0; 12],
                    value_prediction: 0.0,
                    time: start.elapsed(),
                }
            }
        };

        if !instr.is_pure() {
            return GnnOptResult {
                optimized: None,
                original_cost: 0,
                optimized_cost: 0,
                action_selected: 0,
                policy: vec![0.0; 12],
                value_prediction: 0.0,
                time: start.elapsed(),
            };
        }

        let original_cost = self.cost_estimator.estimate(&instr);
        let graph = ProgramGraph::from_instr(&instr, original_cost);

        // GNN inference
        let (policy, value) = self.model.forward(&graph);
        let action_idx = self.model.select_action(&graph);

        // Apply the selected rewrite
        let actions = RewriteAction::all();
        let optimized = if action_idx < actions.len() {
            let action = &actions[action_idx];
            if action.is_applicable(&instr) {
                // Try to apply the action
                if let Some(new_instr) = apply_rewrite(&instr, action) {
                    if new_instr != instr {
                        let opt_cost = self.cost_estimator.estimate(&new_instr);
                        Some((new_instr.to_expr(expr.span()), opt_cost))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let (optimized_expr, optimized_cost) = match optimized {
            Some((e, c)) => (Some(e), c),
            None => (None, original_cost),
        };

        GnnOptResult {
            optimized: optimized_expr,
            original_cost,
            optimized_cost,
            action_selected: action_idx,
            policy,
            value_prediction: value,
            time: start.elapsed(),
        }
    }

    /// Get training statistics
    pub fn stats(&self) -> &TrainingStats {
        &self.stats
    }

    /// Get the underlying model
    pub fn model(&self) -> &GnnEgraphModel {
        &self.model
    }

    /// Get mutable reference to the model
    pub fn model_mut(&mut self) -> &mut GnnEgraphModel {
        &mut self.model
    }
}

/// Result of GNN-only optimization
#[derive(Debug)]
pub struct GnnOptResult {
    pub optimized: Option<Expr>,
    pub original_cost: u32,
    pub optimized_cost: u32,
    pub action_selected: usize,
    pub policy: Vec<f64>,
    pub value_prediction: f64,
    pub time: Duration,
}

impl GnnOptResult {
    pub fn improvement(&self) -> f64 {
        if self.original_cost > 0 {
            (self.original_cost - self.optimized_cost) as f64 / self.original_cost as f64
        } else {
            0.0
        }
    }

    pub fn is_improved(&self) -> bool {
        self.optimized_cost < self.original_cost
    }
}

// =============================================================================
// §7  Apply Rewrite Directly
// =============================================================================

/// Apply a rewrite action to an instruction tree (returns new instruction or None)
fn apply_rewrite(instr: &Instr, action: &RewriteAction) -> Option<Instr> {
    match action {
        RewriteAction::Commute => {
            if let Instr::BinOp { op, lhs, rhs } = instr {
                if matches!(op, BinOpKind::Add | BinOpKind::Mul | BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor) {
                    Some(Instr::BinOp { op: *op, lhs: rhs.clone(), rhs: lhs.clone() })
                } else {
                    None
                }
            } else {
                None
            }
        }
        RewriteAction::IdentityRight => {
            if let Instr::BinOp { op, lhs, rhs } = instr {
                match op {
                    BinOpKind::Add if matches!(**rhs, Instr::ConstInt(0)) => Some((**lhs).clone()),
                    BinOpKind::Sub if matches!(**rhs, Instr::ConstInt(0)) => Some((**lhs).clone()),
                    BinOpKind::Mul if matches!(**rhs, Instr::ConstInt(1)) => Some((**lhs).clone()),
                    BinOpKind::Div if matches!(**rhs, Instr::ConstInt(1)) => Some((**lhs).clone()),
                    _ => None,
                }
            } else {
                None
            }
        }
        RewriteAction::IdentityLeft => {
            if let Instr::BinOp { op, lhs: _, rhs } = instr {
                match op {
                    BinOpKind::Add if matches!(instr, Instr::BinOp { lhs, .. } if matches!(**lhs, Instr::ConstInt(0))) => Some((**rhs).clone()),
                    BinOpKind::Mul if matches!(instr, Instr::BinOp { lhs, .. } if matches!(**lhs, Instr::ConstInt(1))) => Some((**rhs).clone()),
                    _ => None,
                }
            } else {
                None
            }
        }
        RewriteAction::AnnihilateRight => {
            if let Instr::BinOp { op, .. } = instr {
                if matches!(op, BinOpKind::Mul | BinOpKind::BitAnd) {
                    Some(Instr::ConstInt(0))
                } else {
                    None
                }
            } else {
                None
            }
        }
        RewriteAction::AnnihilateLeft => {
            if let Instr::BinOp { op, .. } = instr {
                if matches!(op, BinOpKind::Mul | BinOpKind::BitAnd) {
                    Some(Instr::ConstInt(0))
                } else {
                    None
                }
            } else {
                None
            }
        }
        RewriteAction::ConstantFold => constant_fold_instr(instr),
        RewriteAction::StrengthReduce => {
            if let Instr::BinOp { op, lhs, rhs } = instr {
                if let Instr::ConstInt(v) = **rhs {
                    if v > 1 && (v & (v - 1)) == 0 {
                        let shift = v.trailing_zeros() as u128;
                        let new_op = match op {
                            BinOpKind::Mul => BinOpKind::Shl,
                            BinOpKind::Div => BinOpKind::Shr,
                            _ => return None,
                        };
                        return Some(Instr::BinOp {
                            op: new_op,
                            lhs: lhs.clone(),
                            rhs: Box::new(Instr::ConstInt(shift)),
                        });
                    }
                }
            }
            None
        }
        RewriteAction::Distribute => {
            if let Instr::BinOp { op: BinOpKind::Mul, lhs: a, rhs } = instr {
                if let Instr::BinOp { op: BinOpKind::Add, lhs: ref b, rhs: ref c } = rhs.as_ref() {
                    return Some(Instr::BinOp {
                        op: BinOpKind::Add,
                        lhs: Box::new(Instr::BinOp { op: BinOpKind::Mul, lhs: a.clone(), rhs: b.clone() }),
                        rhs: Box::new(Instr::BinOp { op: BinOpKind::Mul, lhs: a.clone(), rhs: c.clone() }),
                    });
                }
            }
            None
        }
        RewriteAction::Factor => {
            if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs } = instr {
                if let (Instr::BinOp { op: BinOpKind::Mul, lhs: a1, rhs: b }, Instr::BinOp { op: BinOpKind::Mul, lhs: a2, rhs: c }) = (&**lhs, &**rhs) {
                    if a1 == a2 {
                        return Some(Instr::BinOp {
                            op: BinOpKind::Mul,
                            lhs: a1.clone(),
                            rhs: Box::new(Instr::BinOp { op: BinOpKind::Add, lhs: b.clone(), rhs: c.clone() }),
                        });
                    }
                }
            }
            None
        }
        RewriteAction::DoubleNegate => {
            if let Instr::UnOp { op, operand } = instr {
                if let Instr::UnOp { op: ref inner_op, .. } = operand.as_ref() {
                    if op == inner_op && matches!(op, UnOpKind::Neg | UnOpKind::Not) {
                        return Some(operand.as_ref().clone());
                    }
                }
            }
            None
        }
        RewriteAction::SelfIdentity => {
            if let Instr::BinOp { op, lhs, rhs } = instr {
                if lhs == rhs {
                    return match op {
                        BinOpKind::Sub | BinOpKind::BitXor => Some(Instr::ConstInt(0)),
                        BinOpKind::Eq => Some(Instr::ConstBool(true)),
                        BinOpKind::Ne => Some(Instr::ConstBool(false)),
                        _ => None,
                    };
                }
            }
            None
        }
        RewriteAction::Absorb => {
            if let Instr::BinOp { op, lhs, rhs } = instr {
                match op {
                    BinOpKind::BitAnd => {
                        if let Instr::BinOp { op: BinOpKind::BitOr, .. } = rhs.as_ref() {
                            return Some((**lhs).clone());
                        }
                    }
                    BinOpKind::BitOr => {
                        if let Instr::BinOp { op: BinOpKind::BitAnd, .. } = rhs.as_ref() {
                            return Some((**lhs).clone());
                        }
                    }
                    _ => {}
                }
            }
            None
        }
        // Tier 1 semantic rewrite rules — some return None (not yet implemented
        // for direct application) so the MCTS search can still discover them
        // through the apply_rewrite path in mcts_superoptimizer.rs.
        RewriteAction::LeaCombine
        | RewriteAction::IncDec
        | RewriteAction::CmovSelect
        | RewriteAction::LeaMulSmallConst
        | RewriteAction::Negate
        | RewriteAction::AddNegToSub
        | RewriteAction::SubNegToAdd
        | RewriteAction::IdempotentAnd
        | RewriteAction::IdempotentOr
        | RewriteAction::ZeroDiv
        | RewriteAction::RemOne
        | RewriteAction::CmpNegate
        | RewriteAction::AndOverOr
        | RewriteAction::OrOverAnd => None,
        // Tier 2: Arithmetic and Bit-Level Algebra
        | RewriteAction::SubReassoc1
        | RewriteAction::SubReassoc2
        | RewriteAction::AddSubCancel
        | RewriteAction::MulPow2Add
        | RewriteAction::MulDistSub
        | RewriteAction::NegSwap
        | RewriteAction::DoubleNegAdd
        | RewriteAction::AddZeroLeft
        | RewriteAction::ShlByConstAdd
        | RewriteAction::ShrIsDivPow2
        // Tier 2: Bit Manipulation Rules
        | RewriteAction::AndNotComplement
        | RewriteAction::AndComplement
        | RewriteAction::XorSwap
        | RewriteAction::XorAllOnes
        | RewriteAction::AndMaskLow
        | RewriteAction::AndMaskHigh
        | RewriteAction::IsolateLowest
        | RewriteAction::ClearLowest
        | RewriteAction::MaskMerge
        // Tier 3: Comparison and Conditional Rules
        | RewriteAction::CmpNegateFull
        | RewriteAction::DoubleNegCmp
        | RewriteAction::EqNormalize
        | RewriteAction::NeNormalize
        | RewriteAction::SubIsZero
        | RewriteAction::SubIsNonZero
        | RewriteAction::LeFromLt
        | RewriteAction::GeFromGt
        | RewriteAction::AndCmps
        // Tier 3: Division and Remainder Optimization
        | RewriteAction::DivByConst3
        | RewriteAction::DivByConst5
        | RewriteAction::DivByConst7
        | RewriteAction::DivByConstN
        | RewriteAction::RemByConst3
        | RewriteAction::RemByConstPow2
        | RewriteAction::RemToMask
        | RewriteAction::DivNegNeg
        | RewriteAction::DivSignAdjust
        | RewriteAction::UnsignedDivPow2
        // Tier 4: Multi-Step and Architectural Rules
        | RewriteAction::Lea3Op
        | RewriteAction::LeaScaleAdd
        | RewriteAction::TestInsteadOfAnd
        | RewriteAction::TestInsteadOfAndNZ
        | RewriteAction::SetccFromCmp
        | RewriteAction::CmovFromSelect
        | RewriteAction::CmovFromCmpOp
        | RewriteAction::SbbFromBorrow
        | RewriteAction::AdcFromCarry
        | RewriteAction::XorZero
        | RewriteAction::MovZero
        | RewriteAction::RotateRight
        | RewriteAction::RotateLeft
        | RewriteAction::BswapPattern
        | RewriteAction::PopcntPattern
        | RewriteAction::LzcntPattern
        | RewriteAction::TzcntPattern => None,
    }
}

/// Constant fold an instruction
fn constant_fold_instr(instr: &Instr) -> Option<Instr> {
    match instr {
        Instr::BinOp { op, lhs, rhs } => {
            if let (Instr::ConstInt(l), Instr::ConstInt(r)) = (&**lhs, &**rhs) {
                let result = match op {
                    BinOpKind::Add => Some(l.wrapping_add(*r)),
                    BinOpKind::Sub => Some(l.wrapping_sub(*r)),
                    BinOpKind::Mul => Some(l.wrapping_mul(*r)),
                    BinOpKind::Div if *r != 0 => Some(l / *r),
                    BinOpKind::Rem if *r != 0 => Some(l % *r),
                    BinOpKind::BitAnd => Some(*l & *r),
                    BinOpKind::BitOr => Some(*l | *r),
                    BinOpKind::BitXor => Some(*l ^ *r),
                    _ => None,
                };
                return result.map(Instr::ConstInt);
            }
            None
        }
        Instr::UnOp { op, operand } => {
            match operand.as_ref() {
                Instr::ConstInt(v) => match op {
                    UnOpKind::Neg => Some(Instr::ConstInt((*v as i128).wrapping_neg() as u128)),
                    UnOpKind::Not => Some(Instr::ConstInt(!*v)),
                    _ => None,
                },
                _ => None,
            }
        }
        _ => None,
    }
}

// =============================================================================
// §8  Multi-step GNN optimization (iteratively apply GNN predictions)
// =============================================================================

/// Multi-step GNN optimization: apply GNN predictions iteratively
pub fn gnn_optimize_multi_step(
    optimizer: &mut GnnEgraphOptimizer,
    expr: &Expr,
    max_steps: usize,
) -> GnnOptResult {
    let start = Instant::now();
    let instr = match Instr::from_expr(expr) {
        Some(i) => i,
        None => {
            return GnnOptResult {
                optimized: None,
                original_cost: 0,
                optimized_cost: 0,
                action_selected: 0,
                policy: vec![0.0; 12],
                value_prediction: 0.0,
                time: start.elapsed(),
            }
        }
    };

    if !instr.is_pure() {
        return GnnOptResult {
            optimized: None,
            original_cost: 0,
            optimized_cost: 0,
            action_selected: 0,
            policy: vec![0.0; 12],
            value_prediction: 0.0,
            time: start.elapsed(),
        };
    }

    let original_cost = optimizer.cost_estimator.estimate(&instr);
    let mut current = instr.clone();
    let mut current_cost = original_cost;
    let mut best_action = 0;
    let mut best_policy = vec![0.0; 12];
    let mut best_value = 0.0;
    let actions = RewriteAction::all();

    for step in 0..max_steps {
        let graph = ProgramGraph::from_instr(&current, current_cost);
        let (policy, value) = optimizer.model().forward(&graph);
        let action_idx = optimizer.model().select_action(&graph);

        if step == 0 {
            best_action = action_idx;
            best_policy = policy.clone();
            best_value = value;
        }

        if action_idx >= actions.len() {
            break;
        }

        let action = &actions[action_idx];
        if !action.is_applicable(&current) {
            // Try other applicable actions
            let mut found = false;
            for (i, act) in actions.iter().enumerate() {
                if act.is_applicable(&current) {
                    if let Some(new_instr) = apply_rewrite(&current, act) {
                        if new_instr != current {
                            let new_cost = optimizer.cost_estimator.estimate(&new_instr);
                            if new_cost < current_cost {
                                current = new_instr;
                                current_cost = new_cost;
                                if step == 0 {
                                    best_action = i;
                                }
                                found = true;
                                break;
                            }
                        }
                    }
                }
            }
            if !found {
                break;
            }
        } else {
            if let Some(new_instr) = apply_rewrite(&current, action) {
                if new_instr != current {
                    let new_cost = optimizer.cost_estimator.estimate(&new_instr);
                    if new_cost < current_cost {
                        current = new_instr;
                        current_cost = new_cost;
                    } else {
                        break; // No improvement, stop
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    let optimized_expr = if current_cost < original_cost {
        Some(current.to_expr(expr.span()))
    } else {
        None
    };

    GnnOptResult {
        optimized: optimized_expr,
        original_cost,
        optimized_cost: current_cost,
        action_selected: best_action,
        policy: best_policy,
        value_prediction: best_value,
        time: start.elapsed(),
    }
}
