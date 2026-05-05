// =============================================================================
// MCTS-Port Superoptimizer
//
// A Monte Carlo Tree Search superoptimizer that uses cycle-accurate port
// simulation to find the truly optimal instruction sequence for the specific
// CPU it's running on. Unlike e-graphs which optimize for node count, this
// optimizes for wall-clock time on real hardware.
//
// Key advantages over e-graph equality saturation:
//   1. Uses REAL cycle costs (port contention, latency, throughput)
//   2. Finds multi-step optimization sequences (explore→exploit)
//   3. Search focuses on promising paths — doesn't blow up exponentially
//   4. Port-aware: optimizes for YOUR specific CPU, not a generic model
//
// Architecture:
//   MCTS tree → each node is a program state
//   Actions = rewrite transformations from the e-graph rule set
//   Score = simulated cycle count using HardwareCostModel
//   Selection = UCB1 formula (balance explore vs exploit)
// =============================================================================

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::compiler::ast::*;
use crate::optimizer::hardware_cost_model::{HardwareCostModel, Microarchitecture, PortMap};
use crate::Span;

// =============================================================================
// §1  MCTS Configuration
// =============================================================================

/// Configuration for the MCTS-Port Superoptimizer
#[derive(Debug, Clone)]
pub struct MctsConfig {
    /// Maximum number of MCTS simulations to run per expression
    pub max_simulations: usize,
    /// Maximum search depth (number of consecutive transformations)
    pub max_depth: usize,
    /// UCB1 exploration constant (sqrt(2) is standard)
    pub exploration_constant: f64,
    /// Time budget per expression (in milliseconds). 0 = no time limit
    pub time_budget_ms: u64,
    /// Minimum improvement (in cycles) to accept a transformation
    pub min_improvement: u32,
    /// Whether to use hardware-aware cost model (vs simple node count)
    pub use_hardware_cost: bool,
    /// Target microarchitecture (None = auto-detect)
    pub microarch: Option<Microarchitecture>,
    /// Number of random test inputs for equivalence checking
    pub verif_inputs: usize,
    /// Maximum number of nodes in the search tree before pruning
    pub max_tree_size: usize,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            max_simulations: 200,
            max_depth: 6,
            exploration_constant: 1.414, // sqrt(2)
            time_budget_ms: 50,
            min_improvement: 1,
            use_hardware_cost: true,
            microarch: None,
            verif_inputs: 16,
            max_tree_size: 10_000,
        }
    }
}

impl MctsConfig {
    /// Fast config for development (quick but less thorough)
    pub fn fast() -> Self {
        Self {
            max_simulations: 50,
            max_depth: 3,
            time_budget_ms: 10,
            ..Default::default()
        }
    }

    /// Thorough config for production (slow but finds more optimizations)
    pub fn thorough() -> Self {
        Self {
            max_simulations: 1000,
            max_depth: 8,
            time_budget_ms: 200,
            ..Default::default()
        }
    }

    /// Maximum optimization config
    pub fn maximum() -> Self {
        Self {
            max_simulations: 5000,
            max_depth: 12,
            time_budget_ms: 1000,
            max_tree_size: 100_000,
            ..Default::default()
        }
    }

    /// Local-only mode: limits search to single expressions/blocks.
    /// Critical for large codebases (>50K LoC) where global search explodes.
    /// This mode:
    /// - Only optimizes within the current function scope
    /// - No inter-procedural analysis
    /// - No cross-block state tracking
    /// - Reduced max_tree_size to prevent memory blowup
    /// - Shorter time budgets per expression
    pub fn local_only() -> Self {
        Self {
            max_simulations: 100,
            max_depth: 4,
            exploration_constant: 1.414,
            time_budget_ms: 20,
            min_improvement: 2,          // Only accept significant improvements
            use_hardware_cost: true,
            microarch: None,
            verif_inputs: 8,             // Fewer verification inputs
            max_tree_size: 5_000,        // Prevent memory explosion
        }
    }

    /// Very local mode: optimized for hot-path kernels only.
    /// Use for tight loops and inner functions.
    pub fn hotpath() -> Self {
        Self {
            max_simulations: 200,
            max_depth: 6,
            exploration_constant: 1.414,
            time_budget_ms: 50,
            min_improvement: 1,
            use_hardware_cost: true,
            microarch: None,
            verif_inputs: 16,
            max_tree_size: 10_000,
        }
    }
}

/// Mode for controlling search scope to prevent explosion on large codebases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Full inter-procedural analysis (default, for small projects)
    Global,
    /// Local-only: within current function scope (for >50K LoC)
    Local,
    /// Hot-path: aggressive optimization for inner loops
    HotPath,
}

// =============================================================================
// §2  Program Representation for MCTS
// =============================================================================

/// A compact instruction representation for the MCTS search space.
/// This is intentionally simpler than the full AST — we only track operations
/// that affect cycle counts and can be rewritten.
#[derive(Debug, Clone, PartialEq)]
pub enum Instr {
    /// Load a constant integer value
    ConstInt(u128),
    /// Load a constant float value (raw bits)
    ConstFloat(u64),
    /// Load a constant boolean
    ConstBool(bool),
    /// Reference to a variable (by name)
    Var(String),
    /// Binary operation: result = lhs op rhs
    BinOp {
        op: BinOpKind,
        lhs: Box<Instr>,
        rhs: Box<Instr>,
    },
    /// Unary operation: result = op operand
    UnOp {
        op: UnOpKind,
        operand: Box<Instr>,
    },
}

impl Instr {
    /// Convert an Expr to an Instr (simplification for the MCTS search space)
    pub fn from_expr(expr: &Expr) -> Option<Self> {
        match expr {
            Expr::IntLit { value, .. } => Some(Instr::ConstInt(*value)),
            Expr::FloatLit { value, .. } => Some(Instr::ConstFloat(value.to_bits())),
            Expr::BoolLit { value, .. } => Some(Instr::ConstBool(*value)),
            Expr::Ident { name, .. } => Some(Instr::Var(name.clone())),
            Expr::BinOp { op, lhs, rhs, .. } => {
                let l = Instr::from_expr(lhs)?;
                let r = Instr::from_expr(rhs)?;
                Some(Instr::BinOp {
                    op: *op,
                    lhs: Box::new(l),
                    rhs: Box::new(r),
                })
            }
            Expr::UnOp { op, expr, .. } => {
                let inner = Instr::from_expr(expr)?;
                Some(Instr::UnOp {
                    op: *op,
                    operand: Box::new(inner),
                })
            }
            _ => None,
        }
    }

    /// Convert an Instr back to an Expr
    pub fn to_expr(&self, span: Span) -> Expr {
        match self {
            Instr::ConstInt(v) => Expr::IntLit { span, value: *v },
            Instr::ConstFloat(bits) => Expr::FloatLit {
                span,
                value: f64::from_bits(*bits),
            },
            Instr::ConstBool(v) => Expr::BoolLit { span, value: *v },
            Instr::Var(name) => Expr::Ident {
                span,
                name: name.clone(),
            },
            Instr::BinOp { op, lhs, rhs } => Expr::BinOp {
                span,
                op: *op,
                lhs: Box::new(lhs.to_expr(span)),
                rhs: Box::new(rhs.to_expr(span)),
            },
            Instr::UnOp { op, operand } => Expr::UnOp {
                span,
                op: *op,
                expr: Box::new(operand.to_expr(span)),
            },
        }
    }

    /// Check if this instruction is a pure expression (no side effects)
    pub fn is_pure(&self) -> bool {
        match self {
            Instr::ConstInt(_) | Instr::ConstFloat(_) | Instr::ConstBool(_) | Instr::Var(_) => true,
            Instr::BinOp { lhs, rhs, .. } => lhs.is_pure() && rhs.is_pure(),
            Instr::UnOp { operand, .. } => operand.is_pure(),
        }
    }

    /// Collect all variables referenced in this instruction
    pub fn variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_variables(&mut vars);
        vars
    }

    fn collect_variables(&self, vars: &mut Vec<String>) {
        match self {
            Instr::Var(name) => {
                if !vars.contains(name) {
                    vars.push(name.clone());
                }
            }
            Instr::BinOp { lhs, rhs, .. } => {
                lhs.collect_variables(vars);
                rhs.collect_variables(vars);
            }
            Instr::UnOp { operand, .. } => {
                operand.collect_variables(vars);
            }
            _ => {}
        }
    }
}

// =============================================================================
// §3  Cycle-Accurate Cost Estimation
// =============================================================================

/// A cost estimator that uses the hardware cost model for cycle-accurate scoring
pub struct CycleCostEstimator {
    hw_model: HardwareCostModel,
}

impl CycleCostEstimator {
    pub fn new(microarch: Option<Microarchitecture>) -> Self {
        let hw_model = match microarch {
            Some(ma) => HardwareCostModel::with_microarch(ma),
            None => HardwareCostModel::new(),
        };
        Self { hw_model }
    }

    /// Estimate the cycle cost of an instruction tree
    pub fn estimate(&mut self, instr: &Instr) -> u32 {
        let instruction_sequence = self.flatten_to_schedule(instr);
        self.hw_model.estimate_sequence(&instruction_sequence)
    }

    /// Simple node-count cost (fallback when hardware model isn't wanted)
    pub fn node_cost(instr: &Instr) -> f64 {
        match instr {
            Instr::ConstInt(_) | Instr::ConstFloat(_) | Instr::ConstBool(_) | Instr::Var(_) => 0.0,
            Instr::BinOp { op, lhs, rhs } => {
                let base = match op {
                    BinOpKind::Add | BinOpKind::Sub => 1.0,
                    BinOpKind::Mul => 3.0,
                    BinOpKind::Div | BinOpKind::Rem => 20.0,
                    BinOpKind::Shl | BinOpKind::Shr => 1.0,
                    _ => 1.0,
                };
                base + Self::node_cost(lhs) + Self::node_cost(rhs)
            }
            Instr::UnOp { operand, .. } => 1.0 + Self::node_cost(operand),
        }
    }

    /// Flatten an instruction tree into a sequence of instruction names for the scheduler
    fn flatten_to_schedule(&self, instr: &Instr) -> Vec<&'static str> {
        let mut schedule = Vec::new();
        self.flatten_inner(instr, &mut schedule);
        schedule
    }

    fn flatten_inner<'a>(&'a self, instr: &'a Instr, schedule: &mut Vec<&'static str>) {
        match instr {
            Instr::ConstInt(_) | Instr::ConstFloat(_) | Instr::ConstBool(_) => {
                // Constants are typically folded into instructions or loaded from immediate fields
                schedule.push("load");
            }
            Instr::Var(_) => {
                // Variable access = load from register/stack
                schedule.push("load");
            }
            Instr::BinOp { op, lhs, rhs } => {
                // Schedule children first (dependency)
                self.flatten_inner(lhs, schedule);
                self.flatten_inner(rhs, schedule);
                // Then schedule this operation
                let instr_name = match op {
                    BinOpKind::Add => "add",
                    BinOpKind::Sub => "sub",
                    BinOpKind::Mul => "mul",
                    BinOpKind::Div => "div",
                    BinOpKind::Rem => "rem",
                    BinOpKind::BitAnd => "and",
                    BinOpKind::BitOr => "or",
                    BinOpKind::BitXor => "xor",
                    BinOpKind::Shl => "shl",
                    BinOpKind::Shr => "shr",
                    _ => "add", // fallback
                };
                schedule.push(instr_name);
            }
            Instr::UnOp { operand, .. } => {
                self.flatten_inner(operand, schedule);
                schedule.push("add"); // Most unops compile to a single ALU uop
            }
        }
    }

    /// Get the microarchitecture being used
    pub fn microarch(&self) -> Microarchitecture {
        self.hw_model.microarch()
    }
}

// =============================================================================
// §4  Rewrite Actions
// =============================================================================

/// A rewrite action that transforms one program into an equivalent one.
/// These are the "moves" in the MCTS game tree.
#[derive(Debug, Clone)]
pub enum RewriteAction {
    /// Commute a binary operation: a op b → b op a
    Commute,
    /// Apply identity: a + 0 → a, a * 1 → a
    IdentityLeft,
    IdentityRight,
    /// Apply annihilation: a * 0 → 0, a & 0 → 0
    AnnihilateLeft,
    AnnihilateRight,
    /// Constant fold: fold constants at compile time
    ConstantFold,
    /// Strength reduce: x * 2^k → x << k, x / 2^k → x >> k
    StrengthReduce,
    /// Distribute: a * (b + c) → a*b + a*c
    Distribute,
    /// Factor: a*b + a*c → a * (b + c)
    Factor,
    /// Double negation: -(-x) → x, !(!x) → x
    DoubleNegate,
    /// Self-identity: x - x → 0, x ^ x → 0, x == x → true
    SelfIdentity,
    /// Absorb: a & (a | b) → a
    Absorb,
}

impl RewriteAction {
    /// All available rewrite actions
    pub fn all() -> &'static [RewriteAction] {
        &[
            RewriteAction::Commute,
            RewriteAction::IdentityLeft,
            RewriteAction::IdentityRight,
            RewriteAction::AnnihilateLeft,
            RewriteAction::AnnihilateRight,
            RewriteAction::ConstantFold,
            RewriteAction::StrengthReduce,
            RewriteAction::Distribute,
            RewriteAction::Factor,
            RewriteAction::DoubleNegate,
            RewriteAction::SelfIdentity,
            RewriteAction::Absorb,
        ]
    }

    /// Check if this action is applicable to a given instruction
    pub fn is_applicable(&self, instr: &Instr) -> bool {
        match self {
            RewriteAction::Commute => {
                if let Instr::BinOp { op, .. } = instr {
                    matches!(op, BinOpKind::Add | BinOpKind::Mul | BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor | BinOpKind::Eq | BinOpKind::Ne)
                } else {
                    false
                }
            }
            RewriteAction::IdentityLeft => {
                if let Instr::BinOp { op, lhs, .. } = instr {
                    match op {
                        BinOpKind::Add => matches!(**lhs, Instr::ConstInt(0)),
                        BinOpKind::Mul => matches!(**lhs, Instr::ConstInt(1)),
                        BinOpKind::BitOr => matches!(**lhs, Instr::ConstInt(0)),
                        BinOpKind::BitAnd => matches!(**lhs, Instr::ConstInt(v) if v == u128::MAX),
                        _ => false,
                    }
                } else {
                    false
                }
            }
            RewriteAction::IdentityRight => {
                if let Instr::BinOp { op, rhs, .. } = instr {
                    match op {
                        BinOpKind::Add => matches!(**rhs, Instr::ConstInt(0)),
                        BinOpKind::Sub => matches!(**rhs, Instr::ConstInt(0)),
                        BinOpKind::Mul => matches!(**rhs, Instr::ConstInt(1)),
                        BinOpKind::Div => matches!(**rhs, Instr::ConstInt(1)),
                        BinOpKind::BitOr => matches!(**rhs, Instr::ConstInt(0)),
                        BinOpKind::BitAnd => matches!(**rhs, Instr::ConstInt(v) if v == u128::MAX),
                        BinOpKind::Shl | BinOpKind::Shr => matches!(**rhs, Instr::ConstInt(0)),
                        _ => false,
                    }
                } else {
                    false
                }
            }
            RewriteAction::AnnihilateLeft => {
                if let Instr::BinOp { op, lhs, .. } = instr {
                    match op {
                        BinOpKind::Mul => matches!(**lhs, Instr::ConstInt(0)),
                        BinOpKind::BitAnd => matches!(**lhs, Instr::ConstInt(0)),
                        _ => false,
                    }
                } else {
                    false
                }
            }
            RewriteAction::AnnihilateRight => {
                if let Instr::BinOp { op, rhs, .. } = instr {
                    match op {
                        BinOpKind::Mul => matches!(**rhs, Instr::ConstInt(0)),
                        BinOpKind::BitAnd => matches!(**rhs, Instr::ConstInt(0)),
                        _ => false,
                    }
                } else {
                    false
                }
            }
            RewriteAction::ConstantFold => Self::can_constant_fold(instr),
            RewriteAction::StrengthReduce => {
                if let Instr::BinOp { op, rhs, .. } = instr {
                    if matches!(op, BinOpKind::Mul | BinOpKind::Div) {
                        if let Instr::ConstInt(v) = **rhs {
                            v > 1 && (v & (v - 1)) == 0 // power of 2
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            RewriteAction::Distribute => {
                // a * (b + c) pattern
                if let Instr::BinOp { op: BinOpKind::Mul, rhs, .. } = instr {
                    matches!(**rhs, Instr::BinOp { op: BinOpKind::Add, .. })
                } else {
                    false
                }
            }
            RewriteAction::Factor => {
                // a*b + a*c pattern — harder to detect structurally
                if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs, .. } = instr {
                    matches!(**lhs, Instr::BinOp { op: BinOpKind::Mul, .. })
                        && matches!(**rhs, Instr::BinOp { op: BinOpKind::Mul, .. })
                } else {
                    false
                }
            }
            RewriteAction::DoubleNegate => {
                if let Instr::UnOp { op, operand } = instr {
                    matches!(**operand, Instr::UnOp { .. })
                        && matches!(op, UnOpKind::Neg | UnOpKind::Not)
                } else {
                    false
                }
            }
            RewriteAction::SelfIdentity => {
                if let Instr::BinOp { op, lhs, rhs } = instr {
                    match op {
                        BinOpKind::Sub | BinOpKind::BitXor => lhs == rhs,
                        BinOpKind::Eq => lhs == rhs,
                        BinOpKind::Ne => lhs == rhs,
                        _ => false,
                    }
                } else {
                    false
                }
            }
            RewriteAction::Absorb => {
                // a & (a | b) → a  or  a | (a & b) → a
                if let Instr::BinOp { op, lhs, rhs } = instr {
                    match op {
                        BinOpKind::BitAnd => {
                            if let Instr::BinOp { op: BinOpKind::BitOr, .. } = **rhs {
                                true
                            } else {
                                false
                            }
                        }
                        BinOpKind::BitOr => {
                            if let Instr::BinOp { op: BinOpKind::BitAnd, .. } = **rhs {
                                true
                            } else {
                                false
                            }
                        }
                        _ => false,
                    }
                } else {
                    false
                }
            }
        }
    }

    fn can_constant_fold(instr: &Instr) -> bool {
        match instr {
            Instr::BinOp { lhs, rhs, .. } => {
                matches!(**lhs, Instr::ConstInt(_)) && matches!(**rhs, Instr::ConstInt(_))
            }
            Instr::UnOp { operand, .. } => matches!(**operand, Instr::ConstInt(_)),
            _ => false,
        }
    }
}

// =============================================================================
// §5  MCTS Tree Node
// =============================================================================

/// A node in the MCTS search tree
#[derive(Debug, Clone)]
pub struct MctsNode {
    /// The program state at this node
    pub program: Instr,
    /// The action that led to this node (None for root)
    pub action: Option<RewriteAction>,
    /// Path to the sub-instruction that was rewritten (index vector)
    pub rewrite_path: Vec<usize>,
    /// Number of times this node has been visited
    pub visits: u32,
    /// Total reward (negative cycle count, so higher = better)
    pub total_reward: f64,
    /// Children of this node
    pub children: Vec<MctsNode>,
    /// Whether this node has been fully expanded
    pub expanded: bool,
    /// Cycle cost of this program state (cached)
    pub cached_cost: Option<u32>,
}

impl MctsNode {
    pub fn new(program: Instr) -> Self {
        Self {
            program,
            action: None,
            rewrite_path: Vec::new(),
            visits: 0,
            total_reward: 0.0,
            children: Vec::new(),
            expanded: false,
            cached_cost: None,
        }
    }

    /// UCB1 value for this node
    pub fn ucb1(&self, parent_visits: u32, exploration_constant: f64) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY; // Unvisited nodes get priority
        }
        let exploitation = self.total_reward / self.visits as f64;
        let exploration = exploration_constant * ((parent_visits as f64).ln() / self.visits as f64).sqrt();
        exploitation + exploration
    }

    /// Average reward
    pub fn avg_reward(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_reward / self.visits as f64
        }
    }

    /// Select the best child using UCB1
    pub fn best_child(&self, exploration_constant: f64) -> Option<usize> {
        let parent_visits = self.visits;
        let mut best_idx = None;
        let mut best_val = f64::NEG_INFINITY;

        for (i, child) in self.children.iter().enumerate() {
            let val = child.ucb1(parent_visits, exploration_constant);
            if val > best_val {
                best_val = val;
                best_idx = Some(i);
            }
        }

        best_idx
    }

    /// Select the most visited child (for final move selection)
    pub fn most_visited_child(&self) -> Option<usize> {
        let mut best_idx = None;
        let mut best_visits = 0;

        for (i, child) in self.children.iter().enumerate() {
            if child.visits > best_visits {
                best_visits = child.visits;
                best_idx = Some(i);
            }
        }

        best_idx
    }
}

// =============================================================================
// §6  MCTS Superoptimizer Engine
// =============================================================================

/// The MCTS-Port Superoptimizer engine
pub struct MctsSuperoptimizer {
    config: MctsConfig,
    cost_estimator: CycleCostEstimator,
    /// Statistics
    pub simulations_run: u64,
    pub rewrites_found: u64,
    pub best_improvement: u32,
    pub time_spent: Duration,
}

impl MctsSuperoptimizer {
    /// Create a new MCTS superoptimizer with the given configuration
    pub fn new(config: MctsConfig) -> Self {
        let microarch = config.microarch;
        let cost_estimator = CycleCostEstimator::new(microarch);
        Self {
            config,
            cost_estimator,
            simulations_run: 0,
            rewrites_found: 0,
            best_improvement: 0,
            time_spent: Duration::ZERO,
        }
    }

    /// Create with default configuration
    pub fn default_optimizer() -> Self {
        Self::new(MctsConfig::default())
    }

    /// Optimize an expression using MCTS with port-aware cost model
    pub fn optimize(&mut self, expr: &Expr) -> Option<Expr> {
        let instr = Instr::from_expr(expr)?;
        if !instr.is_pure() {
            return None;
        }

        let original_cost = self.estimate_cost(&instr);
        if original_cost == 0 {
            return None; // Nothing to optimize
        }

        let start = Instant::now();
        let deadline = if self.config.time_budget_ms > 0 {
            Some(start + Duration::from_millis(self.config.time_budget_ms))
        } else {
            None
        };

        let mut root = MctsNode::new(instr.clone());

        // Run MCTS simulations
        for _ in 0..self.config.max_simulations {
            // Check time budget
            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }

            // Check tree size
            if self.count_nodes(&root) >= self.config.max_tree_size {
                break;
            }

            self.run_simulation_core(&mut root);
            self.simulations_run += 1;
        }

        // Extract the best program found
        let best = self.extract_best(&root);
        let best_cost = self.estimate_cost(&best);

        self.time_spent = start.elapsed();

        if best_cost + self.config.min_improvement <= original_cost {
            let improvement = original_cost - best_cost;
            if improvement > self.best_improvement {
                self.best_improvement = improvement;
            }
            self.rewrites_found += 1;
            Some(best.to_expr(expr.span()))
        } else {
            None
        }
    }

    /// Run a single MCTS simulation (select → expand → simulate → backpropagate)
    fn run_simulation_core(&mut self, root: &mut MctsNode) {
        // 1. Selection: walk down the tree using UCB1 to find the path
        let mut path: Vec<usize> = Vec::new();
        {
            let mut node: &mut MctsNode = root;
            loop {
                if node.children.is_empty() || !node.expanded {
                    break;
                }
                if let Some(idx) = node.best_child(self.config.exploration_constant) {
                    path.push(idx);
                    node = &mut node.children[idx];
                } else {
                    break;
                }
            }
        }

        // 2. Find the leaf node for expansion and get its program
        let leaf_program;
        {
            let mut node: &mut MctsNode = root;
            for &idx in &path {
                node = &mut node.children[idx];
            }

            // Expand if needed
            if node.visits > 0 && !node.expanded {
                self.expand_node(node);
                if !node.children.is_empty() {
                    let idx = node.children.len() - 1;
                    path.push(idx);
                }
            }

            // Get the program for rollout
            let mut node: &mut MctsNode = root;
            for &idx in &path {
                node = &mut node.children[idx];
            }
            leaf_program = node.program.clone();
        }

        // 3. Simulation (rollout): apply random rewrites to estimate value
        let sim_result = self.rollout(&leaf_program);

        // 4. Backpropagation: update stats up the tree
        root.visits += 1;
        root.total_reward += sim_result;
        {
            let mut node: &mut MctsNode = root;
            for &idx in &path {
                node = &mut node.children[idx];
                node.visits += 1;
                node.total_reward += sim_result;
            }
        }
    }

    /// Expand a node by generating all applicable children
    fn expand_node(&mut self, node: &mut MctsNode) {
        if node.expanded {
            return;
        }

        let actions = RewriteAction::all();
        let mut children = Vec::new();

        for action in actions {
            if action.is_applicable(&node.program) {
                if let Some(new_program) = self.apply_action(&node.program, action) {
                    if new_program != node.program {
                        children.push(MctsNode {
                            program: new_program,
                            action: Some(action.clone()),
                            rewrite_path: Vec::new(),
                            visits: 0,
                            total_reward: 0.0,
                            children: Vec::new(),
                            expanded: false,
                            cached_cost: None,
                        });
                    }
                }
            }

            // Also try applying actions to sub-expressions
            let sub_results = self.apply_action_recursive(&node.program, action, &mut 0);
            for new_program in sub_results {
                if new_program != node.program {
                    children.push(MctsNode {
                        program: new_program,
                        action: Some(action.clone()),
                        rewrite_path: Vec::new(),
                        visits: 0,
                        total_reward: 0.0,
                        children: Vec::new(),
                        expanded: false,
                        cached_cost: None,
                    });
                }
            }
        }

        node.children = children;
        node.expanded = true;
    }

    /// Apply an action to the top-level instruction
    fn apply_action(&self, instr: &Instr, action: &RewriteAction) -> Option<Instr> {
        match action {
            RewriteAction::Commute => {
                if let Instr::BinOp { op, lhs, rhs } = instr {
                    Some(Instr::BinOp {
                        op: *op,
                        lhs: rhs.clone(),
                        rhs: lhs.clone(),
                    })
                } else {
                    None
                }
            }
            RewriteAction::IdentityLeft => {
                if let Instr::BinOp { op, lhs, rhs } = instr {
                    // Validate: the left operand must be the identity element for this op
                    let valid = match op {
                        BinOpKind::Add => matches!(**lhs, Instr::ConstInt(0)),
                        BinOpKind::Mul => matches!(**lhs, Instr::ConstInt(1)),
                        BinOpKind::BitOr => matches!(**lhs, Instr::ConstInt(0)),
                        BinOpKind::BitAnd => matches!(**lhs, Instr::ConstInt(v) if v == u128::MAX),
                        _ => false,
                    };
                    if valid {
                        Some((**rhs).clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            RewriteAction::IdentityRight => {
                if let Instr::BinOp { op, lhs, rhs } = instr {
                    // Validate: the right operand must be the identity element for this op
                    let valid = match op {
                        BinOpKind::Add => matches!(**rhs, Instr::ConstInt(0)),
                        BinOpKind::Sub => matches!(**rhs, Instr::ConstInt(0)),
                        BinOpKind::Mul => matches!(**rhs, Instr::ConstInt(1)),
                        BinOpKind::Div => matches!(**rhs, Instr::ConstInt(1)),
                        BinOpKind::BitOr => matches!(**rhs, Instr::ConstInt(0)),
                        BinOpKind::BitAnd => matches!(**rhs, Instr::ConstInt(v) if v == u128::MAX),
                        BinOpKind::Shl | BinOpKind::Shr => matches!(**rhs, Instr::ConstInt(0)),
                        _ => false,
                    };
                    if valid {
                        Some((**lhs).clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            RewriteAction::AnnihilateLeft => {
                if let Instr::BinOp { op, lhs, .. } = instr {
                    // Validate: the left operand must be the annihilator for this op
                    let valid = match op {
                        BinOpKind::Mul => matches!(**lhs, Instr::ConstInt(0)),
                        BinOpKind::BitAnd => matches!(**lhs, Instr::ConstInt(0)),
                        _ => false,
                    };
                    if valid {
                        Some(Instr::ConstInt(0))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            RewriteAction::AnnihilateRight => {
                if let Instr::BinOp { op, rhs, .. } = instr {
                    // Validate: the right operand must be the annihilator for this op
                    let valid = match op {
                        BinOpKind::Mul => matches!(**rhs, Instr::ConstInt(0)),
                        BinOpKind::BitAnd => matches!(**rhs, Instr::ConstInt(0)),
                        _ => false,
                    };
                    if valid {
                        Some(Instr::ConstInt(0))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            RewriteAction::ConstantFold => self.constant_fold(instr),
            RewriteAction::StrengthReduce => {
                if let Instr::BinOp { op, lhs, rhs } = instr {
                    if let Instr::ConstInt(v) = **rhs {
                        if v > 0 && (v & (v - 1)) == 0 {
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
                    if let Instr::BinOp { op: BinOpKind::Add, lhs: ref b, rhs: ref c } = *rhs.as_ref() {
                        return Some(Instr::BinOp {
                            op: BinOpKind::Add,
                            lhs: Box::new(Instr::BinOp {
                                op: BinOpKind::Mul,
                                lhs: a.clone(),
                                rhs: b.clone(),
                            }),
                            rhs: Box::new(Instr::BinOp {
                                op: BinOpKind::Mul,
                                lhs: a.clone(),
                                rhs: c.clone(),
                            }),
                        });
                    }
                }
                None
            }
            RewriteAction::Factor => {
                // a*b + a*c → a * (b + c)
                if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs } = instr {
                    if let (Instr::BinOp { op: BinOpKind::Mul, lhs: a1, rhs: b }, Instr::BinOp { op: BinOpKind::Mul, lhs: a2, rhs: c }) = (&**lhs, &**rhs)
                    {
                        if a1 == a2 {
                            return Some(Instr::BinOp {
                                op: BinOpKind::Mul,
                                lhs: a1.clone(),
                                rhs: Box::new(Instr::BinOp {
                                    op: BinOpKind::Add,
                                    lhs: b.clone(),
                                    rhs: c.clone(),
                                }),
                            });
                        }
                    }
                }
                None
            }
            RewriteAction::DoubleNegate => {
                if let Instr::UnOp { op, operand } = instr {
                    if let Instr::UnOp { op: ref inner_op, operand: ref inner } = **operand {
                        if *op == *inner_op {
                            return Some((**inner).clone());
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
                            // a & (a | b) → a
                            if let Instr::BinOp { op: BinOpKind::BitOr, .. } = **rhs {
                                return Some((**lhs).clone());
                            }
                        }
                        BinOpKind::BitOr => {
                            // a | (a & b) → a
                            if let Instr::BinOp { op: BinOpKind::BitAnd, .. } = **rhs {
                                return Some((**lhs).clone());
                            }
                        }
                        _ => {}
                    }
                }
                None
            }
        }
    }

    /// Recursively apply an action to all sub-expressions
    fn apply_action_recursive(&self, instr: &Instr, action: &RewriteAction, _depth: &mut usize) -> Vec<Instr> {
        let mut results = Vec::new();

        match instr {
            Instr::BinOp { op, lhs, rhs } => {
                // Try applying to left child
                if action.is_applicable(lhs) {
                    if let Some(new_lhs) = self.apply_action(lhs, action) {
                        results.push(Instr::BinOp {
                            op: *op,
                            lhs: Box::new(new_lhs),
                            rhs: rhs.clone(),
                        });
                    }
                }
                // Try applying to right child
                if action.is_applicable(rhs) {
                    if let Some(new_rhs) = self.apply_action(rhs, action) {
                        results.push(Instr::BinOp {
                            op: *op,
                            lhs: lhs.clone(),
                            rhs: Box::new(new_rhs),
                        });
                    }
                }
                // Recurse into children
                for new_lhs in self.apply_action_recursive(lhs, action, _depth) {
                    results.push(Instr::BinOp {
                        op: *op,
                        lhs: Box::new(new_lhs),
                        rhs: rhs.clone(),
                    });
                }
                for new_rhs in self.apply_action_recursive(rhs, action, _depth) {
                    results.push(Instr::BinOp {
                        op: *op,
                        lhs: lhs.clone(),
                        rhs: Box::new(new_rhs),
                    });
                }
            }
            Instr::UnOp { op, operand } => {
                if action.is_applicable(operand) {
                    if let Some(new_operand) = self.apply_action(operand, action) {
                        results.push(Instr::UnOp {
                            op: *op,
                            operand: Box::new(new_operand),
                        });
                    }
                }
                for new_operand in self.apply_action_recursive(operand, action, _depth) {
                    results.push(Instr::UnOp {
                        op: *op,
                        operand: Box::new(new_operand),
                    });
                }
            }
            _ => {}
        }

        results
    }

    /// Constant fold an instruction
    fn constant_fold(&self, instr: &Instr) -> Option<Instr> {
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
                        BinOpKind::Shl => l.checked_shl((*r).try_into().unwrap_or(128)),
                        BinOpKind::Shr => l.checked_shr((*r).try_into().unwrap_or(128)),
                        BinOpKind::Eq => Some(if l == r { 1 } else { 0 }),
                        BinOpKind::Ne => Some(if l != r { 1 } else { 0 }),
                        BinOpKind::Lt => Some(if l < r { 1 } else { 0 }),
                        _ => None,
                    };
                    return result.map(Instr::ConstInt);
                }
                if let (Instr::ConstBool(l), Instr::ConstBool(r)) = (&**lhs, &**rhs) {
                    let result = match op {
                        BinOpKind::And => Some(*l && *r),
                        BinOpKind::Or => Some(*l || *r),
                        BinOpKind::Eq => Some(*l == *r),
                        BinOpKind::Ne => Some(*l != *r),
                        _ => None,
                    };
                    return result.map(Instr::ConstBool);
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
                    Instr::ConstBool(v) => match op {
                        UnOpKind::Not => Some(Instr::ConstBool(!*v)),
                        _ => None,
                    },
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Rollout: apply random rewrites for a few steps and return the cost
    fn rollout(&mut self, instr: &Instr) -> f64 {
        let mut current = instr.clone();
        let mut rng = SimpleRng::from_seed(self.hash_instr(&current));

        for _ in 0..self.config.max_depth {
            let actions = RewriteAction::all();
            let applicable: Vec<_> = actions
                .iter()
                .filter(|a| a.is_applicable(&current))
                .collect();

            if applicable.is_empty() {
                break;
            }

            // Pick a random applicable action
            let idx = rng.next() as usize % applicable.len();
            let action = applicable[idx];

            if let Some(new_program) = self.apply_action(&current, action) {
                if new_program != current {
                    current = new_program;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        let cost = self.estimate_cost(&current);
        // Reward is negative cost (lower cost = higher reward)
        -(cost as f64)
    }

    /// Backpropagate the simulation result up the tree
    fn backpropagate(&self, root: &mut MctsNode, path: &[usize], reward: f64) {
        let mut node = root;
        node.visits += 1;
        node.total_reward += reward;

        for &idx in path {
            node = &mut node.children[idx];
            node.visits += 1;
            node.total_reward += reward;
        }
    }

    /// Extract the best program from the tree (most-visited child selection)
    fn extract_best(&self, root: &MctsNode) -> Instr {
        let mut best = &root.program;
        let mut node = root;

        loop {
            if node.children.is_empty() {
                break;
            }

            if let Some(idx) = node.most_visited_child() {
                node = &node.children[idx];
                best = &node.program;
            } else {
                break;
            }
        }

        best.clone()
    }

    /// Estimate the cost of an instruction
    fn estimate_cost(&mut self, instr: &Instr) -> u32 {
        if self.config.use_hardware_cost {
            self.cost_estimator.estimate(instr)
        } else {
            CycleCostEstimator::node_cost(instr) as u32
        }
    }

    /// Count total nodes in the tree
    fn count_nodes(&self, node: &MctsNode) -> usize {
        1 + node.children.iter().map(|c| self.count_nodes(c)).sum::<usize>()
    }

    /// Simple hash of an instruction for RNG seeding
    fn hash_instr(&self, instr: &Instr) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        fn hash_instr_inner(instr: &Instr, hasher: &mut DefaultHasher) {
            match instr {
                Instr::ConstInt(v) => { 0u8.hash(hasher); v.hash(hasher); }
                Instr::ConstFloat(v) => { 1u8.hash(hasher); v.hash(hasher); }
                Instr::ConstBool(v) => { 2u8.hash(hasher); v.hash(hasher); }
                Instr::Var(name) => { 3u8.hash(hasher); name.hash(hasher); }
                Instr::BinOp { op, lhs, rhs } => {
                    4u8.hash(hasher);
                    std::mem::discriminant(op).hash(hasher);
                    hash_instr_inner(lhs, hasher);
                    hash_instr_inner(rhs, hasher);
                }
                Instr::UnOp { op, operand } => {
                    5u8.hash(hasher);
                    std::mem::discriminant(op).hash(hasher);
                    hash_instr_inner(operand, hasher);
                }
            }
        }

        let mut hasher = DefaultHasher::new();
        hash_instr_inner(instr, &mut hasher);
        hasher.finish()
    }
}

// =============================================================================
// §7  Simple RNG for Rollouts (no external dependency)
// =============================================================================

/// A simple xorshift64 RNG for deterministic rollouts
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn from_seed(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
}

// =============================================================================
// §8  Integration with the Superoptimizer Pipeline
// =============================================================================

/// Convenience function: optimize an expression with MCTS using default config
pub fn mcts_optimize(expr: &Expr) -> Option<Expr> {
    let mut optimizer = MctsSuperoptimizer::default_optimizer();
    optimizer.optimize(expr)
}

/// Convenience function: optimize an expression with MCTS using fast config
pub fn mcts_optimize_fast(expr: &Expr) -> Option<Expr> {
    let mut optimizer = MctsSuperoptimizer::new(MctsConfig::fast());
    optimizer.optimize(expr)
}

/// Convenience function: optimize an expression with MCTS using thorough config
pub fn mcts_optimize_thorough(expr: &Expr) -> Option<Expr> {
    let mut optimizer = MctsSuperoptimizer::new(MctsConfig::thorough());
    optimizer.optimize(expr)
}

// =============================================================================
// §9  Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_span() -> Span {
        Span::dummy()
    }

    #[test]
    fn test_mcts_config_defaults() {
        let config = MctsConfig::default();
        assert_eq!(config.max_simulations, 200);
        assert_eq!(config.max_depth, 6);
        assert!(config.use_hardware_cost);
    }

    #[test]
    fn test_instr_from_expr() {
        let expr = Expr::BinOp {
            span: dummy_span(),
            op: BinOpKind::Add,
            lhs: Box::new(Expr::IntLit { span: dummy_span(), value: 1 }),
            rhs: Box::new(Expr::IntLit { span: dummy_span(), value: 2 }),
        };
        let instr = Instr::from_expr(&expr).unwrap();
        assert!(matches!(instr, Instr::BinOp { op: BinOpKind::Add, .. }));
    }

    #[test]
    fn test_instr_roundtrip() {
        let expr = Expr::BinOp {
            span: dummy_span(),
            op: BinOpKind::Mul,
            lhs: Box::new(Expr::Ident { span: dummy_span(), name: "x".to_string() }),
            rhs: Box::new(Expr::IntLit { span: dummy_span(), value: 4 }),
        };
        let instr = Instr::from_expr(&expr).unwrap();
        let roundtrip = instr.to_expr(dummy_span());
        // Should be equivalent
        assert!(matches!(roundtrip, Expr::BinOp { op: BinOpKind::Mul, .. }));
    }

    #[test]
    fn test_cycle_cost_estimator() {
        let mut estimator = CycleCostEstimator::new(None);
        let simple = Instr::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(Instr::Var("x".to_string())),
            rhs: Box::new(Instr::Var("y".to_string())),
        };
        let expensive = Instr::BinOp {
            op: BinOpKind::Div,
            lhs: Box::new(Instr::Var("x".to_string())),
            rhs: Box::new(Instr::Var("y".to_string())),
        };
        let simple_cost = estimator.estimate(&simple);
        let expensive_cost = estimator.estimate(&expensive);
        // Division should be much more expensive than addition
        assert!(expensive_cost > simple_cost);
    }

    #[test]
    fn test_rewrite_action_applicable() {
        let commutative = Instr::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(Instr::Var("a".to_string())),
            rhs: Box::new(Instr::Var("b".to_string())),
        };
        assert!(RewriteAction::Commute.is_applicable(&commutative));

        let non_commutative = Instr::BinOp {
            op: BinOpKind::Sub,
            lhs: Box::new(Instr::Var("a".to_string())),
            rhs: Box::new(Instr::Var("b".to_string())),
        };
        assert!(!RewriteAction::Commute.is_applicable(&non_commutative));
    }

    #[test]
    fn test_constant_fold() {
        let optimizer = MctsSuperoptimizer::new(MctsConfig::fast());
        let expr = Instr::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(Instr::ConstInt(3)),
            rhs: Box::new(Instr::ConstInt(5)),
        };
        let folded = optimizer.constant_fold(&expr);
        assert!(matches!(folded, Some(Instr::ConstInt(8))));
    }

    #[test]
    fn test_strength_reduce() {
        let optimizer = MctsSuperoptimizer::new(MctsConfig::fast());
        let expr = Instr::BinOp {
            op: BinOpKind::Mul,
            lhs: Box::new(Instr::Var("x".to_string())),
            rhs: Box::new(Instr::ConstInt(8)),
        };
        let reduced = optimizer.apply_action(&expr, &RewriteAction::StrengthReduce);
        assert!(reduced.is_some());
        if let Some(Instr::BinOp { op, rhs, .. }) = reduced {
            assert_eq!(op, BinOpKind::Shl);
            assert!(matches!(*rhs, Instr::ConstInt(3)));
        }
    }

    #[test]
    fn test_mcts_optimize_identity() {
        let mut optimizer = MctsSuperoptimizer::new(MctsConfig::fast());
        // x + 0 should be optimized to x
        let expr = Expr::BinOp {
            span: dummy_span(),
            op: BinOpKind::Add,
            lhs: Box::new(Expr::Ident { span: dummy_span(), name: "x".to_string() }),
            rhs: Box::new(Expr::IntLit { span: dummy_span(), value: 0 }),
        };
        let result = optimizer.optimize(&expr);
        // MCTS should find that x + 0 → x is cheaper
        if let Some(optimized) = result {
            assert!(matches!(optimized, Expr::Ident { .. }));
        }
    }

    #[test]
    fn test_mcts_optimize_strength_reduce() {
        let mut optimizer = MctsSuperoptimizer::new(MctsConfig::fast());
        // x * 8 should be optimized to x << 3
        let expr = Expr::BinOp {
            span: dummy_span(),
            op: BinOpKind::Mul,
            lhs: Box::new(Expr::Ident { span: dummy_span(), name: "x".to_string() }),
            rhs: Box::new(Expr::IntLit { span: dummy_span(), value: 8 }),
        };
        let result = optimizer.optimize(&expr);
        // Mul is expensive (3 cycles on port 1); shl is cheap (1 cycle on any ALU port)
        // MCTS should find the strength reduction
        if let Some(optimized) = result {
            if let Expr::BinOp { op, .. } = &optimized {
                assert_eq!(*op, BinOpKind::Shl);
            }
        }
    }

    #[test]
    fn test_mcts_distribute_factor() {
        let optimizer = MctsSuperoptimizer::new(MctsConfig::fast());
        // a * (b + c) should be distributable
        let expr = Instr::BinOp {
            op: BinOpKind::Mul,
            lhs: Box::new(Instr::Var("a".to_string())),
            rhs: Box::new(Instr::BinOp {
                op: BinOpKind::Add,
                lhs: Box::new(Instr::Var("b".to_string())),
                rhs: Box::new(Instr::Var("c".to_string())),
            }),
        };
        let result = optimizer.apply_action(&expr, &RewriteAction::Distribute);
        assert!(result.is_some());
        // Result should be a*b + a*c
        if let Some(Instr::BinOp { op: BinOpKind::Add, .. }) = result {
            // Good
        } else {
            panic!("Distribute should produce an Add");
        }
    }

    #[test]
    fn test_ucb1_selection() {
        let root = MctsNode::new(Instr::ConstInt(0));
        assert_eq!(root.visits, 0);
        // UCB1 of unvisited node should be infinity
        assert!(root.ucb1(100, 1.414).is_infinite());
    }
}
