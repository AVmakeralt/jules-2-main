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
//   Selection = PUCT formula (balance explore vs exploit)
// =============================================================================

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

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
    /// PUCT exploration constant (c_puct)
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
            exploration_constant: 1.414, // c_puct
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
    pub fn local_only() -> Self {
        Self {
            max_simulations: 100,
            max_depth: 4,
            exploration_constant: 1.414,
            time_budget_ms: 20,
            min_improvement: 2,
            use_hardware_cost: true,
            microarch: None,
            verif_inputs: 8,
            max_tree_size: 5_000,
        }
    }

    /// Very local mode: optimized for hot-path kernels only.
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

    /// Compute a hash of the config fields to detect changes.
    /// Used to decide whether to recreate the MCTS optimizer or just reset it.
    pub fn config_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        self.max_simulations.hash(&mut hasher);
        self.max_depth.hash(&mut hasher);
        self.exploration_constant.to_bits().hash(&mut hasher);
        self.time_budget_ms.hash(&mut hasher);
        self.min_improvement.hash(&mut hasher);
        self.use_hardware_cost.hash(&mut hasher);
        self.verif_inputs.hash(&mut hasher);
        self.max_tree_size.hash(&mut hasher);
        hasher.finish()
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
// §2a  String Interner for Var(u32) — eliminates heap allocation per variable
// =============================================================================

/// S2 fix: A simple string interner that maps variable names to u32 indices.
///
/// Fix 1: Eliminates the UB from `transmute::<&str, &'static str>`.
/// The old `get()` used transmute which was UB because the thread-local
/// `Vec<String>` could reallocate, invalidating the returned reference.
/// Now uses `Box::leak` to safely obtain `&'static str`, and stores
/// `Arc<str>` internally for reference-counted safety against reallocation.
pub struct StringInterner;

std::thread_local! {
    static INTERN_STORAGE: std::cell::RefCell<Vec<Arc<str>>> = std::cell::RefCell::new(Vec::new());
    static INTERN_LEAKED: std::cell::RefCell<Vec<&'static str>> = std::cell::RefCell::new(Vec::new());
}

static INTERN_COUNTER: AtomicU32 = AtomicU32::new(0);

impl StringInterner {
    /// Intern a string and return its u32 index. If the string is already
    /// interned (by value equality), returns the existing index.
    pub fn intern(s: &str) -> u32 {
        INTERN_STORAGE.with(|storage| {
            let mut vec = storage.borrow_mut();
            // Linear scan is fine: variable count per expression is small (<50)
            for (i, existing) in vec.iter().enumerate() {
                if existing.as_ref() == s {
                    return i as u32;
                }
            }
            let idx = vec.len() as u32;
            vec.push(Arc::from(s));
            // Also store a leaked &'static str for backward-compatible get()
            let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
            INTERN_LEAKED.with(|leaked_vec| {
                leaked_vec.borrow_mut().push(leaked);
            });
            idx
        })
    }

    /// Retrieve the string for a given index as &'static str.
    /// Fix 1: Uses Box::leak instead of transmute for safe &'static str.
    /// The leaked allocation lives for the entire program, which is acceptable
    /// because variable names are bounded. This eliminates the UB from the
    /// old transmute approach where Vec reallocation could invalidate references.
    pub fn get(idx: u32) -> &'static str {
        INTERN_LEAKED.with(|leaked_vec| {
            let vec = leaked_vec.borrow();
            if (idx as usize) < vec.len() {
                vec[idx as usize] // Already &'static str via Box::leak, no transmute
            } else {
                "<invalid>"
            }
        })
    }

    /// Retrieve the Arc<str> for a given index.
    /// Arc<str> is reference-counted and safe against Vec reallocation.
    /// Use this when you need owned, cloneable string data.
    pub fn get_arc(idx: u32) -> Arc<str> {
        INTERN_STORAGE.with(|storage| {
            let vec = storage.borrow();
            if (idx as usize) < vec.len() {
                Arc::clone(&vec[idx as usize])
            } else {
                Arc::from("<invalid>")
            }
        })
    }
}

// =============================================================================
// §2b  Program Representation for MCTS
// =============================================================================

/// A compact instruction representation for the MCTS search space.
#[derive(Debug, Clone, PartialEq)]
pub enum Instr {
    /// Load a constant integer value
    ConstInt(u128),
    /// Load a constant float value (raw bits)
    ConstFloat(u64),
    /// Load a constant boolean
    ConstBool(bool),
    /// Reference to a variable (by interned index into StringInterner)
    Var(u32),
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
            Expr::Ident { name, .. } => Some(Instr::Var(StringInterner::intern(name))),
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
            Instr::Var(idx) => Expr::Ident {
                span,
                name: StringInterner::get(*idx).to_string(),
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

    /// Collect all variables referenced in this instruction.
    /// Fix 8: Internally uses SmallVec<[u32; 4]> for dedup to avoid
    /// intermediate String allocation. Returns Vec<String> for backward
    /// compatibility with callers in other modules.
    pub fn variables(&self) -> Vec<String> {
        let mut var_set = HashSet::new();
        self.collect_variables_dedup(&mut var_set);
        var_set.into_iter().map(|idx| StringInterner::get(idx).to_string()).collect()
    }

    /// Collect all variables as interned u32 indices (no String allocation).
    /// Use this when you only need the indices, not the string names.
    pub fn variable_indices(&self) -> SmallVec<[u32; 4]> {
        let mut var_set = HashSet::new();
        self.collect_variables_dedup(&mut var_set);
        var_set.into_iter().collect()
    }

    fn collect_variables_dedup(&self, vars: &mut HashSet<u32>) {
        match self {
            Instr::Var(idx) => {
                vars.insert(*idx);
            }
            Instr::BinOp { lhs, rhs, .. } => {
                lhs.collect_variables_dedup(vars);
                rhs.collect_variables_dedup(vars);
            }
            Instr::UnOp { operand, .. } => {
                operand.collect_variables_dedup(vars);
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
                schedule.push("load");
            }
            Instr::Var(_) => {
                schedule.push("load");
            }
            Instr::BinOp { op: BinOpKind::Add, lhs, rhs } => {
                if let Instr::BinOp { op: BinOpKind::Shl, .. } = **lhs {
                    self.flatten_inner(lhs, schedule);
                    self.flatten_inner(rhs, schedule);
                    schedule.push("lea");
                    return;
                }
                if let Instr::BinOp { op: BinOpKind::Shl, .. } = **rhs {
                    self.flatten_inner(lhs, schedule);
                    self.flatten_inner(rhs, schedule);
                    schedule.push("lea");
                    return;
                }
                self.flatten_inner(lhs, schedule);
                self.flatten_inner(rhs, schedule);
                if let Instr::ConstInt(1) = **rhs {
                    schedule.push("inc");
                } else {
                    schedule.push("add");
                }
            }
            Instr::BinOp { op: BinOpKind::Sub, lhs, rhs } => {
                self.flatten_inner(lhs, schedule);
                self.flatten_inner(rhs, schedule);
                if let Instr::ConstInt(1) = **rhs {
                    schedule.push("dec");
                } else {
                    schedule.push("sub");
                }
            }
            Instr::BinOp { op, lhs, rhs } => {
                self.flatten_inner(lhs, schedule);
                self.flatten_inner(rhs, schedule);
                let instr_name = match op {
                    BinOpKind::Mul => "mul",
                    BinOpKind::Div => "div",
                    BinOpKind::Rem => "rem",
                    BinOpKind::BitAnd => "and",
                    BinOpKind::BitOr => "or",
                    BinOpKind::BitXor => "xor",
                    BinOpKind::Shl => "shl",
                    BinOpKind::Shr => "shr",
                    _ => "add",
                };
                schedule.push(instr_name);
            }
            Instr::UnOp { operand, .. } => {
                self.flatten_inner(operand, schedule);
                schedule.push("add");
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
    /// LEA combine: x + y*scale + offset → LEA (1 uop, 1 cycle)
    LeaCombine,
    /// INC/DEC: x + 1 → INC x, x - 1 → DEC x
    IncDec,
    /// CMOV select: if cond { a } else { b } → CMOVcc
    CmovSelect,
    /// Multiply by small constant → LEA sequence
    LeaMulSmallConst,
    // ── Fix 7: Tier 1 semantic rules ──
    /// Negate: x * (-1) → -x
    Negate,
    /// AddNegToSub: x + (-y) → x - y
    AddNegToSub,
    /// SubNegToAdd: x - (-y) → x + y
    SubNegToAdd,
    /// IdempotentAnd: x & x → x
    IdempotentAnd,
    /// IdempotentOr: x | x → x
    IdempotentOr,
    /// ZeroDiv: 0 / x → 0
    ZeroDiv,
    /// RemOne: x % 1 → 0
    RemOne,
    /// CmpNegate: !(x < y) → x >= y
    CmpNegate,
    /// AndOverOr: (x & y) | (x & z) → x & (y | z)
    AndOverOr,
    /// OrOverAnd: (x | y) & (x | z) → x | (y & z)
    OrOverAnd,
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
            RewriteAction::LeaCombine,
            RewriteAction::IncDec,
            RewriteAction::CmovSelect,
            RewriteAction::LeaMulSmallConst,
            // Fix 7: Tier 1 semantic rules
            RewriteAction::Negate,
            RewriteAction::AddNegToSub,
            RewriteAction::SubNegToAdd,
            RewriteAction::IdempotentAnd,
            RewriteAction::IdempotentOr,
            RewriteAction::ZeroDiv,
            RewriteAction::RemOne,
            RewriteAction::CmpNegate,
            RewriteAction::AndOverOr,
            RewriteAction::OrOverAnd,
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
                            v > 1 && (v & (v - 1)) == 0
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
                if let Instr::BinOp { op: BinOpKind::Mul, rhs, .. } = instr {
                    matches!(**rhs, Instr::BinOp { op: BinOpKind::Add, .. })
                } else {
                    false
                }
            }
            RewriteAction::Factor => {
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
                if let Instr::BinOp { op, lhs, rhs } = instr {
                    match op {
                        BinOpKind::BitAnd => {
                            if let Instr::BinOp { op: BinOpKind::BitOr, lhs: ref inner_lhs, .. } = **rhs {
                                lhs == inner_lhs
                            } else {
                                false
                            }
                        }
                        BinOpKind::BitOr => {
                            if let Instr::BinOp { op: BinOpKind::BitAnd, lhs: ref inner_lhs, .. } = **rhs {
                                lhs == inner_lhs
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
            RewriteAction::LeaCombine => {
                if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Shl, .. } = **lhs { return true; }
                    if let Instr::BinOp { op: BinOpKind::Shl, .. } = **rhs { return true; }
                }
                false
            }
            RewriteAction::IncDec => {
                if let Instr::BinOp { op, rhs, .. } = instr {
                    match op {
                        BinOpKind::Add => matches!(**rhs, Instr::ConstInt(1)),
                        BinOpKind::Sub => matches!(**rhs, Instr::ConstInt(1)),
                        _ => false,
                    }
                } else {
                    false
                }
            }
            RewriteAction::CmovSelect => {
                // Fix 2: Detect comparison expressions that are CMOV candidates.
                // Since the current Instr representation doesn't have an If/Cond
                // variant, we detect comparison BinOps (Lt, Gt, Le, Ge, Eq, Ne)
                // which represent conditional patterns that could be lowered to CMOV.
                matches!(instr,
                    Instr::BinOp { op: BinOpKind::Lt, .. }
                    | Instr::BinOp { op: BinOpKind::Gt, .. }
                    | Instr::BinOp { op: BinOpKind::Le, .. }
                    | Instr::BinOp { op: BinOpKind::Ge, .. }
                    | Instr::BinOp { op: BinOpKind::Eq, .. }
                    | Instr::BinOp { op: BinOpKind::Ne, .. }
                )
            }
            RewriteAction::LeaMulSmallConst => {
                if let Instr::BinOp { op: BinOpKind::Mul, rhs, .. } = instr {
                    if let Instr::ConstInt(v) = **rhs {
                        matches!(v, 3 | 5 | 9 | 7)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            // ── Fix 7: Tier 1 semantic rules ──
            RewriteAction::Negate => {
                // x * (-1) → -x
                if let Instr::BinOp { op: BinOpKind::Mul, rhs, .. } = instr {
                    matches!(**rhs, Instr::UnOp { op: UnOpKind::Neg, .. })
                } else {
                    false
                }
            }
            RewriteAction::AddNegToSub => {
                if let Instr::BinOp { op: BinOpKind::Add, rhs, .. } = instr {
                    matches!(**rhs, Instr::UnOp { op: UnOpKind::Neg, .. })
                } else {
                    false
                }
            }
            RewriteAction::SubNegToAdd => {
                if let Instr::BinOp { op: BinOpKind::Sub, rhs, .. } = instr {
                    matches!(**rhs, Instr::UnOp { op: UnOpKind::Neg, .. })
                } else {
                    false
                }
            }
            RewriteAction::IdempotentAnd => {
                if let Instr::BinOp { op, lhs, rhs } = instr {
                    *op == BinOpKind::BitAnd && lhs == rhs
                } else {
                    false
                }
            }
            RewriteAction::IdempotentOr => {
                if let Instr::BinOp { op, lhs, rhs } = instr {
                    *op == BinOpKind::BitOr && lhs == rhs
                } else {
                    false
                }
            }
            RewriteAction::ZeroDiv => {
                if let Instr::BinOp { op: BinOpKind::Div, lhs, .. } = instr {
                    matches!(**lhs, Instr::ConstInt(0))
                } else {
                    false
                }
            }
            RewriteAction::RemOne => {
                if let Instr::BinOp { op: BinOpKind::Rem, rhs, .. } = instr {
                    matches!(**rhs, Instr::ConstInt(1))
                } else {
                    false
                }
            }
            RewriteAction::CmpNegate => {
                if let Instr::UnOp { op: UnOpKind::Not, operand } = instr {
                    matches!(**operand, Instr::BinOp { op: BinOpKind::Lt, .. } | Instr::BinOp { op: BinOpKind::Gt, .. })
                } else {
                    false
                }
            }
            RewriteAction::AndOverOr => {
                if let Instr::BinOp { op: BinOpKind::BitOr, lhs, rhs } = instr {
                    if let (Instr::BinOp { op: BinOpKind::BitAnd, lhs: a1, .. },
                            Instr::BinOp { op: BinOpKind::BitAnd, lhs: a2, .. }) = (&**lhs, &**rhs) {
                        a1 == a2
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            RewriteAction::OrOverAnd => {
                if let Instr::BinOp { op: BinOpKind::BitAnd, lhs, rhs } = instr {
                    if let (Instr::BinOp { op: BinOpKind::BitOr, lhs: a1, .. },
                            Instr::BinOp { op: BinOpKind::BitOr, lhs: a2, .. }) = (&**lhs, &**rhs) {
                        a1 == a2
                    } else {
                        false
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
    /// Fix 5: Policy prior from GNN policy head (defaults to uniform 1/N)
    pub policy_prior: Option<f64>,
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
            policy_prior: None,
        }
    }

    /// UCB1 value for this node (kept for reference, selection now uses PUCT)
    pub fn ucb1(&self, parent_visits: u32, exploration_constant: f64) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY;
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

    /// Fix 5: Select the best child using PUCT (Predictor + Upper Confidence Bound)
    /// Replaces UCB1 selection for better exploration with policy priors.
    pub fn best_child(&self, c_puct: f64) -> Option<usize> {
        if self.children.is_empty() {
            return None;
        }
        let total_visits = self.children.iter().map(|c| c.visits).sum::<u32>().max(1);
        let sqrt_parent = (total_visits as f64).sqrt();

        let best = self.children.iter().enumerate().map(|(i, c)| {
            let q = if c.visits > 0 { c.total_reward / c.visits as f64 } else { 0.0 };
            let p = c.policy_prior.unwrap_or(1.0 / self.children.len() as f64);
            let u = c_puct * p * sqrt_parent / (1.0 + c.visits as f64);
            (i, q + u)
        }).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        best.map(|(i, _)| i)
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
// §4b  Transposition Table (Fix 4)
// =============================================================================

/// An entry in the transposition table for reusing search results
/// across nodes with identical program states.
#[derive(Debug, Clone)]
struct TransEntry {
    visits: u32,
    total_reward: f64,
    best_action: Option<usize>,
}

// =============================================================================
// §6  MCTS Superoptimizer Engine
// =============================================================================

/// The MCTS-Port Superoptimizer engine
pub struct MctsSuperoptimizer {
    config: MctsConfig,
    cost_estimator: CycleCostEstimator,
    /// S1 fix: O(1) node count instead of O(N) tree walk every simulation
    node_count: usize,
    /// Fix 4: Transposition table for reusing search results across
    /// nodes with identical program hashes.
    transposition_table: FxHashMap<u64, TransEntry>,
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
            node_count: 0,
            transposition_table: FxHashMap::default(),
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

    /// Reset internal state for reuse across expressions (S10 fix).
    pub fn reset_for_new_expr(&mut self) {
        self.node_count = 0;
        self.simulations_run = 0;
        self.rewrites_found = 0;
        self.best_improvement = 0;
        self.time_spent = Duration::ZERO;
        self.clear_transposition_table(); // Fix 4
    }

    /// Return the config hash so callers can detect config changes
    /// and decide whether to recreate the optimizer or just reset it.
    pub fn config_hash(&self) -> u64 {
        self.config.config_hash()
    }

    /// Fix 4: Clear the transposition table
    fn clear_transposition_table(&mut self) {
        self.transposition_table.clear();
    }

    /// Optimize an expression using MCTS with port-aware cost model
    pub fn optimize(&mut self, expr: &Expr) -> Option<Expr> {
        let instr = Instr::from_expr(expr)?;
        if !instr.is_pure() {
            return None;
        }

        let original_cost = self.estimate_cost(&instr);
        if original_cost == 0 {
            return None;
        }

        let start = Instant::now();
        let deadline = if self.config.time_budget_ms > 0 {
            Some(start + Duration::from_millis(self.config.time_budget_ms))
        } else {
            None
        };

        let mut root = MctsNode::new(instr.clone());
        self.node_count = 1;

        // Run MCTS simulations
        for _ in 0..self.config.max_simulations {
            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }
            if self.node_count >= self.config.max_tree_size {
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
        // 1. Selection: walk down the tree using PUCT
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

        // Fix 4: Check transposition table before rollout
        let leaf_hash = self.hash_instr(&leaf_program);
        let sim_result = if let Some(entry) = self.transposition_table.get(&leaf_hash) {
            if entry.visits > 0 {
                entry.total_reward / entry.visits as f64
            } else {
                self.rollout(&leaf_program)
            }
        } else {
            self.rollout(&leaf_program)
        };

        // Fix 4: Update transposition table after simulation
        let entry = self.transposition_table.entry(leaf_hash).or_insert(TransEntry {
            visits: 0,
            total_reward: 0.0,
            best_action: None,
        });
        entry.visits += 1;
        entry.total_reward += sim_result;

        // 4. Backpropagation
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
        let mut seen: HashSet<u64> = HashSet::new();
        let num_actions = actions.len();

        for action in actions {
            if action.is_applicable(&node.program) {
                if let Some(new_program) = self.apply_action(&node.program, action) {
                    if new_program != node.program {
                        let h = self.hash_instr(&new_program);
                        if seen.insert(h) {
                            let policy_prior = Some(1.0 / num_actions as f64);
                            children.push(MctsNode {
                                program: new_program,
                                action: Some(action.clone()),
                                rewrite_path: Vec::new(),
                                visits: 0,
                                total_reward: 0.0,
                                children: Vec::new(),
                                expanded: false,
                                cached_cost: None,
                                policy_prior,
                            });
                        }
                    }
                }
            }

            // Fix 3: Pass depth counting down instead of always using max_depth
            let sub_results = self.apply_action_recursive(&node.program, action, self.config.max_depth);
            for new_program in sub_results {
                if new_program != node.program {
                    let h = self.hash_instr(&new_program);
                    if seen.insert(h) {
                        let policy_prior = Some(1.0 / num_actions as f64);
                        children.push(MctsNode {
                            program: new_program,
                            action: Some(action.clone()),
                            rewrite_path: Vec::new(),
                            visits: 0,
                            total_reward: 0.0,
                            children: Vec::new(),
                            expanded: false,
                            cached_cost: None,
                            policy_prior,
                        });
                    }
                }
            }
        }

        self.node_count += children.len();
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
                    let valid = match op {
                        BinOpKind::Add => matches!(**lhs, Instr::ConstInt(0)),
                        BinOpKind::Mul => matches!(**lhs, Instr::ConstInt(1)),
                        BinOpKind::BitOr => matches!(**lhs, Instr::ConstInt(0)),
                        BinOpKind::BitAnd => matches!(**lhs, Instr::ConstInt(v) if v == u128::MAX),
                        _ => false,
                    };
                    if valid { Some((**rhs).clone()) } else { None }
                } else { None }
            }
            RewriteAction::IdentityRight => {
                if let Instr::BinOp { op, lhs, rhs } = instr {
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
                    if valid { Some((**lhs).clone()) } else { None }
                } else { None }
            }
            RewriteAction::AnnihilateLeft => {
                if let Instr::BinOp { op, lhs, .. } = instr {
                    let valid = match op {
                        BinOpKind::Mul => matches!(**lhs, Instr::ConstInt(0)),
                        BinOpKind::BitAnd => matches!(**lhs, Instr::ConstInt(0)),
                        _ => false,
                    };
                    if valid { Some(Instr::ConstInt(0)) } else { None }
                } else { None }
            }
            RewriteAction::AnnihilateRight => {
                if let Instr::BinOp { op, rhs, .. } = instr {
                    let valid = match op {
                        BinOpKind::Mul => matches!(**rhs, Instr::ConstInt(0)),
                        BinOpKind::BitAnd => matches!(**rhs, Instr::ConstInt(0)),
                        _ => false,
                    };
                    if valid { Some(Instr::ConstInt(0)) } else { None }
                } else { None }
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
                    if let Instr::UnOp { op: ref inner_op, operand: ref inner } = **operand {
                        if *op == *inner_op { return Some((**inner).clone()); }
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
                            if let Instr::BinOp { op: BinOpKind::BitOr, lhs: ref inner_lhs, .. } = **rhs {
                                if lhs == inner_lhs { return Some((**lhs).clone()); }
                            }
                        }
                        BinOpKind::BitOr => {
                            if let Instr::BinOp { op: BinOpKind::BitAnd, lhs: ref inner_lhs, .. } = **rhs {
                                if lhs == inner_lhs { return Some((**lhs).clone()); }
                            }
                        }
                        _ => {}
                    }
                }
                None
            }
            RewriteAction::LeaCombine => {
                if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Shl, .. } = **lhs { return Some(instr.clone()); }
                    if let Instr::BinOp { op: BinOpKind::Shl, .. } = **rhs { return Some(instr.clone()); }
                }
                None
            }
            RewriteAction::IncDec => {
                if let Instr::BinOp { op, lhs: _, rhs } = instr {
                    match op {
                        BinOpKind::Add if matches!(**rhs, Instr::ConstInt(1)) => return Some(instr.clone()),
                        BinOpKind::Sub if matches!(**rhs, Instr::ConstInt(1)) => return Some(instr.clone()),
                        _ => {}
                    }
                }
                None
            }
            RewriteAction::CmovSelect => {
                // Fix 2: Mark comparison expressions for CMOV lowering (cost model hint)
                if matches!(instr, Instr::BinOp { op: BinOpKind::Lt | BinOpKind::Gt | BinOpKind::Le | BinOpKind::Ge | BinOpKind::Eq | BinOpKind::Ne, .. }) {
                    return Some(instr.clone());
                }
                None
            }
            RewriteAction::LeaMulSmallConst => {
                if let Instr::BinOp { op: BinOpKind::Mul, lhs, rhs } = instr {
                    if let Instr::ConstInt(v) = **rhs {
                        match v {
                            3 => return Some(Instr::BinOp {
                                op: BinOpKind::Add,
                                lhs: Box::new(Instr::BinOp { op: BinOpKind::Shl, lhs: lhs.clone(), rhs: Box::new(Instr::ConstInt(1)) }),
                                rhs: lhs.clone(),
                            }),
                            5 => return Some(Instr::BinOp {
                                op: BinOpKind::Add,
                                lhs: Box::new(Instr::BinOp { op: BinOpKind::Shl, lhs: lhs.clone(), rhs: Box::new(Instr::ConstInt(2)) }),
                                rhs: lhs.clone(),
                            }),
                            9 => return Some(Instr::BinOp {
                                op: BinOpKind::Add,
                                lhs: Box::new(Instr::BinOp { op: BinOpKind::Shl, lhs: lhs.clone(), rhs: Box::new(Instr::ConstInt(3)) }),
                                rhs: lhs.clone(),
                            }),
                            7 => return Some(Instr::BinOp {
                                op: BinOpKind::Sub,
                                lhs: Box::new(Instr::BinOp { op: BinOpKind::Shl, lhs: lhs.clone(), rhs: Box::new(Instr::ConstInt(3)) }),
                                rhs: lhs.clone(),
                            }),
                            _ => {}
                        }
                    }
                }
                None
            }
            // ── Fix 7: Tier 1 semantic rule apply implementations ──
            RewriteAction::Negate => {
                if let Instr::BinOp { op: BinOpKind::Mul, lhs, rhs } = instr {
                    if let Instr::UnOp { op: UnOpKind::Neg, .. } = &**rhs {
                        return Some(Instr::UnOp { op: UnOpKind::Neg, operand: lhs.clone() });
                    }
                }
                None
            }
            RewriteAction::AddNegToSub => {
                if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs } = instr {
                    if let Instr::UnOp { op: UnOpKind::Neg, operand } = &**rhs {
                        return Some(Instr::BinOp { op: BinOpKind::Sub, lhs: lhs.clone(), rhs: operand.clone() });
                    }
                }
                None
            }
            RewriteAction::SubNegToAdd => {
                if let Instr::BinOp { op: BinOpKind::Sub, lhs, rhs } = instr {
                    if let Instr::UnOp { op: UnOpKind::Neg, operand } = &**rhs {
                        return Some(Instr::BinOp { op: BinOpKind::Add, lhs: lhs.clone(), rhs: operand.clone() });
                    }
                }
                None
            }
            RewriteAction::IdempotentAnd => {
                if let Instr::BinOp { op: BinOpKind::BitAnd, lhs, rhs } = instr {
                    if lhs == rhs { return Some((**lhs).clone()); }
                }
                None
            }
            RewriteAction::IdempotentOr => {
                if let Instr::BinOp { op: BinOpKind::BitOr, lhs, rhs } = instr {
                    if lhs == rhs { return Some((**lhs).clone()); }
                }
                None
            }
            RewriteAction::ZeroDiv => {
                if let Instr::BinOp { op: BinOpKind::Div, lhs, .. } = instr {
                    if let Instr::ConstInt(0) = **lhs { return Some(Instr::ConstInt(0)); }
                }
                None
            }
            RewriteAction::RemOne => {
                if let Instr::BinOp { op: BinOpKind::Rem, rhs, .. } = instr {
                    if let Instr::ConstInt(1) = **rhs { return Some(Instr::ConstInt(0)); }
                }
                None
            }
            RewriteAction::CmpNegate => {
                if let Instr::UnOp { op: UnOpKind::Not, operand } = instr {
                    if let Instr::BinOp { op, lhs, rhs } = &**operand {
                        let new_op = match op {
                            BinOpKind::Lt => BinOpKind::Ge,
                            BinOpKind::Gt => BinOpKind::Le,
                            _ => return None,
                        };
                        return Some(Instr::BinOp { op: new_op, lhs: lhs.clone(), rhs: rhs.clone() });
                    }
                }
                None
            }
            RewriteAction::AndOverOr => {
                if let Instr::BinOp { op: BinOpKind::BitOr, lhs, rhs } = instr {
                    if let (Instr::BinOp { op: BinOpKind::BitAnd, lhs: a, rhs: b },
                            Instr::BinOp { op: BinOpKind::BitAnd, lhs: a2, rhs: c }) = (&**lhs, &**rhs) {
                        if a == a2 {
                            return Some(Instr::BinOp {
                                op: BinOpKind::BitAnd,
                                lhs: a.clone(),
                                rhs: Box::new(Instr::BinOp { op: BinOpKind::BitOr, lhs: b.clone(), rhs: c.clone() }),
                            });
                        }
                    }
                }
                None
            }
            RewriteAction::OrOverAnd => {
                if let Instr::BinOp { op: BinOpKind::BitAnd, lhs, rhs } = instr {
                    if let (Instr::BinOp { op: BinOpKind::BitOr, lhs: a, rhs: b },
                            Instr::BinOp { op: BinOpKind::BitOr, lhs: a2, rhs: c }) = (&**lhs, &**rhs) {
                        if a == a2 {
                            return Some(Instr::BinOp {
                                op: BinOpKind::BitOr,
                                lhs: a.clone(),
                                rhs: Box::new(Instr::BinOp { op: BinOpKind::BitAnd, lhs: b.clone(), rhs: c.clone() }),
                            });
                        }
                    }
                }
                None
            }
        }
    }

    /// Recursively apply an action to all sub-expressions.
    /// Fix 3: depth counts DOWN. Base case when depth == 0.
    /// The recursive call passes `depth - 1` so each level reduces
    /// the remaining depth budget, preventing the old bug where every
    /// recursive call used the full max_depth.
    fn apply_action_recursive(&self, instr: &Instr, action: &RewriteAction, depth: usize) -> Vec<Instr> {
        // Fix 3: Base case — stop recursing when depth reaches 0
        if depth == 0 {
            return Vec::new();
        }

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
                // Recurse into children with depth - 1
                for new_lhs in self.apply_action_recursive(lhs, action, depth - 1) {
                    results.push(Instr::BinOp {
                        op: *op,
                        lhs: Box::new(new_lhs),
                        rhs: rhs.clone(),
                    });
                }
                for new_rhs in self.apply_action_recursive(rhs, action, depth - 1) {
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
                        results.push(Instr::UnOp { op: *op, operand: Box::new(new_operand) });
                    }
                }
                for new_operand in self.apply_action_recursive(operand, action, depth - 1) {
                    results.push(Instr::UnOp { op: *op, operand: Box::new(new_operand) });
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

    /// Fix 6: Rollout with value network prediction when available.
    /// Instead of always applying random rewrites for N steps, use the GNN
    /// value head's prediction if available, falling back to random rollout.
    fn rollout(&mut self, instr: &Instr) -> f64 {
        // Fix 6: Try GNN value prediction first (when gnn-optimizer feature is enabled)
        #[cfg(feature = "gnn-optimizer")]
        {
            if let Some(ref gnn) = self.gnn {
                let value = gnn.predict_value(instr);
                return value;
            }
        }

        // Fallback: random rollout
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
    fn count_nodes(&self, _node: &MctsNode) -> usize {
        self.node_count
    }

    /// Simple hash of an instruction for RNG seeding / transposition table
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
        match instr {
            Instr::BinOp { op: BinOpKind::Add, .. } => {}
            _ => panic!("Expected BinOp::Add"),
        }
    }

    #[test]
    fn test_string_interner_safe() {
        let idx = StringInterner::intern("test_var");
        let s = StringInterner::get(idx);
        assert_eq!(s, "test_var");
        // Box::leak ensures the string lives forever (no UB from transmute)
        for i in 0..100 {
            StringInterner::intern(&format!("var_{}", i));
        }
        // Original value still valid after many internments
        assert_eq!(StringInterner::get(idx), "test_var");
        // Arc version also works
        let arc = StringInterner::get_arc(idx);
        assert_eq!(&*arc, "test_var");
    }

    #[test]
    fn test_variables_returns_indices() {
        let x_idx = StringInterner::intern("x");
        let y_idx = StringInterner::intern("y");
        let instr = Instr::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(Instr::Var(x_idx)),
            rhs: Box::new(Instr::Var(y_idx)),
        };
        // variables() returns Vec<String> for backward compat
        let vars = instr.variables();
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));
        assert_eq!(vars.len(), 2);
        // variable_indices() returns SmallVec<[u32; 4]> without String allocation
        let indices = instr.variable_indices();
        assert!(indices.contains(&x_idx));
        assert!(indices.contains(&y_idx));
        assert_eq!(indices.len(), 2);
    }

    #[test]
    fn test_cmov_select_detects_comparisons() {
        // Fix 2: CmovSelect detects comparison BinOps
        let lt = Instr::BinOp {
            op: BinOpKind::Lt,
            lhs: Box::new(Instr::Var(StringInterner::intern("x"))),
            rhs: Box::new(Instr::ConstInt(0)),
        };
        assert!(RewriteAction::CmovSelect.is_applicable(&lt));

        let eq = Instr::BinOp {
            op: BinOpKind::Eq,
            lhs: Box::new(Instr::Var(StringInterner::intern("x"))),
            rhs: Box::new(Instr::ConstInt(0)),
        };
        assert!(RewriteAction::CmovSelect.is_applicable(&eq));

        // Should not match non-comparison
        let add = Instr::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(Instr::ConstInt(1)),
            rhs: Box::new(Instr::ConstInt(2)),
        };
        assert!(!RewriteAction::CmovSelect.is_applicable(&add));
    }

    #[test]
    fn test_apply_action_recursive_depth() {
        let config = MctsConfig::fast();
        let optimizer = MctsSuperoptimizer::new(config);
        // Deeply nested: ((x + 0) + 0) + 0
        let instr = Instr::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(Instr::BinOp {
                op: BinOpKind::Add,
                lhs: Box::new(Instr::BinOp {
                    op: BinOpKind::Add,
                    lhs: Box::new(Instr::Var(StringInterner::intern("x"))),
                    rhs: Box::new(Instr::ConstInt(0)),
                }),
                rhs: Box::new(Instr::ConstInt(0)),
            }),
            rhs: Box::new(Instr::ConstInt(0)),
        };
        // With depth=1, should only find top-level IdentityRight
        let results = optimizer.apply_action_recursive(&instr, &RewriteAction::IdentityRight, 1);
        assert!(results.len() <= 3, "depth=1 should limit recursion");

        // With depth=0, should find nothing
        let results = optimizer.apply_action_recursive(&instr, &RewriteAction::IdentityRight, 0);
        assert!(results.is_empty(), "depth=0 should return empty");
    }

    #[test]
    fn test_tier1_negate() {
        let x = Instr::Var(StringInterner::intern("x"));
        let instr = Instr::BinOp {
            op: BinOpKind::Mul,
            lhs: Box::new(x.clone()),
            rhs: Box::new(Instr::UnOp {
                op: UnOpKind::Neg,
                operand: Box::new(Instr::ConstInt(1)),
            }),
        };
        assert!(RewriteAction::Negate.is_applicable(&instr));
    }

    #[test]
    fn test_tier1_add_neg_to_sub() {
        let instr = Instr::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(Instr::Var(StringInterner::intern("x"))),
            rhs: Box::new(Instr::UnOp {
                op: UnOpKind::Neg,
                operand: Box::new(Instr::Var(StringInterner::intern("y"))),
            }),
        };
        assert!(RewriteAction::AddNegToSub.is_applicable(&instr));
    }

    #[test]
    fn test_tier1_rem_one() {
        let instr = Instr::BinOp {
            op: BinOpKind::Rem,
            lhs: Box::new(Instr::Var(StringInterner::intern("x"))),
            rhs: Box::new(Instr::ConstInt(1)),
        };
        assert!(RewriteAction::RemOne.is_applicable(&instr));
    }

    #[test]
    fn test_transposition_table() {
        let mut optimizer = MctsSuperoptimizer::new(MctsConfig::fast());
        assert!(optimizer.transposition_table.is_empty());
        optimizer.reset_for_new_expr();
        assert!(optimizer.transposition_table.is_empty());
    }

    #[test]
    fn test_puct_selection() {
        let mut root = MctsNode::new(Instr::ConstInt(0));
        root.visits = 10;
        // Add two children with different policy priors
        root.children.push(MctsNode {
            program: Instr::ConstInt(1),
            action: None,
            rewrite_path: Vec::new(),
            visits: 5,
            total_reward: -10.0,
            children: Vec::new(),
            expanded: false,
            cached_cost: None,
            policy_prior: Some(0.7), // Higher prior
        });
        root.children.push(MctsNode {
            program: Instr::ConstInt(2),
            action: None,
            rewrite_path: Vec::new(),
            visits: 5,
            total_reward: -10.0,
            children: Vec::new(),
            expanded: false,
            cached_cost: None,
            policy_prior: Some(0.3), // Lower prior
        });
        // With equal visits but different priors, PUCT should prefer higher prior
        let best = root.best_child(1.414);
        assert_eq!(best, Some(0)); // Child 0 has higher policy prior
    }
}
