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

    /// Instruction shape category for dispatch table filtering.
    /// Categorizes instructions into broad shape classes for fast
    /// pre-filtering of applicable rewrite actions.
    pub fn shape(&self) -> InstrShape {
        match self {
            Instr::ConstInt(_) | Instr::ConstFloat(_) | Instr::ConstBool(_) => InstrShape::Const,
            Instr::Var(_) => InstrShape::Var,
            Instr::BinOp { .. } => InstrShape::BinOp,
            Instr::UnOp { .. } => InstrShape::UnOp,
        }
    }
}

/// Instruction shape categories for dispatch table filtering.
/// Used to quickly narrow down which rewrite actions could possibly
/// apply to a given instruction without running full pattern matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstrShape {
    /// Constant values (ConstInt, ConstFloat, ConstBool)
    Const,
    /// Variable references
    Var,
    /// Binary operations
    BinOp,
    /// Unary operations
    UnOp,
}

// =============================================================================
// §3a  Action Dispatch Table — Pre-filtering for Performance
// =============================================================================

/// Pre-computed dispatch table mapping instruction shapes to applicable actions.
/// Avoids iterating over all actions for every node during expansion.
pub struct ActionDispatchTable {
    table: HashMap<InstrShape, Vec<RewriteAction>>,
}

impl ActionDispatchTable {
    /// Build the dispatch table by testing each action against representative
    /// instructions of each shape category.
    pub fn build() -> Self {
        let mut table = HashMap::new();
        for shape in [InstrShape::Const, InstrShape::Var, InstrShape::BinOp, InstrShape::UnOp] {
            let test_instr = match shape {
                InstrShape::Const => Instr::ConstInt(0),
                InstrShape::Var => Instr::Var(0),
                InstrShape::BinOp => Instr::BinOp {
                    op: BinOpKind::Add,
                    lhs: Box::new(Instr::Var(0)),
                    rhs: Box::new(Instr::Var(1)),
                },
                InstrShape::UnOp => Instr::UnOp {
                    op: UnOpKind::Neg,
                    operand: Box::new(Instr::Var(0)),
                },
            };
            let applicable: Vec<RewriteAction> = RewriteAction::all().iter()
                .filter(|a| a.is_applicable(&test_instr))
                .cloned()
                .collect();
            table.insert(shape, applicable);
        }
        Self { table }
    }

    /// Returns a slice of actions that MIGHT be applicable for this shape.
    /// The caller still needs to call is_applicable() for exact checking
    /// because shape matching is an over-approximation.
    pub fn get_applicable(&self, instr: &Instr) -> &[RewriteAction] {
        self.table.get(&instr.shape()).map(|v| v.as_slice()).unwrap_or(&[])
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
    // ── Tier 2: Arithmetic and Bit-Level Algebra ──
    /// SubReassoc1: a - (b + c) -> (a - b) - c
    SubReassoc1,
    /// SubReassoc2: a - (b - c) -> (a - b) + c
    SubReassoc2,
    /// AddSubCancel: (a + b) - b -> a
    AddSubCancel,
    /// MulPow2Add: x * 2 + x -> x * 3 (LEA pattern)
    MulPow2Add,
    /// MulDistSub: a * (b - c) -> a*b - a*c
    MulDistSub,
    /// NegSwap: -(a - b) -> b - a
    NegSwap,
    /// DoubleNegAdd: -(-a + b) -> a - b
    DoubleNegAdd,
    /// AddZeroLeft: 0 + a -> a
    AddZeroLeft,
    /// ShlByConstAdd: (x << k) + x -> x * (2^k + 1)
    ShlByConstAdd,
    /// ShrIsDivPow2: x >> k -> x / 2^k (semantic equivalence for cost model)
    ShrIsDivPow2,
    // ── Tier 2: Bit Manipulation Rules ──
    /// AndNotComplement: a & (!b) -> ANDN(a, b)
    AndNotComplement,
    /// AndComplement: a & (~b) -> ANDN(a, b)
    AndComplement,
    /// XorSwap: a ^ b ^ b -> a (eliminate XOR-swap residue)
    XorSwap,
    /// XorAllOnes: a ^ ~0 -> ~a (XOR with all-ones is NOT)
    XorAllOnes,
    /// AndMaskLow: x & ((1 << k) - 1) -> extract low k bits
    AndMaskLow,
    /// AndMaskHigh: x & ~((1 << k) - 1) -> clear low k bits
    AndMaskHigh,
    /// IsolateLowest: x & (-x) -> BLSI(x)
    IsolateLowest,
    /// ClearLowest: x & (x - 1) -> BLSR(x)
    ClearLowest,
    /// MaskMerge: (a & mask) | (b & ~mask) -> BFI(a, b, mask)
    MaskMerge,
    // ── Tier 3: Comparison and Conditional Rules ──
    /// CmpNegateFull: !(a <= b) -> a > b and !(a >= b) -> a < b
    CmpNegateFull,
    /// DoubleNegCmp: !!(a < b) -> a < b
    DoubleNegCmp,
    /// EqNormalize: a == b -> b == a (when b is const)
    EqNormalize,
    /// NeNormalize: a != b -> b != a (when b is const)
    NeNormalize,
    /// SubIsZero: (a - b) == 0 -> a == b
    SubIsZero,
    /// SubIsNonZero: (a - b) != 0 -> a != b
    SubIsNonZero,
    /// LeFromLt: a < b + 1 -> a <= b
    LeFromLt,
    /// GeFromGt: a > b - 1 -> a >= b
    GeFromGt,
    /// AndCmps: (a > 0) & (a < N) -> range check pattern
    AndCmps,
    // ── Tier 3: Division and Remainder Optimization ──
    /// DivByConst3: x / 3 -> MULH(x, magic3)
    DivByConst3,
    /// DivByConst5: x / 5 -> MULH(x, magic5)
    DivByConst5,
    /// DivByConst7: x / 7 -> MULH(x, magic7)
    DivByConst7,
    /// DivByConstN: x / N -> MULH(x, magic(N)) — generic for any non-power-of-2 constant > 1
    DivByConstN,
    /// RemByConst3: x % 3 -> x - (x/3)*3
    RemByConst3,
    /// RemByConstPow2: x % 2^k -> x & (2^k - 1)
    RemByConstPow2,
    /// RemToMask: x % N -> x - (x/N)*N for const N
    RemToMask,
    /// DivNegNeg: (-a) / (-b) -> a / b
    DivNegNeg,
    /// DivSignAdjust: x / (-N) -> -(x / N)
    DivSignAdjust,
    /// UnsignedDivPow2: x >>> k -> x / 2^k (logical shift)
    UnsignedDivPow2,
    // ── Tier 4: Multi-Step and Architectural Rules ──
    /// Lea3Op: base + index*scale + offset -> LEA(base, index, scale, offset)
    Lea3Op,
    /// LeaScaleAdd: x*3 -> LEA(x, x*2), x*5 -> LEA(x, x*4), etc.
    LeaScaleAdd,
    /// TestInsteadOfAnd: (x & mask) == 0 -> TEST(x, mask)
    TestInsteadOfAnd,
    /// TestInsteadOfAndNZ: (x & mask) != 0 -> TEST(x, mask) + SETNE
    TestInsteadOfAndNZ,
    /// SetccFromCmp: (a < b) ? 1 : 0 -> SETcc
    SetccFromCmp,
    /// CmovFromSelect: if (c) { a } else { b } -> CMOVcc
    CmovFromSelect,
    /// CmovFromCmpOp: c ? x + y : x -> CMOVcc(c, x+y, x)
    CmovFromCmpOp,
    /// SbbFromBorrow: a - b - carry -> SBB(a, b, carry)
    SbbFromBorrow,
    /// AdcFromCarry: a + b + carry -> ADC(a, b, carry)
    AdcFromCarry,
    /// XorZero: x ^ x -> 0 (register zeroing)
    XorZero,
    /// MovZero: 0 -> XOR(reg, reg)
    MovZero,
    /// RotateRight: (x >> k) | (x << (N-k)) -> ROR(x, k)
    RotateRight,
    /// RotateLeft: (x << k) | (x >> (N-k)) -> ROL(x, k)
    RotateLeft,
    /// BswapPattern: byte-reverse -> BSWAP
    BswapPattern,
    /// PopcntPattern: bit count -> POPCNT
    PopcntPattern,
    /// LzcntPattern: leading zero count -> LZCNT
    LzcntPattern,
    /// TzcntPattern: trailing zero count -> TZCNT
    TzcntPattern,
    // ── Tier 5: Reassociation, Cancellation, and Memory-Form Rules ──
    /// AddReassocConst: (a + C1) + C2 -> a + (C1 + C2) — reassociate constants to fold them
    AddReassocConst,
    /// MulReassocConst: (a * C1) * C2 -> a * (C1 * C2) — reassociate constant multiplies
    MulReassocConst,
    /// AddNegReassoc: (a + b) - a -> b — cancel a + b - a
    AddNegReassoc,
    /// SubSelfCancel: a - (a + b) -> -b — cancel subtraction of self
    SubSelfCancel,
    /// MulAddDistribute: a*b + a*c -> a * (b + c) — factor out common multiplier (saves 1 MUL)
    MulAddDistribute,
    /// MulSubDistribute: a*b - a*c -> a * (b - c) — factor out common multiplier
    MulSubDistribute,
    /// NegateMul: -(a * b) -> (-a) * b — enables further opts
    NegateMul,
    /// AddSubSwap: (a + b) - c -> a + (b - c) — exposes CSE
    AddSubSwap,
    /// SubAddMerge: a - b + b -> a — cancel subtraction then addition
    SubAddMerge,
    /// BlsmaskRule: a ^ (a - 1) -> BLSMSK(a) — bit manipulation pattern
    BlsmaskRule,
    /// MemFormFold: fold binary op with memory-load second operand into memory-form opcode
    MemFormFold,
    /// FlagReuse: eliminate redundant CMP when same operands are compared again
    FlagReuse,
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
            // Tier 2: Arithmetic and Bit-Level Algebra
            RewriteAction::SubReassoc1,
            RewriteAction::SubReassoc2,
            RewriteAction::AddSubCancel,
            RewriteAction::MulPow2Add,
            RewriteAction::MulDistSub,
            RewriteAction::NegSwap,
            RewriteAction::DoubleNegAdd,
            RewriteAction::AddZeroLeft,
            RewriteAction::ShlByConstAdd,
            RewriteAction::ShrIsDivPow2,
            // Tier 2: Bit Manipulation Rules
            RewriteAction::AndNotComplement,
            RewriteAction::AndComplement,
            RewriteAction::XorSwap,
            RewriteAction::XorAllOnes,
            RewriteAction::AndMaskLow,
            RewriteAction::AndMaskHigh,
            RewriteAction::IsolateLowest,
            RewriteAction::ClearLowest,
            RewriteAction::MaskMerge,
            // Tier 3: Comparison and Conditional Rules
            RewriteAction::CmpNegateFull,
            RewriteAction::DoubleNegCmp,
            RewriteAction::EqNormalize,
            RewriteAction::NeNormalize,
            RewriteAction::SubIsZero,
            RewriteAction::SubIsNonZero,
            RewriteAction::LeFromLt,
            RewriteAction::GeFromGt,
            RewriteAction::AndCmps,
            // Tier 3: Division and Remainder Optimization
            RewriteAction::DivByConst3,
            RewriteAction::DivByConst5,
            RewriteAction::DivByConst7,
            RewriteAction::DivByConstN,
            RewriteAction::RemByConst3,
            RewriteAction::RemByConstPow2,
            RewriteAction::RemToMask,
            RewriteAction::DivNegNeg,
            RewriteAction::DivSignAdjust,
            RewriteAction::UnsignedDivPow2,
            // Tier 4: Multi-Step and Architectural Rules
            RewriteAction::Lea3Op,
            RewriteAction::LeaScaleAdd,
            RewriteAction::TestInsteadOfAnd,
            RewriteAction::TestInsteadOfAndNZ,
            RewriteAction::SetccFromCmp,
            RewriteAction::CmovFromSelect,
            RewriteAction::CmovFromCmpOp,
            RewriteAction::SbbFromBorrow,
            RewriteAction::AdcFromCarry,
            RewriteAction::XorZero,
            RewriteAction::MovZero,
            RewriteAction::RotateRight,
            RewriteAction::RotateLeft,
            RewriteAction::BswapPattern,
            RewriteAction::PopcntPattern,
            RewriteAction::LzcntPattern,
            RewriteAction::TzcntPattern,
            // Tier 5: Reassociation, Cancellation, and Memory-Form Rules
            RewriteAction::AddReassocConst,
            RewriteAction::MulReassocConst,
            RewriteAction::AddNegReassoc,
            RewriteAction::SubSelfCancel,
            RewriteAction::MulAddDistribute,
            RewriteAction::MulSubDistribute,
            RewriteAction::NegateMul,
            RewriteAction::AddSubSwap,
            RewriteAction::SubAddMerge,
            RewriteAction::BlsmaskRule,
            RewriteAction::MemFormFold,
            RewriteAction::FlagReuse,
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
            // ── Tier 2: Arithmetic and Bit-Level Algebra ──
            RewriteAction::SubReassoc1 => {
                // a - (b + c) -> (a - b) - c
                if let Instr::BinOp { op: BinOpKind::Sub, rhs, .. } = instr {
                    matches!(**rhs, Instr::BinOp { op: BinOpKind::Add, .. })
                } else { false }
            }
            RewriteAction::SubReassoc2 => {
                // a - (b - c) -> (a - b) + c
                if let Instr::BinOp { op: BinOpKind::Sub, rhs, .. } = instr {
                    matches!(**rhs, Instr::BinOp { op: BinOpKind::Sub, .. })
                } else { false }
            }
            RewriteAction::AddSubCancel => {
                // (a + b) - b -> a
                if let Instr::BinOp { op: BinOpKind::Sub, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Add, rhs: ref add_rhs, .. } = **lhs {
                        add_rhs == rhs
                    } else { false }
                } else { false }
            }
            RewriteAction::MulPow2Add => {
                // x * 2 + x -> x * 3 (LEA pattern)
                if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs } = instr {
                    // Match x * 2 + x
                    if let Instr::BinOp { op: BinOpKind::Mul, lhs: ref mul_lhs, rhs: ref mul_rhs } = &**lhs {
                        if let Instr::ConstInt(2) = **mul_rhs { return mul_lhs == rhs; }
                    }
                    // Match x + x * 2
                    if let Instr::BinOp { op: BinOpKind::Mul, lhs: ref mul_lhs, rhs: ref mul_rhs } = &**rhs {
                        if let Instr::ConstInt(2) = **mul_rhs { return mul_lhs == lhs; }
                    }
                    // Also match x << 1 + x
                    if let Instr::BinOp { op: BinOpKind::Shl, lhs: ref shift_lhs, rhs: ref shift_rhs } = &**lhs {
                        if let Instr::ConstInt(1) = **shift_rhs { return shift_lhs == rhs; }
                    }
                    // Match x + x << 1
                    if let Instr::BinOp { op: BinOpKind::Shl, lhs: ref shift_lhs, rhs: ref shift_rhs } = &**rhs {
                        if let Instr::ConstInt(1) = **shift_rhs { return shift_lhs == lhs; }
                    }
                    false
                } else { false }
            }
            RewriteAction::MulDistSub => {
                // a * (b - c) -> a*b - a*c
                if let Instr::BinOp { op: BinOpKind::Mul, rhs, .. } = instr {
                    matches!(**rhs, Instr::BinOp { op: BinOpKind::Sub, .. })
                } else { false }
            }
            RewriteAction::NegSwap => {
                // -(a - b) -> b - a
                if let Instr::UnOp { op: UnOpKind::Neg, operand } = instr {
                    matches!(**operand, Instr::BinOp { op: BinOpKind::Sub, .. })
                } else { false }
            }
            RewriteAction::DoubleNegAdd => {
                // -(-a + b) -> a - b
                if let Instr::UnOp { op: UnOpKind::Neg, operand } = instr {
                    if let Instr::BinOp { op: BinOpKind::Add, lhs, .. } = &**operand {
                        matches!(**lhs, Instr::UnOp { op: UnOpKind::Neg, .. })
                    } else { false }
                } else { false }
            }
            RewriteAction::AddZeroLeft => {
                // 0 + a -> a
                if let Instr::BinOp { op: BinOpKind::Add, lhs, .. } = instr {
                    matches!(**lhs, Instr::ConstInt(0))
                } else { false }
            }
            RewriteAction::ShlByConstAdd => {
                // (x << k) + x -> x * (2^k + 1)
                if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Shl, lhs: ref shift_lhs, rhs: ref shift_rhs } = &**lhs {
                        if matches!(**shift_rhs, Instr::ConstInt(_)) { return shift_lhs == rhs; }
                    }
                    if let Instr::BinOp { op: BinOpKind::Shl, lhs: ref shift_lhs, rhs: ref shift_rhs } = &**rhs {
                        if matches!(**shift_rhs, Instr::ConstInt(_)) { return shift_lhs == lhs; }
                    }
                    false
                } else { false }
            }
            RewriteAction::ShrIsDivPow2 => {
                // x >> k -> x / 2^k (semantic equivalence for cost model)
                matches!(instr, Instr::BinOp { op: BinOpKind::Shr, .. })
            }
            // ── Tier 2: Bit Manipulation Rules ──
            RewriteAction::AndNotComplement => {
                // a & (!b) -> ANDN(a, b)
                if let Instr::BinOp { op: BinOpKind::BitAnd, rhs, .. } = instr {
                    matches!(**rhs, Instr::UnOp { op: UnOpKind::Not, .. })
                } else { false }
            }
            RewriteAction::AndComplement => {
                // a & (~b) -> ANDN(a, b) — same but with bitwise NOT
                if let Instr::BinOp { op: BinOpKind::BitAnd, rhs, .. } = instr {
                    matches!(**rhs, Instr::UnOp { op: UnOpKind::Not, .. })
                } else { false }
            }
            RewriteAction::XorSwap => {
                // a ^ b ^ b -> a
                if let Instr::BinOp { op: BinOpKind::BitXor, lhs, rhs } = instr {
                    // Match (a ^ b) ^ b or a ^ (b ^ b)
                    if let Instr::BinOp { op: BinOpKind::BitXor, rhs: ref inner_rhs, .. } = **lhs {
                        inner_rhs == rhs
                    } else if let Instr::BinOp { op: BinOpKind::BitXor, lhs: ref inner_lhs, rhs: ref inner_rhs } = **rhs {
                        inner_lhs == inner_rhs
                    } else { false }
                } else { false }
            }
            RewriteAction::XorAllOnes => {
                // a ^ ~0 -> ~a
                if let Instr::BinOp { op: BinOpKind::BitXor, rhs, .. } = instr {
                    matches!(**rhs, Instr::ConstInt(v) if v == u128::MAX)
                } else { false }
            }
            RewriteAction::AndMaskLow => {
                // x & ((1 << k) - 1) -> extract low k bits
                if let Instr::BinOp { op: BinOpKind::BitAnd, rhs, .. } = instr {
                    if let Instr::BinOp { op: BinOpKind::Sub, lhs: ref sub_lhs, rhs: ref sub_rhs } = **rhs {
                        if let Instr::BinOp { op: BinOpKind::Shl, lhs: ref shl_lhs, .. } = &**sub_lhs {
                            matches!(**shl_lhs, Instr::ConstInt(1)) && matches!(**sub_rhs, Instr::ConstInt(1))
                        } else { false }
                    } else { false }
                } else { false }
            }
            RewriteAction::AndMaskHigh => {
                // x & ~((1 << k) - 1) -> clear low k bits
                if let Instr::BinOp { op: BinOpKind::BitAnd, rhs, .. } = instr {
                    if let Instr::UnOp { op: UnOpKind::Not, operand } = &**rhs {
                        if let Instr::BinOp { op: BinOpKind::Sub, lhs: ref sub_lhs, rhs: ref sub_rhs } = &**operand {
                            matches!(**sub_lhs, Instr::BinOp { op: BinOpKind::Shl, .. }) && matches!(**sub_rhs, Instr::ConstInt(1))
                        } else { false }
                    } else { false }
                } else { false }
            }
            RewriteAction::IsolateLowest => {
                // x & (-x) -> BLSI(x)
                if let Instr::BinOp { op: BinOpKind::BitAnd, lhs, rhs } = instr {
                    if let Instr::UnOp { op: UnOpKind::Neg, operand } = &**rhs {
                        operand == lhs
                    } else { false }
                } else { false }
            }
            RewriteAction::ClearLowest => {
                // x & (x - 1) -> BLSR(x)
                if let Instr::BinOp { op: BinOpKind::BitAnd, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Sub, lhs: ref sub_lhs, rhs: ref sub_rhs } = &**rhs {
                        sub_lhs == lhs && matches!(**sub_rhs, Instr::ConstInt(1))
                    } else { false }
                } else { false }
            }
            RewriteAction::MaskMerge => {
                // (a & mask) | (b & ~mask) -> BFI(a, b, mask)
                if let Instr::BinOp { op: BinOpKind::BitOr, lhs, rhs } = instr {
                    if let (Instr::BinOp { op: BinOpKind::BitAnd, rhs: ref mask1, .. },
                            Instr::BinOp { op: BinOpKind::BitAnd, rhs: ref mask2, .. }) = (&**lhs, &**rhs) {
                        if let Instr::UnOp { op: UnOpKind::Not, operand: ref not_operand } = &**mask2 {
                            mask1 == not_operand
                        } else { false }
                    } else { false }
                } else { false }
            }
            // ── Tier 3: Comparison and Conditional Rules ──
            RewriteAction::CmpNegateFull => {
                // !(a <= b) -> a > b and !(a >= b) -> a < b
                if let Instr::UnOp { op: UnOpKind::Not, operand } = instr {
                    matches!(**operand, Instr::BinOp { op: BinOpKind::Le | BinOpKind::Ge, .. })
                } else { false }
            }
            RewriteAction::DoubleNegCmp => {
                // !!(a < b) -> a < b
                if let Instr::UnOp { op: UnOpKind::Not, operand } = instr {
                    if let Instr::UnOp { op: UnOpKind::Not, .. } = &**operand {
                        true
                    } else { false }
                } else { false }
            }
            RewriteAction::EqNormalize => {
                // a == b -> b == a (when b is const)
                if let Instr::BinOp { op: BinOpKind::Eq, rhs, .. } = instr {
                    matches!(**rhs, Instr::ConstInt(_))
                } else { false }
            }
            RewriteAction::NeNormalize => {
                // a != b -> b != a (when b is const)
                if let Instr::BinOp { op: BinOpKind::Ne, rhs, .. } = instr {
                    matches!(**rhs, Instr::ConstInt(_))
                } else { false }
            }
            RewriteAction::SubIsZero => {
                // (a - b) == 0 -> a == b
                if let Instr::BinOp { op: BinOpKind::Eq, lhs, rhs } = instr {
                    matches!(**lhs, Instr::BinOp { op: BinOpKind::Sub, .. }) && matches!(**rhs, Instr::ConstInt(0))
                } else { false }
            }
            RewriteAction::SubIsNonZero => {
                // (a - b) != 0 -> a != b
                if let Instr::BinOp { op: BinOpKind::Ne, lhs, rhs } = instr {
                    matches!(**lhs, Instr::BinOp { op: BinOpKind::Sub, .. }) && matches!(**rhs, Instr::ConstInt(0))
                } else { false }
            }
            RewriteAction::LeFromLt => {
                // a < b + 1 -> a <= b
                if let Instr::BinOp { op: BinOpKind::Lt, rhs, .. } = instr {
                    if let Instr::BinOp { op: BinOpKind::Add, rhs: ref add_rhs, .. } = &**rhs {
                        matches!(**add_rhs, Instr::ConstInt(1))
                    } else { false }
                } else { false }
            }
            RewriteAction::GeFromGt => {
                // a > b - 1 -> a >= b
                if let Instr::BinOp { op: BinOpKind::Gt, rhs, .. } = instr {
                    if let Instr::BinOp { op: BinOpKind::Sub, rhs: ref sub_rhs, .. } = &**rhs {
                        matches!(**sub_rhs, Instr::ConstInt(1))
                    } else { false }
                } else { false }
            }
            RewriteAction::AndCmps => {
                // (a > 0) & (a < N) -> range check pattern
                if let Instr::BinOp { op: BinOpKind::BitAnd, lhs, rhs } = instr {
                    matches!(**lhs, Instr::BinOp { op: BinOpKind::Gt, .. }) && matches!(**rhs, Instr::BinOp { op: BinOpKind::Lt, .. })
                } else { false }
            }
            // ── Tier 3: Division and Remainder Optimization ──
            RewriteAction::DivByConst3 => {
                if let Instr::BinOp { op: BinOpKind::Div, rhs, .. } = instr {
                    matches!(**rhs, Instr::ConstInt(3))
                } else { false }
            }
            RewriteAction::DivByConst5 => {
                if let Instr::BinOp { op: BinOpKind::Div, rhs, .. } = instr {
                    matches!(**rhs, Instr::ConstInt(5))
                } else { false }
            }
            RewriteAction::DivByConst7 => {
                if let Instr::BinOp { op: BinOpKind::Div, rhs, .. } = instr {
                    matches!(**rhs, Instr::ConstInt(7))
                } else { false }
            }
            RewriteAction::DivByConstN => {
                if let Instr::BinOp { op: BinOpKind::Div, rhs, .. } = instr {
                    if let Instr::ConstInt(v) = **rhs {
                        v > 1 && (v & (v - 1)) != 0 // non-power-of-2 constant > 1
                    } else { false }
                } else { false }
            }
            RewriteAction::RemByConst3 => {
                if let Instr::BinOp { op: BinOpKind::Rem, rhs, .. } = instr {
                    matches!(**rhs, Instr::ConstInt(3))
                } else { false }
            }
            RewriteAction::RemByConstPow2 => {
                if let Instr::BinOp { op: BinOpKind::Rem, rhs, .. } = instr {
                    if let Instr::ConstInt(v) = **rhs {
                        v > 1 && (v & (v - 1)) == 0 // power-of-2 constant > 1
                    } else { false }
                } else { false }
            }
            RewriteAction::RemToMask => {
                if let Instr::BinOp { op: BinOpKind::Rem, rhs, .. } = instr {
                    matches!(**rhs, Instr::ConstInt(_))
                } else { false }
            }
            RewriteAction::DivNegNeg => {
                // (-a) / (-b) -> a / b
                if let Instr::BinOp { op: BinOpKind::Div, lhs, rhs } = instr {
                    matches!(**lhs, Instr::UnOp { op: UnOpKind::Neg, .. }) && matches!(**rhs, Instr::UnOp { op: UnOpKind::Neg, .. })
                } else { false }
            }
            RewriteAction::DivSignAdjust => {
                // x / (-N) -> -(x / N)
                if let Instr::BinOp { op: BinOpKind::Div, rhs, .. } = instr {
                    if let Instr::UnOp { op: UnOpKind::Neg, .. } = &**rhs { true }
                    else if let Instr::ConstInt(v) = **rhs { (v as i128) < 0 } else { false }
                } else { false }
            }
            RewriteAction::UnsignedDivPow2 => {
                // x >>> k -> x / 2^k
                if let Instr::BinOp { op: BinOpKind::Shr, rhs, .. } = instr {
                    if let Instr::ConstInt(v) = **rhs { v > 0 } else { false }
                } else { false }
            }
            // ── Tier 4: Multi-Step and Architectural Rules ──
            RewriteAction::Lea3Op => {
                // base + index*scale + offset -> LEA
                if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs } = instr {
                    let lhs_has_shl = matches!(**lhs, Instr::BinOp { op: BinOpKind::Shl, .. });
                    let rhs_has_shl = matches!(**rhs, Instr::BinOp { op: BinOpKind::Shl, .. });
                    // Match nested Add with Shl child
                    if let Instr::BinOp { op: BinOpKind::Add, .. } = **lhs { return true; }
                    if let Instr::BinOp { op: BinOpKind::Add, .. } = **rhs { return true; }
                    lhs_has_shl || rhs_has_shl
                } else { false }
            }
            RewriteAction::LeaScaleAdd => {
                // x*3, x*5, x*7, x*9, x*11, x*13 -> LEA sequences
                if let Instr::BinOp { op: BinOpKind::Mul, rhs, .. } = instr {
                    if let Instr::ConstInt(v) = **rhs {
                        matches!(v, 3 | 5 | 7 | 9 | 11 | 13)
                    } else { false }
                } else { false }
            }
            RewriteAction::TestInsteadOfAnd => {
                // (x & mask) == 0 -> TEST(x, mask)
                if let Instr::BinOp { op: BinOpKind::Eq, lhs, rhs } = instr {
                    matches!(**lhs, Instr::BinOp { op: BinOpKind::BitAnd, .. }) && matches!(**rhs, Instr::ConstInt(0))
                } else { false }
            }
            RewriteAction::TestInsteadOfAndNZ => {
                // (x & mask) != 0 -> TEST(x, mask) + SETNE
                if let Instr::BinOp { op: BinOpKind::Ne, lhs, rhs } = instr {
                    matches!(**lhs, Instr::BinOp { op: BinOpKind::BitAnd, .. }) && matches!(**rhs, Instr::ConstInt(0))
                } else { false }
            }
            RewriteAction::SetccFromCmp => {
                // (a < b) ? 1 : 0 -> SETcc
                if let Instr::BinOp { op, rhs, .. } = instr {
                    matches!(op, BinOpKind::Lt | BinOpKind::Gt | BinOpKind::Le | BinOpKind::Ge | BinOpKind::Eq | BinOpKind::Ne)
                        && matches!(**rhs, Instr::ConstInt(0) | Instr::ConstInt(1))
                } else { false }
            }
            RewriteAction::CmovFromSelect => {
                // if (c) { a } else { b } -> CMOVcc
                matches!(instr, Instr::BinOp { op: BinOpKind::Lt | BinOpKind::Gt | BinOpKind::Le | BinOpKind::Ge | BinOpKind::Eq | BinOpKind::Ne, .. })
            }
            RewriteAction::CmovFromCmpOp => {
                // c ? x + y : x -> CMOVcc(c, x+y, x) — match comparison BinOps
                matches!(instr, Instr::BinOp { op: BinOpKind::Lt | BinOpKind::Gt | BinOpKind::Le | BinOpKind::Ge | BinOpKind::Eq | BinOpKind::Ne, .. })
            }
            RewriteAction::SbbFromBorrow => {
                // a - b - carry -> SBB — match nested Sub
                if let Instr::BinOp { op: BinOpKind::Sub, lhs, .. } = instr {
                    matches!(**lhs, Instr::BinOp { op: BinOpKind::Sub, .. })
                } else { false }
            }
            RewriteAction::AdcFromCarry => {
                // a + b + carry -> ADC — match nested Add
                if let Instr::BinOp { op: BinOpKind::Add, lhs, .. } = instr {
                    matches!(**lhs, Instr::BinOp { op: BinOpKind::Add, .. })
                } else { false }
            }
            RewriteAction::XorZero => {
                // x ^ x -> 0 (register zeroing)
                if let Instr::BinOp { op: BinOpKind::BitXor, lhs, rhs } = instr {
                    lhs == rhs
                } else { false }
            }
            RewriteAction::MovZero => {
                // 0 -> XOR(reg, reg)
                matches!(instr, Instr::ConstInt(0))
            }
            RewriteAction::RotateRight => {
                // (x >> k) | (x << (N-k)) -> ROR(x, k)
                if let Instr::BinOp { op: BinOpKind::BitOr, lhs, rhs } = instr {
                    matches!(**lhs, Instr::BinOp { op: BinOpKind::Shr, .. }) && matches!(**rhs, Instr::BinOp { op: BinOpKind::Shl, .. })
                } else { false }
            }
            RewriteAction::RotateLeft => {
                // (x << k) | (x >> (N-k)) -> ROL(x, k)
                if let Instr::BinOp { op: BinOpKind::BitOr, lhs, rhs } = instr {
                    matches!(**lhs, Instr::BinOp { op: BinOpKind::Shl, .. }) && matches!(**rhs, Instr::BinOp { op: BinOpKind::Shr, .. })
                } else { false }
            }
            RewriteAction::BswapPattern => {
                // byte-reverse pattern detection (simplified)
                matches!(instr, Instr::BinOp { op: BinOpKind::BitOr, .. })
            }
            RewriteAction::PopcntPattern => {
                // bit count pattern (simplified — detect XOR with shift)
                matches!(instr, Instr::BinOp { op: BinOpKind::BitXor, .. })
            }
            RewriteAction::LzcntPattern => {
                // leading zero count pattern (simplified)
                matches!(instr, Instr::BinOp { op: BinOpKind::Shr, .. })
            }
            RewriteAction::TzcntPattern => {
                // trailing zero count pattern (simplified — detect x & -x)
                if let Instr::BinOp { op: BinOpKind::BitAnd, lhs, rhs } = instr {
                    if let Instr::UnOp { op: UnOpKind::Neg, operand } = &**rhs {
                        operand == lhs
                    } else { false }
                } else { false }
            }
            // ── Tier 5: Reassociation, Cancellation, and Memory-Form Rules ──
            RewriteAction::AddReassocConst => {
                // (a + C1) + C2 -> a + (C1 + C2)
                if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Add, rhs: ref inner_rhs, .. } = &**lhs {
                        matches!(**inner_rhs, Instr::ConstInt(_)) && matches!(**rhs, Instr::ConstInt(_))
                    } else { false }
                } else { false }
            }
            RewriteAction::MulReassocConst => {
                // (a * C1) * C2 -> a * (C1 * C2)
                if let Instr::BinOp { op: BinOpKind::Mul, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Mul, rhs: ref inner_rhs, .. } = &**lhs {
                        matches!(**inner_rhs, Instr::ConstInt(_)) && matches!(**rhs, Instr::ConstInt(_))
                    } else { false }
                } else { false }
            }
            RewriteAction::AddNegReassoc => {
                // (a + b) - a -> b
                if let Instr::BinOp { op: BinOpKind::Sub, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Add, lhs: ref add_lhs, .. } = &**lhs {
                        add_lhs == rhs
                    } else { false }
                } else { false }
            }
            RewriteAction::SubSelfCancel => {
                // a - (a + b) -> -b
                if let Instr::BinOp { op: BinOpKind::Sub, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Add, lhs: ref add_lhs, .. } = &**rhs {
                        lhs == add_lhs
                    } else { false }
                } else { false }
            }
            RewriteAction::MulAddDistribute => {
                // a*b + a*c -> a * (b + c)
                if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs } = instr {
                    if let (Instr::BinOp { op: BinOpKind::Mul, lhs: a1, .. },
                            Instr::BinOp { op: BinOpKind::Mul, lhs: a2, .. }) = (&**lhs, &**rhs) {
                        a1 == a2
                    } else { false }
                } else { false }
            }
            RewriteAction::MulSubDistribute => {
                // a*b - a*c -> a * (b - c)
                if let Instr::BinOp { op: BinOpKind::Sub, lhs, rhs } = instr {
                    if let (Instr::BinOp { op: BinOpKind::Mul, lhs: a1, .. },
                            Instr::BinOp { op: BinOpKind::Mul, lhs: a2, .. }) = (&**lhs, &**rhs) {
                        a1 == a2
                    } else { false }
                } else { false }
            }
            RewriteAction::NegateMul => {
                // -(a * b) -> (-a) * b
                if let Instr::UnOp { op: UnOpKind::Neg, operand } = instr {
                    matches!(**operand, Instr::BinOp { op: BinOpKind::Mul, .. })
                } else { false }
            }
            RewriteAction::AddSubSwap => {
                // (a + b) - c -> a + (b - c)
                if let Instr::BinOp { op: BinOpKind::Sub, lhs, .. } = instr {
                    matches!(**lhs, Instr::BinOp { op: BinOpKind::Add, .. })
                } else { false }
            }
            RewriteAction::SubAddMerge => {
                // a - b + b -> a
                if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Sub, rhs: ref sub_rhs, .. } = &**lhs {
                        sub_rhs == rhs
                    } else { false }
                } else { false }
            }
            RewriteAction::BlsmaskRule => {
                // a ^ (a - 1) -> BLSMSK(a)
                if let Instr::BinOp { op: BinOpKind::BitXor, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Sub, lhs: ref sub_lhs, rhs: ref sub_rhs } = &**rhs {
                        lhs == sub_lhs && matches!(**sub_rhs, Instr::ConstInt(1))
                    } else { false }
                } else { false }
            }
            RewriteAction::MemFormFold => {
                // Binary op with memory-load second operand -> memory-form opcode
                // Detect BinOp where rhs is a Var (representing a load from memory)
                matches!(instr, Instr::BinOp { .. })
            }
            RewriteAction::FlagReuse => {
                // Redundant CMP elimination — detect comparison BinOps
                matches!(instr,
                    Instr::BinOp { op: BinOpKind::Lt | BinOpKind::Gt | BinOpKind::Le
                                  | BinOpKind::Ge | BinOpKind::Eq | BinOpKind::Ne, .. }
                )
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
#[derive(Debug)]
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
    /// Virtual loss for parallel MCTS (AtomicU32 for thread-safe access)
    pub virtual_loss: AtomicU32,
}

impl Clone for MctsNode {
    fn clone(&self) -> Self {
        Self {
            program: self.program.clone(),
            action: self.action.clone(),
            rewrite_path: self.rewrite_path.clone(),
            visits: self.visits,
            total_reward: self.total_reward,
            children: self.children.clone(),
            expanded: self.expanded,
            cached_cost: self.cached_cost,
            policy_prior: self.policy_prior,
            virtual_loss: AtomicU32::new(self.virtual_loss.load(Ordering::Relaxed)),
        }
    }
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
            virtual_loss: AtomicU32::new(0),
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
    /// Updated with virtual loss support for parallel MCTS.
    pub fn best_child(&self, c_puct: f64) -> Option<usize> {
        if self.children.is_empty() { return None; }
        let total_visits = self.children.iter().map(|c| c.visits).sum::<u32>().max(1);
        let sqrt_parent = (total_visits as f64).sqrt();
        const VIRTUAL_LOSS_PENALTY: f64 = 5.0;

        let best = self.children.iter().enumerate().map(|(i, c)| {
            let q = if c.visits > 0 { 
                (c.total_reward - c.virtual_loss.load(Ordering::Relaxed) as f64 * VIRTUAL_LOSS_PENALTY) / c.visits as f64 
            } else { 0.0 };
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
/// Updated with structural equality check and LRU eviction support.
#[derive(Debug, Clone)]
struct TransEntry {
    hash: u64,
    /// Store program for structural equality check (two-tier: fast hash + structural)
    program: Instr,
    visits: u32,
    total_reward: f64,
    best_action: Option<usize>,
    /// For LRU eviction — tracks when this entry was last accessed
    last_access: u64,
}

/// Maximum number of entries in the transposition table.
/// When exceeded, the least recently used entry is evicted.
const TRANS_TABLE_MAX_ENTRIES: usize = 4096;

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
    /// Pre-computed dispatch table for action applicability filtering
    dispatch_table: ActionDispatchTable,
    /// Fix 6: Optional GNN value network for rollout (when gnn-optimizer enabled)
    #[cfg(feature = "gnn-optimizer")]
    gnn: Option<crate::optimizer::gnn_egraph_optimizer::GnnEgraphOptimizer>,
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
        let dispatch_table = ActionDispatchTable::build();
        Self {
            config,
            cost_estimator,
            node_count: 0,
            transposition_table: FxHashMap::default(),
            dispatch_table,
            #[cfg(feature = "gnn-optimizer")]
            gnn: None,
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

        // Fix 4: Check transposition table before rollout (two-tier: hash + structural equality)
        let leaf_hash = self.hash_instr(&leaf_program);
        let sim_result = if let Some(entry) = self.transposition_lookup(leaf_hash, &leaf_program) {
            if entry.visits > 0 {
                entry.total_reward / entry.visits as f64
            } else {
                self.rollout(&leaf_program)
            }
        } else {
            self.rollout(&leaf_program)
        };

        // Fix 4: Update transposition table after simulation (with LRU eviction)
        if let Some(entry) = self.transposition_table.get_mut(&leaf_hash) {
            if entry.program == leaf_program {
                entry.visits += 1;
                entry.total_reward += sim_result;
                entry.last_access = self.simulations_run;
            } else {
                // Hash collision — replace with new entry
                self.transposition_insert(leaf_hash, leaf_program.clone(), 1, sim_result, None);
            }
        } else {
            self.transposition_insert(leaf_hash, leaf_program.clone(), 1, sim_result, None);
        }

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

        // Use dispatch table for pre-filtering instead of iterating all actions
        let applicable_actions = self.dispatch_table.get_applicable(&node.program);
        let mut children = Vec::new();
        let mut seen: HashSet<u64> = HashSet::new();
        let num_actions = applicable_actions.len().max(1);

        // Compute cost-model-derived policy priors
        let priors = self.compute_policy_priors(&node.program, applicable_actions);

        for (action_idx, action) in applicable_actions.iter().enumerate() {
            if action.is_applicable(&node.program) {
                if let Some(new_program) = self.apply_action(&node.program, action) {
                    if new_program != node.program {
                        let h = self.hash_instr(&new_program);
                        if seen.insert(h) {
                            let policy_prior = priors.get(action_idx).copied()
                                .or_else(|| Some(1.0 / num_actions as f64));
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
                                virtual_loss: AtomicU32::new(0),
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
                        let policy_prior = priors.get(action_idx).copied()
                            .or_else(|| Some(1.0 / num_actions as f64));
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
                            virtual_loss: AtomicU32::new(0),
                        });
                    }
                }
            }
        }

        self.node_count += children.len();
        node.children = children;
        node.expanded = true;
    }

    /// Generate children lazily, stopping at the first child with non-positive
    /// estimated savings (first-win heuristic). This avoids generating all
    /// possible children when the cost model suggests most are unpromising.
    fn expand_lazy(&mut self, node: &mut MctsNode) {
        if node.expanded {
            return;
        }

        let parent_cost = node.cached_cost.unwrap_or(u32::MAX);
        let applicable_actions = self.dispatch_table.get_applicable(&node.program);

        // Sort actions by estimated savings (from cost model)
        let mut action_savings: Vec<(usize, f64)> = applicable_actions.iter().enumerate()
            .filter_map(|(i, action)| {
                if action.is_applicable(&node.program) {
                    let savings = self.estimate_savings(action, &node.program, parent_cost);
                    Some((i, savings))
                } else {
                    None
                }
            })
            .collect();

        action_savings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut seen: HashSet<u64> = HashSet::new();
        let num_actions = applicable_actions.len().max(1);
        let priors = self.compute_policy_priors(&node.program, applicable_actions);

        for (action_idx, estimated_savings) in action_savings {
            if estimated_savings <= 0.0 {
                break; // First-win: stop at first non-promising action
            }
            if let Some(child_program) = self.apply_action(&node.program, &applicable_actions[action_idx]) {
                if child_program != node.program {
                    let h = self.hash_instr(&child_program);
                    if seen.insert(h) {
                        let policy_prior = priors.get(action_idx).copied()
                            .or_else(|| Some(1.0 / num_actions as f64));
                        node.children.push(MctsNode {
                            program: child_program,
                            action: Some(applicable_actions[action_idx].clone()),
                            rewrite_path: Vec::new(),
                            visits: 0,
                            total_reward: 0.0,
                            children: Vec::new(),
                            expanded: false,
                            cached_cost: None,
                            policy_prior,
                            virtual_loss: AtomicU32::new(0),
                        });
                    }
                }
            }
        }
        node.expanded = true;
    }

    /// Estimate the savings from applying a rewrite action without full materialization.
    /// Uses heuristic cost estimates based on the action type.
    fn estimate_savings(&self, action: &RewriteAction, _program: &Instr, parent_cost: u32) -> f64 {
        let parent_cost_f = parent_cost as f64;
        match action {
            // High-impact rules: division replacement
            RewriteAction::DivByConst3 | RewriteAction::DivByConst5 |
            RewriteAction::DivByConst7 | RewriteAction::DivByConstN => parent_cost_f * 0.5,
            RewriteAction::RemByConst3 | RewriteAction::RemByConstPow2 |
            RewriteAction::RemToMask => parent_cost_f * 0.4,
            // LEA patterns
            RewriteAction::Lea3Op | RewriteAction::LeaScaleAdd |
            RewriteAction::LeaCombine | RewriteAction::LeaMulSmallConst => parent_cost_f * 0.3,
            // Architectural instructions
            RewriteAction::BswapPattern | RewriteAction::PopcntPattern |
            RewriteAction::LzcntPattern | RewriteAction::TzcntPattern |
            RewriteAction::RotateRight | RewriteAction::RotateLeft => parent_cost_f * 0.4,
            // Bit manipulation
            RewriteAction::IsolateLowest | RewriteAction::ClearLowest |
            RewriteAction::AndMaskLow | RewriteAction::AndMaskHigh => parent_cost_f * 0.2,
            // Conditional/cmov
            RewriteAction::CmovFromSelect | RewriteAction::CmovFromCmpOp |
            RewriteAction::CmovSelect | RewriteAction::SetccFromCmp => parent_cost_f * 0.2,
            // Constant folding
            RewriteAction::ConstantFold => parent_cost_f * 0.2,
            RewriteAction::StrengthReduce => parent_cost_f * 0.3,
            // Identity/annihilation (always saves)
            RewriteAction::IdentityLeft | RewriteAction::IdentityRight |
            RewriteAction::AnnihilateLeft | RewriteAction::AnnihilateRight |
            RewriteAction::AddZeroLeft | RewriteAction::AddSubCancel => parent_cost_f * 0.15,
            // Most other rules: small savings estimate
            _ => parent_cost_f * 0.1,
        }
    }

    /// Compute policy prior from cost model for each applicable action.
    /// Uses a softmax over estimated savings to create a probability distribution.
    fn compute_policy_priors(&self, program: &Instr, actions: &[RewriteAction]) -> Vec<f64> {
        let current_cost = CycleCostEstimator::node_cost(program) as f64;
        let temperature = 2.0; // Default temperature for softmax

        let savings: Vec<f64> = actions.iter().map(|action| {
            if action.is_applicable(program) {
                let estimated_savings = match action {
                    // High-impact rules get higher priors
                    RewriteAction::DivByConst3 | RewriteAction::DivByConst5 |
                    RewriteAction::DivByConst7 | RewriteAction::DivByConstN => current_cost * 0.5,
                    RewriteAction::Lea3Op | RewriteAction::LeaScaleAdd |
                    RewriteAction::LeaCombine | RewriteAction::LeaMulSmallConst => current_cost * 0.3,
                    RewriteAction::BswapPattern | RewriteAction::PopcntPattern |
                    RewriteAction::LzcntPattern | RewriteAction::TzcntPattern => current_cost * 0.4,
                    RewriteAction::ConstantFold => current_cost * 0.2,
                    RewriteAction::StrengthReduce => current_cost * 0.3,
                    RewriteAction::CmovFromSelect | RewriteAction::CmovFromCmpOp |
                    RewriteAction::CmovSelect => current_cost * 0.2,
                    RewriteAction::RemByConst3 | RewriteAction::RemByConstPow2 |
                    RewriteAction::RemToMask => current_cost * 0.4,
                    _ => current_cost * 0.1,
                };
                estimated_savings / temperature
            } else {
                0.0
            }
        }).collect();

        // Softmax normalization
        let max_saving = savings.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_savings: Vec<f64> = savings.iter().map(|s| (s - max_saving).exp()).collect();
        let sum: f64 = exp_savings.iter().sum();
        if sum > 0.0 {
            exp_savings.iter().map(|e| e / sum).collect()
        } else {
            // Uniform fallback
            let n = actions.len() as f64;
            vec![1.0 / n; actions.len()]
        }
    }

    /// Two-tier hash lookup: fast u64 bloom filter + structural equality
    fn transposition_lookup(&self, hash: u64, program: &Instr) -> Option<&TransEntry> {
        self.transposition_table.get(&hash).and_then(|entry| {
            if entry.program == *program { Some(entry) } else { None }
        })
    }

    /// Insert with LRU eviction if table is full
    fn transposition_insert(&mut self, hash: u64, program: Instr, visits: u32, reward: f64, best_action: Option<usize>) {
        if self.transposition_table.len() >= TRANS_TABLE_MAX_ENTRIES {
            // Evict LRU entry
            if let Some(lru_key) = self.transposition_table.iter()
                .min_by_key(|(_, e)| e.last_access)
                .map(|(&k, _)| k)
            {
                self.transposition_table.remove(&lru_key);
            }
        }
        self.transposition_table.insert(hash, TransEntry {
            hash,
            program,
            visits,
            total_reward: reward,
            best_action,
            last_access: self.simulations_run,
        });
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
            // ── Tier 2: Arithmetic and Bit-Level Algebra ──
            RewriteAction::SubReassoc1 => {
                // a - (b + c) -> (a - b) - c
                if let Instr::BinOp { op: BinOpKind::Sub, lhs: a, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Add, lhs: b, rhs: c } = &**rhs {
                        return Some(Instr::BinOp {
                            op: BinOpKind::Sub,
                            lhs: Box::new(Instr::BinOp { op: BinOpKind::Sub, lhs: a.clone(), rhs: b.clone() }),
                            rhs: c.clone(),
                        });
                    }
                }
                None
            }
            RewriteAction::SubReassoc2 => {
                // a - (b - c) -> (a - b) + c
                if let Instr::BinOp { op: BinOpKind::Sub, lhs: a, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Sub, lhs: b, rhs: c } = &**rhs {
                        return Some(Instr::BinOp {
                            op: BinOpKind::Add,
                            lhs: Box::new(Instr::BinOp { op: BinOpKind::Sub, lhs: a.clone(), rhs: b.clone() }),
                            rhs: c.clone(),
                        });
                    }
                }
                None
            }
            RewriteAction::AddSubCancel => {
                // (a + b) - b -> a
                if let Instr::BinOp { op: BinOpKind::Sub, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Add, lhs: a, rhs: ref add_rhs } = &**lhs {
                        if add_rhs == rhs { return Some((**a).clone()); }
                    }
                }
                None
            }
            RewriteAction::MulPow2Add => {
                // x * 2 + x -> x * 3
                if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Mul, lhs: ref x, rhs: ref mul_rhs } = &**lhs {
                        if let Instr::ConstInt(2) = **mul_rhs {
                            if x == rhs { return Some(Instr::BinOp { op: BinOpKind::Mul, lhs: x.clone(), rhs: Box::new(Instr::ConstInt(3)) }); }
                        }
                    }
                    if let Instr::BinOp { op: BinOpKind::Mul, lhs: ref x, rhs: ref mul_rhs } = &**rhs {
                        if let Instr::ConstInt(2) = **mul_rhs {
                            if x == lhs { return Some(Instr::BinOp { op: BinOpKind::Mul, lhs: x.clone(), rhs: Box::new(Instr::ConstInt(3)) }); }
                        }
                    }
                    // x << 1 + x -> x * 3
                    if let Instr::BinOp { op: BinOpKind::Shl, lhs: ref x, rhs: ref shift_rhs } = &**lhs {
                        if let Instr::ConstInt(1) = **shift_rhs {
                            if x == rhs { return Some(Instr::BinOp { op: BinOpKind::Mul, lhs: x.clone(), rhs: Box::new(Instr::ConstInt(3)) }); }
                        }
                    }
                }
                None
            }
            RewriteAction::MulDistSub => {
                // a * (b - c) -> a*b - a*c
                if let Instr::BinOp { op: BinOpKind::Mul, lhs: a, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Sub, lhs: b, rhs: c } = &**rhs {
                        return Some(Instr::BinOp {
                            op: BinOpKind::Sub,
                            lhs: Box::new(Instr::BinOp { op: BinOpKind::Mul, lhs: a.clone(), rhs: b.clone() }),
                            rhs: Box::new(Instr::BinOp { op: BinOpKind::Mul, lhs: a.clone(), rhs: c.clone() }),
                        });
                    }
                }
                None
            }
            RewriteAction::NegSwap => {
                // -(a - b) -> b - a
                if let Instr::UnOp { op: UnOpKind::Neg, operand } = instr {
                    if let Instr::BinOp { op: BinOpKind::Sub, lhs, rhs } = &**operand {
                        return Some(Instr::BinOp { op: BinOpKind::Sub, lhs: rhs.clone(), rhs: lhs.clone() });
                    }
                }
                None
            }
            RewriteAction::DoubleNegAdd => {
                // -(-a + b) -> a - b
                if let Instr::UnOp { op: UnOpKind::Neg, operand } = instr {
                    if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs: b } = &**operand {
                        if let Instr::UnOp { op: UnOpKind::Neg, operand: ref inner } = &**lhs {
                            return Some(Instr::BinOp { op: BinOpKind::Sub, lhs: inner.clone(), rhs: b.clone() });
                        }
                    }
                }
                None
            }
            RewriteAction::AddZeroLeft => {
                // 0 + a -> a
                if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs } = instr {
                    if let Instr::ConstInt(0) = **lhs { return Some((**rhs).clone()); }
                }
                None
            }
            RewriteAction::ShlByConstAdd => {
                // (x << k) + x -> x * (2^k + 1)
                if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Shl, lhs: ref x, rhs: ref shift_rhs } = &**lhs {
                        if let Instr::ConstInt(k) = **shift_rhs {
                            if x == rhs {
                                let factor = (1u128 << k) + 1;
                                return Some(Instr::BinOp { op: BinOpKind::Mul, lhs: x.clone(), rhs: Box::new(Instr::ConstInt(factor)) });
                            }
                        }
                    }
                    if let Instr::BinOp { op: BinOpKind::Shl, lhs: ref x, rhs: ref shift_rhs } = &**rhs {
                        if let Instr::ConstInt(k) = **shift_rhs {
                            if x == lhs {
                                let factor = (1u128 << k) + 1;
                                return Some(Instr::BinOp { op: BinOpKind::Mul, lhs: x.clone(), rhs: Box::new(Instr::ConstInt(factor)) });
                            }
                        }
                    }
                }
                None
            }
            RewriteAction::ShrIsDivPow2 => {
                // x >> k -> x / 2^k (semantic equivalence — mark for cost model)
                Some(instr.clone())
            }
            // ── Tier 2: Bit Manipulation Rules ──
            RewriteAction::AndNotComplement | RewriteAction::AndComplement => {
                // a & (!b) / a & (~b) -> ANDN(a, b) — mark for cost model
                Some(instr.clone())
            }
            RewriteAction::XorSwap => {
                // a ^ b ^ b -> a
                if let Instr::BinOp { op: BinOpKind::BitXor, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::BitXor, lhs: ref a, rhs: ref inner_rhs } = &**lhs {
                        if inner_rhs == rhs { return Some((**a).clone()); }
                    }
                }
                None
            }
            RewriteAction::XorAllOnes => {
                // a ^ ~0 -> ~a
                if let Instr::BinOp { op: BinOpKind::BitXor, lhs, rhs } = instr {
                    if let Instr::ConstInt(v) = **rhs {
                        if v == u128::MAX { return Some(Instr::UnOp { op: UnOpKind::Not, operand: lhs.clone() }); }
                    }
                }
                None
            }
            RewriteAction::AndMaskLow | RewriteAction::AndMaskHigh => {
                // Mark for cost model — these are architectural instruction matches
                Some(instr.clone())
            }
            RewriteAction::IsolateLowest => {
                // x & (-x) -> BLSI(x) — mark for cost model
                Some(instr.clone())
            }
            RewriteAction::ClearLowest => {
                // x & (x - 1) -> BLSR(x) — mark for cost model
                Some(instr.clone())
            }
            RewriteAction::MaskMerge => {
                // (a & mask) | (b & ~mask) -> BFI(a, b, mask) — mark for cost model
                Some(instr.clone())
            }
            // ── Tier 3: Comparison and Conditional Rules ──
            RewriteAction::CmpNegateFull => {
                // !(a <= b) -> a > b and !(a >= b) -> a < b
                if let Instr::UnOp { op: UnOpKind::Not, operand } = instr {
                    if let Instr::BinOp { op, lhs, rhs } = &**operand {
                        let new_op = match op {
                            BinOpKind::Le => BinOpKind::Gt,
                            BinOpKind::Ge => BinOpKind::Lt,
                            _ => return None,
                        };
                        return Some(Instr::BinOp { op: new_op, lhs: lhs.clone(), rhs: rhs.clone() });
                    }
                }
                None
            }
            RewriteAction::DoubleNegCmp => {
                // !!(a < b) -> a < b
                if let Instr::UnOp { op: UnOpKind::Not, operand } = instr {
                    if let Instr::UnOp { op: UnOpKind::Not, operand: ref inner } = &**operand {
                        return Some((**inner).clone());
                    }
                }
                None
            }
            RewriteAction::EqNormalize => {
                // a == b -> b == a (when b is const)
                if let Instr::BinOp { op: BinOpKind::Eq, lhs, rhs } = instr {
                    if matches!(**rhs, Instr::ConstInt(_)) {
                        return Some(Instr::BinOp { op: BinOpKind::Eq, lhs: rhs.clone(), rhs: lhs.clone() });
                    }
                }
                None
            }
            RewriteAction::NeNormalize => {
                // a != b -> b != a (when b is const)
                if let Instr::BinOp { op: BinOpKind::Ne, lhs, rhs } = instr {
                    if matches!(**rhs, Instr::ConstInt(_)) {
                        return Some(Instr::BinOp { op: BinOpKind::Ne, lhs: rhs.clone(), rhs: lhs.clone() });
                    }
                }
                None
            }
            RewriteAction::SubIsZero => {
                // (a - b) == 0 -> a == b
                if let Instr::BinOp { op: BinOpKind::Eq, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Sub, lhs: a, rhs: b } = &**lhs {
                        if let Instr::ConstInt(0) = **rhs {
                            return Some(Instr::BinOp { op: BinOpKind::Eq, lhs: a.clone(), rhs: b.clone() });
                        }
                    }
                }
                None
            }
            RewriteAction::SubIsNonZero => {
                // (a - b) != 0 -> a != b
                if let Instr::BinOp { op: BinOpKind::Ne, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Sub, lhs: a, rhs: b } = &**lhs {
                        if let Instr::ConstInt(0) = **rhs {
                            return Some(Instr::BinOp { op: BinOpKind::Ne, lhs: a.clone(), rhs: b.clone() });
                        }
                    }
                }
                None
            }
            RewriteAction::LeFromLt => {
                // a < b + 1 -> a <= b
                if let Instr::BinOp { op: BinOpKind::Lt, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Add, lhs: b, rhs: ref add_rhs } = &**rhs {
                        if let Instr::ConstInt(1) = **add_rhs {
                            return Some(Instr::BinOp { op: BinOpKind::Le, lhs: lhs.clone(), rhs: b.clone() });
                        }
                    }
                }
                None
            }
            RewriteAction::GeFromGt => {
                // a > b - 1 -> a >= b
                if let Instr::BinOp { op: BinOpKind::Gt, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Sub, lhs: b, rhs: ref sub_rhs } = &**rhs {
                        if let Instr::ConstInt(1) = **sub_rhs {
                            return Some(Instr::BinOp { op: BinOpKind::Ge, lhs: lhs.clone(), rhs: b.clone() });
                        }
                    }
                }
                None
            }
            RewriteAction::AndCmps => {
                // (a > 0) & (a < N) -> range check pattern — mark for cost model
                Some(instr.clone())
            }
            // ── Tier 3: Division and Remainder Optimization ──
            RewriteAction::DivByConst3 | RewriteAction::DivByConst5 |
            RewriteAction::DivByConst7 | RewriteAction::DivByConstN => {
                // x / N -> MULH(x, magic(N)) — mark for cost model
                Some(instr.clone())
            }
            RewriteAction::RemByConst3 | RewriteAction::RemToMask => {
                // x % N -> x - (x/N)*N — mark for cost model
                Some(instr.clone())
            }
            RewriteAction::RemByConstPow2 => {
                // x % 2^k -> x & (2^k - 1)
                if let Instr::BinOp { op: BinOpKind::Rem, lhs, rhs } = instr {
                    if let Instr::ConstInt(v) = **rhs {
                        if v > 1 && (v & (v - 1)) == 0 {
                            return Some(Instr::BinOp { op: BinOpKind::BitAnd, lhs: lhs.clone(), rhs: Box::new(Instr::ConstInt(v - 1)) });
                        }
                    }
                }
                None
            }
            RewriteAction::DivNegNeg => {
                // (-a) / (-b) -> a / b
                if let Instr::BinOp { op: BinOpKind::Div, lhs, rhs } = instr {
                    if let (Instr::UnOp { op: UnOpKind::Neg, operand: a },
                            Instr::UnOp { op: UnOpKind::Neg, operand: b }) = (&**lhs, &**rhs) {
                        return Some(Instr::BinOp { op: BinOpKind::Div, lhs: a.clone(), rhs: b.clone() });
                    }
                }
                None
            }
            RewriteAction::DivSignAdjust => {
                // x / (-N) -> -(x / N)
                if let Instr::BinOp { op: BinOpKind::Div, lhs, rhs } = instr {
                    if let Instr::UnOp { op: UnOpKind::Neg, operand: ref inner } = &**rhs {
                        return Some(Instr::UnOp {
                            op: UnOpKind::Neg,
                            operand: Box::new(Instr::BinOp { op: BinOpKind::Div, lhs: lhs.clone(), rhs: inner.clone() }),
                        });
                    }
                }
                None
            }
            RewriteAction::UnsignedDivPow2 => {
                // x >>> k -> x / 2^k — mark for cost model
                Some(instr.clone())
            }
            // ── Tier 4: Multi-Step and Architectural Rules ──
            RewriteAction::Lea3Op | RewriteAction::LeaScaleAdd => {
                // Mark for cost model — LEA pattern detection
                Some(instr.clone())
            }
            RewriteAction::TestInsteadOfAnd | RewriteAction::TestInsteadOfAndNZ => {
                // Mark for cost model — TEST instruction pattern
                Some(instr.clone())
            }
            RewriteAction::SetccFromCmp | RewriteAction::CmovFromSelect |
            RewriteAction::CmovFromCmpOp => {
                // Mark for cost model — conditional instruction pattern
                Some(instr.clone())
            }
            RewriteAction::SbbFromBorrow | RewriteAction::AdcFromCarry => {
                // Mark for cost model — carry instruction pattern
                Some(instr.clone())
            }
            RewriteAction::XorZero => {
                // x ^ x -> 0
                if let Instr::BinOp { op: BinOpKind::BitXor, lhs, rhs } = instr {
                    if lhs == rhs { return Some(Instr::ConstInt(0)); }
                }
                None
            }
            RewriteAction::MovZero => {
                // 0 -> XOR(reg, reg) — mark for cost model
                Some(instr.clone())
            }
            RewriteAction::RotateRight | RewriteAction::RotateLeft => {
                // Mark for cost model — rotation pattern
                Some(instr.clone())
            }
            RewriteAction::BswapPattern | RewriteAction::PopcntPattern |
            RewriteAction::LzcntPattern | RewriteAction::TzcntPattern => {
                // Mark for cost model — special instruction pattern
                Some(instr.clone())
            }
            // ── Tier 5: Reassociation, Cancellation, and Memory-Form Rules ──
            RewriteAction::AddReassocConst => {
                // (a + C1) + C2 -> a + (C1 + C2)
                if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Add, lhs: ref a, rhs: ref c1 } = &**lhs {
                        if let (Instr::ConstInt(v1), Instr::ConstInt(v2)) = (&**c1, &**rhs) {
                            return Some(Instr::BinOp {
                                op: BinOpKind::Add,
                                lhs: a.clone(),
                                rhs: Box::new(Instr::ConstInt(v1.wrapping_add(*v2))),
                            });
                        }
                    }
                }
                None
            }
            RewriteAction::MulReassocConst => {
                // (a * C1) * C2 -> a * (C1 * C2)
                if let Instr::BinOp { op: BinOpKind::Mul, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Mul, lhs: ref a, rhs: ref c1 } = &**lhs {
                        if let (Instr::ConstInt(v1), Instr::ConstInt(v2)) = (&**c1, &**rhs) {
                            return Some(Instr::BinOp {
                                op: BinOpKind::Mul,
                                lhs: a.clone(),
                                rhs: Box::new(Instr::ConstInt(v1.wrapping_mul(*v2))),
                            });
                        }
                    }
                }
                None
            }
            RewriteAction::AddNegReassoc => {
                // (a + b) - a -> b
                if let Instr::BinOp { op: BinOpKind::Sub, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Add, lhs: ref add_lhs, rhs: ref add_rhs, .. } = &**lhs {
                        if add_lhs == rhs { return Some((**add_rhs).clone()); }
                    }
                }
                None
            }
            RewriteAction::SubSelfCancel => {
                // a - (a + b) -> -b
                if let Instr::BinOp { op: BinOpKind::Sub, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Add, lhs: ref add_lhs, rhs: ref add_rhs, .. } = &**rhs {
                        if lhs == add_lhs {
                            return Some(Instr::UnOp { op: UnOpKind::Neg, operand: add_rhs.clone() });
                        }
                    }
                }
                None
            }
            RewriteAction::MulAddDistribute => {
                // a*b + a*c -> a * (b + c)
                if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs } = instr {
                    if let (Instr::BinOp { op: BinOpKind::Mul, lhs: a1, rhs: b },
                            Instr::BinOp { op: BinOpKind::Mul, lhs: a2, rhs: c }) = (&**lhs, &**rhs) {
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
            RewriteAction::MulSubDistribute => {
                // a*b - a*c -> a * (b - c)
                if let Instr::BinOp { op: BinOpKind::Sub, lhs, rhs } = instr {
                    if let (Instr::BinOp { op: BinOpKind::Mul, lhs: a1, rhs: b },
                            Instr::BinOp { op: BinOpKind::Mul, lhs: a2, rhs: c }) = (&**lhs, &**rhs) {
                        if a1 == a2 {
                            return Some(Instr::BinOp {
                                op: BinOpKind::Mul,
                                lhs: a1.clone(),
                                rhs: Box::new(Instr::BinOp { op: BinOpKind::Sub, lhs: b.clone(), rhs: c.clone() }),
                            });
                        }
                    }
                }
                None
            }
            RewriteAction::NegateMul => {
                // -(a * b) -> (-a) * b
                if let Instr::UnOp { op: UnOpKind::Neg, operand } = instr {
                    if let Instr::BinOp { op: BinOpKind::Mul, lhs, rhs } = &**operand {
                        return Some(Instr::BinOp {
                            op: BinOpKind::Mul,
                            lhs: Box::new(Instr::UnOp { op: UnOpKind::Neg, operand: lhs.clone() }),
                            rhs: rhs.clone(),
                        });
                    }
                }
                None
            }
            RewriteAction::AddSubSwap => {
                // (a + b) - c -> a + (b - c)
                if let Instr::BinOp { op: BinOpKind::Sub, lhs, rhs: ref c } = instr {
                    if let Instr::BinOp { op: BinOpKind::Add, lhs: ref a, rhs: ref b } = &**lhs {
                        return Some(Instr::BinOp {
                            op: BinOpKind::Add,
                            lhs: a.clone(),
                            rhs: Box::new(Instr::BinOp { op: BinOpKind::Sub, lhs: b.clone(), rhs: c.clone() }),
                        });
                    }
                }
                None
            }
            RewriteAction::SubAddMerge => {
                // a - b + b -> a
                if let Instr::BinOp { op: BinOpKind::Add, lhs, rhs } = instr {
                    if let Instr::BinOp { op: BinOpKind::Sub, lhs: ref a, rhs: ref sub_rhs } = &**lhs {
                        if sub_rhs == rhs { return Some((**a).clone()); }
                    }
                }
                None
            }
            RewriteAction::BlsmaskRule => {
                // a ^ (a - 1) -> BLSMSK(a) — mark for cost model
                Some(instr.clone())
            }
            RewriteAction::MemFormFold => {
                // Fold binary op with memory-load second operand into memory-form opcode
                // Mark for cost model — memory-form instruction pattern
                Some(instr.clone())
            }
            RewriteAction::FlagReuse => {
                // Eliminate redundant CMP with same operands — mark for cost model
                Some(instr.clone())
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
                let cost = self.cost_estimator.estimate(instr);
                let graph = crate::optimizer::gnn_egraph_optimizer::ProgramGraph::from_instr(instr, cost);
                let (_policy, value) = gnn.model().forward(&graph);
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
            virtual_loss: AtomicU32::new(0),
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
            virtual_loss: AtomicU32::new(0),
        });
        // With equal visits but different priors, PUCT should prefer higher prior
        let best = root.best_child(1.414);
        assert_eq!(best, Some(0)); // Child 0 has higher policy prior
    }
}
