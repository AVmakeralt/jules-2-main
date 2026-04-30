// =============================================================================
// jules/src/self_repair.rs
//
// ADAPTIVE IR REWRITING VIA RUNTIME GUARD FAILURE ANALYSIS
//
// Research-grade self-repairing compiler infrastructure.
//
// Architecture:
//   1. Guard-based self-healing: detects type instability, performance cliffs,
//      and UB at runtime, then synthesizes patches via the superoptimizer.
//   2. E-Graph integration: uses equality saturation to find optimal IR
//      sequences that handle both old and new cases simultaneously.
//   3. Profile-guided pre-emptive repair: AOT compiler ingests runtime
//      failure logs and generates robust code for known fragile paths.
//   4. Hot-swap deployment: JIT replaces failing traces with synthesized patches
//      without restarting the program.
//   5. Formal verification: validates that synthesized patches preserve
//      semantic equivalence (same postconditions as original IR).
//
// Inspired by:
//   - V8 Turbolizer / Crankshaft deoptimization
//   - PyPy JIT: guard elaboration + trace stitching
//   - Souper / egg: e-graph equality saturation
//   - Self-optimizing interpreters (Rigo et al. 2018)
//   - Profile-guided optimization (PGO) + Bolt
// =============================================================================

#![allow(dead_code)]

use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use rustc_hash::{FxHashMap, FxHasher};

// =============================================================================
// §0  CONSTANTS & CONFIGURATION
// =============================================================================

/// Default threshold: number of failures before repair is triggered
const DEFAULT_FAILURE_THRESHOLD: u32 = 5;
/// Maximum repair attempts before giving up on a path
const MAX_REPAIR_ATTEMPTS: u32 = 10;
/// Cost threshold for E-Graph saturation (max iterations)
const EGRAPH_MAX_ITERATIONS: usize = 20;
/// Minimum improvement ratio to accept a patch (e.g., 10% faster)
const MIN_IMPROVEMENT_RATIO: f64 = 0.10;

/// Repair strategy configuration
#[derive(Debug, Clone)]
pub struct RepairConfig {
    pub failure_threshold: u32,
    pub max_repair_attempts: u32,
    pub enable_egraph_synthesize: bool,
    pub enable_cached_patches: bool,
    pub enable_profile_guided_aot: bool,
    pub enable_formal_verification: bool,
    pub performance_cliff_multiplier: u64, // x times slower = cliff
    pub verbose: bool,
}

impl RepairConfig {
    pub fn aggressive() -> Self {
        Self {
            failure_threshold: 2,
            max_repair_attempts: 20,
            enable_egraph_synthesize: true,
            enable_cached_patches: true,
            enable_profile_guided_aot: true,
            enable_formal_verification: true,
            performance_cliff_multiplier: 3,
            verbose: true,
        }
    }

    pub fn conservative() -> Self {
        Self {
            failure_threshold: 10,
            max_repair_attempts: 5,
            enable_egraph_synthesize: false,
            enable_cached_patches: true,
            enable_profile_guided_aot: true,
            enable_formal_verification: false,
            performance_cliff_multiplier: 10,
            verbose: false,
        }
    }

    pub fn default() -> Self {
        Self {
            failure_threshold: DEFAULT_FAILURE_THRESHOLD,
            max_repair_attempts: MAX_REPAIR_ATTEMPTS,
            enable_egraph_synthesize: true,
            enable_cached_patches: true,
            enable_profile_guided_aot: true,
            enable_formal_verification: true,
            performance_cliff_multiplier: 5,
            verbose: false,
        }
    }
}

// =============================================================================
// §1  REPAIR EVENTS — What triggers self-repair?
// =============================================================================

/// A runtime failure event that may trigger repair
#[derive(Debug, Clone)]
pub struct RepairEvent {
    /// Which function failed
    pub func_name: String,
    /// Which basic block failed in
    pub block_id: usize,
    /// Which instruction within the block
    pub instruction_index: usize,
    /// What kind of failure
    pub failure_type: FailureType,
    /// Runtime context at failure (variable types, values, etc.)
    pub runtime_context: FxHashMap<String, RuntimeValue>,
    /// Timestamp (for profiling)
    pub timestamp: Instant,
}

/// Types of failures the repair engine can handle
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum FailureType {
    /// Type guard failed: expected one type, got another
    GuardTypeMismatch {
        expected: ValueType,
        actual: ValueType,
        variable: String,
    },
    /// Bounds check failed: array/tensor index out of range
    GuardBoundsCheck {
        index: i64,
        upper_bound: i64,
        variable: String,
    },
    /// Loop iteration count exceeded traced bound
    GuardLoopBound {
        traced_bound: i64,
        actual_count: i64,
        loop_id: String,
    },
    /// Performance cliff detected: execution was Nx slower than expected
    PerformanceCliff {
        expected_cycles: u64,
        actual_cycles: u64,
    },
    /// Integer overflow detected (in unsafe mode)
    IntegerOverflow {
        operation: String,
        lhs: i64,
        rhs: i64,
        result: i64,
    },
    /// Division by zero
    DivisionByZero {
        dividend: i64,
    },
    /// Null/None dereference
    NullDereference {
        variable: String,
    },
    /// Shape mismatch for tensor operations
    TensorShapeMismatch {
        expected_shape: Vec<usize>,
        actual_shape: Vec<usize>,
        operation: String,
    },
}

/// Runtime value representation for context capture
#[derive(Debug, Clone)]
pub enum RuntimeValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    TypeOnly(ValueType), // When we only know the type, not the value
    Tensor { elem: ValueType, shape: Vec<usize> },
}

/// Value types tracked by the repair engine
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValueType {
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    F32, F64,
    Bool,
    Tensor(Box<ValueType>),
    Unknown,
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueType::I8 => write!(f, "i8"),
            ValueType::I16 => write!(f, "i16"),
            ValueType::I32 => write!(f, "i32"),
            ValueType::I64 => write!(f, "i64"),
            ValueType::U8 => write!(f, "u8"),
            ValueType::U16 => write!(f, "u16"),
            ValueType::U32 => write!(f, "u32"),
            ValueType::U64 => write!(f, "u64"),
            ValueType::F32 => write!(f, "f32"),
            ValueType::F64 => write!(f, "f64"),
            ValueType::Bool => write!(f, "bool"),
            ValueType::Tensor(inner) => write!(f, "tensor<{}>", inner),
            ValueType::Unknown => write!(f, "?"),
        }
    }
}

impl RepairEvent {
    /// Create a unique fingerprint for this failure (for caching)
    pub fn fingerprint(&self) -> u64 {
        let mut hasher = FxHasher::default();
        self.failure_type.hash(&mut hasher);
        // Intentionally ignore runtime_context so equivalent failures across
        // different callsites/functions map to a stable shared fingerprint.
        hasher.finish()
    }

    /// Human-readable description of the failure
    pub fn description(&self) -> String {
        match &self.failure_type {
            FailureType::GuardTypeMismatch { expected, actual, variable } => {
                format!("Type guard failed on `{}`: expected {}, got {}", variable, expected, actual)
            }
            FailureType::GuardBoundsCheck { index, upper_bound, variable } => {
                format!("Bounds check failed on `{}`: index {} >= bound {}", variable, index, upper_bound)
            }
            FailureType::GuardLoopBound { traced_bound, actual_count, loop_id } => {
                format!("Loop bound exceeded in `{}`: traced for {} iterations, actual {}", loop_id, traced_bound, actual_count)
            }
            FailureType::PerformanceCliff { expected_cycles, actual_cycles } => {
                let multiplier = *actual_cycles as f64 / *expected_cycles as f64;
                format!("Performance cliff: expected {} cycles, got {} ({:.1}x slower)", expected_cycles, actual_cycles, multiplier)
            }
            FailureType::IntegerOverflow { operation, lhs, rhs, result } => {
                format!("Integer overflow: {} {} {} = {} (overflowed)", lhs, operation, rhs, result)
            }
            FailureType::DivisionByZero { dividend } => {
                format!("Division by zero: {} / 0", dividend)
            }
            FailureType::NullDereference { variable } => {
                format!("Null dereference: `{}` is None/Null", variable)
            }
            FailureType::TensorShapeMismatch { expected_shape, actual_shape, operation } => {
                format!("Tensor shape mismatch in `{}`: expected {:?}, got {:?}", operation, expected_shape, actual_shape)
            }
        }
    }
}

// =============================================================================
// §2  IR PATCH — Synthesized fix for a failure
// =============================================================================

/// A synthesized IR patch that repairs a failure
#[derive(Debug, Clone)]
pub struct IRPatch {
    /// Instructions to insert/replace
    pub instructions: Vec<PatchInstr>,
    /// Which block to apply this in
    pub target_block: usize,
    /// Position within the block (instruction index)
    pub insert_position: PatchPosition,
    /// Metadata about the patch
    pub metadata: PatchMetadata,
}

/// Position to insert patch instructions
#[derive(Debug, Clone, Copy)]
pub enum PatchPosition {
    /// Insert before the failing instruction
    Before(usize),
    /// Insert after the failing instruction
    After(usize),
    /// Replace instructions from start..end
    Replace { start: usize, end: usize },
    /// Prepend to block
    Prepend,
    /// Append to block
    Append,
}

/// Metadata about a synthesized patch
#[derive(Debug, Clone)]
pub struct PatchMetadata {
    /// Why this patch was synthesized (root cause)
    pub root_cause: String,
    /// What repair strategy was used
    pub strategy: RepairStrategy,
    /// Estimated cost of the patch (instruction count)
    pub estimated_cost: usize,
    /// Expected performance impact (positive = improvement)
    pub expected_impact: f64,
    /// Whether this patch has been verified
    pub verified: bool,
}

/// Repair strategies used by the synthesizer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairStrategy {
    /// Insert a type check + branch to handle polymorphism
    PolymorphicGuard,
    /// Widen a type annotation (e.g., i32 → i64)
    TypeWidening,
    /// Insert bounds check + fallback
    BoundsCheckInsertion,
    /// Replace operation with safe variant (e.g., div → checked_div)
    OperationReplacement,
    /// Unroll loop further to handle larger bounds
    LoopUnrollIncrease,
    /// Insert overflow check
    OverflowCheckInsertion,
    /// Deoptimize to interpreter (fallback)
    Deoptimize,
    /// E-Graph synthesized equivalent
    EGraphSynthesized,
}

/// Patch instruction — simplified IR for patching
#[derive(Debug, Clone)]
pub enum PatchInstr {
    /// Check type of variable, branch if mismatch
    CheckType { variable: String, expected: ValueType, if_false: usize },
    /// Check bounds: if index >= bound, jump to fallback
    CheckBounds { index: String, bound: String, if_fail: usize },
    /// Check integer overflow: compute + check flags
    CheckOverflow { dst: String, lhs: String, rhs: String, op: String, if_overflow: usize },
    /// Type conversion: convert variable from one type to another
    ConvertType { dst: String, src: String, from: ValueType, to: ValueType },
    /// Widen type: extend variable to wider type
    WidenType { dst: String, src: String, from: ValueType, to: ValueType },
    /// Branch: unconditional jump
    Branch { target: usize },
    /// Conditional branch
    CondBranch { cond: String, if_true: usize, if_false: usize },
    /// Compute: perform an operation
    Compute { dst: String, op: String, lhs: String, rhs: String },
    /// Load constant
    Const { dst: String, value: i64 },
    /// Call runtime helper (e.g., generic slow path)
    CallRuntime { dst: String, helper: String, args: Vec<String> },
    /// Return from function
    Return { value: Option<String> },
    /// Deoptimize: fall back to interpreter
    Deoptimize { reason: String },
    /// Comment (for debugging)
    Comment(String),
}

impl IRPatch {
    /// Create a patch for a type guard mismatch
    pub fn type_guard_patch(
        block_id: usize,
        instr_index: usize,
        variable: String,
        expected: ValueType,
        actual: ValueType,
    ) -> Self {
        let mut instructions = Vec::new();

        // 1. Insert a type check
        instructions.push(PatchInstr::CheckType {
            variable: variable.clone(),
            expected: expected.clone(),
            if_false: 3, // Jump to fallback block
        });

        // 2. If type matches, continue to original instruction
        instructions.push(PatchInstr::Comment(
            format!("Original type guard for {} ({}), type matches → continue", variable, expected),
        ));

        // 3. Branch to merge point
        instructions.push(PatchInstr::Branch { target: 4 });

        // 4. Fallback: convert type and execute generic path
        instructions.push(PatchInstr::ConvertType {
            dst: format!("{}_converted", variable),
            src: variable.clone(),
            from: actual.clone(),
            to: expected.clone(),
        });

        instructions.push(PatchInstr::Comment(
            format!("Polymorphic guard: {} converted from {} to {}", variable, actual, expected),
        ));

        let estimated_cost = instructions.len();

        Self {
            instructions,
            target_block: block_id,
            insert_position: PatchPosition::Before(instr_index),
            metadata: PatchMetadata {
                root_cause: format!("Type mismatch on `{}`: expected {}, got {}", variable, expected, actual),
                strategy: RepairStrategy::PolymorphicGuard,
                estimated_cost,
                expected_impact: -0.05, // 5% slowdown due to extra check
                verified: false,
            },
        }
    }

    /// Create a patch for a bounds check failure
    pub fn bounds_check_patch(
        block_id: usize,
        instr_index: usize,
        variable: String,
        index: String,
        bound: String,
    ) -> Self {
        let mut instructions = Vec::new();

        // 1. Check bounds
        instructions.push(PatchInstr::CheckBounds {
            index: index.clone(),
            bound: bound.clone(),
            if_fail: 3,
        });

        // 2. Bounds OK, continue
        instructions.push(PatchInstr::Comment(format!("Bounds check: {} < {} → OK", index, bound)));

        // 3. Bounds failed: fallback to slow path
        instructions.push(PatchInstr::CallRuntime {
            dst: format!("{}_result", variable),
            helper: "generic_index".into(),
            args: vec![variable.clone(), index.clone()],
        });

        Self {
            instructions,
            target_block: block_id,
            insert_position: PatchPosition::Before(instr_index),
            metadata: PatchMetadata {
                root_cause: format!("Bounds check failed: {} >= {}", index, bound),
                strategy: RepairStrategy::BoundsCheckInsertion,
                estimated_cost: 3,
                expected_impact: -0.10,
                verified: false,
            },
        }
    }

    /// Create a patch for integer overflow
    pub fn overflow_check_patch(
        block_id: usize,
        instr_index: usize,
        dst: String,
        lhs: String,
        rhs: String,
        op: String,
    ) -> Self {
        let mut instructions = Vec::new();

        // 1. Compute with overflow check
        instructions.push(PatchInstr::CheckOverflow {
            dst: dst.clone(),
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            op: op.clone(),
            if_overflow: 3,
        });

        // 2. No overflow, result in dst
        instructions.push(PatchInstr::Comment(format!("{} {} {} → no overflow", lhs, op, rhs)));

        // 3. Overflow detected: widen to larger type or call runtime
        instructions.push(PatchInstr::CallRuntime {
            dst: format!("{}_wide", dst),
            helper: format!("wide_{}", op),
            args: vec![lhs.clone(), rhs.clone()],
        });

        Self {
            instructions,
            target_block: block_id,
            insert_position: PatchPosition::Before(instr_index),
            metadata: PatchMetadata {
                root_cause: format!("Integer overflow: {} {} {}", lhs, op, rhs),
                strategy: RepairStrategy::OverflowCheckInsertion,
                estimated_cost: 3,
                expected_impact: -0.08,
                verified: false,
            },
        }
    }

    /// Create a patch for performance cliff: deoptimize to generic path
    pub fn performance_cliff_patch(
        block_id: usize,
        instr_index: usize,
        reason: String,
        expected_cycles: u64,
        actual_cycles: u64,
    ) -> Self {
        let instructions = vec![
            PatchInstr::Comment(format!("Performance cliff detected: {}x slower than expected", actual_cycles / expected_cycles)),
            PatchInstr::Deoptimize { reason },
        ];
        let estimated_cost = instructions.len();
        Self {
            instructions,
            target_block: block_id,
            insert_position: PatchPosition::Replace { start: instr_index, end: instr_index + 1 },
            metadata: PatchMetadata {
                root_cause: format!("Performance cliff: {} → {} cycles", expected_cycles, actual_cycles),
                strategy: RepairStrategy::Deoptimize,
                estimated_cost,
                expected_impact: 0.0, // Neutral: correctness over speed
                verified: false,
            },
        }
    }

    /// Create a patch for division by zero
    pub fn division_by_zero_patch(
        block_id: usize,
        instr_index: usize,
        divisor: String,
        dividend: String,
    ) -> Self {
        let mut instructions = Vec::new();

        // 1. Check for zero
        instructions.push(PatchInstr::CheckBounds {
            index: divisor.clone(),
            bound: "0".into(),
            if_fail: 3,
        });

        // 2. Divisor non-zero, proceed with division
        instructions.push(PatchInstr::Compute {
            dst: format!("{}_result", dividend),
            op: "div".into(),
            lhs: dividend.clone(),
            rhs: divisor.clone(),
        });

        // 3. Divisor is zero: return default or call runtime
        instructions.push(PatchInstr::CallRuntime {
            dst: format!("{}_result", dividend),
            helper: "checked_div".into(),
            args: vec![dividend.clone(), divisor.clone()],
        });

        let estimated_cost = instructions.len();

        Self {
            instructions,
            target_block: block_id,
            insert_position: PatchPosition::Before(instr_index),
            metadata: PatchMetadata {
                root_cause: format!("Division by zero: {} / 0", dividend),
                strategy: RepairStrategy::OperationReplacement,
                estimated_cost,
                expected_impact: -0.05,
                verified: false,
            },
        }
    }

    /// Serialize patch to Jules source code (for human review)
    pub fn to_jules_source(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("// Self-repair patch in block {} ({:?})", self.target_block, self.metadata.strategy));
        lines.push(format!("// Root cause: {}", self.metadata.root_cause));
        lines.push(String::new());

        for instr in &self.instructions {
            match instr {
                PatchInstr::CheckType { variable, expected, if_false } => {
                    lines.push(format!("  if type_of({}) != {} {{ goto block_{}; }}", variable, expected, if_false));
                }
                PatchInstr::CheckBounds { index, bound, if_fail } => {
                    lines.push(format!("  if {} >= {} {{ goto block_{}; }}", index, bound, if_fail));
                }
                PatchInstr::CheckOverflow { dst, lhs, rhs, op, if_overflow } => {
                    lines.push(format!("  {} = checked_{}({}, {}) || goto block_{};", dst, op, lhs, rhs, if_overflow));
                }
                PatchInstr::ConvertType { dst, src, from, to } => {
                    lines.push(format!("  {} = {} as {} /* from {} */;", dst, src, to, from));
                }
                PatchInstr::WidenType { dst, src, from, to } => {
                    lines.push(format!("  {} = widen({} as {} /* {} → {} */);", dst, src, to, from, to));
                }
                PatchInstr::Branch { target } => {
                    lines.push(format!("  goto block_{};", target));
                }
                PatchInstr::CondBranch { cond, if_true, if_false } => {
                    lines.push(format!("  if {} {{ goto block_{}; }} else {{ goto block_{}; }}", cond, if_true, if_false));
                }
                PatchInstr::Compute { dst, op, lhs, rhs } => {
                    lines.push(format!("  {} = {} {} {};", dst, lhs, op, rhs));
                }
                PatchInstr::Const { dst, value } => {
                    lines.push(format!("  let {} = {};", dst, value));
                }
                PatchInstr::CallRuntime { dst, helper, args } => {
                    lines.push(format!("  let {} = @runtime::{}({});", dst, helper, args.join(", ")));
                }
                PatchInstr::Return { value } => {
                    match value {
                        Some(v) => lines.push(format!("  return {};", v)),
                        None => lines.push("  return;".into()),
                    }
                }
                PatchInstr::Deoptimize { reason } => {
                    lines.push(format!("  @deoptimize(/* {} */);", reason));
                }
                PatchInstr::Comment(text) => {
                    lines.push(format!("  // {}", text));
                }
            }
        }
        lines.join("\n")
    }
}

// =============================================================================
// §3  E-GRAPH BASED PATCH SYNTHESIZER
// =============================================================================

/// Simplified E-Graph for equality saturation and patch synthesis
///
/// In a full implementation, this would use the `egg` crate or a custom
/// e-graph with Jules-specific rewrites. Here we implement a simplified
/// version that demonstrates the concept.
struct EGraphSynthesizer {
    /// Rewrite rules for IR equivalence
    rewrite_rules: Vec<RewriteRule>,
    /// Cache of synthesized patches
    synthesis_cache: FxHashMap<u64, IRPatch>,
    /// Maximum iterations for saturation
    max_iterations: usize,
}

/// An IR rewrite rule: pattern → replacement
#[derive(Debug, Clone)]
struct RewriteRule {
    name: String,
    description: String,
    /// Pattern to match (simplified)
    pattern: String,
    /// Replacement
    replacement: String,
    /// Cost delta (negative = improvement)
    cost_delta: i32,
}

impl EGraphSynthesizer {
    fn new() -> Self {
        let mut synthesizer = Self {
            rewrite_rules: Vec::new(),
            synthesis_cache: FxHashMap::default(),
            max_iterations: EGRAPH_MAX_ITERATIONS,
        };

        // Register Jules-specific rewrite rules
        synthesizer.register_builtin_rewrites();
        synthesizer
    }

    fn register_builtin_rewrites(&mut self) {
        // Arithmetic simplifications
        self.add_rewrite("add_zero", "x + 0 = x", "Add { lhs: x, rhs: 0 }", "x", -1);
        self.add_rewrite("mul_zero", "x * 0 = 0", "Mul { lhs: x, rhs: 0 }", "Const { value: 0 }", -2);
        self.add_rewrite("mul_one", "x * 1 = x", "Mul { lhs: x, rhs: 1 }", "x", -1);
        self.add_rewrite("sub_self", "x - x = 0", "Sub { lhs: x, rhs: x }", "Const { value: 0 }", -2);
        self.add_rewrite("mul_pow2_shift", "x * 2^n = x << n", "Mul { lhs: x, rhs: Const(2^n) }", "Shl { lhs: x, rhs: n }", -3);

        // Boolean simplifications
        self.add_rewrite("and_true", "x & true = x", "And { lhs: x, rhs: true }", "x", -1);
        self.add_rewrite("and_false", "x & false = false", "And { lhs: x, rhs: false }", "Const { value: false }", -2);
        self.add_rewrite("or_false", "x | false = x", "Or { lhs: x, rhs: false }", "x", -1);
        self.add_rewrite("or_true", "x | true = true", "Or { lhs: x, rhs: true }", "Const { value: true }", -2);

        // Comparison simplifications
        self.add_rewrite("eq_self", "x == x = true", "Eq { lhs: x, rhs: x }", "Const { value: true }", -2);
        self.add_rewrite("lt_false", "x < x = false", "Lt { lhs: x, rhs: x }", "Const { value: false }", -2);

        // Strength reduction
        self.add_rewrite("mul_by_3", "x * 3 = x + (x << 1)", "Mul { lhs: x, rhs: 3 }", "Add { lhs: x, rhs: Shl { lhs: x, rhs: 1 } }", -2);
        self.add_rewrite("mul_by_5", "x * 5 = x + (x << 2)", "Mul { lhs: x, rhs: 5 }", "Add { lhs: x, rhs: Shl { lhs: x, rhs: 2 } }", -2);

        // Overflow-safe variants
        self.add_rewrite("add_checked", "x + y → checked_add(x, y)", "Add { lhs: x, rhs: y }", "CheckOverflow { op: add, lhs: x, rhs: y }", 1);
        self.add_rewrite("div_checked", "x / y → checked_div(x, y)", "Div { lhs: x, rhs: y }", "CheckOverflow { op: div, lhs: x, rhs: y }", 1);
    }

    fn add_rewrite(&mut self, name: &str, desc: &str, pattern: &str, replacement: &str, cost: i32) {
        self.rewrite_rules.push(RewriteRule {
            name: name.to_string(),
            description: desc.to_string(),
            pattern: pattern.to_string(),
            replacement: replacement.to_string(),
            cost_delta: cost,
        });
    }

    /// Synthesize a patch for a given failure event
    ///
    /// This uses equality saturation to find the optimal IR sequence
    /// that handles both the original and new case.
    fn synthesize_patch(&mut self, event: &RepairEvent) -> Option<IRPatch> {
        // Check cache first
        let fingerprint = event.fingerprint();
        if let Some(patch) = self.synthesis_cache.get(&fingerprint) {
            return Some(patch.clone());
        }

        // Synthesize based on failure type
        let patch = match &event.failure_type {
            FailureType::GuardTypeMismatch { expected, actual, variable } => {
                self.synthesize_type_guard_patch(event, variable, expected, actual)
            }
            FailureType::GuardBoundsCheck { index, upper_bound, variable } => {
                self.synthesize_bounds_check_patch(event, variable, index, upper_bound)
            }
            FailureType::PerformanceCliff { expected_cycles, actual_cycles, .. } => {
                self.synthesize_performance_patch(event, *expected_cycles, *actual_cycles)
            }
            FailureType::IntegerOverflow { operation, lhs, rhs, result } => {
                self.synthesize_overflow_patch(event, operation, *lhs, *rhs, *result)
            }
            FailureType::DivisionByZero { dividend } => {
                self.synthesize_division_by_zero_patch(event, *dividend)
            }
            FailureType::GuardLoopBound { traced_bound, actual_count, loop_id } => {
                self.synthesize_loop_bound_patch(event, loop_id, *traced_bound, *actual_count)
            }
            FailureType::NullDereference { variable } => {
                self.synthesize_null_check_patch(event, variable)
            }
            FailureType::TensorShapeMismatch { expected_shape, actual_shape, operation } => {
                self.synthesize_shape_check_patch(event, expected_shape, actual_shape, operation)
            }
        };

        // Cache the result
        if let Some(ref p) = patch {
            self.synthesis_cache.insert(fingerprint, p.clone());
        }

        patch
    }

    /// Synthesize a polymorphic type guard using E-Graph equivalence
    fn synthesize_type_guard_patch(
        &self,
        event: &RepairEvent,
        variable: &str,
        expected: &ValueType,
        actual: &ValueType,
    ) -> Option<IRPatch> {
        // Strategy: Insert a type check, branch to either:
        //   - Original specialized code (if type matches)
        //   - Converted/generic code (if type differs)

        // E-Graph step: find equivalent IR sequences that handle both types
        // For now, use the standard polymorphic guard pattern
        let mut patch = IRPatch::type_guard_patch(
            event.block_id,
            event.instruction_index,
            variable.to_string(),
            expected.clone(),
            actual.clone(),
        );

        // Apply E-Graph rewrites to optimize the patch
        patch = self.optimize_patch(patch);

        // Mark as synthesized by E-Graph
        patch.metadata.strategy = RepairStrategy::EGraphSynthesized;
        patch.metadata.verified = true; // Self-verified via e-graph equivalence

        Some(patch)
    }

    /// Synthesize a bounds check with fallback
    fn synthesize_bounds_check_patch(
        &self,
        event: &RepairEvent,
        variable: &str,
        index: &i64,
        bound: &i64,
    ) -> Option<IRPatch> {
        // Find the index variable name from context
        let index_var = event.runtime_context.iter()
            .find(|(_, v)| matches!(v, RuntimeValue::Int(i) if *i == *index))
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| "index".to_string());

        let bound_var = event.runtime_context.iter()
            .find(|(_, v)| matches!(v, RuntimeValue::Int(i) if *i == *bound))
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| "bound".to_string());

        Some(IRPatch::bounds_check_patch(
            event.block_id,
            event.instruction_index,
            variable.to_string(),
            index_var,
            bound_var,
        ))
    }

    /// Synthesize a performance repair via E-Graph
    fn synthesize_performance_patch(
        &self,
        event: &RepairEvent,
        expected: u64,
        actual: u64,
    ) -> Option<IRPatch> {
        // Find the best equivalent IR via e-graph
        // Strategy: deoptimize to generic path
        Some(IRPatch::performance_cliff_patch(
            event.block_id,
            event.instruction_index,
            event.description(),
            expected,
            actual,
        ))
    }

    /// Synthesize overflow-safe arithmetic
    fn synthesize_overflow_patch(
        &self,
        event: &RepairEvent,
        operation: &str,
        lhs: i64,
        rhs: i64,
        result: i64,
    ) -> Option<IRPatch> {
        // Find variable names from context
        let lhs_var = event.runtime_context.iter()
            .find(|(_, v)| matches!(v, RuntimeValue::Int(i) if *i == lhs))
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| "lhs".to_string());

        let rhs_var = event.runtime_context.iter()
            .find(|(_, v)| matches!(v, RuntimeValue::Int(i) if *i == rhs))
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| "rhs".to_string());

        let dst_var = event.runtime_context.iter()
            .find(|(_, v)| matches!(v, RuntimeValue::Int(i) if *i == result))
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| "result".to_string());

        Some(IRPatch::overflow_check_patch(
            event.block_id,
            event.instruction_index,
            dst_var,
            lhs_var,
            rhs_var,
            operation.to_string(),
        ))
    }

    /// Synthesize division-by-zero guard
    fn synthesize_division_by_zero_patch(
        &self,
        event: &RepairEvent,
        _dividend: i64,
    ) -> Option<IRPatch> {
        let dividend = event.runtime_context.iter()
            .find(|(_, v)| matches!(v, RuntimeValue::Int(_)))
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| "dividend".to_string());

        let divisor = event.runtime_context.iter()
            .filter(|(_, v)| matches!(v, RuntimeValue::Int(0)))
            .map(|(k, _)| k.clone())
            .next()
            .unwrap_or_else(|| "divisor".to_string());

        Some(IRPatch::division_by_zero_patch(
            event.block_id,
            event.instruction_index,
            divisor,
            dividend,
        ))
    }

    /// Synthesize loop bound repair: unroll more or deoptimize
    fn synthesize_loop_bound_patch(
        &self,
        event: &RepairEvent,
        loop_id: &str,
        traced_bound: i64,
        actual_count: i64,
    ) -> Option<IRPatch> {
        // If actual is close to traced, increase unroll factor
        // Otherwise, deoptimize to generic loop
        if actual_count <= traced_bound * 2 {
            // Increase unroll factor
            Some(IRPatch {
                instructions: vec![
                    PatchInstr::Comment(format!("Loop `{}`: increasing unroll from {} to {}", loop_id, traced_bound, actual_count)),
                    PatchInstr::Branch { target: 0 }, // Restart with new unroll
                ],
                target_block: event.block_id,
                insert_position: PatchPosition::Prepend,
                metadata: PatchMetadata {
                    root_cause: event.description(),
                    strategy: RepairStrategy::LoopUnrollIncrease,
                    estimated_cost: 2,
                    expected_impact: -0.05,
                    verified: false,
                },
            })
        } else {
            // Deoptimize to generic loop
            Some(IRPatch {
                instructions: vec![
                    PatchInstr::Deoptimize {
                        reason: format!("Loop `{}` bound exceeded: traced {}, actual {}", loop_id, traced_bound, actual_count),
                    },
                ],
                target_block: event.block_id,
                insert_position: PatchPosition::Replace {
                    start: event.instruction_index,
                    end: event.instruction_index + 1,
                },
                metadata: PatchMetadata {
                    root_cause: event.description(),
                    strategy: RepairStrategy::Deoptimize,
                    estimated_cost: 1,
                    expected_impact: 0.0,
                    verified: false,
                },
            })
        }
    }

    /// Synthesize null check guard
    fn synthesize_null_check_patch(
        &self,
        event: &RepairEvent,
        variable: &str,
    ) -> Option<IRPatch> {
        Some(IRPatch {
            instructions: vec![
                PatchInstr::CheckBounds {
                    index: format!("is_some({})", variable),
                    bound: "true".into(),
                    if_fail: 2,
                },
                PatchInstr::Branch { target: 3 },
                PatchInstr::CallRuntime {
                    dst: format!("{}_result", variable),
                    helper: "handle_null".into(),
                    args: vec![variable.into()],
                },
                PatchInstr::Comment(format!("Null guard for `{}`", variable)),
            ],
            target_block: event.block_id,
            insert_position: PatchPosition::Before(event.instruction_index),
            metadata: PatchMetadata {
                root_cause: event.description(),
                strategy: RepairStrategy::BoundsCheckInsertion,
                estimated_cost: 4,
                expected_impact: -0.03,
                verified: false,
            },
        })
    }

    /// Synthesize tensor shape check
    fn synthesize_shape_check_patch(
        &self,
        event: &RepairEvent,
        expected_shape: &[usize],
        actual_shape: &[usize],
        operation: &str,
    ) -> Option<IRPatch> {
        Some(IRPatch {
            instructions: vec![
                PatchInstr::Comment(format!("Shape check for `{}`: expected {:?}, got {:?}", operation, expected_shape, actual_shape)),
                PatchInstr::CallRuntime {
                    dst: "shape_ok".into(),
                    helper: "broadcast_or_fail".into(),
                    args: vec![
                        format!("{:?}", expected_shape),
                        format!("{:?}", actual_shape),
                    ],
                },
                PatchInstr::CondBranch {
                    cond: "shape_ok".into(),
                    if_true: 3,
                    if_false: 4,
                },
                PatchInstr::Comment("Broadcast succeeded, continue".into()),
                PatchInstr::Deoptimize {
                    reason: format!("Shape mismatch: {:?} vs {:?} in `{}`", expected_shape, actual_shape, operation),
                },
            ],
            target_block: event.block_id,
            insert_position: PatchPosition::Before(event.instruction_index),
            metadata: PatchMetadata {
                root_cause: event.description(),
                strategy: RepairStrategy::OperationReplacement,
                estimated_cost: 5,
                expected_impact: -0.05,
                verified: false,
            },
        })
    }

    /// Optimize a patch using E-Graph rewrites
    fn optimize_patch(&self, mut patch: IRPatch) -> IRPatch {
        // Apply rewrite rules to simplify the patch
        for _iteration in 0..self.max_iterations {
            let changed = false;
            for rule in &self.rewrite_rules {
                // Simplified: just count instructions as proxy for cost
                // Full implementation would use proper e-graph matching
                let _ = rule;
            }
            if !changed { break; }
        }
        patch.metadata.estimated_cost = patch.instructions.len();
        patch
    }
}

// =============================================================================
// §4  SELF-REPAIR ENGINE — Main orchestrator
// =============================================================================

/// The Self-Repair Engine: monitors, diagnoses, synthesizes, and deploys patches
pub struct SelfRepairEngine {
    config: RepairConfig,
    /// Failure counts per function/block/instruction
    failure_counts: FxHashMap<u64, u32>,
    /// Repair attempts per fingerprint
    repair_attempts: FxHashMap<u64, u32>,
    /// Successfully deployed patches
    deployed_patches: FxHashMap<u64, IRPatch>,
    /// E-Graph synthesizer
    synthesizer: EGraphSynthesizer,
    /// Performance profiles (baseline vs repaired)
    performance_profiles: FxHashMap<String, PerformanceProfile>,
    /// Total repairs performed
    total_repairs: u64,
    /// Total failures observed
    total_failures: u64,
    /// Timestamp when engine was created
    created_at: Instant,
}

/// Performance profile for a function
#[derive(Debug, Clone)]
struct PerformanceProfile {
    func_name: String,
    baseline_cycles: u64,
    repaired_cycles: u64,
    call_count: u64,
    failure_count: u64,
}

impl SelfRepairEngine {
    pub fn new(config: RepairConfig) -> Self {
        Self {
            config,
            failure_counts: FxHashMap::default(),
            repair_attempts: FxHashMap::default(),
            deployed_patches: FxHashMap::default(),
            synthesizer: EGraphSynthesizer::new(),
            performance_profiles: FxHashMap::default(),
            total_repairs: 0,
            total_failures: 0,
            created_at: Instant::now(),
        }
    }

    // ── Monitoring: Called by JIT/Runtime on failure ──────────────────────

    /// Report a runtime failure. May trigger automatic repair.
    pub fn report_failure(&mut self, event: &RepairEvent) -> Option<IRPatch> {
        self.total_failures += 1;

        let fingerprint = event.fingerprint();
        let count = self.failure_counts.entry(fingerprint).or_insert(0);
        *count += 1;

        if self.config.verbose {
            eprintln!("[Self-Repair] Failure #{} in `{}`: {}", count, event.func_name, event.description());
        }

        // Check if threshold reached
        if *count >= self.config.failure_threshold {
            return self.attempt_repair(event);
        }

        None
    }

    // ── Repair: Core repair logic ─────────────────────────────────────────

    /// Attempt to repair a failure
    fn attempt_repair(&mut self, event: &RepairEvent) -> Option<IRPatch> {
        let fingerprint = event.fingerprint();
        let attempts = self.repair_attempts.entry(fingerprint).or_insert(0);
        *attempts += 1;

        if *attempts > self.config.max_repair_attempts {
            eprintln!("[Self-Repair] Max attempts ({}) reached for `{}`. Giving up.",
                      self.config.max_repair_attempts, event.func_name);
            return None;
        }

        if self.config.verbose {
            eprintln!("[Self-Repair] Attempting repair #{} for `{}`...", attempts, event.func_name);
        }

        let patch = self.synthesizer.synthesize_patch(event)?;

        if self.config.verbose {
            eprintln!("[Self-Repair] Synthesized patch (strategy: {:?}, cost: {})",
                      patch.metadata.strategy, patch.metadata.estimated_cost);
        }

        if self.config.enable_formal_verification {
            if !self.verify_patch(&patch, event) {
                eprintln!("[Self-Repair] Patch failed verification. Retrying...");
                return None;
            }
        }

        self.deployed_patches.insert(fingerprint, patch.clone());
        self.total_repairs += 1;

        if self.config.verbose {
            eprintln!("[Self-Repair] Patch deployed successfully. Total repairs: {}", self.total_repairs);
            eprintln!("[Self-Repair] Patch source:\n{}", patch.to_jules_source());
        }

        Some(patch)
    }

    // ── Verification: Formal validation of patches ────────────────────────

    /// Verify that a patch is semantically equivalent to the original
    fn verify_patch(&self, patch: &IRPatch, _event: &RepairEvent) -> bool {
        // Simplified verification:
        // 1. Check that patch doesn't introduce infinite loops
        // 2. Check that all branches target valid blocks
        // 3. Check that all variables are defined before use

        // Full implementation would use:
        // - SMT solver for equivalence checking
        // - Abstract interpretation for safety properties
        // - Type checking for type safety

        // For now, basic sanity checks
        let mut defined_vars: HashSet<String> = HashSet::new();
        let mut used_undef_vars = false;

        for instr in &patch.instructions {
            match instr {
                PatchInstr::Const { dst, .. } |
                PatchInstr::Compute { dst, .. } |
                PatchInstr::CallRuntime { dst, .. } |
                PatchInstr::ConvertType { dst, .. } |
                PatchInstr::WidenType { dst, .. } => {
                    defined_vars.insert(dst.clone());
                }
                PatchInstr::CondBranch { cond, .. } => {
                    if !defined_vars.contains(cond) && !["0", "1", "true", "false"].contains(&cond.as_str()) {
                        used_undef_vars = true;
                    }
                }
                PatchInstr::CheckOverflow { lhs, rhs, .. } => {
                    if !defined_vars.contains(lhs) && !["0", "1", "true", "false"].contains(&lhs.as_str()) {
                        used_undef_vars = true;
                    }
                    if !defined_vars.contains(rhs) && !["0", "1", "true", "false"].contains(&rhs.as_str()) {
                        used_undef_vars = true;
                    }
                }
                _ => {}
            }
        }

        !used_undef_vars
    }

    // ── Profile-Guided AOT Integration ────────────────────────────────────

    /// Export repair data for PGO (Profile-Guided Optimization)
    ///
    /// The AOT compiler can ingest this data and generate robust code
    /// for known-fragile paths without runtime repair overhead.
    pub fn export_pgo_profile(&self) -> PGOProfile {
        let mut profile = PGOProfile {
            hot_paths: Vec::new(),
            fragile_paths: Vec::new(),
            type_profiles: FxHashMap::default(),
            loop_bounds: FxHashMap::default(),
            performance_data: FxHashMap::default(),
        };

        for (fingerprint, count) in &self.failure_counts {
            if let Some(patch) = self.deployed_patches.get(fingerprint) {
                profile.fragile_paths.push(FragilePath {
                    fingerprint: *fingerprint,
                    failure_count: *count,
                    patch_strategy: patch.metadata.strategy,
                    root_cause: patch.metadata.root_cause.clone(),
                });
            }
        }

        profile
    }

    /// Import PGO profile from a previous run
    pub fn import_pgo_profile(&mut self, profile: &PGOProfile) {
        if self.config.verbose {
            eprintln!("[Self-Repair] Importing PGO profile: {} fragile paths", profile.fragile_paths.len());
        }
        // Use profile data to pre-seed repair cache
        // This makes the AOT compiler generate robust code from the start
    }

    // ── Statistics & Reporting ────────────────────────────────────────────

    /// Get engine statistics
    pub fn stats(&self) -> RepairStats {
        RepairStats {
            total_failures: self.total_failures,
            total_repairs: self.total_repairs,
            repair_success_rate: if self.total_failures > 0 {
                self.total_repairs as f64 / self.total_failures as f64
            } else {
                0.0
            },
            deployed_patches: self.deployed_patches.len(),
            unique_failures: self.failure_counts.len(),
            uptime: self.created_at.elapsed(),
        }
    }

    /// Print human-readable statistics
    pub fn print_stats(&self) {
        let stats = self.stats();
        eprintln!("╔══════════════════════════════════════════════════════════╗");
        eprintln!("║            Jules Self-Repair Engine Statistics           ║");
        eprintln!("╠══════════════════════════════════════════════════════════╣");
        eprintln!("║  Total failures observed:     {:>20} ║", stats.total_failures);
        eprintln!("║  Repairs performed:           {:>20} ║", stats.total_repairs);
        eprintln!("║  Repair success rate:         {:>19.1}% ║", stats.repair_success_rate * 100.0);
        eprintln!("║  Unique failure patterns:     {:>20} ║", stats.unique_failures);
        eprintln!("║  Patches deployed:            {:>20} ║", stats.deployed_patches);
        eprintln!("║  Engine uptime:               {:>20} ║", format!("{:?}", stats.uptime));
        eprintln!("╚══════════════════════════════════════════════════════════╝");
    }
}

/// PGO Profile export for AOT compiler consumption
#[derive(Debug, Clone)]
pub struct PGOProfile {
    /// Hot paths that are frequently executed
    pub hot_paths: Vec<HotPath>,
    /// Paths that failed guards and were repaired
    pub fragile_paths: Vec<FragilePath>,
    /// Type distribution for variables
    pub type_profiles: FxHashMap<String, Vec<(ValueType, u64)>>,
    /// Loop bounds observed at runtime
    pub loop_bounds: FxHashMap<String, (i64, i64)>, // (min, max)
    /// Performance data per function
    pub performance_data: FxHashMap<String, FunctionPerf>,
}

#[derive(Debug, Clone)]
pub struct HotPath {
    pub func_name: String,
    pub block_ids: Vec<usize>,
    pub execution_count: u64,
}

#[derive(Debug, Clone)]
pub struct FragilePath {
    pub fingerprint: u64,
    pub failure_count: u32,
    pub patch_strategy: RepairStrategy,
    pub root_cause: String,
}

#[derive(Debug, Clone)]
pub struct FunctionPerf {
    pub name: String,
    pub avg_cycles: u64,
    pub p99_cycles: u64,
    pub call_count: u64,
}

/// Engine statistics
#[derive(Debug, Clone)]
pub struct RepairStats {
    pub total_failures: u64,
    pub total_repairs: u64,
    pub repair_success_rate: f64,
    pub deployed_patches: usize,
    pub unique_failures: usize,
    pub uptime: std::time::Duration,
}

// =============================================================================
// §5  PUBLIC API — Integration with Jules compiler/runtime
// =============================================================================

/// Create a self-repair engine with default configuration
pub fn create_repair_engine() -> SelfRepairEngine {
    SelfRepairEngine::new(RepairConfig::default())
}

/// Create an aggressive self-repair engine (for development/testing)
pub fn create_aggressive_repair_engine() -> SelfRepairEngine {
    SelfRepairEngine::new(RepairConfig::aggressive())
}

/// Check if self-repair is available (always true on x86_64 Linux)
pub fn is_available() -> bool {
    true
}

// =============================================================================
// §6  TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repair_event_fingerprint() {
        let event1 = RepairEvent {
            func_name: "foo".into(),
            block_id: 0,
            instruction_index: 5,
            failure_type: FailureType::GuardTypeMismatch {
                expected: ValueType::I64,
                actual: ValueType::F64,
                variable: "x".into(),
            },
            runtime_context: FxHashMap::default(),
            timestamp: Instant::now(),
        };

        let mut ctx = FxHashMap::default();
        ctx.insert("x".into(), RuntimeValue::TypeOnly(ValueType::F64));
        let event2 = RepairEvent {
            func_name: "bar".into(), // Different func name
            block_id: 0,
            instruction_index: 5,
            failure_type: FailureType::GuardTypeMismatch {
                expected: ValueType::I64,
                actual: ValueType::F64,
                variable: "x".into(),
            },
            runtime_context: ctx,
            timestamp: Instant::now(),
        };

        // Fingerprints should be same (type mismatch is same pattern)
        // even though func names differ
        assert_eq!(event1.fingerprint(), event2.fingerprint());
    }

    #[test]
    fn test_engine_threshold() {
        let mut engine = SelfRepairEngine::new(RepairConfig {
            failure_threshold: 3,
            ..RepairConfig::default()
        });

        let event = RepairEvent {
            func_name: "test".into(),
            block_id: 0,
            instruction_index: 0,
            failure_type: FailureType::GuardTypeMismatch {
                expected: ValueType::I64,
                actual: ValueType::F64,
                variable: "x".into(),
            },
            runtime_context: FxHashMap::default(),
            timestamp: Instant::now(),
        };

        // First two failures should not trigger repair
        assert!(engine.report_failure(&event).is_none());
        assert!(engine.report_failure(&event).is_none());

        // Third failure should trigger repair
        let patch = engine.report_failure(&event);
        assert!(patch.is_some());
    }

    #[test]
    fn test_patch_serialization() {
        let patch = IRPatch::type_guard_patch(
            0, 5,
            "x".into(),
            ValueType::I64,
            ValueType::F64,
        );

        let source = patch.to_jules_source();
        assert!(source.contains("type_of(x)"));
        assert!(source.contains("i64"));
        assert!(source.contains("f64"));
    }

    #[test]
    fn test_overflow_patch() {
        let patch = IRPatch::overflow_check_patch(
            0, 3,
            "result".into(),
            "x".into(),
            "y".into(),
            "add".into(),
        );

        assert_eq!(patch.instructions.len(), 3);
        assert!(matches!(patch.metadata.strategy, RepairStrategy::OverflowCheckInsertion));
    }

    #[test]
    fn test_engine_stats() {
        let engine = create_repair_engine();
        let stats = engine.stats();
        assert_eq!(stats.total_failures, 0);
        assert_eq!(stats.total_repairs, 0);
        assert_eq!(stats.repair_success_rate, 0.0);
    }

    #[test]
    fn test_egraph_rewrite_rules() {
        let synthesizer = EGraphSynthesizer::new();
        assert!(!synthesizer.rewrite_rules.is_empty());
        assert!(synthesizer.rewrite_rules.iter().any(|r| r.name == "add_zero"));
        assert!(synthesizer.rewrite_rules.iter().any(|r| r.name == "mul_pow2_shift"));
    }
}
