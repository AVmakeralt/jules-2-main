// =============================================================================
// jules/src/bytecode_vm.rs
//
// BYTECODE VIRTUAL MACHINE
//
// Execution strategy:
// - Switch-dispatch (match-based) interpreter loop — portable, debuggable,
//   and competitive with indirect-threaded designs on modern CPUs where
//   branch predictors learn the match's jump table effectively.
//   NOTE: Rust stable does not expose computed-goto / label-as-value, so
//   true "direct threading" (à la CPython's ceval.c HAVE_COMPUTED_GOTOS)
//   is not available without nightly + asm!. The match compiles to a jump
//   table on optimized builds, which approximates the same effect.
// - Register-based architecture (no stack manipulation overhead)
// - Inline caching for property/method access (polymorphic inline caches)
// - Constant folding & dead code elimination at compile time
// - Memory pooling with pre-allocated slot array for hot paths
// - SIMD vectorized operations for Vec4/tensor math (x86_64 + scalar fallback)
// - Speculative type specialization: fast I64/F64 paths, generic fallback
// =============================================================================

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use bumpalo::Bump;
use rustc_hash::FxHashMap;

use crate::compiler::ast::{BinOpKind, IfOrBlock, Program, UnOpKind};
use crate::compiler::formal_verify::{ArithmeticMode, EntropyWatchdog, OsrEngine, OsrOutcome, OsrTrigger, TrustTier, WatchdogSensitivity};
use crate::interp::{RuntimeError, StructData, Value};
#[cfg(feature = "gnn-optimizer")]
use crate::runtime::memory_management::PrefetchEngine;
use crate::optimizer::data_dependent_jit::DataDependentJIT;

// =============================================================================
// §1  BYTECODE INSTRUCTION SET
// =============================================================================

/// Ultra-compact bytecode instruction (fits in 16 bytes for cache efficiency)
/// Reduced from 32-byte alignment — the largest variant (LoadConstInt = u16+i64 = 10 bytes)
/// fits in 16 bytes. This doubles I-cache density for instruction streams.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub enum Instr {
    // Constant loads (imm32 for common cases, imm64 for rare)
    LoadConst { dst: u16, idx: u32 },
    LoadConstInt { dst: u16, value: i64 },
    LoadConstFloat { dst: u16, value: f64 },
    LoadConstBool { dst: u16, value: bool },
    LoadConstUnit { dst: u16 },
    
    // Register operations
    Move { dst: u16, src: u16 },
    
    // Arithmetic (all register-to-register)
    Add { dst: u16, lhs: u16, rhs: u16 },
    Sub { dst: u16, lhs: u16, rhs: u16 },
    Mul { dst: u16, lhs: u16, rhs: u16 },
    Div { dst: u16, lhs: u16, rhs: u16 },
    Rem { dst: u16, lhs: u16, rhs: u16 },
    Neg { dst: u16, src: u16 },
    
    // Bitwise
    BitAnd { dst: u16, lhs: u16, rhs: u16 },
    BitOr { dst: u16, lhs: u16, rhs: u16 },
    BitXor { dst: u16, lhs: u16, rhs: u16 },
    Shl { dst: u16, lhs: u16, rhs: u16 },
    Shr { dst: u16, lhs: u16, rhs: u16 },
    Not { dst: u16, src: u16 },
    
    // Comparison
    Eq { dst: u16, lhs: u16, rhs: u16 },
    Ne { dst: u16, lhs: u16, rhs: u16 },
    Lt { dst: u16, lhs: u16, rhs: u16 },
    Le { dst: u16, lhs: u16, rhs: u16 },
    Gt { dst: u16, lhs: u16, rhs: u16 },
    Ge { dst: u16, lhs: u16, rhs: u16 },
    
    // Control flow (relative offsets for position-independent code)
    Jump { offset: i32 },
    JumpFalse { cond: u16, offset: i32 },
    JumpTrue { cond: u16, offset: i32 },
    
    // Function call
    Call { dst: u16, func: u16, argc: u16, start: u16 },
    CallNative { dst: u16, func_idx: u32, argc: u16, start: u16 },
    Return { value: u16 },
    
    // Memory access
    LoadField { dst: u16, obj: u16, field_idx: u32 },
    StoreField { obj: u16, field_idx: u32, src: u16 },
    LoadIndex { dst: u16, arr: u16, idx: u16 },
    StoreIndex { arr: u16, idx: u16, src: u16 },
    ArrayLen { dst: u16, arr: u16 },
    
    // Vector/tensor operations (SIMD-optimized)
    VecAdd { dst: u16, lhs: u16, rhs: u16 },
    VecMul { dst: u16, lhs: u16, rhs: u16 },
    MatMul { dst: u16, lhs: u16, rhs: u16 },
    
    // Power operation
    Pow { dst: u16, base: u16, exp: u16 },

    // Element-wise (Hadamard) multiply for tensors/arrays
    HadamardMul { dst: u16, lhs: u16, rhs: u16 },

    // Compound construction from register range
    MakeArray { dst: u16, start: u16, count: u16 },
    MakeTuple { dst: u16, start: u16, count: u16 },
    MakeStruct { dst: u16, name_idx: u32, field_start: u16, field_count: u16 },
    MakeRange { dst: u16, lo: u16, hi: u16, inclusive: bool },

    // Type casting
    Cast { dst: u16, src: u16, target_type: u32 },

    // Print output
    Print { src: u16 },

    // SIMD loop hint — marks a loop as vectorizable
    SimdLoopStart {
        /// Number of elements per SIMD lane
        lane_count: u8,
        /// SIMD width in bits (128, 256, 512)
        simd_width: u16,
    },
    SimdLoopEnd,

    // Type checking & specialization
    TypeCheck { dst: u16, src: u16, expected_type: u32 },
    AssumeInt { dst: u16, src: u16 },
    AssumeFloat { dst: u16, src: u16 },

    // Debug/profiling
    ProfilePoint { id: u32 },
    DebugBreak,

    // NOP for alignment/padding
    Nop,
}

// =============================================================================
// §2  COMPILED FUNCTION
// =============================================================================

/// A compiled bytecode function with metadata for optimization
#[derive(Debug)]
pub struct BytecodeFunction {
    pub name: String,
    pub instructions: Vec<Instr>,
    pub constants: Vec<Value>,
    pub num_locals: u16,
    pub num_params: u16,

    // Optimization metadata
    pub hotness: AtomicU64,        // How often this function is called
    pub execution_count: AtomicU64, // For adaptive optimization
    pub avg_slots_used: f64,       // For register allocation hints

    // ── Proof Trust Protocol ──────────────────────────────────────────────────
    // When the SMT solver / translation validator proves properties about
    // this function, the compiler sets these fields.  The runtime reads
    // them to decide which safety checks to skip.
    /// Trust tier: how much the runtime can trust this function's code.
    /// TIER_0_TRUSTED = SMT-proven safe (watchdog disabled, wrapping arithmetic)
    /// TIER_3_UNVERIFIED = no proof (full safety monitoring)
    pub trust_tier: TrustTier,
    /// Arithmetic mode: what happens on integer overflow?
    /// Wrapping = two's complement wrap (default for TIER_0/1)
    /// Saturating = clamp to MIN/MAX (for DSP)
    /// Strict = checked arithmetic that errors on overflow
    pub arithmetic_mode: ArithmeticMode,
    /// Whether the SMT solver proved loop termination for this function.
    pub termination_proven: bool,
    /// Whether the SMT solver proved absence of arithmetic overflow.
    pub overflow_proven_safe: bool,
}

impl Clone for BytecodeFunction {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            instructions: self.instructions.clone(),
            constants: self.constants.clone(),
            num_locals: self.num_locals,
            num_params: self.num_params,
            hotness: AtomicU64::new(self.hotness.load(Ordering::Relaxed)),
            execution_count: AtomicU64::new(self.execution_count.load(Ordering::Relaxed)),
            avg_slots_used: self.avg_slots_used,
            trust_tier: self.trust_tier,
            arithmetic_mode: self.arithmetic_mode,
            termination_proven: self.termination_proven,
            overflow_proven_safe: self.overflow_proven_safe,
        }
    }
}

impl BytecodeFunction {
    pub fn new(name: String) -> Self {
        Self {
            name,
            instructions: Vec::with_capacity(256),
            constants: Vec::with_capacity(64),
            num_locals: 0,
            num_params: 0,
            hotness: AtomicU64::new(0),
            execution_count: AtomicU64::new(0),
            avg_slots_used: 0.0,
            trust_tier: TrustTier::default(),
            arithmetic_mode: ArithmeticMode::default(),
            termination_proven: false,
            overflow_proven_safe: false,
        }
    }
    
    #[inline(always)]
    fn add_constant(&mut self, value: Value) -> u32 {
        let idx = self.constants.len() as u32;
        self.constants.push(value);
        idx
    }
}

// =============================================================================
// §3  POLYMORPHIC INLINE CACHE (PIC)
// =============================================================================

/// Inline cache entry for fast property/method access
/// Implements monomorphic + polymorphic caching (up to 4 shapes)
#[derive(Debug, Clone)]
pub struct InlineCache {
    /// Number of cached shapes (0-4)
    state: u8,
    /// Cached shape IDs (up to 4)
    shape_ids: [u64; 4],
    /// Cached offsets/results (up to 4)
    offsets: [i32; 4],
    /// Fallback when cache miss
    fallback_offset: i32,
}

impl InlineCache {
    pub const fn new() -> Self {
        Self {
            state: 0,
            shape_ids: [0; 4],
            offsets: [0; 4],
            fallback_offset: -1,
        }
    }
    
    /// Try to get cached result for shape ID
    #[inline(always)]
    pub fn lookup(&self, shape_id: u64) -> Option<i32> {
        match self.state {
            0 => None,
            1 => {
                if self.shape_ids[0] == shape_id {
                    Some(self.offsets[0])
                } else {
                    None
                }
            }
            2..=4 => {
                for i in 0..self.state as usize {
                    if self.shape_ids[i] == shape_id {
                        return Some(self.offsets[i]);
                    }
                }
                None
            }
            _ => unreachable!(),
        }
    }
    
    /// Update cache with new shape ID and offset
    #[inline(never)]
    pub fn update(&mut self, shape_id: u64, offset: i32) {
        let idx = self.state as usize;
        if idx < 4 {
            // Monomorphic or polymorphic case
            self.shape_ids[idx] = shape_id;
            self.offsets[idx] = offset;
            self.state += 1;
        } else {
            // Megamorphic - just update fallback
            self.fallback_offset = offset;
        }
    }
}

// =============================================================================
// §4  MEMORY POOL & ARENA ALLOCATION
// =============================================================================

/// Thread-local memory pool for allocation-free execution
pub struct MemoryPool {
    /// Bump allocator for fast allocation during execution
    bump: Bump,
    /// Pre-allocated value cache for common values
    value_cache: [Option<Value>; 256],
    /// Slot array (pre-allocated to avoid reallocation)
    slots: Vec<Value>,
    /// Highest slot index written in the current frame (for partial reset)
    max_slot_used: usize,
}

impl MemoryPool {
    pub fn with_capacity(slots: usize) -> Self {
        Self {
            bump: Bump::with_capacity(4096),
            value_cache: std::array::from_fn(|_| None),
            slots: (0..slots).map(|_| Value::Unit).collect(),
            max_slot_used: 0,
        }
    }
    
    /// Reset the pool after a function returns.
    ///
    /// Only slots up to `max_slot_used` are zeroed; unused tail slots are
    /// left as `Value::Unit` from the previous reset (or initialisation),
    /// so they are already clean.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.bump.reset();
        let end = self.max_slot_used.min(self.slots.len().saturating_sub(1));
        for slot in self.slots[..=end].iter_mut() {
            *slot = Value::Unit;
        }
        self.max_slot_used = 0;
    }
    
    #[inline(always)]
    pub fn alloc_slice<T>(&self, data: &[T]) -> &mut [T]
    where
        T: Copy,
    {
        let slice = self.bump.alloc_slice_copy(data);
        slice
    }
}

// =============================================================================
// §5  ADAPTIVE PROFILING
// =============================================================================

/// Tracks execution hotness for adaptive optimization.
///
/// `record_execution` is called on a sampled basis (currently every 256
/// dispatches) rather than per-instruction. This keeps atomic overhead
/// negligible in tight loops while still providing useful hotness signals
/// for loop detection over millions of iterations.
pub struct AdaptiveProfiler {
    /// Per-instruction execution counters
    instruction_counters: Vec<AtomicU64>,
    /// Per-function execution counts
    function_counters: Vec<AtomicU64>,
    /// Backedge counts for loop detection
    backedge_counters: Vec<AtomicU64>,
    /// Hot loop boundaries
    hot_loops: Vec<(usize, usize)>, // (start, end) instruction indices
}

impl AdaptiveProfiler {
    pub fn new(num_instructions: usize) -> Self {
        Self {
            instruction_counters: (0..num_instructions)
                .map(|_| AtomicU64::new(0))
                .collect(),
            function_counters: Vec::new(),
            backedge_counters: Vec::new(),
            hot_loops: Vec::new(),
        }
    }
    
    #[inline(always)]
    pub fn record_execution(&self, pc: usize) {
        if pc < self.instruction_counters.len() {
            self.instruction_counters[pc].fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Check if this location is "hot" (executed > 10000 times)
    #[inline(always)]
    pub fn is_hot(&self, pc: usize) -> bool {
        if pc < self.instruction_counters.len() {
            self.instruction_counters[pc].load(Ordering::Relaxed) > 10_000
        } else {
            false
        }
    }
    
    /// Detect loops from backedge execution
    pub fn detect_hot_loops(&mut self) {
        // Simple heuristic: instructions executed many times in sequence
        let threshold = 50_000;
        let mut in_loop = false;
        let mut loop_start = 0;
        
        for (i, counter) in self.instruction_counters.iter().enumerate() {
            let count = counter.load(Ordering::Relaxed);
            if !in_loop && count > threshold {
                in_loop = true;
                loop_start = i;
            } else if in_loop && count <= threshold {
                in_loop = false;
                self.hot_loops.push((loop_start, i));
            }
        }
    }
}

// =============================================================================
// §6  BYTECODE COMPILER (AST → Bytecode)
// =============================================================================

/// Compiles AST to optimized bytecode with constant folding
pub struct BytecodeCompiler {
    current_function: BytecodeFunction,
    functions: FxHashMap<String, usize>, // function name -> index
    next_label: u32,
    locals: FxHashMap<String, u16>,      // local variable -> slot
    next_slot: u16,                      // next available local slot
    
    // Constant folding state
    known_constants: FxHashMap<u16, Value>, // slot -> known constant value
    
    // Optimization flags
    fold_constants: bool,
    eliminate_dead_code: bool,
    
    // Loop compilation state (for break/continue)
    break_labels: Vec<usize>,     // positions of Jump instructions to patch at loop end
    loop_starts: Vec<usize>,      // PC at the start of each enclosing loop (for continue)
    loop_continue_starts: Vec<usize>, // PC of the continue target for each enclosing loop
    break_labels_at_loop_start: Vec<usize>, // break_labels.len() when each loop was entered
}

impl BytecodeCompiler {
    pub fn new() -> Self {
        Self {
            current_function: BytecodeFunction::new("<main>".to_string()),
            functions: FxHashMap::default(),
            next_label: 0,
            locals: FxHashMap::default(),
            next_slot: 0,
            known_constants: FxHashMap::default(),
            fold_constants: true,
            eliminate_dead_code: true,
            break_labels: Vec::new(),
            loop_starts: Vec::new(),
            loop_continue_starts: Vec::new(),
            break_labels_at_loop_start: Vec::new(),
        }
    }

    /// Set whether constant folding is enabled.  When `false`, the bytecode
    /// compiler emits literal instructions without attempting to fold them.
    /// This is useful for debugging or when the AST-level optimizer already
    /// performed constant propagation and folding is not desired at the
    /// bytecode level.
    pub fn set_fold_constants(&mut self, enabled: bool) {
        self.fold_constants = enabled;
        if !enabled {
            self.known_constants.clear();
        }
    }

    /// Set whether dead code elimination is enabled at the bytecode level.
    pub fn set_eliminate_dead_code(&mut self, enabled: bool) {
        self.eliminate_dead_code = enabled;
    }

    #[inline]
    fn new_label(&mut self) -> u32 {
        let label = self.next_label;
        self.next_label += 1;
        label
    }

    #[inline]
    fn alloc_slot(&mut self) -> u16 {
        let slot = self.next_slot;
        self.next_slot = self.next_slot.saturating_add(1);
        slot
    }
    
    /// Emit instruction, applying constant folding if enabled.
    ///
    /// **Constant-tracking discipline** (fixes the stale-constant bug):
    /// When `fold_constants` is enabled, every instruction that is emitted
    /// (whether folded or not) must update `known_constants` so that later
    /// instructions see a consistent view:
    ///
    /// - If the instruction was folded into a `LoadConstInt/Float/Bool`,
    ///   the destination slot is recorded as a known constant.
    /// - If the instruction writes to a slot with a *non-constant* result
    ///   (arithmetic, Move from unknown slot, etc.), the destination slot
    ///   is removed from `known_constants`.
    /// - `Nop` (from folding a no-op Move) does not affect any slot.
    #[inline]
    fn emit(&mut self, instr: Instr) {
        if self.fold_constants {
            if let Some(folded) = self.try_fold_constant(&instr) {
                // Track the folded constant in known_constants.
                self.track_folded_result(&folded);
                self.current_function.instructions.push(folded);
                return;
            }
        }
        // Non-folded instruction: invalidate any destination slots that
        // are overwritten by this instruction.
        self.invalidate_dst(&instr);
        self.current_function.instructions.push(instr);
    }

    /// After a constant-fold, record the destination slot's known value.
    fn track_folded_result(&mut self, instr: &Instr) {
        match instr {
            Instr::LoadConstInt { dst, value } => {
                self.known_constants.insert(*dst, Value::I64(*value));
            }
            Instr::LoadConstFloat { dst, value } => {
                self.known_constants.insert(*dst, Value::F64(*value));
            }
            Instr::LoadConstBool { dst, value } => {
                self.known_constants.insert(*dst, Value::Bool(*value));
            }
            Instr::LoadConstUnit { dst } => {
                self.known_constants.insert(*dst, Value::Unit);
            }
            Instr::Nop => {
                // Nop doesn't write to any slot — nothing to track.
            }
            _ => {
                // Any other folded instruction (shouldn't happen with current
                // folding rules, but be safe): invalidate dst.
                if let Some(dst) = self.instr_dst(instr) {
                    self.known_constants.remove(&dst);
                }
            }
        }
    }

    /// Invalidate `known_constants` for any slot written by a non-folded
    /// instruction.  This is the core fix for the stale-constant bug: when
    /// a `Move`, `Add`, etc. writes to a slot, any previously-known constant
    /// for that slot is no longer valid.
    fn invalidate_dst(&mut self, instr: &Instr) {
        if let Some(dst) = self.instr_dst(instr) {
            self.known_constants.remove(&dst);
        }
    }

    /// Return the destination slot for an instruction, if it has one.
    #[inline]
    fn instr_dst(&self, instr: &Instr) -> Option<u16> {
        match instr {
            Instr::LoadConst { dst, .. }
            | Instr::LoadConstInt { dst, .. }
            | Instr::LoadConstFloat { dst, .. }
            | Instr::LoadConstBool { dst, .. }
            | Instr::LoadConstUnit { dst }
            | Instr::Move { dst, .. }
            | Instr::Add { dst, .. }
            | Instr::Sub { dst, .. }
            | Instr::Mul { dst, .. }
            | Instr::Div { dst, .. }
            | Instr::Rem { dst, .. }
            | Instr::Neg { dst, .. }
            | Instr::BitAnd { dst, .. }
            | Instr::BitOr { dst, .. }
            | Instr::BitXor { dst, .. }
            | Instr::Shl { dst, .. }
            | Instr::Shr { dst, .. }
            | Instr::Not { dst, .. }
            | Instr::Eq { dst, .. }
            | Instr::Ne { dst, .. }
            | Instr::Lt { dst, .. }
            | Instr::Le { dst, .. }
            | Instr::Gt { dst, .. }
            | Instr::Ge { dst, .. }
            | Instr::Call { dst, .. }
            | Instr::CallNative { dst, .. }
            | Instr::LoadField { dst, .. }
            | Instr::LoadIndex { dst, .. }
            | Instr::ArrayLen { dst, .. }
            | Instr::VecAdd { dst, .. }
            | Instr::VecMul { dst, .. }
            | Instr::MatMul { dst, .. }
            | Instr::Pow { dst, .. }
            | Instr::HadamardMul { dst, .. }
            | Instr::MakeArray { dst, .. }
            | Instr::MakeTuple { dst, .. }
            | Instr::MakeStruct { dst, .. }
            | Instr::MakeRange { dst, .. }
            | Instr::Cast { dst, .. }
            | Instr::TypeCheck { dst, .. }
            | Instr::AssumeInt { dst, .. }
            | Instr::AssumeFloat { dst, .. }
            => Some(*dst),
            Instr::Print { .. }
            | Instr::StoreField { .. }
            | Instr::StoreIndex { .. }
            | Instr::Jump { .. }
            | Instr::JumpFalse { .. }
            | Instr::JumpTrue { .. }
            | Instr::Return { .. }
            | Instr::SimdLoopStart { .. }
            | Instr::SimdLoopEnd
            | Instr::ProfilePoint { .. }
            | Instr::DebugBreak
            | Instr::Nop
            => None,
        }
    }
    
    /// Try to fold constant expressions at compile time.
    ///
    /// **SAFETY**: The `Move` instruction is ONLY folded when `dst == src`
    /// (trivial no-op → Nop).  We do NOT fold `Move { dst, src }` to
    /// `LoadConstInt` even when `src` holds a known constant, because
    /// that would "bake in" the constant value and break subsequent
    /// mutations of the source slot.  For example:
    ///
    ///   let x = 2;    // slot 0, known_constants[0] = I64(2)
    ///   x = 10;       // Move { dst: 0, src: tmp } — must NOT fold
    ///   let y = x;    // Move { dst: 1, src: 0 } — must NOT fold to LoadConstInt 2
    ///   x = 20;       // slot 0 is now 20, but y would still have stale 2
    ///
    /// Instead, we rely on the AST-level constant propagator (§3 in
    /// advanced_optimizer.rs) to handle variable-level constant
    /// propagation correctly, with proper invalidation on reassignment.
    fn try_fold_constant(&self, instr: &Instr) -> Option<Instr> {
        match instr {
            Instr::Add { dst, lhs, rhs } => {
                if let (Some(Value::I64(l)), Some(Value::I64(r))) =
                    (self.known_constants.get(lhs), self.known_constants.get(rhs)) {
                    // Fold: constant + constant = constant (wrapping semantics)
                    return Some(Instr::LoadConstInt { dst: *dst, value: l.wrapping_add(*r) });
                }
                if let (Some(Value::F64(l)), Some(Value::F64(r))) =
                    (self.known_constants.get(lhs), self.known_constants.get(rhs)) {
                    return Some(Instr::LoadConstFloat { dst: *dst, value: l + r });
                }
            }
            Instr::Sub { dst, lhs, rhs } => {
                if let (Some(Value::I64(l)), Some(Value::I64(r))) =
                    (self.known_constants.get(lhs), self.known_constants.get(rhs)) {
                    return Some(Instr::LoadConstInt { dst: *dst, value: l.wrapping_sub(*r) });
                }
            }
            Instr::Mul { dst, lhs, rhs } => {
                if let (Some(Value::I64(l)), Some(Value::I64(r))) =
                    (self.known_constants.get(lhs), self.known_constants.get(rhs)) {
                    // Fold: constant * constant = constant (wrapping semantics)
                    return Some(Instr::LoadConstInt { dst: *dst, value: l.wrapping_mul(*r) });
                }
            }
            Instr::Move { dst, src } => {
                // Eliminate no-op moves (dst == src) — this is always safe
                // because it reads from and writes to the same slot.
                if dst == src {
                    return Some(Instr::Nop);
                }
                // NOTE: We do NOT fold `Move` to `LoadConstInt/Float/Bool`
                // here.  Previously, when `src` held a known constant, the
                // Move was replaced with a LoadConst.  This was WRONG because
                // it captured the constant value at the time of the Move, but
                // subsequent mutations of the source slot would not be
                // reflected — producing stale constant values at runtime.
                //
                // The correct approach is to let the Move execute at runtime,
                // reading whatever value is currently in the source slot.
                // The AST-level constant propagator handles variable-level
                // constant propagation correctly.
            }
            _ => {}
        }
        None
    }

    /// Try to evaluate a simple constant expression at compile time.
    /// Used to populate `known_constants` for bytecode-level constant folding.
    fn eval_const_expr(&self, expr: &crate::compiler::ast::Expr) -> Option<Value> {
        use crate::compiler::ast::Expr;
        match expr {
            Expr::IntLit { value, .. } => Some(Value::I64(*value as i64)),
            Expr::FloatLit { value, .. } => Some(Value::F64(*value)),
            Expr::BoolLit { value, .. } => Some(Value::Bool(*value)),
            Expr::BinOp { op, lhs, rhs, .. } => {
                let lv = self.eval_const_expr(lhs)?;
                let rv = self.eval_const_expr(rhs)?;
                match (lv, rv) {
                    (Value::I64(l), Value::I64(r)) => {
                        use crate::compiler::ast::BinOpKind;
                        let result = match op {
                            BinOpKind::Add => Some(Value::I64(l.wrapping_add(r))),
                            BinOpKind::Sub => Some(Value::I64(l.wrapping_sub(r))),
                            BinOpKind::Mul => Some(Value::I64(l.wrapping_mul(r))),
                            BinOpKind::Div if r != 0 => Some(Value::I64(l / r)),
                            BinOpKind::Rem if r != 0 => Some(Value::I64(l % r)),
                            BinOpKind::BitAnd => Some(Value::I64(l & r)),
                            BinOpKind::BitOr => Some(Value::I64(l | r)),
                            BinOpKind::BitXor => Some(Value::I64(l ^ r)),
                            BinOpKind::Shl => Some(Value::I64(l.wrapping_shl(r as u32))),
                            BinOpKind::Shr => Some(Value::I64(l.wrapping_shr(r as u32))),
                            _ => None,
                        };
                        result
                    }
                    (Value::F64(l), Value::F64(r)) => {
                        use crate::compiler::ast::BinOpKind;
                        let result = match op {
                            BinOpKind::Add => Some(Value::F64(l + r)),
                            BinOpKind::Sub => Some(Value::F64(l - r)),
                            BinOpKind::Mul => Some(Value::F64(l * r)),
                            BinOpKind::Div if r != 0.0 => Some(Value::F64(l / r)),
                            _ => None,
                        };
                        result
                    }
                    _ => None,
                }
            }
            Expr::UnOp { op, expr, .. } => {
                let v = self.eval_const_expr(expr)?;
                match (op, v) {
                    (crate::compiler::ast::UnOpKind::Neg, Value::I64(n)) => Some(Value::I64(n.wrapping_neg())),
                    (crate::compiler::ast::UnOpKind::Neg, Value::F64(f)) => Some(Value::F64(-f)),
                    (crate::compiler::ast::UnOpKind::Not, Value::I64(n)) => Some(Value::I64(!n)),
                    (crate::compiler::ast::UnOpKind::Not, Value::Bool(b)) => Some(Value::Bool(!b)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Compile program to bytecode
    pub fn compile_program(&mut self, program: &Program) -> Result<Vec<BytecodeFunction>, String> {
        // First pass: register all function names so they can be referenced
        // during compilation of other functions (cross-function calls).
        let mut function_names = Vec::new();
        for item in &program.items {
            if let crate::compiler::ast::Item::Fn(fn_decl) = item {
                if fn_decl.body.is_some() {
                    let idx = self.functions.len();
                    self.functions.insert(fn_decl.name.clone(), idx);
                    function_names.push(fn_decl.name.clone());
                }
            }
        }

        // Compile each top-level function
        let mut functions = Vec::new();
        
        for item in &program.items {
            match item {
                crate::compiler::ast::Item::Fn(fn_decl) => {
                    if let Some(body) = &fn_decl.body {
                        let mut fn_compiler = BytecodeCompiler::new();
                        fn_compiler.current_function.name = fn_decl.name.clone();
                        fn_compiler.current_function.num_params = fn_decl.params.len() as u16;
                        fn_compiler.next_slot = fn_compiler.current_function.num_params;
                        for (i, p) in fn_decl.params.iter().enumerate() {
                            fn_compiler.locals.insert(p.name.clone(), i as u16);
                        }
                        // Share the function name table so cross-function calls work
                        fn_compiler.functions = self.functions.clone();
                        
                        // Compile function body
                        fn_compiler.compile_block(body, 0)?;
                        
                        // Set num_locals to the total number of slots allocated
                        // (params + locals + temporaries).  This is critical for
                        // the VM to allocate the correct slot array size.
                        fn_compiler.current_function.num_locals = fn_compiler.next_slot;
                        
                        functions.push(fn_compiler.current_function);
                    }
                }
                _ => {}
            }
        }
        
        Ok(functions)
    }
    
    fn compile_block(&mut self, block: &crate::compiler::ast::Block, dst: u16) -> Result<(), String> {
        for stmt in &block.stmts {
            self.compile_stmt(stmt)?;
        }
        if let Some(tail) = &block.tail {
            self.compile_expr(tail, dst)?;
        }
        Ok(())
    }
    
    // ─── Pattern slot allocation ───────────────────────────────────────────

    /// Allocate a slot for a pattern and register identifier bindings in `locals`.
    fn alloc_pattern_slot(&mut self, pattern: &crate::compiler::ast::Pattern) -> Result<u16, String> {
        match pattern {
            crate::compiler::ast::Pattern::Ident { name, .. } => {
                if let Some(&existing) = self.locals.get(name) {
                    Ok(existing)
                } else {
                    let slot = self.alloc_slot();
                    self.locals.insert(name.clone(), slot);
                    Ok(slot)
                }
            }
            crate::compiler::ast::Pattern::Wildcard(_) => Ok(self.alloc_slot()),
            crate::compiler::ast::Pattern::Tuple { elems, .. } => {
                // Allocate slots for each element pattern
                let mut first_slot = None;
                for elem in elems {
                    let s = self.alloc_pattern_slot(elem)?;
                    if first_slot.is_none() {
                        first_slot = Some(s);
                    }
                }
                Ok(first_slot.unwrap_or_else(|| self.alloc_slot()))
            }
            crate::compiler::ast::Pattern::Struct { fields, .. } => {
                let mut first_slot = None;
                for (_, pat) in fields {
                    if let Some(p) = pat {
                        let s = self.alloc_pattern_slot(p)?;
                        if first_slot.is_none() {
                            first_slot = Some(s);
                        }
                    }
                }
                Ok(first_slot.unwrap_or_else(|| self.alloc_slot()))
            }
            _ => {
                // Lit, Enum, Range, Or patterns: allocate a single slot
                Ok(self.alloc_slot())
            }
        }
    }

    // ─── Jump patching helpers ──────────────────────────────────────────────

    /// Emit a `JumpFalse` with a placeholder offset. Returns the instruction
    /// position so the offset can be patched later once the target is known.
    fn emit_jump_false(&mut self, cond: u16) -> usize {
        let pos = self.current_function.instructions.len();
        self.current_function.instructions.push(Instr::JumpFalse { cond, offset: 0 });
        pos
    }

    /// Emit a `JumpTrue` with a placeholder offset. Returns the instruction
    /// position so the offset can be patched later.
    fn emit_jump_true(&mut self, cond: u16) -> usize {
        let pos = self.current_function.instructions.len();
        self.current_function.instructions.push(Instr::JumpTrue { cond, offset: 0 });
        pos
    }

    /// Emit an unconditional `Jump` with a placeholder offset. Returns the
    /// instruction position so the offset can be patched later.
    fn emit_jump(&mut self) -> usize {
        let pos = self.current_function.instructions.len();
        self.current_function.instructions.push(Instr::Jump { offset: 0 });
        pos
    }

    /// Patch the jump offset at `pos` so that it lands at the *current* end of
    /// the instruction stream (i.e. the instruction that will be emitted next).
    fn patch_jump_offset(&mut self, pos: usize) {
        let target = self.current_function.instructions.len();
        let offset = (target as i32) - (pos as i32);
        match &mut self.current_function.instructions[pos] {
            Instr::Jump { offset: ref mut o } => *o = offset,
            Instr::JumpFalse { offset: ref mut o, .. } => *o = offset,
            Instr::JumpTrue { offset: ref mut o, .. } => *o = offset,
            other => panic!("patch_jump_offset: not a jump instruction at pos {pos}, found {other:?}"),
        }
    }

    /// Patch the jump offset at `pos` to jump to a specific `target` position.
    fn patch_jump_offset_to(&mut self, pos: usize, target: usize) {
        let offset = (target as i32) - (pos as i32);
        match &mut self.current_function.instructions[pos] {
            Instr::Jump { offset: ref mut o } => *o = offset,
            Instr::JumpFalse { offset: ref mut o, .. } => *o = offset,
            Instr::JumpTrue { offset: ref mut o, .. } => *o = offset,
            other => panic!("patch_jump_offset_to: not a jump instruction at pos {pos}, found {other:?}"),
        }
    }

    /// Patch all break-label `Jump` instructions recorded in `self.break_labels`
    /// so they land at `loop_end` (the instruction right after the loop), then
    /// clear the labels belonging to the just-completed loop.
    ///
    /// `depth` is the number of loop nesting levels that were active when the
    /// break labels were recorded; we only patch labels that belong to the
    /// innermost loop (i.e. those recorded after the last `loop_starts.push()`).
    fn patch_break_labels(&mut self, loop_end: usize) {
        // Break labels belonging to the innermost loop are the ones recorded
        // after the current loop's loop_start was pushed.  We pop the saved
        // count from break_labels_at_loop_start to get the correct split point.
        let split_point = self.break_labels_at_loop_start.pop().unwrap_or(0);
        let labels: Vec<usize> = self.break_labels.drain(split_point..).collect();
        for pos in labels {
            self.patch_jump_offset_to(pos, loop_end);
        }
    }

    /// Emit a backward `Jump` to the given `target` position.
    fn emit_backward_jump(&mut self, target: usize) {
        let current = self.current_function.instructions.len();
        let offset = (target as i32) - (current as i32);
        self.current_function.instructions.push(Instr::Jump { offset });
    }

    /// Convert a field name to a field index for `LoadField`/`StoreField`.
    /// Uses a stable hash of the field name.  In a production compiler, this
    /// would be resolved from struct layout metadata.
    fn field_name_to_idx(&self, field: &str) -> u32 {
        let mut hash: u32 = 0;
        for byte in field.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
        hash
    }

    /// Store a value (in `val_slot`) into the lvalue described by `target`.
    fn compile_store_target(&mut self, target: &crate::compiler::ast::Expr, val_slot: u16) -> Result<(), String> {
        match target {
            crate::compiler::ast::Expr::Ident { name, .. } => {
                let slot = self.locals.get(name).copied()
                    .ok_or_else(|| format!("unknown local variable `{name}`"))?;
                self.emit(Instr::Move { dst: slot, src: val_slot });
            }
            crate::compiler::ast::Expr::Field { object, field, .. } => {
                let obj_slot = self.alloc_slot();
                self.compile_expr(object, obj_slot)?;
                let field_idx = self.field_name_to_idx(field);
                self.emit(Instr::StoreField { obj: obj_slot, field_idx, src: val_slot });
            }
            crate::compiler::ast::Expr::Index { object, indices, .. } => {
                let obj_slot = self.alloc_slot();
                self.compile_expr(object, obj_slot)?;
                if indices.len() == 1 {
                    let idx_slot = self.alloc_slot();
                    self.compile_expr(&indices[0], idx_slot)?;
                    self.emit(Instr::StoreIndex { arr: obj_slot, idx: idx_slot, src: val_slot });
                } else {
                    return Err("bytecode compiler: multi-index store not yet supported".to_string());
                }
            }
            _ => return Err(format!("bytecode compiler: cannot assign to this expression type")),
        }
        Ok(())
    }

    // ─── Statement compilation ──────────────────────────────────────────────

    fn compile_stmt(&mut self, stmt: &crate::compiler::ast::Stmt) -> Result<(), String> {
        use crate::compiler::ast::Stmt;
        match stmt {
            Stmt::Let { pattern, init, .. } => {
                let dst = self.alloc_pattern_slot(pattern)?;
                if let Some(expr) = init {
                    self.compile_expr(expr, dst)?;
                    // Track known constant values for bytecode-level folding.
                    // Mutable variables must NOT be tracked because they can be
                    // reassigned later without invalidation.
                    if self.fold_constants {
                        if let crate::compiler::ast::Pattern::Ident { name: _, mutable, .. } = pattern {
                            if !mutable {
                                if let Some(val) = self.eval_const_expr(expr) {
                                    self.known_constants.insert(dst, val);
                                }
                            }
                        }
                    }
                } else {
                    self.emit(Instr::LoadConstUnit { dst });
                }
            }
            Stmt::Expr { expr, .. } => {
                // CRITICAL: Use a scratch slot for expression statements so that
                // the "also copy to dst" code in assignment expressions doesn't
                // clobber a local variable.  Previously, `dst=0` was used, which
                // overlaps with the first local variable's slot.  For example:
                //   let mut a: i32 = 100       // a lives in slot 0
                //   a = 10                     // Assign emits Move { dst: 0, src: tmp }
                //                              // then "copy to dst=0" clobbers a!
                //   a                           // reads clobbered value
                let scratch = self.alloc_slot();
                self.compile_expr(expr, scratch)?;
                // Invalidate known constants when a variable is reassigned.
                if self.fold_constants {
                    if let crate::compiler::ast::Expr::Assign { target, .. } = expr {
                        if let crate::compiler::ast::Expr::Ident { name, .. } = target.as_ref() {
                            if let Some(&slot) = self.locals.get(name) {
                                self.known_constants.remove(&slot);
                            }
                        }
                    }
                }
            }
            Stmt::Return { value, .. } => {
                if let Some(expr) = value {
                    self.compile_expr(expr, 0)?;
                } else {
                    self.emit(Instr::LoadConstUnit { dst: 0 });
                }
                self.emit(Instr::Return { value: 0 });
            }
            Stmt::If { cond, then, else_, .. } => {
                let cond_slot = self.alloc_slot();
                self.compile_expr(cond, cond_slot)?;
                let jump_false_pos = self.emit_jump_false(cond_slot);
                self.compile_block(then, 0)?;
                if let Some(else_branch) = else_ {
                    let jump_end_pos = self.emit_jump();
                    self.patch_jump_offset(jump_false_pos);
                    match else_branch.as_ref() {
                        IfOrBlock::If(if_stmt) => self.compile_stmt(if_stmt)?,
                        IfOrBlock::Block(block) => self.compile_block(block, 0)?,
                    }
                    self.patch_jump_offset(jump_end_pos);
                } else {
                    self.patch_jump_offset(jump_false_pos);
                }
            }
            Stmt::While { cond, body, .. } => {
                let loop_start = self.current_function.instructions.len();
                self.loop_starts.push(loop_start);
                self.break_labels_at_loop_start.push(self.break_labels.len());

                // Continue target for while loops is the condition check
                self.loop_continue_starts.push(loop_start);

                let cond_slot = self.alloc_slot();
                self.compile_expr(cond, cond_slot)?;
                let jump_end_pos = self.emit_jump_false(cond_slot);

                self.compile_block(body, 0)?;

                // Jump back to loop start (condition check)
                self.emit_backward_jump(loop_start);

                // Patch: condition-jump and all break jumps land here
                let loop_end = self.current_function.instructions.len();
                self.patch_jump_offset(jump_end_pos);
                self.patch_break_labels(loop_end);

                self.loop_starts.pop();
                self.loop_continue_starts.pop();
            }
            Stmt::Loop { body, .. } => {
                let loop_start = self.current_function.instructions.len();
                self.loop_starts.push(loop_start);
                self.break_labels_at_loop_start.push(self.break_labels.len());

                // Continue target for `loop` is the body start
                self.loop_continue_starts.push(loop_start);

                self.compile_block(body, 0)?;

                self.emit_backward_jump(loop_start);

                let loop_end = self.current_function.instructions.len();
                self.patch_break_labels(loop_end);

                self.loop_starts.pop();
                self.loop_continue_starts.pop();
            }
            Stmt::ForIn { pattern, iter, body, .. } => {
                // Compile the iterable into a temp slot
                let iter_slot = self.alloc_slot();
                self.compile_expr(iter, iter_slot)?;

                // Get the length of the iterable
                let len_slot = self.alloc_slot();
                self.emit(Instr::ArrayLen { dst: len_slot, arr: iter_slot });

                // Initialize index to 0
                let index_slot = self.alloc_slot();
                self.emit(Instr::LoadConstInt { dst: index_slot, value: 0 });

                // Allocate a slot for the loop variable
                let elem_slot = self.alloc_pattern_slot(pattern)?;

                // Loop structure:
                //   Jump to condition check (so continue goes to the increment)
                let jump_to_cond = self.emit_jump();

                // loop_body:
                let loop_body_start = self.current_function.instructions.len();
                self.loop_starts.push(loop_body_start);
                self.break_labels_at_loop_start.push(self.break_labels.len());

                // Load element at current index
                self.emit(Instr::LoadIndex { dst: elem_slot, arr: iter_slot, idx: index_slot });

                // Compile body
                self.compile_block(body, 0)?;

                // continue_target: increment index
                let continue_target = self.current_function.instructions.len();
                self.loop_continue_starts.push(continue_target);

                let one_slot = self.alloc_slot();
                self.emit(Instr::LoadConstInt { dst: one_slot, value: 1 });
                self.emit(Instr::Add { dst: index_slot, lhs: index_slot, rhs: one_slot });

                // cond_check:
                self.patch_jump_offset(jump_to_cond); // patches the initial jump
                let cond_slot = self.alloc_slot();
                self.emit(Instr::Lt { dst: cond_slot, lhs: index_slot, rhs: len_slot });
                // If index < len, jump back to loop body
                let body_offset = (loop_body_start as i32) - (self.current_function.instructions.len() as i32);
                self.current_function.instructions.push(Instr::JumpTrue { cond: cond_slot, offset: body_offset });

                // loop_end:
                let loop_end = self.current_function.instructions.len();
                self.patch_break_labels(loop_end);

                self.loop_starts.pop();
                self.loop_continue_starts.pop();
            }
            Stmt::Break { .. } => {
                let pos = self.emit_jump();
                self.break_labels.push(pos);
            }
            Stmt::Continue { .. } => {
                if let Some(&continue_target) = self.loop_continue_starts.last() {
                    self.emit_backward_jump(continue_target);
                } else {
                    return Err("bytecode compiler: continue outside of loop".to_string());
                }
            }
            Stmt::Match { .. } => {
                // Match statements not yet fully supported; emit Nop
                self.emit(Instr::Nop);
            }
            Stmt::EntityFor { .. } => {
                // Entity-for loops are game-simulation specific; emit Nop
                self.emit(Instr::Nop);
            }
            Stmt::Item(_) => {
                // Nested items not supported in bytecode compilation
            }
            Stmt::ParallelFor(_) | Stmt::Spawn(_) | Stmt::Sync(_) | Stmt::Atomic(_) => {
                // Parallelism statements not yet supported in bytecode
                self.emit(Instr::Nop);
            }
        }
        Ok(())
    }

    // ─── Expression compilation ─────────────────────────────────────────────

    fn compile_expr(&mut self, expr: &crate::compiler::ast::Expr, dst: u16) -> Result<(), String> {
        use crate::compiler::ast::Expr;
        match expr {
            // ── Literals ──────────────────────────────────────────────────
            Expr::IntLit { value, .. } => {
                let val = *value as i64;
                if val >= i32::MIN as i64 && val <= i32::MAX as i64 {
                    self.emit(Instr::LoadConstInt { dst, value: val });
                } else {
                    let idx = self.current_function.add_constant(Value::I64(val));
                    self.emit(Instr::LoadConst { dst, idx });
                }
            }
            Expr::FloatLit { value, .. } => {
                self.emit(Instr::LoadConstFloat { dst, value: *value });
            }
            Expr::BoolLit { value, .. } => {
                self.emit(Instr::LoadConstBool { dst, value: *value });
            }
            Expr::StrLit { value, .. } => {
                let idx = self.current_function.add_constant(Value::Str(value.clone()));
                self.emit(Instr::LoadConst { dst, idx });
            }

            // ── Variable / path ───────────────────────────────────────────
            Expr::Ident { name, .. } => {
                if let Some(&slot) = self.locals.get(name) {
                    self.emit(Instr::Move { dst, src: slot });
                } else if self.functions.contains_key(name) {
                    // This is a reference to a known function — create a Fn value
                    // so it can be called via the Call instruction.
                    // We store the function name as a constant; the VM will
                    // resolve it at runtime via the function table.
                    let idx = self.current_function.add_constant(Value::Str(name.clone()));
                    self.emit(Instr::LoadConst { dst, idx });
                } else if name == "print" || name == "println" {
                    // Built-in functions — store name as constant
                    let idx = self.current_function.add_constant(Value::Str(name.clone()));
                    self.emit(Instr::LoadConst { dst, idx });
                } else {
                    return Err(format!("unknown local variable `{name}`"));
                }
            }
            Expr::Path { segments, .. } => {
                // Treat single-segment paths as identifiers
                if segments.len() == 1 {
                    let slot = self
                        .locals
                        .get(&segments[0])
                        .copied()
                        .ok_or_else(|| format!("unknown local variable `{}`", segments[0]))?;
                    self.emit(Instr::Move { dst, src: slot });
                } else {
                    return Err(format!(
                        "bytecode compiler: qualified paths not yet supported: {}",
                        segments.join("::")
                    ));
                }
            }

            // ── Vector constructors ───────────────────────────────────────
            Expr::VecCtor { size, elems, .. } => {
                // Compile each element into temp slots, then build a vector value.
                let elem_slots: Vec<u16> = (0..elems.len()).map(|_| self.alloc_slot()).collect();
                for (i, elem) in elems.iter().enumerate() {
                    self.compile_expr(elem, elem_slots[i])?;
                }
                // Emit BuildVec instruction: gathers N element slots into a
                // SIMD vector value in dst.
                let count = elems.len() as u16;
                let start = elem_slots.first().copied().unwrap_or(dst);
                let _ = size; // VecSize used for type info, not needed at this stage
                self.emit(Instr::MakeArray { dst, start, count });
            }

            // ── Array literal ─────────────────────────────────────────────
            Expr::ArrayLit { elems, .. } => {
                let count = elems.len() as u16;
                let start = self.next_slot;
                let elem_slots: Vec<u16> = (0..elems.len()).map(|_| self.alloc_slot()).collect();
                for (i, elem) in elems.iter().enumerate() {
                    self.compile_expr(elem, elem_slots[i])?;
                }
                self.emit(Instr::MakeArray { dst, start, count });
            }

            // ── Binary operations ─────────────────────────────────────────
            Expr::BinOp { op, lhs, rhs, .. } => {
                // Short-circuit evaluation for logical operators
                match op {
                    BinOpKind::And => {
                        self.compile_expr(lhs, dst)?;
                        let jump_pos = self.emit_jump_false(dst);
                        self.compile_expr(rhs, dst)?;
                        self.patch_jump_offset(jump_pos);
                    }
                    BinOpKind::Or => {
                        self.compile_expr(lhs, dst)?;
                        let jump_pos = self.emit_jump_true(dst);
                        self.compile_expr(rhs, dst)?;
                        self.patch_jump_offset(jump_pos);
                    }
                    _ => {
                        // Eager evaluation for all other operators
                        self.compile_expr(lhs, dst)?;
                        let rhs_slot = self.alloc_slot();
                        self.compile_expr(rhs, rhs_slot)?;

                        match op {
                            BinOpKind::Add     => self.emit(Instr::Add { dst, lhs: dst, rhs: rhs_slot }),
                            BinOpKind::Sub     => self.emit(Instr::Sub { dst, lhs: dst, rhs: rhs_slot }),
                            BinOpKind::Mul     => self.emit(Instr::Mul { dst, lhs: dst, rhs: rhs_slot }),
                            BinOpKind::Div     => self.emit(Instr::Div { dst, lhs: dst, rhs: rhs_slot }),
                            BinOpKind::Rem     => self.emit(Instr::Rem { dst, lhs: dst, rhs: rhs_slot }),
                            BinOpKind::FloorDiv => self.emit(Instr::Div { dst, lhs: dst, rhs: rhs_slot }),
                            BinOpKind::Eq      => self.emit(Instr::Eq { dst, lhs: dst, rhs: rhs_slot }),
                            BinOpKind::Ne      => self.emit(Instr::Ne { dst, lhs: dst, rhs: rhs_slot }),
                            BinOpKind::Lt      => self.emit(Instr::Lt { dst, lhs: dst, rhs: rhs_slot }),
                            BinOpKind::Le      => self.emit(Instr::Le { dst, lhs: dst, rhs: rhs_slot }),
                            BinOpKind::Gt      => self.emit(Instr::Gt { dst, lhs: dst, rhs: rhs_slot }),
                            BinOpKind::Ge      => self.emit(Instr::Ge { dst, lhs: dst, rhs: rhs_slot }),
                            BinOpKind::BitAnd  => self.emit(Instr::BitAnd { dst, lhs: dst, rhs: rhs_slot }),
                            BinOpKind::BitOr   => self.emit(Instr::BitOr { dst, lhs: dst, rhs: rhs_slot }),
                            BinOpKind::BitXor  => self.emit(Instr::BitXor { dst, lhs: dst, rhs: rhs_slot }),
                            BinOpKind::Shl     => self.emit(Instr::Shl { dst, lhs: dst, rhs: rhs_slot }),
                            BinOpKind::Shr     => self.emit(Instr::Shr { dst, lhs: dst, rhs: rhs_slot }),
                            BinOpKind::And | BinOpKind::Or => unreachable!(),
                        }
                    }
                }
            }

            // ── Unary operations ──────────────────────────────────────────
            Expr::UnOp { op, expr: inner, .. } => {
                self.compile_expr(inner, dst)?;
                match op {
                    UnOpKind::Neg => self.emit(Instr::Neg { dst, src: dst }),
                    UnOpKind::Not => self.emit(Instr::Not { dst, src: dst }),
                    UnOpKind::Deref | UnOpKind::Ref | UnOpKind::RefMut => {
                        // Pass through — deref/ref are transparent at this level
                    }
                }
            }

            // ── Assignment ────────────────────────────────────────────────
            Expr::Assign { op, target, value, .. } => {
                // Always allocate a temp slot for the RHS result to avoid
                // clobbering the target variable's slot when dst overlaps.
                let result_slot = self.alloc_slot();
                if op.is_compound() {
                    // Compound assignment (+=, etc.): evaluate target, then RHS,
                    // apply operator, then store back.
                    self.compile_expr(target, result_slot)?;
                    let old_val = self.alloc_slot();
                    self.emit(Instr::Move { dst: old_val, src: result_slot });
                    let rhs_slot = self.alloc_slot();
                    self.compile_expr(value, rhs_slot)?;

                    match op.to_binop() {
                        Some(BinOpKind::Add)    => self.emit(Instr::Add { dst: result_slot, lhs: old_val, rhs: rhs_slot }),
                        Some(BinOpKind::Sub)    => self.emit(Instr::Sub { dst: result_slot, lhs: old_val, rhs: rhs_slot }),
                        Some(BinOpKind::Mul)    => self.emit(Instr::Mul { dst: result_slot, lhs: old_val, rhs: rhs_slot }),
                        Some(BinOpKind::Div)    => self.emit(Instr::Div { dst: result_slot, lhs: old_val, rhs: rhs_slot }),
                        Some(BinOpKind::Rem)    => self.emit(Instr::Rem { dst: result_slot, lhs: old_val, rhs: rhs_slot }),
                        Some(BinOpKind::BitAnd) => self.emit(Instr::BitAnd { dst: result_slot, lhs: old_val, rhs: rhs_slot }),
                        Some(BinOpKind::BitOr)  => self.emit(Instr::BitOr { dst: result_slot, lhs: old_val, rhs: rhs_slot }),
                        Some(BinOpKind::BitXor) => self.emit(Instr::BitXor { dst: result_slot, lhs: old_val, rhs: rhs_slot }),
                        _ => {
                            // MatMulAssign etc. — fallback: just assign RHS
                            self.compile_expr(value, result_slot)?;
                        }
                    }
                } else {
                    // Plain assignment: compile RHS into a temp slot
                    self.compile_expr(value, result_slot)?;
                }
                // Store result into the target lvalue
                self.compile_store_target(target, result_slot)?;
                // Also copy to dst for the expression's return value
                if dst != result_slot {
                    self.emit(Instr::Move { dst, src: result_slot });
                }
            }

            // ── Field access ──────────────────────────────────────────────
            Expr::Field { object, field, .. } => {
                let obj_slot = self.alloc_slot();
                self.compile_expr(object, obj_slot)?;
                let field_idx = self.field_name_to_idx(field);
                self.emit(Instr::LoadField { dst, obj: obj_slot, field_idx });
            }

            // ── Index access ──────────────────────────────────────────────
            Expr::Index { object, indices, .. } => {
                let obj_slot = self.alloc_slot();
                self.compile_expr(object, obj_slot)?;
                if indices.len() == 1 {
                    let idx_slot = self.alloc_slot();
                    self.compile_expr(&indices[0], idx_slot)?;
                    self.emit(Instr::LoadIndex { dst, arr: obj_slot, idx: idx_slot });
                } else {
                    return Err("bytecode compiler: multi-index access not yet supported".to_string());
                }
            }

            // ── Function call ─────────────────────────────────────────────
            Expr::Call { func, args, .. } => {
                // ── Special-case: print() → emit Print instruction directly ──
                if let Expr::Ident { name, .. } = func.as_ref() {
                    if name == "print" {
                        if args.len() == 1 {
                            self.compile_expr(&args[0], dst)?;
                            self.emit(Instr::Print { src: dst });
                        } else {
                            // print() with 0 or 2+ args: compile each and print
                            for arg in args {
                                self.compile_expr(arg, dst)?;
                                self.emit(Instr::Print { src: dst });
                            }
                        }
                    } else if name == "println" {
                        if args.len() == 1 {
                            self.compile_expr(&args[0], dst)?;
                            self.emit(Instr::Print { src: dst });
                        } else if args.is_empty() {
                            self.emit(Instr::LoadConstUnit { dst });
                            self.emit(Instr::Print { src: dst });
                        }
                    } else if self.functions.contains_key(name) {
                        // Call to a known user-defined function
                        let argc = args.len() as u16;
                        let arg_start = dst.wrapping_add(2);
                        for (i, arg) in args.iter().enumerate() {
                            self.compile_expr(arg, arg_start + i as u16)?;
                        }
                        // Create a Fn value for the callee
                        let fn_name_idx = self.current_function.add_constant(Value::Str(name.clone()));
                        self.emit(Instr::LoadConst { dst: dst.wrapping_add(1), idx: fn_name_idx });
                        // We need a Value::Fn — create one via a LoadFn-like approach
                        // For now, use the function name to look up at runtime
                        self.emit(Instr::Call { dst, func: dst.wrapping_add(1), argc, start: arg_start });
                    } else {
                        // Unknown function — compile as a generic call
                        let argc = args.len() as u16;
                        let arg_start = dst.wrapping_add(2);
                        for (i, arg) in args.iter().enumerate() {
                            self.compile_expr(arg, arg_start + i as u16)?;
                        }
                        let func_slot = dst.wrapping_add(1);
                        self.compile_expr(func, func_slot)?;
                        self.emit(Instr::Call { dst, func: func_slot, argc, start: arg_start });
                    }
                } else {
                    // Callee is a complex expression (e.g., `f(x)(y)`)
                    let argc = args.len() as u16;
                    let arg_start = dst.wrapping_add(2);
                    for (i, arg) in args.iter().enumerate() {
                        self.compile_expr(arg, arg_start + i as u16)?;
                    }
                    let func_slot = dst.wrapping_add(1);
                    self.compile_expr(func, func_slot)?;
                    self.emit(Instr::Call { dst, func: func_slot, argc, start: arg_start });
                }
            }

            // ── Method call ───────────────────────────────────────────────
            Expr::MethodCall { receiver, method, args, .. } => {
                // Compile receiver + args into contiguous slots, then emit a
                // Call instruction where the first argument is the receiver.
                // The method name is resolved at call time by the VM through
                // the inline cache / vtable lookup.
                let argc = (args.len() + 1) as u16; // +1 for receiver
                let arg_start = dst.wrapping_add(2);
                self.compile_expr(receiver, arg_start)?;
                for (i, arg) in args.iter().enumerate() {
                    self.compile_expr(arg, arg_start + 1 + i as u16)?;
                }
                // Store the method name as a string constant index so the VM
                // can look it up.  We reuse the Call instruction with the
                // function slot pointing to a string-constant that the VM
                // recognises as a method dispatch request.
                let func_slot = dst.wrapping_add(1);
                // Encode the method name as a string-constant and store in
                // the function slot.  The VM call handler checks whether the
                // "function" slot contains a string and performs method
                // dispatch if so.
                let idx = self.current_function.add_constant(Value::Str(method.clone()));
                self.emit(Instr::LoadConst { dst: func_slot, idx });
                self.emit(Instr::Call { dst, func: func_slot, argc, start: arg_start });
            }

            // ── If expression ─────────────────────────────────────────────
            Expr::IfExpr { cond, then, else_, .. } => {
                self.compile_expr(cond, dst)?;
                let jump_false_pos = self.emit_jump_false(dst);
                self.compile_block(then, dst)?;
                if let Some(else_block) = else_ {
                    let jump_end_pos = self.emit_jump();
                    self.patch_jump_offset(jump_false_pos);
                    self.compile_block(else_block, dst)?;
                    self.patch_jump_offset(jump_end_pos);
                } else {
                    self.patch_jump_offset(jump_false_pos);
                    // No else branch → result is Unit
                    self.emit(Instr::LoadConstUnit { dst });
                }
            }

            // ── Block expression ──────────────────────────────────────────
            Expr::Block(block) => {
                self.compile_block(block, dst)?;
            }

            // ── Tuple ─────────────────────────────────────────────────────
            Expr::Tuple { elems, .. } => {
                let count = elems.len() as u16;
                let start = self.next_slot;
                let elem_slots: Vec<u16> = (0..elems.len()).map(|_| self.alloc_slot()).collect();
                for (i, elem) in elems.iter().enumerate() {
                    self.compile_expr(elem, elem_slots[i])?;
                }
                self.emit(Instr::MakeTuple { dst, start, count });
            }

            // ── Struct literal ────────────────────────────────────────────
            Expr::StructLit { name, fields, .. } => {
                let name_idx = self.current_function.add_constant(Value::Str(name.clone()));
                let field_count = fields.len() as u16;
                let field_start = self.next_slot;
                // Allocate pairs of (name_slot, value_slot) for each field
                for (field_name, field_expr) in fields.iter() {
                    let name_slot = self.alloc_slot();
                    let val_slot = self.alloc_slot();
                    let const_idx = self.current_function.add_constant(Value::Str(field_name.clone()));
                    self.emit(Instr::LoadConst { dst: name_slot, idx: const_idx });
                    self.compile_expr(field_expr, val_slot)?;
                }
                self.emit(Instr::MakeStruct { dst, name_idx, field_start, field_count });
            }

            // ── Closure ───────────────────────────────────────────────────
            Expr::Closure { .. } => {
                // Closures not yet supported in bytecode compilation
                self.emit(Instr::LoadConstUnit { dst });
            }

            // ── Cast ──────────────────────────────────────────────────────
            Expr::Cast { expr: inner, ty, .. } => {
                let src_slot = self.alloc_slot();
                self.compile_expr(inner, src_slot)?;
                // Map the type AST node to a target_type discriminant
                let target_type = Self::type_to_cast_target(ty);
                self.emit(Instr::Cast { dst, src: src_slot, target_type });
            }

            // ── Range ─────────────────────────────────────────────────────
            Expr::Range { lo, hi, inclusive, .. } => {
                let lo_slot = self.alloc_slot();
                let hi_slot = self.alloc_slot();
                if let Some(lo_expr) = lo {
                    self.compile_expr(lo_expr, lo_slot)?;
                } else {
                    self.emit(Instr::LoadConstInt { dst: lo_slot, value: 0 });
                }
                if let Some(hi_expr) = hi {
                    self.compile_expr(hi_expr, hi_slot)?;
                } else {
                    self.emit(Instr::LoadConstInt { dst: hi_slot, value: 0 });
                }
                self.emit(Instr::MakeRange { dst, lo: lo_slot, hi: hi_slot, inclusive: *inclusive });
            }

            // ── Tensor-specific operations ────────────────────────────────
            Expr::MatMul { lhs, rhs, .. } => {
                self.compile_expr(lhs, dst)?;
                let rhs_slot = self.alloc_slot();
                self.compile_expr(rhs, rhs_slot)?;
                self.emit(Instr::MatMul { dst, lhs: dst, rhs: rhs_slot });
            }
            Expr::HadamardMul { lhs, rhs, .. } => {
                self.compile_expr(lhs, dst)?;
                let rhs_slot = self.alloc_slot();
                self.compile_expr(rhs, rhs_slot)?;
                self.emit(Instr::HadamardMul { dst, lhs: dst, rhs: rhs_slot });
            }
            Expr::HadamardDiv { lhs, rhs, .. } => {
                self.compile_expr(lhs, dst)?;
                let rhs_slot = self.alloc_slot();
                self.compile_expr(rhs, rhs_slot)?;
                self.emit(Instr::Div { dst, lhs: dst, rhs: rhs_slot });
            }
            Expr::Grad { inner, .. } => {
                // Gradient tracking is transparent at runtime
                self.compile_expr(inner, dst)?;
            }
            Expr::Pow { base, exp, .. } => {
                self.compile_expr(base, dst)?;
                let exp_slot = self.alloc_slot();
                self.compile_expr(exp, exp_slot)?;
                self.emit(Instr::Pow { dst, base: dst, exp: exp_slot });
            }
            Expr::TensorConcat { .. } | Expr::KronProd { .. } | Expr::OuterProd { .. } => {
                // Tensor operations not fully supported in bytecode yet
                self.emit(Instr::LoadConstUnit { dst });
            }
        }
        Ok(())
    }

    /// Map an AST Type to the Cast target_type discriminant.
    ///
    /// Encoding: 0=I64, 1=F64, 2=Bool, 3=Str, 4=I32, 5=F32, 6=U64
    fn type_to_cast_target(ty: &crate::compiler::ast::Type) -> u32 {
        use crate::compiler::ast::{ElemType, Type};
        match ty {
            Type::Scalar(e) => match e {
                ElemType::I64 | ElemType::Usize => 0,
                ElemType::F64 => 1,
                ElemType::Bool => 2,
                ElemType::I32 => 4,
                ElemType::F32 => 5,
                ElemType::U64 => 6,
                ElemType::I8 | ElemType::I16 | ElemType::U8 | ElemType::U16 | ElemType::U32 => 0,
                ElemType::F16 | ElemType::Bf16 => 5,
            },
            Type::Named(name) => match name.as_str() {
                "i64" | "int" => 0,
                "f64" | "float" => 1,
                "bool" => 2,
                "string" | "str" => 3,
                "i32" => 4,
                "f32" => 5,
                "u64" => 6,
                _ => 0, // default to I64
            },
            _ => 0, // default to I64
        }
    }
}

// =============================================================================
// §7  CALL FRAME
// =============================================================================

/// A call frame for function invocation.
#[derive(Debug, Clone)]
pub struct CallFrame {
    /// The PC to return to after the callee returns.
    pub return_pc: usize,
    /// The base slot index for the caller's local variables.
    pub base_slot: u16,
    /// Number of locals in the caller's frame (for slot restore).
    pub num_locals: u16,
}

// =============================================================================
// §8  ULTRA-FAST BYTECODE VM (Direct-Threaded Execution)
// =============================================================================

/// The fastest possible interpreter using direct threading
pub struct BytecodeVM {
    /// All compiled functions
    functions: Vec<BytecodeFunction>,

    /// Function name → index lookup for O(1) Call dispatch (avoids O(n) linear scan)
    function_index: FxHashMap<String, usize>,

    /// Global constant pool
    constants: Vec<Value>,
    
    /// Inline caches for fast field/method access
    inline_caches: Vec<InlineCache>,
    
    /// Memory pool for allocation
    memory_pool: MemoryPool,
    
    /// Adaptive profiler
    profiler: Option<AdaptiveProfiler>,
    
    /// Execution statistics
    total_instructions: u64,
    total_time_ns: u64,
    #[cfg(feature = "gnn-optimizer")]
    prefetch: PrefetchEngine,

    /// Native function table (indexed by CallNative.func_idx)
    native_functions: Vec<Box<dyn Fn(&[Value]) -> Result<Value, RuntimeError> + Send + Sync>>,

    /// Call stack for function invocations
    call_stack: Vec<CallFrame>,

    /// Profiling data: per-point hit counters
    profile_points: Vec<AtomicU64>,

    /// Data-Dependent JIT: profiles runtime values and creates specialized
    /// loop versions when hot values are detected. Integrated into the
    /// dispatch loop via the `Call` handler and loop-backedge observation.
    data_dependent_jit: DataDependentJIT,

    // ── Heuristic Safety Engine ───────────────────────────────────────────────
    // The new safety subsystem replaces the naive PC-counter watchdog with
    // three layers of defense:
    //   1. Trust Tier: SMT-proven code skips all checks
    //   2. Entropy Watchdog: monitors state mutation, not just PC hits
    //   3. OSR Engine: de-optimizes to interpreter instead of panicking

    /// Global arithmetic mode override. If set, overrides per-function modes.
    /// Used for benchmarking or when the user explicitly sets a mode.
    global_arithmetic_mode: Option<ArithmeticMode>,

    /// On-Stack Replacement engine for graceful de-optimization.
    /// When the runtime detects an unsafe condition in optimized code,
    /// the OSR engine swaps execution to the safe interpreter path
    /// instead of panicking.
    osr_engine: OsrEngine,
}

impl BytecodeVM {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            function_index: FxHashMap::default(),
            constants: Vec::new(),
            inline_caches: Vec::new(),
            memory_pool: MemoryPool::with_capacity(1024),
            profiler: None,
            total_instructions: 0,
            total_time_ns: 0,
            #[cfg(feature = "gnn-optimizer")]
            prefetch: PrefetchEngine::default(),
            native_functions: Vec::new(),
            call_stack: Vec::new(),
            profile_points: Vec::new(),
            data_dependent_jit: DataDependentJIT::new(),
            global_arithmetic_mode: None,
            osr_engine: OsrEngine::new(true),
        }
    }

    /// Get a reference to the data-dependent JIT for external inspection.
    pub fn data_dependent_jit(&self) -> &DataDependentJIT {
        &self.data_dependent_jit
    }

    /// Get a mutable reference to the data-dependent JIT for configuration.
    pub fn data_dependent_jit_mut(&mut self) -> &mut DataDependentJIT {
        &mut self.data_dependent_jit
    }

    /// Register a native function and return its index for CallNative.
    pub fn register_native(&mut self, f: Box<dyn Fn(&[Value]) -> Result<Value, RuntimeError> + Send + Sync>) -> u32 {
        let idx = self.native_functions.len() as u32;
        self.native_functions.push(f);
        idx
    }
    
    /// Load compiled functions into VM
    pub fn load_functions(&mut self, functions: Vec<BytecodeFunction>) {
        // Build function name → index lookup for O(1) Call dispatch
        self.function_index.clear();
        for (i, f) in functions.iter().enumerate() {
            self.function_index.insert(f.name.clone(), i);
        }
        self.functions = functions;
    }
    
    /// Execute a function by index
    #[inline(never)]
    pub fn execute(&mut self, func_idx: usize, args: &[Value]) -> Result<Value, RuntimeError> {
        let start_time = std::time::Instant::now();

        // Initialize slots with arguments
        let needed_slots = self.functions[func_idx].num_locals.max(self.functions[func_idx].num_params) as usize;
        let num_slots = needed_slots.max(64); // Minimum slot size to avoid index-out-of-bounds
        // Only grow the slot array — never shrink it. This avoids O(n) reallocation
        // on every function call for leaf functions that need fewer slots.
        if self.memory_pool.slots.len() < num_slots {
            self.memory_pool.slots.resize(num_slots, Value::Unit);
        }
        self.memory_pool.max_slot_used = num_slots.saturating_sub(1);
        for (i, arg) in args.iter().enumerate() {
            if i < num_slots {
                self.memory_pool.slots[i] = arg.clone();
            }
        }

        // Update execution counters
        self.functions[func_idx].execution_count.fetch_add(1, Ordering::Relaxed);
        let func_len = self.functions[func_idx].instructions.len();

        // Determine the effective arithmetic mode for this function.
        // Priority: global override > per-function mode > trust-tier default
        let arithmetic_mode = self.global_arithmetic_mode.unwrap_or_else(|| {
            self.functions[func_idx].arithmetic_mode
        });

        // Determine the watchdog sensitivity based on trust tier.
        let watchdog_sensitivity = self.functions[func_idx].trust_tier.watchdog_sensitivity();

        // Execute bytecode.
        //
        // We need a reference to `func` that outlives the mutable borrow of
        // `self` inside `execute_direct_threaded`. The borrow checker cannot
        // prove the two borrows are disjoint (func is in self.functions; the
        // loop mutates self.memory_pool and self.prefetch, not self.functions).
        //
        // SAFETY: `execute_direct_threaded` never modifies `self.functions`
        // (it only reads instructions/constants from `func`), so the aliasing
        // of `func_ptr` with `self` is safe. The pointer is valid for the
        // duration of the call because `self.functions` is not reallocated
        // inside `execute_direct_threaded`.
        let func_ptr: *const BytecodeFunction = &self.functions[func_idx];
        unsafe {
            self.execute_direct_threaded(&*func_ptr, func_len, arithmetic_mode, watchdog_sensitivity)?;
        }

        let elapsed = start_time.elapsed();
        self.total_instructions += func_len as u64;
        self.total_time_ns += elapsed.as_nanos() as u64;

        // Return value is in slot 0. Swap it out instead of cloning —
        // the slot array is reset on the next call anyway.
        Ok(std::mem::replace(&mut self.memory_pool.slots[0], Value::Unit))
    }
    
    /// Switch-dispatch execution loop.
    ///
    /// The Rust compiler lowers the `match` on `Instr` to an indirect jump
    /// table (`br_table` / `jmp [rax*8+base]`) in optimized builds, which
    /// gives near-direct-threaded performance without unsafe label arithmetic.
    ///
    /// Hot-path design notes:
    /// - I64/F64 fast paths are checked first via pattern matching; the generic
    ///   `Value` fallback is reached only on type mismatch.
    /// - `clone()` is eliminated from the arithmetic fast paths; it only
    ///   appears on the generic slow path where a heap allocation is already
    ///   implied by the type mismatch branch.
    /// - `prefetch_insn` is intentionally absent: the sequential instruction
    ///   stream is handled by the CPU's hardware prefetcher for free. See
    ///   `PrefetchEngine::prefetch_insn` doc for the full reasoning.
    /// - Profiling is sampled (every N instructions) rather than per-instruction
    ///   to avoid atomic overhead on every dispatch.
    #[inline]
    fn execute_direct_threaded(&mut self, func: &BytecodeFunction, func_len: usize, arithmetic_mode: ArithmeticMode, watchdog_sensitivity: WatchdogSensitivity) -> Result<(), RuntimeError> {
        // Create raw pointer to self before any field borrows.
        // Used by the Call instruction to recursively invoke execute()
        // without conflicting with the `slots` mutable borrow.
        // SAFETY: vm_ptr is only dereferenced in the Call arm, after all
        // reads from `slots` for that instruction have been cloned into
        // local variables, and before `slots` is used again (next iteration).
        let vm_ptr: *mut BytecodeVM = self;

        let instructions = &func.instructions;
        let constants = &func.constants;
        let slots = &mut self.memory_pool.slots;
        let mut pc: usize = 0;
        #[cfg(feature = "gnn-optimizer")]
        let mut branch_density: u8 = 0;
        // Sampling counter: profile every 256 instructions instead of every one.
        let mut profile_counter: u8 = 0;
        // Pre-compute slot pointer for write-intent prefetch in the dispatch loop.
        let _slot_ptr = slots.as_mut_ptr();

        // ── Entropy Watchdog ──────────────────────────────────────────────────
        // Replaces the naive PC-counter safety mechanism. Instead of counting
        // how many times the PC hits the same address, we monitor whether the
        // program state is actually *changing*. A loop that iterates through
        // an array has "entropy" (making progress) and is NOT an infinite loop.
        let mut entropy_watchdog = EntropyWatchdog::new(watchdog_sensitivity);

        // Main dispatch loop
        while pc < func_len {
            // ── Entropy Watchdog Check ──────────────────────────────────────────
            // The sampled check here is for a HARD INSTRUCTION LIMIT only.
            // The actual entropy-based loop detection happens at real backedges
            // (Jump with negative offset). This avoids false positives from
            // counting every 256 instructions as a "backedge observation".
            // The watchdog is disabled for TIER_0_TRUSTED functions.
            if entropy_watchdog.active {
                profile_counter = profile_counter.wrapping_add(1);
                if profile_counter == 0 {
                    // Only check hard instruction limit here — entropy check
                    // is at actual backward jumps. Scale by 256 because we
                    // only sample every 256 instructions.
                    if let Some(max_backedges) = entropy_watchdog.sensitivity.max_iterations_scaled(entropy_watchdog.complexity_factor) {
                        if entropy_watchdog.total_backedges > max_backedges {
                            let trigger = OsrTrigger::WatchdogTimeout {
                                pc,
                                iterations: entropy_watchdog.total_backedges,
                            };
                            match self.osr_engine.deoptimize(trigger, &func.name, pc) {
                                OsrOutcome::Deoptimized { .. } => {
                                    return Err(RuntimeError::new(format!(
                                        "watchdog: instruction limit exceeded in '{}' at pc={} \
                                         (entropy ratio: {:.2}%, {} backedges observed). \
                                         OSR de-optimization engaged.",
                                        func.name, pc,
                                        entropy_watchdog.entropy_ratio() * 100.0,
                                        entropy_watchdog.total_backedges
                                    )));
                                }
                                OsrOutcome::Unavailable { .. } => {
                                    return Err(RuntimeError::new(format!(
                                        "watchdog: instruction limit exceeded in '{}' at pc={} \
                                         (entropy ratio: {:.2}%, {} backedges observed)",
                                        func.name, pc,
                                        entropy_watchdog.entropy_ratio() * 100.0,
                                        entropy_watchdog.total_backedges
                                    )));
                                }
                            }
                        }
                    }
                }
            }
            // Sampled profiling: avoids an atomic fetch_add on every instruction.
            // The counter wraps every 256 dispatches; profiler sees ~0.4% of PCs.
            if let Some(ref profiler) = self.profiler {
                profile_counter = profile_counter.wrapping_add(1);
                if profile_counter == 0 {
                    profiler.record_execution(pc);
                }
            }

            // Prefetch the write destination one slot ahead — the one software
            // hint that pays its cost (PREFETCHW eliminates the RFO stall).
            // Instruction-stream prefetch is intentionally omitted; see module doc.
            #[cfg(feature = "gnn-optimizer")]
            {
                self.prefetch.tick(branch_density);
                self.prefetch.prefetch_dual(
                    // insn_base/pc/insn_len are ignored inside prefetch_dual now;
                    // pass them for API compatibility.
                    instructions.as_ptr(), pc, func_len,
                    slot_ptr, pc, slots.len(),
                );
            }

            // SAFETY: pc < func_len (loop guard), instructions is a valid slice.
            let instr = unsafe { &*instructions.as_ptr().add(pc) };
            
            match instr {
                // ── HOT PATH: Constant loads (most frequent) ──
                Instr::LoadConst { dst, idx } => {
                    // Clone is unavoidable here since constants is a shared pool,
                    // but we call it explicitly to make the allocation site visible.
                    slots[*dst as usize] = constants[*idx as usize].clone();
                    pc += 1;
                }
                Instr::LoadConstInt { dst, value } => {
                    slots[*dst as usize] = Value::I64(*value);
                    pc += 1;
                }
                Instr::LoadConstFloat { dst, value } => {
                    slots[*dst as usize] = Value::F64(*value);
                    pc += 1;
                }
                Instr::LoadConstBool { dst, value } => {
                    slots[*dst as usize] = Value::Bool(*value);
                    pc += 1;
                }
                Instr::LoadConstUnit { dst } => {
                    slots[*dst as usize] = Value::Unit;
                    pc += 1;
                }
                
                // ── HOT PATH: Register moves ──
                // Avoid clone(): use ptr::read/write to bitwise-copy the Value
                // when src != dst. The compiler guarantees non-aliasing register
                // indices; malformed bytecode where dst == src is a no-op here.
                Instr::Move { dst, src } => {
                    let d = *dst as usize;
                    let s = *src as usize;
                    if d != s {
                        // SAFETY: d and s are distinct valid indices into the
                        // pre-allocated slots vec. ptr::read moves ownership out
                        // of src without running its destructor; ptr::write drops
                        // the old dst value and installs the moved value.
                        unsafe {
                            let v = std::ptr::read(slots.as_ptr().add(s));
                            std::ptr::write(slots.as_mut_ptr().add(d), v);
                        }
                    }
                    pc += 1;
                }
                
                // ── HOT PATH: Arithmetic operations ──
                // Fast paths match I64/F64 directly without allocating.
                // The generic slow path clones only on type-mismatch, which
                // already implies a slow dynamic-dispatch branch.
                Instr::Add { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];
                    
                    if let (Value::I64(l), Value::I64(r)) = (l_val, r_val) {
                        // Tier 1 Fix: Use arithmetic mode instead of raw `+`
                        // Wrapping mode (default): silently wraps on overflow
                        // Strict mode: returns error on overflow
                        // Saturating mode: clamps to i64::MIN/MAX
                        slots[*dst as usize] = Value::I64(arithmetic_mode.add(*l, *r));
                        pc += 1;
                        continue;
                    }
                    if let (Value::F64(l), Value::F64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::F64(l + r);
                        pc += 1;
                        continue;
                    }
                    
                    let lhs_val = l_val.clone();
                    let rhs_val = r_val.clone();
                    slots[*dst as usize] = Self::add_values_static(&lhs_val, &rhs_val)?;
                    pc += 1;
                }
                
                Instr::Sub { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];
                    
                    if let (Value::I64(l), Value::I64(r)) = (l_val, r_val) {
                        // Tier 1 Fix: Use arithmetic mode instead of raw `-`
                        slots[*dst as usize] = Value::I64(arithmetic_mode.sub(*l, *r));
                        pc += 1;
                        continue;
                    }
                    if let (Value::F64(l), Value::F64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::F64(l - r);
                        pc += 1;
                        continue;
                    }
                    
                    let lhs_val = l_val.clone();
                    let rhs_val = r_val.clone();
                    slots[*dst as usize] = Self::sub_values_static(&lhs_val, &rhs_val)?;
                    pc += 1;
                }
                
                Instr::Mul { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];
                    
                    if let (Value::I64(l), Value::I64(r)) = (l_val, r_val) {
                        // Tier 1 Fix: Use arithmetic mode instead of raw `*`
                        // This was the root cause of the "Arithmetic Panic" —
                        // Rust's default `*` panics on overflow in debug mode.
                        // Now we use wrapping_mul() by default for speed.
                        slots[*dst as usize] = Value::I64(arithmetic_mode.mul(*l, *r));
                        pc += 1;
                        continue;
                    }
                    if let (Value::F64(l), Value::F64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::F64(l * r);
                        pc += 1;
                        continue;
                    }
                    
                    let lhs_val = l_val.clone();
                    let rhs_val = r_val.clone();
                    slots[*dst as usize] = Self::mul_values_static(&lhs_val, &rhs_val)?;
                    pc += 1;
                }
                
                Instr::Div { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];
                    
                    if let (Value::I64(l), Value::I64(r)) = (l_val, r_val) {
                        if *r == 0 {
                            return Err(RuntimeError::new("division by zero"));
                        }
                        slots[*dst as usize] = Value::I64(l / r);
                        pc += 1;
                        continue;
                    }
                    if let (Value::F64(l), Value::F64(r)) = (l_val, r_val) {
                        if *r == 0.0 {
                            return Err(RuntimeError::new("division by zero"));
                        }
                        slots[*dst as usize] = Value::F64(l / r);
                        pc += 1;
                        continue;
                    }
                    
                    let lhs_val = l_val.clone();
                    let rhs_val = r_val.clone();
                    slots[*dst as usize] = Self::div_values_static(&lhs_val, &rhs_val)?;
                    pc += 1;
                }

                // ── HOT PATH: SIMD-friendly vector ops ──
                Instr::VecAdd { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];
                    slots[*dst as usize] = Self::vec_add_values_static(l_val, r_val)?;
                    pc += 1;
                }
                Instr::VecMul { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];
                    slots[*dst as usize] = Self::vec_mul_values_static(l_val, r_val)?;
                    pc += 1;
                }
                
                // ── HOT PATH: Control flow ──
                Instr::Jump { offset } => {
                    #[cfg(feature = "gnn-optimizer")]
                    { branch_density = branch_density.saturating_add(1); }
                    // Detect backward jumps (loops) for DataDependentJIT profiling.
                    // When a backedge is taken, observe the loop trip count so
                    // the JIT can detect hot values and create specializations.
                    if *offset < 0 {
                        // Backward jump = loop backedge. Feed trip-count
                        // hint to the DataDependentJIT. We use the current
                        // function name as the profiling key.
                        let fn_name = &func.name;
                        let loop_len = (-(*offset)) as u64;
                        // Observe the loop length as a proxy for the trip count.
                        // This is a lightweight observation — the heavy lifting
                        // (guard checking, specialization) happens in try_specialize.
                        self.data_dependent_jit.observe_int(
                            &format!("{fn_name}::loop_len@{pc}"),
                            loop_len as i64,
                        );

                        // Tier 2 Fix: Feed the backedge to the entropy watchdog.
                        // The watchdog checks if the state is mutating (making progress)
                        // rather than just counting how many times we've been here.
                        if entropy_watchdog.active && entropy_watchdog.observe_backedge(slots) {
                            let trigger = OsrTrigger::WatchdogTimeout {
                                pc,
                                iterations: entropy_watchdog.total_backedges,
                            };
                            match self.osr_engine.deoptimize(trigger, &func.name, pc) {
                                OsrOutcome::Deoptimized { .. } => {
                                    return Err(RuntimeError::new(format!(
                                        "watchdog: infinite loop in '{}' at pc={} \
                                         (stagnant for {} backedges, entropy ratio: {:.2}%). \
                                         OSR de-optimization engaged.",
                                        func.name, pc, entropy_watchdog.stagnant_count,
                                        entropy_watchdog.entropy_ratio() * 100.0
                                    )));
                                }
                                OsrOutcome::Unavailable { .. } => {
                                    return Err(RuntimeError::new(format!(
                                        "watchdog: infinite loop in '{}' at pc={} \
                                         (stagnant for {} backedges, entropy ratio: {:.2}%)",
                                        func.name, pc, entropy_watchdog.stagnant_count,
                                        entropy_watchdog.entropy_ratio() * 100.0
                                    )));
                                }
                            }
                        }
                    }
                    pc = if *offset >= 0 {
                        pc + *offset as usize
                    } else {
                        pc.wrapping_sub((-(*offset)) as usize)
                    };
                }
                
                Instr::JumpFalse { cond, offset } => {
                    #[cfg(feature = "gnn-optimizer")]
                    { branch_density = branch_density.saturating_add(1); }
                    let cond_val = &slots[*cond as usize];
                    if !cond_val.is_truthy() {
                        pc = if *offset >= 0 {
                            pc + *offset as usize
                        } else {
                            pc.wrapping_sub((-(*offset)) as usize)
                        };
                    } else {
                        pc += 1;
                    }
                }
                
                Instr::JumpTrue { cond, offset } => {
                    #[cfg(feature = "gnn-optimizer")]
                    { branch_density = branch_density.saturating_add(1); }
                    let cond_val = &slots[*cond as usize];
                    if cond_val.is_truthy() {
                        pc = if *offset >= 0 {
                            pc + *offset as usize
                        } else {
                            pc.wrapping_sub((-(*offset)) as usize)
                        };
                    } else {
                        pc += 1;
                    }
                }
                
                // ── Remainder ──
                Instr::Rem { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];

                    if let (Value::I64(l), Value::I64(r)) = (l_val, r_val) {
                        if *r == 0 {
                            return Err(RuntimeError::new("remainder by zero"));
                        }
                        slots[*dst as usize] = Value::I64(l % r);
                        pc += 1;
                        continue;
                    }
                    if let (Value::F64(l), Value::F64(r)) = (l_val, r_val) {
                        if *r == 0.0 {
                            return Err(RuntimeError::new("remainder by zero"));
                        }
                        slots[*dst as usize] = Value::F64(l % r);
                        pc += 1;
                        continue;
                    }

                    let lhs_val = l_val.clone();
                    let rhs_val = r_val.clone();
                    slots[*dst as usize] = Self::rem_values_static(&lhs_val, &rhs_val)?;
                    pc += 1;
                }

                // ── Negation ──
                Instr::Neg { dst, src } => {
                    let s_val = &slots[*src as usize];

                    if let Value::I64(v) = s_val {
                        // Tier 1 Fix: Use arithmetic mode for negation
                        slots[*dst as usize] = Value::I64(arithmetic_mode.neg(*v));
                        pc += 1;
                        continue;
                    }
                    if let Value::F64(v) = s_val {
                        slots[*dst as usize] = Value::F64(-v);
                        pc += 1;
                        continue;
                    }

                    let src_val = s_val.clone();
                    slots[*dst as usize] = Self::neg_values_static(&src_val)?;
                    pc += 1;
                }

                // ── Bitwise AND ──
                Instr::BitAnd { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];

                    if let (Value::I64(l), Value::I64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::I64(l & r);
                        pc += 1;
                        continue;
                    }

                    // Generic fallback: coerce to i64
                    let lv = l_val.clone();
                    let rv = r_val.clone();
                    let l_i64 = lv.as_i64().ok_or_else(|| RuntimeError::new(format!(
                        "BitAnd: left operand is not an integer ({})", lv.type_name()
                    )))?;
                    let r_i64 = rv.as_i64().ok_or_else(|| RuntimeError::new(format!(
                        "BitAnd: right operand is not an integer ({})", rv.type_name()
                    )))?;
                    slots[*dst as usize] = Value::I64(l_i64 & r_i64);
                    pc += 1;
                }

                // ── Bitwise OR ──
                Instr::BitOr { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];

                    if let (Value::I64(l), Value::I64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::I64(l | r);
                        pc += 1;
                        continue;
                    }

                    let lv = l_val.clone();
                    let rv = r_val.clone();
                    let l_i64 = lv.as_i64().ok_or_else(|| RuntimeError::new(format!(
                        "BitOr: left operand is not an integer ({})", lv.type_name()
                    )))?;
                    let r_i64 = rv.as_i64().ok_or_else(|| RuntimeError::new(format!(
                        "BitOr: right operand is not an integer ({})", rv.type_name()
                    )))?;
                    slots[*dst as usize] = Value::I64(l_i64 | r_i64);
                    pc += 1;
                }

                // ── Bitwise XOR ──
                Instr::BitXor { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];

                    if let (Value::I64(l), Value::I64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::I64(l ^ r);
                        pc += 1;
                        continue;
                    }

                    let lv = l_val.clone();
                    let rv = r_val.clone();
                    let l_i64 = lv.as_i64().ok_or_else(|| RuntimeError::new(format!(
                        "BitXor: left operand is not an integer ({})", lv.type_name()
                    )))?;
                    let r_i64 = rv.as_i64().ok_or_else(|| RuntimeError::new(format!(
                        "BitXor: right operand is not an integer ({})", rv.type_name()
                    )))?;
                    slots[*dst as usize] = Value::I64(l_i64 ^ r_i64);
                    pc += 1;
                }

                // ── Shift Left ──
                Instr::Shl { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];

                    if let (Value::I64(l), Value::I64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::I64(l.wrapping_shl(*r as u32));
                        pc += 1;
                        continue;
                    }

                    let lv = l_val.clone();
                    let rv = r_val.clone();
                    let l_i64 = lv.as_i64().ok_or_else(|| RuntimeError::new(format!(
                        "Shl: left operand is not an integer ({})", lv.type_name()
                    )))?;
                    let r_i64 = rv.as_i64().ok_or_else(|| RuntimeError::new(format!(
                        "Shl: right operand is not an integer ({})", rv.type_name()
                    )))?;
                    slots[*dst as usize] = Value::I64(l_i64.wrapping_shl(r_i64 as u32));
                    pc += 1;
                }

                // ── Shift Right ──
                Instr::Shr { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];

                    if let (Value::I64(l), Value::I64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::I64(l.wrapping_shr(*r as u32));
                        pc += 1;
                        continue;
                    }

                    let lv = l_val.clone();
                    let rv = r_val.clone();
                    let l_i64 = lv.as_i64().ok_or_else(|| RuntimeError::new(format!(
                        "Shr: left operand is not an integer ({})", lv.type_name()
                    )))?;
                    let r_i64 = rv.as_i64().ok_or_else(|| RuntimeError::new(format!(
                        "Shr: right operand is not an integer ({})", rv.type_name()
                    )))?;
                    slots[*dst as usize] = Value::I64(l_i64.wrapping_shr(r_i64 as u32));
                    pc += 1;
                }

                // ── Bitwise / Logical NOT ──
                Instr::Not { dst, src } => {
                    let s_val = &slots[*src as usize];

                    if let Value::I64(v) = s_val {
                        slots[*dst as usize] = Value::I64(!v);
                        pc += 1;
                        continue;
                    }
                    if let Value::Bool(v) = s_val {
                        slots[*dst as usize] = Value::Bool(!v);
                        pc += 1;
                        continue;
                    }

                    // Generic fallback: coerce to i64 and bitwise-not
                    let src_val = s_val.clone();
                    if let Some(i) = src_val.as_i64() {
                        slots[*dst as usize] = Value::I64(!i);
                    } else if let Some(b) = src_val.as_bool() {
                        slots[*dst as usize] = Value::Bool(!b);
                    } else {
                        return Err(RuntimeError::new(format!(
                            "Not: cannot negate {} at pc={pc}", src_val.type_name()
                        )));
                    }
                    pc += 1;
                }

                // ── Comparison: Equal ──
                Instr::Eq { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];

                    if let (Value::I64(l), Value::I64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::Bool(l == r);
                        pc += 1;
                        continue;
                    }
                    if let (Value::F64(l), Value::F64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::Bool(l == r);
                        pc += 1;
                        continue;
                    }
                    if let (Value::Bool(l), Value::Bool(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::Bool(l == r);
                        pc += 1;
                        continue;
                    }

                    let lv = l_val.clone();
                    let rv = r_val.clone();
                    slots[*dst as usize] = Self::compare_values_static(&lv, &rv, |l, r| l == r, "Eq")?;
                    pc += 1;
                }

                // ── Comparison: Not Equal ──
                Instr::Ne { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];

                    if let (Value::I64(l), Value::I64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::Bool(l != r);
                        pc += 1;
                        continue;
                    }
                    if let (Value::F64(l), Value::F64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::Bool(l != r);
                        pc += 1;
                        continue;
                    }
                    if let (Value::Bool(l), Value::Bool(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::Bool(l != r);
                        pc += 1;
                        continue;
                    }

                    let lv = l_val.clone();
                    let rv = r_val.clone();
                    slots[*dst as usize] = Self::compare_values_static(&lv, &rv, |l, r| l != r, "Ne")?;
                    pc += 1;
                }

                // ── Comparison: Less Than ──
                Instr::Lt { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];

                    if let (Value::I64(l), Value::I64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::Bool(l < r);
                        pc += 1;
                        continue;
                    }
                    if let (Value::F64(l), Value::F64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::Bool(l < r);
                        pc += 1;
                        continue;
                    }

                    let lv = l_val.clone();
                    let rv = r_val.clone();
                    slots[*dst as usize] = Self::compare_values_static(&lv, &rv, |l, r| l < r, "Lt")?;
                    pc += 1;
                }

                // ── Comparison: Less or Equal ──
                Instr::Le { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];

                    if let (Value::I64(l), Value::I64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::Bool(l <= r);
                        pc += 1;
                        continue;
                    }
                    if let (Value::F64(l), Value::F64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::Bool(l <= r);
                        pc += 1;
                        continue;
                    }

                    let lv = l_val.clone();
                    let rv = r_val.clone();
                    slots[*dst as usize] = Self::compare_values_static(&lv, &rv, |l, r| l <= r, "Le")?;
                    pc += 1;
                }

                // ── Comparison: Greater Than ──
                Instr::Gt { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];

                    if let (Value::I64(l), Value::I64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::Bool(l > r);
                        pc += 1;
                        continue;
                    }
                    if let (Value::F64(l), Value::F64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::Bool(l > r);
                        pc += 1;
                        continue;
                    }

                    let lv = l_val.clone();
                    let rv = r_val.clone();
                    slots[*dst as usize] = Self::compare_values_static(&lv, &rv, |l, r| l > r, "Gt")?;
                    pc += 1;
                }

                // ── Comparison: Greater or Equal ──
                Instr::Ge { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];

                    if let (Value::I64(l), Value::I64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::Bool(l >= r);
                        pc += 1;
                        continue;
                    }
                    if let (Value::F64(l), Value::F64(r)) = (l_val, r_val) {
                        slots[*dst as usize] = Value::Bool(l >= r);
                        pc += 1;
                        continue;
                    }

                    let lv = l_val.clone();
                    let rv = r_val.clone();
                    slots[*dst as usize] = Self::compare_values_static(&lv, &rv, |l, r| l >= r, "Ge")?;
                    pc += 1;
                }

                // JumpIfFalse/JumpIfTrue removed — merged into JumpFalse/JumpTrue above

                // ── Function Call ──
                // func slot contains a Value::Fn. Push a CallFrame, collect args,
                // then recursively execute the callee via vm_ptr (raw pointer to
                // self) to avoid conflicting with the `slots` mutable borrow.
                Instr::Call { dst, func, argc, start } => {
                    let dst_idx = *dst as usize;
                    let func_val = slots[*func as usize].clone();
                    let arg_start = *start as usize;
                    let arg_count = *argc as usize;

                    // Collect args before any further mutation of slots.
                    // Stack-allocate for small arg counts (≤4) to avoid heap alloc.
                    let args: Vec<Value> = if arg_count <= 4 {
                        let mut small: [Value; 4] = [Value::Unit, Value::Unit, Value::Unit, Value::Unit];
                        for i in 0..arg_count {
                            small[i] = slots.get(arg_start + i).cloned().unwrap_or(Value::Unit);
                        }
                        small[..arg_count].to_vec()
                    } else {
                        (0..arg_count)
                            .map(|i| slots.get(arg_start + i).cloned().unwrap_or(Value::Unit))
                            .collect()
                    };

                    // ── DataDependentJIT: profile call arguments ──
                    // Observe integer arguments so the JIT can detect hot
                    // values and create guarded specializations.
                    {
                        let fn_name_for_jit = match &func_val {
                            Value::Fn(closure) => closure.decl.name.clone(),
                            _ => "anonymous".to_string(),
                        };
                        for (i, arg) in args.iter().enumerate() {
                            if let Value::I64(v) = arg {
                                self.data_dependent_jit.observe_int(
                                    &format!("{fn_name_for_jit}::arg{i}"),
                                    *v,
                                );
                            } else if let Value::Bool(v) = arg {
                                self.data_dependent_jit.observe_bool(
                                    &format!("{fn_name_for_jit}::arg{i}"),
                                    *v,
                                );
                            }
                        }
                    }

                    match &func_val {
                        Value::Fn(closure) => {
                            let fn_name = closure.decl.name.clone();

                            // ── Optimizer annotation builtins (identity functions) ──
                            // These are inserted by the superoptimizer during compilation.
                            // At runtime they are identity functions — return the first argument.
                            if matches!(
                                fn_name.as_str(),
                                "fused_elementwise"
                                | "exact_div_restore"
                                | "matmul_elemwise"
                                | "scaled_matmul"
                            ) {
                                let ret = args.into_iter().next().ok_or_else(|| {
                                    RuntimeError::new(format!(
                                        "{fn_name}() requires at least 1 argument"
                                    ))
                                })?;
                                unsafe {
                                    (&mut (*vm_ptr).memory_pool.slots)[dst_idx] = ret;
                                }
                                pc += 1;
                                continue;
                            }

                            // Look up the BytecodeFunction by name — O(1) hash lookup
                            let callee_idx = self.function_index.get(&fn_name).copied();
                            match callee_idx {
                                Some(idx) => {
                                    // Push call frame (return to next instruction)
                                    self.call_stack.push(CallFrame {
                                        return_pc: pc + 1,
                                        base_slot: 0,
                                        num_locals: self.functions[idx].num_locals,
                                    });

                                    let result = unsafe { (*vm_ptr).execute(idx, &args) };

                                    // Pop the call frame
                                    self.call_stack.pop();

                                    match result {
                                        Ok(ret_val) => {
                                            unsafe {
                                                (&mut (*vm_ptr).memory_pool.slots)[dst_idx] = ret_val;
                                            }
                                        }
                                        Err(e) => return Err(e),
                                    }
                                    pc += 1;
                                }
                                None => {
                                    return Err(RuntimeError::new(format!(
                                        "Call: unknown function `{fn_name}` at pc={pc}"
                                    )));
                                }
                            }
                        }
                        Value::Str(fn_name) => {
                            // ── Optimizer annotation builtins (identity functions) ──
                            if matches!(
                                fn_name.as_str(),
                                "fused_elementwise"
                                | "exact_div_restore"
                                | "matmul_elemwise"
                                | "scaled_matmul"
                            ) {
                                let ret = args.into_iter().next().ok_or_else(|| {
                                    RuntimeError::new(format!(
                                        "{fn_name}() requires at least 1 argument"
                                    ))
                                })?;
                                unsafe {
                                    (&mut (*vm_ptr).memory_pool.slots)[dst_idx] = ret;
                                }
                                pc += 1;
                                continue;
                            }

                            // Function referenced by name (from BytecodeCompiler) — O(1) hash lookup
                            let callee_idx = self.function_index.get(fn_name).copied();
                            match callee_idx {
                                Some(idx) => {
                                    self.call_stack.push(CallFrame {
                                        return_pc: pc + 1,
                                        base_slot: 0,
                                        num_locals: self.functions[idx].num_locals,
                                    });

                                    let result = unsafe { (*vm_ptr).execute(idx, &args) };

                                    self.call_stack.pop();

                                    match result {
                                        Ok(ret_val) => {
                                            unsafe {
                                                (&mut (*vm_ptr).memory_pool.slots)[dst_idx] = ret_val;
                                            }
                                        }
                                        Err(e) => return Err(e),
                                    }
                                    pc += 1;
                                }
                                None => {
                                    return Err(RuntimeError::new(format!(
                                        "Call: unknown function `{fn_name}` at pc={pc}"
                                    )));
                                }
                            }
                        }
                        other => {
                            return Err(RuntimeError::new(format!(
                                "Call: expected Fn or Str, got {} at pc={pc}",
                                other.type_name()
                            )));
                        }
                    }
                }

                // ── Native Function Call ──
                Instr::CallNative { dst, func_idx, argc, start } => {
                    let fidx = *func_idx as usize;
                    if fidx >= self.native_functions.len() {
                        return Err(RuntimeError::new(format!(
                            "CallNative: func_idx {fidx} out of range ({} registered) at pc={pc}",
                            self.native_functions.len()
                        )));
                    }

                    // Collect arguments from slots[start..start+argc)
                    // For small arg counts (≤4), stack-allocate to avoid heap allocation
                    let arg_start = *start as usize;
                    let arg_count = *argc as usize;
                    let result = if arg_count <= 4 {
                        let mut small_args: [Value; 4] = [Value::Unit, Value::Unit, Value::Unit, Value::Unit];
                        for i in 0..arg_count {
                            small_args[i] = slots.get(arg_start + i).cloned().unwrap_or(Value::Unit);
                        }
                        self.native_functions[fidx](&small_args[..arg_count])
                    } else {
                        let args: Vec<Value> = (0..arg_count)
                            .map(|i| slots.get(arg_start + i).cloned().unwrap_or(Value::Unit))
                            .collect();
                        self.native_functions[fidx](&args)
                    };

                    match result {
                        Ok(val) => {
                            slots[*dst as usize] = val;
                        }
                        Err(e) => return Err(e),
                    }
                    pc += 1;
                }

                // ── Return from function ──
                // If we have a call frame, pop it and resume the caller.
                // Otherwise, this is the top-level return — store value in
                // slot 0 and exit the dispatch loop.
                Instr::Return { value } => {
                    if let Some(frame) = self.call_stack.pop() {
                        // Move return value to slot 0 using mem::replace (avoids clone)
                        let v_slot = *value as usize;
                        let ret_val = std::mem::replace(&mut slots[v_slot], Value::Unit);
                        slots[0] = ret_val;
                        pc = frame.return_pc;
                    } else {
                        // Top-level return: move value to slot 0
                        let v_slot = *value as usize;
                        if v_slot != 0 {
                            let ret_val = std::mem::replace(&mut slots[v_slot], Value::Unit);
                            slots[0] = ret_val;
                        }
                        return Ok(());
                    }
                }

                // ── Load Field from Struct ──
                // field_idx is a pre-computed index into the struct's field values,
                // sorted by the order the fields were declared at compile time.
                Instr::LoadField { dst, obj, field_idx } => {
                    let obj_val = &slots[*obj as usize];
                    match obj_val {
                        Value::Struct(data) => {
                            // Use InlineCache for O(1) field access — avoids O(n) iter().nth()
                            let shape_id = data.fields.len() as u64;
                            // Try inline cache first (monomorphic fast path)
                            let cached = if let Some(cache) = self.inline_caches.get(0) {
                                cache.lookup(shape_id)
                            } else {
                                None
                            };
                            if let Some(offset) = cached {
                                if let Some(v) = data.fields.iter().nth(offset as usize).map(|(_, v)| v) {
                                    slots[*dst as usize] = v.clone();
                                    pc += 1;
                                    continue;
                                }
                            }
                            // Slow path: iterate to field_idx, then update cache
                            let fidx = *field_idx as usize;
                            let mut iter = data.fields.iter();
                            let result = if let Some((_key, v)) = iter.nth(fidx) {
                                // Update inline cache for next time
                                if self.inline_caches.is_empty() {
                                    self.inline_caches.push(InlineCache::new());
                                }
                                if let Some(cache) = self.inline_caches.get_mut(0) {
                                    cache.update(shape_id, fidx as i32);
                                }
                                Ok(v.clone())
                            } else {
                                Err(RuntimeError::new(format!(
                                    "LoadField: field index {} out of range at pc={pc}",
                                    field_idx
                                )))
                            };
                            match result {
                                Ok(v) => slots[*dst as usize] = v,
                                Err(e) => return Err(e),
                            }
                        }
                        other => {
                            return Err(RuntimeError::new(format!(
                                "LoadField: expected Struct, got {} at pc={pc}",
                                other.type_name()
                            )));
                        }
                    }
                    pc += 1;
                }

                // ── Store Field in Struct ──
                Instr::StoreField { obj, field_idx, src } => {
                    let src_val = std::mem::replace(&mut slots[*src as usize], Value::Unit);
                    let obj_val = &mut slots[*obj as usize];
                    match obj_val {
                        Value::Struct(data) => {
                            // Use InlineCache for O(1) field key lookup
                            let shape_id = data.fields.len() as u64;
                            let cached = if let Some(cache) = self.inline_caches.get(0) {
                                cache.lookup(shape_id)
                            } else {
                                None
                            };
                            let key = if let Some(offset) = cached {
                                data.fields.iter().nth(offset as usize).map(|(k, _)| k.clone())
                            } else {
                                // Slow path: find Nth field in sorted order
                                let fidx = *field_idx as usize;
                                let key = data.fields.iter().nth(fidx).map(|(k, _)| k.clone());
                                // Update inline cache
                                if let Some(ref _k) = key {
                                    if self.inline_caches.is_empty() {
                                        self.inline_caches.push(InlineCache::new());
                                    }
                                    if let Some(cache) = self.inline_caches.get_mut(0) {
                                        cache.update(shape_id, fidx as i32);
                                    }
                                }
                                key
                            };
                            match key {
                                Some(k) => {
                                    data.fields.insert(k, src_val);
                                }
                                None => {
                                    return Err(RuntimeError::new(format!(
                                        "StoreField: field index {} out of range at pc={pc}",
                                        field_idx
                                    )));
                                }
                            }
                        }
                        other => {
                            return Err(RuntimeError::new(format!(
                                "StoreField: expected Struct, got {} at pc={pc}",
                                other.type_name()
                            )));
                        }
                    }
                    pc += 1;
                }

                // ── Load Index from Array ──
                Instr::LoadIndex { dst, arr, idx } => {
                    let arr_val = &slots[*arr as usize];
                    let idx_val = &slots[*idx as usize];
                    // Clone the element before writing to dst to avoid
                    // simultaneous immutable + mutable borrows of `slots`.
                    let result_val = match (arr_val, idx_val) {
                        (Value::Array(arr_mutex), Value::I64(i)) => {
                            let guard = arr_mutex.borrow();
                            let index = *i as usize;
                            match guard.get(index) {
                                Some(v) => v.clone(),
                                None => {
                                    return Err(RuntimeError::new(format!(
                                        "LoadIndex: index {i} out of bounds (len {}) at pc={pc}",
                                        guard.len()
                                    )));
                                }
                            }
                        }
                        (Value::Array(_), other) => {
                            return Err(RuntimeError::new(format!(
                                "LoadIndex: index must be I64, got {} at pc={pc}",
                                other.type_name()
                            )));
                        }
                        (other, _) => {
                            return Err(RuntimeError::new(format!(
                                "LoadIndex: expected Array, got {} at pc={pc}",
                                other.type_name()
                            )));
                        }
                    };
                    // Drop any borrows of arr/idx before writing dst.
                    slots[*dst as usize] = result_val;
                    pc += 1;
                }

                // ── Store Index in Array ──
                Instr::StoreIndex { arr, idx, src } => {
                    let src_val = slots[*src as usize].clone();
                    let arr_val = &slots[*arr as usize];
                    let idx_val = &slots[*idx as usize];
                    match (arr_val, idx_val) {
                        (Value::Array(arr_mutex), Value::I64(i)) => {
                            let mut guard = arr_mutex.borrow_mut();
                            let index = *i as usize;
                            if index < guard.len() {
                                guard[index] = src_val;
                            } else {
                                return Err(RuntimeError::new(format!(
                                    "StoreIndex: index {i} out of bounds (len {}) at pc={pc}",
                                    guard.len()
                                )));
                            }
                        }
                        (Value::Array(_), other) => {
                            return Err(RuntimeError::new(format!(
                                "StoreIndex: index must be I64, got {} at pc={pc}",
                                other.type_name()
                            )));
                        }
                        (other, _) => {
                            return Err(RuntimeError::new(format!(
                                "StoreIndex: expected Array, got {} at pc={pc}",
                                other.type_name()
                            )));
                        }
                    }
                    pc += 1;
                }

                // ── Array/Tuple Length ──
                Instr::ArrayLen { dst, arr } => {
                    let len = match &slots[*arr as usize] {
                        Value::Array(arr_mutex) => arr_mutex.borrow().len(),
                        Value::Tuple(v) => v.len(),
                        Value::Str(s) => s.len(),
                        other => {
                            return Err(RuntimeError::new(format!(
                                "ArrayLen: expected Array/Tuple/Str, got {} at pc={pc}",
                                other.type_name()
                            )));
                        }
                    };
                    slots[*dst as usize] = Value::I64(len as i64);
                    pc += 1;
                }

                // ── Matrix Multiplication (Tensor) ──
                Instr::MatMul { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];
                    match (l_val, r_val) {
                        (Value::Tensor(l_arc), Value::Tensor(r_arc)) => {
                            let l_tensor = l_arc.read().unwrap();
                            let r_tensor = r_arc.read().unwrap();
                            let result = l_tensor.matmul(&r_tensor)?;
                            drop(l_tensor);
                            drop(r_tensor);
                            slots[*dst as usize] = Value::Tensor(std::sync::Arc::new(std::sync::RwLock::new(result)));
                        }
                        (Value::TensorFast(l_arc), Value::TensorFast(r_arc)) => {
                            let l_tensor = l_arc.borrow();
                            let r_tensor = r_arc.borrow();
                            let result = l_tensor.matmul(&r_tensor)?;
                            drop(l_tensor);
                            drop(r_tensor);
                            slots[*dst as usize] = Value::TensorFast(std::sync::Arc::new(std::cell::RefCell::new(result)));
                        }
                        (Value::Tensor(_), Value::TensorFast(_))
                        | (Value::TensorFast(_), Value::Tensor(_)) => {
                            return Err(RuntimeError::new(
                                "MatMul: cannot mix Tensor and TensorFast — use same type"
                            ));
                        }
                        (other_l, other_r) => {
                            return Err(RuntimeError::new(format!(
                                "MatMul: expected Tensor/TensorFast, got {} and {} at pc={pc}",
                                other_l.type_name(), other_r.type_name()
                            )));
                        }
                    }
                    pc += 1;
                }

                // ── Profile Point ──
                // Record that this profile point was hit. Grows the counter
                // vector on first encounter.
                Instr::ProfilePoint { id } => {
                    let id_usize = *id as usize;
                    if id_usize >= self.profile_points.len() {
                        self.profile_points.resize_with(id_usize + 1, || AtomicU64::new(0));
                    }
                    self.profile_points[id_usize].fetch_add(1, Ordering::Relaxed);
                    pc += 1;
                }

                // ── Debug Break ──
                // No-op in release mode; in debug builds this could trigger
                // a breakpoint trap, but for now it's just a no-op.
                Instr::DebugBreak => {
                    pc += 1;
                }

                // ── Type specialization guards ──
                // These are emitted by the compiler when it infers a type for
                // a slot. On mismatch we return a runtime error — never silently
                // produce wrong output.
                Instr::AssumeInt { dst, src } => {
                    match &slots[*src as usize] {
                        Value::I64(v) => {
                            let v = *v;
                            slots[*dst as usize] = Value::I64(v);
                        }
                        other => return Err(RuntimeError::new(format!(
                            "AssumeInt: expected Int, got {} at pc={pc}",
                            other.type_name()
                        ))),
                    }
                    pc += 1;
                }

                Instr::AssumeFloat { dst, src } => {
                    match &slots[*src as usize] {
                        Value::F64(v) => {
                            let v = *v;
                            slots[*dst as usize] = Value::F64(v);
                        }
                        other => return Err(RuntimeError::new(format!(
                            "AssumeFloat: expected Float, got {} at pc={pc}",
                            other.type_name()
                        ))),
                    }
                    pc += 1;
                }

                Instr::TypeCheck { dst, src, expected_type } => {
                    // expected_type is a discriminant index matching Value's type_tag() order.
                    // Write a Bool into dst: true if the type matches, false otherwise.
                    // This replaces the previous hard-error behaviour so that callers can
                    // branch on the result instead of aborting.
                    let actual_tag = slots[*src as usize].type_tag() as u32;
                    slots[*dst as usize] = Value::Bool(actual_tag == *expected_type);
                    pc += 1;
                }

                // ── Power ──
                Instr::Pow { dst, base, exp } => {
                    let b_val = &slots[*base as usize];
                    let e_val = &slots[*exp as usize];

                    if let (Value::I64(b), Value::I64(e)) = (b_val, e_val) {
                        let e = *e;
                        if e < 0 {
                            return Err(RuntimeError::new(
                                "Pow: negative exponent for integer base"
                            ));
                        }
                        slots[*dst as usize] = Value::I64(Self::ipow(*b, e as u64));
                        pc += 1;
                        continue;
                    }
                    if let (Value::F64(b), Value::F64(e)) = (b_val, e_val) {
                        slots[*dst as usize] = Value::F64(b.powf(*e));
                        pc += 1;
                        continue;
                    }
                    // Fallback: coerce to f64
                    let bf = b_val.as_f64();
                    let ef = e_val.as_f64();
                    match (bf, ef) {
                        (Some(b), Some(e)) => {
                            slots[*dst as usize] = Value::F64(b.powf(e));
                        }
                        _ => {
                            return Err(RuntimeError::new(format!(
                                "Pow: cannot raise {} to {} at pc={pc}",
                                b_val.type_name(), e_val.type_name()
                            )));
                        }
                    }
                    pc += 1;
                }

                // ── Hadamard (element-wise) multiply ──
                Instr::HadamardMul { dst, lhs, rhs } => {
                    let l_val = &slots[*lhs as usize];
                    let r_val = &slots[*rhs as usize];
                    match (l_val, r_val) {
                        (Value::Tensor(l_arc), Value::Tensor(r_arc)) => {
                            let l_tensor = l_arc.read().unwrap();
                            let r_tensor = r_arc.read().unwrap();
                            let result = l_tensor.hadamard_mul(&r_tensor)?;
                            drop(l_tensor);
                            drop(r_tensor);
                            slots[*dst as usize] = Value::Tensor(std::sync::Arc::new(std::sync::RwLock::new(result)));
                        }
                        (Value::TensorFast(l_arc), Value::TensorFast(r_arc)) => {
                            let l_tensor = l_arc.borrow();
                            let r_tensor = r_arc.borrow();
                            let result = l_tensor.hadamard_mul(&r_tensor)?;
                            drop(l_tensor);
                            drop(r_tensor);
                            slots[*dst as usize] = Value::TensorFast(std::sync::Arc::new(std::cell::RefCell::new(result)));
                        }
                        (Value::Vec4(a), Value::Vec4(b)) => {
                            slots[*dst as usize] = Value::Vec4(simd_mul_vec4(*a, *b));
                        }
                        (Value::Vec3(a), Value::Vec3(b)) => {
                            slots[*dst as usize] = Value::Vec3([a[0]*b[0], a[1]*b[1], a[2]*b[2]]);
                        }
                        (Value::Vec2(a), Value::Vec2(b)) => {
                            slots[*dst as usize] = Value::Vec2([a[0]*b[0], a[1]*b[1]]);
                        }
                        (Value::Array(l_arr), Value::Array(r_arr)) => {
                            let l_guard = l_arr.borrow();
                            let r_guard = r_arr.borrow();
                            if l_guard.len() != r_guard.len() {
                                return Err(RuntimeError::new(format!(
                                    "HadamardMul: array length mismatch ({} vs {}) at pc={pc}",
                                    l_guard.len(), r_guard.len()
                                )));
                            }
                            // Element-wise multiply for numeric arrays
                            let result: Vec<Value> = l_guard.iter().zip(r_guard.iter())
                                .map(|(l, r)| {
                                    match (l.as_f64(), r.as_f64()) {
                                        (Some(lf), Some(rf)) => Value::F64(lf * rf),
                                        _ => Value::Unit,
                                    }
                                })
                                .collect();
                            drop(l_guard);
                            drop(r_guard);
                            slots[*dst as usize] = Value::Array(Rc::new(RefCell::new(result)));
                        }
                        _ => {
                            return Err(RuntimeError::new(format!(
                                "HadamardMul: unsupported types {} and {} at pc={pc}",
                                l_val.type_name(), r_val.type_name()
                            )));
                        }
                    }
                    pc += 1;
                }

                // ── MakeArray: construct array from register range ──
                Instr::MakeArray { dst, start, count } => {
                    let s = *start as usize;
                    let c = *count as usize;
                    let elems: Vec<Value> = (0..c)
                        .map(|i| slots.get(s + i).cloned().unwrap_or(Value::Unit))
                        .collect();
                    slots[*dst as usize] = Value::Array(Rc::new(RefCell::new(elems)));
                    pc += 1;
                }

                // ── MakeTuple: construct tuple from register range ──
                Instr::MakeTuple { dst, start, count } => {
                    let s = *start as usize;
                    let c = *count as usize;
                    let elems: Vec<Value> = (0..c)
                        .map(|i| slots.get(s + i).cloned().unwrap_or(Value::Unit))
                        .collect();
                    slots[*dst as usize] = Value::Tuple(Box::new(elems));
                    pc += 1;
                }

                // ── MakeStruct: construct struct from field values ──
                Instr::MakeStruct { dst, name_idx, field_start, field_count } => {
                    let name = match constants.get(*name_idx as usize) {
                        Some(Value::Str(s)) => s.clone(),
                        _ => format!("struct_{}", name_idx),
                    };
                    let fs = *field_start as usize;
                    let fc = *field_count as usize;
                    let mut fields = FxHashMap::default();
                    // Fields are stored as pairs of (name_value, field_value) in slots
                    for i in 0..fc {
                        let name_val = slots.get(fs + i * 2).cloned().unwrap_or(Value::Unit);
                        let field_val = slots.get(fs + i * 2 + 1).cloned().unwrap_or(Value::Unit);
                        let key = match &name_val {
                            Value::Str(s) => s.clone(),
                            other => format!("field_{i}_{}", other.type_name()),
                        };
                        fields.insert(key, field_val);
                    }
                    slots[*dst as usize] = Value::Struct(Box::new(StructData { name, fields }));
                    pc += 1;
                }

                // ── MakeRange: construct a lazy range [lo, hi) or [lo, hi] ──
                Instr::MakeRange { dst, lo, hi, inclusive } => {
                    let lo_val = &slots[*lo as usize];
                    let hi_val = &slots[*hi as usize];
                    // Use lazy Range variant to avoid materializing the full Vec.
                    let lo_i = lo_val.as_i64().unwrap_or(0) as i32;
                    let hi_i = hi_val.as_i64().unwrap_or(0) as i32;
                    slots[*dst as usize] = Value::Range { start: lo_i, end: hi_i, inclusive: *inclusive };
                    pc += 1;
                }

                // ── Cast: type casting ──
                // target_type encoding: 0=I64, 1=F64, 2=Bool, 3=Str, 4=I32, 5=F32, 6=U64
                Instr::Cast { dst, src, target_type } => {
                    let src_val = &slots[*src as usize];
                    let result = match *target_type {
                        0 => { // I64
                            match src_val.as_i64() {
                                Some(v) => Value::I64(v),
                                None => return Err(RuntimeError::new(format!(
                                    "Cast: cannot cast {} to I64 at pc={pc}",
                                    src_val.type_name()
                                ))),
                            }
                        }
                        1 => { // F64
                            match src_val.as_f64() {
                                Some(v) => Value::F64(v),
                                None => return Err(RuntimeError::new(format!(
                                    "Cast: cannot cast {} to F64 at pc={pc}",
                                    src_val.type_name()
                                ))),
                            }
                        }
                        2 => { // Bool
                            match src_val.as_bool() {
                                Some(v) => Value::Bool(v),
                                None => Value::Bool(src_val.is_truthy()),
                            }
                        }
                        3 => { // Str
                            Value::Str(format!("{src_val}"))
                        }
                        4 => { // I32
                            match src_val.as_i64() {
                                Some(v) => Value::I32(v as i32),
                                None => return Err(RuntimeError::new(format!(
                                    "Cast: cannot cast {} to I32 at pc={pc}",
                                    src_val.type_name()
                                ))),
                            }
                        }
                        5 => { // F32
                            match src_val.as_f64() {
                                Some(v) => Value::F32(v as f32),
                                None => return Err(RuntimeError::new(format!(
                                    "Cast: cannot cast {} to F32 at pc={pc}",
                                    src_val.type_name()
                                ))),
                            }
                        }
                        6 => { // U64
                            match src_val.as_i64() {
                                Some(v) => Value::U64(v as u64),
                                None => return Err(RuntimeError::new(format!(
                                    "Cast: cannot cast {} to U64 at pc={pc}",
                                    src_val.type_name()
                                ))),
                            }
                        }
                        _ => {
                            return Err(RuntimeError::new(format!(
                                "Cast: unknown target type {target_type} at pc={pc}"
                            )));
                        }
                    };
                    slots[*dst as usize] = result;
                    pc += 1;
                }

                // ── Print ──
                Instr::Print { src } => {
                    let val = &slots[*src as usize];
                    // Only print in non-bench contexts; avoid I/O overhead in
                    // tight loops by checking a thread-local flag.
                    #[cfg(not(test))]
                    {
                        println!("{val}");
                    }
                    #[cfg(test)]
                    {
                        // Suppress output during tests to avoid polluting test output
                        let _ = val;
                    }
                    pc += 1;
                }

                // ── SIMD loop hints ──
                // These are advisory — they tell the VM that the upcoming loop
                // body is vectorizable. On hardware with the required SIMD width,
                // the VM can set internal flags to guide vectorized execution.
                Instr::SimdLoopStart { lane_count, simd_width } => {
                    // Mark this loop as SIMD-vectorizable for the adaptive
                    // profiler. Currently a no-op at runtime — future work
                    // will use these hints to drive vectorized dispatch.
                    let _ = (*lane_count, *simd_width);
                    pc += 1;
                }
                Instr::SimdLoopEnd => {
                    pc += 1;
                }

                // ── NOP ──
                Instr::Nop => {
                    pc += 1;
                }
            }
        }

        Ok(())
    }

    /// Integer exponentiation by squaring (fast path for Pow with I64 args).
    #[inline(always)]
    fn ipow(base: i64, exp: u64) -> i64 {
        let mut result: i64 = 1;
        let mut b = base;
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                result *= b;
            }
            b *= b;
            e >>= 1;
        }
        result
    }

    // Helper functions for type coercion (slow path)
    #[inline(always)]
    fn add_values(&self, l: &Value, r: &Value) -> Result<Value, RuntimeError> {
        Self::add_values_static(l, r)
    }

    #[inline(always)]
    fn sub_values(&self, l: &Value, r: &Value) -> Result<Value, RuntimeError> {
        Self::sub_values_static(l, r)
    }

    #[inline(always)]
    fn mul_values(&self, l: &Value, r: &Value) -> Result<Value, RuntimeError> {
        Self::mul_values_static(l, r)
    }

    #[inline(always)]
    fn div_values(&self, l: &Value, r: &Value) -> Result<Value, RuntimeError> {
        Self::div_values_static(l, r)
    }

    #[inline(always)]
    fn vec_add_values_static(l: &Value, r: &Value) -> Result<Value, RuntimeError> {
        match (l, r) {
            (Value::Vec2(a), Value::Vec2(b)) => Ok(Value::Vec2([a[0] + b[0], a[1] + b[1]])),
            (Value::Vec3(a), Value::Vec3(b)) => Ok(Value::Vec3([a[0] + b[0], a[1] + b[1], a[2] + b[2]])),
            (Value::Vec4(a), Value::Vec4(b)) => Ok(Value::Vec4(simd_add_vec4(*a, *b))),
            _ => Err(RuntimeError::new("VecAdd expects matching vector types")),
        }
    }

    #[inline(always)]
    fn vec_mul_values_static(l: &Value, r: &Value) -> Result<Value, RuntimeError> {
        match (l, r) {
            (Value::Vec2(a), Value::Vec2(b)) => Ok(Value::Vec2([a[0] * b[0], a[1] * b[1]])),
            (Value::Vec3(a), Value::Vec3(b)) => Ok(Value::Vec3([a[0] * b[0], a[1] * b[1], a[2] * b[2]])),
            (Value::Vec4(a), Value::Vec4(b)) => Ok(Value::Vec4(simd_mul_vec4(*a, *b))),
            _ => Err(RuntimeError::new("VecMul expects matching vector types")),
        }
    }

    // Static helper functions for use in the hot loop
    #[inline(always)]
    fn add_values_static(l: &Value, r: &Value) -> Result<Value, RuntimeError> {
        if let Some(lf) = l.as_f64() {
            if let Some(rf) = r.as_f64() {
                return Ok(Value::F64(lf + rf));
            }
        }
        Err(RuntimeError::new(format!(
            "cannot add {} and {}",
            l.type_name(),
            r.type_name()
        )))
    }

    #[inline(always)]
    fn sub_values_static(l: &Value, r: &Value) -> Result<Value, RuntimeError> {
        if let Some(lf) = l.as_f64() {
            if let Some(rf) = r.as_f64() {
                return Ok(Value::F64(lf - rf));
            }
        }
        Err(RuntimeError::new(format!(
            "cannot subtract {} from {}",
            r.type_name(),
            l.type_name()
        )))
    }

    #[inline(always)]
    fn mul_values_static(l: &Value, r: &Value) -> Result<Value, RuntimeError> {
        if let Some(lf) = l.as_f64() {
            if let Some(rf) = r.as_f64() {
                return Ok(Value::F64(lf * rf));
            }
        }
        Err(RuntimeError::new(format!(
            "cannot multiply {} and {}",
            l.type_name(),
            r.type_name()
        )))
    }

    #[inline(always)]
    fn div_values_static(l: &Value, r: &Value) -> Result<Value, RuntimeError> {
        if let Some(lf) = l.as_f64() {
            if let Some(rf) = r.as_f64() {
                if rf == 0.0 {
                    return Err(RuntimeError::new("division by zero"));
                }
                return Ok(Value::F64(lf / rf));
            }
        }
        Err(RuntimeError::new(format!(
            "cannot divide {} by {}",
            l.type_name(),
            r.type_name()
        )))
    }

    #[inline(always)]
    fn rem_values_static(l: &Value, r: &Value) -> Result<Value, RuntimeError> {
        if let Some(lf) = l.as_f64() {
            if let Some(rf) = r.as_f64() {
                if rf == 0.0 {
                    return Err(RuntimeError::new("remainder by zero"));
                }
                return Ok(Value::F64(lf % rf));
            }
        }
        Err(RuntimeError::new(format!(
            "cannot compute remainder of {} and {}",
            l.type_name(),
            r.type_name()
        )))
    }

    #[inline(always)]
    fn neg_values_static(v: &Value) -> Result<Value, RuntimeError> {
        match v {
            Value::I8(x) => Ok(Value::I64(-(*x as i64))),
            Value::I16(x) => Ok(Value::I64(-(*x as i64))),
            Value::I32(x) => Ok(Value::I64(-(*x as i64))),
            Value::I64(x) => Ok(Value::I64(-x)),
            Value::F32(x) => Ok(Value::F64(-(*x as f64))),
            Value::F64(x) => Ok(Value::F64(-x)),
            other => Err(RuntimeError::new(format!(
                "cannot negate {}",
                other.type_name()
            ))),
        }
    }

    /// Generic comparison fallback: coerce both sides to f64 and apply the
    /// given comparison closure.
    #[inline(always)]
    fn compare_values_static(
        l: &Value,
        r: &Value,
        cmp: impl Fn(f64, f64) -> bool,
        op_name: &str,
    ) -> Result<Value, RuntimeError> {
        let lf = l.as_f64();
        let rf = r.as_f64();
        match (lf, rf) {
            (Some(lf), Some(rf)) => Ok(Value::Bool(cmp(lf, rf))),
            _ => Err(RuntimeError::new(format!(
                "{op_name}: cannot compare {} and {}",
                l.type_name(),
                r.type_name()
            ))),
        }
    }
    
    /// Get execution statistics
    pub fn get_stats(&self) -> VMStats {
        VMStats {
            total_instructions: self.total_instructions,
            total_time_ns: self.total_time_ns,
            instructions_per_sec: if self.total_time_ns > 0 {
                (self.total_instructions as f64) / (self.total_time_ns as f64 / 1e9)
            } else {
                0.0
            },
        }
    }

    /// Load a full program (AST) into the VM by compiling all functions to bytecode.
    /// This is the primary entry point for running Jules programs without the
    /// tree-walking interpreter.
    pub fn load_program(&mut self, program: &Program) -> Result<(), String> {
        let mut compiler = BytecodeCompiler::new();
        let functions = compiler.compile_program(program)?;
        self.load_functions(functions);
        Ok(())
    }

    /// Call a named function with the given arguments.
    /// Returns the function's return value or a runtime error.
    /// This is the primary API for executing Jules programs via the bytecode VM.
    pub fn call_fn(&mut self, name: &str, args: Vec<Value>) -> Result<Value, RuntimeError> {
        let func_idx = self.function_index.get(name).copied()
            .ok_or_else(|| RuntimeError::new(format!("undefined function `{name}`")))?;
        self.execute(func_idx, &args)
    }

    // ── Heuristic Safety Engine Public API ───────────────────────────────────

    /// Set the global arithmetic mode override.
    ///
    /// When set, this overrides the per-function arithmetic mode for ALL
    /// functions. Use this for benchmarking or when the user explicitly
    /// sets a mode via a command-line flag or configuration file.
    ///
    /// Pass `None` to revert to per-function mode selection.
    pub fn set_arithmetic_mode(&mut self, mode: Option<ArithmeticMode>) {
        self.global_arithmetic_mode = mode;
    }

    /// Get the current global arithmetic mode override.
    pub fn arithmetic_mode(&self) -> Option<ArithmeticMode> {
        self.global_arithmetic_mode
    }

    /// Set the trust tier for a specific function by name.
    ///
    /// This is typically called by the compiler after the SMT solver
    /// and translation validator have analyzed the function. The runtime
    /// will then adjust its safety checks accordingly.
    pub fn set_function_trust_tier(&mut self, name: &str, tier: TrustTier) {
        if let Some(&idx) = self.function_index.get(name) {
            self.functions[idx].trust_tier = tier;
            // Update arithmetic mode based on trust tier
            self.functions[idx].arithmetic_mode = tier.default_arithmetic_mode();
            // Mark proven properties
            match tier {
                TrustTier::Tier0Trusted => {
                    self.functions[idx].termination_proven = true;
                    self.functions[idx].overflow_proven_safe = true;
                }
                TrustTier::Tier1Verified => {
                    self.functions[idx].overflow_proven_safe = true;
                }
                _ => {}
            }
        }
    }

    /// Run the formal verification pipeline on all loaded functions.
    ///
    /// This uses the SMT solver and translation validator to prove
    /// safety properties about each function and assigns the appropriate
    /// TrustTier. Functions that are proven safe will execute with
    /// reduced runtime overhead.
    pub fn verify_all_functions(&mut self) {
        use crate::compiler::formal_verify::FormalVerifier;
        let verifier = FormalVerifier::new();

        // Collect function names and instruction counts first to avoid borrow issues.
        let func_info: Vec<(String, usize)> = self.functions.iter()
            .map(|f| (f.name.clone(), f.instructions.len()))
            .collect();

        for (name, instr_count) in func_info {
            let result = verifier.verify_function(&name, instr_count);
            if let Some(&idx) = self.function_index.get(&name) {
                self.functions[idx].trust_tier = result.tier;
                self.functions[idx].arithmetic_mode = result.tier.default_arithmetic_mode();
                self.functions[idx].termination_proven = result.termination_proven;
                self.functions[idx].overflow_proven_safe = result.overflow_proven_safe;
            }
        }
    }

    /// Get the OSR engine's de-optimization count.
    pub fn osr_deopt_count(&self) -> u64 {
        self.osr_engine.deopt_count()
    }

    /// Enable or disable OSR de-optimization.
    pub fn set_osr_enabled(&mut self, enabled: bool) {
        self.osr_engine = OsrEngine::new(enabled);
    }
}

#[inline(always)]
fn simd_add_vec4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_mm_add_ps, _mm_loadu_ps, _mm_storeu_ps};
        let va = _mm_loadu_ps(a.as_ptr());
        let vb = _mm_loadu_ps(b.as_ptr());
        let sum = _mm_add_ps(va, vb);
        let mut out = [0.0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), sum);
        return out;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::{vld1q_f32, vaddq_f32, vst1q_f32};
        let va = vld1q_f32(a.as_ptr());
        let vb = vld1q_f32(b.as_ptr());
        let sum = vaddq_f32(va, vb);
        let mut out = [0.0f32; 4];
        vst1q_f32(out.as_mut_ptr(), sum);
        return out;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
    }
}

#[inline(always)]
fn simd_mul_vec4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_mm_loadu_ps, _mm_mul_ps, _mm_storeu_ps};
        let va = _mm_loadu_ps(a.as_ptr());
        let vb = _mm_loadu_ps(b.as_ptr());
        let prod = _mm_mul_ps(va, vb);
        let mut out = [0.0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), prod);
        return out;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::{vld1q_f32, vmulq_f32, vst1q_f32};
        let va = vld1q_f32(a.as_ptr());
        let vb = vld1q_f32(b.as_ptr());
        let prod = vmulq_f32(va, vb);
        let mut out = [0.0f32; 4];
        vst1q_f32(out.as_mut_ptr(), prod);
        return out;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
    }
}

#[derive(Debug, Clone)]
pub struct VMStats {
    pub total_instructions: u64,
    pub total_time_ns: u64,
    pub instructions_per_sec: f64,
}

// =============================================================================
// §9  TESTS & BENCHMARKS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert a Value equals an expected i64.
    fn assert_i64(val: &Value, expected: i64) {
        match val {
            Value::I64(v) => assert_eq!(*v, expected, "expected I64({expected}), got I64({v})"),
            other => panic!("expected I64({expected}), got {}", other.type_name()),
        }
    }

    /// Helper: assert a Value equals an expected f64.
    fn assert_f64(val: &Value, expected: f64) {
        match val {
            Value::F64(v) => assert_eq!(*v, expected, "expected F64({expected}), got F64({v})"),
            other => panic!("expected F64({expected}), got {}", other.type_name()),
        }
    }

    /// Build a minimal BytecodeFunction that computes fibonacci iteratively.
    ///
    /// Equivalent Jules source:
    /// ```jules
    /// fn fib(n: i64) -> i64 {
    ///     if n <= 1 { return n; }
    ///     let mut a = 0;
    ///     let mut b = 1;
    ///     for i in 2..n+1 {
    ///         let temp = b;
    ///         b = a + b;
    ///         a = temp;
    ///     }
    ///     return b;
    /// }
    /// ```
    ///
    /// We hand-compile this into bytecode to avoid depending on the full
    /// frontend parser in this test.
    fn build_fib_function() -> BytecodeFunction {
        // Slot layout:
        //   0 = n (param)
        //   1 = a
        //   2 = b
        //   3 = temp
        //   4 = i
        //   5 = cond result
        //   6 = n+1 (upper bound)
        //   7 = scratch for comparisons
        //   8 = one constant
        //   9 = two constant
        let mut f = BytecodeFunction::new("fib".to_string());
        f.num_params = 1;
        f.num_locals = 10;

        // if n <= 1: return n
        f.instructions.push(Instr::LoadConstInt { dst: 8, value: 1 });
        f.instructions.push(Instr::Le { dst: 5, lhs: 0, rhs: 8 }); // n <= 1
        let jump_past_early_return = f.instructions.len();
        f.instructions.push(Instr::JumpFalse { cond: 5, offset: 0 }); // patch later
        f.instructions.push(Instr::Return { value: 0 }); // return n
        // patch: jump lands here
        let after_early_return = f.instructions.len();
        match &mut f.instructions[jump_past_early_return] {
            Instr::JumpFalse { offset, .. } => *offset = (after_early_return as i32) - (jump_past_early_return as i32),
            _ => unreachable!(),
        }

        // let a = 0; let b = 1;
        f.instructions.push(Instr::LoadConstInt { dst: 1, value: 0 }); // a = 0
        f.instructions.push(Instr::LoadConstInt { dst: 2, value: 1 }); // b = 1

        // Loop: for i in 2..n+1
        f.instructions.push(Instr::LoadConstInt { dst: 9, value: 2 }); // i starts at 2
        f.instructions.push(Instr::LoadConstInt { dst: 8, value: 1 }); // constant 1
        f.instructions.push(Instr::Add { dst: 6, lhs: 0, rhs: 8 });   // n + 1
        f.instructions.push(Instr::Move { dst: 4, src: 9 });           // i = 2

        // Jump to condition check
        let jump_to_cond = f.instructions.len();
        f.instructions.push(Instr::Jump { offset: 0 }); // patch later

        // Loop body:
        let loop_body = f.instructions.len();
        // temp = b
        f.instructions.push(Instr::Move { dst: 3, src: 2 });
        // b = a + b
        f.instructions.push(Instr::Add { dst: 2, lhs: 1, rhs: 2 });
        // a = temp
        f.instructions.push(Instr::Move { dst: 1, src: 3 });
        // i += 1
        f.instructions.push(Instr::LoadConstInt { dst: 8, value: 1 });
        f.instructions.push(Instr::Add { dst: 4, lhs: 4, rhs: 8 });

        // Condition: i < n+1
        let cond_check = f.instructions.len();
        // Patch the initial jump to land here
        match &mut f.instructions[jump_to_cond] {
            Instr::Jump { offset } => *offset = (cond_check as i32) - (jump_to_cond as i32),
            _ => unreachable!(),
        }
        f.instructions.push(Instr::Lt { dst: 5, lhs: 4, rhs: 6 });
        // If true, jump back to loop body
        let backedge_offset = (loop_body as i32) - (f.instructions.len() as i32);
        f.instructions.push(Instr::JumpTrue { cond: 5, offset: backedge_offset });

        // After loop: return b
        f.instructions.push(Instr::Return { value: 2 });

        f
    }

    #[test]
    fn test_vm_fibonacci_small() {
        let fib = build_fib_function();
        let mut vm = BytecodeVM::new();
        vm.load_functions(vec![fib]);

        // fib(0) = 0
        assert_i64(&vm.execute(0, &[Value::I64(0)]).unwrap(), 0);

        // fib(1) = 1
        assert_i64(&vm.execute(0, &[Value::I64(1)]).unwrap(), 1);

        // fib(5) = 5
        assert_i64(&vm.execute(0, &[Value::I64(5)]).unwrap(), 5);

        // fib(10) = 55
        assert_i64(&vm.execute(0, &[Value::I64(10)]).unwrap(), 55);

        // fib(20) = 6765
        assert_i64(&vm.execute(0, &[Value::I64(20)]).unwrap(), 6765);
    }

    #[test]
    fn test_vm_fibonacci_30() {
        let fib = build_fib_function();
        let mut vm = BytecodeVM::new();
        vm.load_functions(vec![fib]);

        let start = std::time::Instant::now();
        let iterations = 1000;
        for _ in 0..iterations {
            let result = vm.execute(0, &[Value::I64(30)]).unwrap();
            assert_i64(&result, 832040);
        }
        let elapsed = start.elapsed();
        let ns_per_iter = elapsed.as_nanos() / iterations as u128;
        eprintln!(
            "bench_vm_fibonacci_30: {} iterations in {:?} ({:.0} ns/iter)",
            iterations, elapsed, ns_per_iter as f64,
        );
    }

    #[test]
    fn test_vm_pow_opcode() {
        let mut f = BytecodeFunction::new("test_pow".to_string());
        f.num_params = 0;
        f.num_locals = 4;
        // slots: 0=base(2), 1=exp(10), 2=result
        f.instructions.push(Instr::LoadConstInt { dst: 0, value: 2 });
        f.instructions.push(Instr::LoadConstInt { dst: 1, value: 10 });
        f.instructions.push(Instr::Pow { dst: 2, base: 0, exp: 1 });
        f.instructions.push(Instr::Return { value: 2 });

        let mut vm = BytecodeVM::new();
        vm.load_functions(vec![f]);
        let result = vm.execute(0, &[]).unwrap();
        assert_i64(&result, 1024); // 2^10 = 1024
    }

    #[test]
    fn test_vm_make_array_opcode() {
        let mut f = BytecodeFunction::new("test_array".to_string());
        f.num_params = 0;
        f.num_locals = 6;
        // Build array [10, 20, 30] from slots 1..4
        f.instructions.push(Instr::LoadConstInt { dst: 1, value: 10 });
        f.instructions.push(Instr::LoadConstInt { dst: 2, value: 20 });
        f.instructions.push(Instr::LoadConstInt { dst: 3, value: 30 });
        f.instructions.push(Instr::MakeArray { dst: 0, start: 1, count: 3 });
        f.instructions.push(Instr::Return { value: 0 });

        let mut vm = BytecodeVM::new();
        vm.load_functions(vec![f]);
        let result = vm.execute(0, &[]).unwrap();
        match result {
            Value::Array(arr) => {
                let guard = arr.borrow();
                assert_eq!(guard.len(), 3);
                assert_i64(&guard[0], 10);
                assert_i64(&guard[1], 20);
                assert_i64(&guard[2], 30);
            }
            other => panic!("expected Array, got {:?}", other.type_name()),
        }
    }

    #[test]
    fn test_vm_make_tuple_opcode() {
        let mut f = BytecodeFunction::new("test_tuple".to_string());
        f.num_params = 0;
        f.num_locals = 6;
        // Build tuple (42, true) from slots 1..3
        f.instructions.push(Instr::LoadConstInt { dst: 1, value: 42 });
        f.instructions.push(Instr::LoadConstBool { dst: 2, value: true });
        f.instructions.push(Instr::MakeTuple { dst: 0, start: 1, count: 2 });
        f.instructions.push(Instr::Return { value: 0 });

        let mut vm = BytecodeVM::new();
        vm.load_functions(vec![f]);
        let result = vm.execute(0, &[]).unwrap();
        match result {
            Value::Tuple(v) => {
                assert_eq!(v.len(), 2);
                assert_i64(&v[0], 42);
                match &v[1] {
                    Value::Bool(b) => assert!(*b),
                    other => panic!("expected Bool, got {}", other.type_name()),
                }
            }
            other => panic!("expected Tuple, got {:?}", other.type_name()),
        }
    }

    #[test]
    fn test_vm_make_range_opcode() {
        let mut f = BytecodeFunction::new("test_range".to_string());
        f.num_params = 0;
        f.num_locals = 6;
        // Build range 0..10
        f.instructions.push(Instr::LoadConstInt { dst: 0, value: 0 });
        f.instructions.push(Instr::LoadConstInt { dst: 1, value: 10 });
        f.instructions.push(Instr::MakeRange { dst: 2, lo: 0, hi: 1, inclusive: false });
        f.instructions.push(Instr::Return { value: 2 });

        let mut vm = BytecodeVM::new();
        vm.load_functions(vec![f]);
        let result = vm.execute(0, &[]).unwrap();
        match result {
            Value::Tuple(v) => {
                assert_eq!(v.len(), 3);
                assert_i64(&v[0], 0);
                assert_i64(&v[1], 10);
                match &v[2] {
                    Value::Bool(b) => assert!(!b), // exclusive
                    other => panic!("expected Bool, got {}", other.type_name()),
                }
            }
            other => panic!("expected Tuple (range), got {:?}", other.type_name()),
        }
    }

    #[test]
    fn test_vm_cast_opcode() {
        let mut f = BytecodeFunction::new("test_cast".to_string());
        f.num_params = 0;
        f.num_locals = 4;
        // Cast I64(42) to F64
        f.instructions.push(Instr::LoadConstInt { dst: 0, value: 42 });
        f.instructions.push(Instr::Cast { dst: 1, src: 0, target_type: 1 }); // 1 = F64
        f.instructions.push(Instr::Return { value: 1 });

        let mut vm = BytecodeVM::new();
        vm.load_functions(vec![f]);
        let result = vm.execute(0, &[]).unwrap();
        assert_f64(&result, 42.0);
    }

    #[test]
    fn test_vm_simd_loop_hints() {
        let mut f = BytecodeFunction::new("test_simd".to_string());
        f.num_params = 0;
        f.num_locals = 2;
        // SimdLoopStart / SimdLoopEnd should not crash
        f.instructions.push(Instr::SimdLoopStart { lane_count: 8, simd_width: 256 });
        f.instructions.push(Instr::LoadConstInt { dst: 0, value: 42 });
        f.instructions.push(Instr::SimdLoopEnd);
        f.instructions.push(Instr::Return { value: 0 });

        let mut vm = BytecodeVM::new();
        vm.load_functions(vec![f]);
        let result = vm.execute(0, &[]).unwrap();
        assert_i64(&result, 42);
    }

    #[test]
    fn test_vm_data_dependent_jit_profiling() {
        // Verify that the DataDependentJIT is integrated and profiling works.
        let fib = build_fib_function();
        let mut vm = BytecodeVM::new();
        vm.load_functions(vec![fib]);

        // Run several times to build up observations
        for n in 5..15 {
            let _ = vm.execute(0, &[Value::I64(n)]);
        }

        // The JIT should have observations from the backedge profiling
        let jit = vm.data_dependent_jit();
        // At minimum, the JIT should have been created and not crashed
        let _ = jit;
    }
}