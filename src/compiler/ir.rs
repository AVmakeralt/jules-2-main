// =============================================================================
// jules/src/compiler/ir.rs
//
// Unified Semantic SSA IR for the Jules programming language.
//
// This is THE canonical IR that all compiler subsystems operate on:
//   - Superoptimizer (MCTS, e-graph, semantic)
//   - JIT backends (tracing, AOT)
//   - ML kernel compiler
//   - Bytecode VM
//   - Vectorizer
//   - Ownership analyzer
//
// Design principles:
//   1. SSA form — every value defined exactly once
//   2. Explicit memory — alloc/load/store/borrow/alias are IR nodes
//   3. Effects are first-class — every node carries effect metadata
//   4. Ownership is in the IR — not a separate analysis pass
//   5. Tensor and scalar unified — same ops, different type parameters
//   6. Cost hooks on every node — for superoptimizer guidance
//   7. Parallel semantics first-class — tasks, regions, dependencies
// =============================================================================

use crate::compiler::ast::{BinOpKind, UnOpKind, ElemType};
use crate::compiler::lexer::Span;
use std::fmt;

// ─── Identifiers ─────────────────────────────────────────────────────────────

/// A value identifier in SSA form. Each ValueId is defined exactly once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// A basic block identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// A function identifier in the IR.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionId(pub String);

// ─── Effects System ───────────────────────────────────────────────────────────

/// Effect flags attached to every IR node.
/// These determine what optimizations are legal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectFlags(u16);

impl EffectFlags {
    pub const PURE:       EffectFlags = EffectFlags(0b0000_0000_0001);
    pub const READONLY:   EffectFlags = EffectFlags(0b0000_0000_0010);
    pub const WRITE:      EffectFlags = EffectFlags(0b0000_0000_0100);
    pub const IO:         EffectFlags = EffectFlags(0b0000_0000_1000);
    pub const ATOMIC:     EffectFlags = EffectFlags(0b0000_0001_0000);
    pub const UNSAFE:     EffectFlags = EffectFlags(0b0000_0010_0000);
    pub const ALLOC:      EffectFlags = EffectFlags(0b0000_0100_0000);
    pub const BORROW:     EffectFlags = EffectFlags(0b0000_1000_0000);
    pub const PARALLEL:   EffectFlags = EffectFlags(0b0001_0000_0000);
    pub const TERMINATES: EffectFlags = EffectFlags(0b0010_0000_0000);

    pub fn none() -> Self { EffectFlags(0) }
    pub fn pure() -> Self { Self::PURE }

    pub fn contains(self, other: EffectFlags) -> bool {
        (self.0 & other.0) != 0
    }

    pub fn union(self, other: EffectFlags) -> EffectFlags {
        EffectFlags(self.0 | other.0)
    }

    pub fn is_pure(self) -> bool {
        self.contains(Self::PURE) && !self.contains(Self::WRITE)
            && !self.contains(Self::IO) && !self.contains(Self::ALLOC)
    }

    pub fn may_reorder_with(self, other: EffectFlags) -> bool {
        // Two nodes can be reordered if neither writes and neither does IO
        let conflict = Self::WRITE.0 | Self::IO.0 | Self::ATOMIC.0 | Self::UNSAFE.0;
        (self.0 & conflict) == 0 && (other.0 & conflict) == 0
    }
}

impl fmt::Display for EffectFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = vec![];
        if self.contains(Self::PURE)       { parts.push("pure"); }
        if self.contains(Self::READONLY)   { parts.push("readonly"); }
        if self.contains(Self::WRITE)      { parts.push("write"); }
        if self.contains(Self::IO)         { parts.push("io"); }
        if self.contains(Self::ATOMIC)     { parts.push("atomic"); }
        if self.contains(Self::UNSAFE)     { parts.push("unsafe"); }
        if self.contains(Self::ALLOC)      { parts.push("alloc"); }
        if self.contains(Self::BORROW)     { parts.push("borrow"); }
        if self.contains(Self::PARALLEL)   { parts.push("parallel"); }
        if self.contains(Self::TERMINATES) { parts.push("terminates"); }
        if parts.is_empty() { write!(f, "none") } else { write!(f, "{}", parts.join("|")) }
    }
}

// ─── Ownership Model ──────────────────────────────────────────────────────────

/// Ownership qualifier attached to value definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ownership {
    /// Single owner, move semantics. Default for heap objects and tensors.
    Own,
    /// No aliasing possible — optimizer may assume exclusive access.
    Unique,
    /// Shared read-only reference. Multiple readers allowed.
    Shared,
    /// Borrowed mutable reference. Exclusive access, no other refs.
    MutBorrow,
    /// Copy type — primitives that are implicitly copied (i32, f64, bool).
    Copy,
}

// ─── IR Types ──────────────────────────────────────────────────────────────────

/// Types in the unified IR. Simpler than AST types — lowered and canonicalized.
#[derive(Debug, Clone, PartialEq)]
pub enum IrType {
    /// Integer type with specified bit width.
    Int { width: u8, signed: bool },
    /// Floating-point type with specified bit width (16, 32, 64).
    Float { width: u8 },
    /// Boolean type.
    Bool,
    /// String type.
    String,
    /// Unit type (void).
    Unit,
    /// Never type (diverges).
    Never,
    /// Tensor type with element type and shape.
    Tensor { elem: ElemType, shape: Vec<IrDim> },
    /// SIMD vector type.
    Vec { elem: Box<IrType>, lanes: u32 },
    /// Reference/pointer type with ownership qualifier.
    Ref { inner: Box<IrType>, ownership: Ownership },
    /// Function pointer type.
    FnPtr { params: Vec<IrType>, ret: Box<IrType> },
    /// Struct type (named).
    Struct { name: String, fields: Vec<(String, IrType)> },
    /// Enum type (named).
    Enum { name: String },
    /// Opaque/external type.
    Opaque(String),
}

/// Dimension in a tensor shape.
#[derive(Debug, Clone, PartialEq)]
pub enum IrDim {
    Static(u64),
    Dynamic,
    Symbolic(String),
}

// ─── Cost Model ────────────────────────────────────────────────────────────────

/// Cost metadata attached to IR nodes for superoptimizer guidance.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CostHint {
    /// Estimated cycle count on target architecture.
    pub estimated_cycles: f32,
    /// SIMD vector width (1 = scalar).
    pub vector_width: u32,
    /// Instruction latency in cycles.
    pub latency: u8,
    /// Throughput (operations per cycle).
    pub throughput: f32,
    /// Cache pressure estimate (0-10 scale).
    pub cache_pressure: u8,
    /// Fusion potential (0-10 scale, 10 = highly fusible).
    pub fusion_potential: u8,
}

// ─── IR Instructions ──────────────────────────────────────────────────────────

/// A single instruction in the unified SSA IR.
#[derive(Debug, Clone, PartialEq)]
pub struct IrInstr {
    /// The result value (None for terminators and stores).
    pub dst: Option<ValueId>,
    /// The instruction opcode.
    pub op: IrOp,
    /// Source span for diagnostics.
    pub span: Span,
    /// Effect flags — determines legal optimizations.
    pub effects: EffectFlags,
    /// Ownership of the result value.
    pub ownership: Ownership,
    /// Cost hint for superoptimizer.
    pub cost: CostHint,
}

/// The operation of an IR instruction.
#[derive(Debug, Clone, PartialEq)]
pub enum IrOp {
    // ── Constants ────────────────────────────────────────────────────────
    /// Constant integer value.
    ConstInt { value: i128, ty: IrType },
    /// Constant float value.
    ConstFloat { bits: u64, ty: IrType },
    /// Constant boolean.
    ConstBool { value: bool },
    /// Constant string (index into string table).
    ConstStr { idx: u32 },
    /// Constant unit value.
    ConstUnit,

    // ── Arithmetic (unified for scalar + tensor) ─────────────────────────
    /// Binary arithmetic: add, sub, mul, div, rem, floor_div
    BinOp { op: BinOpKind, lhs: ValueId, rhs: ValueId },
    /// Unary: neg, not
    UnOp { op: UnOpKind, operand: ValueId },

    // ── Comparison ───────────────────────────────────────────────────────
    /// Integer/float comparison.
    ICmp { cond: ICmpCond, lhs: ValueId, rhs: ValueId },

    // ── Memory Operations ────────────────────────────────────────────────
    /// Stack allocation with alignment.
    Alloca { ty: IrType, align: u32 },
    /// Heap allocation with ownership.
    Alloc { ty: IrType, ownership: Ownership },
    /// Region-scoped allocation (freed when region ends).
    RegionAlloc { region: u32, ty: IrType },
    /// Load from pointer.
    Load { ptr: ValueId, ty: IrType },
    /// Store to pointer.
    Store { ptr: ValueId, value: ValueId },
    /// Move (ownership transfer).
    Move { src: ValueId },
    /// Copy (explicit duplication).
    Copy { src: ValueId },
    /// Borrow read-only reference.
    BorrowRead { src: ValueId },
    /// Borrow mutable reference.
    BorrowWrite { src: ValueId },

    // ── Tensor Operations ────────────────────────────────────────────────
    /// Matrix multiply.
    MatMul { lhs: ValueId, rhs: ValueId },
    /// Element-wise (Hadamard) multiply.
    HadamardMul { lhs: ValueId, rhs: ValueId },
    /// Element-wise divide.
    HadamardDiv { lhs: ValueId, rhs: ValueId },
    /// Tensor concatenation.
    TensorConcat { lhs: ValueId, rhs: ValueId },
    /// Outer product.
    OuterProd { lhs: ValueId, rhs: ValueId },
    /// Kronecker product.
    KronProd { lhs: ValueId, rhs: ValueId },
    /// Tensor reshape.
    Reshape { src: ValueId, new_shape: Vec<IrDim> },
    /// Tensor broadcast.
    Broadcast { src: ValueId, target_shape: Vec<IrDim> },

    // ── Control Flow (terminators) ───────────────────────────────────────
    /// Unconditional jump.
    Jump { target: BlockId },
    /// Conditional branch.
    CondBr { cond: ValueId, if_true: BlockId, if_false: BlockId },
    /// Return from function.
    Ret { value: Option<ValueId> },
    /// Unreachable code marker.
    Unreachable,

    // ── Phi Node (SSA merge) ─────────────────────────────────────────────
    /// SSA phi node: picks value based on which predecessor block was executed.
    Phi { incoming: Vec<(BlockId, ValueId)> },

    // ── Function Calls ───────────────────────────────────────────────────
    /// Call a function by name.
    Call { func: String, args: Vec<ValueId> },
    /// Call a native/extern function.
    CallNative { func_idx: u32, args: Vec<ValueId> },
    /// Tail call optimization.
    TailCall { func: String, args: Vec<ValueId> },

    // ── Parallel / Task Operations ───────────────────────────────────────
    /// Spawn a task (returns a task handle).
    TaskSpawn { func: String, args: Vec<ValueId>, ownership: TaskOwnership },
    /// Wait for task completion, get result.
    TaskJoin { task: ValueId },
    /// Start a parallel region.
    ParallelStart { region_id: u32 },
    /// End a parallel region (implicit join).
    ParallelEnd { region_id: u32 },
    /// Declare a dependency between tasks.
    TaskDepend { from: ValueId, to: ValueId },

    // ── Effects ──────────────────────────────────────────────────────────
    /// Emit an effect (IO, logging, etc.).
    Emit { effect: String, value: ValueId },

    // ── Type Operations ──────────────────────────────────────────────────
    /// Type cast (bit-preserving or widening).
    Cast { src: ValueId, target_ty: IrType },
    /// Type check (runtime guard).
    TypeCheck { value: ValueId, expected: IrType },

    // ── SIMD ─────────────────────────────────────────────────────────────
    /// SIMD vector operation.
    SimdOp { op: BinOpKind, lanes: u32, lhs: ValueId, rhs: ValueId },

    // ── Intrinsics ───────────────────────────────────────────────────────
    /// Invoke a declared intrinsic by name.
    Intrinsic { name: String, args: Vec<ValueId> },

    // ── Debug / Metadata ─────────────────────────────────────────────────
    /// Source location marker.
    DebugLoc { line: u32, col: u32 },
    /// No-op placeholder.
    Nop,
}

/// Integer comparison condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ICmpCond {
    Eq, Ne, Lt, Le, Gt, Ge,
}

/// How ownership transfers when spawning a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskOwnership {
    /// Data is moved into the task, no aliasing.
    Move,
    /// Data is borrowed read-only.
    Ref,
    /// Data is shared mutably via atomic.
    Shared,
}

// ─── Basic Block ───────────────────────────────────────────────────────────────

/// A basic block in the CFG: a sequence of instructions ending with a terminator.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: BlockId,
    pub instrs: Vec<IrInstr>,
    /// Predecessor blocks (filled during CFG construction).
    pub predecessors: Vec<BlockId>,
}

impl BasicBlock {
    pub fn new(id: BlockId) -> Self {
        BasicBlock {
            id,
            instrs: vec![],
            predecessors: vec![],
        }
    }

    pub fn is_terminated(&self) -> bool {
        self.instrs.last().map_or(false, |i| i.op.is_terminator())
    }
}

// ─── Function ──────────────────────────────────────────────────────────────────

/// A function in the unified IR.
#[derive(Debug, Clone)]
pub struct IrFunction {
    pub name: String,
    pub params: Vec<(ValueId, IrType)>,
    pub ret_ty: IrType,
    pub blocks: Vec<BasicBlock>,
    pub entry: BlockId,
    /// Effect annotation.
    pub effects: EffectFlags,
    /// Precondition constraints.
    pub requires: Vec<ValueId>,
    /// Postcondition constraints.
    pub ensures: Vec<ValueId>,
    /// Source span.
    pub span: Span,
}

// ─── Module ────────────────────────────────────────────────────────────────────

/// A compilation module containing functions, types, and intrinsics.
#[derive(Debug, Clone)]
pub struct IrModule {
    pub functions: Vec<IrFunction>,
    pub intrinsics: Vec<IrIntrinsic>,
    pub span: Span,
}

/// An intrinsic declaration.
#[derive(Debug, Clone)]
pub struct IrIntrinsic {
    pub name: String,
    pub param_types: Vec<IrType>,
    pub ret_type: IrType,
    pub effects: EffectFlags,
}

// ─── Op helpers ────────────────────────────────────────────────────────────────

impl IrOp {
    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            IrOp::Jump { .. }
                | IrOp::CondBr { .. }
                | IrOp::Ret { .. }
                | IrOp::Unreachable
        )
    }

    pub fn is_pure(&self) -> bool {
        matches!(
            self,
            IrOp::ConstInt { .. }
                | IrOp::ConstFloat { .. }
                | IrOp::ConstBool { .. }
                | IrOp::ConstStr { .. }
                | IrOp::ConstUnit
                | IrOp::BinOp { .. }
                | IrOp::UnOp { .. }
                | IrOp::ICmp { .. }
                | IrOp::Phi { .. }
                | IrOp::Cast { .. }
                | IrOp::Copy { .. }
                | IrOp::Move { .. }
        )
    }

    /// Returns all ValueIds used by this instruction.
    pub fn used_values(&self) -> Vec<ValueId> {
        match self {
            IrOp::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],
            IrOp::UnOp { operand, .. } => vec![*operand],
            IrOp::ICmp { lhs, rhs, .. } => vec![*lhs, *rhs],
            IrOp::Load { ptr, .. } => vec![*ptr],
            IrOp::Store { ptr, value } => vec![*ptr, *value],
            IrOp::Move { src } | IrOp::Copy { src } => vec![*src],
            IrOp::BorrowRead { src } | IrOp::BorrowWrite { src } => vec![*src],
            IrOp::MatMul { lhs, rhs } => vec![*lhs, *rhs],
            IrOp::HadamardMul { lhs, rhs } => vec![*lhs, *rhs],
            IrOp::HadamardDiv { lhs, rhs } => vec![*lhs, *rhs],
            IrOp::TensorConcat { lhs, rhs } => vec![*lhs, *rhs],
            IrOp::OuterProd { lhs, rhs } => vec![*lhs, *rhs],
            IrOp::KronProd { lhs, rhs } => vec![*lhs, *rhs],
            IrOp::Reshape { src, .. } => vec![*src],
            IrOp::Broadcast { src, .. } => vec![*src],
            IrOp::CondBr { cond, .. } => vec![*cond],
            IrOp::Ret { value } => value.map_or(vec![], |v| vec![v]),
            IrOp::Phi { incoming } => incoming.iter().map(|&(_, v)| v).collect(),
            IrOp::Call { args, .. } | IrOp::CallNative { args, .. } | IrOp::TailCall { args, .. } => args.clone(),
            IrOp::TaskSpawn { args, .. } => args.clone(),
            IrOp::TaskJoin { task } => vec![*task],
            IrOp::TaskDepend { from, to } => vec![*from, *to],
            IrOp::Emit { value, .. } => vec![*value],
            IrOp::Cast { src, .. } => vec![*src],
            IrOp::TypeCheck { value, .. } => vec![*value],
            IrOp::SimdOp { lhs, rhs, .. } => vec![*lhs, *rhs],
            IrOp::Intrinsic { args, .. } => args.clone(),
            _ => vec![],
        }
    }
}
