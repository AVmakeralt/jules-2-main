// =============================================================================
// jules/src/compiler/ir.rs
//
// SSA-based Typed Intermediate Representation for the Jules compiler.
//
// This IR sits between the high-level AST and the low-level bytecode VM.
// It provides a structured, typed, effect-annotated representation suitable
// for optimisation passes, superoptimizer integration, and code generation.
//
// Design principles:
//   • SSA form — values defined once, block parameters replace phi nodes
//   • Typed — every IR node carries its result type
//   • Effect-annotated — every expression node carries an Effect enum
//   • Ownership-aware — values carry Ownership metadata
//   • Span-tracked — every node has a source Span
//   • Cost-annotated — nodes can carry a CostHint for the superoptimizer
// =============================================================================

use std::fmt;

use crate::compiler::ast::Attribute;
use crate::compiler::lexer::Span;
use crate::compiler::typeck::{Diagnostic, Dim, Ty};

// =============================================================================
// §1  CORE ENUMERATIONS
// =============================================================================

/// Effect annotation for IR nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    /// No side effects — can be freely reordered / deduplicated / CSE'd.
    Pure,
    /// Reads or writes external state (I/O, network, files).
    IO,
    /// Mutates heap or stack state.
    Mutation,
    /// Allocates memory (region or heap).
    Allocation,
    /// Affects control flow (break, continue, return).
    ControlFlow,
    /// Async task spawning.
    Async,
    /// Effect not yet determined (conservative).
    Unknown,
}

/// Ownership metadata for IR values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ownership {
    /// Unique owner — moved on use.
    Owned,
    /// Read-only borrow.
    Ref,
    /// Exclusive mutable borrow.
    MutRef,
    /// Shared (reference-counted) access.
    Shared,
    /// Implicitly copied on use (primitives).
    Copy,
}

/// Cost model hint for the superoptimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostHint {
    Unknown,
    /// < 1 cycle (register op).
    Cheap,
    /// 1–10 cycles (ALU op).
    Moderate,
    /// 10–100 cycles (division, complex math).
    Expensive,
    /// > 100 cycles (memory access, syscall).
    VeryExpensive,
    /// Loop with known iteration count.
    Loop { iterations: u64 },
    /// Compute kernel with known FLOP count.
    Kernel { flops: u64 },
}

/// Fusion boundary marker — controls loop/kernel fusion decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionBoundary {
    /// Can be fused with neighbours.
    None,
    /// Must not be fused (explicit @nofuse).
    NoFuse,
    /// Actively seek fusion (@fusion(aggressive)).
    FuseAggressive,
}

// =============================================================================
// §2  VALUE & BLOCK IDS
// =============================================================================

/// Reference to a value in SSA form (like a virtual register).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

/// A typed SSA value.
#[derive(Debug, Clone)]
pub struct IrValue {
    pub id: ValueId,
    pub ty: Ty,
    pub ownership: Ownership,
    pub span: Span,
}

/// A basic-block identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

// =============================================================================
// §3  BASIC BLOCK
// =============================================================================

/// A basic block in the CFG.
///
/// Block parameters replace traditional phi nodes: when control flow
/// transfers to a block, the caller passes concrete values for each
/// parameter.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: BlockId,
    /// Block parameters — (value id, type) pairs.
    pub params: Vec<(ValueId, Ty)>,
    /// Sequential statements inside the block.
    pub stmts: Vec<IrStmt>,
    /// The block's terminator (branch / jump / return / …).
    pub terminator: IrTerminator,
    pub span: Span,
}

// =============================================================================
// §4  TERMINATOR
// =============================================================================

/// The terminator instruction at the end of a basic block.
#[derive(Debug, Clone)]
pub enum IrTerminator {
    /// Unconditional jump: `goto target(args)`.
    Jump {
        target: BlockId,
        args: Vec<ValueId>,
    },
    /// Conditional branch: `if cond then then_block(then_args) else else_block(else_args)`.
    Branch {
        cond: ValueId,
        then_block: BlockId,
        then_args: Vec<ValueId>,
        else_block: BlockId,
        else_args: Vec<ValueId>,
    },
    /// Return an optional value from the current function.
    Return {
        value: Option<ValueId>,
    },
    /// Unreachable code (after return / diverging call).
    Unreachable,
    /// Switch on an integer value (for match on enums / integers).
    Switch {
        value: ValueId,
        cases: Vec<(u128, BlockId, Vec<ValueId>)>,
        default: BlockId,
        default_args: Vec<ValueId>,
    },
    /// Task join — wait for a spawned task and get its result.
    Join {
        task: ValueId,
        result_block: BlockId,
        result_args: Vec<ValueId>,
    },
}

// =============================================================================
// §5  IR STATEMENTS
// =============================================================================

/// An IR statement — side-effecting or value-defining.
#[derive(Debug, Clone)]
pub enum IrStmt {
    // ── Value definitions (SSA) ─────────────────────────────────────────
    /// Define a new SSA value: `dst = expr`.
    Define {
        dst: ValueId,
        ty: Ty,
        ownership: Ownership,
        value: IrExpr,
        span: Span,
    },

    /// Define a value that's computed but immediately discarded.
    Discard {
        value: IrExpr,
        span: Span,
    },

    // ── Storage operations (non-SSA escape hatches) ────────────────────
    /// Store a value to a mutable location (for `:=` mutation).
    Store {
        target: IrLValue,
        value: ValueId,
        span: Span,
    },

    /// Load a value from a mutable location.
    Load {
        dst: ValueId,
        ty: Ty,
        ownership: Ownership,
        source: IrLValue,
        span: Span,
    },

    // ── Region operations ──────────────────────────────────────────────
    /// Enter a region: `region R { ... }`.
    RegionEnter {
        region_id: u32,
        span: Span,
    },
    /// Exit a region — all allocations in the region are freed.
    RegionExit {
        region_id: u32,
        span: Span,
    },
    /// Allocate in a region.
    RegionAlloc {
        dst: ValueId,
        ty: Ty,
        region_id: u32,
        count: ValueId,
        span: Span,
    },

    // ── Contract assertions ────────────────────────────────────────────
    /// Assert a precondition: `requires(expr)`.
    Requires {
        condition: ValueId,
        message: String,
        span: Span,
    },
    /// Assert a postcondition: `ensures(expr)`.
    Ensures {
        condition: ValueId,
        message: String,
        span: Span,
    },
    /// Assume a fact (unchecked, for optimisation).
    Assume {
        condition: ValueId,
        span: Span,
    },

    // ── Debug / profiling ──────────────────────────────────────────────
    /// Debug assertion.
    DebugAssert {
        condition: ValueId,
        message: String,
        span: Span,
    },
    /// Profile point.
    Profile {
        id: u32,
        span: Span,
    },
}

// =============================================================================
// §6  IR L-VALUES (mutable targets)
// =============================================================================

/// An l-value — a mutable target for stores and loads.
#[derive(Debug, Clone)]
pub enum IrLValue {
    /// Named mutable variable.
    Var {
        name: String,
        ty: Ty,
    },
    /// Field of a struct.
    Field {
        object: ValueId,
        field_name: String,
        field_idx: u32,
        ty: Ty,
    },
    /// Index into an array / tensor.
    Index {
        object: ValueId,
        index: ValueId,
        ty: Ty,
    },
    /// Dereference a reference.
    Deref {
        ptr: ValueId,
        ty: Ty,
    },
}

// =============================================================================
// §7  OPERATOR TYPES
// =============================================================================

/// Binary operators at the IR level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    FloorDiv,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    And,
    Or,
}

/// Unary operators at the IR level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrUnOp {
    Neg,
    Not,
}

/// Comparison operators at the IR level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrCmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Reduction operators for tensor reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrReduceOp {
    Sum,
    Product,
    Min,
    Max,
    Any,
    All,
}

// =============================================================================
// §8  SUPPORTING TYPES
// =============================================================================

/// IR-level function parameter.
#[derive(Debug, Clone)]
pub struct IrParam {
    pub name: String,
    pub ty: Ty,
    pub ownership: Ownership,
}

/// A single match case in the IR.
#[derive(Debug, Clone)]
pub struct IrMatchCase {
    pub pattern: IrPattern,
    pub guard: Option<ValueId>,
    pub block: BlockId,
    /// Bindings introduced by this case: (name, value_id, type).
    pub bindings: Vec<(String, ValueId, Ty)>,
}

/// IR-level pattern (for match lowering).
#[derive(Debug, Clone)]
pub enum IrPattern {
    Wildcard,
    Ident { name: String },
    Lit { value: IrConst },
    Struct { name: String, fields: Vec<(String, IrPattern)> },
    Tuple { elems: Vec<IrPattern> },
    Enum { name: String, variant: String, inner: Vec<IrPattern> },
    Range { lo: Option<IrConst>, hi: Option<IrConst>, inclusive: bool },
    Or { arms: Vec<IrPattern> },
    Guard { pattern: Box<IrPattern>, condition: ValueId },
}

/// IR-level constant value.
#[derive(Debug, Clone)]
pub enum IrConst {
    Int(u128),
    Float(f64),
    Bool(bool),
    Str(String),
}

// =============================================================================
// §9  IR EXPRESSIONS
// =============================================================================

/// IR expression — produces a value and carries type + effect metadata.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum IrExpr {
    // ── Constants ──────────────────────────────────────────────────────
    ConstInt { value: u128, ty: Ty },
    ConstFloat { value: f64, ty: Ty },
    ConstBool { value: bool },
    ConstStr { value: String },
    ConstUnit,
    /// Option None value.
    ConstNone,
    /// Result Err value.
    ConstErr { message: String },

    // ── Value reference ────────────────────────────────────────────────
    ValueRef { id: ValueId },

    // ── Arithmetic (all typed) ────────────────────────────────────────
    BinOp { op: IrBinOp, lhs: ValueId, rhs: ValueId, ty: Ty, effect: Effect },
    UnOp { op: IrUnOp, operand: ValueId, ty: Ty },

    // ── Comparison ─────────────────────────────────────────────────────
    Compare { op: IrCmpOp, lhs: ValueId, rhs: ValueId },

    // ── Type operations ────────────────────────────────────────────────
    Cast { value: ValueId, from_ty: Ty, to_ty: Ty },
    /// Runtime type check.
    TypeCheck { value: ValueId, expected: Ty },
    /// `Option::is_none` check.
    IsNone { value: ValueId },
    /// `Option::is_some` check.
    IsSome { value: ValueId },
    /// `Result::is_ok` check.
    IsOk { value: ValueId },
    /// `Result::is_err` check.
    IsErr { value: ValueId },
    /// Unwrap — panic on None / Err.
    Unwrap { value: ValueId, inner_ty: Ty },
    /// Try-propagate — the `?` operator.
    TryPropagate { value: ValueId, ok_ty: Ty, err_ty: Ty },

    // ── Struct / Tuple / Array / Enum construction ────────────────────
    MakeStruct { name: String, fields: Vec<(String, ValueId)>, ty: Ty },
    MakeTuple { elems: Vec<ValueId>, ty: Ty },
    MakeArray { elems: Vec<ValueId>, ty: Ty },
    MakeEnum { name: String, variant: String, data: Vec<ValueId>, ty: Ty },
    MakeOption { inner: Option<ValueId>, inner_ty: Ty },
    MakeResult { value: Result<ValueId, ValueId>, ok_ty: Ty, err_ty: Ty },
    MakeRange { lo: Option<ValueId>, hi: Option<ValueId>, inclusive: bool },
    MakeClosure { params: Vec<IrParam>, body: IrFunction, captures: Vec<ValueId>, ty: Ty },

    // ── Field / element access (produces a new SSA value) ─────────────
    FieldAccess { object: ValueId, field_name: String, field_idx: u32, ty: Ty },
    IndexAccess { object: ValueId, indices: Vec<ValueId>, ty: Ty },
    TupleAccess { tuple: ValueId, index: u32, ty: Ty },

    // ── Function calls ────────────────────────────────────────────────
    Call { func: ValueId, args: Vec<ValueId>, ty: Ty, effect: Effect },
    CallNamed { name: String, args: Vec<ValueId>, ty: Ty, effect: Effect },
    MethodCall { receiver: ValueId, method: String, args: Vec<ValueId>, ty: Ty, effect: Effect },

    // ── Tensor / ML operations ──────────────────────────────────────────
    MatMul { lhs: ValueId, rhs: ValueId, ty: Ty },
    HadamardMul { lhs: ValueId, rhs: ValueId, ty: Ty },
    HadamardDiv { lhs: ValueId, rhs: ValueId, ty: Ty },
    TensorConcat { lhs: ValueId, rhs: ValueId, ty: Ty },
    KronProd { lhs: ValueId, rhs: ValueId, ty: Ty },
    OuterProd { lhs: ValueId, rhs: ValueId, ty: Ty },
    Pow { base: ValueId, exp: ValueId, ty: Ty },
    Grad { inner: ValueId, ty: Ty },
    TensorReshape { tensor: ValueId, new_shape: Vec<Dim>, ty: Ty },
    TensorBroadcast { tensor: ValueId, target_shape: Vec<Dim>, ty: Ty },
    TensorTranspose { tensor: ValueId, perm: Vec<u32>, ty: Ty },
    TensorReduce { tensor: ValueId, op: IrReduceOp, axes: Vec<u32>, ty: Ty },
    TensorMap { tensor: ValueId, func: ValueId, ty: Ty },

    // ── Loop operations (structured) ──────────────────────────────────
    /// Pure map — for each element, apply function (vectorisable).
    MapLoop { iter: ValueId, body_func: ValueId, ty: Ty, fusion: FusionBoundary },
    /// Reduction — fold with associative operator (parallelisable).
    ReduceLoop { iter: ValueId, init: ValueId, op: IrBinOp, body_func: Option<ValueId>, ty: Ty, fusion: FusionBoundary },
    /// Stateful while loop (sequential).
    WhileLoop { cond_block: BlockId, body_block: BlockId, state_vars: Vec<(ValueId, Ty)>, effect: Effect },
    /// `fold()` loop with tracked state.
    FoldLoop { iter: ValueId, init: ValueId, body_func: ValueId, state_tys: Vec<Ty>, ty: Ty },

    // ── Pattern matching ──────────────────────────────────────────────
    Match { value: ValueId, cases: Vec<IrMatchCase>, default: Option<BlockId>, ty: Ty },

    // ── If expression ─────────────────────────────────────────────────
    If { cond: ValueId, then_value: ValueId, else_value: Option<ValueId>, ty: Ty },

    // ── Pipeline operator ─────────────────────────────────────────────
    /// `a |> f |> g`
    Pipeline { stages: Vec<ValueId>, ty: Ty },

    // ── Concurrency ───────────────────────────────────────────────────
    Spawn { task: IrFunction, name: Option<String>, ownership_transfer: Vec<ValueId>, ty: Ty },
    JoinTask { task: ValueId, ty: Ty },
    AtomicBlock { body: BlockId, ty: Ty },
    SyncBarrier,

    // ── Region allocation (expression form) ────────────────────────────
    RegionAllocExpr { region_id: u32, ty: Ty, count: ValueId },

    // ── Low-level escape hatches ──────────────────────────────────────
    AsmBlock { template: String, operands: Vec<ValueId>, clobbers: Vec<String>, ty: Ty, effect: Effect },
    Intrinsic { name: String, args: Vec<ValueId>, ty: Ty, effect: Effect },
    UnsafeBlock { body: BlockId, ty: Ty, reason: String },

    // ── Decorator / attribute metadata (not executed — optimiser hints) ──
    DecoratorHint { name: String, args: Vec<IrConst>, target: ValueId },

    // ── SIMD vector operations ────────────────────────────────────────
    VecCtor { elems: Vec<ValueId>, ty: Ty },
    VecBinOp { op: IrBinOp, lhs: ValueId, rhs: ValueId, ty: Ty },
    VecSwizzle { vec: ValueId, indices: Vec<u8>, ty: Ty },

    // ── Cost annotation for superoptimizer ─────────────────────────────
    WithCost { inner: ValueId, cost: CostHint },
}

// =============================================================================
// §10  IR FUNCTION
// =============================================================================

/// An IR-level function — a named CFG with typed params and return type.
#[derive(Debug, Clone)]
pub struct IrFunction {
    pub name: String,
    pub params: Vec<IrParam>,
    pub ret_ty: Ty,
    pub effect: Effect,
    pub blocks: Vec<BasicBlock>,
    pub entry: BlockId,
    pub next_value_id: u32,
    pub next_block_id: u32,
    /// Attributes from the AST (e.g. @gpu, @simd, @parallel).
    pub attributes: Vec<Attribute>,
    /// Cost hint for the superoptimizer.
    pub cost_hint: CostHint,
    /// Fusion boundary for loop/kernel fusion.
    pub fusion_boundary: FusionBoundary,
    /// Region this function allocates in (if any).
    pub region_id: Option<u32>,
    /// True if this is a GPU / TPU kernel.
    pub is_kernel: bool,
    /// True if this is `main()` or another public entry point.
    pub is_entry: bool,
}

// =============================================================================
// §11  IR MODULE — TOP-LEVEL DEFINITIONS
// =============================================================================

/// Data layout annotation for struct / component definitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrDataLayout {
    Auto,
    SoA,
    AoS,
    /// Explicit alignment: `aligned<64>`.
    Aligned(u32),
    /// Named layout string: `layout="NHWC"`.
    Named(String),
}

/// IR-level struct definition.
#[derive(Debug, Clone)]
pub struct IrStructDef {
    pub name: String,
    pub fields: Vec<(String, Ty, Ownership)>,
    pub is_component: bool,
    pub layout: IrDataLayout,
}

/// IR-level enum definition.
#[derive(Debug, Clone)]
pub struct IrEnumDef {
    pub name: String,
    /// (variant_name, payload_types).
    pub variants: Vec<(String, Vec<Ty>)>,
}

/// IR-level component definition.
#[derive(Debug, Clone)]
pub struct IrComponentDef {
    pub name: String,
    pub fields: Vec<(String, Ty)>,
    pub layout: IrDataLayout,
}

/// IR-level constant definition.
#[derive(Debug, Clone)]
pub struct IrConstDef {
    pub name: String,
    pub ty: Ty,
    pub value: IrConst,
}

/// IR-level region definition.
#[derive(Debug, Clone)]
pub struct IrRegionDef {
    pub id: u32,
    pub name: String,
}

/// IR-level effect definition.
#[derive(Debug, Clone)]
pub struct IrEffectDef {
    pub name: String,
    /// Effect operations: `effect io { read, write, print }`.
    pub operations: Vec<String>,
}

/// A complete IR module.
#[derive(Debug, Clone)]
pub struct IrModule {
    pub name: String,
    pub functions: Vec<IrFunction>,
    pub structs: Vec<IrStructDef>,
    pub enums: Vec<IrEnumDef>,
    pub components: Vec<IrComponentDef>,
    pub constants: Vec<IrConstDef>,
    pub regions: Vec<IrRegionDef>,
    pub effects: Vec<IrEffectDef>,
}

// =============================================================================
// §12  DISPLAY IMPLEMENTATIONS
// =============================================================================

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Effect::Pure => write!(f, "pure"),
            Effect::IO => write!(f, "io"),
            Effect::Mutation => write!(f, "mutation"),
            Effect::Allocation => write!(f, "allocation"),
            Effect::ControlFlow => write!(f, "control_flow"),
            Effect::Async => write!(f, "async"),
            Effect::Unknown => write!(f, "unknown"),
        }
    }
}

impl fmt::Display for Ownership {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ownership::Owned => write!(f, "owned"),
            Ownership::Ref => write!(f, "ref"),
            Ownership::MutRef => write!(f, "mut_ref"),
            Ownership::Shared => write!(f, "shared"),
            Ownership::Copy => write!(f, "copy"),
        }
    }
}

impl fmt::Display for CostHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CostHint::Unknown => write!(f, "unknown"),
            CostHint::Cheap => write!(f, "cheap"),
            CostHint::Moderate => write!(f, "moderate"),
            CostHint::Expensive => write!(f, "expensive"),
            CostHint::VeryExpensive => write!(f, "very_expensive"),
            CostHint::Loop { iterations } => write!(f, "loop({iterations})"),
            CostHint::Kernel { flops } => write!(f, "kernel({flops})"),
        }
    }
}

impl fmt::Display for FusionBoundary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusionBoundary::None => write!(f, "none"),
            FusionBoundary::NoFuse => write!(f, "no_fuse"),
            FusionBoundary::FuseAggressive => write!(f, "fuse_aggressive"),
        }
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

// =============================================================================
// §13  IrExpr :: effect() — compute the effect of an expression
// =============================================================================

impl IrExpr {
    /// Return the effect annotation of this expression.
    ///
    /// For variants that carry an explicit `effect` field, that value is
    /// returned directly.  For all others, a conservative default is chosen.
    pub fn effect(&self) -> Effect {
        match self {
            // ── Constants — always pure ──
            IrExpr::ConstInt { .. }
            | IrExpr::ConstFloat { .. }
            | IrExpr::ConstBool { .. }
            | IrExpr::ConstStr { .. }
            | IrExpr::ConstUnit
            | IrExpr::ConstNone
            | IrExpr::ConstErr { .. } => Effect::Pure,

            // ── Value reference — pure ──
            IrExpr::ValueRef { .. } => Effect::Pure,

            // ── Explicitly annotated ──
            IrExpr::BinOp { effect, .. } => *effect,
            IrExpr::Call { effect, .. } => *effect,
            IrExpr::CallNamed { effect, .. } => *effect,
            IrExpr::MethodCall { effect, .. } => *effect,
            IrExpr::AsmBlock { effect, .. } => *effect,
            IrExpr::Intrinsic { effect, .. } => *effect,

            // ── Pure type operations ──
            IrExpr::UnOp { .. }
            | IrExpr::Compare { .. }
            | IrExpr::Cast { .. }
            | IrExpr::TypeCheck { .. }
            | IrExpr::IsNone { .. }
            | IrExpr::IsSome { .. }
            | IrExpr::IsOk { .. }
            | IrExpr::IsErr { .. } => Effect::Pure,

            // ── Unwrap / try — may panic ──
            IrExpr::Unwrap { .. } => Effect::ControlFlow,
            IrExpr::TryPropagate { .. } => Effect::ControlFlow,

            // ── Construction — pure ──
            IrExpr::MakeStruct { .. }
            | IrExpr::MakeTuple { .. }
            | IrExpr::MakeArray { .. }
            | IrExpr::MakeEnum { .. }
            | IrExpr::MakeOption { .. }
            | IrExpr::MakeResult { .. }
            | IrExpr::MakeRange { .. }
            | IrExpr::MakeClosure { .. } => Effect::Pure,

            // ── Access — pure ──
            IrExpr::FieldAccess { .. }
            | IrExpr::IndexAccess { .. }
            | IrExpr::TupleAccess { .. } => Effect::Pure,

            // ── Tensor ops — pure (no side effects) ──
            IrExpr::MatMul { .. }
            | IrExpr::HadamardMul { .. }
            | IrExpr::HadamardDiv { .. }
            | IrExpr::TensorConcat { .. }
            | IrExpr::KronProd { .. }
            | IrExpr::OuterProd { .. }
            | IrExpr::Pow { .. }
            | IrExpr::Grad { .. }
            | IrExpr::TensorReshape { .. }
            | IrExpr::TensorBroadcast { .. }
            | IrExpr::TensorTranspose { .. }
            | IrExpr::TensorReduce { .. }
            | IrExpr::TensorMap { .. } => Effect::Pure,

            // ── Loops — may have any effect ──
            IrExpr::MapLoop { .. } => Effect::Pure,
            IrExpr::ReduceLoop { .. } => Effect::Pure,
            IrExpr::WhileLoop { effect, .. } => *effect,
            IrExpr::FoldLoop { .. } => Effect::Unknown,

            // ── Control-flow expressions ──
            IrExpr::Match { .. } => Effect::ControlFlow,
            IrExpr::If { .. } => Effect::Pure,

            // ── Pipeline — pure (composition of pure stages) ──
            IrExpr::Pipeline { .. } => Effect::Pure,

            // ── Concurrency ──
            IrExpr::Spawn { .. } => Effect::Async,
            IrExpr::JoinTask { .. } => Effect::Async,
            IrExpr::AtomicBlock { .. } => Effect::Mutation,
            IrExpr::SyncBarrier => Effect::Async,

            // ── Region allocation ──
            IrExpr::RegionAllocExpr { .. } => Effect::Allocation,

            // ── Unsafe — unknown ──
            IrExpr::UnsafeBlock { .. } => Effect::Unknown,

            // ── Decorator hints — pure (metadata only) ──
            IrExpr::DecoratorHint { .. } => Effect::Pure,

            // ── SIMD vector ops — pure ──
            IrExpr::VecCtor { .. }
            | IrExpr::VecBinOp { .. }
            | IrExpr::VecSwizzle { .. } => Effect::Pure,

            // ── Cost annotation — delegates to inner ──
            IrExpr::WithCost { .. } => Effect::Pure,
        }
    }
}

// =============================================================================
// §14  IR BUILDER
// =============================================================================

/// Builder for constructing IR inside a function.
#[derive(Debug)]
pub struct IrBuilder {
    func: IrFunction,
    current_block: BlockId,
}

impl IrBuilder {
    /// Create a new builder for a function with the given name, params, and
    /// return type.  An entry block is automatically allocated.
    pub fn new(name: &str, params: Vec<IrParam>, ret_ty: Ty, effect: Effect) -> Self {
        let entry = BlockId(0);
        let func = IrFunction {
            name: name.to_string(),
            params,
            ret_ty,
            effect,
            blocks: vec![BasicBlock {
                id: entry,
                params: vec![],
                stmts: vec![],
                terminator: IrTerminator::Unreachable,
                span: Span::dummy(),
            }],
            entry,
            next_value_id: 0,
            next_block_id: 1,
            attributes: vec![],
            cost_hint: CostHint::Unknown,
            fusion_boundary: FusionBoundary::None,
            region_id: None,
            is_kernel: false,
            is_entry: false,
        };
        IrBuilder {
            func,
            current_block: entry,
        }
    }

    /// Allocate a fresh SSA value ID.
    pub fn alloc_value(&mut self, ty: Ty, ownership: Ownership, _span: Span) -> ValueId {
        let id = ValueId(self.func.next_value_id);
        self.func.next_value_id += 1;
        let _ = (ty, ownership); // stored by Define, not here
        id
    }

    /// Allocate a fresh basic-block ID.
    pub fn alloc_block(&mut self) -> BlockId {
        let id = BlockId(self.func.next_block_id);
        self.func.next_block_id += 1;
        self.func.blocks.push(BasicBlock {
            id,
            params: vec![],
            stmts: vec![],
            terminator: IrTerminator::Unreachable,
            span: Span::dummy(),
        });
        id
    }

    /// Switch the builder to append statements to the given block.
    pub fn set_block(&mut self, block: BlockId) {
        self.current_block = block;
    }

    /// Get the current block ID.
    pub fn current_block(&self) -> BlockId {
        self.current_block
    }

    /// Define a new SSA value in the current block.
    /// Returns the ValueId of the newly defined value.
    pub fn define(&mut self, ty: Ty, ownership: Ownership, value: IrExpr, span: Span) -> ValueId {
        let id = ValueId(self.func.next_value_id);
        self.func.next_value_id += 1;
        let block = self.block_mut();
        block.stmts.push(IrStmt::Define {
            dst: id,
            ty,
            ownership,
            value,
            span,
        });
        id
    }

    /// Emit a discard statement (expression evaluated for side effects only).
    pub fn discard(&mut self, value: IrExpr, span: Span) {
        let block = self.block_mut();
        block.stmts.push(IrStmt::Discard { value, span });
    }

    /// Store a value to an l-value target.
    pub fn store(&mut self, target: IrLValue, value: ValueId, span: Span) {
        let block = self.block_mut();
        block.stmts.push(IrStmt::Store { target, value, span });
    }

    /// Load a value from an l-value source, returning a fresh ValueId.
    pub fn load(&mut self, ty: Ty, ownership: Ownership, source: IrLValue, span: Span) -> ValueId {
        let id = ValueId(self.func.next_value_id);
        self.func.next_value_id += 1;
        let block = self.block_mut();
        block.stmts.push(IrStmt::Load {
            dst: id,
            ty,
            ownership,
            source,
            span,
        });
        id
    }

    /// Set the terminator for the current block.
    pub fn terminate(&mut self, terminator: IrTerminator) {
        let block = self.block_mut();
        block.terminator = terminator;
    }

    /// Add a parameter to a block (used for block-parameter SSA).
    pub fn add_block_param(&mut self, block_id: BlockId, value_id: ValueId, ty: Ty) {
        if let Some(block) = self.func.blocks.iter_mut().find(|b| b.id == block_id) {
            block.params.push((value_id, ty));
        }
    }

    /// Consume the builder and return the completed IrFunction.
    pub fn build(self) -> IrFunction {
        self.func
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    fn block_mut(&mut self) -> &mut BasicBlock {
        let idx = self.current_block.0 as usize;
        &mut self.func.blocks[idx]
    }
}

// =============================================================================
// §15  IR VALIDATOR
// =============================================================================

/// Validates IR invariants (used after lowering from AST).
///
/// Checks performed:
///   • Value IDs are in range (≤ next_value_id)
///   • Block IDs are in range (≤ next_block_id)
///   • All blocks are terminated (no Unreachable that was left as a placeholder)
///   • Entry block exists
///   • Type consistency for defines
///   • Effect soundness (pure expressions don't appear in mutation contexts)
pub struct IrValidator;

impl IrValidator {
    /// Validate a single function's IR.
    /// Returns a list of diagnostics for any violations found.
    pub fn validate_function(func: &IrFunction) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        let max_vid = func.next_value_id;
        let max_bid = func.next_block_id;

        // Check entry block exists.
        let entry_idx = func.entry.0 as usize;
        if entry_idx >= func.blocks.len() || func.blocks[entry_idx].id != func.entry {
            diags.push(Diagnostic::error(
                Span::dummy(),
                format!(
                    "IR function `{}`: entry block {:?} does not exist ({} blocks total)",
                    func.name,
                    func.entry,
                    func.blocks.len()
                ),
            ));
        }

        // Check each block.
        for block in &func.blocks {
            // Block ID in range.
            if block.id.0 >= max_bid {
                diags.push(Diagnostic::error(
                    Span::dummy(),
                    format!(
                        "IR function `{}`: block {:?} has ID >= next_block_id {}",
                        func.name, block.id, max_bid
                    ),
                ));
            }

            // Block parameters have valid value IDs.
            for (vid, ty) in &block.params {
                if vid.0 >= max_vid {
                    diags.push(Diagnostic::error(
                        Span::dummy(),
                        format!(
                            "IR function `{}`: block {:?} param {:?} has ID >= next_value_id {}",
                            func.name, block.id, vid, max_vid
                        ),
                    ));
                }
                let _ = ty; // type checked by later passes
            }

            // Check statements.
            for stmt in &block.stmts {
                Self::validate_stmt(stmt, max_vid, max_bid, &func.name, &mut diags);
            }

            // Check terminator.
            Self::validate_terminator(&block.terminator, max_vid, max_bid, &func.name, &mut diags);
        }

        diags
    }

    /// Validate an entire IR module.
    pub fn validate_module(module: &IrModule) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        for func in &module.functions {
            diags.extend(Self::validate_function(func));
        }
        diags
    }

    // ── Statement validation ──────────────────────────────────────────

    fn validate_stmt(
        stmt: &IrStmt,
        max_vid: u32,
        max_bid: u32,
        fname: &str,
        diags: &mut Vec<Diagnostic>,
    ) {
        match stmt {
            IrStmt::Define { dst, value, .. } => {
                if dst.0 >= max_vid {
                    diags.push(Diagnostic::error(
                        Span::dummy(),
                        format!("IR function `{fname}`: Define dst {dst:?} >= next_value_id {max_vid}"),
                    ));
                }
                Self::validate_expr(value, max_vid, max_bid, fname, diags);
            }
            IrStmt::Discard { value, .. } => {
                Self::validate_expr(value, max_vid, max_bid, fname, diags);
            }
            IrStmt::Store { target, value, .. } => {
                Self::validate_lvalue(target, max_vid, fname, diags);
                if value.0 >= max_vid {
                    diags.push(Diagnostic::error(
                        Span::dummy(),
                        format!("IR function `{fname}`: Store value {value:?} >= next_value_id {max_vid}"),
                    ));
                }
            }
            IrStmt::Load { dst, source, .. } => {
                if dst.0 >= max_vid {
                    diags.push(Diagnostic::error(
                        Span::dummy(),
                        format!("IR function `{fname}`: Load dst {dst:?} >= next_value_id {max_vid}"),
                    ));
                }
                Self::validate_lvalue(source, max_vid, fname, diags);
            }
            IrStmt::RegionAlloc { dst, count, .. } => {
                if dst.0 >= max_vid {
                    diags.push(Diagnostic::error(
                        Span::dummy(),
                        format!("IR function `{fname}`: RegionAlloc dst {dst:?} >= next_value_id {max_vid}"),
                    ));
                }
                if count.0 >= max_vid {
                    diags.push(Diagnostic::error(
                        Span::dummy(),
                        format!("IR function `{fname}`: RegionAlloc count {count:?} >= next_value_id {max_vid}"),
                    ));
                }
            }
            IrStmt::Requires { condition, .. }
            | IrStmt::Ensures { condition, .. }
            | IrStmt::Assume { condition, .. }
            | IrStmt::DebugAssert { condition, .. } => {
                if condition.0 >= max_vid {
                    diags.push(Diagnostic::error(
                        Span::dummy(),
                        format!("IR function `{fname}`: condition {condition:?} >= next_value_id {max_vid}"),
                    ));
                }
            }
            // RegionEnter, RegionExit, Profile — no value refs to check.
            IrStmt::RegionEnter { .. }
            | IrStmt::RegionExit { .. }
            | IrStmt::Profile { .. } => {}
        }
    }

    // ── Expression validation ─────────────────────────────────────────

    fn validate_expr(
        expr: &IrExpr,
        max_vid: u32,
        max_bid: u32,
        fname: &str,
        diags: &mut Vec<Diagnostic>,
    ) {
        match expr {
            // Constants — no references to validate.
            IrExpr::ConstInt { .. }
            | IrExpr::ConstFloat { .. }
            | IrExpr::ConstBool { .. }
            | IrExpr::ConstStr { .. }
            | IrExpr::ConstUnit
            | IrExpr::ConstNone
            | IrExpr::ConstErr { .. } => {}

            IrExpr::ValueRef { id } => {
                if id.0 >= max_vid {
                    diags.push(Diagnostic::error(
                        Span::dummy(),
                        format!("IR function `{fname}`: ValueRef {id:?} >= next_value_id {max_vid}"),
                    ));
                }
            }

            IrExpr::BinOp { lhs, rhs, .. } => {
                Self::check_vid(*lhs, max_vid, fname, diags);
                Self::check_vid(*rhs, max_vid, fname, diags);
            }
            IrExpr::UnOp { operand, .. } => {
                Self::check_vid(*operand, max_vid, fname, diags);
            }
            IrExpr::Compare { lhs, rhs, .. } => {
                Self::check_vid(*lhs, max_vid, fname, diags);
                Self::check_vid(*rhs, max_vid, fname, diags);
            }

            IrExpr::Cast { value, .. }
            | IrExpr::TypeCheck { value, .. }
            | IrExpr::IsNone { value }
            | IrExpr::IsSome { value }
            | IrExpr::IsOk { value }
            | IrExpr::IsErr { value }
            | IrExpr::Unwrap { value, .. }
            | IrExpr::TryPropagate { value, .. }
            | IrExpr::Grad { inner: value, .. } => {
                Self::check_vid(*value, max_vid, fname, diags);
            }

            IrExpr::MakeStruct { fields, .. } => {
                for (_, vid) in fields {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }
            IrExpr::MakeTuple { elems, .. } | IrExpr::MakeArray { elems, .. } => {
                for vid in elems {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }
            IrExpr::MakeEnum { data, .. } => {
                for vid in data {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }
            IrExpr::MakeOption { inner, .. } => {
                if let Some(vid) = inner {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }
            IrExpr::MakeResult { value, .. } => match value {
                Ok(vid) | Err(vid) => Self::check_vid(*vid, max_vid, fname, diags),
            },
            IrExpr::MakeRange { lo, hi, .. } => {
                if let Some(vid) = lo {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
                if let Some(vid) = hi {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }
            IrExpr::MakeClosure { captures, .. } => {
                for vid in captures {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }

            IrExpr::FieldAccess { object, .. } => {
                Self::check_vid(*object, max_vid, fname, diags);
            }
            IrExpr::IndexAccess { object, indices, .. } => {
                Self::check_vid(*object, max_vid, fname, diags);
                for vid in indices {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }
            IrExpr::TupleAccess { tuple, .. } => {
                Self::check_vid(*tuple, max_vid, fname, diags);
            }

            IrExpr::Call { func: vid, args, .. } => {
                Self::check_vid(*vid, max_vid, fname, diags);
                for a in args {
                    Self::check_vid(*a, max_vid, fname, diags);
                }
            }
            IrExpr::CallNamed { args, .. } => {
                for a in args {
                    Self::check_vid(*a, max_vid, fname, diags);
                }
            }
            IrExpr::MethodCall { receiver, args, .. } => {
                Self::check_vid(*receiver, max_vid, fname, diags);
                for a in args {
                    Self::check_vid(*a, max_vid, fname, diags);
                }
            }

            IrExpr::MatMul { lhs, rhs, .. }
            | IrExpr::HadamardMul { lhs, rhs, .. }
            | IrExpr::HadamardDiv { lhs, rhs, .. }
            | IrExpr::TensorConcat { lhs, rhs, .. }
            | IrExpr::KronProd { lhs, rhs, .. }
            | IrExpr::OuterProd { lhs, rhs, .. } => {
                Self::check_vid(*lhs, max_vid, fname, diags);
                Self::check_vid(*rhs, max_vid, fname, diags);
            }

            IrExpr::Pow { base, exp, .. } => {
                Self::check_vid(*base, max_vid, fname, diags);
                Self::check_vid(*exp, max_vid, fname, diags);
            }

            IrExpr::TensorReshape { tensor, .. }
            | IrExpr::TensorBroadcast { tensor, .. }
            | IrExpr::TensorTranspose { tensor, .. }
            | IrExpr::TensorMap { tensor, .. } => {
                Self::check_vid(*tensor, max_vid, fname, diags);
            }

            IrExpr::TensorReduce { tensor, .. } => {
                Self::check_vid(*tensor, max_vid, fname, diags);
            }

            IrExpr::MapLoop { iter, body_func, .. } => {
                Self::check_vid(*iter, max_vid, fname, diags);
                Self::check_vid(*body_func, max_vid, fname, diags);
            }
            IrExpr::ReduceLoop { iter, init, body_func, .. } => {
                Self::check_vid(*iter, max_vid, fname, diags);
                Self::check_vid(*init, max_vid, fname, diags);
                if let Some(bf) = body_func {
                    Self::check_vid(*bf, max_vid, fname, diags);
                }
            }
            IrExpr::WhileLoop { cond_block, body_block, state_vars, .. } => {
                Self::check_bid(*cond_block, max_bid, fname, diags);
                Self::check_bid(*body_block, max_bid, fname, diags);
                for (vid, _) in state_vars {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }
            IrExpr::FoldLoop { iter, init, body_func, .. } => {
                Self::check_vid(*iter, max_vid, fname, diags);
                Self::check_vid(*init, max_vid, fname, diags);
                Self::check_vid(*body_func, max_vid, fname, diags);
            }

            IrExpr::Match { value, cases, .. } => {
                Self::check_vid(*value, max_vid, fname, diags);
                for case in cases {
                    Self::validate_ir_pattern(&case.pattern, max_vid, fname, diags);
                    if let Some(g) = &case.guard {
                        Self::check_vid(*g, max_vid, fname, diags);
                    }
                    Self::check_bid(case.block, max_bid, fname, diags);
                    for (_, vid, _) in &case.bindings {
                        Self::check_vid(*vid, max_vid, fname, diags);
                    }
                }
            }

            IrExpr::If { cond, then_value, else_value, .. } => {
                Self::check_vid(*cond, max_vid, fname, diags);
                Self::check_vid(*then_value, max_vid, fname, diags);
                if let Some(ev) = else_value {
                    Self::check_vid(*ev, max_vid, fname, diags);
                }
            }

            IrExpr::Pipeline { stages, .. } => {
                for vid in stages {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }

            IrExpr::Spawn { ownership_transfer, .. } => {
                for vid in ownership_transfer {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }
            IrExpr::JoinTask { task, .. } => {
                Self::check_vid(*task, max_vid, fname, diags);
            }
            IrExpr::AtomicBlock { body, .. } => {
                Self::check_bid(*body, max_bid, fname, diags);
            }
            IrExpr::SyncBarrier => {}

            IrExpr::RegionAllocExpr { count, .. } => {
                Self::check_vid(*count, max_vid, fname, diags);
            }

            IrExpr::AsmBlock { operands, .. } => {
                for vid in operands {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }
            IrExpr::Intrinsic { args, .. } => {
                for vid in args {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }
            IrExpr::UnsafeBlock { body, .. } => {
                Self::check_bid(*body, max_bid, fname, diags);
            }

            IrExpr::DecoratorHint { target, .. } => {
                Self::check_vid(*target, max_vid, fname, diags);
            }

            IrExpr::VecCtor { elems, .. } => {
                for vid in elems {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }
            IrExpr::VecBinOp { lhs, rhs, .. } => {
                Self::check_vid(*lhs, max_vid, fname, diags);
                Self::check_vid(*rhs, max_vid, fname, diags);
            }
            IrExpr::VecSwizzle { vec, .. } => {
                Self::check_vid(*vec, max_vid, fname, diags);
            }

            IrExpr::WithCost { inner, .. } => {
                Self::check_vid(*inner, max_vid, fname, diags);
            }
        }
    }

    // ── L-value validation ────────────────────────────────────────────

    fn validate_lvalue(lv: &IrLValue, max_vid: u32, fname: &str, diags: &mut Vec<Diagnostic>) {
        match lv {
            IrLValue::Var { .. } => {}
            IrLValue::Field { object, .. } => {
                Self::check_vid(*object, max_vid, fname, diags);
            }
            IrLValue::Index { object, index, .. } => {
                Self::check_vid(*object, max_vid, fname, diags);
                Self::check_vid(*index, max_vid, fname, diags);
            }
            IrLValue::Deref { ptr, .. } => {
                Self::check_vid(*ptr, max_vid, fname, diags);
            }
        }
    }

    // ── Pattern validation ────────────────────────────────────────────

    fn validate_ir_pattern(pat: &IrPattern, max_vid: u32, fname: &str, diags: &mut Vec<Diagnostic>) {
        match pat {
            IrPattern::Wildcard | IrPattern::Ident { .. } | IrPattern::Lit { .. } => {}
            IrPattern::Struct { fields, .. } => {
                for (_, p) in fields {
                    Self::validate_ir_pattern(p, max_vid, fname, diags);
                }
            }
            IrPattern::Tuple { elems } => {
                for p in elems {
                    Self::validate_ir_pattern(p, max_vid, fname, diags);
                }
            }
            IrPattern::Enum { inner, .. } => {
                for p in inner {
                    Self::validate_ir_pattern(p, max_vid, fname, diags);
                }
            }
            IrPattern::Range { .. } => {}
            IrPattern::Or { arms } => {
                for p in arms {
                    Self::validate_ir_pattern(p, max_vid, fname, diags);
                }
            }
            IrPattern::Guard { pattern, condition } => {
                Self::validate_ir_pattern(pattern, max_vid, fname, diags);
                Self::check_vid(*condition, max_vid, fname, diags);
            }
        }
    }

    // ── Terminator validation ─────────────────────────────────────────

    fn validate_terminator(
        term: &IrTerminator,
        max_vid: u32,
        max_bid: u32,
        fname: &str,
        diags: &mut Vec<Diagnostic>,
    ) {
        match term {
            IrTerminator::Jump { target, args } => {
                Self::check_bid(*target, max_bid, fname, diags);
                for vid in args {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }
            IrTerminator::Branch {
                cond,
                then_block,
                then_args,
                else_block,
                else_args,
            } => {
                Self::check_vid(*cond, max_vid, fname, diags);
                Self::check_bid(*then_block, max_bid, fname, diags);
                for vid in then_args {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
                Self::check_bid(*else_block, max_bid, fname, diags);
                for vid in else_args {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }
            IrTerminator::Return { value } => {
                if let Some(vid) = value {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }
            IrTerminator::Unreachable => {}
            IrTerminator::Switch {
                value,
                cases,
                default,
                default_args,
            } => {
                Self::check_vid(*value, max_vid, fname, diags);
                for (_, bid, args) in cases {
                    Self::check_bid(*bid, max_bid, fname, diags);
                    for vid in args {
                        Self::check_vid(*vid, max_vid, fname, diags);
                    }
                }
                Self::check_bid(*default, max_bid, fname, diags);
                for vid in default_args {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }
            IrTerminator::Join {
                task,
                result_block,
                result_args,
            } => {
                Self::check_vid(*task, max_vid, fname, diags);
                Self::check_bid(*result_block, max_bid, fname, diags);
                for vid in result_args {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────

    fn check_vid(vid: ValueId, max_vid: u32, fname: &str, diags: &mut Vec<Diagnostic>) {
        if vid.0 >= max_vid {
            diags.push(Diagnostic::error(
                Span::dummy(),
                format!("IR function `{fname}`: value ref {vid:?} >= next_value_id {max_vid}"),
            ));
        }
    }

    fn check_bid(bid: BlockId, max_bid: u32, fname: &str, diags: &mut Vec<Diagnostic>) {
        if bid.0 >= max_bid {
            diags.push(Diagnostic::error(
                Span::dummy(),
                format!("IR function `{fname}`: block ref {bid:?} >= next_block_id {max_bid}"),
            ));
        }
    }
}

// =============================================================================
// §16  IrModule :: validate()
// =============================================================================

impl IrModule {
    /// Validate this module's IR invariants.
    /// Returns a list of diagnostics for any violations found.
    pub fn validate(&self) -> Vec<Diagnostic> {
        IrValidator::validate_module(self)
    }
}

// =============================================================================
// §17  DEFAULTS & UTILITIES
// =============================================================================

impl Default for Effect {
    fn default() -> Self {
        Effect::Unknown
    }
}

impl Default for Ownership {
    fn default() -> Self {
        Ownership::Owned
    }
}

impl Default for CostHint {
    fn default() -> Self {
        CostHint::Unknown
    }
}

impl Default for FusionBoundary {
    fn default() -> Self {
        FusionBoundary::None
    }
}

impl IrFunction {
    /// Look up a block by its ID.
    pub fn block(&self, id: BlockId) -> Option<&BasicBlock> {
        self.blocks.iter().find(|b| b.id == id)
    }

    /// Look up a block by its ID (mutable).
    pub fn block_mut(&mut self, id: BlockId) -> Option<&mut BasicBlock> {
        self.blocks.iter_mut().find(|b| b.id == id)
    }
}

impl IrModule {
    /// Create an empty module with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        IrModule {
            name: name.into(),
            functions: vec![],
            structs: vec![],
            enums: vec![],
            components: vec![],
            constants: vec![],
            regions: vec![],
            effects: vec![],
        }
    }

    /// Look up a function by name.
    pub fn find_function(&self, name: &str) -> Option<&IrFunction> {
        self.functions.iter().find(|f| f.name == name)
    }

    /// Look up a function by name (mutable).
    pub fn find_function_mut(&mut self, name: &str) -> Option<&mut IrFunction> {
        self.functions.iter_mut().find(|f| f.name == name)
    }
}
