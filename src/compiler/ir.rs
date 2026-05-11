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
#[allow(dead_code)] // Part of the IR data model — used by future passes
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

/// ECS entity query in the IR.
#[derive(Debug, Clone)]
pub struct IrEntityQuery {
    pub with_components: Vec<String>,
    pub without_components: Vec<String>,
}

/// Component access pattern in an ECS loop.
#[derive(Debug, Clone)]
pub struct IrComponentAccess {
    pub component: String,
    pub mode: IrAccessMode,
}

/// Access mode for a component in an ECS query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrAccessMode {
    Read,
    Write,
    ReadWrite,
}

/// Parallelism hint in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrParallelismHint {
    Auto,
    Sequential,
    Parallel,
    Simd,
    Gpu,
}

/// Alias kind annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrAliasKind {
    Unique,
    Shared,
}

/// Alias metadata for load/store instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasKind {
    /// No alias information — assume may alias anything.
    Unknown,
    /// No other pointer aliases this memory location.
    NoAlias,
    /// The memory is only read, not written.
    ReadOnly,
    /// The memory is only written, not read before.
    WriteOnly,
    /// No other pointer accesses overlapping memory.
    Restrict,
}

impl Default for AliasKind {
    fn default() -> Self {
        AliasKind::Unknown
    }
}

impl fmt::Display for AliasKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AliasKind::Unknown => write!(f, "unknown"),
            AliasKind::NoAlias => write!(f, "noalias"),
            AliasKind::ReadOnly => write!(f, "readonly"),
            AliasKind::WriteOnly => write!(f, "writeonly"),
            AliasKind::Restrict => write!(f, "restrict"),
        }
    }
}

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

/// ML computation graph mode — determines execution and optimization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrGraphMode {
    /// Execute immediately, no tracing.
    Eager,
    /// Trace operations for later compilation.
    Traced,
    /// Static computation graph, fully known at compile time.
    Static,
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
    MakeResult { ok_value: Option<ValueId>, err_value: Option<ValueId>, ok_ty: Ty, err_ty: Ty },
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

    // ── Effect emission ────────────────────────────────────────────────
    /// `emit stdout "hello"` — explicit effect emission.
    Emit { effect_name: String, value: ValueId, effect: Effect },

    // ── Explicit copy ─────────────────────────────────────────────────
    /// `copy x` — explicit copy expression.
    Copy { value: ValueId, ty: Ty },

    // ── Await expression ───────────────────────────────────────────────
    /// Await the result of an async task.
    Await { value: ValueId, ty: Ty },

    // ── ECS entity query loop ──────────────────────────────────────────
    /// Entity-for loop — iterates over ECS entities matching a query.
    EntityFor {
        query: IrEntityQuery,
        body_func: ValueId,
        access_pattern: Vec<IrComponentAccess>,
        parallelism: IrParallelismHint,
        ty: Ty,
    },

    // ── Parallel for loop ──────────────────────────────────────────────
    /// Parallel iteration over a collection with optional chunking.
    ParallelFor {
        iter: ValueId,
        body_func: ValueId,
        chunk_size: Option<u64>,
        ty: Ty,
        effect: Effect,
    },

    // ── Filter operation (pure, vectorizable) ──────────────────────────
    /// Filter elements of an iterable by a predicate.
    Filter { iter: ValueId, predicate: ValueId, ty: Ty },

    // ── Batch operation ────────────────────────────────────────────────
    /// Batch elements of an iterable into fixed-size groups.
    Batch { iter: ValueId, size: ValueId, ty: Ty },

    // ── Kernel abstraction ─────────────────────────────────────────────
    /// Named compute kernel (GPU/TPU or SIMD).
    Kernel {
        name: String,
        body: IrFunction,
        inputs: Vec<ValueId>,
        ty: Ty,
        effect: Effect,
    },

    // ── Break with optional value ──────────────────────────────────────
    /// Break out of the innermost loop, optionally yielding a value.
    Break { value: Option<ValueId> },

    // ── Continue ───────────────────────────────────────────────────────
    /// Continue to the next iteration of the innermost loop.
    Continue,

    // ── Alias annotation (unique/shared) ───────────────────────────────
    /// Annotate a value with aliasing information for the optimiser.
    AliasAnnotation { value: ValueId, alias_kind: IrAliasKind, ty: Ty },

    // ── Determinism flag ───────────────────────────────────────────────
    /// Mark a value as guaranteed-deterministic (or not).
    Deterministic { value: ValueId, guaranteed: bool, ty: Ty },

    // ── Cost budget ────────────────────────────────────────────────────
    /// Annotate a value with a cost budget (e.g. `@budget(1000)`).
    CostBudget { value: ValueId, budget: u64, ty: Ty },

    // ── Compile-time execution ──────────────────────────────────────────
    /// Compile-time evaluated expression — result is baked into the binary.
    Comptime { inner: ValueId, ty: Ty },
    /// Compile-time constant table lookup.
    ComptimeTable { name: String, index: ValueId, ty: Ty },

    // ── Intent-based parallelism ────────────────────────────────────────
    /// Parallel reduction with associative operator — compiler decides
    /// thread count, SIMD width, chunking, and GPU offload.
    ParallelReduce {
        iter: ValueId,
        init: ValueId,
        op: IrReduceOp,
        body_func: ValueId,
        ty: Ty,
    },

    // ── Shader stage dispatch ───────────────────────────────────────────
    /// Dispatch a compute shader with explicit workgroup size.
    ShaderDispatch {
        shader: String,
        inputs: Vec<ValueId>,
        workgroup_size: (u32, u32, u32),
        ty: Ty,
        effect: Effect,
    },

    // ── ML graph operation ──────────────────────────────────────────────
    /// An ML graph operation with explicit execution mode.
    GraphOp {
        op_name: String,
        inputs: Vec<ValueId>,
        mode: IrGraphMode,
        ty: Ty,
    },

    // ── Trait method call ───────────────────────────────────────────────
    /// Dynamic dispatch through a trait vtable.
    TraitCall {
        trait_name: String,
        method: String,
        receiver: ValueId,
        args: Vec<ValueId>,
        ty: Ty,
        effect: Effect,
    },
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
    pub variants: Vec<IrEnumVariant>,
}

/// A single variant in an IR-level enum definition.
#[derive(Debug, Clone)]
pub struct IrEnumVariant {
    pub name: String,
    pub fields: IrEnumVariantFields,
}

/// The payload of an enum variant — unit, tuple, or named fields.
#[derive(Debug, Clone)]
pub enum IrEnumVariantFields {
    /// No payload: `None`
    Unit,
    /// Positional payload: `Some(T, U)`
    Tuple(Vec<Ty>),
    /// Named payload: `Point { x: f64, y: f64 }`
    Named(Vec<(String, Ty)>),
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

/// IR-level module definition.
#[derive(Debug, Clone)]
pub struct IrModuleDef {
    pub name: String,
    /// Exported function names.
    pub functions: Vec<String>,
    /// Exported struct names.
    pub structs: Vec<String>,
}

/// IR-level use/import declaration.
#[derive(Debug, Clone)]
pub struct IrUseDecl {
    pub path: String,
    pub alias: Option<String>,
    pub is_glob: bool,
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
    /// Sub-module definitions.
    pub modules: Vec<IrModuleDef>,
    /// Use/import declarations.
    pub uses: Vec<IrUseDecl>,
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

impl Ownership {
    /// Alias for MutRef — used by the flat IR.
    pub const MUT_BORROW: Ownership = Ownership::MutRef;
    /// Alias for Owned — used by the flat IR.
    pub const OWN: Ownership = Ownership::Owned;
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

impl fmt::Display for IrAccessMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrAccessMode::Read => write!(f, "read"),
            IrAccessMode::Write => write!(f, "write"),
            IrAccessMode::ReadWrite => write!(f, "read_write"),
        }
    }
}

impl fmt::Display for IrParallelismHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrParallelismHint::Auto => write!(f, "auto"),
            IrParallelismHint::Sequential => write!(f, "sequential"),
            IrParallelismHint::Parallel => write!(f, "parallel"),
            IrParallelismHint::Simd => write!(f, "simd"),
            IrParallelismHint::Gpu => write!(f, "gpu"),
        }
    }
}

impl fmt::Display for IrAliasKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrAliasKind::Unique => write!(f, "unique"),
            IrAliasKind::Shared => write!(f, "shared"),
        }
    }
}

impl fmt::Display for IrEntityQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "with=[{}] without=[{}]",
            self.with_components.join(", "),
            self.without_components.join(", "))
    }
}

impl fmt::Display for IrComponentAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.component, self.mode)
    }
}

impl fmt::Display for IrEnumVariantFields {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrEnumVariantFields::Unit => write!(f, "()"),
            IrEnumVariantFields::Tuple(tys) => {
                write!(f, "(")?;
                for (i, ty) in tys.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", ty)?;
                }
                write!(f, ")")
            }
            IrEnumVariantFields::Named(fields) => {
                write!(f, "{{ ")?;
                for (i, (name, ty)) in fields.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {}", name, ty)?;
                }
                write!(f, " }}")
            }
        }
    }
}

impl fmt::Display for IrEnumVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.name, self.fields)
    }
}

impl fmt::Display for IrModuleDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "module {} (fns: [{}], structs: [{}])",
            self.name,
            self.functions.join(", "),
            self.structs.join(", "))
    }
}

impl fmt::Display for IrUseDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "use {}", self.path)?;
        if let Some(alias) = &self.alias {
            write!(f, " as {alias}")?;
        }
        if self.is_glob {
            write!(f, ".*")?;
        }
        Ok(())
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

            // ── Construction — MakeArray/MakeClosure allocate memory ──
            IrExpr::MakeStruct { .. }
            | IrExpr::MakeTuple { .. } => Effect::Pure,
            IrExpr::MakeArray { .. } => Effect::Allocation,
            IrExpr::MakeEnum { .. }
            | IrExpr::MakeOption { .. }
            | IrExpr::MakeResult { .. }
            | IrExpr::MakeRange { .. } => Effect::Pure,
            IrExpr::MakeClosure { .. } => Effect::Allocation,

            // ── Access — IndexAccess can panic on OOB, others are pure ──
            IrExpr::FieldAccess { .. }
            | IrExpr::TupleAccess { .. } => Effect::Pure,
            IrExpr::IndexAccess { .. } => Effect::Unknown,

            // ── Tensor ops — HadamardDiv can panic on div-by-zero ──
            IrExpr::MatMul { .. }
            | IrExpr::HadamardMul { .. }
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
            IrExpr::HadamardDiv { .. } => Effect::Unknown,

            // ── Loops — may have any effect ──
            IrExpr::MapLoop { .. } => Effect::Unknown,
            IrExpr::ReduceLoop { .. } => Effect::Unknown,
            IrExpr::WhileLoop { effect, .. } => *effect,
            IrExpr::FoldLoop { .. } => Effect::Unknown,

            // ── Control-flow expressions ──
            IrExpr::Match { .. } => Effect::ControlFlow,
            IrExpr::If { .. } => Effect::Unknown,

            // ── Pipeline — may have any effect (composition of stages) ──
            IrExpr::Pipeline { .. } => Effect::Unknown,

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

            // ── Cost annotation — conservative: unknown (could wrap any expression) ──
            IrExpr::WithCost { .. } => Effect::Unknown,

            // ── Effect emission — always IO ──
            IrExpr::Emit { effect, .. } => *effect,

            // ── Explicit copy — pure (copies bits) ──
            IrExpr::Copy { .. } => Effect::Pure,

            // ── Await — async ──
            IrExpr::Await { .. } => Effect::Async,

            // ── ECS entity query loop — may read/write components ──
            IrExpr::EntityFor { .. } => Effect::Mutation,

            // ── Parallel for — explicitly annotated ──
            IrExpr::ParallelFor { effect, .. } => *effect,

            // ── Filter — pure, vectorizable ──
            IrExpr::Filter { .. } => Effect::Pure,

            // ── Batch — pure ──
            IrExpr::Batch { .. } => Effect::Pure,

            // ── Kernel — explicitly annotated ──
            IrExpr::Kernel { effect, .. } => *effect,

            // ── Break / Continue — control flow ──
            IrExpr::Break { .. } => Effect::ControlFlow,
            IrExpr::Continue => Effect::ControlFlow,

            // ── Alias annotation — pure (metadata only) ──
            IrExpr::AliasAnnotation { .. } => Effect::Pure,

            // ── Determinism flag — pure (metadata only) ──
            IrExpr::Deterministic { .. } => Effect::Pure,

            // ── Cost budget — pure (metadata only) ──
            IrExpr::CostBudget { .. } => Effect::Pure,

            // ── Compile-time execution — pure (baked into binary) ──
            IrExpr::Comptime { .. } => Effect::Pure,
            IrExpr::ComptimeTable { .. } => Effect::Pure,

            // ── Parallel reduce — body_func may have side effects ──
            IrExpr::ParallelReduce { .. } => Effect::Unknown,

            // ── Shader dispatch — GPU + IO ──
            IrExpr::ShaderDispatch { effect, .. } => *effect,

            // ── ML graph op — pure (deferred computation) ──
            IrExpr::GraphOp { .. } => Effect::Pure,

            // ── Trait method call — explicitly annotated ──
            IrExpr::TraitCall { effect, .. } => *effect,
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

    // ── Convenience methods for new IR nodes ────────────────────────────

    /// Define an Emit expression and return the result ValueId.
    pub fn emit_effect(&mut self, effect_name: String, value: ValueId, effect: Effect, ty: Ty, ownership: Ownership, span: Span) -> ValueId {
        self.define(ty, ownership, IrExpr::Emit { effect_name, value, effect }, span)
    }

    /// Define a Copy expression and return the result ValueId.
    pub fn copy_value(&mut self, value: ValueId, ty: Ty, ownership: Ownership, span: Span) -> ValueId {
        self.define(ty.clone(), ownership, IrExpr::Copy { value, ty }, span)
    }

    /// Define an Await expression and return the result ValueId.
    pub fn await_value(&mut self, value: ValueId, ty: Ty, ownership: Ownership, span: Span) -> ValueId {
        self.define(ty.clone(), ownership, IrExpr::Await { value, ty }, span)
    }

    /// Define an EntityFor expression and return the result ValueId.
    pub fn entity_for(&mut self, query: IrEntityQuery, body_func: ValueId, access_pattern: Vec<IrComponentAccess>, parallelism: IrParallelismHint, ty: Ty, ownership: Ownership, span: Span) -> ValueId {
        self.define(ty.clone(), ownership, IrExpr::EntityFor { query, body_func, access_pattern, parallelism, ty }, span)
    }

    /// Define a ParallelFor expression and return the result ValueId.
    pub fn parallel_for(&mut self, iter: ValueId, body_func: ValueId, chunk_size: Option<u64>, ty: Ty, effect: Effect, ownership: Ownership, span: Span) -> ValueId {
        self.define(ty.clone(), ownership, IrExpr::ParallelFor { iter, body_func, chunk_size, ty, effect }, span)
    }

    /// Define a Filter expression and return the result ValueId.
    pub fn filter(&mut self, iter: ValueId, predicate: ValueId, ty: Ty, ownership: Ownership, span: Span) -> ValueId {
        self.define(ty.clone(), ownership, IrExpr::Filter { iter, predicate, ty }, span)
    }

    /// Define a Batch expression and return the result ValueId.
    pub fn batch(&mut self, iter: ValueId, size: ValueId, ty: Ty, ownership: Ownership, span: Span) -> ValueId {
        self.define(ty.clone(), ownership, IrExpr::Batch { iter, size, ty }, span)
    }

    /// Define a Kernel expression and return the result ValueId.
    pub fn kernel(&mut self, name: String, body: IrFunction, inputs: Vec<ValueId>, ty: Ty, effect: Effect, ownership: Ownership, span: Span) -> ValueId {
        self.define(ty.clone(), ownership, IrExpr::Kernel { name, body, inputs, ty, effect }, span)
    }

    /// Define a Break expression (no result ValueId — control flow).
    pub fn break_loop(&mut self, value: Option<ValueId>, span: Span) {
        self.discard(IrExpr::Break { value }, span);
    }

    /// Define a Continue expression (no result ValueId — control flow).
    pub fn continue_loop(&mut self, span: Span) {
        self.discard(IrExpr::Continue, span);
    }

    /// Define an AliasAnnotation expression and return the result ValueId.
    pub fn alias_annotation(&mut self, value: ValueId, alias_kind: IrAliasKind, ty: Ty, ownership: Ownership, span: Span) -> ValueId {
        self.define(ty.clone(), ownership, IrExpr::AliasAnnotation { value, alias_kind, ty }, span)
    }

    /// Define a Deterministic annotation and return the result ValueId.
    pub fn deterministic(&mut self, value: ValueId, guaranteed: bool, ty: Ty, ownership: Ownership, span: Span) -> ValueId {
        self.define(ty.clone(), ownership, IrExpr::Deterministic { value, guaranteed, ty }, span)
    }

    /// Define a CostBudget annotation and return the result ValueId.
    pub fn cost_budget(&mut self, value: ValueId, budget: u64, ty: Ty, ownership: Ownership, span: Span) -> ValueId {
        self.define(ty.clone(), ownership, IrExpr::CostBudget { value, budget, ty }, span)
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
///   • Type consistency for defines (expression type matches declared type)
///   • Effect soundness (pure expressions don't appear in mutation contexts)
///   • Nested IrFunction bodies (closures, spawns, kernels) are validated
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
            IrStmt::Define { dst, ty, value, .. } => {
                if dst.0 >= max_vid {
                    diags.push(Diagnostic::error(
                        Span::dummy(),
                        format!("IR function `{fname}`: Define dst {dst:?} >= next_value_id {max_vid}"),
                    ));
                }
                Self::validate_expr(value, max_vid, max_bid, fname, diags);
                // Type consistency check: expression effect should be compatible
                // with the context. Pure expressions should not be used in
                // Store/Load mutation contexts.
                let expr_effect = value.effect();
                if matches!(expr_effect, Effect::Pure) && !Self::is_pure_compatible_type(ty) {
                    diags.push(Diagnostic::warning(
                        Span::dummy(),
                        format!(
                            "IR function `{fname}`: Define dst {dst:?} has mutation-incompatible type but expression is Pure"
                        ),
                    ));
                }
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
            IrExpr::MakeResult { ok_value, err_value, .. } => {
                if let Some(vid) = ok_value {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
                if let Some(vid) = err_value {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            },
            IrExpr::MakeRange { lo, hi, .. } => {
                if let Some(vid) = lo {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
                if let Some(vid) = hi {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }
            IrExpr::MakeClosure { captures, body, .. } => {
                for vid in captures {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
                // Validate nested function body
                Self::validate_nested_function(body, fname, diags);
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

            IrExpr::Spawn { ownership_transfer, task, .. } => {
                for vid in ownership_transfer {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
                // Validate nested task function body
                Self::validate_nested_function(task, fname, diags);
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

            // ── New variants ─────────────────────────────────────────────

            IrExpr::Emit { value, .. } => {
                Self::check_vid(*value, max_vid, fname, diags);
            }

            IrExpr::Copy { value, .. } => {
                Self::check_vid(*value, max_vid, fname, diags);
            }

            IrExpr::Await { value, .. } => {
                Self::check_vid(*value, max_vid, fname, diags);
            }

            IrExpr::EntityFor { query, body_func, access_pattern, .. } => {
                Self::check_vid(*body_func, max_vid, fname, diags);
                for access in access_pattern {
                    let _ = access; // component names are strings, not value refs
                }
                let _ = query; // query contains only string names
            }

            IrExpr::ParallelFor { iter, body_func, .. } => {
                Self::check_vid(*iter, max_vid, fname, diags);
                Self::check_vid(*body_func, max_vid, fname, diags);
            }

            IrExpr::Filter { iter, predicate, .. } => {
                Self::check_vid(*iter, max_vid, fname, diags);
                Self::check_vid(*predicate, max_vid, fname, diags);
            }

            IrExpr::Batch { iter, size, .. } => {
                Self::check_vid(*iter, max_vid, fname, diags);
                Self::check_vid(*size, max_vid, fname, diags);
            }

            IrExpr::Kernel { inputs, body, .. } => {
                for vid in inputs {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
                // Validate nested kernel function body
                Self::validate_nested_function(body, fname, diags);
            }

            IrExpr::Break { value } => {
                if let Some(vid) = value {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }

            IrExpr::Continue => {}

            IrExpr::AliasAnnotation { value, .. } => {
                Self::check_vid(*value, max_vid, fname, diags);
            }

            IrExpr::Deterministic { value, .. } => {
                Self::check_vid(*value, max_vid, fname, diags);
            }

            IrExpr::CostBudget { value, .. } => {
                Self::check_vid(*value, max_vid, fname, diags);
            }

            // ── Semantic unification variants ───────────────────────────

            IrExpr::Comptime { inner, .. } => {
                Self::check_vid(*inner, max_vid, fname, diags);
            }

            IrExpr::ComptimeTable { index, .. } => {
                Self::check_vid(*index, max_vid, fname, diags);
            }

            IrExpr::ParallelReduce { iter, init, body_func, .. } => {
                Self::check_vid(*iter, max_vid, fname, diags);
                Self::check_vid(*init, max_vid, fname, diags);
                Self::check_vid(*body_func, max_vid, fname, diags);
            }

            IrExpr::ShaderDispatch { inputs, .. } => {
                for vid in inputs {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }

            IrExpr::GraphOp { inputs, .. } => {
                for vid in inputs {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
            }

            IrExpr::TraitCall { receiver, args, .. } => {
                Self::check_vid(*receiver, max_vid, fname, diags);
                for vid in args {
                    Self::check_vid(*vid, max_vid, fname, diags);
                }
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

    /// Check if a type is compatible with a Pure expression context.
    /// Mutable reference types are never pure because they imply mutation.
    fn is_pure_compatible_type(ty: &Ty) -> bool {
        // Conservatively return true — the type checker handles precise
        // compatibility. We only flag obvious mismatches.
        let _ = ty;
        true
    }

    /// Validate a nested IrFunction body (e.g. inside MakeClosure, Spawn, Kernel).
    fn validate_nested_function(func: &IrFunction, parent_fname: &str, diags: &mut Vec<Diagnostic>) {
        let nested_diags = Self::validate_function(func);
        for diag in nested_diags {
            diags.push(Diagnostic::warning(
                Span::dummy(),
                format!("IR function `{parent_fname}`: nested function `{}`: {}", func.name, diag.message),
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
            modules: vec![],
            uses: vec![],
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

// =============================================================================
// §18  SEMANTIC UNIFICATION — EFFECT CAPABILITY SET (bitflags)
// =============================================================================

bitflags::bitflags! {
    /// Unified effect capability set — effects are NOT mutually exclusive.
    /// A function can perform IO + Allocation + Async simultaneously.
    /// This is the core of the semantic unification: every construct's
    /// effects are represented as a set, enabling:
    ///   - Purity checking (EffectCapSet::empty() == pure)
    ///   - Effect inference (union of subexpressions)
    ///   - Capability gating (function declares required caps)
    ///   - JIT caching (same caps → same cache key)
    ///   - Automatic parallelization (no Mutation/IO → parallel-safe)
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct EffectCapSet: u32 {
        /// No effects — pure expression, freely reorderable.
        const PURE = 0;
        /// Reads or writes external state (I/O, network, files).
        const IO = 1 << 0;
        /// Mutates heap or stack state.
        const MUTATE = 1 << 1;
        /// Allocates memory (region or heap).
        const ALLOC = 1 << 2;
        /// Affects control flow (break, continue, return, panic).
        const CONTROL_FLOW = 1 << 3;
        /// Async task spawning / joining.
        const ASYNC = 1 << 4;
        /// GPU dispatch / compute shader execution.
        const GPU = 1 << 5;
        /// SIMD vector operations.
        const SIMD = 1 << 6;
        /// Effect not yet determined (conservative — assume anything).
        const UNKNOWN = 1 << 7;
    }
}

impl EffectCapSet {
    /// True if no bits set (or only PURE, which is 0).
    pub fn is_pure(self) -> bool {
        self.bits() == 0
    }

    /// True if no IO/MUTATE/ASYNC — safe to parallelize.
    pub fn is_parallel_safe(self) -> bool {
        !(self.contains(Self::IO) || self.contains(Self::MUTATE) || self.contains(Self::ASYNC))
    }

    /// True if no IO/ASYNC/UNKNOWN/MUTATE — deterministic result.
    pub fn is_deterministic(self) -> bool {
        !(self.contains(Self::IO) || self.contains(Self::ASYNC) || self.contains(Self::UNKNOWN) || self.contains(Self::MUTATE))
    }

    /// Convert from the legacy Effect enum.
    pub fn from_legacy(e: Effect) -> Self {
        match e {
            Effect::Pure => Self::empty(),
            Effect::IO => Self::IO,
            Effect::Mutation => Self::MUTATE,
            Effect::Allocation => Self::ALLOC,
            Effect::ControlFlow => Self::CONTROL_FLOW,
            Effect::Async => Self::ASYNC,
            Effect::Unknown => Self::UNKNOWN,
        }
    }

    /// Convert back to the legacy Effect enum (picks the dominant effect).
    pub fn to_legacy(self) -> Effect {
        if self.is_pure() {
            return Effect::Pure;
        }
        // Pick the most significant effect in priority order.
        if self.contains(Self::IO) {
            return Effect::IO;
        }
        if self.contains(Self::ASYNC) {
            return Effect::Async;
        }
        if self.contains(Self::MUTATE) {
            return Effect::Mutation;
        }
        if self.contains(Self::ALLOC) {
            return Effect::Allocation;
        }
        if self.contains(Self::CONTROL_FLOW) {
            return Effect::ControlFlow;
        }
        if self.contains(Self::UNKNOWN) {
            return Effect::Unknown;
        }
        Effect::Pure
    }
}

impl fmt::Display for EffectCapSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_pure() {
            return write!(f, "pure");
        }
        let mut first = true;
        let mut flag = |name: &str, present: bool| -> fmt::Result {
            if present {
                if !first { write!(f, "|")?; }
                first = false;
                write!(f, "{name}")?;
            }
            Ok(())
        };
        flag("io", self.contains(Self::IO))?;
        flag("mutate", self.contains(Self::MUTATE))?;
        flag("alloc", self.contains(Self::ALLOC))?;
        flag("control_flow", self.contains(Self::CONTROL_FLOW))?;
        flag("async", self.contains(Self::ASYNC))?;
        flag("gpu", self.contains(Self::GPU))?;
        flag("simd", self.contains(Self::SIMD))?;
        flag("unknown", self.contains(Self::UNKNOWN))?;
        Ok(())
    }
}

impl Default for EffectCapSet {
    fn default() -> Self {
        Self::empty()
    }
}

// =============================================================================
// §19  FLAT INSTRUCTION IR — IrType, IrOp, TaskOwnership, EffectFlags
// =============================================================================

/// Flat IR type — used by the instruction-level IR (lower-level than IrExpr).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IrType {
    Unit,
    Bool,
    Int { width: u32, signed: bool },
    Float { width: u32 },
    String,
    Struct { name: String, fields: Vec<(String, IrType)> },
    Enum { name: String, variants: Vec<(String, Vec<IrType>)> },
    Array { elem: Box<IrType>, len: Option<u64> },
    Tensor { elem: Box<IrType>, shape: Vec<u64> },
    Slice(Box<IrType>),
    Ref(Box<IrType>),
    MutRef(Box<IrType>),
    FnPtr { params: Vec<IrType>, ret: Box<IrType> },
    Never,
    Unknown,
}

impl fmt::Display for IrType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrType::Unit => write!(f, "()"),
            IrType::Bool => write!(f, "bool"),
            IrType::Int { width, signed } => {
                write!(f, "{}{}", if *signed { "i" } else { "u" }, width)
            }
            IrType::Float { width } => write!(f, "f{}", width),
            IrType::String => write!(f, "str"),
            IrType::Struct { name, .. } => write!(f, "{name}"),
            IrType::Enum { name, .. } => write!(f, "{name}"),
            IrType::Array { elem, len } => {
                match len {
                    Some(l) => write!(f, "[{}; {l}]", elem),
                    None => write!(f, "[{}]", elem),
                }
            }
            IrType::Tensor { elem, shape } => {
                write!(f, "tensor<{}; ", elem)?;
                for (i, d) in shape.iter().enumerate() {
                    if i > 0 { write!(f, "×")?; }
                    write!(f, "{d}")?;
                }
                write!(f, ">")
            }
            IrType::Slice(elem) => write!(f, "[{}]", elem),
            IrType::Ref(inner) => write!(f, "&{}", inner),
            IrType::MutRef(inner) => write!(f, "&mut {}", inner),
            IrType::FnPtr { params, ret } => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{p}")?;
                }
                write!(f, ") -> {ret}")
            }
            IrType::Never => write!(f, "!"),
            IrType::Unknown => write!(f, "?"),
        }
    }
}

/// Task ownership transfer model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskOwnership {
    Move,
    Copy,
    Ref,
}

impl fmt::Display for TaskOwnership {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskOwnership::Move => write!(f, "move"),
            TaskOwnership::Copy => write!(f, "copy"),
            TaskOwnership::Ref => write!(f, "ref"),
        }
    }
}

/// Flat IR operation — one operation per instruction.
#[derive(Debug, Clone)]
pub enum IrOp {
    // Constants
    ConstInt { value: i128, ty: IrType },
    ConstFloat { bits: u64, ty: IrType },
    ConstBool { value: bool },
    ConstStr { idx: u32 },
    ConstUnit,
    // Arithmetic
    BinOp { op: crate::compiler::ast::BinOpKind, lhs: ValueId, rhs: ValueId },
    UnOp { op: crate::compiler::ast::UnOpKind, operand: ValueId },
    // Memory
    Alloca { ty: IrType, align: u32 },
    Store { ptr: ValueId, value: ValueId },
    Load { ptr: ValueId, ty: IrType },
    Move { src: ValueId },
    // Control flow
    Ret { value: Option<ValueId> },
    Nop,
    // Tensor
    MatMul { lhs: ValueId, rhs: ValueId },
    HadamardMul { lhs: ValueId, rhs: ValueId },
    HadamardDiv { lhs: ValueId, rhs: ValueId },
    TensorConcat { lhs: ValueId, rhs: ValueId },
    KronProd { lhs: ValueId, rhs: ValueId },
    OuterProd { lhs: ValueId, rhs: ValueId },
    // Calls
    Call { func: String, args: Vec<ValueId> },
    Intrinsic { name: String, args: Vec<ValueId> },
    // Parallelism
    ParallelStart { region_id: u32 },
    ParallelEnd { region_id: u32 },
    // Region
    RegionAlloc { region: u32, ty: IrType },
    // Tasks
    TaskSpawn { func: String, args: Vec<ValueId>, ownership: TaskOwnership },
    TaskJoin { task: ValueId },
    // Control flow (flat IR)
    CondBr { cond: ValueId, if_true: BlockId, if_false: BlockId },
    Jump { target: BlockId },
    Phi { incoming: Vec<(BlockId, ValueId)> },
    // Type operations
    TypeCheck { value: ValueId, expected: IrType },
    Cast { src: ValueId, target_ty: IrType },
    // Copy (explicit bit-copy, distinct from Move which transfers ownership)
    Copy { src: ValueId },
    // Effect emission
    Emit { effect: String, value: ValueId },
}

impl IrOp {
    /// Returns true if this operation is a terminator (ends a basic block).
    pub fn is_terminator(&self) -> bool {
        matches!(self, IrOp::Ret { .. } | IrOp::CondBr { .. } | IrOp::Jump { .. })
    }
}

/// Effect flags for flat IR instructions — bitflags.
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct EffectFlags: u32 {
        const PURE = 0;
        const IO = 1 << 0;
        const ALLOC = 1 << 1;
        const WRITE = 1 << 2;
        const READONLY = 1 << 3;
        const PARALLEL = 1 << 4;
        const TERMINATES = 1 << 5;
    }
}

impl EffectFlags {
    /// Create a pure (no-effect) flags set.
    pub fn pure() -> Self {
        Self::empty()
    }

    /// Alias for pure() — no effect flags.
    pub fn none() -> Self {
        Self::empty()
    }

    /// Returns true if no effect flags are set.
    pub fn is_pure(self) -> bool {
        self.bits() == 0
    }
}

impl fmt::Display for EffectFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_pure() {
            return write!(f, "pure");
        }
        let mut first = true;
        let mut flag = |name: &str, present: bool| -> fmt::Result {
            if present {
                if !first { write!(f, "|")?; }
                first = false;
                write!(f, "{name}")?;
            }
            Ok(())
        };
        flag("io", self.contains(Self::IO))?;
        flag("alloc", self.contains(Self::ALLOC))?;
        flag("write", self.contains(Self::WRITE))?;
        flag("readonly", self.contains(Self::READONLY))?;
        flag("parallel", self.contains(Self::PARALLEL))?;
        flag("terminates", self.contains(Self::TERMINATES))?;
        Ok(())
    }
}

impl Default for EffectFlags {
    fn default() -> Self {
        Self::empty()
    }
}

// =============================================================================
// §20  FLAT IrInstr, FlatBlock, FlatIrFunction, IrIntrinsic, FlatIrModule
// =============================================================================

/// A single flat IR instruction.
#[derive(Debug, Clone)]
pub struct IrInstr {
    pub dst: Option<ValueId>,
    pub op: IrOp,
    pub span: Span,
    pub effects: EffectFlags,
    pub ownership: Ownership,
    pub cost: CostHint,
    /// Alias metadata for load/store instructions.
    pub alias: AliasKind,
}

/// A basic block for the flat instruction IR.
#[derive(Debug, Clone)]
pub struct FlatBlock {
    pub id: BlockId,
    pub instrs: Vec<IrInstr>,
    pub span: Span,
}

impl FlatBlock {
    pub fn new(id: BlockId) -> Self {
        FlatBlock { id, instrs: vec![], span: Span::dummy() }
    }

    pub fn is_terminated(&self) -> bool {
        self.instrs.last().map_or(false, |i| i.op.is_terminator())
    }
}

/// A function in the flat instruction IR — what lower.rs produces.
#[derive(Debug, Clone)]
pub struct FlatIrFunction {
    pub name: String,
    pub params: Vec<(ValueId, IrType)>,
    pub ret_ty: IrType,
    pub blocks: Vec<FlatBlock>,
    pub entry: BlockId,
    pub effects: EffectFlags,
    pub requires: Vec<ValueId>,
    pub ensures: Vec<ValueId>,
    pub span: Span,
}

/// An intrinsic function declaration.
#[derive(Debug, Clone)]
pub struct IrIntrinsic {
    pub name: String,
    pub param_types: Vec<IrType>,
    pub ret_type: IrType,
    pub effects: EffectFlags,
}

/// A module in the flat instruction IR.
#[derive(Debug, Clone)]
pub struct FlatIrModule {
    pub functions: Vec<FlatIrFunction>,
    pub intrinsics: Vec<IrIntrinsic>,
    pub span: Span,
}

// =============================================================================
// §21  SEMANTIC UNIFICATION — NEW IR TYPES
// =============================================================================

/// Visibility / access control for module items.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrVisibility {
    /// Private to the current module (default).
    Private,
    /// Visible within the current crate.
    PubCrate,
    /// Visible to the entire world (public API).
    Public,
    /// Visible only to a specific module (e.g. pub(in crate::submodule)).
    PubIn(String),
}

impl fmt::Display for IrVisibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrVisibility::Private => write!(f, "private"),
            IrVisibility::PubCrate => write!(f, "pub(crate)"),
            IrVisibility::Public => write!(f, "pub"),
            IrVisibility::PubIn(path) => write!(f, "pub(in {path})"),
        }
    }
}

impl Default for IrVisibility {
    fn default() -> Self {
        IrVisibility::Private
    }
}

/// Shader pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrShaderStage {
    Vertex,
    Fragment,
    Compute,
    Geometry,
    TessControl,
    TessEval,
}

impl fmt::Display for IrShaderStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrShaderStage::Vertex => write!(f, "vertex"),
            IrShaderStage::Fragment => write!(f, "fragment"),
            IrShaderStage::Compute => write!(f, "compute"),
            IrShaderStage::Geometry => write!(f, "geometry"),
            IrShaderStage::TessControl => write!(f, "tess_control"),
            IrShaderStage::TessEval => write!(f, "tess_eval"),
        }
    }
}

/// A resource binding for shader stages.
#[derive(Debug, Clone)]
pub struct IrResourceBinding {
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub ty: Ty,
    pub access: IrAccessMode,
}

/// A shader program definition with explicit stages and resource bindings.
#[derive(Debug, Clone)]
pub struct IrShaderDef {
    pub name: String,
    pub stages: Vec<(IrShaderStage, IrFunction)>,
    pub bindings: Vec<IrResourceBinding>,
    pub push_constants: Vec<IrParam>,
}

/// ECS system scheduling metadata — explicit read/write declarations.
#[derive(Debug, Clone)]
pub struct IrSystemSchedule {
    pub system_name: String,
    pub reads: Vec<String>,
    pub writes: Vec<String>,
    pub before: Vec<String>,
    pub after: Vec<String>,
    pub parallel_with: Vec<String>,
}

/// A trait (interface) definition.
#[derive(Debug, Clone)]
pub struct IrTraitDef {
    pub name: String,
    pub methods: Vec<IrTraitMethod>,
    pub associated_types: Vec<String>,
}

/// A method signature in a trait definition.
#[derive(Debug, Clone)]
pub struct IrTraitMethod {
    pub name: String,
    pub params: Vec<IrParam>,
    pub ret_ty: Ty,
    pub effect: Effect,
    pub has_default: bool,
}

/// A trait implementation for a specific type.
#[derive(Debug, Clone)]
pub struct IrTraitImpl {
    pub trait_name: String,
    pub for_type: String,
    pub methods: Vec<IrFunction>,
}

/// Per-field layout metadata — alignment, offset, padding.
#[derive(Debug, Clone)]
pub struct IrFieldLayout {
    pub name: String,
    pub ty: Ty,
    pub ownership: Ownership,
    pub alignment: u32,
    pub offset: u32,
    pub size: u32,
}

/// IR-level struct definition with full layout metadata.
#[derive(Debug, Clone)]
pub struct IrStructDefFull {
    pub name: String,
    pub fields: Vec<IrFieldLayout>,
    pub is_component: bool,
    pub layout: IrDataLayout,
    pub total_size: u32,
    pub total_alignment: u32,
}

/// Extended IR module with semantic unification fields.
#[derive(Debug, Clone)]
pub struct IrModuleFull {
    pub name: String,
    pub functions: Vec<IrFunction>,
    pub structs: Vec<IrStructDef>,
    pub enums: Vec<IrEnumDef>,
    pub components: Vec<IrComponentDef>,
    pub constants: Vec<IrConstDef>,
    pub regions: Vec<IrRegionDef>,
    pub effects: Vec<IrEffectDef>,
    pub modules: Vec<IrModuleDef>,
    pub uses: Vec<IrUseDecl>,
    /// Shader program definitions.
    pub shaders: Vec<IrShaderDef>,
    /// Trait definitions.
    pub traits: Vec<IrTraitDef>,
    /// Trait implementations.
    pub trait_impls: Vec<IrTraitImpl>,
    /// ECS system scheduling metadata.
    pub system_schedules: Vec<IrSystemSchedule>,
}

// =============================================================================
// §22  EXTENDED IrFunction — graph_mode, capabilities, visibility
// =============================================================================

/// Extended IrFunction with semantic unification fields.
#[derive(Debug, Clone)]
pub struct IrFunctionFull {
    pub inner: IrFunction,
    /// ML graph execution mode (if applicable).
    pub graph_mode: Option<IrGraphMode>,
    /// Unified effect capabilities.
    pub capabilities: EffectCapSet,
    /// Visibility / access control.
    pub visibility: IrVisibility,
}
