// =============================================================================
// jules/src/ast.rs
//
// Abstract Syntax Tree for the Jules programming language.
//
// Every node carries a `Span` so downstream passes (type checker, codegen,
// IDE tooling) always know where in source a construct originated.
//
// The two headline features represented here:
//
//   1. First-class tensors (Feature 1)
//        tensor<f32>[128, 128] A
//        C = A @ B
//
//   2. Deterministic game simulation (Feature 2)
//        system Update(dt: f32):
//            for entity in world:
//                entity.position += entity.velocity * dt
//
//      The compiler uses the `SystemDecl` + `EntityFor` AST nodes to:
//        • infer which components are read vs. written
//        • decide whether iterations are order-independent (→ SIMD / GPU)
//        • emit deterministic tick order when mutation order matters
// =============================================================================

use crate::lexer::Span;

// ─── Re-export Span for convenience in other modules ─────────────────────────

// =============================================================================
// TYPES
// =============================================================================

/// The scalar element types that can appear inside tensors and vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElemType {
    F16,
    F32,
    F64,
    Bf16,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
    Usize,
}

impl ElemType {
    /// Number of bytes this element type occupies.
    pub fn byte_size(&self) -> usize {
        match self {
            ElemType::F16 | ElemType::Bf16 | ElemType::I16 | ElemType::U16 => 2,
            ElemType::F32 | ElemType::I32 | ElemType::U32 => 4,
            ElemType::F64 | ElemType::I64 | ElemType::U64 => 8,
            ElemType::I8 | ElemType::U8 | ElemType::Bool => 1,
            ElemType::Usize => 8, // assume 64-bit target
        }
    }

    /// True for floating-point element types.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            ElemType::F16 | ElemType::F32 | ElemType::F64 | ElemType::Bf16
        )
    }

    /// True for signed integer types.
    pub fn is_signed_int(&self) -> bool {
        matches!(
            self,
            ElemType::I8 | ElemType::I16 | ElemType::I32 | ElemType::I64
        )
    }
}

// -----------------------------------------------------------------------------

/// A single dimension in a tensor or array shape.
/// `tensor<f32>[N, M, _]` — `N` and `M` are named, `_` is dynamic.
#[derive(Debug, Clone, PartialEq)]
pub enum DimExpr {
    /// A concrete integer literal: `[128, 128]`
    Lit(u64),
    /// A named compile-time constant: `[N, M]`
    Named(String),
    /// A dynamic dimension inferred at runtime: `[_]`
    Dynamic,
    /// An expression, e.g. `[N * 2]` — only valid in certain positions.
    Expr(Box<Expr>),
}

// -----------------------------------------------------------------------------

/// The number of lanes in a SIMD vector type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecSize {
    N2,
    N3,
    N4,
}

impl VecSize {
    pub fn lanes(self) -> u32 {
        match self {
            VecSize::N2 => 2,
            VecSize::N3 => 3,
            VecSize::N4 => 4,
        }
    }
}

/// The "family" of a vector type (float / signed int / unsigned int).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecFamily {
    Float,
    Int,
    UInt,
}

// -----------------------------------------------------------------------------

/// The full Jules type language.
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    // ── Scalars ──────────────────────────────────────────────────────────
    Scalar(ElemType),

    // ── Tensor (Feature 1) ───────────────────────────────────────────────
    /// `tensor<f32>[128, 128]`
    Tensor {
        elem: ElemType,
        shape: Vec<DimExpr>,
    },

    // ── Fixed-size vectors / matrices (Feature 2 / game sim) ─────────────
    /// `vec2`, `vec3`, `vec4` (float) — or `ivec3`, `uvec4`, etc.
    Vec {
        size: VecSize,
        family: VecFamily,
    },
    /// `mat2`, `mat3`, `mat4`
    Mat {
        size: VecSize,
    },
    /// Unit quaternion: `quat`
    Quat,

    // ── Compound types ────────────────────────────────────────────────────
    /// A user-defined named type: `Transform`, `Particle`, …
    Named(String),
    /// `(f32, vec3, bool)` — heterogeneous tuple
    Tuple(Vec<Type>),
    /// `[f32; 16]` — fixed-size array
    Array {
        elem: Box<Type>,
        len: Box<Expr>,
    },
    /// `[f32]` — unsized slice
    Slice(Box<Type>),
    /// `&T` or `&mut T`
    Ref {
        mutable: bool,
        inner: Box<Type>,
    },
    /// `Option<T>`
    Option(Box<Type>),
    /// `Result<T, E>`
    Result {
        ok: Box<Type>,
        err: Box<Type>,
    },
    /// `fn(A, B) -> C`
    FnPtr {
        params: Vec<Type>,
        ret: Box<Type>,
    },
    /// The never / bottom type `!`
    Never,
    /// Type to be inferred: `_`
    Infer,
}

impl Type {
    /// Is this type directly SIMD-vectorisable?
    /// (used by the game-sim lowering pass)
    pub fn is_simd_candidate(&self) -> bool {
        matches!(
            self,
            Type::Scalar(ElemType::F32)
                | Type::Scalar(ElemType::F64)
                | Type::Scalar(ElemType::I32)
                | Type::Vec { .. }
                | Type::Quat
        )
    }

    /// Does this type contain a gradient-tracked float?
    pub fn contains_float(&self) -> bool {
        match self {
            Type::Scalar(e) => e.is_float(),
            Type::Tensor { elem, .. } => elem.is_float(),
            Type::Vec { family, .. } => *family == VecFamily::Float,
            Type::Mat { .. } | Type::Quat => true,
            Type::Tuple(ts) => ts.iter().any(|t| t.contains_float()),
            _ => false,
        }
    }
}

// =============================================================================
// ATTRIBUTES
// =============================================================================

/// A Jules attribute, attached to declarations or statements.
///
/// ```text
/// @gpu
/// @simd
/// @parallel
/// @seq              // sequential (deterministic order)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Attribute {
    // Device placement
    Gpu,
    Cpu,
    Tpu,
    // Parallelism hints for the game-sim lowering pass
    Simd,
    Parallel,
    Seq, // force sequential execution for determinism
    // Gradient tracking
    Grad,
    // Free-form for future extensibility: `@inline`, `@deprecated(…)`, etc.
    Named { name: String, args: Vec<Expr> },
}

// =============================================================================
// PATTERNS
// =============================================================================

/// A destructuring pattern used in `let`, `match`, and `for` bindings.
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    /// `_` — wildcard
    Wildcard(Span),
    /// `x`, `mut y`
    Ident {
        span: Span,
        name: String,
        mutable: bool,
    },
    /// `(a, b, c)` — tuple destructure
    Tuple { span: Span, elems: Vec<Pattern> },
    /// `Point { x, y }` — struct destructure
    Struct {
        span: Span,
        path: String,
        fields: Vec<(String, Option<Pattern>)>,
    },
    /// `42`, `3.14`, `true`, `"hello"` — literal guard
    Lit(Span, LitVal),
    /// `Some(x)` — enum variant with inner pattern
    Enum {
        span: Span,
        path: String,
        inner: Vec<Pattern>,
    },
    /// `a..b` — range pattern
    Range {
        span: Span,
        lo: Box<Pattern>,
        hi: Box<Pattern>,
        inclusive: bool,
    },
    /// `pat1 | pat2` — or-pattern
    Or { span: Span, arms: Vec<Pattern> },
}

impl Pattern {
    pub fn span(&self) -> Span {
        match self {
            Pattern::Wildcard(s) => *s,
            Pattern::Ident { span, .. } => *span,
            Pattern::Tuple { span, .. } => *span,
            Pattern::Struct { span, .. } => *span,
            Pattern::Lit(span, _) => *span,
            Pattern::Enum { span, .. } => *span,
            Pattern::Range { span, .. } => *span,
            Pattern::Or { span, .. } => *span,
        }
    }
}

// =============================================================================
// LITERAL VALUES
// =============================================================================

/// A fully-evaluated literal, used in patterns and constant expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum LitVal {
    Int(u128),
    Float(f64),
    Bool(bool),
    Str(String),
    // Inline vec/tensor: `vec3(1.0, 0.0, 0.0)`, `[1, 2, 3]`
    Vec(Vec<LitVal>),
}

// =============================================================================
// EXPRESSIONS
// =============================================================================

/// All Jules expression forms.
/// Every variant carries a `Span` so diagnostics can point at exact source.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // ── Literals ──────────────────────────────────────────────────────────
    IntLit {
        span: Span,
        value: u128,
    },
    FloatLit {
        span: Span,
        value: f64,
    },
    BoolLit {
        span: Span,
        value: bool,
    },
    StrLit {
        span: Span,
        value: String,
    },

    // ── Variable / path ───────────────────────────────────────────────────
    /// `x`, `world`, `dt`
    Ident {
        span: Span,
        name: String,
    },
    /// `std::math::sqrt`
    Path {
        span: Span,
        segments: Vec<String>,
    },

    // ── Tensor / vector constructors ──────────────────────────────────────
    /// `vec3(1.0, 0.0, 0.0)` — vector literal constructor
    VecCtor {
        span: Span,
        size: VecSize,
        elems: Vec<Expr>,
    },
    /// `mat3::identity()` — handled as a call, but the AST tracks the type
    /// `[1.0, 2.0, 3.0, 4.0]` — array/tensor literal
    ArrayLit {
        span: Span,
        elems: Vec<Expr>,
    },

    // ── Arithmetic / logical / bitwise ────────────────────────────────────
    BinOp {
        span: Span,
        op: BinOpKind,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    UnOp {
        span: Span,
        op: UnOpKind,
        expr: Box<Expr>,
    },

    // ── Assignment (used as expression in statement position) ─────────────
    Assign {
        span: Span,
        op: AssignOpKind,
        target: Box<Expr>,
        value: Box<Expr>,
    },

    // ── Field / index access ──────────────────────────────────────────────
    /// `entity.position`, `p.x`
    Field {
        span: Span,
        object: Box<Expr>,
        field: String,
    },
    /// `tensor[i, j]`, `arr[0]`
    Index {
        span: Span,
        object: Box<Expr>,
        indices: Vec<Expr>,
    },

    // ── Calls ─────────────────────────────────────────────────────────────
    /// `f(a, b)`, `sqrt(x)`
    Call {
        span: Span,
        func: Box<Expr>,
        args: Vec<Expr>,
        /// Named arguments: `lerp(a, b, t: 0.5)`
        named: Vec<(String, Expr)>,
    },
    /// `obj.method(args)` — method call syntax
    MethodCall {
        span: Span,
        receiver: Box<Expr>,
        method: String,
        args: Vec<Expr>,
    },

    // ── Tensor-specific (Feature 1) ───────────────────────────────────────
    /// `A @ B` — matrix multiply
    MatMul {
        span: Span,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// `A .* B` — element-wise (Hadamard) multiply
    HadamardMul {
        span: Span,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// `A ./ B` — element-wise divide
    HadamardDiv {
        span: Span,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// `A ++ B` — tensor concatenation along axis 0
    TensorConcat {
        span: Span,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// `kron(A, B)` — Kronecker product
    KronProd {
        span: Span,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// `outer(A, B)` — outer product
    OuterProd {
        span: Span,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// `grad A` — gradient-tracked expression
    Grad {
        span: Span,
        inner: Box<Expr>,
    },
    /// `A ** 2` — scalar/element-wise power
    Pow {
        span: Span,
        base: Box<Expr>,
        exp: Box<Expr>,
    },

    // ── Range ─────────────────────────────────────────────────────────────
    Range {
        span: Span,
        lo: Option<Box<Expr>>,
        hi: Option<Box<Expr>>,
        inclusive: bool,
    },

    // ── Cast ──────────────────────────────────────────────────────────────
    Cast {
        span: Span,
        expr: Box<Expr>,
        ty: Type,
    },

    // ── Conditional (expression form) ────────────────────────────────────
    /// `if cond { a } else { b }`
    IfExpr {
        span: Span,
        cond: Box<Expr>,
        then: Box<Block>,
        else_: Option<Box<Block>>,
    },

    // ── Closures ──────────────────────────────────────────────────────────
    Closure {
        span: Span,
        params: Vec<Param>,
        ret_ty: Option<Type>,
        body: Box<Expr>,
    },

    // ── Block expression ──────────────────────────────────────────────────
    Block(Box<Block>),

    // ── Tuple ─────────────────────────────────────────────────────────────
    Tuple {
        span: Span,
        elems: Vec<Expr>,
    },

    // ── Struct literal ────────────────────────────────────────────────────
    StructLit {
        span: Span,
        name: String,
        fields: Vec<(String, Expr)>,
    },
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::IntLit { span, .. } => *span,
            Expr::FloatLit { span, .. } => *span,
            Expr::BoolLit { span, .. } => *span,
            Expr::StrLit { span, .. } => *span,
            Expr::Ident { span, .. } => *span,
            Expr::Path { span, .. } => *span,
            Expr::VecCtor { span, .. } => *span,
            Expr::ArrayLit { span, .. } => *span,
            Expr::BinOp { span, .. } => *span,
            Expr::UnOp { span, .. } => *span,
            Expr::Assign { span, .. } => *span,
            Expr::Field { span, .. } => *span,
            Expr::Index { span, .. } => *span,
            Expr::Call { span, .. } => *span,
            Expr::MethodCall { span, .. } => *span,
            Expr::MatMul { span, .. } => *span,
            Expr::HadamardMul { span, .. } => *span,
            Expr::HadamardDiv { span, .. } => *span,
            Expr::TensorConcat { span, .. } => *span,
            Expr::KronProd { span, .. } => *span,
            Expr::OuterProd { span, .. } => *span,
            Expr::Grad { span, .. } => *span,
            Expr::Pow { span, .. } => *span,
            Expr::Range { span, .. } => *span,
            Expr::Cast { span, .. } => *span,
            Expr::IfExpr { span, .. } => *span,
            Expr::Closure { span, .. } => *span,
            Expr::Block(b) => b.span,
            Expr::Tuple { span, .. } => *span,
            Expr::StructLit { span, .. } => *span,
        }
    }

    /// Returns the field chain if this expression is a sequence of field
    /// accesses on a single root identifier: `entity.position.x`
    /// → `Some(("entity", vec!["position", "x"]))`
    pub fn as_field_chain(&self) -> Option<(&str, Vec<&str>)> {
        fn collect<'a>(e: &'a Expr, fields: &mut Vec<&'a str>) -> Option<&'a str> {
            match e {
                Expr::Ident { name, .. } => Some(name.as_str()),
                Expr::Field { object, field, .. } => {
                    let root = collect(object, fields)?;
                    fields.push(field.as_str());
                    Some(root)
                }
                _ => None,
            }
        }
        let mut fields = Vec::new();
        let root = collect(self, &mut fields)?;
        Some((root, fields))
    }
}

// ─── Binary Operators ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOpKind {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    FloorDiv,
    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // Logical
    And,
    Or,
    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

// ─── Unary Operators ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnOpKind {
    Neg,    // `-x`
    Not,    // `!x` / `~x`
    Deref,  // `*x`
    Ref,    // `&x`
    RefMut, // `&mut x`
}

// ─── Assignment Operators ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignOpKind {
    Assign,    // =
    AddAssign, // +=
    SubAssign, // -=
    MulAssign, // *=
    DivAssign, // /=
    RemAssign, // %=
    BitAndAssign,
    BitOrAssign,
    BitXorAssign,
    MatMulAssign, // @=
}

impl AssignOpKind {
    /// True if this is a compound assignment (anything except plain `=`).
    pub fn is_compound(self) -> bool {
        !matches!(self, AssignOpKind::Assign)
    }

    /// Return the corresponding binary operator for a compound assignment,
    /// e.g. `+=` → `Add`.
    pub fn to_binop(self) -> Option<BinOpKind> {
        Some(match self {
            AssignOpKind::AddAssign => BinOpKind::Add,
            AssignOpKind::SubAssign => BinOpKind::Sub,
            AssignOpKind::MulAssign => BinOpKind::Mul,
            AssignOpKind::DivAssign => BinOpKind::Div,
            AssignOpKind::RemAssign => BinOpKind::Rem,
            AssignOpKind::BitAndAssign => BinOpKind::BitAnd,
            AssignOpKind::BitOrAssign => BinOpKind::BitOr,
            AssignOpKind::BitXorAssign => BinOpKind::BitXor,
            _ => return None,
        })
    }
}

// =============================================================================
// STATEMENTS
// =============================================================================

/// A single statement. Statements are the basic building block of a `Block`.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    // ── Local binding ─────────────────────────────────────────────────────
    Let {
        span: Span,
        pattern: Pattern,
        ty: Option<Type>,
        init: Option<Expr>,
        mutable: bool,
    },

    // ── Expression statement ──────────────────────────────────────────────
    Expr {
        span: Span,
        expr: Expr,
        /// If true the semicolon was present and the value is discarded.
        has_semi: bool,
    },

    // ── Control flow ──────────────────────────────────────────────────────
    Return {
        span: Span,
        value: Option<Expr>,
    },
    Break {
        span: Span,
        value: Option<Expr>,
        label: Option<String>,
    },
    Continue {
        span: Span,
        label: Option<String>,
    },

    // ── Structured loops ──────────────────────────────────────────────────
    /// `for pat in iter { body }`
    ForIn {
        span: Span,
        pattern: Pattern,
        iter: Expr,
        body: Block,
        label: Option<String>,
    },

    /// `for entity in world { … }` — the game-simulation entity loop.
    ///
    /// Distinguished from `ForIn` because the compiler needs to:
    ///   • infer/check the component query from `body`
    ///   • decide whether iterations are order-independent (→ SIMD / GPU)
    ///   • annotate with `ComponentAccess` during analysis
    ///
    /// The parser initially emits a `ForIn` whose `iter` is an `Ident("world")`
    /// or a `WorldQuery` expression; the analysis pass rewrites it to
    /// `EntityFor`.
    EntityFor {
        span: Span,
        /// The loop variable bound to each entity: `entity`
        var: String,
        /// The query expression. May be:
        ///   • `Ident("world")` — all entities
        ///   • `WorldQuery { with, without, filter }` — refined query
        query: EntityQuery,
        body: Block,
        label: Option<String>,
        /// Filled by analysis: what components the body reads/writes.
        accesses: Vec<ComponentAccess>,
        /// Filled by analysis: the best parallelism strategy for this loop.
        parallelism: ParallelismHint,
    },

    /// `while cond { body }`
    While {
        span: Span,
        cond: Expr,
        body: Block,
        label: Option<String>,
    },

    /// `loop { body }`
    Loop {
        span: Span,
        body: Block,
        label: Option<String>,
    },

    // ── Conditional ───────────────────────────────────────────────────────
    If {
        span: Span,
        cond: Expr,
        then: Block,
        else_: Option<Box<IfOrBlock>>,
    },

    // ── Match ─────────────────────────────────────────────────────────────
    Match {
        span: Span,
        expr: Expr,
        arms: Vec<MatchArm>,
    },

    // ── Item inside a block (nested fn, struct, etc.) ─────────────────────
    Item(Box<Item>),

    // ── Parallelism statements (Feature 4) ───────────────────────────────
    /// `parallel for var in iter { body }`
    ParallelFor(ParallelFor),
    /// `spawn { … }` — fire-and-forget async task
    Spawn(SpawnBlock),
    /// `sync { … }` — barrier that waits for in-flight spawns
    Sync(SyncBlock),
    /// `atomic { … }` — indivisible region
    Atomic(AtomicBlock),
}

/// Used for `else if` and `else { }` chains.
#[derive(Debug, Clone, PartialEq)]
pub enum IfOrBlock {
    If(Stmt), // another If stmt
    Block(Block),
}

/// One arm of a `match` expression.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub span: Span,
    pub pat: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
}

// =============================================================================
// BLOCKS
// =============================================================================

/// A braced sequence of statements with an optional trailing expression.
/// `{ stmt1; stmt2; expr }` — `expr` is the block's value.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub span: Span,
    pub stmts: Vec<Stmt>,
    /// The trailing expression (no semicolon) is the block's value.
    /// `None` if the block returns `()`.
    pub tail: Option<Box<Expr>>,
}

impl Block {
    pub fn new(span: Span) -> Self {
        Block {
            span,
            stmts: vec![],
            tail: None,
        }
    }

    /// True if the block ends with an expression value (not `()`).
    pub fn has_value(&self) -> bool {
        self.tail.is_some()
    }
}

// =============================================================================
// GAME SIMULATION — ENTITY QUERIES & COMPONENT ACCESS
// =============================================================================

/// Describes *which entities* to iterate over in an `EntityFor` loop.
///
/// Example sources:
/// ```text
/// for entity in world                              // all entities
/// for entity in world with (Position, Velocity)    // must have both
/// for entity in world with (Position) without (Dead) // filtered
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct EntityQuery {
    pub span: Span,
    /// Component types that must be present (AND logic).
    pub with: Vec<String>,
    /// Component types that must be absent.
    pub without: Vec<String>,
    /// An optional boolean filter expression evaluated per entity.
    /// Example: `entity.health > 0.0`
    pub filter: Option<Box<Expr>>,
}

impl EntityQuery {
    /// A bare `world` query — no filters at all.
    pub fn all(span: Span) -> Self {
        EntityQuery {
            span,
            with: vec![],
            without: vec![],
            filter: None,
        }
    }

    pub fn is_unconstrained(&self) -> bool {
        self.with.is_empty() && self.without.is_empty() && self.filter.is_none()
    }
}

// -----------------------------------------------------------------------------

/// How a system accesses a single component type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    /// Only reads the component (field appears only on the RHS).
    Read,
    /// Writes to the component (field appears as assignment target).
    Write,
    /// Both reads and writes (e.g., `+=`).
    ReadWrite,
}

impl AccessMode {
    pub fn merge(self, other: AccessMode) -> AccessMode {
        match (self, other) {
            (AccessMode::Read, AccessMode::Read) => AccessMode::Read,
            (AccessMode::Write, AccessMode::Write) => AccessMode::Write,
            _ => AccessMode::ReadWrite,
        }
    }

    pub fn is_write(self) -> bool {
        matches!(self, AccessMode::Write | AccessMode::ReadWrite)
    }
}

/// Records that a system (or entity loop body) accesses a particular component.
/// Populated by the component-access analysis pass.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentAccess {
    /// The component type name: `"Position"`, `"Velocity"`, …
    pub component: String,
    /// How the system uses it.
    pub mode: AccessMode,
    /// The field name used on the entity variable: `"position"`, `"velocity"`.
    /// Useful when the field name differs from the component type name.
    pub field_alias: String,
}

// -----------------------------------------------------------------------------

/// What parallelism strategy the compiler should apply to an `EntityFor` loop.
///
/// The analysis pass infers this from:
///   • Attribute annotations (`@simd`, `@parallel`, `@gpu`, `@seq`)
///   • Whether any two iterations could alias (i.e., write to a component
///     that another iteration reads)
///   • The number of entities (GPU dispatch only pays off at large N)
///   • The inner operation type (vec math → SIMD, tensor → GPU)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelismHint {
    /// Let the compiler decide at code-generation time.
    Auto,
    /// Force single-thread, in-order execution.
    /// Guarantees bit-identical results across runs.
    Sequential,
    /// Emit a multi-threaded loop (e.g., Rayon).
    /// Deterministic iff all iterations are independent.
    Parallel,
    /// Auto-vectorise using SIMD intrinsics (AVX2/NEON/etc.).
    Simd,
    /// Dispatch to the GPU as a compute shader / CUDA kernel.
    Gpu,
    /// Try SIMD first; fall back to GPU above a threshold entity count.
    SimdOrGpu { threshold: u64 },
}

impl Default for ParallelismHint {
    fn default() -> Self {
        ParallelismHint::Auto
    }
}

// =============================================================================
// FUNCTION PARAMETERS & RETURN TYPES
// =============================================================================

/// A single function / system parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub span: Span,
    pub name: String,
    pub ty: Option<Type>,
    pub default: Option<Expr>,
    pub mutable: bool,
}

impl Param {
    pub fn simple(span: Span, name: impl Into<String>, ty: Type) -> Self {
        Param {
            span,
            name: name.into(),
            ty: Some(ty),
            default: None,
            mutable: false,
        }
    }
}

// =============================================================================
// TOP-LEVEL DECLARATIONS
// =============================================================================

// ─── Function ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct FnDecl {
    pub span: Span,
    pub attrs: Vec<Attribute>,
    pub name: String,
    pub generics: Vec<GenericParam>,
    pub params: Vec<Param>,
    pub ret_ty: Option<Type>,
    pub body: Option<Block>, // None for extern / trait declaration
    pub is_async: bool,
}

// ─── System (Feature 2) ──────────────────────────────────────────────────────

/// A Jules game-simulation system declaration.
///
/// ```text
/// @simd
/// system Update(dt: f32):
///     for entity in world with (Position, Velocity):
///         entity.position += entity.velocity * dt
/// ```
///
/// The compiler pipeline for a `SystemDecl`:
///
///   parse → SystemDecl (accesses = [], parallelism = Auto)
///         ↓
///   analysis pass → fills `accesses`, checks for aliasing
///         ↓
///   lowering pass → applies `parallelism` hint, emits SIMD / GPU IR
#[derive(Debug, Clone, PartialEq)]
pub struct SystemDecl {
    pub span: Span,
    pub attrs: Vec<Attribute>,
    pub name: String,
    /// Parameters to the system, e.g. `(dt: f32)`.
    /// These become uniform / push-constant values in GPU codegen.
    pub params: Vec<Param>,
    /// An optional compile-time query annotation.
    /// If `None`, the analysis pass infers the query from the body.
    ///
    /// Explicit form:
    /// ```text
    /// system Update(dt: f32) query(Position, Velocity) without(Dead):
    /// ```
    pub explicit_query: Option<EntityQuery>,
    pub body: Block,
    // ── Filled by analysis pass ──────────────────────────────────────────
    /// All component accesses inferred from the body.
    pub accesses: Vec<ComponentAccess>,
    /// The chosen parallelism strategy.
    pub parallelism: ParallelismHint,
    /// True when the compiler has proved iterations are order-independent.
    /// Enables deterministic-parallel execution.
    pub iterations_independent: bool,
}

impl SystemDecl {
    pub fn new(span: Span, name: impl Into<String>, params: Vec<Param>, body: Block) -> Self {
        SystemDecl {
            span,
            attrs: vec![],
            name: name.into(),
            params,
            explicit_query: None,
            body,
            accesses: vec![],
            parallelism: ParallelismHint::Auto,
            iterations_independent: false,
        }
    }

    /// Returns the effective parallelism: attribute annotations take priority
    /// over the analysis result.
    pub fn effective_parallelism(&self) -> ParallelismHint {
        for attr in &self.attrs {
            match attr {
                Attribute::Gpu => return ParallelismHint::Gpu,
                Attribute::Simd => return ParallelismHint::Simd,
                Attribute::Parallel => return ParallelismHint::Parallel,
                Attribute::Seq => return ParallelismHint::Sequential,
                _ => {}
            }
        }
        self.parallelism
    }

    /// Returns true if any component is written by this system, used to check
    /// for data races when multiple systems run in parallel.
    pub fn has_writes(&self) -> bool {
        self.accesses.iter().any(|a| a.mode.is_write())
    }
}

// ─── Struct ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct StructField {
    pub span: Span,
    pub name: String,
    pub ty: Type,
    pub attrs: Vec<Attribute>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructDecl {
    pub span: Span,
    pub attrs: Vec<Attribute>,
    pub name: String,
    pub generics: Vec<GenericParam>,
    pub fields: Vec<StructField>,
}

// ─── Component (Feature 2) ────────────────────────────────────────────────────

/// An ECS component declaration.
///
/// ```text
/// component Position { x: f32, y: f32, z: f32 }
/// component Velocity { x: f32, y: f32, z: f32 }
/// component Health   { value: f32 }
/// component Dead     {}    // marker component (zero-sized)
/// ```
///
/// Components are semantically identical to structs but the compiler knows
/// they live in component arrays (SoA layout by default) and emits
/// SIMD-friendly load/store patterns for them.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentDecl {
    pub span: Span,
    pub attrs: Vec<Attribute>,
    pub name: String,
    pub fields: Vec<StructField>,
    /// Preferred memory layout for the component array.
    pub layout: ComponentLayout,
}

/// How the component data is laid out in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentLayout {
    /// Structure-of-Arrays: `positions_x[], positions_y[], positions_z[]`
    /// Best for SIMD: loads consecutive floats for each field.
    Soa,
    /// Array-of-Structures: `[{x,y,z}, {x,y,z}, …]`
    /// Easier to reason about but less SIMD-friendly.
    Aos,
}

impl Default for ComponentLayout {
    fn default() -> Self {
        ComponentLayout::Soa
    }
}

// ─── Enum ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub span: Span,
    pub name: String,
    pub fields: EnumVariantFields,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EnumVariantFields {
    Unit,
    Tuple(Vec<Type>),
    Struct(Vec<StructField>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumDecl {
    pub span: Span,
    pub attrs: Vec<Attribute>,
    pub name: String,
    pub generics: Vec<GenericParam>,
    pub variants: Vec<EnumVariant>,
}

// ─── Const / static ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct ConstDecl {
    pub span: Span,
    pub name: String,
    pub ty: Type,
    pub value: Expr,
    pub is_pub: bool,
}

// ─── Generics ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct GenericParam {
    pub span: Span,
    pub name: String,
    pub bounds: Vec<String>, // trait bounds, simplified for now
}

// ─── Use / module ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct UsePath {
    pub span: Span,
    pub segments: Vec<String>,
    pub alias: Option<String>,
    pub is_glob: bool,
}

// =============================================================================
// SHADER DECLARATIONS
// =============================================================================

/// Shader binding kind (uniform, buffer, sampler, texture).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderBindingKind {
    Uniform,
    Buffer,
    Storage,
    Sampler,
    Texture,
}

/// A single shader binding (uniform, buffer, sampler, or texture).
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderBinding {
    pub span: Span,
    pub kind: ShaderBindingKind,
    pub name: String,
    pub ty: Type,
    pub binding_idx: Option<u64>,
}

/// `shader Name { vertex { … } fragment { … } compute { … } }`
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderDecl {
    pub span: Span,
    pub attrs: Vec<Attribute>,
    pub name: String,
    pub bindings: Vec<ShaderBinding>,
    pub vertex: Option<Box<Block>>,
    pub fragment: Option<Box<Block>>,
    pub compute: Option<Box<Block>>,
}

// =============================================================================
// SCENE DECLARATIONS
// =============================================================================

/// An instance of a prefab in a scene.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneInstance {
    pub span: Span,
    pub prefab: String,
    pub overrides: Vec<(String, Expr)>,
}

/// `scene Name { PrefabName { field: val }, … }`
#[derive(Debug, Clone, PartialEq)]
pub struct SceneDecl {
    pub span: Span,
    pub attrs: Vec<Attribute>,
    pub name: String,
    pub instances: Vec<SceneInstance>,
}

// =============================================================================
// PREFAB DECLARATIONS
// =============================================================================

/// A component within a prefab.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefabComponent {
    pub span: Span,
    pub name: String,
    pub fields: Vec<(String, Expr)>,
}

/// `prefab Name { component ComponentType { field: val }, … }`
#[derive(Debug, Clone, PartialEq)]
pub struct PrefabDecl {
    pub span: Span,
    pub attrs: Vec<Attribute>,
    pub name: String,
    pub components: Vec<PrefabComponent>,
}

// =============================================================================
// PHYSICS CONFIGURATION
// =============================================================================

/// `physics { gravity: expr, iterations: u64, substeps: u64, … }`
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsConfigDecl {
    pub span: Span,
    pub attrs: Vec<Attribute>,
    pub gravity: Option<Expr>,
    pub iterations: Option<u64>,
    pub substeps: Option<u64>,
    pub collision_layers: Vec<String>,
}

// =============================================================================
// LOSS FUNCTION DECLARATIONS
// =============================================================================

/// `loss LossName { fn forward(pred, target) -> f32 { … } }`
#[derive(Debug, Clone, PartialEq)]
pub struct LossDecl {
    pub span: Span,
    pub attrs: Vec<Attribute>,
    pub name: String,
    pub methods: Vec<FnDecl>,
}

// =============================================================================
// TOP-LEVEL ITEM
// =============================================================================

/// Everything that can appear at the top level of a Jules source file.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Fn(FnDecl),
    /// `system Update(dt: f32): …`
    System(SystemDecl),
    Struct(StructDecl),
    /// `component Position { x: f32, y: f32, z: f32 }`
    Component(ComponentDecl),
    Enum(EnumDecl),
    Const(ConstDecl),
    Use(UsePath),
    /// `agent Warden: …`
    Agent(AgentDecl),
    /// `model PolicyNet: …`
    Model(ModelDecl),
    /// `train Warden in HollowFacility: …`
    Train(TrainDecl),
    /// `shader Name { … }` — shader definition
    Shader(ShaderDecl),
    /// `scene Name { … }` — scene/level definition
    Scene(SceneDecl),
    /// `prefab Name { … }` — prefab template
    Prefab(PrefabDecl),
    /// `physics { … }` — physics configuration
    PhysicsConfig(PhysicsConfigDecl),
    /// `loss Name { … }` — custom loss function
    Loss(LossDecl),
    /// Module declaration: `mod physics;` or `mod physics { … }`
    Mod {
        span: Span,
        name: String,
        items: Option<Vec<Item>>,
        is_pub: bool,
    },
}

impl Item {
    pub fn span(&self) -> Span {
        match self {
            Item::Fn(f) => f.span,
            Item::System(s) => s.span,
            Item::Struct(s) => s.span,
            Item::Component(c) => c.span,
            Item::Enum(e) => e.span,
            Item::Const(c) => c.span,
            Item::Use(u) => u.span,
            Item::Agent(a) => a.span,
            Item::Model(m) => m.span,
            Item::Train(t) => t.span,
            Item::Shader(s) => s.span,
            Item::Scene(s) => s.span,
            Item::Prefab(p) => p.span,
            Item::PhysicsConfig(p) => p.span,
            Item::Loss(l) => l.span,
            Item::Mod { span, .. } => *span,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Item::Fn(f) => &f.name,
            Item::System(s) => &s.name,
            Item::Struct(s) => &s.name,
            Item::Component(c) => &c.name,
            Item::Enum(e) => &e.name,
            Item::Const(c) => &c.name,
            Item::Use(_) => "<use>",
            Item::Agent(a) => &a.name,
            Item::Model(m) => &m.name,
            Item::Train(t) => &t.agent,
            Item::Shader(s) => &s.name,
            Item::Scene(s) => &s.name,
            Item::Prefab(p) => &p.name,
            Item::PhysicsConfig(_) => "<physics>",
            Item::Loss(l) => &l.name,
            Item::Mod { name, .. } => name,
        }
    }
}

// =============================================================================
// PARALLELISM — FEATURE 4
// =============================================================================

/// The scheduling strategy for a `parallel for` loop or spawn block.
///
/// Inferred by the compiler from data-dependency analysis; can be
/// overridden by attributes (`@gpu`, `@simd`, `@seq`, …).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleKind {
    /// Compiler chooses automatically.
    Auto,
    /// One OS thread per chunk (work-stealing pool).
    ThreadPool,
    /// SIMD lanes (auto-vectorise).
    Simd,
    /// GPU compute dispatch.
    Gpu,
    /// Sequential (deterministic fallback / debug mode).
    Sequential,
    /// Static chunk size: `parallel(chunk=64) for …`
    StaticChunk(u64),
}

impl Default for ScheduleKind {
    fn default() -> Self {
        ScheduleKind::Auto
    }
}

/// `parallel for var in iter { body }`
///
/// Semantics: every iteration is independent; the compiler may execute them
/// in any order on any available compute resource.
///
/// The `schedule` is filled by the analysis pass; it can be overridden by
/// placing `@gpu`, `@simd`, or `@seq` before the statement.
#[derive(Debug, Clone, PartialEq)]
pub struct ParallelFor {
    pub span: Span,
    pub attrs: Vec<Attribute>,
    pub var: Pattern,
    pub iter: Expr,
    pub body: Block,
    pub label: Option<String>,
    /// Chunk size hint: `parallel(chunk=64) for …`
    pub chunk: Option<u64>,
    /// Filled by analysis.
    pub schedule: ScheduleKind,
}

/// `spawn { body }` — fire-and-forget lightweight task.
///
/// The body runs asynchronously.  Any values captured from the outer scope
/// are moved unless they are `Copy`.
#[derive(Debug, Clone, PartialEq)]
pub struct SpawnBlock {
    pub span: Span,
    pub attrs: Vec<Attribute>,
    pub body: Block,
    /// Optional name for diagnostics / cancellation: `spawn("physics") { … }`
    pub name: Option<String>,
}

/// `sync { body }` — barrier that waits for all in-flight spawned tasks
/// created inside `body` to complete before continuing.
#[derive(Debug, Clone, PartialEq)]
pub struct SyncBlock {
    pub span: Span,
    pub body: Block,
}

/// `atomic { body }` — executes `body` as an indivisible unit.
/// Maps to a hardware lock / compare-and-swap region.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomicBlock {
    pub span: Span,
    pub body: Block,
}

// =============================================================================
// AI BEHAVIOUR SYSTEMS — FEATURE 3
// =============================================================================

// ─── Perception ──────────────────────────────────────────────────────────────

/// The kind of perception sensor available to an agent.
#[derive(Debug, Clone, PartialEq)]
pub enum PerceptionKind {
    /// Cone / sphere vision: `perception vision 40`
    Vision,
    /// Sound / proximity: `perception hearing 20`
    Hearing,
    /// Direct knowledge (shared blackboard): `perception omniscient`
    Omniscient,
    /// Named custom sensor: `perception custom "NavMesh"`
    Custom(String),
}

/// A perception declaration inside an agent body.
///
/// ```text
/// perception vision 40        // 40-unit range
/// perception hearing 15       // 15-unit radius
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PerceptionSpec {
    pub span: Span,
    pub kind: PerceptionKind,
    /// Sensing range (world units).  `None` means unlimited.
    pub range: Option<f64>,
    /// Field-of-view angle in degrees (vision only).
    pub fov: Option<f64>,
    /// Optional custom tag string.
    pub tag: Option<String>,
}

// ─── Memory ───────────────────────────────────────────────────────────────────

/// The memory architecture used by an agent.
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryKind {
    /// Episodic memory: stores timestamped event records.
    /// `memory episodic 120s`
    Episodic,
    /// Semantic memory: stores declarative facts (KV store).
    /// `memory semantic`
    Semantic,
    /// Working memory: fixed-size sliding window.
    /// `memory working 32`
    Working,
    /// External memory: backed by a tensor/embedding store.
    /// `memory external tensor<f32>[512, 128]`
    External(Box<Type>),
}

/// A memory configuration declaration inside an agent body.
#[derive(Debug, Clone, PartialEq)]
pub struct MemorySpec {
    pub span: Span,
    pub kind: MemoryKind,
    /// Retention window (seconds) or slot count, depending on kind.
    pub capacity: Option<MemoryCapacity>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryCapacity {
    /// Time-based: `120s`, `30m`
    Duration { seconds: f64 },
    /// Slot-based: `working 32`
    Slots(u64),
}

// ─── Learning ─────────────────────────────────────────────────────────────────

/// The learning strategy wired into an agent.
#[derive(Debug, Clone, PartialEq)]
pub enum LearningKind {
    /// Classic Q-learning / deep-RL.
    Reinforcement,
    /// Imitation learning from demonstration data.
    Imitation,
    /// Evolutionary / neuroevolution.
    Evolutionary,
    /// No online learning; agent behaviour is fully scripted.
    None,
    /// A named custom learning algorithm (plugin/extern).
    Custom(String),
}

/// A learning configuration declaration.
///
/// ```text
/// learning reinforcement
/// learning imitation
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LearningSpec {
    pub span: Span,
    pub kind: LearningKind,
    /// Learning rate (overrides default).
    pub learning_rate: Option<f64>,
    /// Discount factor γ (RL).
    pub gamma: Option<f64>,
    /// Reference to a `model` declaration used as the policy network.
    pub policy_model: Option<String>,
}

// ─── Behavior Rules ────────────────────────────────────────────────────────

/// Priority of a behaviour rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BehaviorPriority(pub u32);

/// A named, prioritised behaviour rule inside an agent.
///
/// ```text
/// behavior Flee(priority: 100):
///     if self.health < 20:
///         return seek(exit)
/// behavior Patrol(priority: 10):
///     return follow(patrol_path)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BehaviorRule {
    pub span: Span,
    pub name: String,
    pub priority: BehaviorPriority,
    pub params: Vec<Param>,
    pub body: Block,
}

// ─── Goal ─────────────────────────────────────────────────────────────────────

/// A declared goal for the agent to pursue.
///
/// Goals are evaluated by the decision-making layer (utility / planner).
/// ```text
/// goal Survive:
///     utility = self.health / 100.0
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GoalDecl {
    pub span: Span,
    pub name: String,
    /// An expression in `[0, 1]` representing how strongly the agent wants
    /// to pursue this goal right now.
    pub utility: Expr,
}

// ─── Agent Declaration ────────────────────────────────────────────────────────

/// Selects which decision-making architecture the agent uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentArchitecture {
    /// Finite-state machine (default).
    Fsm,
    /// Behaviour tree.
    BehaviorTree,
    /// Utility-based (scores all available behaviours each tick).
    Utility,
    /// Goal-oriented action planning (GOAP).
    Goap,
    /// Fully learned policy (uses the attached `learning` spec).
    Learned,
}

impl Default for AgentArchitecture {
    fn default() -> Self {
        AgentArchitecture::Utility
    }
}

/// A Jules AI agent declaration.
///
/// ```text
/// agent Warden:
///     perception vision 40
///     memory episodic 120s
///     learning reinforcement
///     behavior Flee(priority: 100):
///         if self.health < 20.0:
///             return seek(exit)
///     behavior Patrol(priority: 10):
///         return follow(patrol_path)
/// ```
///
/// The compiler lowers an `AgentDecl` to:
///   • a struct with sensor / memory fields
///   • an update function that ticks perception, memory decay, and the
///     decision architecture
///   • a learning update step (if `learning` is not `None`)
#[derive(Debug, Clone, PartialEq)]
pub struct AgentDecl {
    pub span: Span,
    pub attrs: Vec<Attribute>,
    pub name: String,
    /// Which decision-making architecture to use.
    pub architecture: AgentArchitecture,
    /// Perception sensors attached to this agent.
    pub perceptions: Vec<PerceptionSpec>,
    /// Memory subsystem (at most one per agent).
    pub memory: Option<MemorySpec>,
    /// Online learning configuration (optional).
    pub learning: Option<LearningSpec>,
    /// Prioritised behaviour rules, evaluated top-down each tick.
    pub behaviors: Vec<BehaviorRule>,
    /// Agent goals (utility / GOAP).
    pub goals: Vec<GoalDecl>,
    /// Additional named fields (agent state variables).
    pub fields: Vec<StructField>,
}

impl AgentDecl {
    /// Returns true if this agent has any online learning capability.
    pub fn is_learnable(&self) -> bool {
        matches!(
            &self.learning,
            Some(s) if !matches!(s.kind, LearningKind::None)
        )
    }

    /// Returns the highest-priority behaviour rule.
    pub fn top_behavior(&self) -> Option<&BehaviorRule> {
        self.behaviors.iter().max_by_key(|b| b.priority)
    }
}

// =============================================================================
// NEURAL-NETWORK MODEL — UNIQUE FEATURE 1
// =============================================================================

/// Activation function for a neural-network layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Activation {
    Relu,
    LeakyRelu,
    Elu,
    Swish,
    Mish,
    Sigmoid,
    Tanh,
    Gelu,
    Silu,
    Softmax,
    /// No activation / identity.
    Linear,
    /// Named custom activation (user-defined or from a plugin).
    Custom(String),
}

impl Default for Activation {
    fn default() -> Self {
        Activation::Linear
    }
}

/// Padding mode for convolution layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    Same,
    Valid,
}

/// Pooling operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolOp {
    Max,
    Average,
    GlobalMax,
    GlobalAverage,
}

/// A single layer in a `ModelDecl`.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelLayer {
    /// `input 128` — declares input shape
    Input { span: Span, size: u64 },

    /// `dense 256 relu` — fully-connected
    Dense {
        span: Span,
        units: u64,
        activation: Activation,
        /// If `true`, the layer has a bias term.
        bias: bool,
    },

    /// `conv 32 3x3 relu` — 2-D convolution
    Conv2d {
        span: Span,
        filters: u64,
        kernel_h: u64,
        kernel_w: u64,
        stride: u64,
        padding: Padding,
        activation: Activation,
    },

    /// `pool 2x2 max` — pooling
    Pool {
        span: Span,
        size_h: u64,
        size_w: u64,
        op: PoolOp,
    },

    /// `recurrent 128 lstm` or `recurrent 128 gru`
    Recurrent {
        span: Span,
        units: u64,
        cell: RecurrentCell,
        bidirect: bool,
    },

    /// `attention 8 64` — multi-head self-attention (heads, dim_per_head)
    Attention {
        span: Span,
        num_heads: u64,
        head_dim: u64,
    },

    /// `embed 10000 128` — embedding table (vocab_size, embed_dim)
    Embed {
        span: Span,
        vocab_size: u64,
        embed_dim: u64,
    },

    /// `dropout 0.1` — dropout with keep probability
    Dropout { span: Span, rate: f64 },

    /// `norm batch` / `norm layer` / `norm rms`
    Norm { span: Span, kind: NormKind },

    /// `output 12 softmax` — final layer
    Output {
        span: Span,
        units: u64,
        activation: Activation,
    },

    /// `residual { dense 128, dense 128 }` — residual/skip connection
    Residual { span: Span, layers: Vec<ModelLayer> },

    /// `flatten` — reshape tensor to 1D
    Flatten { span: Span },

    /// A named sub-model used as a layer: `residual_block`
    SubModel { span: Span, name: String },
}

impl ModelLayer {
    pub fn span(&self) -> Span {
        match self {
            ModelLayer::Input { span, .. } => *span,
            ModelLayer::Dense { span, .. } => *span,
            ModelLayer::Conv2d { span, .. } => *span,
            ModelLayer::Pool { span, .. } => *span,
            ModelLayer::Recurrent { span, .. } => *span,
            ModelLayer::Attention { span, .. } => *span,
            ModelLayer::Embed { span, .. } => *span,
            ModelLayer::Dropout { span, .. } => *span,
            ModelLayer::Norm { span, .. } => *span,
            ModelLayer::Output { span, .. } => *span,
            ModelLayer::Residual { span, .. } => *span,
            ModelLayer::Flatten { span, .. } => *span,
            ModelLayer::SubModel { span, .. } => *span,
        }
    }

    /// Total number of trainable parameters (approximate, shapes may be dynamic).
    pub fn approx_params(&self) -> Option<u64> {
        match self {
            ModelLayer::Dense { units, .. } => Some(*units), // input_dim * units + units
            ModelLayer::Output { units, .. } => Some(*units),
            _ => None,
        }
    }
}

/// RNN cell type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecurrentCell {
    Lstm,
    Gru,
    SimpleRnn,
}

/// Normalisation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    Batch,
    Layer,
    Rms,
    Group,
}

/// A named neural-network model declaration.
///
/// ```text
/// model PolicyNet:
///     input 128
///     dense 256 relu
///     dense 256 relu
///     output 12 softmax
/// ```
///
/// The compiler lowers a `ModelDecl` to:
///   • Weight tensor allocations (on the chosen device)
///   • A forward-pass function
///   • Gradient buffers (if `@grad` is present)
///   • Optional CUDA/Metal kernel code
#[derive(Debug, Clone, PartialEq)]
pub struct ModelDecl {
    pub span: Span,
    pub attrs: Vec<Attribute>,
    pub name: String,
    pub layers: Vec<ModelLayer>,
    /// Device the model is initialised on.
    pub device: ModelDevice,
    /// Training configuration (optional; set via `train` block).
    pub optimizer: Option<OptimizerSpec>,
}

/// Where model weights live at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelDevice {
    Cpu,
    Gpu,
    Auto,
}
impl Default for ModelDevice {
    fn default() -> Self {
        ModelDevice::Auto
    }
}

/// Optimizer specification attached to a model.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerSpec {
    pub span: Span,
    pub kind: OptimizerKind,
    pub learning_rate: f64,
    pub extra: Vec<(String, f64)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerKind {
    Adam,
    Sgd,
    Rmsprop,
    Adagrad,
    AdamW,
    Lion,
    Sophia,
    Prodigy,
}

impl ModelDecl {
    /// Count layers (excluding Input/Output markers).
    pub fn hidden_layer_count(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| !matches!(l, ModelLayer::Input { .. } | ModelLayer::Output { .. }))
            .count()
    }

    /// True if a gradient attribute is set (model can be trained).
    pub fn is_trainable(&self) -> bool {
        self.attrs.iter().any(|a| matches!(a, Attribute::Grad))
    }
}

// =============================================================================
// SIMULATION TRAINING — UNIQUE FEATURE 2
// =============================================================================

/// A single reward or penalty signal inside a `train` block.
///
/// ```text
/// reward survive   // positive signal when the agent is alive
/// penalty seen     // negative signal when the agent is spotted
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SignalSpec {
    pub span: Span,
    /// `true` = reward, `false` = penalty
    pub is_reward: bool,
    /// The name of the signal, matches an event emitted by the world.
    pub name: String,
    /// Weight multiplier (default 1.0).
    pub weight: f64,
    /// Optional scalar expression for dynamic weighting.
    pub expr: Option<Expr>,
}

/// Episode termination / truncation conditions.
#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeSpec {
    pub span: Span,
    /// Maximum number of simulation ticks per episode.
    pub max_steps: Option<u64>,
    /// Maximum real-time seconds per episode.
    pub max_seconds: Option<f64>,
    /// Condition expression that terminates the episode early.
    pub done_condition: Option<Expr>,
    /// Number of parallel environment instances to run.
    pub num_envs: Option<u64>,
}

/// `train AgentName in WorldName { … }` declaration.
///
/// Describes how to train an agent in a simulation world.
/// The compiler generates:
///   • An environment reset function
///   • A step function that runs one simulation tick
///   • A reward aggregation loop
///   • A gradient-based policy update (using the agent's `learning` spec)
///
/// ```text
/// train Warden in HollowFacility:
///     reward survive
///     penalty seen
///     episode max_steps: 2000, num_envs: 64
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TrainDecl {
    pub span: Span,
    pub attrs: Vec<Attribute>,
    /// The agent type to train.
    pub agent: String,
    /// The simulation world / level to train in.
    pub world: String,
    /// Reward and penalty signal specifications.
    pub signals: Vec<SignalSpec>,
    /// Episode termination configuration.
    pub episode: Option<EpisodeSpec>,
    /// Which model to train (defaults to the agent's `learning.policy_model`).
    pub model: Option<String>,
    /// Optimizer (if not set on the model itself).
    pub optimizer: Option<OptimizerSpec>,
    /// Extra hyperparameters as key/value expressions.
    pub hyper: Vec<(String, Expr)>,
    /// RL algorithm: "ppo", "dqn", "sac", "reinforce", etc.
    pub algorithm: Option<String>,
    /// Value model for dueling/two-headed architectures.
    pub value_model: Option<String>,
}

impl TrainDecl {
    pub fn total_reward_weight(&self) -> f64 {
        self.signals
            .iter()
            .filter(|s| s.is_reward)
            .map(|s| s.weight)
            .sum()
    }

    pub fn total_penalty_weight(&self) -> f64 {
        self.signals
            .iter()
            .filter(|s| !s.is_reward)
            .map(|s| s.weight)
            .sum()
    }
}

// =============================================================================
// PROGRAM ROOT
// =============================================================================

/// The root of a Jules source file's AST.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// The full byte span of the file.
    pub span: Span,
    /// All top-level items in declaration order.
    pub items: Vec<Item>,
}

impl Program {
    pub fn new() -> Self {
        Program {
            span: Span::dummy(),
            items: vec![],
        }
    }

    /// Iterate over all system declarations.
    pub fn systems(&self) -> impl Iterator<Item = &SystemDecl> {
        self.items.iter().filter_map(|i| match i {
            Item::System(s) => Some(s),
            _ => None,
        })
    }

    /// Iterate over all component declarations.
    pub fn components(&self) -> impl Iterator<Item = &ComponentDecl> {
        self.items.iter().filter_map(|i| match i {
            Item::Component(c) => Some(c),
            _ => None,
        })
    }

    /// Iterate over all agent declarations.
    pub fn agents(&self) -> impl Iterator<Item = &AgentDecl> {
        self.items.iter().filter_map(|i| match i {
            Item::Agent(a) => Some(a),
            _ => None,
        })
    }

    /// Iterate over all model declarations.
    pub fn models(&self) -> impl Iterator<Item = &ModelDecl> {
        self.items.iter().filter_map(|i| match i {
            Item::Model(m) => Some(m),
            _ => None,
        })
    }

    /// Iterate over all training session declarations.
    pub fn trains(&self) -> impl Iterator<Item = &TrainDecl> {
        self.items.iter().filter_map(|i| match i {
            Item::Train(t) => Some(t),
            _ => None,
        })
    }
}

// =============================================================================
// VISITOR TRAIT
// =============================================================================

/// A traversal visitor over the Jules AST.
///
/// Default implementations recurse into children via the `walk_*` helpers.
/// Override any method to intercept that node type; call the corresponding
/// `walk_*` function if you still want recursion.
pub trait Visitor: Sized {
    fn visit_program(&mut self, prog: &Program) {
        walk_program(self, prog);
    }
    fn visit_item(&mut self, item: &Item) {
        walk_item(self, item);
    }
    fn visit_fn(&mut self, f: &FnDecl) {
        walk_fn(self, f);
    }
    fn visit_system(&mut self, s: &SystemDecl) {
        walk_system(self, s);
    }
    fn visit_struct(&mut self, s: &StructDecl) {
        walk_struct(self, s);
    }
    fn visit_component(&mut self, c: &ComponentDecl) {
        walk_component(self, c);
    }
    fn visit_agent(&mut self, a: &AgentDecl) {
        walk_agent(self, a);
    }
    fn visit_model(&mut self, m: &ModelDecl) {
        walk_model(self, m);
    }
    fn visit_train(&mut self, t: &TrainDecl) {
        walk_train(self, t);
    }
    fn visit_block(&mut self, b: &Block) {
        walk_block(self, b);
    }
    fn visit_stmt(&mut self, s: &Stmt) {
        walk_stmt(self, s);
    }
    fn visit_expr(&mut self, e: &Expr) {
        walk_expr(self, e);
    }
    fn visit_type(&mut self, _ty: &Type) {}
}

// ─── Walk functions (default recursion) ──────────────────────────────────────

pub fn walk_program<V: Visitor>(v: &mut V, prog: &Program) {
    for item in &prog.items {
        v.visit_item(item);
    }
}

pub fn walk_item<V: Visitor>(v: &mut V, item: &Item) {
    match item {
        Item::Fn(f) => v.visit_fn(f),
        Item::System(s) => v.visit_system(s),
        Item::Struct(s) => v.visit_struct(s),
        Item::Component(c) => v.visit_component(c),
        Item::Agent(a) => v.visit_agent(a),
        Item::Model(m) => v.visit_model(m),
        Item::Train(t) => v.visit_train(t),
        Item::Enum(_) => {}
        Item::Const(_) => {}
        Item::Use(_) => {}
        Item::Mod {
            items: Some(items), ..
        } => {
            for i in items {
                v.visit_item(i);
            }
        }
        Item::Mod { .. } => {}
        _ => {}
    }
}

pub fn walk_fn<V: Visitor>(v: &mut V, f: &FnDecl) {
    if let Some(body) = &f.body {
        v.visit_block(body);
    }
}

pub fn walk_system<V: Visitor>(v: &mut V, s: &SystemDecl) {
    v.visit_block(&s.body);
}

pub fn walk_struct<V: Visitor>(v: &mut V, s: &StructDecl) {
    for field in &s.fields {
        v.visit_type(&field.ty);
    }
}

pub fn walk_component<V: Visitor>(v: &mut V, c: &ComponentDecl) {
    for field in &c.fields {
        v.visit_type(&field.ty);
    }
}

pub fn walk_agent<V: Visitor>(v: &mut V, a: &AgentDecl) {
    for field in &a.fields {
        v.visit_type(&field.ty);
    }
    for rule in &a.behaviors {
        v.visit_block(&rule.body);
    }
    for goal in &a.goals {
        v.visit_expr(&goal.utility);
    }
}

pub fn walk_model<V: Visitor>(_v: &mut V, _m: &ModelDecl) {
    // Layers are leaf data; no sub-expressions to recurse into by default.
}

pub fn walk_train<V: Visitor>(v: &mut V, t: &TrainDecl) {
    for sig in &t.signals {
        if let Some(e) = &sig.expr {
            v.visit_expr(e);
        }
    }
    if let Some(ep) = &t.episode {
        if let Some(e) = &ep.done_condition {
            v.visit_expr(e);
        }
    }
    for (_, e) in &t.hyper {
        v.visit_expr(e);
    }
}

pub fn walk_block<V: Visitor>(v: &mut V, b: &Block) {
    for stmt in &b.stmts {
        v.visit_stmt(stmt);
    }
    if let Some(tail) = &b.tail {
        v.visit_expr(tail);
    }
}

pub fn walk_stmt<V: Visitor>(v: &mut V, s: &Stmt) {
    match s {
        Stmt::Let { init, .. } => {
            if let Some(e) = init {
                v.visit_expr(e);
            }
        }
        Stmt::Expr { expr, .. } => v.visit_expr(expr),
        Stmt::Return { value, .. } => {
            if let Some(e) = value {
                v.visit_expr(e);
            }
        }
        Stmt::ForIn { iter, body, .. } => {
            v.visit_expr(iter);
            v.visit_block(body);
        }
        Stmt::EntityFor { query, body, .. } => {
            if let Some(f) = &query.filter {
                v.visit_expr(f);
            }
            v.visit_block(body);
        }
        Stmt::While { cond, body, .. } => {
            v.visit_expr(cond);
            v.visit_block(body);
        }
        Stmt::Loop { body, .. } => v.visit_block(body),
        Stmt::If {
            cond, then, else_, ..
        } => {
            v.visit_expr(cond);
            v.visit_block(then);
            if let Some(e) = else_ {
                match e.as_ref() {
                    IfOrBlock::If(s) => v.visit_stmt(s),
                    IfOrBlock::Block(b) => v.visit_block(b),
                }
            }
        }
        Stmt::Match { expr, arms, .. } => {
            v.visit_expr(expr);
            for arm in arms {
                v.visit_expr(&arm.body);
            }
        }
        Stmt::Item(i) => v.visit_item(i),
        Stmt::Break { value, .. } => {
            if let Some(e) = value {
                v.visit_expr(e);
            }
        }
        Stmt::Continue { .. } => {}
        Stmt::ParallelFor(pf) => {
            v.visit_expr(&pf.iter);
            v.visit_block(&pf.body);
        }
        Stmt::Spawn(sb) => v.visit_block(&sb.body),
        Stmt::Sync(sb) => v.visit_block(&sb.body),
        Stmt::Atomic(ab) => v.visit_block(&ab.body),
    }
}

pub fn walk_expr<V: Visitor>(v: &mut V, e: &Expr) {
    match e {
        Expr::BinOp { lhs, rhs, .. } => {
            v.visit_expr(lhs);
            v.visit_expr(rhs);
        }
        Expr::UnOp { expr, .. } => v.visit_expr(expr),
        Expr::Assign { target, value, .. } => {
            v.visit_expr(target);
            v.visit_expr(value);
        }
        Expr::Field { object, .. } => v.visit_expr(object),
        Expr::Index {
            object, indices, ..
        } => {
            v.visit_expr(object);
            for i in indices {
                v.visit_expr(i);
            }
        }
        Expr::Call {
            func, args, named, ..
        } => {
            v.visit_expr(func);
            for a in args {
                v.visit_expr(a);
            }
            for (_, a) in named {
                v.visit_expr(a);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            v.visit_expr(receiver);
            for a in args {
                v.visit_expr(a);
            }
        }
        Expr::MatMul { lhs, rhs, .. } => {
            v.visit_expr(lhs);
            v.visit_expr(rhs);
        }
        Expr::HadamardMul { lhs, rhs, .. } => {
            v.visit_expr(lhs);
            v.visit_expr(rhs);
        }
        Expr::HadamardDiv { lhs, rhs, .. } => {
            v.visit_expr(lhs);
            v.visit_expr(rhs);
        }
        Expr::TensorConcat { lhs, rhs, .. } => {
            v.visit_expr(lhs);
            v.visit_expr(rhs);
        }
        Expr::KronProd { lhs, rhs, .. } => {
            v.visit_expr(lhs);
            v.visit_expr(rhs);
        }
        Expr::OuterProd { lhs, rhs, .. } => {
            v.visit_expr(lhs);
            v.visit_expr(rhs);
        }
        Expr::Grad { inner, .. } => v.visit_expr(inner),
        Expr::Pow { base, exp, .. } => {
            v.visit_expr(base);
            v.visit_expr(exp);
        }
        Expr::VecCtor { elems, .. } => {
            for e in elems {
                v.visit_expr(e);
            }
        }
        Expr::ArrayLit { elems, .. } => {
            for e in elems {
                v.visit_expr(e);
            }
        }
        Expr::Cast { expr, ty, .. } => {
            v.visit_expr(expr);
            v.visit_type(ty);
        }
        Expr::IfExpr {
            cond, then, else_, ..
        } => {
            v.visit_expr(cond);
            v.visit_block(then);
            if let Some(b) = else_ {
                v.visit_block(b);
            }
        }
        Expr::Closure { body, .. } => v.visit_expr(body),
        Expr::Block(b) => v.visit_block(b),
        Expr::Tuple { elems, .. } => {
            for e in elems {
                v.visit_expr(e);
            }
        }
        Expr::StructLit { fields, .. } => {
            for (_, e) in fields {
                v.visit_expr(e);
            }
        }
        Expr::Range { lo, hi, .. } => {
            if let Some(e) = lo {
                v.visit_expr(e);
            }
            if let Some(e) = hi {
                v.visit_expr(e);
            }
        }
        // Leaves
        Expr::IntLit { .. }
        | Expr::FloatLit { .. }
        | Expr::BoolLit { .. }
        | Expr::StrLit { .. }
        | Expr::Ident { .. }
        | Expr::Path { .. } => {}
    }
}

// =============================================================================
// COMPONENT ACCESS ANALYSIS HELPER
// =============================================================================

/// A simple visitor that walks an expression tree and records every
/// `entity.<field>` access it finds, classifying each as read or write.
///
/// This is used by the system analysis pass to populate
/// `SystemDecl::accesses` and `EntityFor::accesses`.
///
/// Usage:
/// ```rust,ignore
/// let mut col = ComponentAccessCollector::new("entity");
/// col.visit_block(&system.body);
/// let accesses: Vec<ComponentAccess> = col.finish();
/// ```
pub struct ComponentAccessCollector {
    /// The name of the entity loop variable to track.
    entity_var: String,
    /// Accumulator, keyed by field name.
    accesses: std::collections::HashMap<String, AccessMode>,
    /// True while we are the target of an assignment.
    in_assign_target: bool,
}

impl ComponentAccessCollector {
    pub fn new(entity_var: impl Into<String>) -> Self {
        ComponentAccessCollector {
            entity_var: entity_var.into(),
            accesses: std::collections::HashMap::new(),
            in_assign_target: false,
        }
    }

    /// Return the collected accesses, sorted by field name for determinism.
    pub fn finish(self) -> Vec<ComponentAccess> {
        let mut v: Vec<ComponentAccess> = self
            .accesses
            .into_iter()
            .map(|(field, mode)| ComponentAccess {
                component: to_component_name(&field),
                mode,
                field_alias: field,
            })
            .collect();
        v.sort_by(|a, b| a.component.cmp(&b.component));
        v
    }

    fn record(&mut self, field: &str, mode: AccessMode) {
        let entry = self.accesses.entry(field.to_string()).or_insert(mode);
        *entry = entry.merge(mode);
    }

    /// Returns the first field accessed from the tracked entity variable, e.g.:
    /// - `entity.pos` -> `Some("pos")`
    /// - `entity.transform.pos` -> `Some("transform")`
    fn entity_component_field<'a>(&self, mut expr: &'a Expr) -> Option<&'a str> {
        loop {
            match expr {
                Expr::Field { object, field, .. } => {
                    if matches!(
                        object.as_ref(),
                        Expr::Ident { name, .. } if name == &self.entity_var
                    ) {
                        return Some(field.as_str());
                    }
                    expr = object.as_ref();
                }
                _ => return None,
            }
        }
    }
}

/// Convert a snake_case field name to a PascalCase component name:
/// `"position"` → `"Position"`, `"linear_velocity"` → `"LinearVelocity"`.
fn to_component_name(field: &str) -> String {
    field
        .split('_')
        .map(|word| {
            let mut c = word.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            }
        })
        .collect()
}

impl Visitor for ComponentAccessCollector {
    fn visit_expr(&mut self, e: &Expr) {
        match e {
            // `entity.field = …` or `entity.field += …` — write (or read+write)
            Expr::Assign {
                target, value, op, ..
            } => {
                if let Some(field) = self.entity_component_field(target) {
                    self.record(field, AccessMode::Write);
                    if op.is_compound() {
                        self.record(field, AccessMode::Read);
                    }
                } else {
                    // Mark target side
                    self.in_assign_target = true;
                    self.visit_expr(target);
                    self.in_assign_target = false;
                    // For compound assignments (+=, -=, …) the target is also read
                    if op.is_compound() {
                        self.in_assign_target = false;
                        self.visit_expr(target);
                    }
                }
                self.visit_expr(value);
            }

            // `entity.field` — read (unless in_assign_target)
            Expr::Field { object, field, .. } => {
                if let Expr::Ident { name, .. } = object.as_ref() {
                    if name == &self.entity_var {
                        let mode = if self.in_assign_target {
                            AccessMode::Write
                        } else {
                            AccessMode::Read
                        };
                        self.record(field, mode);
                        return; // don't recurse into the Ident
                    }
                }
                // Recurse for nested field access like `entity.transform.pos`
                self.visit_expr(object);
            }

            _ => walk_expr(self, e),
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy() -> Span {
        Span::dummy()
    }

    // ── Type helpers ──────────────────────────────────────────────────────

    #[test]
    fn test_elem_type_sizes() {
        assert_eq!(ElemType::F32.byte_size(), 4);
        assert_eq!(ElemType::F64.byte_size(), 8);
        assert_eq!(ElemType::F16.byte_size(), 2);
        assert_eq!(ElemType::I8.byte_size(), 1);
        assert_eq!(ElemType::Bool.byte_size(), 1);
    }

    #[test]
    fn test_simd_candidate() {
        assert!(Type::Scalar(ElemType::F32).is_simd_candidate());
        assert!(Type::Vec {
            size: VecSize::N3,
            family: VecFamily::Float
        }
        .is_simd_candidate());
        assert!(Type::Quat.is_simd_candidate());
        assert!(!Type::Scalar(ElemType::Bool).is_simd_candidate());
        assert!(!Type::Named("Foo".into()).is_simd_candidate());
    }

    #[test]
    fn test_tensor_type_node() {
        let t = Type::Tensor {
            elem: ElemType::F32,
            shape: vec![DimExpr::Lit(128), DimExpr::Lit(128)],
        };
        assert!(t.contains_float());
        if let Type::Tensor { shape, .. } = &t {
            assert_eq!(shape.len(), 2);
            assert_eq!(shape[0], DimExpr::Lit(128));
        }
    }

    // ── Expr helpers ──────────────────────────────────────────────────────

    #[test]
    fn test_field_chain() {
        // entity.position
        let e = Expr::Field {
            span: dummy(),
            object: Box::new(Expr::Ident {
                span: dummy(),
                name: "entity".into(),
            }),
            field: "position".into(),
        };
        let (root, fields) = e.as_field_chain().unwrap();
        assert_eq!(root, "entity");
        assert_eq!(fields, vec!["position"]);
    }

    #[test]
    fn test_nested_field_chain() {
        // entity.transform.pos
        let e = Expr::Field {
            span: dummy(),
            object: Box::new(Expr::Field {
                span: dummy(),
                object: Box::new(Expr::Ident {
                    span: dummy(),
                    name: "entity".into(),
                }),
                field: "transform".into(),
            }),
            field: "pos".into(),
        };
        let (root, fields) = e.as_field_chain().unwrap();
        assert_eq!(root, "entity");
        assert_eq!(fields, vec!["transform", "pos"]);
    }

    // ── AssignOpKind ──────────────────────────────────────────────────────

    #[test]
    fn test_assign_op_to_binop() {
        assert_eq!(AssignOpKind::AddAssign.to_binop(), Some(BinOpKind::Add));
        assert_eq!(AssignOpKind::MulAssign.to_binop(), Some(BinOpKind::Mul));
        assert_eq!(AssignOpKind::Assign.to_binop(), None);
        assert!(AssignOpKind::AddAssign.is_compound());
        assert!(!AssignOpKind::Assign.is_compound());
    }

    // ── AccessMode ────────────────────────────────────────────────────────

    #[test]
    fn test_access_mode_merge() {
        assert_eq!(AccessMode::Read.merge(AccessMode::Read), AccessMode::Read);
        assert_eq!(
            AccessMode::Read.merge(AccessMode::Write),
            AccessMode::ReadWrite
        );
        assert_eq!(
            AccessMode::Write.merge(AccessMode::Write),
            AccessMode::Write
        );
        assert_eq!(
            AccessMode::ReadWrite.merge(AccessMode::Read),
            AccessMode::ReadWrite
        );
    }

    // ── ComponentAccessCollector ──────────────────────────────────────────

    /// Build a minimal `entity.position += entity.velocity * dt` AST and
    /// verify the collector correctly identifies position=ReadWrite, velocity=Read.
    #[test]
    fn test_component_access_collector_basic() {
        // entity.position
        let pos = Expr::Field {
            span: dummy(),
            object: Box::new(Expr::Ident {
                span: dummy(),
                name: "entity".into(),
            }),
            field: "position".into(),
        };
        // entity.velocity
        let vel = Expr::Field {
            span: dummy(),
            object: Box::new(Expr::Ident {
                span: dummy(),
                name: "entity".into(),
            }),
            field: "velocity".into(),
        };
        // dt
        let dt = Expr::Ident {
            span: dummy(),
            name: "dt".into(),
        };

        // entity.velocity * dt
        let vel_dt = Expr::BinOp {
            span: dummy(),
            op: BinOpKind::Mul,
            lhs: Box::new(vel),
            rhs: Box::new(dt),
        };

        // entity.position += entity.velocity * dt
        let assign = Expr::Assign {
            span: dummy(),
            op: AssignOpKind::AddAssign,
            target: Box::new(pos),
            value: Box::new(vel_dt),
        };

        let body = Block {
            span: dummy(),
            stmts: vec![Stmt::Expr {
                span: dummy(),
                expr: assign,
                has_semi: true,
            }],
            tail: None,
        };

        let mut col = ComponentAccessCollector::new("entity");
        col.visit_block(&body);
        let accesses = col.finish();

        // Should have exactly two accesses
        assert_eq!(accesses.len(), 2);

        let pos_acc = accesses
            .iter()
            .find(|a| a.field_alias == "position")
            .unwrap();
        assert_eq!(
            pos_acc.mode,
            AccessMode::ReadWrite,
            "position should be ReadWrite because of +="
        );
        assert_eq!(pos_acc.component, "Position");

        let vel_acc = accesses
            .iter()
            .find(|a| a.field_alias == "velocity")
            .unwrap();
        assert_eq!(
            vel_acc.mode,
            AccessMode::Read,
            "velocity should be Read-only"
        );
        assert_eq!(vel_acc.component, "Velocity");
    }

    /// Pure read: `let speed = entity.velocity.length()`
    #[test]
    fn test_component_access_read_only() {
        let vel = Expr::Field {
            span: dummy(),
            object: Box::new(Expr::Ident {
                span: dummy(),
                name: "entity".into(),
            }),
            field: "velocity".into(),
        };
        let call = Expr::MethodCall {
            span: dummy(),
            receiver: Box::new(vel),
            method: "length".into(),
            args: vec![],
        };
        let body = Block {
            span: dummy(),
            stmts: vec![Stmt::Let {
                span: dummy(),
                pattern: Pattern::Ident {
                    span: dummy(),
                    name: "speed".into(),
                    mutable: false,
                },
                ty: None,
                init: Some(call),
                mutable: false,
            }],
            tail: None,
        };
        let mut col = ComponentAccessCollector::new("entity");
        col.visit_block(&body);
        let accesses = col.finish();

        assert_eq!(accesses.len(), 1);
        assert_eq!(accesses[0].mode, AccessMode::Read);
    }

    /// Plain write: `entity.position = vec3(0.0, 0.0, 0.0)`
    #[test]
    fn test_component_access_write_only() {
        let pos = Expr::Field {
            span: dummy(),
            object: Box::new(Expr::Ident {
                span: dummy(),
                name: "entity".into(),
            }),
            field: "position".into(),
        };
        let zero = Expr::VecCtor {
            span: dummy(),
            size: VecSize::N3,
            elems: vec![
                Expr::FloatLit {
                    span: dummy(),
                    value: 0.0,
                },
                Expr::FloatLit {
                    span: dummy(),
                    value: 0.0,
                },
                Expr::FloatLit {
                    span: dummy(),
                    value: 0.0,
                },
            ],
        };
        let assign = Expr::Assign {
            span: dummy(),
            op: AssignOpKind::Assign,
            target: Box::new(pos),
            value: Box::new(zero),
        };
        let body = Block {
            span: dummy(),
            stmts: vec![Stmt::Expr {
                span: dummy(),
                expr: assign,
                has_semi: true,
            }],
            tail: None,
        };
        let mut col = ComponentAccessCollector::new("entity");
        col.visit_block(&body);
        let accesses = col.finish();
        assert_eq!(accesses[0].mode, AccessMode::Write);
    }

    /// Compound assignment through nested fields should still mark the root
    /// entity component as ReadWrite exactly once.
    #[test]
    fn test_component_access_nested_compound_target() {
        let target = Expr::Field {
            span: dummy(),
            object: Box::new(Expr::Field {
                span: dummy(),
                object: Box::new(Expr::Ident {
                    span: dummy(),
                    name: "entity".into(),
                }),
                field: "transform".into(),
            }),
            field: "position".into(),
        };
        let rhs = Expr::FloatLit {
            span: dummy(),
            value: 1.0,
        };
        let assign = Expr::Assign {
            span: dummy(),
            op: AssignOpKind::AddAssign,
            target: Box::new(target),
            value: Box::new(rhs),
        };
        let body = Block {
            span: dummy(),
            stmts: vec![Stmt::Expr {
                span: dummy(),
                expr: assign,
                has_semi: true,
            }],
            tail: None,
        };

        let mut col = ComponentAccessCollector::new("entity");
        col.visit_block(&body);
        let accesses = col.finish();

        assert_eq!(accesses.len(), 1);
        assert_eq!(accesses[0].field_alias, "transform");
        assert_eq!(accesses[0].mode, AccessMode::ReadWrite);
    }

    // ── SystemDecl helpers ────────────────────────────────────────────────

    #[test]
    fn test_system_decl_effective_parallelism_from_attr() {
        let mut sys = SystemDecl::new(
            dummy(),
            "Spawn",
            vec![Param::simple(dummy(), "dt", Type::Scalar(ElemType::F32))],
            Block::new(dummy()),
        );
        // Analysis says Auto, but attribute says Simd → Simd wins.
        sys.attrs = vec![Attribute::Simd];
        sys.parallelism = ParallelismHint::Auto;
        assert_eq!(sys.effective_parallelism(), ParallelismHint::Simd);
    }

    #[test]
    fn test_system_decl_has_writes() {
        let mut sys = SystemDecl::new(dummy(), "Update", vec![], Block::new(dummy()));
        sys.accesses = vec![
            ComponentAccess {
                component: "Position".into(),
                mode: AccessMode::ReadWrite,
                field_alias: "position".into(),
            },
            ComponentAccess {
                component: "Velocity".into(),
                mode: AccessMode::Read,
                field_alias: "velocity".into(),
            },
        ];
        assert!(sys.has_writes());

        // Now make everything read-only
        sys.accesses
            .iter_mut()
            .for_each(|a| a.mode = AccessMode::Read);
        assert!(!sys.has_writes());
    }

    // ── Program helpers ───────────────────────────────────────────────────

    #[test]
    fn test_program_iteration() {
        let mut prog = Program::new();
        prog.items.push(Item::Component(ComponentDecl {
            span: dummy(),
            attrs: vec![],
            name: "Position".into(),
            layout: ComponentLayout::Soa,
            fields: vec![
                StructField {
                    span: dummy(),
                    name: "x".into(),
                    ty: Type::Scalar(ElemType::F32),
                    attrs: vec![],
                },
                StructField {
                    span: dummy(),
                    name: "y".into(),
                    ty: Type::Scalar(ElemType::F32),
                    attrs: vec![],
                },
                StructField {
                    span: dummy(),
                    name: "z".into(),
                    ty: Type::Scalar(ElemType::F32),
                    attrs: vec![],
                },
            ],
        }));
        prog.items.push(Item::System(SystemDecl::new(
            dummy(),
            "Update",
            vec![Param::simple(dummy(), "dt", Type::Scalar(ElemType::F32))],
            Block::new(dummy()),
        )));

        assert_eq!(prog.components().count(), 1);
        assert_eq!(prog.systems().count(), 1);
        assert_eq!(prog.components().next().unwrap().name, "Position");
        assert_eq!(prog.systems().next().unwrap().name, "Update");
    }

    // ── to_component_name ─────────────────────────────────────────────────

    #[test]
    fn test_to_component_name() {
        assert_eq!(to_component_name("position"), "Position");
        assert_eq!(to_component_name("linear_velocity"), "LinearVelocity");
        assert_eq!(to_component_name("health"), "Health");
        assert_eq!(to_component_name("x"), "X");
    }

    // ── EntityQuery ───────────────────────────────────────────────────────

    #[test]
    fn test_entity_query_unconstrained() {
        let q = EntityQuery::all(dummy());
        assert!(q.is_unconstrained());
    }

    #[test]
    fn test_entity_query_constrained() {
        let q = EntityQuery {
            span: dummy(),
            with: vec!["Position".into(), "Velocity".into()],
            without: vec!["Dead".into()],
            filter: None,
        };
        assert!(!q.is_unconstrained());
        assert_eq!(q.with.len(), 2);
        assert_eq!(q.without.len(), 1);
    }

    // ── Visitor ───────────────────────────────────────────────────────────

    /// Count how many Ident nodes exist in an expression tree using the visitor.
    struct IdentCounter(usize);
    impl Visitor for IdentCounter {
        fn visit_expr(&mut self, e: &Expr) {
            if matches!(e, Expr::Ident { .. }) {
                self.0 += 1;
            }
            walk_expr(self, e);
        }
    }

    #[test]
    fn test_visitor_counts_idents() {
        // entity.position += entity.velocity * dt
        let pos = Expr::Field {
            span: dummy(),
            object: Box::new(Expr::Ident {
                span: dummy(),
                name: "entity".into(),
            }),
            field: "position".into(),
        };
        let vel = Expr::Field {
            span: dummy(),
            object: Box::new(Expr::Ident {
                span: dummy(),
                name: "entity".into(),
            }),
            field: "velocity".into(),
        };
        let dt = Expr::Ident {
            span: dummy(),
            name: "dt".into(),
        };
        let expr = Expr::Assign {
            span: dummy(),
            op: AssignOpKind::AddAssign,
            target: Box::new(pos),
            value: Box::new(Expr::BinOp {
                span: dummy(),
                op: BinOpKind::Mul,
                lhs: Box::new(vel),
                rhs: Box::new(dt),
            }),
        };
        let mut counter = IdentCounter(0);
        counter.visit_expr(&expr);
        // "entity" (pos side) + "entity" (vel side) + "dt" = 3
        assert_eq!(counter.0, 3);
    }

    // ── ComponentLayout ───────────────────────────────────────────────────

    #[test]
    fn test_component_layout_default_soa() {
        assert_eq!(ComponentLayout::default(), ComponentLayout::Soa);
    }
}

// =============================================================================
// TESTS — Features 3 & 4, Unique Features 1, 2, 3
// =============================================================================

#[cfg(test)]
mod tests_new {
    use super::*;
    fn d() -> Span {
        Span::dummy()
    }
    fn f32_ty() -> Type {
        Type::Scalar(ElemType::F32)
    }
    fn float_lit(v: f64) -> Expr {
        Expr::FloatLit {
            span: d(),
            value: v,
        }
    }
    fn ident(n: &str) -> Expr {
        Expr::Ident {
            span: d(),
            name: n.into(),
        }
    }

    // =========================================================================
    // FEATURE 3 — AI BEHAVIOUR SYSTEMS
    // =========================================================================

    // ── AgentDecl construction ────────────────────────────────────────────────

    fn make_warden() -> AgentDecl {
        AgentDecl {
            span: d(),
            attrs: vec![],
            name: "Warden".into(),
            architecture: AgentArchitecture::Utility,
            perceptions: vec![PerceptionSpec {
                span: d(),
                kind: PerceptionKind::Vision,
                range: Some(40.0),
                fov: Some(120.0),
                tag: None,
            }],
            memory: Some(MemorySpec {
                span: d(),
                kind: MemoryKind::Episodic,
                capacity: Some(MemoryCapacity::Duration { seconds: 120.0 }),
            }),
            learning: Some(LearningSpec {
                span: d(),
                kind: LearningKind::Reinforcement,
                learning_rate: Some(3e-4),
                gamma: Some(0.99),
                policy_model: Some("PolicyNet".into()),
            }),
            behaviors: vec![
                BehaviorRule {
                    span: d(),
                    name: "Flee".into(),
                    priority: BehaviorPriority(100),
                    params: vec![],
                    body: Block::new(d()),
                },
                BehaviorRule {
                    span: d(),
                    name: "Patrol".into(),
                    priority: BehaviorPriority(10),
                    params: vec![],
                    body: Block::new(d()),
                },
            ],
            goals: vec![],
            fields: vec![StructField {
                span: d(),
                name: "health".into(),
                ty: f32_ty(),
                attrs: vec![],
            }],
        }
    }

    #[test]
    fn test_agent_decl_learnable() {
        let agent = make_warden();
        assert!(agent.is_learnable());
    }

    #[test]
    fn test_agent_decl_not_learnable() {
        let mut agent = make_warden();
        agent.learning = Some(LearningSpec {
            span: d(),
            kind: LearningKind::None,
            learning_rate: None,
            gamma: None,
            policy_model: None,
        });
        assert!(!agent.is_learnable());
    }

    #[test]
    fn test_agent_decl_no_learning() {
        let mut agent = make_warden();
        agent.learning = None;
        assert!(!agent.is_learnable());
    }

    #[test]
    fn test_agent_top_behavior() {
        let agent = make_warden();
        let top = agent.top_behavior().unwrap();
        assert_eq!(top.name, "Flee");
        assert_eq!(top.priority, BehaviorPriority(100));
    }

    #[test]
    fn test_agent_perception_range() {
        let agent = make_warden();
        assert_eq!(agent.perceptions[0].range, Some(40.0));
        assert!(matches!(agent.perceptions[0].kind, PerceptionKind::Vision));
    }

    #[test]
    fn test_agent_memory_duration() {
        let agent = make_warden();
        let mem = agent.memory.as_ref().unwrap();
        assert!(matches!(mem.kind, MemoryKind::Episodic));
        assert!(matches!(
            mem.capacity,
            Some(MemoryCapacity::Duration { seconds: s }) if s == 120.0
        ));
    }

    #[test]
    fn test_agent_learning_spec() {
        let agent = make_warden();
        let ls = agent.learning.as_ref().unwrap();
        assert!(matches!(ls.kind, LearningKind::Reinforcement));
        assert_eq!(ls.policy_model, Some("PolicyNet".into()));
        assert_eq!(ls.gamma, Some(0.99));
    }

    #[test]
    fn test_agent_architecture_default() {
        assert_eq!(AgentArchitecture::default(), AgentArchitecture::Utility);
    }

    #[test]
    fn test_behavior_priority_ordering() {
        assert!(BehaviorPriority(100) > BehaviorPriority(10));
        assert!(BehaviorPriority(0) < BehaviorPriority(1));
    }

    // ── Visitor reaches agent behavior bodies ─────────────────────────────────

    struct BlockCounter(usize);
    impl Visitor for BlockCounter {
        fn visit_block(&mut self, b: &Block) {
            self.0 += 1;
            walk_block(self, b);
        }
    }

    #[test]
    fn test_visitor_walks_agent_behaviors() {
        let agent = make_warden();
        let item = Item::Agent(agent);
        let mut counter = BlockCounter(0);
        counter.visit_item(&item);
        // Two behavior bodies
        assert_eq!(counter.0, 2);
    }

    // =========================================================================
    // FEATURE 4 — PARALLELISM BY DEFAULT
    // =========================================================================

    fn make_parallel_for() -> ParallelFor {
        ParallelFor {
            span: d(),
            attrs: vec![Attribute::Simd],
            var: Pattern::Ident {
                span: d(),
                name: "agent".into(),
                mutable: false,
            },
            iter: ident("swarm"),
            body: Block {
                span: d(),
                stmts: vec![Stmt::Expr {
                    span: d(),
                    expr: Expr::MethodCall {
                        span: d(),
                        receiver: Box::new(ident("agent")),
                        method: "update".into(),
                        args: vec![],
                    },
                    has_semi: true,
                }],
                tail: None,
            },
            label: None,
            chunk: Some(64),
            schedule: ScheduleKind::Simd,
        }
    }

    #[test]
    fn test_parallel_for_basic() {
        let pf = make_parallel_for();
        assert_eq!(pf.chunk, Some(64));
        assert_eq!(pf.schedule, ScheduleKind::Simd);
        assert!(matches!(pf.attrs[0], Attribute::Simd));
    }

    #[test]
    fn test_parallel_for_as_stmt() {
        let pf = make_parallel_for();
        let stmt = Stmt::ParallelFor(pf);
        let mut counter = BlockCounter(0);
        counter.visit_stmt(&stmt);
        assert_eq!(counter.0, 1); // body block visited
    }

    #[test]
    fn test_schedule_kind_default() {
        assert_eq!(ScheduleKind::default(), ScheduleKind::Auto);
    }

    #[test]
    fn test_spawn_block() {
        let sb = SpawnBlock {
            span: d(),
            attrs: vec![Attribute::Gpu],
            body: Block::new(d()),
            name: Some("physics_task".into()),
        };
        let stmt = Stmt::Spawn(sb);
        let mut counter = BlockCounter(0);
        counter.visit_stmt(&stmt);
        assert_eq!(counter.0, 1);
    }

    #[test]
    fn test_atomic_block() {
        let ab = AtomicBlock {
            span: d(),
            body: Block::new(d()),
        };
        let stmt = Stmt::Atomic(ab);
        let mut counter = BlockCounter(0);
        counter.visit_stmt(&stmt);
        assert_eq!(counter.0, 1);
    }

    #[test]
    fn test_sync_block() {
        let sb = SyncBlock {
            span: d(),
            body: Block::new(d()),
        };
        let stmt = Stmt::Sync(sb);
        let mut counter = BlockCounter(0);
        counter.visit_stmt(&stmt);
        assert_eq!(counter.0, 1);
    }

    // =========================================================================
    // UNIQUE FEATURE 1 — NEURAL-NETWORK MODEL
    // =========================================================================

    fn make_policy_net() -> ModelDecl {
        ModelDecl {
            span: d(),
            attrs: vec![Attribute::Grad],
            name: "PolicyNet".into(),
            device: ModelDevice::Auto,
            optimizer: Some(OptimizerSpec {
                span: d(),
                kind: OptimizerKind::Adam,
                learning_rate: 3e-4,
                extra: vec![],
            }),
            layers: vec![
                ModelLayer::Input {
                    span: d(),
                    size: 128,
                },
                ModelLayer::Dense {
                    span: d(),
                    units: 256,
                    activation: Activation::Relu,
                    bias: true,
                },
                ModelLayer::Dense {
                    span: d(),
                    units: 256,
                    activation: Activation::Relu,
                    bias: true,
                },
                ModelLayer::Output {
                    span: d(),
                    units: 12,
                    activation: Activation::Softmax,
                },
            ],
        }
    }

    #[test]
    fn test_model_decl_layers() {
        let m = make_policy_net();
        assert_eq!(m.layers.len(), 4);
        assert_eq!(m.hidden_layer_count(), 2); // excludes Input + Output
    }

    #[test]
    fn test_model_is_trainable() {
        let m = make_policy_net();
        assert!(m.is_trainable()); // has @grad
    }

    #[test]
    fn test_model_not_trainable_without_grad() {
        let mut m = make_policy_net();
        m.attrs.clear();
        assert!(!m.is_trainable());
    }

    #[test]
    fn test_model_layer_span() {
        let layer = ModelLayer::Dense {
            span: d(),
            units: 64,
            activation: Activation::Relu,
            bias: false,
        };
        let _ = layer.span(); // must not panic
    }

    #[test]
    fn test_model_activation_default() {
        assert_eq!(Activation::default(), Activation::Linear);
    }

    #[test]
    fn test_model_device_default() {
        assert_eq!(ModelDevice::default(), ModelDevice::Auto);
    }

    #[test]
    fn test_model_layer_kinds() {
        let layers = vec![
            ModelLayer::Conv2d {
                span: d(),
                filters: 32,
                kernel_h: 3,
                kernel_w: 3,
                stride: 1,
                padding: Padding::Same,
                activation: Activation::Relu,
            },
            ModelLayer::Pool {
                span: d(),
                size_h: 2,
                size_w: 2,
                op: PoolOp::Max,
            },
            ModelLayer::Dropout {
                span: d(),
                rate: 0.1,
            },
            ModelLayer::Norm {
                span: d(),
                kind: NormKind::Batch,
            },
            ModelLayer::Attention {
                span: d(),
                num_heads: 8,
                head_dim: 64,
            },
            ModelLayer::Embed {
                span: d(),
                vocab_size: 10_000,
                embed_dim: 128,
            },
            ModelLayer::Recurrent {
                span: d(),
                units: 128,
                cell: RecurrentCell::Lstm,
                bidirect: false,
            },
        ];
        assert_eq!(layers.len(), 7);
        for layer in &layers {
            let _ = layer.span();
        } // all spans accessible
    }

    // ── Visitor reaches nested model (via Program) ────────────────────────────

    struct ModelCounter(usize);
    impl Visitor for ModelCounter {
        fn visit_model(&mut self, _m: &ModelDecl) {
            self.0 += 1;
            // no walk_model needed — layers are leaf data
        }
    }

    #[test]
    fn test_visitor_counts_models() {
        let mut prog = Program::new();
        prog.items.push(Item::Model(make_policy_net()));
        prog.items.push(Item::Model(make_policy_net()));
        let mut counter = ModelCounter(0);
        counter.visit_program(&prog);
        assert_eq!(counter.0, 2);
    }

    // =========================================================================
    // UNIQUE FEATURE 2 — SIMULATION TRAINING
    // =========================================================================

    fn make_train_decl() -> TrainDecl {
        TrainDecl {
            span: d(),
            attrs: vec![],
            agent: "Warden".into(),
            world: "HollowFacility".into(),
            signals: vec![
                SignalSpec {
                    span: d(),
                    is_reward: true,
                    name: "survive".into(),
                    weight: 1.0,
                    expr: None,
                },
                SignalSpec {
                    span: d(),
                    is_reward: false,
                    name: "seen".into(),
                    weight: 2.0,
                    expr: None,
                },
                SignalSpec {
                    span: d(),
                    is_reward: true,
                    name: "eliminate".into(),
                    weight: 0.5,
                    expr: None,
                },
            ],
            episode: Some(EpisodeSpec {
                span: d(),
                max_steps: Some(2000),
                max_seconds: None,
                done_condition: Some(Expr::BoolLit {
                    span: d(),
                    value: false,
                }),
                num_envs: Some(64),
            }),
            model: Some("PolicyNet".into()),
            optimizer: None,
            hyper: vec![("gamma".into(), float_lit(0.99))],
            algorithm: None,
            value_model: None,
        }
    }

    #[test]
    fn test_train_decl_agent_world() {
        let t = make_train_decl();
        assert_eq!(t.agent, "Warden");
        assert_eq!(t.world, "HollowFacility");
    }

    #[test]
    fn test_train_decl_reward_penalty_weights() {
        let t = make_train_decl();
        // rewards: survive(1.0) + eliminate(0.5)
        assert!((t.total_reward_weight() - 1.5).abs() < 1e-9);
        // penalties: seen(2.0)
        assert!((t.total_penalty_weight() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_train_decl_episode_spec() {
        let t = make_train_decl();
        let ep = t.episode.as_ref().unwrap();
        assert_eq!(ep.max_steps, Some(2000));
        assert_eq!(ep.num_envs, Some(64));
    }

    #[test]
    fn test_train_decl_signals_count() {
        let t = make_train_decl();
        assert_eq!(t.signals.iter().filter(|s| s.is_reward).count(), 2);
        assert_eq!(t.signals.iter().filter(|s| !s.is_reward).count(), 1);
    }

    // ── Visitor reaches train hyper-parameters and episode done-condition ──────

    struct ExprCounter(usize);
    impl Visitor for ExprCounter {
        fn visit_expr(&mut self, e: &Expr) {
            self.0 += 1;
            walk_expr(self, e);
        }
    }

    #[test]
    fn test_visitor_walks_train_exprs() {
        let t = make_train_decl();
        let item = Item::Train(t);
        let mut counter = ExprCounter(0);
        counter.visit_item(&item);
        // done_condition (BoolLit) + hyper gamma (FloatLit) = 2
        assert_eq!(counter.0, 2);
    }

    // =========================================================================
    // UNIQUE FEATURE 3 — GPU KERNELS (AST-LEVEL SMOKE TESTS)
    // =========================================================================
    // GPU kernel lowering is a codegen concern; at the AST level we just verify
    // that the relevant attributes and parallel-loop constructs are representable
    // and visible to the visitor.

    #[test]
    fn test_gpu_kernel_attribute_on_fn() {
        // @kernel fn particle_update(…) { … }
        let f = FnDecl {
            span: d(),
            attrs: vec![Attribute::Named {
                name: "kernel".into(),
                args: vec![],
            }],
            name: "particle_update".into(),
            generics: vec![],
            params: vec![Param::simple(d(), "dt", f32_ty())],
            ret_ty: None,
            body: Some(Block::new(d())),
            is_async: false,
        };
        assert!(!f.attrs.is_empty());
        assert!(matches!(&f.attrs[0], Attribute::Named { name, .. } if name == "kernel"));
    }

    #[test]
    fn test_gpu_parallel_loop_in_fn() {
        // fn update_particles(particles: &[Particle], dt: f32) {
        //     parallel for p in particles { p.pos += p.vel * dt }
        // }
        let pf = ParallelFor {
            span: d(),
            attrs: vec![Attribute::Gpu],
            var: Pattern::Ident {
                span: d(),
                name: "p".into(),
                mutable: true,
            },
            iter: ident("particles"),
            body: Block {
                span: d(),
                stmts: vec![Stmt::Expr {
                    span: d(),
                    expr: Expr::Assign {
                        span: d(),
                        op: AssignOpKind::AddAssign,
                        target: Box::new(Expr::Field {
                            span: d(),
                            object: Box::new(ident("p")),
                            field: "pos".into(),
                        }),
                        value: Box::new(Expr::BinOp {
                            span: d(),
                            op: BinOpKind::Mul,
                            lhs: Box::new(Expr::Field {
                                span: d(),
                                object: Box::new(ident("p")),
                                field: "vel".into(),
                            }),
                            rhs: Box::new(ident("dt")),
                        }),
                    },
                    has_semi: true,
                }],
                tail: None,
            },
            label: None,
            chunk: None,
            schedule: ScheduleKind::Gpu,
        };

        // schedule should be GPU
        assert_eq!(pf.schedule, ScheduleKind::Gpu);

        // visitor should find 3 idents: p (pos), p (vel), dt
        let stmt = Stmt::ParallelFor(pf);
        let mut counter = ExprCounter(0);
        counter.visit_stmt(&stmt);
        // p (lhs field obj) + p (rhs field obj) + dt = 3 Ident leaves
        //   plus BinOp, Field, Field, Assign = several non-leaf Exprs too
        assert!(
            counter.0 >= 3,
            "expected at least 3 expressions, got {}",
            counter.0
        );
    }

    // =========================================================================
    // PROGRAM — all new item types round-trip through Program helpers
    // =========================================================================

    #[test]
    fn test_program_all_new_iterators() {
        let mut prog = Program::new();
        prog.items.push(Item::Agent(make_warden()));
        prog.items.push(Item::Model(make_policy_net()));
        prog.items.push(Item::Train(make_train_decl()));

        assert_eq!(prog.agents().count(), 1);
        assert_eq!(prog.models().count(), 1);
        assert_eq!(prog.trains().count(), 1);
        assert_eq!(prog.systems().count(), 0);

        assert_eq!(prog.agents().next().unwrap().name, "Warden");
        assert_eq!(prog.models().next().unwrap().name, "PolicyNet");
        assert_eq!(prog.trains().next().unwrap().world, "HollowFacility");
    }

    #[test]
    fn test_item_name_helpers() {
        assert_eq!(Item::Agent(make_warden()).name(), "Warden");
        assert_eq!(Item::Model(make_policy_net()).name(), "PolicyNet");
        assert_eq!(Item::Train(make_train_decl()).name(), "Warden");
    }

    #[test]
    fn test_item_span_helpers() {
        let _ = Item::Agent(make_warden()).span();
        let _ = Item::Model(make_policy_net()).span();
        let _ = Item::Train(make_train_decl()).span();
    }
}
