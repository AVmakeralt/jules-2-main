// =============================================================================
// jules/src/lexer.rs
//
// Lexer for the Jules programming language.
//
// Tensor syntax handled here:
//   tensor<float>[128, 128]   — shaped tensor type
//   A @ B                     — matrix-multiply operator
//   A .* B                    — element-wise multiply (Hadamard)
//   A ++ B                    — tensor concatenation
//   grad A                    — automatic gradient marker
//   @gpu / @cpu               — device placement annotations
// =============================================================================

use std::fmt;
use std::iter::Peekable;

// ─── Source Span ─────────────────────────────────────────────────────────────

/// A half-open byte range [start, end) inside the source string, plus
/// the originating line/column for human-readable error messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: u32,
    pub col: u32,
}

impl Span {
    pub fn new(start: usize, end: usize, line: u32, col: u32) -> Self {
        Span { start, end, line, col }
    }

    /// A dummy span used for synthetic tokens.
    pub fn dummy() -> Self {
        Span { start: 0, end: 0, line: 0, col: 0 }
    }

    /// Merge two spans into one that covers both.
    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
            line: self.line,
            col: self.col,
        }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

// ─── Numeric Literal Kinds ───────────────────────────────────────────────────

/// How a numeric literal was written in source.
#[derive(Debug, Clone, PartialEq)]
pub enum NumericBase {
    Decimal,
    Hex,    // 0x…
    Octal,  // 0o…
    Binary, // 0b…
}

// ─── Token Kinds ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // ── Literals ──────────────────────────────────────────────────────────
    IntLit    { value: u128, base: NumericBase },
    FloatLit  { value: f64 },
    StringLit { value: String },
    BoolLit   { value: bool },

    // ── Identifiers & Keywords ────────────────────────────────────────────
    Ident(String),

    // Storage & variable declarations
    KwLet,
    KwMut,
    KwConst,

    // Control flow
    KwIf,
    KwElse,
    KwFor,
    KwWhile,
    KwLoop,
    KwBreak,
    KwContinue,
    KwReturn,
    KwMatch,
    KwIn,

    // Functions & types
    KwFn,
    KwStruct,
    KwEnum,
    KwImpl,
    KwTrait,
    KwType,
    KwWhere,
    KwAs,
    KwPub,
    KwUse,
    KwMod,
    KwSelf,

    // ── Jules-specific keywords ───────────────────────────────────────────

    /// `tensor`  — introduces a tensor type
    KwTensor,

    /// `grad`    — wraps an expression to enable gradient tracking
    KwGrad,

    /// `device`  — explicit device placement block: `device(gpu) { … }`
    KwDevice,

    /// `kernel`  — declares a GPU compute kernel
    KwKernel,

    /// `system`  — declares a deterministic game-simulation system
    KwSystem,

    /// `component` — declares an ECS component type
    KwComponent,

    /// `query`   — explicit component query inside a system
    KwQuery,

    /// `world`   — the ECS world handle (built-in name)
    KwWorld,

    /// `with`    — component filter: `for e in world with (Vel, Pos)`
    KwWith,

    /// `without` — exclusion filter: `for e in world without (Dead,)`
    KwWithout,

    // ── Shader / graphics pipeline (Feature 5) ───────────────────────────
    /// `shader`      — declares a GPU shader program (vertex/fragment/compute)
    KwShader,
    /// `vertex`      — vertex shader stage block
    KwVertex,
    /// `fragment`    — fragment (pixel) shader stage block
    KwFragment,
    /// `compute`     — compute shader stage block
    KwCompute,
    /// `buffer`      — GPU storage/vertex buffer binding
    KwBuffer,
    /// `uniform`     — uniform / constant buffer binding
    KwUniform,
    /// `sampler`     — texture sampler descriptor
    KwSampler,
    /// `texture`     — texture resource (2D / cube / array)
    KwTexture,
    /// `pipeline`    — graphics or compute pipeline descriptor
    KwPipeline,
    /// `pass`        — render pass (attachments + clear values)
    KwPass,
    /// `layout`      — pipeline layout (bind groups, push constants)
    KwLayout,

    // ── Scene / asset / prefab (Feature 6) ───────────────────────────────
    /// `scene`       — declares a named game scene / level
    KwScene,
    /// `prefab`      — reusable entity template
    KwPrefab,
    /// `asset`       — asset reference (mesh, texture, audio, …)
    KwAsset,
    /// `lod`         — level-of-detail configuration block
    KwLod,

    // ── Physics (Feature 7) ───────────────────────────────────────────────
    /// `collider`    — physics collider shape declaration
    KwCollider,
    /// `rigidbody`   — physics rigid body component
    KwRigidbody,
    /// `constraint`  — physics joint / constraint (spring, hinge, …)
    KwConstraint,
    /// `trigger`     — trigger volume (overlap events, no collision response)
    KwTrigger,
    /// `physics`     — physics world configuration block
    KwPhysics,

    // ── ML / data pipeline (Feature 8) ───────────────────────────────────
    /// `loss`        — declares a named loss function
    KwLoss,
    /// `metric`      — declares an evaluation metric
    KwMetric,
    /// `dataloader`  — dataset iteration handle
    KwDataloader,
    /// `transform`   — data preprocessing / augmentation transform
    KwTransform,
    /// `vmap`        — vectorised-map: lift fn over batch dimension (JAX-style)
    KwVmap,
    /// `jit`         — just-in-time compile a function
    KwJit,
    /// `quantize`    — post-training quantisation block
    KwQuantize,
    /// `prune`       — structured / unstructured weight pruning block
    KwPrune,
    /// `noise`       — procedural noise function (perlin, simplex, white)
    KwNoise,
    /// `shuffle`     — shuffle tensor / dataset along an axis
    KwShuffle,
    /// `checkpoint`  — save / restore a model or training state
    KwCheckpoint,
    /// `pipeline` (ML) — data or model pipeline (reuses KwPipeline above)

    // ── AI behavior system keywords (Feature 3) ──────────────────────────
    /// `agent`       — declares an AI behaviour agent
    KwAgent,
    /// `perception`  — perception block inside an agent
    KwPerception,
    /// `memory`      — memory configuration inside an agent
    KwMemory,
    /// `learning`    — learning strategy inside an agent
    KwLearning,
    /// `behavior`    — named behaviour rule inside an agent
    KwBehavior,
    /// `goal`        — declares a named goal for an agent
    KwGoal,
    /// `sensor`      — generic sensor declaration
    KwSensor,

    // ── Parallelism keywords (Feature 4) ─────────────────────────────────
    /// `parallel`    — `parallel for x in iter { … }` dispatch hint
    KwParallel,
    /// `spawn`       — `spawn { … }` lightweight async task
    KwSpawn,
    /// `sync`        — `sync { … }` synchronisation barrier
    KwSync,
    /// `atomic`      — `atomic { … }` single-instruction atomic region
    KwAtomic,

    // ── Neural-network keywords (Unique Feature 1) ────────────────────────
    /// `model`       — declares a neural-network model
    KwModel,
    /// `layer`       — generic layer inside a model
    KwLayer,
    /// `dense`       — fully-connected layer:  `dense 256 relu`
    KwDense,
    /// `conv`        — convolution layer:       `conv 32 3x3 relu`
    KwConv,
    /// `pool`        — pooling layer:            `pool 2x2 max`
    KwPool,
    /// `recurrent`   — recurrent (LSTM/GRU) layer
    KwRecurrent,
    /// `attention`   — self-attention / transformer layer
    KwAttention,
    /// `embed`       — embedding layer
    KwEmbed,
    /// `dropout`     — dropout regularisation layer
    KwDropout,
    /// `norm`        — normalisation layer (batch / layer / rms)
    KwNorm,
    /// `input` / `output` — explicit I/O shape declarations in a model
    KwInput,
    KwOutput,

    // ── Activation functions — treated as keywords so the parser sees them
    //    directly without needing identifier look-up tables.  ─────────────
    KwRelu,
    KwLeakyRelu,
    KwSigmoid,
    KwTanh,
    KwGelu,
    KwSilu,
    KwSoftmax,
    KwLinear,     // identity / no activation

    // ── Simulation training keywords (Unique Feature 2) ──────────────────
    /// `train`       — `train AgentName in WorldName { … }`
    KwTrain,
    /// `reward`      — reward signal declaration inside a train block
    KwReward,
    /// `penalty`     — penalty signal declaration inside a train block
    KwPenalty,
    /// `episode`     — episode configuration
    KwEpisode,
    /// `policy`      — policy specification inside a train block
    KwPolicy,

    /// `async` / `await` — for async dispatch
    KwAsync,
    KwAwait,

    // ── Vector / matrix math types (game simulation, SIMD) ───────────────
    KwVec2, KwVec3, KwVec4,
    KwIVec2, KwIVec3, KwIVec4,
    KwUVec2, KwUVec3, KwUVec4,
    KwMat2, KwMat3, KwMat4,
    KwQuat,

    // ── Primitive element types (used inside tensor<…>) ───────────────────
    KwF16,
    KwF32,
    KwF64,
    KwBf16,
    KwI8,
    KwI16,
    KwI32,
    KwI64,
    KwU8,
    KwU16,
    KwU32,
    KwU64,
    KwBool,
    KwUsize,

    // ── Device / parallelism annotations ─────────────────────────────────
    AtGpu,
    AtCpu,
    AtTpu,
    AtGrad,        // @grad        — gradient annotation
    AtSimd,        // @simd        — force SIMD vectorisation
    AtParallel,    // @parallel    — force parallel (multi-thread) dispatch
    AtSeq,         // @seq         — force sequential (deterministic) execution
    AtUnroll,      // @unroll      — hint to unroll the next loop
    AtInline,      // @inline      — force inlining of a function
    AtNoInline,    // @noinline    — prevent inlining
    AtKernel,      // @kernel      — GPU kernel entry point
    AtJit,         // @jit         — JIT-compile this function
    AtVmap,        // @vmap        — vectorise over batch dimension
    AtCheckpoint,  // @checkpoint  — gradient checkpointing (recompute in backward)
    AtQuantize,    // @quantize    — annotate for post-training quantisation
    AtPrune,       // @prune       — annotate for weight pruning
    AtProfile,     // @profile     — emit profiling instrumentation
    AtTest,        // @test        — mark function as a unit test
    AtBenchmark,   // @benchmark   — mark function as a microbenchmark
    AtDeprecated,  // @deprecated  — emit deprecation warning at call sites
    AtAi,          // @ai          — neural network injection into agent
    AtCustom(String), // @PPO, @DQN, @SAC, etc.

    // ── Operators ─────────────────────────────────────────────────────────

    // Arithmetic
    Plus,        // +
    Minus,       // -
    Star,        // *
    Slash,       // /
    Percent,     // %
    StarStar,    // ** (scalar power)

    // Tensor-specific operators
    MatMul,       // @    — matrix multiply (A @ B)
    HadamardMul,  // .*   — element-wise multiply
    HadamardDiv,  // ./   — element-wise divide
    TensorConcat, // ++   — tensor concatenation along first axis
    KronProd,     // @@   — Kronecker product  (A @@ B)
    OuterProd,    // ^*   — outer product      (a ^* b)
    FloorDiv,     // //   — floor (integer) division

    // Bitwise
    Ampersand,   // &
    Pipe,        // |
    Caret,       // ^
    Tilde,       // ~
    LtLt,        // <<
    GtGt,        // >>

    // Comparison
    EqEq,        // ==
    BangEq,      // !=
    Lt,          // <
    Gt,          // >
    LtEq,        // <=
    GtEq,        // >=

    // Logical
    AmpAmp,      // &&
    PipePipe,    // ||
    Bang,        // !

    // Assignment
    Eq,          // =
    PlusEq,      // +=
    MinusEq,     // -=
    StarEq,      // *=
    SlashEq,     // /=
    PercentEq,   // %=
    AmpEq,       // &=
    PipeEq,      // |=
    CaretEq,     // ^=
    MatMulEq,    // @=  — in-place matrix multiply

    // Other
    Arrow,       // ->
    FatArrow,    // =>
    DotDot,      // ..
    DotDotEq,    // ..=
    Dot,         // .
    Colon,       // :
    ColonColon,  // ::
    Semicolon,   // ;
    Comma,       // ,
    Question,    // ?
    Hash,        // #

    // ── Delimiters ────────────────────────────────────────────────────────
    LParen,      // (
    RParen,      // )
    LBracket,    // [
    RBracket,    // ]
    LBrace,      // {
    RBrace,      // }
    LAngle,      // < used in tensor<…> (also Lt, disambiguated by parser)
    RAngle,      // > used in tensor<…> (also Gt)

    // ── Special ───────────────────────────────────────────────────────────
    /// End of file.
    Eof,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Ident(s)     => write!(f, "identifier `{}`", s),
            TokenKind::IntLit { value, .. }  => write!(f, "integer `{}`", value),
            TokenKind::FloatLit { value }    => write!(f, "float `{}`", value),
            TokenKind::StringLit { value }   => write!(f, "string `\"{}\"`", value),
            TokenKind::BoolLit { value }     => write!(f, "bool `{}`", value),
            TokenKind::MatMul                => write!(f, "`@`"),
            TokenKind::HadamardMul           => write!(f, "`.*`"),
            TokenKind::HadamardDiv           => write!(f, "`./`"),
            TokenKind::TensorConcat          => write!(f, "`++`"),
            TokenKind::Eof                   => write!(f, "<eof>"),
            other                            => write!(f, "{:?}", other),
        }
    }
}

// ─── Token ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    /// The original source text that produced this token, useful for
    /// diagnostics and source-map generation.
    pub raw: String,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span, raw: impl Into<String>) -> Self {
        Token { kind, span, raw: raw.into() }
    }

    pub fn is_eof(&self) -> bool {
        matches!(self.kind, TokenKind::Eof)
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}", self.kind, self.span)
    }
}

// ─── Lexer Error ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LexError {
    pub message: String,
    pub span: Span,
}

impl LexError {
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        LexError { message: message.into(), span }
    }
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LexError at {}: {}", self.span, self.message)
    }
}

pub type LexResult<T> = Result<T, LexError>;

// ─── Keyword Table ────────────────────────────────────────────────────────────

/// Map a bare identifier string to its keyword `TokenKind`, or `None` if it is
/// a user-defined identifier.
#[inline]
fn keyword(s: &str) -> Option<TokenKind> {
    // Fast-path: all keywords are 2-10 chars, skip the match for anything else
    let len = s.len();
    if len < 2 || len > 10 {
        return None;
    }
    // Second fast-path: first char must match a keyword's first char
    // Keywords starting with unusual letters can be skipped
    match s.as_bytes()[0] {
        b'a' | b'b' | b'c' | b'd' | b'e' | b'f' | b'g' | b'i' | b'j'
        | b'k' | b'l' | b'm' | b'n' | b'o' | b'p' | b'q' | b'r' | b's'
        | b't' | b'u' | b'v' | b'w' | b'z' => {} // possible keyword
        _ => return None, // no keyword starts with h, x, y
    }
    
    Some(match s {
        // Jules core
        "tensor"    => TokenKind::KwTensor,
        "grad"      => TokenKind::KwGrad,
        "device"    => TokenKind::KwDevice,
        "kernel"    => TokenKind::KwKernel,
        "system"    => TokenKind::KwSystem,
        "component" => TokenKind::KwComponent,
        "query"     => TokenKind::KwQuery,
        "world"     => TokenKind::KwWorld,
        "with"      => TokenKind::KwWith,
        "without"   => TokenKind::KwWithout,
        // AI behaviour (Feature 3)
        "agent"       => TokenKind::KwAgent,
        "perception"  => TokenKind::KwPerception,
        "memory"      => TokenKind::KwMemory,
        "learning"    => TokenKind::KwLearning,
        "behavior"    => TokenKind::KwBehavior,
        "goal"        => TokenKind::KwGoal,
        "sensor"      => TokenKind::KwSensor,

        // Parallelism (Feature 4)
        "parallel"    => TokenKind::KwParallel,
        "spawn"       => TokenKind::KwSpawn,
        "sync"        => TokenKind::KwSync,
        "atomic"      => TokenKind::KwAtomic,

        // Neural-network (Unique Feature 1)
        "model"       => TokenKind::KwModel,
        "layer"       => TokenKind::KwLayer,
        "dense"       => TokenKind::KwDense,
        "conv"        => TokenKind::KwConv,
        "pool"        => TokenKind::KwPool,
        "recurrent"   => TokenKind::KwRecurrent,
        "attention"   => TokenKind::KwAttention,
        "embed"       => TokenKind::KwEmbed,
        "dropout"     => TokenKind::KwDropout,
        "norm"        => TokenKind::KwNorm,
        "input"       => TokenKind::KwInput,
        "output"      => TokenKind::KwOutput,

        // Activation functions
        "relu"        => TokenKind::KwRelu,
        "leaky_relu"  => TokenKind::KwLeakyRelu,
        "sigmoid"     => TokenKind::KwSigmoid,
        "tanh"        => TokenKind::KwTanh,
        "gelu"        => TokenKind::KwGelu,
        "silu"        => TokenKind::KwSilu,
        "softmax"     => TokenKind::KwSoftmax,
        "linear"      => TokenKind::KwLinear,

        // Simulation training (Unique Feature 2)
        "train"       => TokenKind::KwTrain,
        "reward"      => TokenKind::KwReward,
        "penalty"     => TokenKind::KwPenalty,
        "episode"     => TokenKind::KwEpisode,
        "policy"      => TokenKind::KwPolicy,

        "async"     => TokenKind::KwAsync,
        "await"     => TokenKind::KwAwait,

        // Vector / matrix math types
        "vec2"  => TokenKind::KwVec2,
        "vec3"  => TokenKind::KwVec3,
        "vec4"  => TokenKind::KwVec4,
        "ivec2" => TokenKind::KwIVec2,
        "ivec3" => TokenKind::KwIVec3,
        "ivec4" => TokenKind::KwIVec4,
        "uvec2" => TokenKind::KwUVec2,
        "uvec3" => TokenKind::KwUVec3,
        "uvec4" => TokenKind::KwUVec4,
        "mat2"  => TokenKind::KwMat2,
        "mat3"  => TokenKind::KwMat3,
        "mat4"  => TokenKind::KwMat4,
        "quat"  => TokenKind::KwQuat,

        // Primitive element types
        "f16"      => TokenKind::KwF16,
        "f32"      => TokenKind::KwF32,
        "f64"      => TokenKind::KwF64,
        "bf16"     => TokenKind::KwBf16,
        "i8"       => TokenKind::KwI8,
        "i16"      => TokenKind::KwI16,
        "i32"      => TokenKind::KwI32,
        "i64"      => TokenKind::KwI64,
        "u8"       => TokenKind::KwU8,
        "u16"      => TokenKind::KwU16,
        "u32"      => TokenKind::KwU32,
        "u64"      => TokenKind::KwU64,
        "bool"     => TokenKind::KwBool,
        "usize"    => TokenKind::KwUsize,

        // ── Shader / graphics ──────────────────────────────────────────────
        "shader"     => TokenKind::KwShader,
        "vertex"     => TokenKind::KwVertex,
        "fragment"   => TokenKind::KwFragment,
        "compute"    => TokenKind::KwCompute,
        "buffer"     => TokenKind::KwBuffer,
        "uniform"    => TokenKind::KwUniform,
        "sampler"    => TokenKind::KwSampler,
        "texture"    => TokenKind::KwTexture,
        "pipeline"   => TokenKind::KwPipeline,
        "pass"       => TokenKind::KwPass,
        "layout"     => TokenKind::KwLayout,

        // ── Scene / asset ──────────────────────────────────────────────────
        "scene"      => TokenKind::KwScene,
        "prefab"     => TokenKind::KwPrefab,
        "asset"      => TokenKind::KwAsset,
        "lod"        => TokenKind::KwLod,

        // ── Physics ────────────────────────────────────────────────────────
        "collider"   => TokenKind::KwCollider,
        "rigidbody"  => TokenKind::KwRigidbody,
        "constraint" => TokenKind::KwConstraint,
        "trigger"    => TokenKind::KwTrigger,
        "physics"    => TokenKind::KwPhysics,

        // ── ML / data pipeline ─────────────────────────────────────────────
        "loss"       => TokenKind::KwLoss,
        "metric"     => TokenKind::KwMetric,
        "dataloader" => TokenKind::KwDataloader,
        "transform"  => TokenKind::KwTransform,
        "vmap"       => TokenKind::KwVmap,
        "jit"        => TokenKind::KwJit,
        "quantize"   => TokenKind::KwQuantize,
        "prune"      => TokenKind::KwPrune,
        "noise"      => TokenKind::KwNoise,
        "shuffle"    => TokenKind::KwShuffle,
        "checkpoint" => TokenKind::KwCheckpoint,
        "if"       => TokenKind::KwIf,
        "else"     => TokenKind::KwElse,
        "for"      => TokenKind::KwFor,
        "while"    => TokenKind::KwWhile,
        "loop"     => TokenKind::KwLoop,
        "break"    => TokenKind::KwBreak,
        "continue" => TokenKind::KwContinue,
        "return"   => TokenKind::KwReturn,
        "match"    => TokenKind::KwMatch,
        "in"       => TokenKind::KwIn,

        // Declarations
        "let"      => TokenKind::KwLet,
        "mut"      => TokenKind::KwMut,
        "const"    => TokenKind::KwConst,
        "fn"       => TokenKind::KwFn,
        "struct"   => TokenKind::KwStruct,
        "enum"     => TokenKind::KwEnum,
        "impl"     => TokenKind::KwImpl,
        "trait"    => TokenKind::KwTrait,
        "type"     => TokenKind::KwType,
        "where"    => TokenKind::KwWhere,
        "as"       => TokenKind::KwAs,
        "pub"      => TokenKind::KwPub,
        "use"      => TokenKind::KwUse,
        "mod"      => TokenKind::KwMod,
        "self"     => TokenKind::KwSelf,

        // Bool literals
        "true"     => TokenKind::BoolLit { value: true },
        "false"    => TokenKind::BoolLit { value: false },

        _ => return None,
    })
}

// ─── Lexer ────────────────────────────────────────────────────────────────────

pub struct Lexer<'src> {
    /// The full source text.
    src: &'src str,
    /// Byte offset of the current character.
    pos: usize,
    /// Current line (1-based).
    line: u32,
    /// Current column (1-based, in characters).
    col: u32,
    /// Peekable iterator over `(char_index, char)` pairs.
    chars: Peekable<std::str::CharIndices<'src>>,
}

impl<'src> Lexer<'src> {
    pub fn new(src: &'src str) -> Self {
        Lexer {
            src,
            pos: 0,
            line: 1,
            col: 1,
            chars: src.char_indices().peekable(),
        }
    }

    // ── Low-level character access ────────────────────────────────────────

    /// Peek at the next character without consuming it.
    #[inline]
    fn peek(&mut self) -> Option<char> {
        self.chars.peek().map(|&(_, c)| c)
    }

    /// Peek at the character *after* the next one (two-char lookahead).
    #[inline]
    fn peek2(&mut self) -> Option<char> {
        let mut iter = self.chars.clone();
        iter.next();
        iter.next().map(|(_, c)| c)
    }

    /// Consume and return the next character, updating position tracking.
    fn advance(&mut self) -> Option<char> {
        match self.chars.next() {
            None => None,
            Some((byte_pos, c)) => {
                self.pos = byte_pos + c.len_utf8();
                if c == '\n' {
                    self.line += 1;
                    self.col = 1;
                } else {
                    self.col += 1;
                }
                Some(c)
            }
        }
    }

    /// Consume the next character only if it equals `expected`.
    fn eat(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Return the current byte position without consuming anything.
    #[inline]
    fn current_pos(&mut self) -> usize {
        self.chars.peek().map(|&(i, _)| i).unwrap_or(self.src.len())
    }

    /// Build a Span from a start position to the current position.
    fn span_from(&mut self, start: usize, start_line: u32, start_col: u32) -> Span {
        Span::new(start, self.current_pos(), start_line, start_col)
    }

    /// Get the raw source slice for a span.
    fn raw_slice(&self, span: &Span) -> &str {
        &self.src[span.start..span.end]
    }

    // ── Whitespace & Comments ─────────────────────────────────────────────

    fn skip_whitespace_and_comments(&mut self) -> LexResult<()> {
        loop {
            // Skip ASCII whitespace.
            while matches!(self.peek(), Some(c) if c.is_ascii_whitespace()) {
                self.advance();
            }

            match (self.peek(), self.peek2()) {
                // Line comment: // …
                (Some('/'), Some('/')) => {
                    self.advance();
                    self.advance();
                    while matches!(self.peek(), Some(c) if c != '\n') {
                        self.advance();
                    }
                }
                // Block comment: /* … */  (supports nesting)
                (Some('/'), Some('*')) => {
                    self.advance();
                    self.advance();
                    self.skip_block_comment()?;
                }
                _ => break,
            }
        }
        Ok(())
    }

    fn skip_block_comment(&mut self) -> LexResult<()> {
        let err_line = self.line;
        let err_col  = self.col;
        let mut depth = 1usize;
        loop {
            match (self.advance(), self.peek()) {
                (None, _) => {
                    return Err(LexError::new(
                        "unterminated block comment",
                        Span::new(0, 0, err_line, err_col),
                    ));
                }
                (Some('/'), Some('*')) => {
                    self.advance();
                    depth += 1;
                }
                (Some('*'), Some('/')) => {
                    self.advance();
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    // ── String Literals ───────────────────────────────────────────────────

    fn lex_string(&mut self, start: usize, line: u32, col: u32) -> LexResult<Token> {
        // Opening `"` was already consumed by the caller.
        let mut value = String::new();
        loop {
            match self.advance() {
                None | Some('\n') => {
                    return Err(LexError::new(
                        "unterminated string literal",
                        Span::new(start, self.current_pos(), line, col),
                    ));
                }
                Some('"') => break,
                Some('\\') => {
                    let escaped = match self.advance() {
                        Some('n')  => '\n',
                        Some('t')  => '\t',
                        Some('r')  => '\r',
                        Some('\\') => '\\',
                        Some('"')  => '"',
                        Some('0')  => '\0',
                        Some('u')  => self.lex_unicode_escape(start, line, col)?,
                        Some(c) => {
                            return Err(LexError::new(
                                format!("unknown escape sequence `\\{}`", c),
                                Span::new(start, self.current_pos(), line, col),
                            ));
                        }
                        None => {
                            return Err(LexError::new(
                                "unexpected end of file in escape sequence",
                                Span::new(start, self.current_pos(), line, col),
                            ));
                        }
                    };
                    value.push(escaped);
                }
                Some(c) => value.push(c),
            }
        }
        let span = self.span_from(start, line, col);
        let raw = self.raw_slice(&span).to_string();
        Ok(Token::new(TokenKind::StringLit { value }, span, raw))
    }

    /// Parse `\u{XXXX}` Unicode escape sequences.
    fn lex_unicode_escape(&mut self, start: usize, line: u32, col: u32) -> LexResult<char> {
        if !self.eat('{') {
            return Err(LexError::new(
                "expected `{` after `\\u` in unicode escape",
                Span::new(start, self.current_pos(), line, col),
            ));
        }
        let mut hex = String::new();
        loop {
            match self.peek() {
                Some('}') => {
                    self.advance();
                    break;
                }
                Some(c) if c.is_ascii_hexdigit() => {
                    hex.push(c);
                    self.advance();
                }
                _ => {
                    return Err(LexError::new(
                        "invalid character in unicode escape",
                        Span::new(start, self.current_pos(), line, col),
                    ));
                }
            }
        }
        let code = u32::from_str_radix(&hex, 16).map_err(|_| {
            LexError::new("invalid unicode escape value", Span::new(start, self.current_pos(), line, col))
        })?;
        char::from_u32(code).ok_or_else(|| {
            LexError::new(
                format!("unicode scalar value U+{:X} is invalid", code),
                Span::new(start, self.current_pos(), line, col),
            )
        })
    }

    // ── Numeric Literals ──────────────────────────────────────────────────

    fn lex_number(&mut self, first: char, start: usize, line: u32, col: u32) -> LexResult<Token> {
        let mut raw_buf = String::new();
        raw_buf.push(first);

        // Detect base prefix (0x, 0o, 0b).
        let base = if first == '0' {
            match self.peek() {
                Some('x') | Some('X') => {
                    raw_buf.push(self.advance().unwrap());
                    NumericBase::Hex
                }
                Some('o') | Some('O') => {
                    raw_buf.push(self.advance().unwrap());
                    NumericBase::Octal
                }
                Some('b') | Some('B') => {
                    raw_buf.push(self.advance().unwrap());
                    NumericBase::Binary
                }
                _ => NumericBase::Decimal,
            }
        } else {
            NumericBase::Decimal
        };

        // Consume digits (with optional `_` separators).
        let mut digits = String::new();
        if first != '0' || matches!(base, NumericBase::Decimal) {
            if matches!(base, NumericBase::Decimal) {
                digits.push(first);
            }
        }

        let is_valid_digit = |c: char, base: &NumericBase| -> bool {
            match base {
                NumericBase::Hex    => c.is_ascii_hexdigit(),
                NumericBase::Octal  => matches!(c, '0'..='7'),
                NumericBase::Binary => matches!(c, '0' | '1'),
                NumericBase::Decimal => c.is_ascii_digit(),
            }
        };

        loop {
            match self.peek() {
                Some('_') => {
                    raw_buf.push('_');
                    self.advance();
                }
                Some(c) if is_valid_digit(c, &base) => {
                    digits.push(c);
                    raw_buf.push(c);
                    self.advance();
                }
                _ => break,
            }
        }

        // Check for float (decimal point or exponent), only in base-10.
        let is_float = matches!(base, NumericBase::Decimal)
            && (self.peek() == Some('.')
                && self.peek2().map(|c| c != '.').unwrap_or(false)  // not `..`
                || matches!(self.peek(), Some('e') | Some('E')));

        if is_float {
            // Consume fractional part.
            if self.peek() == Some('.') {
                raw_buf.push('.');
                digits.push('.');
                self.advance();
                loop {
                    match self.peek() {
                        Some('_') => { raw_buf.push('_'); self.advance(); }
                        Some(c) if c.is_ascii_digit() => {
                            digits.push(c); raw_buf.push(c); self.advance();
                        }
                        _ => break,
                    }
                }
            }
            // Consume exponent.
            if matches!(self.peek(), Some('e') | Some('E')) {
                let e = self.advance().unwrap();
                digits.push(e);
                raw_buf.push(e);
                if matches!(self.peek(), Some('+') | Some('-')) {
                    let sign = self.advance().unwrap();
                    digits.push(sign);
                    raw_buf.push(sign);
                }
                loop {
                    match self.peek() {
                        Some('_') => { raw_buf.push('_'); self.advance(); }
                        Some(c) if c.is_ascii_digit() => {
                            digits.push(c); raw_buf.push(c); self.advance();
                        }
                        _ => break,
                    }
                }
            }
            let span = self.span_from(start, line, col);
            let clean: String = digits.chars().filter(|&c| c != '_').collect();
            let value: f64 = clean.parse().map_err(|_| {
                LexError::new(format!("invalid float literal `{}`", raw_buf), span)
            })?;
            return Ok(Token::new(TokenKind::FloatLit { value }, span, raw_buf));
        }

        // Integer literal.
        let span = self.span_from(start, line, col);
        let clean: String = digits.chars().filter(|&c| c != '_').collect();
        if clean.is_empty() {
            return Err(LexError::new(
                format!("no digits after base prefix in `{}`", raw_buf),
                span,
            ));
        }
        let radix = match base {
            NumericBase::Hex     => 16,
            NumericBase::Octal   => 8,
            NumericBase::Binary  => 2,
            NumericBase::Decimal => 10,
        };
        let value = u128::from_str_radix(&clean, radix).map_err(|_| {
            LexError::new(format!("integer literal `{}` out of range", raw_buf), span)
        })?;
        Ok(Token::new(TokenKind::IntLit { value, base }, span, raw_buf))
    }

    // ── Identifier / Keyword ──────────────────────────────────────────────

    fn lex_ident(&mut self, first: char, start: usize, line: u32, col: u32) -> Token {
        let mut s = String::with_capacity(16);
        s.push(first);
        while matches!(self.peek(), Some(c) if c.is_alphanumeric() || c == '_') {
            s.push(self.advance().unwrap());
        }
        let span = self.span_from(start, line, col);
        // Check if it's a keyword first to avoid unnecessary clone
        if let Some(kw) = keyword(&s) {
            Token::new(kw, span, s)  // keyword: move s into raw
        } else {
            // Identifier: the string moves into Ident, raw is a clone
            let raw = s.clone();
            let kind = TokenKind::Ident(s);
            Token::new(kind, span, raw)
        }
    }

    // ── Device Attributes (@gpu, @cpu, @tpu, @grad) ───────────────────────

    fn lex_at_attribute(&mut self, start: usize, line: u32, col: u32) -> LexResult<Token> {
        // `@` already consumed.
        let mut name = String::new();
        while matches!(self.peek(), Some(c) if c.is_alphanumeric() || c == '_') {
            name.push(self.advance().unwrap());
        }
        let span = self.span_from(start, line, col);
        let kind = match name.as_str() {
            "gpu"         => TokenKind::AtGpu,
            "cpu"         => TokenKind::AtCpu,
            "tpu"         => TokenKind::AtTpu,
            "grad"        => TokenKind::AtGrad,
            "simd"        => TokenKind::AtSimd,
            "parallel"    => TokenKind::AtParallel,
            "seq"         => TokenKind::AtSeq,
            "unroll"      => TokenKind::AtUnroll,
            "inline"      => TokenKind::AtInline,
            "noinline"    => TokenKind::AtNoInline,
            "kernel"      => TokenKind::AtKernel,
            "jit"         => TokenKind::AtJit,
            "vmap"        => TokenKind::AtVmap,
            "checkpoint"  => TokenKind::AtCheckpoint,
            "quantize"    => TokenKind::AtQuantize,
            "prune"       => TokenKind::AtPrune,
            "profile"     => TokenKind::AtProfile,
            "test"        => TokenKind::AtTest,
            "benchmark"   => TokenKind::AtBenchmark,
            "deprecated"  => TokenKind::AtDeprecated,
            "ai"          => TokenKind::AtAi,
            // Plain `@` with no recognised name is the MatMul operator.
            "" => TokenKind::MatMul,
            _ => TokenKind::AtCustom(name.clone()),
        };
        let raw = format!("@{}", name);
        Ok(Token::new(kind, span, raw))
    }

    // ── Single Token ──────────────────────────────────────────────────────

    /// Lex the next token from the source stream.
    pub fn next_token(&mut self) -> LexResult<Token> {
        self.skip_whitespace_and_comments()?;

        let start = self.current_pos();
        let line  = self.line;
        let col   = self.col;

        let c = match self.advance() {
            None => {
                let span = Span::new(start, start, line, col);
                return Ok(Token::new(TokenKind::Eof, span, ""));
            }
            Some(c) => c,
        };

        macro_rules! tok {
            ($kind:expr, $raw:expr) => {{
                let span = self.span_from(start, line, col);
                Token::new($kind, span, $raw)
            }};
        }

        let token = match c {
            // ── String literal ────────────────────────────────────────────
            '"' => self.lex_string(start, line, col)?,

            // ── Numbers ───────────────────────────────────────────────────
            c if c.is_ascii_digit() => self.lex_number(c, start, line, col)?,

            // ── Identifiers / keywords ────────────────────────────────────
            c if c.is_alphabetic() || c == '_' => self.lex_ident(c, start, line, col),

            // ── Device / matmul attribute  (@gpu, @cpu, @@, @, …) ────────
            '@' => {
                // `@@` = Kronecker product
                if self.peek() == Some('@') {
                    self.advance();
                    tok!(TokenKind::KronProd, "@@")
                // If followed immediately by an alpha we have an attribute.
                } else if matches!(self.peek(), Some(c) if c.is_alphabetic()) {
                    self.lex_at_attribute(start, line, col)?
                } else if self.eat('=') {
                    tok!(TokenKind::MatMulEq, "@=")
                } else {
                    tok!(TokenKind::MatMul, "@")
                }
            }

            // ── Dot-family ────────────────────────────────────────────────
            '.' => {
                if self.eat('.') {
                    if self.eat('=') {
                        tok!(TokenKind::DotDotEq, "..=")
                    } else {
                        tok!(TokenKind::DotDot, "..")
                    }
                } else if self.eat('*') {
                    tok!(TokenKind::HadamardMul, ".*")
                } else if self.eat('/') {
                    tok!(TokenKind::HadamardDiv, "./")
                } else {
                    tok!(TokenKind::Dot, ".")
                }
            }

            // ── Plus ──────────────────────────────────────────────────────
            '+' => {
                if self.eat('+') {
                    tok!(TokenKind::TensorConcat, "++")
                } else if self.eat('=') {
                    tok!(TokenKind::PlusEq, "+=")
                } else {
                    tok!(TokenKind::Plus, "+")
                }
            }

            // ── Minus / Arrow ─────────────────────────────────────────────
            '-' => {
                if self.eat('>') {
                    tok!(TokenKind::Arrow, "->")
                } else if self.eat('=') {
                    tok!(TokenKind::MinusEq, "-=")
                } else {
                    tok!(TokenKind::Minus, "-")
                }
            }

            // ── Star ──────────────────────────────────────────────────────
            '*' => {
                if self.eat('*') {
                    tok!(TokenKind::StarStar, "**")
                } else if self.eat('=') {
                    tok!(TokenKind::StarEq, "*=")
                } else {
                    tok!(TokenKind::Star, "*")
                }
            }

            // ── Slash / comment (already consumed above) ──────────────────
            '/' => {
                if self.eat('/') {
                    tok!(TokenKind::FloorDiv, "//")
                } else if self.eat('=') {
                    tok!(TokenKind::SlashEq, "/=")
                } else {
                    tok!(TokenKind::Slash, "/")
                }
            }

            // ── Percent ───────────────────────────────────────────────────
            '%' => {
                if self.eat('=') {
                    tok!(TokenKind::PercentEq, "%=")
                } else {
                    tok!(TokenKind::Percent, "%")
                }
            }

            // ── Ampersand ─────────────────────────────────────────────────
            '&' => {
                if self.eat('&') {
                    tok!(TokenKind::AmpAmp, "&&")
                } else if self.eat('=') {
                    tok!(TokenKind::AmpEq, "&=")
                } else {
                    tok!(TokenKind::Ampersand, "&")
                }
            }

            // ── Pipe ──────────────────────────────────────────────────────
            '|' => {
                if self.eat('|') {
                    tok!(TokenKind::PipePipe, "||")
                } else if self.eat('=') {
                    tok!(TokenKind::PipeEq, "|=")
                } else {
                    tok!(TokenKind::Pipe, "|")
                }
            }

            // ── Caret ─────────────────────────────────────────────────────
            '^' => {
                if self.eat('*') {
                    tok!(TokenKind::OuterProd, "^*")
                } else if self.eat('=') {
                    tok!(TokenKind::CaretEq, "^=")
                } else {
                    tok!(TokenKind::Caret, "^")
                }
            }

            // ── Less-than / shift-left ────────────────────────────────────
            '<' => {
                if self.eat('<') {
                    tok!(TokenKind::LtLt, "<<")
                } else if self.eat('=') {
                    tok!(TokenKind::LtEq, "<=")
                } else {
                    tok!(TokenKind::Lt, "<")
                }
            }

            // ── Greater-than / shift-right ────────────────────────────────
            '>' => {
                if self.eat('>') {
                    tok!(TokenKind::GtGt, ">>")
                } else if self.eat('=') {
                    tok!(TokenKind::GtEq, ">=")
                } else {
                    tok!(TokenKind::Gt, ">")
                }
            }

            // ── Equals / fat-arrow ────────────────────────────────────────
            '=' => {
                if self.eat('=') {
                    tok!(TokenKind::EqEq, "==")
                } else if self.eat('>') {
                    tok!(TokenKind::FatArrow, "=>")
                } else {
                    tok!(TokenKind::Eq, "=")
                }
            }

            // ── Bang ──────────────────────────────────────────────────────
            '!' => {
                if self.eat('=') {
                    tok!(TokenKind::BangEq, "!=")
                } else {
                    tok!(TokenKind::Bang, "!")
                }
            }

            // ── Colon ─────────────────────────────────────────────────────
            ':' => {
                if self.eat(':') {
                    tok!(TokenKind::ColonColon, "::")
                } else {
                    tok!(TokenKind::Colon, ":")
                }
            }

            // ── Single-character tokens ───────────────────────────────────
            '~' => tok!(TokenKind::Tilde,     "~"),
            ';' => tok!(TokenKind::Semicolon, ";"),
            ',' => tok!(TokenKind::Comma,     ","),
            '?' => tok!(TokenKind::Question,  "?"),
            '#' => tok!(TokenKind::Hash,      "#"),
            '(' => tok!(TokenKind::LParen,    "("),
            ')' => tok!(TokenKind::RParen,    ")"),
            '[' => tok!(TokenKind::LBracket,  "["),
            ']' => tok!(TokenKind::RBracket,  "]"),
            '{' => tok!(TokenKind::LBrace,    "{"),
            '}' => tok!(TokenKind::RBrace,    "}"),

            unknown => {
                let span = self.span_from(start, line, col);
                return Err(LexError::new(
                    format!("unexpected character `{}`", unknown),
                    span,
                ));
            }
        };

        Ok(token)
    }

    // ── Full tokenisation ─────────────────────────────────────────────────

    /// Tokenise the entire source string into a `Vec<Token>`.
    /// The final element is always `TokenKind::Eof`.
    /// Errors are accumulated; tokenisation continues past individual bad
    /// characters so the caller can report multiple problems at once.
    pub fn tokenize(&mut self) -> (Vec<Token>, Vec<LexError>) {
        // Heuristic capacity to reduce reallocations in large-source lexing.
        let mut tokens = Vec::with_capacity((self.src.len() / 4).max(16));
        let mut errors = Vec::with_capacity(4);

        loop {
            match self.next_token() {
                Ok(tok) => {
                    let is_eof = tok.is_eof();
                    tokens.push(tok);
                    if is_eof {
                        break;
                    }
                }
                Err(e) => {
                    errors.push(e);
                    // Skip one byte and continue so we surface as many
                    // errors as possible in one pass.
                    self.advance();
                }
            }
        }

        (tokens, errors)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(src: &str) -> Vec<TokenKind> {
        let mut l = Lexer::new(src);
        let (tokens, errors) = l.tokenize();
        assert!(errors.is_empty(), "unexpected lex errors: {:?}", errors);
        tokens.into_iter().map(|t| t.kind).collect()
    }

    // ── Tensor type syntax ────────────────────────────────────────────────

    #[test]
    fn test_tensor_type_tokens() {
        // tensor<f32>[128, 128]
        let kinds = lex("tensor<f32>[128, 128]");
        assert_eq!(kinds, vec![
            TokenKind::KwTensor,
            TokenKind::Lt,
            TokenKind::KwF32,
            TokenKind::Gt,
            TokenKind::LBracket,
            TokenKind::IntLit { value: 128, base: NumericBase::Decimal },
            TokenKind::Comma,
            TokenKind::IntLit { value: 128, base: NumericBase::Decimal },
            TokenKind::RBracket,
            TokenKind::Eof,
        ]);
    }

    // ── MatMul operator ───────────────────────────────────────────────────

    #[test]
    fn test_matmul_operator() {
        let kinds = lex("C = A @ B");
        assert_eq!(kinds, vec![
            TokenKind::Ident("C".into()),
            TokenKind::Eq,
            TokenKind::Ident("A".into()),
            TokenKind::MatMul,
            TokenKind::Ident("B".into()),
            TokenKind::Eof,
        ]);
    }

    // ── Device annotations ────────────────────────────────────────────────

    #[test]
    fn test_device_annotations() {
        let kinds = lex("@gpu @cpu @tpu");
        assert_eq!(kinds, vec![
            TokenKind::AtGpu,
            TokenKind::AtCpu,
            TokenKind::AtTpu,
            TokenKind::Eof,
        ]);
    }

    // ── Hadamard operators ────────────────────────────────────────────────

    #[test]
    fn test_hadamard_ops() {
        let kinds = lex("A .* B");
        assert_eq!(kinds, vec![
            TokenKind::Ident("A".into()),
            TokenKind::HadamardMul,
            TokenKind::Ident("B".into()),
            TokenKind::Eof,
        ]);
    }

    // ── Tensor concatenation ──────────────────────────────────────────────

    #[test]
    fn test_tensor_concat() {
        let kinds = lex("A ++ B");
        assert_eq!(kinds, vec![
            TokenKind::Ident("A".into()),
            TokenKind::TensorConcat,
            TokenKind::Ident("B".into()),
            TokenKind::Eof,
        ]);
    }

    // ── Integer bases ─────────────────────────────────────────────────────

    #[test]
    fn test_integer_bases() {
        let kinds = lex("0xFF 0o77 0b1010 42");
        assert_eq!(kinds, vec![
            TokenKind::IntLit { value: 255, base: NumericBase::Hex },
            TokenKind::IntLit { value: 63,  base: NumericBase::Octal },
            TokenKind::IntLit { value: 10,  base: NumericBase::Binary },
            TokenKind::IntLit { value: 42,  base: NumericBase::Decimal },
            TokenKind::Eof,
        ]);
    }

    // ── Float literals ────────────────────────────────────────────────────

    #[test]
    fn test_float_literal() {
        let kinds = lex("3.14 1.0e-3");
        assert!(matches!(kinds[0], TokenKind::FloatLit { .. }));
        assert!(matches!(kinds[1], TokenKind::FloatLit { .. }));
    }

    // ── String literals ───────────────────────────────────────────────────

    #[test]
    fn test_string_literal() {
        let kinds = lex(r#""hello\nworld""#);
        assert_eq!(kinds, vec![
            TokenKind::StringLit { value: "hello\nworld".into() },
            TokenKind::Eof,
        ]);
    }

    // ── Line comments ─────────────────────────────────────────────────────

    #[test]
    fn test_line_comment() {
        let kinds = lex("let x // this is a comment\n= 5");
        assert_eq!(kinds, vec![
            TokenKind::KwLet,
            TokenKind::Ident("x".into()),
            TokenKind::Eq,
            TokenKind::IntLit { value: 5, base: NumericBase::Decimal },
            TokenKind::Eof,
        ]);
    }

    // ── Block comments (nested) ───────────────────────────────────────────

    #[test]
    fn test_nested_block_comment() {
        let kinds = lex("a /* outer /* inner */ still outer */ b");
        assert_eq!(kinds, vec![
            TokenKind::Ident("a".into()),
            TokenKind::Ident("b".into()),
            TokenKind::Eof,
        ]);
    }

    // ── DotDot vs HadamardMul ─────────────────────────────────────────────

    #[test]
    fn test_dotdot_not_hadamard() {
        let kinds = lex("0..10");
        assert_eq!(kinds, vec![
            TokenKind::IntLit { value: 0, base: NumericBase::Decimal },
            TokenKind::DotDot,
            TokenKind::IntLit { value: 10, base: NumericBase::Decimal },
            TokenKind::Eof,
        ]);
    }

    // ── Span tracking ─────────────────────────────────────────────────────

    #[test]
    fn test_span_line_col() {
        let mut l = Lexer::new("let\nx");
        let (tokens, _) = l.tokenize();
        assert_eq!(tokens[0].span.line, 1);
        assert_eq!(tokens[0].span.col,  1);
        assert_eq!(tokens[1].span.line, 2);
        assert_eq!(tokens[1].span.col,  1);
    }

    // ── Grad keyword ──────────────────────────────────────────────────────

    #[test]
    fn test_grad_keyword() {
        let kinds = lex("grad A");
        assert_eq!(kinds, vec![
            TokenKind::KwGrad,
            TokenKind::Ident("A".into()),
            TokenKind::Eof,
        ]);
    }

    // ── Full mini-program ─────────────────────────────────────────────────

    #[test]
    fn test_mini_program() {
        let src = r#"
            @gpu
            fn forward(A: tensor<f32>[128, 128], B: tensor<f32>[128, 128]) -> tensor<f32>[128, 128] {
                let C = A @ B
                return C
            }
        "#;
        let mut l = Lexer::new(src);
        let (tokens, errors) = l.tokenize();
        assert!(errors.is_empty(), "{:?}", errors);
        // Spot-check a few key tokens.
        let kinds: Vec<_> = tokens.iter().map(|t| &t.kind).collect();
        assert!(kinds.contains(&&TokenKind::AtGpu));
        assert!(kinds.contains(&&TokenKind::KwFn));
        assert!(kinds.contains(&&TokenKind::KwTensor));
        assert!(kinds.contains(&&TokenKind::MatMul));
    }
}
