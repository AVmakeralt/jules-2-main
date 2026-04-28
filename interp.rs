// ─────────────────────────────────────────────────────────────────────────────
// Cargo.toml additions required for this optimised build:
//
//   [dependencies]
//   rustc-hash = "1"       # FxHashMap / FxHashSet
//   matrixmultiply = "0.3" # already present
//
// Recommended profile settings for maximum throughput:
//
//   [profile.release]
//   opt-level = 3
//   lto = "fat"            # link-time optimisation across crates
//   codegen-units = 1      # single CGU → best inlining
//   panic = "abort"        # removes unwinding machinery
// ─────────────────────────────────────────────────────────────────────────────

// =============================================================================
// jules/src/interp.rs
//
// Tree-walking Interpreter / Virtual Machine for the Jules programming language.
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │  Design overview                                                         │
// │                                                                          │
// │  Strategy: direct AST evaluation (tree-walking interpreter).             │
// │  This is the fastest-to-get-running execution strategy and serves as     │
// │  the reference implementation.  A compiler backend (Cranelift / WASM)   │
// │  can be layered on top later using the same runtime primitives.          │
// │                                                                          │
// │  Feature targets implemented here:                                       │
// │    • CPU scalar + vector/matrix arithmetic                               │
// │    • Tensor operations (matmul, hadamard, concat, grad tracking)         │
// │    • SIMD hint: interpreted as parallel Rayon-style iteration            │
// │    • ECS World: sparse-set component storage, entity lifecycle           │
// │    • System scheduler: topological ordering + deterministic tick         │
// │    • spawn / sync / atomic concurrency primitives (thread-pool based)    │
// │    • Agent runtime: behaviour tick, perception, memory decay             │
// │    • Neural-network forward pass (CPU tensor execution)                  │
// │    • Training loop: episode rollout + reward aggregation + SGD/Adam      │
// │    • GPU path: trait-based dispatch ready for wgpu/CUDA backends          │
// └─────────────────────────────────────────────────────────────────────────┘
// =============================================================================

#![allow(
    clippy::match_single_binding,
    clippy::large_enum_variant,
    clippy::too_many_arguments
)]

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

// ── Fast HashMap — ~2x faster than SipHash for short string keys ─────────────
use rustc_hash::FxHashMap;

// ── Direct threading: when compiling with nightly, use computed-goto labels ──
//   This eliminates the match dispatch overhead in the hot VM loop.
//   On stable Rust we fall back to the match-based dispatch (still fast).
#[cfg(feature = "nightly")]
macro_rules! direct_threading_dispatch {
    ($pc:ident, $instrs:ident, $next:ident) => {
        goto next;
    };
}

// ── Force inline on hot-path functions ──────────────────────────────────────
#[allow(unused_macros)]
macro_rules! hot_path {
    () => {
        #[inline(always)]
    };
}

use crate::ast::{
    Activation, AgentDecl, AssignOpKind, BinOpKind, Block, ElemType, EntityQuery, Expr, FnDecl,
    Item, ModelDecl, ModelLayer, NormKind, OptimizerKind, ParallelismHint, Pattern, PoolOp,
    Program, RecurrentCell, Stmt, SystemDecl, TrainDecl, UnOpKind, VecSize,
};
use crate::game_systems::{InputState, PhysicsShape, PhysicsWorld, RenderCommand, RenderState};
use crate::lexer::Span;
use crate::ml_engine::{ComputationGraph, Optimizer, OptimizerState};
use matrixmultiply::sgemm;

// =============================================================================
// §1  RUNTIME VALUE
// =============================================================================

/// Every runtime value produced or consumed by Jules programs.
///
/// Cloning is cheap for scalars; tensors are reference-counted so large
/// allocations are not duplicated on every assignment.
#[derive(Debug, Clone)]
pub enum Value {
    // ── Scalars ───────────────────────────────────────────────────────────────
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Unit,

    // ── SIMD vectors (stored as flat f32 arrays) ──────────────────────────────
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    IVec2([i32; 2]),
    IVec3([i32; 3]),
    IVec4([i32; 4]),
    Mat2([[f32; 2]; 2]),
    Mat3([[f32; 3]; 3]),
    Mat4([[f32; 4]; 4]),
    Quat([f32; 4]), // [x, y, z, w]

    // ── Data pipelines / data loaders (Feature 8) ─────────────────────────────
    DataLoader(Arc<Mutex<DataLoader>>),

    // ── Tensors (Feature 1) ───────────────────────────────────────────────────
    /// Tensor with thread-safe access (for parallel execution)
    Tensor(Arc<RwLock<Tensor>>),
    /// Fast single-threaded tensor (no locking overhead)
    TensorFast(Arc<RefCell<Tensor>>),

    // ── Compound ─────────────────────────────────────────────────────────────
    Tuple(Vec<Value>),
    Array(Arc<Mutex<Vec<Value>>>),
    Struct {
        name: String,
        fields: FxHashMap<String, Value>,
    },
    /// HashMap: key -> value pairs (keys currently strings)
    HashMap(Arc<Mutex<FxHashMap<String, Value>>>),

    // ── Option / Result types ─────────────────────────────────────────────────
    /// `Some(value)` or `None` (for Option<T>)
    Some(Box<Value>),
    None,
    /// `Ok(value)` for Result<T, E>
    Ok(Box<Value>),
    /// `Err(value)` for Result<T, E>
    Err(Box<Value>),

    // ── Callable ─────────────────────────────────────────────────────────────
    /// A user-defined function closure (captures its definition scope).
    Fn(Arc<FnClosure>),

    // ── ECS handles ──────────────────────────────────────────────────────────
    Entity(EntityId),
    World(Arc<Mutex<EcsWorld>>),

    // ── Neural-network model handle ───────────────────────────────────────────
    Model(Arc<Mutex<NnModel>>),

    // ── Control flow signals (never escape to user code) ─────────────────────
    Return(Box<Value>),
    Break(Option<Box<Value>>),
    Continue,
}

#[derive(Debug, Clone)]
pub struct DataLoader {
    pub samples: Vec<Value>,
    pub batch_size: usize,
    pub index: usize,
    pub shuffle: bool,
}

impl DataLoader {
    #[inline]
    pub fn next_batch(&mut self) -> Option<Value> {
        if self.index >= self.samples.len() {
            return None;
        }
        let end = (self.index + self.batch_size).min(self.samples.len());
        // Clone only the batch slice; the rest stays in-place.
        let chunk: Vec<Value> = self.samples[self.index..end].to_vec();
        self.index = end;
        Some(Value::Array(Arc::new(Mutex::new(chunk))))
    }

    pub fn has_next(&self) -> bool {
        self.index < self.samples.len()
    }

    pub fn reset(&mut self) {
        self.index = 0;
        if self.shuffle {
            // deterministic shuffle so tests remain stable
            let mut seed = 4211_u64;
            for i in (1..self.samples.len()).rev() {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = (seed % (i as u64 + 1)) as usize;
                self.samples.swap(i, j);
            }
        }
    }
}

impl Value {
    pub fn type_name(&self) -> &str {
        match self {
            Value::I8(_) => "i8",
            Value::I16(_) => "i16",
            Value::I32(_) => "i32",
            Value::I64(_) => "i64",
            Value::U8(_) => "u8",
            Value::U16(_) => "u16",
            Value::U32(_) => "u32",
            Value::U64(_) => "u64",
            Value::F32(_) => "f32",
            Value::F64(_) => "f64",
            Value::Bool(_) => "bool",
            Value::Str(_) => "str",
            Value::Unit => "()",
            Value::Vec2(_) => "vec2",
            Value::Vec3(_) => "vec3",
            Value::Vec4(_) => "vec4",
            Value::IVec2(_) => "ivec2",
            Value::IVec3(_) => "ivec3",
            Value::IVec4(_) => "ivec4",
            Value::Mat2(_) => "mat2",
            Value::Mat3(_) => "mat3",
            Value::Mat4(_) => "mat4",
            Value::Quat(_) => "quat",
            Value::Tensor(_) => "tensor",
            Value::TensorFast(_) => "tensor",
            Value::DataLoader(_) => "dataloader",
            Value::Tuple(_) => "tuple",
            Value::Array(_) => "array",
            Value::HashMap(_) => "map",
            Value::Struct { name, .. } => name,
            Value::Some(_) => "Some",
            Value::None => "None",
            Value::Ok(_) => "Ok",
            Value::Err(_) => "Err",
            Value::Fn(_) => "fn",
            Value::Entity(_) => "entity",
            Value::World(_) => "world",
            Value::Model(_) => "model",
            Value::Return(_) | Value::Break(_) | Value::Continue => "<control-flow>",
        }
    }

    /// Extract f64 for arithmetic, coercing all numeric types.
    #[inline(always)]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F32(x) => Some(*x as f64),
            Value::F64(x) => Some(*x),
            Value::I32(x) => Some(*x as f64),
            Value::I64(x) => Some(*x as f64),
            Value::U32(x) => Some(*x as f64),
            Value::U64(x) => Some(*x as f64),
            Value::I8(x) => Some(*x as f64),
            Value::I16(x) => Some(*x as f64),
            Value::U8(x) => Some(*x as f64),
            Value::U16(x) => Some(*x as f64),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::I32(x) => Some(*x as i64),
            Value::I64(x) => Some(*x),
            Value::U32(x) => Some(*x as i64),
            Value::U64(x) => Some(*x as i64),
            Value::I8(x) => Some(*x as i64),
            Value::I16(x) => Some(*x as i64),
            Value::U8(x) => Some(*x as i64),
            Value::U16(x) => Some(*x as i64),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn as_bool(&self) -> Option<bool> {
        if let Value::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::I32(x) => *x != 0,
            Value::F32(x) => *x != 0.0,
            _ => true,
        }
    }

    /// True for any of the control-flow signal variants.
    #[inline(always)]
    pub fn is_signal(&self) -> bool {
        matches!(self, Value::Return(_) | Value::Break(_) | Value::Continue)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::I8(x) => write!(f, "{x}"),
            Value::I16(x) => write!(f, "{x}"),
            Value::I32(x) => write!(f, "{x}"),
            Value::I64(x) => write!(f, "{x}"),
            Value::U8(x) => write!(f, "{x}"),
            Value::U16(x) => write!(f, "{x}"),
            Value::U32(x) => write!(f, "{x}"),
            Value::U64(x) => write!(f, "{x}"),
            Value::F32(x) => write!(f, "{x}"),
            Value::F64(x) => write!(f, "{x}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Str(s) => write!(f, "{s}"),
            Value::Unit => write!(f, "()"),
            Value::Vec2(v) => write!(f, "vec2({}, {})", v[0], v[1]),
            Value::Vec3(v) => write!(f, "vec3({}, {}, {})", v[0], v[1], v[2]),
            Value::Vec4(v) => write!(f, "vec4({}, {}, {}, {})", v[0], v[1], v[2], v[3]),
            Value::Mat3(_) => write!(f, "mat3(…)"),
            Value::Mat4(_) => write!(f, "mat4(…)"),
            Value::Quat(q) => write!(f, "quat({}, {}, {}, {})", q[0], q[1], q[2], q[3]),
            Value::Tensor(t) => {
                let t = t.read().unwrap();
                write!(f, "tensor<{:?}>{:?}", t.elem, t.shape)
            }
            Value::Tuple(vs) => {
                let inner: Vec<_> = vs.iter().map(|v| v.to_string()).collect();
                write!(f, "({})", inner.join(", "))
            }
            Value::Struct { name, .. } => write!(f, "{name} {{ … }}"),
            Value::DataLoader(d) => {
                let d = d.lock().unwrap();
                write!(
                    f,
                    "dataloader(batch_size={}, index={}, total={})",
                    d.batch_size,
                    d.index,
                    d.samples.len()
                )
            }
            Value::Some(v) => write!(f, "Some({})", v),
            Value::None => write!(f, "None"),
            Value::Ok(v) => write!(f, "Ok({})", v),
            Value::Err(v) => write!(f, "Err({})", v),
            Value::HashMap(m) => {
                let m = m.lock().unwrap();
                write!(f, "{{ {} items }}", m.len())
            }
            Value::Entity(id) => write!(f, "Entity({id})"),
            Value::World(_) => write!(f, "<world>"),
            Value::Model(_) => write!(f, "<model>"),
            Value::Fn(_) => write!(f, "<fn>"),
            Value::Array(a) => {
                let a = a.lock().unwrap();
                write!(f, "[…; {}]", a.len())
            }
            Value::Return(v) => write!(f, "return {v}"),
            Value::Break(_) => write!(f, "break"),
            Value::Continue => write!(f, "continue"),
            _ => write!(f, "<value>"),
        }
    }
}

// =============================================================================
// §2  TENSOR  (Feature 1)
// =============================================================================

/// A dense n-dimensional tensor stored as a flat `f32` vector on the CPU.
/// GPU tensors use the same shape metadata but store data behind a `GpuBuffer`.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub elem: ElemType,
    pub shape: Vec<usize>,
    pub data: TensorStorage,
    /// When `Some`, this tensor has a gradient buffer attached (`@grad`).
    pub grad: Option<Box<Tensor>>,
}

#[derive(Debug, Clone)]
pub enum TensorStorage {
    Cpu(Vec<f32>),
    /// Placeholder for a GPU buffer handle (wgpu BufferId / CUDA pointer).
    Gpu(GpuBufferHandle),
}

/// Opaque handle to a GPU buffer.  Filled in by the GPU backend.
#[derive(Debug, Clone)]
pub struct GpuBufferHandle(pub u64);

impl Tensor {
    #[inline]
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product::<usize>().max(1);
        let mut data = Vec::with_capacity(n);
        data.resize(n, 0.0_f32);
        Tensor {
            elem: ElemType::F32,
            shape,
            data: TensorStorage::Cpu(data),
            grad: None,
        }
    }

    #[inline]
    pub fn from_data(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let expected = shape.iter().product::<usize>().max(1);
        debug_assert_eq!(
            data.len(),
            expected,
            "Tensor::from_data: shape {:?} expects {} elements, got {}",
            shape,
            expected,
            data.len()
        );

        Tensor {
            elem: ElemType::F32,
            shape,
            data: TensorStorage::Cpu(data),
            grad: None,
        }
    }

    #[inline(always)]
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    #[inline(always)]
    fn cpu_data(&self) -> &[f32] {
        match &self.data {
            TensorStorage::Cpu(v) => v,
            TensorStorage::Gpu(_) => panic!("tensor is on GPU; call to_cpu() first"),
        }
    }

    #[inline(always)]
    fn cpu_data_mut(&mut self) -> &mut Vec<f32> {
        match &mut self.data {
            TensorStorage::Cpu(v) => v,
            TensorStorage::Gpu(_) => panic!("tensor is on GPU"),
        }
    }

    /// Matrix multiply  C = A @ B.
    /// A: [M, K], B: [K, N] → C: [M, N].
    /// Uses cache-blocked (tiled) 32×32 GEMM for CPU performance.
    #[inline]
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor, RuntimeError> {
        if self.shape.len() < 2 || rhs.shape.len() < 2 {
            return Err(RuntimeError::new("matmul requires ≥2-D tensors"));
        }

        if self.shape.len() != rhs.shape.len() {
            return Err(RuntimeError::new("matmul requires matching tensor rank"));
        }

        let (m, k) = (
            self.shape[self.shape.len() - 2],
            self.shape[self.shape.len() - 1],
        );
        let (k2, n) = (
            rhs.shape[rhs.shape.len() - 2],
            rhs.shape[rhs.shape.len() - 1],
        );
        if k != k2 {
            return Err(RuntimeError::new(format!(
                "matmul shape mismatch: [{m}, {k}] @ [{k2}, {n}]"
            )));
        }

        let batch_shape = &self.shape[..self.shape.len() - 2];
        let rhs_batch_shape = &rhs.shape[..rhs.shape.len() - 2];
        if batch_shape != rhs_batch_shape {
            return Err(RuntimeError::new(format!(
                "matmul batch shape mismatch: {:?} vs {:?}",
                batch_shape, rhs_batch_shape
            )));
        }

        let batch_count = batch_shape.iter().product::<usize>().max(1);
        let a = self.cpu_data();
        let b = rhs.cpu_data();
        let mut c = vec![0.0_f32; batch_count * m * n];

        // Hybrid GEMM strategy:
        // - matrixmultiply::sgemm for large dense workloads (near-BLAS path)
        // - cache-tiled scalar kernel for smaller matrices
        const TILE: usize = 32;
        let use_sgemm = m >= 64 && n >= 64 && k >= 64;
        for batch in 0..batch_count {
            let a_offset = batch * m * k;
            let b_offset = batch * k * n;
            let c_offset = batch * m * n;

            if use_sgemm {
                unsafe {
                    sgemm(
                        m,
                        k,
                        n,
                        1.0,
                        a[a_offset..].as_ptr(),
                        k as isize,
                        1,
                        b[b_offset..].as_ptr(),
                        n as isize,
                        1,
                        0.0,
                        c[c_offset..].as_mut_ptr(),
                        n as isize,
                        1,
                    );
                }
                continue;
            }

            for ii in (0..m).step_by(TILE) {
                for jj in (0..n).step_by(TILE) {
                    for kk in (0..k).step_by(TILE) {
                        let i_end = (ii + TILE).min(m);
                        let j_end = (jj + TILE).min(n);
                        let k_end = (kk + TILE).min(k);
                        for i in ii..i_end {
                            for t in kk..k_end {
                                let a_it = a[a_offset + i * k + t];
                                for j in jj..j_end {
                                    c[c_offset + i * n + j] += a_it * b[b_offset + t * n + j];
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut out_shape = batch_shape.to_vec();
        out_shape.push(m);
        out_shape.push(n);
        Ok(Tensor::from_data(out_shape, c))
    }

    /// Kronecker product  C = A @@ B.
    /// A: [m, n], B: [p, q] → C: [m*p, n*q]
    pub fn kron(&self, rhs: &Tensor) -> Result<Tensor, RuntimeError> {
        if self.shape.len() != 2 || rhs.shape.len() != 2 {
            return Err(RuntimeError::new("Kronecker product requires 2-D tensors"));
        }
        let (m, n) = (self.shape[0], self.shape[1]);
        let (p, q) = (rhs.shape[0], rhs.shape[1]);
        let a = self.cpu_data();
        let b = rhs.cpu_data();
        let mut c = vec![0.0_f32; m * p * n * q];
        for i in 0..m {
            for j in 0..n {
                let a_ij = a[i * n + j];
                for r in 0..p {
                    for s in 0..q {
                        c[(i * p + r) * (n * q) + (j * q + s)] = a_ij * b[r * q + s];
                    }
                }
            }
        }
        Ok(Tensor::from_data(vec![m * p, n * q], c))
    }

    /// Outer product  C = a ^* b.
    /// a: [m], b: [n] → C: [m, n]
    pub fn outer(&self, rhs: &Tensor) -> Result<Tensor, RuntimeError> {
        if self.shape.len() != 1 || rhs.shape.len() != 1 {
            return Err(RuntimeError::new("outer product requires 1-D tensors"));
        }
        let m = self.shape[0];
        let n = rhs.shape[0];
        let a = self.cpu_data();
        let b = rhs.cpu_data();
        let c: Vec<f32> = a
            .iter()
            .flat_map(|&ai| b.iter().map(move |&bj| ai * bj))
            .collect();
        Ok(Tensor::from_data(vec![m, n], c))
    }

    /// Element-wise multiply (Hadamard).
    pub fn hadamard_mul(&self, rhs: &Tensor) -> Result<Tensor, RuntimeError> {
        self.elementwise(rhs, |a, b| a * b, ".*")
    }

    /// Element-wise divide.
    pub fn hadamard_div(&self, rhs: &Tensor) -> Result<Tensor, RuntimeError> {
        self.elementwise(rhs, |a, b| a / b, "./")
    }

    fn elementwise(
        &self,
        rhs: &Tensor,
        op: impl Fn(f32, f32) -> f32,
        name: &str,
    ) -> Result<Tensor, RuntimeError> {
        // Exact shape match (fast path)
        if self.shape == rhs.shape {
            let a = self.cpu_data();
            let b = rhs.cpu_data();
            let c: Vec<f32> = a.iter().zip(b).map(|(x, y)| op(*x, *y)).collect();
            return Ok(Tensor::from_data(self.shape.clone(), c));
        }

        // Numpy-style broadcasting
        let result_shape = broadcast_shape(&self.shape, &rhs.shape).ok_or_else(|| {
            RuntimeError::new(format!(
                "`{name}` shape mismatch: {:?} vs {:?}",
                self.shape, rhs.shape
            ))
        })?;

        let n: usize = result_shape.iter().product();
        let mut c = vec![0.0_f32; n];
        for idx in 0..n {
            let ai = broadcast_index(idx, &result_shape, &self.shape);
            let bi = broadcast_index(idx, &result_shape, &rhs.shape);
            let a = self.cpu_data();
            let b = rhs.cpu_data();
            c[idx] = op(a[ai], b[bi]);
        }
        Ok(Tensor::from_data(result_shape, c))
    }

    /// Floor division element-wise (integer rounding toward −∞).
    pub fn floor_div(&self, rhs: &Tensor) -> Result<Tensor, RuntimeError> {
        self.elementwise(rhs, |a, b| (a / b).floor(), "//")
    }

    /// Concatenate along axis 0.
    pub fn concat(&self, rhs: &Tensor) -> Result<Tensor, RuntimeError> {
        if self.shape.len() != rhs.shape.len() {
            return Err(RuntimeError::new("tensor concat: rank mismatch"));
        }
        for i in 1..self.shape.len() {
            if self.shape[i] != rhs.shape[i] {
                return Err(RuntimeError::new(format!(
                    "tensor concat: dim {} mismatch ({} vs {})",
                    i, self.shape[i], rhs.shape[i]
                )));
            }
        }
        let mut data = self.cpu_data().to_vec();
        data.extend_from_slice(rhs.cpu_data());
        let mut shape = self.shape.clone();
        shape[0] += rhs.shape[0];
        Ok(Tensor::from_data(shape, data))
    }

    /// Attach a zero-initialised gradient buffer.
    pub fn enable_grad(&mut self) {
        if self.grad.is_none() {
            self.grad = Some(Box::new(Tensor::zeros(self.shape.clone())));
        }
    }

    /// Apply an activation function element-wise.
    pub fn apply_activation(&self, act: &Activation) -> Tensor {
        let data: Vec<f32> = self
            .cpu_data()
            .iter()
            .map(|&x| match act {
                Activation::Relu => x.max(0.0),
                Activation::LeakyRelu => {
                    if x > 0.0 {
                        x
                    } else {
                        0.01 * x
                    }
                }
                Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                Activation::Tanh => x.tanh(),
                Activation::Gelu => {
                    0.5 * x
                        * (1.0
                            + ((2.0_f32 / std::f32::consts::PI).sqrt()
                                * (x + 0.044715 * x * x * x))
                                .tanh())
                }
                Activation::Silu => x / (1.0 + (-x).exp()),
                Activation::Elu => {
                    if x > 0.0 {
                        x
                    } else {
                        x.exp() - 1.0
                    }
                }
                Activation::Swish => x / (1.0 + (-x).exp()),
                Activation::Mish => x * (1.0 + x.exp()).ln().tanh(),
                Activation::Softmax => x, // applied below per-row
                Activation::Linear | Activation::Custom(_) => x,
            })
            .collect();

        let mut t = Tensor::from_data(self.shape.clone(), data);

        // Softmax: apply row-wise along last dim.
        if matches!(act, Activation::Softmax) {
            let d = t.cpu_data_mut();
            let cols = *self.shape.last().unwrap_or(&1);
            let rows = d.len() / cols;
            for r in 0..rows {
                let row = &mut d[r * cols..(r + 1) * cols];
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = row.iter().map(|x| (x - max).exp()).sum();
                for x in row.iter_mut() {
                    *x = (*x - max).exp() / sum;
                }
            }
        }
        t
    }

    /// Reduce sum over all elements → scalar.
    pub fn sum_all(&self) -> f32 {
        self.cpu_data().iter().sum()
    }

    /// Scale all elements.
    pub fn scale(&self, s: f32) -> Tensor {
        let data: Vec<f32> = self.cpu_data().iter().map(|x| x * s).collect();
        Tensor::from_data(self.shape.clone(), data)
    }

    /// In-place scaling of all elements (avoids allocation).
    #[inline]
    pub fn scale_inplace(&mut self, s: f32) {
        for v in self.cpu_data_mut().iter_mut() {
            *v *= s;
        }
    }

    /// Add another tensor (in-place on self).
    pub fn add_assign(&mut self, rhs: &Tensor) -> Result<(), RuntimeError> {
        if self.shape != rhs.shape {
            return Err(RuntimeError::new("tensor += shape mismatch"));
        }
        // Avoid intermediate allocation: zip directly over slices.
        let b_ptr = rhs.cpu_data().as_ptr();
        let a = self.cpu_data_mut();
        for (i, x) in a.iter_mut().enumerate() {
            // SAFETY: shapes are equal so b has the same length as a.
            *x += unsafe { *b_ptr.add(i) };
        }
        Ok(())
    }

    /// MSE loss w.r.t. targets.
    pub fn mse_loss(&self, targets: &Tensor) -> Result<f32, RuntimeError> {
        if self.shape != targets.shape {
            return Err(RuntimeError::new("MSE: shape mismatch"));
        }
        let a = self.cpu_data();
        let b = targets.cpu_data();
        let loss: f32 = a.iter().zip(b).map(|(p, t)| (p - t).powi(2)).sum::<f32>() / a.len() as f32;
        Ok(loss)
    }

    /// Cross-entropy loss (prediction probabilities vs one-hot targets).
    pub fn cross_entropy_loss(&self, targets: &Tensor) -> Result<f32, RuntimeError> {
        if self.shape != targets.shape {
            return Err(RuntimeError::new("CE: shape mismatch"));
        }
        let a = self.cpu_data();
        let b = targets.cpu_data();
        let loss: f32 = a
            .iter()
            .zip(b)
            .map(|(p, t)| -t * p.max(1e-9).ln())
            .sum::<f32>()
            / self.shape[0] as f32;
        Ok(loss)
    }
}

// =============================================================================
// §3  ECS WORLD  (Feature 2)
// =============================================================================

pub type EntityId = u64;
thread_local! {
    static VM_REG_POOL: RefCell<Vec<Value>> = RefCell::new(Vec::new());
}

/// The Entity-Component-System world that backs all `system` and entity-for loops.
///
/// Component storage uses a sparse-set layout per component type:
///   component_name → (dense vec of EntityId, dense vec of component Value)
/// The two vecs stay in sync: `dense_ids[i]` owns `dense_vals[i]`.
#[derive(Debug, Default)]
pub struct EcsWorld {
    next_id: EntityId,
    alive: std::collections::HashSet<EntityId>,
    /// component_type → SparseSet
    components: FxHashMap<String, SparseSet>,
    /// Pending events (signal_name → Vec<EntityId>)
    events: FxHashMap<String, Vec<EntityId>>,
    vec3_plan_cache: FxHashMap<String, Vec3PlanCache>,
    fused_plan_cache: FxHashMap<String, FusedPlanCache>,
}

/// Sparse-set component storage.
#[derive(Debug, Default)]
struct SparseSet {
    /// Maps EntityId → index into `dense_ids` / `dense_vals`.
    sparse: FxHashMap<EntityId, usize>,
    dense_ids: Vec<EntityId>,
    dense_vals: Vec<Value>,
    version: u64,
}

#[derive(Debug, Default)]
struct Vec3PlanCache {
    pos_version: u64,
    vel_version: u64,
    chunk_size: usize,
    pairs: Vec<(usize, usize)>,
}

#[derive(Debug, Default)]
struct FusedPlanCache {
    pos_version: u64,
    vel_version: u64,
    health_version: u64,
    damage_version: u64,
    chunk_size: usize,
    tuples: Vec<(usize, usize, usize, usize)>,
}

impl SparseSet {
    #[inline]
    fn insert(&mut self, id: EntityId, val: Value) {
        if let Some(&idx) = self.sparse.get(&id) {
            self.dense_vals[idx] = val;
        } else {
            let idx = self.dense_ids.len();
            self.sparse.insert(id, idx);
            self.dense_ids.push(id);
            self.dense_vals.push(val);
            self.version = self.version.wrapping_add(1);
        }
    }

    #[inline]
    fn get(&self, id: EntityId) -> Option<&Value> {
        self.sparse.get(&id).map(|&i| &self.dense_vals[i])
    }

    #[inline]
    fn get_mut(&mut self, id: EntityId) -> Option<&mut Value> {
        self.sparse
            .get(&id)
            .copied()
            .map(|i| &mut self.dense_vals[i])
    }

    fn remove(&mut self, id: EntityId) {
        if let Some(idx) = self.sparse.remove(&id) {
            let last = self.dense_ids.len() - 1;
            if idx != last {
                let moved_id = self.dense_ids[last];
                self.dense_ids.swap(idx, last);
                self.dense_vals.swap(idx, last);
                self.sparse.insert(moved_id, idx);
            }
            self.dense_ids.pop();
            self.dense_vals.pop();
            self.version = self.version.wrapping_add(1);
        }
    }

    fn entity_ids(&self) -> &[EntityId] {
        &self.dense_ids
    }
}

impl EcsWorld {
    #[inline]
    fn plan_key2(a: &str, b: &str) -> String {
        format!("{a}|{b}")
    }
    #[inline]
    fn plan_key4(a: &str, b: &str, c: &str, d: &str) -> String {
        format!("{a}|{b}|{c}|{d}")
    }

    pub fn spawn(&mut self) -> EntityId {
        let id = self.next_id;
        self.next_id += 1;
        self.alive.insert(id);
        id
    }

    pub fn despawn(&mut self, id: EntityId) {
        self.alive.remove(&id);
        for set in self.components.values_mut() {
            set.remove(id);
        }
    }

    pub fn is_alive(&self, id: EntityId) -> bool {
        self.alive.contains(&id)
    }

    pub fn insert_component(&mut self, id: EntityId, comp_type: &str, val: Value) {
        self.components
            .entry(comp_type.to_owned())
            .or_default()
            .insert(id, val);
    }

    pub fn get_component(&self, id: EntityId, comp_type: &str) -> Option<&Value> {
        self.components.get(comp_type)?.get(id)
    }

    pub fn get_component_mut(&mut self, id: EntityId, comp_type: &str) -> Option<&mut Value> {
        self.components.get_mut(comp_type)?.get_mut(id)
    }

    pub fn remove_component(&mut self, id: EntityId, comp_type: &str) {
        if let Some(set) = self.components.get_mut(comp_type) {
            set.remove(id);
        }
    }

    /// Returns all live entities matching a query.
    pub fn query(&self, with: &[String], without: &[String]) -> Vec<EntityId> {
        let mut out = Vec::new();
        self.query_into(with, without, &mut out);
        out
    }

    /// Writes all live entities matching a query into a caller-provided output buffer.
    pub fn query_into(&self, with: &[String], without: &[String], out: &mut Vec<EntityId>) {
        out.clear();
        let base: Cow<'_, [EntityId]> = if with.is_empty() {
            Cow::Owned(self.alive.iter().copied().collect())
        } else {
            let smallest = with
                .iter()
                .filter_map(|c| self.components.get(c))
                .min_by_key(|s| s.dense_ids.len());
            match smallest {
                None => return,
                Some(s) => Cow::Borrowed(s.entity_ids()),
            }
        };
        out.reserve(base.len());
        for id in base.iter().copied() {
            if !self.alive.contains(&id) {
                continue;
            }
            if with
                .iter()
                .any(|c| self.components.get(c).map_or(true, |s| s.get(id).is_none()))
            {
                continue;
            }
            if without.iter().any(|c| {
                self.components
                    .get(c)
                    .map_or(false, |s| s.get(id).is_some())
            }) {
                continue;
            }
            out.push(id);
        }
    }

    /// Allocation-lean query for the common 2-component include case.
    #[inline]
    pub fn query2(&self, c1: &str, c2: &str) -> Vec<EntityId> {
        let (Some(s1), Some(s2)) = (self.components.get(c1), self.components.get(c2)) else {
            return Vec::new();
        };
        let (base, other) = if s1.dense_ids.len() <= s2.dense_ids.len() {
            (s1, s2)
        } else {
            (s2, s1)
        };
        let mut out = Vec::with_capacity(base.dense_ids.len());
        for id in &base.dense_ids {
            if self.alive.contains(id) && other.sparse.contains_key(id) {
                out.push(*id);
            }
        }
        out
    }

    /// Emit an event signal for the training loop.
    pub fn emit_event(&mut self, signal: &str, entity: EntityId) {
        self.events
            .entry(signal.to_owned())
            .or_default()
            .push(entity);
    }

    /// Tight linear integration pass for the common `pos += vel * dt` case.
    ///
    /// This avoids per-entity component lookups by iterating directly over the
    /// dense component arrays and joining through sparse indices.
    pub fn integrate_vec3_linear(&mut self, pos_comp: &str, vel_comp: &str, dt: f32) -> usize {
        if pos_comp == vel_comp {
            return 0;
        }
        let Some(mut pos_set) = self.components.remove(pos_comp) else {
            return 0;
        };
        let Some(vel_set) = self.components.get(vel_comp) else {
            self.components.insert(pos_comp.to_owned(), pos_set);
            return 0;
        };

        let mut updated = 0usize;
        if pos_set.dense_ids.len() <= vel_set.dense_ids.len() {
            for i in 0..pos_set.dense_ids.len() {
                let id = pos_set.dense_ids[i];
                let Some(&vi) = vel_set.sparse.get(&id) else {
                    continue;
                };
                if let (Some(Value::Vec3(p)), Some(Value::Vec3(v))) =
                    (pos_set.dense_vals.get_mut(i), vel_set.dense_vals.get(vi))
                {
                    p[0] += v[0] * dt;
                    p[1] += v[1] * dt;
                    p[2] += v[2] * dt;
                    updated += 1;
                }
            }
        } else {
            for vi in 0..vel_set.dense_ids.len() {
                let id = vel_set.dense_ids[vi];
                let Some(&pi) = pos_set.sparse.get(&id) else {
                    continue;
                };
                if let (Some(Value::Vec3(p)), Some(Value::Vec3(v))) =
                    (pos_set.dense_vals.get_mut(pi), vel_set.dense_vals.get(vi))
                {
                    p[0] += v[0] * dt;
                    p[1] += v[1] * dt;
                    p[2] += v[2] * dt;
                    updated += 1;
                }
            }
        }
        self.components.insert(pos_comp.to_owned(), pos_set);
        updated
    }

    /// Batched/fused loop form that keeps hot loops on raw contiguous arrays.
    #[inline]
    pub fn integrate_vec3_linear_fused(
        &mut self,
        pos_comp: &str,
        vel_comp: &str,
        dt: f32,
    ) -> usize {
        if pos_comp == vel_comp {
            return 0;
        }
        let Some(mut pos_set) = self.components.remove(pos_comp) else {
            return 0;
        };
        let Some(vel_set) = self.components.get(vel_comp) else {
            self.components.insert(pos_comp.to_owned(), pos_set);
            return 0;
        };
        let mut updated = 0usize;
        for i in 0..pos_set.dense_ids.len() {
            let id = pos_set.dense_ids[i];
            let Some(&vi) = vel_set.sparse.get(&id) else {
                continue;
            };
            let (p, v) = unsafe {
                let p = pos_set.dense_vals.get_unchecked_mut(i);
                let v = vel_set.dense_vals.get_unchecked(vi);
                (p, v)
            };
            if let (Value::Vec3(p3), Value::Vec3(v3)) = (p, v) {
                p3[0] += v3[0] * dt;
                p3[1] += v3[1] * dt;
                p3[2] += v3[2] * dt;
                updated += 1;
            }
        }
        self.components.insert(pos_comp.to_owned(), pos_set);
        updated
    }

    /// Archetype-like chunked precomputed join for `pos += vel * dt`.
    /// Reuses precomputed dense indices and executes branch-light chunk loops.
    pub fn integrate_vec3_chunked_precomputed(
        &mut self,
        pos_comp: &str,
        vel_comp: &str,
        dt: f32,
        chunk_size: usize,
    ) -> usize {
        if pos_comp == vel_comp {
            return 0;
        }
        let Some(pos_set_ref) = self.components.get(pos_comp) else {
            return 0;
        };
        let Some(vel_set_ref) = self.components.get(vel_comp) else {
            return 0;
        };
        let pos_v = pos_set_ref.version;
        let vel_v = vel_set_ref.version;
        let key = Self::plan_key2(pos_comp, vel_comp);
        let mut cache = self.vec3_plan_cache.remove(&key).unwrap_or_default();
        let chunk_size = chunk_size.max(1);
        if cache.pos_version != pos_v
            || cache.vel_version != vel_v
            || cache.chunk_size != chunk_size
        {
            cache.pairs.clear();
            cache
                .pairs
                .reserve(pos_set_ref.dense_ids.len().min(vel_set_ref.dense_ids.len()));
            for (pi, id) in pos_set_ref.dense_ids.iter().enumerate() {
                if let Some(&vi) = vel_set_ref.sparse.get(id) {
                    cache.pairs.push((pi, vi));
                }
            }
            cache.pos_version = pos_v;
            cache.vel_version = vel_v;
            cache.chunk_size = chunk_size;
        }

        let Some(mut pos_set) = self.components.remove(pos_comp) else {
            self.vec3_plan_cache.insert(key, cache);
            return 0;
        };
        let Some(vel_set) = self.components.get(vel_comp) else {
            self.components.insert(pos_comp.to_owned(), pos_set);
            self.vec3_plan_cache.insert(key, cache);
            return 0;
        };
        let mut updated = 0usize;
        for chunk in cache.pairs.chunks(cache.chunk_size) {
            for (pi, vi) in chunk {
                let (p, v) = unsafe {
                    let p = pos_set.dense_vals.get_unchecked_mut(*pi);
                    let v = vel_set.dense_vals.get_unchecked(*vi);
                    (p, v)
                };
                if let (Value::Vec3(p3), Value::Vec3(v3)) = (p, v) {
                    p3[0] += v3[0] * dt;
                    p3[1] += v3[1] * dt;
                    p3[2] += v3[2] * dt;
                    updated += 1;
                }
            }
        }
        self.components.insert(pos_comp.to_owned(), pos_set);
        self.vec3_plan_cache.insert(key, cache);
        updated
    }

    /// Superoptimizer pass for `pos += vel * dt`.
    ///
    /// Uses the same cached join plan as the chunked pass but executes an
    /// aggressively unrolled inner loop with `mul_add` to reduce instruction
    /// count and improve autovectorization in release mode.
    pub fn integrate_vec3_superoptimizer(
        &mut self,
        pos_comp: &str,
        vel_comp: &str,
        dt: f32,
        chunk_size: usize,
    ) -> usize {
        if pos_comp == vel_comp {
            return 0;
        }
        let Some(pos_set_ref) = self.components.get(pos_comp) else {
            return 0;
        };
        let Some(vel_set_ref) = self.components.get(vel_comp) else {
            return 0;
        };
        let pos_v = pos_set_ref.version;
        let vel_v = vel_set_ref.version;
        let key = Self::plan_key2(pos_comp, vel_comp);
        let mut cache = self.vec3_plan_cache.remove(&key).unwrap_or_default();
        let chunk_size = chunk_size.max(1);
        if cache.pos_version != pos_v
            || cache.vel_version != vel_v
            || cache.chunk_size != chunk_size
        {
            cache.pairs.clear();
            cache
                .pairs
                .reserve(pos_set_ref.dense_ids.len().min(vel_set_ref.dense_ids.len()));
            for (pi, id) in pos_set_ref.dense_ids.iter().enumerate() {
                if let Some(&vi) = vel_set_ref.sparse.get(id) {
                    cache.pairs.push((pi, vi));
                }
            }
            cache.pos_version = pos_v;
            cache.vel_version = vel_v;
            cache.chunk_size = chunk_size;
        }

        let Some(mut pos_set) = self.components.remove(pos_comp) else {
            self.vec3_plan_cache.insert(key, cache);
            return 0;
        };
        let Some(vel_set) = self.components.get(vel_comp) else {
            self.components.insert(pos_comp.to_owned(), pos_set);
            self.vec3_plan_cache.insert(key, cache);
            return 0;
        };
        let mut updated = 0usize;
        for chunk in cache.pairs.chunks(cache.chunk_size) {
            let mut i = 0usize;
            while i + 3 < chunk.len() {
                let slots = [chunk[i], chunk[i + 1], chunk[i + 2], chunk[i + 3]];
                if Self::simd_update_vec3_x4(&mut pos_set, vel_set, slots, dt) {
                    updated += 4;
                    i += 4;
                    continue;
                }
                for (pi, vi) in slots {
                    let (p, v) = unsafe {
                        let p = pos_set.dense_vals.get_unchecked_mut(pi);
                        let v = vel_set.dense_vals.get_unchecked(vi);
                        (p, v)
                    };
                    if let (Value::Vec3(p3), Value::Vec3(v3)) = (p, v) {
                        p3[0] = v3[0].mul_add(dt, p3[0]);
                        p3[1] = v3[1].mul_add(dt, p3[1]);
                        p3[2] = v3[2].mul_add(dt, p3[2]);
                        updated += 1;
                    }
                }
                i += 4;
            }
            while i < chunk.len() {
                let (pi, vi) = chunk[i];
                let (p, v) = unsafe {
                    let p = pos_set.dense_vals.get_unchecked_mut(pi);
                    let v = vel_set.dense_vals.get_unchecked(vi);
                    (p, v)
                };
                if let (Value::Vec3(p3), Value::Vec3(v3)) = (p, v) {
                    p3[0] = v3[0].mul_add(dt, p3[0]);
                    p3[1] = v3[1].mul_add(dt, p3[1]);
                    p3[2] = v3[2].mul_add(dt, p3[2]);
                    updated += 1;
                }
                i += 1;
            }
        }
        self.components.insert(pos_comp.to_owned(), pos_set);
        self.vec3_plan_cache.insert(key, cache);
        updated
    }

    #[inline]
    fn simd_update_vec3_x4(
        pos_set: &mut SparseSet,
        vel_set: &SparseSet,
        slots: [(usize, usize); 4],
        dt: f32,
    ) -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            use std::arch::x86_64::*;

            let [(pi0, vi0), (pi1, vi1), (pi2, vi2), (pi3, vi3)] = slots;
            let pos_ptr = pos_set.dense_vals.as_mut_ptr();
            let vel_ptr = vel_set.dense_vals.as_ptr();
            let p0 = &mut *pos_ptr.add(pi0);
            let p1 = &mut *pos_ptr.add(pi1);
            let p2 = &mut *pos_ptr.add(pi2);
            let p3 = &mut *pos_ptr.add(pi3);
            let v0 = &*vel_ptr.add(vi0);
            let v1 = &*vel_ptr.add(vi1);
            let v2 = &*vel_ptr.add(vi2);
            let v3 = &*vel_ptr.add(vi3);
            let (
                Value::Vec3(p0),
                Value::Vec3(v0),
                Value::Vec3(p1),
                Value::Vec3(v1),
                Value::Vec3(p2),
                Value::Vec3(v2),
                Value::Vec3(p3),
                Value::Vec3(v3),
            ) = (p0, v0, p1, v1, p2, v2, p3, v3)
            else {
                return false;
            };

            let dtv = _mm_set1_ps(dt);
            let px = _mm_set_ps(p3[0], p2[0], p1[0], p0[0]);
            let py = _mm_set_ps(p3[1], p2[1], p1[1], p0[1]);
            let pz = _mm_set_ps(p3[2], p2[2], p1[2], p0[2]);
            let vx = _mm_set_ps(v3[0], v2[0], v1[0], v0[0]);
            let vy = _mm_set_ps(v3[1], v2[1], v1[1], v0[1]);
            let vz = _mm_set_ps(v3[2], v2[2], v1[2], v0[2]);

            let ox = _mm_add_ps(px, _mm_mul_ps(vx, dtv));
            let oy = _mm_add_ps(py, _mm_mul_ps(vy, dtv));
            let oz = _mm_add_ps(pz, _mm_mul_ps(vz, dtv));

            let mut out_x = [0.0_f32; 4];
            let mut out_y = [0.0_f32; 4];
            let mut out_z = [0.0_f32; 4];
            _mm_storeu_ps(out_x.as_mut_ptr(), ox);
            _mm_storeu_ps(out_y.as_mut_ptr(), oy);
            _mm_storeu_ps(out_z.as_mut_ptr(), oz);

            p0[0] = out_x[0];
            p1[0] = out_x[1];
            p2[0] = out_x[2];
            p3[0] = out_x[3];
            p0[1] = out_y[0];
            p1[1] = out_y[1];
            p2[1] = out_y[2];
            p3[1] = out_y[3];
            p0[2] = out_z[0];
            p1[2] = out_z[1];
            p2[2] = out_z[2];
            p3[2] = out_z[3];
            return true;
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            let _ = (pos_set, vel_set, slots, dt);
            false
        }
    }

    /// Fused chunked system pass:
    /// `pos += vel * dt` and `health -= damage * dt` in one tight loop.
    pub fn integrate_vec3_and_health_chunked(
        &mut self,
        pos_comp: &str,
        vel_comp: &str,
        health_comp: &str,
        damage_comp: &str,
        dt: f32,
        chunk_size: usize,
    ) -> usize {
        let Some(pos_ref) = self.components.get(pos_comp) else {
            return 0;
        };
        let Some(vel_ref) = self.components.get(vel_comp) else {
            return 0;
        };
        let Some(health_ref) = self.components.get(health_comp) else {
            return 0;
        };
        let Some(damage_ref) = self.components.get(damage_comp) else {
            return 0;
        };
        let key = Self::plan_key4(pos_comp, vel_comp, health_comp, damage_comp);
        let mut cache = self.fused_plan_cache.remove(&key).unwrap_or_default();
        let chunk_size = chunk_size.max(1);
        if cache.pos_version != pos_ref.version
            || cache.vel_version != vel_ref.version
            || cache.health_version != health_ref.version
            || cache.damage_version != damage_ref.version
            || cache.chunk_size != chunk_size
        {
            cache.tuples.clear();
            cache
                .tuples
                .reserve(pos_ref.dense_ids.len().min(vel_ref.dense_ids.len()));
            for (pi, id) in pos_ref.dense_ids.iter().enumerate() {
                let (Some(&vi), Some(&hi), Some(&di)) = (
                    vel_ref.sparse.get(id),
                    health_ref.sparse.get(id),
                    damage_ref.sparse.get(id),
                ) else {
                    continue;
                };
                cache.tuples.push((pi, vi, hi, di));
            }
            cache.pos_version = pos_ref.version;
            cache.vel_version = vel_ref.version;
            cache.health_version = health_ref.version;
            cache.damage_version = damage_ref.version;
            cache.chunk_size = chunk_size;
        }

        let Some(mut pos_set) = self.components.remove(pos_comp) else {
            self.fused_plan_cache.insert(key, cache);
            return 0;
        };
        let Some(mut health_set) = self.components.remove(health_comp) else {
            self.components.insert(pos_comp.to_owned(), pos_set);
            self.fused_plan_cache.insert(key, cache);
            return 0;
        };
        let (Some(vel_set), Some(damage_set)) = (
            self.components.get(vel_comp),
            self.components.get(damage_comp),
        ) else {
            self.components.insert(pos_comp.to_owned(), pos_set);
            self.components.insert(health_comp.to_owned(), health_set);
            self.fused_plan_cache.insert(key, cache);
            return 0;
        };

        let mut updated = 0usize;
        for chunk in cache.tuples.chunks(cache.chunk_size) {
            for (pi, vi, hi, di) in chunk {
                let (p, v, h, d) = unsafe {
                    let p = pos_set.dense_vals.get_unchecked_mut(*pi);
                    let v = vel_set.dense_vals.get_unchecked(*vi);
                    let h = health_set.dense_vals.get_unchecked_mut(*hi);
                    let d = damage_set.dense_vals.get_unchecked(*di);
                    (p, v, h, d)
                };
                if let (Value::Vec3(p3), Value::Vec3(v3), Value::F32(hp), Value::F32(dmg)) =
                    (p, v, h, d)
                {
                    p3[0] += v3[0] * dt;
                    p3[1] += v3[1] * dt;
                    p3[2] += v3[2] * dt;
                    *hp -= *dmg * dt;
                    updated += 1;
                }
            }
        }
        self.components.insert(pos_comp.to_owned(), pos_set);
        self.components.insert(health_comp.to_owned(), health_set);
        self.fused_plan_cache.insert(key, cache);
        updated
    }

    /// Drain all events for a named signal.
    pub fn drain_events(&mut self, signal: &str) -> Vec<EntityId> {
        self.events.remove(signal).unwrap_or_default()
    }
}

// Lightweight snapshot of the ECS world used by the frame debugger and scene
// editor. This captures the set of live entities and their components.
pub type ComponentMap = FxHashMap<String, Value>;

#[derive(Debug, Clone)]
pub struct WorldSnapshot {
    pub entities: Vec<(EntityId, ComponentMap)>,
}

impl EcsWorld {
    /// Return a list of component type names currently registered in the world.
    pub fn component_types(&self) -> Vec<String> {
        self.components.keys().cloned().collect()
    }

    /// Capture a serializable snapshot of the current world state.
    pub fn snapshot(&self) -> WorldSnapshot {
        let mut entities: Vec<(EntityId, ComponentMap)> = Vec::new();
        for &id in self.alive.iter() {
            let mut comps: ComponentMap = FxHashMap::default();
            for c in self.component_types() {
                if let Some(v) = self.get_component(id, &c) {
                    comps.insert(c.clone(), v.clone());
                }
            }
            entities.push((id, comps));
        }
        WorldSnapshot { entities }
    }

    /// Restore a snapshot into this world, replacing its current contents.
    pub fn restore_snapshot(&mut self, snap: &WorldSnapshot) {
        self.next_id = 1;
        self.alive.clear();
        self.components.clear();
        for (id, comps) in &snap.entities {
            if *id >= self.next_id {
                self.next_id = *id + 1;
            }
            self.alive.insert(*id);
            for (comp_name, val) in comps {
                self.insert_component(*id, comp_name, val.clone());
            }
        }
    }
}

// =============================================================================
// §4  NEURAL NETWORK MODEL RUNTIME  (Unique Feature 1)
// =============================================================================

/// A weight layer in the runtime model.
#[derive(Debug, Clone)]
pub enum WeightLayer {
    Dense {
        w: Tensor,
        b: Tensor,
        act: Activation,
    },
    Conv2d {
        filters: u64,
        kh: u64,
        kw: u64,
        act: Activation,
    },
    Pool {
        ph: u64,
        pw: u64,
        op: PoolOp,
    },
    Dropout {
        rate: f32,
        training: bool,
    },
    Norm {
        kind: NormKind,
        scale: Tensor,
        shift: Tensor,
    },
    Attention {
        num_heads: u64,
        head_dim: u64,
        wq: Tensor,
        wk: Tensor,
        wv: Tensor,
        wo: Tensor,
    },
    Embed {
        table: Tensor,
    },
    Recurrent {
        units: u64,
        cell: RecurrentCell,
        wh: Tensor,
        wx: Tensor,
        bh: Tensor,
    },
    SubModel {
        name: String,
    },
}

/// The live neural network model with allocated weights.
#[derive(Debug)]
pub struct NnModel {
    pub name: String,
    pub layers: Vec<WeightLayer>,
    pub training: bool,
    /// Gradient accumulator: one tensor per weight tensor, same shape.
    pub grads: Vec<Vec<Tensor>>,
    /// Adam state: first and second moment estimates.
    pub m1: Vec<Vec<Tensor>>,
    pub m2: Vec<Vec<Tensor>>,
    pub step: u64,
}

impl NnModel {
    /// Instantiate a model from its AST declaration, initialising all weights.
    pub fn from_decl(decl: &ModelDecl) -> Self {
        let mut layers = Vec::new();
        let mut last_width: usize = 1;

        for layer in &decl.layers {
            match layer {
                ModelLayer::Input { size, .. } => {
                    last_width = *size as usize;
                }
                ModelLayer::Dense {
                    units,
                    activation,
                    bias,
                    ..
                } => {
                    let u = *units as usize;
                    // He initialisation: std = sqrt(2 / fan_in)
                    let std = (2.0_f32 / last_width as f32).sqrt();
                    let w = Tensor::from_data(
                        vec![last_width, u],
                        (0..last_width * u).map(|_| rand_normal(0.0, std)).collect(),
                    );
                    let b = if *bias {
                        Tensor::zeros(vec![u])
                    } else {
                        Tensor::zeros(vec![u])
                    };
                    layers.push(WeightLayer::Dense {
                        w,
                        b,
                        act: activation.clone(),
                    });
                    last_width = u;
                }
                ModelLayer::Output {
                    units, activation, ..
                } => {
                    let u = *units as usize;
                    let std = (2.0_f32 / last_width as f32).sqrt();
                    let w = Tensor::from_data(
                        vec![last_width, u],
                        (0..last_width * u).map(|_| rand_normal(0.0, std)).collect(),
                    );
                    let b = Tensor::zeros(vec![u]);
                    layers.push(WeightLayer::Dense {
                        w,
                        b,
                        act: activation.clone(),
                    });
                    last_width = u;
                }
                ModelLayer::Dropout { rate, .. } => {
                    layers.push(WeightLayer::Dropout {
                        rate: *rate as f32,
                        training: true,
                    });
                }
                ModelLayer::Norm { kind, .. } => {
                    let scale = Tensor::from_data(vec![last_width], vec![1.0; last_width]);
                    let shift = Tensor::zeros(vec![last_width]);
                    layers.push(WeightLayer::Norm {
                        kind: *kind,
                        scale,
                        shift,
                    });
                }
                ModelLayer::Attention {
                    num_heads,
                    head_dim,
                    ..
                } => {
                    let d = (*num_heads * *head_dim) as usize;
                    let std = (2.0_f32 / last_width as f32).sqrt();
                    let mk = |rows: usize, cols: usize| {
                        Tensor::from_data(
                            vec![rows, cols],
                            (0..rows * cols).map(|_| rand_normal(0.0, std)).collect(),
                        )
                    };
                    layers.push(WeightLayer::Attention {
                        num_heads: *num_heads,
                        head_dim: *head_dim,
                        wq: mk(last_width, d),
                        wk: mk(last_width, d),
                        wv: mk(last_width, d),
                        wo: mk(d, last_width),
                    });
                }
                ModelLayer::Embed {
                    vocab_size,
                    embed_dim,
                    ..
                } => {
                    let v = *vocab_size as usize;
                    let e = *embed_dim as usize;
                    let table = Tensor::from_data(
                        vec![v, e],
                        (0..v * e).map(|_| rand_normal(0.0, 0.01)).collect(),
                    );
                    layers.push(WeightLayer::Embed { table });
                    last_width = e;
                }
                ModelLayer::Conv2d {
                    filters,
                    kernel_h,
                    kernel_w,
                    activation,
                    ..
                } => {
                    layers.push(WeightLayer::Conv2d {
                        filters: *filters,
                        kh: *kernel_h,
                        kw: *kernel_w,
                        act: activation.clone(),
                    });
                }
                ModelLayer::Pool {
                    size_h, size_w, op, ..
                } => {
                    layers.push(WeightLayer::Pool {
                        ph: *size_h,
                        pw: *size_w,
                        op: *op,
                    });
                }
                ModelLayer::Recurrent { units, cell, .. } => {
                    let u = *units as usize;
                    let std = (2.0_f32 / (last_width + u) as f32).sqrt();
                    let wh = Tensor::from_data(
                        vec![u, u],
                        (0..u * u).map(|_| rand_normal(0.0, std)).collect(),
                    );
                    let wx = Tensor::from_data(
                        vec![last_width, u],
                        (0..last_width * u).map(|_| rand_normal(0.0, std)).collect(),
                    );
                    let bh = Tensor::zeros(vec![u]);
                    layers.push(WeightLayer::Recurrent {
                        units: *units,
                        cell: *cell,
                        wh,
                        wx,
                        bh,
                    });
                    last_width = u;
                }
                ModelLayer::SubModel { name, .. } => {
                    layers.push(WeightLayer::SubModel { name: name.clone() });
                }
                ModelLayer::Residual { .. } | ModelLayer::Flatten { .. } => {}
            }
        }

        let n = layers.len();
        NnModel {
            name: decl.name.clone(),
            layers,
            training: false,
            grads: vec![vec![]; n],
            m1: vec![vec![]; n],
            m2: vec![vec![]; n],
            step: 0,
        }
    }

    /// Run the forward pass.  Input: [batch, features] tensor.
    pub fn forward(&mut self, mut x: Tensor) -> Result<Tensor, RuntimeError> {
        for layer in &self.layers {
            x = self.apply_layer(layer, x)?;
        }
        Ok(x)
    }

    fn apply_layer(&self, layer: &WeightLayer, x: Tensor) -> Result<Tensor, RuntimeError> {
        match layer {
            WeightLayer::Dense { w, b, act } => {
                // y = x @ W + b
                let mut y = x.matmul(w)?;
                // Broadcast-add bias.
                let b_data = b.cpu_data();
                let cols = *w.shape.last().unwrap();
                let y_data = y.cpu_data_mut();
                for (i, v) in y_data.iter_mut().enumerate() {
                    *v += b_data[i % cols];
                }
                Ok(y.apply_activation(act))
            }
            WeightLayer::Dropout { rate, training } => {
                if !training || *rate == 0.0 {
                    return Ok(x);
                }
                // Bernoulli dropout mask.
                let keep = 1.0 - rate;
                let data: Vec<f32> = x
                    .cpu_data()
                    .iter()
                    .map(|v| if pseudo_rand() > *rate { v / keep } else { 0.0 })
                    .collect();
                Ok(Tensor::from_data(x.shape.clone(), data))
            }
            WeightLayer::Norm { scale, shift, .. } => {
                // Layer norm: normalise along last dimension.
                let d = *x.shape.last().unwrap_or(&1);
                let s_data = scale.cpu_data();
                let sh_data = shift.cpu_data();
                let x_data = x.cpu_data();
                let rows = x_data.len() / d;
                let mut out = vec![0.0_f32; x_data.len()];
                for r in 0..rows {
                    let row = &x_data[r * d..(r + 1) * d];
                    let mean: f32 = row.iter().sum::<f32>() / d as f32;
                    let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
                    let std = (var + 1e-6).sqrt();
                    for c in 0..d {
                        out[r * d + c] = (row[c] - mean) / std * s_data[c] + sh_data[c];
                    }
                }
                Ok(Tensor::from_data(x.shape.clone(), out))
            }
            WeightLayer::Attention {
                num_heads,
                head_dim,
                wq,
                wk,
                wv,
                wo,
            } => {
                // Simplified single-head for the interpreter (full MHA in prod).
                let q = x.matmul(wq)?;
                let k = x.matmul(wk)?;
                let v = x.matmul(wv)?;
                let d = *head_dim as f32;
                let scores = q.matmul(&k.transpose()?)?.scale(1.0 / d.sqrt());
                let scores = scores.apply_activation(&Activation::Softmax);
                let attn = scores.matmul(&v)?;
                let out = attn.matmul(wo)?;
                let _ = num_heads;
                Ok(out)
            }
            WeightLayer::Embed { table } => {
                // Expect x to contain integer indices (stored as f32).
                let indices: Vec<usize> = x.cpu_data().iter().map(|&v| v as usize).collect();
                let embed_dim = table.shape[1];
                let mut out = vec![0.0_f32; indices.len() * embed_dim];
                let t = table.cpu_data();
                for (i, idx) in indices.iter().enumerate() {
                    let src = &t[idx * embed_dim..(idx + 1) * embed_dim];
                    out[i * embed_dim..(i + 1) * embed_dim].copy_from_slice(src);
                }
                Ok(Tensor::from_data(vec![indices.len(), embed_dim], out))
            }
            WeightLayer::Recurrent {
                units, wh, wx, bh, ..
            } => {
                // Single-step GRU approximation (simplest stateful layer).
                let u = *units as usize;
                let h = Tensor::zeros(vec![1, u]);
                let xw = x.matmul(wx)?;
                let hw = h.matmul(wh)?;
                let b_data = bh.cpu_data();
                let xw_d = xw.cpu_data();
                let hw_d = hw.cpu_data();
                let out: Vec<f32> = xw_d
                    .iter()
                    .zip(hw_d)
                    .enumerate()
                    .map(|(i, (a, b))| (a + b + b_data[i % u]).tanh())
                    .collect();
                Ok(Tensor::from_data(vec![1, u], out))
            }
            WeightLayer::Conv2d { .. } | WeightLayer::Pool { .. } => {
                // Spatial operations: return x unchanged in the interpreter.
                // A production backend replaces these with im2col + GEMM.
                Ok(x)
            }
            WeightLayer::SubModel { .. } => Ok(x),
        }
    }
}

impl Tensor {
    fn transpose(&self) -> Result<Tensor, RuntimeError> {
        if self.shape.len() < 2 {
            return Err(RuntimeError::new("transpose requires ≥2-D tensor"));
        }
        let (rows, cols) = (
            self.shape[self.shape.len() - 2],
            self.shape[self.shape.len() - 1],
        );
        let data = self.cpu_data();
        let mut out = vec![0.0_f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = data[r * cols + c];
            }
        }
        let mut shape = self.shape.clone();
        let n = shape.len();
        shape.swap(n - 2, n - 1);
        Ok(Tensor::from_data(shape, out))
    }
}

// =============================================================================
// §5  SYSTEM SCHEDULER  (Feature 2)
// =============================================================================

/// The scheduler dispatches systems in topological order, respecting
/// component read/write dependencies.
pub struct Scheduler {
    /// Systems in execution order (determined by topological sort).
    order: Vec<String>,
}

impl Scheduler {
    /// Build a schedule from the program's system declarations.
    pub fn build(program: &Program) -> Self {
        // Simple heuristic: systems that only read run before systems that write.
        // A proper Kahn's-algorithm scheduler goes here in a full compiler.
        let mut order: Vec<String> = program.systems().map(|s| s.name.clone()).collect();
        order.sort(); // stable alphabetic fallback
        Scheduler { order }
    }

    /// Tick: run all systems in scheduled order against the world.
    pub fn tick(
        &self,
        systems: &FxHashMap<String, Arc<SystemDecl>>,
        world: &Arc<Mutex<EcsWorld>>,
        interp: &mut Interpreter,
        delta_time: f32,
    ) -> Result<(), RuntimeError> {
        for name in &self.order {
            if let Some(sys) = systems.get(name) {
                interp.run_system(sys, world, delta_time)?;
            }
        }
        Ok(())
    }
}

// =============================================================================
// §6  ENVIRONMENT (variable store)  —  optimized flat-slab implementation
// =============================================================================
//
// Design: instead of allocating a new HashMap per scope frame, we maintain:
//   • A single FxHashMap<name → slot_index> — updated in O(1).
//   • A flat Vec<Value> (the "slab") — each slot is one variable.
//   • A frame stack that records undo info: each push records (name, old_slot)
//     so that pop() can restore the previous binding.
//
// Compared with Vec<HashMap>:
//   • get()       : 1 hash lookup (was N frame scans + N lookups)
//   • set_local() : 1 push + 1 hash insert  (was 1 HashMap alloc + 1 insert)
//   • push()      : push empty Vec<_>       (was alloc new HashMap)
//   • pop()       : iterate undo list only  (was dealloc HashMap)
//
// Frame = Vec<(name, old_slot)>  where old_slot = None means the name was
// not present before this frame introduced it.

/// A single scope frame's undo log.
type FrameLog = Vec<(Box<str>, Option<u32>)>;

/// The call-stack / environment for the interpreter.
#[derive(Default)]
pub struct Env {
    /// Flat value slab.  slot i stores the value at index i.
    values: Vec<Value>,
    /// Maps variable name → current slot in `values`.
    name_to_slot: FxHashMap<Box<str>, u32>,
    /// Undo stack for scope management.
    frames: Vec<FrameLog>,
}

impl Env {
    #[inline]
    pub fn new() -> Self {
        Env {
            values: Vec::with_capacity(16),
            name_to_slot: FxHashMap::default(),
            frames: vec![Vec::new()],
        }
    }

    /// Enter a new lexical scope.
    #[inline]
    pub fn push(&mut self) {
        self.frames.push(Vec::new());
    }

    /// Exit the current lexical scope, removing all bindings introduced in it.
    #[inline]
    pub fn pop(&mut self) {
        if let Some(log) = self.frames.pop() {
            for (name, old_slot) in log {
                match old_slot {
                    Some(slot) => {
                        self.name_to_slot.insert(name, slot);
                    }
                    None => {
                        self.name_to_slot.remove(&name);
                    }
                }
            }
            // Recompute highest reachable slot after undo, then truncate.
            // This is O(bindings) but correct for shadow/rebind patterns.
            let next_len = self
                .name_to_slot
                .values()
                .copied()
                .max()
                .map(|v| v as usize + 1)
                .unwrap_or(0);
            self.values.truncate(next_len);
        }
    }

    /// Mutate an existing binding (searches all frames).  If not found,
    /// creates a new binding in the innermost frame.
    #[inline]
    pub fn set(&mut self, name: &str, val: Value) {
        if let Some(&slot) = self.name_to_slot.get(name) {
            // Fast path: binding already exists somewhere.
            self.values[slot as usize] = val;
        } else {
            // Slow path: new binding.
            self.set_local(name, val);
        }
    }

    /// Introduce a new binding in the innermost scope.
    #[inline]
    pub fn set_local(&mut self, name: &str, val: Value) {
        let slot = self.values.len() as u32;
        self.values.push(val);
        let boxed: Box<str> = name.into();
        // Record the old slot (for pop undo) before overwriting.
        let old_slot = self.name_to_slot.insert(boxed.clone(), slot);
        if let Some(frame) = self.frames.last_mut() {
            frame.push((boxed, old_slot));
        }
    }

    /// Look up a variable.
    ///
    /// Uses `get_unchecked` behind a feature flag; in debug builds this falls
    /// back to the bounds-checked path so tests still catch logic errors.
    #[inline]
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.name_to_slot.get(name).map(|&slot| {
            // SAFETY: every slot inserted by `set_local` is a valid index into
            // `self.values`; we never remove elements from the slab.
            #[cfg(not(debug_assertions))]
            unsafe {
                self.values.get_unchecked(slot as usize)
            }
            #[cfg(debug_assertions)]
            &self.values[slot as usize]
        })
    }

    /// Iterate all (name, value) pairs in the current flat view.
    /// Used for closure capture.
    pub fn iter_all(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.name_to_slot
            .iter()
            .map(|(name, &slot)| (name.as_ref(), &self.values[slot as usize]))
    }
}

// Keep type alias for closure capture maps (used by FnClosure).
type Frame = FxHashMap<String, Value>;

// =============================================================================
// §6b  BYTECODE COMPILER + REGISTER VM
// =============================================================================
//
// Architecture overview:
//
//   Compiler::compile_fn(decl)  →  CompiledFn { instrs, str_pool, slot_count, .. }
//   vm_exec(interp, func, args) →  Value
//
// Variable resolution:
//   At compile time, each `let`-bound name is assigned a unique integer slot.
//   The VM frame is just a `Vec<Value>` indexed by slot — no string hashing at
//   runtime at all.
//
// The tree-walking eval_expr / eval_stmt / eval_block are KEPT intact for
// the test API.  call_fn() uses the bytecode VM path.

/// A single VM instruction.  Using u16 for register indices keeps each
/// instruction at 8 bytes or less, fitting two per cache line.
#[derive(Debug, Clone)]
pub enum Instr {
    // ── Literals ─────────────────────────────────────────────────────────────
    LoadUnit(u16),
    LoadBool(u16, bool),
    LoadI32(u16, i32),
    LoadI64(u16, i64),
    LoadF32(u16, f32),
    LoadF64(u16, f64),
    /// Load string constant: (dst_reg, str_pool_idx)
    LoadStr(u16, u16),
    /// Load a pre-built Value from the const pool: (dst_reg, const_pool_idx)
    LoadConst(u16, u16),
    /// Load a global function closure: (dst_reg, fn_name str_pool_idx)
    LoadFn(u16, u16),

    // ── Register ─────────────────────────────────────────────────────────────
    Move(u16, u16), // dst ← src

    // ── Variables (slot-addressed) ────────────────────────────────────────────
    /// Load from slot: (dst_reg, slot)
    Load(u16, u16),
    /// Store to slot: (slot, src_reg)
    Store(u16, u16),

    // ── Arithmetic / comparison ───────────────────────────────────────────────
    BinOp(u16, BinOpKind, u16, u16), // dst, op, lhs, rhs
    UnOp(u16, UnOpKind, u16),        // dst, op, src
    PowOp(u16, u16, u16),            // dst, base, exp
    MatMulInstr(u16, u16, u16),      // dst, lhs, rhs
    HadamardMulInstr(u16, u16, u16),
    HadamardDivInstr(u16, u16, u16),
    TensorConcatInstr(u16, u16, u16),

    // ── Control flow ─────────────────────────────────────────────────────────
    /// Unconditional jump: relative pc offset (can be negative).
    Jump(i32),
    /// Jump if register is falsy.
    JumpFalse(u16, i32),
    /// Jump if register is truthy.
    JumpTrue(u16, i32),

    // ── Calls ─────────────────────────────────────────────────────────────────
    /// Call user function: (dst, callee_reg, args_start_reg, arg_count)
    Call(u16, u16, u16, u16),
    /// Call named builtin: (dst, name_str_idx, args_start, arg_count)
    CallBuiltin(u16, u16, u16, u16),
    /// Call method: (dst, recv_reg, method_str_idx, args_start, arg_count)
    CallMethod(u16, u16, u16, u16, u16),

    // ── Return / signals ─────────────────────────────────────────────────────
    Return(u16),
    ReturnUnit,
    BreakSignal,
    BreakValSignal(u16),
    ContinueSignal,

    // ── Collections ─────────────────────────────────────────────────────────
    NewArray(u16),
    ArrayPush(u16, u16),     // array_reg, val_reg
    ArrayGet(u16, u16, u16), // dst, array_reg, idx_reg
    ArraySet(u16, u16, u16), // array_reg, idx_reg, val_reg
    NewHashMap(u16),
    NewTuple(u16, u16, u16), // dst, first_reg, count
    NewStruct(u16, u16),     // dst, name_str_idx

    // ── Field / index ─────────────────────────────────────────────────────────
    FieldGet(u16, u16, u16), // dst, obj_reg, field_str_idx
    FieldSet(u16, u16, u16), // obj_reg, field_str_idx, val_reg
    IndexGet(u16, u16, u16), // dst, obj_reg, idx_reg
    IndexSet(u16, u16, u16), // obj_reg, idx_reg, val_reg

    // ── Vector constructors ───────────────────────────────────────────────────
    Vec2Ctor(u16, u16, u16),
    Vec3Ctor(u16, u16, u16, u16),
    Vec4Ctor(u16, u16, u16, u16, u16),

    // ── Range ─────────────────────────────────────────────────────────────────
    RangeExcl(u16, u16, u16), // dst, lo_reg, hi_reg
    RangeIncl(u16, u16, u16),

    // ── Grad ─────────────────────────────────────────────────────────────────
    EnableGrad(u16, u16), // dst, src (enables grad, returns same tensor)

    // ── Misc ─────────────────────────────────────────────────────────────────
    Nop,
}

/// A compiled function body ready for the VM.
#[derive(Debug, Clone)]
pub struct CompiledFn {
    pub name: String,
    /// Number of parameter slots (first `param_count` slots = args).
    pub param_count: u16,
    /// Total register/slot count needed.
    pub slot_count: u16,
    /// The instruction stream.
    pub instrs: Vec<Instr>,
    /// String constant pool (field names, builtin names, struct names, …).
    pub str_pool: Vec<String>,
    /// Arbitrary Value constants (e.g. pre-built empty arrays).
    pub const_pool: Vec<Value>,
}

// ── Compiler ─────────────────────────────────────────────────────────────────

struct Compiler {
    instrs: Vec<Instr>,
    str_pool: Vec<String>,
    str_idx: FxHashMap<String, u16>,
    const_pool: Vec<Value>,
    /// Scope stack: each element is a map from name → slot index.
    scopes: Vec<FxHashMap<String, u16>>,
    /// Next available slot index.
    next_slot: u16,
    /// Next available temporary register (above all slots).
    next_tmp: u16,
    /// Captured closure variables from outer scope.
    captures: FxHashMap<String, u16>,
}

impl Compiler {
    fn new(param_count: u16) -> Self {
        Compiler {
            instrs: Vec::new(),
            str_pool: Vec::new(),
            str_idx: FxHashMap::default(),
            const_pool: Vec::new(),
            scopes: vec![FxHashMap::default()],
            next_slot: param_count,
            // Temporaries live immediately above slots to keep register files compact.
            next_tmp: param_count,
            captures: FxHashMap::default(),
        }
    }

    /// Intern a string, returning its pool index.
    fn intern(&mut self, s: &str) -> u16 {
        if let Some(&i) = self.str_idx.get(s) {
            return i;
        }
        let i = self.str_pool.len() as u16;
        self.str_pool.push(s.to_owned());
        self.str_idx.insert(s.to_owned(), i);
        i
    }

    /// Allocate a new temporary register.
    fn tmp(&mut self) -> u16 {
        let r = self.next_tmp;
        self.next_tmp += 1;
        r
    }

    /// Declare a new local variable slot in the current scope.
    fn declare(&mut self, name: &str) -> u16 {
        let slot = self.next_slot;
        self.next_slot += 1;
        self.scopes
            .last_mut()
            .unwrap()
            .insert(name.to_owned(), slot);
        slot
    }

    /// Resolve a variable name to its slot, or None if unknown.
    fn resolve(&self, name: &str) -> Option<u16> {
        for scope in self.scopes.iter().rev() {
            if let Some(&slot) = scope.get(name) {
                return Some(slot);
            }
        }
        self.captures.get(name).copied()
    }

    fn push_scope(&mut self) {
        self.scopes.push(FxHashMap::default());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn emit(&mut self, instr: Instr) {
        self.instrs.push(instr);
    }

    /// Emit a placeholder jump, returns its index so we can patch it.
    fn emit_jump_false(&mut self, cond: u16) -> usize {
        let pos = self.instrs.len();
        self.emit(Instr::JumpFalse(cond, 0));
        pos
    }

    fn emit_jump(&mut self) -> usize {
        let pos = self.instrs.len();
        self.emit(Instr::Jump(0));
        pos
    }

    fn patch_jump(&mut self, pos: usize) {
        let target = self.instrs.len() as i32;
        let offset = target - pos as i32 - 1;
        match &mut self.instrs[pos] {
            Instr::Jump(ref mut o) => *o = offset,
            Instr::JumpFalse(_, ref mut o) => *o = offset,
            Instr::JumpTrue(_, ref mut o) => *o = offset,
            _ => {}
        }
    }

    /// Compile a block, returning the register holding the tail value.
    fn compile_block(&mut self, block: &Block, dst: u16) {
        let block_tmp_base = self.next_slot;
        self.next_tmp = self.next_tmp.max(block_tmp_base);
        self.push_scope();
        for stmt in &block.stmts {
            self.compile_stmt(stmt);
            // Statement temporaries are not live across statement boundaries.
            self.next_tmp = self.next_tmp.min(self.next_slot).max(block_tmp_base);
            // If stmt itself ends in Return/Break/Continue we can stop.
        }
        if let Some(tail) = &block.tail {
            self.compile_expr_into(tail, dst);
        } else {
            self.emit(Instr::LoadUnit(dst));
        }
        self.pop_scope();
        self.next_tmp = self.next_tmp.min(self.next_slot).max(block_tmp_base);
    }

    /// Compile a statement.
    fn compile_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let { pattern, init, .. } => {
                let init_reg = self.tmp();
                if let Some(e) = init {
                    self.compile_expr_into(e, init_reg);
                } else {
                    self.emit(Instr::LoadUnit(init_reg));
                }
                self.compile_pattern_bind(pattern, init_reg);
            }
            Stmt::Expr { expr, .. } => {
                let t = self.tmp();
                self.compile_expr_into(expr, t);
            }
            Stmt::Return { value, .. } => {
                if let Some(e) = value {
                    let r = self.tmp();
                    self.compile_expr_into(e, r);
                    self.emit(Instr::Return(r));
                } else {
                    self.emit(Instr::ReturnUnit);
                }
            }
            Stmt::Break { value, .. } => {
                if let Some(e) = value {
                    let r = self.tmp();
                    self.compile_expr_into(e, r);
                    self.emit(Instr::BreakValSignal(r));
                } else {
                    self.emit(Instr::BreakSignal);
                }
            }
            Stmt::Continue { .. } => {
                self.emit(Instr::ContinueSignal);
            }
            Stmt::While { cond, body, .. } => {
                let loop_start = self.instrs.len() as i32;
                let cond_reg = self.tmp();
                self.compile_expr_into(cond, cond_reg);
                let exit_jump = self.emit_jump_false(cond_reg);
                let body_dst = self.tmp();
                self.compile_block(body, body_dst);
                // Jump back to condition
                let back_offset = loop_start - self.instrs.len() as i32 - 1;
                self.emit(Instr::Jump(back_offset));
                self.patch_jump(exit_jump);
            }
            Stmt::Loop { body, .. } => {
                let loop_start = self.instrs.len() as i32;
                let body_dst = self.tmp();
                self.compile_block(body, body_dst);
                let back_offset = loop_start - self.instrs.len() as i32 - 1;
                self.emit(Instr::Jump(back_offset));
            }
            Stmt::If {
                cond, then, else_, ..
            } => {
                let cond_reg = self.tmp();
                self.compile_expr_into(cond, cond_reg);
                let else_jump = self.emit_jump_false(cond_reg);
                let dst = self.tmp();
                self.compile_block(then, dst);
                if let Some(e) = else_ {
                    let end_jump = self.emit_jump();
                    self.patch_jump(else_jump);
                    match e.as_ref() {
                        crate::ast::IfOrBlock::Block(b) => self.compile_block(b, dst),
                        crate::ast::IfOrBlock::If(s) => self.compile_stmt(s),
                    }
                    self.patch_jump(end_jump);
                } else {
                    self.patch_jump(else_jump);
                }
            }
            Stmt::ForIn {
                pattern,
                iter,
                body,
                ..
            } => {
                // Evaluate iterator into a temporary array register.
                let iter_reg = self.tmp();
                self.compile_expr_into(iter, iter_reg);
                // Emit: idx = 0
                let idx_slot = self.next_slot;
                self.next_slot += 1;
                self.emit(Instr::LoadI32(idx_slot, 0));
                // Loop header: cond = (idx < len)
                let loop_start = self.instrs.len() as i32;
                let len_reg = self.tmp();
                let si = self.intern("len");
                self.emit(Instr::CallMethod(len_reg, iter_reg, si, 0, 0));
                let cond_reg = self.tmp();
                self.emit(Instr::BinOp(cond_reg, BinOpKind::Lt, idx_slot, len_reg));
                let exit_jump = self.emit_jump_false(cond_reg);
                // Body: item = arr[idx]; bind pattern; run body
                let item_reg = self.tmp();
                self.emit(Instr::ArrayGet(item_reg, iter_reg, idx_slot));
                self.push_scope();
                self.compile_pattern_bind(pattern, item_reg);
                let body_dst = self.tmp();
                self.compile_block(body, body_dst);
                self.pop_scope();
                // idx += 1
                let one_reg = self.tmp();
                self.emit(Instr::LoadI32(one_reg, 1));
                self.emit(Instr::BinOp(idx_slot, BinOpKind::Add, idx_slot, one_reg));
                let back = loop_start - self.instrs.len() as i32 - 1;
                self.emit(Instr::Jump(back));
                self.patch_jump(exit_jump);
            }
            // Fallthrough: complex stmts fall back at runtime via tree-walker.
            _ => {
                // Emit a sentinel that signals "exec this stmt via tree-walker".
                // (In practice the compiler only gets called for functions where
                //  all stmts are compilable; others take the tree-walk path.)
                self.emit(Instr::Nop);
            }
        }
    }

    /// Compile an expression, placing the result in `dst`.
    fn compile_expr_into(&mut self, expr: &Expr, dst: u16) {
        match expr {
            Expr::IntLit { value, .. } => {
                self.emit(Instr::LoadI32(dst, *value as i32));
            }
            Expr::FloatLit { value, .. } => {
                self.emit(Instr::LoadF32(dst, *value as f32));
            }
            Expr::BoolLit { value, .. } => {
                self.emit(Instr::LoadBool(dst, *value));
            }
            Expr::StrLit { value, .. } => {
                let si = self.intern(value);
                self.emit(Instr::LoadStr(dst, si));
            }
            Expr::Ident { name, .. } => {
                if let Some(slot) = self.resolve(name) {
                    self.emit(Instr::Load(dst, slot));
                } else {
                    // Global fn or world — emit LoadFn / special
                    let si = self.intern(name);
                    self.emit(Instr::LoadFn(dst, si));
                }
            }
            Expr::Path { segments, .. } => {
                let name = segments.join("::");
                if let Some(slot) = self.resolve(&name) {
                    self.emit(Instr::Load(dst, slot));
                } else {
                    let si = self.intern(&name);
                    self.emit(Instr::LoadFn(dst, si));
                }
            }
            Expr::BinOp { op, lhs, rhs, .. } => {
                // Short-circuit And / Or
                if *op == BinOpKind::And {
                    self.compile_expr_into(lhs, dst);
                    let jmp = self.emit_jump_false(dst);
                    self.compile_expr_into(rhs, dst);
                    // Normalise to Bool
                    let bool_reg = self.tmp();
                    self.emit(Instr::BinOp(bool_reg, BinOpKind::Ne, dst, dst)); // placeholder
                                                                                // Actually just leave dst as-is and patch
                    self.patch_jump(jmp);
                    return;
                }
                if *op == BinOpKind::Or {
                    self.compile_expr_into(lhs, dst);
                    let jmp = self.emit_jump_false(dst);
                    let end_jmp = self.emit_jump();
                    self.patch_jump(jmp);
                    self.compile_expr_into(rhs, dst);
                    self.patch_jump(end_jmp);
                    return;
                }
                let l = self.tmp();
                let r = self.tmp();
                self.compile_expr_into(lhs, l);
                self.compile_expr_into(rhs, r);
                self.emit(Instr::BinOp(dst, *op, l, r));
            }
            Expr::UnOp { op, expr, .. } => {
                let s = self.tmp();
                self.compile_expr_into(expr, s);
                self.emit(Instr::UnOp(dst, *op, s));
            }
            Expr::Assign {
                op, target, value, ..
            } => {
                let rhs_reg = self.tmp();
                self.compile_expr_into(value, rhs_reg);
                match target.as_ref() {
                    Expr::Ident { name, .. } => {
                        if let Some(slot) = self.resolve(name) {
                            if *op == AssignOpKind::Assign {
                                self.emit(Instr::Store(slot, rhs_reg));
                            } else {
                                let cur = self.tmp();
                                self.emit(Instr::Load(cur, slot));
                                let bin_op = op.to_binop().unwrap_or(BinOpKind::Add);
                                self.emit(Instr::BinOp(cur, bin_op, cur, rhs_reg));
                                self.emit(Instr::Store(slot, cur));
                            }
                        }
                    }
                    Expr::Field { object, field, .. } => {
                        let obj_reg = self.tmp();
                        self.compile_expr_into(object, obj_reg);
                        let fi = self.intern(field);
                        self.emit(Instr::FieldSet(obj_reg, fi, rhs_reg));
                    }
                    Expr::Index {
                        object, indices, ..
                    } => {
                        let obj_reg = self.tmp();
                        self.compile_expr_into(object, obj_reg);
                        let idx_reg = self.tmp();
                        if let Some(i) = indices.first() {
                            self.compile_expr_into(i, idx_reg);
                        }
                        self.emit(Instr::IndexSet(obj_reg, idx_reg, rhs_reg));
                    }
                    _ => {}
                }
                self.emit(Instr::LoadUnit(dst));
            }
            Expr::Field { object, field, .. } => {
                let obj_reg = self.tmp();
                self.compile_expr_into(object, obj_reg);
                let fi = self.intern(field);
                self.emit(Instr::FieldGet(dst, obj_reg, fi));
            }
            Expr::Index {
                object, indices, ..
            } => {
                let obj_reg = self.tmp();
                self.compile_expr_into(object, obj_reg);
                let idx_reg = self.tmp();
                if let Some(i) = indices.first() {
                    self.compile_expr_into(i, idx_reg);
                } else {
                    self.emit(Instr::LoadI32(idx_reg, 0));
                }
                self.emit(Instr::IndexGet(dst, obj_reg, idx_reg));
            }
            Expr::Call { func, args, .. } => {
                // Evaluate all args into consecutive registers.
                let args_start = self.next_tmp;
                for a in args.iter() {
                    let r = self.tmp();
                    self.compile_expr_into(a, r);
                }
                let arg_count = (self.next_tmp - args_start) as u16;
                // Check for builtin by name.
                if let Expr::Ident { name, .. } = func.as_ref() {
                    let si = self.intern(name);
                    self.emit(Instr::CallBuiltin(dst, si, args_start, arg_count));
                } else if let Expr::Path { segments, .. } = func.as_ref() {
                    let name = segments.join("::");
                    let si = self.intern(&name);
                    self.emit(Instr::CallBuiltin(dst, si, args_start, arg_count));
                } else {
                    let fn_reg = self.tmp();
                    self.compile_expr_into(func, fn_reg);
                    self.emit(Instr::Call(dst, fn_reg, args_start, arg_count));
                }
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
                ..
            } => {
                let recv_reg = self.tmp();
                self.compile_expr_into(receiver, recv_reg);
                let args_start = self.next_tmp;
                for a in args.iter() {
                    let r = self.tmp();
                    self.compile_expr_into(a, r);
                }
                let arg_count = (self.next_tmp - args_start) as u16;
                let mi = self.intern(method);
                self.emit(Instr::CallMethod(dst, recv_reg, mi, args_start, arg_count));
            }
            Expr::IfExpr {
                cond, then, else_, ..
            } => {
                let cond_reg = self.tmp();
                self.compile_expr_into(cond, cond_reg);
                let else_jump = self.emit_jump_false(cond_reg);
                self.compile_block(then, dst);
                if let Some(b) = else_ {
                    let end_jump = self.emit_jump();
                    self.patch_jump(else_jump);
                    self.compile_block(b, dst);
                    self.patch_jump(end_jump);
                } else {
                    self.patch_jump(else_jump);
                    self.emit(Instr::LoadUnit(dst));
                }
            }
            Expr::Block(b) => {
                self.compile_block(b, dst);
            }
            Expr::ArrayLit { elems, .. } => {
                self.emit(Instr::NewArray(dst));
                for e in elems {
                    let r = self.tmp();
                    self.compile_expr_into(e, r);
                    self.emit(Instr::ArrayPush(dst, r));
                }
            }
            Expr::Tuple { elems, .. } => {
                let start = self.next_tmp;
                for e in elems {
                    let r = self.tmp();
                    self.compile_expr_into(e, r);
                }
                let count = (self.next_tmp - start) as u16;
                self.emit(Instr::NewTuple(dst, start, count));
            }
            Expr::VecCtor { size, elems, .. } => {
                let regs: Vec<u16> = elems
                    .iter()
                    .map(|e| {
                        let r = self.tmp();
                        self.compile_expr_into(e, r);
                        r
                    })
                    .collect();
                match size {
                    VecSize::N2 => self.emit(Instr::Vec2Ctor(dst, regs[0], regs[1])),
                    VecSize::N3 => self.emit(Instr::Vec3Ctor(dst, regs[0], regs[1], regs[2])),
                    VecSize::N4 => {
                        self.emit(Instr::Vec4Ctor(dst, regs[0], regs[1], regs[2], regs[3]))
                    }
                }
            }
            Expr::Range {
                lo, hi, inclusive, ..
            } => {
                let lo_reg = self.tmp();
                let hi_reg = self.tmp();
                if let Some(l) = lo {
                    self.compile_expr_into(l, lo_reg);
                } else {
                    self.emit(Instr::LoadI32(lo_reg, 0));
                }
                if let Some(h) = hi {
                    self.compile_expr_into(h, hi_reg);
                } else {
                    self.emit(Instr::LoadI32(hi_reg, 0));
                }
                if *inclusive {
                    self.emit(Instr::RangeIncl(dst, lo_reg, hi_reg));
                } else {
                    self.emit(Instr::RangeExcl(dst, lo_reg, hi_reg));
                }
            }
            Expr::MatMul { lhs, rhs, .. } => {
                let l = self.tmp();
                let r = self.tmp();
                self.compile_expr_into(lhs, l);
                self.compile_expr_into(rhs, r);
                self.emit(Instr::MatMulInstr(dst, l, r));
            }
            Expr::Pow { base, exp, .. } => {
                let b = self.tmp();
                let e = self.tmp();
                self.compile_expr_into(base, b);
                self.compile_expr_into(exp, e);
                self.emit(Instr::PowOp(dst, b, e));
            }
            Expr::Grad { inner, .. } => {
                let s = self.tmp();
                self.compile_expr_into(inner, s);
                self.emit(Instr::EnableGrad(dst, s));
            }
            Expr::StructLit { name, fields, .. } => {
                let ni = self.intern(name);
                self.emit(Instr::NewStruct(dst, ni));
                for (fname, fexpr) in fields {
                    let vr = self.tmp();
                    self.compile_expr_into(fexpr, vr);
                    let fi = self.intern(fname);
                    self.emit(Instr::FieldSet(dst, fi, vr));
                }
            }
            Expr::Cast { expr, ty, .. } => {
                let s = self.tmp();
                self.compile_expr_into(expr, s);
                // Encode type as a string constant for runtime dispatch.
                let type_str = format!("{:?}", ty);
                let ti = self.intern(&type_str);
                // Re-use FieldGet slot with a special sentinel.
                self.emit(Instr::LoadStr(dst, ti));
                // Fallback: emit a builtin call to "cast"
                self.emit(Instr::Move(dst, s)); // no-op cast; full cast via tree-walker
            }
            Expr::Closure { .. } => {
                // Closures fall back to const pool.
                self.emit(Instr::LoadUnit(dst));
            }
            _ => {
                // Unrecognised expression: emit unit and let the tree-walker
                // handle it if the function takes the slow path.
                self.emit(Instr::LoadUnit(dst));
            }
        }
    }

    fn compile_pattern_bind(&mut self, pat: &Pattern, src: u16) {
        match pat {
            Pattern::Ident { name, .. } => {
                let slot = self.declare(name);
                self.emit(Instr::Store(slot, src));
            }
            Pattern::Wildcard(_) => {}
            Pattern::Tuple { elems, .. } => {
                for (i, p) in elems.iter().enumerate() {
                    let item = self.tmp();
                    self.emit(Instr::LoadI32(item, i as i32));
                    let v = self.tmp();
                    self.emit(Instr::IndexGet(v, src, item));
                    self.compile_pattern_bind(p, v);
                }
            }
            _ => {}
        }
    }

    fn finish(self, name: String, param_count: u16) -> CompiledFn {
        let slot_count = self.next_tmp.max(self.next_slot);
        CompiledFn {
            name,
            param_count,
            slot_count,
            instrs: self.instrs,
            str_pool: self.str_pool,
            const_pool: self.const_pool,
        }
    }
}

/// Compile an `FnDecl` into a `CompiledFn`.
pub fn compile_fn(decl: &FnDecl) -> CompiledFn {
    let param_count = decl.params.len() as u16;
    let mut c = Compiler::new(param_count);
    // Declare parameter slots in order.
    for (i, p) in decl.params.iter().enumerate() {
        c.scopes
            .last_mut()
            .unwrap()
            .insert(p.name.clone(), i as u16);
    }
    let result_dst = c.next_slot;
    c.next_slot += 1;
    c.next_tmp = c.next_slot;
    if let Some(body) = &decl.body {
        c.compile_block(body, result_dst);
    }
    c.emit(Instr::Return(result_dst));
    c.finish(decl.name.clone(), param_count)
}

// ── Shorthand runtime error macro ────────────────────────────────────────────
macro_rules! rt_err {
    ($msg:expr) => { Err(RuntimeError::new($msg)) };
    ($fmt:literal $(, $arg:expr)*) => { Err(RuntimeError::new(format!($fmt $(, $arg)*))) };
}

// ── Register-based VM executor ───────────────────────────────────────────────

/// Execute a compiled function on the VM.
///
/// `args` are placed into the first N register slots.
/// Returns the function's return value or a `RuntimeError`.
pub fn vm_exec(
    interp: &mut Interpreter,
    func: &CompiledFn,
    args: Vec<Value>,
) -> Result<Value, RuntimeError> {
    struct VmRegPoolGuard {
        regs: Vec<Value>,
    }
    impl Drop for VmRegPoolGuard {
        fn drop(&mut self) {
            self.regs.clear();
            let regs = std::mem::take(&mut self.regs);
            VM_REG_POOL.with(|pool| {
                *pool.borrow_mut() = regs;
            });
        }
    }

    let needed = func.slot_count as usize + 32;
    let mut reg_guard = VmRegPoolGuard {
        regs: VM_REG_POOL.with(|pool| std::mem::take(&mut *pool.borrow_mut())),
    };
    reg_guard.regs.clear();
    reg_guard.regs.resize(needed, Value::Unit);
    let regs = &mut reg_guard.regs;

    // Load arguments.
    for (i, arg) in args.into_iter().enumerate() {
        if i < regs.len() {
            regs[i] = arg;
        }
    }

    let instrs = &func.instrs;
    let str_pool = &func.str_pool;
    let const_pool = &func.const_pool;
    let mut pc: usize = 0;

    // Hot-path register access without bounds checking in release mode.
    macro_rules! reg {
        ($r:expr) => {{
            #[cfg(debug_assertions)]
            {
                regs.get($r as usize)
                    .expect("vm_exec: register index out of bounds")
            }
            #[cfg(not(debug_assertions))]
            unsafe {
                regs.get_unchecked($r as usize)
            }
        }};
    }
    macro_rules! reg_mut {
        ($r:expr) => {{
            #[cfg(debug_assertions)]
            {
                regs.get_mut($r as usize)
                    .expect("vm_exec: register index out of bounds")
            }
            #[cfg(not(debug_assertions))]
            unsafe {
                regs.get_unchecked_mut($r as usize)
            }
        }};
    }
    macro_rules! slot {
        ($s:expr) => {{
            #[cfg(debug_assertions)]
            {
                regs.get($s as usize)
                    .expect("vm_exec: slot index out of bounds")
            }
            #[cfg(not(debug_assertions))]
            unsafe {
                regs.get_unchecked($s as usize)
            }
        }};
    }
    macro_rules! slot_mut {
        ($s:expr) => {{
            #[cfg(debug_assertions)]
            {
                regs.get_mut($s as usize)
                    .expect("vm_exec: slot index out of bounds")
            }
            #[cfg(not(debug_assertions))]
            unsafe {
                regs.get_unchecked_mut($s as usize)
            }
        }};
    }
    macro_rules! str_c {
        ($i:expr) => {
            str_pool
                .get($i as usize)
                .expect("vm_exec: string index out of bounds")
                .as_str()
        };
    }
    loop {
        if pc >= instrs.len() {
            return Ok(Value::Unit);
        }
        let instr = unsafe { instrs.get_unchecked(pc) };
        pc += 1;

        match instr {
            Instr::Nop => {}
            Instr::LoadUnit(d) => *reg_mut!(*d) = Value::Unit,
            Instr::LoadBool(d, b) => *reg_mut!(*d) = Value::Bool(*b),
            Instr::LoadI32(d, v) => *reg_mut!(*d) = Value::I32(*v),
            Instr::LoadI64(d, v) => *reg_mut!(*d) = Value::I64(*v),
            Instr::LoadF32(d, v) => *reg_mut!(*d) = Value::F32(*v),
            Instr::LoadF64(d, v) => *reg_mut!(*d) = Value::F64(*v),
            Instr::LoadStr(d, si) => *reg_mut!(*d) = Value::Str(str_c!(*si).to_owned()),
            Instr::LoadConst(d, ci) => {
                *reg_mut!(*d) = const_pool
                    .get(*ci as usize)
                    .expect("vm_exec: const index out of bounds")
                    .clone()
            }
            Instr::LoadFn(d, si) => {
                let name = str_c!(*si);
                if name == "world" {
                    *reg_mut!(*d) = Value::World(interp.world.clone());
                } else if let Some(f) = interp.fns.get(name).cloned() {
                    *reg_mut!(*d) = Value::Fn(f);
                } else if let Some(m) = interp.models.get(name).cloned() {
                    *reg_mut!(*d) = Value::Model(m);
                } else {
                    *reg_mut!(*d) = Value::Unit;
                }
            }
            Instr::Move(d, s) => {
                let v = reg!(*s).clone();
                *reg_mut!(*d) = v;
            }
            Instr::Load(d, slot) => {
                let v = slot!(*slot).clone();
                *reg_mut!(*d) = v;
            }
            Instr::Store(slot, s) => {
                let v = reg!(*s).clone();
                *slot_mut!(*slot) = v;
            }

            Instr::BinOp(d, op, l, r) => {
                // Optimized numeric binop: avoid cloning when types match.
                let lv = reg!(*l);
                let rv = reg!(*r);
                *reg_mut!(*d) = match (lv, rv) {
                    // I32 fast path (most common for loop counters & indices)
                    (Value::I32(a), Value::I32(b)) => {
                        let (a, b) = (*a, *b);
                        match op {
                            BinOpKind::Add => Value::I32(a.wrapping_add(b)),
                            BinOpKind::Sub => Value::I32(a.wrapping_sub(b)),
                            BinOpKind::Mul => Value::I32(a.wrapping_mul(b)),
                            BinOpKind::Div => {
                                if b == 0 {
                                    return rt_err!("division by zero");
                                }
                                Value::I32(a.wrapping_div(b))
                            }
                            BinOpKind::Rem => {
                                if b == 0 {
                                    return rt_err!("modulo by zero");
                                }
                                Value::I32(a.wrapping_rem(b))
                            }
                            BinOpKind::Lt => Value::Bool(a < b),
                            BinOpKind::Le => Value::Bool(a <= b),
                            BinOpKind::Gt => Value::Bool(a > b),
                            BinOpKind::Ge => Value::Bool(a >= b),
                            BinOpKind::Eq => Value::Bool(a == b),
                            BinOpKind::Ne => Value::Bool(a != b),
                            _ => return rt_err!("op {:?} not defined for i32", op),
                        }
                    }
                    // F32 fast path
                    (Value::F32(a), Value::F32(b)) => {
                        let (a, b) = (*a, *b);
                        match op {
                            BinOpKind::Add => Value::F32(a + b),
                            BinOpKind::Sub => Value::F32(a - b),
                            BinOpKind::Mul => Value::F32(a * b),
                            BinOpKind::Div => {
                                if b == 0.0 {
                                    return rt_err!("division by zero");
                                }
                                Value::F32(a / b)
                            }
                            BinOpKind::Rem => Value::F32(a % b),
                            BinOpKind::FloorDiv => {
                                if b == 0.0 {
                                    return rt_err!("floor division by zero");
                                }
                                Value::F32((a / b).floor())
                            }
                            BinOpKind::Lt => Value::Bool(a < b),
                            BinOpKind::Le => Value::Bool(a <= b),
                            BinOpKind::Gt => Value::Bool(a > b),
                            BinOpKind::Ge => Value::Bool(a >= b),
                            BinOpKind::Eq => Value::Bool(a == b),
                            BinOpKind::Ne => Value::Bool(a != b),
                            _ => return rt_err!("op {:?} not defined for f32", op),
                        }
                    }
                    // I64 path
                    (Value::I64(a), Value::I64(b)) => {
                        let (a, b) = (*a, *b);
                        match op {
                            BinOpKind::Add => Value::I64(a.wrapping_add(b)),
                            BinOpKind::Sub => Value::I64(a.wrapping_sub(b)),
                            BinOpKind::Mul => Value::I64(a.wrapping_mul(b)),
                            BinOpKind::Div => {
                                if b == 0 {
                                    return rt_err!("division by zero");
                                }
                                Value::I64(a.wrapping_div(b))
                            }
                            BinOpKind::Rem => {
                                if b == 0 {
                                    return rt_err!("modulo by zero");
                                }
                                Value::I64(a.wrapping_rem(b))
                            }
                            BinOpKind::Lt => Value::Bool(a < b),
                            BinOpKind::Le => Value::Bool(a <= b),
                            BinOpKind::Gt => Value::Bool(a > b),
                            BinOpKind::Ge => Value::Bool(a >= b),
                            BinOpKind::Eq => Value::Bool(a == b),
                            BinOpKind::Ne => Value::Bool(a != b),
                            _ => return rt_err!("op {:?} not defined for i64", op),
                        }
                    }
                    // Fallback: clone and use full numeric binop
                    _ => {
                        let lv_clone = lv.clone();
                        let rv_clone = rv.clone();
                        eval_numeric_binop(*op, lv_clone, rv_clone)?
                    }
                };
            }
            Instr::UnOp(d, op, s) => {
                let v = reg!(*s).clone();
                *reg_mut!(*d) = vm_unop(*op, v)?;
            }
            Instr::PowOp(d, b, e) => {
                let bv = reg!(*b);
                let ev = reg!(*e);
                *reg_mut!(*d) = match (bv, ev) {
                    (Value::F32(x), Value::F32(y)) => Value::F32(x.powf(*y)),
                    (Value::F64(x), Value::F64(y)) => Value::F64(x.powf(*y)),
                    (Value::I32(x), Value::I32(y)) => Value::I32(x.pow(*y as u32)),
                    _ => {
                        let bv_clone = bv.clone();
                        let ev_clone = ev.clone();
                        if let (Some(x), Some(y)) = (bv_clone.as_f64(), ev_clone.as_f64()) {
                            Value::F64(x.powf(y))
                        } else {
                            return rt_err!("** requires numeric operands");
                        }
                    }
                };
            }

            Instr::Jump(offset) => {
                pc = (pc as i32 + *offset) as usize;
            }
            Instr::JumpFalse(cond, offset) => {
                if !reg!(*cond).is_truthy() {
                    pc = (pc as i32 + *offset) as usize;
                }
            }
            Instr::JumpTrue(cond, offset) => {
                if reg!(*cond).is_truthy() {
                    pc = (pc as i32 + *offset) as usize;
                }
            }

            Instr::Return(r) => {
                return Ok(reg!(*r).clone());
            }
            Instr::ReturnUnit => {
                return Ok(Value::Unit);
            }
            Instr::BreakSignal => {
                return Ok(Value::Break(None));
            }
            Instr::BreakValSignal(r) => {
                return Ok(Value::Break(Some(Box::new(reg!(*r).clone()))));
            }
            Instr::ContinueSignal => {
                return Ok(Value::Continue);
            }

            Instr::Call(d, fn_reg, args_start, arg_count) => {
                let func_v = reg!(*fn_reg).clone();
                let mut call_args = Vec::with_capacity(*arg_count as usize);
                for i in 0..*arg_count {
                    call_args.push(slot!(*args_start + i).clone());
                }
                let mut env = Env::new();
                *reg_mut!(*d) = interp.eval_call(func_v, call_args, &mut env)?;
            }
            Instr::CallBuiltin(d, si, args_start, arg_count) => {
                let name = str_c!(*si);
                let mut call_args = Vec::with_capacity(*arg_count as usize);
                for i in 0..*arg_count {
                    call_args.push(slot!(*args_start + i).clone());
                }
                *reg_mut!(*d) = interp.eval_builtin(name, call_args)?;
            }
            Instr::CallMethod(d, recv_r, mi, args_start, arg_count) => {
                let recv = reg!(*recv_r).clone();
                let method = str_c!(*mi);
                let mut call_args = Vec::with_capacity(*arg_count as usize);
                for i in 0..*arg_count {
                    call_args.push(slot!(*args_start + i).clone());
                }
                let mut env = Env::new();
                *reg_mut!(*d) = interp.eval_method(recv, method, call_args, &mut env)?;
            }

            Instr::NewArray(d) => *reg_mut!(*d) = Value::Array(Arc::new(Mutex::new(Vec::new()))),
            Instr::ArrayPush(arr, v) => {
                if let Value::Array(a) = &reg!(*arr) {
                    let val = reg!(*v).clone();
                    a.lock().unwrap().push(val);
                }
            }
            Instr::ArrayGet(d, arr, idx) => {
                let a = reg!(*arr).clone();
                let i = reg!(*idx).clone();
                *reg_mut!(*d) = interp.eval_index(a, vec![i])?;
            }
            Instr::ArraySet(arr, idx, val) => {
                let i = reg!(*idx).as_i64().unwrap_or(0) as usize;
                let v = reg!(*val).clone();
                if let Value::Array(a) = &reg!(*arr) {
                    let mut lock = a.lock().unwrap();
                    if i < lock.len() {
                        lock[i] = v;
                    }
                }
            }
            Instr::NewHashMap(d) => {
                *reg_mut!(*d) = Value::HashMap(Arc::new(Mutex::new(FxHashMap::default())));
            }
            Instr::NewTuple(d, start, count) => {
                let mut vals = Vec::with_capacity(*count as usize);
                for i in 0..*count {
                    vals.push(regs[(*start + i) as usize].clone());
                }
                *reg_mut!(*d) = Value::Tuple(vals);
            }
            Instr::NewStruct(d, ni) => {
                let name = str_c!(*ni).to_owned();
                *reg_mut!(*d) = Value::Struct {
                    name,
                    fields: FxHashMap::default(),
                };
            }
            Instr::FieldGet(d, obj, fi) => {
                let o = reg!(*obj).clone();
                let field = str_c!(*fi);
                *reg_mut!(*d) = interp.eval_field(o, field)?;
            }
            Instr::FieldSet(obj, fi, val) => {
                let field = str_pool[*fi as usize].clone();
                let v = reg!(*val).clone();
                match &mut regs[*obj as usize] {
                    Value::Struct { fields, .. } => {
                        fields.insert(field, v);
                    }
                    _ => {}
                }
            }
            Instr::IndexGet(d, obj, idx) => {
                let o = reg!(*obj).clone();
                let i = reg!(*idx).clone();
                *reg_mut!(*d) = interp.eval_index(o, vec![i])?;
            }
            Instr::IndexSet(obj, idx, val) => {
                let i = reg!(*idx).as_i64().unwrap_or(0) as usize;
                let v = reg!(*val).clone();
                if let Value::Array(a) = &regs[*obj as usize] {
                    let mut lock = a.lock().unwrap();
                    if i < lock.len() {
                        lock[i] = v;
                    }
                }
            }
            Instr::Vec2Ctor(d, x, y) => {
                let xv = reg!(*x).as_f64().unwrap_or(0.0) as f32;
                let yv = reg!(*y).as_f64().unwrap_or(0.0) as f32;
                *reg_mut!(*d) = Value::Vec2([xv, yv]);
            }
            Instr::Vec3Ctor(d, x, y, z) => {
                let xv = reg!(*x).as_f64().unwrap_or(0.0) as f32;
                let yv = reg!(*y).as_f64().unwrap_or(0.0) as f32;
                let zv = reg!(*z).as_f64().unwrap_or(0.0) as f32;
                *reg_mut!(*d) = Value::Vec3([xv, yv, zv]);
            }
            Instr::Vec4Ctor(d, x, y, z, w) => {
                let xv = reg!(*x).as_f64().unwrap_or(0.0) as f32;
                let yv = reg!(*y).as_f64().unwrap_or(0.0) as f32;
                let zv = reg!(*z).as_f64().unwrap_or(0.0) as f32;
                let wv = reg!(*w).as_f64().unwrap_or(0.0) as f32;
                *reg_mut!(*d) = Value::Vec4([xv, yv, zv, wv]);
            }
            Instr::RangeExcl(d, lo, hi) => {
                let s = reg!(*lo).as_i64().unwrap_or(0) as i32;
                let e = reg!(*hi).as_i64().unwrap_or(0) as i32;
                *reg_mut!(*d) =
                    Value::Array(Arc::new(Mutex::new((s..e).map(Value::I32).collect())));
            }
            Instr::RangeIncl(d, lo, hi) => {
                let s = reg!(*lo).as_i64().unwrap_or(0) as i32;
                let e = reg!(*hi).as_i64().unwrap_or(0) as i32;
                *reg_mut!(*d) =
                    Value::Array(Arc::new(Mutex::new((s..=e).map(Value::I32).collect())));
            }
            Instr::MatMulInstr(d, l, r) => {
                let lv = reg!(*l).clone();
                let rv = reg!(*r).clone();
                *reg_mut!(*d) = interp.eval_matmul(lv, rv)?;
            }
            Instr::HadamardMulInstr(d, l, r) => {
                if let (Value::Tensor(a), Value::Tensor(b)) = (reg!(*l).clone(), reg!(*r).clone()) {
                    let out = a.read().unwrap().hadamard_mul(&b.read().unwrap())?;
                    *reg_mut!(*d) = Value::Tensor(Arc::new(RwLock::new(out)));
                }
            }
            Instr::HadamardDivInstr(d, l, r) => {
                if let (Value::Tensor(a), Value::Tensor(b)) = (reg!(*l).clone(), reg!(*r).clone()) {
                    let out = a.read().unwrap().hadamard_div(&b.read().unwrap())?;
                    *reg_mut!(*d) = Value::Tensor(Arc::new(RwLock::new(out)));
                }
            }
            Instr::TensorConcatInstr(d, l, r) => {
                if let (Value::Tensor(a), Value::Tensor(b)) = (reg!(*l).clone(), reg!(*r).clone()) {
                    let out = a.read().unwrap().concat(&b.read().unwrap())?;
                    *reg_mut!(*d) = Value::Tensor(Arc::new(RwLock::new(out)));
                }
            }
            Instr::EnableGrad(d, s) => {
                let v = reg!(*s).clone();
                if let Value::Tensor(t) = &v {
                    t.write().unwrap().enable_grad();
                }
                *reg_mut!(*d) = v;
            }
        }
    }
}

// ── Specialized i32 VM — runs pure-integer bytecode at ~10x the speed ────────

/// Fast integer-only VM. Uses raw i32 arrays, zero Value allocations per instruction.
/// Returns None if non-i32 values are encountered (caller should fall back to vm_exec).
pub fn vm_exec_i32(
    interp: &mut Interpreter,
    func: &CompiledFn,
    args: &[Value],
) -> Option<Result<Value, RuntimeError>> {
    // Check that all args are i32-compatible.
    let i32_args: Vec<i32> = args
        .iter()
        .map(|v| match v {
            Value::I32(n) => Some(*n),
            Value::I64(n) => Some(*n as i32),
            Value::Unit => Some(0),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;

    let needed = func.slot_count as usize + 32;
    let mut slots = vec![0i32; needed];
    let regs = &mut vec![0i32; needed];

    // Load arguments.
    for (i, &arg) in i32_args.iter().enumerate() {
        if i < regs.len() {
            regs[i] = arg;
        }
    }

    let instrs = &func.instrs;
    let const_pool = &func.const_pool;
    let str_pool = &func.str_pool;
    let mut pc = 0usize;

    macro_rules! reg {
        ($i:expr) => {
            unsafe { *regs.get_unchecked($i as usize) }
        };
    }
    macro_rules! reg_mut {
        ($i:expr) => {
            unsafe { regs.get_unchecked_mut($i as usize) }
        };
    }
    macro_rules! slot {
        ($i:expr) => {
            unsafe { *slots.get_unchecked($i as usize) }
        };
    }
    macro_rules! slot_mut {
        ($i:expr) => {
            unsafe { slots.get_unchecked_mut($i as usize) }
        };
    }
    macro_rules! str_c {
        ($i:expr) => {
            str_pool
                .get($i as usize)
                .expect("vm_i32: string index out of bounds")
                .as_str()
        };
    }
    macro_rules! rt_err {
        ($($arg:tt)*) => {{
            return Some(Err(RuntimeError {
                message: format!($($arg)*),
                span: None,
            }));
        }};
    }

    loop {
        if pc >= instrs.len() {
            return Some(Ok(Value::I32(0)));
        }
        let instr = unsafe { instrs.get_unchecked(pc) };
        pc += 1;

        match instr {
            Instr::Nop => {}
            Instr::LoadUnit(d) => *reg_mut!(*d) = 0,
            Instr::LoadBool(d, b) => *reg_mut!(*d) = if *b { 1 } else { 0 },
            Instr::LoadI32(d, v) => *reg_mut!(*d) = *v,
            Instr::LoadI64(d, v) => *reg_mut!(*d) = *v as i32,
            Instr::LoadF32(..)
            | Instr::LoadF64(..)
            | Instr::LoadStr(..)
            | Instr::LoadConst(..)
            | Instr::LoadFn(..) => {
                return None; // fall back to regular VM
            }
            Instr::Move(d, s) => {
                let v = reg!(*s);
                *reg_mut!(*d) = v;
            }
            Instr::Load(d, slot) => {
                let v = slot!(*slot);
                *reg_mut!(*d) = v;
            }
            Instr::Store(slot, s) => {
                let v = reg!(*s);
                *slot_mut!(*slot) = v;
            }
            Instr::BinOp(d, op, l, r) => {
                let a = reg!(*l);
                let b = reg!(*r);
                *reg_mut!(*d) = match op {
                    BinOpKind::Add => a.wrapping_add(b),
                    BinOpKind::Sub => a.wrapping_sub(b),
                    BinOpKind::Mul => a.wrapping_mul(b),
                    BinOpKind::Div => {
                        if b == 0 {
                            return Some(Err(RuntimeError {
                                message: "division by zero".into(),
                                span: None,
                            }));
                        }
                        a.wrapping_div(b)
                    }
                    BinOpKind::Rem => {
                        if b == 0 {
                            return Some(Err(RuntimeError {
                                message: "modulo by zero".into(),
                                span: None,
                            }));
                        }
                        a.wrapping_rem(b)
                    }
                    BinOpKind::Lt => {
                        if a < b {
                            1
                        } else {
                            0
                        }
                    }
                    BinOpKind::Le => {
                        if a <= b {
                            1
                        } else {
                            0
                        }
                    }
                    BinOpKind::Gt => {
                        if a > b {
                            1
                        } else {
                            0
                        }
                    }
                    BinOpKind::Ge => {
                        if a >= b {
                            1
                        } else {
                            0
                        }
                    }
                    BinOpKind::Eq => {
                        if a == b {
                            1
                        } else {
                            0
                        }
                    }
                    BinOpKind::Ne => {
                        if a != b {
                            1
                        } else {
                            0
                        }
                    }
                    _ => return None,
                };
            }
            Instr::UnOp(d, op, s) => {
                let v = reg!(*s);
                *reg_mut!(*d) = match op {
                    UnOpKind::Neg => v.wrapping_neg(),
                    UnOpKind::Not => {
                        if v == 0 {
                            1
                        } else {
                            0
                        }
                    }
                    _ => return None,
                };
            }
            Instr::Jump(off) => {
                pc = (pc as i32 + *off) as usize;
            }
            Instr::JumpFalse(s, off) => {
                if reg!(*s) == 0 {
                    pc = (pc as i32 + *off) as usize;
                }
            }
            Instr::JumpTrue(s, off) => {
                if reg!(*s) != 0 {
                    pc = (pc as i32 + *off) as usize;
                }
            }
            Instr::Return(s) => {
                return Some(Ok(Value::I32(reg!(*s))));
            }
            Instr::ReturnUnit => {
                return Some(Ok(Value::Unit));
            }
            Instr::BreakSignal | Instr::BreakValSignal(_) | Instr::ContinueSignal => {
                return None;
            }
            // All other instructions need the full VM
            _ => return None,
        }
    }
}

#[inline]
fn vm_unop(op: UnOpKind, v: Value) -> Result<Value, RuntimeError> {
    match op {
        UnOpKind::Neg => match v {
            Value::F32(x) => Ok(Value::F32(-x)),
            Value::F64(x) => Ok(Value::F64(-x)),
            Value::I32(x) => Ok(Value::I32(-x)),
            Value::I64(x) => Ok(Value::I64(-x)),
            Value::Vec3(v) => Ok(Value::Vec3([-v[0], -v[1], -v[2]])),
            Value::Tensor(t) => {
                let data: Vec<f32> = t.read().unwrap().cpu_data().iter().map(|x| -x).collect();
                let shape = t.read().unwrap().shape.clone();
                Ok(Value::Tensor(Arc::new(RwLock::new(Tensor::from_data(
                    shape, data,
                )))))
            }
            _ => rt_err!("unary `-` on `{}`", v.type_name()),
        },
        UnOpKind::Not => match v {
            Value::Bool(b) => Ok(Value::Bool(!b)),
            Value::I32(x) => Ok(Value::I32(!x)),
            Value::I64(x) => Ok(Value::I64(!x)),
            _ => rt_err!("unary `!` on `{}`", v.type_name()),
        },
        UnOpKind::Deref | UnOpKind::Ref | UnOpKind::RefMut => Ok(v),
    }
}

// =============================================================================
// §7  RUNTIME ERROR
// =============================================================================

#[derive(Debug, Clone)]
pub struct RuntimeError {
    pub message: String,
    pub span: Option<Span>,
}

impl RuntimeError {
    #[cold]
    pub fn new(msg: impl Into<String>) -> Self {
        RuntimeError {
            message: msg.into(),
            span: None,
        }
    }
    pub fn at(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.span {
            Some(s) => write!(f, "runtime error at {}: {}", s, self.message),
            None => write!(f, "runtime error: {}", self.message),
        }
    }
}

// =============================================================================
// §8  FUNCTION CLOSURE
// =============================================================================

pub struct FnClosure {
    pub decl: FnDecl,
    pub capture: Frame, // captured environment for closures
}

impl fmt::Debug for FnClosure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<fn {}>", self.decl.name)
    }
}

#[derive(Debug, Clone)]
struct SimEntityState {
    pos: [f32; 2],
    vel: [f32; 2],
    half_extents: [f32; 2],
}

#[derive(Debug, Clone)]
struct SimWorldState {
    dt: f32,
    entities: BTreeMap<i64, SimEntityState>,
    next_entity_id: i64,
    seed: u64,
    step_count: u64,
}

impl SimWorldState {
    fn new(dt: f32, seed: u64) -> Self {
        Self {
            dt,
            entities: BTreeMap::new(),
            next_entity_id: 1,
            seed,
            step_count: 0,
        }
    }

    fn spawn(&mut self, pos: [f32; 2], vel: [f32; 2], half_extents: [f32; 2]) -> i64 {
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        self.entities.insert(
            id,
            SimEntityState {
                pos,
                vel,
                half_extents,
            },
        );
        id
    }

    fn step(&mut self, dt_override: Option<f32>) {
        let dt = dt_override.unwrap_or(self.dt).max(0.0);
        self.step_count = self.step_count.wrapping_add(1);
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        for e in self.entities.values_mut() {
            e.pos[0] += e.vel[0] * dt;
            e.pos[1] += e.vel[1] * dt;
        }

        // Deterministic broadphase via uniform grid + local-neighborhood checks.
        // This keeps behavior deterministic while reducing pair checks in dense worlds.
        let mut max_extent = 0.5_f32;
        for e in self.entities.values() {
            max_extent = max_extent.max(e.half_extents[0]).max(e.half_extents[1]);
        }
        let cell_size = (max_extent * 2.0).max(0.25);

        let mut grid: FxHashMap<(i32, i32), Vec<i64>> =
            FxHashMap::with_capacity_and_hasher(self.entities.len(), Default::default());
        let ids = self.entities.keys().copied().collect::<Vec<_>>();
        for id in &ids {
            if let Some(e) = self.entities.get(id) {
                let cx = (e.pos[0] / cell_size).floor() as i32;
                let cy = (e.pos[1] / cell_size).floor() as i32;
                grid.entry((cx, cy)).or_default().push(*id);
            }
        }

        for id in &ids {
            let e = match self.entities.get(id) {
                Some(e) => e,
                None => continue,
            };
            let cx = (e.pos[0] / cell_size).floor() as i32;
            let cy = (e.pos[1] / cell_size).floor() as i32;
            for ox in -1..=1 {
                for oy in -1..=1 {
                    if let Some(candidates) = grid.get(&(cx + ox, cy + oy)) {
                        for other_id in candidates {
                            if other_id <= id {
                                continue;
                            }
                            let (a_vx, a_vy, b_vx, b_vy, overlap_x, overlap_y) =
                                match (self.entities.get(id), self.entities.get(other_id)) {
                                    (Some(a), Some(b)) => {
                                        let overlap_x = (a.pos[0] - b.pos[0]).abs()
                                            <= (a.half_extents[0] + b.half_extents[0]);
                                        let overlap_y = (a.pos[1] - b.pos[1]).abs()
                                            <= (a.half_extents[1] + b.half_extents[1]);
                                        (
                                            a.vel[0], a.vel[1], b.vel[0], b.vel[1], overlap_x,
                                            overlap_y,
                                        )
                                    }
                                    _ => continue,
                                };
                            if overlap_x && overlap_y {
                                if let Some(a_mut) = self.entities.get_mut(id) {
                                    a_mut.vel[0] = -a_vx;
                                    a_mut.vel[1] = -a_vy;
                                }
                                if let Some(b_mut) = self.entities.get_mut(other_id) {
                                    b_mut.vel[0] = -b_vx;
                                    b_mut.vel[1] = -b_vy;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct WindowState {
    width: i64,
    height: i64,
    title: String,
    is_open: bool,
    frame_count: u64,
}

// =============================================================================
// §9  INTERPRETER
// =============================================================================

/// The main tree-walking interpreter.
pub struct Interpreter {
    /// Top-level function registry.
    pub fns: FxHashMap<String, Arc<FnClosure>>,
    /// Top-level model registry (AST decls; instantiated on demand).
    pub model_decls: FxHashMap<String, ModelDecl>,
    /// Live model instances.
    pub models: FxHashMap<String, Arc<Mutex<NnModel>>>,
    /// Agent declarations.
    pub agent_decls: FxHashMap<String, AgentDecl>,
    /// Struct/component type registry (name → field list).
    pub types: FxHashMap<String, Vec<String>>,
    /// ECS world (global singleton for now).
    pub world: Arc<Mutex<EcsWorld>>,
    /// GPU dispatch backend (None = CPU-only).
    pub gpu: Option<Box<dyn GpuBackend>>,
    /// Thread pool size for `spawn` / parallel loops.
    pub n_threads: usize,
    /// Physics engine (game systems integration)
    pub physics_world: Option<Arc<Mutex<PhysicsWorld>>>,
    /// Rendering state (game systems integration)
    pub render_state: Option<Arc<Mutex<RenderState>>>,
    /// Input state (game systems integration)
    pub input_state: Option<Arc<Mutex<InputState>>>,
    /// Computation graph for autodiff (ML integration)
    pub computation_graph: Option<Arc<Mutex<ComputationGraph>>>,
    /// Active optimizers indexed by ID (ML integration)
    pub optimizers: FxHashMap<String, (Optimizer, OptimizerState)>,
    /// Deterministic simulation worlds (`sim` module)
    sim_worlds: FxHashMap<i64, SimWorldState>,
    next_sim_world_id: i64,
    /// Headless window state handles (`window` module)
    windows: FxHashMap<i64, WindowState>,
    next_window_id: i64,
    // ── Bytecode cache ────────────────────────────────────────────────────────
    /// Maps function name → compiled bytecode (compiled once, reused forever).
    compiled_fns: FxHashMap<String, Arc<CompiledFn>>,
    #[cfg(feature = "phase3-jit")]
    native_fns: FxHashMap<String, Arc<crate::phase3_jit::NativeCode>>,
    #[cfg(feature = "phase3-jit")]
    pgo_started_at: Instant,
    #[cfg(feature = "phase3-jit")]
    pgo_window_done: bool,
    #[cfg(feature = "phase3-jit")]
    pgo_call_counts: FxHashMap<String, u64>,
    /// Runtime function profiler (built-in hotspot weighting).
    runtime_profile_enabled: bool,
    runtime_profile: FxHashMap<String, (u64, u128)>, // fn -> (calls, total_nanos)
    /// Global VM/JIT switch. When disabled, execution falls back to tree-walking.
    jit_enabled: bool,
    /// Native machine-code JIT switch (phase3). Off by default for safety.
    native_jit_enabled: bool,
    /// If enabled, compile all top-level functions at load time to remove first-call latency.
    advance_jit_enabled: bool,
    /// Runtime path counters used by benchmarking/telemetry.
    jit_native_calls: u64,
    jit_vm_calls: u64,
    jit_fallback_calls: u64,
    ecs_query_scratch: Vec<EntityId>,
}

impl Interpreter {
    pub fn new() -> Self {
        Interpreter {
            fns: FxHashMap::default(),
            model_decls: FxHashMap::default(),
            models: FxHashMap::default(),
            agent_decls: FxHashMap::default(),
            types: FxHashMap::default(),
            world: Arc::new(Mutex::new(EcsWorld::default())),
            gpu: Some(Box::new(JulesGpuAdapter::new())),
            n_threads: 4,
            physics_world: Some(Arc::new(Mutex::new(PhysicsWorld::new()))),
            render_state: Some(Arc::new(Mutex::new(RenderState::new()))),
            input_state: Some(Arc::new(Mutex::new(InputState::new()))),
            computation_graph: Some(Arc::new(Mutex::new(ComputationGraph::new()))),
            optimizers: FxHashMap::default(),
            sim_worlds: FxHashMap::default(),
            next_sim_world_id: 1,
            windows: FxHashMap::default(),
            next_window_id: 1,
            compiled_fns: FxHashMap::default(),
            #[cfg(feature = "phase3-jit")]
            native_fns: FxHashMap::default(),
            #[cfg(feature = "phase3-jit")]
            pgo_started_at: Instant::now(),
            #[cfg(feature = "phase3-jit")]
            pgo_window_done: false,
            #[cfg(feature = "phase3-jit")]
            pgo_call_counts: FxHashMap::default(),
            runtime_profile_enabled: false,
            runtime_profile: FxHashMap::default(),
            jit_enabled: true,
            native_jit_enabled: std::env::var("JULES_ENABLE_NATIVE_JIT")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            advance_jit_enabled: true,
            jit_native_calls: 0,
            jit_vm_calls: 0,
            jit_fallback_calls: 0,
            ecs_query_scratch: Vec::new(),
        }
    }

    /// Reset runtime execution-path counters used by benchmarks.
    pub fn reset_jit_counters(&mut self) {
        self.jit_native_calls = 0;
        self.jit_vm_calls = 0;
        self.jit_fallback_calls = 0;
    }

    /// Return `(native_jit, vm_jit, treewalk_fallback)` call counts.
    pub fn jit_counters(&self) -> (u64, u64, u64) {
        (
            self.jit_native_calls,
            self.jit_vm_calls,
            self.jit_fallback_calls,
        )
    }

    fn record_runtime_profile(&mut self, name: &str, elapsed: Duration) {
        if !self.runtime_profile_enabled {
            return;
        }
        let entry = self
            .runtime_profile
            .entry(name.to_string())
            .or_insert((0, 0));
        entry.0 = entry.0.saturating_add(1);
        entry.1 = entry.1.saturating_add(elapsed.as_nanos());
    }

    /// Enable/disable VM/JIT execution globally.
    pub fn set_jit_enabled(&mut self, enabled: bool) {
        self.jit_enabled = enabled;
        if !enabled {
            self.compiled_fns.clear();
            #[cfg(feature = "phase3-jit")]
            self.native_fns.clear();
            #[cfg(feature = "phase3-jit")]
            {
                self.pgo_window_done = false;
                self.pgo_started_at = Instant::now();
                self.pgo_call_counts.clear();
            }
        }
    }

    /// Enable/disable native machine-code JIT execution.
    pub fn set_native_jit_enabled(&mut self, enabled: bool) {
        self.native_jit_enabled = enabled;
        #[cfg(feature = "phase3-jit")]
        if !enabled {
            self.native_fns.clear();
        }
    }

    /// Enable/disable advance JIT (eager pre-compilation on program load).
    pub fn set_advance_jit_enabled(&mut self, enabled: bool) {
        self.advance_jit_enabled = enabled;
    }

    fn precompile_loaded_functions(&mut self) {
        if !self.jit_enabled || !self.advance_jit_enabled {
            return;
        }
        let names: Vec<String> = self.fns.keys().cloned().collect();
        for name in names {
            if self.compiled_fns.contains_key(&name) {
                continue;
            }
            if let Some(closure) = self.fns.get(&name) {
                let compiled = compile_fn(&closure.decl);
                self.compiled_fns.insert(name, Arc::new(compiled));
            }
        }
    }

    // ── Program loading ────────────────────────────────────────────────────

    /// Load all top-level declarations from a parsed program into the interpreter.
    pub fn load_program(&mut self, program: &Program) {
        // Avoid stale bytecode when reloading/redefining functions.
        self.compiled_fns.clear();
        #[cfg(feature = "phase3-jit")]
        {
            self.native_fns.clear();
            self.pgo_window_done = false;
            self.pgo_started_at = Instant::now();
            self.pgo_call_counts.clear();
        }
        for item in &program.items {
            self.load_item(item);
        }
        self.precompile_loaded_functions();
    }

    fn load_item(&mut self, item: &Item) {
        match item {
            Item::Fn(f) => {
                let closure = FnClosure {
                    decl: f.clone(),
                    capture: FxHashMap::default(),
                };
                self.fns.insert(f.name.clone(), Arc::new(closure));
            }
            Item::Component(c) => {
                let fields: Vec<String> = c.fields.iter().map(|f| f.name.clone()).collect();
                self.types.insert(c.name.clone(), fields);
            }
            Item::Struct(s) => {
                let fields: Vec<String> = s.fields.iter().map(|f| f.name.clone()).collect();
                self.types.insert(s.name.clone(), fields);
            }
            Item::Agent(a) => {
                self.agent_decls.insert(a.name.clone(), a.clone());
            }
            Item::Model(m) => {
                self.model_decls.insert(m.name.clone(), m.clone());
            }
            Item::Mod {
                items: Some(inner), ..
            } => {
                for i in inner {
                    self.load_item(i);
                }
            }
            _ => {}
        }
    }

    // ── Run a function by name ──────────────────────────────────────────────

    /// Call a named top-level function.
    ///
    /// **Fast path**: compile the function to bytecode on first invocation,
    /// cache the `CompiledFn`, then run via the register VM on all subsequent
    /// calls.  This eliminates tree-walking overhead (no recursive Rust calls
    /// per AST node, no string-keyed HashMap lookups for local variables).
    ///
    /// Falls back to the tree-walker if compilation fails or the function is
    /// a complex closure.
    pub fn call_fn(&mut self, name: &str, args: Vec<Value>) -> Result<Value, RuntimeError> {
        let started = Instant::now();
        let closure = self
            .fns
            .get(name)
            .cloned()
            .ok_or_else(|| RuntimeError::new(format!("undefined function `{name}`")))?;

        // ── Bytecode fast path ────────────────────────────────────────────────
        if self.jit_enabled {
            // Check if we already have a compiled version.
            if !self.compiled_fns.contains_key(name) {
                // Compile and cache.
                let compiled = compile_fn(&closure.decl);
                self.compiled_fns
                    .insert(name.to_owned(), Arc::new(compiled));
            }
            let compiled = self.compiled_fns[name].clone();
            #[cfg(feature = "phase3-jit")]
            {
                if let Some(count) = self.pgo_call_counts.get_mut(name) {
                    *count += 1;
                } else {
                    self.pgo_call_counts.insert(name.to_owned(), 1);
                }
                if !self.pgo_window_done
                    && self.pgo_started_at.elapsed() >= Duration::from_millis(5)
                {
                    self.pgo_window_done = true;
                    if let Some((hot_name, _)) = self
                        .pgo_call_counts
                        .iter()
                        .max_by_key(|(_, count)| *count)
                        .map(|(k, v)| (k.clone(), *v))
                    {
                        if let Some(hot_compiled) = self.compiled_fns.get(&hot_name).cloned() {
                            if let Some(native) = crate::phase3_jit::translate(&hot_compiled) {
                                self.native_fns.insert(hot_name, Arc::new(native));
                            }
                        }
                    }
                }
            }

            // If the arg count matches expectation, run the VM.
            if closure.capture.is_empty() && args.len() == closure.decl.params.len() {
                #[cfg(feature = "phase3-jit")]
                if self.native_jit_enabled {
                    if let Some(native) = self.native_fns.get(name).cloned() {
                        if let Ok(v) = crate::phase3_jit::execute(&native, &args) {
                            self.jit_native_calls = self.jit_native_calls.saturating_add(1);
                            self.record_runtime_profile(name, started.elapsed());
                            return Ok(v);
                        }
                    } else if let Some(native) = crate::phase3_jit::translate(&compiled) {
                        let native = Arc::new(native);
                        self.native_fns.insert(name.to_owned(), native.clone());
                        if let Ok(v) = crate::phase3_jit::execute(&native, &args) {
                            self.jit_native_calls = self.jit_native_calls.saturating_add(1);
                            self.record_runtime_profile(name, started.elapsed());
                            return Ok(v);
                        }
                    }
                }
                self.jit_vm_calls = self.jit_vm_calls.saturating_add(1);
                // Try the specialized i32 VM first (zero Value allocations).
                if let Some(result) = vm_exec_i32(self, &compiled, &args) {
                    let result = result.map(|r| match r {
                        Value::Return(v) => *v,
                        other => other,
                    });
                    self.record_runtime_profile(name, started.elapsed());
                    return result;
                }
                // Fall back to full VM.
                let result = vm_exec(self, &compiled, args).map(|r| match r {
                    Value::Return(v) => *v,
                    other => other,
                });
                self.record_runtime_profile(name, started.elapsed());
                return result;
            }
        }

        // ── Fallback: tree-walker (captures, mismatched args, etc.) ──────────
        self.jit_fallback_calls = self.jit_fallback_calls.saturating_add(1);
        let mut env = Env::new();
        env.push();
        // Inject captured environment for closures.
        for (k, v) in &closure.capture {
            env.set_local(k, v.clone());
        }
        for (param, val) in closure.decl.params.iter().zip(args) {
            env.set_local(&param.name, val);
        }
        if let Some(body) = &closure.decl.body.clone() {
            let result = self.eval_block(body, &mut env)?;
            env.pop();
            let out = match result {
                Value::Return(v) => Ok(*v),
                other => Ok(other),
            };
            self.record_runtime_profile(name, started.elapsed());
            out
        } else {
            env.pop();
            self.record_runtime_profile(name, started.elapsed());
            Ok(Value::Unit)
        }
    }

    // ── System execution ────────────────────────────────────────────────────

    pub fn run_system(
        &mut self,
        sys: &SystemDecl,
        world: &Arc<Mutex<EcsWorld>>,
        delta_time: f32,
    ) -> Result<(), RuntimeError> {
        let mut env = Env::new();
        env.push();
        // Bind system parameters.
        for param in &sys.params {
            if param.name == "dt" || param.name == "delta_time" {
                env.set_local(&param.name, Value::F32(delta_time));
            } else {
                env.set_local(&param.name, Value::Unit);
            }
        }
        // Bind the world handle.
        env.set_local("world", Value::World(world.clone()));
        self.eval_block(&sys.body, &mut env)?;
        env.pop();
        Ok(())
    }

    // =========================================================================
    // §10  BLOCK EVALUATION
    // =========================================================================

    #[inline]
    pub fn eval_block(&mut self, block: &Block, env: &mut Env) -> Result<Value, RuntimeError> {
        env.push();
        let mut result = Value::Unit;
        for stmt in &block.stmts {
            result = self.eval_stmt(stmt, env)?;
            if result.is_signal() {
                break;
            }
        }
        if !result.is_signal() {
            if let Some(tail) = &block.tail {
                result = self.eval_expr(tail, env)?;
            }
        }
        env.pop();
        Ok(result)
    }

    // =========================================================================
    // §11  STATEMENT EVALUATION
    // =========================================================================

    #[inline(never)] // Large match: keep out of the hot inlining budget.
    pub fn eval_stmt(&mut self, stmt: &Stmt, env: &mut Env) -> Result<Value, RuntimeError> {
        match stmt {
            Stmt::Let { pattern, init, .. } => {
                let val = if let Some(e) = init {
                    self.eval_expr(e, env)?
                } else {
                    Value::Unit
                };
                self.bind_pattern(pattern, val, env);
                Ok(Value::Unit)
            }

            Stmt::Expr { expr, .. } => self.eval_expr(expr, env),

            Stmt::Return { value, .. } => {
                let v = if let Some(e) = value {
                    self.eval_expr(e, env)?
                } else {
                    Value::Unit
                };
                Ok(Value::Return(Box::new(v)))
            }

            Stmt::Break { value, .. } => {
                let v = if let Some(e) = value {
                    Some(Box::new(self.eval_expr(e, env)?))
                } else {
                    None
                };
                Ok(Value::Break(v))
            }

            Stmt::Continue { .. } => Ok(Value::Continue),

            Stmt::ForIn {
                pattern,
                iter,
                body,
                ..
            } => {
                let iter_val = self.eval_expr(iter, env)?;
                match iter_val {
                    // Fast path: iterate string chars directly to avoid collecting a temporary Vec.
                    Value::Str(s) => {
                        for ch in s.chars() {
                            env.push();
                            self.bind_pattern(pattern, Value::Str(ch.to_string()), env);
                            let r = self.eval_block(body, env)?;
                            env.pop();
                            match r {
                                Value::Break(_) => break,
                                Value::Continue => continue,
                                v if v.is_signal() => return Ok(v),
                                _ => {}
                            }
                        }
                    }
                    other => {
                        let items = self.value_to_iter(other)?;
                        for item in items {
                            env.push();
                            self.bind_pattern(pattern, item, env);
                            let r = self.eval_block(body, env)?;
                            env.pop();
                            match r {
                                Value::Break(_) => break,
                                Value::Continue => continue,
                                v if v.is_signal() => return Ok(v),
                                _ => {}
                            }
                        }
                    }
                }
                Ok(Value::Unit)
            }

            Stmt::EntityFor {
                var,
                query,
                body,
                parallelism,
                ..
            } => self.eval_entity_for(var, query, body, *parallelism, env),

            Stmt::While { cond, body, .. } => {
                loop {
                    let c = self.eval_expr(cond, env)?;
                    if !c.is_truthy() {
                        break;
                    }
                    let r = self.eval_block(body, env)?;
                    match r {
                        Value::Break(_) => break,
                        Value::Continue => continue,
                        v if v.is_signal() => return Ok(v),
                        _ => {}
                    }
                }
                Ok(Value::Unit)
            }

            Stmt::Loop { body, .. } => loop {
                let r = self.eval_block(body, env)?;
                match r {
                    Value::Break(v) => return Ok(v.map(|b| *b).unwrap_or(Value::Unit)),
                    Value::Continue => continue,
                    v if v.is_signal() => return Ok(v),
                    _ => {}
                }
            },

            Stmt::If {
                cond, then, else_, ..
            } => {
                let c = self.eval_expr(cond, env)?;
                if c.is_truthy() {
                    self.eval_block(then, env)
                } else if let Some(e) = else_ {
                    match e.as_ref() {
                        crate::ast::IfOrBlock::If(s) => self.eval_stmt(s, env),
                        crate::ast::IfOrBlock::Block(b) => self.eval_block(b, env),
                    }
                } else {
                    Ok(Value::Unit)
                }
            }

            Stmt::Match { expr, arms, .. } => {
                let scrutinee = self.eval_expr(expr, env)?;
                for arm in arms {
                    if self.pattern_matches(&arm.pat, &scrutinee) {
                        env.push();
                        self.bind_pattern(&arm.pat, scrutinee.clone(), env);
                        let guard_ok = if let Some(g) = &arm.guard {
                            self.eval_expr(g, env)?.is_truthy()
                        } else {
                            true
                        };
                        let result = if guard_ok {
                            self.eval_expr(&arm.body, env)?
                        } else {
                            env.pop();
                            continue;
                        };
                        env.pop();
                        return Ok(result);
                    }
                }
                Ok(Value::Unit)
            }

            Stmt::Item(i) => {
                self.load_item(i);
                Ok(Value::Unit)
            }

            Stmt::ParallelFor(pf) => {
                let iter_val = self.eval_expr(&pf.iter, env)?;
                let items = self.value_to_iter(iter_val)?;
                // In the interpreter we execute sequentially;
                // a rayon par_iter() call replaces this in the compiled backend.
                for item in items {
                    env.push();
                    self.bind_pattern(&pf.var, item, env);
                    let r = self.eval_block(&pf.body, env)?;
                    env.pop();
                    match r {
                        Value::Break(_) => break,
                        Value::Continue => continue,
                        v if v.is_signal() => return Ok(v),
                        _ => {}
                    }
                }
                Ok(Value::Unit)
            }

            Stmt::Spawn(sb) => {
                // In the interpreter we run the body inline (no true async).
                self.eval_block(&sb.body, env)
            }

            Stmt::Sync(sb) => self.eval_block(&sb.body, env),
            Stmt::Atomic(ab) => self.eval_block(&ab.body, env),
        }
    }

    // ── Entity-for evaluation ──────────────────────────────────────────────

    fn eval_entity_for(
        &mut self,
        var: &str,
        query: &EntityQuery,
        body: &Block,
        parallelism: ParallelismHint,
        env: &mut Env,
    ) -> Result<Value, RuntimeError> {
        // Snapshot the matching entity list (safe: world locked briefly).
        self.ecs_query_scratch.clear();
        {
            let w = self.world.lock().unwrap();
            w.query_into(&query.with, &query.without, &mut self.ecs_query_scratch);
        }

        // Apply optional filter expression.
        let mut ids_to_run = Vec::new();
        if let Some(filter_expr) = &query.filter {
            for idx in 0..self.ecs_query_scratch.len() {
                let id = self.ecs_query_scratch[idx];
                env.push();
                env.set_local(var, Value::Entity(id));
                let ok = self.eval_expr(filter_expr, env)?.is_truthy();
                env.pop();
                if ok {
                    ids_to_run.push(id);
                }
            }
        } else {
            ids_to_run = self.ecs_query_scratch.clone();
        };

        match parallelism {
            ParallelismHint::Sequential | ParallelismHint::Auto => {
                // Sequential: deterministic order guaranteed.
                if query.filter.is_some() {
                    for id in ids_to_run.iter().copied() {
                        env.push();
                        env.set_local(var, Value::Entity(id));
                        let r = self.eval_block(body, env)?;
                        env.pop();
                        match r {
                            Value::Break(_) => break,
                            Value::Continue => continue,
                            v if v.is_signal() => return Ok(v),
                            _ => {}
                        }
                    }
                } else {
                    for idx in 0..self.ecs_query_scratch.len() {
                        let id = self.ecs_query_scratch[idx];
                        env.push();
                        env.set_local(var, Value::Entity(id));
                        let r = self.eval_block(body, env)?;
                        env.pop();
                        match r {
                            Value::Break(_) => break,
                            Value::Continue => continue,
                            v if v.is_signal() => return Ok(v),
                            _ => {}
                        }
                    }
                }
            }
            ParallelismHint::Parallel
            | ParallelismHint::Simd
            | ParallelismHint::Gpu
            | ParallelismHint::SimdOrGpu { .. } => {
                // Parallel: interpreter falls back to sequential with a note.
                // A real backend uses rayon::par_iter() or GPU dispatch here.
                if query.filter.is_some() {
                    for id in ids_to_run.iter().copied() {
                        env.push();
                        env.set_local(var, Value::Entity(id));
                        let r = self.eval_block(body, env)?;
                        env.pop();
                        if r.is_signal() {
                            return Ok(r);
                        }
                    }
                } else {
                    for idx in 0..self.ecs_query_scratch.len() {
                        let id = self.ecs_query_scratch[idx];
                        env.push();
                        env.set_local(var, Value::Entity(id));
                        let r = self.eval_block(body, env)?;
                        env.pop();
                        if r.is_signal() {
                            return Ok(r);
                        }
                    }
                }
            }
        }
        Ok(Value::Unit)
    }

    // =========================================================================
    // §12  EXPRESSION EVALUATION
    // =========================================================================

    #[inline]
    pub fn eval_expr(&mut self, expr: &Expr, env: &mut Env) -> Result<Value, RuntimeError> {
        match expr {
            Expr::IntLit { value, .. } => Ok(Value::I32(*value as i32)),
            Expr::FloatLit { value, .. } => Ok(Value::F32(*value as f32)),
            Expr::BoolLit { value, .. } => Ok(Value::Bool(*value)),
            Expr::StrLit { value, .. } => Ok(Value::Str(value.clone())),

            Expr::Ident { name, span: _ } => {
                // Check local env first, then built-ins.
                if let Some(v) = env.get(name) {
                    return Ok(v.clone());
                }
                if name == "world" {
                    return Ok(Value::World(self.world.clone()));
                }
                if let Some(f) = self.fns.get(name.as_str()).cloned() {
                    return Ok(Value::Fn(f));
                }
                if let Some(m) = self.models.get(name.as_str()).cloned() {
                    return Ok(Value::Model(m));
                }
                rt_err!("undefined variable `{name}`")
            }

            Expr::Path { segments, .. } => {
                let name = segments.join("::");
                if let Some(v) = env.get(&name) {
                    return Ok(v.clone());
                }
                if let Some(f) = self.fns.get(name.as_str()).cloned() {
                    return Ok(Value::Fn(f));
                }
                rt_err!("undefined path `{name}`")
            }

            // ── Vector constructors ────────────────────────────────────────────
            Expr::VecCtor {
                size,
                elems,
                span: _,
            } => {
                let vals: Vec<f32> = elems
                    .iter()
                    .map(|e| {
                        self.eval_expr(e, env).and_then(|v| {
                            v.as_f64()
                                .map(|f| f as f32)
                                .ok_or_else(|| RuntimeError::new("vec element must be numeric"))
                        })
                    })
                    .collect::<Result<_, _>>()?;
                match size {
                    VecSize::N2 => Ok(Value::Vec2([vals[0], vals[1]])),
                    VecSize::N3 => Ok(Value::Vec3([vals[0], vals[1], vals[2]])),
                    VecSize::N4 => Ok(Value::Vec4([vals[0], vals[1], vals[2], vals[3]])),
                }
            }

            Expr::ArrayLit { elems, .. } => {
                let vals: Vec<Value> = elems
                    .iter()
                    .map(|e| self.eval_expr(e, env))
                    .collect::<Result<_, _>>()?;
                Ok(Value::Array(Arc::new(Mutex::new(vals))))
            }

            Expr::Tuple { elems, .. } => {
                let vals: Vec<Value> = elems
                    .iter()
                    .map(|e| self.eval_expr(e, env))
                    .collect::<Result<_, _>>()?;
                Ok(Value::Tuple(vals))
            }

            // ── Arithmetic ────────────────────────────────────────────────────
            Expr::BinOp { op, lhs, rhs, span } => {
                self.eval_binop(*op, lhs, rhs, env).map_err(|e| e.at(*span))
            }

            Expr::UnOp { op, expr, span } => {
                let v = self.eval_expr(expr, env)?;
                self.eval_unop(*op, v).map_err(|e| e.at(*span))
            }

            // ── Assignment ────────────────────────────────────────────────────
            Expr::Assign {
                op, target, value, ..
            } => {
                let rhs = self.eval_expr(value, env)?;
                self.eval_assign(*op, target, rhs, env)
            }

            // ── Field access ──────────────────────────────────────────────────
            Expr::Field {
                object,
                field,
                span,
            } => {
                let obj = self.eval_expr(object, env)?;
                self.eval_field(obj, field).map_err(|e| e.at(*span))
            }

            // ── Index ─────────────────────────────────────────────────────────
            Expr::Index {
                object,
                indices,
                span,
            } => {
                let obj = self.eval_expr(object, env)?;
                let idxs: Vec<Value> = indices
                    .iter()
                    .map(|i| self.eval_expr(i, env))
                    .collect::<Result<_, _>>()?;
                self.eval_index(obj, idxs).map_err(|e| e.at(*span))
            }

            // ── Calls ─────────────────────────────────────────────────────────
            Expr::Call {
                func,
                args,
                named: _,
                span,
            } => {
                let args_v: Vec<Value> = args
                    .iter()
                    .map(|a| self.eval_expr(a, env))
                    .collect::<Result<_, _>>()?;
                // Check for built-in functions by name first
                if let Expr::Ident { name, .. } = func.as_ref() {
                    if let Ok(result) = self.eval_builtin(name, args_v.clone()) {
                        return Ok(result);
                    }
                }
                let func_v = self.eval_expr(func, env)?;
                self.eval_call(func_v, args_v, env).map_err(|e| e.at(*span))
            }

            Expr::MethodCall {
                receiver,
                method,
                args,
                span,
            } => {
                let recv = self.eval_expr(receiver, env)?;
                let args_v: Vec<Value> = args
                    .iter()
                    .map(|a| self.eval_expr(a, env))
                    .collect::<Result<_, _>>()?;
                self.eval_method(recv, method, args_v, env)
                    .map_err(|e| e.at(*span))
            }

            // ── Tensor-specific (Feature 1) ───────────────────────────────────
            Expr::MatMul { lhs, rhs, span } => {
                let l = self.eval_expr(lhs, env)?;
                let r = self.eval_expr(rhs, env)?;
                self.eval_matmul(l, r).map_err(|e| e.at(*span))
            }

            Expr::HadamardMul { lhs, rhs, span: _ } => {
                let l = self.eval_tensor(lhs, env)?;
                let r = self.eval_tensor(rhs, env)?;
                let out = l.read().unwrap().hadamard_mul(&r.read().unwrap())?;
                Ok(Value::Tensor(Arc::new(RwLock::new(out))))
            }

            Expr::HadamardDiv { lhs, rhs, span: _ } => {
                let l = self.eval_tensor(lhs, env)?;
                let r = self.eval_tensor(rhs, env)?;
                let out = l.read().unwrap().hadamard_div(&r.read().unwrap())?;
                Ok(Value::Tensor(Arc::new(RwLock::new(out))))
            }

            Expr::TensorConcat { lhs, rhs, span: _ } => {
                let l = self.eval_tensor(lhs, env)?;
                let r = self.eval_tensor(rhs, env)?;
                let out = l.read().unwrap().concat(&r.read().unwrap())?;
                Ok(Value::Tensor(Arc::new(RwLock::new(out))))
            }

            Expr::Grad { inner, .. } => {
                let v = self.eval_expr(inner, env)?;
                if let Value::Tensor(t) = &v {
                    t.write().unwrap().enable_grad();
                }
                Ok(v)
            }

            Expr::Pow { base, exp, .. } => {
                let b = self.eval_expr(base, env)?;
                let e = self.eval_expr(exp, env)?;
                match (&b, &e) {
                    (Value::F32(x), Value::F32(y)) => Ok(Value::F32(x.powf(*y))),
                    (Value::F64(x), Value::F64(y)) => Ok(Value::F64(x.powf(*y))),
                    (Value::I32(x), Value::I32(y)) => Ok(Value::I32(x.pow(*y as u32))),
                    _ => {
                        if let (Some(x), Some(y)) = (b.as_f64(), e.as_f64()) {
                            Ok(Value::F64(x.powf(y)))
                        } else {
                            rt_err!("** requires numeric operands")
                        }
                    }
                }
            }

            Expr::Range {
                lo, hi, inclusive, ..
            } => {
                let lo_v = lo.as_ref().map(|e| self.eval_expr(e, env)).transpose()?;
                let hi_v = hi.as_ref().map(|e| self.eval_expr(e, env)).transpose()?;
                let start = lo_v.and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                let end = hi_v.and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                let range: Vec<Value> = if *inclusive {
                    (start..=end).map(Value::I32).collect()
                } else {
                    (start..end).map(Value::I32).collect()
                };
                Ok(Value::Array(Arc::new(Mutex::new(range))))
            }

            Expr::Cast { expr, ty, .. } => {
                let v = self.eval_expr(expr, env)?;
                self.eval_cast(v, ty)
            }

            Expr::IfExpr {
                cond, then, else_, ..
            } => {
                let c = self.eval_expr(cond, env)?;
                if c.is_truthy() {
                    self.eval_block(then, env)
                } else if let Some(b) = else_ {
                    self.eval_block(b, env)
                } else {
                    Ok(Value::Unit)
                }
            }

            Expr::Closure { params, body, .. } => {
                // Capture current environment.
                let mut capture = FxHashMap::default();
                for (k, v) in env.iter_all() {
                    capture.insert(k.to_owned(), v.clone());
                }
                let decl = FnDecl {
                    span: crate::lexer::Span::dummy(),
                    attrs: vec![],
                    name: "<closure>".into(),
                    generics: vec![],
                    params: params.clone(),
                    ret_ty: None,
                    body: Some(Block {
                        span: crate::lexer::Span::dummy(),
                        stmts: vec![],
                        tail: Some(body.clone()),
                    }),
                    is_async: false,
                };
                Ok(Value::Fn(Arc::new(FnClosure { decl, capture })))
            }

            Expr::Block(b) => self.eval_block(b, env),

            Expr::StructLit { name, fields, .. } => {
                let mut field_vals = FxHashMap::default();
                for (fname, fexpr) in fields {
                    field_vals.insert(fname.clone(), self.eval_expr(fexpr, env)?);
                }
                Ok(Value::Struct {
                    name: name.clone(),
                    fields: field_vals,
                })
            }
            Expr::KronProd { lhs, rhs, .. } => {
                let l = self.eval_tensor(lhs, env)?;
                let r = self.eval_tensor(rhs, env)?;
                let out = l.read().unwrap().kron(&r.read().unwrap())?;
                Ok(Value::Tensor(Arc::new(RwLock::new(out))))
            }
            Expr::OuterProd { lhs, rhs, .. } => {
                let l = self.eval_tensor(lhs, env)?;
                let r = self.eval_tensor(rhs, env)?;
                let out = l.read().unwrap().outer(&r.read().unwrap())?;
                Ok(Value::Tensor(Arc::new(RwLock::new(out))))
            }
        }
    }

    // ── Tensor helper ──────────────────────────────────────────────────────

    fn eval_tensor(
        &mut self,
        expr: &Expr,
        env: &mut Env,
    ) -> Result<Arc<RwLock<Tensor>>, RuntimeError> {
        match self.eval_expr(expr, env)? {
            Value::Tensor(t) => Ok(t),
            other => rt_err!("expected tensor, got `{}`", other.type_name()),
        }
    }

    // =========================================================================
    // §13  OPERATOR EVALUATION
    // =========================================================================

    #[inline]
    fn eval_binop(
        &mut self,
        op: BinOpKind,
        lhs: &Expr,
        rhs: &Expr,
        env: &mut Env,
    ) -> Result<Value, RuntimeError> {
        // Short-circuit logical operators.
        if op == BinOpKind::And {
            let l = self.eval_expr(lhs, env)?;
            return if !l.is_truthy() {
                Ok(Value::Bool(false))
            } else {
                Ok(Value::Bool(self.eval_expr(rhs, env)?.is_truthy()))
            };
        }
        if op == BinOpKind::Or {
            let l = self.eval_expr(lhs, env)?;
            return if l.is_truthy() {
                Ok(Value::Bool(true))
            } else {
                Ok(Value::Bool(self.eval_expr(rhs, env)?.is_truthy()))
            };
        }

        let l = self.eval_expr(lhs, env)?;
        let r = self.eval_expr(rhs, env)?;
        eval_numeric_binop(op, l, r)
    }

    #[inline]
    fn eval_unop(&self, op: UnOpKind, v: Value) -> Result<Value, RuntimeError> {
        match op {
            UnOpKind::Neg => match v {
                Value::F32(x) => Ok(Value::F32(-x)),
                Value::F64(x) => Ok(Value::F64(-x)),
                Value::I32(x) => Ok(Value::I32(-x)),
                Value::I64(x) => Ok(Value::I64(-x)),
                Value::Vec3(v) => Ok(Value::Vec3([-v[0], -v[1], -v[2]])),
                Value::Tensor(t) => {
                    let data: Vec<f32> = t.read().unwrap().cpu_data().iter().map(|x| -x).collect();
                    let shape = t.read().unwrap().shape.clone();
                    Ok(Value::Tensor(Arc::new(RwLock::new(Tensor::from_data(
                        shape, data,
                    )))))
                }
                _ => rt_err!("unary `-` on `{}`", v.type_name()),
            },
            UnOpKind::Not => match v {
                Value::Bool(b) => Ok(Value::Bool(!b)),
                Value::I32(x) => Ok(Value::I32(!x)),
                Value::I64(x) => Ok(Value::I64(!x)),
                _ => rt_err!("unary `!` on `{}`", v.type_name()),
            },
            UnOpKind::Deref | UnOpKind::Ref | UnOpKind::RefMut => Ok(v),
        }
    }

    fn eval_matmul(&mut self, l: Value, r: Value) -> Result<Value, RuntimeError> {
        match (l, r) {
            (Value::Tensor(a), Value::Tensor(b)) => {
                let a_guard = a.read().unwrap();
                let b_guard = b.read().unwrap();
                let a_shape = a_guard.shape.clone();
                let b_shape = b_guard.shape.clone();
                let a_cpu: Vec<f32> = match &a_guard.data {
                    TensorStorage::Cpu(v) => v.clone(),
                    TensorStorage::Gpu(h) => {
                        if let Some(gpu) = &self.gpu {
                            gpu.download(h)
                        } else {
                            return rt_err!("tensor is on GPU but no backend is configured");
                        }
                    }
                };
                let b_cpu: Vec<f32> = match &b_guard.data {
                    TensorStorage::Cpu(v) => v.clone(),
                    TensorStorage::Gpu(h) => {
                        if let Some(gpu) = &self.gpu {
                            gpu.download(h)
                        } else {
                            return rt_err!("tensor is on GPU but no backend is configured");
                        }
                    }
                };
                let out = if let Some(gpu) = &self.gpu {
                    let ga = gpu.upload(&a_cpu, a_shape.clone());
                    let gb = gpu.upload(&b_cpu, b_shape.clone());
                    let gout = gpu.matmul(&ga, &gb, &a_shape, &b_shape);
                    let out_data = gpu.download(&gout);
                    let mut out_shape = if a_shape.len() > 2 {
                        a_shape[..a_shape.len() - 2].to_vec()
                    } else {
                        Vec::new()
                    };
                    out_shape.push(a_shape[a_shape.len() - 2]);
                    out_shape.push(b_shape[b_shape.len() - 1]);
                    Tensor::from_data(out_shape, out_data)
                } else {
                    a_guard.matmul(&b_guard)?
                };
                Ok(Value::Tensor(Arc::new(RwLock::new(out))))
            }
            (Value::Mat3(a), Value::Mat3(b)) => Ok(Value::Mat3(mat3_mul(a, b))),
            (Value::Mat4(a), Value::Mat4(b)) => Ok(Value::Mat4(mat4_mul(a, b))),
            (Value::Mat3(m), Value::Vec3(v)) => Ok(Value::Vec3(mat3_vec3_mul(m, v))),
            (Value::Mat4(m), Value::Vec4(v)) => Ok(Value::Vec4(mat4_vec4_mul(m, v))),
            (l, r) => rt_err!(
                "@ requires tensor/matrix operands, got `{}` @ `{}`",
                l.type_name(),
                r.type_name()
            ),
        }
    }

    // =========================================================================
    // §14  ASSIGNMENT
    // =========================================================================

    fn eval_assign(
        &mut self,
        op: AssignOpKind,
        target: &Expr,
        rhs: Value,
        env: &mut Env,
    ) -> Result<Value, RuntimeError> {
        // For compound assignments, read current value first.
        let effective_rhs = if op == AssignOpKind::Assign {
            rhs
        } else {
            let current = self.eval_expr(target, env)?;
            let bin_op = op
                .to_binop()
                .ok_or_else(|| RuntimeError::new("unknown compound assignment"))?;
            // MatMulAssign: @=
            if op == AssignOpKind::MatMulAssign {
                self.eval_matmul(current, rhs)?
            } else {
                eval_numeric_binop(bin_op, current, rhs)?
            }
        };

        // Write back to the target.
        match target {
            Expr::Ident { name, .. } => {
                env.set(name, effective_rhs);
            }
            Expr::Field { object, field, .. } => {
                if let Expr::Ident { name, .. } = object.as_ref() {
                    // For entity field access, write to the world.
                    if let Some(Value::Entity(id)) = env.get(name).cloned() {
                        let mut w = self.world.lock().unwrap();
                        if let Some(comp) = w.get_component_mut(id, field) {
                            *comp = effective_rhs;
                        } else {
                            w.insert_component(id, field, effective_rhs);
                        }
                    } else if let Some(Value::Struct { fields: _, .. }) = env.get(name).cloned() {
                        let mut s = env.get(name).unwrap().clone();
                        if let Value::Struct { ref mut fields, .. } = s {
                            fields.insert(field.clone(), effective_rhs);
                        }
                        env.set(name, s);
                    }
                }
            }
            Expr::Index {
                object, indices, ..
            } => {
                if let Expr::Ident { name, .. } = object.as_ref() {
                    let idx = if let Some(e) = indices.first() {
                        self.eval_expr(e, env)?.as_i64().unwrap_or(0) as usize
                    } else {
                        0
                    };
                    if let Some(Value::Array(arr)) = env.get(name).cloned() {
                        arr.lock().unwrap()[idx] = effective_rhs;
                    }
                }
            }
            _ => {}
        }
        Ok(Value::Unit)
    }

    // =========================================================================
    // §15  FIELD AND INDEX ACCESS
    // =========================================================================

    fn eval_field(&mut self, obj: Value, field: &str) -> Result<Value, RuntimeError> {
        match obj {
            Value::Struct { ref fields, .. } => fields
                .get(field)
                .cloned()
                .ok_or_else(|| RuntimeError::new(format!("no field `{field}`"))),
            Value::Entity(id) => {
                let w = self.world.lock().unwrap();
                // Field name maps to component type by convention (lowercase → CamelCase).
                // We try the field name directly, then a title-cased version.
                let comp = w.get_component(id, field).or_else(|| {
                    let titled = title_case(field);
                    w.get_component(id, &titled)
                });
                comp.cloned()
                    .ok_or_else(|| RuntimeError::new(format!("entity has no component `{field}`")))
            }
            Value::Vec2(v) => swizzle_vec(&v, field).map_err(RuntimeError::new),
            Value::Vec3(v) => swizzle_vec(&v, field).map_err(RuntimeError::new),
            Value::Vec4(v) => swizzle_vec(&v, field).map_err(RuntimeError::new),
            Value::Quat(q) => match field {
                "x" => Ok(Value::F32(q[0])),
                "y" => Ok(Value::F32(q[1])),
                "z" => Ok(Value::F32(q[2])),
                "w" => Ok(Value::F32(q[3])),
                _ => rt_err!("quat has no field `{field}`"),
            },
            Value::Tuple(vs) => {
                let idx: usize = field
                    .parse()
                    .map_err(|_| RuntimeError::new(format!("bad tuple field `{field}`")))?;
                vs.into_iter()
                    .nth(idx)
                    .ok_or_else(|| RuntimeError::new(format!("tuple index {idx} out of range")))
            }
            other => rt_err!("`{}` has no field `{field}`", other.type_name()),
        }
    }

    fn eval_index(&self, obj: Value, indices: Vec<Value>) -> Result<Value, RuntimeError> {
        match obj {
            Value::Array(arr) => {
                let idx = indices.first().and_then(|v| v.as_i64()).unwrap_or(0) as usize;
                let a = arr.lock().unwrap();
                a.get(idx)
                    .cloned()
                    .ok_or_else(|| RuntimeError::new(format!("index {idx} out of bounds")))
            }
            Value::Tensor(t) => {
                let t = t.read().unwrap();
                let flat_idx = tensor_flat_index(&t.shape, &indices)?;
                Ok(Value::F32(t.cpu_data()[flat_idx]))
            }
            Value::Str(s) => {
                let idx = indices.first().and_then(|v| v.as_i64()).unwrap_or(0) as usize;
                s.chars()
                    .nth(idx)
                    .map(|c| Value::Str(c.to_string()))
                    .ok_or_else(|| RuntimeError::new("string index out of range"))
            }
            other => rt_err!("cannot index `{}`", other.type_name()),
        }
    }

    // =========================================================================
    // §16  FUNCTION CALLS
    // =========================================================================

    fn eval_call(
        &mut self,
        func: Value,
        args: Vec<Value>,
        _env: &mut Env,
    ) -> Result<Value, RuntimeError> {
        match func {
            Value::Fn(closure) => {
                let closure = closure.clone();
                let mut call_env = Env::new();
                // Inject capture.
                for (k, v) in &closure.capture {
                    call_env.set_local(k, v.clone());
                }
                call_env.push();
                for (param, arg) in closure.decl.params.iter().zip(args) {
                    call_env.set_local(&param.name, arg);
                }
                let body = closure.decl.body.clone();
                let result = if let Some(b) = body {
                    self.eval_block(&b, &mut call_env)?
                } else {
                    Value::Unit
                };
                call_env.pop();
                match result {
                    Value::Return(v) => Ok(*v),
                    other => Ok(other),
                }
            }
            _ => {
                // Built-in functions.
                rt_err!("not callable")
            }
        }
    }

    // ── Built-in function dispatch ────────────────────────────────────────────

    fn canonical_builtin_name<'a>(name: &'a str) -> &'a str {
        static ALIASES: OnceLock<FxHashMap<&'static str, &'static str>> = OnceLock::new();
        let aliases = ALIASES.get_or_init(|| {
            FxHashMap::from_iter([
                ("core::some", "Some"),
                ("core::Some", "Some"),
                ("core::none", "None"),
                ("core::None", "None"),
                ("core::ok", "Ok"),
                ("core::Ok", "Ok"),
                ("core::err", "Err"),
                ("core::Err", "Err"),
                ("core::unwrap", "unwrap"),
                ("core::is_some", "is_some"),
                ("core::is_none", "is_none"),
                ("core::is_ok", "is_ok"),
                ("core::is_err", "is_err"),
                ("collections::map_new", "HashMap::new"),
                ("collections::len", "len"),
                ("collections::range", "range"),
                ("math::sin", "sin"),
                ("math::cos", "cos"),
                ("math::tan", "tan"),
                ("math::exp", "exp"),
                ("math::log", "log"),
                ("math::sqrt", "sqrt"),
                ("math::tanh", "tanh"),
                ("math::lerp", "mix"),
                ("math::smoothstep", "smoothstep"),
                ("math::clamp01", "math::clamp01"),
                ("math::approach", "math::approach"),
                ("math::move_towards2", "math::move_towards2"),
                ("math::angle_to", "math::angle_to"),
                ("math::rand_unit2", "math::rand_unit2"),
                ("tensor::zeros", "zeros"),
                ("tensor::ones", "ones"),
                ("tensor::random_seed", "math::random_seed"),
                ("nn::cross_entropy", "loss::cross_entropy"),
                ("nn::mse", "loss::mse"),
                ("train::optimizer_create", "optimizer::create"),
                ("train::optimizer_step", "optimizer::step"),
                ("data::dataloader", "dataloader"),
                ("data::pipeline", "pipeline"),
                ("io::read_text", "read_file"),
                ("io::write_text", "write_file"),
                ("model::load_ir", "read_file"),
                ("model::save_ir", "write_file"),
                ("model::load_weights", "sys::read_bytes"),
                ("model::save_weights", "sys::write_bytes"),
                ("debug::trace", "dbg"),
                ("sim::world_new", "sim::world"),
                ("window::new", "window::create"),
                ("math::vec2", "math::vec2"),
                ("math::vec3", "math::vec3"),
                ("math::vec4", "math::vec4"),
                ("math::dot2", "math::dot2"),
                ("math::dot3", "math::dot3"),
                ("math::cross3", "math::cross3"),
                ("math::normalize2", "math::normalize2"),
                ("math::normalize3", "math::normalize3"),
                ("math::normalize4", "math::normalize4"),
                ("math::lerp", "math::lerp"),
                ("math::quat", "math::quat"),
                ("math::quat_mul", "math::quat_mul"),
                ("math::quat_slerp", "math::quat_slerp"),
                ("math::mat4_identity", "math::mat4_identity"),
                ("math::mat4_mul", "math::mat4_mul"),
                ("math::mat4_translate", "math::mat4_translate"),
                ("math::mat4_scale", "math::mat4_scale"),
                ("math::mat4_perspective", "math::mat4_perspective"),
                ("geom::ray", "geom::ray"),
                ("geom::aabb", "geom::aabb"),
                ("geom::sphere", "geom::sphere"),
                ("geom::plane", "geom::plane"),
                ("geom::sdf_sphere", "geom::sdf_sphere"),
                ("geom::sdf_box", "geom::sdf_box"),
                ("random::rand", "random::rand"),
                ("random::rand_int", "random::rand_int"),
                ("random::rand_normal", "random::rand_normal"),
                ("random::choice", "random::choice"),
                ("random::shuffle", "random::shuffle"),
                ("noise::perlin2", "noise::perlin2"),
                ("noise::perlin3", "noise::perlin3"),
                ("noise::simplex2", "noise::simplex2"),
                ("noise::fbm2", "noise::fbm2"),
                ("spatial::hash_new", "spatial::hash_new"),
                ("spatial::quadtree_new", "spatial::quadtree_new"),
                ("path::astar", "path::astar"),
                ("path::dijkstra", "path::dijkstra"),
                ("ai::seek", "ai::seek"),
                ("ai::flee", "ai::flee"),
                ("ai::boid_separation", "ai::boid_separation"),
                ("ai::utility_score", "ai::utility_score"),
                ("net::tcp_listen", "net::tcp_listen"),
                ("net::udp_bind", "net::udp_bind"),
                ("alloc::arena_new", "alloc::arena_new"),
                ("collections::mpsc_new", "collections::mpsc_new"),
                ("render::begin_frame", "render::begin_frame"),
                ("render::clear", "render::clear"),
                ("render::rect", "render::rect"),
                ("render::sprite", "render::sprite"),
                ("render::flush", "render::flush"),
                ("render::stats", "render::stats"),
            ])
        });
        aliases.get(name).copied().unwrap_or(name)
    }

    fn stdlib_modules_value(&self) -> Value {
        let std_modules = crate::jules_std::modules_value();
        let modules: [(&str, &[&str]); 19] = [
            (
                "core",
                &[
                    "Some", "None", "Ok", "Err", "unwrap", "is_some", "is_none", "is_ok", "is_err",
                ],
            ),
            (
                "math",
                &[
                    "sin",
                    "cos",
                    "tan",
                    "exp",
                    "log",
                    "sqrt",
                    "tanh",
                    "softmax",
                    "random",
                    "random_seed",
                    "rand_int",
                    "sigmoid",
                    "relu",
                    "lerp",
                    "smoothstep",
                    "dot2",
                    "length2",
                    "distance2",
                    "remap",
                    "clamp01",
                    "approach",
                    "move_towards2",
                    "angle_to",
                    "rand_unit2",
                ],
            ),
            (
                "tensor",
                &[
                    "zeros",
                    "ones",
                    "sum",
                    "mean",
                    "max",
                    "softmax",
                    "normalize",
                ],
            ),
            (
                "nn",
                &["relu", "gelu", "sigmoid", "tanh", "cross_entropy", "mse"],
            ),
            ("train", &["optimizer::create", "optimizer::step"]),
            ("data", &["dataloader", "pipeline"]),
            ("io", &["read_file", "write_file", "append_file"]),
            (
                "sys",
                &["sys::cwd", "sys::os", "sys::arch", "sys::list_dir"],
            ),
            ("error", &["Ok", "Err", "unwrap"]),
            ("diag", &["diag::warn", "diag::perf_hint"]),
            ("collections", &["HashMap::new", "len", "range"]),
            ("compute", &["compute::device", "compute::parallel_map"]),
            ("quant", &["quant::int8_export"]),
            (
                "model",
                &[
                    "read_file",
                    "write_file",
                    "sys::read_bytes",
                    "sys::write_bytes",
                ],
            ),
            (
                "debug",
                &[
                    "dbg",
                    "debug::tensor_shape",
                    "debug::disable_jit",
                    "debug::enable_jit",
                    "debug::set_advance_jit",
                    "debug::jit_state",
                    "debug::runtime_profile",
                    "debug::runtime_hotspots",
                    "debug::runtime_profile_reset",
                ],
            ),
            (
                "sim",
                &[
                    "sim::world",
                    "sim::spawn",
                    "sim::step",
                    "sim::get_state",
                    "sim::state_tensor",
                    "sim::apply",
                    "sim::entity_count",
                    "sim::nearest_entity",
                    "sim::query_radius",
                ],
            ),
            (
                "std",
                &[
                    "std::modules",
                    "math::",
                    "geom::",
                    "random::",
                    "noise::",
                    "spatial::",
                    "path::",
                    "ai::",
                    "net::",
                    "alloc::",
                    "collections::",
                ],
            ),
            (
                "window",
                &[
                    "window::create",
                    "window::open",
                    "window::clear",
                    "window::draw_rect",
                    "window::present",
                    "window::close",
                    "window::input_key_down",
                    "window::size",
                    "window::title",
                    "window::frames",
                ],
            ),
            (
                "render",
                &[
                    "render::begin_frame",
                    "render::clear",
                    "render::rect",
                    "render::sprite",
                    "render::flush",
                    "render::stats",
                ],
            ),
        ];
        let mut out = FxHashMap::default();
        for (module, names) in modules {
            let vals = names
                .iter()
                .map(|n| Value::Str((*n).to_string()))
                .collect::<Vec<_>>();
            out.insert(module.to_string(), Value::Array(Arc::new(Mutex::new(vals))));
        }
        // Merge std library modules into the output without overwriting
        // interpreter-specific catalogs (which intentionally include short names).
        if let Value::HashMap(std_map) = std_modules {
            if let Ok(map) = std_map.lock() {
                for (k, v) in map.iter() {
                    out.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
        }
        Value::HashMap(Arc::new(Mutex::new(out)))
    }

    pub fn eval_builtin(&mut self, name: &str, args: Vec<Value>) -> Result<Value, RuntimeError> {
        use std::f64::consts;
        let name = Self::canonical_builtin_name(name);

        match name {
            // ── Math functions ────────────────────────────────────────────────
            "sin" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.sin() as f32))
                } else {
                    rt_err!("sin() requires a number")
                }
            }
            "cos" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.cos() as f32))
                } else {
                    rt_err!("cos() requires a number")
                }
            }
            "tan" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.tan() as f32))
                } else {
                    rt_err!("tan() requires a number")
                }
            }
            "asin" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.asin() as f32))
                } else {
                    rt_err!("asin() requires a number")
                }
            }
            "acos" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.acos() as f32))
                } else {
                    rt_err!("acos() requires a number")
                }
            }
            "atan" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.atan() as f32))
                } else {
                    rt_err!("atan() requires a number")
                }
            }
            "atan2" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                ) {
                    (Some(y), Some(x)) => Ok(Value::F32(y.atan2(x) as f32)),
                    _ => rt_err!("atan2() requires two numbers"),
                }
            }
            "sqrt" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.sqrt() as f32))
                } else {
                    rt_err!("sqrt() requires a number")
                }
            }
            "cbrt" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.cbrt() as f32))
                } else {
                    rt_err!("cbrt() requires a number")
                }
            }
            "pow" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                ) {
                    (Some(x), Some(y)) => Ok(Value::F32(x.powf(y) as f32)),
                    _ => rt_err!("pow() requires two numbers"),
                }
            }
            "exp" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.exp() as f32))
                } else {
                    rt_err!("exp() requires a number")
                }
            }
            "exp2" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.exp2() as f32))
                } else {
                    rt_err!("exp2() requires a number")
                }
            }
            "exp10" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.log10().exp() as f32))
                } else {
                    rt_err!("exp10() requires a number")
                }
            }
            "ln" | "log" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.ln() as f32))
                } else {
                    rt_err!("ln() requires a number")
                }
            }
            "log2" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.log2() as f32))
                } else {
                    rt_err!("log2() requires a number")
                }
            }
            "log10" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.log10() as f32))
                } else {
                    rt_err!("log10() requires a number")
                }
            }
            "abs" => match args.first() {
                Some(Value::F32(x)) => Ok(Value::F32(x.abs())),
                Some(Value::F64(x)) => Ok(Value::F64(x.abs())),
                Some(Value::I32(x)) => Ok(Value::I32(x.abs())),
                Some(Value::I64(x)) => Ok(Value::I64(x.abs())),
                Some(v) => {
                    if let Some(x) = v.as_f64() {
                        Ok(Value::F32(x.abs() as f32))
                    } else {
                        rt_err!("abs() requires a number")
                    }
                }
                _ => rt_err!("abs() requires a number"),
            },
            "floor" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.floor() as f32))
                } else {
                    rt_err!("floor() requires a number")
                }
            }
            "ceil" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.ceil() as f32))
                } else {
                    rt_err!("ceil() requires a number")
                }
            }
            "round" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.round() as f32))
                } else {
                    rt_err!("round() requires a number")
                }
            }
            "trunc" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.trunc() as f32))
                } else {
                    rt_err!("trunc() requires a number")
                }
            }
            "fract" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.fract() as f32))
                } else {
                    rt_err!("fract() requires a number")
                }
            }
            "min" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                ) {
                    (Some(x), Some(y)) => Ok(Value::F32(x.min(y) as f32)),
                    _ => rt_err!("min() requires two numbers"),
                }
            }
            "max" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                ) {
                    (Some(x), Some(y)) => Ok(Value::F32(x.max(y) as f32)),
                    _ => rt_err!("max() requires two numbers"),
                }
            }
            "clamp" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                ) {
                    (Some(x), Some(lo), Some(hi)) => Ok(Value::F32((x.max(lo).min(hi)) as f32)),
                    _ => rt_err!("clamp() requires three numbers"),
                }
            }
            "degrees" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32((x * 180.0 / consts::PI) as f32))
                } else {
                    rt_err!("degrees() requires a number")
                }
            }
            "radians" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32((x * consts::PI / 180.0) as f32))
                } else {
                    rt_err!("radians() requires a number")
                }
            }
            "sign" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    if x > 0.0 {
                        Ok(Value::F32(1.0))
                    } else if x < 0.0 {
                        Ok(Value::F32(-1.0))
                    } else {
                        Ok(Value::F32(0.0))
                    }
                } else {
                    rt_err!("sign() requires a number")
                }
            }
            "step" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                ) {
                    (Some(edge), Some(x)) => Ok(Value::F32(if x >= edge { 1.0 } else { 0.0 })),
                    _ => rt_err!("step() requires two numbers"),
                }
            }
            "smoothstep" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                ) {
                    (Some(edge0), Some(edge1), Some(x)) => {
                        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
                        Ok(Value::F32((t * t * (3.0 - 2.0 * t)) as f32))
                    }
                    _ => rt_err!("smoothstep() requires three numbers"),
                }
            }
            "mix" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                ) {
                    (Some(x), Some(y), Some(a)) => Ok(Value::F32((x * (1.0 - a) + y * a) as f32)),
                    _ => rt_err!("mix() requires three numbers"),
                }
            }
            "tanh" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.tanh() as f32))
                } else {
                    rt_err!("tanh() requires a number")
                }
            }
            "math::random_seed" => {
                if let Some(seed) = args.first().and_then(|v| v.as_i64()) {
                    use std::sync::atomic::Ordering::Relaxed;
                    RAND_STATE.store(seed as u64, Relaxed);
                    Ok(Value::Bool(true))
                } else {
                    rt_err!("math::random_seed(seed) requires integer seed")
                }
            }
            "math::random" => Ok(Value::F32(pseudo_rand())),
            "math::rand_int" => {
                let lo = args.get(0).and_then(|v| v.as_i64()).unwrap_or(0);
                let hi = args.get(1).and_then(|v| v.as_i64()).unwrap_or(lo + 1);
                if hi <= lo {
                    return rt_err!("math::rand_int(lo, hi) requires hi > lo");
                }
                let r = pseudo_rand();
                let span = (hi - lo) as f32;
                Ok(Value::I64(lo + (r * span).floor() as i64))
            }
            "math::sigmoid" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32((1.0 / (1.0 + (-x).exp())) as f32))
                } else {
                    rt_err!("math::sigmoid(x) requires a number")
                }
            }
            "math::relu" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32((x.max(0.0)) as f32))
                } else {
                    rt_err!("math::relu(x) requires a number")
                }
            }
            "math::clamp01" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x.clamp(0.0, 1.0) as f32))
                } else {
                    rt_err!("math::clamp01(x) requires a number")
                }
            }
            "math::approach" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                ) {
                    (Some(current), Some(target), Some(max_delta)) => {
                        let d = (target - current).abs();
                        if d <= max_delta.max(0.0) {
                            Ok(Value::F32(target as f32))
                        } else if target > current {
                            Ok(Value::F32((current + max_delta.max(0.0)) as f32))
                        } else {
                            Ok(Value::F32((current - max_delta.max(0.0)) as f32))
                        }
                    }
                    _ => {
                        rt_err!("math::approach(current, target, max_delta) requires three numbers")
                    }
                }
            }
            "math::dot2" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                    args.get(3).and_then(|v| v.as_f64()),
                ) {
                    (Some(ax), Some(ay), Some(bx), Some(by)) => {
                        Ok(Value::F32((ax * bx + ay * by) as f32))
                    }
                    _ => rt_err!("math::dot2(ax, ay, bx, by) requires four numbers"),
                }
            }
            "math::length2" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                ) {
                    (Some(x), Some(y)) => Ok(Value::F32((x * x + y * y).sqrt() as f32)),
                    _ => rt_err!("math::length2(x, y) requires two numbers"),
                }
            }
            "math::distance2" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                    args.get(3).and_then(|v| v.as_f64()),
                ) {
                    (Some(ax), Some(ay), Some(bx), Some(by)) => {
                        let dx = ax - bx;
                        let dy = ay - by;
                        Ok(Value::F32((dx * dx + dy * dy).sqrt() as f32))
                    }
                    _ => rt_err!("math::distance2(ax, ay, bx, by) requires four numbers"),
                }
            }
            "math::remap" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                    args.get(3).and_then(|v| v.as_f64()),
                    args.get(4).and_then(|v| v.as_f64()),
                ) {
                    (Some(x), Some(in0), Some(in1), Some(out0), Some(out1)) => {
                        let denom = (in1 - in0).abs().max(1e-12);
                        let t = (x - in0) / denom;
                        Ok(Value::F32((out0 + (out1 - out0) * t) as f32))
                    }
                    _ => rt_err!("math::remap(x, in0, in1, out0, out1) requires five numbers"),
                }
            }
            "math::move_towards2" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                    args.get(3).and_then(|v| v.as_f64()),
                    args.get(4).and_then(|v| v.as_f64()),
                ) {
                    (Some(cx), Some(cy), Some(tx), Some(ty), Some(max_delta)) => {
                        let dx = tx - cx;
                        let dy = ty - cy;
                        let dist = (dx * dx + dy * dy).sqrt();
                        if dist <= max_delta.max(0.0) || dist <= 1e-12 {
                            Ok(Value::Array(Arc::new(Mutex::new(vec![
                                Value::F32(tx as f32),
                                Value::F32(ty as f32),
                            ]))))
                        } else {
                            let s = max_delta.max(0.0) / dist;
                            Ok(Value::Array(Arc::new(Mutex::new(vec![
                                Value::F32((cx + dx * s) as f32),
                                Value::F32((cy + dy * s) as f32),
                            ]))))
                        }
                    }
                    _ => rt_err!(
                        "math::move_towards2(cx, cy, tx, ty, max_delta) requires five numbers"
                    ),
                }
            }
            "math::angle_to" => {
                match (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                    args.get(3).and_then(|v| v.as_f64()),
                ) {
                    (Some(ax), Some(ay), Some(bx), Some(by)) => {
                        Ok(Value::F32((by - ay).atan2(bx - ax) as f32))
                    }
                    _ => rt_err!("math::angle_to(ax, ay, bx, by) requires four numbers"),
                }
            }
            "math::rand_unit2" => {
                let theta = pseudo_rand() as f64 * (2.0 * consts::PI);
                Ok(Value::Array(Arc::new(Mutex::new(vec![
                    Value::F32(theta.cos() as f32),
                    Value::F32(theta.sin() as f32),
                ]))))
            }
            "std::modules" => Ok(self.stdlib_modules_value()),

            "range" => {
                let start = args.get(0).and_then(|v| v.as_i64()).unwrap_or(0);
                let end = args.get(1).and_then(|v| v.as_i64()).unwrap_or(start);
                let step = args.get(2).and_then(|v| v.as_i64()).unwrap_or(1);
                if step == 0 {
                    return rt_err!("range() step must be nonzero");
                }
                let mut items = Vec::new();
                if step > 0 {
                    let mut i = start;
                    while i < end {
                        items.push(Value::I32(i as i32));
                        i += step;
                    }
                } else {
                    let mut i = start;
                    while i > end {
                        items.push(Value::I32(i as i32));
                        i += step;
                    }
                }
                Ok(Value::Array(Arc::new(Mutex::new(items))))
            }
            "arange" => {
                let start = args.get(0).and_then(|v| v.as_i64()).unwrap_or(0);
                let end = args.get(1).and_then(|v| v.as_i64()).unwrap_or(start);
                let step = args.get(2).and_then(|v| v.as_i64()).unwrap_or(1);
                if step == 0 {
                    return rt_err!("arange() step must be nonzero");
                }
                let mut items = Vec::new();
                if step > 0 {
                    let mut i = start;
                    while i < end {
                        items.push(Value::I32(i as i32));
                        i += step;
                    }
                } else {
                    let mut i = start;
                    while i > end {
                        items.push(Value::I32(i as i32));
                        i += step;
                    }
                }
                Ok(Value::Array(Arc::new(Mutex::new(items))))
            }
            "zeros" => {
                let len = args.get(0).and_then(|v| v.as_i64()).unwrap_or(0).max(0) as usize;
                let v = vec![Value::F32(0.0); len];
                Ok(Value::Array(Arc::new(Mutex::new(v))))
            }
            "ones" => {
                let len = args.get(0).and_then(|v| v.as_i64()).unwrap_or(0).max(0) as usize;
                let v = vec![Value::F32(1.0); len];
                Ok(Value::Array(Arc::new(Mutex::new(v))))
            }
            "math::softmax" | "tensor::softmax" => {
                if let Some(Value::Array(a)) = args.first() {
                    let values = a.lock().unwrap();
                    let nums: Vec<f32> = values
                        .iter()
                        .map(|v| v.as_f64().map(|x| x as f32))
                        .collect::<Option<Vec<f32>>>()
                        .ok_or_else(|| RuntimeError::new("softmax requires numeric array"))?;
                    let max = nums.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
                    let exps: Vec<f32> = nums.iter().map(|x| (x - max).exp()).collect();
                    let sum: f32 = exps.iter().sum();
                    let out = exps
                        .into_iter()
                        .map(|x| Value::F32(x / sum.max(1e-12)))
                        .collect::<Vec<_>>();
                    Ok(Value::Array(Arc::new(Mutex::new(out))))
                } else if let Some(Value::Tensor(t)) = args.first() {
                    let out = t.read().unwrap().apply_activation(&Activation::Softmax);
                    Ok(Value::Tensor(Arc::new(RwLock::new(out))))
                } else {
                    rt_err!("softmax requires Array[number] or Tensor")
                }
            }
            "tensor::sum" => match args.first() {
                Some(Value::Tensor(t)) => Ok(Value::F32(t.read().unwrap().sum_all())),
                Some(Value::Array(a)) => {
                    let mut s = 0.0f32;
                    for v in a.lock().unwrap().iter() {
                        s += v
                            .as_f64()
                            .ok_or_else(|| RuntimeError::new("tensor::sum expects numeric array"))?
                            as f32;
                    }
                    Ok(Value::F32(s))
                }
                _ => rt_err!("tensor::sum expects Tensor or Array[number]"),
            },
            "tensor::mean" => match args.first() {
                Some(Value::Tensor(t)) => {
                    let tt = t.read().unwrap();
                    Ok(Value::F32(tt.sum_all() / tt.numel().max(1) as f32))
                }
                Some(Value::Array(a)) => {
                    let values = a.lock().unwrap();
                    if values.is_empty() {
                        return Ok(Value::F32(0.0));
                    }
                    let mut s = 0.0f32;
                    for v in values.iter() {
                        s += v.as_f64().ok_or_else(|| {
                            RuntimeError::new("tensor::mean expects numeric array")
                        })? as f32;
                    }
                    Ok(Value::F32(s / values.len() as f32))
                }
                _ => rt_err!("tensor::mean expects Tensor or Array[number]"),
            },
            "tensor::max" => match args.first() {
                Some(Value::Tensor(t)) => {
                    let tt = t.read().unwrap();
                    let m = tt
                        .cpu_data()
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    Ok(Value::F32(m))
                }
                Some(Value::Array(a)) => {
                    let mut max_v = f32::NEG_INFINITY;
                    for v in a.lock().unwrap().iter() {
                        let x = v
                            .as_f64()
                            .ok_or_else(|| RuntimeError::new("tensor::max expects numeric array"))?
                            as f32;
                        max_v = max_v.max(x);
                    }
                    Ok(Value::F32(max_v))
                }
                _ => rt_err!("tensor::max expects Tensor or Array[number]"),
            },
            "tensor::normalize" => match args.first() {
                Some(Value::Tensor(t)) => {
                    let src = t.read().unwrap();
                    let norm = src
                        .cpu_data()
                        .iter()
                        .map(|x| x * x)
                        .sum::<f32>()
                        .sqrt()
                        .max(1e-12);
                    let data = src.cpu_data().iter().map(|x| x / norm).collect::<Vec<_>>();
                    Ok(Value::Tensor(Arc::new(RwLock::new(Tensor::from_data(
                        src.shape.clone(),
                        data,
                    )))))
                }
                Some(Value::Array(a)) => {
                    let values = a.lock().unwrap();
                    let nums = values
                        .iter()
                        .map(|v| v.as_f64().map(|x| x as f32))
                        .collect::<Option<Vec<f32>>>()
                        .ok_or_else(|| {
                            RuntimeError::new("tensor::normalize expects numeric array")
                        })?;
                    let norm = nums.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                    let out = nums
                        .into_iter()
                        .map(|x| Value::F32(x / norm))
                        .collect::<Vec<_>>();
                    Ok(Value::Array(Arc::new(Mutex::new(out))))
                }
                _ => rt_err!("tensor::normalize expects Tensor or Array[number]"),
            },
            "dataloader" | "pipeline" => {
                let source = args.get(0).ok_or_else(|| {
                    RuntimeError::new("dataloader() requires a source array or tensor")
                })?;
                let batch_size = args
                    .get(1)
                    .and_then(|v| v.as_i64())
                    .map(|v| v.max(1) as usize)
                    .unwrap_or(1);
                let shuffle = args.get(2).and_then(|v| v.as_bool()).unwrap_or(false);

                let mut samples: Vec<Value> = Vec::new();
                match source {
                    Value::Array(a) => samples = a.lock().unwrap().clone(),
                    Value::Tensor(t) => {
                        let t = t.read().unwrap();
                        if t.shape.is_empty() {
                            return rt_err!("dataloader() requires tensor with rank >=1");
                        }
                        let rows = t.shape[0];
                        let single = t.shape.iter().skip(1).product::<usize>();
                        if rows * single != t.numel() {
                            return rt_err!("dataloader(): invalid tensor shape");
                        }
                        for i in 0..rows {
                            let start = i * single;
                            let end = start + single;
                            let chunk = t.cpu_data()[start..end].to_vec();
                            let mut row_shape = t.shape.clone();
                            row_shape.remove(0);
                            samples.push(Value::Tensor(Arc::new(RwLock::new(Tensor::from_data(
                                row_shape, chunk,
                            )))));
                        }
                    }
                    _ => return rt_err!("dataloader() source must be Array or Tensor"),
                }

                let mut loader = DataLoader {
                    samples,
                    batch_size,
                    index: 0,
                    shuffle,
                };
                if shuffle {
                    loader.reset();
                }
                Ok(Value::DataLoader(Arc::new(Mutex::new(loader))))
            }

            // ── I/O functions ─────────────────────────────────────────────────
            "print" => {
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        print!(" ");
                    }
                    print!("{}", arg);
                }
                Ok(Value::Unit)
            }
            "println" => {
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        print!(" ");
                    }
                    print!("{}", arg);
                }
                println!();
                Ok(Value::Unit)
            }
            "dbg" => {
                println!("[DEBUG] {:?}", args);
                Ok(args.first().cloned().unwrap_or(Value::Unit))
            }
            "diag::warn" => {
                if let Some(msg) = args.first().map(|v| v.to_string()) {
                    eprintln!("[diag::warn] {}", msg);
                    Ok(Value::Bool(true))
                } else {
                    rt_err!("diag::warn(message) requires a message")
                }
            }
            "diag::perf_hint" => {
                if let Some(msg) = args.first().map(|v| v.to_string()) {
                    eprintln!("[diag::perf_hint] {}", msg);
                    Ok(Value::Bool(true))
                } else {
                    rt_err!("diag::perf_hint(message) requires a message")
                }
            }
            "compute::device" => Ok(Value::Str("cpu".to_string())),
            "compute::parallel_map" => {
                if let Some(Value::Array(a)) = args.first() {
                    Ok(Value::Array(Arc::new(Mutex::new(
                        a.lock().unwrap().clone(),
                    ))))
                } else {
                    rt_err!("compute::parallel_map currently expects an array")
                }
            }
            "debug::tensor_shape" => {
                if let Some(Value::Tensor(t)) = args.first() {
                    let shape_vals = t
                        .read()
                        .unwrap()
                        .shape
                        .iter()
                        .map(|d| Value::I64(*d as i64))
                        .collect::<Vec<_>>();
                    Ok(Value::Array(Arc::new(Mutex::new(shape_vals))))
                } else {
                    rt_err!("debug::tensor_shape requires a Tensor")
                }
            }
            "debug::disable_jit" => {
                self.set_jit_enabled(false);
                Ok(Value::Bool(true))
            }
            "debug::enable_jit" => {
                self.set_jit_enabled(true);
                Ok(Value::Bool(true))
            }
            "debug::set_advance_jit" => {
                let enabled = args.first().and_then(|v| v.as_bool()).ok_or_else(|| {
                    RuntimeError::new("debug::set_advance_jit(bool) requires bool")
                })?;
                self.set_advance_jit_enabled(enabled);
                Ok(Value::Bool(enabled))
            }
            "debug::jit_state" => {
                let mut state = FxHashMap::default();
                state.insert("jit_enabled".to_string(), Value::Bool(self.jit_enabled));
                state.insert(
                    "advance_jit_enabled".to_string(),
                    Value::Bool(self.advance_jit_enabled),
                );
                state.insert(
                    "compiled_fn_count".to_string(),
                    Value::I64(self.compiled_fns.len() as i64),
                );
                Ok(Value::HashMap(Arc::new(Mutex::new(state))))
            }
            "debug::runtime_profile" => {
                let enabled = args.first().and_then(|v| v.as_bool()).unwrap_or(true);
                self.runtime_profile_enabled = enabled;
                Ok(Value::Bool(enabled))
            }
            "debug::runtime_profile_reset" => {
                self.runtime_profile.clear();
                Ok(Value::Bool(true))
            }
            "debug::runtime_hotspots" => {
                let limit = args.first().and_then(|v| v.as_i64()).unwrap_or(8).max(1) as usize;
                let total_nanos: u128 = self.runtime_profile.values().map(|(_, n)| *n).sum();
                let mut rows = self
                    .runtime_profile
                    .iter()
                    .map(|(name, (calls, nanos))| (name.clone(), *calls, *nanos))
                    .collect::<Vec<_>>();
                rows.sort_by(|a, b| b.2.cmp(&a.2));
                let mut out = Vec::new();
                for (name, calls, nanos) in rows.into_iter().take(limit) {
                    let mut row = FxHashMap::default();
                    let total_ms = nanos as f64 / 1_000_000.0;
                    let avg_us = if calls == 0 {
                        0.0
                    } else {
                        (nanos as f64 / calls as f64) / 1_000.0
                    };
                    let weight = if total_nanos == 0 {
                        0.0
                    } else {
                        nanos as f64 / total_nanos as f64
                    };
                    row.insert("fn".to_string(), Value::Str(name));
                    row.insert("calls".to_string(), Value::I64(calls as i64));
                    row.insert("total_ms".to_string(), Value::F64(total_ms));
                    row.insert("avg_us".to_string(), Value::F64(avg_us));
                    row.insert("weight".to_string(), Value::F64(weight));
                    out.push(Value::HashMap(Arc::new(Mutex::new(row))));
                }
                Ok(Value::Array(Arc::new(Mutex::new(out))))
            }
            "quant::int8_export" => Ok(Value::Bool(true)),
            "sim::world" => {
                let dt = args
                    .first()
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(0.016);
                let seed = args.get(1).and_then(|v| v.as_i64()).unwrap_or(12345) as u64;
                let world_id = self.next_sim_world_id;
                self.next_sim_world_id += 1;
                self.sim_worlds
                    .insert(world_id, SimWorldState::new(dt.max(0.0), seed));
                Ok(Value::I64(world_id))
            }
            "sim::spawn" => {
                let world_id = args.get(0).and_then(|v| v.as_i64()).ok_or_else(|| {
                    RuntimeError::new("sim::spawn(world_id, entity) requires world_id")
                })?;
                let world = self
                    .sim_worlds
                    .get_mut(&world_id)
                    .ok_or_else(|| RuntimeError::new("sim::spawn: unknown world id"))?;
                let mut pos = [0.0_f32, 0.0_f32];
                let mut vel = [0.0_f32, 0.0_f32];
                let mut size = [0.5_f32, 0.5_f32];
                if let Some(Value::HashMap(map)) = args.get(1) {
                    let m = map.lock().unwrap();
                    if let Some(Value::Array(arr)) = m.get("position") {
                        let vals = arr.lock().unwrap();
                        if vals.len() >= 2 {
                            pos[0] = vals[0].as_f64().unwrap_or(0.0) as f32;
                            pos[1] = vals[1].as_f64().unwrap_or(0.0) as f32;
                        }
                    }
                    if let Some(Value::Array(arr)) = m.get("velocity") {
                        let vals = arr.lock().unwrap();
                        if vals.len() >= 2 {
                            vel[0] = vals[0].as_f64().unwrap_or(0.0) as f32;
                            vel[1] = vals[1].as_f64().unwrap_or(0.0) as f32;
                        }
                    }
                    if let Some(Value::Array(arr)) = m.get("size") {
                        let vals = arr.lock().unwrap();
                        if vals.len() >= 2 {
                            size[0] = vals[0].as_f64().unwrap_or(0.5) as f32 * 0.5;
                            size[1] = vals[1].as_f64().unwrap_or(0.5) as f32 * 0.5;
                        }
                    }
                }
                let id = world.spawn(pos, vel, size);
                Ok(Value::I64(id))
            }
            "sim::step" => {
                let world_id = args.first().and_then(|v| v.as_i64()).ok_or_else(|| {
                    RuntimeError::new("sim::step(world_id, dt?) requires world_id")
                })?;
                let dt = args.get(1).and_then(|v| v.as_f64()).map(|v| v as f32);
                let world = self
                    .sim_worlds
                    .get_mut(&world_id)
                    .ok_or_else(|| RuntimeError::new("sim::step: unknown world id"))?;
                world.step(dt);
                Ok(Value::Bool(true))
            }
            "sim::reset" => {
                let world_id = args
                    .first()
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| RuntimeError::new("sim::reset(world_id) requires world_id"))?;
                let seed = self
                    .sim_worlds
                    .get(&world_id)
                    .map(|w| w.seed)
                    .unwrap_or(12345);
                let dt = self
                    .sim_worlds
                    .get(&world_id)
                    .map(|w| w.dt)
                    .unwrap_or(0.016);
                self.sim_worlds
                    .insert(world_id, SimWorldState::new(dt, seed));
                Ok(Value::Bool(true))
            }
            "sim::apply" => {
                let world_id = args.get(0).and_then(|v| v.as_i64()).ok_or_else(|| {
                    RuntimeError::new("sim::apply(world_id, entity_id, action) requires world_id")
                })?;
                let entity_id = args
                    .get(1)
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| RuntimeError::new("sim::apply requires entity_id"))?;
                let world = self
                    .sim_worlds
                    .get_mut(&world_id)
                    .ok_or_else(|| RuntimeError::new("sim::apply: unknown world id"))?;
                let ent = world
                    .entities
                    .get_mut(&entity_id)
                    .ok_or_else(|| RuntimeError::new("sim::apply: unknown entity id"))?;
                if let Some(Value::Array(action)) = args.get(2) {
                    let a = action.lock().unwrap();
                    if a.len() >= 2 {
                        ent.vel[0] += a[0].as_f64().unwrap_or(0.0) as f32;
                        ent.vel[1] += a[1].as_f64().unwrap_or(0.0) as f32;
                    }
                }
                Ok(Value::Bool(true))
            }
            "sim::get_state" => {
                let world_id = args.first().and_then(|v| v.as_i64()).ok_or_else(|| {
                    RuntimeError::new("sim::get_state(world_id) requires world_id")
                })?;
                let world = self
                    .sim_worlds
                    .get(&world_id)
                    .ok_or_else(|| RuntimeError::new("sim::get_state: unknown world id"))?;
                let mut ids = world.entities.keys().copied().collect::<Vec<_>>();
                ids.sort_unstable();
                let mut out = Vec::with_capacity(ids.len());
                for id in ids {
                    if let Some(e) = world.entities.get(&id) {
                        let mut row = FxHashMap::default();
                        row.insert("id".into(), Value::I64(id));
                        row.insert(
                            "position".into(),
                            Value::Array(Arc::new(Mutex::new(vec![
                                Value::F32(e.pos[0]),
                                Value::F32(e.pos[1]),
                            ]))),
                        );
                        row.insert(
                            "velocity".into(),
                            Value::Array(Arc::new(Mutex::new(vec![
                                Value::F32(e.vel[0]),
                                Value::F32(e.vel[1]),
                            ]))),
                        );
                        out.push(Value::HashMap(Arc::new(Mutex::new(row))));
                    }
                }
                Ok(Value::Array(Arc::new(Mutex::new(out))))
            }
            "sim::state_tensor" => {
                let world_id = args.first().and_then(|v| v.as_i64()).ok_or_else(|| {
                    RuntimeError::new("sim::state_tensor(world_id) requires world_id")
                })?;
                let world = self
                    .sim_worlds
                    .get(&world_id)
                    .ok_or_else(|| RuntimeError::new("sim::state_tensor: unknown world id"))?;
                let mut ids = world.entities.keys().copied().collect::<Vec<_>>();
                ids.sort_unstable();
                let mut data = Vec::with_capacity(ids.len() * 4);
                for id in ids {
                    if let Some(e) = world.entities.get(&id) {
                        data.extend_from_slice(&[e.pos[0], e.pos[1], e.vel[0], e.vel[1]]);
                    }
                }
                let t = Tensor::from_data(vec![world.entities.len(), 4], data);
                Ok(Value::Tensor(Arc::new(RwLock::new(t))))
            }
            "sim::entity_count" => {
                let world_id = args.first().and_then(|v| v.as_i64()).ok_or_else(|| {
                    RuntimeError::new("sim::entity_count(world_id) requires world_id")
                })?;
                let world = self
                    .sim_worlds
                    .get(&world_id)
                    .ok_or_else(|| RuntimeError::new("sim::entity_count: unknown world id"))?;
                Ok(Value::I64(world.entities.len() as i64))
            }
            "sim::nearest_entity" => {
                let world_id = args.first().and_then(|v| v.as_i64()).ok_or_else(|| {
                    RuntimeError::new(
                        "sim::nearest_entity(world_id, [x,y], radius?) requires world_id",
                    )
                })?;
                let pos = match args.get(1) {
                    Some(Value::Array(arr)) => {
                        let vals = arr.lock().unwrap();
                        if vals.len() < 2 {
                            return rt_err!("sim::nearest_entity requires [x,y] position");
                        }
                        [
                            vals[0].as_f64().unwrap_or(0.0) as f32,
                            vals[1].as_f64().unwrap_or(0.0) as f32,
                        ]
                    }
                    _ => return rt_err!("sim::nearest_entity requires [x,y] position"),
                };
                let radius = args
                    .get(2)
                    .and_then(|v| v.as_f64())
                    .unwrap_or(f64::INFINITY) as f32;
                let radius2 = radius * radius;
                let world = self
                    .sim_worlds
                    .get(&world_id)
                    .ok_or_else(|| RuntimeError::new("sim::nearest_entity: unknown world id"))?;
                let mut best: Option<(i64, f32)> = None;
                for (id, e) in &world.entities {
                    let dx = e.pos[0] - pos[0];
                    let dy = e.pos[1] - pos[1];
                    let d2 = dx * dx + dy * dy;
                    if d2 > radius2 {
                        continue;
                    }
                    match best {
                        Some((_, bd2)) if d2 >= bd2 => {}
                        _ => best = Some((*id, d2)),
                    }
                }
                Ok(Value::I64(best.map(|(id, _)| id).unwrap_or(-1)))
            }
            "sim::query_radius" => {
                let world_id = args.first().and_then(|v| v.as_i64()).ok_or_else(|| {
                    RuntimeError::new(
                        "sim::query_radius(world_id, [x,y], radius) requires world_id",
                    )
                })?;
                let pos = match args.get(1) {
                    Some(Value::Array(arr)) => {
                        let vals = arr.lock().unwrap();
                        if vals.len() < 2 {
                            return rt_err!("sim::query_radius requires [x,y] position");
                        }
                        [
                            vals[0].as_f64().unwrap_or(0.0) as f32,
                            vals[1].as_f64().unwrap_or(0.0) as f32,
                        ]
                    }
                    _ => return rt_err!("sim::query_radius requires [x,y] position"),
                };
                let radius = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                let radius2 = radius.max(0.0) * radius.max(0.0);
                let world = self
                    .sim_worlds
                    .get(&world_id)
                    .ok_or_else(|| RuntimeError::new("sim::query_radius: unknown world id"))?;
                let mut ids = Vec::new();
                for (id, e) in &world.entities {
                    let dx = e.pos[0] - pos[0];
                    let dy = e.pos[1] - pos[1];
                    let d2 = dx * dx + dy * dy;
                    if d2 <= radius2 {
                        ids.push(*id);
                    }
                }
                ids.sort_unstable();
                let vals = ids.into_iter().map(Value::I64).collect::<Vec<_>>();
                Ok(Value::Array(Arc::new(Mutex::new(vals))))
            }
            "window::create" => {
                let width = args.get(0).and_then(|v| v.as_i64()).unwrap_or(800).max(1);
                let height = args.get(1).and_then(|v| v.as_i64()).unwrap_or(600).max(1);
                let title = args
                    .get(2)
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "Jules".to_string());
                let id = self.next_window_id;
                self.next_window_id += 1;
                self.windows.insert(
                    id,
                    WindowState {
                        width,
                        height,
                        title,
                        is_open: true,
                        frame_count: 0,
                    },
                );
                Ok(Value::I64(id))
            }
            "window::open" => {
                let id = args.first().and_then(|v| v.as_i64()).ok_or_else(|| {
                    RuntimeError::new("window::open(window_id) requires window_id")
                })?;
                let is_open = self.windows.get(&id).map(|w| w.is_open).unwrap_or(false);
                Ok(Value::Bool(is_open))
            }
            "window::clear" | "window::draw_rect" => {
                let id = args.first().and_then(|v| v.as_i64()).ok_or_else(|| {
                    RuntimeError::new("window::clear/draw_rect requires window_id")
                })?;
                if self.windows.contains_key(&id) {
                    Ok(Value::Bool(true))
                } else {
                    rt_err!("window: unknown window id")
                }
            }
            "window::present" => {
                let id = args.first().and_then(|v| v.as_i64()).ok_or_else(|| {
                    RuntimeError::new("window::present(window_id) requires window_id")
                })?;
                if let Some(w) = self.windows.get_mut(&id) {
                    w.frame_count = w.frame_count.saturating_add(1);
                    let _ = (w.width, w.height, w.title.len());
                    // Keep deterministic upper bound to avoid accidental infinite loops in headless mode.
                    if w.frame_count > 120_000 {
                        w.is_open = false;
                    }
                    Ok(Value::Bool(true))
                } else {
                    rt_err!("window::present: unknown window id")
                }
            }
            "window::close" => {
                let id = args.first().and_then(|v| v.as_i64()).ok_or_else(|| {
                    RuntimeError::new("window::close(window_id) requires window_id")
                })?;
                if let Some(w) = self.windows.get_mut(&id) {
                    w.is_open = false;
                    Ok(Value::Bool(true))
                } else {
                    rt_err!("window::close: unknown window id")
                }
            }
            "window::input_key_down" => Ok(Value::Bool(false)),
            "window::size" => {
                let id = args.first().and_then(|v| v.as_i64()).ok_or_else(|| {
                    RuntimeError::new("window::size(window_id) requires window_id")
                })?;
                if let Some(w) = self.windows.get(&id) {
                    Ok(Value::Array(Arc::new(Mutex::new(vec![
                        Value::I64(w.width),
                        Value::I64(w.height),
                    ]))))
                } else {
                    rt_err!("window::size: unknown window id")
                }
            }
            "window::title" => {
                let id = args.first().and_then(|v| v.as_i64()).ok_or_else(|| {
                    RuntimeError::new("window::title(window_id) requires window_id")
                })?;
                if let Some(w) = self.windows.get(&id) {
                    Ok(Value::Str(w.title.clone()))
                } else {
                    rt_err!("window::title: unknown window id")
                }
            }
            "window::frames" => {
                let id = args.first().and_then(|v| v.as_i64()).ok_or_else(|| {
                    RuntimeError::new("window::frames(window_id) requires window_id")
                })?;
                if let Some(w) = self.windows.get(&id) {
                    Ok(Value::I64(w.frame_count as i64))
                } else {
                    rt_err!("window::frames: unknown window id")
                }
            }

            // ── Type conversion ───────────────────────────────────────────────
            "i32" => {
                if let Some(x) = args.first().and_then(|v| v.as_i64()) {
                    Ok(Value::I32(x as i32))
                } else if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::I32(x as i32))
                } else {
                    rt_err!("i32() requires a number")
                }
            }
            "i64" => {
                if let Some(x) = args.first().and_then(|v| v.as_i64()) {
                    Ok(Value::I64(x))
                } else if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::I64(x as i64))
                } else {
                    rt_err!("i64() requires a number")
                }
            }
            "f32" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F32(x as f32))
                } else {
                    rt_err!("f32() requires a number")
                }
            }
            "f64" => {
                if let Some(x) = args.first().and_then(|v| v.as_f64()) {
                    Ok(Value::F64(x))
                } else {
                    rt_err!("f64() requires a number")
                }
            }
            "bool" => Ok(Value::Bool(
                args.first().map(|v| v.is_truthy()).unwrap_or(false),
            )),
            "str" => Ok(Value::Str(
                args.first().map(|v| v.to_string()).unwrap_or_default(),
            )),

            // ── Collection functions ──────────────────────────────────────────
            "len" => match args.first() {
                Some(Value::Array(a)) => Ok(Value::I32(a.lock().unwrap().len() as i32)),
                Some(Value::Str(s)) => Ok(Value::I32(s.len() as i32)),
                Some(v) => rt_err!("len() not applicable to {}", v.type_name()),
                None => rt_err!("len() requires an argument"),
            },
            // ── Option / Result constructors ───────────────────────────────────
            "Some" => Ok(Value::Some(Box::new(
                args.into_iter().next().unwrap_or(Value::Unit),
            ))),
            "None" => Ok(Value::None),
            "Ok" => Ok(Value::Ok(Box::new(
                args.into_iter().next().unwrap_or(Value::Unit),
            ))),
            "Err" => Ok(Value::Err(Box::new(
                args.into_iter().next().unwrap_or(Value::Unit),
            ))),
            "unwrap" => match args.first() {
                Some(Value::Some(v)) => Ok((**v).clone()),
                Some(Value::Ok(v)) => Ok((**v).clone()),
                Some(Value::None) => rt_err!("called unwrap() on None"),
                Some(Value::Err(e)) => rt_err!("called unwrap() on Err: {}", e),
                _ => rt_err!("unwrap() requires Option or Result"),
            },
            "is_some" => match args.first() {
                Some(Value::Some(_)) => Ok(Value::Bool(true)),
                Some(Value::None) => Ok(Value::Bool(false)),
                _ => rt_err!("is_some() requires Option"),
            },
            "is_none" => match args.first() {
                Some(Value::Some(_)) => Ok(Value::Bool(false)),
                Some(Value::None) => Ok(Value::Bool(true)),
                _ => rt_err!("is_none() requires Option"),
            },
            "is_ok" => match args.first() {
                Some(Value::Ok(_)) => Ok(Value::Bool(true)),
                Some(Value::Err(_)) => Ok(Value::Bool(false)),
                _ => rt_err!("is_ok() requires Result"),
            },
            "is_err" => match args.first() {
                Some(Value::Ok(_)) => Ok(Value::Bool(false)),
                Some(Value::Err(_)) => Ok(Value::Bool(true)),
                _ => rt_err!("is_err() requires Result"),
            },

            // ── String functions ──────────────────────────────────────────────
            "concat" => {
                let strs: Vec<String> = args.iter().map(|v| v.to_string()).collect();
                Ok(Value::Str(strs.join("")))
            }

            // ── HashMap / Collection constructors ──────────────────────────────
            "HashMap::new" => Ok(Value::HashMap(Arc::new(Mutex::new(FxHashMap::default())))),

            // ── File I/O ───────────────────────────────────────────────────────
            "read_file" => {
                if let Some(Value::Str(path)) = args.first() {
                    match std::fs::read_to_string(path) {
                        Ok(content) => Ok(Value::Str(content)),
                        Err(e) => rt_err!("read_file failed: {}", e),
                    }
                } else {
                    rt_err!("read_file requires a path string")
                }
            }
            "write_file" => match (args.get(0), args.get(1)) {
                (Some(Value::Str(path)), Some(Value::Str(content))) => {
                    match std::fs::write(path, content) {
                        Ok(_) => Ok(Value::Bool(true)),
                        Err(e) => rt_err!("write_file failed: {}", e),
                    }
                }
                _ => rt_err!("write_file requires (path, content) strings"),
            },
            "file_exists" => {
                if let Some(Value::Str(path)) = args.first() {
                    Ok(Value::Bool(std::path::Path::new(path).exists()))
                } else {
                    rt_err!("file_exists requires a path string")
                }
            }
            "delete_file" => {
                if let Some(Value::Str(path)) = args.first() {
                    match std::fs::remove_file(path) {
                        Ok(_) => Ok(Value::Bool(true)),
                        Err(e) => rt_err!("delete_file failed: {}", e),
                    }
                } else {
                    rt_err!("delete_file requires a path string")
                }
            }
            "append_file" => match (args.get(0), args.get(1)) {
                (Some(Value::Str(path)), Some(Value::Str(content))) => {
                    use std::io::Write;
                    match std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)
                    {
                        Ok(mut f) => {
                            if f.write_all(content.as_bytes()).is_ok() {
                                Ok(Value::Bool(true))
                            } else {
                                rt_err!("append_file write failed")
                            }
                        }
                        Err(e) => rt_err!("append_file failed: {}", e),
                    }
                }
                _ => rt_err!("append_file requires (path, content) strings"),
            },
            "sys::cwd" => match std::env::current_dir() {
                Ok(path) => Ok(Value::Str(path.to_string_lossy().into_owned())),
                Err(e) => rt_err!("sys::cwd failed: {}", e),
            },
            "sys::os" => Ok(Value::Str(std::env::consts::OS.to_string())),
            "sys::arch" => Ok(Value::Str(std::env::consts::ARCH.to_string())),
            "sys::temp_dir" => Ok(Value::Str(
                std::env::temp_dir().to_string_lossy().into_owned(),
            )),
            "sys::set_cwd" => {
                if let Some(Value::Str(path)) = args.first() {
                    match std::env::set_current_dir(path) {
                        Ok(_) => Ok(Value::Bool(true)),
                        Err(e) => rt_err!("sys::set_cwd failed: {}", e),
                    }
                } else {
                    rt_err!("sys::set_cwd requires a path string")
                }
            }
            "sys::mkdir" => {
                if let Some(Value::Str(path)) = args.first() {
                    match std::fs::create_dir_all(path) {
                        Ok(_) => Ok(Value::Bool(true)),
                        Err(e) => rt_err!("sys::mkdir failed: {}", e),
                    }
                } else {
                    rt_err!("sys::mkdir requires a path string")
                }
            }
            "sys::rmdir" => {
                if let Some(Value::Str(path)) = args.first() {
                    match std::fs::remove_dir_all(path) {
                        Ok(_) => Ok(Value::Bool(true)),
                        Err(e) => rt_err!("sys::rmdir failed: {}", e),
                    }
                } else {
                    rt_err!("sys::rmdir requires a path string")
                }
            }
            "sys::remove_path" => {
                if let Some(Value::Str(path)) = args.first() {
                    let p = std::path::Path::new(path);
                    if !p.exists() {
                        Ok(Value::Bool(false))
                    } else if p.is_dir() {
                        match std::fs::remove_dir_all(p) {
                            Ok(_) => Ok(Value::Bool(true)),
                            Err(e) => rt_err!("sys::remove_path failed: {}", e),
                        }
                    } else {
                        match std::fs::remove_file(p) {
                            Ok(_) => Ok(Value::Bool(true)),
                            Err(e) => rt_err!("sys::remove_path failed: {}", e),
                        }
                    }
                } else {
                    rt_err!("sys::remove_path requires a path string")
                }
            }
            "sys::list_dir" => {
                if let Some(Value::Str(path)) = args.first() {
                    match std::fs::read_dir(path) {
                        Ok(entries) => {
                            let mut names = vec![];
                            for entry in entries {
                                let entry = entry.map_err(|e| {
                                    RuntimeError::new(format!("sys::list_dir failed: {}", e))
                                })?;
                                names.push(Value::Str(
                                    entry.file_name().to_string_lossy().into_owned(),
                                ));
                            }
                            names.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
                            Ok(Value::Array(Arc::new(Mutex::new(names))))
                        }
                        Err(e) => rt_err!("sys::list_dir failed: {}", e),
                    }
                } else {
                    rt_err!("sys::list_dir requires a path string")
                }
            }
            "sys::read_bytes" => {
                if let Some(Value::Str(path)) = args.first() {
                    match std::fs::read(path) {
                        Ok(bytes) => {
                            let values = bytes.into_iter().map(|b| Value::U8(b)).collect();
                            Ok(Value::Array(Arc::new(Mutex::new(values))))
                        }
                        Err(e) => rt_err!("sys::read_bytes failed: {}", e),
                    }
                } else {
                    rt_err!("sys::read_bytes requires a path string")
                }
            }
            "sys::copy" => match (args.get(0), args.get(1)) {
                (Some(Value::Str(from)), Some(Value::Str(to))) => match std::fs::copy(from, to) {
                    Ok(n) => Ok(Value::I64(n as i64)),
                    Err(e) => rt_err!("sys::copy failed: {}", e),
                },
                _ => rt_err!("sys::copy requires (from_path, to_path) strings"),
            },
            "sys::rename" => match (args.get(0), args.get(1)) {
                (Some(Value::Str(from)), Some(Value::Str(to))) => match std::fs::rename(from, to) {
                    Ok(_) => Ok(Value::Bool(true)),
                    Err(e) => rt_err!("sys::rename failed: {}", e),
                },
                _ => rt_err!("sys::rename requires (from_path, to_path) strings"),
            },
            "sys::metadata" => {
                if let Some(Value::Str(path)) = args.first() {
                    match std::fs::metadata(path) {
                        Ok(md) => {
                            let mut out = FxHashMap::default();
                            out.insert("is_file".into(), Value::Bool(md.is_file()));
                            out.insert("is_dir".into(), Value::Bool(md.is_dir()));
                            out.insert("len".into(), Value::I64(md.len() as i64));
                            out.insert("readonly".into(), Value::Bool(md.permissions().readonly()));
                            let modified = md
                                .modified()
                                .ok()
                                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                                .map(|d| d.as_secs() as i64)
                                .unwrap_or(0);
                            out.insert("modified_unix".into(), Value::I64(modified));
                            Ok(Value::HashMap(Arc::new(Mutex::new(out))))
                        }
                        Err(e) => rt_err!("sys::metadata failed: {}", e),
                    }
                } else {
                    rt_err!("sys::metadata requires a path string")
                }
            }
            "sys::write_bytes" => match (args.get(0), args.get(1)) {
                (Some(Value::Str(path)), Some(Value::Array(data))) => {
                    let bytes: Vec<u8> = data
                        .lock()
                        .unwrap()
                        .iter()
                        .map(|v| match v {
                            Value::U8(x) => Ok(*x),
                            Value::I32(x) if *x >= 0 && *x <= 255 => Ok(*x as u8),
                            _ => Err(RuntimeError::new(
                                "sys::write_bytes data must contain only byte values",
                            )),
                        })
                        .collect::<Result<Vec<u8>, RuntimeError>>()?;
                    match std::fs::write(path, bytes) {
                        Ok(_) => Ok(Value::Bool(true)),
                        Err(e) => rt_err!("sys::write_bytes failed: {}", e),
                    }
                }
                _ => rt_err!("sys::write_bytes requires (path, byte_array)"),
            },
            "sys::env_get" => {
                if let Some(Value::Str(key)) = args.first() {
                    match std::env::var(key) {
                        Ok(v) => Ok(Value::Some(Box::new(Value::Str(v)))),
                        Err(std::env::VarError::NotPresent) => Ok(Value::None),
                        Err(e) => rt_err!("sys::env_get failed: {}", e),
                    }
                } else {
                    rt_err!("sys::env_get requires a key string")
                }
            }
            "sys::env_set" => match (args.get(0), args.get(1)) {
                (Some(Value::Str(key)), Some(Value::Str(val))) => {
                    std::env::set_var(key, val);
                    Ok(Value::Bool(true))
                }
                _ => rt_err!("sys::env_set requires (key, value) strings"),
            },
            "sys::env_remove" => {
                if let Some(Value::Str(key)) = args.first() {
                    std::env::remove_var(key);
                    Ok(Value::Bool(true))
                } else {
                    rt_err!("sys::env_remove requires a key string")
                }
            }
            "sys::process_id" => Ok(Value::I64(std::process::id() as i64)),
            "sys::sleep_ms" => {
                if let Some(ms) = args.first().and_then(|v| v.as_i64()) {
                    if ms < 0 {
                        rt_err!("sys::sleep_ms requires a non-negative integer")
                    } else {
                        std::thread::sleep(std::time::Duration::from_millis(ms as u64));
                        Ok(Value::Bool(true))
                    }
                } else {
                    rt_err!("sys::sleep_ms requires an integer")
                }
            }
            "sys::exec" => {
                if let Some(Value::Str(command)) = args.first() {
                    let output = std::process::Command::new("sh")
                        .arg("-c")
                        .arg(command)
                        .output();
                    match output {
                        Ok(out) => {
                            let mut result = FxHashMap::default();
                            result.insert("ok".into(), Value::Bool(out.status.success()));
                            result.insert(
                                "code".into(),
                                Value::I64(out.status.code().unwrap_or(-1) as i64),
                            );
                            result.insert(
                                "stdout".into(),
                                Value::Str(String::from_utf8_lossy(&out.stdout).into_owned()),
                            );
                            result.insert(
                                "stderr".into(),
                                Value::Str(String::from_utf8_lossy(&out.stderr).into_owned()),
                            );
                            Ok(Value::HashMap(Arc::new(Mutex::new(result))))
                        }
                        Err(e) => rt_err!("sys::exec failed: {}", e),
                    }
                } else {
                    rt_err!("sys::exec requires a command string")
                }
            }
            "sys::exec_argv" => match (args.get(0), args.get(1)) {
                (Some(Value::Str(program)), Some(Value::Array(argv))) => {
                    let arg_values = argv.lock().unwrap();
                    let mut parsed = Vec::with_capacity(arg_values.len());
                    for v in arg_values.iter() {
                        if let Value::Str(s) = v {
                            parsed.push(s.clone());
                        } else {
                            return rt_err!("sys::exec_argv args array must contain only strings");
                        }
                    }
                    let output = std::process::Command::new(program).args(&parsed).output();
                    match output {
                        Ok(out) => {
                            let mut result = FxHashMap::default();
                            result.insert("ok".into(), Value::Bool(out.status.success()));
                            result.insert(
                                "code".into(),
                                Value::I64(out.status.code().unwrap_or(-1) as i64),
                            );
                            result.insert(
                                "stdout".into(),
                                Value::Str(String::from_utf8_lossy(&out.stdout).into_owned()),
                            );
                            result.insert(
                                "stderr".into(),
                                Value::Str(String::from_utf8_lossy(&out.stderr).into_owned()),
                            );
                            Ok(Value::HashMap(Arc::new(Mutex::new(result))))
                        }
                        Err(e) => rt_err!("sys::exec_argv failed: {}", e),
                    }
                }
                _ => rt_err!("sys::exec_argv requires (program, args_array)"),
            },
            "sys::exec_argv_in" => match (args.get(0), args.get(1), args.get(2)) {
                (Some(Value::Str(program)), Some(Value::Array(argv)), Some(Value::Str(cwd))) => {
                    let arg_values = argv.lock().unwrap();
                    let mut parsed = Vec::with_capacity(arg_values.len());
                    for v in arg_values.iter() {
                        if let Value::Str(s) = v {
                            parsed.push(s.clone());
                        } else {
                            return rt_err!(
                                "sys::exec_argv_in args array must contain only strings"
                            );
                        }
                    }
                    let output = std::process::Command::new(program)
                        .args(&parsed)
                        .current_dir(cwd)
                        .output();
                    match output {
                        Ok(out) => {
                            let mut result = FxHashMap::default();
                            result.insert("ok".into(), Value::Bool(out.status.success()));
                            result.insert(
                                "code".into(),
                                Value::I64(out.status.code().unwrap_or(-1) as i64),
                            );
                            result.insert(
                                "stdout".into(),
                                Value::Str(String::from_utf8_lossy(&out.stdout).into_owned()),
                            );
                            result.insert(
                                "stderr".into(),
                                Value::Str(String::from_utf8_lossy(&out.stderr).into_owned()),
                            );
                            Ok(Value::HashMap(Arc::new(Mutex::new(result))))
                        }
                        Err(e) => rt_err!("sys::exec_argv_in failed: {}", e),
                    }
                }
                _ => rt_err!("sys::exec_argv_in requires (program, args_array, cwd)"),
            },

            // ── Physics functions ──────────────────────────────────────────────
            "physics::world_new" => Ok(Value::I64(1)),
            "physics::create_body" => {
                if let (Some(mass), Some(x), Some(y), Some(z)) = (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                    args.get(3).and_then(|v| v.as_f64()),
                ) {
                    let mut world = self.physics_world.as_ref().unwrap().lock().unwrap();
                    let body_id = world.create_rigid_body(
                        mass as f32,
                        PhysicsShape::Sphere { radius: 0.5 },
                        [x as f32, y as f32, z as f32],
                    );
                    Ok(Value::I64(body_id as i64))
                } else {
                    rt_err!("physics::create_body requires (mass, x, y, z) numbers")
                }
            }
            "physics::set_velocity" => {
                if let (Some(body_id), Some(vx), Some(vy), Some(vz)) = (
                    args.get(0).and_then(|v| v.as_i64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                    args.get(3).and_then(|v| v.as_f64()),
                ) {
                    let mut world = self.physics_world.as_ref().unwrap().lock().unwrap();
                    world.set_velocity(body_id as u32, vx as f32, vy as f32, vz as f32);
                    Ok(Value::Bool(true))
                } else {
                    rt_err!("physics::set_velocity requires (body_id, vx, vy, vz)")
                }
            }
            "physics::get_position" => {
                if let Some(body_id) = args.first().and_then(|v| v.as_i64()) {
                    let world = self.physics_world.as_ref().unwrap().lock().unwrap();
                    if let Some([x, y, z]) = world.get_position(body_id as u32) {
                        Ok(Value::Vec3([x, y, z]))
                    } else {
                        rt_err!("physics::get_position: body not found")
                    }
                } else {
                    rt_err!("physics::get_position requires body_id")
                }
            }
            "physics::get_velocity" => {
                if let Some(body_id) = args.first().and_then(|v| v.as_i64()) {
                    let world = self.physics_world.as_ref().unwrap().lock().unwrap();
                    if let Some([vx, vy, vz]) = world.get_velocity(body_id as u32) {
                        Ok(Value::Vec3([vx, vy, vz]))
                    } else {
                        rt_err!("physics::get_velocity: body not found")
                    }
                } else {
                    rt_err!("physics::get_velocity requires body_id")
                }
            }
            "physics::step" => {
                if let Some(dt) = args.first().and_then(|v| v.as_f64()) {
                    let mut world = self.physics_world.as_ref().unwrap().lock().unwrap();
                    world.step(dt as f32);
                    Ok(Value::Bool(true))
                } else {
                    rt_err!("physics::step requires dt (number)")
                }
            }
            "physics::apply_force" => {
                if let (Some(body_id), Some(fx), Some(fy), Some(fz)) = (
                    args.get(0).and_then(|v| v.as_i64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                    args.get(3).and_then(|v| v.as_f64()),
                ) {
                    let _world = self.physics_world.as_ref().unwrap().lock().unwrap();
                    // Placeholder: force integration API is not currently exposed.
                    let _ = (body_id, fx, fy, fz);
                    Ok(Value::Bool(false))
                } else {
                    rt_err!("physics::apply_force requires (body_id, fx, fy, fz)")
                }
            }

            // ── Graphics functions ─────────────────────────────────────────────
            "graphics::set_camera" => {
                if let (Some(x), Some(y), Some(z)) = (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                ) {
                    let mut render = self.render_state.as_ref().unwrap().lock().unwrap();
                    render.update_camera_position(x as f32, y as f32, z as f32);
                    Ok(Value::Bool(true))
                } else {
                    rt_err!("graphics::set_camera requires (x, y, z)")
                }
            }
            "graphics::set_chunked_grid_cell" => {
                if let (Some(map_id), Some(x), Some(y), Some(object_id)) = (
                    args.get(0).and_then(|v| v.as_i64()),
                    args.get(1).and_then(|v| v.as_i64()),
                    args.get(2).and_then(|v| v.as_i64()),
                    args.get(3).and_then(|v| v.as_i64()),
                ) {
                    let mut render = self.render_state.as_ref().unwrap().lock().unwrap();
                    match render.set_chunked_grid_cell(
                        map_id as u32,
                        x as usize,
                        y as usize,
                        object_id as u32,
                    ) {
                        Ok(()) => Ok(Value::Bool(true)),
                        Err(e) => rt_err!("graphics::set_chunked_grid_cell failed: {}", e),
                    }
                } else {
                    rt_err!("graphics::set_chunked_grid_cell requires (map_id, x, y, object_id)")
                }
            }
            "graphics::create_material" => {
                if let (Some(r), Some(g), Some(b), Some(a)) = (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                    args.get(3).and_then(|v| v.as_f64()),
                ) {
                    let mut render = self.render_state.as_ref().unwrap().lock().unwrap();
                    let material_id =
                        render.create_material([r as f32, g as f32, b as f32, a as f32]);
                    Ok(Value::I64(material_id as i64))
                } else {
                    rt_err!("graphics::create_material requires (r, g, b, a)")
                }
            }
            "graphics::create_sprite" => {
                if let (Some(Value::Str(name)), Some(w), Some(h)) = (
                    args.get(0),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                ) {
                    let mut render = self.render_state.as_ref().unwrap().lock().unwrap();
                    let sprite_id =
                        render.create_sprite(name.clone(), w.max(0.0) as f32, h.max(0.0) as f32);
                    Ok(Value::I64(sprite_id as i64))
                } else {
                    rt_err!(
                        "graphics::create_sprite(name, width, height) requires string + numbers"
                    )
                }
            }
            "graphics::render_mesh" => {
                if let (Some(mesh_id), Some(mat_id)) = (
                    args.get(0).and_then(|v| v.as_i64()),
                    args.get(1).and_then(|v| v.as_i64()),
                ) {
                    let render = self.render_state.as_ref().unwrap().lock().unwrap();
                    let _ = (render, mesh_id, mat_id);
                    Ok(Value::Bool(false))
                } else {
                    rt_err!("graphics::render_mesh requires (mesh_id, material_id)")
                }
            }
            "graphics::clear" => Ok(Value::Bool(true)),

            // ── Render command-buffer API (AoT/host friendly) ───────────────
            "render::begin_frame" => {
                if let (Some(w), Some(h)) = (
                    args.get(0).and_then(|v| v.as_i64()),
                    args.get(1).and_then(|v| v.as_i64()),
                ) {
                    let mut render = self.render_state.as_ref().unwrap().lock().unwrap();
                    render.begin_frame(w.max(1) as u32, h.max(1) as u32);
                    Ok(Value::Bool(true))
                } else {
                    rt_err!("render::begin_frame(width, height) requires integer width/height")
                }
            }
            "render::clear" => {
                let r = args.get(0).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                let g = args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                let b = args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                let a = args.get(3).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                let mut render = self.render_state.as_ref().unwrap().lock().unwrap();
                render.queue_clear([r, g, b, a]);
                Ok(Value::Bool(true))
            }
            "render::rect" => {
                if let (Some(x), Some(y), Some(w), Some(h)) = (
                    args.get(0).and_then(|v| v.as_f64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                    args.get(3).and_then(|v| v.as_f64()),
                ) {
                    let r = args.get(4).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                    let g = args.get(5).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                    let b = args.get(6).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                    let a = args.get(7).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                    let layer = args.get(8).and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                    let mut render = self.render_state.as_ref().unwrap().lock().unwrap();
                    render.queue_rect(
                        x as f32,
                        y as f32,
                        w.max(0.0) as f32,
                        h.max(0.0) as f32,
                        [r, g, b, a],
                        layer,
                    );
                    Ok(Value::Bool(true))
                } else {
                    rt_err!("render::rect(x, y, w, h, r?, g?, b?, a?, layer?) requires at least x/y/w/h")
                }
            }
            "render::sprite" => {
                if let (Some(sprite_id), Some(x), Some(y), Some(w), Some(h)) = (
                    args.get(0).and_then(|v| v.as_i64()),
                    args.get(1).and_then(|v| v.as_f64()),
                    args.get(2).and_then(|v| v.as_f64()),
                    args.get(3).and_then(|v| v.as_f64()),
                    args.get(4).and_then(|v| v.as_f64()),
                ) {
                    let rotation = args.get(5).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                    let layer = args.get(6).and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                    let mut render = self.render_state.as_ref().unwrap().lock().unwrap();
                    match render.queue_sprite(
                        sprite_id as u32,
                        x as f32,
                        y as f32,
                        w.max(0.0) as f32,
                        h.max(0.0) as f32,
                        rotation,
                        layer,
                    ) {
                        Ok(()) => Ok(Value::Bool(true)),
                        Err(e) => rt_err!("render::sprite failed: {}", e),
                    }
                } else {
                    rt_err!("render::sprite(sprite_id, x, y, w, h, rotation_deg?, layer?) requires sprite_id/x/y/w/h")
                }
            }
            "render::flush" => {
                let mut render = self.render_state.as_ref().unwrap().lock().unwrap();
                let mut out = Vec::with_capacity(render.command_buffer.len());
                for cmd in render.command_buffer.drain(..) {
                    let mut entry = FxHashMap::default();
                    match cmd {
                        RenderCommand::Clear { color } => {
                            entry.insert("kind".into(), Value::Str("clear".into()));
                            entry.insert(
                                "color".into(),
                                Value::Array(Arc::new(Mutex::new(vec![
                                    Value::F32(color[0]),
                                    Value::F32(color[1]),
                                    Value::F32(color[2]),
                                    Value::F32(color[3]),
                                ]))),
                            );
                        }
                        RenderCommand::Rect {
                            x,
                            y,
                            w,
                            h,
                            color,
                            layer,
                        } => {
                            entry.insert("kind".into(), Value::Str("rect".into()));
                            entry.insert("x".into(), Value::F32(x));
                            entry.insert("y".into(), Value::F32(y));
                            entry.insert("w".into(), Value::F32(w));
                            entry.insert("h".into(), Value::F32(h));
                            entry.insert("layer".into(), Value::I64(layer as i64));
                            entry.insert(
                                "color".into(),
                                Value::Array(Arc::new(Mutex::new(vec![
                                    Value::F32(color[0]),
                                    Value::F32(color[1]),
                                    Value::F32(color[2]),
                                    Value::F32(color[3]),
                                ]))),
                            );
                        }
                        RenderCommand::Sprite {
                            sprite_id,
                            x,
                            y,
                            w,
                            h,
                            rotation_deg,
                            layer,
                        } => {
                            entry.insert("kind".into(), Value::Str("sprite".into()));
                            entry.insert("sprite_id".into(), Value::I64(sprite_id as i64));
                            entry.insert("x".into(), Value::F32(x));
                            entry.insert("y".into(), Value::F32(y));
                            entry.insert("w".into(), Value::F32(w));
                            entry.insert("h".into(), Value::F32(h));
                            entry.insert("rotation_deg".into(), Value::F32(rotation_deg));
                            entry.insert("layer".into(), Value::I64(layer as i64));
                        }
                    }
                    out.push(Value::HashMap(Arc::new(Mutex::new(entry))));
                }
                Ok(Value::Array(Arc::new(Mutex::new(out))))
            }
            "render::stats" => {
                let render = self.render_state.as_ref().unwrap().lock().unwrap();
                let mut map = FxHashMap::default();
                map.insert("width".into(), Value::I64(render.width as i64));
                map.insert("height".into(), Value::I64(render.height as i64));
                map.insert(
                    "queued_commands".into(),
                    Value::I64(render.command_buffer.len() as i64),
                );
                map.insert("sprites".into(), Value::I64(render.sprites.len() as i64));
                map.insert(
                    "materials".into(),
                    Value::I64(render.materials.len() as i64),
                );
                Ok(Value::HashMap(Arc::new(Mutex::new(map))))
            }

            // ── Input functions ───────────────────────────────────────────────────
            "input::is_key_pressed" => {
                if let Some(Value::Str(key)) = args.first() {
                    let _input = self.input_state.as_ref().unwrap().lock().unwrap();
                    let _ = key;
                    let pressed = false;
                    Ok(Value::Bool(pressed))
                } else {
                    rt_err!("input::is_key_pressed requires a key name string")
                }
            }
            "input::get_mouse_position" => Ok(Value::Vec3([0.0, 0.0, 0.0])),
            "input::get_mouse_scroll" => Ok(Value::F32(0.0)),
            "input::get_gamepad_axis" => {
                if let (Some(gamepad_id), Some(axis_id)) = (
                    args.get(0).and_then(|v| v.as_i64()),
                    args.get(1).and_then(|v| v.as_i64()),
                ) {
                    let _ = (gamepad_id, axis_id);
                    Ok(Value::F32(0.0))
                } else {
                    rt_err!("input::get_gamepad_axis requires (gamepad_id, axis_id)")
                }
            }
            "input::get_gamepad_button" => {
                if let (Some(gamepad_id), Some(btn_id)) = (
                    args.get(0).and_then(|v| v.as_i64()),
                    args.get(1).and_then(|v| v.as_i64()),
                ) {
                    let _ = (gamepad_id, btn_id);
                    Ok(Value::Bool(false))
                } else {
                    rt_err!("input::get_gamepad_button requires (gamepad_id, button_id)")
                }
            }

            // ── Autodiff functions ─────────────────────────────────────────────────
            "autodiff::enable" => {
                if let Some(Value::Tensor(t)) = args.first() {
                    if let Some(graph) = &self.computation_graph {
                        let tensor = t.read().unwrap().clone();
                        let ml_tensor = crate::ml_engine::Tensor {
                            shape: tensor.shape.clone(),
                            data: tensor.cpu_data().to_vec(),
                        };
                        let node_id = graph.lock().unwrap().add_input(ml_tensor);
                        Ok(Value::I64(node_id as i64))
                    } else {
                        rt_err!("autodiff graph unavailable")
                    }
                } else {
                    rt_err!("autodiff::enable requires a tensor")
                }
            }
            "autodiff::backward" => {
                if let Some(node_id) = args.first().and_then(|v| v.as_i64()) {
                    if let Some(graph) = &self.computation_graph {
                        graph.lock().unwrap().backward(node_id as u64);
                        Ok(Value::Bool(true))
                    } else {
                        rt_err!("autodiff graph unavailable")
                    }
                } else {
                    rt_err!("autodiff::backward requires node_id")
                }
            }
            "autodiff::get_gradient" => {
                if let Some(node_id) = args.first().and_then(|v| v.as_i64()) {
                    if let Some(graph) = &self.computation_graph {
                        let graph = graph.lock().unwrap();
                        let grad = graph
                            .nodes
                            .get(&(node_id as u64))
                            .and_then(|node| node.gradient.clone())
                            .unwrap_or_else(|| crate::ml_engine::Tensor::zeros(vec![1]));
                        let interp_grad = Tensor::from_data(grad.shape, grad.data);
                        Ok(Value::Tensor(Arc::new(RwLock::new(interp_grad))))
                    } else {
                        rt_err!("autodiff graph unavailable")
                    }
                } else {
                    rt_err!("autodiff::get_gradient requires node_id")
                }
            }

            // ── Optimizer functions ────────────────────────────────────────────────
            "optimizer::create" => {
                if let (Some(Value::Str(opt_type)), Some(lr)) =
                    (args.get(0), args.get(1).and_then(|v| v.as_f64()))
                {
                    let _ = (opt_type, lr);
                    Ok(Value::I64(0)) // Return optimizer handle
                } else {
                    rt_err!("optimizer::create requires (optimizer_type, learning_rate)")
                }
            }
            "optimizer::step" => {
                // Simplified: actual implementation would update weights based on gradients
                Ok(Value::Bool(true))
            }

            // ── Loss functions ─────────────────────────────────────────────────────
            "loss::mse" => {
                if let (Some(Value::Tensor(pred)), Some(Value::Tensor(target))) =
                    (args.get(0), args.get(1))
                {
                    let pred_t = pred.read().unwrap();
                    let target_t = target.read().unwrap();

                    let pred_data = pred_t.cpu_data();
                    let target_data = target_t.cpu_data();

                    let mut mse = 0.0f32;
                    for (p, t) in pred_data.iter().zip(target_data.iter()) {
                        let diff = p - t;
                        mse += diff * diff;
                    }
                    mse /= pred_data.len() as f32;

                    Ok(Value::F32(mse))
                } else {
                    rt_err!("loss::mse requires (predictions, targets) tensors")
                }
            }
            "loss::cross_entropy" => {
                if let (Some(Value::Tensor(logits)), Some(Value::Tensor(targets))) =
                    (args.get(0), args.get(1))
                {
                    let logits_t = logits.read().unwrap();
                    let targets_t = targets.read().unwrap();

                    let logits_data = logits_t.cpu_data();
                    let targets_data = targets_t.cpu_data();

                    let mut ce = 0.0f32;
                    let eps = 1e-7f32;
                    for (logit, target) in logits_data.iter().zip(targets_data.iter()) {
                        let exp_logit = logit.exp();
                        let softmax = exp_logit / (1.0 + exp_logit);
                        ce += -target * softmax.ln().max(eps.ln());
                    }
                    ce /= logits_data.len() as f32;

                    Ok(Value::F32(ce))
                } else {
                    rt_err!("loss::cross_entropy requires (logits, targets) tensors")
                }
            }

            // ── Metrics functions ──────────────────────────────────────────────────
            "metrics::accuracy" => {
                if let (Some(Value::Tensor(pred)), Some(Value::Tensor(target))) =
                    (args.get(0), args.get(1))
                {
                    let pred_t = pred.read().unwrap();
                    let target_t = target.read().unwrap();

                    let pred_data = pred_t.cpu_data();
                    let target_data = target_t.cpu_data();

                    let mut correct = 0;
                    for (p, t) in pred_data.iter().zip(target_data.iter()) {
                        if (p.round() - t.round()).abs() < 0.01 {
                            correct += 1;
                        }
                    }

                    let accuracy = correct as f32 / pred_data.len() as f32;
                    Ok(Value::F32(accuracy))
                } else {
                    rt_err!("metrics::accuracy requires (predictions, targets) tensors")
                }
            }
            "metrics::precision" => {
                if let (Some(Value::Tensor(pred)), Some(Value::Tensor(target))) =
                    (args.get(0), args.get(1))
                {
                    let pred_t = pred.read().unwrap();
                    let target_t = target.read().unwrap();

                    let pred_data = pred_t.cpu_data();
                    let target_data = target_t.cpu_data();

                    let mut tp = 0;
                    let mut fp = 0;
                    for (p, t) in pred_data.iter().zip(target_data.iter()) {
                        let pred_pos = p.round() > 0.5;
                        let target_pos = t.round() > 0.5;
                        if pred_pos && target_pos {
                            tp += 1;
                        }
                        if pred_pos && !target_pos {
                            fp += 1;
                        }
                    }

                    let precision = if tp + fp > 0 {
                        tp as f32 / (tp + fp) as f32
                    } else {
                        0.0
                    };
                    Ok(Value::F32(precision))
                } else {
                    rt_err!("metrics::precision requires (predictions, targets) tensors")
                }
            }
            "metrics::recall" => {
                if let (Some(Value::Tensor(pred)), Some(Value::Tensor(target))) =
                    (args.get(0), args.get(1))
                {
                    let pred_t = pred.read().unwrap();
                    let target_t = target.read().unwrap();

                    let pred_data = pred_t.cpu_data();
                    let target_data = target_t.cpu_data();

                    let mut tp = 0;
                    let mut fn_count = 0;
                    for (p, t) in pred_data.iter().zip(target_data.iter()) {
                        let pred_pos = p.round() > 0.5;
                        let target_pos = t.round() > 0.5;
                        if pred_pos && target_pos {
                            tp += 1;
                        }
                        if !pred_pos && target_pos {
                            fn_count += 1;
                        }
                    }

                    let recall = if tp + fn_count > 0 {
                        tp as f32 / (tp + fn_count) as f32
                    } else {
                        0.0
                    };
                    Ok(Value::F32(recall))
                } else {
                    rt_err!("metrics::recall requires (predictions, targets) tensors")
                }
            }
            "metrics::f1_score" => {
                if let (Some(Value::Tensor(pred)), Some(Value::Tensor(target))) =
                    (args.get(0), args.get(1))
                {
                    // Compute precision and recall
                    let pred_t = pred.read().unwrap();
                    let target_t = target.read().unwrap();

                    let pred_data = pred_t.cpu_data();
                    let target_data = target_t.cpu_data();

                    let mut tp = 0;
                    let mut fp = 0;
                    let mut fn_count = 0;
                    for (p, t) in pred_data.iter().zip(target_data.iter()) {
                        let pred_pos = p.round() > 0.5;
                        let target_pos = t.round() > 0.5;
                        if pred_pos && target_pos {
                            tp += 1;
                        }
                        if pred_pos && !target_pos {
                            fp += 1;
                        }
                        if !pred_pos && target_pos {
                            fn_count += 1;
                        }
                    }

                    let precision = if tp + fp > 0 {
                        tp as f32 / (tp + fp) as f32
                    } else {
                        0.0
                    };
                    let recall = if tp + fn_count > 0 {
                        tp as f32 / (tp + fn_count) as f32
                    } else {
                        0.0
                    };
                    let f1 = if precision + recall > 0.0 {
                        2.0 * (precision * recall) / (precision + recall)
                    } else {
                        0.0
                    };

                    Ok(Value::F32(f1))
                } else {
                    rt_err!("metrics::f1_score requires (predictions, targets) tensors")
                }
            }

            // ── Std library dispatch ────────────────────────────────────────────
            name if name.starts_with("math::")
                || name.starts_with("geom::")
                || name.starts_with("random::")
                || name.starts_with("noise::")
                || name.starts_with("spatial::")
                || name.starts_with("path::")
                || name.starts_with("ai::")
                || name.starts_with("net::")
                || name.starts_with("alloc::")
                || name.starts_with("collections::") =>
            {
                crate::jules_std::dispatch(name, &args).unwrap_or_else(|| {
                    Err(RuntimeError {
                        message: format!("unknown std function: {}", name),
                        span: None,
                    })
                })
            }

            // Not a built-in
            _ => Err(RuntimeError {
                message: format!("unknown function: {}", name),
                span: None,
            }),
        }
    }

    // ── Built-in method dispatch ───────────────────────────────────────────

    fn eval_method(
        &mut self,
        recv: Value,
        method: &str,
        args: Vec<Value>,
        _env: &mut Env,
    ) -> Result<Value, RuntimeError> {
        match (&recv, method) {
            // ── Tensor methods ─────────────────────────────────────────────────
            (Value::Tensor(_), "transpose") => {
                if let Value::Tensor(t) = recv {
                    let out = t.read().unwrap().transpose()?;
                    Ok(Value::Tensor(Arc::new(RwLock::new(out))))
                } else {
                    unreachable!()
                }
            }
            (Value::Tensor(_), "sum") => {
                if let Value::Tensor(t) = recv {
                    Ok(Value::F32(t.read().unwrap().sum_all()))
                } else {
                    unreachable!()
                }
            }
            (Value::Tensor(_), "mean") => {
                if let Value::Tensor(t) = recv {
                    let tt = t.read().unwrap();
                    Ok(Value::F32(tt.sum_all() / tt.numel() as f32))
                } else {
                    unreachable!()
                }
            }
            (Value::Tensor(_), "relu") => {
                if let Value::Tensor(t) = recv {
                    let out = t.read().unwrap().apply_activation(&Activation::Relu);
                    Ok(Value::Tensor(Arc::new(RwLock::new(out))))
                } else {
                    unreachable!()
                }
            }
            (Value::Tensor(_), "softmax") => {
                if let Value::Tensor(t) = recv {
                    let out = t.read().unwrap().apply_activation(&Activation::Softmax);
                    Ok(Value::Tensor(Arc::new(RwLock::new(out))))
                } else {
                    unreachable!()
                }
            }
            (Value::Tensor(_), "reshape") => {
                if let Value::Tensor(t) = recv {
                    let new_shape: Vec<usize> = args
                        .iter()
                        .filter_map(|v| v.as_i64().map(|i| i as usize))
                        .collect();
                    let data = t.read().unwrap().cpu_data().to_vec();
                    Ok(Value::Tensor(Arc::new(RwLock::new(Tensor::from_data(
                        new_shape, data,
                    )))))
                } else {
                    unreachable!()
                }
            }
            // ── Vec methods ────────────────────────────────────────────────────
            (Value::Vec3(v), "dot") => {
                if let Some(Value::Vec3(r)) = args.first() {
                    Ok(Value::F32(v[0] * r[0] + v[1] * r[1] + v[2] * r[2]))
                } else {
                    rt_err!("dot() expects vec3")
                }
            }
            (Value::Vec3(v), "cross") => {
                if let Some(Value::Vec3(r)) = args.first() {
                    Ok(Value::Vec3([
                        v[1] * r[2] - v[2] * r[1],
                        v[2] * r[0] - v[0] * r[2],
                        v[0] * r[1] - v[1] * r[0],
                    ]))
                } else {
                    rt_err!("cross() expects vec3")
                }
            }
            (Value::Vec3(v), "normalize") => {
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                if len < 1e-8 {
                    return Ok(Value::Vec3(*v));
                }
                Ok(Value::Vec3([v[0] / len, v[1] / len, v[2] / len]))
            }
            (Value::Vec3(v), "length" | "magnitude") => {
                Ok(Value::F32((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()))
            }
            // ── Model forward pass ─────────────────────────────────────────────
            (Value::Model(_), "forward") => {
                if let Value::Model(m) = recv {
                    if let Some(Value::Tensor(x)) = args.into_iter().next() {
                        let x_owned = Arc::try_unwrap(x)
                            .unwrap_or_else(|a| {
                                let t = a.read().unwrap().clone();
                                RwLock::new(t)
                            })
                            .into_inner()
                            .unwrap();
                        let out = m.lock().unwrap().forward(x_owned)?;
                        Ok(Value::Tensor(Arc::new(RwLock::new(out))))
                    } else {
                        rt_err!("forward() expects tensor")
                    }
                } else {
                    unreachable!()
                }
            }
            // ── DataLoader / data pipeline methods ─────────────────────────────
            (Value::DataLoader(_), "next") => {
                if let Value::DataLoader(dl) = recv {
                    let mut dl = dl.lock().unwrap();
                    if let Some(batch) = dl.next_batch() {
                        Ok(Value::Some(Box::new(batch)))
                    } else {
                        Ok(Value::None)
                    }
                } else {
                    unreachable!()
                }
            }
            (Value::DataLoader(_), "has_next") => {
                if let Value::DataLoader(dl) = recv {
                    Ok(Value::Bool(dl.lock().unwrap().has_next()))
                } else {
                    unreachable!()
                }
            }
            (Value::DataLoader(_), "reset") => {
                if let Value::DataLoader(dl) = recv {
                    dl.lock().unwrap().reset();
                    Ok(Value::Unit)
                } else {
                    unreachable!()
                }
            }
            (Value::DataLoader(_), "shuffle") => {
                if let Value::DataLoader(dl) = recv {
                    let mut d = dl.lock().unwrap();
                    d.shuffle = true;
                    d.reset();
                    Ok(Value::Unit)
                } else {
                    unreachable!()
                }
            }
            (Value::DataLoader(_), "batch") => {
                if let Value::DataLoader(dl) = recv {
                    let size = args
                        .get(0)
                        .and_then(|v| v.as_i64())
                        .map(|i| i.max(1) as usize)
                        .unwrap_or(1);
                    let (samples, shuffle) = {
                        let dl = dl.lock().unwrap();
                        (dl.samples.clone(), dl.shuffle)
                    };
                    Ok(Value::DataLoader(Arc::new(Mutex::new(DataLoader {
                        samples,
                        batch_size: size,
                        index: 0,
                        shuffle,
                    }))))
                } else {
                    unreachable!()
                }
            }
            (Value::DataLoader(_), "repeat") => {
                if let Value::DataLoader(dl) = recv {
                    let times = args
                        .get(0)
                        .and_then(|v| v.as_i64())
                        .map(|i| i.max(0) as usize)
                        .unwrap_or(1);
                    let (samples, batch_size, shuffle) = {
                        let dl = dl.lock().unwrap();
                        (dl.samples.clone(), dl.batch_size, dl.shuffle)
                    };
                    let mut repeated = Vec::new();
                    for _ in 0..times {
                        repeated.extend_from_slice(&samples);
                    }
                    Ok(Value::DataLoader(Arc::new(Mutex::new(DataLoader {
                        samples: repeated,
                        batch_size,
                        index: 0,
                        shuffle,
                    }))))
                } else {
                    unreachable!()
                }
            }
            (Value::DataLoader(_), "collect") => {
                if let Value::DataLoader(dl) = recv {
                    let data = dl.lock().unwrap().samples.clone();
                    Ok(Value::Array(Arc::new(Mutex::new(data))))
                } else {
                    unreachable!()
                }
            }
            (Value::DataLoader(_), "map") => {
                if let Value::DataLoader(dl) = recv {
                    let func = match args.get(0) {
                        Some(Value::Fn(f)) => f.clone(),
                        _ => return rt_err!("map() requires a function argument"),
                    };
                    let (batch_size, shuffle, samples) = {
                        let dl = dl.lock().unwrap();
                        (dl.batch_size, dl.shuffle, dl.samples.clone())
                    };
                    let mut mapped = Vec::new();
                    for sample in samples {
                        let mut call_env = Env::new();
                        let mapped_val =
                            self.eval_call(Value::Fn(func.clone()), vec![sample], &mut call_env)?;
                        mapped.push(mapped_val);
                    }
                    Ok(Value::DataLoader(Arc::new(Mutex::new(DataLoader {
                        samples: mapped,
                        batch_size,
                        index: 0,
                        shuffle,
                    }))))
                } else {
                    unreachable!()
                }
            }
            (Value::DataLoader(_), "filter") => {
                if let Value::DataLoader(dl) = recv {
                    let func = match args.get(0) {
                        Some(Value::Fn(f)) => f.clone(),
                        _ => return rt_err!("filter() requires a function argument"),
                    };
                    let (batch_size, shuffle, samples) = {
                        let dl = dl.lock().unwrap();
                        (dl.batch_size, dl.shuffle, dl.samples.clone())
                    };
                    let mut filtered = Vec::new();
                    for sample in samples {
                        let mut call_env = Env::new();
                        let result = self.eval_call(
                            Value::Fn(func.clone()),
                            vec![sample.clone()],
                            &mut call_env,
                        )?;
                        if result.as_bool().unwrap_or(false) {
                            filtered.push(sample);
                        }
                    }
                    Ok(Value::DataLoader(Arc::new(Mutex::new(DataLoader {
                        samples: filtered,
                        batch_size,
                        index: 0,
                        shuffle,
                    }))))
                } else {
                    unreachable!()
                }
            }

            // ── World methods ──────────────────────────────────────────────────
            (Value::World(_), "spawn") => {
                if let Value::World(w) = recv {
                    let id = w.lock().unwrap().spawn();
                    Ok(Value::Entity(id))
                } else {
                    unreachable!()
                }
            }
            (Value::World(_), "despawn") => {
                if let Value::World(w) = recv {
                    if let Some(Value::Entity(id)) = args.first() {
                        w.lock().unwrap().despawn(*id);
                    }
                    Ok(Value::Unit)
                } else {
                    unreachable!()
                }
            }
            // ── Array / string methods ─────────────────────────────────────────
            (Value::Array(_), "len") => {
                if let Value::Array(a) = recv {
                    Ok(Value::I32(a.lock().unwrap().len() as i32))
                } else {
                    unreachable!()
                }
            }
            (Value::Array(_), "push") => {
                if let Value::Array(a) = recv {
                    if let Some(v) = args.into_iter().next() {
                        a.lock().unwrap().push(v);
                    }
                    Ok(Value::Unit)
                } else {
                    unreachable!()
                }
            }
            (Value::Array(_), "pop") => {
                if let Value::Array(a) = recv {
                    Ok(a.lock().unwrap().pop().unwrap_or(Value::Unit))
                } else {
                    unreachable!()
                }
            }
            (Value::Array(_), "clear") => {
                if let Value::Array(a) = recv {
                    a.lock().unwrap().clear();
                    Ok(Value::Unit)
                } else {
                    unreachable!()
                }
            }

            // ── HashMap methods ───────────────────────────────────────────────
            (Value::HashMap(_), "insert") => {
                if let Value::HashMap(m) = recv {
                    match (args.get(0), args.get(1)) {
                        (Some(Value::Str(k)), Some(v)) => {
                            m.lock().unwrap().insert(k.clone(), v.clone());
                            Ok(Value::Unit)
                        }
                        _ => rt_err!("insert() requires (string_key, value)"),
                    }
                } else {
                    unreachable!()
                }
            }
            (Value::HashMap(_), "get") => {
                if let Value::HashMap(m) = recv {
                    if let Some(Value::Str(k)) = args.first() {
                        let map = m.lock().unwrap();
                        Ok(map.get(k).cloned().unwrap_or(Value::None))
                    } else {
                        rt_err!("get() requires string key")
                    }
                } else {
                    unreachable!()
                }
            }
            (Value::HashMap(_), "remove") => {
                if let Value::HashMap(m) = recv {
                    if let Some(Value::Str(k)) = args.first() {
                        Ok(m.lock().unwrap().remove(k).unwrap_or(Value::None))
                    } else {
                        rt_err!("remove() requires string key")
                    }
                } else {
                    unreachable!()
                }
            }
            (Value::HashMap(_), "len") => {
                if let Value::HashMap(m) = recv {
                    Ok(Value::I32(m.lock().unwrap().len() as i32))
                } else {
                    unreachable!()
                }
            }
            (Value::HashMap(_), "clear") => {
                if let Value::HashMap(m) = recv {
                    m.lock().unwrap().clear();
                    Ok(Value::Unit)
                } else {
                    unreachable!()
                }
            }
            (Value::HashMap(_), "keys") => {
                if let Value::HashMap(m) = recv {
                    let keys: Vec<Value> = m
                        .lock()
                        .unwrap()
                        .keys()
                        .map(|k| Value::Str(k.clone()))
                        .collect();
                    Ok(Value::Array(Arc::new(Mutex::new(keys))))
                } else {
                    unreachable!()
                }
            }
            (Value::HashMap(_), "values") => {
                if let Value::HashMap(m) = recv {
                    let values: Vec<Value> = m.lock().unwrap().values().cloned().collect();
                    Ok(Value::Array(Arc::new(Mutex::new(values))))
                } else {
                    unreachable!()
                }
            }
            (Value::HashMap(_), "contains_key") => {
                if let Value::HashMap(m) = recv {
                    if let Some(Value::Str(k)) = args.first() {
                        Ok(Value::Bool(m.lock().unwrap().contains_key(k)))
                    } else {
                        rt_err!("contains_key() requires string key")
                    }
                } else {
                    unreachable!()
                }
            }
            (Value::Str(s), "len") => Ok(Value::I32(s.len() as i32)),

            // ── String methods ─────────────────────────────────────────────────
            (Value::Str(s), "to_upper") => Ok(Value::Str(s.to_uppercase())),
            (Value::Str(s), "to_lower") => Ok(Value::Str(s.to_lowercase())),
            (Value::Str(s), "trim") => Ok(Value::Str(s.trim().to_string())),
            (Value::Str(s), "trim_start") => Ok(Value::Str(s.trim_start().to_string())),
            (Value::Str(s), "trim_end") => Ok(Value::Str(s.trim_end().to_string())),
            (Value::Str(s), "chars") => {
                let chars: Vec<Value> = s.chars().map(|c| Value::Str(c.to_string())).collect();
                Ok(Value::Array(Arc::new(Mutex::new(chars))))
            }
            (Value::Str(s), "reverse") => Ok(Value::Str(s.chars().rev().collect())),
            (Value::Str(s), "starts_with") => {
                if let Some(Value::Str(prefix)) = args.first() {
                    Ok(Value::Bool(s.starts_with(prefix)))
                } else {
                    rt_err!("starts_with() requires a string argument")
                }
            }
            (Value::Str(s), "ends_with") => {
                if let Some(Value::Str(suffix)) = args.first() {
                    Ok(Value::Bool(s.ends_with(suffix)))
                } else {
                    rt_err!("ends_with() requires a string argument")
                }
            }
            (Value::Str(s), "contains") => {
                if let Some(Value::Str(needle)) = args.first() {
                    Ok(Value::Bool(s.contains(needle)))
                } else {
                    rt_err!("contains() requires a string argument")
                }
            }
            (Value::Str(s), "split") => {
                if let Some(Value::Str(delim)) = args.first() {
                    let parts: Vec<Value> = s
                        .split(delim.as_str())
                        .map(|part| Value::Str(part.to_string()))
                        .collect();
                    Ok(Value::Array(Arc::new(Mutex::new(parts))))
                } else {
                    rt_err!("split() requires a string argument")
                }
            }
            (Value::Str(s), "replace") => match (args.get(0), args.get(1)) {
                (Some(Value::Str(from)), Some(Value::Str(to))) => {
                    Ok(Value::Str(s.replace(from, to)))
                }
                _ => rt_err!("replace() requires two string arguments"),
            },

            // ── Option/Result methods ───────────────────────────────────────────
            (Value::Some(v), "unwrap") => Ok((**v).clone()),
            (Value::None, "unwrap") => rt_err!("called unwrap() on None"),
            (Value::Ok(v), "unwrap") => Ok((**v).clone()),
            (Value::Err(e), "unwrap") => rt_err!("called unwrap() on Err: {}", e),

            (Value::Some(_), "is_some") => Ok(Value::Bool(true)),
            (Value::None, "is_some") => Ok(Value::Bool(false)),
            (Value::Some(_), "is_none") => Ok(Value::Bool(false)),
            (Value::None, "is_none") => Ok(Value::Bool(true)),

            (Value::Ok(_), "is_ok") => Ok(Value::Bool(true)),
            (Value::Err(_), "is_ok") => Ok(Value::Bool(false)),
            (Value::Ok(_), "is_err") => Ok(Value::Bool(false)),
            (Value::Err(_), "is_err") => Ok(Value::Bool(true)),

            // ── Fallback ───────────────────────────────────────────────────────
            (_, method) => {
                rt_err!("no method `{method}` on `{}`", recv.type_name())
            }
        }
    }

    // =========================================================================
    // §17  CAST
    // =========================================================================

    fn eval_cast(&self, v: Value, ty: &crate::ast::Type) -> Result<Value, RuntimeError> {
        use crate::ast::{ElemType as E, Type};
        let f = v.as_f64().unwrap_or(0.0);
        match ty {
            Type::Scalar(E::F32) => Ok(Value::F32(f as f32)),
            Type::Scalar(E::F64) => Ok(Value::F64(f)),
            Type::Scalar(E::I32) => Ok(Value::I32(f as i32)),
            Type::Scalar(E::I64) => Ok(Value::I64(f as i64)),
            Type::Scalar(E::U32) => Ok(Value::U32(f as u32)),
            Type::Scalar(E::U64) => Ok(Value::U64(f as u64)),
            Type::Scalar(E::Bool) => Ok(Value::Bool(f != 0.0)),
            _ => Ok(v),
        }
    }

    // =========================================================================
    // §18  PATTERN MATCHING
    // =========================================================================

    fn pattern_matches(&self, pat: &Pattern, val: &Value) -> bool {
        match (pat, val) {
            (Pattern::Wildcard(_), _) => true,
            (Pattern::Ident { .. }, _) => true,
            (Pattern::Lit(_, lit), v) => {
                use crate::ast::LitVal;
                match (lit, v) {
                    (LitVal::Int(n), Value::I32(x)) => *n == *x as u128,
                    (LitVal::Int(n), Value::I64(x)) => *n == *x as u128,
                    (LitVal::Float(f), Value::F32(x)) => (*f as f32 - x).abs() < f32::EPSILON,
                    (LitVal::Bool(b), Value::Bool(x)) => b == x,
                    (LitVal::Str(s), Value::Str(x)) => s == x,
                    _ => false,
                }
            }
            (Pattern::Tuple { elems, .. }, Value::Tuple(vs)) => {
                elems.len() == vs.len()
                    && elems
                        .iter()
                        .zip(vs)
                        .all(|(p, v)| self.pattern_matches(p, v))
            }
            (Pattern::Or { arms, .. }, v) => arms.iter().any(|p| self.pattern_matches(p, v)),
            _ => false,
        }
    }

    fn bind_pattern(&self, pat: &Pattern, val: Value, env: &mut Env) {
        match pat {
            Pattern::Ident { name, .. } => env.set_local(name, val),
            Pattern::Wildcard(_) => {}
            Pattern::Tuple { elems, .. } => {
                if let Value::Tuple(vs) = val {
                    for (p, v) in elems.iter().zip(vs) {
                        self.bind_pattern(p, v, env);
                    }
                }
            }
            Pattern::Struct { fields, .. } => {
                if let Value::Struct { fields: fmap, .. } = val {
                    for (fname, maybe_pat) in fields {
                        if let Some(fval) = fmap.get(fname) {
                            if let Some(p) = maybe_pat {
                                self.bind_pattern(p, fval.clone(), env);
                            } else {
                                env.set_local(fname, fval.clone());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // ── Collect iterable values into a Vec ────────────────────────────────

    fn value_to_iter(&self, v: Value) -> Result<Vec<Value>, RuntimeError> {
        match v {
            Value::Array(a) => Ok(a.lock().unwrap().clone()),
            Value::Tuple(vs) => Ok(vs),
            Value::Str(s) => Ok(s.chars().map(|c| Value::Str(c.to_string())).collect()),
            _ => rt_err!("value `{}` is not iterable", v.type_name()),
        }
    }
}

// =============================================================================
// §19  TRAINING LOOP RUNTIME  (Unique Feature 2)
// =============================================================================

/// Configuration resolved from a `TrainDecl`.
pub struct TrainingConfig {
    pub agent_name: String,
    pub world_name: String,
    pub model_name: Option<String>,
    pub max_steps: u64,
    pub num_envs: u64,
    pub lr: f32,
    pub gamma: f32,
    pub signals: Vec<(String, f32, bool)>, // (name, weight, is_reward)
    pub optimizer: OptimizerKind,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            agent_name: String::new(),
            world_name: String::new(),
            model_name: None,
            max_steps: 1000,
            num_envs: 1,
            lr: 3e-4,
            gamma: 0.99,
            signals: vec![],
            optimizer: OptimizerKind::Adam,
        }
    }
}

impl Interpreter {
    /// Execute a full training session described by a `TrainDecl`.
    pub fn run_training(
        &mut self,
        decl: &TrainDecl,
        interp_env: &mut Env,
    ) -> Result<TrainingStats, RuntimeError> {
        let max_steps = decl
            .episode
            .as_ref()
            .and_then(|e| e.max_steps)
            .unwrap_or(1000);
        let num_envs = decl.episode.as_ref().and_then(|e| e.num_envs).unwrap_or(1);
        let gamma = decl
            .hyper
            .iter()
            .find(|(k, _)| k == "gamma")
            .and_then(|(_, e)| {
                if let Expr::FloatLit { value, .. } = e {
                    Some(*value as f32)
                } else {
                    None
                }
            })
            .unwrap_or(0.99);

        // Instantiate model (if named).
        if let Some(model_name) = &decl.model {
            if !self.models.contains_key(model_name.as_str()) {
                if let Some(decl) = self.model_decls.get(model_name).cloned() {
                    let model = NnModel::from_decl(&decl);
                    self.models
                        .insert(model_name.clone(), Arc::new(Mutex::new(model)));
                }
            }
        }

        let mut stats = TrainingStats::default();
        let mut total_reward = 0.0_f32;

        // Run `num_envs` parallel episodes (sequential in interpreter).
        for _env_idx in 0..num_envs {
            // Reset world state for each episode.
            {
                let mut w = self.world.lock().unwrap();
                *w = EcsWorld::default();
            }

            let mut episode_reward = 0.0_f32;
            let mut done = false;

            for step in 0..max_steps {
                if done {
                    break;
                }

                // 1. Collect observations from world events (fallback to zeros).
                let obs = Tensor::zeros(vec![1, 4]);

                // 2. Forward pass through policy network.
                let _action = if let Some(model_name) = &decl.model {
                    if let Some(m) = self.models.get(model_name).cloned() {
                        let out = m.lock().unwrap().forward(obs)?;
                        Value::Tensor(Arc::new(RwLock::new(out)))
                    } else {
                        Value::Unit
                    }
                } else {
                    Value::Unit
                };

                // 3. Step the simulation (tick all systems).
                // In the full runtime this calls scheduler.tick().

                // 4. Accumulate reward signals.
                let step_reward: f32 = decl
                    .signals
                    .iter()
                    .map(|sig| {
                        let w = self.world.lock().unwrap();
                        let count = w
                            .events
                            .get(&sig.name)
                            .map(|v| v.len() as f32)
                            .unwrap_or(0.0);
                        if sig.is_reward {
                            count * sig.weight as f32
                        } else {
                            -count * sig.weight as f32
                        }
                    })
                    .sum();
                episode_reward += step_reward * gamma.powi(step as i32);

                // 5. Check done condition.
                if let Some(ep) = &decl.episode {
                    if let Some(cond) = &ep.done_condition {
                        done = self.eval_expr(cond, interp_env)?.is_truthy();
                    }
                }
            }

            total_reward += episode_reward;
            stats.episode_rewards.push(episode_reward);
        }

        // 6. Policy gradient update (simplified REINFORCE).
        if let Some(model_name) = &decl.model {
            if let Some(model) = self.models.get(model_name).cloned() {
                let lr = decl
                    .optimizer
                    .as_ref()
                    .map(|o| o.learning_rate as f32)
                    .unwrap_or(3e-4);
                update_model_weights(
                    &mut *model.lock().unwrap(),
                    total_reward / num_envs as f32,
                    lr,
                );
            }
        }

        stats.mean_reward = total_reward / num_envs as f32;
        stats.total_steps = max_steps * num_envs;
        Ok(stats)
    }
}

/// Summary statistics from one training session.
#[derive(Debug, Default)]
pub struct TrainingStats {
    pub mean_reward: f32,
    pub total_steps: u64,
    pub episode_rewards: Vec<f32>,
}

impl TrainingStats {
    pub fn display(&self) {
        println!("Training complete:");
        println!("  mean reward = {:.4}", self.mean_reward);
        println!("  total steps = {}", self.total_steps);
        println!("  episodes    = {}", self.episode_rewards.len());
    }
}

/// Simple gradient update: scale all weights by the policy gradient signal.
fn update_model_weights(model: &mut NnModel, reward: f32, lr: f32) {
    model.step += 1;
    for layer in &mut model.layers {
        match layer {
            WeightLayer::Dense { w, b, .. } => {
                // REINFORCE-style update: w ← w + lr * reward * ε
                let grad_scale = lr * reward.clamp(-1.0, 1.0);
                let w_data = w.cpu_data_mut();
                for x in w_data.iter_mut() {
                    *x += grad_scale * pseudo_rand_small();
                }
                let b_data = b.cpu_data_mut();
                for x in b_data.iter_mut() {
                    *x += grad_scale * pseudo_rand_small();
                }
            }
            _ => {}
        }
    }
}

// =============================================================================
// §20  GPU BACKEND TRAIT
// =============================================================================

/// Trait implemented by GPU backends (wgpu, CUDA via FFI, Metal).
/// The interpreter calls these when a tensor or loop is annotated for GPU.
pub trait GpuBackend: Send + Sync {
    /// Allocate a buffer on the GPU and copy data from the CPU.
    fn upload(&self, data: &[f32], shape: Vec<usize>) -> GpuBufferHandle;
    /// Copy a GPU buffer back to a CPU Vec<f32>.
    fn download(&self, handle: &GpuBufferHandle) -> Vec<f32>;
    /// Dispatch a matrix-multiply kernel: C = A @ B.
    fn matmul(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        shape_a: &[usize],
        shape_b: &[usize],
    ) -> GpuBufferHandle;
    /// Dispatch an element-wise kernel.
    fn elementwise(&self, a: &GpuBufferHandle, b: &GpuBufferHandle, op: GpuOp) -> GpuBufferHandle;
    /// Dispatch a parallel entity-loop kernel (SPIR-V / WGSL / CUDA PTX).
    fn dispatch_entity_loop(&self, entities: &[u64], workgroup_size: u32);
}

/// Primitive operations the GPU backend can dispatch.
#[derive(Debug, Clone, Copy)]
pub enum GpuOp {
    Add,
    Sub,
    Mul,
    Div,
    HadamardMul,
    HadamardDiv,
}

/// Jules-native GPU backend adapter used by the interpreter runtime.
struct JulesGpuAdapter {
    backend: crate::gpu_backend::GpuBackend,
}

impl JulesGpuAdapter {
    fn new() -> Self {
        JulesGpuAdapter {
            backend: crate::gpu_backend::GpuBackend::auto_select(),
        }
    }
}

impl GpuBackend for JulesGpuAdapter {
    fn upload(&self, data: &[f32], shape: Vec<usize>) -> GpuBufferHandle {
        let h = self.backend.upload(data, shape);
        GpuBufferHandle(h.id)
    }

    fn download(&self, handle: &GpuBufferHandle) -> Vec<f32> {
        self.backend
            .download(&crate::gpu_backend::GpuBufferHandle { id: handle.0 })
    }

    fn matmul(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        shape_a: &[usize],
        shape_b: &[usize],
    ) -> GpuBufferHandle {
        let m = shape_a[shape_a.len() - 2];
        let n = shape_b[shape_b.len() - 1];
        let out = self.backend.upload(&vec![0.0; m * n], vec![m, n]);
        let ga = crate::gpu_backend::GpuBufferHandle { id: a.0 };
        let gb = crate::gpu_backend::GpuBufferHandle { id: b.0 };
        let go = crate::gpu_backend::GpuBufferHandle { id: out.id };
        let _ = self.backend.matmul(&ga, &gb, &go);
        GpuBufferHandle(out.id)
    }

    fn elementwise(&self, a: &GpuBufferHandle, b: &GpuBufferHandle, op: GpuOp) -> GpuBufferHandle {
        let a_data = self.download(a);
        let out = self
            .backend
            .upload(&vec![0.0; a_data.len()], vec![a_data.len()]);
        let ga = crate::gpu_backend::GpuBufferHandle { id: a.0 };
        let gb = crate::gpu_backend::GpuBufferHandle { id: b.0 };
        let go = crate::gpu_backend::GpuBufferHandle { id: out.id };
        let mapped = match op {
            GpuOp::Add => crate::gpu_backend::GpuOp::Add,
            GpuOp::Sub => crate::gpu_backend::GpuOp::Sub,
            GpuOp::Mul | GpuOp::HadamardMul => crate::gpu_backend::GpuOp::Mul,
            GpuOp::Div | GpuOp::HadamardDiv => crate::gpu_backend::GpuOp::Div,
        };
        let _ = self.backend.elementwise(&ga, &gb, mapped, &go);
        GpuBufferHandle(out.id)
    }

    fn dispatch_entity_loop(&self, _entities: &[u64], _workgroup_size: u32) {
        // Hook kept for future parallel ECS dispatch.
    }
}

// =============================================================================
// §21  PURE HELPERS
// =============================================================================

/// Evaluate a binary operator on numeric Values.
/// Evaluate a binary arithmetic or comparison operation on two Values.
///
/// Hot path: I32 × I32 and F32 × F32 are handled first to avoid the full
/// numeric type dispatch on every inner-loop iteration.
#[inline]
fn eval_numeric_binop(op: BinOpKind, l: Value, r: Value) -> Result<Value, RuntimeError> {
    // ── Ultra-fast path: I32 × I32 (loop counters, indices, most arithmetic) ──
    if let (Value::I32(a), Value::I32(b)) = (&l, &r) {
        let (a, b) = (*a, *b);
        return match op {
            BinOpKind::Add => Ok(Value::I32(a.wrapping_add(b))),
            BinOpKind::Sub => Ok(Value::I32(a.wrapping_sub(b))),
            BinOpKind::Mul => Ok(Value::I32(a.wrapping_mul(b))),
            BinOpKind::Div => {
                if b == 0 {
                    rt_err!("division by zero")
                } else {
                    Ok(Value::I32(a / b))
                }
            }
            BinOpKind::Rem => {
                if b == 0 {
                    rt_err!("modulo by zero")
                } else {
                    Ok(Value::I32(a % b))
                }
            }
            BinOpKind::Lt => Ok(Value::Bool(a < b)),
            BinOpKind::Le => Ok(Value::Bool(a <= b)),
            BinOpKind::Gt => Ok(Value::Bool(a > b)),
            BinOpKind::Ge => Ok(Value::Bool(a >= b)),
            BinOpKind::Eq => Ok(Value::Bool(a == b)),
            BinOpKind::Ne => Ok(Value::Bool(a != b)),
            BinOpKind::BitAnd => Ok(Value::I32(a & b)),
            BinOpKind::BitOr => Ok(Value::I32(a | b)),
            BinOpKind::BitXor => Ok(Value::I32(a ^ b)),
            BinOpKind::Shl => Ok(Value::I32(a << (b as u32))),
            BinOpKind::Shr => Ok(Value::I32(a >> (b as u32))),
            _ => Err(RuntimeError::new(format!(
                "op {:?} not defined for i32",
                op
            ))),
        };
    }
    // ── Fast path: F32 × F32 ─────────────────────────────────────────────────
    if let (Value::F32(a), Value::F32(b)) = (&l, &r) {
        let (a, b) = (*a, *b);
        return match op {
            BinOpKind::Add => Ok(Value::F32(a + b)),
            BinOpKind::Sub => Ok(Value::F32(a - b)),
            BinOpKind::Mul => Ok(Value::F32(a * b)),
            BinOpKind::Div => {
                if b == 0.0 {
                    rt_err!("division by zero")
                } else {
                    Ok(Value::F32(a / b))
                }
            }
            BinOpKind::Rem => Ok(Value::F32(a % b)),
            BinOpKind::FloorDiv => {
                if b == 0.0 {
                    rt_err!("floor division by zero")
                } else {
                    Ok(Value::F32((a / b).floor()))
                }
            }
            BinOpKind::Lt => Ok(Value::Bool(a < b)),
            BinOpKind::Le => Ok(Value::Bool(a <= b)),
            BinOpKind::Gt => Ok(Value::Bool(a > b)),
            BinOpKind::Ge => Ok(Value::Bool(a >= b)),
            BinOpKind::Eq => Ok(Value::Bool(a == b)),
            BinOpKind::Ne => Ok(Value::Bool(a != b)),
            _ => Err(RuntimeError::new(format!(
                "op {:?} not defined for f32",
                op
            ))),
        };
    }
    // Tensor arithmetic.
    if let (Value::Tensor(a), Value::Tensor(b)) = (&l, &r) {
        let at = a.read().unwrap();
        let bt = b.read().unwrap();
        let result = match op {
            BinOpKind::Add => at.elementwise(&bt, |x, y| x + y, "+"),
            BinOpKind::Sub => at.elementwise(&bt, |x, y| x - y, "-"),
            BinOpKind::Mul => at.elementwise(&bt, |x, y| x * y, "*"),
            BinOpKind::Div => at.elementwise(&bt, |x, y| x / y, "/"),
            _ => return rt_err!("operator not supported for tensors"),
        }?;
        return Ok(Value::Tensor(Arc::new(RwLock::new(result))));
    }

    // Tensor + scalar broadcast.
    if let (Value::Tensor(t), ref scalar) | (ref scalar, Value::Tensor(t)) = (&l, &r) {
        if let Some(s) = scalar.as_f64() {
            let tt = t.read().unwrap();
            let data: Vec<f32> = tt
                .cpu_data()
                .iter()
                .map(|x| match op {
                    BinOpKind::Add => x + s as f32,
                    BinOpKind::Sub => x - s as f32,
                    BinOpKind::Mul => x * s as f32,
                    BinOpKind::Div => x / s as f32,
                    _ => *x,
                })
                .collect();
            let out = Tensor::from_data(tt.shape.clone(), data);
            return Ok(Value::Tensor(Arc::new(RwLock::new(out))));
        }
    }

    // Vec3 arithmetic.
    if let (Value::Vec3(a), Value::Vec3(b)) = (&l, &r) {
        return Ok(Value::Vec3(match op {
            BinOpKind::Add => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
            BinOpKind::Sub => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
            BinOpKind::Mul => [a[0] * b[0], a[1] * b[1], a[2] * b[2]],
            BinOpKind::Div => [a[0] / b[0], a[1] / b[1], a[2] / b[2]],
            _ => return rt_err!("operator not supported for vec3"),
        }));
    }

    // Vec3 * scalar.
    if let (Value::Vec3(v), ref s) | (ref s, Value::Vec3(v)) = (&l, &r) {
        if let Some(s) = s.as_f64().map(|f| f as f32) {
            return Ok(Value::Vec3([v[0] * s, v[1] * s, v[2] * s]));
        }
    }

    // Fast-path integer comparisons to avoid numeric widening/conversion.
    let int_cmp = match (&l, &r, op) {
        (Value::I64(a), Value::I64(b), BinOpKind::Lt) => Some(*a < *b),
        (Value::I64(a), Value::I64(b), BinOpKind::Le) => Some(*a <= *b),
        (Value::I64(a), Value::I64(b), BinOpKind::Gt) => Some(*a > *b),
        (Value::I64(a), Value::I64(b), BinOpKind::Ge) => Some(*a >= *b),
        (Value::I32(a), Value::I32(b), BinOpKind::Lt) => Some(*a < *b),
        (Value::I32(a), Value::I32(b), BinOpKind::Le) => Some(*a <= *b),
        (Value::I32(a), Value::I32(b), BinOpKind::Gt) => Some(*a > *b),
        (Value::I32(a), Value::I32(b), BinOpKind::Ge) => Some(*a >= *b),
        (Value::I64(a), Value::I32(b), BinOpKind::Lt) => Some(*a < *b as i64),
        (Value::I64(a), Value::I32(b), BinOpKind::Le) => Some(*a <= *b as i64),
        (Value::I64(a), Value::I32(b), BinOpKind::Gt) => Some(*a > *b as i64),
        (Value::I64(a), Value::I32(b), BinOpKind::Ge) => Some(*a >= *b as i64),
        (Value::I32(a), Value::I64(b), BinOpKind::Lt) => Some((*a as i64) < *b),
        (Value::I32(a), Value::I64(b), BinOpKind::Le) => Some((*a as i64) <= *b),
        (Value::I32(a), Value::I64(b), BinOpKind::Gt) => Some((*a as i64) > *b),
        (Value::I32(a), Value::I64(b), BinOpKind::Ge) => Some((*a as i64) >= *b),
        _ => None,
    };
    if let Some(b) = int_cmp {
        return Ok(Value::Bool(b));
    }

    // Comparison operators — always return Bool.
    let cmp_result = match op {
        BinOpKind::Eq => Some(value_eq(&l, &r)),
        BinOpKind::Ne => Some(!value_eq(&l, &r)),
        BinOpKind::Lt => l.as_f64().zip(r.as_f64()).map(|(a, b)| a < b),
        BinOpKind::Le => l.as_f64().zip(r.as_f64()).map(|(a, b)| a <= b),
        BinOpKind::Gt => l.as_f64().zip(r.as_f64()).map(|(a, b)| a > b),
        BinOpKind::Ge => l.as_f64().zip(r.as_f64()).map(|(a, b)| a >= b),
        _ => None,
    };
    if let Some(b) = cmp_result {
        return Ok(Value::Bool(b));
    }

    // Fast-path integer arithmetic for common loop-heavy workloads.
    match (&l, &r) {
        (Value::I64(a), Value::I64(b)) => return Ok(Value::I64(arith_i64(op, *a, *b)?)),
        (Value::I32(a), Value::I32(b)) => return Ok(Value::I32(arith_i32(op, *a, *b)?)),
        (Value::I64(a), Value::I32(b)) => return Ok(Value::I64(arith_i64(op, *a, *b as i64)?)),
        (Value::I32(a), Value::I64(b)) => return Ok(Value::I64(arith_i64(op, *a as i64, *b)?)),
        _ => {}
    }

    // Numeric arithmetic — promote to widest common type.
    match (&l, &r) {
        (Value::F64(a), _) | (_, Value::F64(a)) => {
            let b = r.as_f64().or_else(|| l.as_f64()).unwrap_or(*a);
            let a = l.as_f64().unwrap_or(*a);
            Ok(Value::F64(arith_f64(op, a, b)?))
        }
        (Value::F32(a), _) | (_, Value::F32(a)) => {
            let b = r.as_f64().or_else(|| l.as_f64()).unwrap_or(*a as f64);
            let a = l.as_f64().unwrap_or(*a as f64);
            Ok(Value::F32(arith_f64(op, a, b)? as f32))
        }
        _ => {
            let a = l.as_i64().unwrap_or(0);
            let b = r.as_i64().unwrap_or(0);
            Ok(Value::I32(arith_i64(op, a, b)? as i32))
        }
    }
}

#[inline(always)]
fn arith_f64(op: BinOpKind, a: f64, b: f64) -> Result<f64, RuntimeError> {
    Ok(match op {
        BinOpKind::Add => a + b,
        BinOpKind::Sub => a - b,
        BinOpKind::Mul => a * b,
        BinOpKind::Div => {
            if b == 0.0 {
                return rt_err!("division by zero");
            }
            a / b
        }
        BinOpKind::Rem => a % b,
        BinOpKind::FloorDiv => {
            if b == 0.0 {
                return rt_err!("floor division by zero");
            }
            (a / b).floor()
        }
        _ => return rt_err!("operator {:?} not defined for floats", op),
    })
}

#[inline(always)]
fn arith_i64(op: BinOpKind, a: i64, b: i64) -> Result<i64, RuntimeError> {
    Ok(match op {
        BinOpKind::Add => a.wrapping_add(b),
        BinOpKind::Sub => a.wrapping_sub(b),
        BinOpKind::Mul => a.wrapping_mul(b),
        BinOpKind::Div => {
            if b == 0 {
                return rt_err!("division by zero");
            }
            a / b
        }
        BinOpKind::Rem => {
            if b == 0 {
                return rt_err!("modulo by zero");
            }
            a % b
        }
        BinOpKind::BitAnd => a & b,
        BinOpKind::BitOr => a | b,
        BinOpKind::BitXor => a ^ b,
        BinOpKind::Shl => a << (b as u32),
        BinOpKind::Shr => a >> (b as u32),
        _ => return rt_err!("operator not defined for integers"),
    })
}

#[inline(always)]
fn arith_i32(op: BinOpKind, a: i32, b: i32) -> Result<i32, RuntimeError> {
    Ok(match op {
        BinOpKind::Add => a.wrapping_add(b),
        BinOpKind::Sub => a.wrapping_sub(b),
        BinOpKind::Mul => a.wrapping_mul(b),
        BinOpKind::Div => {
            if b == 0 {
                return rt_err!("division by zero");
            }
            a / b
        }
        BinOpKind::Rem => {
            if b == 0 {
                return rt_err!("modulo by zero");
            }
            a % b
        }
        BinOpKind::BitAnd => a & b,
        BinOpKind::BitOr => a | b,
        BinOpKind::BitXor => a ^ b,
        BinOpKind::Shl => a << (b as u32),
        BinOpKind::Shr => a >> (b as u32),
        _ => return rt_err!("operator not defined for integers"),
    })
}

#[inline(always)]
fn value_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::I32(x), Value::I32(y)) => x == y,
        (Value::I64(x), Value::I64(y)) => x == y,
        (Value::F32(x), Value::F32(y)) => x == y,
        (Value::F64(x), Value::F64(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Unit, Value::Unit) => true,
        _ => false,
    }
}

/// Compute the flat index into a tensor given multi-dimensional indices.
fn tensor_flat_index(shape: &[usize], indices: &[Value]) -> Result<usize, RuntimeError> {
    if indices.len() != shape.len() {
        return rt_err!(
            "tensor has rank {} but {} indices given",
            shape.len(),
            indices.len()
        );
    }
    let mut flat = 0;
    let mut stride = 1;
    for (i, (&dim, idx)) in shape.iter().zip(indices).enumerate().rev() {
        let idx_val = idx.as_i64().unwrap_or(0) as usize;
        if idx_val >= dim {
            return rt_err!(
                "index {} out of bounds for dimension {} (size {})",
                idx_val,
                i,
                dim
            );
        }
        flat += idx_val * stride;
        stride *= dim;
    }
    Ok(flat)
}

/// Vec swizzle: extract x/y/z/w components or multi-component swizzles.
fn swizzle_vec(components: &[f32], field: &str) -> Result<Value, String> {
    let mapped: Vec<f32> = field
        .chars()
        .map(|c| match c {
            'x' | 'r' => components.get(0).copied(),
            'y' | 'g' => components.get(1).copied(),
            'z' | 'b' => components.get(2).copied(),
            'w' | 'a' => components.get(3).copied(),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| format!("invalid swizzle `{field}`"))?;
    Ok(match mapped.len() {
        1 => Value::F32(mapped[0]),
        2 => Value::Vec2([mapped[0], mapped[1]]),
        3 => Value::Vec3([mapped[0], mapped[1], mapped[2]]),
        4 => Value::Vec4([mapped[0], mapped[1], mapped[2], mapped[3]]),
        _ => return Err(format!("swizzle `{field}` exceeds 4 components")),
    })
}

fn title_case(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

// =============================================================================
// BROADCAST HELPERS (numpy-style)
// =============================================================================

/// Compute the broadcast result shape of two shapes. Returns None if incompatible.
pub(crate) fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let len = a.len().max(b.len());
    let mut result = vec![0usize; len];
    for i in 0..len {
        let da = if i < len - a.len() {
            1
        } else {
            a[i - (len - a.len())]
        };
        let db = if i < len - b.len() {
            1
        } else {
            b[i - (len - b.len())]
        };
        result[i] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return None;
        };
    }
    Some(result)
}

/// Map a flat linear index in `result_shape` back to a flat index in `src_shape`,
/// respecting broadcast rules (dimensions of size 1 always map to index 0).
pub(crate) fn broadcast_index(flat: usize, result_shape: &[usize], src_shape: &[usize]) -> usize {
    let len = result_shape.len();
    let off = len - src_shape.len();
    let mut src_idx = 0usize;
    let mut stride = 1usize;
    let mut rem = flat;

    // Decompose flat index into multi-dim, then re-compose for src.
    let mut multi = vec![0usize; len];
    for i in (0..len).rev() {
        multi[i] = rem % result_shape[i];
        rem /= result_shape[i];
    }

    let mut src_strides = vec![1usize; src_shape.len()];
    for i in (0..src_shape.len().saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    }

    for i in 0..src_shape.len() {
        let ri = i + off;
        let idx = if src_shape[i] == 1 { 0 } else { multi[ri] };
        src_idx += idx * src_strides[i];
        let _ = stride;
        stride = 1;
    }
    src_idx
}

// =============================================================================
// VMAP  —  vectorise a closure over the batch (first) dimension
// =============================================================================

/// Apply a function to each slice along axis 0 of a tensor, collecting results.
///
/// If `input` has shape [B, ...], the function receives tensors of shape [...]
/// and the output is stacked to [B, output_dims...].
pub fn vmap_tensor<F>(input: &Tensor, f: F) -> Result<Tensor, RuntimeError>
where
    F: Fn(Tensor) -> Result<Tensor, RuntimeError>,
{
    if input.shape.is_empty() {
        return Err(RuntimeError::new("vmap requires at least a 1-D tensor"));
    }
    let batch = input.shape[0];
    let inner_shape = input.shape[1..].to_vec();
    let inner_n: usize = inner_shape.iter().product::<usize>().max(1);
    let data = input.cpu_data();

    // Collect slices first; the caller's `f` may be parallelised by a
    // Rayon-aware backend by replacing this loop with par_iter.
    let mut outputs = Vec::with_capacity(batch);
    for b in 0..batch {
        let start = b * inner_n;
        // Avoid one allocation by reusing the pre-sized slice copy.
        let slice = data[start..start + inner_n].to_vec();
        let t = Tensor::from_data(inner_shape.clone(), slice);
        outputs.push(f(t)?);
    }

    // Stack: all outputs must have the same shape
    let out_shape = outputs.first().map(|t| t.shape.clone()).unwrap_or_default();
    let out_n: usize = out_shape.iter().product::<usize>().max(1);
    let mut stacked = Vec::with_capacity(batch * out_n);
    for t in &outputs {
        stacked.extend_from_slice(t.cpu_data());
    }
    let mut final_shape = vec![batch];
    final_shape.extend(&out_shape);
    Ok(Tensor::from_data(final_shape, stacked))
}

// =============================================================================
// NOISE BUILTINS  (Perlin / value noise, no external crate)
// =============================================================================

/// Value noise in [0, 1] — cheap, no external dependency.
pub fn value_noise_2d(x: f32, y: f32) -> f32 {
    fn fade(t: f32) -> f32 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + t * (b - a)
    }
    fn hash(ix: i32, iy: i32) -> f32 {
        let n = ix
            .wrapping_mul(1619)
            .wrapping_add(iy.wrapping_mul(31337))
            .wrapping_mul(1013904223i32);
        (n as u32 as f32) / u32::MAX as f32
    }

    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let ux = fade(fx);
    let uy = fade(fy);

    lerp(
        lerp(hash(ix, iy), hash(ix + 1, iy), ux),
        lerp(hash(ix, iy + 1), hash(ix + 1, iy + 1), ux),
        uy,
    )
}

/// White noise uniform in [0, 1].
pub fn white_noise() -> f32 {
    pseudo_rand()
}

/// Fractional Brownian Motion — sum of octaves of value noise.
pub fn fbm_2d(x: f32, y: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut val = 0.0_f32;
    let mut amp = 0.5_f32;
    let mut freq = 1.0_f32;
    for _ in 0..octaves {
        val += amp * value_noise_2d(x * freq, y * freq);
        amp *= gain;
        freq *= lacunarity;
    }
    val
}

// =============================================================================
// PPO  —  Proximal Policy Optimisation training loop helpers
// =============================================================================

/// Compute Generalised Advantage Estimation (GAE).
///
/// `rewards`, `values`, `dones` are all length T (timestep).
/// Returns (advantages, returns) each of length T.
pub fn gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[f32], // 1.0 if episode ended, 0.0 otherwise
    gamma: f32,
    lam: f32,
    last_value: f32,
) -> (Vec<f32>, Vec<f32>) {
    let t = rewards.len();
    let mut advantages = vec![0.0_f32; t];
    let mut last_gae = 0.0_f32;
    let mut next_val = last_value;

    for i in (0..t).rev() {
        let mask = 1.0 - dones[i];
        let delta = rewards[i] + gamma * next_val * mask - values[i];
        last_gae = delta + gamma * lam * mask * last_gae;
        advantages[i] = last_gae;
        next_val = values[i];
    }

    let returns: Vec<f32> = advantages.iter().zip(values).map(|(a, v)| a + v).collect();
    (advantages, returns)
}

/// PPO surrogate loss (clipped objective).
///
/// Returns (policy_loss, value_loss, entropy).
pub fn ppo_loss(
    log_probs_old: &[f32],
    log_probs_new: &[f32],
    advantages: &[f32],
    returns: &[f32],
    values: &[f32],
    clip_eps: f32,
    vf_coef: f32,
    ent_coef: f32,
    entropy: &[f32],
) -> (f32, f32, f32) {
    let n = log_probs_old.len() as f32;

    // Policy loss (clipped surrogate)
    let policy_loss = log_probs_old
        .iter()
        .zip(log_probs_new)
        .zip(advantages)
        .map(|((old, new), adv)| {
            let ratio = (new - old).exp();
            let surr1 = ratio * adv;
            let surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv;
            -surr1.min(surr2)
        })
        .sum::<f32>()
        / n;

    // Value function loss (MSE, clipped)
    let value_loss = returns
        .iter()
        .zip(values)
        .map(|(r, v)| (r - v).powi(2))
        .sum::<f32>()
        / n;

    // Entropy bonus
    let mean_entropy = entropy.iter().sum::<f32>() / n;

    (
        policy_loss + vf_coef * value_loss - ent_coef * mean_entropy,
        value_loss,
        mean_entropy,
    )
}

// =============================================================================
// SIMPLE PHYSICS STEP  (Feature 7 — game dev integration)
// =============================================================================

/// Rigid body state for a single entity.
#[derive(Debug, Clone, Default)]
pub struct RigidBody {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub mass: f32,
    pub drag: f32, // linear damping coefficient
    pub is_static: bool,
}

impl RigidBody {
    pub fn new(mass: f32) -> Self {
        RigidBody {
            mass,
            drag: 0.02,
            ..Default::default()
        }
    }

    /// Semi-implicit Euler integration.
    pub fn integrate(&mut self, gravity: [f32; 3], dt: f32) {
        if self.is_static {
            return;
        }
        // a = F/m (gravity only here; external forces accumulated elsewhere)
        for i in 0..3 {
            self.vel[i] += gravity[i] * dt;
            self.vel[i] *= 1.0 - self.drag * dt;
            self.pos[i] += self.vel[i] * dt;
        }
    }

    /// Apply an impulse directly to velocity: v += impulse / mass.
    pub fn apply_impulse(&mut self, impulse: [f32; 3]) {
        if self.is_static || self.mass == 0.0 {
            return;
        }
        let inv_mass = 1.0 / self.mass;
        for i in 0..3 {
            self.vel[i] += impulse[i] * inv_mass;
        }
    }
}

/// Axis-aligned bounding box collider.
#[derive(Debug, Clone)]
pub struct AabbCollider {
    pub half_extents: [f32; 3],
}

impl AabbCollider {
    /// Test overlap of two AABBs at given positions.
    pub fn overlaps(&self, pos_a: [f32; 3], other: &AabbCollider, pos_b: [f32; 3]) -> bool {
        for i in 0..3 {
            let dist = (pos_a[i] - pos_b[i]).abs();
            if dist > self.half_extents[i] + other.half_extents[i] {
                return false;
            }
        }
        true
    }

    /// Compute penetration depth and normal for collision response.
    pub fn penetration(
        &self,
        pos_a: [f32; 3],
        other: &AabbCollider,
        pos_b: [f32; 3],
    ) -> Option<([f32; 3], f32)> {
        let mut min_pen = f32::INFINITY;
        let mut normal = [0.0_f32; 3];
        for i in 0..3 {
            let delta = pos_b[i] - pos_a[i];
            let overlap = (self.half_extents[i] + other.half_extents[i]) - delta.abs();
            if overlap <= 0.0 {
                return None;
            }
            if overlap < min_pen {
                min_pen = overlap;
                normal = [0.0; 3];
                normal[i] = delta.signum();
            }
        }
        Some((normal, min_pen))
    }
}

/// Resolve elastic collision between two rigid bodies.
pub fn resolve_collision(a: &mut RigidBody, b: &mut RigidBody, normal: [f32; 3], restitution: f32) {
    let rel_vel: f32 = (0..3).map(|i| (a.vel[i] - b.vel[i]) * normal[i]).sum();
    if rel_vel > 0.0 {
        return;
    } // separating

    let inv_a = if a.is_static { 0.0 } else { 1.0 / a.mass };
    let inv_b = if b.is_static { 0.0 } else { 1.0 / b.mass };
    let j = -(1.0 + restitution) * rel_vel / (inv_a + inv_b);

    for i in 0..3 {
        a.vel[i] += j * inv_a * normal[i];
        b.vel[i] -= j * inv_b * normal[i];
    }
}

// ── Matrix arithmetic helpers ──────────────────────────────────────────────

fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut c = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut c = [[0.0_f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn mat3_vec3_mul(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat4_vec4_mul(m: [[f32; 4]; 4], v: [f32; 4]) -> [f32; 4] {
    let mut r = [0.0_f32; 4];
    for i in 0..4 {
        for j in 0..4 {
            r[i] += m[i][j] * v[j];
        }
    }
    r
}

// ── PRNG (LCG, no external crate needed) ───────────────────────────────────

static RAND_STATE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(12345678);

fn pseudo_rand() -> f32 {
    use std::sync::atomic::Ordering::Relaxed;
    let s = RAND_STATE
        .fetch_add(2891336453, Relaxed)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    RAND_STATE.store(s, Relaxed);
    (s >> 33) as f32 / u32::MAX as f32
}

fn pseudo_rand_small() -> f32 {
    pseudo_rand() * 2.0 - 1.0
}

fn rand_normal(mean: f32, std: f32) -> f32 {
    // Box–Muller (two uniform → one normal).
    let u1 = pseudo_rand().max(1e-7);
    let u2 = pseudo_rand();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    mean + std * z
}

// =============================================================================
// §22  PUBLIC API
// =============================================================================

/// Convenience: load a program, evaluate a named function, and return the result.
pub fn jules_run(program: &Program, entry: &str, args: Vec<Value>) -> Result<Value, RuntimeError> {
    let mut interp = Interpreter::new();
    interp.load_program(program);
    interp.call_fn(entry, args)
}

/// Execute all `train` blocks found in the program.
pub fn jules_train(program: &Program) -> Result<Vec<TrainingStats>, RuntimeError> {
    let mut interp = Interpreter::new();
    interp.load_program(program);
    let mut all_stats = Vec::new();
    let trains: Vec<_> = program
        .items
        .iter()
        .filter_map(|i| {
            if let Item::Train(t) = i {
                Some(t.clone())
            } else {
                None
            }
        })
        .collect();
    let mut env = Env::new();
    for t in &trains {
        let stats = interp.run_training(t, &mut env)?;
        all_stats.push(stats);
    }
    Ok(all_stats)
}

// =============================================================================
// §23  TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;
    use crate::lexer::Span;

    fn sp() -> Span {
        Span::dummy()
    }

    // ── Value helpers ─────────────────────────────────────────────────────────

    #[test]
    fn test_value_as_f64() {
        assert_eq!(Value::F32(1.5).as_f64(), Some(1.5));
        assert_eq!(Value::I32(7).as_f64(), Some(7.0));
        assert_eq!(Value::Bool(true).as_f64(), None);
    }

    #[test]
    fn test_value_is_truthy() {
        assert!(Value::Bool(true).is_truthy());
        assert!(!Value::Bool(false).is_truthy());
        assert!(Value::I32(1).is_truthy());
        assert!(!Value::I32(0).is_truthy());
    }

    #[test]
    fn test_value_display() {
        assert_eq!(Value::I32(42).to_string(), "42");
        assert_eq!(Value::Bool(true).to_string(), "true");
        assert_eq!(Value::Unit.to_string(), "()");
        assert_eq!(Value::Vec3([1.0, 2.0, 3.0]).to_string(), "vec3(1, 2, 3)");
    }

    // ── Tensor operations ─────────────────────────────────────────────────────

    #[test]
    fn test_tensor_matmul_2x2() {
        // [1 2; 3 4] @ [1 0; 0 1] = [1 2; 3 4]
        let a = Tensor::from_data(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::from_data(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]); // identity
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        let d = c.cpu_data();
        assert!((d[0] - 1.0).abs() < 1e-5);
        assert!((d[1] - 2.0).abs() < 1e-5);
        assert!((d[2] - 3.0).abs() < 1e-5);
        assert!((d[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_tensor_matmul_shape_mismatch() {
        let a = Tensor::from_data(vec![2, 3], vec![0.0; 6]);
        let b = Tensor::from_data(vec![2, 2], vec![0.0; 4]);
        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn test_tensor_matmul_batched() {
        let a = Tensor::from_data(
            vec![2, 2, 3],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // batch 1
            ],
        );
        let b = Tensor::from_data(
            vec![2, 3, 2],
            vec![
                1.0, 0.0, 0.0, 1.0, 1.0, 1.0, // batch 0
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 1
            ],
        );
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape, vec![2, 2, 2]);
        let d = c.cpu_data();
        assert!((d[0] - 4.0).abs() < 1e-6);
        assert!((d[1] - 5.0).abs() < 1e-6);
        assert!((d[2] - 10.0).abs() < 1e-6);
        assert!((d[3] - 11.0).abs() < 1e-6);
        assert!((d[4] - 1.0).abs() < 1e-6);
        assert!((d[5] - 2.0).abs() < 1e-6);
        assert!((d[6] - 3.0).abs() < 1e-6);
        assert!((d[7] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_hadamard_mul() {
        let a = Tensor::from_data(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::from_data(vec![4], vec![2.0, 2.0, 2.0, 2.0]);
        let c = a.hadamard_mul(&b).unwrap();
        let d = c.cpu_data();
        assert_eq!(d, &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_tensor_concat_axis0() {
        let a = Tensor::from_data(vec![2, 3], vec![1.0; 6]);
        let b = Tensor::from_data(vec![3, 3], vec![2.0; 9]);
        let c = a.concat(&b).unwrap();
        assert_eq!(c.shape, vec![5, 3]);
        assert_eq!(c.numel(), 15);
    }

    #[test]
    fn test_tensor_concat_inner_mismatch() {
        let a = Tensor::from_data(vec![2, 3], vec![0.0; 6]);
        let b = Tensor::from_data(vec![2, 4], vec![0.0; 8]);
        assert!(a.concat(&b).is_err());
    }

    #[test]
    fn test_tensor_activation_relu() {
        let t = Tensor::from_data(vec![4], vec![-1.0, 0.0, 1.0, 2.0]);
        let out = t.apply_activation(&Activation::Relu);
        assert_eq!(out.cpu_data(), &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_tensor_activation_sigmoid() {
        let t = Tensor::from_data(vec![1], vec![0.0]);
        let out = t.apply_activation(&Activation::Sigmoid);
        assert!((out.cpu_data()[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_tensor_softmax_sums_to_one() {
        let t = Tensor::from_data(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]);
        let out = t.apply_activation(&Activation::Softmax);
        let sum: f32 = out.cpu_data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
    }

    #[test]
    fn test_tensor_mse_loss() {
        let pred = Tensor::from_data(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let tgt = Tensor::from_data(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let loss = pred.mse_loss(&tgt).unwrap();
        assert!(loss.abs() < 1e-6);
    }

    #[test]
    fn test_tensor_grad_attach() {
        let mut t = Tensor::zeros(vec![3, 3]);
        assert!(t.grad.is_none());
        t.enable_grad();
        assert!(t.grad.is_some());
    }

    #[test]
    fn test_tensor_transpose() {
        let t = Tensor::from_data(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t2 = t.transpose().unwrap();
        assert_eq!(t2.shape, vec![3, 2]);
        assert_eq!(t2.cpu_data()[0], 1.0);
        assert_eq!(t2.cpu_data()[1], 4.0); // first column of original
    }

    // ── ECS World ─────────────────────────────────────────────────────────────

    #[test]
    fn test_ecs_spawn_despawn() {
        let mut world = EcsWorld::default();
        let e1 = world.spawn();
        let e2 = world.spawn();
        assert!(world.is_alive(e1));
        world.despawn(e1);
        assert!(!world.is_alive(e1));
        assert!(world.is_alive(e2));
    }

    #[test]
    fn test_ecs_component_insert_get() {
        let mut world = EcsWorld::default();
        let e = world.spawn();
        world.insert_component(e, "health", Value::F32(100.0));
        let val = world.get_component(e, "health").cloned().unwrap();
        assert!(matches!(val, Value::F32(v) if (v - 100.0).abs() < 1e-5));
    }

    #[test]
    fn test_ecs_query_with_filter() {
        let mut world = EcsWorld::default();
        let e1 = world.spawn();
        let e2 = world.spawn();
        world.insert_component(e1, "Position", Value::Vec3([0.0, 0.0, 0.0]));
        world.insert_component(e1, "Velocity", Value::Vec3([1.0, 0.0, 0.0]));
        world.insert_component(e2, "Position", Value::Vec3([1.0, 0.0, 0.0]));
        // e1 has both; e2 only has Position.
        let with_vel = world.query(&["Position".into(), "Velocity".into()], &[]);
        assert_eq!(with_vel.len(), 1);
        assert_eq!(with_vel[0], e1);
    }

    #[test]
    fn test_ecs_query2() {
        let mut world = EcsWorld::default();
        let e1 = world.spawn();
        let e2 = world.spawn();
        world.insert_component(e1, "pos", Value::Vec3([0.0, 0.0, 0.0]));
        world.insert_component(e1, "vel", Value::Vec3([1.0, 0.0, 0.0]));
        world.insert_component(e2, "pos", Value::Vec3([1.0, 0.0, 0.0]));
        let ids = world.query2("pos", "vel");
        assert_eq!(ids, vec![e1]);
    }

    #[test]
    fn test_ecs_query_without() {
        let mut world = EcsWorld::default();
        let alive = world.spawn();
        let dead = world.spawn();
        world.insert_component(alive, "Health", Value::F32(100.0));
        world.insert_component(dead, "Health", Value::F32(0.0));
        world.insert_component(dead, "Dead", Value::Unit);
        let result = world.query(&["Health".into()], &["Dead".into()]);
        assert_eq!(result, vec![alive]);
    }

    #[test]
    fn test_ecs_events() {
        let mut world = EcsWorld::default();
        let e = world.spawn();
        world.emit_event("killed", e);
        world.emit_event("killed", e);
        let evts = world.drain_events("killed");
        assert_eq!(evts.len(), 2);
        let evts2 = world.drain_events("killed");
        assert!(evts2.is_empty());
    }

    #[test]
    fn test_ecs_integrate_vec3_linear() {
        let mut world = EcsWorld::default();
        let a = world.spawn();
        world.insert_component(a, "pos", Value::Vec3([0.0, 0.0, 0.0]));
        world.insert_component(a, "vel", Value::Vec3([2.0, 0.0, 0.0]));
        let b = world.spawn();
        world.insert_component(b, "pos", Value::Vec3([1.0, 1.0, 1.0]));
        world.insert_component(b, "vel", Value::Vec3([0.0, -2.0, 0.0]));

        let n = world.integrate_vec3_linear("pos", "vel", 0.5);
        assert_eq!(n, 2);
        assert!(
            matches!(world.get_component(a, "pos"), Some(Value::Vec3(p)) if (p[0] - 1.0).abs() < 1e-6)
        );
        assert!(
            matches!(world.get_component(b, "pos"), Some(Value::Vec3(p)) if (p[1] - 0.0).abs() < 1e-6)
        );
    }

    #[test]
    fn test_ecs_integrate_vec3_linear_fused() {
        let mut world = EcsWorld::default();
        let a = world.spawn();
        world.insert_component(a, "pos", Value::Vec3([0.0, 0.0, 0.0]));
        world.insert_component(a, "vel", Value::Vec3([2.0, 0.0, 0.0]));
        let b = world.spawn();
        world.insert_component(b, "pos", Value::Vec3([1.0, 1.0, 1.0]));
        world.insert_component(b, "vel", Value::Vec3([0.0, -2.0, 0.0]));

        let n = world.integrate_vec3_linear_fused("pos", "vel", 0.5);
        assert_eq!(n, 2);
        assert!(
            matches!(world.get_component(a, "pos"), Some(Value::Vec3(p)) if (p[0] - 1.0).abs() < 1e-6)
        );
        assert!(
            matches!(world.get_component(b, "pos"), Some(Value::Vec3(p)) if (p[1] - 0.0).abs() < 1e-6)
        );
    }

    #[test]
    fn test_ecs_integrate_vec3_chunked_precomputed() {
        let mut world = EcsWorld::default();
        let a = world.spawn();
        world.insert_component(a, "pos", Value::Vec3([0.0, 0.0, 0.0]));
        world.insert_component(a, "vel", Value::Vec3([2.0, 0.0, 0.0]));
        let b = world.spawn();
        world.insert_component(b, "pos", Value::Vec3([1.0, 1.0, 1.0]));
        world.insert_component(b, "vel", Value::Vec3([0.0, -2.0, 0.0]));

        let n = world.integrate_vec3_chunked_precomputed("pos", "vel", 0.5, 64);
        assert_eq!(n, 2);
        assert!(
            matches!(world.get_component(a, "pos"), Some(Value::Vec3(p)) if (p[0] - 1.0).abs() < 1e-6)
        );
        assert!(
            matches!(world.get_component(b, "pos"), Some(Value::Vec3(p)) if (p[1] - 0.0).abs() < 1e-6)
        );
    }

    #[test]
    fn test_ecs_integrate_vec3_superoptimizer() {
        let mut world = EcsWorld::default();
        let a = world.spawn();
        world.insert_component(a, "pos", Value::Vec3([0.0, 0.0, 0.0]));
        world.insert_component(a, "vel", Value::Vec3([2.0, 0.0, 0.0]));
        let b = world.spawn();
        world.insert_component(b, "pos", Value::Vec3([1.0, 1.0, 1.0]));
        world.insert_component(b, "vel", Value::Vec3([0.0, -2.0, 0.0]));

        let n = world.integrate_vec3_superoptimizer("pos", "vel", 0.5, 64);
        assert_eq!(n, 2);
        assert!(
            matches!(world.get_component(a, "pos"), Some(Value::Vec3(p)) if (p[0] - 1.0).abs() < 1e-6)
        );
        assert!(
            matches!(world.get_component(b, "pos"), Some(Value::Vec3(p)) if (p[1] - 0.0).abs() < 1e-6)
        );
    }

    #[test]
    fn test_ecs_integrate_vec3_and_health_chunked() {
        let mut world = EcsWorld::default();
        let e = world.spawn();
        world.insert_component(e, "pos", Value::Vec3([0.0, 0.0, 0.0]));
        world.insert_component(e, "vel", Value::Vec3([1.0, 0.0, 0.0]));
        world.insert_component(e, "health", Value::F32(10.0));
        world.insert_component(e, "damage", Value::F32(2.0));
        let n = world.integrate_vec3_and_health_chunked("pos", "vel", "health", "damage", 0.5, 64);
        assert_eq!(n, 1);
        assert!(
            matches!(world.get_component(e, "pos"), Some(Value::Vec3(p)) if (p[0] - 0.5).abs() < 1e-6)
        );
        assert!(
            matches!(world.get_component(e, "health"), Some(Value::F32(h)) if (h - 9.0).abs() < 1e-6)
        );
    }

    // ── Interpreter ───────────────────────────────────────────────────────────

    fn mk_interp() -> Interpreter {
        Interpreter::new()
    }

    fn eval(expr: &Expr) -> Value {
        let mut i = mk_interp();
        let mut env = Env::new();
        i.eval_expr(expr, &mut env).unwrap()
    }

    #[test]
    fn test_interp_int_lit() {
        assert!(matches!(
            eval(&Expr::IntLit {
                span: sp(),
                value: 42
            }),
            Value::I32(42)
        ));
    }

    #[test]
    fn test_interp_float_lit() {
        assert!(matches!(
            eval(&Expr::FloatLit {
                span: sp(),
                value: 3.14
            }),
            Value::F32(_)
        ));
    }

    #[test]
    fn test_interp_bool_lit() {
        assert!(matches!(
            eval(&Expr::BoolLit {
                span: sp(),
                value: true
            }),
            Value::Bool(true)
        ));
    }

    #[test]
    fn test_interp_binop_add() {
        let e = Expr::BinOp {
            span: sp(),
            op: BinOpKind::Add,
            lhs: Box::new(Expr::IntLit {
                span: sp(),
                value: 3,
            }),
            rhs: Box::new(Expr::IntLit {
                span: sp(),
                value: 4,
            }),
        };
        assert!(matches!(eval(&e), Value::I32(7)));
    }

    #[test]
    fn test_interp_binop_compare() {
        let e = Expr::BinOp {
            span: sp(),
            op: BinOpKind::Lt,
            lhs: Box::new(Expr::IntLit {
                span: sp(),
                value: 3,
            }),
            rhs: Box::new(Expr::IntLit {
                span: sp(),
                value: 5,
            }),
        };
        assert!(matches!(eval(&e), Value::Bool(true)));
    }

    #[test]
    fn test_interp_binop_div_zero() {
        let e = Expr::BinOp {
            span: sp(),
            op: BinOpKind::Div,
            lhs: Box::new(Expr::IntLit {
                span: sp(),
                value: 10,
            }),
            rhs: Box::new(Expr::IntLit {
                span: sp(),
                value: 0,
            }),
        };
        let mut i = mk_interp();
        let mut env = Env::new();
        assert!(i.eval_expr(&e, &mut env).is_err());
    }

    #[test]
    fn test_interp_vec3_ctor() {
        let e = Expr::VecCtor {
            span: sp(),
            size: VecSize::N3,
            elems: vec![
                Expr::FloatLit {
                    span: sp(),
                    value: 1.0,
                },
                Expr::FloatLit {
                    span: sp(),
                    value: 2.0,
                },
                Expr::FloatLit {
                    span: sp(),
                    value: 3.0,
                },
            ],
        };
        assert!(matches!(eval(&e), Value::Vec3([1.0, 2.0, 3.0])));
    }

    #[test]
    fn test_interp_unop_neg() {
        let e = Expr::UnOp {
            span: sp(),
            op: UnOpKind::Neg,
            expr: Box::new(Expr::FloatLit {
                span: sp(),
                value: 5.0,
            }),
        };
        assert!(matches!(eval(&e), Value::F32(v) if (v + 5.0).abs() < 1e-6));
    }

    #[test]
    fn test_interp_let_and_ident() {
        let mut i = mk_interp();
        let mut env = Env::new();
        let stmt = Stmt::Let {
            span: sp(),
            pattern: Pattern::Ident {
                span: sp(),
                name: "x".into(),
                mutable: false,
            },
            ty: None,
            init: Some(Expr::IntLit {
                span: sp(),
                value: 99,
            }),
            mutable: false,
        };
        i.eval_stmt(&stmt, &mut env).unwrap();
        let v = i
            .eval_expr(
                &Expr::Ident {
                    span: sp(),
                    name: "x".into(),
                },
                &mut env,
            )
            .unwrap();
        assert!(matches!(v, Value::I32(99)));
    }

    #[test]
    fn test_interp_if_expr() {
        let mut i = mk_interp();
        let mut env = Env::new();
        let e = Expr::IfExpr {
            span: sp(),
            cond: Box::new(Expr::BoolLit {
                span: sp(),
                value: true,
            }),
            then: Box::new(Block {
                span: sp(),
                stmts: vec![],
                tail: Some(Box::new(Expr::IntLit {
                    span: sp(),
                    value: 1,
                })),
            }),
            else_: Some(Box::new(Block {
                span: sp(),
                stmts: vec![],
                tail: Some(Box::new(Expr::IntLit {
                    span: sp(),
                    value: 2,
                })),
            })),
        };
        assert!(matches!(i.eval_expr(&e, &mut env).unwrap(), Value::I32(1)));
    }

    #[test]
    fn test_interp_dataloader_basics() {
        let mut interp = mk_interp();
        let mut env = Env::new();
        let source = Value::Array(Arc::new(Mutex::new(vec![
            Value::I32(1),
            Value::I32(2),
            Value::I32(3),
            Value::I32(4),
        ])));

        let loader = interp
            .eval_builtin(
                "dataloader",
                vec![source.clone(), Value::I32(2), Value::Bool(false)],
            )
            .unwrap();

        let has_next = interp
            .eval_method(loader.clone(), "has_next", vec![], &mut env)
            .unwrap();
        assert!(matches!(has_next, Value::Bool(true)));

        let next_batch = interp
            .eval_method(loader.clone(), "next", vec![], &mut env)
            .unwrap();
        match next_batch {
            Value::Some(boxed_batch) => {
                if let Value::Array(arr) = *boxed_batch {
                    let v = arr.lock().unwrap();
                    assert_eq!(v.len(), 2);
                } else {
                    panic!("expected array batch");
                }
            }
            _ => panic!("expected Some(batch)"),
        }

        let collected = interp
            .eval_method(loader.clone(), "collect", vec![], &mut env)
            .unwrap();
        match collected {
            Value::Array(arr) => assert_eq!(arr.lock().unwrap().len(), 4),
            _ => panic!("expected array from collect"),
        }

        let repeated = interp
            .eval_method(loader.clone(), "repeat", vec![Value::I32(2)], &mut env)
            .unwrap();
        if let Value::DataLoader(dl) = repeated {
            assert_eq!(dl.lock().unwrap().samples.len(), 8);
        } else {
            panic!("expected DataLoader from repeat");
        }

        let b = interp
            .eval_method(loader, "batch", vec![Value::I32(1)], &mut env)
            .unwrap();
        if let Value::DataLoader(dl) = b {
            assert_eq!(dl.lock().unwrap().batch_size, 1);
        } else {
            panic!("expected DataLoader from batch");
        }
    }

    #[test]
    fn test_interp_range_zeros_ones() {
        let mut interp = mk_interp();

        let r = interp
            .eval_builtin("range", vec![Value::I32(0), Value::I32(5), Value::I32(2)])
            .unwrap();
        if let Value::Array(arr) = r {
            let a = arr.lock().unwrap();
            let ints: Vec<i32> = a
                .iter()
                .filter_map(|v| match v {
                    Value::I32(x) => Some(*x),
                    _ => None,
                })
                .collect();
            assert_eq!(ints, vec![0, 2, 4]);
        } else {
            panic!("expected array from range");
        }

        let z = interp.eval_builtin("zeros", vec![Value::I32(3)]).unwrap();
        if let Value::Array(arr) = z {
            assert_eq!(arr.lock().unwrap().len(), 3);
        } else {
            panic!("expected array from zeros");
        }

        let o = interp.eval_builtin("ones", vec![Value::I32(2)]).unwrap();
        if let Value::Array(arr) = o {
            let values: Vec<f32> = arr
                .lock()
                .unwrap()
                .iter()
                .filter_map(|v| match v {
                    Value::F32(x) => Some(*x),
                    _ => None,
                })
                .collect();
            assert_eq!(values, vec![1.0, 1.0]);
        } else {
            panic!("expected array from ones");
        }
    }

    #[test]
    fn test_stdlib_module_aliases_and_catalog() {
        let mut interp = mk_interp();

        let modules = interp.eval_builtin("std::modules", vec![]).unwrap();
        if let Value::HashMap(map) = modules {
            let m = map.lock().unwrap();
            assert!(m.contains_key("core"));
            assert!(m.contains_key("math"));
            assert!(m.contains_key("tensor"));
            assert!(m.contains_key("nn"));
            assert!(m.contains_key("train"));
            assert!(m.contains_key("render"));
            if let Some(Value::Array(math_mod)) = m.get("math") {
                let vals = math_mod.lock().unwrap();
                assert!(vals
                    .iter()
                    .any(|v| matches!(v, Value::Str(s) if s == "distance2")));
            } else {
                panic!("expected math module list");
            }
            if let Some(Value::Array(render_mod)) = m.get("render") {
                let vals = render_mod.lock().unwrap();
                assert!(vals
                    .iter()
                    .any(|v| matches!(v, Value::Str(s) if s == "render::flush")));
            } else {
                panic!("expected render module list");
            }
        } else {
            panic!("expected hashmap from std::modules");
        }

        let seeded = interp
            .eval_builtin("math::random_seed", vec![Value::I64(1234)])
            .unwrap();
        assert!(matches!(seeded, Value::Bool(true)));
        let r1 = interp.eval_builtin("math::random", vec![]).unwrap();
        interp
            .eval_builtin("math::random_seed", vec![Value::I64(1234)])
            .unwrap();
        let r2 = interp.eval_builtin("math::random", vec![]).unwrap();
        match (r1, r2) {
            (Value::F32(a), Value::F32(b)) => assert!((a - b).abs() < 1e-12),
            _ => panic!("expected random values"),
        }

        let zeroes = interp
            .eval_builtin("tensor::zeros", vec![Value::I32(4)])
            .unwrap();
        assert!(matches!(zeroes, Value::Array(_)));
        let s = interp
            .eval_builtin(
                "tensor::sum",
                vec![Value::Array(Arc::new(Mutex::new(vec![
                    Value::I32(1),
                    Value::I32(2),
                    Value::I32(3),
                ])))],
            )
            .unwrap();
        assert!(matches!(s, Value::F32(v) if (v - 6.0).abs() < 1e-6));
    }

    #[test]
    fn test_sim_and_window_minimal_loop() {
        let mut interp = mk_interp();
        let world = interp
            .eval_builtin("sim::world", vec![Value::F32(0.1), Value::I64(7)])
            .unwrap();
        let world_id = match world {
            Value::I64(id) => id,
            _ => panic!("expected world id"),
        };

        let mut entity = FxHashMap::default();
        entity.insert(
            "position".into(),
            Value::Array(Arc::new(Mutex::new(vec![Value::F32(0.0), Value::F32(0.0)]))),
        );
        entity.insert(
            "velocity".into(),
            Value::Array(Arc::new(Mutex::new(vec![Value::F32(1.0), Value::F32(0.0)]))),
        );
        let e = interp
            .eval_builtin(
                "sim::spawn",
                vec![
                    Value::I64(world_id),
                    Value::HashMap(Arc::new(Mutex::new(entity))),
                ],
            )
            .unwrap();
        let entity_id = match e {
            Value::I64(id) => id,
            _ => panic!("expected entity id"),
        };

        interp
            .eval_builtin("sim::step", vec![Value::I64(world_id), Value::F32(0.1)])
            .unwrap();
        let state = interp
            .eval_builtin("sim::get_state", vec![Value::I64(world_id)])
            .unwrap();
        if let Value::Array(rows) = state {
            let rows = rows.lock().unwrap();
            assert_eq!(rows.len(), 1);
            if let Value::HashMap(map) = &rows[0] {
                let m = map.lock().unwrap();
                assert!(matches!(m.get("id"), Some(Value::I64(id)) if *id == entity_id));
            } else {
                panic!("expected state row hashmap");
            }
        } else {
            panic!("expected sim state array");
        }

        let win = interp
            .eval_builtin(
                "window::create",
                vec![Value::I32(640), Value::I32(360), Value::Str("Sim".into())],
            )
            .unwrap();
        let win_id = match win {
            Value::I64(id) => id,
            _ => panic!("expected window id"),
        };
        assert!(matches!(
            interp
                .eval_builtin("window::open", vec![Value::I64(win_id)])
                .unwrap(),
            Value::Bool(true)
        ));
        interp
            .eval_builtin("window::clear", vec![Value::I64(win_id)])
            .unwrap();
        interp
            .eval_builtin(
                "window::draw_rect",
                vec![
                    Value::I64(win_id),
                    Value::F32(0.0),
                    Value::F32(0.0),
                    Value::F32(10.0),
                    Value::F32(10.0),
                ],
            )
            .unwrap();
        interp
            .eval_builtin("window::present", vec![Value::I64(win_id)])
            .unwrap();
        interp
            .eval_builtin("window::close", vec![Value::I64(win_id)])
            .unwrap();
        assert!(matches!(
            interp
                .eval_builtin("window::open", vec![Value::I64(win_id)])
                .unwrap(),
            Value::Bool(false)
        ));
    }

    #[test]
    fn test_expanded_stdlib_helpers() {
        let mut interp = mk_interp();

        interp
            .eval_builtin("math::random_seed", vec![Value::I64(99)])
            .unwrap();
        let r = interp
            .eval_builtin("math::rand_int", vec![Value::I64(10), Value::I64(20)])
            .unwrap();
        assert!(matches!(r, Value::I64(v) if (10..20).contains(&v)));

        assert!(matches!(
            interp.eval_builtin("math::clamp01", vec![Value::F32(2.5)]).unwrap(),
            Value::F32(v) if (v - 1.0).abs() < 1e-6
        ));
        assert!(matches!(
            interp.eval_builtin("math::sigmoid", vec![Value::F32(0.0)]).unwrap(),
            Value::F32(v) if (v - 0.5).abs() < 1e-6
        ));

        let norm = interp
            .eval_builtin(
                "tensor::normalize",
                vec![Value::Array(Arc::new(Mutex::new(vec![
                    Value::F32(3.0),
                    Value::F32(4.0),
                ])))],
            )
            .unwrap();
        if let Value::Array(arr) = norm {
            let vals = arr.lock().unwrap();
            let a = match vals[0] {
                Value::F32(v) => v,
                _ => 0.0,
            };
            let b = match vals[1] {
                Value::F32(v) => v,
                _ => 0.0,
            };
            assert!((a - 0.6).abs() < 1e-5);
            assert!((b - 0.8).abs() < 1e-5);
        } else {
            panic!("expected normalized array");
        }

        let w = interp
            .eval_builtin(
                "window::create",
                vec![Value::I32(320), Value::I32(200), Value::Str("T".into())],
            )
            .unwrap();
        let id = match w {
            Value::I64(i) => i,
            _ => panic!("expected id"),
        };
        interp
            .eval_builtin("window::present", vec![Value::I64(id)])
            .unwrap();
        assert!(matches!(
            interp
                .eval_builtin("window::frames", vec![Value::I64(id)])
                .unwrap(),
            Value::I64(1)
        ));
        assert!(matches!(
            interp.eval_builtin("window::title", vec![Value::I64(id)]).unwrap(),
            Value::Str(ref s) if s == "T"
        ));
        if let Value::Array(sz) = interp
            .eval_builtin("window::size", vec![Value::I64(id)])
            .unwrap()
        {
            let s = sz.lock().unwrap();
            assert!(matches!(s[0], Value::I64(320)));
            assert!(matches!(s[1], Value::I64(200)));
        } else {
            panic!("expected window size array");
        }
    }

    #[test]
    fn test_runtime_hotspot_weighting_builtin() {
        let mut interp = mk_interp();
        interp
            .eval_builtin("debug::runtime_profile", vec![Value::Bool(true)])
            .unwrap();
        interp
            .eval_builtin("debug::runtime_profile_reset", vec![])
            .unwrap();
        interp
            .runtime_profile
            .insert("sim::step".to_string(), (120, 9_000_000));
        interp
            .runtime_profile
            .insert("render::flush".to_string(), (120, 3_000_000));
        let hotspots = interp
            .eval_builtin("debug::runtime_hotspots", vec![Value::I64(4)])
            .unwrap();
        if let Value::Array(rows) = hotspots {
            let vals = rows.lock().unwrap();
            assert!(!vals.is_empty());
        } else {
            panic!("expected array from debug::runtime_hotspots");
        }
    }

    #[test]
    fn test_sim_step_many_entities_stable() {
        let mut interp = mk_interp();
        let world = interp
            .eval_builtin("sim::world", vec![Value::F32(0.016), Value::I64(123)])
            .unwrap();
        let world_id = match world {
            Value::I64(id) => id,
            _ => panic!("expected world id"),
        };

        for i in 0..128 {
            let mut entity = FxHashMap::default();
            let x = (i % 16) as f32 * 0.4;
            let y = (i / 16) as f32 * 0.4;
            entity.insert(
                "position".into(),
                Value::Array(Arc::new(Mutex::new(vec![Value::F32(x), Value::F32(y)]))),
            );
            entity.insert(
                "velocity".into(),
                Value::Array(Arc::new(Mutex::new(vec![
                    Value::F32(0.2),
                    Value::F32(-0.1),
                ]))),
            );
            let _ = interp
                .eval_builtin(
                    "sim::spawn",
                    vec![
                        Value::I64(world_id),
                        Value::HashMap(Arc::new(Mutex::new(entity))),
                    ],
                )
                .unwrap();
        }

        for _ in 0..50 {
            interp
                .eval_builtin("sim::step", vec![Value::I64(world_id)])
                .unwrap();
        }

        let state = interp
            .eval_builtin("sim::state_tensor", vec![Value::I64(world_id)])
            .unwrap();
        if let Value::Tensor(t) = state {
            let tt = t.read().unwrap();
            assert_eq!(tt.shape, vec![128, 4]);
            assert_eq!(tt.numel(), 512);
        } else {
            panic!("expected tensor from sim::state_tensor");
        }
    }

    #[test]
    fn test_render_command_buffer_api() {
        let mut interp = mk_interp();
        interp
            .eval_builtin(
                "render::begin_frame",
                vec![Value::I64(1280), Value::I64(720)],
            )
            .unwrap();
        interp
            .eval_builtin(
                "render::clear",
                vec![
                    Value::F32(0.1),
                    Value::F32(0.2),
                    Value::F32(0.3),
                    Value::F32(1.0),
                ],
            )
            .unwrap();
        interp
            .eval_builtin(
                "render::rect",
                vec![
                    Value::F32(20.0),
                    Value::F32(30.0),
                    Value::F32(64.0),
                    Value::F32(16.0),
                    Value::F32(1.0),
                    Value::F32(0.0),
                    Value::F32(0.0),
                    Value::F32(1.0),
                    Value::I64(2),
                ],
            )
            .unwrap();
        let sprite_id = match interp
            .eval_builtin(
                "graphics::create_sprite",
                vec![
                    Value::Str("hero".into()),
                    Value::F32(32.0),
                    Value::F32(32.0),
                ],
            )
            .unwrap()
        {
            Value::I64(id) => id,
            _ => panic!("expected sprite id"),
        };
        interp
            .eval_builtin(
                "render::sprite",
                vec![
                    Value::I64(sprite_id),
                    Value::F32(48.0),
                    Value::F32(64.0),
                    Value::F32(32.0),
                    Value::F32(32.0),
                    Value::F32(45.0),
                    Value::I64(3),
                ],
            )
            .unwrap();

        let stats = interp.eval_builtin("render::stats", vec![]).unwrap();
        if let Value::HashMap(m) = stats {
            let m = m.lock().unwrap();
            assert!(matches!(m.get("width"), Some(Value::I64(1280))));
            assert!(matches!(m.get("height"), Some(Value::I64(720))));
            assert!(matches!(m.get("queued_commands"), Some(Value::I64(3))));
        } else {
            panic!("expected render::stats map");
        }

        let flushed = interp.eval_builtin("render::flush", vec![]).unwrap();
        if let Value::Array(cmds) = flushed {
            let cmds = cmds.lock().unwrap();
            assert_eq!(cmds.len(), 3);
        } else {
            panic!("expected command list from render::flush");
        }

        let stats_after = interp.eval_builtin("render::stats", vec![]).unwrap();
        if let Value::HashMap(m) = stats_after {
            let m = m.lock().unwrap();
            assert!(matches!(m.get("queued_commands"), Some(Value::I64(0))));
        } else {
            panic!("expected render::stats map");
        }
    }

    #[test]
    fn test_game_math_helpers() {
        let mut interp = mk_interp();
        let dot = interp
            .eval_builtin(
                "math::dot2",
                vec![
                    Value::F32(1.0),
                    Value::F32(2.0),
                    Value::F32(3.0),
                    Value::F32(4.0),
                ],
            )
            .unwrap();
        assert!(matches!(dot, Value::F32(v) if (v - 11.0).abs() < 1e-6));

        let dist = interp
            .eval_builtin(
                "math::distance2",
                vec![
                    Value::F32(0.0),
                    Value::F32(0.0),
                    Value::F32(3.0),
                    Value::F32(4.0),
                ],
            )
            .unwrap();
        assert!(matches!(dist, Value::F32(v) if (v - 5.0).abs() < 1e-6));

        let remap = interp
            .eval_builtin(
                "math::remap",
                vec![
                    Value::F32(0.5),
                    Value::F32(0.0),
                    Value::F32(1.0),
                    Value::F32(-1.0),
                    Value::F32(1.0),
                ],
            )
            .unwrap();
        assert!(matches!(remap, Value::F32(v) if v.abs() < 1e-6));

        let approach = interp
            .eval_builtin(
                "math::approach",
                vec![Value::F32(0.0), Value::F32(10.0), Value::F32(3.0)],
            )
            .unwrap();
        assert!(matches!(approach, Value::F32(v) if (v - 3.0).abs() < 1e-6));

        let move_towards = interp
            .eval_builtin(
                "math::move_towards2",
                vec![
                    Value::F32(0.0),
                    Value::F32(0.0),
                    Value::F32(3.0),
                    Value::F32(4.0),
                    Value::F32(2.5),
                ],
            )
            .unwrap();
        if let Value::Array(v) = move_towards {
            let v = v.lock().unwrap();
            let x = match v[0] {
                Value::F32(n) => n,
                _ => panic!("x should be f32"),
            };
            let y = match v[1] {
                Value::F32(n) => n,
                _ => panic!("y should be f32"),
            };
            assert!((x - 1.5).abs() < 1e-6);
            assert!((y - 2.0).abs() < 1e-6);
        } else {
            panic!("expected [x, y] from move_towards2");
        }

        let angle = interp
            .eval_builtin(
                "math::angle_to",
                vec![
                    Value::F32(0.0),
                    Value::F32(0.0),
                    Value::F32(0.0),
                    Value::F32(1.0),
                ],
            )
            .unwrap();
        assert!(matches!(angle, Value::F32(v) if (v - std::f32::consts::FRAC_PI_2).abs() < 1e-6));

        interp
            .eval_builtin("math::random_seed", vec![Value::I64(123)])
            .unwrap();
        let unit = interp.eval_builtin("math::rand_unit2", vec![]).unwrap();
        if let Value::Array(v) = unit {
            let v = v.lock().unwrap();
            let x = match v[0] {
                Value::F32(n) => n,
                _ => panic!("x should be f32"),
            };
            let y = match v[1] {
                Value::F32(n) => n,
                _ => panic!("y should be f32"),
            };
            let len = (x * x + y * y).sqrt();
            assert!((len - 1.0).abs() < 1e-5);
        } else {
            panic!("expected unit vector");
        }
    }

    #[test]
    fn test_sim_spatial_queries() {
        let mut interp = mk_interp();
        let world_id = match interp
            .eval_builtin("sim::world", vec![Value::F32(0.016), Value::I64(123)])
            .unwrap()
        {
            Value::I64(id) => id,
            _ => panic!("expected world id"),
        };

        let mk_ent = |x: f32, y: f32| {
            let mut entity = FxHashMap::default();
            entity.insert(
                "position".into(),
                Value::Array(Arc::new(Mutex::new(vec![Value::F32(x), Value::F32(y)]))),
            );
            entity.insert(
                "velocity".into(),
                Value::Array(Arc::new(Mutex::new(vec![Value::F32(0.0), Value::F32(0.0)]))),
            );
            Value::HashMap(Arc::new(Mutex::new(entity)))
        };
        let _ = interp
            .eval_builtin("sim::spawn", vec![Value::I64(world_id), mk_ent(0.0, 0.0)])
            .unwrap();
        let e2 = match interp
            .eval_builtin("sim::spawn", vec![Value::I64(world_id), mk_ent(2.0, 0.0)])
            .unwrap()
        {
            Value::I64(id) => id,
            _ => panic!("expected entity id"),
        };

        let count = interp
            .eval_builtin("sim::entity_count", vec![Value::I64(world_id)])
            .unwrap();
        assert!(matches!(count, Value::I64(2)));

        let near = interp
            .eval_builtin(
                "sim::nearest_entity",
                vec![
                    Value::I64(world_id),
                    Value::Array(Arc::new(Mutex::new(vec![Value::F32(1.8), Value::F32(0.0)]))),
                    Value::F32(10.0),
                ],
            )
            .unwrap();
        assert!(matches!(near, Value::I64(id) if id == e2));

        let qr = interp
            .eval_builtin(
                "sim::query_radius",
                vec![
                    Value::I64(world_id),
                    Value::Array(Arc::new(Mutex::new(vec![Value::F32(0.0), Value::F32(0.0)]))),
                    Value::F32(1.0),
                ],
            )
            .unwrap();
        match qr {
            Value::Array(ids) => assert_eq!(ids.lock().unwrap().len(), 1),
            _ => panic!("expected id array"),
        }
    }

    #[test]
    fn test_sys_level_builtins_io_env_exec() {
        let mut interp = mk_interp();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tmp_root = std::env::temp_dir().join(format!("jules_sys_{pid}_{nanos}"));
        let tmp_root_s = tmp_root.to_string_lossy().to_string();
        assert!(matches!(
            interp.eval_builtin("sys::os", vec![]).unwrap(),
            Value::Str(_)
        ));
        assert!(matches!(
            interp.eval_builtin("sys::arch", vec![]).unwrap(),
            Value::Str(_)
        ));
        assert!(matches!(
            interp.eval_builtin("sys::temp_dir", vec![]).unwrap(),
            Value::Str(_)
        ));

        assert!(matches!(
            interp
                .eval_builtin("sys::mkdir", vec![Value::Str(tmp_root_s.clone())])
                .unwrap(),
            Value::Bool(true)
        ));

        let bytes = Value::Array(Arc::new(Mutex::new(vec![
            Value::U8(0),
            Value::U8(1),
            Value::U8(2),
            Value::U8(255),
        ])));
        let bin_path = tmp_root.join("blob.bin").to_string_lossy().to_string();
        assert!(matches!(
            interp
                .eval_builtin(
                    "sys::write_bytes",
                    vec![Value::Str(bin_path.clone()), bytes.clone()]
                )
                .unwrap(),
            Value::Bool(true)
        ));
        let read_back = interp
            .eval_builtin("sys::read_bytes", vec![Value::Str(bin_path)])
            .unwrap();
        if let Value::Array(arr) = read_back {
            let got: Vec<u8> = arr
                .lock()
                .unwrap()
                .iter()
                .map(|v| match v {
                    Value::U8(b) => *b,
                    _ => panic!("expected byte values"),
                })
                .collect();
            assert_eq!(got, vec![0, 1, 2, 255]);
        } else {
            panic!("expected byte array");
        }

        let env_key = format!("JULES_TEST_KEY_{pid}_{nanos}");
        interp
            .eval_builtin(
                "sys::env_set",
                vec![Value::Str(env_key.clone()), Value::Str("abc".into())],
            )
            .unwrap();
        let env_val = interp
            .eval_builtin("sys::env_get", vec![Value::Str(env_key.clone())])
            .unwrap();
        assert!(matches!(env_val, Value::Some(v) if matches!(*v, Value::Str(ref s) if s == "abc")));
        interp
            .eval_builtin("sys::env_remove", vec![Value::Str(env_key.clone())])
            .unwrap();
        let env_missing = interp
            .eval_builtin("sys::env_get", vec![Value::Str(env_key)])
            .unwrap();
        assert!(matches!(env_missing, Value::None));

        let exec_result = interp
            .eval_builtin("sys::exec", vec![Value::Str("printf jules_sys_ok".into())])
            .unwrap();
        if let Value::HashMap(map) = exec_result {
            let m = map.lock().unwrap();
            assert!(matches!(m.get("ok"), Some(Value::Bool(true))));
            assert!(matches!(m.get("stdout"), Some(Value::Str(s)) if s == "jules_sys_ok"));
        } else {
            panic!("expected hashmap for exec result");
        }
        let exec_argv = interp
            .eval_builtin(
                "sys::exec_argv",
                vec![
                    Value::Str("printf".into()),
                    Value::Array(Arc::new(Mutex::new(vec![Value::Str(" argv_ok".into())]))),
                ],
            )
            .unwrap();
        if let Value::HashMap(map) = exec_argv {
            let m = map.lock().unwrap();
            assert!(matches!(m.get("ok"), Some(Value::Bool(true))));
            assert!(matches!(m.get("stdout"), Some(Value::Str(s)) if s == " argv_ok"));
        } else {
            panic!("expected hashmap for exec_argv result");
        }

        let list_result = interp
            .eval_builtin("sys::list_dir", vec![Value::Str(tmp_root_s.clone())])
            .unwrap();
        if let Value::Array(arr) = list_result {
            let names: Vec<String> = arr
                .lock()
                .unwrap()
                .iter()
                .map(|v| match v {
                    Value::Str(s) => s.clone(),
                    _ => panic!("expected string entry"),
                })
                .collect();
            assert!(names.iter().any(|n| n == "blob.bin"));
        } else {
            panic!("expected directory listing array");
        }
        let metadata = interp
            .eval_builtin(
                "sys::metadata",
                vec![Value::Str(
                    tmp_root.join("blob.bin").to_string_lossy().to_string(),
                )],
            )
            .unwrap();
        if let Value::HashMap(map) = metadata {
            let m = map.lock().unwrap();
            assert!(matches!(m.get("is_file"), Some(Value::Bool(true))));
            assert!(matches!(m.get("len"), Some(Value::I64(4))));
        } else {
            panic!("expected metadata hashmap");
        }

        let copied_path = tmp_root.join("copy.bin").to_string_lossy().to_string();
        let copied_bytes = interp
            .eval_builtin(
                "sys::copy",
                vec![
                    Value::Str(tmp_root.join("blob.bin").to_string_lossy().to_string()),
                    Value::Str(copied_path.clone()),
                ],
            )
            .unwrap();
        assert!(matches!(copied_bytes, Value::I64(4)));
        assert!(matches!(
            interp
                .eval_builtin(
                    "sys::rename",
                    vec![
                        Value::Str(copied_path),
                        Value::Str(tmp_root.join("moved.bin").to_string_lossy().to_string())
                    ]
                )
                .unwrap(),
            Value::Bool(true)
        ));

        assert!(matches!(
            interp
                .eval_builtin("sys::rmdir", vec![Value::Str(tmp_root_s.clone())])
                .unwrap(),
            Value::Bool(true)
        ));
        assert!(matches!(
            interp
                .eval_builtin("sys::remove_path", vec![Value::Str(tmp_root_s)])
                .unwrap(),
            Value::Bool(false)
        ));
    }

    #[test]
    fn test_interp_return_propagation() {
        let mut i = mk_interp();
        let mut env = Env::new();
        let block = Block {
            span: sp(),
            stmts: vec![
                Stmt::Return {
                    span: sp(),
                    value: Some(Expr::IntLit {
                        span: sp(),
                        value: 42,
                    }),
                },
                Stmt::Expr {
                    span: sp(),
                    has_semi: true,
                    expr: Expr::IntLit {
                        span: sp(),
                        value: 0,
                    },
                },
            ],
            tail: None,
        };
        let r = i.eval_block(&block, &mut env).unwrap();
        assert!(matches!(r, Value::Return(v) if matches!(*v, Value::I32(42))));
    }

    #[test]
    fn test_interp_matmul_tensors() {
        let mut i = mk_interp();
        let mut env = Env::new();
        let a = Tensor::from_data(vec![2, 3], vec![1.0; 6]);
        let b = Tensor::from_data(vec![3, 2], vec![1.0; 6]);
        env.set_local("A", Value::Tensor(Arc::new(RwLock::new(a))));
        env.set_local("B", Value::Tensor(Arc::new(RwLock::new(b))));
        let e = Expr::MatMul {
            span: sp(),
            lhs: Box::new(Expr::Ident {
                span: sp(),
                name: "A".into(),
            }),
            rhs: Box::new(Expr::Ident {
                span: sp(),
                name: "B".into(),
            }),
        };
        let result = i.eval_expr(&e, &mut env).unwrap();
        if let Value::Tensor(t) = result {
            assert_eq!(t.read().unwrap().shape, vec![2, 2]);
        } else {
            panic!("expected tensor");
        }
    }

    #[test]
    fn test_interp_range_for_loop() {
        let mut i = mk_interp();
        let mut env = Env::new();
        // Accumulate sum of 0..5 into `acc`.
        let block = Block {
            span: sp(),
            stmts: vec![
                Stmt::Let {
                    span: sp(),
                    pattern: Pattern::Ident {
                        span: sp(),
                        name: "acc".into(),
                        mutable: true,
                    },
                    ty: None,
                    init: Some(Expr::IntLit {
                        span: sp(),
                        value: 0,
                    }),
                    mutable: true,
                },
                Stmt::ForIn {
                    span: sp(),
                    pattern: Pattern::Ident {
                        span: sp(),
                        name: "i".into(),
                        mutable: false,
                    },
                    iter: Expr::Range {
                        span: sp(),
                        lo: Some(Box::new(Expr::IntLit {
                            span: sp(),
                            value: 0,
                        })),
                        hi: Some(Box::new(Expr::IntLit {
                            span: sp(),
                            value: 5,
                        })),
                        inclusive: false,
                    },
                    body: Block {
                        span: sp(),
                        stmts: vec![Stmt::Expr {
                            span: sp(),
                            has_semi: true,
                            expr: Expr::Assign {
                                span: sp(),
                                op: AssignOpKind::AddAssign,
                                target: Box::new(Expr::Ident {
                                    span: sp(),
                                    name: "acc".into(),
                                }),
                                value: Box::new(Expr::Ident {
                                    span: sp(),
                                    name: "i".into(),
                                }),
                            },
                        }],
                        tail: None,
                    },
                    label: None,
                },
            ],
            tail: Some(Box::new(Expr::Ident {
                span: sp(),
                name: "acc".into(),
            })),
        };
        let result = i.eval_block(&block, &mut env).unwrap();
        assert!(
            matches!(result, Value::I32(10)),
            "0+1+2+3+4 = 10, got {result}"
        );
    }

    #[test]
    fn test_interp_string_for_loop() {
        let mut i = mk_interp();
        let mut env = Env::new();
        let block = Block {
            span: sp(),
            stmts: vec![
                Stmt::Let {
                    span: sp(),
                    pattern: Pattern::Ident {
                        span: sp(),
                        name: "acc".into(),
                        mutable: true,
                    },
                    ty: None,
                    init: Some(Expr::IntLit {
                        span: sp(),
                        value: 0,
                    }),
                    mutable: true,
                },
                Stmt::ForIn {
                    span: sp(),
                    pattern: Pattern::Ident {
                        span: sp(),
                        name: "ch".into(),
                        mutable: false,
                    },
                    iter: Expr::StrLit {
                        span: sp(),
                        value: "abc".into(),
                    },
                    body: Block {
                        span: sp(),
                        stmts: vec![Stmt::Expr {
                            span: sp(),
                            has_semi: true,
                            expr: Expr::Assign {
                                span: sp(),
                                op: AssignOpKind::AddAssign,
                                target: Box::new(Expr::Ident {
                                    span: sp(),
                                    name: "acc".into(),
                                }),
                                value: Box::new(Expr::IntLit {
                                    span: sp(),
                                    value: 1,
                                }),
                            },
                        }],
                        tail: None,
                    },
                    label: None,
                },
            ],
            tail: Some(Box::new(Expr::Ident {
                span: sp(),
                name: "acc".into(),
            })),
        };
        let result = i.eval_block(&block, &mut env).unwrap();
        assert!(
            matches!(result, Value::I32(3)),
            "expected 3 chars, got {result}"
        );
    }

    // ── Neural network forward pass ───────────────────────────────────────────

    #[test]
    fn test_nn_forward_pass() {
        let decl = ModelDecl {
            span: sp(),
            attrs: vec![],
            name: "TestNet".into(),
            layers: vec![
                ModelLayer::Input {
                    span: sp(),
                    size: 4,
                },
                ModelLayer::Dense {
                    span: sp(),
                    units: 8,
                    activation: Activation::Relu,
                    bias: true,
                },
                ModelLayer::Output {
                    span: sp(),
                    units: 2,
                    activation: Activation::Softmax,
                },
            ],
            device: ModelDevice::Cpu,
            optimizer: None,
        };
        let mut model = NnModel::from_decl(&decl);
        model.training = false;
        let input = Tensor::from_data(vec![1, 4], vec![1.0, 0.5, -1.0, 0.0]);
        let out = model.forward(input).unwrap();
        assert_eq!(out.shape, vec![1, 2]);
        // Softmax: outputs should sum to 1.
        let sum: f32 = out.cpu_data().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "output should be softmax, sum={sum}"
        );
    }

    #[test]
    fn test_nn_dropout_training_mode() {
        // In inference mode, dropout should be a no-op.
        let decl = ModelDecl {
            span: sp(),
            attrs: vec![],
            name: "DropNet".into(),
            layers: vec![
                ModelLayer::Input {
                    span: sp(),
                    size: 4,
                },
                ModelLayer::Dropout {
                    span: sp(),
                    rate: 0.0,
                }, // rate=0 → no drop
                ModelLayer::Output {
                    span: sp(),
                    units: 2,
                    activation: Activation::Linear,
                },
            ],
            device: ModelDevice::Cpu,
            optimizer: None,
        };
        let mut model = NnModel::from_decl(&decl);
        let input = Tensor::from_data(vec![1, 4], vec![1.0; 4]);
        let out = model.forward(input).unwrap();
        assert_eq!(out.shape, vec![1, 2]);
    }

    // ── Vec swizzle ───────────────────────────────────────────────────────────

    #[test]
    fn test_swizzle_x() {
        assert!(matches!(
            swizzle_vec(&[1.0, 2.0, 3.0], "x"),
            Ok(Value::F32(1.0))
        ));
    }

    #[test]
    fn test_swizzle_xyz() {
        assert!(matches!(
            swizzle_vec(&[1.0, 2.0, 3.0], "xyz"),
            Ok(Value::Vec3(_))
        ));
    }

    #[test]
    fn test_swizzle_invalid() {
        assert!(swizzle_vec(&[1.0, 2.0], "q").is_err());
    }

    // ── Scope / environment ───────────────────────────────────────────────────

    #[test]
    fn test_env_scoping() {
        let mut env = Env::new();
        env.set_local("x", Value::I32(1));
        env.push();
        env.set_local("x", Value::I32(2));
        assert!(matches!(env.get("x"), Some(Value::I32(2))));
        env.pop();
        assert!(matches!(env.get("x"), Some(Value::I32(1))));
    }

    // ── Matrix multiply ───────────────────────────────────────────────────────

    #[test]
    fn test_mat3_identity_mul() {
        let id = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let a = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let r = mat3_mul(a, id);
        assert_eq!(r, a);
    }

    #[test]
    fn test_mat3_vec3_mul() {
        let id = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let v = [3.0_f32, 5.0, 7.0];
        assert_eq!(mat3_vec3_mul(id, v), v);
    }

    // ── Training stats ────────────────────────────────────────────────────────

    #[test]
    fn test_training_stats_default() {
        let s = TrainingStats::default();
        assert_eq!(s.total_steps, 0);
        assert!(s.episode_rewards.is_empty());
    }

    // ── Tensor flat index ──────────────────────────────────────────────────────

    #[test]
    fn test_tensor_flat_index_2d() {
        let shape = vec![3, 4];
        let idx = vec![Value::I32(1), Value::I32(2)];
        assert_eq!(tensor_flat_index(&shape, &idx).unwrap(), 1 * 4 + 2);
    }

    #[test]
    fn test_tensor_flat_index_out_of_bounds() {
        let shape = vec![2, 2];
        let idx = vec![Value::I32(0), Value::I32(5)]; // col 5 ≥ 2
        assert!(tensor_flat_index(&shape, &idx).is_err());
    }
}
