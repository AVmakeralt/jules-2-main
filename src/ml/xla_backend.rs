// =========================================================================
// XLA Backend for Jules ML Engine
// Provides XLA (Accelerated Linear Algebra) compilation and execution
// =========================================================================

use crate::ml::ml_engine::{ComputationGraph, ComputeNode, Operation, Tensor};
use crate::ml::superopt_xla_bridge::SuperoptXlaBridge;
use rustc_hash::FxHashMap;
use std::sync::{Arc, Mutex};

#[cfg(feature = "xla")]
use xla_ffi as xla;

/// XLA computation handle
#[derive(Debug)]
pub struct XlaComputation {
    pub id: u64,
    pub hlo_ir: String, // HLO (High Level Optimizer) IR representation
    pub compiled: bool,
    /// Structured CPU-executable operations for fallback execution.
    /// These are populated during compile_graph and used by cpu_execute
    /// when native XLA is not available.
    pub cpu_ops: Vec<CpuOp>,
    #[cfg(feature = "xla")]
    pub native_handle: Option<*mut std::ffi::c_void>, // Native XLA computation handle
}

impl Clone for XlaComputation {
    fn clone(&self) -> Self {
        XlaComputation {
            id: self.id,
            hlo_ir: self.hlo_ir.clone(),
            compiled: self.compiled,
            cpu_ops: self.cpu_ops.clone(),
            #[cfg(feature = "xla")]
            native_handle: None, // Do NOT clone raw pointers — avoid double-free
        }
    }
}

// =========================================================================
// CPU Fallback Operation Types
// =========================================================================

/// A structured CPU-executable operation derived from the computation graph.
/// Used by `cpu_execute` when native XLA is not available.
#[derive(Debug, Clone)]
pub struct CpuOp {
    /// Name used to store/retrieve the result tensor
    pub result_name: String,
    /// What operation to perform
    pub op_kind: CpuOpKind,
    /// Names of operand tensors (must be computed before this op)
    pub operand_names: Vec<String>,
    /// Output shape
    pub output_shape: Vec<usize>,
}

/// CPU operation kinds that actually compute results.
#[derive(Debug, Clone)]
pub enum CpuOpKind {
    /// Load the n-th input tensor
    Parameter(usize),
    /// A constant tensor with given data
    Constant(Vec<f32>),
    /// Element-wise addition
    Add,
    /// Element-wise subtraction
    Sub,
    /// Element-wise multiplication
    Mul,
    /// Element-wise division
    Div,
    /// Matrix multiplication (dot)
    Dot,
    /// ReLU activation
    Relu,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Numerically stable softmax
    Softmax,
    /// Sum reduction over all elements
    ReduceSum,
    /// Mean reduction over all elements
    ReduceMean,
    /// Reshape to new shape
    Reshape { new_shape: Vec<usize> },
    /// Transpose with given permutation
    Transpose { axes: Vec<usize> },
    /// No-op / identity (copy)
    Identity,
}

// =========================================================================
// XLA Configuration
// =========================================================================

/// XLA backend configuration
#[derive(Debug, Clone)]
pub struct XlaConfig {
    pub device: XlaDevice,
    pub optimization_level: u8, // 0-3, higher = more optimization
    pub enable_autotuning: bool,
    pub enable_superopt: bool, // Enable superoptimizer integration
}

#[derive(Debug, Clone, PartialEq)]
pub enum XlaDevice {
    Cpu,
    Gpu(u32), // GPU device ID
    Tpu,
}

impl Default for XlaConfig {
    fn default() -> Self {
        Self {
            device: XlaDevice::Cpu,
            optimization_level: 2,
            enable_autotuning: true,
            enable_superopt: true,
        }
    }
}

// =========================================================================
// XLA Backend
// =========================================================================

/// XLA backend executor
pub struct XlaBackend {
    config: XlaConfig,
    computations: FxHashMap<u64, XlaComputation>,
    next_id: u64,
    superopt_bridge: Option<SuperoptXlaBridge>,
    #[cfg(feature = "xla")]
    client: Option<Arc<Mutex<XlaClient>>>,
    #[cfg(not(feature = "xla"))]
    _client: Option<Arc<Mutex<()>>>, // Placeholder when XLA FFI is disabled
    /// Whether libxla.so was found at runtime via dlopen
    #[allow(dead_code)] // Used when feature = "xla" is enabled
    xla_native_available: bool,
}

#[cfg(feature = "xla")]
struct XlaClient {
    device: XlaDevice,
    // Native XLA client handle would go here
    native_client: Option<*mut std::ffi::c_void>,
}

#[cfg(feature = "xla")]
impl XlaClient {
    fn new(device: XlaDevice) -> Self {
        // In production, this would initialize the actual XLA client via FFI
        Self {
            device,
            native_client: None,
        }
    }
}

impl XlaBackend {
    pub fn new(config: XlaConfig) -> Self {
        #[cfg(feature = "xla")]
        let client = Some(Arc::new(Mutex::new(XlaClient::new(config.device.clone()))));
        #[cfg(not(feature = "xla"))]
        let _client = None;

        let superopt_bridge = if config.enable_superopt {
            Some(SuperoptXlaBridge::new())
        } else {
            None
        };

        let xla_native_available = Self::xla_library_available();

        Self {
            config,
            computations: FxHashMap::default(),
            next_id: 0,
            superopt_bridge,
            #[cfg(feature = "xla")]
            client,
            #[cfg(not(feature = "xla"))]
            _client,
            xla_native_available,
        }
    }

    pub fn with_default_device() -> Self {
        Self::new(XlaConfig::default())
    }

    /// Check if an XLA native library is available at runtime.
    ///
    /// Probes for libxla.so, xla_extension.so, or libxla_c.so via dlopen.
    /// Returns true if any of these libraries can be loaded.
    fn xla_library_available() -> bool {
        #[cfg(unix)]
        {
            extern "C" {
                fn dlopen(filename: *const i8, flags: i32) -> *mut std::ffi::c_void;
                fn dlclose(handle: *mut std::ffi::c_void) -> i32;
            }
            const RTLD_LAZY: i32 = 1;
            let libs: &[&[u8]] = &[
                b"libxla.so\0",
                b"libxla_c.so\0",
                b"xla_extension.so\0",
            ];
            unsafe {
                for lib_name in libs {
                    let handle = dlopen(lib_name.as_ptr() as *const i8, RTLD_LAZY);
                    if !handle.is_null() {
                        dlclose(handle);
                        return true;
                    }
                }
            }
            false
        }
        #[cfg(not(unix))]
        {
            false
        }
    }

    /// Convert Jules computation graph to XLA HLO IR
    pub fn compile_graph(&mut self, graph: &ComputationGraph) -> Result<XlaComputation, String> {
        let hlo_ir = self.graph_to_hlo(graph)?;

        // Build structured CPU ops from the graph for fallback execution
        let cpu_ops = self.graph_to_cpu_ops(graph)?;

        let id = self.next_id;
        self.next_id += 1;

        // Apply superoptimizer optimizations if enabled
        let optimized_hlo = if self.config.enable_superopt {
            self.apply_superoptimizer(&hlo_ir)?
        } else {
            hlo_ir.clone()
        };

        #[cfg(feature = "xla")]
        let native_handle = if self.xla_native_available {
            self.compile_native_xla(&optimized_hlo)?
        } else {
            None
        };

        #[cfg(not(feature = "xla"))]
        let _native_handle: Option<*mut std::ffi::c_void> = None;

        let computation = XlaComputation {
            id,
            hlo_ir: optimized_hlo,
            compiled: true,
            cpu_ops,
            #[cfg(feature = "xla")]
            native_handle,
        };

        self.computations.insert(id, computation.clone());
        Ok(computation)
    }

    /// Execute a compiled XLA computation
    pub fn execute(&self, comp: &XlaComputation, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        if !comp.compiled {
            return Err("Computation not compiled".to_string());
        }

        #[cfg(feature = "xla")]
        {
            if let Some(_handle) = comp.native_handle {
                // Real XLA execution via FFI
                self.execute_native_xla(comp, inputs)
            } else {
                // Fallback to CPU execution that actually computes
                self.cpu_execute(comp, inputs)
            }
        }

        #[cfg(not(feature = "xla"))]
        {
            // CPU execution when XLA FFI is disabled — actually computes results
            self.cpu_execute(comp, inputs)
        }
    }

    /// Apply superoptimizer optimizations to HLO IR
    #[cfg(feature = "xla")]
    fn apply_superoptimizer(&self, hlo_ir: &str) -> Result<String, String> {
        if let Some(bridge) = &self.superopt_bridge {
            bridge.optimize_hlo_ir(hlo_ir)
        } else {
            Ok(hlo_ir.to_string())
        }
    }

    /// Apply superoptimizer optimizations to HLO IR (when XLA FFI is disabled)
    #[cfg(not(feature = "xla"))]
    fn apply_superoptimizer(&self, hlo_ir: &str) -> Result<String, String> {
        if let Some(bridge) = &self.superopt_bridge {
            bridge.optimize_hlo_ir(hlo_ir)
        } else {
            Ok(hlo_ir.to_string())
        }
    }

    /// Compile HLO IR to native XLA computation.
    ///
    /// Checks if libxla.so is available. If not, returns None so that
    /// the CPU fallback execution path is used instead.
    #[cfg(feature = "xla")]
    fn compile_native_xla(&self, hlo_ir: &str) -> Result<Option<*mut std::ffi::c_void>, String> {
        // Attempt to compile via XLA FFI if the native library was found.
        // In production, this would:
        //   1. dlsym the XLA compile function from the loaded library
        //   2. Call xla::compile_computation(hlo_ir, &self.config)
        //   3. Return the compiled handle
        // For now, even if the library exists, we don't have the FFI bridge
        // to actually invoke it, so fall back to CPU execution.
        let _ = hlo_ir;
        Ok(None)
    }

    /// Execute native XLA computation
    #[cfg(feature = "xla")]
    fn execute_native_xla(&self, comp: &XlaComputation, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        // In production, this would:
        // 1. Transfer inputs to device via XLA FFI
        // 2. Execute compiled XLA computation
        // 3. Transfer outputs back to host
        // For now, fall back to CPU execution that actually computes
        self.cpu_execute(comp, inputs)
    }

    // =====================================================================
    // CPU Execution — actually computes results (no more identity returns)
    // =====================================================================

    /// Execute the computation graph on CPU using the structured ops.
    ///
    /// This interprets the CPU ops derived from the original computation graph
    /// and computes actual results for every operation.
    fn cpu_execute(&self, comp: &XlaComputation, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        // Map from result_name -> computed Tensor
        let mut values: FxHashMap<String, Tensor> = FxHashMap::default();

        for op in &comp.cpu_ops {
            let result = match &op.op_kind {
                CpuOpKind::Parameter(idx) => {
                    let idx = *idx;
                    if idx >= inputs.len() {
                        return Err(format!(
                            "cpu_execute: parameter index {} out of range ({} inputs)",
                            idx,
                            inputs.len()
                        ));
                    }
                    inputs[idx].clone()
                }
                CpuOpKind::Constant(data) => Tensor {
                    shape: op.output_shape.clone(),
                    data: data.clone(),
                },
                CpuOpKind::Add => {
                    let a = Self::lookup_operand(&values, &op.operand_names, 0, "add")?;
                    let b = Self::lookup_operand(&values, &op.operand_names, 1, "add")?;
                    cpu_elementwise(&a, &b, |x, y| x + y)?
                }
                CpuOpKind::Sub => {
                    let a = Self::lookup_operand(&values, &op.operand_names, 0, "sub")?;
                    let b = Self::lookup_operand(&values, &op.operand_names, 1, "sub")?;
                    cpu_elementwise(&a, &b, |x, y| x - y)?
                }
                CpuOpKind::Mul => {
                    let a = Self::lookup_operand(&values, &op.operand_names, 0, "mul")?;
                    let b = Self::lookup_operand(&values, &op.operand_names, 1, "mul")?;
                    cpu_elementwise(&a, &b, |x, y| x * y)?
                }
                CpuOpKind::Div => {
                    let a = Self::lookup_operand(&values, &op.operand_names, 0, "div")?;
                    let b = Self::lookup_operand(&values, &op.operand_names, 1, "div")?;
                    cpu_elementwise(&a, &b, |x, y| {
                        if y == 0.0 { 0.0 } else { x / y }
                    })?
                }
                CpuOpKind::Dot => {
                    let a = Self::lookup_operand(&values, &op.operand_names, 0, "dot")?;
                    let b = Self::lookup_operand(&values, &op.operand_names, 1, "dot")?;
                    cpu_matmul(&a, &b)?
                }
                CpuOpKind::Relu => {
                    let a = Self::lookup_operand(&values, &op.operand_names, 0, "relu")?;
                    cpu_unary(&a, |x| x.max(0.0))
                }
                CpuOpKind::Sigmoid => {
                    let a = Self::lookup_operand(&values, &op.operand_names, 0, "sigmoid")?;
                    cpu_unary(&a, |x| 1.0 / (1.0 + (-x).exp()))
                }
                CpuOpKind::Tanh => {
                    let a = Self::lookup_operand(&values, &op.operand_names, 0, "tanh")?;
                    cpu_unary(&a, |x| x.tanh())
                }
                CpuOpKind::Softmax => {
                    let a = Self::lookup_operand(&values, &op.operand_names, 0, "softmax")?;
                    cpu_softmax(&a)?
                }
                CpuOpKind::ReduceSum => {
                    let a = Self::lookup_operand(&values, &op.operand_names, 0, "reduce-sum")?;
                    let sum = a.data.iter().sum();
                    Tensor {
                        shape: vec![1],
                        data: vec![sum],
                    }
                }
                CpuOpKind::ReduceMean => {
                    let a = Self::lookup_operand(&values, &op.operand_names, 0, "reduce-mean")?;
                    let sum: f32 = a.data.iter().sum();
                    let mean = sum / a.data.len().max(1) as f32;
                    Tensor {
                        shape: vec![1],
                        data: vec![mean],
                    }
                }
                CpuOpKind::Reshape { new_shape } => {
                    let a = Self::lookup_operand(&values, &op.operand_names, 0, "reshape")?;
                    let new_numel: usize = new_shape.iter().product();
                    if new_numel != a.data.len() {
                        return Err(format!(
                            "reshape: element count mismatch ({} -> {})",
                            a.data.len(),
                            new_numel
                        ));
                    }
                    Tensor {
                        shape: new_shape.clone(),
                        data: a.data.clone(),
                    }
                }
                CpuOpKind::Transpose { axes } => {
                    let a = Self::lookup_operand(&values, &op.operand_names, 0, "transpose")?;
                    cpu_transpose(&a, axes)?
                }
                CpuOpKind::Identity => {
                    Self::lookup_operand(&values, &op.operand_names, 0, "identity")?
                }
            };
            values.insert(op.result_name.clone(), result);
        }

        // Collect all non-parameter, non-constant results as outputs
        let mut outputs = Vec::new();
        for op in &comp.cpu_ops {
            match &op.op_kind {
                CpuOpKind::Parameter(_) | CpuOpKind::Constant(_) => continue,
                _ => {
                    if let Some(tensor) = values.get(&op.result_name) {
                        outputs.push(tensor.clone());
                    }
                }
            }
        }

        // If no outputs were produced (e.g. graph is all inputs/constants),
        // return the last computed value or the inputs
        if outputs.is_empty() && !inputs.is_empty() {
            return Ok(inputs.to_vec());
        }

        Ok(outputs)
    }

    /// Helper to look up an operand by name and index
    fn lookup_operand<'a>(
        values: &'a FxHashMap<String, Tensor>,
        operand_names: &[String],
        idx: usize,
        op_name: &str,
    ) -> Result<Tensor, String> {
        let name = operand_names.get(idx).ok_or_else(|| {
            format!(
                "cpu_execute: operand index {} missing for {} operation",
                idx, op_name
            )
        })?;
        values.get(name).cloned().ok_or_else(|| {
            format!(
                "cpu_execute: operand '{}' not yet computed (needed for {})",
                name, op_name
            )
        })
    }

    // =====================================================================
    // Graph to CPU Ops Conversion
    // =====================================================================

    /// Convert a ComputationGraph into a sequence of CPU-executable operations.
    ///
    /// The operations are topologically sorted (by node ID, which is assumed
    /// to be monotonically increasing for forward declarations).
    fn graph_to_cpu_ops(&self, graph: &ComputationGraph) -> Result<Vec<CpuOp>, String> {
        let mut ops = Vec::new();
        let mut input_param_idx = 0;

        // Sort nodes by ID for deterministic ordering
        let mut node_ids: Vec<_> = graph.nodes.keys().cloned().collect();
        node_ids.sort();

        // Build a map from node ID to its result name
        let mut id_to_name: FxHashMap<u64, String> = FxHashMap::default();

        for node_id in node_ids {
            let node = graph.nodes.get(&node_id).ok_or("Missing node")?;
            let result_name = match &node.op {
                Operation::Input => format!("param_{}", input_param_idx),
                Operation::Constant => format!("const_{}", node_id),
                _ => format!("node_{}", node_id),
            };

            // Build operand names from input node IDs
            let operand_names: Vec<String> = node
                .inputs
                .iter()
                .map(|&input_id| {
                    id_to_name
                        .get(&input_id)
                        .cloned()
                        .unwrap_or_else(|| format!("unknown_{}", input_id))
                })
                .collect();

            let op_kind = match &node.op {
                Operation::Input => {
                    let idx = input_param_idx;
                    input_param_idx += 1;
                    CpuOpKind::Parameter(idx)
                }
                Operation::Constant => CpuOpKind::Constant(node.value.data.clone()),
                Operation::Add => CpuOpKind::Add,
                Operation::Sub => CpuOpKind::Sub,
                Operation::Mul => CpuOpKind::Mul,
                Operation::Div => CpuOpKind::Div,
                Operation::MatMul => CpuOpKind::Dot,
                Operation::ReLU => CpuOpKind::Relu,
                Operation::Sigmoid => CpuOpKind::Sigmoid,
                Operation::Tanh => CpuOpKind::Tanh,
                Operation::Softmax => CpuOpKind::Softmax,
                Operation::Sum => CpuOpKind::ReduceSum,
                Operation::Mean => CpuOpKind::ReduceMean,
                Operation::Reshape { new_shape } => CpuOpKind::Reshape {
                    new_shape: new_shape.clone(),
                },
                Operation::Transpose { axes } => CpuOpKind::Transpose {
                    axes: axes.clone(),
                },
            };

            id_to_name.insert(node_id, result_name.clone());

            ops.push(CpuOp {
                result_name,
                op_kind,
                operand_names,
                output_shape: node.value.shape.clone(),
            });
        }

        Ok(ops)
    }

    // =====================================================================
    // HLO IR Generation (for documentation / future native XLA use)
    // =====================================================================

    /// Convert ComputationGraph to HLO IR string
    fn graph_to_hlo(&self, graph: &ComputationGraph) -> Result<String, String> {
        let mut hlo = String::new();
        hlo.push_str("HloModule jules_computation {\n");

        // Sort nodes by ID for deterministic output
        let mut node_ids: Vec<_> = graph.nodes.keys().cloned().collect();
        node_ids.sort();

        for node_id in node_ids {
            if let Some(node) = graph.nodes.get(&node_id) {
                hlo.push_str(&self.node_to_hlo(node, graph)?);
            }
        }

        hlo.push_str("}\n");
        Ok(hlo)
    }

    /// Convert a single ComputeNode to HLO
    fn node_to_hlo(&self, node: &ComputeNode, _graph: &ComputationGraph) -> Result<String, String> {
        let op_name = match &node.op {
            Operation::Input => format!("input_{}", node.id),
            Operation::Constant => format!("constant_{}", node.id),
            Operation::Add => "add".to_string(),
            Operation::Sub => "subtract".to_string(),
            Operation::Mul => "multiply".to_string(),
            Operation::Div => "divide".to_string(),
            Operation::MatMul => "dot".to_string(),
            Operation::ReLU => "relu".to_string(),
            Operation::Sigmoid => "sigmoid".to_string(),
            Operation::Tanh => "tanh".to_string(),
            Operation::Softmax => "softmax".to_string(),
            Operation::Sum => "reduce-sum".to_string(),
            Operation::Mean => "reduce-mean".to_string(),
            Operation::Reshape { .. } => "reshape".to_string(),
            Operation::Transpose { .. } => "transpose".to_string(),
        };

        let shape_str = self.shape_to_hlo(&node.value.shape);
        let mut hlo = format!("  {} = {}({})", op_name, self.op_to_hlo(&node.op), shape_str);

        if !node.inputs.is_empty() {
            hlo.push_str(" operands=");
            let input_strs: Vec<String> = node
                .inputs
                .iter()
                .map(|id| format!("input_{}", id))
                .collect();
            hlo.push_str(&input_strs.join(", "));
        }

        hlo.push_str("\n");
        Ok(hlo)
    }

    /// Convert Operation to HLO opcode
    fn op_to_hlo(&self, op: &Operation) -> &'static str {
        match op {
            Operation::Input => "parameter",
            Operation::Constant => "constant",
            Operation::Add => "add",
            Operation::Sub => "subtract",
            Operation::Mul => "multiply",
            Operation::Div => "divide",
            Operation::MatMul => "dot",
            Operation::ReLU => "maximum", // ReLU = max(x, 0)
            Operation::Sigmoid => "sigmoid",
            Operation::Tanh => "tanh",
            Operation::Softmax => "softmax",
            Operation::Sum => "reduce",
            Operation::Mean => "reduce",
            Operation::Reshape { .. } => "reshape",
            Operation::Transpose { .. } => "transpose",
        }
    }

    /// Convert shape to HLO format
    fn shape_to_hlo(&self, shape: &[usize]) -> String {
        if shape.is_empty() {
            "[]".to_string()
        } else {
            format!(
                "[{}]",
                shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }

    /// Get backend information
    pub fn backend_info(&self) -> BackendInfo {
        BackendInfo {
            name: "XLA".to_string(),
            device: format!("{:?}", self.config.device),
            optimization_level: self.config.optimization_level,
            autotuning_enabled: self.config.enable_autotuning,
            superopt_enabled: self.config.enable_superopt,
        }
    }
}

// =========================================================================
// CPU Computation Helpers — these actually compute, no identity/empty returns
// =========================================================================

/// Element-wise binary operation on two tensors
fn cpu_elementwise<F>(a: &Tensor, b: &Tensor, op: F) -> Result<Tensor, String>
where
    F: Fn(f32, f32) -> f32,
{
    if a.shape != b.shape {
        return Err(format!(
            "cpu_elementwise: shape mismatch {:?} vs {:?}",
            a.shape, b.shape
        ));
    }
    let data: Vec<f32> = a.data.iter().zip(&b.data).map(|(&x, &y)| op(x, y)).collect();
    Ok(Tensor {
        shape: a.shape.clone(),
        data,
    })
}

/// Unary operation on a tensor
fn cpu_unary<F>(a: &Tensor, op: F) -> Tensor
where
    F: Fn(f32) -> f32,
{
    let data: Vec<f32> = a.data.iter().map(|&x| op(x)).collect();
    Tensor {
        shape: a.shape.clone(),
        data,
    }
}

/// Matrix multiplication (dot) on 2D tensors.
/// Uses nested loops to actually compute the result.
fn cpu_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(format!(
            "cpu_matmul: expected 2D tensors, got {:?} and {:?}",
            a.shape.len(),
            b.shape.len()
        ));
    }
    let m = a.shape[0];
    let k = a.shape[1];
    let k2 = b.shape[0];
    let n = b.shape[1];
    if k != k2 {
        return Err(format!(
            "cpu_matmul: dimension mismatch ({}x{}) @ ({}x{})",
            m, k, k2, n
        ));
    }

    let mut data = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a.data[i * k + p] * b.data[p * n + j];
            }
            data[i * n + j] = acc;
        }
    }

    Ok(Tensor {
        shape: vec![m, n],
        data,
    })
}

/// Numerically stable softmax
fn cpu_softmax(a: &Tensor) -> Result<Tensor, String> {
    if a.data.is_empty() {
        return Err("cpu_softmax: empty tensor".into());
    }
    let max_val = a.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_vals: Vec<f32> = a.data.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let data: Vec<f32> = if sum > 0.0 {
        exp_vals.iter().map(|e| e / sum).collect()
    } else {
        // Uniform distribution if all inputs are -inf
        vec![1.0 / a.data.len() as f32; a.data.len()]
    };
    Ok(Tensor {
        shape: a.shape.clone(),
        data,
    })
}

/// Transpose a tensor according to the given axes permutation.
fn cpu_transpose(a: &Tensor, axes: &[usize]) -> Result<Tensor, String> {
    if axes.len() != a.shape.len() {
        return Err(format!(
            "cpu_transpose: axes length {} doesn't match tensor rank {}",
            axes.len(),
            a.shape.len()
        ));
    }

    let rank = a.shape.len();
    let new_shape: Vec<usize> = axes.iter().map(|&ax| a.shape[ax]).collect();

    // Compute strides for old and new shapes
    let mut old_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        old_strides[i] = old_strides[i + 1] * a.shape[i + 1];
    }

    let mut new_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    let total: usize = a.shape.iter().product();
    let mut data = vec![0.0f32; total];

    // For each element in the output, compute its source index
    for out_idx in 0..total {
        // Decompose out_idx into multi-dim coordinates using new_strides
        let mut in_multi = vec![0usize; rank];
        let mut remaining = out_idx;
        for i in 0..rank {
            in_multi[axes[i]] = remaining / new_strides[i];
            remaining %= new_strides[i];
        }
        // Compute linear source index
        let in_idx: usize = in_multi
            .iter()
            .zip(old_strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum();
        data[out_idx] = a.data[in_idx];
    }

    Ok(Tensor {
        shape: new_shape,
        data,
    })
}

// =========================================================================
// Public Types
// =========================================================================

#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub name: String,
    pub device: String,
    pub optimization_level: u8,
    pub autotuning_enabled: bool,
    pub superopt_enabled: bool,
}

/// Convenience function to compile and execute a graph in one step
pub fn compile_and_execute(
    graph: &ComputationGraph,
    inputs: &[Tensor],
) -> Result<Vec<Tensor>, String> {
    let mut backend = XlaBackend::with_default_device();
    let comp = backend.compile_graph(graph)?;
    backend.execute(&comp, inputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xla_backend_creation() {
        let backend = XlaBackend::with_default_device();
        let info = backend.backend_info();
        assert_eq!(info.name, "XLA");
    }

    #[test]
    fn test_simple_graph_compilation() {
        let mut graph = ComputationGraph {
            nodes: FxHashMap::default(),
            next_id: 0,
        };

        // Add a simple constant node
        let node = ComputeNode {
            id: 0,
            op: Operation::Constant,
            inputs: vec![],
            value: Tensor {
                shape: vec![2, 2],
                data: vec![1.0, 2.0, 3.0, 4.0],
            },
            gradient: None,
            requires_grad: false,
        };
        graph.nodes.insert(0, node);

        let mut backend = XlaBackend::with_default_device();
        let result = backend.compile_graph(&graph);
        assert!(result.is_ok());
        let comp = result.unwrap();
        assert!(!comp.cpu_ops.is_empty());
    }

    #[test]
    fn test_shape_to_hlo() {
        let backend = XlaBackend::with_default_device();
        assert_eq!(backend.shape_to_hlo(&[]), "[]");
        assert_eq!(backend.shape_to_hlo(&[2, 3]), "[2, 3]");
    }

    #[test]
    fn test_cpu_matmul() {
        let a = Tensor {
            shape: vec![2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let b = Tensor {
            shape: vec![2, 2],
            data: vec![5.0, 6.0, 7.0, 8.0],
        };
        let result = cpu_matmul(&a, &b).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_cpu_softmax() {
        let a = Tensor {
            shape: vec![3],
            data: vec![1.0, 2.0, 3.0],
        };
        let result = cpu_softmax(&a).unwrap();
        // Sum should be approximately 1.0
        let sum: f32 = result.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Largest probability should be for the last element
        assert!(result.data[2] > result.data[1]);
        assert!(result.data[1] > result.data[0]);
    }

    #[test]
    fn test_cpu_elementwise() {
        let a = Tensor {
            shape: vec![3],
            data: vec![1.0, 2.0, 3.0],
        };
        let b = Tensor {
            shape: vec![3],
            data: vec![4.0, 5.0, 6.0],
        };
        let result = cpu_elementwise(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_cpu_transpose() {
        let a = Tensor {
            shape: vec![2, 3],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let result = cpu_transpose(&a, &[1, 0]).unwrap();
        assert_eq!(result.shape, vec![3, 2]);
        assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_cpu_execute_with_operations() {
        // Build a graph: input -> relu -> output
        let mut graph = ComputationGraph {
            nodes: FxHashMap::default(),
            next_id: 0,
        };

        // Input node
        let input_node = ComputeNode {
            id: 0,
            op: Operation::Input,
            inputs: vec![],
            value: Tensor {
                shape: vec![2, 2],
                data: vec![0.0; 4],
            },
            gradient: None,
            requires_grad: false,
        };
        graph.nodes.insert(0, input_node);

        // ReLU node
        let relu_node = ComputeNode {
            id: 1,
            op: Operation::ReLU,
            inputs: vec![0],
            value: Tensor {
                shape: vec![2, 2],
                data: vec![0.0; 4],
            },
            gradient: None,
            requires_grad: false,
        };
        graph.nodes.insert(1, relu_node);

        let mut backend = XlaBackend::with_default_device();
        let comp = backend.compile_graph(&graph).unwrap();

        let input = Tensor {
            shape: vec![2, 2],
            data: vec![-1.0, 2.0, -3.0, 4.0],
        };

        let outputs = backend.execute(&comp, &[input]).unwrap();
        // The relu output should be [0.0, 2.0, 0.0, 4.0]
        assert!(!outputs.is_empty());
        let relu_output = &outputs[0];
        assert_eq!(relu_output.data, vec![0.0, 2.0, 0.0, 4.0]);
    }
}
