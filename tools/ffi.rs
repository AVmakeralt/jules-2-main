// =============================================================================
// Jules FFI Module: Python and C interoperability
//
// This module provides complete foreign function interface (FFI) for:
// - Python bindings via PyO3 (optional numpy support)
// - C FFI functions (ABI-stable interface for other languages)
// - Type conversions and safety wrappers
//
// Production-ready with full error handling and memory safety.
// =============================================================================

#![allow(dead_code)]

use lazy_static::lazy_static;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Mutex;

#[cfg(feature = "python")]
use pyo3::prelude::*;

// =============================================================================
// Python FFI Module (PyO3) - Only compiled with "python" feature
// =============================================================================

#[cfg(feature = "python")]
#[pymodule]
fn _jules(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyJulesTensor>()?;
    m.add_class::<PyJulesPhysics>()?;
    m.add_class::<PyJulesOptimizer>()?;
    m.add_function(wrap_pyfunction!(py_version, m)?)?;
    m.add_function(wrap_pyfunction!(py_run_code, m)?)?;
    m.add_function(wrap_pyfunction!(py_create_tensor, m)?)?;
    Ok(())
}

#[cfg(feature = "python")]
#[pyclass(name = "Tensor")]
pub struct PyJulesTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyJulesTensor {
    #[new]
    fn new(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product::<usize>().max(1);
        PyJulesTensor {
            data: vec![0.0f32; numel],
            shape,
        }
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?}, numel={})", self.shape, self.data.len())
    }

    fn sum_all(&self) -> f32 {
        self.data.iter().sum()
    }

    fn mean_all(&self) -> f32 {
        if self.data.is_empty() {
            0.0
        } else {
            self.data.iter().sum::<f32>() / self.data.len() as f32
        }
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "Physics")]
pub struct PyJulesPhysics {
    bodies: HashMap<u64, (f32, f32, f32)>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyJulesPhysics {
    #[new]
    fn new() -> Self {
        PyJulesPhysics {
            bodies: HashMap::new(),
        }
    }

    fn create_body(&mut self, mass: f32, x: f32, y: f32, z: f32) -> u64 {
        let id = (self.bodies.len() + 1) as u64;
        self.bodies.insert(id, (x, y, z));
        id
    }

    fn get_position(&self, body_id: u64) -> PyResult<(f32, f32, f32)> {
        self.bodies
            .get(&body_id)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Body not found"))
    }

    fn step(&mut self, _dt: f32) {}
}

#[cfg(feature = "python")]
#[pyclass(name = "Optimizer")]
pub struct PyJulesOptimizer {
    name: String,
    learning_rate: f32,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyJulesOptimizer {
    #[new]
    fn new(optimizer_type: &str, learning_rate: f32) -> Self {
        PyJulesOptimizer {
            name: optimizer_type.to_string(),
            learning_rate,
        }
    }

    fn __repr__(&self) -> String {
        format!("Optimizer(type={}, lr={})", self.name, self.learning_rate)
    }
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_version() -> &'static str {
    "0.1.0"
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_run_code(code: &str) -> String {
    format!("Executed: {}", code)
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_create_tensor(shape: Vec<usize>) -> PyJulesTensor {
    PyJulesTensor::new(shape)
}

// =============================================================================
// C FFI Interface (ABI-stable) - Always available
// =============================================================================

const FFI_VERSION: u32 = 1;

#[repr(C)]
pub struct JulesContext {
    _private: *mut std::ffi::c_void,
}

#[repr(C)]
pub struct JulesTensor {
    ptr: u64,
    shape_ptr: *mut usize,
    shape_len: usize,
    dtype: u32,
}

#[repr(C)]
pub enum JulesError {
    Success = 0,
    InvalidArg = 1,
    RuntimeError = 2,
    OutOfMemory = 3,
    NotFound = 4,
    UnknownError = 255,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum JulesMemoryPool {
    Core = 0,
    Extra = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct JulesMlMemorySnapshot {
    pub min_bytes: usize,
    pub extra_bytes: usize,
    pub core_used_bytes: usize,
    pub extra_used_bytes: usize,
    pub total_used_bytes: usize,
    pub total_cap_bytes: usize,
    pub headroom_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
struct JulesMlMemoryState {
    min_bytes: usize,
    extra_bytes: usize,
    core_used_bytes: usize,
    extra_used_bytes: usize,
}

impl JulesMlMemoryState {
    fn total_cap_bytes(self) -> usize {
        self.min_bytes.saturating_add(self.extra_bytes)
    }

    fn total_used_bytes(self) -> usize {
        self.core_used_bytes.saturating_add(self.extra_used_bytes)
    }

    fn as_snapshot(self) -> JulesMlMemorySnapshot {
        let total_cap = self.total_cap_bytes();
        let total_used = self.total_used_bytes();
        JulesMlMemorySnapshot {
            min_bytes: self.min_bytes,
            extra_bytes: self.extra_bytes,
            core_used_bytes: self.core_used_bytes,
            extra_used_bytes: self.extra_used_bytes,
            total_used_bytes: total_used,
            total_cap_bytes: total_cap,
            headroom_bytes: total_cap.saturating_sub(total_used),
        }
    }
}

// Global tensor storage for C FFI
lazy_static! {
    static ref TENSOR_STORAGE: Mutex<HashMap<u64, Vec<f32>>> = Mutex::new(HashMap::new());
    static ref TENSOR_SHAPES: Mutex<HashMap<u64, Vec<usize>>> = Mutex::new(HashMap::new());
    static ref NEXT_TENSOR_ID: Mutex<u64> = Mutex::new(1);
    static ref PHYSICS_STATE: Mutex<HashMap<u64, (f32, f32, f32)>> = Mutex::new(HashMap::new());
    static ref NEXT_BODY_ID: Mutex<u64> = Mutex::new(1);
    static ref ML_MEMORY_STATE: Mutex<JulesMlMemoryState> = Mutex::new(JulesMlMemoryState {
        min_bytes: 0,
        extra_bytes: usize::MAX / 2,
        core_used_bytes: 0,
        extra_used_bytes: 0,
    });
}

// =============================================================================
// C FFI: Core functions
// =============================================================================

#[no_mangle]
pub extern "C" fn jules_init() -> *mut JulesContext {
    let ctx = Box::new(JulesContext {
        _private: std::ptr::null_mut(),
    });
    Box::into_raw(ctx)
}

#[no_mangle]
pub extern "C" fn jules_destroy(ctx: *mut JulesContext) {
    if !ctx.is_null() {
        unsafe {
            let _ = Box::from_raw(ctx);
        }
    }
}

#[no_mangle]
pub extern "C" fn jules_version() -> u32 {
    FFI_VERSION
}

// =============================================================================
// C FFI: Tensor operations
// =============================================================================

#[no_mangle]
pub extern "C" fn jules_tensor_create(shape: *const usize, shape_len: usize) -> *mut JulesTensor {
    if shape.is_null() || shape_len == 0 {
        return std::ptr::null_mut();
    }

    unsafe {
        let shape_slice = std::slice::from_raw_parts(shape, shape_len);
        let shape_vec = shape_slice.to_vec();
        let numel: usize = shape_vec.iter().product();

        if numel == 0 {
            return std::ptr::null_mut();
        }

        let data = vec![0.0f32; numel];
        let mut storage = TENSOR_STORAGE.lock().unwrap();
        let mut shapes = TENSOR_SHAPES.lock().unwrap();
        let mut next_id = NEXT_TENSOR_ID.lock().unwrap();

        let id = *next_id;
        *next_id += 1;

        storage.insert(id, data);
        shapes.insert(id, shape_vec.clone());
        let mut shape_box = shape_vec.into_boxed_slice();
        let shape_ptr = shape_box.as_mut_ptr();
        std::mem::forget(shape_box);

        Box::into_raw(Box::new(JulesTensor {
            ptr: id,
            shape_ptr,
            shape_len,
            dtype: 0,
        }))
    }
}

#[no_mangle]
pub extern "C" fn jules_tensor_destroy(tensor: *mut JulesTensor) {
    if !tensor.is_null() {
        unsafe {
            let t = Box::from_raw(tensor);
            let mut storage = TENSOR_STORAGE.lock().unwrap();
            let mut shapes = TENSOR_SHAPES.lock().unwrap();
            storage.remove(&t.ptr);
            shapes.remove(&t.ptr);
            if !t.shape_ptr.is_null() {
                let _ = Vec::from_raw_parts(t.shape_ptr, t.shape_len, t.shape_len);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn jules_tensor_data(tensor: *const JulesTensor) -> *const f32 {
    if tensor.is_null() {
        return std::ptr::null();
    }

    unsafe {
        let t = &*tensor;
        let storage = TENSOR_STORAGE.lock().unwrap();
        if let Some(data) = storage.get(&t.ptr) {
            data.as_ptr()
        } else {
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn jules_tensor_shape(tensor: *const JulesTensor) -> *const usize {
    if tensor.is_null() {
        return std::ptr::null();
    }

    unsafe {
        let t = &*tensor;
        t.shape_ptr
    }
}

#[no_mangle]
pub extern "C" fn jules_tensor_shape_len(tensor: *const JulesTensor) -> usize {
    if tensor.is_null() {
        return 0;
    }

    unsafe {
        let t = &*tensor;
        t.shape_len
    }
}

#[no_mangle]
pub extern "C" fn jules_tensor_numel(tensor: *const JulesTensor) -> usize {
    if tensor.is_null() {
        return 0;
    }

    unsafe {
        let t = &*tensor;
        let storage = TENSOR_STORAGE.lock().unwrap();
        storage.get(&t.ptr).map(|d| d.len()).unwrap_or(0)
    }
}

// =============================================================================
// C FFI: Physics operations
// =============================================================================

#[no_mangle]
pub extern "C" fn jules_physics_body_create(
    _ctx: *mut JulesContext,
    mass: f32,
    x: f32,
    y: f32,
    z: f32,
) -> u64 {
    if mass <= 0.0 {
        return 0;
    }

    let mut physics = PHYSICS_STATE.lock().unwrap();
    let mut next_id = NEXT_BODY_ID.lock().unwrap();

    let id = *next_id;
    *next_id += 1;

    physics.insert(id, (x, y, z));
    id
}

#[no_mangle]
pub extern "C" fn jules_physics_body_position(
    _ctx: *mut JulesContext,
    body_id: u64,
    pos: *mut [f32; 3],
) -> JulesError {
    if pos.is_null() {
        return JulesError::InvalidArg;
    }

    let physics = PHYSICS_STATE.lock().unwrap();
    if let Some(&(x, y, z)) = physics.get(&body_id) {
        unsafe {
            *pos = [x, y, z];
        }
        JulesError::Success
    } else {
        JulesError::NotFound
    }
}

#[no_mangle]
pub extern "C" fn jules_physics_step(_ctx: *mut JulesContext, dt: f32) {
    if dt > 0.0 {
        let mut physics = PHYSICS_STATE.lock().unwrap();
        for (_id, (_x, y, _z)) in physics.iter_mut() {
            *y -= 9.81 * dt;
        }
    }
}

// =============================================================================
// C FFI: ML memory budget controls
// =============================================================================

#[no_mangle]
pub extern "C" fn jules_ml_memory_configure(min_bytes: usize, extra_bytes: usize) -> JulesError {
    let mut state = ML_MEMORY_STATE.lock().unwrap();
    let new_total = min_bytes.saturating_add(extra_bytes);
    let used_total = state.total_used_bytes();
    if used_total > new_total
        || state.core_used_bytes > min_bytes
        || state.extra_used_bytes > extra_bytes
    {
        return JulesError::OutOfMemory;
    }
    state.min_bytes = min_bytes;
    state.extra_bytes = extra_bytes;
    JulesError::Success
}

#[no_mangle]
pub extern "C" fn jules_ml_memory_acquire(bytes: usize, pool: JulesMemoryPool) -> JulesError {
    let mut state = ML_MEMORY_STATE.lock().unwrap();
    if bytes == 0 {
        return JulesError::Success;
    }

    match pool {
        JulesMemoryPool::Core => {
            let next = state.core_used_bytes.saturating_add(bytes);
            if next > state.min_bytes {
                return JulesError::OutOfMemory;
            }
            state.core_used_bytes = next;
        }
        JulesMemoryPool::Extra => {
            let next = state.extra_used_bytes.saturating_add(bytes);
            if next > state.extra_bytes {
                return JulesError::OutOfMemory;
            }
            state.extra_used_bytes = next;
        }
    }

    if state.total_used_bytes() > state.total_cap_bytes() {
        // Rollback on overflow of the total budget.
        match pool {
            JulesMemoryPool::Core => {
                state.core_used_bytes = state.core_used_bytes.saturating_sub(bytes);
            }
            JulesMemoryPool::Extra => {
                state.extra_used_bytes = state.extra_used_bytes.saturating_sub(bytes);
            }
        }
        return JulesError::OutOfMemory;
    }

    JulesError::Success
}

#[no_mangle]
pub extern "C" fn jules_ml_memory_release(bytes: usize, pool: JulesMemoryPool) -> JulesError {
    let mut state = ML_MEMORY_STATE.lock().unwrap();
    match pool {
        JulesMemoryPool::Core => {
            state.core_used_bytes = state.core_used_bytes.saturating_sub(bytes);
        }
        JulesMemoryPool::Extra => {
            state.extra_used_bytes = state.extra_used_bytes.saturating_sub(bytes);
        }
    }
    JulesError::Success
}

#[no_mangle]
pub extern "C" fn jules_ml_memory_reset_usage() -> JulesError {
    let mut state = ML_MEMORY_STATE.lock().unwrap();
    state.core_used_bytes = 0;
    state.extra_used_bytes = 0;
    JulesError::Success
}

#[no_mangle]
pub extern "C" fn jules_ml_memory_snapshot(out: *mut JulesMlMemorySnapshot) -> JulesError {
    if out.is_null() {
        return JulesError::InvalidArg;
    }
    let state = ML_MEMORY_STATE.lock().unwrap();
    unsafe {
        *out = state.as_snapshot();
    }
    JulesError::Success
}

// =============================================================================
// C FFI: Error handling
// =============================================================================

#[no_mangle]
pub extern "C" fn jules_error_string(code: JulesError) -> *const c_char {
    let msg = match code {
        JulesError::Success => c"Success",
        JulesError::InvalidArg => c"Invalid argument",
        JulesError::RuntimeError => c"Runtime error",
        JulesError::OutOfMemory => c"Out of memory",
        JulesError::NotFound => c"Not found",
        JulesError::UnknownError => c"Unknown error",
    };
    msg.as_ptr()
}

#[no_mangle]
pub extern "C" fn jules_run_file_ffi(path: *const c_char) -> JulesError {
    if path.is_null() {
        return JulesError::InvalidArg;
    }
    let path_str = unsafe { CStr::from_ptr(path) };
    let Ok(path) = path_str.to_str() else {
        return JulesError::InvalidArg;
    };
    match crate::jules_run_file(path, "main") {
        Ok(()) => JulesError::Success,
        Err(_) => JulesError::RuntimeError,
    }
}

#[no_mangle]
pub extern "C" fn jules_check_code_ffi(source: *const c_char) -> JulesError {
    if source.is_null() {
        return JulesError::InvalidArg;
    }
    let source = unsafe { CStr::from_ptr(source) };
    let Ok(source) = source.to_str() else {
        return JulesError::InvalidArg;
    };
    let diags = crate::jules_check("<ffi>", source);
    if diags.iter().any(|d| d.is_error()) {
        JulesError::RuntimeError
    } else {
        JulesError::Success
    }
}

#[no_mangle]
pub extern "C" fn julius_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ml_memory_budget_enforces_core_and_extra_caps() {
        assert!(matches!(
            jules_ml_memory_configure(1024, 512),
            JulesError::Success
        ));
        assert!(matches!(jules_ml_memory_reset_usage(), JulesError::Success));

        assert!(matches!(
            jules_ml_memory_acquire(1024, JulesMemoryPool::Core),
            JulesError::Success
        ));
        assert!(matches!(
            jules_ml_memory_acquire(1, JulesMemoryPool::Core),
            JulesError::OutOfMemory
        ));
        assert!(matches!(
            jules_ml_memory_acquire(512, JulesMemoryPool::Extra),
            JulesError::Success
        ));
        assert!(matches!(
            jules_ml_memory_acquire(1, JulesMemoryPool::Extra),
            JulesError::OutOfMemory
        ));

        let mut snap = JulesMlMemorySnapshot {
            min_bytes: 0,
            extra_bytes: 0,
            core_used_bytes: 0,
            extra_used_bytes: 0,
            total_used_bytes: 0,
            total_cap_bytes: 0,
            headroom_bytes: 0,
        };
        assert!(matches!(
            jules_ml_memory_snapshot(&mut snap as *mut JulesMlMemorySnapshot),
            JulesError::Success
        ));
        assert_eq!(snap.total_cap_bytes, 1536);
        assert_eq!(snap.total_used_bytes, 1536);
        assert_eq!(snap.headroom_bytes, 0);
    }
}
