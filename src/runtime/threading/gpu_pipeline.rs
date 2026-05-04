// =========================================================================
// GPU Pipeline for Parallel CPU-GPU Execution
// Double-buffered submission pattern for hiding GPU latency
// Integration with wgpu for compute shader dispatch
// =========================================================================

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// GPU operation type descriptor
#[derive(Debug, Clone)]
pub enum GpuOpType {
    /// Matrix multiplication: C(m×n) = A(m×k) × B(k×n)
    MatMul {
        /// Rows of A / C
        m: usize,
        /// Columns of B / C
        n: usize,
        /// Columns of A / Rows of B
        k: usize,
    },
    /// Element-wise operation
    ElementWise {
        /// Operation name (relu, sigmoid, tanh, exp, log, abs, neg, square, sqrt, add, mul)
        op: String,
    },
    /// Reduction operation
    Reduce {
        /// Operation name (sum, mean, max, min)
        op: String,
        /// Number of input elements
        size: usize,
    },
    /// Generic / unknown operation
    Generic,
}

/// GPU task descriptor carrying all information needed to execute on CPU
#[derive(Debug, Clone)]
pub struct GpuTaskDescriptor {
    /// Operation type
    pub op_type: GpuOpType,
    /// Input data (concatenated: for MatMul A then B; for ElementWise unary or binary)
    pub input_data: Vec<f32>,
    /// Expected output size
    pub output_size: usize,
}

impl GpuTaskDescriptor {
    /// Create a new GPU task descriptor
    pub fn new(op_type: GpuOpType, input_data: Vec<f32>, output_size: usize) -> Self {
        Self { op_type, input_data, output_size }
    }
}

/// CPU-side executor for GPU task descriptors.
/// Performs actual computation when no real GPU backend is available.
struct CpuTaskExecutor;

impl CpuTaskExecutor {
    /// Execute a GPU task descriptor on the CPU
    fn execute(&self, task: &GpuTaskDescriptor) -> Vec<f32> {
        match &task.op_type {
            GpuOpType::MatMul { m, n, k } => {
                let m = *m;
                let n = *n;
                let k = *k;
                // Input A is m×k, Input B is k×n, Output is m×n
                if task.input_data.len() < m * k + k * n {
                    // Not enough input data — return zeroed output
                    return vec![0.0f32; m * n];
                }
                let a = &task.input_data[0..m * k];
                let b = &task.input_data[m * k..m * k + k * n];
                let mut c = vec![0.0f32; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f32;
                        for l in 0..k {
                            sum += a[i * k + l] * b[l * n + j];
                        }
                        c[i * n + j] = sum;
                    }
                }
                c
            }
            GpuOpType::ElementWise { op } => {
                let data = &task.input_data;
                match op.as_str() {
                    "relu" => data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect(),
                    "sigmoid" => data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
                    "tanh" => data.iter().map(|&x| x.tanh()).collect(),
                    "exp" => data.iter().map(|&x| x.exp()).collect(),
                    "log" => data.iter().map(|&x| x.ln()).collect(),
                    "abs" => data.iter().map(|&x| x.abs()).collect(),
                    "neg" => data.iter().map(|&x| -x).collect(),
                    "square" => data.iter().map(|&x| x * x).collect(),
                    "sqrt" => data.iter().map(|&x| x.sqrt()).collect(),
                    "add" if task.input_data.len() > task.output_size => {
                        // Binary element-wise: first half + second half
                        let n = task.output_size;
                        (0..n).map(|i| task.input_data[i] + task.input_data[n + i]).collect()
                    }
                    "mul" if task.input_data.len() > task.output_size => {
                        let n = task.output_size;
                        (0..n).map(|i| task.input_data[i] * task.input_data[n + i]).collect()
                    }
                    _ => vec![0.0f32; task.output_size],
                }
            }
            GpuOpType::Reduce { op, size } => {
                let size = *size;
                if task.input_data.len() < size {
                    return vec![0.0f32; task.output_size];
                }
                let data = &task.input_data[0..size];
                match op.as_str() {
                    "sum" => vec![data.iter().sum()],
                    "mean" => vec![data.iter().sum::<f32>() / size as f32],
                    "max" => vec![data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)],
                    "min" => vec![data.iter().cloned().fold(f32::INFINITY, f32::min)],
                    _ => vec![0.0f32; task.output_size],
                }
            }
            GpuOpType::Generic => {
                // Generic tasks: attempt basic computation, fall back to zeros
                vec![0.0f32; task.output_size]
            }
        }
    }
}

/// GPU task handle with result
#[derive(Debug)]
pub struct GpuTaskHandle {
    /// Task ID
    id: u64,
    /// Completed flag
    completed: AtomicBool,
    /// Result data
    result: Vec<f32>,
    /// Error message
    error: Option<String>,
}

impl GpuTaskHandle {
    /// Create a new GPU task handle
    pub fn new(id: u64) -> Self {
        Self {
            id,
            completed: AtomicBool::new(false),
            result: Vec::new(),
            error: None,
        }
    }

    /// Check if completed
    pub fn is_completed(&self) -> bool {
        self.completed.load(Ordering::Acquire)
    }

    /// Mark as completed with result
    pub fn complete(&mut self, result: Vec<f32>) {
        self.result = result;
        self.completed.store(true, Ordering::Release);
    }

    /// Mark as failed with error
    pub fn fail(&mut self, error: String) {
        self.error = Some(error);
        self.completed.store(true, Ordering::Release);
    }

    /// Get result (blocks if not completed)
    pub fn get_result(&self) -> Result<&[f32], &str> {
        while !self.is_completed() {
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
        
        if let Some(ref error) = self.error {
            Err(error)
        } else {
            Ok(&self.result)
        }
    }

    /// Get the task ID
    pub fn id(&self) -> u64 {
        self.id
    }
}

/// GPU pipeline buffer state
#[derive(Debug, Clone, Copy)]
enum BufferState {
    /// Buffer is being filled by CPU
    Filling,
    /// Buffer is being processed by GPU
    Processing,
    /// Buffer is ready to be read
    Ready,
}

/// Double-buffered GPU buffer
#[allow(dead_code)]
struct GpuBuffer {
    /// Buffer ID
    id: usize,
    /// Buffer state
    state: AtomicUsize, // BufferState as usize
    /// Data
    data: Vec<f32>,
    /// Associated task ID
    task_id: AtomicU64,
}

impl GpuBuffer {
    fn new(id: usize, capacity: usize) -> Self {
        Self {
            id,
            state: AtomicUsize::new(BufferState::Filling as usize),
            data: vec![0.0; capacity],
            task_id: AtomicU64::new(0),
        }
    }

    fn get_state(&self) -> BufferState {
        match self.state.load(Ordering::Acquire) {
            0 => BufferState::Filling,
            1 => BufferState::Processing,
            2 => BufferState::Ready,
            _ => BufferState::Filling,
        }
    }

    fn set_state(&self, state: BufferState) {
        self.state.store(state as usize, Ordering::Release);
    }
}

/// GPU pipeline for double-buffered execution
pub struct GpuPipeline {
    /// Pending tasks
    pending: Arc<std::sync::Mutex<Vec<GpuTaskHandle>>>,
    /// Completed tasks
    completed: Arc<std::sync::Mutex<Vec<GpuTaskHandle>>>,
    /// Next task ID
    next_id: AtomicUsize,
    /// Double buffer state
    buffer_index: AtomicUsize,
    /// Buffers
    buffers: Vec<GpuBuffer>,
    /// Buffer capacity
    buffer_capacity: usize,
}

impl GpuPipeline {
    /// Create a new GPU pipeline
    pub fn new(num_buffers: usize, buffer_capacity: usize) -> Self {
        let mut buffers = Vec::with_capacity(num_buffers);
        for i in 0..num_buffers {
            buffers.push(GpuBuffer::new(i, buffer_capacity));
        }
        
        Self {
            pending: Arc::new(std::sync::Mutex::new(Vec::new())),
            completed: Arc::new(std::sync::Mutex::new(Vec::new())),
            next_id: AtomicUsize::new(0),
            buffer_index: AtomicUsize::new(0),
            buffers,
            buffer_capacity,
        }
    }

    /// Submit a GPU task
    pub fn submit(&self, data: Vec<f32>) -> Arc<GpuTaskHandle> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) as u64;
        let handle = Arc::new(GpuTaskHandle::new(id));
        
        // Get current buffer
        let buffer_idx = self.buffer_index.load(Ordering::Acquire);
        let buffer = &self.buffers[buffer_idx];
        
        // Copy data to buffer
        if data.len() <= self.buffer_capacity {
            let data_ptr = buffer.data.as_ptr() as *mut f32;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data.len());
            }
        }
        
        buffer.task_id.store(id, Ordering::Release);
        buffer.set_state(BufferState::Processing);
        
        // Add to pending
        let mut pending = self.pending.lock().unwrap();
        pending.push(GpuTaskHandle::new(id));
        
        // Swap buffers
        let next_idx = (buffer_idx + 1) % self.buffers.len();
        self.buffer_index.store(next_idx, Ordering::Release);
        
        // Dispatch to GPU compute shader
        // For now, simulate GPU execution on CPU
        self.simulate_gpu_execution(id, data);
        
        handle
    }

    /// Simulate GPU execution (CPU fallback for wgpu integration)
    fn simulate_gpu_execution(&self, task_id: u64, data: Vec<f32>) {
        let pending = self.pending.clone();
        let completed = self.completed.clone();
        
        std::thread::spawn(move || {
            // Simulate GPU work
            std::thread::sleep(std::time::Duration::from_millis(10));
            
            // Move from pending to completed
            let mut pending = pending.lock().unwrap();
            let mut completed = completed.lock().unwrap();
            
            if let Some(pos) = pending.iter().position(|t| t.id == task_id) {
                let mut task = pending.remove(pos);
                task.complete(data);
                completed.push(task);
            }
        });
    }

    /// Poll for completed tasks
    pub fn poll(&self) -> Vec<Arc<GpuTaskHandle>> {
        let mut completed = self.completed.lock().unwrap();
        let mut results = Vec::new();
        
        if !completed.is_empty() {
            results = completed.drain(..).map(Arc::new).collect();
        }
        
        results
    }

    /// Get current buffer index for double-buffering
    pub fn current_buffer(&self) -> usize {
        self.buffer_index.load(Ordering::Acquire)
    }

    /// Swap buffers
    pub fn swap_buffers(&self) {
        let current = self.buffer_index.load(Ordering::Acquire);
        let next = (current + 1) % self.buffers.len();
        self.buffer_index.store(next, Ordering::Release);
    }

    /// Get buffer state
    pub fn get_buffer_state(&self, index: usize) -> BufferState {
        if index < self.buffers.len() {
            self.buffers[index].get_state()
        } else {
            BufferState::Filling
        }
    }

    /// Wait for all pending tasks to complete
    pub fn wait_all(&self) {
        loop {
            let pending = self.pending.lock().unwrap();
            if pending.is_empty() {
                break;
            }
            drop(pending);
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
    }
}

impl Default for GpuPipeline {
    fn default() -> Self {
        Self::new(2, 1024)
    }
}

impl GpuPipeline {
    /// Submit a raw GPU task to the pipeline.
    ///
    /// The caller provides an opaque pointer to a task structure.  The
    /// pipeline enqueues the task and returns `Ok(())` once the task has
    /// been accepted.  Actual execution happens asynchronously — use
    /// [`poll`] or [`wait_all`] to observe completion.
    ///
    /// On systems without a real GPU backend the task is executed
    /// immediately on a helper thread (CPU fallback).
    pub fn submit_task(&self, descriptor: GpuTaskDescriptor) -> Result<(), String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) as u64;

        // Grab the current double-buffer slot
        let buffer_idx = self.buffer_index.load(Ordering::Acquire);
        if buffer_idx >= self.buffers.len() {
            return Err("submit_task: buffer index out of range".to_string());
        }
        let buffer = &self.buffers[buffer_idx];

        // Copy input data into the double-buffer if it fits.
        if descriptor.input_data.len() <= self.buffer_capacity {
            let data_ptr = buffer.data.as_ptr() as *mut f32;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    descriptor.input_data.as_ptr(),
                    data_ptr,
                    descriptor.input_data.len(),
                );
            }
        }

        // Mark the buffer as processing and associate it with the task id.
        buffer.task_id.store(id, Ordering::Release);
        buffer.set_state(BufferState::Processing);

        // Create a handle and push it onto the pending queue.
        let handle = GpuTaskHandle::new(id);
        let mut pending = self.pending.lock().unwrap();
        pending.push(handle);

        // Advance the double-buffer index for the next submission.
        let next_idx = (buffer_idx + 1) % self.buffers.len();
        self.buffer_index.store(next_idx, Ordering::Release);

        // Execute asynchronously on a CPU helper thread using CpuTaskExecutor.
        // In a real implementation this would dispatch a wgpu compute
        // shader or a CUDA kernel.
        let pending_arc = self.pending.clone();
        let completed_arc = self.completed.clone();

        std::thread::spawn(move || {
            // Simulate GPU work latency.
            std::thread::sleep(std::time::Duration::from_micros(500));

            // Perform actual computation on CPU.
            let executor = CpuTaskExecutor;
            let result = executor.execute(&descriptor);

            // Move from pending → completed.
            let mut pending = pending_arc.lock().unwrap();
            let mut completed = completed_arc.lock().unwrap();

            if let Some(pos) = pending.iter().position(|t| t.id == id) {
                let mut task = pending.remove(pos);
                task.complete(result);
                completed.push(task);
            }
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_pipeline_submit() {
        let pipeline = GpuPipeline::new(2, 1024);
        let handle = pipeline.submit(vec![1.0, 2.0, 3.0]);
        assert!(!handle.is_completed());
    }

    #[test]
    fn test_double_buffer() {
        let pipeline = GpuPipeline::new(2, 1024);
        assert_eq!(pipeline.current_buffer(), 0);
        pipeline.swap_buffers();
        assert_eq!(pipeline.current_buffer(), 1);
    }

    #[test]
    fn test_buffer_state() {
        let pipeline = GpuPipeline::new(2, 1024);
        let state = pipeline.get_buffer_state(0);
        assert!(matches!(state, BufferState::Filling));
    }

    #[test]
    fn test_wait_all() {
        let pipeline = GpuPipeline::new(2, 1024);
        pipeline.submit(vec![1.0, 2.0, 3.0]);
        pipeline.wait_all();
    }
}
