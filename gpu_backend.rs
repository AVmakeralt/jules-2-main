#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

// GPU buffer handle (opaque to users)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuBufferHandle {
    pub id: u64,
}

#[repr(C)]
pub enum GpuOp {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
}

pub enum GpuMemoryType {
    Float32,
    Float64,
    Int32,
    Int64,
}

#[derive(Clone)]
pub struct GpuBuffer {
    pub id: u64,
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub device: String, // "cuda", "metal", "wgpu", "cpu"
}

/// Main trait for GPU backends
pub trait GpuBackendImpl: Send + Sync {
    /// Upload data to GPU
    fn upload(&self, data: &[f32], shape: Vec<usize>) -> GpuBufferHandle;

    /// Download data from GPU
    fn download(&self, handle: &GpuBufferHandle) -> Vec<f32>;

    /// Download data from GPU into caller-provided output buffer.
    fn download_into(&self, handle: &GpuBufferHandle, out: &mut [f32]) -> Result<(), String>;

    /// Overwrite an existing GPU buffer with new data
    fn write(&self, handle: &GpuBufferHandle, data: &[f32]) -> Result<(), String>;

    /// Matrix multiplication: C = A @ B
    fn matmul(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        out: &GpuBufferHandle,
    ) -> Result<(), String>;

    /// Element-wise operation
    fn elementwise(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        op: GpuOp,
        out: &GpuBufferHandle,
    ) -> Result<(), String>;

    /// Convolution operation (for neural networks)
    fn conv2d(
        &self,
        input: &GpuBufferHandle,
        kernel: &GpuBufferHandle,
        out: &GpuBufferHandle,
        stride: u32,
        padding: u32,
    ) -> Result<(), String>;

    /// Pool operation (max or avg)
    fn pool(
        &self,
        input: &GpuBufferHandle,
        out: &GpuBufferHandle,
        pool_size: u32,
        is_max: bool,
    ) -> Result<(), String>;

    /// Activation function
    fn activation(
        &self,
        input: &GpuBufferHandle,
        out: &GpuBufferHandle,
        activation: &str, // "relu", "sigmoid", "tanh", "softmax"
    ) -> Result<(), String>;

    /// Get backend name
    fn backend_name(&self) -> &'static str;

    /// Check if backend is available
    fn is_available(&self) -> bool;
}

// =============================================================================
// CPU Backend (CPU fallback for development/testing)
// =============================================================================

pub struct CpuBackend {
    buffers: Arc<Mutex<HashMap<u64, GpuBuffer>>>,
    next_id: Arc<Mutex<u64>>,
}

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend {
            buffers: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(1)),
        }
    }
}

impl GpuBackendImpl for CpuBackend {
    fn upload(&self, data: &[f32], shape: Vec<usize>) -> GpuBufferHandle {
        let mut buffers = self.buffers.lock().unwrap();
        let mut next_id = self.next_id.lock().unwrap();

        let id = *next_id;
        *next_id += 1;

        buffers.insert(
            id,
            GpuBuffer {
                id,
                data: data.to_vec(),
                shape,
                device: "cpu".to_string(),
            },
        );

        GpuBufferHandle { id }
    }

    fn download(&self, handle: &GpuBufferHandle) -> Vec<f32> {
        let buffers = self.buffers.lock().unwrap();
        buffers
            .get(&handle.id)
            .map(|b| b.data.clone())
            .unwrap_or_default()
    }

    fn download_into(&self, handle: &GpuBufferHandle, out: &mut [f32]) -> Result<(), String> {
        let buffers = self.buffers.lock().unwrap();
        let src = buffers
            .get(&handle.id)
            .ok_or("Buffer not found for download_into")?;
        if src.data.len() != out.len() {
            return Err("download_into output size mismatch".into());
        }
        out.copy_from_slice(&src.data);
        Ok(())
    }

    fn write(&self, handle: &GpuBufferHandle, data: &[f32]) -> Result<(), String> {
        let mut buffers = self.buffers.lock().unwrap();
        let buf = buffers
            .get_mut(&handle.id)
            .ok_or("Buffer not found for write")?;
        if buf.data.len() != data.len() {
            return Err("Write size does not match buffer size".into());
        }
        buf.data.copy_from_slice(data);
        Ok(())
    }

    fn matmul(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        out: &GpuBufferHandle,
    ) -> Result<(), String> {
        let mut buffers = self.buffers.lock().unwrap();
        let buf_a = buffers.get(&a.id).ok_or("Buffer A not found")?;
        let buf_b = buffers.get(&b.id).ok_or("Buffer B not found")?;

        let (out_data, out_shape) = batched_matmul(
            &buf_a.data,
            &buf_a.shape,
            &buf_b.data,
            &buf_b.shape,
        )?;

        let out_buf = buffers.get_mut(&out.id).ok_or("Output buffer not found")?;
        out_buf.data = out_data;
        out_buf.shape = out_shape;

        Ok(())
    }

    fn elementwise(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        op: GpuOp,
        out: &GpuBufferHandle,
    ) -> Result<(), String> {
        let mut buffers = self.buffers.lock().unwrap();
        let (a_len, a_shape) = {
            let a_buf = buffers.get(&a.id).ok_or("Buffer A not found")?;
            let b_buf = buffers.get(&b.id).ok_or("Buffer B not found")?;
            if a_buf.data.len() != b_buf.data.len() {
                return Err("Elementwise op requires equal-sized tensors".into());
            }
            (a_buf.data.len(), a_buf.shape.clone())
        };
        let a_buf = buffers.get(&a.id).unwrap();
        let b_buf = buffers.get(&b.id).unwrap();
        let out_data: Vec<f32> = a_buf
            .data
            .iter()
            .zip(b_buf.data.iter())
            .map(|(x, y)| match op {
                GpuOp::Add => x + y,
                GpuOp::Sub => x - y,
                GpuOp::Mul => x * y,
                GpuOp::Div => {
                    if *y == 0.0 {
                        0.0
                    } else {
                        x / y
                    }
                }
                GpuOp::MatMul => *x,
            })
            .collect();
        let out_buf = buffers.get_mut(&out.id).ok_or("Output buffer not found")?;
        out_buf.data = out_data;
        out_buf.shape.clone_from(&a_shape);
        let _ = a_len;
        Ok(())
    }

    fn conv2d(
        &self,
        input: &GpuBufferHandle,
        kernel: &GpuBufferHandle,
        out: &GpuBufferHandle,
        stride: u32,
        padding: u32,
    ) -> Result<(), String> {
        let mut buffers = self.buffers.lock().unwrap();
        let inp = buffers
            .get(&input.id)
            .ok_or("Input buffer not found")?;
        let ker = buffers
            .get(&kernel.id)
            .ok_or("Kernel buffer not found")?;
        if inp.shape.len() != 2 || ker.shape.len() != 2 {
            return Err("conv2d expects [H,W] input and [KH,KW] kernel".into());
        }
        let (h, w) = (inp.shape[0] as isize, inp.shape[1] as isize);
        let (kh, kw) = (ker.shape[0] as isize, ker.shape[1] as isize);
        let stride = stride.max(1) as isize;
        let pad = padding as isize;
        let out_h = (((h + 2 * pad - kh) / stride) + 1).max(0) as usize;
        let out_w = (((w + 2 * pad - kw) / stride) + 1).max(0) as usize;
        let mut out_data = vec![0.0f32; out_h * out_w];
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut acc = 0.0f32;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let iy = oy as isize * stride + ky - pad;
                        let ix = ox as isize * stride + kx - pad;
                        if iy >= 0 && iy < h && ix >= 0 && ix < w {
                            let iidx = iy as usize * inp.shape[1] + ix as usize;
                            let kidx = ky as usize * ker.shape[1] + kx as usize;
                            acc += inp.data[iidx] * ker.data[kidx];
                        }
                    }
                }
                out_data[oy * out_w + ox] = acc;
            }
        }
        let out_buf = buffers.get_mut(&out.id).ok_or("Output buffer not found")?;
        out_buf.data = out_data;
        out_buf.shape = vec![out_h, out_w];
        Ok(())
    }

    fn pool(
        &self,
        input: &GpuBufferHandle,
        out: &GpuBufferHandle,
        pool_size: u32,
        is_max: bool,
    ) -> Result<(), String> {
        let mut buffers = self.buffers.lock().unwrap();
        let inp = buffers
            .get(&input.id)
            .ok_or("Input buffer not found")?;
        if inp.shape.len() != 2 {
            return Err("pool expects [H,W] input".into());
        }
        let p = pool_size.max(1) as usize;
        let out_h = inp.shape[0] / p;
        let out_w = inp.shape[1] / p;
        let mut out_data = vec![0.0f32; out_h * out_w];
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut acc = if is_max { f32::NEG_INFINITY } else { 0.0 };
                for py in 0..p {
                    for px in 0..p {
                        let iy = oy * p + py;
                        let ix = ox * p + px;
                        let v = inp.data[iy * inp.shape[1] + ix];
                        if is_max {
                            acc = acc.max(v);
                        } else {
                            acc += v;
                        }
                    }
                }
                if !is_max {
                    acc /= (p * p) as f32;
                }
                out_data[oy * out_w + ox] = acc;
            }
        }
        let out_buf = buffers.get_mut(&out.id).ok_or("Output buffer not found")?;
        out_buf.data = out_data;
        out_buf.shape = vec![out_h, out_w];
        Ok(())
    }

    fn activation(
        &self,
        input: &GpuBufferHandle,
        out: &GpuBufferHandle,
        activation: &str,
    ) -> Result<(), String> {
        let mut buffers = self.buffers.lock().unwrap();
        let inp = buffers
            .get(&input.id)
            .ok_or("Input buffer not found")?;
        let mut out_data = inp.data.clone();
        match activation {
            "relu" => {
                for v in &mut out_data {
                    *v = v.max(0.0);
                }
            }
            "sigmoid" => {
                for v in &mut out_data {
                    *v = 1.0 / (1.0 + (-*v).exp());
                }
            }
            "tanh" => {
                for v in &mut out_data {
                    *v = v.tanh();
                }
            }
            "softmax" => {
                let max_v = out_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0f32;
                for v in &mut out_data {
                    *v = (*v - max_v).exp();
                    sum += *v;
                }
                if sum != 0.0 {
                    for v in &mut out_data {
                        *v /= sum;
                    }
                }
            }
            other => return Err(format!("unsupported activation `{other}`")),
        }
        let inp_shape = inp.shape.clone();
        let _ = inp; // keep reference alive until end of scope
        let out_buf = buffers.get_mut(&out.id).ok_or("Output buffer not found")?;
        out_buf.data = out_data;
        out_buf.shape.clone_from(&inp_shape);
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "cpu"
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }
}

fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

fn matmul_blocked_rows(
    a_data: &[f32],
    bt_data: &[f32],
    row_start: usize,
    row_end: usize,
    k: usize,
    n: usize,
    out_chunk: &mut [f32],
    out_row_base: usize,
) {
    const BK: usize = 64;
    const BN: usize = 64;
    for row in row_start..row_end {
        let out_row_local = row - out_row_base;
        let out_row = &mut out_chunk[out_row_local * n..(out_row_local + 1) * n];
        let a_row = &a_data[row * k..(row + 1) * k];
        for col_block in (0..n).step_by(BN) {
            let col_end = (col_block + BN).min(n);
            for col in col_block..col_end {
                let b_row = &bt_data[col * k..(col + 1) * k];
                let mut acc = 0.0f32;
                for kk_block in (0..k).step_by(BK) {
                    let kk_end = (kk_block + BK).min(k);
                    acc += dot_unrolled_8(&a_row[kk_block..kk_end], &b_row[kk_block..kk_end]);
                }
                out_row[col] = acc;
            }
        }
    }
}

fn accelerated_matmul(a_data: &[f32], b_data: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];

    // Route larger GEMMs to the optimized `matrixmultiply` kernel.
    let ops = m.saturating_mul(k).saturating_mul(n);
    if ops >= 16_384 {
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a_data.as_ptr(),
                k as isize,
                1,
                b_data.as_ptr(),
                n as isize,
                1,
                0.0,
                out.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        return out;
    }

    // For small matrices, blocked+transpose often wins by reducing setup overhead.
    let bt = transpose_2d(b_data, k, n);
    matmul_blocked_rows(a_data, &bt, 0, m, k, n, &mut out, 0);
    out
}

fn batched_matmul(
    a_data: &[f32],
    a_shape: &[usize],
    b_data: &[f32],
    b_shape: &[usize],
) -> Result<(Vec<f32>, Vec<usize>), String> {
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err("Matmul requires rank-2+ tensors".into());
    }
    if a_shape.len() != b_shape.len() {
        return Err("Matmul requires matching tensor rank".into());
    }

    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let k2 = b_shape[b_shape.len() - 2];
    let n = b_shape[b_shape.len() - 1];
    if k != k2 {
        return Err("Matmul dimension mismatch".into());
    }
    if a_shape[..a_shape.len() - 2] != b_shape[..b_shape.len() - 2] {
        return Err("Matmul batch shape mismatch".into());
    }

    let batch: usize = a_shape[..a_shape.len() - 2].iter().product();
    let a_mat = m * k;
    let b_mat = k * n;
    let mut out_data = vec![0.0f32; batch * m * n];

    let batch_mat_size = m * n;
    let ops_per_batch = m.saturating_mul(k).saturating_mul(n);
    let threads = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    let parallel_batches = threads > 1 && batch > 1 && ops_per_batch >= 65_536;

    if parallel_batches {
        let batches_per_chunk = batch.div_ceil(threads);
        thread::scope(|scope| {
            for (chunk_idx, out_chunk) in out_data
                .chunks_mut(batches_per_chunk * batch_mat_size)
                .enumerate()
            {
                let batch_start = chunk_idx * batches_per_chunk;
                let batch_end = (batch_start + batches_per_chunk).min(batch);
                scope.spawn(move || {
                    for bi in batch_start..batch_end {
                        let local_bi = bi - batch_start;
                        let a_off = bi * a_mat;
                        let b_off = bi * b_mat;
                        let out_off = local_bi * batch_mat_size;
                        let out = accelerated_matmul(
                            &a_data[a_off..a_off + a_mat],
                            &b_data[b_off..b_off + b_mat],
                            m,
                            k,
                            n,
                        );
                        out_chunk[out_off..out_off + batch_mat_size].copy_from_slice(&out);
                    }
                });
            }
        });
    } else {
        for bi in 0..batch {
            let a_off = bi * a_mat;
            let b_off = bi * b_mat;
            let out_off = bi * batch_mat_size;
            let out = accelerated_matmul(
                &a_data[a_off..a_off + a_mat],
                &b_data[b_off..b_off + b_mat],
                m,
                k,
                n,
            );
            out_data[out_off..out_off + batch_mat_size].copy_from_slice(&out);
        }
    }

    let mut out_shape = a_shape[..a_shape.len() - 2].to_vec();
    out_shape.push(m);
    out_shape.push(n);
    Ok((out_data, out_shape))
}

fn dot_unrolled_8(lhs: &[f32], rhs: &[f32]) -> f32 {
    let len = lhs.len();
    let chunks = len / 8;
    let mut i = 0;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut s4 = 0.0f32;
    let mut s5 = 0.0f32;
    let mut s6 = 0.0f32;
    let mut s7 = 0.0f32;

    for _ in 0..chunks {
        s0 += lhs[i] * rhs[i];
        s1 += lhs[i + 1] * rhs[i + 1];
        s2 += lhs[i + 2] * rhs[i + 2];
        s3 += lhs[i + 3] * rhs[i + 3];
        s4 += lhs[i + 4] * rhs[i + 4];
        s5 += lhs[i + 5] * rhs[i + 5];
        s6 += lhs[i + 6] * rhs[i + 6];
        s7 += lhs[i + 7] * rhs[i + 7];
        i += 8;
    }

    let mut tail = 0.0f32;
    while i < len {
        tail += lhs[i] * rhs[i];
        i += 1;
    }

    s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + tail
}

// =============================================================================
// Jules GPU Backend (native Jules compute runtime)
// =============================================================================

pub struct WgpuBackend {
    // Jules native GPU backend runtime state (backend-agnostic compute path).
    buffers: Arc<Mutex<HashMap<u64, GpuBuffer>>>,
    next_id: Arc<Mutex<u64>>,
}

impl WgpuBackend {
    pub fn new() -> Result<Self, String> {
        // In real implementation: pollster::block_on(Self::new_async())
        Ok(WgpuBackend {
            buffers: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(1)),
        })
    }
}

impl GpuBackendImpl for WgpuBackend {
    fn upload(&self, data: &[f32], shape: Vec<usize>) -> GpuBufferHandle {
        let mut buffers = self.buffers.lock().unwrap();
        let mut next_id = self.next_id.lock().unwrap();

        let id = *next_id;
        *next_id += 1;

        buffers.insert(
            id,
            GpuBuffer {
                id,
                data: data.to_vec(),
                shape,
                device: "jules-gpu".to_string(),
            },
        );

        GpuBufferHandle { id }
    }

    fn download(&self, handle: &GpuBufferHandle) -> Vec<f32> {
        let buffers = self.buffers.lock().unwrap();
        buffers
            .get(&handle.id)
            .map(|buf| buf.data.clone())
            .unwrap_or_default()
    }

    fn download_into(&self, handle: &GpuBufferHandle, out: &mut [f32]) -> Result<(), String> {
        let buffers = self.buffers.lock().unwrap();
        let src = buffers
            .get(&handle.id)
            .ok_or("Buffer not found for download_into")?;
        if src.data.len() != out.len() {
            return Err("download_into output size mismatch".into());
        }
        out.copy_from_slice(&src.data);
        Ok(())
    }

    fn write(&self, handle: &GpuBufferHandle, data: &[f32]) -> Result<(), String> {
        let mut buffers = self.buffers.lock().unwrap();
        let buf = buffers
            .get_mut(&handle.id)
            .ok_or("Buffer not found for write")?;
        if buf.data.len() != data.len() {
            return Err("Write size does not match buffer size".into());
        }
        buf.data.copy_from_slice(data);
        Ok(())
    }

    fn matmul(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        out: &GpuBufferHandle,
    ) -> Result<(), String> {
        let mut buffers = self.buffers.lock().unwrap();
        let a_buf = buffers.get(&a.id).ok_or("Buffer A not found")?;
        let b_buf = buffers.get(&b.id).ok_or("Buffer B not found")?;
        let (out_data, out_shape) = batched_matmul(
            &a_buf.data,
            &a_buf.shape,
            &b_buf.data,
            &b_buf.shape,
        )?;
        let out_buf = buffers.get_mut(&out.id).ok_or("Output buffer not found")?;
        out_buf.data = out_data;
        out_buf.shape = out_shape;
        Ok(())
    }

    fn elementwise(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        op: GpuOp,
        out: &GpuBufferHandle,
    ) -> Result<(), String> {
        let mut buffers = self.buffers.lock().unwrap();
        let (a_len, a_shape) = {
            let a_buf = buffers.get(&a.id).ok_or("Buffer A not found")?;
            let b_buf = buffers.get(&b.id).ok_or("Buffer B not found")?;
            if a_buf.data.len() != b_buf.data.len() {
                return Err("Elementwise op requires equal-sized tensors".into());
            }
            (a_buf.data.len(), a_buf.shape.clone())
        };
        let a_buf = buffers.get(&a.id).unwrap();
        let b_buf = buffers.get(&b.id).unwrap();
        let out_data: Vec<f32> = a_buf
            .data
            .iter()
            .zip(b_buf.data.iter())
            .map(|(x, y)| match op {
                GpuOp::Add => x + y,
                GpuOp::Sub => x - y,
                GpuOp::Mul => x * y,
                GpuOp::Div => {
                    if *y == 0.0 {
                        0.0
                    } else {
                        x / y
                    }
                }
                GpuOp::MatMul => *x,
            })
            .collect();
        let out_buf = buffers.get_mut(&out.id).ok_or("Output buffer not found")?;
        out_buf.data = out_data;
        out_buf.shape.clone_from(&a_shape);
        let _ = a_len;
        Ok(())
    }

    fn conv2d(
        &self,
        input: &GpuBufferHandle,
        kernel: &GpuBufferHandle,
        out: &GpuBufferHandle,
        stride: u32,
        padding: u32,
    ) -> Result<(), String> {
        let mut buffers = self.buffers.lock().unwrap();
        let inp = buffers
            .get(&input.id)
            .ok_or("Input buffer not found")?;
        let ker = buffers
            .get(&kernel.id)
            .ok_or("Kernel buffer not found")?;
        if inp.shape.len() != 2 || ker.shape.len() != 2 {
            return Err("conv2d expects [H,W] input and [KH,KW] kernel".into());
        }
        let (h, w) = (inp.shape[0] as isize, inp.shape[1] as isize);
        let (kh, kw) = (ker.shape[0] as isize, ker.shape[1] as isize);
        let stride = stride.max(1) as isize;
        let pad = padding as isize;
        let out_h = (((h + 2 * pad - kh) / stride) + 1).max(0) as usize;
        let out_w = (((w + 2 * pad - kw) / stride) + 1).max(0) as usize;
        let mut out_data = vec![0.0f32; out_h * out_w];
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut acc = 0.0f32;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let iy = oy as isize * stride + ky - pad;
                        let ix = ox as isize * stride + kx - pad;
                        if iy >= 0 && iy < h && ix >= 0 && ix < w {
                            let iidx = iy as usize * inp.shape[1] + ix as usize;
                            let kidx = ky as usize * ker.shape[1] + kx as usize;
                            acc += inp.data[iidx] * ker.data[kidx];
                        }
                    }
                }
                out_data[oy * out_w + ox] = acc;
            }
        }
        let out_buf = buffers.get_mut(&out.id).ok_or("Output buffer not found")?;
        out_buf.data = out_data;
        out_buf.shape = vec![out_h, out_w];
        Ok(())
    }

    fn pool(
        &self,
        input: &GpuBufferHandle,
        out: &GpuBufferHandle,
        pool_size: u32,
        is_max: bool,
    ) -> Result<(), String> {
        let mut buffers = self.buffers.lock().unwrap();
        let inp = buffers
            .get(&input.id)
            .ok_or("Input buffer not found")?;
        if inp.shape.len() != 2 {
            return Err("pool expects [H,W] input".into());
        }
        let p = pool_size.max(1) as usize;
        let out_h = inp.shape[0] / p;
        let out_w = inp.shape[1] / p;
        let mut out_data = vec![0.0f32; out_h * out_w];
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut acc = if is_max { f32::NEG_INFINITY } else { 0.0 };
                for py in 0..p {
                    for px in 0..p {
                        let iy = oy * p + py;
                        let ix = ox * p + px;
                        let v = inp.data[iy * inp.shape[1] + ix];
                        if is_max {
                            acc = acc.max(v);
                        } else {
                            acc += v;
                        }
                    }
                }
                if !is_max {
                    acc /= (p * p) as f32;
                }
                out_data[oy * out_w + ox] = acc;
            }
        }
        let out_buf = buffers.get_mut(&out.id).ok_or("Output buffer not found")?;
        out_buf.data = out_data;
        out_buf.shape = vec![out_h, out_w];
        Ok(())
    }

    fn activation(
        &self,
        input: &GpuBufferHandle,
        out: &GpuBufferHandle,
        activation: &str,
    ) -> Result<(), String> {
        let mut buffers = self.buffers.lock().unwrap();
        let inp = buffers
            .get(&input.id)
            .ok_or("Input buffer not found")?;
        let mut out_data = inp.data.clone();
        match activation {
            "relu" => {
                for v in &mut out_data {
                    *v = v.max(0.0);
                }
            }
            "sigmoid" => {
                for v in &mut out_data {
                    *v = 1.0 / (1.0 + (-*v).exp());
                }
            }
            "tanh" => {
                for v in &mut out_data {
                    *v = v.tanh();
                }
            }
            "softmax" => {
                let max_v = out_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0f32;
                for v in &mut out_data {
                    *v = (*v - max_v).exp();
                    sum += *v;
                }
                if sum != 0.0 {
                    for v in &mut out_data {
                        *v /= sum;
                    }
                }
            }
            other => return Err(format!("unsupported activation `{other}`")),
        }
        let inp_shape = inp.shape.clone();
        let _ = inp; // keep reference alive until end of scope
        let out_buf = buffers.get_mut(&out.id).ok_or("Output buffer not found")?;
        out_buf.data = out_data;
        out_buf.shape.clone_from(&inp_shape);
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "jules-gpu"
    }

    fn is_available(&self) -> bool {
        true
    }
}

// =============================================================================
// Multi-Backend Selector
// =============================================================================

pub enum GpuBackend {
    Cpu(Arc<CpuBackend>),
    Wgpu(Arc<WgpuBackend>),
}

impl GpuBackend {
    /// Auto-select best available GPU backend
    pub fn auto_select() -> Self {
        // Prefer Jules native GPU backend, then CPU fallback.
        match WgpuBackend::new() {
            Ok(backend) => GpuBackend::Wgpu(Arc::new(backend)),
            Err(_) => GpuBackend::Cpu(Arc::new(CpuBackend::new())),
        }
    }

    /// Force CPU backend
    pub fn cpu() -> Self {
        GpuBackend::Cpu(Arc::new(CpuBackend::new()))
    }

    /// Get backend implementation trait object
    pub fn as_impl(&self) -> &dyn GpuBackendImpl {
        match self {
            GpuBackend::Cpu(backend) => backend.as_ref(),
            GpuBackend::Wgpu(backend) => backend.as_ref(),
        }
    }

    pub fn upload(&self, data: &[f32], shape: Vec<usize>) -> GpuBufferHandle {
        self.as_impl().upload(data, shape)
    }

    pub fn download(&self, handle: &GpuBufferHandle) -> Vec<f32> {
        self.as_impl().download(handle)
    }

    pub fn download_into(&self, handle: &GpuBufferHandle, out: &mut [f32]) -> Result<(), String> {
        self.as_impl().download_into(handle, out)
    }

    pub fn write(&self, handle: &GpuBufferHandle, data: &[f32]) -> Result<(), String> {
        self.as_impl().write(handle, data)
    }

    pub fn matmul(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        out: &GpuBufferHandle,
    ) -> Result<(), String> {
        self.as_impl().matmul(a, b, out)
    }

    pub fn elementwise(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        op: GpuOp,
        out: &GpuBufferHandle,
    ) -> Result<(), String> {
        self.as_impl().elementwise(a, b, op, out)
    }

    pub fn backend_name(&self) -> &'static str {
        self.as_impl().backend_name()
    }

    pub fn is_available(&self) -> bool {
        self.as_impl().is_available()
    }
}

// =============================================================================
// GPU Memory Manager (handles allocation and cleanup)
// =============================================================================

pub struct GpuMemoryManager {
    backend: GpuBackend,
    allocated: Arc<Mutex<HashMap<u64, GpuBuffer>>>,
    current_bytes: Arc<Mutex<usize>>,
    config: Arc<Mutex<GpuMemoryConfig>>,
}

#[derive(Debug, Clone, Copy)]
pub struct GpuMemoryConfig {
    /// Memory reserved for runtime + model base footprint.
    pub base_reserved_bytes: usize,
    /// Additional bytes allowed above `base_reserved_bytes`.
    pub max_extra_bytes: usize,
}

impl GpuMemoryConfig {
    pub fn total_budget_bytes(self) -> usize {
        self.base_reserved_bytes
            .saturating_add(self.max_extra_bytes)
    }
}

impl GpuMemoryManager {
    pub fn new(backend: GpuBackend) -> Self {
        Self::new_with_config(
            backend,
            GpuMemoryConfig {
                base_reserved_bytes: 0,
                max_extra_bytes: usize::MAX / 2,
            },
        )
    }

    pub fn new_with_config(backend: GpuBackend, config: GpuMemoryConfig) -> Self {
        GpuMemoryManager {
            backend,
            allocated: Arc::new(Mutex::new(HashMap::new())),
            current_bytes: Arc::new(Mutex::new(0)),
            config: Arc::new(Mutex::new(config)),
        }
    }

    pub fn set_base_reserved_bytes(&self, bytes: usize) {
        let mut cfg = self.config.lock().unwrap();
        cfg.base_reserved_bytes = bytes;
    }

    pub fn set_max_extra_bytes(&self, bytes: usize) {
        let mut cfg = self.config.lock().unwrap();
        cfg.max_extra_bytes = bytes;
    }

    pub fn memory_budget_bytes(&self) -> usize {
        self.config.lock().unwrap().total_budget_bytes()
    }

    pub fn used_bytes(&self) -> usize {
        *self.current_bytes.lock().unwrap()
    }

    pub fn allocate(&self, shape: Vec<usize>, init_val: f32) -> Result<GpuBufferHandle, String> {
        let numel: usize = shape.iter().product();
        let data = vec![init_val; numel];
        self.allocate_from_data(shape, data)
    }

    pub fn allocate_from_data(
        &self,
        shape: Vec<usize>,
        data: Vec<f32>,
    ) -> Result<GpuBufferHandle, String> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err("Data length does not match shape".into());
        }
        let requested_bytes = numel.saturating_mul(std::mem::size_of::<f32>());
        let budget_bytes = self.memory_budget_bytes();
        let mut used = self.current_bytes.lock().unwrap();
        if used.saturating_add(requested_bytes) > budget_bytes {
            return Err(format!(
                "GPU memory budget exceeded: requested={} used={} budget={}",
                requested_bytes, *used, budget_bytes
            ));
        }

        let shape_clone = shape.clone();
        let handle = self.backend.upload(&data, shape);
        self.allocated.lock().unwrap().insert(
            handle.id,
            GpuBuffer {
                id: handle.id,
                data,
                shape: shape_clone,
                device: self.backend.backend_name().to_string(),
            },
        );
        *used += requested_bytes;
        Ok(handle)
    }

    pub fn free(&self, handle: &GpuBufferHandle) {
        let removed = self.allocated.lock().unwrap().remove(&handle.id);
        if let Some(buffer) = removed {
            let mut used = self.current_bytes.lock().unwrap();
            let bytes = buffer.data.len().saturating_mul(std::mem::size_of::<f32>());
            *used = used.saturating_sub(bytes);
        }
    }

    pub fn get_stats(&self) -> (usize, usize) {
        let allocated = self.allocated.lock().unwrap();
        let count = allocated.len();
        let total_elements: usize = allocated.values().map(|b| b.data.len()).sum();
        (count, total_elements)
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.backend_name()
    }

    pub fn matmul(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        out: &GpuBufferHandle,
    ) -> Result<(), String> {
        self.backend.matmul(a, b, out)
    }

    pub fn download(&self, handle: &GpuBufferHandle) -> Vec<f32> {
        self.backend.download(handle)
    }

    pub fn download_into(&self, handle: &GpuBufferHandle, out: &mut [f32]) -> Result<(), String> {
        self.backend.download_into(handle, out)
    }

    pub fn write(&self, handle: &GpuBufferHandle, data: &[f32]) -> Result<(), String> {
        let mut allocated = self.allocated.lock().unwrap();
        let local = allocated
            .get_mut(&handle.id)
            .ok_or("Buffer not managed by this memory manager")?;
        if local.data.len() != data.len() {
            return Err("Write size does not match managed buffer size".into());
        }
        local.data.copy_from_slice(data);
        self.backend.write(handle, data)
    }
}

// =============================================================================
// Kernel implementations (compute shaders for common operations)
// =============================================================================

pub struct GpuKernels;

impl GpuKernels {
    /// WGSL (WebGPU Shading Language) for matrix multiplication
    pub const MATMUL_KERNEL: &'static str = r#"
@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_out: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    var sum = 0.0;
    for (var k = 0u; k < 256u; k = k + 1u) {
        sum += matrix_a[row * 256u + k] * matrix_b[k * 256u + col];
    }

    matrix_out[row * 256u + col] = sum;
}
    "#;

    /// WGSL for ReLU activation
    pub const RELU_KERNEL: &'static str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    output[idx] = max(0.0, input[idx]);
}
    "#;

    /// WGSL for element-wise addition
    pub const ADD_KERNEL: &'static str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    out[idx] = a[idx] + b[idx];
}
    "#;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend() {
        let backend = CpuBackend::new();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let handle = backend.upload(&data, vec![2, 2]);
        let downloaded = backend.download(&handle);
        assert_eq!(downloaded, data);
    }

    #[test]
    fn test_auto_select() {
        let backend = GpuBackend::auto_select();
        assert!(backend.is_available());
        assert!(!backend.backend_name().is_empty());
    }

    #[test]
    fn test_cpu_backend_matmul_correctness() {
        let backend = CpuBackend::new();
        let a = backend.upload(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = backend.upload(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let out = backend.upload(&[0.0, 0.0, 0.0, 0.0], vec![2, 2]);
        backend.matmul(&a, &b, &out).unwrap();
        let got = backend.download(&out);
        assert_eq!(got, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_cpu_backend_elementwise_conv_pool_activation() {
        let backend = CpuBackend::new();

        // elementwise add
        let a = backend.upload(&[1.0, -2.0, 3.0, -4.0], vec![2, 2]);
        let b = backend.upload(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let ew_out = backend.upload(&[0.0, 0.0, 0.0, 0.0], vec![2, 2]);
        backend.elementwise(&a, &b, GpuOp::Add, &ew_out).unwrap();
        assert_eq!(backend.download(&ew_out), vec![1.5, -1.5, 3.5, -3.5]);

        // activation relu
        let act_out = backend.upload(&[0.0, 0.0, 0.0, 0.0], vec![2, 2]);
        backend.activation(&ew_out, &act_out, "relu").unwrap();
        assert_eq!(backend.download(&act_out), vec![1.5, 0.0, 3.5, 0.0]);

        // conv2d 3x3 input with 2x2 kernel
        let input = backend.upload(
            &[1.0, 2.0, 3.0,
              4.0, 5.0, 6.0,
              7.0, 8.0, 9.0],
            vec![3, 3],
        );
        let kernel = backend.upload(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let conv_out = backend.upload(&[0.0, 0.0, 0.0, 0.0], vec![2, 2]);
        backend.conv2d(&input, &kernel, &conv_out, 1, 0).unwrap();
        assert_eq!(backend.download(&conv_out), vec![6.0, 8.0, 12.0, 14.0]);

        // max-pool 2x2
        let pool_in = backend.upload(
            &[1.0, 2.0, 3.0, 4.0,
              5.0, 6.0, 7.0, 8.0,
              9.0, 10.0, 11.0, 12.0,
              13.0, 14.0, 15.0, 16.0],
            vec![4, 4],
        );
        let pool_out = backend.upload(&[0.0, 0.0, 0.0, 0.0], vec![2, 2]);
        backend.pool(&pool_in, &pool_out, 2, true).unwrap();
        assert_eq!(backend.download(&pool_out), vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_memory_manager() {
        let backend = GpuBackend::cpu();
        let manager = GpuMemoryManager::new(backend);
        let handle = manager.allocate(vec![10, 10], 0.0).unwrap();
        let (count, total) = manager.get_stats();
        assert!(count > 0);
        assert_eq!(total, 100);
        assert!(manager.used_bytes() >= 400);
        manager.free(&handle);
    }

    #[test]
    fn test_memory_budget_limit() {
        let backend = GpuBackend::cpu();
        let manager = GpuMemoryManager::new_with_config(
            backend,
            GpuMemoryConfig {
                base_reserved_bytes: 128,
                max_extra_bytes: 128,
            },
        );
        let too_large = manager.allocate(vec![100], 0.0);
        assert!(too_large.is_err());
    }

    #[test]
    fn test_jules_gpu_elementwise_and_activation() {
        let backend = WgpuBackend::new().unwrap();
        let a = backend.upload(&[1.0, -2.0, 3.0, -4.0], vec![2, 2]);
        let b = backend.upload(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
        let out = backend.upload(&[0.0, 0.0, 0.0, 0.0], vec![2, 2]);
        backend.elementwise(&a, &b, GpuOp::Add, &out).unwrap();
        let sum = backend.download(&out);
        assert_eq!(sum, vec![1.5, -1.5, 3.5, -3.5]);

        backend.activation(&out, &out, "relu").unwrap();
        let relu = backend.download(&out);
        assert_eq!(relu, vec![1.5, 0.0, 3.5, 0.0]);
    }

    #[test]
    fn test_jules_gpu_batched_matmul() {
        let backend = WgpuBackend::new().unwrap();
        let a = backend.upload(
            &[
                1.0, 2.0, 3.0, 4.0, // batch 0
                2.0, 0.0, 1.0, 2.0, // batch 1
            ],
            vec![2, 2, 2],
        );
        let b = backend.upload(
            &[
                5.0, 6.0, 7.0, 8.0, // batch 0
                1.0, 0.0, 0.0, 1.0, // batch 1 identity
            ],
            vec![2, 2, 2],
        );
        let out = backend.upload(&[0.0; 8], vec![2, 2, 2]);
        backend.matmul(&a, &b, &out).unwrap();
        let got = backend.download(&out);
        assert_eq!(got, vec![19.0, 22.0, 43.0, 50.0, 2.0, 0.0, 1.0, 2.0]);
    }
}
