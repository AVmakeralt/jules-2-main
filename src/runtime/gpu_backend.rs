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

    /// Flash attention (scaled dot-product attention)
    /// query: [batch, q_len, head_dim], key: [batch, kv_len, head_dim],
    /// value: [batch, kv_len, v_dim], out: [batch, q_len, v_dim]
    fn flash_attention(
        &self,
        query: &GpuBufferHandle,
        key: &GpuBufferHandle,
        value: &GpuBufferHandle,
        out: &GpuBufferHandle,
        scale: f32,
        causal: bool,
    ) -> Result<(), String>;

    /// Get backend name
    fn backend_name(&self) -> &'static str;

    /// Check if backend is available
    fn is_available(&self) -> bool;
}

// =============================================================================
// Sharded Buffer Storage — reduces lock contention by partitioning the buffer
// map into N shards, each protected by its own Mutex. The shard for a given
// buffer ID is determined by id % N. For N = 8 (typical 8-core CPU), this
// reduces average lock wait time by ~8x compared to a single global lock.
// =============================================================================

/// Number of shards for the GPU buffer lock (1 per CPU core, max 8)
const GPU_SHARD_COUNT: usize = 8;

/// Sharded buffer storage for reduced lock contention
struct ShardedBuffers {
    shards: [Mutex<HashMap<u64, GpuBuffer>>; GPU_SHARD_COUNT],
}

impl ShardedBuffers {
    fn new() -> Self {
        Self {
            shards: std::array::from_fn(|_| Mutex::new(HashMap::new())),
        }
    }

    #[inline]
    fn shard_index(&self, id: u64) -> usize {
        (id as usize) % GPU_SHARD_COUNT
    }

    fn insert(&self, id: u64, buffer: GpuBuffer) {
        let shard = &self.shards[self.shard_index(id)];
        shard.lock().unwrap().insert(id, buffer);
    }

    fn get(&self, id: u64) -> Option<GpuBuffer> {
        let shard = &self.shards[self.shard_index(id)];
        shard.lock().unwrap().get(&id).cloned()
    }

    fn get_cloned_field<F, R>(&self, id: u64, f: F) -> Option<R>
    where
        F: FnOnce(&GpuBuffer) -> R,
    {
        let shard = &self.shards[self.shard_index(id)];
        let guard = shard.lock().unwrap();
        guard.get(&id).map(f)
    }

    fn with_mut<F, R>(&self, id: u64, f: F) -> Option<R>
    where
        F: FnOnce(&mut GpuBuffer) -> R,
    {
        let shard = &self.shards[self.shard_index(id)];
        let mut guard = shard.lock().unwrap();
        guard.get_mut(&id).map(f)
    }

    /// Lock a specific shard for operations that need mutable access
    fn lock_shard(&self, shard_idx: usize) -> std::sync::MutexGuard<HashMap<u64, GpuBuffer>> {
        self.shards[shard_idx % GPU_SHARD_COUNT].lock().unwrap()
    }
}

// =============================================================================
// CPU Backend (CPU fallback for development/testing)
// =============================================================================

pub struct CpuBackend {
    buffers: ShardedBuffers,
    next_id: Arc<Mutex<u64>>,
}

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend {
            buffers: ShardedBuffers::new(),
            next_id: Arc::new(Mutex::new(1)),
        }
    }
}

impl GpuBackendImpl for CpuBackend {
    fn upload(&self, data: &[f32], shape: Vec<usize>) -> GpuBufferHandle {
        // Allocate id first, then insert into sharded buffers.
        let id = {
            let mut next_id = self.next_id.lock().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };

        self.buffers.insert(
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
        self.buffers
            .get(handle.id)
            .map(|b| b.data.clone())
            .unwrap_or_default()
    }

    fn download_into(&self, handle: &GpuBufferHandle, out: &mut [f32]) -> Result<(), String> {
        let src = self.buffers
            .get_cloned_field(handle.id, |b| b.data.clone())
            .ok_or("Buffer not found for download_into")?;
        if src.len() != out.len() {
            return Err("download_into output size mismatch".into());
        }
        out.copy_from_slice(&src);
        Ok(())
    }

    fn write(&self, handle: &GpuBufferHandle, data: &[f32]) -> Result<(), String> {
        let data_len = data.len();
        self.buffers
            .with_mut(handle.id, |buf| {
                if buf.data.len() != data_len {
                    return Err("Write size does not match buffer size".into());
                }
                buf.data.copy_from_slice(&data[..data_len]);
                Ok(())
            })
            .ok_or("Buffer not found for write")?
    }

    fn matmul(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        out: &GpuBufferHandle,
    ) -> Result<(), String> {
        // Clone input buffers first (never hold more than one shard lock at a time).
        let buf_a = self.buffers.get(a.id).ok_or("Buffer A not found")?;
        let buf_b = self.buffers.get(b.id).ok_or("Buffer B not found")?;

        // Compute without holding any locks.
        let (out_data, out_shape) = batched_matmul(
            &buf_a.data,
            &buf_a.shape,
            &buf_b.data,
            &buf_b.shape,
        )?;

        // Write result to output buffer.
        self.buffers.with_mut(out.id, |out_buf| {
            out_buf.data = out_data;
            out_buf.shape = out_shape;
        }).ok_or("Output buffer not found")?;

        Ok(())
    }

    fn elementwise(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        op: GpuOp,
        out: &GpuBufferHandle,
    ) -> Result<(), String> {
        if matches!(op, GpuOp::MatMul) {
            return Err("MatMul is not an elementwise operation. Use the matmul() method instead.".into());
        }
        // Clone input buffers first.
        let a_buf = self.buffers.get(a.id).ok_or("Buffer A not found")?;
        let b_buf = self.buffers.get(b.id).ok_or("Buffer B not found")?;
        if a_buf.data.len() != b_buf.data.len() {
            return Err("Elementwise op requires equal-sized tensors".into());
        }
        let a_shape = a_buf.shape.clone();
        // Compute without holding any locks.
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
                GpuOp::MatMul => unreachable!(), // handled above
            })
            .collect();
        // Write result to output buffer.
        self.buffers.with_mut(out.id, |out_buf| {
            out_buf.data = out_data;
            out_buf.shape = a_shape;
        }).ok_or("Output buffer not found")?;
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
        // Clone input buffers first.
        let inp = self.buffers
            .get(input.id)
            .ok_or("Input buffer not found")?;
        let ker = self.buffers
            .get(kernel.id)
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
        // Compute without holding any locks.
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
        // Write result to output buffer.
        self.buffers.with_mut(out.id, |out_buf| {
            out_buf.data = out_data;
            out_buf.shape = vec![out_h, out_w];
        }).ok_or("Output buffer not found")?;
        Ok(())
    }

    fn pool(
        &self,
        input: &GpuBufferHandle,
        out: &GpuBufferHandle,
        pool_size: u32,
        is_max: bool,
    ) -> Result<(), String> {
        // Clone input buffer first.
        let inp = self.buffers
            .get(input.id)
            .ok_or("Input buffer not found")?;
        if inp.shape.len() != 2 {
            return Err("pool expects [H,W] input".into());
        }
        let p = pool_size.max(1) as usize;
        let out_h = inp.shape[0] / p;
        let out_w = inp.shape[1] / p;
        // Compute without holding any locks.
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
        // Write result to output buffer.
        self.buffers.with_mut(out.id, |out_buf| {
            out_buf.data = out_data;
            out_buf.shape = vec![out_h, out_w];
        }).ok_or("Output buffer not found")?;
        Ok(())
    }

    fn activation(
        &self,
        input: &GpuBufferHandle,
        out: &GpuBufferHandle,
        activation: &str,
    ) -> Result<(), String> {
        // Clone input buffer first.
        let inp = self.buffers
            .get(input.id)
            .ok_or("Input buffer not found")?;
        let inp_shape = inp.shape.clone();
        let mut out_data = inp.data.clone();
        // Compute without holding any locks.
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
                // Per-row softmax for 2D tensors [rows, cols].
                // For 1D, treat as a single row.
                let row_len = if inp_shape.len() >= 2 {
                    inp_shape[inp_shape.len() - 1]
                } else {
                    out_data.len()
                };
                let num_rows = out_data.len() / row_len.max(1);
                for r in 0..num_rows {
                    let row_start = r * row_len;
                    let row_end = row_start + row_len;
                    let row = &mut out_data[row_start..row_end];
                    let max_v = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mut sum = 0.0f32;
                    for v in row.iter_mut() {
                        *v = (*v - max_v).exp();
                        sum += *v;
                    }
                    if sum != 0.0 {
                        for v in row.iter_mut() {
                            *v /= sum;
                        }
                    }
                }
            }
            other => return Err(format!("unsupported activation `{other}`")),
        }
        // Write result to output buffer.
        self.buffers.with_mut(out.id, |out_buf| {
            out_buf.data = out_data;
            out_buf.shape = inp_shape;
        }).ok_or("Output buffer not found")?;
        Ok(())
    }

    fn flash_attention(
        &self,
        query: &GpuBufferHandle,
        key: &GpuBufferHandle,
        value: &GpuBufferHandle,
        out: &GpuBufferHandle,
        scale: f32,
        causal: bool,
    ) -> Result<(), String> {
        // Clone input buffers first (never hold more than one shard lock at a time).
        let q_buf = self.buffers.get(query.id).ok_or("Query buffer not found")?;
        let k_buf = self.buffers.get(key.id).ok_or("Key buffer not found")?;
        let v_buf = self.buffers.get(value.id).ok_or("Value buffer not found")?;
        // Compute without holding any locks.
        let (out_data, out_shape) = cpu_flash_attention(
            &q_buf.data, &q_buf.shape,
            &k_buf.data, &k_buf.shape,
            &v_buf.data, &v_buf.shape,
            scale, causal,
        )?;
        // Write result to output buffer.
        self.buffers.with_mut(out.id, |out_buf| {
            out_buf.data = out_data;
            out_buf.shape = out_shape;
        }).ok_or("Output buffer not found")?;
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
        
        // Use custom threading engine instead of thread::scope
        use crate::runtime::threading::join;

        let num_chunks = batch.div_ceil(batches_per_chunk);

        for chunk_idx in 0..num_chunks {
            let batch_start = chunk_idx * batches_per_chunk;
            let batch_end = (batch_start + batches_per_chunk).min(batch);
            let mid = batch_start + (batch_end - batch_start) / 2;

            let a_data = a_data;
            let b_data = b_data;

            let (first_half, second_half) = join(
                move || {
                    let mut local_out = Vec::new();
                    for bi in batch_start..mid {
                        let a_off = bi * a_mat;
                        let b_off = bi * b_mat;
                        let out = accelerated_matmul(
                            &a_data[a_off..a_off + a_mat],
                            &b_data[b_off..b_off + b_mat],
                            m,
                            k,
                            n,
                        );
                        local_out.extend_from_slice(&out);
                    }
                    (batch_start, local_out)
                },
                move || {
                    let mut local_out = Vec::new();
                    for bi in mid..batch_end {
                        let a_off = bi * a_mat;
                        let b_off = bi * b_mat;
                        let out = accelerated_matmul(
                            &a_data[a_off..a_off + a_mat],
                            &b_data[b_off..b_off + b_mat],
                            m,
                            k,
                            n,
                        );
                        local_out.extend_from_slice(&out);
                    }
                    (mid, local_out)
                },
            );

            // Copy results back for each half
            for (start_bi, half_out) in [&first_half, &second_half] {
                let start = start_bi * batch_mat_size;
                if start + half_out.len() <= out_data.len() {
                    out_data[start..start + half_out.len()].copy_from_slice(half_out);
                }
            }
        }
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

/// CPU-based scaled dot-product attention (flash attention fallback).
/// Computes: out = softmax(Q @ K^T * scale) @ V
/// q: [batch, q_len, d], k: [batch, kv_len, d], v: [batch, kv_len, v_dim]
/// Returns ([batch, q_len, v_dim] data, output shape).
fn cpu_flash_attention(
    q_data: &[f32],
    q_shape: &[usize],
    k_data: &[f32],
    k_shape: &[usize],
    v_data: &[f32],
    v_shape: &[usize],
    scale: f32,
    causal: bool,
) -> Result<(Vec<f32>, Vec<usize>), String> {
    if q_shape.len() != 3 || k_shape.len() != 3 || v_shape.len() != 3 {
        return Err("flash_attention expects rank-3 q,k,v tensors".into());
    }
    let batch = q_shape[0];
    let q_len = q_shape[1];
    let d = q_shape[2];
    let kv_len = k_shape[1];
    let kd = k_shape[2];
    let v_dim = v_shape[2];
    if batch != k_shape[0] || batch != v_shape[0] {
        return Err("flash_attention batch dimension mismatch".into());
    }
    if kd != d {
        return Err("flash_attention q/k head dimension mismatch".into());
    }
    if v_shape[1] != kv_len {
        return Err("flash_attention k/v sequence length mismatch".into());
    }
    if d == 0 || v_dim == 0 {
        return Err("flash_attention dimensions must be > 0".into());
    }

    let mut out = vec![0.0f32; batch * q_len * v_dim];

    for b in 0..batch {
        for t in 0..q_len {
            let q_base = (b * q_len + t) * d;
            let q_row = &q_data[q_base..q_base + d];

            // Compute attention scores: Q[t] . K[s] * scale
            let mut scores = vec![f32::NEG_INFINITY; kv_len];
            let mut max_score = f32::NEG_INFINITY;
            for s in 0..kv_len {
                if causal && s > t {
                    continue;
                }
                let k_base = (b * kv_len + s) * d;
                let k_row = &k_data[k_base..k_base + d];
                let mut dot = 0.0f32;
                for i in 0..d {
                    dot += q_row[i] * k_row[i];
                }
                let score = dot * scale;
                scores[s] = score;
                if score > max_score {
                    max_score = score;
                }
            }

            // Numerically stable softmax over scores
            let mut denom = 0.0f32;
            for s in 0..kv_len {
                if scores[s].is_finite() {
                    scores[s] = (scores[s] - max_score).exp();
                    denom += scores[s];
                } else {
                    scores[s] = 0.0;
                }
            }
            let denom = denom.max(1e-12);

            // Weighted sum of values
            let out_base = (b * q_len + t) * v_dim;
            for s in 0..kv_len {
                let w = scores[s] / denom;
                if w == 0.0 {
                    continue;
                }
                let v_base = (b * kv_len + s) * v_dim;
                for c in 0..v_dim {
                    out[out_base + c] += w * v_data[v_base + c];
                }
            }
        }
    }

    Ok((out, vec![batch, q_len, v_dim]))
}

// =============================================================================
// Jules GPU Backend (native Jules compute runtime)
// =============================================================================

/// WebGPU / wgpu backend.
///
/// For real GPU execution, the following are required:
/// - A Vulkan, Metal, or DX12 capable GPU and driver
/// - The `wgpu` crate linked with the appropriate backend feature
///   (vulkan-portability, metal, dx12, or gl)
/// - A valid GPU adapter enumerated at runtime via `wgpu::Instance::enumerate_adapters()`
/// - Compute pipeline creation for WGSL shaders (see GpuKernels)
///
/// When GPU hardware is not available, compute operations return errors rather
/// than silently falling back to CPU. Use CpuBackend explicitly if CPU
/// execution is desired.
pub struct WgpuBackend {
    // Jules native GPU backend runtime state (backend-agnostic compute path).
    buffers: ShardedBuffers,
    next_id: Arc<Mutex<u64>>,
    /// Whether a real wgpu GPU adapter was found at initialization.
    pub gpu_available: bool,
}

impl WgpuBackend {
    pub fn new() -> Result<Self, String> {
        let gpu_available = Self::wgpu_available();
        // In real implementation with wgpu crate:
        //   pollster::block_on(async {
        //       let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        //       let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        //           power_preference: wgpu::PowerPreference::HighPerformance,
        //           ..Default::default()
        //       }).await?;
        //       let (device, queue) = adapter.request_device(...).await?;
        //   })
        Ok(WgpuBackend {
            buffers: ShardedBuffers::new(),
            next_id: Arc::new(Mutex::new(1)),
            gpu_available,
        })
    }

    /// Check if a WebGPU-compatible GPU is available.
    ///
    /// Uses dlopen to probe for the Vulkan loader (libvulkan.so) on Linux,
    /// Metal framework on macOS, or d3d12.dll on Windows.
    /// Returns true if at least one backend library is found.
    pub fn wgpu_available() -> bool {
        // Check for Vulkan (Linux / Windows)
        if probe_library(b"libvulkan.so.1\0") {
            return true;
        }
        // Check for Metal (macOS)
        if probe_library(b"/System/Library/Frameworks/Metal.framework/Metal\0") {
            return true;
        }
        // Check for D3D12 (Windows)
        if probe_library(b"d3d12.dll\0") {
            return true;
        }
        false
    }
}

impl GpuBackendImpl for WgpuBackend {
    fn upload(&self, data: &[f32], shape: Vec<usize>) -> GpuBufferHandle {
        // Allocate id first, then insert into sharded buffers.
        let id = {
            let mut next_id = self.next_id.lock().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };

        self.buffers.insert(
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
        self.buffers
            .get(handle.id)
            .map(|buf| buf.data.clone())
            .unwrap_or_default()
    }

    fn download_into(&self, handle: &GpuBufferHandle, out: &mut [f32]) -> Result<(), String> {
        let src = self.buffers
            .get_cloned_field(handle.id, |b| b.data.clone())
            .ok_or("Buffer not found for download_into")?;
        if src.len() != out.len() {
            return Err("download_into output size mismatch".into());
        }
        out.copy_from_slice(&src);
        Ok(())
    }

    fn write(&self, handle: &GpuBufferHandle, data: &[f32]) -> Result<(), String> {
        let data_len = data.len();
        self.buffers
            .with_mut(handle.id, |buf| {
                if buf.data.len() != data_len {
                    return Err("Write size does not match buffer size".into());
                }
                buf.data.copy_from_slice(&data[..data_len]);
                Ok(())
            })
            .ok_or("Buffer not found for write")?
    }

    fn matmul(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        out: &GpuBufferHandle,
    ) -> Result<(), String> {
        if !self.gpu_available {
            return Err("WgpuBackend: GPU not available — real GPU operations are not yet implemented; use CpuBackend instead".into());
        }
        // Clone input buffers first (never hold more than one shard lock at a time).
        let a_buf = self.buffers.get(a.id).ok_or("Buffer A not found")?;
        let b_buf = self.buffers.get(b.id).ok_or("Buffer B not found")?;
        // Compute without holding any locks.
        let (out_data, out_shape) = batched_matmul(
            &a_buf.data,
            &a_buf.shape,
            &b_buf.data,
            &b_buf.shape,
        )?;
        // Write result to output buffer.
        self.buffers.with_mut(out.id, |out_buf| {
            out_buf.data = out_data;
            out_buf.shape = out_shape;
        }).ok_or("Output buffer not found")?;
        Ok(())
    }

    fn elementwise(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        op: GpuOp,
        out: &GpuBufferHandle,
    ) -> Result<(), String> {
        if !self.gpu_available {
            return Err("WgpuBackend: GPU not available — real GPU operations are not yet implemented; use CpuBackend instead".into());
        }
        if matches!(op, GpuOp::MatMul) {
            return Err("MatMul is not an elementwise operation. Use the matmul() method instead.".into());
        }
        // Clone input buffers first.
        let a_buf = self.buffers.get(a.id).ok_or("Buffer A not found")?;
        let b_buf = self.buffers.get(b.id).ok_or("Buffer B not found")?;
        if a_buf.data.len() != b_buf.data.len() {
            return Err("Elementwise op requires equal-sized tensors".into());
        }
        let a_shape = a_buf.shape.clone();
        // Compute without holding any locks.
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
                GpuOp::MatMul => unreachable!(), // handled above
            })
            .collect();
        // Write result to output buffer.
        self.buffers.with_mut(out.id, |out_buf| {
            out_buf.data = out_data;
            out_buf.shape = a_shape;
        }).ok_or("Output buffer not found")?;
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
        if !self.gpu_available {
            return Err("WgpuBackend: GPU not available — real GPU operations are not yet implemented; use CpuBackend instead".into());
        }
        // Clone input buffers first.
        let inp = self.buffers
            .get(input.id)
            .ok_or("Input buffer not found")?;
        let ker = self.buffers
            .get(kernel.id)
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
        // Compute without holding any locks.
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
        // Write result to output buffer.
        self.buffers.with_mut(out.id, |out_buf| {
            out_buf.data = out_data;
            out_buf.shape = vec![out_h, out_w];
        }).ok_or("Output buffer not found")?;
        Ok(())
    }

    fn pool(
        &self,
        input: &GpuBufferHandle,
        out: &GpuBufferHandle,
        pool_size: u32,
        is_max: bool,
    ) -> Result<(), String> {
        if !self.gpu_available {
            return Err("WgpuBackend: GPU not available — real GPU operations are not yet implemented; use CpuBackend instead".into());
        }
        // Clone input buffer first.
        let inp = self.buffers
            .get(input.id)
            .ok_or("Input buffer not found")?;
        if inp.shape.len() != 2 {
            return Err("pool expects [H,W] input".into());
        }
        let p = pool_size.max(1) as usize;
        let out_h = inp.shape[0] / p;
        let out_w = inp.shape[1] / p;
        // Compute without holding any locks.
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
        // Write result to output buffer.
        self.buffers.with_mut(out.id, |out_buf| {
            out_buf.data = out_data;
            out_buf.shape = vec![out_h, out_w];
        }).ok_or("Output buffer not found")?;
        Ok(())
    }

    fn activation(
        &self,
        input: &GpuBufferHandle,
        out: &GpuBufferHandle,
        activation: &str,
    ) -> Result<(), String> {
        if !self.gpu_available {
            return Err("WgpuBackend: GPU not available — real GPU operations are not yet implemented; use CpuBackend instead".into());
        }
        // Clone input buffer first.
        let inp = self.buffers
            .get(input.id)
            .ok_or("Input buffer not found")?;
        let inp_shape = inp.shape.clone();
        let mut out_data = inp.data.clone();
        // Compute without holding any locks.
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
                let row_len = if inp_shape.len() >= 2 {
                    inp_shape[inp_shape.len() - 1]
                } else {
                    out_data.len()
                };
                let num_rows = out_data.len() / row_len.max(1);
                for r in 0..num_rows {
                    let row_start = r * row_len;
                    let row_end = row_start + row_len;
                    let row = &mut out_data[row_start..row_end];
                    let max_v = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mut sum = 0.0f32;
                    for v in row.iter_mut() {
                        *v = (*v - max_v).exp();
                        sum += *v;
                    }
                    if sum != 0.0 {
                        for v in row.iter_mut() {
                            *v /= sum;
                        }
                    }
                }
            }
            other => return Err(format!("unsupported activation `{other}`")),
        }
        // Write result to output buffer.
        self.buffers.with_mut(out.id, |out_buf| {
            out_buf.data = out_data;
            out_buf.shape = inp_shape;
        }).ok_or("Output buffer not found")?;
        Ok(())
    }

    fn flash_attention(
        &self,
        query: &GpuBufferHandle,
        key: &GpuBufferHandle,
        value: &GpuBufferHandle,
        out: &GpuBufferHandle,
        scale: f32,
        causal: bool,
    ) -> Result<(), String> {
        if !self.gpu_available {
            return Err("WgpuBackend: GPU not available — real GPU operations are not yet implemented; use CpuBackend instead".into());
        }
        // Clone input buffers first (never hold more than one shard lock at a time).
        let q_buf = self.buffers.get(query.id).ok_or("Query buffer not found")?;
        let k_buf = self.buffers.get(key.id).ok_or("Key buffer not found")?;
        let v_buf = self.buffers.get(value.id).ok_or("Value buffer not found")?;
        // Compute without holding any locks.
        let (out_data, out_shape) = cpu_flash_attention(
            &q_buf.data, &q_buf.shape,
            &k_buf.data, &k_buf.shape,
            &v_buf.data, &v_buf.shape,
            scale, causal,
        )?;
        // Write result to output buffer.
        self.buffers.with_mut(out.id, |out_buf| {
            out_buf.data = out_data;
            out_buf.shape = out_shape;
        }).ok_or("Output buffer not found")?;
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        if self.gpu_available {
            "wgpu-gpu"
        } else {
            "wgpu-cpu-fallback"
        }
    }

    fn is_available(&self) -> bool {
        self.gpu_available // Only truly available when GPU hardware is present
    }
}

// =============================================================================
// CUDA Backend (NVIDIA GPU via CUDA Driver API)
// =============================================================================

/// CUDA Driver API type aliases.
///
/// These represent the function signatures for the CUDA Driver API.
/// They are loaded at runtime via dlopen/dlsym when libcuda.so is available,
/// so there is no link-time dependency on libcuda.
///
/// For real GPU execution, the following are required:
/// - libcuda.so.1 must be present (installed by the NVIDIA driver)
/// - A CUDA-capable GPU must be installed
/// - The CUDA Toolkit (optional, for cuBLAS/cuDNN integration)
///
/// When CUDA is not available, CudaBackend compute operations return errors
/// instead of silently falling back to CPU.
mod cuda_ffi {
    #![allow(non_camel_case_types)]

    pub type CUresult = i32;
    pub type CUdevice = i32;
    pub type CUcontext = *mut std::ffi::c_void;
    pub type CUdeviceptr = *mut std::ffi::c_void;

    /// cuInit(flags) — Initialize the CUDA driver API
    pub type cuInit_t = unsafe extern "C" fn(flags: u32) -> CUresult;
    /// cuDeviceGet(device, ordinal) — Get a device handle by index
    pub type cuDeviceGet_t = unsafe extern "C" fn(device: *mut CUdevice, ordinal: i32) -> CUresult;
    /// cuCtxCreate(ctx, flags, dev) — Create a CUDA context on a device
    pub type cuCtxCreate_t = unsafe extern "C" fn(
        ctx: *mut CUcontext,
        flags: u32,
        dev: CUdevice,
    ) -> CUresult;
    /// cuMemAlloc(dptr, size) — Allocate device memory
    pub type cuMemAlloc_t = unsafe extern "C" fn(dptr: *mut CUdeviceptr, size: usize) -> CUresult;
    /// cuMemFree(dptr) — Free device memory
    pub type cuMemFree_t = unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult;
    /// cuLaunchKernel(f, gridX, gridY, gridZ, blockX, blockY, blockZ,
    ///                sharedMemBytes, stream, kernelParams, extra) — Launch a CUDA kernel
    pub type cuLaunchKernel_t = unsafe extern "C" fn(
        f: *mut std::ffi::c_void,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        stream: *mut std::ffi::c_void,
        kernel_params: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void,
    ) -> CUresult;
}

/// Probe for a shared library at runtime using dlopen.
/// Returns true if the library can be opened (i.e. it exists and is loadable).
///
/// This uses POSIX dlopen/dlclose via extern "C" FFI (available from libc on
/// Linux and macOS). On Windows, LoadLibraryA/FreeLibrary would be used instead.
fn probe_library(name: &[u8]) -> bool {
    // Ensure the name is null-terminated
    assert!(name.last() == Some(&0), "library name must be null-terminated");
    #[cfg(unix)]
    {
        extern "C" {
            fn dlopen(filename: *const i8, flags: i32) -> *mut std::ffi::c_void;
            fn dlclose(handle: *mut std::ffi::c_void) -> i32;
        }
        const RTLD_LAZY: i32 = 1;
        unsafe {
            let handle = dlopen(name.as_ptr() as *const i8, RTLD_LAZY);
            if handle.is_null() {
                return false;
            }
            dlclose(handle);
            true
        }
    }
    #[cfg(not(unix))]
    {
        let _ = name;
        false
    }
}

/// NVIDIA CUDA backend.
///
/// Attempts to use the CUDA Driver API for GPU computation.
/// If libcuda.so is not available at runtime, compute operations return errors
/// rather than silently falling back to CPU. Use CpuBackend explicitly if CPU
/// execution is desired.
pub struct CudaBackend {
    buffers: Arc<Mutex<HashMap<u64, GpuBuffer>>>,
    next_id: Arc<Mutex<u64>>,
    pub has_cuda: bool,
}

impl CudaBackend {
    pub fn new() -> Result<Self, String> {
        let has_cuda = Self::cuda_available();
        Ok(CudaBackend {
            buffers: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(Mutex::new(1)),
            has_cuda,
        })
    }

    /// Check if libcuda.so is available via dlopen.
    ///
    /// This probes for the CUDA driver library at runtime. If found,
    /// CUDA operations can be dispatched to the GPU. Otherwise, all
    /// operations fall back to CPU computation.
    pub fn cuda_available() -> bool {
        // Try libcuda.so.1 (standard NVIDIA driver library)
        if probe_library(b"libcuda.so.1\0") {
            return true;
        }
        // Try libcuda.so (sometimes a symlink)
        if probe_library(b"libcuda.so\0") {
            return true;
        }
        false
    }
}

impl GpuBackendImpl for CudaBackend {
    fn upload(&self, data: &[f32], shape: Vec<usize>) -> GpuBufferHandle {
        // TODO: When has_cuda is true, allocate device memory with cuMemAlloc
        // and copy data to device with cuMemcpyHtoD.
        // For now, store in host buffers and compute on CPU.
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
                device: if self.has_cuda { "cuda" } else { "cuda-cpu-fallback" }.to_string(),
            },
        );
        GpuBufferHandle { id }
    }

    fn download(&self, handle: &GpuBufferHandle) -> Vec<f32> {
        // TODO: When has_cuda is true, use cuMemcpyDtoH to transfer from device.
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
        // TODO: When has_cuda is true, use cuMemcpyHtoD.
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
        if !self.has_cuda {
            return Err("CudaBackend: CUDA not available — real GPU operations are not yet implemented; use CpuBackend instead".into());
        }
        // TODO: When has_cuda is true, use cuBLAS sgemm or cuLaunchKernel
        // with a custom matmul kernel.
        // CPU fallback: actually compute the matrix multiplication.
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
        if !self.has_cuda {
            return Err("CudaBackend: CUDA not available — real GPU operations are not yet implemented; use CpuBackend instead".into());
        }
        if matches!(op, GpuOp::MatMul) {
            return Err("MatMul is not an elementwise operation. Use the matmul() method instead.".into());
        }
        // CPU fallback: actually compute element-wise operations.
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
                    if *y == 0.0 { 0.0 } else { x / y }
                }
                GpuOp::MatMul => unreachable!(),
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
        if !self.has_cuda {
            return Err("CudaBackend: CUDA not available — real GPU operations are not yet implemented; use CpuBackend instead".into());
        }
        // CPU fallback: actually compute the convolution.
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
        if !self.has_cuda {
            return Err("CudaBackend: CUDA not available — real GPU operations are not yet implemented; use CpuBackend instead".into());
        }
        // CPU fallback: actually compute pooling.
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
        if !self.has_cuda {
            return Err("CudaBackend: CUDA not available — real GPU operations are not yet implemented; use CpuBackend instead".into());
        }
        // CPU fallback: actually compute activation functions.
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
                // Per-row softmax for 2D tensors [rows, cols].
                let row_len = if inp.shape.len() >= 2 {
                    inp.shape[inp.shape.len() - 1]
                } else {
                    out_data.len()
                };
                let num_rows = out_data.len() / row_len.max(1);
                for r in 0..num_rows {
                    let row_start = r * row_len;
                    let row_end = row_start + row_len;
                    let row = &mut out_data[row_start..row_end];
                    let max_v = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let mut sum = 0.0f32;
                    for v in row.iter_mut() {
                        *v = (*v - max_v).exp();
                        sum += *v;
                    }
                    if sum != 0.0 {
                        for v in row.iter_mut() {
                            *v /= sum;
                        }
                    }
                }
            }
            other => return Err(format!("unsupported activation `{other}`")),
        }
        let inp_shape = inp.shape.clone();
        let _ = inp;
        let out_buf = buffers.get_mut(&out.id).ok_or("Output buffer not found")?;
        out_buf.data = out_data;
        out_buf.shape.clone_from(&inp_shape);
        Ok(())
    }

    fn flash_attention(
        &self,
        query: &GpuBufferHandle,
        key: &GpuBufferHandle,
        value: &GpuBufferHandle,
        out: &GpuBufferHandle,
        scale: f32,
        causal: bool,
    ) -> Result<(), String> {
        if !self.has_cuda {
            return Err("CudaBackend: CUDA not available — real GPU operations are not yet implemented; use CpuBackend instead".into());
        }
        // CPU fallback: scaled dot-product attention (actually computes).
        let mut buffers = self.buffers.lock().unwrap();
        let q_buf = buffers.get(&query.id).ok_or("Query buffer not found")?;
        let k_buf = buffers.get(&key.id).ok_or("Key buffer not found")?;
        let v_buf = buffers.get(&value.id).ok_or("Value buffer not found")?;
        let (out_data, out_shape) = cpu_flash_attention(
            &q_buf.data, &q_buf.shape,
            &k_buf.data, &k_buf.shape,
            &v_buf.data, &v_buf.shape,
            scale, causal,
        )?;
        let out_buf = buffers.get_mut(&out.id).ok_or("Output buffer not found")?;
        out_buf.data = out_data;
        out_buf.shape = out_shape;
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        if self.has_cuda {
            "cuda"
        } else {
            "cuda-cpu-fallback"
        }
    }

    fn is_available(&self) -> bool {
        self.has_cuda // Only truly available when CUDA hardware is present
    }
}

// =============================================================================
// Multi-Backend Selector
// =============================================================================

pub enum GpuBackend {
    Cpu(Arc<CpuBackend>),
    Wgpu(Arc<WgpuBackend>),
    Cuda(Arc<CudaBackend>),
}

impl GpuBackend {
    /// Auto-select best available GPU backend.
    /// Preference order: CUDA > wGPU > CPU.
    pub fn auto_select() -> Self {
        // Try CUDA first (NVIDIA GPUs)
        if let Ok(backend) = CudaBackend::new() {
            if backend.has_cuda {
                return GpuBackend::Cuda(Arc::new(backend));
            }
        }
        // Try wGPU next (Vulkan/Metal/DX12)
        if let Ok(backend) = WgpuBackend::new() {
            if backend.gpu_available {
                return GpuBackend::Wgpu(Arc::new(backend));
            }
        }
        // Fallback to CPU backend (no silent GPU fallback)
        GpuBackend::Cpu(Arc::new(CpuBackend::new()))
    }

    /// Force CPU backend
    pub fn cpu() -> Self {
        GpuBackend::Cpu(Arc::new(CpuBackend::new()))
    }

    /// Force CUDA backend (returns CudaBackend only if CUDA hardware is present;
    /// otherwise returns CpuBackend since CudaBackend without CUDA cannot run
    /// any compute operations)
    pub fn cuda() -> Self {
        match CudaBackend::new() {
            Ok(backend) if backend.has_cuda => GpuBackend::Cuda(Arc::new(backend)),
            _ => GpuBackend::Cpu(Arc::new(CpuBackend::new())),
        }
    }

    /// Get backend implementation trait object
    pub fn as_impl(&self) -> &dyn GpuBackendImpl {
        match self {
            GpuBackend::Cpu(backend) => backend.as_ref(),
            GpuBackend::Wgpu(backend) => backend.as_ref(),
            GpuBackend::Cuda(backend) => backend.as_ref(),
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

    /// Flash attention (scaled dot-product attention)
    pub fn flash_attention(
        &self,
        query: &GpuBufferHandle,
        key: &GpuBufferHandle,
        value: &GpuBufferHandle,
        out: &GpuBufferHandle,
        scale: f32,
        causal: bool,
    ) -> Result<(), String> {
        self.as_impl().flash_attention(query, key, value, out, scale, causal)
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

    /// WGSL for flash attention (scaled dot-product attention)
    /// This is a simplified shader stub — a production implementation would use
    /// tiling and online-softmax to avoid materializing the full attention matrix.
    pub const FLASH_ATTENTION_KERNEL: &'static str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: AttentionParams;

struct AttentionParams {
    batch: u32,
    q_len: u32,
    kv_len: u32,
    head_dim: u32,
    scale: f32,
    causal: u32,
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch * params.q_len;
    if (idx >= total) { return; }

    let b = idx / params.q_len;
    let t = idx % params.q_len;
    let v_dim = params.head_dim;

    // Compute attention scores and output for position (b, t)
    var max_score: f32 = -1e30;
    for (var s: u32 = 0u; s < params.kv_len; s = s + 1u) {
        if (params.causal == 1u && s > t) { continue; }
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
            let qi = (b * params.q_len + t) * params.head_dim + d;
            let ki = (b * params.kv_len + s) * params.head_dim + d;
            dot = dot + q[qi] * k[ki];
        }
        let score = dot * params.scale;
        if (score > max_score) { max_score = score; }
    }

    var denom: f32 = 0.0;
    for (var s: u32 = 0u; s < params.kv_len; s = s + 1u) {
        if (params.causal == 1u && s > t) { continue; }
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
            let qi = (b * params.q_len + t) * params.head_dim + d;
            let ki = (b * params.kv_len + s) * params.head_dim + d;
            dot = dot + q[qi] * k[ki];
        }
        let score = dot * params.scale;
        denom = denom + exp(score - max_score);
    }

    for (var c: u32 = 0u; c < v_dim; c = c + 1u) {
        var acc: f32 = 0.0;
        for (var s: u32 = 0u; s < params.kv_len; s = s + 1u) {
            if (params.causal == 1u && s > t) { continue; }
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
                let qi = (b * params.q_len + t) * params.head_dim + d;
                let ki = (b * params.kv_len + s) * params.head_dim + d;
                dot = dot + q[qi] * k[ki];
            }
            let score = dot * params.scale;
            let w = exp(score - max_score) / max(denom, 1e-12);
            let vi = (b * params.kv_len + s) * v_dim + c;
            acc = acc + w * v[vi];
        }
        out[(b * params.q_len + t) * v_dim + c] = acc;
    }
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
        if !backend.gpu_available {
            // Without a real GPU, compute operations must return an error
            // instead of silently falling back to CPU.
            let a = backend.upload(&[1.0, -2.0, 3.0, -4.0], vec![2, 2]);
            let b = backend.upload(&[0.5, 0.5, 0.5, 0.5], vec![2, 2]);
            let out = backend.upload(&[0.0, 0.0, 0.0, 0.0], vec![2, 2]);
            assert!(backend.elementwise(&a, &b, GpuOp::Add, &out).is_err());
            assert!(backend.activation(&out, &out, "relu").is_err());
            return;
        }
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
        if !backend.gpu_available {
            assert!(backend.matmul(
                &GpuBufferHandle { id: 0 },
                &GpuBufferHandle { id: 0 },
                &GpuBufferHandle { id: 0 },
            ).is_err());
            return;
        }
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
