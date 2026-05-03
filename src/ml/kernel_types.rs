// =========================================================================
// Jules Custom ML Kernel Types
//
// This module defines extensible kernel abstractions for hardware-specific
// acceleration. Unlike the ad-hoc kernel enums in ml_engine.rs, this follows
// a unified trait-based architecture that makes adding new kernel types
// straightforward and type-safe.
//
// Kernel Categories:
//   1. Precision Kernels    - Different numeric formats (fp32, fp16, bf16, int8, int4)
//   2. Domain Kernels        - Problem-specific optimizations (transformers, convolutions)
//   3. Layout Kensors        - Memory access pattern optimizations (SoA, AoS, blocked)
//
// Architecture:
//   KernelRegistry -> dispatches to appropriate kernel implementation
//   KernelTraits   -> defines the interface all kernels must implement
//   KernelBuffer   -> abstraction for pre-packed kernel data
// =========================================================================

use std::collections::HashMap;

// =========================================================================
// §1 Kernel Registry and Dispatch
// =========================================================================

/// Central registry for all kernel types.
/// Maintains a mapping from kernel identifier to the actual kernel implementation.
/// This allows runtime selection of the optimal kernel based on hardware and problem size.
pub struct KernelRegistry {
    matmul_kernels: HashMap<String, Box<dyn MatmulKernel>>,
    attention_kernels: HashMap<String, Box<dyn AttentionKernel>>,
    activation_kernels: HashMap<String, Box<dyn ActivationKernel>>,
    quantization_kernels: HashMap<String, Box<dyn QuantizedKernel>>,
}

impl KernelRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            matmul_kernels: HashMap::new(),
            attention_kernels: HashMap::new(),
            activation_kernels: HashMap::new(),
            quantization_kernels: HashMap::new(),
        };
        registry.register_default_kernels();
        registry
    }

    fn register_default_kernels(&mut self) {
        // Register FP32 kernels
        self.register_matmul_kernel("fp32_scalar", Fp32ScalarKernel);
        self.register_matmul_kernel("fp32_avx2", Fp32Avx2Kernel);
        self.register_matmul_kernel("fp32_avx512", Fp32Avx512Kernel);

        // Register BF16 kernels
        self.register_matmul_kernel("bf16_avx512", Bf16Avx512Kernel);

        // Register FP16 kernels  
        self.register_matmul_kernel("fp16_avx512", Fp16Avx512Kernel);

        // Register INT8 kernels
        self.register_quantized_kernel("int8_avx2", Int8Avx2Kernel);
        self.register_quantized_kernel("int8_avx512vnni", Int8Avx512VnniKernel);

        // Register attention kernels
        self.register_attention_kernel("flash_attention", FlashAttentionKernel);
        self.register_attention_kernel("ring_attention", RingAttentionKernel);

        // Register activation kernels
        self.register_activation_kernel("silu_gelu_avx512", SiluGeluAvx512Kernel);
    }

    pub fn register_matmul_kernel(&mut self, name: &'static str, kernel: impl MatmulKernel + 'static) {
        self.matmul_kernels.insert(name.to_string(), Box::new(kernel));
    }

    pub fn register_attention_kernel(&mut self, name: &'static str, kernel: impl AttentionKernel + 'static) {
        self.attention_kernels.insert(name.to_string(), Box::new(kernel));
    }

    pub fn register_activation_kernel(&mut self, name: &'static str, kernel: impl ActivationKernel + 'static) {
        self.activation_kernels.insert(name.to_string(), Box::new(kernel));
    }

    pub fn register_quantized_kernel(&mut self, name: &'static str, kernel: impl QuantizedKernel + 'static) {
        self.quantization_kernels.insert(name.to_string(), Box::new(kernel));
    }

    /// Auto-select the best kernel for the given problem characteristics.
    pub fn select_matmul_kernel(&self, m: usize, k: usize, n: usize, dtype: DataType) -> &dyn MatmulKernel {
        // Heuristic selection based on problem size and hardware
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") {
                match dtype {
                    DataType::Fp32 => {
                        if m * k * n > 1_000_000 {
                            return self.matmul_kernels.get("fp32_avx512").unwrap().as_ref();
                        }
                    }
                    DataType::Bf16 => {
                        return self.matmul_kernels.get("bf16_avx512").unwrap().as_ref();
                    }
                    DataType::Fp16 => {
                        return self.matmul_kernels.get("fp16_avx512").unwrap().as_ref();
                    }
                    _ => {}
                }
            }
            if is_x86_feature_detected!("avx2") {
                return self.matmul_kernels.get("fp32_avx2").unwrap().as_ref();
            }
        }
        self.matmul_kernels.get("fp32_scalar").unwrap().as_ref()
    }

    /// Select best attention kernel based on sequence length and hardware.
    pub fn select_attention_kernel(&self, seq_len: usize, dtype: DataType) -> &dyn AttentionKernel {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") && seq_len <= 4096 {
                return self.attention_kernels.get("flash_attention").unwrap().as_ref();
            }
        }
        self.attention_kernels.get("flash_attention").unwrap().as_ref()
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// §2 Data Types
// =========================================================================

/// Supported data types for kernel operations.
/// Each type has different tradeoffs in precision, speed, and memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Fp32,   // Standard 32-bit float (highest precision, most memory)
    Fp16,   // 16-bit float (half precision, faster but limited range)
    Bf16,   // Brain float 16 (same range as fp32, reduced precision)
    Int8,   // 8-bit integer (quantized, requires scaling)
    Int4,   // 4-bit integer (extreme compression, needs careful scaling)
    UInt8,  // Unsigned 8-bit (for non-negative activations)
}

impl DataType {
    /// Size in bytes per element.
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::Fp32 => 4,
            DataType::Fp16 | DataType::Bf16 => 2,
            DataType::Int8 | DataType::UInt8 => 1,
            DataType::Int4 => 1, // 2 elements per byte
        }
    }

    /// Maximum value for quantization types.
    pub fn max_quantized_value(&self) -> f32 {
        match self {
            DataType::Int8 => 127.0,
            DataType::UInt8 => 255.0,
            DataType::Int4 => 7.0,
            _ => 1.0,
        }
    }
}

// =========================================================================
// §3 Memory Layout Types
// =========================================================================

/// Memory layout for tensor data.
/// Different layouts optimize for different access patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    /// Row-major (standard C-style): stride[i] = product(shape[i+1:])
    RowMajor,
    /// Column-major (Fortran-style): stride[i] = product(shape[..i])
    ColMajor,
    /// Structure of Arrays: groups same-index elements contiguously
    SoA,
    /// Array of Structures: groups different-index elements
    AoS,
    /// Blocked layout: divides matrix into cache-friendly blocks
    Blocked { block_size: usize },
    /// Tiled layout for TMA (Tensor Memory Access) on NVIDIA/AMD
    Tiled { tile_m: usize, tile_n: usize },
}

impl MemoryLayout {
    /// Compute stride for row-major layout.
    pub fn row_major_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Compute stride for column-major layout.
    pub fn col_major_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in 1..shape.len() {
            strides[i] = strides[i - 1] * shape[i - 1];
        }
        strides
    }
}

// =========================================================================
// §4 Kernel Traits
// =========================================================================

/// Trait that all matrix multiplication kernels must implement.
/// Defines the interface for GEMM (General Matrix Multiply) operations.
pub trait MatmulKernel: Send + Sync {
    /// Execute matrix multiplication: C = alpha * A @ B + beta * C
    /// 
    /// Arguments:
    ///   a: Left matrix (m x k)
    ///   b: Right matrix (k x n)
    ///   c: Output matrix (m x n)
    ///   m, k, n: Dimensions
    ///   lda, ldb, ldc: Leading dimensions (stride between rows)
    fn gemm(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
    );

    /// Get the preferred block sizes for this kernel.
    fn preferred_block_sizes(&self) -> (usize, usize, usize) {
        (64, 64, 64) // default: 64x64x64 blocking
    }

    /// Minimum operation count to justify kernel overhead.
    fn min_ops_for_parallelism(&self) -> usize {
        100_000
    }

    /// Name identifier for this kernel.
    fn name(&self) -> &'static str;
}

/// Trait for attention mechanism kernels.
/// Supports various attention implementations (flash attention, ring attention, etc.).
pub trait AttentionKernel: Send + Sync {
    /// Compute scaled dot-product attention with optional causal masking.
    /// 
    /// Arguments:
    ///   q: Query tensor [batch, q_len, head_dim]
    ///   k: Key tensor [batch, kv_len, head_dim]
    ///   v: Value tensor [batch, kv_len, head_dim]
    ///   scale: 1/sqrt(head_dim)
    ///   causal: Whether to apply causal (upper triangular) masking
    /// 
    /// Returns: Output tensor [batch, q_len, head_dim]
    fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        q_len: usize,
        kv_len: usize,
        head_dim: usize,
        scale: f32,
        causal: bool,
    ) -> Vec<f32>;

    /// Estimate memory access pattern for cache optimization.
    fn memory_access_pattern(&self, seq_len: usize) -> MemoryAccessEstimate {
        MemoryAccessEstimate {
            reads: seq_len * seq_len,
            writes: seq_len * seq_len,
            cache_lines: seq_len * seq_len / 64,
        }
    }

    fn name(&self) -> &'static str;
}

/// Trait for activation function kernels.
/// Optimized implementations for common activations.
pub trait ActivationKernel: Send + Sync {
    /// Apply activation element-wise.
    fn forward(&self, data: &[f32]) -> Vec<f32>;

    /// Apply activation and compute gradient in one pass (fused).
    fn forward_backward(&self, data: &[f32], upstream_grad: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let output = self.forward(data);
        let grad = self.backward(data, &output, upstream_grad);
        (output, grad)
    }

    /// Compute backward pass.
    fn backward(&self, data: &[f32], output: &[f32], upstream_grad: &[f32]) -> Vec<f32>;

    fn name(&self) -> &'static str;
}

/// Trait for quantized matrix multiplication kernels.
/// Supports INT8, INT4, and other low-precision formats.
pub trait QuantizedKernel: Send + Sync {
    /// Execute quantized matrix multiplication with dequantization.
    fn gemm_quantized(
        &self,
        a: &[f32],
        qweight: &[i8],
        scales: &[f32],
        bias: Option<&[f32]>,
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    );

    /// Get the effective bits per parameter (including scales).
    fn effective_bits_per_param(&self) -> f32;

    fn name(&self) -> &'static str;
}

/// Memory access pattern estimate for cache optimization.
#[derive(Debug, Clone)]
pub struct MemoryAccessEstimate {
    pub reads: usize,
    pub writes: usize,
    pub cache_lines: usize,
}

// =========================================================================
// §5 Concrete Kernel Implementations
// =========================================================================

// --- FP32 Kernels ---

pub struct Fp32ScalarKernel;

impl MatmulKernel for Fp32ScalarKernel {
    fn gemm(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, _lda: usize, _ldb: usize, ldc: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * ldc + j] = sum;
            }
        }
    }

    fn name(&self) -> &'static str { "fp32_scalar" }
}

pub struct Fp32Avx2Kernel;

impl MatmulKernel for Fp32Avx2Kernel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn gemm(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, _lda: usize, _ldb: usize, ldc: usize) {
        unsafe {
            use std::arch::x86_64::*;
            for i in (0..m).step_by(8) {
                for j in (0..n).step_by(4) {
                    let mut acc = [0.0f32; 32];
                    for p in 0..k {
                        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * k + p));
                        for jj in 0..4 {
                            if i + jj < m && j + jj < n {
                                let b_val = _mm256_set1_ps(b[p * n + j + jj]);
                                let mul = _mm256_mul_ps(a_vec, b_val);
                                acc[jj] += _mm256_cvtss_f32(_mm256_hadd_ps(mul, mul));
                            }
                        }
                    }
                    for jj in 0..4 {
                        if i + jj < m && j + jj < n {
                            c[(i + jj) * ldc + (j + jj)] = acc[jj];
                        }
                    }
                }
            }
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn gemm(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, _lda: usize, _ldb: usize, ldc: usize) {
        Fp32ScalarKernel.gemm(a, b, c, m, k, n, _lda, _ldb, ldc);
    }

    fn preferred_block_sizes(&self) -> (usize, usize, usize) {
        (64, 96, 192)
    }

    fn name(&self) -> &'static str { "fp32_avx2" }
}

pub struct Fp32Avx512Kernel;

impl MatmulKernel for Fp32Avx512Kernel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn gemm(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, _lda: usize, _ldb: usize, ldc: usize) {
        // Full AVX-512 implementation would use _mm512_load_ps, _mm512_mul_ps, etc.
        // For brevity, using blocked SIMD approach
        Fp32Avx2Kernel.gemm(a, b, c, m, k, n, _lda, _ldb, ldc);
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn gemm(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, _lda: usize, _ldb: usize, ldc: usize) {
        Fp32ScalarKernel.gemm(a, b, c, m, k, n, _lda, _ldb, ldc);
    }

    fn preferred_block_sizes(&self) -> (usize, usize, usize) {
        (96, 128, 256)
    }

    fn name(&self) -> &'static str { "fp32_avx512" }
}

// --- BF16 Kernels ---

pub struct Bf16Avx512Kernel;

impl MatmulKernel for Bf16Avx512Kernel {
    fn gemm(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, _lda: usize, _ldb: usize, ldc: usize) {
        // BF16 conversion followed by AVX-512 BF16 matmul
        // AVX-512 BF16 provides _mm256_cvtne2ps_pbh for converting float to bf16
        // and _mm256_dpbf16_ps for dot product of bf16 vectors
        // BF16 kernel: requires avx512bf16 intrinsics with proper __m256bh types.
        // Fall back to FP32 AVX-512 for now; full BF16 implementation would use
        // _mm256_cvtne2ps_pbh for float-to-bf16 and _mm256_dpbf16_ps for dot product.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx512f") {
            Fp32Avx512Kernel.gemm(a, b, c, m, k, n, _lda, _ldb, ldc);
        } else {
            Fp32Avx512Kernel.gemm(a, b, c, m, k, n, _lda, _ldb, ldc);
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        Fp32ScalarKernel.gemm(a, b, c, m, k, n, _lda, _ldb, ldc);
    }

    fn name(&self) -> &'static str { "bf16_avx512" }
}

// --- FP16 Kernels ---

pub struct Fp16Avx512Kernel;

impl MatmulKernel for Fp16Avx512Kernel {
    fn gemm(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, _lda: usize, _ldb: usize, ldc: usize) {
        // Similar to BF16 but using standard FP16 conversion
        // FP16 kernel: _mm256_cvtph_ps expects __m128i (not __m256i from _mm256_castps_si256).
        // Fall back to FP32 AVX-512 for now; full FP16 implementation would use proper
        // 128-bit loads and _mm256_cvtph_ps for half-to-float conversion.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx512f") {
            Fp32Avx512Kernel.gemm(a, b, c, m, k, n, _lda, _ldb, ldc);
        } else {
            Fp32Avx512Kernel.gemm(a, b, c, m, k, n, _lda, _ldb, ldc);
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        Fp32ScalarKernel.gemm(a, b, c, m, k, n, _lda, _ldb, ldc);
    }

    fn name(&self) -> &'static str { "fp16_avx512" }
}

// --- INT8 Quantized Kernels ---

pub struct Int8Avx2Kernel;

impl QuantizedKernel for Int8Avx2Kernel {
    fn gemm_quantized(
        &self,
        a: &[f32],
        qweight: &[i8],
        scales: &[f32],
        bias: Option<&[f32]>,
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
        // INT8 AVX2 kernel: _mm256_cvtepi8_ps does not exist; proper approach uses
        // _mm_cvtepi8_epi32 (_mm128i -> _mm256i) then _mm256_cvtepi32_ps.
        // Fall back to scalar for now.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx2") {
            Self::scalar_fallback(a, qweight, scales, bias, c, m, k, n);
        } else {
            // Fallback to scalar
            Self::scalar_fallback(a, qweight, scales, bias, c, m, k, n);
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        Self::scalar_fallback(a, qweight, scales, bias, c, m, k, n);
    }

    fn effective_bits_per_param(&self) -> f32 {
        // INT8 with per-channel scales: 8 bits + 4 bits for scale (fp16) = ~1.5 bits
        1.5
    }

    fn name(&self) -> &'static str { "int8_avx2" }
}

impl Int8Avx2Kernel {
    fn scalar_fallback(
        a: &[f32],
        qweight: &[i8],
        scales: &[f32],
        bias: Option<&[f32]>,
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
        for i in 0..m {
            for o in 0..n {
                let scale = scales[o];
                let mut acc = 0.0f32;
                for idx in 0..k {
                    let w = qweight[idx * n + o] as f32;
                    acc += a[i * k + idx] * w;
                }
                acc *= scale;
                if let Some(b) = bias {
                    acc += b[o];
                }
                c[i * n + o] = acc;
            }
        }
    }
}

pub struct Int8Avx512VnniKernel;

impl QuantizedKernel for Int8Avx512VnniKernel {
    fn gemm_quantized(
        &self,
        a: &[f32],
        qweight: &[i8],
        scales: &[f32],
        bias: Option<&[f32]>,
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
        // INT8 AVX512-VNNI kernel: _mm256_dpbusd_epi32 expects (__m256i, __m256i, __m256i)
        // but we have __m256 and __m512i args. Fall back to INT8 AVX2 scalar for now.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx512vnni") {
            Int8Avx2Kernel.gemm_quantized(a, qweight, scales, bias, c, m, k, n);
        } else {
            Int8Avx2Kernel.gemm_quantized(a, qweight, scales, bias, c, m, k, n);
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        Int8Avx2Kernel.gemm_quantized(a, qweight, scales, bias, c, m, k, n);
    }

    fn effective_bits_per_param(&self) -> f32 {
        1.5
    }

    fn name(&self) -> &'static str { "int8_avx512vnni" }
}

// =========================================================================
// §6 Attention Kernels
// =========================================================================

pub struct FlashAttentionKernel;

impl AttentionKernel for FlashAttentionKernel {
    fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        q_len: usize,
        kv_len: usize,
        head_dim: usize,
        scale: f32,
        causal: bool,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; batch * q_len * head_dim];

        for b in 0..batch {
            // Process in blocks for better cache locality
            let block_size = 64usize.min(kv_len);
            
            for tb in 0..((kv_len + block_size - 1) / block_size) {
                let start_k = tb * block_size;
                let end_k = (start_k + block_size).min(kv_len);
                
                for t in 0..q_len {
                    let q_base = (b * q_len + t) * head_dim;
                    
                    // Compute attention scores for this block
                    let mut block_max = f32::NEG_INFINITY;
                    let mut block_sum = 0.0f32;
                    let mut scores = vec![0.0f32; end_k - start_k];
                    
                    for s in start_k..end_k {
                        if causal && s > t {
                            scores[s - start_k] = f32::NEG_INFINITY;
                            continue;
                        }
                        
                        let k_base = (b * kv_len + s) * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_base + d] * k[k_base + d];
                        }
                        let score = dot * scale;
                        scores[s - start_k] = score;
                        block_max = block_max.max(score);
                    }
                    
                    // Numerically stable softmax
                    for s in 0..end_k - start_k {
                        scores[s] = (scores[s] - block_max).exp();
                        block_sum += scores[s];
                    }
                    block_sum = block_sum.max(1e-12);
                    
                    // Apply attention and accumulate
                    let out_base = (b * q_len + t) * head_dim;
                    for s in start_k..end_k {
                        let w = scores[s - start_k] / block_sum;
                        if w < 1e-12 {
                            continue;
                        }
                        let v_base = (b * kv_len + s) * head_dim;
                        for d in 0..head_dim {
                            output[out_base + d] += w * v[v_base + d];
                        }
                    }
                }
            }
        }

        output
    }

    fn name(&self) -> &'static str { "flash_attention" }
}

pub struct RingAttentionKernel;

impl AttentionKernel for RingAttentionKernel {
    fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        q_len: usize,
        kv_len: usize,
        head_dim: usize,
        scale: f32,
        causal: bool,
    ) -> Vec<f32> {
        // Ring attention distributes KV heads across devices
        // This is a placeholder that falls back to flash attention
        // Full implementation would handle cross-device communication
        let mut output = vec![0.0f32; batch * q_len * head_dim];
        
        // Simplified: process locally like flash attention
        let flash = FlashAttentionKernel;
        let local_output = flash.forward(q, k, v, batch, q_len, kv_len, head_dim, scale, causal);
        
        // In full ring attention, would aggregate from other devices here
        output.copy_from_slice(&local_output);
        
        output
    }

    fn name(&self) -> &'static str { "ring_attention" }
}

// =========================================================================
// §7 Activation Kernels
// =========================================================================

pub struct SiluGeluAvx512Kernel;

impl ActivationKernel for SiluGeluAvx512Kernel {
    fn forward(&self, data: &[f32]) -> Vec<f32> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                use std::arch::x86_64::*;
                let mut output = vec![0.0f32; data.len()];
                let mut i = 0usize;
                
                while i + 16 <= data.len() {
                    // AVX-512 does not provide _mm512_exp_ps, _mm512_neg_ps, or _mm512_tanh_ps.
                    // Use scalar computation for each element in the 16-wide chunk.
                    for j in 0..16 {
                        let x = *data.as_ptr().add(i + j);
                        let k = (2.0 / std::f32::consts::PI).sqrt();
                        let cubic = 0.044715 * x * x * x;
                        let inner = k * (x + cubic);
                        output[i + j] = 0.5 * x * (1.0 + inner.tanh());
                    }
                    i += 16;
                }
                
                // Handle remainder
                while i < data.len() {
                    let x = data[i];
                    let k = (2.0 / std::f32::consts::PI).sqrt();
                    let cubic = 0.044715 * x * x * x;
                    let inner = k * (x + cubic);
                    output[i] = 0.5 * x * (1.0 + inner.tanh());
                    i += 1;
                }
                
                output
            }
        } else {
            // Scalar fallback
            data.iter().map(|&x| {
                let k = (2.0 / std::f32::consts::PI).sqrt();
                let cubic = 0.044715 * x * x * x;
                let inner = k * (x + cubic);
                0.5 * x * (1.0 + inner.tanh())
            }).collect()
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        data.iter().map(|&x| {
            let k = (2.0 / std::f32::consts::PI).sqrt();
            let cubic = 0.044715 * x * x * x;
            let inner = k * (x + cubic);
            0.5 * x * (1.0 + inner.tanh())
        }).collect()
    }

    fn backward(&self, data: &[f32], output: &[f32], upstream_grad: &[f32]) -> Vec<f32> {
        data.iter().zip(output).zip(upstream_grad).map(|((&x, &out), &g)| {
            // GELU gradient: 0.5 * tanh + 0.5 * x * (1 - tanh^2) * (sqrt(2/pi) * (1 + 3*0.044715*x^2))
            let k = (2.0 / std::f32::consts::PI).sqrt();
            let x2 = x * x;
            let x3 = x2 * x;
            let inner = k * (x + 0.044715 * x3);
            let tanh = inner.tanh();
            let d_gelu = 0.5 * (1.0 + tanh) + 0.5 * x * (1.0 - tanh * tanh) * k * (1.0 + 3.0 * 0.044715 * x2);
            g * d_gelu
        }).collect()
    }

    fn name(&self) -> &'static str { "silu_gelu_avx512" }
}

// =========================================================================
// §8 Kernel Pre-Packing Infrastructure
// =========================================================================

/// Pre-packed kernel data structure.
/// Allows kernels to pre-compute and cache data in optimal formats.
pub struct KernelBuffer {
    pub data: Vec<u8>,
    pub dtype: DataType,
    pub shape: Vec<usize>,
    pub layout: MemoryLayout,
}

impl KernelBuffer {
    /// Pack weights for optimized GEMM execution.
    /// Reorganizes weight matrix for better cache behavior.
    pub fn pack_weights_fp32(weights: &[f32], m: usize, k: usize, n: usize) -> Self {
        // Pack in register blocks (e.g., 8x4 for AVX-512)
        let block_m = 8;
        let block_n = 4;
        let packed_m = (m + block_m - 1) / block_m;
        let packed_n = (n + block_n - 1) / block_n;
        
        let mut packed = vec![0.0f32; packed_m * packed_n * block_m * block_n];
        
        for pb in 0..packed_m {
            for qb in 0..packed_n {
                for i in 0..block_m {
                    for j in 0..block_n {
                        let row = pb * block_m + i;
                        let col = qb * block_n + j;
                        if row < m && col < n {
                            packed[(pb * packed_n + qb) * block_m * block_n + i * block_n + j] =
                                weights[row * n + col];
                        }
                    }
                }
            }
        }
        
        KernelBuffer {
            data: unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4).to_vec() },
            dtype: DataType::Fp32,
            shape: vec![packed_m * block_m, packed_n * block_n],
            layout: MemoryLayout::Blocked { block_size: block_m },
        }
    }

    /// Pack weights for INT4 quantized execution.
    pub fn pack_weights_int4(weights: &[f32], scales: &[f32], m: usize, k: usize, n: usize) -> Self {
        let mut packed = vec![0u8; m * ((n + 1) / 2)];
        
        for row in 0..m {
            for col in 0..n {
                let qweight = (weights[row * n + col] / scales[col]).round().clamp(-8.0, 7.0) as i8;
                let byte_idx = col / 2;
                let shift = if col % 2 == 0 { 0 } else { 4 };
                packed[row * ((n + 1) / 2) + byte_idx] |= ((qweight as u8) & 0x0F) << shift;
            }
        }
        
        KernelBuffer {
            data: packed,
            dtype: DataType::Int4,
            shape: vec![m, n],
            layout: MemoryLayout::RowMajor,
        }
    }
}

// =========================================================================
// §9 Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datatype_sizes() {
        assert_eq!(DataType::Fp32.size_bytes(), 4);
        assert_eq!(DataType::Fp16.size_bytes(), 2);
        assert_eq!(DataType::Bf16.size_bytes(), 2);
        assert_eq!(DataType::Int8.size_bytes(), 1);
        assert_eq!(DataType::Int4.size_bytes(), 1);
    }

    #[test]
    fn test_kernel_registry_creation() {
        let registry = KernelRegistry::new();
        assert!(registry.matmul_kernels.contains_key("fp32_scalar"));
        assert!(registry.matmul_kernels.contains_key("bf16_avx512"));
    }

    #[test]
    fn test_memory_layout_strides() {
        let shape = vec![3, 4, 5];
        let strides = MemoryLayout::row_major_strides(&shape);
        assert_eq!(strides, vec![20, 5, 1]);
    }

    #[test]
    fn test_fp32_kernel_gemm() {
        let kernel = Fp32ScalarKernel;
        
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let mut c = vec![0.0, 0.0, 0.0, 0.0];
        
        kernel.gemm(&a, &b, &mut c, 2, 2, 2, 2, 2, 2);
        
        // Expected: [[1*5+3*7, 1*6+3*8], [2*5+4*7, 2*6+4*8]] = [[26, 30], [38, 44]]
        assert_eq!(c[0], 26.0);
        assert_eq!(c[1], 30.0);
        assert_eq!(c[2], 38.0);
        assert_eq!(c[3], 44.0);
    }

    #[test]
    fn test_flash_attention_basic() {
        let kernel = FlashAttentionKernel;
        
        let q = vec![1.0, 0.0, 0.5, 0.0];  // batch=1, q_len=2, head_dim=2
        let k = vec![1.0, 0.0, 0.0, 1.0];  // batch=1, kv_len=2, head_dim=2
        let v = vec![1.0, 2.0, 3.0, 4.0];  // batch=1, kv_len=2, head_dim=2
        
        let output = kernel.forward(&q, &k, &v, 1, 2, 2, 2, 0.7071, false);
        
        // Basic sanity check - output should have correct shape
        assert_eq!(output.len(), 4);
        // Output should be reasonable values (softmax of attention weights applied to values)
        for v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_kernel_buffer_pack_weights() {
        let weights = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ]; // 2x4
        
        let packed = KernelBuffer::pack_weights_fp32(&weights, 2, 4, 4);
        
        assert_eq!(packed.dtype, DataType::Fp32);
        assert_eq!(packed.layout, MemoryLayout::Blocked { block_size: 8 });
    }

    #[test]
    fn test_quantized_kernel_effective_bits() {
        let int8_kernel = Int8Avx2Kernel;
        assert!((int8_kernel.effective_bits_per_param() - 1.5).abs() < 0.1);
    }
}

// =========================================================================
// §10 Integration with ML Engine
// =========================================================================

/// Trait for converting between ML engine tensors and kernel buffers.
pub trait KernelTensorConversion {
    fn to_kernel_buffer(&self, dtype: DataType, layout: MemoryLayout) -> KernelBuffer;
    fn from_kernel_buffer(buffer: &KernelBuffer) -> Vec<f32>;
}

impl KernelTensorConversion for Vec<f32> {
    fn to_kernel_buffer(&self, dtype: DataType, layout: MemoryLayout) -> KernelBuffer {
        let packed_data: Vec<u8> = match dtype {
            DataType::Fp32 => self.iter().flat_map(|&f| f.to_le_bytes().to_vec()).collect(),
            DataType::Fp16 | DataType::Bf16 => {
                self.iter().map(|&f| {
                    let bits = f.to_bits();
                    let fp16 = ((bits >> 16) & 0xFFFF) as u16;
                    fp16.to_le_bytes().to_vec()
                }).flatten().collect()
            }
            DataType::Int8 => {
                // INT8 quantization: scale float values to [-128, 127] range
                // Per-tensor quantization with a single scale factor
                let max_abs = self.iter().map(|f| f.abs()).fold(0.0f32, f32::max);
                let scale = max_abs / 127.0;
                if scale > 0.0 {
                    self.iter().flat_map(|&f| {
                        let q = (f / scale).round().clamp(-128.0, 127.0) as i8;
                        q.to_le_bytes().to_vec()
                    }).collect()
                } else {
                    self.iter().flat_map(|_| 0i8.to_le_bytes().to_vec()).collect()
                }
            }
            DataType::Int4 => {
                // INT4 quantization: pack two values per byte
                let max_abs = self.iter().map(|f| f.abs()).fold(0.0f32, f32::max);
                let scale = max_abs / 7.0;
                let mut packed = Vec::with_capacity((self.len() + 1) / 2);
                for chunk in self.chunks(2) {
                    let lo = if scale > 0.0 {
                        ((chunk[0] / scale).round().clamp(-8.0, 7.0) as i8 & 0x0F) as u8
                    } else { 0 };
                    let hi = if chunk.len() > 1 && scale > 0.0 {
                        ((chunk[1] / scale).round().clamp(-8.0, 7.0) as i8 & 0x0F) as u8
                    } else { 0 };
                    packed.push(lo | (hi << 4));
                }
                packed
            }
            DataType::UInt8 => {
                // UInt8 quantization: scale to [0, 255] range
                let min_val = self.iter().cloned().fold(f32::MAX, f32::min);
                let max_val = self.iter().cloned().fold(f32::MIN, f32::max);
                let range = max_val - min_val;
                let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
                self.iter().flat_map(|&f| {
                    let q = ((f - min_val) / scale).round().clamp(0.0, 255.0) as u8;
                    q.to_le_bytes().to_vec()
                }).collect()
            }
        };
        KernelBuffer {
            data: packed_data,
            dtype,
            shape: vec![self.len()],
            layout,
        }
    }

    fn from_kernel_buffer(buffer: &KernelBuffer) -> Vec<f32> {
        match buffer.dtype {
            DataType::Fp32 => {
                buffer.data.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            }
            DataType::Fp16 | DataType::Bf16 => {
                buffer.data.chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect()
            }
            DataType::Int8 => {
                let max_abs = buffer.data.iter().map(|&b| (b as i8).abs() as f32).fold(0.0f32, f32::max);
                let scale = max_abs / 127.0;
                buffer.data.iter().map(|&b| (b as i8) as f32 * scale).collect()
            }
            DataType::Int4 => {
                let scale = 1.0 / 7.0;
                buffer.data.iter().flat_map(|&byte| {
                    let lo = ((byte & 0x0F) as i8) as f32 * scale;
                    let hi = (((byte >> 4) & 0x0F) as i8) as f32 * scale;
                    vec![lo, hi]
                }).collect()
            }
            DataType::UInt8 => {
                let min_val = 0.0f32;
                let scale = 1.0f32;
                buffer.data.iter().map(|&b| min_val + b as f32 * scale).collect()
            }
        }
    }
}