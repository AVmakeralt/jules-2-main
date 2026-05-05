// =========================================================================
// Jules ML Engine - ENHANCED MAXIMUM VERSION
// Full-featured automatic differentiation, optimizers, and neural network primitives
// =========================================================================

use std::collections::HashMap;

// =========================================================================
// TENSOR STRUCTURE - Foundation of all ML operations
// =========================================================================

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub requires_grad: bool,
}

impl Tensor {
    // =========================================================================
    // Creation and Initialization Methods
    // =========================================================================

    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Tensor {
            shape,
            data: vec![0.0; numel],
            requires_grad: false,
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Tensor {
            shape,
            data: vec![1.0; numel],
            requires_grad: false,
        }
    }

    pub fn randn(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        let data = (0..numel)
            .map(|i| {
                let u1 = (i as f32 * 12.9898).sin() * 43758.545;
                let u1_frac = u1 - u1.floor();
                let u2 = ((i + 1) as f32 * 78.233).sin() * 43758.545;
                let u2_frac = u2 - u2.floor();
                (-2.0 * u1_frac.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2_frac).cos()
            })
            .collect();
        Tensor {
            shape,
            data,
            requires_grad: false,
        }
    }

    /// Xavier (Glorot) initialization - optimal for sigmoid/tanh
    pub fn xavier(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        let fan_in = shape.get(0).copied().unwrap_or(1) as f32;
        let fan_out = shape.get(1).copied().unwrap_or(1) as f32;
        let limit = (6.0 / (fan_in + fan_out)).sqrt();

        let data = (0..numel)
            .map(|i| {
                let r = ((i as f32 * 12.9898).sin() * 43758.545) % 1.0;
                (r * 2.0 - 1.0) * limit
            })
            .collect();
        Tensor {
            shape,
            data,
            requires_grad: false,
        }
    }

    /// He initialization - optimal for ReLU networks
    pub fn he(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        let fan_in = shape.get(0).copied().unwrap_or(1) as f32;
        let std = (2.0 / fan_in).sqrt();

        let data = (0..numel)
            .map(|i| {
                let u1 = (i as f32 * 12.9898).sin() * 43758.545;
                let u1_frac = u1 - u1.floor();
                let u2 = ((i + 1) as f32 * 78.233).sin() * 43758.545;
                let u2_frac = u2 - u2.floor();
                let z = (-2.0 * u1_frac.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2_frac).cos();
                (z * std).max(-3.0 * std).min(3.0 * std)
            })
            .collect();
        Tensor {
            shape,
            data,
            requires_grad: false,
        }
    }

    /// Orthogonal initialization - preserves gradient flow
    pub fn orthogonal(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        let rows = shape.get(0).copied().unwrap_or(1);
        let cols = shape.get(1).copied().unwrap_or(1);
        let flat_size = rows.min(cols);

        let mut data = vec![0.0; numel];
        for i in 0..flat_size {
            let val = if (i as u32 & 1) == 0 { 1.0 } else { -1.0 };
            data[i * flat_size + i] = val;
        }

        // Add small perturbations for non-square matrices
        for i in flat_size..numel {
            let u = (i as f32 * 12.9898).sin() * 43758.545;
            data[i] = (u - u.floor() - 0.5) * 0.01;
        }

        Tensor {
            shape,
            data,
            requires_grad: false,
        }
    }

    /// Uniform initialization between min and max
    pub fn uniform(shape: Vec<usize>, min: f32, max: f32) -> Self {
        let numel: usize = shape.iter().product();
        let data = (0..numel)
            .map(|i| {
                let u = (i as f32 * 12.9898).sin() * 43758.545;
                let u_norm = u - u.floor();
                min + u_norm * (max - min)
            })
            .collect();
        Tensor {
            shape,
            data,
            requires_grad: false,
        }
    }

    // =========================================================================
    // Basic Properties and Utilities
    // =========================================================================

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn requires_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            new_numel,
            "Cannot reshape: {} elements into {} elements",
            self.numel(),
            new_numel
        );
        Tensor {
            shape: new_shape,
            data: self.data.clone(),
            requires_grad: self.requires_grad,
        }
    }

    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Transpose only works for 2D tensors");
        let (m, n) = (self.shape[0], self.shape[1]);
        let data: Vec<f32> = (0..n)
            .flat_map(|j| (0..m).map(move |i| self.data[i * n + j]))
            .collect();
        Tensor {
            shape: vec![n, m],
            data,
            requires_grad: self.requires_grad,
        }
    }

    pub fn flatten(&self) -> Tensor {
        Tensor {
            shape: vec![self.numel()],
            data: self.data.clone(),
            requires_grad: self.requires_grad,
        }
    }

    pub fn broadcast_to(&self, target_shape: Vec<usize>) -> Tensor {
        if self.shape == target_shape {
            return self.clone();
        }

        let target_numel: usize = target_shape.iter().product();
        let self_numel = self.numel();

        assert_eq!(target_numel % self_numel, 0, "Cannot broadcast shape");

        let mut data = Vec::with_capacity(target_numel);
        let repeat_factor = target_numel / self_numel;
        for _ in 0..repeat_factor {
            data.extend_from_slice(&self.data);
        }

        Tensor {
            shape: target_shape,
            data,
            requires_grad: self.requires_grad,
        }
    }

    // =========================================================================
    // Basic Arithmetic Operations
    // =========================================================================

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch in addition");
        let data = self.data.iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad || other.requires_grad,
        }
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch in subtraction");
        let data = self.data.iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad || other.requires_grad,
        }
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch in multiplication");
        let data = self.data.iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad || other.requires_grad,
        }
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch in division");
        let data = self.data.iter()
            .zip(&other.data)
            .map(|(a, b)| {
                let safe_b = if b.abs() < 1e-10 {
                    1e-10 * if *b >= 0.0 { 1.0 } else { -1.0 }
                } else {
                    *b
                };
                a / safe_b
            })
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad || other.requires_grad,
        }
    }

    pub fn scale(&self, scalar: f32) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| x * scalar).collect(),
            requires_grad: self.requires_grad,
        }
    }

    // =========================================================================
    // Matrix Operations
    // =========================================================================

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Left operand must be 2D");
        assert_eq!(other.shape.len(), 2, "Right operand must be 2D");
        assert_eq!(self.shape[1], other.shape[0], "Incompatible shapes for matmul");

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];
        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += self.data[i * k + p] * other.data[p * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Tensor {
            shape: vec![m, n],
            data: result,
            requires_grad: self.requires_grad || other.requires_grad,
        }
    }

    pub fn batch_matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 3, "Left operand must be 3D");
        assert_eq!(other.shape.len(), 3, "Right operand must be 3D");
        assert_eq!(self.shape[0], other.shape[0], "Batch size mismatch");
        assert_eq!(self.shape[2], other.shape[1], "Incompatible shapes for batch matmul");

        let batch_size = self.shape[0];
        let m = self.shape[1];
        let k = self.shape[2];
        let n = other.shape[2];

        let mut result = vec![0.0; batch_size * m * n];

        for b in 0..batch_size {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for p in 0..k {
                        sum += self.data[b * m * k + i * k + p] * other.data[b * k * n + p * n + j];
                    }
                    result[b * m * n + i * n + j] = sum;
                }
            }
        }

        Tensor {
            shape: vec![batch_size, m, n],
            data: result,
            requires_grad: self.requires_grad || other.requires_grad,
        }
    }

    pub fn matmul_grad_a(&self, upstream_grad: &Tensor, b: &Tensor) -> Tensor {
        let b_t = b.transpose();
        upstream_grad.matmul(&b_t)
    }

    pub fn matmul_grad_b(&self, a: &Tensor, upstream_grad: &Tensor) -> Tensor {
        let a_t = a.transpose();
        a_t.matmul(upstream_grad)
    }

    // =========================================================================
    // Reduction Operations
    // =========================================================================

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn sum_axis(&self, axis: usize) -> Tensor {
        assert!(axis < self.shape.len(), "Axis out of range");

        if self.shape.len() == 1 {
            return Tensor::ones(vec![1]).scale(self.sum());
        }

        let mut new_shape = self.shape.clone();
        new_shape[axis] = 1;

        let stride: usize = self.shape.iter().skip(axis + 1).product();
        let outer_stride: usize = self.shape.iter().take(axis).product();
        let reduce_dim = self.shape[axis];

        let mut result = vec![0.0; self.numel() / reduce_dim];

        for outer_idx in 0..outer_stride {
            for elem_idx in 0..(self.numel() / reduce_dim / outer_stride) {
                let mut sum = 0.0;
                for reduce_idx in 0..reduce_dim {
                    let flat_idx = outer_idx * (reduce_dim * stride) + reduce_idx * stride + elem_idx;
                    if flat_idx < self.data.len() {
                        sum += self.data[flat_idx];
                    }
                }
                result[outer_idx * (self.numel() / reduce_dim / outer_stride) + elem_idx] = sum;
            }
        }

        Tensor {
            shape: new_shape,
            data: result,
            requires_grad: self.requires_grad,
        }
    }

    pub fn mean(&self) -> f32 {
        self.sum() / self.numel().max(1) as f32
    }

    pub fn mean_axis(&self, axis: usize) -> Tensor {
        let sum_tensor = self.sum_axis(axis);
        sum_tensor.scale(1.0 / self.shape[axis] as f32)
    }

    pub fn max(&self) -> f32 {
        self.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    }

    pub fn min(&self) -> f32 {
        self.data.iter().fold(f32::INFINITY, |a, &b| a.min(b))
    }

    pub fn argmax(&self) -> usize {
        self.data.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    pub fn argmin(&self) -> usize {
        self.data.iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    pub fn variance(&self) -> f32 {
        let mean = self.mean();
        let sum_sq_diff: f32 = self.data.iter()
            .map(|x| (x - mean).powi(2))
            .sum();
        sum_sq_diff / self.numel().max(1) as f32
    }

    pub fn std(&self) -> f32 {
        self.variance().sqrt()
    }

    // =========================================================================
    // Activation Functions
    // =========================================================================

    pub fn relu(&self) -> Tensor {
        let data = self.data.iter().map(|x| x.max(0.0)).collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad,
        }
    }

    pub fn leaky_relu(&self, alpha: f32) -> Tensor {
        let data = self.data.iter()
            .map(|x| if *x > 0.0 { *x } else { alpha * x })
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad,
        }
    }

    pub fn elu(&self, alpha: f32) -> Tensor {
        let data = self.data.iter()
            .map(|x| if *x > 0.0 { *x } else { alpha * (x.exp() - 1.0) })
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad,
        }
    }

    pub fn gelu(&self) -> Tensor {
        let data = self.data.iter()
            .map(|x| {
                let cdf = 0.5 * (1.0 + (2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh();
                x * cdf
            })
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad,
        }
    }

    pub fn sigmoid(&self) -> Tensor {
        let data = self.data.iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad,
        }
    }

    pub fn tanh(&self) -> Tensor {
        let data = self.data.iter().map(|x| x.tanh()).collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad,
        }
    }

    pub fn swish(&self) -> Tensor {
        let data = self.data.iter()
            .map(|x| x / (1.0 + (-x).exp()))
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad,
        }
    }

    pub fn mish(&self) -> Tensor {
        let data = self.data.iter()
            .map(|x| x * (1.0 + x.exp().max(1e-7)).ln().tanh())
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad,
        }
    }

    pub fn softmax(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Softmax expects 2D input");

        let batch_size = self.shape[0];
        let features = self.shape[1];
        let mut data = vec![0.0; self.numel()];

        for b in 0..batch_size {
            let max_val = (b * features..(b + 1) * features)
                .map(|i| self.data[i])
                .fold(f32::NEG_INFINITY, |a, b| a.max(b));

            let mut sum_exp = 0.0;
            for i in b * features..(b + 1) * features {
                sum_exp += (self.data[i] - max_val).exp();
            }

            for i in b * features..(b + 1) * features {
                data[i] = (self.data[i] - max_val).exp() / sum_exp;
            }
        }

        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad,
        }
    }

    pub fn log_softmax(&self) -> Tensor {
        let softmax = self.softmax();
        Tensor {
            shape: self.shape.clone(),
            data: softmax.data.iter().map(|x| (x + 1e-12).ln()).collect(),
            requires_grad: self.requires_grad,
        }
    }

    // =========================================================================
    // Activation Gradients
    // =========================================================================

    pub fn relu_grad(&self, upstream_grad: &Tensor) -> Tensor {
        let data = upstream_grad.data.iter()
            .zip(&self.data)
            .map(|(g, x)| if *x > 0.0 { *g } else { 0.0 })
            .collect();
        Tensor {
            shape: upstream_grad.shape.clone(),
            data,
            requires_grad: upstream_grad.requires_grad,
        }
    }

    pub fn leaky_relu_grad(&self, upstream_grad: &Tensor, alpha: f32) -> Tensor {
        let data = upstream_grad.data.iter()
            .zip(&self.data)
            .map(|(g, x)| if *x > 0.0 { *g } else { alpha * g })
            .collect();
        Tensor {
            shape: upstream_grad.shape.clone(),
            data,
            requires_grad: upstream_grad.requires_grad,
        }
    }

    pub fn sigmoid_grad(&self, upstream_grad: &Tensor) -> Tensor {
        let data = upstream_grad.data.iter()
            .zip(&self.data)
            .map(|(g, x)| g * x * (1.0 - x))
            .collect();
        Tensor {
            shape: upstream_grad.shape.clone(),
            data,
            requires_grad: upstream_grad.requires_grad,
        }
    }

    pub fn tanh_grad(&self, upstream_grad: &Tensor) -> Tensor {
        let data = upstream_grad.data.iter()
            .zip(&self.data)
            .map(|(g, x)| g * (1.0 - x * x))
            .collect();
        Tensor {
            shape: upstream_grad.shape.clone(),
            data,
            requires_grad: upstream_grad.requires_grad,
        }
    }

    // =========================================================================
    // Normalization Operations
    // =========================================================================

    /// Batch normalization: computes mean and variance **per channel** (across
    /// batch and spatial dims). Expects tensor shape `[N, C, …]` where dim 1 is
    /// the channel dimension. `gamma`/`beta` should have shape `[C]`.
    pub fn batch_norm(&self, gamma: &Tensor, beta: &Tensor, _momentum: f32, epsilon: f32) -> (Tensor, f32, f32) {
        assert!(self.shape.len() >= 2, "batch_norm requires at least 2D tensor");
        let n = self.shape[0];       // batch size
        let c = self.shape[1];       // channels
        let spatial: usize = if self.shape.len() > 2 {
            self.shape[2..].iter().product()
        } else {
            1
        };
        let count_per_channel = (n * spatial).max(1);

        // Per-channel mean and variance
        let mut channel_means = vec![0.0f32; c];
        let mut channel_vars = vec![0.0f32; c];

        for ch in 0..c {
            let mut sum = 0.0f32;
            for batch in 0..n {
                for sp in 0..spatial {
                    let idx = batch * c * spatial + ch * spatial + sp;
                    sum += self.data[idx];
                }
            }
            channel_means[ch] = sum / count_per_channel as f32;
        }

        for ch in 0..c {
            let mut var_sum = 0.0f32;
            let mean = channel_means[ch];
            for batch in 0..n {
                for sp in 0..spatial {
                    let idx = batch * c * spatial + ch * spatial + sp;
                    let d = self.data[idx] - mean;
                    var_sum += d * d;
                }
            }
            channel_vars[ch] = var_sum / count_per_channel as f32;
        }

        // Normalize per channel
        let data: Vec<f32> = (0..self.numel())
            .map(|i| {
                let ch = (i / spatial) % c;
                let normalized = (self.data[i] - channel_means[ch]) / (channel_vars[ch] + epsilon).sqrt();
                let g = if ch < gamma.data.len() { gamma.data[ch] } else { gamma.data[0] };
                let b_val = if ch < beta.data.len() { beta.data[ch] } else { beta.data[0] };
                g * normalized + b_val
            })
            .collect();

        // Return average mean/variance for API compatibility
        let avg_mean = channel_means.iter().sum::<f32>() / c.max(1) as f32;
        let avg_var = channel_vars.iter().sum::<f32>() / c.max(1) as f32;

        (
            Tensor {
                shape: self.shape.clone(),
                data,
                requires_grad: self.requires_grad,
            },
            avg_mean,
            avg_var,
        )
    }

    /// Layer normalization: computes mean and variance **per sample** (across
    /// feature dims). Expects tensor shape `[N, D]` or `[N, …]` where each
    /// sample (row) is normalized independently. `gamma`/`beta` should have
    /// shape `[D]` (or `[last_dim]`).
    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, epsilon: f32) -> Tensor {
        assert!(self.shape.len() >= 2, "layer_norm requires at least 2D tensor");
        let n = self.shape[0];       // number of samples
        let feat: usize = self.shape[1..].iter().product(); // features per sample

        let mut data = vec![0.0f32; self.numel()];

        for sample in 0..n {
            let base = sample * feat;
            // Per-sample mean
            let mut sum = 0.0f32;
            for d in 0..feat {
                sum += self.data[base + d];
            }
            let mean = sum / feat.max(1) as f32;

            // Per-sample variance
            let mut var_sum = 0.0f32;
            for d in 0..feat {
                let diff = self.data[base + d] - mean;
                var_sum += diff * diff;
            }
            let variance = var_sum / feat.max(1) as f32;

            // Normalize per sample
            for d in 0..feat {
                let normalized = (self.data[base + d] - mean) / (variance + epsilon).sqrt();
                let g = if d < gamma.data.len() { gamma.data[d] } else { gamma.data[0] };
                let b_val = if d < beta.data.len() { beta.data[d] } else { beta.data[0] };
                data[base + d] = g * normalized + b_val;
            }
        }

        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad,
        }
    }

    pub fn normalize(&self) -> Tensor {
        let mean = self.mean();
        let std = self.std() + 1e-8;
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| (x - mean) / std).collect(),
            requires_grad: self.requires_grad,
        }
    }

    // =========================================================================
    // Regularization and Gradient Techniques
    // =========================================================================

    pub fn clip_by_value(&self, min: f32, max: f32) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter()
                .map(|x| x.max(min).min(max))
                .collect(),
            requires_grad: self.requires_grad,
        }
    }

    pub fn clip_by_norm(&self, max_norm: f32) -> Tensor {
        let norm = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm <= max_norm || norm < 1e-8 {
            self.clone()
        } else {
            Tensor {
                shape: self.shape.clone(),
                data: self.data.iter().map(|x| x * (max_norm / norm)).collect(),
                requires_grad: self.requires_grad,
            }
        }
    }

    pub fn dropout(&self, dropout_rate: f32, training: bool) -> Tensor {
        if !training || dropout_rate == 0.0 {
            return self.clone();
        }

        let keep_prob = 1.0 - dropout_rate;
        let scale = 1.0 / keep_prob;

        let data: Vec<f32> = self.data.iter()
            .enumerate()
            .map(|(i, x)| {
                let r = ((i as f32 * 12.9898).sin() * 43758.545) % 1.0;
                if r < keep_prob {
                    x * scale
                } else {
                    0.0
                }
            })
            .collect();

        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad,
        }
    }

    // =========================================================================
    // Concatenation and Stacking
    // =========================================================================

    pub fn concatenate(&self, other: &Tensor, axis: usize) -> Tensor {
        assert_eq!(self.shape.len(), other.shape.len());
        assert!(axis < self.shape.len());

        for i in 0..self.shape.len() {
            if i != axis {
                assert_eq!(self.shape[i], other.shape[i]);
            }
        }

        let mut new_shape = self.shape.clone();
        new_shape[axis] += other.shape[axis];

        let outer_size: usize = if axis > 0 { self.shape[..axis].iter().product() } else { 1 };
        let self_axis = self.shape[axis];
        let other_axis = other.shape[axis];
        let inner_size: usize = if axis + 1 < self.shape.len() { self.shape[axis + 1..].iter().product() } else { 1 };

        let mut data = Vec::with_capacity(self.numel() + other.numel());
        for g in 0..outer_size {
            // Copy self's slice for this group
            let src_offset = g * self_axis * inner_size;
            data.extend_from_slice(&self.data[src_offset..src_offset + self_axis * inner_size]);
            // Copy other's slice for this group
            let src_offset = g * other_axis * inner_size;
            data.extend_from_slice(&other.data[src_offset..src_offset + other_axis * inner_size]);
        }

        Tensor {
            shape: new_shape,
            data,
            requires_grad: self.requires_grad || other.requires_grad,
        }
    }

    pub fn split(&self, axis: usize, indices: Vec<usize>) -> Vec<Tensor> {
        let mut result = Vec::new();
        let mut start = 0;

        let outer_size: usize = if axis > 0 { self.shape[..axis].iter().product() } else { 1 };
        let inner_size: usize = if axis + 1 < self.shape.len() { self.shape[axis + 1..].iter().product() } else { 1 };

        for &idx in &indices {
            let chunk_axis = idx - start;
            let mut new_shape = self.shape.clone();
            new_shape[axis] = chunk_axis;

            let mut data = Vec::with_capacity(outer_size * chunk_axis * inner_size);
            for g in 0..outer_size {
                let src_offset = (g * self.shape[axis] + start) * inner_size;
                data.extend_from_slice(&self.data[src_offset..src_offset + chunk_axis * inner_size]);
            }

            result.push(Tensor {
                shape: new_shape,
                data,
                requires_grad: self.requires_grad,
            });

            start = idx;
        }

        result
    }

    pub fn stack(tensors: &[Tensor], axis: usize) -> Tensor {
        assert!(!tensors.is_empty());

        let first_shape = &tensors[0].shape;
        for tensor in tensors.iter().skip(1) {
            assert_eq!(tensor.shape, *first_shape);
        }

        let mut new_shape = first_shape.clone();
        new_shape.insert(axis, tensors.len());

        let outer_size: usize = if axis > 0 { first_shape[..axis].iter().product() } else { 1 };
        let inner_size: usize = if axis + 1 < first_shape.len() { first_shape[axis..].iter().product() } else { 1 };

        let mut data = Vec::with_capacity(tensors.len() * tensors[0].numel());
        for g in 0..outer_size {
            for tensor in tensors {
                let src_offset = g * inner_size;
                data.extend_from_slice(&tensor.data[src_offset..src_offset + inner_size]);
            }
        }

        Tensor {
            shape: new_shape,
            data,
            requires_grad: tensors.iter().any(|t| t.requires_grad),
        }
    }

    // =========================================================================
    // Utility Methods
    // =========================================================================

    pub fn clone_shape(&self) -> Tensor {
        self.clone()
    }

    pub fn zeros_like(&self) -> Tensor {
        Tensor::zeros(self.shape.clone())
    }

    pub fn ones_like(&self) -> Tensor {
        Tensor::ones(self.shape.clone())
    }

    pub fn abs(&self) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| x.abs()).collect(),
            requires_grad: self.requires_grad,
        }
    }

    pub fn exp(&self) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| x.exp()).collect(),
            requires_grad: self.requires_grad,
        }
    }

    pub fn log(&self) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| (x + 1e-12).ln()).collect(),
            requires_grad: self.requires_grad,
        }
    }

    pub fn sqrt(&self) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| x.sqrt()).collect(),
            requires_grad: self.requires_grad,
        }
    }

    pub fn pow(&self, exponent: f32) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| x.powf(exponent)).collect(),
            requires_grad: self.requires_grad,
        }
    }
}

// =========================================================================
// OPTIMIZATION TECHNIQUES
// =========================================================================

pub struct GradientAccumulator {
    accumulated: HashMap<u64, Tensor>,
    accumulation_steps: u64,
    current_step: u64,
}

impl GradientAccumulator {
    pub fn new(accumulation_steps: u64) -> Self {
        GradientAccumulator {
            accumulated: HashMap::new(),
            accumulation_steps,
            current_step: 0,
        }
    }

    pub fn accumulate(&mut self, param_id: u64, gradient: Tensor) {
        if let Some(acc_grad) = self.accumulated.get_mut(&param_id) {
            *acc_grad = acc_grad.add(&gradient);
        } else {
            self.accumulated.insert(param_id, gradient);
        }
        self.current_step += 1;
    }

    pub fn step(&mut self) {
        self.current_step = 0;
    }

    pub fn should_update(&self) -> bool {
        self.current_step >= self.accumulation_steps
    }

    pub fn get_accumulated(&self, param_id: u64) -> Option<Tensor> {
        self.accumulated.get(&param_id).cloned()
    }

    pub fn clear(&mut self) {
        self.accumulated.clear();
        self.current_step = 0;
    }
}

// =========================================================================
// LOSS FUNCTIONS - Comprehensive Suite
// =========================================================================

pub struct LossFunctions;

impl LossFunctions {
    /// Mean Squared Error
    pub fn mse(predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(predictions.shape, targets.shape);
        predictions.data.iter()
            .zip(&targets.data)
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f32>() / predictions.numel().max(1) as f32
    }

    pub fn mse_gradient(predictions: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(predictions.shape, targets.shape);
        Tensor {
            shape: predictions.shape.clone(),
            data: predictions.data.iter()
                .zip(&targets.data)
                .map(|(p, t)| 2.0 * (p - t) / predictions.numel().max(1) as f32)
                .collect(),
            requires_grad: true,
        }
    }

    /// Cross-Entropy Loss (with numerical stability)
    pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(logits.shape, targets.shape);

        let max_val = logits.max();
        let exp_logits: Vec<f32> = logits.data.iter()
            .map(|x| (x - max_val).exp())
            .collect();
        let sum_exp: f32 = exp_logits.iter().sum();

        logits.data.iter()
            .zip(&targets.data)
            .enumerate()
            .map(|(i, (_logit, target))| {
                let prob = exp_logits[i] / sum_exp;
                -target * (prob.max(1e-12).ln())
            })
            .sum::<f32>() / logits.numel().max(1) as f32
    }

    pub fn cross_entropy_gradient(logits: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(logits.shape, targets.shape);

        let max_val = logits.max();
        let exp_logits: Vec<f32> = logits.data.iter()
            .map(|x| (x - max_val).exp())
            .collect();
        let sum_exp: f32 = exp_logits.iter().sum();

        Tensor {
            shape: logits.shape.clone(),
            data: exp_logits.iter()
                .zip(&targets.data)
                .map(|(prob_unnorm, target)| {
                    let prob = prob_unnorm / sum_exp;
                    (prob - target) / logits.numel().max(1) as f32
                })
                .collect(),
            requires_grad: true,
        }
    }

    /// Binary Cross-Entropy
    pub fn binary_cross_entropy(predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(predictions.shape, targets.shape);
        predictions.data.iter()
            .zip(&targets.data)
            .map(|(p, t)| {
                let p_clipped = p.max(1e-8).min(1.0 - 1e-8);
                -t * p_clipped.ln() - (1.0 - t) * (1.0 - p_clipped).ln()
            })
            .sum::<f32>() / predictions.numel().max(1) as f32
    }

    /// Huber Loss (robust to outliers)
    pub fn huber(predictions: &Tensor, targets: &Tensor, delta: f32) -> f32 {
        assert_eq!(predictions.shape, targets.shape);
        predictions.data.iter()
            .zip(&targets.data)
            .map(|(p, t)| {
                let error = (p - t).abs();
                if error < delta {
                    0.5 * error.powi(2)
                } else {
                    delta * (error - 0.5 * delta)
                }
            })
            .sum::<f32>() / predictions.numel().max(1) as f32
    }

    /// Focal Loss (for imbalanced classification)
    pub fn focal(predictions: &Tensor, targets: &Tensor, gamma: f32) -> f32 {
        assert_eq!(predictions.shape, targets.shape);
        predictions.data.iter()
            .zip(&targets.data)
            .map(|(p, t)| {
                let p_clipped = p.max(1e-8).min(1.0 - 1e-8);
                let ce = -t * p_clipped.ln() - (1.0 - t) * (1.0 - p_clipped).ln();
                ce * (1.0 - p_clipped).powf(gamma)
            })
            .sum::<f32>() / predictions.numel().max(1) as f32
    }

    /// Contrastive Loss (for similarity learning)
    pub fn contrastive(anchor: &Tensor, positive: &Tensor, margin: f32) -> f32 {
        let pos_dist: f32 = anchor.data.iter()
            .zip(&positive.data)
            .map(|(a, p)| (a - p).powi(2))
            .sum::<f32>()
            .sqrt();

        (1.0 * pos_dist.powi(2) + (margin - pos_dist).max(0.0).powi(2)) / 2.0
    }

    /// Ranking Loss (pairwise ranking)
    pub fn ranking(positive_score: f32, negative_score: f32, margin: f32) -> f32 {
        (1.0 + (-margin * (positive_score - negative_score)).exp()).ln()
    }
}

// =========================================================================
// REGULARIZATION TECHNIQUES
// =========================================================================

pub struct Regularization;

impl Regularization {
    /// L2 (Ridge) Regularization
    pub fn l2_penalty(weights: &Tensor, lambda: f32) -> f32 {
        lambda * weights.data.iter()
            .map(|w| w * w)
            .sum::<f32>()
    }

    pub fn l2_gradient(weights: &Tensor, lambda: f32) -> Tensor {
        Tensor {
            shape: weights.shape.clone(),
            data: weights.data.iter().map(|w| 2.0 * lambda * w).collect(),
            requires_grad: true,
        }
    }

    /// L1 (Lasso) Regularization
    pub fn l1_penalty(weights: &Tensor, lambda: f32) -> f32 {
        lambda * weights.data.iter().map(|w| w.abs()).sum::<f32>()
    }

    pub fn l1_gradient(weights: &Tensor, lambda: f32) -> Tensor {
        Tensor {
            shape: weights.shape.clone(),
            data: weights.data.iter().map(|w| lambda * w.signum()).collect(),
            requires_grad: true,
        }
    }

    /// Elastic Net Regularization
    pub fn elastic_net(weights: &Tensor, l1_lambda: f32, l2_lambda: f32) -> f32 {
        Self::l1_penalty(weights, l1_lambda) + Self::l2_penalty(weights, l2_lambda)
    }

    /// Mixup Data Augmentation
    pub fn mixup(x1: &Tensor, x2: &Tensor, y1: f32, y2: f32, alpha: f32) -> (Tensor, f32) {
        let lambda = alpha;
        let x_mixed = x1.scale(lambda).add(&x2.scale(1.0 - lambda));
        let y_mixed = y1 * lambda + y2 * (1.0 - lambda);
        (x_mixed, y_mixed)
    }

    /// CutMix Data Augmentation (simplified for 1D)
    pub fn cutmix(x1: &Tensor, x2: &Tensor, alpha: f32, cut_ratio: f32) -> (Tensor, f32) {
        let lambda = alpha;
        let mut mixed = x1.clone();

        let cut_size = ((x1.numel() as f32) * cut_ratio) as usize;
        let start_idx = (x1.numel() as f32 * (1.0 - cut_ratio) / 2.0) as usize;

        for i in 0..cut_size {
            if start_idx + i < x1.numel() {
                mixed.data[start_idx + i] = x2.data[start_idx + i];
            }
        }

        (mixed, lambda)
    }

    /// Label Smoothing
    pub fn label_smoothing(labels: &Tensor, smoothing: f32) -> Tensor {
        let smoothed = labels.scale(1.0 - smoothing)
            .add(&Tensor::ones(labels.shape.clone()).scale(smoothing / labels.numel() as f32));
        smoothed
    }
}

// =========================================================================
// OPTIMIZER ENHANCEMENTS
// =========================================================================

pub enum Optimizer {
    SGD { learning_rate: f32, momentum: f32, nesterov: bool },
    Adam { learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32 },
    AdamW { learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32 },
    RMSprop { learning_rate: f32, rho: f32, epsilon: f32 },
    LAMB { learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32 },
}

pub struct OptimizerState {
    pub momentum: HashMap<u64, Vec<f32>>,
    pub velocity_m: HashMap<u64, Vec<f32>>,
    pub velocity_v: HashMap<u64, Vec<f32>>,
    pub step_count: u64,
}

impl OptimizerState {
    pub fn new() -> Self {
        OptimizerState {
            momentum: HashMap::new(),
            velocity_m: HashMap::new(),
            velocity_v: HashMap::new(),
            step_count: 0,
        }
    }
}

impl Optimizer {
    pub fn update_weights(
        &self,
        weights: &mut Vec<f32>,
        gradients: &[f32],
        state: &mut OptimizerState,
    ) {
        match self {
            Optimizer::SGD { learning_rate, momentum, nesterov } => {
                Self::sgd_step(weights, gradients, *learning_rate, *momentum, *nesterov, state);
            }
            Optimizer::Adam { learning_rate, beta1, beta2, epsilon } => {
                Self::adam_step(weights, gradients, *learning_rate, *beta1, *beta2, *epsilon, state);
            }
            Optimizer::AdamW { learning_rate, beta1, beta2, epsilon, weight_decay } => {
                Self::adamw_step(weights, gradients, *learning_rate, *beta1, *beta2, *epsilon, *weight_decay, state);
            }
            Optimizer::RMSprop { learning_rate, rho, epsilon } => {
                Self::rmsprop_step(weights, gradients, *learning_rate, *rho, *epsilon, state);
            }
            Optimizer::LAMB { learning_rate, beta1, beta2, epsilon, weight_decay } => {
                Self::lamb_step(weights, gradients, *learning_rate, *beta1, *beta2, *epsilon, *weight_decay, state);
            }
        }
    }

    fn sgd_step(weights: &mut [f32], grads: &[f32], lr: f32, momentum: f32, nesterov: bool, state: &mut OptimizerState) {
        let momentum_vec = state.momentum
            .entry(0)
            .or_insert_with(|| vec![0.0; weights.len()]);

        if momentum_vec.len() != weights.len() {
            momentum_vec.resize(weights.len(), 0.0);
        }

        for i in 0..weights.len() {
            let grad = if i < grads.len() { grads[i] } else { 0.0 };
            momentum_vec[i] = momentum * momentum_vec[i] - lr * grad;

            if nesterov {
                weights[i] += momentum * momentum_vec[i] - lr * grad;
            } else {
                weights[i] += momentum_vec[i];
            }
        }
    }

    fn adam_step(
        weights: &mut [f32],
        grads: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        state: &mut OptimizerState,
    ) {
        state.step_count += 1;
        let step = state.step_count as f32;
        let bias_correction1 = 1.0 - beta1.powf(step);
        let bias_correction2 = 1.0 - beta2.powf(step);

        let m = state.velocity_m
            .entry(0)
            .or_insert_with(|| vec![0.0; weights.len()]);
        let v = state.velocity_v
            .entry(0)
            .or_insert_with(|| vec![0.0; weights.len()]);

        for i in 0..weights.len() {
            if i < grads.len() {
                m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];

                let m_hat = m[i] / bias_correction1;
                let v_hat = v[i] / bias_correction2;

                weights[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
    }

    fn adamw_step(
        weights: &mut [f32],
        grads: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        state: &mut OptimizerState,
    ) {
        Self::adam_step(weights, grads, lr, beta1, beta2, eps, state);
        for w in weights {
            *w *= 1.0 - wd * lr;
        }
    }

    fn rmsprop_step(
        weights: &mut [f32],
        grads: &[f32],
        lr: f32,
        rho: f32,
        eps: f32,
        state: &mut OptimizerState,
    ) {
        let v = state.velocity_v
            .entry(0)
            .or_insert_with(|| vec![0.0; weights.len()]);

        for i in 0..weights.len() {
            if i < grads.len() {
                v[i] = rho * v[i] + (1.0 - rho) * grads[i] * grads[i];
                weights[i] -= lr * grads[i] / (v[i].sqrt() + eps);
            }
        }
    }

    fn lamb_step(
        weights: &mut [f32],
        grads: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        state: &mut OptimizerState,
    ) {
        // LAMB = Layer-wise Adaptive Moments optimizer for Batch training
        state.step_count += 1;
        let step = state.step_count as f32;
        let bias_correction1 = 1.0 - beta1.powf(step);
        let bias_correction2 = 1.0 - beta2.powf(step);

        let m = state.velocity_m
            .entry(0)
            .or_insert_with(|| vec![0.0; weights.len()]);
        let v = state.velocity_v
            .entry(0)
            .or_insert_with(|| vec![0.0; weights.len()]);

        let mut weight_norm = 0.0;
        let mut grad_norm = 0.0;

        for i in 0..weights.len() {
            weight_norm += weights[i] * weights[i];
            if i < grads.len() {
                m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
                grad_norm += (wd * weights[i] + grads[i]) * (wd * weights[i] + grads[i]);
            }
        }

        let weight_norm = weight_norm.sqrt();
        let grad_norm = grad_norm.sqrt();
        let trust_ratio = if grad_norm > 0.0 && weight_norm > 0.0 {
            weight_norm / grad_norm
        } else {
            1.0
        };

        for i in 0..weights.len() {
            if i < grads.len() {
                let m_hat = m[i] / bias_correction1;
                let v_hat = v[i] / bias_correction2;
                let adaptive_lr = trust_ratio * lr;
                weights[i] -= adaptive_lr * m_hat / (v_hat.sqrt() + eps);
                weights[i] *= 1.0 - wd * adaptive_lr;
            }
        }
    }
}

// =========================================================================
// LEARNING RATE SCHEDULERS
// =========================================================================

pub struct LearningRateScheduler {
    initial_lr: f32,
    schedule: ScheduleType,
    step: u64,
}

pub enum ScheduleType {
    Constant,
    Linear { final_lr: f32, total_steps: u64 },
    Exponential { decay_rate: f32 },
    StepDecay { step_size: u64, gamma: f32 },
    CosineAnnealing { total_steps: u64 },
    WarmupLinear { warmup_steps: u64, total_steps: u64 },
    PolynomialDecay { final_lr: f32, total_steps: u64, power: f32 },
}

impl LearningRateScheduler {
    pub fn new(initial_lr: f32, schedule: ScheduleType) -> Self {
        LearningRateScheduler {
            initial_lr,
            schedule,
            step: 0,
        }
    }

    pub fn get_lr(&self) -> f32 {
        match &self.schedule {
            ScheduleType::Constant => self.initial_lr,
            ScheduleType::Linear { final_lr, total_steps } => {
                let progress = (self.step as f32) / (*total_steps as f32);
                self.initial_lr + (final_lr - self.initial_lr) * progress
            }
            ScheduleType::Exponential { decay_rate } => {
                self.initial_lr * decay_rate.powf(self.step as f32)
            }
            ScheduleType::StepDecay { step_size, gamma } => {
                let decay_steps = self.step / step_size;
                self.initial_lr * gamma.powf(decay_steps as f32)
            }
            ScheduleType::CosineAnnealing { total_steps } => {
                let progress = (self.step as f32) / (*total_steps as f32);
                let pi = std::f32::consts::PI;
                self.initial_lr * 0.5 * (1.0 + (pi * progress).cos())
            }
            ScheduleType::WarmupLinear { warmup_steps, total_steps } => {
                if self.step < *warmup_steps {
                    self.initial_lr * (self.step as f32) / (*warmup_steps as f32)
                } else {
                    let remaining = *total_steps - self.step;
                    self.initial_lr * (remaining as f32) / (*total_steps as f32)
                }
            }
            ScheduleType::PolynomialDecay { final_lr, total_steps, power } => {
                let progress = (self.step as f32) / (*total_steps as f32);
                self.initial_lr + (final_lr - self.initial_lr) * progress.powf(*power)
            }
        }
    }

    pub fn step(&mut self) {
        self.step += 1;
    }
}

// =========================================================================
// MODEL CHECKPOINTING
// =========================================================================

pub struct ModelCheckpoint {
    pub weights: Vec<Vec<f32>>,
    pub optimizer_state: OptimizerState,
    pub step: u64,
    pub best_loss: f32,
}

impl ModelCheckpoint {
    pub fn new() -> Self {
        ModelCheckpoint {
            weights: Vec::new(),
            optimizer_state: OptimizerState::new(),
            step: 0,
            best_loss: f32::INFINITY,
        }
    }

    pub fn save_weights(&mut self, weights: &[Vec<f32>]) {
        self.weights = weights.to_vec();
    }

    pub fn load_weights(&self) -> Vec<Vec<f32>> {
        self.weights.clone()
    }
}

// =========================================================================
// METRICS AND EVALUATION
// =========================================================================

pub struct Metrics;

impl Metrics {
    pub fn accuracy(predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(predictions.numel(), targets.numel());
        let correct = predictions.data.iter()
            .zip(&targets.data)
            .filter(|&(p, t)| (p.round() - t).abs() < 0.5)
            .count();
        correct as f32 / predictions.numel() as f32
    }

    pub fn precision(predictions: &Tensor, targets: &Tensor) -> f32 {
        let true_positives = predictions.data.iter()
            .zip(&targets.data)
            .filter(|&(p, t)| *p > 0.5 && *t > 0.5)
            .count() as f32;

        let false_positives = predictions.data.iter()
            .zip(&targets.data)
            .filter(|&(p, t)| *p > 0.5 && *t < 0.5)
            .count() as f32;

        if (true_positives + false_positives).abs() < 1e-8 {
            0.0
        } else {
            true_positives / (true_positives + false_positives)
        }
    }

    pub fn recall(predictions: &Tensor, targets: &Tensor) -> f32 {
        let true_positives = predictions.data.iter()
            .zip(&targets.data)
            .filter(|&(p, t)| *p > 0.5 && *t > 0.5)
            .count() as f32;

        let false_negatives = predictions.data.iter()
            .zip(&targets.data)
            .filter(|&(p, t)| *p < 0.5 && *t > 0.5)
            .count() as f32;

        if (true_positives + false_negatives).abs() < 1e-8 {
            0.0
        } else {
            true_positives / (true_positives + false_negatives)
        }
    }

    pub fn f1_score(predictions: &Tensor, targets: &Tensor) -> f32 {
        let precision = Self::precision(predictions, targets);
        let recall = Self::recall(predictions, targets);

        if (precision + recall).abs() < 1e-8 {
            0.0
        } else {
            2.0 * (precision * recall) / (precision + recall)
        }
    }

    pub fn auc_roc(predictions: &Tensor, targets: &Tensor) -> f32 {
        let mut pairs: Vec<(f32, f32)> = predictions.data.iter()
            .zip(&targets.data)
            .map(|(&p, &t)| (p, t))
            .collect();

        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut prev_tp = 0.0;
        let mut prev_fp = 0.0;
        let mut auc = 0.0;

        let total_pos = pairs.iter().filter(|(_, t)| *t > 0.5).count() as f32;
        let total_neg = pairs.iter().filter(|(_, t)| *t < 0.5).count() as f32;

        if total_pos < 1.0 || total_neg < 1.0 {
            return 0.5;
        }

        for (_, t) in pairs {
            if t > 0.5 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }

            let tpr = tp / total_pos;
            let fpr = fp / total_neg;

            auc += (fpr - prev_fp) * (tpr + prev_tp) / 2.0;

            prev_tp = tpr;
            prev_fp = fpr;
        }

        auc
    }

    pub fn confusion_matrix(predictions: &Tensor, targets: &Tensor) -> (f32, f32, f32, f32) {
        let tp = predictions.data.iter()
            .zip(&targets.data)
            .filter(|&(p, t)| *p > 0.5 && *t > 0.5)
            .count() as f32;

        let fp = predictions.data.iter()
            .zip(&targets.data)
            .filter(|&(p, t)| *p > 0.5 && *t < 0.5)
            .count() as f32;

        let tn = predictions.data.iter()
            .zip(&targets.data)
            .filter(|&(p, t)| *p < 0.5 && *t < 0.5)
            .count() as f32;

        let fn_ = predictions.data.iter()
            .zip(&targets.data)
            .filter(|&(p, t)| *p < 0.5 && *t > 0.5)
            .count() as f32;

        (tp, fp, tn, fn_)
    }
}
