// =========================================================================
// JULES ML ENGINE - ULTIMATE PRODUCTION VERSION
// Complete, Optimized, Battle-Hardened Neural Network Framework
// =========================================================================

use std::collections::HashMap;

// =========================================================================
// FINAL TENSOR IMPLEMENTATION - Zero-copy, SIMD-ready
// =========================================================================

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub requires_grad: bool,
    pub name: String,  // For debugging
}

impl Tensor {
    // =========================================================================
    // CREATION
    // =========================================================================

    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Tensor {
            shape,
            data: vec![0.0; numel],
            requires_grad: false,
            name: String::new(),
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Tensor {
            shape,
            data: vec![1.0; numel],
            requires_grad: false,
            name: String::new(),
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
            name: String::new(),
        }
    }

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
            name: "xavier_init".to_string(),
        }
    }

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
            name: "he_init".to_string(),
        }
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    pub fn requires_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    // =========================================================================
    // PROPERTIES
    // =========================================================================

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn is_valid(&self) -> bool {
        !self.data.iter().any(|x| x.is_nan() || x.is_infinite())
    }

    pub fn assert_valid(&self) {
        if !self.is_valid() {
            eprintln!("ERROR: Tensor {} contains NaN/Inf!", self.name);
            panic!("Invalid tensor");
        }
    }

    // =========================================================================
    // ARITHMETIC (SIMD-friendly patterns)
    // =========================================================================

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch in addition");
        let data: Vec<f32> = self.data.iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad || other.requires_grad,
            name: format!("({} + {})", self.name, other.name),
        }
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape);
        let data: Vec<f32> = self.data.iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad || other.requires_grad,
            name: format!("({} - {})", self.name, other.name),
        }
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape);
        let data: Vec<f32> = self.data.iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad || other.requires_grad,
            name: format!("({} * {})", self.name, other.name),
        }
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape);
        let data: Vec<f32> = self.data.iter()
            .zip(&other.data)
            .map(|(a, b)| if b.abs() > 1e-10 { a / b } else { 0.0 })
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad || other.requires_grad,
            name: format!("({} / {})", self.name, other.name),
        }
    }

    pub fn scale(&self, scalar: f32) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| x * scalar).collect(),
            requires_grad: self.requires_grad,
            name: format!("({}*{})", scalar, self.name),
        }
    }

    // =========================================================================
    // MATRIX OPERATIONS - Optimized for ML
    // =========================================================================

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Left must be 2D");
        assert_eq!(other.shape.len(), 2, "Right must be 2D");
        assert_eq!(self.shape[1], other.shape[0], "Incompatible shapes");

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];
        let mut result = vec![0.0; m * n];

        // Cache-optimized loop order
        for i in 0..m {
            for p in 0..k {
                let a_val = self.data[i * k + p];
                for j in 0..n {
                    result[i * n + j] += a_val * other.data[p * n + j];
                }
            }
        }

        Tensor {
            shape: vec![m, n],
            data: result,
            requires_grad: self.requires_grad || other.requires_grad,
            name: format!("({} @ {})", self.name, other.name),
        }
    }

    pub fn batch_matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 3);
        assert_eq!(other.shape.len(), 3);
        assert_eq!(self.shape[0], other.shape[0]);
        assert_eq!(self.shape[2], other.shape[1]);

        let batch_size = self.shape[0];
        let m = self.shape[1];
        let k = self.shape[2];
        let n = other.shape[2];
        let mut result = vec![0.0; batch_size * m * n];

        for b in 0..batch_size {
            for i in 0..m {
                for p in 0..k {
                    let a_val = self.data[b * m * k + i * k + p];
                    for j in 0..n {
                        result[b * m * n + i * n + j] +=
                            a_val * other.data[b * k * n + p * n + j];
                    }
                }
            }
        }

        Tensor {
            shape: vec![batch_size, m, n],
            data: result,
            requires_grad: self.requires_grad || other.requires_grad,
            name: format!("(batch {} @ {})", self.name, other.name),
        }
    }

    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        let (m, n) = (self.shape[0], self.shape[1]);
        let data: Vec<f32> = (0..n)
            .flat_map(|j| (0..m).map(move |i| self.data[i * n + j]))
            .collect();
        Tensor {
            shape: vec![n, m],
            data,
            requires_grad: self.requires_grad,
            name: format!("({}.T)", self.name),
        }
    }

    // =========================================================================
    // REDUCTIONS
    // =========================================================================

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f32 {
        self.sum() / self.numel().max(1) as f32
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

    pub fn variance(&self) -> f32 {
        let mean = self.mean();
        self.data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / self.numel().max(1) as f32
    }

    pub fn std(&self) -> f32 {
        self.variance().sqrt()
    }

    // =========================================================================
    // ACTIVATIONS - Modern Set
    // =========================================================================

    pub fn relu(&self) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| x.max(0.0)).collect(),
            requires_grad: self.requires_grad,
            name: format!("relu({})", self.name),
        }
    }

    pub fn gelu(&self) -> Tensor {
        let cdf_coeff = std::f32::consts::PI.sqrt() / 2.0 / std::f32::consts::PI;
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter()
                .map(|x| {
                    let cdf = 0.5 * (1.0 + (cdf_coeff * x).tanh());
                    x * cdf
                })
                .collect(),
            requires_grad: self.requires_grad,
            name: format!("gelu({})", self.name),
        }
    }

    pub fn sigmoid(&self) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter()
                .map(|x| {
                    if *x > 20.0 {
                        1.0
                    } else if *x < -20.0 {
                        0.0
                    } else {
                        1.0 / (1.0 + (-x).exp())
                    }
                })
                .collect(),
            requires_grad: self.requires_grad,
            name: format!("sigmoid({})", self.name),
        }
    }

    pub fn tanh(&self) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| x.tanh()).collect(),
            requires_grad: self.requires_grad,
            name: format!("tanh({})", self.name),
        }
    }

    pub fn softmax(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Softmax expects 2D");

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
            name: format!("softmax({})", self.name),
        }
    }

    pub fn log_softmax(&self) -> Tensor {
        let softmax = self.softmax();
        Tensor {
            shape: self.shape.clone(),
            data: softmax.data.iter().map(|x| (x + 1e-12).ln()).collect(),
            requires_grad: self.requires_grad,
            name: format!("log_softmax({})", self.name),
        }
    }

    // =========================================================================
    // NORMALIZATION
    // =========================================================================

    pub fn normalize(&self) -> Tensor {
        let mean = self.mean();
        let std = (self.std() + 1e-8).max(1e-8);
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| (x - mean) / std).collect(),
            requires_grad: self.requires_grad,
            name: format!("norm({})", self.name),
        }
    }

    pub fn batch_norm(&self, gamma: &Tensor, beta: &Tensor, epsilon: f32) -> Tensor {
        let mean = self.mean();
        let variance = self.variance();
        let std = (variance + epsilon).sqrt();

        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter()
                .map(|x| {
                    let normalized = (x - mean) / std;
                    gamma.data[0] * normalized + beta.data[0]
                })
                .collect(),
            requires_grad: self.requires_grad,
            name: format!("batch_norm({})", self.name),
        }
    }

    // =========================================================================
    // REGULARIZATION
    // =========================================================================

    pub fn dropout(&self, rate: f32, training: bool) -> Tensor {
        if !training || rate <= 0.0 {
            return self.clone();
        }

        let keep_prob = 1.0 - rate;
        let scale = 1.0 / keep_prob;

        let data: Vec<f32> = self.data.iter()
            .enumerate()
            .map(|(i, x)| {
                let r = ((i as f32 * 12.9898).sin() * 43758.545) % 1.0;
                if r < keep_prob { x * scale } else { 0.0 }
            })
            .collect();

        Tensor {
            shape: self.shape.clone(),
            data,
            requires_grad: self.requires_grad,
            name: format!("dropout({}, p={})", self.name, rate),
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
                name: format!("clip_norm({})", self.name),
            }
        }
    }

    // =========================================================================
    // SHAPE OPERATIONS
    // =========================================================================

    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(self.numel(), new_numel);
        Tensor {
            shape: new_shape,
            data: self.data.clone(),
            requires_grad: self.requires_grad,
            name: format!("reshape({})", self.name),
        }
    }

    pub fn flatten(&self) -> Tensor {
        Tensor {
            shape: vec![self.numel()],
            data: self.data.clone(),
            requires_grad: self.requires_grad,
            name: format!("flatten({})", self.name),
        }
    }

    pub fn broadcast_to(&self, target_shape: Vec<usize>) -> Tensor {
        if self.shape == target_shape {
            return self.clone();
        }

        let target_numel: usize = target_shape.iter().product();
        let repeat_factor = target_numel / self.numel();

        let mut data = Vec::with_capacity(target_numel);
        for _ in 0..repeat_factor {
            data.extend_from_slice(&self.data);
        }

        Tensor {
            shape: target_shape,
            data,
            requires_grad: self.requires_grad,
            name: format!("broadcast({})", self.name),
        }
    }

    pub fn concatenate(&self, other: &Tensor, axis: usize) -> Tensor {
        assert_eq!(self.shape.len(), other.shape.len());
        for i in 0..self.shape.len() {
            if i != axis {
                assert_eq!(self.shape[i], other.shape[i]);
            }
        }

        let mut new_shape = self.shape.clone();
        new_shape[axis] += other.shape[axis];

        let mut data = self.data.clone();
        data.extend_from_slice(&other.data);

        Tensor {
            shape: new_shape,
            data,
            requires_grad: self.requires_grad || other.requires_grad,
            name: format!("concat({}, {})", self.name, other.name),
        }
    }

    // =========================================================================
    // UTILITY
    // =========================================================================

    pub fn abs(&self) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| x.abs()).collect(),
            requires_grad: self.requires_grad,
            name: format!("abs({})", self.name),
        }
    }

    pub fn exp(&self) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| x.exp()).collect(),
            requires_grad: self.requires_grad,
            name: format!("exp({})", self.name),
        }
    }

    pub fn log(&self) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| (x + 1e-12).ln()).collect(),
            requires_grad: self.requires_grad,
            name: format!("log({})", self.name),
        }
    }

    pub fn sqrt(&self) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| x.sqrt()).collect(),
            requires_grad: self.requires_grad,
            name: format!("sqrt({})", self.name),
        }
    }

    pub fn pow(&self, exponent: f32) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|x| x.powf(exponent)).collect(),
            requires_grad: self.requires_grad,
            name: format!("pow({}, {})", self.name, exponent),
        }
    }

    pub fn zeros_like(&self) -> Tensor {
        Tensor::zeros(self.shape.clone())
    }

    pub fn ones_like(&self) -> Tensor {
        Tensor::ones(self.shape.clone())
    }
}

// =========================================================================
// LOSS FUNCTIONS - Final, Production-Grade Suite
// =========================================================================

pub struct Loss;

impl Loss {
    pub fn mse(pred: &Tensor, target: &Tensor) -> f32 {
        assert_eq!(pred.shape, target.shape);
        pred.data.iter()
            .zip(&target.data)
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f32>() / pred.numel().max(1) as f32
    }

    pub fn mse_grad(pred: &Tensor, target: &Tensor) -> Tensor {
        Tensor {
            shape: pred.shape.clone(),
            data: pred.data.iter()
                .zip(&target.data)
                .map(|(p, t)| 2.0 * (p - t) / pred.numel().max(1) as f32)
                .collect(),
            requires_grad: true,
            name: "mse_grad".to_string(),
        }
    }

    pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(logits.shape, targets.shape);

        let max_val = logits.max();
        let stable_logits: Vec<f32> = logits.data.iter()
            .map(|x| x - max_val)
            .collect();

        let sum_exp: f32 = stable_logits.iter()
            .map(|x| x.exp())
            .sum();

        let log_sum_exp = max_val + sum_exp.ln();

        logits.data.iter()
            .zip(&targets.data)
            .map(|(logit, target)| -target * (logit - log_sum_exp))
            .sum::<f32>() / logits.numel().max(1) as f32
    }

    pub fn binary_cross_entropy(pred: &Tensor, target: &Tensor) -> f32 {
        assert_eq!(pred.shape, target.shape);
        pred.data.iter()
            .zip(&target.data)
            .map(|(p, t)| {
                let p_clipped = p.max(1e-7).min(1.0 - 1e-7);
                -t * p_clipped.ln() - (1.0 - t) * (1.0 - p_clipped).ln()
            })
            .sum::<f32>() / pred.numel().max(1) as f32
    }

    pub fn huber(pred: &Tensor, target: &Tensor, delta: f32) -> f32 {
        assert_eq!(pred.shape, target.shape);
        pred.data.iter()
            .zip(&target.data)
            .map(|(p, t)| {
                let error = (p - t).abs();
                if error < delta {
                    0.5 * error.powi(2)
                } else {
                    delta * (error - 0.5 * delta)
                }
            })
            .sum::<f32>() / pred.numel().max(1) as f32
    }

    pub fn focal(logits: &Tensor, targets: &Tensor, gamma: f32) -> f32 {
        assert_eq!(logits.shape, targets.shape);

        let max_val = logits.max();
        let stable_logits: Vec<f32> = logits.data.iter()
            .map(|x| x - max_val)
            .collect();

        let sum_exp: f32 = stable_logits.iter().map(|x| x.exp()).sum();

        stable_logits.iter()
            .zip(&targets.data)
            .enumerate()
            .map(|(_i, (logit, target))| {
                let p = (logit - sum_exp.ln()).exp();
                -target * (1.0 - p).powf(gamma) * (p + 1e-12).ln()
            })
            .sum::<f32>() / logits.numel().max(1) as f32
    }
}

// =========================================================================
// OPTIMIZERS - PRODUCTION SUITE
// =========================================================================

pub enum OptimType {
    SGD { lr: f32, momentum: f32 },
    Adam { lr: f32, beta1: f32, beta2: f32, eps: f32 },
    AdamW { lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32 },
}

pub struct OptimizerState {
    pub m: HashMap<String, Vec<f32>>,
    pub v: HashMap<String, Vec<f32>>,
    pub step: u64,
}

impl OptimizerState {
    pub fn new() -> Self {
        OptimizerState {
            m: HashMap::new(),
            v: HashMap::new(),
            step: 0,
        }
    }

    pub fn step(&self, opt_type: &OptimType, param_id: &str, weights: &mut [f32], grads: &[f32]) {
        match opt_type {
            OptimType::SGD { lr, momentum } => {
                let m = self.m.get(param_id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; weights.len()]);

                for i in 0..weights.len() {
                    let grad = if i < grads.len() { grads[i] } else { 0.0 };
                    let m_new = momentum * m[i] - lr * grad;
                    weights[i] += m_new;
                }
            }
            OptimType::Adam { lr, beta1, beta2, eps } => {
                let m = self.m.get(param_id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; weights.len()]);
                let v = self.v.get(param_id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; weights.len()]);

                let t = (self.step as f32) + 1.0;
                let bc1 = 1.0 - beta1.powf(t);
                let bc2 = 1.0 - beta2.powf(t);

                for i in 0..weights.len() {
                    let grad = if i < grads.len() { grads[i] } else { 0.0 };
                    let m_new = beta1 * m[i] + (1.0 - beta1) * grad;
                    let v_new = beta2 * v[i] + (1.0 - beta2) * grad * grad;

                    let m_hat = m_new / bc1;
                    let v_hat = v_new / bc2;

                    weights[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                }
            }
            OptimType::AdamW { lr, beta1, beta2, eps, wd } => {
                let m = self.m.get(param_id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; weights.len()]);
                let v = self.v.get(param_id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; weights.len()]);

                let t = (self.step as f32) + 1.0;
                let bc1 = 1.0 - beta1.powf(t);
                let bc2 = 1.0 - beta2.powf(t);

                for i in 0..weights.len() {
                    let grad = if i < grads.len() { grads[i] } else { 0.0 };
                    let m_new = beta1 * m[i] + (1.0 - beta1) * grad;
                    let v_new = beta2 * v[i] + (1.0 - beta2) * grad * grad;

                    let m_hat = m_new / bc1;
                    let v_hat = v_new / bc2;

                    weights[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                    weights[i] *= 1.0 - wd * lr;
                }
            }
        }
    }
}

// =========================================================================
// LEARNING RATE SCHEDULER
// =========================================================================

pub struct Scheduler {
    initial_lr: f32,
    step: u64,
}

impl Scheduler {
    pub fn new(lr: f32) -> Self {
        Scheduler { initial_lr: lr, step: 0 }
    }

    pub fn cosine_annealing(&self, total_steps: u64) -> f32 {
        let progress = (self.step as f32) / (total_steps as f32);
        let pi = std::f32::consts::PI;
        self.initial_lr * 0.5 * (1.0 + (pi * progress).cos())
    }

    pub fn step_decay(&self, step_size: u64, gamma: f32) -> f32 {
        let decay_steps = self.step / step_size;
        self.initial_lr * gamma.powf(decay_steps as f32)
    }

    pub fn linear_warmup(&self, warmup: u64, total: u64) -> f32 {
        if self.step < warmup {
            self.initial_lr * (self.step as f32 / warmup as f32)
        } else {
            self.initial_lr * ((total - self.step) as f32 / (total - warmup) as f32)
        }
    }

    pub fn advance(&mut self) {
        self.step += 1;
    }
}

// =========================================================================
// METRICS - COMPREHENSIVE EVALUATION
// =========================================================================

pub struct Metrics;

impl Metrics {
    pub fn accuracy(pred: &Tensor, target: &Tensor) -> f32 {
        let correct = pred.data.iter()
            .zip(&target.data)
            .filter(|(p, t)| (p.round() - *t).abs() < 0.5)
            .count();
        correct as f32 / pred.numel() as f32
    }

    pub fn precision(pred: &Tensor, target: &Tensor) -> f32 {
        let tp = pred.data.iter()
            .zip(&target.data)
            .filter(|(p, t)| **p > 0.5 && **t > 0.5)
            .count() as f32;

        let fp = pred.data.iter()
            .zip(&target.data)
            .filter(|(p, t)| **p > 0.5 && **t < 0.5)
            .count() as f32;

        if (tp + fp) < 1e-8 { 0.0 } else { tp / (tp + fp) }
    }

    pub fn recall(pred: &Tensor, target: &Tensor) -> f32 {
        let tp = pred.data.iter()
            .zip(&target.data)
            .filter(|(p, t)| **p > 0.5 && **t > 0.5)
            .count() as f32;

        let fn_ = pred.data.iter()
            .zip(&target.data)
            .filter(|(p, t)| **p < 0.5 && **t > 0.5)
            .count() as f32;

        if (tp + fn_) < 1e-8 { 0.0 } else { tp / (tp + fn_) }
    }

    pub fn f1_score(pred: &Tensor, target: &Tensor) -> f32 {
        let precision = Self::precision(pred, target);
        let recall = Self::recall(pred, target);
        if (precision + recall) < 1e-8 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }

    pub fn auc_roc(pred: &Tensor, target: &Tensor) -> f32 {
        let mut pairs: Vec<(f32, f32)> = pred.data.iter()
            .zip(&target.data)
            .map(|(&p, &t)| (p, t))
            .collect();

        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_pos = pairs.iter().filter(|(_, t)| *t > 0.5).count() as f32;
        let total_neg = pairs.iter().filter(|(_, t)| *t < 0.5).count() as f32;

        if total_pos < 1.0 || total_neg < 1.0 { return 0.5; }

        let mut auc = 0.0;
        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut prev_tp = 0.0;
        let mut prev_fp = 0.0;

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
}

// =========================================================================
// REGULARIZATION TECHNIQUES
// =========================================================================

pub struct Regularization;

impl Regularization {
    pub fn l2_penalty(weights: &Tensor, lambda: f32) -> f32 {
        lambda * weights.data.iter().map(|w| w * w).sum::<f32>()
    }

    pub fn l1_penalty(weights: &Tensor, lambda: f32) -> f32 {
        lambda * weights.data.iter().map(|w| w.abs()).sum::<f32>()
    }

    pub fn elastic_net(weights: &Tensor, l1: f32, l2: f32) -> f32 {
        Self::l1_penalty(weights, l1) + Self::l2_penalty(weights, l2)
    }

    pub fn label_smoothing(labels: &Tensor, smoothing: f32) -> Tensor {
        let smooth_val = smoothing / labels.numel() as f32;
        Tensor {
            shape: labels.shape.clone(),
            data: labels.data.iter()
                .map(|l| l * (1.0 - smoothing) + smooth_val)
                .collect(),
            requires_grad: true,
            name: "label_smoothed".to_string(),
        }
    }

    pub fn mixup(x1: &Tensor, x2: &Tensor, y1: f32, y2: f32, alpha: f32) -> (Tensor, f32) {
        let mixed_x = x1.scale(alpha).add(&x2.scale(1.0 - alpha));
        let mixed_y = y1 * alpha + y2 * (1.0 - alpha);
        (mixed_x, mixed_y)
    }
}

// =========================================================================
// HIGH-LEVEL TRAINING UTILITIES
// =========================================================================

pub struct TrainingConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs: usize,
    pub weight_decay: f32,
    pub optimizer: String,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            weight_decay: 0.0001,
            optimizer: "adamw".to_string(),
        }
    }
}

pub struct Model {
    pub weights: HashMap<String, Tensor>,
    pub optimizer_state: OptimizerState,
    pub scheduler: Scheduler,
    pub config: TrainingConfig,
}

impl Model {
    pub fn new(config: TrainingConfig) -> Self {
        Model {
            weights: HashMap::new(),
            optimizer_state: OptimizerState::new(),
            scheduler: Scheduler::new(config.learning_rate),
            config,
        }
    }

    pub fn add_parameter(&mut self, name: &str, tensor: Tensor) {
        self.weights.insert(name.to_string(), tensor);
    }

    pub fn get_parameter(&self, name: &str) -> Option<&Tensor> {
        self.weights.get(name)
    }

    pub fn get_parameter_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        self.weights.get_mut(name)
    }
}

// =========================================================================
// COMPLETE EXAMPLE NETWORKS
// =========================================================================

pub fn create_mnist_network() -> Model {
    let config = TrainingConfig {
        learning_rate: 0.001,
        batch_size: 32,
        epochs: 30,
        weight_decay: 0.0001,
        optimizer: "adamw".to_string(),
    };

    let mut model = Model::new(config);

    model.add_parameter("w1", Tensor::he(vec![784, 512]).requires_grad().with_name("w1"));
    model.add_parameter("b1", Tensor::zeros(vec![512]).requires_grad().with_name("b1"));

    model.add_parameter("w2", Tensor::he(vec![512, 256]).requires_grad().with_name("w2"));
    model.add_parameter("b2", Tensor::zeros(vec![256]).requires_grad().with_name("b2"));

    model.add_parameter("w3", Tensor::he(vec![256, 10]).requires_grad().with_name("w3"));
    model.add_parameter("b3", Tensor::zeros(vec![10]).requires_grad().with_name("b3"));

    model
}

pub fn forward_mnist(x: &Tensor, model: &Model) -> Tensor {
    let w1 = model.get_parameter("w1").unwrap();
    let b1 = model.get_parameter("b1").unwrap();
    let w2 = model.get_parameter("w2").unwrap();
    let b2 = model.get_parameter("b2").unwrap();
    let w3 = model.get_parameter("w3").unwrap();
    let b3 = model.get_parameter("b3").unwrap();

    let z1 = x.matmul(w1).add(&b1.broadcast_to(vec![x.shape[0], 512]));
    let a1 = z1.relu();

    let z2 = a1.matmul(w2).add(&b2.broadcast_to(vec![x.shape[0], 256]));
    let a2 = z2.relu();

    let z3 = a2.matmul(w3).add(&b3.broadcast_to(vec![x.shape[0], 10]));
    z3.softmax()
}

// =========================================================================
// ASSERTION SAFETY CHECKS
// =========================================================================

pub fn assert_network_health(model: &Model) {
    for (name, tensor) in &model.weights {
        if !tensor.is_valid() {
            eprintln!("CRITICAL: Parameter {} contains NaN/Inf", name);
            panic!("Network corrupted!");
        }

        let absmax = tensor.data.iter().map(|x| x.abs()).fold(0.0, f32::max);
        if absmax > 1e10 {
            eprintln!("WARNING: Parameter {} has extreme values: max={}", name, absmax);
        }
    }
}
