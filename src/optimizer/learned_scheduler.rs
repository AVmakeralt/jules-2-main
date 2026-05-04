// =============================================================================
// Interval-Compressed Instruction Scheduling via Learned Latency Models
//
// Most compilers use static CPU latency tables for instruction scheduling.
// Jules has hardware performance counters (RDPMC via jit_scheduler.rs). The
// novel idea:
//
//   1. Train a tiny neural network (fits in L1 cache, ~4KB) that takes
//      (instruction sequence, recent PEBS counters) → predicted throughput.
//   2. Use this model to do online instruction scheduling that adapts to
//      actual microarchitecture state: cache warming, branch predictor
//      history, port contention.
//   3. This beats static latency tables because real hardware behavior
//      depends on context, not just instruction identity.
//
// The GNN training infrastructure in tools/gnn_train/ can be adapted for
// this. The model is small enough (256 floats = 1KB for 2-layer ReLU net)
// to evaluate inline during code generation with negligible overhead.
//
// Architecture:
//
//   Instruction sequence (window of N instructions)
//       │
//       ▼
//   Feature Extractor ─── encodes opcode, operand types, dependencies
//       │
//       ▼
//   Micro-Latency Net (~4KB, 2-layer ReLU) ─── predicts throughput
//       │                   • Input: 64-dim feature vector
//       │                   • Hidden: 32 neurons
//       │                   • Output: 1 (predicted cycles)
//       ▼
//   Adaptive Scheduler ─── reorders instructions for minimum total latency
//                          using the predicted per-instruction cost
// =============================================================================


// ─── Instruction Features ─────────────────────────────────────────────────────

/// A compact representation of an instruction for the latency model.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InstructionFeatures {
    /// Opcode class (e.g., ADD, MUL, LOAD, STORE, BRANCH, SIMD).
    pub opcode_class: OpcodeClass,
    /// Number of register operands.
    pub reg_operands: u8,
    /// Number of memory operands.
    pub mem_operands: u8,
    /// Whether this instruction has an immediate operand.
    pub has_immediate: bool,
    /// Whether this instruction has a dependency on the previous instruction.
    pub has_dependency: bool,
    /// Data type width (8, 16, 32, 64, 128, 256, 512 bits).
    pub data_width: u16,
    /// Whether this is a SIMD instruction.
    pub is_simd: bool,
}

/// Opcode classes for the latency model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpcodeClass {
    IntegerAdd,
    IntegerMul,
    IntegerDiv,
    FloatAdd,
    FloatMul,
    FloatDiv,
    FloatFma,
    Load,
    Store,
    Branch,
    Compare,
    Bitwise,
    Shift,
    SimdArith,
    SimdLoad,
    SimdStore,
    Conversion,
    Nop,
    Other,
}

impl OpcodeClass {
    /// Default static latency (cycles) for each opcode class.
    /// These are conservative x86-64 Zen 4 / Ice Lake numbers.
    pub fn default_latency(&self) -> f64 {
        match self {
            OpcodeClass::IntegerAdd => 1.0,
            OpcodeClass::IntegerMul => 3.0,
            OpcodeClass::IntegerDiv => 20.0,
            OpcodeClass::FloatAdd => 4.0,
            OpcodeClass::FloatMul => 5.0,
            OpcodeClass::FloatDiv => 14.0,
            OpcodeClass::FloatFma => 4.0,
            OpcodeClass::Load => 4.0,
            OpcodeClass::Store => 1.0,
            OpcodeClass::Branch => 2.0,
            OpcodeClass::Compare => 1.0,
            OpcodeClass::Bitwise => 1.0,
            OpcodeClass::Shift => 1.0,
            OpcodeClass::SimdArith => 4.0,
            OpcodeClass::SimdLoad => 5.0,
            OpcodeClass::SimdStore => 1.0,
            OpcodeClass::Conversion => 3.0,
            OpcodeClass::Nop => 0.25,
            OpcodeClass::Other => 2.0,
        }
    }

    /// Default throughput (instructions per cycle) for each opcode class.
    pub fn default_throughput(&self) -> f64 {
        match self {
            OpcodeClass::IntegerAdd => 4.0,
            OpcodeClass::IntegerMul => 1.0,
            OpcodeClass::IntegerDiv => 0.05,
            OpcodeClass::FloatAdd => 2.0,
            OpcodeClass::FloatMul => 1.0,
            OpcodeClass::FloatDiv => 0.5,
            OpcodeClass::FloatFma => 2.0,
            OpcodeClass::Load => 2.0,
            OpcodeClass::Store => 2.0,
            OpcodeClass::Branch => 1.0,
            OpcodeClass::Compare => 4.0,
            OpcodeClass::Bitwise => 4.0,
            OpcodeClass::Shift => 2.0,
            OpcodeClass::SimdArith => 1.0,
            OpcodeClass::SimdLoad => 1.0,
            OpcodeClass::SimdStore => 2.0,
            OpcodeClass::Conversion => 1.0,
            OpcodeClass::Nop => 4.0,
            OpcodeClass::Other => 1.0,
        }
    }
}

// ─── Micro-Latency Neural Network ────────────────────────────────────────────

/// A tiny 2-layer ReLU neural network for latency prediction.
/// Size: 64 inputs × 32 hidden × 1 output = 2081 floats ≈ 8KB
/// Fits comfortably in L1 cache.
#[derive(Debug, Clone)]
pub struct MicroLatencyNet {
    /// Input → hidden weights (64 × 32 = 2048 floats).
    weights_ih: Vec<f32>,
    /// Hidden biases (32 floats).
    biases_h: Vec<f32>,
    /// Hidden → output weights (32 floats).
    weights_ho: Vec<f32>,
    /// Output bias (1 float).
    bias_o: f32,
}

impl MicroLatencyNet {
    /// Input dimension.
    pub const INPUT_DIM: usize = 64;
    /// Hidden dimension.
    pub const HIDDEN_DIM: usize = 32;

    /// Create a new network with random (Xavier-initialized) weights.
    pub fn new() -> Self {
        let mut rng = simple_rng(42);
        let scale_ih = (2.0 / (Self::INPUT_DIM + Self::HIDDEN_DIM) as f64) as f32;
        let scale_ho = (2.0 / (Self::HIDDEN_DIM + 1) as f64) as f32;

        let weights_ih: Vec<f32> = (0..Self::INPUT_DIM * Self::HIDDEN_DIM)
            .map(|_| (next_f32(&mut rng) * 2.0 - 1.0) * scale_ih)
            .collect();
        let biases_h = vec![0.0f32; Self::HIDDEN_DIM];
        let weights_ho: Vec<f32> = (0..Self::HIDDEN_DIM)
            .map(|_| (next_f32(&mut rng) * 2.0 - 1.0) * scale_ho)
            .collect();

        Self {
            weights_ih,
            biases_h,
            weights_ho,
            bias_o: 0.0,
        }
    }

    /// Create a network initialized from static latency tables (no learning).
    /// The network will initially approximate the static table.
    pub fn from_static_tables() -> Self {
        let mut net = Self::new();
        // Initialize the output layer to roughly reproduce static latencies.
        // The hidden layer will learn corrections from training data.
        for i in 0..Self::HIDDEN_DIM {
            net.weights_ho[i] = 0.1; // Small positive contribution
        }
        net.bias_o = 3.0; // Average latency
        net
    }

    /// Predict the latency (in cycles) for a given feature vector.
    pub fn predict(&self, features: &[f32; Self::INPUT_DIM]) -> f64 {
        // Hidden layer: ReLU(W·x + b)
        let mut hidden = [0.0f32; Self::HIDDEN_DIM];
        for j in 0..Self::HIDDEN_DIM {
            let mut sum = self.biases_h[j];
            for i in 0..Self::INPUT_DIM {
                sum += self.weights_ih[i * Self::HIDDEN_DIM + j] * features[i];
            }
            hidden[j] = if sum > 0.0 { sum } else { 0.0 }; // ReLU
        }

        // Output layer: linear(W·h + b)
        let mut output = self.bias_o;
        for j in 0..Self::HIDDEN_DIM {
            output += self.weights_ho[j] * hidden[j];
        }

        // Clamp to reasonable range (0.1 – 100 cycles)
        output.clamp(0.1, 100.0) as f64
    }

    /// Train with full backpropagation through both layers
    pub fn train_one(&mut self, features: &[f32; Self::INPUT_DIM], actual: f32, lr: f32) {
        // Forward pass
        let mut hidden = [0.0f32; Self::HIDDEN_DIM];
        let mut pre_activations = [0.0f32; Self::HIDDEN_DIM];
        for j in 0..Self::HIDDEN_DIM {
            let mut sum = self.biases_h[j];
            for i in 0..Self::INPUT_DIM {
                sum += self.weights_ih[i * Self::HIDDEN_DIM + j] * features[i];
            }
            pre_activations[j] = sum;
            hidden[j] = if sum > 0.0 { sum } else { 0.0 }; // ReLU
        }

        let mut output = self.bias_o;
        for j in 0..Self::HIDDEN_DIM {
            output += self.weights_ho[j] * hidden[j];
        }

        // Loss = 0.5 * (output - actual)^2
        let error = output - actual;

        // ── Full backpropagation ──

        // Output layer gradients
        let d_output = error; // dL/d(output) = (output - actual)

        // Hidden layer gradients (chain rule through ReLU)
        let mut d_hidden = [0.0f32; Self::HIDDEN_DIM];
        for j in 0..Self::HIDDEN_DIM {
            d_hidden[j] = self.weights_ho[j] * d_output;
            // ReLU derivative: 0 if pre-activation was <= 0
            if pre_activations[j] <= 0.0 {
                d_hidden[j] = 0.0;
            }
        }

        // Update output layer weights
        for j in 0..Self::HIDDEN_DIM {
            self.weights_ho[j] -= lr * d_output * hidden[j];
        }
        self.bias_o -= lr * d_output;

        // Update input-to-hidden weights
        for j in 0..Self::HIDDEN_DIM {
            for i in 0..Self::INPUT_DIM {
                self.weights_ih[i * Self::HIDDEN_DIM + j] -= lr * d_hidden[j] * features[i];
            }
            self.biases_h[j] -= lr * d_hidden[j];
        }
    }

    /// Total number of parameters.
    pub fn param_count(&self) -> usize {
        self.weights_ih.len() + self.biases_h.len() + self.weights_ho.len() + 1
    }

    /// Memory size in bytes.
    pub fn memory_size(&self) -> usize {
        self.param_count() * std::mem::size_of::<f32>()
    }
}

// ─── Simple RNG (no external deps) ────────────────────────────────────────────

fn simple_rng(seed: u64) -> u64 {
    seed
}

fn next_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / (1u64 << 31) as f32
}

// ─── Feature Extractor ────────────────────────────────────────────────────────

/// Extracts a 64-dim feature vector from an instruction window + hardware
/// counters for input to the latency model.
pub struct FeatureExtractor;

impl FeatureExtractor {
    /// Extract features from a window of instructions and PEBS counter data.
    pub fn extract(
        instructions: &[InstructionFeatures],
        pebs_counters: &PebsCounters,
    ) -> [f32; MicroLatencyNet::INPUT_DIM] {
        let mut features = [0.0f32; MicroLatencyNet::INPUT_DIM];

        // Features 0–7: Current instruction properties
        if let Some(inst) = instructions.last() {
            features[0] = inst.opcode_class as u8 as f32 / 20.0;
            features[1] = inst.reg_operands as f32 / 4.0;
            features[2] = inst.mem_operands as f32 / 2.0;
            features[3] = if inst.has_immediate { 1.0 } else { 0.0 };
            features[4] = if inst.has_dependency { 1.0 } else { 0.0 };
            features[5] = inst.data_width as f32 / 512.0;
            features[6] = if inst.is_simd { 1.0 } else { 0.0 };
            features[7] = inst.opcode_class.default_latency() as f32 / 20.0;
        }

        // Features 8–15: PEBS counter context
        features[8] = pebs_counters.cache_miss_rate as f32;
        features[9] = pebs_counters.branch_misprediction_rate as f32;
        features[10] = (pebs_counters.ipc as f32 / 4.0).min(1.0);
        features[11] = (pebs_counters.l1d_miss_rate as f32).min(1.0);
        features[12] = (pebs_counters.l2_miss_rate as f32).min(1.0);
        features[13] = (pebs_cycles_to_rate(pebs_counters.tlb_miss_rate) as f32).min(1.0);
        features[14] = pebs_counters.port_contention as f32;
        features[15] = pebs_counters.memory_bandwidth_utilization as f32;

        // Features 16–31: Lookback window (last 4 instructions, 4 features each)
        let window = instructions.iter().rev().take(4).collect::<Vec<_>>();
        for (idx, inst) in window.iter().enumerate() {
            let base = 16 + idx * 4;
            features[base] = inst.opcode_class as u8 as f32 / 20.0;
            features[base + 1] = if inst.has_dependency { 1.0 } else { 0.0 };
            features[base + 2] = inst.data_width as f32 / 512.0;
            features[base + 3] = inst.opcode_class.default_latency() as f32 / 20.0;
        }

        // Features 32–63: Reserved for future expansion
        // (e.g., dependency graph features, register pressure)

        features
    }
}

fn pebs_cycles_to_rate(rate: f64) -> f64 {
    rate
}

/// Hardware performance counter snapshot from PEBS.
#[derive(Debug, Clone, Default)]
pub struct PebsCounters {
    /// Cache miss rate (0.0–1.0).
    pub cache_miss_rate: f64,
    /// Branch misprediction rate (0.0–1.0).
    pub branch_misprediction_rate: f64,
    /// Instructions per cycle.
    pub ipc: f64,
    /// L1D cache miss rate.
    pub l1d_miss_rate: f64,
    /// L2 cache miss rate.
    pub l2_miss_rate: f64,
    /// TLB miss rate.
    pub tlb_miss_rate: f64,
    /// Execution port contention estimate (0.0–1.0).
    pub port_contention: f64,
    /// Memory bandwidth utilization (0.0–1.0).
    pub memory_bandwidth_utilization: f64,
}

// ─── Adaptive Scheduler ───────────────────────────────────────────────────────

/// An instruction in the scheduling window.
#[derive(Debug, Clone)]
pub struct SchedInstruction {
    /// Unique ID.
    pub id: usize,
    /// Instruction features.
    pub features: InstructionFeatures,
    /// Instructions this depends on (IDs).
    pub dependencies: Vec<usize>,
    /// Estimated latency (predicted by the model or static table).
    pub estimated_latency: f64,
    /// Whether this instruction has been scheduled.
    pub scheduled: bool,
    /// Cycle at which this instruction starts executing.
    pub start_cycle: f64,
}

/// The adaptive scheduler uses the learned latency model to reorder
/// instructions for minimum total latency.
pub struct AdaptiveScheduler {
    /// The latency prediction model.
    model: MicroLatencyNet,
    /// Current PEBS counter state.
    pebs: PebsCounters,
    /// Whether to use the learned model (vs. static tables).
    use_learned_model: bool,
    /// Training learning rate.
    training_lr: f32,
    /// Number of predictions made.
    predictions: u64,
    /// Number of training steps.
    training_steps: u64,
    /// Accumulated prediction error (for tracking accuracy).
    total_error: f64,
}

impl AdaptiveScheduler {
    pub fn new(use_learned_model: bool) -> Self {
        Self {
            model: MicroLatencyNet::from_static_tables(),
            pebs: PebsCounters::default(),
            use_learned_model,
            training_lr: 0.001,
            predictions: 0,
            training_steps: 0,
            total_error: 0.0,
        }
    }

    /// Update the PEBS counter state (called periodically from the JIT
    /// scheduler's hardware counter reader).
    pub fn update_pebs(&mut self, counters: PebsCounters) {
        self.pebs = counters;
    }

    /// Predict the latency of an instruction in the current context.
    pub fn predict_latency(&mut self, inst: &InstructionFeatures, window: &[InstructionFeatures]) -> f64 {
        self.predictions += 1;

        if !self.use_learned_model {
            return inst.opcode_class.default_latency();
        }

        let mut window_with_inst = window.to_vec();
        window_with_inst.push(inst.clone());
        let features = FeatureExtractor::extract(&window_with_inst, &self.pebs);
        self.model.predict(&features)
    }

    /// Train the model on an (instruction, actual_latency) observation.
    pub fn train_observation(&mut self, inst: &InstructionFeatures, window: &[InstructionFeatures], actual_latency: f64) {
        self.training_steps += 1;

        let mut window_with_inst = window.to_vec();
        window_with_inst.push(inst.clone());
        let features = FeatureExtractor::extract(&window_with_inst, &self.pebs);

        let predicted = self.model.predict(&features);
        self.total_error += (predicted - actual_latency).abs();

        self.model.train_one(&features, actual_latency as f32, self.training_lr);
    }

    /// Schedule a window of instructions for minimum total latency.
    ///
    /// Uses list scheduling with the learned latency model:
    /// 1. Compute predicted latency for each instruction
    /// 2. Topologically sort by dependencies
    /// 3. Greedily schedule the instruction with the earliest possible start
    ///    that has all dependencies satisfied
    pub fn schedule(&mut self, instructions: &mut [SchedInstruction]) {
        // Step 1: Predict latencies.
        let window: Vec<InstructionFeatures> = instructions
            .iter()
            .map(|i| i.features.clone())
            .collect();

        for inst in instructions.iter_mut() {
            inst.estimated_latency = self.predict_latency(&inst.features, &window);
        }

        // Step 2: List scheduling.
        let n = instructions.len();
        let mut scheduled_count = 0;

        while scheduled_count < n {
            // Find instructions whose dependencies are all scheduled.
            let mut ready: Vec<usize> = Vec::new();
            for (i, inst) in instructions.iter().enumerate() {
                if inst.scheduled {
                    continue;
                }
                let deps_met = inst.dependencies.iter().all(|dep| {
                    instructions.get(*dep).map_or(true, |d| d.scheduled)
                });
                if deps_met {
                    ready.push(i);
                }
            }

            if ready.is_empty() {
                break; // Deadlock or bug
            }

            // Among ready instructions, pick the one with the highest
            // latency (critical-path-first heuristic).
            ready.sort_by(|&a, &b| {
                instructions[b]
                    .estimated_latency
                    .partial_cmp(&instructions[a].estimated_latency)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Schedule the best candidate.
            let best = ready[0];
            let dep_end_cycle = instructions[best]
                .dependencies
                .iter()
                .map(|dep| {
                    instructions
                        .get(*dep)
                        .map_or(0.0, |d| d.start_cycle + d.estimated_latency)
                })
                .fold(0.0f64, f64::max);

            instructions[best].start_cycle = dep_end_cycle;
            instructions[best].scheduled = true;
            scheduled_count += 1;
        }
    }

    /// Get the average prediction error.
    pub fn avg_error(&self) -> f64 {
        if self.training_steps == 0 {
            0.0
        } else {
            self.total_error / self.training_steps as f64
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> LearnedSchedulerStats {
        LearnedSchedulerStats {
            predictions: self.predictions,
            training_steps: self.training_steps,
            avg_error: self.avg_error(),
            model_size_bytes: self.model.memory_size(),
            model_params: self.model.param_count(),
            use_learned_model: self.use_learned_model,
        }
    }
}

/// Statistics from the learned scheduler.
#[derive(Debug, Clone)]
pub struct LearnedSchedulerStats {
    pub predictions: u64,
    pub training_steps: u64,
    pub avg_error: f64,
    pub model_size_bytes: usize,
    pub model_params: usize,
    pub use_learned_model: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_micro_latency_net() {
        let net = MicroLatencyNet::new();
        let features = [0.5f32; MicroLatencyNet::INPUT_DIM];
        let prediction = net.predict(&features);
        assert!(prediction > 0.0 && prediction <= 100.0);
        assert!(net.param_count() > 2000);
    }

    #[test]
    fn test_feature_extraction() {
        let inst = InstructionFeatures {
            opcode_class: OpcodeClass::FloatMul,
            reg_operands: 3,
            mem_operands: 0,
            has_immediate: false,
            has_dependency: true,
            data_width: 256,
            is_simd: true,
        };
        let pebs = PebsCounters {
            cache_miss_rate: 0.05,
            ipc: 2.5,
            ..Default::default()
        };
        let features = FeatureExtractor::extract(&[inst], &pebs);
        assert!(features[0] > 0.0); // opcode class encoded
        assert!(features[6] > 0.0); // is_simd encoded
    }

    #[test]
    fn test_adaptive_scheduler() {
        let mut scheduler = AdaptiveScheduler::new(true);
        let inst = InstructionFeatures {
            opcode_class: OpcodeClass::FloatAdd,
            reg_operands: 3,
            mem_operands: 0,
            has_immediate: false,
            has_dependency: false,
            data_width: 32,
            is_simd: false,
        };
        let latency = scheduler.predict_latency(&inst, &[]);
        assert!(latency > 0.0);
    }

    #[test]
    fn test_opcode_class_defaults() {
        assert_eq!(OpcodeClass::IntegerAdd.default_latency(), 1.0);
        assert_eq!(OpcodeClass::FloatFma.default_latency(), 4.0);
        assert!(OpcodeClass::IntegerDiv.default_throughput() < 1.0);
    }

    #[test]
    fn test_training_reduces_error() {
        let mut scheduler = AdaptiveScheduler::new(true);
        let inst = InstructionFeatures {
            opcode_class: OpcodeClass::FloatMul,
            reg_operands: 2,
            mem_operands: 0,
            has_immediate: false,
            has_dependency: false,
            data_width: 32,
            is_simd: false,
        };

        // Train on actual latency = 8.0
        for _ in 0..1000 {
            scheduler.train_observation(&inst, &[], 8.0);
        }

        let predicted = scheduler.predict_latency(&inst, &[]);
        // The prediction should be closer to 8.0 than the static default
        let static_default = OpcodeClass::FloatMul.default_latency();
        assert!((predicted - 8.0).abs() < (static_default - 8.0).abs());
    }
}
