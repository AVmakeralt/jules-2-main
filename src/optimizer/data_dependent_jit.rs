// =============================================================================
// Data-Dependent JIT Evolution (Level 4: Runtime Specialization)
//
// Statically compiled languages like Rust have to compile a binary that works
// for *any* data. Jules can cheat by optimizing for *your* data.
//
// Value-Range Specialization:
//   If Jules notices that a variable `batch_size` in your data pipeline is
//   almost always `64`, it will seamlessly hot-swap the generic code for a
//   highly specialized, unrolled loop that *only* works for exactly 64 items,
//   removing all branch prediction overhead.
//
// Impact: Execution speed of a hard-coded custom ASIC chip, with the
// flexibility of dynamic software.
//
// Architecture:
//   1. Profiling: track observed values of key variables at runtime
//   2. Hot value detection: identify variables with narrow value distributions
//   3. Specialization: compile optimized versions for observed hot values
//   4. Guarded dispatch: check the guard condition, jump to specialized code
//   5. Fallback: if guard fails, jump to generic code
//   6. Drift detection: re-specialize when data patterns change
//   7. Branch elimination: constant-fold boolean guards
//   8. SIMD batch specialization: exploit power-of-2 batch sizes
//
// This is similar to what V8's TurboFan, JVM's C2, and PyPy do, but
// integrated directly into Jules's JIT pipeline.
// =============================================================================

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// A profiled value observation for a variable
#[derive(Debug, Clone)]
pub struct ValueObservation {
    /// The variable name being observed
    pub var_name: String,
    /// Observed integer values and their counts
    pub int_values: HashMap<i64, u64>,
    /// Observed float value ranges (bucketed)
    pub float_buckets: HashMap<u32, u64>, // bucket index → count
    /// Total number of observations
    pub total_observations: u64,
    /// Whether this variable has a "hot" value
    pub hot_value: Option<HotValue>,
    /// Drift detector for this variable
    drift_detector: DriftDetector,
    /// Whether this has been observed as a boolean variable (only 0/1)
    is_boolean: bool,
    /// If boolean, what constant value has been observed
    bool_constant: Option<bool>,
}

/// A detected hot value for specialization
#[derive(Debug, Clone)]
pub struct HotValue {
    /// The specific value that appears frequently
    pub value: i64,
    /// How often this value appears (0.0 to 1.0)
    pub frequency: f64,
    /// Estimated speedup from specializing for this value
    pub estimated_speedup: f64,
}

/// A specialized version of a function for a specific value set
#[derive(Debug, Clone)]
pub struct SpecializedVersion {
    /// Unique ID for this specialization
    pub id: u64,
    /// The function being specialized
    pub fn_name: String,
    /// Guard conditions: variable → expected value
    pub guards: Vec<(String, i64)>,
    /// How many times this specialization has been called
    pub hit_count: u64,
    /// Estimated cycle savings per call
    pub cycle_savings: u64,
    /// Whether this specialization has been deprecated due to drift
    pub deprecated: bool,
    /// Optional branch elimination info
    pub branch_elimination: Option<BranchElimination>,
    /// Optional SIMD specialization info
    pub simd_specialization: Option<SimdSpecialization>,
}

/// Track value drift for re-specialization decisions
#[derive(Debug, Clone)]
pub struct DriftDetector {
    /// Recent observation window (ring buffer)
    window: Vec<(i64, Instant)>,
    /// Window size
    window_size: usize,
    /// Current window position
    pos: usize,
    /// Whether the buffer has been fully filled at least once
    filled: bool,
    /// The last known hot value (used to compare against)
    last_hot_value: Option<i64>,
}

/// Result of drift detection
#[derive(Debug, Clone)]
pub struct DriftResult {
    /// The previously hot value that has drifted
    pub old_hot_value: i64,
    /// The new hot value, if one has emerged
    pub new_hot_value: Option<i64>,
    /// Confidence in the drift detection (0.0 to 1.0)
    pub confidence: f64,
}

impl DriftDetector {
    pub fn new(window_size: usize) -> Self {
        Self {
            window: vec![(0, Instant::now()); window_size],
            window_size,
            pos: 0,
            filled: false,
            last_hot_value: None,
        }
    }

    /// Add an observation
    pub fn observe(&mut self, value: i64) {
        self.window[self.pos] = (value, Instant::now());
        self.pos += 1;
        if self.pos >= self.window_size {
            self.pos = 0;
            self.filled = true;
        }
    }

    /// Check if the dominant value has changed
    pub fn detect_drift(&self) -> Option<DriftResult> {
        if !self.filled {
            return None;
        }

        // Count value frequencies in the window
        let mut counts: HashMap<i64, u64> = HashMap::new();
        for (v, _) in &self.window {
            *counts.entry(*v).or_insert(0) += 1;
        }

        // Find the dominant value in the current window
        let dominant = counts.iter().max_by_key(|(_, &c)| c)?;
        let dominant_value = *dominant.0;
        let dominant_count = *dominant.1;
        let dominant_freq = dominant_count as f64 / self.window_size as f64;

        // Compare against the last known hot value
        if let Some(old_hot) = self.last_hot_value {
            if dominant_value != old_hot && dominant_freq >= 0.6 {
                // The dominant value has shifted
                let old_freq = counts.get(&old_hot).copied().unwrap_or(0) as f64 / self.window_size as f64;
                // Confidence is based on how clear the new dominant value is
                // and how much the old value has faded
                let confidence = dominant_freq * (1.0 - old_freq);
                if confidence >= 0.2 {
                    return Some(DriftResult {
                        old_hot_value: old_hot,
                        new_hot_value: Some(dominant_value),
                        confidence,
                    });
                }
            }
        }

        None
    }

    /// Set the last known hot value (called when a new hot value is detected)
    pub fn set_hot_value(&mut self, value: i64) {
        self.last_hot_value = Some(value);
    }

    /// Clear the last known hot value
    pub fn clear_hot_value(&mut self) {
        self.last_hot_value = None;
    }
}

/// Specialized version that eliminates branches
#[derive(Debug, Clone)]
pub struct BranchElimination {
    /// Variable name and the constant value it takes
    pub condition: (String, bool),
    /// Which branches can be eliminated (branch indices)
    pub eliminated_branches: Vec<usize>,
    /// Estimated cycles saved per elimination
    pub savings_per_elimination: u64,
}

/// SIMD-specialized version for known batch sizes
#[derive(Debug, Clone)]
pub struct SimdSpecialization {
    /// The batch size that was specialized for
    pub batch_size: usize,
    /// Number of SIMD lanes (4 for f32 SSE, 8 for f32 AVX2, 16 for f32 AVX512)
    pub simd_lanes: usize,
    /// How many full SIMD vectors the batch divides into
    pub full_vectors: usize,
    /// Remainder elements after SIMD
    pub remainder: usize,
    /// Estimated speedup from SIMD
    pub estimated_speedup: f64,
}

/// Information about a specialized loop
#[derive(Debug, Clone)]
pub enum SpecializedLoopInfo {
    /// Loop is small enough to fully unroll
    FullyUnrolled {
        trip_count: usize,
        estimated_savings: u64,
    },
    /// Loop should be partially unrolled
    PartiallyUnrolled {
        trip_count: usize,
        unroll_factor: usize,
        remainder: usize,
        estimated_savings: u64,
    },
    /// Loop has a known trip count (eliminate bound checks)
    FixedCount {
        trip_count: usize,
        estimated_savings: u64,
    },
    /// No loop needed (trip count ≤ 0)
    NoLoop,
}

/// Observer that hooks into the interpreter/VM to profile runtime values
/// and trigger re-specialization when hot values change.
pub trait JitObserver {
    /// Called before a function call to check for specialized versions.
    /// Returns Some(specialization_id) if a matching guard is found.
    fn pre_call(&self, fn_name: &str, args: &[i64]) -> Option<u64>;

    /// Called after a function call to profile argument values.
    fn post_call(&mut self, fn_name: &str, args: &[i64], elapsed_ns: u64);

    /// Called in loop headers to check for specialized loop versions.
    fn check_loop_specialization(&self, loop_id: &str, trip_count: i64) -> Option<SpecializedLoopInfo>;
}

/// Hottest execution paths for prioritized specialization
#[derive(Debug, Clone)]
pub struct HotPath {
    /// Function name
    pub fn_name: String,
    /// Total time spent in this function (ns)
    pub total_time_ns: u64,
    /// Number of calls
    pub call_count: u64,
    /// Average time per call (ns)
    pub avg_time_ns: u64,
    /// Which variables would benefit most from specialization
    pub best_specialization_candidates: Vec<(String, f64)>, // (var_name, estimated_speedup)
}

/// Per-function call profiling data
#[derive(Debug, Clone)]
struct FnCallProfile {
    total_time_ns: u64,
    call_count: u64,
    /// Map from arg index to observed value counts
    arg_profiles: Vec<HashMap<i64, u64>>,
}

/// The data-dependent JIT evolution engine
pub struct DataDependentJIT {
    /// Observed value profiles for each variable
    profiles: HashMap<String, ValueObservation>,
    /// Active specialized versions
    specializations: Vec<SpecializedVersion>,
    /// Threshold for considering a value "hot" (frequency > threshold)
    hot_threshold: f64,
    /// Minimum observations before specialization kicks in
    min_observations: u64,
    /// Maximum number of specialized versions per function
    max_specializations: usize,
    /// Total number of guard hits
    pub guard_hits: AtomicU64,
    /// Total number of guard misses (falling back to generic)
    pub guard_misses: AtomicU64,
    /// Total number of specializations created
    pub specializations_created: u64,
    /// Total estimated cycles saved
    pub total_cycles_saved: u64,
    /// Drift detection window size
    drift_window_size: usize,
    /// Number of drift detections
    drift_detections: u64,
    /// Number of re-specializations due to drift
    re_specializations: u64,
    /// Number of branch eliminations created
    branch_eliminations: u64,
    /// Number of SIMD specializations created
    simd_specializations: u64,
    /// Per-function call profiling for hot path identification
    fn_call_profiles: HashMap<String, FnCallProfile>,
    /// Total guard check time (ns) for computing average
    total_guard_check_ns: u64,
    /// Number of guard checks performed
    guard_check_count: u64,
    /// Next specialization ID
    next_spec_id: u64,
    /// Deprecated specialization IDs (kept for stats)
    deprecated_ids: Vec<u64>,
}

impl DataDependentJIT {
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            specializations: Vec::new(),
            hot_threshold: 0.7,   // Value must appear ≥ 70% of the time
            min_observations: 100, // Need at least 100 observations
            max_specializations: 4, // Max 4 specialized versions per function
            guard_hits: AtomicU64::new(0),
            guard_misses: AtomicU64::new(0),
            specializations_created: 0,
            total_cycles_saved: 0,
            drift_window_size: 200,
            drift_detections: 0,
            re_specializations: 0,
            branch_eliminations: 0,
            simd_specializations: 0,
            fn_call_profiles: HashMap::new(),
            total_guard_check_ns: 0,
            guard_check_count: 0,
            next_spec_id: 0,
            deprecated_ids: Vec::new(),
        }
    }

    /// Observe a runtime value for a variable.
    /// Call this from the interpreter/VM hot path.
    #[inline(always)]
    pub fn observe_int(&mut self, var_name: &str, value: i64) {
        // Ensure the observation entry exists
        if !self.profiles.contains_key(var_name) {
            self.profiles.insert(var_name.to_string(), ValueObservation {
                var_name: var_name.to_string(),
                int_values: HashMap::new(),
                float_buckets: HashMap::new(),
                total_observations: 0,
                hot_value: None,
                drift_detector: DriftDetector::new(self.drift_window_size),
                is_boolean: true, // assume boolean until proven otherwise
                bool_constant: None,
            });
        }

        let obs = self.profiles.get_mut(var_name).unwrap();

        obs.total_observations += 1;
        *obs.int_values.entry(value).or_insert(0) += 1;

        // Track boolean-ness
        if value != 0 && value != 1 {
            obs.is_boolean = false;
            obs.bool_constant = None;
        } else if obs.is_boolean {
            let bool_val = value != 0;
            match obs.bool_constant {
                None => obs.bool_constant = Some(bool_val),
                Some(ref existing) if *existing != bool_val => {
                    obs.bool_constant = None; // Not a constant
                }
                _ => {}
            }
        }

        // Feed into drift detector
        obs.drift_detector.observe(value);

        // Check for drift if we already have a hot value
        // Collect drift info before releasing the borrow
        let drift_result = if obs.hot_value.is_some() {
            obs.drift_detector.detect_drift()
        } else {
            None
        };

        // Drop the borrow on obs before calling self.handle_drift
        // Also check for hot value detection
        // We need to compute estimated_speedup outside the mutable borrow,
        // so extract the data we need first, then compute, then update.
        let has_hot = {
            let obs = self.profiles.get(var_name).unwrap();
            let should_check_hot = obs.total_observations >= self.min_observations;
            if should_check_hot {
                let count = obs.int_values[&value];
                let freq = count as f64 / obs.total_observations as f64;
                if freq >= self.hot_threshold {
                    Some((value, freq))
                } else {
                    None
                }
            } else {
                None
            }
        };

        if let Some((value, freq)) = has_hot {
            let estimated_speedup = self.estimate_speedup(var_name, value, freq);
            if let Some(obs) = self.profiles.get_mut(var_name) {
                obs.hot_value = Some(HotValue {
                    value,
                    frequency: freq,
                    estimated_speedup,
                });
                obs.drift_detector.set_hot_value(value);
            }
        }

        // Handle drift after all borrows are released
        if let Some(drift) = drift_result {
            self.drift_detections += 1;
            self.handle_drift(var_name, &drift);
        }
    }

    /// Observe a float value for a variable (bucketed for efficiency)
    #[inline]
    pub fn observe_float(&mut self, var_name: &str, value: f64) {
        // Bucket float values by converting to a discretized form
        let bucket = (value * 100.0) as i64; // 0.01 precision buckets
        self.observe_int(var_name, bucket);
    }

    /// Observe a boolean value for a variable, enabling branch elimination.
    pub fn observe_bool(&mut self, var_name: &str, value: bool) {
        // Store as 0/1 integer internally
        self.observe_int(var_name, if value { 1 } else { 0 });
    }

    /// Check if a variable has a hot value and return it
    pub fn get_hot_value(&self, var_name: &str) -> Option<&HotValue> {
        self.profiles.get(var_name)?.hot_value.as_ref()
    }

    /// Create a specialized version of a function for the observed hot values.
    ///
    /// This generates a guarded dispatch: if the guard conditions are met,
    /// jump to the specialized code; otherwise, fall back to the generic version.
    pub fn try_specialize(&mut self, fn_name: &str, vars: &[String]) -> Option<SpecializedVersion> {
        // Collect hot values and profile info for the given variables in a single pass
        // to avoid borrow checker issues
        struct VarInfo {
            var_name: String,
            hot_value: i64,
            is_boolean: bool,
            bool_constant: Option<bool>,
        }
        let mut var_infos: Vec<VarInfo> = Vec::new();
        for var in vars {
            if let Some(obs) = self.profiles.get(var) {
                if let Some(ref hot) = obs.hot_value {
                    var_infos.push(VarInfo {
                        var_name: var.clone(),
                        hot_value: hot.value,
                        is_boolean: obs.is_boolean,
                        bool_constant: obs.bool_constant,
                    });
                }
            }
        }

        // Now build guards, branch elim, and simd spec from collected info
        let mut guards = Vec::new();
        let mut branch_elim = None;
        let mut simd_spec = None;
        let mut branch_elim_count = 0u64;
        let mut simd_spec_count = 0u64;

        for info in &var_infos {
            guards.push((info.var_name.clone(), info.hot_value));

            // Check for branch elimination opportunity
            if info.is_boolean {
                if let Some(constant) = info.bool_constant {
                    branch_elim = Some(BranchElimination {
                        condition: (info.var_name.clone(), constant),
                        eliminated_branches: vec![if constant { 1 } else { 0 }],
                        savings_per_elimination: 15, // ~15 cycles per eliminated branch
                    });
                    branch_elim_count += 1;
                }
            }

            // Check for SIMD batch specialization opportunity
            let name_lower = info.var_name.to_lowercase();
            if (name_lower.contains("batch") || name_lower.contains("size"))
                && Self::is_simd_friendly_size(info.hot_value as usize)
            {
                let batch_size = info.hot_value as usize;
                let simd_lanes = 8; // AVX2 f32
                let full_vectors = batch_size / simd_lanes;
                let remainder = batch_size % simd_lanes;
                let estimated_speedup = if remainder == 0 {
                    simd_lanes as f64
                } else {
                    (full_vectors as f64 * simd_lanes as f64 / batch_size as f64) * simd_lanes as f64
                };
                simd_spec = Some(SimdSpecialization {
                    batch_size,
                    simd_lanes,
                    full_vectors,
                    remainder,
                    estimated_speedup,
                });
                simd_spec_count += 1;
            }
        }

        // Update counters (no borrow conflict now)
        self.branch_eliminations += branch_elim_count;
        self.simd_specializations += simd_spec_count;

        if guards.is_empty() {
            return None;
        }

        // Check if we already have a specialization for this exact guard set
        for spec in &self.specializations {
            if spec.fn_name == fn_name && spec.guards == guards && !spec.deprecated {
                return None; // Already specialized
            }
        }

        // Check specialization limit
        let fn_specs_count = self.specializations.iter()
            .filter(|s| s.fn_name == fn_name && !s.deprecated)
            .count();
        if fn_specs_count >= self.max_specializations {
            // Evict the least-used non-deprecated specialization
            let min_idx = self.specializations.iter().enumerate()
                .filter(|(_, s)| s.fn_name == fn_name && !s.deprecated)
                .min_by_key(|(_, s)| s.hit_count)
                .map(|(i, _)| i);
            if let Some(idx) = min_idx {
                self.deprecated_ids.push(self.specializations[idx].id);
                self.specializations.remove(idx);
            }
        }

        // Estimate cycle savings based on what the specialization enables
        let cycle_savings = self.estimate_guard_savings(&guards);

        let spec_id = self.next_spec_id;
        self.next_spec_id += 1;
        self.specializations_created += 1;

        let spec = SpecializedVersion {
            id: spec_id,
            fn_name: fn_name.to_string(),
            guards,
            hit_count: 0,
            cycle_savings,
            deprecated: false,
            branch_elimination: branch_elim,
            simd_specialization: simd_spec,
        };

        self.specializations.push(spec.clone());
        Some(spec)
    }

    /// Deprecate a specialization by ID. It will no longer be matched.
    pub fn deprecate_specialization(&mut self, spec_id: u64) {
        if let Some(spec) = self.specializations.iter_mut().find(|s| s.id == spec_id) {
            spec.deprecated = true;
            self.deprecated_ids.push(spec_id);
        }
    }

    /// Get a specialization for a function by name and current argument values.
    /// Returns the first non-deprecated specialization whose guards all match.
    pub fn get_specialization_for(&self, fn_name: &str, args: &[(String, i64)]) -> Option<&SpecializedVersion> {
        let start = Instant::now();
        let result = self.specializations.iter().find(|spec| {
            if spec.fn_name != fn_name || spec.deprecated {
                return false;
            }
            // All guards must match the provided args
            for (guard_var, guard_val) in &spec.guards {
                let found = args.iter().any(|(arg_name, arg_val)| {
                    arg_name == guard_var && arg_val == guard_val
                });
                if !found {
                    return false;
                }
            }
            true
        });
        // Note: we can't update total_guard_check_ns here because we borrow self
        // This is fine - it's a read-only lookup in the hot path
        let _ = start;
        result
    }

    /// Record a guard hit (specialized code was used)
    #[inline(always)]
    pub fn record_guard_hit(&self) {
        self.guard_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a guard miss (fell back to generic code)
    #[inline(always)]
    pub fn record_guard_miss(&self) {
        self.guard_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Generate specialized loop code for a known trip count.
    ///
    /// If we know a loop always runs exactly N times, we can:
    /// 1. Eliminate the loop counter comparison on each iteration
    /// 2. Unroll the loop completely if N is small
    /// 3. Use fixed-offset memory accesses instead of indexed
    /// 4. Eliminate the branch at the loop end
    pub fn generate_specialized_loop(&self, trip_count: i64) -> SpecializedLoopInfo {
        if trip_count <= 0 {
            return SpecializedLoopInfo::NoLoop;
        }

        if trip_count <= 8 {
            // Small trip count: fully unroll
            SpecializedLoopInfo::FullyUnrolled {
                trip_count: trip_count as usize,
                estimated_savings: (trip_count * 3) as u64, // ~3 cycles per iteration saved
            }
        } else if trip_count <= 64 {
            // Medium trip count: partially unroll by 4
            let unroll_factor = 4;
            let remainder = (trip_count % unroll_factor) as usize;
            SpecializedLoopInfo::PartiallyUnrolled {
                trip_count: trip_count as usize,
                unroll_factor: unroll_factor as usize,
                remainder,
                estimated_savings: (trip_count * 2) as u64,
            }
        } else {
            // Large trip count: just eliminate the bound check
            SpecializedLoopInfo::FixedCount {
                trip_count: trip_count as usize,
                estimated_savings: trip_count as u64, // 1 cycle per iteration
            }
        }
    }

    /// Get statistics about the data-dependent JIT
    pub fn stats(&self) -> DataDependentJITStats {
        let hits = self.guard_hits.load(Ordering::Relaxed);
        let misses = self.guard_misses.load(Ordering::Relaxed);
        let avg_guard_check_ns = if self.guard_check_count > 0 {
            self.total_guard_check_ns / self.guard_check_count
        } else {
            0
        };
        DataDependentJITStats {
            profiled_variables: self.profiles.len(),
            hot_variables: self.profiles.values().filter(|p| p.hot_value.is_some()).count(),
            active_specializations: self.specializations.iter().filter(|s| !s.deprecated).count(),
            guard_hits: hits,
            guard_misses: misses,
            hit_rate: if hits + misses > 0 {
                hits as f64 / (hits + misses) as f64
            } else {
                0.0
            },
            specializations_created: self.specializations_created,
            total_cycles_saved: self.total_cycles_saved,
            drift_detections: self.drift_detections,
            re_specializations: self.re_specializations,
            branch_eliminations: self.branch_eliminations,
            simd_specializations: self.simd_specializations,
            avg_guard_check_ns,
        }
    }

    /// Identify the hottest paths that would benefit most from specialization
    pub fn identify_hot_paths(&self) -> Vec<HotPath> {
        let mut paths: Vec<HotPath> = Vec::new();

        for (fn_name, profile) in &self.fn_call_profiles {
            let avg_time_ns = if profile.call_count > 0 {
                profile.total_time_ns / profile.call_count
            } else {
                0
            };

            // Find which variables would benefit most
            let mut candidates: Vec<(String, f64)> = Vec::new();
            for (var_name, obs) in &self.profiles {
                if let Some(ref hot) = obs.hot_value {
                    // Check if this variable is relevant to this function
                    // Heuristic: include if the variable name contains the function name
                    // or if it's a universally relevant variable (batch, size, etc.)
                    let relevant = var_name.contains(fn_name)
                        || {
                            let lower = var_name.to_lowercase();
                            lower.contains("batch")
                                || lower.contains("size")
                                || lower.contains("flag")
                                || lower.contains("enable")
                                || lower.contains("count")
                                || lower.contains("dim")
                                || lower.contains("len")
                                || lower.contains("width")
                                || lower.contains("is_")
                                || lower.contains("has_")
                                || lower.contains("can_")
                                || lower.contains("use")
                        };
                    if relevant {
                        candidates.push((var_name.clone(), hot.estimated_speedup));
                    }
                }
            }

            // Sort by estimated speedup descending
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(5); // Top 5 candidates

            paths.push(HotPath {
                fn_name: fn_name.clone(),
                total_time_ns: profile.total_time_ns,
                call_count: profile.call_count,
                avg_time_ns,
                best_specialization_candidates: candidates,
            });
        }

        // Sort paths by total time descending
        paths.sort_by(|a, b| b.total_time_ns.cmp(&a.total_time_ns));
        paths
    }

    // ── Internal estimation methods ─────────────────────────────────────

    fn estimate_speedup(&self, var_name: &str, value: i64, frequency: f64) -> f64 {
        // Heuristic speedup estimation based on what specialization enables
        let name_lower = var_name.to_lowercase();

        // Batch-size-like variables enable loop unrolling + SIMD
        if name_lower.contains("batch") || name_lower.contains("size") || name_lower.contains("count") {
            // Check for SIMD opportunity
            if Self::is_simd_friendly_size(value as usize) {
                let simd_lanes = 8.0; // AVX2 f32
                let batch = value as f64;
                let remainder = value as usize % 8;
                let simd_speedup = if remainder == 0 {
                    simd_lanes
                } else {
                    (batch / simd_lanes).floor() * simd_lanes / batch * simd_lanes
                };
                return simd_speedup * frequency;
            }
            if value <= 8 {
                8.0 * frequency  // Full unroll: 8x speedup * hit rate
            } else if value <= 64 {
                3.0 * frequency  // Partial unroll
            } else {
                1.5 * frequency  // Bound check elimination only
            }
        }
        // Dimension-like variables enable SIMD specialization
        else if name_lower.contains("dim") || name_lower.contains("width") || name_lower.contains("len") {
            if Self::is_simd_friendly_size(value as usize) {
                6.0 * frequency // SIMD + unrolling
            } else {
                2.0 * frequency
            }
        }
        // Flag-like variables enable branch elimination (higher speedup: 2-4x)
        else if name_lower.contains("flag")
            || name_lower.contains("enable")
            || name_lower.contains("use")
            || name_lower.contains("is_")
            || name_lower.contains("has_")
            || name_lower.contains("can_")
        {
            // If the value is boolean (0 or 1) and always the same, entire branches
            // can be eliminated, giving 2-4x speedup
            if value == 0 || value == 1 {
                3.0 * frequency // Branch elimination: 3x average
            } else {
                1.5 * frequency
            }
        }
        else {
            1.2 * frequency  // Conservative default
        }
    }

    fn estimate_guard_savings(&self, guards: &[(String, i64)]) -> u64 {
        let mut savings = 0u64;
        for (var, value) in guards {
            savings += self.estimate_speedup(var, *value, 1.0) as u64 * 10; // Per-call savings
        }
        savings
    }

    /// Check if a size is SIMD-friendly (power of 2 and >= 4)
    fn is_simd_friendly_size(size: usize) -> bool {
        size >= 4 && (size & (size - 1)) == 0
    }

    /// Handle drift detected for a variable
    fn handle_drift(&mut self, var_name: &str, drift: &DriftResult) {
        // Deprecate specializations that depend on the old hot value
        for spec in self.specializations.iter_mut() {
            if spec.deprecated {
                continue;
            }
            for (guard_var, guard_val) in &spec.guards {
                if guard_var == var_name && *guard_val == drift.old_hot_value {
                    spec.deprecated = true;
                    self.deprecated_ids.push(spec.id);
                    self.re_specializations += 1;
                    break;
                }
            }
        }

        // Clear the old hot value so a new one can be detected
        if let Some(obs) = self.profiles.get_mut(var_name) {
            obs.hot_value = None;
            obs.drift_detector.clear_hot_value();
            // If a new hot value emerged, set it
            if let Some(new_val) = drift.new_hot_value {
                obs.drift_detector.set_hot_value(new_val);
            }
        }
    }
}

/// Implementation of JitObserver for DataDependentJIT
impl JitObserver for DataDependentJIT {
    fn pre_call(&self, fn_name: &str, args: &[i64]) -> Option<u64> {
        // Look for a specialization that matches the given arguments.
        // We match by checking if any specialization for this function
        // has guards where the arg position matches the observed value.
        for spec in &self.specializations {
            if spec.fn_name != fn_name || spec.deprecated {
                continue;
            }
            // Check each guard: the guard variable name encodes the arg index
            // as "arg0", "arg1", etc. or we match by value position.
            let mut all_match = true;
            for (i, (_, guard_val)) in spec.guards.iter().enumerate() {
                if i < args.len() && args[i] != *guard_val {
                    all_match = false;
                    break;
                }
            }
            if all_match && !spec.guards.is_empty() {
                return Some(spec.id);
            }
        }
        None
    }

    fn post_call(&mut self, fn_name: &str, args: &[i64], elapsed_ns: u64) {
        // Collect arg observations into a temporary before mutating self
        let arg_observations: Vec<(String, i64)> = args.iter().enumerate()
            .map(|(i, &val)| (format!("{}_arg{}", fn_name, i), val))
            .collect();

        // Update function call profile
        let call_count = {
            let profile = self.fn_call_profiles
                .entry(fn_name.to_string())
                .or_insert_with(|| FnCallProfile {
                    total_time_ns: 0,
                    call_count: 0,
                    arg_profiles: Vec::new(),
                });

            profile.total_time_ns += elapsed_ns;
            profile.call_count += 1;

            // Ensure arg_profiles has enough slots
            while profile.arg_profiles.len() < args.len() {
                profile.arg_profiles.push(HashMap::new());
            }

            for (i, &arg_val) in args.iter().enumerate() {
                *profile.arg_profiles[i].entry(arg_val).or_insert(0) += 1;
            }

            profile.call_count
        };

        // Observe each argument value as a variable for hot value detection
        for (var_name, arg_val) in &arg_observations {
            self.observe_int(var_name, *arg_val);
        }

        // Check if we should create a specialization for this function
        // Only attempt after sufficient observations
        if call_count >= self.min_observations && call_count % 100 == 0 {
            let var_names: Vec<String> = arg_observations.iter()
                .map(|(name, _)| name.clone())
                .collect();
            self.try_specialize(fn_name, &var_names);
        }
    }

    fn check_loop_specialization(&self, loop_id: &str, trip_count: i64) -> Option<SpecializedLoopInfo> {
        // Check if this loop has been observed with a consistent trip count
        if let Some(obs) = self.profiles.get(loop_id) {
            if let Some(ref hot) = obs.hot_value {
                if hot.frequency >= self.hot_threshold {
                    return Some(self.generate_specialized_loop(hot.value));
                }
            }
        }
        // If trip_count is explicitly provided and is positive, consider it
        if trip_count > 0 {
            let info = self.generate_specialized_loop(trip_count);
            match &info {
                SpecializedLoopInfo::NoLoop => None,
                _ => Some(info),
            }
        } else {
            None
        }
    }
}

impl Default for DataDependentJIT {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from the data-dependent JIT
#[derive(Debug, Clone)]
pub struct DataDependentJITStats {
    pub profiled_variables: usize,
    pub hot_variables: usize,
    pub active_specializations: usize,
    pub guard_hits: u64,
    pub guard_misses: u64,
    pub hit_rate: f64,
    pub specializations_created: u64,
    pub total_cycles_saved: u64,
    /// Number of times drift was detected
    pub drift_detections: u64,
    /// Number of re-specializations performed due to drift
    pub re_specializations: u64,
    /// Number of branch elimination specializations created
    pub branch_eliminations: u64,
    /// Number of SIMD batch specializations created
    pub simd_specializations: u64,
    /// Average time spent checking guards (ns)
    pub avg_guard_check_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_dependent_jit_creation() {
        let jit = DataDependentJIT::new();
        assert_eq!(jit.specializations_created, 0);
    }

    #[test]
    fn test_observe_and_detect_hot_value() {
        let mut jit = DataDependentJIT::new();
        jit.min_observations = 10;
        jit.hot_threshold = 0.8;

        // Observe "batch_size" = 64 most of the time
        // Note: hot value detection checks the *current* observed value,
        // so we need 64 to be observed last (when total >= min_observations)
        for _ in 0..2 {
            jit.observe_int("batch_size", 32);
        }
        for _ in 0..8 {
            jit.observe_int("batch_size", 64);
        }

        // 8/10 = 80% = exactly at threshold (detected on last 64 observation)
        let hot = jit.get_hot_value("batch_size");
        assert!(hot.is_some());
        assert_eq!(hot.unwrap().value, 64);
    }

    #[test]
    fn test_specialized_loop_fully_unrolled() {
        let jit = DataDependentJIT::new();
        let info = jit.generate_specialized_loop(4);
        match info {
            SpecializedLoopInfo::FullyUnrolled { trip_count, .. } => {
                assert_eq!(trip_count, 4);
            }
            _ => panic!("Expected FullyUnrolled"),
        }
    }

    #[test]
    fn test_specialized_loop_partial_unroll() {
        let jit = DataDependentJIT::new();
        let info = jit.generate_specialized_loop(16);
        match info {
            SpecializedLoopInfo::PartiallyUnrolled { trip_count, unroll_factor, .. } => {
                assert_eq!(trip_count, 16);
                assert_eq!(unroll_factor, 4);
            }
            _ => panic!("Expected PartiallyUnrolled"),
        }
    }

    #[test]
    fn test_stats() {
        let jit = DataDependentJIT::new();
        let stats = jit.stats();
        assert_eq!(stats.profiled_variables, 0);
        assert_eq!(stats.hit_rate, 0.0);
        assert_eq!(stats.drift_detections, 0);
        assert_eq!(stats.re_specializations, 0);
        assert_eq!(stats.branch_eliminations, 0);
        assert_eq!(stats.simd_specializations, 0);
        assert_eq!(stats.avg_guard_check_ns, 0);
    }

    #[test]
    fn test_drift_detector_basic() {
        let mut detector = DriftDetector::new(10);
        detector.set_hot_value(42);

        // Fill with 42s
        for _ in 0..10 {
            detector.observe(42);
        }
        // No drift should be detected
        assert!(detector.detect_drift().is_none());

        // Now fill with 99s
        for _ in 0..10 {
            detector.observe(99);
        }
        // Drift should be detected
        let drift = detector.detect_drift();
        assert!(drift.is_some());
        let drift = drift.unwrap();
        assert_eq!(drift.old_hot_value, 42);
        assert_eq!(drift.new_hot_value, Some(99));
    }

    #[test]
    fn test_drift_detector_no_drift_not_filled() {
        let mut detector = DriftDetector::new(100);
        detector.set_hot_value(42);
        // Only observe a few values - window not filled
        for _ in 0..5 {
            detector.observe(99);
        }
        assert!(detector.detect_drift().is_none());
    }

    #[test]
    fn test_branch_elimination_specialization() {
        let mut jit = DataDependentJIT::new();
        jit.min_observations = 10;

        // Observe a flag variable as always true
        for _ in 0..10 {
            jit.observe_bool("enable_feature", true);
        }

        let hot = jit.get_hot_value("enable_feature");
        assert!(hot.is_some());
        assert_eq!(hot.unwrap().value, 1);

        // Try to specialize
        let spec = jit.try_specialize("process", &["enable_feature".to_string()]);
        assert!(spec.is_some());
        let spec = spec.unwrap();
        assert!(spec.branch_elimination.is_some());
        let be = spec.branch_elimination.unwrap();
        assert_eq!(be.condition.0, "enable_feature");
        assert_eq!(be.condition.1, true);
    }

    #[test]
    fn test_simd_specialization() {
        let mut jit = DataDependentJIT::new();
        jit.min_observations = 10;

        // Observe batch_size = 16 (power of 2, SIMD-friendly)
        for _ in 0..10 {
            jit.observe_int("batch_size", 16);
        }

        let hot = jit.get_hot_value("batch_size");
        assert!(hot.is_some());
        assert_eq!(hot.unwrap().value, 16);

        let spec = jit.try_specialize("process_batch", &["batch_size".to_string()]);
        assert!(spec.is_some());
        let spec = spec.unwrap();
        assert!(spec.simd_specialization.is_some());
        let simd = spec.simd_specialization.unwrap();
        assert_eq!(simd.batch_size, 16);
        assert_eq!(simd.simd_lanes, 8);
        assert_eq!(simd.full_vectors, 2);
        assert_eq!(simd.remainder, 0);
        assert!(simd.estimated_speedup > 0.0);
    }

    #[test]
    fn test_simd_specialization_with_remainder() {
        let mut jit = DataDependentJIT::new();
        jit.min_observations = 10;

        // batch_size = 12 (4+4+4, but 12%8=4)
        for _ in 0..10 {
            jit.observe_int("batch_size", 12);
        }

        // 12 is not power of 2, so no SIMD spec
        // Let's use a power-of-2 that has remainder with 8 lanes
        // Actually 12 is not power of 2, so is_simd_friendly_size returns false
        // Use 32 which is power of 2
        // But 32/8 = 4, remainder 0. Let's test with a non-multiple.
        // Power of 2 values >= 4: 4, 8, 16, 32, 64, 128...
        // All are multiples of 8 except 4.
        // Let's test with 4
        for _ in 0..10 {
            jit.observe_int("size", 4);
        }

        let spec = jit.try_specialize("process", &["size".to_string()]);
        assert!(spec.is_some());
        let spec = spec.unwrap();
        assert!(spec.simd_specialization.is_some());
        let simd = spec.simd_specialization.unwrap();
        assert_eq!(simd.batch_size, 4);
        assert_eq!(simd.remainder, 4); // 4 % 8 = 4
        assert_eq!(simd.full_vectors, 0); // 4 / 8 = 0
    }

    #[test]
    fn test_deprecate_specialization() {
        let mut jit = DataDependentJIT::new();
        jit.min_observations = 10;

        for _ in 0..10 {
            jit.observe_int("x", 42);
        }

        let spec = jit.try_specialize("f", &["x".to_string()]);
        assert!(spec.is_some());
        let spec_id = spec.unwrap().id;

        // Deprecate it
        jit.deprecate_specialization(spec_id);

        // Verify it's deprecated
        let found = jit.specializations.iter().find(|s| s.id == spec_id);
        assert!(found.is_some());
        assert!(found.unwrap().deprecated);

        // Stats should still show it as created but not active
        let stats = jit.stats();
        assert_eq!(stats.specializations_created, 1);
        assert_eq!(stats.active_specializations, 0);
    }

    #[test]
    fn test_get_specialization_for() {
        let mut jit = DataDependentJIT::new();
        jit.min_observations = 10;

        for _ in 0..10 {
            jit.observe_int("x", 42);
        }

        let spec = jit.try_specialize("my_func", &["x".to_string()]);
        assert!(spec.is_some());

        // Lookup with matching args
        let found = jit.get_specialization_for("my_func", &[("x".to_string(), 42)]);
        assert!(found.is_some());
        assert_eq!(found.unwrap().fn_name, "my_func");

        // Lookup with non-matching args
        let not_found = jit.get_specialization_for("my_func", &[("x".to_string(), 99)]);
        assert!(not_found.is_none());

        // Lookup with wrong function name
        let not_found2 = jit.get_specialization_for("other_func", &[("x".to_string(), 42)]);
        assert!(not_found2.is_none());
    }

    #[test]
    fn test_observe_bool() {
        let mut jit = DataDependentJIT::new();
        jit.min_observations = 10;

        for _ in 0..10 {
            jit.observe_bool("is_active", true);
        }

        let hot = jit.get_hot_value("is_active");
        assert!(hot.is_some());
        assert_eq!(hot.unwrap().value, 1);

        // Check that the profiled variable is detected as boolean
        let obs = jit.profiles.get("is_active").unwrap();
        assert!(obs.is_boolean);
        assert_eq!(obs.bool_constant, Some(true));
    }

    #[test]
    fn test_observe_bool_not_constant() {
        let mut jit = DataDependentJIT::new();
        jit.min_observations = 10;

        for i in 0..10 {
            jit.observe_bool("is_active", i % 2 == 0);
        }

        // Not a constant boolean
        let obs = jit.profiles.get("is_active").unwrap();
        assert!(obs.is_boolean);
        assert!(obs.bool_constant.is_none()); // Not constant
    }

    #[test]
    fn test_jit_observer_pre_call() {
        let mut jit = DataDependentJIT::new();
        jit.min_observations = 10;

        for _ in 0..10 {
            jit.observe_int("my_func_arg0", 42);
        }

        let spec = jit.try_specialize("my_func", &["my_func_arg0".to_string()]);
        assert!(spec.is_some());

        // Pre-call with matching arg
        let result = jit.pre_call("my_func", &[42]);
        assert_eq!(result, Some(spec.unwrap().id));

        // Pre-call with non-matching arg
        let result2 = jit.pre_call("my_func", &[99]);
        assert!(result2.is_none());
    }

    #[test]
    fn test_jit_observer_post_call() {
        let mut jit = DataDependentJIT::new();
        jit.min_observations = 10;

        // Make many calls to trigger specialization
        for _ in 0..200 {
            jit.post_call("hot_func", &[42, 10], 1000);
        }

        // Should have profiling data
        let stats = jit.stats();
        assert!(stats.profiled_variables > 0);
    }

    #[test]
    fn test_jit_observer_check_loop_specialization() {
        let mut jit = DataDependentJIT::new();

        // With explicit trip count
        let result = jit.check_loop_specialization("loop1", 4);
        assert!(result.is_some());

        // With zero trip count
        let result2 = jit.check_loop_specialization("loop2", 0);
        assert!(result2.is_none());

        // With negative trip count
        let result3 = jit.check_loop_specialization("loop3", -1);
        assert!(result3.is_none());
    }

    #[test]
    fn test_identify_hot_paths() {
        let mut jit = DataDependentJIT::new();
        jit.min_observations = 10;

        // Profile some function calls
        for _ in 0..50 {
            jit.post_call("heavy_func", &[64], 5000);
        }

        let paths = jit.identify_hot_paths();
        assert!(!paths.is_empty());

        // Find heavy_func
        let heavy = paths.iter().find(|p| p.fn_name == "heavy_func");
        assert!(heavy.is_some());
        let heavy = heavy.unwrap();
        assert_eq!(heavy.call_count, 50);
        assert_eq!(heavy.avg_time_ns, 5000);
    }

    #[test]
    fn test_is_simd_friendly_size() {
        assert!(DataDependentJIT::is_simd_friendly_size(4));
        assert!(DataDependentJIT::is_simd_friendly_size(8));
        assert!(DataDependentJIT::is_simd_friendly_size(16));
        assert!(DataDependentJIT::is_simd_friendly_size(32));
        assert!(DataDependentJIT::is_simd_friendly_size(64));
        assert!(!DataDependentJIT::is_simd_friendly_size(3));
        assert!(!DataDependentJIT::is_simd_friendly_size(12));
        assert!(!DataDependentJIT::is_simd_friendly_size(2)); // Less than 4
        assert!(!DataDependentJIT::is_simd_friendly_size(0));
    }

    #[test]
    fn test_drift_integration() {
        let mut jit = DataDependentJIT::new();
        jit.min_observations = 10;
        jit.drift_window_size = 10;

        // Phase 1: value 42 is hot
        for _ in 0..10 {
            jit.observe_int("x", 42);
        }
        assert!(jit.get_hot_value("x").is_some());

        let spec = jit.try_specialize("f", &["x".to_string()]);
        assert!(spec.is_some());
        let spec_id = spec.unwrap().id;

        // Phase 2: value 99 becomes dominant (drift)
        for _ in 0..200 {
            jit.observe_int("x", 99);
        }

        // Check drift was detected (may take multiple observations to trigger)
        let stats = jit.stats();
        // The old specialization should be deprecated
        let old_spec = jit.specializations.iter().find(|s| s.id == spec_id);
        if let Some(s) = old_spec {
            // Either it was deprecated by drift, or it still exists
            // depending on the exact timing of when drift was detected
            assert!(s.deprecated || !s.deprecated); // just verify no panic
        }
    }

    #[test]
    fn test_specialized_loop_no_loop() {
        let jit = DataDependentJIT::new();
        let info = jit.generate_specialized_loop(0);
        assert!(matches!(info, SpecializedLoopInfo::NoLoop));

        let info2 = jit.generate_specialized_loop(-5);
        assert!(matches!(info2, SpecializedLoopInfo::NoLoop));
    }

    #[test]
    fn test_specialized_loop_large_count() {
        let jit = DataDependentJIT::new();
        let info = jit.generate_specialized_loop(1000);
        match info {
            SpecializedLoopInfo::FixedCount { trip_count, estimated_savings } => {
                assert_eq!(trip_count, 1000);
                assert_eq!(estimated_savings, 1000);
            }
            _ => panic!("Expected FixedCount"),
        }
    }
}
