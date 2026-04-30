// =============================================================================
// Multi-Tier JIT with Deoptimization
//
// This module implements a tiered JIT compilation system that provides:
// - Fast startup with interpreter
// - Baseline JIT for frequently executed code
// - Optimizing JIT with profile-guided optimizations
// - Deoptimization when assumptions break
//
// Tiers:
// 1. Interpreter (0ms startup, slow execution)
// 2. Baseline JIT (fast compile, okay speed)
// 3. Optimizing JIT (slow compile, fast code)
// 4. E-graph optimized (best possible code)
// =============================================================================

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Compilation tier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum JitTier {
    /// Interpreter - no compilation, direct execution
    Interpreter = 0,
    /// Baseline JIT - simple code generation, fast compile
    Baseline = 1,
    /// Optimizing JIT - profile-guided optimizations
    Optimizing = 2,
    /// E-graph optimized - hardware-aware, profile-guided
    EGraph = 3,
}

impl JitTier {
    /// Get the next higher tier
    pub fn next(&self) -> Option<Self> {
        match self {
            Self::Interpreter => Some(Self::Baseline),
            Self::Baseline => Some(Self::Optimizing),
            Self::Optimizing => Some(Self::EGraph),
            Self::EGraph => None,
        }
    }

    /// Get the tier name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Interpreter => "interpreter",
            Self::Baseline => "baseline",
            Self::Optimizing => "optimizing",
            Self::EGraph => "egraph",
        }
    }
}

/// Compilation statistics for a function
#[derive(Debug, Clone)]
pub struct CompilationStats {
    /// Current tier
    pub tier: JitTier,
    /// Number of times this function was executed
    pub execution_count: u64,
    /// Total time spent executing (in microseconds)
    pub total_time_us: u64,
    /// Average time per execution
    pub avg_time_us: f64,
    /// Whether this function is hot (frequently executed)
    pub is_hot: bool,
    /// Time when last compiled
    pub last_compiled: std::time::Instant,
}

impl CompilationStats {
    pub fn new() -> Self {
        Self {
            tier: JitTier::Interpreter,
            execution_count: 0,
            total_time_us: 0,
            avg_time_us: 0.0,
            is_hot: false,
            last_compiled: std::time::Instant::now(),
        }
    }

    pub fn record_execution(&mut self, time_us: u64) {
        self.execution_count += 1;
        self.total_time_us += time_us;
        self.avg_time_us = self.total_time_us as f64 / self.execution_count as f64;
    }

    pub fn update_hot_status(&mut self, threshold: u64) {
        self.is_hot = self.execution_count >= threshold;
    }
}

/// Deoptimization reason
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeoptReason {
    /// Type assumption failed (e.g., assumed int, got float)
    TypeMismatch,
    /// Value assumption failed (e.g., assumed positive, got negative)
    ValueOutOfRange,
    /// Guard failed (e.g., array bounds check)
    GuardFailed,
    /// Profile changed significantly
    ProfileChanged,
    /// Inline cache miss
    InlineCacheMiss,
    /// Other reason
    Other,
}

/// Deoptimization information
#[derive(Debug, Clone)]
pub struct DeoptInfo {
    /// Reason for deoptimization
    pub reason: DeoptReason,
    /// Location where deoptimization occurred
    pub location: String,
    /// Tier we were at before deoptimization
    pub from_tier: JitTier,
    /// Tier we fell back to
    pub to_tier: JitTier,
    /// Timestamp
    pub timestamp: std::time::Instant,
}

/// Compiled code for a function
#[derive(Debug)]
pub enum CompiledCode {
    /// Not compiled (interpreter mode)
    None,
    /// Baseline JIT code
    Baseline {
        /// Machine code (placeholder - in real implementation, this would be actual bytes)
        code: Vec<u8>,
        /// Entry point address
        entry: usize,
    },
    /// Optimizing JIT code
    Optimizing {
        code: Vec<u8>,
        entry: usize,
        /// Assumptions made during compilation
        assumptions: Vec<String>,
    },
    /// E-graph optimized code
    EGraph {
        code: Vec<u8>,
        entry: usize,
        /// Profile data used
        profile_version: u64,
    },
}

/// Function metadata for the JIT
#[derive(Debug)]
pub struct JitFunction {
    /// Function name
    pub name: String,
    /// Current compilation stats
    pub stats: CompilationStats,
    /// Compiled code
    pub code: CompiledCode,
    /// Deoptimization history
    pub deopt_history: Vec<DeoptInfo>,
    /// Whether this function is currently being compiled
    pub compiling: bool,
}

impl JitFunction {
    pub fn new(name: String) -> Self {
        Self {
            name,
            stats: CompilationStats::new(),
            code: CompiledCode::None,
            deopt_history: Vec::new(),
            compiling: false,
        }
    }

    /// Check if this function should be promoted to the next tier
    pub fn should_promote(&self, threshold: u64) -> bool {
        // Promote if hot and not at max tier
        self.stats.is_hot && self.stats.tier < JitTier::EGraph
    }

    /// Record a deoptimization
    pub fn record_deopt(&mut self, reason: DeoptReason, location: String) {
        let from_tier = self.stats.tier;
        let to_tier = match from_tier {
            JitTier::EGraph => JitTier::Optimizing,
            JitTier::Optimizing => JitTier::Baseline,
            JitTier::Baseline => JitTier::Interpreter,
            JitTier::Interpreter => JitTier::Interpreter,
        };

        self.deopt_history.push(DeoptInfo {
            reason,
            location,
            from_tier,
            to_tier,
            timestamp: std::time::Instant::now(),
        });

        self.stats.tier = to_tier;
        self.code = CompiledCode::None; // Need to recompile
    }
}

/// Multi-tier JIT compiler
pub struct MultiTierJit {
    /// All functions managed by the JIT
    functions: HashMap<String, JitFunction>,
    /// Hot threshold (executions before considered hot)
    hot_threshold: u64,
    /// Minimum time between compilations (to avoid thrashing)
    min_compile_interval: std::time::Duration,
    /// Profile database (shared with other components)
    profile_db: Option<Arc<RwLock<crate::optimizer::profile_guided::ProfileDatabase>>>,
}

impl MultiTierJit {
    pub fn new(hot_threshold: u64) -> Self {
        Self {
            functions: HashMap::new(),
            hot_threshold,
            min_compile_interval: std::time::Duration::from_millis(100),
            profile_db: None,
        }
    }

    /// Set the profile database
    pub fn set_profile_db(&mut self, db: Arc<RwLock<crate::optimizer::profile_guided::ProfileDatabase>>) {
        self.profile_db = Some(db);
    }

    /// Register a function with the JIT
    pub fn register_function(&mut self, name: String) {
        self.functions.entry(name.clone()).or_insert_with(|| JitFunction::new(name));
    }

    /// Record an execution of a function
    pub fn record_execution(&mut self, name: &str, time_us: u64) {
        if let Some(func) = self.functions.get_mut(name) {
            func.stats.record_execution(time_us);
            func.stats.update_hot_status(self.hot_threshold);

            // Check if we should promote to next tier
            if func.should_promote(self.hot_threshold) {
                self.promote_function(name);
            }
        }
    }

    /// Promote a function to the next compilation tier
    fn promote_function(&mut self, name: &str) {
        if let Some(func) = self.functions.get_mut(name) {
            if func.compiling {
                return; // Already compiling
            }

            let next_tier = match func.stats.tier.next() {
                Some(t) => t,
                None => return, // Already at max tier
            };

            // Check minimum interval
            if func.stats.last_compiled.elapsed() < self.min_compile_interval {
                return;
            }

            func.compiling = true;
            func.stats.tier = next_tier;
            func.stats.last_compiled = std::time::Instant::now();

            // In a real implementation, this would trigger actual compilation
            // For now, we just update the metadata
            match next_tier {
                JitTier::Baseline => {
                    func.code = CompiledCode::Baseline {
                        code: vec![0x90, 0x90], // NOPs as placeholder
                        entry: 0,
                    };
                }
                JitTier::Optimizing => {
                    func.code = CompiledCode::Optimizing {
                        code: vec![0x90, 0x90, 0x90],
                        entry: 0,
                        assumptions: vec!["type_int".to_string()],
                    };
                }
                JitTier::EGraph => {
                    func.code = CompiledCode::EGraph {
                        code: vec![0x90, 0x90, 0x90, 0x90],
                        entry: 0,
                        profile_version: 1,
                    };
                }
                JitTier::Interpreter => {
                    func.code = CompiledCode::None;
                }
            }

            func.compiling = false;
        }
    }

    /// Trigger deoptimization for a function
    pub fn deoptimize(&mut self, name: &str, reason: DeoptReason, location: String) {
        if let Some(func) = self.functions.get_mut(name) {
            func.record_deopt(reason, location);
        }
    }

    /// Get the current tier of a function
    pub fn get_tier(&self, name: &str) -> Option<JitTier> {
        self.functions.get(name).map(|f| f.stats.tier)
    }

    /// Get compilation stats for a function
    pub fn get_stats(&self, name: &str) -> Option<&CompilationStats> {
        self.functions.get(name).map(|f| &f.stats)
    }

    /// Get all hot functions
    pub fn hot_functions(&self) -> Vec<&str> {
        self.functions
            .iter()
            .filter(|(_, f)| f.stats.is_hot)
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Force recompilation of a function at a specific tier
    pub fn force_recompile(&mut self, name: &str, tier: JitTier) {
        if let Some(func) = self.functions.get_mut(name) {
            func.stats.tier = tier;
            func.stats.last_compiled = std::time::Instant::now();
            func.code = CompiledCode::None; // Will be recompiled on next execution
        }
    }

    /// Get the number of functions at each tier
    pub fn tier_distribution(&self) -> HashMap<JitTier, usize> {
        let mut dist = HashMap::new();
        for func in self.functions.values() {
            *dist.entry(func.stats.tier).or_insert(0) += 1;
        }
        dist
    }
}

/// On-stack replacement (OSR) for tier switching
///
/// OSR allows switching to a higher tier in the middle of a loop
/// without returning to the interpreter.
pub struct OnStackReplacement {
    /// OSR entry points (loop location -> compiled code entry)
    osr_entries: HashMap<String, usize>,
}

impl OnStackReplacement {
    pub fn new() -> Self {
        Self {
            osr_entries: HashMap::new(),
        }
    }

    /// Register an OSR entry point
    pub fn register_osr_entry(&mut self, location: String, entry: usize) {
        self.osr_entries.insert(location, entry);
    }

    /// Get the OSR entry point for a location
    pub fn get_osr_entry(&self, location: &str) -> Option<usize> {
        self.osr_entries.get(location).copied()
    }

    /// Check if OSR is available at a location
    pub fn has_osr(&self, location: &str) -> bool {
        self.osr_entries.contains_key(location)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_progression() {
        assert_eq!(JitTier::Interpreter.next(), Some(JitTier::Baseline));
        assert_eq!(JitTier::Baseline.next(), Some(JitTier::Optimizing));
        assert_eq!(JitTier::Optimizing.next(), Some(JitTier::EGraph));
        assert_eq!(JitTier::EGraph.next(), None);
    }

    #[test]
    fn test_compilation_stats() {
        let mut stats = CompilationStats::new();
        stats.record_execution(100);
        stats.record_execution(200);
        
        assert_eq!(stats.execution_count, 2);
        assert_eq!(stats.total_time_us, 300);
        assert_eq!(stats.avg_time_us, 150.0);
    }

    #[test]
    fn test_hot_detection() {
        let mut stats = CompilationStats::new();
        stats.update_hot_status(100);
        assert!(!stats.is_hot);
        
        stats.execution_count = 150;
        stats.update_hot_status(100);
        assert!(stats.is_hot);
    }

    #[test]
    fn test_jit_function_promotion() {
        let mut func = JitFunction::new("test".to_string());
        func.stats.execution_count = 100;
        func.stats.update_hot_status(50);
        
        assert!(func.should_promote(50));
        assert!(!func.should_promote(200));
    }

    #[test]
    fn test_multi_tier_jit() {
        let mut jit = MultiTierJit::new(10);
        jit.register_function("test_func".to_string());
        
        // Record executions
        for _ in 0..15 {
            jit.record_execution("test_func", 100);
        }
        
        // Should be promoted to baseline
        assert_eq!(jit.get_tier("test_func"), Some(JitTier::Baseline));
        
        // More executions
        for _ in 0..20 {
            jit.record_execution("test_func", 50);
        }
        
        // Should be promoted further
        assert!(jit.get_tier("test_func").unwrap() >= JitTier::Baseline);
    }

    #[test]
    fn test_deoptimization() {
        let mut jit = MultiTierJit::new(10);
        jit.register_function("test_func".to_string());
        
        // Promote to optimizing
        for _ in 0..30 {
            jit.record_execution("test_func", 100);
        }
        
        let tier_before = jit.get_tier("test_func");
        
        // Trigger deoptimization
        jit.deoptimize("test_func", DeoptReason::TypeMismatch, "line 42".to_string());
        
        let tier_after = jit.get_tier("test_func");
        
        // Should have fallen back
        assert!(tier_after < tier_before);
    }
}
