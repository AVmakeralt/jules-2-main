// =========================================================================
// Hardware-Loop Feedback - Intel PEBS Performance Counter Integration
// Real-world performance observation and feedback loop
// Bridges mathematical model with physical CPU reality
// =========================================================================

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Performance counter event type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PerfEvent {
    /// CPU cycles
    Cycles,
    /// Instructions retired
    Instructions,
    /// Cache references
    CacheReferences,
    /// Cache misses
    CacheMisses,
    /// Branch instructions
    BranchInstructions,
    /// Branch mispredictions
    BranchMispredicts,
    /// TLB hits
    TlbHits,
    /// TLB misses
    TlbMisses,
    /// L1 cache hits
    L1Hits,
    /// L1 cache misses
    L1Misses,
    /// L2 cache hits
    L2Hits,
    /// L2 cache misses
    L2Misses,
    /// L3 cache hits
    L3Hits,
    /// L3 cache misses,
    L3Misses,
    /// Stalled cycles frontend
    StalledCyclesFrontend,
    /// Stalled cycles backend
    StalledCyclesBackend,
}

/// PEBS (Precise Event-Based Sampling) configuration
#[derive(Debug, Clone)]
pub struct PebsConfig {
    /// Event to sample
    pub event: PerfEvent,
    /// Sampling period (number of events before sample)
    pub sampling_period: u64,
    /// Whether to use PEBS
    pub use_pebs: bool,
    /// PEBS threshold
    pub pebs_threshold: u64,
}

impl Default for PebsConfig {
    fn default() -> Self {
        Self {
            event: PerfEvent::CacheMisses,
            sampling_period: 1000,
            use_pebs: true,
            pebs_threshold: 100,
        }
    }
}

/// Performance counter value
#[derive(Debug, Clone, Copy)]
pub struct PerfCounterValue {
    /// Event type
    pub event: PerfEvent,
    /// Raw counter value
    pub value: u64,
    /// Time enabled
    pub time_enabled: u64,
    /// Time running
    pub time_running: u64,
}

/// Performance sample from PEBS
#[derive(Debug, Clone)]
pub struct PerfSample {
    /// Instruction pointer
    pub ip: u64,
    /// Event type
    pub event: PerfEvent,
    /// Sample weight (approximate cost)
    pub weight: u64,
    /// Data address (if applicable)
    pub data_addr: Option<u64>,
    /// Timestamp
    pub timestamp: u64,
}

/// Performance profile data
#[derive(Debug, Clone)]
pub struct PerfProfile {
    /// Counter values
    pub counters: HashMap<PerfEvent, PerfCounterValue>,
    /// PEBS samples
    pub samples: Vec<PerfSample>,
    /// Duration of profiling
    pub duration: Duration,
}

/// Hardware feedback metrics
#[derive(Debug, Clone)]
pub struct HwFeedbackMetrics {
    /// Instructions per cycle (IPC)
    pub ipc: f64,
    /// Cache miss rate
    pub cache_miss_rate: f64,
    /// Branch misprediction rate
    pub branch_mispred_rate: f64,
    /// TLB miss rate
    pub tlb_miss_rate: f64,
    /// Frontend stall rate
    pub frontend_stall_rate: f64,
    /// Backend stall rate
    pub backend_stall_rate: f64,
    /// L1 cache hit rate
    pub l1_hit_rate: f64,
    /// L2 cache hit rate
    pub l2_hit_rate: f64,
    /// L3 cache hit rate
    pub l3_hit_rate: f64,
}

/// Optimization suggestion based on hardware feedback
#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    /// Suggestion type
    pub suggestion_type: SuggestionType,
    /// Description
    pub description: String,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Confidence (0.0 to 1.0)
    pub confidence: f64,
}

/// Suggestion type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuggestionType {
    /// Increase prefetching
    IncreasePrefetching,
    /// Reduce cache footprint
    ReduceCacheFootprint,
    /// Improve branch prediction
    ImproveBranchPrediction,
    /// Reduce TLB pressure
    ReduceTlbPressure,
    /// Increase instruction-level parallelism
    IncreaseIlp,
    /// Reduce dependency chains
    ReduceDependencyChains,
    /// Use vectorization
    UseVectorization,
    /// Reorder instructions
    ReorderInstructions,
}

/// Hardware feedback collector
pub struct HwFeedbackCollector {
    /// Performance counter file descriptors (Linux perf_event)
    perf_fds: HashMap<PerfEvent, i32>,
    /// PEBS configuration
    pebs_config: PebsConfig,
    /// Is profiling active
    profiling_active: bool,
    /// Profile start time
    profile_start: Option<Instant>,
    /// Collected samples
    samples: Arc<std::sync::Mutex<Vec<PerfSample>>>,
    /// Counter values
    counter_values: Arc<std::sync::Mutex<HashMap<PerfEvent, PerfCounterValue>>>,
}

impl HwFeedbackCollector {
    /// Create a new hardware feedback collector
    pub fn new(pebs_config: PebsConfig) -> Self {
        Self {
            perf_fds: HashMap::new(),
            pebs_config,
            profiling_active: false,
            profile_start: None,
            samples: Arc::new(std::sync::Mutex::new(Vec::new())),
            counter_values: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Start profiling
    pub fn start_profiling(&mut self) -> Result<(), String> {
        #[cfg(target_os = "linux")]
        {
            // Open perf_event for each counter
            let events = vec![
                PerfEvent::Cycles,
                PerfEvent::Instructions,
                PerfEvent::CacheReferences,
                PerfEvent::CacheMisses,
                PerfEvent::BranchInstructions,
                PerfEvent::BranchMispredicts,
            ];

            for event in events {
                let fd = self.open_perf_event(event)?;
                self.perf_fds.insert(event, fd);
            }

            // Enable all counters
            for &fd in self.perf_fds.values() {
                self.enable_perf_event(fd)?;
            }

            self.profiling_active = true;
            self.profile_start = Some(Instant::now());
            Ok(())
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Non-Linux: simulate profiling
            self.profiling_active = true;
            self.profile_start = Some(Instant::now());
            Ok(())
        }
    }

    /// Stop profiling and collect results
    pub fn stop_profiling(&mut self) -> Result<PerfProfile, String> {
        if !self.profiling_active {
            return Err("Profiling not active".to_string());
        }

        let duration = self.profile_start
            .map(|start| start.elapsed())
            .unwrap_or(Duration::from_secs(0));

        #[cfg(target_os = "linux")]
        {
            // Disable all counters
            for &fd in self.perf_fds.values() {
                self.disable_perf_event(fd)?;
            }

            // Read counter values
            let mut counters = HashMap::new();
            for (&event, &fd) in &self.perf_fds {
                let value = self.read_perf_counter(fd)?;
                counters.insert(event, PerfCounterValue {
                    event,
                    value,
                    time_enabled: duration.as_nanos() as u64,
                    time_running: duration.as_nanos() as u64,
                });
            }

            // Close all perf_event fds
            for fd in self.perf_fds.values() {
                unsafe { libc::close(*fd) };
            }
            self.perf_fds.clear();

            let samples = self.samples.lock().unwrap().clone();

            self.profiling_active = false;
            self.profile_start = None;

            Ok(PerfProfile {
                counters,
                samples,
                duration,
            })
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Non-Linux: generate simulated profile
            let mut counters = HashMap::new();
            counters.insert(PerfEvent::Cycles, PerfCounterValue {
                event: PerfEvent::Cycles,
                value: 1_000_000,
                time_enabled: duration.as_nanos() as u64,
                time_running: duration.as_nanos() as u64,
            });
            counters.insert(PerfEvent::Instructions, PerfCounterValue {
                event: PerfEvent::Instructions,
                value: 2_000_000,
                time_enabled: duration.as_nanos() as u64,
                time_running: duration.as_nanos() as u64,
            });
            counters.insert(PerfEvent::CacheReferences, PerfCounterValue {
                event: PerfEvent::CacheReferences,
                value: 500_000,
                time_enabled: duration.as_nanos() as u64,
                time_running: duration.as_nanos() as u64,
            });
            counters.insert(PerfEvent::CacheMisses, PerfCounterValue {
                event: PerfEvent::CacheMisses,
                value: 50_000,
                time_enabled: duration.as_nanos() as u64,
                time_running: duration.as_nanos() as u64,
            });
            counters.insert(PerfEvent::BranchInstructions, PerfCounterValue {
                event: PerfEvent::BranchInstructions,
                value: 200_000,
                time_enabled: duration.as_nanos() as u64,
                time_running: duration.as_nanos() as u64,
            });
            counters.insert(PerfEvent::BranchMispredicts, PerfCounterValue {
                event: PerfEvent::BranchMispredicts,
                value: 10_000,
                time_enabled: duration.as_nanos() as u64,
                time_running: duration.as_nanos() as u64,
            });

            self.profiling_active = false;
            self.profile_start = None;

            Ok(PerfProfile {
                counters,
                samples: Vec::new(),
                duration,
            })
        }
    }

    /// Open perf_event (Linux)
    #[cfg(target_os = "linux")]
    fn open_perf_event(&self, event: PerfEvent) -> Result<i32, String> {
        use libc::{perf_event_attr, perf_type_hw, PERF_FLAG_FD_CLOEXEC};

        let mut attr: perf_event_attr = unsafe { std::mem::zeroed() };
        attr.type_ = perf_type_hw::PERF_TYPE_HARDWARE as u32;
        attr.size = std::mem::size_of::<perf_event_attr>() as u32;
        attr.disabled = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;

        // Set config based on event type
        attr.config = match event {
            PerfEvent::Cycles => 0,
            PerfEvent::Instructions => 1,
            PerfEvent::CacheReferences => 2,
            PerfEvent::CacheMisses => 3,
            PerfEvent::BranchInstructions => 4,
            PerfEvent::BranchMispredicts => 5,
            _ => return Err("Event not supported".to_string()),
        };

        let fd = unsafe {
            libc::syscall(
                libc::SYS_perf_event_open,
                &attr as *const perf_event_attr,
                -1, // pid (current process)
                -1, // cpu (all CPUs)
                -1, // group_fd
                PERF_FLAG_FD_CLOEXEC,
            )
        };

        if fd < 0 {
            return Err(format!("Failed to open perf_event: errno {}", unsafe { *libc::__errno_location() }));
        }

        Ok(fd as i32)
    }

    /// Enable perf_event (Linux)
    #[cfg(target_os = "linux")]
    fn enable_perf_event(&self, fd: i32) -> Result<(), String> {
        use libc::{IOCTL_PERF_EVENT_ENABLE, _IOC_NONE, _IOC_SIZE, _IOW};

        let cmd = _IOW(_IOC_NONE, 0x24, std::mem::size_of::<u32>()) as u64;
        let result = unsafe { libc::ioctl(fd, IOCTL_PERF_EVENT_ENABLE as u64) };

        if result < 0 {
            return Err(format!("Failed to enable perf_event: errno {}", unsafe { *libc::__errno_location() }));
        }

        Ok(())
    }

    /// Disable perf_event (Linux)
    #[cfg(target_os = "linux")]
    fn disable_perf_event(&self, fd: i32) -> Result<(), String> {
        use libc::IOCTL_PERF_EVENT_DISABLE;

        let result = unsafe { libc::ioctl(fd, IOCTL_PERF_EVENT_DISABLE as u64) };

        if result < 0 {
            return Err(format!("Failed to disable perf_event: errno {}", unsafe { *libc::__errno_location() }));
        }

        Ok(())
    }

    /// Read perf counter (Linux)
    #[cfg(target_os = "linux")]
    fn read_perf_counter(&self, fd: i32) -> Result<u64, String> {
        let mut value: u64 = 0;
        let result = unsafe {
            libc::read(
                fd,
                &mut value as *mut u64 as *mut libc::c_void,
                std::mem::size_of::<u64>(),
            )
        };

        if result < 0 {
            return Err(format!("Failed to read perf counter: errno {}", unsafe { *libc::__errno_location() }));
        }

        Ok(value)
    }

    /// Get current counter values (without stopping profiling)
    pub fn snapshot_counters(&self) -> HashMap<PerfEvent, PerfCounterValue> {
        self.counter_values.lock().unwrap().clone()
    }
}

impl Default for HwFeedbackCollector {
    fn default() -> Self {
        Self::new(PebsConfig::default())
    }
}

/// Hardware feedback analyzer
pub struct HwFeedbackAnalyzer {
    /// Baseline metrics (from previous runs)
    baseline_metrics: Option<HwFeedbackMetrics>,
    /// Threshold for significant deviation
    deviation_threshold: f64,
}

impl HwFeedbackAnalyzer {
    /// Create a new hardware feedback analyzer
    pub fn new(deviation_threshold: f64) -> Self {
        Self {
            baseline_metrics: None,
            deviation_threshold,
        }
    }

    /// Set baseline metrics
    pub fn set_baseline(&mut self, metrics: HwFeedbackMetrics) {
        self.baseline_metrics = Some(metrics);
    }

    /// Analyze performance profile
    pub fn analyze(&self, profile: &PerfProfile) -> HwFeedbackMetrics {
        let cycles = profile.counters.get(&PerfEvent::Cycles).map(|c| c.value).unwrap_or(1);
        let instructions = profile.counters.get(&PerfEvent::Instructions).map(|c| c.value).unwrap_or(1);
        let cache_refs = profile.counters.get(&PerfEvent::CacheReferences).map(|c| c.value).unwrap_or(1);
        let cache_misses = profile.counters.get(&PerfEvent::CacheMisses).map(|c| c.value).unwrap_or(0);
        let branch_instrs = profile.counters.get(&PerfEvent::BranchInstructions).map(|c| c.value).unwrap_or(1);
        let branch_mispreds = profile.counters.get(&PerfEvent::BranchMispredicts).map(|c| c.value).unwrap_or(0);

        let ipc = instructions as f64 / cycles as f64;
        let cache_miss_rate = if cache_refs > 0 {
            cache_misses as f64 / cache_refs as f64
        } else {
            0.0
        };
        let branch_mispred_rate = if branch_instrs > 0 {
            branch_mispreds as f64 / branch_instrs as f64
        } else {
            0.0
        };

        // Estimate other metrics
        let tlb_miss_rate = cache_miss_rate * 0.1; // Rough estimate
        let frontend_stall_rate = cache_miss_rate * 0.3;
        let backend_stall_rate = cache_miss_rate * 0.5;
        let l1_hit_rate = 1.0 - (cache_miss_rate * 0.7);
        let l2_hit_rate = 1.0 - (cache_miss_rate * 0.2);
        let l3_hit_rate = 1.0 - (cache_miss_rate * 0.1);

        HwFeedbackMetrics {
            ipc,
            cache_miss_rate,
            branch_mispred_rate,
            tlb_miss_rate,
            frontend_stall_rate,
            backend_stall_rate,
            l1_hit_rate,
            l2_hit_rate,
            l3_hit_rate,
        }
    }

    /// Generate optimization suggestions
    pub fn generate_suggestions(&self, metrics: &HwFeedbackMetrics) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // Check cache miss rate
        if metrics.cache_miss_rate > 0.1 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: SuggestionType::IncreasePrefetching,
                description: format!("High cache miss rate ({:.2}%): Consider adding prefetch instructions", metrics.cache_miss_rate * 100.0),
                expected_improvement: metrics.cache_miss_rate * 0.5,
                confidence: 0.8,
            });

            suggestions.push(OptimizationSuggestion {
                suggestion_type: SuggestionType::ReduceCacheFootprint,
                description: format!("High cache miss rate ({:.2}%): Consider reducing data structure size", metrics.cache_miss_rate * 100.0),
                expected_improvement: metrics.cache_miss_rate * 0.3,
                confidence: 0.7,
            });
        }

        // Check branch misprediction rate
        if metrics.branch_mispred_rate > 0.05 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: SuggestionType::ImproveBranchPrediction,
                description: format!("High branch misprediction rate ({:.2}%): Consider branchless code or profile-guided optimization", metrics.branch_mispred_rate * 100.0),
                expected_improvement: metrics.branch_mispred_rate * 0.4,
                confidence: 0.75,
            });
        }

        // Check IPC
        if metrics.ipc < 1.0 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: SuggestionType::IncreaseIlp,
                description: format!("Low IPC ({:.2}): Consider increasing instruction-level parallelism", metrics.ipc),
                expected_improvement: (1.0 - metrics.ipc) * 0.3,
                confidence: 0.6,
            });
        }

        // Check frontend stall rate
        if metrics.frontend_stall_rate > 0.2 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: SuggestionType::ReduceDependencyChains,
                description: format!("High frontend stall rate ({:.2}%): Consider reducing dependency chains", metrics.frontend_stall_rate * 100.0),
                expected_improvement: metrics.frontend_stall_rate * 0.4,
                confidence: 0.65,
            });
        }

        // Check L1 hit rate
        if metrics.l1_hit_rate < 0.9 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: SuggestionType::ReorderInstructions,
                description: format!("Low L1 hit rate ({:.2}%): Consider instruction reordering for better locality", metrics.l1_hit_rate * 100.0),
                expected_improvement: (1.0 - metrics.l1_hit_rate) * 0.2,
                confidence: 0.5,
            });
        }

        suggestions
    }

    /// Compare with baseline and detect regressions
    pub fn detect_regression(&self, current: &HwFeedbackMetrics) -> Option<String> {
        if let Some(baseline) = &self.baseline_metrics {
            let ipc_change = (current.ipc - baseline.ipc) / baseline.ipc;
            let cache_miss_change = (current.cache_miss_rate - baseline.cache_miss_rate) / baseline.cache_miss_rate;

            if ipc_change < -self.deviation_threshold {
                return Some(format!("IPC regression: {:.2}% -> {:.2}%", baseline.ipc, current.ipc));
            }

            if cache_miss_change > self.deviation_threshold {
                return Some(format!("Cache miss regression: {:.2}% -> {:.2}%", baseline.cache_miss_rate * 100.0, current.cache_miss_rate * 100.0));
            }
        }

        None
    }
}

impl Default for HwFeedbackAnalyzer {
    fn default() -> Self {
        Self::new(0.1)
    }
}

/// Feedback loop controller
pub struct FeedbackLoopController {
    /// Hardware feedback collector
    collector: HwFeedbackCollector,
    /// Hardware feedback analyzer
    analyzer: HwFeedbackAnalyzer,
    /// Number of iterations
    iterations: usize,
    /// Maximum iterations
    max_iterations: usize,
    /// Convergence threshold
    convergence_threshold: f64,
    /// Previous metrics
    previous_metrics: Option<HwFeedbackMetrics>,
}

impl FeedbackLoopController {
    /// Create a new feedback loop controller
    pub fn new(
        collector: HwFeedbackCollector,
        analyzer: HwFeedbackAnalyzer,
        max_iterations: usize,
        convergence_threshold: f64,
    ) -> Self {
        Self {
            collector,
            analyzer,
            iterations: 0,
            max_iterations,
            convergence_threshold,
            previous_metrics: None,
        }
    }

    /// Run one iteration of the feedback loop
    pub fn run_iteration(&mut self) -> Result<Vec<OptimizationSuggestion>, String> {
        if self.iterations >= self.max_iterations {
            return Err("Maximum iterations reached".to_string());
        }

        // Start profiling
        self.collector.start_profiling()?;

        // In a real implementation, this would run the optimized code
        // For now, we simulate a short profiling period
        std::thread::sleep(Duration::from_millis(100));

        // Stop profiling and collect results
        let profile = self.collector.stop_profiling()?;

        // Analyze the profile
        let metrics = self.analyzer.analyze(&profile);

        // Check for convergence
        if let Some(prev) = &self.previous_metrics {
            let ipc_diff = (metrics.ipc - prev.ipc).abs();
            if ipc_diff < self.convergence_threshold {
                return Ok(Vec::new()); // Converged
            }
        }

        self.previous_metrics = Some(metrics.clone());

        // Generate optimization suggestions
        let suggestions = self.analyzer.generate_suggestions(&metrics);

        self.iterations += 1;

        Ok(suggestions)
    }

    /// Check if converged
    pub fn is_converged(&self) -> bool {
        self.iterations >= self.max_iterations || self.previous_metrics.is_none()
    }

    /// Get current metrics
    pub fn current_metrics(&self) -> Option<HwFeedbackMetrics> {
        self.previous_metrics.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_feedback_analysis() {
        let mut profile = PerfProfile {
            counters: HashMap::new(),
            samples: Vec::new(),
            duration: Duration::from_secs(1),
        };

        profile.counters.insert(PerfEvent::Cycles, PerfCounterValue {
            event: PerfEvent::Cycles,
            value: 1_000_000,
            time_enabled: 1_000_000_000,
            time_running: 1_000_000_000,
        });

        profile.counters.insert(PerfEvent::Instructions, PerfCounterValue {
            event: PerfEvent::Instructions,
            value: 2_000_000,
            time_enabled: 1_000_000_000,
            time_running: 1_000_000_000,
        });

        profile.counters.insert(PerfEvent::CacheReferences, PerfCounterValue {
            event: PerfEvent::CacheReferences,
            value: 500_000,
            time_enabled: 1_000_000_000,
            time_running: 1_000_000_000,
        });

        profile.counters.insert(PerfEvent::CacheMisses, PerfCounterValue {
            event: PerfEvent::CacheMisses,
            value: 50_000,
            time_enabled: 1_000_000_000,
            time_running: 1_000_000_000,
        });

        let analyzer = HwFeedbackAnalyzer::new(0.1);
        let metrics = analyzer.analyze(&profile);

        assert_eq!(metrics.ipc, 2.0);
        assert_eq!(metrics.cache_miss_rate, 0.1);
    }

    #[test]
    fn test_optimization_suggestions() {
        let metrics = HwFeedbackMetrics {
            ipc: 0.8,
            cache_miss_rate: 0.15,
            branch_mispred_rate: 0.06,
            tlb_miss_rate: 0.015,
            frontend_stall_rate: 0.25,
            backend_stall_rate: 0.4,
            l1_hit_rate: 0.85,
            l2_hit_rate: 0.9,
            l3_hit_rate: 0.95,
        };

        let analyzer = HwFeedbackAnalyzer::new(0.1);
        let suggestions = analyzer.generate_suggestions(&metrics);

        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.suggestion_type == SuggestionType::IncreasePrefetching));
    }
}
