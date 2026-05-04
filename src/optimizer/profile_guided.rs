// =============================================================================
// Profile-Guided Equality Saturation
//
// This module implements runtime profiling that feeds real execution data
// back into the e-graph, enabling it to select the actually fastest equivalent
// program instead of the theoretically cheapest one.
//
// Features:
// - Hot path detection and profiling
// - Real cycle count collection
// - Profile-weighted cost model
// - Re-saturation trigger when behavior changes
// =============================================================================

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Profile data for a specific code location
#[derive(Debug, Clone)]
pub struct ProfileData {
    /// Number of times this code was executed
    pub execution_count: u64,
    /// Total cycles spent (if available)
    pub total_cycles: u64,
    /// Average cycles per execution
    pub avg_cycles: f64,
    /// Last time this was profiled
    pub last_profiled: Instant,
    /// Whether this is a hot path (frequently executed)
    pub is_hot: bool,
}

impl ProfileData {
    pub fn new() -> Self {
        Self {
            execution_count: 0,
            total_cycles: 0,
            avg_cycles: 0.0,
            last_profiled: Instant::now(),
            is_hot: false,
        }
    }

    pub fn record_execution(&mut self, cycles: u64) {
        self.execution_count += 1;
        self.total_cycles += cycles;
        self.avg_cycles = self.total_cycles as f64 / self.execution_count as f64;
        self.last_profiled = Instant::now();
    }

    /// Mark as hot if execution count exceeds threshold
    pub fn update_hot_status(&mut self, threshold: u64) {
        self.is_hot = self.execution_count >= threshold;
    }
}

/// Profile database for the entire program
#[derive(Debug)]
pub struct ProfileDatabase {
    /// Map from code location (e.g., function name or instruction address) to profile data
    profiles: HashMap<String, ProfileData>,
    /// Threshold for considering a path "hot"
    hot_threshold: u64,
    /// Whether profiling is enabled
    enabled: bool,
}

impl ProfileDatabase {
    pub fn new(hot_threshold: u64) -> Self {
        Self {
            profiles: HashMap::new(),
            hot_threshold,
            enabled: true,
        }
    }

    /// Record an execution at a specific location
    pub fn record_execution(&mut self, location: &str, cycles: u64) {
        if !self.enabled {
            return;
        }

        let profile = self.profiles
            .entry(location.to_string())
            .or_insert_with(ProfileData::new);
        profile.record_execution(cycles);
        profile.update_hot_status(self.hot_threshold);
    }

    /// Get profile data for a location
    pub fn get_profile(&self, location: &str) -> Option<&ProfileData> {
        self.profiles.get(location)
    }

    /// Get all hot paths
    pub fn hot_paths(&self) -> Vec<(String, &ProfileData)> {
        self.profiles
            .iter()
            .filter(|(_, p)| p.is_hot)
            .map(|(loc, p)| (loc.clone(), p))
            .collect()
    }

    /// Check if a location is hot
    pub fn is_hot(&self, location: &str) -> bool {
        self.profiles
            .get(location)
            .map(|p| p.is_hot)
            .unwrap_or(false)
    }

    /// Enable/disable profiling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Clear all profile data
    pub fn clear(&mut self) {
        self.profiles.clear();
    }
}

/// Thread-safe profile database
pub type SharedProfileDatabase = Arc<RwLock<ProfileDatabase>>;

/// Profile-weighted cost model for e-graph
///
/// This cost model combines static hardware costs with dynamic profile data
/// to select the actually fastest instruction sequence.
pub struct ProfileWeightedCostModel {
    /// Profile database
    profiles: SharedProfileDatabase,
    /// Base hardware cost model (from hardware_cost_model.rs)
    /// In a real implementation, this would reference the HardwareCostModel
    base_cost_multiplier: f64,
    /// How much to weight profile data vs static costs (0.0 = static only, 1.0 = profile only)
    profile_weight: f64,
}

impl ProfileWeightedCostModel {
    pub fn new(profiles: SharedProfileDatabase) -> Self {
        Self {
            profiles,
            base_cost_multiplier: 1.0,
            profile_weight: 0.5, // Balance between static and profile data
        }
    }

    /// Set the profile weight (0.0 to 1.0)
    pub fn set_profile_weight(&mut self, weight: f64) {
        self.profile_weight = weight.clamp(0.0, 1.0);
    }

    /// Estimate the cost of an instruction sequence with profile data
    ///
    /// This combines:
    /// 1. Static hardware cost (latency, port contention)
    /// 2. Dynamic profile data (actual cycle counts from runtime)
    pub fn estimate_cost(&self, location: &str, static_cost: f64) -> f64 {
        let db = self.profiles.read().unwrap();
        
        if let Some(profile) = db.get_profile(location) {
            // Blend static cost with profile data
            let profile_cost = profile.avg_cycles;
            let blended = (1.0 - self.profile_weight) * static_cost + self.profile_weight * profile_cost;
            
            // If this is a hot path, trust the profile data more
            if profile.is_hot {
                blended * 0.8 // Discount cost for hot paths (they're worth optimizing)
            } else {
                blended
            }
        } else {
            // No profile data available, use static cost
            static_cost * self.base_cost_multiplier
        }
    }

    /// Check if re-saturation is needed for a location
    ///
    /// Re-saturation is triggered when:
    /// 1. A path becomes hot (crosses threshold)
    /// 2. Profile data changes significantly (e.g., avg cycles changes by >20%)
    pub fn needs_resaturation(&self, location: &str, previous_avg: Option<f64>) -> bool {
        let db = self.profiles.read().unwrap();
        
        if let Some(profile) = db.get_profile(location) {
            // Re-saturate if just became hot
            if profile.is_hot && previous_avg.is_none() {
                return true;
            }
            
            // Re-saturate if profile changed significantly
            if let Some(prev) = previous_avg {
                let change = (profile.avg_cycles - prev).abs() / prev;
                if change > 0.2 { // 20% change threshold
                    return true;
                }
            }
        }
        
        false
    }

    /// Get the profile database
    pub fn profiles(&self) -> &SharedProfileDatabase {
        &self.profiles
    }
}

/// Re-saturation trigger
///
/// Determines when the e-graph should be re-saturated with new profile data.
pub struct ResaturationTrigger {
    /// Minimum time between re-saturations
    min_interval: Duration,
    /// Last re-saturation time
    last_resaturation: Instant,
    /// Minimum number of new executions before re-saturation
    min_new_executions: u64,
}

impl ResaturationTrigger {
    pub fn new(min_interval: Duration, min_new_executions: u64) -> Self {
        Self {
            min_interval,
            last_resaturation: Instant::now() - min_interval, // Allow immediate first re-saturation
            min_new_executions,
        }
    }

    /// Check if re-saturation should be triggered
    pub fn should_trigger(&mut self, profile_db: &ProfileDatabase) -> bool {
        // Check time interval
        if self.last_resaturation.elapsed() < self.min_interval {
            return false;
        }

        // Check if there are enough new executions on hot paths
        let total_hot_executions: u64 = profile_db
            .hot_paths()
            .iter()
            .map(|(_, p)| p.execution_count)
            .sum();

        if total_hot_executions >= self.min_new_executions {
            self.last_resaturation = Instant::now();
            true
        } else {
            false
        }
    }
}

/// Cycle counter for profiling
///
/// In a real implementation, this would use hardware performance counters
/// (e.g., RDTSC on x86, PMU on ARM) for accurate cycle counting.
pub struct CycleCounter {
    /// Whether cycle counting is available
    available: bool,
}

impl CycleCounter {
    pub fn new() -> Self {
        // In a real implementation, check if performance counters are available
        Self {
            available: true, // Assume available for now
        }
    }

    /// Read the current cycle count with lfence serialization
    pub fn read_cycles(&self) -> u64 {
        if !self.available {
            return 0;
        }

        #[cfg(target_arch = "x86_64")]
        unsafe {
            let mut high: u32;
            let mut low: u32;
            std::arch::asm!(
                "lfence",  // Serialize: prevent out-of-order execution before RDTSC
                "rdtsc",
                out("eax") low,
                out("edx") high,
                options(nomem, nostack)
            );
            ((high as u64) << 32) | (low as u64)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        }
    }

    /// Serializing cycle read (more accurate but slower)
    pub fn read_cycles_serializing(&self) -> u64 {
        if !self.available {
            return 0;
        }

        #[cfg(target_arch = "x86_64")]
        unsafe {
            let mut high: u32;
            let mut low: u32;
            std::arch::asm!(
                "push rbx",     // Save RBX (reserved by LLVM for GOT in PIC)
                "cpuid",        // Full serialization barrier (writes EAX, EBX, ECX, EDX)
                "pop rbx",      // Restore RBX
                "rdtsc",
                out("eax") low,
                out("edx") high,
                out("ecx") _,
                options(nostack)
            );
            ((high as u64) << 32) | (low as u64)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.read_cycles()
        }
    }

    /// Estimate CPU frequency in GHz by measuring TSC over a known sleep duration
    pub fn estimate_frequency_ghz(&self) -> f64 {
        let start = self.read_cycles();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let end = self.read_cycles();
        let cycles = end.wrapping_sub(start) as f64;
        cycles / 10_000_000.0  // 10ms = 10,000,000 ns
    }

    /// Convert cycles to nanoseconds using estimated CPU frequency
    pub fn cycles_to_ns(&self, cycles: u64) -> f64 {
        let freq = self.estimate_frequency_ghz();
        if freq > 0.0 {
            cycles as f64 / freq
        } else {
            cycles as f64  // Fallback: assume 1 GHz
        }
    }

    /// Measure cycles for a closure
    pub fn measure<F, R>(&self, f: F) -> (R, u64)
    where
        F: FnOnce() -> R,
    {
        let start = self.read_cycles();
        let result = f();
        let end = self.read_cycles();
        (result, end.saturating_sub(start))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_data() {
        let mut profile = ProfileData::new();
        profile.record_execution(100);
        profile.record_execution(200);
        
        assert_eq!(profile.execution_count, 2);
        assert_eq!(profile.total_cycles, 300);
        assert_eq!(profile.avg_cycles, 150.0);
    }

    #[test]
    fn test_hot_detection() {
        let mut profile = ProfileData::new();
        profile.update_hot_status(100);
        assert!(!profile.is_hot);
        
        profile.execution_count = 150;
        profile.update_hot_status(100);
        assert!(profile.is_hot);
    }

    #[test]
    fn test_profile_database() {
        let mut db = ProfileDatabase::new(100);
        db.record_execution("func1", 50);
        db.record_execution("func1", 50);
        
        assert_eq!(db.get_profile("func1").unwrap().execution_count, 2);
        assert!(!db.is_hot("func1"));
        
        for _ in 0..101 {
            db.record_execution("func1", 10);
        }
        assert!(db.is_hot("func1"));
    }

    #[test]
    fn test_resaturation_trigger() {
        let mut db = ProfileDatabase::new(10);
        let mut trigger = ResaturationTrigger::new(Duration::from_millis(100), 50);
        
        // Should not trigger immediately (not enough executions)
        assert!(!trigger.should_trigger(&db));
        
        // Add executions
        for _ in 0..60 {
            db.record_execution("hot_func", 10);
        }
        
        // Should trigger now
        assert!(trigger.should_trigger(&db));
    }
}
