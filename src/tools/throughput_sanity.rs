// =============================================================================
// Throughput Sanity Checker
//
// Flags benchmark numbers that exceed PHYSICAL LIMITS of x86-64 hardware.
// Any benchmark claiming throughput above these ceilings is measuring
// compiler optimization (dead code elimination), not actual work.
//
// Ceilings derived from:
//   - Intel® 64 and IA-32 Architectures Optimization Reference Manual
//   - AMD64 Architecture Programmer's Manual Volume 2
//   - uops.info latency/throughput database
//   - Agner Fog's instruction tables
// =============================================================================

use std::time::Duration;

/// Physical limits for x86-64 CPUs (conservative values that apply
/// to all modern x86-64 microarchitectures from 2015+).
pub struct HardwareCeilings {
    /// Maximum instructions per cycle (IPC).
    /// Skylake: 4-wide decode. Golden Cove: 6-wide. Zen 4: 5-wide.
    pub max_ipc: f64,

    /// Minimum L1 data cache access latency in nanoseconds.
    /// 4 cycles at 3 GHz = 1.33 ns. We use 1.0 ns as absolute floor.
    pub min_l1_access_ns: f64,

    /// Minimum L2 cache access latency in nanoseconds.
    /// 12 cycles at 3 GHz = 4 ns.
    pub min_l2_access_ns: f64,

    /// Minimum DRAM access latency in nanoseconds.
    /// 200 cycles at 3 GHz = 67 ns. Typically 80-150 ns on modern DDR4.
    pub min_dram_access_ns: f64,

    /// Minimum time for a cache-line-sized write (64 bytes).
    /// At best store bandwidth, ~1-2 ns.
    pub min_cacheline_write_ns: f64,

    /// Maximum memory bandwidth in GB/s (DDR4-3200 dual-channel).
    pub max_memory_bandwidth_gbps: f64,

    /// Maximum branch instructions per second (1/cycle at 3 GHz = 3 G/s).
    pub max_branch_throughput: f64,

    /// Minimum branch misprediction penalty in nanoseconds (~15 cycles at 3 GHz = 5 ns).
    pub min_misprediction_penalty_ns: f64,

    /// Maximum integer ALU operations per second (4/cycle at 3 GHz = 12 G/s).
    pub max_alu_throughput: f64,

    /// Minimum multiplication latency in nanoseconds (3 cycles at 3 GHz = 1 ns).
    pub min_multiply_latency_ns: f64,

    /// Minimum division latency in nanoseconds (25-40 cycles at 3 GHz = 8-13 ns).
    pub min_division_latency_ns: f64,

    /// CPU frequency assumption (GHz) for cycle-to-nanosecond conversion.
    pub assumed_freq_ghz: f64,
}

impl Default for HardwareCeilings {
    fn default() -> Self {
        Self {
            max_ipc: 6.0,
            min_l1_access_ns: 1.0,
            min_l2_access_ns: 4.0,
            min_dram_access_ns: 60.0,
            min_cacheline_write_ns: 2.0,
            max_memory_bandwidth_gbps: 45.0,
            max_branch_throughput: 3_000_000_000.0,
            min_misprediction_penalty_ns: 5.0,
            max_alu_throughput: 12_000_000_000.0,
            min_multiply_latency_ns: 1.0,
            min_division_latency_ns: 8.0,
            assumed_freq_ghz: 3.0,
        }
    }
}

/// The result of a sanity check on a benchmark measurement.
#[derive(Debug, Clone)]
pub struct SanityCheckResult {
    /// Name of the benchmark.
    pub name: String,
    /// Measured nanoseconds per iteration.
    pub measured_ns_per_iter: f64,
    /// Measured throughput (operations per second), if applicable.
    pub measured_throughput: Option<f64>,
    /// What type of operation was measured.
    pub operation_type: OperationType,
    /// Whether the measurement passes sanity checks.
    pub passes: bool,
    /// Warning message if the measurement is suspect.
    pub warning: Option<String>,
}

/// Type of operation being benchmarked, used to determine which
/// physical limits apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    /// Pure register ALU operation (add, sub, and, or, xor).
    RegisterALU,
    /// Integer multiplication.
    IntegerMultiply,
    /// Integer division.
    IntegerDivision,
    /// L1 data cache access (load or store).
    L1Access,
    /// L2 cache access.
    L2Access,
    /// Main memory (DRAM) access.
    DRAMAccess,
    /// Cache-line-sized write (64 bytes).
    CacheLineWrite,
    /// Branch instruction.
    Branch,
    /// Memory bandwidth (streaming).
    MemoryBandwidth,
    /// Compound operation (multiple sub-operations).
    Compound,
    /// Unknown / unclassified.
    Unknown,
}

impl HardwareCeilings {
    /// Check a benchmark measurement against physical limits.
    pub fn check(
        &self,
        name: &str,
        measured_ns_per_iter: f64,
        measured_throughput: Option<f64>,
        op_type: OperationType,
    ) -> SanityCheckResult {
        let mut warning = None;
        let mut passes = true;

        match op_type {
            OperationType::RegisterALU => {
                // A single ALU op should take at least ~0.25 ns (1/IPC at 3 GHz)
                // Anything below 0.1 ns is physically impossible
                if measured_ns_per_iter < 0.1 {
                    passes = false;
                    warning = Some(format!(
                        "ALU op at {:.2} ns — physically impossible (min ~0.25 ns at peak IPC). Likely DCE.",
                        measured_ns_per_iter
                    ));
                }
            }
            OperationType::IntegerMultiply => {
                if measured_ns_per_iter < self.min_multiply_latency_ns {
                    passes = false;
                    warning = Some(format!(
                        "Multiply at {:.2} ns — below min latency of {:.2} ns. Likely DCE or folded.",
                        measured_ns_per_iter, self.min_multiply_latency_ns
                    ));
                }
            }
            OperationType::IntegerDivision => {
                if measured_ns_per_iter < self.min_division_latency_ns {
                    passes = false;
                    warning = Some(format!(
                        "Division at {:.2} ns — below min latency of {:.2} ns. Likely DCE.",
                        measured_ns_per_iter, self.min_division_latency_ns
                    ));
                }
            }
            OperationType::L1Access => {
                if measured_ns_per_iter < self.min_l1_access_ns {
                    passes = false;
                    warning = Some(format!(
                        "L1 access at {:.2} ns — below min latency of {:.2} ns. Likely DCE.",
                        measured_ns_per_iter, self.min_l1_access_ns
                    ));
                }
            }
            OperationType::L2Access => {
                if measured_ns_per_iter < self.min_l2_access_ns {
                    passes = false;
                    warning = Some(format!(
                        "L2 access at {:.2} ns — below min latency of {:.2} ns. Likely DCE.",
                        measured_ns_per_iter, self.min_l2_access_ns
                    ));
                }
            }
            OperationType::DRAMAccess => {
                if measured_ns_per_iter < self.min_dram_access_ns {
                    passes = false;
                    warning = Some(format!(
                        "DRAM access at {:.2} ns — below min latency of {:.2} ns. Likely DCE.",
                        measured_ns_per_iter, self.min_dram_access_ns
                    ));
                }
            }
            OperationType::CacheLineWrite => {
                if measured_ns_per_iter < self.min_cacheline_write_ns {
                    passes = false;
                    warning = Some(format!(
                        "64B write at {:.2} ns — below physical limit of {:.2} ns.",
                        measured_ns_per_iter, self.min_cacheline_write_ns
                    ));
                }
            }
            OperationType::Branch => {
                if let Some(tp) = measured_throughput {
                    if tp > self.max_branch_throughput {
                        passes = false;
                        warning = Some(format!(
                            "Branch throughput {:.0}/s — exceeds ceiling of {:.0}/s. Likely DCE.",
                            tp, self.max_branch_throughput
                        ));
                    }
                }
            }
            OperationType::MemoryBandwidth => {
                if let Some(tp) = measured_throughput {
                    let tp_gb = tp * 64.0 / 1_000_000_000.0; // Assume 64B ops
                    if tp_gb > self.max_memory_bandwidth_gbps {
                        passes = false;
                        warning = Some(format!(
                            "Memory bandwidth {:.1} GB/s — exceeds ceiling of {:.1} GB/s. Likely DCE.",
                            tp_gb, self.max_memory_bandwidth_gbps
                        ));
                    }
                }
            }
            OperationType::Compound | OperationType::Unknown => {
                // Can't check against specific limits, but flag sub-nanosecond
                // results for compound operations as highly suspect.
                if measured_ns_per_iter < 0.5 && op_type == OperationType::Compound {
                    passes = false;
                    warning = Some(format!(
                        "Compound op at {:.2} ns — sub-nanosecond compound operations are physically impossible. Likely DCE.",
                        measured_ns_per_iter
                    ));
                }
            }
        }

        SanityCheckResult {
            name: name.to_string(),
            measured_ns_per_iter,
            measured_throughput,
            operation_type: op_type,
            passes,
            warning,
        }
    }
}

/// Format a sanity check result for display.
impl std::fmt::Display for SanityCheckResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.passes { "✓ REAL" } else { "⚠ FAKE" };
        write!(f, "[{}] {} — {:.1} ns/iter", status, self.name, self.measured_ns_per_iter)?;
        if let Some(ref w) = self.warning {
            write!(f, " — {}", w)?;
        }
        Ok(())
    }
}
