// =============================================================================
// Deep Microarchitectural Cost Model with Dependency-DAG Scheduling
//
// Replaces the simplistic hardware_cost_model with a real dependency-graph-based
// latency estimator that models:
//   1. Critical path through a DAG weighted by instruction latency
//   2. Port pressure (throughput bottleneck)
//   3. Frontend decode width
//   4. Macro-fusion (CMP+JCC, TEST+JCC)
//   5. Register pressure penalties
//
// Cost data sourced from uops.info for Skylake, Zen 4, and Golden Cove.
// =============================================================================

#![cfg(feature = "core-superopt")]

use std::collections::HashMap;

// =============================================================================
// Structures
// =============================================================================

/// Per-instruction cost entry from uops.info
#[derive(Debug, Clone, Copy)]
pub struct CostEntry {
    /// Latency in cycles
    pub latency: u8,
    /// Reciprocal throughput (1/IPC) — smaller is better;
    /// 0.25 means 4 instructions per cycle, 1.0 means 1 per cycle
    pub throughput: f32,
    /// Bitmask of execution ports (bit 0 = port 0, bit 1 = port 1, etc.)
    pub ports: u8,
    /// Number of micro-ops this instruction decodes to
    pub num_uops: u8,
    /// Whether this instruction can macro-fuse with a subsequent Jcc
    pub can_macro_fuse: bool,
}

/// Target microarchitecture configuration
#[derive(Debug, Clone, Copy)]
pub struct TargetConfig {
    /// Human-readable name of the microarchitecture
    pub name: &'static str,
    /// Instructions decoded per cycle (4-6 on modern CPUs)
    pub decode_width: u8,
    /// Number of execution ports
    pub num_ports: u8,
    /// Number of general-purpose registers (16 for x86-64)
    pub num_gprs: u8,
    /// Spill penalty kicks in above this many live registers
    pub register_pressure_threshold: u8,
    /// Extra cycles per spill/reload pair
    pub spill_penalty: f64,
}

/// A machine instruction in the flat representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MachineInstr {
    /// Operation to perform
    pub opcode: Opcode,
    /// Destination register (virtual register number)
    pub dst: u8,
    /// First source register (virtual register number)
    pub src1: u8,
    /// Second source register (virtual register number, or unused if has_imm)
    pub src2: u8,
    /// Immediate value (0 if not used)
    pub imm: i32,
    /// Whether the immediate field is active (replaces src2)
    pub has_imm: bool,
}

/// Opcodes supported by the cost model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Opcode {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Neg,
    Not,
    Cmp,
    Test,
    LoadConst,
    Mov,
    Nop,
}

/// Dependency DAG tracking register data dependencies
#[derive(Debug, Clone)]
pub struct DepDag {
    /// Number of nodes (instructions)
    num_nodes: usize,
    /// Opcode for each node (needed for cost lookup during traversal)
    opcodes: Vec<Opcode>,
    /// For each node index, the set of predecessor node indices
    /// (edges pred -> node, meaning node depends on pred)
    predecessors: Vec<Vec<usize>>,
    /// For each node index, the set of successor node indices
    /// (edges node -> succ, meaning succ depends on node)
    successors: Vec<Vec<usize>>,
}

/// Dependency type (used internally during DAG construction)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DepKind {
    /// Read-After-Write: consumer reads a register written by producer
    Raw,
    /// Write-After-Write: later writer must wait for earlier writer
    Waw,
    /// Write-After-Read: later writer must wait for earlier reader
    War,
}

// =============================================================================
// Target presets
// =============================================================================

impl TargetConfig {
    /// Intel Skylake (client, 2015) — 4-wide decode, 7 execution ports
    pub fn skylake() -> Self {
        Self {
            name: "skylake",
            decode_width: 4,
            num_ports: 7,
            num_gprs: 16,
            register_pressure_threshold: 14,
            spill_penalty: 2.0,
        }
    }

    /// Intel Golden Cove (Alder Lake P-core, 2021) — 6-wide decode, 8 execution ports
    pub fn golden_cove() -> Self {
        Self {
            name: "golden_cove",
            decode_width: 6,
            num_ports: 8,
            num_gprs: 16,
            register_pressure_threshold: 14,
            spill_penalty: 2.0,
        }
    }

    /// AMD Zen 4 (2022) — 5-wide decode, 6 execution ports
    pub fn zen4() -> Self {
        Self {
            name: "zen4",
            decode_width: 5,
            num_ports: 6,
            num_gprs: 16,
            register_pressure_threshold: 14,
            spill_penalty: 2.0,
        }
    }

    /// Auto-detect the current CPU via CPUID and return the closest TargetConfig.
    /// Falls back to Skylake if the microarchitecture cannot be identified.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let result = std::arch::x86_64::__cpuid(0);
            let ebx = result.ebx;
            let edx = result.edx;
            let ecx = result.ecx;

            let vendor = [
                (ebx & 0xFF) as u8,
                ((ebx >> 8) & 0xFF) as u8,
                ((ebx >> 16) & 0xFF) as u8,
                ((ebx >> 24) & 0xFF) as u8,
                (edx & 0xFF) as u8,
                ((edx >> 8) & 0xFF) as u8,
                ((edx >> 16) & 0xFF) as u8,
                ((edx >> 24) & 0xFF) as u8,
                (ecx & 0xFF) as u8,
                ((ecx >> 8) & 0xFF) as u8,
                ((ecx >> 16) & 0xFF) as u8,
                ((ecx >> 24) & 0xFF) as u8,
            ];

            let cpuid1 = std::arch::x86_64::__cpuid(1);
            let eax = cpuid1.eax;

            let base_family = (eax >> 8) & 0xF;
            let base_model = (eax >> 4) & 0xF;
            let extended_family = (eax >> 20) & 0xFF;
            let extended_model = (eax >> 16) & 0xF;

            let family = if base_family == 0xF {
                base_family + extended_family
            } else {
                base_family
            };

            let model = if base_family == 0x6 || base_family == 0xF {
                (extended_model << 4) + base_model
            } else {
                base_model
            };

            if &vendor == b"GenuineIntel" {
                match (family, model) {
                    // Alder Lake / Raptor Lake P-core
                    (6, 0x97) | (6, 0x9A) | (6, 0xB7) => Self::golden_cove(),
                    // Skylake / Kaby Lake
                    (6, 0x5E) | (6, 0x8E) | (6, 0x9E) => Self::skylake(),
                    // Skylake-X / Cascade Lake
                    (6, 0x55) => Self::skylake(),
                    // Ice Lake
                    (6, 0x7D) | (6, 0x7E) => Self::golden_cove(),
                    // Default modern Intel → Skylake as safe baseline
                    (6, _) => Self::skylake(),
                    _ => Self::skylake(),
                }
            } else if &vendor == b"AuthenticAMD" {
                match (family, model) {
                    // Zen 4 (Ryzen 7000, EPYC Genoa)
                    (0x19, 0x10..=0x1F) | (0x19, 0x60..=0x6F) => Self::zen4(),
                    // Zen 3 → fall through to Zen 4 as close approximation
                    (0x19, 0x00..=0x0F) | (0x19, 0x20..=0x5F) => Self::zen4(),
                    // Zen 2 / older
                    (0x17, _) => Self::zen4(),
                    _ => Self::skylake(),
                }
            } else {
                Self::skylake()
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            // Non-x86 platforms: default to Skylake parameters as a
            // reasonable generic baseline.
            Self::skylake()
        }
    }
}

// =============================================================================
// Cost database — uops.info data per target
// =============================================================================

/// Port bitmask constants for Skylake integer ALU operations (ports 0, 1, 5, 6)
const SKL_ALU_PORTS: u8 = (1 << 0) | (1 << 1) | (1 << 5) | (1 << 6); // 0b01100011 = 99

/// Port bitmask for Skylake shift operations (ports 0, 6)
const SKL_SHIFT_PORTS: u8 = (1 << 0) | (1 << 6); // 0b01000001 = 65

/// Port bitmask for Skylake multiply (port 1 only)
const SKL_MUL_PORTS: u8 = 1 << 1; // 2

/// Port bitmask for Skylake divide (port 0 only)
const SKL_DIV_PORTS: u8 = 1 << 0; // 1

/// Port bitmask constants for Zen 4 integer ALU operations (ports 0, 1, 2, 3)
const ZN4_ALU_PORTS: u8 = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3); // 0b00001111 = 15

/// Port bitmask for Zen 4 multiply (ports 0, 1 — two integer multiply units)
const ZN4_MUL_PORTS: u8 = (1 << 0) | (1 << 1); // 3

/// Port bitmask for Zen 4 divide (port 0 only)
const ZN4_DIV_PORTS: u8 = 1 << 0; // 1

/// Port bitmask for Golden Cove integer ALU (ports 0, 1, 5, 6)
const GNC_ALU_PORTS: u8 = (1 << 0) | (1 << 1) | (1 << 5) | (1 << 6); // 99

/// Port bitmask for Golden Cove shift (ports 0, 6)
const GNC_SHIFT_PORTS: u8 = (1 << 0) | (1 << 6); // 65

/// Port bitmask for Golden Cove multiply (port 1)
const GNC_MUL_PORTS: u8 = 1 << 1; // 2

/// Port bitmask for Golden Cove divide (port 0)
const GNC_DIV_PORTS: u8 = 1 << 0; // 1

/// Cost lookup for Intel Skylake (client).
///
/// Data from uops.info for SKL:
///   ADD/SUB r64,r64:  latency 1, rthroughput 0.25, ports 0|1|5|6
///   IMUL r64,r64:     latency 3, rthroughput 1,    port 1
///   IDIV r64:         latency 35-40, rthroughput 35-40, port 0
///   SHL/SHR r64,imm8: latency 1, rthroughput 0.5,  ports 0|6
fn skylake_cost(opcode: Opcode) -> CostEntry {
    match opcode {
        Opcode::Add => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: SKL_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Sub => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: SKL_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Mul => CostEntry {
            latency: 3,
            throughput: 1.0,
            ports: SKL_MUL_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Div => CostEntry {
            latency: 35,
            throughput: 35.0,
            ports: SKL_DIV_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Rem => CostEntry {
            latency: 35,
            throughput: 35.0,
            ports: SKL_DIV_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::And => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: SKL_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Or => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: SKL_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Xor => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: SKL_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Shl => CostEntry {
            latency: 1,
            throughput: 0.5,
            ports: SKL_SHIFT_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Shr => CostEntry {
            latency: 1,
            throughput: 0.5,
            ports: SKL_SHIFT_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Neg => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: SKL_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Not => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: SKL_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Cmp => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: SKL_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: true,
        },
        Opcode::Test => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: SKL_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: true,
        },
        Opcode::LoadConst => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: SKL_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Mov => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: SKL_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Nop => CostEntry {
            latency: 0,
            throughput: 0.0,
            ports: 0,
            num_uops: 1,
            can_macro_fuse: false,
        },
    }
}

/// Cost lookup for Intel Golden Cove (Alder Lake P-core).
///
/// Data from uops.info for GNC:
///   ADD/SUB r64,r64:  latency 1, rthroughput 0.25, ports 0|1|5|6
///   IMUL r64,r64:     latency 3, rthroughput 1,    port 1
///   IDIV r64:         latency ~28-30, rthroughput ~28-30, port 0
///   SHL/SHR r64,imm8: latency 1, rthroughput 0.5,  ports 0|6
fn golden_cove_cost(opcode: Opcode) -> CostEntry {
    match opcode {
        Opcode::Add => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: GNC_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Sub => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: GNC_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Mul => CostEntry {
            latency: 3,
            throughput: 1.0,
            ports: GNC_MUL_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Div => CostEntry {
            latency: 30,
            throughput: 30.0,
            ports: GNC_DIV_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Rem => CostEntry {
            latency: 30,
            throughput: 30.0,
            ports: GNC_DIV_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::And => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: GNC_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Or => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: GNC_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Xor => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: GNC_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Shl => CostEntry {
            latency: 1,
            throughput: 0.5,
            ports: GNC_SHIFT_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Shr => CostEntry {
            latency: 1,
            throughput: 0.5,
            ports: GNC_SHIFT_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Neg => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: GNC_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Not => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: GNC_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Cmp => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: GNC_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: true,
        },
        Opcode::Test => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: GNC_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: true,
        },
        Opcode::LoadConst => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: GNC_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Mov => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: GNC_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Nop => CostEntry {
            latency: 0,
            throughput: 0.0,
            ports: 0,
            num_uops: 1,
            can_macro_fuse: false,
        },
    }
}

/// Cost lookup for AMD Zen 4.
///
/// Data from uops.info for ZN4:
///   ADD/SUB r64,r64:  latency 1, rthroughput 0.25, ports 0|1|2|3
///   IMUL r64,r64:     latency 3, rthroughput 1,    ports 0|1
///   IDIV r64:         latency ~25 (variable 17-39), rthroughput ~25, port 0
///   SHL/SHR r64,imm8: latency 1, rthroughput 0.25, ports 0|1|2|3
fn zen4_cost(opcode: Opcode) -> CostEntry {
    match opcode {
        Opcode::Add => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: ZN4_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Sub => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: ZN4_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Mul => CostEntry {
            latency: 3,
            throughput: 1.0,
            ports: ZN4_MUL_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Div => CostEntry {
            latency: 25,
            throughput: 25.0,
            ports: ZN4_DIV_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Rem => CostEntry {
            latency: 25,
            throughput: 25.0,
            ports: ZN4_DIV_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::And => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: ZN4_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Or => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: ZN4_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Xor => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: ZN4_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Shl => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: ZN4_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Shr => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: ZN4_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Neg => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: ZN4_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Not => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: ZN4_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Cmp => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: ZN4_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: true,
        },
        Opcode::Test => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: ZN4_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: true,
        },
        Opcode::LoadConst => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: ZN4_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Mov => CostEntry {
            latency: 1,
            throughput: 0.25,
            ports: ZN4_ALU_PORTS,
            num_uops: 1,
            can_macro_fuse: false,
        },
        Opcode::Nop => CostEntry {
            latency: 0,
            throughput: 0.0,
            ports: 0,
            num_uops: 1,
            can_macro_fuse: false,
        },
    }
}

/// Look up the cost entry for a given opcode on the specified target.
///
/// Dispatches to the appropriate per-target cost table based on `target.name`.
/// Falls back to Skylake data for unknown targets.
pub fn cost_entry(opcode: Opcode, target: TargetConfig) -> CostEntry {
    match target.name {
        "skylake" => skylake_cost(opcode),
        "golden_cove" => golden_cove_cost(opcode),
        "zen4" => zen4_cost(opcode),
        // Unknown target → fall back to Skylake
        _ => skylake_cost(opcode),
    }
}

// =============================================================================
// Instruction operand helpers
// =============================================================================

/// Returns true if the opcode reads src1 as a register operand.
fn reads_src1(opcode: Opcode) -> bool {
    match opcode {
        // LoadConst loads an immediate; src1 is not read as a register
        Opcode::LoadConst | Opcode::Nop => false,
        _ => true,
    }
}

/// Returns true if the opcode reads src2 as a register operand.
/// If has_imm is true, src2 is not a register (the immediate is used instead).
fn reads_src2(opcode: Opcode, has_imm: bool) -> bool {
    if has_imm {
        return false;
    }
    match opcode {
        // Neg and Not are unary operations — only src1 is used
        Opcode::Neg | Opcode::Not | Opcode::LoadConst | Opcode::Nop => false,
        _ => true,
    }
}

/// Returns true if the opcode writes to the dst register.
fn writes_dst(opcode: Opcode) -> bool {
    match opcode {
        // Cmp/Test write to flags, modeled as writing to dst in our representation
        // Nop writes nothing
        Opcode::Nop => false,
        _ => true,
    }
}

/// Returns true if the opcode represents a conditional branch that can
/// participate in macro-fusion with a preceding CMP or TEST.
///
/// Currently our opcode set has no branch instructions; this returns false
/// for all opcodes. When branch opcodes are added, this function should be
/// updated to return true for them.
fn is_conditional_branch(_opcode: Opcode) -> bool {
    false
}

// =============================================================================
// DAG builder
// =============================================================================

/// Build a dependency DAG from a sequence of machine instructions.
///
/// Tracks three kinds of register dependencies:
/// - **RAW** (Read-After-Write): consumer reads a register written by a producer
/// - **WAW** (Write-After-Write): later writer must come after earlier writer
/// - **WAR** (Write-After-Read): later writer must come after earlier reader
///
/// On an out-of-order CPU with register renaming, only RAW is a true
/// dependency; WAW and WAR are "name" dependencies resolved by renaming.
/// However, we track all three for a conservative cost estimate, which is
/// appropriate for superoptimization purposes.
pub fn build_dependency_dag(instrs: &[MachineInstr]) -> DepDag {
    let n = instrs.len();
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let opcodes: Vec<Opcode> = instrs.iter().map(|i| i.opcode).collect();

    // Maps: register → last instruction index that defined (wrote) it
    let mut last_def: HashMap<u8, usize> = HashMap::new();
    // Maps: register → last instruction index that used (read) it
    let mut last_use: HashMap<u8, usize> = HashMap::new();

    /// Add a dependency edge from `from` to `to`, avoiding duplicates
    fn add_edge(
        from: usize,
        to: usize,
        predecessors: &mut [Vec<usize>],
        successors: &mut [Vec<usize>],
    ) {
        if from == to {
            return;
        }
        // Avoid duplicate edges
        if !predecessors[to].contains(&from) {
            predecessors[to].push(from);
            successors[from].push(to);
        }
    }

    for (i, instr) in instrs.iter().enumerate() {
        // --- RAW dependencies: this instruction reads a register written earlier ---
        if reads_src1(instr.opcode) {
            if let Some(&prev) = last_def.get(&instr.src1) {
                add_edge(prev, i, &mut predecessors, &mut successors);
            }
        }
        if reads_src2(instr.opcode, instr.has_imm) {
            if let Some(&prev) = last_def.get(&instr.src2) {
                add_edge(prev, i, &mut predecessors, &mut successors);
            }
        }

        // --- WAW dependency: this instruction writes a register written earlier ---
        if writes_dst(instr.opcode) {
            if let Some(&prev) = last_def.get(&instr.dst) {
                add_edge(prev, i, &mut predecessors, &mut successors);
            }
        }

        // --- WAR dependency: this instruction writes a register read earlier ---
        if writes_dst(instr.opcode) {
            if let Some(&prev) = last_use.get(&instr.dst) {
                // Only add WAR edge if the reader is not the same as the last
                // writer (otherwise it's already covered by RAW/WAW)
                if last_def.get(&instr.dst).map_or(true, |&d| d != prev) {
                    add_edge(prev, i, &mut predecessors, &mut successors);
                }
            }
        }

        // --- Update maps ---
        // Record reads before the write so that same-instruction read-after-write
        // within the same instruction doesn't create a false self-loop
        if reads_src1(instr.opcode) {
            last_use.insert(instr.src1, i);
        }
        if reads_src2(instr.opcode, instr.has_imm) {
            last_use.insert(instr.src2, i);
        }
        if writes_dst(instr.opcode) {
            last_def.insert(instr.dst, i);
        }
    }

    DepDag {
        num_nodes: n,
        opcodes,
        predecessors,
        successors,
    }
}

// =============================================================================
// Critical path computation
// =============================================================================

/// Compute the critical path length through the dependency DAG.
///
/// The critical path is the longest path through the DAG where each edge
/// is weighted by the latency of the source node. Uses topological sort
/// followed by dynamic programming (longest path in a DAG).
///
/// Returns the length in cycles as a floating-point value.
pub fn critical_path_length(dag: &DepDag, cost_db: &dyn Fn(Opcode) -> CostEntry) -> f64 {
    if dag.num_nodes == 0 {
        return 0.0;
    }

    // --- Topological sort (Kahn's algorithm) ---
    let mut in_degree: Vec<usize> = vec![0; dag.num_nodes];
    for i in 0..dag.num_nodes {
        in_degree[i] = dag.predecessors[i].len();
    }

    let mut queue: Vec<usize> = Vec::with_capacity(dag.num_nodes);
    for i in 0..dag.num_nodes {
        if in_degree[i] == 0 {
            queue.push(i);
        }
    }

    let mut topo_order: Vec<usize> = Vec::with_capacity(dag.num_nodes);
    while let Some(node) = queue.pop() {
        topo_order.push(node);
        for &succ in &dag.successors[node] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push(succ);
            }
        }
    }

    // If topo_order is shorter than num_nodes, there's a cycle.
    // In that case, fall back to sequential execution.
    if topo_order.len() != dag.num_nodes {
        // Sum all latencies sequentially as worst case
        let mut total = 0.0;
        for i in 0..dag.num_nodes {
            let cost = cost_db(dag.opcodes[i]);
            total += cost.latency as f64;
        }
        return total;
    }

    // --- DP: dist[i] = critical path length ending at node i ---
    let mut dist: Vec<f64> = vec![0.0; dag.num_nodes];

    for &node in &topo_order {
        let cost = cost_db(dag.opcodes[node]);
        let node_latency = cost.latency as f64;

        // dist[node] = latency[node] + max(dist[pred] for all predecessors)
        let max_pred = dag.predecessors[node]
            .iter()
            .map(|&pred| dist[pred])
            .fold(0.0f64, f64::max);

        dist[node] = node_latency + max_pred;
    }

    // The critical path is the maximum dist across all nodes
    dist.iter().cloned().fold(0.0f64, f64::max)
}

// =============================================================================
// Port pressure calculation
// =============================================================================

/// Compute the port-pressure throughput bound for the instruction sequence.
///
/// Distributes each instruction's uops evenly across its available execution
/// ports, then finds the maximum pressure on any single port. This represents
/// the minimum number of cycles needed considering only port contention.
pub fn port_pressure_bound(
    instrs: &[MachineInstr],
    target: &TargetConfig,
    cost_db: &dyn Fn(Opcode) -> CostEntry,
) -> f64 {
    if instrs.is_empty() {
        return 0.0;
    }

    // Track pressure on each port (port 0 through port num_ports-1)
    let mut port_pressure: Vec<f64> = vec![0.0; target.num_ports as usize];

    for instr in instrs {
        let cost = cost_db(instr.opcode);
        if cost.num_uops == 0 || cost.ports == 0 {
            continue;
        }

        // Count how many ports this instruction can use
        let num_available_ports = cost.ports.count_ones() as f64;

        // Distribute uops evenly across available ports.
        // Each available port gets (num_uops / num_available_ports) of pressure.
        let pressure_per_port = cost.num_uops as f64 / num_available_ports;

        for port_idx in 0..target.num_ports as usize {
            if cost.ports & (1 << port_idx) != 0 {
                port_pressure[port_idx] += pressure_per_port;
            }
        }
    }

    // The throughput bound is the maximum pressure on any single port
    port_pressure.iter().cloned().fold(0.0f64, f64::max)
}

// =============================================================================
// Register pressure estimation
// =============================================================================

/// Estimate the maximum register pressure (number of simultaneously live
/// virtual registers) in the instruction sequence using interval analysis.
///
/// A register is considered "live" from its first definition to its last use.
/// If a register is defined but never used, it is conservatively considered
/// live until the end of the block (it may be a live-out value needed by
/// successor blocks).
///
/// Returns the maximum number of simultaneously live registers.
pub fn register_pressure(instrs: &[MachineInstr]) -> u8 {
    if instrs.is_empty() {
        return 0;
    }

    let n = instrs.len();

    // --- Phase 1: Compute live intervals for each register ---
    // def[r] = first instruction index where r is defined (written as dst)
    // last_use[r] = last instruction index where r is used (read as src1 or src2)
    let mut def: HashMap<u8, usize> = HashMap::new();
    let mut last_use: HashMap<u8, usize> = HashMap::new();

    for (i, instr) in instrs.iter().enumerate() {
        // Record definition
        if writes_dst(instr.opcode) {
            def.entry(instr.dst).or_insert(i);
        }

        // Record uses
        if reads_src1(instr.opcode) {
            last_use.insert(instr.src1, i);
        }
        if reads_src2(instr.opcode, instr.has_imm) {
            last_use.insert(instr.src2, i);
        }
    }

    // --- Phase 2: Compute max live registers at each instruction ---
    // A register r is live at instruction i if:
    //   def[r] <= i <= last_use[r]
    // If defined but never used, conservatively live until end of block.
    let mut max_pressure: u8 = 0;

    for i in 0..n {
        let mut live_count: u8 = 0;
        for (&reg, &def_idx) in &def {
            let end = last_use.get(&reg).copied().unwrap_or(n - 1);
            if def_idx <= i && i <= end {
                live_count += 1;
            }
        }
        max_pressure = max_pressure.max(live_count);
    }

    max_pressure
}

// =============================================================================
// Macro-fusion detection
// =============================================================================

/// Detect macro-fusion opportunities in the instruction sequence.
///
/// Macro-fusion occurs when a CMP or TEST instruction is immediately followed
/// by a conditional branch (Jcc). The pair fuses into a single uop, saving
/// 1 uop from the frontend and reducing port pressure.
///
/// Returns a list of (cmp_or_test_index, branch_index) pairs where fusion
/// occurs. Since the current Opcode set has no branch instructions, this
/// function will not detect any fusions unless branch opcodes are added.
///
/// Additionally, if the last instruction is a CMP/TEST with can_macro_fuse=true,
/// it is assumed to fuse with the implicit block-ending branch. In this case
/// the branch index is set to `usize::MAX` as a sentinel.
fn detect_macro_fusions(
    instrs: &[MachineInstr],
    cost_db: &dyn Fn(Opcode) -> CostEntry,
) -> Vec<(usize, usize)> {
    let mut fusions = Vec::new();

    for i in 0..instrs.len() {
        let cost = cost_db(instrs[i].opcode);
        if !cost.can_macro_fuse {
            continue;
        }

        // Check if the next instruction is a conditional branch
        if i + 1 < instrs.len() && is_conditional_branch(instrs[i + 1].opcode) {
            fusions.push((i, i + 1));
        }

        // If this is the last instruction, assume it fuses with the
        // implicit block-ending conditional branch
        if i == instrs.len() - 1 {
            fusions.push((i, usize::MAX));
        }
    }

    fusions
}

/// Count the number of uops saved by macro-fusion.
///
/// Each fusion saves 1 uop (the branch uop is absorbed into the CMP/TEST uop).
fn fusion_uop_savings(
    instrs: &[MachineInstr],
    cost_db: &dyn Fn(Opcode) -> CostEntry,
) -> u8 {
    let fusions = detect_macro_fusions(instrs, cost_db);
    // Each fusion saves 1 uop (the branch), but for the implicit end-of-block
    // fusion (branch_index == usize::MAX), we save 1 uop from the implicit branch.
    fusions.len() as u8
}

// =============================================================================
// Main estimation function
// =============================================================================

/// Estimate the cycle count for a basic block of instructions on the given
/// target microarchitecture.
///
/// Returns: `max(critical_path, throughput_bound, frontend_bound) + register_pressure_penalty`
///
/// Where:
/// - `critical_path` = longest dependency chain weighted by instruction latency
/// - `throughput_bound` = max port pressure across all execution ports
/// - `frontend_bound` = total_uops / decode_width
/// - `register_pressure_penalty` = if pressure > threshold then
///   `(pressure - threshold) * spill_penalty` else 0
pub fn estimate_block_cycles(instrs: &[MachineInstr], target: TargetConfig) -> f64 {
    if instrs.is_empty() {
        return 0.0;
    }

    let cost_db = |op: Opcode| cost_entry(op, target);

    // 1. Build dependency DAG
    let dag = build_dependency_dag(instrs);

    // 2. Compute critical path length
    let critical_path = critical_path_length(&dag, &cost_db);

    // 3. Compute port-pressure throughput bound
    let throughput_bound = port_pressure_bound(instrs, &target, &cost_db);

    // 4. Compute frontend decode bound
    let total_uops: u32 = instrs
        .iter()
        .map(|instr| cost_db(instr.opcode).num_uops as u32)
        .sum();
    let savings = fusion_uop_savings(instrs, &cost_db) as u32;
    let effective_uops = total_uops.saturating_sub(savings);
    let frontend_bound = if target.decode_width > 0 {
        effective_uops as f64 / target.decode_width as f64
    } else {
        effective_uops as f64
    };

    // 5. Compute register pressure penalty
    let pressure = register_pressure(instrs) as f64;
    let threshold = target.register_pressure_threshold as f64;
    let reg_pressure_penalty = if pressure > threshold {
        (pressure - threshold) * target.spill_penalty
    } else {
        0.0
    };

    // 6. Final estimate: the bottleneck plus register pressure penalty
    let bottleneck = critical_path.max(throughput_bound).max(frontend_bound);
    bottleneck + reg_pressure_penalty
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a simple machine instruction
    fn make_instr(opcode: Opcode, dst: u8, src1: u8, src2: u8) -> MachineInstr {
        MachineInstr {
            opcode,
            dst,
            src1,
            src2,
            imm: 0,
            has_imm: false,
        }
    }

    /// Helper to create an instruction with an immediate operand
    fn make_instr_imm(opcode: Opcode, dst: u8, src1: u8, imm: i32) -> MachineInstr {
        MachineInstr {
            opcode,
            dst,
            src1,
            src2: 0,
            imm,
            has_imm: true,
        }
    }

    // -------------------------------------------------------------------------
    // Test: Independent adds should parallelize → critical path = 1 cycle
    // -------------------------------------------------------------------------
    #[test]
    fn test_independent_adds_critical_path() {
        // ADD r0, r1, r2  (writes r0, reads r1 and r2)
        // ADD r3, r4, r5  (writes r3, reads r4 and r5)
        // These are independent — no data dependencies between them.
        let instrs = vec![
            make_instr(Opcode::Add, 0, 1, 2),
            make_instr(Opcode::Add, 3, 4, 5),
        ];

        let target = TargetConfig::skylake();
        let dag = build_dependency_dag(&instrs);
        let cost_db = |op: Opcode| cost_entry(op, target);

        let cp = critical_path_length(&dag, &cost_db);

        // Each ADD has latency 1, and they are independent,
        // so the critical path is just 1 cycle.
        assert_eq!(cp, 1.0, "Independent adds should have critical path of 1 cycle");
    }

    // -------------------------------------------------------------------------
    // Test: Dependent add chain → critical path = 2 cycles
    // -------------------------------------------------------------------------
    #[test]
    fn test_dependent_add_chain_critical_path() {
        // ADD r0, r1, r2  (writes r0)
        // ADD r3, r0, r4  (reads r0 — depends on first ADD)
        let instrs = vec![
            make_instr(Opcode::Add, 0, 1, 2),
            make_instr(Opcode::Add, 3, 0, 4), // reads r0 produced by first ADD
        ];

        let target = TargetConfig::skylake();
        let dag = build_dependency_dag(&instrs);
        let cost_db = |op: Opcode| cost_entry(op, target);

        let cp = critical_path_length(&dag, &cost_db);

        // ADD latency 1 + ADD latency 1 = 2 cycles on the critical path
        assert_eq!(
            cp, 2.0,
            "Dependent add chain should have critical path of 2 cycles"
        );
    }

    // -------------------------------------------------------------------------
    // Test: Division dominates cost
    // -------------------------------------------------------------------------
    #[test]
    fn test_division_dominates_cost() {
        let target = TargetConfig::skylake();

        // A block with just ADDs
        let add_block = vec![
            make_instr(Opcode::Add, 0, 1, 2),
            make_instr(Opcode::Add, 3, 0, 4),
            make_instr(Opcode::Add, 5, 3, 6),
        ];

        // A block with a DIV that the last ADD depends on
        let div_block = vec![
            make_instr(Opcode::Div, 0, 1, 2), // DIV r0, r1, r2
            make_instr(Opcode::Add, 3, 0, 4),  // ADD r3, r0, r4  (depends on DIV)
        ];

        let add_cost = estimate_block_cycles(&add_block, target);
        let div_cost = estimate_block_cycles(&div_block, target);

        assert!(
            div_cost > add_cost,
            "Block with DIV (cost={div_cost}) should be more expensive than block with ADDs (cost={add_cost})"
        );

        // The critical path through the DIV block should be at least 35 (DIV latency) + 1 (ADD)
        let dag = build_dependency_dag(&div_block);
        let cost_db = |op: Opcode| cost_entry(op, target);
        let cp = critical_path_length(&dag, &cost_db);
        assert!(
            cp >= 36.0,
            "Critical path through DIV+ADD should be at least 36, got {cp}"
        );
    }

    // -------------------------------------------------------------------------
    // Test: Register pressure penalty kicks in when > 14 GPRs live
    // -------------------------------------------------------------------------
    #[test]
    fn test_register_pressure_penalty() {
        let target = TargetConfig::skylake();

        // Create 10 LoadConst instructions — 10 live registers, below threshold of 14
        let mut low_pressure_instrs: Vec<MachineInstr> = Vec::new();
        for i in 0u8..10 {
            low_pressure_instrs.push(make_instr_imm(Opcode::LoadConst, i, 0, 0));
        }

        let pressure_low = register_pressure(&low_pressure_instrs);
        assert!(
            pressure_low <= 14,
            "Low pressure block should have <= 14 live registers, got {pressure_low}"
        );

        let cost_low = estimate_block_cycles(&low_pressure_instrs, target);

        // Create 15 LoadConst instructions — 15 live registers, exceeds threshold of 14
        let mut high_pressure_instrs: Vec<MachineInstr> = Vec::new();
        for i in 0u8..15 {
            high_pressure_instrs.push(make_instr_imm(Opcode::LoadConst, i, 0, 0));
        }

        let pressure_high = register_pressure(&high_pressure_instrs);
        assert_eq!(
            pressure_high, 15,
            "High pressure block should have 15 live registers, got {pressure_high}"
        );

        let cost_high = estimate_block_cycles(&high_pressure_instrs, target);

        // The high-pressure block should be more expensive due to spill penalty
        assert!(
            cost_high > cost_low,
            "High register pressure (cost={cost_high}) should be more expensive than low pressure (cost={cost_low})"
        );

        // Verify the penalty is exactly (15 - 14) * 2.0 = 2.0 extra cycles
        let expected_penalty = (15.0 - 14.0) * target.spill_penalty;
        let cost_without_penalty = {
            let cost_db = |op: Opcode| cost_entry(op, target);
            let dag = build_dependency_dag(&high_pressure_instrs);
            let cp = critical_path_length(&dag, &cost_db);
            let tp = port_pressure_bound(&high_pressure_instrs, &target, &cost_db);
            let fb = 15.0 / target.decode_width as f64;
            cp.max(tp).max(fb)
        };
        let diff = cost_high - cost_without_penalty;
        assert!(
            (diff - expected_penalty).abs() < 0.01,
            "Spill penalty should be {expected_penalty}, got {diff}"
        );
    }

    // -------------------------------------------------------------------------
    // Test: Cost entry data integrity
    // -------------------------------------------------------------------------
    #[test]
    fn test_cost_entry_data() {
        let skylake = TargetConfig::skylake();

        // Verify required minimum data from the spec
        let add = cost_entry(Opcode::Add, skylake);
        assert_eq!(add.latency, 1);
        assert!((add.throughput - 0.25).abs() < 0.01);
        assert_ne!(add.ports & SKL_ALU_PORTS, 0); // uses at least some ALU ports

        let sub = cost_entry(Opcode::Sub, skylake);
        assert_eq!(sub.latency, 1);
        assert!((sub.throughput - 0.25).abs() < 0.01);

        let mul = cost_entry(Opcode::Mul, skylake);
        assert_eq!(mul.latency, 3);
        assert!((mul.throughput - 1.0).abs() < 0.01);

        let div = cost_entry(Opcode::Div, skylake);
        assert!(div.latency >= 35 && div.latency <= 40);
        assert!(div.throughput >= 35.0 && div.throughput <= 40.0);

        let and = cost_entry(Opcode::And, skylake);
        assert_eq!(and.latency, 1);
        assert!((and.throughput - 0.25).abs() < 0.01);

        let or = cost_entry(Opcode::Or, skylake);
        assert_eq!(or.latency, 1);
        assert!((or.throughput - 0.25).abs() < 0.01);

        let xor = cost_entry(Opcode::Xor, skylake);
        assert_eq!(xor.latency, 1);
        assert!((xor.throughput - 0.25).abs() < 0.01);

        let shl = cost_entry(Opcode::Shl, skylake);
        assert_eq!(shl.latency, 1);
        assert!((shl.throughput - 0.5).abs() < 0.01);

        let shr = cost_entry(Opcode::Shr, skylake);
        assert_eq!(shr.latency, 1);
        assert!((shr.throughput - 0.5).abs() < 0.01);

        let neg = cost_entry(Opcode::Neg, skylake);
        assert_eq!(neg.latency, 1);
        assert!((neg.throughput - 0.25).abs() < 0.01);

        let not = cost_entry(Opcode::Not, skylake);
        assert_eq!(not.latency, 1);
        assert!((not.throughput - 0.25).abs() < 0.01);

        let cmp = cost_entry(Opcode::Cmp, skylake);
        assert_eq!(cmp.latency, 1);
        assert!((cmp.throughput - 0.25).abs() < 0.01);
        assert!(cmp.can_macro_fuse);

        let test = cost_entry(Opcode::Test, skylake);
        assert_eq!(test.latency, 1);
        assert!((test.throughput - 0.25).abs() < 0.01);
        assert!(test.can_macro_fuse);
    }

    // -------------------------------------------------------------------------
    // Test: Target presets
    // -------------------------------------------------------------------------
    #[test]
    fn test_target_presets() {
        let skylake = TargetConfig::skylake();
        assert_eq!(skylake.name, "skylake");
        assert_eq!(skylake.decode_width, 4);
        assert_eq!(skylake.num_gprs, 16);
        assert_eq!(skylake.register_pressure_threshold, 14);

        let gnc = TargetConfig::golden_cove();
        assert_eq!(gnc.name, "golden_cove");
        assert_eq!(gnc.decode_width, 6);

        let zen4 = TargetConfig::zen4();
        assert_eq!(zen4.name, "zen4");
        assert_eq!(zen4.decode_width, 5);
    }

    // -------------------------------------------------------------------------
    // Test: DAG construction with RAW dependency
    // -------------------------------------------------------------------------
    #[test]
    fn test_dag_raw_dependency() {
        // ADD r0, r1, r2
        // ADD r3, r0, r4  ← RAW on r0
        let instrs = vec![
            make_instr(Opcode::Add, 0, 1, 2),
            make_instr(Opcode::Add, 3, 0, 4),
        ];

        let dag = build_dependency_dag(&instrs);

        // Second instruction should depend on the first
        assert!(
            dag.predecessors[1].contains(&0),
            "Second ADD should have RAW dependency on first ADD"
        );
        assert!(
            dag.successors[0].contains(&1),
            "First ADD should have successor edge to second ADD"
        );
    }

    // -------------------------------------------------------------------------
    // Test: DAG construction with WAW dependency
    // -------------------------------------------------------------------------
    #[test]
    fn test_dag_waw_dependency() {
        // ADD r0, r1, r2
        // ADD r0, r3, r4  ← WAW on r0 (both write r0)
        let instrs = vec![
            make_instr(Opcode::Add, 0, 1, 2),
            make_instr(Opcode::Add, 0, 3, 4),
        ];

        let dag = build_dependency_dag(&instrs);

        // Second instruction should depend on the first (WAW)
        assert!(
            dag.predecessors[1].contains(&0),
            "Second ADD should have WAW dependency on first ADD"
        );
    }

    // -------------------------------------------------------------------------
    // Test: DAG with no dependencies
    // -------------------------------------------------------------------------
    #[test]
    fn test_dag_no_dependencies() {
        let instrs = vec![
            make_instr(Opcode::Add, 0, 1, 2),
            make_instr(Opcode::Add, 3, 4, 5),
            make_instr(Opcode::Add, 6, 7, 8),
        ];

        let dag = build_dependency_dag(&instrs);

        for i in 0..3 {
            assert!(
                dag.predecessors[i].is_empty(),
                "Instruction {i} should have no predecessors (all independent)"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Test: Critical path with longer chain
    // -------------------------------------------------------------------------
    #[test]
    fn test_critical_path_chain() {
        let target = TargetConfig::skylake();
        let cost_db = |op: Opcode| cost_entry(op, target);

        // ADD r0, r1, r2   (latency 1)
        // ADD r3, r0, r4   (latency 1, depends on r0)
        // MUL r5, r3, r6   (latency 3, depends on r3)
        let instrs = vec![
            make_instr(Opcode::Add, 0, 1, 2),
            make_instr(Opcode::Add, 3, 0, 4),
            make_instr(Opcode::Mul, 5, 3, 6),
        ];

        let dag = build_dependency_dag(&instrs);
        let cp = critical_path_length(&dag, &cost_db);

        // Critical path: 1 + 1 + 3 = 5 cycles
        assert_eq!(cp, 5.0, "Chain ADD→ADD→MUL should have critical path of 5");
    }

    // -------------------------------------------------------------------------
    // Test: Port pressure bound
    // -------------------------------------------------------------------------
    #[test]
    fn test_port_pressure_bound() {
        let target = TargetConfig::skylake();
        let cost_db = |op: Opcode| cost_entry(op, target);

        // 4 independent ADDs on Skylake
        // Each ADD: 1 uop, ports {0,1,5,6} (4 ports)
        // Pressure per port: 4 * (1/4) = 1.0
        let instrs = vec![
            make_instr(Opcode::Add, 0, 1, 2),
            make_instr(Opcode::Add, 3, 4, 5),
            make_instr(Opcode::Add, 6, 7, 8),
            make_instr(Opcode::Add, 9, 10, 11),
        ];

        let tp = port_pressure_bound(&instrs, &target, &cost_db);

        // Each port gets 4 * (1/4) = 1.0 pressure
        assert!(
            (tp - 1.0).abs() < 0.01,
            "4 ADDs on Skylake should have port pressure of ~1.0, got {tp}"
        );
    }

    // -------------------------------------------------------------------------
    // Test: MUL bottlenecks on port 1 on Skylake
    // -------------------------------------------------------------------------
    #[test]
    fn test_mul_port_bottleneck() {
        let target = TargetConfig::skylake();
        let cost_db = |op: Opcode| cost_entry(op, target);

        // 4 independent MULs on Skylake
        // Each MUL: 1 uop, port 1 only (1 port)
        // Pressure on port 1: 4 * 1.0 = 4.0
        let instrs = vec![
            make_instr(Opcode::Mul, 0, 1, 2),
            make_instr(Opcode::Mul, 3, 4, 5),
            make_instr(Opcode::Mul, 6, 7, 8),
            make_instr(Opcode::Mul, 9, 10, 11),
        ];

        let tp = port_pressure_bound(&instrs, &target, &cost_db);

        // Port 1 gets 4.0 pressure (the bottleneck)
        assert!(
            (tp - 4.0).abs() < 0.01,
            "4 MULs on Skylake should have port pressure of ~4.0, got {tp}"
        );
    }

    // -------------------------------------------------------------------------
    // Test: Full estimate_block_cycles for simple cases
    // -------------------------------------------------------------------------
    #[test]
    fn test_estimate_simple_block() {
        let target = TargetConfig::skylake();

        // Single ADD — should be approximately 1 cycle
        let instrs = vec![make_instr(Opcode::Add, 0, 1, 2)];
        let cost = estimate_block_cycles(&instrs, target);
        assert!(
            cost >= 1.0 && cost < 3.0,
            "Single ADD should cost ~1 cycle, got {cost}"
        );
    }

    // -------------------------------------------------------------------------
    // Test: estimate_block_cycles — DIV dominates
    // -------------------------------------------------------------------------
    #[test]
    fn test_estimate_div_dominates() {
        let target = TargetConfig::skylake();

        let div_instrs = vec![make_instr(Opcode::Div, 0, 1, 2)];
        let add_instrs = vec![make_instr(Opcode::Add, 0, 1, 2)];

        let div_cost = estimate_block_cycles(&div_instrs, target);
        let add_cost = estimate_block_cycles(&add_instrs, target);

        assert!(
            div_cost > add_cost * 10.0,
            "DIV cost ({div_cost}) should be >> ADD cost ({add_cost})"
        );
    }

    // -------------------------------------------------------------------------
    // Test: Zen 4 has different costs than Skylake
    // -------------------------------------------------------------------------
    #[test]
    fn test_target_cost_differences() {
        let skylake = TargetConfig::skylake();
        let zen4 = TargetConfig::zen4();
        let golden_cove = TargetConfig::golden_cove();

        // DIV latency differs across targets
        let skl_div = cost_entry(Opcode::Div, skylake);
        let zn4_div = cost_entry(Opcode::Div, zen4);
        let gnc_div = cost_entry(Opcode::Div, golden_cove);

        assert_ne!(skl_div.latency, zn4_div.latency, "Skylake and Zen4 DIV latencies should differ");
        assert_ne!(skl_div.latency, gnc_div.latency, "Skylake and Golden Cove DIV latencies should differ");

        // Zen4 has better shift throughput (0.25 vs 0.5)
        let skl_shl = cost_entry(Opcode::Shl, skylake);
        let zn4_shl = cost_entry(Opcode::Shl, zen4);
        assert!(
            zn4_shl.throughput < skl_shl.throughput,
            "Zen4 shift throughput ({}) should be better than Skylake ({})",
            zn4_shl.throughput,
            skl_shl.throughput,
        );
    }

    // -------------------------------------------------------------------------
    // Test: NOP has zero latency and doesn't affect critical path
    // -------------------------------------------------------------------------
    #[test]
    fn test_nop_zero_latency() {
        let target = TargetConfig::skylake();

        let nop = cost_entry(Opcode::Nop, target);
        assert_eq!(nop.latency, 0);
        assert_eq!(nop.num_uops, 1); // still consumes a uop slot

        // NOP + ADD (no dependency since NOP doesn't write to a register)
        let instrs = vec![
            MachineInstr {
                opcode: Opcode::Nop,
                dst: 0,
                src1: 0,
                src2: 0,
                imm: 0,
                has_imm: false,
            },
            make_instr(Opcode::Add, 1, 2, 3),
        ];

        let cost = estimate_block_cycles(&instrs, target);
        // The ADD latency is 1; NOP doesn't add to critical path.
        // Frontend: 2 uops / 4 decode = 0.5
        // So the result should be max(1.0, ...) = 1.0
        assert!(
            cost >= 1.0,
            "NOP + independent ADD should cost at least 1 cycle, got {cost}"
        );
    }

    // -------------------------------------------------------------------------
    // Test: Register pressure with uses that kill registers
    // -------------------------------------------------------------------------
    #[test]
    fn test_register_pressure_with_uses() {
        // LoadConst r0, 0   ← r0 live from 0
        // LoadConst r1, 0   ← r1 live from 1
        // ADD r2, r0, r1    ← r2 live from 2; r0 last used at 2, r1 last used at 2
        // After instruction 2, r0 and r1 are dead (no more uses), only r2 is live.
        let instrs = vec![
            make_instr_imm(Opcode::LoadConst, 0, 0, 0),
            make_instr_imm(Opcode::LoadConst, 1, 0, 0),
            make_instr(Opcode::Add, 2, 0, 1),
        ];

        let pressure = register_pressure(&instrs);

        // At instruction 0: r0 live → 1
        // At instruction 1: r0, r1 live → 2
        // At instruction 2: r0, r1, r2 live → 3 (r0 and r1 are still live at 2 because they're used there)
        assert_eq!(pressure, 3, "Expected max 3 live registers");
    }

    // -------------------------------------------------------------------------
    // Test: Macro-fusion with implicit block-ending branch
    // -------------------------------------------------------------------------
    #[test]
    fn test_macro_fusion_end_of_block() {
        let target = TargetConfig::skylake();
        let cost_db = |op: Opcode| cost_entry(op, target);

        // Block ending with CMP — should detect fusion with implicit branch
        let instrs = vec![
            make_instr(Opcode::Add, 0, 1, 2),
            make_instr(Opcode::Cmp, 3, 0, 4),
        ];

        let fusions = detect_macro_fusions(&instrs, &cost_db);
        assert!(
            !fusions.is_empty(),
            "CMP at end of block should fuse with implicit branch"
        );
        assert_eq!(fusions[0].0, 1, "Fusion should involve instruction index 1 (CMP)");
        assert_eq!(fusions[0].1, usize::MAX, "Branch index should be sentinel");

        // Block ending with ADD — should NOT fuse
        let instrs_no_fuse = vec![
            make_instr(Opcode::Add, 0, 1, 2),
            make_instr(Opcode::Add, 3, 0, 4),
        ];

        let fusions_no = detect_macro_fusions(&instrs_no_fuse, &cost_db);
        assert!(
            fusions_no.is_empty(),
            "Block ending with ADD should have no macro-fusions"
        );
    }

    // -------------------------------------------------------------------------
    // Test: estimate_block_cycles gives consistent results
    // -------------------------------------------------------------------------
    #[test]
    fn test_estimate_consistency() {
        let target = TargetConfig::skylake();

        // Same instruction sequence should give the same cost every time
        let instrs = vec![
            make_instr(Opcode::Add, 0, 1, 2),
            make_instr(Opcode::Add, 3, 0, 4),
            make_instr(Opcode::Mul, 5, 3, 6),
        ];

        let cost1 = estimate_block_cycles(&instrs, target);
        let cost2 = estimate_block_cycles(&instrs, target);

        assert!(
            (cost1 - cost2).abs() < 0.001,
            "Repeated estimates should be identical: {cost1} vs {cost2}"
        );
    }

    // -------------------------------------------------------------------------
    // Test: Empty block has zero cost
    // -------------------------------------------------------------------------
    #[test]
    fn test_empty_block() {
        let target = TargetConfig::skylake();
        let cost = estimate_block_cycles(&[], target);
        assert_eq!(cost, 0.0, "Empty block should have zero cost");
    }

    // -------------------------------------------------------------------------
    // Test: DepDag with WAR dependency
    // -------------------------------------------------------------------------
    #[test]
    fn test_dag_war_dependency() {
        // ADD r0, r1, r2   (reads r1, writes r0)
        // ADD r1, r3, r4   (writes r1 — WAR with first ADD's read of r1)
        let instrs = vec![
            make_instr(Opcode::Add, 0, 1, 2),
            make_instr(Opcode::Add, 1, 3, 4),
        ];

        let dag = build_dependency_dag(&instrs);

        // Second instruction should depend on the first (WAR on r1)
        assert!(
            dag.predecessors[1].contains(&0),
            "Second ADD should have WAR dependency on first ADD (r1)"
        );
    }

    // -------------------------------------------------------------------------
    // Test: Detect function returns a valid target
    // -------------------------------------------------------------------------
    #[test]
    fn test_detect_target() {
        let target = TargetConfig::detect();
        // Should return one of the known targets
        assert!(
            target.name == "skylake" || target.name == "golden_cove" || target.name == "zen4",
            "Detected target should be a known uarch, got '{}'",
            target.name,
        );
    }

    // -------------------------------------------------------------------------
    // Test: Cross-target estimation — Zen4 DIV is cheaper than Skylake DIV
    // -------------------------------------------------------------------------
    #[test]
    fn test_cross_target_div_cost() {
        let skylake = TargetConfig::skylake();
        let zen4 = TargetConfig::zen4();

        let div_instrs = vec![make_instr(Opcode::Div, 0, 1, 2)];

        let skl_cost = estimate_block_cycles(&div_instrs, skylake);
        let zn4_cost = estimate_block_cycles(&div_instrs, zen4);

        assert!(
            zn4_cost < skl_cost,
            "Zen4 DIV ({zn4_cost}) should be cheaper than Skylake DIV ({skl_cost})"
        );
    }

    // -------------------------------------------------------------------------
    // Test: LoadConst doesn't create false source dependencies
    // -------------------------------------------------------------------------
    #[test]
    fn test_loadconst_no_src_dependency() {
        // LoadConst r0, 42   (writes r0, no src dependency)
        // LoadConst r1, 42   (writes r1, no src dependency)
        let instrs = vec![
            make_instr_imm(Opcode::LoadConst, 0, 0, 42),
            make_instr_imm(Opcode::LoadConst, 1, 0, 42),
        ];

        let dag = build_dependency_dag(&instrs);

        // No dependencies between the two LoadConsts
        assert!(
            dag.predecessors[0].is_empty(),
            "First LoadConst should have no predecessors"
        );
        assert!(
            dag.predecessors[1].is_empty(),
            "Second LoadConst should have no predecessors"
        );
    }

    // -------------------------------------------------------------------------
    // Test: Immediate operand doesn't create src2 dependency
    // -------------------------------------------------------------------------
    #[test]
    fn test_immediate_no_src2_dependency() {
        // ADD r0, r1, imm:42  (has_imm=true, src2 is not a register)
        // ADD r3, r0, r4       (depends on r0 via RAW)
        let instrs = vec![
            MachineInstr {
                opcode: Opcode::Add,
                dst: 0,
                src1: 1,
                src2: 99, // would be a register if has_imm=false
                imm: 42,
                has_imm: true,
            },
            make_instr(Opcode::Add, 3, 0, 4),
        ];

        let dag = build_dependency_dag(&instrs);

        // Second ADD depends on first (RAW on r0)
        assert!(dag.predecessors[1].contains(&0));

        // First ADD should NOT depend on the definition of virtual reg 99
        // (because has_imm=true means src2 is not a register)
        assert!(
            dag.predecessors[0].is_empty(),
            "First ADD with immediate should have no predecessors (no reg 99 def)"
        );
    }

    // -------------------------------------------------------------------------
    // Test: Long dependent chain of mixed operations
    // -------------------------------------------------------------------------
    #[test]
    fn test_long_mixed_chain() {
        let target = TargetConfig::skylake();
        let cost_db = |op: Opcode| cost_entry(op, target);

        // r0 = ADD r1, r2          (lat 1)
        // r3 = MUL r0, r4          (lat 3, RAW on r0)
        // r5 = SUB r3, r6          (lat 1, RAW on r3)
        // r7 = DIV r5, r8          (lat 35, RAW on r5)
        let instrs = vec![
            make_instr(Opcode::Add, 0, 1, 2),
            make_instr(Opcode::Mul, 3, 0, 4),
            make_instr(Opcode::Sub, 5, 3, 6),
            make_instr(Opcode::Div, 7, 5, 8),
        ];

        let dag = build_dependency_dag(&instrs);
        let cp = critical_path_length(&dag, &cost_db);

        // Expected: 1 + 3 + 1 + 35 = 40 cycles
        assert_eq!(cp, 40.0, "Chain ADD→MUL→SUB→DIV should have critical path of 40");
    }

    // -------------------------------------------------------------------------
    // Test: Register pressure — no false pressure from NOP
    // -------------------------------------------------------------------------
    #[test]
    fn test_nop_no_register_pressure() {
        let instrs = vec![MachineInstr {
            opcode: Opcode::Nop,
            dst: 0,
            src1: 0,
            src2: 0,
            imm: 0,
            has_imm: false,
        }];

        let pressure = register_pressure(&instrs);
        assert_eq!(pressure, 0, "NOP should contribute zero register pressure");
    }
}
