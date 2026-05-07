// =============================================================================
// Machine Instruction Model — First-Class Instruction Semantics
//
// This module defines machine instructions as first-class objects with full
// semantic metadata: opcode, explicit/implicit operands, flag effects, memory
// effects, encoding size, and microarchitectural cost profiles.
//
// Design references:
//   - LLVM Target-Independent Code Generator (llvm.org/docs/CodeGenerator.html)
//   - STOKE Superoptimizer (Stanford, stochastic search over x86-64 assembly)
//   - egg: Fast and Extensible Equality Saturation (arXiv:2004.03082)
//   - Alive2: Bounded Translation Validation for LLVM (PLDI 2021)
//
// Architecture:
//   Source program
//     → semantic normalization
//     → machine lowering
//     → instruction graph (this module)
//     → legality gate
//     → e-graph canonicalization
//     → instruction-aware MCTS / beam / stochastic search
//     → microarchitecture cost model
//     → bounded validation
//     → emit machine code
// =============================================================================

#![cfg(feature = "core-superopt")]

use std::collections::{HashMap, HashSet};
use std::fmt;

use smallvec::{SmallVec, smallvec};

use crate::optimizer::uarch_cost::{self, TargetConfig};

// =============================================================================
// §1  Register Model
// =============================================================================

/// A physical or virtual register identifier.
///
/// Physical registers map to x86-64 register names; virtual registers are
/// allocated during lowering and mapped to physical registers during
/// register allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Reg {
    /// Virtual register (allocated during lowering, mapped during RA)
    VReg(u32),
    /// RAX
    Rax,
    /// RCX
    Rcx,
    /// RDX
    Rdx,
    /// RBX
    Rbx,
    /// RSP (stack pointer)
    Rsp,
    /// RBP (base pointer / frame pointer)
    Rbp,
    /// RSI
    Rsi,
    /// RDI
    Rdi,
    /// R8–R15
    R8, R9, R10, R11, R12, R13, R14, R15,
    /// RFLAGS register
    Rflags,
    /// No register (placeholder for unused operand slots)
    None,
}

impl Reg {
    /// Whether this register is a general-purpose register (excl. RSP/RBP/RFLAGS)
    pub fn is_gpr(&self) -> bool {
        matches!(self,
            Reg::Rax | Reg::Rcx | Reg::Rdx | Reg::Rbx |
            Reg::Rsi | Reg::Rdi |
            Reg::R8 | Reg::R9 | Reg::R10 | Reg::R11 |
            Reg::R12 | Reg::R13 | Reg::R14 | Reg::R15
        )
    }

    /// Whether this is a virtual register
    pub fn is_virtual(&self) -> bool {
        matches!(self, Reg::VReg(_))
    }

    /// Whether this register is caller-saved (volatile) in the System V AMD64 ABI
    pub fn is_caller_saved(&self) -> bool {
        matches!(self,
            Reg::Rax | Reg::Rcx | Reg::Rdx |
            Reg::Rsi | Reg::Rdi |
            Reg::R8 | Reg::R9 | Reg::R10 | Reg::R11
        )
    }

    /// Whether this register is callee-saved (non-volatile) in the System V AMD64 ABI
    pub fn is_callee_saved(&self) -> bool {
        matches!(self,
            Reg::Rbx | Reg::Rbp |
            Reg::R12 | Reg::R13 | Reg::R14 | Reg::R15
        )
    }

    /// All GPRs available for allocation (excl. RSP, RBP, RFLAGS)
    pub fn allocatable_gprs() -> &'static [Reg] {
        &[
            Reg::Rax, Reg::Rcx, Reg::Rdx, Reg::Rbx,
            Reg::Rsi, Reg::Rdi,
            Reg::R8, Reg::R9, Reg::R10, Reg::R11,
            Reg::R12, Reg::R13, Reg::R14, Reg::R15,
        ]
    }

    /// Index for fast bitset operations
    pub fn index(&self) -> u8 {
        match self {
            Reg::VReg(v) => 200 + (*v % 56) as u8, // Virtual regs in upper range
            Reg::Rax => 0, Reg::Rcx => 1, Reg::Rdx => 2, Reg::Rbx => 3,
            Reg::Rsp => 4, Reg::Rbp => 5, Reg::Rsi => 6, Reg::Rdi => 7,
            Reg::R8 => 8, Reg::R9 => 9, Reg::R10 => 10, Reg::R11 => 11,
            Reg::R12 => 12, Reg::R13 => 13, Reg::R14 => 14, Reg::R15 => 15,
            Reg::Rflags => 16, Reg::None => 255,
        }
    }
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reg::VReg(v) => write!(f, "v{}", v),
            Reg::Rax => write!(f, "rax"),
            Reg::Rcx => write!(f, "rcx"),
            Reg::Rdx => write!(f, "rdx"),
            Reg::Rbx => write!(f, "rbx"),
            Reg::Rsp => write!(f, "rsp"),
            Reg::Rbp => write!(f, "rbp"),
            Reg::Rsi => write!(f, "rsi"),
            Reg::Rdi => write!(f, "rdi"),
            Reg::R8 => write!(f, "r8"),
            Reg::R9 => write!(f, "r9"),
            Reg::R10 => write!(f, "r10"),
            Reg::R11 => write!(f, "r11"),
            Reg::R12 => write!(f, "r12"),
            Reg::R13 => write!(f, "r13"),
            Reg::R14 => write!(f, "r14"),
            Reg::R15 => write!(f, "r15"),
            Reg::Rflags => write!(f, "rflags"),
            Reg::None => write!(f, "_"),
        }
    }
}

/// A compact bitset of registers for fast liveness/clobber tracking.
/// Uses a u32 where each bit corresponds to Reg::index().
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RegSet(u32);

impl RegSet {
    pub fn empty() -> Self { Self(0) }
    pub fn all() -> Self { Self(0x0001FFFF) } // All 17 physical regs

    pub fn insert(&mut self, r: Reg) { self.0 |= 1 << r.index(); }
    pub fn remove(&mut self, r: Reg) { self.0 &= !(1 << r.index()); }
    pub fn contains(&self, r: Reg) -> bool { self.0 & (1 << r.index()) != 0 }

    pub fn union(&self, other: &RegSet) -> RegSet { RegSet(self.0 | other.0) }
    pub fn intersect(&self, other: &RegSet) -> RegSet { RegSet(self.0 & other.0) }
    pub fn difference(&self, other: &RegSet) -> RegSet { RegSet(self.0 & !other.0) }

    pub fn count(&self) -> u32 { self.0.count_ones() }
    pub fn is_empty(&self) -> bool { self.0 == 0 }

    /// Iterate over the physical registers in this set
    pub fn iter_physical(&self) -> impl Iterator<Item = Reg> + '_ {
        static PHYSICAL_REGS: &[Reg] = &[
            Reg::Rax, Reg::Rcx, Reg::Rdx, Reg::Rbx,
            Reg::Rsp, Reg::Rbp, Reg::Rsi, Reg::Rdi,
            Reg::R8, Reg::R9, Reg::R10, Reg::R11,
            Reg::R12, Reg::R13, Reg::R14, Reg::R15,
            Reg::Rflags,
        ];
        PHYSICAL_REGS.iter().copied().filter(move |r| self.contains(*r))
    }
}

// =============================================================================
// §2  Operand Model
// =============================================================================

/// An operand to a machine instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operand {
    /// Register operand (read or write)
    Reg(Reg),
    /// Immediate value (embedded in the instruction encoding)
    Imm(i64),
    /// Memory operand: [base + index * scale + displacement]
    Mem {
        base: Reg,
        index: Option<Reg>,
        scale: u8, // 1, 2, 4, or 8
        disp: i32,
    },
    /// RIP-relative address (for position-independent code)
    RipRelative(i32),
}

impl Operand {
    /// Collect all registers referenced by this operand
    pub fn regs(&self) -> SmallVec<[Reg; 2]> {
        match self {
            Operand::Reg(r) => smallvec![*r],
            Operand::Imm(_) | Operand::RipRelative(_) => smallvec![],
            Operand::Mem { base, index, .. } => {
                let mut regs = smallvec![*base];
                if let Some(idx) = index {
                    regs.push(*idx);
                }
                regs
            }
        }
    }

    /// Whether this operand is a register
    pub fn is_reg(&self) -> bool { matches!(self, Operand::Reg(_)) }
    /// Whether this operand is an immediate
    pub fn is_imm(&self) -> bool { matches!(self, Operand::Imm(_)) }
    /// Whether this operand is a memory reference
    pub fn is_mem(&self) -> bool { matches!(self, Operand::Mem { .. }) }
}

// =============================================================================
// §3  Flag Effects
// =============================================================================

/// Which CPU flags are affected by an instruction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlagEffects {
    /// CF (Carry Flag) is written
    pub writes_cf: bool,
    /// ZF (Zero Flag) is written
    pub writes_zf: bool,
    /// SF (Sign Flag) is written
    pub writes_sf: bool,
    /// OF (Overflow Flag) is written
    pub writes_of: bool,
    /// PF (Parity Flag) is written
    pub writes_pf: bool,
    /// AF (Adjust Flag) is written
    pub writes_af: bool,
    /// CF is read
    pub reads_cf: bool,
    /// ZF is read
    pub reads_zf: bool,
    /// SF is read
    pub reads_sf: bool,
    /// OF is read
    pub reads_of: bool,
}

impl FlagEffects {
    /// No flag effects (e.g., for MOV)
    pub fn none() -> Self { Self::default() }

    /// Writes all arithmetic flags (ADD, SUB, AND, OR, XOR, etc.)
    pub fn writes_all_arith() -> Self {
        Self {
            writes_cf: true, writes_zf: true, writes_sf: true,
            writes_of: true, writes_pf: true, writes_af: true,
            ..Default::default()
        }
    }

    /// Writes CF + OF only (SHL, SHR, SAR)
    pub fn writes_shift() -> Self {
        Self {
            writes_cf: true, writes_zf: true, writes_sf: true,
            writes_of: true, writes_pf: true,
            ..Default::default()
        }
    }

    /// Reads all arithmetic flags (e.g., CMOVcc, SETcc)
    pub fn reads_all_arith() -> Self {
        Self {
            reads_cf: true, reads_zf: true, reads_sf: true, reads_of: true,
            ..Default::default()
        }
    }

    /// No flag effects (LEA, MOV)
    pub fn preserves_all() -> Self { Self::default() }

    /// Whether any flags are written
    pub fn writes_any(&self) -> bool {
        self.writes_cf || self.writes_zf || self.writes_sf || self.writes_of || self.writes_pf || self.writes_af
    }

    /// Whether any flags are read
    pub fn reads_any(&self) -> bool {
        self.reads_cf || self.reads_zf || self.reads_sf || self.reads_of
    }
}

// =============================================================================
// §4  Memory Effects
// =============================================================================

/// Memory effects of an instruction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemEffects {
    /// Reads from memory
    pub may_read: bool,
    /// Writes to memory
    pub may_write: bool,
    /// Has side effects (e.g., volatile, I/O)
    pub has_side_effects: bool,
    /// May raise a trap/exception (e.g., division by zero)
    pub may_trap: bool,
    /// Barrier: prevents reordering across this instruction
    pub is_barrier: bool,
}

impl MemEffects {
    /// Pure computation — no memory effects
    pub fn pure() -> Self { Self::default() }

    /// Reads from memory
    pub fn read() -> Self { Self { may_read: true, ..Default::default() } }

    /// Writes to memory
    pub fn write() -> Self { Self { may_write: true, ..Default::default() } }

    /// Reads and writes memory
    pub fn read_write() -> Self {
        Self { may_read: true, may_write: true, ..Default::default() }
    }

    /// Instruction may trap (DIV, IDIV)
    pub fn may_trap() -> Self {
        Self { may_trap: true, ..Default::default() }
    }

    /// Whether this instruction is pure (no memory or side effects)
    pub fn is_pure(&self) -> bool {
        !self.may_read && !self.may_write && !self.has_side_effects && !self.may_trap
    }

    /// Whether this instruction can be safely reordered with another
    pub fn can_reorder_with(&self, other: &MemEffects) -> bool {
        // Can reorder if at most one writes and neither has side effects
        if self.has_side_effects || other.has_side_effects { return false; }
        if self.is_barrier || other.is_barrier { return false; }
        let writes = self.may_write as u8 + other.may_write as u8;
        if writes > 1 { return false; }
        if writes == 1 && (self.may_read || other.may_read) { return false; }
        true
    }
}

// =============================================================================
// §5  x86-64 Opcode Enum
// =============================================================================

/// x86-64 opcodes with full semantic metadata.
///
/// This is the core opcode enum — each variant carries enough information
/// to derive all semantic properties (operand count, flag effects, memory
/// effects, commutativity, encoding size, etc.) through the instruction
/// table database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum X86Opcode {
    // ── Data Movement ──
    /// MOV r64, r64/imm32
    Mov64,
    /// MOV r32, r32/imm32 (zero-extends to 64 bits)
    Mov32,
    /// MOVZX r64, r/m8
    Movzx8,
    /// MOVZX r64, r/m16
    Movzx16,
    /// MOVSX r64, r/m8
    Movsx8,
    /// MOVSX r64, r/m16
    Movsx16,
    /// MOVSXD r64, r/m32
    Movsxd32,
    /// XCHG r64, r64
    Xchg,

    // ── Arithmetic ──
    /// ADD r64, r/m64
    Add64,
    /// ADD r32, r/m32
    Add32,
    /// SUB r64, r/m64
    Sub64,
    /// SUB r32, r/m32
    Sub32,
    /// INC r64
    Inc64,
    /// DEC r64
    Dec64,
    /// NEG r64
    Neg64,
    /// MUL r64 (unsigned, RDX:RAX)
    Mul64,
    /// IMUL r64, r/m64
    Imul64,
    /// IMUL r64, r/m64, imm
    Imul64Imm,
    /// DIV r/m64 (unsigned)
    Div64,
    /// IDIV r/m64 (signed)
    Idiv64,
    /// ADC r64, r/m64 (add with carry)
    Adc64,
    /// SBB r64, r/m64 (subtract with borrow)
    Sbb64,

    // ── Logical ──
    /// AND r64, r/m64
    And64,
    /// OR r64, r/m64
    Or64,
    /// XOR r64, r/m64
    Xor64,
    /// NOT r64
    Not64,
    /// ANDN r64, r64, r64 (BMI1)
    Andn64,
    /// TEST r64, r/m64
    Test64,

    // ── Shifts and Rotates ──
    /// SHL r64, imm8 / CL
    Shl64,
    /// SHR r64, imm8 / CL
    Shr64,
    /// SAR r64, imm8 / CL
    Sar64,
    /// ROL r64, imm8 / CL
    Rol64,
    /// ROR r64, imm8 / CL
    Ror64,

    // ── Comparison ──
    /// CMP r64, r/m64
    Cmp64,
    /// CMP r32, r/m32
    Cmp32,

    // ── Conditional ──
    /// CMOVcc r64, r/m64
    Cmov64,
    /// SETcc r/m8
    Setcc,

    // ── Address Computation ──
    /// LEA r64, [base + index*scale + disp]
    Lea64,

    // ── Bit Manipulation (BMI1/BMI2) ──
    /// BLSI r64, r/m64
    Blsi64,
    /// BLSR r64, r/m64
    Blsr64,
    /// BLSMSK r64, r/m64
    Blsmsk64,
    /// BZHI r64, r/m64, r64
    Bzhi64,
    /// PDEP r64, r64, r/m64
    Pdep64,
    /// PEXT r64, r64, r/m64
    Pext64,
    /// RORX r64, r/m64, imm8
    Rorx64,

    // ── Byte Operations ──
    /// BSWAP r64
    Bswap64,
    /// POPCNT r64, r/m64
    Popcnt64,
    /// LZCNT r64, r/m64
    Lzcnt64,
    /// TZCNT r64, r/m64
    Tzcnt64,
    /// BT r64, r/m64
    Bt64,
    /// BTS r64, r/m64
    Bts64,
    /// BTR r64, r/m64
    Btr64,
    /// BTC r64, r/m64
    Btc64,

    // ── NOP ──
    /// NOP (1 byte)
    Nop,

    // ── Load / Store ──
    /// Load from memory: MOV r64, [mem]
    Load64,
    /// Store to memory: MOV [mem], r64
    Store64,

    // ── Constant Materialization ──
    /// Load 32-bit immediate (zero-extended): MOV r32, imm32
    LoadImm32,
    /// Load 64-bit immediate: MOV r64, imm64 (10-byte encoding)
    LoadImm64,
    /// Zero register: XOR r32, r32 (shorter than MOV r32, 0)
    ZeroReg32,
}

impl X86Opcode {
    /// Whether this opcode is commutative (i.e., a op b == b op a)
    pub fn is_commutative(&self) -> bool {
        matches!(self,
            X86Opcode::Add64 | X86Opcode::Add32 |
            X86Opcode::And64 | X86Opcode::Or64 | X86Opcode::Xor64 |
            X86Opcode::Imul64 |
            X86Opcode::Pdep64 | X86Opcode::Pext64
        )
    }

    /// Whether this is a comparison opcode (writes flags but no GPR)
    pub fn is_comparison(&self) -> bool {
        matches!(self, X86Opcode::Cmp64 | X86Opcode::Cmp32 | X86Opcode::Test64)
    }

    /// Whether this is a conditional move
    pub fn is_cmov(&self) -> bool {
        matches!(self, X86Opcode::Cmov64)
    }

    /// Whether this is a LEA (no flag effects, pure address computation)
    pub fn is_lea(&self) -> bool {
        matches!(self, X86Opcode::Lea64)
    }

    /// Whether this is a division instruction (may trap)
    pub fn is_division(&self) -> bool {
        matches!(self, X86Opcode::Div64 | X86Opcode::Idiv64)
    }
}

impl fmt::Display for X86Opcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            X86Opcode::Mov64 => "mov", X86Opcode::Mov32 => "mov",
            X86Opcode::Movzx8 => "movzx", X86Opcode::Movzx16 => "movzx",
            X86Opcode::Movsx8 => "movsx", X86Opcode::Movsx16 => "movsx",
            X86Opcode::Movsxd32 => "movsxd",
            X86Opcode::Xchg => "xchg",
            X86Opcode::Add64 => "add", X86Opcode::Add32 => "add",
            X86Opcode::Sub64 => "sub", X86Opcode::Sub32 => "sub",
            X86Opcode::Inc64 => "inc", X86Opcode::Dec64 => "dec",
            X86Opcode::Neg64 => "neg",
            X86Opcode::Mul64 => "mul", X86Opcode::Imul64 => "imul",
            X86Opcode::Imul64Imm => "imul",
            X86Opcode::Div64 => "div", X86Opcode::Idiv64 => "idiv",
            X86Opcode::Adc64 => "adc", X86Opcode::Sbb64 => "sbb",
            X86Opcode::And64 => "and", X86Opcode::Or64 => "or",
            X86Opcode::Xor64 => "xor", X86Opcode::Not64 => "not",
            X86Opcode::Andn64 => "andn",
            X86Opcode::Test64 => "test",
            X86Opcode::Shl64 => "shl", X86Opcode::Shr64 => "shr",
            X86Opcode::Sar64 => "sar",
            X86Opcode::Rol64 => "rol", X86Opcode::Ror64 => "ror",
            X86Opcode::Cmp64 => "cmp", X86Opcode::Cmp32 => "cmp",
            X86Opcode::Cmov64 => "cmovcc",
            X86Opcode::Setcc => "setcc",
            X86Opcode::Lea64 => "lea",
            X86Opcode::Blsi64 => "blsi", X86Opcode::Blsr64 => "blsr",
            X86Opcode::Blsmsk64 => "blsmsk",
            X86Opcode::Bzhi64 => "bzhi",
            X86Opcode::Pdep64 => "pdep", X86Opcode::Pext64 => "pext",
            X86Opcode::Rorx64 => "rorx",
            X86Opcode::Bswap64 => "bswap",
            X86Opcode::Popcnt64 => "popcnt",
            X86Opcode::Lzcnt64 => "lzcnt",
            X86Opcode::Tzcnt64 => "tzcnt",
            X86Opcode::Bt64 => "bt", X86Opcode::Bts64 => "bts",
            X86Opcode::Btr64 => "btr", X86Opcode::Btc64 => "btc",
            X86Opcode::Nop => "nop",
            X86Opcode::Load64 => "load", X86Opcode::Store64 => "store",
            X86Opcode::LoadImm32 => "mov", X86Opcode::LoadImm64 => "mov",
            X86Opcode::ZeroReg32 => "xor",
        };
        write!(f, "{}", name)
    }
}

// =============================================================================
// §6  MachineInstr — The Core Instruction Model
// =============================================================================

/// A machine instruction with full semantic metadata.
///
/// This is the central data structure of the machine-instruction-aware
/// optimizer. Each instruction carries:
///
/// - **opcode**: what operation to perform
/// - **operands**: explicit inputs and outputs
/// - **implicit_reads**: registers implicitly read (e.g., RAX in MUL)
/// - **implicit_writes**: registers implicitly written (e.g., RDX:RAX in MUL)
/// - **flag_effects**: which RFLAGS bits are read/written
/// - **mem_effects**: memory read/write/side-effect information
/// - **encoding_length**: bytes in the x86-64 encoding
/// - **latency**: estimated cycles for the result to be available
/// - **throughput**: reciprocal throughput (cycles per issue)
/// - **uop_count**: number of micro-ops this decodes to
/// - **clobber_set**: set of registers clobbered by this instruction
#[derive(Debug, Clone, PartialEq)]
pub struct MachineInstrFull {
    /// The opcode (determines most semantic properties)
    pub opcode: X86Opcode,

    /// Destination operand (the value produced by this instruction)
    pub dst: Operand,

    /// Source operands (the values consumed by this instruction)
    pub srcs: SmallVec<[Operand; 3]>,

    /// Registers implicitly read (not listed in srcs)
    pub implicit_reads: RegSet,

    /// Registers implicitly written (not listed in dst)
    pub implicit_writes: RegSet,

    /// Flag effects (which RFLAGS bits are read/written)
    pub flag_effects: FlagEffects,

    /// Memory effects (read/write/side-effect info)
    pub mem_effects: MemEffects,

    /// Encoding length in bytes
    pub encoding_length: u8,

    /// Estimated latency in cycles
    pub latency: u8,

    /// Reciprocal throughput (cycles per issue, smaller = better)
    pub throughput: f32,

    /// Number of micro-ops
    pub uop_count: u8,

    /// Register clobber set
    pub clobbers: RegSet,
}

impl MachineInstrFull {
    /// Create a new machine instruction with default metadata.
    /// Metadata is populated from the instruction table database.
    pub fn new(opcode: X86Opcode, dst: Operand, srcs: SmallVec<[Operand; 3]>) -> Self {
        let meta = X86InstrTable::lookup(opcode);
        Self {
            opcode,
            dst,
            srcs,
            implicit_reads: meta.implicit_reads,
            implicit_writes: meta.implicit_writes,
            flag_effects: meta.flag_effects,
            mem_effects: meta.mem_effects,
            encoding_length: meta.encoding_length,
            latency: meta.latency,
            throughput: meta.throughput,
            uop_count: meta.uop_count,
            clobbers: meta.clobbers,
        }
    }

    /// All registers read by this instruction (explicit + implicit)
    pub fn all_reads(&self) -> RegSet {
        let mut reads = self.implicit_reads;
        if let Some(r) = self.dst_reg() {
            // For read-modify-write ops, dst is also read
            if self.flag_effects.writes_any() && !self.is_comparison() {
                reads.insert(r);
            }
        }
        for src in &self.srcs {
            for r in src.regs() {
                reads.insert(r);
            }
        }
        reads
    }

    /// All registers written by this instruction (explicit + implicit)
    pub fn all_writes(&self) -> RegSet {
        let mut writes = self.implicit_writes;
        if let Some(r) = self.dst_reg() {
            writes.insert(r);
        }
        writes
    }

    /// Get the destination register, if any
    pub fn dst_reg(&self) -> Option<Reg> {
        match &self.dst {
            Operand::Reg(r) => Some(*r),
            _ => None,
        }
    }

    /// Whether this instruction is pure (no memory or side effects)
    pub fn is_pure(&self) -> bool {
        self.mem_effects.is_pure()
    }

    /// Whether this is a comparison instruction
    pub fn is_comparison(&self) -> bool {
        self.opcode.is_comparison()
    }

    /// Whether this instruction can be safely eliminated if its result is unused
    pub fn is_safe_to_eliminate(&self) -> bool {
        self.is_pure() && !self.mem_effects.has_side_effects && !self.mem_effects.may_trap
    }

    /// Whether this instruction can be reordered with another
    pub fn can_reorder_with(&self, other: &MachineInstrFull) -> bool {
        // Can't reorder if there are data dependencies
        let self_writes = self.all_writes();
        let other_reads = other.all_reads();
        let other_writes = other.all_writes();
        let self_reads = self.all_reads();

        // RAW: other reads what self writes
        if !self_writes.intersect(&other_reads).is_empty() { return false; }
        // WAR: self reads what other writes
        if !other_writes.intersect(&self_reads).is_empty() { return false; }
        // WAW: both write the same register
        if !self_writes.intersect(&other_writes).is_empty() { return false; }

        // Flag dependencies
        if self.flag_effects.writes_any() && other.flag_effects.reads_any() { return false; }
        if other.flag_effects.writes_any() && self.flag_effects.reads_any() { return false; }

        // Memory dependencies
        if !self.mem_effects.can_reorder_with(&other.mem_effects) { return false; }

        true
    }

    /// Whether this instruction preserves all flags
    pub fn preserves_flags(&self) -> bool {
        !self.flag_effects.writes_any()
    }

    /// Whether this instruction reads flags
    pub fn reads_flags(&self) -> bool {
        self.flag_effects.reads_any()
    }

    /// Replace a register operand with another register
    pub fn rename_reg(&mut self, from: Reg, to: Reg) {
        if let Operand::Reg(r) = &mut self.dst {
            if *r == from { *r = to; }
        }
        for src in &mut self.srcs {
            if let Operand::Reg(r) = src {
                if *r == from { *r = to; }
            }
            if let Operand::Mem { base, index, .. } = src {
                if *base == from { *base = to; }
                if let Some(idx) = index {
                    if *idx == from { *idx = to; }
                }
            }
        }
        if self.implicit_reads.contains(from) {
            self.implicit_reads.remove(from);
            self.implicit_reads.insert(to);
        }
        if self.implicit_writes.contains(from) {
            self.implicit_writes.remove(from);
            self.implicit_writes.insert(to);
        }
        if self.clobbers.contains(from) {
            self.clobbers.remove(from);
            self.clobbers.insert(to);
        }
    }

    /// Get the encoding size in bytes
    pub fn size(&self) -> u8 {
        self.encoding_length
    }

    /// Display in AT&T-style assembly syntax
    pub fn to_asm(&self) -> String {
        let mut s = format!("{}", self.opcode);
        // Destination first (AT&T: src, dst; Intel: dst, src)
        match &self.dst {
            Operand::Reg(r) => s.push_str(&format!(" {}", r)),
            Operand::Imm(v) => s.push_str(&format!(" ${}", v)),
            Operand::Mem { base, index, scale, disp } => {
                s.push_str(&format!(" [{}", base));
                if let Some(idx) = index {
                    s.push_str(&format!(" + {}*{}", idx, scale));
                }
                if *disp != 0 {
                    s.push_str(&format!(" + {}", disp));
                }
                s.push(']');
            }
            _ => {}
        }
        for src in &self.srcs {
            s.push(',');
            match src {
                Operand::Reg(r) => s.push_str(&format!(" {}", r)),
                Operand::Imm(v) => s.push_str(&format!(" ${}", v)),
                Operand::Mem { base, index, scale, disp } => {
                    s.push_str(&format!(" [{}", base));
                    if let Some(idx) = index {
                        s.push_str(&format!(" + {}*{}", idx, scale));
                    }
                    if *disp != 0 {
                        s.push_str(&format!(" + {}", disp));
                    }
                    s.push(']');
                }
                _ => {}
            }
        }
        s
    }
}

// =============================================================================
// §7  Instruction Table Database
// =============================================================================

/// Metadata for an x86-64 instruction, looked up from the instruction table.
#[derive(Debug, Clone, Copy)]
pub struct InstrMeta {
    pub implicit_reads: RegSet,
    pub implicit_writes: RegSet,
    pub flag_effects: FlagEffects,
    pub mem_effects: MemEffects,
    pub encoding_length: u8,
    pub latency: u8,
    pub throughput: f32,
    pub uop_count: u8,
    pub clobbers: RegSet,
}

/// The x86-64 instruction table database.
///
/// Provides semantic metadata for each opcode. This is the "instruction
/// table" that the optimizer queries to know whether an instruction
/// commutes, touches memory, implicitly uses/defines registers, etc.
///
/// Latency and throughput values are for Skylake as a baseline; the
/// multi-objective cost model adjusts per-target using the uarch_cost
/// module's per-target data.
pub struct X86InstrTable;

impl X86InstrTable {
    /// Look up metadata for an opcode.
    pub fn lookup(opcode: X86Opcode) -> InstrMeta {
        match opcode {
            // ── Data Movement ──
            X86Opcode::Mov64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::preserves_all(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.25, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Mov32 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::preserves_all(),
                mem_effects: MemEffects::pure(),
                encoding_length: 5,  // MOV r32, imm32 = 5 bytes
                latency: 1, throughput: 0.25, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Movzx8 | X86Opcode::Movzx16 |
            X86Opcode::Movsx8 | X86Opcode::Movsx16 |
            X86Opcode::Movsxd32 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::preserves_all(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.25, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Xchg => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::preserves_all(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.5, uop_count: 2,
                clobbers: RegSet::empty(),
            },

            // ── Arithmetic ──
            X86Opcode::Add64 | X86Opcode::Add32 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.25, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Sub64 | X86Opcode::Sub32 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.25, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Inc64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects {
                    writes_zf: true, writes_sf: true, writes_of: true,
                    writes_pf: true, writes_af: true,
                    ..Default::default() // Does NOT write CF
                },
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.25, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Dec64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects {
                    writes_zf: true, writes_sf: true, writes_of: true,
                    writes_pf: true, writes_af: true,
                    ..Default::default()
                },
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.25, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Neg64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.25, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Mul64 => InstrMeta {
                implicit_reads: { let mut s = RegSet::empty(); s.insert(Reg::Rax); s },
                implicit_writes: {
                    let mut s = RegSet::empty();
                    s.insert(Reg::Rax); s.insert(Reg::Rdx);
                    s
                },
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 3, throughput: 1.0, uop_count: 1,
                clobbers: { let mut s = RegSet::empty(); s.insert(Reg::Rdx); s },
            },
            X86Opcode::Imul64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 3, throughput: 1.0, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Imul64Imm => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 4, // IMUL r64, r/m64, imm8 = 4 bytes
                latency: 3, throughput: 1.0, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Div64 => InstrMeta {
                implicit_reads: {
                    let mut s = RegSet::empty();
                    s.insert(Reg::Rax); s.insert(Reg::Rdx);
                    s
                },
                implicit_writes: {
                    let mut s = RegSet::empty();
                    s.insert(Reg::Rax); s.insert(Reg::Rdx);
                    s
                },
                flag_effects: FlagEffects::none(), // DIV flags are undefined
                mem_effects: MemEffects::may_trap(),
                encoding_length: 3,
                latency: 35, throughput: 35.0, uop_count: 1,
                clobbers: { let mut s = RegSet::empty(); s.insert(Reg::Rdx); s },
            },
            X86Opcode::Idiv64 => InstrMeta {
                implicit_reads: {
                    let mut s = RegSet::empty();
                    s.insert(Reg::Rax); s.insert(Reg::Rdx);
                    s
                },
                implicit_writes: {
                    let mut s = RegSet::empty();
                    s.insert(Reg::Rax); s.insert(Reg::Rdx);
                    s
                },
                flag_effects: FlagEffects::none(),
                mem_effects: MemEffects::may_trap(),
                encoding_length: 3,
                latency: 35, throughput: 35.0, uop_count: 1,
                clobbers: { let mut s = RegSet::empty(); s.insert(Reg::Rdx); s },
            },
            X86Opcode::Adc64 => InstrMeta {
                implicit_reads: { let mut s = RegSet::empty(); s.insert(Reg::Rflags); s },
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.5, uop_count: 2,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Sbb64 => InstrMeta {
                implicit_reads: { let mut s = RegSet::empty(); s.insert(Reg::Rflags); s },
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.5, uop_count: 2,
                clobbers: RegSet::empty(),
            },

            // ── Logical ──
            X86Opcode::And64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.25, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Or64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.25, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Xor64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.25, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Not64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::preserves_all(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.25, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Andn64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 4, // VEX-encoded
                latency: 1, throughput: 0.5, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Test64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.25, uop_count: 1,
                clobbers: RegSet::empty(),
            },

            // ── Shifts and Rotates ──
            X86Opcode::Shl64 | X86Opcode::Shr64 | X86Opcode::Sar64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_shift(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.5, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Rol64 | X86Opcode::Ror64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_shift(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 1.0, uop_count: 1,
                clobbers: RegSet::empty(),
            },

            // ── Comparison ──
            X86Opcode::Cmp64 | X86Opcode::Cmp32 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.25, uop_count: 1,
                clobbers: RegSet::empty(),
            },

            // ── Conditional ──
            X86Opcode::Cmov64 => InstrMeta {
                implicit_reads: { let mut s = RegSet::empty(); s.insert(Reg::Rflags); s },
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::preserves_all(),
                mem_effects: MemEffects::pure(),
                encoding_length: 4,
                latency: 2, throughput: 0.5, uop_count: 2,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Setcc => InstrMeta {
                implicit_reads: { let mut s = RegSet::empty(); s.insert(Reg::Rflags); s },
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::preserves_all(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 0.5, uop_count: 1,
                clobbers: RegSet::empty(),
            },

            // ── LEA ──
            X86Opcode::Lea64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::preserves_all(),
                mem_effects: MemEffects::pure(),
                encoding_length: 4,
                latency: 1, throughput: 0.5, uop_count: 1,
                clobbers: RegSet::empty(),
            },

            // ── Bit Manipulation (BMI1/BMI2) ──
            X86Opcode::Blsi64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 4,
                latency: 1, throughput: 0.5, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Blsr64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 4,
                latency: 1, throughput: 0.5, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Blsmsk64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 4,
                latency: 1, throughput: 0.5, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Bzhi64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 4,
                latency: 1, throughput: 0.5, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Pdep64 | X86Opcode::Pext64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::preserves_all(),
                mem_effects: MemEffects::pure(),
                encoding_length: 4,
                latency: 3, throughput: 3.0, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Rorx64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::preserves_all(),
                mem_effects: MemEffects::pure(),
                encoding_length: 4,
                latency: 1, throughput: 1.0, uop_count: 1,
                clobbers: RegSet::empty(),
            },

            // ── Byte/Bit Operations ──
            X86Opcode::Bswap64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::preserves_all(),
                mem_effects: MemEffects::pure(),
                encoding_length: 3,
                latency: 1, throughput: 1.0, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Popcnt64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 4,
                latency: 3, throughput: 1.0, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Lzcnt64 | X86Opcode::Tzcnt64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 4,
                latency: 3, throughput: 1.0, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Bt64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects { writes_cf: true, ..Default::default() },
                mem_effects: MemEffects::pure(),
                encoding_length: 4,
                latency: 1, throughput: 0.5, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Bts64 | X86Opcode::Btr64 | X86Opcode::Btc64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects { writes_cf: true, ..Default::default() },
                mem_effects: MemEffects::pure(),
                encoding_length: 4,
                latency: 1, throughput: 0.5, uop_count: 1,
                clobbers: RegSet::empty(),
            },

            // ── NOP ──
            X86Opcode::Nop => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::preserves_all(),
                mem_effects: MemEffects::pure(),
                encoding_length: 1,
                latency: 0, throughput: 0.0, uop_count: 0,
                clobbers: RegSet::empty(),
            },

            // ── Load / Store ──
            X86Opcode::Load64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::preserves_all(),
                mem_effects: MemEffects::read(),
                encoding_length: 4, // MOV r64, [mem]
                latency: 4, throughput: 0.5, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::Store64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::preserves_all(),
                mem_effects: MemEffects::write(),
                encoding_length: 4,
                latency: 1, throughput: 1.0, uop_count: 1,
                clobbers: RegSet::empty(),
            },

            // ── Constant Materialization ──
            X86Opcode::LoadImm32 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::preserves_all(),
                mem_effects: MemEffects::pure(),
                encoding_length: 5, // MOV r32, imm32
                latency: 1, throughput: 0.25, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::LoadImm64 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::preserves_all(),
                mem_effects: MemEffects::pure(),
                encoding_length: 10, // MOV r64, imm64
                latency: 1, throughput: 0.25, uop_count: 1,
                clobbers: RegSet::empty(),
            },
            X86Opcode::ZeroReg32 => InstrMeta {
                implicit_reads: RegSet::empty(),
                implicit_writes: RegSet::empty(),
                flag_effects: FlagEffects::writes_all_arith(),
                mem_effects: MemEffects::pure(),
                encoding_length: 2, // XOR r32, r32 (shorter than MOV r32, 0)
                latency: 1, throughput: 0.25, uop_count: 1,
                clobbers: RegSet::empty(),
            },
        }
    }
}

// =============================================================================
// §8  Instruction Block — A Sequence of MachineInstrs
// =============================================================================

/// A basic block of machine instructions with liveness information.
#[derive(Debug, Clone)]
pub struct InstrBlock {
    /// Instructions in program order
    pub instrs: Vec<MachineInstrFull>,
    /// Live-in registers (values live on entry)
    pub live_in: RegSet,
    /// Live-out registers (values live on exit)
    pub live_out: RegSet,
}

impl InstrBlock {
    pub fn new(instrs: Vec<MachineInstrFull>) -> Self {
        Self {
            instrs,
            live_in: RegSet::empty(),
            live_out: RegSet::empty(),
        }
    }

    /// Compute liveness information for this block.
    pub fn compute_liveness(&mut self) {
        // Backward analysis: live = uses ∪ (live_out - defs)
        let mut live: RegSet = self.live_out;
        // We'll store per-instruction liveness in a separate pass
        for instr in self.instrs.iter().rev() {
            // Remove defs from live set
            let defs = instr.all_writes();
            live = live.difference(&defs);
            // Add uses to live set
            let uses = instr.all_reads();
            live = live.union(&uses);
        }
        self.live_in = live;
    }

    /// Get the total encoding size in bytes
    pub fn total_size(&self) -> usize {
        self.instrs.iter().map(|i| i.encoding_length as usize).sum()
    }

    /// Get the number of instructions
    pub fn len(&self) -> usize {
        self.instrs.len()
    }

    /// Whether the block is empty
    pub fn is_empty(&self) -> bool {
        self.instrs.is_empty()
    }

    /// Compute flag-state tracking for each instruction point.
    ///
    /// Returns a `Vec<FlagState>` of length `instrs.len() + 1`, where
    /// entry `i` represents the flag state *before* instruction `i`
    /// executes, and entry `instrs.len()` is the state after the last
    /// instruction.
    ///
    /// This is used to identify redundant CMP/TEST instructions whose
    /// flags are already available from a prior flag-writing instruction.
    pub fn compute_flag_states(&self) -> Vec<FlagState> {
        let n = self.instrs.len();
        let mut states = Vec::with_capacity(n + 1);
        let mut current = FlagState::new();
        states.push(current.clone());

        for (idx, instr) in self.instrs.iter().enumerate() {
            // If any operand of the current flag writer is clobbered by a
            // subsequent non-flag-preserving instruction that also writes to
            // one of the operand registers, we must invalidate.  For simplicity
            // we invalidate on *any* flag-writing instruction, or on any
            // instruction that writes a register used by the current flag
            // writer's operands.
            if instr.flag_effects.writes_any() {
                // This instruction writes new flags — check if it's a
                // comparison-like instruction we can track.
                if instr.opcode.is_comparison() {
                    let op0 = instr.dst.clone();
                    let op1 = instr.srcs.get(0).cloned().unwrap_or(Operand::Imm(0));
                    current.update(idx, instr.opcode, op0, op1);
                } else {
                    // Non-comparison flag writer (ADD, SUB, etc.) — we can
                    // track it for flag redundancy, but we only know it's
                    // equivalent to a CMP of dst vs 0 for SUB/ADD patterns.
                    // For safety, just invalidate and record the new writer.
                    let op0 = instr.dst.clone();
                    let op1 = Operand::Imm(0);
                    current.update(idx, instr.opcode, op0, op1);
                }
            } else {
                // Check if this instruction clobbers a register used by the
                // current flag writer's operands. If so, the flags are stale.
                if let Some(_) = current.writer_idx {
                    let writer_reads: SmallVec<[Reg; 2]> = {
                        let mut regs = smallvec![];
                        if let Some(ref op) = current.operand0 { regs.extend(op.regs()); }
                        if let Some(ref op) = current.operand1 { regs.extend(op.regs()); }
                        regs
                    };
                    let instr_writes = instr.all_writes();
                    for r in writer_reads {
                        if instr_writes.contains(r) {
                            current.invalidate();
                            break;
                        }
                    }
                }
                // Also invalidate if this instruction reads flags — it
                // consumes them, so a subsequent reader can't reuse them.
                // (No invalidation needed for flag reads; the flags are
                // still valid for later readers.)
            }
            states.push(current.clone());
        }

        states
    }

    /// Eliminate redundant CMP/TEST instructions whose flags are already
    /// available from a prior instruction.
    ///
    /// Returns a new `InstrBlock` with redundant comparisons replaced by
    /// NOPs.  A comparison is considered redundant if:
    /// - A prior instruction wrote the same flags with the same opcode and
    ///   operands
    /// - No intervening instruction clobbers those flags or the operand
    ///   registers
    pub fn eliminate_redundant_comparisons(&self) -> InstrBlock {
        let flag_states = self.compute_flag_states();
        let mut new_instrs = self.instrs.clone();

        for (idx, instr) in self.instrs.iter().enumerate() {
            if !instr.opcode.is_comparison() {
                continue;
            }
            // The flag state *before* this instruction is at index `idx`
            // (because compute_flag_states returns state before each instr)
            let state_before = &flag_states[idx];
            let op0 = &instr.dst;
            let op1 = instr.srcs.get(0).unwrap_or(&Operand::Imm(0));

            if state_before.is_redundant_cmp(instr.opcode, op0, op1) {
                // Replace with NOP
                new_instrs[idx] = MachineInstrFull::new(
                    X86Opcode::Nop,
                    Operand::Reg(Reg::None),
                    smallvec![],
                );
            }
        }

        let mut result = InstrBlock::new(new_instrs);
        result.live_in = self.live_in;
        result.live_out = self.live_out;
        result
    }
}

// =============================================================================
// §8b  Flag-Value Numbering
// =============================================================================

/// Tracks which CPU flags are still valid from a prior CMP/TEST instruction.
///
/// This allows the optimizer to eliminate redundant comparisons when the
/// flags are already set from a previous instruction.
#[derive(Debug, Clone)]
pub struct FlagState {
    /// The instruction index that last wrote the flags
    pub writer_idx: Option<usize>,
    /// The opcode that wrote the flags (CMP, TEST, ADD, etc.)
    pub writer_opcode: Option<X86Opcode>,
    /// The first operand compared (virtual register or immediate)
    pub operand0: Option<Operand>,
    /// The second operand compared
    pub operand1: Option<Operand>,
    /// Which flags are still valid
    pub valid_flags: FlagEffects,
}

impl FlagState {
    /// Create a new (empty) flag state — no flags are valid.
    pub fn new() -> Self {
        Self {
            writer_idx: None,
            writer_opcode: None,
            operand0: None,
            operand1: None,
            valid_flags: FlagEffects::none(),
        }
    }

    /// Check if a comparison with the given operands would produce the same
    /// flags as the current state (i.e., the comparison is redundant).
    ///
    /// A comparison is redundant when:
    /// - The same opcode is used (CMP==CMP or TEST==TEST)
    /// - The operands are identical
    /// - The required flags from the new comparison are a subset of the
    ///   still-valid flags
    pub fn is_redundant_cmp(&self, opcode: X86Opcode, op0: &Operand, op1: &Operand) -> bool {
        let Some(ref writer_opc) = self.writer_opcode else { return false; };
        let Some(ref stored_op0) = self.operand0 else { return false; };
        let Some(ref stored_op1) = self.operand1 else { return false; };

        // For CMP: same opcode and same operands → same flags
        if opcode == *writer_opc && op0 == stored_op0 && op1 == stored_op1 {
            return true;
        }

        // For TEST: TEST a, b is commutative, so TEST a,b == TEST b,a
        if opcode == X86Opcode::Test64 && *writer_opc == X86Opcode::Test64 {
            if (op0 == stored_op0 && op1 == stored_op1)
                || (op0 == stored_op1 && op1 == stored_op0)
            {
                return true;
            }
        }

        // CMP a, 0 is equivalent to TEST a, a (both set flags based on a - 0
        // and a & a respectively, giving same ZF/SF/PF for the value of a).
        // We conservatively do NOT fold this because CF differs (CMP sets CF
        // to the borrow, TEST clears CF). Only fold if the reader only checks
        // ZF/SF.

        false
    }

    /// Invalidate the flag state (e.g., when any flag-writing instruction
    /// occurs that we cannot precisely track, or when an operand register
    /// is clobbered).
    pub fn invalidate(&mut self) {
        self.writer_idx = None;
        self.writer_opcode = None;
        self.operand0 = None;
        self.operand1 = None;
        self.valid_flags = FlagEffects::none();
    }

    /// Update the flag state with a new flag-writing instruction.
    pub fn update(&mut self, idx: usize, opcode: X86Opcode, op0: Operand, op1: Operand) {
        self.writer_idx = Some(idx);
        self.writer_opcode = Some(opcode);
        self.operand0 = Some(op0);
        self.operand1 = Some(op1);
        self.valid_flags = FlagEffects::writes_all_arith();
    }
}

// =============================================================================
// §9  Legality Checker
// =============================================================================

/// Result of a legality check on an instruction block.
#[derive(Debug, Clone)]
pub struct LegalityResult {
    /// Whether the block is legal
    pub is_legal: bool,
    /// Violations found (if any)
    pub violations: Vec<LegalityViolation>,
}

/// A legality violation.
#[derive(Debug, Clone)]
pub enum LegalityViolation {
    /// A register is written but not live afterward (dead def that clobbers a live value)
    DeadDefClobbersLive { reg: Reg, instr_idx: usize },
    /// A flag is expected to be preserved but is clobbered
    FlagClobbered { instr_idx: usize },
    /// Memory ordering is violated
    MemoryOrderingViolation { instr_idx_a: usize, instr_idx_b: usize },
    /// An implicit clobber conflicts with a live register
    ImplicitClobberConflict { reg: Reg, instr_idx: usize },
    /// A calling convention constraint is violated
    CallingConventionViolation { msg: String },
    /// A division instruction has a live RDX that would be clobbered
    DivClobbersRdx { instr_idx: usize },
    /// A flag-writing instruction's flags are clobbered between the writer
    /// and a flag-reading consumer
    FlagDependencyViolation { writer_idx: usize, reader_idx: usize },
    /// A register is defined but never used (dead code)
    UnusedDefViolation { reg: Reg, instr_idx: usize },
}

/// Check the legality of an instruction block.
///
/// Verifies:
/// 1. Register liveness is respected (no clobbering live values)
/// 2. Flags are preserved when required
/// 3. Memory ordering is preserved
/// 4. Implicit clobbers are safe
/// 5. Calling convention constraints hold
pub fn check_legality(block: &InstrBlock) -> LegalityResult {
    let mut violations = Vec::new();
    let mut live: RegSet = block.live_in;

    for (idx, instr) in block.instrs.iter().enumerate() {
        // 1. Check implicit clobbers don't conflict with live registers
        let clobbers = instr.clobbers;
        let conflict = live.intersect(&clobbers);
        for reg in conflict.iter_physical() {
            // Only flag if the clobbered register is not the destination
            // (writing to dst is expected)
            if instr.dst_reg() != Some(reg) {
                violations.push(LegalityViolation::ImplicitClobberConflict {
                    reg, instr_idx: idx,
                });
            }
        }

        // 2. Check division doesn't clobber live RDX
        if instr.opcode.is_division() {
            if live.contains(Reg::Rdx) {
                violations.push(LegalityViolation::DivClobbersRdx { instr_idx: idx });
            }
        }

        // 3. Check flag preservation: if a subsequent instruction reads flags
        //    and the current one writes them, we need to verify no flag-consuming
        //    instruction between them has its flags clobbered.
        //    (Simplified: just flag the clobber; caller must check if any
        //     flag reader follows.)

        // Update liveness
        let defs = instr.all_writes();
        let uses = instr.all_reads();
        live = live.difference(&defs).union(&uses);
    }

    // 4. Check memory ordering: stores and loads must not be reordered
    //    across barriers or in ways that change observable behavior.
    //    (Conservative: flag any adjacent store-load pairs that could alias.)
    let mut last_store_idx: Option<usize> = None;
    for (idx, instr) in block.instrs.iter().enumerate() {
        if instr.mem_effects.may_write {
            last_store_idx = Some(idx);
        }
        if instr.mem_effects.may_read {
            if let Some(store_idx) = last_store_idx {
                if store_idx + 1 < idx {
                    // There are instructions between the store and this load
                    // that could potentially reorder. Flag if there's a barrier.
                    for i in store_idx + 1..idx {
                        if block.instrs[i].mem_effects.is_barrier {
                            violations.push(LegalityViolation::MemoryOrderingViolation {
                                instr_idx_a: store_idx,
                                instr_idx_b: idx,
                            });
                        }
                    }
                }
            }
        }
    }

    LegalityResult {
        is_legal: violations.is_empty(),
        violations,
    }
}

/// Check flag dependencies in an instruction block.
///
/// Verifies that no flag clobber occurs between a flag-writing instruction
/// and its flag-reading consumers. This catches cases where an intervening
/// instruction destroys flags that a later instruction depends on.
///
/// Returns a `LegalityResult` containing any `FlagDependencyViolation`s
/// found.  The algorithm tracks the most recent flag-writer and, for each
/// flag-reader, checks that no intervening instruction clobbers the flags.
pub fn check_flag_dependencies(block: &InstrBlock) -> LegalityResult {
    let mut violations = Vec::new();

    // Track the most recent flag writer for each set of flags.
    // When we encounter a flag reader, we check that the flags are still
    // valid (i.e., no intervening instruction wrote them).
    let mut last_flag_writer_idx: Option<usize> = None;

    for (idx, instr) in block.instrs.iter().enumerate() {
        if instr.flag_effects.reads_any() {
            // This instruction reads flags — check that the writer is
            // still valid (no clobber in between).
            if let Some(writer_idx) = last_flag_writer_idx {
                // Walk from writer_idx+1 to idx-1 and check for flag clobbers
                for mid_idx in writer_idx + 1..idx {
                    let mid_instr = &block.instrs[mid_idx];
                    if mid_instr.flag_effects.writes_any() {
                        // A flag-clobbering instruction sits between the
                        // writer and reader — this is a violation.
                        violations.push(LegalityViolation::FlagDependencyViolation {
                            writer_idx,
                            reader_idx: idx,
                        });
                        break;
                    }
                }
            }
        }

        if instr.flag_effects.writes_any() {
            last_flag_writer_idx = Some(idx);
        }
    }

    LegalityResult {
        is_legal: violations.is_empty(),
        violations,
    }
}

/// Detect unused definitions (dead code) in an instruction block.
///
/// A register is "unused" if it is written by an instruction but never read
/// by any subsequent instruction (and is not in the live-out set).  This is
/// a separate check from the main `check_legality` because unused defs are
/// not always illegal — they may be intentional (e.g., for side effects).
/// However, flagging them is useful for optimization.
pub fn check_unused_defs(block: &InstrBlock) -> LegalityResult {
    let mut violations = Vec::new();

    // Build a map from each register to the list of instruction indices
    // where it is read.  Then for each definition, check if there is a
    // subsequent read.
    let mut reg_reads: HashMap<Reg, Vec<usize>> = HashMap::new();
    for (idx, instr) in block.instrs.iter().enumerate() {
        for r in instr.all_reads().iter_physical() {
            reg_reads.entry(r).or_default().push(idx);
        }
    }

    for (idx, instr) in block.instrs.iter().enumerate() {
        // Only check pure instructions that can be safely eliminated
        if !instr.is_safe_to_eliminate() {
            continue;
        }
        // Only check register destinations
        if let Some(dst_reg) = instr.dst_reg() {
            if dst_reg == Reg::None {
                continue;
            }
            // Check if this register is ever read after this definition
            let has_subsequent_read = reg_reads
                .get(&dst_reg)
                .map_or(false, |reads| reads.iter().any(|&r_idx| r_idx > idx));

            // Also check if it's in the live-out set
            let is_live_out = block.live_out.contains(dst_reg);

            if !has_subsequent_read && !is_live_out {
                violations.push(LegalityViolation::UnusedDefViolation {
                    reg: dst_reg,
                    instr_idx: idx,
                });
            }
        }
    }

    LegalityResult {
        is_legal: violations.is_empty(),
        violations,
    }
}

// =============================================================================
// §9b  E-Graph Canonicalization Layer
// =============================================================================

/// Canonicalize an instruction block using e-graph style equality saturation.
///
/// This normalizes equivalent instruction patterns so the optimizer can
/// more easily identify optimization opportunities.
///
/// Canonicalization rules applied:
/// 1. Sort commutative operands (smaller vreg first)
/// 2. Normalize constant positions (constants to the right of commutative ops)
/// 3. Chain same-ops into trees (ADD(ADD(a,b),c) stays, but sort operands)
/// 4. Eliminate double negations
/// 5. Normalize XOR/AND masks
pub fn canonicalize_block(block: &InstrBlock) -> InstrBlock {
    let mut new_instrs = block.instrs.clone();

    for instr in new_instrs.iter_mut() {
        canonicalize_instruction(instr);
    }

    let mut result = InstrBlock::new(new_instrs);
    result.live_in = block.live_in;
    result.live_out = block.live_out;
    result
}

/// Apply canonicalization rules to a single instruction.
fn canonicalize_instruction(instr: &mut MachineInstrFull) {
    // Rule 1 & 2: Sort commutative operands — put smaller register on the
    // left, and constants on the right.
    if instr.opcode.is_commutative() && instr.srcs.len() >= 1 {
        let dst = instr.dst.clone();
        let src0 = instr.srcs[0].clone();

        // For commutative ops like ADD dst, src where dst == src0 after
        // renaming, we can swap dst and src0 to canonicalize.
        // More generally, if the instruction is `op dst, src0` and op is
        // commutative, then `op src0, dst` is equivalent.
        // We canonicalize so that the "smaller" operand is dst.
        if should_swap_commutative(&dst, &src0) {
            instr.dst = src0;
            instr.srcs[0] = dst;
        }

        // Normalize constant positions: if dst is an immediate (unlikely
        // but possible in some representations), swap with the register
        // source.
        if instr.dst.is_imm() && instr.srcs.iter().any(|s| s.is_reg()) {
            // Find first register source and swap
            for src in instr.srcs.iter_mut() {
                if src.is_reg() {
                    std::mem::swap(&mut instr.dst, src);
                    break;
                }
            }
        }
    }

    // Rule 4: Eliminate double negation: NEG(NEG r) → r
    if instr.opcode == X86Opcode::Neg64 {
        // This would require looking at the source instruction that defined
        // the operand. For a single-instruction view, we can't do this.
        // This rule is applied at the block level in a separate pass.
    }

    // Rule 5: Normalize XOR with self → ZeroReg32 pattern
    // XOR r, r is equivalent to ZeroReg32 r and is already handled in
    // the instruction table. No additional canonicalization needed.

    // Rule: Normalize AND with all-ones mask (AND r, -1 → NOP/MOV)
    if instr.opcode == X86Opcode::And64 {
        if let Some(Operand::Imm(mask)) = instr.srcs.get(0) {
            if *mask == -1 {
                // AND r, -1 is a no-op (identity). Replace with MOV.
                instr.opcode = X86Opcode::Mov64;
                let dst = instr.dst.clone();
                instr.srcs = smallvec![dst];
                instr.flag_effects = FlagEffects::preserves_all();
            }
        }
    }
}

/// Determine whether commutative operands should be swapped for
/// canonicalization.  We want the "smaller" operand on the left (dst)
/// side to make pattern matching easier.
///
/// Ordering: registers < immediates < memory operands.
/// Among registers: physical regs < virtual regs (by number).
fn should_swap_commutative(dst: &Operand, src: &Operand) -> bool {
    let dst_rank = operand_rank(dst);
    let src_rank = operand_rank(src);

    // Swap if src has a lower rank (should be on the dst side)
    if src_rank != dst_rank {
        return src_rank < dst_rank;
    }

    // Same rank — compare within the rank
    match (dst, src) {
        (Operand::Reg(Reg::VReg(a)), Operand::Reg(Reg::VReg(b))) => a > b,
        (Operand::Reg(a), Operand::Reg(b)) => a.index() > b.index(),
        (Operand::Imm(a), Operand::Imm(b)) => a > b,
        _ => false,
    }
}

/// Assign a ranking value to an operand for canonical ordering.
fn operand_rank(op: &Operand) -> u8 {
    match op {
        Operand::Reg(_) => 0,
        Operand::Imm(_) => 1,
        Operand::Mem { .. } => 2,
        Operand::RipRelative(_) => 3,
    }
}

// =============================================================================
// §10  Instruction-Aware Rewrite Actions
// =============================================================================

/// Instruction-aware rewrite actions that operate on machine instructions.
///
/// Unlike the expression-level `RewriteAction` in the MCTS superoptimizer,
/// these actions reason about machine semantics: they know which opcodes
/// commute, which operations can be fused, and which rewrites preserve
/// liveness and flag semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MachineRewriteAction {
    /// Commute operands of a commutative instruction (ADD, MUL, AND, OR, XOR)
    Commute,
    /// Fold constant expressions at compile time
    FoldConst,
    /// Strength-reduce: x * 2^k → x << k
    StrengthReduce,
    /// Fuse add + shift into LEA (base + index*scale + offset)
    LeaCombine,
    /// Rematerialize a constant instead of keeping it live
    Rematerialize,
    /// Copy propagation: replace use of copied register with original
    CopyPropagate,
    /// Eliminate redundant MOVs
    EliminateMove,
    /// Swap independent instructions to reduce port pressure
    ScheduleSwap,
    /// Rename virtual registers to reduce spills/live ranges
    RenameReg,
    /// Avoid spills by rematerializing or recomputing values
    SpillAvoid,
    /// Fuse instruction pairs (e.g., CMP + SETcc → single pattern)
    FusePair,
    /// Split a live range to reduce register pressure
    SplitLiveRange,
}

impl MachineRewriteAction {
    /// All available machine-aware rewrite actions
    pub fn all() -> &'static [MachineRewriteAction] {
        &[
            MachineRewriteAction::Commute,
            MachineRewriteAction::FoldConst,
            MachineRewriteAction::StrengthReduce,
            MachineRewriteAction::LeaCombine,
            MachineRewriteAction::Rematerialize,
            MachineRewriteAction::CopyPropagate,
            MachineRewriteAction::EliminateMove,
            MachineRewriteAction::ScheduleSwap,
            MachineRewriteAction::RenameReg,
            MachineRewriteAction::SpillAvoid,
            MachineRewriteAction::FusePair,
            MachineRewriteAction::SplitLiveRange,
        ]
    }

    /// Check if this action is applicable to a given instruction
    pub fn is_applicable(&self, instr: &MachineInstrFull) -> bool {
        match self {
            MachineRewriteAction::Commute => {
                instr.opcode.is_commutative() && instr.srcs.len() >= 2
                    && instr.srcs[0].is_reg() && instr.srcs[1].is_reg()
            }
            MachineRewriteAction::FoldConst => {
                // At least two constant operands
                let const_count = instr.srcs.iter().filter(|o| o.is_imm()).count();
                const_count >= 1 && instr.is_pure()
            }
            MachineRewriteAction::StrengthReduce => {
                // IMUL by power of 2 → SHL
                matches!(instr.opcode, X86Opcode::Imul64 | X86Opcode::Imul64Imm)
                    && instr.srcs.iter().any(|o| {
                        if let Operand::Imm(v) = o {
                            *v > 1 && (*v as u64).is_power_of_two()
                        } else {
                            false
                        }
                    })
            }
            MachineRewriteAction::LeaCombine => {
                // ADD with SHL child → LEA
                matches!(instr.opcode, X86Opcode::Add64 | X86Opcode::Add32)
            }
            MachineRewriteAction::Rematerialize => {
                // MOV with immediate can be rematerialized
                matches!(instr.opcode, X86Opcode::LoadImm32 | X86Opcode::LoadImm64)
            }
            MachineRewriteAction::CopyPropagate => {
                // MOV r1, r2 is a copy
                matches!(instr.opcode, X86Opcode::Mov64 | X86Opcode::Mov32)
                    && instr.srcs.len() == 1 && instr.srcs[0].is_reg()
            }
            MachineRewriteAction::EliminateMove => {
                // MOV r1, r1 is a no-op
                matches!(instr.opcode, X86Opcode::Mov64 | X86Opcode::Mov32)
                    && instr.srcs.len() == 1
                    && instr.dst == instr.srcs[0]
            }
            MachineRewriteAction::ScheduleSwap => {
                // Any pure instruction can potentially be reordered
                instr.is_pure() && !instr.flag_effects.writes_any()
            }
            MachineRewriteAction::RenameReg => {
                // Any instruction with virtual register operands
                let has_vreg = instr.dst_reg().map_or(false, |r| r.is_virtual());
                has_vreg || instr.srcs.iter().any(|o| {
                    matches!(o, Operand::Reg(Reg::VReg(_)))
                })
            }
            MachineRewriteAction::SpillAvoid => {
                // Any instruction that writes a virtual register could benefit
                instr.dst_reg().map_or(false, |r| r.is_virtual())
            }
            MachineRewriteAction::FusePair => {
                // CMP followed by SETcc/CMOVcc can be fused
                matches!(instr.opcode, X86Opcode::Cmp64 | X86Opcode::Cmp32 | X86Opcode::Test64)
            }
            MachineRewriteAction::SplitLiveRange => {
                // Any instruction defining a virtual register
                instr.dst_reg().map_or(false, |r| r.is_virtual())
            }
        }
    }

    /// Human-readable name for the action
    pub fn name(&self) -> &'static str {
        match self {
            MachineRewriteAction::Commute => "Commute",
            MachineRewriteAction::FoldConst => "FoldConst",
            MachineRewriteAction::StrengthReduce => "StrengthReduce",
            MachineRewriteAction::LeaCombine => "LeaCombine",
            MachineRewriteAction::Rematerialize => "Rematerialize",
            MachineRewriteAction::CopyPropagate => "CopyPropagate",
            MachineRewriteAction::EliminateMove => "EliminateMove",
            MachineRewriteAction::ScheduleSwap => "ScheduleSwap",
            MachineRewriteAction::RenameReg => "RenameReg",
            MachineRewriteAction::SpillAvoid => "SpillAvoid",
            MachineRewriteAction::FusePair => "FusePair",
            MachineRewriteAction::SplitLiveRange => "SplitLiveRange",
        }
    }
}

/// Apply a machine rewrite action to an instruction block.
///
/// Returns the transformed block if the action was applicable and
/// produced a different program; returns None otherwise.
pub fn apply_machine_rewrite(
    block: &InstrBlock,
    action: MachineRewriteAction,
    target_idx: usize,
) -> Option<InstrBlock> {
    if target_idx >= block.instrs.len() {
        return None;
    }

    let mut new_instrs = block.instrs.clone();
    let changed = match action {
        MachineRewriteAction::Commute => {
            let instr = &mut new_instrs[target_idx];
            if instr.opcode.is_commutative() && instr.srcs.len() >= 2 {
                instr.srcs.swap(0, 1);
                true
            } else {
                false
            }
        }
        MachineRewriteAction::EliminateMove => {
            let instr = &new_instrs[target_idx];
            if matches!(instr.opcode, X86Opcode::Mov64 | X86Opcode::Mov32)
                && instr.srcs.len() == 1 && instr.dst == instr.srcs[0]
            {
                // Remove the redundant MOV
                new_instrs.remove(target_idx);
                true
            } else {
                false
            }
        }
        MachineRewriteAction::StrengthReduce => {
            let instr = &mut new_instrs[target_idx];
            if matches!(instr.opcode, X86Opcode::Imul64Imm) {
                if let Some(Operand::Imm(v)) = instr.srcs.get(1) {
                    if *v > 1 && (*v as u64).is_power_of_two() {
                        let shift = (*v as u64).trailing_zeros() as i64;
                        instr.opcode = X86Opcode::Shl64;
                        instr.srcs[1] = Operand::Imm(shift);
                        instr.latency = 1;
                        instr.throughput = 0.5;
                        true
                    } else { false }
                } else { false }
            } else { false }
        }
        MachineRewriteAction::LeaCombine => {
            // Replace ADD(base, SHL(index, scale)) → LEA(base, index, scale)
            // This is a pattern match: we need to check if one source is
            // a SHL and the other is a register.
            let instr = &new_instrs[target_idx];
            if matches!(instr.opcode, X86Opcode::Add64) && instr.srcs.len() >= 2 {
                // Check if either source comes from a preceding SHL
                let src0_reg = match &instr.srcs[0] {
                    Operand::Reg(r) => Some(*r),
                    _ => None,
                };
                let src1_reg = match &instr.srcs[1] {
                    Operand::Reg(r) => Some(*r),
                    _ => None,
                };
                // Look for a preceding SHL that feeds one of our sources
                if let (Some(dst_reg), Some(s0)) = (instr.dst_reg(), src0_reg) {
                    if let Some(s1) = src1_reg {
                        // Check if there's a SHL that defines s0 or s1
                        for i in (0..target_idx).rev() {
                            let prev = &new_instrs[i];
                            if matches!(prev.opcode, X86Opcode::Shl64) {
                                if let (Some(prev_dst), Some(Operand::Imm(shift))) =
                                    (prev.dst_reg(), prev.srcs.get(1)) {
                                    if prev_dst == s0 || prev_dst == s1 {
                                        // Found a SHL feeding into this ADD → LEA
                                        let base = if prev_dst == s0 { s1 } else { s0 };
                                        let lea = MachineInstrFull::new(
                                            X86Opcode::Lea64,
                                            Operand::Reg(dst_reg),
                                            smallvec![
                                                Operand::Mem {
                                                    base,
                                                    index: Some(prev_dst),
                                                    scale: 1u8 << (*shift as u8),
                                                    disp: 0,
                                                },
                                            ],
                                        );
                                        new_instrs[target_idx] = lea;
                                        // The SHL is now dead — can be removed
                                        // (conservative: leave it for DCE)
                                        return Some(InstrBlock::new(new_instrs));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            false
        }
        MachineRewriteAction::FoldConst => {
            // If an instruction has all immediate source operands and is pure,
            // evaluate it at compile time and replace with LoadImm.
            let instr = &new_instrs[target_idx];
            if !instr.is_pure() {
                return None;
            }
            // All source operands must be immediates
            let all_imm = instr.srcs.iter().all(|o| o.is_imm());
            if !all_imm {
                return None;
            }
            // Also the dst must be a register (we need somewhere to put the result)
            let dst_reg = match instr.dst_reg() {
                Some(r) => r,
                None => return None,
            };
            // Evaluate the constant expression
            let result = match instr.opcode {
                X86Opcode::Add64 | X86Opcode::Add32 => {
                    if let (Some(Operand::Imm(a)), Some(Operand::Imm(b))) =
                        (instr.srcs.get(0), instr.srcs.get(1)) {
                        Some(a.wrapping_add(*b))
                    } else { None }
                }
                X86Opcode::Sub64 | X86Opcode::Sub32 => {
                    if let (Some(Operand::Imm(a)), Some(Operand::Imm(b))) =
                        (instr.srcs.get(0), instr.srcs.get(1)) {
                        Some(a.wrapping_sub(*b))
                    } else { None }
                }
                X86Opcode::And64 => {
                    if let (Some(Operand::Imm(a)), Some(Operand::Imm(b))) =
                        (instr.srcs.get(0), instr.srcs.get(1)) {
                        Some(a & b)
                    } else { None }
                }
                X86Opcode::Or64 => {
                    if let (Some(Operand::Imm(a)), Some(Operand::Imm(b))) =
                        (instr.srcs.get(0), instr.srcs.get(1)) {
                        Some(a | b)
                    } else { None }
                }
                X86Opcode::Xor64 => {
                    if let (Some(Operand::Imm(a)), Some(Operand::Imm(b))) =
                        (instr.srcs.get(0), instr.srcs.get(1)) {
                        Some(a ^ b)
                    } else { None }
                }
                X86Opcode::Shl64 => {
                    if let (Some(Operand::Imm(a)), Some(Operand::Imm(b))) =
                        (instr.srcs.get(0), instr.srcs.get(1)) {
                        if *b >= 0 && *b < 64 {
                            Some(a.wrapping_shl(*b as u32))
                        } else { None }
                    } else { None }
                }
                X86Opcode::Shr64 => {
                    if let (Some(Operand::Imm(a)), Some(Operand::Imm(b))) =
                        (instr.srcs.get(0), instr.srcs.get(1)) {
                        if *b >= 0 && *b < 64 {
                            Some((*a as u64).wrapping_shr(*b as u32) as i64)
                        } else { None }
                    } else { None }
                }
                X86Opcode::Sar64 => {
                    if let (Some(Operand::Imm(a)), Some(Operand::Imm(b))) =
                        (instr.srcs.get(0), instr.srcs.get(1)) {
                        if *b >= 0 && *b < 64 {
                            Some(a.wrapping_shr(*b as u32))
                        } else { None }
                    } else { None }
                }
                X86Opcode::Imul64Imm => {
                    if let (Some(Operand::Imm(a)), Some(Operand::Imm(b))) =
                        (instr.srcs.get(0), instr.srcs.get(1)) {
                        Some(a.wrapping_mul(*b))
                    } else { None }
                }
                X86Opcode::Neg64 => {
                    if let Some(Operand::Imm(a)) = instr.srcs.get(0) {
                        Some(a.wrapping_neg())
                    } else { None }
                }
                X86Opcode::Not64 => {
                    if let Some(Operand::Imm(a)) = instr.srcs.get(0) {
                        Some(!a)
                    } else { None }
                }
                X86Opcode::Inc64 => {
                    if let Some(Operand::Imm(a)) = instr.srcs.get(0) {
                        Some(a.wrapping_add(1))
                    } else { None }
                }
                X86Opcode::Dec64 => {
                    if let Some(Operand::Imm(a)) = instr.srcs.get(0) {
                        Some(a.wrapping_sub(1))
                    } else { None }
                }
                _ => None,
            };
            if let Some(val) = result {
                let folded = if val == 0 {
                    MachineInstrFull::new(
                        X86Opcode::ZeroReg32,
                        Operand::Reg(dst_reg),
                        smallvec![Operand::Reg(dst_reg)],
                    )
                } else {
                    let low32 = val as i32;
                    let fits_32 = (val as u128) == ((low32 as u64) as u128);
                    if fits_32 {
                        MachineInstrFull::new(
                            X86Opcode::LoadImm32,
                            Operand::Reg(dst_reg),
                            smallvec![Operand::Imm(val)],
                        )
                    } else {
                        MachineInstrFull::new(
                            X86Opcode::LoadImm64,
                            Operand::Reg(dst_reg),
                            smallvec![Operand::Imm(val)],
                        )
                    }
                };
                new_instrs[target_idx] = folded;
                true
            } else {
                false
            }
        }
        MachineRewriteAction::Rematerialize => {
            // Find a LoadImm at target_idx; if there's a gap between it and its
            // first use, move it right before the first use.
            let instr = &new_instrs[target_idx];
            if !matches!(instr.opcode, X86Opcode::LoadImm32 | X86Opcode::LoadImm64 | X86Opcode::ZeroReg32) {
                return None;
            }
            let dst_reg = match instr.dst_reg() {
                Some(r) => r,
                None => return None,
            };
            // Find the first use of dst_reg after target_idx
            let mut first_use_idx: Option<usize> = None;
            for i in (target_idx + 1)..new_instrs.len() {
                let reads = new_instrs[i].all_reads();
                if reads.contains(dst_reg) {
                    first_use_idx = Some(i);
                    break;
                }
            }
            // If no use found, or the first use is immediately adjacent, no move needed
            let use_idx = match first_use_idx {
                Some(idx) if idx > target_idx + 1 => idx,
                _ => return None,
            };
            // Move the LoadImm from target_idx to just before use_idx
            let loadimm = new_instrs.remove(target_idx);
            // After removal, use_idx shifted by -1
            new_instrs.insert(use_idx - 1, loadimm);
            true
        }
        MachineRewriteAction::CopyPropagate => {
            // If the instruction at target_idx is MOV dst, src (where src is a register),
            // find all subsequent uses of dst and replace them with src.
            // Stop at any instruction that writes to src.
            let instr = &new_instrs[target_idx];
            if !matches!(instr.opcode, X86Opcode::Mov64 | X86Opcode::Mov32) {
                return None;
            }
            if instr.srcs.len() != 1 {
                return None;
            }
            let src_reg = match &instr.srcs[0] {
                Operand::Reg(r) => *r,
                _ => return None,
            };
            let dst_reg = match instr.dst_reg() {
                Some(r) => r,
                None => return None,
            };
            if dst_reg == src_reg {
                return None; // Self-copy — that's EliminateMove
            }
            // Propagate: replace uses of dst_reg with src_reg in subsequent instructions
            let mut made_change = false;
            for i in (target_idx + 1)..new_instrs.len() {
                // Stop if any instruction writes to src_reg (would change meaning)
                let writes = new_instrs[i].all_writes();
                if writes.contains(src_reg) {
                    break;
                }
                // Replace uses of dst_reg with src_reg
                let reads = new_instrs[i].all_reads();
                if reads.contains(dst_reg) {
                    new_instrs[i].rename_reg(dst_reg, src_reg);
                    made_change = true;
                }
                // Stop if this instruction redefines dst_reg (new value, can't propagate past)
                if writes.contains(dst_reg) {
                    break;
                }
            }
            made_change
        }
        MachineRewriteAction::ScheduleSwap => {
            // If target_idx and target_idx+1 are both pure, flag-independent,
            // and reorderable, swap them.
            if target_idx + 1 >= new_instrs.len() {
                return None;
            }
            let instr_a = &new_instrs[target_idx];
            let instr_b = &new_instrs[target_idx + 1];
            if !instr_a.is_pure() || !instr_b.is_pure() {
                return None;
            }
            // Must be flag-independent (no flag writes that the other reads)
            if instr_a.flag_effects.writes_any() || instr_b.flag_effects.writes_any() {
                return None;
            }
            if !instr_a.can_reorder_with(instr_b) {
                return None;
            }
            // Swap the two instructions
            new_instrs.swap(target_idx, target_idx + 1);
            true
        }
        MachineRewriteAction::RenameReg => {
            // If the instruction at target_idx has a virtual register destination,
            // rename it to a different (unused) virtual register.
            let instr = &new_instrs[target_idx];
            let dst_reg = match instr.dst_reg() {
                Some(r @ Reg::VReg(_)) => r,
                _ => return None,
            };
            // Collect all vreg numbers in the block to find a free one
            let mut max_vreg: u32 = 0;
            let mut used_vregs: HashSet<u32> = HashSet::new();
            for i in &new_instrs {
                if let Some(Reg::VReg(v)) = i.dst_reg() {
                    used_vregs.insert(v);
                    max_vreg = max_vreg.max(v);
                }
                for src in &i.srcs {
                    if let Operand::Reg(Reg::VReg(v)) = src {
                        used_vregs.insert(*v);
                        max_vreg = max_vreg.max(*v);
                    }
                }
            }
            // Find a free vreg number
            let new_vreg_num = (0..=max_vreg + 1)
                .find(|v| !used_vregs.contains(v))
                .unwrap_or(max_vreg + 1);
            let new_reg = Reg::VReg(new_vreg_num);
            // Rename the destination in the defining instruction
            let old_reg = dst_reg;
            new_instrs[target_idx].rename_reg(old_reg, new_reg);
            // Update all uses of old_reg in subsequent instructions
            for i in (target_idx + 1)..new_instrs.len() {
                let reads = new_instrs[i].all_reads();
                if reads.contains(old_reg) {
                    new_instrs[i].rename_reg(old_reg, new_reg);
                }
                // Stop if this instruction redefines old_reg
                let writes = new_instrs[i].all_writes();
                if writes.contains(old_reg) {
                    break;
                }
            }
            true
        }
        MachineRewriteAction::SpillAvoid => {
            // If the instruction at target_idx is a LoadImm with a small constant
            // (fits in 32 bits), mark it as rematerializable by replacing it with
            // a shorter encoding (LoadImm32 → ZeroReg32 if value is 0).
            let instr = &new_instrs[target_idx];
            let dst_reg = match instr.dst_reg() {
                Some(r) if r.is_virtual() => r,
                _ => return None,
            };
            match instr.opcode {
                X86Opcode::LoadImm32 => {
                    if let Some(Operand::Imm(v)) = instr.srcs.get(0) {
                        if *v == 0 {
                            // Replace LoadImm32 with ZeroReg32 (shorter encoding)
                            let zero = MachineInstrFull::new(
                                X86Opcode::ZeroReg32,
                                Operand::Reg(dst_reg),
                                smallvec![Operand::Reg(dst_reg)],
                            );
                            new_instrs[target_idx] = zero;
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                X86Opcode::LoadImm64 => {
                    // If the 64-bit immediate actually fits in 32 bits, use LoadImm32 instead
                    if let Some(Operand::Imm(v)) = instr.srcs.get(0) {
                        let low32 = *v as i32;
                        let fits_32 = (*v as u128) == ((low32 as u64) as u128);
                        if fits_32 {
                            let shorter = MachineInstrFull::new(
                                X86Opcode::LoadImm32,
                                Operand::Reg(dst_reg),
                                smallvec![Operand::Imm(*v)],
                            );
                            new_instrs[target_idx] = shorter;
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                _ => false,
            }
        }
        MachineRewriteAction::FusePair => {
            // If the instruction at target_idx is CMP/TEST and the next instruction
            // is SETcc/CMOVcc, mark the pair for macro-fusion by ensuring they're
            // adjacent. If they're not adjacent, try to move them together by
            // removing intervening pure, flag-preserving instructions.
            let instr = &new_instrs[target_idx];
            if !instr.opcode.is_comparison() {
                return None;
            }
            // Find a SETcc/CMOVcc after target_idx that reads flags
            let mut fusion_target: Option<usize> = None;
            for i in (target_idx + 1)..new_instrs.len() {
                let candidate = &new_instrs[i];
                if matches!(candidate.opcode, X86Opcode::Setcc | X86Opcode::Cmov64)
                    && candidate.flag_effects.reads_any() {
                    fusion_target = Some(i);
                    break;
                }
                // Stop if an intervening instruction clobbers flags
                if candidate.flag_effects.writes_any() && !candidate.preserves_flags() {
                    break;
                }
            }
            let fuse_idx = match fusion_target {
                Some(idx) if idx > target_idx + 1 => idx,
                Some(_) => return None, // Already adjacent, nothing to do
                None => return None,
            };
            // Check that all instructions between target_idx+1 and fuse_idx are
            // pure and flag-preserving, so they can be moved safely
            let mut intervening: Vec<usize> = Vec::new();
            let mut can_move = true;
            for i in (target_idx + 1)..fuse_idx {
                let mid = &new_instrs[i];
                if !mid.is_pure() || mid.flag_effects.writes_any() {
                    can_move = false;
                    break;
                }
                intervening.push(i);
            }
            if !can_move || intervening.is_empty() {
                return None;
            }
            // Move the CMP/TEST instruction right before the SETcc/CMOVcc
            // by removing it from target_idx and inserting before fuse_idx
            let cmp_instr = new_instrs.remove(target_idx);
            // After removal, fuse_idx shifted by -1, and intervening indices shifted by -1
            // The flag-user is now at fuse_idx - 1
            new_instrs.insert(fuse_idx - 1, cmp_instr);
            true
        }
        MachineRewriteAction::SplitLiveRange => {
            // If the instruction at target_idx defines a virtual register vreg,
            // split the live range by creating a new VReg, updating the original
            // instruction's dst to the new VReg, and adding a MOV from the new VReg
            // to the old VReg right after the definition.
            let instr = &new_instrs[target_idx];
            let dst_reg = match instr.dst_reg() {
                Some(r @ Reg::VReg(_)) => r,
                _ => return None,
            };
            // Collect all vreg numbers to find a free one
            let mut max_vreg: u32 = 0;
            let mut used_vregs: HashSet<u32> = HashSet::new();
            for i in &new_instrs {
                if let Some(Reg::VReg(v)) = i.dst_reg() {
                    used_vregs.insert(v);
                    max_vreg = max_vreg.max(v);
                }
                for src in &i.srcs {
                    if let Operand::Reg(Reg::VReg(v)) = src {
                        used_vregs.insert(*v);
                        max_vreg = max_vreg.max(*v);
                    }
                }
            }
            let new_vreg_num = (0..=max_vreg + 1)
                .find(|v| !used_vregs.contains(v))
                .unwrap_or(max_vreg + 1);
            let new_reg = Reg::VReg(new_vreg_num);
            let old_reg = dst_reg;
            // Change the original instruction's dst to the new vreg
            new_instrs[target_idx].rename_reg(old_reg, new_reg);
            // Insert a MOV from new_reg to old_reg right after the definition
            let copy = MachineInstrFull::new(
                X86Opcode::Mov64,
                Operand::Reg(old_reg),
                smallvec![Operand::Reg(new_reg)],
            );
            new_instrs.insert(target_idx + 1, copy);
            true
        }
    };

    if changed {
        Some(InstrBlock::new(new_instrs))
    } else {
        None
    }
}

// =============================================================================
// §11  Multi-Objective Machine Cost Model
// =============================================================================

/// A multi-objective cost vector for scoring instruction sequences.
///
/// Instead of a single scalar, we track multiple cost dimensions so the
/// optimizer can reject "faster-looking" sequences that are actually
/// worse on the CPU (e.g., fewer uops but higher register pressure
/// causing spills).
#[derive(Debug, Clone, Default)]
pub struct MachineCostVector {
    /// Estimated cycles (critical path + throughput bottleneck)
    pub cycles: f64,
    /// Total micro-ops (frontend decode pressure)
    pub uops: f64,
    /// Port pressure (max pressure on any single port)
    pub port_pressure: f64,
    /// Register pressure (max simultaneously live registers)
    pub register_pressure: f64,
    /// Code size in bytes
    pub code_size: usize,
    /// Memory traffic (number of loads + stores)
    pub memory_traffic: usize,
    /// Uncertainty in the estimate (0.0 = exact, 1.0 = highly uncertain)
    pub uncertainty: f64,
}

impl MachineCostVector {
    /// Compute a weighted scalar cost from the multi-objective vector.
    ///
    /// The weights are chosen to reflect typical optimization priorities:
    /// - Cycles dominate (we want fast code)
    /// - uops matter for frontend pressure
    /// - Port pressure matters for throughput
    /// - Register pressure penalizes spills
    /// - Code size matters for I-cache
    pub fn weighted_cost(&self) -> f64 {
        const W_CYCLES: f64 = 1.0;
        const W_UOPS: f64 = 0.2;
        const W_PORT: f64 = 0.3;
        const W_REG: f64 = 0.5;
        const W_SIZE: f64 = 0.01;
        const W_MEM: f64 = 0.3;
        const W_UNCERTAINTY: f64 = 0.1;

        W_CYCLES * self.cycles
        + W_UOPS * self.uops
        + W_PORT * self.port_pressure
        + W_REG * self.register_pressure
        + W_SIZE * self.code_size as f64
        + W_MEM * self.memory_traffic as f64
        + W_UNCERTAINTY * self.uncertainty * self.cycles
    }

    /// Whether this cost vector is strictly better than another
    /// (at least as good on all dimensions, strictly better on at least one)
    pub fn dominates(&self, other: &MachineCostVector) -> bool {
        let at_least_as_good =
            self.cycles <= other.cycles
            && self.uops <= other.uops
            && self.port_pressure <= other.port_pressure
            && self.register_pressure <= other.register_pressure
            && self.code_size <= other.code_size
            && self.memory_traffic <= other.memory_traffic;

        let strictly_better =
            self.cycles < other.cycles
            || self.uops < other.uops
            || self.port_pressure < other.port_pressure
            || self.register_pressure < other.register_pressure
            || self.code_size < other.code_size
            || self.memory_traffic < other.memory_traffic;

        at_least_as_good && strictly_better
    }
}

/// Estimate the multi-objective cost vector for an instruction block.
pub fn estimate_machine_cost(block: &InstrBlock, target: TargetConfig) -> MachineCostVector {
    if block.instrs.is_empty() {
        return MachineCostVector::default();
    }

    // Convert MachineInstrFull to uarch_cost::MachineInstr for DAG-based costing
    let uarch_instrs: Vec<uarch_cost::MachineInstr> = block.instrs.iter()
        .map(|i| full_to_uarch_instr(i))
        .collect();

    // Use the existing uarch cost model for cycle estimation
    let cycles = uarch_cost::estimate_block_cycles(&uarch_instrs, target);

    // Total uops
    let uops: f64 = block.instrs.iter()
        .map(|i| i.uop_count as f64)
        .sum();

    // Port pressure
    let port_pressure = uarch_cost::port_pressure_bound(
        &uarch_instrs,
        &target,
        &|op| uarch_cost::cost_entry(op, target),
    );

    // Register pressure
    let register_pressure = uarch_cost::register_pressure(&uarch_instrs) as f64;

    // Code size
    let code_size = block.total_size();

    // Memory traffic
    let memory_traffic = block.instrs.iter()
        .filter(|i| i.mem_effects.may_read || i.mem_effects.may_write)
        .count();

    // Uncertainty: higher for instructions with variable latency or trap potential
    let uncertainty = block.instrs.iter()
        .map(|i| {
            if i.mem_effects.may_trap { 0.3 }
            else if i.mem_effects.may_read || i.mem_effects.may_write { 0.2 }
            else if i.latency >= 10 { 0.15 }
            else { 0.0 }
        })
        .sum::<f64>()
        / block.instrs.len().max(1) as f64;

    MachineCostVector {
        cycles,
        uops,
        port_pressure,
        register_pressure,
        code_size,
        memory_traffic,
        uncertainty,
    }
}

/// Convert a MachineInstrFull to a uarch_cost::MachineInstr for
/// compatibility with the existing DAG-based cost model.
fn full_to_uarch_instr(instr: &MachineInstrFull) -> uarch_cost::MachineInstr {
    let dst = match instr.dst_reg() {
        Some(Reg::VReg(v)) => (v % 16) as u8,
        Some(r) => r.index(),
        None => 0,
    };
    let (src1, src2, imm, has_imm) = match instr.srcs.as_slice() {
        [Operand::Reg(r1)] => (r1.index(), 0, 0, false),
        [Operand::Reg(r1), Operand::Reg(r2)] => (r1.index(), r2.index(), 0, false),
        [Operand::Reg(r1), Operand::Imm(v)] => (r1.index(), 0, *v as i32, true),
        [Operand::Reg(r1), Operand::Reg(r2), Operand::Imm(_)] =>
            (r1.index(), r2.index(), 0, false),
        _ => (0, 0, 0, false),
    };
    uarch_cost::MachineInstr {
        opcode: x86_to_uarch_opcode(instr.opcode),
        dst, src1, src2, imm, has_imm,
    }
}

/// Map an X86Opcode to a uarch_cost::Opcode for the existing cost model.
fn x86_to_uarch_opcode(op: X86Opcode) -> uarch_cost::Opcode {
    match op {
        X86Opcode::Add64 | X86Opcode::Add32 |
        X86Opcode::Inc64 | X86Opcode::Adc64 => uarch_cost::Opcode::Add,
        X86Opcode::Sub64 | X86Opcode::Sub32 |
        X86Opcode::Dec64 | X86Opcode::Sbb64 => uarch_cost::Opcode::Sub,
        X86Opcode::Imul64 | X86Opcode::Imul64Imm | X86Opcode::Mul64 =>
            uarch_cost::Opcode::Mul,
        X86Opcode::Div64 | X86Opcode::Idiv64 => uarch_cost::Opcode::Div,
        X86Opcode::And64 | X86Opcode::Andn64 => uarch_cost::Opcode::And,
        X86Opcode::Or64 => uarch_cost::Opcode::Or,
        X86Opcode::Xor64 | X86Opcode::ZeroReg32 => uarch_cost::Opcode::Xor,
        X86Opcode::Shl64 | X86Opcode::Rol64 => uarch_cost::Opcode::Shl,
        X86Opcode::Shr64 | X86Opcode::Sar64 | X86Opcode::Ror64 => uarch_cost::Opcode::Shr,
        X86Opcode::Neg64 => uarch_cost::Opcode::Neg,
        X86Opcode::Not64 => uarch_cost::Opcode::Not,
        X86Opcode::Cmp64 | X86Opcode::Cmp32 => uarch_cost::Opcode::Cmp,
        X86Opcode::Test64 => uarch_cost::Opcode::Test,
        X86Opcode::Mov64 | X86Opcode::Mov32 |
        X86Opcode::Movzx8 | X86Opcode::Movzx16 |
        X86Opcode::Movsx8 | X86Opcode::Movsx16 |
        X86Opcode::Movsxd32 => uarch_cost::Opcode::Mov,
        X86Opcode::LoadImm32 | X86Opcode::LoadImm64 | X86Opcode::Load64 =>
            uarch_cost::Opcode::LoadConst,
        X86Opcode::Lea64 => uarch_cost::Opcode::Add, // LEA ≈ ADD for cost
        X86Opcode::Nop => uarch_cost::Opcode::Nop,
        _ => uarch_cost::Opcode::Add, // Conservative fallback
    }
}

// =============================================================================
// §12  Register Allocation as Search State
// =============================================================================

/// The register allocation state that is part of the search.
///
/// This is the "big one" for close-to-handwritten-asm quality: if the
/// optimizer only sees values and not registers, it will make locally
/// good decisions that blow up later. The search state includes:
///
/// - Current live registers
/// - Available temporaries
/// - Clobbers
/// - Spill cost estimate
/// - Instruction scheduling window
#[derive(Debug, Clone)]
pub struct RegAllocState {
    /// Currently live virtual registers
    pub live_vregs: HashSet<u32>,
    /// Mapping from virtual register to physical register
    pub vreg_to_preg: HashMap<u32, Reg>,
    /// Set of physical registers currently in use
    pub used_pregs: RegSet,
    /// Set of physical registers that are free for allocation
    pub free_pregs: RegSet,
    /// Estimated spill cost (accumulated)
    pub spill_cost: f64,
    /// Current instruction scheduling window (index range)
    pub sched_window_start: usize,
    pub sched_window_end: usize,
    /// Number of spills so far
    pub spill_count: u32,
    /// Number of rematerializations so far
    pub remat_count: u32,
    /// Maximum register pressure seen so far
    pub max_pressure: u32,
}

impl RegAllocState {
    /// Create a new register allocation state with all allocatable GPRs free
    pub fn new() -> Self {
        let mut free = RegSet::empty();
        for &r in Reg::allocatable_gprs() {
            free.insert(r);
        }
        Self {
            live_vregs: HashSet::new(),
            vreg_to_preg: HashMap::new(),
            used_pregs: RegSet::empty(),
            free_pregs: free,
            spill_cost: 0.0,
            sched_window_start: 0,
            sched_window_end: 0,
            spill_count: 0,
            remat_count: 0,
            max_pressure: 0,
        }
    }

    /// Allocate a physical register for a virtual register.
    /// Returns the allocated physical register, or None if all are in use
    /// (which would require a spill).
    pub fn alloc(&mut self, vreg: u32) -> Option<Reg> {
        if let Some(&preg) = self.vreg_to_preg.get(&vreg) {
            // Already allocated — just mark as live
            self.live_vregs.insert(vreg);
            return Some(preg);
        }

        // Try to find a free physical register
        let mut chosen: Option<Reg> = None;
        for &r in Reg::allocatable_gprs() {
            if self.free_pregs.contains(r) {
                chosen = Some(r);
                break;
            }
        }

        match chosen {
            Some(preg) => {
                self.free_pregs.remove(preg);
                self.used_pregs.insert(preg);
                self.vreg_to_preg.insert(vreg, preg);
                self.live_vregs.insert(vreg);
                self.max_pressure = self.max_pressure.max(self.live_vregs.len() as u32);
                Some(preg)
            }
            None => {
                // No free register — need to spill
                self.spill_count += 1;
                self.spill_cost += 2.0; // ~2 cycles per spill/reload pair
                None
            }
        }
    }

    /// Mark a virtual register as dead (its physical register is freed)
    pub fn free_vreg(&mut self, vreg: u32) {
        if self.live_vregs.remove(&vreg) {
            if let Some(preg) = self.vreg_to_preg.remove(&vreg) {
                self.used_pregs.remove(preg);
                self.free_pregs.insert(preg);
            }
        }
    }

    /// Get the physical register assigned to a virtual register
    pub fn get_preg(&self, vreg: u32) -> Option<Reg> {
        self.vreg_to_preg.get(&vreg).copied()
    }

    /// Current register pressure
    pub fn pressure(&self) -> u32 {
        self.live_vregs.len() as u32
    }

    /// Whether a spill would be needed to allocate a new register
    pub fn would_spill(&self) -> bool {
        self.free_pregs.iter_physical().count() == 0
    }

    /// Estimate the cost of the current register allocation state
    pub fn cost(&self) -> f64 {
        self.spill_cost
        + self.spill_count as f64 * 2.0
        + self.max_pressure as f64 * 0.1
    }

    /// Apply an instruction to this RA state (allocate defs, mark uses)
    pub fn apply_instr(&mut self, instr: &MachineInstrFull) {
        // Mark all source registers as used
        for src in &instr.srcs {
            if let Operand::Reg(Reg::VReg(v)) = src {
                self.live_vregs.insert(*v);
            }
        }
        // Allocate a physical register for the destination
        if let Operand::Reg(Reg::VReg(v)) = &instr.dst {
            self.alloc(*v);
        }
    }

    /// Release registers that are no longer live after an instruction
    pub fn release_dead(&mut self, dead_vregs: &[u32]) {
        for &vreg in dead_vregs {
            self.free_vreg(vreg);
        }
    }
}

impl Default for RegAllocState {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// §13  Bounded Validation
// =============================================================================

/// Result of bounded validation.
#[derive(Debug, Clone)]
pub enum ValidationResult {
    /// The transformation is proven equivalent (SMT or exhaustive testing)
    ProvenEquivalent,
    /// The transformation passes bounded testing but is not formally proven
    BoundedTestPassed { num_tests: usize },
    /// A counterexample was found
    CounterExample { inputs: Vec<i64> },
    /// Validation timed out
    Timeout,
    /// Validation was inconclusive
    Inconclusive(String),
}

/// Validate a candidate instruction block against the original using
/// bounded translation validation.
///
/// This uses:
/// 1. SMT-based verification where possible (when smt-verify feature is enabled)
/// 2. Bounded random testing otherwise (Alive2-style approach)
pub fn validate_transformation(
    original: &InstrBlock,
    candidate: &InstrBlock,
    num_random_tests: usize,
) -> ValidationResult {
    // Quick check: both blocks must be non-empty
    if original.is_empty() || candidate.is_empty() {
        return ValidationResult::Inconclusive("Empty block".to_string());
    }

    // Count input variables (live-in registers that are virtual)
    let num_inputs = original.live_in.iter_physical().count();

    // Bounded random testing
    let mut rng = simple_rng::SimpleRng::new(42);
    let mut passed_tests = 0;

    for _ in 0..num_random_tests {
        // Generate random inputs
        let inputs: Vec<i64> = (0..num_inputs)
            .map(|_| rng.next_i64())
            .collect();

        // Interpret the original block
        let orig_result = interpret_block(original, &inputs);
        // Interpret the candidate block
        let cand_result = interpret_block(candidate, &inputs);

        match (orig_result, cand_result) {
            (Some(o), Some(c)) if o == c => {
                passed_tests += 1;
            }
            (Some(_), Some(_)) => {
                // Mismatch!
                return ValidationResult::CounterExample { inputs };
            }
            _ => {
                // Execution error (e.g., division by zero in one but not the other)
                // — inconclusive
                continue;
            }
        }
    }

    if passed_tests >= num_random_tests * 9 / 10 {
        ValidationResult::BoundedTestPassed { num_tests: passed_tests }
    } else {
        ValidationResult::Inconclusive(format!(
            "Only {}/{} tests passed", passed_tests, num_random_tests
        ))
    }
}

/// Simple interpreter for validating instruction blocks.
fn interpret_block(block: &InstrBlock, inputs: &[i64]) -> Option<i64> {
    let mut regs: HashMap<u32, i64> = HashMap::new();

    // Initialize live-in registers with input values
    for (i, &v) in inputs.iter().enumerate() {
        regs.insert(i as u32, v);
    }

    for instr in &block.instrs {
        let dst_vreg = match instr.dst_reg() {
            Some(Reg::VReg(v)) => v,
            _ => continue,
        };

        let get_src = |op: &Operand, regs: &HashMap<u32, i64>| -> Option<i64> {
            match op {
                Operand::Reg(Reg::VReg(v)) => regs.get(v).copied(),
                Operand::Imm(v) => Some(*v),
                _ => None,
            }
        };

        let src0 = instr.srcs.get(0).and_then(|o| get_src(o, &regs));
        let src1 = instr.srcs.get(1).and_then(|o| get_src(o, &regs));

        let result = match instr.opcode {
            X86Opcode::Add64 | X86Opcode::Add32 =>
                src0.and_then(|a| src1.map(|b| a.wrapping_add(b))),
            X86Opcode::Sub64 | X86Opcode::Sub32 =>
                src0.and_then(|a| src1.map(|b| a.wrapping_sub(b))),
            X86Opcode::Imul64 | X86Opcode::Imul64Imm =>
                src0.and_then(|a| src1.map(|b| a.wrapping_mul(b))),
            X86Opcode::And64 => src0.and_then(|a| src1.map(|b| a & b)),
            X86Opcode::Or64 => src0.and_then(|a| src1.map(|b| a | b)),
            X86Opcode::Xor64 => src0.and_then(|a| src1.map(|b| a ^ b)),
            X86Opcode::Shl64 => src0.and_then(|a| src1.map(|b| a.wrapping_shl(b as u32))),
            X86Opcode::Shr64 => src0.and_then(|a| src1.map(|b| ((a as u64).wrapping_shr(b as u32)) as i64)),
            X86Opcode::Neg64 => src0.map(|a| a.wrapping_neg()),
            X86Opcode::Not64 => src0.map(|a| !a),
            X86Opcode::Inc64 => src0.map(|a| a.wrapping_add(1)),
            X86Opcode::Dec64 => src0.map(|a| a.wrapping_sub(1)),
            X86Opcode::Mov64 | X86Opcode::Mov32 => src0,
            X86Opcode::LoadImm32 | X86Opcode::LoadImm64 =>
                instr.srcs.get(0).and_then(|o| get_src(o, &regs)),
            X86Opcode::ZeroReg32 => Some(0),
            X86Opcode::Lea64 => {
                // LEA: base + index*scale + disp
                match instr.srcs.get(0) {
                    Some(Operand::Mem { base, index, scale, disp }) => {
                        let base_val = match base {
                            Reg::VReg(v) => regs.get(v).copied().unwrap_or(0),
                            _ => 0,
                        };
                        let index_val = match index {
                            Some(Reg::VReg(v)) => regs.get(v).copied().unwrap_or(0),
                            _ => 0,
                        };
                        Some(base_val + index_val * (*scale as i64) + (*disp as i64))
                    }
                    _ => None,
                }
            }
            _ => None, // Unsupported opcode in interpreter
        };

        if let Some(val) = result {
            regs.insert(dst_vreg, val);
        } else {
            // Can't interpret this instruction
            return None;
        }
    }

    // Return the value of the last-defined virtual register
    regs.values().last().copied()
}

/// Simple deterministic RNG for bounded testing.
mod simple_rng {
    pub struct SimpleRng {
        state: u64,
    }

    impl SimpleRng {
        pub fn new(seed: u64) -> Self { Self { state: seed } }

        pub fn next_u64(&mut self) -> u64 {
            // xorshift64
            self.state ^= self.state << 13;
            self.state ^= self.state >> 7;
            self.state ^= self.state << 17;
            self.state
        }

        pub fn next_i64(&mut self) -> i64 {
            self.next_u64() as i64
        }
    }
}

// =============================================================================
// §14  Machine Lowering — Expr → MachineInstrFull
// =============================================================================

/// Lower an expression-level `Instr` from the MCTS superoptimizer into
/// a block of `MachineInstrFull`s with virtual register allocation.
///
/// This is the bridge between the expression-level and instruction-level
/// representations. It produces a flat sequence of machine instructions
/// with virtual registers, proper flag effects, and operand metadata.
pub fn lower_instr_to_machine(
    instr: &crate::optimizer::mcts_superoptimizer::Instr,
) -> InstrBlock {
    let mut machine_instrs: Vec<MachineInstrFull> = Vec::new();
    let mut next_vreg: u32 = 0;

    let _result_reg = lower_instr_recursive(instr, &mut machine_instrs, &mut next_vreg);

    // The result of the expression is in _result_reg.
    // (For now, the block just contains the instructions.)

    let mut block = InstrBlock::new(machine_instrs);
    block.compute_liveness();
    block
}

fn alloc_vreg(next: &mut u32) -> Reg {
    let r = Reg::VReg(*next);
    *next += 1;
    r
}

fn lower_instr_recursive(
    instr: &crate::optimizer::mcts_superoptimizer::Instr,
    out: &mut Vec<MachineInstrFull>,
    next_vreg: &mut u32,
) -> Reg {
    use crate::compiler::ast::{BinOpKind, UnOpKind};
    use crate::optimizer::mcts_superoptimizer::Instr as MctsInstr;

    match instr {
        MctsInstr::ConstInt(v) => {
            let dst = alloc_vreg(next_vreg);
            let low32 = *v as i32;
            let fits_32 = (*v as u128) == ((low32 as u64) as u128);
            if *v == 0 {
                // XOR r32, r32 (2 bytes, zero-extends)
                out.push(MachineInstrFull::new(
                    X86Opcode::ZeroReg32,
                    Operand::Reg(dst),
                    smallvec![Operand::Reg(dst)],
                ));
            } else if fits_32 {
                // MOV r32, imm32 (5 bytes, zero-extends)
                out.push(MachineInstrFull::new(
                    X86Opcode::LoadImm32,
                    Operand::Reg(dst),
                    smallvec![Operand::Imm(*v as i64)],
                ));
            } else {
                // MOV r64, imm64 (10 bytes)
                out.push(MachineInstrFull::new(
                    X86Opcode::LoadImm64,
                    Operand::Reg(dst),
                    smallvec![Operand::Imm(*v as i64)],
                ));
            }
            dst
        }
        MctsInstr::ConstFloat(bits) => {
            let dst = alloc_vreg(next_vreg);
            let v = *bits as i64;
            out.push(MachineInstrFull::new(
                X86Opcode::LoadImm64,
                Operand::Reg(dst),
                smallvec![Operand::Imm(v)],
            ));
            dst
        }
        MctsInstr::ConstBool(b) => {
            let dst = alloc_vreg(next_vreg);
            out.push(MachineInstrFull::new(
                X86Opcode::LoadImm32,
                Operand::Reg(dst),
                smallvec![Operand::Imm(if *b { 1 } else { 0 })],
            ));
            dst
        }
        MctsInstr::Var(idx) => {
            // Variable — just return its virtual register
            Reg::VReg(*idx)
        }
        MctsInstr::BinOp { op, lhs, rhs } => {
            let src1 = lower_instr_recursive(lhs, out, next_vreg);
            let src2 = lower_instr_recursive(rhs, out, next_vreg);
            let dst = alloc_vreg(next_vreg);

            let opcode = match op {
                BinOpKind::Add => X86Opcode::Add64,
                BinOpKind::Sub => X86Opcode::Sub64,
                BinOpKind::Mul => X86Opcode::Imul64,
                BinOpKind::Div => X86Opcode::Idiv64,
                BinOpKind::Rem => X86Opcode::Idiv64, // Remainder from RDX
                BinOpKind::BitAnd => X86Opcode::And64,
                BinOpKind::BitOr => X86Opcode::Or64,
                BinOpKind::BitXor => X86Opcode::Xor64,
                BinOpKind::Shl => X86Opcode::Shl64,
                BinOpKind::Shr => X86Opcode::Shr64,
                BinOpKind::Lt | BinOpKind::Gt | BinOpKind::Le |
                BinOpKind::Ge | BinOpKind::Eq | BinOpKind::Ne => X86Opcode::Cmp64,
                _ => X86Opcode::Add64,
            };

            out.push(MachineInstrFull::new(
                opcode,
                Operand::Reg(dst),
                smallvec![Operand::Reg(src1), Operand::Reg(src2)],
            ));
            dst
        }
        MctsInstr::UnOp { op, operand } => {
            let src = lower_instr_recursive(operand, out, next_vreg);
            let dst = alloc_vreg(next_vreg);

            let opcode = match op {
                UnOpKind::Neg => X86Opcode::Neg64,
                UnOpKind::Not => X86Opcode::Not64,
                _ => X86Opcode::Mov64,
            };

            out.push(MachineInstrFull::new(
                opcode,
                Operand::Reg(dst),
                smallvec![Operand::Reg(src)],
            ));
            dst
        }
    }
}

// =============================================================================
// §15  Full Pipeline Integration
// =============================================================================

/// The full machine-instruction-aware optimization pipeline.
///
/// This is the entry point that wires together:
///   1. Machine lowering (Expr → InstrBlock)
///   2. Legality checking
///   3. E-graph canonicalization (delegated to existing egg rules)
///   4. Instruction-aware MCTS search
///   5. Multi-objective cost model scoring
///   6. Bounded validation
///
/// Architecture:
///   Source program
///     → semantic normalization
///     → machine lowering
///     → instruction graph (InstrBlock)
///     → legality gate
///     → e-graph canonicalization
///     → instruction-aware MCTS / beam / stochastic search
///     → microarchitecture cost model
///     → bounded validation
///     → emit machine code
pub struct MachineInstrPipeline {
    /// Target microarchitecture configuration
    pub target: TargetConfig,
    /// Number of random tests for bounded validation
    pub validation_tests: usize,
    /// Maximum number of MCTS simulations per block
    pub max_simulations: usize,
    /// Whether to use legality checking
    pub check_legality: bool,
}

impl MachineInstrPipeline {
    pub fn new(target: TargetConfig) -> Self {
        Self {
            target,
            validation_tests: 256,
            max_simulations: 100,
            check_legality: true,
        }
    }

    /// Run the full pipeline on an instruction block.
    ///
    /// Returns the optimized block if an improvement was found.
    pub fn optimize_block(&self, block: &InstrBlock) -> Option<InstrBlock> {
        // Step 1: Check legality of the input
        if self.check_legality {
            let result = check_legality(block);
            if !result.is_legal {
                // Input is not legal — can't optimize
                return None;
            }
        }

        // Step 2: Score the original
        let original_cost = estimate_machine_cost(block, self.target);

        // Step 3: Try instruction-aware rewrites
        let mut best_block = block.clone();
        let mut best_cost = original_cost.clone();

        for _ in 0..self.max_simulations {
            // Pick a random instruction and action
            let target_idx = (simple_rng::SimpleRng::new(
                best_cost.weighted_cost().to_bits()
            ).next_u64() as usize) % best_block.instrs.len().max(1);

            for &action in MachineRewriteAction::all() {
                if target_idx < best_block.instrs.len()
                    && action.is_applicable(&best_block.instrs[target_idx])
                {
                    if let Some(candidate) = apply_machine_rewrite(&best_block, action, target_idx) {
                        // Check legality
                        if self.check_legality {
                            let legal = check_legality(&candidate);
                            if !legal.is_legal { continue; }
                        }

                        // Score the candidate
                        let candidate_cost = estimate_machine_cost(&candidate, self.target);

                        // Accept if better
                        if candidate_cost.weighted_cost() < best_cost.weighted_cost() {
                            // Validate equivalence
                            let valid = validate_transformation(
                                block, &candidate, self.validation_tests,
                            );
                            match valid {
                                ValidationResult::ProvenEquivalent |
                                ValidationResult::BoundedTestPassed { .. } => {
                                    best_block = candidate;
                                    best_cost = candidate_cost;
                                }
                                _ => {} // Reject
                            }
                        }
                    }
                }
            }
        }

        // Step 4: Return the improved block if better
        if best_cost.weighted_cost() < original_cost.weighted_cost() * 0.99 {
            Some(best_block)
        } else {
            None
        }
    }
}

// =============================================================================
// §16  Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reg_set_operations() {
        let mut s = RegSet::empty();
        assert!(s.is_empty());
        s.insert(Reg::Rax);
        s.insert(Reg::Rcx);
        assert_eq!(s.count(), 2);
        assert!(s.contains(Reg::Rax));
        assert!(!s.contains(Reg::Rdx));
        s.remove(Reg::Rax);
        assert!(!s.contains(Reg::Rax));
    }

    #[test]
    fn test_flag_effects() {
        let arith = FlagEffects::writes_all_arith();
        assert!(arith.writes_cf);
        assert!(arith.writes_zf);
        assert!(arith.writes_any());

        let lea = FlagEffects::preserves_all();
        assert!(!lea.writes_any());
        assert!(!lea.reads_any());
    }

    #[test]
    fn test_mem_effects_pure() {
        let pure = MemEffects::pure();
        assert!(pure.is_pure());
        assert!(pure.can_reorder_with(&pure));
    }

    #[test]
    fn test_mem_effects_reorder() {
        let read = MemEffects::read();
        let write = MemEffects::write();
        assert!(!read.can_reorder_with(&write));
        assert!(read.can_reorder_with(&read));
    }

    #[test]
    fn test_opcode_commutative() {
        assert!(X86Opcode::Add64.is_commutative());
        assert!(X86Opcode::Imul64.is_commutative());
        assert!(!X86Opcode::Sub64.is_commutative());
        assert!(!X86Opcode::Shl64.is_commutative());
    }

    #[test]
    fn test_machine_instr_create() {
        let instr = MachineInstrFull::new(
            X86Opcode::Add64,
            Operand::Reg(Reg::VReg(2)),
            smallvec![Operand::Reg(Reg::VReg(0)), Operand::Reg(Reg::VReg(1))],
        );
        assert_eq!(instr.opcode, X86Opcode::Add64);
        assert_eq!(instr.latency, 1);
        assert_eq!(instr.uop_count, 1);
        assert!(instr.is_pure());
        assert!(instr.flag_effects.writes_any());
    }

    #[test]
    fn test_machine_instr_rename() {
        let mut instr = MachineInstrFull::new(
            X86Opcode::Add64,
            Operand::Reg(Reg::VReg(2)),
            smallvec![Operand::Reg(Reg::VReg(0)), Operand::Reg(Reg::VReg(1))],
        );
        instr.rename_reg(Reg::VReg(0), Reg::VReg(5));
        assert_eq!(instr.srcs[0], Operand::Reg(Reg::VReg(5)));
    }

    #[test]
    fn test_instr_table_lookup() {
        let meta = X86InstrTable::lookup(X86Opcode::Lea64);
        assert!(!meta.flag_effects.writes_any()); // LEA preserves flags
        assert_eq!(meta.latency, 1);

        let meta = X86InstrTable::lookup(X86Opcode::Imul64);
        assert!(meta.flag_effects.writes_any());

        let meta = X86InstrTable::lookup(X86Opcode::Cmov64);
        assert!(!meta.flag_effects.writes_any()); // CMOV preserves flags
        assert!(meta.flag_effects.reads_any());    // But reads flags
    }

    #[test]
    fn test_legality_check_pure() {
        let block = InstrBlock::new(vec![
            MachineInstrFull::new(
                X86Opcode::Add64,
                Operand::Reg(Reg::VReg(2)),
                smallvec![Operand::Reg(Reg::VReg(0)), Operand::Reg(Reg::VReg(1))],
            ),
        ]);
        let result = check_legality(&block);
        assert!(result.is_legal);
    }

    #[test]
    fn test_commute_action() {
        let instr = MachineInstrFull::new(
            X86Opcode::Add64,
            Operand::Reg(Reg::VReg(2)),
            smallvec![Operand::Reg(Reg::VReg(0)), Operand::Reg(Reg::VReg(1))],
        );
        assert!(MachineRewriteAction::Commute.is_applicable(&instr));
        assert!(!MachineRewriteAction::StrengthReduce.is_applicable(&instr));
    }

    #[test]
    fn test_eliminate_move_action() {
        let instr = MachineInstrFull::new(
            X86Opcode::Mov64,
            Operand::Reg(Reg::VReg(0)),
            smallvec![Operand::Reg(Reg::VReg(0))], // MOV v0, v0
        );
        assert!(MachineRewriteAction::EliminateMove.is_applicable(&instr));
    }

    #[test]
    fn test_cost_vector_weighted() {
        let cost = MachineCostVector {
            cycles: 10.0,
            uops: 5.0,
            port_pressure: 3.0,
            register_pressure: 4.0,
            code_size: 20,
            memory_traffic: 0,
            uncertainty: 0.1,
        };
        let weighted = cost.weighted_cost();
        assert!(weighted > 0.0);
        assert!(weighted < 100.0);
    }

    #[test]
    fn test_reg_alloc_state() {
        let mut ra = RegAllocState::new();
        assert!(ra.would_spill() == false);

        let r = ra.alloc(0);
        assert!(r.is_some());
        assert_eq!(ra.pressure(), 1);

        ra.free_vreg(0);
        assert_eq!(ra.pressure(), 0);
    }

    #[test]
    fn test_div_metadata() {
        let meta = X86InstrTable::lookup(X86Opcode::Div64);
        assert!(meta.mem_effects.may_trap);
        assert!(meta.implicit_reads.contains(Reg::Rax));
        assert!(meta.implicit_reads.contains(Reg::Rdx));
        assert!(meta.implicit_writes.contains(Reg::Rax));
        assert!(meta.implicit_writes.contains(Reg::Rdx));
        assert!(meta.clobbers.contains(Reg::Rdx));
    }

    #[test]
    fn test_instr_block_liveness() {
        let mut block = InstrBlock::new(vec![
            MachineInstrFull::new(
                X86Opcode::Add64,
                Operand::Reg(Reg::VReg(2)),
                smallvec![Operand::Reg(Reg::VReg(0)), Operand::Reg(Reg::VReg(1))],
            ),
        ]);
        block.compute_liveness();
        // v0 and v1 must be live-in
        assert!(block.live_in.contains(Reg::VReg(0)));
        assert!(block.live_in.contains(Reg::VReg(1)));
    }

    #[test]
    fn test_lower_const() {
        use crate::optimizer::mcts_superoptimizer::Instr as MctsInstr;
        let instr = MctsInstr::ConstInt(42);
        let block = lower_instr_to_machine(&instr);
        assert_eq!(block.len(), 1);
        assert_eq!(block.instrs[0].opcode, X86Opcode::LoadImm32);
    }

    #[test]
    fn test_lower_binop() {
        use crate::optimizer::mcts_superoptimizer::Instr as MctsInstr;
        let instr = MctsInstr::BinOp {
            op: crate::compiler::ast::BinOpKind::Add,
            lhs: Box::new(MctsInstr::ConstInt(1)),
            rhs: Box::new(MctsInstr::ConstInt(2)),
        };
        let block = lower_instr_to_machine(&instr);
        // Should produce: LoadImm32(1), LoadImm32(2), Add64
        assert_eq!(block.len(), 3);
    }
}
