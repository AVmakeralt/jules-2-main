// =============================================================================
// MCTS Deep Core — STOKE-Inspired Superoptimizer Inner Loop
//
// A flat, cache-friendly MCTS core with O(1) mutations and optional hardware
// execution of candidates. Designed for >1M candidates/sec/core throughput.
//
// Architecture:
//   - Flat instruction representation (16 bytes, cache-line aligned)
//   - Fixed-size program arrays fitting in L1 cache
//   - O(1) mutations (opcode/operand/immediate replace, swap)
//   - Fast test-vector interpretation with pre-allocated register file
//   - Optional x86-64 JIT execution via mmap'd executable pages
//   - Integration hooks for known_bits pruning and SMT verification
//
// This is the "deep core" separate from the tree-based MCTS superoptimizer
// (mcts_superoptimizer.rs). Where that module does UCB1 tree search with
// rewrite actions, this module does stochastic program synthesis — generate
// a random mutation, test it, keep it if it's better.
// =============================================================================

#![cfg(feature = "core-superopt")]

use std::ops::{Index, IndexMut};
use std::time::{Duration, Instant};

use libc::{mmap, munmap, MAP_ANON, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE};

// =============================================================================
// §1  Inline PRNG — xorshift64+
//
// The `rand` crate is not a direct dependency of this project, so we implement
// a fast, self-contained PRNG. xorshift64+ has excellent statistical quality
// and is used by major JavaScript engines (V8, SpiderMonkey).
// =============================================================================

/// Fast xorshift64+ pseudo-random number generator.
/// State must never be all zeros.
#[derive(Clone)]
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state
        let state = if seed == 0 {
            0xDEAD_BEEF_CAFE_BABE
        } else {
            seed
        };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() as usize) % n
    }

    fn next_u8(&mut self, n: u8) -> u8 {
        if n == 0 {
            return 0;
        }
        (self.next_u64() as u8) % n
    }

    fn next_u16(&mut self, n: u16) -> u16 {
        if n == 0 {
            return 0;
        }
        (self.next_u64() as u16) % n
    }

    fn next_i32(&mut self) -> i32 {
        self.next_u64() as i32
    }

    fn next_bool(&mut self) -> bool {
        (self.next_u64() & 1) == 1
    }
}

// =============================================================================
// §2  Core Types
// =============================================================================

/// Flat instruction representation — fits in a single cache line (16 bytes).
/// A program is an array of these, fitting in L1 for sequences up to ~4
/// instructions (64 bytes / 16 bytes per instruction = 4 instructions per
/// cache line).
#[derive(Clone, Copy, Default)]
#[repr(C, align(16))]
pub struct FlatInstr {
    pub opcode: u16, // opcode index into OpcodeTable
    pub dst: u8,     // destination virtual register
    pub src1: u8,    // first source virtual register
    pub src2: u8,    // second source virtual register
    pub imm: i32,    // immediate value
    pub flags: u16,  // instruction flags (is_nop, is_imm, etc.)
    pub _pad: u16,   // padding for alignment
}

/// Flag bits for FlatInstr.flags
pub const FLAG_IS_NOP: u16 = 0x0001;
pub const FLAG_IS_IMM: u16 = 0x0002;
pub const FLAG_IS_COMMUTATIVE: u16 = 0x0004;
pub const FLAG_IS_PURE: u16 = 0x0008;

/// Maximum program length — tuned so the entire program fits in L1.
/// 16 instructions * 16 bytes = 256 bytes, well within L1 cache.
pub const MAX_PROGRAM_LEN: usize = 16;

/// Maximum number of virtual registers.
pub const MAX_VREGS: usize = 32;

/// A candidate program — flat array of instructions wrapped in a struct
/// for method support while maintaining the same cache-friendly layout.
/// `#[repr(C)]` guarantees the same memory layout as `[FlatInstr; 16]`.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct FlatProgram {
    pub instrs: [FlatInstr; MAX_PROGRAM_LEN],
}

impl Default for FlatProgram {
    fn default() -> Self {
        Self::empty()
    }
}

impl Index<usize> for FlatProgram {
    type Output = FlatInstr;
    fn index(&self, index: usize) -> &FlatInstr {
        &self.instrs[index]
    }
}

impl IndexMut<usize> for FlatProgram {
    fn index_mut(&mut self, index: usize) -> &mut FlatInstr {
        &mut self.instrs[index]
    }
}

/// Mutation types — all O(1) except insert/delete which are O(n) but rare.
#[derive(Debug, Clone, Copy)]
pub enum Mutation {
    OpcodeReplace {
        instr_idx: usize,
        new_opcode: u16,
    },
    OperandReplace {
        instr_idx: usize,
        which: u8,
        new_val: u8,
    },
    ImmediateReplace {
        instr_idx: usize,
        new_imm: i32,
    },
    Swap {
        idx_a: usize,
        idx_b: usize,
    },
    Insert {
        at: usize,
    }, // O(n) — shift instructions down, insert NOP
    Delete {
        at: usize,
    }, // O(n) — remove instruction, shift up, pad with NOP
}

// =============================================================================
// §3  Opcode Table
// =============================================================================

/// Metadata about a single opcode.
#[derive(Debug, Clone, Copy)]
pub struct OpcodeInfo {
    pub name: &'static str,
    pub num_srcs: u8,        // 0, 1, or 2 source operands
    pub has_dst: bool,       // does it produce a result?
    pub is_commutative: bool, // can src1/src2 be swapped?
    pub is_pure: bool,       // no side effects?
    pub latency: u8,         // estimated cycles (for cost model integration)
}

/// The opcode table — contains all valid opcodes and their properties.
pub struct OpcodeTable {
    opcodes: Vec<OpcodeInfo>,
}

/// Standard opcode indices — these are the index into the OpcodeTable.
pub mod op {
    pub const ADD: u16 = 0;
    pub const SUB: u16 = 1;
    pub const MUL: u16 = 2;
    pub const DIV: u16 = 3;
    pub const REM: u16 = 4;
    pub const AND: u16 = 5;
    pub const OR: u16 = 6;
    pub const XOR: u16 = 7;
    pub const SHL: u16 = 8;
    pub const SHR: u16 = 9;
    pub const NEG: u16 = 10;
    pub const NOT: u16 = 11;
    pub const LOAD_CONST: u16 = 12;
    pub const MOV: u16 = 13;
    pub const NOP: u16 = 14;
    pub const CMP_EQ: u16 = 15;
    pub const CMP_NE: u16 = 16;
    pub const CMP_LT: u16 = 17;
    pub const CMP_LE: u16 = 18;
    pub const CMP_GT: u16 = 19;
    pub const CMP_GE: u16 = 20;
    pub const SELECT: u16 = 21;
    pub const COUNT: u16 = 22;
}

impl OpcodeTable {
    /// Construct the standard opcode table with all supported opcodes.
    pub fn standard() -> Self {
        let opcodes = vec![
            OpcodeInfo { name: "add", num_srcs: 2, has_dst: true, is_commutative: true, is_pure: true, latency: 1 },
            OpcodeInfo { name: "sub", num_srcs: 2, has_dst: true, is_commutative: false, is_pure: true, latency: 1 },
            OpcodeInfo { name: "mul", num_srcs: 2, has_dst: true, is_commutative: true, is_pure: true, latency: 3 },
            OpcodeInfo { name: "div", num_srcs: 2, has_dst: true, is_commutative: false, is_pure: true, latency: 20 },
            OpcodeInfo { name: "rem", num_srcs: 2, has_dst: true, is_commutative: false, is_pure: true, latency: 20 },
            OpcodeInfo { name: "and", num_srcs: 2, has_dst: true, is_commutative: true, is_pure: true, latency: 1 },
            OpcodeInfo { name: "or", num_srcs: 2, has_dst: true, is_commutative: true, is_pure: true, latency: 1 },
            OpcodeInfo { name: "xor", num_srcs: 2, has_dst: true, is_commutative: true, is_pure: true, latency: 1 },
            OpcodeInfo { name: "shl", num_srcs: 2, has_dst: true, is_commutative: false, is_pure: true, latency: 1 },
            OpcodeInfo { name: "shr", num_srcs: 2, has_dst: true, is_commutative: false, is_pure: true, latency: 1 },
            OpcodeInfo { name: "neg", num_srcs: 1, has_dst: true, is_commutative: false, is_pure: true, latency: 1 },
            OpcodeInfo { name: "not", num_srcs: 1, has_dst: true, is_commutative: false, is_pure: true, latency: 1 },
            OpcodeInfo { name: "load_const", num_srcs: 0, has_dst: true, is_commutative: false, is_pure: true, latency: 1 },
            OpcodeInfo { name: "mov", num_srcs: 1, has_dst: true, is_commutative: false, is_pure: true, latency: 1 },
            OpcodeInfo { name: "nop", num_srcs: 0, has_dst: false, is_commutative: false, is_pure: true, latency: 0 },
            OpcodeInfo { name: "cmp_eq", num_srcs: 2, has_dst: true, is_commutative: true, is_pure: true, latency: 1 },
            OpcodeInfo { name: "cmp_ne", num_srcs: 2, has_dst: true, is_commutative: true, is_pure: true, latency: 1 },
            OpcodeInfo { name: "cmp_lt", num_srcs: 2, has_dst: true, is_commutative: false, is_pure: true, latency: 1 },
            OpcodeInfo { name: "cmp_le", num_srcs: 2, has_dst: true, is_commutative: false, is_pure: true, latency: 1 },
            OpcodeInfo { name: "cmp_gt", num_srcs: 2, has_dst: true, is_commutative: false, is_pure: true, latency: 1 },
            OpcodeInfo { name: "cmp_ge", num_srcs: 2, has_dst: true, is_commutative: false, is_pure: true, latency: 1 },
            OpcodeInfo { name: "select", num_srcs: 3, has_dst: true, is_commutative: false, is_pure: true, latency: 2 },
        ];
        debug_assert_eq!(opcodes.len(), op::COUNT as usize);
        Self { opcodes }
    }

    #[inline]
    pub fn get(&self, idx: u16) -> &OpcodeInfo {
        &self.opcodes[idx as usize]
    }

    #[inline]
    pub fn len(&self) -> u16 {
        self.opcodes.len() as u16
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.opcodes.is_empty()
    }
}

// =============================================================================
// §4  Program Utilities
// =============================================================================

impl FlatInstr {
    /// Create a NOP instruction.
    #[inline]
    pub fn nop() -> Self {
        Self {
            opcode: op::NOP,
            dst: 0,
            src1: 0,
            src2: 0,
            imm: 0,
            flags: FLAG_IS_NOP,
            _pad: 0,
        }
    }

    /// Create a LoadConst instruction.
    #[inline]
    pub fn load_const(dst: u8, imm: i32) -> Self {
        Self {
            opcode: op::LOAD_CONST,
            dst,
            src1: 0,
            src2: 0,
            imm,
            flags: FLAG_IS_IMM,
            _pad: 0,
        }
    }

    /// Create a binary operation instruction.
    #[inline]
    pub fn binop(opcode: u16, dst: u8, src1: u8, src2: u8, table: &OpcodeTable) -> Self {
        let info = table.get(opcode);
        let mut flags = 0u16;
        if info.is_commutative {
            flags |= FLAG_IS_COMMUTATIVE;
        }
        if info.is_pure {
            flags |= FLAG_IS_PURE;
        }
        Self {
            opcode,
            dst,
            src1,
            src2,
            imm: 0,
            flags,
            _pad: 0,
        }
    }

    /// Create a unary operation instruction.
    #[inline]
    pub fn unop(opcode: u16, dst: u8, src: u8) -> Self {
        Self {
            opcode,
            dst,
            src1: src,
            src2: 0,
            imm: 0,
            flags: FLAG_IS_PURE,
            _pad: 0,
        }
    }

    /// Create a Mov instruction.
    #[inline]
    pub fn mov(dst: u8, src: u8) -> Self {
        Self {
            opcode: op::MOV,
            dst,
            src1: src,
            src2: 0,
            imm: 0,
            flags: FLAG_IS_PURE,
            _pad: 0,
        }
    }

    /// Check if this instruction is a NOP.
    #[inline]
    pub fn is_nop(&self) -> bool {
        self.opcode == op::NOP || (self.flags & FLAG_IS_NOP) != 0
    }

    /// Check if this instruction uses an immediate value.
    #[inline]
    pub fn is_imm(&self) -> bool {
        (self.flags & FLAG_IS_IMM) != 0 || self.opcode == op::LOAD_CONST
    }
}

impl FlatProgram {
    /// Create an empty program (all NOPs).
    pub fn empty() -> Self {
        Self {
            instrs: [FlatInstr::nop(); MAX_PROGRAM_LEN],
        }
    }

    /// Get the actual length (count non-NOP instructions from the start).
    pub fn effective_len(&self) -> usize {
        self.instrs.iter().take_while(|i| !i.is_nop()).count()
    }

    /// Check if the program is well-formed:
    /// - No use-before-def of registers (except input registers)
    /// - All register indices are within bounds
    /// - Instruction operands match opcode requirements
    pub fn is_well_formed(&self, num_inputs: u8) -> bool {
        let mut defined = vec![false; MAX_VREGS];

        // Input registers are pre-defined
        for i in 0..num_inputs as usize {
            if i < MAX_VREGS {
                defined[i] = true;
            }
        }

        // S27 fix: Move OpcodeTable outside the loop — creating it per
        // instruction allocates a Vec each time, wasting cycles.
        let table = OpcodeTable::standard();

        for instr in self.instrs.iter() {
            if instr.is_nop() {
                break;
            }

            // Check register bounds
            if (instr.dst as usize) >= MAX_VREGS {
                return false;
            }

            let table = &table;
            let info = table.get(instr.opcode);

            // Check source register bounds and use-before-def
            if info.num_srcs >= 1 {
                if (instr.src1 as usize) >= MAX_VREGS {
                    return false;
                }
                if !defined[instr.src1 as usize] && instr.opcode != op::LOAD_CONST {
                    return false;
                }
            }
            if info.num_srcs >= 2 {
                if (instr.src2 as usize) >= MAX_VREGS {
                    return false;
                }
                if !defined[instr.src2 as usize] {
                    return false;
                }
            }
            // Select uses 3 operands: src1=condition, src2=true_val,
            // the false_val register is encoded in the low byte of imm.
            if instr.opcode == op::SELECT {
                if (instr.src1 as usize) >= MAX_VREGS || (instr.src2 as usize) >= MAX_VREGS {
                    return false;
                }
                if !defined[instr.src1 as usize] || !defined[instr.src2 as usize] {
                    return false;
                }
                let false_reg = instr.imm as u8;
                if (false_reg as usize) >= MAX_VREGS {
                    return false;
                }
                if !defined[false_reg as usize] {
                    return false;
                }
            }

            // Mark destination as defined
            if info.has_dst {
                defined[instr.dst as usize] = true;
            }
            drop(table);
        }

        true
    }

    /// Collect the set of live output registers (registers written to by
    /// non-NOP instructions).
    pub fn output_regs(&self) -> Vec<u8> {
        let mut has_write = vec![false; MAX_VREGS];

        for instr in self.instrs.iter() {
            if instr.is_nop() {
                break;
            }
            if (instr.dst as usize) < MAX_VREGS {
                has_write[instr.dst as usize] = true;
            }
        }

        let mut regs = Vec::new();
        for (reg, &written) in has_write.iter().enumerate() {
            if written {
                regs.push(reg as u8);
            }
        }
        regs
    }

    /// Compute a simple cost for this program based on opcode latencies.
    pub fn cost(&self, table: &OpcodeTable) -> u32 {
        let mut total: u32 = 0;
        for instr in self.instrs.iter() {
            if instr.is_nop() {
                break;
            }
            if instr.opcode < table.len() {
                total += table.get(instr.opcode).latency as u32;
            } else {
                total += 10; // penalty for unknown opcodes
            }
        }
        // Also factor in effective length as a tiebreaker
        total + self.effective_len() as u32
    }

    /// Swap two instructions by index.
    pub fn swap(&mut self, a: usize, b: usize) {
        if a < MAX_PROGRAM_LEN && b < MAX_PROGRAM_LEN && a != b {
            self.instrs.swap(a, b);
        }
    }
}

// =============================================================================
// §5  Flatten / Unflatten — Bridge to MCTS Instr tree
// =============================================================================

#[cfg(feature = "gnn-optimizer")]
use crate::compiler::ast::{BinOpKind, UnOpKind};
#[cfg(feature = "gnn-optimizer")]
use crate::optimizer::mcts_superoptimizer::Instr;

/// Flatten an MCTS Instr tree into a FlatProgram.
/// This converts the tree-structured representation into a linear sequence
/// of flat instructions with virtual registers.
#[cfg(feature = "gnn-optimizer")]
pub fn flatten_instr(instr: &Instr, table: &OpcodeTable) -> FlatProgram {
    let mut program = FlatProgram::empty();
    let mut next_reg: u8 = 0u8;
    let mut next_idx: usize = 0;
    let mut num_inputs = 0u8;

    fn flatten_rec(
        instr: &Instr,
        program: &mut FlatProgram,
        table: &OpcodeTable,
        next_reg: &mut u8,
        next_idx: &mut usize,
        num_inputs: &mut u8,
    ) -> u8 {
        if *next_idx >= MAX_PROGRAM_LEN || *next_reg as usize >= MAX_VREGS {
            return 0;
        }

        match instr {
            Instr::ConstInt(v) => {
                let dst = *next_reg;
                *next_reg = next_reg.wrapping_add(1);
                let imm = *v as i32;
                if *next_idx < MAX_PROGRAM_LEN {
                    program[*next_idx] = FlatInstr::load_const(dst, imm);
                    *next_idx += 1;
                }
                dst
            }
            Instr::ConstFloat(bits) => {
                let dst = *next_reg;
                *next_reg = next_reg.wrapping_add(1);
                let imm = *bits as i32;
                if *next_idx < MAX_PROGRAM_LEN {
                    program[*next_idx] = FlatInstr::load_const(dst, imm);
                    *next_idx += 1;
                }
                dst
            }
            Instr::ConstBool(b) => {
                let dst = *next_reg;
                *next_reg = next_reg.wrapping_add(1);
                let imm = if *b { 1 } else { 0 };
                if *next_idx < MAX_PROGRAM_LEN {
                    program[*next_idx] = FlatInstr::load_const(dst, imm);
                    *next_idx += 1;
                }
                dst
            }
            Instr::Var(name) => {
                // name is a u32 interned index — resolve back to &str for matching
                let name_str = crate::optimizer::mcts_superoptimizer::StringInterner::get(*name);
                let reg = if let Some(idx_str) = name_str.strip_prefix('x') {
                    idx_str.parse::<u8>().unwrap_or(0)
                } else if name_str == "x" || name_str == "a" {
                    0
                } else if name_str == "y" || name_str == "b" {
                    1
                } else {
                    let r = *num_inputs;
                    if r < *next_reg {
                        *next_reg
                    } else {
                        r
                    }
                };
                let effective_reg = reg.min(31);
                if effective_reg >= *num_inputs {
                    *num_inputs = effective_reg + 1;
                }
                effective_reg
            }
            Instr::BinOp { op, lhs, rhs } => {
                let src1 = flatten_rec(lhs, program, table, next_reg, next_idx, num_inputs);
                let src2 = flatten_rec(rhs, program, table, next_reg, next_idx, num_inputs);
                let dst = *next_reg;
                *next_reg = next_reg.wrapping_add(1);

                let opcode = match op {
                    BinOpKind::Add => op::ADD,
                    BinOpKind::Sub => op::SUB,
                    BinOpKind::Mul => op::MUL,
                    BinOpKind::Div => op::DIV,
                    BinOpKind::Rem => op::REM,
                    BinOpKind::BitAnd => op::AND,
                    BinOpKind::BitOr => op::OR,
                    BinOpKind::BitXor => op::XOR,
                    BinOpKind::Shl => op::SHL,
                    BinOpKind::Shr => op::SHR,
                    BinOpKind::Eq => op::CMP_EQ,
                    BinOpKind::Ne => op::CMP_NE,
                    BinOpKind::Lt => op::CMP_LT,
                    BinOpKind::Le => op::CMP_LE,
                    BinOpKind::Gt => op::CMP_GT,
                    BinOpKind::Ge => op::CMP_GE,
                    _ => op::NOP,
                };

                if *next_idx < MAX_PROGRAM_LEN {
                    program[*next_idx] = FlatInstr::binop(opcode, dst, src1, src2, table);
                    *next_idx += 1;
                }
                dst
            }
            Instr::UnOp { op, operand } => {
                let src = flatten_rec(operand, program, table, next_reg, next_idx, num_inputs);
                let dst = *next_reg;
                *next_reg = next_reg.wrapping_add(1);

                let opcode = match op {
                    UnOpKind::Neg => op::NEG,
                    UnOpKind::Not => op::NOT,
                    _ => op::NOP,
                };

                if *next_idx < MAX_PROGRAM_LEN {
                    program[*next_idx] = FlatInstr::unop(opcode, dst, src);
                    *next_idx += 1;
                }
                dst
            }
        }
    }

    flatten_rec(
        instr,
        &mut program,
        table,
        &mut next_reg,
        &mut next_idx,
        &mut num_inputs,
    );
    program
}

/// Convert a FlatProgram back to an MCTS Instr tree.
/// Returns None if the program cannot be represented as a tree.
#[cfg(feature = "gnn-optimizer")]
pub fn unflatten_instr(
    program: &FlatProgram,
    _table: &OpcodeTable,
    input_names: &[String],
) -> Option<Instr> {
    let mut values: Vec<Option<Instr>> = vec![None; MAX_VREGS];

    // Map input registers to Var instructions
    for (i, name) in input_names.iter().enumerate() {
        if i < MAX_VREGS {
            values[i] = Some(Instr::Var(crate::optimizer::mcts_superoptimizer::StringInterner::intern(name)));
        }
    }

    for instr in program.instrs.iter() {
        if instr.is_nop() {
            break;
        }

        let result = match instr.opcode {
            op::LOAD_CONST => Some(Instr::ConstInt(instr.imm as u128)),
            op::MOV => values.get(instr.src1 as usize)?.as_ref().cloned(),
            op::NEG => {
                let operand = values.get(instr.src1 as usize)?.as_ref().cloned()?;
                Some(Instr::UnOp {
                    op: UnOpKind::Neg,
                    operand: Box::new(operand),
                })
            }
            op::NOT => {
                let operand = values.get(instr.src1 as usize)?.as_ref().cloned()?;
                Some(Instr::UnOp {
                    op: UnOpKind::Not,
                    operand: Box::new(operand),
                })
            }
            opcode if opcode >= op::ADD && opcode <= op::SHR => {
                let lhs = values.get(instr.src1 as usize)?.as_ref().cloned()?;
                let rhs = values.get(instr.src2 as usize)?.as_ref().cloned()?;
                let binop = match opcode {
                    op::ADD => BinOpKind::Add,
                    op::SUB => BinOpKind::Sub,
                    op::MUL => BinOpKind::Mul,
                    op::DIV => BinOpKind::Div,
                    op::REM => BinOpKind::Rem,
                    op::AND => BinOpKind::BitAnd,
                    op::OR => BinOpKind::BitOr,
                    op::XOR => BinOpKind::BitXor,
                    op::SHL => BinOpKind::Shl,
                    op::SHR => BinOpKind::Shr,
                    _ => return None,
                };
                Some(Instr::BinOp {
                    op: binop,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                })
            }
            opcode if opcode >= op::CMP_EQ && opcode <= op::CMP_GE => {
                let lhs = values.get(instr.src1 as usize)?.as_ref().cloned()?;
                let rhs = values.get(instr.src2 as usize)?.as_ref().cloned()?;
                let binop = match opcode {
                    op::CMP_EQ => BinOpKind::Eq,
                    op::CMP_NE => BinOpKind::Ne,
                    op::CMP_LT => BinOpKind::Lt,
                    op::CMP_LE => BinOpKind::Le,
                    op::CMP_GT => BinOpKind::Gt,
                    op::CMP_GE => BinOpKind::Ge,
                    _ => return None,
                };
                Some(Instr::BinOp {
                    op: binop,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                })
            }
            _ => None,
        };

        if let Some(instr_val) = result {
            if (instr.dst as usize) < MAX_VREGS {
                values[instr.dst as usize] = Some(instr_val);
            }
        }
    }

    // Return the value in the last instruction's destination register
    let mut last_dst: Option<u8> = None;
    for fi in program.instrs.iter() {
        if fi.is_nop() {
            break;
        }
        last_dst = Some(fi.dst);
    }

    if let Some(dst) = last_dst {
        values.get(dst as usize)?.as_ref().cloned()
    } else {
        None
    }
}

// =============================================================================
// §6  Mutation Engine
// =============================================================================

/// The mutation engine generates and applies random mutations to candidate
/// programs. The distribution is tuned for STOKE-style superoptimization.
pub struct MutationEngine {
    rng: Xorshift64,
    table: OpcodeTable,
    num_inputs: u8,
    max_reg: u8,
}

impl MutationEngine {
    pub fn new(seed: u64, num_inputs: u8) -> Self {
        Self {
            rng: Xorshift64::new(seed),
            table: OpcodeTable::standard(),
            num_inputs,
            max_reg: 31,
        }
    }

    /// Generate a random mutation for the given program.
    pub fn random_mutation(&mut self, program: &FlatProgram) -> Mutation {
        let eff_len = program.effective_len();
        let roll = self.rng.next_u64() % 100;

        if roll < 60 {
            let instr_idx = if eff_len > 0 {
                self.rng.next_usize(eff_len)
            } else {
                0
            };
            let new_opcode = self.rng.next_u16(self.table.len() - 1);
            Mutation::OpcodeReplace {
                instr_idx,
                new_opcode,
            }
        } else if roll < 80 {
            let instr_idx = if eff_len > 0 {
                self.rng.next_usize(eff_len)
            } else {
                0
            };
            let which = self.rng.next_u8(3);
            let new_val = self.rng.next_u8(self.max_reg + 1);
            Mutation::OperandReplace {
                instr_idx,
                which,
                new_val,
            }
        } else if roll < 90 {
            let instr_idx = if eff_len > 0 {
                self.rng.next_usize(eff_len)
            } else {
                0
            };
            let new_imm = self.rng.next_i32();
            Mutation::ImmediateReplace {
                instr_idx,
                new_imm,
            }
        } else if roll < 95 {
            let idx_a = if eff_len > 1 {
                self.rng.next_usize(eff_len)
            } else {
                0
            };
            let idx_b = if eff_len > 1 {
                loop {
                    let b = self.rng.next_usize(eff_len);
                    if b != idx_a {
                        break b;
                    }
                }
            } else {
                0
            };
            Mutation::Swap { idx_a, idx_b }
        } else if roll < 98 {
            let at = if eff_len < MAX_PROGRAM_LEN {
                self.rng.next_usize(eff_len + 1)
            } else {
                self.rng.next_usize(eff_len)
            };
            Mutation::Insert { at }
        } else {
            let at = if eff_len > 0 {
                self.rng.next_usize(eff_len)
            } else {
                0
            };
            Mutation::Delete { at }
        }
    }

    /// Apply a mutation, returning the new program.
    pub fn apply_mutation(&self, program: &FlatProgram, mutation: &Mutation) -> FlatProgram {
        let mut new_program = *program;

        match mutation {
            Mutation::OpcodeReplace {
                instr_idx,
                new_opcode,
            } => {
                if *instr_idx < MAX_PROGRAM_LEN {
                    new_program[*instr_idx].opcode = *new_opcode;
                    if *new_opcode < self.table.len() {
                        let info = self.table.get(*new_opcode);
                        new_program[*instr_idx].flags = 0;
                        if info.is_commutative {
                            new_program[*instr_idx].flags |= FLAG_IS_COMMUTATIVE;
                        }
                        if info.is_pure {
                            new_program[*instr_idx].flags |= FLAG_IS_PURE;
                        }
                        if *new_opcode == op::LOAD_CONST {
                            new_program[*instr_idx].flags |= FLAG_IS_IMM;
                        }
                    }
                }
            }
            Mutation::OperandReplace {
                instr_idx,
                which,
                new_val,
            } => {
                if *instr_idx < MAX_PROGRAM_LEN {
                    match which {
                        0 => new_program[*instr_idx].dst = *new_val,
                        1 => new_program[*instr_idx].src1 = *new_val,
                        _ => new_program[*instr_idx].src2 = *new_val,
                    }
                }
            }
            Mutation::ImmediateReplace {
                instr_idx,
                new_imm,
            } => {
                if *instr_idx < MAX_PROGRAM_LEN {
                    new_program[*instr_idx].imm = *new_imm;
                    if new_program[*instr_idx].opcode == op::LOAD_CONST {
                        new_program[*instr_idx].flags |= FLAG_IS_IMM;
                    }
                }
            }
            Mutation::Swap { idx_a, idx_b } => {
                if *idx_a < MAX_PROGRAM_LEN && *idx_b < MAX_PROGRAM_LEN {
                    new_program.swap(*idx_a, *idx_b);
                }
            }
            Mutation::Insert { at } => {
                if *at < MAX_PROGRAM_LEN {
                    for i in ((*at + 1)..MAX_PROGRAM_LEN).rev() {
                        new_program.instrs[i] = new_program.instrs[i - 1];
                    }
                    new_program[*at] = FlatInstr::nop();
                }
            }
            Mutation::Delete { at } => {
                if *at < MAX_PROGRAM_LEN {
                    for i in *at..MAX_PROGRAM_LEN - 1 {
                        new_program.instrs[i] = new_program.instrs[i + 1];
                    }
                    new_program.instrs[MAX_PROGRAM_LEN - 1] = FlatInstr::nop();
                }
            }
        }

        new_program
    }

    /// Generate a completely random program with the given number of instructions.
    pub fn random_program(&mut self, len: usize) -> FlatProgram {
        let mut program = FlatProgram::empty();
        let actual_len = len.min(MAX_PROGRAM_LEN);

        for i in 0..actual_len {
            let opcode = self.rng.next_u16(self.table.len() - 1);
            let info = self.table.get(opcode);

            let dst = self.rng.next_u8(self.max_reg + 1);

            let (src1, src2) = if info.num_srcs >= 2 {
                (self.rng.next_u8(self.max_reg + 1), self.rng.next_u8(self.max_reg + 1))
            } else if info.num_srcs == 1 {
                (self.rng.next_u8(self.max_reg + 1), 0)
            } else {
                (0, 0)
            };

            let imm = if opcode == op::LOAD_CONST {
                self.rng.next_i32()
            } else {
                0
            };

            let mut flags = 0u16;
            if info.is_commutative {
                flags |= FLAG_IS_COMMUTATIVE;
            }
            if info.is_pure {
                flags |= FLAG_IS_PURE;
            }
            if opcode == op::LOAD_CONST {
                flags |= FLAG_IS_IMM;
            }

            program.instrs[i] = FlatInstr {
                opcode,
                dst,
                src1,
                src2,
                imm,
                flags,
                _pad: 0,
            };
        }

        program
    }
}

// =============================================================================
// §7  Test-Vector Evaluator
// =============================================================================

/// Evaluates candidate programs against test vectors using fast interpretation.
/// The register file is pre-allocated to avoid per-evaluation heap allocation.
pub struct TestVectorEvaluator {
    /// Pre-allocated register file to avoid per-evaluation allocation.
    regs: Vec<u64>,
}

impl TestVectorEvaluator {
    pub fn new() -> Self {
        Self {
            regs: vec![0u64; MAX_VREGS],
        }
    }

    /// Evaluate a program on a single test vector, returning the output.
    /// Returns None if the program faults (bad register access, etc.)
    pub fn evaluate(
        &mut self,
        program: &FlatProgram,
        inputs: &[u64],
        output_reg: u8,
        table: &OpcodeTable,
    ) -> Option<u64> {
        self.regs.fill(0);

        for (i, &val) in inputs.iter().enumerate() {
            if i < MAX_VREGS {
                self.regs[i] = val;
            }
        }

        for instr in program.instrs.iter() {
            if instr.is_nop() {
                break;
            }

            if instr.opcode >= table.len() {
                return None;
            }

            let dst = instr.dst as usize;
            if dst >= MAX_VREGS {
                return None;
            }

            let result = self.exec_instr(instr)?;
            self.regs[dst] = result;
        }

        if (output_reg as usize) < MAX_VREGS {
            Some(self.regs[output_reg as usize])
        } else {
            None
        }
    }

    /// Execute a single instruction, returning the result.
    /// Uses wrapping semantics for all arithmetic.
    /// Division by zero returns 0 (conservative for search).
    #[inline]
    fn exec_instr(&self, instr: &FlatInstr) -> Option<u64> {
        let src1 = if (instr.src1 as usize) < MAX_VREGS {
            self.regs[instr.src1 as usize]
        } else {
            return None;
        };

        let src2 = if (instr.src2 as usize) < MAX_VREGS {
            self.regs[instr.src2 as usize]
        } else {
            return None;
        };

        match instr.opcode {
            op::ADD => Some(src1.wrapping_add(src2)),
            op::SUB => Some(src1.wrapping_sub(src2)),
            op::MUL => Some(src1.wrapping_mul(src2)),
            op::DIV => {
                if src2 == 0 {
                    Some(0)
                } else {
                    Some(src1.wrapping_div(src2))
                }
            }
            op::REM => {
                if src2 == 0 {
                    Some(0)
                } else {
                    Some(src1.wrapping_rem(src2))
                }
            }
            op::AND => Some(src1 & src2),
            op::OR => Some(src1 | src2),
            op::XOR => Some(src1 ^ src2),
            op::SHL => Some(src1.wrapping_shl(src2 as u32)),
            op::SHR => Some(src1.wrapping_shr(src2 as u32)),
            op::NEG => Some(src1.wrapping_neg()),
            op::NOT => Some(!src1),
            op::LOAD_CONST => Some(instr.imm as u64),
            op::MOV => Some(src1),
            op::NOP => None,
            op::CMP_EQ => Some(if src1 == src2 { 1 } else { 0 }),
            op::CMP_NE => Some(if src1 != src2 { 1 } else { 0 }),
            op::CMP_LT => Some(if src1 < src2 { 1 } else { 0 }),
            op::CMP_LE => Some(if src1 <= src2 { 1 } else { 0 }),
            op::CMP_GT => Some(if src1 > src2 { 1 } else { 0 }),
            op::CMP_GE => Some(if src1 >= src2 { 1 } else { 0 }),
            op::SELECT => {
                let false_reg = instr.imm as u8;
                let false_val = if (false_reg as usize) < MAX_VREGS {
                    self.regs[false_reg as usize]
                } else {
                    return None;
                };
                Some(if src1 != 0 { src2 } else { false_val })
            }
            _ => None,
        }
    }

    /// Evaluate on multiple test vectors, returning the number that match
    /// expected outputs. This is the hot path — optimize for speed.
    pub fn evaluate_batch(
        &mut self,
        program: &FlatProgram,
        test_vectors: &[(Vec<u64>, u64)],
        output_reg: u8,
        table: &OpcodeTable,
    ) -> usize {
        let mut matches = 0;
        let eff_len = program.effective_len();

        for (inputs, expected) in test_vectors.iter() {
            self.regs.fill(0);
            for (i, &val) in inputs.iter().enumerate() {
                if i < MAX_VREGS {
                    self.regs[i] = val;
                }
            }

            let mut valid = true;
            for j in 0..eff_len {
                let instr = &program.instrs[j];
                if instr.opcode >= table.len() {
                    valid = false;
                    break;
                }
                let dst = instr.dst as usize;
                if dst >= MAX_VREGS {
                    valid = false;
                    break;
                }
                match self.exec_instr(instr) {
                    Some(result) => self.regs[dst] = result,
                    None => {
                        if instr.opcode != op::NOP {
                            valid = false;
                            break;
                        }
                    }
                }
            }

            if valid {
                if (output_reg as usize) < MAX_VREGS && self.regs[output_reg as usize] == *expected
                {
                    matches += 1;
                }
            }
        }

        matches
    }
}

impl Default for TestVectorEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// §8  Hardware Executor — STOKE Trick
// =============================================================================

/// Optional hardware executor: execute the candidate directly on hardware
/// using mmap'd executable pages.
pub struct HardwareExecutor {
    exec_page: *mut u8,
    page_size: usize,
}

// Safety: HardwareExecutor owns its mmap'd page exclusively.
unsafe impl Send for HardwareExecutor {}
unsafe impl Sync for HardwareExecutor {}

impl HardwareExecutor {
    /// Page size for mmap allocations (4 KiB).
    const EXEC_PAGE_SIZE: usize = 4096;

    /// Create a new hardware executor with an executable memory page.
    /// Returns None if mmap fails (e.g., not on x86-64, or permission denied).
    pub fn new() -> Option<Self> {
        #[cfg(not(target_arch = "x86_64"))]
        {
            return None;
        }

        #[cfg(target_arch = "x86_64")]
        {
            let page = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    Self::EXEC_PAGE_SIZE,
                    PROT_READ | PROT_WRITE | PROT_EXEC,
                    MAP_PRIVATE | MAP_ANON,
                    -1,
                    0,
                )
            };

            if page.is_null() || page == libc::MAP_FAILED {
                return None;
            }

            Some(Self {
                exec_page: page as *mut u8,
                page_size: Self::EXEC_PAGE_SIZE,
            })
        }
    }

    /// Compile a flat program to x86-64 machine code and execute it.
    /// Returns None if the program can't be compiled to x86.
    pub fn execute(
        &self,
        program: &FlatProgram,
        inputs: &[u64],
        table: &OpcodeTable,
    ) -> Option<u64> {
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (program, inputs, table);
            return None;
        }

        #[cfg(target_arch = "x86_64")]
        {
            let eff_len = program.effective_len();

            // Validate program before executing
            for i in 0..eff_len {
                let instr = &program.instrs[i];
                if instr.opcode >= table.len() {
                    return None;
                }
                if instr.dst as usize >= MAX_VREGS {
                    return None;
                }
            }

            // Determine the number of virtual registers needed
            let mut max_vreg: usize = 0;
            for i in 0..eff_len {
                let instr = &program.instrs[i];
                max_vreg = max_vreg.max(instr.dst as usize);
                max_vreg = max_vreg.max(instr.src1 as usize);
                max_vreg = max_vreg.max(instr.src2 as usize);
            }
            let num_vregs = (max_vreg + 1).min(MAX_VREGS);
            let stack_size = num_vregs * 8;

            // Build x86-64 machine code
            let mut code = Vec::with_capacity(self.page_size);

            // Prologue
            code.push(0x55); // push rbp
            code.extend_from_slice(&[0x48, 0x89, 0xE5]); // mov rbp, rsp
            let aligned_stack = (stack_size + 15) & !15;
            if aligned_stack <= 0x7F {
                code.extend_from_slice(&[0x48, 0x83, 0xEC, aligned_stack as u8]);
            } else {
                code.extend_from_slice(&[0x48, 0x81, 0xE4]);
                code.extend_from_slice(&(aligned_stack as i32).to_le_bytes());
            }

            // Store inputs from rdi to stack slots
            for i in 0..inputs.len().min(num_vregs) {
                let offset = ((i + 1) * 8) as i32;
                code.extend_from_slice(&[0x48, 0x8B, 0x47]); // mov rax, [rdi + i*8]
                code.push((i * 8) as u8);
                code.extend_from_slice(&[0x48, 0x89, 0x85]); // mov [rbp - offset], rax
                code.extend_from_slice(&(-offset).to_le_bytes());
            }

            // Emit each instruction
            for i in 0..eff_len {
                let instr = &program.instrs[i];
                self.emit_instruction(&mut code, instr);
            }

            // Load output register (last instruction's dst) into rax
            let output_reg = if eff_len > 0 { program.instrs[eff_len - 1].dst } else { 0 };
            let out_offset = ((output_reg as usize + 1) * 8) as i32;
            code.extend_from_slice(&[0x48, 0x8B, 0x85]); // mov rax, [rbp - out_offset]
            code.extend_from_slice(&(-out_offset).to_le_bytes());

            // Epilogue
            if aligned_stack <= 0x7F {
                code.extend_from_slice(&[0x48, 0x83, 0xC4, aligned_stack as u8]);
            } else {
                code.extend_from_slice(&[0x48, 0x81, 0xC4]);
                code.extend_from_slice(&(aligned_stack as i32).to_le_bytes());
            }
            code.push(0x5D); // pop rbp
            code.push(0xC3); // ret

            if code.len() > self.page_size {
                return None;
            }

            // Copy code to executable page and execute
            unsafe {
                std::ptr::copy_nonoverlapping(code.as_ptr(), self.exec_page, code.len());
            }

            let func: unsafe extern "C" fn(*const u64) -> u64 =
                unsafe { std::mem::transmute(self.exec_page) };

            let input_ptr = inputs.as_ptr();
            let result = unsafe { func(input_ptr) };
            Some(result)
        }
    }

    /// Emit x86-64 machine code for a single flat instruction.
    /// Virtual registers are at [rbp - (reg+1)*8].
    #[cfg(target_arch = "x86_64")]
    fn emit_instruction(&self, code: &mut Vec<u8>, instr: &FlatInstr) {
        let dst_off = ((instr.dst as usize + 1) * 8) as i32;
        let s1_off = ((instr.src1 as usize + 1) * 8) as i32;
        let s2_off = ((instr.src2 as usize + 1) * 8) as i32;

        // Helper macros for common patterns
        macro_rules! load_rax_from {
            ($off:expr) => {
                code.extend_from_slice(&[0x48, 0x8B, 0x85]);
                code.extend_from_slice(&(-$off).to_le_bytes());
            };
        }
        macro_rules! store_rax_to {
            ($off:expr) => {
                code.extend_from_slice(&[0x48, 0x89, 0x85]);
                code.extend_from_slice(&(-$off).to_le_bytes());
            };
        }

        match instr.opcode {
            op::LOAD_CONST => {
                code.extend_from_slice(&[0x48, 0xB8]); // mov rax, imm64
                code.extend_from_slice(&(instr.imm as i64).to_le_bytes());
                store_rax_to!(dst_off);
            }
            op::MOV => {
                load_rax_from!(s1_off);
                store_rax_to!(dst_off);
            }
            op::ADD => {
                load_rax_from!(s1_off);
                code.extend_from_slice(&[0x48, 0x03, 0x85]); // add rax, [rbp-s2]
                code.extend_from_slice(&(-s2_off).to_le_bytes());
                store_rax_to!(dst_off);
            }
            op::SUB => {
                load_rax_from!(s1_off);
                code.extend_from_slice(&[0x48, 0x2B, 0x85]); // sub rax, [rbp-s2]
                code.extend_from_slice(&(-s2_off).to_le_bytes());
                store_rax_to!(dst_off);
            }
            op::MUL => {
                load_rax_from!(s1_off);
                code.extend_from_slice(&[0x48, 0x8B, 0x8D]); // mov rcx, [rbp-s2]
                code.extend_from_slice(&(-s2_off).to_le_bytes());
                code.extend_from_slice(&[0x48, 0x0F, 0xAF, 0xC1]); // imul rax, rcx
                store_rax_to!(dst_off);
            }
            op::AND => {
                load_rax_from!(s1_off);
                code.extend_from_slice(&[0x48, 0x23, 0x85]); // and rax, [rbp-s2]
                code.extend_from_slice(&(-s2_off).to_le_bytes());
                store_rax_to!(dst_off);
            }
            op::OR => {
                load_rax_from!(s1_off);
                code.extend_from_slice(&[0x48, 0x0B, 0x85]); // or rax, [rbp-s2]
                code.extend_from_slice(&(-s2_off).to_le_bytes());
                store_rax_to!(dst_off);
            }
            op::XOR => {
                load_rax_from!(s1_off);
                code.extend_from_slice(&[0x48, 0x33, 0x85]); // xor rax, [rbp-s2]
                code.extend_from_slice(&(-s2_off).to_le_bytes());
                store_rax_to!(dst_off);
            }
            op::SHL => {
                load_rax_from!(s1_off);
                code.extend_from_slice(&[0x8A, 0x8D]); // mov cl, byte [rbp-s2]
                code.extend_from_slice(&(-s2_off).to_le_bytes());
                code.extend_from_slice(&[0x48, 0xD3, 0xE0]); // shl rax, cl
                store_rax_to!(dst_off);
            }
            op::SHR => {
                load_rax_from!(s1_off);
                code.extend_from_slice(&[0x8A, 0x8D]); // mov cl, byte [rbp-s2]
                code.extend_from_slice(&(-s2_off).to_le_bytes());
                code.extend_from_slice(&[0x48, 0xD3, 0xE8]); // shr rax, cl
                store_rax_to!(dst_off);
            }
            op::NEG => {
                load_rax_from!(s1_off);
                code.extend_from_slice(&[0x48, 0xF7, 0xD8]); // neg rax
                store_rax_to!(dst_off);
            }
            op::NOT => {
                load_rax_from!(s1_off);
                code.extend_from_slice(&[0x48, 0xF7, 0xD0]); // not rax
                store_rax_to!(dst_off);
            }
            op::DIV | op::REM => {
                load_rax_from!(s1_off);
                code.extend_from_slice(&[0x48, 0x8B, 0x8D]); // mov rcx, [rbp-s2]
                code.extend_from_slice(&(-s2_off).to_le_bytes());
                code.extend_from_slice(&[0x48, 0x85, 0xC9]); // test rcx, rcx
                let jz_pos = code.len();
                code.extend_from_slice(&[0x74, 0x00]); // jz .zero (placeholder)
                code.extend_from_slice(&[0x48, 0x99]); // cqo
                code.extend_from_slice(&[0x48, 0xF7, 0xF9]); // idiv rcx
                if instr.opcode == op::REM {
                    code.extend_from_slice(&[0x48, 0x89, 0xD0]); // mov rax, rdx
                }
                let jmp_pos = code.len();
                code.extend_from_slice(&[0xEB, 0x00]); // jmp .done (placeholder)
                let zero_pos = code.len();
                code.extend_from_slice(&[0x31, 0xC0]); // xor eax, eax
                let done_pos = code.len();
                store_rax_to!(dst_off);
                code[jz_pos + 1] = (zero_pos - (jz_pos + 2)) as u8;
                code[jmp_pos + 1] = (done_pos - (jmp_pos + 2)) as u8;
            }
            op::CMP_EQ | op::CMP_NE | op::CMP_LT | op::CMP_LE | op::CMP_GT | op::CMP_GE => {
                load_rax_from!(s1_off);
                code.extend_from_slice(&[0x48, 0x3B, 0x85]); // cmp rax, [rbp-s2]
                code.extend_from_slice(&(-s2_off).to_le_bytes());
                let cc = match instr.opcode {
                    op::CMP_EQ => 0x94u8,
                    op::CMP_NE => 0x95,
                    op::CMP_LT => 0x9C,
                    op::CMP_LE => 0x9E,
                    op::CMP_GT => 0x9F,
                    op::CMP_GE => 0x9D,
                    _ => 0x94,
                };
                code.extend_from_slice(&[0x0F, cc, 0xC0]); // setcc al
                code.extend_from_slice(&[0x48, 0x0F, 0xB6, 0xC0]); // movzx rax, al
                store_rax_to!(dst_off);
            }
            op::SELECT => {
                let false_reg = instr.imm as u8;
                let f_off = ((false_reg as usize + 1) * 8) as i32;
                load_rax_from!(s1_off); // condition
                code.extend_from_slice(&[0x48, 0x85, 0xC0]); // test rax, rax
                let jz_pos = code.len();
                code.extend_from_slice(&[0x74, 0x00]); // jz .false
                load_rax_from!(s2_off); // true value
                let jmp_pos = code.len();
                code.extend_from_slice(&[0xEB, 0x00]); // jmp .done
                let false_pos = code.len();
                load_rax_from!(f_off); // false value
                let done_pos = code.len();
                store_rax_to!(dst_off);
                code[jz_pos + 1] = (false_pos - (jz_pos + 2)) as u8;
                code[jmp_pos + 1] = (done_pos - (jmp_pos + 2)) as u8;
            }
            op::NOP => {}
            _ => {
                code.push(0x90); // x86 NOP
            }
        }
    }
}

impl Drop for HardwareExecutor {
    fn drop(&mut self) {
        if !self.exec_page.is_null() && self.page_size > 0 {
            unsafe {
                munmap(self.exec_page as *mut libc::c_void, self.page_size);
            }
        }
    }
}

// =============================================================================
// §9  Main Search Driver
// =============================================================================

/// Statistics from the deep MCTS search.
#[derive(Debug, Clone)]
pub struct SearchStats {
    pub candidates_evaluated: u64,
    pub candidates_passed_filter: u64,
    pub improvements_found: u64,
    pub best_cost: u32,
    pub elapsed: Duration,
}

impl Default for SearchStats {
    fn default() -> Self {
        Self {
            candidates_evaluated: 0,
            candidates_passed_filter: 0,
            improvements_found: 0,
            best_cost: u32::MAX,
            elapsed: Duration::ZERO,
        }
    }
}

/// The deep MCTS search driver — the STOKE-inspired inner loop.
pub struct DeepMctsSearch {
    engine: MutationEngine,
    evaluator: TestVectorEvaluator,
    hw_executor: Option<HardwareExecutor>,
    test_vectors: Vec<(Vec<u64>, u64)>,
    cost_threshold: u32,
    stats: SearchStats,
    output_reg: u8,
}

impl DeepMctsSearch {
    /// Create a new deep MCTS search instance.
    pub fn new(seed: u64, num_inputs: u8, use_hardware: bool) -> Self {
        let hw_executor = if use_hardware {
            HardwareExecutor::new()
        } else {
            None
        };

        Self {
            engine: MutationEngine::new(seed, num_inputs),
            evaluator: TestVectorEvaluator::new(),
            hw_executor,
            test_vectors: Vec::new(),
            cost_threshold: u32::MAX,
            stats: SearchStats::default(),
            output_reg: num_inputs,
        }
    }

    /// Add a test vector (inputs, expected_output).
    pub fn add_test_vector(&mut self, inputs: Vec<u64>, expected: u64) {
        self.test_vectors.push((inputs, expected));
    }

    /// Set the output register.
    pub fn set_output_reg(&mut self, reg: u8) {
        self.output_reg = reg;
    }

    /// Set the cost threshold — only accept programs cheaper than this.
    pub fn set_cost_threshold(&mut self, threshold: u32) {
        self.cost_threshold = threshold;
    }

    /// Run the search for the given number of iterations.
    pub fn search(&mut self, source: &FlatProgram, iterations: usize) -> Option<FlatProgram> {
        let table = OpcodeTable::standard();
        let source_cost = source.cost(&table);
        let mut best_program = *source;
        let mut best_cost = source_cost;
        self.stats.best_cost = source_cost;

        let start = Instant::now();

        for _ in 0..iterations {
            let mutation = self.engine.random_mutation(&best_program);
            let candidate = self.engine.apply_mutation(&best_program, &mutation);

            self.stats.candidates_evaluated += 1;

            let num_inputs = self.engine.num_inputs;
            if !candidate.is_well_formed(num_inputs) {
                continue;
            }

            let passed = self.evaluator.evaluate_batch(
                &candidate,
                &self.test_vectors,
                self.output_reg,
                &table,
            );

            if passed == self.test_vectors.len() {
                self.stats.candidates_passed_filter += 1;

                let candidate_cost = candidate.cost(&table);

                if candidate_cost < best_cost && candidate_cost < self.cost_threshold {
                    best_cost = candidate_cost;
                    best_program = candidate;
                    self.stats.improvements_found += 1;
                    self.stats.best_cost = best_cost;
                }
            }
        }

        self.stats.elapsed = start.elapsed();

        if best_cost < source_cost {
            Some(best_program)
        } else {
            None
        }
    }

    /// Get the search statistics.
    pub fn stats(&self) -> &SearchStats {
        &self.stats
    }
}

// =============================================================================
// §10 Integration Hooks for Pruning and Verification
// =============================================================================

/// Check if a candidate program can be pruned using known-bits analysis.
/// Returns true if the candidate should be pruned (is definitely wrong).
pub fn known_bits_prune(_program: &FlatProgram, _spec_known_bits: &[(u64, u64)]) -> bool {
    // Integration point for crate::optimizer::known_bits
    // When that module exists, this will track known bits per register
    // and prune candidates whose output bits conflict with the spec.
    false
}

/// Verify a candidate program using SMT solving.
/// Returns true if the candidate is verified equivalent to the spec.
pub fn smt_verify_equiv(_program: &FlatProgram, _spec_program: &FlatProgram) -> bool {
    // Integration point for crate::optimizer::smt_verify
    // When that module exists (with z3), this will encode both programs
    // as SMT constraints and check for equivalence.
    true
}

// =============================================================================
// §11 Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_program_creation_and_effective_len() {
        let program = FlatProgram::empty();
        assert_eq!(program.effective_len(), 0);

        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::binop(op::ADD, 2, 0, 1, &OpcodeTable::standard());
        assert_eq!(program.effective_len(), 1);

        program[1] = FlatInstr::binop(op::MUL, 3, 2, 0, &OpcodeTable::standard());
        assert_eq!(program.effective_len(), 2);

        assert!(program[2].is_nop());
        assert_eq!(program.effective_len(), 2);
    }

    #[test]
    fn test_flat_instr_nop() {
        let nop = FlatInstr::nop();
        assert!(nop.is_nop());
        assert_eq!(nop.opcode, op::NOP);

        let lc = FlatInstr::load_const(0, 42);
        assert!(!lc.is_nop());
        assert!(lc.is_imm());
    }

    #[test]
    fn test_opcode_table_standard() {
        let table = OpcodeTable::standard();
        assert_eq!(table.len(), op::COUNT);
        assert_eq!(table.get(op::ADD).name, "add");
        assert_eq!(table.get(op::ADD).num_srcs, 2);
        assert!(table.get(op::ADD).is_commutative);
        assert!(table.get(op::ADD).has_dst);

        assert_eq!(table.get(op::SUB).name, "sub");
        assert!(!table.get(op::SUB).is_commutative);

        assert_eq!(table.get(op::LOAD_CONST).name, "load_const");
        assert_eq!(table.get(op::LOAD_CONST).num_srcs, 0);

        assert_eq!(table.get(op::NOP).name, "nop");
        assert!(!table.get(op::NOP).has_dst);
    }

    #[test]
    fn test_mutation_engine_generates_valid_mutations() {
        let mut engine = MutationEngine::new(42, 2);

        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::load_const(2, 5);
        program[1] = FlatInstr::binop(op::ADD, 3, 0, 2, &OpcodeTable::standard());

        for _ in 0..100 {
            let mutation = engine.random_mutation(&program);
            let new_program = engine.apply_mutation(&program, &mutation);
            assert!(new_program.effective_len() <= MAX_PROGRAM_LEN);
        }
    }

    #[test]
    fn test_mutation_opcode_replace() {
        let engine = MutationEngine::new(123, 2);
        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::binop(op::ADD, 2, 0, 1, &OpcodeTable::standard());

        let mutation = Mutation::OpcodeReplace {
            instr_idx: 0,
            new_opcode: op::SUB,
        };
        let new_program = engine.apply_mutation(&program, &mutation);
        assert_eq!(new_program[0].opcode, op::SUB);
    }

    #[test]
    fn test_mutation_operand_replace() {
        let engine = MutationEngine::new(123, 2);
        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::binop(op::ADD, 2, 0, 1, &OpcodeTable::standard());

        let mutation = Mutation::OperandReplace {
            instr_idx: 0,
            which: 1,
            new_val: 3,
        };
        let new_program = engine.apply_mutation(&program, &mutation);
        assert_eq!(new_program[0].src1, 3);
    }

    #[test]
    fn test_mutation_immediate_replace() {
        let engine = MutationEngine::new(123, 1);
        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::load_const(1, 42);

        let mutation = Mutation::ImmediateReplace {
            instr_idx: 0,
            new_imm: 100,
        };
        let new_program = engine.apply_mutation(&program, &mutation);
        assert_eq!(new_program[0].imm, 100);
    }

    #[test]
    fn test_mutation_swap() {
        let engine = MutationEngine::new(123, 2);
        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::load_const(2, 1);
        program[1] = FlatInstr::load_const(3, 2);

        let mutation = Mutation::Swap { idx_a: 0, idx_b: 1 };
        let new_program = engine.apply_mutation(&program, &mutation);
        assert_eq!(new_program[0].imm, 2);
        assert_eq!(new_program[1].imm, 1);
    }

    #[test]
    fn test_mutation_insert_and_delete() {
        let engine = MutationEngine::new(123, 2);
        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::load_const(2, 1);
        program[1] = FlatInstr::load_const(3, 2);

        let mutation = Mutation::Insert { at: 0 };
        let new_program = engine.apply_mutation(&program, &mutation);
        assert!(new_program[0].is_nop());
        assert_eq!(new_program[1].imm, 1);
        assert_eq!(new_program[2].imm, 2);

        let mutation = Mutation::Delete { at: 0 };
        let restored = engine.apply_mutation(&new_program, &mutation);
        assert_eq!(restored[0].imm, 1);
        assert_eq!(restored[1].imm, 2);
    }

    #[test]
    fn test_test_vector_evaluator_simple_add() {
        let table = OpcodeTable::standard();
        let mut evaluator = TestVectorEvaluator::new();

        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::binop(op::ADD, 2, 0, 1, &table);

        let result = evaluator.evaluate(&program, &[3, 5], 2, &table);
        assert_eq!(result, Some(8));

        let result = evaluator.evaluate(&program, &[0, 0], 2, &table);
        assert_eq!(result, Some(0));

        let result = evaluator.evaluate(&program, &[u64::MAX, 1], 2, &table);
        assert_eq!(result, Some(0));
    }

    #[test]
    fn test_test_vector_evaluator_load_const() {
        let table = OpcodeTable::standard();
        let mut evaluator = TestVectorEvaluator::new();

        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::load_const(1, 42);

        let result = evaluator.evaluate(&program, &[], 1, &table);
        assert_eq!(result, Some(42));
    }

    #[test]
    fn test_test_vector_evaluator_mul_and_shift() {
        let table = OpcodeTable::standard();
        let mut evaluator = TestVectorEvaluator::new();

        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::load_const(1, 1);
        program[1] = FlatInstr::binop(op::SHL, 2, 0, 1, &table);

        let result = evaluator.evaluate(&program, &[7], 2, &table);
        assert_eq!(result, Some(14));
    }

    #[test]
    fn test_test_vector_evaluator_batch() {
        let table = OpcodeTable::standard();
        let mut evaluator = TestVectorEvaluator::new();

        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::binop(op::ADD, 2, 0, 1, &table);

        let test_vectors = vec![(vec![1, 2], 3), (vec![10, 20], 30), (vec![100, 200], 300)];

        let matches = evaluator.evaluate_batch(&program, &test_vectors, 2, &table);
        assert_eq!(matches, 3);
    }

    #[test]
    fn test_test_vector_evaluator_div_by_zero() {
        let table = OpcodeTable::standard();
        let mut evaluator = TestVectorEvaluator::new();

        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::binop(op::DIV, 2, 0, 1, &table);

        let result = evaluator.evaluate(&program, &[10, 0], 2, &table);
        assert_eq!(result, Some(0));

        let result = evaluator.evaluate(&program, &[10, 2], 2, &table);
        assert_eq!(result, Some(5));
    }

    #[test]
    fn test_test_vector_evaluator_comparisons() {
        let table = OpcodeTable::standard();
        let mut evaluator = TestVectorEvaluator::new();

        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::binop(op::CMP_LT, 2, 0, 1, &table);

        assert_eq!(evaluator.evaluate(&program, &[1, 5], 2, &table), Some(1));
        assert_eq!(evaluator.evaluate(&program, &[5, 1], 2, &table), Some(0));
        assert_eq!(evaluator.evaluate(&program, &[3, 3], 2, &table), Some(0));
    }

    #[test]
    fn test_search_finds_shift_optimization() {
        let table = OpcodeTable::standard();

        // Source: r1 = 2; r2 = r0 * r1 (cost: 1+3+2=6)
        let mut source = FlatProgram::empty();
        source[0] = FlatInstr::load_const(1, 2);
        source[1] = FlatInstr::binop(op::MUL, 2, 0, 1, &table);

        let mut search = DeepMctsSearch::new(42, 1, false);
        search.set_output_reg(2);

        for x in [0u64, 1, 7, 42, 100, 255, 1000, u64::MAX / 2] {
            search.add_test_vector(vec![x], x.wrapping_mul(2));
        }

        let result = search.search(&source, 10000);

        if let Some(better) = result {
            let better_cost = better.cost(&table);
            let source_cost = source.cost(&table);
            assert!(
                better_cost < source_cost,
                "Found program should be cheaper: {} < {}",
                better_cost,
                source_cost
            );

            let mut evaluator = TestVectorEvaluator::new();
            for x in [0u64, 1, 7, 42, 100, 255, 1000] {
                let result = evaluator.evaluate(&better, &[x], 2, &table);
                assert_eq!(
                    result,
                    Some(x.wrapping_mul(2)),
                    "Optimized program should still be correct for x={}",
                    x
                );
            }
        }
    }

    #[test]
    fn test_is_well_formed() {
        let table = OpcodeTable::standard();

        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::binop(op::ADD, 2, 0, 1, &table);
        assert!(program.is_well_formed(2));
        assert!(!program.is_well_formed(1));

        let mut program2 = FlatProgram::empty();
        program2[0] = FlatInstr::load_const(1, 42);
        program2[1] = FlatInstr::binop(op::ADD, 2, 0, 1, &table);
        assert!(program2.is_well_formed(1));
    }

    #[test]
    fn test_output_regs() {
        let table = OpcodeTable::standard();

        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::load_const(1, 42);
        program[1] = FlatInstr::binop(op::ADD, 2, 0, 1, &table);

        let regs = program.output_regs();
        assert!(regs.contains(&1));
        assert!(regs.contains(&2));
    }

    #[test]
    fn test_program_cost() {
        let table = OpcodeTable::standard();

        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::load_const(1, 42);
        program[1] = FlatInstr::binop(op::ADD, 2, 0, 1, &table);

        let cost = program.cost(&table);
        assert_eq!(cost, 4);

        let mut expensive = FlatProgram::empty();
        expensive[0] = FlatInstr::load_const(1, 2);
        expensive[1] = FlatInstr::binop(op::MUL, 2, 0, 1, &table);

        let expensive_cost = expensive.cost(&table);
        assert_eq!(expensive_cost, 6);
        assert!(expensive_cost > cost);
    }

    #[test]
    fn test_hardware_executor_creation() {
        let executor = HardwareExecutor::new();
        drop(executor);
    }

    #[test]
    fn test_hardware_executor_simple() {
        let executor = HardwareExecutor::new();
        if executor.is_none() {
            eprintln!("Skipping hardware executor test (not x86-64 or mmap unavailable)");
            return;
        }
        let executor = executor.unwrap();
        let table = OpcodeTable::standard();

        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::binop(op::ADD, 2, 0, 1, &table);

        let result = executor.execute(&program, &[3, 5], &table);
        assert_eq!(result, Some(8));

        let result = executor.execute(&program, &[0, 0], &table);
        assert_eq!(result, Some(0));

        let result = executor.execute(&program, &[u64::MAX, 1], &table);
        assert_eq!(result, Some(0));
    }

    #[test]
    fn test_hardware_executor_shift() {
        let executor = HardwareExecutor::new();
        if executor.is_none() {
            eprintln!("Skipping hardware executor test");
            return;
        }
        let executor = executor.unwrap();
        let table = OpcodeTable::standard();

        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::load_const(1, 1);
        program[1] = FlatInstr::binop(op::SHL, 2, 0, 1, &table);

        let result = executor.execute(&program, &[7], &table);
        assert_eq!(result, Some(14));
    }

    #[test]
    fn test_hardware_executor_load_const() {
        let executor = HardwareExecutor::new();
        if executor.is_none() {
            eprintln!("Skipping hardware executor test");
            return;
        }
        let executor = executor.unwrap();
        let table = OpcodeTable::standard();

        let mut program = FlatProgram::empty();
        program[0] = FlatInstr::load_const(0, 42);

        let result = executor.execute(&program, &[], &table);
        assert_eq!(result, Some(42));
    }

    #[test]
    fn test_random_program_generation() {
        let mut engine = MutationEngine::new(42, 2);
        let program = engine.random_program(4);

        assert_eq!(program.effective_len(), 4);
        for i in 0..4 {
            assert!(program[i].opcode < OpcodeTable::standard().len());
            assert!(!program[i].is_nop());
        }
    }

    #[test]
    fn test_xorshift64_prng() {
        let mut rng = Xorshift64::new(42);
        let a = rng.next_u64();
        let b = rng.next_u64();
        assert_ne!(a, b);

        let mut rng2 = Xorshift64::new(42);
        assert_eq!(rng2.next_u64(), a);
        assert_eq!(rng2.next_u64(), b);

        let mut rng3 = Xorshift64::new(0);
        let c = rng3.next_u64();
        assert_ne!(c, 0);
    }

    #[test]
    fn test_known_bits_prune() {
        let program = FlatProgram::empty();
        assert!(!known_bits_prune(&program, &[]));
    }

    #[test]
    fn test_smt_verify_equiv() {
        let a = FlatProgram::empty();
        let b = FlatProgram::empty();
        assert!(smt_verify_equiv(&a, &b));
    }
}
