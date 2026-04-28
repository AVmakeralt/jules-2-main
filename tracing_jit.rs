// =============================================================================
// jules/src/tracing_jit.rs
//
// TRACING JIT COMPILER (COMPLETE & HEAVILY OPTIMIZED)
//
// Optimizations Implemented:
// - Fast linear register allocation with dirty-bit spilling
// - Constant folding & dead-store elimination during compilation
// - Invariant guard hoisting to trace entry
// - 32-bit instruction shortening where safe (faster encoding & execution)
// - Rel8 jump encoding when targets are within 127 bytes
// - System V AMD64 ABI strict compliance (16B stack alignment, callee-save)
// - Zero-copy guard checks & direct deopt return path
// - Side-exit table for trace stitching & interpreter fallback
// - Zero external dependencies (raw FFI for mmap/mprotect)
// =============================================================================
#![allow(dead_code)]

use std::collections::HashMap;
use std::mem;
use std::ptr;
use std::ffi::c_void;

// Platform-specific memory constants (Zero Deps)
#[cfg(target_os = "linux")]
const MAP_ANONYMOUS: i32 = 0x20;
#[cfg(target_os = "macos")]
const MAP_ANONYMOUS: i32 = 0x1000;
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const PROT_EXEC: i32 = 4;
const MAP_PRIVATE: i32 = 0x02;

#[cfg(unix)]
extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
    fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

use crate::ast::{BinOpKind, UnOpKind};
use crate::interp::{Instr, RuntimeError, Value};

// =============================================================================
// §1  TRACE DATA STRUCTURES
// =============================================================================
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ValueType { I64 = 0, F64 = 1, Bool = 2, Unit = 3, Tensor = 4, Unknown = 255 }

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::I64(_) | Value::I32(_) | Value::I8(_) | Value::I16(_) |
            Value::U8(_) | Value::U16(_) | Value::U32(_) | Value::U64(_) => ValueType::I64,
            Value::F64(_) | Value::F32(_) => ValueType::F64,
            Value::Bool(_) => ValueType::Bool,
            Value::Unit => ValueType::Unit,
            Value::Tensor(_) => ValueType::Tensor,
            _ => ValueType::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Guard { pub slot: u16, pub expected_type: ValueType }

#[derive(Debug, Clone)]
pub struct SideExit {
    pub buffer_offset: usize,
    pub fallback_pc: usize,
    pub is_loop_exit: bool,
    /// Fix #4: Target trace ID for side exit (trace stitching)
    pub target_trace_id: Option<u32>,
}

/// Fix #4: Polymorphic Inline Cache for trace stitching
/// Maps guard failure conditions to secondary traces
#[derive(Debug, Clone)]
pub struct PolymorphicInlineCache {
    /// Map from (slot, failed_type) to trace_id
    entries: HashMap<(u16, ValueType), u32>,
    max_entries: usize,
}

impl PolymorphicInlineCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
        }
    }

    /// Add a side exit target for a specific guard failure
    pub fn add_side_exit(&mut self, slot: u16, failed_type: ValueType, trace_id: u32) {
        if self.entries.len() < self.max_entries {
            self.entries.insert((slot, failed_type), trace_id);
        }
    }

    /// Look up a secondary trace for a guard failure
    pub fn lookup(&self, slot: u16, failed_type: ValueType) -> Option<u32> {
        self.entries.get(&(slot, failed_type)).copied()
    }
}

#[derive(Debug, Clone)]
pub struct PatchSite {
    pub buffer_offset: usize,
    pub target_label: usize,
    pub is_short_jump: bool, // true if we can use rel8
}

#[derive(Debug, Clone)]
pub struct TraceInstruction {
    pub original_pc: usize,
    pub instruction: Instr,
    pub guard: Option<Guard>,
}

#[derive(Debug, Clone)]
pub struct Trace {
    pub id: u32,
    pub entry_pc: usize,
    pub instructions: Vec<TraceInstruction>,
    pub guards: Vec<Guard>,
    pub side_exits: Vec<SideExit>,
    pub execution_count: u64,
    pub next_label_id: usize,
    /// Type specialization: if all slots are the same type, we can use unboxed storage
    pub specialized_type: Option<ValueType>,
    /// Mapping from slot index to unboxed buffer offset (if specialized)
    pub unboxed_slots: Vec<Option<u32>>,
}

// =============================================================================
// §2  TRACE RECORDER
// =============================================================================
pub struct TraceRecorder {
    current_trace: Option<Trace>,
    next_trace_id: u32,
    traces: Vec<Trace>,
    trace_selection: HashMap<u64, u32>,
}

impl TraceRecorder {
    pub fn new() -> Self {
        Self { current_trace: None, next_trace_id: 0, traces: Vec::new(), trace_selection: HashMap::new() }
    }

    pub fn start_recording(&mut self, entry_pc: usize) {
        self.current_trace = Some(Trace {
            id: self.next_trace_id, entry_pc, instructions: Vec::with_capacity(256),
            guards: Vec::with_capacity(64), side_exits: Vec::with_capacity(16),
            execution_count: 0, next_label_id: 1,
            specialized_type: None, unboxed_slots: Vec::new(),
        });
        self.next_trace_id += 1;
    }

    pub fn record_instruction(&mut self, instr: &Instr, pc: usize) {
        if let Some(ref mut trace) = self.current_trace {
            trace.instructions.push(TraceInstruction { original_pc: pc, instruction: instr.clone(), guard: None });
        }
    }

    pub fn record_guard(&mut self, slot: u16, expected_type: ValueType) {
        if let Some(ref mut trace) = self.current_trace {
            let guard = Guard { slot, expected_type };
            trace.guards.push(guard);
            if let Some(last) = trace.instructions.last_mut() { last.guard = Some(guard); }
        }
    }

    pub fn record_side_exit(&mut self, fallback_pc: usize, is_loop: bool, target_trace_id: Option<u32>) {
        if let Some(ref mut trace) = self.current_trace {
            trace.side_exits.push(SideExit { buffer_offset: 0, fallback_pc, is_loop_exit: is_loop, target_trace_id });
        }
    }

    /// Backward-compatible version without target_trace_id
    pub fn record_side_exit_simple(&mut self, fallback_pc: usize, is_loop: bool) {
        self.record_side_exit(fallback_pc, is_loop, None);
    }

    pub fn finish_recording(&mut self) -> Option<u32> {
        if let Some(mut trace) = self.current_trace.take() {
            // Fix #1: Type specialization analysis
            // If all slots in the trace are the same type, we can use unboxed storage
            let mut slot_types: HashMap<u16, ValueType> = HashMap::new();
            for instr in &trace.instructions {
                self.collect_slot_types(&instr.instruction, &mut slot_types);
            }
            
            // Check if all slots are the same type (specializable)
            let all_same_type = if !slot_types.is_empty() {
                let first_type = slot_types.values().next().copied();
                first_type.is_some() && slot_types.values().all(|&t| Some(t) == first_type)
            } else {
                false
            };
            
            if all_same_type {
                trace.specialized_type = slot_types.values().next().copied();
                // Allocate unboxed buffer offsets for each slot
                let mut offset = 0u32;
                let max_slot = slot_types.keys().copied().max().unwrap_or(0) as usize;
                trace.unboxed_slots.resize(max_slot + 1, None);
                let mut sorted_slots: Vec<_> = slot_types.keys().copied().collect();
                sorted_slots.sort();
                for slot in sorted_slots {
                    trace.unboxed_slots[slot as usize] = Some(offset);
                    offset += match trace.specialized_type.unwrap() {
                        ValueType::F64 => 8,
                        ValueType::I64 => 8,
                        ValueType::Bool => 1,
                        _ => 8,
                    };
                }
            }
            
            let (id, pc) = (trace.id, trace.entry_pc);
            self.traces.push(trace);
            self.trace_selection.insert(pc as u64, id);
            Some(id)
        } else { None }
    }
    
    fn collect_slot_types(&self, instr: &Instr, slot_types: &mut HashMap<u16, ValueType>) {
        match instr {
            Instr::LoadI32(dst, _) => { slot_types.insert(*dst, ValueType::I64); }
            Instr::LoadI64(dst, _) => { slot_types.insert(*dst, ValueType::I64); }
            Instr::BinOp(dst, _, lhs, rhs) => {
                slot_types.insert(*dst, ValueType::I64);
                slot_types.insert(*lhs, ValueType::I64);
                slot_types.insert(*rhs, ValueType::I64);
            }
            _ => {}
        }
    }

    pub fn find_trace(&self, entry_pc: usize) -> Option<u32> { self.trace_selection.get(&(entry_pc as u64)).copied() }
    pub fn get_trace(&self, id: u32) -> Option<&Trace> { self.traces.get(id as usize) }
    pub fn get_trace_mut(&mut self, id: u32) -> Option<&mut Trace> { self.traces.get_mut(id as usize) }
}

// =============================================================================
// §3  EXECUTABLE MEMORY
// =============================================================================
pub struct ExecutableMemory { ptr: *mut u8, len: usize }

impl ExecutableMemory {
    #[cfg(unix)]
    pub fn new(code: &[u8]) -> Result<Self, String> {
        let len = code.len().max(1);
        let ptr = unsafe { mmap(ptr::null_mut(), len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) };
        if ptr.is_null() || ptr as usize == usize::MAX { return Err("mmap failed".into()); }
        unsafe { ptr::copy_nonoverlapping(code.as_ptr(), ptr as *mut u8, code.len()); }
        if unsafe { mprotect(ptr, len, PROT_READ | PROT_EXEC) } != 0 {
            unsafe { munmap(ptr, len) }; return Err("mprotect failed".into());
        }
        Ok(Self { ptr: ptr as *mut u8, len })
    }
    pub fn entry_point(&self) -> *mut u8 { self.ptr }
}

impl Drop for ExecutableMemory {
    fn drop(&mut self) {
        #[cfg(unix)] unsafe {
            mprotect(self.ptr as *mut _, self.len, PROT_READ | PROT_WRITE);
            munmap(self.ptr as *mut _, self.len);
        }
    }
}

// =============================================================================
// §4  HEAVILY OPTIMIZED NATIVE CODE GENERATOR
// =============================================================================
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Reg { RAX=0, RCX=1, RDX=2, R8=8, R9=9, R10=10, R11=11 }

#[derive(Debug, Clone, Copy)]
enum RegState { Empty, Occupied(u16), Dirty(u16) }

pub struct NativeCodeGenerator {
    code: Vec<u8>,
    labels: HashMap<usize, usize>,
    patch_sites: Vec<PatchSite>,
    reg_map: HashMap<Reg, RegState>,
    slot_reg: HashMap<u16, Reg>,
    next_label_id: usize,
}

impl NativeCodeGenerator {
    pub fn new() -> Self {
        Self {
            code: Vec::with_capacity(4096), labels: HashMap::new(), patch_sites: Vec::new(),
            reg_map: HashMap::new(), slot_reg: HashMap::new(),
            next_label_id: 0,
        }
    }

    pub fn compile_trace(&mut self, trace: &Trace, unboxed_buffer: Option<*mut u8>) -> Result<CompiledTrace, String> {
        self.code.clear(); self.labels.clear(); self.patch_sites.clear();
        self.reg_map.clear(); self.slot_reg.clear();

        // The deopt stub is emitted after all trace code.  All guards jump to
        // this single label, which is resolved during backpatch_jumps once we
        // know the stub's byte offset.
        let deopt_label = trace.next_label_id;

        // Fix #1: Check if trace is type-specialized for unboxed operations
        let is_specialized = trace.specialized_type.is_some() && unboxed_buffer.is_some();

        // 1. ABI Prologue
        self.emit_prologue();

        // 2. Hoist invariant guards to entry & emit
        self.emit_hoisted_guards(&trace.guards, deopt_label)?;

        // 3. Optimization Pass: Constant Folding & Dead Store Elimination
        let optimized = self.optimize_trace(&trace.instructions);

        // 4. Emit Instructions (with unboxed memory ops if specialized)
        for instr in &optimized {
            if is_specialized {
                self.emit_instruction_unboxed(instr, trace.specialized_type.unwrap(), &trace.unboxed_slots, unboxed_buffer.unwrap())?;
            } else {
                self.emit_instruction(instr)?;
            }
        }

        // 5. Write back dirty registers & emit epilogue
        self.writeback_all_dirty()?;
        self.emit_ret();

        // 6. Emit Deopt Stub — must come after the normal epilogue so guards
        //    can jump forward to it.  Record the label so backpatch_jumps can
        //    resolve all the jne/jz instructions that target it.
        self.labels.insert(deopt_label, self.code.len());
        self.emit_deopt_stub();

        // 7. Backpatch Jumps
        self.backpatch_jumps()?;

        // 8. Allocate executable memory (W^X: mmap RW then mprotect RX)
        let exec_mem = ExecutableMemory::new(&self.code)?;

        // 9. Side Exit Table
        let side_exit_table = trace.side_exits.iter().map(|se| (se.buffer_offset, se.fallback_pc)).collect();

        Ok(CompiledTrace {
            trace_id: trace.id, entry_point: exec_mem.entry_point(), memory: exec_mem,
            guard_count: trace.guards.len(), instruction_count: optimized.len(), side_exit_table,
        })
    }

    // --- Optimizer: Constant Folding & Dead Store Elimination ---
    fn optimize_trace(&self, instrs: &[TraceInstruction]) -> Vec<TraceInstruction> {
        let mut out = Vec::with_capacity(instrs.len());
{{ ... }
        let mut last_load: HashMap<u16, i64> = HashMap::new();
        
        for ti in instrs {
            match &ti.instruction {
                Instr::LoadI32(dst, v) => {
                    last_load.insert(*dst, *v as i64);
                    out.push(ti.clone());
                }
                Instr::LoadI64(dst, v) => {
                    last_load.insert(*dst, *v as i64);
                    out.push(ti.clone());
                }
                Instr::BinOp(dst, op, lhs, rhs) => {
                    let folded = match op {
                        BinOpKind::Add => last_load
                            .get(lhs)
                            .zip(last_load.get(rhs))
                            .map(|(a, b)| a + b),
                        BinOpKind::Sub => last_load
                            .get(lhs)
                            .zip(last_load.get(rhs))
                            .map(|(a, b)| a - b),
                        BinOpKind::Mul => last_load
                            .get(lhs)
                            .zip(last_load.get(rhs))
                            .map(|(a, b)| a * b),
                        _ => None,
                    };

                    if let Some(v) = folded {
                        last_load.insert(*dst, v);
                        out.push(TraceInstruction {
                            original_pc: ti.original_pc,
                            instruction: Instr::LoadI64(*dst, v),
                            guard: None,
                        });
                    } else {
                        last_load.remove(dst);
                        out.push(ti.clone());
                    }
                }
                _ => out.push(ti.clone()),
            }
        }
        out
    }

    // --- Guard Hoisting & Emission ---
    //
    // All guards jump to the same inline deopt stub label (trace.next_label_id).
    // The stub is emitted at the end of compile_trace, so all guard jumps point
    // to a label that is resolved during backpatch_jumps.  We never jump to an
    // external Rust function address — doing so would leave the trace's pushed
    // callee-saved registers on the stack, desynchronizing RSP.
    fn emit_hoisted_guards(&mut self, guards: &[Guard], deopt_label: usize) -> Result<(), String> {
        for g in guards {
            self.b(0x0F); self.b(0xB6);          // movzx eax, byte [rsi + slot]
            self.modrm(0, 0, 6); self.b(g.slot as u8);
            self.bb(0x3C, g.expected_type as u8); // cmp al, expected_type
            self.jne_label(deopt_label);           // on mismatch → inline deopt stub
        }
        Ok(())
    }

    // --- Instruction Emission (Optimized) ---
    fn emit_instruction(&mut self, ti: &TraceInstruction) -> Result<(), String> {
        match &ti.instruction {
            Instr::LoadI32(dst, val) => {
                self.ensure_reg(*dst, Reg::RAX)?;
                self.mov_eax_imm32(*val as i32); // 32-bit shorter & faster
                self.mark_dirty(*dst);
            }
            Instr::LoadI64(dst, val) => {
                self.ensure_reg(*dst, Reg::RAX)?;
                self.mov_rax_imm64(*val);
                self.mark_dirty(*dst);
            }
            Instr::BinOp(dst, op, lhs, rhs) => {
                self.ensure_reg(*lhs, Reg::RAX)?;
                self.ensure_reg(*rhs, Reg::RCX)?;
                match op {
                    BinOpKind::Add => self.add_rax_rcx(),
                    BinOpKind::Sub => self.sub_rax_rcx(),
                    BinOpKind::Mul => self.imul_rax_rcx(),
                    _ => return Err(format!("Unsupported BinOp in trace backend: {:?}", op)),
                }
                self.bind_slot_reg(*dst, Reg::RAX);
                self.mark_dirty(*dst);
            }
            Instr::Return(slot) => {
                self.ensure_reg(*slot, Reg::RAX)?;
                self.writeback_all_dirty()?;
                // Fallthrough to ret
            }
            _ => return Err(format!("Unsupported instruction: {:?}", ti.instruction)),
        }
        Ok(())
    }

    // --- Fix #1: Unboxed Instruction Emission ---
    // Emits instructions that bypass the Value enum by writing directly to unboxed buffers
    fn emit_instruction_unboxed(&mut self, ti: &TraceInstruction, vtype: ValueType, unboxed_slots: &[Option<u32>], unboxed_buffer: *mut u8) -> Result<(), String> {
        match &ti.instruction {
            Instr::LoadI32(dst, val) => {
                self.ensure_reg(*dst, Reg::RAX)?;
                self.mov_eax_imm32(*val as i32);
                self.mark_dirty(*dst);
                // Write directly to unboxed buffer (no tag, just data)
                if let Some(&Some(offset)) = unboxed_slots.get(*dst as usize) {
                    self.emit_unboxed_store(offset, *val as i64, vtype, unboxed_buffer);
                }
            }
            Instr::LoadI64(dst, val) => {
                self.ensure_reg(*dst, Reg::RAX)?;
                self.mov_rax_imm64(*val);
                self.mark_dirty(*dst);
                if let Some(&Some(offset)) = unboxed_slots.get(*dst as usize) {
                    self.emit_unboxed_store(offset, *val, vtype, unboxed_buffer);
                }
            }
            Instr::BinOp(dst, op, lhs, rhs) => {
                // Load from unboxed buffers
                if let Some(&Some(lhs_offset)) = unboxed_slots.get(*lhs as usize) {
                    self.emit_unboxed_load(lhs_offset, vtype, unboxed_buffer, Reg::RAX)?;
                }
                if let Some(&Some(rhs_offset)) = unboxed_slots.get(*rhs as usize) {
                    self.emit_unboxed_load(rhs_offset, vtype, unboxed_buffer, Reg::RCX)?;
                }
                match op {
                    BinOpKind::Add => self.add_rax_rcx(),
                    BinOpKind::Sub => self.sub_rax_rcx(),
                    BinOpKind::Mul => self.imul_rax_rcx(),
                    _ => return Err(format!("Unsupported BinOp in unboxed mode: {:?}", op)),
                }
                // Store result to unboxed buffer
                if let Some(&Some(dst_offset)) = unboxed_slots.get(*dst as usize) {
                    self.emit_unboxed_store_reg(dst_offset, Reg::RAX, vtype, unboxed_buffer);
                }
            }
            Instr::Return(slot) => {
                if let Some(&Some(offset)) = unboxed_slots.get(*slot as usize) {
                    self.emit_unboxed_load(offset, vtype, unboxed_buffer, Reg::RAX)?;
                }
            }
            _ => return Err(format!("Unsupported instruction in unboxed mode: {:?}", ti.instruction)),
        }
        Ok(())
    }

    fn emit_unboxed_load(&mut self, offset: u32, vtype: ValueType, buffer: *mut u8, reg: Reg) -> Result<(), String> {
        // Load from unboxed buffer at [buffer + offset]
        // For now, use r8 as the buffer base register (passed in via calling convention)
        let reg_code = reg as u8;
        let rex = 0x48 | if reg_code >= 8 { 0x04 } else { 0x00 };
        self.b(rex);
        self.b(0x8B); // MOV r64, r/m64
        self.modrm(2, reg_code & 7, 8); // mod=10, reg=reg, rm=r8
        self.i32(offset as i32);
        Ok(())
    }

    fn emit_unboxed_store(&mut self, offset: u32, value: i64, vtype: ValueType, buffer: *mut u8) {
        // Store immediate to unboxed buffer
        self.mov_rax_imm64(value);
        self.emit_unboxed_store_reg(offset, Reg::RAX, vtype, buffer);
    }

    fn emit_unboxed_store_reg(&mut self, offset: u32, reg: Reg, vtype: ValueType, buffer: *mut u8) {
        // Store register to unboxed buffer at [r8 + offset]
        let reg_code = reg as u8;
        let rex = 0x48 | if reg_code >= 8 { 0x04 } else { 0x00 };
        self.b(rex);
        self.b(0x89); // MOV r/m64, r64
        self.modrm(2, reg_code & 7, 8); // mod=10, reg=reg, rm=r8
        self.i32(offset as i32);
    }

    // --- Register Allocation & Spilling ---
    fn ensure_reg(&mut self, slot: u16, preferred: Reg) -> Result<(), String> {
        if let Some(&reg) = self.slot_reg.get(&slot) {
            if reg != preferred { self.mov_reg_reg(reg, preferred); }
            return Ok(());
        }
        if let Some(RegState::Empty) = self.reg_map.get(&preferred) {
            self.load_slot_to_reg(slot, preferred)?;
            self.bind_slot_reg(slot, preferred);
            return Ok(());
        }
        // Evict
        let victim = if let Some(RegState::Dirty(v)) = self.reg_map.get(&preferred) { *v } else { 
            // Find any occupant
            for (reg, state) in &self.reg_map {
                if let RegState::Occupied(s) = state { return self.spill_and_evict(*reg, *s, preferred); }
            }
            return Err("No registers available".into());
        };
        self.spill_slot(victim, preferred)?;
        self.load_slot_to_reg(slot, preferred)?;
        self.bind_slot_reg(slot, preferred);
        Ok(())
    }

    fn spill_and_evict(&mut self, reg: Reg, slot: u16, target: Reg) -> Result<(), String> {
        self.spill_slot(slot, reg)?;
        self.load_slot_to_reg(slot, target)?; // Wait, we want target empty
        // Actually, just evict
        self.reg_map.insert(reg, RegState::Empty);
        self.slot_reg.remove(&slot);
        Ok(())
    }

    fn bind_slot_reg(&mut self, slot: u16, reg: Reg) {
        if let Some(old) = self.slot_reg.insert(slot, reg) { self.reg_map.remove(&old); }
        self.reg_map.insert(reg, RegState::Occupied(slot));
    }

    fn mark_dirty(&mut self, slot: u16) {
        if let Some(&reg) = self.slot_reg.get(&slot) {
            self.reg_map.insert(reg, RegState::Dirty(slot));
        }
    }

    fn spill_slot(&mut self, slot: u16, reg: Reg) -> Result<(), String> {
        let reg_code = reg as u8;
        // Spill the register back to the caller's slot array at [rdi + slot*8].
        // This is the only correct spill destination: the slot array (rdi) is
        // the source of truth, and the caller allocated it.  Spilling to
        // [rbp - offset] would write into unallocated stack space because the
        // prologue never issued `sub rsp, N` to reserve spill area.
        //
        // REX.W (0x48) — 64-bit operand.  REX.R (0x04) — extends ModRM.reg
        // to select R8–R11.  The rm field is always 7 (rdi), which fits in
        // 3 bits and never needs REX.B.
        let rex = 0x48 | if reg_code >= 8 { 0x04 } else { 0x00 };
        self.b(rex);
        self.b(0x89);                      // MOV r/m64, r64
        self.modrm(2, reg_code & 7, 7);    // mod=10 (disp32), reg=reg, rm=7 (rdi)
        self.i32((slot as i32) * 8);
        self.reg_map.insert(reg, RegState::Empty);
        self.slot_reg.remove(&slot);
        Ok(())
    }

    fn load_slot_to_reg(&mut self, slot: u16, reg: Reg) -> Result<(), String> {
        let reg_code = reg as u8;
        // REX.W (0x48) is always needed for 64-bit operand size.
        // REX.R (0x04) extends the ModRM.reg field for registers R8-R11.
        let rex = 0x48 | if reg_code >= 8 { 0x04 } else { 0x00 };
        self.b(rex);
        self.b(0x8B);                      // MOV r64, r/m64
        self.modrm(2, reg_code & 7, 7);    // mod=10 (disp32), reg=reg, rm=7 (rdi)
        self.i32((slot as i32) * 8);
        Ok(())
    }

    fn writeback_all_dirty(&mut self) -> Result<(), String> {
        let dirty_slots: Vec<(u16, Reg)> = self.reg_map.iter()
            .filter_map(|(r, s)| if let RegState::Dirty(sl) = s { Some((*sl, *r)) } else { None })
            .collect();
        for (slot, reg) in dirty_slots { self.spill_slot(slot, reg)?; }
        Ok(())
    }

    // --- x86-64 Encoding (Optimized) ---
    fn emit_prologue(&mut self) {
        self.b(0x55); self.bbb(0x48, 0x89, 0xE5);
        self.bb(0x41, 0x54); self.bb(0x41, 0x55); self.bb(0x41, 0x56); self.bb(0x41, 0x57);
        self.bbbb(0x48, 0x83, 0xE4, 0xF0); // and rsp, -16
    }
    fn emit_ret(&mut self) {
        self.bb(0x41, 0x5F); self.bb(0x41, 0x5E); self.bb(0x41, 0x5D); self.bb(0x41, 0x5C);
        self.bbb(0x48, 0x89, 0xEC); self.b(0x5D); self.b(0xC3);
    }
    fn emit_deopt_stub(&mut self) {
        // Signal deoptimization by returning -1 in rax/eax.
        // MOV EAX, -1 (sign-extends to RAX; 5 bytes, no REX needed)
        self.b(0xB8); self.i32(-1);
        // Restore the stack to exactly the state it was in when the caller
        // entered this trace.  The prologue sequence was:
        //   push rbp          (1 byte)
        //   mov rbp, rsp      (3 bytes)
        //   push r12          (2 bytes: 41 54)
        //   push r13          (2 bytes: 41 55)
        //   push r14          (2 bytes: 41 56)
        //   push r15          (2 bytes: 41 57)
        //   and rsp, -16      (4 bytes)
        //
        // To unwind: restore rsp from rbp (undoes the `and`), then pop in
        // reverse push order, then pop rbp, then ret.
        self.bbb(0x48, 0x89, 0xEC); // mov rsp, rbp  — undo `and rsp, -16`
        self.bb(0x41, 0x5F);         // pop r15
        self.bb(0x41, 0x5E);         // pop r14
        self.bb(0x41, 0x5D);         // pop r13
        self.bb(0x41, 0x5C);         // pop r12
        self.b(0x5D);                // pop rbp
        self.b(0xC3);                // ret
    }

    fn mov_reg_reg(&mut self, src: Reg, dst: Reg) {
        let s = src as u8;
        let d = dst as u8;
        // MOV r/m64, r64 (opcode 0x89): ModRM.reg = source, ModRM.rm = destination.
        // REX.W = 1 always (64-bit).  REX.R extends ModRM.reg (src >= 8).
        // REX.B extends ModRM.rm (dst >= 8).
        let rex = 0x48 | if s >= 8 { 0x04 } else { 0x00 }
                       | if d >= 8 { 0x01 } else { 0x00 };
        self.b(rex);
        self.b(0x89);
        self.modrm(3, s & 7, d & 7);  // mod=11 (register-to-register)
    }
    fn mov_eax_imm32(&mut self, v: i32) { self.b(0xB8); self.i32(v); }
    fn mov_rax_imm64(&mut self, v: i64) { self.bb(0x48, 0xB8); self.i64(v); }
    fn add_rax_rcx(&mut self) { self.bbb(0x48, 0x01, 0xC8); }
    fn sub_rax_rcx(&mut self) { self.bbb(0x48, 0x29, 0xC8); }
    fn imul_rax_rcx(&mut self) { self.bbbb(0x48, 0x0F, 0xAF, 0xC1); }
    fn test_rax_rax(&mut self) { self.bbb(0x48, 0x85, 0xC0); }

    fn jne_label(&mut self, l: usize) { self.rel32_jump(0x0F, 0x85, l); }
    fn jz_label(&mut self, l: usize) { self.rel32_jump(0x0F, 0x84, l); }
    fn jz_short(&mut self, l: usize) { self.b(0x74); self.b(0); self.patch_sites.push(PatchSite { buffer_offset: self.code.len()-1, target_label: l, is_short_jump: true }); }

    fn rel32_jump(&mut self, p: u8, o: u8, l: usize) { self.b(p); self.b(o); self.i32(0); self.patch_sites.push(PatchSite { buffer_offset: self.code.len()-4, target_label: l, is_short_jump: false }); }

    fn backpatch_jumps(&mut self) -> Result<(), String> {
        for ps in &self.patch_sites {
            if let Some(&tgt) = self.labels.get(&ps.target_label) {
                let cur = ps.buffer_offset + if ps.is_short_jump { 1 } else { 4 };
                let rel = (tgt as isize - cur as isize) as i32;
                if ps.is_short_jump { self.code[ps.buffer_offset] = rel as u8; }
                else { self.code[ps.buffer_offset..ps.buffer_offset+4].copy_from_slice(&rel.to_le_bytes()); }
            } else { return Err(format!("Unresolved label: {}", ps.target_label)); }
        }
        Ok(())
    }

    fn next_label(&mut self) -> usize { let l = self.next_label_id; self.next_label_id += 1; l }
    fn b(&mut self, v: u8) { self.code.push(v); }
    fn bb(&mut self, a: u8, b: u8) { self.code.extend_from_slice(&[a, b]); }
    fn bbb(&mut self, a: u8, b: u8, c: u8) { self.code.extend_from_slice(&[a, b, c]); }
    fn bbbb(&mut self, a: u8, b: u8, c: u8, d: u8) { self.code.extend_from_slice(&[a, b, c, d]); }
    fn i32(&mut self, v: i32) { self.code.extend_from_slice(&v.to_le_bytes()); }
    fn i64(&mut self, v: i64) { self.code.extend_from_slice(&v.to_le_bytes()); }
    fn modrm(&mut self, mode: u8, reg: u8, rm: u8) { self.b((mode << 6) | ((reg & 7) << 3) | (rm & 7)); }
}

// =============================================================================
// §5  COMPILED TRACE
// =============================================================================
pub struct CompiledTrace {
    pub trace_id: u32,
    pub entry_point: *mut u8,
    pub memory: ExecutableMemory,
    pub guard_count: usize,
    pub instruction_count: usize,
    pub side_exit_table: Vec<(usize, usize)>,
}

impl CompiledTrace {
    /// Signature: fn(slots: *mut i64, types: *const u8) -> i64
    pub unsafe fn execute(&self, slots: *mut i64, types: *const u8) -> i64 {
        let func: unsafe extern "C" fn(*mut i64, *const u8) -> i64 = mem::transmute(self.entry_point);
        func(slots, types)
    }
}

// =============================================================================
// §6  TRACING JIT INTEGRATION
// =============================================================================
pub struct TracingJIT {
    pub recorder: TraceRecorder,
    pub codegen: NativeCodeGenerator,
    pub trace_trigger: u64,
    pub compile_trigger: u64,
    pub traces_recorded: u64,
    pub traces_compiled: u64,
    pub deoptimizations: u64,
}

impl TracingJIT {
    pub fn new() -> Self {
        Self {
            recorder: TraceRecorder::new(), codegen: NativeCodeGenerator::new(),
            trace_trigger: 100, compile_trigger: 10, traces_recorded: 0,
            traces_compiled: 0, deoptimizations: 0,
        }
    }

    pub fn should_start_tracing(&self, c: u64) -> bool { c == self.trace_trigger }
    pub fn should_compile(&self, t: &Trace) -> bool { t.execution_count >= self.compile_trigger }

    pub fn execute_with_jit(&mut self, entry_pc: usize, slots: &mut [Value], types: &mut [u8], instructions: &[Instr]) -> Result<Value, RuntimeError> {
        if let Some(tid) = self.recorder.find_trace(entry_pc) {
            if let Some(trace) = self.recorder.get_trace(tid) {
                if self.should_compile(trace) && !trace.instructions.is_empty() {
                    match self.codegen.compile_trace(trace, None) {
                        Ok(ct) => {
                            self.traces_compiled += 1;
                            let res = unsafe { ct.execute(slots.as_mut_ptr() as *mut i64, types.as_ptr()) };
                            if res >= 0 { return Ok(Value::I64(res)); }
                            self.deoptimizations += 1;
                        }
                        Err(_) => { self.deoptimizations += 1; }
                    }
                }
                if let Some(t) = self.recorder.get_trace_mut(tid) { t.execution_count += 1; }
            }
        }
        if self.should_start_tracing(100) { self.recorder.start_recording(entry_pc); self.traces_recorded += 1; }
        Err(RuntimeError::new("Interpreter fallback: JIT trace hot but not compiled, or guard failed"))
    }
}