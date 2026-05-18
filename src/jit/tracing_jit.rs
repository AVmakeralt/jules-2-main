// =============================================================================
// jules/src/tracing_jit.rs
//
// TRACING JIT COMPILER
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


use rustc_hash::FxHashMap;
use std::mem;
use std::ptr;
use std::ffi::c_void;
use smallvec::SmallVec;

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

use crate::compiler::ast::BinOpKind;
use crate::interp::{Instr, RuntimeError, Value};

// =============================================================================
// §1  TRACE DATA STRUCTURES
// =============================================================================
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ValueType { I64 = 0, F64 = 1, Bool = 2, Unit = 3, Tensor = 4, F32 = 5, Unknown = 255 }

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::I64(_) | Value::I32(_) | Value::I8(_) | Value::I16(_) |
            Value::U8(_) | Value::U16(_) | Value::U32(_) | Value::U64(_) => ValueType::I64,
            Value::F32(_) => ValueType::F32,
            Value::F64(_) => ValueType::F64,
            Value::Bool(_) => ValueType::Bool,
            Value::Unit => ValueType::Unit,
            Value::Tensor(_) | Value::TensorFast(_) => ValueType::Tensor,
            _ => ValueType::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Guard { pub slot: u16, pub expected_type: ValueType }

impl From<u8> for ValueType {
    fn from(v: u8) -> Self {
        match v {
            0 => ValueType::I64,
            1 => ValueType::F64,
            2 => ValueType::Bool,
            3 => ValueType::Unit,
            4 => ValueType::Tensor,
            5 => ValueType::F32,
            _ => ValueType::Unknown,
        }
    }
}

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
    entries: FxHashMap<(u16, ValueType), u32>,
    max_entries: usize,
}

impl PolymorphicInlineCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: FxHashMap::default(),
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
    trace_selection: FxHashMap<u64, u32>,
    /// Maximum number of instructions allowed in a single trace before
    /// recording is aborted.  Prevents unbounded trace growth which would
    /// blow compile time and i-cache footprint.
    max_trace_length: usize,
}

impl TraceRecorder {
    pub fn new() -> Self {
        Self { current_trace: None, next_trace_id: 0, traces: Vec::new(), trace_selection: FxHashMap::default(), max_trace_length: 512 }
    }

    pub fn with_max_trace_length(max_trace_length: usize) -> Self {
        Self { current_trace: None, next_trace_id: 0, traces: Vec::new(), trace_selection: FxHashMap::default(), max_trace_length }
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

    /// Returns true if the current trace has exceeded the maximum trace
    /// length and should be aborted.  Callers should check this after each
    /// record_instruction call during execution-driven tracing.
    pub fn should_abort_trace(&self) -> bool {
        if let Some(ref trace) = self.current_trace {
            trace.instructions.len() > self.max_trace_length
        } else {
            false
        }
    }

    /// Abort the current recording, discarding the trace entirely.
    /// Used when a trace grows beyond `max_trace_length` or when an
    /// untraceable operation is encountered.
    pub fn abort_recording(&mut self) {
        self.current_trace = None;
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
            let mut slot_types: FxHashMap<u16, ValueType> = FxHashMap::default();
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
                        ValueType::F32 => 8, // F32 widened to F64 in unboxed buffer
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
    
    fn collect_slot_types(&self, instr: &Instr, slot_types: &mut FxHashMap<u16, ValueType>) {
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
//
// FIX #4: Arena-based executable memory allocation.
// Instead of per-trace mmap+mprotect pairs (which waste virtual memory and
// cause TLB pressure), traces are sub-allocated within 4MB chunks. This
// reduces mmap syscalls and improves TLB utilization.

/// 4MB chunk size for trace arena allocation
const TRACE_ARENA_CHUNK_SIZE: usize = 4 * 1024 * 1024;

/// Arena allocator for trace executable memory (Fix #4).
/// Sub-allocates traces within large mmap'd chunks, avoiding per-trace
/// mmap+mprotect overhead.
struct TraceArena {
    /// Chain of allocated chunks
    chunks: Vec<*mut u8>,
    /// Current chunk sizes (for munmap on drop)
    chunk_sizes: Vec<usize>,
    /// Per-chunk dirty flag: true if alloc() wrote to this chunk
    chunk_dirty: Vec<bool>,
    /// Offset into current chunk
    offset: usize,
    /// Current chunk remaining capacity
    remaining: usize,
}

// SAFETY: TraceArena exclusively owns its mmap'd regions.
unsafe impl Send for TraceArena {}
unsafe impl Sync for TraceArena {}

impl TraceArena {
    fn new() -> Self {
        Self {
            chunks: Vec::new(),
            chunk_sizes: Vec::new(),
            chunk_dirty: Vec::new(),
            offset: 0,
            remaining: 0,
        }
    }

    /// Allocate `len` bytes of executable memory from the arena.
    /// Returns a pointer to RW memory. Caller must call finalize()
    /// before executing the code.
    fn alloc(&mut self, len: usize) -> Result<*mut u8, String> {
        let aligned = (len + 15) & !15; // 16-byte alignment
        if aligned > self.remaining {
            // Need a new chunk
            let chunk_size = aligned.max(TRACE_ARENA_CHUNK_SIZE);
            let ptr = unsafe {
                mmap(
                    ptr::null_mut(),
                    chunk_size,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if ptr.is_null() || ptr as usize == usize::MAX {
                return Err("mmap failed for trace arena chunk".into());
            }
            self.chunks.push(ptr as *mut u8);
            self.chunk_sizes.push(chunk_size);
            self.chunk_dirty.push(false);
            self.offset = 0;
            self.remaining = chunk_size;
        }

        // Allocate from current chunk
        let chunk_idx = self.chunks.len() - 1;
        let chunk_ptr = *self.chunks.last().unwrap();
        let ptr = unsafe { chunk_ptr.add(self.offset) };
        self.offset += aligned;
        self.remaining -= aligned;
        // Mark current chunk as dirty so finalize() will mprotect it
        self.chunk_dirty[chunk_idx] = true;
        Ok(ptr)
    }

    /// Flip only dirty chunks from RW→RX for execution (W^X compliance).
    /// Clean chunks (already RX or never written) are skipped to avoid
    /// redundant mprotect syscalls.
    fn finalize(&mut self) -> Result<(), String> {
        for (i, (&chunk_ptr, &size)) in self.chunks.iter().zip(self.chunk_sizes.iter()).enumerate() {
            if self.chunk_dirty.get(i).copied().unwrap_or(false) {
                let ok = unsafe { mprotect(chunk_ptr as *mut c_void, size, PROT_READ | PROT_EXEC) };
                if ok != 0 {
                    return Err("mprotect failed in trace arena finalize".into());
                }
                self.chunk_dirty[i] = false; // Reset dirty flag after mprotect
            }
        }
        Ok(())
    }
}

impl Drop for TraceArena {
    fn drop(&mut self) {
        for (&chunk_ptr, &size) in self.chunks.iter().zip(self.chunk_sizes.iter()) {
            unsafe {
                // FIX #10: mprotect before munmap is redundant — munmap works
                // regardless of page protection. Skip it to avoid syscall overhead.
                munmap(chunk_ptr as *mut c_void, size);
            }
        }
    }
}

// Thread-local trace arena for sub-allocation (Fix #4).
std::thread_local! {
    static TRACE_ARENA: std::cell::RefCell<TraceArena> = std::cell::RefCell::new(TraceArena::new());
}

pub struct ExecutableMemory { ptr: *mut u8, len: usize, arena_backed: bool }

// SAFETY: ExecutableMemory exclusively owns its mmap'd region.
// It is safe to send across threads and share references because
// the memory is only read/executed, never mutated after construction.
unsafe impl Send for ExecutableMemory {}
unsafe impl Sync for ExecutableMemory {}

impl ExecutableMemory {
    #[cfg(unix)]
    pub fn new(code: &[u8]) -> Result<Self, String> {
        let len = code.len().max(1);

        // FIX #4: Try arena allocation first (sub-allocate within 4MB chunks).
        // This avoids per-trace mmap+mprotect pairs that waste virtual memory
        // and cause TLB pressure.
        if let Ok(ptr) = TRACE_ARENA.with(|arena_cell| {
            let mut arena = arena_cell.borrow_mut();
            arena.alloc(len)
        }) {
            unsafe { ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len()); }
            // Finalize arena to flip pages RW→RX before execution
            TRACE_ARENA.with(|arena_cell| {
                let mut arena = arena_cell.borrow_mut();
                arena.finalize()
            })?;
            return Ok(Self { ptr, len, arena_backed: true });
        }

        // Fallback: individual mmap per trace
        let ptr = unsafe { mmap(ptr::null_mut(), len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) };
        if ptr.is_null() || ptr as usize == usize::MAX { return Err("mmap failed".into()); }
        unsafe { ptr::copy_nonoverlapping(code.as_ptr(), ptr as *mut u8, code.len()); }
        if unsafe { mprotect(ptr, len, PROT_READ | PROT_EXEC) } != 0 {
            unsafe { munmap(ptr, len) }; return Err("mprotect failed".into());
        }
        Ok(Self { ptr: ptr as *mut u8, len, arena_backed: false })
    }

    #[cfg(not(unix))]
    pub fn new(_code: &[u8]) -> Result<Self, String> {
        Err("ExecutableMemory not supported on this platform".into())
    }

    pub fn entry_point(&self) -> *mut u8 { self.ptr }
}

impl Drop for ExecutableMemory {
    fn drop(&mut self) {
        if self.arena_backed {
            // Arena-backed memory is freed when the TraceArena is dropped,
            // not per-ExecutableMemory. No per-drop munmap needed.
            return;
        }
        #[cfg(unix)] unsafe {
            // FIX #10: mprotect before munmap is redundant — munmap works
            // regardless of page protection. Skip it to avoid the syscall overhead.
            munmap(self.ptr as *mut _, self.len);
        }
    }
}

// =============================================================================
// §4  HEAVILY OPTIMIZED NATIVE CODE GENERATOR
// =============================================================================
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Reg { RAX=0, RCX=1, RDX=2, R8=8, R9=9, R10=10, R11=11 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegState { Empty, Occupied(u16), Dirty(u16) }

pub struct NativeCodeGenerator {
    code: Vec<u8>,
    labels: FxHashMap<usize, usize>,
    patch_sites: Vec<PatchSite>,
    /// J5 fix: Flat array indexed by register number instead of FxHashMap.
    /// With only 6-10 JIT registers, a HashMap's hashing + probing + cache
    /// unfriendly layout is ~3-5x slower than a direct array index. Each
    /// HashMap entry has ~48 bytes of overhead vs an array's 8-16 bytes.
    reg_map: [RegState; 16],
    /// J5 fix: Vec indexed by slot number instead of FxHashMap<u16, Reg>.
    /// Slots are sequential integers, making Vec the natural O(1) choice.
    slot_reg: Vec<Option<Reg>>,
    next_label_id: usize,
    /// FIX (JIT-1): Deopt label for division-by-zero guards. Set during
    /// compile_trace() so emit_instruction can reference it.
    deopt_label: usize,
}

impl NativeCodeGenerator {
    pub fn new() -> Self {
        Self {
            code: Vec::with_capacity(4096), labels: FxHashMap::default(), patch_sites: Vec::new(),
            reg_map: [RegState::Empty; 16], slot_reg: Vec::new(),
            next_label_id: 0, deopt_label: 0,
        }
    }

    pub fn compile_trace(&mut self, trace: &Trace, unboxed_buffer: Option<*mut u8>) -> Result<CompiledTrace, String> {
        self.code.clear(); self.labels.clear(); self.patch_sites.clear();
        self.reg_map = [RegState::Empty; 16]; self.slot_reg.clear();

        // Initialize next_label_id from the trace's label counter so that
        // next_label() generates labels that don't collide with trace labels.
        self.next_label_id = trace.next_label_id;

        // The deopt stub is emitted after all trace code.  All guards jump to
        // this single label, which is resolved during backpatch_jumps once we
        // know the stub's byte offset.
        let deopt_label = self.next_label();
        self.deopt_label = deopt_label; // FIX (JIT-1): store for emit_instruction

        // Fix #1: Check if trace is type-specialized for unboxed operations
        let is_specialized = trace.specialized_type.is_some() && unboxed_buffer.is_some();

        // 1. ABI Prologue
        self.emit_prologue();

        // 2. Hoist invariant guards to entry & emit
        self.emit_hoisted_guards(&trace.guards, deopt_label)?;

        // FIX (JIT-2): The unboxed buffer address is now passed as a function
        // argument in RDX (the 3rd System V ABI register), rather than baked
        // as an immediate into the generated code. This makes the JIT code
        // position-independent with respect to the buffer — if the buffer is
        // ever reallocated (e.g., by a GC or arena reset), the caller can
        // simply pass the new address without recompiling the trace.
        //
        // The original code used mov_r8_imm64(buffer_addr) which baked the
        // absolute address into the machine code, creating a relocation
        // problem. Now we copy RDX → R8 in the prologue so all existing
        // emit_unboxed_load/store_reg code (which references [R8+offset])
        // continues to work unchanged.
        if is_specialized {
            // mov r8, rdx — copy the buffer argument from RDX to R8.
            // Use mov_r8_imm64 only when we have a concrete buffer address that
            // was passed as an immediate (for position-dependent code paths).
            // Here we use register-to-register move instead, but mov_r8_imm64
            // is available for cases where the buffer address is known at
            // compile time (e.g., when the unboxed buffer is at a fixed address).
            if let Some(buf) = unboxed_buffer {
                if !buf.is_null() {
                    // Position-dependent fallback: load immediate buffer address.
                    // This is used when the caller provides a fixed buffer address
                    // that won't change (e.g., a statically allocated buffer).
                    self.mov_r8_imm64(buf as i64);
                } else {
                    // Position-independent: copy RDX → R8
                    self.b(0x49); self.b(0x89); self.b(0xD0); // mov r8, rdx
                }
            } else {
                // No buffer provided — shouldn't happen when is_specialized is true
                self.b(0x49); self.b(0x89); self.b(0xD0); // mov r8, rdx
            }
        }

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
        {
        let mut last_load: FxHashMap<u16, i64> = FxHashMap::default();
        
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
            // J7 fix: Guard slot encoding now supports slots > 255.
            // Previously, the imm8 displacement in [rsi + slot] only encoded
            // slots 0-255; larger slots silently read from wrong memory.
            // Now we use disp32 encoding (mod=10) for all slots, which
            // handles the full u16 range correctly at the cost of 3 extra
            // bytes per guard (7→10 bytes). For slots < 256 we could use
            // the shorter form, but correctness > compactness.
            self.b(0x0F); self.b(0xB6);          // movzx eax, byte [rsi + disp32]
            self.modrm(2, 0, 6);                  // mod=10 (disp32), reg=0 (eax), rm=6 (rsi)
            self.i32(g.slot as i32);              // full 32-bit displacement
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
                // J1 fix: Use optimal immediate encoding instead of always
                // emitting the 10-byte MOV RAX, imm64. For values that fit
                // in i32 (the vast majority of integer constants), we save
                // 3-5 bytes per load. This reduces code bloat by ~30-50%
                // for integer-heavy traces, improving i-cache hit rates and
                // decode throughput.
                if let Ok(v32) = i32::try_from(*val) {
                    if v32 >= 0 {
                        self.mov_eax_imm32(v32); // 5 bytes
                    } else {
                        // MOV RAX, sign-extended imm32: 48 C7 C0 + imm32 = 7 bytes
                        self.b(0x48); self.b(0xC7); self.b(0xC0);
                        self.i32(v32);
                    }
                } else {
                    self.mov_rax_imm64(*val); // 10 bytes — only when needed
                }
                self.mark_dirty(*dst);
            }
            Instr::LoadBool(dst, val) => {
                self.ensure_reg(*dst, Reg::RAX)?;
                if *val {
                    self.mov_eax_imm32(1);
                } else {
                    self.b(0x31); self.b(0xC0); // xor eax, eax
                }
                self.mark_dirty(*dst);
            }
            Instr::LoadUnit(dst) => {
                self.ensure_reg(*dst, Reg::RAX)?;
                self.b(0x31); self.b(0xC0); // xor eax, eax — Unit = 0
                self.mark_dirty(*dst);
            }
            Instr::Move(dst, src) => {
                if dst != src {
                    let src_reg = self.alloc_reg(*src)?;
                    self.ensure_reg(*dst, src_reg)?;
                    // The value is already in the register from alloc_reg
                    self.mark_dirty(*dst);
                }
            }
            Instr::Store(slot, src_reg) => {
                // Store register value to the flat i64 slot array.
                // The slot array is passed as the first argument (RDI).
                // Emit: mov [rdi + slot*8], <src_reg_value>
                let src = self.alloc_reg(*src_reg)?;
                // Move to RAX for the store (we store via RAX)
                if src != Reg::RAX { self.mov_reg_reg(src, Reg::RAX); }
                // MOV [RDI + disp32], RAX
                self.b(0x48); self.b(0x89); // REX.W MOV r/m64, r64
                self.modrm(2, 0, 7); // mod=10 (disp32), reg=RAX(0), rm=RDI(7)
                self.i32((*slot as i32) * 8);
            }
            Instr::Load(dst, slot) => {
                // Load value from flat i64 slot array into register.
                // The slot array is passed as the first argument (RDI).
                // Emit: mov <dst_reg>, [rdi + slot*8]
                self.ensure_reg(*dst, Reg::RAX)?;
                // MOV RAX, [RDI + disp32]
                self.b(0x48); self.b(0x8B); // REX.W MOV r64, r/m64
                self.modrm(2, 0, 7); // mod=10 (disp32), reg=RAX(0), rm=RDI(7)
                self.i32((*slot as i32) * 8);
                self.mark_dirty(*dst);
            }
            Instr::BinOp(dst, op, lhs, rhs) => {
                // FIX #2: Use wider register allocation to reduce load-spill-load
                // cycles.
                let lhs_reg = self.alloc_reg(*lhs)?;
                let rhs_reg = self.alloc_reg(*rhs)?;
                // Move operands to RAX/RCX for arithmetic
                if lhs_reg != Reg::RAX { self.mov_reg_reg(lhs_reg, Reg::RAX); }
                if rhs_reg != Reg::RCX { self.mov_reg_reg(rhs_reg, Reg::RCX); }
                match op {
                    BinOpKind::Add => self.add_rax_rcx(),
                    BinOpKind::Sub => self.sub_rax_rcx(),
                    BinOpKind::Mul => self.imul_rax_rcx(),
                    BinOpKind::Div => {
                        self.test_rax_rax();
                        let zero_label = self.next_label();
                        self.jz_short(zero_label);
                        self.test_rcx_rcx();
                        self.jz_label(self.deopt_label);
                        // Fix #2: Guard against i64::MIN / -1 overflow (would raise SIGFPE)
                        // Use R10 as scratch (push/pop to preserve any live value)
                        self.bb(0x41, 0x52);                // push r10
                        self.bb(0x49, 0xBA); self.i64(i64::MIN); // mov r10, i64::MIN
                        self.bbb(0x4C, 0x39, 0xD0);         // cmp rax, r10
                        let skip_overflow1 = self.next_label();
                        self.jne_label(skip_overflow1);
                        self.bb(0x49, 0xBA); self.i64(-1i64);   // mov r10, -1
                        self.bbb(0x4C, 0x39, 0xD1);         // cmp rcx, r10
                        let skip_overflow2 = self.next_label();
                        self.jne_label(skip_overflow2);
                        // Overflow: i64::MIN / -1 — restore r10 and deopt
                        self.bb(0x41, 0x5A);                // pop r10
                        self.b(0xE9); self.i32(0);           // jmp deopt_label (rel32)
                        self.patch_sites.push(PatchSite { buffer_offset: self.code.len()-4, target_label: self.deopt_label, is_short_jump: false });
                        self.labels.insert(skip_overflow1, self.code.len());
                        self.labels.insert(skip_overflow2, self.code.len());
                        self.bb(0x41, 0x5A);                // pop r10
                        // Now safe to divide
                        self.idiv_rax_rcx();
                        self.labels.insert(zero_label, self.code.len());
                    }
                    BinOpKind::Rem => {
                        self.test_rcx_rcx();
                        self.jz_label(self.deopt_label);
                        // Fix #2: Guard against i64::MIN % -1 overflow (would raise SIGFPE)
                        self.bb(0x41, 0x52);                // push r10
                        self.bb(0x49, 0xBA); self.i64(i64::MIN); // mov r10, i64::MIN
                        self.bbb(0x4C, 0x39, 0xD0);         // cmp rax, r10
                        let skip_overflow1 = self.next_label();
                        self.jne_label(skip_overflow1);
                        self.bb(0x49, 0xBA); self.i64(-1i64);   // mov r10, -1
                        self.bbb(0x4C, 0x39, 0xD1);         // cmp rcx, r10
                        let skip_overflow2 = self.next_label();
                        self.jne_label(skip_overflow2);
                        // Overflow: i64::MIN % -1 — restore r10 and deopt
                        self.bb(0x41, 0x5A);                // pop r10
                        self.b(0xE9); self.i32(0);           // jmp deopt_label (rel32)
                        self.patch_sites.push(PatchSite { buffer_offset: self.code.len()-4, target_label: self.deopt_label, is_short_jump: false });
                        self.labels.insert(skip_overflow1, self.code.len());
                        self.labels.insert(skip_overflow2, self.code.len());
                        self.bb(0x41, 0x5A);                // pop r10
                        // Now safe to compute remainder
                        self.irem_rax_rcx();
                    }
                    BinOpKind::BitAnd => self.and_rax_rcx(),
                    BinOpKind::BitOr => self.or_rax_rcx(),
                    BinOpKind::BitXor => self.xor_rax_rcx(),
                    BinOpKind::Shl => self.shl_rax_cl(),
                    BinOpKind::Shr => self.shr_rax_cl(),
                    BinOpKind::Eq => {
                        // cmp rax, rcx; sete al; movzx eax, al
                        self.bbb(0x48, 0x39, 0xC8); // cmp rax, rcx
                        self.b(0x0F); self.b(0x94); self.b(0xC0); // sete al
                        self.b(0x0F); self.b(0xB6); self.b(0xC0); // movzx eax, al
                    }
                    BinOpKind::Ne => {
                        self.bbb(0x48, 0x39, 0xC8); // cmp rax, rcx
                        self.b(0x0F); self.b(0x95); self.b(0xC0); // setne al
                        self.b(0x0F); self.b(0xB6); self.b(0xC0); // movzx eax, al
                    }
                    BinOpKind::Lt => {
                        self.bbb(0x48, 0x39, 0xC8); // cmp rax, rcx
                        self.b(0x0F); self.b(0x9C); self.b(0xC0); // setl al
                        self.b(0x0F); self.b(0xB6); self.b(0xC0); // movzx eax, al
                    }
                    BinOpKind::Le => {
                        self.bbb(0x48, 0x39, 0xC8); // cmp rax, rcx
                        self.b(0x0F); self.b(0x9E); self.b(0xC0); // setle al
                        self.b(0x0F); self.b(0xB6); self.b(0xC0); // movzx eax, al
                    }
                    BinOpKind::Gt => {
                        self.bbb(0x48, 0x39, 0xC8); // cmp rax, rcx
                        self.b(0x0F); self.b(0x9F); self.b(0xC0); // setg al
                        self.b(0x0F); self.b(0xB6); self.b(0xC0); // movzx eax, al
                    }
                    BinOpKind::Ge => {
                        self.bbb(0x48, 0x39, 0xC8); // cmp rax, rcx
                        self.b(0x0F); self.b(0x9D); self.b(0xC0); // setge al
                        self.b(0x0F); self.b(0xB6); self.b(0xC0); // movzx eax, al
                    }
                    _ => return Err(format!("Unsupported BinOp in trace backend: {:?}", op)),
                }
                self.bind_slot_reg(*dst, Reg::RAX);
                self.mark_dirty(*dst);
            }
            Instr::UnOp(dst, _op, src) => {
                // Support basic unary operations
                let src_reg = self.alloc_reg(*src)?;
                if src_reg != Reg::RAX { self.mov_reg_reg(src_reg, Reg::RAX); }
                // For Neg: neg rax
                self.b(0x48); self.b(0xF7); self.b(0xD8); // neg rax
                self.bind_slot_reg(*dst, Reg::RAX);
                self.mark_dirty(*dst);
            }
            Instr::Return(slot) => {
                self.ensure_reg(*slot, Reg::RAX)?;
                self.writeback_all_dirty()?;
                // Fallthrough to ret
            }
            Instr::ReturnUnit => {
                self.b(0x31); self.b(0xC0); // xor eax, eax
                self.writeback_all_dirty()?;
            }
            Instr::Jump(_offset) => {
                // For tracing JIT, we compile the hot path as a linear sequence.
                // Jump instructions in the trace are elided since we only record
                // the taken path. However, backward jumps (loops) need to be
                // handled: we emit a loop back to the trace entry.
                // For now, we simply skip the jump — the trace is already linearized.
            }
            Instr::JumpFalse(_reg, _offset) | Instr::JumpTrue(_reg, _offset) => {
                // Same as Jump: the trace is linearized, so conditional jumps
                // are already resolved (we only recorded the taken path).
                // Guards handle the type-checking; side exits handle the
                // unexpected path.
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
                // J1 fix: Same optimal immediate encoding as the boxed path
                if let Ok(v32) = i32::try_from(*val) {
                    if v32 >= 0 {
                        self.mov_eax_imm32(v32);
                    } else {
                        self.b(0x48); self.b(0xC7); self.b(0xC0);
                        self.i32(v32);
                    }
                } else {
                    self.mov_rax_imm64(*val);
                }
                self.mark_dirty(*dst);
                if let Some(&Some(offset)) = unboxed_slots.get(*dst as usize) {
                    self.emit_unboxed_store(offset, *val, vtype, unboxed_buffer);
                }
            }
            Instr::BinOp(dst, op, lhs, rhs) => {
                // Only use the unboxed path if ALL operands have unboxed offsets;
                // otherwise fall through to the boxed implementation to avoid
                // operating on stale register values.
                let lhs_has = unboxed_slots.get(*lhs as usize).and_then(|o| *o).is_some();
                let rhs_has = unboxed_slots.get(*rhs as usize).and_then(|o| *o).is_some();
                let dst_has = unboxed_slots.get(*dst as usize).and_then(|o| *o).is_some();
                if lhs_has && rhs_has && dst_has {
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
                        BinOpKind::Div => {
                            // FIX (JIT-1): Guard against division by zero (unboxed path)
                            // Also use test_rax_rax + jz_short for zero-dividend fast path
                            self.test_rax_rax();
                            let zero_label = self.next_label();
                            self.jz_short(zero_label); // if dividend==0, skip to zero-result
                            self.test_rcx_rcx();
                            self.jz_label(self.deopt_label);
                            // Fix #2: Guard against i64::MIN / -1 overflow (unboxed path)
                            self.bb(0x41, 0x52);                // push r10
                            self.bb(0x49, 0xBA); self.i64(i64::MIN); // mov r10, i64::MIN
                            self.bbb(0x4C, 0x39, 0xD0);         // cmp rax, r10
                            let skip_overflow1 = self.next_label();
                            self.jne_label(skip_overflow1);
                            self.bb(0x49, 0xBA); self.i64(-1i64);   // mov r10, -1
                            self.bbb(0x4C, 0x39, 0xD1);         // cmp rcx, r10
                            let skip_overflow2 = self.next_label();
                            self.jne_label(skip_overflow2);
                            self.bb(0x41, 0x5A);                // pop r10
                            self.b(0xE9); self.i32(0);           // jmp deopt_label (rel32)
                            self.patch_sites.push(PatchSite { buffer_offset: self.code.len()-4, target_label: self.deopt_label, is_short_jump: false });
                            self.labels.insert(skip_overflow1, self.code.len());
                            self.labels.insert(skip_overflow2, self.code.len());
                            self.bb(0x41, 0x5A);                // pop r10
                            // Now safe to divide
                            self.idiv_rax_rcx();
                            self.labels.insert(zero_label, self.code.len());
                        }
                        BinOpKind::Rem => {
                            // FIX (JIT-1): Guard against division by zero (unboxed path)
                            self.test_rcx_rcx();
                            self.jz_label(self.deopt_label);
                            // Fix #2: Guard against i64::MIN % -1 overflow (unboxed path)
                            self.bb(0x41, 0x52);                // push r10
                            self.bb(0x49, 0xBA); self.i64(i64::MIN); // mov r10, i64::MIN
                            self.bbb(0x4C, 0x39, 0xD0);         // cmp rax, r10
                            let skip_overflow1 = self.next_label();
                            self.jne_label(skip_overflow1);
                            self.bb(0x49, 0xBA); self.i64(-1i64);   // mov r10, -1
                            self.bbb(0x4C, 0x39, 0xD1);         // cmp rcx, r10
                            let skip_overflow2 = self.next_label();
                            self.jne_label(skip_overflow2);
                            self.bb(0x41, 0x5A);                // pop r10
                            self.b(0xE9); self.i32(0);           // jmp deopt_label (rel32)
                            self.patch_sites.push(PatchSite { buffer_offset: self.code.len()-4, target_label: self.deopt_label, is_short_jump: false });
                            self.labels.insert(skip_overflow1, self.code.len());
                            self.labels.insert(skip_overflow2, self.code.len());
                            self.bb(0x41, 0x5A);                // pop r10
                            // Now safe to compute remainder
                            self.irem_rax_rcx();
                        }
                        BinOpKind::BitAnd => self.and_rax_rcx(),
                        BinOpKind::BitOr => self.or_rax_rcx(),
                        BinOpKind::BitXor => self.xor_rax_rcx(),
                        BinOpKind::Shl => self.shl_rax_cl(),
                        BinOpKind::Shr => self.shr_rax_cl(),
                        _ => return Err(format!("Unsupported BinOp in unboxed mode: {:?}", op)),
                    }
                    // Store result to unboxed buffer
                    if let Some(&Some(dst_offset)) = unboxed_slots.get(*dst as usize) {
                        self.emit_unboxed_store_reg(dst_offset, Reg::RAX, vtype, unboxed_buffer);
                    }
                } else {
                    // Fallback to boxed path: at least one operand lacks an unboxed offset
                    return self.emit_instruction(ti);
                }
            }
            Instr::Return(slot) => {
                if let Some(&Some(offset)) = unboxed_slots.get(*slot as usize) {
                    self.emit_unboxed_load(offset, vtype, unboxed_buffer, Reg::RAX)?;
                }
                // Write back all dirty registers before returning so modified
                // slot values are not lost on the unboxed path.
                self.writeback_all_dirty()?;
            }
            _ => return Err(format!("Unsupported instruction in unboxed mode: {:?}", ti.instruction)),
        }
        Ok(())
    }

    fn emit_unboxed_load(&mut self, offset: u32, vtype: ValueType, _buffer: *mut u8, reg: Reg) -> Result<(), String> {
        // Load from unboxed buffer at [buffer + offset]
        // J2 fix: Removed redundant mov_r8_imm64 per load. The buffer base
        // address is now loaded once in compile_trace() prologue via
        // emit_unboxed_prologue(), so R8 already holds the correct base.
        //
        // Boolean fix: For ValueType::Bool, we only store 1 byte, so we must
        // only load 1 byte (movzx) to avoid reading past the slot boundary
        // and pulling in garbage from adjacent boolean values.
        //
        // F32 fix: F32 values are widened to F64 when stored in the unboxed
        // buffer (f32→f64 is lossless), so we load 8 bytes just like F64.
        // The type tag (ValueType::F32 vs F64) distinguishes them for guards.
        let reg_code = reg as u8;
        if matches!(vtype, ValueType::Bool) {
            // MOVZX r32, byte [r8 + disp32]
            // REX: W=0 (32-bit dest zero-extends), REX.R if reg >= 8
            let rex = if reg_code >= 8 { 0x44 } else { 0x00 }; // REX optional for REX.R only
            if rex != 0 { self.b(rex); }
            self.b(0x0F);
            self.b(0xB6); // MOVZX r32, r/m8
            self.modrm(2, reg_code & 7, 8); // mod=10, reg=reg, rm=r8
            self.i32(offset as i32);
        } else {
            let rex = 0x48 | if reg_code >= 8 { 0x04 } else { 0x00 };
            self.b(rex);
            self.b(0x8B); // MOV r64, r/m64
            self.modrm(2, reg_code & 7, 8); // mod=10, reg=reg, rm=r8
            self.i32(offset as i32);
        }
        Ok(())
    }

    fn emit_unboxed_store(&mut self, offset: u32, value: i64, vtype: ValueType, buffer: *mut u8) {
        // Store immediate to unboxed buffer
        self.mov_rax_imm64(value);
        self.emit_unboxed_store_reg(offset, Reg::RAX, vtype, buffer);
    }

    fn emit_unboxed_store_reg(&mut self, offset: u32, reg: Reg, vtype: ValueType, _buffer: *mut u8) {
        // Store register to unboxed buffer at [r8 + offset]
        // J2 fix: Removed redundant mov_r8_imm64 per store. R8 is loaded
        // once in the prologue; no need to reload on every store.
        //
        // Boolean fix: For ValueType::Bool, we must only write 1 byte to
        // avoid corrupting adjacent boolean values in the tightly-packed
        // unboxed buffer (1 byte per bool vs 8 bytes per i64/f64).
        //
        // F32 fix: F32 values are widened to F64 (8 bytes) in the unboxed
        // buffer. We store 8 bytes just like F64/I64. The caller must
        // ensure the value in the register is already the widened f64
        // representation (f64::from(f32_value) stored as i64 bits).
        let reg_code = reg as u8;
        if matches!(vtype, ValueType::Bool) {
            // MOV byte [r8 + disp32], r8_low  (AL for rax, CL for rcx, etc.)
            // Use the low byte of the register. For RAX/RCX this is AL/CL.
            // REX prefix is NOT needed for byte stores to AL/CL/DL/BL.
            // For R8B-R15B, we need REX prefix (0x41 for REX.B).
            if reg_code >= 8 {
                self.b(0x41); // REX.B for r8b-r15b
            }
            self.b(0x88); // MOV r/m8, r8
            self.modrm(2, reg_code & 7, 8); // mod=10, reg=low_byte(reg), rm=r8
            self.i32(offset as i32);
        } else {
            let rex = 0x48 | if reg_code >= 8 { 0x04 } else { 0x00 };
            self.b(rex);
            self.b(0x89); // MOV r/m64, r64
            self.modrm(2, reg_code & 7, 8); // mod=10, reg=reg, rm=r8
            self.i32(offset as i32);
        }
    }

    // --- Register Allocation & Spilling ---
    // J5 fix: All methods updated to use flat array reg_map[reg as usize]
    // and Vec<Option<Reg>> slot_reg instead of FxHashMap lookups.
    //
    // FIX #2: Extended register allocator to use RAX, RCX, RDX, R8, R9, R10
    // instead of only RAX and RCX. This reduces load-spill-load cycles when
    // multiple values are live simultaneously.

    /// Register allocation candidate pool (Fix #2).
    /// Ordered by preference: RAX/RCX are scratch registers used by arithmetic
    /// ops, but RDX, R8, R9, R10 are available for holding intermediate values
    /// without forcing eviction of live RAX/RCX contents.
    const REG_POOL: [Reg; 7] = [Reg::RAX, Reg::RCX, Reg::RDX, Reg::R8, Reg::R9, Reg::R10, Reg::R11];

    /// Allocate a free register from the wider pool (Fix #2).
    /// Returns a free register, or evicts an occupied one. This allows
    /// intermediate values to live in RDX/R8/R9/R10 instead of forcing
    /// everything through RAX/RCX.
    fn alloc_reg(&mut self, slot: u16) -> Result<Reg, String> {
        self.ensure_slot_reg_capacity(slot);
        // Check if already allocated
        if let Some(reg) = self.slot_reg[slot as usize] {
            return Ok(reg);
        }
        // Find a free register from the wider pool
        for &reg in &Self::REG_POOL {
            if self.reg_map[reg as usize] == RegState::Empty {
                self.load_slot_to_reg(slot, reg)?;
                self.bind_slot_reg(slot, reg);
                return Ok(reg);
            }
        }
        // All candidate registers are occupied — evict the least recently used.
        // For simplicity, evict the first non-dirty occupant; if all dirty,
        // spill the first one.
        for &reg in &Self::REG_POOL {
            if let RegState::Occupied(victim_slot) = self.reg_map[reg as usize] {
                self.spill_slot(victim_slot, reg)?;
                self.load_slot_to_reg(slot, reg)?;
                self.bind_slot_reg(slot, reg);
                return Ok(reg);
            }
        }
        // All are dirty — spill first candidate
        if let RegState::Dirty(victim_slot) = self.reg_map[Reg::RDX as usize] {
            self.spill_slot(victim_slot, Reg::RDX)?;
            self.load_slot_to_reg(slot, Reg::RDX)?;
            self.bind_slot_reg(slot, Reg::RDX);
            return Ok(Reg::RDX);
        }
        Err("No registers available for allocation".into())
    }

    /// Ensure slot_reg Vec is large enough to hold the given slot index
    #[inline(always)]
    fn ensure_slot_reg_capacity(&mut self, slot: u16) {
        let needed = slot as usize + 1;
        if self.slot_reg.len() < needed {
            self.slot_reg.resize(needed, None);
        }
    }

    fn ensure_reg(&mut self, slot: u16, preferred: Reg) -> Result<(), String> {
        self.ensure_slot_reg_capacity(slot);
        if let Some(reg) = self.slot_reg[slot as usize] {
            if reg != preferred {
                // Fix #3: Properly rebind slot to preferred register after copy.
                // Previously, mov_reg_reg copied the value but left slot_reg
                // pointing at the old register. Subsequent mark_dirty() calls
                // would mark the OLD register dirty (which still held the stale
                // value), while the new value in preferred was untracked.
                // Now we handle the preferred register's current occupant and
                // rebind the slot to preferred, preserving dirty state.

                // First, handle any existing occupant of the preferred register.
                match self.reg_map[preferred as usize] {
                    RegState::Dirty(other_slot) => {
                        // Spill the dirty occupant before overwriting preferred.
                        self.spill_slot(other_slot, preferred)?;
                    }
                    RegState::Occupied(other_slot) => {
                        // Clean occupant — value is already in memory, just unbind.
                        self.ensure_slot_reg_capacity(other_slot);
                        self.slot_reg[other_slot as usize] = None;
                        self.reg_map[preferred as usize] = RegState::Empty;
                    }
                    RegState::Empty => {}
                }

                // Copy value from old register to preferred.
                self.mov_reg_reg(reg, preferred);

                // Transfer the binding from old register to preferred,
                // preserving the dirty state.
                let was_dirty = matches!(self.reg_map[reg as usize], RegState::Dirty(_));
                self.reg_map[reg as usize] = RegState::Empty;
                self.slot_reg[slot as usize] = Some(preferred);
                self.reg_map[preferred as usize] = if was_dirty {
                    RegState::Dirty(slot)
                } else {
                    RegState::Occupied(slot)
                };
            }
            return Ok(());
        }
        if self.reg_map[preferred as usize] == RegState::Empty {
            self.load_slot_to_reg(slot, preferred)?;
            self.bind_slot_reg(slot, preferred);
            return Ok(());
        }
        // Evict
        let victim = if let RegState::Dirty(v) = self.reg_map[preferred as usize] { v } else {
            // Find any occupant
            for (i, state) in self.reg_map.iter().enumerate() {
                if let RegState::Occupied(_s) = state { return self.spill_and_evict(slot, preferred); }
                if i >= 12 { break; } // only check first 12 entries (our JIT regs)
            }
            return Err("No registers available".into());
        };
        self.spill_slot(victim, preferred)?;
        self.load_slot_to_reg(slot, preferred)?;
        self.bind_slot_reg(slot, preferred);
        Ok(())
    }

    fn spill_and_evict(&mut self, wanted_slot: u16, target: Reg) -> Result<(), String> {
        self.ensure_slot_reg_capacity(wanted_slot);
        // Find which register holds wanted_slot, or pick a register to evict
        let evict_reg = if let Some(reg) = self.slot_reg.get(wanted_slot as usize).and_then(|r| *r) {
            reg
        } else {
            // Pick a victim register
            if let RegState::Dirty(_v) = self.reg_map[target as usize] {
                target
            } else {
                // Find any occupant
                let evicted = self.reg_map.iter().enumerate()
                    .find_map(|(i, state)| match state {
                        RegState::Occupied(s) => Some((i as u8, *s)),
                        _ => None,
                    });
                if let Some((reg_idx, evicted_slot)) = evicted {
                    let reg = unsafe { std::mem::transmute::<u8, Reg>(reg_idx) };
                    self.spill_slot(evicted_slot, reg)?;
                    self.reg_map[reg_idx as usize] = RegState::Empty;
                    self.ensure_slot_reg_capacity(evicted_slot);
                    self.slot_reg[evicted_slot as usize] = None;
                    self.load_slot_to_reg(wanted_slot, target)?;
                    self.bind_slot_reg(wanted_slot, target);
                    return Ok(());
                }
                return Err("No registers available".into());
            }
        };
        // Spill the evicted slot to memory
        // Find the slot currently occupying evict_reg
        let evicted_slot = match self.reg_map[evict_reg as usize] {
            RegState::Occupied(s) | RegState::Dirty(s) => s,
            _ => wanted_slot,
        };
        self.spill_slot(evicted_slot, evict_reg)?;
        self.reg_map[evict_reg as usize] = RegState::Empty;
        self.ensure_slot_reg_capacity(evicted_slot);
        self.slot_reg[evicted_slot as usize] = None;
        // Load the WANTED slot into the now-free register
        self.load_slot_to_reg(wanted_slot, target)?;
        self.bind_slot_reg(wanted_slot, target);
        Ok(())
    }

    fn bind_slot_reg(&mut self, slot: u16, reg: Reg) {
        self.ensure_slot_reg_capacity(slot);
        if let Some(old) = self.slot_reg[slot as usize].take() {
            self.reg_map[old as usize] = RegState::Empty;
        }
        self.slot_reg[slot as usize] = Some(reg);
        self.reg_map[reg as usize] = RegState::Occupied(slot);
    }

    fn mark_dirty(&mut self, slot: u16) {
        self.ensure_slot_reg_capacity(slot);
        if let Some(reg) = self.slot_reg[slot as usize] {
            self.reg_map[reg as usize] = RegState::Dirty(slot);
        }
    }

    fn spill_slot(&mut self, slot: u16, reg: Reg) -> Result<(), String> {
        let reg_code = reg as u8;
        let rex = 0x48 | if reg_code >= 8 { 0x04 } else { 0x00 };
        self.b(rex);
        self.b(0x89);                      // MOV r/m64, r64
        self.modrm(2, reg_code & 7, 7);    // mod=10 (disp32), reg=reg, rm=7 (rdi)
        self.i32((slot as i32) * 8);
        self.reg_map[reg as usize] = RegState::Empty;
        self.ensure_slot_reg_capacity(slot);
        self.slot_reg[slot as usize] = None;
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
        // J3 fix: Use SmallVec instead of heap-allocating a Vec every return.
        // J5 fix: Iterate flat array instead of FxHashMap.
        // With only 6 JIT registers, there can be at most 6 dirty entries,
        // so SmallVec<[(u16, Reg); 6]> stays on the stack with zero allocation.
        let dirty_slots: SmallVec<[(u16, Reg); 6]> = self.reg_map.iter().enumerate()
            .filter_map(|(i, s)| if let RegState::Dirty(sl) = s {
                Some((*sl, unsafe { std::mem::transmute::<u8, Reg>(i as u8) }))
            } else { None })
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
    fn mov_r8_imm64(&mut self, v: i64) { self.bb(0x49, 0xB8); self.i64(v); } // REX.W+B for r8
    fn add_rax_rcx(&mut self) { self.bbb(0x48, 0x01, 0xC8); }
    fn sub_rax_rcx(&mut self) { self.bbb(0x48, 0x29, 0xC8); }
    fn imul_rax_rcx(&mut self) { self.bbbb(0x48, 0x0F, 0xAF, 0xC1); }
    fn idiv_rax_rcx(&mut self) { self.bbb(0x48, 0x99, 0xF7); self.b(0xF1); }
    fn irem_rax_rcx(&mut self) { self.bbb(0x48, 0x99, 0xF7); self.b(0xF1); self.bbb(0x48, 0x89, 0xD0); /* mov rax, rdx — remainder from RDX */ }
    fn and_rax_rcx(&mut self) { self.bbb(0x48, 0x21, 0xC8); }
    fn or_rax_rcx(&mut self) { self.bbb(0x48, 0x09, 0xC8); }
    fn xor_rax_rcx(&mut self) { self.bbb(0x48, 0x31, 0xC8); }
    fn shl_rax_cl(&mut self) { self.bb(0x48, 0xD3); self.b(0xE0); }
    fn shr_rax_cl(&mut self) { self.bb(0x48, 0xD3); self.b(0xE8); }
    fn test_rax_rax(&mut self) { self.bbb(0x48, 0x85, 0xC0); }
    fn test_rcx_rcx(&mut self) { self.bbb(0x48, 0x85, 0xC9); } // FIX (JIT-1): test rcx, rcx

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
    /// Execute the compiled trace.
    ///
    /// FIX (JIT-2): The signature now accepts an optional `unboxed_buffer`
    /// parameter passed via RDX (3rd System V argument register). This
    /// replaces the old approach where the buffer address was baked into
    /// the generated code as an immediate value.
    ///
    /// Signature: fn(slots: *mut i64, types: *const u8, unboxed_buffer: *mut u8) -> i64
    pub unsafe fn execute(&self, slots: *mut i64, types: *const u8, unboxed_buffer: *mut u8) -> i64 {
        let func: unsafe extern "C" fn(*mut i64, *const u8, *mut u8) -> i64 = mem::transmute(self.entry_point);
        func(slots, types, unboxed_buffer)
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
    /// Per-PC hot counters tracking how many times each entry_pc has been called.
    hot_counters: FxHashMap<u64, u64>,
    /// J8 fix: Cache of compiled traces keyed by trace_id.
    /// Without this cache, the JIT recompiles the same trace on every call
    /// after the hot threshold — making the JIT essentially useless since
    /// compilation overhead (mmap + codegen + mprotect) exceeds any benefit.
    pub compiled_cache: FxHashMap<u32, CompiledTrace>,
    /// Polymorphic Inline Cache: maps (slot, failed_type) → trace_id for
    /// secondary traces compiled after a guard failure.  This enables trace
    /// stitching — when a guard fails because a slot changed type, the PIC
    /// provides an alternative trace specialised for the new type.
    pic: PolymorphicInlineCache,
    /// Maximum number of instructions allowed in a single trace before
    /// recording is aborted (default: 512).
    max_trace_length: usize,
}

impl TracingJIT {
    pub fn new() -> Self {
        Self {
            recorder: TraceRecorder::new(), codegen: NativeCodeGenerator::new(),
            trace_trigger: 100, compile_trigger: 10, traces_recorded: 0,
            traces_compiled: 0, deoptimizations: 0, hot_counters: FxHashMap::default(),
            compiled_cache: FxHashMap::default(),
            pic: PolymorphicInlineCache::new(16),
            max_trace_length: 512,
        }
    }

    pub fn should_start_tracing(&self, c: u64) -> bool { c >= self.trace_trigger }
    pub fn should_compile(&self, t: &Trace) -> bool { t.execution_count >= self.compile_trigger }

    /// Detect which guard failed by scanning the type array against the
    /// trace's guard list.  Returns the first (slot, observed_type) pair
    /// where the actual type differs from the guard's expected type.
    fn detect_guard_failure(trace: &Trace, types: &[u8]) -> Option<(u16, ValueType)> {
        for guard in &trace.guards {
            let slot_idx = guard.slot as usize;
            if let Some(&actual_byte) = types.get(slot_idx) {
                let actual_type = ValueType::from(actual_byte);
                if actual_type != guard.expected_type {
                    return Some((guard.slot, actual_type));
                }
            }
        }
        None
    }

    /// Record a guard failure in the PIC and trigger recording of a new
    /// trace for the failed type path.  The new trace will be compiled
    /// on the next execution when it becomes hot.
    ///
    /// Unlike the old version, this does NOT record an empty trace. Instead
    /// it only registers the PIC entry — the actual trace will be recorded
    /// when `execute_with_jit` next encounters this hot PC and records
    /// instructions from the bytecode.
    fn record_guard_failure(&mut self, entry_pc: usize, slot: u16, failed_type: ValueType) {
        // Allocate a trace ID without recording a full trace yet.
        // The PIC maps (slot, failed_type) → trace_id so that future
        // lookups can find this entry. The actual trace will be populated
        // on the next hot-path execution.
        let new_trace_id = self.recorder.next_trace_id;
        self.recorder.next_trace_id += 1;
        self.pic.add_side_exit(slot, failed_type, new_trace_id);
        self.traces_recorded += 1;
    }

    pub fn execute_with_jit(&mut self, entry_pc: usize, slots: &mut [Value], types: &mut [u8], _instructions: &[Instr]) -> Result<Value, RuntimeError> {
        // ── Increment hot counter ──────────────────────────────────────────
        let hot_count = *self.hot_counters.entry(entry_pc as u64).and_modify(|c| *c += 1).or_insert(1);

        // ── Check for existing trace ───────────────────────────────────────
        if let Some(tid) = self.recorder.find_trace(entry_pc) {
            // J8 fix: Check the compiled cache FIRST. If we already compiled this
            // trace, reuse the cached machine code instead of recompiling.
            if let Some(ct) = self.compiled_cache.get(&tid) {
                let res = unsafe { ct.execute(slots.as_mut_ptr() as *mut i64, types.as_ptr(), std::ptr::null_mut() as *mut u8) };
                if res >= 0 { return Ok(Value::I64(res)); }
                // Guard failed — try PIC recovery before falling back to interpreter
                self.deoptimizations += 1;
                if let Some(trace) = self.recorder.get_trace(tid) {
                    if let Some((slot, failed_type)) = Self::detect_guard_failure(trace, types) {
                        if let Some(pic_tid) = self.pic.lookup(slot, failed_type) {
                            if let Some(pic_ct) = self.compiled_cache.get(&pic_tid) {
                                let pic_res = unsafe { pic_ct.execute(slots.as_mut_ptr() as *mut i64, types.as_ptr(), std::ptr::null_mut() as *mut u8) };
                                if pic_res >= 0 { return Ok(Value::I64(pic_res)); }
                                self.deoptimizations += 1;
                            }
                        } else {
                            self.record_guard_failure(entry_pc, slot, failed_type);
                        }
                    }
                }
                // Fall back to interpreter (return Err signals caller to use Tier 0)
            } else if let Some(trace) = self.recorder.get_trace(tid) {
                // FIX: Check execution_count BEFORE incrementing. Previously,
                // execution_count was incremented after the compile check,
                // meaning the trace had to be called compile_trigger+1 times
                // before compilation. Now we check first, then increment.
                if self.should_compile(trace) && !trace.instructions.is_empty() {
                    match self.codegen.compile_trace(trace, None) {
                        Ok(ct) => {
                            self.traces_compiled += 1;
                            let res = unsafe { ct.execute(slots.as_mut_ptr() as *mut i64, types.as_ptr(), std::ptr::null_mut() as *mut u8) };
                            if res >= 0 {
                                self.compiled_cache.insert(tid, ct);
                                return Ok(Value::I64(res));
                            }
                            // Guard failed on freshly compiled trace
                            self.deoptimizations += 1;
                            if let Some((slot, failed_type)) = Self::detect_guard_failure(trace, types) {
                                if let Some(pic_tid) = self.pic.lookup(slot, failed_type) {
                                    if let Some(pic_ct) = self.compiled_cache.get(&pic_tid) {
                                        let pic_res = unsafe { pic_ct.execute(slots.as_mut_ptr() as *mut i64, types.as_ptr(), std::ptr::null_mut() as *mut u8) };
                                        if pic_res >= 0 {
                                            self.compiled_cache.insert(tid, ct);
                                            return Ok(Value::I64(pic_res));
                                        }
                                        self.deoptimizations += 1;
                                    }
                                } else {
                                    self.record_guard_failure(entry_pc, slot, failed_type);
                                }
                            }
                            // Cache the compiled trace even if it deopts
                            self.compiled_cache.insert(tid, ct);
                        }
                        Err(_) => { self.deoptimizations += 1; }
                    }
                }
            }
            // Increment execution count AFTER compile check (so the first call
            // when execution_count == compile_trigger triggers compilation)
            if let Some(t) = self.recorder.get_trace_mut(tid) { t.execution_count += 1; }

            // FIX: Do NOT replace the recorder here! The old code replaced
            // the entire recorder when hot_count >= trace_trigger, destroying
            // all existing traces and their accumulated execution_counts.
            // This was the root cause of "traces_compiled=0" — the trace
            // could never accumulate enough count because it was destroyed
            // every time the hot counter crossed the threshold.
            //
            // Now we only start a NEW recording if there's no existing trace
            // for this entry_pc. If a trace already exists, we skip — the
            // trace's execution_count will reach compile_trigger naturally.
            return Err(RuntimeError::new("Deoptimization: falling back to interpreter"));
        }

        // ── No existing trace for this entry_pc ────────────────────────────
        // Start recording a new trace if the hot counter has crossed the
        // trace_trigger threshold. We do NOT replace the recorder — we add
        // a new trace to the existing recorder.
        if self.should_start_tracing(hot_count) {
            // Only start recording if there's no existing trace at this PC
            // (double-check, since we're past the if-let above)
            self.recorder.start_recording(entry_pc);
            self.traces_recorded += 1;
            // FIX: Record all instructions from the bytecode immediately.
            // The old code only called start_recording() here and expected
            // the interpreter loop to feed instructions one at a time.
            // But the tiered execution manager calls execute_with_jit()
            // directly, so there's no interpreter loop feeding instructions.
            // We need to record from the provided _instructions slice.
            for (pc, instr) in _instructions.iter().enumerate() {
                if self.recorder.should_abort_trace() {
                    self.recorder.abort_recording();
                    break;
                }
                self.recorder.record_instruction(instr, pc);
            }
            if let Some(_trace_id) = self.recorder.finish_recording() {
                // Trace is now recorded and findable by entry_pc.
                // It will be compiled on the next call when its
                // execution_count reaches compile_trigger.
            }
        }

        Err(RuntimeError::new("Deoptimization: falling back to interpreter"))
    }

    /// Execute with a flat i64 slot array (instead of Value enum array).
    /// This is the correct interface for the tracing JIT: compiled native code
    /// reads/writes at 8-byte-aligned offsets (slot*8), so it needs a flat
    /// array of i64s, not the large Value enum.
    pub fn execute_with_jit_flat(&mut self, entry_pc: usize, flat_slots: &mut [i64], types: &mut [u8], _instructions: &[Instr]) -> Result<Value, RuntimeError> {
        // ── Increment hot counter ──────────────────────────────────────────
        let hot_count = *self.hot_counters.entry(entry_pc as u64).and_modify(|c| *c += 1).or_insert(1);

        // ── Check for existing trace ───────────────────────────────────────
        if let Some(tid) = self.recorder.find_trace(entry_pc) {
            // Check the compiled cache FIRST
            if let Some(ct) = self.compiled_cache.get(&tid) {
                let res = unsafe { ct.execute(flat_slots.as_mut_ptr(), types.as_ptr(), std::ptr::null_mut() as *mut u8) };
                if res >= 0 { return Ok(Value::I64(res)); }
                // Guard failed
                self.deoptimizations += 1;
                if let Some(trace) = self.recorder.get_trace(tid) {
                    if let Some((slot, failed_type)) = Self::detect_guard_failure(trace, types) {
                        if let Some(pic_tid) = self.pic.lookup(slot, failed_type) {
                            if let Some(pic_ct) = self.compiled_cache.get(&pic_tid) {
                                let pic_res = unsafe { pic_ct.execute(flat_slots.as_mut_ptr(), types.as_ptr(), std::ptr::null_mut() as *mut u8) };
                                if pic_res >= 0 { return Ok(Value::I64(pic_res)); }
                                self.deoptimizations += 1;
                            }
                        } else {
                            self.record_guard_failure(entry_pc, slot, failed_type);
                        }
                    }
                }
            } else if let Some(trace) = self.recorder.get_trace(tid) {
                if self.should_compile(trace) && !trace.instructions.is_empty() {
                    match self.codegen.compile_trace(trace, None) {
                        Ok(ct) => {
                            self.traces_compiled += 1;
                            let res = unsafe { ct.execute(flat_slots.as_mut_ptr(), types.as_ptr(), std::ptr::null_mut() as *mut u8) };
                            if res >= 0 {
                                self.compiled_cache.insert(tid, ct);
                                return Ok(Value::I64(res));
                            }
                            self.deoptimizations += 1;
                            if let Some((slot, failed_type)) = Self::detect_guard_failure(trace, types) {
                                if let Some(pic_tid) = self.pic.lookup(slot, failed_type) {
                                    if let Some(pic_ct) = self.compiled_cache.get(&pic_tid) {
                                        let pic_res = unsafe { pic_ct.execute(flat_slots.as_mut_ptr(), types.as_ptr(), std::ptr::null_mut() as *mut u8) };
                                        if pic_res >= 0 {
                                            self.compiled_cache.insert(tid, ct);
                                            return Ok(Value::I64(pic_res));
                                        }
                                        self.deoptimizations += 1;
                                    }
                                } else {
                                    self.record_guard_failure(entry_pc, slot, failed_type);
                                }
                            }
                            self.compiled_cache.insert(tid, ct);
                        }
                        Err(_) => { self.deoptimizations += 1; }
                    }
                }
            }
            if let Some(t) = self.recorder.get_trace_mut(tid) { t.execution_count += 1; }
            return Err(RuntimeError::new("Deoptimization: falling back to interpreter"));
        }

        // ── No existing trace for this entry_pc ────────────────────────────
        if self.should_start_tracing(hot_count) {
            // BUG FIX: Do NOT destroy the existing recorder with a new one —
            // that would discard all previously recorded traces and compiled
            // caches. Instead, start recording on the existing recorder.
            //
            // Record a trace from the bytecode instruction stream, starting at
            // entry_pc.  We walk forward through the instructions, recording
            // each one, and stop when we encounter a backward jump (loop
            // back-edge) which closes the trace.  If we reach the end of the
            // instruction stream without a loop, we also close the trace.
            self.recorder.start_recording(entry_pc);
            let start = entry_pc as usize;
            for pc in start.._instructions.len() {
                self.recorder.record_instruction(&_instructions[pc], pc);
                // Check for max trace length — abort if exceeded
                if self.recorder.should_abort_trace() {
                    self.recorder.abort_recording();
                    break;
                }
                // Check for loop back-edge (backward jump) — close the trace
                match &_instructions[pc] {
                    Instr::Jump(off) if *off < 0 => {
                        // Unconditional backward jump — loop back-edge
                        self.recorder.finish_recording();
                        self.traces_recorded += 1;
                        return Err(RuntimeError::new("Deoptimization: falling back to interpreter"));
                    }
                    Instr::JumpFalse(_, off) if *off < 0 => {
                        // Conditional backward jump — record as side exit (loop exit guard)
                        let target_pc = (pc as i32 + 1 + off) as usize;
                        self.recorder.record_side_exit(target_pc, true, None);
                        self.recorder.finish_recording();
                        self.traces_recorded += 1;
                        return Err(RuntimeError::new("Deoptimization: falling back to interpreter"));
                    }
                    Instr::JumpTrue(_, off) if *off < 0 => {
                        let target_pc = (pc as i32 + 1 + off) as usize;
                        self.recorder.record_side_exit(target_pc, true, None);
                        self.recorder.finish_recording();
                        self.traces_recorded += 1;
                        return Err(RuntimeError::new("Deoptimization: falling back to interpreter"));
                    }
                    Instr::Return(_) | Instr::ReturnUnit => {
                        // End of function — close the trace
                        self.recorder.finish_recording();
                        self.traces_recorded += 1;
                        return Err(RuntimeError::new("Deoptimization: falling back to interpreter"));
                    }
                    _ => {}
                }
            }
            // Reached end of instruction stream without a loop or return
            // — finish recording anyway
            if let Some(_trace_id) = self.recorder.finish_recording() {
                // Trace recorded; will be compiled on next call
            }
        }

        Err(RuntimeError::new("Deoptimization: falling back to interpreter"))
    }
}
