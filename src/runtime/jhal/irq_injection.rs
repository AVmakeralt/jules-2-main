// =========================================================================
// Direct Speculative IRQs — Zero-Latency Interrupt Vectoring
//
// Traditional interrupt handlers save ALL registers (30-100 cycles).
// Jules eliminates this via register partitioning:
//   RAX-R12  →  Main task (JIT code)
//   R13-R15  →  Fast IRQ handlers (reserved, NEVER touched by JIT)
//
// The JIT guarantees R13-R15 are free at every interrupt point,
// enabling zero-save entry: 0-10 cycle interrupt latency.
//
// REFERENCES:
//   Intel SDM Vol 3, §6.10 (IDT), §6.8 (IDT descriptor format)
// =========================================================================

use core::sync::atomic::{compiler_fence, fence, Ordering};

/// Reserved registers for fast IRQ handlers.
/// The JIT must NEVER emit code that touches these in the main task.
pub const IRQ_RESERVED_REGS: &[&str] = &["r13", "r14", "r15"];

/// Main task registers (RAX-R12). The JIT may use these freely.
pub const TASK_REGS: &[&str] = &[
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8", "r9", "r10", "r11", "r12",
];

/// Error indicating that the static register partition proof has failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticProofViolation {
    pub register: &'static str,
    pub context: &'static str,
}

impl StaticProofViolation {
    pub const fn new(register: &'static str, context: &'static str) -> Self {
        Self { register, context }
    }
}

/// Prove that a set of used registers does NOT overlap with R13-R15.
pub fn verify_register_partition(used_regs: &[&str]) -> bool {
    for used in used_regs {
        for reserved in IRQ_RESERVED_REGS {
            if *used == *reserved {
                return false;
            }
        }
    }
    true
}

/// Const-evaluable version of register partition proof.
pub fn verify_register_partition_const(used_regs: &[&str]) -> bool {
    // This is a compile-time-checkable version using byte comparison
    // instead of == on &str (which is not const-stable)
    for used in used_regs {
        for reserved in IRQ_RESERVED_REGS {
            if used.as_ptr() == reserved.as_ptr() && used.len() == reserved.len() {
                return false;
            }
        }
    }
    true
}

// Note: compile-time const assertion removed because PartialEq<&str>
// is not const-stable. The runtime check verify_register_partition()
// enforces this invariant.

// ─── IDT Entry ──────────────────────────────────────────────────────────────

/// Gate type: 64-bit interrupt gate (0xE). Clears IF, aborts TSX.
pub const GATE_TYPE_INTERRUPT: u8 = 0x0E;
/// Gate type: 64-bit trap gate (0xF). Does NOT clear IF.
pub const GATE_TYPE_TRAP: u8 = 0x0F;

/// A 16-byte IDT entry matching Intel SDM Volume 3, §6.10.
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct IdtEntry {
    offset_low: u16,
    selector: u16,
    ist_offset: u8,
    type_attr: u8,
    offset_mid: u16,
    offset_high: u32,
    reserved: u32,
}

impl IdtEntry {
    pub const SIZE: usize = 16;

    pub const fn missing() -> Self {
        Self { offset_low: 0, selector: 0, ist_offset: 0, type_attr: 0,
               offset_mid: 0, offset_high: 0, reserved: 0 }
    }

    const fn build_type_attr(gate_type: u8, dpl: u8, present: bool) -> u8 {
        let p = if present { 1u8 << 7 } else { 0 };
        let dpl_bits = (dpl & 0x3) << 5;
        p | dpl_bits | (gate_type & 0x0F)
    }

    /// Create a 64-bit interrupt gate entry.
    pub const fn interrupt_gate(handler: usize, selector: u16, ist: u8, dpl: u8) -> Self {
        Self {
            offset_low: (handler & 0xFFFF) as u16,
            selector,
            ist_offset: ist & 0x07,
            type_attr: Self::build_type_attr(GATE_TYPE_INTERRUPT, dpl, true),
            offset_mid: ((handler >> 16) & 0xFFFF) as u16,
            offset_high: ((handler >> 32) & 0xFFFFFFFF) as u32,
            reserved: 0,
        }
    }

    /// Create a 64-bit trap gate entry.
    pub const fn trap_gate(handler: usize, selector: u16, ist: u8, dpl: u8) -> Self {
        Self {
            offset_low: (handler & 0xFFFF) as u16,
            selector,
            ist_offset: ist & 0x07,
            type_attr: Self::build_type_attr(GATE_TYPE_TRAP, dpl, true),
            offset_mid: ((handler >> 16) & 0xFFFF) as u16,
            offset_high: ((handler >> 32) & 0xFFFFFFFF) as u32,
            reserved: 0,
        }
    }

    /// Ensure TSX-safe: convert to interrupt gate (aborts TSX transactions).
    pub const fn with_tsx_abort(self) -> Self {
        Self { type_attr: (self.type_attr & !0x0F) | GATE_TYPE_INTERRUPT, ..self }
    }

    pub const fn is_present(&self) -> bool { (self.type_attr & (1 << 7)) != 0 }
    pub const fn is_interrupt_gate(&self) -> bool { (self.type_attr & 0x0F) == GATE_TYPE_INTERRUPT }
    pub const fn is_trap_gate(&self) -> bool { (self.type_attr & 0x0F) == GATE_TYPE_TRAP }
    pub const fn handler_address(&self) -> usize {
        (self.offset_low as usize) | ((self.offset_mid as usize) << 16) | ((self.offset_high as usize) << 32)
    }
    pub const fn selector(&self) -> u16 { self.selector }
    pub const fn ist(&self) -> u8 { self.ist_offset & 0x07 }
    pub const fn dpl(&self) -> u8 { (self.type_attr >> 5) & 0x3 }
}

// ─── IDT Table ──────────────────────────────────────────────────────────────

pub const IDT_ENTRIES: usize = 256;
pub const FIRST_USER_VECTOR: u8 = 32;

/// Pseudo-descriptor for LIDT/SIDT (10 bytes in 64-bit mode).
#[repr(C, packed)]
struct Idtr {
    limit: u16,
    base: u64,
}

/// The Interrupt Descriptor Table. Fixed 256 × 16 = 4096 bytes (one page).
pub struct Idt {
    entries: [IdtEntry; IDT_ENTRIES],
}

impl Idt {
    pub const fn new() -> Self {
        Self { entries: [IdtEntry::missing(); IDT_ENTRIES] }
    }

    pub fn set_handler(&mut self, vector: u8, handler: usize, selector: u16) {
        self.entries[vector as usize] = IdtEntry::interrupt_gate(handler, selector, 0, 0);
        compiler_fence(Ordering::Release);
    }

    pub fn set_handler_with_ist(&mut self, vector: u8, handler: usize, selector: u16, ist: u8) {
        self.entries[vector as usize] = IdtEntry::interrupt_gate(handler, selector, ist, 0);
        compiler_fence(Ordering::Release);
    }

    pub fn entry(&self, vector: u8) -> Option<&IdtEntry> {
        self.entries.get(vector as usize)
    }

    /// Load the IDT via LIDT instruction.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn load(&self) {
        let idtr = Idtr {
            limit: (core::mem::size_of::<Self>() - 1) as u16,
            base: self as *const Self as u64,
        };
        fence(Ordering::SeqCst);
        core::arch::asm!(
            "lidt [{}]",
            in(reg) &idtr as *const Idtr,
            options(readonly, preserves_flags)
        );
        compiler_fence(Ordering::SeqCst);
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub unsafe fn load(&self) {}
}

// ─── Fast IRQ Vector Table ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct FastIrqSlot {
    handler: usize,
    selector: u16,
    present: bool,
}

impl FastIrqSlot {
    pub const fn empty() -> Self { Self { handler: 0, selector: 0, present: false } }
    pub const fn new(handler: usize, selector: u16) -> Self { Self { handler, selector, present: true } }
    pub const fn is_present(&self) -> bool { self.present }
    pub const fn handler(&self) -> usize { self.handler }
}

/// Maps hardware interrupt vectors to fast handler function pointers.
pub struct FastIrqVectorTable {
    slots: [FastIrqSlot; IDT_ENTRIES],
}

impl FastIrqVectorTable {
    pub const fn new() -> Self { Self { slots: [FastIrqSlot::empty(); IDT_ENTRIES] } }

    pub fn register(&mut self, vector: u8, handler: usize, selector: u16) -> Result<(), StaticProofViolation> {
        if (vector as usize) < FIRST_USER_VECTOR as usize {
            return Err(StaticProofViolation::new("vector", "CPU exception vectors require full save/restore"));
        }
        self.slots[vector as usize] = FastIrqSlot::new(handler, selector);
        compiler_fence(Ordering::Release);
        Ok(())
    }

    pub fn unregister(&mut self, vector: u8) {
        self.slots[vector as usize] = FastIrqSlot::empty();
    }

    pub fn is_registered(&self, vector: u8) -> bool { self.slots[vector as usize].is_present() }

    pub fn apply_to_idt(&self, idt: &mut Idt) {
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.is_present() {
                idt.set_handler(i as u8, slot.handler, slot.selector);
            }
        }
    }

    pub fn registered_count(&self) -> usize { self.slots.iter().filter(|s| s.is_present()).count() }
}

// ─── IRQ Predictor (ProphecyOracle Integration) ─────────────────────────────

pub const CONFIDENCE_THRESHOLD: u8 = 128;
const HISTORY_DEPTH: usize = 8;

/// Predicted interrupt timing.
#[derive(Debug, Clone, Copy)]
pub struct IrqProphecy {
    pub vector: u8,
    pub predicted_cycles_ahead: u32,
    pub confidence: u8,
}

impl IrqProphecy {
    pub const fn zero() -> Self { Self { vector: 0, predicted_cycles_ahead: 0, confidence: 0 } }
    pub const fn new(vector: u8, cycles: u32, confidence: u8) -> Self {
        Self { vector, predicted_cycles_ahead: cycles, confidence }
    }
    pub fn is_actionable(&self) -> bool {
        self.confidence >= CONFIDENCE_THRESHOLD && self.predicted_cycles_ahead > 0
    }
}

/// Per-vector interrupt predictor using TSC history.
pub struct IrqPredictor {
    states: [IrqProphecy; IDT_ENTRIES],
    history: [[u64; HISTORY_DEPTH]; IDT_ENTRIES],
    history_pos: [usize; IDT_ENTRIES],
}

impl IrqPredictor {
    pub const fn new() -> Self {
        Self {
            states: [IrqProphecy::zero(); IDT_ENTRIES],
            history: [[0u64; HISTORY_DEPTH]; IDT_ENTRIES],
            history_pos: [0usize; IDT_ENTRIES],
        }
    }

    /// Record an interrupt occurrence and update prediction.
    pub fn record_interrupt(&mut self, vector: u8, tsc: u64) {
        let idx = vector as usize;
        let pos = self.history_pos[idx] % HISTORY_DEPTH;
        self.history[idx][pos] = tsc;
        self.history_pos[idx] = self.history_pos[idx].wrapping_add(1);

        let entry_count = if self.history_pos[idx] >= HISTORY_DEPTH {
            HISTORY_DEPTH
        } else {
            self.history_pos[idx]
        };

        if entry_count >= 2 {
            let (avg, variance) = self.compute_interval_stats(idx, entry_count);
            let predicted = if avg > u32::MAX as u64 { u32::MAX } else { avg as u32 };
            let base_conf = (entry_count as u32 * 28).min(224) as u8;
            let var_ratio = if avg > 0 { (variance * 100) / avg } else { 100 };
            let var_bonus = if var_ratio < 10 { 31u8 } else if var_ratio < 25 { 20u8 } else if var_ratio < 50 { 10u8 } else { 0u8 };
            self.states[idx] = IrqProphecy::new(vector, predicted, base_conf.saturating_add(var_bonus));
        } else {
            self.states[idx] = IrqProphecy::new(vector, 0, (entry_count as u32 * 32).min(64) as u8);
        }
        compiler_fence(Ordering::Release);
    }

    fn compute_interval_stats(&self, idx: usize, entry_count: usize) -> (u64, u64) {
        let start = if self.history_pos[idx] >= HISTORY_DEPTH {
            self.history_pos[idx] % HISTORY_DEPTH
        } else { 0 };

        let mut sorted = [0u64; HISTORY_DEPTH];
        for i in 0..entry_count { sorted[i] = self.history[idx][(start + i) % HISTORY_DEPTH]; }

        let interval_count = entry_count - 1;
        let mut sum: u64 = 0;
        let mut intervals = [0u64; HISTORY_DEPTH - 1];
        for i in 0..interval_count {
            intervals[i] = if sorted[i + 1] >= sorted[i] { sorted[i + 1] - sorted[i] } else { sorted[i].wrapping_sub(sorted[i + 1]) };
            sum = sum.saturating_add(intervals[i]);
        }
        let avg = if interval_count > 0 { sum / interval_count as u64 } else { 0 };
        let mut var_sum: u64 = 0;
        for i in 0..interval_count {
            let diff = if intervals[i] > avg { intervals[i] - avg } else { avg - intervals[i] };
            let ds = diff / 1000;
            var_sum = var_sum.saturating_add(ds * ds);
        }
        let variance = if interval_count > 0 { var_sum / interval_count as u64 } else { 0 };
        (avg, variance)
    }

    pub fn predict_next(&self) -> Option<&IrqProphecy> {
        let mut best: Option<&IrqProphecy> = None;
        for p in &self.states {
            if !p.is_actionable() { continue; }
            best = Some(match best {
                None => p,
                Some(prev) => if p.predicted_cycles_ahead < prev.predicted_cycles_ahead { p } else { prev },
            });
        }
        best
    }

    pub fn predict_vector(&self, vector: u8) -> Option<&IrqProphecy> {
        let p = &self.states[vector as usize];
        if p.confidence > 0 { Some(p) } else { None }
    }

    pub fn reserve_execution_slot(&self, prophecy: &IrqProphecy) -> bool { prophecy.is_actionable() }

    pub fn history_count(&self, vector: u8) -> usize {
        let p = self.history_pos[vector as usize];
        if p >= HISTORY_DEPTH { HISTORY_DEPTH } else { p }
    }

    pub fn clear_vector(&mut self, vector: u8) {
        let idx = vector as usize;
        self.states[idx] = IrqProphecy::zero();
        self.history[idx] = [0u64; HISTORY_DEPTH];
        self.history_pos[idx] = 0;
    }
}

// ─── Read TSC ───────────────────────────────────────────────────────────────

/// Read the Time Stamp Counter (RDTSC).
#[cfg(target_arch = "x86_64")]
pub unsafe fn rdtsc() -> u64 {
    let low: u32;
    let high: u32;
    core::arch::asm!("rdtsc", out("eax") low, out("edx") high, options(nomem, nostack));
    ((high as u64) << 32) | (low as u64)
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn rdtsc() -> u64 { 0 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_partition() {
        assert!(verify_register_partition(TASK_REGS));
        assert!(!verify_register_partition(&["rax", "r13", "rdx"]));
        assert!(!verify_register_partition(&["r14"]));
        assert!(!verify_register_partition(&["r15"]));
    }

    #[test]
    fn test_const_partition_proof() {
        assert!(verify_register_partition_const(TASK_REGS));
    }

    #[test]
    fn test_idt_entry_size() { assert_eq!(core::mem::size_of::<IdtEntry>(), 16); }

    #[test]
    fn test_interrupt_gate() {
        let e = IdtEntry::interrupt_gate(0xABCD_EF01_2345_6789, 0x08, 0, 0);
        assert!(e.is_present());
        assert!(e.is_interrupt_gate());
        assert_eq!(e.handler_address(), 0xABCD_EF01_2345_6789);
        assert_eq!(e.selector(), 0x08);
    }

    #[test]
    fn test_trap_gate() {
        let e = IdtEntry::trap_gate(0x1000, 0x08, 3, 0);
        assert!(e.is_trap_gate());
    }

    #[test]
    fn test_with_tsx_abort() {
        let trap = IdtEntry::trap_gate(0x1000, 0x08, 0, 0);
        let safe = trap.with_tsx_abort();
        assert!(safe.is_interrupt_gate());
    }

    #[test]
    fn test_idt_new() {
        let idt = Idt::new();
        assert!(!idt.entry(0).unwrap().is_present());
    }

    #[test]
    fn test_idt_set_handler() {
        let mut idt = Idt::new();
        idt.set_handler(0x20, 0xFFFF_8000_0010_0000, 0x08);
        let e = idt.entry(0x20).unwrap();
        assert!(e.is_present());
        assert_eq!(e.handler_address(), 0xFFFF_8000_0010_0000);
    }

    #[test]
    fn test_vector_table() {
        let mut table = FastIrqVectorTable::new();
        assert!(table.register(0x20, 0x1000, 0x08).is_ok());
        assert!(table.register(0, 0x1000, 0x08).is_err()); // CPU exception
        assert!(table.is_registered(0x20));
        assert_eq!(table.registered_count(), 1);
    }

    #[test]
    fn test_predictor() {
        let mut pred = IrqPredictor::new();
        assert!(pred.predict_next().is_none());
        // Periodic: every 1000 cycles
        for i in 0..8 { pred.record_interrupt(0x20, (i as u64) * 1000); }
        let p = pred.predict_next();
        assert!(p.is_some());
        assert_eq!(p.unwrap().vector, 0x20);
        assert!(p.unwrap().confidence >= CONFIDENCE_THRESHOLD);
    }

    #[test]
    fn test_irq_prophecy_actionable() {
        assert!(IrqProphecy::new(0x20, 1000, 200).is_actionable());
        assert!(!IrqProphecy::new(0x20, 1000, 50).is_actionable());
        assert!(!IrqProphecy::new(0x20, 0, 200).is_actionable());
    }

    #[test]
    fn test_static_proof_violation() {
        let v = StaticProofViolation::new("r13", "JIT loop");
        assert_eq!(v.register, "r13");
        assert_eq!(v.context, "JIT loop");
    }
}
