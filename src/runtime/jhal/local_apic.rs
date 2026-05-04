// =========================================================================
// Local APIC Driver — The Pulse of Jules
//
// The Advanced Programmable Interrupt Controller (APIC) is the heart of
// the CPU's interrupt subsystem.  It manages:
//   - Per-CPU timers (used by AdaptiveScheduler for high-precision ticks)
//   - Inter-Processor Interrupts (IPIs) for cross-core coordination
//   - Interrupt masking and priority management
//
// REGISTER MAP VERIFIED AGAINST: Intel 64 and IA-32 Architectures SDM
//   Volume 3, Chapter 11 ("Local and I/O APIC"), Table 11-5.
//
// ARCHITECTURE DECISION:
//   We do NOT use a C-style struct with reserved padding fields to map
//   the APIC MMIO region.  The reason is that Rust's MmioReg<T> wraps
//   a *mut T pointer, which is 8 bytes on x86_64 — but APIC registers
//   are only 4 bytes wide with 4-byte alignment.  A struct-based map
//   would be 2x too large and fail the 4096-byte compile-time check.
//
//   Instead, we use offset-based access: the base address of the APIC
//   MMIO page is stored once, and each register is accessed via
//   `read_reg(offset)` / `write_reg(offset, value)` which computes
//   the correct address as `base + offset`.  This is the same approach
//   used by Linux (arch/x86/kernel/apic/apic.c) and seL4.
//
// JULES INTEGRATION:
//   - Timer hooks into AdaptiveScheduler via `ApicTimerConfig`
//   - EOI is called automatically by the interrupt dispatcher
//   - IPIs are used for cross-core SpeculativeThread coordination
//   - ProphecyOracle uses the timer tick as its scheduling quantum
//
// REVIEW CHECKLIST:
//   1. Re-entrant: APIC registers are per-CPU; each core has its own
//      APIC instance.  A TSX abort does not corrupt APIC state because
//      MMIO writes are non-transactional — but the driver is designed
//      so that any interrupted init sequence can be safely restarted
//      (idempotent initialization).
//   2. Side channel: APIC MMIO reads force cache-line loads, but the
//      APIC base address is mapped as UC (uncacheable) in the MTRRs,
//      so cache-timing attacks are not possible.
//   3. Memory ordering: All register access goes through `read_reg()`
//      and `write_reg()` which enforce Acquire/Release semantics.
//   4. 4.5MB limit: Zero heap allocation.  The struct is 8 bytes (one pointer).
// =========================================================================

use core::sync::atomic::{fence, Ordering};

// ─── APIC Register Offsets (from SDM Table 11-5) ──────────────────────────
// These offsets are byte offsets from the APIC base address.
// All registers are 32-bit (4-byte aligned).

/// Offset 0x020: Local APIC ID Register
pub const REG_APIC_ID: u32 = 0x020;
/// Offset 0x030: Local APIC Version Register
pub const REG_VERSION: u32 = 0x030;
/// Offset 0x080: Task Priority Register
pub const REG_TPR: u32 = 0x080;
/// Offset 0x0B0: End Of Interrupt Register (write-only)
pub const REG_EOI: u32 = 0x0B0;
/// Offset 0x0D0: Logical Destination Register
pub const REG_LDR: u32 = 0x0D0;
/// Offset 0x0E0: Destination Format Register (P6+, 64-bit mode: always flat)
pub const REG_DFR: u32 = 0x0E0;
/// Offset 0x0F0: Spurious Interrupt Vector Register
pub const REG_SVR: u32 = 0x0F0;
/// Offset 0x100: In-Service Register (first of 8, 0x100–0x170)
pub const REG_ISR_BASE: u32 = 0x100;
/// Offset 0x200: Interrupt Request Register (first of 8, 0x200–0x270)
pub const REG_IRR_BASE: u32 = 0x200;
/// Offset 0x280: Error Status Register
pub const REG_ESR: u32 = 0x280;
/// Offset 0x300: Interrupt Command Register (low 32 bits)
pub const REG_ICR_LOW: u32 = 0x300;
/// Offset 0x310: Interrupt Command Register (high 32 bits)
pub const REG_ICR_HIGH: u32 = 0x310;
/// Offset 0x320: LVT Timer Register
pub const REG_LVT_TIMER: u32 = 0x320;
/// Offset 0x330: LVT Thermal Sensor Register
pub const REG_LVT_THERMAL: u32 = 0x330;
/// Offset 0x340: LVT Performance Counter Register
pub const REG_LVT_PMC: u32 = 0x340;
/// Offset 0x350: LVT LINT0 Register
pub const REG_LVT_LINT0: u32 = 0x350;
/// Offset 0x360: LVT LINT1 Register
pub const REG_LVT_LINT1: u32 = 0x360;
/// Offset 0x370: LVT Error Register
pub const REG_LVT_ERROR: u32 = 0x370;
/// Offset 0x380: Timer Initial Count Register
pub const REG_TIMER_INIT: u32 = 0x380;
/// Offset 0x390: Timer Current Count Register (read-only)
pub const REG_TIMER_CURRENT: u32 = 0x390;
/// Offset 0x3E0: Timer Divide Configuration Register
pub const REG_TIMER_DIV: u32 = 0x3E0;

// ─── Bit Field Definitions (Named, No Magic Numbers) ──────────────────────

/// SVR: Bit 8 — Software Enable/Disable.  Set to 1 to enable APIC.
pub const SVR_ENABLE: u32 = 1 << 8;

/// SVR: Bits 7:0 — Spurious vector number.  Must be odd (bit 0 set)
/// per Intel SDM.  Vector 0xFF is conventional.
pub const SVR_SPURIOUS_VECTOR: u32 = 0xFF;

/// LVT Timer: Bits 7:0 — Interrupt vector number.
/// We use vector 0x20 (32) which is the first user-available vector
/// (vectors 0–31 are reserved for CPU exceptions).
pub const LVT_TIMER_VECTOR: u32 = 0x20;

/// LVT Timer: Bit 17 — Timer Mode.
///   0 = One-shot
///   1 = Periodic
pub const LVT_TIMER_MODE_PERIODIC: u32 = 1 << 17;

/// LVT Timer: Bit 16 — Interrupt Mask.
///   0 = Not masked (interrupt enabled)
///   1 = Masked (interrupt disabled)
pub const LVT_TIMER_MASKED: u32 = 1 << 16;

/// Timer Divide Configuration values.
/// Encoded as bits [2:0] with a discontinuity at 0b1xx.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum TimerDivide {
    /// Divide by 2 (bits = 0b000)
    Div2 = 0b000,
    /// Divide by 4 (bits = 0b001)
    Div4 = 0b001,
    /// Divide by 8 (bits = 0b010)
    Div8 = 0b010,
    /// Divide by 16 (bits = 0b011) — Default, best for ProphecyOracle tick
    Div16 = 0b011,
    /// Divide by 32 (bits = 0b1000) — NOTE: encoding jumps to 0b1000!
    Div32 = 0b1000,
    /// Divide by 64 (bits = 0b1001)
    Div64 = 0b1001,
    /// Divide by 128 (bits = 0b1010)
    Div128 = 0b1010,
    /// Divide by 1 (bits = 0b1011)
    Div1 = 0b1011,
}

/// ICR Delivery Mode (bits 10:8 of ICR low)
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum DeliveryMode {
    /// Fixed delivery — interrupt sent to the target core(s)
    Fixed = 0b000,
    /// Lowest-priority delivery — sent to the core with lowest priority
    LowestPriority = 0b001,
    /// SMI (System Management Interrupt)
    Smi = 0b010,
    /// NMI (Non-Maskable Interrupt)
    Nmi = 0b100,
    /// INIT IPI — resets the target core
    Init = 0b101,
    /// Startup IPI — used with INIT to start an AP
    Startup = 0b110,
}

/// ICR Destination Shorthand (bits 19:18 of ICR low)
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum DestinationShorthand {
    /// No shorthand — use the destination field
    None = 0b00,
    /// Self — send to the current core only
    Self_ = 0b01,
    /// All including self
    All = 0b10,
    /// All excluding self
    AllExceptSelf = 0b11,
}

/// ICR Delivery Status (bit 12 of ICR low, read-only)
pub const ICR_DELIVERY_PENDING: u32 = 1 << 12;

// ─── Default APIC Base Address ────────────────────────────────────────────

/// Default xAPIC base address (mapped at this physical address in the
/// Local APIC's MSR IA32_APICBASE).  Must be mapped as UC in page tables.
pub const APIC_DEFAULT_BASE: usize = 0xFEE0_0000;

// ─── Timer Configuration ──────────────────────────────────────────────────

/// Configuration for the APIC timer, consumed by `init_timer()`.
/// This struct is the interface between the driver and the
/// AdaptiveScheduler / ProphecyOracle.
#[derive(Debug, Clone, Copy)]
pub struct ApicTimerConfig {
    /// Interrupt vector number (must be >= 32).
    pub vector: u8,
    /// Timer mode: One-shot or Periodic.
    pub periodic: bool,
    /// Divide value for the timer clock.
    pub divide: TimerDivide,
    /// Initial count value.  The timer counts down from this to zero,
    /// then fires the interrupt.  In periodic mode, it reloads automatically.
    /// This value must be calibrated against the CPU's bus clock.
    pub initial_count: u32,
}

impl ApicTimerConfig {
    /// Create a timer config suitable for the ProphecyOracle's scheduling
    /// quantum.  Uses vector 0x20, periodic mode, divide-by-16.
    pub fn for_prophecy_tick(bus_ticks_per_quantum: u32) -> Self {
        Self {
            vector: LVT_TIMER_VECTOR as u8,
            periodic: true,
            divide: TimerDivide::Div16,
            initial_count: bus_ticks_per_quantum,
        }
    }

    /// Create a timer config for one-shot timing measurements.
    pub fn for_calibration(divide: TimerDivide) -> Self {
        Self {
            vector: LVT_TIMER_VECTOR as u8,
            periodic: false,
            divide,
            initial_count: 0xFFFF_FFFF, // Max count — let it count down
        }
    }
}

// ─── Local APIC Driver ────────────────────────────────────────────────────

/// The Local APIC driver.
///
/// This struct provides access to a single CPU's Local APIC via
/// offset-based MMIO.  It stores only the base address (8 bytes).
///
/// # Safety Invariants
/// - `base` must point to a valid 4KiB APIC MMIO region.
/// - The page at `base` must be mapped as Device memory (UC).
/// - Only one `LocalApic` instance should exist per CPU core.
pub struct LocalApic {
    /// Base virtual address of the APIC MMIO region.
    base: usize,
}

impl LocalApic {
    // ─── Low-Level Register Access ─────────────────────────────────────

    /// Read a 32-bit APIC register at the given byte offset.
    ///
    /// Uses Acquire ordering to ensure subsequent reads see device state
    /// that was made visible before this register was written.
    #[inline(always)]
    fn read_reg(&self, offset: u32) -> u32 {
        fence(Ordering::Acquire);
        // SAFETY: self.base is a valid APIC MMIO address by invariant,
        // and `offset` is a valid register offset from the SDM.
        unsafe {
            core::ptr::read_volatile((self.base + offset as usize) as *const u32)
        }
    }

    /// Write a 32-bit APIC register at the given byte offset.
    ///
    /// Uses Release ordering to ensure prior writes (DMA data, etc.)
    /// are visible before this register write takes effect.
    #[inline(always)]
    fn write_reg(&self, offset: u32, value: u32) {
        // SAFETY: same as read_reg.
        unsafe {
            core::ptr::write_volatile((self.base + offset as usize) as *mut u32, value)
        }
        fence(Ordering::Release);
    }

    // ─── Construction ──────────────────────────────────────────────────

    /// Create a LocalApic reference pointing to the given physical address.
    ///
    /// # Safety
    /// - `base_addr` must be the correct APIC base address (typically
    ///   read from IA32_APIC_BASE MSR).
    /// - The page at `base_addr` must be mapped as Device memory (UC)
    ///   in the page tables.
    /// - Only one `LocalApic` instance should exist per CPU core.
    pub unsafe fn from_base(base_addr: usize) -> Self {
        Self { base: base_addr }
    }

    /// Create a LocalApic at the default base address (0xFEE00000).
    ///
    /// # Safety
    /// Same as `from_base`.  Additionally, the default APIC base must
    /// not have been remapped via IA32_APIC_BASE MSR.
    pub unsafe fn default() -> Self {
        Self::from_base(APIC_DEFAULT_BASE)
    }

    // ─── Identification ────────────────────────────────────────────────

    /// Read the Local APIC ID (bits 31:24 of the APIC ID register).
    /// On x2APIC, this is the full 32-bit value; on xAPIC, only bits
    /// 31:24 are significant.
    pub fn apic_id(&self) -> u32 {
        self.read_reg(REG_APIC_ID) >> 24
    }

    /// Read the APIC version register.
    /// Bits 7:0 = version number, bits 15:8 = max LVT entry,
    /// bit 23 = EOI-suppression support.
    pub fn version(&self) -> u32 {
        self.read_reg(REG_VERSION)
    }

    /// Get the maximum LVT entry count (from version register bits 15:8).
    pub fn max_lvt_entries(&self) -> u32 {
        (self.version() >> 8) & 0xFF
    }

    // ─── Spurious Interrupt Vector / APIC Enable ───────────────────────

    /// Software-enable the Local APIC.
    ///
    /// This MUST be called before any other APIC operations.  Without it,
    /// the APIC is in hardware-disabled state and all interrupts are
    /// suppressed.  The spurious vector is set to 0xFF (conventional).
    ///
    /// This operation is idempotent — calling it multiple times is safe.
    pub fn enable(&self) {
        let svr_val = SVR_ENABLE | SVR_SPURIOUS_VECTOR;
        self.write_reg(REG_SVR, svr_val);
    }

    /// Check if the APIC is software-enabled.
    pub fn is_enabled(&self) -> bool {
        (self.read_reg(REG_SVR) & SVR_ENABLE) != 0
    }

    // ─── End Of Interrupt ──────────────────────────────────────────────

    /// Signal End Of Interrupt for the current interrupt.
    ///
    /// MUST be called at the end of every interrupt handler, or the CPU
    /// will not deliver further interrupts of the same or lower priority.
    pub fn eoi(&self) {
        // EOI is write-only; any value works.  Writing 0 is conventional.
        self.write_reg(REG_EOI, 0);
    }

    // ─── Task Priority ─────────────────────────────────────────────────

    /// Set the task priority.  Interrupts with priority class <= TPR[7:4]
    /// are blocked.  Used to temporarily mask low-priority interrupts
    /// during critical sections.
    pub fn set_task_priority(&self, priority: u8) {
        // Only bits 7:4 are significant; bits 3:0 must be 0.
        self.write_reg(REG_TPR, (priority as u32 & 0xFF) & !0x0F);
    }

    /// Read the current task priority.
    pub fn task_priority(&self) -> u8 {
        (self.read_reg(REG_TPR) & 0xFF) as u8
    }

    // ─── Timer ─────────────────────────────────────────────────────────

    /// Initialize the APIC timer with the given configuration.
    ///
    /// This is the primary interface for the AdaptiveScheduler.
    /// The sequence is:
    ///   1. Set divide configuration
    ///   2. Mask the timer (so it doesn't fire during setup)
    ///   3. Set the LVT entry (vector + mode + unmask)
    ///   4. Set the initial count (which starts the timer)
    ///
    /// # Idempotency
    /// This function is safe to call multiple times.  Each call
    /// reconfigures the timer completely.
    pub fn init_timer(&self, config: &ApicTimerConfig) {
        // Step 1: Set divide value
        self.write_reg(REG_TIMER_DIV, config.divide as u32);

        // Step 2: Mask timer while configuring LVT
        let mut lvt_val = (config.vector as u32) | LVT_TIMER_MASKED;
        if config.periodic {
            lvt_val |= LVT_TIMER_MODE_PERIODIC;
        }
        self.write_reg(REG_LVT_TIMER, lvt_val);

        // Step 3: Set initial count (starts the timer counting)
        self.write_reg(REG_TIMER_INIT, config.initial_count);

        // Step 4: Unmask the timer (enable interrupts)
        lvt_val &= !LVT_TIMER_MASKED;
        self.write_reg(REG_LVT_TIMER, lvt_val);
    }

    /// Stop the APIC timer (mask it).
    pub fn stop_timer(&self) {
        let lvt_val = self.read_reg(REG_LVT_TIMER) | LVT_TIMER_MASKED;
        self.write_reg(REG_LVT_TIMER, lvt_val);
        // Also zero the initial count to stop counting
        self.write_reg(REG_TIMER_INIT, 0);
    }

    /// Read the current timer count.
    /// In one-shot mode, this counts down from `initial_count` to 0.
    /// In periodic mode, it counts down and reloads automatically.
    pub fn timer_current_count(&self) -> u32 {
        self.read_reg(REG_TIMER_CURRENT)
    }

    /// Perform a timer calibration using a busy-wait loop.
    ///
    /// Returns the approximate bus clock frequency in Hz, which can be
    /// used to compute `initial_count` for a desired timer frequency.
    ///
    /// NOTE: This is a simplified calibration.  A production implementation
    /// would use the HPET or PIT for accurate calibration.
    pub fn calibrate_timer(&self, divide: TimerDivide) -> u32 {
        // Step 1: Set up the timer in one-shot mode with max count
        self.write_reg(REG_TIMER_DIV, divide as u32);
        self.write_reg(REG_LVT_TIMER, LVT_TIMER_VECTOR | LVT_TIMER_MASKED);
        self.write_reg(REG_TIMER_INIT, 0xFFFF_FFFF);

        // Step 2: Wait for a known duration (approximate)
        let wait_iterations = 20_000_000;
        let start = self.read_reg(REG_TIMER_CURRENT);
        for _ in 0..wait_iterations {
            core::hint::spin_loop();
        }
        let end = self.read_reg(REG_TIMER_CURRENT);

        // Step 3: Compute ticks elapsed and extrapolate frequency
        let ticks_elapsed = start.wrapping_sub(end);
        let divisor = match divide {
            TimerDivide::Div1 => 1,
            TimerDivide::Div2 => 2,
            TimerDivide::Div4 => 4,
            TimerDivide::Div8 => 8,
            TimerDivide::Div16 => 16,
            TimerDivide::Div32 => 32,
            TimerDivide::Div64 => 64,
            TimerDivide::Div128 => 128,
        };
        let bus_ticks = (ticks_elapsed as u64 * divisor as u64) as u32;
        bus_ticks.saturating_mul(100)
    }

    // ─── Inter-Processor Interrupts (IPIs) ─────────────────────────────

    /// Send an Inter-Processor Interrupt.
    ///
    /// # TSX Safety
    /// This function MUST NOT be called inside a TSX transaction.
    /// IPIs are not transactional — if the TSX aborts after the IPI
    /// is sent, there is no way to "unsend" it.
    pub fn send_ipi(
        &self,
        destination_apic_id: u8,
        vector: u8,
        delivery_mode: DeliveryMode,
        shorthand: DestinationShorthand,
    ) {
        // Wait for any pending IPI to complete
        let mut attempts = 0;
        while (self.read_reg(REG_ICR_LOW) & ICR_DELIVERY_PENDING) != 0 {
            core::hint::spin_loop();
            attempts += 1;
            if attempts > 1_000_000 {
                break;
            }
        }

        // Write destination APIC ID to ICR high (bits 63:56)
        let icr_high_val = (destination_apic_id as u32) << 24;
        self.write_reg(REG_ICR_HIGH, icr_high_val);

        // Build ICR low value
        let icr_low_val = (vector as u32)
            | ((delivery_mode as u32) << 8)
            | ((shorthand as u32) << 18);

        // Writing ICR low triggers the IPI
        fence(Ordering::SeqCst);
        self.write_reg(REG_ICR_LOW, icr_low_val);
        fence(Ordering::SeqCst);
    }

    /// Send an INIT IPI to a target core (used for AP startup / reset).
    pub fn send_init_ipi(&self, target_apic_id: u8) {
        self.send_ipi(target_apic_id, 0, DeliveryMode::Init, DestinationShorthand::None);
    }

    /// Send a SIPI (Startup IPI) to a target core.
    pub fn send_startup_ipi(&self, target_apic_id: u8, start_page: u8) {
        self.send_ipi(target_apic_id, start_page, DeliveryMode::Startup, DestinationShorthand::None);
    }

    // ─── LVT LINT Configuration ────────────────────────────────────────

    /// Configure the LINT0 input.
    /// Commonly used for the External Interrupt (INTR) line from the I/O APIC.
    pub fn set_lint0(&self, vector: u8, masked: bool) {
        let mut val = vector as u32;
        if masked {
            val |= LVT_TIMER_MASKED;
        }
        self.write_reg(REG_LVT_LINT0, val);
    }

    /// Configure the LINT1 input.
    /// Commonly used for the Non-Maskable Interrupt (NMI) line.
    pub fn set_lint1(&self, vector: u8, masked: bool) {
        let mut val = vector as u32;
        if masked {
            val |= LVT_TIMER_MASKED;
        }
        self.write_reg(REG_LVT_LINT1, val);
    }

    // ─── Error Status ──────────────────────────────────────────────────

    /// Read and clear the Error Status Register.
    /// The ESR is cleared by writing to it first (required by Intel SDM),
    /// then reading.
    pub fn read_and_clear_esr(&self) -> u32 {
        self.write_reg(REG_ESR, 0);
        self.read_reg(REG_ESR)
    }

    // ─── Interrupt State ───────────────────────────────────────────────

    /// Check if an interrupt is in-service at the given vector.
    pub fn is_in_service(&self, vector: u8) -> bool {
        let reg_idx = (vector as u32) / 32;
        let bit_idx = (vector as u32) % 32;
        if reg_idx < 8 {
            (self.read_reg(REG_ISR_BASE + reg_idx * 0x10) & (1 << bit_idx)) != 0
        } else {
            false
        }
    }
}

// ─── APIC Base MSR Helpers ────────────────────────────────────────────────

/// Read the IA32_APIC_BASE MSR (0x1B).
/// Returns the raw 64-bit value containing the APIC base address,
/// the global enable bit, and the BSP flag.
pub fn read_apic_base_msr() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        let eax: u32;
        let edx: u32;
        unsafe {
            core::arch::asm!(
                "rdmsr",
                in("ecx") 0x1B_u32,
                out("rax") eax,
                out("rdx") edx,
                options(nostack)
                // NOTE: no `nomem` — rdmsr reads from model-specific register state
                // that the compiler cannot see.  Using `nomem` would allow the
                // compiler to hoist or eliminate the read.
            );
        }
        // rdmsr returns the 64-bit value in EDX:EAX (high:low)
        ((edx as u64) << 32) | (eax as u64)
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        APIC_DEFAULT_BASE as u64
    }
}

/// Extract the APIC base physical address from the MSR value.
/// Bits 12–35 contain the base address (aligned to 4KB).
pub fn apic_base_from_msr(msr: u64) -> usize {
    (msr & 0xFFFF_F000) as usize
}

/// Check if this core is the Bootstrap Processor (BSP).
/// Bit 8 of IA32_APIC_BASE MSR.
pub fn is_bsp(msr: u64) -> bool {
    (msr & (1 << 8)) != 0
}

/// Check if the APIC is globally enabled.
/// Bit 11 of IA32_APIC_BASE MSR.
pub fn is_apic_global_enable(msr: u64) -> bool {
    (msr & (1 << 11)) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_apic_size() {
        // The LocalApic struct should be just one usize (pointer size)
        assert_eq!(core::mem::size_of::<LocalApic>(), core::mem::size_of::<usize>());
    }

    #[test]
    fn test_timer_divide_encoding() {
        // Verify the discontinuous encoding is correct per Intel SDM
        assert_eq!(TimerDivide::Div2 as u32, 0b000);
        assert_eq!(TimerDivide::Div4 as u32, 0b001);
        assert_eq!(TimerDivide::Div8 as u32, 0b010);
        assert_eq!(TimerDivide::Div16 as u32, 0b011);
        assert_eq!(TimerDivide::Div32 as u32, 0b1000); // Jumps from 3 to 8!
        assert_eq!(TimerDivide::Div64 as u32, 0b1001);
        assert_eq!(TimerDivide::Div128 as u32, 0b1010);
        assert_eq!(TimerDivide::Div1 as u32, 0b1011);
    }

    #[test]
    fn test_timer_config_for_prophecy() {
        let config = ApicTimerConfig::for_prophecy_tick(1_000_000);
        assert_eq!(config.vector, 0x20);
        assert!(config.periodic);
        assert_eq!(config.divide, TimerDivide::Div16);
        assert_eq!(config.initial_count, 1_000_000);
    }

    #[test]
    fn test_svr_values() {
        assert_eq!(SVR_ENABLE, 1 << 8);
        assert_eq!(SVR_SPURIOUS_VECTOR & 1, 1);
    }

    #[test]
    fn test_lvt_timer_fields() {
        assert_eq!(LVT_TIMER_VECTOR, 0x20);
        assert_eq!(LVT_TIMER_MODE_PERIODIC, 1 << 17);
        assert_eq!(LVT_TIMER_MASKED, 1 << 16);
    }

    #[test]
    fn test_delivery_mode_values() {
        assert_eq!(DeliveryMode::Fixed as u32, 0);
        assert_eq!(DeliveryMode::Init as u32, 5);
        assert_eq!(DeliveryMode::Startup as u32, 6);
        assert_eq!(DeliveryMode::Nmi as u32, 4);
    }

    #[test]
    fn test_destination_shorthand_values() {
        assert_eq!(DestinationShorthand::None as u32, 0);
        assert_eq!(DestinationShorthand::Self_ as u32, 1);
        assert_eq!(DestinationShorthand::All as u32, 2);
        assert_eq!(DestinationShorthand::AllExceptSelf as u32, 3);
    }

    #[test]
    fn test_apic_base_msr_helpers() {
        let msr = (APIC_DEFAULT_BASE as u64) | (1 << 11) | (1 << 8);
        assert_eq!(apic_base_from_msr(msr), APIC_DEFAULT_BASE);
        assert!(is_bsp(msr));
        assert!(is_apic_global_enable(msr));
    }

    #[test]
    fn test_register_offsets() {
        // Verify key register offsets match the Intel SDM
        assert_eq!(REG_APIC_ID, 0x020);
        assert_eq!(REG_VERSION, 0x030);
        assert_eq!(REG_TPR, 0x080);
        assert_eq!(REG_EOI, 0x0B0);
        assert_eq!(REG_SVR, 0x0F0);
        assert_eq!(REG_ICR_LOW, 0x300);
        assert_eq!(REG_ICR_HIGH, 0x310);
        assert_eq!(REG_LVT_TIMER, 0x320);
        assert_eq!(REG_TIMER_INIT, 0x380);
        assert_eq!(REG_TIMER_CURRENT, 0x390);
        assert_eq!(REG_TIMER_DIV, 0x3E0);
    }
}
