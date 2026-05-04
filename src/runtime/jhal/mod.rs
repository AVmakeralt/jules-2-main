// =========================================================================
// Jules Hardware Abstraction Layer (JHAL)
//
// The JHAL is the single entry point for all hardware interaction in Jules.
// It provides:
//
//   1. Local APIC — Timer and interrupt management for the AdaptiveScheduler
//   2. UART 16550 — Serial console for early boot and runtime logging
//   3. PCIe Enumerator — Bus discovery for device registration
//   4. MMIO Primitives — Volatile register access with memory ordering
//   5. I/O Port Primitives — `outb`/`inb` with compiler fences
//
// DESIGN PHILOSOPHY ("Provable Hardware Modeling"):
//
//   In Linux, drivers use "defensive programming" — they check for errors
//   at every step and try to recover.  In Jules, we use "Provable Hardware
//   Modeling" — the driver IS the formal specification of the hardware.
//
//   This means:
//     - Every bit is named and typed (no magic numbers)
//     - Every register access enforces Acquire/Release ordering
//     - Every `unsafe` block is limited to raw port or pointer access
//     - All logic outside `unsafe` blocks is proven safe by the type system
//     - Zero heap allocation (respects the 4.5MB limit)
//
// TSX INTEGRATION:
//   All JHAL drivers are designed for use with Jules's TSX-based speculative
//   execution.  Operations that are not transactional (I/O port writes,
//   MMIO writes, IPI sends) are clearly marked and must NOT be called
//   inside a TSX transaction.  Per-CPU ring buffers provide TSX-safe
//   alternatives (write to buffer inside TX, flush outside TX).
//
// REVIEW CHECKLIST (4 Questions for Peer Reviewers):
//
//   1. Is it Re-entrant?
//      A SpeculativeThread can be interrupted while in any JHAL driver.
//      TSX will roll back the transactional state.  The only risk is
//      non-transactional I/O, which each driver mitigates.
//
//   2. Is there a Side Channel?
//      MMIO is mapped as UC (uncacheable), preventing cache-timing attacks.
//      Port I/O is inherently non-cacheable.  The AliasLayout engine
//      ensures MMIO regions are never mapped as WB.
//
//   3. Is the Memory Ordering Correct?
//      All MMIO writes use Release ordering.  All MMIO reads use Acquire.
//      Port I/O uses compiler_fence(SeqCst).  Write-then-read patterns
//      use fence(SeqCst) between operations.
//
//   4. Does it respect the 4.5MB limit?
//      Zero heap allocation.  All buffers are static arrays.
//      The LocalApic struct is 4KB (one MMIO page, not heap).
//      The Console has 256 bytes per core (static array).
//      The DeviceRegistry has 256 entries × ~200 bytes (static).
// =========================================================================

pub mod io_port;
pub mod mmio;
pub mod local_apic;
pub mod serial_uart;
pub mod pcie;
pub mod sfi;
pub mod tsx;
pub mod irq_injection;
pub mod identity_map;

// Re-export the primary types for ergonomic access.

// I/O Port primitives
pub use io_port::{outb, outw, outl, inb, inw, inl, io_wait};

// MMIO primitives
pub use mmio::{MmioReg, MmioDevice};

// Local APIC
pub use local_apic::{
    LocalApic,
    ApicTimerConfig,
    TimerDivide,
    DeliveryMode,
    DestinationShorthand,
    APIC_DEFAULT_BASE,
    SVR_ENABLE,
    LVT_TIMER_VECTOR,
    LVT_TIMER_MODE_PERIODIC,
    LVT_TIMER_MASKED,
    read_apic_base_msr,
    apic_base_from_msr,
    is_bsp,
    is_apic_global_enable,
};

// UART 16550
pub use serial_uart::{
    SerialPort,
    SerialRingBuffer,
    BaudRate,
    Console,
    COM1, COM2, COM3, COM4,
};

// PCIe Enumerator
pub use pcie::{
    PciEnumerator,
    PciDevice,
    PciHeaderType,
    PciBdf,
    BoundedBus,
    BoundedDevice,
    BoundedFunction,
    DeviceRegistry,
    MAX_BUS, MAX_DEVICE, MAX_FUNCTION, MAX_BARS,
    read_pci_config,
    write_pci_config,
    read_pci_ecam,
    assign_device_latency_prophecies,
};

// Software Fault Isolation
pub use sfi::{
    SfiConfig,
    DEFAULT_SANCTUARY_BASE,
    DEFAULT_SANCTUARY_SIZE,
    emit_sfi_mask_inline_asm,
    verify_sfi_invariant,
    verify_sfi_config,
    sfi_load_with_barrier,
    sfi_store_with_barrier,
    apply_mask_asm,
};

// TSX Transaction Wrappers & AMX
pub use tsx::{
    TsxStatus,
    TsxTransaction,
    xbegin, xend, xabort, xtest,
    tsx_begin_safe, tsx_commit_safe,
    is_in_software_transaction,
    prove_transaction_bound,
    L1D_CACHE_SIZE,
    AmxTileConfig,
    amx_init, amx_tile_release, amx_tile_load, amx_tile_store, amx_tdpbssd,
    scheduler_matmul_fallback,
};

// Direct Speculative IRQs
pub use irq_injection::{
    IRQ_RESERVED_REGS, TASK_REGS,
    StaticProofViolation,
    verify_register_partition, verify_register_partition_const,
    IdtEntry, Idt,
    IDT_ENTRIES, FIRST_USER_VECTOR,
    GATE_TYPE_INTERRUPT, GATE_TYPE_TRAP,
    FastIrqSlot, FastIrqVectorTable,
    IrqProphecy, IrqPredictor,
    CONFIDENCE_THRESHOLD,
    rdtsc,
};

// Identity Mapping, HugePage, IOMMU, NMI, CFI
pub use identity_map::{
    IdentityMap,
    HugePageAllocator, HugePageSize,
    IommuDropZone,
    NmiWatchdog, NMI_DEFAULT_INTERVAL,
    CfiJumpTable, CfiReport, verify_cfi_compliance,
    PTE_PRESENT, PTE_WRITABLE, PTE_HUGE_PAGE, PTE_NO_EXECUTE,
    SANCTUARY_SIZE,
};

// ─── JHAL Instance ────────────────────────────────────────────────────────

/// The top-level Jules Hardware Abstraction Layer.
///
/// This struct owns all hardware drivers and provides a unified interface.
/// It is designed to be instantiated once at boot and shared across all
/// cores via `Arc<Jhal>` or a static reference.
///
/// # 4.5MB Budget Analysis
/// - LocalApic: 4 KiB (MMIO, not counted)
/// - Console: ~64 KiB (256 cores × 256 bytes ring)
/// - DeviceRegistry: ~50 KiB (256 devices × ~200 bytes)
/// - Total: ~114 KiB — well within budget
pub struct Jhal {
    /// Serial console.
    pub console: Console,
    /// PCIe device registry (populated during boot).
    pub device_registry: DeviceRegistry,
    /// Whether the APIC has been initialized.
    apic_initialized: bool,
}

impl Jhal {
    /// Create a new JHAL instance.
    ///
    /// This initializes:
    /// - The serial console on COM1 at 115200 baud
    /// - An empty device registry (call `enumerate_pci()` to populate)
    ///
    /// The Local APIC is NOT initialized here because it requires
    /// per-CPU setup.  Call `init_apic()` from each CPU's boot code.
    pub fn new() -> Self {
        Self {
            console: Console::new(),
            device_registry: DeviceRegistry::new(),
            apic_initialized: false,
        }
    }

    /// Initialize the Local APIC for the current CPU.
    ///
    /// # Safety
    /// - Must be called once per CPU during boot.
    /// - The APIC must be mapped as UC in the page tables.
    pub unsafe fn init_apic(&mut self, timer_config: &ApicTimerConfig) {
        let apic = LocalApic::default();
        apic.enable();
        apic.init_timer(timer_config);
        self.apic_initialized = true;

        // Log APIC initialization (no format! — zero heap)
        self.console.write_line_direct("[JHAL] Local APIC initialized");
        // Use a static buffer for hex formatting without heap allocation
        let mut buf = [0u8; 64];
        let msg = jhal_format_apic_info(apic.apic_id(), apic.version(), &mut buf);
        self.console.write_line_direct(msg);
    }

    /// Enumerate all PCI devices.
    ///
    /// Populates the device registry and assigns latency prophecies.
    pub fn enumerate_pci(&mut self) {
        self.console.write_line_direct("[JHAL] Starting PCI enumeration...");

        let mut enumerator = PciEnumerator::new(false); // Use port I/O
        enumerator.enumerate();

        // Copy discovered devices to our registry
        let source = enumerator.registry();
        for i in 0..source.len() {
            if let Some(device) = source.get(i) {
                self.device_registry.register(device.clone());
            }
        }

        // Assign latency prophecies
        assign_device_latency_prophecies(&self.device_registry);

        // Log completion (no format! — zero heap)
        let mut buf = [0u8; 64];
        let msg = jhal_format_enum_result(self.device_registry.len(), &mut buf);
        self.console.write_line_direct(msg);
    }

    /// Check if the APIC has been initialized.
    pub fn is_apic_initialized(&self) -> bool {
        self.apic_initialized
    }

    /// Flush all console buffers.
    pub fn flush_console(&self) {
        self.console.flush_all();
    }
}

impl Default for Jhal {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Zero-Heap Formatting Helpers ─────────────────────────────────────────
//
// These functions format integers into a stack-allocated byte buffer
// WITHOUT using format!() or String, which would allocate on the heap.
// They return a &str slice into the provided buffer.

/// Format APIC info: "[JHAL] APIC ID: N, Version: 0xHHHHHHHH"
fn jhal_format_apic_info<'a>(apic_id: u32, version: u32, buf: &'a mut [u8; 64]) -> &'a str {
    let prefix = b"[JHAL] APIC ID: ";
    let mut pos = 0;
    for &b in prefix {
        if pos < buf.len() { buf[pos] = b; pos += 1; }
    }
    pos = write_u32_decimal(buf, pos, apic_id);
    let sep = b", Version: 0x";
    for &b in sep {
        if pos < buf.len() { buf[pos] = b; pos += 1; }
    }
    pos = write_u32_hex(buf, pos, version);
    // Trim to actual length
    core::str::from_utf8(&buf[..pos]).unwrap_or("[JHAL] APIC initialized")
}

/// Format enum result: "[JHAL] PCI enumeration complete: N devices found"
fn jhal_format_enum_result<'a>(count: usize, buf: &'a mut [u8; 64]) -> &'a str {
    let prefix = b"[JHAL] PCI enum complete: ";
    let mut pos = 0;
    for &b in prefix {
        if pos < buf.len() { buf[pos] = b; pos += 1; }
    }
    pos = write_usize_decimal(buf, pos, count);
    let suffix = b" devices";
    for &b in suffix {
        if pos < buf.len() { buf[pos] = b; pos += 1; }
    }
    core::str::from_utf8(&buf[..pos]).unwrap_or("[JHAL] PCI enum complete")
}

/// Write a u32 in decimal to the buffer, returning the new position.
fn write_u32_decimal(buf: &mut [u8], mut pos: usize, val: u32) -> usize {
    if val == 0 {
        if pos < buf.len() { buf[pos] = b'0'; pos += 1; }
        return pos;
    }
    let mut digits = [0u8; 10];
    let mut n = val;
    let mut i = 0;
    while n > 0 {
        digits[i] = (n % 10) as u8;
        n /= 10;
        i += 1;
    }
    for j in (0..i).rev() {
        if pos < buf.len() {
            buf[pos] = b'0' + digits[j];
            pos += 1;
        }
    }
    pos
}

/// Write a usize in decimal to the buffer, returning the new position.
fn write_usize_decimal(buf: &mut [u8], pos: usize, val: usize) -> usize {
    write_u32_decimal(buf, pos, val as u32)
}

/// Write a u32 in hexadecimal (8 digits, zero-padded) to the buffer.
fn write_u32_hex(buf: &mut [u8], mut pos: usize, val: u32) -> usize {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    for shift in (0..8).rev() {
        let nibble = ((val >> (shift * 4)) & 0xF) as usize;
        if pos < buf.len() {
            buf[pos] = HEX[nibble];
            pos += 1;
        }
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jhal_creation() {
        let jhal = Jhal::new();
        assert!(!jhal.is_apic_initialized());
        assert!(jhal.device_registry.is_empty());
    }

    #[test]
    fn test_re_exports() {
        // Verify all re-exports are accessible
        let _port: u16 = COM1;
        let _baud: BaudRate = BaudRate::Baud115200;
        let _divide: TimerDivide = TimerDivide::Div16;
        let _max_bus: usize = MAX_BUS;
        let _max_dev: usize = MAX_DEVICE;
        let _max_fn: usize = MAX_FUNCTION;
    }

    #[test]
    fn test_zero_heap_formatting_decimal() {
        let mut buf = [0u8; 64];
        let pos = write_u32_decimal(&mut buf, 0, 42);
        assert_eq!(core::str::from_utf8(&buf[..pos]).unwrap(), "42");

        let mut buf2 = [0u8; 64];
        let pos2 = write_u32_decimal(&mut buf2, 0, 0);
        assert_eq!(core::str::from_utf8(&buf2[..pos2]).unwrap(), "0");

        let mut buf3 = [0u8; 64];
        let pos3 = write_u32_decimal(&mut buf3, 0, 123456);
        assert_eq!(core::str::from_utf8(&buf3[..pos3]).unwrap(), "123456");
    }

    #[test]
    fn test_zero_heap_formatting_hex() {
        let mut buf = [0u8; 64];
        let pos = write_u32_hex(&mut buf, 0, 0xDEAD_BEEF);
        assert_eq!(core::str::from_utf8(&buf[..pos]).unwrap(), "DEADBEEF");

        let mut buf2 = [0u8; 64];
        let pos2 = write_u32_hex(&mut buf2, 0, 0x0000_00FF);
        assert_eq!(core::str::from_utf8(&buf2[..pos2]).unwrap(), "000000FF");
    }

    #[test]
    fn test_zero_heap_formatting_apic_info() {
        let mut buf = [0u8; 64];
        let msg = jhal_format_apic_info(3, 0x0005_0011, &mut buf);
        assert!(msg.contains("3"));
        assert!(msg.contains("00050011"));
    }

    #[test]
    fn test_zero_heap_formatting_enum_result() {
        let mut buf = [0u8; 64];
        let msg = jhal_format_enum_result(7, &mut buf);
        assert!(msg.contains("7"));
        assert!(msg.contains("devices"));
    }
}
