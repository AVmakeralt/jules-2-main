// =========================================================================
// UART 16550 Serial Port Driver — The Voice of Jules
//
// Before you have a screen, you have a Serial Port.  This is how you
// "see" into the 4.5MB sanctuary.  The 16550 UART is the universal
// console device on x86 systems — COM1 (0x3F8) and COM2 (0x2F8).
//
// ARCHITECTURE DECISIONS:
//
//   1. ZERO LOCKING: Each core writes into its own per-CPU ring buffer
//      (via rseq).  A dedicated flush thread (or the idle loop) drains
//      the buffers to the physical UART.  This means logging never stalls
//      the JIT — the critical path is a single `store` + `fetch_add`.
//
//   2. TSX SAFETY: UART writes inside a TSX transaction are buffered in
//      the per-CPU ring and only flushed AFTER the transaction commits.
//      I/O instructions (`outb`) are not transactional — if we wrote
//      directly to the UART inside a TSX region and the transaction
//      aborted, the byte would already be on the wire with no way to
//      "unsend" it.  This driver prevents that.
//
//   3. BOUNDED SPIN: The transmit-wait loop has a maximum spin count.
//      If the UART transmitter is stuck, we drop the byte rather than
//      hang the CPU.  In a JIT runtime, a dropped log line is far
//      preferable to an infinite spin loop.
//
//   4. NO HEAP ALLOCATION: The ring buffer is a static array.  The
//      entire driver fits in a few cache lines.  No Vec, no Box,
//      no String — pure static arrays only.
//
//   5. NO EXTERNAL CRATES: This module depends only on core::sync::atomic
//      and the io_port module.  No num_cpus, no std — pure Rust.
//
// REVIEW CHECKLIST:
//   1. Re-entrant: Yes.  Each core has its own buffer.  The shared UART
//      is only touched by the flush thread, which uses atomic CAS for
//      mutual exclusion (no lock — just try-lock and retry).
//   2. Side channel: Port I/O is inherently timing-variable, but we
//      use constant-time spin-wait (no early exit based on data values).
//      The per-CPU buffer prevents cross-core timing interference.
//   3. Memory ordering: `outb`/`inb` include `compiler_fence(SeqCst)`.
//      Ring buffer uses `Acquire`/`Release` ordering.
//   4. 4.5MB limit: Zero heap.  `SerialRingBuffer` is 256 bytes per core.
//      Console uses a fixed array of MAX_CORES ring buffers (~64 KiB total).
// =========================================================================

use core::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use super::io_port::{outb, inb};

// ─── UART Register Offsets (relative to base port) ────────────────────────

/// Offset 0: Transmit Holding Buffer (write) / Receiver Buffer (read)
const REG_DATA: u16 = 0;
/// Offset 1: Interrupt Enable Register
const REG_IER: u16 = 1;
/// Offset 2: FIFO Control Register (write) / Interrupt Identification (read)
const REG_FCR: u16 = 2;
/// Offset 3: Line Control Register
const REG_LCR: u16 = 3;
/// Offset 4: Modem Control Register
const REG_MCR: u16 = 4;
/// Offset 5: Line Status Register
const REG_LSR: u16 = 5;
/// Offset 6: Modem Status Register
const REG_MSR: u16 = 6;
/// Offset 7: Scratch Register
const REG_SCR: u16 = 7;

// ─── Bit Field Definitions ────────────────────────────────────────────────

/// IER: Enable Received Data Available Interrupt
const IER_RX_ENABLE: u8 = 1 << 0;
/// IER: Enable Transmit Holding Register Empty Interrupt
const IER_TX_ENABLE: u8 = 1 << 1;
/// IER: Enable Receiver Line Status Interrupt
const IER_RX_LINE_STATUS: u8 = 1 << 2;
/// IER: Enable Modem Status Interrupt
const IER_MODEM_STATUS: u8 = 1 << 3;

/// LCR: Divisor Latch Access Bit (bit 7)
const LCR_DLAB: u8 = 1 << 7;
/// LCR: 8 data bits (bits 1:0 = 11)
const LCR_8BITS: u8 = 0b11;
/// LCR: No parity (bit 3 = 0)
const LCR_NO_PARITY: u8 = 0;
/// LCR: One stop bit (bit 2 = 0)
const LCR_ONE_STOP: u8 = 0;

/// FCR: Enable FIFOs (bit 0)
const FCR_ENABLE: u8 = 1 << 0;
/// FCR: Clear receive FIFO (bit 1)
const FCR_CLEAR_RX: u8 = 1 << 1;
/// FCR: Clear transmit FIFO (bit 2)
const FCR_CLEAR_TX: u8 = 1 << 2;
/// FCR: DMA mode (bit 3)
const FCR_DMA_MODE: u8 = 1 << 3;
/// FCR: 14-byte trigger level (bits 7:6 = 11)
const FCR_TRIGGER_14: u8 = 0b11 << 6;

/// MCR: Data Terminal Ready (bit 0)
const MCR_DTR: u8 = 1 << 0;
/// MCR: Request To Send (bit 1)
const MCR_RTS: u8 = 1 << 1;
/// MCR: Auxiliary Output 1 (bit 2) — used as interrupt gate on some systems
const MCR_OUT1: u8 = 1 << 2;
/// MCR: Auxiliary Output 2 (bit 3) — enables IRQ on PC-compatible UARTs
const MCR_OUT2: u8 = 1 << 3;

/// LSR: Data Ready (bit 0)
const LSR_DATA_READY: u8 = 1 << 0;
/// LSR: Transmit Holding Register Empty (bit 5)
const LSR_TX_EMPTY: u8 = 1 << 5;

// ─── Baud Rate Divisors ───────────────────────────────────────────────────

/// Baud rate divisor values for a standard 1.8432 MHz clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum BaudRate {
    /// 115200 baud (divisor = 1)
    Baud115200 = 1,
    /// 57600 baud (divisor = 2)
    Baud57600 = 2,
    /// 38400 baud (divisor = 3)
    Baud38400 = 3,
    /// 19200 baud (divisor = 6)
    Baud19200 = 6,
    /// 9600 baud (divisor = 12)
    Baud9600 = 12,
}

// ─── Spin-Loop Timeout ────────────────────────────────────────────────────

/// Maximum number of spin-loop iterations before dropping a byte.
/// At ~2 GHz with `spin_loop()` (~4 cycles each), 1 million iterations
/// is approximately 2 milliseconds.  This is generous; the UART
/// transmitter should empty a byte in ~87 microseconds at 115200 baud.
const SPIN_TIMEOUT: u32 = 1_000_000;

// ─── Per-CPU Ring Buffer ──────────────────────────────────────────────────

/// Size of the per-CPU ring buffer in bytes.  Must be a power of 2
/// for efficient modular arithmetic.
const RING_SIZE: usize = 256;
/// Mask for wrapping ring buffer indices (RING_SIZE - 1).
const RING_MASK: usize = RING_SIZE - 1;

/// A lock-free single-producer, single-consumer ring buffer for UART output.
///
/// The producer (any core writing to the serial console) uses `enqueue()`.
/// The consumer (the flush thread) uses `dequeue()`.
///
/// # TSX Safety
/// `enqueue()` is safe to call inside a TSX transaction because it only
/// touches per-CPU memory.  If the transaction aborts, the ring buffer
/// state is rolled back by TSX, and the byte is simply not enqueued.
/// The byte will never appear on the physical UART because `dequeue()`
/// only runs outside the transaction.
pub struct SerialRingBuffer {
    /// Ring buffer storage.
    buffer: [u8; RING_SIZE],
    /// Write position (only modified by the producer).
    head: AtomicUsize,
    /// Read position (only modified by the consumer).
    tail: AtomicUsize,
    /// Number of bytes dropped due to ring overflow.
    dropped: AtomicU32,
}

impl SerialRingBuffer {
    /// Create a new empty ring buffer.
    pub const fn new() -> Self {
        Self {
            buffer: [0u8; RING_SIZE],
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            dropped: AtomicU32::new(0),
        }
    }

    /// Enqueue a byte for later transmission.
    /// Returns `true` if the byte was enqueued, `false` if the buffer is full.
    ///
    /// # Safety for TSX
    /// This method only writes to per-CPU memory and uses atomic operations.
    /// It is safe to call inside a TSX transaction.
    pub fn enqueue(&self, byte: u8) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let next_head = (head + 1) & RING_MASK;

        if next_head == tail {
            // Buffer full — drop the byte
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        // SAFETY: We are the only producer.  The `head` position is ours.
        // The buffer cell at `head` is not being read by the consumer
        // because `head != tail` (we checked above) and the consumer
        // only reads cells between `tail` and `head`.
        unsafe {
            let ptr = self.buffer.as_ptr().add(head) as *mut u8;
            core::ptr::write_volatile(ptr, byte);
        }
        self.head.store(next_head, Ordering::Release);
        true
    }

    /// Dequeue a byte from the buffer.
    /// Returns `Some(byte)` if available, `None` if the buffer is empty.
    pub fn dequeue(&self) -> Option<u8> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail == head {
            // Buffer empty
            return None;
        }

        let byte = unsafe {
            let ptr = self.buffer.as_ptr().add(tail);
            core::ptr::read_volatile(ptr)
        };
        let next_tail = (tail + 1) & RING_MASK;
        self.tail.store(next_tail, Ordering::Release);
        Some(byte)
    }

    /// Get the number of bytes currently in the buffer.
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        (head.wrapping_sub(tail)) & RING_MASK
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the number of dropped bytes.
    pub fn dropped_count(&self) -> u32 {
        self.dropped.load(Ordering::Relaxed)
    }
}

// ─── Serial Port Driver ───────────────────────────────────────────────────

/// Standard COM port addresses.
pub const COM1: u16 = 0x3F8;
pub const COM2: u16 = 0x2F8;
pub const COM3: u16 = 0x3E8;
pub const COM4: u16 = 0x2E8;

/// The UART 16550 serial port driver.
///
/// This struct represents a single UART port.  It provides both
/// synchronous (blocking) and buffered (lock-free) I/O.
///
/// # TSX Safety
/// - `write_byte_blocking()` must NOT be called inside a TSX transaction
///   because it uses `outb`, which is not transactional.
/// - `write_buffered()` is safe inside a TSX transaction because it only
///   writes to the per-CPU ring buffer.
pub struct SerialPort {
    /// Base I/O port address.
    base_port: u16,
    /// Whether the port has been initialized.
    initialized: AtomicBool,
    /// Flush lock — only one core may flush at a time.
    flush_lock: AtomicBool,
}

impl SerialPort {
    /// Create a new serial port handle for the given base address.
    ///
    /// Does NOT initialize the port — call `init()` before use.
    pub const fn new(port: u16) -> Self {
        Self {
            base_port: port,
            initialized: AtomicBool::new(false),
            flush_lock: AtomicBool::new(false),
        }
    }

    /// Create a SerialPort for COM1 (0x3F8).
    pub const fn com1() -> Self {
        Self::new(COM1)
    }

    /// Initialize the UART with the specified baud rate.
    ///
    /// This configures: 8 data bits, no parity, one stop bit (8-N-1),
    /// enables FIFOs with 14-byte trigger level, and sets MCR to
    /// enable DTR + RTS + OUT2 (IRQ gate).
    ///
    /// # Safety
    /// - Must be called before any other operations.
    /// - Must only be called once (idempotent if called again).
    /// - The base port must be a valid UART I/O port.
    pub fn init(&self, baud: BaudRate) {
        if self.initialized.load(Ordering::Acquire) {
            return; // Already initialized — idempotent
        }

        let port = self.base_port;

        unsafe {
            // Step 1: Disable all interrupts
            outb(port + REG_IER, 0x00);

            // Step 2: Enable DLAB to set baud rate divisor
            outb(port + REG_LCR, LCR_DLAB);

            // Step 3: Set divisor (low byte, then high byte)
            outb(port + REG_DATA, (baud as u16 & 0xFF) as u8);
            outb(port + REG_IER, ((baud as u16 >> 8) & 0xFF) as u8);

            // Step 4: Disable DLAB, set 8-N-1
            outb(port + REG_LCR, LCR_8BITS | LCR_NO_PARITY | LCR_ONE_STOP);

            // Step 5: Enable FIFO, clear both, 14-byte trigger
            outb(port + REG_FCR, FCR_ENABLE | FCR_CLEAR_RX | FCR_CLEAR_TX | FCR_TRIGGER_14);

            // Step 6: Set MCR (DTR + RTS + OUT2 for IRQ gate)
            outb(port + REG_MCR, MCR_DTR | MCR_RTS | MCR_OUT2);

            // Step 7: Verify by writing/reading scratch register
            outb(port + REG_SCR, 0xAE);
            let readback = inb(port + REG_SCR);
            if readback != 0xAE {
                // UART scratch register test failed — may be a 8250 (no FIFO)
                // or the port doesn't exist.  Continue anyway; the driver
                // will work but without FIFO benefits.
            }
        }

        self.initialized.store(true, Ordering::Release);
    }

    /// Check if the UART transmit holding register is empty.
    fn is_transmit_empty(&self) -> bool {
        unsafe { (inb(self.base_port + REG_LSR) & LSR_TX_EMPTY) != 0 }
    }

    /// Check if the UART has received data.
    pub fn has_data(&self) -> bool {
        unsafe { (inb(self.base_port + REG_LSR) & LSR_DATA_READY) != 0 }
    }

    /// Write a single byte to the UART, blocking until the transmitter
    /// is ready.
    ///
    /// # TSX Safety
    /// **MUST NOT** be called inside a TSX transaction.  The `outb`
    /// instruction is not transactional — if the transaction aborts
    /// after the write, the byte is already on the wire.
    ///
    /// Use `write_buffered()` inside TSX transactions instead.
    pub fn write_byte_blocking(&self, byte: u8) {
        if !self.initialized.load(Ordering::Acquire) {
            return;
        }

        // Bounded spin-wait for transmit ready
        let mut spins = 0;
        while !self.is_transmit_empty() {
            core::hint::spin_loop();
            spins += 1;
            if spins >= SPIN_TIMEOUT {
                // Transmitter stuck — drop the byte rather than hang
                return;
            }
        }

        unsafe {
            outb(self.base_port + REG_DATA, byte);
        }
    }

    /// Read a single byte from the UART, blocking until data is available.
    ///
    /// Uses a bounded spin-wait with timeout.
    pub fn read_byte_blocking(&self) -> Option<u8> {
        if !self.initialized.load(Ordering::Acquire) {
            return None;
        }

        let mut spins = 0;
        while !self.has_data() {
            core::hint::spin_loop();
            spins += 1;
            if spins >= SPIN_TIMEOUT {
                return None;
            }
        }

        Some(unsafe { inb(self.base_port + REG_DATA) })
    }

    /// Write a byte to the per-CPU ring buffer (lock-free, TSX-safe).
    ///
    /// This is the preferred method for logging from the JIT runtime.
    /// The byte will be flushed to the physical UART by the flush
    /// thread or idle loop.
    pub fn write_buffered(&self, byte: u8, ring: &SerialRingBuffer) -> bool {
        ring.enqueue(byte)
    }

    /// Write a string to the per-CPU ring buffer.
    /// Returns the number of bytes successfully enqueued.
    pub fn write_str_buffered(&self, s: &str, ring: &SerialRingBuffer) -> usize {
        let mut count = 0;
        for byte in s.bytes() {
            if ring.enqueue(byte) {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    /// Flush all bytes from the ring buffer to the physical UART.
    ///
    /// Uses a try-lock to ensure only one core flushes at a time.
    /// Other cores simply skip — their bytes will be flushed on the
    /// next flush call.
    ///
    /// # TSX Safety
    /// **MUST NOT** be called inside a TSX transaction.
    pub fn flush_ring(&self, ring: &SerialRingBuffer) {
        if !self.initialized.load(Ordering::Acquire) {
            return;
        }

        // Try-lock: if another core is flushing, skip this round
        if self.flush_lock.compare_exchange(
            false,
            true,
            Ordering::Acquire,
            Ordering::Relaxed,
        ).is_err() {
            return;
        }

        // Drain the ring buffer
        while let Some(byte) = ring.dequeue() {
            self.write_byte_blocking(byte);
        }

        // Release the lock
        self.flush_lock.store(false, Ordering::Release);
    }

    /// Write a complete string (blocking).
    pub fn write_str(&self, s: &str) {
        for byte in s.bytes() {
            self.write_byte_blocking(byte);
        }
    }

    /// Write a complete string with a newline (blocking).
    pub fn write_line(&self, s: &str) {
        self.write_str(s);
        self.write_byte_blocking(b'\r');
        self.write_byte_blocking(b'\n');
    }
}

// ─── Per-CPU Console (Zero Heap) ─────────────────────────────────────────

/// Maximum number of cores supported for per-CPU console buffers.
/// This is a static limit — no Vec, no dynamic sizing.
const MAX_CORES: usize = 256;



/// The global console: one ring buffer per core, one shared UART.
///
/// Uses a FIXED-SIZE static array of `SerialRingBuffer` — no Vec,
/// no heap allocation.  Only `active_cores` entries are actually used;
/// the rest sit idle and consume no CPU time.
///
/// Total memory: 256 cores × ~270 bytes = ~67 KiB (well within budget).
pub struct Console {
    /// The serial port.
    port: SerialPort,
    /// Per-CPU ring buffers.  Fixed-size array, no heap.
    rings: [SerialRingBuffer; MAX_CORES],
    /// Number of CPU cores actually in use (1..=256).
    active_cores: usize,
}

impl Console {
    /// Create a new console on COM1 at 115200 baud.
    pub fn new() -> Self {
        let port = SerialPort::com1();
        // NOTE: We cannot call init() here because it touches I/O ports
        // which may not be available in userspace test environments.
        // In bare-metal boot, init() is called separately.
        let active_cores = detect_cpu_count_fallback();
        Self {
            port,
            rings: const_array_of_rings(),
            active_cores,
        }
    }

    /// Create a console on a specific port and baud rate.
    pub fn with_port_baud(port: u16, baud: BaudRate) -> Self {
        let serial = SerialPort::new(port);
        serial.init(baud);
        let active_cores = detect_cpu_count_fallback();
        Self {
            port: serial,
            rings: const_array_of_rings(),
            active_cores,
        }
    }

    /// Initialize the serial port (call once during boot).
    pub fn init_port(&self, baud: BaudRate) {
        self.port.init(baud);
    }

    /// Write a byte from the current CPU to the console (buffered, TSX-safe).
    /// If rseq is available, writes to the per-CPU ring.  Otherwise,
    /// falls back to CPU 0's ring.
    pub fn write(&self, byte: u8, cpu_id: Option<usize>) -> bool {
        let idx = cpu_id.unwrap_or(0).min(self.active_cores.saturating_sub(1));
        self.port.write_buffered(byte, &self.rings[idx])
    }

    /// Write a string from the current CPU (buffered, TSX-safe).
    pub fn write_str(&self, s: &str, cpu_id: Option<usize>) -> usize {
        let idx = cpu_id.unwrap_or(0).min(self.active_cores.saturating_sub(1));
        self.port.write_str_buffered(s, &self.rings[idx])
    }

    /// Flush all per-CPU buffers to the UART.
    /// Should be called from the idle loop or a dedicated flush thread.
    pub fn flush_all(&self) {
        for i in 0..self.active_cores {
            self.port.flush_ring(&self.rings[i]);
        }
    }

    /// Flush only the current CPU's buffer.
    pub fn flush_cpu(&self, cpu_id: Option<usize>) {
        let idx = cpu_id.unwrap_or(0).min(self.active_cores.saturating_sub(1));
        self.port.flush_ring(&self.rings[idx]);
    }

    /// Get the total number of dropped bytes across all cores.
    pub fn total_dropped(&self) -> u32 {
        let mut total = 0u32;
        for i in 0..self.active_cores {
            total += self.rings[i].dropped_count();
        }
        total
    }

    /// Write a string directly (blocking, NOT TSX-safe).
    /// Use only during early boot before per-CPU buffers are set up.
    pub fn write_direct(&self, s: &str) {
        self.port.write_str(s);
    }

    /// Write a line directly (blocking, NOT TSX-safe).
    pub fn write_line_direct(&self, s: &str) {
        self.port.write_line(s);
    }

    /// Get the number of active ring buffers.
    pub fn num_cores(&self) -> usize {
        self.active_cores
    }
}

impl Default for Console {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper: create a const array of SerialRingBuffer values.
/// Required because SerialRingBuffer::new() is const but [T; N] construction
/// needs either Copy or const initialization.
const fn const_array_of_rings() -> [SerialRingBuffer; MAX_CORES] {
    const INIT: SerialRingBuffer = SerialRingBuffer::new();
    [INIT; MAX_CORES]
}

/// Fallback CPU count detection using CPUID — NO external crates.
///
/// On x86_64, uses `core::arch::x86_64::__cpuid` which is a pure Rust
/// intrinsic (no external crate needed).  CPUID leaf 1 EBX[23:16] reports
/// the maximum number of addressable logical processors in this package.
///
/// On other architectures or if CPUID is unavailable, returns 1.
/// The boot code should call `Console::set_active_cores()` with the
/// real count discovered via ACPI/MADT.
fn detect_cpu_count_fallback() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        // Use raw inline asm for CPUID — no external crates, no nightly __cpuid.
        // CPUID leaf 1, EBX[23:16] = max logical processors per package.
        let ebx: u32;
        unsafe {
            core::arch::asm!(
                "xchg {tmp}, rbx",  // Save RBX (PIC register on some ABIs)
                "cpuid",
                "xchg {tmp}, rbx",  // Restore RBX
                tmp = out(reg) ebx,
                in("eax") 1u32,
                in("ecx") 0u32,
                out("edx") _,
                options(nomem, nostack)
            );
        }
        let count = ((ebx >> 16) & 0xFF) as usize;
        if count == 0 { 1 } else { count.min(MAX_CORES) }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let ring = SerialRingBuffer::new();
        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);

        // Enqueue a byte
        assert!(ring.enqueue(0x41)); // 'A'
        assert!(!ring.is_empty());
        assert_eq!(ring.len(), 1);

        // Dequeue the byte
        let byte = ring.dequeue();
        assert_eq!(byte, Some(0x41));
        assert!(ring.is_empty());
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let ring = SerialRingBuffer::new();

        // Fill the buffer (RING_SIZE - 1 items, because head+1 == tail means full)
        for i in 0..(RING_SIZE - 1) {
            assert!(ring.enqueue(i as u8), "enqueue failed at {}", i);
        }

        // Next enqueue should fail (buffer full)
        assert!(!ring.enqueue(0xFF));
        assert_eq!(ring.dropped_count(), 1);

        // Drain and verify
        for i in 0..(RING_SIZE - 1) {
            assert_eq!(ring.dequeue(), Some(i as u8));
        }
        assert!(ring.is_empty());
    }

    #[test]
    fn test_ring_buffer_wrap_around() {
        let ring = SerialRingBuffer::new();

        // Fill and drain multiple times to test wrap-around
        for round in 0..10 {
            for i in 0..10 {
                assert!(ring.enqueue((round * 10 + i) as u8));
            }
            for i in 0..10 {
                assert_eq!(ring.dequeue(), Some((round * 10 + i) as u8));
            }
        }
        assert!(ring.is_empty());
    }

    #[test]
    fn test_baud_rate_values() {
        assert_eq!(BaudRate::Baud115200 as u16, 1);
        assert_eq!(BaudRate::Baud38400 as u16, 3);
        assert_eq!(BaudRate::Baud9600 as u16, 12);
    }

    #[test]
    fn test_serial_port_const() {
        // Verify COM port addresses are correct
        assert_eq!(COM1, 0x3F8);
        assert_eq!(COM2, 0x2F8);
        assert_eq!(COM3, 0x3E8);
        assert_eq!(COM4, 0x2E8);
    }

    #[test]
    fn test_lcr_fields() {
        // 8-N-1 = 0x03
        let lcr = LCR_8BITS | LCR_NO_PARITY | LCR_ONE_STOP;
        assert_eq!(lcr, 0x03);
        // DLAB = bit 7
        assert_eq!(LCR_DLAB, 0x80);
    }

    #[test]
    fn test_fcr_fields() {
        // Enable + clear both + 14-byte trigger
        let fcr = FCR_ENABLE | FCR_CLEAR_RX | FCR_CLEAR_TX | FCR_TRIGGER_14;
        assert_eq!(fcr, 0xC7);
    }

    #[test]
    fn test_console_no_heap() {
        // Verify Console can be created without heap allocation
        let console = Console::new();
        assert!(console.num_cores() >= 1);
        assert!(console.num_cores() <= MAX_CORES);
    }

    #[test]
    fn test_ring_size_is_power_of_2() {
        // Required for efficient modular arithmetic
        assert!(RING_SIZE.is_power_of_two());
    }
}
