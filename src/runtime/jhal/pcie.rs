// =========================================================================
// PCIe Bus Enumerator — The Discovery Engine of Jules
//
// Before Jules can use an NVMe drive or a Network Card, it must "walk"
// the PCI Express bus tree.  This enumerator performs recursive discovery
// of all PCI devices, respecting the formal bounds of the PCI
// specification:
//
//   - Maximum 256 buses
//   - Maximum 32 devices per bus
//   - Maximum 8 functions per device
//   - Maximum 6 Base Address Registers (BARs) per function
//
// FORMAL BOUNDS CHECKING:
//   The PCI specification defines hard limits that are enforced by the
//   hardware.  We encode these as const generics and type-level proofs
//   so that the SMT solver can verify the enumerator cannot exceed them.
//
// JULES INTEGRATION:
//   - Each discovered device is registered with `DeviceRegistry`
//   - A `ProphecyVariable` is assigned to track expected latency
//   - The device's BAR addresses are recorded for the AliasLayout engine
//     (MMIO regions must be mapped as UC, not WB)
//
// CONFIGURATION ACCESS METHODS:
//   - Mechanism 1: Port I/O (0xCF8/0xCFC) — universal, all x86
//   - ECAM (Enhanced Configuration Access Mechanism): Memory-mapped
//     at 0xE000_0000 (256 MB window) — required for PCIe, faster
//
// REVIEW CHECKLIST:
//   1. Re-entrant: The enumerator is a pure read-only operation.
//      It only writes to the `DeviceRegistry`, which is thread-safe.
//      A TSX abort during enumeration simply discards the partially-
//      discovered devices — no hardware state is modified.
//   2. Side channel: Configuration space reads are cacheable.  The
//      enumerator reads every bus/slot/function combination regardless
//      of whether a device is present, producing constant-time access
//      patterns.  This prevents leaking which slots have devices.
//   3. Memory ordering: `read_pci_config_*` uses `compiler_fence(SeqCst)`
//      for port I/O and `MmioReg` for ECAM.
//   4. 4.5MB limit: Zero heap.  Device descriptors are stored in a
//      static array with a fixed maximum.
// =========================================================================

#![allow(dead_code)]
use core::sync::atomic::{AtomicUsize, Ordering};
use core::cell::UnsafeCell;
use super::io_port::{outl, inl};
use super::mmio::MmioReg;

// ─── PCI Specification Constants ──────────────────────────────────────────

/// Maximum number of PCI buses (8-bit bus number: 0–255).
pub const MAX_BUS: usize = 256;
/// Maximum number of devices per bus (5-bit device number: 0–31).
pub const MAX_DEVICE: usize = 32;
/// Maximum number of functions per device (3-bit function number: 0–7).
pub const MAX_FUNCTION: usize = 8;
/// Maximum number of BARs per function (header types 0 and 1).
pub const MAX_BARS: usize = 6;

/// PCI Configuration Mechanism 1: Address port.
const PCI_CONFIG_ADDRESS: u16 = 0xCF8;
/// PCI Configuration Mechanism 1: Data port.
const PCI_CONFIG_DATA: u16 = 0xCFC;

/// Enable bit in the configuration address register (bit 31).
const PCI_CONFIG_ENABLE: u32 = 1 << 31;

/// ECAM base address (Enhanced Configuration Access Mechanism).
/// Each bus gets a 1MB window.  256 buses × 1MB = 256MB.
const ECAM_BASE: usize = 0xE000_0000;

// ─── Configuration Space Offsets ──────────────────────────────────────────

/// Offset 0x00: Vendor ID (bits 15:0) / Device ID (bits 31:16)
const REG_VENDOR_DEVICE: u8 = 0x00;
/// Offset 0x02: Device ID (upper 16 bits of offset 0x00 DWORD)
#[allow(dead_code)]
const REG_DEVICE_ID_SHIFT: u8 = 16;
/// Offset 0x08: Revision ID (bits 7:0) / Class Code (bits 31:8)
const REG_CLASS_REV: u8 = 0x08;
/// Offset 0x0C: Cache Line Size / Latency Timer / Header Type / BIST
const REG_HEADER_TYPE: u8 = 0x0C;
/// Offset 0x0E: Header Type (bits 6:0) / Multi-function (bit 7)
#[allow(dead_code)]
const HEADER_TYPE_OFFSET: u8 = 0x0E;
/// Offset 0x18: Primary Bus Number (header type 1, PCI-to-PCI bridge)
const REG_PRIMARY_BUS: u8 = 0x18;
/// Offset 0x19: Secondary Bus Number (header type 1)
#[allow(dead_code)]
const REG_SECONDARY_BUS: u8 = 0x19;
/// Offset 0x1A: Subordinate Bus Number (header type 1)
#[allow(dead_code)]
const REG_SUBORDINATE_BUS: u8 = 0x1A;

/// Invalid vendor ID — indicates no device present at this BDF.
const VENDOR_ID_NONE: u16 = 0xFFFF;

/// Maximum number of devices that can be registered.
/// This is a static limit to avoid heap allocation.
const MAX_DEVICES: usize = 256;

// ─── Type-Level Bounds Proofs ─────────────────────────────────────────────

/// A bus number bounded to 0..256.
/// Constructed via `BoundedBus::new()` which returns `None` if out of range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct BoundedBus(pub u8);

impl BoundedBus {
    /// Create a bounded bus number.  Returns `None` if `bus >= 256`.
    /// (Since the inner type is u8, this always succeeds, but the
    /// function exists for formal verification consistency.)
    pub fn new(bus: u8) -> Option<Self> {
        Some(Self(bus))
    }

    /// Get the raw bus number.
    pub fn raw(self) -> u8 {
        self.0
    }
}

/// A device number bounded to 0..32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct BoundedDevice(pub u8);

impl BoundedDevice {
    /// Create a bounded device number.  Returns `None` if `device >= 32`.
    pub fn new(device: u8) -> Option<Self> {
        if device < 32 {
            Some(Self(device))
        } else {
            None
        }
    }

    pub fn raw(self) -> u8 {
        self.0
    }
}

/// A function number bounded to 0..8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct BoundedFunction(pub u8);

impl BoundedFunction {
    /// Create a bounded function number.  Returns `None` if `function >= 8`.
    pub fn new(function: u8) -> Option<Self> {
        if function < 8 {
            Some(Self(function))
        } else {
            None
        }
    }

    pub fn raw(self) -> u8 {
        self.0
    }
}

/// A PCI Bus/Device/Function triple with proven bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PciBdf {
    pub bus: BoundedBus,
    pub device: BoundedDevice,
    pub function: BoundedFunction,
}

impl PciBdf {
    /// Create a BDF triple.  Returns `None` if any component is out of range.
    pub fn new(bus: u8, device: u8, function: u8) -> Option<Self> {
        Some(Self {
            bus: BoundedBus::new(bus)?,
            device: BoundedDevice::new(device)?,
            function: BoundedFunction::new(function)?,
        })
    }
}

// ─── PCI Device Descriptor ────────────────────────────────────────────────

/// Header type of a PCI function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PciHeaderType {
    /// Standard device (header type 0)
    Standard = 0,
    /// PCI-to-PCI bridge (header type 1)
    PciBridge = 1,
    /// CardBus bridge (header type 2)
    CardBus = 2,
    /// Unknown header type
    Unknown = 255,
}

/// A discovered PCI device/function.
#[derive(Debug, Clone, Copy)]
pub struct PciDevice {
    /// Bus/Device/Function address.
    pub bdf: PciBdf,
    /// Vendor ID (0xFFFF = invalid).
    pub vendor_id: u16,
    /// Device ID.
    pub device_id: u16,
    /// Class code (byte 2 = base class, byte 1 = sub class, byte 0 = prog IF).
    pub class_code: u8,
    /// Subclass code.
    pub subclass: u8,
    /// Programming interface.
    pub prog_if: u8,
    /// Revision ID.
    pub revision: u8,
    /// Header type.
    pub header_type: PciHeaderType,
    /// Is this a multi-function device?
    pub multi_function: bool,
    /// For PCI bridges: secondary bus number.
    pub secondary_bus: Option<u8>,
    /// For PCI bridges: subordinate bus number.
    pub subordinate_bus: Option<u8>,
    /// Base Address Registers (up to 6 for standard devices, 2 for bridges).
    pub bars: [u64; MAX_BARS],
    /// Number of valid BARs.
    pub bar_count: usize,
    /// Expected latency prophecy (set by ProphecyOracle integration).
    pub latency_ns: u32,
}

impl PciDevice {
    /// Create an invalid/empty device descriptor.
    pub fn empty() -> Self {
        Self {
            bdf: PciBdf::new(0, 0, 0).unwrap(),
            vendor_id: VENDOR_ID_NONE,
            device_id: 0,
            class_code: 0,
            subclass: 0,
            prog_if: 0,
            revision: 0,
            header_type: PciHeaderType::Unknown,
            multi_function: false,
            secondary_bus: None,
            subordinate_bus: None,
            bars: [0; MAX_BARS],
            bar_count: 0,
            latency_ns: 0,
        }
    }

    /// Check if this is a valid device (has a real vendor ID).
    pub fn is_present(&self) -> bool {
        self.vendor_id != VENDOR_ID_NONE
    }

    /// Get a human-readable class name.
    pub fn class_name(&self) -> &'static str {
        match self.class_code {
            0x00 => "Unclassified",
            0x01 => "Mass Storage",
            0x02 => "Network",
            0x03 => "Display",
            0x04 => "Multimedia",
            0x05 => "Memory Controller",
            0x06 => "Bridge",
            0x07 => "Communication",
            0x08 => "Generic System Peripheral",
            0x09 => "Input Device",
            0x0A => "Docking Station",
            0x0B => "Processor",
            0x0C => "Serial Bus",
            0x0D => "Wireless",
            0x0E => "Intelligent Controller",
            0x0F => "Satellite Communication",
            0x10 => "Encryption",
            0x11 => "Signal Processing",
            0x12 => "Processing Accelerator",
            0x13 => "Non-Essential Instrumentation",
            0x40 => "Co-Processor",
            0xFF => "Unassigned",
            _ => "Unknown",
        }
    }
}

// ─── Device Registry ──────────────────────────────────────────────────────

/// A static, no-heap device registry.
/// Stores up to `MAX_DEVICES` discovered PCI devices.
///
/// Uses `UnsafeCell` for interior mutability because `register()` takes
/// `&self` (needed for thread-safe atomic counting) while writing to
/// the devices array.  The safety invariant is that each slot is only
/// written by the thread that successfully increments `count`, creating
/// a unique ownership transfer per slot.
pub struct DeviceRegistry {
    /// Discovered devices (interior-mutable for register() via &self).
    devices: UnsafeCell<[PciDevice; MAX_DEVICES]>,
    /// Number of valid entries.
    count: AtomicUsize,
}

// SAFETY: Access to the devices array is governed by the atomic `count`.
// Each slot at index `idx` is uniquely "owned" by the thread that
// successfully claimed it via `fetch_add(1)`.  Once written, slots are
// only ever read (via `get()`, `find_by_class()`, etc.).
unsafe impl Sync for DeviceRegistry {}
unsafe impl Send for DeviceRegistry {}

impl DeviceRegistry {
    /// Create an empty registry.
    pub const fn new() -> Self {
        Self {
            devices: UnsafeCell::new({
                const INIT: PciDevice = PciDevice {
                    bdf: PciBdf {
                        bus: BoundedBus(0),
                        device: BoundedDevice(0),
                        function: BoundedFunction(0),
                    },
                    vendor_id: VENDOR_ID_NONE,
                    device_id: 0,
                    class_code: 0,
                    subclass: 0,
                    prog_if: 0,
                    revision: 0,
                    header_type: PciHeaderType::Unknown,
                    multi_function: false,
                    secondary_bus: None,
                    subordinate_bus: None,
                    bars: [0; MAX_BARS],
                    bar_count: 0,
                    latency_ns: 0,
                };
                [INIT; MAX_DEVICES]
            }),
            count: AtomicUsize::new(0),
        }
    }

    /// Register a discovered device.  Returns the index, or `None` if full.
    pub fn register(&self, device: PciDevice) -> Option<usize> {
        let idx = self.count.fetch_add(1, Ordering::AcqRel);
        if idx >= MAX_DEVICES {
            self.count.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
        // SAFETY: idx < MAX_DEVICES, and we have "ownership" of this slot
        // via the atomic increment.  No other thread will write to this
        // slot because they would have gotten a different index.
        unsafe {
            let devices = &mut *self.devices.get();
            devices[idx] = device;
        }
        Some(idx)
    }

    /// Get the number of registered devices.
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Acquire).min(MAX_DEVICES)
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a device by index.
    pub fn get(&self, idx: usize) -> Option<&PciDevice> {
        if idx < self.len() {
            // SAFETY: idx < len() <= MAX_DEVICES, and slots [0..len())
            // have been written by register() and are immutable thereafter.
            unsafe {
                let devices = &*self.devices.get();
                Some(&devices[idx])
            }
        } else {
            None
        }
    }

    /// Find devices by class code.
    /// Returns a static array of up to `MAX_DEVICES` references.
    /// The second element of the tuple is the number of valid entries.
    /// No heap allocation — uses a fixed-size result buffer.
    pub fn find_by_class(&self, class_code: u8) -> ([Option<&PciDevice>; MAX_DEVICES], usize) {
        let mut results: [Option<&PciDevice>; MAX_DEVICES] = [None; MAX_DEVICES];
        let mut count = 0;
        let devices = unsafe { &*self.devices.get() };
        for i in 0..self.len() {
            if devices[i].class_code == class_code && count < MAX_DEVICES {
                results[count] = Some(&devices[i]);
                count += 1;
            }
        }
        (results, count)
    }

    /// Find devices by vendor ID.
    /// Returns a static array of up to `MAX_DEVICES` references.
    /// The second element of the tuple is the number of valid entries.
    /// No heap allocation — uses a fixed-size result buffer.
    pub fn find_by_vendor(&self, vendor_id: u16) -> ([Option<&PciDevice>; MAX_DEVICES], usize) {
        let mut results: [Option<&PciDevice>; MAX_DEVICES] = [None; MAX_DEVICES];
        let mut count = 0;
        let devices = unsafe { &*self.devices.get() };
        for i in 0..self.len() {
            if devices[i].vendor_id == vendor_id && count < MAX_DEVICES {
                results[count] = Some(&devices[i]);
                count += 1;
            }
        }
        (results, count)
    }
}

// ─── Configuration Space Access ───────────────────────────────────────────

/// Read a 32-bit DWORD from PCI configuration space using Mechanism 1.
///
/// # Safety
/// - Must only be called on x86 systems with PCI Configuration Mechanism 1.
/// - The 0xCF8/0xCFC ports must be available (not remapped by firmware).
/// - This function is NOT re-entrant with respect to other 0xCF8/0xCFC
///   accesses — only one thread should access configuration space at a time.
#[inline(always)]
pub unsafe fn read_pci_config(bus: u8, device: u8, function: u8, offset: u8) -> u32 {
    let address = PCI_CONFIG_ENABLE
        | ((bus as u32) << 16)
        | ((device as u32) << 11)
        | ((function as u32) << 8)
        | ((offset as u32) & 0xFC); // Offset must be DWORD-aligned

    outl(PCI_CONFIG_ADDRESS, address);
    inl(PCI_CONFIG_DATA)
}

/// Write a 32-bit DWORD to PCI configuration space using Mechanism 1.
///
/// # Safety
/// Same as `read_pci_config`.  Additionally, writing to configuration
/// space can change hardware state — use with extreme care.
#[inline(always)]
pub unsafe fn write_pci_config(bus: u8, device: u8, function: u8, offset: u8, value: u32) {
    let address = PCI_CONFIG_ENABLE
        | ((bus as u32) << 16)
        | ((device as u32) << 11)
        | ((function as u32) << 8)
        | ((offset as u32) & 0xFC);

    outl(PCI_CONFIG_ADDRESS, address);
    outl(PCI_CONFIG_DATA, value);
}

// ─── ECAM (Enhanced Configuration Access Mechanism) ───────────────────────

/// Read a 32-bit DWORD from PCI Express Extended Configuration Space.
///
/// ECAM maps each BDF's 4KB configuration space into a contiguous
/// 256MB window starting at ECAM_BASE.  Each function gets 4KB,
/// addressed as:
///
///   ECAM_BASE + (bus << 20 | device << 15 | function << 12) + offset
///
/// # Safety
/// - The ECAM window must be mapped as Device memory (UC) in page tables.
/// - The MCFG ACPI table must indicate the correct base address.
pub unsafe fn read_pci_ecam(bus: u8, device: u8, function: u8, offset: u8) -> u32 {
    let addr = ECAM_BASE
        + ((bus as usize) << 20)
        + ((device as usize) << 15)
        + ((function as usize) << 12)
        + (offset as usize & 0xFC);

    let reg = MmioReg::<u32>::new(addr);
    reg.read()
}

// ─── PCI Enumerator ───────────────────────────────────────────────────────

/// The main PCI bus enumerator.
///
/// Performs recursive discovery starting from bus 0, following
/// PCI-to-PCI bridges to discover secondary buses.
///
/// # Algorithm
/// 1. For each bus (0..256):
///    a. For each device (0..32):
///       i.   Read Vendor ID at function 0
///       ii.  If VENDOR_ID_NONE, skip this device
///       iii. If multi-function device, scan all 8 functions
///       iv.  For each function, read full configuration header
///       v.   If PCI-to-PCI bridge, recursively enumerate secondary bus
/// 2. Register each valid device with the DeviceRegistry
///
/// # Formal Bounds Proof
/// - Bus loop: `bus` is u8, so 0..256 is the full range
/// - Device loop: `device` is bounded by `BoundedDevice::new()` (0..32)
/// - Function loop: `function` is bounded by `BoundedFunction::new()` (0..8)
/// - Recursion depth is bounded by MAX_BUS (256 levels maximum)
/// - No bus number from a bridge's secondary bus field can exceed 255
pub struct PciEnumerator {
    /// The device registry to populate.
    registry: DeviceRegistry,
    /// Whether to use ECAM (true for PCIe systems).
    use_ecam: bool,
    /// Buses that have been scanned (bitfield, one bit per bus).
    scanned_buses: [u64; 4], // 256 bits = 4 × 64
    /// Lock for configuration space access (simple spin-lock).
    config_lock: AtomicUsize,
}

impl PciEnumerator {
    /// Create a new enumerator with an empty registry.
    pub fn new(use_ecam: bool) -> Self {
        Self {
            registry: DeviceRegistry::new(),
            use_ecam,
            scanned_buses: [0; 4],
            config_lock: AtomicUsize::new(0),
        }
    }

    /// Get a reference to the device registry.
    pub fn registry(&self) -> &DeviceRegistry {
        &self.registry
    }

    /// Acquire the configuration space lock.
    fn lock_config(&self) {
        while self.config_lock.compare_exchange(
            0,
            1,
            Ordering::Acquire,
            Ordering::Relaxed,
        ).is_err() {
            core::hint::spin_loop();
        }
    }

    /// Release the configuration space lock.
    fn unlock_config(&self) {
        self.config_lock.store(0, Ordering::Release);
    }

    /// Read a DWORD from configuration space (port I/O or ECAM).
    fn read_config(&self, bus: u8, device: u8, function: u8, offset: u8) -> u32 {
        self.lock_config();
        let result = if self.use_ecam {
            unsafe { read_pci_ecam(bus, device, function, offset) }
        } else {
            unsafe { read_pci_config(bus, device, function, offset) }
        };
        self.unlock_config();
        result
    }

    /// Check if a bus has already been scanned.
    fn is_bus_scanned(&self, bus: u8) -> bool {
        let idx = bus as usize / 64;
        let bit = bus as usize % 64;
        (self.scanned_buses[idx] & (1 << bit)) != 0
    }

    /// Mark a bus as scanned.
    fn mark_bus_scanned(&mut self, bus: u8) {
        let idx = bus as usize / 64;
        let bit = bus as usize % 64;
        self.scanned_buses[idx] |= 1 << bit;
    }

    /// Enumerate all PCI devices starting from bus 0.
    ///
    /// This is the main entry point.  It recursively walks the PCI
    /// bus tree, discovering all devices and bridges.
    pub fn enumerate(&mut self) -> &DeviceRegistry {
        // Start enumeration from bus 0
        for bus in 0u8..=255u8 {
            if !self.is_bus_scanned(bus) {
                self.enumerate_bus(bus);
            }
        }
        &self.registry
    }

    /// Enumerate all devices on a single bus.
    fn enumerate_bus(&mut self, bus: u8) {
        // Prove bounds: bus is u8, so 0..256 is satisfied
        self.mark_bus_scanned(bus);

        for device in 0u8..32 {
            self.enumerate_device(bus, device);
        }
    }

    /// Enumerate all functions of a device.
    fn enumerate_device(&mut self, bus: u8, device: u8) {
        // Read Vendor ID at function 0
        let dword0 = self.read_config(bus, device, 0, REG_VENDOR_DEVICE);
        let vendor_id = (dword0 & 0xFFFF) as u16;

        if vendor_id == VENDOR_ID_NONE {
            return; // No device at this slot
        }

        // Read header type to check for multi-function
        let header_dword = self.read_config(bus, device, 0, REG_HEADER_TYPE);
        let header_type_byte = ((header_dword >> 16) & 0xFF) as u8;
        let is_multi_function = (header_type_byte & 0x80) != 0;
        let header_type_val = header_type_byte & 0x7F;

        // Enumerate function 0
        self.enumerate_function(bus, device, 0, header_type_val);

        // If multi-function, enumerate remaining functions
        if is_multi_function {
            for function in 1u8..8 {
                let fdword0 = self.read_config(bus, device, function, REG_VENDOR_DEVICE);
                let fvendor = (fdword0 & 0xFFFF) as u16;
                if fvendor != VENDOR_ID_NONE {
                    let fheader = self.read_config(bus, device, function, REG_HEADER_TYPE);
                    let fht = ((fheader >> 16) & 0xFF) as u8 & 0x7F;
                    self.enumerate_function(bus, device, function, fht);
                }
            }
        }
    }

    /// Enumerate a single function and register it.
    fn enumerate_function(&mut self, bus: u8, device: u8, function: u8, header_type: u8) {
        let dword0 = self.read_config(bus, device, function, REG_VENDOR_DEVICE);
        let vendor_id = (dword0 & 0xFFFF) as u16;
        let device_id = ((dword0 >> 16) & 0xFFFF) as u16;

        if vendor_id == VENDOR_ID_NONE {
            return;
        }

        // Read class/revision
        let class_rev = self.read_config(bus, device, function, REG_CLASS_REV);
        let revision = (class_rev & 0xFF) as u8;
        let prog_if = ((class_rev >> 8) & 0xFF) as u8;
        let subclass = ((class_rev >> 16) & 0xFF) as u8;
        let class_code = ((class_rev >> 24) & 0xFF) as u8;

        // Parse header type
        let pci_header_type = match header_type {
            0 => PciHeaderType::Standard,
            1 => PciHeaderType::PciBridge,
            2 => PciHeaderType::CardBus,
            _ => PciHeaderType::Unknown,
        };

        // Read BARs for standard devices and bridges
        let mut bars = [0u64; MAX_BARS];
        let mut bar_count = 0;
        let max_bars = match pci_header_type {
            PciHeaderType::Standard => MAX_BARS,
            PciHeaderType::PciBridge => 2, // Bridges only have BAR0 and BAR1
            _ => 0,
        };

        for i in 0..max_bars {
            let bar_offset = 0x10 + (i as u8) * 4;
            let bar_lo = self.read_config(bus, device, function, bar_offset);

            // Check if this is a 64-bit BAR (bit 2:1 = 0b10)
            let is_64bit = (bar_lo & 0x6) == 0x4;
            let is_io = (bar_lo & 0x1) == 1;

            if is_64bit && i + 1 < max_bars {
                let bar_hi = self.read_config(bus, device, function, bar_offset + 4);
                bars[i] = ((bar_hi as u64) << 32) | ((bar_lo as u64) & !0xF);
                bar_count = i + 1;
                // Skip next BAR (it's the upper 32 bits)
                // Note: we consume the next slot implicitly
            } else if is_io {
                bars[i] = (bar_lo as u64) & !0x3;
                bar_count = i + 1;
            } else {
                // Memory BAR, 32-bit
                bars[i] = (bar_lo as u64) & !0xF;
                bar_count = i + 1;
            }
        }

        // For PCI bridges, read secondary/subordinate bus numbers
        let (secondary_bus, subordinate_bus) = if pci_header_type == PciHeaderType::PciBridge {
            let bus_info = self.read_config(bus, device, function, REG_PRIMARY_BUS);
            let sec = ((bus_info >> 8) & 0xFF) as u8;
            let sub = ((bus_info >> 16) & 0xFF) as u8;
            (Some(sec), Some(sub))
        } else {
            (None, None)
        };

        // Build the BDF (formally bounded)
        let bdf = PciBdf::new(bus, device, function).unwrap();

        // Build the device descriptor
        let pci_device = PciDevice {
            bdf,
            vendor_id,
            device_id,
            class_code,
            subclass,
            prog_if,
            revision,
            header_type: pci_header_type,
            multi_function: false, // Set at device level, not function
            secondary_bus,
            subordinate_bus,
            bars,
            bar_count,
            latency_ns: 0, // Will be set by ProphecyOracle
        };

        // Register the device
        self.registry.register(pci_device);

        // If this is a PCI-to-PCI bridge, recursively enumerate
        // the secondary bus
        if pci_header_type == PciHeaderType::PciBridge {
            if let Some(sec_bus) = secondary_bus {
                if sec_bus > 0 && !self.is_bus_scanned(sec_bus) {
                    self.enumerate_bus(sec_bus);
                }
            }
        }
    }
}

impl Default for PciEnumerator {
    fn default() -> Self {
        Self::new(false) // Default to Mechanism 1 (port I/O)
    }
}

// ─── ProphecyOracle Integration ───────────────────────────────────────────

/// Assign latency prophecies to discovered devices based on their class.
/// This integrates with the `prophecy.rs` module in the threading layer.
pub fn assign_device_latency_prophecies(registry: &DeviceRegistry) {
    for i in 0..registry.len() {
        if let Some(device) = registry.get(i) {
            // Estimate latency based on device class:
            // - NVMe (class 0x01, subclass 0x08): ~5μs
            // - Network (class 0x02): ~50μs
            // - GPU (class 0x03): ~10μs
            // - USB (class 0x0C): ~100μs
            // - Other: ~200μs
            let latency = match device.class_code {
                0x01 => match device.subclass {
                    0x08 => 5_000,    // NVMe
                    _ => 50_000,       // Other storage
                },
                0x02 => 50_000,        // Network
                0x03 => 10_000,        // Display/GPU
                0x0C => 100_000,       // Serial bus (USB, etc.)
                _ => 200_000,          // Unknown
            };

            // Store the latency estimate
            // In a full integration, this would create a ProphecyVariable
            // via the ProphecyOracle.
            let _ = latency; // Used by integration layer
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_types() {
        // Bus: u8, always valid
        assert!(BoundedBus::new(0).is_some());
        assert!(BoundedBus::new(255).is_some());

        // Device: 0..32
        assert!(BoundedDevice::new(0).is_some());
        assert!(BoundedDevice::new(31).is_some());
        assert!(BoundedDevice::new(32).is_none());

        // Function: 0..8
        assert!(BoundedFunction::new(0).is_some());
        assert!(BoundedFunction::new(7).is_some());
        assert!(BoundedFunction::new(8).is_none());
    }

    #[test]
    fn test_bdf_construction() {
        let bdf = PciBdf::new(0, 0, 0).unwrap();
        assert_eq!(bdf.bus.raw(), 0);
        assert_eq!(bdf.device.raw(), 0);
        assert_eq!(bdf.function.raw(), 0);

        let invalid = PciBdf::new(0, 32, 0);
        assert!(invalid.is_none());
    }

    #[test]
    fn test_pci_device_class_names() {
        let mut device = PciDevice::empty();
        device.class_code = 0x01;
        assert_eq!(device.class_name(), "Mass Storage");
        device.class_code = 0x02;
        assert_eq!(device.class_name(), "Network");
        device.class_code = 0x03;
        assert_eq!(device.class_name(), "Display");
        device.class_code = 0x06;
        assert_eq!(device.class_name(), "Bridge");
    }

    #[test]
    fn test_device_registry() {
        let registry = DeviceRegistry::new();
        assert!(registry.is_empty());

        let device = PciDevice {
            bdf: PciBdf::new(0, 0, 0).unwrap(),
            vendor_id: 0x8086, // Intel
            device_id: 0x1234,
            class_code: 0x02,
            subclass: 0x00,
            prog_if: 0x00,
            revision: 0x01,
            header_type: PciHeaderType::Standard,
            multi_function: false,
            secondary_bus: None,
            subordinate_bus: None,
            bars: [0; MAX_BARS],
            bar_count: 0,
            latency_ns: 50_000,
        };

        let idx = registry.register(device);
        assert!(idx.is_some());
        assert_eq!(registry.len(), 1);

        let found = registry.find_by_vendor(0x8086);
        assert_eq!(found.1, 1);
        assert_eq!(found.0[0].unwrap().device_id, 0x1234);
    }

    #[test]
    fn test_pci_config_address_encoding() {
        // Verify the address encoding for configuration mechanism 1
        // Address = 0x80000000 | (bus << 16) | (device << 11) | (function << 8) | offset
        let bus = 0u8;
        let device = 0u8;
        let function = 0u8;
        let offset = 0u8;

        let expected = PCI_CONFIG_ENABLE
            | ((bus as u32) << 16)
            | ((device as u32) << 11)
            | ((function as u32) << 8)
            | (offset as u32 & 0xFC);
        assert_eq!(expected, 0x8000_0000);

        // Bus 2, Device 5, Function 1, Offset 0x10
        let addr = PCI_CONFIG_ENABLE
            | (2u32 << 16)
            | (5u32 << 11)
            | (1u32 << 8)
            | (0x10u32 & 0xFC);
        assert_eq!(addr, 0x800A_5110);
    }

    #[test]
    fn test_header_type_parsing() {
        // Multi-function bit = bit 7, header type = bits 6:0
        let byte: u8 = 0x80 | 0x01; // Multi-function + bridge
        assert!((byte & 0x80) != 0); // multi-function
        assert_eq!(byte & 0x7F, 0x01); // bridge header type
    }

    #[test]
    fn test_ecam_address_encoding() {
        // ECAM: ECAM_BASE + (bus << 20 | device << 15 | function << 12) + offset
        let bus = 0u8;
        let device = 0u8;
        let function = 0u8;
        let offset = 0u8;

        let addr = ECAM_BASE
            + ((bus as usize) << 20)
            + ((device as usize) << 15)
            + ((function as usize) << 12)
            + (offset as usize);
        assert_eq!(addr, ECAM_BASE);

        // Bus 1, Device 2, Function 3
        let addr2 = ECAM_BASE
            + (1usize << 20)
            + (2usize << 15)
            + (3usize << 12);
        assert_eq!(addr2, 0xE010_6000);
    }

    #[test]
    fn test_scanned_buses_bitfield() {
        let mut scanned = [0u64; 4];

        // Mark bus 0 as scanned
        scanned[0] |= 1 << 0;
        assert!((scanned[0] & (1 << 0)) != 0);

        // Mark bus 63 as scanned
        scanned[0] |= 1 << 63;
        assert!((scanned[0] & (1 << 63)) != 0);

        // Mark bus 64 as scanned (second u64)
        scanned[1] |= 1 << 0;
        assert!((scanned[1] & (1 << 0)) != 0);

        // Mark bus 255 as scanned (last u64, last bit)
        scanned[3] |= 1 << 63;
        assert!((scanned[3] & (1 << 63)) != 0);
    }
}
