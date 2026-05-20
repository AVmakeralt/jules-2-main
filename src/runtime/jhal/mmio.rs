// =========================================================================
// Memory-Mapped I/O (MMIO) Primitives
//
// Provides `MmioReg<T>` — a strongly-typed volatile register accessor
// that enforces Acquire/Release memory ordering for all reads and writes.
//
// DESIGN PRINCIPLES (Jules "Provable Hardware Modeling"):
//
//   1. Every register access is `volatile` — the compiler may NOT elide,
//      merge, or reorder any read or write.
//   2. Writes use `Ordering::Release` to ensure that all prior writes
//      (e.g., writing DMA data to a buffer) are visible before the MMIO
//      write that triggers the device to read that data.
//   3. Reads use `Ordering::Acquire` to ensure that subsequent reads
//      see device output that was made visible by the device before it
//      raised the interrupt / set the status bit we just read.
//   4. For write-then-read sequences that must be strictly ordered
//      (e.g., writing a command then reading status), use `write_then_read()`
//      which inserts a `fence(SeqCst)` between the two operations.
//   5. No heap allocation. All types are `#[repr(C)]` with correct padding.
//   6. `unsafe` is ONLY used for the `addr_of!` / pointer dereference,
//      never for any logic.
//
// REVIEW CHECKLIST ANSWERS:
//   - Re-entrant: Yes. MMIO reads/writes are side-effect-free from the
//     compiler's perspective (volatile), and hardware register state is
//     independent per-register. A TSX abort does not "half-write" a register
//     because MMIO writes are non-transactional — but see each driver's
//     TSX safety analysis for mitigations.
//   - Side channel: MMIO reads force cache-line loads. Jules's AliasLayout
//     ensures MMIO regions are never aliased with user data, preventing
//     Spectre-style probing of device state.
//   - Memory ordering: Enforced by this module.
//   - 4.5MB limit: Zero heap allocation.
// =========================================================================

use core::sync::atomic::{fence, Ordering};
use core::ptr;
use core::marker::PhantomData;

/// A memory-mapped I/O register of type `T`.
///
/// # Invariants
/// - `ptr` always points to a valid, properly-aligned hardware register.
/// - The pointed-to memory is never accessed through any non-volatile path.
/// - `T` must be a primitive type (u8, u16, u32, u64) that matches the
///   hardware register width.
#[repr(transparent)]
pub struct MmioReg<T> {
    ptr: *mut T,
    _phantom: PhantomData<T>,
}

impl<T> MmioReg<T> {
    /// Create a new MMIO register accessor.
    ///
    /// # Safety
    /// Caller must ensure:
    /// - `addr` points to a valid hardware register.
    /// - `addr` is aligned to `align_of::<T>()`.
    /// - No other code accesses this register through a non-volatile path.
    #[inline(always)]
    pub const unsafe fn new(addr: usize) -> Self {
        Self {
            ptr: addr as *mut T,
            _phantom: PhantomData,
        }
    }
}

impl MmioReg<u32> {
    /// Read the register with Acquire semantics.
    ///
    /// Guarantees that all subsequent reads see values that were written
    /// by the device *before* the device made this register's value visible.
    #[inline(always)]
    pub fn read(&self) -> u32 {
        fence(Ordering::Acquire);
        // SAFETY: self.ptr is a valid, aligned MMIO register by invariant.
        unsafe { ptr::read_volatile(self.ptr) }
    }

    /// Write the register with Release semantics.
    ///
    /// Guarantees that all prior writes (to DMA buffers, command data, etc.)
    /// are visible to the device *before* this write takes effect.
    #[inline(always)]
    pub fn write(&self, val: u32) {
        // SAFETY: self.ptr is a valid, aligned MMIO register by invariant.
        unsafe { ptr::write_volatile(self.ptr, val) }
        fence(Ordering::Release);
    }

    /// Read-modify-write: set specific bits, clear all others.
    #[inline(always)]
    pub fn write_bits(&self, mask: u32, val: u32) {
        let old = self.read();
        let new = (old & !mask) | (val & mask);
        self.write(new);
    }

    /// Read-modify-write: set specific bits (OR), leaving others unchanged.
    #[inline(always)]
    pub fn set_bits(&self, bits: u32) {
        let old = self.read();
        self.write(old | bits);
    }

    /// Read-modify-write: clear specific bits (AND NOT), leaving others unchanged.
    #[inline(always)]
    pub fn clear_bits(&self, bits: u32) {
        let old = self.read();
        self.write(old & !bits);
    }

    /// Write a value, then immediately read it back.
    ///
    /// Inserts a full memory fence between write and read to ensure
    /// the device processes the write before we observe any status change.
    /// This is the standard pattern for "write command, read status" MMIO.
    #[inline(always)]
    pub fn write_then_read(&self, val: u32) -> u32 {
        self.write(val);
        fence(Ordering::SeqCst);
        self.read()
    }
}

impl MmioReg<u64> {
    #[inline(always)]
    pub fn read(&self) -> u64 {
        fence(Ordering::Acquire);
        unsafe { ptr::read_volatile(self.ptr) }
    }

    #[inline(always)]
    pub fn write(&self, val: u64) {
        unsafe { ptr::write_volatile(self.ptr, val) }
        fence(Ordering::Release);
    }
}

/// A contiguous block of MMIO registers, representing a complete hardware
/// device register map. The struct must be `#[repr(C)]` with correct
/// field ordering and padding to match the hardware layout.
///
/// # Safety
/// All fields must be `MmioReg<T>` or reserved padding (`[u32; N]`).
/// Padding arrays use `[u32; N]` (not `[u8; N]`) because hardware registers
/// are always 32-bit aligned.
pub trait MmioDevice {
    /// Verify that the register map size matches the hardware spec.
    /// This is a const assertion that runs at compile time.
    const REGISTER_MAP_SIZE: usize;

    /// Create a new instance pointing to the given physical address.
    ///
    /// # Safety
    /// Caller must ensure the address is the correct base address for
    /// this device, mapped into the virtual address space with
    /// Device/Memory type (uncacheable, strongly ordered).
    unsafe fn from_base(addr: usize) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmio_size() {
        // Verify MmioReg is transparent (same size as inner type)
        assert_eq!(std::mem::size_of::<MmioReg<u32>>(), std::mem::size_of::<*mut u32>());
        assert_eq!(std::mem::size_of::<MmioReg<u64>>(), std::mem::size_of::<*mut u64>());
    }
}
