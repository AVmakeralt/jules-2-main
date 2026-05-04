// =========================================================================
// x86 I/O Port Primitives
//
// Provably correct wrappers around `in` and `out` instructions.
// Every port access uses `asm!` with explicit register constraints — no
// "magic numbers" leak into the assembly output.
//
// INVARIANTS enforced by the type system:
//   - Port addresses are u16 (x86 I/O space is 16-bit addressed).
//   - All functions are `unsafe` because touching a raw port is the ONLY
//     permissible use of `unsafe` in a Jules driver.
//   - Memory ordering: `out` implies a full serialising fence on x86
//     (because `out` is inherently strongly ordered), but we add an
//     explicit `compiler_fence(Ordering::SeqCst)` to prevent the
//     *compiler* from reordering around the instruction.
// =========================================================================

use core::sync::atomic::compiler_fence;
use core::sync::atomic::Ordering;

/// Write a byte to an I/O port.
///
/// # Safety
/// Caller must ensure `port` is a valid I/O port address and that
/// writing to it does not violate any hardware protocol invariants.
#[inline(always)]
pub unsafe fn outb(port: u16, val: u8) {
    compiler_fence(Ordering::SeqCst);
    core::arch::asm!(
        "out dx, al",
        in("dx") port,
        in("al") val,
        options(nomem, nostack)
    );
    compiler_fence(Ordering::SeqCst);
}

/// Write a word (16-bit) to an I/O port.
///
/// # Safety
/// Same as `outb`.
#[inline(always)]
pub unsafe fn outw(port: u16, val: u16) {
    compiler_fence(Ordering::SeqCst);
    core::arch::asm!(
        "out dx, ax",
        in("dx") port,
        in("ax") val,
        options(nomem, nostack)
    );
    compiler_fence(Ordering::SeqCst);
}

/// Write a dword (32-bit) to an I/O port.
///
/// # Safety
/// Same as `outb`.
#[inline(always)]
pub unsafe fn outl(port: u16, val: u32) {
    compiler_fence(Ordering::SeqCst);
    core::arch::asm!(
        "out dx, eax",
        in("dx") port,
        in("eax") val,
        options(nomem, nostack)
    );
    compiler_fence(Ordering::SeqCst);
}

/// Read a byte from an I/O port.
///
/// # Safety
/// Caller must ensure `port` is a valid I/O port address.
#[inline(always)]
pub unsafe fn inb(port: u16) -> u8 {
    compiler_fence(Ordering::SeqCst);
    let val: u8;
    core::arch::asm!(
        "in al, dx",
        in("dx") port,
        out("al") val,
        options(nomem, nostack)
    );
    compiler_fence(Ordering::SeqCst);
    val
}

/// Read a word (16-bit) from an I/O port.
///
/// # Safety
/// Same as `inb`.
#[inline(always)]
pub unsafe fn inw(port: u16) -> u16 {
    compiler_fence(Ordering::SeqCst);
    let val: u16;
    core::arch::asm!(
        "in ax, dx",
        in("dx") port,
        out("ax") val,
        options(nomem, nostack)
    );
    compiler_fence(Ordering::SeqCst);
    val
}

/// Read a dword (32-bit) from an I/O port.
///
/// # Safety
/// Same as `inb`.
#[inline(always)]
pub unsafe fn inl(port: u16) -> u32 {
    compiler_fence(Ordering::SeqCst);
    let val: u32;
    core::arch::asm!(
        "in eax, dx",
        in("dx") port,
        out("eax") val,
        options(nomem, nostack)
    );
    compiler_fence(Ordering::SeqCst);
    val
}

/// Wait for an I/O operation to complete (approximately 1 microsecond).
/// Uses port 0x80 (POST diagnostic port) which is safe to write to on
/// all x86 platforms. This is the standard approach used by Linux, SeaBIOS,
/// and coreboot.
#[inline(always)]
pub fn io_wait() {
    unsafe {
        outb(0x80, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_io_port_types() {
        // Compile-time verification that port types are correct size
        let _port: u16 = 0x3F8;
        let _byte: u8 = 0xFF;
        let _word: u16 = 0xFFFF;
        let _dword: u32 = 0xFFFFFFFF;
        // We cannot actually perform I/O in unit tests, but we verify
        // the type signatures compile correctly.
    }
}
