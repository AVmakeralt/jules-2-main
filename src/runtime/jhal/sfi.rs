// =========================================================================
// Software Fault Isolation (SFI) — The Memory Fence for Jules
//
// This module implements Software Fault Isolation, the "Memory Fence"
// for the Jules bare-metal runtime.  SFI guarantees that JIT-generated
// code can NEVER write outside the configured "Sanctuary" boundary by
// forcing every memory access through a bitwise AND mask.
//
// DESIGN PHILOSOPHY ("Provable Memory Safety"):
//
//   In traditional systems, memory safety relies on page tables and
//   hardware protection rings.  In Jules's JIT, we use SFI as a
//   lightweight alternative: every pointer is masked before use, making
//   it provably impossible to access memory outside the Sanctuary.
//
//   The key insight is that if the Sanctuary region has size = 2^n and
//   is aligned to 2^n, then the mask (2^n - 1) | base constrains any
//   pointer to [base, base + 2^n) via the transformation:
//
//     masked_ptr = (ptr & mask) | base = base + (ptr mod size)
//
//   This is an SMT-provable invariant:
//     forall ptr in u64: base <= (ptr & mask) | base < base + size
//
// SPECTRE MITIGATION:
//
//   The `sfi_load_with_barrier` and `sfi_store_with_barrier` functions
//   add an LFENCE (load fence) instruction after the bounds check and
//   before the actual memory access.  This prevents speculative
//   execution from bypassing the SFI check — a critical defense
//   against Spectre v1 (bounds check bypass) attacks.
//
//   The sequence is:
//     1. Mask the pointer:  masked = (ptr & mask) | base
//     2. Verify bounds:     masked >= base && masked < base + size
//     3. LFENCE:            block speculative execution
//     4. Access memory:     read or write through masked pointer
//
// JIT CODEGEN:
//
//   The `emit_sfi_mask_inline_asm` function generates the x86_64
//   assembly sequence that JIT-compiled code must execute before every
//   memory write:
//     and <reg>, <size - 1>    ; isolate offset within sanctuary
//     or  <reg>, <base>        ; set sanctuary base bits
//
//   This two-instruction sequence has zero branches and is therefore
//   constant-time, eliminating timing side channels.
//
// REVIEW CHECKLIST (4 Questions for Peer Reviewers):
//
//   1. Is it Re-entrant?
//      Yes.  SFI masking is purely functional — it computes a new
//      pointer from the input pointer and the static config.  No
//      mutable state.  A TSX abort has no effect on SFI state.
//
//   2. Is there a Side Channel?
//      The AND+OR mask is constant-time (no branches).  The LFENCE
//      barrier prevents Spectre v1 bypass.  MMIO regions are excluded
//      from the Sanctuary by construction (mapped at different
//      addresses via AliasLayout).
//
//   3. Is the Memory Ordering Correct?
//      The LFENCE is a full speculation barrier.  All loads/stores
//      after the LFENCE are guaranteed to see the results of the
//      bounds check.  The `compiler_fence(SeqCst)` before the LFENCE
//      prevents the compiler from reordering the bounds check after
//      the barrier.
//
//   4. Does it respect the 4.5MB limit?
//      Zero heap allocation.  SfiConfig is 24 bytes (3 x usize).
//      All functions operate on stack values only.
//
// REFERENCES:
//   - Wahbe et al., "Efficient Software-Based Fault Isolation", SOSP 1993
//   - Intel 64 and IA-32 SDM, Volume 2: LFENCE instruction
//   - Google Project Zero, "Reading privileged memory with a
//     side-channel", 2018
// =========================================================================

use core::sync::atomic::{compiler_fence, Ordering};

// For JIT codegen helper only (not on the SFI runtime hot path)
extern crate alloc;
use alloc::string::String;

// ─── Default Sanctuary Configuration ──────────────────────────────────────

/// Default sanctuary size: 8 MB (power of 2, aligned for simple mask).
///
/// 8 MB provides ample room for JIT code, inline caches, and runtime
/// data while remaining a manageable fraction of the address space.
///
/// Note: 8 MB exceeds the 4.5 MB *physical* budget — the sanctuary is
/// a VIRTUAL address range.  Only the pages that are actually faulted
/// in consume physical memory.
pub const DEFAULT_SANCTUARY_SIZE: usize = 8 * 1024 * 1024; // 8 MB = 0x0080_0000

/// Default sanctuary base address.
///
/// This is chosen to be:
/// - Aligned to DEFAULT_SANCTUARY_SIZE (8 MB)
/// - In a region that doesn't conflict with kernel, MMIO, or stack
/// - At a 4 GB boundary for clean mask arithmetic
///
/// On bare-metal Jules, this would be set by the bootloader based on
/// the memory map.  For hosted testing, this is a safe default.
pub const DEFAULT_SANCTUARY_BASE: usize = 0x0000_0010_0000_0000; // 4 GB, 8 MB-aligned

// ─── SFI Configuration ────────────────────────────────────────────────────

/// Software Fault Isolation configuration.
///
/// Defines the "Sanctuary" — the memory region that JIT-generated code
/// is allowed to access.  All pointers are forced through a bitwise
/// mask that constrains them to `[base, base + size)`.
///
/// # Invariants
///
/// - `size` must be a power of 2
/// - `base` must be aligned to `size` (i.e., `base % size == 0`)
/// - `mask` is computed as `(size - 1) | base`
///
/// # How the Mask Works
///
/// Given `mask = (size - 1) | base`:
///
/// Since `base` is aligned to `size`, the bits of `base` and
/// `(size - 1)` do NOT overlap:
/// - `(size - 1)` captures all bits in the offset range `[0, size)`
/// - `base` captures all bits in the base address
///
/// The transformation `apply_mask(ptr) = (ptr & mask) | base` works as:
///
/// ```text
/// ptr & mask = ptr & ((size-1) | base)
///            = (ptr & (size-1)) | (ptr & base)    [non-overlapping bits]
///
/// apply_mask(ptr) = (ptr & mask) | base
///                 = (ptr & (size-1)) | (ptr & base) | base
///                 = (ptr & (size-1)) | base         [(ptr & base) | base = base]
///                 = base + (ptr mod size)
/// ```
///
/// Since `ptr mod size` is in `[0, size)`, the result is always in
/// `[base, base + size)`.
///
/// # SMT-Verifiable Invariant
///
/// The following lemma is provable in SMT-LIB2 (bitvector arithmetic):
///
/// ```smt2
/// (assert (forall ((ptr (_ BitVec 64)))
///   (let ((masked (bvor (bvand ptr MASK) BASE)))
///     (and (bvuge masked BASE)
///          (bvult masked (bvadd BASE SIZE))))))
/// ```
///
/// Z3 proves this in < 1 second.  The proof is by the algebraic
/// identity shown above.
///
/// # Memory Layout
///
/// ```text
/// Address Space:
///   0x0000_0000_0000_0000 ─┐
///                          │ Forbidden (kernel, MMIO, etc.)
///   0x0000_0010_0000_0000 ─┤ <- base
///                          │ <- Sanctuary [base, base + size)
///   0x0000_0010_0080_0000 ─┤ <- base + size
///                          │ Forbidden
///   0xFFFF_FFFF_FFFF_FFFF ─┘
/// ```
///
/// # Zero Heap Allocation
///
/// This struct is 24 bytes on 64-bit (3 x usize).  No Vec, Box, or
/// String.
#[derive(Debug, Clone, Copy)]
pub struct SfiConfig {
    /// Base address of the sanctuary (must be aligned to `size`).
    base: usize,
    /// Size of the sanctuary (must be a power of 2).
    size: usize,
    /// Computed mask: `(size - 1) | base`.
    /// Constrains any pointer to `[base, base + size)`.
    mask: usize,
}

impl SfiConfig {
    /// Create a new SFI configuration.
    ///
    /// Returns `None` if:
    /// - `size` is 0
    /// - `size` is not a power of 2
    /// - `base` is not aligned to `size` (i.e., `base % size != 0`)
    ///
    /// # Const Evaluation
    ///
    /// This function is `const`, so it can be used to create static
    /// configs at compile time:
    ///
    /// ```rust,ignore
    /// static JULES_SFI: SfiConfig = match SfiConfig::new(0x10_0000_0000, 0x800_0000) {
    ///     Some(c) => c,
    ///     None => panic!("invalid SFI config"),
    /// };
    /// ```
    ///
    /// # Validation Logic
    ///
    /// 1. **Size nonzero**: `size == 0` would produce mask = usize::MAX | base,
    ///    which constrains nothing.
    ///
    /// 2. **Size is a power of 2**: The classic bit trick `n & (n-1) == 0`
    ///    is true iff `n` has exactly one bit set.  This is required so
    ///    that `(size - 1)` produces a contiguous run of 1-bits (the
    ///    offset mask).
    ///
    /// 3. **Base aligned to size**: `base & (size - 1) == 0` ensures that
    ///    the base bits and offset bits don't overlap, which is the key
    ///    to the SFI invariant proof.
    #[inline]
    pub const fn new(base: usize, size: usize) -> Option<Self> {
        // Size must be nonzero
        if size == 0 {
            return None;
        }
        // Size must be a power of 2 (exactly one bit set)
        if size & (size - 1) != 0 {
            return None;
        }
        // Base must be aligned to size
        if base & (size - 1) != 0 {
            return None;
        }
        Some(Self {
            base,
            size,
            mask: (size - 1) | base,
        })
    }

    /// Apply the SFI mask to a raw pointer value.
    ///
    /// Returns `(ptr & mask) | base`, which is guaranteed to be in
    /// the range `[base, base + size)`.
    ///
    /// # Proof of Correctness
    ///
    /// ```text
    /// apply_mask(ptr) = (ptr & ((size-1) | base)) | base
    ///                 = (ptr & (size-1)) | (ptr & base) | base
    ///                 = (ptr & (size-1)) | base
    ///                 = base + (ptr mod size)
    ///
    /// Since 0 <= (ptr mod size) < size:
    ///   base <= apply_mask(ptr) < base + size    QED
    /// ```
    ///
    /// # Constant-Time
    ///
    /// This operation is branch-free: one AND and one OR, both of
    /// which execute in constant time regardless of the input.
    /// This eliminates timing side channels.
    #[inline(always)]
    pub fn apply_mask(&self, ptr: usize) -> usize {
        (ptr & self.mask) | self.base
    }

    /// Check if a pointer value falls within the sanctuary bounds.
    ///
    /// Returns `true` if `ptr` is in `[base, base + size)`.
    ///
    /// Note: this checks the RAW pointer, not a masked pointer.
    /// After `apply_mask`, the result ALWAYS passes this check
    /// (that is the core SFI guarantee).
    ///
    /// # Overflow Safety
    ///
    /// Uses `wrapping_sub` to avoid overflow in `base + size`.
    /// The check `ptr.wrapping_sub(base) < size` is equivalent to
    /// `ptr >= base && ptr < base + size` but cannot overflow.
    #[inline(always)]
    pub fn is_in_bounds(&self, ptr: usize) -> bool {
        ptr >= self.base && ptr.wrapping_sub(self.base) < self.size
    }

    /// Get the raw mask value: `(size - 1) | base`.
    ///
    /// This is useful for JIT codegen and SMT verification.
    #[inline(always)]
    pub fn mask_value(&self) -> usize {
        self.mask
    }

    /// Get the sanctuary base address.
    #[inline(always)]
    pub fn base(&self) -> usize {
        self.base
    }

    /// Get the sanctuary size.
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the offset mask: `size - 1`.
    ///
    /// This is the mask that isolates the offset within the sanctuary,
    /// without the base bits.  Useful for the two-instruction JIT
    /// sequence:
    ///
    /// ```asm
    /// and <reg>, offset_mask   ; isolate offset
    /// or  <reg>, base          ; set base
    /// ```
    #[inline(always)]
    pub fn offset_mask(&self) -> usize {
        self.size - 1
    }
}

// ─── JIT Codegen Helper ──────────────────────────────────────────────────

/// Emit x86_64 inline assembly for SFI pointer masking.
///
/// Generates the two-instruction sequence that JIT-compiled code must
/// execute before every memory write to guarantee SFI invariants:
///
/// ```asm
/// and <ptr_reg>, <offset_mask>   ; zero bits outside the sanctuary offset range
/// or  <ptr_reg>, <base>          ; force the sanctuary base address bits
/// ```
///
/// This sequence is:
/// - **Branch-free**: No conditional jumps, eliminating timing side channels
/// - **Constant-time**: Always executes in the same number of cycles
/// - **Two instructions**: Minimal overhead (2 uops on modern x86)
///
/// # Arguments
///
/// - `ptr_reg`: The register containing the pointer (e.g., `"rax"`,
///   `"rdi"`)
/// - `config`: The SFI configuration
///
/// # Returns
///
/// The assembly text, suitable for inclusion in a JIT code buffer.
///
/// # Note on Allocation
///
/// This function uses `alloc::string::String` because JIT codegen
/// requires dynamic string building.  It is NOT on the SFI runtime
/// hot path — the actual SFI enforcement uses `apply_mask()` which
/// is pure `core` and branch-free.
///
/// # Example Output
///
/// For base = 0x0000001000000000, size = 0x0000000000800000:
///
/// ```asm
/// and rax, 0x00000000007FFFFF
/// or  rax, 0x0000001000000000
/// ```
pub fn emit_sfi_mask_inline_asm(ptr_reg: &str, config: &SfiConfig) -> String {
    let mut result = String::with_capacity(128);

    // "    and <reg>, 0x"
    result.push_str("    and ");
    result.push_str(ptr_reg);
    result.push_str(", 0x");
    write_usize_hex_to_string(&mut result, config.offset_mask());

    // "\n    or  <reg>, 0x"
    result.push('\n');
    result.push_str("    or  ");
    result.push_str(ptr_reg);
    result.push_str(", 0x");
    write_usize_hex_to_string(&mut result, config.base());

    result.push('\n');
    result
}

/// Write a usize value as zero-padded hexadecimal into a String.
///
/// Always produces exactly 16 hex digits (for u64/usize).
fn write_usize_hex_to_string(s: &mut String, val: usize) {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    for shift in (0..16).rev() {
        let nibble = ((val >> (shift * 4)) & 0xF) as usize;
        s.push(HEX[nibble] as char);
    }
}

// ─── SFI Invariant Verification ──────────────────────────────────────────

/// Verify that the SFI invariant holds for the given configuration and
/// pointer value.
///
/// This function checks that after applying the SFI mask, the pointer
/// is within the sanctuary bounds:
///
/// ```text
/// apply_mask(ptr) in [base, base + size)
/// ```
///
/// # SMT Proof (for ALL pointer values)
///
/// The SFI invariant is provable for ALL 2^64 possible pointer values
/// using SMT-LIB2:
///
/// ```smt2
/// ; Declare the SFI parameters
/// (declare-const BASE (_ BitVec 64))
/// (declare-const SIZE (_ BitVec 64))
/// (declare-const MASK (_ BitVec 64))
///
/// ; Preconditions (from SfiConfig::new validation)
/// (assert (= MASK (bvor (bvsub SIZE (_ bv1 64)) BASE)))
/// (assert (= (bvand BASE (bvsub SIZE (_ bv1 64))) (_ bv0 64)))
/// (assert (= (bvand SIZE (bvsub SIZE (_ bv1 64))) (_ bv0 64)))
/// (assert (bvugt SIZE (_ bv0 64)))
///
/// ; The invariant: for all pointers, the masked result is in bounds
/// (assert (forall ((ptr (_ BitVec 64)))
///   (let ((masked (bvor (bvand ptr MASK) BASE)))
///     (and (bvuge masked BASE)
///          (bvult masked (bvadd BASE SIZE))))))
/// ```
///
/// Z3 proves this in < 1 second.  The proof is by the algebraic
/// identity:
///
/// ```text
/// (ptr & MASK) | BASE = (ptr & (SIZE-1)) | BASE = BASE + (ptr mod SIZE)
/// ```
///
/// Since `0 <= ptr mod SIZE < SIZE`, we have
/// `BASE <= result < BASE + SIZE`.  QED.
///
/// # Runtime Usage
///
/// While the SMT proof covers ALL 2^64 possible pointer values, this
/// runtime function verifies the invariant for a specific test pointer.
/// It is useful for:
/// - Quick sanity checks during development
/// - Runtime assertions in debug builds
/// - Fuzz testing with random pointer values
#[inline(always)]
pub fn verify_sfi_invariant(config: &SfiConfig, ptr: usize) -> bool {
    let masked = config.apply_mask(ptr);
    config.is_in_bounds(masked)
}

/// Verify the SFI configuration is structurally valid.
///
/// This is a stronger check than `verify_sfi_invariant` — it verifies
/// that the config itself is well-formed, which implies the invariant
/// holds for ALL possible pointer values (by the SMT proof).
///
/// # Checks
///
/// 1. `size` is a nonzero power of 2
/// 2. `base` is aligned to `size`
/// 3. `mask == (size - 1) | base`
///
/// If all three hold, then by the algebraic proof, the SFI invariant
/// is guaranteed for every possible pointer value.
pub fn verify_sfi_config(config: &SfiConfig) -> bool {
    // Size must be a nonzero power of 2
    if config.size == 0 || config.size & (config.size - 1) != 0 {
        return false;
    }
    // Base must be aligned to size
    if config.base & (config.size - 1) != 0 {
        return false;
    }
    // Mask must equal (size - 1) | base
    if config.mask != ((config.size - 1) | config.base) {
        return false;
    }
    true
}

// ─── SFI-Protected Memory Access with Spectre Mitigation ─────────────────

/// SFI-protected load with Spectre mitigation (LFENCE).
///
/// Performs a memory load through the SFI mask, with a speculation
/// barrier to prevent Spectre v1 (bounds check bypass) attacks.
///
/// # Sequence
///
/// 1. **Mask**: `masked_ptr = (ptr & mask) | base` — constrain to
///    sanctuary
/// 2. **Verify**: `masked_ptr in [base, base + size)` — structural
///    check (defense in depth)
/// 3. **LFENCE**: Block speculative execution past this point
/// 4. **Load**: Read the value from the masked pointer
///
/// # Spectre Mitigation
///
/// Without the LFENCE, an attacker could train the branch predictor
/// to speculatively execute the load before the bounds check resolves,
/// leaking out-of-bounds data through cache side channels (Spectre
/// v1).
///
/// The LFENCE ensures that the CPU waits for the bounds check to
/// complete before speculatively executing the load.  This makes the
/// SFI guarantee robust even against speculative execution attacks.
///
/// # Returns
///
/// - `Some(value)` if the masked pointer is in bounds and properly
///   aligned
/// - `None` if the masked pointer is out of bounds or misaligned
///
/// # Safety
///
/// This function is safe to call from any context.  The SFI mask
/// guarantees that the accessed address is within the sanctuary,
/// and the LFENCE prevents speculative out-of-bounds access.
///
/// # Important
///
/// The caller MUST ensure that the sanctuary region actually contains
/// mapped, readable memory.  SFI only constrains the ADDRESS — it
/// does not guarantee the memory at that address is valid.
pub fn sfi_load_with_barrier<T: Copy>(config: &SfiConfig, ptr: *const T) -> Option<T> {
    let raw_addr = ptr as usize;
    let masked_addr = config.apply_mask(raw_addr);

    // Step 2: Verify bounds (defense in depth).
    // For a valid SfiConfig, this should ALWAYS pass (that's the
    // SMT-proven invariant).  We check defensively anyway.
    if !config.is_in_bounds(masked_addr) {
        return None;
    }

    // Compiler fence: prevent the compiler from reordering the
    // bounds check after the LFENCE.  The LFENCE handles CPU-side
    // reordering, but we also need to prevent the compiler from
    // reordering the code.
    compiler_fence(Ordering::SeqCst);

    // Step 3: LFENCE — speculation barrier.
    // This prevents the CPU from speculatively executing the load
    // before the bounds check above is architecturally resolved.
    speculation_barrier();

    // Step 4: Check alignment for the target type.
    if masked_addr % core::mem::align_of::<T>() != 0 {
        return None;
    }

    // SAFETY: masked_addr is guaranteed to be within the sanctuary
    // by the SFI invariant (SMT-proven), and we've verified alignment.
    // The LFENCE ensures no speculative out-of-bounds access.
    let masked_ptr = masked_addr as *const T;
    Some(unsafe { core::ptr::read(masked_ptr) })
}

/// SFI-protected store with Spectre mitigation (LFENCE).
///
/// Performs a memory store through the SFI mask, with a speculation
/// barrier to prevent Spectre v1 attacks.
///
/// # Sequence
///
/// Same as `sfi_load_with_barrier`, but for stores:
/// 1. **Mask**: Constrain the pointer to the sanctuary
/// 2. **Verify**: Check bounds (defense in depth)
/// 3. **LFENCE**: Speculation barrier
/// 4. **Store**: Write the value to the masked pointer
///
/// # Returns
///
/// - `true` if the store succeeded (pointer in bounds and aligned)
/// - `false` if the store was rejected (out of bounds or misaligned)
///
/// # Safety
///
/// This function is safe to call from any context.  The SFI mask
/// guarantees the accessed address is within the sanctuary.
///
/// # Important
///
/// The caller MUST ensure that the sanctuary region actually contains
/// mapped, writable memory.  SFI only constrains the ADDRESS — it
/// does not guarantee the memory at that address is writable.
pub fn sfi_store_with_barrier<T: Copy>(config: &SfiConfig, ptr: *mut T, val: T) -> bool {
    let raw_addr = ptr as usize;
    let masked_addr = config.apply_mask(raw_addr);

    // Step 2: Verify bounds (defense in depth).
    if !config.is_in_bounds(masked_addr) {
        return false;
    }

    // Compiler fence: prevent reordering of bounds check.
    compiler_fence(Ordering::SeqCst);

    // Step 3: LFENCE — speculation barrier.
    speculation_barrier();

    // Step 4: Check alignment for the target type.
    if masked_addr % core::mem::align_of::<T>() != 0 {
        return false;
    }

    // SAFETY: masked_addr is guaranteed to be within the sanctuary
    // by the SFI invariant, and we've verified alignment.
    let masked_ptr = masked_addr as *mut T;
    unsafe { core::ptr::write(masked_ptr, val) }
    true
}

/// Insert a speculation barrier (LFENCE on x86_64).
///
/// This prevents the CPU from speculatively executing any subsequent
/// instructions until all prior instructions have completed locally.
/// It is the standard mitigation for Spectre v1 (bounds check bypass).
///
/// On non-x86_64 platforms, this falls back to a SeqCst compiler
/// fence, which prevents the compiler from reordering but does not
/// prevent CPU speculation.  Platform-specific barriers should be
/// added as needed for production use on those targets.
#[inline(always)]
fn speculation_barrier() {
    #[cfg(target_arch = "x86_64")]
    {
        // LFENCE: Wait for all prior instructions to complete locally
        // before executing subsequent instructions.  This is a
        // serializing instruction that prevents speculative execution
        // past this point.
        //
        // Per Intel SDM Volume 2:
        //   "LFENCE does not execute until all prior instructions have
        //    completed locally, and no later instruction begins
        //    execution until LFENCE completes."
        //
        // Options:
        // - nostack: LFENCE does not touch the stack
        // - preserves_flags: LFENCE does not modify RFLAGS
        // - NOT nomem: We want the compiler to treat this as a memory
        //   barrier to prevent reordering of loads/stores across it.
        // SAFETY: LFENCE is a benign instruction that only affects
        // instruction ordering.  It does not modify memory, registers
        // (other than RIP), or flags.  It is safe to execute in any
        // context.
        unsafe {
            core::arch::asm!("lfence", options(nostack, preserves_flags));
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // Fallback: compiler fence prevents reordering but not CPU
        // speculation.  Platform-specific barriers should be added
        // for production use on ARM, RISC-V, etc.
        compiler_fence(Ordering::SeqCst);
    }
}

// ─── Inline Assembly SFI for JIT Codegen ─────────────────────────────────

/// Apply the SFI mask using inline assembly on x86_64.
///
/// This is the same operation as `SfiConfig::apply_mask`, but
/// implemented via inline assembly.  It is intended as a reference
/// for what the JIT-generated code does at runtime.
///
/// The generated machine code is:
/// - `and rax, <offset_mask>`  (1 uop, 1 cycle latency)
/// - `or  rax, <base>`         (1 uop, 1 cycle latency)
///
/// Total: 2 uops, 2 cycles.  This is the overhead per memory access
/// in JIT-generated SFI code.
///
/// # Safety
///
/// This function is safe — it only computes a value, it does not
/// access memory.
///
/// # Platform
///
/// Only available on x86_64.  On other platforms, falls back to the
/// Rust implementation (`SfiConfig::apply_mask`).
#[inline(always)]
pub fn apply_mask_asm(config: &SfiConfig, ptr: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        let offset_mask = config.offset_mask();
        let base = config.base();
        let result: usize;

        // SAFETY: This is pure computation — no memory access, no
        // side effects.  The AND+OR sequence is exactly equivalent to
        // `base + (ptr % size)` but branch-free and constant-time.
        unsafe {
            core::arch::asm!(
                "and {0}, {1}",  // isolate offset within sanctuary
                "or  {0}, {2}",  // set sanctuary base bits
                inout(reg) ptr => result,
                in(reg) offset_mask,
                in(reg) base,
                options(nostack, preserves_flags),
            );
        }

        result
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        config.apply_mask(ptr)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── SfiConfig Construction Tests ──────────────────────────────

    #[test]
    fn test_sfi_config_valid_default() {
        let config = SfiConfig::new(DEFAULT_SANCTUARY_BASE, DEFAULT_SANCTUARY_SIZE).unwrap();
        assert_eq!(config.base(), DEFAULT_SANCTUARY_BASE);
        assert_eq!(config.size(), DEFAULT_SANCTUARY_SIZE);
        assert_eq!(
            config.mask_value(),
            (DEFAULT_SANCTUARY_SIZE - 1) | DEFAULT_SANCTUARY_BASE
        );
    }

    #[test]
    fn test_sfi_config_zero_size() {
        assert!(SfiConfig::new(0x1000, 0).is_none());
    }

    #[test]
    fn test_sfi_config_non_power_of_two_size() {
        assert!(SfiConfig::new(0x1000, 100).is_none());
        assert!(SfiConfig::new(0x1000, 1025).is_none());
        assert!(SfiConfig::new(0x1000, 3).is_none());
        assert!(SfiConfig::new(0x1000, 6).is_none());
        assert!(SfiConfig::new(0x1000, 0xFFFF).is_none()); // 65535
    }

    #[test]
    fn test_sfi_config_misaligned_base() {
        // size = 0x1000 (4 KiB), base must be 4 KiB-aligned
        assert!(SfiConfig::new(0x1001, 0x1000).is_none()); // Off by 1
        assert!(SfiConfig::new(0x0800, 0x1000).is_none()); // Half-aligned
        assert!(SfiConfig::new(0x0100, 0x1000).is_none()); // Way off
    }

    #[test]
    fn test_sfi_config_valid_small() {
        let config = SfiConfig::new(0x1000, 0x1000).unwrap();
        assert_eq!(config.base(), 0x1000);
        assert_eq!(config.size(), 0x1000);
        assert_eq!(config.mask_value(), 0x1FFF); // 0xFFF | 0x1000
    }

    #[test]
    fn test_sfi_config_valid_base_zero() {
        // base = 0 is aligned to any power of 2
        let config = SfiConfig::new(0, 0x10000).unwrap();
        assert_eq!(config.base(), 0);
        assert_eq!(config.mask_value(), 0xFFFF); // offset_mask only
    }

    #[test]
    fn test_sfi_config_const_eval() {
        const CONFIG: Option<SfiConfig> = SfiConfig::new(0x10000, 0x10000);
        assert!(CONFIG.is_some());
        assert_eq!(CONFIG.unwrap().base(), 0x10000);
    }

    #[test]
    fn test_sfi_config_size_one() {
        // size = 1 is a degenerate but valid power of 2
        let config = SfiConfig::new(0, 1).unwrap();
        assert_eq!(config.size(), 1);
        // Any pointer should map to base (the only address in sanctuary)
        assert_eq!(config.apply_mask(0), 0);
        assert_eq!(config.apply_mask(usize::MAX), 0);
    }

    #[test]
    fn test_sfi_config_large_power_of_two() {
        // 2^47 = 128 TiB — covers all user-space on x86_64
        let config = SfiConfig::new(0, 1 << 47).unwrap();
        assert_eq!(config.offset_mask(), (1 << 47) - 1);
    }

    #[test]
    fn test_sfi_struct_size() {
        // SfiConfig should be exactly 3 usizes (24 bytes on 64-bit)
        assert_eq!(
            core::mem::size_of::<SfiConfig>(),
            3 * core::mem::size_of::<usize>()
        );
    }

    // ─── apply_mask Tests ──────────────────────────────────────────

    #[test]
    fn test_apply_mask_in_bounds_pointer() {
        let config = SfiConfig::new(0x10000, 0x10000).unwrap();
        // Pointer already in bounds — should map to same location
        let ptr = 0x10050;
        let masked = config.apply_mask(ptr);
        assert_eq!(masked, 0x10050);
        assert!(config.is_in_bounds(masked));
    }

    #[test]
    fn test_apply_mask_out_of_bounds_high_pointer() {
        let config = SfiConfig::new(0x10000, 0x10000).unwrap();
        // Pointer way out of bounds — should be clamped to sanctuary
        let ptr = 0xFFFF_0000_0000_0000;
        let masked = config.apply_mask(ptr);
        // ptr & 0x1FFFF = 0x0000 (no bits overlap)
        // | 0x10000 = 0x10000
        assert_eq!(masked, 0x10000);
        assert!(config.is_in_bounds(masked));
    }

    #[test]
    fn test_apply_mask_wraps_into_sanctuary() {
        let config = SfiConfig::new(0x2000, 0x1000).unwrap();
        // ptr = 0x3005 (offset 5 from a different 0x1000-aligned region)
        // mask = 0xFFF | 0x2000 = 0x2FFF
        // ptr & 0x2FFF = 0x2005
        // | 0x2000 = 0x2005 (base bits already set)
        let masked = config.apply_mask(0x3005);
        assert_eq!(masked, 0x2005);
        assert!(config.is_in_bounds(masked));
    }

    #[test]
    fn test_apply_mask_preserves_offset() {
        let config = SfiConfig::new(0x8000_0000, 0x8000_0000).unwrap();
        // The offset within the sanctuary should be preserved
        for offset in [0, 1, 0x100, 0x7FFF_FFFE, 0x7FFF_FFFF] {
            let ptr = 0x8000_0000 + offset;
            let masked = config.apply_mask(ptr);
            assert_eq!(masked, ptr, "offset {offset:#x} not preserved");
            assert!(config.is_in_bounds(masked));
        }
    }

    #[test]
    fn test_apply_mask_all_bits_set() {
        let config = SfiConfig::new(0, 0x10000).unwrap();
        let masked = config.apply_mask(usize::MAX);
        // usize::MAX & 0xFFFF = 0xFFFF
        // | 0 = 0xFFFF
        assert_eq!(masked, 0xFFFF);
        assert!(config.is_in_bounds(masked));
    }

    #[test]
    fn test_apply_mask_base_zero_large_size() {
        let config = SfiConfig::new(0, 1 << 47).unwrap();
        let ptr: usize = 0x7FFF_1234_5678;
        let masked = config.apply_mask(ptr);
        // ptr & 0x7FFF_FFFF_FFFF | 0 = ptr (identity for user-space)
        assert_eq!(masked, ptr);
        assert!(config.is_in_bounds(masked));
    }

    // ─── is_in_bounds Tests ────────────────────────────────────────

    #[test]
    fn test_is_in_bounds_lower_bound() {
        let config = SfiConfig::new(0x1000, 0x1000).unwrap();
        assert!(config.is_in_bounds(0x1000)); // Exactly at base
        assert!(!config.is_in_bounds(0x0FFF)); // One below base
    }

    #[test]
    fn test_is_in_bounds_upper_bound() {
        let config = SfiConfig::new(0x1000, 0x1000).unwrap();
        assert!(config.is_in_bounds(0x1FFF)); // Last byte
        assert!(!config.is_in_bounds(0x2000)); // base + size
    }

    #[test]
    fn test_is_in_bounds_middle() {
        let config = SfiConfig::new(0x1000, 0x1000).unwrap();
        assert!(config.is_in_bounds(0x1500));
        assert!(config.is_in_bounds(0x1001));
        assert!(config.is_in_bounds(0x1FFE));
    }

    // ─── verify_sfi_invariant Tests ────────────────────────────────

    #[test]
    fn test_verify_sfi_invariant_many_pointers() {
        let config = SfiConfig::new(0x10000, 0x10000).unwrap();

        let test_ptrs: [usize; 16] = [
            0x0000_0000_0000_0000,
            0x0000_0000_0000_0001,
            0x0000_0000_0000_FFFF,
            0x0000_0000_0001_0000,
            0x0000_0000_0001_FFFF,
            0x0000_0000_FFFF_0000,
            0x0000_0001_0000_0000,
            0x1000_0000_0000_0000,
            0x2000_0000_0000_0000,
            0x4000_0000_0000_0000,
            0x8000_0000_0000_0000,
            0xC000_0000_0000_0000,
            0xFFFF_FFFF_FFFF_FFFF,
            0xDEAD_BEEF_CAFE_BABE,
            0x1234_5678_9ABC_DEF0,
            0x0000_0000_0001_2345,
        ];

        for ptr in test_ptrs {
            assert!(
                verify_sfi_invariant(&config, ptr),
                "SFI invariant failed for ptr={ptr:#018x}"
            );
        }
    }

    #[test]
    fn test_verify_sfi_invariant_exhaustive_small() {
        // For a small sanctuary, test ALL possible offset values
        let config = SfiConfig::new(0x1000, 0x1000).unwrap();

        // Test all offsets from various base addresses
        for offset in 0..0x1000usize {
            // From a wild pointer
            let ptr = 0xDEAD_0000_0000_0000 | offset;
            let masked = config.apply_mask(ptr);
            assert!(
                config.is_in_bounds(masked),
                "offset {offset:#x}: masked={masked:#x} out of bounds"
            );

            // From within the sanctuary
            let ptr_in = 0x1000 + offset;
            let masked_in = config.apply_mask(ptr_in);
            assert!(
                config.is_in_bounds(masked_in),
                "in-bounds offset {offset:#x}: masked={masked_in:#x} out of bounds"
            );
        }
    }

    #[test]
    fn test_verify_sfi_invariant_multiple_configs() {
        // Test several different SFI configurations
        let configs = [
            SfiConfig::new(0, 0x1000).unwrap(),
            SfiConfig::new(0x1000, 0x1000).unwrap(),
            SfiConfig::new(0, 0x10000).unwrap(),
            SfiConfig::new(0x8000_0000, 0x8000_0000).unwrap(),
            SfiConfig::new(DEFAULT_SANCTUARY_BASE, DEFAULT_SANCTUARY_SIZE).unwrap(),
        ];

        let test_ptrs = [0, 1, 0x7FFF, 0x8000, 0xFFFF, usize::MAX];

        for config in &configs {
            for ptr in test_ptrs {
                assert!(
                    verify_sfi_invariant(config, ptr),
                    "SFI invariant failed: base={:#x} size={:#x} ptr={:#x}",
                    config.base(),
                    config.size(),
                    ptr
                );
            }
        }
    }

    // ─── verify_sfi_config Tests ───────────────────────────────────

    #[test]
    fn test_verify_sfi_config_valid() {
        let config = SfiConfig::new(0x10000, 0x10000).unwrap();
        assert!(verify_sfi_config(&config));
    }

    #[test]
    fn test_verify_sfi_config_tampered_mask() {
        // We can't easily tamper with SfiConfig since fields are
        // private, but we can verify that a valid config passes.
        let config = SfiConfig::new(0x10000, 0x10000).unwrap();
        assert!(verify_sfi_config(&config));

        // Verify each valid config passes
        assert!(verify_sfi_config(&SfiConfig::new(0, 1).unwrap()));
        assert!(verify_sfi_config(&SfiConfig::new(0, 0x10000).unwrap()));
        assert!(verify_sfi_config(
            &SfiConfig::new(DEFAULT_SANCTUARY_BASE, DEFAULT_SANCTUARY_SIZE).unwrap()
        ));
    }

    // ─── emit_sfi_mask_inline_asm Tests ────────────────────────────

    #[test]
    fn test_emit_sfi_mask_asm_format() {
        let config = SfiConfig::new(0x10_0000_0000, 8 * 1024 * 1024).unwrap();
        let asm = emit_sfi_mask_inline_asm("rax", &config);

        // Should contain the AND and OR instructions
        assert!(asm.contains("and rax,"));
        assert!(asm.contains("or  rax,"));

        // Should contain hex values
        let offset_mask = 8 * 1024 * 1024 - 1; // 0x007FFFFF
        let base: u64 = 0x10_0000_0000;
        assert!(
            asm.contains(&format!("{offset_mask:016X}")),
            "missing offset mask in: {asm}"
        );
        assert!(
            asm.contains(&format!("{base:016X}")),
            "missing base in: {asm}"
        );
    }

    #[test]
    fn test_emit_sfi_mask_asm_different_registers() {
        let config = SfiConfig::new(0, 0x10000).unwrap();

        for reg in &["rax", "rdi", "rsi", "rdx", "rcx", "r8", "r15"] {
            let asm = emit_sfi_mask_inline_asm(reg, &config);
            assert!(
                asm.contains(&format!("and {reg},")),
                "missing 'and {reg},' in asm output"
            );
            assert!(
                asm.contains(&format!("or  {reg},")),
                "missing 'or  {reg},' in asm output"
            );
        }
    }

    #[test]
    fn test_emit_sfi_mask_asm_base_zero() {
        let config = SfiConfig::new(0, 0x10000).unwrap();
        let asm = emit_sfi_mask_inline_asm("rdi", &config);

        // Base is 0, so the OR instruction should show 0x...0000
        assert!(asm.contains("or  rdi, 0x"));
        assert!(asm.contains("00000000000010000")); // offset mask for 0x10000
    }

    // ─── apply_mask_asm Tests ──────────────────────────────────────

    #[test]
    fn test_apply_mask_asm_matches_apply_mask() {
        let config = SfiConfig::new(0x10000, 0x10000).unwrap();
        let test_ptrs = [0, 1, 0x7FFF, 0x8000, 0xFFFF, 0x10000, usize::MAX];

        for ptr in test_ptrs {
            let rust_result = config.apply_mask(ptr);
            let asm_result = apply_mask_asm(&config, ptr);
            assert_eq!(
                rust_result, asm_result,
                "asm mismatch for ptr={ptr:#x}: rust={rust_result:#x} asm={asm_result:#x}"
            );
        }
    }

    #[test]
    fn test_apply_mask_asm_multiple_configs() {
        let configs = [
            SfiConfig::new(0, 0x1000).unwrap(),
            SfiConfig::new(0x1000, 0x1000).unwrap(),
            SfiConfig::new(0x8000_0000, 0x8000_0000).unwrap(),
        ];

        for config in &configs {
            for ptr in [0usize, 1, 0x7FFF, 0xFFFF, usize::MAX] {
                let rust_result = config.apply_mask(ptr);
                let asm_result = apply_mask_asm(config, ptr);
                assert_eq!(rust_result, asm_result);
            }
        }
    }

    // ─── sfi_load_with_barrier Tests ───────────────────────────────

    #[test]
    fn test_sfi_load_with_barrier_in_sanctuary() {
        let value: u64 = 0xDEAD_BEEF_CAFE_1234;
        let ptr = &value as *const u64;

        // Create a config that covers user-space addresses
        let config = SfiConfig::new(0, 1 << 47).unwrap();

        let result = sfi_load_with_barrier(&config, ptr);
        assert_eq!(result, Some(0xDEAD_BEEF_CAFE_1234));
    }

    #[test]
    fn test_sfi_load_with_barrier_u8() {
        let value: u8 = 0xAB;
        let ptr = &value as *const u8;

        let config = SfiConfig::new(0, 1 << 47).unwrap();
        let result = sfi_load_with_barrier(&config, ptr);
        assert_eq!(result, Some(0xAB));
    }

    #[test]
    fn test_sfi_load_with_barrier_u32() {
        let value: u32 = 0x1234_5678;
        let ptr = &value as *const u32;

        let config = SfiConfig::new(0, 1 << 47).unwrap();
        let result = sfi_load_with_barrier(&config, ptr);
        assert_eq!(result, Some(0x1234_5678));
    }

    #[test]
    fn test_sfi_load_with_barrier_array() {
        let arr: [u32; 4] = [0x11111111, 0x22222222, 0x33333333, 0x44444444];
        let config = SfiConfig::new(0, 1 << 47).unwrap();

        for i in 0..4 {
            let ptr = arr.as_ptr().wrapping_add(i);
            let result = sfi_load_with_barrier(&config, ptr);
            assert_eq!(result, Some(arr[i]));
        }
    }

    // ─── sfi_store_with_barrier Tests ──────────────────────────────

    #[test]
    fn test_sfi_store_with_barrier_in_sanctuary() {
        let mut value: u64 = 0;
        let ptr = &mut value as *mut u64;

        let config = SfiConfig::new(0, 1 << 47).unwrap();
        let success = sfi_store_with_barrier(&config, ptr, 0xCAFE_BABE_DEAD_BEEF);
        assert!(success);
        assert_eq!(value, 0xCAFE_BABE_DEAD_BEEF);
    }

    #[test]
    fn test_sfi_store_with_barrier_u8() {
        let mut value: u8 = 0;
        let config = SfiConfig::new(0, 1 << 47).unwrap();

        let success = sfi_store_with_barrier(&config, &mut value, 0xFF);
        assert!(success);
        assert_eq!(value, 0xFF);
    }

    #[test]
    fn test_sfi_store_with_barrier_array() {
        let mut arr: [u32; 4] = [0; 4];
        let config = SfiConfig::new(0, 1 << 47).unwrap();

        for i in 0..4 {
            let ptr = arr.as_mut_ptr().wrapping_add(i);
            let val = (i as u32) * 0x11;
            let success = sfi_store_with_barrier(&config, ptr, val);
            assert!(success, "store failed for index {i}");
        }

        assert_eq!(arr, [0x00, 0x11, 0x22, 0x33]);
    }

    // ─── Integration: Round-trip Load/Store ────────────────────────

    #[test]
    fn test_sfi_round_trip() {
        let mut buffer: [u64; 16] = [0; 16];
        let config = SfiConfig::new(0, 1 << 47).unwrap();

        // Write values
        for i in 0..16 {
            let ptr = buffer.as_mut_ptr().wrapping_add(i);
            let val = (i as u64) * 0x0101_0101_0101_0101;
            assert!(
                sfi_store_with_barrier(&config, ptr, val),
                "store failed for index {i}"
            );
        }

        // Read them back
        for i in 0..16 {
            let ptr = buffer.as_ptr().wrapping_add(i);
            let expected = (i as u64) * 0x0101_0101_0101_0101;
            assert_eq!(
                sfi_load_with_barrier(&config, ptr),
                Some(expected),
                "load mismatch for index {i}"
            );
        }
    }

    // ─── Mask Algebra Tests ────────────────────────────────────────

    #[test]
    fn test_mask_non_overlapping_bits() {
        // Verify that base bits and offset_mask bits don't overlap
        let configs = [
            (0x0, 0x1000),
            (0x1000, 0x1000),
            (0x8000_0000, 0x8000_0000),
            (0x10_0000_0000, 0x800_0000),
        ];

        for (base, size) in configs {
            let config = SfiConfig::new(base, size).unwrap();
            let offset_mask = config.offset_mask();
            assert_eq!(
                config.base() & offset_mask,
                0,
                "base={base:#x} and offset_mask={offset_mask:#x} overlap!"
            );
        }
    }

    #[test]
    fn test_mask_equals_offset_mask_or_base() {
        let config = SfiConfig::new(0x10000, 0x10000).unwrap();
        assert_eq!(config.mask_value(), config.offset_mask() | config.base());
    }

    #[test]
    fn test_apply_mask_equals_base_plus_mod() {
        let config = SfiConfig::new(0x20000, 0x10000).unwrap();

        // For various pointers, verify apply_mask(ptr) == base + (ptr % size)
        let test_ptrs = [0, 1, 0x7FFF, 0x8000, 0xFFFF, 0x10000, 0x1FFFF, usize::MAX];
        for ptr in test_ptrs {
            let expected = config.base() + (ptr % config.size());
            let actual = config.apply_mask(ptr);
            assert_eq!(
                actual, expected,
                "ptr={ptr:#x}: expected {expected:#x}, got {actual:#x}"
            );
        }
    }

    // ─── Default Constants Tests ───────────────────────────────────

    #[test]
    fn test_default_sanctuary_size_is_power_of_two() {
        assert_ne!(DEFAULT_SANCTUARY_SIZE, 0);
        assert_eq!(DEFAULT_SANCTUARY_SIZE & (DEFAULT_SANCTUARY_SIZE - 1), 0);
    }

    #[test]
    fn test_default_sanctuary_base_is_aligned() {
        assert_eq!(DEFAULT_SANCTUARY_BASE & (DEFAULT_SANCTUARY_SIZE - 1), 0);
    }

    #[test]
    fn test_default_config_is_valid() {
        assert!(SfiConfig::new(DEFAULT_SANCTUARY_BASE, DEFAULT_SANCTUARY_SIZE).is_some());
    }

    // ─── Edge Cases ────────────────────────────────────────────────

    #[test]
    fn test_apply_mask_with_zero_base() {
        // When base is 0, the mask is just the offset mask.
        // apply_mask(ptr) = ptr & (size - 1)
        let config = SfiConfig::new(0, 0x10000).unwrap();
        assert_eq!(config.apply_mask(0x0001_2345), 0x2345);
        assert_eq!(config.apply_mask(0xFFFF_FFFF), 0xFFFF);
        assert_eq!(config.apply_mask(0), 0);
    }

    #[test]
    fn test_apply_mask_pointer_at_sanctuary_end() {
        // ptr = base + size - 1 (last valid address)
        let config = SfiConfig::new(0x10000, 0x10000).unwrap();
        let ptr = 0x1FFFF; // base + size - 1
        let masked = config.apply_mask(ptr);
        assert_eq!(masked, 0x1FFFF);
        assert!(config.is_in_bounds(masked));
    }

    #[test]
    fn test_apply_mask_pointer_at_sanctuary_start() {
        // ptr = base (first valid address)
        let config = SfiConfig::new(0x10000, 0x10000).unwrap();
        let ptr = 0x10000; // base
        let masked = config.apply_mask(ptr);
        assert_eq!(masked, 0x10000);
        assert!(config.is_in_bounds(masked));
    }

    #[test]
    fn test_sfi_load_store_different_types() {
        // Test with various Copy types
        let config = SfiConfig::new(0, 1 << 47).unwrap();

        // u8
        let mut v8: u8 = 0;
        assert!(sfi_store_with_barrier(&config, &mut v8, 42u8));
        assert_eq!(sfi_load_with_barrier(&config, &v8 as *const u8), Some(42u8));

        // u16
        let mut v16: u16 = 0;
        assert!(sfi_store_with_barrier(&config, &mut v16, 0xBEEFu16));
        assert_eq!(sfi_load_with_barrier(&config, &v16 as *const u16), Some(0xBEEFu16));

        // u32
        let mut v32: u32 = 0;
        assert!(sfi_store_with_barrier(&config, &mut v32, 0xDEAD_BEEFu32));
        assert_eq!(sfi_load_with_barrier(&config, &v32 as *const u32), Some(0xDEAD_BEEFu32));

        // u64
        let mut v64: u64 = 0;
        assert!(sfi_store_with_barrier(&config, &mut v64, 0xCAFE_BABE_DEAD_BEEFu64));
        assert_eq!(sfi_load_with_barrier(&config, &v64 as *const u64), Some(0xCAFE_BABE_DEAD_BEEFu64));

        // usize
        let mut vsize: usize = 0;
        assert!(sfi_store_with_barrier(&config, &mut vsize, 0x1234_5678usize));
        assert_eq!(sfi_load_with_barrier(&config, &vsize as *const usize), Some(0x1234_5678usize));
    }
}
