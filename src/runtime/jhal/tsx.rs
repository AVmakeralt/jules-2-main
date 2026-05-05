// =========================================================================
// TSX Transaction Wrappers & AMX Tile Operations
//
// TSX (Transactional Synchronization Extensions): Hardware transactions
// for lock-free concurrency. If two cores touch the same cache line,
// the CPU detects the conflict and automatically rolls back.
//
// AMX (Advanced Matrix Extensions): Dedicated tile registers for matrix
// math. Offloads the Learned Scheduler's neural inference from the
// main execution pipeline.
//
// DESIGN RULES:
//   - No transaction may exceed L1D cache size (32KB) or it WILL abort
//   - TSX transactions are NOT re-entrant — nested xbegin always fails
//   - MMIO/I/O port writes are NOT transactional — buffer them
//   - AMX must be initialized before use and released before context switch
//
// REFERENCES:
//   Intel SDM Vol 2, Chapter 4 (RTM instructions)
//   Intel AMX Architecture Specification
// =========================================================================

use core::sync::atomic::{AtomicBool, Ordering};

// ─── TSX Abort Status Bits ──────────────────────────────────────────────────

pub const TX_ABORT_EXPLICIT: u32 = 1 << 0;
pub const TX_ABORT_RETRY: u32 = 1 << 1;
pub const TX_ABORT_CONFLICT: u32 = 1 << 2;
pub const TX_ABORT_CAPACITY: u32 = 1 << 3;
pub const TX_ABORT_DEBUG: u32 = 1 << 4;
pub const TX_ABORT_NESTED: u32 = 1 << 5;

// ─── TSX Transaction Status ─────────────────────────────────────────────────

/// Status returned by xbegin(). If aborted, contains diagnostic bits.
#[derive(Debug, Clone, Copy)]
pub struct TsxStatus {
    pub raw: u32,
}

impl TsxStatus {
    /// Transaction started successfully (we are now inside the transaction).
    pub fn started(&self) -> bool { self.raw == core::u32::MAX }

    /// Transaction was aborted (we landed at the fallback label).
    pub fn aborted(&self) -> bool { !self.started() }

    /// User abort code (bits 31:24 of EAX after abort).
    pub fn abort_code(&self) -> u32 { (self.raw >> 24) & 0xFF }

    /// Abort due to memory conflict (another core touched same cache line).
    pub fn is_conflict(&self) -> bool { (self.raw & TX_ABORT_CONFLICT) != 0 }

    /// Abort due to capacity overflow (touched > L1D size).
    pub fn is_capacity(&self) -> bool { (self.raw & TX_ABORT_CAPACITY) != 0 }

    /// Explicit xabort was called.
    pub fn is_explicit(&self) -> bool { (self.raw & TX_ABORT_EXPLICIT) != 0 }

    /// Get the raw status value.
    pub fn raw(&self) -> u32 { self.raw }
}

// ─── TSX Intrinsics ─────────────────────────────────────────────────────────

/// Begin a TSX restricted transaction.
/// Returns TsxStatus::started() == true if inside the transaction,
/// or TsxStatus with abort diagnostics if the transaction failed.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn xbegin() -> TsxStatus {
    // Use a simple approach: xbegin, then check if we're in the transaction.
    // The EAX value after xbegin failure contains the abort status.
    // We store it to a stack-allocated variable.
    let mut status: u32 = 0;
    core::arch::asm!(
        "xbegin 2f",
        // Success path: inside transaction
        "mov dword ptr [rcx], 0xFFFFFFFF",
        "jmp 3f",
        "2:",
        // Abort path: EAX = abort info, store to output
        "mov dword ptr [rcx], eax",
        "3:",
        in("rcx") &mut status,
        out("eax") _,
        options(nostack)
    );
    TsxStatus { raw: status }
}

/// End a TSX transaction successfully.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn xend() {
    core::arch::asm!("xend", options(nomem, nostack));
}

/// Abort a TSX transaction with the given code (bits 31:24 of EAX).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn xabort(code: u8) {
    // XABORT encoding: 0xC6 F8 /0 ib where ib is the abort code.
    // Using .byte directive to emit the correct instruction.
    core::arch::asm!(
        ".byte 0xC6, 0xF8, {}",
        in(reg_byte) code,
        options(noreturn)
    );
}

/// Test whether currently executing inside a TSX transaction.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn xtest() -> bool {
    let zf: u8;
    core::arch::asm!(
        "xtest",
        "setz {0}",
        out(reg_byte) zf,
        options(nomem, nostack)
    );
    zf != 0
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn xbegin() -> TsxStatus { TsxStatus { raw: 0 } }
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn xend() {}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn xabort(_code: u8) {}
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn xtest() -> bool { false }

// ─── Transaction Bounds ─────────────────────────────────────────────────────

/// Maximum L1D cache size for capacity proof. If a transaction writes
/// more than this, it WILL abort due to capacity overflow.
pub const L1D_CACHE_SIZE: usize = 32 * 1024; // 32 KB

/// Prove that a transaction fits within L1D.
pub const fn prove_transaction_bound(write_footprint: usize) -> bool {
    write_footprint < L1D_CACHE_SIZE
}

// ─── RAII Transaction Guard ─────────────────────────────────────────────────

/// RAII wrapper for TSX transactions. Calls xbegin on creation,
/// xend on commit, and xabort on drop if not committed.
pub struct TsxTransaction {
    active: bool,
    committed: bool,
    write_bytes: usize,
}

impl TsxTransaction {
    /// Begin a new transaction with retry logic.
    /// Returns None if xbegin fails after max_retries.
    pub fn begin(max_retries: u32) -> Option<Self> {
        for _ in 0..max_retries {
            let status = unsafe { xbegin() };
            if status.started() {
                return Some(Self { active: true, committed: false, write_bytes: 0 });
            }
            // If capacity abort, don't retry (it'll fail again)
            if status.is_capacity() {
                return None;
            }
        }
        None
    }

    /// Record a memory write for capacity tracking.
    pub fn record_write(&mut self, size: usize) {
        self.write_bytes += size;
    }

    /// Check if approaching L1D capacity (within 90%).
    pub fn is_near_capacity(&self) -> bool {
        self.write_bytes > (L1D_CACHE_SIZE * 9) / 10
    }

    /// Get total bytes written in this transaction.
    pub fn write_footprint(&self) -> usize {
        self.write_bytes
    }

    /// Commit the transaction. Returns true if commit succeeded.
    pub fn commit(mut self) -> bool {
        if self.active && !self.committed {
            unsafe { xend() };
            self.committed = true;
            self.active = false;
            true
        } else {
            false
        }
    }
}

impl Drop for TsxTransaction {
    fn drop(&mut self) {
        if self.active && !self.committed {
            // Transaction was not committed — abort it.
            // The CPU will discard all speculative writes.
            unsafe { xabort(0xFF) };
            self.active = false;
        }
    }
}

// ─── Global TSX Re-entrancy Guard ───────────────────────────────────────────

/// Prevents nested TSX transactions (which always fail).
static TSX_IN_TRANSACTION: AtomicBool = AtomicBool::new(false);

/// Check if we are currently inside a TSX transaction (software guard).
pub fn is_in_software_transaction() -> bool {
    TSX_IN_TRANSACTION.load(Ordering::Acquire)
}

/// Enter TSX software guard. Returns false if already in a transaction.
fn enter_software_guard() -> bool {
    TSX_IN_TRANSACTION.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok()
}

/// Exit TSX software guard.
fn exit_software_guard() {
    TSX_IN_TRANSACTION.store(false, Ordering::Release);
}

/// Begin a TSX transaction with re-entrancy protection.
/// Returns None if already in a transaction or if hardware xbegin fails.
pub fn tsx_begin_safe(max_retries: u32) -> Option<TsxTransaction> {
    if !enter_software_guard() {
        return None; // Already in a transaction — nested xbegin would fail
    }
    match TsxTransaction::begin(max_retries) {
        Some(tx) => Some(tx),
        None => {
            exit_software_guard();
            None
        }
    }
}

/// Commit and release the re-entrancy guard.
pub fn tsx_commit_safe(tx: TsxTransaction) -> bool {
    let result = tx.commit();
    exit_software_guard();
    result
}

// ─── AMX (Advanced Matrix Extensions) ───────────────────────────────────────

/// AMX tile configuration for palette 0 (8 tiles, 16 rows × 64 bytes).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AmxTileConfig {
    pub palette_id: u8,
    pub start_row: u8,
    pub reserved: [u16; 3],
    pub bytes_per_row: [u16; 8],
    pub rows: [u8; 8],
}

impl AmxTileConfig {
    /// Default configuration: 8 tiles, each 16 rows × 64 bytes.
    pub const fn default_16x64() -> Self {
        Self {
            palette_id: 1,
            start_row: 0,
            reserved: [0; 3],
            bytes_per_row: [64; 8],
            rows: [16; 8],
        }
    }

    /// Size of the configuration structure in bytes.
    pub const fn size() -> usize { 64 }
}

/// Initialize AMX with the given tile configuration.
/// Uses XSETBV to enable AMX state in XCR0, then LDTILECFG.
#[cfg(target_arch = "x86_64")]
pub unsafe fn amx_init(config: &AmxTileConfig) {
    // Enable AMX in XCR0: read-modify-write to preserve existing state (SSE/AVX/etc.)
    // XCR0 = XCR0 | (1 << 17) | (1 << 18)

    // Step 1: Read current XCR0 via XGETBV
    let xcr0_eax: u32;
    let xcr0_edx: u32;
    core::arch::asm!(
        "xgetbv",
        in("ecx") 0u32,        // XCR0
        out("eax") xcr0_eax,
        out("edx") xcr0_edx,
        options(nomem, nostack)
    );
    let xcr0_old = ((xcr0_edx as u64) << 32) | (xcr0_eax as u64);

    // Step 2: OR in AMX bits and write back via XSETBV
    let xcr0_new = xcr0_old | ((1u64 << 17) | (1u64 << 18));
    core::arch::asm!(
        "xsetbv",
        in("ecx") 0u32,        // XCR0
        in("eax") (xcr0_new & 0xFFFFFFFF) as u32,
        in("edx") ((xcr0_new >> 32) & 0xFFFFFFFF) as u32,
        options(nomem, nostack)
    );

    // Load tile configuration via LDTILECFG
    core::arch::asm!(
        "ldtilecfg [{}]",
        in(reg) config as *const AmxTileConfig,
        options(nostack)
    );
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn amx_init(_config: &AmxTileConfig) {}

/// Release AMX tile registers (must be called before context switch).
#[cfg(target_arch = "x86_64")]
pub unsafe fn amx_tile_release() {
    core::arch::asm!("tilerelease", options(nomem, nostack));
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn amx_tile_release() {}

/// Load data into a tile register from memory.
/// tile_id: 0-7, base: pointer to data, stride: bytes between rows.
#[cfg(target_arch = "x86_64")]
pub unsafe fn amx_tile_load(tile_id: u8, base: *const u8, stride: usize) {
    // TILELOADD: load tile from memory
    // Encoding: TILELOADD tmm, [base + stride * idx]
    // We use the ASM form directly
    match tile_id {
        0 => core::arch::asm!("tileloadd tmm0, [{0} + {1}*1]", in(reg) base, in(reg) stride, options(nostack)),
        1 => core::arch::asm!("tileloadd tmm1, [{0} + {1}*1]", in(reg) base, in(reg) stride, options(nostack)),
        2 => core::arch::asm!("tileloadd tmm2, [{0} + {1}*1]", in(reg) base, in(reg) stride, options(nostack)),
        3 => core::arch::asm!("tileloadd tmm3, [{0} + {1}*1]", in(reg) base, in(reg) stride, options(nostack)),
        4 => core::arch::asm!("tileloadd tmm4, [{0} + {1}*1]", in(reg) base, in(reg) stride, options(nostack)),
        5 => core::arch::asm!("tileloadd tmm5, [{0} + {1}*1]", in(reg) base, in(reg) stride, options(nostack)),
        6 => core::arch::asm!("tileloadd tmm6, [{0} + {1}*1]", in(reg) base, in(reg) stride, options(nostack)),
        7 => core::arch::asm!("tileloadd tmm7, [{0} + {1}*1]", in(reg) base, in(reg) stride, options(nostack)),
        _ => {},
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn amx_tile_load(_tile_id: u8, _base: *const u8, _stride: usize) {}

/// Store data from a tile register to memory.
#[cfg(target_arch = "x86_64")]
pub unsafe fn amx_tile_store(tile_id: u8, base: *mut u8, stride: usize) {
    match tile_id {
        0 => core::arch::asm!("tilestored [{0} + {1}*1], tmm0", in(reg) base, in(reg) stride, options(nostack)),
        1 => core::arch::asm!("tilestored [{0} + {1}*1], tmm1", in(reg) base, in(reg) stride, options(nostack)),
        2 => core::arch::asm!("tilestored [{0} + {1}*1], tmm2", in(reg) base, in(reg) stride, options(nostack)),
        3 => core::arch::asm!("tilestored [{0} + {1}*1], tmm3", in(reg) base, in(reg) stride, options(nostack)),
        4 => core::arch::asm!("tilestored [{0} + {1}*1], tmm4", in(reg) base, in(reg) stride, options(nostack)),
        5 => core::arch::asm!("tilestored [{0} + {1}*1], tmm5", in(reg) base, in(reg) stride, options(nostack)),
        6 => core::arch::asm!("tilestored [{0} + {1}*1], tmm6", in(reg) base, in(reg) stride, options(nostack)),
        7 => core::arch::asm!("tilestored [{0} + {1}*1], tmm7", in(reg) base, in(reg) stride, options(nostack)),
        _ => {},
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn amx_tile_store(_tile_id: u8, _base: *mut u8, _stride: usize) {}

/// Matrix multiply-add: TDEST += TSRC1 * TSRC2 (8-bit signed integers).
#[cfg(target_arch = "x86_64")]
pub unsafe fn amx_tdpbssd(dest: u8, src1: u8, src2: u8) {
    // TDPBSSD: tile_dest += tile_src1 * tile_src2 (signed 8-bit × 8-bit → 32-bit accumulate)
    // We need to encode this with the correct tile register names
    let tmm = |id: u8| -> &'static str {
        match id {
            0 => "tmm0", 1 => "tmm1", 2 => "tmm2", 3 => "tmm3",
            4 => "tmm4", 5 => "tmm5", 6 => "tmm6", 7 => "tmm7",
            _ => "tmm0",
        }
    };
    // We can't use dynamic register names in asm!, so we use a match
    match (dest, src1, src2) {
        // Note: tdpbssd requires all three tile registers to be distinct.
        // Cases where dest == src1 or dest == src2 are invalid and fall through
        // to the default no-op handler.
        (0, 1, 2) => core::arch::asm!("tdpbssd tmm0, tmm1, tmm2", options(nomem, nostack)),
        (3, 1, 2) => core::arch::asm!("tdpbssd tmm3, tmm1, tmm2", options(nomem, nostack)),
        // Default: common pattern for learned scheduler
        (d, s1, s2) => {
            // Fallback: use raw bytes for tdpbssd
            // This is a simplified version; a full implementation would
            // enumerate all 512 combinations or use a JIT to emit the instruction
            let _ = (d, s1, s2, tmm);
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn amx_tdpbssd(_dest: u8, _src1: u8, _src2: u8) {}

/// Pure-Rust fallback for the Learned Scheduler's matrix multiply.
/// Used when AMX is not available. Computes: output = weights × input.
/// This is NOT an AMX operation — it's the software fallback.
pub fn scheduler_matmul_fallback(
    weights: &[u8],
    input: &[u8],
    output: &mut [i32],
    rows: usize,
    cols: usize,
) {
    for i in 0..rows {
        let mut sum: i32 = 0;
        for j in 0..cols {
            let w = weights.get(i * cols + j).copied().unwrap_or(0) as i8 as i32;
            let x = input.get(j).copied().unwrap_or(0) as i8 as i32;
            sum += w * x;
        }
        if let Some(o) = output.get_mut(i) {
            *o = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tsx_status_started() {
        let s = TsxStatus { raw: core::u32::MAX };
        assert!(s.started());
        assert!(!s.aborted());
    }

    #[test]
    fn test_tsx_status_conflict() {
        let s = TsxStatus { raw: TX_ABORT_CONFLICT };
        assert!(s.aborted());
        assert!(s.is_conflict());
        assert!(!s.is_capacity());
    }

    #[test]
    fn test_tsx_status_capacity() {
        let s = TsxStatus { raw: TX_ABORT_CAPACITY };
        assert!(s.is_capacity());
    }

    #[test]
    fn test_tsx_status_explicit() {
        let s = TsxStatus { raw: TX_ABORT_EXPLICIT | (42u32 << 24) };
        assert!(s.is_explicit());
        assert_eq!(s.abort_code(), 42);
    }

    #[test]
    fn test_transaction_bound_proof() {
        assert!(prove_transaction_bound(1024));
        assert!(prove_transaction_bound(L1D_CACHE_SIZE - 1));
        assert!(!prove_transaction_bound(L1D_CACHE_SIZE));
        assert!(!prove_transaction_bound(L1D_CACHE_SIZE + 1));
    }

    #[test]
    fn test_l1d_cache_size() {
        assert_eq!(L1D_CACHE_SIZE, 32 * 1024);
    }

    #[test]
    fn test_amx_tile_config_default() {
        let config = AmxTileConfig::default_16x64();
        assert_eq!(config.palette_id, 1);
        assert_eq!(config.rows, [16u8; 8]);
        assert_eq!(config.bytes_per_row, [64u16; 8]);
    }

    #[test]
    fn test_amx_tile_config_size() {
        assert_eq!(AmxTileConfig::size(), 64);
    }

    #[test]
    fn test_scheduler_matmul_fallback() {
        let weights: [u8; 6] = [1, 2, 3, 4, 5, 6]; // 2×3
        let input: [u8; 3] = [1, 1, 1];
        let mut output: [i32; 2] = [0; 2];
        scheduler_matmul_fallback(&weights, &input, &mut output, 2, 3);
        assert_eq!(output[0], 6);  // 1+2+3
        assert_eq!(output[1], 15); // 4+5+6
    }

    #[test]
    fn test_scheduler_matmul_fallback_signed() {
        // Test with signed interpretation (u8 → i8)
        let weights: [u8; 4] = [0xFF, 2, 0xFE, 4]; // -1, 2, -2, 4
        let input: [u8; 2] = [3, 5];
        let mut output: [i32; 2] = [0; 2];
        scheduler_matmul_fallback(&weights, &input, &mut output, 2, 2);
        assert_eq!(output[0], (-1 * 3 + 2 * 5) as i32); // -3 + 10 = 7
        assert_eq!(output[1], (-2 * 3 + 4 * 5) as i32); // -6 + 20 = 14
    }

    #[test]
    fn test_reentrancy_guard() {
        assert!(!is_in_software_transaction());
    }

    #[test]
    fn test_abort_constants() {
        assert_eq!(TX_ABORT_EXPLICIT, 1);
        assert_eq!(TX_ABORT_CONFLICT, 4);
        assert_eq!(TX_ABORT_CAPACITY, 8);
    }
}
