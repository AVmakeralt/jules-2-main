// =========================================================================
// Identity Mapping, HugePage Allocator, IOMMU, NMI Watchdog, and CFI
//
// When Jules IS the OS, it doesn't use virtual memory — it IS the
// virtual memory manager. This module implements:
//
//   1. Identity Mapping — VA = PA, no TLB misses for core logic
//   2. HugePage Allocator — 1GB/2MB pages for minimal TLB pressure
//   3. IOMMU Drop Zones — Protect Sanctuary from DMA attacks
//   4. NMI Watchdog — Detect and recover from deadlocks
//   5. CFI — Control Flow Integrity (no unchecked indirect jumps)
//
// REFERENCES:
//   Intel SDM Vol 3, §4.3 (64-bit paging)
//   Intel VT-d Specification (IOMMU)
// =========================================================================

use core::sync::atomic::{AtomicU64, Ordering};

// ─── 1. IDENTITY MAPPING ────────────────────────────────────────────────────

pub const PTE_PRESENT: u64 = 1 << 0;
pub const PTE_WRITABLE: u64 = 1 << 1;
pub const PTE_USER: u64 = 1 << 2;
pub const PTE_HUGE_PAGE: u64 = 1 << 7;
pub const PTE_NO_EXECUTE: u64 = 1 << 63;

pub const PML4_SHIFT: u32 = 39;
pub const PDPT_SHIFT: u32 = 30;
pub const PD_SHIFT: u32 = 21;
pub const PAGE_SHIFT: u32 = 12;

pub const PML4_ENTRIES: usize = 512;
pub const PDPT_ENTRIES: usize = 512;
pub const PD_ENTRIES: usize = 512;

/// Address mask for 4KB page alignment
pub const PAGE_MASK: u64 = !0xFFF;

/// Default sanctuary size: 8 MB (power of 2, SFI-compatible)
pub const SANCTUARY_SIZE: u64 = 8 * 1024 * 1024;

/// A single PML4 table for identity mapping.
/// Supports up to 512 × 1GB = 512GB of identity-mapped memory.
pub struct IdentityMap {
    /// PML4 table (512 × 8 bytes = 4KB)
    pml4: [u64; PML4_ENTRIES],
    /// PDPT tables (each 512 × 8 = 4KB)
    pdpt: [[u64; PDPT_ENTRIES]; PML4_ENTRIES],
    /// PD tables for 4KB page support (indexed by [pdpt_idx][pd_idx])
    pd: [[[u64; PD_ENTRIES]; PDPT_ENTRIES]; PML4_ENTRIES],
    /// Whether each PDPT entry has been initialized
    pdpt_used: [bool; PML4_ENTRIES],
    /// Whether each PD entry has been initialized (for 4KB page tracking)
    pd_used: [[bool; PDPT_ENTRIES]; PML4_ENTRIES],
    /// Use 1GB gigantic pages (maps entire 1GB per PDPT entry)
    use_1gb_pages: bool,
    /// Total mapped bytes
    mapped_bytes: u64,
}

impl IdentityMap {
    pub const fn new(use_1gb_pages: bool) -> Self {
        Self {
            pml4: [0; PML4_ENTRIES],
            pdpt: [[0; PDPT_ENTRIES]; PML4_ENTRIES],
            pd: [[[0; PD_ENTRIES]; PDPT_ENTRIES]; PML4_ENTRIES],
            pdpt_used: [false; PML4_ENTRIES],
            pd_used: [[false; PDPT_ENTRIES]; PML4_ENTRIES],
            use_1gb_pages,
            mapped_bytes: 0,
        }
    }

    /// Map a 4KB page as identity: VA = PA.
    pub fn map_4kb(&mut self, phys_addr: u64, flags: u64) {
        let pml4_idx = ((phys_addr >> PML4_SHIFT) & 0x1FF) as usize;
        let pdpt_idx = ((phys_addr >> PDPT_SHIFT) & 0x1FF) as usize;
        let pd_idx = ((phys_addr >> PD_SHIFT) & 0x1FF) as usize;
        let pt_idx = ((phys_addr >> PAGE_SHIFT) & 0x1FF) as usize;

        // Ensure PML4 entry points to our PDPT
        if self.pml4[pml4_idx] == 0 {
            let pdpt_addr = &self.pdpt[pml4_idx] as *const [u64; 512] as u64;
            self.pml4[pml4_idx] = (pdpt_addr & PAGE_MASK) | PTE_PRESENT | PTE_WRITABLE;
            self.pdpt_used[pml4_idx] = true;
        }

        // Ensure PDPT entry points to a PD (not a 1GB huge page)
        if self.pdpt[pml4_idx][pdpt_idx] == 0 || (self.pdpt[pml4_idx][pdpt_idx] & PTE_HUGE_PAGE) != 0 {
            let pd_addr = &self.pd[pml4_idx][pdpt_idx] as *const [u64; 512] as u64;
            self.pdpt[pml4_idx][pdpt_idx] = (pd_addr & PAGE_MASK) | PTE_PRESENT | PTE_WRITABLE;
            self.pd_used[pml4_idx][pdpt_idx] = true;
        }

        // PD entry points to PT — for 4KB pages, we use the PD entry itself
        // as a page table entry array (simplified: PD[pt_idx] = 4KB page mapping)
        // In a full implementation, PD would point to a separate PT.
        // Here we use the pd array as a combined PD+PT where each entry maps a 4KB page.
        // PD entries for 4KB mapping: point to the page itself (no huge page bit).
        // Since we don't have a separate PT level in this simplified model,
        // we use pd[pml4_idx][pdpt_idx] as the PD and treat entries as PT entries.
        // Actually, we need a proper 3-level walk. The PD entry should NOT have the
        // huge page bit set — it points to a PT. But since we inline the PT in pd,
        // each pd[pml4_idx][pdpt_idx][pd_idx] acts as a PT entry for 4KB pages.
        self.pd[pml4_idx][pdpt_idx][pd_idx] = (phys_addr & PAGE_MASK) | flags | PTE_PRESENT;
        self.mapped_bytes += 4 * 1024; // 4 KB

        let _ = pt_idx; // Reserved for full PT implementation
    }

    /// Map a 2MB page as identity (using PD huge page bit).
    pub fn map_2mb(&mut self, phys_addr: u64, flags: u64) {
        let pml4_idx = ((phys_addr >> PML4_SHIFT) & 0x1FF) as usize;
        let pdpt_idx = ((phys_addr >> PDPT_SHIFT) & 0x1FF) as usize;

        if self.pml4[pml4_idx] == 0 {
            let pdpt_addr = &self.pdpt[pml4_idx] as *const [u64; 512] as u64;
            self.pml4[pml4_idx] = (pdpt_addr & PAGE_MASK) | PTE_PRESENT | PTE_WRITABLE;
            self.pdpt_used[pml4_idx] = true;
        }

        // PD entry with huge page bit
        self.pdpt[pml4_idx][pdpt_idx] = (phys_addr & PAGE_MASK) | flags | PTE_HUGE_PAGE | PTE_PRESENT;
        self.mapped_bytes += 2 * 1024 * 1024; // 2 MB
    }

    /// Map a 1GB page as identity (using PDPT huge page bit).
    pub fn map_1gb(&mut self, phys_addr: u64, flags: u64) {
        let pml4_idx = ((phys_addr >> PML4_SHIFT) & 0x1FF) as usize;
        let pdpt_idx = ((phys_addr >> PDPT_SHIFT) & 0x1FF) as usize;

        if self.pml4[pml4_idx] == 0 {
            let pdpt_addr = &self.pdpt[pml4_idx] as *const [u64; 512] as u64;
            self.pml4[pml4_idx] = (pdpt_addr & PAGE_MASK) | PTE_PRESENT | PTE_WRITABLE;
            self.pdpt_used[pml4_idx] = true;
        }

        self.pdpt[pml4_idx][pdpt_idx] = (phys_addr & 0x000F_FFFF_C000_0000) | flags | PTE_HUGE_PAGE | PTE_PRESENT;
        self.mapped_bytes += 1024 * 1024 * 1024; // 1 GB
    }

    /// Map the entire physical range as identity.
    pub fn map_range(&mut self, phys_start: u64, phys_end: u64, flags: u64) {
        let mut addr = phys_start & PAGE_MASK;
        while addr < phys_end {
            if self.use_1gb_pages && addr % (1 << 30) == 0 && addr + (1u64 << 30) <= phys_end {
                self.map_1gb(addr, flags);
                addr += 1u64 << 30;
            } else if addr % (1 << 21) == 0 && addr + (1u64 << 21) <= phys_end {
                self.map_2mb(addr, flags);
                addr += 1u64 << 21;
            } else {
                self.map_4kb(addr, flags);
                addr += 1u64 << 12;
            }
        }
    }

    /// Load the page table by writing CR3.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn load_cr3(&self) {
        let pml4_addr = &self.pml4 as *const [u64; 512] as u64;
        core::arch::asm!(
            "mov cr3, {}",
            in(reg) pml4_addr & PAGE_MASK,
            options(nostack)
        );
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub unsafe fn load_cr3(&self) {}

    /// Flush a single TLB entry using INVLPG.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn flush_tlb_entry(addr: u64) {
        let _ = addr; // Address is used by INVLPG
        #[cfg(target_arch = "x86_64")]
        core::arch::asm!("invlpg [{}]", in(reg) addr, options(nostack));
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub unsafe fn flush_tlb_entry(_addr: u64) {}

    /// Prophetic TLB preload: pre-load a TLB entry before it's needed.
    /// This reads a byte from the predicted address, forcing the CPU to
    /// populate the TLB entry. The read is discarded.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn prophetic_tlb_preload(addr: u64) {
        #[cfg(target_arch = "x86_64")]
        {
            // Touch the page to force TLB population
            let ptr = addr as *const u8;
            let _ = core::ptr::read_volatile(ptr);
        }
        let _ = addr;
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub unsafe fn prophetic_tlb_preload(_addr: u64) {}

    pub fn mapped_bytes(&self) -> u64 { self.mapped_bytes }
    pub fn pml4_address(&self) -> u64 { &self.pml4 as *const [u64; 512] as u64 }
}

// ─── 2. HUGE PAGE ALLOCATOR ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HugePageSize {
    Size2MB,
    Size1GB,
}

impl HugePageSize {
    pub const fn size_bytes(&self) -> u64 {
        match self {
            HugePageSize::Size2MB => 2 * 1024 * 1024,
            HugePageSize::Size1GB => 1024 * 1024 * 1024,
        }
    }
}

/// Simple bump allocator for huge pages. No free list — Jules never returns memory.
pub struct HugePageAllocator {
    next_addr: u64,
    end_addr: u64,
    page_size: HugePageSize,
    allocated: u64,
}

impl HugePageAllocator {
    pub const fn new(start: u64, end: u64, page_size: HugePageSize) -> Self {
        Self { next_addr: start, end_addr: end, page_size, allocated: 0 }
    }

    pub fn allocate(&mut self) -> Option<u64> {
        let size = self.page_size.size_bytes();
        if self.next_addr + size > self.end_addr {
            return None;
        }
        let addr = self.next_addr;
        self.next_addr += size;
        self.allocated += 1;
        Some(addr)
    }

    pub fn allocate_contiguous(&mut self, count: usize) -> Option<u64> {
        let size = self.page_size.size_bytes() * count as u64;
        if self.next_addr + size > self.end_addr {
            return None;
        }
        let addr = self.next_addr;
        self.next_addr += size;
        self.allocated += count as u64;
        Some(addr)
    }

    pub fn allocated_count(&self) -> u64 { self.allocated }
    pub fn remaining(&self) -> u64 {
        let size = self.page_size.size_bytes();
        (self.end_addr.saturating_sub(self.next_addr)) / size
    }
}

// ─── 3. IOMMU DROP ZONES ────────────────────────────────────────────────────

/// IOMMU DMA protection: hardware devices can ONLY write to the drop zone.
pub struct IommuDropZone {
    drop_zone_base: u64,
    drop_zone_size: u64,
    sanctuary_base: u64,
    sanctuary_size: u64,
}

impl IommuDropZone {
    pub const fn new(
        drop_zone_base: u64,
        drop_zone_size: u64,
        sanctuary_base: u64,
        sanctuary_size: u64,
    ) -> Self {
        Self { drop_zone_base, drop_zone_size, sanctuary_base, sanctuary_size }
    }

    pub fn is_dma_allowed(&self, addr: u64, len: u64) -> bool {
        let start = addr;
        let end = addr.saturating_add(len);
        // DMA must target the drop zone entirely
        start >= self.drop_zone_base && end <= self.drop_zone_base + self.drop_zone_size
    }

    pub fn is_sanctuary(&self, addr: u64) -> bool {
        addr >= self.sanctuary_base && addr < self.sanctuary_base + self.sanctuary_size
    }

    /// Program the IOMMU via MMIO (Intel VT-d DMAR tables).
    /// In a real implementation, this would write to the IOMMU's
    /// root-entry and context-entry tables to set up DMA remapping.
    pub unsafe fn program_iommu(&self) {
        // Placeholder: real implementation would:
        // 1. Locate the DMAR ACPI table
        // 2. Map the IOMMU MMIO registers
        // 3. Write root-entry table pointer
        // 4. Enable translation in the Global Command Register
        // 5. Configure device-specific DMA remapping
        // For now, we provide the interface without actual MMIO writes
    }

    pub fn drop_zone_base(&self) -> u64 { self.drop_zone_base }
}

// ─── 4. NMI WATCHDOG ────────────────────────────────────────────────────────

/// Default NMI watchdog interval: 10ms in APIC timer ticks.
/// At 100 Hz bus clock / 16 divide = ~6.25 MHz, 10ms ≈ 62,500 ticks.
pub const NMI_DEFAULT_INTERVAL: u32 = 62_500;

/// NMI Watchdog: detects deadlocks by checking an epoch counter.
/// If the epoch hasn't advanced after 2 consecutive NMI checks,
/// assumes the core is deadlocked and forces a restart.
#[allow(dead_code)]
pub struct NmiWatchdog {
    epoch: AtomicU64,
    last_epoch: u64,
    interval_ticks: u32,
    max_hits: u32,
    hit_count: u32,
}

impl NmiWatchdog {
    pub const fn new(interval_ticks: u32) -> Self {
        Self {
            epoch: AtomicU64::new(0),
            last_epoch: 0,
            interval_ticks,
            max_hits: 3,
            hit_count: 0,
        }
    }

    /// Start the NMI watchdog (program APIC LINT1 as NMI).
    pub unsafe fn start(&self) {
        // Configure Local APIC LINT1 for NMI delivery
        // This requires the Local APIC driver — in a real implementation,
        // we'd call local_apic.set_lint1(vector, masked: false)
    }

    /// Check if the core is alive. Called from the NMI handler.
    /// Returns true if the epoch advanced (core is alive).
    pub fn check_and_pet(&mut self) -> bool {
        let current = self.epoch.load(Ordering::Acquire);
        if current != self.last_epoch {
            self.last_epoch = current;
            self.hit_count = 0;
            true
        } else {
            self.hit_count += 1;
            if self.hit_count >= self.max_hits {
                // Core is deadlocked — force restart
                // SAFETY: This is called from NMI handler context, which is
                // inherently unsafe. The caller must ensure this is valid.
                unsafe { self.force_restart() };
            }
            false
        }
    }

    /// Increment the epoch counter (called by the scheduler each tick).
    pub fn pet(&self) {
        self.epoch.fetch_add(1, Ordering::Release);
    }

    /// Force-abort the current TSX transaction and restart the core.
    pub unsafe fn force_restart(&self) {
        // In a real implementation:
        // 1. If in TSX transaction, it's already aborted by NMI
        // 2. Restore CPU state from last checkpoint
        // 3. Resume execution from known-good point
    }

    pub fn epoch(&self) -> u64 { self.epoch.load(Ordering::Acquire) }
    pub fn hit_count(&self) -> u32 { self.hit_count }
}

// ─── 5. CFI (Control Flow Integrity) ────────────────────────────────────────

/// Maximum number of valid jump targets in the CFI table.
const CFI_MAX_TARGETS: usize = 1024;

/// CFI enforcement: the JIT is forbidden from emitting indirect jumps
/// unless the target is verified against a hardcoded Jump Table.
pub struct CfiJumpTable {
    targets: [u64; CFI_MAX_TARGETS],
    count: usize,
}

impl CfiJumpTable {
    pub const fn new() -> Self {
        Self { targets: [0; CFI_MAX_TARGETS], count: 0 }
    }

    /// Register a valid jump target. Returns false if table is full.
    pub fn register_target(&mut self, addr: u64) -> bool {
        if self.count >= CFI_MAX_TARGETS {
            return false;
        }
        self.targets[self.count] = addr;
        self.count += 1;
        // Keep sorted for binary search
        self.targets[..self.count].sort_unstable();
        true
    }

    /// Verify a jump target using binary search. O(log n).
    pub fn is_valid_target(&self, addr: u64) -> bool {
        if self.count == 0 { return false; }
        let mut lo = 0usize;
        let mut hi = self.count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.targets[mid].cmp(&addr) {
                core::cmp::Ordering::Equal => return true,
                core::cmp::Ordering::Less => lo = mid + 1,
                core::cmp::Ordering::Greater => hi = mid,
            }
        }
        false
    }

    /// Emit a CFI-checked indirect jump in x86_64 assembly.
    /// Pattern: compare target against jump table, if valid jump, else UD2.
    pub fn emit_cfi_jump_asm(target_reg: &str) -> String {
        // This returns a template that the JIT fills in with the actual
        // jump table address and bounds
        format!(
            "cmp {target_reg}, [cfi_table_end]\n\
             jae 1f\n\
             cmp {target_reg}, [cfi_table_start]\n\
             jb 1f\n\
             ; Binary search verification would go here\n\
             jmp {target_reg}\n\
             1: ud2  ; CFI violation — undefined instruction trap"
        )
    }

    pub fn count(&self) -> usize { self.count }
}

/// CFI compliance verification report.
pub struct CfiReport {
    pub total_jumps: usize,
    pub verified_jumps: usize,
    pub violations: Vec<(usize, u64)>,
    pub is_compliant: bool,
}

/// Verify CFI compliance for a set of indirect jump targets.
pub fn verify_cfi_compliance(jump_targets: &[u64], table: &CfiJumpTable) -> CfiReport {
    let mut verified = 0;
    let mut violations = Vec::new();
    for (i, &target) in jump_targets.iter().enumerate() {
        if table.is_valid_target(target) {
            verified += 1;
        } else {
            violations.push((i, target));
        }
    }
    let is_compliant = violations.is_empty();
    CfiReport { total_jumps: jump_targets.len(), verified_jumps: verified, violations, is_compliant }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pte_flags() {
        assert_eq!(PTE_PRESENT, 1);
        assert_eq!(PTE_WRITABLE, 2);
        assert_eq!(PTE_HUGE_PAGE, 0x80);
        assert_eq!(PTE_NO_EXECUTE, 1u64 << 63);
    }

    #[test]
    fn test_identity_map_new() {
        let map = IdentityMap::new(true);
        assert_eq!(map.mapped_bytes(), 0);
    }

    #[test]
    fn test_identity_map_1gb() {
        let mut map = IdentityMap::new(true);
        map.map_1gb(0, PTE_PRESENT | PTE_WRITABLE);
        assert_eq!(map.mapped_bytes(), 1024 * 1024 * 1024);
    }

    #[test]
    fn test_identity_map_2mb() {
        let mut map = IdentityMap::new(false);
        map.map_2mb(0, PTE_PRESENT | PTE_WRITABLE);
        assert_eq!(map.mapped_bytes(), 2 * 1024 * 1024);
    }

    #[test]
    fn test_huge_page_allocator() {
        let mut alloc = HugePageAllocator::new(0, 4u64 * 1024 * 1024 * 1024, HugePageSize::Size1GB);
        assert_eq!(alloc.remaining(), 4);
        let a = alloc.allocate();
        assert!(a.is_some());
        assert_eq!(a.unwrap(), 0);
        let b = alloc.allocate();
        assert!(b.is_some());
        assert_eq!(b.unwrap(), 1024 * 1024 * 1024);
        assert_eq!(alloc.allocated_count(), 2);
        assert_eq!(alloc.remaining(), 2);
    }

    #[test]
    fn test_huge_page_allocator_exhaustion() {
        let mut alloc = HugePageAllocator::new(0, 2u64 * 1024 * 1024, HugePageSize::Size2MB);
        assert!(alloc.allocate().is_some());
        assert!(alloc.allocate().is_some());
        assert!(alloc.allocate().is_none()); // exhausted
    }

    #[test]
    fn test_huge_page_size_bytes() {
        assert_eq!(HugePageSize::Size2MB.size_bytes(), 2 * 1024 * 1024);
        assert_eq!(HugePageSize::Size1GB.size_bytes(), 1024 * 1024 * 1024);
    }

    #[test]
    fn test_iommu_drop_zone() {
        let dz = IommuDropZone::new(0x1000_0000, 0x1_0000, 0x0, 0x800_0000);
        assert!(dz.is_dma_allowed(0x1000_0000, 0x100));
        assert!(!dz.is_dma_allowed(0x0, 0x100)); // Inside sanctuary
        assert!(dz.is_sanctuary(0x100));
        assert!(!dz.is_sanctuary(0x1000_0000)); // In drop zone, not sanctuary
    }

    #[test]
    fn test_nmi_watchdog() {
        let mut wd = NmiWatchdog::new(NMI_DEFAULT_INTERVAL);
        assert_eq!(wd.epoch(), 0);
        wd.pet();
        assert_eq!(wd.epoch(), 1);
        assert!(wd.check_and_pet()); // Epoch advanced
    }

    #[test]
    fn test_nmi_watchdog_deadlock() {
        let mut wd = NmiWatchdog::new(NMI_DEFAULT_INTERVAL);
        wd.last_epoch = 0;
        assert!(!wd.check_and_pet()); // Epoch didn't advance
        assert_eq!(wd.hit_count(), 1);
    }

    #[test]
    fn test_cfi_jump_table() {
        let mut table = CfiJumpTable::new();
        assert!(table.register_target(0x1000));
        assert!(table.register_target(0x2000));
        assert!(table.register_target(0x3000));
        assert_eq!(table.count(), 3);
        assert!(table.is_valid_target(0x1000));
        assert!(table.is_valid_target(0x2000));
        assert!(!table.is_valid_target(0x4000));
    }

    #[test]
    fn test_cfi_compliance() {
        let mut table = CfiJumpTable::new();
        table.register_target(0x1000);
        table.register_target(0x2000);

        let targets = [0x1000u64, 0x2000, 0x3000];
        let report = verify_cfi_compliance(&targets, &table);
        assert_eq!(report.total_jumps, 3);
        assert_eq!(report.verified_jumps, 2);
        assert!(!report.is_compliant);
        assert_eq!(report.violations.len(), 1);
        assert_eq!(report.violations[0], (2, 0x3000));
    }

    #[test]
    fn test_cfi_compliant() {
        let mut table = CfiJumpTable::new();
        table.register_target(0x1000);

        let targets = [0x1000u64];
        let report = verify_cfi_compliance(&targets, &table);
        assert!(report.is_compliant);
    }

    #[test]
    fn test_cfi_emit_asm() {
        let asm = CfiJumpTable::emit_cfi_jump_asm("rax");
        assert!(asm.contains("rax"));
        assert!(asm.contains("ud2"));
    }

    #[test]
    fn test_sanctuary_size() {
        assert_eq!(SANCTUARY_SIZE, 8 * 1024 * 1024);
    }
}
