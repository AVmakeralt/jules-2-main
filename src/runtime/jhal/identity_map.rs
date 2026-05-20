// =========================================================================
// Identity Mapping, HugePage Allocator, IOMMU, and CFI
//
// When Jules IS the OS, it doesn't use virtual memory — it IS the
// virtual memory manager. This module implements:
//
//   1. Identity Mapping — VA = PA, no TLB misses for core logic
//   2. HugePage Allocator — 1GB/2MB pages for minimal TLB pressure
//   3. IOMMU Drop Zones — Protect Sanctuary from DMA attacks
//   4. CFI — Control Flow Integrity (no unchecked indirect jumps)
//
// NOTE: The NMI watchdog has been removed. Infinite loop / deadlock
// detection is now handled by the EntropyWatchdog in formal_verify.rs,
// which monitors state mutation entropy at loop backedges instead of
// relying on a hardware NMI timer.
//
// REFERENCES:
//   Intel SDM Vol 3, §4.3 (64-bit paging)
//   Intel VT-d Specification (IOMMU)
// =========================================================================

// Zero-heap design: no HashMap, Vec, Box, Arc, or std::collections.
// All data structures use fixed-size arrays with linear scan.

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

/// Maximum number of on-demand PD tables. Each PD table covers 1GB of
/// address space when using 2MB pages, or 1GB using 512 × 2MB entries.
/// 16 PD tables covers 16 GB, sufficient for the Jules bare-metal runtime.
const MAX_PD_TABLES: usize = 16;

/// C8 fix: Maximum number of on-demand PT (Page Table) tables for 4KB mapping.
/// Each PT table covers 2MB of address space (512 × 4KB entries).
/// 64 PT tables covers 128 MB of 4KB-mapped address space.
const MAX_PT_TABLES: usize = 64;

/// A single PML4 table for identity mapping.
/// Supports up to 512 × 1GB = 512GB of identity-mapped memory.
/// Zero-heap: all data is in fixed-size inline arrays.
pub struct IdentityMap {
    /// PML4 table (512 × 8 bytes = 4KB)
    pml4: [u64; PML4_ENTRIES],
    /// PDPT tables (each 512 × 8 = 4KB)
    pdpt: [[u64; PDPT_ENTRIES]; PML4_ENTRIES],
    /// PD tables for 4KB/2MB page support, allocated on demand.
    /// Each entry stores (pml4_idx, pdpt_idx) and the 512-entry page table.
    /// Linear scan lookup — acceptable for ≤16 entries.
    pd_keys: [(usize, usize); MAX_PD_TABLES],
    pd_tables: [[u64; PD_ENTRIES]; MAX_PD_TABLES],
    pd_count: usize,
    /// C8 fix: PT (Page Table) tables for 4KB page mapping, allocated on demand.
    /// Each PT table covers 2MB. Key is (pml4_idx, pdpt_idx, pd_idx).
    pt_keys: [(usize, usize, usize); MAX_PT_TABLES],
    pt_tables: [[u64; 512]; MAX_PT_TABLES],
    pt_count: usize,
    /// Whether each PDPT entry has been initialized
    pdpt_used: [bool; PML4_ENTRIES],
    /// Use 1GB gigantic pages (maps entire 1GB per PDPT entry)
    use_1gb_pages: bool,
    /// Total mapped bytes
    mapped_bytes: u64,
}

impl IdentityMap {
    pub fn new(use_1gb_pages: bool) -> Self {
        Self {
            pml4: [0; PML4_ENTRIES],
            pdpt: [[0; PDPT_ENTRIES]; PML4_ENTRIES],
            pd_keys: [(0, 0); MAX_PD_TABLES],
            pd_tables: [[0; PD_ENTRIES]; MAX_PD_TABLES],
            pd_count: 0,
            pt_keys: [(0, 0, 0); MAX_PT_TABLES],
            pt_tables: [[0; 512]; MAX_PT_TABLES],
            pt_count: 0,
            pdpt_used: [false; PML4_ENTRIES],
            use_1gb_pages,
            mapped_bytes: 0,
        }
    }

    /// Look up a PD table by (pml4_idx, pdpt_idx). Returns None if not found.
    fn _get_pd(&self, key: (usize, usize)) -> Option<&[u64; PD_ENTRIES]> {
        for i in 0..self.pd_count {
            if self.pd_keys[i] == key {
                return Some(&self.pd_tables[i]);
            }
        }
        None
    }

    /// Look up a PD table by (pml4_idx, pdpt_idx) mutably. Returns None if not found.
    fn get_pd_mut(&mut self, key: (usize, usize)) -> Option<&mut [u64; PD_ENTRIES]> {
        for i in 0..self.pd_count {
            if self.pd_keys[i] == key {
                return Some(&mut self.pd_tables[i]);
            }
        }
        None
    }

    /// Get or create a PD table for the given key. Returns the table index.
    /// Returns None if MAX_PD_TABLES is exhausted.
    fn get_or_create_pd(&mut self, key: (usize, usize)) -> Option<usize> {
        // Check if it already exists
        for i in 0..self.pd_count {
            if self.pd_keys[i] == key {
                return Some(i);
            }
        }
        // Allocate a new one
        if self.pd_count >= MAX_PD_TABLES {
            return None;
        }
        let idx = self.pd_count;
        self.pd_keys[idx] = key;
        self.pd_tables[idx] = [0; PD_ENTRIES];
        self.pd_count += 1;
        Some(idx)
    }

    /// Map a 4KB page as identity: VA = PA.
    ///
    /// C8 fix: Now properly uses a 4-level page table (PML4 → PDPT → PD → PT).
    /// The old code computed pt_idx but never used it, writing the physical
    /// address directly into the PD entry as if it were a 2MB page. Now we
    /// allocate PT (Page Table) structures on demand, write the 4KB PTE into
    /// the PT, and set the PD entry to point to the PT (without the HUGE_PAGE bit).
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
            let pd_table_idx = self.get_or_create_pd((pml4_idx, pdpt_idx))
                .expect("PD table limit exhausted — increase MAX_PD_TABLES");
            let pd_addr = self.pd_tables[pd_table_idx].as_ptr() as u64;
            self.pdpt[pml4_idx][pdpt_idx] = (pd_addr & PAGE_MASK) | PTE_PRESENT | PTE_WRITABLE;
        }

        // C8 fix: PD entry must point to a PT (Page Table), NOT directly to the phys addr.
        // The PD entry for 4KB mapping points to a PT; the HUGE_PAGE bit must NOT be set.
        // We need to get/create the PT first, then write the PD entry, to avoid
        // holding a mutable borrow on pd_table while creating the PT.
        let pt_key = (pml4_idx, pdpt_idx, pd_idx);

        // First, ensure the PT table exists and get its address
        let pd_table = self.get_pd_mut((pml4_idx, pdpt_idx))
            .expect("pd table must exist after initialization");
        let pd_entry = pd_table[pd_idx];

        let pt_addr = if pd_entry == 0 || (pd_entry & PTE_HUGE_PAGE) != 0 {
            // Need to allocate a PT for this 2MB region
            // Compute the PT address first before borrowing
            let pt_table_idx = self.get_or_create_pt(pt_key)
                .expect("PT table limit exhausted — increase MAX_PT_TABLES");
            self.pt_tables[pt_table_idx].as_ptr() as u64
        } else {
            // PD entry already points to a PT; extract the address
            pd_entry & PAGE_MASK
        };

        // Now write the PD entry to point to the PT (without HUGE_PAGE bit)
        {
            let pd_table = self.get_pd_mut((pml4_idx, pdpt_idx))
                .expect("pd table must exist after initialization");
            let pd_entry = pd_table[pd_idx];
            if pd_entry == 0 || (pd_entry & PTE_HUGE_PAGE) != 0 {
                pd_table[pd_idx] = (pt_addr & PAGE_MASK) | PTE_PRESENT | PTE_WRITABLE;
            }
        }

        // Write the 4KB PTE into the PT
        let pt_table = self.get_pt_mut(pt_key)
            .expect("pt table must exist after initialization");
        pt_table[pt_idx] = (phys_addr & PAGE_MASK) | flags | PTE_PRESENT;
        self.mapped_bytes += 4 * 1024; // 4 KB
    }

    /// Get or create a PT table for the given key. Returns the table index.
    /// Returns None if MAX_PT_TABLES is exhausted.
    fn get_or_create_pt(&mut self, key: (usize, usize, usize)) -> Option<usize> {
        // Check if it already exists
        for i in 0..self.pt_count {
            if self.pt_keys[i] == key {
                return Some(i);
            }
        }
        // Allocate a new one
        if self.pt_count >= MAX_PT_TABLES {
            return None;
        }
        let idx = self.pt_count;
        self.pt_keys[idx] = key;
        self.pt_tables[idx] = [0; 512];
        self.pt_count += 1;
        Some(idx)
    }

    /// Look up a PT table by key mutably. Returns None if not found.
    fn get_pt_mut(&mut self, key: (usize, usize, usize)) -> Option<&mut [u64; 512]> {
        for i in 0..self.pt_count {
            if self.pt_keys[i] == key {
                return Some(&mut self.pt_tables[i]);
            }
        }
        None
    }

    /// Map a 2MB page as identity (using PD huge page bit).
    pub fn map_2mb(&mut self, phys_addr: u64, flags: u64) {
        let pml4_idx = ((phys_addr >> PML4_SHIFT) & 0x1FF) as usize;
        let pdpt_idx = ((phys_addr >> PDPT_SHIFT) & 0x1FF) as usize;
        let pd_idx = ((phys_addr >> PD_SHIFT) & 0x1FF) as usize;

        if self.pml4[pml4_idx] == 0 {
            let pdpt_addr = &self.pdpt[pml4_idx] as *const [u64; 512] as u64;
            self.pml4[pml4_idx] = (pdpt_addr & PAGE_MASK) | PTE_PRESENT | PTE_WRITABLE;
            self.pdpt_used[pml4_idx] = true;
        }

        // PDPT entry must point to a PD (no huge page bit at this level)
        if self.pdpt[pml4_idx][pdpt_idx] == 0 || (self.pdpt[pml4_idx][pdpt_idx] & PTE_HUGE_PAGE) != 0 {
            // Allocate a PD table on demand (zero-heap: fixed array)
            let pd_idx = self.get_or_create_pd((pml4_idx, pdpt_idx))
                .expect("PD table limit exhausted — increase MAX_PD_TABLES");
            let pd_addr = self.pd_tables[pd_idx].as_ptr() as u64;
            self.pdpt[pml4_idx][pdpt_idx] = (pd_addr & PAGE_MASK) | PTE_PRESENT | PTE_WRITABLE;
        }

        // PD entry with huge page bit (2MB page)
        if let Some(pd_table) = self.get_pd_mut((pml4_idx, pdpt_idx)) {
            pd_table[pd_idx] = (phys_addr & !0x1F_FFFF) | flags | PTE_HUGE_PAGE | PTE_PRESENT;
        }
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
        if self.next_addr > self.end_addr {
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

// ─── 4. CFI (Control Flow Integrity) ────────────────────────────────────────

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

/// Maximum CFI violations that can be recorded.
const CFI_MAX_VIOLATIONS: usize = 256;

/// CFI compliance verification report.
/// Zero-heap: uses fixed-size arrays instead of Vec.
pub struct CfiReport {
    pub total_jumps: usize,
    pub verified_jumps: usize,
    pub violations: [(usize, u64); CFI_MAX_VIOLATIONS],
    pub violation_count: usize,
    pub is_compliant: bool,
}

/// Verify CFI compliance for a set of indirect jump targets.
pub fn verify_cfi_compliance(jump_targets: &[u64], table: &CfiJumpTable) -> CfiReport {
    let mut verified = 0;
    let mut violations = [(0usize, 0u64); CFI_MAX_VIOLATIONS];
    let mut violation_count = 0;
    for (i, &target) in jump_targets.iter().enumerate() {
        if table.is_valid_target(target) {
            verified += 1;
        } else if violation_count < CFI_MAX_VIOLATIONS {
            violations[violation_count] = (i, target);
            violation_count += 1;
        }
    }
    let is_compliant = violation_count == 0;
    CfiReport { total_jumps: jump_targets.len(), verified_jumps: verified, violations, violation_count, is_compliant }
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
        // IdentityMap is ~2MB due to inline PDPT arrays; use larger stack
        let child = std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let map = IdentityMap::new(true);
                assert_eq!(map.mapped_bytes(), 0);
            })
            .expect("thread spawn failed");
        child.join().expect("test thread panicked");
    }

    #[test]
    fn test_identity_map_1gb() {
        // IdentityMap is ~2MB due to inline PDPT arrays; use spawn_with_stack
        // to get enough stack space for the test
        let child = std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let mut map = IdentityMap::new(true);
                map.map_1gb(0, PTE_PRESENT | PTE_WRITABLE);
                assert_eq!(map.mapped_bytes(), 1024 * 1024 * 1024);
            })
            .expect("thread spawn failed");
        child.join().expect("test thread panicked");
    }

    #[test]
    fn test_identity_map_2mb() {
        // IdentityMap is ~2MB due to inline PDPT arrays; use larger stack
        let child = std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let mut map = IdentityMap::new(false);
                map.map_2mb(0, PTE_PRESENT | PTE_WRITABLE);
                assert_eq!(map.mapped_bytes(), 2 * 1024 * 1024);
            })
            .expect("thread spawn failed");
        child.join().expect("test thread panicked");
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
        assert_eq!(report.violation_count, 1);
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
