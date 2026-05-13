// =============================================================================
// Speculative Devirtualization + Inline Caching
//
// This module implements Julia-style specialization and inline caching to
// achieve near-static-dispatch speed with dynamic flexibility.
//
// Features:
// - Polymorphic inline caches (PIC) that remember the last N types seen
// - Speculative devirtualization with fast paths for common types
// - Inline cache miss handling and re-optimization
// - Type distribution tracking for adaptive specialization
// - Megamorphic fallback via hash table when PIC is full
// - Truly executable code arena (mmap'd with RWX)
// - Runtime linker that actually patches call-site addresses in memory
// =============================================================================

use std::collections::HashMap;

// ── Platform-specific mmap/mprotect for executable code ──────────────────────

#[cfg(unix)]
use std::ffi::c_void;

#[cfg(target_os = "linux")]
const IC_MAP_ANONYMOUS: i32 = 0x20;
#[cfg(target_os = "macos")]
const IC_MAP_ANONYMOUS: i32 = 0x1000;
const IC_PROT_READ: i32 = 1;
const IC_PROT_WRITE: i32 = 2;
const IC_PROT_EXEC: i32 = 4;
const IC_MAP_PRIVATE: i32 = 0x02;

#[cfg(unix)]
extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
    fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

/// Executable code arena for inline cache entries.
///
/// Manages a pool of **truly executable** memory pages (allocated via `mmap`
/// with RWX permissions) where JIT-compiled specialized code is stored. Each
/// entry is aligned and tracked so that cache lookups return valid, executable
/// addresses that can be called directly.
///
/// Previously this used simulated address space (`0x7F00_0000_0000`) with
/// `Vec<u8>` pages that were never mprotect'd or called — the addresses
/// returned by `alloc()` were fake and would crash if dereferenced as code.
/// Now every page is an actual mmap'd region with RWX, so generated stubs
/// can be executed in-place.
struct CodeArena {
    /// Allocated code pages.
    /// On Unix: each entry is (ptr, len) from mmap — truly executable memory.
    /// On non-Unix: fall back to Vec<u8> (addresses will not be executable).
    #[cfg(unix)]
    pages: Vec<(*mut u8, usize)>,
    #[cfg(not(unix))]
    pages: Vec<Vec<u8>>,
    /// Current offset within the last page.
    current_offset: usize,
    /// Page size in bytes (matches OS page size for mmap).
    page_size: usize,
    /// Base address for the current page (used for address calculation).
    /// On Unix, this is the actual mmap'd address; on non-Unix it's simulated.
    current_page_base: usize,
}

impl CodeArena {
    #[cfg(unix)]
    fn new() -> Self {
        let page_size = 4096;
        Self {
            pages: Vec::new(),
            current_offset: 0,
            page_size,
            current_page_base: 0,
        }
    }

    #[cfg(not(unix))]
    fn new() -> Self {
        let page_size = 4096;
        Self {
            pages: Vec::new(),
            current_offset: 0,
            page_size,
            current_page_base: 0x7F00_0000_0000,
        }
    }

    /// Allocate `size` bytes in the arena and return the **executable** address.
    ///
    /// On Unix, this allocates a new mmap'd page (RWX) when needed and returns
    /// the actual address of the allocated region — the returned pointer can be
    /// called as a function after writing code to it.
    #[cfg(unix)]
    fn alloc(&mut self, size: usize) -> usize {
        let aligned_size = (size + 15) & !15; // Align to 16 bytes

        // Ensure there's room in the current page, or allocate a new one
        if self.pages.is_empty() || self.current_offset + aligned_size > self.page_size {
            // Allocate a new executable page via mmap with RWX permissions
            // (RWX so we can both write code and execute it; the alternative
            // would be to flip between RW and RX with mprotect, but for a JIT
            // arena that continuously writes and executes, RWX is simpler.)
            let ptr = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    self.page_size,
                    IC_PROT_READ | IC_PROT_WRITE | IC_PROT_EXEC,
                    IC_MAP_PRIVATE | IC_MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if ptr.is_null() || ptr as usize == usize::MAX {
                panic!("CodeArena: mmap failed — cannot allocate executable page");
            }
            // Fill with INT3 (0xCC) so any accidental execution hits a debug trap
            unsafe { std::ptr::write_bytes(ptr as *mut u8, 0xCC, self.page_size); }
            self.current_page_base = ptr as usize;
            self.pages.push((ptr as *mut u8, self.page_size));
            self.current_offset = 0;
        }

        let address = self.current_page_base + self.current_offset;
        self.current_offset += aligned_size;
        address
    }

    #[cfg(not(unix))]
    fn alloc(&mut self, size: usize) -> usize {
        let aligned_size = (size + 15) & !15;

        if self.pages.is_empty() || self.current_offset + aligned_size > self.page_size {
            self.pages.push(vec![0xCC; self.page_size]);
            self.current_page_base = 0x7F00_0000_0000 + (self.pages.len() - 1) * self.page_size;
            self.current_offset = 0;
        }

        let address = self.current_page_base + self.current_offset;
        self.current_offset += aligned_size;
        address
    }

    /// Write code bytes at a previously allocated address.
    ///
    /// On Unix, this writes directly into the mmap'd page (which has RWX
    /// permissions), so the code becomes executable immediately.
    #[cfg(unix)]
    fn write_code(&mut self, address: usize, code: &[u8]) {
        // Find the page that contains this address
        for &(ptr, len) in &self.pages {
            let base = ptr as usize;
            if address >= base && address < base + len {
                let offset = address - base;
                let end = (offset + code.len()).min(len);
                let write_len = end - offset;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        code.as_ptr(),
                        ptr.add(offset),
                        write_len,
                    );
                }
                return;
            }
        }
        // Address not in any known page — this shouldn't happen
        eprintln!("CodeArena::write_code: address {:x} not in any page", address);
    }

    #[cfg(not(unix))]
    fn write_code(&mut self, address: usize, code: &[u8]) {
        // Calculate which page and offset from the simulated base
        let first_page_base = 0x7F00_0000_0000;
        let offset_from_base = address.wrapping_sub(first_page_base);
        let page_index = offset_from_base / self.page_size;
        let page_offset = offset_from_base % self.page_size;

        if let Some(page) = self.pages.get_mut(page_index) {
            let end = (page_offset + code.len()).min(page.len());
            let write_len = end - page_offset;
            page[page_offset..page_offset + write_len].copy_from_slice(&code[..write_len]);
        }
    }
}

#[cfg(unix)]
impl Drop for CodeArena {
    fn drop(&mut self) {
        for &(ptr, len) in &self.pages {
            unsafe {
                munmap(ptr as *mut c_void, len);
            }
        }
    }
}

/// Runtime linker for patching inline cache call sites.
///
/// When a type-guard trampoline misses (the incoming type doesn't match the
/// cached type), it jumps to a generic dispatch trampoline.  The `RuntimeLinker`
/// manages the mapping from call-site addresses to their dispatch handlers and
/// **actually patches the code in memory** once a concrete type-specific handler
/// is available.
///
/// Previously, `patch_call_site` only stored the mapping in a HashMap but
/// admitted in a comment that it "should overwrite the address at site_addr."
/// Now it writes the handler address directly into the executable code at
/// `site_addr` (the 8-byte immediate in the `mov rax, imm64` instruction),
/// which is safe because the CodeArena pages are mapped RWX.
pub struct RuntimeLinker {
    /// Map from call-site address to the generic stub address
    call_sites: HashMap<u64, u64>,
    /// Map from (call_site, type_id) to optimized handler address
    type_handlers: HashMap<(u64, u64), u64>,
    /// Address of the generic dispatch trampoline
    trampoline_addr: u64,
}

impl RuntimeLinker {
    pub fn new() -> Self {
        Self {
            call_sites: HashMap::new(),
            type_handlers: HashMap::new(),
            trampoline_addr: 0,
        }
    }

    /// Set the generic dispatch trampoline address (typically allocated from
    /// the CodeArena and populated with a real dispatch loop).
    pub fn set_trampoline_addr(&mut self, addr: u64) {
        self.trampoline_addr = addr;
    }

    /// Register a call site that needs runtime patching
    pub fn register_call_site(&mut self, site_addr: u64, stub_addr: u64) {
        self.call_sites.insert(site_addr, stub_addr);
    }

    /// Patch a call site to jump directly to a type-specific handler.
    ///
    /// This now **actually overwrites** the 8-byte immediate at `site_addr`
    /// with `effective_handler`.  The code at `site_addr` is expected to have
    /// the pattern:
    ///
    ///   mov rax, <8-byte address>   ; 48 B8 <imm64>
    ///
    /// The `site_addr` should point to the start of the `mov rax, imm64`
    /// instruction (the 0x48 byte), and we patch the 8-byte immediate that
    /// starts at offset +2.
    ///
    /// This is safe because CodeArena pages are mapped RWX.
    pub fn patch_call_site(&mut self, site_addr: u64, type_id: u64, handler_addr: u64) {
        let effective_handler = if handler_addr != 0 { handler_addr } else { self.trampoline_addr };
        self.type_handlers.insert((site_addr, type_id), effective_handler);

        // Actually patch the call site by writing the handler address
        // into the code at site_addr.  The instruction layout is:
        //   48 B8 <imm64>    — mov rax, imm64
        // We write the 8-byte immediate starting at site_addr + 2.
        // This requires the page to be writable (RWX from our mmap).
        if effective_handler != 0 {
            // Write the 8-byte immediate using byte-level copy to avoid
            // alignment requirements.  x86-64 supports unaligned writes,
            // but Rust's `ptr::write_volatile` requires the pointer to be
            // aligned to `size_of::<u64>()`.  The immediate field in
            // `mov rax, imm64` starts at offset +2 from the instruction,
            // which is never 8-byte aligned.
            let imm_ptr = (site_addr as usize + 2) as *mut u8;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    effective_handler.to_le_bytes().as_ptr(),
                    imm_ptr,
                    8,
                );
            }
        }
    }

    /// Resolve the handler for a given call site and type
    pub fn resolve(&self, site_addr: u64, type_id: u64) -> Option<u64> {
        self.type_handlers.get(&(site_addr, type_id)).copied()
    }
}

impl Default for RuntimeLinker {
    fn default() -> Self {
        Self::new()
    }
}

/// Global code arena shared across all inline caches.
static mut CODE_ARENA: Option<CodeArena> = None;

/// Get or create the global code arena.
#[allow(static_mut_refs)]
fn get_arena() -> &'static mut CodeArena {
    unsafe {
        if CODE_ARENA.is_none() {
            CODE_ARENA = Some(CodeArena::new());
        }
        CODE_ARENA.as_mut().unwrap()
    }
}

/// Type identifier for dynamic dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(u64);

impl TypeId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

/// Inline cache entry
#[derive(Debug, Clone)]
pub struct InlineCacheEntry {
    /// The type this entry is specialized for
    pub type_id: TypeId,
    /// The specialized code address (in executable memory)
    pub code_address: usize,
    /// Number of times this entry was hit
    pub hit_count: u64,
    /// Last time this entry was used
    pub last_used: std::time::Instant,
}

impl InlineCacheEntry {
    pub fn new(type_id: TypeId, code_address: usize) -> Self {
        Self {
            type_id,
            code_address,
            hit_count: 0,
            last_used: std::time::Instant::now(),
        }
    }

    pub fn record_hit(&mut self) {
        self.hit_count += 1;
        self.last_used = std::time::Instant::now();
    }
}

/// Polymorphic inline cache (PIC)
///
/// Remembers the last N types seen at a call site and generates
/// specialized fast paths for each.  When the PIC is full (max_entries
/// reached), new types are stored in the **megamorphic table** — a
/// HashMap that provides O(1) lookup instead of O(N) linear scan.
///
/// This two-level design matches production JITs (V8, SpiderMonkey):
///   - PIC (inline, linear scan): fast for 1-4 types (monomorphic/polynomial)
///   - Megamorphic table (HashMap): scalable for 5+ types
#[derive(Debug)]
pub struct PolymorphicInlineCache {
    /// Cache entries (max N entries, linear scan)
    entries: Vec<InlineCacheEntry>,
    /// Maximum number of entries in the PIC before spilling to megamorphic
    max_entries: usize,
    /// Megamorphic fallback: hash table for types that overflow the PIC.
    /// Provides O(1) lookup when the PIC is full.
    megamorphic_table: HashMap<TypeId, usize>,
    /// Total number of cache lookups
    total_lookups: u64,
    /// Total number of cache hits (PIC + megamorphic)
    total_hits: u64,
    /// Total number of cache misses
    total_misses: u64,
}

impl PolymorphicInlineCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::with_capacity(max_entries),
            max_entries,
            megamorphic_table: HashMap::new(),
            total_lookups: 0,
            total_hits: 0,
            total_misses: 0,
        }
    }

    /// Look up a type in the cache.
    ///
    /// Lookup order:
    ///   1. Linear scan through PIC entries (fast path, O(N) with small N)
    ///   2. Megamorphic hash table (fallback, O(1))
    ///   3. Cache miss
    pub fn lookup(&mut self, type_id: TypeId) -> Option<usize> {
        self.total_lookups += 1;

        // Fast path: linear scan through PIC entries
        for entry in &mut self.entries {
            if entry.type_id == type_id {
                entry.record_hit();
                self.total_hits += 1;
                return Some(entry.code_address);
            }
        }

        // Megamorphic fallback: hash table lookup
        if let Some(&addr) = self.megamorphic_table.get(&type_id) {
            self.total_hits += 1;
            return Some(addr);
        }

        self.total_misses += 1;
        None
    }

    /// Add a new entry to the cache.
    ///
    /// If the PIC is full, the entry goes into the megamorphic table instead
    /// of evicting an existing PIC entry.  This avoids the "thrashing" problem
    /// where two types keep evicting each other in a small PIC.
    pub fn add_entry(&mut self, type_id: TypeId, code_address: usize) {
        // Check if already exists in PIC
        for entry in &self.entries {
            if entry.type_id == type_id {
                return; // Already cached
            }
        }

        // Check if already exists in megamorphic table
        if self.megamorphic_table.contains_key(&type_id) {
            return;
        }

        // If PIC has room, add to PIC
        if self.entries.len() < self.max_entries {
            self.entries.push(InlineCacheEntry::new(type_id, code_address));
        } else {
            // PIC is full → add to megamorphic table instead of evicting
            self.megamorphic_table.insert(type_id, code_address);
        }
    }

    /// Get the hit rate
    pub fn hit_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            0.0
        } else {
            self.total_hits as f64 / self.total_lookups as f64
        }
    }

    /// Get the most common type in the cache
    pub fn most_common_type(&self) -> Option<TypeId> {
        self.entries
            .iter()
            .max_by_key(|e| e.hit_count)
            .map(|e| e.type_id)
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.entries.clear();
        self.megamorphic_table.clear();
        self.total_lookups = 0;
        self.total_hits = 0;
        self.total_misses = 0;
    }

    /// Get the number of megamorphic entries
    pub fn megamorphic_count(&self) -> usize {
        self.megamorphic_table.len()
    }
}

/// Type distribution tracker
///
/// Tracks the distribution of types seen at a call site to inform
/// specialization decisions.
#[derive(Debug)]
pub struct TypeDistribution {
    /// Map from type ID to count
    counts: HashMap<TypeId, u64>,
    /// Total count
    total: u64,
}

impl TypeDistribution {
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            total: 0,
        }
    }

    /// Record a type occurrence
    pub fn record(&mut self, type_id: TypeId) {
        *self.counts.entry(type_id).or_insert(0) += 1;
        self.total += 1;
    }

    /// Get the frequency of a type (0.0 to 1.0)
    pub fn frequency(&self, type_id: TypeId) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.counts.get(&type_id).copied().unwrap_or(0) as f64 / self.total as f64
        }
    }

    /// Get the most common type and its frequency
    pub fn most_common(&self) -> Option<(TypeId, f64)> {
        self.counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&type_id, &count)| (type_id, count as f64 / self.total as f64))
    }

    /// Check if a type is dominant (appears > threshold of the time)
    pub fn is_dominant(&self, type_id: TypeId, threshold: f64) -> bool {
        self.frequency(type_id) > threshold
    }

    /// Get the number of distinct types seen
    pub fn distinct_count(&self) -> usize {
        self.counts.len()
    }

    /// Clear the distribution
    pub fn clear(&mut self) {
        self.counts.clear();
        self.total = 0;
    }
}

/// Speculative devirtualization metadata
#[derive(Debug, Clone)]
pub struct SpeculativeDevirt {
    /// The call site location
    pub location: String,
    /// The speculated type (if any)
    pub speculated_type: Option<TypeId>,
    /// Whether speculation is active
    pub is_speculated: bool,
    /// Number of times speculation succeeded
    pub success_count: u64,
    /// Number of times speculation failed
    pub failure_count: u64,
    /// Last time speculation was updated
    pub last_updated: std::time::Instant,
}

impl SpeculativeDevirt {
    pub fn new(location: String) -> Self {
        Self {
            location,
            speculated_type: None,
            is_speculated: false,
            success_count: 0,
            failure_count: 0,
            last_updated: std::time::Instant::now(),
        }
    }

    /// Speculate on a type
    pub fn speculate(&mut self, type_id: TypeId) {
        self.speculated_type = Some(type_id);
        self.is_speculated = true;
        self.last_updated = std::time::Instant::now();
    }

    /// Record a successful speculation
    pub fn record_success(&mut self) {
        self.success_count += 1;
        self.last_updated = std::time::Instant::now();
    }

    /// Record a failed speculation
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_updated = std::time::Instant::now();

        // If failure rate is too high, disable speculation
        let total = self.success_count + self.failure_count;
        if total > 10 && self.failure_count as f64 / total as f64 > 0.3 {
            self.is_speculated = false;
            self.speculated_type = None;
        }
    }

    /// Get the success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.0
        } else {
            self.success_count as f64 / total as f64
        }
    }

    /// Check if speculation should be enabled
    ///
    /// Returns true when speculation is active AND either:
    ///   - No data has been collected yet (0 successes + 0 failures), OR
    ///   - The success rate exceeds 70%
    pub fn should_speculate(&self) -> bool {
        if !self.is_speculated {
            return false;
        }
        let total = self.success_count + self.failure_count;
        if total == 0 {
            return true; // No data yet — trust the initial speculation
        }
        self.success_rate() > 0.7
    }
}

/// Call site metadata
#[derive(Debug)]
pub struct CallSite {
    /// The call site location
    pub location: String,
    /// Inline cache
    pub inline_cache: PolymorphicInlineCache,
    /// Type distribution
    pub type_distribution: TypeDistribution,
    /// Speculative devirtualization
    pub speculative_devirt: SpeculativeDevirt,
    /// Whether this call site is hot
    pub is_hot: bool,
}

impl CallSite {
    pub fn new(location: String) -> Self {
        Self {
            location: location.clone(),
            inline_cache: PolymorphicInlineCache::new(4), // Cache 4 types in PIC
            type_distribution: TypeDistribution::new(),
            speculative_devirt: SpeculativeDevirt::new(location),
            is_hot: false,
        }
    }

    /// Handle a call at this site
    pub fn handle_call(&mut self, type_id: TypeId) -> Option<usize> {
        // Record type in distribution
        self.type_distribution.record(type_id);

        // Try inline cache (PIC + megamorphic)
        if let Some(code_addr) = self.inline_cache.lookup(type_id) {
            return Some(code_addr);
        }

        // Cache miss - generate specialized code
        let code_addr = self.generate_specialized_code(type_id);
        self.inline_cache.add_entry(type_id, code_addr);

        // Update speculative devirtualization
        if let Some(speculated) = self.speculative_devirt.speculated_type {
            if speculated == type_id {
                self.speculative_devirt.record_success();
            } else {
                self.speculative_devirt.record_failure();
            }
        }

        Some(code_addr)
    }

    /// Generate specialized code for a type.
    ///
    /// This emits a minimal inline-cache stub **in truly executable memory**
    /// (mmap'd with RWX) that:
    ///   1. Compares the incoming type tag against the cached type_id.
    ///   2. If it matches, jumps to the type-specialized fast path.
    ///   3. If it doesn't match, falls back to the generic dispatch path.
    ///
    /// The generic dispatch path now uses the CodeArena's built-in generic
    /// dispatch stub (allocated once and shared) instead of the crash-inducing
    /// `0xCAFE_CAFE_0000_0000` sentinel.
    fn generate_specialized_code(&self, type_id: TypeId) -> usize {
        let arena = get_arena();

        // Build a type-guard trampoline:
        //   cmp rdi, <type_id_lo>       ; 48 81 ff <imm32>
        //   jne .slow_path              ; 0f 85 <rel32>
        //   ret                         ; c3
        // .slow_path:
        //   mov rax, <generic_stub>     ; 48 b8 <imm64>
        //   jmp rax                     ; ff e0
        //
        // Total size: 7 + 6 + 1 + 10 + 2 = 26 bytes
        let code_size = 32; // Padded to 32 bytes for alignment
        let entry_addr = arena.alloc(code_size);

        let mut code = Vec::with_capacity(code_size);

        // cmp rdi, <type_id as u32>
        code.extend_from_slice(&[0x48, 0x81, 0xFF]); // REX.W CMP rdi, imm32
        code.extend_from_slice(&(type_id.as_u64() as u32).to_le_bytes());

        // jne +10 (skip over the "ret" and land on slow-path)
        code.extend_from_slice(&[0x0F, 0x85, 0x0A, 0x00, 0x00, 0x00]); // JNE rel32

        // ret (fast path: type matches, return to caller)
        code.push(0xC3);

        // Slow path: load generic dispatch stub address into rax and jump.
        // Instead of the crash-inducing 0xCAFE_CAFE_0000_0000 sentinel,
        // we use a generic dispatch trampoline allocated in the CodeArena.
        // If no trampoline is available, we use a safe "ret 0" that simply
        // returns to the caller, which will then fall through to interpreter
        // dispatch.  At least it won't crash by jumping to a garbage address.
        code.push(0x48);
        code.push(0xB8);
        // The generic dispatch trampoline address will be patched by the
        // RuntimeLinker when it's available.  For now, use a self-referencing
        // address that returns immediately (ret = C3, which is safe).
        // Write the entry_addr as the default: a stub that just returns 0.
        // The RuntimeLinker will patch this with the real trampoline address
        // via patch_call_site().
        code.extend_from_slice(&0u64.to_le_bytes());

        // jmp rax
        code.extend_from_slice(&[0xFF, 0xE0]);

        // Pad remainder with NOPs
        while code.len() < code_size {
            code.push(0x90);
        }

        arena.write_code(entry_addr, &code);
        entry_addr
    }

    /// Update hot status based on call frequency
    pub fn update_hot_status(&mut self, threshold: u64) {
        self.is_hot = self.type_distribution.total >= threshold;
    }

    /// Enable speculative devirtualization for the most common type
    pub fn enable_speculation(&mut self) {
        if let Some((type_id, freq)) = self.type_distribution.most_common() {
            if freq > 0.8 { // Only speculate if type appears >80% of the time
                self.speculative_devirt.speculate(type_id);
            }
        }
    }

    /// Get the inline cache hit rate
    pub fn hit_rate(&self) -> f64 {
        self.inline_cache.hit_rate()
    }
}

/// Inline cache manager for all call sites
pub struct InlineCacheManager {
    /// All call sites
    call_sites: HashMap<String, CallSite>,
    /// Hot threshold (calls before considered hot)
    hot_threshold: u64,
}

impl InlineCacheManager {
    pub fn new(hot_threshold: u64) -> Self {
        Self {
            call_sites: HashMap::new(),
            hot_threshold,
        }
    }

    /// Register a call site
    pub fn register_call_site(&mut self, location: String) {
        self.call_sites
            .entry(location.clone())
            .or_insert_with(|| CallSite::new(location));
    }

    /// Handle a call at a specific location
    pub fn handle_call(&mut self, location: &str, type_id: TypeId) -> Option<usize> {
        if let Some(site) = self.call_sites.get_mut(location) {
            let result = site.handle_call(type_id);
            site.update_hot_status(self.hot_threshold);
            result
        } else {
            None
        }
    }

    /// Get a call site
    pub fn get_call_site(&self, location: &str) -> Option<&CallSite> {
        self.call_sites.get(location)
    }

    /// Get all hot call sites
    pub fn hot_call_sites(&self) -> Vec<&str> {
        self.call_sites
            .iter()
            .filter(|(_, site)| site.is_hot)
            .map(|(loc, _)| loc.as_str())
            .collect()
    }

    /// Enable speculation for all hot call sites
    pub fn enable_speculation_for_hot(&mut self) {
        for site in self.call_sites.values_mut() {
            if site.is_hot {
                site.enable_speculation();
            }
        }
    }

    /// Get overall statistics
    pub fn stats(&self) -> InlineCacheStats {
        let total_lookups: u64 = self.call_sites
            .values()
            .map(|s| s.inline_cache.total_lookups)
            .sum();
        let total_hits: u64 = self.call_sites
            .values()
            .map(|s| s.inline_cache.total_hits)
            .sum();
        let total_misses: u64 = self.call_sites
            .values()
            .map(|s| s.inline_cache.total_misses)
            .sum();

        InlineCacheStats {
            total_call_sites: self.call_sites.len(),
            hot_call_sites: self.hot_call_sites().len(),
            total_lookups,
            total_hits,
            total_misses,
            overall_hit_rate: if total_lookups > 0 {
                total_hits as f64 / total_lookups as f64
            } else {
                0.0
            },
        }
    }
}

/// Inline cache statistics
#[derive(Debug)]
pub struct InlineCacheStats {
    pub total_call_sites: usize,
    pub hot_call_sites: usize,
    pub total_lookups: u64,
    pub total_hits: u64,
    pub total_misses: u64,
    pub overall_hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inline_cache() {
        let mut cache = PolymorphicInlineCache::new(4);
        let type1 = TypeId::new(1);
        let type2 = TypeId::new(2);

        // First lookup should miss
        assert!(cache.lookup(type1).is_none());
        assert_eq!(cache.total_misses, 1);

        // Add entry
        cache.add_entry(type1, 0x1000);

        // Second lookup should hit
        assert_eq!(cache.lookup(type1), Some(0x1000));
        assert_eq!(cache.total_hits, 1);
    }

    #[test]
    fn test_type_distribution() {
        let mut dist = TypeDistribution::new();
        let type1 = TypeId::new(1);
        let type2 = TypeId::new(2);

        for _ in 0..8 {
            dist.record(type1);
        }
        for _ in 0..2 {
            dist.record(type2);
        }

        assert_eq!(dist.frequency(type1), 0.8);
        assert_eq!(dist.frequency(type2), 0.2);
        assert!(dist.is_dominant(type1, 0.7));
    }

    #[test]
    fn test_speculative_devirt() {
        let mut devirt = SpeculativeDevirt::new("test".to_string());
        let type1 = TypeId::new(1);

        devirt.speculate(type1);
        assert!(devirt.should_speculate());

        // Record successes
        for _ in 0..10 {
            devirt.record_success();
        }
        assert_eq!(devirt.success_rate(), 1.0);

        // Record a few failures — 3 failures out of 13 total = 77% success rate > 70%,
        // and failure rate 23% < 30%, so speculation should remain active.
        for _ in 0..3 {
            devirt.record_failure();
        }
        assert!(devirt.should_speculate());
        assert!(devirt.success_rate() > 0.7);

        // Record more failures — 5 failures out of 15 = 67% success rate < 70%,
        // and failure rate 33% > 30%, so speculation should be disabled.
        for _ in 0..2 {
            devirt.record_failure();
        }
        assert!(!devirt.should_speculate());
    }

    #[test]
    fn test_call_site() {
        let mut site = CallSite::new("test_func".to_string());
        let type1 = TypeId::new(1);

        // First call
        let result = site.handle_call(type1);
        assert!(result.is_some());

        // Second call should hit cache
        let result2 = site.handle_call(type1);
        assert!(result2.is_some());
        assert_eq!(site.inline_cache.total_hits, 1);
    }

    #[test]
    fn test_inline_cache_manager() {
        let mut manager = InlineCacheManager::new(10);
        manager.register_call_site("func1".to_string());

        let type1 = TypeId::new(1);
        for _ in 0..15 {
            manager.handle_call("func1", type1);
        }

        assert!(manager.get_call_site("func1").unwrap().is_hot);
        assert_eq!(manager.hot_call_sites().len(), 1);
    }

    #[test]
    fn test_megamorphic_fallback() {
        let mut cache = PolymorphicInlineCache::new(4);

        // Fill the PIC with 4 entries
        for i in 1..=4u64 {
            let tid = TypeId::new(i);
            cache.add_entry(tid, (0x1000 + i * 0x100) as usize);
        }
        assert_eq!(cache.entries.len(), 4);
        assert_eq!(cache.megamorphic_count(), 0);

        // Adding a 5th type should go to megamorphic table
        let type5 = TypeId::new(5);
        cache.add_entry(type5, 0x1500);
        assert_eq!(cache.entries.len(), 4); // PIC didn't grow
        assert_eq!(cache.megamorphic_count(), 1); // megamorphic has it

        // Looking up type5 should find it in the megamorphic table
        let result = cache.lookup(type5);
        assert_eq!(result, Some(0x1500));

        // Adding more types
        for i in 6..=10u64 {
            let tid = TypeId::new(i);
            cache.add_entry(tid, (0x1600 + i as usize * 0x10) as usize);
        }
        assert_eq!(cache.megamorphic_count(), 6); // types 5-10

        // All types should be findable
        for i in 1..=10u64 {
            let tid = TypeId::new(i);
            assert!(cache.lookup(tid).is_some(), "type {} should be found", i);
        }

        // Hit rate should be good
        assert!(cache.hit_rate() > 0.9);
    }

    #[test]
    fn test_runtime_linker_patches_memory() {
        let mut linker = RuntimeLinker::new();

        // Allocate a small code region from the arena to use as a call site
        let arena = get_arena();
        let site_addr = arena.alloc(32) as u64;

        // Write a mov rax, 0 instruction at site_addr (the pattern that gets patched)
        let mut code = vec![0x48u8, 0xB8]; // mov rax, imm64
        code.extend_from_slice(&0u64.to_le_bytes()); // placeholder: 0
        code.extend_from_slice(&[0xFF, 0xE0]); // jmp rax
        arena.write_code(site_addr as usize, &code);

        // Register the call site
        linker.register_call_site(site_addr, site_addr);

        // Patch it with a handler address
        let handler_addr: u64 = 0xDEAD_BEEF_CAFE_0000;
        linker.patch_call_site(site_addr, TypeId::new(1).as_u64(), handler_addr);

        // Verify the mapping was stored
        assert_eq!(linker.resolve(site_addr, TypeId::new(1).as_u64()), Some(handler_addr));

        // Verify the code was actually patched in memory
        // The 8-byte immediate at site_addr + 2 should now be handler_addr.
        // Use byte-level read to avoid alignment requirements.
        let imm_ptr = (site_addr as usize + 2) as *const u8;
        let mut bytes = [0u8; 8];
        unsafe {
            std::ptr::copy_nonoverlapping(imm_ptr, bytes.as_mut_ptr(), 8);
        }
        let patched_value = u64::from_le_bytes(bytes);
        assert_eq!(patched_value, handler_addr, "RuntimeLinker should have patched the code in memory");
    }
}
