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
// =============================================================================

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Executable code arena for inline cache entries.
///
/// Manages a pool of executable memory pages where JIT-compiled specialized
/// code is stored. Each entry is aligned and tracked so that cache lookups
/// return valid, executable addresses.
struct CodeArena {
    /// Allocated code pages (each page is a Vec<u8> of machine code).
    pages: Vec<Vec<u8>>,
    /// Current offset within the last page.
    current_offset: usize,
    /// Page size in bytes (matches OS page size for potential mmap use).
    page_size: usize,
    /// Base virtual address for the arena (simulated; in a real JIT this
    /// would be an mmap'd region with RWX permissions).
    base_address: usize,
}

impl CodeArena {
    fn new() -> Self {
        let page_size = 4096;
        Self {
            pages: Vec::new(),
            current_offset: 0,
            page_size,
            base_address: 0x7F00_0000_0000, // Simulated high address
        }
    }

    /// Allocate `size` bytes in the arena and return the executable address.
    fn alloc(&mut self, size: usize) -> usize {
        let aligned_size = (size + 15) & !15; // Align to 16 bytes

        // Ensure there's room in the current page, or allocate a new one
        if self.pages.is_empty() || self.current_offset + aligned_size > self.page_size {
            self.pages.push(vec![0xCC; self.page_size]); // Fill with INT3 (debug trap)
            self.current_offset = 0;
        }

        let page_index = self.pages.len() - 1;
        let address = self.base_address + page_index * self.page_size + self.current_offset;
        self.current_offset += aligned_size;
        address
    }

    /// Write code bytes at a previously allocated address.
    fn write_code(&mut self, address: usize, code: &[u8]) {
        let offset = address - self.base_address;
        let page_index = offset / self.page_size;
        let page_offset = offset % self.page_size;

        if let Some(page) = self.pages.get_mut(page_index) {
            let end = (page_offset + code.len()).min(page.len());
            let write_len = end - page_offset;
            page[page_offset..page_offset + write_len].copy_from_slice(&code[..write_len]);
        }
    }
}

/// Runtime linker for patching inline cache call sites.
///
/// When a type-guard trampoline misses (the incoming type doesn't match the
/// cached type), it jumps to a generic dispatch trampoline.  The `RuntimeLinker`
/// manages the mapping from call-site addresses to their dispatch handlers and
/// can patch call sites once a concrete type-specific handler is available.
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
            trampoline_addr: 0, // Would be set to actual trampoline in real impl
        }
    }

    /// Register a call site that needs runtime patching
    pub fn register_call_site(&mut self, site_addr: u64, stub_addr: u64) {
        self.call_sites.insert(site_addr, stub_addr);
    }

    /// Patch a call site to jump directly to a type-specific handler
    pub fn patch_call_site(&mut self, site_addr: u64, type_id: u64, handler_addr: u64) {
        self.type_handlers.insert((site_addr, type_id), handler_addr);
        // In a real implementation, this would overwrite the 8-byte address
        // at site_addr with handler_addr using mprotect + memcpy
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
    /// The specialized code address (in a real implementation)
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
/// specialized fast paths for each.
#[derive(Debug)]
pub struct PolymorphicInlineCache {
    /// Cache entries (max N entries)
    entries: Vec<InlineCacheEntry>,
    /// Maximum number of entries in the cache
    max_entries: usize,
    /// Total number of cache lookups
    total_lookups: u64,
    /// Total number of cache hits
    total_hits: u64,
    /// Total number of cache misses
    total_misses: u64,
}

impl PolymorphicInlineCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::with_capacity(max_entries),
            max_entries,
            total_lookups: 0,
            total_hits: 0,
            total_misses: 0,
        }
    }

    /// Look up a type in the cache
    pub fn lookup(&mut self, type_id: TypeId) -> Option<usize> {
        self.total_lookups += 1;

        for entry in &mut self.entries {
            if entry.type_id == type_id {
                entry.record_hit();
                self.total_hits += 1;
                return Some(entry.code_address);
            }
        }

        self.total_misses += 1;
        None
    }

    /// Add a new entry to the cache
    pub fn add_entry(&mut self, type_id: TypeId, code_address: usize) {
        // Check if already exists
        for entry in &self.entries {
            if entry.type_id == type_id {
                return; // Already cached
            }
        }

        // If cache is full, evict the least recently used entry
        if self.entries.len() >= self.max_entries {
            let lru_index = self.entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.entries.remove(lru_index);
        }

        self.entries.push(InlineCacheEntry::new(type_id, code_address));
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
        self.total_lookups = 0;
        self.total_hits = 0;
        self.total_misses = 0;
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
    pub fn should_speculate(&self) -> bool {
        self.is_speculated && self.success_rate() > 0.7
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
            inline_cache: PolymorphicInlineCache::new(4), // Cache 4 types
            type_distribution: TypeDistribution::new(),
            speculative_devirt: SpeculativeDevirt::new(location),
            is_hot: false,
        }
    }

    /// Handle a call at this site
    pub fn handle_call(&mut self, type_id: TypeId) -> Option<usize> {
        // Record type in distribution
        self.type_distribution.record(type_id);

        // Try inline cache
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
    /// This emits a minimal inline-cache stub in the code arena that:
    ///   1. Compares the incoming type tag against the cached type_id.
    ///   2. If it matches, jumps to the type-specialized fast path.
    ///   3. If it doesn't match, falls back to the generic dispatch path.
    ///
    /// The emitted x86-64 machine code is a type-guard trampoline that
    /// the inline cache can jump to on a cache hit.
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

        // Slow path: load generic dispatch stub address into rax and jump
        // mov rax, <address>
        code.push(0x48);
        code.push(0xB8);
        // Placeholder: RuntimeLinker will patch this with the actual generic dispatch address.
        // Using a recognizable pattern for debugging: 0xCAFE_CAFE_0000_0000
        // The generic dispatch trampoline reads the type ID from RDI, looks it up
        // in a global dispatch table, and jumps to the type-specific handler.
        code.extend_from_slice(&0xCAFE_CAFE_0000_0000u64.to_le_bytes());

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

        // Record failures
        for _ in 0..5 {
            devirt.record_failure();
        }
        // Should still speculate (success rate > 70%)
        assert!(devirt.should_speculate());
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
}
