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
            location,
            inline_cache: PolymorphicInlineCache::new(4), // Cache 4 types
            type_distribution: TypeDistribution::new(),
            speculative_devirt: SpeculativeDevirt::new(location.clone()),
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

    /// Generate specialized code for a type
    fn generate_specialized_code(&self, type_id: TypeId) -> usize {
        // In a real implementation, this would JIT-compile specialized code
        // For now, return a placeholder address
        (type_id.as_u64() as usize) & 0xFFFF
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
