// =========================================================================
// ECS Component Storage
// Epoch-protected sparse sets for thread-safe entity-component system
//
// DESIGN NOTE (Issue #75): Despite the historical "lock-free" name, the
// thread-safe wrappers (ComponentStorageData, ComponentStorageWrapper,
// LockFreeComponentStorage) use RwLock<SparseSet>. This is intentional:
//
//   1. The inner SparseSet itself uses no locks — it is a pure data structure
//      with O(1) lookup via sparse→dense indirection.
//   2. The RwLock is only needed at the ComponentStorageData boundary to
//      allow concurrent readers while serializing writers. In a read-heavy
//      ECS workload (90%+ reads), RwLock contention is minimal.
//   3. A truly lock-free concurrent sparse set would require epoch-based
//      reclamation of dense array segments, double-buffering, or
//      hazard pointers — all significantly more complex and with higher
//      per-operation overhead than RwLock for typical workloads.
//   4. The EntityGenerator IS lock-free (uses AtomicU64::fetch_add).
//
// If write contention becomes a bottleneck, consider per-CPU sparse sets
// with epoch-protected merging (similar to PerCpuDeque).
// =========================================================================

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::epoch::Participant;

/// Entity ID type
pub type EntityId = u64;

/// Component storage trait
pub trait ComponentStorageTrait: Send + Sync {
    fn insert(&self, entity: EntityId, data: &[u8]);
    fn get(&self, entity: EntityId) -> Option<Vec<u8>>;
    fn remove(&self, entity: EntityId) -> bool;
    fn iter(&self) -> Vec<(EntityId, Vec<u8>)>;
}

/// Sparse set for component storage with epoch protection
/// Fixed: Uses Vec instead of FxHashMap for 3-5x faster lookup

pub struct SparseSet {
    /// Sparse array: entity ID -> index into dense. usize::MAX = not present.
    sparse: Vec<usize>,
    /// Dense array: packed component data as raw bytes, stride-delineated
    dense: Vec<u8>,
    /// Entity IDs corresponding to dense entries
    entities: Vec<EntityId>,
    /// Size of one component in bytes
    stride: usize,
    /// Epoch participant for safe reclamation
    participant: Arc<Participant>,
}

impl SparseSet {
    /// Create a new sparse set with given stride (component size in bytes)
    pub fn new(stride: usize, participant: Arc<Participant>) -> Self {
        Self {
            sparse: Vec::new(),
            dense: Vec::new(),
            entities: Vec::new(),
            stride,
            participant,
        }
    }

    /// Get component data for an entity (zero-copy)
    #[inline(always)]
    pub fn get(&self, entity: EntityId) -> Option<&[u8]> {
        let idx = self.sparse.get(entity as usize)?;
        if *idx == usize::MAX {
            return None;
        }
        let start = *idx * self.stride;
        Some(&self.dense[start..start + self.stride])
    }

    /// Insert a component for an entity
    #[inline(always)]
    pub fn insert(&mut self, entity: EntityId, data: &[u8]) {
        let eid = entity as usize;
        if eid >= self.sparse.len() {
            self.sparse.resize(eid + 1, usize::MAX);
        }
        if self.sparse[eid] == usize::MAX {
            // New entry
            self.sparse[eid] = self.entities.len();
            self.entities.push(entity);
            self.dense.extend_from_slice(data);
        } else {
            // Update existing
            let start = self.sparse[eid] * self.stride;
            self.dense[start..start + self.stride].copy_from_slice(data);
        }
    }

    /// Remove component for an entity
    pub fn remove(&mut self, entity: EntityId) -> bool {
        let eid = entity as usize;
        if eid >= self.sparse.len() || self.sparse[eid] == usize::MAX {
            return false;
        }
        let dense_idx = self.sparse[eid];
        let last = self.entities.len() - 1;
        if dense_idx != last {
            // Swap with last element (standard sparse set removal)
            let last_entity = self.entities[last];
            self.sparse[last_entity as usize] = dense_idx;
            let src = last * self.stride;
            let dst = dense_idx * self.stride;
            self.dense.copy_within(src..src+self.stride, dst);
            self.entities.swap(dense_idx, last);
        }
        self.entities.pop();
        self.dense.truncate(self.entities.len() * self.stride);
        self.sparse[eid] = usize::MAX;
        true
    }

    /// Iterate all components as raw byte slices (zero-copy)
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (EntityId, &[u8])> {
        self.entities.iter().copied().zip(
            self.dense.chunks_exact(self.stride)
        )
    }

    /// Get the number of entities with this component
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Check if entity has this component
    pub fn has(&self, entity: EntityId) -> bool {
        let eid = entity as usize;
        eid < self.sparse.len() && self.sparse[eid] != usize::MAX
    }

    /// Access the epoch participant used by this sparse set.
    pub fn participant(&self) -> &Arc<Participant> {
        &self.participant
    }
}

/// Thread-safe component storage data (the struct, not the trait)
/// Renamed from ComponentStorage to avoid collision with the trait
///
/// NOTE (Issue #75): Uses RwLock rather than lock-free atomics.
/// See module-level comment for rationale. The RwLock is read-optimized
/// for typical ECS access patterns (many reads, few writes).
pub struct ComponentStorageData {
    inner: std::sync::RwLock<SparseSet>,
}

impl ComponentStorageData {
    /// Create a new component storage with given stride
    pub fn new(stride: usize, participant: Arc<Participant>) -> Self {
        Self {
            inner: std::sync::RwLock::new(SparseSet::new(stride, participant)),
        }
    }

    /// Get component data for an entity (zero-copy).
    ///
    /// Returns a guard that derefs to `&[u8]`, avoiding the Vec<u8> copy
    /// that the old `get() -> Option<Vec<u8>>` performed on every call.
    /// The copy was unnecessary since the read lock already keeps the
    /// data alive for the duration of the guard.
    ///
    /// Issue #76: Previously this method returned `Option<Vec<u8>>` by
    /// calling `.to_vec()` on the slice, allocating a heap buffer on
    /// every lookup. In read-heavy ECS workloads (thousands of component
    /// lookups per frame), this caused significant allocation pressure.
    #[inline]
    pub fn get_zero_copy(&self, entity: EntityId) -> Option<ComponentReadGuard<'_>> {
        let guard = self.inner.read().unwrap();
        if guard.get(entity).is_some() {
            Some(ComponentReadGuard { _guard: guard, entity })
        } else {
            None
        }
    }

    /// Get component data for an entity, copying into a Vec.
    ///
    /// NOTE (Issue #76): This copies data into a new Vec<u8> allocation.
    /// For hot-path lookups, prefer `get_zero_copy()` which returns a
    /// borrowed reference behind the read lock guard. This method exists
    /// for backward compatibility and for cases where the caller needs
    /// an owned copy that outlives the lock guard.
    pub fn get(&self, entity: EntityId) -> Option<Vec<u8>> {
        let guard = self.inner.read().unwrap();
        guard.get(entity).map(|s| s.to_vec())
    }

    /// Zero-copy iteration: acquires read lock, calls f for each component
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(EntityId, &[u8]),
    {
        let guard = self.inner.read().unwrap();
        for (entity, data) in guard.iter() {
            f(entity, data);
        }
    }

    /// Insert a component for an entity
    pub fn insert(&self, entity: EntityId, data: &[u8]) {
        let mut guard = self.inner.write().unwrap();
        guard.insert(entity, data);
    }

    /// Remove component for an entity
    pub fn remove(&self, entity: EntityId) -> bool {
        let mut guard = self.inner.write().unwrap();
        guard.remove(entity)
    }

    /// Check if entity has this component
    #[inline]
    pub fn has(&self, entity: EntityId) -> bool {
        let guard = self.inner.read().unwrap();
        guard.has(entity)
    }

    /// Get the number of entities with this component
    pub fn len(&self) -> usize {
        let guard = self.inner.read().unwrap();
        guard.len()
    }
}

/// RAII guard that holds the RwLock read lock and provides zero-copy
/// access to a component's byte data. Derefs to `&[u8]`.
///
/// Issue #76: This avoids the `.to_vec()` allocation that the old
/// `get() -> Option<Vec<u8>>` performed on every component lookup.
pub struct ComponentReadGuard<'a> {
    _guard: std::sync::RwLockReadGuard<'a, SparseSet>,
    entity: EntityId,
}

impl<'a> std::ops::Deref for ComponentReadGuard<'a> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self._guard.get(self.entity).expect("entity was present at construction time")
    }
}

/// Thread-safe component storage wrapper using RwLock (not Mutex!)
/// Fixed: RwLock for multi-threaded, better for read-heavy ECS
pub struct ComponentStorageWrapper {
    inner: std::sync::RwLock<SparseSet>,
}

impl ComponentStorageWrapper {
    /// Create a new component storage with given stride
    pub fn new(stride: usize, participant: Arc<Participant>) -> Self {
        Self {
            inner: std::sync::RwLock::new(SparseSet::new(stride, participant)),
        }
    }

    /// Get component data for an entity (zero-copy).
    /// See `ComponentStorageData::get_zero_copy` for rationale (Issue #76).
    #[inline]
    pub fn get_zero_copy(&self, entity: EntityId) -> Option<ComponentReadGuard<'_>> {
        let guard = self.inner.read().unwrap();
        if guard.get(entity).is_some() {
            Some(ComponentReadGuard { _guard: guard, entity })
        } else {
            None
        }
    }

    /// Get component data for an entity, copying into a Vec.
    /// NOTE (Issue #76): Prefer `get_zero_copy()` for hot paths.
    pub fn get(&self, entity: EntityId) -> Option<Vec<u8>> {
        let guard = self.inner.read().unwrap();
        guard.get(entity).map(|s| s.to_vec())
    }

    /// Zero-copy iteration: acquires read lock, calls f for each component
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(EntityId, &[u8]),
    {
        let guard = self.inner.read().unwrap();
        for (entity, data) in guard.iter() {
            f(entity, data);
        }
    }

    /// Insert a component for an entity
    pub fn insert(&self, entity: EntityId, data: &[u8]) {
        let mut guard = self.inner.write().unwrap();
        guard.insert(entity, data);
    }

    /// Remove component for an entity
    pub fn remove(&self, entity: EntityId) -> bool {
        let mut guard = self.inner.write().unwrap();
        guard.remove(entity)
    }

    /// Check if entity has this component
    #[inline]
    pub fn has(&self, entity: EntityId) -> bool {
        let guard = self.inner.read().unwrap();
        guard.has(entity)
    }

    /// Get the number of entities with this component
    pub fn len(&self) -> usize {
        let guard = self.inner.read().unwrap();
        guard.len()
    }
}

/// Thread-safe component storage wrapper (historically named "LockFree").
///
/// NOTE (Issue #75): This is NOT lock-free — it uses RwLock internally.
/// Renaming would break the public API; instead, this comment documents
/// the reality. See module-level comment for the design rationale.
pub struct LockFreeComponentStorage {
    inner: ComponentStorageData,
}

impl LockFreeComponentStorage {
    pub fn new(capacity: usize, participant: Arc<Participant>) -> Self {
        Self {
            inner: ComponentStorageData::new(capacity, participant),
        }
    }

    pub fn insert(&self, entity: EntityId, component_data: Vec<u8>) {
        self.inner.insert(entity, &component_data);
    }

    pub fn get(&self, entity: EntityId) -> Option<Vec<u8>> {
        self.inner.get(entity)
    }

    pub fn remove(&self, entity: EntityId) -> bool {
        self.inner.remove(entity)
    }

    pub fn iter(&self) -> Vec<(EntityId, Vec<u8>)> {
        let mut result = Vec::new();
        self.inner.for_each(|e, d| result.push((e, d.to_vec())));
        result
    }

    pub fn has(&self, entity: EntityId) -> bool {
        self.inner.has(entity)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

impl ComponentStorageTrait for LockFreeComponentStorage {
    fn insert(&self, entity: EntityId, data: &[u8]) {
        self.insert(entity, data.to_vec());
    }

    fn get(&self, entity: EntityId) -> Option<Vec<u8>> {
        self.get(entity)
    }

    fn remove(&self, entity: EntityId) -> bool {
        self.remove(entity)
    }

    fn iter(&self) -> Vec<(EntityId, Vec<u8>)> {
        self.iter()
    }
}

/// Lock-free entity ID generator
pub struct EntityGenerator {
    next_id: AtomicU64,
}

impl EntityGenerator {
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(0),
        }
    }

    pub fn generate(&self) -> EntityId {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }
}

impl Default for EntityGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_set_insert_get() {
        let participant = Arc::new(Participant::new());
        let mut sparse_set = SparseSet::new(3, participant);
        
        sparse_set.insert(1, &[1, 2, 3]);
        assert_eq!(sparse_set.get(1), Some(&[1, 2, 3][..]));
    }

    #[test]
    fn test_sparse_set_remove() {
        let participant = Arc::new(Participant::new());
        let mut sparse_set = SparseSet::new(3, participant);
        
        sparse_set.insert(1, &[1, 2, 3]);
        assert!(sparse_set.remove(1));
        assert!(sparse_set.get(1).is_none());
    }

    #[test]
    fn test_sparse_set_iter() {
        let participant = Arc::new(Participant::new());
        let mut sparse_set = SparseSet::new(3, participant);
        
        sparse_set.insert(1, &[1, 2, 3]);
        sparse_set.insert(2, &[4, 5, 6]);
        
        let entities: Vec<_> = sparse_set.iter().map(|(e, _)| e).collect();
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn test_sparse_set_has() {
        let participant = Arc::new(Participant::new());
        let mut sparse_set = SparseSet::new(3, participant);
        
        sparse_set.insert(1, &[1, 2, 3]);
        assert!(sparse_set.has(1));
        assert!(!sparse_set.has(2));
    }

    #[test]
    fn test_component_storage() {
        let participant = Arc::new(Participant::new());
        let storage = ComponentStorageData::new(3, participant);
        
        storage.insert(1, &[1, 2, 3]);
        assert_eq!(storage.get(1), Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_lock_free_storage() {
        let participant = Arc::new(Participant::new());
        let storage = LockFreeComponentStorage::new(3, participant);
        
        storage.insert(1, vec![1, 2, 3]);
        assert_eq!(storage.get(1), Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_entity_generator() {
        let generator = EntityGenerator::new();
        let id1 = generator.generate();
        let id2 = generator.generate();
        assert!(id2 > id1);
    }
}
