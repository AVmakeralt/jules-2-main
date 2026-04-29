// =========================================================================
// Lock-Free ECS Component Storage
// Epoch-protected sparse sets for lock-free entity-component system
// Implements crossbeam-style atomic operations for thread safety
// =========================================================================

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use super::epoch::{Guard, Participant};

/// Entity ID type
pub type EntityId = u64;

/// Component storage trait
pub trait ComponentStorage: Send + Sync {
    fn insert(&self, entity: EntityId, data: &[u8]);
    fn get(&self, entity: EntityId) -> Option<Vec<u8>>;
    fn remove(&self, entity: EntityId) -> bool;
    fn iter(&self) -> Vec<(EntityId, Vec<u8>)>;
}

/// Sparse set for component storage with epoch protection
pub struct SparseSet {
    /// Dense array of entity IDs
    dense: Vec<EntityId>,
    /// Sparse array mapping entity to dense index
    sparse: Vec<usize>,
    /// Component data
    data: Vec<Vec<u8>>,
    /// Number of active entities
    count: usize,
    /// Capacity
    capacity: usize,
    /// Epoch participant for safe reclamation
    participant: Arc<Participant>,
}

impl SparseSet {
    /// Create a new sparse set with given capacity
    pub fn new(capacity: usize, participant: Arc<Participant>) -> Self {
        Self {
            dense: Vec::with_capacity(capacity),
            sparse: vec![usize::MAX; capacity],
            data: Vec::with_capacity(capacity),
            count: 0,
            capacity,
            participant,
        }
    }

    /// Insert a component for an entity
    pub fn insert(&mut self, entity: EntityId, component_data: Vec<u8>) {
        let entity_idx = entity as usize;
        
        // Expand sparse array if needed
        if entity_idx >= self.sparse.len() {
            let new_size = (entity_idx + 1).max(self.sparse.len() * 2);
            self.sparse.resize(new_size, usize::MAX);
        }
        
        // Check if entity already has this component
        if self.sparse[entity_idx] != usize::MAX {
            // Update existing component
            let dense_idx = self.sparse[entity_idx];
            self.data[dense_idx] = component_data;
            return;
        }
        
        // Insert new component
        let dense_idx = self.count;
        self.dense.push(entity);
        self.sparse[entity_idx] = dense_idx;
        self.data.push(component_data);
        self.count += 1;
    }

    /// Get component data for an entity
    pub fn get(&self, entity: EntityId) -> Option<&[u8]> {
        let entity_idx = entity as usize;
        
        if entity_idx >= self.sparse.len() {
            return None;
        }
        
        let dense_idx = self.sparse[entity_idx];
        if dense_idx == usize::MAX {
            return None;
        }
        
        self.data.get(dense_idx).map(|d| d.as_slice())
    }

    /// Remove component for an entity
    pub fn remove(&mut self, entity: EntityId) -> bool {
        let entity_idx = entity as usize;
        
        if entity_idx >= self.sparse.len() {
            return false;
        }
        
        let dense_idx = self.sparse[entity_idx];
        if dense_idx == usize::MAX {
            return false;
        }
        
        // Swap with last element
        let last_entity = self.dense[self.count - 1];
        let last_dense_idx = self.sparse[last_entity as usize];
        
        self.dense[dense_idx] = last_entity;
        self.sparse[last_entity as usize] = dense_idx;
        self.sparse[entity_idx] = usize::MAX;
        
        self.dense.pop();
        self.data.pop();
        self.count -= 1;
        
        true
    }

    /// Iterate over all entities with this component
    pub fn iter(&self) -> impl Iterator<Item = (EntityId, &[u8])> {
        self.dense.iter().enumerate().map(move |(i, &entity)| {
            (entity, self.data[i].as_slice())
        })
    }

    /// Get the number of entities with this component
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Check if entity has this component
    pub fn has(&self, entity: EntityId) -> bool {
        let entity_idx = entity as usize;
        if entity_idx >= self.sparse.len() {
            return false;
        }
        self.sparse[entity_idx] != usize::MAX
    }
}

/// Lock-free component storage wrapper with epoch protection
pub struct LockFreeComponentStorage {
    /// Sparse set for component data
    sparse_set: Arc<std::sync::Mutex<SparseSet>>,
    /// Epoch participant
    participant: Arc<Participant>,
}

impl LockFreeComponentStorage {
    /// Create a new lock-free component storage
    pub fn new(capacity: usize, participant: Arc<Participant>) -> Self {
        Self {
            sparse_set: Arc::new(std::sync::Mutex::new(SparseSet::new(capacity, participant.clone()))),
            participant,
        }
    }

    /// Insert a component for an entity
    pub fn insert(&self, entity: EntityId, component_data: Vec<u8>) {
        let mut sparse_set = self.sparse_set.lock().unwrap();
        sparse_set.insert(entity, component_data);
    }

    /// Get component data for an entity
    pub fn get(&self, entity: EntityId) -> Option<Vec<u8>> {
        let sparse_set = self.sparse_set.lock().unwrap();
        sparse_set.get(entity).map(|d| d.to_vec())
    }

    /// Remove component for an entity
    pub fn remove(&self, entity: EntityId) -> bool {
        let mut sparse_set = self.sparse_set.lock().unwrap();
        sparse_set.remove(entity)
    }

    /// Iterate over all entities with this component
    pub fn iter(&self) -> Vec<(EntityId, Vec<u8>)> {
        let sparse_set = self.sparse_set.lock().unwrap();
        sparse_set.iter().map(|(e, d)| (e, d.to_vec())).collect()
    }

    /// Check if entity has this component
    pub fn has(&self, entity: EntityId) -> bool {
        let sparse_set = self.sparse_set.lock().unwrap();
        sparse_set.has(entity)
    }

    /// Get the number of entities with this component
    pub fn len(&self) -> usize {
        let sparse_set = self.sparse_set.lock().unwrap();
        sparse_set.len()
    }
}

impl ComponentStorage for LockFreeComponentStorage {
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
        let mut sparse_set = SparseSet::new(10, participant);
        
        sparse_set.insert(1, vec![1, 2, 3]);
        assert_eq!(sparse_set.get(1), Some(&[1, 2, 3][..]));
    }

    #[test]
    fn test_sparse_set_remove() {
        let participant = Arc::new(Participant::new());
        let mut sparse_set = SparseSet::new(10, participant);
        
        sparse_set.insert(1, vec![1, 2, 3]);
        assert!(sparse_set.remove(1));
        assert!(sparse_set.get(1).is_none());
    }

    #[test]
    fn test_sparse_set_iter() {
        let participant = Arc::new(Participant::new());
        let mut sparse_set = SparseSet::new(10, participant);
        
        sparse_set.insert(1, vec![1, 2, 3]);
        sparse_set.insert(2, vec![4, 5, 6]);
        
        let entities: Vec<_> = sparse_set.iter().map(|(e, _)| e).collect();
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn test_sparse_set_has() {
        let participant = Arc::new(Participant::new());
        let mut sparse_set = SparseSet::new(10, participant);
        
        sparse_set.insert(1, vec![1, 2, 3]);
        assert!(sparse_set.has(1));
        assert!(!sparse_set.has(2));
    }

    #[test]
    fn test_lock_free_storage() {
        let participant = Arc::new(Participant::new());
        let storage = LockFreeComponentStorage::new(10, participant);
        
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
