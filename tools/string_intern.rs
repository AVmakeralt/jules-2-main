// =============================================================================
// jules/src/string_intern.rs
//
// STRING INTERNING SYSTEM
//
// Eliminates repeated string allocations for field names, component names,
// variable names, etc. Converts string comparisons to pointer comparisons.
//
// Performance benefits:
// - String comparisons: O(n) -> O(1) (pointer equality)
// - HashMap lookups: eliminated hash computation for interned strings
// - Memory usage: shared strings stored once globally
// =============================================================================

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{LazyLock, Mutex};

/// Global string interning arena (thread-safe via Mutex)
pub static GLOBAL_INTERNER: LazyLock<Mutex<StringInterner>> = 
    LazyLock::new(|| Mutex::new(StringInterner::default()));

/// Global string interning arena
pub struct StringInterner {
    strings: Vec<String>,
    lookup: HashMap<String, StringId>,
}

/// Opaque handle to an interned string
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct StringId(u32);

impl StringId {
    /// Reserved ID for empty/missing strings
    pub const NONE: Self = StringId(0);

    #[inline(always)]
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self {
            // Reserve index 0 for NONE
            strings: vec![String::new()],
            lookup: HashMap::new(),
        }
    }
}

impl StringInterner {
    /// Intern a string, returning its ID
    #[inline(always)]
    pub fn intern(&mut self, s: &str) -> StringId {
        if let Some(&id) = self.lookup.get(s) {
            return id;
        }
        let id = StringId(self.strings.len() as u32);
        self.strings.push(s.to_string());
        self.lookup.insert(s.to_string(), id);
        id
    }

    /// Resolve a StringId back to &str
    #[inline(always)]
    pub fn resolve(&self, id: StringId) -> &str {
        &self.strings[id.0 as usize]
    }

    /// Intern a string from an owned String (avoids double allocation)
    #[inline(always)]
    pub fn intern_owned(&mut self, s: String) -> StringId {
        if let Some(&id) = self.lookup.get(&s) {
            return id;
        }
        let id = StringId(self.strings.len() as u32);
        self.lookup.insert(s.clone(), id);
        self.strings.push(s);
        id
    }
}

/// Thread-local string interning for hot paths (no locking overhead)
thread_local! {
    static THREAD_LOCAL_INTERNER: std::cell::RefCell<StringInterner> =
        std::cell::RefCell::new(StringInterner::default());
}

/// Fast thread-local string interning
#[inline(always)]
pub fn intern_thread_local(s: &str) -> StringId {
    THREAD_LOCAL_INTERNER.with(|cell| {
        cell.borrow_mut().intern(s)
    })
}

/// Fast thread-local string interning with owned string
#[inline(always)]
pub fn intern_thread_local_owned(s: String) -> StringId {
    THREAD_LOCAL_INTERNER.with(|cell| {
        cell.borrow_mut().intern_owned(s)
    })
}

/// Resolve a StringId using thread-local interner
#[inline(always)]
pub fn resolve_thread_local(id: StringId) -> String {
    THREAD_LOCAL_INTERNER.with(|cell| {
        cell.borrow().resolve(id).to_string()
    })
}

// Atomic counter for generating unique IDs without interning
static NEXT_ID: AtomicU32 = AtomicU32::new(1);

#[inline(always)]
pub fn generate_id() -> u32 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interning() {
        let mut interner = StringInterner::default();
        let id1 = interner.intern("hello");
        let id2 = interner.intern("hello");
        let id3 = interner.intern("world");
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert_eq!(interner.resolve(id1), "hello");
        assert_eq!(interner.resolve(id3), "world");
    }
}
