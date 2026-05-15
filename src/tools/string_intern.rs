// =============================================================================
// jules/src/tools/string_intern.rs
//
// STRING INTERNING SYSTEM — delegates to the canonical interner in symbol.rs
//
// Issue #90: This module previously maintained its own separate StringId(u32)
// interner, duplicating the functionality already in runtime::symbol::Symbol(u32).
// Having three separate interners (tools::string_intern, runtime::symbol,
// optimizer::mcts_superoptimizer) meant that the same string could be interned
// with three different IDs across the codebase, preventing O(1) cross-module
// comparisons and wasting memory.
//
// This module now delegates to runtime::symbol::StringInterner, which is the
// canonical (single) interner. The StringId type is a newtype wrapper around
// Symbol for backward compatibility with any code that references StringId.
// =============================================================================

use std::sync::atomic::{AtomicU32, Ordering};

/// Opaque handle to an interned string.
///
/// Internally wraps `crate::runtime::symbol::Symbol` so that all interners
/// across the codebase share the same ID space. Two StringIds are equal
/// iff they refer to the same interned string (via the canonical Symbol).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct StringId(pub crate::runtime::symbol::Symbol);

impl StringId {
    /// Reserved ID for empty/missing strings
    pub const NONE: Self = StringId(crate::runtime::symbol::Symbol::EMPTY);

    #[inline(always)]
    pub fn as_u32(self) -> u32 {
        self.0.as_u32()
    }

    /// Convert to the underlying Symbol (canonical interner handle).
    #[inline(always)]
    pub fn as_symbol(self) -> crate::runtime::symbol::Symbol {
        self.0
    }
}

// Re-export the canonical global interner functions under the old names
// so any code using `string_intern::global_intern` etc. continues to work.

/// Intern a string using the canonical global interner.
/// Returns a Symbol that can be used for fast comparisons.
/// Thread-safe but should be avoided in ultra-hot paths.
#[inline(always)]
pub fn global_intern(s: &str) -> StringId {
    StringId(crate::runtime::symbol::global_intern(s))
}

/// Resolve a StringId back to its string using the canonical global interner.
#[inline(always)]
pub fn global_resolve(id: StringId) -> String {
    crate::runtime::symbol::global_resolve(id.0)
}

/// Look up a string in the canonical global interner.
#[inline(always)]
pub fn global_get(s: &str) -> Option<StringId> {
    crate::runtime::symbol::global_get(s).map(StringId)
}

// Atomic counter for generating unique IDs without interning
// (kept for backward compatibility with any code using generate_id)
static NEXT_ID: AtomicU32 = AtomicU32::new(1);

#[inline(always)]
pub fn generate_id() -> u32 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interning_via_symbol() {
        let id1 = global_intern("hello");
        let id2 = global_intern("hello");
        let id3 = global_intern("world");
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert_eq!(global_resolve(id1), "hello");
        assert_eq!(global_resolve(id3), "world");
    }

    #[test]
    fn test_cross_module_compatibility() {
        // StringId should be compatible with Symbol from the canonical interner
        let string_id = global_intern("test_cross");
        let symbol = crate::runtime::symbol::global_intern("test_cross");
        // Both should resolve to the same string
        assert_eq!(global_resolve(string_id), crate::runtime::symbol::global_resolve(symbol));
    }
}
