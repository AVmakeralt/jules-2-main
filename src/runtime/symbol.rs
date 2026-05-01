// =============================================================================
// jules/src/runtime/symbol.rs
//
// SYMBOL TABLE & STRING INTERNER
//
// Replaces hot-path String comparisons with O(1) u32 comparisons.
// Every component name, field name, and frequently-compared string is
// interned once and then referred to by its Symbol (u32) ID.
//
// Performance impact:
// - ECS component lookup: FxHashMap<Symbol, SparseSet> instead of FxHashMap<String, SparseSet>
//   eliminates string hashing on every component access
// - Struct field access: FxHashMap<Symbol, Value> instead of FxHashMap<String, Value>
// - String equality: u32 comparison instead of byte-by-byte comparison
// =============================================================================

use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU32, Ordering};

/// A Symbol is a unique identifier for an interned string.
/// Two Symbols are equal iff they refer to the same interned string.
/// Symbol(0) is reserved for the empty/invalid symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Symbol(pub u32);

impl Symbol {
    /// The empty/invalid symbol.
    pub const EMPTY: Symbol = Symbol(0);

    #[inline(always)]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    #[inline(always)]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Global symbol counter for generating unique IDs across all threads.
static GLOBAL_SYMBOL_COUNTER: AtomicU32 = AtomicU32::new(1); // 0 reserved for EMPTY

/// Thread-local string interner for fast symbol lookup and deduplication.
///
/// Usage pattern:
/// 1. At compile time / program load time, intern all known identifiers
/// 2. At runtime, look up symbols by string for fast comparison
/// 3. For debugging, resolve symbols back to strings
#[derive(Debug)]
pub struct StringInterner {
    /// String → Symbol mapping for O(1) internment
    string_to_symbol: FxHashMap<String, Symbol>,
    /// Symbol → String mapping for O(1) resolution
    symbol_to_string: Vec<String>,
}

impl StringInterner {
    pub fn new() -> Self {
        let mut interner = Self {
            string_to_symbol: FxHashMap::default(),
            symbol_to_string: Vec::with_capacity(256),
        };
        // Reserve slot 0 for EMPTY
        interner.symbol_to_string.push(String::new());
        interner
    }

    /// Intern a string, returning its Symbol.
    /// If the string was already interned, returns the existing Symbol.
    /// O(1) amortized lookup via FxHashMap.
    #[inline(always)]
    pub fn intern(&mut self, s: &str) -> Symbol {
        if let Some(&sym) = self.string_to_symbol.get(s) {
            return sym;
        }
        let sym = Symbol(GLOBAL_SYMBOL_COUNTER.fetch_add(1, Ordering::Relaxed));
        self.string_to_symbol.insert(s.to_string(), sym);
        // Ensure the vec is large enough
        let idx = sym.0 as usize;
        if idx >= self.symbol_to_string.len() {
            self.symbol_to_string.resize(idx + 1, String::new());
        }
        self.symbol_to_string[idx] = s.to_string();
        sym
    }

    /// Look up an already-interned string. Returns None if not interned.
    #[inline(always)]
    pub fn get(&self, s: &str) -> Option<Symbol> {
        self.string_to_symbol.get(s).copied()
    }

    /// Resolve a Symbol back to its string. Returns "" for EMPTY.
    #[inline(always)]
    pub fn resolve(&self, sym: Symbol) -> &str {
        self.symbol_to_string
            .get(sym.0 as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Number of interned strings (excluding EMPTY).
    pub fn len(&self) -> usize {
        self.symbol_to_string.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Pre-intern common component names for ECS hot path.
    pub fn pre_intern_ecs_components(&mut self) {
        for name in &[
            "position", "pos", "velocity", "vel", "acceleration",
            "health", "damage", "score", "mass", "radius",
            "color", "alpha", "rotation", "scale", "tag",
            "player", "enemy", "bullet", "item", "platform",
            "x", "y", "z", "width", "height", "depth",
            "speed", "direction", "alive", "active", "visible",
        ] {
            self.intern(name);
        }
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// GLOBAL STRING INTERNER (lazy-init, thread-safe via Mutex)
// =============================================================================

use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref GLOBAL_INTERNER: Mutex<StringInterner> = Mutex::new({
        let mut i = StringInterner::new();
        i.pre_intern_ecs_components();
        i
    });
}

/// Intern a string using the global interner.
/// Returns a Symbol that can be used for fast comparisons.
/// Thread-safe but should be avoided in ultra-hot paths (use thread-local interner instead).
pub fn global_intern(s: &str) -> Symbol {
    GLOBAL_INTERNER.lock().unwrap().intern(s)
}

/// Resolve a Symbol using the global interner.
pub fn global_resolve(sym: Symbol) -> String {
    GLOBAL_INTERNER.lock().unwrap().resolve(sym).to_string()
}

/// Look up a string in the global interner.
pub fn global_get(s: &str) -> Option<Symbol> {
    GLOBAL_INTERNER.lock().unwrap().get(s)
}
