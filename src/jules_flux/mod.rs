//! Jules-Flux: Zero-Copy, Tiled-Graph Compiler
//! 
//! A custom XLA/Triton equivalent for Jules that integrates with the existing
//! Phase 3 JIT system and Prophetic Prefetch Engine for maximum performance.
//!
//! Uses the existing JIT infrastructure with AMX kernel generation.

pub mod tile;
pub mod amx_kernels;

pub use tile::*;
pub use amx_kernels::*;

// Re-export from existing Jules prefetch engine
pub use crate::runtime::threading::prophetic_prefetch::{
    PrefetchLevel, PrefetchHint, AccessPattern
};

use crate::jit::phase3_jit::{compile_ops};
use crate::interp::{Instr, CompiledFn};

/// Size of the static sanctuary for tile operations (4.5 MB)
pub const SANCTUARY_SIZE: usize = 4_718_592;

/// Cache line size (64 bytes)
pub const CACHE_LINE_SIZE: usize = 64;

/// Default tile dimensions for AMX
pub const TILE_ROWS: usize = 16;
pub const TILE_COLS: usize = 64;

/// Compile a fused tile operation using the existing JIT
pub fn compile_tile_kernel(name: &str, ops: &[Instr]) -> Option<CompiledFn> {
    compile_ops(name, ops)
}

/// Flux error types
#[derive(Debug, Clone)]
pub enum FluxError {
    TileOutOfBounds,
    AmxNotSupported,
    KernelCompilationFailed(String),
}

impl std::fmt::Display for FluxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FluxError::TileOutOfBounds => write!(f, "Tile operation out of bounds"),
            FluxError::AmxNotSupported => write!(f, "AMX not supported on this CPU"),
            FluxError::KernelCompilationFailed(msg) => write!(f, "JIT compilation failed: {}", msg),
        }
    }
}

impl std::error::Error for FluxError {}
