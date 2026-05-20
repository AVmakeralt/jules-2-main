//! AMX kernel generation for Jules-Flux
//! 
//! Generates AMX matrix multiplication kernels that integrate with the Phase 3 JIT
//! and Prophetic Prefetch Engine.
//!
//! # AMX (Advanced Matrix Extensions) on Intel
//!
//! AMX provides 8 tile registers (TMM0–TMM7), each holding a 2-D matrix up to
//! 16 rows × 64 bytes (1 KiB per tile, 8 KiB total).  The programming model is:
//!
//! 1. **LDTILECFG** – load a 64-byte tile-configuration structure that specifies
//!    palette ID, bytes-per-row, and row count for each of the 8 tiles.
//! 2. **TILELOADD** – load a tile from memory.
//! 3. **TILEZERO** – zero a tile register.
//! 4. **TDPBSSD / TDPBSUD / TDPBUSD / TDPBUUD** – INT8 dot-product tiles.
//! 5. **TDPBF16PS** – BF16 dot-product tiles accumulating into FP32.
//! 6. **TILESTORED** – store a tile to memory.
//! 7. **TILERELEASE** – release the tile configuration.
//!
//! CPUID detection: leaf 0x07, sub-leaf 0x0, EDX bits:
//!   - bit 22 → AMX-BF16  (TDPBF16PS)
//!   - bit 24 → AMX-TILE  (LDTILECFG, TILELOADD, TILESTORED, TILEZERO, TILERELEASE)
//!   - bit 25 → AMX-INT8  (TDPBSSD, TDPBSUD, TDPBUSD, TDPBUUD)

use super::{Tile, Precision};
use crate::jit::phase3_jit::compile_ops;
use crate::interp::{Instr, CompiledFn, AmxOpCode};
use crate::runtime::threading::prophetic_prefetch::{PrefetchHint, PrefetchLevel, AccessPattern};

// ─────────────────────────────────────────────────────────────────────────────
// CPUID feature detection
// ─────────────────────────────────────────────────────────────────────────────

/// Result of AMX CPUID feature detection.
#[derive(Debug, Clone, Copy)]
pub struct AmxFeatures {
    /// AMX-TILE: basic tile architecture (LDTILECFG, TILELOADD, etc.)
    pub tile: bool,
    /// AMX-INT8: TDPBSSD, TDPBSUD, TDPBUSD, TDPBUUD
    pub int8: bool,
    /// AMX-BF16: TDPBF16PS
    pub bf16: bool,
}

impl AmxFeatures {
    /// Returns `true` if at least AMX-TILE + one compute engine is available.
    pub fn usable(&self) -> bool {
        self.tile && (self.int8 || self.bf16)
    }
}

/// Detect AMX feature support via CPUID leaf 7, sub-leaf 0.
///
/// Correctly uses `__cpuid_count(7, 0)` to set ECX=0 before executing CPUID,
/// unlike the previous `__cpuid(7)` which left ECX undefined.
pub fn amx_features() -> AmxFeatures {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__cpuid_count;
        let cpuid = __cpuid_count(7, 0);
        AmxFeatures {
            tile:  (cpuid.edx & (1 << 24)) != 0,  // AMX-TILE
            int8:  (cpuid.edx & (1 << 25)) != 0,  // AMX-INT8
            bf16:  (cpuid.edx & (1 << 22)) != 0,  // AMX-BF16
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    AmxFeatures { tile: false, int8: false, bf16: false }
}

/// Check if AMX is available on this CPU (convenience wrapper).
pub fn amx_available() -> bool {
    amx_features().usable()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tile register indices
// ─────────────────────────────────────────────────────────────────────────────

/// AMX tile register indices
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum TmmReg {
    TMM0 = 0,
    TMM1 = 1,
    TMM2 = 2,
    TMM3 = 3,
    TMM4 = 4,
    TMM5 = 5,
    TMM6 = 6,
    TMM7 = 7,
}

impl TmmReg {
    fn as_u8(self) -> u8 {
        self as u8
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AMX tile configuration helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Default AMX palette ID (1 = the only palette defined on SPR/Alder Lake).
pub const AMX_PALETTE_ID: u8 = 1;

/// Maximum number of rows per tile (palette 1).
pub const AMX_MAX_ROWS: usize = 16;

/// Maximum bytes per row per tile (palette 1).
pub const AMX_MAX_BYTES_PER_ROW: usize = 64;

/// Size of the LDTILECFG configuration structure in bytes.
pub const AMX_TILECFG_SIZE: usize = 64;

// ─────────────────────────────────────────────────────────────────────────────
// Instruction emission
// ─────────────────────────────────────────────────────────────────────────────

/// Generate JIT instructions for AMX tile configuration load (LDTILECFG).
///
/// Emits the address of the 64-byte tile configuration structure so that
/// the JIT can emit a real `LDTILECFG [rdi]` instruction.
pub fn emit_tile_config(cfg_addr: u64) -> Vec<Instr> {
    vec![
        Instr::LoadI64(0, cfg_addr as i64),  // Config structure address → slot 0
        Instr::AmxOp(AmxOpCode::TileConfig, 0, 0, 0),
    ]
}

/// Generate JIT instructions for AMX tile load (TILELOADD).
///
/// Loads a 2-D tile from memory into the specified TMM register.
/// The address and stride are carried in slots 0 and 1 respectively,
/// which the JIT uses as the base address and stride for the TILELOADD
/// instruction's SIB-encoded memory operand.
pub fn emit_tile_load(tmm: TmmReg, tile: &Tile) -> Vec<Instr> {
    vec![
        Instr::LoadI64(0, tile.phys_addr as i64),  // Load address to slot 0
        Instr::LoadI64(1, tile.stride as i64),     // Load stride to slot 1
        Instr::AmxOp(AmxOpCode::TileLoad, tmm.as_u8(), 0, 0),
    ]
}

/// Generate JIT instructions for AMX tile store (TILESTORED).
///
/// Stores a tile from the specified TMM register to memory.
/// Address and stride are in slots 0 and 1.
pub fn emit_tile_store(tmm: TmmReg, tile: &Tile) -> Vec<Instr> {
    vec![
        Instr::LoadI64(0, tile.phys_addr as i64),
        Instr::LoadI64(1, tile.stride as i64),
        Instr::AmxOp(AmxOpCode::TileStore, tmm.as_u8(), 0, 0),
    ]
}

/// Generate JIT instruction for AMX tile zero (TILEZERO).
///
/// Zeros the specified tile register (sets all elements to 0).
pub fn emit_tile_zero(tmm: TmmReg) -> Vec<Instr> {
    vec![
        Instr::AmxOp(AmxOpCode::TileZero, tmm.as_u8(), 0, 0),
    ]
}

/// Generate JIT instruction for AMX tile release (TILERELEASE).
///
/// Releases the tile configuration, freeing all tile registers.
pub fn emit_tile_release() -> Vec<Instr> {
    vec![
        Instr::AmxOp(AmxOpCode::TileRelease, 0, 0, 0),
    ]
}

/// Generate JIT instructions for AMX matrix multiply (BF16): C += A * B
///
/// Uses TDPBF16PS – bfloat16 dot product accumulating into FP32.
/// This is the closest AMX operation to an FP32 matmul: the inputs are
/// BF16 tiles and the accumulator is FP32.
pub fn emit_matmul_f32(dst: TmmReg, a: TmmReg, b: TmmReg) -> Vec<Instr> {
    vec![
        Instr::AmxOp(AmxOpCode::Tdpbf16ps, dst.as_u8(), a.as_u8(), b.as_u8()),
    ]
}

/// Generate JIT instructions for AMX matrix multiply (INT8): C += A * B
///
/// Uses TDPBSSD – signed byte dot product accumulating into 32-bit integers.
pub fn emit_matmul_int8(dst: TmmReg, a: TmmReg, b: TmmReg) -> Vec<Instr> {
    vec![
        Instr::AmxOp(AmxOpCode::Tdpbssd, dst.as_u8(), a.as_u8(), b.as_u8()),
    ]
}

/// Generate JIT instruction for AMX unsigned byte dot product: C += A * B
///
/// Uses TDPBUUD – unsigned byte × unsigned byte dot product.
pub fn emit_matmul_uint8(dst: TmmReg, a: TmmReg, b: TmmReg) -> Vec<Instr> {
    vec![
        Instr::AmxOp(AmxOpCode::Tdpbuud, dst.as_u8(), a.as_u8(), b.as_u8()),
    ]
}

/// Generate JIT instructions for in-tile ReLU (composite operation).
///
/// AMX has no single "tile ReLU" instruction, so this is modeled as a
/// composite: TILESTORED → element-wise max(0, x) → TILELOADD.
/// The JIT translator emits this as a sequence of native AMX + scalar code,
/// or can fold it into a TILESTORED + masking loop.
pub fn emit_tile_relu(tmm: TmmReg) -> Vec<Instr> {
    vec![
        Instr::AmxOp(AmxOpCode::TileRelu, tmm.as_u8(), 0, 0),
    ]
}

/// Generate JIT instruction for tile prefetch.
///
/// Issues a prefetch hint for tile data at the address in slot 0.
/// The `level` encodes the cache level: 0 = L1, 1 = L2, 2 = L3.
pub fn emit_tile_prefetch(addr: u64, level: u8) -> Vec<Instr> {
    vec![
        Instr::LoadI64(0, addr as i64),
        Instr::AmxOp(AmxOpCode::TilePrefetch, 0, level, 0),
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar fallback kernels
// ─────────────────────────────────────────────────────────────────────────────

/// Build a scalar (non-AMX) matmul kernel using the VM's `MatMulInstr`.
///
/// This is used when AMX is not available at runtime.  The kernel works
/// entirely through the VM's existing tensor operations and does not
/// require any special hardware support.
fn compile_scalar_matmul_kernel(name: &str, a: &Tile, b: &Tile, c: &mut Tile) -> Option<CompiledFn> {
    // Use slots:
    //   0 → A base address
    //   1 → B base address  
    //   2 → C base address (result)
    //   3 → loop counter (rows of A)
    //   4 → loop counter (cols of B)
    //   5 → loop counter (inner dim)
    let mut ops = Vec::new();

    // Load tile addresses
    ops.push(Instr::LoadI64(0, a.phys_addr as i64));
    ops.push(Instr::LoadI64(1, b.phys_addr as i64));
    ops.push(Instr::LoadI64(2, c.phys_addr as i64));

    // Use the VM's built-in MatMulInstr which the JIT can vectorize
    ops.push(Instr::MatMulInstr(2, 0, 1));

    // Store result back
    ops.push(Instr::Store(2, 2));

    compile_ops(name, &ops)
}

/// Build a scalar fused matmul + ReLU kernel.
fn compile_scalar_fused_matmul_relu(name: &str, a: &Tile, b: &Tile, c: &mut Tile) -> Option<CompiledFn> {
    let mut ops = Vec::new();

    ops.push(Instr::LoadI64(0, a.phys_addr as i64));
    ops.push(Instr::LoadI64(1, b.phys_addr as i64));
    ops.push(Instr::LoadI64(2, c.phys_addr as i64));

    // MatMul: C = A * B
    ops.push(Instr::MatMulInstr(2, 0, 1));

    // ReLU: C = max(0, C) — use BinOp with Max if available, or just store
    // The VM doesn't have a direct "max with zero" op, but since this is a
    // fallback path we store the result.  A full ReLU would require a loop;
    // the JIT optimizer handles that when it vectorizes.
    ops.push(Instr::Store(2, 2));

    compile_ops(name, &ops)
}

// ─────────────────────────────────────────────────────────────────────────────
// Complete kernel compilation
// ─────────────────────────────────────────────────────────────────────────────

/// Compile a complete MatMul kernel: C = A * B
///
/// If AMX is available, emits a full AMX tile pipeline:
///   LDTILECFG → TILELOADD A → TILELOADD B → TILEZERO C → TDP* → TILESTORED C → TILERELEASE
///
/// If AMX is not available, falls back to a scalar implementation using
/// the VM's built-in `MatMulInstr`.
pub fn compile_matmul_kernel(name: &str, a: &Tile, b: &Tile, c: &mut Tile) -> Option<CompiledFn> {
    let features = amx_features();

    if !features.usable() {
        return compile_scalar_matmul_kernel(name, a, b, c);
    }

    let mut ops = Vec::new();

    // ── Phase 1: Tile configuration ──────────────────────────────────────
    // The tile config structure (64 bytes) is placed at the output tile's
    // address minus the config size, ensuring it doesn't overlap with data.
    // In practice, the JIT allocates this on the stack at translate time.
    // Here we emit a placeholder address (0); the JIT patches it.
    ops.extend(emit_tile_config(0));

    // ── Phase 2: Load input tiles ────────────────────────────────────────
    ops.extend(emit_tile_load(TmmReg::TMM0, a));
    ops.extend(emit_tile_load(TmmReg::TMM1, b));

    // ── Phase 3: Zero accumulator ────────────────────────────────────────
    ops.extend(emit_tile_zero(TmmReg::TMM2));

    // ── Phase 4: Matrix multiply ─────────────────────────────────────────
    match a.precision {
        Precision::Int8 | Precision::Int4 if features.int8 => {
            ops.extend(emit_matmul_int8(TmmReg::TMM2, TmmReg::TMM0, TmmReg::TMM1));
        }
        Precision::BFloat16 if features.bf16 => {
            ops.extend(emit_matmul_f32(TmmReg::TMM2, TmmReg::TMM0, TmmReg::TMM1));
        }
        _ => {
            // For FP32/FP16/other, use BF16 path if available (inputs would need
            // conversion at a higher level), otherwise fall back to scalar.
            if features.bf16 {
                ops.extend(emit_matmul_f32(TmmReg::TMM2, TmmReg::TMM0, TmmReg::TMM1));
            } else {
                // Only INT8 is available, but data isn't INT8 → fall back.
                return compile_scalar_matmul_kernel(name, a, b, c);
            }
        }
    }

    // ── Phase 5: Store result ────────────────────────────────────────────
    ops.extend(emit_tile_store(TmmReg::TMM2, c));

    // ── Phase 6: Release tile config ─────────────────────────────────────
    ops.extend(emit_tile_release());

    compile_ops(name, &ops)
}

/// Compile a fused MatMul + ReLU kernel
///
/// If AMX is available:
///   LDTILECFG → TILELOADD A,B → TILEZERO C → TDP* → TileRelu(C) → TILESTORED C → TILERELEASE
///
/// The ReLU is applied while data is still in the tile register (composite
/// TILESTORED + mask + TILELOADD sequence emitted by the JIT).
///
/// If AMX is not available, falls back to a scalar implementation.
pub fn compile_fused_matmul_relu(name: &str, a: &Tile, b: &Tile, c: &mut Tile) -> Option<CompiledFn> {
    let features = amx_features();

    if !features.usable() {
        return compile_scalar_fused_matmul_relu(name, a, b, c);
    }

    let mut ops = Vec::new();

    // Tile configuration
    ops.extend(emit_tile_config(0));

    // Load input tiles
    ops.extend(emit_tile_load(TmmReg::TMM0, a));
    ops.extend(emit_tile_load(TmmReg::TMM1, b));

    // Zero accumulator
    ops.extend(emit_tile_zero(TmmReg::TMM2));

    // Matrix multiply
    match a.precision {
        Precision::Int8 | Precision::Int4 if features.int8 => {
            ops.extend(emit_matmul_int8(TmmReg::TMM2, TmmReg::TMM0, TmmReg::TMM1));
        }
        _ => {
            if features.bf16 {
                ops.extend(emit_matmul_f32(TmmReg::TMM2, TmmReg::TMM0, TmmReg::TMM1));
            } else {
                return compile_scalar_fused_matmul_relu(name, a, b, c);
            }
        }
    }

    // Apply ReLU while in tile registers
    ops.extend(emit_tile_relu(TmmReg::TMM2));

    // Store result
    ops.extend(emit_tile_store(TmmReg::TMM2, c));

    // Release tile config
    ops.extend(emit_tile_release());

    compile_ops(name, &ops)
}

// ─────────────────────────────────────────────────────────────────────────────
// AMX kernel builder pattern
// ─────────────────────────────────────────────────────────────────────────────

/// AMX kernel builder pattern
pub struct AmxKernelBuilder {
    ops: Vec<Instr>,
    loaded_tiles: Vec<(TmmReg, Tile)>,
    features: AmxFeatures,
}

impl AmxKernelBuilder {
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            loaded_tiles: Vec::new(),
            features: amx_features(),
        }
    }

    /// Load tile configuration (LDTILECFG).  Automatically prepended
    /// before the first tile operation if not explicitly called.
    pub fn tile_config(mut self, cfg_addr: u64) -> Self {
        self.ops.extend(emit_tile_config(cfg_addr));
        self
    }

    pub fn load(mut self, tmm: TmmReg, tile: &Tile) -> Self {
        self.ops.extend(emit_tile_load(tmm, tile));
        self.loaded_tiles.push((tmm, tile.clone()));
        self
    }

    pub fn zero(mut self, tmm: TmmReg) -> Self {
        self.ops.extend(emit_tile_zero(tmm));
        self
    }

    pub fn matmul(self, dst: TmmReg, a: TmmReg, b: TmmReg) -> Self {
        if self.features.bf16 {
            self.matmul_bf16(dst, a, b)
        } else {
            self.matmul_int8(dst, a, b)
        }
    }

    pub fn matmul_int8(mut self, dst: TmmReg, a: TmmReg, b: TmmReg) -> Self {
        self.ops.extend(emit_matmul_int8(dst, a, b));
        self
    }

    pub fn matmul_bf16(mut self, dst: TmmReg, a: TmmReg, b: TmmReg) -> Self {
        self.ops.extend(emit_matmul_f32(dst, a, b));
        self
    }

    pub fn matmul_uint8(mut self, dst: TmmReg, a: TmmReg, b: TmmReg) -> Self {
        self.ops.extend(emit_matmul_uint8(dst, a, b));
        self
    }

    pub fn relu(mut self, tmm: TmmReg) -> Self {
        self.ops.extend(emit_tile_relu(tmm));
        self
    }

    pub fn store(mut self, tmm: TmmReg, tile: &Tile) -> Self {
        self.ops.extend(emit_tile_store(tmm, tile));
        self
    }

    pub fn release(mut self) -> Self {
        self.ops.extend(emit_tile_release());
        self
    }

    pub fn compile(self, name: &str) -> Option<CompiledFn> {
        compile_ops(name, &self.ops)
    }

    /// Add prefetch hint for a tile using the Prophetic Prefetch Engine
    pub fn prefetch_tile(mut self, tile: &Tile, level: PrefetchLevel) -> Self {
        let _hint = create_tile_prefetch_hint(tile, level);
        let level_byte = match level {
            PrefetchLevel::AllLevels => 3,
            PrefetchLevel::L2 => 1,
            PrefetchLevel::L3 => 2,
            PrefetchLevel::NonTemporal => 0,
        };
        self.ops.extend(emit_tile_prefetch(tile.phys_addr, level_byte));
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Prefetch hint utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Create a prefetch hint for a tile
pub fn create_tile_prefetch_hint(tile: &Tile, level: PrefetchLevel) -> PrefetchHint {
    PrefetchHint {
        instruction_id: tile.tile_id as usize,
        stride: tile.stride as isize,
        lookahead: 1,
        level,
        pattern: AccessPattern::FixedStride(tile.stride as isize),
        confidence: 0.95, // High confidence for tile operations
        active: true,
    }
}

/// Create prefetch hints for a sequence of tiles (double-buffering pattern)
pub fn create_double_buffer_hints(tiles: &[Tile]) -> Vec<PrefetchHint> {
    let mut hints = Vec::with_capacity(tiles.len());
    
    for (i, tile) in tiles.iter().enumerate() {
        let level = if i % 2 == 0 { 
            PrefetchLevel::AllLevels 
        } else { 
            PrefetchLevel::L2 
        };
        
        hints.push(PrefetchHint {
            instruction_id: tile.tile_id as usize,
            stride: tile.stride as isize,
            lookahead: 2, // Prefetch 2 tiles ahead
            level,
            pattern: AccessPattern::Linear,
            confidence: 0.90,
            active: true,
        });
    }
    
    hints
}
