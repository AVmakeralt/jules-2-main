//! AMX kernel generation for Jules-Flux
//! 
//! Generates AMX matrix multiplication kernels that integrate with the Phase 3 JIT
//! and Prophetic Prefetch Engine.

use super::{FluxError, Tile, Precision, TILE_ROWS, TILE_COLS, CACHE_LINE_SIZE};
use crate::jit::phase3_jit::{compile_ops, NativeCode};
use crate::interp::{Instr, CompiledFn};
use crate::runtime::threading::prophetic_prefetch::{PrefetchHint, PrefetchLevel, AccessPattern};

/// Check if AMX is available on this CPU
pub fn amx_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__cpuid;
        unsafe {
            let cpuid = __cpuid(7);
            (cpuid.edx & (1 << 24)) != 0  // AMX-TILE bit
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

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

/// Generate JIT instructions for AMX tile load
/// 
/// This creates bytecode that the Phase 3 JIT will compile to native AMX instructions
pub fn emit_tile_load(tmm: TmmReg, tile: &Tile) -> Vec<Instr> {
    // The JIT will inline the actual AMX instruction
    vec![
        Instr::LoadI(0, tile.phys_addr as i64),  // Load address to slot 0
        Instr::LoadI(1, tile.stride as i64),     // Load stride to slot 1
        Instr::NativeCall(0x100 + tmm as u8, vec![0, 1]), // AMX load call
    ]
}

/// Generate JIT instructions for AMX tile store
pub fn emit_tile_store(tmm: TmmReg, tile: &Tile) -> Vec<Instr> {
    vec![
        Instr::LoadI(0, tile.phys_addr as i64),
        Instr::LoadI(1, tile.stride as i64),
        Instr::NativeCall(0x200 + tmm as u8, vec![0, 1]),
    ]
}

/// Generate JIT instructions for AMX matrix multiply: C = A * B
pub fn emit_matmul_f32(dst: TmmReg, a: TmmReg, b: TmmReg) -> Vec<Instr> {
    // TDPF32PS - dot product of FP32 tiles
    vec![
        Instr::NativeCall(0x300 | ((dst as u8) << 4) | ((a as u8) << 2) | (b as u8), vec![]),
    ]
}

/// Generate JIT instructions for AMX matrix multiply: C = A * B (INT8)
pub fn emit_matmul_int8(dst: TmmReg, a: TmmReg, b: TmmReg) -> Vec<Instr> {
    // TDPBSSD - dot product of signed bytes to signed dwords
    vec![
        Instr::NativeCall(0x400 | ((dst as u8) << 4) | ((a as u8) << 2) | (b as u8), vec![]),
    ]
}

/// Compile a complete MatMul kernel: C = A * B
pub fn compile_matmul_kernel(name: &str, a: &Tile, b: &Tile, c: &mut Tile) -> Option<CompiledFn> {
    if !amx_available() {
        return None;
    }

    let mut ops = Vec::new();

    // Load tiles A and B
    ops.extend(emit_tile_load(TmmReg::TMM0, a));
    ops.extend(emit_tile_load(TmmReg::TMM1, b));
    
    // Zero accumulator C
    ops.push(Instr::NativeCall(0x500, vec![])); // tilezero TMM2
    
    // Matrix multiply based on precision
    match a.precision {
        Precision::Float32 => {
            ops.extend(emit_matmul_f32(TmmReg::TMM2, TmmReg::TMM0, TmmReg::TMM1));
        }
        Precision::Int8 | Precision::Int4 => {
            ops.extend(emit_matmul_int8(TmmReg::TMM2, TmmReg::TMM0, TmmReg::TMM1));
        }
        _ => {
            // Convert other types to f32 path
            ops.extend(emit_matmul_f32(TmmReg::TMM2, TmmReg::TMM0, TmmReg::TMM1));
        }
    }
    
    // Store result
    ops.extend(emit_tile_store(TmmReg::TMM2, c));
    
    // Release AMX
    ops.push(Instr::NativeCall(0x600, vec![])); // tilerelease

    compile_ops(name, &ops)
}

/// Compile a fused MatMul + ReLU kernel
pub fn compile_fused_matmul_relu(name: &str, a: &Tile, b: &Tile, c: &mut Tile) -> Option<CompiledFn> {
    let mut ops = Vec::new();

    ops.extend(emit_tile_load(TmmReg::TMM0, a));
    ops.extend(emit_tile_load(TmmReg::TMM1, b));
    ops.push(Instr::NativeCall(0x500, vec![])); // tilezero
    ops.extend(emit_matmul_f32(TmmReg::TMM2, TmmReg::TMM0, TmmReg::TMM1));
    
    // Apply ReLU while in registers (native call with RELU flag)
    ops.push(Instr::NativeCall(0x700, vec![])); // tile_relu TMM2
    
    ops.extend(emit_tile_store(TmmReg::TMM2, c));
    ops.push(Instr::NativeCall(0x600, vec![])); // tilerelease

    compile_ops(name, &ops)
}

/// AMX kernel builder pattern
pub struct AmxKernelBuilder {
    ops: Vec<Instr>,
    loaded_tiles: Vec<(TmmReg, Tile)>,
}

impl AmxKernelBuilder {
    pub fn new() -> Self {
        Self { ops: Vec::new(), loaded_tiles: Vec::new() }
    }

    pub fn load(mut self, tmm: TmmReg, tile: &Tile) -> Self {
        self.ops.extend(emit_tile_load(tmm, tile));
        self.loaded_tiles.push((tmm, tile.clone()));
        self
    }

    pub fn zero(mut self, tmm: TmmReg) -> Self {
        self.ops.push(Instr::NativeCall(0x500 | (tmm as u8) as i64, vec![]));
        self
    }

    pub fn matmul(mut self, dst: TmmReg, a: TmmReg, b: TmmReg) -> Self {
        self.ops.extend(emit_matmul_f32(dst, a, b));
        self
    }

    pub fn matmul_int8(mut self, dst: TmmReg, a: TmmReg, b: TmmReg) -> Self {
        self.ops.extend(emit_matmul_int8(dst, a, b));
        self
    }

    pub fn relu(mut self, tmm: TmmReg) -> Self {
        self.ops.push(Instr::NativeCall(0x700 | (tmm as u8) as i64, vec![]));
        self
    }

    pub fn store(mut self, tmm: TmmReg, tile: &Tile) -> Self {
        self.ops.extend(emit_tile_store(tmm, tile));
        self
    }

    pub fn release(mut self) -> Self {
        self.ops.push(Instr::NativeCall(0x600, vec![]));
        self
    }

    pub fn compile(self, name: &str) -> Option<CompiledFn> {
        compile_ops(name, &self.ops)
    }

    /// Add prefetch hint for a tile using the Prophetic Prefetch Engine
    pub fn prefetch_tile(mut self, tile: &Tile, level: PrefetchLevel) -> Self {
        let hint = create_tile_prefetch_hint(tile, level);
        // Emit prefetch as native call with hint data
        self.ops.push(Instr::LoadI(0, tile.phys_addr as i64));
        self.ops.push(Instr::NativeCall(0x800 | (level as i64), vec![0]));
        self
    }
}

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
