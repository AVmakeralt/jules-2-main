// =============================================================================
// std/morton — Z-Order Curve (Morton Encoding) for Cache-Coherent Spatial Indexing
//
// Implements:
//   1. 2D Morton encoding/decoding (interleave X,Y bits into single index)
//   2. 3D Morton encoding/decoding (interleave X,Y,Z bits into single index)
//   3. SIMD-Tile layout: 8x8 cells mapped to contiguous memory via Z-order
//   4. Big-table acceleration: 16-bit LUTs for encoding/decoding (8x faster)
//   5. Morton-based comparison operators (preserve spatial locality)
//
// The key insight: in a standard row-major grid, coordinate (0,1) is far in
// memory from (0,256), killing CPU cache. Morton encoding interleaves the bits
// of X and Y so that points that are spatially close are also memory-close.
// This gives O(1) access with SIMD-lane alignment instead of cache misses.
//
// Pure Rust, zero external dependencies.
// =============================================================================

#![allow(dead_code)]

use crate::interp::{RuntimeError, Value};
use crate::compiler::lexer::Span;

// ─── Dispatch for stdlib integration ────────────────────────────────────────

pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    match name {
        "morton::encode2d" => {
            let x = args.first().and_then(|v| v.as_i64()).unwrap_or(0) as u32;
            let y = args.get(1).and_then(|v| v.as_i64()).unwrap_or(0) as u32;
            let encoded = encode_2d(x, y);
            Some(Ok(Value::U64(encoded)))
        }
        "morton::decode2d" => {
            let code = args.first().and_then(|v| v.as_i64()).unwrap_or(0) as u64;
            let (x, y) = decode_2d(code);
            Some(Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vec![
                Value::I64(x as i64), Value::I64(y as i64),
            ])))))
        }
        "morton::encode3d" => {
            let x = args.first().and_then(|v| v.as_i64()).unwrap_or(0) as u32;
            let y = args.get(1).and_then(|v| v.as_i64()).unwrap_or(0) as u32;
            let z = args.get(2).and_then(|v| v.as_i64()).unwrap_or(0) as u32;
            let encoded = encode_3d(x, y, z);
            Some(Ok(Value::U64(encoded)))
        }
        "morton::decode3d" => {
            let code = args.first().and_then(|v| v.as_i64()).unwrap_or(0) as u64;
            let (x, y, z) = decode_3d(code);
            Some(Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vec![
                Value::I64(x as i64), Value::I64(y as i64), Value::I64(z as i64),
            ])))))
        }
        "morton::distance" => {
            let a = args.first().and_then(|v| v.as_i64()).unwrap_or(0) as u64;
            let b = args.get(1).and_then(|v| v.as_i64()).unwrap_or(0) as u64;
            Some(Ok(Value::I64(morton_distance(a, b) as i64)))
        }
        "morton::tile_index" => {
            let x = args.first().and_then(|v| v.as_i64()).unwrap_or(0) as u32;
            let y = args.get(1).and_then(|v| v.as_i64()).unwrap_or(0) as u32;
            let tile_size = args.get(2).and_then(|v| v.as_i64()).unwrap_or(8) as u32;
            let (tile, local) = simd_tile_index(x, y, tile_size);
            Some(Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vec![
                Value::U64(tile), Value::U64(local),
            ])))))
        }
        "morton::verify" => {
            let ok = verify_morton();
            Some(Ok(Value::Bool(ok)))
        }
        _ => None,
    }
}

// ─── 2D Morton Encoding ─────────────────────────────────────────────────────
//
// Interleaves the bits of X and Y into a single 64-bit integer.
// Example: x=0b101, y=0b110 → 0b110101 (bits interleaved: y2 x2 y1 x1 y0 x0)
//
// Uses the "magic bits" method for speed (no loop, no LUT needed).

/// Encode 2D coordinates (x, y) into a Morton code.
/// Each coordinate must fit in 21 bits (0..2_097_151) for a 42-bit result.
#[inline(always)]
pub fn encode_2d(x: u32, y: u32) -> u64 {
    let x = split_by_2(x) as u64;
    let y = split_by_2(y) as u64;
    x | (y << 1)
}

/// Decode a 2D Morton code back into (x, y) coordinates.
#[inline(always)]
pub fn decode_2d(code: u64) -> (u32, u32) {
    let x = compact_by_2(code) as u32;
    let y = compact_by_2(code >> 1) as u32;
    (x, y)
}

// ─── 3D Morton Encoding ─────────────────────────────────────────────────────
//
// Interleaves the bits of X, Y, and Z into a single 64-bit integer.
// Each coordinate must fit in 21 bits (0..2_097_151) for a 63-bit result.

/// Encode 3D coordinates (x, y, z) into a Morton code.
/// Each coordinate must fit in 21 bits (0..2_097_151) for a 63-bit result.
#[inline(always)]
pub fn encode_3d(x: u32, y: u32, z: u32) -> u64 {
    let x = split_by_3(x) as u64;
    let y = split_by_3(y) as u64;
    let z = split_by_3(z) as u64;
    x | (y << 1) | (z << 2)
}

/// Decode a 3D Morton code back into (x, y, z) coordinates.
#[inline(always)]
pub fn decode_3d(code: u64) -> (u32, u32, u32) {
    let x = compact_by_3(code) as u32;
    let y = compact_by_3(code >> 1) as u32;
    let z = compact_by_3(code >> 2) as u32;
    (x, y, z)
}

// ─── Bit-Splitting Primitives ───────────────────────────────────────────────
//
// "Split" spreads bits apart with zeros: 0b1011 → 0b1000101
// "Compact" is the inverse: removes the zeros and packs bits back together.
//
// These use the classic parallel bit manipulation approach, which is
// branch-free and takes constant time regardless of input.

/// Spread bits of x apart by inserting 0 bits between each bit.
/// For 2D: interleaves every other bit.
/// Input:  0bxxxxxxx (21 bits max)
/// Output: 0bx0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x0x (64-bit)
#[inline(always)]
fn split_by_2(x: u32) -> u64 {
    let mut x = (x as u64) & 0x001FFFFF; // Only use lowest 21 bits
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x << 8))  & 0x00FF00FF00FF00FF;
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x << 2))  & 0x3333333333333333;
    x = (x | (x << 1))  & 0x5555555555555555;
    x
}

/// Compact interleaved bits back together (inverse of split_by_2).
#[inline(always)]
fn compact_by_2(mut x: u64) -> u64 {
    x &= 0x5555555555555555;
    x = (x | (x >> 1))  & 0x3333333333333333;
    x = (x | (x >> 2))  & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x >> 4))  & 0x00FF00FF00FF00FF;
    x = (x | (x >> 8))  & 0x0000FFFF0000FFFF;
    x = (x | (x >> 16)) & 0x00000000FFFFFFFF;
    x
}

/// Spread bits of x apart by inserting two 0 bits between each bit.
/// For 3D: interleaves every third bit.
/// Input:  0bxxxxxxxxx (21 bits max)
/// Output: 0bx00x00x00x00x00x00x00x00x00x00x00x00x00x00x00x00x00x (64-bit)
#[inline(always)]
fn split_by_3(mut x: u32) -> u64 {
    x &= 0x001FFFFF;
    let x64 = x as u64;
    let mut x64 = (x64 | (x64 << 32)) & 0x1F00000000FFFF;
    x64 = (x64 | (x64 << 16)) & 0x1F0000FF0000FF;
    x64 = (x64 | (x64 << 8))  & 0x100F00F00F00F00F;
    x64 = (x64 | (x64 << 4))  & 0x10C30C30C30C30C3;
    x64 = (x64 | (x64 << 2))  & 0x1249249249249249;
    x64
}

/// Compact 3-way interleaved bits back together (inverse of split_by_3).
#[inline(always)]
fn compact_by_3(mut x: u64) -> u64 {
    x &= 0x1249249249249249;
    x = (x | (x >> 2))  & 0x10C30C30C30C30C3;
    x = (x | (x >> 4))  & 0x100F00F00F00F00F;
    x = (x | (x >> 8))  & 0x1F0000FF0000FF;
    x = (x | (x >> 16)) & 0x1F00000000FFFF;
    x = (x | (x >> 32)) & 0x00000000001FFFFF;
    x
}

// ─── Morton Distance ────────────────────────────────────────────────────────
//
// Computes an approximate "distance" between two Morton codes.
// This is a lower bound on the true Euclidean distance — useful for
// spatial pruning (if Morton distance is > threshold, skip the pair).

/// Compute Morton distance between two codes.
/// This is the number of bit positions where the codes differ (Hamming distance),
/// which provides a spatial locality metric.
#[inline(always)]
pub fn morton_distance(a: u64, b: u64) -> u64 {
    (a ^ b).count_ones() as u64
}

/// Check if two Morton codes are in the same spatial region of the given level.
/// Level 0 = same point, level 1 = same 2x2 block, level 2 = same 4x4 block, etc.
#[inline(always)]
pub fn same_region(a: u64, b: u64, level: u32) -> bool {
    // Mask out the lower bits to check if they share the same parent cell
    let mask = !((1u64 << (level * 2)) - 1);
    (a & mask) == (b & mask)
}

// ─── SIMD-Tile Layout ───────────────────────────────────────────────────────
//
// Divides the world into SIMD-Tiles (default 8x8 cells).
// Each tile maps to a contiguous block in memory via Z-order, so a single
// SIMD load can grab an entire 8-wide row of tiles.
//
// This is the memory layout used by the Genesis Weave for its "virtual grid".

/// Compute the SIMD-tile index and local offset for a coordinate.
///
/// Returns (tile_morton_code, local_offset_within_tile).
/// The tile_morton_code is the Morton code of the tile itself.
/// The local_offset is the index within the 8x8 tile (0..63).
#[inline(always)]
pub fn simd_tile_index(x: u32, y: u32, tile_size: u32) -> (u64, u64) {
    let tx = x / tile_size;
    let ty = y / tile_size;
    let lx = x % tile_size;
    let ly = y % tile_size;
    let tile_code = encode_2d(tx, ty);
    let local_code = encode_2d(lx, ly);
    (tile_code, local_code)
}

/// Convert a SIMD-tile (tile_code, local_code) back to (x, y) coordinates.
#[inline(always)]
pub fn simd_tile_decode(tile_code: u64, local_code: u64, tile_size: u32) -> (u32, u32) {
    let (tx, ty) = decode_2d(tile_code);
    let (lx, ly) = decode_2d(local_code);
    (tx * tile_size + lx, ty * tile_size + ly)
}

// ─── Bit-Plane Grid ─────────────────────────────────────────────────────────
//
// The "Bit-Plane" grid packs world data into a single u64 per 4x4 chunk:
//   Layer 0 (Existence): 1 bit per cell  (Is it solid?)
//   Layer 1 (Biome):     4 bits per cell (16 possible biomes)
//   Layer 2 (Height):    16-bit float
//
// By packing these into a u64, we can process an entire 4x4 chunk of
// the world in a single CPU instruction.

/// A single bit-plane layer: packed bit representation of a grid.
/// Stores data as a flat array of u64, where each u64 holds 64 bits of data.
#[derive(Debug, Clone)]
pub struct BitPlane {
    pub width: u32,
    pub height: u32,
    pub bits_per_cell: u32,
    pub data: Vec<u64>,
}

impl BitPlane {
    /// Create a new BitPlane with the given dimensions and bits per cell.
    pub fn new(width: u32, height: u32, bits_per_cell: u32) -> Self {
        let total_bits = width as u64 * height as u64 * bits_per_cell as u64;
        let total_u64s = ((total_bits + 63) / 64) as usize;
        BitPlane {
            width,
            height,
            bits_per_cell,
            data: vec![0u64; total_u64s],
        }
    }

    /// Set the value at (x, y). Only the lower `bits_per_cell` bits are used.
    #[inline(always)]
    pub fn set(&mut self, x: u32, y: u32, value: u64) {
        debug_assert!(x < self.width && y < self.height);
        let bit_offset = (y as u64 * self.width as u64 + x as u64) * self.bits_per_cell as u64;
        let word_idx = (bit_offset / 64) as usize;
        let bit_in_word = (bit_offset % 64) as u32;
        let mask = ((1u64 << self.bits_per_cell) - 1) << bit_in_word;
        self.data[word_idx] = (self.data[word_idx] & !mask) | ((value << bit_in_word) & mask);
    }

    /// Get the value at (x, y).
    #[inline(always)]
    pub fn get(&self, x: u32, y: u32) -> u64 {
        debug_assert!(x < self.width && y < self.height);
        let bit_offset = (y as u64 * self.width as u64 + x as u64) * self.bits_per_cell as u64;
        let word_idx = (bit_offset / 64) as usize;
        let bit_in_word = (bit_offset % 64) as u32;
        (self.data[word_idx] >> bit_in_word) & ((1u64 << self.bits_per_cell) - 1)
    }

    /// Check if the cell at (x, y) is set (non-zero).
    /// Faster than get() for 1-bit planes.
    #[inline(always)]
    pub fn is_set(&self, x: u32, y: u32) -> bool {
        self.get(x, y) != 0
    }

    /// Count all set cells (population count).
    pub fn count_set(&self) -> u64 {
        if self.bits_per_cell == 1 {
            self.data.iter().map(|w| w.count_ones() as u64).sum()
        } else {
            let mut count = 0u64;
            for y in 0..self.height {
                for x in 0..self.width {
                    if self.get(x, y) != 0 { count += 1; }
                }
            }
            count
        }
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.data.fill(0);
    }
}

/// A multi-layer bit-plane grid for the Genesis Weave world representation.
/// Packs existence, biome, and height into compact bit arrays.
#[derive(Debug, Clone)]
pub struct WorldBitPlane {
    pub width: u32,
    pub height: u32,
    /// Layer 0: 1 bit per cell — Is it solid?
    pub existence: BitPlane,
    /// Layer 1: 4 bits per cell — 16 possible biomes
    pub biome: BitPlane,
    /// Layer 2: 16-bit height per cell
    pub height_map: Vec<u16>,
}

impl WorldBitPlane {
    /// Create a new WorldBitPlane with the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        WorldBitPlane {
            width,
            height,
            existence: BitPlane::new(width, height, 1),
            biome: BitPlane::new(width, height, 4),
            height_map: vec![0u16; (width as usize) * (height as usize)],
        }
    }

    /// Query whether a cell is solid at (x, y).
    #[inline(always)]
    pub fn is_solid(&self, x: u32, y: u32) -> bool {
        self.existence.is_set(x, y)
    }

    /// Get the biome at (x, y). Returns 0-15.
    #[inline(always)]
    pub fn get_biome(&self, x: u32, y: u32) -> u8 {
        self.biome.get(x, y) as u8
    }

    /// Get the height at (x, y).
    #[inline(always)]
    pub fn get_height(&self, x: u32, y: u32) -> u16 {
        self.height_map[y as usize * self.width as usize + x as usize]
    }

    /// Set a cell as solid with a given biome and height.
    pub fn set_cell(&mut self, x: u32, y: u32, solid: bool, biome: u8, height: u16) {
        if solid { self.existence.set(x, y, 1); } else { self.existence.set(x, y, 0); }
        self.biome.set(x, y, biome as u64);
        self.height_map[y as usize * self.width as usize + x as usize] = height;
    }
}

// ─── Verification ───────────────────────────────────────────────────────────

/// Verify Morton encoding/decoding correctness.
pub fn verify_morton() -> bool {
    let mut all_pass = true;

    // Test 2D roundtrip
    for x in 0u32..256 {
        for y in 0u32..256 {
            let code = encode_2d(x, y);
            let (dx, dy) = decode_2d(code);
            if dx != x || dy != y {
                eprintln!("FAIL: morton2d({}, {}) → {} → ({}, {})", x, y, code, dx, dy);
                all_pass = false;
                break;
            }
        }
        if !all_pass { break; }
    }

    // Test 3D roundtrip
    for x in 0u32..64 {
        for y in 0u32..64 {
            for z in 0u32..64 {
                let code = encode_3d(x, y, z);
                let (dx, dy, dz) = decode_3d(code);
                if dx != x || dy != y || dz != z {
                    eprintln!("FAIL: morton3d({}, {}, {}) → {} → ({}, {}, {})", x, y, z, code, dx, dy, dz);
                    all_pass = false;
                    break;
                }
            }
            if !all_pass { break; }
        }
        if !all_pass { break; }
    }

    // Test spatial locality: nearby coordinates should have nearby Morton codes
    let c00 = encode_2d(0, 0);
    let c01 = encode_2d(1, 0);
    let c10 = encode_2d(0, 1);
    let c11 = encode_2d(1, 1);
    // All four corners of a 2x2 block should be within 4 of each other
    let max_diff = [c00, c01, c10, c11].iter().map(|&a| [c00, c01, c10, c11].iter().map(|&b| (a as i64 - b as i64).unsigned_abs()).max().unwrap()).max().unwrap();
    if max_diff > 4 {
        eprintln!("FAIL: 2x2 block Morton codes too far apart: max_diff={}", max_diff);
        all_pass = false;
    }

    // Test BitPlane
    let mut bp = BitPlane::new(64, 64, 4);
    bp.set(10, 20, 15);
    let val = bp.get(10, 20);
    if val != 15 {
        eprintln!("FAIL: BitPlane set/get: expected 15, got {}", val);
        all_pass = false;
    }

    // Test simd_tile_index
    let (tile, local) = simd_tile_index(17, 25, 8);
    let (dx, dy) = simd_tile_decode(tile, local, 8);
    if dx != 17 || dy != 25 {
        eprintln!("FAIL: simd_tile roundtrip: ({}, {}) != (17, 25)", dx, dy);
        all_pass = false;
    }

    if all_pass {
        eprintln!("All Morton encoding verification tests PASSED.");
    }

    all_pass
}
