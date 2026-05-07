//! Tile descriptors for Jules-Flux
//! 
//! Static tile descriptors that map to physical memory for zero-copy operations.

use super::{SANCTUARY_SIZE, CACHE_LINE_SIZE};

/// Precision types for tile data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Int4,
    Int8,
    BFloat16,
    Float16,
    Float32,
}

impl Precision {
    pub fn size_bytes(&self) -> usize {
        match self {
            Precision::Int4 => 1,     // 2 per byte
            Precision::Int8 => 1,
            Precision::BFloat16 => 2,
            Precision::Float16 => 2,
            Precision::Float32 => 4,
        }
    }
}

/// Tile descriptor - references a region of physical memory
#[derive(Debug, Clone)]
pub struct Tile {
    pub phys_addr: u64,
    pub rows: usize,
    pub cols: usize,
    pub stride: usize,
    pub precision: Precision,
    pub tile_id: u64,
}

impl Tile {
    pub fn new(phys_addr: u64, rows: usize, cols: usize, precision: Precision) -> Self {
        Self {
            phys_addr,
            rows,
            cols,
            stride: cols * precision.size_bytes(),
            precision,
            tile_id: phys_addr, // Use address as ID
        }
    }

    pub fn size_bytes(&self) -> usize {
        self.rows * self.cols * self.precision.size_bytes()
    }

    pub fn cache_lines(&self) -> usize {
        (self.size_bytes() + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE
    }

    pub fn end_addr(&self) -> u64 {
        self.phys_addr + self.size_bytes() as u64
    }

    pub fn fits_in_sanctuary(&self, base: u64) -> bool {
        self.phys_addr >= base && self.end_addr() <= base + SANCTUARY_SIZE as u64
    }
}

/// Sanctuary - 4.5 MB reserved physical memory region
pub struct Sanctuary {
    base: u64,
    cursor: u64,
}

impl Sanctuary {
    pub fn new(base: u64) -> Self {
        Self { base, cursor: base }
    }

    pub fn allocate(&mut self, rows: usize, cols: usize, precision: Precision) -> Option<Tile> {
        let tile = Tile::new(self.cursor, rows, cols, precision);
        if !tile.fits_in_sanctuary(self.base) {
            return None;
        }
        self.cursor = tile.end_addr();
        Some(tile)
    }

    pub fn reset(&mut self) {
        self.cursor = self.base;
    }

    pub fn base(&self) -> u64 { self.base }
    pub fn used(&self) -> usize { (self.cursor - self.base) as usize }
    pub fn available(&self) -> usize { SANCTUARY_SIZE - self.used() }
}
