// =============================================================================
// std/alloc — Jules Standard Library: Memory Allocators
//
// Arena (bump) allocator, pool allocator, slab allocator.
// Pure Rust, zero external dependencies.
// =============================================================================

#![allow(dead_code)]

use crate::interp::{RuntimeError, Value};
use crate::lexer::Span;

macro_rules! rt_err {
    ($msg:expr) => {
        RuntimeError { span: Some(Span::dummy()), message: $msg.to_string() }
    };
}

fn i64_arg(args: &[Value], i: usize) -> Option<i64> {
    args.get(i).and_then(|v| v.as_i64())
}

fn f64_arg(args: &[Value], i: usize) -> Option<f64> {
    args.get(i).and_then(|v| v.as_f64())
}

// ─── Arena (Bump) Allocator ──────────────────────────────────────────────────

pub struct Arena {
    buffer: Vec<u8>,
    cursor: usize,
    alignment: usize,
}

impl Arena {
    pub fn new(capacity: usize, alignment: usize) -> Self {
        let mut buf = Vec::with_capacity(capacity);
        buf.resize(capacity, 0);
        Arena { buffer: buf, cursor: 0, alignment }
    }

    pub fn alloc(&mut self, size: usize) -> Option<usize> {
        let align = self.alignment.max(1);
        let aligned_cursor = (self.cursor + align - 1) & !(align - 1);
        let next = aligned_cursor.checked_add(size)?;
        if next > self.buffer.len() { return None; }
        let offset = aligned_cursor;
        self.cursor = next;
        Some(offset)
    }

    pub fn alloc_zero(&mut self, size: usize) -> Option<usize> {
        let offset = self.alloc(size)?;
        for i in offset..offset + size {
            self.buffer[i] = 0;
        }
        Some(offset)
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    pub fn used(&self) -> usize { self.cursor }
    pub fn capacity(&self) -> usize { self.buffer.len() }
    pub fn free(&self) -> usize { self.buffer.len() - self.cursor }

    pub fn write_u8(&mut self, offset: usize, val: u8) -> bool {
        if offset + 1 <= self.buffer.len() { self.buffer[offset] = val; true } else { false }
    }
    pub fn write_u16(&mut self, offset: usize, val: u16) -> bool {
        if offset + 2 <= self.buffer.len() {
            self.buffer[offset..offset+2].copy_from_slice(&val.to_le_bytes()); true
        } else { false }
    }
    pub fn write_u32(&mut self, offset: usize, val: u32) -> bool {
        if offset + 4 <= self.buffer.len() {
            self.buffer[offset..offset+4].copy_from_slice(&val.to_le_bytes()); true
        } else { false }
    }
    pub fn write_u64(&mut self, offset: usize, val: u64) -> bool {
        if offset + 8 <= self.buffer.len() {
            self.buffer[offset..offset+8].copy_from_slice(&val.to_le_bytes()); true
        } else { false }
    }
    pub fn write_f32(&mut self, offset: usize, val: f32) -> bool {
        if offset + 4 <= self.buffer.len() {
            self.buffer[offset..offset+4].copy_from_slice(&val.to_le_bytes()); true
        } else { false }
    }
    pub fn write_f64(&mut self, offset: usize, val: f64) -> bool {
        if offset + 8 <= self.buffer.len() {
            self.buffer[offset..offset+8].copy_from_slice(&val.to_le_bytes()); true
        } else { false }
    }

    pub fn read_u8(&self, offset: usize) -> Option<u8> {
        if offset + 1 <= self.buffer.len() { Some(self.buffer[offset]) } else { None }
    }
    pub fn read_u32(&self, offset: usize) -> Option<u32> {
        if offset + 4 <= self.buffer.len() {
            Some(u32::from_le_bytes([
                self.buffer[offset], self.buffer[offset+1], self.buffer[offset+2], self.buffer[offset+3]
            ]))
        } else { None }
    }
    pub fn read_f32(&self, offset: usize) -> Option<f32> {
        if offset + 4 <= self.buffer.len() {
            Some(f32::from_le_bytes([
                self.buffer[offset], self.buffer[offset+1], self.buffer[offset+2], self.buffer[offset+3]
            ]))
        } else { None }
    }
    pub fn read_f64(&self, offset: usize) -> Option<f64> {
        if offset + 8 <= self.buffer.len() {
            Some(f64::from_le_bytes([
                self.buffer[offset], self.buffer[offset+1], self.buffer[offset+2], self.buffer[offset+3],
                self.buffer[offset+4], self.buffer[offset+5], self.buffer[offset+6], self.buffer[offset+7],
            ]))
        } else { None }
    }
}

// ─── Pool Allocator (fixed-size blocks) ──────────────────────────────────────

pub struct Pool {
    blocks: Vec<Vec<u8>>,
    block_size: usize,
    free_list: Vec<usize>,
}

impl Pool {
    pub fn new(block_size: usize, capacity: usize) -> Self {
        let mut blocks = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            blocks.push(vec![0; block_size]);
        }
        let free_list: Vec<usize> = (0..capacity).collect();
        Pool { blocks, block_size, free_list }
    }

    pub fn alloc(&mut self) -> Option<usize> {
        self.free_list.pop()
    }

    pub fn free(&mut self, idx: usize) {
        if idx < self.blocks.len() {
            self.free_list.push(idx);
        }
    }

    pub fn write(&mut self, idx: usize, offset: usize, data: &[u8]) -> bool {
        if idx < self.blocks.len() && offset + data.len() <= self.block_size {
            self.blocks[idx][offset..offset+data.len()].copy_from_slice(data);
            true
        } else { false }
    }

    pub fn read(&self, idx: usize, offset: usize, len: usize) -> Option<Vec<u8>> {
        if idx < self.blocks.len() && offset + len <= self.block_size {
            Some(self.blocks[idx][offset..offset+len].to_vec())
        } else { None }
    }

    pub fn used(&self) -> usize { self.blocks.len() - self.free_list.len() }
    pub fn capacity(&self) -> usize { self.blocks.len() }
}

// ─── Slab Allocator (growable, indexed by stable key) ────────────────────────

pub struct Slab {
    entries: Vec<Option<Vec<u8>>>,
    free_list: Vec<usize>,
}

impl Slab {
    pub fn new() -> Self {
        Slab { entries: Vec::new(), free_list: Vec::new() }
    }

    pub fn insert(&mut self, data: Vec<u8>) -> usize {
        if let Some(idx) = self.free_list.pop() {
            self.entries[idx] = Some(data);
            idx
        } else {
            let idx = self.entries.len();
            self.entries.push(Some(data));
            idx
        }
    }

    pub fn remove(&mut self, idx: usize) -> bool {
        if idx < self.entries.len() && self.entries[idx].is_some() {
            self.entries[idx] = None;
            self.free_list.push(idx);
            true
        } else { false }
    }

    pub fn get(&self, idx: usize) -> Option<&[u8]> {
        self.entries.get(idx).and_then(|e| e.as_ref()).map(|v| v.as_slice())
    }

    pub fn len(&self) -> usize {
        self.entries.iter().filter(|e| e.is_some()).count()
    }

    pub fn capacity(&self) -> usize { self.entries.len() }
}

// ─── Thread-local registries ────────────────────────────────────────────────

thread_local! {
    static ARENAS: std::cell::RefCell<Vec<Arena>> = std::cell::RefCell::new(Vec::new());
    static POOLS: std::cell::RefCell<Vec<Pool>> = std::cell::RefCell::new(Vec::new());
    static SLABS: std::cell::RefCell<Vec<Slab>> = std::cell::RefCell::new(Vec::new());
}

// ─── Builtin dispatch ───────────────────────────────────────────────────────

pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    match name {
        // ── Arena ────────────────────────────────────────────────────────
        "alloc::arena_new" => {
            let cap = i64_arg(args, 0).unwrap_or(65536) as usize;
            let align = i64_arg(args, 1).unwrap_or(16) as usize;
            ARENAS.with(|a| {
                let mut a = a.borrow_mut();
                a.push(Arena::new(cap, align));
                Some(Ok(Value::U64(a.len() as u64)))
            })
        }
        "alloc::arena_alloc" => {
            if args.len() < 2 { return Some(Err(rt_err!("arena_alloc() requires handle, size"))); }
            if let (Some(h), Some(size)) = (i64_arg(args,0), i64_arg(args,1)) {
                ARENAS.with(|a| {
                    let mut a = a.borrow_mut();
                    if let Some(arena) = a.get_mut(h as usize - 1) {
                        match arena.alloc(size as usize) {
                            Some(offset) => Some(Ok(Value::U64(offset as u64))),
                            None => Some(Err(rt_err!("arena_alloc(): out of memory"))),
                        }
                    } else { Some(Err(rt_err!("arena_alloc(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("arena_alloc() requires handle, size"))) }
        }
        "alloc::arena_reset" => {
            if let Some(h) = i64_arg(args, 0) {
                ARENAS.with(|a| {
                    let mut a = a.borrow_mut();
                    if let Some(arena) = a.get_mut(h as usize - 1) {
                        arena.reset(); Some(Ok(Value::Unit))
                    } else { Some(Err(rt_err!("arena_reset(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("arena_reset() requires handle"))) }
        }
        "alloc::arena_used" => {
            if let Some(h) = i64_arg(args, 0) {
                ARENAS.with(|a| {
                    let a = a.borrow();
                    if let Some(arena) = a.get(h as usize - 1) {
                        Some(Ok(Value::U64(arena.used() as u64)))
                    } else { Some(Err(rt_err!("arena_used(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("arena_used() requires handle"))) }
        }
        "alloc::arena_write_f32" => {
            if args.len() < 3 { return Some(Err(rt_err!("arena_write_f32() requires handle, offset, value"))); }
            if let (Some(h), Some(off), Some(val)) = (i64_arg(args,0), i64_arg(args,1), f64_arg(args,2)) {
                ARENAS.with(|a| {
                    let mut a = a.borrow_mut();
                    if let Some(arena) = a.get_mut(h as usize - 1) {
                        if arena.write_f32(off as usize, val as f32) {
                            Some(Ok(Value::Unit))
                        } else { Some(Err(rt_err!("arena_write_f32(): out of bounds"))) }
                    } else { Some(Err(rt_err!("arena_write_f32(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("arena_write_f32() requires handle, offset, value"))) }
        }
        "alloc::arena_read_f32" => {
            if args.len() < 2 { return Some(Err(rt_err!("arena_read_f32() requires handle, offset"))); }
            if let (Some(h), Some(off)) = (i64_arg(args,0), i64_arg(args,1)) {
                ARENAS.with(|a| {
                    let a = a.borrow();
                    if let Some(arena) = a.get(h as usize - 1) {
                        match arena.read_f32(off as usize) {
                            Some(v) => Some(Ok(Value::F32(v))),
                            None => Some(Err(rt_err!("arena_read_f32(): out of bounds"))),
                        }
                    } else { Some(Err(rt_err!("arena_read_f32(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("arena_read_f32() requires handle, offset"))) }
        }

        // ── Pool ─────────────────────────────────────────────────────────
        "alloc::pool_new" => {
            let bs = i64_arg(args, 0).unwrap_or(256) as usize;
            let cap = i64_arg(args, 1).unwrap_or(1024) as usize;
            POOLS.with(|p| {
                let mut p = p.borrow_mut();
                p.push(Pool::new(bs, cap));
                Some(Ok(Value::U64(p.len() as u64)))
            })
        }
        "alloc::pool_alloc" => {
            if let Some(h) = i64_arg(args, 0) {
                POOLS.with(|p| {
                    let mut p = p.borrow_mut();
                    if let Some(pool) = p.get_mut(h as usize - 1) {
                        match pool.alloc() {
                            Some(idx) => Some(Ok(Value::U64(idx as u64))),
                            None => Some(Err(rt_err!("pool_alloc(): pool exhausted"))),
                        }
                    } else { Some(Err(rt_err!("pool_alloc(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("pool_alloc() requires handle"))) }
        }
        "alloc::pool_free" => {
            if args.len() < 2 { return Some(Err(rt_err!("pool_free() requires handle, index"))); }
            if let (Some(h), Some(idx)) = (i64_arg(args,0), i64_arg(args,1)) {
                POOLS.with(|p| {
                    let mut p = p.borrow_mut();
                    if let Some(pool) = p.get_mut(h as usize - 1) {
                        pool.free(idx as usize);
                        Some(Ok(Value::Unit))
                    } else { Some(Err(rt_err!("pool_free(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("pool_free() requires handle, index"))) }
        }
        "alloc::pool_used" => {
            if let Some(h) = i64_arg(args, 0) {
                POOLS.with(|p| {
                    let p = p.borrow();
                    if let Some(pool) = p.get(h as usize - 1) {
                        Some(Ok(Value::U64(pool.used() as u64)))
                    } else { Some(Err(rt_err!("pool_used(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("pool_used() requires handle"))) }
        }

        // ── Slab ─────────────────────────────────────────────────────────
        "alloc::slab_new" => Some(Ok(Value::U64({
            SLABS.with(|s| { let mut s = s.borrow_mut(); s.push(Slab::new()); s.len() as u64 })
        }))),
        "alloc::slab_insert" => {
            if args.len() < 2 { return Some(Err(rt_err!("slab_insert() requires handle, data"))); }
            if let (Some(h), Value::Str(data)) = (i64_arg(args,0), &args[1]) {
                SLABS.with(|s| {
                    let mut s = s.borrow_mut();
                    if let Some(slab) = s.get_mut(h as usize - 1) {
                        Some(Ok(Value::U64(slab.insert(data.as_bytes().to_vec()) as u64)))
                    } else { Some(Err(rt_err!("slab_insert(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("slab_insert() requires handle, string"))) }
        }
        "alloc::slab_remove" => {
            if args.len() < 2 { return Some(Err(rt_err!("slab_remove() requires handle, key"))); }
            if let (Some(h), Some(k)) = (i64_arg(args,0), i64_arg(args,1)) {
                SLABS.with(|s| {
                    let mut s = s.borrow_mut();
                    if let Some(slab) = s.get_mut(h as usize - 1) {
                        if slab.remove(k as usize) { Some(Ok(Value::Unit)) }
                        else { Some(Err(rt_err!("slab_remove(): invalid key"))) }
                    } else { Some(Err(rt_err!("slab_remove(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("slab_remove() requires handle, key"))) }
        }
        "alloc::slab_len" => {
            if let Some(h) = i64_arg(args, 0) {
                SLABS.with(|s| {
                    let s = s.borrow();
                    if let Some(slab) = s.get(h as usize - 1) {
                        Some(Ok(Value::U64(slab.len() as u64)))
                    } else { Some(Err(rt_err!("slab_len(): invalid handle"))) }
                })
            } else { Some(Err(rt_err!("slab_len() requires handle"))) }
        }

        _ => None,
    }
}
