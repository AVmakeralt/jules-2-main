// =============================================================================
// std/random — Jules Standard Library: Random Number Generation
//
// RNGs: PCG32, SplitMix64, XorShift64
// Distributions: uniform (int/float), normal (Box-Muller), bernoulli,
//                weighted choice, shuffle, sample
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

fn f64_arg(args: &[Value], i: usize) -> Option<f64> {
    args.get(i).and_then(|v| v.as_f64())
}

fn i64_arg(args: &[Value], i: usize) -> Option<i64> {
    args.get(i).and_then(|v| v.as_i64())
}

// ─── PCG32 (Permuted Congruential Generator) ────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    pub fn new(seed: u64, seq: u64) -> Self {
        let mut rng = Pcg32 { state: 0, inc: (seq << 1) | 1 };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xor = ((old >> 18) ^ old) >> 27;
        let rot = (old >> 59) as u32;
        ((xor >> rot) | (xor << (rot.wrapping_neg() & 31))) as u32
    }

    pub fn next_u64(&mut self) -> u64 {
        ((self.next_u32() as u64) << 32) | (self.next_u32() as u64)
    }

    /// Float in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        let v = self.next_u64();
        // Use the top 53 bits for full precision
        (v >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Float in [0, 1)
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32) / (1u64 << 24) as f32
    }

    /// Integer in [lo, hi)
    pub fn next_range(&mut self, lo: u32, hi: u32) -> u32 {
        if hi <= lo { return lo; }
        let range = hi - lo;
        if range & (range - 1) == 0 {
            lo + (self.next_u32() & (range - 1))
        } else {
            let max_valid = u32::MAX - (u32::MAX % range);
            loop {
                let v = self.next_u32();
                if v < max_valid {
                    return lo + (v % range);
                }
            }
        }
    }

    /// Standard normal (mean=0, stddev=1) via Box-Muller
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Boolean with probability p
    pub fn next_bool(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }
}

// ─── SplitMix64 ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ─── Global RNG state (thread-local) ─────────────────────────────────────────

thread_local! {
    static GLOBAL_PCG: std::cell::RefCell<Pcg32> = std::cell::RefCell::new(
        Pcg32::new(42, 54)
    );
}

/// Dispatch a random:: builtin.
pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    match name {
        // ── Seeding ──────────────────────────────────────────────────────
        "random::seed" => {
            if let Some(s) = i64_arg(args, 0) {
                GLOBAL_PCG.with(|r| *r.borrow_mut() = Pcg32::new(s as u64, 54));
                Some(Ok(Value::Unit))
            } else {
                let seed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(42);
                GLOBAL_PCG.with(|r| *r.borrow_mut() = Pcg32::new(seed, 54));
                Some(Ok(Value::Unit))
            }
        }

        // ── Uniform float [0, 1) ─────────────────────────────────────────
        "random::rand" | "random::rand_f32" => {
            Some(Ok(Value::F32(GLOBAL_PCG.with(|r| r.borrow_mut().next_f32()))))
        }
        "random::rand_f64" => {
            Some(Ok(Value::F64(GLOBAL_PCG.with(|r| r.borrow_mut().next_f64()))))
        }

        // ── Uniform float [lo, hi) ───────────────────────────────────────
        "random::rand_range" => {
            if args.len() < 2 {
                return Some(Err(rt_err!("random::rand_range() requires lo, hi")));
            }
            if let (Some(lo), Some(hi)) = (f64_arg(args, 0), f64_arg(args, 1)) {
                let v = GLOBAL_PCG.with(|r| r.borrow_mut().next_f64() * (hi - lo) + lo);
                Some(Ok(Value::F64(v)))
            } else { Some(Err(rt_err!("random::rand_range() requires two numbers"))) }
        }

        // ── Uniform int [lo, hi) ─────────────────────────────────────────
        "random::rand_int" => {
            if args.len() < 2 {
                return Some(Err(rt_err!("random::rand_int() requires lo, hi")));
            }
            if let (Some(lo), Some(hi)) = (i64_arg(args, 0), i64_arg(args, 1)) {
                if lo >= hi {
                    return Some(Err(rt_err!("random::rand_int(): lo >= hi")));
                }
                let v = GLOBAL_PCG.with(|r| r.borrow_mut().next_range(lo as u32, hi as u32));
                Some(Ok(Value::I32(v as i32)))
            } else { Some(Err(rt_err!("random::rand_int() requires two integers"))) }
        }

        // ── Normal distribution ──────────────────────────────────────────
        "random::rand_normal" => {
            let mean = f64_arg(args, 0).unwrap_or(0.0);
            let stddev = f64_arg(args, 1).unwrap_or(1.0);
            let v = GLOBAL_PCG.with(|r| r.borrow_mut().next_normal() * stddev + mean);
            Some(Ok(Value::F64(v)))
        }

        // ── Bernoulli ────────────────────────────────────────────────────
        "random::rand_bool" => {
            let p = f64_arg(args, 0).unwrap_or(0.5);
            Some(Ok(Value::Bool(GLOBAL_PCG.with(|r| r.borrow_mut().next_bool(p)))))
        }

        // ── Choice from array ────────────────────────────────────────────
        "random::choice" => {
            if args.is_empty() {
                return Some(Err(rt_err!("random::choice() requires an array")));
            }
            if let Value::Array(arr) = &args[0] {
                let arr = arr.lock().unwrap();
                if arr.is_empty() {
                    return Some(Err(rt_err!("random::choice(): empty array")));
                }
                let idx = GLOBAL_PCG.with(|r| r.borrow_mut().next_range(0, arr.len() as u32)) as usize;
                Some(Ok(arr[idx].clone()))
            } else { Some(Err(rt_err!("random::choice() requires an array"))) }
        }

        // ── Weighted choice ──────────────────────────────────────────────
        "random::weighted_choice" => {
            if args.len() < 2 {
                return Some(Err(rt_err!("random::weighted_choice() requires items, weights")));
            }
            if let (Value::Array(items_arr), Value::Array(weights_arr)) = (&args[0], &args[1]) {
                let items = items_arr.lock().unwrap();
                let weights = weights_arr.lock().unwrap();
                if items.is_empty() || weights.is_empty() || items.len() != weights.len() {
                    return Some(Err(rt_err!("random::weighted_choice(): items/weights length mismatch")));
                }
                let total: f64 = weights.iter().filter_map(|v| v.as_f64()).sum();
                if total <= 0.0 {
                    return Some(Err(rt_err!("random::weighted_choice(): total weight <= 0")));
                }
                let mut pick = GLOBAL_PCG.with(|r| r.borrow_mut().next_f64() * total);
                for (i, w) in weights.iter().enumerate() {
                    if let Some(wv) = w.as_f64() {
                        pick -= wv;
                        if pick <= 0.0 {
                            return Some(Ok(items[i].clone()));
                        }
                    }
                }
                Some(Ok(items.last().cloned().unwrap_or(Value::Unit)))
            } else { Some(Err(rt_err!("random::weighted_choice() requires two arrays"))) }
        }

        // ── Shuffle (Fisher-Yates) ───────────────────────────────────────
        "random::shuffle" => {
            if let Value::Array(arr) = &args[0] {
                let mut arr = arr.lock().unwrap();
                for i in (1..arr.len()).rev() {
                    let j = GLOBAL_PCG.with(|r| r.borrow_mut().next_range(0, (i + 1) as u32)) as usize;
                    arr.swap(i, j);
                }
                Some(Ok(Value::Unit))
            } else { Some(Err(rt_err!("random::shuffle() requires an array"))) }
        }

        // ── Sample n without replacement ─────────────────────────────────
        "random::sample" => {
            if args.len() < 2 {
                return Some(Err(rt_err!("random::sample() requires array, count")));
            }
            if let (Value::Array(arr), Some(n)) = (&args[0], i64_arg(args, 1)) {
                let arr = arr.lock().unwrap();
                let n = n as usize;
                if n > arr.len() {
                    return Some(Err(rt_err!("random::sample(): n > array length")));
                }
                // Floyd's sampling algorithm
                let mut selected = std::collections::HashSet::with_capacity(n);
                for j in (arr.len() - n)..arr.len() {
                    let r = GLOBAL_PCG.with(|r| r.borrow_mut().next_range(0, (j + 1) as u32)) as usize;
                    if selected.contains(&r) {
                        selected.insert(j);
                    } else {
                        selected.insert(r);
                    }
                }
                let result: Vec<Value> = selected.into_iter().map(|i| arr[i].clone()).collect();
                Some(Ok(Value::Array(std::sync::Arc::new(std::sync::Mutex::new(result)))))
            } else { Some(Err(rt_err!("random::sample() requires array + int"))) }
        }

        // ── PCG32 constructor ────────────────────────────────────────────
        "random::pcg32" => {
            let seed = i64_arg(args, 0).unwrap_or(42) as u64;
            let seq = i64_arg(args, 1).unwrap_or(54) as u64;
            let handle = register_rng(Pcg32::new(seed, seq));
            Some(Ok(Value::U64(handle)))
        }
        "random::pcg32_next" => {
            if let Some(h) = i64_arg(args, 0) {
                let v = RNG_REGISTRY.with(|r| {
                    let mut v = r.borrow_mut();
                    let idx = h as usize;
                    if idx < v.len() { Some(Value::U32(v[idx - 1].next_u32())) } else { None }
                });
                v.map(Ok).or_else(|| Some(Err(rt_err!("random::pcg32_next(): invalid handle"))))
            } else { Some(Err(rt_err!("random::pcg32_next() requires handle"))) }
        }
        "random::pcg32_next_f32" => {
            if let Some(h) = i64_arg(args, 0) {
                let v = RNG_REGISTRY.with(|r| {
                    let mut v = r.borrow_mut();
                    let idx = h as usize;
                    if idx < v.len() { Some(Value::F32(v[idx - 1].next_f32())) } else { None }
                });
                v.map(Ok).or_else(|| Some(Err(rt_err!("random::pcg32_next_f32(): invalid handle"))))
            } else { Some(Err(rt_err!("random::pcg32_next_f32() requires handle"))) }
        }
        "random::pcg32_range" => {
            if args.len() < 3 { return Some(Err(rt_err!("pcg32_range() requires handle, lo, hi"))); }
            if let (Some(h), Some(lo), Some(hi)) = (i64_arg(args,0), i64_arg(args,1), i64_arg(args,2)) {
                let v = RNG_REGISTRY.with(|r| {
                    let mut v = r.borrow_mut();
                    let idx = h as usize;
                    if idx < v.len() { Some(Value::U32(v[idx - 1].next_range(lo as u32, hi as u32))) } else { None }
                });
                v.map(Ok).or_else(|| Some(Err(rt_err!("pcg32_range(): invalid handle"))))
            } else { Some(Err(rt_err!("pcg32_range() requires handle, lo, hi"))) }
        }

        _ => None,
    }
}

// ─── RNG Registry (for user-created RNG handles) ────────────────────────────

thread_local! {
    static RNG_REGISTRY: std::cell::RefCell<Vec<Pcg32>> =
        std::cell::RefCell::new(Vec::new());
}

fn register_rng(rng: Pcg32) -> u64 {
    RNG_REGISTRY.with(|r| {
        let mut v = r.borrow_mut();
        let handle = (v.len() + 1) as u64;
        v.push(rng);
        handle
    })
}


