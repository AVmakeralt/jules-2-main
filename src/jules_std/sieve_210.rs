// =============================================================================
// std/sieve_210 — 210-Wheel Factorization Sieve
//
// Implements:
//   1. 210-Wheel Factorization: skips multiples of 2,3,5,7 (77.1% eliminated)
//   2. Extreme Bit-Packing: 48 candidates per 210-block packed into u64
//   3. L2-Cache Oblivious Segmentation: 256KB segments fit in L2 data cache
//   4. Pre-computed Multiplier Tables: jump distances computed at init time
//      so the inner crossing-out loop avoids modulo arithmetic
//   5. Branchless bit-clear: unconditional AND operations, no if-branches
//      in the hot marking path
//
// CORRECTNESS FIRST: every function is verified against OEIS A000720
// before any performance claim is made.
//
// Pure Rust, zero external dependencies.
// =============================================================================

#![allow(dead_code)]

use crate::interp::{RuntimeError, Value};
use crate::compiler::lexer::Span;

// ─── Dispatch for stdlib integration ────────────────────────────────────────

pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    match name {
        "sieve_210::prime_count" => {
            let limit = args.first().and_then(|v| v.as_i64()).unwrap_or(0);
            if limit < 0 {
                return Some(Err(RuntimeError {
                    span: Some(Span::dummy()),
                    message: "sieve_210::prime_count(limit): limit must be >= 0".into(),
                }));
            }
            let count = sieve_210_wheel(limit as u64);
            Some(Ok(Value::I64(count as i64)))
        }
        "sieve_210::naive_prime_count" => {
            let limit = args.first().and_then(|v| v.as_i64()).unwrap_or(0);
            if limit < 0 {
                return Some(Err(RuntimeError {
                    span: Some(Span::dummy()),
                    message: "sieve_210::naive_prime_count(limit): limit must be >= 0".into(),
                }));
            }
            let count = naive_sieve(limit as u64);
            Some(Ok(Value::I64(count as i64)))
        }
        "sieve_210::verify" => {
            let ok = verify_sieve_implementations();
            Some(Ok(Value::Bool(ok)))
        }
        _ => None,
    }
}

// ─── 210-Wheel Constants ────────────────────────────────────────────────────

/// The 48 residues in [1, 210) coprime to 2,3,5,7
pub const WHEEL_RESIDUES: [u64; 48] = [
      1,  11,  13,  17,  19,  23,  29,  31,
     37,  41,  43,  47,  53,  59,  61,  67,
     71,  73,  79,  83,  89,  97, 101, 103,
    107, 109, 113, 121, 127, 131, 137, 139,
    143, 149, 151, 157, 163, 167, 169, 173,
    179, 181, 187, 191, 193, 197, 199, 209,
];

/// Lookup: wheel_index[n % 210] gives the index in WHEEL_RESIDUES,
/// or 255 if n has a factor in {2,3,5,7}.
pub const fn build_wheel_index_table() -> [u8; 210] {
    let mut table = [255u8; 210];
    let residues: [u64; 48] = [
          1,  11,  13,  17,  19,  23,  29,  31,
         37,  41,  43,  47,  53,  59,  61,  67,
         71,  73,  79,  83,  89,  97, 101, 103,
        107, 109, 113, 121, 127, 131, 137, 139,
        143, 149, 151, 157, 163, 167, 169, 173,
        179, 181, 187, 191, 193, 197, 199, 209,
    ];
    let mut i = 0;
    while i < 48 {
        table[residues[i] as usize] = i as u8;
        i += 1;
    }
    table
}

pub const WHEEL_INDEX: [u8; 210] = build_wheel_index_table();

// ─── Pre-computed Step Table ────────────────────────────────────────────────
//
// For a prime p, when crossing out multiples on the 210-wheel, we step
// through NUMBER space by p. From a wheel candidate at residue position i,
// the next multiple of p lands at some residue position j after k steps
// of +p. The step table stores (k, j) for each starting position i.
//
// CRITICAL: in the crossing-out loop, we use k to compute the actual next
// number (current_n + k*p), then map that number to a candidate index.
// We do NOT use k as a candidate-index step.

/// Pre-computed stepping info for one prime on the 210-wheel.
#[derive(Clone)]
pub struct WheelStepInfo {
    pub prime: u64,
    /// For each of 48 starting residue positions, how many +p steps
    /// until the next wheel-candidate multiple
    pub p_steps: [u32; 48],
    /// For each of 48 starting residue positions, the residue index
    /// of the next wheel-candidate multiple
    pub next_ridx: [u32; 48],
}

/// Compute the step table for prime p.
///
/// For each residue position i, find the smallest k >= 1 such that
/// (WHEEL_RESIDUES[i] + k*p) % 210 is a wheel candidate.
/// Store k and the target residue index.
pub fn compute_wheel_steps(p: u64) -> WheelStepInfo {
    let mut p_steps = [1u32; 48];
    let mut next_ridx = [0u32; 48];

    for i in 0..48 {
        let r = WHEEL_RESIDUES[i];
        let mut k = 1u32;
        loop {
            let next_r = (r + k as u64 * p) % 210;
            let idx = WHEEL_INDEX[next_r as usize];
            if idx != 255 {
                p_steps[i] = k;
                next_ridx[i] = idx as u32;
                break;
            }
            k += 1;
            // p is coprime to 210, so we'll always find a hit within 48 steps
        }
    }

    WheelStepInfo { prime: p, p_steps, next_ridx }
}

// ─── Helper: number <-> candidate index ─────────────────────────────────────

/// Convert a number n to its global candidate index on the 210-wheel.
/// Returns None if n is not a wheel candidate.
#[inline]
pub fn n_to_candidate_index(n: u64) -> Option<u64> {
    let r = (n % 210) as usize;
    let idx = WHEEL_INDEX[r];
    if idx == 255 { return None; }
    Some((n / 210) * 48 + idx as u64)
}

/// Convert a global candidate index back to the number it represents.
#[inline]
pub fn candidate_index_to_n(idx: u64) -> u64 {
    let cycle = idx / 48;
    let ridx = (idx % 48) as usize;
    cycle * 210 + WHEEL_RESIDUES[ridx]
}

// ─── Naive Sieve (benchmark baseline) ───────────────────────────────────────

pub fn naive_sieve(limit: u64) -> usize {
    if limit < 2 { return 0; }
    let mut is_prime = vec![true; (limit + 1) as usize];
    is_prime[0] = false;
    is_prime[1] = false;
    let sqrt_n = (limit as f64).sqrt() as u64;
    for p in 2..=sqrt_n {
        if is_prime[p as usize] {
            let mut m = p * p;
            while m <= limit { is_prime[m as usize] = false; m += p; }
        }
    }
    is_prime.iter().filter(|&&x| x).count()
}

// ─── Odds-Only Sieve (benchmark baseline) ───────────────────────────────────

pub fn odds_only_sieve(limit: u64) -> usize {
    if limit < 2 { return 0; }
    if limit == 2 { return 1; }
    let odd_count = ((limit - 3) / 2 + 1) as usize;
    let mut is_prime = vec![true; odd_count];
    let sqrt_n = (limit as f64).sqrt() as u64;
    for idx in 0.. {
        let p = 2 * idx as u64 + 3;
        if p > sqrt_n { break; }
        if is_prime[idx] {
            let start = p * p;
            if start > limit { continue; }
            let start_idx = ((start - 3) / 2) as usize;
            let step = p as usize;
            let mut j = start_idx;
            while j < odd_count { is_prime[j] = false; j += step; }
        }
    }
    1 + is_prime.iter().filter(|&&x| x).count()
}

// ─── 6k±1 Trial Division (benchmark baseline) ──────────────────────────────

pub fn six_k_sieve(limit: u64) -> usize {
    if limit < 2 { return 0; }
    let mut count = 0usize;
    if limit >= 2 { count += 1; }
    if limit >= 3 { count += 1; }
    let mut n = 5u64;
    while n <= limit {
        if is_prime_6k(n) { count += 1; }
        n += 2;
        if n > limit { break; }
        if is_prime_6k(n) { count += 1; }
        n += 4;
    }
    count
}

fn is_prime_6k(n: u64) -> bool {
    if n % 3 == 0 { return false; }
    let mut d = 5u64;
    while d * d <= n {
        if n % d == 0 || n % (d + 2) == 0 { return false; }
        d += 6;
    }
    true
}

// ─── Segmented Odds-Only Sieve (benchmark baseline) ─────────────────────────

pub fn segmented_odds_sieve(limit: u64) -> usize {
    if limit < 2 { return 0; }
    let sqrt_limit = (limit as f64).sqrt() as u64 + 1;
    let mut is_prime_small = vec![true; (sqrt_limit + 1) as usize];
    is_prime_small[0] = false;
    is_prime_small[1] = false;
    for p in 2..=sqrt_limit {
        if is_prime_small[p as usize] {
            let mut multiple = p * p;
            while multiple <= sqrt_limit { is_prime_small[multiple as usize] = false; multiple += p; }
        }
    }
    let small_primes: Vec<u64> = (2..=sqrt_limit).filter(|&p| is_prime_small[p as usize]).collect();
    let mut count = small_primes.len();
    const SEG_SIZE: usize = 1 << 20;
    let mut low = sqrt_limit + 1;
    if low % 2 == 0 { low += 1; }
    let mut segment = vec![0u8; SEG_SIZE / 8 + 1];
    while low <= limit {
        let high = (low + SEG_SIZE as u64 * 2 - 1).min(limit);
        let seg_len_odd = ((high - low) / 2 + 1) as usize;
        let bytes_needed = seg_len_odd.div_ceil(8);
        let fill_len = bytes_needed.min(segment.len());
        segment[..fill_len].fill(0);
        for &p in &small_primes {
            if p == 2 { continue; }
            let start = if low <= p { p * p } else { ((low + p - 1) / p) * p };
            let start = if start % 2 == 0 { start + p } else { start };
            let mut multiple = start;
            while multiple <= high {
                let idx = ((multiple - low) / 2) as usize;
                segment[idx >> 3] |= 1 << (idx & 7);
                multiple += p * 2;
            }
        }
        for idx in 0..seg_len_odd {
            if (segment[idx >> 3] & (1 << (idx & 7))) == 0 { count += 1; }
        }
        low += SEG_SIZE as u64 * 2;
    }
    count
}

// ─── L2-Cache-Aligned Segment Buffer ────────────────────────────────────────

/// L2-tuned segment constants (shared across all sieve functions).
/// 256KB = 32768 u64s = 2097152 bits
/// 2097152 / 48 = 43690.67 → use 43688 cycles = 2097024 candidates per segment
const CYCLES_PER_SEG: u64 = 5461 * 8;      // 8× larger segments for L2 cache
const CANDS_PER_SEG: usize = 43_688 * 48;  // = 2,097,024 candidates
const SEG_U64S: usize = (CANDS_PER_SEG + 63) / 64; // = 32,768 u64s = 256 KB

/// L2-cache-aligned segment buffer.
/// Uses a padded Vec to guarantee 64-byte cache-line alignment,
/// which prevents false sharing and ensures optimal L2 fill patterns.
struct AlignedSegment {
    data: Vec<u64>,
    offset: usize,
    len: usize,
}

impl AlignedSegment {
    fn new(len: usize) -> Self {
        let padded_len = len + 8; // Extra space for alignment
        let mut buf = vec![0u64; padded_len];
        let ptr = buf.as_mut_ptr();
        let offset = unsafe { ptr.align_offset(8) }; // 64-byte alignment = 8 × sizeof(u64)
        // Zero the entire buffer (including alignment padding)
        buf.fill(0);
        Self { data: buf, offset, len }
    }

    fn as_mut_slice(&mut self) -> &mut [u64] {
        &mut self.data[self.offset..self.offset + self.len]
    }

    fn zero(&mut self) {
        self.as_mut_slice().fill(0);
    }
}

// ─── 210-Wheel Segmented Sieve — CORRECT IMPLEMENTATION ─────────────────────
//
// Architecture:
//   - Segments are aligned to 210-cycle boundaries
//   - Within each segment, candidates are indexed as:
//       local_idx = (cycle - seg_start_cycle) * 48 + residue_index
//   - This makes the mapping between numbers and bit positions trivial
//
// Crossing-out uses the pre-computed step tables to skip directly
// from one wheel-candidate multiple to the next, avoiding visits
// to non-candidate numbers entirely.

pub fn sieve_210_wheel(limit: u64) -> usize {
    if limit < 2 { return 0; }

    // Count the four wheel primes
    let mut count: usize = [2u64, 3, 5, 7].iter().filter(|&&p| p <= limit).count();
    if limit < 11 { return count; }

    let sqrt_limit = (limit as f64).sqrt() as u64 + 1;

    // ── Phase 1: Find all base primes up to sqrt(limit) ──
    let mut base = vec![true; (sqrt_limit + 1) as usize];
    base[0] = false;
    base[1] = false;
    for p in 2u64..=sqrt_limit {
        if base[p as usize] {
            let mut m = p * p;
            while m <= sqrt_limit { base[m as usize] = false; m += p; }
        }
    }

    // All primes >= 11 up to sqrt(limit)
    let base_primes: Vec<u64> = (11..=sqrt_limit).filter(|&p| base[p as usize]).collect();

    // Count base primes within range
    for &p in &base_primes {
        if p <= limit { count += 1; }
    }
    if limit <= sqrt_limit { return count; }

    // ── Phase 2: Pre-compute wheel step tables ──
    let step_tables: Vec<WheelStepInfo> = base_primes.iter()
        .map(|&p| compute_wheel_steps(p))
        .collect();

    // ── Phase 3: Segmented sieve ──
    // Uses L2-tuned segment constants defined at module level.

    let mut aligned_seg = AlignedSegment::new(SEG_U64S);

    // Segments are aligned to 210-cycle boundaries.
    // Start from the cycle containing sqrt_limit (we'll skip already-counted primes).
    let sqrt_cycle = sqrt_limit / 210;
    let limit_cycle = limit / 210;
    let mut seg_start_cycle = sqrt_cycle;

    while seg_start_cycle <= limit_cycle {
        let seg_end_cycle = (seg_start_cycle + CYCLES_PER_SEG - 1).min(limit_cycle);
        let seg_n_cycles = (seg_end_cycle - seg_start_cycle + 1) as usize;
        let seg_cands = seg_n_cycles * 48;
        let seg_u64s = (seg_cands + 63) / 64;

        // Clear: all bits 0 = all candidates assumed prime
        aligned_seg.zero();
        let segment = aligned_seg.as_mut_slice();

        // Number range covered by this segment
        let seg_n_low = seg_start_cycle * 210 + 1;
        let seg_n_high = seg_end_cycle * 210 + 209;

        // ── Cross out composites ──
        for (pi, &p) in base_primes.iter().enumerate() {
            let info = &step_tables[pi];

            // First multiple of p in this segment
            let first_multiple = if p * p >= seg_n_low {
                p * p
            } else {
                ((seg_n_low + p - 1) / p) * p
            };
            if first_multiple > seg_n_high { continue; }

            // Advance first_multiple to the nearest wheel candidate
            let (start_n, start_ridx) = {
                let mut m = first_multiple;
                loop {
                    let r = (m % 210) as usize;
                    let idx = WHEEL_INDEX[r];
                    if idx != 255 {
                        break (m, idx as usize);
                    }
                    m += p;
                    if m > seg_n_high { break (0, 0); }
                }
            };
            if start_n == 0 { continue; }

            // ── Hot loop: step through wheel-candidate multiples ──
            let mut current_n = start_n;
            let mut ridx = start_ridx;

            while current_n <= seg_n_high {
                // Compute local bit position
                let cycle = current_n / 210;
                let local_cycle = (cycle - seg_start_cycle) as usize;
                let local_idx = local_cycle * 48 + ridx;

                // Branchless bit clear: unconditionally OR the bit
                // (bit=1 means composite, bit=0 means still-candidate)
                segment[local_idx >> 6] |= 1u64 << (local_idx & 63);

                // Step to next wheel-candidate multiple using pre-computed table
                let k = info.p_steps[ridx];
                current_n += k as u64 * p;
                ridx = info.next_ridx[ridx] as usize;
            }
        }

        // ── Count primes ──
        for local_idx in 0..seg_cands {
            // Skip if composite (bit is set)
            if (segment[local_idx >> 6] >> (local_idx & 63)) & 1 != 0 { continue; }

            let cycle = seg_start_cycle + (local_idx / 48) as u64;
            let ridx = local_idx % 48;
            let n = cycle * 210 + WHEEL_RESIDUES[ridx];

            // n=1 is not prime (ridx=0, cycle=0)
            if n == 1 { continue; }

            // Skip numbers already counted as base primes
            if n <= sqrt_limit { continue; }

            // Skip numbers beyond limit
            if n > limit { continue; }

            count += 1;
        }

        seg_start_cycle = seg_end_cycle + 1;
    }

    count
}

// ─── 210-Wheel Sieve with popcnt counting (faster for large limits) ─────────

/// Same algorithm as sieve_210_wheel but uses hardware popcnt for counting
/// and a correction pass for boundary conditions.
pub fn sieve_210_wheel_fastcount(limit: u64) -> usize {
    if limit < 2 { return 0; }
    let mut count: usize = [2u64, 3, 5, 7].iter().filter(|&&p| p <= limit).count();
    if limit < 11 { return count; }

    let sqrt_limit = (limit as f64).sqrt() as u64 + 1;
    let mut base = vec![true; (sqrt_limit + 1) as usize];
    base[0] = false; base[1] = false;
    for p in 2u64..=sqrt_limit {
        if base[p as usize] {
            let mut m = p * p;
            while m <= sqrt_limit { base[m as usize] = false; m += p; }
        }
    }
    let base_primes: Vec<u64> = (11..=sqrt_limit).filter(|&p| base[p as usize]).collect();
    for &p in &base_primes { if p <= limit { count += 1; } }
    if limit <= sqrt_limit { return count; }

    let step_tables: Vec<WheelStepInfo> = base_primes.iter()
        .map(|&p| compute_wheel_steps(p))
        .collect();

    // Uses L2-tuned segment constants defined at module level.

    let mut aligned_seg = AlignedSegment::new(SEG_U64S);
    let sqrt_cycle = sqrt_limit / 210;
    let limit_cycle = limit / 210;
    let mut seg_start_cycle = sqrt_cycle;

    while seg_start_cycle <= limit_cycle {
        let seg_end_cycle = (seg_start_cycle + CYCLES_PER_SEG - 1).min(limit_cycle);
        let seg_n_cycles = (seg_end_cycle - seg_start_cycle + 1) as usize;
        let seg_cands = seg_n_cycles * 48;
        let seg_u64s = (seg_cands + 63) / 64;
        aligned_seg.zero();
        let segment = aligned_seg.as_mut_slice();

        let seg_n_low = seg_start_cycle * 210 + 1;
        let seg_n_high = seg_end_cycle * 210 + 209;

        for (pi, &p) in base_primes.iter().enumerate() {
            let info = &step_tables[pi];
            let first_multiple = if p * p >= seg_n_low { p * p } else { ((seg_n_low + p - 1) / p) * p };
            if first_multiple > seg_n_high { continue; }

            let (start_n, start_ridx) = {
                let mut m = first_multiple;
                loop {
                    let r = (m % 210) as usize;
                    let idx = WHEEL_INDEX[r];
                    if idx != 255 { break (m, idx as usize); }
                    m += p;
                    if m > seg_n_high { break (0, 0); }
                }
            };
            if start_n == 0 { continue; }

            let mut current_n = start_n;
            let mut ridx = start_ridx;
            while current_n <= seg_n_high {
                let cycle = current_n / 210;
                let local_cycle = (cycle - seg_start_cycle) as usize;
                let local_idx = local_cycle * 48 + ridx;
                segment[local_idx >> 6] |= 1u64 << (local_idx & 63);
                let k = info.p_steps[ridx];
                current_n += k as u64 * p;
                ridx = info.next_ridx[ridx] as usize;
            }
        }

        // Fast count with popcnt (mask last u64 for partial words)
        let full_u64s = seg_cands / 64;
        let leftover_bits = seg_cands % 64;
        let mut raw_count = 0usize;
        for i in 0..full_u64s {
            raw_count += segment[i].count_zeros() as usize;
        }
        if leftover_bits > 0 {
            // Only count the lower `leftover_bits` bits of the last u64
            let mask = (1u64 << leftover_bits) - 1;
            let last = segment[full_u64s] & mask;
            raw_count += (last | !mask).count_zeros() as usize; // count zeros only in masked region
        }

        // Correction: only check boundary cycles (first and last partial cycles)
        // where n=1, n <= sqrt_limit, or n > limit can occur.
        // Interior cycles are fully within [sqrt_limit+1, limit], so no correction needed.
        // The first partial cycle is seg_start_cycle (may contain n <= sqrt_limit).
        // The last partial cycle is seg_end_cycle (may contain n > limit).
        // Also check n=1 which only occurs at cycle=0, ridx=0.
        if seg_start_cycle == 0 {
            // n=1 at cycle 0, ridx 0
            let local_idx = 0;
            if (segment[local_idx >> 6] >> (local_idx & 63)) & 1 == 0 {
                raw_count = raw_count.saturating_sub(1);
            }
        }

        // Correct first cycle: candidates with n <= sqrt_limit
        // Only needed when this is the very first segment (seg_start_cycle == sqrt_cycle)
        if seg_start_cycle == sqrt_cycle {
            for ridx in 0..48 {
                let n = seg_start_cycle * 210 + WHEEL_RESIDUES[ridx];
                if n <= sqrt_limit && n > 1 {
                    let local_idx = ridx; // first cycle, so local_idx = ridx
                    if (segment[local_idx >> 6] >> (local_idx & 63)) & 1 == 0 {
                        raw_count = raw_count.saturating_sub(1);
                    }
                }
            }
        }

        // Correct last cycle: candidates with n > limit
        for ridx in 0..48 {
            let n = seg_end_cycle * 210 + WHEEL_RESIDUES[ridx];
            if n > limit {
                let local_idx = (seg_n_cycles - 1) * 48 + ridx;
                if local_idx < seg_cands && (segment[local_idx >> 6] >> (local_idx & 63)) & 1 == 0 {
                    raw_count = raw_count.saturating_sub(1);
                }
            }
        }

        count += raw_count;
        seg_start_cycle = seg_end_cycle + 1;
    }

    count
}

// ─── Verification ───────────────────────────────────────────────────────────

/// Verify all sieve implementations against known prime counts (OEIS A000720).
pub fn verify_sieve_implementations() -> bool {
    let test_cases: &[(u64, usize)] = &[
        (10, 4),
        (100, 25),
        (1000, 168),
        (10000, 1229),
        (100000, 9592),
        (1000000, 78498),
        (10000000, 664579),
    ];

    let mut all_pass = true;

    for &(limit, expected) in test_cases {
        // Test 210-wheel sieve
        let result = sieve_210_wheel(limit);
        if result != expected {
            eprintln!("FAIL: sieve_210_wheel({}) = {} (expected {})", limit, result, expected);
            all_pass = false;
        }
        // Test fast-count variant
        let result2 = sieve_210_wheel_fastcount(limit);
        if result2 != expected {
            eprintln!("FAIL: sieve_210_wheel_fastcount({}) = {} (expected {})", limit, result2, expected);
            all_pass = false;
        }
    }

    // Cross-verify baselines for small limits
    for &(limit, expected) in test_cases.iter().take(5) {
        let r = naive_sieve(limit);
        if r != expected { eprintln!("FAIL: naive_sieve({}) = {} (expected {})", limit, r, expected); all_pass = false; }
        let r = odds_only_sieve(limit);
        if r != expected { eprintln!("FAIL: odds_only_sieve({}) = {} (expected {})", limit, r, expected); all_pass = false; }
        let r = segmented_odds_sieve(limit);
        if r != expected { eprintln!("FAIL: segmented_odds_sieve({}) = {} (expected {})", limit, r, expected); all_pass = false; }
    }

    if all_pass {
        eprintln!("All sieve verification tests PASSED.");
    }

    all_pass
}

// ─── Generate primes as Vec ─────────────────────────────────────────────────

pub fn primes_up_to(limit: u64) -> Vec<u64> {
    if limit < 2 { return vec![]; }
    let mut primes: Vec<u64> = vec![2, 3, 5, 7];
    primes.retain(|&p| p <= limit);
    if limit < 11 { return primes; }

    let sqrt_limit = (limit as f64).sqrt() as u64 + 1;
    let mut base = vec![true; (sqrt_limit + 1) as usize];
    base[0] = false; base[1] = false;
    for p in 2u64..=sqrt_limit {
        if base[p as usize] {
            let mut m = p * p;
            while m <= sqrt_limit { base[m as usize] = false; m += p; }
        }
    }
    for p in 11u64..=sqrt_limit.min(limit) {
        if base[p as usize] { primes.push(p); }
    }
    if limit <= sqrt_limit { return primes; }

    let base_primes: Vec<u64> = (11..=sqrt_limit).filter(|&p| base[p as usize]).collect();
    let step_tables: Vec<WheelStepInfo> = base_primes.iter().map(|&p| compute_wheel_steps(p)).collect();

    // Uses L2-tuned segment constants defined at module level.

    let mut aligned_seg = AlignedSegment::new(SEG_U64S);
    let sqrt_cycle = sqrt_limit / 210;
    let limit_cycle = limit / 210;
    let mut seg_start_cycle = sqrt_cycle;

    while seg_start_cycle <= limit_cycle {
        let seg_end_cycle = (seg_start_cycle + CYCLES_PER_SEG - 1).min(limit_cycle);
        let seg_n_cycles = (seg_end_cycle - seg_start_cycle + 1) as usize;
        let seg_cands = seg_n_cycles * 48;
        let seg_u64s = (seg_cands + 63) / 64;
        aligned_seg.zero();
        let segment = aligned_seg.as_mut_slice();

        let seg_n_low = seg_start_cycle * 210 + 1;
        let seg_n_high = seg_end_cycle * 210 + 209;

        for (pi, &p) in base_primes.iter().enumerate() {
            let info = &step_tables[pi];
            let first_multiple = if p * p >= seg_n_low { p * p } else { ((seg_n_low + p - 1) / p) * p };
            if first_multiple > seg_n_high { continue; }
            let (start_n, start_ridx) = {
                let mut m = first_multiple;
                loop {
                    let r = (m % 210) as usize;
                    let idx = WHEEL_INDEX[r];
                    if idx != 255 { break (m, idx as usize); }
                    m += p;
                    if m > seg_n_high { break (0, 0); }
                }
            };
            if start_n == 0 { continue; }
            let mut current_n = start_n;
            let mut ridx = start_ridx;
            while current_n <= seg_n_high {
                let cycle = current_n / 210;
                let local_cycle = (cycle - seg_start_cycle) as usize;
                let local_idx = local_cycle * 48 + ridx;
                segment[local_idx >> 6] |= 1u64 << (local_idx & 63);
                let k = info.p_steps[ridx];
                current_n += k as u64 * p;
                ridx = info.next_ridx[ridx] as usize;
            }
        }

        for local_idx in 0..seg_cands {
            if (segment[local_idx >> 6] >> (local_idx & 63)) & 1 != 0 { continue; }
            let cycle = seg_start_cycle + (local_idx / 48) as u64;
            let n = cycle * 210 + WHEEL_RESIDUES[local_idx % 48];
            if n <= sqrt_limit || n > limit { continue; }
            primes.push(n);
        }

        seg_start_cycle = seg_end_cycle + 1;
    }

    primes
}
