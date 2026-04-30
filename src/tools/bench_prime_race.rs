use std::hint::black_box;
use std::time::Instant;

use jules::interp::{Interpreter, Value};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fair_limit: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let showcase_limit: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000_000);

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   PRIME GENERATION RACE                                 ║");
    println!("║   Segmented sieve for both sides                         ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("Fair limit: {}", fair_limit);
    println!("Showcase limit: {}", showcase_limit);
    println!();

    println!("─── Round 1: Jules segmented sieve ───");
    let jules_start = Instant::now();
    let jules_count = match jules_segmented_sieve(fair_limit) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[FAIL] Jules segmented sieve failed: {e}");
            std::process::exit(1);
        }
    };
    let jules_runtime = jules_start.elapsed();
    let jules_tp = fair_limit as f64 / jules_runtime.as_secs_f64().max(1e-12);
    println!("  Runtime:        {:.6} s", jules_runtime.as_secs_f64());
    println!("  Primes found:   {}", jules_count);
    println!("  Throughput:     {:.0} numbers/s", jules_tp);
    println!();

    println!("─── Round 2: Rust segmented sieve ───");
    let rust_start = Instant::now();
    let rust_count = rust_segmented_sieve(fair_limit);
    let rust_runtime = rust_start.elapsed();
    let rust_tp = fair_limit as f64 / rust_runtime.as_secs_f64().max(1e-12);
    println!("  Runtime:        {:.6} s", rust_runtime.as_secs_f64());
    println!("  Primes found:   {}", rust_count);
    println!("  Throughput:     {:.0} numbers/s", rust_tp);
    println!();

    println!("─── Round 3: Rust 6k±1 trial division (showcase) ───");
    let trial_start = Instant::now();
    let trial_count = rust_trial_division_6k(showcase_limit);
    let trial_runtime = trial_start.elapsed();
    let trial_tp = showcase_limit as f64 / trial_runtime.as_secs_f64().max(1e-12);
    println!("  Runtime:        {:.3} s", trial_runtime.as_secs_f64());
    println!("  Primes found:   {}", trial_count);
    println!("  Throughput:     {:.0} numbers/s", trial_tp);
    println!();

    let slowdown = jules_runtime.as_secs_f64() / rust_runtime.as_secs_f64().max(1e-12);
    let algo_speedup = rust_tp / trial_tp.max(1e-12);
    println!(
        "Same algorithm slowdown (Jules/Rust segmented): {:.1}x",
        slowdown
    );
    println!(
        "Algorithm speedup (Rust segmented vs Rust trial): {:.1}x",
        algo_speedup
    );
}

/// Jules-side segmented sieve via built-in math::prime_count_segmented.
fn jules_segmented_sieve(limit: usize) -> Result<usize, String> {
    let mut interp = Interpreter::new();
    let seg_size = 1 << 20;
    let out = interp
        .eval_builtin(
            "math::prime_count_segmented",
            vec![Value::I64(limit as i64), Value::I64(seg_size as i64)],
        )
        .map_err(|e| e.message)?;

    match out {
        Value::I32(v) => Ok(v.max(0) as usize),
        Value::I64(v) => Ok(v.max(0) as usize),
        other => Err(format!("unexpected return type: {:?}", other)),
    }
}

fn rust_segmented_sieve(limit: usize) -> usize {
    segmented_sieve_core(limit)
}

fn segmented_sieve_core(limit: usize) -> usize {
    if limit < 2 {
        return 0;
    }

    let sqrt_limit = (limit as f64).sqrt() as usize + 1;
    let mut is_prime_small = vec![true; sqrt_limit + 1];
    is_prime_small[0] = false;
    is_prime_small[1] = false;
    for p in 2..=sqrt_limit {
        if is_prime_small[p] {
            let mut multiple = p * p;
            while multiple <= sqrt_limit {
                is_prime_small[multiple] = false;
                multiple += p;
            }
        }
    }
    let small_primes: Vec<usize> = (2..=sqrt_limit).filter(|&p| is_prime_small[p]).collect();

    let mut count = small_primes.len();
    const SEG_SIZE: usize = 1 << 20;
    let mut segment = vec![0u8; SEG_SIZE / 8 + 1];

    let mut low = sqrt_limit + 1;
    if low % 2 == 0 {
        low += 1;
    }

    while low <= limit {
        let high = (low + SEG_SIZE * 2 - 1).min(limit);
        let seg_len_odd = ((high - low) / 2) + 1;
        let bytes_needed = seg_len_odd.div_ceil(8);
        let clear_len = bytes_needed.min(segment.len());
        for b in segment.iter_mut().take(clear_len) {
            *b = 0;
        }

        for &p in &small_primes {
            if p == 2 {
                continue;
            }
            let start = if low <= p { p * p } else { low.div_ceil(p) * p };
            let start = if start % 2 == 0 { start + p } else { start };
            let mut multiple = start;
            while multiple <= high {
                let idx = (multiple - low) / 2;
                segment[idx >> 3] |= 1 << (idx & 7);
                multiple += p * 2;
            }
        }

        for idx in 0..seg_len_odd {
            if (segment[idx >> 3] & (1 << (idx & 7))) == 0 {
                count += 1;
            }
        }

        low += SEG_SIZE * 2;
    }

    black_box(count)
}

fn rust_trial_division_6k(limit: usize) -> usize {
    if limit < 2 {
        return black_box(0);
    }

    let mut count = 0usize;
    if limit >= 2 {
        count += 1;
    }
    if limit >= 3 {
        count += 1;
    }

    let mut n = 5usize;
    while n <= limit {
        let mut is_prime = true;
        if n % 3 == 0 {
            is_prime = false;
        } else {
            let mut d = 5usize;
            while d * d <= n {
                if n % d == 0 || n % (d + 2) == 0 {
                    is_prime = false;
                    break;
                }
                d += 6;
            }
        }
        if is_prime {
            count += 1;
        }
        n += 2;
        black_box(());
    }
    black_box(count)
}
