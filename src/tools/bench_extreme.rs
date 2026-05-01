// =============================================================================
// bench_extreme.rs — Comprehensive Benchmark: Extreme Algorithms
//
// Benchmarks ALL sieve and PRNG implementations against each other:
//
// Sieve Algorithms:
//   1. Naive Sieve of Eratosthenes
//   2. Odds-Only Sieve
//   3. 6k±1 Trial Division
//   4. Segmented Odds-Only Sieve (current repo implementation)
//   5. 210-Wheel Segmented Sieve (production implementation)
//
// PRNG Algorithms:
//   1. XorShift64
//   2. PCG32
//   3. Mersenne Twister MT19937-64
//   4. Squares Counter-Based (scalar)
//   5. Shishiua Counter-Based (scalar)
//   6. SimdPrng4 (4-lane batched)
//   7. SimdPrng8 (8-lane batched)
//
// Usage:
//   cargo run --bin bench-extreme [sieve_limit] [prng_count]
//   Defaults: sieve_limit=10_000_000, prng_count=100_000_000
// =============================================================================

use std::hint::black_box;
use std::time::Instant;

use jules::jules_std::sieve_210::*;
use jules::jules_std::prng_simd::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sieve_limit: u64 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000);
    let prng_count: u64 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000_000);

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║   EXTREME ALGORITHM BENCHMARK SUITE                              ║");
    println!("║   210-Wheel Sieve + SIMD Counter-Based PRNG                      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Sieve limit:  {:>15}", format_number(sieve_limit));
    println!("PRNG count:   {:>15}", format_number(prng_count));
    println!();

    // ── Verification ──────────────────────────────────────────────────────────
    println!("═══ VERIFICATION ═══════════════════════════════════════════════════");
    let sieve_ok = verify_sieve_implementations();
    println!("  Sieve implementations: {}", if sieve_ok { "PASS ✓" } else { "FAIL ✗" });

    let prng_quality = test_prng_quality();
    println!("  PRNG quality test:     {}", if prng_quality { "PASS ✓" } else { "FAIL ✗" });

    let counter_ok = test_counter_based();
    println!("  Counter-based test:    {}", if counter_ok { "PASS ✓" } else { "FAIL ✗" });

    let simd_ok = test_simd8_consistency();
    println!("  SIMD8 consistency:     {}", if simd_ok { "PASS ✓" } else { "FAIL ✗" });
    println!();

    if !sieve_ok || !prng_quality || !counter_ok || !simd_ok {
        eprintln!("WARNING: Some verification tests failed. Results may be incorrect.");
        eprintln!();
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  SIEVE BENCHMARKS
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══ SIEVE BENCHMARKS ═══════════════════════════════════════════════");
    println!();

    let mut sieve_results: Vec<(&str, f64, usize)> = Vec::new();

    // ── 1. Naive Sieve ──────────────────────────────────────────────────────
    if sieve_limit <= 100_000_000 {
        print!("  Running Naive Sieve..............");
        let start = Instant::now();
        let count = naive_sieve(sieve_limit);
        let elapsed = start.elapsed().as_secs_f64();
        println!(" {:>8.4} s  ({} primes)", elapsed, format_number(count as u64));
        sieve_results.push(("Naive", elapsed, count));
    } else {
        println!("  Naive Sieve...................... SKIPPED (too slow for limit > 100M)");
    }

    // ── 2. Odds-Only Sieve ──────────────────────────────────────────────────
    if sieve_limit <= 1_000_000_000 {
        print!("  Running Odds-Only Sieve..........");
        let start = Instant::now();
        let count = odds_only_sieve(sieve_limit);
        let elapsed = start.elapsed().as_secs_f64();
        println!(" {:>8.4} s  ({} primes)", elapsed, format_number(count as u64));
        sieve_results.push(("Odds-Only", elapsed, count));
    } else {
        println!("  Odds-Only Sieve.................. SKIPPED (too slow for limit > 1B)");
    }

    // ── 3. 6k±1 Trial Division ──────────────────────────────────────────────
    if sieve_limit <= 10_000_000 {
        print!("  Running 6k±1 Trial Division......");
        let start = Instant::now();
        let count = six_k_sieve(sieve_limit);
        let elapsed = start.elapsed().as_secs_f64();
        println!(" {:>8.4} s  ({} primes)", elapsed, format_number(count as u64));
        sieve_results.push(("6k±1", elapsed, count));
    } else {
        println!("  6k±1 Trial Division.............. SKIPPED (too slow for limit > 10M)");
    }

    // ── 4. Segmented Odds-Only Sieve ────────────────────────────────────────
    print!("  Running Segmented Odds-Only......");
    let start = Instant::now();
    let count = segmented_odds_sieve(sieve_limit);
    let elapsed = start.elapsed().as_secs_f64();
    println!(" {:>8.4} s  ({} primes)", elapsed, format_number(count as u64));
    sieve_results.push(("Seg-Odds", elapsed, count));

    // ── 5. 210-Wheel Production Sieve ───────────────────────────────────────
    print!("  Running 210-Wheel Sieve..........");
    let start = Instant::now();
    let count = sieve_210_wheel(sieve_limit);
    let elapsed = start.elapsed().as_secs_f64();
    println!(" {:>8.4} s  ({} primes)", elapsed, format_number(count as u64));
    sieve_results.push(("210-Wheel", elapsed, count));

    // ── Verify all sieve counts match ───────────────────────────────────────
    println!();
    if sieve_results.len() >= 2 {
        let reference_count = sieve_results.last().unwrap().2;
        let all_match = sieve_results.iter().all(|(_, _, c)| *c == reference_count);
        if all_match {
            println!("  ✓ All sieve implementations agree: {} primes", format_number(reference_count as u64));
        } else {
            println!("  ✗ WARNING: Sieve implementations disagree!");
            for (name, _, count) in &sieve_results {
                println!("    {}: {} primes", name, format_number(*count as u64));
            }
        }
    }

    // ── Speedup table ───────────────────────────────────────────────────────
    println!();
    println!("  Sieve Speedup Table (relative to slowest):");
    if let Some(&(_, slowest, _)) = sieve_results.first() {
        let slowest = slowest.max(1e-12);
        for (name, elapsed, _) in &sieve_results {
            let speedup = slowest / elapsed.max(1e-12);
            println!("    {:<25} {:>8.4} s  ({:>6.1}x)", name, elapsed, speedup);
        }
    }

    println!();

    // ══════════════════════════════════════════════════════════════════════════
    //  PRNG BENCHMARKS
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══ PRNG BENCHMARKS ════════════════════════════════════════════════");
    println!("  Generating {} u64 values each", format_number(prng_count));
    println!();

    let mut prng_results: Vec<(&str, f64)> = Vec::new();

    // ── 1. XorShift64 ───────────────────────────────────────────────────────
    print!("  Running XorShift64...............");
    let mut rng = XorShift64::new(42);
    let start = Instant::now();
    let mut sum = 0u64;
    for _ in 0..prng_count {
        sum = sum.wrapping_add(rng.next_u64());
    }
    let elapsed = start.elapsed().as_secs_f64();
    black_box(sum);
    println!(" {:>8.4} s  ({:.1} M/s)", elapsed, prng_count as f64 / elapsed / 1e6);
    prng_results.push(("XorShift64", elapsed));

    // ── 2. PCG32 ────────────────────────────────────────────────────────────
    print!("  Running PCG32....................");
    let mut rng = Pcg32Bench::new(42, 54);
    let start = Instant::now();
    let mut sum = 0u64;
    for _ in 0..prng_count {
        sum = sum.wrapping_add(rng.next_u64());
    }
    let elapsed = start.elapsed().as_secs_f64();
    black_box(sum);
    println!(" {:>8.4} s  ({:.1} M/s)", elapsed, prng_count as f64 / elapsed / 1e6);
    prng_results.push(("PCG32", elapsed));

    // ── 3. Mersenne Twister MT19937-64 ──────────────────────────────────────
    print!("  Running Mersenne Twister 64......");
    let mut rng = MersenneTwister64::new(42);
    let start = Instant::now();
    let mut sum = 0u64;
    for _ in 0..prng_count {
        sum = sum.wrapping_add(rng.next_u64());
    }
    let elapsed = start.elapsed().as_secs_f64();
    black_box(sum);
    println!(" {:>8.4} s  ({:.1} M/s)", elapsed, prng_count as f64 / elapsed / 1e6);
    prng_results.push(("MT19937-64", elapsed));

    // ── 4. Squares Counter-Based ────────────────────────────────────────────
    print!("  Running Squares Counter-Based....");
    let mut rng = SquaresRng::new(42);
    let start = Instant::now();
    let mut sum = 0u64;
    for _ in 0..prng_count {
        sum = sum.wrapping_add(rng.next_u64());
    }
    let elapsed = start.elapsed().as_secs_f64();
    black_box(sum);
    println!(" {:>8.4} s  ({:.1} M/s)", elapsed, prng_count as f64 / elapsed / 1e6);
    prng_results.push(("Squares", elapsed));

    // ── 5. Shishiua Counter-Based ───────────────────────────────────────────
    print!("  Running Shishiua Counter-Based...");
    let mut rng = ShishiuaRng::new(42);
    let start = Instant::now();
    let mut sum = 0u64;
    for _ in 0..prng_count {
        sum = sum.wrapping_add(rng.next_u64());
    }
    let elapsed = start.elapsed().as_secs_f64();
    black_box(sum);
    println!(" {:>8.4} s  ({:.1} M/s)", elapsed, prng_count as f64 / elapsed / 1e6);
    prng_results.push(("Shishiua", elapsed));

    // ── 6. SimdPrng4 (4-lane batched) ──────────────────────────────────────
    print!("  Running SimdPrng4 (4-lane).......");
    let mut rng = SimdPrng4::new(42);
    let start = Instant::now();
    let mut sum = 0u64;
    for _ in 0..prng_count {
        sum = sum.wrapping_add(rng.next_u64());
    }
    let elapsed = start.elapsed().as_secs_f64();
    black_box(sum);
    println!(" {:>8.4} s  ({:.1} M/s)", elapsed, prng_count as f64 / elapsed / 1e6);
    prng_results.push(("SimdPrng4", elapsed));

    // ── 7. SimdPrng8 (8-lane batched) ──────────────────────────────────────
    print!("  Running SimdPrng8 (8-lane).......");
    let mut rng = SimdPrng8::new(42);
    let start = Instant::now();
    let mut sum = 0u64;
    for _ in 0..prng_count {
        sum = sum.wrapping_add(rng.next_u64());
    }
    let elapsed = start.elapsed().as_secs_f64();
    black_box(sum);
    println!(" {:>8.4} s  ({:.1} M/s)", elapsed, prng_count as f64 / elapsed / 1e6);
    prng_results.push(("SimdPrng8", elapsed));

    // ── 8. SimdPrng8 Batch (8 at a time) ───────────────────────────────────
    print!("  Running SimdPrng8 Batched........");
    let mut rng = SimdPrng8::new(42);
    let start = Instant::now();
    let mut sum = [0u64; 8];
    let batches = prng_count / 8;
    for _ in 0..batches {
        let vals = rng.next_8x_u64();
        for i in 0..8 {
            sum[i] = sum[i].wrapping_add(vals[i]);
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    black_box(sum);
    println!(" {:>8.4} s  ({:.1} M/s)", elapsed, prng_count as f64 / elapsed / 1e6);
    prng_results.push(("SimdPrng8-Batch", elapsed));

    // ── 9. Bulk fill benchmark ─────────────────────────────────────────────
    print!("  Running SimdPrng8 Fill...........");
    let mut rng = SimdPrng8::new(42);
    let mut buf = vec![0u64; prng_count as usize];
    let start = Instant::now();
    rng.fill_u64(&mut buf);
    let elapsed = start.elapsed().as_secs_f64();
    black_box(&buf);
    let throughput_gib = (prng_count as f64 * 8.0) / elapsed / (1024.0 * 1024.0 * 1024.0);
    println!(" {:>8.4} s  ({:.1} GiB/s)", elapsed, throughput_gib);
    prng_results.push(("SimdPrng8-Fill", elapsed));

    // ── 10. Counter-based jump benchmark ────────────────────────────────────
    print!("  Running Counter Jump (Squares)...");
    let rng = SquaresRng::new(42);
    let start = Instant::now();
    let mut sum = 0u64;
    // Jump to 1000 random positions (simulating parallel access)
    for i in 0..1000 {
        let idx = i * 1_000_000_000_000; // trillion-scale jumps
        sum = sum.wrapping_add(rng.at_index(idx));
    }
    let elapsed = start.elapsed().as_secs_f64();
    black_box(sum);
    println!(" {:>8.4} s  (1000 jumps to trillion-scale indices)", elapsed);

    // ── PRNG Speedup table ──────────────────────────────────────────────────
    println!();
    println!("  PRNG Speedup Table (relative to slowest):");
    if let Some(&(_, slowest)) = prng_results.first() {
        let slowest = slowest.max(1e-12);
        for (name, elapsed) in &prng_results {
            let speedup = slowest / elapsed.max(1e-12);
            println!("    {:<25} {:>8.4} s  ({:>6.1}x)", name, elapsed, speedup);
        }
    }

    println!();

    // ══════════════════════════════════════════════════════════════════════════
    //  THROUGHPUT SUMMARY
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══ THROUGHPUT SUMMARY ═════════════════════════════════════════════");

    if let Some(&(_, wheel_time, _)) = sieve_results.iter().find(|(n, _, _)| *n == "210-Wheel") {
        let sieve_tp = sieve_limit as f64 / wheel_time.max(1e-12);
        println!("  210-Wheel Sieve:  {:.0} numbers/s  ({:.1}M/s)", sieve_tp, sieve_tp / 1e6);
    }

    if let Some(&(_, fill_time)) = prng_results.iter().find(|(n, _)| *n == "SimdPrng8-Fill") {
        let prng_tp = prng_count as f64 * 8.0 / fill_time.max(1e-12);
        println!("  SimdPrng8 Fill:   {:.0} bytes/s    ({:.1} GiB/s)", prng_tp, prng_tp / (1024.0 * 1024.0 * 1024.0));
    }

    println!();

    // ══════════════════════════════════════════════════════════════════════════
    //  LARGE-SCALE SIEVE BENCHMARK (optional)
    // ══════════════════════════════════════════════════════════════════════════
    if sieve_limit >= 100_000_000 {
        println!("═══ LARGE-SCALE SIEVE (1 Billion) ════════════════════════════════");
        let big_limit = 1_000_000_000u64;

        print!("  Running Segmented Odds-Only (1B)...");
        let start = Instant::now();
        let count = segmented_odds_sieve(big_limit);
        let elapsed = start.elapsed().as_secs_f64();
        println!(" {:>8.4} s  ({} primes)", elapsed, format_number(count as u64));

        print!("  Running 210-Wheel (1B).............");
        let start = Instant::now();
        let count = sieve_210_wheel(big_limit);
        let elapsed = start.elapsed().as_secs_f64();
        println!(" {:>8.4} s  ({} primes)", elapsed, format_number(count as u64));

        println!();
    }

    println!("Benchmark complete.");
}

fn format_number(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}
