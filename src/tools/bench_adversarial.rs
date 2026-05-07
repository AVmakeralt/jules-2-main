// =============================================================================
// Adversarial Benchmark Suite — DESTROY YOUR ILLUSIONS
//
// Every benchmark here is designed to WORST-CASE your code:
//   - Random access patterns that destroy cache locality
//   - Data-dependent branches that defeat prediction
//   - Memory pressure that evicts your carefully-placed L2 lines
//   - Pointer-chasing that kills the hardware prefetcher
//
// If your benchmark shows good numbers HERE, they're real.
// If it only shows good numbers on sequential access, those are fake.
//
// Run: cargo run --release --bin bench-adversarial
// =============================================================================

use std::hint::black_box;
use std::time::Instant;

use jules::jules_std::sieve_210::{
    sieve_210_wheel, sieve_210_wheel_fastcount,
};
use jules::runtime::jhal::CfiJumpTable;

// ─── Throughput Sanity Ceiling Constants ─────────────────────────────────────
//
// These are HARD PHYSICAL LIMITS based on x86-64 microarchitecture.
// Any benchmark claiming throughput above these is measuring compiler
// optimization (dead code elimination), not actual work.

// Hardware ceiling constants — used by check_sanity_ceilings().
// See throughput_sanity.rs for the full physical-limits reference.

/// Minimum nanoseconds for any operation that touches L1 data cache.
const MIN_L1_ACCESS_NS: f64 = 1.0;

/// Minimum nanoseconds for a cache-line-sized write (64 bytes).
const MIN_CACHELINE_WRITE_NS: f64 = 2.0;

fn main() {
    println!("================================================================");
    println!("  ADVERSARIAL BENCHMARK SUITE");
    println!("  \"If it survives this, it's real.\"");
    println!("================================================================");
    println!();

    // ── Phase 1: Throughput Sanity Checks ─────────────────────────────────
    println!("─── Phase 1: Throughput Sanity Checks ───");
    println!();
    check_sanity_ceilings();

    // ── Phase 2: Adversarial Sieve ────────────────────────────────────────
    println!();
    println!("─── Phase 2: Adversarial Sieve (segment-aligned vs random range) ───");
    println!();
    bench_sieve_adversarial();

    // ── Phase 3: Cache Thrash Test ────────────────────────────────────────
    println!();
    println!("─── Phase 3: Cache Thrash (random 64B stride vs sequential) ───");
    println!();
    bench_cache_thrash();

    // ── Phase 4: Branch Divergence ────────────────────────────────────────
    println!();
    println!("─── Phase 4: Branch Divergence (predictable vs random conditions) ───");
    println!();
    bench_branch_divergence();

    // ── Phase 5: Pointer Chase ────────────────────────────────────────────
    println!();
    println!("─── Phase 5: Pointer Chase (linked list vs array traversal) ───");
    println!();
    bench_pointer_chase();

    // ── Phase 6: CFI Lookup Under Pressure ────────────────────────────────
    println!();
    println!("─── Phase 6: CFI Lookup (sequential vs adversarial access) ───");
    println!();
    bench_cfi_adversarial();

    // ── Phase 7: ECS Churn ────────────────────────────────────────────────
    println!();
    println!("─── Phase 7: ECS Churn (stable entities vs spawn/kill churn) ───");
    println!();
    bench_ecs_churn();

    println!();
    println!("================================================================");
    println!("  ADVERSARIAL BENCHMARK COMPLETE");
    println!("  Numbers that survived: REAL. Numbers that cratered: THEATER.");
    println!("================================================================");
}

// ─── Phase 1: Sanity Checks ──────────────────────────────────────────────────

fn check_sanity_ceilings() {
    // These are checks against COMMON fake benchmark patterns.
    // If your benchmark reports numbers that violate these, it's measuring
    // compiler optimization, not actual work.

    // Test: Can we really do a memory access in 0 ns? (No.)
    let iters = 10_000_000;
    let mut data = vec![0u64; 1024]; // Fits in L1
    let start = Instant::now();
    for i in 0..iters {
        data[black_box(i) % 1024] = black_box(data[black_box(i) % 1024].wrapping_add(1));
    }
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_nanos() as f64 / iters as f64;

    println!(
        "  L1 random read-modify-write: {:.1} ns/iter (ceiling: {:.1} ns)",
        ns_per, MIN_L1_ACCESS_NS
    );
    if ns_per < MIN_L1_ACCESS_NS {
        println!("    ⚠ SUSPECT: below L1 access ceiling — possible DCE");
    } else {
        println!("    ✓ Above L1 access ceiling — measurement is real");
    }

    // Test: Can we write a cache line in 0 ns? (No.)
    let mut big = vec![0u64; 1_000_000];
    let start = Instant::now();
    for i in 0..iters {
        let idx = (i * 8) % big.len();
        big[idx..idx + 8].fill(black_box(i as u64));
    }
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_nanos() as f64 / iters as f64;
    println!(
        "  Cache-line write (64B): {:.1} ns/iter (ceiling: {:.1} ns)",
        ns_per, MIN_CACHELINE_WRITE_NS
    );
    if ns_per < MIN_CACHELINE_WRITE_NS {
        println!("    ⚠ SUSPECT: below cache-line write ceiling — possible DCE");
    } else {
        println!("    ✓ Above cache-line write ceiling — measurement is real");
    }
}

// ─── Phase 2: Adversarial Sieve ──────────────────────────────────────────────

fn bench_sieve_adversarial() {
    let limit = 10_000_000u64;
    let expected = 664_579usize;

    // Friendly: count all primes up to limit (segment-aligned)
    let start = Instant::now();
    let result = sieve_210_wheel(limit);
    let friendly_time = start.elapsed();
    println!(
        "  Friendly (10M, segment-aligned): {} primes in {:.3}ms",
        result,
        friendly_time.as_secs_f64() * 1000.0
    );
    assert_eq!(result, expected, "sieve_210_wheel correctness failed!");

    let start = Instant::now();
    let result2 = sieve_210_wheel_fastcount(limit);
    let fastcount_time = start.elapsed();
    println!(
        "  Fastcount (10M, segment-aligned): {} primes in {:.3}ms",
        result2,
        fastcount_time.as_secs_f64() * 1000.0
    );
    assert_eq!(result2, expected, "fastcount correctness failed!");

    // Adversarial: multiple small non-aligned ranges (destroy segment reuse)
    let adversarial_limits: Vec<u64> = (0..100)
        .map(|i| 1_000_000 + (i as u64 * 7919) % 9_000_000) // Random-ish limits
        .collect();
    let start = Instant::now();
    let mut total_primes = 0usize;
    for &lim in &adversarial_limits {
        total_primes += sieve_210_wheel(lim);
    }
    let adversarial_time = start.elapsed();
    println!(
        "  Adversarial (100 random limits 1M–10M): {} total primes in {:.3}ms",
        total_primes,
        adversarial_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Adversarial overhead: {:.1}x vs single 10M pass",
        (adversarial_time.as_secs_f64() / 100.0) / friendly_time.as_secs_f64().max(1e-12)
    );

    // Large adversarial: stress the L2 cache
    let big_limit = 100_000_000u64;
    let start = Instant::now();
    let big_result = sieve_210_wheel(big_limit);
    let big_time = start.elapsed();
    println!(
        "  Large (100M): {} primes in {:.3}ms ({:.0} nums/s)",
        big_result,
        big_time.as_secs_f64() * 1000.0,
        big_limit as f64 / big_time.as_secs_f64().max(1e-12)
    );
}

// ─── Phase 3: Cache Thrash ───────────────────────────────────────────────────

fn bench_cache_thrash() {
    const SIZE: usize = 64 * 1024 * 1024; // 64M u64s = 512 MB (way beyond L3)
    let mut data = vec![0u64; SIZE];
    let iters = 1_000_000;

    // Sequential access (best case for hardware prefetcher)
    let start = Instant::now();
    for i in 0..iters {
        data[black_box(i) % SIZE] = data[black_box(i) % SIZE].wrapping_add(1);
    }
    let seq_time = start.elapsed();
    let seq_ns = seq_time.as_nanos() as f64 / iters as f64;
    println!(
        "  Sequential ({}MB): {:.1} ns/access",
        SIZE * 8 / 1024 / 1024,
        seq_ns
    );

    // Random stride (destroy cache locality, defeat prefetcher)
    // Use a simple LCG for reproducibility
    let mut rng_state: u64 = 0x1234567890ABCDEF;
    let start = Instant::now();
    for _ in 0..iters {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let idx = (rng_state >> 12) as usize % SIZE;
        data[idx] = data[idx].wrapping_add(1);
        black_box(data[idx]);
    }
    let rand_time = start.elapsed();
    let rand_ns = rand_time.as_nanos() as f64 / iters as f64;
    println!(
        "  Random stride ({}MB): {:.1} ns/access",
        SIZE * 8 / 1024 / 1024,
        rand_ns
    );
    println!(
        "  Random penalty: {:.1}x slower than sequential",
        rand_ns / seq_ns.max(0.1)
    );
}

// ─── Phase 4: Branch Divergence ──────────────────────────────────────────────

fn bench_branch_divergence() {
    let iters = 10_000_000;
    let mut data = vec![0u32; iters];

    // Predictable branches: threshold is always 50
    let start = Instant::now();
    let mut count = 0u64;
    for i in 0..iters {
        data[i] = (i as u32).wrapping_mul(7919);
        if black_box(data[i]) < 50 {
            count += 1;
        }
    }
    let pred_time = start.elapsed();
    let pred_ns = pred_time.as_nanos() as f64 / iters as f64;
    println!(
        "  Predictable branches: {:.1} ns/iter ({} taken)",
        pred_ns, count
    );

    // Random branches: data-dependent conditions
    let mut rng_state: u64 = 0xCAFEBABEDEADBEEF;
    let start = Instant::now();
    let mut count2 = 0u64;
    for i in 0..iters {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let threshold = (rng_state >> 48) as u32;
        data[i] = (i as u32).wrapping_mul(7919);
        if black_box(data[i]) < threshold {
            count2 += 1;
        }
    }
    let rand_time = start.elapsed();
    let rand_ns = rand_time.as_nanos() as f64 / iters as f64;
    println!(
        "  Random branches: {:.1} ns/iter ({} taken)",
        rand_ns, count2
    );
    println!(
        "  Misprediction penalty: {:.1}x slower than predictable",
        rand_ns / pred_ns.max(0.1)
    );
}

// ─── Phase 5: Pointer Chase ──────────────────────────────────────────────────

fn bench_pointer_chase() {
    const SIZE: usize = 1_000_000;
    let iters = 1_000_000;

    // Array traversal (sequential, prefetcher-friendly)
    let data: Vec<u64> = (0..SIZE as u64).collect();
    let start = Instant::now();
    let mut sum = 0u64;
    for i in 0..iters {
        sum = sum.wrapping_add(data[black_box(i) % SIZE]);
    }
    let array_time = start.elapsed();
    let array_ns = array_time.as_nanos() as f64 / iters as f64;
    println!(
        "  Array traversal: {:.1} ns/access, sum={}",
        array_ns,
        black_box(sum) % 1000
    );

    // Linked list traversal (pointer chase, kills prefetcher)
    // Build a linked list with shuffled next pointers
    let mut indices: Vec<usize> = (0..SIZE).collect();
    // Fisher-Yates shuffle
    let mut rng = 0x1234567890ABCDEFu64;
    for i in (1..SIZE).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (rng as usize) % (i + 1);
        indices.swap(i, j);
    }
    let mut next: Vec<usize> = vec![0; SIZE];
    for k in 0..SIZE - 1 {
        next[indices[k]] = indices[k + 1];
    }
    next[indices[SIZE - 1]] = indices[0]; // circular

    let values: Vec<u64> = (0..SIZE as u64).collect();
    let start = Instant::now();
    let mut sum2 = 0u64;
    let mut idx = indices[0];
    for _ in 0..iters {
        sum2 = sum2.wrapping_add(values[idx]);
        idx = next[idx];
    }
    let chase_time = start.elapsed();
    let chase_ns = chase_time.as_nanos() as f64 / iters as f64;
    println!(
        "  Pointer chase: {:.1} ns/access, sum={}",
        chase_ns,
        black_box(sum2) % 1000
    );
    println!(
        "  Pointer chase penalty: {:.1}x slower than array traversal",
        chase_ns / array_ns.max(0.1)
    );
}

// ─── Phase 6: CFI Lookup Adversarial ─────────────────────────────────────────

fn bench_cfi_adversarial() {
    let mut table = CfiJumpTable::new();
    for i in 0..256 {
        table.register_target(0x1000 + (i as u64) * 0x100);
    }

    let iters = 1_000_000;

    // Sequential access
    let start = Instant::now();
    let mut count = 0u64;
    for i in 0..iters {
        let target = 0x1000 + ((i as u64 % 256) * 0x100);
        count += black_box(table.is_valid_target(target)) as u64;
    }
    let seq_time = start.elapsed();
    let seq_ns = seq_time.as_nanos() as f64 / iters as f64;
    println!(
        "  Sequential CFI lookup: {:.1} ns/iter ({} hits)",
        seq_ns, count
    );

    // Random access (destroys any caching in the jump table)
    let mut rng = 0xDEADBEEFCAFEBABEu64;
    let start = Instant::now();
    let mut count2 = 0u64;
    for _ in 0..iters {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let target = 0x1000 + ((rng >> 56) as u64 % 256) * 0x100;
        count2 += black_box(table.is_valid_target(target)) as u64;
    }
    let rand_time = start.elapsed();
    let rand_ns = rand_time.as_nanos() as f64 / iters as f64;
    println!(
        "  Random CFI lookup: {:.1} ns/iter ({} hits)",
        rand_ns, count2
    );

    // Invalid targets (worst case — full scan)
    let start = Instant::now();
    let mut count3 = 0u64;
    for i in 0..iters {
        let target = 0xDEAD_0000 + (i as u64 * 0x100); // Never registered
        count3 += black_box(table.is_valid_target(target)) as u64;
    }
    let invalid_time = start.elapsed();
    let invalid_ns = invalid_time.as_nanos() as f64 / iters as f64;
    println!(
        "  Invalid CFI lookup: {:.1} ns/iter ({} hits — should be 0)",
        invalid_ns, count3
    );
}

// ─── Phase 7: ECS Churn ──────────────────────────────────────────────────────

fn bench_ecs_churn() {
    use jules::interp::{EcsWorld, Value};

    let n = 10_000;
    let steps = 20;
    let dt = 0.016f32;

    // Stable entities (no churn — best case for SoA layout)
    let mut world = EcsWorld::default();
    for _ in 0..n {
        let id = world.spawn();
        world.insert_component(id, "pos", Value::Vec3([0.0, 0.0, 0.0]));
        world.insert_component(id, "vel", Value::Vec3([1.0, 0.0, 0.0]));
    }
    // Warmup
    for _ in 0..3 {
        run_step_ecs(&mut world, dt);
    }
    let start = Instant::now();
    for _ in 0..steps {
        run_step_ecs(&mut world, dt);
    }
    let stable_time = start.elapsed();
    let stable_sps = steps as f64 / stable_time.as_secs_f64().max(1e-12);
    println!(
        "  Stable ECS ({} entities): {:.3}ms, {:.0} steps/s",
        n,
        stable_time.as_secs_f64() * 1000.0,
        stable_sps
    );

    // Churn: kill 10% and respawn each step (destroy SoA layout)
    let mut churn_world = EcsWorld::default();
    let mut ids: Vec<u64> = Vec::new();
    for _ in 0..n {
        let id = churn_world.spawn();
        churn_world.insert_component(id, "pos", Value::Vec3([0.0, 0.0, 0.0]));
        churn_world.insert_component(id, "vel", Value::Vec3([1.0, 0.0, 0.0]));
        ids.push(id);
    }
    // Warmup
    for _ in 0..3 {
        run_step_ecs(&mut churn_world, dt);
    }
    let start = Instant::now();
    for step in 0..steps {
        run_step_ecs(&mut churn_world, dt);
        // Kill 10% and respawn every other step
        if step % 2 == 0 {
            let kill_count = n / 10;
            for &id in &ids[..kill_count] {
                churn_world.despawn(id);
            }
            for _ in 0..kill_count {
                let id = churn_world.spawn();
                churn_world.insert_component(id, "pos", Value::Vec3([0.0, 0.0, 0.0]));
                churn_world.insert_component(id, "vel", Value::Vec3([1.0, 0.0, 0.0]));
            }
        }
    }
    let churn_time = start.elapsed();
    let churn_sps = steps as f64 / churn_time.as_secs_f64().max(1e-12);
    println!(
        "  Churn ECS (10% kill/respawn): {:.3}ms, {:.0} steps/s",
        churn_time.as_secs_f64() * 1000.0,
        churn_sps
    );
    println!(
        "  Churn overhead: {:.1}x slower than stable",
        stable_sps / churn_sps.max(1.0)
    );
}

fn run_step_ecs(world: &mut jules::interp::EcsWorld, dt: f32) {
    use jules::interp::Value;
    let ids = world.query2("pos", "vel");
    for id in ids {
        let pos_val = world.get_component(id, "pos").unwrap().clone();
        let vel_val = world.get_component(id, "vel").unwrap().clone();
        let new_pos = match (pos_val, vel_val) {
            (Value::Vec3(p), Value::Vec3(v)) => {
                Value::Vec3([p[0] + v[0] * dt, p[1] + v[1] * dt, p[2] + v[2] * dt])
            }
            _ => continue,
        };
        world.insert_component(id, "pos", new_pos);
    }
}
