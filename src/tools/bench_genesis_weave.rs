// =============================================================================
// bench-genesis-weave — Comprehensive Benchmark for Genesis Weave Stack
//
// Benchmarks all components of the Jules-spec world generation system:
//   1. 210-Wheel Sieve (prime counting at scale)
//   2. SIMD PRNG (throughput in GiB/s)
//   3. Morton Encoding (2D/3D encode/decode throughput)
//   4. Genesis Weave (terrain, biome, entity, exclusion queries/s)
//   5. Collision System (probes/s, resolution throughput)
// =============================================================================

use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     JULES GENESIS WEAVE — Comprehensive Benchmark Suite        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Verification Phase ──────────────────────────────────────────────
    println!("▶ Phase 0: Verification");
    let sieve_ok = jules::jules_std::sieve_210::verify_sieve_implementations();
    let prng_ok = jules::jules_std::prng_simd::test_prng_quality()
        && jules::jules_std::prng_simd::test_counter_based()
        && jules::jules_std::prng_simd::test_simd8_consistency();
    let morton_ok = jules::jules_std::morton::verify_morton();
    let genesis_ok = jules::jules_std::genesis_weave::verify_genesis();
    let collision_ok = jules::jules_std::collision::verify_collision();

    println!("  Sieve 210:      {}", if sieve_ok { "✓ PASS" } else { "✗ FAIL" });
    println!("  PRNG SIMD:      {}", if prng_ok { "✓ PASS" } else { "✗ FAIL" });
    println!("  Morton:         {}", if morton_ok { "✓ PASS" } else { "✗ FAIL" });
    println!("  Genesis Weave:  {}", if genesis_ok { "✓ PASS" } else { "✗ FAIL" });
    println!("  Collision:      {}", if collision_ok { "✓ PASS" } else { "✗ FAIL" });
    println!();

    if !(sieve_ok && prng_ok && morton_ok && genesis_ok && collision_ok) {
        eprintln!("⚠ Some verification tests failed — benchmarks may be unreliable.");
        println!();
    }

    // ── Benchmark 1: 210-Wheel Sieve ───────────────────────────────────
    println!("▶ Benchmark 1: 210-Wheel Factorization Sieve");
    {
        let limits = [1_000_000u64, 10_000_000, 100_000_000];

        for &limit in &limits {
            let t = Instant::now();
            let count = jules::jules_std::sieve_210::sieve_210_wheel(limit);
            let elapsed = t.elapsed();
            let rate = (limit as f64) / elapsed.as_secs_f64();
            println!("  sieve_210_wheel({:>12}) = {:>8} primes  [{:>8.2}M nums/s  {:.2?}]",
                limit, count, rate / 1_000_000.0, elapsed);
        }

        // Compare against naive
        let t = Instant::now();
        let count = jules::jules_std::sieve_210::naive_sieve(10_000_000);
        let elapsed = t.elapsed();
        println!("  naive_sieve(10M)             = {:>8} primes  [{:.2?}]", count, elapsed);

        // Compare against odds-only
        let t = Instant::now();
        let count = jules::jules_std::sieve_210::odds_only_sieve(10_000_000);
        let elapsed = t.elapsed();
        println!("  odds_only_sieve(10M)         = {:>8} primes  [{:.2?}]", count, elapsed);

        // Compare against segmented odds
        let t = Instant::now();
        let count = jules::jules_std::sieve_210::segmented_odds_sieve(10_000_000);
        let elapsed = t.elapsed();
        println!("  segmented_odds_sieve(10M)    = {:>8} primes  [{:.2?}]", count, elapsed);
    }
    println!();

    // ── Benchmark 2: SIMD PRNG ─────────────────────────────────────────
    println!("▶ Benchmark 2: Counter-Based PRNG (SIMD8)");
    {
        let n = 10_000_000u64;
        let mut buf = vec![0u64; n as usize];

        // SimdPrng8 fill
        let mut rng = jules::jules_std::prng_simd::SimdPrng8::new(42);
        let t = Instant::now();
        rng.fill_u64(&mut buf);
        let elapsed = t.elapsed();
        let bytes = (n * 8) as f64;
        let rate_gib = bytes / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
        println!("  SimdPrng8 Fill:  {:>12} u64s in {:.2?}  ({:.2} GiB/s)", n, elapsed, rate_gib);

        // Squares scalar
        let mut rng = jules::jules_std::prng_simd::SquaresRng::new(42);
        let t = Instant::now();
        for item in buf.iter_mut().take(1_000_000) {
            *item = rng.next_u64();
        }
        let elapsed = t.elapsed();
        let bytes = (1_000_000 * 8) as f64;
        let rate_gib = bytes / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
        println!("  Squares Scalar:  {:>12} u64s in {:.2?}  ({:.2} GiB/s)", 1_000_000, elapsed, rate_gib);

        // Shishiua scalar
        let mut rng = jules::jules_std::prng_simd::ShishiuaRng::new(42);
        let t = Instant::now();
        for item in buf.iter_mut().take(1_000_000) {
            *item = rng.next_u64();
        }
        let elapsed = t.elapsed();
        let bytes = (1_000_000 * 8) as f64;
        let rate_gib = bytes / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
        println!("  Shishiua Scalar: {:>12} u64s in {:.2?}  ({:.2} GiB/s)", 1_000_000, elapsed, rate_gib);

        // Mersenne Twister baseline
        let mut rng = jules::jules_std::prng_simd::MersenneTwister64::new(42);
        let t = Instant::now();
        for item in buf.iter_mut().take(1_000_000) {
            *item = rng.next_u64();
        }
        let elapsed = t.elapsed();
        let bytes = (1_000_000 * 8) as f64;
        let rate_gib = bytes / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
        println!("  MT19937-64:      {:>12} u64s in {:.2?}  ({:.2} GiB/s)", 1_000_000, elapsed, rate_gib);

        // XorShift64 baseline
        let mut rng = jules::jules_std::prng_simd::XorShift64::new(42);
        let t = Instant::now();
        for item in buf.iter_mut().take(1_000_000) {
            *item = rng.next_u64();
        }
        let elapsed = t.elapsed();
        let bytes = (1_000_000 * 8) as f64;
        let rate_gib = bytes / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
        println!("  XorShift64:      {:>12} u64s in {:.2?}  ({:.2} GiB/s)", 1_000_000, elapsed, rate_gib);

        // Counter-based jump-to-N (unique to Jules)
        let rng = jules::jules_std::prng_simd::SquaresRng::new(42);
        let t = Instant::now();
        for item in buf.iter_mut().take(1_000_000) {
            *item = rng.at_index(*item);
        }
        let elapsed = t.elapsed();
        let bytes = (1_000_000 * 8) as f64;
        let rate_gib = bytes / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
        println!("  Squares at_index:{:>12} u64s in {:.2?}  ({:.2} GiB/s) [JUMP-TO-N]", 1_000_000, elapsed, rate_gib);
    }
    println!();

    // ── Benchmark 3: Morton Encoding ───────────────────────────────────
    println!("▶ Benchmark 3: Morton Z-Order Encoding");
    {
        let n = 10_000_000u32;

        // 2D encode
        let t = Instant::now();
        let mut sum = 0u64;
        for i in 0..n {
            sum = sum.wrapping_add(jules::jules_std::morton::encode_2d(i, i));
        }
        let elapsed = t.elapsed();
        let rate = (n as f64) / elapsed.as_secs_f64();
        println!("  encode_2d:   {:>12} ops in {:.2?}  ({:.0}M ops/s)  [checksum: {}]",
            n, elapsed, rate / 1_000_000.0, sum);

        // 2D decode
        let t = Instant::now();
        let mut sum_x = 0u32;
        for i in 0..n {
            let (x, _) = jules::jules_std::morton::decode_2d(i as u64);
            sum_x = sum_x.wrapping_add(x);
        }
        let elapsed = t.elapsed();
        let rate = (n as f64) / elapsed.as_secs_f64();
        println!("  decode_2d:   {:>12} ops in {:.2?}  ({:.0}M ops/s)  [checksum: {}]",
            n, elapsed, rate / 1_000_000.0, sum_x);

        // 3D encode
        let t = Instant::now();
        let mut sum = 0u64;
        for i in 0..n {
            sum = sum.wrapping_add(jules::jules_std::morton::encode_3d(i, i, i));
        }
        let elapsed = t.elapsed();
        let rate = (n as f64) / elapsed.as_secs_f64();
        println!("  encode_3d:   {:>12} ops in {:.2?}  ({:.0}M ops/s)  [checksum: {}]",
            n, elapsed, rate / 1_000_000.0, sum);

        // BitPlane set/get
        let mut bp = jules::jules_std::morton::BitPlane::new(256, 256, 4);
        let t = Instant::now();
        for y in 0..256u32 {
            for x in 0..256u32 {
                bp.set(x, y, ((x + y) % 16) as u64);
            }
        }
        let elapsed = t.elapsed();
        let ops = 256 * 256;
        let rate = (ops as f64) / elapsed.as_secs_f64();
        println!("  BitPlane set: {:>12} ops in {:.2?}  ({:.0}M ops/s)", ops, elapsed, rate / 1_000_000.0);

        let t = Instant::now();
        let mut sum = 0u64;
        for y in 0..256u32 {
            for x in 0..256u32 {
                sum += bp.get(x, y);
            }
        }
        let elapsed = t.elapsed();
        let rate = (ops as f64) / elapsed.as_secs_f64();
        println!("  BitPlane get: {:>12} ops in {:.2?}  ({:.0}M ops/s)  [checksum: {}]", ops, elapsed, rate / 1_000_000.0, sum);
    }
    println!();

    // ── Benchmark 4: Genesis Weave ─────────────────────────────────────
    println!("▶ Benchmark 4: Genesis Weave (Stateless World Generation)");
    {
        let world = jules::jules_std::genesis_weave::GenesisWeave::new(42);
        let n = 1_000_000u64;

        // Terrain height
        let t = Instant::now();
        let mut sum = 0.0f64;
        for i in 0..n {
            let x = (i as f64) * 0.01;
            sum += world.height(x, x * 0.7, 0.0);
        }
        let elapsed = t.elapsed();
        let rate = (n as f64) / elapsed.as_secs_f64();
        println!("  terrain_height:   {:>12} queries in {:.2?}  ({:.0}K queries/s)  [avg: {:.4}]",
            n, elapsed, rate / 1_000.0, sum / n as f64);

        // Biome query
        let t = Instant::now();
        let mut biome_counts = [0usize; 16];
        for i in 0..n {
            let x = (i as f64) * 0.01;
            let biome = world.biome(x, x * 0.7);
            biome_counts[biome as usize] += 1;
        }
        let elapsed = t.elapsed();
        let rate = (n as f64) / elapsed.as_secs_f64();
        println!("  biome_at:         {:>12} queries in {:.2?}  ({:.0}K queries/s)", n, elapsed, rate / 1_000.0);

        // Entity oracle
        let n_entity = 100_000u64;
        let t = Instant::now();
        let mut entity_count = 0usize;
        for i in 0..n_entity {
            let x = (i as i64) * 3;
            let y = (i as i64) * 7;
            let entity = world.entity(x, y);
            if entity as u8 != 0 { entity_count += 1; }
        }
        let elapsed = t.elapsed();
        let rate = (n_entity as f64) / elapsed.as_secs_f64();
        println!("  entity_at:        {:>12} queries in {:.2?}  ({:.0}K queries/s)  [entities: {}]",
            n_entity, elapsed, rate / 1_000.0, entity_count);

        // Exclusion sieve
        let n_excl = 100_000u64;
        let t = Instant::now();
        let mut excluded = 0usize;
        for i in 0..n_excl {
            if world.is_excluded((i * 3) as i64, (i * 7) as i64) {
                excluded += 1;
            }
        }
        let elapsed = t.elapsed();
        let rate = (n_excl as f64) / elapsed.as_secs_f64();
        println!("  check_exclusion:  {:>12} queries in {:.2?}  ({:.0}K queries/s)  [excluded: {}]",
            n_excl, elapsed, rate / 1_000.0, excluded);

        // Jittered grid
        let n_jitter = 100_000u64;
        let t = Instant::now();
        for i in 0..n_jitter {
            let _ = world.jittered(i as i64, i as i64);
        }
        let elapsed = t.elapsed();
        let rate = (n_jitter as f64) / elapsed.as_secs_f64();
        println!("  jittered_position:{:>12} queries in {:.2?}  ({:.0}K queries/s)",
            n_jitter, elapsed, rate / 1_000.0);

        // Structure generation
        let n_struct = 10_000u64;
        let t = Instant::now();
        let mut total_segments = 0usize;
        for i in 0..n_struct {
            let segments = world.structure(i as i64, i as i64);
            total_segments += segments.len() / 6; // 6 values per segment
        }
        let elapsed = t.elapsed();
        let rate = (n_struct as f64) / elapsed.as_secs_f64();
        println!("  generate_struct:  {:>12} structs in {:.2?}  ({:.0} structs/s)  [avg segments: {:.1}]",
            n_struct, elapsed, rate, total_segments as f64 / n_struct as f64);

        // Biome distribution
        println!("  Biome distribution:");
        let biome_names = ["Ocean", "Beach", "Plains", "Forest", "DenseForest",
            "Hills", "Mountains", "SnowCaps", "Desert", "Savanna", "Jungle",
            "Tundra", "Swamp", "Volcanic", "Mushroom", "Crystal"];
        for (i, &count) in biome_counts.iter().enumerate() {
            if count > 0 {
                let pct = count as f64 / n as f64 * 100.0;
                println!("    {:>14}: {:>6} ({:.1}%)", biome_names[i], count, pct);
            }
        }
    }
    println!();

    // ── Benchmark 5: Collision System ──────────────────────────────────
    println!("▶ Benchmark 5: Stateless Collision System");
    {
        let seed = 42u64;
        let n = 100_000u64;

        // Single probe
        let t = Instant::now();
        let mut hit_count = 0usize;
        for i in 0..n {
            let x = (i as f64) * 1.5;
            let y = (i as f64) * 2.3;
            if jules::jules_std::collision::probe_collision(seed, x, y, 1.0) {
                hit_count += 1;
            }
        }
        let elapsed = t.elapsed();
        let rate = (n as f64) / elapsed.as_secs_f64();
        println!("  probe_collision:  {:>12} probes in {:.2?}  ({:.0}K probes/s)  [hits: {}]",
            n, elapsed, rate / 1_000.0, hit_count);

        // 8-probe SIMD
        let n8 = 50_000u64;
        let t = Instant::now();
        for i in 0..n8 {
            let x = (i as f64) * 3.0;
            let y = (i as f64) * 5.0;
            let _ = jules::jules_std::collision::probe_collision_8(seed, x, y, 1.0);
        }
        let elapsed = t.elapsed();
        let total_probes = n8 * 8;
        let rate = (total_probes as f64) / elapsed.as_secs_f64();
        println!("  probe_8:          {:>12} probes in {:.2?}  ({:.0}K probes/s)  [8x SIMD]",
            total_probes, elapsed, rate / 1_000.0);

        // 3D probe
        let n3d = 50_000u64;
        let t = Instant::now();
        for i in 0..n3d {
            let x = (i as f64) * 1.5;
            let y = (i as f64) * 2.3;
            let z = (i as f64) * 0.7;
            let _ = jules::jules_std::collision::probe_collision_3d(seed, x, y, z, 1.0);
        }
        let elapsed = t.elapsed();
        let rate = (n3d as f64) / elapsed.as_secs_f64();
        println!("  probe_3d:         {:>12} probes in {:.2?}  ({:.0}K probes/s)",
            n3d, elapsed, rate / 1_000.0);

        // Collision resolution
        let n_resolve = 10_000u64;
        let t = Instant::now();
        for i in 0..n_resolve {
            let x = (i as f64) * 5.0;
            let y = (i as f64) * 7.0;
            let _ = jules::jules_std::collision::resolve_collision(seed, x, y, 1.0, 8);
        }
        let elapsed = t.elapsed();
        let rate = (n_resolve as f64) / elapsed.as_secs_f64();
        println!("  resolve_collision:{:>12} resolves in {:.2?}  ({:.0}K resolves/s)",
            n_resolve, elapsed, rate / 1_000.0);

        // Morton-ordered batch probe
        let n_morton = 10_000u64;
        let t = Instant::now();
        for i in 0..n_morton {
            let _ = jules::jules_std::collision::morton_batch_probe(seed, i as u32, i as u32, 3);
        }
        let elapsed = t.elapsed();
        let rate = (n_morton as f64) / elapsed.as_secs_f64();
        println!("  morton_batch:     {:>12} batches in {:.2?}  ({:.0}K batches/s)",
            n_morton, elapsed, rate / 1_000.0);

        // Batch 8-entity collision
        let positions = [
            (100.0, 100.0), (200.0, 200.0), (300.0, 300.0), (400.0, 400.0),
            (500.0, 500.0), (600.0, 600.0), (700.0, 700.0), (800.0, 800.0),
        ];
        let n_batch = 10_000u64;
        let t = Instant::now();
        for _ in 0..n_batch {
            let _ = jules::jules_std::collision::batch_collision_8(seed, &positions, 1.0);
        }
        let elapsed = t.elapsed();
        let total = n_batch * 8;
        let rate = (total as f64) / elapsed.as_secs_f64();
        println!("  batch_8:          {:>12} entities in {:.2?}  ({:.0}K entities/s)  [8x batch]",
            total, elapsed, rate / 1_000.0);
    }
    println!();

    // ── Summary ────────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK COMPLETE                           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  All Genesis Weave algorithms are now part of the built-in library.");
    println!("  Modules: sieve_210, prng_simd, morton, genesis_weave, collision");
    println!("  Each module is available via stdlib dispatch (e.g., genesis::terrain_height)");
    println!();
    println!("  Run with 'cargo run --bin bench-genesis-weave --release' for");
    println!("  release-mode numbers (expected 2-4x speedup over debug).");
}
