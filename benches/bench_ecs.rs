// Lightweight ECS microbenchmark harness.
// Run with: `cargo run --release --bin bench-ecs -- <entities> <steps> <dt> <mode>`

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::hint::black_box;
use std::time::{Duration, Instant};

use jules::interp::{EcsWorld, Value};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let dt: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.016);
    let mode = args.get(4).map(String::as_str).unwrap_or("both");

    println!(
        "bench-ecs: entities={}, steps={}, dt={}, mode={}",
        n, steps, dt, mode
    );

    let mut world = EcsWorld::default();

    for _ in 0..n {
        let id = world.spawn();
        world.insert_component(id, "pos", Value::Vec3([0.0, 0.0, 0.0]));
        world.insert_component(id, "vel", Value::Vec3([1.0, 0.0, 0.0]));
        world.insert_component(id, "health", Value::F32(100.0));
        world.insert_component(id, "damage", Value::F32(0.25));
    }

    // Warmup
    for _ in 0..3 {
        run_step(&mut world, dt);
    }

    if mode == "baseline" || mode == "both" || mode == "rust-compare" {
        let t0 = Instant::now();
        for _ in 0..steps {
            run_step(&mut world, dt);
        }
        let elapsed = t0.elapsed();
        let sps = steps as f64 / elapsed.as_secs_f64().max(1e-12);
        println!(
            "baseline elapsed: {:.3}s ({:.1} steps/s)",
            elapsed.as_secs_f64(),
            sps
        );
    }

    if mode == "soa-linear" || mode == "both" || mode == "rust-compare" {
        let mut world_soa = EcsWorld::default();
        for _ in 0..n {
            let id = world_soa.spawn();
            world_soa.insert_component(id, "pos", Value::Vec3([0.0, 0.0, 0.0]));
            world_soa.insert_component(id, "vel", Value::Vec3([1.0, 0.0, 0.0]));
            world_soa.insert_component(id, "health", Value::F32(100.0));
            world_soa.insert_component(id, "damage", Value::F32(0.25));
        }
        let ts = Instant::now();
        for _ in 0..steps {
            let _ = world_soa.integrate_vec3_linear("pos", "vel", dt);
        }
        let elapsed = ts.elapsed().as_secs_f64();
        let sps = steps as f64 / elapsed.max(1e-12);
        println!("soa-linear elapsed: {:.3}s ({:.1} steps/s)", elapsed, sps);
    }

    let mut fused_elapsed_s = None;
    if mode == "fused-linear" || mode == "both" || mode == "rust-compare" {
        let mut world_fused = EcsWorld::default();
        for _ in 0..n {
            let id = world_fused.spawn();
            world_fused.insert_component(id, "pos", Value::Vec3([0.0, 0.0, 0.0]));
            world_fused.insert_component(id, "vel", Value::Vec3([1.0, 0.0, 0.0]));
            world_fused.insert_component(id, "health", Value::F32(100.0));
            world_fused.insert_component(id, "damage", Value::F32(0.25));
        }
        let tf = Instant::now();
        for _ in 0..steps {
            let _ = world_fused.integrate_vec3_linear_fused("pos", "vel", dt);
        }
        let elapsed = tf.elapsed().as_secs_f64();
        fused_elapsed_s = Some(elapsed);
        let sps = steps as f64 / elapsed.max(1e-12);
        println!("fused-linear elapsed: {:.3}s ({:.1} steps/s)", elapsed, sps);
    }

    let mut chunked_elapsed_s = None;
    if mode == "chunked" || mode == "both" || mode == "rust-compare" {
        let mut world_chunked = EcsWorld::default();
        for _ in 0..n {
            let id = world_chunked.spawn();
            world_chunked.insert_component(id, "pos", Value::Vec3([0.0, 0.0, 0.0]));
            world_chunked.insert_component(id, "vel", Value::Vec3([1.0, 0.0, 0.0]));
            world_chunked.insert_component(id, "health", Value::F32(100.0));
            world_chunked.insert_component(id, "damage", Value::F32(0.25));
        }
        let tc = Instant::now();
        for _ in 0..steps {
            let _ = world_chunked.integrate_vec3_chunked_precomputed("pos", "vel", dt, 64);
            let _ = world_chunked
                .integrate_vec3_and_health_chunked("pos", "vel", "health", "damage", dt, 64);
        }
        let elapsed = tc.elapsed().as_secs_f64();
        chunked_elapsed_s = Some(elapsed);
        let sps = steps as f64 / elapsed.max(1e-12);
        println!(
            "chunked-fused elapsed: {:.3}s ({:.1} steps/s)",
            elapsed, sps
        );
    }

    let mut superopt_elapsed_s = None;
    if mode == "superoptimizer" || mode == "both" || mode == "rust-compare" {
        let mut world_superopt = EcsWorld::default();
        for _ in 0..n {
            let id = world_superopt.spawn();
            world_superopt.insert_component(id, "pos", Value::Vec3([0.0, 0.0, 0.0]));
            world_superopt.insert_component(id, "vel", Value::Vec3([1.0, 0.0, 0.0]));
            world_superopt.insert_component(id, "health", Value::F32(100.0));
            world_superopt.insert_component(id, "damage", Value::F32(0.25));
        }
        let ts = Instant::now();
        for _ in 0..steps {
            let _ = world_superopt.integrate_vec3_superoptimizer("pos", "vel", dt, 64);
        }
        let elapsed = ts.elapsed().as_secs_f64();
        superopt_elapsed_s = Some(elapsed);
        let sps = steps as f64 / elapsed.max(1e-12);
        println!(
            "superoptimizer elapsed: {:.3}s ({:.1} steps/s)",
            elapsed, sps
        );
    }

    let mut aot_elapsed_s = None;
    if mode == "aot-hash" || mode == "both" || mode == "rust-compare" {
        let mut world_aot = EcsWorld::default();
        for _ in 0..n {
            let id = world_aot.spawn();
            world_aot.insert_component(id, "pos", Value::Vec3([0.0, 0.0, 0.0]));
            world_aot.insert_component(id, "vel", Value::Vec3([1.0, 0.0, 0.0]));
            world_aot.insert_component(id, "health", Value::F32(100.0));
            world_aot.insert_component(id, "damage", Value::F32(0.25));
        }
        let cache = AotStepCache::build(&world_aot);
        let mut hotspot = StepHotspot::default();
        let t1 = Instant::now();
        for _ in 0..steps {
            run_step_aot_hash(&mut world_aot, dt, &cache, &mut hotspot);
        }
        let elapsed = t1.elapsed();
        aot_elapsed_s = Some(elapsed.as_secs_f64());
        let sps = steps as f64 / elapsed.as_secs_f64().max(1e-12);
        println!(
            "aot-hash elapsed: {:.3}s ({:.1} steps/s) cache_hash={:016x}",
            elapsed.as_secs_f64(),
            sps,
            cache.layout_hash
        );
        println!(
            "hotspot weights: query={:.1}% fetch={:.1}% math={:.1}% write={:.1}%",
            hotspot.weight_query() * 100.0,
            hotspot.weight_fetch() * 100.0,
            hotspot.weight_math() * 100.0,
            hotspot.weight_write() * 100.0,
        );
    }

    if mode == "rust-compare" || mode == "both" {
        let mut rw = RustWorld::new(n);
        for _ in 0..3 {
            run_step_rust(&mut rw, dt);
        }
        let min_measure_window = Duration::from_millis(200);
        let mut measured_steps = 0usize;
        let tr = Instant::now();
        while measured_steps < steps || tr.elapsed() < min_measure_window {
            for _ in 0..steps {
                run_step_rust(&mut rw, dt);
            }
            measured_steps += steps;
        }
        // Keep final state observable so the optimizer cannot delete updates.
        let checksum = rust_checksum(&rw);
        black_box(checksum);
        black_box(&rw);
        let elapsed = tr.elapsed().as_secs_f64();
        let sps = measured_steps as f64 / elapsed.max(1e-12);
        println!(
            "rust elapsed: {:.3}s ({:.1} steps/s, measured_steps={})",
            elapsed, sps, measured_steps
        );
        let rust_sec_per_step = elapsed / (measured_steps as f64).max(1.0);
        if let Some(aot_s) = aot_elapsed_s {
            let aot_sec_per_step = aot_s / (steps as f64).max(1.0);
            println!(
                "aot-vs-rust ratio (sec/step): {:.2}x",
                aot_sec_per_step / rust_sec_per_step.max(1e-12)
            );
        }
        if let Some(fused_s) = fused_elapsed_s {
            let fused_sec_per_step = fused_s / (steps as f64).max(1.0);
            println!(
                "fused-vs-rust ratio (sec/step): {:.2}x",
                fused_sec_per_step / rust_sec_per_step.max(1e-12)
            );
        }
        if let Some(chunked_s) = chunked_elapsed_s {
            let chunked_sec_per_step = chunked_s / (steps as f64).max(1.0);
            println!(
                "chunked-vs-rust ratio (sec/step): {:.2}x",
                chunked_sec_per_step / rust_sec_per_step.max(1e-12)
            );
        }
        if let Some(superopt_s) = superopt_elapsed_s {
            let superopt_sec_per_step = superopt_s / (steps as f64).max(1.0);
            println!(
                "superoptimizer-vs-rust ratio (sec/step): {:.2}x",
                superopt_sec_per_step / rust_sec_per_step.max(1e-12)
            );
        }
    }

    // If SIMD feature enabled, run the gather/process/scatter path too
    #[cfg(feature = "phase6-simd")]
    {
        println!("running SIMD-path comparison (feature=phase6-simd)");
        let mut world2 = EcsWorld::default();
        for _ in 0..n {
            let id = world2.spawn();
            world2.insert_component(id, "pos", Value::Vec3([0.0, 0.0, 0.0]));
            world2.insert_component(id, "vel", Value::Vec3([1.0, 0.0, 0.0]));
        }
        let ids = world2.query(&vec!["pos".to_string(), "vel".to_string()], &vec![]);
        let mut positions = Vec::with_capacity(ids.len());
        let mut velocities = Vec::with_capacity(ids.len());
        let t1 = Instant::now();
        for _ in 0..steps {
            run_step_simd_precomputed(&mut world2, dt, &ids, &mut positions, &mut velocities);
        }
        let elapsed_simd = t1.elapsed();
        println!("simd elapsed: {:.3}s", elapsed_simd.as_secs_f64());
    }
}

#[derive(Debug, Clone)]
struct AotStepCache {
    ids: Vec<u64>,
    layout_hash: u64,
}

struct RustWorld {
    pos: Vec<[f32; 3]>,
    vel: Vec<[f32; 3]>,
    health: Vec<f32>,
    damage: Vec<f32>,
}

impl RustWorld {
    fn new(n: usize) -> Self {
        Self {
            pos: vec![[0.0, 0.0, 0.0]; n],
            vel: vec![[1.0, 0.0, 0.0]; n],
            health: vec![100.0; n],
            damage: vec![0.25; n],
        }
    }
}

impl AotStepCache {
    fn build(world: &EcsWorld) -> Self {
        let ids = world.query2("pos", "vel");
        let mut hasher = DefaultHasher::new();
        ids.hash(&mut hasher);
        let layout_hash = hasher.finish();
        Self { ids, layout_hash }
    }
}

#[derive(Default)]
struct StepHotspot {
    query_ns: u128,
    fetch_ns: u128,
    math_ns: u128,
    write_ns: u128,
}

impl StepHotspot {
    fn total_ns(&self) -> u128 {
        self.query_ns + self.fetch_ns + self.math_ns + self.write_ns
    }
    fn weight_query(&self) -> f64 {
        self.query_ns as f64 / self.total_ns().max(1) as f64
    }
    fn weight_fetch(&self) -> f64 {
        self.fetch_ns as f64 / self.total_ns().max(1) as f64
    }
    fn weight_math(&self) -> f64 {
        self.math_ns as f64 / self.total_ns().max(1) as f64
    }
    fn weight_write(&self) -> f64 {
        self.write_ns as f64 / self.total_ns().max(1) as f64
    }
}

fn run_step_aot_hash(
    world: &mut EcsWorld,
    dt: f32,
    cache: &AotStepCache,
    hotspot: &mut StepHotspot,
) {
    let t_query = Instant::now();
    let mut hasher = DefaultHasher::new();
    cache.ids.hash(&mut hasher);
    let live_hash = hasher.finish();
    hotspot.query_ns += t_query.elapsed().as_nanos();
    if live_hash != cache.layout_hash {
        run_step(world, dt);
        return;
    }

    for id in &cache.ids {
        let t_fetch = Instant::now();
        let vel = match world.get_component(*id, "vel") {
            Some(Value::Vec3(v)) => *v,
            _ => continue,
        };
        hotspot.fetch_ns += t_fetch.elapsed().as_nanos();

        let t_math = Instant::now();
        let dv = [vel[0] * dt, vel[1] * dt, vel[2] * dt];
        hotspot.math_ns += t_math.elapsed().as_nanos();

        let t_write = Instant::now();
        if let Some(Value::Vec3(p)) = world.get_component_mut(*id, "pos") {
            p[0] += dv[0];
            p[1] += dv[1];
            p[2] += dv[2];
        }
        hotspot.write_ns += t_write.elapsed().as_nanos();
    }
}

fn run_step_rust(world: &mut RustWorld, dt: f32) {
    for ((p, v), (h, d)) in world
        .pos
        .iter_mut()
        .zip(&world.vel)
        .zip(world.health.iter_mut().zip(&world.damage))
    {
        p[0] += v[0] * dt;
        p[1] += v[1] * dt;
        p[2] += v[2] * dt;
        *h -= *d * dt;
    }
}

fn rust_checksum(world: &RustWorld) -> f64 {
    let pos_sum: f64 = world
        .pos
        .iter()
        .map(|p| (p[0] + p[1] + p[2]) as f64)
        .sum();
    let health_sum: f64 = world.health.iter().map(|h| *h as f64).sum();
    pos_sum + health_sum
}

fn run_step(world: &mut EcsWorld, dt: f32) {
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

#[cfg(feature = "phase6-simd")]
fn run_step_simd_precomputed(
    world: &mut EcsWorld,
    dt: f32,
    ids: &[u64],
    positions: &mut Vec<[f32; 3]>,
    velocities: &mut Vec<[f32; 3]>,
) {
    // Keep buffers hot and fixed-size across iterations so the benchmark
    // measures SIMD math + ECS gather/scatter, not allocator churn.
    if positions.len() != ids.len() {
        positions.resize(ids.len(), [0.0, 0.0, 0.0]);
    }
    if velocities.len() != ids.len() {
        velocities.resize(ids.len(), [0.0, 0.0, 0.0]);
    }

    // gather
    for (i, id) in ids.iter().enumerate() {
        let p = world.get_component(*id, "pos").cloned();
        let v = world.get_component(*id, "vel").cloned();
        match (p, v) {
            (Some(Value::Vec3(pv)), Some(Value::Vec3(vv))) => {
                positions[i] = pv;
                velocities[i] = vv;
            }
            _ => {
                positions[i] = [0.0, 0.0, 0.0];
                velocities[i] = [0.0, 0.0, 0.0];
            }
        }
    }

    // call centralized SIMD helper
    jules::phase6_simd::update_positions(positions, velocities, dt);

    // scatter
    for (i, id) in ids.iter().enumerate() {
        world.insert_component(*id, "pos", Value::Vec3(positions[i]));
    }
}
