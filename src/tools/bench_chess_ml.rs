// Chess learning-environment benchmark.
// Run with: cargo run --release --bin bench-chess-ml -- <episodes> <max_steps> <batch> <envs> <device>

use jules::chess_ml::{
    eval_policy_vs_random, train_chess_policy_batched, train_chess_policy_batched_from,
    train_chess_policy_gpu, train_chess_policy_soa,
};
use std::hint::black_box;
use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;
use std::time::Instant;

const BASELINE_PARAM_COUNT: usize = 8;
const F32_BYTES: u64 = std::mem::size_of::<f32>() as u64;

#[derive(Debug, Clone, Copy)]
struct ResourceSnapshot {
    ram_available_bytes: Option<u64>,
    gpu_available_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct TrainEstimate {
    estimated_duration: Duration,
    estimated_steps_per_sec: f64,
    total_steps: u64,
    memory_required_bytes: u64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let episodes: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let max_steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let batch_raw: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let envs_raw: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let batch = batch_raw.max(1).min(4096);
    let envs = envs_raw.max(1).min(4096);
    let device = args.get(5).map(String::as_str).unwrap_or("cpu");
    let mut params = BASELINE_PARAM_COUNT;
    let mut estimate_only = false;
    let mut skip_strength_eval = false;
    let mut i = 6usize;
    while i < args.len() {
        match args[i].as_str() {
            "--params" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse::<usize>().ok()) {
                    params = v.max(1);
                    i += 2;
                } else {
                    println!("⚠️ --params requires a positive integer; using default {params}");
                    i += 1;
                }
            }
            "--estimate-only" => {
                estimate_only = true;
                i += 1;
            }
            "--skip-strength-eval" => {
                skip_strength_eval = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    if batch != batch_raw || envs != envs_raw {
        println!("⚠️ safety clamp applied: batch {batch_raw}->{batch}, envs {envs_raw}->{envs}");
    }

    println!(
        "bench-chess-ml episodes={episodes} max_steps={max_steps} batch={batch} envs={envs} device={device}"
    );
    println!("estimator: params={params}, estimate_only={estimate_only}");

    let resources = detect_resources(device);
    print_resource_report(resources, device);
    if let Some(estimate) = estimate_training_time(episodes, max_steps, batch, envs, device, params)
    {
        print_estimate(estimate);
        maybe_print_speed_tips(estimate, episodes, max_steps, batch, envs, params, device);
    } else {
        println!("⚠️ estimator unavailable: failed to run calibration pass");
    }
    if estimate_only {
        return;
    }

    let jules = if device == "gpu" {
        train_chess_policy_gpu(episodes, envs, max_steps, batch, 0xC0FFEE).unwrap_or_else(|e| {
            println!("⚠️ GPU path unavailable ({e}), falling back to CPU SoA path.");
            train_chess_policy_soa(episodes, envs, max_steps, batch, 0xC0FFEE)
        })
    } else if envs > 1 {
        train_chess_policy_soa(episodes, envs, max_steps, batch, 0xC0FFEE)
    } else {
        train_chess_policy_batched(episodes, max_steps, batch, 0xC0FFEE)
    };
    println!(
        "Jules engine: {:.3}s, steps={}, {:.0} steps/s, win_rate={:.2}%",
        jules.elapsed.as_secs_f64(),
        jules.total_steps,
        jules.steps_per_sec,
        jules.win_rate * 100.0
    );

    match python_baseline(episodes, max_steps) {
        Ok(py) => {
            println!("Python baseline: {:.3}s, {:.0} steps/s", py.0, py.1);
            let speedup = jules.steps_per_sec / py.1.max(1e-9);
            println!("Speedup (Jules/Python): {:.2}x", speedup);
            if speedup >= 1.0 {
                println!("✅ Jules is faster than Python on this benchmark.");
            } else {
                println!("⚠️ Jules is slower than Python on this machine/run.");
            }
        }
        Err(e) => {
            println!("⚠️ Python baseline skipped: {e}");
        }
    }

    match jax_baseline(episodes, max_steps, envs.max(1)) {
        Ok((elapsed, steps_per_sec, backend)) => {
            println!(
                "JAX baseline ({backend}): {:.3}s, {:.0} steps/s",
                elapsed, steps_per_sec
            );
            let speedup = jules.steps_per_sec / steps_per_sec.max(1e-9);
            println!("Speedup (Jules/JAX): {:.2}x", speedup);
            if speedup >= 1.0 {
                println!("✅ Jules is faster than JAX on this benchmark.");
            } else {
                println!("⚠️ Jules is slower than JAX on this machine/run.");
            }
        }
        Err(e) => {
            println!("⚠️ JAX baseline skipped: {e}");
        }
    }
    let rust = rust_baseline(episodes, max_steps, envs.max(1));
    println!(
        "Rust baseline: {:.3}s, {:.0} steps/s, checksum={:.4}",
        rust.0, rust.1, rust.2
    );
    println!(
        "Speedup (Jules/Rust): {:.2}x",
        jules.steps_per_sec / rust.1.max(1e-9)
    );

    if skip_strength_eval {
        println!("Skipping long strength evaluation loop (--skip-strength-eval).");
        return;
    }

    let target_strength = 0.70f32;
    let mut train_seed = 0xC0FFEEu64;
    let mut total_train_episodes = 0usize;
    let mut weights = [0.03f32, 0.0, 0.0, 0.2, 0.01, -0.01, 0.0, 0.0];
    let eval_games = 1_500usize;
    let chunk = 100_000usize;
    println!(
        "Strength training: target win-rate vs random = {:.0}% (eval games={eval_games})",
        target_strength * 100.0
    );
    loop {
        let (chunk_result, new_weights) = train_chess_policy_batched_from(
            chunk,
            max_steps.max(32),
            batch.max(64),
            train_seed,
            weights,
        );
        weights = new_weights;
        total_train_episodes += chunk;
        train_seed = train_seed.wrapping_add(1);
        let eval_wr =
            eval_policy_vs_random(weights, eval_games, max_steps.max(32), train_seed ^ 0xA11CE);
        println!(
            "  episodes={} chunk_steps/s={:.0} eval_win_rate={:.2}%",
            total_train_episodes,
            chunk_result.steps_per_sec,
            eval_wr * 100.0
        );
        if eval_wr >= target_strength || total_train_episodes >= 2_000_000 {
            println!(
                "Final strength: eval_win_rate={:.2}% after {} episodes",
                eval_wr * 100.0,
                total_train_episodes
            );
            break;
        }
    }
}

fn detect_resources(device: &str) -> ResourceSnapshot {
    ResourceSnapshot {
        ram_available_bytes: read_available_ram_bytes(),
        gpu_available_bytes: if device == "gpu" {
            read_available_gpu_bytes()
        } else {
            None
        },
    }
}

fn print_resource_report(resources: ResourceSnapshot, device: &str) {
    match resources.ram_available_bytes {
        Some(v) => println!("resource: RAM available ≈ {:.2} GiB", bytes_to_gib(v)),
        None => println!("resource: RAM available unknown"),
    }
    if device == "gpu" {
        match resources.gpu_available_bytes {
            Some(v) => println!(
                "resource: GPU memory available ≈ {:.2} GiB",
                bytes_to_gib(v)
            ),
            None => println!("resource: GPU memory available unknown"),
        }
    }
}

fn estimate_training_time(
    episodes: usize,
    max_steps: usize,
    batch: usize,
    envs: usize,
    device: &str,
    params: usize,
) -> Option<TrainEstimate> {
    let calibration_episodes = episodes.clamp(32, 2_048);
    let calibration = if device == "gpu" {
        train_chess_policy_gpu(calibration_episodes, envs, max_steps, batch, 0xE57E_0001).ok()
    } else if envs > 1 {
        Some(train_chess_policy_soa(
            calibration_episodes,
            envs,
            max_steps,
            batch,
            0xE57E_0001,
        ))
    } else {
        Some(train_chess_policy_batched(
            calibration_episodes,
            max_steps,
            batch,
            0xE57E_0001,
        ))
    }?;

    let calibration_sps = calibration.steps_per_sec.max(1.0);
    // Compute cost scales near-linearly with parameter count for dense layers.
    let param_scale = params as f64 / BASELINE_PARAM_COUNT as f64;
    let estimated_steps_per_sec = (calibration_sps / param_scale.max(1e-9)).max(1.0);
    let total_steps = episodes as u64 * max_steps as u64 * envs.max(1) as u64;
    let secs = total_steps as f64 / estimated_steps_per_sec;
    let estimated_duration = Duration::from_secs_f64(secs.max(0.0));
    let memory_required_bytes = estimate_memory_bytes(params, batch, envs);
    Some(TrainEstimate {
        estimated_duration,
        estimated_steps_per_sec,
        total_steps,
        memory_required_bytes,
    })
}

fn print_estimate(est: TrainEstimate) {
    println!(
        "estimate: total_steps={} est_steps/s={:.0} est_time={} est_memory={:.2} GiB",
        est.total_steps,
        est.estimated_steps_per_sec,
        format_duration(est.estimated_duration),
        bytes_to_gib(est.memory_required_bytes)
    );
}

fn maybe_print_speed_tips(
    est: TrainEstimate,
    episodes: usize,
    max_steps: usize,
    batch: usize,
    envs: usize,
    params: usize,
    device: &str,
) {
    if est.estimated_duration < Duration::from_secs(30 * 60) {
        return;
    }
    println!("⚠️ estimated training time exceeds 30 minutes. Speed-up suggestions:");
    println!(
        "  • Reduce episodes: {episodes} -> {}",
        ((episodes as f64) * 0.7) as usize
    );
    println!(
        "  • Reduce max_steps: {max_steps} -> {}",
        (max_steps as f64 * 0.8).max(1.0) as usize
    );
    println!(
        "  • Reduce params: {params} -> {}",
        (params as f64 * 0.75).max(1.0) as usize
    );
    if envs == 1 {
        println!("  • Increase envs (parallel sims): 1 -> 8 (CPU) or 32+ (GPU)");
    } else {
        println!("  • Tune envs to saturate hardware: current envs={envs}");
    }
    if device != "gpu" {
        println!("  • Use `gpu` device mode if available for larger batches.");
    }
    println!("  • Batch tuning: current batch={batch}; test 64, 128, 256 and keep best steps/s.");
}

fn estimate_memory_bytes(params: usize, batch: usize, envs: usize) -> u64 {
    let params = params as u64;
    let batch = batch as u64;
    let envs = envs.max(1) as u64;
    // Weight + gradient + optimizer state (Adam-like) ~= 4 tensors.
    let model_bytes = params * F32_BYTES * 4;
    // Activation/workspace estimate: scales with params and active batch/env lanes.
    let activation_bytes = params
        .saturating_mul((batch.max(1) + envs).max(1))
        .saturating_mul(F32_BYTES / 2 + 1);
    // Safety factor for allocator fragmentation and temporary buffers.
    (model_bytes + activation_bytes).saturating_mul(13) / 10
}

fn read_available_ram_bytes() -> Option<u64> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

fn read_available_gpu_bytes() -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut total_mib = 0u64;
    for line in text.lines() {
        if let Ok(v) = line.trim().parse::<u64>() {
            total_mib = total_mib.saturating_add(v);
        }
    }
    if total_mib == 0 {
        None
    } else {
        Some(total_mib.saturating_mul(1024 * 1024))
    }
}

fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn format_duration(d: Duration) -> String {
    let total = d.as_secs();
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}h {m}m {s}s")
    } else {
        format!("{m}m {s}s")
    }
}

fn python_baseline(episodes: usize, max_steps: usize) -> Result<(f64, f64), String> {
    let script = r#"
import time, random, sys
E = int(sys.argv[1]); M = int(sys.argv[2])
w = [0.03,0.0,0.0,0.2,0.01,-0.01,0.0,0.0]
steps = 0
t0 = time.perf_counter()
for _ in range(E):
    wp, bp = 8, 8
    wa, ba = 0.0, 0.0
    white = True
    for _ in range(M):
        f = [1.0, wp, bp, wp-bp, wa, ba, 0.0, 0.0]
        score = sum(fi*wi for fi,wi in zip(f,w)) + random.random()*0.01
        reward = 0.001 if score > 0 else -0.001
        if white:
            wa += 1.0
            if random.random() < 0.03 and bp > 0:
                bp -= 1; reward += 0.1
        else:
            ba += 1.0
            if random.random() < 0.03 and wp > 0:
                wp -= 1; reward -= 0.1
        for i in range(8):
            x = f[i]
            if x > 32: x = 32
            if x < -32: x = -32
            w[i] += 0.002 * reward * x
        white = not white
        steps += 1
elapsed = time.perf_counter() - t0
print(f"{elapsed} {steps/elapsed if elapsed>0 else 0}")
"#;

    let mut child = Command::new("python3")
        .arg("-")
        .arg(episodes.to_string())
        .arg(max_steps.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("python3 not available: {e}"))?;

    {
        let stdin = child.stdin.as_mut().ok_or("failed to open stdin")?;
        stdin
            .write_all(script.as_bytes())
            .map_err(|e| format!("failed to write python script: {e}"))?;
    }

    let out = child
        .wait_with_output()
        .map_err(|e| format!("python failed: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).to_string());
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.split_whitespace();
    let elapsed: f64 = it
        .next()
        .ok_or("missing elapsed")?
        .parse()
        .map_err(|_| "bad elapsed")?;
    let steps_per_sec: f64 = it
        .next()
        .ok_or("missing steps/s")?
        .parse()
        .map_err(|_| "bad steps/s")?;
    Ok((elapsed, steps_per_sec))
}

fn jax_baseline(
    episodes: usize,
    max_steps: usize,
    envs: usize,
) -> Result<(f64, f64, String), String> {
    let script = r#"
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys, time

try:
    import jax
    import jax.numpy as jnp
except Exception as e:
    print(f"ERR:jax_missing:{e}")
    raise SystemExit(7)

E = int(sys.argv[1]); M = int(sys.argv[2]); N = max(1, int(sys.argv[3]))

weights = jnp.array([0.03,0.0,0.0,0.2,0.01,-0.01,0.0,0.0], dtype=jnp.float32)

@jax.jit
def run_training(weights, key):
    def episode_fn(_, state):
        w, key = state
        wp = jnp.full((N,), 8.0, dtype=jnp.float32)
        bp = jnp.full((N,), 8.0, dtype=jnp.float32)
        wa = jnp.zeros((N,), dtype=jnp.float32)
        ba = jnp.zeros((N,), dtype=jnp.float32)

        def step_fn(_, carry):
            wp, bp, wa, ba, w, key = carry
            key, k1, k2 = jax.random.split(key, 3)
            f = jnp.stack([jnp.ones_like(wp), wp, bp, wp - bp, wa, ba, wa * 0.1, ba * 0.1], axis=1)
            score = jnp.sum(f * w[None, :], axis=1) + (jax.random.uniform(k1, (N,), dtype=jnp.float32) - 0.5) * 0.01
            reward = jnp.where(score > 0, 0.001, -0.001)

            hit_prob = jax.random.uniform(k2, (N,), dtype=jnp.float32)
            white_hits = (hit_prob < 0.03) & (bp > 0)
            bp = bp - white_hits.astype(jnp.float32)
            reward = reward + white_hits.astype(jnp.float32) * 0.1
            wa = wa + 1.0

            grad = jnp.mean(f * reward[:, None], axis=0)
            w = w + 0.002 * grad
            return (wp, bp, wa, ba, w, key)

        wp, bp, wa, ba, w, key = jax.lax.fori_loop(0, M, step_fn, (wp, bp, wa, ba, w, key))
        return (w, key)

    return jax.lax.fori_loop(0, E, episode_fn, (weights, key))

key = jax.random.PRNGKey(0xC0FFEE)
run_training(weights, key)[0].block_until_ready()  # warmup compile

t0 = time.perf_counter()
weights, key = run_training(weights, key)
weights.block_until_ready()
elapsed = time.perf_counter() - t0
steps = E * M * N
backend = jax.default_backend()
print(f"{elapsed} {steps/elapsed if elapsed>0 else 0} {backend}")
"#;

    let mut child = Command::new("python3")
        .arg("-")
        .arg(episodes.to_string())
        .arg(max_steps.to_string())
        .arg(envs.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("python3 not available: {e}"))?;

    child
        .stdin
        .as_mut()
        .ok_or("failed to open stdin")?
        .write_all(script.as_bytes())
        .map_err(|e| format!("failed to write JAX script: {e}"))?;

    let out = child
        .wait_with_output()
        .map_err(|e| format!("JAX run failed: {e}"))?;

    if !out.status.success() {
        let msg = format!(
            "{} {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        if msg.contains("ERR:jax_missing:") {
            return Err("JAX not installed in python3 environment".to_string());
        }
        return Err(msg);
    }

    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.split_whitespace();
    let elapsed: f64 = it
        .next()
        .ok_or("missing elapsed")?
        .parse()
        .map_err(|_| "bad elapsed")?;
    let steps_per_sec: f64 = it
        .next()
        .ok_or("missing steps/s")?
        .parse()
        .map_err(|_| "bad steps/s")?;
    let backend = it.next().unwrap_or("unknown").to_string();

    Ok((elapsed, steps_per_sec, backend))
}

fn rust_baseline(episodes: usize, max_steps: usize, envs: usize) -> (f64, f64, f32) {
    let e = episodes;
    let m = max_steps;
    let n = envs.max(1);
    let mut rng = 0xC0FFEEu64;
    let mut w = [0.03f32, 0.0, 0.0, 0.2, 0.01, -0.01, 0.0, 0.0];
    let t0 = Instant::now();
    let mut checksum = 0.0f32;

    for _ in 0..e {
        let wp = vec![8.0f32; n];
        let mut bp = vec![8.0f32; n];
        let mut wa = vec![0.0f32; n];
        let ba = vec![0.0f32; n];
        for _ in 0..m {
            let mut grad = [0.0f32; 8];
            for i in 0..n {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let r0 = (((rng >> 32) as u32) as f32 / u32::MAX as f32) - 0.5;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let hit = ((rng >> 32) as u32) as f32 / u32::MAX as f32;
                let f = [1.0f32, wp[i], bp[i], wp[i] - bp[i], wa[i], ba[i], wa[i] * 0.1, ba[i] * 0.1];
                let score = f
                    .iter()
                    .zip(w.iter())
                    .map(|(fi, wi)| fi * wi)
                    .sum::<f32>()
                    + r0 * 0.01;
                let mut reward = if score > 0.0 { 0.001 } else { -0.001 };
                if hit < 0.03 && bp[i] > 0.0 {
                    bp[i] -= 1.0;
                    reward += 0.1;
                }
                wa[i] += 1.0;
                for k in 0..8 {
                    grad[k] += f[k] * reward;
                }
                checksum += score * 1e-6 + reward;
            }
            let inv_n = 1.0 / n as f32;
            for k in 0..8 {
                w[k] += 0.002 * grad[k] * inv_n;
            }
        }
    }
    black_box(w);
    let elapsed = t0.elapsed().as_secs_f64();
    let steps = (e * m * n) as f64;
    (elapsed, steps / elapsed.max(1e-9), checksum)
}
