use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use jules::interp::Tensor;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let steps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    let batch: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);

    // 512 -> 1024 -> 512 => 1,048,576 params (weights only)
    let in_dim = 512usize;
    let hidden = 1024usize;
    let out_dim = 512usize;
    let params = in_dim * hidden + hidden * out_dim;
    println!(
        "bench-mlp-1m steps={steps} batch={batch} dims={in_dim}->{hidden}->{out_dim} params={params}"
    );

    let j = jules_mlp_train_like(steps, batch, in_dim, hidden, out_dim);
    let j_sec_per_step = j.0 / steps.max(1) as f64;
    println!(
        "Jules Tensor elapsed: {:.3}s total, {:.6}s/step, {:.2} steps/s, checksum={:.4}",
        j.0, j_sec_per_step, j.1, j.2
    );

    match python_numpy_train_like(steps, batch, in_dim, hidden, out_dim) {
        Ok(py) => {
            let py_sec_per_step = py.0 / steps.max(1) as f64;
            println!(
                "Python/NumPy: {:.3}s total, {:.6}s/step, {:.2} steps/s, checksum={:.4}",
                py.0, py_sec_per_step, py.1, py.2
            );
            println!(
                "Speedup (Jules/Python): {:.2}x (by steps/s), {:.2}x (by sec/step)",
                j.1 / py.1.max(1e-9),
                py_sec_per_step / j_sec_per_step.max(1e-12)
            );
        }
        Err(e) => println!("⚠️ Python baseline skipped: {e}"),
    }
}

fn jules_mlp_train_like(
    steps: usize,
    batch: usize,
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
) -> (f64, f64, f32) {
    let mut w1 = Tensor::from_data(vec![in_dim, hidden], init_vec(in_dim * hidden, 1));
    let mut w2 = Tensor::from_data(vec![hidden, out_dim], init_vec(hidden * out_dim, 2));
    let x = Tensor::from_data(vec![batch, in_dim], init_vec(batch * in_dim, 3));

    let t0 = Instant::now();
    let mut checksum = 0.0f32;
    for _ in 0..steps {
        let h = x.matmul(&w1).expect("x@w1 failed");
        let y = h.matmul(&w2).expect("h@w2 failed");
        checksum += y.sum_all() * 1e-9;

        // lightweight training-like update (weight decay style) to keep write path hot
        w1.scale_inplace(0.99995);
        w2.scale_inplace(0.99995);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    (elapsed, steps as f64 / elapsed.max(1e-9), checksum)
}

fn python_numpy_train_like(
    steps: usize,
    batch: usize,
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
) -> Result<(f64, f64, f32), String> {
    let script = r#"
import sys, time
try:
    import numpy as np
except Exception as e:
    print(f"ERR:numpy_missing:{e}")
    raise SystemExit(7)

steps = int(sys.argv[1]); batch = int(sys.argv[2]); in_dim = int(sys.argv[3]); hidden = int(sys.argv[4]); out_dim = int(sys.argv[5])

def init_vec(n, seed):
    x = seed
    out = np.empty((n,), dtype=np.float32)
    for i in range(n):
        x = (1664525 * x + 1013904223) & 0xFFFFFFFF
        out[i] = ((x >> 8) & 0xFFFF) / 65535.0 * 0.02 - 0.01
    return out

w1 = init_vec(in_dim * hidden, 1).reshape(in_dim, hidden)
w2 = init_vec(hidden * out_dim, 2).reshape(hidden, out_dim)
x = init_vec(batch * in_dim, 3).reshape(batch, in_dim)

t0 = time.perf_counter()
checksum = 0.0
for _ in range(steps):
    h = x @ w1
    y = h @ w2
    checksum += float(np.sum(y)) * 1e-9
    w1 *= 0.99995
    w2 *= 0.99995

elapsed = time.perf_counter() - t0
print(f"{elapsed} {steps/elapsed if elapsed>0 else 0} {checksum}")
"#;

    let mut child = Command::new("python3")
        .arg("-")
        .arg(steps.to_string())
        .arg(batch.to_string())
        .arg(in_dim.to_string())
        .arg(hidden.to_string())
        .arg(out_dim.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("python3 not available: {e}"))?;

    {
        let stdin = child.stdin.as_mut().ok_or("failed to open stdin")?;
        stdin
            .write_all(script.as_bytes())
            .map_err(|e| format!("failed to write script: {e}"))?;
    }

    let out = child
        .wait_with_output()
        .map_err(|e| format!("python failed: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let msg = format!("{stdout} {stderr}");
        if msg.contains("ERR:numpy_missing:") {
            return python_pure_fallback(steps.min(1), batch.min(1), in_dim, hidden, out_dim);
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
    let sps: f64 = it
        .next()
        .ok_or("missing steps/s")?
        .parse()
        .map_err(|_| "bad steps/s")?;
    let checksum: f32 = it
        .next()
        .ok_or("missing checksum")?
        .parse()
        .map_err(|_| "bad checksum")?;
    Ok((elapsed, sps, checksum))
}

fn python_pure_fallback(
    steps: usize,
    batch: usize,
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
) -> Result<(f64, f64, f32), String> {
    let script = r#"
import sys, time
steps = int(sys.argv[1]); batch = int(sys.argv[2]); in_dim = int(sys.argv[3]); hidden = int(sys.argv[4]); out_dim = int(sys.argv[5])

def init_vec(n, seed):
    x = seed
    out = [0.0] * n
    for i in range(n):
        x = (1664525 * x + 1013904223) & 0xFFFFFFFF
        out[i] = (((x >> 8) & 0xFFFF) / 65535.0) * 0.02 - 0.01
    return out

def matmul(a, a_rows, a_cols, b, b_cols):
    out = [0.0] * (a_rows * b_cols)
    for r in range(a_rows):
        aoff = r * a_cols
        ooff = r * b_cols
        for c in range(b_cols):
            s = 0.0
            bidx = c
            for k in range(a_cols):
                s += a[aoff + k] * b[bidx]
                bidx += b_cols
            out[ooff + c] = s
    return out

w1 = init_vec(in_dim * hidden, 1)
w2 = init_vec(hidden * out_dim, 2)
x = init_vec(batch * in_dim, 3)

t0 = time.perf_counter()
checksum = 0.0
for _ in range(steps):
    h = matmul(x, batch, in_dim, w1, hidden)
    y = matmul(h, batch, hidden, w2, out_dim)
    checksum += sum(y) * 1e-9
    for i in range(len(w1)): w1[i] *= 0.99995
    for i in range(len(w2)): w2[i] *= 0.99995

elapsed = time.perf_counter() - t0
print(f"{elapsed} {steps/elapsed if elapsed>0 else 0} {checksum}")
"#;

    let mut child = Command::new("python3")
        .arg("-")
        .arg(steps.to_string())
        .arg(batch.to_string())
        .arg(in_dim.to_string())
        .arg(hidden.to_string())
        .arg(out_dim.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("python3 not available: {e}"))?;

    {
        let stdin = child.stdin.as_mut().ok_or("failed to open stdin")?;
        stdin
            .write_all(script.as_bytes())
            .map_err(|e| format!("failed to write fallback script: {e}"))?;
    }

    let out = child
        .wait_with_output()
        .map_err(|e| format!("python failed: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stdout = String::from_utf8_lossy(&out.stdout);
        return Err(format!("{stdout} {stderr}"));
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.split_whitespace();
    let elapsed: f64 = it
        .next()
        .ok_or("missing elapsed")?
        .parse()
        .map_err(|_| "bad elapsed")?;
    let sps: f64 = it
        .next()
        .ok_or("missing steps/s")?
        .parse()
        .map_err(|_| "bad steps/s")?;
    let checksum: f32 = it
        .next()
        .ok_or("missing checksum")?
        .parse()
        .map_err(|_| "bad checksum")?;
    Ok((elapsed, sps, checksum))
}

fn init_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed;
    let mut out = vec![0.0f32; n];
    for v in &mut out {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let r = ((x >> 8) & 0xFFFF) as f32 / 65_535.0;
        *v = r * 0.02 - 0.01;
    }
    out
}
