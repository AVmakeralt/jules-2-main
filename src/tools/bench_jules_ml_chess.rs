use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let script = args.get(1).cloned().unwrap_or_else(|| "ml_chess.jules".to_string());

    let t0 = Instant::now();
    match jules::jules_run_file(&script, "main") {
        Ok(()) => {
            let dt = t0.elapsed().as_secs_f64();
            println!("Jules script `{}` elapsed: {:.3}s", script, dt);
            match python_baseline() {
                Ok(py) => {
                    println!("Python baseline elapsed: {:.3}s", py);
                    println!("Speedup (Python / Jules): {:.2}x", py / dt.max(1e-9));
                }
                Err(e) => println!("⚠️ Python baseline skipped: {e}"),
            }
        }
        Err(e) => {
            eprintln!("failed running script: {e}");
            std::process::exit(1);
        }
    }
}

fn python_baseline() -> Result<f64, String> {
    let code = r#"
import time

def run():
    # intentionally close to a full-board training-style inner loop
    w0, w1, w2 = 0.03, 0.10, -0.07
    checksum = 0.0
    for ep in range(5000):
        board = [1]*16 + [0]*32 + [2]*16
        white_adv = 0.0
        black_adv = 0.0
        for step in range(64):
            white_pawns = sum(1 for v in board if v == 1)
            black_pawns = sum(1 for v in board if v == 2)
            f0 = 1.0
            f1 = white_pawns - black_pawns
            f2 = white_adv - black_adv
            score = w0 * f0 + w1 * f1 + w2 * f2
            rew = 0.0
            if score >= 0.0:
                for i in range(63, -1, -1):
                    if board[i] == 2:
                        board[i] = 0
                        rew += 0.25
                        break
                white_adv += 1.0
            else:
                for i in range(64):
                    if board[i] == 1:
                        board[i] = 0
                        rew -= 0.20
                        break
                black_adv += 1.0
            eta = 0.005
            w0 += eta * rew * f0
            w1 += eta * rew * f1
            w2 += eta * rew * f2
            checksum += score * 0.001 + rew
    return checksum

t0 = time.perf_counter()
run()
print(time.perf_counter() - t0)
"#;

    let mut child = Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("python3 missing: {e}"))?;
    child
        .stdin
        .as_mut()
        .ok_or("stdin unavailable")?
        .write_all(code.as_bytes())
        .map_err(|e| format!("write failed: {e}"))?;
    let out = child.wait_with_output().map_err(|e| format!("wait failed: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).to_string());
    }
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<f64>()
        .map_err(|e| format!("parse failed: {e}"))
}
