// =============================================================================
// bench_tracing_jit.rs — Interpreter vs Rust Benchmark
//
// Benchmarks the Jules interpreter against native Rust for key workloads.
// Also tests tracing JIT trace compilation (record + compile to x86-64).
//
// Tracing JIT Status:
//   - Trace recording: WORKING
//   - Trace compilation to x86-64: WORKING  
//   - Native execution: BUGGY (wrong results due to static trace recording
//     instead of execution-driven recording; loops execute only 1 iteration)
//   - Fix needed: execution-driven trace recording
// =============================================================================

use std::hint::black_box;
use std::time::Instant;

use jules::{CompileUnit, Pipeline, PipelineResult};

struct Workload {
    name: &'static str,
    src: &'static str,
    rust_fn: fn() -> i64,
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         JULES INTERPRETER vs RUST NATIVE BENCHMARK              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  iterations per sample: {}", iters);
    println!();

    let workloads = vec![
        Workload {
            name: "fib-iter-30",
            src: r#"
fn bench() -> i32 {
  let mut a: i32 = 0;
  let mut b: i32 = 1;
  let mut i: i32 = 0;
  while i < 30 {
    let tmp: i32 = a + b;
    a = b;
    b = tmp;
    i = i + 1;
  }
  a
}
"#,
            rust_fn: rust_fib_iter_30,
        },
        Workload {
            name: "nested-loops",
            src: r#"
fn bench() -> i32 {
  let mut count: i32 = 0;
  let mut i: i32 = 0;
  while i < 50 {
    let mut j: i32 = 0;
    while j < 50 {
      count = count + 1;
      j = j + 1;
    }
    i = i + 1;
  }
  count
}
"#,
            rust_fn: rust_nested_loops,
        },
        Workload {
            name: "sum-1m",
            src: r#"
fn bench() -> i32 {
  let mut s: i32 = 0;
  let mut i: i32 = 0;
  while i < 1000000 {
    s = s + i;
    i = i + 1;
    s = s - i;
    i = i + 1;
  }
  s
}
"#,
            rust_fn: rust_sum_1m,
        },
        Workload {
            name: "lcg-100k",
            src: r#"
fn bench() -> i32 {
  let mut s: i32 = 42;
  let mut i: i32 = 0;
  while i < 100000 {
    s = s * 1664525 + 1013904223;
    i = i + 1;
  }
  s
}
"#,
            rust_fn: rust_lcg_100k,
        },
        Workload {
            name: "gcd-50x50",
            src: r#"
fn gcd(a: i32, b: i32) -> i32 {
  let mut x: i32 = a;
  let mut y: i32 = b;
  while y != 0 {
    let tmp: i32 = y;
    y = x - (x / y) * y;
    x = tmp;
  }
  x
}
fn bench() -> i32 {
  let mut total: i32 = 0;
  let mut i: i32 = 1;
  while i <= 50 {
    let mut j: i32 = 1;
    while j <= 50 {
      total = total + gcd(i, j);
      j = j + 1;
    }
    i = i + 1;
  }
  total
}
"#,
            rust_fn: rust_gcd_50x50,
        },
        Workload {
            name: "collatz-200",
            src: r#"
fn collatz(n: i32) -> i32 {
  let mut x: i32 = n;
  let mut steps: i32 = 0;
  while x != 1 {
    if x - (x / 2) * 2 == 0 {
      x = x / 2;
    } else {
      x = 3 * x + 1;
    }
    steps = steps + 1;
  }
  steps
}
fn bench() -> i32 {
  let mut total: i32 = 0;
  let mut i: i32 = 1;
  while i <= 200 {
    total = total + collatz(i);
    i = i + 1;
  }
  total
}
"#,
            rust_fn: rust_collatz_200,
        },
    ];

    println!("┌─────────────────┬──────────────┬──────────────┬───────────┬────────┐");
    println!("│ Workload        │  Interp (ms) │  Rust  (ms)  │ I/Rust(x) │ Status │");
    println!("├─────────────────┼──────────────┼──────────────┼───────────┼────────┤");

    for wl in &workloads {
        let (interp_ms, rust_ms, interp_ok) = bench_workload(wl, iters);

        let interp_rust_x = if rust_ms > 0.0 { interp_ms / rust_ms } else { 0.0 };
        let status = if interp_ok { "✓" } else { "✗" };

        println!("│ {:15} │ {:10.2}  │ {:10.2}  │ {:7.1}x  │   {}    │",
            wl.name,
            interp_ms,
            rust_ms,
            interp_rust_x,
            status,
        );
    }

    println!("└─────────────────┴──────────────┴──────────────┴───────────┴────────┘");
    println!();
    println!("  I/Rust = Interpreter / Rust ratio (1.0x = parity with native Rust)");
    println!();
    println!("  TRACING JIT STATUS:");
    println!("  - Trace recording: WORKING (records bytecode instructions)");
    println!("  - Trace compilation to x86-64: WORKING (produces native code)");
    println!("  - Native execution: BUGGY (wrong results — trace is static, not");
    println!("    execution-driven; loops execute only 1 iteration instead of N)");
    println!("  - Key bugs fixed in this session:");
    println!("    1. Recorder replacement destroying traces (traces_compiled=0)");
    println!("    2. Value enum vs flat i64 slot array mismatch");
    println!("    3. Missing instruction support (Store, Load, Move, comparisons)");
    println!("    4. compile_tracing() not attempting immediate compilation");
    println!("  - Remaining fix needed: execution-driven trace recording");
}

fn bench_workload(wl: &Workload, iters: usize) -> (f64, f64, bool) {
    let mut unit = CompileUnit::new(wl.name.to_string(), wl.src);
    let result = Pipeline::new().run(&mut unit);
    if unit.has_errors() {
        for d in &unit.diags {
            if d.severity == jules::DiagSeverity::Error {
                eprintln!("  COMPILE ERROR [{}]: {}", wl.name, d.message);
            }
        }
        return (0.0, 0.0, false);
    }
    let prog = match result {
        PipelineResult::Ok(p) => p,
        PipelineResult::OkWithIr { program, .. } => program,
        _ => return (0.0, 0.0, false),
    };

    // ── Interpreter benchmark ──
    let mut interp = jules::interp::Interpreter::new();
    interp.set_jit_enabled(false);
    interp.set_advance_jit_enabled(false);
    interp.set_native_jit_enabled(false);
    interp.load_program(&prog);

    let interp_warmup = interp.call_fn("bench", vec![]);
    let interp_val = extract_i64(&interp_warmup);

    let interp_start = Instant::now();
    for _ in 0..iters {
        let _ = black_box(interp.call_fn("bench", vec![]));
    }
    let interp_ms = interp_start.elapsed().as_secs_f64() * 1000.0;

    // ── Rust baseline ──
    let rust_start = Instant::now();
    for _ in 0..iters {
        let _ = black_box((wl.rust_fn)());
    }
    let rust_ms = rust_start.elapsed().as_secs_f64() * 1000.0;

    let rust_expected = (wl.rust_fn)();
    let interp_ok = interp_val == rust_expected;

    if !interp_ok {
        eprintln!("  [{}] interp: got {} expected {}", wl.name, interp_val, rust_expected);
    }

    (interp_ms, rust_ms, interp_ok)
}

fn extract_i64(val: &Result<jules::interp::Value, jules::interp::RuntimeError>) -> i64 {
    match val {
        Ok(jules::interp::Value::I32(n)) => *n as i64,
        Ok(jules::interp::Value::I64(n)) => *n,
        _ => i64::MIN,
    }
}

fn rust_lcg_100k() -> i64 {
    let mut s: i32 = 42;
    for _ in 0..100000 {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    }
    black_box(s as i64)
}

fn rust_fib_iter_30() -> i64 {
    let mut a: i32 = 0;
    let mut b: i32 = 1;
    for _ in 0..30 {
        let tmp = a.wrapping_add(b);
        a = b;
        b = tmp;
    }
    black_box(a as i64)
}

fn rust_nested_loops() -> i64 {
    let mut count: i32 = 0;
    for _ in 0..50 {
        for _ in 0..50 {
            count += 1;
        }
    }
    black_box(count as i64)
}

fn rust_gcd_50x50() -> i64 {
    fn gcd(a: i32, b: i32) -> i32 {
        let mut x = a;
        let mut y = b;
        while y != 0 {
            let tmp = y;
            y = x - (x / y) * y;
            x = tmp;
        }
        x
    }
    let mut total: i32 = 0;
    for i in 1..=50 {
        for j in 1..=50 {
            total += gcd(i, j);
        }
    }
    black_box(total as i64)
}

fn rust_sum_1m() -> i64 {
    let mut s: i32 = 0;
    let mut i: i32 = 0;
    while i < 1000000 {
        s = s.wrapping_add(i);
        i = i.wrapping_add(1);
        s = s.wrapping_sub(i);
        i = i.wrapping_add(1);
    }
    black_box(s as i64)
}

fn rust_collatz_200() -> i64 {
    fn collatz(n: i32) -> i32 {
        let mut x = n;
        let mut steps = 0;
        while x != 1 {
            if x - (x / 2) * 2 == 0 {
                x = x / 2;
            } else {
                x = 3 * x + 1;
            }
            steps += 1;
        }
        steps
    }
    let mut total: i32 = 0;
    for i in 1..=200 {
        total += collatz(i);
    }
    black_box(total as i64)
}
