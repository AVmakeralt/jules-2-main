// =============================================================================
// bench_jit_fast.rs — Fast JIT Performance Benchmark
//
// Streamlined benchmark that measures JIT vs Interpreter vs Rust in a single
// pass per workload. Designed to complete quickly.
//
// Usage:
//   cargo run --release --bin bench-jit-fast [loops]
//   Default: loops=10_000_000 (10M)
// =============================================================================

use std::hint::black_box;
use std::time::Instant;

use jules::{CompileUnit, DiagSeverity, Pipeline, PipelineResult};

fn extract_i64(val: &Result<jules::interp::Value, jules::interp::RuntimeError>) -> i64 {
    match val {
        Ok(jules::interp::Value::I32(n)) => *n as i64,
        Ok(jules::interp::Value::I64(n)) => *n,
        Ok(v) => { eprintln!("  unexpected type: {:?}", v); i64::MIN }
        Err(e) => { eprintln!("  runtime error: {}", e.message); i64::MIN }
    }
}

fn fmt_dur(secs: f64) -> String {
    if secs >= 1.0 { format!("{:.3}s", secs) }
    else if secs >= 0.001 { format!("{:.1}ms", secs * 1000.0) }
    else { format!("{:.0}us", secs * 1_000_000.0) }
}

struct BenchResult {
    name: &'static str,
    jit_s: f64,
    interp_s: f64,
    rust_s: f64,
    jit_correct: bool,
    interp_correct: bool,
    jit_native_calls: u64,
    jit_vm_calls: u64,
    jit_fallback_calls: u64,
    loop_count: i32,
    interp_loop_count: i32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: i32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10_000_000);
    let interp_n: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100_000);

    println!();
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║        JULES JIT PERFORMANCE BENCHMARK                    ║");
    println!("║        Phase 3 JIT vs Interpreter vs Rust Native          ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("  JIT/Rust loops:  {:>12} ({:.0}M)", n, n as f64 / 1e6);
    println!("  Interp loops:    {:>12} ({:.0}K)", interp_n, interp_n as f64 / 1e3);
    println!("  PID:             {}", std::process::id());
    println!();

    let mut results: Vec<BenchResult> = Vec::new();

    // ── COUNT ──
    results.push(run_bench(
        "COUNT",
        &format!(r#"
fn bench() -> i32 {{
  let mut count: i32 = 0;
  let mut i: i32 = 0;
  while i < {n} {{
    count = count + 1;
    i = i + 1;
  }}
  count
}}
"#),
        &format!(r#"
fn bench() -> i32 {{
  let mut count: i32 = 0;
  let mut i: i32 = 0;
  while i < {interp_n} {{
    count = count + 1;
    i = i + 1;
  }}
  count
}}
"#),
        rust_count, n, interp_n,
    ));

    // ── LCG ──
    results.push(run_bench(
        "LCG",
        &format!(r#"
fn bench() -> i32 {{
  let mut s: i32 = 42;
  let mut i: i32 = 0;
  while i < {n} {{
    s = s * 1664525 + 1013904223;
    i = i + 1;
  }}
  s
}}
"#),
        &format!(r#"
fn bench() -> i32 {{
  let mut s: i32 = 42;
  let mut i: i32 = 0;
  while i < {interp_n} {{
    s = s * 1664525 + 1013904223;
    i = i + 1;
  }}
  s
}}
"#),
        rust_lcg, n, interp_n,
    ));

    // ── SUM ──
    results.push(run_bench(
        "SUM",
        &format!(r#"
fn bench() -> i32 {{
  let mut s: i32 = 0;
  let mut i: i32 = 0;
  while i < {n} {{
    s = s + i;
    i = i + 1;
    s = s - i;
    i = i + 1;
  }}
  s
}}
"#),
        &format!(r#"
fn bench() -> i32 {{
  let mut s: i32 = 0;
  let mut i: i32 = 0;
  while i < {interp_n} {{
    s = s + i;
    i = i + 1;
    s = s - i;
    i = i + 1;
  }}
  s
}}
"#),
        rust_sum, n, interp_n,
    ));

    // ── MULADD ──
    results.push(run_bench(
        "MULADD",
        &format!(r#"
fn bench() -> i32 {{
  let mut a: i32 = 1;
  let mut b: i32 = 2;
  let mut i: i32 = 0;
  while i < {n} {{
    let tmp: i32 = a * 3 + b;
    a = b;
    b = tmp;
    i = i + 1;
  }}
  a
}}
"#),
        &format!(r#"
fn bench() -> i32 {{
  let mut a: i32 = 1;
  let mut b: i32 = 2;
  let mut i: i32 = 0;
  while i < {interp_n} {{
    let tmp: i32 = a * 3 + b;
    a = b;
    b = tmp;
    i = i + 1;
  }}
  a
}}
"#),
        rust_muladd, n, interp_n,
    ));

    // ── COLLATZ-200 ──
    results.push(run_bench(
        "COLLATZ-200",
        r#"
fn collatz(x: i32) -> i32 {
  let mut n: i32 = x;
  let mut steps: i32 = 0;
  while n != 1 {
    if n - (n / 2) * 2 == 0 {
      n = n / 2;
    } else {
      n = 3 * n + 1;
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
        r#"
fn collatz(x: i32) -> i32 {
  let mut n: i32 = x;
  let mut steps: i32 = 0;
  while n != 1 {
    if n - (n / 2) * 2 == 0 {
      n = n / 2;
    } else {
      n = 3 * n + 1;
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
        rust_collatz_200, 200, 200,
    ));

    // ── GCD-50x50 ──
    results.push(run_bench(
        "GCD-50x50",
        r#"
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
        r#"
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
        rust_gcd_50x50, 2500, 2500,
    ));

    // ── COND-ADD ──
    results.push(run_bench(
        "COND-ADD",
        &format!(r#"
fn bench() -> i32 {{
  let mut s: i32 = 0;
  let mut i: i32 = 0;
  while i < {n} {{
    if i - (i / 2) * 2 == 0 {{
      s = s + i;
    }} else {{
      s = s - 1;
    }}
    i = i + 1;
  }}
  s
}}
"#),
        &format!(r#"
fn bench() -> i32 {{
  let mut s: i32 = 0;
  let mut i: i32 = 0;
  while i < {interp_n} {{
    if i - (i / 2) * 2 == 0 {{
      s = s + i;
    }} else {{
      s = s - 1;
    }}
    i = i + 1;
  }}
  s
}}
"#),
        rust_cond_add, n, interp_n,
    ));

    // ── Main results table ──
    println!();
    println!("┌──────────────┬────────────┬────────────┬────────────┬──────────┬──────────┬──────────┬─────────┐");
    println!("│ Workload     │  JIT (s)   │ Interp(s)* │  Rust (s)  │ J/R (x)  │ I/R (x)* │ J/I (x)* │Correct  │");
    println!("├──────────────┼────────────┼────────────┼──────────┼──────────┼──────────┼──────────┼─────────┤");

    for r in &results {
        let jit_rust = if r.rust_s > 0.0 { r.jit_s / r.rust_s } else { 0.0 };
        let interp_rust = if r.rust_s > 0.0 && r.interp_s > 0.0 {
            let scale = r.loop_count as f64 / r.interp_loop_count as f64;
            (r.interp_s * scale) / r.rust_s
        } else { 0.0 };
        let jit_interp = if r.interp_s > 0.0 {
            let scale = r.loop_count as f64 / r.interp_loop_count as f64;
            r.jit_s / (r.interp_s * scale)
        } else { 0.0 };
        let correct_str = if r.jit_correct && r.interp_correct { "PASS" }
                          else if r.jit_correct { "JIT-ok" }
                          else { "FAIL" };

        println!("│ {:12} │ {:>10} │ {:>10} │ {:>10} │ {:>8.2} │ {:>8.1} │ {:>8.1} │ {:>7} │",
            r.name,
            fmt_dur(r.jit_s),
            fmt_dur(r.interp_s),
            fmt_dur(r.rust_s),
            jit_rust,
            interp_rust,
            jit_interp,
            correct_str);
    }

    println!("└──────────────┴────────────┴────────────┴────────────┴──────────┴──────────┴──────────┴─────────┘");
    println!("  * Interp times extrapolated from interp_loops to JIT loops");

    // ── JIT Counters ──
    println!();
    println!("═══ JIT COUNTERS (which execution path was used?) ══════════════");
    println!("┌──────────────┬─────────┬─────────┬──────────┐");
    println!("│ Workload     │ Native  │   VM    │ Fallback │");
    println!("├──────────────┼─────────┼─────────┼──────────┤");
    for r in &results {
        println!("│ {:12} │ {:>7} │ {:>7} │ {:>8} │",
            r.name, r.jit_native_calls, r.jit_vm_calls, r.jit_fallback_calls);
    }
    println!("└──────────────┴─────────┴─────────┴──────────┘");

    // ── Throughput ──
    println!();
    println!("═══ THROUGHPUT (M iterations/second) ═══════════════════════════");
    println!("┌──────────────┬───────────┬───────────┬───────────┐");
    println!("│ Workload     │    JIT    │   Rust    │  Ratio    │");
    println!("├──────────────┼───────────┼───────────┼───────────┤");
    for r in &results {
        if r.jit_s > 0.0 && r.rust_s > 0.0 && r.jit_correct {
            let mips_jit = (r.loop_count as f64 / 1e6) / r.jit_s;
            let mips_rust = (r.loop_count as f64 / 1e6) / r.rust_s;
            let ratio = r.jit_s / r.rust_s;
            println!("│ {:12} │ {:>7.0}   │ {:>7.0}   │ {:>7.2}x  │",
                r.name, mips_jit, mips_rust, ratio);
        }
    }
    println!("└──────────────┴───────────┴───────────┴───────────┘");

    // ── JIT Warmup Test ──
    println!();
    println!("═══ JIT WARMUP: time to first native execution ════════════════");

    let warmup_src = r#"
fn bench() -> i32 {
  let mut s: i32 = 42;
  let mut i: i32 = 0;
  while i < 10000 {
    s = s * 1664525 + 1013904223;
    i = i + 1;
  }
  s
}
"#;

    let mut unit = CompileUnit::new("<bench-warmup>".to_string(), warmup_src);
    let result = Pipeline::new().run(&mut unit);
    if !unit.has_errors() {
        if let Some(prog) = result.program() {
            // Interpreted
            let mut interp_no_jit = jules::interp::Interpreter::new();
            interp_no_jit.set_jit_enabled(false);
            interp_no_jit.load_program(prog);
            let start = Instant::now();
            let _ = interp_no_jit.call_fn("bench", vec![]);
            let interp_first = start.elapsed();

            // JIT cold
            let mut interp_jit = jules::interp::Interpreter::new();
            interp_jit.set_jit_enabled(true);
            interp_jit.set_advance_jit_enabled(true);
            interp_jit.set_native_jit_enabled(true);
            interp_jit.load_program(prog);

            let start = Instant::now();
            let _ = interp_jit.call_fn("bench", vec![]);
            let call1 = start.elapsed();

            // Wait for PGO
            std::thread::sleep(std::time::Duration::from_millis(60));

            let start = Instant::now();
            let _ = interp_jit.call_fn("bench", vec![]);
            let call2 = start.elapsed();

            let start = Instant::now();
            let _ = interp_jit.call_fn("bench", vec![]);
            let call3 = start.elapsed();

            let (nc, vc, fc) = interp_jit.jit_counters();

            println!("  Interpreted (no JIT):  {}", fmt_dur(interp_first.as_secs_f64()));
            println!("  JIT call 1 (cold):     {}", fmt_dur(call1.as_secs_f64()));
            println!("  JIT call 2 (after PGO):{}", fmt_dur(call2.as_secs_f64()));
            println!("  JIT call 3 (warm):     {}", fmt_dur(call3.as_secs_f64()));
            println!("  Counters: native={} vm={} fallback={}", nc, vc, fc);
            if call3.as_secs_f64() > 0.0 && interp_first.as_secs_f64() > 0.0 {
                println!("  JIT speedup (warm):    {:.1}x over interpreted",
                    interp_first.as_secs_f64() / call3.as_secs_f64());
            }
        }
    }

    // ── Summary ──
    let pass = results.iter().filter(|r| r.jit_correct).count();
    let total = results.len();
    let avg_jr: f64 = {
        let valid: Vec<f64> = results.iter()
            .filter(|r| r.jit_correct && r.jit_s > 0.0 && r.rust_s > 0.0)
            .map(|r| r.jit_s / r.rust_s)
            .collect();
        if valid.is_empty() { 0.0 } else { valid.iter().sum::<f64>() / valid.len() as f64 }
    };

    println!();
    println!("  ═══ SUMMARY ═══");
    println!("  Correctness:       {}/{} workloads pass", pass, total);
    println!("  Avg JIT/Rust:      {:.2}x  (lower = closer to native)", avg_jr);
    println!();
}

// =============================================================================
// BENCH RUNNER
// =============================================================================

fn run_bench(
    name: &'static str,
    jit_src: &str,
    interp_src: &str,
    rust_fn: fn(i32) -> i64,
    n: i32,
    interp_n: i32,
) -> BenchResult {
    eprint!("  {:12} bench...", name);

    // Compile JIT source
    let mut jit_unit = CompileUnit::new(format!("<jit:{}>", name), jit_src);
    let jit_result = Pipeline::new().run(&mut jit_unit);
    if jit_unit.has_errors() {
        for d in &jit_unit.diags {
            if d.severity == DiagSeverity::Error {
                eprintln!(" JIT COMPILE ERROR: {}", d.message);
            }
        }
        return BenchResult {
            name, jit_s: 0.0, interp_s: 0.0, rust_s: 0.0,
            jit_correct: false, interp_correct: false,
            jit_native_calls: 0, jit_vm_calls: 0, jit_fallback_calls: 0,
            loop_count: n, interp_loop_count: interp_n,
        };
    }
    let jit_prog = match jit_result.program() {
        Some(p) => p,
        None => {
            eprintln!(" NO PROGRAM");
            return BenchResult {
                name, jit_s: 0.0, interp_s: 0.0, rust_s: 0.0,
                jit_correct: false, interp_correct: false,
                jit_native_calls: 0, jit_vm_calls: 0, jit_fallback_calls: 0,
                loop_count: n, interp_loop_count: interp_n,
            };
        }
    };

    // ── JIT BENCHMARK ──
    let mut jit_interp = jules::interp::Interpreter::new();
    jit_interp.set_jit_enabled(true);
    jit_interp.set_advance_jit_enabled(true);
    jit_interp.set_native_jit_enabled(true);
    jit_interp.load_program(jit_prog);
    // Load IR module so JIT can use the SSA-direct path (register-allocated)
    // instead of falling back to the bytecode JIT (slot-array based)
    // NOTE: SSA path has a branch fixup bug that causes segfaults,
    // so we skip IR module loading until that's fixed.
    // if let PipelineResult::OkWithIr { ir_module, .. } = &jit_result {
    //     jit_interp.load_ir_module(ir_module.clone());
    // }

    // Warmup: multiple calls to trigger PGO windows
    let jit_val = extract_i64(&jit_interp.call_fn("bench", vec![]));
    let _ = jit_interp.call_fn("bench", vec![]);
    std::thread::sleep(std::time::Duration::from_millis(60));
    let _ = jit_interp.call_fn("bench", vec![]);

    let (jit_native_calls, jit_vm_calls, jit_fallback_calls) = jit_interp.jit_counters();

    let jit_start = Instant::now();
    let _ = black_box(jit_interp.call_fn("bench", vec![]));
    let jit_s = jit_start.elapsed().as_secs_f64();

    let rust_expected = rust_fn(n);
    let jit_correct = jit_val == rust_expected;

    // ── INTERPRETER BENCHMARK ──
    let mut interp_unit = CompileUnit::new(format!("<interp:{}>", name), interp_src);
    let interp_result = Pipeline::new().run(&mut interp_unit);
    let (interp_s, interp_correct) = if !interp_unit.has_errors() {
        if let Some(iprog) = interp_result.program() {
            let mut interp = jules::interp::Interpreter::new();
            interp.set_jit_enabled(false);
            interp.set_advance_jit_enabled(false);
            interp.set_native_jit_enabled(false);
            interp.load_program(iprog);

            let interp_val = extract_i64(&interp.call_fn("bench", vec![]));
            let interp_rust = rust_fn(interp_n);
            let correct = interp_val == interp_rust;

            let _ = interp.call_fn("bench", vec![]); // warmup

            let interp_start = Instant::now();
            let _ = black_box(interp.call_fn("bench", vec![]));
            let interp_s = interp_start.elapsed().as_secs_f64();

            (interp_s, correct)
        } else {
            (0.0, false)
        }
    } else {
        (0.0, false)
    };

    // ── RUST BASELINE ──
    let rust_start = Instant::now();
    let _ = black_box(rust_fn(n));
    let rust_s = rust_start.elapsed().as_secs_f64();

    eprintln!(" done");

    BenchResult {
        name, jit_s, interp_s, rust_s,
        jit_correct, interp_correct,
        jit_native_calls, jit_vm_calls, jit_fallback_calls,
        loop_count: n, interp_loop_count: interp_n,
    }
}

// =============================================================================
// RUST BASELINES
// =============================================================================

fn rust_count(n: i32) -> i64 {
    let mut count: i32 = 0;
    let mut i: i32 = 0;
    while i < n {
        count = count.wrapping_add(1);
        i = i.wrapping_add(1);
        black_box(&mut count);
    }
    count as i64
}

fn rust_lcg(n: i32) -> i64 {
    let mut s: i32 = 42;
    let mut i: i32 = 0;
    while i < n {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        i = i.wrapping_add(1);
        black_box(&mut s);
    }
    s as i64
}

fn rust_sum(n: i32) -> i64 {
    let mut s: i32 = 0;
    let mut i: i32 = 0;
    while i < n {
        s = s.wrapping_add(i);
        i = i.wrapping_add(1);
        s = s.wrapping_sub(i);
        i = i.wrapping_add(1);
    }
    black_box(s as i64)
}

fn rust_muladd(n: i32) -> i64 {
    let mut a: i32 = 1;
    let mut b: i32 = 2;
    let mut i: i32 = 0;
    while i < n {
        let tmp = a.wrapping_mul(3).wrapping_add(b);
        a = b;
        b = tmp;
        i = i.wrapping_add(1);
    }
    black_box(a as i64)
}

fn rust_collatz_200(_n: i32) -> i64 {
    fn collatz(x: i32) -> i32 {
        let mut v = x;
        let mut steps = 0;
        while v != 1 {
            if v % 2 == 0 { v = v / 2; } else { v = 3 * v + 1; }
            steps += 1;
        }
        steps
    }
    let mut total: i32 = 0;
    for i in 1..=200 { total += collatz(i); }
    black_box(total as i64)
}

fn rust_gcd_50x50(_n: i32) -> i64 {
    fn gcd(a: i32, b: i32) -> i32 {
        let mut x = a; let mut y = b;
        while y != 0 { let tmp = y; y = x - (x / y) * y; x = tmp; }
        x
    }
    let mut total: i32 = 0;
    for i in 1..=50 { for j in 1..=50 { total += gcd(i, j); } }
    black_box(total as i64)
}

fn rust_cond_add(n: i32) -> i64 {
    let mut s: i32 = 0;
    let mut i: i32 = 0;
    while i < n {
        if i % 2 == 0 { s = s.wrapping_add(i); } else { s = s.wrapping_sub(1); }
        i = i.wrapping_add(1);
    }
    black_box(s as i64)
}
