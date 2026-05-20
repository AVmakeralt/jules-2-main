// =============================================================================
// bench_jit_comprehensive.rs — Comprehensive JIT Performance Benchmark
//
// Measures Jules JIT performance across multiple dimensions:
//   1. JIT Native vs Interpreter vs BytecodeVM vs Rust Native
//   2. Compilation throughput (how fast can the JIT compile?)
//   3. Loop-heavy workloads (LCG, SUM, COUNT, MULADD)
//   4. Nested loops (matrix-like patterns)
//   5. Function call overhead (identity, multi-arg)
//   6. Control flow (if-else, Collatz)
//   7. Arithmetic intensity (div/mod heavy)
//   8. Correctness validation for all paths
//
// Usage:
//   cargo run --release --bin bench-jit-comprehensive
//   cargo run --release --bin bench-jit-comprehensive [loops] [samples]
// =============================================================================

use std::hint::black_box;
use std::time::Instant;

use jules::{CompileUnit, DiagSeverity, Pipeline, PipelineResult};

const DEFAULT_LOOPS: usize = 100_000_000;  // 100M for JIT/Rust
const DEFAULT_SAMPLES: usize = 3;
const INTERP_LOOPS: usize = 1_000_000;     // 1M for interpreter

// =============================================================================
// WORKLOAD DEFINITIONS
// =============================================================================

struct Workload {
    name: &'static str,
    category: &'static str,
    jules_src: String,
    interp_src: String,
    rust_fn: fn(i32) -> i64,
    loop_count: i32,
    interp_loop_count: i32,
}

fn build_workloads(n: i32, interp_n: i32) -> Vec<Workload> {
    vec![
        // ── CATEGORY: LOOP COUNTERS ──
        Workload {
            name: "COUNT",
            category: "loop-counter",
            jules_src: format!(r#"
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
            interp_src: format!(r#"
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
            rust_fn: rust_count,
            loop_count: n,
            interp_loop_count: interp_n,
        },
        // ── CATEGORY: ARITHMETIC ──
        Workload {
            name: "LCG",
            category: "arithmetic",
            jules_src: format!(r#"
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
            interp_src: format!(r#"
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
            rust_fn: rust_lcg,
            loop_count: n,
            interp_loop_count: interp_n,
        },
        Workload {
            name: "SUM",
            category: "arithmetic",
            jules_src: format!(r#"
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
            interp_src: format!(r#"
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
            rust_fn: rust_sum,
            loop_count: n,
            interp_loop_count: interp_n,
        },
        Workload {
            name: "MULADD",
            category: "arithmetic",
            jules_src: format!(r#"
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
            interp_src: format!(r#"
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
            rust_fn: rust_muladd,
            loop_count: n,
            interp_loop_count: interp_n,
        },
        // ── CATEGORY: DIVISION ──
        Workload {
            name: "DIVMOD",
            category: "division",
            jules_src: format!(r#"
fn bench() -> i32 {{
  let mut s: i32 = 42;
  let mut i: i32 = 1;
  while i < {n} {{
    s = s + (s / i);
    s = s - (i / 7);
    i = i + 1;
  }}
  s
}}
"#),
            interp_src: format!(r#"
fn bench() -> i32 {{
  let mut s: i32 = 42;
  let mut i: i32 = 1;
  while i < {interp_n} {{
    s = s + (s / i);
    s = s - (i / 7);
    i = i + 1;
  }}
  s
}}
"#),
            rust_fn: rust_divmod,
            loop_count: n,
            interp_loop_count: interp_n,
        },
        // ── CATEGORY: CONTROL FLOW ──
        Workload {
            name: "COLLATZ",
            category: "control-flow",
            jules_src: format!(r#"
fn collatz(x: i32) -> i32 {{
  let mut n: i32 = x;
  let mut steps: i32 = 0;
  while n != 1 {{
    if n - (n / 2) * 2 == 0 {{
      n = n / 2;
    }} else {{
      n = 3 * n + 1;
    }}
    steps = steps + 1;
  }}
  steps
}}
fn bench() -> i32 {{
  let mut total: i32 = 0;
  let mut i: i32 = 1;
  while i <= {n} {{
    total = total + collatz(i);
    i = i + 1;
  }}
  total
}}
"#),
            interp_src: format!(r#"
fn collatz(x: i32) -> i32 {{
  let mut n: i32 = x;
  let mut steps: i32 = 0;
  while n != 1 {{
    if n - (n / 2) * 2 == 0 {{
      n = n / 2;
    }} else {{
      n = 3 * n + 1;
    }}
    steps = steps + 1;
  }}
  steps
}}
fn bench() -> i32 {{
  let mut total: i32 = 0;
  let mut i: i32 = 1;
  while i <= {interp_n} {{
    total = total + collatz(i);
    i = i + 1;
  }}
  total
}}
"#),
            rust_fn: rust_collatz,
            loop_count: n,
            interp_loop_count: interp_n,
        },
        // ── CATEGORY: NESTED LOOPS ──
        Workload {
            name: "NESTED-50x50",
            category: "nested-loop",
            jules_src: r#"
fn bench() -> i32 {
  let mut total: i32 = 0;
  let mut i: i32 = 0;
  while i < 50 {
    let mut j: i32 = 0;
    while j < 50 {
      total = total + i * j + 1;
      j = j + 1;
    }
    i = i + 1;
  }
  total
}
"#.to_string(),
            interp_src: r#"
fn bench() -> i32 {
  let mut total: i32 = 0;
  let mut i: i32 = 0;
  while i < 50 {
    let mut j: i32 = 0;
    while j < 50 {
      total = total + i * j + 1;
      j = j + 1;
    }
    i = i + 1;
  }
  total
}
"#.to_string(),
            rust_fn: rust_nested_50x50,
            loop_count: 2500, // 50*50
            interp_loop_count: 2500,
        },
        // ── CATEGORY: FUNCTION CALLS ──
        Workload {
            name: "CALL-ID",
            category: "calls",
            jules_src: format!(r#"
fn identity(x: i32) -> i32 {{ x }}
fn bench() -> i32 {{
  let mut s: i32 = 0;
  let mut i: i32 = 0;
  while i < {n} {{
    s = identity(s + 1);
    i = i + 1;
  }}
  s
}}
"#),
            interp_src: format!(r#"
fn identity(x: i32) -> i32 {{ x }}
fn bench() -> i32 {{
  let mut s: i32 = 0;
  let mut i: i32 = 0;
  while i < {interp_n} {{
    s = identity(s + 1);
    i = i + 1;
  }}
  s
}}
"#),
            rust_fn: rust_call_id,
            loop_count: n,
            interp_loop_count: interp_n,
        },
        // ── CATEGORY: BITWISE / CONDITIONAL ──
        Workload {
            name: "COND-ADD",
            category: "conditional",
            jules_src: format!(r#"
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
            interp_src: format!(r#"
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
            rust_fn: rust_cond_add,
            loop_count: n,
            interp_loop_count: interp_n,
        },
    ]
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

fn rust_divmod(n: i32) -> i64 {
    let mut s: i32 = 42;
    let mut i: i32 = 1;
    while i < n {
        s = s.wrapping_add(s / i);
        s = s.wrapping_sub(i / 7);
        i = i.wrapping_add(1);
    }
    black_box(s as i64)
}

fn rust_collatz(n: i32) -> i64 {
    fn collatz(x: i32) -> i32 {
        let mut v = x;
        let mut steps = 0;
        while v != 1 {
            if v % 2 == 0 {
                v = v / 2;
            } else {
                v = 3 * v + 1;
            }
            steps += 1;
        }
        steps
    }
    let mut total: i32 = 0;
    for i in 1..=n {
        total += collatz(i);
    }
    black_box(total as i64)
}

fn rust_nested_50x50(_n: i32) -> i64 {
    let mut total: i32 = 0;
    let mut i: i32 = 0;
    while i < 50 {
        let mut j: i32 = 0;
        while j < 50 {
            total = total.wrapping_add(i.wrapping_mul(j).wrapping_add(1));
            j = j.wrapping_add(1);
        }
        i = i.wrapping_add(1);
    }
    black_box(total as i64)
}

fn rust_call_id(n: i32) -> i64 {
    #[inline(never)]
    fn identity(x: i32) -> i32 { x }
    let mut s: i32 = 0;
    let mut i: i32 = 0;
    while i < n {
        s = identity(s.wrapping_add(1));
        i = i.wrapping_add(1);
    }
    black_box(s as i64)
}

fn rust_cond_add(n: i32) -> i64 {
    let mut s: i32 = 0;
    let mut i: i32 = 0;
    while i < n {
        if i % 2 == 0 {
            s = s.wrapping_add(i);
        } else {
            s = s.wrapping_sub(1);
        }
        i = i.wrapping_add(1);
    }
    black_box(s as i64)
}

// =============================================================================
// HELPERS
// =============================================================================

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
    else if secs >= 0.001 { format!("{:.2}ms", secs * 1000.0) }
    else { format!("{:.0}us", secs * 1_000_000.0) }
}

fn compile_src(name: &str, src: &str) -> Option<PipelineResult> {
    let mut unit = CompileUnit::new(format!("<bench-jit:{}>", name), src);
    let result = Pipeline::new().run(&mut unit);
    if unit.has_errors() {
        for d in &unit.diags {
            if d.severity == DiagSeverity::Error {
                eprintln!("  COMPILE ERROR: {}", d.message);
            }
        }
        return None;
    }
    if result.program().is_none() {
        return None;
    }
    Some(result)
}

// =============================================================================
// BENCHMARK: JIT NATIVE (Phase 3 JIT)
// =============================================================================

fn bench_jit_native(wl: &Workload, samples: usize) -> (f64, i64, bool) {
    let result = match compile_src(wl.name, &wl.jules_src) {
        Some(r) => r,
        None => return (0.0, i64::MIN, false),
    };
    let prog = result.program().unwrap();

    // Warmup: let PGO kick in and compile the function natively
    let mut interp = jules::interp::Interpreter::new();
    interp.set_jit_enabled(true);
    interp.set_advance_jit_enabled(true);
    interp.set_native_jit_enabled(true);
    interp.load_program(prog);

    // Multiple warmup calls to trigger PGO windows
    let warmup_val = extract_i64(&interp.call_fn("bench", vec![]));
    // Second warmup to ensure JIT compilation is done
    let _ = interp.call_fn("bench", vec![]);
    // Small sleep to let any async compilation finish
    std::thread::sleep(std::time::Duration::from_millis(60));

    let rust_expected = (wl.rust_fn)(wl.loop_count);
    let correct = warmup_val == rust_expected;

    if !correct {
        eprintln!("  [{}] Native JIT mismatch (got {} expected {})",
            wl.name, warmup_val, rust_expected);
    }

    // Get JIT counters
    let (native_calls, vm_calls, fallback_calls) = interp.jit_counters();

    // Timed runs (reuse the same interpreter to keep native code cached)
    let mut total_s = 0.0f64;
    for _ in 0..samples {
        let start = Instant::now();
        let _ = black_box(interp.call_fn("bench", vec![]));
        total_s += start.elapsed().as_secs_f64();
    }
    let avg_s = total_s / samples as f64;

    eprintln!("  [{}] JIT counters: native={} vm={} fallback={}",
        wl.name, native_calls, vm_calls, fallback_calls);

    (avg_s, warmup_val, correct)
}

// =============================================================================
// BENCHMARK: INTERPRETER ONLY (no JIT)
// =============================================================================

fn bench_interp(wl: &Workload, samples: usize) -> (f64, i64, bool) {
    let result = match compile_src(wl.name, &wl.interp_src) {
        Some(r) => r,
        None => return (0.0, i64::MIN, false),
    };
    let prog = result.program().unwrap();

    let mut interp = jules::interp::Interpreter::new();
    interp.set_jit_enabled(false);
    interp.set_advance_jit_enabled(false);
    interp.set_native_jit_enabled(false);
    interp.load_program(prog);

    let warmup_val = extract_i64(&interp.call_fn("bench", vec![]));
    let rust_expected = (wl.rust_fn)(wl.interp_loop_count);
    let correct = warmup_val == rust_expected;

    let mut total_s = 0.0f64;
    for _ in 0..samples {
        let start = Instant::now();
        let _ = black_box(interp.call_fn("bench", vec![]));
        total_s += start.elapsed().as_secs_f64();
    }
    let avg_s = total_s / samples as f64;

    (avg_s, warmup_val, correct)
}

// =============================================================================
// BENCHMARK: BYTECODE VM ONLY
// =============================================================================

fn bench_bytecode_vm(wl: &Workload, samples: usize) -> (f64, i64, bool) {
    let result = match compile_src(wl.name, &wl.interp_src) {
        Some(r) => r,
        None => return (0.0, i64::MIN, false),
    };
    let prog = match result {
        PipelineResult::Ok(p) => p,
        PipelineResult::OkWithIr { program, .. } => program,
        _ => return (0.0, i64::MIN, false),
    };

    let mut vm = jules::runtime::bytecode_vm::BytecodeVM::new();
    if vm.load_program(&prog).is_err() {
        return (0.0, i64::MIN, false);
    }

    let warmup_val = extract_i64(&vm.call_fn("bench", vec![]));
    let rust_expected = (wl.rust_fn)(wl.interp_loop_count);
    let correct = warmup_val == rust_expected;

    let mut total_s = 0.0f64;
    for _ in 0..samples {
        let start = Instant::now();
        let _ = black_box(vm.call_fn("bench", vec![]));
        total_s += start.elapsed().as_secs_f64();
    }
    let avg_s = total_s / samples as f64;

    (avg_s, warmup_val, correct)
}

// =============================================================================
// BENCHMARK: RUST NATIVE BASELINE
// =============================================================================

fn bench_rust(wl: &Workload, samples: usize) -> f64 {
    let mut total_s = 0.0f64;
    for _ in 0..samples {
        let start = Instant::now();
        let _ = black_box((wl.rust_fn)(wl.loop_count));
        total_s += start.elapsed().as_secs_f64();
    }
    total_s / samples as f64
}

fn bench_rust_interp(wl: &Workload, samples: usize) -> f64 {
    let mut total_s = 0.0f64;
    for _ in 0..samples {
        let start = Instant::now();
        let _ = black_box((wl.rust_fn)(wl.interp_loop_count));
        total_s += start.elapsed().as_secs_f64();
    }
    total_s / samples as f64
}

// =============================================================================
// BENCHMARK: JIT COMPILATION THROUGHPUT
// =============================================================================

fn bench_compile_throughput(samples: usize) {
    println!();
    println!("═══ JIT COMPILATION THROUGHPUT ═══════════════════════════════════");
    println!();

    let test_sources = [
        ("tiny", "fn bench() -> i32 { 42 }"),
        ("small-loop", r#"fn bench() -> i32 {
  let mut s: i32 = 0;
  let mut i: i32 = 0;
  while i < 100 { s = s + i; i = i + 1; }
  s
}"#),
        ("medium-loop", r#"fn bench() -> i32 {
  let mut s: i32 = 0;
  let mut i: i32 = 0;
  while i < 10000 { s = s * 1664525 + 1013904223; i = i + 1; }
  s
}"#),
        ("nested-3", r#"fn bench() -> i32 {
  let mut c: i32 = 0;
  let mut i: i32 = 0;
  while i < 10 {
    let mut j: i32 = 0;
    while j < 10 {
      let mut k: i32 = 0;
      while k < 10 { c = c + 1; k = k + 1; }
      j = j + 1;
    }
    i = i + 1;
  }
  c
}"#),
    ];

    println!("┌─────────────┬───────────────┬───────────────┬────────────────┐");
    println!("│ Program     │ Compile (us)  │ Full Pipe(us) │ Native JIT(us) │");
    println!("├─────────────┼───────────────┼───────────────┼────────────────┤");

    for (name, src) in &test_sources {
        let mut compile_times = Vec::new();
        let mut full_times = Vec::new();
        let mut jit_times = Vec::new();

        for _ in 0..samples {
            // Measure full pipeline (compile + JIT)
            let full_start = Instant::now();
            let mut unit = CompileUnit::new(format!("<bench-compile:{}>", name), *src);
            let result = Pipeline::new().run(&mut unit);
            let compile_time = full_start.elapsed();
            let compile_us = compile_time.as_micros() as f64;
            compile_times.push(compile_us);

            if result.program().is_none() { continue; }
            full_times.push(compile_us);

            // Measure JIT compilation specifically
            let prog = result.program().unwrap();
            let mut interp = jules::interp::Interpreter::new();
            interp.set_jit_enabled(true);
            interp.set_advance_jit_enabled(true);
            interp.set_native_jit_enabled(true);
            interp.load_program(prog);

            // Force JIT compilation by calling once
            let jit_start = Instant::now();
            let _ = interp.call_fn("bench", vec![]);
            // Wait for PGO windows
            std::thread::sleep(std::time::Duration::from_millis(60));
            let _ = interp.call_fn("bench", vec![]);
            let jit_elapsed = jit_start.elapsed().as_micros() as f64;
            jit_times.push(jit_elapsed);
        }

        let compile_avg = compile_times.iter().sum::<f64>() / compile_times.len() as f64;
        let full_avg = full_times.iter().sum::<f64>() / full_times.len().max(1) as f64;
        let jit_avg = jit_times.iter().sum::<f64>() / jit_times.len().max(1) as f64;

        println!("│ {:11} │ {:>11.0}   │ {:>11.0}   │ {:>12.0}   │",
            name, compile_avg, full_avg, jit_avg);
    }

    println!("└─────────────┴───────────────┴───────────────┴────────────────┘");
}

// =============================================================================
// BENCHMARK: JIT WARMUP COST
// =============================================================================

fn bench_jit_warmup() {
    println!();
    println!("═══ JIT WARMUP LATENCY (time to first native execution) ═════════");
    println!();

    let src = r#"
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

    // Measure: compile + load + first call (interpreted)
    let mut unit = CompileUnit::new("<bench-warmup>".to_string(), src);
    let result = Pipeline::new().run(&mut unit);
    if result.program().is_none() {
        println!("  COMPILE FAILED");
        return;
    }
    let prog = result.program().unwrap();

    // Interpreted first call (no JIT)
    let mut interp_no_jit = jules::interp::Interpreter::new();
    interp_no_jit.set_jit_enabled(false);
    interp_no_jit.load_program(prog);
    let start = Instant::now();
    let _ = interp_no_jit.call_fn("bench", vec![]);
    let interp_first = start.elapsed();

    // JIT warmup sequence: measure each call individually
    let mut interp_jit = jules::interp::Interpreter::new();
    interp_jit.set_jit_enabled(true);
    interp_jit.set_advance_jit_enabled(true);
    interp_jit.set_native_jit_enabled(true);
    interp_jit.load_program(prog);

    let call1_start = Instant::now();
    let _ = interp_jit.call_fn("bench", vec![]);
    let call1 = call1_start.elapsed();

    // Wait for PGO window (5ms)
    std::thread::sleep(std::time::Duration::from_millis(60));

    let call2_start = Instant::now();
    let _ = interp_jit.call_fn("bench", vec![]);
    let call2 = call2_start.elapsed();

    let call3_start = Instant::now();
    let _ = interp_jit.call_fn("bench", vec![]);
    let call3 = call3_start.elapsed();

    let (native, vm, fallback) = interp_jit.jit_counters();

    println!("  Interpreted first call:  {}", fmt_dur(interp_first.as_secs_f64()));
    println!("  JIT call 1 (cold):       {}", fmt_dur(call1.as_secs_f64()));
    println!("  JIT call 2 (after PGO):  {}", fmt_dur(call2.as_secs_f64()));
    println!("  JIT call 3 (warm):       {}", fmt_dur(call3.as_secs_f64()));
    println!("  JIT counters: native={} vm={} fallback={}", native, vm, fallback);

    if call3.as_secs_f64() > 0.0 && interp_first.as_secs_f64() > 0.0 {
        let speedup = interp_first.as_secs_f64() / call3.as_secs_f64();
        println!("  JIT warm speedup:        {:.1}x over interpreted", speedup);
    }
}

// =============================================================================
// MAIN
// =============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: i32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_LOOPS as i32);
    let interp_n: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(INTERP_LOOPS as i32);
    let samples: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_SAMPLES);

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║        COMPREHENSIVE JIT PERFORMANCE BENCHMARK                          ║");
    println!("║        Jules Language — Phase 3 JIT vs Interpreter vs Rust              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  JIT/Rust loops:     {} ({:.0}M)", n, n as f64 / 1e6);
    println!("  Interp loops:       {} ({:.0}M) [extrapolated to JIT N]", interp_n, interp_n as f64 / 1e6);
    println!("  Samples per test:   {}", samples);
    println!("  PID:                {}", std::process::id());
    println!();

    let workloads = build_workloads(n, interp_n);

    // ══════════════════════════════════════════════════════════════════════════
    // PHASE 1: JIT NATIVE vs RUST (at full N)
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══ PHASE 1: JIT NATIVE vs RUST ({} iterations) ════════════════", n);
    println!();
    println!("┌──────────────┬────────────┬────────────┬──────────┬─────────┬──────────┐");
    println!("│ Workload     │  JIT (s)   │  Rust (s)  │ J/R (x)  │ Correct │ Miter/s  │");
    println!("├──────────────┼────────────┼────────────┼──────────┼─────────┼──────────┤");

    let mut jit_total = 0.0f64;
    let mut rust_total = 0.0f64;
    let mut ok_count = 0usize;
    let mut phase1_results: Vec<(&str, f64, f64, bool)> = Vec::new();

    for wl in &workloads {
        eprint!("  {:12} JIT+Rust...", wl.name);
        let (jit_s, _val, correct) = bench_jit_native(wl, samples);
        let rust_s = bench_rust(wl, samples);
        let ratio = if rust_s > 0.0 { jit_s / rust_s } else { 0.0 };
        let mips = if jit_s > 0.0 { (wl.loop_count as f64 / 1e6) / jit_s } else { 0.0 };
        eprintln!(" {:.2}x", ratio);

        println!("│ {:12} │ {:>10} │ {:>10} │ {:>8.2} │ {:>7} │ {:>7.0}  │",
            wl.name, fmt_dur(jit_s), fmt_dur(rust_s), ratio,
            if correct { "PASS" } else { "FAIL" }, mips);

        phase1_results.push((wl.name, jit_s, rust_s, correct));
        if correct && jit_s > 0.0 && rust_s > 0.0 {
            jit_total += jit_s;
            rust_total += rust_s;
            ok_count += 1;
        }
    }
    println!("└──────────────┴────────────┴────────────┴──────────┴─────────┴──────────┘");

    if ok_count > 0 {
        println!();
        println!("  Aggregate JIT/Rust: {:.2}x ({} workloads, {:.0}M iterations each)",
            jit_total / rust_total, ok_count, n as f64 / 1e6);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // PHASE 2: INTERPRETER vs RUST (at reduced N, extrapolate)
    // ══════════════════════════════════════════════════════════════════════════
    println!();
    println!("═══ PHASE 2: INTERPRETER vs RUST ({} iterations, extrapolate to {}) ══", interp_n, n);
    println!();
    println!("┌──────────────┬────────────┬────────────┬──────────┬──────────┬──────────┐");
    println!("│ Workload     │ Interp (s) │  Rust (s)  │ I/R (x)  │Extrap J  │ Correct  │");
    println!("├──────────────┼────────────┼────────────┼──────────┼──────────┤");

    for wl in &workloads {
        eprint!("  {:12} Interp...", wl.name);
        let (interp_s, _val, correct) = bench_interp(wl, samples);
        let rust_s = bench_rust_interp(wl, samples);
        let ratio = if rust_s > 0.0 { interp_s / rust_s } else { 0.0 };
        let scale = if wl.interp_loop_count > 0 {
            wl.loop_count as f64 / wl.interp_loop_count as f64
        } else {
            1.0
        };
        let extrap = interp_s * scale;
        let extrap_jit_ratio = if let Some((_, jit_s, _, _)) = phase1_results.iter().find(|(n, _, _, _)| *n == wl.name) {
            if *jit_s > 0.0 && extrap > 0.0 { extrap / jit_s } else { 0.0 }
        } else { 0.0 };
        eprintln!(" {:.1}x", ratio);

        println!("│ {:12} │ {:>10} │ {:>10} │ {:>8.1} │ {:>6.1}x   │ {:>7}  │",
            wl.name,
            fmt_dur(interp_s),
            fmt_dur(rust_s),
            ratio,
            extrap_jit_ratio,
            if correct { "PASS" } else { "FAIL" });
    }
    println!("└──────────────┴────────────┴────────────┴──────────┴──────────┴──────────┘");

    // ══════════════════════════════════════════════════════════════════════════
    // PHASE 3: BYTECODE VM vs RUST (at reduced N)
    // ══════════════════════════════════════════════════════════════════════════
    println!();
    println!("═══ PHASE 3: BYTECODE VM vs RUST ({} iterations) ════════════════", interp_n);
    println!();
    println!("┌──────────────┬────────────┬────────────┬──────────┬─────────┐");
    println!("│ Workload     │  VM (s)    │  Rust (s)  │ V/R (x)  │ Correct │");
    println!("├──────────────┼────────────┼────────────┼──────────┼─────────┤");

    for wl in &workloads {
        eprint!("  {:12} BytecodeVM...", wl.name);
        let (vm_s, _val, correct) = bench_bytecode_vm(wl, samples);
        let rust_s = bench_rust_interp(wl, samples);
        let ratio = if rust_s > 0.0 { vm_s / rust_s } else { 0.0 };
        eprintln!(" {:.1}x", ratio);

        println!("│ {:12} │ {:>10} │ {:>10} │ {:>8.1} │ {:>7} │",
            wl.name, fmt_dur(vm_s), fmt_dur(rust_s), ratio,
            if correct { "PASS" } else { "FAIL" });
    }
    println!("└──────────────┴────────────┴────────────┴──────────┴─────────┘");

    // ══════════════════════════════════════════════════════════════════════════
    // PHASE 4: JIT SPEEDUP OVER INTERPRETER
    // ══════════════════════════════════════════════════════════════════════════
    println!();
    println!("═══ PHASE 4: JIT SPEEDUP OVER INTERPRETER ══════════════════════");
    println!();
    println!("┌──────────────┬─────────────┬─────────────┬────────────┐");
    println!("│ Workload     │ Interp(s)★  │  JIT(s)     │Speedup(x)  │");
    println!("├──────────────┼─────────────┼─────────────┼────────────┤");

    for wl in &workloads {
        let (interp_s, _, _) = bench_interp(wl, samples);
        let (jit_s, _, _) = bench_jit_native(wl, 1); // quick

        // Scale interpreter time to JIT's loop count
        let scale = if wl.interp_loop_count > 0 {
            wl.loop_count as f64 / wl.interp_loop_count as f64
        } else { 1.0 };
        let interp_scaled = interp_s * scale;
        let speedup = if jit_s > 0.0 { interp_scaled / jit_s } else { 0.0 };

        println!("│ {:12} │ {:>11} │ {:>11} │ {:>10.1} │",
            wl.name, fmt_dur(interp_scaled), fmt_dur(jit_s), speedup);
    }
    println!("└──────────────┴─────────────┴─────────────┴────────────┘");
    println!("  ★ Interp time extrapolated from {} to {} iterations", interp_n, n);

    // ══════════════════════════════════════════════════════════════════════════
    // PHASE 5: JIT WARMUP
    // ══════════════════════════════════════════════════════════════════════════
    bench_jit_warmup();

    // ══════════════════════════════════════════════════════════════════════════
    // PHASE 6: COMPILATION THROUGHPUT
    // ══════════════════════════════════════════════════════════════════════════
    bench_compile_throughput(samples);

    // ══════════════════════════════════════════════════════════════════════════
    // FINAL SUMMARY
    // ══════════════════════════════════════════════════════════════════════════
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    PERFORMANCE SUMMARY                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    if ok_count > 0 {
        println!("  JIT / Rust aggregate:  {:.2}x  (lower is better; 1.0x = Rust parity)",
            jit_total / rust_total);
    }

    let pass_count = phase1_results.iter().filter(|(_, _, _, ok)| *ok).count();
    let fail_count = phase1_results.len() - pass_count;
    println!("  Correctness:           {} passed, {} failed", pass_count, fail_count);

    // Per-workload throughput
    println!();
    println!("  Per-workload throughput (M iterations/second):");
    println!("  ┌──────────────┬───────────┬───────────┬───────────┐");
    println!("  │ Workload     │    JIT    │   Rust    │  Ratio    │");
    println!("  ├──────────────┼───────────┼───────────┼───────────┤");
    for (name, jit_s, rust_s, ok) in &phase1_results {
        if *ok && *jit_s > 0.0 && *rust_s > 0.0 {
            let mips_jit = (n as f64 / 1e6) / jit_s;
            let mips_rust = (n as f64 / 1e6) / rust_s;
            println!("  │ {:12} │ {:>7.0}   │ {:>7.0}   │ {:>7.2}x  │",
                name, mips_jit, mips_rust, jit_s / rust_s);
        }
    }
    println!("  └──────────────┴───────────┴───────────┴───────────┘");

    println!();
    println!("  Benchmark complete. N = {} ({:.0}M iterations per workload)", n, n as f64 / 1e6);
}
