// =============================================================================
// bench_1b_jit.rs — 1-Billion Loop JIT vs Rust Benchmark
//
// Measures JIT-compiled Jules code against native Rust across multiple
// compute-heavy workloads at ~1B loop iterations.
// Interpreter is tested at 1M loops and scaled.
//
// Usage:
//   cargo run --release --bin bench-1b-jit [loops] [samples]
//   Default: loops=1_000_000_000, samples=3
// =============================================================================

use std::hint::black_box;
use std::time::Instant;

use jules::{CompileUnit, Pipeline, PipelineResult};

const DEFAULT_LOOPS: usize = 1_000_000_000;
const DEFAULT_SAMPLES: usize = 3;
const INTERP_SCALE: i32 = 1_000_000; // 1M for interpreter (then extrapolate)

struct Workload {
    name: &'static str,
    jules_src: String,
    interp_src: String, // same but with smaller N for interpreter
    rust_fn: fn(i32) -> i64,
    loop_count: i32,
}

fn build_workloads(n: i32) -> Vec<Workload> {
    vec![
        Workload {
            name: "LCG",
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
  while i < {INTERP_SCALE} {{
    s = s * 1664525 + 1013904223;
    i = i + 1;
  }}
  s
}}
"#),
            rust_fn: rust_lcg,
            loop_count: n,
        },
        Workload {
            name: "SUM",
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
  while i < {INTERP_SCALE} {{
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
        },
        Workload {
            name: "COUNT",
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
  while i < {INTERP_SCALE} {{
    count = count + 1;
    i = i + 1;
  }}
  count
}}
"#),
            rust_fn: rust_count,
            loop_count: n,
        },
        Workload {
            name: "MULADD",
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
  while i < {INTERP_SCALE} {{
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
        },
    ]
}

fn rust_lcg(n: i32) -> i64 {
    let mut s: i32 = 42;
    for _ in 0..n { s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223); }
    black_box(s as i64)
}
fn rust_sum(n: i32) -> i64 {
    let mut s: i32 = 0; let mut i: i32 = 0;
    while i < n { s = s.wrapping_add(i); i = i.wrapping_add(1); s = s.wrapping_sub(i); i = i.wrapping_add(1); }
    black_box(s as i64)
}
fn rust_count(n: i32) -> i64 {
    let mut count: i32 = 0; let mut i: i32 = 0;
    while i < n { count = count.wrapping_add(1); i = i.wrapping_add(1); }
    black_box(count as i64)
}
fn rust_muladd(n: i32) -> i64 {
    let mut a: i32 = 1; let mut b: i32 = 2;
    for _ in 0..n { let tmp = a.wrapping_mul(3).wrapping_add(b); a = b; b = tmp; }
    black_box(a as i64)
}

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

fn compile_src(name: &str, src: &str) -> Option<(PipelineResult, CompileUnit)> {
    let mut unit = CompileUnit::new(format!("<bench-1b:{}>", name), src);
    let result = Pipeline::new().run(&mut unit);
    if unit.has_errors() {
        for d in &unit.diags { if d.severity == jules::DiagSeverity::Error { eprintln!("  COMPILE ERROR: {}", d.message); } }
        return None;
    }
    if result.program().is_none() { return None; }
    Some((result, unit))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: i32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_LOOPS as i32);
    let samples: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_SAMPLES);

    println!();
    println!("================================================================");
    println!("  1-BILLION LOOP JIT vs RUST BENCHMARK");
    println!("================================================================");
    println!("  Loops (JIT/Rust):   {} ({:.2}B)", n, n as f64 / 1e9);
    println!("  Loops (Interp):     {} ({:.2}M) [extrapolated]", INTERP_SCALE, INTERP_SCALE as f64 / 1e6);
    println!("  Samples per test:   {}", samples);
    println!("  PID:                {}", std::process::id());
    println!();

    let workloads = build_workloads(n);

    // ══════════════════════════════════════════════════════════════════════════
    // PHASE 1: JIT vs Rust at 1B loops
    // ══════════════════════════════════════════════════════════════════════════
    println!("┌──────────────┬────────────┬────────────┬──────────┬─────────┐");
    println!("│ Workload     │  JIT (s)   │  Rust (s)  │ J/R (x)  │ Correct │");
    println!("├──────────────┼────────────┼────────────┼──────────┼─────────┤");

    let mut jit_total = 0.0f64;
    let mut rust_total = 0.0f64;
    let mut ok_count = 0usize;
    let mut results: Vec<(&str, f64, f64, bool)> = Vec::new();

    for wl in &workloads {
        eprint!("  {:12} JIT+Rust...", wl.name);
        let (jit_s, rust_s, correct) = bench_jit_vs_rust(wl, samples);
        let ratio = if rust_s > 0.0 { jit_s / rust_s } else { 0.0 };
        eprintln!(" {:.2}x", ratio);

        println!("│ {:12} │ {:>10} │ {:>10} │ {:>8.2} │ {:>7} │",
            wl.name, fmt_dur(jit_s), fmt_dur(rust_s), ratio,
            if correct { "PASS" } else { "FAIL" });

        results.push((wl.name, jit_s, rust_s, correct));
        if correct { jit_total += jit_s; rust_total += rust_s; ok_count += 1; }
    }
    println!("└──────────────┴────────────┴────────────┴──────────┴─────────┘");
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // PHASE 2: Interpreter at 1M loops (extrapolate to 1B)
    // ══════════════════════════════════════════════════════════════════════════
    println!("┌──────────────┬────────────┬────────────┬──────────┬──────────┐");
    println!("│ Workload     │ Interp(ms) │  Rust(ms)  │ I/R (x)  │Extrap 1B │");
    println!("├──────────────┼────────────┼────────────┼──────────┼──────────┤");

    for wl in &workloads {
        eprint!("  {:12} Interp...", wl.name);
        let (interp_s, rust_s, correct) = bench_interp_scaled(wl, samples);
        let ratio = if rust_s > 0.0 { interp_s / rust_s } else { 0.0 };
        let scale = n as f64 / INTERP_SCALE as f64;
        let extrap_1b = interp_s * scale;
        eprintln!(" {:.1}x", ratio);

        println!("│ {:12} │ {:>10} │ {:>10} │ {:>8.1} │ {:>7.1}s │",
            wl.name,
            fmt_dur(interp_s),
            fmt_dur(rust_s),
            ratio,
            fmt_dur(extrap_1b));

        let _ = correct;
    }
    println!("└──────────────┴────────────┴────────────┴──────────┴──────────┘");
    println!();

    // ══════════════════════════════════════════════════════════════════════════
    // AGGREGATE
    // ══════════════════════════════════════════════════════════════════════════
    if ok_count > 0 {
        let avg_jit_rust = jit_total / rust_total;
        println!("  ═══ AGGREGATE ═══");
        println!();
        println!("  JIT / Rust average: {:.2}x  ({} workloads, N={:.2}B)", avg_jit_rust, ok_count, n as f64 / 1e9);
        println!();
        println!("  Per-workload throughput:");
        for (name, jit_s, rust_s, ok) in &results {
            if *ok {
                let mips_jit = if *jit_s > 0.0 { (n as f64 / 1e6) / jit_s } else { 0.0 };
                let mips_rust = if *rust_s > 0.0 { (n as f64 / 1e6) / rust_s } else { 0.0 };
                println!("  {:12}: JIT {:>6.0} Miter/s | Rust {:>6.0} Miter/s | {:.2}x",
                    name, mips_jit, mips_rust, jit_s / rust_s.max(1e-9));
            }
        }
    }

    // JIT counters
    println!();
    println!("  ═══ JIT COUNTERS ═══");
    for wl in &workloads {
        if let Some((native, vm, fallback)) = bench_jit_counters(wl) {
            println!("  {:12}: native={} vm={} fallback={}", wl.name, native, vm, fallback);
        }
    }
    println!();
    println!("  Benchmark complete. N = {} ({:.2}B iterations per workload)", n, n as f64 / 1e9);
}

fn bench_jit_vs_rust(wl: &Workload, samples: usize) -> (f64, f64, bool) {
    let (result, _unit) = match compile_src(wl.name, &wl.jules_src) {
        Some(r) => r,
        None => return (0.0, 0.0, false),
    };
    let prog = result.program().unwrap();

    // Get the IR module for IR-direct JIT compilation
    let ir_module = result.ir_module().cloned();

    // JIT warmup + correctness
    let mut interp_jit = jules::interp::Interpreter::new();
    interp_jit.set_jit_enabled(true);
    interp_jit.set_advance_jit_enabled(true);
    interp_jit.set_native_jit_enabled(true);
    interp_jit.load_program(prog);
    if let Some(ref ir_mod) = ir_module {
        interp_jit.load_ir_module(ir_mod.clone());
    }
    let jit_val = extract_i64(&interp_jit.call_fn("bench", vec![]));
    let rust_expected = (wl.rust_fn)(wl.loop_count);

    let mut using_native = true;
    let mut correct = jit_val == rust_expected;

    if !correct {
        eprintln!("\n  [{}] Native JIT mismatch (got {} expected {}), trying VM JIT",
            wl.name, jit_val, rust_expected);
        let mut interp_vm = jules::interp::Interpreter::new();
        interp_vm.set_jit_enabled(true);
        interp_vm.set_advance_jit_enabled(true);
        interp_vm.set_native_jit_enabled(false);
        interp_vm.load_program(prog);
        if let Some(ref ir_mod) = ir_module {
            interp_vm.load_ir_module(ir_mod.clone());
        }
        let vm_val = extract_i64(&interp_vm.call_fn("bench", vec![]));
        if vm_val == rust_expected {
            using_native = false;
            correct = true;
        } else {
            eprintln!("  [{}] VM JIT also mismatched (got {})", wl.name, vm_val);
        }
    }

    // Timed JIT runs
    let mut jit_s = 0.0f64;
    for _ in 0..samples {
        let mut interp = jules::interp::Interpreter::new();
        interp.set_jit_enabled(true);
        interp.set_advance_jit_enabled(true);
        interp.set_native_jit_enabled(using_native);
        interp.load_program(prog);
        if let Some(ref ir_mod) = ir_module {
            interp.load_ir_module(ir_mod.clone());
        }
        let _ = interp.call_fn("bench", vec![]); // warmup
        interp.reset_jit_counters();

        let start = Instant::now();
        let _ = black_box(interp.call_fn("bench", vec![]));
        jit_s += start.elapsed().as_secs_f64();
    }
    jit_s /= samples as f64;

    // Rust baseline
    let mut rust_s = 0.0f64;
    for _ in 0..samples {
        let start = Instant::now();
        let _ = black_box((wl.rust_fn)(wl.loop_count));
        rust_s += start.elapsed().as_secs_f64();
    }
    rust_s /= samples as f64;

    (jit_s, rust_s, correct)
}

fn bench_interp_scaled(wl: &Workload, samples: usize) -> (f64, f64, bool) {
    let (result, _unit) = match compile_src(wl.name, &wl.interp_src) {
        Some(r) => r,
        None => return (0.0, 0.0, false),
    };
    let prog = result.program().unwrap();

    let mut interp = jules::interp::Interpreter::new();
    interp.set_jit_enabled(false);
    interp.set_advance_jit_enabled(false);
    interp.set_native_jit_enabled(false);
    interp.load_program(prog);
    let interp_val = extract_i64(&interp.call_fn("bench", vec![]));
    let rust_expected = (wl.rust_fn)(INTERP_SCALE);
    let correct = interp_val == rust_expected;

    if !correct {
        eprintln!("  [{}] Interp mismatch: got {} expected {}", wl.name, interp_val, rust_expected);
    }

    let mut interp_s = 0.0f64;
    for _ in 0..samples {
        let start = Instant::now();
        let _ = black_box(interp.call_fn("bench", vec![]));
        interp_s += start.elapsed().as_secs_f64();
    }
    interp_s /= samples as f64;

    let mut rust_s = 0.0f64;
    for _ in 0..samples {
        let start = Instant::now();
        let _ = black_box((wl.rust_fn)(INTERP_SCALE));
        rust_s += start.elapsed().as_secs_f64();
    }
    rust_s /= samples as f64;

    (interp_s, rust_s, correct)
}

fn bench_jit_counters(wl: &Workload) -> Option<(u64, u64, u64)> {
    let (result, _unit) = compile_src(wl.name, &wl.jules_src)?;
    let prog = result.program()?;
    let ir_module = result.ir_module().cloned();
    let mut interp = jules::interp::Interpreter::new();
    interp.set_jit_enabled(true);
    interp.set_advance_jit_enabled(true);
    interp.set_native_jit_enabled(true);
    interp.load_program(prog);
    if let Some(ir_mod) = ir_module {
        interp.load_ir_module(ir_mod);
    }
    interp.reset_jit_counters();
    let _ = interp.call_fn("bench", vec![]);
    Some(interp.jit_counters())
}
