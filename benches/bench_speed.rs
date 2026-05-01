// =============================================================================
// JULES QUICK SPEED BENCHMARK
// =============================================================================

use std::hint::black_box;
use std::time::Instant;

use jules::{CompileUnit, DiagSeverity, Pipeline, PipelineResult};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         JULES QUICK SPEED BENCHMARK                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Test 1: Simple loop
    bench_simple_loop(1_000);

    // Test 2: JIT vs interpreter
    bench_jit_vs_interp(1_000);

    // Test 3: Compile speed
    bench_compile_speed();

    // Test 4: Rust baselines
    bench_rust_baselines();

    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  Done.");
}

fn bench_simple_loop(n: i32) {
    let jules_src = format!(
        r#"
fn bench() -> i32 {{
  let mut s: i32 = 1;
  let mut i: i32 = 0;
  while i < {n} {{
    s = s + i;
    i = i + 1;
  }}
  s
}}
"#
    );

    println!("┌─ Simple Loop ({} iterations) ──────────────────────┐", n);

    let pipeline = Pipeline::new();
    let mut unit = CompileUnit::new("<loop>".to_string(), &jules_src);
    let result = pipeline.run(&mut unit);

    if unit.has_errors() {
        for d in &unit.diags {
            if d.severity == DiagSeverity::Error {
                eprintln!("  COMPILE ERROR: {}", d.message);
            }
        }
        println!("│  Jules: COMPILE FAILED");
        println!("└────────────────────────────────────────────────────────────┘");
        println!();
        return;
    }

    let program = match result {
        PipelineResult::Ok(p) => p,
        _ => {
            println!("│  Jules: pipeline failed");
            println!("└────────────────────────────────────────────────────────────┘");
            println!();
            return;
        }
    };

    let mut interp = jules::interp::Interpreter::new();
    interp.load_program(&program);

    // Warmup
    let _ = interp.call_fn("bench", vec![]);

    // Measure
    let samples = 3;
    let mut jules_times = Vec::new();
    let mut jules_result = 0i64;
    for _ in 0..samples {
        let start = Instant::now();
        match interp.call_fn("bench", vec![]) {
            Ok(jules::interp::Value::I32(v)) => jules_result = v as i64,
            Ok(jules::interp::Value::I64(v)) => jules_result = v,
            Ok(v) => eprintln!("  unexpected: {:?}", v.type_name()),
            Err(e) => eprintln!("  runtime error: {}", e.message),
        }
        jules_times.push(start.elapsed());
    }
    let jules_best = *jules_times.iter().min().unwrap();
    let jules_avg: std::time::Duration = jules_times.iter().sum::<std::time::Duration>() / samples as u32;

    // Rust baseline
    let mut rust_times = Vec::new();
    let mut rust_result = 0i64;
    for _ in 0..samples {
        let start = Instant::now();
        rust_result = rust_simple_loop(n);
        rust_times.push(start.elapsed());
    }
    let rust_best = *rust_times.iter().min().unwrap();
    let rust_avg: std::time::Duration = rust_times.iter().sum::<std::time::Duration>() / samples as u32;

    let ratio = jules_avg.as_secs_f64() / rust_avg.as_secs_f64();
    let ns_per_iter_jules = jules_avg.as_nanos() as f64 / n as f64;
    let ns_per_iter_rust = rust_avg.as_nanos() as f64 / n as f64;

    println!("│  Jules: avg={:?}  best={:?}  ({:.1} ns/iter)", jules_avg, jules_best, ns_per_iter_jules);
    println!("│  Rust:  avg={:?}  best={:?}  ({:.1} ns/iter)", rust_avg, rust_best, ns_per_iter_rust);
    println!("│  Ratio: {:.1}x (Jules/Rust)", ratio);
    if jules_result == rust_result {
        println!("│  Correctness: PASS");
    } else {
        println!("│  Correctness: MISMATCH (jules={} rust={})", jules_result, rust_result);
    }
    println!("└────────────────────────────────────────────────────────────┘");
    println!();
}

fn rust_simple_loop(n: i32) -> i64 {
    let mut s: i32 = 1;
    for i in 0..black_box(n) {
        s = black_box(s + i);
    }
    black_box(s as i64)
}

fn bench_jit_vs_interp(n: i32) {
    let jules_src = format!(
        r#"
fn bench() -> i32 {{
  let mut s: i32 = 1;
  let mut i: i32 = 0;
  while i < {n} {{
    s = s + i;
    i = i + 1;
  }}
  s
}}
"#
    );

    println!("┌─ JIT vs Interpreter ({} iterations) ───────────────────┐", n);

    let pipeline = Pipeline::new();
    let mut unit = CompileUnit::new("<jit-bench>".to_string(), &jules_src);
    let result = pipeline.run(&mut unit);

    if unit.has_errors() {
        println!("│  Compile failed, skipping");
        println!("└────────────────────────────────────────────────────────────┘");
        println!();
        return;
    }

    let program = match result {
        PipelineResult::Ok(p) => p,
        _ => {
            println!("│  Pipeline failed, skipping");
            println!("└────────────────────────────────────────────────────────────┘");
            println!();
            return;
        }
    };

    // Interpreter mode
    let mut interp = jules::interp::Interpreter::new();
    interp.set_jit_enabled(false);
    interp.load_program(&program);
    let _ = interp.call_fn("bench", vec![]);

    let mut interp_times = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        let _ = interp.call_fn("bench", vec![]);
        interp_times.push(start.elapsed());
    }
    let interp_avg: std::time::Duration = interp_times.iter().sum::<std::time::Duration>() / 3;

    // JIT mode
    let mut interp_jit = jules::interp::Interpreter::new();
    interp_jit.set_jit_enabled(true);
    interp_jit.set_advance_jit_enabled(true);
    interp_jit.load_program(&program);
    let _ = interp_jit.call_fn("bench", vec![]);

    let mut jit_times = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        let _ = interp_jit.call_fn("bench", vec![]);
        jit_times.push(start.elapsed());
    }
    let jit_avg: std::time::Duration = jit_times.iter().sum::<std::time::Duration>() / 3;

    let speedup = interp_avg.as_secs_f64() / jit_avg.as_secs_f64();

    println!("│  Interpreter: avg={:?}", interp_avg);
    println!("│  JIT:         avg={:?}", jit_avg);
    println!("│  JIT speedup: {:.2}x over interpreter", speedup);
    println!("└────────────────────────────────────────────────────────────┘");
    println!();
}

fn bench_compile_speed() {
    println!("┌─ Compile Speed ────────────────────────────────────────────┐");

    let sources = vec![
        ("empty fn", "fn main() {}\n"),
        ("arith expr", "fn main() -> i32 { 2 + 3 * 4 }\n"),
        ("let x = 1", "fn main() { let x: i32 = 1; }\n"),
        ("if/else", "fn main() -> i32 { if 1 > 0 { 1 } else { 0 } }\n"),
    ];

    for (name, src) in sources {
        let pipeline = Pipeline::new();
        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            let mut unit = CompileUnit::new("<bench>".to_string(), src);
            let _ = pipeline.run(&mut unit);
        }
        let total = start.elapsed();
        let per_iter = total / iters as u32;
        println!("│  {:20}  avg={:?}", name, per_iter);
    }
    println!("└────────────────────────────────────────────────────────────┘");
    println!();
}

fn bench_rust_baselines() {
    println!("┌─ Rust Native Baselines ────────────────────────────────────┐");

    // Integer loop (10k)
    let start = Instant::now();
    let mut s: i32 = 0;
    for i in 0..10_000i32 { s = s.wrapping_add(i); }
    let loop_time = start.elapsed();
    println!("│  Integer loop (10k iters): {:?}", loop_time);
    let _ = black_box(s);

    // Fibonacci
    fn fib(n: i32) -> i32 { if n <= 1 { n } else { fib(n-1) + fib(n-2) } }
    let start = Instant::now();
    let f = fib(black_box(30));
    let fib_time = start.elapsed();
    println!("│  Fibonacci(30): {:?}  result={}", fib_time, black_box(f));

    // Memory throughput
    let data: Vec<f32> = vec![1.0; 1_000_000];
    let start = Instant::now();
    let sum: f32 = data.iter().sum();
    let mem_time = start.elapsed();
    println!("│  Sum 1M f32:    {:?}  result={:.1}", mem_time, black_box(sum));

    println!("└────────────────────────────────────────────────────────────┘");
}
