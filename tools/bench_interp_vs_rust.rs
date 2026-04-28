use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;
use std::time::SystemTime;

use jules::{CompileUnit, Pipeline, PipelineResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchMode {
    NativeProbe,
    NativeJit,
    FullJit,
    FullInterp,
    TieredTracing,
    AotTime,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);
    let mode = parse_mode(args.get(3).map(String::as_str));
    if mode == BenchMode::NativeProbe {
        std::process::exit(run_native_probe());
    }
    let seed: i64 = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(default_seed);
    let samples: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(10);

    println!("bench-interp-vs-rust n={n} iters={iters} mode={mode:?} seed={seed}");

    let jules_src = if mode == BenchMode::TieredTracing {
        format!(
            r#"
fn bench() -> i32 {{
  let s = {seed};
  s * 1664525 + 97
}}
"#,
            seed = seed
        )
    } else {
        format!(
            r#"
fn bench() -> i32 {{
  let mut s = {seed};
  let mut i = 0;
  while i < {n} {{
    s = s * 1664525 + (i * 1013904223) + 97;
    i = i + 1;
  }}
  s
}}
"#,
            seed = seed,
            n = n
        )
    };

    // Jules compile benchmark (10 runs, averaged)
    let mut jules_compile_s = 0.0f64;
    let mut program = None;
    for sample in 0..samples {
        let pipeline = Pipeline::new();
        let compile_start = Instant::now();
        for i in 0..iters {
            let mut unit = CompileUnit::new(format!("<bench:{sample}:{i}>"), &jules_src);
            let result = pipeline.run(&mut unit);
            if unit.has_errors() {
                eprintln!(
                    "compile diagnostics: {} => {:?}",
                    unit.diags.len(),
                    unit.diags
                );
                std::process::exit(1);
            }
            if i + 1 == iters && sample + 1 == samples {
                match result {
                    PipelineResult::Ok(p) => program = Some(p),
                    _ => {
                        eprintln!("pipeline failed to produce executable program");
                        std::process::exit(1);
                    }
                }
            }
        }
        jules_compile_s += compile_start.elapsed().as_secs_f64();
    }
    jules_compile_s /= samples as f64;

    if mode == BenchMode::AotTime {
        let rust_compile_s = if mode == BenchMode::TieredTracing {
            None
        } else {
            rustc_compile_baseline(n)
        };
        println!(
            "Jules AoT compile(avg {samples}): {:.6}s total ({:.6}s/iter)",
            jules_compile_s,
            jules_compile_s / iters as f64
        );
        if let Some(rc) = rust_compile_s {
            println!("Rust AoT compile:            {:.6}s total (rustc -O)", rc);
            println!(
                "AoT compile ratio (Jules/Rust): {:.2}x",
                jules_compile_s / rc.max(1e-9)
            );
        } else {
            println!("Rust AoT compile:            skipped (rustc unavailable)");
        }
        return;
    }

    // Jules runtime benchmark
    let program = program.expect("program should exist");
    let mut interp = jules::interp::Interpreter::new();
    let mut using_jit = matches!(mode, BenchMode::FullJit | BenchMode::NativeJit);
    let using_native_jit = matches!(mode, BenchMode::NativeJit) && native_jit_available();
    if matches!(mode, BenchMode::NativeJit) && !using_native_jit {
        eprintln!("native JIT probe failed; falling back to VM JIT for this benchmark process.");
    }
    interp.set_jit_enabled(using_jit);
    interp.set_advance_jit_enabled(using_jit);
    interp.set_native_jit_enabled(using_native_jit);
    interp.load_program(&program);
    interp.reset_jit_counters();

    let (mut check_jules, effective_mode, mut jules_runtime_s, mut jules_checksum, jit_counts) = if mode == BenchMode::TieredTracing {
        let mut tiered = jules::tiered_compilation::TieredExecutionManager::new(
            jules::tiered_compilation::PromotionPolicy::max_performance(),
        );
        tiered.load_program(&program);
        tiered.force_compile_all(jules::tiered_compilation::Tier::Tier3_TracingJIT);

        let check = call_bench_i64_tiered(&mut tiered);
        let mut runtime_s = 0.0f64;
        let mut checksum = 0i64;
        for _ in 0..samples {
            let run_start = Instant::now();
            for _ in 0..iters {
                let v = call_bench_i64_tiered(&mut tiered);
                checksum = checksum.wrapping_add(black_box(v));
            }
            runtime_s += run_start.elapsed().as_secs_f64();
        }
        runtime_s /= samples as f64;
        let counts = (
            tiered.tracing_jit.traces_compiled,
            tiered.tracing_jit.traces_recorded,
            tiered.tracing_jit.deoptimizations,
        );
        (check, "Tracing-JIT-tier3", runtime_s, checksum, counts)
    } else {
        let mut check = call_bench_i64(&mut interp);
        if check == i64::MIN && using_jit {
            eprintln!("JIT path returned Unit for `bench`; retrying benchmark on interpreter path.");
            using_jit = false;
            interp.set_jit_enabled(false);
            interp.set_advance_jit_enabled(false);
            interp.set_native_jit_enabled(false);
            check = call_bench_i64(&mut interp);
        }
        let mut runtime_s = 0.0f64;
        let mut checksum = 0i64;
        for _ in 0..samples {
            let run_start = Instant::now();
            for _ in 0..iters {
                let v = call_bench_i64(&mut interp);
                checksum = checksum.wrapping_add(black_box(v));
            }
            runtime_s += run_start.elapsed().as_secs_f64();
        }
        runtime_s /= samples as f64;
        let (native_calls, vm_calls, fallback_calls) = interp.jit_counters();
        let mode_name = if !using_jit {
            "interp"
        } else if native_calls > 0 {
            "JIT-native"
        } else if vm_calls > 0 {
            "JIT-vm"
        } else {
            "interp-fallback"
        };
        (check, mode_name, runtime_s, checksum, (native_calls, vm_calls, fallback_calls))
    };
    let check_rust = if mode == BenchMode::TieredTracing {
        rust_kernel_trace(seed)
    } else {
        rust_kernel(n, seed)
    };
    if check_jules != check_rust {
        eprintln!(
            "correctness mismatch: jules={check_jules}, rust={check_rust}, n={n}, seed={seed}"
        );
        std::process::exit(2);
    }

    // Rust runtime baseline
    let mut rust_runtime_s = 0.0f64;
    let mut rust_checksum = 0i64;
    for _ in 0..samples {
        let rust_runtime_start = Instant::now();
        for _ in 0..iters {
            rust_checksum = rust_checksum.wrapping_add(if mode == BenchMode::TieredTracing {
                rust_kernel_trace(seed)
            } else {
                rust_kernel(n, seed)
            });
        }
        rust_runtime_s += rust_runtime_start.elapsed().as_secs_f64();
    }
    rust_runtime_s /= samples as f64;

    // Rust compile baseline (single rustc -O for equivalent source)
    let rust_compile_s = if mode == BenchMode::TieredTracing {
        None
    } else {
        rustc_compile_baseline(n)
    };

    println!(
        "Jules compile(avg {samples}): {:.6}s total ({:.6}s/iter)",
        jules_compile_s,
        jules_compile_s / iters as f64
    );
    println!(
        "Jules runtime [{}](avg {samples}): {:.6}s total ({:.6}s/iter) checksum={}",
        effective_mode,
        jules_runtime_s,
        jules_runtime_s / iters as f64,
        jules_checksum
    );
    println!(
        "Rust runtime(avg {samples}): {:.6}s total ({:.6}s/iter) checksum={}",
        rust_runtime_s,
        rust_runtime_s / iters as f64,
        rust_checksum
    );
    if let Some(rc) = rust_compile_s {
        println!("Rust compile:   {:.6}s total (rustc -O)", rc);
        println!(
            "Compile ratio (Jules/Rust): {:.2}x",
            jules_compile_s / rc.max(1e-9)
        );
    } else {
        println!("Rust compile:   skipped (rustc unavailable)");
    }
    println!(
        "Runtime ratio (Jules {} / Rust native): {:.2}x",
        effective_mode,
        jules_runtime_s / rust_runtime_s.max(1e-9)
    );
    println!(
        "JIT counters: a={} b={} c={}",
        jit_counts.0, jit_counts.1, jit_counts.2
    );
}

fn parse_mode(raw: Option<&str>) -> BenchMode {
    match raw {
        Some("aot") | Some("aot-time") => BenchMode::AotTime,
        Some("interp") => BenchMode::FullInterp,
        Some("native-probe") => BenchMode::NativeProbe,
        Some("native") | Some("native-jit") | Some("jit-native") => BenchMode::NativeJit,
        Some("jit") => BenchMode::FullJit,
        Some("tracing") | Some("tiered-tracing") | Some("trace") => BenchMode::TieredTracing,
        _ => BenchMode::NativeJit,
    }
}

fn call_bench_i64_tiered(mgr: &mut jules::tiered_compilation::TieredExecutionManager) -> i64 {
    match mgr.call_function("bench", vec![]) {
        Ok(jules::interp::Value::I32(n)) => n as i64,
        Ok(jules::interp::Value::I64(n)) => n,
        Ok(jules::interp::Value::Unit) => i64::MIN,
        Ok(_) | Err(_) => i64::MIN,
    }
}

fn native_jit_available() -> bool {
    let Ok(exe) = std::env::current_exe() else {
        return false;
    };
    let status = Command::new(exe)
        .arg("32")
        .arg("1")
        .arg("native-probe")
        .status();
    matches!(status, Ok(s) if s.success())
}

fn run_native_probe() -> i32 {
    let src = r#"
fn bench() -> i32 {
  let mut s = 1;
  let mut i = 0;
  while i < 32 {
    s = s * 1664525 + i + 97;
    i = i + 1;
  }
  s
}
"#;
    let pipeline = Pipeline::new();
    let mut unit = CompileUnit::new("<native-probe>", src);
    let program = match pipeline.run(&mut unit) {
        PipelineResult::Ok(p) if !unit.has_errors() => p,
        _ => return 2,
    };
    let mut interp = jules::interp::Interpreter::new();
    interp.set_jit_enabled(true);
    interp.set_advance_jit_enabled(true);
    interp.set_native_jit_enabled(true);
    interp.load_program(&program);
    let _ = interp.call_fn("bench", Vec::new());
    0
}

fn call_bench_i64(interp: &mut jules::interp::Interpreter) -> i64 {
    match interp.call_fn("bench", Vec::new()) {
        Ok(jules::interp::Value::I64(v)) => v,
        Ok(jules::interp::Value::I32(v)) => v as i64,
        Ok(jules::interp::Value::Unit) => i64::MIN,
        Ok(v) => {
            eprintln!("unexpected return value from jules kernel: {v:?}");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("runtime error: {}", e.message);
            std::process::exit(1);
        }
    }
}

fn rust_kernel(n: usize, seed: i64) -> i64 {
    let n = black_box(n as i32);
    let mut s: i32 = black_box(seed as i32);
    for i in 0..n {
        s = black_box(
            s.wrapping_mul(1_664_525)
                .wrapping_add(i.wrapping_mul(1_013_904_223))
                .wrapping_add(97),
        );
    }
    black_box(s as i64)
}

fn rust_kernel_trace(seed: i64) -> i64 {
    let s = seed as i32;
    s.wrapping_mul(1_664_525).wrapping_add(97) as i64
}

fn rustc_compile_baseline(n: usize) -> Option<f64> {
    let mut src_path = PathBuf::from(std::env::temp_dir());
    src_path.push(format!("jules_bench_{n}.rs"));
    let mut bin_path = PathBuf::from(std::env::temp_dir());
    bin_path.push(format!("jules_bench_{n}.bin"));

    let src = format!(
        r#"
#[inline(never)]
fn kernel(n: usize, seed: i64) -> i64 {{
    let mut s: i32 = std::hint::black_box(seed as i32);
    for i in 0..n as i32 {{
        s = std::hint::black_box(
            s.wrapping_mul(1_664_525)
                .wrapping_add(i.wrapping_mul(1_013_904_223))
                .wrapping_add(97)
        );
    }}
    std::hint::black_box(s as i64)
}}
fn main() {{
    let seed = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    println!("{{}}", kernel({n}, seed));
}}
"#
    );
    if fs::write(&src_path, src).is_err() {
        return None;
    }

    let start = Instant::now();
    let status = Command::new("rustc")
        .arg("-O")
        .arg(&src_path)
        .arg("-o")
        .arg(&bin_path)
        .status()
        .ok()?;
    let elapsed = start.elapsed().as_secs_f64();

    let _ = fs::remove_file(&src_path);
    let _ = fs::remove_file(&bin_path);

    if status.success() {
        Some(elapsed)
    } else {
        None
    }
}

fn default_seed() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(1)
}
