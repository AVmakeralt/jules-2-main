use std::fs;
use std::time::{Duration, Instant};

use jules::{CompileUnit, Pipeline, PipelineResult};

#[derive(Debug, Clone)]
struct Sample {
    name: &'static str,
    source: String,
    run_main: bool,
}

#[derive(Debug, Clone)]
struct TimingSummary {
    min: Duration,
    p50: Duration,
    p95: Duration,
    max: Duration,
    mean: Duration,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iterations = args
        .get(1)
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(30);

    println!("Jules micro-benchmark");
    println!("  iterations per sample: {iterations}");
    println!("  pid: {}", std::process::id());
    println!(
        "  initial rss: {:.2} MiB",
        rss_bytes().map(bytes_to_mib).unwrap_or(0.0)
    );
    println!();

    let samples = vec![
        Sample {
            name: "compile:tiny-main",
            source: "fn main() {}\n".to_string(),
            run_main: false,
        },
        Sample {
            name: "compile+run:tiny-main",
            source: "fn main() {}\n".to_string(),
            run_main: true,
        },
        Sample {
            name: "compile:stress-lets",
            source: build_stress_source(400),
            run_main: false,
        },
        Sample {
            name: "compile:ml-kernel",
            source: build_ml_source(64),
            run_main: false,
        },
        Sample {
            name: "compile+run:prime-sieve",
            source: build_prime_sieve_source(),
            run_main: true,
        },
        Sample {
            name: "compile+run:prime-check",
            source: build_prime_check_source(),
            run_main: true,
        },
    ];

    for sample in &samples {
        match run_sample(sample, iterations) {
            Ok(report) => print_report(sample, &report),
            Err(e) => {
                eprintln!("[FAIL] {}: {}", sample.name, e);
                std::process::exit(1);
            }
        }
    }
}

#[derive(Debug)]
struct SampleReport {
    compile: TimingSummary,
    runtime: Option<TimingSummary>,
    rss_before: u64,
    rss_after: u64,
}

const COMPILE_SLA: Duration = Duration::from_millis(1500);

fn run_sample(sample: &Sample, iterations: usize) -> Result<SampleReport, String> {
    let mut compile_times = Vec::with_capacity(iterations);
    let mut runtime_times = if sample.run_main {
        Some(Vec::with_capacity(iterations))
    } else {
        None
    };

    let rss_before = rss_bytes().unwrap_or(0);
    let pipeline = Pipeline::new();

    for i in 0..iterations {
        let mut unit = CompileUnit::new(format!("<bench:{}:{i}>", sample.name), &sample.source);

        let compile_start = Instant::now();
        let result = pipeline.run(&mut unit);
        let compile_elapsed = compile_start.elapsed();
        compile_times.push(compile_elapsed);

        if unit.has_errors() {
            return Err(format!(
                "compile errors in sample `{}` at iteration {} ({} diagnostics)",
                sample.name,
                i,
                unit.diags.len()
            ));
        }

        let PipelineResult::Ok(program) = result else {
            return Err(format!(
                "pipeline halted before codegen/runtime in sample `{}` at iteration {}",
                sample.name, i
            ));
        };

        if sample.run_main {
            let mut interp = jules::interp::Interpreter::new();
            interp.load_program(&program);
            let run_start = Instant::now();
            interp.call_fn("main", vec![]).map_err(|e| {
                format!(
                    "runtime error in `{}` iteration {}: {}",
                    sample.name, i, e.message
                )
            })?;
            let run_elapsed = run_start.elapsed();
            runtime_times.as_mut().unwrap().push(run_elapsed);
        }
    }

    let rss_after = rss_bytes().unwrap_or(rss_before);
    Ok(SampleReport {
        compile: summarize(&compile_times),
        runtime: runtime_times.as_ref().map(|v| summarize(v)),
        rss_before,
        rss_after,
    })
}

fn summarize(samples: &[Duration]) -> TimingSummary {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let p50 = percentile(&sorted, 50.0);
    let p95 = percentile(&sorted, 95.0);

    let total_nanos: u128 = sorted.iter().map(|d| d.as_nanos()).sum();
    let mean_nanos = total_nanos / sorted.len() as u128;
    let mean = Duration::from_nanos(mean_nanos.min(u64::MAX as u128) as u64);

    TimingSummary {
        min,
        p50,
        p95,
        max,
        mean,
    }
}

fn percentile(sorted: &[Duration], pct: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::from_nanos(0);
    }
    let rank = (pct / 100.0) * ((sorted.len() - 1) as f64);
    let idx = rank.round() as usize;
    sorted[idx]
}

fn print_report(sample: &Sample, report: &SampleReport) {
    println!("== {} ==", sample.name);
    print_timing("compile", &report.compile);
    if sample.name.contains("prime") {
        print_sla_custom("compile", &report.compile, PRIME_SLA);
    } else {
        print_sla("compile", &report.compile);
    }
    if let Some(runtime) = &report.runtime {
        print_timing("runtime", runtime);
    }

    let rss_delta = report.rss_after.saturating_sub(report.rss_before);
    println!(
        "rss: {:.2} MiB -> {:.2} MiB (delta: +{:.2} MiB)",
        bytes_to_mib(report.rss_before),
        bytes_to_mib(report.rss_after),
        bytes_to_mib(rss_delta)
    );
    println!();
}

fn print_timing(label: &str, t: &TimingSummary) {
    println!(
        "{}: min={} p50={} p95={} max={} mean={}",
        label,
        fmt_duration(t.min),
        fmt_duration(t.p50),
        fmt_duration(t.p95),
        fmt_duration(t.max),
        fmt_duration(t.mean)
    );
}

fn print_sla(label: &str, t: &TimingSummary) {
    if t.p95 <= COMPILE_SLA {
        println!(
            "{label}-sla: pass (p95 {} <= {})",
            fmt_duration(t.p95),
            fmt_duration(COMPILE_SLA)
        );
    } else {
        println!(
            "{label}-sla: FAIL (p95 {} > {})",
            fmt_duration(t.p95),
            fmt_duration(COMPILE_SLA)
        );
    }
}

fn print_sla_custom(label: &str, t: &TimingSummary, sla: Duration) {
    if t.p95 <= sla {
        println!(
            "{label}-sla: pass (p95 {} <= {})",
            fmt_duration(t.p95),
            fmt_duration(sla)
        );
    } else {
        println!(
            "{label}-sla: FAIL (p95 {} > {})",
            fmt_duration(t.p95),
            fmt_duration(sla)
        );
    }
}

fn fmt_duration(d: Duration) -> String {
    if d.as_micros() >= 1_000 {
        format!("{:.3} ms", d.as_secs_f64() * 1_000.0)
    } else {
        format!("{:.1} µs", d.as_secs_f64() * 1_000_000.0)
    }
}

fn bytes_to_mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn rss_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb = rest
                .split_whitespace()
                .next()
                .and_then(|n| n.parse::<u64>().ok())?;
            return Some(kb * 1024);
        }
    }
    None
}

fn build_stress_source(n_lets: usize) -> String {
    let mut out = String::from("fn main() {\n");
    for i in 0..n_lets {
        let line = format!("  let v{i} = {} + {};\n", i, i + 1);
        out.push_str(&line);
    }
    out.push_str("}\n");
    out
}

fn build_ml_source(width: usize) -> String {
    let mut out = String::from("fn main() {\n");
    out.push_str("  let scale = 2;\n");
    for i in 0..width {
        out.push_str(&format!("  let x{i} = {} * scale;\n", i + 1));
    }
    for i in 0..width {
        let a = i;
        let b = (i + 3) % width;
        out.push_str(&format!("  let m{i} = x{a} * x{b} + scale;\n"));
    }
    out.push_str("}\n");
    out
}

const PRIME_SLA: Duration = Duration::from_millis(2000);

fn build_prime_sieve_source() -> String {
    // Sieve of Eratosthenes to find primes up to 1000
    let mut out = String::from("fn main() {\n");
    out.push_str("  let limit = 1000;\n");
    out.push_str("  let i = 2;\n");
    // Simple iterative prime checking
    out.push_str("  while i < limit {\n");
    out.push_str("    let is_prime = 1;\n");
    out.push_str("    let j = 2;\n");
    out.push_str("    while j < i {\n");
    out.push_str("      let rem = i - (i / j) * j;\n");
    out.push_str("      if rem == 0 {\n");
    out.push_str("        let is_prime = 0;\n");
    out.push_str("      }\n");
    out.push_str("      let j = j + 1;\n");
    out.push_str("    }\n");
    out.push_str("    if is_prime == 1 {\n");
    out.push_str("      i;\n");
    out.push_str("    }\n");
    out.push_str("    let i = i + 1;\n");
    out.push_str("  }\n");
    out.push_str("}\n");
    out
}

fn build_prime_check_source() -> String {
    // Check if specific numbers are prime
    let mut out = String::from("fn main() {\n");
    let test_numbers = [97, 101, 103, 107, 109, 113, 127, 131, 137, 139];
    for num in test_numbers.iter() {
        out.push_str(&format!("  let n = {num};\n"));
        out.push_str("  let is_prime = 1;\n");
        out.push_str("  let i = 2;\n");
        out.push_str("  while i < n {\n");
        out.push_str("    let rem = n - (n / i) * i;\n");
        out.push_str("    if rem == 0 {\n");
        out.push_str("      let is_prime = 0;\n");
        out.push_str("    }\n");
        out.push_str("    let i = i + 1;\n");
        out.push_str("  }\n");
        out.push_str("  is_prime;\n");
    }
    out.push_str("}\n");
    out
}
