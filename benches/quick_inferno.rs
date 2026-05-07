use std::time::Instant;
use jules::{CompileUnit, DiagSeverity, Pipeline, PipelineResult};

fn main() {
    let benches: Vec<(&str, &str, Option<i32>)> = vec![
        ("deep-arith-chain", r#"
fn bench() -> i32 {
    let x: i32 = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10
                + 11 + 12 + 13 + 14 + 15 + 16 + 17 + 18 + 19 + 20
                + 21 + 22 + 23 + 24 + 25 + 26 + 27 + 28 + 29 + 30
                + 31 + 32 + 33 + 34 + 35 + 36 + 37 + 38 + 39 + 40
                + 41 + 42 + 43 + 44 + 45 + 46 + 47 + 48 + 49 + 50;
    x * 2
}
"#, Some(2550)),
        ("fibonacci-30", r#"
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
"#, Some(514229)),
        ("gcd-euclidean", r#"
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
"#, None),
        ("const-fold-heavy", r#"
fn bench() -> i32 {
    let a: i32 = 1 + 2 + 3 + 4 + 5;
    let b: i32 = a * 100 / 15;
    let c: i32 = b - 50 + 25;
    let d: i32 = c * 2 + 10;
    let e: i32 = d / 3 - 5;
    e
}
"#, None),
        ("sum-to-1000", r#"
fn bench() -> i32 {
    let mut s: i32 = 0;
    let mut i: i32 = 1;
    while i <= 1000 {
        s = s + i;
        i = i + 1;
    }
    s
}
"#, Some(500500)),
        ("prime-sieve-500", r#"
fn is_prime(n: i32) -> i32 {
    if n < 2 { return 0; }
    if n < 4 { return 1; }
    if n - (n / 2) * 2 == 0 { return 0; }
    if n - (n / 3) * 3 == 0 { return 0; }
    let mut i: i32 = 5;
    while i * i <= n {
        if n - (n / i) * i == 0 { return 0; }
        if n - (n / (i + 2)) * (i + 2) == 0 { return 0; }
        i = i + 6;
    }
    1
}
fn bench() -> i32 {
    let mut count: i32 = 0;
    let mut i: i32 = 2;
    while i <= 500 {
        count = count + is_prime(i);
        i = i + 1;
    }
    count
}
"#, Some(95)),
        ("collatz-200", r#"
fn bench() -> i32 {
    let mut total: i32 = 0;
    let mut n: i32 = 1;
    while n <= 200 {
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
        total = total + steps;
        n = n + 1;
    }
    total
}
"#, None),
    ];

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              JULES QUICK BENCHMARK SUITE                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let iterations = 10;
    let mut pass = 0;
    let mut fail = 0;
    let mut total_compile_us: u128 = 0;
    let mut total_run_us: f64 = 0.0;

    for (name, src, expected) in &benches {
        print!("  {:25} ", name);
        let _ = std::io::Write::flush(&mut std::io::stdout());

        let t0 = Instant::now();
        let mut unit = CompileUnit::new(name.to_string(), *src);
        let result = Pipeline::new().run(&mut unit);

        if unit.has_errors() {
            let msgs: Vec<String> = unit.diags.iter()
                .filter(|d| d.severity == DiagSeverity::Error)
                .map(|d| d.message.clone())
                .collect();
            println!("COMPILE ERROR: {}", msgs.join("; "));
            fail += 1;
            continue;
        }

        let prog = match result {
            PipelineResult::Ok(p) => p,
            _ => { println!("PIPELINE FAIL"); fail += 1; continue; }
        };

        let compile_us = t0.elapsed().as_micros();
        total_compile_us += compile_us;

        let mut vm = jules::runtime::bytecode_vm::BytecodeVM::new();
        let vm_ok = vm.load_program(&prog).is_ok();

        let (run_time, last_val, engine) = if vm_ok {
            let _ = vm.call_fn("bench", vec![]);
            let run_start = Instant::now();
            let mut last = jules::interp::Value::I32(0);
            for _ in 0..iterations {
                last = match vm.call_fn("bench", vec![]) {
                    Ok(v) => v,
                    Err(e) => {
                        println!("VM ERROR: {}", e.message);
                        break;
                    }
                };
            }
            (run_start.elapsed().as_secs_f64(), last, "vm")
        } else {
            let mut interp = jules::interp::Interpreter::new();
            interp.set_jit_enabled(false);
            interp.load_program(&prog);
            let _ = interp.call_fn("bench", vec![]);
            let run_start = Instant::now();
            let mut last = jules::interp::Value::I32(0);
            for _ in 0..iterations {
                last = match interp.call_fn("bench", vec![]) {
                    Ok(v) => v,
                    Err(e) => {
                        println!("INTERP ERROR: {}", e.message);
                        break;
                    }
                };
            }
            (run_start.elapsed().as_secs_f64(), last, "interp")
        };

        let result_val = match last_val {
            jules::interp::Value::I32(n) => n,
            jules::interp::Value::I64(n) => n as i32,
            jules::interp::Value::F64(f) => f as i32,
            other => { println!("TYPE: {:?}", other); fail += 1; continue; }
        };

        let correct = match expected {
            Some(exp) => result_val == *exp,
            None => true,
        };
        let avg_us = (run_time / iterations as f64) * 1_000_000.0;
        total_run_us += avg_us;

        if correct {
            pass += 1;
            println!("PASS  result={:<12} compile={}us  avg={:.1}us/iter [{}]", result_val, compile_us, avg_us, engine);
        } else {
            fail += 1;
            println!("FAIL  expected={} got={}", expected.unwrap(), result_val);
        }
    }

    println!();
    println!("  ─────────────────────────────────────────────────────────────");
    println!("  Total: {}/{} passed, total compile: {}us, total avg run: {:.1}us", pass, pass+fail, total_compile_us, total_run_us);
    if fail == 0 { println!("  *** ALL BENCHMARKS PASSED ***"); }
    else { println!("  *** {} BENCHMARKS FAILED ***", fail); }
}
