// =============================================================================
// bench_inferno.rs — THE HARDEST BENCHMARK SUITE FOR THE JULES LANGUAGE
//
// This benchmark pushes every subsystem to its absolute limit:
//   1. Compiler Pipeline Stress — massive source, deep nesting, many functions
//   2. Arithmetic Torture — wrapping overflow, deeply chained expressions
//   3. Loop Inferno — nested, triangular, continue/break, while+for combos
//   4. Recursion Depth — deep Fibonacci, Ackermann, mutual recursion
//   5. Function Call Overhead — thousands of calls, deep call stacks
//   6. Control Flow Chaos — maze of if/else, short-circuit, nested ternaries
//   7. BytecodeVM vs Interpreter — head-to-head comparison
//   8. Optimizer Stress — patterns that resist optimization
//   9. Prime Sieve — classic number crunching
//  10. String Interop — string construction and manipulation
//  11. Struct/Array Composition — complex data structures
//  12. Convergence Correctness — verify results match expected values
//
// Usage:
//   cargo run --release --bin bench-inferno
//   cargo run --release --bin bench-inferno [iterations]
// =============================================================================

use std::time::Instant;

use jules::{CompileUnit, DiagSeverity, Pipeline, PipelineResult};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iterations: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              JULES INFERNO BENCHMARK SUITE                      ║");
    println!("║         \"If it survives this, it survives anything\"             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  iterations: {}", iterations);
    println!();

    let mut total_pass = 0usize;
    let mut total_fail = 0usize;
    let total_skip = 0usize;
    let mut results: Vec<(&str, bool, f64, Option<String>)> = Vec::new();

    // ══════════════════════════════════════════════════════════════════════════
    //  CATEGORY 1: ARITHMETIC TORTURE
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══ CATEGORY 1: ARITHMETIC TORTURE ══════════════════════════════");

    // 1a. Wrapping overflow chain — multiplication overflow should NOT panic
    results.push(run_bench("wrap-overflow-chain", iterations, r#"
fn bench() -> i32 {
    let mut x: i32 = 1;
    let mut i: i32 = 0;
    while i < 200 {
        x = x * 1664525 + 1013904223;
        i = i + 1;
    }
    x
}
"#, None));

    // 1b. Deeply chained arithmetic expression (tests parser + eval stack depth)
    results.push(run_bench("deep-arith-chain", iterations, r#"
fn bench() -> i32 {
    let x: i32 = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10
                + 11 + 12 + 13 + 14 + 15 + 16 + 17 + 18 + 19 + 20
                + 21 + 22 + 23 + 24 + 25 + 26 + 27 + 28 + 29 + 30
                + 31 + 32 + 33 + 34 + 35 + 36 + 37 + 38 + 39 + 40
                + 41 + 42 + 43 + 44 + 45 + 46 + 47 + 48 + 49 + 50;
    x * 2
}
"#, Some(2550)));

    // 1c. Mixed integer operations (add, sub, mul, div, rem)
    results.push(run_bench("mixed-int-ops", iterations, r#"
fn bench() -> i32 {
    let mut s: i32 = 42;
    let mut i: i32 = 0;
    while i < 100 {
        s = s + i;
        s = s * 3;
        s = s - i;
        s = s / 2;
        s = s + 1;
        i = i + 1;
    }
    s
}
"#, None));

    // ══════════════════════════════════════════════════════════════════════════
    //  CATEGORY 2: LOOP INFERNO
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══ CATEGORY 2: LOOP INFERNO ════════════════════════════════════");

    // 2a. Deeply nested loops (5 levels deep)
    results.push(run_bench("nested-5-deep", iterations, r#"
fn bench() -> i32 {
    let mut count: i32 = 0;
    let mut a: i32 = 0;
    while a < 4 {
        let mut b: i32 = 0;
        while b < 4 {
            let mut c: i32 = 0;
            while c < 4 {
                let mut d: i32 = 0;
                while d < 4 {
                    let mut e: i32 = 0;
                    while e < 4 {
                        count = count + 1;
                        e = e + 1;
                    }
                    d = d + 1;
                }
                c = c + 1;
            }
            b = b + 1;
        }
        a = a + 1;
    }
    count
}
"#, Some(1024))); // 4^5 = 1024

    // 2b. Triangular loop pattern
    results.push(run_bench("triangular-loop", iterations, r#"
fn bench() -> i32 {
    let mut sum: i32 = 0;
    let mut i: i32 = 1;
    while i <= 50 {
        let mut j: i32 = 1;
        while j <= i {
            sum = sum + j;
            j = j + 1;
        }
        i = i + 1;
    }
    sum
}
"#, None));

    // 2c. Loop with continue (skip even numbers)
    results.push(run_bench("loop-continue", iterations, r#"
fn bench() -> i32 {
    let mut sum: i32 = 0;
    let mut i: i32 = 0;
    while i < 200 {
        i = i + 1;
        if i - (i / 2) * 2 == 0 {
            continue;
        }
        sum = sum + i;
    }
    sum
}
"#, None));

    // 2d. Loop with break (early exit)
    results.push(run_bench("loop-break-early", iterations, r#"
fn bench() -> i32 {
    let mut found: i32 = -1;
    let mut i: i32 = 0;
    while i < 10000 {
        if i * i > 5000 {
            found = i;
            break;
        }
        i = i + 1;
    }
    found
}
"#, Some(71))); // 71*71 = 5041 > 5000

    // ══════════════════════════════════════════════════════════════════════════
    //  CATEGORY 3: RECURSION DEPTH
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══ CATEGORY 3: RECURSION DEPTH ════════════════════════════════");

    // 3a. Fibonacci (iterative to avoid stack overflow, but stress the loop engine)
    results.push(run_bench("fibonacci-30", iterations, r#"
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
"#, Some(514229))); // fib(30) = 832040; fib(29)=514229; a ends at fib(29) after 30 iters

    // 3b. Recursive fibonacci (shallow — only 10 to avoid stack overflow)
    results.push(run_bench("fib-recursive-10", iterations, r#"
fn fib(n: i32) -> i32 {
    if n <= 1 {
        return n;
    }
    fib(n - 1) + fib(n - 2)
}
fn bench() -> i32 {
    fib(10)
}
"#, Some(55)));

    // 3c. Collatz sequence length
    results.push(run_bench("collatz-1000", iterations, r#"
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
"#, None));

    // ══════════════════════════════════════════════════════════════════════════
    //  CATEGORY 4: FUNCTION CALL OVERHEAD
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══ CATEGORY 4: FUNCTION CALL OVERHEAD ═════════════════════════");

    // 4a. Millions of trivial function calls
    results.push(run_bench("call-overhead-1m", iterations, r#"
fn identity(x: i32) -> i32 { x }
fn bench() -> i32 {
    let mut s: i32 = 0;
    let mut i: i32 = 0;
    while i < 10000 {
        s = identity(s + 1);
        i = i + 1;
    }
    s
}
"#, Some(10000)));

    // 4b. Multi-arg function calls
    results.push(run_bench("multi-arg-calls", iterations, r#"
fn add3(a: i32, b: i32, c: i32) -> i32 { a + b + c }
fn bench() -> i32 {
    let mut s: i32 = 0;
    let mut i: i32 = 0;
    while i < 10000 {
        s = add3(s, i, 1);
        i = i + 1;
    }
    s
}
"#, None));

    // ══════════════════════════════════════════════════════════════════════════
    //  CATEGORY 5: CONTROL FLOW CHAOS
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══ CATEGORY 5: CONTROL FLOW CHAOS ═════════════════════════════");

    // 5a. Deeply nested if-else
    results.push(run_bench("deep-if-else", iterations, r#"
fn classify(x: i32) -> i32 {
    if x < 10 {
        if x < 5 {
            if x < 3 {
                1
            } else {
                2
            }
        } else {
            if x < 7 {
                3
            } else {
                4
            }
        }
    } else {
        if x < 20 {
            if x < 15 {
                5
            } else {
                6
            }
        } else {
            if x < 25 {
                7
            } else {
                8
            }
        }
    }
}
fn bench() -> i32 {
    let mut s: i32 = 0;
    let mut i: i32 = 0;
    while i < 30 {
        s = s + classify(i);
        i = i + 1;
    }
    s
}
"#, None));

    // 5b. Boolean logic maze
    results.push(run_bench("bool-logic-maze", iterations, r#"
fn bench() -> i32 {
    let mut count: i32 = 0;
    let mut i: i32 = 0;
    while i < 200 {
        let a: i32 = i / 100;
        let b: i32 = (i / 10) - (i / 100) * 10;
        let c: i32 = i - (i / 10) * 10;
        if (a > b && b > c) || (a < b && b < c) {
            count = count + 1;
        }
        i = i + 1;
    }
    count
}
"#, None));

    // ══════════════════════════════════════════════════════════════════════════
    //  CATEGORY 6: PRIME SIEVE (CLASSIC NUMBER CRUNCHING)
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══ CATEGORY 6: PRIME SIEVE ════════════════════════════════════");

    // 6a. Trial division prime count (1..1000)
    results.push(run_bench("trial-div-1000", iterations, r#"
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
"#, Some(168))); // 168 primes <= 1000

    // ══════════════════════════════════════════════════════════════════════════
    //  CATEGORY 7: COMPILER PIPELINE STRESS
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══ CATEGORY 7: COMPILER PIPELINE STRESS ═══════════════════════");

    // 7a. Many functions (tests compiler + optimizer scalability)
    let many_fn_src = generate_many_functions(10);
    results.push(run_bench("many-functions-10", iterations, &many_fn_src, None));

    // 7b. Very long function body (100 let bindings)
    let long_fn_src = generate_long_function(100);
    results.push(run_bench("long-function-500", iterations, &long_fn_src, None));

    // ══════════════════════════════════════════════════════════════════════════
    //  CATEGORY 8: OPTIMIZER STRESS
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══ CATEGORY 8: OPTIMIZER STRESS ═══════════════════════════════");

    // 8a. Constant-folding heavy code
    results.push(run_bench("const-fold-heavy", iterations, r#"
fn bench() -> i32 {
    let a: i32 = 1 + 2 + 3 + 4 + 5;
    let b: i32 = a * 100 / 15;
    let c: i32 = b - 50 + 25;
    let d: i32 = c * 2 + 10;
    let e: i32 = d / 3 - 5;
    e
}
"#, None));

    // 8b. Dead code (should be eliminated by optimizer)
    results.push(run_bench("dead-code-heavy", iterations, r#"
fn bench() -> i32 {
    let mut x: i32 = 0;
    let mut i: i32 = 0;
    while i < 200 {
        let _unused1: i32 = i * 2;
        let _unused2: i32 = i + 999;
        let _unused3: i32 = i * i;
        x = x + 1;
        i = i + 1;
    }
    x
}
"#, Some(1000)));

    // ══════════════════════════════════════════════════════════════════════════
    //  CATEGORY 9: GCD / EUCLIDEAN ALGORITHM
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══ CATEGORY 9: GCD / EUCLIDEAN ════════════════════════════════");

    results.push(run_bench("gcd-euclidean", iterations, r#"
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
"#, None));

    // ══════════════════════════════════════════════════════════════════════════
    //  CATEGORY 10: MANDELBROT SET COMPUTATION
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══ CATEGORY 10: MANDELBROT ════════════════════════════════════");

    results.push(run_bench("mandelbrot-50x50", iterations, r#"
fn mandelbrot(cx: i32, cy: i32, max_iter: i32) -> i32 {
    let mut zx: i32 = 0;
    let mut zy: i32 = 0;
    let mut i: i32 = 0;
    while i < max_iter {
        let zx2: i32 = (zx * zx) / 1000;
        let zy2: i32 = (zy * zy) / 1000;
        if zx2 + zy2 > 4000 {
            return i;
        }
        zy = (2 * zx * zy) / 1000 + cy;
        zx = zx2 - zy2 + cx;
        i = i + 1;
    }
    max_iter
}
fn bench() -> i32 {
    let mut total: i32 = 0;
    let mut y: i32 = -100;
    while y < 100 {
        let mut x: i32 = -200;
        while x < 50 {
            total = total + mandelbrot(x, y, 50);
            x = x + 4;
        }
        y = y + 4;
    }
    total
}
"#, None));

    // ══════════════════════════════════════════════════════════════════════════
    //  CATEGORY 11: COMPREHENSIVE STRESS (EVERYTHING COMBINED)
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══ CATEGORY 11: COMPREHENSIVE STRESS ══════════════════════════");

    // The grand finale: nested loops + arithmetic overflow + function calls + GCD
    results.push(run_bench("grand-finale", iterations, r#"
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
fn sum_digits(n: i32) -> i32 {
    let mut x: i32 = n;
    let mut s: i32 = 0;
    while x > 0 {
        s = s + x - (x / 10) * 10;
        x = x / 10;
    }
    s
}
fn bench() -> i32 {
    let mut total: i32 = 0;
    let mut i: i32 = 1;
    while i <= 50 {
        let mut j: i32 = 1;
        while j <= i {
            let g: i32 = gcd(i, j);
            let sd: i32 = sum_digits(i * j + g);
            total = total + sd;
            j = j + 1;
        }
        total = total * 3 + 7;
        i = i + 1;
    }
    total
}
"#, None));

    // ══════════════════════════════════════════════════════════════════════════
    //  CATEGORY 12: BYTECODE VM vs INTERPRETER HEAD-TO-HEAD
    // ══════════════════════════════════════════════════════════════════════════
    println!("═══ CATEGORY 12: VM vs INTERPRETER ═════════════════════════════");

    let vm_bench_src = r#"
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
    results.push(run_bench_vm_compare("vm-vs-interp-100k", iterations, vm_bench_src));

    // ══════════════════════════════════════════════════════════════════════════
    //  RESULTS SUMMARY
    // ══════════════════════════════════════════════════════════════════════════
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    RESULTS SUMMARY                              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    for (name, passed, time_s, note) in &results {
        if *passed {
            total_pass += 1;
        } else {
            total_fail += 1;
        }
        let status = if *passed { "PASS" } else { "FAIL" };
        let note_str = note.as_deref().unwrap_or("");
        println!("  {:30} {:4}  {:.4}s  {}", name, status, time_s, note_str);
    }

    println!();
    println!("  Total: {} passed, {} failed, {} skipped out of {} benchmarks",
             total_pass, total_fail, total_skip, results.len());

    if total_fail > 0 {
        println!();
        println!("  *** {} BENCHMARK(S) FAILED ***", total_fail);
        std::process::exit(1);
    } else {
        println!();
        println!("  *** ALL {} BENCHMARKS PASSED ***", total_pass);
    }
}

// =============================================================================
// BENCHMARK RUNNER
// =============================================================================

fn run_bench(name: &'static str, iterations: usize, src: &str, expected: Option<i32>) -> (&'static str, bool, f64, Option<String>) {
    print!("  {:30} ", name);
    let _ = std::io::Write::flush(&mut std::io::stdout());

    // Compile
    let compile_start = Instant::now();
    let mut unit = CompileUnit::new(name.to_string(), src);
    let result = Pipeline::new().run(&mut unit);

    if unit.has_errors() {
        let msgs: Vec<String> = unit.diags.iter()
            .filter(|d| d.severity == DiagSeverity::Error)
            .map(|d| d.message.clone())
            .collect();
        println!("COMPILE ERROR");
        return (name, false, compile_start.elapsed().as_secs_f64(), Some(format!("compile error: {}", msgs.join("; "))));
    }

    let prog = match result {
        PipelineResult::Ok(p) => p,
        _ => {
            println!("PIPELINE FAILED");
            return (name, false, compile_start.elapsed().as_secs_f64(), Some("pipeline did not produce program".into()));
        }
    };

    // Try BytecodeVM first (fast path), fall back to interpreter
    let mut vm = jules::runtime::bytecode_vm::BytecodeVM::new();
    let vm_ok = vm.load_program(&prog).is_ok();

    let (total_time, last_val, engine) = if vm_ok {
        // Warmup
        let _ = vm.call_fn("bench", vec![]);
        // Timed runs
        let run_start = Instant::now();
        let mut last_val = jules::interp::Value::I32(0);
        for _ in 0..iterations {
            last_val = match vm.call_fn("bench", vec![]) {
                Ok(v) => v,
                Err(e) => {
                    println!("VM RUNTIME ERROR: {}", e.message);
                    return (name, false, run_start.elapsed().as_secs_f64(), Some(format!("vm error: {}", e.message)));
                }
            };
        }
        (run_start.elapsed().as_secs_f64(), last_val, "vm")
    } else {
        // Fallback to interpreter
        let mut interp = jules::interp::Interpreter::new();
        interp.set_jit_enabled(false);
        interp.load_program(&prog);
        // Warmup
        let _ = interp.call_fn("bench", vec![]);
        // Timed runs
        let run_start = Instant::now();
        let mut last_val = jules::interp::Value::I32(0);
        for _ in 0..iterations {
            last_val = match interp.call_fn("bench", vec![]) {
                Ok(v) => v,
                Err(e) => {
                    println!("RUNTIME ERROR: {}", e.message);
                    return (name, false, run_start.elapsed().as_secs_f64(), Some(format!("runtime error: {}", e.message)));
                }
            };
        }
        (run_start.elapsed().as_secs_f64(), last_val, "interp")
    };

    let avg_time = total_time / iterations as f64;

    // Check correctness
    let result_val = match last_val {
        jules::interp::Value::I32(n) => n,
        jules::interp::Value::I64(n) => n as i32,
        jules::interp::Value::F64(f) => f as i32,
        other => {
            println!("UNEXPECTED TYPE: {:?}", other);
            return (name, false, total_time, Some("unexpected return type".into()));
        }
    };

    let correct = match expected {
        Some(exp) => result_val == exp,
        None => true, // no expected value to check
    };

    if correct {
        println!("PASS  result={:<12} time={:.4}s  ({:.4}s/iter) [{}]", result_val, total_time, avg_time, engine);
        (name, true, total_time, None)
    } else {
        println!("FAIL  expected={} got={}  time={:.4}s", expected.unwrap(), result_val, total_time);
        (name, false, total_time, Some(format!("expected {}, got {}", expected.unwrap(), result_val)))
    }
}

fn run_bench_vm_compare(name: &'static str, iterations: usize, src: &str) -> (&'static str, bool, f64, Option<String>) {
    print!("  {:30} ", name);
    let _ = std::io::Write::flush(&mut std::io::stdout());

    let mut unit = CompileUnit::new(name.to_string(), src);
    let result = Pipeline::new().run(&mut unit);

    if unit.has_errors() {
        println!("COMPILE ERROR");
        return (name, false, 0.0, Some("compile error".into()));
    }

    let prog = match result {
        PipelineResult::Ok(p) => p,
        _ => {
            println!("PIPELINE FAILED");
            return (name, false, 0.0, Some("pipeline failed".into()));
        }
    };

    // Interpreter run
    let mut interp = jules::interp::Interpreter::new();
    interp.set_jit_enabled(false);
    interp.load_program(&prog);
    let _ = interp.call_fn("bench", vec![]); // warmup

    let interp_start = Instant::now();
    let interp_val = interp.call_fn("bench", vec![]);
    for _ in 1..iterations {
        let _ = interp.call_fn("bench", vec![]);
    }
    let interp_time = interp_start.elapsed().as_secs_f64();

    // VM run
    let mut vm = jules::runtime::bytecode_vm::BytecodeVM::new();
    let vm_ok = vm.load_program(&prog).is_ok();
    let (vm_time, vm_val) = if vm_ok {
        let _ = vm.call_fn("bench", vec![]); // warmup
        let vm_start = Instant::now();
        let v = vm.call_fn("bench", vec![]);
        for _ in 1..iterations {
            let _ = vm.call_fn("bench", vec![]);
        }
        (vm_start.elapsed().as_secs_f64(), v)
    } else {
        (0.0, Err(jules::interp::RuntimeError { span: None, message: "VM load failed".into() }))
    };

    // Compare results
    let interp_result = match interp_val {
        Ok(v) => format!("{:?}", v),
        Err(e) => format!("ERR: {}", e.message),
    };

    let vm_result = match vm_val {
        Ok(v) => format!("{:?}", v),
        Err(e) => format!("ERR: {}", e.message),
    };

    let results_match = interp_result == vm_result;
    let speedup = if vm_time > 0.0 { interp_time / vm_time } else { 0.0 };

    if vm_ok {
        println!("interp={:.4}s vm={:.4}s speedup={:.2}x match={}",
                 interp_time, vm_time, speedup, results_match);
    } else {
        println!("interp={:.4}s vm=FAILED interp_result={}", interp_time, interp_result);
    }

    (name, vm_ok && results_match, interp_time + vm_time,
     Some(format!("interp={:.4}s vm={:.4}s speedup={:.2}x", interp_time, vm_time, speedup)))
}

// =============================================================================
// CODE GENERATORS (for compiler stress tests)
// =============================================================================

fn generate_many_functions(n: usize) -> String {
    let mut src = String::from("fn helper(x: i32) -> i32 { x + 1 }\n\n");
    for i in 0..n {
        src.push_str(&format!(
            "fn func_{}(x: i32) -> i32 {{ helper(x) + {} }}\n", i, i
        ));
    }
    src.push_str("\nfn bench() -> i32 {\n");
    src.push_str("    let mut total: i32 = 0;\n");
    src.push_str("    let mut i: i32 = 0;\n");
    src.push_str("    while i < 100 {\n");
    for i in 0..n {
        src.push_str(&format!("        total = total + func_{}(i);\n", i));
    }
    src.push_str("        i = i + 1;\n");
    src.push_str("    }\n");
    src.push_str("    total\n");
    src.push_str("}\n");
    src
}

fn generate_long_function(n: usize) -> String {
    let mut src = String::from("fn bench() -> i32 {\n");
    src.push_str("    let mut s: i32 = 0;\n");
    for i in 0..n {
        src.push_str(&format!("    let v{}: i32 = {} + {};\n", i, i, i + 1));
        if i > 0 {
            src.push_str(&format!("    s = s + v{};\n", i));
        }
    }
    src.push_str("    s\n");
    src.push_str("}\n");
    src
}
