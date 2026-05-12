// Debug tool: compare optimized vs unoptimized results
use jules::{CompileUnit, DiagSeverity, Pipeline, PipelineResult};

fn main() {
    println!("=== PIPELINE OPTIMIZATION BUG HUNT ===\n");

    // Test: cond-mutation (fails with default opt_level=2)
    let src = r#"
fn bench() -> i32 {
    let mut x: i32 = 10;
    let flag: i32 = 1;
    if flag != 0 {
        x = 99;
    }
    x
}
"#;

    // Test with opt_level 0 (no optimization)
    println!("--- opt_level=0 (no optimization) ---");
    test_with_opt_level(src, 0, Some(99));

    // Test with opt_level 1 (fast compile) 
    println!("--- opt_level=1 (fast compile) ---");
    test_with_opt_level(src, 1, Some(99));

    // Test with opt_level 2 (balanced)
    println!("--- opt_level=2 (balanced) ---");
    test_with_opt_level(src, 2, Some(99));

    // Test with opt_level 3 (maximum)
    println!("--- opt_level=3 (maximum) ---");
    test_with_opt_level(src, 3, Some(99));

    // Test fibonacci with different opt levels
    let fib_src = r#"
fn bench() -> i32 {
    let mut a: i32 = 0;
    let mut b: i32 = 1;
    let mut i: i32 = 0;
    while i < 10 {
        let tmp: i32 = a + b;
        a = b;
        b = tmp;
        i = i + 1;
    }
    a
}
"#;
    println!("\n--- Fibonacci opt_level=0 ---");
    test_with_opt_level(fib_src, 0, Some(55));
    println!("--- Fibonacci opt_level=1 ---");
    test_with_opt_level(fib_src, 1, Some(55));
    println!("--- Fibonacci opt_level=2 ---");
    test_with_opt_level(fib_src, 2, Some(55));

    println!("\n=== DONE ===");
}

fn test_with_opt_level(src: &str, opt_level: u8, expected: Option<i32>) {
    let mut unit = CompileUnit::new("test".to_string(), src);
    let mut pipeline = Pipeline::new();
    pipeline.opt_level = opt_level;
    let result = pipeline.run(&mut unit);

    if unit.has_errors() {
        for d in &unit.diags {
            if d.severity == DiagSeverity::Error {
                println!("  COMPILE ERROR: {}", d.message);
            }
        }
        return;
    }

    let prog = match result {
        PipelineResult::Ok(p) => p,
        _ => { println!("  PIPELINE FAILED"); return; }
    };

    // Use interpreter
    let mut interp = jules::interp::Interpreter::new();
    interp.set_jit_enabled(false);
    interp.load_program(&prog);
    match interp.call_fn("bench", vec![]) {
        Ok(v) => {
            let val = match v {
                jules::interp::Value::I32(n) => n as i64,
                jules::interp::Value::I64(n) => n,
                jules::interp::Value::F64(f) => f as i64,
                other => { println!("  TYPE: {:?}", other); return; }
            };
            let correct = expected.map_or(true, |e| val == e as i64);
            if correct {
                println!("  OK  result={}", val);
            } else {
                println!("  FAIL  expected={} got={}", expected.unwrap(), val);
            }
        }
        Err(e) => {
            println!("  ERROR: {}", e.message);
        }
    }
}
