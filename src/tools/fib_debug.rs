// Debug fibonacci at opt_level=2
use jules::{CompileUnit, DiagSeverity, Pipeline, PipelineResult};

fn main() {
    println!("=== FIBONACCI BUG HUNT ===\n");

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

    for opt in 0..=3 {
        println!("--- opt_level={} ---", opt);
        test_with_opt_level(fib_src, opt, Some(55));
    }
    
    // Test simpler while loop
    let simple_src = r#"
fn bench() -> i32 {
    let mut s: i32 = 0;
    let mut i: i32 = 0;
    while i < 5 {
        s = s + i;
        i = i + 1;
    }
    s
}
"#;
    println!("\n--- Simple while loop ---");
    for opt in 0..=3 {
        println!("opt_level={}", opt);
        test_with_opt_level(simple_src, opt, Some(10));
    }
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
        PipelineResult::OkWithIr { program, .. } => program,
        _ => { println!("  PIPELINE FAILED"); return; }
    };

    let mut interp = jules::interp::Interpreter::new();
    interp.set_jit_enabled(false);
    interp.load_program(&prog);
    match interp.call_fn("bench", vec![]) {
        Ok(v) => {
            let val = match v {
                jules::interp::Value::I32(n) => n as i64,
                jules::interp::Value::I64(n) => n,
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
