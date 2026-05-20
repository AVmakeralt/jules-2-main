// Test what DSE eliminates in fibonacci
use jules::compiler::ast::Program;
use jules::optimizer::advanced_optimizer::{Superoptimizer, SuperoptimizerConfig};
use jules::{CompileUnit, Pipeline, PipelineResult};

fn main() {
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

    let mut unit = CompileUnit::new("test".to_string(), fib_src);
    let mut pipeline = Pipeline::new();
    pipeline.opt_level = 0;
    let prog = match pipeline.run(&mut unit) {
        PipelineResult::Ok(p) => p,
        PipelineResult::OkWithIr { program, .. } => program,
        _ => { println!("PIPELINE FAILED"); return; }
    };

    // Run just fast_compile (works)
    let mut p1 = prog.clone();
    let mut opt1 = Superoptimizer::new(SuperoptimizerConfig::fast_compile());
    opt1.optimize_program(&mut p1);
    println!("After fast_compile superopt:");
    dump_fn(&p1);
    test_interp(&p1, Some(55));

    // Run with DSE enabled (breaks)
    let mut p2 = prog.clone();
    let mut config2 = SuperoptimizerConfig::fast_compile();
    config2.enable_dse = true;
    config2.iterations = 1;
    let mut opt2 = Superoptimizer::new(config2);
    opt2.optimize_program(&mut p2);
    println!("\nAfter fast_compile + DSE:");
    dump_fn(&p2);
    test_interp(&p2, Some(55));
}

fn dump_fn(prog: &Program) {
    for item in &prog.items {
        if let jules::compiler::ast::Item::Fn(fn_decl) = item {
            println!("Function: {}", fn_decl.name);
            if let Some(body) = &fn_decl.body {
                for (i, stmt) in body.stmts.iter().enumerate() {
                    println!("  [{}] {:?}", i, stmt);
                }
                if let Some(tail) = &body.tail {
                    println!("  tail: {:?}", tail);
                }
            }
        }
    }
}

fn test_interp(prog: &Program, expected: Option<i32>) {
    let mut interp = jules::interp::Interpreter::new();
    interp.set_jit_enabled(false);
    interp.load_program(prog);
    match interp.call_fn("bench", vec![]) {
        Ok(v) => {
            let val = match v {
                jules::interp::Value::I32(n) => n as i64,
                jules::interp::Value::I64(n) => n,
                other => { println!("TYPE: {:?}", other); return; }
            };
            let correct = expected.map_or(true, |e| val == e as i64);
            if correct { println!("  OK  result={}", val); }
            else { println!("  FAIL  expected={} got={}", expected.unwrap(), val); }
        }
        Err(e) => { println!("  ERROR: {}", e.message); }
    }
}
