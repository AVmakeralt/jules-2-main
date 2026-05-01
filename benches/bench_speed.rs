// JULES DEBUG — isolate the infinite loop bug
use jules::{CompileUnit, DiagSeverity, Pipeline, PipelineResult};

fn main() {
    println!("=== ISOLATING THE BUG ===");

    // Test: Does assignment work at all in a block?
    test("assign-in-if", "fn bench() -> i32 { let mut x: i32 = 0; if true { x = 1; } x }");

    // Test: Simple while loop with manual break
    test("while-break", "fn bench() -> i32 { let mut i: i32 = 0; while true { i = i + 1; if i > 2 { break; } } i }");

    // Test: What about a for loop?
    // Jules may not have for loops in the same way

    // Test: while loop condition check
    test("while-cond", "fn bench() -> i32 { let mut i: i32 = 0; while i < 1 { let i: i32 = i + 1; } 42 }");
}

fn test(name: &str, src: &str) {
    print!("  {:25} ", name);
    let mut unit = CompileUnit::new("<test>".to_string(), src);
    let result = Pipeline::new().run(&mut unit);

    if unit.has_errors() {
        for d in &unit.diags {
            if d.severity == DiagSeverity::Error {
                println!("COMPILE ERROR: {}", d.message);
            }
        }
        return;
    }

    let prog = match result {
        PipelineResult::Ok(p) => p,
        _ => { println!("PIPELINE FAILED"); return; }
    };

    let mut interp = jules::interp::Interpreter::new();
    interp.set_jit_enabled(false);
    interp.load_program(&prog);

    use std::time::Instant;
    let start = Instant::now();
    match interp.call_fn("bench", vec![]) {
        Ok(v) => {
            let elapsed = start.elapsed();
            println!("OK  result={}  time={:?}", v, elapsed);
        }
        Err(e) => {
            println!("RUNTIME ERROR: {}", e.message);
        }
    }
}
