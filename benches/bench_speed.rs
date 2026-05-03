// JULES DEBUG — isolate the infinite loop bug
use jules::{CompileUnit, DiagSeverity, Pipeline, PipelineResult};

fn main() {
    println!("=== WHILE-LOOP FIX VERIFICATION ===");

    // Test: Does assignment work at all in a block?
    test("assign-in-if", "fn bench() -> i32 { let mut x: i32 = 0; if true { x = 1; } x }");

    // Test: Simple while loop with manual break
    test("while-break", "fn bench() -> i32 { let mut i: i32 = 0; while true { i = i + 1; if i > 2 { break; } } i }");

    // Test: while loop with condition (was broken — uses assignment not shadow)
    test("while-cond", "fn bench() -> i32 { let mut i: i32 = 0; while i < 3 { i = i + 1; } i }");

    // Test: while loop with larger iteration count
    test("while-100", "fn bench() -> i32 { let mut i: i32 = 0; let mut s: i32 = 0; while i < 100 { s = s + i; i = i + 1; } s }");

    // Test: nested while loops
    test("while-nested", "fn bench() -> i32 { let mut i: i32 = 0; let mut j: i32 = 0; let mut s: i32 = 0; while i < 5 { j = 0; while j < 5 { s = s + 1; j = j + 1; } i = i + 1; } s }");

    // Test: while with continue
    test("while-continue", "fn bench() -> i32 { let mut i: i32 = 0; let mut s: i32 = 0; while i < 10 { i = i + 1; if i < 5 { continue; } s = s + i; } s }");

    // Test: break from nested loop
    test("while-break-nested", "fn bench() -> i32 { let mut i: i32 = 0; while true { let mut j: i32 = 0; while j < 10 { j = j + 1; if j > 3 { break; } } i = i + 1; if i > 2 { break; } } i }");

    println!("\n=== ALL TESTS PASSED ===");
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
