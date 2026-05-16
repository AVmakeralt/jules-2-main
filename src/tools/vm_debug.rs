// Debug tool: trace VM execution to find bugs
use jules::{CompileUnit, DiagSeverity, Pipeline, PipelineResult};

fn main() {
    println!("=== VM DEBUG: Tracing bytecode execution ===\n");

    // Test 1: cond-mutation (simplest failure)
    test("cond-mutation", r#"
fn bench() -> i32 {
    let mut x: i32 = 10;
    let flag: i32 = 1;
    if flag != 0 {
        x = 99;
    }
    x
}
"#, Some(99));

    // Test 2: fibonacci (loop with variable swaps)
    test("fibonacci-10", r#"
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
"#, Some(55));

    // Test 3: simple function call
    test("call-simple", r#"
fn identity(x: i32) -> i32 { x }
fn bench() -> i32 {
    identity(42)
}
"#, Some(42));

    // Test 4: loop-break-early
    test("loop-break-early", r#"
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
"#, Some(71));

    // Test 5: dead-code-heavy
    test("dead-code-heavy", r#"
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
"#, Some(200));

    // Test 6: recursive fibonacci
    test("fib-recursive-10", r#"
fn fib(n: i32) -> i32 {
    if n <= 1 {
        return n;
    }
    fib(n - 1) + fib(n - 2)
}
fn bench() -> i32 {
    fib(10)
}
"#, Some(55));

    // Test 7: double call (calls same function twice)
    test("double-call", r#"
fn add1(x: i32) -> i32 { x + 1 }
fn bench() -> i32 {
    add1(add1(0))
}
"#, Some(2));

    println!("\n=== DEBUG COMPLETE ===");
}

fn test(name: &str, src: &str, expected: Option<i32>) {
    println!("--- {} ---", name);
    let mut unit = CompileUnit::new(name.to_string(), src);
    let result = Pipeline::new().run(&mut unit);

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

    // Try VM
    let mut vm = jules::runtime::bytecode_vm::BytecodeVM::new();
    let vm_ok = vm.load_program(&prog).is_ok();

    if vm_ok {
        // Dump the compiled bytecode
        println!("  [VM bytecode loaded]");
        
        match vm.call_fn("bench", vec![]) {
            Ok(v) => {
                let val = match v {
                    jules::interp::Value::I32(n) => n as i64,
                    jules::interp::Value::I64(n) => n,
                    jules::interp::Value::F64(f) => f as i64,
                    other => { println!("  TYPE: {:?}", other); return; }
                };
                let correct = expected.map_or(true, |e| val == e as i64);
                if correct {
                    println!("  VM OK  result={}", val);
                } else {
                    println!("  VM FAIL  expected={} got={}", expected.unwrap(), val);
                }
            }
            Err(e) => {
                println!("  VM ERROR: {}", e.message);
            }
        }
    } else {
        println!("  VM LOAD FAILED, trying interpreter...");
    }

    // Also try interpreter for comparison
    let mut interp = jules::interp::Interpreter::new();
    interp.set_jit_enabled(false);
    interp.load_program(&prog);
    match interp.call_fn("bench", vec![]) {
        Ok(v) => {
            let val = match v {
                jules::interp::Value::I32(n) => n as i64,
                jules::interp::Value::I64(n) => n,
                jules::interp::Value::F64(f) => f as i64,
                _ => return,
            };
            println!("  INTERP result={}", val);
        }
        Err(e) => {
            println!("  INTERP ERROR: {}", e.message);
        }
    }
}
