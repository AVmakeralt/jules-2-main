//! Minimal test to reproduce JIT hang with while-loop bytecode
use jules::compiler::ast::BinOpKind;
use jules::interp::{CompiledFn, Instr, Value};

fn main() {
    test_simple_loop();
    test_bench_loop();
}

fn test_simple_loop() {
    println!("=== Test 1: Simple while loop ===");
    // fn count() -> i32 {
    //   let mut i: i32 = 0;
    //   while i < 10 { i = i + 1; }
    //   i
    // }
    
    let instrs = vec![
        Instr::LoadI32(1, 0),                    // pc=0: i = 0
        Instr::Load(2, 1),                       // pc=1: tmp2 = i
        Instr::LoadI32(3, 10),                   // pc=2: tmp3 = 10
        Instr::BinOp(4, BinOpKind::Lt, 2, 3),   // pc=3: tmp4 = (i < 10)
        Instr::JumpFalse(4, 5),                  // pc=4: if !tmp4, jump to pc=10
        Instr::Load(5, 1),                       // pc=5: tmp5 = i
        Instr::LoadI32(6, 1),                    // pc=6: tmp6 = 1
        Instr::BinOp(7, BinOpKind::Add, 5, 6),  // pc=7: tmp7 = i + 1
        Instr::Store(1, 7),                      // pc=8: i = tmp7
        Instr::Jump(-9),                         // pc=9: back to pc=1
        Instr::Load(8, 1),                       // pc=10: tmp8 = i
        Instr::Return(8),                        // pc=11: return i
    ];
    
    let compiled = CompiledFn {
        name: "count".to_string(),
        param_count: 0,
        slot_count: 9,
        instrs,
        str_pool: vec![],
        const_pool: vec![],
    };
    
    run_test(&compiled, "count", 10);
}

fn test_bench_loop() {
    println!("\n=== Test 2: Bench while loop (s * 1664525 + i + 97) ===");
    // fn bench() -> i32 {
    //   let mut s: i32 = 1;
    //   let mut i: i32 = 0;
    //   while i < 32 { s = s * 1664525 + i + 97; i = i + 1; }
    //   s
    // }
    
    // Compute expected value using Rust
    let mut s: i32 = 1;
    let mut i: i32 = 0;
    while i < 32 {
        s = s.wrapping_mul(1664525).wrapping_add(i).wrapping_add(97);
        i = i + 1;
    }
    let expected = s;
    println!("Expected result (Rust): {}", expected);
    
    let instrs = vec![
        Instr::LoadI32(1, 1),                       // pc=0: s = 1
        Instr::LoadI32(2, 0),                       // pc=1: i = 0
        // loop_start (pc=2):
        Instr::Load(3, 2),                          // pc=2: tmp3 = i
        Instr::LoadI32(4, 32),                      // pc=3: tmp4 = 32
        Instr::BinOp(5, BinOpKind::Lt, 3, 4),      // pc=4: tmp5 = (i < 32)
        Instr::JumpFalse(5, 13),                    // pc=5: if !tmp5, jump to pc=19; offset = 19 - 5 - 1 = 13
        // body:
        Instr::Load(6, 1),                          // pc=6: tmp6 = s
        Instr::LoadI32(7, 1664525),                 // pc=7: tmp7 = 1664525
        Instr::BinOp(8, BinOpKind::Mul, 6, 7),     // pc=8: tmp8 = s * 1664525
        Instr::Load(9, 2),                          // pc=9: tmp9 = i
        Instr::BinOp(10, BinOpKind::Add, 8, 9),    // pc=10: tmp10 = (s*1664525) + i
        Instr::LoadI32(11, 97),                     // pc=11: tmp11 = 97
        Instr::BinOp(12, BinOpKind::Add, 10, 11),  // pc=12: tmp12 = (s*1664525+i) + 97
        Instr::Store(1, 12),                        // pc=13: s = tmp12
        Instr::Load(13, 2),                         // pc=14: tmp13 = i
        Instr::LoadI32(14, 1),                      // pc=15: tmp14 = 1
        Instr::BinOp(15, BinOpKind::Add, 13, 14),  // pc=16: tmp15 = i + 1
        Instr::Store(2, 15),                        // pc=17: i = tmp15
        Instr::Jump(-17),                          // pc=18: back to pc=2; offset = 2 - 18 - 1 = -17
        // loop_end (pc=19):
        Instr::Load(16, 1),                         // pc=19: tmp16 = s
        Instr::Return(16),                          // pc=20: return s
    ];
    
    let compiled = CompiledFn {
        name: "bench".to_string(),
        param_count: 0,
        slot_count: 17,
        instrs,
        str_pool: vec![],
        const_pool: vec![],
    };
    
    run_test(&compiled, "bench", expected);
}

fn run_test(compiled: &CompiledFn, name: &str, expected: i32) {
    // Test with VM first
    let mut interp = jules::interp::Interpreter::new();
    match jules::interp::vm_exec_i32(&mut interp, compiled, &[]) {
        Some(Ok(val)) => {
            if let Value::I32(v) = val {
                println!("VM i32 result: {} ({})", v, if v == expected { "OK" } else { "MISMATCH" });
            } else {
                println!("VM i32 result: {:?} (unexpected type)", val);
            }
        }
        Some(Err(e)) => println!("VM i32 error: {:?}", e),
        None => println!("VM i32: fell back (unsupported instruction)"),
    }
    
    // Test with JIT
    match jules::jit::phase3_jit::translate(compiled) {
        Some(native) => {
            if !jules::jit::phase3_jit::finalize_arena() {
                eprintln!("WARNING: finalize_arena() failed!");
            }
            
            match jules::jit::phase3_jit::execute(&native, &[]) {
                Ok(val) => {
                    if let Value::I32(v) = val {
                        println!("JIT result: {} ({})", v, if v == expected { "OK" } else { "MISMATCH" });
                    } else {
                        println!("JIT result: {:?} (unexpected type)", val);
                    }
                }
                Err(e) => eprintln!("JIT execution error: {:?}", e),
            }
        }
        None => {
            eprintln!("JIT translation returned None");
        }
    }
}
