// Debug tool: test individual optimization passes
use jules::{CompileUnit, DiagSeverity, Pipeline, PipelineResult};

fn main() {
    println!("=== ISOLATE WHICH OPT PASS CAUSES THE BUG ===\n");

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

    // Baseline: opt_level=1 (works)
    println!("Baseline (opt_level=1):");
    test_with_opt_level(src, 1, Some(99));

    // Now test opt_level=2 but manually disable individual passes
    // by modifying the pipeline. Since we can't easily disable individual
    // passes through the API, let's test by running a custom pipeline
    // that only does the superoptimizer (like opt_level=1)
    
    // Alternative approach: check what the AST looks like after opt_level=2
    println!("\n--- Dumping optimized AST at opt_level=2 ---");
    let mut unit = CompileUnit::new("test".to_string(), src);
    let mut pipeline = Pipeline::new();
    pipeline.opt_level = 2;
    pipeline.print_opt_stats = true;
    let result = pipeline.run(&mut unit);
    
    if let Some(prog) = result.program() {
        for item in &prog.items {
            if let jules::compiler::ast::Item::Fn(fn_decl) = item {
                println!("Function: {}", fn_decl.name);
                if let Some(body) = &fn_decl.body {
                    dump_block(body, 2);
                }
            }
        }
    }

    println!("\n--- Dumping optimized AST at opt_level=1 ---");
    let mut unit2 = CompileUnit::new("test".to_string(), src);
    let mut pipeline2 = Pipeline::new();
    pipeline2.opt_level = 1;
    pipeline2.print_opt_stats = true;
    let result2 = pipeline2.run(&mut unit2);
    
    if let Some(prog) = result2.program() {
        for item in &prog.items {
            if let jules::compiler::ast::Item::Fn(fn_decl) = item {
                println!("Function: {}", fn_decl.name);
                if let Some(body) = &fn_decl.body {
                    dump_block(body, 2);
                }
            }
        }
    }
}

fn dump_block(block: &jules::compiler::ast::Block, indent: usize) {
    let pad = " ".repeat(indent);
    for stmt in &block.stmts {
        dump_stmt(stmt, indent);
    }
    if let Some(tail) = &block.tail {
        println!("{}tail: {:?}", pad, tail);
    }
}

fn dump_stmt(stmt: &jules::compiler::ast::Stmt, indent: usize) {
    let pad = " ".repeat(indent);
    match stmt {
        jules::compiler::ast::Stmt::Let { pattern, init, mutable, .. } => {
            let mut_str = if *mutable { "mut " } else { "" };
            match init {
                Some(e) => println!("{}let {}{:?} = {:?}", pad, mut_str, pattern, e),
                None => println!("{}let {}{:?}", pad, mut_str, pattern),
            }
        }
        jules::compiler::ast::Stmt::Expr { expr, .. } => {
            println!("{}expr: {:?}", pad, expr);
        }
        jules::compiler::ast::Stmt::If { cond, then, else_, .. } => {
            println!("{}if {:?}", pad, cond);
            dump_block(then, indent + 2);
            if let Some(eb) = else_ {
                println!("{}else", pad);
                match &**eb {
                    jules::compiler::ast::IfOrBlock::Block(b) => dump_block(b, indent + 2),
                    jules::compiler::ast::IfOrBlock::If(s) => dump_stmt(s, indent + 2),
                }
            }
        }
        jules::compiler::ast::Stmt::While { cond, body, .. } => {
            println!("{}while {:?}", pad, cond);
            dump_block(body, indent + 2);
        }
        jules::compiler::ast::Stmt::Return { value, .. } => {
            println!("{}return {:?}", pad, value);
        }
        _ => println!("{}[other stmt]", pad),
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
