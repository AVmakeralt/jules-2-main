// Debug fibonacci recursive call bug — trace the exact pass that removes fib
use jules::CompileUnit;
use jules::compiler::ast::{Item, Expr, Stmt, IfOrBlock};
use jules::optimizer::advanced_optimizer::{Superoptimizer, SuperoptimizerConfig};

fn main() {
    let src = r#"
fn fib(n: i32) -> i32 {
    if n <= 1 {
        return n;
    }
    fib(n - 1) + fib(n - 2)
}
fn bench() -> i32 {
    fib(10)
}
"#;

    // Parse the source WITHOUT optimization
    let mut unit = CompileUnit::new("test".to_string(), src);
    let mut pipeline = jules::Pipeline::new();
    pipeline.opt_level = 0; // No optimization
    let result = pipeline.run(&mut unit);
    let prog = match result {
        jules::PipelineResult::Ok(p) => p,
        jules::PipelineResult::OkWithIr { program, .. } => program,
        _ => { eprintln!("PIPELINE FAILED"); return; }
    };

    let mut program = prog;
    
    eprintln!("=== BEFORE Superoptimizer ===");
    print_fns(&program);

    let config = SuperoptimizerConfig::fast_compile(); // opt_level=1
    let mut opt = Superoptimizer::new(config);
    opt.optimize_program(&mut program);
    
    eprintln!("\n=== AFTER Superoptimizer (dead_fn={}, inline={}, const_fold={}) ===", 
        opt.dead_functions_eliminated, opt.inlinings, opt.constant_folds);
    print_fns(&program);
}

fn print_fns(prog: &jules::compiler::ast::Program) {
    for item in &prog.items {
        if let Item::Fn(f) = item {
            eprintln!("  Function: {} (params: {})", f.name, f.params.len());
            if let Some(body) = &f.body {
                let has_fib_call = check_fib_call_in_block(body);
                eprintln!("    Contains fib call: {}", has_fib_call);
            }
        }
    }
}

fn check_fib_call_in_block(block: &jules::compiler::ast::Block) -> bool {
    for stmt in &block.stmts {
        if check_fib_call_in_stmt(stmt) { return true; }
    }
    if let Some(tail) = &block.tail {
        if check_fib_call_in_expr(tail) { return true; }
    }
    false
}

fn check_fib_call_in_stmt(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Let { init: Some(e), .. } | Stmt::Expr { expr: e, .. } => check_fib_call_in_expr(e),
        Stmt::Return { value: Some(e), .. } => check_fib_call_in_expr(e),
        Stmt::If { cond, then, else_, .. } => {
            let mut found = check_fib_call_in_expr(cond) || check_fib_call_in_block(then);
            if let Some(eb) = else_ {
                match eb.as_ref() {
                    IfOrBlock::If(if_stmt) => found = found || check_fib_call_in_stmt(if_stmt),
                    IfOrBlock::Block(b) => found = found || check_fib_call_in_block(b),
                }
            }
            found
        }
        Stmt::While { cond, body, .. } => check_fib_call_in_expr(cond) || check_fib_call_in_block(body),
        _ => false,
    }
}

fn check_fib_call_in_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Call { func, args, .. } => {
            if let Expr::Ident { name, .. } = func.as_ref() {
                if name == "fib" { return true; }
            }
            args.iter().any(|a| check_fib_call_in_expr(a))
        }
        Expr::BinOp { lhs, rhs, .. } => check_fib_call_in_expr(lhs) || check_fib_call_in_expr(rhs),
        Expr::UnOp { expr: e, .. } => check_fib_call_in_expr(e),
        Expr::IfExpr { cond, then, else_, .. } => {
            let mut found = check_fib_call_in_expr(cond) || check_fib_call_in_block(then);
            if let Some(eb) = else_ {
                found = found || check_fib_call_in_block(eb);
            }
            found
        }
        Expr::Block(b) => check_fib_call_in_block(b),
        _ => false,
    }
}
