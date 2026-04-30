// =============================================================================
// MCTS-Port Superoptimizer Benchmark
//
// Run: cargo run --release --bin bench-mcts-superopt
// =============================================================================

use std::hint::black_box;
use std::time::Instant;

use jules::compiler::ast::*;
use jules::compiler::lexer::Span;
use jules::optimizer::hardware_cost_model::Microarchitecture;
use jules::optimizer::mcts_superoptimizer::*;

fn sp() -> Span {
    Span::dummy()
}

fn make_test_expressions() -> Vec<(&'static str, Expr)> {
    let mut exprs = Vec::new();

    // 1. x + 0 (identity)
    exprs.push((
        "identity_add",
        Expr::BinOp {
            span: sp(), op: BinOpKind::Add,
            lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
            rhs: Box::new(Expr::IntLit { span: sp(), value: 0 }),
        },
    ));

    // 2. x * 1 (identity)
    exprs.push((
        "identity_mul",
        Expr::BinOp {
            span: sp(), op: BinOpKind::Mul,
            lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
            rhs: Box::new(Expr::IntLit { span: sp(), value: 1 }),
        },
    ));

    // 3. x * 0 (annihilation)
    exprs.push((
        "annihilate_mul",
        Expr::BinOp {
            span: sp(), op: BinOpKind::Mul,
            lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
            rhs: Box::new(Expr::IntLit { span: sp(), value: 0 }),
        },
    ));

    // 4. x - x (self-identity)
    exprs.push((
        "self_sub",
        Expr::BinOp {
            span: sp(), op: BinOpKind::Sub,
            lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
            rhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        },
    ));

    // 5. Constant fold: 3 + 5
    exprs.push((
        "const_fold_add",
        Expr::BinOp {
            span: sp(), op: BinOpKind::Add,
            lhs: Box::new(Expr::IntLit { span: sp(), value: 3 }),
            rhs: Box::new(Expr::IntLit { span: sp(), value: 5 }),
        },
    ));

    // 6. Strength reduce: x * 8
    exprs.push((
        "strength_reduce_shift",
        Expr::BinOp {
            span: sp(), op: BinOpKind::Mul,
            lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
            rhs: Box::new(Expr::IntLit { span: sp(), value: 8 }),
        },
    ));

    // 7. Nested: (x + 0) * 1
    exprs.push((
        "nested_identity",
        Expr::BinOp {
            span: sp(), op: BinOpKind::Mul,
            lhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Add,
                lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
                rhs: Box::new(Expr::IntLit { span: sp(), value: 0 }),
            }),
            rhs: Box::new(Expr::IntLit { span: sp(), value: 1 }),
        },
    ));

    // 8. Deep: ((x * 1) + 0) * (y + 0)
    exprs.push((
        "deep_identity",
        Expr::BinOp {
            span: sp(), op: BinOpKind::Mul,
            lhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Add,
                lhs: Box::new(Expr::BinOp {
                    span: sp(), op: BinOpKind::Mul,
                    lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
                    rhs: Box::new(Expr::IntLit { span: sp(), value: 1 }),
                }),
                rhs: Box::new(Expr::IntLit { span: sp(), value: 0 }),
            }),
            rhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Add,
                lhs: Box::new(Expr::Ident { span: sp(), name: "y".into() }),
                rhs: Box::new(Expr::IntLit { span: sp(), value: 0 }),
            }),
        },
    ));

    // 9. Distribute: x * (y + z)
    exprs.push((
        "distribute",
        Expr::BinOp {
            span: sp(), op: BinOpKind::Mul,
            lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
            rhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Add,
                lhs: Box::new(Expr::Ident { span: sp(), name: "y".into() }),
                rhs: Box::new(Expr::Ident { span: sp(), name: "z".into() }),
            }),
        },
    ));

    // 10. Factor: x*y + x*z
    exprs.push((
        "factor",
        Expr::BinOp {
            span: sp(), op: BinOpKind::Add,
            lhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Mul,
                lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
                rhs: Box::new(Expr::Ident { span: sp(), name: "y".into() }),
            }),
            rhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Mul,
                lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
                rhs: Box::new(Expr::Ident { span: sp(), name: "z".into() }),
            }),
        },
    ));

    // 11. Division by power of 2: x / 16
    exprs.push((
        "strength_reduce_div",
        Expr::BinOp {
            span: sp(), op: BinOpKind::Div,
            lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
            rhs: Box::new(Expr::IntLit { span: sp(), value: 16 }),
        },
    ));

    // 12. Complex: (a + 0) * (b * 1) + (x - x)
    exprs.push((
        "complex_multi_opt",
        Expr::BinOp {
            span: sp(), op: BinOpKind::Add,
            lhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Mul,
                lhs: Box::new(Expr::BinOp {
                    span: sp(), op: BinOpKind::Add,
                    lhs: Box::new(Expr::Ident { span: sp(), name: "a".into() }),
                    rhs: Box::new(Expr::IntLit { span: sp(), value: 0 }),
                }),
                rhs: Box::new(Expr::BinOp {
                    span: sp(), op: BinOpKind::Mul,
                    lhs: Box::new(Expr::Ident { span: sp(), name: "b".into() }),
                    rhs: Box::new(Expr::IntLit { span: sp(), value: 1 }),
                }),
            }),
            rhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Sub,
                lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
                rhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
            }),
        },
    ));

    // 13. No optimization needed: x + y
    exprs.push((
        "no_opt",
        Expr::BinOp {
            span: sp(), op: BinOpKind::Add,
            lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
            rhs: Box::new(Expr::Ident { span: sp(), name: "y".into() }),
        },
    ));

    // 14. Double negate: -(-x)
    exprs.push((
        "double_negate",
        Expr::UnOp {
            span: sp(), op: UnOpKind::Neg,
            expr: Box::new(Expr::UnOp {
                span: sp(), op: UnOpKind::Neg,
                expr: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
            }),
        },
    ));

    // 15. Massive: (((x + 0) * 1) + (y * 0)) * ((z / 1) + (w - w))
    exprs.push((
        "massive_multi_opt",
        Expr::BinOp {
            span: sp(), op: BinOpKind::Mul,
            lhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Add,
                lhs: Box::new(Expr::BinOp {
                    span: sp(), op: BinOpKind::Mul,
                    lhs: Box::new(Expr::BinOp {
                        span: sp(), op: BinOpKind::Add,
                        lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
                        rhs: Box::new(Expr::IntLit { span: sp(), value: 0 }),
                    }),
                    rhs: Box::new(Expr::IntLit { span: sp(), value: 1 }),
                }),
                rhs: Box::new(Expr::BinOp {
                    span: sp(), op: BinOpKind::Mul,
                    lhs: Box::new(Expr::Ident { span: sp(), name: "y".into() }),
                    rhs: Box::new(Expr::IntLit { span: sp(), value: 0 }),
                }),
            }),
            rhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Add,
                lhs: Box::new(Expr::BinOp {
                    span: sp(), op: BinOpKind::Div,
                    lhs: Box::new(Expr::Ident { span: sp(), name: "z".into() }),
                    rhs: Box::new(Expr::IntLit { span: sp(), value: 1 }),
                }),
                rhs: Box::new(Expr::BinOp {
                    span: sp(), op: BinOpKind::Sub,
                    lhs: Box::new(Expr::Ident { span: sp(), name: "w".into() }),
                    rhs: Box::new(Expr::Ident { span: sp(), name: "w".into() }),
                }),
            }),
        },
    ));

    exprs
}

fn main() {
    println!("================================================================");
    println!("  MCTS-Port Superoptimizer Benchmark");
    println!("================================================================");
    println!();

    let test_exprs = make_test_expressions();

    // ── Phase 1: Fast Config ──
    println!("-- Phase 1: Fast Configuration (50 sims, depth 3, 10ms) --");
    println!();

    let mut total_original_cycles = 0u32;
    let mut total_optimized_cycles = 0u32;
    let mut total_improved = 0usize;
    let total_exprs = test_exprs.len();

    let mut fast_opt = MctsSuperoptimizer::new(MctsConfig::fast());

    for (name, expr) in &test_exprs {
        let mut cost_est = CycleCostEstimator::new(None);
        let instr = Instr::from_expr(expr).unwrap();
        let original_cost = cost_est.estimate(&instr);

        let t0 = Instant::now();
        let result = fast_opt.optimize(expr);
        let elapsed = t0.elapsed();

        let optimized_cost = if let Some(ref optimized) = result {
            let opt_instr = Instr::from_expr(optimized).unwrap();
            cost_est.estimate(&opt_instr)
        } else {
            original_cost
        };

        let improved = optimized_cost < original_cost;
        let reduction = if original_cost > 0 {
            (original_cost - optimized_cost) as f64 / original_cost as f64 * 100.0
        } else { 0.0 };

        total_original_cycles += original_cost;
        total_optimized_cycles += optimized_cost;
        if improved { total_improved += 1; }

        let status = if improved { "IMPROVED" } else if result.is_some() { "SAME" } else { "NO-OPT" };
        println!(
            "  {:25}  cycles: {:4} -> {:4}  ({:5.1}% reduction)  {:?}  {}",
            name, original_cost, optimized_cost, reduction, elapsed, status,
        );
    }

    let fast_reduction = if total_original_cycles > 0 {
        (total_original_cycles - total_optimized_cycles) as f64 / total_original_cycles as f64 * 100.0
    } else { 0.0 };

    println!();
    println!(
        "  Fast summary: {}/{} improved, cycles: {} -> {} ({:.1}% reduction)",
        total_improved, total_exprs, total_original_cycles, total_optimized_cycles, fast_reduction,
    );
    println!(
        "  Simulations: {}, Rewrites: {}, Best improvement: {} cycles, Time: {:?}",
        fast_opt.simulations_run, fast_opt.rewrites_found, fast_opt.best_improvement, fast_opt.time_spent,
    );

    // ── Phase 2: Default Config ──
    println!();
    println!("-- Phase 2: Default Configuration (200 sims, depth 6, 50ms) --");
    println!();

    let mut total_original_cycles2 = 0u32;
    let mut total_optimized_cycles2 = 0u32;
    let mut total_improved2 = 0usize;

    let mut default_opt = MctsSuperoptimizer::new(MctsConfig::default());

    for (name, expr) in &test_exprs {
        let mut cost_est = CycleCostEstimator::new(None);
        let instr = Instr::from_expr(expr).unwrap();
        let original_cost = cost_est.estimate(&instr);

        let t0 = Instant::now();
        let result = default_opt.optimize(expr);
        let elapsed = t0.elapsed();

        let optimized_cost = if let Some(ref optimized) = result {
            let opt_instr = Instr::from_expr(optimized).unwrap();
            cost_est.estimate(&opt_instr)
        } else {
            original_cost
        };

        let improved = optimized_cost < original_cost;
        let reduction = if original_cost > 0 {
            (original_cost - optimized_cost) as f64 / original_cost as f64 * 100.0
        } else { 0.0 };

        total_original_cycles2 += original_cost;
        total_optimized_cycles2 += optimized_cost;
        if improved { total_improved2 += 1; }

        let status = if improved { "IMPROVED" } else if result.is_some() { "SAME" } else { "NO-OPT" };
        println!(
            "  {:25}  cycles: {:4} -> {:4}  ({:5.1}% reduction)  {:?}  {}",
            name, original_cost, optimized_cost, reduction, elapsed, status,
        );
    }

    let default_reduction = if total_original_cycles2 > 0 {
        (total_original_cycles2 - total_optimized_cycles2) as f64 / total_original_cycles2 as f64 * 100.0
    } else { 0.0 };

    println!();
    println!(
        "  Default summary: {}/{} improved, cycles: {} -> {} ({:.1}% reduction)",
        total_improved2, total_exprs, total_original_cycles2, total_optimized_cycles2, default_reduction,
    );
    println!(
        "  Simulations: {}, Rewrites: {}, Best improvement: {} cycles, Time: {:?}",
        default_opt.simulations_run, default_opt.rewrites_found, default_opt.best_improvement, default_opt.time_spent,
    );

    // ── Phase 3: Cross-Microarchitecture Comparison ──
    println!();
    println!("-- Phase 3: Cross-Microarchitecture Cycle Costs --");
    println!();

    let microarchs = [
        Microarchitecture::Skylake,
        Microarchitecture::SkylakeX,
        Microarchitecture::IceLake,
        Microarchitecture::GoldenCove,
        Microarchitecture::Zen2,
        Microarchitecture::Zen3,
        Microarchitecture::Zen4,
    ];

    let cross_test_exprs: Vec<(&str, Expr)> = vec![
        ("x*8 (shift)", Expr::BinOp {
            span: sp(), op: BinOpKind::Mul,
            lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
            rhs: Box::new(Expr::IntLit { span: sp(), value: 8 }),
        }),
        ("x/16 (shift)", Expr::BinOp {
            span: sp(), op: BinOpKind::Div,
            lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
            rhs: Box::new(Expr::IntLit { span: sp(), value: 16 }),
        }),
        ("x+y+z (adds)", Expr::BinOp {
            span: sp(), op: BinOpKind::Add,
            lhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Add,
                lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
                rhs: Box::new(Expr::Ident { span: sp(), name: "y".into() }),
            }),
            rhs: Box::new(Expr::Ident { span: sp(), name: "z".into() }),
        }),
    ];

    print!("  {:20}", "Expression");
    for ma in &microarchs {
        print!("  {:>12}", format!("{:?}", ma));
    }
    println!();

    for (name, expr) in &cross_test_exprs {
        print!("  {:20}", name);
        for ma in &microarchs {
            let mut cost_est = CycleCostEstimator::new(Some(*ma));
            let instr = Instr::from_expr(expr).unwrap();
            let cost = cost_est.estimate(&instr);
            print!("  {:>12}", format!("{}cyc", cost));
        }
        println!();
    }

    // ── Phase 4: Hardware Cost vs Node Count ──
    println!();
    println!("-- Phase 4: Hardware Cost vs Simple Node Count --");
    println!();

    for (name, expr) in &test_exprs {
        let instr = Instr::from_expr(expr).unwrap();
        let node_cost = CycleCostEstimator::node_cost(&instr);
        let mut cost_est = CycleCostEstimator::new(None);
        let hw_cost = cost_est.estimate(&instr);
        println!(
            "  {:25}  node_cost={:6.1}  hw_cost={:4} cycles",
            name, node_cost, hw_cost,
        );
    }

    // ── Phase 5: Throughput Stress Test ──
    println!();
    println!("-- Phase 5: Throughput Stress Test --");
    println!();

    let stress_count = 100;
    let stress_expr = Expr::BinOp {
        span: sp(), op: BinOpKind::Add,
        lhs: Box::new(Expr::BinOp {
            span: sp(), op: BinOpKind::Mul,
            lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
            rhs: Box::new(Expr::IntLit { span: sp(), value: 1 }),
        }),
        rhs: Box::new(Expr::IntLit { span: sp(), value: 0 }),
    };

    let mut stress_opt = MctsSuperoptimizer::new(MctsConfig::fast());
    let t0 = Instant::now();
    let mut stress_improved = 0usize;
    for _ in 0..stress_count {
        if stress_opt.optimize(&stress_expr).is_some() {
            stress_improved += 1;
        }
    }
    let stress_elapsed = t0.elapsed();
    let stress_per_expr = stress_elapsed / stress_count as u32;
    println!(
        "  {} expressions optimized in {:?} ({:?}/expr)",
        stress_count, stress_elapsed, stress_per_expr,
    );
    println!(
        "  Improved: {}/{} ({:.0}%), Throughput: {:.0} exprs/sec",
        stress_improved, stress_count,
        stress_improved as f64 / stress_count as f64 * 100.0,
        stress_count as f64 / stress_elapsed.as_secs_f64().max(1e-12),
    );

    // ── Phase 6: Rewrite Rule Coverage ──
    println!();
    println!("-- Phase 6: Rewrite Rule Coverage --");
    println!();

    let actions = RewriteAction::all();
    println!("  Available rewrite actions: {}", actions.len());
    for action in actions {
        let mut count = 0;
        for (_, expr) in &test_exprs {
            let instr = Instr::from_expr(expr).unwrap();
            if action.is_applicable(&instr) {
                count += 1;
            }
        }
        println!("  {:20} applicable to {}/{} test expressions", format!("{:?}", action), count, test_exprs.len());
    }

    // ── Final Summary ──
    println!();
    println!("================================================================");
    println!("  BENCHMARK COMPLETE");
    println!("================================================================");
    println!();
    println!("  Fast config:     {}/{} improved, {:.1}% cycle reduction", total_improved, total_exprs, fast_reduction);
    println!("  Default config:  {}/{} improved, {:.1}% cycle reduction", total_improved2, total_exprs, default_reduction);
    println!("  Stress test:     {} exprs at {:?}/expr, {:.0} exprs/sec",
        stress_count, stress_per_expr,
        stress_count as f64 / stress_elapsed.as_secs_f64().max(1e-12));
    println!();
    println!("  MCTS-Port Superoptimizer status: OPERATIONAL");
}
