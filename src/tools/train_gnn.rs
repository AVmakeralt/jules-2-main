// =============================================================================
// GNN E-Graph Superoptimizer — Training & Benchmark
//
// Run: cargo run --release --bin train-gnn --features gnn-optimizer
// =============================================================================

#![cfg(feature = "gnn-optimizer")]

use std::hint::black_box;
use std::time::Instant;

use jules::compiler::ast::*;
use jules::compiler::lexer::Span;
use jules::optimizer::gnn_egraph_optimizer::*;
use jules::optimizer::gnn_trained_weights::load_pretrained_gnn;
use jules::optimizer::hardware_cost_model::Microarchitecture;
use jules::optimizer::mcts_superoptimizer::*;

fn sp() -> Span {
    Span::dummy()
}

fn make_training_expressions() -> Vec<(String, Expr)> {
    let mut exprs = Vec::new();

    // Basic identities
    exprs.push(("x+0".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Add,
        lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        rhs: Box::new(Expr::IntLit { span: sp(), value: 0, ty: None }),
    }));
    exprs.push(("x*1".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Mul,
        lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        rhs: Box::new(Expr::IntLit { span: sp(), value: 1, ty: None }),
    }));
    exprs.push(("x*0".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Mul,
        lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        rhs: Box::new(Expr::IntLit { span: sp(), value: 0, ty: None }),
    }));
    exprs.push(("x-x".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Sub,
        lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        rhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
    }));

    // Constant folding
    exprs.push(("3+5".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Add,
        lhs: Box::new(Expr::IntLit { span: sp(), value: 3, ty: None }),
        rhs: Box::new(Expr::IntLit { span: sp(), value: 5, ty: None }),
    }));
    exprs.push(("10-3".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Sub,
        lhs: Box::new(Expr::IntLit { span: sp(), value: 10, ty: None }),
        rhs: Box::new(Expr::IntLit { span: sp(), value: 3, ty: None }),
    }));

    // Strength reduction
    exprs.push(("x*8".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Mul,
        lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        rhs: Box::new(Expr::IntLit { span: sp(), value: 8, ty: None }),
    }));
    exprs.push(("x*4".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Mul,
        lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        rhs: Box::new(Expr::IntLit { span: sp(), value: 4, ty: None }),
    }));
    exprs.push(("x/16".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Div,
        lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        rhs: Box::new(Expr::IntLit { span: sp(), value: 16, ty: None }),
    }));
    exprs.push(("x/32".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Div,
        lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        rhs: Box::new(Expr::IntLit { span: sp(), value: 32, ty: None }),
    }));

    // Nested patterns
    exprs.push(("(x+0)*1".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Mul,
        lhs: Box::new(Expr::BinOp {
            span: sp(), op: BinOpKind::Add,
            lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
            rhs: Box::new(Expr::IntLit { span: sp(), value: 0, ty: None }),
        }),
        rhs: Box::new(Expr::IntLit { span: sp(), value: 1, ty: None }),
    }));

    // Deep nested
    exprs.push(("((x*1)+0)*(y+0)".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Mul,
        lhs: Box::new(Expr::BinOp {
            span: sp(), op: BinOpKind::Add,
            lhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Mul,
                lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
                rhs: Box::new(Expr::IntLit { span: sp(), value: 1, ty: None }),
            }),
            rhs: Box::new(Expr::IntLit { span: sp(), value: 0, ty: None }),
        }),
        rhs: Box::new(Expr::BinOp {
            span: sp(), op: BinOpKind::Add,
            lhs: Box::new(Expr::Ident { span: sp(), name: "y".into() }),
            rhs: Box::new(Expr::IntLit { span: sp(), value: 0, ty: None }),
        }),
    }));

    // Factor
    exprs.push(("x*y+x*z".into(), Expr::BinOp {
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
    }));

    // Complex
    exprs.push(("(a+0)*(b*1)+(x-x)".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Add,
        lhs: Box::new(Expr::BinOp {
            span: sp(), op: BinOpKind::Mul,
            lhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Add,
                lhs: Box::new(Expr::Ident { span: sp(), name: "a".into() }),
                rhs: Box::new(Expr::IntLit { span: sp(), value: 0, ty: None }),
            }),
            rhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Mul,
                lhs: Box::new(Expr::Ident { span: sp(), name: "b".into() }),
                rhs: Box::new(Expr::IntLit { span: sp(), value: 1, ty: None }),
            }),
        }),
        rhs: Box::new(Expr::BinOp {
            span: sp(), op: BinOpKind::Sub,
            lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
            rhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        }),
    }));

    // Massive
    exprs.push(("(((x+0)*1)+(y*0))*((z/1)+(w-w))".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Mul,
        lhs: Box::new(Expr::BinOp {
            span: sp(), op: BinOpKind::Add,
            lhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Mul,
                lhs: Box::new(Expr::BinOp {
                    span: sp(), op: BinOpKind::Add,
                    lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
                    rhs: Box::new(Expr::IntLit { span: sp(), value: 0, ty: None }),
                }),
                rhs: Box::new(Expr::IntLit { span: sp(), value: 1, ty: None }),
            }),
            rhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Mul,
                lhs: Box::new(Expr::Ident { span: sp(), name: "y".into() }),
                rhs: Box::new(Expr::IntLit { span: sp(), value: 0, ty: None }),
            }),
        }),
        rhs: Box::new(Expr::BinOp {
            span: sp(), op: BinOpKind::Add,
            lhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Div,
                lhs: Box::new(Expr::Ident { span: sp(), name: "z".into() }),
                rhs: Box::new(Expr::IntLit { span: sp(), value: 1, ty: None }),
            }),
            rhs: Box::new(Expr::BinOp {
                span: sp(), op: BinOpKind::Sub,
                lhs: Box::new(Expr::Ident { span: sp(), name: "w".into() }),
                rhs: Box::new(Expr::Ident { span: sp(), name: "w".into() }),
            }),
        }),
    }));

    // No-opt cases (for negative examples)
    exprs.push(("x+y".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Add,
        lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        rhs: Box::new(Expr::Ident { span: sp(), name: "y".into() }),
    }));
    exprs.push(("x*y".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Mul,
        lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        rhs: Box::new(Expr::Ident { span: sp(), name: "y".into() }),
    }));

    // Double negate
    exprs.push(("-(-x)".into(), Expr::UnOp {
        span: sp(), op: UnOpKind::Neg,
        expr: Box::new(Expr::UnOp {
            span: sp(), op: UnOpKind::Neg,
            expr: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        }),
    }));

    // Distribute
    exprs.push(("x*(y+z)".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Mul,
        lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        rhs: Box::new(Expr::BinOp {
            span: sp(), op: BinOpKind::Add,
            lhs: Box::new(Expr::Ident { span: sp(), name: "y".into() }),
            rhs: Box::new(Expr::Ident { span: sp(), name: "z".into() }),
        }),
    }));

    // x-0, x/1
    exprs.push(("x-0".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Sub,
        lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        rhs: Box::new(Expr::IntLit { span: sp(), value: 0, ty: None }),
    }));
    exprs.push(("x/1".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::Div,
        lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        rhs: Box::new(Expr::IntLit { span: sp(), value: 1, ty: None }),
    }));

    // x^x
    exprs.push(("x^x".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::BitXor,
        lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        rhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
    }));

    // x&0
    exprs.push(("x&0".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::BitAnd,
        lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        rhs: Box::new(Expr::IntLit { span: sp(), value: 0, ty: None }),
    }));

    // x|0
    exprs.push(("x|0".into(), Expr::BinOp {
        span: sp(), op: BinOpKind::BitOr,
        lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
        rhs: Box::new(Expr::IntLit { span: sp(), value: 0, ty: None }),
    }));

    exprs
}

fn main() {
    println!("================================================================");
    println!("  GNN E-Graph Superoptimizer — Training & Benchmark");
    println!("================================================================");
    println!();

    let training_exprs = make_training_expressions();
    println!("  Training expressions: {}", training_exprs.len());

    // ── Phase 1: Model Architecture ──
    println!();
    println!("-- Phase 1: Model Architecture --");
    println!();

    let small_config = GnnConfig::small();
    let medium_config = GnnConfig::medium();
    let large_config = GnnConfig::large();

    println!("  Small config:  ~{} params (hidden={}, layers={}, heads={})",
        small_config.estimate_params(), small_config.hidden_dim, small_config.num_layers, small_config.num_heads);
    println!("  Medium config: ~{} params (hidden={}, layers={}, heads={})",
        medium_config.estimate_params(), medium_config.hidden_dim, medium_config.num_layers, medium_config.num_heads);
    println!("  Large config:  ~{} params (hidden={}, layers={}, heads={})",
        large_config.estimate_params(), large_config.hidden_dim, large_config.num_layers, large_config.num_heads);

    // ── Phase 2: Generate Training Data ──
    println!();
    println!("-- Phase 2: Generate Training Data --");
    println!();

    let t0 = Instant::now();
    let episodes = generate_training_data(&training_exprs, 5);
    let gen_time = t0.elapsed();
    println!("  Generated {} training episodes in {:?}", episodes.len(), gen_time);

    let improved = episodes.iter().filter(|e| e.reward > 0.0).count();
    println!("  Episodes with improvement: {}/{} ({:.0}%)",
        improved, episodes.len(),
        improved as f64 / episodes.len().max(1) as f64 * 100.0);

    // ── Phase 3: Train Medium Model ──
    println!();
    println!("-- Phase 3: Train Medium Model (~50k params) --");
    println!();

    let mut gnn_opt = GnnEgraphOptimizer::new(medium_config.clone());
    println!("  Actual parameter count: {}", gnn_opt.param_count());

    // Fast training on pre-generated data
    gnn_opt.train_fast(&episodes, 50);

    // ── Phase 4: GNN Inference Benchmark ──
    println!();
    println!("-- Phase 4: GNN Inference vs MCTS Comparison --");
    println!();

    let mut mcts_opt = MctsSuperoptimizer::new(MctsConfig::fast());
    let mut cost_est = CycleCostEstimator::new(None);

    println!("  {:25}  {:>8}  {:>8}  {:>8}  {:>8}  {:>6}  {:>6}",
        "Expression", "OrigCyc", "MCTSCyc", "GNNCyc", "GNN2xCyc", "MCTS", "GNN");
    println!("  {}", "-".repeat(80));

    let mut total_orig = 0u32;
    let mut total_mcts = 0u32;
    let mut total_gnn = 0u32;
    let mut total_gnn2 = 0u32;
    let mut mcts_improved = 0usize;
    let mut gnn_improved = 0usize;
    let mut gnn2_improved = 0usize;

    for (name, expr) in &training_exprs {
        let instr = Instr::from_expr(expr).unwrap();
        let original_cost = cost_est.estimate(&instr);

        // MCTS optimization
        let t_mcts = Instant::now();
        let mcts_result = mcts_opt.optimize(expr);
        let mcts_time = t_mcts.elapsed();
        let mcts_cost = if let Some(ref opt) = mcts_result {
            cost_est.estimate(&Instr::from_expr(opt).unwrap_or_else(|| instr.clone()))
        } else { original_cost };

        // GNN single-step optimization
        let t_gnn = Instant::now();
        let gnn_result = gnn_opt.optimize_gnn_only(expr);
        let gnn_time = t_gnn.elapsed();
        let gnn_cost = gnn_result.optimized_cost;

        // GNN multi-step optimization
        let t_gnn2 = Instant::now();
        let gnn2_result = gnn_optimize_multi_step(&mut gnn_opt, expr, 5);
        let gnn2_time = t_gnn2.elapsed();
        let gnn2_cost = gnn2_result.optimized_cost;

        total_orig += original_cost;
        total_mcts += mcts_cost;
        total_gnn += gnn_cost;
        total_gnn2 += gnn2_cost;
        if mcts_cost < original_cost { mcts_improved += 1; }
        if gnn_cost < original_cost { gnn_improved += 1; }
        if gnn2_cost < original_cost { gnn2_improved += 1; }

        println!("  {:25}  {:>8}  {:>8}  {:>8}  {:>8}  {:>5?}  {:>5?}",
            name, original_cost, mcts_cost, gnn_cost, gnn2_cost,
            mcts_time, gnn_time,
        );
    }

    let mcts_red = if total_orig > 0 { (total_orig - total_mcts) as f64 / total_orig as f64 * 100.0 } else { 0.0 };
    let gnn_red = if total_orig > 0 { (total_orig - total_gnn) as f64 / total_orig as f64 * 100.0 } else { 0.0 };
    let gnn2_red = if total_orig > 0 { (total_orig - total_gnn2) as f64 / total_orig as f64 * 100.0 } else { 0.0 };

    println!();
    println!("  MCTS:  {}/{} improved, {:.1}% cycle reduction", mcts_improved, training_exprs.len(), mcts_red);
    println!("  GNN-1: {}/{} improved, {:.1}% cycle reduction", gnn_improved, training_exprs.len(), gnn_red);
    println!("  GNN-5: {}/{} improved, {:.1}% cycle reduction", gnn2_improved, training_exprs.len(), gnn2_red);

    // ── Phase 5: Throughput Comparison ──
    println!();
    println!("-- Phase 5: Throughput Stress Test --");
    println!();

    let stress_count = 100;
    let stress_expr = Expr::BinOp {
        span: sp(), op: BinOpKind::Add,
        lhs: Box::new(Expr::BinOp {
            span: sp(), op: BinOpKind::Mul,
            lhs: Box::new(Expr::Ident { span: sp(), name: "x".into() }),
            rhs: Box::new(Expr::IntLit { span: sp(), value: 1, ty: None }),
        }),
        rhs: Box::new(Expr::IntLit { span: sp(), value: 0, ty: None }),
    };

    // MCTS throughput
    let mut mcts_stress = MctsSuperoptimizer::new(MctsConfig::fast());
    let t0 = Instant::now();
    let mut mcts_improved_count = 0;
    for _ in 0..stress_count {
        if mcts_stress.optimize(&stress_expr).is_some() {
            mcts_improved_count += 1;
        }
    }
    let mcts_elapsed = t0.elapsed();
    let mcts_per = mcts_elapsed / stress_count as u32;

    // GNN throughput
    let t0 = Instant::now();
    let mut gnn_improved_count = 0;
    for _ in 0..stress_count {
        let result = gnn_opt.optimize_gnn_only(&stress_expr);
        if result.is_improved() { gnn_improved_count += 1; }
    }
    let gnn_elapsed = t0.elapsed();
    let gnn_per = gnn_elapsed / stress_count as u32;

    // GNN multi-step throughput
    let t0 = Instant::now();
    let mut gnn2_improved_count = 0;
    for _ in 0..stress_count {
        let result = gnn_optimize_multi_step(&mut gnn_opt, &stress_expr, 3);
        if result.is_improved() { gnn2_improved_count += 1; }
    }
    let gnn2_elapsed = t0.elapsed();
    let gnn2_per = gnn2_elapsed / stress_count as u32;

    println!("  MCTS:   {} exprs in {:?} ({:?}/expr, {:.0} exprs/sec)",
        stress_count, mcts_elapsed, mcts_per,
        stress_count as f64 / mcts_elapsed.as_secs_f64().max(1e-12));
    println!("  GNN-1:  {} exprs in {:?} ({:?}/expr, {:.0} exprs/sec)",
        stress_count, gnn_elapsed, gnn_per,
        stress_count as f64 / gnn_elapsed.as_secs_f64().max(1e-12));
    println!("  GNN-5:  {} exprs in {:?} ({:?}/expr, {:.0} exprs/sec)",
        stress_count, gnn2_elapsed, gnn2_per,
        stress_count as f64 / gnn2_elapsed.as_secs_f64().max(1e-12));

    let speedup = mcts_elapsed.as_secs_f64() / gnn_elapsed.as_secs_f64().max(1e-12);
    println!("  GNN speedup over MCTS: {:.1}x", speedup);

    // ── Phase 6: GNN Policy Analysis ──
    println!();
    println!("-- Phase 6: GNN Policy Analysis --");
    println!();

    let actions = RewriteAction::all();
    println!("  {:25}  {:>10}  {:>10}  {:>8}", "Expression", "Top Action", "Confidence", "Value");
    println!("  {}", "-".repeat(60));

    for (name, expr) in &training_exprs {
        let result = gnn_opt.optimize_gnn_only(expr);
        let top_action_name = if result.action_selected < actions.len() {
            format!("{:?}", actions[result.action_selected])
        } else {
            "None".to_string()
        };
        let confidence = result.policy.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("  {:25}  {:>10}  {:>10.3}  {:>8.3}",
            name, top_action_name, confidence, result.value_prediction);
    }

    // ── Phase 7: Pre-Trained 100k-Param GNN Benchmark ──
    println!();
    println!("-- Phase 7: Pre-Trained 100k-Param GNN (Baked-In Weights) --");
    println!();

    let pretrained_model = load_pretrained_gnn();
    println!("  Pre-trained model loaded: {} params", {
        let mut c = 0;
        c += pretrained_model.embedding.len() + pretrained_model.embedding_bias.len();
        for layer in &pretrained_model.gnn_layers {
            c += layer.w_self.len() + layer.w_neigh.len() + layer.bias.len();
            c += layer.layer_norm.gamma.len() + layer.layer_norm.beta.len();
            if let Some(k) = &layer.attn_key { c += k.len(); }
            if let Some(q) = &layer.attn_query { c += q.len(); }
        }
        c += pretrained_model.readout.w.len() + pretrained_model.readout.b.len();
        c += pretrained_model.policy_head.w1.len() + pretrained_model.policy_head.b1.len()
           + pretrained_model.policy_head.w2.len() + pretrained_model.policy_head.b2.len();
        c += pretrained_model.value_head.w1.len() + pretrained_model.value_head.b1.len()
           + pretrained_model.value_head.w2.len() + pretrained_model.value_head.b2.len();
        c
    });

    // Create optimizer with pre-trained model
    let pretrained_config = GnnConfig {
        hidden_dim: 96,
        num_layers: 4,
        num_heads: 4,
        num_actions: 12,
        node_feature_dim: 33,
        learning_rate: 0.001,
        gamma: 0.99,
        num_episodes: 0,
        batch_size: 32,
        use_attention: true,
        value_loss_coef: 0.5,
        entropy_coef: 0.01,
    };

    let mut pretrained_opt = GnnEgraphOptimizer::new(pretrained_config);
    // Replace the model with the pre-trained one
    *pretrained_opt.model_mut() = pretrained_model;

    // Benchmark pre-trained GNN
    let mut pretrained_improved = 0usize;
    let mut total_pretrained = 0u32;

    println!("  {:25}  {:>8}  {:>8}  {:>8}  {:>8}", "Expression", "OrigCyc", "MCTSCyc", "PreGNN1", "PreGNN5");
    println!("  {}", "-".repeat(65));

    for (name, expr) in &training_exprs {
        let instr = Instr::from_expr(expr).unwrap();
        let original_cost = cost_est.estimate(&instr);

        let mcts_cost = {
            let res = mcts_opt.optimize(expr);
            if let Some(ref opt) = res {
                cost_est.estimate(&Instr::from_expr(opt).unwrap_or_else(|| instr.clone()))
            } else { original_cost }
        };

        // Pre-trained GNN single-step
        let pretrained1_result = pretrained_opt.optimize_gnn_only(expr);
        let pretrained1_cost = pretrained1_result.optimized_cost;

        // Pre-trained GNN multi-step
        let pretrained5_result = gnn_optimize_multi_step(&mut pretrained_opt, expr, 5);
        let pretrained5_cost = pretrained5_result.optimized_cost;

        total_pretrained += pretrained5_cost;
        if pretrained5_cost < original_cost { pretrained_improved += 1; }

        println!("  {:25}  {:>8}  {:>8}  {:>8}  {:>8}",
            name, original_cost, mcts_cost, pretrained1_cost, pretrained5_cost);
    }

    let pretrained_red = if total_orig > 0 { (total_orig - total_pretrained) as f64 / total_orig as f64 * 100.0 } else { 0.0 };
    println!();
    println!("  Pre-trained GNN (100k params): {}/{} improved, {:.1}% cycle reduction", 
        pretrained_improved, training_exprs.len(), pretrained_red);

    // Pre-trained throughput
    let t0 = Instant::now();
    let mut pretrained_improved_count = 0;
    for _ in 0..stress_count {
        let result = pretrained_opt.optimize_gnn_only(&stress_expr);
        if result.is_improved() { pretrained_improved_count += 1; }
    }
    let pretrained_elapsed = t0.elapsed();

    println!("  Pre-trained GNN throughput: {} exprs in {:?} ({:.0} exprs/sec)",
        stress_count, pretrained_elapsed,
        stress_count as f64 / pretrained_elapsed.as_secs_f64().max(1e-12));

    let pretrained_speedup = mcts_elapsed.as_secs_f64() / pretrained_elapsed.as_secs_f64().max(1e-12);
    println!("  Pre-trained GNN speedup over MCTS: {:.1}x", pretrained_speedup);

    // ── Final Summary ──
    println!();
    println!("================================================================");
    println!("  GNN E-GRAPH SUPEROPTIMIZER — FINAL RESULTS");
    println!("================================================================");
    println!();
    println!("  ┌────────────────────────────────────────────────────────────┐");
    println!("  │  Model: {} params (hidden={}, layers={}, heads={})  │",
        gnn_opt.param_count(), medium_config.hidden_dim, medium_config.num_layers, medium_config.num_heads);
    println!("  ├────────────────────────────────────────────────────────────┤");
    println!("  │  MCTS:  {}/{} improved, {:.1}% cycle reduction               │",
        mcts_improved, training_exprs.len(), mcts_red);
    println!("  │  GNN-1: {}/{} improved, {:.1}% cycle reduction               │",
        gnn_improved, training_exprs.len(), gnn_red);
    println!("  │  GNN-5: {}/{} improved, {:.1}% cycle reduction               │",
        gnn2_improved, training_exprs.len(), gnn2_red);
    println!("  ├────────────────────────────────────────────────────────────┤");
    println!("  │  Throughput: MCTS={:.0}/s  GNN={:.0}/s  ({:.1}x faster)      │",
        stress_count as f64 / mcts_elapsed.as_secs_f64().max(1e-12),
        stress_count as f64 / gnn_elapsed.as_secs_f64().max(1e-12),
        speedup);
    println!("  └────────────────────────────────────────────────────────────┘");
    println!();
    println!("  GNN E-Graph Superoptimizer status: OPERATIONAL");
}
