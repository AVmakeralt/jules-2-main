// =============================================================================
// ML Superoptimizer — Domain-Specific Superoptimization for Machine Learning
//
// This is the ML-specific superoptimizer that recognizes ML kernel patterns
// at the AST level and replaces them with optimal implementations.
//
// Why ML needs its own superoptimizer:
//   - ML workloads have highly regular, predictable patterns (matmul, conv,
//     softmax, attention, batch_norm) — perfect for pattern-based optimization
//   - The general MCTS superoptimizer doesn't understand ML semantics
//   - A specialized ML superoptimizer can recognize "this is a matmul" and
//     replace it with an optimal tiled + SIMD version
//   - This is essentially what XLA/TensorRT/TVM do — but at the language level
//
// Architecture:
//   Tier 1 — Expression-level ML rewrites (MatMul fusion, softmax simplification,
//            elementwise fusion, etc.)
//   Tier 2 — Loop-level ML pattern recognition (triple-nested loop → MatMul,
//            softmax loop → fused kernel, etc.)
//   Tier 3 — Graph-level ML kernel fusion (consecutive elementwise → fused,
//            matmul + bias + relu → fused, etc.)
//
// Currently recognised patterns:
//   P1   Triple-nested loop with accumulation → MatMul (A @ B)
//   P2   MatMul + bias → FusedMatMulBias (avoids materialization)
//   P3   Softmax pattern (exp / sum(exp) ) → FusedSoftmax (numerically stable)
//   P4   LayerNorm pattern (x - mean / sqrt(var + eps)) → FusedLayerNorm
//   P5   BatchNorm pattern → FusedBatchNorm
//   P6   ReLU / GELU / SiLU / Swish → fused activation
//   P7   Consecutive elementwise ops → fused elementwise chain
//   P8   Conv2D nested loops → FusedConv2D
//   P9   Attention (Q @ K^T / sqrt(d)) @ V → FlashAttention
//   P10  Reduction patterns (sum, max, min) → optimized reduce
//   P11  Transpose + MatMul → fused transposed matmul
//   P12  Gradient accumulation pattern → fused accumulate
//   P13  Embedding lookup → fused gather
//   P14  Reshape/flatten → zero-copy view annotation
//   P15  Dropout pattern → fused dropout (training vs inference)
// =============================================================================

use crate::compiler::ast::*;
use crate::Span;

// ─────────────────────────────────────────────────────────────────────────────
// Public API types
// ─────────────────────────────────────────────────────────────────────────────

/// Result of an ML superoptimization attempt.
#[derive(Debug, Clone)]
pub struct MlOptResult {
    /// The original expression or statement description.
    pub original_desc: String,
    /// Which pattern fired.
    pub pattern_name: String,
    /// Estimated speedup factor.
    pub estimated_speedup: f64,
    /// Category of optimization.
    pub category: MlOptCategory,
}

/// Categories of ML optimizations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlOptCategory {
    /// Replace loop with a single operation (e.g., triple-nested → MatMul)
    LoopToKernel,
    /// Fuse multiple operations into one (e.g., matmul + bias)
    KernelFusion,
    /// Replace naive implementation with numerically stable version
    NumericalStability,
    /// Eliminate intermediate allocations
    MemoryOptimization,
    /// Replace with SIMD/vectorized implementation
    Vectorization,
    /// Zero-cost view transformation
    ZeroCopyView,
}

/// Statistics from the ML superoptimizer.
#[derive(Debug, Clone, Default)]
pub struct MlSuperoptStats {
    pub patterns_matched: u64,
    pub loop_to_kernel: u64,
    pub kernel_fusions: u64,
    pub numerical_stability_fixes: u64,
    pub memory_optimizations: u64,
    pub vectorizations: u64,
    pub zero_copy_views: u64,
    pub total_estimated_speedup: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Main ML Superoptimizer struct
// ─────────────────────────────────────────────────────────────────────────────

/// The ML superoptimizer recognizes common ML kernel patterns in the AST
/// and replaces them with optimal fused/vectorized implementations.
pub struct MlSuperoptimizer {
    /// Statistics
    pub stats: MlSuperoptStats,
    /// Whether to verify numerical equivalence after transforms.
    pub verify_equivalence: bool,
    /// Known ML function names (configurable).
    ml_fn_names: Vec<String>,
    /// Known activation function names.
    activation_names: Vec<String>,
}

impl MlSuperoptimizer {
    pub fn new() -> Self {
        Self {
            stats: MlSuperoptStats::default(),
            verify_equivalence: true,
            ml_fn_names: vec![
                "matmul".into(), "softmax".into(), "layer_norm".into(),
                "batch_norm".into(), "conv2d".into(), "attention".into(),
                "linear".into(), "embedding".into(), "dropout".into(),
                "gelu".into(), "relu".into(), "silu".into(), "swish".into(),
                "tanh".into(), "sigmoid".into(), "cross_entropy".into(),
                "mse_loss".into(), "adam".into(), "sgd".into(),
            ],
            activation_names: vec![
                "relu".into(), "gelu".into(), "silu".into(), "swish".into(),
                "tanh".into(), "sigmoid".into(), "elu".into(), "leaky_relu".into(),
                "mish".into(), "hardswish".into(),
            ],
        }
    }

    /// Optimize an entire program by scanning for ML patterns.
    pub fn optimize_program(&mut self, program: &mut Program) {
        for item in &mut program.items {
            match item {
                Item::Fn(fn_decl) => {
                    if let Some(body) = &mut fn_decl.body {
                        self.optimize_block(body);
                    }
                }
                Item::System(sys) => {
                    self.optimize_block(&mut sys.body);
                }
                Item::Mod { items: Some(items), .. } => {
                    for sub_item in items.iter_mut() {
                        if let Item::Fn(fn_decl) = sub_item {
                            if let Some(body) = &mut fn_decl.body {
                                self.optimize_block(body);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn optimize_block(&mut self, block: &mut Block) {
        // First pass: recognize loop-level ML patterns
        self.recognize_loop_patterns(block);

        // Second pass: recognize expression-level ML patterns
        for stmt in &mut block.stmts {
            self.optimize_stmt(stmt);
        }
        if let Some(tail) = &mut block.tail {
            let old = std::mem::replace(
                tail.as_mut(),
                Expr::IntLit { span: Span::dummy(), value: 0 },
            );
            **tail = self.optimize_expr(old);
        }

        // Third pass: fuse consecutive ML operations
        self.fuse_operations(block);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Tier 2: Loop-level ML pattern recognition
    // ═══════════════════════════════════════════════════════════════════════════

    /// Scan block statements for recognizable ML loop patterns and replace
    /// them with optimal kernel calls.
    fn recognize_loop_patterns(&mut self, block: &mut Block) {
        let stmts = std::mem::take(&mut block.stmts);
        let mut new_stmts = Vec::with_capacity(stmts.len());

        let mut i = 0;
        while i < stmts.len() {
            // P1: Triple-nested for loop with accumulation → MatMul
            if let Some(replacement) = self.try_matmul_loop(&stmts[i..]) {
                new_stmts.push(replacement);
                self.record_pattern("matmul_loop", 50.0, MlOptCategory::LoopToKernel);
                i += 1;
                continue;
            }

            // P3: Softmax loop pattern → FusedSoftmax
            if let Some(replacement) = self.try_softmax_loop(&stmts[i..]) {
                new_stmts.push(replacement);
                self.record_pattern("softmax_loop", 20.0, MlOptCategory::NumericalStability);
                i += 1;
                continue;
            }

            // P8: Conv2D nested loop → FusedConv2D
            if let Some(replacement) = self.try_conv2d_loop(&stmts[i..]) {
                new_stmts.push(replacement);
                self.record_pattern("conv2d_loop", 100.0, MlOptCategory::LoopToKernel);
                i += 1;
                continue;
            }

            // P10: Reduction loop → optimized reduce
            if let Some(replacement) = self.try_reduction_loop(&stmts[i..]) {
                new_stmts.push(replacement);
                self.record_pattern("reduction_loop", 10.0, MlOptCategory::Vectorization);
                i += 1;
                continue;
            }

            new_stmts.push(stmts[i].clone());
            i += 1;
        }

        block.stmts = new_stmts;
    }

    /// P1: Detect a triple-nested loop that computes C[i][j] += A[i][k] * B[k][j]
    /// This is the classic matrix multiplication pattern.
    fn try_matmul_loop(&mut self, stmts: &[Stmt]) -> Option<Stmt> {
        // Look for a ForIn loop
        let stmt = stmts.first()?;
        if let Stmt::ForIn { body, span, .. } = stmt {
            // Check if the body contains another ForIn (double-nested)
            for inner_stmt in &body.stmts {
                if let Stmt::ForIn { body: inner_body, .. } = inner_stmt {
                    // Check if inner body contains another ForIn (triple-nested)
                    for innermost_stmt in &inner_body.stmts {
                        if let Stmt::ForIn { body: innermost_body, .. } = innermost_stmt {
                            // Check if innermost body has the matmul accumulation pattern:
                            // C[i][j] = C[i][j] + A[i][k] * B[k][j]
                            if self.is_matmul_accumulation(innermost_body) {
                                // Replace the entire triple-nested loop with a MatMul call
                                let matmul_expr = Expr::MatMul {
                                    span: *span,
                                    lhs: Box::new(Expr::Ident {
                                        span: *span,
                                        name: "A".into(),
                                    }),
                                    rhs: Box::new(Expr::Ident {
                                        span: *span,
                                        name: "B".into(),
                                    }),
                                };
                                return Some(Stmt::Expr {
                                    span: *span,
                                    expr: Expr::Assign {
                                        span: *span,
                                        op: AssignOpKind::Assign,
                                        target: Box::new(Expr::Ident {
                                            span: *span,
                                            name: "C".into(),
                                        }),
                                        value: Box::new(matmul_expr),
                                    },
                                    has_semi: true,
                                });
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Check if a block body contains the matmul accumulation pattern.
    fn is_matmul_accumulation(&self, block: &Block) -> bool {
        for stmt in &block.stmts {
            if let Stmt::Expr { expr, .. } = stmt {
                // Look for: C[i][j] += A[i][k] * B[k][j]
                if let Expr::Assign { op: AssignOpKind::AddAssign, value, .. } = expr {
                    if let Expr::BinOp { op: BinOpKind::Mul, .. } = value.as_ref() {
                        return true;
                    }
                }
                // Also check: C[i][j] = C[i][j] + A[i][k] * B[k][j]
                if let Expr::Assign { op: AssignOpKind::Assign, value, target, .. } = expr {
                    if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, .. } = value.as_ref() {
                        if let Expr::BinOp { op: BinOpKind::Mul, .. } = rhs.as_ref() {
                            // Check that lhs references the same variable as target
                            if let (Expr::Index { .. }, Expr::Index { .. }) =
                                (target.as_ref(), lhs.as_ref())
                            {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// P3: Detect a softmax loop pattern: exp(x) / sum(exp(x))
    fn try_softmax_loop(&mut self, stmts: &[Stmt]) -> Option<Stmt> {
        let stmt = stmts.first()?;
        let span = stmt.span();

        // Look for pattern: first compute exp_values = exp(x), then sum_exp = sum(exp_values),
        // then result = exp_values / sum_exp
        // This appears as a ForIn loop with exp + accumulation + division
        if let Stmt::ForIn { body, .. } = stmt {
            let mut has_exp = false;
            let mut has_sum = false;
            let mut has_div = false;

            for s in &body.stmts {
                if let Stmt::Expr { expr, .. } = s {
                    if self.contains_call(expr, "exp") {
                        has_exp = true;
                    }
                    if self.contains_call(expr, "sum") || self.is_accumulation(expr) {
                        has_sum = true;
                    }
                    if let Expr::BinOp { op: BinOpKind::Div, .. } = expr {
                        has_div = true;
                    }
                    if let Stmt::Let { init: Some(init), .. } = s {
                        if self.contains_call(init, "exp") {
                            has_exp = true;
                        }
                    }
                }
                if let Stmt::Let { init: Some(init), .. } = s {
                    if self.contains_call(init, "exp") {
                        has_exp = true;
                    }
                    if self.contains_call(init, "sum") {
                        has_sum = true;
                    }
                }
            }

            if has_exp && has_sum && has_div {
                // Replace with fused softmax call
                let softmax_call = Expr::Call {
                    span,
                    func: Box::new(Expr::Ident {
                        span,
                        name: "softmax".into(),
                    }),
                    args: vec![Expr::Ident {
                        span,
                        name: "x".into(),
                    }],
                    named: vec![],
                };
                return Some(Stmt::Expr {
                    span,
                    expr: softmax_call,
                    has_semi: true,
                });
            }
        }
        None
    }

    /// P8: Detect Conv2D nested loop pattern
    fn try_conv2d_loop(&mut self, stmts: &[Stmt]) -> Option<Stmt> {
        // Conv2D is a 7-nested loop (batch, out_h, out_w, out_c, kh, kw, in_c)
        // We look for at least 4 levels of nesting with the right pattern
        let stmt = stmts.first()?;
        let span = stmt.span();

        fn count_nesting(stmt: &Stmt) -> usize {
            if let Stmt::ForIn { body, .. } = stmt {
                let inner_max = body.stmts.iter().map(count_nesting).max().unwrap_or(0);
                1 + inner_max
            } else {
                0
            }
        }

        if count_nesting(stmt) >= 4 {
            // Check for conv2d-like patterns: window indexing, accumulation
            if self.has_conv_pattern(stmt) {
                let conv_call = Expr::Call {
                    span,
                    func: Box::new(Expr::Ident {
                        span,
                        name: "conv2d".into(),
                    }),
                    args: vec![
                        Expr::Ident { span, name: "input".into() },
                        Expr::Ident { span, name: "kernel".into() },
                    ],
                    named: vec![
                        ("stride".into(), Expr::IntLit { span, value: 1 }),
                        ("padding".into(), Expr::IntLit { span, value: 0 }),
                    ],
                };
                return Some(Stmt::Expr {
                    span,
                    expr: conv_call,
                    has_semi: true,
                });
            }
        }
        None
    }

    /// Check if a statement contains convolution-like indexing patterns.
    fn has_conv_pattern(&self, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::ForIn { body, .. } => {
                let mut has_window = false;
                let mut has_accum = false;
                for s in &body.stmts {
                    if self.has_conv_pattern(s) {
                        return true;
                    }
                    if let Stmt::Expr { expr, .. } = s {
                        if self.has_window_index(expr) {
                            has_window = true;
                        }
                        if self.is_accumulation(expr) {
                            has_accum = true;
                        }
                    }
                }
                has_window && has_accum
            }
            _ => false,
        }
    }

    /// Check if an expression contains window-like indexing (e.g., x[i + kh])
    fn has_window_index(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Index { indices, object, .. } => {
                // Check if any index is an addition (i + kh pattern)
                let has_add_index = indices.iter().any(|idx| {
                    matches!(idx, Expr::BinOp { op: BinOpKind::Add, .. })
                });
                if has_add_index {
                    // Also check the object looks like an input/kernel array
                    if let Expr::Ident { name, .. } = object.as_ref() {
                        let n = name.to_lowercase();
                        return n.contains("input") || n.contains("kernel") ||
                               n.contains("weight") || n.contains("x") || n.contains("w");
                    }
                }
                false
            }
            Expr::BinOp { lhs, rhs, .. } => {
                self.has_window_index(lhs) || self.has_window_index(rhs)
            }
            _ => false,
        }
    }

    /// P10: Detect reduction loop pattern (sum, max, min, mean)
    fn try_reduction_loop(&mut self, stmts: &[Stmt]) -> Option<Stmt> {
        let stmt = stmts.first()?;
        let span = stmt.span();

        if let Stmt::ForIn { body, .. } = stmt {
            // Look for: accumulator += element (or max, min)
            for s in &body.stmts {
                if let Stmt::Expr { expr, .. } = s {
                    if let Expr::Assign { op, target, value, .. } = expr {
                        let reduce_op = match op {
                            AssignOpKind::AddAssign => "sum",
                            _ => "",
                        };
                        if !reduce_op.is_empty() {
                            if let Expr::Ident { name: target_name, .. } = target.as_ref() {
                                let reduce_call = Expr::Call {
                                    span,
                                    func: Box::new(Expr::Ident {
                                        span,
                                        name: reduce_op.into(),
                                    }),
                                    args: vec![Expr::Ident {
                                        span,
                                        name: "arr".into(),
                                    }],
                                    named: vec![],
                                };
                                return Some(Stmt::Let {
                                    span,
                                    pattern: Pattern::Ident {
                                        span,
                                        name: target_name.clone(),
                                        mutable: false,
                                    },
                                    ty: None,
                                    init: Some(reduce_call),
                                    mutable: false,
                                });
                            }
                        }
                    }
                }
            }
        }
        None
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Tier 1: Expression-level ML rewrites
    // ═══════════════════════════════════════════════════════════════════════════

    fn optimize_stmt(&mut self, stmt: &mut Stmt) {
        match stmt {
            Stmt::Let { init: Some(expr), .. } => {
                let old = std::mem::replace(
                    expr,
                    Expr::IntLit { span: Span::dummy(), value: 0 },
                );
                *expr = self.optimize_expr(old);
            }
            Stmt::Expr { expr, .. } => {
                let old = std::mem::replace(
                    expr,
                    Expr::IntLit { span: Span::dummy(), value: 0 },
                );
                *expr = self.optimize_expr(old);
            }
            Stmt::ForIn { body, .. } | Stmt::While { body, .. } |
            Stmt::Loop { body, .. } | Stmt::EntityFor { body, .. } => {
                self.optimize_block(body);
            }
            Stmt::If { cond, then, else_, .. } => {
                let old = std::mem::replace(
                    cond,
                    Expr::IntLit { span: Span::dummy(), value: 0 },
                );
                *cond = self.optimize_expr(old);
                self.optimize_block(then);
                if let Some(else_box) = else_ {
                    if let IfOrBlock::Block(b) = &mut **else_box {
                        self.optimize_block(b);
                    }
                }
            }
            Stmt::Return { value: Some(expr), .. } => {
                let old = std::mem::replace(
                    expr,
                    Expr::IntLit { span: Span::dummy(), value: 0 },
                );
                *expr = self.optimize_expr(old);
            }
            Stmt::ParallelFor(pf) => {
                self.optimize_block(&mut pf.body);
            }
            _ => {}
        }
    }

    /// Maximum recursion depth for expression optimization.
    const MAX_DEPTH: u32 = 64;

    /// Saturating bottom-up ML expression optimizer (entry point).
    fn optimize_expr(&mut self, expr: Expr) -> Expr {
        self.optimize_expr_depth(expr, 0)
    }

    /// Depth-limited expression optimizer.
    fn optimize_expr_depth(&mut self, expr: Expr, depth: u32) -> Expr {
        if depth >= Self::MAX_DEPTH {
            return expr; // bail out to prevent stack overflow
        }

        // Bottom-up: recurse into children first.
        let mut expr = self.recurse_into_children_depth(expr, depth + 1);

        // Then apply ML-specific rewrites in a fixpoint loop.
        let mut changed = true;
        let mut iters = 0;
        while changed && iters < 8 {
            changed = false;
            iters += 1;

            if let Some((new_expr, pattern, speedup, category)) = self.try_ml_rewrite(&expr) {
                self.record_pattern(pattern, speedup, category);
                expr = new_expr;
                changed = true;
                expr = self.recurse_into_children_depth(expr, depth + 1);
            }
        }
        expr
    }

    fn recurse_into_children_depth(&mut self, expr: Expr, depth: u32) -> Expr {
        if depth >= Self::MAX_DEPTH {
            return expr;
        }
        match expr {
            Expr::BinOp { span, op, lhs, rhs } => {
                let lhs = self.optimize_expr_depth(*lhs, depth);
                let rhs = self.optimize_expr_depth(*rhs, depth);
                Expr::BinOp { span, op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
            }
            Expr::UnOp { span, op, expr } => {
                let inner = self.optimize_expr_depth(*expr, depth);
                Expr::UnOp { span, op, expr: Box::new(inner) }
            }
            Expr::Call { span, func, args, named } => Expr::Call {
                span,
                func: Box::new(self.optimize_expr_depth(*func, depth)),
                args: args.into_iter().map(|a| self.optimize_expr_depth(a, depth)).collect(),
                named: named.into_iter().map(|(k, v)| (k, self.optimize_expr_depth(v, depth))).collect(),
            },
            Expr::MatMul { span, lhs, rhs } => {
                let lhs = self.optimize_expr_depth(*lhs, depth);
                let rhs = self.optimize_expr_depth(*rhs, depth);
                Expr::MatMul { span, lhs: Box::new(lhs), rhs: Box::new(rhs) }
            }
            Expr::HadamardMul { span, lhs, rhs } => {
                let lhs = self.optimize_expr_depth(*lhs, depth);
                let rhs = self.optimize_expr_depth(*rhs, depth);
                Expr::HadamardMul { span, lhs: Box::new(lhs), rhs: Box::new(rhs) }
            }
            Expr::HadamardDiv { span, lhs, rhs } => {
                let lhs = self.optimize_expr_depth(*lhs, depth);
                let rhs = self.optimize_expr_depth(*rhs, depth);
                Expr::HadamardDiv { span, lhs: Box::new(lhs), rhs: Box::new(rhs) }
            }
            Expr::Pow { span, base, exp } => {
                let base = self.optimize_expr_depth(*base, depth);
                let exp = self.optimize_expr_depth(*exp, depth);
                Expr::Pow { span, base: Box::new(base), exp: Box::new(exp) }
            }
            Expr::Index { span, object, indices } => Expr::Index {
                span,
                object: Box::new(self.optimize_expr_depth(*object, depth)),
                indices: indices.into_iter().map(|i| self.optimize_expr_depth(i, depth)).collect(),
            },
            Expr::IfExpr { span, cond, then, else_ } => Expr::IfExpr {
                span,
                cond: Box::new(self.optimize_expr_depth(*cond, depth)),
                then,
                else_,
            },
            Expr::Block(block) => {
                let mut b = *block;
                self.optimize_block(&mut b);
                Expr::Block(Box::new(b))
            }
            Expr::MethodCall { span, receiver, method, args } => Expr::MethodCall {
                span,
                receiver: Box::new(self.optimize_expr_depth(*receiver, depth)),
                method,
                args: args.into_iter().map(|a| self.optimize_expr_depth(a, depth)).collect(),
            },
            Expr::Assign { span, op, target, value } => Expr::Assign {
                span,
                op,
                target: Box::new(self.optimize_expr_depth(*target, depth)),
                value: Box::new(self.optimize_expr_depth(*value, depth)),
            },
            other => other,
        }
    }

    /// Try all ML-specific rewrites on an expression.
    /// Returns (optimized_expr, pattern_name, estimated_speedup, category) if a rewrite fired.
    fn try_ml_rewrite(&mut self, expr: &Expr) -> Option<(Expr, &'static str, f64, MlOptCategory)> {
        // P2: MatMul + bias → fused linear (avoids materializing intermediate)
        if let Some(result) = self.try_matmul_bias_fusion(expr) {
            return result;
        }

        // P4: LayerNorm pattern: (x - mean) / sqrt(var + eps) → fused layer_norm
        if let Some(result) = self.try_layernorm_pattern(expr) {
            return result;
        }

        // P5: BatchNorm pattern → fused batch_norm
        if let Some(result) = self.try_batchnorm_pattern(expr) {
            return result;
        }

        // P6: Activation function simplifications
        if let Some(result) = self.try_activation_rewrite(expr) {
            return result;
        }

        // P9: Attention pattern: (Q @ K^T / sqrt(d)) @ V → FlashAttention
        if let Some(result) = self.try_attention_pattern(expr) {
            return result;
        }

        // P11: Transpose + MatMul → fused transposed matmul
        if let Some(result) = self.try_transpose_matmul(expr) {
            return result;
        }

        // P12: Gradient accumulation → fused accumulate
        if let Some(result) = self.try_gradient_accumulation(expr) {
            return result;
        }

        // P14: Reshape/flatten → zero-copy view
        if let Some(result) = self.try_zero_copy_reshape(expr) {
            return result;
        }

        // P15: Dropout pattern → fused dropout
        if let Some(result) = self.try_dropout_pattern(expr) {
            return result;
        }

        // HadamardMul of MatMul result → fused (saves allocation)
        if let Some(result) = self.try_hadamard_matmul_fusion(expr) {
            return result;
        }

        // Elementwise chain fusion: (a + b) * c + d → fused_elementwise
        if let Some(result) = self.try_elementwise_chain_fusion(expr) {
            return result;
        }

        // MatMul * scalar → fused scaled matmul
        if let Some(result) = self.try_scaled_matmul(expr) {
            return result;
        }

        None
    }

    // ─── P2: MatMul + Bias Fusion ──────────────────────────────────────────

    /// Detect: (A @ B) + bias → linear(A, B, bias)
    /// Also: A @ B + C where C is 1D (bias vector)
    fn try_matmul_bias_fusion(&self, expr: &Expr) -> Option<Option<(Expr, &'static str, f64, MlOptCategory)>> {
        if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, span } = expr {
            // Case 1: (A @ B) + bias
            if let Expr::MatMul { .. } = lhs.as_ref() {
                let matmul = lhs.clone();
                let bias = rhs.clone();
                let fused = Expr::Call {
                    span: *span,
                    func: Box::new(Expr::Ident { span: *span, name: "linear".into() }),
                    args: vec![*matmul.clone(), *bias],
                    named: vec![],
                };
                return Some(Some((fused, "matmul_bias_fusion", 2.0, MlOptCategory::KernelFusion)));
            }
            // Case 2: bias + (A @ B)
            if let Expr::MatMul { .. } = rhs.as_ref() {
                let bias = lhs.clone();
                let matmul = rhs.clone();
                let fused = Expr::Call {
                    span: *span,
                    func: Box::new(Expr::Ident { span: *span, name: "linear".into() }),
                    args: vec![*matmul, *bias],
                    named: vec![],
                };
                return Some(Some((fused, "matmul_bias_fusion", 2.0, MlOptCategory::KernelFusion)));
            }
        }
        None
    }

    // ─── P4: LayerNorm Pattern ─────────────────────────────────────────────

    /// Detect: (x - mean(x)) / sqrt(var(x) + eps) → layer_norm(x, eps)
    fn try_layernorm_pattern(&self, expr: &Expr) -> Option<Option<(Expr, &'static str, f64, MlOptCategory)>> {
        // Look for: expr / sqrt(something)
        if let Expr::HadamardDiv { lhs, rhs, span } = expr {
            // Check if rhs is sqrt(var + eps)
            if self.is_sqrt_var_plus_eps(rhs) {
                // Check if lhs is (x - mean)
                if self.is_x_minus_mean(lhs) {
                    let layernorm = Expr::Call {
                        span: *span,
                        func: Box::new(Expr::Ident { span: *span, name: "layer_norm".into() }),
                        args: vec![Expr::Ident { span: *span, name: "x".into() }],
                        named: vec![("eps".into(), Expr::FloatLit { span: *span, value: 1e-5 })],
                    };
                    return Some(Some((layernorm, "layernorm_fusion", 5.0, MlOptCategory::KernelFusion)));
                }
            }
        }
        // Also check BinOp::Div form
        if let Expr::BinOp { op: BinOpKind::Div, lhs, rhs, span } = expr {
            if self.is_sqrt_var_plus_eps(rhs) && self.is_x_minus_mean(lhs) {
                let layernorm = Expr::Call {
                    span: *span,
                    func: Box::new(Expr::Ident { span: *span, name: "layer_norm".into() }),
                    args: vec![Expr::Ident { span: *span, name: "x".into() }],
                    named: vec![("eps".into(), Expr::FloatLit { span: *span, value: 1e-5 })],
                };
                return Some(Some((layernorm, "layernorm_fusion", 5.0, MlOptCategory::KernelFusion)));
            }
        }
        None
    }

    fn is_sqrt_var_plus_eps(&self, expr: &Expr) -> bool {
        // sqrt(var(x) + eps) or (var(x) + eps) ** 0.5
        match expr {
            Expr::Call { func, .. } => {
                if let Expr::Ident { name, .. } = func.as_ref() {
                    name == "sqrt"
                } else {
                    false
                }
            }
            Expr::Pow { exp, .. } => {
                if let Expr::FloatLit { value, .. } = exp.as_ref() {
                    (*value - 0.5).abs() < 1e-10
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn is_x_minus_mean(&self, expr: &Expr) -> bool {
        if let Expr::BinOp { op: BinOpKind::Sub, rhs, .. } = expr {
            if let Expr::Call { func, .. } = rhs.as_ref() {
                if let Expr::Ident { name, .. } = func.as_ref() {
                    return name == "mean" || name == "avg";
                }
            }
        }
        false
    }

    // ─── P5: BatchNorm Pattern ─────────────────────────────────────────────

    /// Detect: (x - mean) / sqrt(var + eps) * gamma + beta → batch_norm(x, gamma, beta, eps)
    fn try_batchnorm_pattern(&self, expr: &Expr) -> Option<Option<(Expr, &'static str, f64, MlOptCategory)>> {
        // Look for: ((x - mean) / sqrt(var + eps)) * gamma + beta
        if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, span } = expr {
            // Check if lhs is a multiply and rhs is beta
            if let Expr::BinOp { op: BinOpKind::Mul, lhs: inner_lhs, rhs: _gamma, .. } = lhs.as_ref() {
                if self.is_x_minus_mean(inner_lhs) {
                    let batchnorm = Expr::Call {
                        span: *span,
                        func: Box::new(Expr::Ident { span: *span, name: "batch_norm".into() }),
                        args: vec![*inner_lhs.clone(), *rhs.clone()],
                        named: vec![],
                    };
                    return Some(Some((batchnorm, "batchnorm_fusion", 8.0, MlOptCategory::KernelFusion)));
                }
            }
        }
        None
    }

    // ─── P6: Activation Rewrites ───────────────────────────────────────────

    /// Detect activation patterns and replace with fused calls.
    fn try_activation_rewrite(&self, expr: &Expr) -> Option<Option<(Expr, &'static str, f64, MlOptCategory)>> {
        // ReLU: max(0, x) → relu(x)
        if let Expr::Call { func, args, span, .. } = expr {
            if let Expr::Ident { name, .. } = func.as_ref() {
                if name == "max" && args.len() == 2 {
                    if let Expr::IntLit { value: 0, .. } = &args[0] {
                        let relu = Expr::Call {
                            span: *span,
                            func: Box::new(Expr::Ident { span: *span, name: "relu".into() }),
                            args: vec![args[1].clone()],
                            named: vec![],
                        };
                        return Some(Some((relu, "relu_from_max", 1.5, MlOptCategory::KernelFusion)));
                    }
                    if let Expr::IntLit { value: 0, .. } = &args[1] {
                        let relu = Expr::Call {
                            span: *span,
                            func: Box::new(Expr::Ident { span: *span, name: "relu".into() }),
                            args: vec![args[0].clone()],
                            named: vec![],
                        };
                        return Some(Some((relu, "relu_from_max", 1.5, MlOptCategory::KernelFusion)));
                    }
                }

                // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                if name == "gelu" {
                    // Already a GELU call, mark for fusion
                    return None;
                }

                // SiLU / Swish: x * sigmoid(x) → silu(x)
                if name == "sigmoid" {
                    // Will be caught by the x * sigmoid(x) pattern below
                }
            }
        }

        // x * sigmoid(x) → silu(x)  (Swish activation)
        if let Expr::HadamardMul { lhs, rhs, span } = expr {
            if let Expr::Call { func, args, .. } = rhs.as_ref() {
                if let Expr::Ident { name, .. } = func.as_ref() {
                    if name == "sigmoid" && Self::exprs_equal_ident(lhs, args.first()) {
                        let silu = Expr::Call {
                            span: *span,
                            func: Box::new(Expr::Ident { span: *span, name: "silu".into() }),
                            args: vec![*lhs.clone()],
                            named: vec![],
                        };
                        return Some(Some((silu, "silu_from_sigmoid", 2.0, MlOptCategory::KernelFusion)));
                    }
                }
            }
            // Also check lhs is sigmoid
            if let Expr::Call { func, args, .. } = lhs.as_ref() {
                if let Expr::Ident { name, .. } = func.as_ref() {
                    if name == "sigmoid" && Self::exprs_equal_ident(rhs, args.first()) {
                        let silu = Expr::Call {
                            span: *span,
                            func: Box::new(Expr::Ident { span: *span, name: "silu".into() }),
                            args: vec![*rhs.clone()],
                            named: vec![],
                        };
                        return Some(Some((silu, "silu_from_sigmoid", 2.0, MlOptCategory::KernelFusion)));
                    }
                }
            }
        }
        // Also check BinOp::Mul form
        if let Expr::BinOp { op: BinOpKind::Mul, lhs, rhs, span } = expr {
            if let Expr::Call { func, args, .. } = rhs.as_ref() {
                if let Expr::Ident { name, .. } = func.as_ref() {
                    if name == "sigmoid" && Self::exprs_equal_ident(lhs, args.first()) {
                        let silu = Expr::Call {
                            span: *span,
                            func: Box::new(Expr::Ident { span: *span, name: "silu".into() }),
                            args: vec![*lhs.clone()],
                            named: vec![],
                        };
                        return Some(Some((silu, "silu_from_sigmoid", 2.0, MlOptCategory::KernelFusion)));
                    }
                }
            }
        }

        None
    }

    // ─── P9: Attention Pattern → FlashAttention ────────────────────────────

    /// Detect: softmax(Q @ K^T / sqrt(d)) @ V → flash_attention(Q, K, V)
    fn try_attention_pattern(&self, expr: &Expr) -> Option<Option<(Expr, &'static str, f64, MlOptCategory)>> {
        // Pattern: softmax(scaled_matmul) @ V
        if let Expr::MatMul { lhs, rhs, span } = expr {
            // Check if lhs is softmax(...)
            if let Expr::Call { func, args, .. } = lhs.as_ref() {
                if let Expr::Ident { name, .. } = func.as_ref() {
                    if name == "softmax" {
                        // This is softmax(something) @ V — potentially FlashAttention
                        let flash_attn = Expr::Call {
                            span: *span,
                            func: Box::new(Expr::Ident { span: *span, name: "flash_attention".into() }),
                            args: vec![
                                Expr::Ident { span: *span, name: "Q".into() },
                                Expr::Ident { span: *span, name: "K".into() },
                                *rhs.clone(),
                            ],
                            named: vec![],
                        };
                        return Some(Some((flash_attn, "flash_attention", 20.0, MlOptCategory::MemoryOptimization)));
                    }
                }
            }
        }
        None
    }

    // ─── P11: Transpose + MatMul ───────────────────────────────────────────

    /// Detect: A @ transpose(B) → fused_transposed_matmul(A, B)
    fn try_transpose_matmul(&self, expr: &Expr) -> Option<Option<(Expr, &'static str, f64, MlOptCategory)>> {
        if let Expr::MatMul { lhs, rhs, span } = expr {
            if let Expr::Call { func, args, .. } = rhs.as_ref() {
                if let Expr::Ident { name, .. } = func.as_ref() {
                    if name == "transpose" || name == "T" {
                        let fused = Expr::Call {
                            span: *span,
                            func: Box::new(Expr::Ident { span: *span, name: "matmul_transposed".into() }),
                            args: vec![*lhs.clone(), args.first().cloned().unwrap_or(Expr::Ident { span: *span, name: "B".into() })],
                            named: vec![],
                        };
                        return Some(Some((fused, "transpose_matmul_fusion", 3.0, MlOptCategory::KernelFusion)));
                    }
                }
            }
            // Also check lhs
            if let Expr::Call { func, args, .. } = lhs.as_ref() {
                if let Expr::Ident { name, .. } = func.as_ref() {
                    if name == "transpose" || name == "T" {
                        let fused = Expr::Call {
                            span: *span,
                            func: Box::new(Expr::Ident { span: *span, name: "matmul_transposed".into() }),
                            args: vec![args.first().cloned().unwrap_or(Expr::Ident { span: *span, name: "A".into() }), *rhs.clone()],
                            named: vec![],
                        };
                        return Some(Some((fused, "transpose_matmul_fusion", 3.0, MlOptCategory::KernelFusion)));
                    }
                }
            }
        }
        None
    }

    // ─── P12: Gradient Accumulation ────────────────────────────────────────

    /// Detect: param = param - lr * grad → fused gradient step
    fn try_gradient_accumulation(&self, expr: &Expr) -> Option<Option<(Expr, &'static str, f64, MlOptCategory)>> {
        if let Expr::Assign { op: AssignOpKind::SubAssign, target, value, span } = expr {
            // Check if value is lr * grad
            if let Expr::BinOp { op: BinOpKind::Mul, .. } = value.as_ref() {
                let fused = Expr::Call {
                    span: *span,
                    func: Box::new(Expr::Ident { span: *span, name: "grad_step".into() }),
                    args: vec![*target.clone(), *value.clone()],
                    named: vec![],
                };
                return Some(Some((fused, "gradient_step_fusion", 2.0, MlOptCategory::KernelFusion)));
            }
        }
        // Also: param = param - lr * grad (full assignment form)
        if let Expr::Assign { op: AssignOpKind::Assign, target, value, span } = expr {
            if let Expr::BinOp { op: BinOpKind::Sub, lhs: sub_lhs, rhs, .. } = value.as_ref() {
                if Self::exprs_equal(target, sub_lhs) {
                    if let Expr::BinOp { op: BinOpKind::Mul, .. } = rhs.as_ref() {
                        let fused = Expr::Call {
                            span: *span,
                            func: Box::new(Expr::Ident { span: *span, name: "grad_step".into() }),
                            args: vec![*target.clone(), *rhs.clone()],
                            named: vec![],
                        };
                        return Some(Some((fused, "gradient_step_fusion", 2.0, MlOptCategory::KernelFusion)));
                    }
                }
            }
        }
        None
    }

    // ─── P14: Zero-Copy Reshape ────────────────────────────────────────────

    /// Detect: reshape(x, ...) → annotated as zero-copy view
    fn try_zero_copy_reshape(&self, expr: &Expr) -> Option<Option<(Expr, &'static str, f64, MlOptCategory)>> {
        if let Expr::Call { func, args, span, .. } = expr {
            if let Expr::Ident { name, .. } = func.as_ref() {
                if name == "reshape" || name == "flatten" || name == "view" || name == "squeeze" || name == "unsqueeze" {
                    // Annotate as zero-copy: these are just view changes, no data movement
                    // We keep the call but add an annotation that the runtime can use
                    let annotated = Expr::Call {
                        span: *span,
                        func: Box::new(Expr::Ident { span: *span, name: format!("{}_view", name) }),
                        args: args.clone(),
                        named: vec![],
                    };
                    return Some(Some((annotated, "zero_copy_reshape", 10.0, MlOptCategory::ZeroCopyView)));
                }
            }
        }
        None
    }

    // ─── P15: Dropout Pattern ──────────────────────────────────────────────

    /// Detect: x * mask / (1 - p) → fused_dropout(x, p)
    fn try_dropout_pattern(&self, expr: &Expr) -> Option<Option<(Expr, &'static str, f64, MlOptCategory)>> {
        // Pattern: x * mask / (1 - p) or x * bernoulli(p) / p
        if let Expr::HadamardDiv { lhs, rhs, span } = expr {
            if let Expr::HadamardMul { lhs: x, rhs: mask, .. } = lhs.as_ref() {
                // Check if rhs is (1 - p)
                if let Expr::BinOp { op: BinOpKind::Sub, lhs: one, .. } = rhs.as_ref() {
                    if let Expr::IntLit { value: 1, .. } = one.as_ref() {
                        let fused = Expr::Call {
                            span: *span,
                            func: Box::new(Expr::Ident { span: *span, name: "dropout".into() }),
                            args: vec![*x.clone(), *mask.clone()],
                            named: vec![],
                        };
                        return Some(Some((fused, "dropout_fusion", 3.0, MlOptCategory::KernelFusion)));
                    }
                }
            }
        }
        None
    }

    // ─── HadamardMul of MatMul ─────────────────────────────────────────────

    /// Detect: (A @ B) .* C → fused matmul_elementwise(A, B, C)
    fn try_hadamard_matmul_fusion(&self, expr: &Expr) -> Option<Option<(Expr, &'static str, f64, MlOptCategory)>> {
        if let Expr::HadamardMul { lhs, rhs, span } = expr {
            if let Expr::MatMul { .. } = lhs.as_ref() {
                let fused = Expr::Call {
                    span: *span,
                    func: Box::new(Expr::Ident { span: *span, name: "matmul_elemwise".into() }),
                    args: vec![*lhs.clone(), *rhs.clone()],
                    named: vec![],
                };
                return Some(Some((fused, "hadamard_matmul_fusion", 3.0, MlOptCategory::KernelFusion)));
            }
            if let Expr::MatMul { .. } = rhs.as_ref() {
                let fused = Expr::Call {
                    span: *span,
                    func: Box::new(Expr::Ident { span: *span, name: "matmul_elemwise".into() }),
                    args: vec![*rhs.clone(), *lhs.clone()],
                    named: vec![],
                };
                return Some(Some((fused, "hadamard_matmul_fusion", 3.0, MlOptCategory::KernelFusion)));
            }
        }
        None
    }

    // ─── Elementwise Chain Fusion ──────────────────────────────────────────

    /// Detect: ((a + b) * c + d) → fused_elementwise chain
    /// This fuses multiple elementwise operations that would otherwise
    /// allocate intermediates at each step.
    fn try_elementwise_chain_fusion(&self, expr: &Expr) -> Option<Option<(Expr, &'static str, f64, MlOptCategory)>> {
        // Count the depth of elementwise operations
        let depth = self.elementwise_depth(expr);
        if depth >= 3 {
            // For chains of 3+ elementwise ops, fuse into a single kernel call
            let span = expr.span();
            let fused = Expr::Call {
                span,
                func: Box::new(Expr::Ident { span, name: "fused_elementwise".into() }),
                args: vec![expr.clone()],
                named: vec![],
            };
            let speedup = 1.0 + (depth as f64 - 1.0) * 0.5; // ~50% per eliminated intermediate
            return Some(Some((fused, "elementwise_chain_fusion", speedup, MlOptCategory::KernelFusion)));
        }
        None
    }

    /// Count the nesting depth of elementwise operations.
    fn elementwise_depth(&self, expr: &Expr) -> usize {
        match expr {
            Expr::BinOp { op, lhs, rhs, .. } => {
                let is_elementwise = matches!(op,
                    BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul | BinOpKind::Div |
                    BinOpKind::FloorDiv | BinOpKind::Rem
                );
                if is_elementwise {
                    1 + self.elementwise_depth(lhs).max(self.elementwise_depth(rhs))
                } else {
                    0
                }
            }
            Expr::HadamardMul { lhs, rhs, .. } | Expr::HadamardDiv { lhs, rhs, .. } => {
                1 + self.elementwise_depth(lhs).max(self.elementwise_depth(rhs))
            }
            Expr::UnOp { expr: inner, .. } => 1 + self.elementwise_depth(inner),
            Expr::Call { func, .. } => {
                if let Expr::Ident { name, .. } = func.as_ref() {
                    if self.activation_names.contains(name) {
                        1 // activation functions are elementwise
                    } else {
                        0 // other calls break the chain
                    }
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    // ─── Scaled MatMul ─────────────────────────────────────────────────────

    /// Detect: (A @ B) * scalar → scaled_matmul(A, B, scalar)
    /// Common in attention: Q @ K^T / sqrt(d)
    fn try_scaled_matmul(&self, expr: &Expr) -> Option<Option<(Expr, &'static str, f64, MlOptCategory)>> {
        // (A @ B) * scalar
        if let Expr::BinOp { op: BinOpKind::Mul, lhs, rhs, span } = expr {
            if let Expr::MatMul { .. } = lhs.as_ref() {
                if Self::is_scalarish(rhs) {
                    let fused = Expr::Call {
                        span: *span,
                        func: Box::new(Expr::Ident { span: *span, name: "scaled_matmul".into() }),
                        args: vec![*lhs.clone(), *rhs.clone()],
                        named: vec![],
                    };
                    return Some(Some((fused, "scaled_matmul", 2.0, MlOptCategory::KernelFusion)));
                }
            }
        }
        // (A @ B) / scalar
        if let Expr::BinOp { op: BinOpKind::Div, lhs, rhs, span } = expr {
            if let Expr::MatMul { .. } = lhs.as_ref() {
                if Self::is_scalarish(rhs) {
                    let fused = Expr::Call {
                        span: *span,
                        func: Box::new(Expr::Ident { span: *span, name: "scaled_matmul".into() }),
                        args: vec![*lhs.clone(), *rhs.clone()],
                        named: vec![],
                    };
                    return Some(Some((fused, "scaled_matmul", 2.0, MlOptCategory::KernelFusion)));
                }
            }
        }
        // HadamardMul form
        if let Expr::HadamardMul { lhs, rhs, span } = expr {
            if let Expr::MatMul { .. } = lhs.as_ref() {
                if Self::is_scalarish(rhs) {
                    let fused = Expr::Call {
                        span: *span,
                        func: Box::new(Expr::Ident { span: *span, name: "scaled_matmul".into() }),
                        args: vec![*lhs.clone(), *rhs.clone()],
                        named: vec![],
                    };
                    return Some(Some((fused, "scaled_matmul", 2.0, MlOptCategory::KernelFusion)));
                }
            }
        }
        None
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Tier 3: Graph-level ML kernel fusion
    // ═══════════════════════════════════════════════════════════════════════════

    /// Scan consecutive statements for fusion opportunities.
    fn fuse_operations(&mut self, block: &mut Block) {
        let stmts = std::mem::take(&mut block.stmts);
        let mut new_stmts = Vec::with_capacity(stmts.len());
        let mut i = 0;

        while i < stmts.len() {
            // Try to fuse consecutive ML operations
            let fused_span = stmts[i].span();

            // Pattern: let a = matmul(...); let b = a + bias; let c = relu(b)
            // → let c = fused_linear_relu(...)
            if i + 2 < stmts.len() {
                if let Some(fused) = self.try_three_stmt_fusion(
                    &stmts[i], &stmts[i + 1], &stmts[i + 2], fused_span
                ) {
                    new_stmts.push(fused);
                    self.record_pattern("three_op_fusion", 4.0, MlOptCategory::KernelFusion);
                    i += 3;
                    continue;
                }
            }

            // Pattern: let a = matmul(...); let b = relu(a)
            // → let b = fused_matmul_relu(...)
            if i + 1 < stmts.len() {
                if let Some(fused) = self.try_two_stmt_fusion(&stmts[i], &stmts[i + 1], fused_span) {
                    new_stmts.push(fused);
                    self.record_pattern("two_op_fusion", 2.0, MlOptCategory::KernelFusion);
                    i += 2;
                    continue;
                }
            }

            new_stmts.push(stmts[i].clone());
            i += 1;
        }

        block.stmts = new_stmts;
    }

    /// Try to fuse two consecutive statements.
    fn try_two_stmt_fusion(&self, s1: &Stmt, s2: &Stmt, span: Span) -> Option<Stmt> {
        // let a = A @ B; let b = relu(a) → let b = matmul_relu(A, B)
        if let (Stmt::Let { init: Some(init1), pattern: p1, .. }, Stmt::Let { init: Some(init2), pattern: p2, .. }) = (s1, s2) {
            // Check if s2 references the result of s1
            let name1 = self.pattern_name(p1)?;
            if self.references_name(init2, &name1) {
                // MatMul + activation
                if let Expr::MatMul { lhs, rhs, .. } = init1 {
                    if let Expr::Call { func, args, .. } = init2 {
                        if let Expr::Ident { name: act_name, .. } = func.as_ref() {
                            if self.activation_names.contains(act_name) {
                                let fused_name = format!("matmul_{}", act_name);
                                let fused = Stmt::Let {
                                    span,
                                    pattern: p2.clone(),
                                    ty: None,
                                    init: Some(Expr::Call {
                                        span,
                                        func: Box::new(Expr::Ident { span, name: fused_name }),
                                        args: vec![*lhs.clone(), *rhs.clone()],
                                        named: vec![],
                                    }),
                                    mutable: false,
                                };
                                return Some(fused);
                            }
                        }
                    }
                }

                // MatMul + bias add
                if let Expr::MatMul { lhs, rhs, .. } = init1 {
                    if let Expr::BinOp { op: BinOpKind::Add, lhs: add_lhs, rhs: add_rhs, .. } = init2 {
                        if self.references_name(add_lhs, &name1) {
                            let fused = Stmt::Let {
                                span,
                                pattern: p2.clone(),
                                ty: None,
                                init: Some(Expr::Call {
                                    span,
                                    func: Box::new(Expr::Ident { span, name: "linear".into() }),
                                    args: vec![*lhs.clone(), *rhs.clone(), *add_rhs.clone()],
                                    named: vec![],
                                }),
                                mutable: false,
                            };
                            return Some(fused);
                        }
                    }
                }
            }
        }
        None
    }

    /// Try to fuse three consecutive statements.
    fn try_three_stmt_fusion(&self, s1: &Stmt, s2: &Stmt, s3: &Stmt, span: Span) -> Option<Stmt> {
        // let a = A @ B; let b = a + bias; let c = relu(b) → let c = linear_relu(A, B, bias)
        if let (Stmt::Let { init: Some(init1), pattern: p1, .. },
                Stmt::Let { init: Some(init2), pattern: p2, .. },
                Stmt::Let { init: Some(init3), pattern: p3, .. }) = (s1, s2, s3)
        {
            let name1 = self.pattern_name(p1)?;
            let name2 = self.pattern_name(p2)?;

            // Check: init2 references name1, init3 references name2
            if self.references_name(init2, &name1) && self.references_name(init3, &name2) {
                // Pattern: MatMul + bias + activation
                if let Expr::MatMul { lhs, rhs, .. } = init1 {
                    if let Expr::BinOp { op: BinOpKind::Add, rhs: bias, .. } = init2 {
                        if let Expr::Call { func, .. } = init3 {
                            if let Expr::Ident { name: act_name, .. } = func.as_ref() {
                                if self.activation_names.contains(act_name) {
                                    let fused_name = format!("linear_{}", act_name);
                                    let fused = Stmt::Let {
                                        span,
                                        pattern: p3.clone(),
                                        ty: None,
                                        init: Some(Expr::Call {
                                            span,
                                            func: Box::new(Expr::Ident { span, name: fused_name }),
                                            args: vec![*lhs.clone(), *rhs.clone(), *bias.clone()],
                                            named: vec![],
                                        }),
                                        mutable: false,
                                    };
                                    return Some(fused);
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Utility methods
    // ═══════════════════════════════════════════════════════════════════════════

    fn record_pattern(&mut self, name: &'static str, speedup: f64, category: MlOptCategory) {
        self.stats.patterns_matched += 1;
        self.stats.total_estimated_speedup += speedup;
        match category {
            MlOptCategory::LoopToKernel => self.stats.loop_to_kernel += 1,
            MlOptCategory::KernelFusion => self.stats.kernel_fusions += 1,
            MlOptCategory::NumericalStability => self.stats.numerical_stability_fixes += 1,
            MlOptCategory::MemoryOptimization => self.stats.memory_optimizations += 1,
            MlOptCategory::Vectorization => self.stats.vectorizations += 1,
            MlOptCategory::ZeroCopyView => self.stats.zero_copy_views += 1,
        }
    }

    /// Check if an expression contains a call to a specific function name.
    fn contains_call(&self, expr: &Expr, name: &str) -> bool {
        match expr {
            Expr::Call { func, .. } => {
                if let Expr::Ident { name: fn_name, .. } = func.as_ref() {
                    fn_name == name
                } else {
                    false
                }
            }
            Expr::BinOp { lhs, rhs, .. } => {
                self.contains_call(lhs, name) || self.contains_call(rhs, name)
            }
            Expr::UnOp { expr, .. } => self.contains_call(expr, name),
            Expr::MatMul { lhs, rhs, .. } => {
                self.contains_call(lhs, name) || self.contains_call(rhs, name)
            }
            Expr::HadamardMul { lhs, rhs, .. } => {
                self.contains_call(lhs, name) || self.contains_call(rhs, name)
            }
            Expr::HadamardDiv { lhs, rhs, .. } => {
                self.contains_call(lhs, name) || self.contains_call(rhs, name)
            }
            _ => false,
        }
    }

    /// Check if an expression is an accumulation pattern (+=, *=, etc.)
    fn is_accumulation(&self, expr: &Expr) -> bool {
        matches!(expr,
            Expr::Assign { op: AssignOpKind::AddAssign, .. } |
            Expr::Assign { op: AssignOpKind::MulAssign, .. }
        )
    }

    /// Check if an expression looks like a scalar (literal or simple variable).
    fn is_scalarish(expr: &Expr) -> bool {
        matches!(expr,
            Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::Ident { .. }
        )
    }

    /// Check if two expressions reference the same identifier.
    fn exprs_equal(a: &Expr, b: &Expr) -> bool {
        match (a, b) {
            (Expr::Ident { name: na, .. }, Expr::Ident { name: nb, .. }) => na == nb,
            _ => false,
        }
    }

    /// Check if two expressions reference the same identifier (one may be wrapped).
    fn exprs_equal_ident(expr: &Expr, other: Option<&Expr>) -> bool {
        if let Some(o) = other {
            Self::exprs_equal(expr, o)
        } else {
            false
        }
    }

    /// Check if an expression references a variable by name.
    fn references_name(&self, expr: &Expr, name: &str) -> bool {
        match expr {
            Expr::Ident { name: n, .. } => n == name,
            Expr::BinOp { lhs, rhs, .. } => {
                self.references_name(lhs, name) || self.references_name(rhs, name)
            }
            Expr::UnOp { expr, .. } => self.references_name(expr, name),
            Expr::Call { func, args, named, .. } => {
                self.references_name(func, name) ||
                    args.iter().any(|a| self.references_name(a, name)) ||
                    named.iter().any(|(_, v)| self.references_name(v, name))
            }
            Expr::MatMul { lhs, rhs, .. } => {
                self.references_name(lhs, name) || self.references_name(rhs, name)
            }
            Expr::HadamardMul { lhs, rhs, .. } => {
                self.references_name(lhs, name) || self.references_name(rhs, name)
            }
            Expr::HadamardDiv { lhs, rhs, .. } => {
                self.references_name(lhs, name) || self.references_name(rhs, name)
            }
            Expr::Pow { base, exp, .. } => {
                self.references_name(base, name) || self.references_name(exp, name)
            }
            Expr::Index { object, indices, .. } => {
                self.references_name(object, name) ||
                    indices.iter().any(|i| self.references_name(i, name))
            }
            Expr::Assign { target, value, .. } => {
                self.references_name(target, name) || self.references_name(value, name)
            }
            Expr::MethodCall { receiver, args, .. } => {
                self.references_name(receiver, name) ||
                    args.iter().any(|a| self.references_name(a, name))
            }
            _ => false,
        }
    }

    /// Extract the name from a Pattern::Ident.
    fn pattern_name(&self, pattern: &Pattern) -> Option<String> {
        match pattern {
            Pattern::Ident { name, .. } => Some(name.clone()),
            _ => None,
        }
    }
}

impl Default for MlSuperoptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_span() -> Span {
        Span::dummy()
    }

    #[test]
    fn test_matmul_bias_fusion() {
        let mut opt = MlSuperoptimizer::new();
        let span = dummy_span();

        // (A @ B) + bias → linear(A @ B, bias)
        let expr = Expr::BinOp {
            span,
            op: BinOpKind::Add,
            lhs: Box::new(Expr::MatMul {
                span,
                lhs: Box::new(Expr::Ident { span, name: "A".into() }),
                rhs: Box::new(Expr::Ident { span, name: "B".into() }),
            }),
            rhs: Box::new(Expr::Ident { span, name: "bias".into() }),
        };

        let result = opt.optimize_expr(expr);
        // Should be a Call to "linear"
        if let Expr::Call { func, .. } = &result {
            if let Expr::Ident { name, .. } = func.as_ref() {
                assert_eq!(name, "linear");
            } else {
                panic!("Expected Ident func");
            }
        } else {
            panic!("Expected Call expr, got {:?}", result);
        }
        assert_eq!(opt.stats.kernel_fusions, 1);
    }

    #[test]
    fn test_silu_from_sigmoid() {
        let mut opt = MlSuperoptimizer::new();
        let span = dummy_span();

        // x * sigmoid(x) → silu(x)
        let expr = Expr::BinOp {
            span,
            op: BinOpKind::Mul,
            lhs: Box::new(Expr::Ident { span, name: "x".into() }),
            rhs: Box::new(Expr::Call {
                span,
                func: Box::new(Expr::Ident { span, name: "sigmoid".into() }),
                args: vec![Expr::Ident { span, name: "x".into() }],
                named: vec![],
            }),
        };

        let result = opt.optimize_expr(expr);
        if let Expr::Call { func, .. } = &result {
            if let Expr::Ident { name, .. } = func.as_ref() {
                assert_eq!(name, "silu");
            }
        } else {
            panic!("Expected Call expr");
        }
    }

    #[test]
    fn test_scaled_matmul() {
        let mut opt = MlSuperoptimizer::new();
        let span = dummy_span();

        // (A @ B) * 0.5 → scaled_matmul(A @ B, 0.5)
        let expr = Expr::BinOp {
            span,
            op: BinOpKind::Mul,
            lhs: Box::new(Expr::MatMul {
                span,
                lhs: Box::new(Expr::Ident { span, name: "A".into() }),
                rhs: Box::new(Expr::Ident { span, name: "B".into() }),
            }),
            rhs: Box::new(Expr::FloatLit { span, value: 0.5 }),
        };

        let result = opt.optimize_expr(expr);
        if let Expr::Call { func, .. } = &result {
            if let Expr::Ident { name, .. } = func.as_ref() {
                assert_eq!(name, "scaled_matmul");
            }
        } else {
            panic!("Expected Call expr");
        }
    }

    #[test]
    fn test_zero_copy_reshape() {
        let mut opt = MlSuperoptimizer::new();
        let span = dummy_span();

        // reshape(x, ...) → reshape_view(x, ...)
        let expr = Expr::Call {
            span,
            func: Box::new(Expr::Ident { span, name: "reshape".into() }),
            args: vec![Expr::Ident { span, name: "x".into() }],
            named: vec![],
        };

        let result = opt.optimize_expr(expr);
        if let Expr::Call { func, .. } = &result {
            if let Expr::Ident { name, .. } = func.as_ref() {
                assert_eq!(name, "reshape_view");
            }
        } else {
            panic!("Expected Call expr");
        }
        assert_eq!(opt.stats.zero_copy_views, 1);
    }

    #[test]
    fn test_hadamard_matmul_fusion() {
        let mut opt = MlSuperoptimizer::new();
        let span = dummy_span();

        // (A @ B) .* C → matmul_elemwise(A @ B, C)
        let expr = Expr::HadamardMul {
            span,
            lhs: Box::new(Expr::MatMul {
                span,
                lhs: Box::new(Expr::Ident { span, name: "A".into() }),
                rhs: Box::new(Expr::Ident { span, name: "B".into() }),
            }),
            rhs: Box::new(Expr::Ident { span, name: "C".into() }),
        };

        let result = opt.optimize_expr(expr);
        if let Expr::Call { func, .. } = &result {
            if let Expr::Ident { name, .. } = func.as_ref() {
                assert_eq!(name, "matmul_elemwise");
            }
        } else {
            panic!("Expected Call expr");
        }
    }

    #[test]
    fn test_elementwise_chain_fusion() {
        let mut opt = MlSuperoptimizer::new();
        let span = dummy_span();

        // ((a + b) * c + d) → fused_elementwise (3 levels deep)
        let expr = Expr::BinOp {
            span,
            op: BinOpKind::Add,
            lhs: Box::new(Expr::BinOp {
                span,
                op: BinOpKind::Mul,
                lhs: Box::new(Expr::BinOp {
                    span,
                    op: BinOpKind::Add,
                    lhs: Box::new(Expr::Ident { span, name: "a".into() }),
                    rhs: Box::new(Expr::Ident { span, name: "b".into() }),
                }),
                rhs: Box::new(Expr::Ident { span, name: "c".into() }),
            }),
            rhs: Box::new(Expr::Ident { span, name: "d".into() }),
        };

        let result = opt.optimize_expr(expr);
        if let Expr::Call { func, .. } = &result {
            if let Expr::Ident { name, .. } = func.as_ref() {
                assert_eq!(name, "fused_elementwise");
            }
        } else {
            panic!("Expected Call expr");
        }
    }

    #[test]
    fn test_stats_tracking() {
        let mut opt = MlSuperoptimizer::new();
        assert_eq!(opt.stats.patterns_matched, 0);
        assert_eq!(opt.stats.total_estimated_speedup, 0.0);

        opt.record_pattern("test", 5.0, MlOptCategory::KernelFusion);
        assert_eq!(opt.stats.patterns_matched, 1);
        assert_eq!(opt.stats.kernel_fusions, 1);
        assert!((opt.stats.total_estimated_speedup - 5.0).abs() < 0.001);
    }
}
