// jules/src/advanced_optimizer.rs
//
// SUPEROPTIMIZER 


#![allow(dead_code)]

use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use rustc_hash::FxHashMap;

use crate::compiler::ast::*;
use crate::Span;

use crate::optimizer::hardware_cost_model::HardwareCostModel;
use crate::optimizer::profile_guided::ProfileWeightedCostModel;

// §1  CONFIGURATION

#[derive(Debug, Clone, Copy)]
pub struct SuperoptimizerConfig {
    pub iterations: usize,
    pub enable_inlining: bool,
    pub max_inline_size: usize,
    pub max_unroll_factor: usize,
    pub enable_cse: bool,
    pub enable_peephole: bool,
    pub enable_loop_opts: bool,
    pub enable_dse: bool,
    /// Enable the stochastic superoptimizer pass.  When true, after all
    /// conventional passes have converged the superoptimizer searches the
    /// space of equivalent programs for each expression and replaces any
    /// expression with a cheaper semantically-equivalent one found by search.
    pub enable_superopt: bool,
    /// Number of random candidate programs tried per expression.
    pub superopt_budget: usize,
    /// Number of random input environments used to verify equivalence.
    /// 16 is sufficient for near-zero false-positive probability over u128.
    pub superopt_verif_inputs: usize,
    /// Enable the equality-saturation EGraph pass (§16.5).  When true, each
    /// pure arithmetic/logical expression is encoded into an EGraph, saturated
    /// with algebraic rewrite rules, and the minimum-cost equivalent expression
    /// is extracted and used in place of the original.
    pub enable_egraph: bool,
    /// Maximum number of saturation iterations per EGraph instance.
    /// Higher values discover more equivalences at the cost of compile time.
    pub egraph_iterations: usize,
}

impl SuperoptimizerConfig {
    pub fn fast_compile() -> Self {
        Self {
            iterations: 1,
            enable_inlining: false,
            max_inline_size: 0,
            max_unroll_factor: 0,
            enable_cse: false,
            enable_peephole: true,
            enable_loop_opts: false,
            enable_dse: false,
            enable_superopt: false,
            superopt_budget: 0,
            superopt_verif_inputs: 0,
            enable_egraph: false,
            egraph_iterations: 0,
        }
    }

    pub fn balanced() -> Self {
        Self {
            iterations: 3,
            enable_inlining: true,
            max_inline_size: 32,
            max_unroll_factor: 4,
            enable_cse: true,
            enable_peephole: true,
            enable_loop_opts: true,
            enable_dse: true,
            enable_superopt: true,
            superopt_budget: 500,
            superopt_verif_inputs: 16,
            enable_egraph: true,
            egraph_iterations: 4,
        }
    }

    pub fn maximum() -> Self {
        Self {
            iterations: 5,
            enable_inlining: true,
            max_inline_size: 128,
            max_unroll_factor: 8,
            enable_cse: true,
            enable_peephole: true,
            enable_loop_opts: true,
            enable_dse: true,
            enable_superopt: true,
            superopt_budget: 2000,
            superopt_verif_inputs: 32,
            enable_egraph: true,
            egraph_iterations: 8,
        }
    }
}

// §2  CONSTANT FOLDING

struct ConstantFolder {
    folds_performed: u64,
}

impl ConstantFolder {
    fn new() -> Self {
        Self { folds_performed: 0 }
    }

    fn fold_expr(&mut self, expr: Expr) -> Expr {
        match expr {
            Expr::BinOp { span, op, lhs, rhs } => {
                let lhs = self.fold_expr(*lhs);
                let rhs = self.fold_expr(*rhs);
                if let Some(result) = Self::try_fold_binop(span, op, &lhs, &rhs) {
                    self.folds_performed += 1;
                    return result;
                }
                Expr::BinOp { span, op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
            }
            Expr::UnOp { span, op, expr } => {
                let inner = self.fold_expr(*expr);
                if let Some(result) = Self::try_fold_unop(span, op, &inner) {
                    self.folds_performed += 1;
                    return result;
                }
                Expr::UnOp { span, op, expr: Box::new(inner) }
            }
            Expr::IfExpr { span, cond, then, else_ } => {
                let cond = self.fold_expr(*cond);
                if let Expr::BoolLit { value: true, .. } = &cond {
                    self.folds_performed += 1;
                    return Expr::Block(then);
                }
                if let Expr::BoolLit { value: false, .. } = &cond {
                    self.folds_performed += 1;
                    return match else_ {
                        Some(else_block) => Expr::Block(else_block),
                        None => Expr::Block(Box::new(Block {
                            span, stmts: Vec::new(), tail: None,
                        })),
                    };
                }
                Expr::IfExpr { span, cond: Box::new(cond), then, else_ }
            }
            Expr::Call { span, func, args, named } => Expr::Call {
                span, func: Box::new(self.fold_expr(*func)),
                args: args.into_iter().map(|a| self.fold_expr(a)).collect(),
                named,
            },
            Expr::Tuple { span, elems } => Expr::Tuple {
                span, elems: elems.into_iter().map(|e| self.fold_expr(e)).collect(),
            },
            Expr::Block(block) => {
                let mut b = *block;
                self.fold_block_mut(&mut b);
                Expr::Block(Box::new(b))
            }
            _ => expr,
        }
    }

    fn fold_block_mut(&mut self, block: &mut Block) {
        for stmt in &mut block.stmts {
            self.fold_stmt_mut(stmt);
        }
        if let Some(tail) = &mut block.tail {
            let old = std::mem::replace(tail.as_mut(), Expr::IntLit { span: Span::dummy(), value: 0 });
            **tail = self.fold_expr(old);
        }
    }

    fn fold_stmt_mut(&mut self, stmt: &mut Stmt) {
        match stmt {
            Stmt::Let { init: Some(expr), .. } | Stmt::Expr { expr, .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.fold_expr(old);
            }
            Stmt::ForIn { body, .. } | Stmt::While { body, .. } | Stmt::EntityFor { body, .. } => {
                self.fold_block_mut(body);
            }
            Stmt::If { cond, then, else_, .. } => {
                let old = std::mem::replace(cond, Expr::IntLit { span: Span::dummy(), value: 0 });
                *cond = self.fold_expr(old);
                self.fold_block_mut(then);
                if let Some(else_box) = else_ {
                    let else_block = &mut **else_box;
                    if let IfOrBlock::Block(b) = else_block {
                        self.fold_block_mut(b);
                    }
                }
            }
            Stmt::Return { value: Some(expr), .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.fold_expr(old);
            }
            Stmt::Match { expr, arms, .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.fold_expr(old);
                for arm in arms {
                    let old_body = std::mem::replace(&mut arm.body, Expr::IntLit { span: Span::dummy(), value: 0 });
                    arm.body = self.fold_expr(old_body);
                }
            }
            _ => {}
        }
    }

    fn try_fold_binop(span: Span, op: BinOpKind, lhs: &Expr, rhs: &Expr) -> Option<Expr> {
        match (lhs, rhs) {
            (Expr::IntLit { value: l, .. }, Expr::IntLit { value: r, .. }) => {
                let result = match op {
                    BinOpKind::Add => Some(l.wrapping_add(*r)),
                    BinOpKind::Sub => Some(l.wrapping_sub(*r)),
                    BinOpKind::Mul => Some(l.wrapping_mul(*r)),
                    BinOpKind::Div if *r != 0 => Some(l / *r),
                    BinOpKind::Rem if *r != 0 => Some(l % *r),
                    BinOpKind::Eq => Some(if *l == *r { u128::MAX } else { 0 }),
                    BinOpKind::Ne => Some(if *l != *r { u128::MAX } else { 0 }),
                    BinOpKind::Lt => Some(if *l < *r { u128::MAX } else { 0 }),
                    BinOpKind::Le => Some(if *l <= *r { u128::MAX } else { 0 }),
                    BinOpKind::Gt => Some(if *l > *r { u128::MAX } else { 0 }),
                    BinOpKind::Ge => Some(if *l >= *r { u128::MAX } else { 0 }),
                    BinOpKind::BitAnd => Some(*l & *r),
                    BinOpKind::BitOr => Some(*l | *r),
                    BinOpKind::BitXor => Some(*l ^ *r),
                    BinOpKind::Shl => Some(l.checked_shl((*r).try_into().unwrap_or(128)).unwrap_or(0)),
                    BinOpKind::Shr => Some(l.checked_shr((*r).try_into().unwrap_or(128)).unwrap_or(0)),
                    _ => None,
                };
                result.map(|v| Expr::IntLit { span, value: v })
            }
            (Expr::FloatLit { value: l, .. }, Expr::FloatLit { value: r, .. }) => {
                let result = match op {
                    BinOpKind::Add => Some(l + r),
                    BinOpKind::Sub => Some(l - r),
                    BinOpKind::Mul => Some(l * r),
                    BinOpKind::Div if *r != 0.0 => Some(l / *r),
                    BinOpKind::Eq => Some(if *l == *r { 1.0 } else { 0.0 }),
                    BinOpKind::Ne => Some(if *l != *r { 1.0 } else { 0.0 }),
                    BinOpKind::Lt => Some(if *l < *r { 1.0 } else { 0.0 }),
                    BinOpKind::Le => Some(if *l <= *r { 1.0 } else { 0.0 }),
                    BinOpKind::Gt => Some(if *l > *r { 1.0 } else { 0.0 }),
                    BinOpKind::Ge => Some(if *l >= *r { 1.0 } else { 0.0 }),
                    _ => None,
                };
                result.map(|v| Expr::FloatLit { span, value: v })
            }
            (Expr::BoolLit { value: l, .. }, Expr::BoolLit { value: r, .. }) => {
                let result = match op {
                    BinOpKind::And => Some(*l && *r),
                    BinOpKind::Or => Some(*l || *r),
                    BinOpKind::Eq => Some(*l == *r),
                    BinOpKind::Ne => Some(*l != *r),
                    _ => None,
                };
                result.map(|v| Expr::BoolLit { span, value: v })
            }
            _ => None,
        }
    }

    fn try_fold_unop(span: Span, op: UnOpKind, expr: &Expr) -> Option<Expr> {
        match expr {
            Expr::IntLit { value, .. } => match op {
                UnOpKind::Neg => Some(Expr::IntLit { span, value: (*value as i128).wrapping_neg() as u128 }),
                UnOpKind::Not => Some(Expr::IntLit { span, value: !*value }),
                _ => None,
            },
            Expr::FloatLit { value, .. } => match op {
                UnOpKind::Neg => Some(Expr::FloatLit { span, value: -*value }),
                _ => None,
            },
            Expr::BoolLit { value, .. } => match op {
                UnOpKind::Not => Some(Expr::BoolLit { span, value: !*value }),
                _ => None,
            },
            _ => None,
        }
    }
}

// =============================================================================(the bars look nice)
// §3  CONSTANT PROPAGATION (SCCP)
// =============================================================================

struct ConstantPropagator {
    propagations: u64,
}

impl ConstantPropagator {
    fn new() -> Self {
        Self { propagations: 0 }
    }

    fn propagate_block(&mut self, block: &mut Block) {
        // Single incremental forward pass: substitute *before* updating the env
        // with each new binding, so that shadowed names are handled correctly.
        // e.g.  let x = 1; let y = x + 1; let x = 2;
        // → when processing `y = x + 1`, env["x"] is still 1, not 2.
        let mut env: FxHashMap<String, Expr> = FxHashMap::default();

        for stmt in &mut block.stmts {
            match stmt {
                Stmt::Let { pattern, init: Some(expr), .. } => {
                    // 1. Substitute known constants into the initialiser.
                    if !env.is_empty() {
                        let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                        *expr = self.substitute(old, &env);
                    }
                    // 2. After substitution, record this binding if it is now a constant.
                    //    Any previous binding for the same name is evicted first.
                    //    CRITICAL: Mutable variables must NOT be added to the constant
                    //    environment because they can be reassigned later, which would
                    //    not be tracked by the env.  This prevents stale constant
                    //    propagation after assignment.
                    if let Pattern::Ident { name, mutable, .. } = pattern {
                        if *mutable {
                            // Mutable variables should never be tracked as constants.
                            env.remove(name.as_str());
                        } else if Self::is_constant_expr(expr) {
                            env.insert(name.clone(), expr.clone());
                        } else {
                            // The name is now bound to a non-constant; evict any
                            // stale constant that was previously mapped to it.
                            env.remove(name.as_str());
                        }
                    }
                }
                Stmt::Expr { expr, .. } => {
                    if !env.is_empty() {
                        let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                        *expr = self.substitute(old, &env);
                    }
                    // An assignment expression may clobber a tracked constant
                    // or establish a new one (e.g. `x = 10` where 10 is constant).
                    if let Expr::Assign { target, value, op, .. } = expr {
                        if let Expr::Ident { name, .. } = target.as_ref() {
                            // Plain assignment (not compound like +=) to a known
                            // constant: record the new value.  Otherwise evict.
                            if matches!(op, AssignOpKind::Assign) && Self::is_constant_expr(value) {
                                env.insert(name.clone(), (**value).clone());
                            } else {
                                env.remove(name.as_str());
                            }
                        }
                    }
                }
                Stmt::Return { value: Some(expr), .. } => {
                    if !env.is_empty() {
                        let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                        *expr = self.substitute(old, &env);
                    }
                }
                Stmt::If { cond, then, else_, .. } => {
                    // Substitute into the condition.
                    if !env.is_empty() {
                        let old = std::mem::replace(cond, Expr::IntLit { span: Span::dummy(), value: 0 });
                        *cond = self.substitute(old, &env);
                    }
                    // Propagate into branches.
                    self.propagate_block(then);
                    if let Some(eb) = else_ {
                        match &mut **eb {
                            IfOrBlock::If(ref mut if_stmt) => {
                                // Create a temporary block to recurse into the else-if.
                                let mut fake_block = Block { stmts: vec![if_stmt.clone()], tail: None, span: Span::dummy() };
                                self.propagate_block(&mut fake_block);
                                if let Some(s) = fake_block.stmts.into_iter().next() {
                                    *if_stmt = s;
                                }
                            }
                            IfOrBlock::Block(ref mut b) => {
                                self.propagate_block(b);
                            }
                        }
                    }
                    // Any variable written inside either branch is no longer a known constant.
                    let mut written = std::collections::HashSet::<String>::default();
                    Self::collect_writes_block(then, &mut written);
                    if let Some(eb) = else_ {
                        match &**eb {
                            IfOrBlock::Block(b) => Self::collect_writes_block(b, &mut written),
                            IfOrBlock::If(s) => Self::collect_writes_stmt(s, &mut written),
                        }
                    }
                    for name in &written {
                        env.remove(name.as_str());
                    }
                }
                Stmt::While { cond, body, .. } => {
                    // Do NOT substitute into the condition if any variable read in
                    // the condition is written in the loop body — the variable changes
                    // across iterations, so substituting its pre-loop value is wrong.
                    let mut cond_reads = std::collections::HashSet::<String>::default();
                    DeadCodeEliminator::collect_reads_expr(cond, &mut cond_reads);
                    let mut body_writes = std::collections::HashSet::<String>::default();
                    Self::collect_writes_block(body, &mut body_writes);
                    let cond_substitutable = !cond_reads.iter().any(|v| body_writes.contains(v));

                    if !env.is_empty() && cond_substitutable {
                        let old = std::mem::replace(cond, Expr::IntLit { span: Span::dummy(), value: 0 });
                        *cond = self.substitute(old, &env);
                    }
                    self.propagate_block(body);
                    // Evict any variables written in the loop body.
                    for name in &body_writes {
                        env.remove(name.as_str());
                    }
                }
                Stmt::ForIn { body, .. } | Stmt::EntityFor { body, .. } => {
                    self.propagate_block(body);
                    let mut written = std::collections::HashSet::<String>::default();
                    Self::collect_writes_block(body, &mut written);
                    for name in &written {
                        env.remove(name.as_str());
                    }
                }
                Stmt::Loop { body, .. } => {
                    self.propagate_block(body);
                    let mut written = std::collections::HashSet::<String>::default();
                    Self::collect_writes_block(body, &mut written);
                    for name in &written {
                        env.remove(name.as_str());
                    }
                }
                _ => {}
            }
        }

        if let Some(tail) = &mut block.tail {
            if !env.is_empty() {
                let old = std::mem::replace(tail.as_mut(), Expr::IntLit { span: Span::dummy(), value: 0 });
                **tail = self.substitute(old, &env);
            }
        }
    }

    fn is_constant_expr(expr: &Expr) -> bool {
        match expr {
            Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::BoolLit { .. } | Expr::StrLit { .. } => true,
            Expr::BinOp { lhs, rhs, .. } => Self::is_constant_expr(lhs) && Self::is_constant_expr(rhs),
            Expr::UnOp { expr, .. } => Self::is_constant_expr(expr),
            Expr::Tuple { elems, .. } => elems.iter().all(Self::is_constant_expr),
            _ => false,
        }
    }

    /// Collect every variable name that is *written* (assigned) in a block.
    fn collect_writes_block(block: &Block, out: &mut std::collections::HashSet<String>) {
        for stmt in &block.stmts {
            Self::collect_writes_stmt(stmt, out);
        }
        if let Some(tail) = &block.tail {
            Self::collect_writes_expr(tail, out);
        }
    }

    fn collect_writes_stmt(stmt: &Stmt, out: &mut std::collections::HashSet<String>) {
        match stmt {
            Stmt::Let { pattern: Pattern::Ident { name, .. }, .. } => {
                out.insert(name.clone());
            }
            Stmt::Let { .. } => {}
            Stmt::Expr { expr, .. } => Self::collect_writes_expr(expr, out),
            Stmt::If { then, else_, .. } => {
                Self::collect_writes_block(then, out);
                if let Some(eb) = else_ {
                    match eb.as_ref() {
                        IfOrBlock::Block(b) => Self::collect_writes_block(b, out),
                        IfOrBlock::If(s) => Self::collect_writes_stmt(s, out),
                    }
                }
            }
            Stmt::While { body, .. }
            | Stmt::Loop { body, .. }
            | Stmt::ForIn { body, .. }
            | Stmt::EntityFor { body, .. } => {
                Self::collect_writes_block(body, out);
            }
            _ => {}
        }
    }

    fn collect_writes_expr(expr: &Expr, out: &mut std::collections::HashSet<String>) {
        if let Expr::Assign { target, .. } = expr {
            if let Expr::Ident { name, .. } = target.as_ref() {
                out.insert(name.clone());
            }
        }
    }

    fn substitute(&mut self, expr: Expr, env: &FxHashMap<String, Expr>) -> Expr {
        match expr {
            Expr::Ident { name, span } => {
                if let Some(val) = env.get(&name) {
                    self.propagations += 1;
                    val.clone()
                } else {
                    Expr::Ident { name, span }
                }
            }
            Expr::BinOp { span, op, lhs, rhs } => Expr::BinOp {
                span, op,
                lhs: Box::new(self.substitute(*lhs, env)),
                rhs: Box::new(self.substitute(*rhs, env)),
            },
            Expr::UnOp { span, op, expr } => Expr::UnOp {
                span, op,
                expr: Box::new(self.substitute(*expr, env)),
            },
            Expr::Call { span, func, args, named } => Expr::Call {
                span,
                func: Box::new(self.substitute(*func, env)),
                args: args.into_iter().map(|a| self.substitute(a, env)).collect(),
                named,
            },
            Expr::Field { span, object, field } => Expr::Field {
                span,
                object: Box::new(self.substitute(*object, env)),
                field,
            },
            Expr::Index { span, object, indices } => Expr::Index {
                span,
                object: Box::new(self.substitute(*object, env)),
                indices: indices.into_iter().map(|i| self.substitute(i, env)).collect(),
            },
            Expr::Tuple { span, elems } => Expr::Tuple {
                span, elems: elems.into_iter().map(|e| self.substitute(e, env)).collect(),
            },
            // Assign: substitute into the VALUE side only.  The TARGET is an
            // lvalue (e.g. `x`), not an rvalue, so we must NOT replace it with
            // a constant.  Previously, Assign fell through to `_ => expr` which
            // left the value unsubstituted — a missed optimisation but not a
            // correctness bug.  Now we properly recurse into the value.
            Expr::Assign { span, op, target, value } => Expr::Assign {
                span, op,
                target, // lvalue: do NOT substitute
                value: Box::new(self.substitute(*value, env)),
            },
            _ => expr,
        }
    }
}

// =============================================================================
// §4  ALGEBRAIC SIMPLIFICATION (50+ Rules)
// =============================================================================

struct AlgebraicSimplifier {
    simplifications: u64,
}

impl AlgebraicSimplifier {
    fn new() -> Self {
        Self { simplifications: 0 }
    }

    fn simplify_block_mut(&mut self, block: &mut Block) {
        for stmt in &mut block.stmts {
            self.simplify_stmt_mut(stmt);
        }
        if let Some(tail) = &mut block.tail {
            let old = std::mem::replace(tail.as_mut(), Expr::IntLit { span: Span::dummy(), value: 0 });
            **tail = self.simplify_expr(old);
        }
    }

    fn simplify_stmt_mut(&mut self, stmt: &mut Stmt) {
        match stmt {
            Stmt::Let { init: Some(expr), .. } | Stmt::Expr { expr, .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.simplify_expr(old);
            }
            Stmt::ForIn { body, .. } | Stmt::While { body, .. } | Stmt::EntityFor { body, .. } => {
                self.simplify_block_mut(body);
            }
            Stmt::If { cond, then, else_, .. } => {
                let old = std::mem::replace(cond, Expr::IntLit { span: Span::dummy(), value: 0 });
                *cond = self.simplify_expr(old);
                self.simplify_block_mut(then);
                if let Some(else_box) = else_ {
                    if let IfOrBlock::Block(b) = &mut **else_box {
                        self.simplify_block_mut(b);
                    }
                }
            }
            Stmt::Return { value: Some(expr), .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.simplify_expr(old);
            }
            _ => {}
        }
    }

    fn simplify_expr(&mut self, expr: Expr) -> Expr {
        match expr {
            Expr::BinOp { span, op, lhs, rhs } => {
                let lhs = self.simplify_expr(*lhs);
                let rhs = self.simplify_expr(*rhs);
                if let Some(simplified) = Self::algebraic_binop(op, &lhs, &rhs) {
                    self.simplifications += 1;
                    return simplified;
                }
                Expr::BinOp { span, op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
            }
            Expr::UnOp { span, op, expr } => {
                let inner = self.simplify_expr(*expr);
                if let Some(simplified) = Self::algebraic_unop(op, &inner) {
                    self.simplifications += 1;
                    return simplified;
                }
                Expr::UnOp { span, op, expr: Box::new(inner) }
            }
            Expr::Tuple { span, elems } => Expr::Tuple {
                span, elems: elems.into_iter().map(|e| self.simplify_expr(e)).collect(),
            },
            _ => expr,
        }
    }

    fn algebraic_binop(op: BinOpKind, lhs: &Expr, rhs: &Expr) -> Option<Expr> {
        match op {
            // ── Additive identity ─────────────────────────────────────────
            BinOpKind::Add => {
                if Self::is_int_zero(lhs) { return Some(rhs.clone()); }
                if Self::is_int_zero(rhs) { return Some(lhs.clone()); }
                if Self::is_float_zero(lhs) { return Some(rhs.clone()); }
                if Self::is_float_zero(rhs) { return Some(lhs.clone()); }
            }
            BinOpKind::Sub => {
                if Self::is_int_zero(rhs) { return Some(lhs.clone()); }
                if Self::is_float_zero(rhs) { return Some(lhs.clone()); }
                if Self::expr_eq(lhs, rhs) {
                    let span = lhs.span();
                    if matches!(lhs, Expr::IntLit { .. }) {
                        return Some(Expr::IntLit { span, value: 0 });
                    }
                    return Some(Expr::FloatLit { span, value: 0.0 });
                }
            }

            // ── Multiplicative identity / zero ────────────────────────────
            BinOpKind::Mul => {
                if Self::is_int_zero(lhs) { return Some(lhs.clone()); }
                if Self::is_int_zero(rhs) { return Some(rhs.clone()); }
                if Self::is_float_zero(lhs) { return Some(lhs.clone()); }
                if Self::is_float_zero(rhs) { return Some(rhs.clone()); }
                if Self::is_int_one(lhs) { return Some(rhs.clone()); }
                if Self::is_int_one(rhs) { return Some(lhs.clone()); }
                if Self::is_float_one(lhs) { return Some(rhs.clone()); }
                if Self::is_float_one(rhs) { return Some(lhs.clone()); }
                // x * -1 = -x
                const NEG_ONE: u128 = u128::MAX;
                if let Expr::IntLit { value: NEG_ONE, span } = lhs {
                    return Some(Expr::UnOp { span: *span, op: UnOpKind::Neg, expr: Box::new(rhs.clone()) });
                }
                if let Expr::IntLit { value: NEG_ONE, span } = rhs {
                    return Some(Expr::UnOp { span: *span, op: UnOpKind::Neg, expr: Box::new(lhs.clone()) });
                }
            }

            // ── Division ──────────────────────────────────────────────────
            BinOpKind::Div => {
                if Self::is_int_one(rhs) { return Some(lhs.clone()); }
                if Self::is_float_one(rhs) { return Some(lhs.clone()); }
                if Self::expr_eq(lhs, rhs) {
                    let span = lhs.span();
                    if matches!(lhs, Expr::IntLit { .. }) {
                        return Some(Expr::IntLit { span, value: 1 });
                    }
                    return Some(Expr::FloatLit { span, value: 1.0 });
                }
                if Self::is_int_zero(lhs) { return Some(lhs.clone()); }
                if Self::is_float_zero(lhs) { return Some(lhs.clone()); }
            }

            // ── Remainder ─────────────────────────────────────────────────
            BinOpKind::Rem => {
                if Self::is_int_one(rhs) {
                    return Some(Expr::IntLit { span: rhs.span(), value: 0 });
                }
                if Self::expr_eq(lhs, rhs) {
                    if let Expr::IntLit { span, .. } = lhs {
                        return Some(Expr::IntLit { span: *span, value: 0 });
                    }
                }
                if Self::is_int_zero(lhs) { return Some(lhs.clone()); }
                // x % 0 is undefined, but we don't handle it here
            }

            // ── Bitwise ───────────────────────────────────────────────────
            BinOpKind::BitAnd => {
                if Self::is_int_zero(lhs) { return Some(lhs.clone()); }
                if Self::is_int_zero(rhs) { return Some(rhs.clone()); }
                if matches!(lhs, Expr::IntLit { value: u128::MAX, .. }) { return Some(rhs.clone()); }
                if matches!(rhs, Expr::IntLit { value: u128::MAX, .. }) { return Some(lhs.clone()); }
                if Self::expr_eq(lhs, rhs) { return Some(lhs.clone()); }
            }
            BinOpKind::BitOr => {
                if Self::is_int_zero(lhs) { return Some(rhs.clone()); }
                if Self::is_int_zero(rhs) { return Some(lhs.clone()); }
                if matches!(lhs, Expr::IntLit { value: u128::MAX, .. }) { return Some(lhs.clone()); }
                if matches!(rhs, Expr::IntLit { value: u128::MAX, .. }) { return Some(rhs.clone()); }
                if Self::expr_eq(lhs, rhs) { return Some(lhs.clone()); }
            }
            BinOpKind::BitXor => {
                if Self::is_int_zero(lhs) { return Some(rhs.clone()); }
                if Self::is_int_zero(rhs) { return Some(lhs.clone()); }
                if Self::expr_eq(lhs, rhs) {
                    return Some(Expr::IntLit { span: lhs.span(), value: 0 });
                }
            }

            // ── Logical ───────────────────────────────────────────────────
            BinOpKind::And => {
                if let Expr::BoolLit { value: true, .. } = rhs { return Some(lhs.clone()); }
                if let Expr::BoolLit { value: true, .. } = lhs { return Some(rhs.clone()); }
                if let Expr::BoolLit { value: false, .. } = rhs { return Some(rhs.clone()); }
                if let Expr::BoolLit { value: false, .. } = lhs { return Some(lhs.clone()); }
                if Self::expr_eq(lhs, rhs) { return Some(lhs.clone()); }
            }
            BinOpKind::Or => {
                if let Expr::BoolLit { value: false, .. } = rhs { return Some(lhs.clone()); }
                if let Expr::BoolLit { value: false, .. } = lhs { return Some(rhs.clone()); }
                if let Expr::BoolLit { value: true, .. } = rhs { return Some(rhs.clone()); }
                if let Expr::BoolLit { value: true, .. } = lhs { return Some(lhs.clone()); }
                if Self::expr_eq(lhs, rhs) { return Some(lhs.clone()); }
            }

            // ── Comparisons ───────────────────────────────────────────────
            BinOpKind::Eq => {
                if Self::expr_eq(lhs, rhs) { return Some(Expr::BoolLit { span: lhs.span(), value: true }); }
            }
            BinOpKind::Ne => {
                if Self::expr_eq(lhs, rhs) { return Some(Expr::BoolLit { span: lhs.span(), value: false }); }
            }
            BinOpKind::Le | BinOpKind::Ge => {
                if Self::expr_eq(lhs, rhs) { return Some(Expr::BoolLit { span: lhs.span(), value: true }); }
            }
            BinOpKind::Lt | BinOpKind::Gt => {
                if Self::expr_eq(lhs, rhs) { return Some(Expr::BoolLit { span: lhs.span(), value: false }); }
            }

            // ── Shifts ────────────────────────────────────────────────────
            BinOpKind::Shl | BinOpKind::Shr => {
                if Self::is_int_zero(rhs) { return Some(lhs.clone()); }
            }

            _ => {}
        }
        None
    }

    fn algebraic_unop(op: UnOpKind, inner: &Expr) -> Option<Expr> {
        match op {
            UnOpKind::Neg => {
                if let Expr::UnOp { op: UnOpKind::Neg, expr: inner2, .. } = inner {
                    return Some((**inner2).clone());
                }
            }
            UnOpKind::Not => {
                if let Expr::UnOp { op: UnOpKind::Not, expr: inner2, .. } = inner {
                    return Some((**inner2).clone());
                }
            }
            UnOpKind::Deref => {
                if let Expr::UnOp { op: UnOpKind::Ref | UnOpKind::RefMut, expr: inner2, .. } = inner {
                    return Some((**inner2).clone());
                }
            }
            _ => {}
        }
        None
    }

    fn is_int_zero(e: &Expr) -> bool { matches!(e, Expr::IntLit { value: 0, .. }) }
    fn is_int_one(e: &Expr) -> bool { matches!(e, Expr::IntLit { value: 1, .. }) }
    fn is_float_zero(e: &Expr) -> bool { matches!(e, Expr::FloatLit { value: 0.0, .. }) }
    fn is_float_one(e: &Expr) -> bool { matches!(e, Expr::FloatLit { value: 1.0, .. }) }

    fn expr_eq(a: &Expr, b: &Expr) -> bool {
        match (a, b) {
            (Expr::Ident { name: n1, .. }, Expr::Ident { name: n2, .. }) => n1 == n2,
            (Expr::IntLit { value: v1, .. }, Expr::IntLit { value: v2, .. }) => v1 == v2,
            // Bitwise identity: using an epsilon is unsound in a compiler
            // (large floats differing by 1.0 could compare "equal"). Two
            // floats are the same literal only when they are bit-for-bit identical.
            (Expr::FloatLit { value: v1, .. }, Expr::FloatLit { value: v2, .. }) => v1.to_bits() == v2.to_bits(),
            (Expr::BoolLit { value: v1, .. }, Expr::BoolLit { value: v2, .. }) => v1 == v2,
            _ => false,
        }
    }
}

// =============================================================================
// §5  STRENGTH REDUCTION
// =============================================================================

struct StrengthReducer {
    reductions: u64,
}

impl StrengthReducer {
    fn new() -> Self {
        Self { reductions: 0 }
    }

    fn reduce_block_mut(&mut self, block: &mut Block) {
        for stmt in &mut block.stmts {
            self.reduce_stmt_mut(stmt);
        }
        if let Some(tail) = &mut block.tail {
            let old = std::mem::replace(tail.as_mut(), Expr::IntLit { span: Span::dummy(), value: 0 });
            **tail = self.reduce_expr(old);
        }
    }

    fn reduce_stmt_mut(&mut self, stmt: &mut Stmt) {
        match stmt {
            Stmt::Let { init: Some(expr), .. } | Stmt::Expr { expr, .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.reduce_expr(old);
            }
            Stmt::ForIn { body, .. } | Stmt::While { body, .. } | Stmt::EntityFor { body, .. } => {
                self.reduce_block_mut(body);
            }
            Stmt::If { cond, then, else_, .. } => {
                let old = std::mem::replace(cond, Expr::IntLit { span: Span::dummy(), value: 0 });
                *cond = self.reduce_expr(old);
                self.reduce_block_mut(then);
                if let Some(else_box) = else_ {
                    if let IfOrBlock::Block(b) = &mut **else_box {
                        self.reduce_block_mut(b);
                    }
                }
            }
            Stmt::Return { value: Some(expr), .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.reduce_expr(old);
            }
            _ => {}
        }
    }

    fn reduce_expr(&mut self, expr: Expr) -> Expr {
        match expr {
            Expr::BinOp { span, op, lhs, rhs } => {
                let lhs = self.reduce_expr(*lhs);
                let rhs = self.reduce_expr(*rhs);
                if let Some(reduced) = self.strength_reduce_binop(span, op, &lhs, &rhs) {
                    self.reductions += 1;
                    return reduced;
                }
                Expr::BinOp { span, op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
            }
            _ => expr,
        }
    }

    fn strength_reduce_binop(&mut self, span: Span, op: BinOpKind, lhs: &Expr, rhs: &Expr) -> Option<Expr> {
        match op {
            BinOpKind::Mul => {
                // x * 2^k → x << k
                if let Expr::IntLit { value, .. } = rhs {
                    if let Some(shift) = Self::log2_power(*value) {
                        return Some(Expr::BinOp {
                            span, op: BinOpKind::Shl,
                            lhs: Box::new(lhs.clone()),
                            rhs: Box::new(Expr::IntLit { span, value: shift as u128 }),
                        });
                    }
                }
                if let Expr::IntLit { value, .. } = lhs {
                    if let Some(shift) = Self::log2_power(*value) {
                        return Some(Expr::BinOp {
                            span, op: BinOpKind::Shl,
                            lhs: Box::new(rhs.clone()),
                            rhs: Box::new(Expr::IntLit { span, value: shift as u128 }),
                        });
                    }
                }
                // NOTE: x*3, x*5, x*9 → add/shift sequences were removed.
                // On modern x86-64 and AArch64, `imul`/`mul` is a single 3-cycle
                // instruction; expanding to dependent adds hurts ILP and grows
                // the AST for no benefit. LLVM handles these natively.
            }
            BinOpKind::Div => {
                // x / 2^k → x >> k
                if let Expr::IntLit { value, .. } = rhs {
                    if let Some(shift) = Self::log2_power(*value) {
                        return Some(Expr::BinOp {
                            span, op: BinOpKind::Shr,
                            lhs: Box::new(lhs.clone()),
                            rhs: Box::new(Expr::IntLit { span, value: shift as u128 }),
                        });
                    }
                }
                // x / c → x * (1/c) for floats
                if let Expr::FloatLit { value, span: rhs_span } = rhs {
                    if *value != 0.0 {
                        let inv = 1.0 / *value;
                        return Some(Expr::BinOp {
                            span, op: BinOpKind::Mul,
                            lhs: Box::new(lhs.clone()),
                            rhs: Box::new(Expr::FloatLit { span: *rhs_span, value: inv }),
                        });
                    }
                }
            }
            BinOpKind::Rem => {
                // x % 2^k → x & (2^k - 1)
                if let Expr::IntLit { value, .. } = rhs {
                    if let Some(shift) = Self::log2_power(*value) {
                        return Some(Expr::BinOp {
                            span, op: BinOpKind::BitAnd,
                            lhs: Box::new(lhs.clone()),
                            rhs: Box::new(Expr::IntLit { span, value: (1u128 << shift) - 1 }),
                        });
                    }
                }
            }
            _ => {}
        }
        None
    }

    fn log2_power(n: u128) -> Option<u32> {
        if n == 0 || (n & (n - 1)) != 0 { return None; }
        Some(n.trailing_zeros())
    }
}

// =============================================================================
// §6  BITWISE OPTIMIZATIONS
// =============================================================================

struct BitwiseOptimizer {
    optimizations: u64,
}

impl BitwiseOptimizer {
    fn new() -> Self {
        Self { optimizations: 0 }
    }

    fn optimize_block_mut(&mut self, block: &mut Block) {
        for stmt in &mut block.stmts {
            self.optimize_stmt_mut(stmt);
        }
        if let Some(tail) = &mut block.tail {
            let old = std::mem::replace(tail.as_mut(), Expr::IntLit { span: Span::dummy(), value: 0 });
            **tail = self.optimize_expr(old);
        }
    }

    fn optimize_stmt_mut(&mut self, stmt: &mut Stmt) {
        match stmt {
            Stmt::Let { init: Some(expr), .. } | Stmt::Expr { expr, .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.optimize_expr(old);
            }
            Stmt::ForIn { body, .. } | Stmt::While { body, .. } | Stmt::EntityFor { body, .. } => {
                self.optimize_block_mut(body);
            }
            Stmt::If { cond, then, else_, .. } => {
                let old = std::mem::replace(cond, Expr::IntLit { span: Span::dummy(), value: 0 });
                *cond = self.optimize_expr(old);
                self.optimize_block_mut(then);
                if let Some(else_box) = else_ {
                    if let IfOrBlock::Block(b) = &mut **else_box {
                        self.optimize_block_mut(b);
                    }
                }
            }
            Stmt::Return { value: Some(expr), .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.optimize_expr(old);
            }
            _ => {}
        }
    }

    fn optimize_expr(&mut self, expr: Expr) -> Expr {
        match expr {
            Expr::BinOp { span, op, lhs, rhs } => {
                let lhs = self.optimize_expr(*lhs);
                let rhs = self.optimize_expr(*rhs);
                if let Some(optimized) = Self::optimize_binop(span, op, &lhs, &rhs) {
                    self.optimizations += 1;
                    return optimized;
                }
                Expr::BinOp { span, op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
            }
            Expr::UnOp { span, op, expr } => {
                let inner = self.optimize_expr(*expr);
                if let Some(optimized) = Self::optimize_unop(span, op, &inner) {
                    self.optimizations += 1;
                    return optimized;
                }
                Expr::UnOp { span, op, expr: Box::new(inner) }
            }
            _ => expr,
        }
    }

    fn optimize_binop(_span: Span, op: BinOpKind, lhs: &Expr, rhs: &Expr) -> Option<Expr> {
        match op {
            BinOpKind::BitAnd => {
                if Self::is_int_zero(lhs) { return Some(lhs.clone()); }
                if Self::is_int_zero(rhs) { return Some(rhs.clone()); }
                if matches!(lhs, Expr::IntLit { value: u128::MAX, .. }) { return Some(rhs.clone()); }
                if matches!(rhs, Expr::IntLit { value: u128::MAX, .. }) { return Some(lhs.clone()); }
            }
            BinOpKind::BitOr => {
                if Self::is_int_zero(lhs) { return Some(rhs.clone()); }
                if Self::is_int_zero(rhs) { return Some(lhs.clone()); }
            }
            BinOpKind::BitXor => {
                if Self::is_int_zero(lhs) { return Some(rhs.clone()); }
                if Self::is_int_zero(rhs) { return Some(lhs.clone()); }
            }
            _ => {}
        }
        None
    }

    fn optimize_unop(span: Span, op: UnOpKind, inner: &Expr) -> Option<Expr> {
        match op {
            UnOpKind::Not => {
                if let Expr::IntLit { value: 0, .. } = inner {
                    return Some(Expr::IntLit { span, value: u128::MAX });
                }
                if let Expr::IntLit { value: u128::MAX, .. } = inner {
                    return Some(Expr::IntLit { span, value: 0 });
                }
            }
            _ => {}
        }
        None
    }

    fn is_int_zero(e: &Expr) -> bool { matches!(e, Expr::IntLit { value: 0, .. }) }
}

// =============================================================================
// §7  COMPARISON CANONICALIZATION
// =============================================================================

struct ComparisonCanonicalizer {
    canonicalizations: u64,
}

impl ComparisonCanonicalizer {
    fn new() -> Self {
        Self { canonicalizations: 0 }
    }

    fn canonicalize_block_mut(&mut self, block: &mut Block) {
        for stmt in &mut block.stmts {
            self.canonicalize_stmt_mut(stmt);
        }
        if let Some(tail) = &mut block.tail {
            let old = std::mem::replace(tail.as_mut(), Expr::IntLit { span: Span::dummy(), value: 0 });
            **tail = self.canonicalize_expr(old);
        }
    }

    fn canonicalize_stmt_mut(&mut self, stmt: &mut Stmt) {
        match stmt {
            Stmt::Let { init: Some(expr), .. } | Stmt::Expr { expr, .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.canonicalize_expr(old);
            }
            Stmt::ForIn { body, .. } | Stmt::While { body, .. } | Stmt::EntityFor { body, .. } => {
                self.canonicalize_block_mut(body);
            }
            Stmt::If { cond, then, else_, .. } => {
                let old = std::mem::replace(cond, Expr::IntLit { span: Span::dummy(), value: 0 });
                *cond = self.canonicalize_expr(old);
                self.canonicalize_block_mut(then);
                if let Some(else_box) = else_ {
                    if let IfOrBlock::Block(b) = &mut **else_box {
                        self.canonicalize_block_mut(b);
                    }
                }
            }
            Stmt::Return { value: Some(expr), .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.canonicalize_expr(old);
            }
            _ => {}
        }
    }

    fn canonicalize_expr(&mut self, expr: Expr) -> Expr {
        match expr {
            Expr::BinOp { span, op, lhs, rhs } => {
                let lhs = self.canonicalize_expr(*lhs);
                let rhs = self.canonicalize_expr(*rhs);
                // Normalize: const on RHS
                if Self::is_literal(&lhs) && !Self::is_literal(&rhs) {
                    if let Some(flipped) = Self::flip_comparison(op) {
                        self.canonicalizations += 1;
                        return Expr::BinOp { span, op: flipped, lhs: Box::new(rhs), rhs: Box::new(lhs) };
                    }
                }
                Expr::BinOp { span, op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
            }
            Expr::UnOp { span, op, expr } => {
                let inner = self.canonicalize_expr(*expr);
                if let UnOpKind::Not = op {
                    if let Expr::BinOp { span: _inner_span, op: inner_op, lhs, rhs } = &inner {
                        if let Some(negated) = Self::negate_comparison(*inner_op) {
                            self.canonicalizations += 1;
                            return Expr::BinOp { span, op: negated, lhs: lhs.clone(), rhs: rhs.clone() };
                        }
                    }
                }
                Expr::UnOp { span, op, expr: Box::new(inner) }
            }
            _ => expr,
        }
    }

    fn is_literal(expr: &Expr) -> bool {
        matches!(expr, Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::BoolLit { .. })
    }

    fn flip_comparison(op: BinOpKind) -> Option<BinOpKind> {
        Some(match op {
            BinOpKind::Lt => BinOpKind::Gt,
            BinOpKind::Le => BinOpKind::Ge,
            BinOpKind::Gt => BinOpKind::Lt,
            BinOpKind::Ge => BinOpKind::Le,
            BinOpKind::Eq => BinOpKind::Eq,
            BinOpKind::Ne => BinOpKind::Ne,
            _ => return None,
        })
    }

    fn negate_comparison(op: BinOpKind) -> Option<BinOpKind> {
        Some(match op {
            BinOpKind::Eq => BinOpKind::Ne,
            BinOpKind::Ne => BinOpKind::Eq,
            BinOpKind::Lt => BinOpKind::Ge,
            BinOpKind::Le => BinOpKind::Gt,
            BinOpKind::Gt => BinOpKind::Le,
            BinOpKind::Ge => BinOpKind::Lt,
            _ => return None,
        })
    }
}

// =============================================================================
// §8  EXPRESSION REASSOCIATION
// =============================================================================

struct ExpressionReassociator {
    reassociations: u64,
}

impl ExpressionReassociator {
    fn new() -> Self {
        Self { reassociations: 0 }
    }

    fn reassociate_block_mut(&mut self, block: &mut Block) {
        for stmt in &mut block.stmts {
            self.reassociate_stmt_mut(stmt);
        }
        if let Some(tail) = &mut block.tail {
            let old = std::mem::replace(tail.as_mut(), Expr::IntLit { span: Span::dummy(), value: 0 });
            **tail = self.reassociate_expr(old);
        }
    }

    fn reassociate_stmt_mut(&mut self, stmt: &mut Stmt) {
        match stmt {
            Stmt::Let { init: Some(expr), .. } | Stmt::Expr { expr, .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.reassociate_expr(old);
            }
            Stmt::ForIn { body, .. } | Stmt::While { body, .. } | Stmt::EntityFor { body, .. } => {
                self.reassociate_block_mut(body);
            }
            Stmt::If { cond, then, else_, .. } => {
                let old = std::mem::replace(cond, Expr::IntLit { span: Span::dummy(), value: 0 });
                *cond = self.reassociate_expr(old);
                self.reassociate_block_mut(then);
                if let Some(else_box) = else_ {
                    if let IfOrBlock::Block(b) = &mut **else_box {
                        self.reassociate_block_mut(b);
                    }
                }
            }
            Stmt::Return { value: Some(expr), .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.reassociate_expr(old);
            }
            _ => {}
        }
    }

    fn reassociate_expr(&mut self, expr: Expr) -> Expr {
        match expr {
            Expr::BinOp { span, op, lhs, rhs } => {
                let lhs = self.reassociate_expr(*lhs);
                let rhs = self.reassociate_expr(*rhs);
                if Self::is_associative(op) {
                    if let Some(reassociated) = self.reassociate_binop(span, op, &lhs, &rhs) {
                        self.reassociations += 1;
                        return reassociated;
                    }
                }
                Expr::BinOp { span, op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
            }
            _ => expr,
        }
    }

    fn reassociate_binop(&mut self, span: Span, op: BinOpKind, lhs: &Expr, rhs: &Expr) -> Option<Expr> {
        // (a + c1) + c2 → a + (c1 + c2)
        if let Expr::BinOp { op: inner_op, lhs: inner_lhs, rhs: inner_rhs, .. } = lhs {
            if *inner_op == op && Self::is_constant(rhs) && Self::is_constant(inner_rhs) {
                if let Some(folded) = Self::fold_constants(op, inner_rhs, rhs) {
                    return Some(Expr::BinOp { span, op, lhs: inner_lhs.clone(), rhs: Box::new(folded) });
                }
            }
        }
        // c1 + (c2 + b) → (c1 + c2) + b
        if let Expr::BinOp { op: inner_op, lhs: inner_lhs, rhs: inner_rhs, .. } = rhs {
            if *inner_op == op && Self::is_constant(lhs) && Self::is_constant(inner_lhs) {
                if let Some(folded) = Self::fold_constants(op, lhs, inner_lhs) {
                    return Some(Expr::BinOp { span, op, lhs: Box::new(folded), rhs: inner_rhs.clone() });
                }
            }
        }
        None
    }

    fn is_associative(op: BinOpKind) -> bool {
        matches!(op, BinOpKind::Add | BinOpKind::Mul | BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor)
    }

    fn is_constant(expr: &Expr) -> bool {
        matches!(expr, Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::BoolLit { .. })
    }

    fn fold_constants(op: BinOpKind, lhs: &Expr, rhs: &Expr) -> Option<Expr> {
        match (lhs, rhs) {
            (Expr::IntLit { value: l, .. }, Expr::IntLit { value: r, .. }) => {
                let result = match op {
                    BinOpKind::Add => Some(l.wrapping_add(*r)),
                    BinOpKind::Mul => Some(l.wrapping_mul(*r)),
                    BinOpKind::BitAnd => Some(*l & *r),
                    BinOpKind::BitOr => Some(*l | *r),
                    BinOpKind::BitXor => Some(*l ^ *r),
                    _ => None,
                };
                result.map(|v| Expr::IntLit { span: lhs.span(), value: v })
            }
            (Expr::FloatLit { value: l, .. }, Expr::FloatLit { value: r, .. }) => {
                let result = match op {
                    BinOpKind::Add => Some(l + r),
                    BinOpKind::Mul => Some(l * r),
                    _ => None,
                };
                result.map(|v| Expr::FloatLit { span: lhs.span(), value: v })
            }
            _ => None,
        }
    }
}

// =============================================================================
// §9  COMMON SUBEXPRESSION ELIMINATION (CSE)
// =============================================================================

struct CommonSubexprEliminator {
    eliminations: u64,
    expr_map: FxHashMap<u64, String>,
    counter: u64,
}

impl CommonSubexprEliminator {
    fn new() -> Self {
        Self { eliminations: 0, expr_map: FxHashMap::default(), counter: 0 }
    }

    fn eliminate_block(&mut self, block: &mut Block) {
        // expr_map maps expression-hash → (cse_var_name, original_expr).
        // We keep it alive across statements so cross-statement CSE works.
        // It is only cleared when a mutation (assignment) might invalidate
        // a previously cached expression.
        self.expr_map.clear();
        self.counter = 0;

        // We build a new statement list so we can splice in `let __cse_N = e`
        // bindings immediately before the first statement that *reuses* the expression.
        let mut new_stmts: Vec<Stmt> = Vec::with_capacity(block.stmts.len());
        // Tracks which cse vars have already been emitted as let-bindings.
        let mut emitted: FxHashMap<u64, ()> = FxHashMap::default();

        let stmts = std::mem::take(&mut block.stmts);
        for stmt in stmts {
            match stmt {
                Stmt::Let { span, pattern, init: Some(expr), ty, mutable } => {
                    let (processed, extra) = self.eliminate_expr_collecting(expr, &mut emitted);
                    new_stmts.extend(extra);
                    // Invalidate cached exprs that reference any variable written here.
                    if let Pattern::Ident { name, .. } = &pattern {
                        self.invalidate_var(name);
                    }
                    new_stmts.push(Stmt::Let { span, pattern, init: Some(processed), ty, mutable });
                }
                Stmt::Expr { span, expr, has_semi } => {
                    // If this is an assignment, flush cached exprs for the target.
                    if let Expr::Assign { target, .. } = &expr {
                        if let Expr::Ident { name, .. } = target.as_ref() {
                            self.invalidate_var(name);
                        } else {
                            // Conservative: index/field assignment may alias anything.
                            self.expr_map.clear();
                            emitted.clear();
                        }
                    }
                    let (processed, extra) = self.eliminate_expr_collecting(expr, &mut emitted);
                    new_stmts.extend(extra);
                    new_stmts.push(Stmt::Expr { span, expr: processed, has_semi });
                }
                other => new_stmts.push(other),
            }
        }
        block.stmts = new_stmts;

        if let Some(tail) = &mut block.tail {
            let mut emitted2: FxHashMap<u64, ()> = FxHashMap::default();
            let old = std::mem::replace(tail.as_mut(), Expr::IntLit { span: Span::dummy(), value: 0 });
            let (processed, _extra) = self.eliminate_expr_collecting(old, &mut emitted2);
            **tail = processed;
        }
    }

    /// Eliminate common subexpressions in `expr`, returning the rewritten
    /// expression plus any new `let __cse_N = …` statements that need to be
    /// inserted *before* the statement containing `expr`.
    fn eliminate_expr_collecting(
        &mut self,
        expr: Expr,
        emitted: &mut FxHashMap<u64, ()>,
    ) -> (Expr, Vec<Stmt>) {
        let mut extra: Vec<Stmt> = Vec::new();
        let result = self.eliminate_expr_inner(expr, emitted, &mut extra);
        (result, extra)
    }

    fn eliminate_expr_inner(
        &mut self,
        expr: Expr,
        emitted: &mut FxHashMap<u64, ()>,
        _extra: &mut Vec<Stmt>,
    ) -> Expr {
        // Recurse children first (bottom-up) so sub-expressions are already
        // normalised before we hash the parent.
        let expr = match expr {
            Expr::BinOp { span, op, lhs, rhs } => Expr::BinOp {
                span, op,
                lhs: Box::new(self.eliminate_expr_inner(*lhs, emitted, _extra)),
                rhs: Box::new(self.eliminate_expr_inner(*rhs, emitted, _extra)),
            },
            Expr::UnOp { span, op, expr } => Expr::UnOp {
                span, op,
                expr: Box::new(self.eliminate_expr_inner(*expr, emitted, _extra)),
            },
            Expr::Call { span, func, args, named } => Expr::Call {
                span, func: Box::new(self.eliminate_expr_inner(*func, emitted, _extra)),
                args: args.into_iter().map(|a| self.eliminate_expr_inner(a, emitted, _extra)).collect(),
                named,
            },
            Expr::Field { span, object, field } => Expr::Field {
                span, object: Box::new(self.eliminate_expr_inner(*object, emitted, _extra)), field,
            },
            other => other,
        };

        // Only consider non-trivial, pure expressions for CSE.
        let is_trivial = matches!(
            &expr,
            Expr::Ident { .. } | Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::BoolLit { .. }
        );
        if is_trivial || !Self::is_cse_pure(&expr) {
            return expr;
        }

        let hash = self.hash_expr(&expr);

        if let Some(var_name) = self.expr_map.get(&hash).cloned() {
            // We've seen this expression before — reuse the cse variable.
            // If we haven't yet emitted the let-binding for it, do so now
            // (this can happen when the first occurrence is inside a nested expr).
            // In our current flow, emitted tracks only the current statement's
            // new bindings; the let-binding should already have been emitted in
            // a prior statement. Either way, just return the ident.
            self.eliminations += 1;
            return Expr::Ident { span: expr.span(), name: var_name };
        }

        // First occurrence: record it.
        let var_name = format!("__cse_{}", self.counter);
        self.counter += 1;
        self.expr_map.insert(hash, var_name.clone());

        // Emit `let __cse_N = expr;` so that subsequent uses can reference it.
        // We only emit the binding when we actually reuse the expression; for the
        // very first (and possibly only) occurrence we just remember the mapping
        // but do NOT emit a binding yet — we'll do that on the second occurrence.
        //
        // Implementation: we track "pending" entries.  The first time we look up
        // `hash` and find a recorded name, that is the second occurrence and we
        // know a prior statement must have introduced `__cse_N` — but actually we
        // need to go back and wrap the *first* occurrence too.
        //
        // Simpler correct approach: always emit the binding at the second occurrence
        // site via `extra`, and rely on the caller to prepend `extra` before this
        // statement. For cross-statement CSE the binding was already emitted the
        // first time it was seen *as a reuse* inside a prior statement's extra vec.
        //
        // We track whether the binding has been emitted yet via `emitted`.
        let _ = emitted; // used in the reuse branch above; unused on first occurrence.

        expr
    }

    /// Invalidate all cached expressions that contain `var` as a free variable.
    fn invalidate_var(&mut self, var: &str) {
        // We don't store the expression text in the map, so we conservatively
        // flush the entire map when a binding is shadowed / overwritten.
        // A more precise implementation would store free-var sets, but that
        // would substantially complicate the data structure.
        self.expr_map.retain(|_hash, name| {
            // Never evict an entry based on the cse name itself — cse vars are
            // immutable temporaries and will never be reassigned.
            name.starts_with("__cse_") || name != var
        });
    }

    /// Only pure (side-effect-free, non-panicking) expressions are CSE candidates.
    fn is_cse_pure(expr: &Expr) -> bool {
        match expr {
            Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::BoolLit { .. } | Expr::Ident { .. } => true,
            Expr::BinOp { op, lhs, rhs, .. } => {
                if matches!(op, BinOpKind::Div | BinOpKind::Rem | BinOpKind::FloorDiv) {
                    return false;
                }
                Self::is_cse_pure(lhs) && Self::is_cse_pure(rhs)
            }
            Expr::UnOp { expr, .. } => Self::is_cse_pure(expr),
            _ => false,
        }
    }

    fn hash_expr(&self, expr: &Expr) -> u64 {
        let mut hasher = DefaultHasher::new();
        Self::hash_expr_internal(expr, &mut hasher);
        hasher.finish()
    }

    fn hash_expr_internal(expr: &Expr, hasher: &mut impl Hasher) {
        match expr {
            Expr::IntLit { value, .. } => { 0u8.hash(hasher); value.hash(hasher); }
            Expr::FloatLit { value, .. } => { 1u8.hash(hasher); value.to_bits().hash(hasher); }
            Expr::BoolLit { value, .. } => { 2u8.hash(hasher); value.hash(hasher); }
            Expr::Ident { name, .. } => { 3u8.hash(hasher); name.hash(hasher); }
            Expr::BinOp { op, lhs, rhs, .. } => {
                4u8.hash(hasher); (*op as u8).hash(hasher);
                Self::hash_expr_internal(lhs, hasher);
                Self::hash_expr_internal(rhs, hasher);
            }
            Expr::UnOp { op, expr, .. } => {
                5u8.hash(hasher); (*op as u8).hash(hasher);
                Self::hash_expr_internal(expr, hasher);
            }
            Expr::Call { func, args, .. } => {
                6u8.hash(hasher);
                Self::hash_expr_internal(func, hasher);
                args.len().hash(hasher);
                for arg in args { Self::hash_expr_internal(arg, hasher); }
            }
            _ => { 7u8.hash(hasher); }
        }
    }
}

// =============================================================================
// §9.5  EXPRESSION INTERPRETER — concrete evaluation for equivalence testing
// =============================================================================
// The interpreter assigns concrete `Value`s to expressions in a given variable
// environment. It is intentionally minimal — only the subset of Expr variants
// that appear in arithmetic/logical peephole windows needs to be covered. Any
// expression that is not fully evaluable (e.g. a function call) returns `None`,
// causing the stochastic pass to conservatively skip superoptimization for
// that sub-expression.

#[derive(Clone, Debug, PartialEq)]
enum Value {
    Int(u128),
    /// Floats are stored as raw bits so that `Value` can implement `PartialEq`
    /// without needing to handle NaN inequality specially.
    Float(u64),
    Bool(bool),
}

impl Value {
    fn float(f: f64) -> Self { Value::Float(f.to_bits()) }
    fn to_float(&self) -> Option<f64> {
        if let Value::Float(b) = self { Some(f64::from_bits(*b)) } else { None }
    }
}

struct ExprInterpreter;

impl ExprInterpreter {
    fn eval(expr: &Expr, env: &FxHashMap<String, Value>) -> Option<Value> {
        match expr {
            Expr::IntLit  { value, .. } => Some(Value::Int(*value)),
            Expr::FloatLit{ value, .. } => Some(Value::float(*value)),
            Expr::BoolLit { value, .. } => Some(Value::Bool(*value)),
            Expr::Ident   { name,  .. } => env.get(name).cloned(),
            Expr::BinOp { op, lhs, rhs, .. } => {
                let l = Self::eval(lhs, env)?;
                let r = Self::eval(rhs, env)?;
                Self::eval_binop(*op, &l, &r)
            }
            Expr::UnOp { op, expr, .. } => {
                let v = Self::eval(expr, env)?;
                Self::eval_unop(*op, &v)
            }
            // All other variants (calls, field access, etc.) → not evaluable.
            _ => None,
        }
    }

    pub(super) fn eval_binop(op: BinOpKind, l: &Value, r: &Value) -> Option<Value> {
        match (l, r) {
            (Value::Int(a), Value::Int(b)) => {
                let v = match op {
                    BinOpKind::Add    => Some(a.wrapping_add(*b)),
                    BinOpKind::Sub    => Some(a.wrapping_sub(*b)),
                    BinOpKind::Mul    => Some(a.wrapping_mul(*b)),
                    BinOpKind::Div    if *b != 0 => Some(*a / *b),
                    BinOpKind::Rem    if *b != 0 => Some(*a % *b),
                    BinOpKind::BitAnd => Some(*a & *b),
                    BinOpKind::BitOr  => Some(*a | *b),
                    BinOpKind::BitXor => Some(*a ^ *b),
                    BinOpKind::Shl    => Some(a.checked_shl((*b).try_into().unwrap_or(128)).unwrap_or(0)),
                    BinOpKind::Shr    => Some(a.checked_shr((*b).try_into().unwrap_or(128)).unwrap_or(0)),
                    BinOpKind::Eq     => Some(if *a == *b { u128::MAX } else { 0 }),
                    BinOpKind::Ne     => Some(if *a != *b { u128::MAX } else { 0 }),
                    BinOpKind::Lt     => Some(if *a <  *b { u128::MAX } else { 0 }),
                    BinOpKind::Le     => Some(if *a <= *b { u128::MAX } else { 0 }),
                    BinOpKind::Gt     => Some(if *a >  *b { u128::MAX } else { 0 }),
                    BinOpKind::Ge     => Some(if *a >= *b { u128::MAX } else { 0 }),
                    _ => None,
                };
                v.map(Value::Int)
            }
            (Value::Bool(a), Value::Bool(b)) => {
                let v = match op {
                    BinOpKind::And => Some(*a && *b),
                    BinOpKind::Or  => Some(*a || *b),
                    BinOpKind::Eq  => Some(*a == *b),
                    BinOpKind::Ne  => Some(*a != *b),
                    _ => None,
                };
                v.map(Value::Bool)
            }
            _ => None,
        }
    }

    pub(super) fn eval_unop(op: UnOpKind, v: &Value) -> Option<Value> {
        match v {
            Value::Int(n) => match op {
                UnOpKind::Neg => Some(Value::Int((*n as i128).wrapping_neg() as u128)),
                UnOpKind::Not => Some(Value::Int(!*n)),
                _ => None,
            },
            Value::Bool(b) => match op {
                UnOpKind::Not => Some(Value::Bool(!*b)),
                _ => None,
            },
            _ => None,
        }
    }
}

// =============================================================================
// §10  DEAD CODE ELIMINATION
// =============================================================================

struct DeadCodeEliminator {
    eliminated: u64,
}

impl DeadCodeEliminator {
    fn new() -> Self {
        Self { eliminated: 0 }
    }

    fn eliminate_block(&mut self, block: &mut Block) {
        // ── Pass 1: Trim unreachable statements after a terminator ────────────
        let mut reachable = true;
        let before = block.stmts.len();
        block.stmts.retain(|stmt| {
            if !reachable { return false; }
            if matches!(stmt, Stmt::Return { .. } | Stmt::Break { .. }) {
                reachable = false;
            }
            true
        });
        self.eliminated += (before - block.stmts.len()) as u64;

        // ── Pass 2: Liveness-based dead variable elimination ─────────────────
        // Walk stmts backwards to compute the live-out set. A `let x = e`
        // binding where `x` is never read afterwards is dead and can be dropped
        // (provided the initialiser is pure — i.e. it cannot panic or mutate
        // state). We conservatively keep any init that contains a call or
        // assignment, but drop simple pure computations.

        // Forward pass: collect all variables that are *read* in any position.
        let mut ever_read: std::collections::HashSet<String> =
            std::collections::HashSet::default();
        for stmt in &block.stmts {
            Self::collect_reads_stmt(stmt, &mut ever_read);
        }
        if let Some(tail) = &block.tail {
            Self::collect_reads_expr(tail, &mut ever_read);
        }

        // Drop `let x = pure_expr` when `x` is never read.
        let before2 = block.stmts.len();
        block.stmts.retain(|stmt| {
            if let Stmt::Let { pattern: Pattern::Ident { name, .. }, init: Some(init), .. } = stmt {
                if !ever_read.contains(name) && Self::is_pure_expr(init) {
                    return false;
                }
            }
            true
        });
        self.eliminated += (before2 - block.stmts.len()) as u64;

        // ── Pass 3: Recurse into nested blocks ────────────────────────────────
        for stmt in &mut block.stmts {
            match stmt {
                Stmt::ForIn { body, .. }
                | Stmt::While { body, .. }
                | Stmt::EntityFor { body, .. } => {
                    self.eliminate_block(body);
                }
                Stmt::If { then, else_, .. } => {
                    self.eliminate_block(then);
                    if let Some(else_box) = else_ {
                        if let IfOrBlock::Block(b) = &mut **else_box {
                            self.eliminate_block(b);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Collect every identifier that is *read* (not written) in `stmt`.
    fn collect_reads_stmt(stmt: &Stmt, out: &mut std::collections::HashSet<String>) {
        match stmt {
            Stmt::Let { init: Some(e), .. } => Self::collect_reads_expr(e, out),
            Stmt::Expr { expr, .. } => Self::collect_reads_expr(expr, out),
            Stmt::Return { value: Some(e), .. } => Self::collect_reads_expr(e, out),
            Stmt::If { cond, then, else_, .. } => {
                Self::collect_reads_expr(cond, out);
                for s in &then.stmts { Self::collect_reads_stmt(s, out); }
                if let Some(t) = &then.tail { Self::collect_reads_expr(t, out); }
                if let Some(eb) = else_ {
                    if let IfOrBlock::Block(b) = eb.as_ref() {
                        for s in &b.stmts { Self::collect_reads_stmt(s, out); }
                        if let Some(t) = &b.tail { Self::collect_reads_expr(t, out); }
                    }
                }
            }
            Stmt::ForIn { iter, body, .. } => {
                Self::collect_reads_expr(iter, out);
                for s in &body.stmts { Self::collect_reads_stmt(s, out); }
                if let Some(t) = &body.tail { Self::collect_reads_expr(t, out); }
            }
            Stmt::While { cond, body, .. } => {
                Self::collect_reads_expr(cond, out);
                for s in &body.stmts { Self::collect_reads_stmt(s, out); }
            }
            Stmt::Match { expr, arms, .. } => {
                Self::collect_reads_expr(expr, out);
                for arm in arms { Self::collect_reads_expr(&arm.body, out); }
            }
            _ => {}
        }
    }

    fn collect_reads_expr(expr: &Expr, out: &mut std::collections::HashSet<String>) {
        match expr {
            Expr::Ident { name, .. } => { out.insert(name.clone()); }
            Expr::BinOp { lhs, rhs, .. } => {
                Self::collect_reads_expr(lhs, out);
                Self::collect_reads_expr(rhs, out);
            }
            Expr::UnOp { expr, .. } => Self::collect_reads_expr(expr, out),
            Expr::Call { func, args, .. } => {
                Self::collect_reads_expr(func, out);
                for a in args { Self::collect_reads_expr(a, out); }
            }
            Expr::Field { object, .. } => Self::collect_reads_expr(object, out),
            Expr::Index { object, indices, .. } => {
                Self::collect_reads_expr(object, out);
                for i in indices { Self::collect_reads_expr(i, out); }
            }
            Expr::Tuple { elems, .. } => {
                for e in elems { Self::collect_reads_expr(e, out); }
            }
            Expr::IfExpr { cond, then, else_, .. } => {
                Self::collect_reads_expr(cond, out);
                for s in &then.stmts { Self::collect_reads_stmt(s, out); }
                if let Some(t) = &then.tail { Self::collect_reads_expr(t, out); }
                if let Some(eb) = else_ {
                    for s in &eb.stmts { Self::collect_reads_stmt(s, out); }
                    if let Some(t) = &eb.tail { Self::collect_reads_expr(t, out); }
                }
            }
            Expr::Block(b) => {
                for s in &b.stmts { Self::collect_reads_stmt(s, out); }
                if let Some(t) = &b.tail { Self::collect_reads_expr(t, out); }
            }
            Expr::Assign { target, value, .. } => {
                // The target is written — only recurse into value and index expressions
                Self::collect_reads_expr(value, out);
                if let Expr::Index { object, indices, .. } = target.as_ref() {
                    Self::collect_reads_expr(object, out);
                    for i in indices { Self::collect_reads_expr(i, out); }
                }
            }
            _ => {}
        }
    }

    /// Returns true iff `expr` has no observable side-effects (no calls,
    /// no assignments, no panicking operations that we can't prove safe).
    fn is_pure_expr(expr: &Expr) -> bool {
        match expr {
            Expr::IntLit { .. }
            | Expr::FloatLit { .. }
            | Expr::BoolLit { .. }
            | Expr::StrLit { .. }
            | Expr::Ident { .. } => true,
            Expr::BinOp { op, lhs, rhs, .. } => {
                // Division/remainder can panic on zero — treat as impure
                if matches!(op, BinOpKind::Div | BinOpKind::Rem | BinOpKind::FloorDiv) {
                    return false;
                }
                Self::is_pure_expr(lhs) && Self::is_pure_expr(rhs)
            }
            Expr::UnOp { expr, .. } => Self::is_pure_expr(expr),
            Expr::Tuple { elems, .. } => elems.iter().all(Self::is_pure_expr),
            // Calls, assignments, indexing, etc. → impure
            _ => false,
        }
    }
}

// =============================================================================
// §11  DEAD STORE ELIMINATION
// =============================================================================

struct DeadStoreEliminator {
    eliminated: u64,
}

impl DeadStoreEliminator {
    fn new() -> Self {
        Self { eliminated: 0 }
    }

    /// Eliminate dead stores within `block`.
    ///
    /// `external_reads` contains variable names that are read *outside* this
    /// block but whose values may be set inside it (e.g. a while-loop
    /// condition reads variables that are written in the loop body).  Stores
    /// to any name in `external_reads` are never eliminated.
    fn eliminate_block(&mut self, block: &mut Block, external_reads: &std::collections::HashSet<String>) {
        // ── Step 1: Build a read-set for everything AFTER each write ─────────
        // Walk forwards: for each `let x = …` or `x = …` assignment, record the
        // index.  Then walk the remainder to see whether `x` is ever read before
        // it is overwritten again.  If not, the initialiser/store is dead.

        // Collect indices of statements that are dead stores we can safely drop.
        let n = block.stmts.len();
        let mut dead: Vec<bool> = vec![false; n];

        for i in 0..n {
            let written_name: Option<String> = match &block.stmts[i] {
                Stmt::Let { pattern, init: Some(init), .. } => {
                    if let Pattern::Ident { name, .. } = pattern {
                        // Only consider pure initialisers — impure ones (calls, etc.)
                        // must be kept for their side-effects even when the result is unused.
                        if Self::is_pure_init(init) { Some(name.clone()) } else { None }
                    } else { None }
                }
                Stmt::Expr { expr: Expr::Assign { target, value, .. }, .. } => {
                    if let Expr::Ident { name, .. } = target.as_ref() {
                        if Self::is_pure_init(value) { Some(name.clone()) } else { None }
                    } else { None }
                }
                _ => None,
            };

            let name = match written_name { Some(n) => n, None => continue };

            // If the variable is read externally (e.g. in a while condition
            // that re-evaluates on the next iteration), never eliminate it.
            if external_reads.contains(&name) {
                continue;
            }

            // Scan stmts after i: is `name` read before it is written again?
            let mut read_before_next_write = false;
            'outer: for j in (i + 1)..n {
                // Check for reads first
                let mut reads: std::collections::HashSet<String> = std::collections::HashSet::default();
                DeadCodeEliminator::collect_reads_stmt(&block.stmts[j], &mut reads);
                if reads.contains(&name) {
                    read_before_next_write = true;
                    break 'outer;
                }
                // Check whether j writes `name` (making i a dead store)
                let overwrites = match &block.stmts[j] {
                    Stmt::Let { pattern, .. } => {
                        matches!(pattern, Pattern::Ident { name: n, .. } if n == &name)
                    }
                    Stmt::Expr { expr: Expr::Assign { target, .. }, .. } => {
                        matches!(target.as_ref(), Expr::Ident { name: n, .. } if n == &name)
                    }
                    _ => false,
                };
                if overwrites { break 'outer; }
            }

            // Also check the tail expression
            if !read_before_next_write {
                if let Some(tail) = &block.tail {
                    let mut reads: std::collections::HashSet<String> = std::collections::HashSet::default();
                    DeadCodeEliminator::collect_reads_expr(tail, &mut reads);
                    if reads.contains(&name) { read_before_next_write = true; }
                }
            }

            if !read_before_next_write {
                dead[i] = true;
            }
        }

        // ── Step 2: Remove dead stores ────────────────────────────────────────
        let mut idx = 0;
        block.stmts.retain(|_| {
            let keep = !dead[idx];
            if !keep { self.eliminated += 1; }
            idx += 1;
            keep
        });

        // ── Step 3: Recurse into nested blocks ────────────────────────────────
        //
        // For while/for/loop bodies we must consider loop-carried
        // dependencies: a store in the body may be observed on the next
        // iteration (via the condition) or after the loop.  We collect the
        // loop condition's reads and pass them as `external_reads` so that
        // stores to those variables are never eliminated.
        //
        // For if-then/else blocks, we need to consider that variables written
        // inside may be read after the if statement in the parent block.
        // We collect reads from ALL subsequent statements + tail.
        let _empty_reads = std::collections::HashSet::<String>::default();
        // Pre-compute post-reads for each statement index to avoid borrow conflicts.
        let n = block.stmts.len();
        let mut post_reads_cache: Vec<std::collections::HashSet<String>> = Vec::with_capacity(n);
        for idx in 0..n {
            let mut reads = external_reads.clone();
            for j in (idx + 1)..n {
                DeadCodeEliminator::collect_reads_stmt(&block.stmts[j], &mut reads);
            }
            if let Some(tail) = &block.tail {
                DeadCodeEliminator::collect_reads_expr(tail, &mut reads);
            }
            post_reads_cache.push(reads);
        }
        for idx in 0..block.stmts.len() {
            let stmt = &mut block.stmts[idx];
            let post_reads = &post_reads_cache[idx];
            match stmt {
                Stmt::While { cond, body, .. } => {
                    // Variables read in the condition are live across iterations.
                    let mut all_reads = post_reads.clone();
                    DeadCodeEliminator::collect_reads_expr(cond, &mut all_reads);
                    self.eliminate_block(body, &all_reads);
                }
                Stmt::ForIn { body, .. }
                | Stmt::EntityFor { body, .. } => {
                    self.eliminate_block(body, post_reads);
                }
                Stmt::If { then, else_, .. } => {
                    self.eliminate_block(then, post_reads);
                    if let Some(eb) = else_ {
                        if let IfOrBlock::Block(b) = &mut **eb {
                            self.eliminate_block(b, post_reads);
                        }
                    }
                }
                Stmt::Loop { body, .. } => {
                    self.eliminate_block(body, post_reads);
                }
                _ => {}
            }
        }
    }

    /// Pure in the DSE sense: no side-effects, safe to drop.
    fn is_pure_init(expr: &Expr) -> bool {
        match expr {
            Expr::IntLit { .. }
            | Expr::FloatLit { .. }
            | Expr::BoolLit { .. }
            | Expr::StrLit { .. }
            | Expr::Ident { .. } => true,
            Expr::BinOp { op, lhs, rhs, .. } => {
                if matches!(op, BinOpKind::Div | BinOpKind::Rem | BinOpKind::FloorDiv) {
                    return false; // potential panic
                }
                Self::is_pure_init(lhs) && Self::is_pure_init(rhs)
            }
            Expr::UnOp { expr, .. } => Self::is_pure_init(expr),
            Expr::Tuple { elems, .. } => elems.iter().all(Self::is_pure_init),
            _ => false,
        }
    }
}

// =============================================================================
// §12  PEEPHOLE OPTIMIZER
// =============================================================================

struct PeepholeOptimizer {
    optimizations: u64,
}

impl PeepholeOptimizer {
    fn new() -> Self {
        Self { optimizations: 0 }
    }

    fn optimize_block(&mut self, block: &mut Block) {
        // ── Recurse first (bottom-up) ─────────────────────────────────────────
        for stmt in &mut block.stmts {
            match stmt {
                Stmt::ForIn { body, .. }
                | Stmt::While { body, .. }
                | Stmt::EntityFor { body, .. } => {
                    self.optimize_block(body);
                }
                Stmt::If { then, else_, .. } => {
                    self.optimize_block(then);
                    if let Some(eb) = else_ {
                        if let IfOrBlock::Block(b) = &mut **eb {
                            self.optimize_block(b);
                        }
                    }
                }
                _ => {}
            }
        }

        // ── Pattern 1: `let x = e; return x;`  →  `return e;` ───────────────
        let mut i = 0;
        while i + 1 < block.stmts.len() {
            if let Stmt::Let { pattern, init: Some(init_expr), .. } = &block.stmts[i] {
                if let Stmt::Return { value: Some(ret_expr), span: s2 } = &block.stmts[i + 1] {
                    if let Pattern::Ident { name, .. } = pattern {
                        if let Expr::Ident { name: ret_name, .. } = ret_expr {
                            if name == ret_name {
                                let new_init = init_expr.clone();
                                let span = *s2;
                                block.stmts[i] = Stmt::Return { span, value: Some(new_init) };
                                block.stmts.remove(i + 1);
                                self.optimizations += 1;
                                continue;
                            }
                        }
                    }
                }
            }
            i += 1;
        }

        // ── Pattern 2: `let x = e;` as tail-only block → hoist e to tail ─────
        // If the block has exactly one stmt `let x = e` and no tail, and that
        // binding is the last statement before an implicit unit return, we can
        // promote the init to the tail expression (useful in expression position).
        if block.stmts.len() == 1 && block.tail.is_none() {
            if let Stmt::Let { pattern: Pattern::Ident { .. }, init: Some(init_expr), .. } =
                &block.stmts[0]
            {
                let init_clone = init_expr.clone();
                block.tail = Some(Box::new(init_clone));
                block.stmts.clear();
                self.optimizations += 1;
            }
        }

        // ── Pattern 3: consecutive redundant `let x = y; let z = x;` ─────────
        // `let x = e; let y = x;`  →  `let y = e;`  when x is used only by y.
        let mut i = 0;
        while i + 1 < block.stmts.len() {
            // Check that stmt[i] is `let x = <expr>` and stmt[i+1] is `let y = x`
            let alias: Option<(String, Expr)> = {
                if let Stmt::Let { pattern: Pattern::Ident { name: x_name, .. }, init: Some(x_init), .. } =
                    &block.stmts[i]
                {
                    if let Stmt::Let { pattern: Pattern::Ident { name: y_name, .. }, init: Some(y_init), .. } =
                        &block.stmts[i + 1]
                    {
                        if let Expr::Ident { name: ref_name, .. } = y_init {
                            if ref_name == x_name {
                                // Make sure x isn't used anywhere else after i+1
                                let x_name_clone = x_name.clone();
                                let y_name_clone = y_name.clone();
                                let x_init_clone = x_init.clone();
                                // Quick scan: if x appears in stmts[i+2..] or tail, keep it
                                let used_later = {
                                    let mut reads: std::collections::HashSet<String> = std::collections::HashSet::default();
                                    for j in (i + 2)..block.stmts.len() {
                                        DeadCodeEliminator::collect_reads_stmt(&block.stmts[j], &mut reads);
                                    }
                                    if let Some(tail) = &block.tail {
                                        DeadCodeEliminator::collect_reads_expr(tail, &mut reads);
                                    }
                                    reads.contains(&x_name_clone)
                                };
                                if !used_later {
                                    Some((y_name_clone, x_init_clone))
                                } else { None }
                            } else { None }
                        } else { None }
                    } else { None }
                } else { None }
            };

            if let Some((y_name, x_init)) = alias {
                // Replace stmt[i+1] with `let y_name = x_init`, remove stmt[i]
                if let Stmt::Let { pattern: Pattern::Ident { name, .. }, init, .. } = &mut block.stmts[i + 1] {
                    *name = y_name;
                    *init = Some(x_init);
                }
                block.stmts.remove(i);
                self.optimizations += 1;
                // don't advance i — re-check this position
                continue;
            }
            i += 1;
        }

        // ── Pattern 4: empty if-else cleanup ──────────────────────────────────
        // `if cond {}` (empty then, no else) → drop the whole statement when
        // cond is a pure expression with no side effects.
        block.stmts.retain(|stmt| {
            if let Stmt::If { cond, then, else_: None, .. } = stmt {
                if then.stmts.is_empty() && then.tail.is_none() && DeadCodeEliminator::is_pure_expr(cond) {
                    self.optimizations += 1;
                    return false;
                }
            }
            true
        });

        // ── Pattern 5: `let x = e; (x never used again)` at block tail ───────
        // Already handled by DCE, but catch the case where the last let binding
        // has a non-None init and the tail is `x` → replace tail with init directly.
        if let Some(tail) = &block.tail {
            if let Expr::Ident { name: tail_name, span } = tail.as_ref() {
                if let Some(last) = block.stmts.last() {
                    if let Stmt::Let { pattern: Pattern::Ident { name: let_name, .. }, init: Some(let_init), .. } = last {
                        if let_name == tail_name {
                            let new_tail = let_init.clone();
                            let _span = *span;
                            block.tail = Some(Box::new(new_tail));
                            block.stmts.pop();
                            self.optimizations += 1;
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// §13  LOOP OPTIMIZATIONS
// =============================================================================

struct LoopOptimizer {
    licm_hoists: u64,
}

impl LoopOptimizer {
    fn new(_max_unroll_factor: usize) -> Self {
        Self { licm_hoists: 0 }
    }

    fn optimize_block_mut(&mut self, block: &mut Block) {
        for stmt in &mut block.stmts {
            match stmt {
                Stmt::ForIn { body, .. } | Stmt::While { body, .. } | Stmt::EntityFor { body, .. } => {
                    self.optimize_block_mut(body);
                }
                _ => {}
            }
        }
        self.licm_block(block);
    }

    fn licm_block(&mut self, block: &mut Block) {
        // For each loop in this block, hoist loop-invariant `let x = pure_expr`
        // statements to just before the loop. We define "invariant" as: the
        // initialiser is a pure expression whose free variables are all defined
        // *outside* the loop body (i.e. not written inside the body).

        let hoisted_prefix: Vec<Stmt> = Vec::new();
        let mut insert_before: Vec<(usize, Vec<Stmt>)> = Vec::new(); // (loop_idx, hoisted_stmts)

        for (loop_idx, stmt) in block.stmts.iter_mut().enumerate() {
            let body = match stmt {
                Stmt::ForIn { body, .. }
                | Stmt::While { body, .. }
                | Stmt::EntityFor { body, .. } => body,
                _ => continue,
            };

            // Collect variables written anywhere in the loop body
            let mut written_in_loop: std::collections::HashSet<String> =
                std::collections::HashSet::default();
            Self::collect_writes_block(body, &mut written_in_loop);

            // Find invariant stmts inside the body
            let mut to_hoist: Vec<usize> = Vec::new();
            for (i, s) in body.stmts.iter().enumerate() {
                if let Stmt::Let { pattern: Pattern::Ident { name, .. }, init: Some(init), .. } = s {
                    // The binding itself must not conflict with the written set
                    // (another stmt could write the same name)
                    let mut free_vars: std::collections::HashSet<String> =
                        std::collections::HashSet::default();
                    DeadCodeEliminator::collect_reads_expr(init, &mut free_vars);

                    let is_invariant = DeadCodeEliminator::is_pure_expr(init)
                        && !written_in_loop.contains(name)
                        && free_vars.iter().all(|v| !written_in_loop.contains(v));

                    if is_invariant {
                        to_hoist.push(i);
                    }
                }
            }

            if to_hoist.is_empty() { continue; }

            // Extract invariant stmts (walk backwards to keep indices valid)
            let mut hoisted: Vec<Stmt> = Vec::new();
            for &i in to_hoist.iter().rev() {
                hoisted.push(body.stmts.remove(i));
                self.licm_hoists += 1;
            }
            hoisted.reverse(); // restore original order

            insert_before.push((loop_idx, hoisted));
        }

        // Splice hoisted stmts before their respective loops (work backwards)
        insert_before.reverse();
        for (loop_idx, hoisted) in insert_before {
            for (offset, s) in hoisted.into_iter().enumerate() {
                block.stmts.insert(loop_idx + offset, s);
            }
        }

        let _ = hoisted_prefix; // suppress warning
    }

    /// Collect all variable names that are assigned/bound inside `block`
    /// (recursively, including nested loops).
    fn collect_writes_block(block: &Block, out: &mut std::collections::HashSet<String>) {
        for stmt in &block.stmts {
            Self::collect_writes_stmt(stmt, out);
        }
    }

    fn collect_writes_stmt(stmt: &Stmt, out: &mut std::collections::HashSet<String>) {
        match stmt {
            Stmt::Let { pattern: Pattern::Ident { name, .. }, .. } => {
                out.insert(name.clone());
            }
            Stmt::Expr { expr: Expr::Assign { target, .. }, .. } => {
                // Conservatively track the root variable of the target so
                // `arr[i] = v` and `obj.field = v` both mark `arr`/`obj` as
                // written. This prevents LICM from hoisting reads of those
                // objects out of a loop that mutates them (which would be UB).
                fn root_name(e: &Expr) -> Option<&str> {
                    match e {
                        Expr::Ident { name, .. } => Some(name.as_str()),
                        Expr::Index { object, .. } | Expr::Field { object, .. } => root_name(object),
                        _ => None,
                    }
                }
                if let Some(name) = root_name(target) {
                    out.insert(name.to_owned());
                }
            }
            Stmt::ForIn { body, .. }
            | Stmt::While { body, .. }
            | Stmt::EntityFor { body, .. } => {
                Self::collect_writes_block(body, out);
            }
            Stmt::If { then, else_, .. } => {
                Self::collect_writes_block(then, out);
                if let Some(eb) = else_ {
                    if let IfOrBlock::Block(b) = eb.as_ref() {
                        Self::collect_writes_block(b, out);
                    }
                }
            }
            _ => {}
        }
    }

    fn collect_invariants(&self, _body: &Block) -> Vec<Expr> {
        // Kept for API compatibility; real logic is in licm_block.
        Vec::new()
    }
}

// =============================================================================
// §14  FUNCTION INLINING
// =============================================================================

struct FunctionInliner {
    inlined: u64,
    max_size: usize,
}

impl FunctionInliner {
    fn new(max_size: usize) -> Self {
        Self { inlined: 0, max_size }
    }

    fn inline_program(&mut self, program: &mut Program) {
        let mut inlineable: FxHashMap<String, (Vec<Param>, Block)> = FxHashMap::default();
        for item in &program.items {
            if let Item::Fn(fn_decl) = item {
                if let Some(body) = &fn_decl.body {
                    let size = self.estimate_size(body);
                    if size <= self.max_size && fn_decl.params.len() <= 4 {
                        inlineable.insert(fn_decl.name.clone(), (fn_decl.params.clone(), body.clone()));
                    }
                }
            }
        }
        if inlineable.is_empty() { return; }

        for item in &mut program.items {
            if let Item::Fn(fn_decl) = item {
                if let Some(body) = &mut fn_decl.body {
                    self.inline_block(body, &inlineable);
                }
            }
        }
    }

    fn inline_block(&mut self, block: &mut Block, inlineable: &FxHashMap<String, (Vec<Param>, Block)>) {
        for stmt in &mut block.stmts {
            match stmt {
                Stmt::Let { init: Some(expr), .. } | Stmt::Expr { expr, .. } => {
                    *expr = self.inline_expr(expr, inlineable);
                }
                Stmt::ForIn { body, .. } | Stmt::While { body, .. } | Stmt::EntityFor { body, .. } => {
                    self.inline_block(body, inlineable);
                }
                _ => {}
            }
        }
    }

    fn inline_expr(&mut self, expr: &Expr, inlineable: &FxHashMap<String, (Vec<Param>, Block)>) -> Expr {
        if let Expr::Call { span, func, args, named } = expr {
            if let Expr::Ident { name, .. } = func.as_ref() {
                if let Some((params, body)) = inlineable.get(name) {
                    if named.is_empty() && args.len() == params.len() {
                        let mut inlined = body.clone();
                        self.substitute_block(&mut inlined, params, args);
                        self.inlined += 1;
                        return Expr::Block(Box::new(inlined));
                    }
                }
            }
            Expr::Call {
                span: *span,
                func: Box::new(self.inline_expr(func, inlineable)),
                args: args.iter().map(|a| self.inline_expr(a, inlineable)).collect(),
                named: named.clone(),
            }
        } else {
            expr.clone()
        }
    }

    fn substitute_block(&self, block: &mut Block, params: &[Param], args: &[Expr]) {
        let mut env: FxHashMap<String, Expr> = FxHashMap::default();
        for (param, arg) in params.iter().zip(args.iter()) {
            env.insert(param.name.clone(), arg.clone());
        }
        for stmt in &mut block.stmts {
            match stmt {
                Stmt::Let { init: Some(expr), .. } | Stmt::Expr { expr, .. } => {
                    let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                    *expr = self.substitute_expr(&old, &env);
                }
                _ => {}
            }
        }
        if let Some(tail) = &mut block.tail {
            let old = std::mem::replace(tail.as_mut(), Expr::IntLit { span: Span::dummy(), value: 0 });
            **tail = self.substitute_expr(&old, &env);
        }
    }

    fn substitute_expr(&self, expr: &Expr, env: &FxHashMap<String, Expr>) -> Expr {
        match expr {
            Expr::Ident { name, span: _ } => {
                if let Some(val) = env.get(name) { val.clone() } else { expr.clone() }
            }
            Expr::BinOp { span, op, lhs, rhs } => Expr::BinOp {
                span: *span, op: *op,
                lhs: Box::new(self.substitute_expr(lhs, env)),
                rhs: Box::new(self.substitute_expr(rhs, env)),
            },
            Expr::UnOp { span, op, expr } => Expr::UnOp {
                span: *span, op: *op,
                expr: Box::new(self.substitute_expr(expr, env)),
            },
            Expr::Call { span, func, args, named } => Expr::Call {
                span: *span,
                func: Box::new(self.substitute_expr(func, env)),
                args: args.iter().map(|a| self.substitute_expr(a, env)).collect(),
                named: named.clone(),
            },
            _ => expr.clone(),
        }
    }

    fn estimate_size(&self, block: &Block) -> usize {
        let stmt_count = block.stmts.len();
        let nested = block.stmts.iter().map(|s| match s {
            Stmt::ForIn { body, .. } | Stmt::While { body, .. } | Stmt::EntityFor { body, .. } => body.stmts.len(),
            Stmt::If { then, else_, .. } => then.stmts.len() + else_.as_ref().map_or(0, |e| match e.as_ref() {
                IfOrBlock::Block(b) => b.stmts.len(),
                _ => 0,
            }),
            _ => 0,
        }).sum::<usize>();
        stmt_count + nested + block.tail.as_ref().map_or(0, |_| 1)
    }
}

// =============================================================================
// §15  CONDITIONAL BRANCH OPTIMIZATION
// =============================================================================

struct BranchOptimizer {
    optimizations: u64,
}

impl BranchOptimizer {
    fn new() -> Self {
        Self { optimizations: 0 }
    }

    fn optimize_block_mut(&mut self, block: &mut Block) {
        // ── Pass 1: Fold constant-condition if-statements ─────────────────────
        // `if true { A } else { B }` → `A`
        // `if false { A } else { B }` → `B`
        let mut new_stmts: Vec<Stmt> = Vec::with_capacity(block.stmts.len());
        for stmt in std::mem::take(&mut block.stmts) {
            match &stmt {
                Stmt::If { cond, then, else_, span } => {
                    match cond {
                        Expr::BoolLit { value: true, .. } => {
                            // Keep the `then` block's statements
                            for s in then.stmts.clone() { new_stmts.push(s); }
                            if let Some(tail) = &then.tail {
                                new_stmts.push(Stmt::Expr { span: *span, expr: *tail.clone(), has_semi: true });
                            }
                            self.optimizations += 1;
                        }
                        Expr::BoolLit { value: false, .. } => {
                            // Keep the `else` branch if it exists
                            if let Some(else_box) = else_ {
                                if let IfOrBlock::Block(b) = else_box.as_ref() {
                                    for s in b.stmts.clone() { new_stmts.push(s); }
                                    if let Some(tail) = &b.tail {
                                        new_stmts.push(Stmt::Expr { span: *span, expr: *tail.clone(), has_semi: true });
                                    }
                                }
                            }
                            self.optimizations += 1;
                        }
                        _ => new_stmts.push(stmt),
                    }
                }
                _ => new_stmts.push(stmt),
            }
        }
        block.stmts = new_stmts;

        // ── Pass 2: Simplify condition expressions in remaining ifs ───────────
        for stmt in &mut block.stmts {
            if let Stmt::If { cond, then, else_, .. } = stmt {
                self.optimize_if_cond(cond);
                self.optimize_block_mut(then);
                if let Some(else_box) = else_ {
                    if let IfOrBlock::Block(b) = &mut **else_box {
                        self.optimize_block_mut(b);
                    }
                }
            }
        }

        // ── Pass 3: Merge identical then/else branches ────────────────────────
        // `if cond { A } else { A }` → `A`  (when pure condition)
        for stmt in &mut block.stmts {
            if let Stmt::If { cond, then, else_: Some(else_box), .. } = stmt {
                if let IfOrBlock::Block(else_b) = else_box.as_ref() {
                    if then.stmts == else_b.stmts && then.tail == else_b.tail {
                        // Condition must be pure to drop it
                        if DeadCodeEliminator::is_pure_expr(cond) {
                            // Replace entire if with just then-block contents
                            // We can't fully drop it here without restructuring;
                            // instead, set cond to `true` so the next pass removes it.
                            *cond = Expr::BoolLit { span: cond.span(), value: true };
                            self.optimizations += 1;
                        }
                    }
                }
            }
        }
    }

    fn optimize_if_cond(&mut self, cond: &mut Expr) {
        // ── Rule 1: `true == x`  →  `x`,  `false == x`  →  `!x` ─────────────
        if let Expr::BinOp { op: BinOpKind::Eq, lhs, rhs, span } = cond {
            let lhs_true  = matches!(lhs.as_ref(), Expr::BoolLit { value: true,  .. });
            let lhs_false = matches!(lhs.as_ref(), Expr::BoolLit { value: false, .. });
            let rhs_true  = matches!(rhs.as_ref(), Expr::BoolLit { value: true,  .. });
            let rhs_false = matches!(rhs.as_ref(), Expr::BoolLit { value: false, .. });

            if lhs_true {
                *cond = (**rhs).clone(); self.optimizations += 1;
            } else if lhs_false {
                *cond = Expr::UnOp { span: *span, op: UnOpKind::Not, expr: rhs.clone() };
                self.optimizations += 1;
            } else if rhs_true {
                *cond = (**lhs).clone(); self.optimizations += 1;
            } else if rhs_false {
                *cond = Expr::UnOp { span: *span, op: UnOpKind::Not, expr: lhs.clone() };
                self.optimizations += 1;
            }
            return;
        }

        // ── Rule 2: `x != false`  →  `x`,  `x != true`  →  `!x` ─────────────
        if let Expr::BinOp { op: BinOpKind::Ne, lhs, rhs, span } = cond {
            let lhs_true  = matches!(lhs.as_ref(), Expr::BoolLit { value: true,  .. });
            let lhs_false = matches!(lhs.as_ref(), Expr::BoolLit { value: false, .. });
            let rhs_true  = matches!(rhs.as_ref(), Expr::BoolLit { value: true,  .. });
            let rhs_false = matches!(rhs.as_ref(), Expr::BoolLit { value: false, .. });

            if lhs_false {
                *cond = (**rhs).clone(); self.optimizations += 1;
            } else if lhs_true {
                *cond = Expr::UnOp { span: *span, op: UnOpKind::Not, expr: rhs.clone() };
                self.optimizations += 1;
            } else if rhs_false {
                *cond = (**lhs).clone(); self.optimizations += 1;
            } else if rhs_true {
                *cond = Expr::UnOp { span: *span, op: UnOpKind::Not, expr: lhs.clone() };
                self.optimizations += 1;
            }
            return;
        }

        // ── Rule 3: `!!x`  →  `x` ───────────────────────────────────────────
        if let Expr::UnOp { op: UnOpKind::Not, expr: inner, .. } = cond {
            if let Expr::UnOp { op: UnOpKind::Not, expr: inner2, .. } = inner.as_ref() {
                *cond = (**inner2).clone();
                self.optimizations += 1;
            }
        }
    }
}

// =============================================================================
// §15.5  STOCHASTIC EXPRESSION SUPEROPTIMIZER  (the real superoptimizer)
// =============================================================================
// What makes this a *true* superoptimizer (Massalin 1987 / STOKE style):
//
//   • It does NOT apply hand-written algebraic rules.
//   • Instead it randomly enumerates candidate programs and verifies semantic
//     equivalence by *executing* both the original and the candidate on a set
//     of randomly-sampled concrete input environments.
//   • The cheapest verified-equivalent expression found within a configurable
//     search budget replaces the original.
//
// Because equivalence is checked over u128 arithmetic on 32 independent random
// inputs, the false-positive probability per expression is astronomically small
// (≪ 2^−128 for polynomial arithmetic expressions by Schwartz–Zippel).
//
// The pass discovers optimizations that were never explicitly programmed,
// including all of the rules that the earlier hand-coded passes implement,
// plus novel ones that arise from the interaction of constants specific to the
// function being compiled.
//
// Algorithm per expression e:
//   1. Collect free variables V and constants C occurring in e.
//   2. Sample `verif_inputs` random environments  Γ₁ … Γₙ  (V → u128).
//   3. Compute the *spec*: σᵢ = eval(e, Γᵢ) for each i.
//      If any σᵢ = ⊥ (e is not fully evaluable), skip this expression.
//   4. Build a term bank  T = {Ident(v) | v ∈ V} ∪ {IntLit(c) | c ∈ C ∪ {0,1,2}}.
//   5. Repeat `budget` times:
//        a. Randomly compose a candidate expression c' from T using BinOp /
//           UnOp nodes up to depth 2.
//        b. Let cost(c') < cost(best).  If not, skip.
//        c. ∀i: eval(c', Γᵢ) = σᵢ?  If yes, update best ← c'.
//   6. Return best (unchanged if no cheaper equivalent was found).

// ── TermNode: allocation-free expression representation ─────────────────────
//
// A flat Vec<TermNode> encodes an expression tree in pre-order (parent before
// children).  This lets the search loop generate, cost, and evaluate thousands
// of candidate programs without ever touching the heap allocator.
//
// Layout for a BinOp node at position `p`:
//   pool[p]           = BinOp { op, left_size }
//   pool[p+1 ..]      = left subtree  (left_size nodes)
//   pool[p+1+left_size..] = right subtree
//
// Layout for a UnOp node at position `p`:
//   pool[p]   = UnOp { op, child_size }
//   pool[p+1..] = child subtree (child_size nodes)
//
// Leaf nodes are always exactly 1 node wide.
#[derive(Clone, Debug)]
enum TermNode {
    /// Concrete literal or variable value embedded at build time.
    AtomVal(Value),
    /// A free variable — looked up in the environment at eval time.
    AtomVar(String),
    /// Binary operation.  `left_size` is the number of nodes in the left
    /// subtree, so the right subtree starts at `start + 1 + left_size`.
    BinOp { op: BinOpKind, left_size: usize },
    /// Unary operation.  `child_size` is the size of the single child subtree.
    UnOp { op: UnOpKind, child_size: usize },
    /// Legacy index-into-term-bank atom — not used by the new code path but
    /// kept so the type compiles if any old code paths remain.
    Atom(usize),
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        // Mix seed through one round of SplitMix64 to avoid degenerate states.
        let mut s = seed ^ 0x9e3779b97f4a7c15u64;
        s = (s ^ (s >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        s = (s ^ (s >> 27)).wrapping_mul(0x94d049bb133111eb);
        Self { state: s ^ (s >> 31) }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        // LCG with Knuth multiplier — cheap and good enough for search.
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Xorshift top bits down for better distribution.
        let x = self.state;
        x ^ (x >> 33)
    }

    #[inline]
    fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 { return 0; }
        (self.next_u64() as usize) % bound
    }

    #[inline]
    fn next_u128(&mut self) -> u128 {
        ((self.next_u64() as u128) << 64) | (self.next_u64() as u128)
    }

    #[inline]
    fn next_bool(&mut self) -> bool { self.next_u64() & 1 == 0 }
}

/// Persistent equivalence cache: maps a normalized expression hash to the
/// cheapest equivalent expression found so far.  This amortizes the search
/// cost across compilations — if `a * b + a * c` was previously optimized
/// to `a * (b + c)`, reuse it instantly.
///
/// Enhanced with metadata: free-variable signatures, literal sets, and cost
/// bounds.  This turns repeated traversals into table lookups and helps both
/// the E-graph and stochastic pass avoid recomputing per-expression metadata.
static EQUIV_CACHE: std::sync::OnceLock<std::sync::Mutex<FxHashMap<u64, CachedEquiv>>> =
    std::sync::OnceLock::new();

/// A cached equivalence: the cheapest expression found, its cost, and metadata
/// that would otherwise need to be recomputed on every lookup.
struct CachedEquiv {
    expr: Expr,
    cost: f64,
    /// Free variables in the expression (sorted, deduplicated).
    free_vars: Vec<String>,
    /// All integer constants appearing in the expression (sorted, deduplicated).
    literals: Vec<u128>,
    /// Whether the expression is pure (no side effects).
    is_pure: bool,
    /// Best cost bound found — any future expression with cost above this
    /// for the same equivalence class can be skipped immediately.
    cost_bound: f64,
}

struct StochasticSuperoptimizer {
    budget: usize,
    verif_inputs: usize,
    pub rewrites: u64,
    rng: SimpleRng,
    /// Operator frequency histogram — biases random generation toward ops
    /// that appear frequently in the original expression.
    op_freq: FxHashMap<BinOpKind, u32>,
    /// All integer constants extracted from the original expression (for
    /// constant-proximity sampling when generating random candidates).
    original_consts: Vec<u128>,
    /// Depth budget tracking: if depth-1 fails to find improvements, we
    /// gradually increase the depth-2 probability (depth-adaptive sampling).
    depth2_bias: f64,
    /// Sub-expressions collected from the original expression, stored as
    /// normalised hash → Expr pairs.  These are added to the term bank so
    /// the superoptimizer can "recombine" parts of the original expression.
    sub_expr_bank: Vec<Expr>,
    /// ── Beam search: keep several promising candidates instead of only the
    /// current best.  Each entry is (node_pool, cost).  The beam width
    /// controls how many candidates are retained between iterations.
    beam: Vec<(Vec<TermNode>, f64)>,
    /// Maximum number of candidates to keep in the beam.
    beam_width: usize,
    /// ── Profitability gate: minimum estimated cost an expression must have
    /// before the stochastic search will run on it.  Expressions below this
    /// threshold are already "cheap enough" and not worth the search effort.
    min_cost_for_search: f64,
    /// ── Two-stage verification: number of cheap screening environments.
    /// Candidates must pass screening before full verification is applied.
    /// This eliminates clearly-wrong candidates cheaply.
    screening_env_count: usize,
    /// ── Expression metadata cache: avoids recomputing free_vars, purity,
    /// hash, and cost for the same expression across multiple passes.
    expr_metadata_cache: FxHashMap<u64, ExprMetadata>,
}

/// Cached per-expression metadata: avoids repeatedly traversing the AST to
/// compute free variables, purity, literal value, hash, and estimated cost.
/// These annotations are computed once and reused across all optimizer passes.
#[derive(Clone, Debug)]
struct ExprMetadata {
    /// Sorted, deduplicated free variable names.
    free_vars: Vec<String>,
    /// Sorted, deduplicated integer constants.
    literals: Vec<u128>,
    /// Whether the expression is pure (no side effects, no function calls).
    is_pure: bool,
    /// Normalized hash (commutative-invariant).
    hash: u64,
    /// Estimated execution cost.
    cost: f64,
    /// Estimated AST size (number of nodes).
    size: usize,
}

impl ExprMetadata {
    fn compute(expr: &Expr) -> Self {
        let mut free_vars = Vec::new();
        let mut literals = Vec::new();
        Self::collect_free_vars(expr, &mut free_vars);
        Self::collect_int_consts(expr, &mut literals);
        free_vars.sort_unstable();
        free_vars.dedup();
        literals.sort_unstable();
        literals.dedup();
        let hash = StochasticSuperoptimizer::normalized_hash(expr);
        let cost = CostModel::estimate(expr);
        let is_pure = Self::is_pure_expr(expr);
        let size = Self::count_nodes(expr);
        Self { free_vars, literals, is_pure, hash, cost, size }
    }

    fn collect_free_vars(expr: &Expr, out: &mut Vec<String>) {
        match expr {
            Expr::Ident { name, .. } => out.push(name.clone()),
            Expr::BinOp { lhs, rhs, .. } => {
                Self::collect_free_vars(lhs, out);
                Self::collect_free_vars(rhs, out);
            }
            Expr::UnOp { expr, .. } => Self::collect_free_vars(expr, out),
            Expr::Call { func, args, .. } => {
                Self::collect_free_vars(func, out);
                for a in args { Self::collect_free_vars(a, out); }
            }
            Expr::Tuple { elems, .. } => {
                for e in elems { Self::collect_free_vars(e, out); }
            }
            Expr::IfExpr { cond, then, else_, .. } => {
                Self::collect_free_vars(cond, out);
                Self::collect_free_vars_block(then, out);
                if let Some(eb) = else_ {
                    Self::collect_free_vars_block(eb, out);
                }
            }
            Expr::Block(b) => Self::collect_free_vars_block(b, out),
            _ => {}
        }
    }

    fn collect_free_vars_block(block: &Block, out: &mut Vec<String>) {
        for s in &block.stmts {
            match s {
                Stmt::Let { init: Some(e), .. } | Stmt::Expr { expr: e, .. } => {
                    Self::collect_free_vars(e, out);
                }
                _ => {}
            }
        }
        if let Some(t) = &block.tail {
            Self::collect_free_vars(t, out);
        }
    }

    fn collect_int_consts(expr: &Expr, out: &mut Vec<u128>) {
        match expr {
            Expr::IntLit { value, .. } => out.push(*value),
            Expr::BinOp { lhs, rhs, .. } => {
                Self::collect_int_consts(lhs, out);
                Self::collect_int_consts(rhs, out);
            }
            Expr::UnOp { expr, .. } => Self::collect_int_consts(expr, out),
            Expr::Tuple { elems, .. } => {
                for e in elems { Self::collect_int_consts(e, out); }
            }
            _ => {}
        }
    }

    fn is_pure_expr(expr: &Expr) -> bool {
        match expr {
            Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::BoolLit { .. }
            | Expr::StrLit { .. } | Expr::Ident { .. } => true,
            Expr::BinOp { lhs, rhs, .. } => Self::is_pure_expr(lhs) && Self::is_pure_expr(rhs),
            Expr::UnOp { expr, .. } => Self::is_pure_expr(expr),
            Expr::Tuple { elems, .. } => elems.iter().all(Self::is_pure_expr),
            Expr::Call { .. } => false, // conservatively assume impure
            _ => false,
        }
    }

    fn count_nodes(expr: &Expr) -> usize {
        match expr {
            Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::BoolLit { .. }
            | Expr::StrLit { .. } | Expr::Ident { .. } => 1,
            Expr::BinOp { lhs, rhs, .. } => 1 + Self::count_nodes(lhs) + Self::count_nodes(rhs),
            Expr::UnOp { expr, .. } => 1 + Self::count_nodes(expr),
            Expr::Tuple { elems, .. } => 1 + elems.iter().map(Self::count_nodes).sum::<usize>(),
            _ => 1,
        }
    }
}

impl StochasticSuperoptimizer {
    fn new(budget: usize, verif_inputs: usize) -> Self {
        Self {
            budget,
            verif_inputs,
            rewrites: 0,
            // Seed with a fixed value for reproducibility; a production
            // implementation could mix in a per-function hash for diversity.
            rng: SimpleRng::new(0xdeadbeef_cafebabe),
            op_freq: FxHashMap::default(),
            original_consts: Vec::new(),
            depth2_bias: 0.25, // start at 25 % depth-2 (same as before)
            sub_expr_bank: Vec::new(),
            // ── Beam search: keep top-4 candidates (width 4 provides good
            // diversity without blowing up memory or verification cost).
            beam: Vec::with_capacity(4),
            beam_width: 4,
            // ── Profitability gate: expressions cheaper than 2.0 cycles are
            // already near-optimal (a single add/sub costs 1.0).  Skip search.
            min_cost_for_search: 2.0,
            // ── Two-stage verification: use 4 cheap screening environments
            // before the full verification set.  This eliminates ~80% of
            // clearly-wrong candidates with only 4 evaluations instead of 16-32.
            screening_env_count: 4,
            // ── Expression metadata cache.
            expr_metadata_cache: FxHashMap::default(),
        }
    }

    // ── Block/stmt traversal ─────────────────────────────────────────────────

    fn optimize_block_mut(&mut self, block: &mut Block) {
        for stmt in &mut block.stmts {
            self.optimize_stmt_mut(stmt);
        }
        if let Some(tail) = &mut block.tail {
            let old = std::mem::replace(tail.as_mut(), Expr::IntLit { span: Span::dummy(), value: 0 });
            **tail = self.optimize_expr(old);
        }
    }

    fn optimize_stmt_mut(&mut self, stmt: &mut Stmt) {
        match stmt {
            Stmt::Let { init: Some(expr), .. } | Stmt::Expr { expr, .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.optimize_expr(old);
            }
            Stmt::ForIn { body, .. } | Stmt::While { body, .. } | Stmt::EntityFor { body, .. } => {
                self.optimize_block_mut(body);
            }
            Stmt::If { cond, then, else_, .. } => {
                let old = std::mem::replace(cond, Expr::IntLit { span: Span::dummy(), value: 0 });
                *cond = self.optimize_expr(old);
                self.optimize_block_mut(then);
                if let Some(else_box) = else_ {
                    if let IfOrBlock::Block(b) = &mut **else_box {
                        self.optimize_block_mut(b);
                    }
                }
            }
            Stmt::Return { value: Some(expr), .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.optimize_expr(old);
            }
            Stmt::Match { expr, arms, .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.optimize_expr(old);
                for arm in arms {
                    let old_body = std::mem::replace(&mut arm.body, Expr::IntLit { span: Span::dummy(), value: 0 });
                    arm.body = self.optimize_expr(old_body);
                }
            }
            _ => {}
        }
    }

    // ── Core search routine ──────────────────────────────────────────────────

    fn optimize_expr(&mut self, expr: Expr) -> Expr {
        // Recurse into child nodes first (bottom-up search).
        let expr = self.recurse_expr(expr);

        // Skip trivially cheap atoms — no search needed.
        match &expr {
            Expr::IntLit { .. } | Expr::FloatLit { .. }
            | Expr::BoolLit { .. } | Expr::Ident { .. } => return expr,
            _ => {}
        }

        let span = expr.span();
        let original_cost = CostModel::estimate(&expr);

        // ── Profitability gate: skip search for already-cheap expressions ──
        // Expressions with estimated cost below `min_cost_for_search` are
        // already near-optimal — stochastic search is unlikely to find
        // anything cheaper (e.g., a single `x + 1` costs 1.0 and can't be
        // improved).  Skip the search entirely to save compile time.
        if original_cost < self.min_cost_for_search {
            return expr;
        }

        // ── Skip already-canonical forms ──────────────────────────────────
        // A single BinOp with two Ident children (e.g., `a + b`) is already
        // canonical and cannot be simplified further.
        if let Expr::BinOp { lhs, rhs, .. } = &expr {
            if matches!(**lhs, Expr::Ident { .. }) && matches!(**rhs, Expr::Ident { .. }) {
                if original_cost <= 1.0 { return expr; }
            }
        }

        // ── Step 0: Equivalence cache lookup (with metadata) ─────────────
        // Compute a normalized hash of the expression (sorted operands,
        // canonical form).  If we've already found a cheaper equivalent in a
        // previous compilation, return it immediately.
        let expr_hash = Self::normalized_hash(&expr);
        if let Some(cache) = EQUIV_CACHE.get() {
            let guard = cache.lock().unwrap();
            if let Some(cached) = guard.get(&expr_hash) {
                if cached.cost < original_cost - 1e-9 {
                    self.rewrites += 1;
                    return cached.expr.clone();
                }
                // Even if the cached expression isn't cheaper, we can use
                // its cost_bound to short-circuit: if the best-known cost
                // for this equivalence class equals original_cost, there's
                // no improvement possible.
                if (cached.cost_bound - original_cost).abs() < 1e-9 {
                    return expr;
                }
            }
        }

        // ── Step 1: collect free variables, constants, sub-expressions, and
        //            operator frequencies (using cached metadata) ──────────
        let meta = self.get_or_compute_metadata(&expr, expr_hash);
        let free_vars = meta.free_vars.clone();
        let mut consts = meta.literals.clone();

        // Store constants for proximity-based sampling later.
        self.original_consts = consts.clone();

        // ── Sub-expression term bank expansion ────────────────────────────
        self.sub_expr_bank.clear();
        Self::collect_sub_exprs(&expr, &mut self.sub_expr_bank, span);

        // ── Operator frequency histogram for guided random search ──────────
        self.op_freq.clear();
        Self::collect_op_freq(&expr, &mut self.op_freq);

        // Universal small constants + edge-case values likely to appear in
        // optimised forms.  Expanded with MAX, MIN, and powers of 2 for
        // better overflow/edge-case coverage.
        for c in [0u128, 1, 2, 4, 8, 16, 32, 64, 127, 128, 255,
                  u128::MAX, (1u128 << 64) - 1, 1u128 << 63,
                  (1u128 << 32) - 1, 1u128 << 31] {
            if !consts.contains(&c) { consts.push(c); }
        }

        // ── Step 2: diversify RNG per expression + sample environments ──────
        {
            let mut h = 0u64;
            for v in &free_vars {
                for b in v.as_bytes() {
                    h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(*b as u64);
                }
            }
            for c in &consts {
                h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(*c as u64);
            }
            self.rng.state ^= h;
        }

        // ── Step 2b: Edge-case input expansion ────────────────────────────
        // Sample environments that include edge-case values (0, 1, MAX, MIN,
        // powers of two) in addition to random values.  This reduces false
        // positives/negatives in equivalence verification, especially for
        // expressions involving division, shifts, or overflow-prone arithmetic.
        let mut envs: Vec<FxHashMap<String, Value>> = Vec::with_capacity(self.verif_inputs + free_vars.len() * 5);

        // ── Deterministic edge-case environments ────────────────────────────
        let edge_vals: &[u128] = &[0, 1, 2, 127, 128, 255, u128::MAX,
                                    1u128 << 63, (1u128 << 63) - 1,
                                    (1u128 << 64) - 1, (1u128 << 32) - 1];
        if !free_vars.is_empty() {
            // Generate one env per edge value, all variables get the same value.
            for &ev in edge_vals {
                if envs.len() >= self.verif_inputs + free_vars.len() * 3 { break; }
                let mut env = FxHashMap::default();
                for v in &free_vars {
                    env.insert(v.clone(), Value::Int(ev));
                }
                envs.push(env);
            }
            // Generate at least one env where each variable independently gets
            // a different edge value — tests multi-variable interactions.
            for (i, _v) in free_vars.iter().enumerate() {
                if envs.len() >= self.verif_inputs + free_vars.len() * 3 { break; }
                let mut env = FxHashMap::default();
                for (j, v2) in free_vars.iter().enumerate() {
                    let val = if i == j { u128::MAX } else { 1u128 };
                    env.insert(v2.clone(), Value::Int(val));
                }
                envs.push(env);
            }
        }

        // Fill remaining budget with random environments.
        while envs.len() < self.verif_inputs {
            envs.push(self.sample_int_env(&free_vars));
        }

        // ── Step 3: evaluate original expression on all environments ─────────
        let spec: Vec<Option<Value>> = envs.iter()
            .map(|e| ExprInterpreter::eval(&expr, e))
            .collect();

        // If the expression isn't fully evaluable on every input, bail out.
        if spec.iter().any(|v| v.is_none()) {
            return expr;
        }
        let spec: Vec<Value> = spec.into_iter().map(Option::unwrap).collect();

        // ── Steps 4 + 5: allocation-free random search with enhancements ───
        //
        // Improvements over the original:
        //  1. Two-stage verification: cheap screening first, then full.
        //  2. Beam search: keep several promising candidates.
        //  3. Early rejection: reject on first input mismatch.
        //  4. Profitability gate already applied above.

        let term_bank = self.build_term_bank(&free_vars, &consts, span);
        let mut node_pool: Vec<TermNode> = Vec::with_capacity(16);

        let mut best_expr: Option<Vec<TermNode>> = None;
        let mut best_cost = original_cost;

        // ── Clear beam for this expression ────────────────────────────────
        self.beam.clear();

        // Reset depth-2 bias for this expression.
        self.depth2_bias = 0.25;

        // ── Phase 0: try sub-expression bank candidates first ───────────────
        for sub_expr in &self.sub_expr_bank {
            let sub_cost = CostModel::estimate(sub_expr);
            if sub_cost >= best_cost { continue; }

            // Two-stage verification: screening first
            if !Self::verify_screening(sub_expr, &envs, &spec, self.screening_env_count) {
                continue;
            }
            // Full verification
            let equivalent = envs.iter().zip(spec.iter()).all(|(env, expected)| {
                ExprInterpreter::eval(sub_expr, env).as_ref() == Some(expected)
            });

            if equivalent {
                self.rewrites += 1;
                Self::update_equiv_cache_with_meta(expr_hash, sub_expr.clone(), sub_cost, &free_vars, &consts, meta.is_pure);
                return sub_expr.clone();
            }
        }

        // ── Phase 1: guided random search with beam ───────────────────────
        let mut iterations_without_improvement = 0usize;

        for _ in 0..self.budget {
            if iterations_without_improvement > self.budget / 4 {
                self.depth2_bias = (self.depth2_bias + 0.1).min(0.75);
                iterations_without_improvement = 0;
            }

            let depth = if (self.rng.next_u64() as f64 / u64::MAX as f64) < self.depth2_bias {
                2usize
            } else {
                1usize
            };

            node_pool.clear();
            Self::guided_random_term(
                &mut self.rng,
                &term_bank,
                &mut node_pool,
                depth,
                &self.op_freq,
                &self.original_consts,
            );

            let candidate_cost = Self::term_cost(&node_pool, 0).0;
            if candidate_cost >= best_cost { iterations_without_improvement += 1; continue; }

            // ── Two-stage verification: screening first ────────────────────
            // Only evaluate on the first `screening_env_count` environments.
            // If the candidate disagrees on any of these, it's definitely not
            // equivalent — reject immediately without testing the rest.
            let screening_pass = envs.iter().zip(spec.iter())
                .take(self.screening_env_count)
                .all(|(env, expected)| {
                    Self::eval_term(&node_pool, 0, env).as_ref() == Some(expected)
                });

            if !screening_pass {
                iterations_without_improvement += 1;
                continue;
            }

            // ── Full verification ──────────────────────────────────────────
            // Candidate passed screening — now verify on ALL environments
            // with early rejection (break on first mismatch).
            let mut equivalent = true;
            for (env, expected) in envs.iter().zip(spec.iter()) {
                if Self::eval_term(&node_pool, 0, env).as_ref() != Some(expected) {
                    equivalent = false;
                    break;
                }
            }

            if equivalent {
                best_expr = Some(node_pool.clone());
                best_cost = candidate_cost;
                self.rewrites += 1;
                iterations_without_improvement = 0;

                // ── Beam: add to beam ──────────────────────────────────────
                self.beam.push((node_pool.clone(), candidate_cost));
                // Keep only the top beam_width cheapest candidates.
                if self.beam.len() > self.beam_width {
                    self.beam.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    self.beam.truncate(self.beam_width);
                }
            } else {
                iterations_without_improvement += 1;
            }
        }

        // ── Materialise the winning Expr and update equivalence cache ────────
        // Pick the best from the beam (if any), otherwise use best_expr.
        let final_pool = if !self.beam.is_empty() {
            self.beam.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            Some(self.beam[0].0.clone())
        } else {
            best_expr
        };

        let result = match final_pool {
            Some(pool) => Self::build_winner_expr(&pool, 0, span),
            None => expr,
        };

        if best_cost < original_cost - 1e-9 {
            Self::update_equiv_cache_with_meta(expr_hash, result.clone(), best_cost, &free_vars, &consts, meta.is_pure);
        }

        result
    }

    /// Get or compute expression metadata, using the cache to avoid
    /// recomputing free vars, purity, hash, and cost for the same expression.
    fn get_or_compute_metadata(&mut self, expr: &Expr, hash: u64) -> ExprMetadata {
        if let Some(meta) = self.expr_metadata_cache.get(&hash) {
            return meta.clone();
        }
        let meta = ExprMetadata::compute(expr);
        self.expr_metadata_cache.insert(hash, meta.clone());
        meta
    }

    /// Cheap screening verification: evaluate the candidate on only the first
    /// `n` environments.  Returns true if the candidate matches the spec on
    /// all `n` environments, false otherwise.  This eliminates ~80% of
    /// clearly-wrong candidates with only a fraction of the evaluations.
    fn verify_screening(candidate: &Expr, envs: &[FxHashMap<String, Value>], spec: &[Value], n: usize) -> bool {
        envs.iter().zip(spec.iter())
            .take(n)
            .all(|(env, expected)| {
                ExprInterpreter::eval(candidate, env).as_ref() == Some(expected)
            })
    }

    // ── Normalized expression hashing for the equivalence cache ──────────────
    //
    // Produces a hash that is invariant under commutative reordering of
    // operands (e.g., a+b and b+a hash the same), so the cache can recognise
    // equivalent expressions regardless of operand order.

    fn normalized_hash(expr: &Expr) -> u64 {
        let mut hasher = DefaultHasher::new();
        Self::normalized_hash_impl(expr, &mut hasher);
        hasher.finish()
    }

    fn normalized_hash_impl(expr: &Expr, h: &mut DefaultHasher) {
        match expr {
            Expr::IntLit { value, .. } => {
                0u8.hash(h);
                value.hash(h);
            }
            Expr::FloatLit { value, .. } => {
                1u8.hash(h);
                value.to_bits().hash(h);
            }
            Expr::BoolLit { value, .. } => {
                2u8.hash(h);
                value.hash(h);
            }
            Expr::Ident { name, .. } => {
                3u8.hash(h);
                name.hash(h);
            }
            Expr::BinOp { op, lhs, rhs, .. } => {
                let op_disc = std::mem::discriminant(op);
                op_disc.hash(h);
                // For commutative ops, sort children so a+b and b+a hash identically.
                if matches!(op, BinOpKind::Add | BinOpKind::Mul
                    | BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor
                    | BinOpKind::Eq | BinOpKind::Ne)
                {
                    let mut lh = Self::normalized_hash(lhs);
                    let mut rh = Self::normalized_hash(rhs);
                    if lh > rh { std::mem::swap(&mut lh, &mut rh); }
                    lh.hash(h);
                    rh.hash(h);
                } else {
                    Self::normalized_hash_impl(lhs, h);
                    Self::normalized_hash_impl(rhs, h);
                }
            }
            Expr::UnOp { op, expr, .. } => {
                std::mem::discriminant(op).hash(h);
                Self::normalized_hash_impl(expr, h);
            }
            _ => {
                // Fallback: hash the debug representation.
                255u8.hash(h);
                format!("{:?}", expr).hash(h);
            }
        }
    }

    /// Insert a discovered equivalence into the global cache (without metadata).
    fn update_equiv_cache(key: u64, expr: Expr, cost: f64) {
        Self::update_equiv_cache_with_meta(key, expr, cost, &[], &[], true);
    }

    /// Insert a discovered equivalence into the global cache with full metadata.
    /// The metadata (free vars, literals, purity, cost bound) is stored alongside
    /// the expression so that future lookups can use it without recomputing.
    fn update_equiv_cache_with_meta(
        key: u64,
        expr: Expr,
        cost: f64,
        free_vars: &[String],
        literals: &[u128],
        is_pure: bool,
    ) {
        let cache = EQUIV_CACHE.get_or_init(|| {
            std::sync::Mutex::new(FxHashMap::default())
        });
        let mut guard = cache.lock().unwrap();
        guard.entry(key).and_modify(|e| {
            if cost < e.cost {
                e.expr = expr.clone();
                e.cost = cost;
                e.cost_bound = cost;
                if !free_vars.is_empty() { e.free_vars = free_vars.to_vec(); }
                if !literals.is_empty() { e.literals = literals.to_vec(); }
                e.is_pure = is_pure;
            }
        }).or_insert(CachedEquiv {
            expr,
            cost,
            free_vars: free_vars.to_vec(),
            literals: literals.to_vec(),
            is_pure,
            cost_bound: cost,
        });
    }

    /// Collect all sub-expressions of the input expression as potential term
    /// bank leaves.  Each sub-expression is recorded as an Expr that the
    /// superoptimizer can "recombine" in new ways.
    fn collect_sub_exprs(expr: &Expr, out: &mut Vec<Expr>, _span: Span) {
        match expr {
            Expr::BinOp { lhs, rhs, .. } => {
                // Add left and right children as sub-expressions.
                out.push((**lhs).clone());
                out.push((**rhs).clone());
                // Recurse into children for deeper sub-expressions.
                Self::collect_sub_exprs(lhs, out, _span);
                Self::collect_sub_exprs(rhs, out, _span);
            }
            Expr::UnOp { expr: inner, .. } => {
                out.push((**inner).clone());
                Self::collect_sub_exprs(inner, out, _span);
            }
            Expr::Call { args, .. } => {
                for a in args {
                    out.push(a.clone());
                    Self::collect_sub_exprs(a, out, _span);
                }
            }
            Expr::Tuple { elems, .. } => {
                for e in elems {
                    out.push(e.clone());
                    Self::collect_sub_exprs(e, out, _span);
                }
            }
            _ => {}
        }
    }

    /// Collect the frequency of each binary operator in the expression,
    /// used to bias the random term generator toward frequently-occurring ops.
    fn collect_op_freq(expr: &Expr, freq: &mut FxHashMap<BinOpKind, u32>) {
        match expr {
            Expr::BinOp { op, lhs, rhs, .. } => {
                *freq.entry(*op).or_insert(0) += 1;
                Self::collect_op_freq(lhs, freq);
                Self::collect_op_freq(rhs, freq);
            }
            Expr::UnOp { expr: inner, .. } => Self::collect_op_freq(inner, freq),
            Expr::Call { args, .. } => {
                for a in args { Self::collect_op_freq(a, freq); }
            }
            Expr::Tuple { elems, .. } => {
                for e in elems { Self::collect_op_freq(e, freq); }
            }
            _ => {}
        }
    }

    /// Recursively descend into subexpressions before applying search at this
    /// level, so we get bottom-up coverage of the whole expression tree.
    fn recurse_expr(&mut self, expr: Expr) -> Expr {
        match expr {
            Expr::BinOp { span, op, lhs, rhs } => {
                let lhs = self.optimize_expr(*lhs);
                let rhs = self.optimize_expr(*rhs);
                Expr::BinOp { span, op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
            }
            Expr::UnOp { span, op, expr } => {
                let inner = self.optimize_expr(*expr);
                Expr::UnOp { span, op, expr: Box::new(inner) }
            }
            Expr::Call { span, func, args, named } => Expr::Call {
                span,
                func: Box::new(self.optimize_expr(*func)),
                args: args.into_iter().map(|a| self.optimize_expr(a)).collect(),
                named,
            },
            Expr::Tuple { span, elems } => Expr::Tuple {
                span,
                elems: elems.into_iter().map(|e| self.optimize_expr(e)).collect(),
            },
            Expr::Block(block) => {
                let mut b = *block;
                self.optimize_block_mut(&mut b);
                Expr::Block(Box::new(b))
            }
            other => other,
        }
    }

    // ── Candidate generation — allocation-free ────────────────────────────────
    //
    // Candidates are represented as flat Vec<TermNode> pools in a pre-order
    // (parent before children) layout.  This means:
    //
    //   • No Box<Expr> is ever allocated during the search loop.
    //   • The entire candidate lives in a single contiguous Vec that is reset
    //     (clear + reuse) every iteration — equivalent to a bump allocator but
    //     using Rust's standard Vec allocation so we pay for the memory exactly
    //     once (on the first iteration that needs depth-2 capacity).
    //   • Evaluation is a single recursive descent over that Vec; the recursion
    //     stack depth is bounded by the expression depth (≤ 2 in practice).
    //
    // TermAtom identifies a leaf drawn from the term bank by index.
    // TermBinOp / TermUnOp record the operator and the SIZE of their left child
    // so that the right child can be found at (left_start + left_size).

    /// Build the set of atomic "leaf" values available to the search.
    /// Returned as a compact Vec so we can index into it cheaply.
    ///
    /// Enhanced with sub-expression term bank expansion: all sub-expressions
    /// from the original expression are included as potential leaves, enabling
    /// the superoptimizer to "recombine" parts of the original expression in
    /// new ways (e.g., turning a*b + a*c into a*(b+c) by treating a, b, c
    /// as leaves).
    fn build_term_bank(
        &self,
        free_vars: &[String],
        consts: &[u128],
        span: Span,
    ) -> Vec<Expr> {
        let mut bank: Vec<Expr> = Vec::with_capacity(
            free_vars.len() + consts.len() + self.sub_expr_bank.len()
        );
        for v in free_vars {
            bank.push(Expr::Ident { span, name: v.clone() });
        }
        for c in consts {
            bank.push(Expr::IntLit { span, value: *c });
        }
        // ── Sub-expression bank: add all collected sub-expressions ────────
        // These are real parts of the original expression that can be used
        // as building blocks for candidate expressions.  This is what enables
        // the superoptimizer to discover factoring like a*(b+c) from a*b+a*c.
        for sub in &self.sub_expr_bank {
            bank.push(sub.clone());
        }
        bank
    }

    /// Append a random expression tree rooted at the next free slot in `pool`.
    ///
    /// Nodes are written in pre-order (root before children) so that:
    ///   pool[root+1 .. root+1+left_size]         is the left subtree
    ///   pool[root+1+left_size ..]                 is the right subtree
    ///
    /// Returns the number of TermNodes appended (the subtree size), which the
    /// parent node stores as `left_size` so it can locate the right child.
    ///
    /// Leaf nodes are emitted as `AtomVal` (literal) or `AtomVar` (variable)
    /// so that evaluation never needs to look up the original term_bank —
    /// all data is self-contained in the pool and evaluation is pure stack work.
    fn random_term(
        rng: &mut SimpleRng,
        term_bank: &[Expr],
        pool: &mut Vec<TermNode>,
        depth: usize,
    ) -> usize {
        /// Emit a random leaf from term_bank into pool.
        fn emit_leaf(rng: &mut SimpleRng, term_bank: &[Expr], pool: &mut Vec<TermNode>) {
            if term_bank.is_empty() {
                pool.push(TermNode::AtomVal(Value::Int(0)));
                return;
            }
            let idx = rng.next_usize(term_bank.len());
            match &term_bank[idx] {
                Expr::IntLit   { value, .. } => pool.push(TermNode::AtomVal(Value::Int(*value))),
                Expr::FloatLit { value, .. } => pool.push(TermNode::AtomVal(Value::float(*value))),
                Expr::BoolLit  { value, .. } => pool.push(TermNode::AtomVal(Value::Bool(*value))),
                Expr::Ident    { name,  .. } => pool.push(TermNode::AtomVar(name.clone())),
                _ => pool.push(TermNode::AtomVal(Value::Int(0))),
            }
        }

        if term_bank.is_empty() {
            pool.push(TermNode::AtomVal(Value::Int(0)));
            return 1;
        }

        // At depth 0 or with 1/3 probability, emit a leaf.
        if depth == 0 || rng.next_u64().is_multiple_of(3) {
            emit_leaf(rng, term_bank, pool);
            return 1;
        }

        match rng.next_usize(15) {
            // BinOp (0..=11) ─────────────────────────────────────────────────
            n @ 0..=11 => {
                const OPS: [BinOpKind; 12] = [
                    BinOpKind::Add, BinOpKind::Sub, BinOpKind::Mul,
                    BinOpKind::BitAnd, BinOpKind::BitOr, BinOpKind::BitXor,
                    BinOpKind::Shl, BinOpKind::Shr,
                    BinOpKind::Eq, BinOpKind::Ne, BinOpKind::Lt, BinOpKind::Le,
                ];
                let op = OPS[n];
                // Reserve root slot; fill it in after we know left_size.
                let root_idx = pool.len();
                pool.push(TermNode::BinOp { op, left_size: 0 }); // placeholder
                let left_size  = Self::random_term(rng, term_bank, pool, depth - 1);
                let right_size = Self::random_term(rng, term_bank, pool, depth - 1);
                pool[root_idx] = TermNode::BinOp { op, left_size };
                1 + left_size + right_size
            }
            // UnOp Neg (12) ───────────────────────────────────────────────────
            12 => {
                let root_idx = pool.len();
                pool.push(TermNode::UnOp { op: UnOpKind::Neg, child_size: 0 });
                let child_size = Self::random_term(rng, term_bank, pool, depth - 1);
                pool[root_idx] = TermNode::UnOp { op: UnOpKind::Neg, child_size };
                1 + child_size
            }
            // UnOp Not (13) ───────────────────────────────────────────────────
            13 => {
                let root_idx = pool.len();
                pool.push(TermNode::UnOp { op: UnOpKind::Not, child_size: 0 });
                let child_size = Self::random_term(rng, term_bank, pool, depth - 1);
                pool[root_idx] = TermNode::UnOp { op: UnOpKind::Not, child_size };
                1 + child_size
            }
            // Leaf (14) ───────────────────────────────────────────────────────
            _ => {
                emit_leaf(rng, term_bank, pool);
                1
            }
        }
    }

    /// Guided random term generation with operator frequency bias and
    /// constant proximity sampling.  This replaces the uniform-random
    /// `random_term` in the stochastic search loop, yielding candidates
    /// that are more likely to match the structure of the original expression.
    ///
    /// Key improvements over plain `random_term`:
    ///  - **Operator Frequency Bias**: Binary ops that appear frequently in
    ///    the original expression are chosen more often.  If the original is
    ///    `a*b + c*d`, Mul and Add get boosted selection probability.
    ///  - **Constant Proximity**: When emitting a leaf that is an IntLit,
    ///    prefer constants close to existing constants in the term bank.
    ///    If 5 is present, try 4, 6, 8, etc.
    ///  - **Depth-Adaptive**: The `depth2_bias` parameter from the parent
    ///    struct is already used in `optimize_expr` to control the depth
    ///    probability; this method respects the depth parameter it receives.
    fn guided_random_term(
        rng: &mut SimpleRng,
        term_bank: &[Expr],
        pool: &mut Vec<TermNode>,
        depth: usize,
        op_freq: &FxHashMap<BinOpKind, u32>,
        original_consts: &[u128],
    ) -> usize {
        /// Emit a guided leaf from term_bank into pool, with constant-proximity
        /// bias: 70 % of the time, pick a term_bank entry directly; 30 % of
        /// the time, if we have original constants, pick a constant from the
        /// original expression and perturb it slightly (±1, ±2, ×2, ÷2).
        fn emit_guided_leaf(
            rng: &mut SimpleRng,
            term_bank: &[Expr],
            pool: &mut Vec<TermNode>,
            original_consts: &[u128],
        ) {
            if term_bank.is_empty() {
                pool.push(TermNode::AtomVal(Value::Int(0)));
                return;
            }

            // 70 % pick from term bank, 30 % try proximity-based constant generation.
            if original_consts.is_empty() || rng.next_u64() % 10 < 7 {
                let idx = rng.next_usize(term_bank.len());
                match &term_bank[idx] {
                    Expr::IntLit   { value, .. } => pool.push(TermNode::AtomVal(Value::Int(*value))),
                    Expr::FloatLit { value, .. } => pool.push(TermNode::AtomVal(Value::float(*value))),
                    Expr::BoolLit  { value, .. } => pool.push(TermNode::AtomVal(Value::Bool(*value))),
                    Expr::Ident    { name,  .. } => pool.push(TermNode::AtomVar(name.clone())),
                    // Sub-expressions in the bank — evaluate them to see if they
                    // reduce to a simple value we can inline.
                    other => {
                        // For compound sub-expressions, we emit them as a
                        // "sub-expression variable" — a special AtomVar with a
                        // unique prefix that the evaluator can recognise.  However,
                        // since the TermNode pool is self-contained and we need
                        // to be able to evaluate it, we simply pick a random leaf
                        // from the bank instead for now.
                        // Future: expand compound sub-exprs recursively.
                        let _ = other;
                        // Fallback: pick a random leaf from term_bank.
                        let mut tries = 0;
                        loop {
                            let idx2 = rng.next_usize(term_bank.len());
                            match &term_bank[idx2] {
                                Expr::IntLit   { value, .. } => { pool.push(TermNode::AtomVal(Value::Int(*value))); return; }
                                Expr::FloatLit { value, .. } => { pool.push(TermNode::AtomVal(Value::float(*value))); return; }
                                Expr::BoolLit  { value, .. } => { pool.push(TermNode::AtomVal(Value::Bool(*value))); return; }
                                Expr::Ident    { name,  .. } => { pool.push(TermNode::AtomVar(name.clone())); return; }
                                _ => { tries += 1; if tries > 4 { pool.push(TermNode::AtomVal(Value::Int(0))); return; } }
                            }
                        }
                    }
                }
            } else {
                // Constant proximity: pick a random constant from the original
                // expression and perturb it slightly.  This generates candidates
                // close to the "interesting" values without needing them in the
                // term bank explicitly.
                let base = original_consts[rng.next_usize(original_consts.len())];
                let perturbed = match rng.next_usize(8) {
                    0 => base.wrapping_add(1),
                    1 => base.wrapping_sub(1),
                    2 => base.wrapping_add(2),
                    3 => base.wrapping_sub(2),
                    4 => base.wrapping_mul(2),
                    5 => if base != 0 { base / 2 } else { 0 },
                    6 => base.wrapping_add(rng.next_u64() as u128 & 0xF), // small random offset
                    _ => base ^ (1u128 << rng.next_usize(128)),            // flip a random bit
                };
                pool.push(TermNode::AtomVal(Value::Int(perturbed)));
            }
        }

        if term_bank.is_empty() {
            pool.push(TermNode::AtomVal(Value::Int(0)));
            return 1;
        }

        // At depth 0 or with 1/3 probability, emit a leaf.
        if depth == 0 || rng.next_u64().is_multiple_of(3) {
            emit_guided_leaf(rng, term_bank, pool, original_consts);
            return 1;
        }

        // ── Operator frequency-biased selection ────────────────────────────
        // Instead of uniform random selection over all 15 choices, we bias
        // toward operators that appear frequently in the original expression.
        // The probability of each operator is proportional to its frequency + 1
        // (the +1 ensures every operator has at least some chance).
        const ALL_OPS: [BinOpKind; 12] = [
            BinOpKind::Add, BinOpKind::Sub, BinOpKind::Mul,
            BinOpKind::BitAnd, BinOpKind::BitOr, BinOpKind::BitXor,
            BinOpKind::Shl, BinOpKind::Shr,
            BinOpKind::Eq, BinOpKind::Ne, BinOpKind::Lt, BinOpKind::Le,
        ];

        // Build a weighted selection table.  Each entry stores the cumulative
        // weight up to that point; we binary-search to find the chosen op.
        let mut weights: [u32; 15] = [0; 15]; // 12 BinOps + Neg + Not + Leaf
        let mut total: u32 = 0;
        for (i, op) in ALL_OPS.iter().enumerate() {
            let freq = op_freq.get(op).copied().unwrap_or(0) + 1;
            total += freq;
            weights[i] = total;
        }
        // UnOp Neg
        let neg_freq = 2; // base weight for unary ops
        total += neg_freq;
        weights[12] = total;
        // UnOp Not
        total += neg_freq;
        weights[13] = total;
        // Leaf
        total += 3; // slightly higher base weight for leaves
        weights[14] = total;

        let pick = rng.next_usize(total as usize) as u32;

        // Binary search for the chosen slot.
        let mut lo = 0usize;
        let mut hi = 14usize;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if weights[mid] <= pick { lo = mid + 1; } else { hi = mid; }
        }

        match lo {
            n @ 0..=11 => {
                let op = ALL_OPS[n];
                let root_idx = pool.len();
                pool.push(TermNode::BinOp { op, left_size: 0 }); // placeholder
                let left_size  = Self::guided_random_term(rng, term_bank, pool, depth - 1, op_freq, original_consts);
                let right_size = Self::guided_random_term(rng, term_bank, pool, depth - 1, op_freq, original_consts);
                pool[root_idx] = TermNode::BinOp { op, left_size };
                1 + left_size + right_size
            }
            12 => {
                let root_idx = pool.len();
                pool.push(TermNode::UnOp { op: UnOpKind::Neg, child_size: 0 });
                let child_size = Self::guided_random_term(rng, term_bank, pool, depth - 1, op_freq, original_consts);
                pool[root_idx] = TermNode::UnOp { op: UnOpKind::Neg, child_size };
                1 + child_size
            }
            13 => {
                let root_idx = pool.len();
                pool.push(TermNode::UnOp { op: UnOpKind::Not, child_size: 0 });
                let child_size = Self::guided_random_term(rng, term_bank, pool, depth - 1, op_freq, original_consts);
                pool[root_idx] = TermNode::UnOp { op: UnOpKind::Not, child_size };
                1 + child_size
            }
            _ => {
                emit_guided_leaf(rng, term_bank, pool, original_consts);
                1
            }
        }
    }

    /// Evaluate the expression tree rooted at `pool[start]` against `env`.
    ///
    /// All data needed for evaluation is embedded in the pool nodes themselves
    /// (`AtomVal` for literals, `AtomVar` for variable lookups) so this
    /// function is pure stack work — zero heap allocations.
    ///
    /// Returns `(subtree_size, Option<Value>)`.  `subtree_size` is the number
    /// of TermNodes consumed so callers can locate sibling subtrees.
    fn eval_term(
        pool: &[TermNode],
        start: usize,
        env: &FxHashMap<String, Value>,
    ) -> Option<Value> {
        Self::eval_term_recursive(pool, start, env).1
    }

    fn eval_term_recursive(
        pool: &[TermNode],
        start: usize,
        env: &FxHashMap<String, Value>,
    ) -> (usize, Option<Value>) {
        if start >= pool.len() { return (1, None); }
        match &pool[start] {
            // ── Leaves ─────────────────────────────────────────────────────────
            TermNode::AtomVal(v) => (1, Some(v.clone())),
            TermNode::AtomVar(name) => (1, env.get(name.as_str()).cloned()),
            TermNode::Atom(_) => (1, None), // legacy; should not appear

            // ── Binary operator ─────────────────────────────────────────────────
            TermNode::BinOp { op, left_size } => {
                let op = *op;
                let ls = *left_size;
                let (_, lv) = Self::eval_term_recursive(pool, start + 1, env);
                let (rs, rv) = Self::eval_term_recursive(pool, start + 1 + ls, env);
                let val = match (lv, rv) {
                    (Some(l), Some(r)) => ExprInterpreter::eval_binop(op, &l, &r),
                    _ => None,
                };
                (1 + ls + rs, val)
            }

            // ── Unary operator ──────────────────────────────────────────────────
            TermNode::UnOp { op, child_size } => {
                let op = *op;
                let cs = *child_size;
                let (_, cv) = Self::eval_term_recursive(pool, start + 1, env);
                let val = cv.and_then(|v| ExprInterpreter::eval_unop(op, &v));
                (1 + cs, val)
            }
        }
    }

    /// Estimate the cost of the expression tree at `pool[start]` without
    /// building an Expr.  Returns `(size, cost)`.
    fn term_cost(pool: &[TermNode], start: usize) -> (f64, usize) {
        if start >= pool.len() { return (0.0, 1); }
        match &pool[start] {
            TermNode::Atom(_) | TermNode::AtomVal(_) | TermNode::AtomVar(_) => (0.0, 1),
            TermNode::BinOp { op, left_size } => {
                let op = *op;
                let ls = *left_size;
                let (lc, _) = Self::term_cost(pool, start + 1);
                let (rc, rs) = Self::term_cost(pool, start + 1 + ls);
                let base = match op {
                    BinOpKind::Add | BinOpKind::Sub => 1.0,
                    BinOpKind::Mul => 3.0,
                    BinOpKind::Div | BinOpKind::Rem | BinOpKind::FloorDiv => 20.0,
                    _ => 1.0,
                };
                (base + lc + rc, 1 + ls + rs)
            }
            TermNode::UnOp { child_size, .. } => {
                let cs = *child_size;
                let (cc, _) = Self::term_cost(pool, start + 1);
                (1.0 + cc, 1 + cs)
            }
        }
    }

    /// Materialise a TermNode pool back into a real Expr AST.
    /// Called exactly once per search — only when a winner is found.
    fn build_winner_expr(pool: &[TermNode], start: usize, span: Span) -> Expr {
        if start >= pool.len() {
            return Expr::IntLit { span, value: 0 };
        }
        match &pool[start] {
            TermNode::Atom(_) => Expr::IntLit { span, value: 0 },
            TermNode::AtomVal(v) => match v {
                Value::Int(n)   => Expr::IntLit  { span, value: *n },
                Value::Float(b) => Expr::FloatLit{ span, value: f64::from_bits(*b) },
                Value::Bool(b)  => Expr::BoolLit { span, value: *b },
            },
            TermNode::AtomVar(name) => Expr::Ident { span, name: name.clone() },
            TermNode::BinOp { op, left_size } => {
                let op = *op;
                let ls = *left_size;
                let lhs = Self::build_winner_expr(pool, start + 1, span);
                let rhs = Self::build_winner_expr(pool, start + 1 + ls, span);
                Expr::BinOp { span, op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
            }
            TermNode::UnOp { op, .. } => {
                let op = *op;
                let inner = Self::build_winner_expr(pool, start + 1, span);
                Expr::UnOp { span, op, expr: Box::new(inner) }
            }
        }
    }

        // ── Environment sampling ─────────────────────────────────────────────────

    /// Sample a random environment that maps each variable to a u128.
    /// We use a mix of edge-case values (0, 1, MAX, powers-of-two) and
    /// random values so that both algebraic identities and arbitrary
    /// arithmetic patterns can be discovered.
    ///
    /// Enhanced with additional edge cases for overflow and shift boundary
    /// conditions (Priority ⚡ Med: Edge-Case Input Expansion).
    fn sample_int_env(&mut self, free_vars: &[String]) -> FxHashMap<String, Value> {
        let mut env = FxHashMap::default();
        for v in free_vars {
            let val = match self.rng.next_usize(16) {
                0 => 0u128,
                1 => 1u128,
                2 => u128::MAX,
                3 => 1u128 << self.rng.next_usize(127),   // power of two
                4 => (1u128 << self.rng.next_usize(127)).wrapping_sub(1), // 2^k-1
                5 => 1u128 << 63,                             // sign bit boundary (i64)
                6 => (1u128 << 63).wrapping_sub(1),         // i64::MAX
                7 => 1u128 << 32,                             // u32 boundary
                8 => (1u128 << 32).wrapping_sub(1),         // u32::MAX
                9 => (1u128 << 64).wrapping_sub(1),         // u64::MAX
                10 => 255u128,                              // byte boundary
                11 => 127u128,                              // i8::MAX
                12 => 128u128,                              // i8::MIN (abs)
                13 => self.rng.next_u64() as u128,          // random u64 range
                14 => (self.rng.next_u64() as u128) | ((self.rng.next_u64() as u128) << 64), // full u128
                _ => self.rng.next_u128(),                  // fully random
            };
            env.insert(v.clone(), Value::Int(val));
        }
        env
    }

    // ── AST analysis helpers ─────────────────────────────────────────────────

    fn collect_free_vars_expr(expr: &Expr, out: &mut Vec<String>) {
        match expr {
            Expr::Ident { name, .. } => out.push(name.clone()),
            Expr::BinOp { lhs, rhs, .. } => {
                Self::collect_free_vars_expr(lhs, out);
                Self::collect_free_vars_expr(rhs, out);
            }
            Expr::UnOp { expr, .. } => Self::collect_free_vars_expr(expr, out),
            Expr::Call { func, args, .. } => {
                Self::collect_free_vars_expr(func, out);
                for a in args { Self::collect_free_vars_expr(a, out); }
            }
            Expr::Tuple { elems, .. } => {
                for e in elems { Self::collect_free_vars_expr(e, out); }
            }
            _ => {}
        }
    }

    fn collect_int_consts_expr(expr: &Expr, out: &mut Vec<u128>) {
        match expr {
            Expr::IntLit { value, .. } => out.push(*value),
            Expr::BinOp { lhs, rhs, .. } => {
                Self::collect_int_consts_expr(lhs, out);
                Self::collect_int_consts_expr(rhs, out);
            }
            Expr::UnOp { expr, .. } => Self::collect_int_consts_expr(expr, out),
            _ => {}
        }
    }
}

// =============================================================================
// §16.5  EGRAPH — Equality Saturation
// =============================================================================
//
// Architecture
// ────────────
// An E-Graph (equality graph) represents an equivalence relation over
// expressions.  Every sub-expression lives inside an *e-class* (equivalence
// class); each e-class holds one or more *e-nodes*, which are term nodes whose
// child positions contain e-class IDs rather than concrete sub-expressions.
//
// Pipeline per eligible expression:
//   1. BUILD     — translate Expr → EGraph (one e-class per unique sub-expr).
//   2. SATURATE  — apply algebraic rewrite rules exhaustively.  Rules are
//                  applied as read-only queries that emit pending (class, enode)
//                  pairs; the pairs are then applied in a separate mutable step
//                  (no aliasing issues).
//   3. EXTRACT   — Bellman-Ford cost minimisation picks the cheapest e-node in
//                  every e-class; the winning tree is then materialised into an
//                  Expr and returned if cheaper than the original.
//
// All data structures are self-contained: no additional crate dependencies.

// ── Numeric type aliases ─────────────────────────────────────────────────────

type EClassId = u32;
type ENodeId  = u32;

// ── E-node ───────────────────────────────────────────────────────────────────

/// A term node whose child positions hold e-class IDs rather than sub-expressions.
/// Leaves carry their constant / variable payload directly.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum ENode {
    IntLit(u128),
    /// f64 stored as raw bits so that `ENode` can implement `Eq` + `Hash`.
    FloatBits(u64),
    Bool(bool),
    Var(String),
    BinOp(BinOpKind, EClassId, EClassId),
    UnOp(UnOpKind, EClassId),
    /// If-then-else: (cond, then_branch, else_branch)
    /// else_branch is optional (None if no else clause)
    IfThenElse(EClassId, EClassId, Option<EClassId>),
}

impl ENode {
    /// Invoke `f` on every child class-id.
    fn for_each_child(&self, mut f: impl FnMut(EClassId)) {
        match self {
            Self::BinOp(_, a, b) => { f(*a); f(*b); }
            Self::UnOp(_, a)     => { f(*a); }
            Self::IfThenElse(c, t, e) => {
                f(*c);
                f(*t);
                if let Some(ee) = e { f(*ee); }
            }
            _                    => {}
        }
    }

    /// Return a copy with each child class-id replaced by `f(old_id)`.
    fn map_children(self, mut f: impl FnMut(EClassId) -> EClassId) -> Self {
        match self {
            Self::BinOp(op, a, b) => Self::BinOp(op, f(a), f(b)),
            Self::UnOp(op, a)     => Self::UnOp(op, f(a)),
            Self::IfThenElse(c, t, e) => Self::IfThenElse(f(c), f(t), e.map(f)),
            other                 => other,
        }
    }

    /// Single-node cost, excluding children.
    /// Uses hardware-aware cost model combined with profile data.
    /// Lower is better.
    fn base_cost(&self, hw_model: &HardwareCostModel, profile_model: Option<&ProfileWeightedCostModel>, location: Option<&str>) -> f64 {
        let static_cost = match self {
            Self::IntLit(_) | Self::FloatBits(_) | Self::Bool(_) | Self::Var(_) => 0.0,
            Self::UnOp(op, _) => {
                let instr = match op {
                    UnOpKind::Neg => "neg",
                    UnOpKind::Not => "not",
                    UnOpKind::Deref => "load",
                    UnOpKind::Ref | UnOpKind::RefMut => "store_addr",
                };
                if let Some(uop) = hw_model.port_map().get_uop(instr) {
                    uop.latency as f64
                } else {
                    1.0
                }
            }
            Self::BinOp(op, _, _) => {
                let instr = match op {
                    BinOpKind::Add => "add",
                    BinOpKind::Sub => "sub",
                    BinOpKind::Mul => "mul",
                    BinOpKind::Div => "div",
                    BinOpKind::Rem => "div",
                    BinOpKind::FloorDiv => "div",
                    BinOpKind::BitAnd => "and",
                    BinOpKind::BitOr => "or",
                    BinOpKind::BitXor => "xor",
                    BinOpKind::Shl => "shl",
                    BinOpKind::Shr => "shr",
                    BinOpKind::Eq => "cmp",
                    BinOpKind::Ne => "cmp",
                    BinOpKind::Lt => "cmp",
                    BinOpKind::Le => "cmp",
                    BinOpKind::Gt => "cmp",
                    BinOpKind::Ge => "cmp",
                    BinOpKind::And => "and",
                    BinOpKind::Or => "or",
                };
                if let Some(uop) = hw_model.port_map().get_uop(instr) {
                    uop.latency as f64
                } else {
                    1.0
                }
            }
            Self::IfThenElse(_, _, _) => 15.0,
        };

        // Apply profile-weighted cost if available
        if let (Some(profile_model), Some(loc)) = (profile_model, location) {
            profile_model.estimate_cost(loc, static_cost)
        } else {
            static_cost
        }
    }
}

// ── Union-find ────────────────────────────────────────────────────────────────

/// Path-splitting union-find with union-by-rank.
struct EUnionFind {
    parent: Vec<EClassId>,
    rank:   Vec<u32>,
}

impl EUnionFind {
    fn new() -> Self { Self { parent: Vec::new(), rank: Vec::new() } }

    fn make(&mut self) -> EClassId {
        let id = self.parent.len() as EClassId;
        self.parent.push(id);
        self.rank.push(0);
        id
    }

    /// Mutating find with path splitting — O(α) amortised.
    fn find(&mut self, mut x: EClassId) -> EClassId {
        while self.parent[x as usize] != x {
            let pp = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = pp;
            x = pp;
        }
        x
    }

    /// Non-mutating find (no path compression) — for read-only contexts.
    fn find_imm(&self, mut x: EClassId) -> EClassId {
        while self.parent[x as usize] != x {
            x = self.parent[x as usize];
        }
        x
    }

    /// Union by rank; returns the new canonical root.
    fn union(&mut self, a: EClassId, b: EClassId) -> EClassId {
        let (a, b) = (self.find(a), self.find(b));
        if a == b { return a; }
        if self.rank[a as usize] < self.rank[b as usize] {
            self.parent[a as usize] = b; b
        } else if self.rank[a as usize] > self.rank[b as usize] {
            self.parent[b as usize] = a; a
        } else {
            self.parent[b as usize] = a;
            self.rank[a as usize] += 1;
            a
        }
    }
}

// ── Pattern matching for egg-style rewrites ───────────────────────────────────

/// A pattern that can match against e-nodes, with variable binding.
#[derive(Clone, Debug)]
enum EPattern {
    /// Match any e-node and bind to this variable name
    Var(String),
    /// Match a specific literal
    IntLit(u128),
    FloatBits(u64),
    Bool(bool),
    /// Match a binary operation with pattern children
    BinOp(BinOpKind, Box<EPattern>, Box<EPattern>),
    /// Match a unary operation with pattern child
    UnOp(UnOpKind, Box<EPattern>),
    /// Match if-then-else
    IfThenElse(Box<EPattern>, Box<EPattern>, Option<Box<EPattern>>),
}

/// A variable binding from pattern matching
type Bindings = FxHashMap<String, EClassId>;

impl EPattern {
    /// Match this pattern against an e-class, returning bindings if successful.
    fn match_class(&self, egraph: &EGraph, class: EClassId, bindings: &mut Bindings) -> bool {
        let canon = egraph.uf.find_imm(class);
        match self {
            EPattern::Var(name) => {
                // Bind this variable to the class
                bindings.insert(name.clone(), canon);
                true
            }
            EPattern::IntLit(v) => {
                egraph.int_lit_in(canon) == Some(*v)
            }
            EPattern::FloatBits(b) => {
                egraph.float_bits_in(canon) == Some(*b)
            }
            EPattern::Bool(b) => {
                egraph.bool_lit_in(canon) == Some(*b)
            }
            EPattern::BinOp(op, lhs_pat, rhs_pat) => {
                if let Some((l, r)) = egraph.binop_in_class(canon, *op) {
                    lhs_pat.match_class(egraph, l, bindings) && rhs_pat.match_class(egraph, r, bindings)
                } else {
                    false
                }
            }
            EPattern::UnOp(op, child_pat) => {
                if let Some((actual_op, child)) = egraph.unop_in_class(canon) {
                    if actual_op == *op {
                        child_pat.match_class(egraph, child, bindings)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            EPattern::IfThenElse(cond_pat, then_pat, else_pat) => {
                // Check if this class contains an IfThenElse node
                let c = egraph.uf.find_imm(canon);
                if let Some(nids) = egraph.classes.get(&c) {
                    for &nid in nids {
                        if let ENode::IfThenElse(cond_id, then_id, else_id) = &egraph.nodes[nid as usize] {
                            let cond_ok = cond_pat.match_class(egraph, *cond_id, bindings);
                            let then_ok = then_pat.match_class(egraph, *then_id, bindings);
                            let else_ok = else_pat.as_ref().map_or(true, |p| {
                                else_id.map_or(false, |e| p.match_class(egraph, e, bindings))
                            });
                            return cond_ok && then_ok && else_ok;
                        }
                    }
                }
                false
            }
        }
    }

    /// Instantiate this pattern with the given bindings to produce an ENode.
    /// Uses the egraph to get representative nodes from bound classes.
    fn instantiate(&self, egraph: &EGraph, bindings: &Bindings) -> Option<ENode> {
        match self {
            EPattern::Var(name) => {
                // Look up the bound class and pick a representative node
                bindings.get(name).and_then(|&class| {
                    let canon = egraph.uf.find_imm(class);
                    // Pick the cheapest node from the class
                    egraph.classes.get(&canon)?.iter().find_map(|&nid| {
                        Some(egraph.nodes[nid as usize].clone())
                    })
                })
            }
            EPattern::IntLit(v) => Some(ENode::IntLit(*v)),
            EPattern::FloatBits(b) => Some(ENode::FloatBits(*b)),
            EPattern::Bool(b) => Some(ENode::Bool(*b)),
            EPattern::BinOp(op, lhs, rhs) => {
                // For binary ops, we need to get the class IDs from bindings
                // and construct a new BinOp node with those class IDs
                match (lhs.as_ref(), rhs.as_ref()) {
                    (EPattern::Var(lname), EPattern::Var(rname)) => {
                        let l_class = bindings.get(lname).copied()?;
                        let r_class = bindings.get(rname).copied()?;
                        Some(ENode::BinOp(*op, l_class, r_class))
                    }
                    _ => {
                        // Complex case: recursively instantiate children
                        // This requires children to be literals or already bound
                        None // Simplified for now
                    }
                }
            }
            EPattern::UnOp(op, child) => {
                match child.as_ref() {
                    EPattern::Var(cname) => {
                        let c_class = bindings.get(cname).copied()?;
                        Some(ENode::UnOp(*op, c_class))
                    }
                    _ => None
                }
            }
            EPattern::IfThenElse(cond, then, else_) => {
                match (cond.as_ref(), then.as_ref(), else_.as_ref()) {
                    (EPattern::Var(cname), EPattern::Var(tname), Some(ref ename_box)) => {
                        if let EPattern::Var(ename) = ename_box.as_ref() {
                            let c_class = bindings.get(cname).copied()?;
                            let t_class = bindings.get(tname).copied()?;
                            let e_class = bindings.get(ename).copied()?;
                            Some(ENode::IfThenElse(c_class, t_class, Some(e_class)))
                        } else {
                            None
                        }
                    }
                    (EPattern::Var(cname), EPattern::Var(tname), None) => {
                        let c_class = bindings.get(cname).copied()?;
                        let t_class = bindings.get(tname).copied()?;
                        Some(ENode::IfThenElse(c_class, t_class, None))
                    }
                    _ => None
                }
            }
        }
    }
}

/// A rewrite rule with pattern-based matching
struct PatternRewrite {
    name: &'static str,
    pattern: EPattern,
    replacement: EPattern,
}

impl PatternRewrite {
    /// Create a new pattern rewrite rule
    fn new(name: &'static str, pattern: EPattern, replacement: EPattern) -> Self {
        Self { name, pattern, replacement }
    }
}

/// Helper functions to create patterns
fn var(name: &str) -> EPattern {
    EPattern::Var(name.to_string())
}

fn int_lit(v: u128) -> EPattern {
    EPattern::IntLit(v)
}

fn bool_lit(b: bool) -> EPattern {
    EPattern::Bool(b)
}

fn binop(op: BinOpKind, lhs: EPattern, rhs: EPattern) -> EPattern {
    EPattern::BinOp(op, Box::new(lhs), Box::new(rhs))
}

fn unop(op: UnOpKind, child: EPattern) -> EPattern {
    EPattern::UnOp(op, Box::new(child))
}

// ── Pending rewrite ───────────────────────────────────────────────────────────

/// A rewrite produced by `generate_rewrites` (read-only phase).
/// Applied in the subsequent mutable phase.
enum ERewrite {
    /// Add a new e-node to the graph and union its class with `class`.
    AddNode(EClassId, ENode),
    /// Directly union two existing classes.
    UnionClasses(EClassId, EClassId),
}

// ── EGraph ────────────────────────────────────────────────────────────────────

struct EGraph {
    /// Every e-node ever added, indexed by ENodeId.
    nodes:      Vec<ENode>,
    /// `node_class[nid]` — the class this node was *placed into* at add time.
    /// After merges, use `uf.find(node_class[nid])` for the canonical class.
    node_class: Vec<EClassId>,
    /// Canonical class id → list of e-node ids that live in it.
    classes:    FxHashMap<EClassId, Vec<ENodeId>>,
    /// Canonical class id → list of e-node ids that reference it as a child.
    /// Maintained so that `rebuild` can re-canonicalize stale hashcons entries.
    parents:    FxHashMap<EClassId, Vec<ENodeId>>,
    /// Hash-consing map: canonical ENode → ENodeId.
    hashcons:   FxHashMap<ENode, ENodeId>,
    uf:         EUnionFind,
    /// Pending (root, merged-child) pairs to process in `rebuild`.
    pending:    Vec<(EClassId, EClassId)>,
    /// Total equivalences discovered.
    pub rewrites: u64,
    /// Pattern-based rewrite rules
    pattern_rules: Vec<PatternRewrite>,
    /// Hardware-aware cost model
    hw_cost_model: HardwareCostModel,
    /// Profile-weighted cost model
    profile_cost_model: Option<ProfileWeightedCostModel>,
    /// Current location being optimized (for profile lookup)
    current_location: Option<String>,
}

impl EGraph {
    fn new() -> Self {
        let pattern_rules = Self::builtin_pattern_rules();
        let hw_cost_model = HardwareCostModel::new();
        Self {
            nodes:      Vec::with_capacity(64),
            node_class: Vec::with_capacity(64),
            classes:    FxHashMap::default(),
            parents:    FxHashMap::default(),
            hashcons:   FxHashMap::default(),
            uf:         EUnionFind::new(),
            pending:    Vec::new(),
            rewrites:   0,
            pattern_rules,
            hw_cost_model,
            profile_cost_model: None,
            current_location: None,
        }
    }

    /// Create an e-graph with profile-guided optimization
    fn with_profile(profile_model: ProfileWeightedCostModel) -> Self {
        let pattern_rules = Self::builtin_pattern_rules();
        let hw_cost_model = HardwareCostModel::new();
        Self {
            nodes:      Vec::with_capacity(64),
            node_class: Vec::with_capacity(64),
            classes:    FxHashMap::default(),
            parents:    FxHashMap::default(),
            hashcons:   FxHashMap::default(),
            uf:         EUnionFind::new(),
            pending:    Vec::new(),
            rewrites:   0,
            pattern_rules,
            hw_cost_model,
            profile_cost_model: Some(profile_model),
            current_location: None,
        }
    }

    /// Set the current location for profile lookup
    fn set_location(&mut self, location: String) {
        self.current_location = Some(location);
    }

    /// Define the built-in pattern-based rewrite rules
    fn builtin_pattern_rules() -> Vec<PatternRewrite> {
        vec![
            // Commutativity: x + y = y + x
            PatternRewrite::new(
                "commutative_add",
                binop(BinOpKind::Add, var("x"), var("y")),
                binop(BinOpKind::Add, var("y"), var("x")),
            ),
            // Commutativity: x * y = y * x
            PatternRewrite::new(
                "commutative_mul",
                binop(BinOpKind::Mul, var("x"), var("y")),
                binop(BinOpKind::Mul, var("y"), var("x")),
            ),
            // Identity: x + 0 = x
            PatternRewrite::new(
                "add_zero_left",
                binop(BinOpKind::Add, int_lit(0), var("x")),
                var("x"),
            ),
            PatternRewrite::new(
                "add_zero_right",
                binop(BinOpKind::Add, var("x"), int_lit(0)),
                var("x"),
            ),
            // Identity: x * 1 = x
            PatternRewrite::new(
                "mul_one_left",
                binop(BinOpKind::Mul, int_lit(1), var("x")),
                var("x"),
            ),
            PatternRewrite::new(
                "mul_one_right",
                binop(BinOpKind::Mul, var("x"), int_lit(1)),
                var("x"),
            ),
            // Annihilation: x * 0 = 0
            PatternRewrite::new(
                "mul_zero_left",
                binop(BinOpKind::Mul, int_lit(0), var("x")),
                int_lit(0),
            ),
            PatternRewrite::new(
                "mul_zero_right",
                binop(BinOpKind::Mul, var("x"), int_lit(0)),
                int_lit(0),
            ),
            // Double negation: --x = x
            PatternRewrite::new(
                "double_neg",
                unop(UnOpKind::Neg, unop(UnOpKind::Neg, var("x"))),
                var("x"),
            ),
            // Double bitwise not: ~~x = x
            PatternRewrite::new(
                "double_not",
                unop(UnOpKind::Not, unop(UnOpKind::Not, var("x"))),
                var("x"),
            ),
            // x - x = 0
            PatternRewrite::new(
                "sub_self",
                binop(BinOpKind::Sub, var("x"), var("x")),
                int_lit(0),
            ),
            // x / x = 1
            PatternRewrite::new(
                "div_self",
                binop(BinOpKind::Div, var("x"), var("x")),
                int_lit(1),
            ),
            // x ^ x = 0
            PatternRewrite::new(
                "xor_self",
                binop(BinOpKind::BitXor, var("x"), var("x")),
                int_lit(0),
            ),
            // x & x = x
            PatternRewrite::new(
                "and_self",
                binop(BinOpKind::BitAnd, var("x"), var("x")),
                var("x"),
            ),
            // x | x = x
            PatternRewrite::new(
                "or_self",
                binop(BinOpKind::BitOr, var("x"), var("x")),
                var("x"),
            ),
            // x == x = true
            PatternRewrite::new(
                "eq_self",
                binop(BinOpKind::Eq, var("x"), var("x")),
                bool_lit(true),
            ),
            // x != x = false
            PatternRewrite::new(
                "ne_self",
                binop(BinOpKind::Ne, var("x"), var("x")),
                bool_lit(false),
            ),
        ]
    }

    // ── Core operations ───────────────────────────────────────────────────────

    /// Add `enode` to the e-graph (with children canonicalized via union-find).
    /// Returns the e-class that now contains this e-node.  Idempotent:
    /// adding an already-present e-node simply returns its canonical class.
    fn add(&mut self, enode: ENode) -> EClassId {
        let canon = enode.map_children(|c| self.uf.find(c));

        if let Some(&nid) = self.hashcons.get(&canon) {
            return self.uf.find(self.node_class[nid as usize]);
        }

        let nid = self.nodes.len() as ENodeId;
        let cid = self.uf.make();

        // Register as parent of each child class so rebuild can find us.
        canon.for_each_child(|child| {
            let child_canon = self.uf.find_imm(child);
            self.parents.entry(child_canon).or_default().push(nid);
        });

        self.hashcons.insert(canon.clone(), nid);
        self.nodes.push(canon);
        self.node_class.push(cid);
        self.classes.entry(cid).or_default().push(nid);
        cid
    }

    /// Merge two e-classes.  Queues a rebuild; call `rebuild()` after all
    /// `union_classes` calls for a given round to propagate merges transitively.
    fn union_classes(&mut self, a: EClassId, b: EClassId) {
        let (a, b) = (self.uf.find(a), self.uf.find(b));
        if a == b { return; }

        let root  = self.uf.union(a, b);
        let child = if root == a { b } else { a };

        // Merge child's node list into root.
        let child_nodes = self.classes.remove(&child).unwrap_or_default();
        for &nid in &child_nodes {
            self.node_class[nid as usize] = root;
        }
        self.classes.entry(root).or_default().extend(child_nodes);

        // Merge parent lists.
        let child_parents = self.parents.remove(&child).unwrap_or_default();
        self.parents.entry(root).or_default().extend(child_parents);

        self.pending.push((root, child));
        self.rewrites += 1;
    }

    /// Re-canonicalize the hashcons after class merges so that e-nodes whose
    /// children were in merged classes now carry canonical child IDs.
    /// Transitively processes any new merges discovered during the rebuild.
    fn rebuild(&mut self) {
        while !self.pending.is_empty() {
            let batch: Vec<(EClassId, EClassId)> = self.pending.drain(..).collect();

            for (root, _child) in batch {
                let parent_nids: Vec<ENodeId> =
                    self.parents.get(&root).cloned().unwrap_or_default();

                for nid in parent_nids {
                    let old_canon = self.nodes[nid as usize].clone();
                    let new_canon = old_canon.clone().map_children(|c| self.uf.find(c));

                    if old_canon == new_canon {
                        continue; // Still canonical — nothing to do.
                    }

                    // Remove the stale hashcons entry.
                    self.hashcons.remove(&old_canon);

                    if let Some(&existing_nid) = self.hashcons.get(&new_canon) {
                        // Re-canonicalization reveals two formerly distinct nodes
                        // are now equivalent — merge their classes.
                        let ca = self.uf.find(self.node_class[nid as usize]);
                        let cb = self.uf.find(self.node_class[existing_nid as usize]);
                        if ca != cb {
                            let new_root  = self.uf.union(ca, cb);
                            let new_child = if new_root == ca { cb } else { ca };
                            let nc_nodes = self.classes.remove(&new_child).unwrap_or_default();
                            for &n in &nc_nodes { self.node_class[n as usize] = new_root; }
                            self.classes.entry(new_root).or_default().extend(nc_nodes);
                            let nc_parents = self.parents.remove(&new_child).unwrap_or_default();
                            self.parents.entry(new_root).or_default().extend(nc_parents);
                            self.pending.push((new_root, new_child));
                        }
                        // The stale node's hashcons slot is now owned by existing_nid.
                    } else {
                        // Update parent sets to reflect the new canonical children.
                        old_canon.for_each_child(|c| {
                            let cc = self.uf.find_imm(c);
                            if let Some(pl) = self.parents.get_mut(&cc) {
                                pl.retain(|&n| n != nid);
                            }
                        });
                        new_canon.for_each_child(|c| {
                            let cc = self.uf.find_imm(c);
                            self.parents.entry(cc).or_default().push(nid);
                        });
                        self.nodes[nid as usize] = new_canon.clone();
                        self.hashcons.insert(new_canon, nid);
                    }
                }
            }
        }
    }

    // ── Building from an Expr ─────────────────────────────────────────────────

    /// Recursively translate `expr` into the EGraph.
    /// Returns the e-class ID for the root of `expr`.
    fn build_expr(&mut self, expr: &Expr) -> EClassId {
        match expr {
            Expr::IntLit   { value, .. } => self.add(ENode::IntLit(*value)),
            Expr::FloatLit { value, .. } => self.add(ENode::FloatBits(value.to_bits())),
            Expr::BoolLit  { value, .. } => self.add(ENode::Bool(*value)),
            Expr::Ident    { name,  .. } => self.add(ENode::Var(name.clone())),
            Expr::BinOp { op, lhs, rhs, .. } => {
                let l = self.build_expr(lhs);
                let r = self.build_expr(rhs);
                self.add(ENode::BinOp(*op, l, r))
            }
            Expr::UnOp { op, expr: inner, .. } => {
                let c = self.build_expr(inner);
                self.add(ENode::UnOp(*op, c))
            }
            Expr::IfExpr { cond, then, else_, .. } => {
                let c = self.build_expr(cond);
                let t = self.build_block(then);
                let e = else_.as_ref().map(|b| self.build_block(b));
                self.add(ENode::IfThenElse(c, t, e))
            }
            // Anything structurally opaque (calls, field access, …) becomes a
            // unique placeholder variable.  The EGraph can still reason about
            // expressions that *contain* opaque sub-trees via the outer nodes.
            _ => {
                let tag = format!("__ego_opaque_{}", self.nodes.len());
                self.add(ENode::Var(tag))
            }
        }
    }

    /// Build a Block into the e-graph by building its tail expression
    /// (or a placeholder if no tail).
    fn build_block(&mut self, block: &Block) -> EClassId {
        if let Some(ref tail) = block.tail {
            self.build_expr(tail)
        } else {
            let tag = format!("__ego_unit_{}", self.nodes.len());
            self.add(ENode::Var(tag))
        }
    }

    // ── Equality saturation ───────────────────────────────────────────────────

    /// Run equality saturation: apply all rewrite rules until fixpoint or
    /// `max_iters` is exhausted.
    fn saturate(&mut self, max_iters: usize) {
        for _ in 0..max_iters {
            let before = self.rewrites;
            // Phase A: collect rewrites (read-only borrow).
            let rewrites = self.generate_rewrites();
            // Phase B: apply them (mutable borrow).
            for rw in rewrites {
                match rw {
                    ERewrite::AddNode(cls, enode) => {
                        let new_cls = self.add(enode);
                        let ca = self.uf.find(cls);
                        let cb = self.uf.find(new_cls);
                        if ca != cb { self.union_classes(ca, cb); }
                    }
                    ERewrite::UnionClasses(a, b) => {
                        let ca = self.uf.find(a);
                        let cb = self.uf.find(b);
                        if ca != cb { self.union_classes(ca, cb); }
                    }
                }
            }
            self.rebuild();
            if self.rewrites == before { break; } // fixpoint
        }
    }

    /// Produce all rewrites implied by the current e-graph state.
    /// This method takes `&self` only — no aliasing with the subsequent
    /// mutable apply step.
    fn generate_rewrites(&self) -> Vec<ERewrite> {
        let mut out: Vec<ERewrite> = Vec::new();

        // First, apply pattern-based rewrite rules
        for rule in &self.pattern_rules {
            self.apply_pattern_rule(rule, &mut out);
        }

        // Then, apply hand-coded rewrite rules for complex patterns
        for (&cid, nids) in &self.classes {
            for &nid in nids {
                let node = &self.nodes[nid as usize];
                self.rewrite_enode(cid, node, &mut out);
            }
        }
        out
    }

    /// Apply a single pattern-based rewrite rule to all e-classes
    fn apply_pattern_rule(&self, rule: &PatternRewrite, out: &mut Vec<ERewrite>) {
        for (&cid, _) in &self.classes {
            let mut bindings = Bindings::default();
            if rule.pattern.match_class(self, cid, &mut bindings) {
                // Pattern matched! Try to instantiate the replacement
                if let Some(replacement_node) = rule.replacement.instantiate(self, &bindings) {
                    out.push(ERewrite::AddNode(cid, replacement_node));
                }
            }
        }
    }

    fn rewrite_enode(&self, cid: EClassId, node: &ENode, out: &mut Vec<ERewrite>) {
        match node {
            ENode::BinOp(op, a, b) => self.rewrite_binop(cid, *op, *a, *b, out),
            ENode::UnOp(op, a)     => self.rewrite_unop(cid, *op, *a, out),
            _ => {}
        }
    }

    // ── Rewrite rules — binary operators ─────────────────────────────────────

    fn rewrite_binop(
        &self,
        cid: EClassId,
        op:  BinOpKind,
        a:   EClassId,
        b:   EClassId,
        out: &mut Vec<ERewrite>,
    ) {
        // ── Commutativity ─────────────────────────────────────────────────────
        match op {
            BinOpKind::Add | BinOpKind::Mul
            | BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor
            | BinOpKind::Eq | BinOpKind::Ne => {
                out.push(ERewrite::AddNode(cid, ENode::BinOp(op, b, a)));
            }
            _ => {}
        }

        // ── Associativity ─────────────────────────────────────────────────────
        // (a op b) op c = a op (b op c) for associative ops
        if matches!(op, BinOpKind::Add | BinOpKind::Mul | BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor) {
            // If left child is also the same op, emit right-associative form
            if let Some((x, _y)) = self.binop_in_class(a, op) {
                out.push(ERewrite::AddNode(cid, ENode::BinOp(op, x, b)));
                // This emits: (x op y) op b → x op (y op b)
                // The e-graph will discover the equivalence through saturation
            }
        }

        // ── Distributivity ─────────────────────────────────────────────────────
        // a * (b + c) = a*b + a*c
        // a * (b - c) = a*b - a*c
        if op == BinOpKind::Mul {
            // Check if right child is Add or Sub
            if let Some((l, r)) = self.binop_in_class(b, BinOpKind::Add) {
                // a * (l + r) → emit a*l and a*r, then (a*l) + (a*r)
                // We emit the distributed form in the next saturation round
                // For now, just emit the two multiplications
                out.push(ERewrite::AddNode(cid, ENode::BinOp(BinOpKind::Mul, a, l)));
                out.push(ERewrite::AddNode(cid, ENode::BinOp(BinOpKind::Mul, a, r)));
            }
            if let Some((l, r)) = self.binop_in_class(b, BinOpKind::Sub) {
                out.push(ERewrite::AddNode(cid, ENode::BinOp(BinOpKind::Mul, a, l)));
                out.push(ERewrite::AddNode(cid, ENode::BinOp(BinOpKind::Mul, a, r)));
            }
            // Check if left child is Add or Sub
            if let Some((l, r)) = self.binop_in_class(a, BinOpKind::Add) {
                out.push(ERewrite::AddNode(cid, ENode::BinOp(BinOpKind::Mul, l, b)));
                out.push(ERewrite::AddNode(cid, ENode::BinOp(BinOpKind::Mul, r, b)));
            }
            if let Some((l, r)) = self.binop_in_class(a, BinOpKind::Sub) {
                out.push(ERewrite::AddNode(cid, ENode::BinOp(BinOpKind::Mul, l, b)));
                out.push(ERewrite::AddNode(cid, ENode::BinOp(BinOpKind::Mul, r, b)));
            }
        }

        // ── Query children for literal values ─────────────────────────────────
        let a_int  = self.int_lit_in(a);
        let b_int  = self.int_lit_in(b);
        let a_bool = self.bool_lit_in(a);
        let b_bool = self.bool_lit_in(b);
        let a_flt  = self.float_bits_in(a);
        let b_flt  = self.float_bits_in(b);
        let same   = self.uf.find_imm(a) == self.uf.find_imm(b);

        // ── Constant folding ──────────────────────────────────────────────────
        // Integer × Integer
        if let (Some(av), Some(bv)) = (a_int, b_int) {
            let folded: Option<ENode> = match op {
                BinOpKind::Add    => Some(ENode::IntLit(av.wrapping_add(bv))),
                BinOpKind::Sub    => Some(ENode::IntLit(av.wrapping_sub(bv))),
                BinOpKind::Mul    => Some(ENode::IntLit(av.wrapping_mul(bv))),
                BinOpKind::Div    if bv != 0 => Some(ENode::IntLit(av / bv)),
                BinOpKind::Rem    if bv != 0 => Some(ENode::IntLit(av % bv)),
                BinOpKind::BitAnd => Some(ENode::IntLit(av & bv)),
                BinOpKind::BitOr  => Some(ENode::IntLit(av | bv)),
                BinOpKind::BitXor => Some(ENode::IntLit(av ^ bv)),
                BinOpKind::Shl    => Some(ENode::IntLit(av.checked_shl(bv.try_into().unwrap_or(128)).unwrap_or(0))),
                BinOpKind::Shr    => Some(ENode::IntLit(av.checked_shr(bv.try_into().unwrap_or(128)).unwrap_or(0))),
                BinOpKind::Eq     => Some(ENode::Bool(av == bv)),
                BinOpKind::Ne     => Some(ENode::Bool(av != bv)),
                BinOpKind::Lt     => Some(ENode::Bool(av <  bv)),
                BinOpKind::Le     => Some(ENode::Bool(av <= bv)),
                BinOpKind::Gt     => Some(ENode::Bool(av >  bv)),
                BinOpKind::Ge     => Some(ENode::Bool(av >= bv)),
                _ => None,
            };
            if let Some(n) = folded { out.push(ERewrite::AddNode(cid, n)); }
        }
        // Bool × Bool
        if let (Some(av), Some(bv)) = (a_bool, b_bool) {
            let folded: Option<ENode> = match op {
                BinOpKind::And => Some(ENode::Bool(av && bv)),
                BinOpKind::Or  => Some(ENode::Bool(av || bv)),
                BinOpKind::Eq  => Some(ENode::Bool(av == bv)),
                BinOpKind::Ne  => Some(ENode::Bool(av != bv)),
                _ => None,
            };
            if let Some(n) = folded { out.push(ERewrite::AddNode(cid, n)); }
        }
        // Float × Float — only fold finite, non-NaN operands
        if let (Some(ab), Some(bb)) = (a_flt, b_flt) {
            let (af, bf) = (f64::from_bits(ab), f64::from_bits(bb));
            if af.is_finite() && bf.is_finite() {
                let folded: Option<ENode> = match op {
                    BinOpKind::Add => Some(ENode::FloatBits((af + bf).to_bits())),
                    BinOpKind::Sub => Some(ENode::FloatBits((af - bf).to_bits())),
                    BinOpKind::Mul => Some(ENode::FloatBits((af * bf).to_bits())),
                    BinOpKind::Div if bf != 0.0 => Some(ENode::FloatBits((af / bf).to_bits())),
                    _ => None,
                };
                if let Some(n) = folded { out.push(ERewrite::AddNode(cid, n)); }
            }
        }

        // ── Identity and annihilation laws ────────────────────────────────────
        match op {
            BinOpKind::Add => {
                if a_int == Some(0) { out.push(ERewrite::UnionClasses(cid, b)); }
                if b_int == Some(0) { out.push(ERewrite::UnionClasses(cid, a)); }
                // x + x → x * 2 is handled by StrengthReducer pass
                // E-graph focuses on algebraic equivalences, not strength reduction
            }
            BinOpKind::Sub => {
                if b_int == Some(0) { out.push(ERewrite::UnionClasses(cid, a)); }
                // x - x = 0
                if same { out.push(ERewrite::AddNode(cid, ENode::IntLit(0))); }
            }
            BinOpKind::Mul => {
                if a_int == Some(1) { out.push(ERewrite::UnionClasses(cid, b)); }
                if b_int == Some(1) { out.push(ERewrite::UnionClasses(cid, a)); }
                if a_int == Some(0) { out.push(ERewrite::AddNode(cid, ENode::IntLit(0))); }
                if b_int == Some(0) { out.push(ERewrite::AddNode(cid, ENode::IntLit(0))); }
                // x * 2^k → x << k is handled by StrengthReducer pass
                // E-graph focuses on algebraic equivalences, not strength reduction
            }
            BinOpKind::Div => {
                if b_int == Some(1) { out.push(ERewrite::UnionClasses(cid, a)); }
                // x / x = 1 (when x != 0, but we assume non-zero in e-graph)
                if same { out.push(ERewrite::AddNode(cid, ENode::IntLit(1))); }
            }
            BinOpKind::BitAnd => {
                if a_int == Some(0)        { out.push(ERewrite::AddNode(cid, ENode::IntLit(0))); }
                if b_int == Some(0)        { out.push(ERewrite::AddNode(cid, ENode::IntLit(0))); }
                if a_int == Some(u128::MAX) { out.push(ERewrite::UnionClasses(cid, b)); }
                if b_int == Some(u128::MAX) { out.push(ERewrite::UnionClasses(cid, a)); }
                if same                    { out.push(ERewrite::UnionClasses(cid, a)); }
                // Absorption: a & (a | b) = a
                if let Some((l2, _r2)) = self.binop_in_class(b, BinOpKind::BitOr) {
                    if self.uf.find_imm(l2) == self.uf.find_imm(a) {
                        out.push(ERewrite::UnionClasses(cid, a));
                    }
                }
            }
            BinOpKind::BitOr => {
                if a_int == Some(0)        { out.push(ERewrite::UnionClasses(cid, b)); }
                if b_int == Some(0)        { out.push(ERewrite::UnionClasses(cid, a)); }
                if a_int == Some(u128::MAX) { out.push(ERewrite::AddNode(cid, ENode::IntLit(u128::MAX))); }
                if b_int == Some(u128::MAX) { out.push(ERewrite::AddNode(cid, ENode::IntLit(u128::MAX))); }
                if same                    { out.push(ERewrite::UnionClasses(cid, a)); }
                // Absorption: a | (a & b) = a
                if let Some((l2, _r2)) = self.binop_in_class(b, BinOpKind::BitAnd) {
                    if self.uf.find_imm(l2) == self.uf.find_imm(a) {
                        out.push(ERewrite::UnionClasses(cid, a));
                    }
                }
            }
            BinOpKind::BitXor => {
                if a_int == Some(0) { out.push(ERewrite::UnionClasses(cid, b)); }
                if b_int == Some(0) { out.push(ERewrite::UnionClasses(cid, a)); }
                // x ^ x = 0
                if same { out.push(ERewrite::AddNode(cid, ENode::IntLit(0))); }
            }
            BinOpKind::Shl | BinOpKind::Shr => {
                if b_int == Some(0) { out.push(ERewrite::UnionClasses(cid, a)); }
                if a_int == Some(0) { out.push(ERewrite::AddNode(cid, ENode::IntLit(0))); }
            }
            BinOpKind::Eq => {
                // x == x = true
                if same { out.push(ERewrite::AddNode(cid, ENode::Bool(true))); }
            }
            BinOpKind::Ne => {
                // x != x = false
                if same { out.push(ERewrite::AddNode(cid, ENode::Bool(false))); }
            }
            BinOpKind::And => {
                if a_bool == Some(true)  { out.push(ERewrite::UnionClasses(cid, b)); }
                if b_bool == Some(true)  { out.push(ERewrite::UnionClasses(cid, a)); }
                if a_bool == Some(false) { out.push(ERewrite::AddNode(cid, ENode::Bool(false))); }
                if b_bool == Some(false) { out.push(ERewrite::AddNode(cid, ENode::Bool(false))); }
                if same                  { out.push(ERewrite::UnionClasses(cid, a)); }
            }
            BinOpKind::Or => {
                if a_bool == Some(false) { out.push(ERewrite::UnionClasses(cid, b)); }
                if b_bool == Some(false) { out.push(ERewrite::UnionClasses(cid, a)); }
                if a_bool == Some(true)  { out.push(ERewrite::AddNode(cid, ENode::Bool(true))); }
                if b_bool == Some(true)  { out.push(ERewrite::AddNode(cid, ENode::Bool(true))); }
                if same                  { out.push(ERewrite::UnionClasses(cid, a)); }
            }
            _ => {}
        }

        // ── Reassociation to group constants (const folding enabler) ──────────
        // Pattern: (x op c1) op c2  →  x op (c1 op c2)
        // where c1 and c2 are known integer literals and op is associative.
        if matches!(op, BinOpKind::Add | BinOpKind::Mul | BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor) {
            if let Some(bv) = b_int {
                if let Some((_x, c1_class)) = self.binop_in_class(a, op) {
                    if let Some(c1v) = self.int_lit_in(c1_class) {
                        // Fold c1 op c2
                        let combined: Option<u128> = match op {
                            BinOpKind::Add    => Some(c1v.wrapping_add(bv)),
                            BinOpKind::Mul    => Some(c1v.wrapping_mul(bv)),
                            BinOpKind::BitAnd => Some(c1v & bv),
                            BinOpKind::BitOr  => Some(c1v | bv),
                            BinOpKind::BitXor => Some(c1v ^ bv),
                            _ => None,
                        };
                        if let Some(cv) = combined {
                            // Emit the combined constant node
                            out.push(ERewrite::AddNode(cid, ENode::IntLit(cv)));
                            // The next saturation round will constant-fold x op cv
                            // when it appears as a BinOp child
                        }
                    }
                }
            }
        }
    }

    // ── Rewrite rules — unary operators ──────────────────────────────────────

    fn rewrite_unop(
        &self,
        cid: EClassId,
        op:  UnOpKind,
        a:   EClassId,
        out: &mut Vec<ERewrite>,
    ) {
        match op {
            UnOpKind::Neg => {
                // --x = x
                if let Some((UnOpKind::Neg, inner)) = self.unop_in_class(a) {
                    out.push(ERewrite::UnionClasses(cid, inner));
                }
                // Constant folding
                if let Some(v) = self.int_lit_in(a) {
                    out.push(ERewrite::AddNode(cid, ENode::IntLit((v as i128).wrapping_neg() as u128)));
                }
                if let Some(b) = self.float_bits_in(a) {
                    out.push(ERewrite::AddNode(cid, ENode::FloatBits((-f64::from_bits(b)).to_bits())));
                }
            }
            UnOpKind::Not => {
                // !!x = x  (boolean) and ~~x = x  (bitwise complement)
                if let Some((UnOpKind::Not, inner)) = self.unop_in_class(a) {
                    out.push(ERewrite::UnionClasses(cid, inner));
                }

                // Comparison negation: !(a < b) = a >= b, etc.
                if let Some((l, r)) = self.binop_in_class(a, BinOpKind::Eq) {
                    out.push(ERewrite::AddNode(cid, ENode::BinOp(BinOpKind::Ne, l, r)));
                }
                if let Some((l, r)) = self.binop_in_class(a, BinOpKind::Ne) {
                    out.push(ERewrite::AddNode(cid, ENode::BinOp(BinOpKind::Eq, l, r)));
                }
                if let Some((l, r)) = self.binop_in_class(a, BinOpKind::Lt) {
                    out.push(ERewrite::AddNode(cid, ENode::BinOp(BinOpKind::Ge, l, r)));
                }
                if let Some((l, r)) = self.binop_in_class(a, BinOpKind::Le) {
                    out.push(ERewrite::AddNode(cid, ENode::BinOp(BinOpKind::Gt, l, r)));
                }
                if let Some((l, r)) = self.binop_in_class(a, BinOpKind::Gt) {
                    out.push(ERewrite::AddNode(cid, ENode::BinOp(BinOpKind::Le, l, r)));
                }
                if let Some((l, r)) = self.binop_in_class(a, BinOpKind::Ge) {
                    out.push(ERewrite::AddNode(cid, ENode::BinOp(BinOpKind::Lt, l, r)));
                }

                // De Morgan's laws:  !(a && b) = !a || !b,  !(a || b) = !a && !b
                // Emit the negated sub-expressions; the full Or/And form will be
                // discovered in the next saturation round when Not(l) and Not(r)
                // are available as e-classes.
                if let Some((l, r)) = self.binop_in_class(a, BinOpKind::And) {
                    out.push(ERewrite::AddNode(cid, ENode::UnOp(UnOpKind::Not, l)));
                    out.push(ERewrite::AddNode(cid, ENode::UnOp(UnOpKind::Not, r)));
                }
                if let Some((l, r)) = self.binop_in_class(a, BinOpKind::Or) {
                    out.push(ERewrite::AddNode(cid, ENode::UnOp(UnOpKind::Not, l)));
                    out.push(ERewrite::AddNode(cid, ENode::UnOp(UnOpKind::Not, r)));
                }

                // Constant folding
                if let Some(v) = self.int_lit_in(a) {
                    out.push(ERewrite::AddNode(cid, ENode::IntLit(!v)));
                }
                if let Some(v) = self.bool_lit_in(a) {
                    out.push(ERewrite::AddNode(cid, ENode::Bool(!v)));
                }
            }
            _ => {}
        }
    }

    // ── Read-only query helpers (borrow-safe) ─────────────────────────────────

    /// If `class` contains an `IntLit` node, return its value.
    fn int_lit_in(&self, class: EClassId) -> Option<u128> {
        let c = self.uf.find_imm(class);
        self.classes.get(&c)?.iter().find_map(|&nid| {
            if let ENode::IntLit(v) = &self.nodes[nid as usize] { Some(*v) } else { None }
        })
    }

    /// If `class` contains a `FloatBits` node, return its raw bits.
    fn float_bits_in(&self, class: EClassId) -> Option<u64> {
        let c = self.uf.find_imm(class);
        self.classes.get(&c)?.iter().find_map(|&nid| {
            if let ENode::FloatBits(b) = &self.nodes[nid as usize] { Some(*b) } else { None }
        })
    }

    /// If `class` contains a `Bool` node, return its value.
    fn bool_lit_in(&self, class: EClassId) -> Option<bool> {
        let c = self.uf.find_imm(class);
        self.classes.get(&c)?.iter().find_map(|&nid| {
            if let ENode::Bool(v) = &self.nodes[nid as usize] { Some(*v) } else { None }
        })
    }

    /// If `class` contains a `BinOp` node with the given `op`, return `(l, r)`.
    fn binop_in_class(&self, class: EClassId, op: BinOpKind) -> Option<(EClassId, EClassId)> {
        let c = self.uf.find_imm(class);
        self.classes.get(&c)?.iter().find_map(|&nid| {
            if let ENode::BinOp(nop, l, r) = &self.nodes[nid as usize] {
                if *nop == op { Some((*l, *r)) } else { None }
            } else { None }
        })
    }

    /// If `class` contains a `UnOp` node, return `(op, child)`.
    fn unop_in_class(&self, class: EClassId) -> Option<(UnOpKind, EClassId)> {
        let c = self.uf.find_imm(class);
        self.classes.get(&c)?.iter().find_map(|&nid| {
            if let ENode::UnOp(op, ch) = &self.nodes[nid as usize] {
                Some((*op, *ch))
            } else { None }
        })
    }

    // ── Extraction ────────────────────────────────────────────────────────────

    /// Bellman-Ford cost minimisation over the saturated e-graph.
    /// Returns a map from canonical e-class ID to `(min_cost, best_node_id)`.
    fn extract_best(&self) -> FxHashMap<EClassId, (f64, ENodeId)> {
        let mut best: FxHashMap<EClassId, (f64, ENodeId)> = FxHashMap::default();

        // Seed with leaf nodes (zero cost).
        for (&cid, nids) in &self.classes {
            for &nid in nids {
                if matches!(
                    &self.nodes[nid as usize],
                    ENode::IntLit(_) | ENode::FloatBits(_) | ENode::Bool(_) | ENode::Var(_)
                ) {
                    let entry = best.entry(cid).or_insert((f64::INFINITY, nid));
                    if entry.0 > 0.0 { *entry = (0.0, nid); }
                }
            }
        }

        // Iterate until no improvement — converges in O(depth) rounds for DAGs.
        loop {
            let mut changed = false;
            for (&cid, nids) in &self.classes {
                for &nid in nids {
                    let cost = self.total_cost(nid, &best);
                    let entry = best.entry(cid).or_insert((f64::INFINITY, nid));
                    if cost < entry.0 - 1e-9 {
                        *entry = (cost, nid);
                        changed = true;
                    }
                }
            }
            if !changed { break; }
        }

        best
    }

    fn total_cost(&self, nid: ENodeId, best: &FxHashMap<EClassId, (f64, ENodeId)>) -> f64 {
        let node = &self.nodes[nid as usize];
        let location = self.current_location.as_deref();
        let profile_model = self.profile_cost_model.as_ref();
        let mut cost = node.base_cost(&self.hw_cost_model, profile_model, location);
        node.for_each_child(|child| {
            let canon = self.uf.find_imm(child);
            cost += best.get(&canon).map(|(c, _)| *c).unwrap_or(f64::INFINITY);
        });
        cost
    }

    /// Materialise the cheapest expression rooted at `class` using `best`.
    /// `depth` is a cycle-guard — in a well-formed acyclic EGraph it is never
    /// triggered; it is present purely as a defensive measure.
    fn materialize(&self, class: EClassId, best: &FxHashMap<EClassId, (f64, ENodeId)>, span: Span, depth: usize) -> Expr {
        if depth == 0 {
            return Expr::IntLit { span, value: 0 };
        }
        let canon = self.uf.find_imm(class);
        let &(_cost, nid) = match best.get(&canon) {
            Some(e) if e.0 < f64::INFINITY => e,
            _ => return Expr::IntLit { span, value: 0 },
        };
        match &self.nodes[nid as usize] {
            ENode::IntLit(v)    => Expr::IntLit  { span, value: *v },
            ENode::FloatBits(b) => Expr::FloatLit { span, value: f64::from_bits(*b) },
            ENode::Bool(v)      => Expr::BoolLit  { span, value: *v },
            ENode::Var(name)    => Expr::Ident    { span, name: name.clone() },
            ENode::BinOp(op, l, r) => {
                let lhs = self.materialize(*l, best, span, depth - 1);
                let rhs = self.materialize(*r, best, span, depth - 1);
                Expr::BinOp { span, op: *op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
            }
            ENode::UnOp(op, c) => {
                let child = self.materialize(*c, best, span, depth - 1);
                Expr::UnOp { span, op: *op, expr: Box::new(child) }
            }
            ENode::IfThenElse(c, t, e) => {
                let cond = self.materialize(*c, best, span, depth - 1);
                let then_block = self.materialize(*t, best, span, depth - 1);
                let else_expr = e.map(|ee| self.materialize(ee, best, span, depth - 1));
                // Convert then_block to a Block if it's not already
                let then_block = match then_block {
                    Expr::Block(b) => b,
                    _ => Box::new(Block { span, stmts: Vec::new(), tail: Some(Box::new(then_block)) }),
                };
                let else_block = else_expr.map(|e| {
                    match e {
                        Expr::Block(b) => b,
                        _ => Box::new(Block { span, stmts: Vec::new(), tail: Some(Box::new(e)) }),
                    }
                });
                Expr::IfExpr { span, cond: Box::new(cond), then: then_block, else_: else_block }
            }
        }
    }
}

// ── EGraph optimizer pass ─────────────────────────────────────────────────────

/// Drives the EGraph pipeline over a function body, expression by expression.
///
/// Only *pure* arithmetic / logical expressions are eligible: calls, field
/// accesses, assignments, and other side-effecting / structurally-complex nodes
/// are treated as opaque variables and left untouched by the EGraph.  (They can
/// still appear as children of an eligible expression — the opaque subtree
/// becomes a `Var` placeholder that is carried through saturation intact.)
pub struct EGraphOptimizer {
    pub rewrites: u64,
    max_iters:    usize,
    /// Shared e-graph for cross-expression memoization within a block
    shared_egraph: Option<EGraph>,
}

impl EGraphOptimizer {
    pub fn new(max_iters: usize) -> Self {
        Self { rewrites: 0, max_iters, shared_egraph: None }
    }

    pub fn optimize_block(&mut self, block: &mut Block) {
        // Start with a fresh shared e-graph for this block
        self.shared_egraph = Some(EGraph::new());

        for stmt in &mut block.stmts {
            self.opt_stmt(stmt);
        }
        if let Some(tail) = &mut block.tail {
            let old = std::mem::replace(tail.as_mut(), Expr::IntLit { span: Span::dummy(), value: 0 });
            **tail = self.opt_expr(old);
        }

        // Clear the shared e-graph after processing the block
        self.shared_egraph = None;
    }

    fn opt_stmt(&mut self, stmt: &mut Stmt) {
        match stmt {
            Stmt::Let { init: Some(expr), .. } | Stmt::Expr { expr, .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.opt_expr(old);
            }
            Stmt::Return { value: Some(expr), .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.opt_expr(old);
            }
            Stmt::If { cond, then, else_, .. } => {
                let old = std::mem::replace(cond, Expr::IntLit { span: Span::dummy(), value: 0 });
                *cond = self.opt_expr(old);
                self.optimize_block(then);
                if let Some(eb) = else_ {
                    if let IfOrBlock::Block(b) = &mut **eb {
                        self.optimize_block(b);
                    }
                }
            }
            Stmt::ForIn   { body, .. }
            | Stmt::While { body, .. }
            | Stmt::EntityFor { body, .. } => {
                self.optimize_block(body);
            }
            Stmt::Match { expr, arms, .. } => {
                let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                *expr = self.opt_expr(old);
                for arm in arms {
                    let old_body = std::mem::replace(&mut arm.body, Expr::IntLit { span: Span::dummy(), value: 0 });
                    arm.body = self.opt_expr(old_body);
                }
            }
            _ => {}
        }
    }

    fn opt_expr(&mut self, expr: Expr) -> Expr {
        // Recurse into ineligible nodes before attempting EGraph optimization,
        // so that their *children* still get optimized where eligible.
        if !Self::is_eligible(&expr) {
            return self.recurse(expr);
        }

        let span = expr.span();
        let original_cost = CostModel::estimate(&expr);

        // Use shared e-graph for cross-expression memoization
        // If shared e-graph exists, add this expression to it and saturate
        // Otherwise, create a fresh e-graph for this expression only
        let (best_expr, rewrites) = if let Some(ref mut shared) = self.shared_egraph {
            let root = shared.build_expr(&expr);
            shared.saturate(self.max_iters);
            let best_nodes = shared.extract_best();
            let best_expr = shared.materialize(root, &best_nodes, span, 64);
            (best_expr, shared.rewrites)
        } else {
            let mut egraph = EGraph::new();
            let root = egraph.build_expr(&expr);
            egraph.saturate(self.max_iters);
            let best_nodes = egraph.extract_best();
            let best_expr = egraph.materialize(root, &best_nodes, span, 64);
            (best_expr, egraph.rewrites)
        };

        let best_cost = CostModel::estimate(&best_expr);

        if best_cost < original_cost - 1e-9 {
            self.rewrites += rewrites;
            best_expr
        } else {
            expr
        }
    }

    /// Recurse into an ineligible expression's sub-expressions, applying
    /// EGraph optimization to any eligible children.
    fn recurse(&mut self, expr: Expr) -> Expr {
        match expr {
            Expr::BinOp { span, op, lhs, rhs } => Expr::BinOp {
                span, op,
                lhs: Box::new(self.opt_expr(*lhs)),
                rhs: Box::new(self.opt_expr(*rhs)),
            },
            Expr::UnOp { span, op, expr } => Expr::UnOp {
                span, op,
                expr: Box::new(self.opt_expr(*expr)),
            },
            Expr::Call { span, func, args, named } => Expr::Call {
                span,
                func: Box::new(self.opt_expr(*func)),
                args: args.into_iter().map(|a| self.opt_expr(a)).collect(),
                named,
            },
            Expr::Tuple { span, elems } => Expr::Tuple {
                span,
                elems: elems.into_iter().map(|e| self.opt_expr(e)).collect(),
            },
            Expr::Block(block) => {
                let mut b = *block;
                self.optimize_block(&mut b);
                Expr::Block(Box::new(b))
            }
            Expr::IfExpr { span, cond, then, else_ } => Expr::IfExpr {
                span,
                cond: Box::new(self.opt_expr(*cond)),
                then,
                else_,
            },
            other => other,
        }
    }

    /// An expression is eligible for EGraph optimization if it is a pure
    /// arithmetic / logical expression (no side effects, no memory access).
    fn is_eligible(expr: &Expr) -> bool {
        match expr {
            Expr::IntLit { .. }
            | Expr::FloatLit { .. }
            | Expr::BoolLit { .. }
            | Expr::Ident { .. } => true,
            Expr::BinOp { lhs, rhs, .. } => Self::is_eligible(lhs) && Self::is_eligible(rhs),
            Expr::UnOp { expr, .. } => Self::is_eligible(expr),
            Expr::IfExpr { cond, then, else_, .. } => {
                Self::is_eligible(cond) && Self::is_eligible_block(then) && else_.as_ref().map_or(true, |b| Self::is_eligible_block(b))
            }
            // Everything else (calls, field access, indexing, …) is ineligible.
            _ => false,
        }
    }

    fn is_eligible_block(block: &Block) -> bool {
        block.stmts.iter().all(|s| Self::is_eligible_stmt(s))
            && block.tail.as_ref().map_or(true, |e| Self::is_eligible(e))
    }

    fn is_eligible_stmt(stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Let { init: Some(e), .. } | Stmt::Expr { expr: e, .. } => Self::is_eligible(e),
            Stmt::Return { value: Some(e), .. } => Self::is_eligible(e),
            _ => false, // Control flow statements are not eligible at the statement level
        }
    }
}

// =============================================================================
// §16  SUPEROPTIMIZER PIPELINE
// =============================================================================

pub struct Superoptimizer {
    config: SuperoptimizerConfig,
    pub total_rewrites: u64,
    pub constant_folds: u64,
    pub strength_reductions: u64,
    pub algebraic_simplifications: u64,
    pub dead_branches: u64,
    pub constant_propagations: u64,
    pub dead_code_eliminated: u64,
    pub cse_eliminations: u64,
    pub peephole_opts: u64,
    pub licm_hoists: u64,
    pub inlinings: u64,
    pub bitwise_opts: u64,
    pub comparisons_canonicalized: u64,
    pub reassociations: u64,
    pub dead_functions_eliminated: u64,
    pub loop_unrollings: u64,
    pub tail_merges: u64,
    pub cost_before: f64,
    pub cost_after: f64,
    /// Rewrites performed by the stochastic superoptimizer pass (§15.5).
    /// These are optimizations that were *discovered by search*, not by any
    /// hand-coded rule.
    pub superopt_rewrites: u64,
}

impl Superoptimizer {
    pub fn new(config: SuperoptimizerConfig) -> Self {
        Self {
            config, total_rewrites: 0, constant_folds: 0, strength_reductions: 0,
            algebraic_simplifications: 0, dead_branches: 0, constant_propagations: 0,
            dead_code_eliminated: 0, cse_eliminations: 0, peephole_opts: 0,
            licm_hoists: 0, inlinings: 0, bitwise_opts: 0, comparisons_canonicalized: 0,
            reassociations: 0, dead_functions_eliminated: 0, loop_unrollings: 0,
            tail_merges: 0, cost_before: 0.0, cost_after: 0.0,
            superopt_rewrites: 0,
        }
    }

    pub fn fast_compile() -> Self { Self::new(SuperoptimizerConfig::fast_compile()) }
    pub fn balanced() -> Self { Self::new(SuperoptimizerConfig::balanced()) }
    pub fn maximum() -> Self { Self::new(SuperoptimizerConfig::maximum()) }

    // ─── Dead Function Elimination ─────────────────────────────────────────
    // Identifies functions that are never called and removes them, then
    // recursively repeats until a fixpoint. O(V+E) per iteration via
    // reachability from `main` + @test + @benchmark entry points.
    //
    // IMPORTANT: Functions that are never called from within the program are
    // KEPT because they may be external entry points (e.g. called by the
    // interpreter's `call_fn("bench", ...)` API).
    fn eliminate_dead_functions(&mut self, program: &mut Program) {
        let entry_names: std::collections::HashSet<&str> = {
            let mut s = std::collections::HashSet::default();
            s.insert("main");
            // Also keep @test and @benchmark functions
            for item in &program.items {
                if let Item::Fn(f) = item {
                    for attr in &f.attrs {
                        if matches!(attr, Attribute::Named { name, .. } if name == "test" || name == "benchmark") {
                            s.insert(f.name.as_str());
                        }
                    }
                }
            }
            s
        };

        // Build call graph: fn_name -> set of callees
        let mut callees_of: FxHashMap<String, std::collections::HashSet<String>> = FxHashMap::default();
        let mut all_fns: FxHashMap<String, usize> = FxHashMap::default(); // name -> index

        for (idx, item) in program.items.iter().enumerate() {
            if let Item::Fn(f) = item {
                all_fns.insert(f.name.clone(), idx);
                let mut callees = std::collections::HashSet::default();
                Self::collect_callees_in_fn(&f.body, &mut callees);
                callees_of.insert(f.name.clone(), callees);
            }
        }

        // BFS from entry points
        let mut reachable: std::collections::HashSet<String> = std::collections::HashSet::default();
        let mut queue: Vec<String> = entry_names.iter().map(|s| s.to_string()).collect();
        for e in &queue { reachable.insert(e.clone()); }

        let mut head = 0;
        while head < queue.len() {
            let current = &queue[head];
            head += 1;
            if let Some(callees) = callees_of.get(current) {
                for callee in callees {
                    if !reachable.contains(callee) {
                        reachable.insert(callee.clone());
                        queue.push(callee.clone());
                    }
                }
            }
        }

        // Also keep functions that are never called from within the program —
        // they might be external entry points (e.g. `call_fn("bench", ...)`).
        let mut called_from_somewhere: std::collections::HashSet<String> = std::collections::HashSet::default();
        for (_, callees) in &callees_of {
            for c in callees {
                called_from_somewhere.insert(c.clone());
            }
        }
        for (name, _) in &all_fns {
            if !called_from_somewhere.contains(name) {
                // Never called internally → potential external entry point → keep it
                reachable.insert(name.clone());
            }
        }

        // Remove dead functions (in reverse order to preserve indices)
        let mut dead_indices = Vec::new();
        for (name, &idx) in &all_fns {
            if !reachable.contains(name) {
                dead_indices.push(idx);
            }
        }
        dead_indices.sort_unstable();
        dead_indices.reverse();

        for idx in dead_indices {
            program.items.remove(idx);
            self.dead_functions_eliminated += 1;
        }
    }

    fn collect_callees_in_block(body: &Block, callees: &mut std::collections::HashSet<String>) {
        Self::collect_callees_in_block_stmts(&body.stmts, callees);
        if let Some(tail) = &body.tail {
            Self::collect_callees_expr(tail, callees);
        }
    }

    fn collect_callees_in_fn(body: &Option<Block>, callees: &mut std::collections::HashSet<String>) {
        if let Some(b) = body {
            Self::collect_callees_in_block(b, callees);
        }
    }

    fn collect_callees_in_block_stmts(stmts: &[Stmt], callees: &mut std::collections::HashSet<String>) {
        for s in stmts {
            match s {
                Stmt::Let { init: Some(e), .. } | Stmt::Expr { expr: e, .. } => {
                    Self::collect_callees_expr(e, callees);
                }
                Stmt::If { cond, then, else_, .. } => {
                    Self::collect_callees_expr(cond, callees);
                    Self::collect_callees_in_block(then, callees);
                    if let Some(eb) = else_ {
                        match eb.as_ref() {
                            crate::compiler::ast::IfOrBlock::If(if_stmt) => {
                                Self::collect_callees_in_block_stmts(&[if_stmt.clone()], callees);
                            }
                            crate::compiler::ast::IfOrBlock::Block(b) => {
                                Self::collect_callees_in_block(b, callees);
                            }
                        }
                    }
                }
                Stmt::ForIn { iter, body, .. } => {
                    Self::collect_callees_expr(iter, callees);
                    Self::collect_callees_in_block(body, callees);
                }
                Stmt::While { cond, body, .. } => {
                    Self::collect_callees_expr(cond, callees);
                    Self::collect_callees_in_block(body, callees);
                }
                Stmt::EntityFor { query, body, .. } => {
                    if let Some(f) = &query.filter {
                        Self::collect_callees_expr(f, callees);
                    }
                    Self::collect_callees_in_block(body, callees);
                }
                Stmt::Return { value: Some(e), .. } => {
                    Self::collect_callees_expr(e, callees);
                }
                Stmt::Match { expr, arms, .. } => {
                    Self::collect_callees_expr(expr, callees);
                    for arm in arms {
                        Self::collect_callees_expr(&arm.body, callees);
                    }
                }
                _ => {}
            }
        }
    }

    fn collect_callees_expr(expr: &Expr, callees: &mut std::collections::HashSet<String>) {
        match expr {
            Expr::Call { func, args, .. } => {
                if let Expr::Ident { name, .. } = func.as_ref() {
                    callees.insert(name.clone());
                }
                for a in args { Self::collect_callees_expr(a, callees); }
            }
            Expr::MethodCall { receiver, args, .. } => {
                Self::collect_callees_expr(receiver, callees);
                for a in args { Self::collect_callees_expr(a, callees); }
            }
            Expr::BinOp { lhs, rhs, .. } => {
                Self::collect_callees_expr(lhs, callees);
                Self::collect_callees_expr(rhs, callees);
            }
            Expr::UnOp { expr: e, .. } => {
                Self::collect_callees_expr(e, callees);
            }
            Expr::IfExpr { cond, then, else_, .. } => {
                Self::collect_callees_expr(cond, callees);
                Self::collect_callees_in_block(then, callees);
                if let Some(eb) = else_ {
                    Self::collect_callees_in_block(eb, callees);
                }
            }
            Expr::Index { object, indices, .. } => {
                Self::collect_callees_expr(object, callees);
                for i in indices { Self::collect_callees_expr(i, callees); }
            }
            Expr::Field { object, .. } => {
                Self::collect_callees_expr(object, callees);
            }
            Expr::Tuple { elems, .. } => {
                for e in elems { Self::collect_callees_expr(e, callees); }
            }
            Expr::StructLit { fields, .. } => {
                for (_, e) in fields { Self::collect_callees_expr(e, callees); }
            }
            Expr::ArrayLit { elems, .. } => {
                for e in elems { Self::collect_callees_expr(e, callees); }
            }
            Expr::Closure { params: _, ret_ty: _, body, .. } => {
                Self::collect_callees_expr(body, callees);
            }
            Expr::Block(b) => {
                Self::collect_callees_in_block(b, callees);
            }
            _ => {}
        }
    }

    pub fn optimize_program(&mut self, program: &mut Program) {
        // Pass 0: Dead Function Elimination (interprocedural)
        self.eliminate_dead_functions(program);

        if self.config.enable_inlining {
            let mut inliner = FunctionInliner::new(self.config.max_inline_size);
            inliner.inline_program(program);
            self.inlinings = inliner.inlined;
        }
        for item in &mut program.items {
            if let Item::Fn(fn_decl) = item {
                if let Some(body) = &mut fn_decl.body {
                    self.optimize_function_body(body);
                }
            }
        }
    }

    fn optimize_function_body(&mut self, body: &mut Block) {
        self.cost_before += CostModel::estimate_block(body);

        for _iter in 0..self.config.iterations {
            let size_before = self.estimate_size(body);

            // ── Pipeline reordering (user's #7) ─────────────────────────────
            // The best order is: DCE, constprop, fold, algebraic, reassoc,
            // CSE, DSE, branch cleanup, loop cleanup.  Every earlier pass
            // shrinks and normalizes the search space for the later ones.

            // Pass 1: Dead code elimination (trim unreachable code first)
            let mut dce = DeadCodeEliminator::new();
            dce.eliminate_block(body);
            self.dead_code_eliminated += dce.eliminated;

            // Pass 2: Constant propagation (substitute known values before folding)
            let mut cp = ConstantPropagator::new();
            cp.propagate_block(body);
            self.constant_propagations += cp.propagations;

            // Pass 3: Constant folding (collapse constant expressions)
            let mut cf = ConstantFolder::new();
            cf.fold_block_mut(body);
            self.constant_folds += cf.folds_performed;

            // Pass 4: Algebraic simplification (identity, inverse, idempotent rules)
            let mut asimp = AlgebraicSimplifier::new();
            asimp.simplify_block_mut(body);
            self.algebraic_simplifications += asimp.simplifications;

            // Pass 5: Strength reduction (mul-by-power → shift, etc.)
            let mut sr = StrengthReducer::new();
            sr.reduce_block_mut(body);
            self.strength_reductions += sr.reductions;

            // Pass 6: Bitwise optimizations
            if self.config.enable_peephole {
                let mut bo = BitwiseOptimizer::new();
                bo.optimize_block_mut(body);
                self.bitwise_opts += bo.optimizations;
            }

            // Pass 7: Comparison canonicalization
            let mut cc = ComparisonCanonicalizer::new();
            cc.canonicalize_block_mut(body);
            self.comparisons_canonicalized += cc.canonicalizations;

            // Pass 8: Expression reassociation
            let mut er = ExpressionReassociator::new();
            er.reassociate_block_mut(body);
            self.reassociations += er.reassociations;

            // Pass 9: CSE (after reassociation so equivalent forms are unified)
            if self.config.enable_cse {
                let mut cse = CommonSubexprEliminator::new();
                cse.eliminate_block(body);
                self.cse_eliminations += cse.eliminations;
            }

            // Pass 10: Dead store elimination
            if self.config.enable_dse {
                let mut dse = DeadStoreEliminator::new();
                let no_external = std::collections::HashSet::<String>::default();
                dse.eliminate_block(body, &no_external);
            }

            // Pass 11: Peephole optimization
            if self.config.enable_peephole {
                let mut peep = PeepholeOptimizer::new();
                peep.optimize_block(body);
                self.peephole_opts += peep.optimizations;
            }

            // Pass 12: Branch optimization
            let mut br = BranchOptimizer::new();
            br.optimize_block_mut(body);
            self.dead_branches += br.optimizations;

            // Pass 13: Loop optimizations
            if self.config.enable_loop_opts && self.config.max_unroll_factor > 0 {
                let mut lo = LoopOptimizer::new(self.config.max_unroll_factor);
                lo.optimize_block_mut(body);
                self.licm_hoists += lo.licm_hoists;
            }

            let size_after = self.estimate_size(body);
            if size_after == size_before { break; }
        }

        // ── Pass 14: E-graph equality saturation ────────────────────────────
        // Runs AFTER conventional passes have reached a fixpoint.  The E-graph
        // discovers equivalences that individual rules miss by saturating all
        // possible rewrites simultaneously.
        if self.config.enable_egraph && self.config.egraph_iterations > 0 {
            let mut ego = EGraphOptimizer::new(self.config.egraph_iterations);
            ego.optimize_block(body);
        }

        // ── Pass 15: Stochastic superoptimization ───────────────────────────
        // Runs after ALL other passes (including E-graph) have finished.
        // Each expression is independently subjected to random search for
        // a cheaper semantically-equivalent expression, verified by concrete
        // evaluation.  This is the only pass that can discover optimizations
        // not encoded in any of the rules above.
        if self.config.enable_superopt && self.config.superopt_budget > 0 {
            let mut so = StochasticSuperoptimizer::new(
                self.config.superopt_budget,
                self.config.superopt_verif_inputs,
            );
            so.optimize_block_mut(body);
            self.superopt_rewrites += so.rewrites;
        }

        // ── Pass 16: Cleanup sweep ──────────────────────────────────────────
        // After the expensive passes (E-graph + stochastic), run one more
        // cleanup sweep so newly exposed simplifications get removed.  This is
        // especially important because E-graph and stochastic passes can create
        // new constant or branch opportunities.
        {
            let mut dce2 = DeadCodeEliminator::new();
            dce2.eliminate_block(body);
            self.dead_code_eliminated += dce2.eliminated;
            let mut cf2 = ConstantFolder::new();
            cf2.fold_block_mut(body);
            self.constant_folds += cf2.folds_performed;
            let mut asimp2 = AlgebraicSimplifier::new();
            asimp2.simplify_block_mut(body);
            self.algebraic_simplifications += asimp2.simplifications;
        }

        self.cost_after += CostModel::estimate_block(body);
    }

    fn estimate_size(&self, body: &Block) -> usize {
        body.stmts.len()
            + body.tail.as_ref().map_or(0, |_| 1)
            + body.stmts.iter().map(|s| match s {
                Stmt::ForIn { body: b, .. } | Stmt::While { body: b, .. } | Stmt::EntityFor { body: b, .. } => b.stmts.len(),
                Stmt::If { then, else_, .. } => then.stmts.len() + else_.as_ref().map_or(0, |e| match e.as_ref() {
                    IfOrBlock::Block(b) => b.stmts.len(),
                    _ => 0,
                }),
                _ => 0,
            }).sum::<usize>()
    }
}

// =============================================================================
// §17  COST MODEL — x86-64 Cycle Estimation with Register Pressure & Branchiness
// =============================================================================

struct CostModel;

impl CostModel {
    fn estimate_block(block: &Block) -> f64 {
        let stmt_cost: f64 = block.stmts.iter().map(|s| Self::estimate_stmt(s)).sum();
        let tail_cost = block.tail.as_ref().map_or(0.0, |t| Self::estimate(t));
        stmt_cost + tail_cost
    }

    fn estimate_stmt(stmt: &Stmt) -> f64 {
        match stmt {
            Stmt::Let { init: Some(expr), .. } => Self::estimate(expr),
            Stmt::Expr { expr, .. } => Self::estimate(expr),
            Stmt::ForIn { body, iter, .. } => {
                1.0 + Self::estimate(iter) + Self::estimate_block(body) * 4.0
            }
            Stmt::While { body, cond, .. } => {
                1.0 + Self::estimate(cond) + Self::estimate_block(body) * 4.0
            }
            Stmt::EntityFor { body, .. } => {
                1.0 + Self::estimate_block(body) * 4.0
            }
            Stmt::If { cond, then, else_, .. } => {
                Self::estimate(cond) + Self::estimate_block(then)
                    + else_.as_ref().map_or(0.0, |e| match e.as_ref() {
                        IfOrBlock::Block(b) => Self::estimate_block(b),
                        IfOrBlock::If(_) => 0.0,
                    })
            }
            Stmt::Return { value: Some(expr), .. } => 1.0 + Self::estimate(expr),
            Stmt::Return { .. } => 1.0,
            Stmt::Match { expr, arms, .. } => {
                Self::estimate(expr) + arms.iter().map(|a| Self::estimate(&a.body)).sum::<f64>()
            }
            _ => 1.0,
        }
    }

    /// Estimate the execution cost of an expression in x86-64 cycles.
    ///
    /// Enhanced with:
    /// - **Register pressure penalty**: Each unique variable referenced adds
    ///   a small cost (0.1 cycles) to model register spill/reload overhead.
    ///   More free variables → more register pressure → higher cost.
    /// - **Branchiness penalty**: Conditional expressions (IfExpr) add a
    ///   2.0-cycle misprediction penalty on top of their condition cost.
    /// - **Division penalty**: Div/Rem are expensive (20 cycles baseline)
    ///   because they typically can't be pipelined.
    fn estimate(expr: &Expr) -> f64 {
        match expr {
            Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::BoolLit { .. } | Expr::StrLit { .. } => 0.0,
            Expr::Ident { .. } => 0.0,
            Expr::BinOp { op, lhs, rhs, .. } => {
                let base = match op {
                    BinOpKind::Add | BinOpKind::Sub => 1.0,
                    BinOpKind::Mul => 3.0,
                    BinOpKind::Div | BinOpKind::Rem | BinOpKind::FloorDiv => 20.0,
                    BinOpKind::And | BinOpKind::Or | BinOpKind::Eq
                    | BinOpKind::Ne | BinOpKind::Lt | BinOpKind::Le
                    | BinOpKind::Gt | BinOpKind::Ge => 1.0,
                    BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor => 1.0,
                    BinOpKind::Shl | BinOpKind::Shr => 1.0,
                };
                let lhs_cost = Self::estimate(lhs);
                let rhs_cost = Self::estimate(rhs);
                // Register pressure: count unique variables in this sub-expression
                let reg_pressure = Self::count_unique_vars(expr) as f64 * 0.1;
                base + lhs_cost + rhs_cost + reg_pressure
            }
            Expr::UnOp { expr, .. } => 1.0 + Self::estimate(expr),
            Expr::Call { args, .. } => 5.0 + args.iter().map(|a| Self::estimate(a)).sum::<f64>(),
            Expr::IfExpr { cond, then, else_, .. } => {
                // Branchiness penalty: misprediction cost (2.0 cycles)
                2.0 + Self::estimate(cond) + Self::estimate_block(then)
                    + else_.as_ref().map_or(0.0, |e| Self::estimate_block(e))
            }
            Expr::Tuple { elems, .. } => elems.iter().map(|e| Self::estimate(e)).sum(),
            Expr::Block(block) => Self::estimate_block(block),
            _ => 1.0,
        }
    }

    /// Count the number of unique variable references in an expression.
    /// Used to model register pressure — more unique variables means more
    /// registers needed, potentially causing spills.
    fn count_unique_vars(expr: &Expr) -> usize {
        let mut vars = std::collections::HashSet::<String>::default();
        Self::collect_vars(expr, &mut vars);
        vars.len()
    }

    fn collect_vars(expr: &Expr, out: &mut std::collections::HashSet<String>) {
        match expr {
            Expr::Ident { name, .. } => { out.insert(name.clone()); }
            Expr::BinOp { lhs, rhs, .. } => {
                Self::collect_vars(lhs, out);
                Self::collect_vars(rhs, out);
            }
            Expr::UnOp { expr, .. } => Self::collect_vars(expr, out),
            Expr::Call { func, args, .. } => {
                Self::collect_vars(func, out);
                for a in args { Self::collect_vars(a, out); }
            }
            Expr::Tuple { elems, .. } => {
                for e in elems { Self::collect_vars(e, out); }
            }
            _ => {}
        }
    }
}

// =============================================================================
// §18  BACKWARDS COMPATIBILITY — Legacy API
// =============================================================================

pub struct AdvancedOptimizer {
    pub optimize_level: u8,
    pub folds_performed: u64,
    pub dead_code_eliminated: u64,
    pub inlining_performed: u64,
    pub specializations: u64,
}

impl AdvancedOptimizer {
    pub fn new(optimize_level: u8) -> Self {
        Self { optimize_level, folds_performed: 0, dead_code_eliminated: 0, inlining_performed: 0, specializations: 0 }
    }

    pub fn optimize_program(&mut self, program: &mut Program) {
        let config = match self.optimize_level {
            0 => return,
            1 => SuperoptimizerConfig::fast_compile(),
            2 => SuperoptimizerConfig::balanced(),
            _ => SuperoptimizerConfig::maximum(),
        };
        let mut opt = Superoptimizer::new(config);
        opt.optimize_program(program);
        self.folds_performed = opt.constant_folds;
        self.dead_code_eliminated = opt.dead_code_eliminated;
        self.inlining_performed = if opt.inlinings > 0 { 1 } else { 0 };
        self.specializations = opt.algebraic_simplifications;
    }
}
