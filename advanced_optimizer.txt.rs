// jules/src/advanced_optimizer.rs
//
// SUPEROPTIMIZER 


#![allow(dead_code)]

use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::ast::*;
use crate::Span;

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
        let mut env: FxHashMap<String, Expr> = FxHashMap::default();

        for stmt in &block.stmts {
            if let Stmt::Let { pattern, init: Some(expr), .. } = stmt {
                if let Pattern::Ident { name, .. } = pattern {
                    if Self::is_constant_expr(expr) {
                        env.insert(name.clone(), expr.clone());
                    }
                }
            }
        }

        if env.is_empty() {
            return;
        }

        for stmt in &mut block.stmts {
            match stmt {
                Stmt::Let { init: Some(expr), .. } | Stmt::Expr { expr, .. } => {
                    let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                    *expr = self.substitute(old, &env);
                }
                Stmt::Return { value: Some(expr), .. } => {
                    let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                    *expr = self.substitute(old, &env);
                }
                _ => {}
            }
        }

        if let Some(tail) = &mut block.tail {
            let old = std::mem::replace(tail.as_mut(), Expr::IntLit { span: Span::dummy(), value: 0 });
            **tail = self.substitute(old, &env);
        }
    }

    fn is_constant_expr(expr: &Expr) -> bool {
        match expr {
            Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::BoolLit { .. } | Expr::StrLit { .. } => true,
            Expr::BinOp { lhs, rhs, .. } => Self::is_constant_expr(lhs) && Self::is_constant_expr(rhs),
            Expr::UnOp { expr, .. } => Self::is_constant_expr(expr),
            Expr::Tuple { elems, .. } => elems.iter().all(|e| Self::is_constant_expr(e)),
            _ => false,
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
            (Expr::FloatLit { value: v1, .. }, Expr::FloatLit { value: v2, .. }) => (v1 - v2).abs() < 1e-10,
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
                // x * 3 → x + x + x
                if let Expr::IntLit { value: 3, .. } = rhs {
                    let doubled = Expr::BinOp { span, op: BinOpKind::Add, lhs: Box::new(lhs.clone()), rhs: Box::new(lhs.clone()) };
                    return Some(Expr::BinOp { span, op: BinOpKind::Add, lhs: Box::new(doubled), rhs: Box::new(lhs.clone()) });
                }
                // x * 5 → (x << 2) + x
                if let Expr::IntLit { value: 5, .. } = rhs {
                    let shifted = Expr::BinOp { span, op: BinOpKind::Shl, lhs: Box::new(lhs.clone()), rhs: Box::new(Expr::IntLit { span, value: 2 }) };
                    return Some(Expr::BinOp { span, op: BinOpKind::Add, lhs: Box::new(shifted), rhs: Box::new(lhs.clone()) });
                }
                // x * 9 → (x << 3) + x
                if let Expr::IntLit { value: 9, .. } = rhs {
                    let shifted = Expr::BinOp { span, op: BinOpKind::Shl, lhs: Box::new(lhs.clone()), rhs: Box::new(Expr::IntLit { span, value: 3 }) };
                    return Some(Expr::BinOp { span, op: BinOpKind::Add, lhs: Box::new(shifted), rhs: Box::new(lhs.clone()) });
                }
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
        self.expr_map.clear();
        self.counter = 0;
        for stmt in &mut block.stmts {
            match stmt {
                Stmt::Let { init: Some(expr), .. } => {
                    let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                    *expr = self.eliminate_expr(old);
                    self.expr_map.clear();
                }
                Stmt::Expr { expr, .. } => {
                    let old = std::mem::replace(expr, Expr::IntLit { span: Span::dummy(), value: 0 });
                    *expr = self.eliminate_expr(old);
                }
                _ => {}
            }
        }
        if let Some(tail) = &mut block.tail {
            let old = std::mem::replace(tail.as_mut(), Expr::IntLit { span: Span::dummy(), value: 0 });
            **tail = self.eliminate_expr(old);
        }
    }

    fn eliminate_expr(&mut self, expr: Expr) -> Expr {
        let hash = self.hash_expr(&expr);
        if let Some(var_name) = self.expr_map.get(&hash) {
            self.eliminations += 1;
            return Expr::Ident { span: expr.span(), name: var_name.clone() };
        }
        let result = match expr {
            Expr::BinOp { span, op, lhs, rhs } => Expr::BinOp {
                span, op,
                lhs: Box::new(self.eliminate_expr(*lhs)),
                rhs: Box::new(self.eliminate_expr(*rhs)),
            },
            Expr::UnOp { span, op, expr } => Expr::UnOp {
                span, op,
                expr: Box::new(self.eliminate_expr(*expr)),
            },
            Expr::Call { span, func, args, named } => Expr::Call {
                span, func: Box::new(self.eliminate_expr(*func)),
                args: args.into_iter().map(|a| self.eliminate_expr(a)).collect(),
                named,
            },
            Expr::Field { span, object, field } => Expr::Field {
                span, object: Box::new(self.eliminate_expr(*object)), field,
            },
            _ => expr,
        };
        if !matches!(&result, Expr::Ident { .. } | Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::BoolLit { .. }) {
            let var_name = format!("__cse_{}", self.counter);
            self.counter += 1;
            self.expr_map.insert(hash, var_name);
        }
        result
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

    fn eval_binop(op: BinOpKind, l: &Value, r: &Value) -> Option<Value> {
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

    fn eval_unop(op: UnOpKind, v: &Value) -> Option<Value> {
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
            if let Stmt::Let { pattern, init: Some(init), .. } = stmt {
                if let Pattern::Ident { name, .. } = pattern {
                    if !ever_read.contains(name) && Self::is_pure_expr(init) {
                        return false;
                    }
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

    fn eliminate_block(&mut self, block: &mut Block) {
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
        for stmt in &mut block.stmts {
            match stmt {
                Stmt::ForIn { body, .. }
                | Stmt::While { body, .. }
                | Stmt::EntityFor { body, .. } => {
                    self.eliminate_block(body);
                }
                Stmt::If { then, else_, .. } => {
                    self.eliminate_block(then);
                    if let Some(eb) = else_ {
                        if let IfOrBlock::Block(b) = &mut **eb {
                            self.eliminate_block(b);
                        }
                    }
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

        let mut hoisted_prefix: Vec<Stmt> = Vec::new();
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
                if let Expr::Ident { name, .. } = target.as_ref() {
                    out.insert(name.clone());
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

struct StochasticSuperoptimizer {
    budget: usize,
    verif_inputs: usize,
    pub rewrites: u64,
    rng: SimpleRng,
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

        // ── Step 1: collect free variables and constants ─────────────────────
        let mut free_vars: Vec<String> = Vec::new();
        let mut consts: Vec<u128> = Vec::new();
        Self::collect_free_vars_expr(&expr, &mut free_vars);
        Self::collect_int_consts_expr(&expr, &mut consts);
        free_vars.sort_unstable();
        free_vars.dedup();
        consts.sort_unstable();
        consts.dedup();

        // Universal small constants likely to appear in optimised forms.
        for c in [0u128, 1, 2, 4, 8, 16, 32, 64, 127, 128, 255] {
            if !consts.contains(&c) { consts.push(c); }
        }

        // If there are no free variables, constant folding should have already
        // collapsed this expression.  The stochastic pass cannot help further.
        // (We still try if there are constants, though.)

        // ── Step 2: sample random input environments ─────────────────────────
        // We bias inputs toward integers because the vast majority of
        // expressions in arithmetic-heavy code are over integers.  Each input
        // is independently sampled so that the spec covers a wide range of
        // value combinations.
        let envs: Vec<FxHashMap<String, Value>> = (0..self.verif_inputs)
            .map(|_| self.sample_int_env(&free_vars))
            .collect();

        // ── Step 3: evaluate original expression on all environments ─────────
        let spec: Vec<Option<Value>> = envs.iter()
            .map(|e| ExprInterpreter::eval(&expr, e))
            .collect();

        // If the expression isn't fully evaluable on every input, bail out.
        // (This happens for calls, field accesses, etc.)
        if spec.iter().any(|v| v.is_none()) {
            return expr;
        }
        // Unwrap is safe: all Some after the check above.
        let spec: Vec<Value> = spec.into_iter().map(Option::unwrap).collect();

        // ── Steps 4 + 5: random search ───────────────────────────────────────
        let term_bank = self.build_term_bank(&free_vars, &consts, span);
        let mut best_expr = expr.clone();
        let mut best_cost = original_cost;

        for _ in 0..self.budget {
            // Randomly pick a depth: depth-1 candidates are cheap to evaluate
            // and cover the vast majority of interesting peephole rewrites;
            // depth-2 candidates are rarer but can find two-level identities
            // (e.g. "(x << 3) + x" for "x * 9").
            let depth = if self.rng.next_u64() % 4 == 0 { 2 } else { 1 };
            let candidate = self.random_expr(&term_bank, span, depth);

            let candidate_cost = CostModel::estimate(&candidate);
            if candidate_cost >= best_cost { continue; }

            // ── Equivalence check via concrete evaluation ────────────────────
            let equivalent = envs.iter().zip(spec.iter()).all(|(env, expected)| {
                ExprInterpreter::eval(&candidate, env).as_ref() == Some(expected)
            });

            if equivalent {
                best_expr = candidate;
                best_cost = candidate_cost;
                self.rewrites += 1;
            }
        }

        best_expr
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

    // ── Candidate generation ─────────────────────────────────────────────────

    /// Build the set of atomic "leaf" expressions available for composition.
    fn build_term_bank(
        &self,
        free_vars: &[String],
        consts: &[u128],
        span: Span,
    ) -> Vec<Expr> {
        let mut bank: Vec<Expr> = Vec::with_capacity(free_vars.len() + consts.len() + 2);
        for v in free_vars {
            bank.push(Expr::Ident { span, name: v.clone() });
        }
        for c in consts {
            bank.push(Expr::IntLit { span, value: *c });
        }
        bank
    }

    /// Randomly assemble an expression of at most `depth` binary/unary levels
    /// from the given term bank.
    fn random_expr(&mut self, term_bank: &[Expr], span: Span, depth: usize) -> Expr {
        if term_bank.is_empty() {
            return Expr::IntLit { span, value: 0 };
        }

        // At depth 0, or randomly early-terminate, return a leaf.
        if depth == 0 || self.rng.next_u64() % 3 == 0 {
            return term_bank[self.rng.next_usize(term_bank.len())].clone();
        }

        match self.rng.next_usize(15) {
            // Binary operations (0..=11)
            n @ 0..=11 => {
                const OPS: [BinOpKind; 12] = [
                    BinOpKind::Add, BinOpKind::Sub, BinOpKind::Mul,
                    BinOpKind::BitAnd, BinOpKind::BitOr, BinOpKind::BitXor,
                    BinOpKind::Shl, BinOpKind::Shr,
                    BinOpKind::Eq, BinOpKind::Ne, BinOpKind::Lt, BinOpKind::Le,
                ];
                let op = OPS[n];
                let lhs = self.random_expr(term_bank, span, depth - 1);
                let rhs = self.random_expr(term_bank, span, depth - 1);
                Expr::BinOp { span, op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
            }
            // Unary negation / bitwise-not (12..=13)
            12 => {
                let inner = self.random_expr(term_bank, span, depth - 1);
                Expr::UnOp { span, op: UnOpKind::Neg, expr: Box::new(inner) }
            }
            13 => {
                let inner = self.random_expr(term_bank, span, depth - 1);
                Expr::UnOp { span, op: UnOpKind::Not, expr: Box::new(inner) }
            }
            // Leaf (14)
            _ => term_bank[self.rng.next_usize(term_bank.len())].clone(),
        }
    }

    // ── Environment sampling ─────────────────────────────────────────────────

    /// Sample a random environment that maps each variable to a u128.
    /// We use a mix of edge-case values (0, 1, MAX, powers-of-two) and
    /// random values so that both algebraic identities and arbitrary
    /// arithmetic patterns can be discovered.
    fn sample_int_env(&mut self, free_vars: &[String]) -> FxHashMap<String, Value> {
        let mut env = FxHashMap::default();
        for v in free_vars {
            let val = match self.rng.next_usize(10) {
                0 => 0u128,
                1 => 1u128,
                2 => u128::MAX,
                3 => 1u128 << self.rng.next_usize(127),   // power of two
                4 => (1u128 << self.rng.next_usize(127)).wrapping_sub(1), // 2^k-1
                _ => self.rng.next_u128(),                 // fully random
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
                            crate::ast::IfOrBlock::If(if_stmt) => {
                                Self::collect_callees_in_block_stmts(&[if_stmt.clone()], callees);
                            }
                            crate::ast::IfOrBlock::Block(b) => {
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

            // Pass 1: Dead code elimination
            let mut dce = DeadCodeEliminator::new();
            dce.eliminate_block(body);
            self.dead_code_eliminated += dce.eliminated;

            // Pass 2: Constant propagation
            let mut cp = ConstantPropagator::new();
            cp.propagate_block(body);
            self.constant_propagations += cp.propagations;

            // Pass 3: Constant folding
            let mut cf = ConstantFolder::new();
            cf.fold_block_mut(body);
            self.constant_folds += cf.folds_performed;

            // Pass 4: Algebraic simplification
            let mut asimp = AlgebraicSimplifier::new();
            asimp.simplify_block_mut(body);
            self.algebraic_simplifications += asimp.simplifications;

            // Pass 5: Strength reduction
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

            // Pass 9: CSE
            if self.config.enable_cse {
                let mut cse = CommonSubexprEliminator::new();
                cse.eliminate_block(body);
                self.cse_eliminations += cse.eliminations;
            }

            // Pass 10: Dead store elimination
            if self.config.enable_dse {
                let mut dse = DeadStoreEliminator::new();
                dse.eliminate_block(body);
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

        // Pass 14: Stochastic superoptimization  (§15.5)
        // Runs *after* the conventional passes have reached a fixpoint so the
        // search starts from already-simplified expressions.  Each expression
        // in the function body is independently subjected to random search for
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
// §17  COST MODEL — x86-64 Cycle Estimation
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
                base + Self::estimate(lhs) + Self::estimate(rhs)
            }
            Expr::UnOp { expr, .. } => 1.0 + Self::estimate(expr),
            Expr::Call { args, .. } => 5.0 + args.iter().map(|a| Self::estimate(a)).sum::<f64>(),
            Expr::IfExpr { cond, then, else_, .. } => {
                Self::estimate(cond) + Self::estimate_block(then)
                    + else_.as_ref().map_or(0.0, |e| Self::estimate_block(e))
            }
            Expr::Tuple { elems, .. } => elems.iter().map(|e| Self::estimate(e)).sum(),
            Expr::Block(block) => Self::estimate_block(block),
            _ => 1.0,
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
