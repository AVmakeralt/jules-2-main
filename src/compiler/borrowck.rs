//! Borrow checker pass for Jules.
//!
//! This pass is intentionally conservative and lexical:
//! - enforces `&mut` exclusivity
//! - rejects writes while borrowed
//! - tracks simple moves (`let y = x`, `y = x`) and use-after-move

use rustc_hash::FxHashMap;

use crate::compiler::ast::{AssignOpKind, Block, Expr, Item, Pattern, Program, Stmt, UnOpKind};
use crate::compiler::lexer::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub span: Span,
    pub message: String,
    pub labels: Vec<(Span, String)>,
    /// Error code (e.g. "E4001") from the error_codes module.
    pub code: Option<&'static str>,
    /// Suggested fix hint (optional, displayed as `help:` in the renderer).
    pub hint: Option<String>,
}

impl Diagnostic {
    /// Attach an error code (e.g. `"E4001"`).
    pub fn with_code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }
    /// Attach a suggested fix hint.
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
}

#[derive(Debug, Default, Clone)]
pub struct Diagnostics {
    pub items: Vec<Diagnostic>,
}

impl Diagnostics {
    fn error(&mut self, span: Span, message: impl Into<String>) {
        self.items.push(Diagnostic {
            severity: Severity::Error,
            span,
            message: message.into(),
            labels: vec![],
            code: None,
            hint: None,
        });
    }

    fn error_with_fix(&mut self, span: Span, message: impl Into<String>, fix: impl Into<String>) {
        self.items.push(Diagnostic {
            severity: Severity::Error,
            span,
            message: message.into(),
            labels: vec![],
            code: None,
            hint: Some(fix.into()),
        });
    }

    /// Push a fully-constructed diagnostic (with code and hint already set).
    fn push_diag(&mut self, d: Diagnostic) {
        self.items.push(d);
    }
}

#[derive(Debug, Default, Clone)]
struct LoanState {
    imm: usize,
    mut_: usize,
}

#[derive(Debug, Default, Clone)]
struct VarState {
    moved: bool,
    is_copy: bool,
}

#[derive(Debug)]
struct BorrowChecker {
    diag: Diagnostics,
    loans_by_target: FxHashMap<String, LoanState>,
    ref_binding_to_target: FxHashMap<String, (String, bool)>, // ref_var -> (target, mut?)
    var_state: FxHashMap<String, VarState>,
    scope_bindings: Vec<Vec<String>>, // variable names declared in lexical scopes
    /// Depth counter for parallel constructs (ParallelFor, Spawn).
    /// Any value > 0 means we are inside a concurrent body where moves and
    /// `&mut` borrows of outer variables are forbidden.
    parallel_depth: usize,
}

impl Default for BorrowChecker {
    fn default() -> Self {
        Self {
            diag: Diagnostics::default(),
            loans_by_target: FxHashMap::default(),
            ref_binding_to_target: FxHashMap::default(),
            var_state: FxHashMap::default(),
            scope_bindings: vec![],
            parallel_depth: 0,
        }
    }
}

impl BorrowChecker {
    fn push_scope(&mut self) {
        self.scope_bindings.push(vec![]);
    }

    fn pop_scope(&mut self) {
        if let Some(vars) = self.scope_bindings.pop() {
            for v in vars {
                // Dropping a reference variable ends its loan.
                self.release_ref_binding(&v);
                self.var_state.remove(&v);
            }
        }
    }

    fn bind_var(&mut self, name: &str) {
        // If this name already holds a reference, drop that loan before rebinding.
        // Without this, shadowing a ref variable leaks its loan forever.
        self.release_ref_binding(name);
        let st = self.var_state.entry(name.to_string()).or_default();
        st.moved = false;
        st.is_copy = false;
        if let Some(scope) = self.scope_bindings.last_mut() {
            scope.push(name.to_string());
        }
    }

    fn mark_moved(&mut self, name: &str) {
        if let Some(st) = self.var_state.get_mut(name) {
            st.moved = true;
        }
    }

    fn mark_initialized(&mut self, name: &str) {
        self.var_state.entry(name.to_string()).or_default().moved = false;
    }

    fn set_copy_flag(&mut self, name: &str, is_copy: bool) {
        self.var_state.entry(name.to_string()).or_default().is_copy = is_copy;
    }

    fn release_ref_binding(&mut self, ref_name: &str) {
        let Some((target, is_mut)) = self.ref_binding_to_target.remove(ref_name) else {
            return;
        };
        let mut remove_target = false;
        if let Some(st) = self.loans_by_target.get_mut(&target) {
            if is_mut {
                st.mut_ = st.mut_.saturating_sub(1);
            } else {
                st.imm = st.imm.saturating_sub(1);
            }
            remove_target = st.imm == 0 && st.mut_ == 0;
        }
        if remove_target {
            self.loans_by_target.remove(&target);
        }
    }

    fn try_borrow(&mut self, target: &str, is_mut: bool, span: Span) -> bool {
        if self.var_state.get(target).map(|v| v.moved).unwrap_or(false) {
            self.diag.push_diag(
                Diagnostic {
                    severity: Severity::Error,
                    span,
                    message: format!("cannot borrow `{target}` because it has been moved"),
                    labels: vec![],
                    code: Some("E4002"),
                    hint: Some(format!("borrow `{target}` before moving it with `&{target}` or `.clone()`")),
                }
            );
            return false;
        }
        // Mutable borrows inside parallel/spawned bodies are forbidden: they
        // could alias with borrows in other concurrent threads.
        if is_mut && self.parallel_depth > 0 {
            self.diag.push_diag(
                Diagnostic {
                    severity: Severity::Error,
                    span,
                    message: format!("cannot mutably borrow `{target}` inside a parallel or spawned block"),
                    labels: vec![],
                    code: Some("E4003"),
                    hint: Some("mutable borrows of outer variables are not allowed inside parallel blocks; use `atomic { }` for shared mutation".into()),
                }
            );
            return false;
        }
        let st = self.loans_by_target.entry(target.to_string()).or_default();
        if is_mut {
            if st.mut_ > 0 || st.imm > 0 {
                self.diag.push_diag(
                    Diagnostic {
                        severity: Severity::Error,
                        span,
                        message: format!("cannot mutably borrow `{target}` because it is already borrowed"),
                        labels: vec![],
                        code: Some("E4004"),
                        hint: Some("wait for the existing borrow to end before mutably borrowing, or use a different variable".into()),
                    }
                );
                return false;
            }
            st.mut_ += 1;
            true
        } else {
            if st.mut_ > 0 {
                self.diag.push_diag(
                    Diagnostic {
                        severity: Severity::Error,
                        span,
                        message: format!(
                            "cannot immutably borrow `{target}` because it is already mutably borrowed"
                        ),
                        labels: vec![],
                        code: Some("E4005"),
                        hint: Some("wait for the mutable borrow to end, or reborrow from the same `&mut`".into()),
                    }
                );
                return false;
            }
            st.imm += 1;
            true
        }
    }

    fn check_read(&mut self, name: &str, span: Span) {
        if self.var_state.get(name).map(|v| v.moved).unwrap_or(false) {
            self.diag.push_diag(
                Diagnostic {
                    severity: Severity::Error,
                    span,
                    message: format!("use of moved value `{name}`"),
                    labels: vec![],
                    code: Some("E4001"),
                    hint: Some("use a reference `&name` instead of moving, or `.clone()` to make a copy".into()),
                }
            );
        }
        if let Some(st) = self.loans_by_target.get(name) {
            if st.mut_ > 0 {
                self.diag.error_with_fix(
                    span,
                    format!("cannot use `{name}` while it is mutably borrowed"),
                    "wait for the mutable borrow to end before accessing this variable",
                );
            }
        }
    }

    fn check_write(&mut self, name: &str, span: Span) {
        // Writing to a moved value is always an error: the storage no longer
        // belongs to this binding (it may have been transferred elsewhere).
        if self.var_state.get(name).map(|v| v.moved).unwrap_or(false) {
            self.diag.push_diag(
                Diagnostic {
                    severity: Severity::Error,
                    span,
                    message: format!("cannot assign to `{name}`: value has been moved"),
                    labels: vec![],
                    code: Some("E4006"),
                    hint: Some("reinitialize the variable before assignment, or use `let mut` with a new binding".into()),
                }
            );
        }
        if let Some(st) = self.loans_by_target.get(name) {
            if st.mut_ > 0 || st.imm > 0 {
                self.diag.error_with_fix(
                    span,
                    format!("cannot assign to `{name}` while it is borrowed"),
                    "wait for the borrow to end before assigning",
                );
            }
        }
    }

    fn move_ident(&mut self, name: &str, span: Span) {
        if self.var_state.get(name).map(|v| v.is_copy).unwrap_or(false) {
            self.check_read(name, span);
            return;
        }
        // Moving out of a variable inside a parallel body is unsafe: another
        // concurrent thread may still hold a reference to the same value.
        if self.parallel_depth > 0 {
            self.diag.push_diag(
                Diagnostic {
                    severity: Severity::Error,
                    span,
                    message: format!("cannot move `{name}` inside a parallel or spawned block"),
                    labels: vec![],
                    code: Some("E4008"),
                    hint: Some("moving outer variables into parallel blocks is unsafe; use `.clone()` or pass by reference".into()),
                }
            );
            return;
        }
        if let Some(st) = self.loans_by_target.get(name) {
            if st.mut_ > 0 || st.imm > 0 {
                self.diag.push_diag(
                    Diagnostic {
                        severity: Severity::Error,
                        span,
                        message: format!("cannot move `{name}` while it is borrowed"),
                        labels: vec![],
                        code: Some("E4009"),
                        hint: Some("wait for the borrow to end before moving".into()),
                    }
                );
                return;
            }
        }
        self.check_read(name, span);
        self.mark_moved(name);
    }

    /// Validate that a borrow is legal *without* mutating loan state.
    /// This is used by `check_expr` for inline borrow expressions (&x, &mut x)
    /// where the enclosing statement handler is responsible for the actual
    /// loan-state update.  Splitting validation from mutation prevents the
    /// double-borrow bug that previously caused loans to never be released.
    fn validate_borrow(&mut self, target: &str, is_mut: bool, span: Span) {
        if self.var_state.get(target).map(|v| v.moved).unwrap_or(false) {
            self.diag.push_diag(
                Diagnostic {
                    severity: Severity::Error,
                    span,
                    message: format!("cannot borrow `{target}` because it has been moved"),
                    labels: vec![],
                    code: Some("E4002"),
                    hint: Some(format!("borrow the value before moving it with `&{target}` or `.clone()`")),
                }
            );
            return;
        }
        if is_mut && self.parallel_depth > 0 {
            self.diag.push_diag(
                Diagnostic {
                    severity: Severity::Error,
                    span,
                    message: format!("cannot mutably borrow `{target}` inside a parallel or spawned block"),
                    labels: vec![],
                    code: Some("E4003"),
                    hint: Some("mutable borrows of outer variables are not allowed inside parallel blocks; use `atomic { }` for shared mutation".into()),
                }
            );
            return;
        }
        if let Some(st) = self.loans_by_target.get(target) {
            if is_mut && (st.mut_ > 0 || st.imm > 0) {
                self.diag.push_diag(
                    Diagnostic {
                        severity: Severity::Error,
                        span,
                        message: format!("cannot mutably borrow `{target}` because it is already borrowed"),
                        labels: vec![],
                        code: Some("E4004"),
                        hint: Some("wait for the existing borrow to end before mutably borrowing, or use a different variable".into()),
                    }
                );
            } else if !is_mut && st.mut_ > 0 {
                self.diag.push_diag(
                    Diagnostic {
                        severity: Severity::Error,
                        span,
                        message: format!(
                            "cannot immutably borrow `{target}` because it is already mutably borrowed"
                        ),
                        labels: vec![],
                        code: Some("E4005"),
                        hint: Some("wait for the mutable borrow to end, or reborrow from the same `&mut`".into()),
                    }
                );
            }
        }
    }

    fn bind_reference_var(&mut self, ref_var: &str, target: &str, is_mut: bool) {
        self.release_ref_binding(ref_var);
        self.ref_binding_to_target
            .insert(ref_var.to_string(), (target.to_string(), is_mut));
    }

    fn expr_as_simple_borrow(expr: &Expr) -> Option<(&str, bool, Span)> {
        if let Expr::UnOp { op, expr, span } = expr {
            if let Expr::Ident { name, .. } = &**expr {
                return match op {
                    UnOpKind::Ref => Some((name, false, *span)),
                    UnOpKind::RefMut => Some((name, true, *span)),
                    _ => None,
                };
            }
        }
        None
    }

    fn expr_as_simple_move(expr: &Expr) -> Option<(&str, Span)> {
        if let Expr::Ident { name, span } = expr {
            Some((name, *span))
        } else {
            None
        }
    }

    fn bind_pattern_vars(&mut self, pat: &Pattern) {
        match pat {
            Pattern::Ident { name, .. } => self.bind_var(name),
            Pattern::Tuple { elems, .. } => {
                for p in elems {
                    self.bind_pattern_vars(p);
                }
            }
            Pattern::Struct { fields, .. } => {
                for (_, p) in fields {
                    if let Some(p) = p {
                        self.bind_pattern_vars(p);
                    }
                }
            }
            Pattern::Enum { inner, .. } => {
                for p in inner {
                    self.bind_pattern_vars(p);
                }
            }
            Pattern::Or { arms, .. } => {
                // Conservative: bind all names from all arms.
                for p in arms {
                    self.bind_pattern_vars(p);
                }
            }
            _ => {}
        }
    }

    fn first_ident_in_pattern<'a>(pat: &'a Pattern) -> Option<&'a str> {
        match pat {
            Pattern::Ident { name, .. } => Some(name),
            Pattern::Tuple { elems, .. } => elems.iter().find_map(Self::first_ident_in_pattern),
            Pattern::Struct { fields, .. } => fields
                .iter()
                .find_map(|(_, p)| p.as_ref().and_then(Self::first_ident_in_pattern)),
            Pattern::Enum { inner, .. } => inner.iter().find_map(Self::first_ident_in_pattern),
            Pattern::Or { arms, .. } => arms.iter().find_map(Self::first_ident_in_pattern),
            _ => None,
        }
    }

    fn check_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Ident { name, span } => self.check_read(name, *span),
            Expr::UnOp { op, expr, span } => {
                match op {
                    UnOpKind::Ref | UnOpKind::RefMut => {
                        let is_mut = matches!(op, UnOpKind::RefMut);
                        if let Expr::Ident { name, .. } = &**expr {
                            // SAFETY: Only *validate* here — do not call try_borrow.
                            // try_borrow (which mutates loan state) is called by the
                            // enclosing statement handler (Stmt::Let / Expr::Assign)
                            // after it receives the binding name.  Calling try_borrow
                            // here too would double-count the loan, leaving it
                            // permanently non-zero even after the ref goes out of scope.
                            self.validate_borrow(name, is_mut, *span);
                        } else {
                            // Complex operand (&obj.field, etc.) — recurse normally.
                            self.check_expr(expr);
                        }
                    }
                    _ => self.check_expr(expr),
                }
            }
            Expr::Assign {
                op,
                target,
                value,
                span,
            } => {
                match &**target {
                    Expr::Ident { name, .. } => {
                        self.check_write(name, *span);
                        if matches!(op, AssignOpKind::Assign) {
                            self.mark_initialized(name);
                            self.set_copy_flag(name, self.expr_is_copy(value));
                        } else {
                            self.check_read(name, *span);
                        }
                        // Rebinding a reference variable drops the old borrow.
                        self.release_ref_binding(name);
                        if let Some((src, mv_span)) = Self::expr_as_simple_move(value) {
                            self.move_ident(src, mv_span);
                        }
                        if let Some((borrowed, is_mut, bspan)) = Self::expr_as_simple_borrow(value)
                        {
                            if self.try_borrow(borrowed, is_mut, bspan) {
                                self.bind_reference_var(name, borrowed, is_mut);
                            }
                        }
                    }
                    Expr::UnOp {
                        op: UnOpKind::Deref,
                        expr: inner,
                        ..
                    } => {
                        if let Expr::Ident {
                            name: ref_name,
                            span: ref_span,
                        } = &**inner
                        {
                            self.check_read(ref_name, *ref_span);
                            match self.ref_binding_to_target.get(ref_name) {
                                Some((_, true)) => {}
                                Some((target, false)) => self.diag.push_diag(
                                    Diagnostic {
                                        severity: Severity::Error,
                                        span: *span,
                                        message: format!(
                                            "cannot assign through immutable reference `{ref_name}` to `{target}`"
                                        ),
                                        labels: vec![],
                                        code: Some("E4010"),
                                        hint: Some("use `&mut name` to create a mutable reference".into()),
                                    }
                                ),
                                None => self.diag.push_diag(
                                    Diagnostic {
                                        severity: Severity::Error,
                                        span: *span,
                                        message: format!("cannot dereference non-reference `{ref_name}` for assignment"),
                                        labels: vec![],
                                        code: Some("E4011"),
                                        hint: Some("ensure the variable is a reference type before dereferencing for assignment".into()),
                                    }
                                ),
                            }
                        } else {
                            self.diag.error(
                                *span,
                                "unsupported assignment target through dereference",
                            );
                        }
                    }
                    _ => self.check_expr(target),
                }
                self.check_expr(value);
            }
            Expr::BinOp { lhs, rhs, .. }
            | Expr::MatMul { lhs, rhs, .. }
            | Expr::HadamardMul { lhs, rhs, .. }
            | Expr::HadamardDiv { lhs, rhs, .. }
            | Expr::TensorConcat { lhs, rhs, .. }
            | Expr::KronProd { lhs, rhs, .. }
            | Expr::OuterProd { lhs, rhs, .. }
            | Expr::Pow {
                base: lhs,
                exp: rhs,
                ..
            } => {
                self.check_expr(lhs);
                self.check_expr(rhs);
            }
            Expr::Field { object, .. } | Expr::Grad { inner: object, .. } => {
                self.check_expr(object)
            }
            Expr::Index {
                object, indices, ..
            } => {
                self.check_expr(object);
                for i in indices {
                    self.check_expr(i);
                }
            }
            Expr::Call {
                func, args, named, ..
            } => {
                self.check_expr(func);
                for a in args {
                    self.check_expr(a);
                    if let Some((name, sp)) = Self::expr_as_simple_move(a) {
                        self.move_ident(name, sp);
                    }
                }
                for (_, a) in named {
                    self.check_expr(a);
                }
            }
            Expr::MethodCall { receiver, args, .. } => {
                self.check_expr(receiver);
                for a in args {
                    self.check_expr(a);
                    if let Some((name, sp)) = Self::expr_as_simple_move(a) {
                        self.move_ident(name, sp);
                    }
                }
            }
            Expr::Range { lo, hi, .. } => {
                if let Some(lo) = lo {
                    self.check_expr(lo);
                }
                if let Some(hi) = hi {
                    self.check_expr(hi);
                }
            }
            Expr::Cast { expr, .. } => self.check_expr(expr),
            Expr::IfExpr {
                cond, then, else_, ..
            } => {
                self.check_expr(cond);
                self.check_block(then);
                if let Some(else_) = else_ {
                    self.check_block(else_);
                }
            }
            Expr::Closure { body, .. } => self.check_expr(body),
            Expr::Block(b) => self.check_block(b),
            Expr::Tuple { elems, .. } | Expr::ArrayLit { elems, .. } => {
                for e in elems {
                    self.check_expr(e);
                }
            }
            Expr::StructLit { fields, .. } => {
                for (_, e) in fields {
                    self.check_expr(e);
                }
            }
            Expr::VecCtor { elems, .. } => {
                for e in elems {
                    self.check_expr(e);
                }
            }
            Expr::IntLit { .. }
            | Expr::FloatLit { .. }
            | Expr::BoolLit { .. }
            | Expr::StrLit { .. }
            | Expr::Path { .. } => {}

            // ── v2: pipeline / emit / copy — recurse into sub-expressions ──
            Expr::Pipeline { stages, .. } => {
                for e in stages {
                    self.check_expr(e);
                }
            }
            Expr::Emit { value, .. } => self.check_expr(value),
            Expr::Copy { inner, .. } => self.check_expr(inner),
        }
    }

    fn expr_is_copy(&self, expr: &Expr) -> bool {
        match expr {
            Expr::IntLit { .. }
            | Expr::FloatLit { .. }
            | Expr::BoolLit { .. }
            | Expr::StrLit { .. } => true,
            Expr::Ident { name, .. } => {
                self.var_state.get(name).map(|v| v.is_copy).unwrap_or(false)
            }
            Expr::UnOp { op, expr, .. } => match op {
                UnOpKind::Ref | UnOpKind::RefMut => true,
                _ => self.expr_is_copy(expr),
            },
            Expr::BinOp { lhs, rhs, .. }
            | Expr::MatMul { lhs, rhs, .. }
            | Expr::HadamardMul { lhs, rhs, .. }
            | Expr::HadamardDiv { lhs, rhs, .. }
            | Expr::TensorConcat { lhs, rhs, .. }
            | Expr::KronProd { lhs, rhs, .. }
            | Expr::OuterProd { lhs, rhs, .. }
            | Expr::Pow {
                base: lhs,
                exp: rhs,
                ..
            } => self.expr_is_copy(lhs) && self.expr_is_copy(rhs),
            Expr::Tuple { elems, .. } | Expr::ArrayLit { elems, .. } => {
                elems.iter().all(|e| self.expr_is_copy(e))
            }
            _ => false,
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let { pattern, init, .. } => {
                if let Some(init) = init {
                    self.check_expr(init);
                }
                self.bind_pattern_vars(pattern);
                if let Some(binding_name) = Self::first_ident_in_pattern(pattern) {
                    if let Some(init) = init {
                        self.set_copy_flag(binding_name, self.expr_is_copy(init));
                        if let Some((src, span)) = Self::expr_as_simple_move(init) {
                            self.move_ident(src, span);
                        }
                        if let Some((borrowed, is_mut, bspan)) = Self::expr_as_simple_borrow(init) {
                            if self.try_borrow(borrowed, is_mut, bspan) {
                                self.bind_reference_var(binding_name, borrowed, is_mut);
                            }
                        }
                    }
                    self.mark_initialized(binding_name);
                }
            }
            Stmt::Expr { expr, .. } => self.check_expr(expr),
            Stmt::Return { value, .. } | Stmt::Break { value, .. } => {
                if let Some(v) = value {
                    self.check_expr(v);
                    if let Some((name, span)) = Self::expr_as_simple_move(v) {
                        self.move_ident(name, span);
                    }
                }
            }
            Stmt::Continue { .. } => {}
            Stmt::ForIn {
                pattern,
                iter,
                body,
                ..
            } => {
                self.check_expr(iter);
                // The for-in loop consumes the iterator by value.  Mark it moved
                // so any subsequent use is caught as use-after-move.
                if let Some((name, span)) = Self::expr_as_simple_move(iter) {
                    self.move_ident(name, span);
                }
                self.push_scope();
                self.bind_pattern_vars(pattern);
                self.check_block(body);
                self.pop_scope();
            }
            Stmt::EntityFor { var, body, .. } => {
                self.push_scope();
                self.bind_var(var);
                self.check_block(body);
                self.pop_scope();
            }
            Stmt::While { body, .. } | Stmt::Loop { body, .. } => {
                self.check_block(body)
            }
            Stmt::If {
                cond, then, else_, ..
            } => {
                self.check_expr(cond);
                self.check_block(then);
                if let Some(else_) = else_ {
                    match &**else_ {
                        crate::compiler::ast::IfOrBlock::If(s) => self.check_stmt(s),
                        crate::compiler::ast::IfOrBlock::Block(b) => self.check_block(b),
                    }
                }
            }
            Stmt::Match { expr, arms, .. } => {
                self.check_expr(expr);
                for arm in arms {
                    // Each arm gets its own scope so pattern bindings from one
                    // arm cannot leak into sibling arms or code after the match.
                    self.push_scope();
                    self.bind_pattern_vars(&arm.pat);
                    if let Some(g) = &arm.guard {
                        self.check_expr(g);
                    }
                    self.check_expr(&arm.body);
                    self.pop_scope();
                }
            }
            Stmt::Item(_) => {}
            Stmt::ParallelFor(pf) => {
                self.check_expr(&pf.iter);
                // Entering a parallel body: increment depth so that move_ident
                // and try_borrow(mut) will reject any unsafe access inside.
                self.parallel_depth += 1;
                self.check_block(&pf.body);
                self.parallel_depth -= 1;
            }
            Stmt::Spawn(b) => {
                // Spawned blocks run concurrently; treat them like parallel bodies.
                self.parallel_depth += 1;
                self.check_block(&b.body);
                self.parallel_depth -= 1;
            }
            Stmt::Sync(b) => self.check_block(&b.body),
            Stmt::Atomic(b) => self.check_block(&b.body),

            // ── v2: effect / region / task / unsafe / intrinsics / constraints ──
            Stmt::Effect { body, .. } => self.check_block(body),
            Stmt::Region { body, .. } => self.check_block(body),
            Stmt::TaskSpawn { task_expr, .. } => self.check_expr(task_expr),
            Stmt::TaskJoin { .. } => {}
            Stmt::UnsafeBlock { body, .. } => self.check_block(body),
            Stmt::IntrinsicsBlock { .. } => {}
            Stmt::Requires { condition, .. } => self.check_expr(condition),
            Stmt::Ensures { condition, .. } => self.check_expr(condition),
        }
    }

    fn check_block(&mut self, block: &Block) {
        self.push_scope();
        for s in &block.stmts {
            self.check_stmt(s);
        }
        if let Some(tail) = &block.tail {
            self.check_expr(tail);
            if let Some((name, span)) = Self::expr_as_simple_move(tail) {
                self.move_ident(name, span);
            }
        }
        self.pop_scope();
    }
}

pub fn jules_borrowck(program: &Program) -> Diagnostics {
    let mut ck = BorrowChecker::default();
    ck.push_scope();
    for item in &program.items {
        match item {
            Item::Fn(f) => {
                if let Some(body) = &f.body {
                    ck.push_scope();
                    for p in &f.params {
                        ck.bind_var(&p.name);
                    }
                    ck.check_block(body);
                    ck.pop_scope();
                }
            }
            Item::System(s) => {
                ck.push_scope();
                for p in &s.params {
                    ck.bind_var(&p.name);
                }
                ck.check_block(&s.body);
                ck.pop_scope();
            }
            Item::Mod {
                items: Some(items), ..
            } => {
                for it in items {
                    if let Item::Fn(f) = it {
                        if let Some(body) = &f.body {
                            ck.push_scope();
                            for p in &f.params {
                                ck.bind_var(&p.name);
                            }
                            ck.check_block(body);
                            ck.pop_scope();
                        }
                    }
                }
            }
            _ => {}
        }
    }
    ck.pop_scope();
    ck.diag
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::lexer::Lexer;
    use crate::compiler::parser::Parser;

    fn borrow_errors(source: &str) -> usize {
        let mut lx = Lexer::new(source);
        let (tokens, lex_errs) = lx.tokenize();
        assert!(lex_errs.is_empty(), "lexer errors: {lex_errs:?}");
        let mut p = Parser::new(tokens);
        let prog = p.parse_program();
        assert!(p.errors.is_empty(), "parser errors: {:?}", p.errors);
        jules_borrowck(&prog)
            .items
            .into_iter()
            .filter(|d| matches!(d.severity, Severity::Error))
            .count()
    }

    fn borrow_diags(source: &str) -> Diagnostics {
        let mut lx = Lexer::new(source);
        let (tokens, lex_errs) = lx.tokenize();
        assert!(lex_errs.is_empty(), "lexer errors: {lex_errs:?}");
        let mut p = Parser::new(tokens);
        let prog = p.parse_program();
        assert!(p.errors.is_empty(), "parser errors: {:?}", p.errors);
        jules_borrowck(&prog)
    }

    #[test]
    fn catches_write_while_borrowed() {
        let src = r#"
            fn main() {
                let mut x = 1;
                let r = &x;
                x = 2;
            }
        "#;
        assert!(borrow_errors(src) >= 1);
    }

    #[test]
    fn catches_conflicting_mut_borrow() {
        let src = r#"
            fn main() {
                let mut x = 1;
                let a = &mut x;
                let b = &x;
            }
        "#;
        assert!(borrow_errors(src) >= 1);
    }

    #[test]
    fn allows_copy_scalars_without_move_errors() {
        let src = r#"
            fn main() {
                let x = 1;
                let y = x;
                let z = x;
            }
        "#;
        assert_eq!(borrow_errors(src), 0);
    }

    #[test]
    fn catches_move_while_borrowed_for_non_copy_values() {
        let src = r#"
            fn main() {
                let x = make_board();
                let r = &x;
                let y = x;
            }
        "#;
        assert!(borrow_errors(src) >= 1);
    }

    #[test]
    fn catches_borrow_after_move_for_non_copy_values() {
        let src = r#"
            fn main() {
                let x = make_board();
                let y = x;
                let r = &x;
            }
        "#;
        assert!(borrow_errors(src) >= 1);
    }

    #[test]
    fn catches_assign_through_immutable_ref() {
        let src = r#"
            fn main() {
                let mut x = 1;
                let r = &x;
                *r = 3;
            }
        "#;
        assert!(borrow_errors(src) >= 1);
    }

    #[test]
    fn moved_value_error_contains_autofix_hint() {
        let src = r#"
            fn main() {
                let x = make_board();
                let y = x;
                let z = x;
            }
        "#;
        let diags = borrow_diags(src);
        assert!(diags.items.iter().any(|d| {
            d.message.contains("use of moved value")
                && d.hint.as_ref().map_or(false, |h| h.contains("clone"))
                && d.code == Some("E4001")
        }));
    }
}
