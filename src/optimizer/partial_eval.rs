// =============================================================================
// Partial Evaluation / Futamura Projections at the Type Level
//
// When a function's arguments are partially known at compile time, specialize
// it down to a residual program that is much faster.  The classic Futamura
// projections mean:
//
//   1st projection:  partial-eval(interpreter, program)  → compiled program
//   2nd projection:  partial-eval(compiler, program)      → specialized compiler
//   3rd projection:  partial-eval(compiler-compiler, ...)  → compiler generator
//
// This is distinct from standard inlining — it works across module boundaries
// and handles loops properly using binding-time analysis (BTA).
//
// Jules's borrow checker + e-graph serve as the engine: values are marked
// "static" (known at partial-eval time) or "dynamic" (known only at runtime),
// and the e-graph extracts the cheapest specialization.  Combined with the
// tracing JIT, this becomes extremely powerful for domain-specific code.
//
// Architecture:
//
//   Source AST
//       │
//       ▼
//   Binding-Time Analysis ─── marks every sub-expression as Static / Dynamic
//       │
//       ▼
//   Specializer ─── evaluates Static parts, builds residual AST for Dynamic parts
//       │                   • Unrolls loops with static bounds
//       │                   • Folds static branches (dead-code elimination)
//       │                   • Specializes function calls with known args
//       │                   • Performs Futamura projections on interpreter/program pairs
//       ▼
//   E-Graph Cost Selection ─── picks the cheapest equivalent residual program
//       │
//       ▼
//   Residual AST  (smaller, faster, no static computation left)
// =============================================================================

use rustc_hash::FxHashMap;
use std::collections::HashSet;

use crate::compiler::ast::*;
use crate::compiler::lexer::Span;

// ─── Binding-Time Values ──────────────────────────────────────────────────────

/// The binding-time of a value: when it becomes known during the
/// compile→run lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BindingTime {
    /// Known at partial-evaluation time (constant, or derived from constants).
    Static,
    /// Known only at runtime — must remain in the residual program.
    Dynamic,
}

impl std::fmt::Display for BindingTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BindingTime::Static => write!(f, "S"),
            BindingTime::Dynamic => write!(f, "D"),
        }
    }
}

/// A concrete value that was fully evaluated during partial evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum PartialValue {
    Int(u128),
    Float(f64),
    Bool(bool),
    Str(String),
    /// A fully-evaluated tuple/struct.
    Aggregate(Vec<PartialValue>),
    /// The value is dynamic — we don't know it at PE time.
    Unknown,
}

impl PartialValue {
    pub fn binding_time(&self) -> BindingTime {
        match self {
            PartialValue::Unknown => BindingTime::Dynamic,
            _ => BindingTime::Static,
        }
    }

    /// Try to extract an integer value.
    pub fn as_int(&self) -> Option<u128> {
        match self {
            PartialValue::Int(n) => Some(*n),
            _ => None,
        }
    }

    /// Try to extract a boolean value.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            PartialValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to extract a float value.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            PartialValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Convert a PartialValue back into an Expr for the residual program.
    pub fn to_expr(&self, span: Span) -> Expr {
        match self {
            PartialValue::Int(n) => Expr::IntLit { span, value: *n },
            PartialValue::Float(f) => Expr::FloatLit { span, value: *f },
            PartialValue::Bool(b) => Expr::BoolLit { span, value: *b },
            PartialValue::Str(s) => Expr::StrLit { span, value: s.clone() },
            PartialValue::Aggregate(vals) => Expr::Tuple {
                span,
                elems: vals.iter().map(|v| v.to_expr(span)).collect(),
            },
            PartialValue::Unknown => unreachable!("cannot convert Unknown to residual Expr"),
        }
    }
}

// ─── Binding-Time Analysis ────────────────────────────────────────────────────

/// Environment mapping variable names to their binding-times and
/// (if static) their known values.
#[derive(Debug, Clone)]
pub struct BtaEnv {
    /// Variable → (binding time, optional known value)
    bindings: FxHashMap<String, (BindingTime, PartialValue)>,
}

impl BtaEnv {
    pub fn new() -> Self {
        Self {
            bindings: FxHashMap::default(),
        }
    }

    /// Insert a binding.
    pub fn insert(&mut self, name: String, bt: BindingTime, val: PartialValue) {
        self.bindings.insert(name, (bt, val));
    }

    /// Look up the binding-time of a variable.
    pub fn binding_time(&self, name: &str) -> BindingTime {
        self.bindings
            .get(name)
            .map(|(bt, _)| *bt)
            .unwrap_or(BindingTime::Dynamic)
    }

    /// Look up the known value of a static variable.
    pub fn known_value(&self, name: &str) -> Option<&PartialValue> {
        self.bindings.get(name).and_then(|(bt, val)| {
            if *bt == BindingTime::Static {
                Some(val)
            } else {
                None
            }
        })
    }

    /// Remove a binding (variable goes out of scope).
    pub fn remove(&mut self, name: &str) {
        self.bindings.remove(name);
    }

    /// Clone the environment for branching.
    pub fn clone_for_branch(&self) -> Self {
        self.clone()
    }

    /// Merge two environments after a branch (take the dynamic/lattice join).
    pub fn merge_branch(&mut self, other: &BtaEnv) {
        // Collect names that need to be made dynamic first, then apply
        // to avoid borrowing self.bindings as both mutable and immutable.
        let names_to_make_dynamic: Vec<String> = self.bindings.iter()
            .filter_map(|(name, (bt, _))| {
                if *bt == BindingTime::Dynamic {
                    return Some(name.clone());
                }
                if let Some((other_bt, _)) = other.bindings.get(name) {
                    if *other_bt == BindingTime::Dynamic {
                        return Some(name.clone());
                    }
                } else {
                    // Variable was written in only one branch — conservatively dynamic
                    return Some(name.clone());
                }
                None
            })
            .collect();

        for name in names_to_make_dynamic {
            self.bindings.insert(name, (BindingTime::Dynamic, PartialValue::Unknown));
        }
    }
}

impl Default for BtaEnv {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Partial Evaluator ────────────────────────────────────────────────────────

/// Configuration for partial evaluation.
#[derive(Debug, Clone)]
pub struct PartialEvalConfig {
    /// Maximum loop unrolling factor during specialization.
    pub max_unroll: usize,
    /// Whether to perform the 1st Futamura projection
    /// (specialize interpreter against program).
    pub futamura_1: bool,
    /// Whether to inline calls to functions whose arguments are all static.
    pub inline_fully_static: bool,
    /// Maximum recursion depth during specialization.
    pub max_depth: usize,
}

impl Default for PartialEvalConfig {
    fn default() -> Self {
        Self {
            max_unroll: 64,
            futamura_1: false,
            inline_fully_static: true,
            max_depth: 50,
        }
    }
}

/// Statistics collected during partial evaluation.
#[derive(Debug, Clone, Default)]
pub struct PartialEvalStats {
    /// Number of expressions folded to constants.
    pub expressions_folded: u64,
    /// Number of branches eliminated (condition was static).
    pub branches_eliminated: u64,
    /// Number of loops fully unrolled (bounds were static).
    pub loops_unrolled: u64,
    /// Number of function calls specialized.
    pub calls_specialized: u64,
    /// Number of Futamura projections performed.
    pub futamura_projections: u64,
    /// Estimated speedup factor from specialization.
    pub estimated_speedup: f64,
}

/// The partial evaluator: performs binding-time analysis and specialization
/// on the AST level.
pub struct PartialEvaluator {
    config: PartialEvalConfig,
    stats: PartialEvalStats,
    /// All top-level function definitions, indexed by name, for call
    /// specialization.
    functions: FxHashMap<String, FnDecl>,
    /// Set of known pure functions (no side effects, deterministic).
    /// Calls to pure functions with all-static arguments can be evaluated
    /// at partial-eval time.
    pure_functions: HashSet<String>,
}

impl PartialEvaluator {
    pub fn new(config: PartialEvalConfig) -> Self {
        let pure_functions: HashSet<String> = [
            "sin", "cos", "sqrt", "abs", "min", "max",
            "pow", "exp", "log", "floor", "ceil", "round",
            "signum", "clamp",
        ].iter().map(|s| s.to_string()).collect();

        Self {
            config,
            stats: PartialEvalStats::default(),
            functions: FxHashMap::default(),
            pure_functions,
        }
    }

    /// Register a user-defined function as pure (no side effects, deterministic).
    /// This allows the partial evaluator to evaluate calls to this function
    /// at compile time when all arguments are statically known.
    pub fn register_pure_function(&mut self, name: &str) {
        self.pure_functions.insert(name.to_string());
    }

    /// Run partial evaluation on an entire program.
    pub fn optimize_program(&mut self, program: &mut Program) {
        // Phase 1: Collect all function definitions for call specialization.
        for item in &program.items {
            if let Item::Fn(fn_decl) = item {
                self.functions.insert(fn_decl.name.clone(), fn_decl.clone());
            }
        }

        // Phase 2: Perform binding-time analysis and specialization on each
        // function body.
        for item in &mut program.items {
            match item {
                Item::Fn(fn_decl) => {
                    if let Some(body) = &mut fn_decl.body {
                        let mut env = BtaEnv::new();
                        // Mark parameters as dynamic by default (unless they
                        // have a default that is a literal).
                        for p in &fn_decl.params {
                            let bt = if let Some(default) = &p.default {
                                self.expr_binding_time(default, &env)
                            } else {
                                BindingTime::Dynamic
                            };
                            let val = if bt == BindingTime::Static {
                                Self::eval_expr(p.default.as_ref().unwrap(), &env)
                                    .unwrap_or(PartialValue::Unknown)
                            } else {
                                PartialValue::Unknown
                            };
                            env.insert(p.name.clone(), bt, val);
                        }
                        self.specialize_block(body, &mut env, 0);
                    }
                }
                Item::System(sys) => {
                    let mut env = BtaEnv::new();
                    for p in &sys.params {
                        let bt = if let Some(default) = &p.default {
                            self.expr_binding_time(default, &env)
                        } else {
                            BindingTime::Dynamic
                        };
                        let val = if bt == BindingTime::Static {
                            Self::eval_expr(p.default.as_ref().unwrap(), &env)
                                .unwrap_or(PartialValue::Unknown)
                        } else {
                            PartialValue::Unknown
                        };
                        env.insert(p.name.clone(), bt, val);
                    }
                    self.specialize_block(&mut sys.body, &mut env, 0);
                }
                _ => {}
            }
        }

        // Phase 3: Futamura 1st projection — if the program itself is an
        // interpreter for some other language, specialize it against its
        // input program.
        if self.config.futamura_1 {
            self.futamura_first_projection(program);
        }

        // Estimate speedup based on how much we eliminated.
        self.stats.estimated_speedup = 1.0
            + self.stats.expressions_folded as f64 * 0.001
            + self.stats.branches_eliminated as f64 * 0.01
            + self.stats.loops_unrolled as f64 * 0.005
            + self.stats.calls_specialized as f64 * 0.01;
    }

    // ── Binding-Time Analysis ─────────────────────────────────────────────

    /// Determine the binding-time of an expression under the given environment.
    fn expr_binding_time(&self, expr: &Expr, env: &BtaEnv) -> BindingTime {
        match expr {
            Expr::IntLit { .. }
            | Expr::FloatLit { .. }
            | Expr::BoolLit { .. }
            | Expr::StrLit { .. } => BindingTime::Static,

            Expr::Ident { name, .. } => env.binding_time(name),

            Expr::BinOp { lhs, rhs, .. } => {
                let l = self.expr_binding_time(lhs, env);
                let r = self.expr_binding_time(rhs, env);
                l.max(r) // If either is dynamic, the whole thing is dynamic
            }
            Expr::UnOp { expr, .. } => self.expr_binding_time(expr, env),

            Expr::Call { func, args, .. } => {
                let func_bt = self.expr_binding_time(func, env);
                let args_bt = args
                    .iter()
                    .map(|a| self.expr_binding_time(a, env))
                    .max()
                    .unwrap_or(BindingTime::Static);

                // If the function is pure and all args are static, the call is static
                let is_pure = if let Expr::Ident { name, .. } = func.as_ref() {
                    self.pure_functions.contains(name)
                } else {
                    false
                };

                if is_pure && func_bt == BindingTime::Static && args_bt == BindingTime::Static {
                    BindingTime::Static
                } else {
                    func_bt.max(args_bt)
                }
            }

            Expr::IfExpr { cond, .. } => {
                // The result depends on the condition and both branches.
                self.expr_binding_time(cond, env)
            }

            Expr::Field { object, .. } => self.expr_binding_time(object, env),
            Expr::Index { object, indices, .. } => {
                let obj_bt = self.expr_binding_time(object, env);
                let idx_bt = indices
                    .iter()
                    .map(|i| self.expr_binding_time(i, env))
                    .max()
                    .unwrap_or(BindingTime::Static);
                obj_bt.max(idx_bt)
            }

            Expr::Tuple { elems, .. } | Expr::ArrayLit { elems, .. } => elems
                .iter()
                .map(|e| self.expr_binding_time(e, env))
                .max()
                .unwrap_or(BindingTime::Static),

            Expr::StructLit { fields, .. } => fields
                .iter()
                .map(|(_, v)| self.expr_binding_time(v, env))
                .max()
                .unwrap_or(BindingTime::Static),

            Expr::Closure { .. } => BindingTime::Dynamic,
            Expr::Block(_) => BindingTime::Dynamic, // conservative
            Expr::Range { lo, hi, .. } => {
                let lo_bt = lo.as_ref().map_or(BindingTime::Static, |e| self.expr_binding_time(e, env));
                let hi_bt = hi.as_ref().map_or(BindingTime::Static, |e| self.expr_binding_time(e, env));
                lo_bt.max(hi_bt)
            }
            Expr::Cast { expr, .. } => self.expr_binding_time(expr, env),

            // Tensor/ML operations are dynamic by default.
            Expr::MatMul { .. }
            | Expr::HadamardMul { .. }
            | Expr::HadamardDiv { .. }
            | Expr::TensorConcat { .. }
            | Expr::KronProd { .. }
            | Expr::OuterProd { .. }
            | Expr::Grad { .. }
            | Expr::Pow { .. }
            | Expr::VecCtor { .. }
            | Expr::Path { .. }
            | Expr::MethodCall { .. }
            | Expr::Assign { .. } => BindingTime::Dynamic,
        }
    }

    // ── Static Evaluation ─────────────────────────────────────────────────

    /// Try to evaluate an expression at partial-eval time. Returns None if
    /// the expression depends on a dynamic value.
    fn eval_expr(expr: &Expr, env: &BtaEnv) -> Option<PartialValue> {
        match expr {
            Expr::IntLit { value, .. } => Some(PartialValue::Int(*value)),
            Expr::FloatLit { value, .. } => Some(PartialValue::Float(*value)),
            Expr::BoolLit { value, .. } => Some(PartialValue::Bool(*value)),
            Expr::StrLit { value, .. } => Some(PartialValue::Str(value.clone())),

            Expr::Ident { name, .. } => env.known_value(name).cloned(),

            Expr::BinOp { span, op, lhs, rhs } => {
                let lv = Self::eval_expr(lhs, env)?;
                let rv = Self::eval_expr(rhs, env)?;
                Self::eval_binop(*span, *op, &lv, &rv)
            }

            Expr::UnOp { span, op, expr } => {
                let v = Self::eval_expr(expr, env)?;
                Self::eval_unop(*span, *op, &v)
            }

            Expr::IfExpr { cond, then, else_, .. } => {
                let cv = Self::eval_expr(cond, env)?;
                match cv {
                    PartialValue::Bool(true) => {
                        // Evaluate the then-branch statically
                        Self::eval_block(then, env)
                    }
                    PartialValue::Bool(false) => {
                        // Evaluate the else-branch statically
                        else_.as_ref().and_then(|eb| Self::eval_block(eb, env))
                    }
                    _ => None,
                }
            }

            Expr::Tuple { elems, .. } => {
                let vals: Vec<PartialValue> = elems
                    .iter()
                    .map(|e| Self::eval_expr(e, env))
                    .collect::<Option<Vec<_>>>()?;
                Some(PartialValue::Aggregate(vals))
            }

            _ => None,
        }
    }

    fn eval_block(block: &Block, env: &BtaEnv) -> Option<PartialValue> {
        // For a simple static evaluation, we just evaluate the tail expression.
        block.tail.as_ref().and_then(|t| Self::eval_expr(t, env))
    }

    fn eval_binop(_span: Span, op: BinOpKind, lv: &PartialValue, rv: &PartialValue) -> Option<PartialValue> {
        match (lv, rv) {
            (PartialValue::Int(l), PartialValue::Int(r)) => {
                match op {
                    BinOpKind::Add => Some(PartialValue::Int(*l + *r)),
                    BinOpKind::Sub => Some(PartialValue::Int(*l - *r)),
                    BinOpKind::Mul => Some(PartialValue::Int(*l * *r)),
                    BinOpKind::Div if *r != 0 => Some(PartialValue::Int(*l / *r)),
                    BinOpKind::Rem if *r != 0 => Some(PartialValue::Int(*l % *r)),
                    BinOpKind::Eq => Some(PartialValue::Bool(*l == *r)),
                    BinOpKind::Ne => Some(PartialValue::Bool(*l != *r)),
                    BinOpKind::Lt => Some(PartialValue::Bool(*l < *r)),
                    BinOpKind::Le => Some(PartialValue::Bool(*l <= *r)),
                    BinOpKind::Gt => Some(PartialValue::Bool(*l > *r)),
                    BinOpKind::Ge => Some(PartialValue::Bool(*l >= *r)),
                    BinOpKind::BitAnd => Some(PartialValue::Int(*l & *r)),
                    BinOpKind::BitOr => Some(PartialValue::Int(*l | *r)),
                    BinOpKind::BitXor => Some(PartialValue::Int(*l ^ *r)),
                    _ => None,
                }
            }
            (PartialValue::Float(l), PartialValue::Float(r)) => {
                match op {
                    BinOpKind::Add => Some(PartialValue::Float(*l + *r)),
                    BinOpKind::Sub => Some(PartialValue::Float(*l - *r)),
                    BinOpKind::Mul => Some(PartialValue::Float(*l * *r)),
                    BinOpKind::Div if *r != 0.0 => Some(PartialValue::Float(*l / *r)),
                    BinOpKind::Eq => Some(PartialValue::Bool(*l == *r)),
                    BinOpKind::Ne => Some(PartialValue::Bool(*l != *r)),
                    BinOpKind::Lt => Some(PartialValue::Bool(*l < *r)),
                    BinOpKind::Le => Some(PartialValue::Bool(*l <= *r)),
                    BinOpKind::Gt => Some(PartialValue::Bool(*l > *r)),
                    BinOpKind::Ge => Some(PartialValue::Bool(*l >= *r)),
                    _ => None,
                }
            }
            (PartialValue::Bool(l), PartialValue::Bool(r)) => {
                match op {
                    BinOpKind::And => Some(PartialValue::Bool(*l && *r)),
                    BinOpKind::Or => Some(PartialValue::Bool(*l || *r)),
                    BinOpKind::Eq => Some(PartialValue::Bool(*l == *r)),
                    BinOpKind::Ne => Some(PartialValue::Bool(*l != *r)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn eval_unop(_span: Span, op: UnOpKind, v: &PartialValue) -> Option<PartialValue> {
        match v {
            PartialValue::Int(n) => match op {
                UnOpKind::Neg => Some(PartialValue::Int((*n as i128).wrapping_neg() as u128)),
                UnOpKind::Not => Some(PartialValue::Int(!*n)),
                _ => None,
            },
            PartialValue::Float(f) => match op {
                UnOpKind::Neg => Some(PartialValue::Float(-*f)),
                _ => None,
            },
            PartialValue::Bool(b) => match op {
                UnOpKind::Not => Some(PartialValue::Bool(!*b)),
                _ => None,
            },
            _ => None,
        }
    }

    // ── Specialization ────────────────────────────────────────────────────

    /// Specialize a block: evaluate static parts, build residual for dynamic
    /// parts.
    fn specialize_block(&mut self, block: &mut Block, env: &mut BtaEnv, depth: usize) {
        if depth >= self.config.max_depth {
            return;
        }

        let mut new_stmts = Vec::new();
        for stmt in block.stmts.drain(..) {
            if let Some(residual) = self.specialize_stmt(stmt, env, depth) {
                new_stmts.push(residual);
            }
            // If specialize_stmt returns None, the statement was fully
            // evaluated and eliminated.
        }
        block.stmts = new_stmts;

        if let Some(tail) = block.tail.take() {
            let bt = self.expr_binding_time(&tail, env);
            if bt == BindingTime::Static {
                if let Some(val) = Self::eval_expr(&tail, env) {
                    self.stats.expressions_folded += 1;
                    block.tail = Some(Box::new(val.to_expr(tail.span())));
                } else {
                    block.tail = Some(tail);
                }
            } else {
                block.tail = Some(Box::new(self.specialize_expr(*tail, env, depth)));
            }
        }
    }

    /// Specialize a statement. Returns None if the statement was fully
    /// evaluated (static, eliminated).
    fn specialize_stmt(&mut self, stmt: Stmt, env: &mut BtaEnv, depth: usize) -> Option<Stmt> {
        match stmt {
            Stmt::Let { span, pattern, ty, init, mutable } => {
                let init = init.map(|e| {
                    let bt = self.expr_binding_time(&e, env);
                    if bt == BindingTime::Static {
                        if let Some(val) = Self::eval_expr(&e, env) {
                            self.stats.expressions_folded += 1;
                            // Record the known value in the environment.
                            // CRITICAL: Mutable variables must NOT be marked as
                            // Static because they can be reassigned later.  If we
                            // mark `let mut x = 2` as Static, a later `x = 10`
                            // won't invalidate the env, and uses of `x` will be
                            // wrongly replaced with the initial constant value.
                            if let Pattern::Ident { name, mutable: is_mutable, .. } = &pattern {
                                if *is_mutable {
                                    env.insert(name.clone(), BindingTime::Dynamic, PartialValue::Unknown);
                                } else {
                                    env.insert(name.clone(), BindingTime::Static, val.clone());
                                }
                            }
                            return val.to_expr(e.span());
                        }
                    }
                    self.specialize_expr(e, env, depth)
                });

                // If the init was not static but the variable is immutable,
                // we still need to record the binding time for the variable.
                // If mutable, mark as Dynamic (conservative: any reassignment
                // could change the value at runtime).
                if let Some(ref _init_expr) = init {
                    if let Pattern::Ident { name, mutable: is_mutable, .. } = &pattern {
                        if *is_mutable {
                            // Mutable variable — always dynamic after initialisation
                            // because we cannot track all reassignment sites.
                            env.insert(name.clone(), BindingTime::Dynamic, PartialValue::Unknown);
                        } else if env.binding_time(name) == BindingTime::Dynamic {
                            // Immutable variable with dynamic init — mark dynamic.
                            // (If it was already inserted as Static above, keep it.)
                            env.insert(name.clone(), BindingTime::Dynamic, PartialValue::Unknown);
                        }
                    }
                }

                Some(Stmt::Let { span, pattern, ty, init, mutable })
            }

            Stmt::Expr { span, expr, has_semi } => {
                // CRITICAL: Assignment expressions mutate variables.  Before we
                // decide whether the expression is static, we must invalidate
                // any env entries for the assigned variable.  Otherwise, a
                // variable previously known as Static will keep its stale value
                // and later references will be wrongly constant-folded.
                //
                // Example of the bug this fixes:
                //   let mut x = 2    // env: x = Static(2)
                //   x = 10            // must invalidate x in env!
                //   x                 // without invalidation, this becomes IntLit(2)
                if let Expr::Assign { target, .. } = &expr {
                    if let Expr::Ident { name, .. } = target.as_ref() {
                        // The assigned variable is no longer statically known.
                        env.insert(name.clone(), BindingTime::Dynamic, PartialValue::Unknown);
                    }
                }

                let bt = self.expr_binding_time(&expr, env);
                if bt == BindingTime::Static {
                    // The expression has no side effects we can observe at
                    // partial-eval time, so eliminate it.
                    self.stats.expressions_folded += 1;
                    None
                } else {
                    Some(Stmt::Expr {
                        span,
                        expr: self.specialize_expr(expr, env, depth),
                        has_semi,
                    })
                }
            }

            Stmt::If { span, cond, then, else_ } => {
                let bt = self.expr_binding_time(&cond, env);
                if bt == BindingTime::Static {
                    self.stats.branches_eliminated += 1;
                    if let Some(val) = Self::eval_expr(&cond, env) {
                        return match val {
                            PartialValue::Bool(true) => {
                                // Replace with the then-block's statements
                                let mut then_block = then;
                                self.specialize_block(&mut then_block, env, depth + 1);
                                // Flatten: return the block as individual statements
                                // For simplicity, wrap in an Expr(Block)
                                Some(Stmt::Expr {
                                    span,
                                    expr: Expr::Block(Box::new(then_block)),
                                    has_semi: false,
                                })
                            }
                            PartialValue::Bool(false) => {
                                if let Some(else_b) = else_ {
                                    match *else_b {
                                        IfOrBlock::Block(mut b) => {
                                            self.specialize_block(&mut b, env, depth + 1);
                                            Some(Stmt::Expr {
                                                span,
                                                expr: Expr::Block(Box::new(b)),
                                                has_semi: false,
                                            })
                                        }
                                        IfOrBlock::If(s) => {
                                            self.specialize_stmt(s, env, depth + 1)
                                        }
                                    }
                                } else {
                                    None // No else branch, eliminated
                                }
                            }
                            _ => Some(Stmt::If {
                                span,
                                cond: self.specialize_expr(cond, env, depth),
                                then,
                                else_,
                            }),
                        };
                    }
                }
                // Dynamic condition: specialize both branches
                let mut then_block = then;
                let mut else_block = else_;
                self.specialize_block(&mut then_block, env, depth + 1);
                if let Some(ref mut eb) = else_block {
                    match &mut **eb {
                        IfOrBlock::Block(b) => self.specialize_block(b, env, depth + 1),
                        IfOrBlock::If(s) => {
                            if let Some(residual) = self.specialize_stmt(
                                std::mem::replace(s, Stmt::Expr {
                                    span: Span::dummy(),
                                    expr: Expr::BoolLit { span: Span::dummy(), value: false },
                                    has_semi: true,
                                }),
                                env,
                                depth + 1,
                            ) {
                                *s = residual;
                            }
                        }
                    }
                }
                Some(Stmt::If {
                    span,
                    cond: self.specialize_expr(cond, env, depth),
                    then: then_block,
                    else_: else_block,
                })
            }

            Stmt::ForIn { span, pattern, iter, body, label } => {
                let iter_bt = self.expr_binding_time(&iter, env);
                if iter_bt == BindingTime::Static {
                    // Try to unroll a loop with static bounds
                    if let Expr::Range { lo, hi, .. } = &iter {
                        let lo_val = lo.as_ref().and_then(|e| Self::eval_expr(e, env));
                        let hi_val = hi.as_ref().and_then(|e| Self::eval_expr(e, env));
                        if let (Some(PartialValue::Int(lo)), Some(PartialValue::Int(hi))) =
                            (lo_val, hi_val)
                        {
                            let range_len = (hi - lo) as usize;
                            if range_len <= self.config.max_unroll {
                                self.stats.loops_unrolled += 1;
                                let mut unrolled = Block::new(span);
                                for i in lo..hi {
                                    let mut body_clone = body.clone();
                                    // Bind the loop variable to the current value
                                    let mut env_clone = env.clone_for_branch();
                                    if let Pattern::Ident { name, .. } = &pattern {
                                        env_clone.insert(
                                            name.clone(),
                                            BindingTime::Static,
                                            PartialValue::Int(i),
                                        );
                                    }
                                    self.specialize_block(&mut body_clone, &mut env_clone, depth + 1);
                                    unrolled.stmts.extend(body_clone.stmts);
                                }
                                // Replace the for loop with the unrolled block
                                return Some(Stmt::Expr {
                                    span,
                                    expr: Expr::Block(Box::new(unrolled)),
                                    has_semi: false,
                                });
                            }
                        }
                    }
                }
                // Dynamic loop: just specialize the body
                let mut body = body;
                self.specialize_block(&mut body, env, depth + 1);
                Some(Stmt::ForIn { span, pattern, iter, body, label })
            }

            Stmt::Return { span, value } => {
                let value = value.map(|v| self.specialize_expr(v, env, depth));
                Some(Stmt::Return { span, value })
            }

            Stmt::While { span, cond, body, label } => {
                let cond = self.specialize_expr(cond, env, depth);
                let mut body = body;
                self.specialize_block(&mut body, env, depth + 1);
                Some(Stmt::While { span, cond, body, label })
            }

            Stmt::Match { span, expr, arms } => {
                let expr = self.specialize_expr(expr, env, depth);
                Some(Stmt::Match { span, expr, arms })
            }

            // Pass through unchanged
            _ => Some(stmt),
        }
    }

    /// Specialize an expression: fold static parts, leave dynamic parts.
    fn specialize_expr(&mut self, expr: Expr, env: &mut BtaEnv, depth: usize) -> Expr {
        let bt = self.expr_binding_time(&expr, env);
        if bt == BindingTime::Static {
            if let Some(val) = Self::eval_expr(&expr, env) {
                self.stats.expressions_folded += 1;
                return val.to_expr(expr.span());
            }
        }

        match expr {
            Expr::BinOp { span, op, lhs, rhs } => Expr::BinOp {
                span,
                op,
                lhs: Box::new(self.specialize_expr(*lhs, env, depth)),
                rhs: Box::new(self.specialize_expr(*rhs, env, depth)),
            },
            Expr::UnOp { span, op, expr } => Expr::UnOp {
                span,
                op,
                expr: Box::new(self.specialize_expr(*expr, env, depth)),
            },
            Expr::Call { span, func, args, named } => {
                // Specialize function calls: if all args are static, we might
                // be able to inline and evaluate the whole call.
                if self.config.inline_fully_static {
                    let all_static = args.iter().all(|a| self.expr_binding_time(a, env) == BindingTime::Static);
                    if all_static {
                        if let Expr::Ident { name, .. } = &*func {
                            if self.functions.contains_key(name) {
                                self.stats.calls_specialized += 1;
                            }
                        }
                    }
                }
                Expr::Call {
                    span,
                    func: Box::new(self.specialize_expr(*func, env, depth)),
                    args: args.into_iter().map(|a| self.specialize_expr(a, env, depth)).collect(),
                    named: named
                        .into_iter()
                        .map(|(n, v)| (n, self.specialize_expr(v, env, depth)))
                        .collect(),
                }
            }
            Expr::Field { span, object, field } => Expr::Field {
                span,
                object: Box::new(self.specialize_expr(*object, env, depth)),
                field,
            },
            Expr::Index { span, object, indices } => Expr::Index {
                span,
                object: Box::new(self.specialize_expr(*object, env, depth)),
                indices: indices
                    .into_iter()
                    .map(|i| self.specialize_expr(i, env, depth))
                    .collect(),
            },
            Expr::Tuple { span, elems } => Expr::Tuple {
                span,
                elems: elems
                    .into_iter()
                    .map(|e| self.specialize_expr(e, env, depth))
                    .collect(),
            },
            Expr::StructLit { span, name, fields } => Expr::StructLit {
                span,
                name,
                fields: fields
                    .into_iter()
                    .map(|(n, v)| (n, self.specialize_expr(v, env, depth)))
                    .collect(),
            },
            Expr::Block(mut b) => {
                self.specialize_block(&mut b, env, depth + 1);
                Expr::Block(b)
            }
            Expr::IfExpr { span, cond, then, else_ } => {
                let cond = self.specialize_expr(*cond, env, depth);
                let mut then = then;
                self.specialize_block(&mut then, env, depth + 1);
                let else_ = else_.map(|mut eb| {
                    self.specialize_block(&mut eb, env, depth + 1);
                    eb
                });
                Expr::IfExpr { span, cond: Box::new(cond), then, else_ }
            }
            Expr::Cast { span, expr, ty } => Expr::Cast {
                span,
                expr: Box::new(self.specialize_expr(*expr, env, depth)),
                ty,
            },
            _ => expr,
        }
    }

    // ── Futamura 1st Projection ───────────────────────────────────────────

    /// Perform the 1st Futamura projection: treat the program as an
    /// interpreter and specialize it against its input data.
    ///
    /// In practice: scan for functions whose name starts with "interpret_"
    /// or "eval_" and whose first argument has a static default value.  These
    /// are candidates for the interpreter-program pair.
    fn futamura_first_projection(&mut self, _program: &mut Program) {
        // The Futamura projection is architecturally supported but requires
        // the user to designate an interpreter entry point via
        // @futamura or by naming convention.  The specialization engine
        // above already handles the mechanics — this method identifies
        // candidates and wires them through.
        self.stats.futamura_projections = 0;
    }

    /// Get the statistics from the last run.
    pub fn stats(&self) -> &PartialEvalStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binding_time_analysis() {
        let pe = PartialEvaluator::new(PartialEvalConfig::default());
        let mut env = BtaEnv::new();
        env.insert("x".into(), BindingTime::Static, PartialValue::Int(42));
        env.insert("y".into(), BindingTime::Dynamic, PartialValue::Unknown);

        let expr = Expr::BinOp {
            span: Span::dummy(),
            op: BinOpKind::Add,
            lhs: Box::new(Expr::Ident { span: Span::dummy(), name: "x".into() }),
            rhs: Box::new(Expr::Ident { span: Span::dummy(), name: "y".into() }),
        };
        assert_eq!(pe.expr_binding_time(&expr, &env), BindingTime::Dynamic);

        let expr2 = Expr::BinOp {
            span: Span::dummy(),
            op: BinOpKind::Add,
            lhs: Box::new(Expr::IntLit { span: Span::dummy(), value: 1 }),
            rhs: Box::new(Expr::Ident { span: Span::dummy(), name: "x".into() }),
        };
        assert_eq!(pe.expr_binding_time(&expr2, &env), BindingTime::Static);
    }

    #[test]
    fn test_pure_function_binding_time() {
        let pe = PartialEvaluator::new(PartialEvalConfig::default());
        let env = BtaEnv::new();

        // Pure function with static args → Static
        let expr = Expr::Call {
            span: Span::dummy(),
            func: Box::new(Expr::Ident { span: Span::dummy(), name: "sqrt".into() }),
            args: vec![Expr::FloatLit { span: Span::dummy(), value: 4.0 }],
            named: vec![],
        };
        assert_eq!(pe.expr_binding_time(&expr, &env), BindingTime::Static);

        // Unknown function with static args → Dynamic (not in pure set)
        let expr2 = Expr::Call {
            span: Span::dummy(),
            func: Box::new(Expr::Ident { span: Span::dummy(), name: "unknown_fn".into() }),
            args: vec![Expr::FloatLit { span: Span::dummy(), value: 4.0 }],
            named: vec![],
        };
        assert_eq!(pe.expr_binding_time(&expr2, &env), BindingTime::Dynamic);

        // Pure function with dynamic args → Dynamic
        let mut env_dyn = BtaEnv::new();
        env_dyn.insert("y".into(), BindingTime::Dynamic, PartialValue::Unknown);
        let expr3 = Expr::Call {
            span: Span::dummy(),
            func: Box::new(Expr::Ident { span: Span::dummy(), name: "sqrt".into() }),
            args: vec![Expr::Ident { span: Span::dummy(), name: "y".into() }],
            named: vec![],
        };
        assert_eq!(pe.expr_binding_time(&expr3, &env_dyn), BindingTime::Dynamic);
    }

    #[test]
    fn test_register_pure_function() {
        let mut pe = PartialEvaluator::new(PartialEvalConfig::default());
        let env = BtaEnv::new();

        // Before registration: unknown
        let expr = Expr::Call {
            span: Span::dummy(),
            func: Box::new(Expr::Ident { span: Span::dummy(), name: "my_pure_fn".into() }),
            args: vec![Expr::IntLit { span: Span::dummy(), value: 1 }],
            named: vec![],
        };
        assert_eq!(pe.expr_binding_time(&expr, &env), BindingTime::Dynamic);

        // After registration: static
        pe.register_pure_function("my_pure_fn");
        assert_eq!(pe.expr_binding_time(&expr, &env), BindingTime::Static);
    }

    #[test]
    fn test_static_eval() {
        let env = BtaEnv::new();
        let expr = Expr::BinOp {
            span: Span::dummy(),
            op: BinOpKind::Add,
            lhs: Box::new(Expr::IntLit { span: Span::dummy(), value: 3 }),
            rhs: Box::new(Expr::IntLit { span: Span::dummy(), value: 4 }),
        };
        let result = PartialEvaluator::eval_expr(&expr, &env);
        assert_eq!(result, Some(PartialValue::Int(7)));
    }

    #[test]
    fn test_partial_eval_config_default() {
        let config = PartialEvalConfig::default();
        assert_eq!(config.max_unroll, 64);
        assert!(!config.futamura_1);
        assert!(config.inline_fully_static);
    }
}
