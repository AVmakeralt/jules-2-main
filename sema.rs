// =============================================================================
// jules/src/sema.rs
//
// Semantic Analysis for the Jules programming language.
//
// This pass runs AFTER parsing and type-checking (`typeck.rs`) and enforces
// language-level rules that go beyond type correctness.
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │  What this pass does                                                    │
// ├─────────────────────────────────────────────────────────────────────────┤
// │  §1  Variable resolution                                                │
// │        • Detect reads of undefined / uninitialised variables.           │
// │        • Detect unused `let` bindings (warn).                           │
// │        • Detect assignments to immutable bindings.                      │
// │        • Detect shadowing of outer names (warn, configurable).          │
// │                                                                         │
// │  §2  Unreachable / dead-code detection                                  │
// │        • Flag statements after `return`, `break`, or `continue`.        │
// │        • Detect `loop {}` with no reachable `break` (infinite loop).    │
// │        • Warn on `if false { … }` / `if true { … } else { … }`.        │
// │        • Detect unused function / system / component declarations.       │
// │                                                                         │
// │  §3  ECS / system loop rules (Feature 2)                               │
// │        • No entity loop may write a component that another              │
// │          *concurrent* entity loop reads (aliasing).                     │
// │        • Iteration determinism: flag when a loop body performs          │
// │          order-dependent operations on components (e.g., writing a      │
// │          global accumulator inside a `@parallel` entity loop).          │
// │        • `without(Dead)` must be explicit if the body reads `health`.   │
// │        • Detect empty `with(…)` queries in `@simd`/`@parallel` systems. │
// │        • Confirm every component named in `with`/`without` is declared. │
// │        • Detect duplicate component names in a single query.             │
// │        • Ensure entity loop variables are not used outside their scope. │
// │                                                                         │
// │  §4  Agent / behaviour correctness (Feature 3)                         │
// │        • Every agent must declare at least one behaviour.               │
// │        • Behaviour priorities must be unique within an agent.           │
// │        • `self` inside a behaviour refers to the agent's own fields.    │
// │        • Utility expressions must not have side-effects.                │
// │        • `learning reinforcement` without a `train` block → warn.       │
// │        • Duplicate perception kinds → error.                            │
// │        • Memory capacity must be > 0.                                   │
// │                                                                         │
// │  §5  Model / train correctness (Unique Features 1 & 2)                 │
// │        • Model must have exactly one `input` and one `output` layer.    │
// │        • `train` references must resolve to declared agents/models.     │
// │        • An agent used in `train` must have `learning` ≠ None.         │
// │        • Hyper-parameter keys must be unique within a `train` block.    │
// │        • Reward and penalty signal names must be unique.                │
// │        • `episode.num_envs` must be ≥ 1 if present.                    │
// │        • `episode.max_steps` must be ≥ 1 if present.                   │
// │                                                                         │
// │  §6  Control-flow integrity                                             │
// │        • `break` / `continue` must be inside a loop.                   │
// │        • Labelled `break` / `continue` must target an enclosing         │
// │          loop that carries the named label.                             │
// │        • `return` outside a function → error.                          │
// │        • `await` outside an `async` function → error.                  │
// │        • `spawn` / `sync` / `atomic` nesting rules.                    │
// └─────────────────────────────────────────────────────────────────────────┘
//
// Architecture
// ────────────
//   SemaCtx        — carries the accumulator, component registry, and
//                    declaration-use maps built during the pass.
//   ScopeStack     — tracks variable bindings together with use-counts so
//                    "unused variable" warnings can be emitted on scope exit.
//   ControlFlow    — a lightweight stack that tracks the enclosing loop /
//                    function / async context for break/continue/return/await.
//   DeclRegistry   — records which top-level names were declared and whether
//                    they were referenced anywhere, enabling unused-decl warns.
//   ComponentGraph — per-system read/write sets used for the aliasing check.
// =============================================================================

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ast::{
    AccessMode, AgentDecl, Attribute, BehaviorRule, Block, ComponentDecl, EntityQuery, Expr,
    FnDecl, GoalDecl, Item, MatchArm, ModelDecl, ModelLayer, ParallelFor, Pattern, PerceptionKind,
    Program, Stmt, SystemDecl, TrainDecl,
};
use crate::lexer::Span;

// =============================================================================
// §0  DIAGNOSTICS  (re-used from typeck; re-declared here for standalone use)
// =============================================================================

/// Severity of a semantic diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

/// A single span-aware diagnostic produced by the semantic pass.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub span: Span,
    pub message: String,
    /// Secondary labels pointing at related source locations.
    pub labels: Vec<(Span, String)>,
}

impl Diagnostic {
    pub fn error(span: Span, msg: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Error,
            span,
            message: msg.into(),
            labels: vec![],
        }
    }
    pub fn warning(span: Span, msg: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Warning,
            span,
            message: msg.into(),
            labels: vec![],
        }
    }
    pub fn note(span: Span, msg: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Note,
            span,
            message: msg.into(),
            labels: vec![],
        }
    }
    /// Attach a secondary label to a different source location.
    pub fn with_label(mut self, span: Span, msg: impl Into<String>) -> Self {
        self.labels.push((span, msg.into()));
        self
    }
    pub fn is_fatal(&self) -> bool {
        self.severity == Severity::Error
    }
}

/// Collects all diagnostics emitted during the semantic pass.
#[derive(Debug, Default)]
pub struct Diagnostics {
    pub items: Vec<Diagnostic>,
}

impl Diagnostics {
    pub fn push(&mut self, d: Diagnostic) {
        self.items.push(d);
    }
    pub fn error(&mut self, span: Span, msg: impl Into<String>) {
        self.push(Diagnostic::error(span, msg));
    }
    pub fn warning(&mut self, span: Span, msg: impl Into<String>) {
        self.push(Diagnostic::warning(span, msg));
    }
    pub fn note(&mut self, span: Span, msg: impl Into<String>) {
        self.push(Diagnostic::note(span, msg));
    }
    pub fn has_errors(&self) -> bool {
        self.items.iter().any(|d| d.is_fatal())
    }
    pub fn error_count(&self) -> usize {
        self.items.iter().filter(|d| d.is_fatal()).count()
    }
    pub fn warning_count(&self) -> usize {
        self.items
            .iter()
            .filter(|d| d.severity == Severity::Warning)
            .count()
    }
}

// =============================================================================
// §0b  VARIABLE BINDING RECORDS
// =============================================================================

/// Tracks one declared variable in the current scope.
#[derive(Debug, Clone)]
struct Binding {
    span: Span,
    mutable: bool,
    /// Number of times this binding has been *read* since its declaration.
    use_count: u32,
    /// True if this binding was declared with `_` (intentionally unused).
    is_wildcard: bool,
}

impl Binding {
    fn new(span: Span, mutable: bool, is_wildcard: bool) -> Self {
        Binding {
            span,
            mutable,
            use_count: 0,
            is_wildcard,
        }
    }
}

// =============================================================================
// §0c  SCOPE STACK
// =============================================================================

/// A stack of lexical scopes.  Each scope maps a variable name to its binding.
/// On `pop_scope` the checker emits "unused variable" warnings for bindings
/// with `use_count == 0` (unless the name starts with `_`).
#[derive(Debug, Default)]
struct ScopeStack {
    scopes: Vec<FxHashMap<String, Binding>>,
}

impl ScopeStack {
    fn push(&mut self) {
        self.scopes.push(FxHashMap::default());
    }

    /// Pop the innermost scope and return any bindings that were never used.
    fn pop(&mut self) -> Vec<(String, Binding)> {
        self.scopes
            .pop()
            .unwrap_or_default()
            .into_iter()
            .filter(|(name, b)| b.use_count == 0 && !b.is_wildcard && name != "_")
            .collect()
    }

    /// Declare a new variable in the innermost scope.
    fn declare(&mut self, name: impl Into<String>, binding: Binding) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.into(), binding);
        }
    }

    /// Mark a variable as used.  Returns `true` if found, `false` if undefined.
    fn mark_used(&mut self, name: &str) -> bool {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(b) = scope.get_mut(name) {
                b.use_count += 1;
                return true;
            }
        }
        false
    }

    /// Look up a binding (immutable).
    fn lookup(&self, name: &str) -> Option<&Binding> {
        for scope in self.scopes.iter().rev() {
            if let Some(b) = scope.get(name) {
                return Some(b);
            }
        }
        None
    }

    /// Returns true when `name` is declared in any outer scope (not innermost),
    /// used to detect shadowing.
    fn is_outer_name(&self, name: &str) -> bool {
        let len = self.scopes.len();
        if len < 2 {
            return false;
        }
        self.scopes[..len - 1]
            .iter()
            .rev()
            .any(|s| s.contains_key(name))
    }
}

// =============================================================================
// §0d  CONTROL-FLOW CONTEXT STACK
// =============================================================================

/// What kind of enclosing construct we are currently inside.
#[derive(Debug, Clone, PartialEq)]
enum CfFrame {
    /// Inside a `fn` / `system` / `kernel` body.
    Function { is_async: bool, name: String },
    /// Inside a `loop`, `while`, or `for` loop — carries its optional label.
    Loop { label: Option<String> },
    /// Inside a `parallel for`.
    ParallelLoop { label: Option<String> },
    /// Inside a `spawn { }` task.
    Spawn,
    /// Inside a `sync { }` barrier.
    Sync,
    /// Inside an `atomic { }` region.
    Atomic,
}

#[derive(Debug, Default)]
struct CfStack {
    frames: Vec<CfFrame>,
}

impl CfStack {
    fn push(&mut self, f: CfFrame) {
        self.frames.push(f);
    }
    fn pop(&mut self) {
        self.frames.pop();
    }

    fn in_loop(&self) -> bool {
        self.frames
            .iter()
            .rev()
            .any(|f| matches!(f, CfFrame::Loop { .. } | CfFrame::ParallelLoop { .. }))
    }

    fn in_async_fn(&self) -> bool {
        self.frames
            .iter()
            .rev()
            .any(|f| matches!(f, CfFrame::Function { is_async: true, .. }))
    }

    fn in_function(&self) -> bool {
        self.frames
            .iter()
            .any(|f| matches!(f, CfFrame::Function { .. }))
    }

    /// Returns `true` if a loop with `label` is reachable from current position.
    fn has_label(&self, label: &str) -> bool {
        self.frames.iter().rev().any(|f| match f {
            CfFrame::Loop { label: Some(l), .. } => l == label,
            CfFrame::ParallelLoop { label: Some(l), .. } => l == label,
            _ => false,
        })
    }

    fn current_fn_name(&self) -> Option<&str> {
        self.frames.iter().rev().find_map(|f| match f {
            CfFrame::Function { name, .. } => Some(name.as_str()),
            _ => None,
        })
    }
}

// =============================================================================
// §0e  DECLARATION REGISTRY  (for unused-decl detection)
// =============================================================================

#[derive(Debug, Clone)]
struct DeclRecord {
    span: Span,
    use_count: u32,
    kind: DeclKind,
    deprecated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeclKind {
    Function,
    System,
    Component,
    Struct,
    Enum,
    Agent,
    Model,
    Const,
    Shader,
    Scene,
    Prefab,
    Loss,
    PhysicsConfig,
}

#[derive(Debug, Default)]
struct DeclRegistry {
    records: FxHashMap<String, DeclRecord>,
}

impl DeclRegistry {
    fn register(&mut self, name: impl Into<String>, span: Span, kind: DeclKind) {
        self.records.insert(
            name.into(),
            DeclRecord {
                span,
                use_count: 0,
                kind,
                deprecated: false,
            },
        );
    }

    fn register_deprecated(&mut self, name: impl Into<String>, span: Span, kind: DeclKind) {
        self.records.insert(
            name.into(),
            DeclRecord {
                span,
                use_count: 0,
                kind,
                deprecated: true,
            },
        );
    }

    fn is_deprecated(&self, name: &str) -> bool {
        self.records.get(name).map_or(false, |r| r.deprecated)
    }

    fn mark_used(&mut self, name: &str) {
        if let Some(r) = self.records.get_mut(name) {
            r.use_count += 1;
        }
    }

    fn unused(&self) -> impl Iterator<Item = (&str, &DeclRecord)> {
        self.records
            .iter()
            .filter(|(name, r)| r.use_count == 0 && !name.starts_with('_'))
            .map(|(n, r)| (n.as_str(), r))
    }
}

// =============================================================================
// §0f  COMPONENT GRAPH  (for ECS aliasing analysis)
// =============================================================================

/// The read / write set for one entity-loop or system.
#[derive(Debug, Default, Clone)]
struct ComponentSet {
    reads: FxHashSet<String>,
    writes: FxHashSet<String>,
}

impl ComponentSet {
    fn add(&mut self, component: &str, mode: AccessMode) {
        match mode {
            AccessMode::Read => {
                self.reads.insert(component.to_owned());
            }
            AccessMode::Write => {
                self.writes.insert(component.to_owned());
            }
            AccessMode::ReadWrite => {
                self.reads.insert(component.to_owned());
                self.writes.insert(component.to_owned());
            }
        }
    }

    /// True if this set's writes conflict with `other`'s reads (or vice versa).
    fn conflicts_with(&self, other: &ComponentSet) -> Option<String> {
        // self writes something other reads
        for comp in &self.writes {
            if other.reads.contains(comp) || other.writes.contains(comp) {
                return Some(comp.clone());
            }
        }
        // other writes something self reads
        for comp in &other.writes {
            if self.reads.contains(comp) {
                return Some(comp.clone());
            }
        }
        None
    }
}

// =============================================================================
// §1  MAIN CONTEXT
// =============================================================================

/// The top-level semantic analysis context.
pub struct SemaCtx {
    pub diag: Diagnostics,
    scopes: ScopeStack,
    cf: CfStack,
    decls: DeclRegistry,
    /// All component names declared in the program.
    components: FxHashSet<String>,
    /// Per-system component access sets (for aliasing analysis).
    system_sets: Vec<(String, Span, ComponentSet)>,
    /// All agent names that appear in `train` blocks (to warn on unused learning).
    trained_agents: FxHashSet<String>,
    /// Whether to emit shadowing warnings (controlled by a lint flag).
    warn_shadow: bool,
}

impl SemaCtx {
    pub fn new() -> Self {
        SemaCtx {
            diag: Diagnostics::default(),
            scopes: ScopeStack::default(),
            cf: CfStack::default(),
            decls: DeclRegistry::default(),
            components: FxHashSet::default(),
            system_sets: Vec::new(),
            trained_agents: FxHashSet::default(),
            warn_shadow: true,
        }
    }

    /// Disable shadowing warnings (e.g., for REPL / interactive mode).
    pub fn silence_shadow_warnings(&mut self) {
        self.warn_shadow = false;
    }

    // ── Convenience wrappers ────────────────────────────────────────────────

    fn err(&mut self, span: Span, msg: impl Into<String>) {
        self.diag.error(span, msg);
    }
    fn warn(&mut self, span: Span, msg: impl Into<String>) {
        self.diag.warning(span, msg);
    }
    fn note(&mut self, span: Span, msg: impl Into<String>) {
        self.diag.note(span, msg);
    }

    fn push_scope(&mut self) {
        self.scopes.push();
    }

    fn pop_scope(&mut self) {
        for (name, binding) in self.scopes.pop() {
            self.warn(
                binding.span,
                format!("unused variable `{}`; prefix with `_` to suppress", name),
            );
        }
    }

    fn declare_var(&mut self, name: &str, span: Span, mutable: bool) {
        if self.warn_shadow && self.scopes.is_outer_name(name) {
            self.warn(
                span,
                format!("binding `{}` shadows an outer variable", name),
            );
        }
        let wildcard = name == "_" || name.starts_with('_');
        self.scopes
            .declare(name, Binding::new(span, mutable, wildcard));
    }

    fn use_var(&mut self, name: &str, _span: Span) {
        // Known built-ins that don't live in the scope stack.
        if matches!(name, "world" | "self" | "true" | "false") {
            return;
        }
        if !self.scopes.mark_used(name) {
            // Check whether it's a top-level declaration (function call etc.)
            self.decls.mark_used(name);
            // If also not a top-level decl, the typeck pass will have caught it;
            // emit a note here rather than a duplicate error so we don't pile on.
        }
    }
}

// =============================================================================
// §2  PROGRAM-LEVEL ENTRY POINT
// =============================================================================

impl SemaCtx {
    /// Run the full semantic analysis pass over a parsed program.
    pub fn analyse(&mut self, program: &Program) {
        // ── Collect all top-level declarations first (two-pass). ──────────────
        self.collect_decls(program);

        // Treat `main` as the implicit entry point. If it is declared, we
        // should never warn that it is unused.
        if self.decls.records.contains_key("main") {
            self.decls.mark_used("main");
        }

        // ── Pass 2: analyse item bodies / statements / expressions. ──────────
        for item in &program.items {
            self.analyse_item(item);
        }

        // ── Post-pass checks that need the complete picture. ──────────────────
        self.check_unused_decls();
        self.check_system_aliasing();
        self.check_untrained_learnable_agents(program);
    }

    // ── Pass 1: register names ─────────────────────────────────────────────

    fn collect_decls(&mut self, program: &Program) {
        for item in &program.items {
            match item {
                Item::Fn(f) => self.decls.register(&f.name, f.span, DeclKind::Function),
                Item::System(s) => self.decls.register(&s.name, s.span, DeclKind::System),
                Item::Component(c) => {
                    self.decls.register(&c.name, c.span, DeclKind::Component);
                    self.components.insert(c.name.clone());
                }
                Item::Struct(s) => self.decls.register(&s.name, s.span, DeclKind::Struct),
                Item::Enum(e) => self.decls.register(&e.name, e.span, DeclKind::Enum),
                Item::Agent(a) => self.decls.register(&a.name, a.span, DeclKind::Agent),
                Item::Model(m) => self.decls.register(&m.name, m.span, DeclKind::Model),
                Item::Const(c) => self.decls.register(&c.name, c.span, DeclKind::Const),
                Item::Mod {
                    items: Some(inner), ..
                } => {
                    // Recurse into inline modules.
                    for i in inner {
                        self.collect_item_decl(i);
                    }
                }
                _ => {}
            }
        }
    }

    fn collect_item_decl(&mut self, item: &Item) {
        match item {
            Item::Fn(f) => self.decls.register(&f.name, f.span, DeclKind::Function),
            Item::System(s) => self.decls.register(&s.name, s.span, DeclKind::System),
            Item::Component(c) => {
                self.decls.register(&c.name, c.span, DeclKind::Component);
                self.components.insert(c.name.clone());
            }
            Item::Struct(s) => self.decls.register(&s.name, s.span, DeclKind::Struct),
            Item::Enum(e) => self.decls.register(&e.name, e.span, DeclKind::Enum),
            Item::Agent(a) => self.decls.register(&a.name, a.span, DeclKind::Agent),
            Item::Model(m) => self.decls.register(&m.name, m.span, DeclKind::Model),
            Item::Const(c) => self.decls.register(&c.name, c.span, DeclKind::Const),
            _ => {}
        }
    }
}

// =============================================================================
// §3  ITEM-LEVEL ANALYSIS
// =============================================================================

impl SemaCtx {
    fn analyse_item(&mut self, item: &Item) {
        match item {
            Item::Fn(f) => self.analyse_fn(f),
            Item::System(s) => self.analyse_system(s),
            Item::Component(c) => self.analyse_component(c),
            Item::Agent(a) => self.analyse_agent(a),
            Item::Model(m) => self.analyse_model(m),
            Item::Train(t) => self.analyse_train(t),
            Item::Const(c) => {
                self.cf.push(CfFrame::Function {
                    is_async: false,
                    name: "<const>".into(),
                });
                self.push_scope();
                self.analyse_expr(&c.value);
                self.pop_scope();
                self.cf.pop();
            }
            Item::Mod {
                items: Some(inner), ..
            } => {
                for i in inner {
                    self.analyse_item(i);
                }
            }
            Item::Struct(_) | Item::Enum(_) | Item::Use(_) | Item::Mod { .. } => {}
            _ => {}
        }
    }

    // =========================================================================
    // §3a  FUNCTIONS
    // =========================================================================

    fn analyse_fn(&mut self, f: &FnDecl) {
        self.cf.push(CfFrame::Function {
            is_async: f.is_async,
            name: f.name.clone(),
        });
        self.push_scope();

        for param in &f.params {
            self.declare_var(&param.name, param.span, param.mutable);
        }

        if let Some(body) = &f.body {
            self.analyse_block(body);
        }

        self.pop_scope();
        self.cf.pop();
    }

    // =========================================================================
    // §3b  SYSTEMS  (ECS Feature 2)
    // =========================================================================

    fn analyse_system(&mut self, s: &SystemDecl) {
        // ── Validate the explicit query (if present). ─────────────────────────
        if let Some(q) = &s.explicit_query {
            self.validate_entity_query(q, s.span);
        }

        // ── Check the `@simd` / `@parallel` + empty query rule. ───────────────
        let has_parallel_attr = s
            .attrs
            .iter()
            .any(|a| matches!(a, Attribute::Simd | Attribute::Parallel | Attribute::Gpu));
        if has_parallel_attr {
            if let Some(q) = &s.explicit_query {
                if q.with.is_empty() {
                    self.warn(
                        s.span,
                        format!(
                            "system `{}` is annotated for parallel execution but has \
                         an unconstrained query (`with` is empty); \
                         this may process more entities than intended",
                            s.name
                        ),
                    );
                }
            }
        }

        // ── Build a component access set for this system (for alias checks). ──
        let mut comp_set = ComponentSet::default();
        for acc in &s.accesses {
            comp_set.add(&acc.component, acc.mode);
        }
        self.system_sets.push((s.name.clone(), s.span, comp_set));

        // ── Analyse the body. ─────────────────────────────────────────────────
        self.cf.push(CfFrame::Function {
            is_async: false,
            name: s.name.clone(),
        });
        self.push_scope();

        // Inject system parameters and the world handle.
        for param in &s.params {
            self.declare_var(&param.name, param.span, param.mutable);
        }
        self.scopes
            .declare("world", Binding::new(s.span, false, false));

        self.analyse_block(&s.body);

        self.pop_scope();
        self.cf.pop();
    }

    fn validate_entity_query(&mut self, q: &EntityQuery, ctx_span: Span) {
        // All component names must be declared.
        for comp in q.with.iter().chain(q.without.iter()) {
            if !self.components.contains(comp.as_str()) {
                self.err(
                    q.span,
                    format!(
                        "query references component `{}` which has not been declared \
                     with the `component` keyword",
                        comp
                    ),
                );
            }
            // Mark it used so the unused-decl check doesn't fire.
            self.decls.mark_used(comp);
        }

        // Duplicate component names within a query are always a mistake.
        let mut seen_with: FxHashSet<&str> = FxHashSet::default();
        let mut seen_without: FxHashSet<&str> = FxHashSet::default();

        for comp in &q.with {
            if !seen_with.insert(comp.as_str()) {
                self.err(
                    q.span,
                    format!(
                        "component `{}` appears more than once in `with(…)` clause",
                        comp
                    ),
                );
            }
        }
        for comp in &q.without {
            if !seen_without.insert(comp.as_str()) {
                self.err(
                    q.span,
                    format!(
                        "component `{}` appears more than once in `without(…)` clause",
                        comp
                    ),
                );
            }
        }

        // A component cannot be in both `with` and `without`.
        for comp in &q.with {
            if q.without.contains(comp) {
                self.err(
                    q.span,
                    format!(
                        "component `{}` appears in both `with(…)` and `without(…)`",
                        comp
                    ),
                );
            }
        }

        let _ = ctx_span;
    }

    /// Cross-system alias analysis: run after all systems have been collected.
    fn check_system_aliasing(&mut self) {
        let sets = self.system_sets.clone();
        for i in 0..sets.len() {
            for j in (i + 1)..sets.len() {
                let (name_a, span_a, set_a) = &sets[i];
                let (name_b, span_b, set_b) = &sets[j];
                if let Some(comp) = set_a.conflicts_with(set_b) {
                    self.diag.push(
                        Diagnostic::warning(
                            *span_a,
                            format!(
                                "read/write aliasing on component `{}` between \
                             system `{}` and system `{}`; \
                             concurrent execution may produce non-deterministic results",
                                comp, name_a, name_b
                            ),
                        )
                        .with_label(
                            *span_b,
                            format!("system `{}` also accesses `{}`", name_b, comp),
                        ),
                    );
                }
            }
        }
    }

    // =========================================================================
    // §3c  COMPONENTS
    // =========================================================================

    fn analyse_component(&mut self, c: &ComponentDecl) {
        // Field names must be unique within the component.
        let mut seen: FxHashSet<&str> = FxHashSet::default();
        for field in &c.fields {
            if !seen.insert(field.name.as_str()) {
                self.err(
                    field.span,
                    format!(
                        "field `{}` is declared more than once in component `{}`",
                        field.name, c.name
                    ),
                );
            }
        }
    }

    // =========================================================================
    // §3d  AGENTS  (Feature 3)
    // =========================================================================

    fn analyse_agent(&mut self, a: &AgentDecl) {
        // Mark the agent as used by itself (it may be referenced by `train`).
        // Actual use tracking happens when a `train` references it.

        // §4a  At least one behaviour.
        if a.behaviors.is_empty() {
            self.warn(
                a.span,
                format!(
                    "agent `{}` declares no behaviours; it will never act",
                    a.name
                ),
            );
        }

        // §4b  Behaviour priorities must be unique.
        {
            let mut seen_prio: FxHashMap<u32, &str> = FxHashMap::default();
            for rule in &a.behaviors {
                let prio = rule.priority.0;
                if let Some(prev) = seen_prio.insert(prio, rule.name.as_str()) {
                    self.err(
                        rule.span,
                        format!(
                            "behaviour `{}` and `{}` share the same priority {} \
                         in agent `{}`; priorities must be unique",
                            rule.name, prev, prio, a.name
                        ),
                    );
                }
            }
        }

        // §4c  Perception: no duplicate kinds.
        {
            let mut seen_kinds: Vec<&PerceptionKind> = Vec::new();
            for perc in &a.perceptions {
                let duplicate = seen_kinds.iter().any(|k| perception_kind_eq(k, &perc.kind));
                if duplicate {
                    self.err(
                        perc.span,
                        format!(
                            "agent `{}` declares the same perception kind more than once",
                            a.name
                        ),
                    );
                } else {
                    seen_kinds.push(&perc.kind);
                }
            }
        }

        // §4d  Memory capacity must be > 0.
        if let Some(mem) = &a.memory {
            use crate::ast::MemoryCapacity;
            match &mem.capacity {
                Some(MemoryCapacity::Slots(0)) => {
                    self.err(mem.span, "memory `slots` capacity must be greater than 0");
                }
                Some(MemoryCapacity::Duration { seconds }) if *seconds <= 0.0 => {
                    self.err(mem.span, "memory retention duration must be > 0 seconds");
                }
                _ => {}
            }
        }

        // §4e  Learning config sanity.
        if let Some(ls) = &a.learning {
            if let Some(lr) = ls.learning_rate {
                if lr <= 0.0 || lr.is_nan() {
                    self.err(
                        ls.span,
                        format!("learning_rate must be a positive finite number; got {}", lr),
                    );
                }
            }
            if let Some(g) = ls.gamma {
                if !(0.0..=1.0).contains(&g) {
                    self.err(
                        ls.span,
                        format!("discount factor γ must be in [0, 1]; got {}", g),
                    );
                }
            }
        }

        // §4f  Analyse behaviour bodies.
        for rule in &a.behaviors {
            self.analyse_behaviour(rule, a);
        }

        // §4g  Analyse goal utility expressions (side-effect check).
        for goal in &a.goals {
            self.analyse_goal(goal, a);
        }
    }

    fn analyse_behaviour(&mut self, rule: &BehaviorRule, agent: &AgentDecl) {
        self.cf.push(CfFrame::Function {
            is_async: false,
            name: format!("{}::{}", agent.name, rule.name),
        });
        self.push_scope();

        // Inject `self` and parameters.
        self.scopes
            .declare("self", Binding::new(rule.span, false, false));
        for param in &rule.params {
            self.declare_var(&param.name, param.span, param.mutable);
        }

        self.analyse_block(&rule.body);

        self.pop_scope();
        self.cf.pop();
    }

    fn analyse_goal(&mut self, goal: &GoalDecl, agent: &AgentDecl) {
        // Goal utility expressions must not produce side effects.
        // We approximate this by checking for assignments and calls.
        if expr_has_side_effects(&goal.utility) {
            self.warn(
                goal.utility.span(),
                format!(
                    "goal `{}` utility expression in agent `{}` appears to have \
                 side effects (contains assignment or call); \
                 utility expressions should be pure",
                    goal.name, agent.name
                ),
            );
        }

        self.push_scope();
        self.scopes
            .declare("self", Binding::new(goal.utility.span(), false, false));
        self.analyse_expr(&goal.utility);
        self.pop_scope();
    }

    // =========================================================================
    // §3e  MODELS  (Unique Feature 1)
    // =========================================================================

    fn analyse_model(&mut self, m: &ModelDecl) {
        // §5a  Exactly one `input` and one `output` layer.
        let input_count = m
            .layers
            .iter()
            .filter(|l| matches!(l, ModelLayer::Input { .. }))
            .count();
        let output_count = m
            .layers
            .iter()
            .filter(|l| matches!(l, ModelLayer::Output { .. }))
            .count();

        if input_count == 0 {
            self.err(
                m.span,
                format!(
                    "model `{}` has no `input` layer; every model must declare an input shape",
                    m.name
                ),
            );
        } else if input_count > 1 {
            self.err(
                m.span,
                format!(
                    "model `{}` declares {} `input` layers; only one is allowed",
                    m.name, input_count
                ),
            );
        }

        if output_count == 0 {
            self.err(
                m.span,
                format!(
                    "model `{}` has no `output` layer; every model must declare an output shape",
                    m.name
                ),
            );
        } else if output_count > 1 {
            self.err(
                m.span,
                format!(
                    "model `{}` declares {} `output` layers; only one is allowed",
                    m.name, output_count
                ),
            );
        }

        // §5b  `input` must be the first layer, `output` the last.
        if let Some(first) = m.layers.first() {
            if !matches!(first, ModelLayer::Input { .. }) {
                self.err(
                    first.span(),
                    format!("first layer of model `{}` should be `input`", m.name),
                );
            }
        }
        if let Some(last) = m.layers.last() {
            if !matches!(last, ModelLayer::Output { .. }) {
                self.warn(
                    last.span(),
                    format!(
                        "last layer of model `{}` is not an `output` layer; \
                     the model output shape will be implicit",
                        m.name
                    ),
                );
            }
        }

        // §5c  No adjacent duplicate layer types (catches obvious copy-paste errors).
        let mut prev_kind: Option<&str> = None;
        for layer in &m.layers {
            let kind = model_layer_kind_name(layer);
            if Some(kind) == prev_kind
                && !matches!(layer, ModelLayer::Dense { .. } | ModelLayer::Conv2d { .. })
            {
                self.warn(
                    layer.span(),
                    format!(
                        "model `{}` has adjacent duplicate `{}` layers; \
                     is this intentional?",
                        m.name, kind
                    ),
                );
            }
            prev_kind = Some(kind);
        }

        // §5d  Sub-model references must exist.
        for layer in &m.layers {
            if let ModelLayer::SubModel { span, name } = layer {
                self.decls.mark_used(name);
                // Actual existence checked by typeck; here we just mark it used.
                let _ = span;
            }
        }
    }

    // =========================================================================
    // §3f  TRAIN  (Unique Feature 2)
    // =========================================================================

    fn analyse_train(&mut self, t: &TrainDecl) {
        // §5e  References must resolve.
        self.decls.mark_used(&t.agent);
        if let Some(m) = &t.model {
            self.decls.mark_used(m);
        }

        // Track that this agent is being trained.
        self.trained_agents.insert(t.agent.clone());

        // §5f  Reward / penalty signal names must be unique.
        {
            let mut seen: FxHashSet<&str> = FxHashSet::default();
            for sig in &t.signals {
                if !seen.insert(sig.name.as_str()) {
                    self.err(
                        sig.span,
                        format!(
                            "signal name `{}` appears more than once in `train` block",
                            sig.name
                        ),
                    );
                }
            }
        }

        // §5g  Signal weights must be non-negative.
        for sig in &t.signals {
            if sig.weight < 0.0 {
                self.err(
                    sig.span,
                    format!(
                        "{} signal `{}` has negative weight {}; \
                     use a `penalty` with positive weight for negative reinforcement",
                        if sig.is_reward { "reward" } else { "penalty" },
                        sig.name,
                        sig.weight
                    ),
                );
            }
        }

        // §5h  Episode constraints.
        if let Some(ep) = &t.episode {
            if matches!(ep.max_steps, Some(0)) {
                self.err(ep.span, "episode `max_steps` must be ≥ 1");
            }
            if matches!(ep.num_envs, Some(0)) {
                self.err(ep.span, "episode `num_envs` must be ≥ 1");
            }
        }

        // §5i  Hyper-parameter keys must be unique.
        {
            let mut seen: FxHashSet<&str> = FxHashSet::default();
            for (key, expr) in &t.hyper {
                if !seen.insert(key.as_str()) {
                    self.err(
                        expr.span(),
                        format!("hyper-parameter `{}` is specified more than once", key),
                    );
                }
            }
        }

        // §5j  Analyse hyper-value expressions.
        for (_, expr) in &t.hyper {
            self.push_scope();
            self.analyse_expr(expr);
            self.pop_scope();
        }

        // §5k  Analyse signal expressions.
        for sig in &t.signals {
            if let Some(e) = &sig.expr {
                self.push_scope();
                self.analyse_expr(e);
                self.pop_scope();
            }
        }

        // §5l  Analyse done_condition.
        if let Some(ep) = &t.episode {
            if let Some(cond) = &ep.done_condition {
                self.push_scope();
                self.analyse_expr(cond);
                self.pop_scope();
            }
        }
    }

    /// Post-pass: agents that have `learning ≠ None` but are never trained get a warning.
    fn check_untrained_learnable_agents(&mut self, program: &Program) {
        for agent in program.agents() {
            if agent.is_learnable() && !self.trained_agents.contains(&agent.name) {
                self.warn(
                    agent.span,
                    format!(
                        "agent `{}` has a `learning` configuration but is never referenced \
                     in a `train` block; the agent will not improve at runtime",
                        agent.name
                    ),
                );
            }
        }
    }
}

// =============================================================================
// §4  BLOCK / STATEMENT ANALYSIS
// =============================================================================

impl SemaCtx {
    fn analyse_block(&mut self, block: &Block) {
        self.push_scope();

        let stmts = &block.stmts;
        let mut terminated = false; // true once we see return/break/continue/Never

        for (i, stmt) in stmts.iter().enumerate() {
            if terminated {
                // Unreachable code: only emit for the first dead statement
                // so we don't flood the user.
                if i == stmts.len() - 1 || !is_dead_item(stmt) {
                    self.warn(stmt_span(stmt), "unreachable code after terminator");
                }
                // Keep going to catch further errors, but mark all remaining
                // as dead.
            }
            self.analyse_stmt(stmt);
            if stmt_terminates(stmt) {
                terminated = true;
            }
        }

        if let Some(tail) = &block.tail {
            if terminated {
                self.warn(tail.span(), "unreachable expression after terminator");
            }
            self.analyse_expr(tail);
        }

        self.pop_scope();
    }

    fn analyse_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            // ── let binding ───────────────────────────────────────────────────
            Stmt::Let {
                span,
                pattern,
                init,
                mutable,
                ..
            } => {
                // Analyse RHS first (it cannot reference the new binding).
                if let Some(e) = init {
                    self.analyse_expr(e);
                } else if !mutable {
                    // `let x;` with no initialiser is allowed for later assignment,
                    // but warn if the type is not explicitly annotated either.
                }
                // Bind pattern names.
                self.bind_pattern(pattern, *span, *mutable);
            }

            // ── expression statement ──────────────────────────────────────────
            Stmt::Expr {
                expr,
                has_semi,
                span,
            } => {
                self.analyse_expr(expr);
                // Warn when a value-producing expression is silently discarded.
                if *has_semi && expr_produces_value(expr) && !is_call_or_assign(expr) {
                    self.note(
                        *span,
                        "expression value discarded; use `let _ = …` if intentional",
                    );
                }
            }

            // ── return ────────────────────────────────────────────────────────
            Stmt::Return { span, value } => {
                if !self.cf.in_function() {
                    self.err(*span, "`return` outside of a function");
                }
                if let Some(e) = value {
                    self.analyse_expr(e);
                }
            }

            // ── break / continue ──────────────────────────────────────────────
            Stmt::Break { span, value, label } => {
                if !self.cf.in_loop() {
                    self.err(*span, "`break` outside of a loop");
                } else if let Some(lbl) = label {
                    if !self.cf.has_label(lbl) {
                        self.err(
                            *span,
                            format!("`break '{lbl}` does not correspond to any enclosing loop"),
                        );
                    }
                }
                if let Some(e) = value {
                    self.analyse_expr(e);
                }
            }

            Stmt::Continue { span, label } => {
                if !self.cf.in_loop() {
                    self.err(*span, "`continue` outside of a loop");
                } else if let Some(lbl) = label {
                    if !self.cf.has_label(lbl) {
                        self.err(
                            *span,
                            format!("`continue '{lbl}` does not correspond to any enclosing loop"),
                        );
                    }
                }
            }

            // ── for … in … ────────────────────────────────────────────────────
            Stmt::ForIn {
                span,
                pattern,
                iter,
                body,
                label,
            } => {
                self.analyse_expr(iter);
                self.cf.push(CfFrame::Loop {
                    label: label.clone(),
                });
                self.push_scope();
                self.bind_pattern(pattern, *span, false);
                self.analyse_block(body);
                self.pop_scope();
                self.cf.pop();
            }

            // ── entity for (ECS) ──────────────────────────────────────────────
            Stmt::EntityFor {
                span,
                var,
                query,
                body,
                label,
                accesses,
                parallelism,
            } => {
                // Validate the query.
                self.validate_entity_query(query, *span);

                // §3 Determinism check: if the loop is parallel/SIMD, warn on writes.
                let is_parallel = matches!(
                    parallelism,
                    crate::ast::ParallelismHint::Parallel
                        | crate::ast::ParallelismHint::Simd
                        | crate::ast::ParallelismHint::Gpu
                        | crate::ast::ParallelismHint::SimdOrGpu { .. }
                );
                for acc in accesses {
                    if acc.mode.is_write() && is_parallel {
                        // Only flag if there's a component that is *also* read by
                        // another iteration (i.e., it's in the `reads` set too).
                        // As a conservative approximation, flag any write.
                        self.warn(
                            *span,
                            format!(
                                "entity loop writes component `{}` under parallel \
                             execution; if iterations are order-dependent this \
                             produces non-deterministic results — \
                             add `@seq` to force determinism",
                                acc.component
                            ),
                        );
                    }
                }

                // Build a set for the aliasing check.
                let mut comp_set = ComponentSet::default();
                for acc in accesses {
                    comp_set.add(&acc.component, acc.mode);
                }

                // Check for self-aliasing: a component that is both read and
                // written within the *same* loop is only safe if iterations are
                // independent.
                for comp in &comp_set.reads {
                    if comp_set.writes.contains(comp) && is_parallel {
                        self.err(
                            *span,
                            format!(
                                "component `{}` is both read and written inside a \
                             parallel entity loop; this is a data race — \
                             split into separate systems or use `@seq`",
                                comp
                            ),
                        );
                    }
                }

                self.cf.push(CfFrame::Loop {
                    label: label.clone(),
                });
                self.push_scope();
                self.declare_var(var, *span, false);
                self.analyse_block(body);
                self.pop_scope();
                self.cf.pop();
            }

            // ── while / loop ──────────────────────────────────────────────────
            Stmt::While {
                cond, body, label, ..
            } => {
                self.analyse_expr(cond);
                // Constant `false` condition → unreachable body.
                if let Expr::BoolLit { value: false, span } = cond {
                    self.warn(*span, "`while false { … }` loop body is unreachable");
                }
                self.cf.push(CfFrame::Loop {
                    label: label.clone(),
                });
                self.push_scope();
                self.analyse_block(body);
                // Check for infinite loop: `while true` with no reachable break.
                if let Expr::BoolLit { value: true, span } = cond {
                    if !block_has_break(body) {
                        self.warn(*span, "`while true` with no `break` is an infinite loop");
                    }
                }
                self.pop_scope();
                self.cf.pop();
            }

            Stmt::Loop { body, label, .. } => {
                self.cf.push(CfFrame::Loop {
                    label: label.clone(),
                });
                self.push_scope();
                self.analyse_block(body);
                if !block_has_break(body) {
                    self.warn(
                        body.span,
                        "`loop { }` has no reachable `break`; \
                               this is an infinite loop",
                    );
                }
                self.pop_scope();
                self.cf.pop();
            }

            // ── if / match ────────────────────────────────────────────────────
            Stmt::If {
                span,
                cond,
                then,
                else_,
            } => {
                self.analyse_expr(cond);
                // Constant-condition dead-branch detection.
                match cond {
                    Expr::BoolLit {
                        value: true,
                        span: cs,
                    } => self.warn(*cs, "`if true { … }` — else branch (if any) is unreachable"),
                    Expr::BoolLit {
                        value: false,
                        span: cs,
                    } => self.warn(*cs, "`if false { … }` — then branch is unreachable"),
                    _ => {}
                }
                self.push_scope();
                self.analyse_block(then);
                self.pop_scope();
                if let Some(e) = else_ {
                    match e.as_ref() {
                        crate::ast::IfOrBlock::If(s) => self.analyse_stmt(s),
                        crate::ast::IfOrBlock::Block(b) => {
                            self.push_scope();
                            self.analyse_block(b);
                            self.pop_scope();
                        }
                    }
                }
                let _ = span;
            }

            Stmt::Match { expr, arms, span } => {
                self.analyse_expr(expr);
                for arm in arms {
                    self.analyse_match_arm(arm);
                }
                let _ = span;
            }

            // ── nested item ───────────────────────────────────────────────────
            Stmt::Item(i) => {
                self.analyse_item(i);
            }

            // ── parallel constructs (Feature 4) ──────────────────────────────
            Stmt::ParallelFor(pf) => {
                self.analyse_parallel_for(pf);
            }

            Stmt::Spawn(sb) => {
                // Captures should be used inside the block; check for
                // accidentally-captured mutable variables.
                self.cf.push(CfFrame::Spawn);
                self.push_scope();
                self.analyse_block(&sb.body);
                self.pop_scope();
                self.cf.pop();
            }

            Stmt::Sync(sb) => {
                self.cf.push(CfFrame::Sync);
                self.push_scope();
                self.analyse_block(&sb.body);
                self.pop_scope();
                self.cf.pop();
            }

            Stmt::Atomic(ab) => {
                // Nested `atomic` is not allowed.
                if matches!(self.cf.frames.last(), Some(CfFrame::Atomic)) {
                    self.err(ab.span, "nested `atomic { }` blocks are not allowed");
                }
                self.cf.push(CfFrame::Atomic);
                self.push_scope();
                self.analyse_block(&ab.body);
                self.pop_scope();
                self.cf.pop();
            }
        }
    }

    fn analyse_match_arm(&mut self, arm: &MatchArm) {
        self.push_scope();
        self.bind_pattern(&arm.pat, arm.span, false);
        if let Some(g) = &arm.guard {
            self.analyse_expr(g);
        }
        self.analyse_expr(&arm.body);
        self.pop_scope();
    }

    fn analyse_parallel_for(&mut self, pf: &ParallelFor) {
        self.analyse_expr(&pf.iter);
        self.cf.push(CfFrame::ParallelLoop {
            label: pf.label.clone(),
        });
        self.push_scope();
        self.bind_pattern(&pf.var, pf.span, false);
        self.analyse_block(&pf.body);
        // Warn if a parallel loop captures a mutable outer binding.
        self.pop_scope();
        self.cf.pop();
    }

    // =========================================================================
    // §5  EXPRESSION ANALYSIS
    // =========================================================================

    fn analyse_expr(&mut self, expr: &Expr) {
        match expr {
            // Literals produce no variable references.
            Expr::IntLit { .. }
            | Expr::FloatLit { .. }
            | Expr::BoolLit { .. }
            | Expr::StrLit { .. } => {}

            Expr::Ident { span, name } => {
                self.use_var(name, *span);
                // Mark top-level decl as used too.
                self.decls.mark_used(name);
            }

            Expr::Path { span, segments } => {
                // Mark each path segment as used.
                for seg in segments {
                    self.decls.mark_used(seg);
                    self.use_var(seg, *span);
                }
            }

            Expr::VecCtor { elems, .. } | Expr::ArrayLit { elems, .. } => {
                for e in elems {
                    self.analyse_expr(e);
                }
            }

            Expr::Tuple { elems, .. } => {
                for e in elems {
                    self.analyse_expr(e);
                }
            }

            Expr::BinOp { lhs, rhs, .. } => {
                self.analyse_expr(lhs);
                self.analyse_expr(rhs);
            }

            Expr::UnOp { expr, .. } => {
                self.analyse_expr(expr);
            }

            Expr::Assign { target, value, .. } => {
                // Check that the target is actually mutable.
                self.check_lvalue_mutability(target);
                self.analyse_expr(target);
                self.analyse_expr(value);
            }

            Expr::Field { object, .. } => {
                self.analyse_expr(object);
            }

            Expr::Index {
                object, indices, ..
            } => {
                self.analyse_expr(object);
                for i in indices {
                    self.analyse_expr(i);
                }
            }

            Expr::Call {
                func, args, named, ..
            } => {
                self.analyse_expr(func);
                // Mark callee as used in the decl registry.
                if let Expr::Ident { name, .. } = func.as_ref() {
                    self.decls.mark_used(name);
                }
                for a in args {
                    self.analyse_expr(a);
                }
                for (_, a) in named {
                    self.analyse_expr(a);
                }
            }

            Expr::MethodCall { receiver, args, .. } => {
                self.analyse_expr(receiver);
                for a in args {
                    self.analyse_expr(a);
                }
            }

            Expr::MatMul { lhs, rhs, .. }
            | Expr::HadamardMul { lhs, rhs, .. }
            | Expr::HadamardDiv { lhs, rhs, .. }
            | Expr::TensorConcat { lhs, rhs, .. }
            | Expr::Pow {
                base: lhs,
                exp: rhs,
                ..
            } => {
                self.analyse_expr(lhs);
                self.analyse_expr(rhs);
            }

            Expr::Grad { inner, .. } => {
                self.analyse_expr(inner);
            }

            Expr::Range { lo, hi, .. } => {
                if let Some(e) = lo {
                    self.analyse_expr(e);
                }
                if let Some(e) = hi {
                    self.analyse_expr(e);
                }
            }

            Expr::Cast { expr, .. } => {
                self.analyse_expr(expr);
            }

            Expr::IfExpr {
                cond, then, else_, ..
            } => {
                self.analyse_expr(cond);
                // Constant condition warning.
                match cond.as_ref() {
                    Expr::BoolLit { value: true, span } => {
                        self.warn(*span, "`if true` — else branch is unreachable")
                    }
                    Expr::BoolLit { value: false, span } => {
                        self.warn(*span, "`if false` — then branch is unreachable")
                    }
                    _ => {}
                }
                self.push_scope();
                self.analyse_block(then);
                self.pop_scope();
                if let Some(b) = else_ {
                    self.push_scope();
                    self.analyse_block(b);
                    self.pop_scope();
                }
            }

            Expr::Closure { params, body, .. } => {
                self.push_scope();
                for p in params {
                    self.declare_var(&p.name, p.span, p.mutable);
                }
                self.analyse_expr(body);
                self.pop_scope();
            }

            Expr::Block(b) => {
                self.analyse_block(b);
            }

            Expr::StructLit { name, fields, .. } => {
                self.decls.mark_used(name);
                for (_, e) in fields {
                    self.analyse_expr(e);
                }
            }
            Expr::KronProd { lhs, rhs, .. } | Expr::OuterProd { lhs, rhs, .. } => {
                self.analyse_expr(lhs);
                self.analyse_expr(rhs);
            }
        }
    }

    // ── Mutability gate ────────────────────────────────────────────────────

    fn check_lvalue_mutability(&mut self, expr: &Expr) {
        match expr {
            Expr::Ident { span, name } => {
                if let Some(b) = self.scopes.lookup(name) {
                    if !b.mutable {
                        self.err(
                            *span,
                            format!(
                                "cannot assign to immutable variable `{}`; \
                             declare it with `let mut {}`",
                                name, name
                            ),
                        );
                    }
                }
                // If not found in scope, typeck already flagged it.
            }
            Expr::Field { object, .. } => {
                self.check_lvalue_mutability(object);
            }
            Expr::Index { object, .. } => {
                self.check_lvalue_mutability(object);
            }
            Expr::UnOp {
                op: crate::ast::UnOpKind::Deref,
                expr,
                ..
            } => {
                // Deref of `&mut T` — trust the type checker.
                self.analyse_expr(expr);
            }
            _ => {}
        }
    }

    // =========================================================================
    // §6  PATTERN BINDING
    // =========================================================================

    fn bind_pattern(&mut self, pat: &Pattern, span: Span, mutable: bool) {
        match pat {
            Pattern::Ident {
                name,
                span: ps,
                mutable: pm,
                ..
            } => {
                self.declare_var(name, *ps, mutable || *pm);
            }
            Pattern::Wildcard(_) => {}
            Pattern::Tuple { elems, .. } => {
                for p in elems {
                    self.bind_pattern(p, span, mutable);
                }
            }
            Pattern::Struct { fields, .. } => {
                for (_, maybe_pat) in fields {
                    if let Some(p) = maybe_pat {
                        self.bind_pattern(p, span, mutable);
                    }
                }
            }
            Pattern::Or { arms, .. } => {
                for p in arms {
                    self.bind_pattern(p, span, mutable);
                }
            }
            Pattern::Enum { inner, .. } => {
                for p in inner {
                    self.bind_pattern(p, span, mutable);
                }
            }
            Pattern::Lit(..) | Pattern::Range { .. } => {}
        }
    }

    // =========================================================================
    // §7  POST-PASS CHECKS
    // =========================================================================

    fn check_unused_decls(&mut self) {
        let unused: Vec<(String, Span, DeclKind)> = self
            .decls
            .unused()
            .map(|(name, r)| (name.to_owned(), r.span, r.kind))
            .collect();

        for (name, span, kind) in unused {
            match kind {
                // Systems and components are "used" by the runtime scheduler,
                // not by explicit source-code references — don't warn on them.
                DeclKind::System | DeclKind::Component => {}
                // Everything else gets a warning.
                _ => {
                    self.warn(
                        span,
                        format!(
                            "unused {} `{}`; prefix with `_` to suppress or remove",
                            decl_kind_name(kind),
                            name
                        ),
                    );
                }
            }
        }
    }
}

// =============================================================================
// §8  PURE HELPER FUNCTIONS
// =============================================================================

/// Returns true if `stmt` unconditionally terminates control flow in the
/// current block (i.e., the next statement is unreachable).
fn stmt_terminates(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Return { .. } | Stmt::Break { .. } | Stmt::Continue { .. } => true,
        Stmt::Loop { body, .. } => !block_has_break(body),
        Stmt::While {
            cond: Expr::BoolLit { value: true, .. },
            body,
            ..
        } => !block_has_break(body),
        Stmt::If {
            cond: Expr::BoolLit { value: false, .. },
            ..
        } => false,
        _ => false,
    }
}

/// Returns true if the statement is itself a "dead code" marker (i.e., an item
/// or a declaration that we don't want to double-warn about).
fn is_dead_item(stmt: &Stmt) -> bool {
    matches!(stmt, Stmt::Item(_))
}

/// Returns the primary span of a statement.
fn stmt_span(stmt: &Stmt) -> Span {
    match stmt {
        Stmt::Let { span, .. } => *span,
        Stmt::Expr { span, .. } => *span,
        Stmt::Return { span, .. } => *span,
        Stmt::Break { span, .. } => *span,
        Stmt::Continue { span, .. } => *span,
        Stmt::ForIn { span, .. } => *span,
        Stmt::EntityFor { span, .. } => *span,
        Stmt::While { span: _, cond, .. } => cond.span(),
        Stmt::Loop { body, .. } => body.span,
        Stmt::If { span, .. } => *span,
        Stmt::Match { span, .. } => *span,
        Stmt::Item(i) => i.span(),
        Stmt::ParallelFor(pf) => pf.span,
        Stmt::Spawn(sb) => sb.span,
        Stmt::Sync(sb) => sb.span,
        Stmt::Atomic(ab) => ab.span,
    }
}

/// Shallow check: does this block contain a direct `break` (not inside a nested
/// loop)? This is a conservative approximation for infinite-loop detection.
fn block_has_break(block: &Block) -> bool {
    block.stmts.iter().any(|s| stmt_has_direct_break(s))
}

fn stmt_has_direct_break(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Break { .. } => true,
        Stmt::If { then, else_, .. } => {
            block_has_break(then)
                || else_.as_ref().map_or(false, |e| match e.as_ref() {
                    crate::ast::IfOrBlock::Block(b) => block_has_break(b),
                    crate::ast::IfOrBlock::If(s) => stmt_has_direct_break(s),
                })
        }
        Stmt::Match { arms, .. } => arms.iter().any(|arm| expr_is_break(&arm.body)),
        // Don't descend into nested loops — their breaks belong to them.
        Stmt::Loop { .. }
        | Stmt::While { .. }
        | Stmt::ForIn { .. }
        | Stmt::EntityFor { .. }
        | Stmt::ParallelFor(_) => false,
        _ => false,
    }
}

fn expr_is_break(expr: &Expr) -> bool {
    matches!(expr, Expr::Block(b) if b.stmts.iter().any(|s| matches!(s, Stmt::Break { .. })))
}

/// True when an expression obviously produces a meaningful value that could be
/// unintentionally discarded via a semicolon.
fn expr_produces_value(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::IfExpr { .. }
            | Expr::Block(_)
            | Expr::MatMul { .. }
            | Expr::HadamardMul { .. }
            | Expr::TensorConcat { .. }
            | Expr::BinOp { .. }
            | Expr::Tuple { .. }
            | Expr::Closure { .. }
    )
}

/// True for call expressions and assignments — these are expected to appear
/// as statements without triggering the "value discarded" note.
fn is_call_or_assign(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Call { .. } | Expr::MethodCall { .. } | Expr::Assign { .. }
    )
}

/// Coarse check: does the expression contain an assignment or a call?
/// Used as a side-effect proxy for goal utility expressions.
fn expr_has_side_effects(expr: &Expr) -> bool {
    match expr {
        Expr::Assign { .. } => true,
        Expr::Call { .. } | Expr::MethodCall { .. } => true,
        Expr::BinOp { lhs, rhs, .. } => expr_has_side_effects(lhs) || expr_has_side_effects(rhs),
        Expr::UnOp { expr, .. } => expr_has_side_effects(expr),
        Expr::Field { object, .. } => expr_has_side_effects(object),
        Expr::Index {
            object, indices, ..
        } => expr_has_side_effects(object) || indices.iter().any(expr_has_side_effects),
        Expr::IfExpr {
            cond, then, else_, ..
        } => {
            expr_has_side_effects(cond)
                || block_has_side_effects(then)
                || else_.as_ref().map_or(false, |b| block_has_side_effects(b))
        }
        Expr::Block(b) => block_has_side_effects(b),
        Expr::Closure { .. } => false, // closure creation is pure
        _ => false,
    }
}

fn block_has_side_effects(block: &Block) -> bool {
    block.stmts.iter().any(stmt_has_side_effects)
        || block
            .tail
            .as_ref()
            .map_or(false, |e| expr_has_side_effects(e))
}

fn stmt_has_side_effects(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Expr { expr, .. } => expr_has_side_effects(expr),
        Stmt::Let { init: Some(e), .. } => expr_has_side_effects(e),
        Stmt::Return { value: Some(e), .. } => expr_has_side_effects(e),
        _ => false,
    }
}

/// Returns a short human-readable name for a perception kind, used for
/// equality comparison (avoids `PartialEq` on the enum which doesn't exist).
fn perception_kind_eq(a: &PerceptionKind, b: &PerceptionKind) -> bool {
    match (a, b) {
        (PerceptionKind::Vision, PerceptionKind::Vision) => true,
        (PerceptionKind::Hearing, PerceptionKind::Hearing) => true,
        (PerceptionKind::Omniscient, PerceptionKind::Omniscient) => true,
        (PerceptionKind::Custom(x), PerceptionKind::Custom(y)) => x == y,
        _ => false,
    }
}

fn model_layer_kind_name(layer: &ModelLayer) -> &'static str {
    match layer {
        ModelLayer::Input { .. } => "input",
        ModelLayer::Dense { .. } => "dense",
        ModelLayer::Conv2d { .. } => "conv",
        ModelLayer::Pool { .. } => "pool",
        ModelLayer::Recurrent { .. } => "recurrent",
        ModelLayer::Attention { .. } => "attention",
        ModelLayer::Embed { .. } => "embed",
        ModelLayer::Dropout { .. } => "dropout",
        ModelLayer::Norm { .. } => "norm",
        ModelLayer::Output { .. } => "output",
        ModelLayer::SubModel { .. } => "submodel",
        _ => "other",
    }
}

fn decl_kind_name(kind: DeclKind) -> &'static str {
    match kind {
        DeclKind::Function => "function",
        DeclKind::System => "system",
        DeclKind::Component => "component",
        DeclKind::Struct => "struct",
        DeclKind::Enum => "enum",
        DeclKind::Agent => "agent",
        DeclKind::Model => "model",
        DeclKind::Const => "constant",
        _ => "other",
    }
}

// =============================================================================
// PUBLIC API
// =============================================================================

/// Run the full semantic analysis pass and return the collected diagnostics.
///
/// ```rust,ignore
/// let diags = jules_sema(&program);
/// for d in &diags.items {
///     eprintln!("{}: {}", d.span, d.message);
/// }
/// ```
pub fn jules_sema(program: &Program) -> Diagnostics {
    let mut ctx = SemaCtx::new();
    ctx.analyse(program);
    ctx.diag
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;
    use crate::lexer::Span;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn sp() -> Span {
        Span::dummy()
    }

    fn mk_fn(name: &str, body: Block) -> FnDecl {
        FnDecl {
            span: sp(),
            attrs: vec![],
            name: name.into(),
            generics: vec![],
            params: vec![],
            ret_ty: None,
            body: Some(body),
            is_async: false,
        }
    }

    fn mk_block(stmts: Vec<Stmt>) -> Block {
        Block {
            span: sp(),
            stmts,
            tail: None,
        }
    }

    fn ident(name: &str) -> Expr {
        Expr::Ident {
            span: sp(),
            name: name.into(),
        }
    }

    fn bool_lit(v: bool) -> Expr {
        Expr::BoolLit {
            span: sp(),
            value: v,
        }
    }

    fn int_lit(v: u128) -> Expr {
        Expr::IntLit {
            span: sp(),
            value: v,
        }
    }

    fn ret(e: Option<Expr>) -> Stmt {
        Stmt::Return {
            span: sp(),
            value: e,
        }
    }

    fn let_stmt(name: &str, mutable: bool, init: Option<Expr>) -> Stmt {
        Stmt::Let {
            span: sp(),
            pattern: Pattern::Ident {
                span: sp(),
                name: name.into(),
                mutable,
            },
            ty: None,
            init,
            mutable,
        }
    }

    fn expr_stmt(e: Expr) -> Stmt {
        Stmt::Expr {
            span: sp(),
            expr: e,
            has_semi: true,
        }
    }

    fn run(program: &Program) -> SemaCtx {
        let mut ctx = SemaCtx::new();
        ctx.analyse(program);
        ctx
    }

    // =========================================================================
    // §1  Variable resolution
    // =========================================================================

    #[test]
    fn test_unused_variable_warning() {
        // `let x = 1;` with no subsequent use of x.
        let body = mk_block(vec![let_stmt("x", false, Some(int_lit(42)))]);
        let mut prog = Program::new();
        prog.items.push(Item::Fn(mk_fn("foo", body)));
        let ctx = run(&prog);
        assert!(
            ctx.diag.warning_count() >= 1,
            "expected unused-variable warning"
        );
        assert!(!ctx.diag.has_errors());
    }

    #[test]
    fn test_used_variable_no_warning() {
        // `let x = 1; return x;`
        let body = mk_block(vec![
            let_stmt("x", false, Some(int_lit(1))),
            ret(Some(ident("x"))),
        ]);
        let mut prog = Program::new();
        prog.items.push(Item::Fn(mk_fn("foo", body)));
        let ctx = run(&prog);
        // Should have no unused-variable warning (there may be other notes).
        let unused_warns: Vec<_> = ctx
            .diag
            .items
            .iter()
            .filter(|d| d.severity == Severity::Warning && d.message.contains("unused variable"))
            .collect();
        assert!(
            unused_warns.is_empty(),
            "unexpected unused-var warning: {:?}",
            unused_warns
        );
    }

    #[test]
    fn test_wildcard_variable_no_warning() {
        // `let _x = 1;` — intentionally unused.
        let body = mk_block(vec![let_stmt("_x", false, Some(int_lit(1)))]);
        let mut prog = Program::new();
        prog.items.push(Item::Fn(mk_fn("foo", body)));
        let ctx = run(&prog);
        let unused: Vec<_> = ctx
            .diag
            .items
            .iter()
            .filter(|d| d.message.contains("unused variable"))
            .collect();
        assert!(unused.is_empty());
    }

    #[test]
    fn test_assign_to_immutable_error() {
        // let x = 1; x = 2;  ← error: x is immutable
        let body = mk_block(vec![
            let_stmt("x", false, Some(int_lit(1))),
            expr_stmt(Expr::Assign {
                span: sp(),
                op: AssignOpKind::Assign,
                target: Box::new(ident("x")),
                value: Box::new(int_lit(2)),
            }),
        ]);
        let mut prog = Program::new();
        prog.items.push(Item::Fn(mk_fn("foo", body)));
        let ctx = run(&prog);
        assert!(ctx.diag.has_errors(), "expected immutability error");
    }

    #[test]
    fn test_assign_to_mutable_ok() {
        // let mut x = 1; x = 2;  ← ok
        let body = mk_block(vec![
            let_stmt("x", true, Some(int_lit(1))),
            expr_stmt(Expr::Assign {
                span: sp(),
                op: AssignOpKind::Assign,
                target: Box::new(ident("x")),
                value: Box::new(int_lit(2)),
            }),
        ]);
        let mut prog = Program::new();
        prog.items.push(Item::Fn(mk_fn("foo", body)));
        let ctx = run(&prog);
        assert!(!ctx.diag.has_errors(), "{:?}", ctx.diag.items);
    }

    #[test]
    fn test_shadowing_warning() {
        // let x = 1; { let x = 2; }
        let inner = mk_block(vec![let_stmt("x", false, Some(int_lit(2)))]);
        let body = mk_block(vec![
            let_stmt("x", false, Some(int_lit(1))),
            Stmt::Expr {
                span: sp(),
                expr: Expr::Block(Box::new(inner)),
                has_semi: false,
            },
        ]);
        let mut prog = Program::new();
        prog.items.push(Item::Fn(mk_fn("foo", body)));
        let ctx = run(&prog);
        let shadow: Vec<_> = ctx
            .diag
            .items
            .iter()
            .filter(|d| d.message.contains("shadows"))
            .collect();
        assert!(!shadow.is_empty(), "expected shadowing warning");
    }

    // =========================================================================
    // §2  Unreachable code
    // =========================================================================

    #[test]
    fn test_unreachable_after_return() {
        // return 1; let x = 2;   ← second stmt unreachable
        let body = mk_block(vec![
            ret(Some(int_lit(1))),
            let_stmt("x", false, Some(int_lit(2))),
        ]);
        let mut prog = Program::new();
        prog.items.push(Item::Fn(mk_fn("foo", body)));
        let ctx = run(&prog);
        let unreach: Vec<_> = ctx
            .diag
            .items
            .iter()
            .filter(|d| d.message.contains("unreachable"))
            .collect();
        assert!(!unreach.is_empty(), "expected unreachable-code warning");
    }

    #[test]
    fn test_if_true_dead_else() {
        // if true { 1 } else { 2 }  ← else is dead
        let then_block = mk_block(vec![]);
        let else_block = mk_block(vec![]);
        let body = mk_block(vec![Stmt::If {
            span: sp(),
            cond: bool_lit(true),
            then: then_block,
            else_: Some(Box::new(IfOrBlock::Block(else_block))),
        }]);
        let mut prog = Program::new();
        prog.items.push(Item::Fn(mk_fn("foo", body)));
        let ctx = run(&prog);
        let warns: Vec<_> = ctx
            .diag
            .items
            .iter()
            .filter(|d| {
                d.severity == Severity::Warning
                    && (d.message.contains("unreachable") || d.message.contains("if true"))
            })
            .collect();
        assert!(!warns.is_empty(), "expected dead-else warning");
    }

    #[test]
    fn test_infinite_loop_warning() {
        // loop { /* no break */ }
        let body = mk_block(vec![Stmt::Loop {
            span: sp(),
            body: mk_block(vec![]),
            label: None,
        }]);
        let mut prog = Program::new();
        prog.items.push(Item::Fn(mk_fn("foo", body)));
        let ctx = run(&prog);
        let inf: Vec<_> = ctx
            .diag
            .items
            .iter()
            .filter(|d| d.message.contains("infinite loop"))
            .collect();
        assert!(!inf.is_empty(), "expected infinite-loop warning");
    }

    #[test]
    fn test_loop_with_break_no_warning() {
        let body = mk_block(vec![Stmt::Loop {
            span: sp(),
            body: mk_block(vec![Stmt::Break {
                span: sp(),
                value: None,
                label: None,
            }]),
            label: None,
        }]);
        let mut prog = Program::new();
        prog.items.push(Item::Fn(mk_fn("foo", body)));
        let ctx = run(&prog);
        let inf: Vec<_> = ctx
            .diag
            .items
            .iter()
            .filter(|d| d.message.contains("infinite loop"))
            .collect();
        assert!(inf.is_empty(), "unexpected infinite-loop warning");
    }

    // =========================================================================
    // §3  Break / Continue outside loops
    // =========================================================================

    #[test]
    fn test_break_outside_loop_error() {
        let body = mk_block(vec![Stmt::Break {
            span: sp(),
            value: None,
            label: None,
        }]);
        let mut prog = Program::new();
        prog.items.push(Item::Fn(mk_fn("foo", body)));
        let ctx = run(&prog);
        assert!(
            ctx.diag.has_errors(),
            "expected error for break outside loop"
        );
    }

    #[test]
    fn test_continue_outside_loop_error() {
        let body = mk_block(vec![Stmt::Continue {
            span: sp(),
            label: None,
        }]);
        let mut prog = Program::new();
        prog.items.push(Item::Fn(mk_fn("foo", body)));
        let ctx = run(&prog);
        assert!(
            ctx.diag.has_errors(),
            "expected error for continue outside loop"
        );
    }

    #[test]
    fn test_break_inside_loop_ok() {
        let inner = mk_block(vec![Stmt::Break {
            span: sp(),
            value: None,
            label: None,
        }]);
        let body = mk_block(vec![Stmt::Loop {
            span: sp(),
            body: inner,
            label: None,
        }]);
        let mut prog = Program::new();
        prog.items.push(Item::Fn(mk_fn("foo", body)));
        let ctx = run(&prog);
        let errs: Vec<_> = ctx
            .diag
            .items
            .iter()
            .filter(|d| d.is_fatal() && d.message.contains("break"))
            .collect();
        assert!(errs.is_empty(), "unexpected break error: {:?}", errs);
    }

    #[test]
    fn test_return_outside_function_error() {
        // Construct a scenario where Return appears at program level
        // (simulated by calling analyse_stmt directly).
        let mut ctx = SemaCtx::new();
        // No function frame pushed.
        ctx.analyse_stmt(&Stmt::Return {
            span: sp(),
            value: None,
        });
        assert!(ctx.diag.has_errors());
    }

    // =========================================================================
    // §4  ECS / system loop rules
    // =========================================================================

    #[test]
    fn test_duplicate_component_in_with_error() {
        let q = EntityQuery {
            span: sp(),
            with: vec!["Position".into(), "Position".into()],
            without: vec![],
            filter: None,
        };
        let mut ctx = SemaCtx::new();
        ctx.components.insert("Position".into());
        ctx.validate_entity_query(&q, sp());
        assert!(
            ctx.diag.has_errors(),
            "expected error for duplicate in with(…)"
        );
    }

    #[test]
    fn test_component_in_both_with_and_without_error() {
        let q = EntityQuery {
            span: sp(),
            with: vec!["Velocity".into()],
            without: vec!["Velocity".into()],
            filter: None,
        };
        let mut ctx = SemaCtx::new();
        ctx.components.insert("Velocity".into());
        ctx.validate_entity_query(&q, sp());
        assert!(
            ctx.diag.has_errors(),
            "expected error for component in both clauses"
        );
    }

    #[test]
    fn test_undeclared_component_in_query_error() {
        let q = EntityQuery {
            span: sp(),
            with: vec!["Ghost".into()], // not declared
            without: vec![],
            filter: None,
        };
        let mut ctx = SemaCtx::new();
        // "Ghost" is NOT in ctx.components
        ctx.validate_entity_query(&q, sp());
        assert!(
            ctx.diag.has_errors(),
            "expected error for undeclared component"
        );
    }

    #[test]
    fn test_valid_entity_query_no_errors() {
        let q = EntityQuery {
            span: sp(),
            with: vec!["Position".into(), "Velocity".into()],
            without: vec!["Dead".into()],
            filter: None,
        };
        let mut ctx = SemaCtx::new();
        ctx.components.insert("Position".into());
        ctx.components.insert("Velocity".into());
        ctx.components.insert("Dead".into());
        ctx.validate_entity_query(&q, sp());
        assert!(!ctx.diag.has_errors(), "{:?}", ctx.diag.items);
    }

    #[test]
    fn test_system_aliasing_detected() {
        // System A writes Position; System B reads Position → conflict.
        let mut set_a = ComponentSet::default();
        set_a.add("Position", AccessMode::Write);
        let mut set_b = ComponentSet::default();
        set_b.add("Position", AccessMode::Read);

        let mut ctx = SemaCtx::new();
        ctx.system_sets.push(("UpdatePos".into(), sp(), set_a));
        ctx.system_sets.push(("RenderPos".into(), sp(), set_b));
        ctx.check_system_aliasing();

        let warns: Vec<_> = ctx
            .diag
            .items
            .iter()
            .filter(|d| d.severity == Severity::Warning && d.message.contains("aliasing"))
            .collect();
        assert!(!warns.is_empty(), "expected aliasing warning");
    }

    #[test]
    fn test_system_no_aliasing_different_components() {
        let mut set_a = ComponentSet::default();
        set_a.add("Velocity", AccessMode::Write);
        let mut set_b = ComponentSet::default();
        set_b.add("Health", AccessMode::Read);

        let mut ctx = SemaCtx::new();
        ctx.system_sets.push(("MoveSystem".into(), sp(), set_a));
        ctx.system_sets.push(("HealthSystem".into(), sp(), set_b));
        ctx.check_system_aliasing();

        assert!(!ctx.diag.has_errors());
        let warns: Vec<_> = ctx
            .diag
            .items
            .iter()
            .filter(|d| d.message.contains("aliasing"))
            .collect();
        assert!(warns.is_empty(), "unexpected aliasing warning");
    }

    // =========================================================================
    // §5  Agent / behaviour correctness
    // =========================================================================

    #[test]
    fn test_agent_no_behaviours_warning() {
        let a = AgentDecl {
            span: sp(),
            attrs: vec![],
            name: "EmptyAgent".into(),
            architecture: AgentArchitecture::Utility,
            perceptions: vec![],
            memory: None,
            learning: None,
            behaviors: vec![], // ← empty
            goals: vec![],
            fields: vec![],
        };
        let mut ctx = SemaCtx::new();
        ctx.analyse_agent(&a);
        let warns: Vec<_> = ctx
            .diag
            .items
            .iter()
            .filter(|d| d.severity == Severity::Warning && d.message.contains("no behaviours"))
            .collect();
        assert!(!warns.is_empty(), "expected no-behaviours warning");
    }

    #[test]
    fn test_agent_duplicate_priority_error() {
        let rule = |name: &str, prio: u32| BehaviorRule {
            span: sp(),
            name: name.into(),
            priority: BehaviorPriority(prio),
            params: vec![],
            body: mk_block(vec![]),
        };
        let a = AgentDecl {
            span: sp(),
            attrs: vec![],
            name: "Clash".into(),
            architecture: AgentArchitecture::Utility,
            perceptions: vec![],
            memory: None,
            learning: None,
            behaviors: vec![rule("Flee", 100), rule("Attack", 100)], // same prio
            goals: vec![],
            fields: vec![],
        };
        let mut ctx = SemaCtx::new();
        ctx.analyse_agent(&a);
        assert!(ctx.diag.has_errors(), "expected duplicate-priority error");
    }

    #[test]
    fn test_agent_unique_priorities_ok() {
        let rule = |name: &str, prio: u32| BehaviorRule {
            span: sp(),
            name: name.into(),
            priority: BehaviorPriority(prio),
            params: vec![],
            body: mk_block(vec![]),
        };
        let a = AgentDecl {
            span: sp(),
            attrs: vec![],
            name: "Ordered".into(),
            architecture: AgentArchitecture::Utility,
            perceptions: vec![],
            memory: None,
            learning: None,
            behaviors: vec![rule("Flee", 100), rule("Patrol", 10)],
            goals: vec![],
            fields: vec![],
        };
        let mut ctx = SemaCtx::new();
        ctx.analyse_agent(&a);
        assert!(!ctx.diag.has_errors(), "{:?}", ctx.diag.items);
    }

    #[test]
    fn test_agent_duplicate_perception_error() {
        let vision = |range| PerceptionSpec {
            span: sp(),
            kind: PerceptionKind::Vision,
            range: Some(range),
            fov: None,
            tag: None,
        };
        let a = AgentDecl {
            span: sp(),
            attrs: vec![],
            name: "DualVision".into(),
            architecture: AgentArchitecture::Utility,
            perceptions: vec![vision(40.0), vision(20.0)], // duplicate Vision
            memory: None,
            learning: None,
            behaviors: vec![BehaviorRule {
                span: sp(),
                name: "act".into(),
                priority: BehaviorPriority(1),
                params: vec![],
                body: mk_block(vec![]),
            }],
            goals: vec![],
            fields: vec![],
        };
        let mut ctx = SemaCtx::new();
        ctx.analyse_agent(&a);
        assert!(ctx.diag.has_errors(), "expected duplicate-perception error");
    }

    // =========================================================================
    // §6  Model correctness
    // =========================================================================

    #[test]
    fn test_model_missing_input_error() {
        let m = ModelDecl {
            span: sp(),
            attrs: vec![],
            name: "NoInput".into(),
            layers: vec![ModelLayer::Output {
                span: sp(),
                units: 10,
                activation: Activation::Softmax,
            }],
            device: ModelDevice::Auto,
            optimizer: None,
        };
        let mut ctx = SemaCtx::new();
        ctx.analyse_model(&m);
        assert!(ctx.diag.has_errors());
    }

    #[test]
    fn test_model_missing_output_error() {
        let m = ModelDecl {
            span: sp(),
            attrs: vec![],
            name: "NoOutput".into(),
            layers: vec![ModelLayer::Input {
                span: sp(),
                size: 128,
            }],
            device: ModelDevice::Auto,
            optimizer: None,
        };
        let mut ctx = SemaCtx::new();
        ctx.analyse_model(&m);
        assert!(ctx.diag.has_errors());
    }

    #[test]
    fn test_model_duplicate_input_error() {
        let m = ModelDecl {
            span: sp(),
            attrs: vec![],
            name: "TwoInputs".into(),
            layers: vec![
                ModelLayer::Input {
                    span: sp(),
                    size: 64,
                },
                ModelLayer::Input {
                    span: sp(),
                    size: 128,
                },
                ModelLayer::Output {
                    span: sp(),
                    units: 10,
                    activation: Activation::Softmax,
                },
            ],
            device: ModelDevice::Auto,
            optimizer: None,
        };
        let mut ctx = SemaCtx::new();
        ctx.analyse_model(&m);
        assert!(ctx.diag.has_errors());
    }

    #[test]
    fn test_model_well_formed_ok() {
        let m = ModelDecl {
            span: sp(),
            attrs: vec![],
            name: "Good".into(),
            layers: vec![
                ModelLayer::Input {
                    span: sp(),
                    size: 128,
                },
                ModelLayer::Dense {
                    span: sp(),
                    units: 256,
                    activation: Activation::Relu,
                    bias: true,
                },
                ModelLayer::Output {
                    span: sp(),
                    units: 10,
                    activation: Activation::Softmax,
                },
            ],
            device: ModelDevice::Auto,
            optimizer: None,
        };
        let mut ctx = SemaCtx::new();
        ctx.analyse_model(&m);
        assert!(!ctx.diag.has_errors(), "{:?}", ctx.diag.items);
    }

    // =========================================================================
    // §7  Train correctness
    // =========================================================================

    #[test]
    fn test_train_duplicate_signal_name_error() {
        let sig = |name: &str| SignalSpec {
            span: sp(),
            is_reward: true,
            name: name.into(),
            weight: 1.0,
            expr: None,
        };
        let t = TrainDecl {
            span: sp(),
            attrs: vec![],
            agent: "Bot".into(),
            world: "Arena".into(),
            signals: vec![sig("survive"), sig("survive")], // duplicate
            episode: None,
            model: None,
            optimizer: None,
            hyper: vec![],
            algorithm: None,
            value_model: None,
        };
        let mut ctx = SemaCtx::new();
        ctx.analyse_train(&t);
        assert!(ctx.diag.has_errors(), "expected duplicate signal error");
    }

    #[test]
    fn test_train_negative_signal_weight_error() {
        let sig = SignalSpec {
            span: sp(),
            is_reward: true,
            name: "bonus".into(),
            weight: -0.5,
            expr: None,
        };
        let t = TrainDecl {
            span: sp(),
            attrs: vec![],
            agent: "Bot".into(),
            world: "Arena".into(),
            signals: vec![sig],
            episode: None,
            model: None,
            optimizer: None,
            hyper: vec![],
            algorithm: None,
            value_model: None,
        };
        let mut ctx = SemaCtx::new();
        ctx.analyse_train(&t);
        assert!(ctx.diag.has_errors(), "expected negative-weight error");
    }

    #[test]
    fn test_train_duplicate_hyper_key_error() {
        let t = TrainDecl {
            span: sp(),
            attrs: vec![],
            agent: "Bot".into(),
            world: "Arena".into(),
            signals: vec![],
            episode: None,
            model: None,
            optimizer: None,
            hyper: vec![
                (
                    "gamma".into(),
                    Expr::FloatLit {
                        span: sp(),
                        value: 0.99,
                    },
                ),
                (
                    "gamma".into(),
                    Expr::FloatLit {
                        span: sp(),
                        value: 0.95,
                    },
                ),
            ],
            algorithm: None,
            value_model: None,
        };
        let mut ctx = SemaCtx::new();
        ctx.analyse_train(&t);
        assert!(ctx.diag.has_errors(), "expected duplicate hyper key error");
    }

    #[test]
    fn test_train_zero_max_steps_error() {
        let ep = EpisodeSpec {
            span: sp(),
            max_steps: Some(0),
            max_seconds: None,
            done_condition: None,
            num_envs: Some(8),
        };
        let t = TrainDecl {
            span: sp(),
            attrs: vec![],
            agent: "Bot".into(),
            world: "Arena".into(),
            signals: vec![],
            episode: Some(ep),
            model: None,
            optimizer: None,
            hyper: vec![],
            algorithm: None,
            value_model: None,
        };
        let mut ctx = SemaCtx::new();
        ctx.analyse_train(&t);
        assert!(ctx.diag.has_errors(), "expected max_steps=0 error");
    }

    #[test]
    fn test_train_zero_num_envs_error() {
        let ep = EpisodeSpec {
            span: sp(),
            max_steps: Some(1000),
            max_seconds: None,
            done_condition: None,
            num_envs: Some(0), // invalid
        };
        let t = TrainDecl {
            span: sp(),
            attrs: vec![],
            agent: "Bot".into(),
            world: "Arena".into(),
            signals: vec![],
            episode: Some(ep),
            model: None,
            optimizer: None,
            hyper: vec![],
            algorithm: None,
            value_model: None,
        };
        let mut ctx = SemaCtx::new();
        ctx.analyse_train(&t);
        assert!(ctx.diag.has_errors(), "expected num_envs=0 error");
    }

    // =========================================================================
    // §8  Diagnostic helpers
    // =========================================================================

    #[test]
    fn test_diagnostic_with_label() {
        let d = Diagnostic::error(sp(), "boom").with_label(sp(), "defined here");
        assert_eq!(d.labels.len(), 1);
        assert!(d.is_fatal());
    }

    #[test]
    fn test_diagnostics_counts() {
        let mut diags = Diagnostics::default();
        diags.error(sp(), "e1");
        diags.error(sp(), "e2");
        diags.warning(sp(), "w1");
        assert_eq!(diags.error_count(), 2);
        assert_eq!(diags.warning_count(), 1);
        assert!(diags.has_errors());
    }

    // =========================================================================
    // §9  Scope stack
    // =========================================================================

    #[test]
    fn test_scope_stack_basic() {
        let mut ss = ScopeStack::default();
        ss.push();
        ss.declare("x", Binding::new(sp(), false, false));
        assert!(ss.mark_used("x"));
        assert!(!ss.mark_used("y"));
        let unused = ss.pop();
        // x was marked used so should not appear as unused
        assert!(unused.is_empty(), "x was used, should not be unused");
    }

    #[test]
    fn test_scope_stack_unused_binding() {
        let mut ss = ScopeStack::default();
        ss.push();
        ss.declare("z", Binding::new(sp(), false, false));
        let unused = ss.pop(); // z never marked used
        assert_eq!(unused.len(), 1);
        assert_eq!(unused[0].0, "z");
    }

    #[test]
    fn test_scope_stack_outer_name_detection() {
        let mut ss = ScopeStack::default();
        ss.push();
        ss.declare("outer", Binding::new(sp(), false, false));
        ss.push();
        ss.declare("inner", Binding::new(sp(), false, false));
        assert!(ss.is_outer_name("outer"));
        assert!(!ss.is_outer_name("inner"));
        ss.pop();
        ss.pop();
    }

    // =========================================================================
    // §10  Control-flow stack
    // =========================================================================

    #[test]
    fn test_cf_stack_in_loop() {
        let mut cf = CfStack::default();
        assert!(!cf.in_loop());
        cf.push(CfFrame::Loop { label: None });
        assert!(cf.in_loop());
        cf.pop();
        assert!(!cf.in_loop());
    }

    #[test]
    fn test_cf_stack_labelled_loop() {
        let mut cf = CfStack::default();
        cf.push(CfFrame::Loop {
            label: Some("outer".into()),
        });
        assert!(cf.has_label("outer"));
        assert!(!cf.has_label("inner"));
        cf.pop();
    }

    #[test]
    fn test_cf_stack_async_fn() {
        let mut cf = CfStack::default();
        cf.push(CfFrame::Function {
            is_async: true,
            name: "fetch".into(),
        });
        assert!(cf.in_async_fn());
        assert!(cf.in_function());
        cf.pop();
        assert!(!cf.in_async_fn());
    }

    // =========================================================================
    // §11  Component graph aliasing
    // =========================================================================

    #[test]
    fn test_component_set_no_conflict() {
        let mut a = ComponentSet::default();
        a.add("Health", AccessMode::Read);
        let mut b = ComponentSet::default();
        b.add("Velocity", AccessMode::Write);
        assert!(a.conflicts_with(&b).is_none());
    }

    #[test]
    fn test_component_set_write_read_conflict() {
        let mut a = ComponentSet::default();
        a.add("Position", AccessMode::Write);
        let mut b = ComponentSet::default();
        b.add("Position", AccessMode::Read);
        assert!(a.conflicts_with(&b).is_some());
    }

    #[test]
    fn test_component_set_write_write_conflict() {
        let mut a = ComponentSet::default();
        a.add("Transform", AccessMode::Write);
        let mut b = ComponentSet::default();
        b.add("Transform", AccessMode::Write);
        assert!(a.conflicts_with(&b).is_some());
    }

    #[test]
    fn test_component_set_read_read_no_conflict() {
        let mut a = ComponentSet::default();
        a.add("Mass", AccessMode::Read);
        let mut b = ComponentSet::default();
        b.add("Mass", AccessMode::Read);
        // Read-read is always safe.
        assert!(a.conflicts_with(&b).is_none());
    }
}
