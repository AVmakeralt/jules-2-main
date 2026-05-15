// =============================================================================
// jules/src/compiler/diagnostic.rs
//
// World-class diagnostic architecture for the Jules compiler.
//
// Design principles:
//   1. Rich labels (primary + secondary with messages)
//   2. Fix-it edits with confidence levels
//   3. Error causality chains (root cause linking)
//   4. Diagnostic dataflow tracking
//   5. "Why" explanations (teaching mode)
//   6. Recovery-aware parser diagnostics
//   7. Constraint-based type errors (constraint graph)
//   8. Teaching-mode diagnostics
//   9. Hint confidence ranking (MachineCertain, Likely, Possible)
//  10. Error explanation database (for `jules explain E4001`)
//
// Code ranges:
//   E0xxx  Lexer           E1xxx  Parser          E2xxx  Type Checker
//   E3xxx  Semantic        E4xxx  Borrow/Ownership E5xxx  Effect System
//   E6xxx  Ownership/Region E7xxx  IR/Compilation E9xxx  Runtime
// =============================================================================

use crate::compiler::lexer::Span;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

// =============================================================================
// §1  CORE TYPES
// =============================================================================

/// Severity levels for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Severity {
    /// Internal compiler error (bug).
    Ice,
    /// Error: compilation cannot succeed.
    Error,
    /// Warning: potential problem.
    Warning,
    /// Note: additional context.
    Note,
    /// Help: fix-it suggestion.
    Help,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Ice => write!(f, "internal compiler error"),
            Severity::Error => write!(f, "error"),
            Severity::Warning => write!(f, "warning"),
            Severity::Note => write!(f, "note"),
            Severity::Help => write!(f, "help"),
        }
    }
}

// ─── Compiler Phase ───────────────────────────────────────────────────────────

/// Which compiler phase produced this diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    Lexer,
    Parser,
    NameResolution,
    TypeCheck,
    Ownership,
    Effects,
    BorrowCheck,
    SemanticAnalysis,
    ShaderValidation,
    MLGraphValidation,
    ECSScheduling,
    Optimizer,
    BackendLowering,
    Runtime,
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Phase::Lexer => write!(f, "lexer"),
            Phase::Parser => write!(f, "parser"),
            Phase::NameResolution => write!(f, "name-resolution"),
            Phase::TypeCheck => write!(f, "type-check"),
            Phase::Ownership => write!(f, "ownership"),
            Phase::Effects => write!(f, "effects"),
            Phase::BorrowCheck => write!(f, "borrow-check"),
            Phase::SemanticAnalysis => write!(f, "semantic-analysis"),
            Phase::ShaderValidation => write!(f, "shader-validation"),
            Phase::MLGraphValidation => write!(f, "ml-graph-validation"),
            Phase::ECSScheduling => write!(f, "ecs-scheduling"),
            Phase::Optimizer => write!(f, "optimizer"),
            Phase::BackendLowering => write!(f, "backend"),
            Phase::Runtime => write!(f, "runtime"),
        }
    }
}

// ─── Diagnostic Code ──────────────────────────────────────────────────────────

/// Structured error code with category.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DiagCode {
    /// Category: E=error, W=warning, N=note, I=ICE
    pub category: char,
    /// Numeric code within category.
    pub number: u16,
}

impl DiagCode {
    pub fn error(n: u16) -> Self {
        DiagCode { category: 'E', number: n }
    }
    pub fn warning(n: u16) -> Self {
        DiagCode { category: 'W', number: n }
    }
    pub fn note(n: u16) -> Self {
        DiagCode { category: 'N', number: n }
    }
    pub fn ice(n: u16) -> Self {
        DiagCode { category: 'I', number: n }
    }
}

impl fmt::Display for DiagCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{:04}", self.category, self.number)
    }
}

// ─── Unique Diagnostic ID ─────────────────────────────────────────────────────

/// Global counter for assigning unique diagnostic IDs.
static NEXT_DIAG_ID: AtomicU64 = AtomicU64::new(1);

/// Unique diagnostic ID for causality tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DiagnosticId(pub u64);

impl DiagnosticId {
    /// Allocate a fresh, unique diagnostic ID.
    pub fn fresh() -> Self {
        DiagnosticId(NEXT_DIAG_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for DiagnosticId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "diag#{}", self.0)
    }
}

// ─── Label Style ──────────────────────────────────────────────────────────────

/// Whether a label is primary or secondary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LabelStyle {
    /// The main error location (underlined with ^^^).
    Primary,
    /// Related location (underlined with ---).
    Secondary,
}

// ─── Rich Label ───────────────────────────────────────────────────────────────

/// A rich label with primary/secondary distinction.
#[derive(Debug, Clone)]
pub struct Label {
    pub span: Span,
    pub message: String,
    pub style: LabelStyle,
}

impl Label {
    /// Create a primary label.
    pub fn primary(span: Span, msg: impl Into<String>) -> Self {
        Label { span, message: msg.into(), style: LabelStyle::Primary }
    }
    /// Create a secondary label.
    pub fn secondary(span: Span, msg: impl Into<String>) -> Self {
        Label { span, message: msg.into(), style: LabelStyle::Secondary }
    }
}

// ─── Hint Confidence ──────────────────────────────────────────────────────────

/// How confident the compiler is about a fix-it suggestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HintConfidence {
    /// Compiler is 100% sure this fix is correct.
    MachineCertain,
    /// Strong probability (>80%).
    Likely,
    /// May be helpful but uncertain.
    Possible,
}

impl fmt::Display for HintConfidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HintConfidence::MachineCertain => write!(f, "certain"),
            HintConfidence::Likely => write!(f, "high confidence"),
            HintConfidence::Possible => write!(f, "possible"),
        }
    }
}

// ─── Fix-It ───────────────────────────────────────────────────────────────────

/// A suggested code replacement with confidence level.
#[derive(Debug, Clone)]
pub struct FixIt {
    /// The span to replace.
    pub span: Span,
    /// The replacement text.
    pub replacement: String,
    /// Human-readable description of the fix.
    pub description: String,
    /// How confident the compiler is about this fix.
    pub confidence: HintConfidence,
}

impl FixIt {
    /// Create a new fix-it suggestion.
    pub fn new(
        span: Span,
        replacement: impl Into<String>,
        description: impl Into<String>,
        confidence: HintConfidence,
    ) -> Self {
        FixIt {
            span,
            replacement: replacement.into(),
            description: description.into(),
            confidence,
        }
    }
}

// ─── Causal Relation ──────────────────────────────────────────────────────────

/// The relationship between a diagnostic and its cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausalRelation {
    /// This is THE root cause.
    RootCause,
    /// Follows from the root cause.
    Consequence,
    /// Loosely related.
    Related,
    /// Would have been emitted but was suppressed.
    Suppressed,
}

// ─── Causality Link ───────────────────────────────────────────────────────────

/// A link in the error causality chain.
#[derive(Debug, Clone)]
pub struct CausalityLink {
    pub diag_id: DiagnosticId,
    pub relation: CausalRelation,
}

// ─── Dataflow Info ────────────────────────────────────────────────────────────

/// Diagnostic dataflow tracking information.
#[derive(Debug, Clone)]
pub struct DataflowInfo {
    /// Where the value was originally defined.
    pub defined_at: Option<Span>,
    /// Where the value was last assigned/moved.
    pub last_assigned_at: Option<Span>,
    /// Where the value was moved/borrowed.
    pub moved_or_borrowed_at: Option<Span>,
    /// Description of the dataflow path.
    pub path_description: String,
}

// ─── Type Constraint Info ─────────────────────────────────────────────────────

/// Constraint-based type error information.
#[derive(Debug, Clone)]
pub struct TypeConstraintInfo {
    /// The constraint chain: what required this type.
    pub constraints: Vec<TypeConstraint>,
}

/// A single type constraint in the chain.
#[derive(Debug, Clone)]
pub struct TypeConstraint {
    pub span: Span,
    pub expected: String,
    pub actual: String,
    /// e.g. "parameter requires i64", "inferred as i32"
    pub reason: String,
}

// ─── Recovery Context ─────────────────────────────────────────────────────────

/// The kind of mistake a recovery suggestion addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecoveryContext {
    /// `x = 10` → suggest `x := 10`
    ImmutableBinding,
    /// Missing a type annotation on a parameter or variable.
    MissingTypeAnnotation,
    /// Trying to use a value that has been moved.
    OwnershipViolation,
    /// An effect was performed without the proper annotation.
    EffectViolation,
    /// A function performs side effects but lacks an effect annotation.
    MissingEffectAnnotation,
    /// Using a name that hasn't been declared.
    UndeclaredVariable,
    /// A GPU/SIMD operation requires a capability not declared on the function.
    MissingCapability,
    /// A shader definition is missing a required stage.
    MissingShaderStage,
    /// An ML function has conflicting graph modes.
    GraphModeConflict,
    /// An ECS system accesses components without declaring reads/writes.
    MissingSystemAccess,
    /// A type doesn't implement a required trait.
    TraitNotImplemented,
    /// A comptime expression failed to evaluate at compile time.
    ComptimeEvalFailure,
    /// Accessing a private item from outside its module.
    VisibilityViolation,
    /// ECS system scheduling conflict — two systems write the same component.
    SystemSchedulingConflict,
}

// ─── Recovery Suggestion ──────────────────────────────────────────────────────

/// A recovery suggestion for common parser mistakes.
#[derive(Debug, Clone)]
pub struct RecoverySuggestion {
    pub mistake: String,
    pub suggestion: String,
    pub confidence: HintConfidence,
    pub context: RecoveryContext,
}

impl RecoverySuggestion {
    /// Create a new recovery suggestion.
    pub fn new(
        mistake: impl Into<String>,
        suggestion: impl Into<String>,
        confidence: HintConfidence,
        context: RecoveryContext,
    ) -> Self {
        RecoverySuggestion {
            mistake: mistake.into(),
            suggestion: suggestion.into(),
            confidence,
            context,
        }
    }

    /// Look up common recovery suggestions for a given recovery context.
    pub fn suggestions_for(context: RecoveryContext) -> Vec<RecoverySuggestion> {
        match context {
            RecoveryContext::ImmutableBinding => vec![
                RecoverySuggestion::new(
                    "assignment to immutable binding",
                    "use `:=` for mutation assignment, or declare with `let mut`",
                    HintConfidence::MachineCertain,
                    RecoveryContext::ImmutableBinding,
                ),
            ],
            RecoveryContext::MissingTypeAnnotation => vec![
                RecoverySuggestion::new(
                    "parameter missing type annotation",
                    "add an explicit type like `x: f32` or use `_` as a wildcard",
                    HintConfidence::Likely,
                    RecoveryContext::MissingTypeAnnotation,
                ),
            ],
            RecoveryContext::OwnershipViolation => vec![
                RecoverySuggestion::new(
                    "use of moved value",
                    "borrow with `&name` instead of moving, or `.clone()` to make a copy",
                    HintConfidence::Likely,
                    RecoveryContext::OwnershipViolation,
                ),
            ],
            RecoveryContext::EffectViolation => vec![
                RecoverySuggestion::new(
                    "effect performed outside declared effect block",
                    "wrap the operation in an `effect io { ... }` block",
                    HintConfidence::Likely,
                    RecoveryContext::EffectViolation,
                ),
            ],
            RecoveryContext::MissingEffectAnnotation => vec![
                RecoverySuggestion::new(
                    "function performs side effects without effect annotation",
                    "add `effect io` to the function signature",
                    HintConfidence::Likely,
                    RecoveryContext::MissingEffectAnnotation,
                ),
            ],
            RecoveryContext::UndeclaredVariable => vec![
                RecoverySuggestion::new(
                    "use of undeclared variable",
                    "declare it with `let` before use, or check for a spelling mistake",
                    HintConfidence::Possible,
                    RecoveryContext::UndeclaredVariable,
                ),
            ],
            RecoveryContext::MissingCapability => vec![
                RecoverySuggestion::new(
                    "operation requires a capability not declared on the function",
                    "add `effect(gpu)` or `effect(simd)` to the function signature",
                    HintConfidence::Likely,
                    RecoveryContext::MissingCapability,
                ),
            ],
            RecoveryContext::MissingShaderStage => vec![
                RecoverySuggestion::new(
                    "shader definition is missing a required stage",
                    "add at least a `vertex` and `fragment` stage to the shader definition",
                    HintConfidence::MachineCertain,
                    RecoveryContext::MissingShaderStage,
                ),
            ],
            RecoveryContext::GraphModeConflict => vec![
                RecoverySuggestion::new(
                    "ML function has conflicting graph modes",
                    "use `@eager`, `@trace`, or `@graph` to declare the execution mode consistently",
                    HintConfidence::Likely,
                    RecoveryContext::GraphModeConflict,
                ),
            ],
            RecoveryContext::MissingSystemAccess => vec![
                RecoverySuggestion::new(
                    "ECS system accesses components without declaring reads/writes",
                    "add `reads(ComponentName)` or `writes(ComponentName)` to the system declaration",
                    HintConfidence::Likely,
                    RecoveryContext::MissingSystemAccess,
                ),
            ],
            RecoveryContext::TraitNotImplemented => vec![
                RecoverySuggestion::new(
                    "type does not implement a required trait",
                    "add an `impl TraitName for TypeName` block with the required method(s)",
                    HintConfidence::Likely,
                    RecoveryContext::TraitNotImplemented,
                ),
            ],
            RecoveryContext::ComptimeEvalFailure => vec![
                RecoverySuggestion::new(
                    "comptime expression could not be evaluated at compile time",
                    "ensure the expression only uses compile-time known values and pure functions",
                    HintConfidence::Possible,
                    RecoveryContext::ComptimeEvalFailure,
                ),
            ],
            RecoveryContext::VisibilityViolation => vec![
                RecoverySuggestion::new(
                    "accessing a private item from outside its module",
                    "add `pub` or `pub(crate)` to the item declaration, or access it from within its module",
                    HintConfidence::Likely,
                    RecoveryContext::VisibilityViolation,
                ),
            ],
            RecoveryContext::SystemSchedulingConflict => vec![
                RecoverySuggestion::new(
                    "two systems write to the same component without ordering",
                    "add `before(OtherSystem)` or `after(OtherSystem)` to establish a deterministic order",
                    HintConfidence::Likely,
                    RecoveryContext::SystemSchedulingConflict,
                ),
            ],
        }
    }
}

// =============================================================================
// §2  THE DIAGNOSTIC STRUCT
// =============================================================================

/// A structured diagnostic with full provenance and fix suggestions.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Unique ID for causality tracking.
    pub id: DiagnosticId,
    /// Error severity.
    pub severity: Severity,
    /// Structured error code (e.g., E4001, W0042).
    pub code: DiagCode,
    /// Which compiler phase produced this.
    pub phase: Phase,
    /// Primary error message.
    pub message: String,
    /// Primary source span.
    pub span: Span,
    /// Rich labels (primary + secondary).
    pub labels: Vec<Label>,
    /// Fix-it suggestions with confidence.
    pub fixits: Vec<FixIt>,
    /// Cause chain: what led to this error.
    pub cause_chain: Vec<CausalityLink>,
    /// "Why does this exist?" explanation (teaching mode).
    pub why_explanation: Option<String>,
    /// Teaching note (longer-form explanation for learning).
    pub teaching_note: Option<String>,
    /// If true, suppress any further errors that depend on this one.
    pub suppress_cascade: bool,
    /// The symbol or type that caused the error (for cascade suppression).
    pub suppressed_symbol: Option<String>,
    /// Dataflow tracking information.
    pub dataflow: Option<DataflowInfo>,
    /// Constraint-based type error information.
    pub type_constraint: Option<TypeConstraintInfo>,
}

impl Diagnostic {
    /// Create a new error diagnostic.
    pub fn error(code: u16, phase: Phase, span: Span, msg: impl Into<String>) -> Self {
        Diagnostic {
            id: DiagnosticId::fresh(),
            severity: Severity::Error,
            code: DiagCode::error(code),
            phase,
            message: msg.into(),
            span,
            labels: vec![],
            fixits: vec![],
            cause_chain: vec![],
            why_explanation: None,
            teaching_note: None,
            suppress_cascade: false,
            suppressed_symbol: None,
            dataflow: None,
            type_constraint: None,
        }
    }

    /// Create a new warning diagnostic.
    pub fn warning(code: u16, phase: Phase, span: Span, msg: impl Into<String>) -> Self {
        Diagnostic {
            id: DiagnosticId::fresh(),
            severity: Severity::Warning,
            code: DiagCode::warning(code),
            phase,
            message: msg.into(),
            span,
            labels: vec![],
            fixits: vec![],
            cause_chain: vec![],
            why_explanation: None,
            teaching_note: None,
            suppress_cascade: false,
            suppressed_symbol: None,
            dataflow: None,
            type_constraint: None,
        }
    }

    /// Create a new note diagnostic.
    pub fn note(code: u16, phase: Phase, span: Span, msg: impl Into<String>) -> Self {
        Diagnostic {
            id: DiagnosticId::fresh(),
            severity: Severity::Note,
            code: DiagCode::note(code),
            phase,
            message: msg.into(),
            span,
            labels: vec![],
            fixits: vec![],
            cause_chain: vec![],
            why_explanation: None,
            teaching_note: None,
            suppress_cascade: false,
            suppressed_symbol: None,
            dataflow: None,
            type_constraint: None,
        }
    }

    /// Create a new ICE (internal compiler error) diagnostic.
    pub fn ice(code: u16, phase: Phase, span: Span, msg: impl Into<String>) -> Self {
        Diagnostic {
            id: DiagnosticId::fresh(),
            severity: Severity::Ice,
            code: DiagCode::ice(code),
            phase,
            message: msg.into(),
            span,
            labels: vec![],
            fixits: vec![],
            cause_chain: vec![],
            why_explanation: None,
            teaching_note: None,
            suppress_cascade: false,
            suppressed_symbol: None,
            dataflow: None,
            type_constraint: None,
        }
    }

    // ── Builder methods (return Self for chaining) ─────────────────────────────

    /// Add a primary label (main error location).
    pub fn primary_label(mut self, span: Span, msg: impl Into<String>) -> Self {
        self.labels.push(Label::primary(span, msg));
        self
    }

    /// Add a secondary label (related location).
    pub fn secondary_label(mut self, span: Span, msg: impl Into<String>) -> Self {
        self.labels.push(Label::secondary(span, msg));
        self
    }

    /// Add a fix-it suggestion with explicit confidence.
    pub fn fix(
        mut self,
        span: Span,
        replacement: impl Into<String>,
        desc: impl Into<String>,
        confidence: HintConfidence,
    ) -> Self {
        self.fixits.push(FixIt::new(span, replacement, desc, confidence));
        self
    }

    /// Add a MachineCertain fix-it.
    pub fn fix_certain(
        mut self,
        span: Span,
        replacement: impl Into<String>,
        desc: impl Into<String>,
    ) -> Self {
        self.fixits
            .push(FixIt::new(span, replacement, desc, HintConfidence::MachineCertain));
        self
    }

    /// Add a Likely fix-it.
    pub fn fix_likely(
        mut self,
        span: Span,
        replacement: impl Into<String>,
        desc: impl Into<String>,
    ) -> Self {
        self.fixits
            .push(FixIt::new(span, replacement, desc, HintConfidence::Likely));
        self
    }

    /// Add a Possible fix-it.
    pub fn fix_possible(
        mut self,
        span: Span,
        replacement: impl Into<String>,
        desc: impl Into<String>,
    ) -> Self {
        self.fixits
            .push(FixIt::new(span, replacement, desc, HintConfidence::Possible));
        self
    }

    /// Add a causality link.
    pub fn caused_by(mut self, id: DiagnosticId, relation: CausalRelation) -> Self {
        self.cause_chain.push(CausalityLink { diag_id: id, relation });
        self
    }

    /// Add a "why does this exist?" explanation.
    pub fn why(mut self, explanation: impl Into<String>) -> Self {
        self.why_explanation = Some(explanation.into());
        self
    }

    /// Add a teaching note.
    pub fn teaching(mut self, note: impl Into<String>) -> Self {
        self.teaching_note = Some(note.into());
        self
    }

    /// Mark this diagnostic as cascade-suppressing for the given symbol.
    pub fn suppress_after(mut self, symbol: impl Into<String>) -> Self {
        self.suppress_cascade = true;
        self.suppressed_symbol = Some(symbol.into());
        self
    }

    /// Attach dataflow tracking information.
    pub fn dataflow(mut self, info: DataflowInfo) -> Self {
        self.dataflow = Some(info);
        self
    }

    /// Attach constraint-based type error information.
    pub fn type_constraint(mut self, info: TypeConstraintInfo) -> Self {
        self.type_constraint = Some(info);
        self
    }

    /// True if this diagnostic would prevent code generation.
    pub fn is_fatal(&self) -> bool {
        matches!(self.severity, Severity::Ice | Severity::Error)
    }
}

// =============================================================================
// §3  DIAGNOSTIC COLLECTOR
// =============================================================================

/// Collects diagnostics with cascade suppression, deduplication, and statistics.
#[derive(Debug, Clone)]
pub struct DiagnosticCollector {
    /// All collected diagnostics.
    pub diagnostics: Vec<Diagnostic>,
    /// Symbols that have been flagged as errors — suppress further errors about them.
    suppressed_symbols: Vec<String>,
    /// Seen (code_number, span.start, span.end) for deduplication.
    seen: Vec<(u16, u32, u32)>,
    /// Total error count.
    error_count: usize,
    /// Total warning count.
    warning_count: usize,
    /// Total note count.
    notes_count: usize,
}

impl DiagnosticCollector {
    pub fn new() -> Self {
        DiagnosticCollector {
            diagnostics: vec![],
            suppressed_symbols: vec![],
            seen: vec![],
            error_count: 0,
            warning_count: 0,
            notes_count: 0,
        }
    }

    /// Add a diagnostic, applying cascade suppression and deduplication.
    pub fn emit(&mut self, diag: Diagnostic) {
        // Deduplication: skip if same code + span was already emitted.
        let dedup_key = (diag.code.number, diag.span.start, diag.span.end);
        if self.seen.contains(&dedup_key) {
            return;
        }
        self.seen.push(dedup_key);

        // Check cascade suppression.
        if matches!(diag.severity, Severity::Error | Severity::Ice) {
            if let Some(ref sym) = diag.suppressed_symbol {
                if self.suppressed_symbols.contains(sym) {
                    return; // Suppressed — root cause already reported
                }
            }
            self.error_count += 1;

            // If this error suppresses cascade, register the symbol.
            if diag.suppress_cascade {
                if let Some(ref sym) = diag.suppressed_symbol {
                    if !self.suppressed_symbols.contains(sym) {
                        self.suppressed_symbols.push(sym.clone());
                    }
                }
            }
        } else if diag.severity == Severity::Warning {
            self.warning_count += 1;
        } else if diag.severity == Severity::Note {
            self.notes_count += 1;
        }

        self.diagnostics.push(diag);
    }

    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    pub fn error_count(&self) -> usize {
        self.error_count
    }

    pub fn warning_count(&self) -> usize {
        self.warning_count
    }

    pub fn notes_count(&self) -> usize {
        self.notes_count
    }

    /// Count the root-cause diagnostics vs dependent diagnostics.
    pub fn root_vs_dependent(&self) -> (usize, usize) {
        let root = self
            .diagnostics
            .iter()
            .filter(|d| {
                d.cause_chain
                    .iter()
                    .any(|l| l.relation == CausalRelation::RootCause)
                    || d.cause_chain.is_empty()
            })
            .count();
        let dependent = self.diagnostics.len() - root;
        (root, dependent)
    }

    /// Render all diagnostics as a human-readable string with source context.
    pub fn render(&self, source: &str, filename: &str) -> String {
        let renderer = DiagnosticRenderer::new(source, filename, RenderConfig::default());
        renderer.render_all(&self.diagnostics)
    }

    /// Render all diagnostics as JSON for IDE/LSP consumption.
    pub fn to_json(&self, filename: &str) -> String {
        let items: Vec<serde_json::Value> = self
            .diagnostics
            .iter()
            .map(|d| {
                let mut obj = serde_json::json!({
                    "id": d.id.0,
                    "code": d.code.to_string(),
                    "severity": format!("{:?}", d.severity).to_lowercase(),
                    "phase": d.phase.to_string(),
                    "message": d.message,
                    "span": {
                        "start": d.span.start,
                        "end": d.span.end,
                        "line": d.span.line,
                        "column": d.span.col,
                    },
                    "file": filename,
                });

                if !d.labels.is_empty() {
                    obj["labels"] = serde_json::Value::Array(
                        d.labels
                            .iter()
                            .map(|l| {
                                serde_json::json!({
                                    "message": l.message,
                                    "style": match l.style {
                                        LabelStyle::Primary => "primary",
                                        LabelStyle::Secondary => "secondary",
                                    },
                                    "span": {
                                        "start": l.span.start,
                                        "end": l.span.end,
                                        "line": l.span.line,
                                        "column": l.span.col,
                                    }
                                })
                            })
                            .collect(),
                    );
                }

                if !d.fixits.is_empty() {
                    obj["fixits"] = serde_json::Value::Array(
                        d.fixits
                            .iter()
                            .map(|f| {
                                serde_json::json!({
                                    "description": f.description,
                                    "replacement": f.replacement,
                                    "confidence": format!("{:?}", f.confidence).to_lowercase(),
                                    "span": {
                                        "start": f.span.start,
                                        "end": f.span.end,
                                        "line": f.span.line,
                                        "column": f.span.col,
                                    }
                                })
                            })
                            .collect(),
                    );
                }

                if !d.cause_chain.is_empty() {
                    obj["causeChain"] = serde_json::Value::Array(
                        d.cause_chain
                            .iter()
                            .map(|c| {
                                serde_json::json!({
                                    "id": c.diag_id.0,
                                    "relation": match c.relation {
                                        CausalRelation::RootCause => "root_cause",
                                        CausalRelation::Consequence => "consequence",
                                        CausalRelation::Related => "related",
                                        CausalRelation::Suppressed => "suppressed",
                                    }
                                })
                            })
                            .collect(),
                    );
                }

                if let Some(ref why) = d.why_explanation {
                    obj["why"] = serde_json::Value::String(why.clone());
                }
                if let Some(ref teaching) = d.teaching_note {
                    obj["teachingNote"] = serde_json::Value::String(teaching.clone());
                }
                if let Some(ref df) = d.dataflow {
                    obj["dataflow"] = serde_json::json!({
                        "definedAt": df.defined_at.map(|s| serde_json::json!({"line": s.line, "col": s.col})),
                        "lastAssignedAt": df.last_assigned_at.map(|s| serde_json::json!({"line": s.line, "col": s.col})),
                        "movedOrBorrowedAt": df.moved_or_borrowed_at.map(|s| serde_json::json!({"line": s.line, "col": s.col})),
                        "pathDescription": df.path_description,
                    });
                }
                if let Some(ref tc) = d.type_constraint {
                    obj["typeConstraint"] = serde_json::json!({
                        "constraints": tc.constraints.iter().map(|c| serde_json::json!({
                            "expected": c.expected,
                            "actual": c.actual,
                            "reason": c.reason,
                            "span": {"line": c.span.line, "col": c.span.col}
                        })).collect::<Vec<_>>()
                    });
                }

                obj
            })
            .collect();
        serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string())
    }

    /// Render a summary line with root/dependent error counts.
    pub fn summary_line(&self) -> String {
        let (root, dep) = self.root_vs_dependent();
        let mut parts = Vec::new();
        if self.error_count > 0 {
            parts.push(format!("{} error(s)", self.error_count));
        }
        if self.warning_count > 0 {
            parts.push(format!("{} warning(s)", self.warning_count));
        }
        if self.notes_count > 0 {
            parts.push(format!("{} note(s)", self.notes_count));
        }
        if root > 0 || dep > 0 {
            parts.push(format!("({} root, {} dependent)", root, dep));
        }
        if parts.is_empty() {
            "no diagnostics".to_string()
        } else {
            parts.join(", ")
        }
    }
}

impl Default for DiagnosticCollector {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// §4  DIAGNOSTIC RENDERER
// =============================================================================

/// Configuration for the diagnostic renderer.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Whether to use ANSI colors.
    pub color: bool,
    /// Tab width for display.
    pub tab_width: usize,
    /// Number of context lines above/below.
    pub context: usize,
    /// Whether to show teaching notes.
    pub teaching_mode: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        RenderConfig {
            color: true,
            tab_width: 4,
            context: 1,
            teaching_mode: false,
        }
    }
}

/// ANSI color codes.
struct Ansi;
impl Ansi {
    const RESET: &'static str = "\x1b[0m";
    const BOLD: &'static str = "\x1b[1m";
    const DIM: &'static str = "\x1b[2m";
    const RED: &'static str = "\x1b[31m";
    const BRIGHT_RED: &'static str = "\x1b[91m";
    const GREEN: &'static str = "\x1b[32m";
    const BRIGHT_YELLOW: &'static str = "\x1b[93m";
    const BLUE: &'static str = "\x1b[34m";
    const BRIGHT_CYAN: &'static str = "\x1b[96m";
    const MAGENTA: &'static str = "\x1b[35m";
}

/// Renders diagnostics with Rust-compiler-style formatting, rich labels,
/// fix-its, causality chains, dataflow info, and teaching notes.
pub struct DiagnosticRenderer<'src> {
    source: &'src str,
    filename: &'src str,
    cfg: RenderConfig,
    /// Pre-split source lines for O(1) access.
    lines: Vec<&'src str>,
}

impl<'src> DiagnosticRenderer<'src> {
    pub fn new(source: &'src str, filename: &'src str, cfg: RenderConfig) -> Self {
        let lines: Vec<&str> = source.split('\n').collect();
        DiagnosticRenderer { source, filename, cfg, lines }
    }

    /// Render all diagnostics.
    pub fn render_all(&self, diags: &[Diagnostic]) -> String {
        let mut out = String::new();
        for d in diags {
            out.push_str(&self.render(d));
            out.push('\n');
        }
        out
    }

    /// Render a single diagnostic.
    pub fn render(&self, d: &Diagnostic) -> String {
        let mut buf = String::new();

        // ── Header line  "error[E4001]: use of moved value `data`" ─────────
        let (sev_tag, sev_color) = match d.severity {
            Severity::Ice => ("internal compiler error", Ansi::BRIGHT_RED),
            Severity::Error => ("error", Ansi::BRIGHT_RED),
            Severity::Warning => ("warning", Ansi::BRIGHT_YELLOW),
            Severity::Note => ("note", Ansi::BRIGHT_CYAN),
            Severity::Help => ("help", Ansi::GREEN),
        };
        let code_str = d.code.to_string();
        let header = format!("{}[{}]: {}", sev_tag, code_str, d.message);
        writeln!(buf, "{}", self.paint_bold(sev_color, &header)).unwrap();

        // ── File + line location ─────────────────────────────────────────────
        if d.span.line > 0 {
            let loc = format!("  --> {}:{}:{}", self.filename, d.span.line, d.span.col);
            writeln!(buf, "{}", self.dim(&loc)).unwrap();
            writeln!(buf, "   {}", self.dim("|")).unwrap();
            self.render_source_snippet(&mut buf, d.span, LabelStyle::Primary);
        }

        // ── Secondary labels with their own source snippets ──────────────────
        for label in &d.labels {
            if label.span.line > 0 && label.span.line as usize <= self.lines.len() {
                // Context lines above
                let ctx_start = label.span.line as usize;
                if ctx_start > 1 {
                    let above = ctx_start.saturating_sub(self.cfg.context + 1);
                    for i in above..ctx_start.saturating_sub(1) {
                        if i < self.lines.len() {
                            writeln!(
                                buf,
                                "{:3} {} {}",
                                i + 1,
                                self.dim("|"),
                                self.expand_tabs(self.lines[i])
                            )
                            .unwrap();
                        }
                    }
                }
                // The label line
                let line_idx = (label.span.line as usize).saturating_sub(1);
                if line_idx < self.lines.len() {
                    writeln!(
                        buf,
                        "{:3} {} {}",
                        label.span.line,
                        self.dim("|"),
                        self.expand_tabs(self.lines[line_idx])
                    )
                    .unwrap();
                    // Underline
                    write!(buf, "   {} ", self.dim("|")).unwrap();
                    let line_text = self.expand_tabs(self.lines[line_idx]);
                    let col_chars = count_visible_cols(&line_text, label.span.col as usize, self.cfg.tab_width);
                    for _ in 0..col_chars {
                        buf.push(' ');
                    }
                    let len = (label.span.end - label.span.start).max(1) as usize;
                    match label.style {
                        LabelStyle::Primary => {
                            let underline = "^".repeat(len);
                            write!(buf, "{} {}", self.paint(Ansi::BRIGHT_RED, &underline), label.message).unwrap();
                        }
                        LabelStyle::Secondary => {
                            let underline = "-".repeat(len);
                            write!(buf, "{} {}", self.paint(Ansi::BLUE, &underline), label.message).unwrap();
                        }
                    }
                    writeln!(buf).unwrap();
                }
            }
        }

        // ── Dataflow info ────────────────────────────────────────────────────
        if let Some(ref df) = d.dataflow {
            writeln!(buf, "   {} = note: dataflow: {}", self.dim("|"), df.path_description).unwrap();
            if let Some(def_span) = df.defined_at {
                writeln!(
                    buf,
                    "   {} = note: value defined at {}:{}",
                    self.dim("|"),
                    def_span.line,
                    def_span.col
                )
                .unwrap();
            }
            if let Some(last) = df.last_assigned_at {
                writeln!(
                    buf,
                    "   {} = note: last assigned at {}:{}",
                    self.dim("|"),
                    last.line,
                    last.col
                )
                .unwrap();
            }
            if let Some(moved) = df.moved_or_borrowed_at {
                writeln!(
                    buf,
                    "   {} = note: moved/borrowed at {}:{}",
                    self.dim("|"),
                    moved.line,
                    moved.col
                )
                .unwrap();
            }
        }

        // ── Type constraint info ─────────────────────────────────────────────
        if let Some(ref tc) = d.type_constraint {
            writeln!(buf, "   {} = note: type constraint chain:", self.dim("|")).unwrap();
            for constraint in &tc.constraints {
                writeln!(
                    buf,
                    "   {} = note:   expected `{}`, found `{}` ({})",
                    self.dim("|"),
                    constraint.expected,
                    constraint.actual,
                    constraint.reason
                )
                .unwrap();
            }
        }

        // ── Causality chain ──────────────────────────────────────────────────
        if !d.cause_chain.is_empty() {
            writeln!(buf, "   {} = note: causality chain:", self.dim("|")).unwrap();
            for link in &d.cause_chain {
                let relation_str = match link.relation {
                    CausalRelation::RootCause => "root cause",
                    CausalRelation::Consequence => "consequence of",
                    CausalRelation::Related => "related to",
                    CausalRelation::Suppressed => "suppressed from",
                };
                writeln!(
                    buf,
                    "   {} = note:   {} {}",
                    self.dim("|"),
                    relation_str,
                    link.diag_id
                )
                .unwrap();
            }
        }

        // ── "Why" explanation ────────────────────────────────────────────────
        if let Some(ref why) = d.why_explanation {
            writeln!(
                buf,
                "   {} = note: {}",
                self.dim("|"),
                self.paint(Ansi::DIM, why)
            )
            .unwrap();
        }

        // ── Teaching note ────────────────────────────────────────────────────
        if self.cfg.teaching_mode {
            if let Some(ref teaching) = d.teaching_note {
                writeln!(buf, "   {}", self.dim("|")).unwrap();
                writeln!(
                    buf,
                    "   {} = teaching: {}",
                    self.dim("|"),
                    self.paint(Ansi::MAGENTA, teaching)
                )
                .unwrap();
            }
        }

        // ── Fix-it suggestions ───────────────────────────────────────────────
        for fixit in &d.fixits {
            let confidence_str = match fixit.confidence {
                HintConfidence::MachineCertain => "certain",
                HintConfidence::Likely => "high confidence",
                HintConfidence::Possible => "possible",
            };
            writeln!(
                buf,
                "   {} help: {} ({})",
                self.dim("|"),
                self.paint(Ansi::BRIGHT_CYAN, &fixit.description),
                confidence_str
            )
            .unwrap();

            // Show the fix-it as an inline diff if on the same line.
            if fixit.span.line > 0 && (fixit.span.line as usize) <= self.lines.len() {
                let line_idx = (fixit.span.line as usize).saturating_sub(1);
                let original = self.expand_tabs(self.lines[line_idx]);
                // Build the fixed line by replacing the span region.
                let byte_start = fixit.span.start as usize;  let byte_start = byte_start.min(self.source.len());
                let byte_end = fixit.span.end as usize;  let byte_end = byte_end.min(self.source.len());
                if byte_start < byte_end && byte_end <= self.source.len() {
                    let fixed = format!(
                        "{}{}{}",
                        &self.source[..byte_start],
                        fixit.replacement,
                        &self.source[byte_end..]
                    );
                    let fixed_line = fixed.split('\n').next().unwrap_or("");
                    let fixed_expanded = self.expand_tabs(fixed_line);
                    // Show diff-style: "   | - old" / "   | + new"
                    writeln!(
                        buf,
                        "   {} {}",
                        self.dim("|"),
                        self.paint(Ansi::RED, &format!("- {}", original.trim_end()))
                    )
                    .unwrap();
                    writeln!(
                        buf,
                        "   {} {}",
                        self.dim("|"),
                        self.paint(Ansi::GREEN, &format!("+ {}", fixed_expanded.trim_end()))
                    )
                    .unwrap();
                }
            }
        }

        // ── Cascade note ─────────────────────────────────────────────────────
        if d.suppress_cascade {
            if let Some(ref sym) = d.suppressed_symbol {
                writeln!(
                    buf,
                    "   {} = note: additional errors about `{}` are suppressed",
                    self.dim("|"),
                    sym
                )
                .unwrap();
            }
        }

        buf
    }

    // ── Internal rendering helpers ────────────────────────────────────────────

    fn render_source_snippet(&self, buf: &mut String, span: Span, style: LabelStyle) {
        if span.line == 0 || (span.line as usize) > self.lines.len() {
            return;
        }
        let line_idx = (span.line as usize).saturating_sub(1);
        let line_text = self.expand_tabs(self.lines[line_idx]);

        // Context lines above
        if self.cfg.context > 0 && line_idx > 0 {
            let start = line_idx.saturating_sub(self.cfg.context);
            for i in start..line_idx {
                writeln!(
                    buf,
                    "{:3} {} {}",
                    i + 1,
                    self.dim("|"),
                    self.expand_tabs(self.lines[i])
                )
                .unwrap();
            }
        }

        // Source line
        writeln!(buf, "{:3} {} {}", span.line, self.dim("|"), line_text).unwrap();

        // Underline
        write!(buf, "   {} ", self.dim("|")).unwrap();
        let col_chars = count_visible_cols(&line_text, span.col as usize, self.cfg.tab_width);
        for _ in 0..col_chars {
            buf.push(' ');
        }
        let len = (span.end - span.start).max(1) as usize;
        match style {
            LabelStyle::Primary => {
                let underline = "^".repeat(len);
                buf.push_str(&self.paint(Ansi::BRIGHT_RED, &underline));
            }
            LabelStyle::Secondary => {
                let underline = "-".repeat(len);
                buf.push_str(&self.paint(Ansi::BLUE, &underline));
            }
        }
        writeln!(buf).unwrap();

        // Context lines below
        if self.cfg.context > 0 && line_idx + 1 < self.lines.len() {
            let end = (line_idx + 1 + self.cfg.context).min(self.lines.len());
            for i in line_idx + 1..end {
                writeln!(
                    buf,
                    "{:3} {} {}",
                    i + 1,
                    self.dim("|"),
                    self.expand_tabs(self.lines[i])
                )
                .unwrap();
            }
        }
    }

    fn paint(&self, code: &str, text: &str) -> String {
        if self.cfg.color {
            format!("{}{}{}", code, text, Ansi::RESET)
        } else {
            text.to_owned()
        }
    }

    fn paint_bold(&self, color: &str, text: &str) -> String {
        if self.cfg.color {
            format!("{}{}{}{}{}", Ansi::BOLD, color, text, Ansi::RESET, Ansi::RESET)
        } else {
            text.to_owned()
        }
    }

    fn dim(&self, text: &str) -> String {
        if self.cfg.color {
            format!("{}{}{}", Ansi::DIM, text, Ansi::RESET)
        } else {
            text.to_owned()
        }
    }

    fn expand_tabs(&self, s: &str) -> String {
        s.replace('\t', &" ".repeat(self.cfg.tab_width))
    }
}

/// Count visible column positions up to a given byte offset, accounting for tabs.
fn count_visible_cols(line: &str, byte_offset: usize, tab_width: usize) -> usize {
    let mut cols = 0;
    let expanded = line.replace('\t', &" ".repeat(tab_width));
    // Approximate: just use the byte offset within the expanded line,
    // but cap it at the expanded line length.
    let byte_start_in_line = byte_offset.saturating_sub(1);
    // Count characters up to byte_start_in_line in the expanded line.
    for (i, ch) in expanded.char_indices() {
        if i >= byte_start_in_line {
            break;
        }
        cols += ch.len_utf8();
    }
    cols
}

// =============================================================================
// §5  ERROR EXPLANATION DATABASE
// =============================================================================

/// An example showing bad code and good code for an error.
#[derive(Debug, Clone)]
pub struct ErrorExample {
    pub bad_code: String,
    pub good_code: String,
    pub explanation: String,
}

/// A detailed explanation for a specific error code.
#[derive(Debug, Clone)]
pub struct ErrorExplanation {
    pub code: u16,
    pub title: String,
    pub description: String,
    pub examples: Vec<ErrorExample>,
    pub fixes: Vec<String>,
    pub related_codes: Vec<u16>,
    pub category: String,
}

/// The complete error explanation database for `jules explain Exxxx`.
#[derive(Debug, Clone)]
pub struct ErrorExplanationDatabase {
    explanations: HashMap<u16, ErrorExplanation>,
}

impl ErrorExplanationDatabase {
    /// Build the complete database with all known error codes.
    pub fn new() -> Self {
        let mut db = ErrorExplanationDatabase { explanations: HashMap::new() };

        // ── E0xxx — Lexer ─────────────────────────────────────────────────
        db.add(ErrorExplanation {
            code: 1,
            title: "Invalid token".into(),
            description: "The lexer encountered a character or sequence that does not form a valid Jules token. \
                This could be a stray Unicode character, an invalid operator, or an unexpected symbol.".into(),
            examples: vec![ErrorExample {
                bad_code: "let x = 5 @ 3".into(),
                good_code: "let x = 5 * 3".into(),
                explanation: "The `@` operator is the matrix multiply operator and requires tensor operands; use `*` for scalar multiplication.".into(),
            }],
            fixes: vec!["Remove the invalid character".into(), "Check if you meant to use a different operator".into()],
            related_codes: vec![2, 3, 4, 5],
            category: "Lexer".into(),
        });
        db.add(ErrorExplanation {
            code: 2,
            title: "Unterminated string literal".into(),
            description: "A string literal was opened with `\"` but never closed. \
                String literals must be terminated on the same line.".into(),
            examples: vec![ErrorExample {
                bad_code: "let s = \"hello".into(),
                good_code: "let s = \"hello\"".into(),
                explanation: "Add a closing `\"` to terminate the string.".into(),
            }],
            fixes: vec!["Add a closing double-quote".into()],
            related_codes: vec![1, 3],
            category: "Lexer".into(),
        });
        db.add(ErrorExplanation {
            code: 3,
            title: "Invalid escape sequence".into(),
            description: "A backslash in a string literal is followed by a character that is not a valid escape sequence.".into(),
            examples: vec![ErrorExample {
                bad_code: "let s = \"hello\\q\"".into(),
                good_code: "let s = \"hello\\n\"".into(),
                explanation: "Valid escape sequences: \\\\ \\n \\t \\r \\\" \\0 \\u{XXXX}".into(),
            }],
            fixes: vec!["Use a valid escape sequence".into(), "Escape the backslash with \\\\\\\\".into()],
            related_codes: vec![2],
            category: "Lexer".into(),
        });
        db.add(ErrorExplanation {
            code: 4,
            title: "Invalid number literal".into(),
            description: "A numeric literal is malformed, e.g. `0xGG` or `123.456.789`.".into(),
            examples: vec![ErrorExample {
                bad_code: "let x = 0xGH".into(),
                good_code: "let x = 0xFF".into(),
                explanation: "Hex literals may only contain digits 0-9 and letters A-F/a-f.".into(),
            }],
            fixes: vec!["Fix the number literal syntax".into()],
            related_codes: vec![1],
            category: "Lexer".into(),
        });
        db.add(ErrorExplanation {
            code: 5,
            title: "Unterminated block comment".into(),
            description: "A block comment `/* ... */` was opened but never closed. \
                Jules supports nested block comments, so every `/*` must have a matching `*/`.".into(),
            examples: vec![ErrorExample {
                bad_code: "/* this comment never ends".into(),
                good_code: "/* this comment does end */".into(),
                explanation: "Add `*/` to close the block comment.".into(),
            }],
            fixes: vec!["Add `*/` to close the comment".into()],
            related_codes: vec![1],
            category: "Lexer".into(),
        });

        // ── E1xxx — Parser ────────────────────────────────────────────────
        Self::add_parser_explanations(&mut db);

        // ── E2xxx — Type Checker ──────────────────────────────────────────
        Self::add_typeck_explanations(&mut db);

        // ── E3xxx — Semantic Analysis ─────────────────────────────────────
        Self::add_semantic_explanations(&mut db);

        // ── E4xxx — Borrow/Ownership ──────────────────────────────────────
        Self::add_borrowck_explanations(&mut db);

        // ── E5xxx — Effect System ─────────────────────────────────────────
        Self::add_effect_explanations(&mut db);

        // ── E6xxx — Ownership/Region ──────────────────────────────────────
        Self::add_region_explanations(&mut db);

        // ── E7xxx — Extended IR/Compilation ───────────────────────────────
        Self::add_extended_ir_explanations(&mut db);

        // ── E8xxx — Shader/GPU/ML Pipeline ───────────────────────────────
        Self::add_shader_gpu_ml_explanations(&mut db);

        db
    }

    /// Look up an explanation by numeric code.
    pub fn get(&self, code: u16) -> Option<&ErrorExplanation> {
        self.explanations.get(&code)
    }

    /// Look up by string code (e.g., "E4001").
    pub fn get_by_str(&self, code: &str) -> Option<&ErrorExplanation> {
        if code.len() < 2 || !code.starts_with('E') {
            return None;
        }
        code[1..].parse::<u16>().ok().and_then(|n| self.get(n))
    }

    /// Print all known error codes organized by category.
    pub fn print_all_codes(&self) {
        let mut codes: Vec<&ErrorExplanation> = self.explanations.values().collect();
        codes.sort_by_key(|e| e.code);

        let categories = [
            ("Lexer", 0u16..=999u16),
            ("Parser", 1000..=1999),
            ("Type Checker", 2000..=2999),
            ("Semantic Analysis", 3000..=3999),
            ("Borrow / Ownership", 4000..=4999),
            ("Effect System", 5000..=5999),
            ("Ownership / Region", 6000..=6999),
            ("IR / Compilation", 7000..=7999),
            ("Runtime", 9000..=9999),
        ];

        for (cat_name, range) in &categories {
            let in_range: Vec<&&ErrorExplanation> = codes.iter()
                .filter(|e| range.contains(&e.code))
                .collect();
            if !in_range.is_empty() {
                println!("\n  {} (E{}–E{}):", cat_name, range.start(), range.end());
                for expl in in_range {
                    println!("    E{:04}  {}", expl.code, expl.title);
                }
            }
        }
        println!();
    }

    fn add(&mut self, explanation: ErrorExplanation) {
        self.explanations.insert(explanation.code, explanation);
    }

    fn add_parser_explanations(db: &mut ErrorExplanationDatabase) {
        let explanations = vec![
            (1001, "Expected token", "The parser expected a specific token that was not found at the current position.", "Missing a semicolon, closing bracket, or keyword."),
            (1002, "Unexpected token", "The parser encountered a token that does not belong in the current grammar position.", "An extra or misplaced keyword/operator."),
            (1003, "Expected identifier", "The parser expected an identifier name but found something else.", "Using a keyword where a name is required."),
            (1004, "Expected type", "The parser expected a type annotation but found something else.", "Missing type after `:` in a parameter or variable declaration."),
            (1005, "Expected expression", "The parser expected an expression but found a keyword, delimiter, or end of file.", "Missing right-hand side of an assignment or argument."),
            (1006, "Expected item", "The parser expected a top-level declaration (`fn`, `struct`, `component`, etc.) but found something else.", "A statement at the top level of the file."),
            (1007, "Expected pattern", "The parser expected a pattern (in a `let` binding or `match` arm) but found something else.", "An expression where a pattern is expected."),
            (1008, "Duplicate field", "A struct literal or pattern contains the same field name more than once.", "Accidental copy-paste of a field."),
            (1009, "Invalid assignment target", "The left-hand side of an assignment is not a valid place expression.", "Trying to assign to a literal, call result, or other non-place expression."),
        ];
        for (code, title, desc, common) in explanations {
            db.add(ErrorExplanation {
                code,
                title: title.into(),
                description: desc.into(),
                examples: vec![ErrorExample {
                    bad_code: format!("// {common}"),
                    good_code: "// corrected version".into(),
                    explanation: common.into(),
                }],
                fixes: vec![format!("Check the syntax near the error location")],
                related_codes: vec![],
                category: "Parser".into(),
            });
        }
    }

    fn add_typeck_explanations(db: &mut ErrorExplanationDatabase) {
        let explanations: Vec<(u16, &str, &str)> = vec![
            (2001, "Type mismatch", "The expected type does not match the actual type of the expression."),
            (2002, "Unknown type", "A type name was used that does not correspond to any declared struct, component, or enum."),
            (2003, "Undefined variable", "A variable name was used that has not been declared in the current scope."),
            (2004, "Binary operator type mismatch", "The operands of a binary operator have incompatible types."),
            (2005, "Unary operator type mismatch", "The operand of a unary operator has an incompatible type."),
            (2006, "Cannot compare types", "Comparison operators require compatible types on both sides."),
            (2007, "Cannot index type", "The index operator `[]` was applied to a non-indexable type."),
            (2008, "No such field", "A field access refers to a field that does not exist on the type."),
            (2009, "Cannot assign to immutable binding", "An attempt was made to reassign a `let` binding (immutable by default in Jules)."),
            (2010, "Call on a non-function type", "A value was called as a function but its type is not callable."),
            (2011, "Wrong argument count", "A function was called with the wrong number of arguments."),
            (2012, "Argument type mismatch", "A function argument has a type that does not match the parameter type."),
            (2013, "Return type mismatch", "The function body produces a type that differs from the declared return type."),
            (2014, "Tensor shape mismatch", "Two tensors have incompatible static shapes for the operation."),
            (2015, "Tensor element type mismatch", "Two tensors have different element types (e.g. f32 vs f64)."),
            (2016, "Tensor rank mismatch", "Two tensors have different numbers of dimensions."),
            (2017, "Invalid tensor operation", "The operation cannot be applied to the given tensor types."),
            (2018, "Array length not compile-time constant", "An array type uses a length that cannot be evaluated at compile time."),
            (2019, "Parameter missing type annotation", "A function parameter lacks a type annotation."),
            (2020, "System parameter requires explicit type", "System parameters must have explicit type annotations."),
            (2021, "Operation on non-tensor type", "A tensor-specific operation was applied to a non-tensor type."),
            (2022, "@grad on non-differentiable type", "The @grad annotation was applied to a type that does not support automatic differentiation."),
            (2023, "Swizzle error", "An invalid swizzle mask or component count was used."),
            (2024, "Model layer error", "A model layer has inconsistent type or shape."),
            (2025, "Train block error", "A train block has a semantic error."),
            (2026, "Potential data race", "Two systems may concurrently access the same component."),
            (2027, "Generic bounds not implemented", "Generic bounds are not yet supported."),
            (2028, "Occurs check failed (infinite type)", "Type inference would create an infinite type (e.g. `T = Option<T>`)."),
            (2029, "Non-exhaustive match", "A match expression does not cover all possible cases."),
        ];
        for (code, title, desc) in explanations {
            db.add(ErrorExplanation {
                code,
                title: title.into(),
                description: desc.into(),
                examples: vec![],
                fixes: vec![],
                related_codes: vec![],
                category: "Type Checker".into(),
            });
        }
    }

    fn add_semantic_explanations(db: &mut ErrorExplanationDatabase) {
        let explanations: Vec<(u16, &str, &str)> = vec![
            (3001, "Unused variable", "A variable was declared but never used. Prefix with `_` to suppress."),
            (3002, "Variable shadowing", "A binding shadows an outer variable with the same name."),
            (3003, "Undeclared component", "A query references a component that has not been declared."),
            (3004, "Duplicate component in query", "A component name appears more than once in a single query clause."),
            (3005, "Component in both with/without", "A component cannot be both required and excluded."),
            (3006, "Unconstrained parallel query", "A parallel system has an empty `with` clause."),
            (3007, "Read-write aliasing between systems", "Two systems access the same component in conflicting ways."),
            (3008, "Duplicate struct/component field", "A field name appears more than once in a declaration."),
            (3009, "Agent declares no behaviours", "An agent with no behaviours will never act."),
            (3010, "Duplicate behaviour priority", "Two behaviours in the same agent share the same priority."),
            (3011, "Duplicate perception kind", "The same perception kind is declared more than once in an agent."),
            (3012, "Invalid memory capacity", "Memory capacity must be a positive number."),
            (3013, "Invalid learning rate", "Learning rate must be a positive finite number."),
            (3014, "Invalid discount factor", "Discount factor gamma must be in [0, 1]."),
            (3015, "Side effect in utility expression", "Utility expressions in behaviours must not have side effects."),
            (3016, "Model layer consistency error", "Model layers have inconsistent input/output shapes."),
            (3017, "Agent with learning but no train block", "An agent declares learning but is never used in a train block."),
            (3018, "Break outside loop", "`break` can only be used inside a loop."),
            (3019, "Continue outside loop", "`continue` can only be used inside a loop."),
            (3020, "Return outside function", "`return` can only be used inside a function body."),
            (3021, "Await outside async function", "`await` can only be used inside an `async` function."),
            (3022, "Nested spawn block", "`spawn` blocks cannot be nested."),
            (3023, "Nested sync block", "`sync` blocks cannot be nested."),
            (3024, "Nested atomic block", "`atomic` blocks cannot be nested."),
        ];
        for (code, title, desc) in explanations {
            db.add(ErrorExplanation {
                code,
                title: title.into(),
                description: desc.into(),
                examples: vec![],
                fixes: vec![],
                related_codes: vec![],
                category: "Semantic Analysis".into(),
            });
        }
    }

    fn add_borrowck_explanations(db: &mut ErrorExplanationDatabase) {
        let explanations: Vec<(u16, &str, &str)> = vec![
            (4001, "Use after move", "A value was used after it was moved to another binding."),
            (4002, "Borrow after move", "An attempt was made to borrow a value that has already been moved."),
            (4003, "Mutable borrow in parallel block", "Mutable borrows of outer variables are not allowed inside parallel/spawned blocks."),
            (4004, "Borrow while borrowed", "An attempt was made to borrow a value that is already borrowed."),
            (4005, "Immutable borrow while mutably borrowed", "Cannot immutably borrow a value that is already mutably borrowed."),
            (4006, "Assign to moved value", "Cannot assign to a binding whose value has been moved."),
            (4007, "Assign while borrowed", "Cannot assign to a binding while it is borrowed."),
            (4008, "Move in parallel block", "Moving outer variables into parallel blocks is unsafe."),
            (4009, "Move while borrowed", "Cannot move a value while it is borrowed."),
            (4010, "Assign through immutable reference", "Cannot assign through an immutable reference."),
            (4011, "Deref non-reference for assignment", "Cannot dereference a non-reference for assignment."),
            (4012, "Unsupported deref assignment target", "This dereference pattern is not supported for assignment."),
        ];
        for (code, title, desc) in explanations {
            db.add(ErrorExplanation {
                code,
                title: title.into(),
                description: desc.into(),
                examples: vec![],
                fixes: vec![],
                related_codes: vec![],
                category: "Borrow/Ownership".into(),
            });
        }
        // Add detailed E4001 explanation with examples
        if let Some(e) = db.explanations.get_mut(&4001) {
            e.examples = vec![ErrorExample {
                bad_code: "let data = compute()\nprocess(data)\nprint(data)".into(),
                good_code: "let data = compute()\nprocess(&data)\nprint(data)".into(),
                explanation: "In Jules, `process(data)` moves `data` into the function. \
                    Use `&data` to borrow instead, or `.clone()` to make a copy before the move.".into(),
            }];
            e.fixes = vec![
                "borrow with `&name` instead of moving".into(),
                "Use `.clone()` to create a copy before the move".into(),
                "Mark the type as `copy` if it should be implicitly copyable".into(),
            ];
        }
    }

    fn add_effect_explanations(db: &mut ErrorExplanationDatabase) {
        let explanations: Vec<(u16, &str, &str)> = vec![
            (5001, "Pure code depends on IO", "A function declared as pure performs IO operations."),
            (5002, "IO effect not declared", "A function performs IO but does not declare the `io` effect."),
            (5003, "Effect boundary violation", "An effect operation crosses a purity boundary."),
            (5004, "Mutation outside declared effect", "A mutation was performed outside of a declared effect block."),
            (5005, "Side effect in pure context", "A side effect was performed in a context declared as pure."),
            (5006, "Unsafe IO access", "IO was accessed without proper annotation or an unsafe block."),
        ];
        for (code, title, desc) in explanations {
            db.add(ErrorExplanation {
                code,
                title: title.into(),
                description: desc.into(),
                examples: vec![],
                fixes: vec![],
                related_codes: vec![],
                category: "Effect System".into(),
            });
        }
    }

    fn add_region_explanations(db: &mut ErrorExplanationDatabase) {
        let explanations: Vec<(u16, &str, &str)> = vec![
            (6001, "Region lifetime exceeded", "A reference outlives the region it was created in."),
            (6002, "Region borrow conflict", "Two conflicting borrows exist within the same region."),
            (6003, "Owned value used after transfer", "A value was used after its ownership was transferred to another task."),
            (6004, "Copy of non-copy type", "An attempt was made to implicitly copy a type that does not implement copy semantics."),
            (6005, "Mutable aliasing violation", "Two mutable references to the same data exist simultaneously."),
            (6006, "Thread ownership conflict", "Two threads may access the same data without proper synchronization."),
            (6007, "Shared mutable without atomic", "Shared mutable access without atomic operations is unsafe."),
        ];
        for (code, title, desc) in explanations {
            db.add(ErrorExplanation {
                code,
                title: title.into(),
                description: desc.into(),
                examples: vec![],
                fixes: vec![],
                related_codes: vec![],
                category: "Ownership/Region".into(),
            });
        }
    }

    fn add_shader_gpu_ml_explanations(db: &mut ErrorExplanationDatabase) {
        for code in [8001u16, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010, 8011, 8012] {
            let (title, desc, fixes, related) = match code {
                8001 => (
                    "shader stage missing",
                    "A shader definition must contain at least one stage (vertex, fragment, or compute). \
                     Without stages, the shader has no entry point and cannot be dispatched to the GPU.",
                    vec!["add a `vertex` and `fragment` stage to the shader definition",
                         "for compute workloads, add a `compute` stage with a workgroup size"],
                    vec![8003u16, 8004],
                ),
                8002 => (
                    "shader resource binding conflict",
                    "Two resources in the same shader are bound to the same group and binding index. \
                     Each resource binding must have a unique (group, binding) pair within a shader. \
                     The GPU uses these indices to locate resources at dispatch time.",
                    vec!["assign a unique group/binding pair to each resource",
                         "check for duplicate @group() @binding() annotations"],
                    vec![8001u16],
                ),
                8003 => (
                    "shader vertex stage missing required output",
                    "The vertex stage must output a position (vec4) for the rasterizer. \
                     Without a position output, the GPU cannot determine where to draw triangles.",
                    vec!["ensure the vertex function returns a struct with a position field at @location(0)"],
                    vec![8001u16, 8004],
                ),
                8004 => (
                    "shader fragment stage missing required output",
                    "The fragment stage must output at least one color target for rendering. \
                     A fragment shader that produces no output has no visible effect.",
                    vec!["add a color output at @location(0) to the fragment function"],
                    vec![8001u16, 8003],
                ),
                8005 => (
                    "invalid shader workgroup size",
                    "The workgroup size for a compute shader must be a triplet of positive integers \
                     (e.g., workgroup_size(64, 1, 1)). Zero or negative values are invalid. \
                     The total workgroup invocations must not exceed the GPU device limit (typically 256-1024).",
                    vec!["use a valid workgroup size like (64, 1, 1) or (8, 8, 1)",
                         "keep total invocations (x*y*z) within device limits"],
                    vec![8001u16, 8006],
                ),
                8006 => (
                    "GPU dispatch tensor layout mismatch",
                    "A GPU dispatch requires tensors in a layout compatible with the kernel. \
                     If the kernel expects NHWC but the tensor is NCHW, the computation will produce \
                     incorrect results. The compiler cannot automatically transpose at dispatch boundaries.",
                    vec!["use @layout(\"NHWC\") or @layout(\"NCHW\") on the tensor to match the kernel expectation",
                         "add an explicit TensorTranspose before dispatch to convert layouts"],
                    vec![8009u16, 8010],
                ),
                8007 => (
                    "ML graph mode conflict",
                    "An ML function declared with one graph mode (e.g., @graph) contains operations \
                     that require a different mode (e.g., @eager). Mixing modes within a single function \
                     is not allowed because it breaks the compilation contract for the graph.",
                    vec!["separate eager and traced operations into different functions",
                         "use @trace for the entire function if it needs mixed evaluation",
                         "annotate the function with the most permissive mode needed"],
                    vec![8008u16],
                ),
                8008 => (
                    "ML autodiff on non-differentiable operation",
                    "The autodiff system cannot compute gradients through this operation because it is \
                     not differentiable (e.g., integer comparison, side-effecting IO, control flow with \
                     non-smooth branches). Gradients require continuous, smooth mathematical operations.",
                    vec!["use differentiable operations (floating-point arithmetic, softmax, etc.)",
                         "detach non-differentiable subexpressions from the gradient computation",
                         "use @nofuse on the non-differentiable portion to isolate it"],
                    vec![8007u16],
                ),
                8009 => (
                    "tensor layout mismatch for kernel",
                    "A compute kernel expects input tensors in a specific memory layout. The provided \
                     tensor has a different layout, which would cause the kernel to read data incorrectly. \
                     Layout mismatches are one of the most common sources of silent correctness bugs in \
                     GPU computing, because the data appears valid but is interpreted in the wrong order.",
                    vec!["add @layout(\"NHWC\") or @layout(\"NCHW\") to match the kernel's expectation",
                         "insert a reshape or transpose operation before the kernel call",
                         "use @soa on the struct definition to ensure contiguous memory layout"],
                    vec![8006u16, 8010],
                ),
                8010 => (
                    "kernel fusion boundary violation",
                    "A kernel marked with @nofuse was included in a fusion group. Fusion boundaries \
                     exist because some operations cannot be safely combined (e.g., they have side effects, \
                     require global synchronization, or have conflicting memory access patterns).",
                    vec!["remove @nofuse from the kernel if fusion is actually safe",
                         "restructure the code so the non-fusible kernel is dispatched separately",
                         "use @fuse(aggressive) on surrounding kernels to maximize other fusion opportunities"],
                    vec![8006u16, 8009],
                ),
                8011 => (
                    "GPU capability not declared",
                    "A function uses GPU operations (shader dispatch, tensor operations on GPU) but \
                     does not declare the `gpu` capability in its effect signature. Without this \
                     declaration, the compiler cannot ensure the GPU is available at runtime, and \
                     the JIT cache key will be incorrect, potentially causing cache collisions.",
                    vec!["add `effect(gpu)` to the function signature",
                         "wrap GPU operations in a function that declares `effect(gpu, alloc, async)`"],
                    vec![8012u16],
                ),
                8012 => (
                    "SIMD capability not declared",
                    "A function uses SIMD vector operations but does not declare the `simd` capability \
                     in its effect signature. The compiler needs to know about SIMD requirements to \
                     correctly schedule vectorized code and to avoid inserting scalar fallbacks.",
                    vec!["add `effect(simd)` to the function signature",
                         "use @vectorize on the function to request SIMD codegen"],
                    vec![8011u16],
                ),
                _ => continue,
            };
            db.explanations.insert(code, ErrorExplanation {
                code,
                title: title.to_string(),
                description: desc.to_string(),
                examples: vec![],
                fixes: fixes.into_iter().map(|s| s.to_string()).collect(),
                related_codes: related,
                category: "Shader/GPU/ML Pipeline".into(),
            });
        }
    }

    fn add_extended_ir_explanations(db: &mut ErrorExplanationDatabase) {
        for code in [7008u16, 7009, 7010, 7011, 7012, 7013] {
            let (title, desc, fixes, related) = match code {
                7008 => (
                    "effect capability violation",
                    "An operation requires a capability (e.g., gpu, simd, io) that is not declared \
                     in the current function's effect signature. Jules uses effect capabilities as a \
                     permission system: a function can only perform operations whose capabilities it \
                     has explicitly declared. This enables the compiler to reason about purity, cache \
                     JIT-compiled code correctly, and automatically parallelize pure computations.",
                    vec!["add the required capability to the function's effect signature",
                         "e.g., `fn foo() effect(gpu, alloc)` for GPU+allocation effects",
                         "check if the operation can be replaced with a pure alternative"],
                    vec![5001u16, 5003],
                ),
                7009 => (
                    "comptime evaluation failed",
                    "A compile-time expression could not be fully evaluated during compilation. \
                     This typically happens when a comptime expression depends on runtime values, \
                     performs IO, calls an impure function, or exceeds the compiler's evaluation budget.",
                    vec!["ensure the expression only uses compile-time known constants",
                         "use only pure functions within comptime blocks",
                         "avoid IO or mutation in comptime expressions"],
                    vec![7008u16],
                ),
                7010 => (
                    "trait method not satisfied",
                    "A type does not implement a required trait method. In Jules, traits define \
                     interfaces that types must satisfy to be used in generic contexts. When a type \
                     is used where a trait bound is expected but the method is not implemented, the \
                     compiler cannot generate the correct dispatch code.",
                    vec!["add an `impl TraitName for TypeName` block",
                         "implement the missing method with the correct signature",
                         "check that the method signature matches the trait declaration exactly"],
                    vec![2001u16, 2010],
                ),
                7011 => (
                    "module visibility violation",
                    "An attempt was made to access a private item from outside its declaring module. \
                     Jules uses visibility modifiers (private, pub(crate), pub) to control access. \
                     By default, all items are private to their module. This prevents accidental \
                     coupling and enables the compiler to reason about API boundaries for optimization.",
                    vec!["add `pub` to the item declaration if it should be publicly accessible",
                         "add `pub(crate)` if it should be accessible within the crate but not externally",
                         "access the item from within its declaring module instead"],
                    vec![3001u16],
                ),
                7012 => (
                    "ECS system scheduling conflict",
                    "Two or more ECS systems write to the same component without an explicit ordering \
                     constraint. This creates a data race: the final state of the component depends on \
                     which system runs last, which is non-deterministic. The compiler refuses to generate \
                     code with implicit ordering because it would silently produce different results on \
                     different runs.",
                    vec!["add `before(OtherSystem)` or `after(OtherSystem)` to establish deterministic order",
                         "split the write access: one system writes, others only read",
                         "merge the conflicting systems into one if they must run in a specific order"],
                    vec![3007u16, 5004],
                ),
                7013 => (
                    "capability set inconsistency",
                    "The inferred effect capabilities of a function do not match its declared capabilities. \
                     This means the function performs operations that require capabilities it doesn't \
                     declare, or it declares capabilities it never uses. Both cases are problematic: \
                     missing capabilities cause soundness violations, while extra capabilities prevent \
                     optimization opportunities.",
                    vec!["update the function's effect signature to match its actual behavior",
                         "use `effect auto` to let the compiler infer capabilities",
                         "remove unused capabilities and add missing ones"],
                    vec![7008u16, 5003],
                ),
                _ => continue,
            };
            db.explanations.insert(code, ErrorExplanation {
                code,
                title: title.to_string(),
                description: desc.to_string(),
                examples: vec![],
                fixes: fixes.into_iter().map(|s| s.to_string()).collect(),
                related_codes: related,
                category: "IR/Compilation".into(),
            });
        }
    }
}

impl Default for ErrorExplanationDatabase {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// §6  DIAGNOSTIC KNOWLEDGE ENGINE
// =============================================================================

/// A common mistake pattern observed across compilations.
#[derive(Debug, Clone)]
pub struct CommonMistake {
    pub pattern: String,
    pub frequency: u64,
    pub suggestion: String,
    pub confidence: HintConfidence,
}

/// The diagnostic knowledge engine tracks error frequencies and common mistake
/// patterns to improve the ranking and relevance of fix-it suggestions.
#[derive(Debug, Clone)]
pub struct DiagnosticKnowledgeEngine {
    /// Frequency of each error code.
    error_frequencies: HashMap<u16, u64>,
    /// Common mistake patterns observed.
    common_mistakes: Vec<CommonMistake>,
    /// Parser ambiguity stats.
    ambiguity_stats: HashMap<String, u64>,
}

impl DiagnosticKnowledgeEngine {
    pub fn new() -> Self {
        let mut engine = DiagnosticKnowledgeEngine {
            error_frequencies: HashMap::new(),
            common_mistakes: Vec::new(),
            ambiguity_stats: HashMap::new(),
        };
        // Seed with known common mistakes.
        engine.common_mistakes = vec![
            CommonMistake {
                pattern: "assignment to immutable binding".into(),
                frequency: 1000,
                suggestion: "use `:=` for mutation, or `let mut` for mutable binding".into(),
                confidence: HintConfidence::MachineCertain,
            },
            CommonMistake {
                pattern: "use of moved value".into(),
                frequency: 800,
                suggestion: "borrow with `&name` or `.clone()` before move".into(),
                confidence: HintConfidence::Likely,
            },
            CommonMistake {
                pattern: "missing semicolon".into(),
                frequency: 700,
                suggestion: "add `;` to end the statement".into(),
                confidence: HintConfidence::MachineCertain,
            },
            CommonMistake {
                pattern: "undeclared variable".into(),
                frequency: 600,
                suggestion: "declare with `let` or check spelling".into(),
                confidence: HintConfidence::Possible,
            },
            CommonMistake {
                pattern: "type mismatch".into(),
                frequency: 500,
                suggestion: "check expected vs actual types; add explicit cast if needed".into(),
                confidence: HintConfidence::Possible,
            },
        ];
        engine
    }

    /// Record that an error code was emitted.
    pub fn record_error(&mut self, code: u16) {
        *self.error_frequencies.entry(code).or_insert(0) += 1;
        // Also update common mistake frequency if matched.
        for mistake in &mut self.common_mistakes {
            // We match loosely by checking if the pattern relates to this code.
            mistake.frequency += 1;
        }
    }

    /// Get the frequency of a specific error code.
    pub fn frequency(&self, code: u16) -> u64 {
        *self.error_frequencies.get(&code).unwrap_or(&0)
    }

    /// Get the top-N most common mistakes, sorted by frequency.
    pub fn top_mistakes(&self, n: usize) -> Vec<&CommonMistake> {
        let mut mistakes: Vec<&CommonMistake> = self.common_mistakes.iter().collect();
        mistakes.sort_by(|a, b| b.frequency.cmp(&a.frequency));
        mistakes.into_iter().take(n).collect()
    }

    /// Record a parser ambiguity.
    pub fn record_ambiguity(&mut self, context: &str) {
        *self.ambiguity_stats.entry(context.into()).or_insert(0) += 1;
    }

    /// Look up a suggestion for a given pattern, ranked by frequency.
    pub fn suggest_for(&self, pattern: &str) -> Option<&CommonMistake> {
        self.common_mistakes
            .iter()
            .filter(|m| m.pattern.contains(pattern) || pattern.contains(&m.pattern))
            .max_by_key(|m| m.frequency)
    }

    /// Merge statistics from another engine (e.g. across compilation sessions).
    pub fn merge(&mut self, other: &DiagnosticKnowledgeEngine) {
        for (code, freq) in &other.error_frequencies {
            *self.error_frequencies.entry(*code).or_insert(0) += freq;
        }
        for (ctx, count) in &other.ambiguity_stats {
            *self.ambiguity_stats.entry(ctx.clone()).or_insert(0) += count;
        }
    }
}

impl Default for DiagnosticKnowledgeEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// §7  FROM IMPLEMENTATIONS
// =============================================================================

impl From<crate::compiler::lexer::LexError> for Diagnostic {
    fn from(e: crate::compiler::lexer::LexError) -> Self {
        Diagnostic::error(1, Phase::Lexer, e.span, e.message)
            .primary_label(e.span, "invalid token")
    }
}

impl From<crate::compiler::parser::ParseError> for Diagnostic {
    fn from(e: crate::compiler::parser::ParseError) -> Self {
        let mut diag = Diagnostic::error(1001, Phase::Parser, e.span, &e.message)
            .primary_label(e.span, "here");
        if let Some(ref hint) = e.hint {
            diag = diag.fix_possible(e.span, "", hint);
        }
        diag
    }
}

// issue #33: All three passes now share SimpleDiagnostic, so we need only
// one From impl.  The individual From impls for typeck/sema/borrowck
// were removed to avoid conflicts.

impl From<crate::compiler::SimpleDiagnostic> for Diagnostic {
    fn from(d: crate::compiler::SimpleDiagnostic) -> Self {
        let code_num = d.code.and_then(|c| c[1..].parse::<u16>().ok()).unwrap_or(9999);
        let sev = match d.severity {
            crate::compiler::SimpleSeverity::Error => Severity::Error,
            crate::compiler::SimpleSeverity::Warning => Severity::Warning,
            crate::compiler::SimpleSeverity::Note => Severity::Note,
        };
        let mut diag = Diagnostic {
            id: DiagnosticId::fresh(),
            severity: sev,
            code: DiagCode::error(code_num),
            phase: Phase::TypeCheck,
            message: d.message,
            span: d.span,
            labels: vec![],
            fixits: vec![],
            cause_chain: vec![],
            why_explanation: None,
            teaching_note: None,
            suppress_cascade: false,
            suppressed_symbol: None,
            dataflow: None,
            type_constraint: None,
        };
        for (span, msg) in d.labels {
            diag = diag.secondary_label(span, msg);
        }
        if let Some(ref hint) = d.hint {
            diag = diag.fix_likely(d.span, "", hint);
        }
        diag
    }
}

// =============================================================================
// §8  WELL-KNOWN ERROR CODES
// =============================================================================

pub mod codes {
    // Lexer errors (E0001-E0099)
    pub const LEX_UNTERMINATED_STRING: u16 = 1;
    pub const LEX_UNTERMINATED_COMMENT: u16 = 2;
    pub const LEX_INVALID_ESCAPE: u16 = 3;
    pub const LEX_INVALID_NUMBER: u16 = 4;
    pub const LEX_INVALID_CHAR: u16 = 5;

    // Parser errors (E0100-E0199)
    pub const PARSE_UNEXPECTED_TOKEN: u16 = 100;
    pub const PARSE_EXPECTED_TOKEN: u16 = 101;
    pub const PARSE_MISSING_SEMI: u16 = 102;
    pub const PARSE_UNCLOSED_BLOCK: u16 = 103;

    // Name resolution errors (E0200-E0299)
    pub const NAME_UNDECLARED: u16 = 200;
    pub const NAME_ALREADY_DECLARED: u16 = 201;
    pub const NAME_SHADOW: u16 = 202;
    pub const NAME_RESULT_OUTSIDE_CONTRACT: u16 = 203;

    // Type errors (E0300-E0399)
    pub const TYPE_MISMATCH: u16 = 300;
    pub const TYPE_OPTION_IN_ARITH: u16 = 301;
    pub const TYPE_OPTION_COMPARE: u16 = 302;
    pub const TYPE_INT_WIDTH_MISMATCH: u16 = 303;
    pub const TYPE_MISSING_RETURN: u16 = 304;
    pub const TYPE_NOT_CALLABLE: u16 = 305;
    pub const TYPE_WRONG_ARG_COUNT: u16 = 306;

    // Ownership errors (E0400-E0499)
    pub const OWN_USE_AFTER_MOVE: u16 = 400;
    pub const OWN_MUTATE_IMMUTABLE: u16 = 401;
    pub const OWN_BORROW_CONFLICT: u16 = 402;
    pub const OWN_REGION_ESCAPE: u16 = 403;
    pub const OWN_TASK_TRANSFER: u16 = 404;

    // Effect errors (E0500-E0599)
    pub const EFFECT_PURE_VIOLATION: u16 = 500;
    pub const EFFECT_EMIT_OUTSIDE_BLOCK: u16 = 501;
    pub const EFFECT_IO_IN_PURE: u16 = 502;

    // Borrow check errors (E0600-E0699)
    pub const BORROW_WRITE_WHILE_BORROWED: u16 = 600;
    pub const BORROW_MUT_CONFLICT: u16 = 601;
    pub const BORROW_DEREF_IMMUTABLE: u16 = 602;

    // Optimizer errors (E0700-E0799)
    pub const OPT_INVALID_REWRITE: u16 = 700;
    pub const OPT_SEMANTIC_CHANGE: u16 = 701;

    // Backend errors (E0800-E0899)
    pub const BACKEND_UNSUPPORTED: u16 = 800;
    pub const BACKEND_LOWER_FAIL: u16 = 801;

    // Runtime errors (E9000-E9999)
    pub const RUNTIME_ARITH_ERROR: u16 = 9000;
    pub const RUNTIME_INDEX_OOB: u16 = 9001;
}

// =============================================================================
// §9  TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_span(line: u32, col: u32) -> Span {
        Span { start: 0, end: 5, line: line as u16, col: col as u16 }
    }

    // ── Label rendering ────────────────────────────────────────────────────

    #[test]
    fn test_label_primary_vs_secondary() {
        let primary = Label::primary(test_span(1, 5), "main error");
        assert_eq!(primary.style, LabelStyle::Primary);
        assert_eq!(primary.message, "main error");

        let secondary = Label::secondary(test_span(2, 3), "related");
        assert_eq!(secondary.style, LabelStyle::Secondary);
        assert_eq!(secondary.message, "related");
    }

    // ── Fix-it confidence display ──────────────────────────────────────────

    #[test]
    fn test_fixit_confidence() {
        let fix_certain = FixIt::new(
            test_span(1, 5),
            "&data",
            "borrow instead",
            HintConfidence::MachineCertain,
        );
        assert_eq!(fix_certain.confidence, HintConfidence::MachineCertain);
        assert_eq!(format!("{}", fix_certain.confidence), "certain");

        let fix_likely = FixIt::new(
            test_span(1, 5),
            ".clone()",
            "clone the value",
            HintConfidence::Likely,
        );
        assert_eq!(fix_likely.confidence, HintConfidence::Likely);

        let fix_possible = FixIt::new(
            test_span(1, 5),
            "/* fix */",
            "try this",
            HintConfidence::Possible,
        );
        assert_eq!(fix_possible.confidence, HintConfidence::Possible);
        assert_eq!(format!("{}", fix_possible.confidence), "possible");
    }

    // ── Causality chain tracking ───────────────────────────────────────────

    #[test]
    fn test_causality_chain() {
        let root_id = DiagnosticId::fresh();
        let root_diag = Diagnostic::error(4001, Phase::BorrowCheck, test_span(10, 5), "use of moved value `data`")
            .caused_by(root_id, CausalRelation::RootCause)
            .suppress_after("data");

        let dep_id = DiagnosticId::fresh();
        let dep_diag = Diagnostic::error(4001, Phase::BorrowCheck, test_span(15, 3), "use of moved value `data`")
            .caused_by(root_id, CausalRelation::Consequence);

        assert!(root_diag.suppress_cascade);
        assert_eq!(root_diag.suppressed_symbol.as_deref(), Some("data"));
        assert!(!dep_diag.cause_chain.is_empty());
        assert_eq!(dep_diag.cause_chain[0].relation, CausalRelation::Consequence);
    }

    // ── Cascade suppression ────────────────────────────────────────────────

    #[test]
    fn test_cascade_suppression() {
        let mut collector = DiagnosticCollector::new();

        let root = Diagnostic::error(4001, Phase::BorrowCheck, test_span(10, 5), "use of moved value `data`")
            .suppress_after("data");
        collector.emit(root);
        assert_eq!(collector.error_count(), 1);

        // This should be suppressed because it references "data"
        let dep = Diagnostic::error(4001, Phase::BorrowCheck, test_span(15, 3), "use of moved value `data` again")
            .suppress_after("data");
        collector.emit(dep);
        // Still 1 because the dependent was suppressed
        assert_eq!(collector.error_count(), 1);
    }

    // ── Dataflow info rendering ────────────────────────────────────────────

    #[test]
    fn test_dataflow_info() {
        let df = DataflowInfo {
            defined_at: Some(test_span(5, 9)),
            last_assigned_at: Some(test_span(10, 5)),
            moved_or_borrowed_at: Some(test_span(12, 13)),
            path_description: "value moved from `data` into `process()`".into(),
        };

        let diag = Diagnostic::error(4001, Phase::BorrowCheck, test_span(18, 9), "use of moved value `data`")
            .dataflow(df);

        assert!(diag.dataflow.is_some());
        let df_ref = diag.dataflow.as_ref().unwrap();
        assert_eq!(df_ref.path_description, "value moved from `data` into `process()`");
        assert!(df_ref.defined_at.is_some());
    }

    #[test]
    fn test_dataflow_rendering() {
        let source = "let data = compute()\nprocess(data)\nprint(data)\n";
        let df = DataflowInfo {
            defined_at: Some(test_span(1, 5)),
            last_assigned_at: None,
            moved_or_borrowed_at: Some(test_span(2, 9)),
            path_description: "value moved from `data` into `process()`".into(),
        };
        let diag = Diagnostic::error(4001, Phase::BorrowCheck, test_span(3, 7), "use of moved value `data`")
            .primary_label(test_span(3, 7), "value used after move")
            .secondary_label(test_span(2, 9), "value moved here")
            .dataflow(df)
            .fix_certain(test_span(2, 9), "&data", "borrow instead")
            .why("immutable bindings are the default in Jules; values are moved, not copied, unless the type implements copy semantics");

        let renderer = DiagnosticRenderer::new(source, "main.jules", RenderConfig { color: false, ..RenderConfig::default() });
        let output = renderer.render(&diag);
        assert!(output.contains("dataflow"));
        assert!(output.contains("value moved from `data` into `process()`"));
        assert!(output.contains("defined at"));
    }

    // ── Type constraint info rendering ─────────────────────────────────────

    #[test]
    fn test_type_constraint_info() {
        let tc = TypeConstraintInfo {
            constraints: vec![
                TypeConstraint {
                    span: test_span(5, 10),
                    expected: "i64".into(),
                    actual: "i32".into(),
                    reason: "parameter requires i64".into(),
                },
                TypeConstraint {
                    span: test_span(3, 7),
                    expected: "i32".into(),
                    actual: "f32".into(),
                    reason: "inferred from literal `3.14`".into(),
                },
            ],
        };

        let diag = Diagnostic::error(2001, Phase::TypeCheck, test_span(5, 10), "type mismatch: expected `i64`, found `i32`")
            .type_constraint(tc);

        assert!(diag.type_constraint.is_some());
        let tc_ref = diag.type_constraint.as_ref().unwrap();
        assert_eq!(tc_ref.constraints.len(), 2);
        assert_eq!(tc_ref.constraints[0].expected, "i64");
        assert_eq!(tc_ref.constraints[0].actual, "i32");
    }

    #[test]
    fn test_type_constraint_rendering() {
        let source = "fn foo(x: i64) {}\nfoo(3)\n";
        let tc = TypeConstraintInfo {
            constraints: vec![TypeConstraint {
                span: test_span(2, 5),
                expected: "i64".into(),
                actual: "i32".into(),
                reason: "parameter requires i64".into(),
            }],
        };
        let diag = Diagnostic::error(2001, Phase::TypeCheck, test_span(2, 5), "type mismatch")
            .type_constraint(tc);

        let renderer = DiagnosticRenderer::new(source, "test.jules", RenderConfig { color: false, ..RenderConfig::default() });
        let output = renderer.render(&diag);
        assert!(output.contains("type constraint chain"));
        assert!(output.contains("expected `i64`, found `i32`"));
    }

    // ── "Why" explanation rendering ────────────────────────────────────────

    #[test]
    fn test_why_explanation() {
        let diag = Diagnostic::error(4001, Phase::BorrowCheck, test_span(1, 1), "use of moved value")
            .why("Jules uses move semantics by default; values are moved, not copied");

        assert!(diag.why_explanation.is_some());
        assert!(diag.why_explanation.unwrap().contains("move semantics"));
    }

    #[test]
    fn test_why_rendering() {
        let source = "let x = 1\n";
        let diag = Diagnostic::error(4001, Phase::BorrowCheck, test_span(1, 1), "use of moved value")
            .why("immutable bindings are the default in Jules");

        let renderer = DiagnosticRenderer::new(source, "test.jules", RenderConfig { color: false, ..RenderConfig::default() });
        let output = renderer.render(&diag);
        assert!(output.contains("note:"));
        assert!(output.contains("immutable bindings"));
    }

    // ── Error explanation database lookup ──────────────────────────────────

    #[test]
    fn test_error_explanation_database() {
        let db = ErrorExplanationDatabase::new();

        // Lookup by numeric code
        let exp = db.get(4001).expect("E4001 should exist");
        assert_eq!(exp.title, "Use after move");
        assert_eq!(exp.category, "Borrow/Ownership");
        assert!(!exp.examples.is_empty());

        // Lookup by string
        let exp2 = db.get_by_str("E4001").expect("E4001 string lookup should work");
        assert_eq!(exp2.code, 4001);

        // Non-existent code
        assert!(db.get(9998).is_none());
        assert!(db.get_by_str("garbage").is_none());
    }

    #[test]
    fn test_explanations_cover_all_ranges() {
        let db = ErrorExplanationDatabase::new();
        // Spot check key codes from each range
        assert!(db.get(1).is_some(), "Lexer E0001");
        assert!(db.get(1001).is_some(), "Parser E1001");
        assert!(db.get(2001).is_some(), "TypeCheck E2001");
        assert!(db.get(3001).is_some(), "Semantic E3001");
        assert!(db.get(4001).is_some(), "Borrow E4001");
        assert!(db.get(5001).is_some(), "Effect E5001");
        assert!(db.get(6001).is_some(), "Region E6001");
    }

    // ── Recovery suggestion generation ─────────────────────────────────────

    #[test]
    fn test_recovery_suggestions() {
        let suggestions = RecoverySuggestion::suggestions_for(RecoveryContext::ImmutableBinding);
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].confidence, HintConfidence::MachineCertain);
        assert!(suggestions[0].suggestion.contains(":="));

        let suggestions = RecoverySuggestion::suggestions_for(RecoveryContext::OwnershipViolation);
        assert!(!suggestions.is_empty());
        assert!(suggestions[0].suggestion.contains("borrow") || suggestions[0].suggestion.contains("clone"));
    }

    // ── Knowledge engine frequency tracking ────────────────────────────────

    #[test]
    fn test_knowledge_engine_frequency() {
        let mut engine = DiagnosticKnowledgeEngine::new();

        engine.record_error(4001);
        engine.record_error(4001);
        engine.record_error(2001);

        assert_eq!(engine.frequency(4001), 2);
        assert_eq!(engine.frequency(2001), 1);
        assert_eq!(engine.frequency(9999), 0);
    }

    #[test]
    fn test_knowledge_engine_top_mistakes() {
        let engine = DiagnosticKnowledgeEngine::new();
        let top = engine.top_mistakes(3);
        assert!(!top.is_empty());
        assert!(top.len() <= 3);
        // Should be sorted by frequency descending
        for i in 1..top.len() {
            assert!(top[i - 1].frequency >= top[i].frequency);
        }
    }

    #[test]
    fn test_knowledge_engine_suggest() {
        let engine = DiagnosticKnowledgeEngine::new();
        let suggestion = engine.suggest_for("moved value");
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().suggestion.contains("borrow") || suggestion.unwrap().suggestion.contains("clone"));
    }

    // ── Diagnostic deduplication ───────────────────────────────────────────

    #[test]
    fn test_deduplication() {
        let mut collector = DiagnosticCollector::new();
        let span = test_span(5, 3);

        let d1 = Diagnostic::error(2001, Phase::TypeCheck, span, "type mismatch");
        let d2 = Diagnostic::error(2001, Phase::TypeCheck, span, "type mismatch (duplicate)");
        let d3 = Diagnostic::error(2002, Phase::TypeCheck, span, "different error");

        collector.emit(d1);
        collector.emit(d2); // Should be deduplicated (same code + span)
        collector.emit(d3); // Different code, should pass

        assert_eq!(collector.diagnostics.len(), 2);
    }

    // ── JSON output ────────────────────────────────────────────────────────

    #[test]
    fn test_json_output() {
        let mut collector = DiagnosticCollector::new();
        let diag = Diagnostic::error(4001, Phase::BorrowCheck, test_span(10, 5), "use of moved value `data`")
            .primary_label(test_span(12, 9), "value moved here")
            .fix_certain(test_span(12, 9), "&data", "borrow instead")
            .why("values are moved by default");

        collector.emit(diag);

        let json = collector.to_json("main.jules");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON should be valid");
        let items = parsed.as_array().expect("should be an array");
        assert_eq!(items.len(), 1);

        let first = &items[0];
        assert_eq!(first["code"].as_str().unwrap(), "E4001");
        assert_eq!(first["severity"].as_str().unwrap(), "error");
        assert_eq!(first["phase"].as_str().unwrap(), "borrow-check");
        assert_eq!(first["message"].as_str().unwrap(), "use of moved value `data`");
        assert!(first["labels"].is_array());
        assert!(first["fixits"].is_array());
        assert_eq!(first["why"].as_str().unwrap(), "values are moved by default");
    }

    // ── Summary line format ────────────────────────────────────────────────

    #[test]
    fn test_summary_line() {
        let mut collector = DiagnosticCollector::new();
        assert_eq!(collector.summary_line(), "no diagnostics");

        collector.emit(Diagnostic::error(4001, Phase::BorrowCheck, test_span(1, 1), "err1"));
        collector.emit(Diagnostic::error(4002, Phase::BorrowCheck, test_span(2, 1), "err2"));
        collector.emit(Diagnostic::warning(3001, Phase::SemanticAnalysis, test_span(3, 1), "warn1"));

        let summary = collector.summary_line();
        assert!(summary.contains("2 error(s)"));
        assert!(summary.contains("1 warning(s)"));
    }

    // ── Builder pattern ────────────────────────────────────────────────────

    #[test]
    fn test_builder_pattern() {
        let root_id = DiagnosticId::fresh();
        let diag = Diagnostic::error(4001, Phase::BorrowCheck, test_span(18, 9), "use of moved value `data`")
            .primary_label(test_span(18, 9), "value used after move")
            .secondary_label(test_span(12, 9), "value moved here")
            .fix_certain(test_span(12, 9), "&data", "borrow instead")
            .fix_likely(test_span(12, 9), "data.clone()", "clone the value")
            .caused_by(root_id, CausalRelation::RootCause)
            .why("immutable bindings are the default in Jules")
            .teaching("In Jules, values are moved by default. Use & to borrow.")
            .suppress_after("data")
            .dataflow(DataflowInfo {
                defined_at: Some(test_span(5, 9)),
                last_assigned_at: None,
                moved_or_borrowed_at: Some(test_span(12, 9)),
                path_description: "value moved from `data` into `process()`".into(),
            })
            .type_constraint(TypeConstraintInfo {
                constraints: vec![TypeConstraint {
                    span: test_span(5, 9),
                    expected: "&T".into(),
                    actual: "T".into(),
                    reason: "borrow required".into(),
                }],
            });

        assert_eq!(diag.severity, Severity::Error);
        assert_eq!(diag.code.number, 4001);
        assert_eq!(diag.phase, Phase::BorrowCheck);
        assert_eq!(diag.labels.len(), 2);
        assert_eq!(diag.fixits.len(), 2);
        assert_eq!(diag.fixits[0].confidence, HintConfidence::MachineCertain);
        assert_eq!(diag.fixits[1].confidence, HintConfidence::Likely);
        assert!(!diag.cause_chain.is_empty());
        assert!(diag.why_explanation.is_some());
        assert!(diag.teaching_note.is_some());
        assert!(diag.suppress_cascade);
        assert!(diag.dataflow.is_some());
        assert!(diag.type_constraint.is_some());
    }

    // ── Teaching mode rendering ────────────────────────────────────────────

    #[test]
    fn test_teaching_mode_rendering() {
        let source = "let x = 1\n";
        let diag = Diagnostic::error(4001, Phase::BorrowCheck, test_span(1, 1), "use of moved value")
            .teaching("In Jules, values are moved by default. Use & to borrow.");

        // With teaching mode ON
        let renderer = DiagnosticRenderer::new(source, "test.jules", RenderConfig {
            color: false,
            teaching_mode: true,
            ..RenderConfig::default()
        });
        let output = renderer.render(&diag);
        assert!(output.contains("teaching:"));
        assert!(output.contains("values are moved by default"));

        // With teaching mode OFF
        let renderer_no_teach = DiagnosticRenderer::new(source, "test.jules", RenderConfig {
            color: false,
            teaching_mode: false,
            ..RenderConfig::default()
        });
        let output_no_teach = renderer_no_teach.render(&diag);
        assert!(!output_no_teach.contains("teaching:"));
    }

    // ── Diagnostic ID uniqueness ───────────────────────────────────────────

    #[test]
    fn test_diagnostic_id_uniqueness() {
        let id1 = DiagnosticId::fresh();
        let id2 = DiagnosticId::fresh();
        assert_ne!(id1, id2);
    }

    // ── DiagCode formatting ────────────────────────────────────────────────

    #[test]
    fn test_diag_code_formatting() {
        assert_eq!(DiagCode::error(4001).to_string(), "E4001");
        assert_eq!(DiagCode::warning(3001).to_string(), "W3001");
        assert_eq!(DiagCode::note(100).to_string(), "N0100");
        assert_eq!(DiagCode::ice(1).to_string(), "I0001");
    }

    // ── From LexError ─────────────────────────────────────────────────────

    #[test]
    fn test_from_lex_error() {
        let lex_err = crate::compiler::lexer::LexError::new("bad token", test_span(1, 5));
        let diag: Diagnostic = lex_err.into();
        assert_eq!(diag.severity, Severity::Error);
        assert_eq!(diag.code.number, 1);
        assert_eq!(diag.phase, Phase::Lexer);
    }

    // ── From ParseError ────────────────────────────────────────────────────

    #[test]
    fn test_from_parse_error() {
        let parse_err = crate::compiler::parser::ParseError::new(test_span(3, 7), "expected `;`");
        let diag: Diagnostic = parse_err.into();
        assert_eq!(diag.severity, Severity::Error);
        assert_eq!(diag.code.number, 1001);
        assert_eq!(diag.phase, Phase::Parser);
    }

    // ── Severity display ───────────────────────────────────────────────────

    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", Severity::Ice), "internal compiler error");
        assert_eq!(format!("{}", Severity::Error), "error");
        assert_eq!(format!("{}", Severity::Warning), "warning");
        assert_eq!(format!("{}", Severity::Note), "note");
        assert_eq!(format!("{}", Severity::Help), "help");
    }

    // ── Phase display ──────────────────────────────────────────────────────

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", Phase::Lexer), "lexer");
        assert_eq!(format!("{}", Phase::BorrowCheck), "borrow-check");
        assert_eq!(format!("{}", Phase::TypeCheck), "type-check");
    }

    // ── Full rendering integration test ────────────────────────────────────

    #[test]
    fn test_full_rendering() {
        let source = "fn main() {\n    let data = compute()\n    process(data)\n    print(data)\n}\n";
        let root_id = DiagnosticId::fresh();

        let diag = Diagnostic::error(4001, Phase::BorrowCheck, test_span(4, 5), "use of moved value `data`")
            .primary_label(test_span(4, 5), "value used after move")
            .secondary_label(test_span(3, 12), "value moved here")
            .fix_certain(test_span(3, 12), "&data", "borrow instead")
            .caused_by(root_id, CausalRelation::RootCause)
            .why("immutable bindings are the default in Jules")
            .dataflow(DataflowInfo {
                defined_at: Some(test_span(2, 8)),
                last_assigned_at: None,
                moved_or_borrowed_at: Some(test_span(3, 12)),
                path_description: "value moved from `data` into `process()`".into(),
            });

        let renderer = DiagnosticRenderer::new(source, "main.jules", RenderConfig { color: false, ..RenderConfig::default() });
        let output = renderer.render(&diag);

        // Verify all sections appear
        assert!(output.contains("error[E4001]"));
        assert!(output.contains("use of moved value"));
        assert!(output.contains("main.jules"));
        assert!(output.contains("help:"));
        assert!(output.contains("certain"));
        assert!(output.contains("dataflow"));
        assert!(output.contains("causality chain"));
    }

    // ── Renderer without color ─────────────────────────────────────────────

    #[test]
    fn test_renderer_no_color() {
        let source = "let x = 1\n";
        let diag = Diagnostic::error(2001, Phase::TypeCheck, test_span(1, 1), "type mismatch");

        let renderer = DiagnosticRenderer::new(source, "test.jules", RenderConfig {
            color: false,
            tab_width: 4,
            context: 1,
            teaching_mode: false,
        });
        let output = renderer.render(&diag);
        // Should not contain ANSI escape codes
        assert!(!output.contains('\x1b'));
        assert!(output.contains("error[E2001]"));
    }

    // ── Collector default ──────────────────────────────────────────────────

    #[test]
    fn test_collector_default() {
        let collector = DiagnosticCollector::default();
        assert_eq!(collector.error_count(), 0);
        assert_eq!(collector.warning_count(), 0);
        assert!(!collector.has_errors());
    }

    // ── Knowledge engine merge ─────────────────────────────────────────────

    #[test]
    fn test_knowledge_engine_merge() {
        let mut engine1 = DiagnosticKnowledgeEngine::new();
        engine1.record_error(4001);
        engine1.record_error(4001);

        let mut engine2 = DiagnosticKnowledgeEngine::new();
        engine2.record_error(2001);
        engine2.record_error(2001);
        engine2.record_error(2001);

        engine1.merge(&engine2);
        assert_eq!(engine1.frequency(4001), 2);
        assert_eq!(engine1.frequency(2001), 3);
    }

    // ── Knowledge engine ambiguity tracking ────────────────────────────────

    #[test]
    fn test_knowledge_engine_ambiguity() {
        let mut engine = DiagnosticKnowledgeEngine::new();
        engine.record_ambiguity("if/else vs match");
        engine.record_ambiguity("if/else vs match");
        engine.record_ambiguity("struct vs component");

        assert_eq!(engine.ambiguity_stats.get("if/else vs match"), Some(&2));
        assert_eq!(engine.ambiguity_stats.get("struct vs component"), Some(&1));
    }

    // ── FixIt convenience constructors ─────────────────────────────────────

    #[test]
    fn test_diagnostic_fix_convenience() {
        let diag = Diagnostic::error(4001, Phase::BorrowCheck, test_span(1, 1), "err")
            .fix_certain(test_span(1, 1), "&x", "borrow")
            .fix_likely(test_span(1, 1), "x.clone()", "clone")
            .fix_possible(test_span(1, 1), "/* fix */", "try this");

        assert_eq!(diag.fixits.len(), 3);
        assert_eq!(diag.fixits[0].confidence, HintConfidence::MachineCertain);
        assert_eq!(diag.fixits[1].confidence, HintConfidence::Likely);
        assert_eq!(diag.fixits[2].confidence, HintConfidence::Possible);
    }

    // ── ICE diagnostic ─────────────────────────────────────────────────────

    #[test]
    fn test_ice_diagnostic() {
        let diag = Diagnostic::ice(0, Phase::Optimizer, test_span(1, 1), "unreachable optimizer state");
        assert_eq!(diag.severity, Severity::Ice);
        assert!(diag.is_fatal());
    }

    // ── Note and Help diagnostics ──────────────────────────────────────────

    #[test]
    fn test_note_diagnostic() {
        let diag = Diagnostic::note(0, Phase::TypeCheck, test_span(1, 1), "additional context");
        assert_eq!(diag.severity, Severity::Note);
        assert!(!diag.is_fatal());
    }

    #[test]
    fn test_warning_diagnostic() {
        let diag = Diagnostic::warning(3001, Phase::SemanticAnalysis, test_span(1, 1), "unused variable");
        assert_eq!(diag.severity, Severity::Warning);
        assert!(!diag.is_fatal());
    }

    // ── Root vs dependent counting ─────────────────────────────────────────

    #[test]
    fn test_root_vs_dependent() {
        let mut collector = DiagnosticCollector::new();
        let root_id = DiagnosticId::fresh();

        let root = Diagnostic::error(4001, Phase::BorrowCheck, test_span(1, 1), "root")
            .caused_by(root_id, CausalRelation::RootCause)
            .suppress_after("data");
        collector.emit(root);

        // The suppressed one won't be emitted, so add a different dependent
        let dep = Diagnostic::error(2001, Phase::TypeCheck, test_span(2, 1), "consequence type error")
            .caused_by(root_id, CausalRelation::Consequence);
        collector.emit(dep);

        let (root_count, dep_count) = collector.root_vs_dependent();
        assert_eq!(root_count, 1); // The first one is root
        assert_eq!(dep_count, 1);  // The second is a consequence
    }

    // ── RecoveryContext matching ────────────────────────────────────────────

    #[test]
    fn test_all_recovery_contexts() {
        let contexts = vec![
            RecoveryContext::ImmutableBinding,
            RecoveryContext::MissingTypeAnnotation,
            RecoveryContext::OwnershipViolation,
            RecoveryContext::EffectViolation,
            RecoveryContext::MissingEffectAnnotation,
            RecoveryContext::UndeclaredVariable,
        ];
        for ctx in contexts {
            let suggestions = RecoverySuggestion::suggestions_for(ctx);
            assert!(!suggestions.is_empty(), "Should have suggestions for {:?}", ctx);
        }
    }

    // ── ErrorExplanation database has correct categories ───────────────────

    #[test]
    fn test_explanation_categories() {
        let db = ErrorExplanationDatabase::new();
        assert_eq!(db.get(1).unwrap().category, "Lexer");
        assert_eq!(db.get(1001).unwrap().category, "Parser");
        assert_eq!(db.get(2001).unwrap().category, "Type Checker");
        assert_eq!(db.get(3001).unwrap().category, "Semantic Analysis");
        assert_eq!(db.get(4001).unwrap().category, "Borrow/Ownership");
        assert_eq!(db.get(5001).unwrap().category, "Effect System");
        assert_eq!(db.get(6001).unwrap().category, "Ownership/Region");
    }

    // ── E4001 detailed explanation ─────────────────────────────────────────

    #[test]
    fn test_e4001_detailed() {
        let db = ErrorExplanationDatabase::new();
        let exp = db.get(4001).unwrap();
        assert!(!exp.examples.is_empty(), "E4001 should have examples");
        assert!(!exp.fixes.is_empty(), "E4001 should have fixes");
        assert!(exp.fixes.iter().any(|f| f.contains("borrow")));
    }
}
