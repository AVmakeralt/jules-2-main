// =============================================================================
// Shared lightweight diagnostic types (issue #33 — consolidation)
//
// Previously, `Severity`, `Diagnostic`, and `Diagnostics` were duplicated
// across typeck.rs, borrowck.rs, and sema.rs with only minor field-name
// differences (`notes` vs `labels`).  All three passes now use the single
// definition below, re-exported from each module for backward compatibility.
// =============================================================================

use crate::compiler::lexer::Span;

/// Severity of a diagnostic message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimpleSeverity {
    Error,
    Warning,
    Note,
}

/// A single span-aware diagnostic produced by any lightweight compiler pass.
#[derive(Debug, Clone)]
pub struct SimpleDiagnostic {
    pub severity: SimpleSeverity,
    pub span: Span,
    pub message: String,
    /// Secondary labels / notes pointing at related source locations.
    pub labels: Vec<(Span, String)>,
    /// Error code (e.g. "E2001") from the error_codes module.
    pub code: Option<&'static str>,
    /// Suggested fix hint (optional, displayed as `help:` in the renderer).
    pub hint: Option<String>,
}

impl SimpleDiagnostic {
    pub fn error(span: Span, msg: impl Into<String>) -> Self {
        SimpleDiagnostic {
            severity: SimpleSeverity::Error,
            span,
            message: msg.into(),
            labels: vec![],
            code: None,
            hint: None,
        }
    }
    pub fn warning(span: Span, msg: impl Into<String>) -> Self {
        SimpleDiagnostic {
            severity: SimpleSeverity::Warning,
            span,
            message: msg.into(),
            labels: vec![],
            code: None,
            hint: None,
        }
    }
    pub fn note(span: Span, msg: impl Into<String>) -> Self {
        SimpleDiagnostic {
            severity: SimpleSeverity::Note,
            span,
            message: msg.into(),
            labels: vec![],
            code: None,
            hint: None,
        }
    }
    /// Attach a secondary label to a different source location.
    pub fn with_label(mut self, span: Span, msg: impl Into<String>) -> Self {
        self.labels.push((span, msg.into()));
        self
    }
    /// Alias for `with_label` — used by typeck which historically called them "notes".
    pub fn with_note(self, span: Span, msg: impl Into<String>) -> Self {
        self.with_label(span, msg)
    }
    /// Attach an error code (e.g. `"E2001"`).
    pub fn with_code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }
    /// Attach a suggested fix hint.
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
    pub fn is_fatal(&self) -> bool {
        self.severity == SimpleSeverity::Error
    }
}

/// Collects all diagnostics emitted during a compiler pass.
#[derive(Debug, Default)]
pub struct SimpleDiagnostics {
    pub items: Vec<SimpleDiagnostic>,
}

impl SimpleDiagnostics {
    pub fn push(&mut self, d: SimpleDiagnostic) {
        self.items.push(d);
    }
    pub fn error(&mut self, span: Span, msg: impl Into<String>) {
        self.push(SimpleDiagnostic::error(span, msg));
    }
    pub fn warning(&mut self, span: Span, msg: impl Into<String>) {
        self.push(SimpleDiagnostic::warning(span, msg));
    }
    pub fn note(&mut self, span: Span, msg: impl Into<String>) {
        self.push(SimpleDiagnostic::note(span, msg));
    }
    pub fn has_errors(&self) -> bool {
        self.items.iter().any(|d| d.is_fatal())
    }
    pub fn error_count(&self) -> usize {
        self.items.iter().filter(|d| d.is_fatal()).count()
    }
}

// ── Module declarations ────────────────────────────────────────────────────────

pub mod ast;
pub mod borrowck;
pub mod diagnostic;
pub mod diff_opt;
pub mod error_codes;
pub mod formal_verify;
pub mod hw_feedback;
pub mod ir;
pub mod ir_to_bytecode;
pub mod ir_borrowck;
pub mod ir_typeck;
pub mod lexer;
pub mod lower;
pub mod loom_model_check;
pub mod parser;
pub mod sat_smt_solver;
pub mod sema;
pub mod translation_validation;
pub mod typeck;
