// =============================================================================
// jules/src/compiler/diagnostic.rs
//
// Structured diagnostic subsystem for the Jules compiler.
//
// Design principles:
//   1. One root error, suppress cascading dependent errors
//   2. Every diagnostic knows which phase produced it
//   3. Constraint-aware: explains which semantic rule was violated
//   4. Fix-it suggestions that actually compile
//   5. Source span + secondary spans + cause chain
//   6. Machine-readable JSON for IDEs/LSP
// =============================================================================

use crate::compiler::lexer::Span;
use std::fmt;

// ─── Diagnostic Severity ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    pub fn error(n: u16) -> Self { DiagCode { category: 'E', number: n } }
    pub fn warning(n: u16) -> Self { DiagCode { category: 'W', number: n } }
    pub fn note(n: u16) -> Self { DiagCode { category: 'N', number: n } }
    pub fn ice(n: u16) -> Self { DiagCode { category: 'I', number: n } }
}

impl fmt::Display for DiagCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{:04}", self.category, self.number)
    }
}

// ─── Secondary Label ──────────────────────────────────────────────────────────

/// A secondary span with a message, pointing at related code.
#[derive(Debug, Clone)]
pub struct SecondaryLabel {
    pub span: Span,
    pub message: String,
}

// ─── Fix-It ────────────────────────────────────────────────────────────────────

/// A suggested code replacement.
#[derive(Debug, Clone)]
pub struct FixIt {
    /// The span to replace.
    pub span: Span,
    /// The replacement text.
    pub replacement: String,
    /// Human-readable description of the fix.
    pub description: String,
}

// ─── Structured Diagnostic ────────────────────────────────────────────────────

/// A structured diagnostic with full provenance and fix suggestions.
#[derive(Debug, Clone)]
pub struct StructuredDiag {
    /// Error severity.
    pub severity: Severity,
    /// Structured error code (e.g., E0001, W0042).
    pub code: DiagCode,
    /// Which compiler phase produced this.
    pub phase: Phase,
    /// Primary error message.
    pub message: String,
    /// Primary source span.
    pub span: Span,
    /// Secondary labels pointing at related code.
    pub labels: Vec<SecondaryLabel>,
    /// Cause chain: what led to this error.
    pub cause: Vec<String>,
    /// Fix-it suggestions.
    pub fixits: Vec<FixIt>,
    /// If true, suppress any further errors that depend on this one.
    pub suppress_cascade: bool,
    /// The symbol or type that caused the error (for cascade suppression).
    pub suppressed_symbol: Option<String>,
}

impl StructuredDiag {
    /// Create a new error diagnostic.
    pub fn error(code: u16, phase: Phase, span: Span, message: impl Into<String>) -> Self {
        StructuredDiag {
            severity: Severity::Error,
            code: DiagCode::error(code),
            phase,
            message: message.into(),
            span,
            labels: vec![],
            cause: vec![],
            fixits: vec![],
            suppress_cascade: false,
            suppressed_symbol: None,
        }
    }

    /// Create a new warning diagnostic.
    pub fn warning(code: u16, phase: Phase, span: Span, message: impl Into<String>) -> Self {
        StructuredDiag {
            severity: Severity::Warning,
            code: DiagCode::warning(code),
            phase,
            message: message.into(),
            span,
            labels: vec![],
            cause: vec![],
            fixits: vec![],
            suppress_cascade: false,
            suppressed_symbol: None,
        }
    }

    /// Add a secondary label.
    pub fn label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(SecondaryLabel { span, message: message.into() });
        self
    }

    /// Add a cause to the chain.
    pub fn caused_by(mut self, cause: impl Into<String>) -> Self {
        self.cause.push(cause.into());
        self
    }

    /// Add a fix-it suggestion.
    pub fn fix(mut self, span: Span, replacement: impl Into<String>, desc: impl Into<String>) -> Self {
        self.fixits.push(FixIt {
            span,
            replacement: replacement.into(),
            description: desc.into(),
        });
        self
    }

    /// Mark this diagnostic as cascade-suppressing.
    /// Any subsequent errors that reference `symbol` will be suppressed.
    pub fn suppress_after(mut self, symbol: impl Into<String>) -> Self {
        self.suppress_cascade = true;
        self.suppressed_symbol = Some(symbol.into());
        self
    }

    /// Render this diagnostic as a human-readable string with source context.
    pub fn render(&self, source: &str, filename: &str) -> String {
        let mut out = String::new();
        let severity_str = match self.severity {
            Severity::Ice => "internal compiler error",
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
            Severity::Help => "help",
        };

        out.push_str(&format!("{}[{}]: {} (phase: {})\n", severity_str, self.code, self.message, self.phase));
        out.push_str(&format!("  --> {}:{}\n", filename, self.span));

        // Show source line
        if let Some(line) = source.lines().nth(self.span.line as usize - 1) {
            out.push_str(&format!("   |\n"));
            out.push_str(&format!("{:3}| {}\n", self.span.line, line));
            out.push_str(&format!("   | "));
            for _ in 1..self.span.col {
                out.push(' ');
            }
            let len = (self.span.end - self.span.start).max(1);
            for _ in 0..len {
                out.push('^');
            }
            out.push('\n');
        }

        // Show cause chain
        for cause in &self.cause {
            out.push_str(&format!("  = note: {}\n", cause));
        }

        // Show secondary labels
        for label in &self.labels {
            out.push_str(&format!("  --> {}:{}: {}\n", filename, label.span, label.message));
        }

        // Show fix-its
        for fixit in &self.fixits {
            out.push_str(&format!("  help: {}\n", fixit.description));
            out.push_str(&format!("       replace with: `{}`\n", fixit.replacement));
        }

        // Cascade note
        if self.suppress_cascade {
            if let Some(ref sym) = self.suppressed_symbol {
                out.push_str(&format!("  note: additional errors about `{}` are suppressed\n", sym));
            }
        }

        out
    }
}

// ─── Diagnostic Collector ─────────────────────────────────────────────────────

/// Collects diagnostics with cascade suppression.
#[derive(Debug, Clone)]
pub struct DiagCollector {
    pub diagnostics: Vec<StructuredDiag>,
    /// Symbols that have been flagged as errors — suppress further errors about them.
    suppressed_symbols: Vec<String>,
    /// Total error count.
    error_count: usize,
    /// Total warning count.
    warning_count: usize,
}

impl DiagCollector {
    pub fn new() -> Self {
        DiagCollector {
            diagnostics: vec![],
            suppressed_symbols: vec![],
            error_count: 0,
            warning_count: 0,
        }
    }

    /// Add a diagnostic, applying cascade suppression.
    pub fn emit(&mut self, diag: StructuredDiag) {
        // Check if this error should be suppressed due to a prior root cause.
        if diag.severity == Severity::Error {
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

    /// Render all diagnostics as a human-readable string.
    pub fn render(&self, source: &str, filename: &str) -> String {
        let mut out = String::new();
        for diag in &self.diagnostics {
            out.push_str(&diag.render(source, filename));
            out.push('\n');
        }
        out
    }

    /// Render all diagnostics as JSON for IDE/LSP consumption.
    pub fn to_json(&self, filename: &str) -> String {
        let items: Vec<serde_json::Value> = self.diagnostics.iter().map(|d| {
            serde_json::json!({
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
                "labels": d.labels.iter().map(|l| serde_json::json!({
                    "message": l.message,
                    "span": { "start": l.span.start, "end": l.span.end, "line": l.span.line, "column": l.span.col }
                })).collect::<Vec<_>>(),
                "fixits": d.fixits.iter().map(|f| serde_json::json!({
                    "description": f.description,
                    "replacement": f.replacement,
                    "span": { "start": f.span.start, "end": f.span.end, "line": f.span.line, "column": f.span.col }
                })).collect::<Vec<_>>(),
                "cause": d.cause,
                "file": filename,
            })
        }).collect();
        serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string())
    }
}

impl Default for DiagCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Well-known error codes ───────────────────────────────────────────────────

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
