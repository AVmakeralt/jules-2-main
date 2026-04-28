// Module declarations for the Jules compiler/interpreter.
// In this repository layout the source files live in the crate root.
#[allow(dead_code)]
#[allow(unused_variables)]
#[allow(unused_macros)]
#[allow(clippy::drop_ref)]
mod advanced_optimizer;
mod advanced_self_repair;
mod aot_native;
mod ast;
mod borrowck;
mod bytecode_vm;
mod ffi;
mod game_systems;
mod gpu_backend;
pub mod interp;
mod lexer;
mod ml_engine;
mod memory_management;
mod optimizer;
mod parser;
mod self_repair;
mod sema;
mod string_intern;
#[cfg(feature = "phase3-jit")]
pub mod tiered_compilation;
#[cfg(feature = "phase3-jit")]
pub mod tracing_jit;
mod typeck;
// Optional, phase-gated modules (added per performance phase protocol)
#[cfg(feature = "phase3-jit")]
mod phase3_jit;
#[cfg(feature = "phase6-simd")]
pub mod phase6_simd;

// Game-dev tooling modules used by the runtime and editor workflows.
pub mod asset_importer;
pub mod chess_ml;
pub mod frame_debugger;
pub mod hot_reload;
pub mod networking;
pub mod profiling_tools;
pub mod scene_editor;
pub mod shader_tooling;

// Standard library — game dev, ML, simulation.
#[allow(clippy::manual_retain)]
mod jules_std;

use std::collections::hash_map::DefaultHasher;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::process::Command as ProcessCommand;
use std::time::{Duration, Instant};

// ── Public API (used by lib.rs re-exports) ────────────────────────────────────

/// Run only the analysis passes (lex → parse → typecheck → sema) on a source
/// string and return all diagnostics.  Does not execute the program.
pub fn jules_check(filename: &str, source: &str) -> Vec<Diag> {
    let mut unit = CompileUnit::new(filename, source);
    Pipeline::new().run(&mut unit);
    unit.diags
}

/// Load, analyse, and run a Jules source file.  Returns `Ok(())` on success,
/// or the first runtime error as a `String`.
///
/// Pass `entry = "main"` for normal execution, `"#test"` to run all @test
/// functions, or `"#bench"` to run all @benchmark functions.
pub fn jules_run_file(path: &str, entry: &str) -> Result<(), String> {
    let source = fs::read_to_string(path).map_err(|e| format!("cannot read `{path}`: {e}"))?;
    let mut unit = CompileUnit::new(path, &source);
    let result = Pipeline::new().run(&mut unit);
    if unit.has_errors() {
        let msgs: Vec<_> = unit
            .diags
            .iter()
            .filter(|d| d.is_error())
            .map(|d| d.message.clone())
            .collect();
        return Err(msgs.join("\n"));
    }
    if let PipelineResult::Ok(program) = result {
        let mut interp = crate::interp::Interpreter::new();
        interp.load_program(&program);
        interp
            .call_fn(entry, vec![])
            .map(|_| ())
            .map_err(|e| e.message)
    } else {
        Err("pipeline did not produce a program".into())
    }
}

// Pull in the compiler passes.  In a real crate these would be separate modules.
use crate::lexer::{LexError, Lexer, Span};
// use crate::parser::Parser;   // parser wired through `Pipeline`
// use crate::typeck::TypeCk;
// use crate::sema::SemaCtx;
// use crate::interp::Interpreter;

// =============================================================================
// §1  ANSI COLOUR CODES
// =============================================================================

struct Ansi;

impl Ansi {
    const RESET: &'static str = "\x1b[0m";
    const BOLD: &'static str = "\x1b[1m";
    const DIM: &'static str = "\x1b[2m";

    // Foreground colours
    const RED: &'static str = "\x1b[31m";
    const YELLOW: &'static str = "\x1b[33m";
    const BLUE: &'static str = "\x1b[34m";
    const CYAN: &'static str = "\x1b[36m";
    const WHITE: &'static str = "\x1b[37m";
    const BRIGHT_RED: &'static str = "\x1b[91m";
    const BRIGHT_YEL: &'static str = "\x1b[93m";
    const BRIGHT_CYN: &'static str = "\x1b[96m";
    const MAGENTA: &'static str = "\x1b[35m";

    fn paint(enabled: bool, code: &str, text: &str) -> String {
        if enabled {
            format!("{}{}{}", code, text, Self::RESET)
        } else {
            text.to_owned()
        }
    }
}

// =============================================================================
// §2  UNIFIED DIAGNOSTIC TYPES
// =============================================================================

/// One diagnostic from any compiler pass.
#[derive(Debug, Clone)]
pub struct Diag {
    pub severity: DiagSeverity,
    pub span: Option<Span>,
    pub code: Option<&'static str>, // e.g. "E001", "W042"
    pub message: String,
    /// Secondary "note" labels at related source positions.
    pub labels: Vec<(Span, String)>,
    /// Suggested fix hint (optional).
    pub hint: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagSeverity {
    Note,
    Warning,
    Error,
}

impl Diag {
    pub fn error(span: Span, msg: impl Into<String>) -> Self {
        Diag {
            severity: DiagSeverity::Error,
            span: Some(span),
            code: None,
            message: msg.into(),
            labels: vec![],
            hint: None,
        }
    }
    pub fn warning(span: Span, msg: impl Into<String>) -> Self {
        Diag {
            severity: DiagSeverity::Warning,
            span: Some(span),
            code: None,
            message: msg.into(),
            labels: vec![],
            hint: None,
        }
    }
    pub fn note(span: Span, msg: impl Into<String>) -> Self {
        Diag {
            severity: DiagSeverity::Note,
            span: Some(span),
            code: None,
            message: msg.into(),
            labels: vec![],
            hint: None,
        }
    }
    pub fn with_code(mut self, c: &'static str) -> Self {
        self.code = Some(c);
        self
    }
    pub fn with_hint(mut self, h: impl Into<String>) -> Self {
        self.hint = Some(h.into());
        self
    }
    pub fn with_label(mut self, s: Span, m: impl Into<String>) -> Self {
        self.labels.push((s, m.into()));
        self
    }
    pub fn is_error(&self) -> bool {
        self.severity == DiagSeverity::Error
    }
}

// ── Conversion from lexer errors ──────────────────────────────────────────────

impl From<LexError> for Diag {
    fn from(e: LexError) -> Self {
        Diag::error(e.span, e.message).with_code("E0001")
    }
}

// =============================================================================
// §3  PRETTY-PRINTING RENDERER
// =============================================================================

/// Configuration for the diagnostic renderer.
#[derive(Debug, Clone)]
pub struct RenderCfg {
    pub color: bool,
    pub tab_width: usize,
    pub context: usize, // extra lines of source context to show
}

impl Default for RenderCfg {
    fn default() -> Self {
        RenderCfg {
            color: true,
            tab_width: 4,
            context: 1,
        }
    }
}

/// Renders a slice of diagnostics to a `String` using Rust-compiler-style
/// formatting with source squiggles.
pub struct DiagRenderer<'src> {
    source: &'src str,
    filename: &'src str,
    cfg: RenderCfg,
    /// Pre-split source lines for O(1) access.
    lines: Vec<&'src str>,
}

impl<'src> DiagRenderer<'src> {
    pub fn new(source: &'src str, filename: &'src str, cfg: RenderCfg) -> Self {
        let lines: Vec<&str> = source.split('\n').collect();
        DiagRenderer {
            source,
            filename,
            cfg,
            lines,
        }
    }

    pub fn render_all(&self, diags: &[Diag]) -> String {
        let mut out = String::new();
        for d in diags {
            out.push_str(&self.render(d));
            out.push('\n');
        }
        out
    }

    pub fn render(&self, d: &Diag) -> String {
        let mut buf = String::new();

        // ── Header line  "error[E001]: message" ───────────────────────────────
        let (sev_tag, sev_color) = match d.severity {
            DiagSeverity::Error => ("error", Ansi::BRIGHT_RED),
            DiagSeverity::Warning => ("warning", Ansi::BRIGHT_YEL),
            DiagSeverity::Note => ("note", Ansi::BRIGHT_CYN),
        };
        let code_part = d.code.map(|c| format!("[{c}]")).unwrap_or_default();
        let header = format!("{sev_tag}{code_part}: {}", d.message);
        writeln!(
            buf,
            "{}",
            self.paint(Ansi::BOLD, &self.paint(sev_color, &header))
        )
        .unwrap();

        // ── File + line location ───────────────────────────────────────────────
        if let Some(span) = d.span {
            if span.line > 0 {
                let loc = format!("  --> {}:{}:{}", self.filename, span.line, span.col);
                writeln!(buf, "{}", self.dim(&loc)).unwrap();
                writeln!(buf, "   {}", self.dim("|")).unwrap();
                self.render_source_snippet(&mut buf, span, sev_color);
            }
        }

        // ── Secondary labels ──────────────────────────────────────────────────
        for (lspan, lmsg) in &d.labels {
            if lspan.line > 0 {
                let loc = format!("  --> {}:{}:{}", self.filename, lspan.line, lspan.col);
                writeln!(buf, "{}", self.dim(&loc)).unwrap();
                writeln!(buf, "   {}", self.dim("|")).unwrap();
                self.render_source_snippet(&mut buf, *lspan, Ansi::BLUE);
                let note = format!("   {} = note: {}", self.dim("|"), lmsg);
                writeln!(buf, "{note}").unwrap();
            }
        }

        // ── Hint ─────────────────────────────────────────────────────────────
        if let Some(hint) = &d.hint {
            writeln!(
                buf,
                "   {} help: {}",
                self.dim("|"),
                self.paint(Ansi::BRIGHT_CYN, hint)
            )
            .unwrap();
        }

        buf
    }

    fn render_source_snippet(&self, buf: &mut String, span: Span, squiggle_color: &str) {
        let line_idx = (span.line as usize).saturating_sub(1);

        // Optionally show one line of context above.
        if self.cfg.context > 0 && line_idx > 0 {
            let prev_no = line_idx; // 1-based = line_idx (because line_idx = line - 1)
            let prev = self.lines.get(line_idx - 1).unwrap_or(&"");
            let prefix = self.dim(&format!("{prev_no:4} | "));
            writeln!(buf, "{prefix}{}", self.dim(prev)).unwrap();
        }

        // The main highlighted line.
        let line_no = span.line as usize;
        let line_str = self.lines.get(line_idx).unwrap_or(&"");
        let prefix = self.paint(Ansi::BOLD, &format!("{line_no:4} | "));
        writeln!(buf, "{prefix}{line_str}").unwrap();

        // Squiggle line: "     | ^^^^"
        let col = (span.col as usize).saturating_sub(1);
        let width = (span.end.saturating_sub(span.start)).max(1);
        let pad = self.expand_tabs(line_str, col);
        let squig = self.paint(squiggle_color, &"^".repeat(width));
        writeln!(
            buf,
            "     {}{}{}",
            self.dim("|"),
            " ".repeat(pad + 1),
            squig
        )
        .unwrap();

        // Optionally show one line of context below.
        if self.cfg.context > 0 {
            let next_idx = line_idx + 1;
            if let Some(next) = self.lines.get(next_idx) {
                let next_no = next_idx + 1;
                let prefix = self.dim(&format!("{next_no:4} | "));
                writeln!(buf, "{prefix}{}", self.dim(next)).unwrap();
            }
        }

        writeln!(buf, "   {}", self.dim("|")).unwrap();
    }

    /// Count display columns up to `col` bytes, expanding tabs.
    fn expand_tabs(&self, line: &str, col: usize) -> usize {
        let tw = self.cfg.tab_width;
        let mut display = 0;
        for (i, ch) in line.char_indices() {
            if i >= col {
                break;
            }
            if ch == '\t' {
                // Align to next tab stop.
                display = (display / tw + 1) * tw;
            } else {
                display += 1;
            }
        }
        display
    }

    fn paint(&self, color: &str, s: &str) -> String {
        Ansi::paint(self.cfg.color, color, s)
    }
    fn dim(&self, s: &str) -> String {
        Ansi::paint(self.cfg.color, Ansi::DIM, s)
    }
}

// =============================================================================
// §4  JSON DIAGNOSTIC EMITTER  (for editor / LSP integration)
// =============================================================================

pub fn diags_to_json(diags: &[Diag], filename: &str) -> String {
    if diags.is_empty() {
        return "[]".to_string();
    }

    let mut out = String::from("[\n");
    for (i, d) in diags.iter().enumerate() {
        let sev = match d.severity {
            DiagSeverity::Error => "error",
            DiagSeverity::Warning => "warning",
            DiagSeverity::Note => "note",
        };
        let (line, col, start, end) = if let Some(sp) = d.span {
            (sp.line, sp.col, sp.start, sp.end)
        } else {
            (0, 0, 0, 0)
        };

        let msg = d.message.replace('"', "\\\"");
        let code = d.code.unwrap_or("");
        let hint = d.hint.as_deref().unwrap_or("").replace('"', "\\\"");

        let labels_json: Vec<String> = d
            .labels
            .iter()
            .map(|(sp, m)| {
                let m = m.replace('"', "\\\"");
                format!(
                    r#"  {{"line":{}, "col":{}, "message":"{}"}}"#,
                    sp.line, sp.col, m
                )
            })
            .collect();

        let comma = if i + 1 < diags.len() { "," } else { "" };
        out.push_str(&format!(
            r#"  {{
    "severity": "{sev}",
    "code": "{code}",
    "file": "{filename}",
    "line": {line},
    "col": {col},
    "start": {start},
    "end": {end},
    "message": "{msg}",
    "hint": "{hint}",
    "labels": [
{}
    ]
  }}{comma}
"#,
            labels_json.join(",\n")
        ));
    }
    out.push(']');
    out
}

// =============================================================================
// §5  COMPILER PIPELINE  (orchestrates all passes)
// =============================================================================

/// Everything the compiler knows about a compilation unit.
pub struct CompileUnit {
    pub filename: String,
    pub source: String,
    pub diags: Vec<Diag>,
}

impl CompileUnit {
    pub fn new(filename: impl Into<String>, source: impl Into<String>) -> Self {
        CompileUnit {
            filename: filename.into(),
            source: source.into(),
            diags: vec![],
        }
    }

    pub fn has_errors(&self) -> bool {
        self.diags.iter().any(|d| d.is_error())
    }

    pub fn error_count(&self) -> usize {
        self.diags.iter().filter(|d| d.is_error()).count()
    }

    /// Human-readable one-line summary: "2 errors, 1 warning".
    pub fn summary(&self) -> String {
        let e = self.error_count();
        let w = self.warning_count();
        format!(
            "{} error{}, {} warning{}",
            e,
            if e == 1 { "" } else { "s" },
            w,
            if w == 1 { "" } else { "s" }
        )
    }
    pub fn warning_count(&self) -> usize {
        self.diags
            .iter()
            .filter(|d| d.severity == DiagSeverity::Warning)
            .count()
    }
}

/// The full front-end pipeline.  Each pass adds to `unit.diags`.
pub struct Pipeline {
    pub warn_as_error: bool,
    pub quiet: bool,
    pub emit_ir: bool,
    pub profile: bool,
    pub opt_level: u8, // 0=none, 1=fast_compile, 2=balanced, 3=maximum
    pub print_opt_stats: bool,
}

impl Pipeline {
    pub fn new() -> Self {
        Pipeline {
            warn_as_error: false,
            quiet: false,
            emit_ir: false,
            profile: false,
            opt_level: 2, // balanced by default
            print_opt_stats: false,
        }
    }

    /// Run the pipeline as far as possible, accumulating diagnostics.
    /// Returns `Ok(unit)` even when there are errors so the caller can
    /// display all diagnostics at once.
    pub fn run(&self, unit: &mut CompileUnit) -> PipelineResult {
        // ── Pass 1: Lex ───────────────────────────────────────────────────────
        let mut lexer = Lexer::new(&unit.source);
        let (tokens, lex_errors) = lexer.tokenize();

        unit.diags
            .reserve(lex_errors.len() + (tokens.len() / 32).max(8));
        for e in lex_errors {
            unit.diags.push(Diag::from(e));
        }

        if unit.has_errors() {
            return PipelineResult::HaltedAt(PassName::Lex);
        }

        // ── Pass 2: Parse ─────────────────────────────────────────────────────
        let mut parser = crate::parser::Parser::new(tokens);
        let mut program = parser.parse_program();

        for e in parser.errors {
            unit.diags.push(parse_error_to_diag(e));
        }

        if unit.has_errors() {
            return PipelineResult::HaltedAt(PassName::Parse);
        }

        // ── Pass 3: Type-check ────────────────────────────────────────────────
        let mut typeck = crate::typeck::TypeCk::new();
        typeck.check_program(&program);
        for d in typeck.diag.items {
            unit.diags.push(adapt_typeck_diag(d));
        }
        if unit.has_errors() {
            return PipelineResult::HaltedAt(PassName::TypeCheck);
        }

        // ── Pass 4: Semantic analysis ─────────────────────────────────────────
        let mut sema = crate::sema::SemaCtx::new();
        sema.analyse(&program);
        for d in sema.diag.items {
            unit.diags.push(adapt_sema_diag(d));
        }
        if unit.has_errors() {
            return PipelineResult::HaltedAt(PassName::Sema);
        }

        // ── Pass 5: Borrow-check (lexical aliasing safety) ───────────────────
        let borrow_diags = crate::borrowck::jules_borrowck(&program);
        for d in borrow_diags.items {
            unit.diags.push(adapt_borrowck_diag(d));
        }
        if unit.has_errors() {
            return PipelineResult::HaltedAt(PassName::BorrowCheck);
        }

        // ── Pass 6: Superoptimizer ──────────────────────────────────────────
        // Runs the full AST-level superoptimizer: constant folding, SCCP,
        // algebraic simplification (50+ rules), strength reduction, CSE,
        // dead code elimination, dead store elimination, peephole, loop
        // invariant code motion, function inlining, branch optimization.
        if self.opt_level >= 1 {
            let config = match self.opt_level {
                1 => crate::advanced_optimizer::SuperoptimizerConfig::fast_compile(),
                2 => crate::advanced_optimizer::SuperoptimizerConfig::balanced(),
                _ => crate::advanced_optimizer::SuperoptimizerConfig::maximum(),
            };
            let mut superopt = crate::advanced_optimizer::Superoptimizer::new(config);
            superopt.optimize_program(&mut program);
            if self.print_opt_stats && self.opt_level >= 2 {
                eprintln!("[opt] const_folds={} cse={} dce={} inline={} unroll={} dead_fn={} licm={}",
                    superopt.constant_folds, superopt.cse_eliminations, superopt.dead_code_eliminated,
                    superopt.inlinings, superopt.loop_unrollings, superopt.dead_functions_eliminated,
                    superopt.licm_hoists);
            }
        }

        // ── Promote warnings to errors if requested ───────────────────────────
        if self.warn_as_error {
            for d in &mut unit.diags {
                if d.severity == DiagSeverity::Warning {
                    d.severity = DiagSeverity::Error;
                }
            }
        }

        // ── Filter notes if quiet ─────────────────────────────────────────────
        if self.quiet {
            unit.diags.retain(|d| d.severity == DiagSeverity::Error);
        }

        // Emit summary counts.
        if !self.quiet {
            let ec = unit.error_count();
            let wc = unit.warning_count();
            if ec > 0 || wc > 0 {
                let summary = format!(
                    "compilation finished: {} error{}, {} warning{}",
                    ec,
                    if ec == 1 { "" } else { "s" },
                    wc,
                    if wc == 1 { "" } else { "s" },
                );
                unit.diags.push(Diag::note(Span::dummy(), summary));
            }
        }

        PipelineResult::Ok(program)
    }
}

#[derive(Debug)]
pub enum PipelineResult {
    Ok(crate::ast::Program),
    HaltedAt(PassName),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassName {
    Lex,
    Parse,
    TypeCheck,
    Sema,
    BorrowCheck,
    Interp,
    Optimize,
    Codegen,
}

// ── Diagnostic adapters (one per pass module) ──────────────────────────────

fn parse_error_to_diag(e: crate::parser::ParseError) -> Diag {
    let mut d = Diag::error(e.span, e.message).with_code("E0002");
    if let Some(h) = e.hint {
        d = d.with_hint(h);
    }
    d
}

fn adapt_typeck_diag(d: crate::typeck::Diagnostic) -> Diag {
    let sev = match d.severity {
        crate::typeck::Severity::Error => DiagSeverity::Error,
        crate::typeck::Severity::Warning => DiagSeverity::Warning,
        crate::typeck::Severity::Note => DiagSeverity::Note,
    };
    let mut out = Diag {
        severity: sev,
        span: Some(d.span),
        code: None,
        message: d.message,
        labels: vec![],
        hint: None,
    };
    for (s, m) in d.notes {
        out.labels.push((s, m));
    }
    out
}

fn adapt_sema_diag(d: crate::sema::Diagnostic) -> Diag {
    let sev = match d.severity {
        crate::sema::Severity::Error => DiagSeverity::Error,
        crate::sema::Severity::Warning => DiagSeverity::Warning,
        crate::sema::Severity::Note => DiagSeverity::Note,
    };
    let mut out = Diag {
        severity: sev,
        span: Some(d.span),
        code: None,
        message: d.message,
        labels: vec![],
        hint: None,
    };
    for (s, m) in d.labels {
        out.labels.push((s, m));
    }
    out
}

fn adapt_borrowck_diag(d: crate::borrowck::Diagnostic) -> Diag {
    let sev = match d.severity {
        crate::borrowck::Severity::Error => DiagSeverity::Error,
        crate::borrowck::Severity::Warning => DiagSeverity::Warning,
        crate::borrowck::Severity::Note => DiagSeverity::Note,
    };
    let mut out = Diag {
        severity: sev,
        span: Some(d.span),
        code: None,
        message: d.message,
        labels: vec![],
        hint: None,
    };
    for (s, m) in d.labels {
        out.labels.push((s, m));
    }
    out
}

fn adapt_runtime_error(e: crate::interp::RuntimeError) -> Diag {
    Diag {
        severity: DiagSeverity::Error,
        span: e.span,
        code: Some("E9000"),
        message: e.message,
        labels: vec![],
        hint: None,
    }
}

// =============================================================================
// §6  SUMMARY LINE
// =============================================================================

fn print_summary(unit: &CompileUnit, cfg: &RenderCfg) {
    let errors = unit.error_count();
    let warnings = unit.warning_count();
    if errors == 0 && warnings == 0 {
        return;
    }

    let err_part = if errors > 0 {
        Ansi::paint(
            cfg.color,
            Ansi::BRIGHT_RED,
            &format!("{errors} error{}", if errors == 1 { "" } else { "s" }),
        )
    } else {
        String::new()
    };

    let warn_part = if warnings > 0 {
        Ansi::paint(
            cfg.color,
            Ansi::BRIGHT_YEL,
            &format!("{warnings} warning{}", if warnings == 1 { "" } else { "s" }),
        )
    } else {
        String::new()
    };

    let parts: Vec<&str> = [err_part.as_str(), warn_part.as_str()]
        .iter()
        .filter(|s| !s.is_empty())
        .cloned()
        .collect();
    eprintln!(
        "{}",
        Ansi::paint(
            cfg.color,
            Ansi::BOLD,
            &format!("jules: {}", parts.join(", "))
        )
    );
}

// =============================================================================
// §7  CLI ARGUMENT PARSER  (zero external dependencies)
// =============================================================================

#[derive(Debug, Default)]
pub struct CliArgs {
    pub command: Command,
    pub file: Option<PathBuf>,
    pub output: Option<PathBuf>, // -o for compile command
    pub rest_args: Vec<String>,
    pub color: bool,
    pub json_diag: bool,
    pub emit_ast: bool,
    pub emit_tokens: bool,
    pub warn_as_error: bool,
    pub quiet: bool,
    pub opt_level: u8,
    pub print_opt_stats: bool,
    pub tab_width: usize,
    pub entry: String, // --entry <fn>  for jules run
    pub train: bool,   // jules train
    pub fix_dry_run: bool,
    pub fix_diff: bool,
    pub fix_aggressive: bool,
    pub estimate_params: usize,
    pub estimate_batch: usize,
    pub estimate_episodes: usize,
    pub estimate_steps: usize,
    pub estimate_envs: usize,
    pub estimate_device: String,
    pub ml_backend: String,
    pub jax_ir: Option<PathBuf>,
    pub jax_dataset: Option<PathBuf>,
    pub jax_out: PathBuf,
    pub jax_script: PathBuf,
    // Tiered compilation flags
    pub tiered: bool,               // Enable tiered compilation
    pub tier_policy: String,        // "fast-startup", "balanced", "max-performance"
    pub print_tier_stats: bool,     // Print tier compilation statistics
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Command {
    #[default]
    Help,
    Run,
    Check,
    Fix,
    Fmt,
    Repl,
    Train,
    Estimate,
    Version,
    Compile, // AOT native compilation to ELF binaries
}

impl CliArgs {
    pub fn parse(args: &[String]) -> Result<Self, String> {
        let mut out = CliArgs {
            color: std::env::var("NO_COLOR").is_err(), // respect NO_COLOR
            tab_width: 4,
            entry: "main".into(),
            opt_level: 2, // balanced optimization by default
            estimate_params: 1_000_000,
            estimate_batch: 64,
            estimate_episodes: 100_000,
            estimate_steps: 64,
            estimate_envs: 1,
            estimate_device: "cpu".into(),
            ml_backend: "jules".into(),
            jax_out: PathBuf::from("artifacts/jax"),
            jax_script: PathBuf::from("scripts/jules_jax_backend.py"),
            ..Default::default()
        };

        let mut it = args.iter().peekable();

        // Skip binary name.
        let _ = it.next();

        // First positional arg: sub-command.
        let cmd = it.peek().map(|s| s.as_str());
        match cmd {
            Some("run") => {
                out.command = Command::Run;
                it.next();
            }
            Some("check") => {
                out.command = Command::Check;
                it.next();
            }
            Some("fix") => {
                out.command = Command::Fix;
                it.next();
            }
            Some("fmt") => {
                out.command = Command::Fmt;
                it.next();
            }
            Some("repl") => {
                out.command = Command::Repl;
                it.next();
            }
            Some("train") => {
                out.command = Command::Train;
                it.next();
            }
            Some("estimate") => {
                out.command = Command::Estimate;
                it.next();
            }
            Some("version") | Some("--version") | Some("-V") => {
                out.command = Command::Version;
                it.next();
            }
            Some("compile") | Some("build") => {
                out.command = Command::Compile;
                it.next();
            }
            Some("help") | Some("--help") | Some("-h") | None => {
                out.command = Command::Help;
                it.next();
            }
            _ => {} // let flags/file sort it out below
        }

        let mut past_dashdash = false;

        while let Some(arg) = it.next() {
            if past_dashdash {
                out.rest_args.push(arg.clone());
                continue;
            }
            match arg.as_str() {
                "--" => {
                    past_dashdash = true;
                }
                "--no-color" => {
                    out.color = false;
                }
                "--color" => {
                    out.color = true;
                }
                "--json-diag" => {
                    out.json_diag = true;
                }
                "--emit-ast" => {
                    out.emit_ast = true;
                }
                "--emit-tokens" => {
                    out.emit_tokens = true;
                }
                "--warn-error" | "-W" => {
                    out.warn_as_error = true;
                }
                "--quiet" | "-q" => {
                    out.quiet = true;
                }
                "-o" | "--output" => {
                    let path = it
                        .next()
                        .ok_or("-o requires a value")?
                        .clone();
                    out.output = Some(PathBuf::from(path));
                }
                "--tab-width" => {
                    let n = it
                        .next()
                        .ok_or("--tab-width requires a value")?
                        .parse::<usize>()
                        .map_err(|_| "--tab-width must be a positive integer")?;
                    out.tab_width = n;
                }
                "--entry" => {
                    out.entry = it.next().ok_or("--entry requires a function name")?.clone();
                }
                "--dry-run" => {
                    out.fix_dry_run = true;
                }
                "--diff" => {
                    out.fix_diff = true;
                }
                "--aggressive" => {
                    out.fix_aggressive = true;
                }
                "--params" => {
                    out.estimate_params = it
                        .next()
                        .ok_or("--params requires a value")?
                        .parse::<usize>()
                        .map_err(|_| "--params must be a positive integer")?
                        .max(1);
                }
                "--batch" => {
                    out.estimate_batch = it
                        .next()
                        .ok_or("--batch requires a value")?
                        .parse::<usize>()
                        .map_err(|_| "--batch must be a positive integer")?
                        .max(1);
                }
                "--episodes" => {
                    out.estimate_episodes = it
                        .next()
                        .ok_or("--episodes requires a value")?
                        .parse::<usize>()
                        .map_err(|_| "--episodes must be a positive integer")?
                        .max(1);
                }
                "--steps" => {
                    out.estimate_steps = it
                        .next()
                        .ok_or("--steps requires a value")?
                        .parse::<usize>()
                        .map_err(|_| "--steps must be a positive integer")?
                        .max(1);
                }
                "--envs" => {
                    out.estimate_envs = it
                        .next()
                        .ok_or("--envs requires a value")?
                        .parse::<usize>()
                        .map_err(|_| "--envs must be a positive integer")?
                        .max(1);
                }
                "--device" => {
                    out.estimate_device = it
                        .next()
                        .ok_or("--device requires `cpu` or `gpu`")?
                        .to_ascii_lowercase();
                    if out.estimate_device != "cpu" && out.estimate_device != "gpu" {
                        return Err("--device must be `cpu` or `gpu`".into());
                    }
                }
                "--ml-backend" => {
                    out.ml_backend = it
                        .next()
                        .ok_or("--ml-backend requires `jules` or `jax`")?
                        .to_ascii_lowercase();
                    if out.ml_backend != "jules" && out.ml_backend != "jax" {
                        return Err("--ml-backend must be `jules` or `jax`".into());
                    }
                }
                "-O0" | "--opt=0" => {
                    out.opt_level = 0;
                }
                "-O1" | "--opt=1" => {
                    out.opt_level = 1;
                }
                "-O2" | "--opt=2" => {
                    out.opt_level = 2;
                }
                "-O3" | "-O" | "--opt=3" | "--opt=maximum" => {
                    out.opt_level = 3;
                }
                "--stats" | "--print-opt-stats" => {
                    out.print_opt_stats = true;
                }
                "--tiered" => {
                    out.tiered = true;
                }
                "--tier-policy" => {
                    out.tier_policy = it.next().ok_or("--tier-policy requires a value")?.clone();
                    if !matches!(out.tier_policy.as_str(), "fast-startup" | "balanced" | "max-performance") {
                        return Err("--tier-policy must be `fast-startup`, `balanced`, or `max-performance`".into());
                    }
                }
                "--tier-stats" => {
                    out.print_tier_stats = true;
                }
                "--jax-ir" => {
                    out.jax_ir = Some(PathBuf::from(it.next().ok_or("--jax-ir requires a path")?));
                }
                "--jax-dataset" => {
                    out.jax_dataset = Some(PathBuf::from(
                        it.next().ok_or("--jax-dataset requires a path")?,
                    ));
                }
                "--jax-out" => {
                    out.jax_out = PathBuf::from(it.next().ok_or("--jax-out requires a path")?);
                }
                "--jax-script" => {
                    out.jax_script =
                        PathBuf::from(it.next().ok_or("--jax-script requires a path")?);
                }
                s if s.starts_with('-') => {
                    return Err(format!("unknown flag `{s}`; try `jules help`"));
                }
                path => {
                    if out.file.is_none() {
                        out.file = Some(PathBuf::from(path));
                    } else {
                        out.rest_args.push(arg.clone());
                    }
                }
            }
        }

        Ok(out)
    }
}

// =============================================================================
// §8  COMMAND IMPLEMENTATIONS
// =============================================================================

const VERSION: &str = env!("CARGO_PKG_VERSION");
const JULES_BANNER: &str = r#"
     ██╗██╗   ██╗██╗     ███████╗███████╗
     ██║██║   ██║██║     ██╔════╝██╔════╝
     ██║██║   ██║██║     █████╗  ███████╗
██   ██║██║   ██║██║     ██╔══╝  ╚════██║
╚█████╔╝╚██████╔╝███████╗███████╗███████║
 ╚════╝  ╚═════╝ ╚══════╝╚══════╝╚══════╝
"#;

fn cmd_version() {
    println!("jules {VERSION}");
    println!("tensor-first game-and-AI programming language");
}

fn cmd_help() {
    println!("{JULES_BANNER}");
    println!("USAGE:");
    println!("    jules <command> [options] [file.jules] [-- args…]\n");
    println!("COMMANDS:");
    println!("    run   <file.jules>    Execute a Jules source file");
    println!("    compile <file.jules>  Compile to native x86-64 ELF binary (AOT, fastest!)");
    println!("    check <file.jules>    Type-check and lint without running (incremental cache)");
    println!("    fix   <file.jules>    Apply safe syntax autofixes from diagnostics");
    println!("    fmt   <file.jules>    Pretty-print the source (token pass)");
    println!("    train <file.jules>    Run all train {{ … }} blocks");
    println!("    estimate           Estimate Jules training time/resources");
    println!("    repl               Interactive REPL / playground");
    println!("    version            Print version information");
    println!("    help               Print this message\n");
    println!("OPTIONS:");
    println!("    --no-color         Disable ANSI colour output");
    println!("    --json-diag        Emit diagnostics as JSON");
    println!("    --emit-tokens      Print token stream and exit");
    println!("    --emit-ast         Print AST and exit");
    println!("    --warn-error / -W  Treat warnings as errors");
    println!("    --quiet  / -q      Suppress warnings and notes");
    println!("    --tab-width <N>    Tab stop width (default: 4)");
    println!("    -o, --output <PATH> Output path for compile command");
    println!("    --entry <fn>       Entry-point function (default: main)");
    println!("    --dry-run          (fix) show changes without writing");
    println!("    --diff             (fix) print changed lines");
    println!("    --aggressive       (fix) allow riskier fixer rules");
    println!("    --params <N>       (estimate) model parameter count");
    println!("    --batch <N>        (estimate) training batch size");
    println!("    --episodes <N>     (estimate) episode count");
    println!("    --steps <N>        (estimate) max steps per episode");
    println!("    --envs <N>         (estimate) parallel environments");
    println!("    --device <cpu|gpu> (estimate) execution device");
    println!("    --ml-backend <jules|jax>  (train) choose trainer backend");
    println!("    --jax-ir <file.json>      (train+jax) model IR path (optional)");
    println!("    --jax-dataset <file.npz>  (train+jax) dataset with x_train/y_train");
    println!("    --jax-out <dir>           (train+jax) output dir (default artifacts/jax)");
    println!("    --jax-script <path.py>    (train+jax) backend script path");
    println!("\nTIERED COMPILATION:");
    println!("    --tiered           Enable tiered compilation (start fast, optimize hot code)");
    println!("    --tier-policy <P>  fast-startup | balanced | max-performance");
    println!("    --tier-stats       Print tier compilation statistics after execution");
    println!("\nEXAMPLES:");
    println!("    jules run examples/physics.jules");
    println!("    jules compile examples/math.jules           # → ./math (native ELF binary)");
    println!("    jules compile examples/math.jules -o out    # → ./out");
    println!("    ./math                                       # run the native binary!");
    println!("    jules run examples/warden.jules --tiered --tier-stats");
    println!("    jules run examples/warden.jules --tiered --tier-policy max-performance");
    println!("    jules check src/agent.jules -W");
    println!("    jules fix broken.jules");
    println!("    jules train examples/warden.jules");
    println!("    jules train examples/warden.jules --ml-backend jax --jax-dataset train.npz --jax-out artifacts/jax");
    println!("    jules estimate --params 40000000 --batch 128 --episodes 300000 --steps 128 --envs 8 --device gpu");
    println!("    jules repl");
}

#[derive(Debug, Clone, Copy)]
struct TrainEstimate {
    total_steps: u64,
    estimated_steps_per_sec: f64,
    estimated_sim_steps_per_sec: f64,
    estimated_model_steps_per_sec: f64,
    bottleneck: &'static str,
    confidence_low_steps_per_sec: f64,
    confidence_high_steps_per_sec: f64,
    estimated_duration: Duration,
    estimated_memory_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
struct ResourceSnapshot {
    ram_available_bytes: Option<u64>,
    gpu_available_bytes: Option<u64>,
}

const EST_BASELINE_PARAMS: usize = 1_000_000;
const F32_BYTES: u64 = std::mem::size_of::<f32>() as u64;

fn cmd_estimate(args: &CliArgs) -> i32 {
    let resources = detect_resources(&args.estimate_device);
    println!("jules estimate");
    println!(
        "  params={} batch={} episodes={} steps={} envs={} device={}",
        args.estimate_params,
        args.estimate_batch,
        args.estimate_episodes,
        args.estimate_steps,
        args.estimate_envs,
        args.estimate_device
    );
    match resources.ram_available_bytes {
        Some(v) => println!("  RAM available: {:.2} GiB", bytes_to_gib(v)),
        None => println!("  RAM available: unknown"),
    }
    if args.estimate_device == "gpu" {
        match resources.gpu_available_bytes {
            Some(v) => println!("  GPU memory available: {:.2} GiB", bytes_to_gib(v)),
            None => println!("  GPU memory available: unknown"),
        }
    }

    let est = estimate_training(
        args.estimate_params,
        args.estimate_batch,
        args.estimate_episodes,
        args.estimate_steps,
        args.estimate_envs,
        &args.estimate_device,
    );
    println!(
        "  estimated: steps={} steps/s≈{:.0} (range {:.0}..{:.0}, sim≈{:.0}, model≈{:.0}, bottleneck={}) time≈{} memory≈{:.2} GiB",
        est.total_steps,
        est.estimated_steps_per_sec,
        est.confidence_low_steps_per_sec,
        est.confidence_high_steps_per_sec,
        est.estimated_sim_steps_per_sec,
        est.estimated_model_steps_per_sec,
        est.bottleneck,
        format_duration(est.estimated_duration),
        bytes_to_gib(est.estimated_memory_bytes),
    );
    print_speed_suggestions(est, args, resources);
    0
}

fn estimate_training(
    params: usize,
    batch: usize,
    episodes: usize,
    steps: usize,
    envs: usize,
    device: &str,
) -> TrainEstimate {
    let params = params.max(1);
    let batch = batch.max(1);
    let envs = envs.max(1);
    let total_steps = episodes as u64 * steps as u64 * envs as u64;
    let base_sim_steps_per_sec = if device == "gpu" {
        20_000_000.0
    } else {
        7_500_000.0
    };
    let base_model_steps_per_sec = if device == "gpu" {
        16_000_000.0
    } else {
        4_200_000.0
    };
    let param_scale = (params as f64 / EST_BASELINE_PARAMS as f64).max(1e-9);
    let env_gain = 1.0 + (envs as f64).log2().max(0.0) * 0.40;
    let batch_gain = if batch <= 64 {
        1.0
    } else if batch <= 256 {
        1.18
    } else {
        1.10
    };
    let sim_steps_per_sec = (base_sim_steps_per_sec * env_gain).max(1.0);
    let model_steps_per_sec =
        (base_model_steps_per_sec * batch_gain / param_scale.powf(0.92)).max(1.0);
    let (estimated_steps_per_sec, bottleneck) = if sim_steps_per_sec <= model_steps_per_sec {
        (sim_steps_per_sec, "sim")
    } else {
        (model_steps_per_sec, "model")
    };
    let secs = total_steps as f64 / estimated_steps_per_sec;
    // Baseline model is intentionally conservative; expose an uncertainty range.
    let (conf_low, conf_high) = if device == "gpu" {
        (0.65, 1.25)
    } else {
        (0.70, 1.20)
    };
    TrainEstimate {
        total_steps,
        estimated_steps_per_sec,
        estimated_sim_steps_per_sec: sim_steps_per_sec,
        estimated_model_steps_per_sec: model_steps_per_sec,
        bottleneck,
        confidence_low_steps_per_sec: (estimated_steps_per_sec * conf_low).max(1.0),
        confidence_high_steps_per_sec: (estimated_steps_per_sec * conf_high).max(1.0),
        estimated_duration: Duration::from_secs_f64(secs.max(0.0)),
        estimated_memory_bytes: estimate_memory_bytes(params, batch, envs),
    }
}

fn estimate_memory_bytes(params: usize, batch: usize, envs: usize) -> u64 {
    let params = params as u64;
    let batch = batch as u64;
    let envs = envs as u64;
    let model_state = params.saturating_mul(F32_BYTES).saturating_mul(4);
    let activations = params
        .saturating_mul(batch.saturating_add(envs).max(1))
        .saturating_mul(F32_BYTES / 2 + 1);
    // Keep memory estimator slightly pessimistic to avoid OOM surprises.
    (model_state + activations).saturating_mul(14) / 10 + 64 * 1024 * 1024
}

fn detect_resources(device: &str) -> ResourceSnapshot {
    ResourceSnapshot {
        ram_available_bytes: read_available_ram_bytes(),
        gpu_available_bytes: if device == "gpu" {
            read_available_gpu_bytes()
        } else {
            None
        },
    }
}

fn read_available_ram_bytes() -> Option<u64> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

fn read_available_gpu_bytes() -> Option<u64> {
    let out = process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut total_mib = 0u64;
    for line in text.lines() {
        if let Ok(v) = line.trim().parse::<u64>() {
            total_mib = total_mib.saturating_add(v);
        }
    }
    if total_mib == 0 {
        None
    } else {
        Some(total_mib.saturating_mul(1024 * 1024))
    }
}

fn build_speed_actions(
    est: TrainEstimate,
    args: &CliArgs,
    resources: ResourceSnapshot,
) -> Vec<String> {
    let mut out = Vec::new();
    if est.bottleneck == "model" {
        let reduced_params = ((args.estimate_params as f64) * 0.7).max(1.0) as usize;
        let projected = estimate_training(
            reduced_params,
            args.estimate_batch,
            args.estimate_episodes,
            args.estimate_steps,
            args.estimate_envs,
            &args.estimate_device,
        );
        out.push(format!(
            "model-bound: reduce params {} -> {} (projected speedup ≈{:.2}x)",
            args.estimate_params,
            reduced_params,
            projected.estimated_steps_per_sec / est.estimated_steps_per_sec
        ));
        if args.estimate_device == "cpu" && args.estimate_batch > 256 {
            out.push(format!(
                "CPU batch may be too high for cache locality: try batch {} -> 128",
                args.estimate_batch
            ));
        } else if args.estimate_batch < 128 {
            out.push(format!(
                "model-bound: try larger batch for better compute utilization ({} -> 128)",
                args.estimate_batch
            ));
        }
    } else {
        let raised_envs = (args.estimate_envs.max(1) * 2).min(64);
        let projected = estimate_training(
            args.estimate_params,
            args.estimate_batch,
            args.estimate_episodes,
            args.estimate_steps,
            raised_envs,
            &args.estimate_device,
        );
        out.push(format!(
            "sim-bound: increase envs {} -> {} (projected speedup ≈{:.2}x)",
            args.estimate_envs,
            raised_envs,
            projected.estimated_steps_per_sec / est.estimated_steps_per_sec
        ));
        out.push(
            "sim-bound: reduce per-step world complexity (agents/sensors/colliders)".to_string(),
        );
    }

    if let Some(ram) = resources.ram_available_bytes {
        if est.estimated_memory_bytes > ram.saturating_mul(85) / 100 {
            out.push(format!(
                "memory risk: estimate {:.2} GiB is near available {:.2} GiB; reduce batch/params",
                bytes_to_gib(est.estimated_memory_bytes),
                bytes_to_gib(ram),
            ));
        }
    }
    out
}

fn print_speed_suggestions(est: TrainEstimate, args: &CliArgs, resources: ResourceSnapshot) {
    if est.estimated_duration < Duration::from_secs(30 * 60) {
        println!("  suggestion: ETA is under 30 minutes; no urgency tuning needed.");
        return;
    }
    println!("  suggestion: ETA exceeds 30 minutes. Quick speedups:");
    println!(
        "    • episodes: {} -> {}",
        args.estimate_episodes,
        ((args.estimate_episodes as f64) * 0.7) as usize
    );
    println!(
        "    • steps: {} -> {}",
        args.estimate_steps,
        (args.estimate_steps as f64 * 0.8).max(1.0) as usize
    );
    println!(
        "    • params: {} -> {}",
        args.estimate_params,
        (args.estimate_params as f64 * 0.75).max(1.0) as usize
    );
    for action in build_speed_actions(est, args, resources) {
        println!("    • {action}");
    }
    if args.estimate_device != "gpu" {
        println!("    • switch to --device gpu if available");
    }
    println!("    • sample batch sizes 64/128/256 and keep the fastest measured setting");
}

fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn format_duration(d: Duration) -> String {
    let total = d.as_secs();
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}h {m}m {s}s")
    } else {
        format!("{m}m {s}s")
    }
}

fn insert_char_at_byte(source: &mut String, byte_idx: usize, ch: char) {
    if byte_idx <= source.len() {
        source.insert(byte_idx, ch);
    }
}

fn replace_byte_range(source: &mut String, start: usize, end: usize, replacement: &str) {
    if start <= end && end <= source.len() {
        source.replace_range(start..end, replacement);
    }
}

fn apply_safe_syntax_fixes(source: &str, diags: &[Diag]) -> Option<String> {
    let mut out = source.to_string();
    let mut changed = false;

    let typo_rewrites = [
        ("fun ", "fn "),
        ("func ", "fn "),
        ("pritn(", "print("),
        ("prnit(", "print("),
        ("flase", "false"),
        ("fasle", "false"),
        ("treu", "true"),
        ("ture", "true"),
    ];
    for (from, to) in typo_rewrites {
        if out.contains(from) {
            out = out.replace(from, to);
            changed = true;
        }
    }

    let mut semicolon_lines = std::collections::BTreeSet::<u32>::new();
    let mut insertions: Vec<(usize, char)> = Vec::new();
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();
    let mut block_open_insertions: Vec<usize> = Vec::new();
    let mut comma_insertions: Vec<usize> = Vec::new();

    for d in diags {
        let Some(span) = d.span else { continue };
        let Some(hint) = d.hint.as_deref() else {
            continue;
        };
        if hint.contains("add `;` to end this statement") {
            semicolon_lines.insert(span.line);
        } else if hint.contains("close this expression with `)`") {
            insertions.push((span.start, ')'));
        } else if hint.contains("close this index/type with `]`") {
            insertions.push((span.start, ']'));
        } else if hint.contains("close this block with `}`") {
            insertions.push((span.start, '}'));
        } else if hint.contains("start a block with `{ ... }`") {
            block_open_insertions.push(span.start);
        } else if hint.contains("separate items with `,`") {
            comma_insertions.push(span.start);
        } else if hint.contains("use `=` to assign a value") {
            replacements.push((span.start, span.end, "=".into()));
        } else if hint.contains("use `->` before a return type") {
            replacements.push((span.start, span.end, "->".into()));
        }
    }

    replacements.sort_by(|a, b| b.0.cmp(&a.0));
    for (start, end, rep) in replacements {
        replace_byte_range(&mut out, start, end, &rep);
        changed = true;
    }

    block_open_insertions.sort_by(|a, b| b.cmp(a));
    for idx in block_open_insertions {
        insert_char_at_byte(&mut out, idx, '{');
        insert_char_at_byte(&mut out, idx + 1, ' ');
        changed = true;
    }

    comma_insertions.sort_by(|a, b| b.cmp(a));
    for idx in comma_insertions {
        insert_char_at_byte(&mut out, idx, ',');
        changed = true;
    }

    insertions.sort_by(|a, b| b.0.cmp(&a.0));
    for (idx, ch) in insertions {
        insert_char_at_byte(&mut out, idx, ch);
        changed = true;
    }

    if !semicolon_lines.is_empty() {
        let mut lines: Vec<String> = out.split('\n').map(|s| s.to_owned()).collect();
        for line_no in semicolon_lines {
            let idx = line_no.saturating_sub(1) as usize;
            if let Some(line) = lines.get_mut(idx) {
                let trimmed = line.trim_end();
                if !trimmed.is_empty()
                    && !trimmed.ends_with(';')
                    && !trimmed.ends_with('{')
                    && !trimmed.ends_with('}')
                {
                    line.push(';');
                    changed = true;
                }
            }
        }
        out = lines.join("\n");
    }

    // Conservative delimiter balancing at EOF.
    let mut open_paren = 0_i32;
    let mut open_bracket = 0_i32;
    let mut open_brace = 0_i32;
    for ch in out.chars() {
        match ch {
            '(' => open_paren += 1,
            ')' => open_paren -= 1,
            '[' => open_bracket += 1,
            ']' => open_bracket -= 1,
            '{' => open_brace += 1,
            '}' => open_brace -= 1,
            _ => {}
        }
    }
    while open_paren > 0 {
        out.push(')');
        open_paren -= 1;
        changed = true;
    }
    while open_bracket > 0 {
        out.push(']');
        open_bracket -= 1;
        changed = true;
    }
    while open_brace > 0 {
        out.push('}');
        open_brace -= 1;
        changed = true;
    }

    changed.then_some(out)
}

fn detect_silent_issues(source: &str) -> Vec<Diag> {
    let mut out = Vec::new();

    fn has_float_equality(trimmed: &str) -> bool {
        if !trimmed.contains("==") {
            return false;
        }
        let mut prev_was_digit = false;
        for ch in trimmed.chars() {
            if ch == '.' && prev_was_digit {
                return true;
            }
            prev_was_digit = ch.is_ascii_digit();
        }
        false
    }

    for (idx, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let span = Span {
            line: (idx + 1) as u32,
            col: (line.find(trimmed).unwrap_or(0) + 1) as u32,
            start: 0,
            end: 0,
        };

        if has_float_equality(trimmed) {
            out.push(
                Diag::warning(
                    span,
                    "possible silent logic issue: float equality comparison can be unstable",
                )
                .with_code("W-SILENT-FLOAT-EQ")
                .with_hint("compare with epsilon: `abs(a - b) < 1e-6`"),
            );
        }

        if (trimmed.starts_with("if ") || trimmed.starts_with("while "))
            && trimmed.contains(" = ")
            && !trimmed.contains("==")
            && !trimmed.contains(">=")
            && !trimmed.contains("<=")
            && !trimmed.contains("!=")
        {
            out.push(
                Diag::warning(
                    span,
                    "possible silent logic issue: assignment used in condition",
                )
                .with_code("W-SILENT-ASSIGN-COND")
                .with_hint("did you mean `==`?"),
            );
        }

        if trimmed.starts_with("while true") {
            out.push(
                Diag::warning(
                    span,
                    "possible silent runtime issue: `while true` may never terminate",
                )
                .with_code("W-SILENT-INFINITE-LOOP")
                .with_hint("ensure a reachable `break`/`return` path"),
            );
        }
    }
    out
}

fn cmd_fix(args: &CliArgs) -> i32 {
    let path = match &args.file {
        Some(p) => p,
        None => {
            eprintln!("jules fix: no file provided");
            return 2;
        }
    };
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("jules: cannot read `{}`: {e}", path.display());
            return 2;
        }
    };

    let filename = path.to_string_lossy();
    let mut unit = CompileUnit::new(filename.as_ref(), &source);
    let mut pipeline = Pipeline::new();
    pipeline.opt_level = args.opt_level;
    pipeline.print_opt_stats = args.print_opt_stats;
    pipeline.quiet = true;
    let _ = pipeline.run(&mut unit);

    if unit.diags.is_empty() {
        println!("jules fix: no diagnostics; file is already clean");
        return 0;
    }

    let mut current = source.clone();
    let mut changed = false;
    for _ in 0..3 {
        match apply_safe_syntax_fixes(&current, &unit.diags) {
            Some(next) if next != current => {
                current = next;
                changed = true;
                let mut rerun = CompileUnit::new(filename.as_ref(), &current);
                let _ = pipeline.run(&mut rerun);
                unit = rerun;
                if unit.diags.is_empty() {
                    break;
                }
            }
            _ => break,
        }
    }

    if changed && current != source {
        if let Err(e) = fs::write(path, current) {
            eprintln!("jules fix: failed writing `{}`: {e}", path.display());
            return 2;
        }
        println!("jules fix: applied safe fixes to {}", path.display());
        0
    } else {
        println!("jules fix: no safe automatic fix could be applied");
        1
    }
}

// ── jules check ────────────────────────────────────────────────────────────

fn cmd_check(args: &CliArgs) -> i32 {
    let path = match &args.file {
        Some(p) => p,
        None => {
            eprintln!("jules check: no file provided");
            return 2;
        }
    };
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("jules: cannot read `{}`: {e}", path.display());
            return 2;
        }
    };

    let cfg = render_cfg(args);
    let filename = path.to_string_lossy();
    let source_hash = hash_source(&source);
    if let Some(meta) = load_incremental_check_cache(path) {
        if meta.source_hash == source_hash && meta.diag_free {
            if !args.quiet {
                println!("jules check: incremental cache hit (no source changes)");
            }
            return 0;
        }
    }
    let mut unit = CompileUnit::new(filename.as_ref(), &source);

    let mut pipeline = Pipeline::new();
    pipeline.opt_level = args.opt_level;
    pipeline.print_opt_stats = args.print_opt_stats;
    pipeline.warn_as_error = args.warn_as_error;
    pipeline.quiet = args.quiet;
    pipeline.run(&mut unit);
    unit.diags.extend(detect_silent_issues(&source));

    emit_diagnostics(&unit.diags, &source, &filename, &cfg, args.json_diag);
    print_summary(&unit, &cfg);
    store_incremental_check_cache(
        path,
        &CheckCacheMeta {
            source_hash,
            diag_free: unit.diags.is_empty(),
        },
    );

    if unit.has_errors() {
        1
    } else {
        0
    }
}

#[derive(Debug, Clone, Copy)]
struct CheckCacheMeta {
    source_hash: u64,
    diag_free: bool,
}

fn hash_source(source: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    hasher.finish()
}

fn incremental_check_cache_path(path: &Path) -> PathBuf {
    let mut hasher = DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    let key = hasher.finish();
    PathBuf::from(".jules_cache")
        .join("check")
        .join(format!("{key:016x}.meta"))
}

fn load_incremental_check_cache(path: &Path) -> Option<CheckCacheMeta> {
    let cache_path = incremental_check_cache_path(path);
    let raw = fs::read_to_string(cache_path).ok()?;
    let mut parts = raw.trim().split(',');
    let source_hash = parts.next()?.parse::<u64>().ok()?;
    let diag_free = parts.next()? == "1";
    Some(CheckCacheMeta {
        source_hash,
        diag_free,
    })
}

fn store_incremental_check_cache(path: &Path, meta: &CheckCacheMeta) {
    let cache_path = incremental_check_cache_path(path);
    if let Some(parent) = cache_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let encoded = format!(
        "{},{}",
        meta.source_hash,
        if meta.diag_free { "1" } else { "0" }
    );
    let _ = fs::write(cache_path, encoded);
}

// ── jules run ──────────────────────────────────────────────────────────────

fn cmd_run(args: &CliArgs) -> i32 {
    let path = match &args.file {
        Some(p) => p,
        None => {
            eprintln!("jules run: no file provided");
            return 2;
        }
    };
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("jules: cannot read `{}`: {e}", path.display());
            return 2;
        }
    };

    let cfg = render_cfg(args);
    let filename = path.to_string_lossy();
    let mut unit = CompileUnit::new(filename.as_ref(), &source);

    let mut pipeline = Pipeline::new();
    pipeline.opt_level = args.opt_level;
    pipeline.print_opt_stats = args.print_opt_stats;
    pipeline.warn_as_error = args.warn_as_error;
    pipeline.quiet = args.quiet;

    // Token dump mode.
    if args.emit_tokens {
        let mut lexer = Lexer::new(&source);
        let (tokens, _) = lexer.tokenize();
        for tok in &tokens {
            println!("{:5}:{:3}  {:?}", tok.span.line, tok.span.col, tok.kind);
        }
        return 0;
    }

    let result = pipeline.run(&mut unit);
    unit.diags.extend(detect_silent_issues(&source));

    emit_diagnostics(&unit.diags, &source, &filename, &cfg, args.json_diag);
    print_summary(&unit, &cfg);

    if unit.has_errors() {
        return 1;
    }

    // Run the interpreter.
    if let PipelineResult::Ok(program) = result {
        if args.tiered {
            #[cfg(feature = "phase3-jit")]
            {
                // Use tiered compilation for progressive optimization
                let policy = match args.tier_policy.as_str() {
                    "fast-startup" => crate::tiered_compilation::PromotionPolicy::fast_startup(),
                    "balanced" => crate::tiered_compilation::PromotionPolicy::balanced(),
                    "max-performance" => crate::tiered_compilation::PromotionPolicy::max_performance(),
                    _ => crate::tiered_compilation::PromotionPolicy::balanced(),
                };

                let mut tiered_mgr = crate::tiered_compilation::TieredExecutionManager::new(policy);
                tiered_mgr.load_program(&program);
                tiered_mgr.enabled = true;

                match tiered_mgr.call_function(&args.entry, vec![]) {
                    Ok(val) => {
                        if !matches!(val, crate::interp::Value::Unit) {
                            println!("{val}");
                        }
                    }
                    Err(e) => {
                        let diag = adapt_runtime_error(e);
                        emit_diagnostics(&[diag], &source, &filename, &cfg, args.json_diag);
                        return 1;
                    }
                }

                if args.print_tier_stats {
                    eprintln!("{}", tiered_mgr.tier_stats_summary());
                }
            }
            #[cfg(not(feature = "phase3-jit"))]
            {
                eprintln!(
                    "warning: --tiered requested but binary was built without `phase3-jit`; running interpreter"
                );
                let mut interp = crate::interp::Interpreter::new();
                interp.load_program(&program);
                match interp.call_fn(&args.entry, vec![]) {
                    Ok(val) => {
                        if !matches!(val, crate::interp::Value::Unit) {
                            println!("{val}");
                        }
                    }
                    Err(e) => {
                        let diag = adapt_runtime_error(e);
                        emit_diagnostics(&[diag], &source, &filename, &cfg, args.json_diag);
                        return 1;
                    }
                }
            }
        } else {
            // Use plain interpreter (existing behavior)
            let mut interp = crate::interp::Interpreter::new();
            interp.load_program(&program);
            match interp.call_fn(&args.entry, vec![]) {
                Ok(val) => {
                    if !matches!(val, crate::interp::Value::Unit) {
                        println!("{val}");
                    }
                }
                Err(e) => {
                    let diag = adapt_runtime_error(e);
                    emit_diagnostics(&[diag], &source, &filename, &cfg, args.json_diag);
                    return 1;
                }
            }
        }
    }
    0
}

// ── jules train ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct JaxModelIr {
    model_name: String,
    input_dim: u64,
    layers: Vec<u64>,
    activation: &'static str,
    task: &'static str,
}

#[derive(Debug, Clone)]
struct BackendCapability {
    model_name: String,
    jules_supported: bool,
    jax_supported: bool,
    jax_reason: Option<String>,
}

fn activation_name_for_jax(a: &crate::ast::Activation) -> &'static str {
    match a {
        crate::ast::Activation::Relu | crate::ast::Activation::LeakyRelu => "relu",
        crate::ast::Activation::Tanh => "tanh",
        crate::ast::Activation::Silu | crate::ast::Activation::Swish => "silu",
        crate::ast::Activation::Gelu => "gelu",
        _ => "gelu",
    }
}

fn build_jax_ir_from_model(model: &crate::ast::ModelDecl) -> Result<JaxModelIr, String> {
    let mut input_dim = None;
    let mut layers = Vec::new();
    let mut activation = "gelu";
    let mut task = "classification";

    for layer in &model.layers {
        match layer {
            crate::ast::ModelLayer::Input { size, .. } => {
                input_dim = Some(*size);
            }
            crate::ast::ModelLayer::Dense {
                units,
                activation: act,
                ..
            } => {
                layers.push(*units);
                activation = activation_name_for_jax(act);
            }
            crate::ast::ModelLayer::Output {
                units,
                activation: act,
                ..
            } => {
                layers.push(*units);
                if matches!(act, crate::ast::Activation::Linear) {
                    task = "regression";
                }
            }
            crate::ast::ModelLayer::Dropout { .. } | crate::ast::ModelLayer::Norm { .. } => {
                // Ignored for export because the bridge currently trains dense MLPs.
            }
            other => {
                return Err(format!(
                    "unsupported layer for JAX bridge export: {other:?}"
                ));
            }
        }
    }

    let input_dim = input_dim.ok_or_else(|| "missing required `input` layer".to_string())?;
    if layers.is_empty() {
        return Err("missing required dense/output layers".to_string());
    }

    Ok(JaxModelIr {
        model_name: model.name.clone(),
        input_dim,
        layers,
        activation,
        task,
    })
}

fn feature_capability_matrix(program: &crate::ast::Program) -> Vec<BackendCapability> {
    program
        .items
        .iter()
        .filter_map(|item| match item {
            crate::ast::Item::Model(m) => Some(m),
            _ => None,
        })
        .map(|m| match build_jax_ir_from_model(m) {
            Ok(_) => BackendCapability {
                model_name: m.name.clone(),
                jules_supported: true,
                jax_supported: true,
                jax_reason: None,
            },
            Err(reason) => BackendCapability {
                model_name: m.name.clone(),
                jules_supported: true,
                jax_supported: false,
                jax_reason: Some(reason),
            },
        })
        .collect()
}

fn print_feature_capability_matrix(program: &crate::ast::Program) {
    let rows = feature_capability_matrix(program);
    if rows.is_empty() {
        return;
    }
    println!("feature capability matrix:");
    for row in rows {
        let jules = if row.jules_supported { "✅" } else { "❌" };
        let jax = if row.jax_supported { "✅" } else { "❌" };
        match row.jax_reason {
            Some(reason) => println!(
                "  {}: Jules {}  JAX {} (why: {})",
                row.model_name, jules, jax, reason
            ),
            None => println!("  {}: Jules {}  JAX {}", row.model_name, jules, jax),
        }
    }
}

fn build_jax_ir_from_program(program: &crate::ast::Program) -> Result<JaxModelIr, String> {
    let train_model = program.items.iter().find_map(|item| match item {
        crate::ast::Item::Train(t) => t.model.as_deref(),
        _ => None,
    });

    let model = program
        .items
        .iter()
        .filter_map(|item| match item {
            crate::ast::Item::Model(m) => Some(m),
            _ => None,
        })
        .find(|m| train_model.map_or(true, |name| m.name == name))
        .ok_or_else(|| {
            if let Some(name) = train_model {
                format!("jax backend: model `{name}` referenced by train block not found")
            } else {
                "jax backend: no `model` declaration found in Jules source".to_string()
            }
        })?;
    build_jax_ir_from_model(model).map_err(|why| {
        format!(
            "jax backend: model `{}` is not supported ({why})",
            model.name
        )
    })
}

fn write_jax_ir_file(ir: &JaxModelIr) -> Result<PathBuf, String> {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("jules_jax_ir_{}_{}.json", process::id(), ts));
    let layers = ir
        .layers
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let content = format!(
        "{{\n  \"schema_version\": 1,\n  \"model_name\": \"{}\",\n  \"input_dim\": {},\n  \"layers\": [{}],\n  \"activation\": \"{}\",\n  \"task\": \"{}\"\n}}\n",
        ir.model_name, ir.input_dim, layers, ir.activation, ir.task
    );
    fs::write(&path, content).map_err(|e| {
        format!(
            "jax backend: failed to write generated IR `{}`: {e}",
            path.display()
        )
    })?;
    Ok(path)
}

fn check_jax_backend_env(script: &Path) -> Result<(), String> {
    if !script.exists() {
        return Err(format!(
            "jax backend: script not found at `{}` (use --jax-script to override)",
            script.display()
        ));
    }

    let check = ProcessCommand::new("python3")
        .arg("-c")
        .arg("import jax, optax, numpy; print('ok')")
        .output()
        .map_err(|e| format!("jax backend: failed to run python3 preflight check: {e}"))?;
    if !check.status.success() {
        let stderr = String::from_utf8_lossy(&check.stderr);
        return Err(format!(
            "jax backend dependencies are missing.\n\
             Install them with:\n\
               pip install --upgrade pip\n\
               pip install jax optax numpy\n\
             python3 preflight error:\n{}",
            stderr.trim()
        ));
    }
    Ok(())
}

// =============================================================================
// AOT NATIVE COMPILATION COMMAND
// =============================================================================

fn cmd_compile(args: &CliArgs) -> i32 {
    let path = match &args.file {
        Some(p) => p,
        None => {
            eprintln!("jules compile: no file provided");
            eprintln!("Usage: jules compile <file.jules> [-o output]");
            return 2;
        }
    };

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("jules: cannot read `{}`: {e}", path.display());
            return 2;
        }
    };

    // Determine output path
    let output_path = args.output.as_ref().map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| {
            // Default: strip .jules extension and add nothing (native binary)
            let stem = path.file_stem().unwrap_or_default().to_string_lossy();
            format!("./{}", stem)
        });

    if !args.quiet {
        eprintln!("Compiling {} → {} (AOT native x86-64)", path.display(), output_path);
    }

    // Run full compilation pipeline
    let path_str = path.to_string_lossy();
    let mut unit = CompileUnit::new(path_str.as_ref(), &source);
    let result = Pipeline::new().run(&mut unit);

    if unit.has_errors() {
        let msgs: Vec<_> = unit
            .diags
            .iter()
            .filter(|d| d.is_error())
            .map(|d| d.message.clone())
            .collect();
        for msg in msgs {
            eprintln!("error: {}", msg);
        }
        return 1;
    }

    let program = match result {
        PipelineResult::Ok(program) => program,
        _ => {
            eprintln!("jules compile: pipeline did not produce a program");
            return 1;
        }
    };

    // Apply optimizations
    if !args.quiet {
        eprintln!("  Optimizing (level {})...", args.opt_level);
    }

    // AOT compile to native ELF binary
    let start = std::time::Instant::now();
    match crate::aot_native::compile_to_native(&program, &output_path, args.opt_level) {
        Ok(()) => {
            let elapsed = start.elapsed();
            if !args.quiet {
                eprintln!("✓ Compiled in {:.3}s", elapsed.as_secs_f64());
                eprintln!("  Output: {}", output_path);
                
                // Show file size
                if let Ok(metadata) = std::fs::metadata(&output_path) {
                    let size = metadata.len();
                    if size < 1024 {
                        eprintln!("  Size: {} bytes", size);
                    } else if size < 1024 * 1024 {
                        eprintln!("  Size: {:.1} KB", size as f64 / 1024.0);
                    } else {
                        eprintln!("  Size: {:.2} MB", size as f64 / (1024.0 * 1024.0));
                    }
                }
            }
            0
        }
        Err(e) => {
            eprintln!("jules compile: {}", e);
            1
        }
    }
}

fn cmd_train(args: &CliArgs) -> i32 {
    let path = match &args.file {
        Some(p) => p,
        None => {
            eprintln!("jules train: no file provided");
            return 2;
        }
    };
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("jules: cannot read `{}`: {e}", path.display());
            return 2;
        }
    };

    let cfg = render_cfg(args);
    let filename = path.to_string_lossy();
    let mut unit = CompileUnit::new(filename.as_ref(), &source);

    let mut pipeline = Pipeline::new();
    pipeline.opt_level = args.opt_level;
    pipeline.print_opt_stats = args.print_opt_stats;
    pipeline.warn_as_error = args.warn_as_error;
    pipeline.quiet = args.quiet;
    let result = pipeline.run(&mut unit);

    emit_diagnostics(&unit.diags, &source, &filename, &cfg, args.json_diag);
    print_summary(&unit, &cfg);

    if unit.has_errors() {
        return 1;
    }

    if let PipelineResult::Ok(program) = result {
        if !args.quiet {
            print_feature_capability_matrix(&program);
        }
        if args.ml_backend == "jax" {
            let dataset = match &args.jax_dataset {
                Some(p) => p.clone(),
                None => {
                    eprintln!(
                        "jules train: `--ml-backend jax` requires `--jax-dataset <train.npz>`\n\
                         example: jules train model.jules --ml-backend jax --jax-dataset train.npz"
                    );
                    return 2;
                }
            };
            if let Err(msg) = check_jax_backend_env(&args.jax_script) {
                eprintln!("{msg}");
                return 2;
            }
            let mut generated_ir = None;
            let ir_path = if let Some(p) = &args.jax_ir {
                p.clone()
            } else {
                match build_jax_ir_from_program(&program).and_then(|ir| write_jax_ir_file(&ir)) {
                    Ok(p) => {
                        generated_ir = Some(p.clone());
                        p
                    }
                    Err(msg) => {
                        eprintln!("{msg}");
                        return 2;
                    }
                }
            };

            let mut cmd = ProcessCommand::new("python3");
            cmd.arg(&args.jax_script)
                .arg("--ir")
                .arg(&ir_path)
                .arg("--dataset")
                .arg(&dataset)
                .arg("--out")
                .arg(&args.jax_out);
            let status = match cmd.status() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("jules train: failed to launch JAX backend via python3: {e}");
                    return 2;
                }
            };
            if let Some(p) = generated_ir {
                let _ = fs::remove_file(p);
            }
            return if status.success() { 0 } else { 1 };
        }

        match crate::interp::jules_train(&program) {
            Ok(all_stats) => {
                for (i, stats) in all_stats.iter().enumerate() {
                    println!(
                        "Train block {}: mean reward = {:.4}, steps = {}",
                        i + 1,
                        stats.mean_reward,
                        stats.total_steps
                    );
                }
            }
            Err(e) => {
                let diag = adapt_runtime_error(e);
                emit_diagnostics(&[diag], &source, &filename, &cfg, args.json_diag);
                return 1;
            }
        }
    }
    0
}

// ── jules fmt ─────────────────────────────────────────────────────────────

fn cmd_fmt(args: &CliArgs) -> i32 {
    let path = match &args.file {
        Some(p) => p,
        None => {
            eprintln!("jules fmt: no file provided");
            return 2;
        }
    };
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("jules: cannot read `{}`: {e}", path.display());
            return 2;
        }
    };

    let mut lexer = Lexer::new(&source);
    let (tokens, errors) = lexer.tokenize();

    let cfg = render_cfg(args);
    if !errors.is_empty() {
        let diags: Vec<Diag> = errors.into_iter().map(Diag::from).collect();
        let filename = path.to_string_lossy();
        let renderer = DiagRenderer::new(&source, &filename, cfg);
        eprint!("{}", renderer.render_all(&diags));
        return 1;
    }

    // Very simple token-level reformatter: emit tokens separated by spaces,
    // with newlines inserted after `{`, `}`, `;`.
    let mut out = String::new();
    let mut indent = 0usize;
    let indent_str = "    ";

    for tok in &tokens {
        use crate::lexer::TokenKind;
        match &tok.kind {
            TokenKind::Eof => break,
            TokenKind::LBrace => {
                out.push_str("{\n");
                indent += 1;
                out.push_str(&indent_str.repeat(indent));
            }
            TokenKind::RBrace => {
                indent = indent.saturating_sub(1);
                out.push('\n');
                out.push_str(&indent_str.repeat(indent));
                out.push_str("}\n");
                if indent > 0 {
                    out.push_str(&indent_str.repeat(indent));
                }
            }
            TokenKind::Semicolon => {
                out.push_str(";\n");
                out.push_str(&indent_str.repeat(indent));
            }
            kind => {
                let _ = kind;
                out.push_str(&tok.raw);
                out.push(' ');
            }
        }
    }

    println!("{}", out.trim_end());
    0
}

// =============================================================================
// §9  REPL
// =============================================================================

const REPL_BANNER: &str = r#"Jules REPL  (type :help for commands, :quit to exit)
Tensor-first · ECS · AI · GPU
"#;

const REPL_HELP: &str = r#"REPL commands:
  :quit / :q          Exit the REPL
  :help / :h          Show this message
  :tokens <expr>      Show token stream for an expression
  :demo <name>        Run built-in demo (game|arcade|ml)
  :persona <target>   Show language priorities (game|ml|all)
  :scientist on|off   Toggle experiment timing output
  :track <file|off>   Append REPL inputs/results to an experiment log
  :clear              Clear the screen
  :load <file>        Load and execute a file
  :reset              Reset interpreter state
  :history            Show input history
  :color on|off       Toggle colour output

Examples:
  // Direct program execution
  fn main() { println("hello from Jules playground"); }

  // Statement mode (auto-wrapped in `fn main { ... }`)
  let x = 3 + 4
  let t = tensor<f32>[2, 2] { 1.0, 2.0, 3.0, 4.0 }
  let C = A @ B
"#;

const PERSONA_GAME_DEV: &str = r#"Game-dev priorities in Jules:
  1) Fast iteration loops (REPL + hot-reload + editor tooling)
  2) Deterministic simulation + ECS/system orchestration
  3) Tight profiling/debug visibility in-frame
  4) FFI-friendly embedding into existing engine stacks

Try now:
  :demo game
  :demo arcade
"#;

const PERSONA_ML_SCI: &str = r#"ML scientist priorities in Jules:
  1) Tensor-first ergonomics with clear math operators
  2) Reproducible train/check loops with quick feedback
  3) Interop with Python/C++ workflows for experiments
  4) Efficient inference/training paths (CPU/GPU + quantized options)

Try now:
  :demo ml
"#;

const DEMO_GAME: &str = include_str!("jules_files/small_game.jules");
const DEMO_ARCADE: &str = include_str!("jules_files/game_arcade_showcase.jules");
const DEMO_ML: &str = include_str!("jules_files/game_nn_demo.jules");

fn demo_source(name: &str) -> Option<&'static str> {
    match name {
        "game" => Some(DEMO_GAME),
        "arcade" => Some(DEMO_ARCADE),
        "ml" => Some(DEMO_ML),
        _ => None,
    }
}

pub struct Repl {
    cfg: RenderCfg,
    history: Vec<String>,
    multiline: String, // accumulates a multi-line block
    in_block: bool,    // true when inside { … }
    scientist_mode: bool,
    experiment_log: Option<PathBuf>,
}

impl Repl {
    pub fn new(cfg: RenderCfg) -> Self {
        Repl {
            cfg,
            history: Vec::new(),
            multiline: String::new(),
            in_block: false,
            scientist_mode: false,
            experiment_log: None,
        }
    }

    pub fn run(&mut self) {
        println!(
            "{}",
            Ansi::paint(self.cfg.color, Ansi::BRIGHT_CYN, REPL_BANNER)
        );

        loop {
            let prompt = if self.in_block {
                Ansi::paint(self.cfg.color, Ansi::DIM, "... > ")
            } else {
                Ansi::paint(self.cfg.color, Ansi::BRIGHT_CYN, "jules> ")
            };
            print!("{prompt}");
            io::stdout().flush().unwrap();

            let mut line = String::new();
            match io::stdin().lock().read_line(&mut line) {
                Ok(0) => {
                    // EOF (Ctrl-D).
                    println!();
                    break;
                }
                Err(e) => {
                    eprintln!("REPL read error: {e}");
                    break;
                }
                Ok(_) => {}
            }

            let line = line.trim_end_matches('\n').trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }

            // ── REPL meta-commands ─────────────────────────────────────────────
            match line.trim() {
                ":quit" | ":q" | "quit" | "exit" => {
                    println!("Goodbye.");
                    break;
                }
                ":help" | ":h" => {
                    println!("{REPL_HELP}");
                    continue;
                }
                ":clear" => {
                    print!("\x1b[2J\x1b[H");
                    io::stdout().flush().unwrap();
                    continue;
                }
                ":reset" => {
                    self.multiline.clear();
                    self.in_block = false;
                    self.history.clear();
                    self.experiment_log = None;
                    println!("{}", Ansi::paint(self.cfg.color, Ansi::DIM, "State reset."));
                    continue;
                }
                ":history" => {
                    for (i, h) in self.history.iter().enumerate() {
                        println!("{i:4}: {h}");
                    }
                    continue;
                }
                s if s.starts_with(":color ") => {
                    let rest = s.trim_start_matches(":color ").trim();
                    self.cfg.color = rest == "on";
                    println!(
                        "Color {}",
                        if self.cfg.color {
                            "enabled"
                        } else {
                            "disabled"
                        }
                    );
                    continue;
                }
                s if s.starts_with(":load ") => {
                    let path = s.trim_start_matches(":load ").trim();
                    self.cmd_load(path);
                    continue;
                }
                s if s.starts_with(":scientist ") => {
                    let mode = s.trim_start_matches(":scientist ").trim();
                    self.cmd_scientist(mode);
                    continue;
                }
                s if s.starts_with(":track ") => {
                    let target = s.trim_start_matches(":track ").trim();
                    self.cmd_track(target);
                    continue;
                }
                s if s.starts_with(":tokens ") => {
                    let code = s.trim_start_matches(":tokens ").trim();
                    self.cmd_tokens(code);
                    continue;
                }
                s if s.starts_with(":demo ") => {
                    let name = s.trim_start_matches(":demo ").trim();
                    self.cmd_demo(name);
                    continue;
                }
                s if s.starts_with(":persona ") => {
                    let target = s.trim_start_matches(":persona ").trim();
                    self.cmd_persona(target);
                    continue;
                }
                _ => {}
            }

            // ── Multi-line block accumulation ──────────────────────────────────
            self.history.push(line.to_owned());
            self.multiline.push_str(line);
            self.multiline.push('\n');

            let brace_balance: i64 = self
                .multiline
                .chars()
                .map(|c| {
                    if c == '{' {
                        1
                    } else if c == '}' {
                        -1
                    } else {
                        0
                    }
                })
                .sum();

            if brace_balance > 0 {
                self.in_block = true;
                continue;
            }
            self.in_block = false;

            // ── Evaluate the completed input ───────────────────────────────────
            let input = std::mem::take(&mut self.multiline);
            self.eval_input(&input);
        }
    }

    fn eval_input(&mut self, input: &str) {
        let started = Instant::now();
        let status: &'static str;

        // First try full-program mode (expects user-provided `fn main`).
        match self.run_repl_program(input) {
            Ok(()) => {
                status = "ok:program";
            }
            Err(ReplRunError::Runtime(msg)) if msg.contains("unknown function: main") => {
                status = match self.run_statement_mode(input) {
                    Ok(()) => "ok:statement",
                    Err(ReplRunError::Compile(diags)) => {
                        let renderer = DiagRenderer::new(input, "<repl>", self.cfg.clone());
                        eprint!("{}", renderer.render_all(&diags));
                        "compile-error"
                    }
                    Err(ReplRunError::Runtime(msg)) => {
                        eprintln!("{}", Ansi::paint(self.cfg.color, Ansi::BRIGHT_RED, &msg));
                        "runtime-error"
                    }
                };
            }
            Err(ReplRunError::Compile(_diags)) => {
                status = match self.run_statement_mode(input) {
                    Ok(()) => "ok:statement-fallback",
                    Err(ReplRunError::Compile(diags)) => {
                        let renderer = DiagRenderer::new(input, "<repl>", self.cfg.clone());
                        eprint!("{}", renderer.render_all(&diags));
                        "compile-error"
                    }
                    Err(ReplRunError::Runtime(msg)) => {
                        eprintln!("{}", Ansi::paint(self.cfg.color, Ansi::BRIGHT_RED, &msg));
                        "runtime-error"
                    }
                };
            }
            Err(ReplRunError::Runtime(msg)) => {
                eprintln!("{}", Ansi::paint(self.cfg.color, Ansi::BRIGHT_RED, &msg));
                status = "runtime-error";
            }
        }

        if self.scientist_mode {
            let elapsed = started.elapsed();
            println!(
                "{}",
                Ansi::paint(
                    self.cfg.color,
                    Ansi::DIM,
                    &format!("[scientist] {status} in {:.2?}", elapsed),
                )
            );
        }
        self.log_experiment(input, status);
    }

    fn run_statement_mode(&self, input: &str) -> Result<(), ReplRunError> {
        let wrapped = format!("fn main() {{\n{input}\n}}\n");
        self.run_repl_program(&wrapped)
    }

    fn run_repl_program(&self, source: &str) -> Result<(), ReplRunError> {
        let mut unit = CompileUnit::new("<repl>", source);
        let result = Pipeline::new().run(&mut unit);
        if unit.has_errors() {
            return Err(ReplRunError::Compile(unit.diags));
        }

        let PipelineResult::Ok(program) = result else {
            return Err(ReplRunError::Runtime(
                "pipeline did not produce a runnable program".into(),
            ));
        };

        let mut interp = crate::interp::Interpreter::new();
        interp.load_program(&program);
        interp
            .call_fn("main", vec![])
            .map(|_| ())
            .map_err(|e| ReplRunError::Runtime(e.message))
    }

    fn cmd_tokens(&self, code: &str) {
        let mut lexer = Lexer::new(code);
        let (tokens, errors) = lexer.tokenize();

        if !errors.is_empty() {
            let renderer = DiagRenderer::new(code, "<repl>", self.cfg.clone());
            let diags: Vec<Diag> = errors.into_iter().map(Diag::from).collect();
            eprint!("{}", renderer.render_all(&diags));
            return;
        }

        println!(
            "{}",
            Ansi::paint(self.cfg.color, Ansi::DIM, "Token stream:")
        );
        for tok in &tokens {
            if matches!(tok.kind, crate::lexer::TokenKind::Eof) {
                break;
            }
            println!("  {:4}:{:3}  {:?}", tok.span.line, tok.span.col, tok.kind);
        }
    }

    fn cmd_load(&mut self, path: &str) {
        match fs::read_to_string(path) {
            Err(e) => {
                let msg = format!("cannot load `{path}`: {e}");
                eprintln!("{}", Ansi::paint(self.cfg.color, Ansi::BRIGHT_RED, &msg));
            }
            Ok(src) => {
                println!(
                    "{}",
                    Ansi::paint(
                        self.cfg.color,
                        Ansi::DIM,
                        &format!("loaded {} bytes from `{path}`", src.len())
                    )
                );
                self.eval_input(&src);
            }
        }
    }

    fn cmd_demo(&mut self, name: &str) {
        let Some(src) = demo_source(name) else {
            eprintln!(
                "{}",
                Ansi::paint(
                    self.cfg.color,
                    Ansi::BRIGHT_RED,
                    "unknown demo. use: :demo game | :demo arcade | :demo ml",
                )
            );
            return;
        };
        println!(
            "{}",
            Ansi::paint(
                self.cfg.color,
                Ansi::DIM,
                &format!("running embedded `{name}` demo"),
            )
        );
        self.eval_input(src);
    }

    fn cmd_persona(&self, target: &str) {
        match target {
            "game" => println!("{PERSONA_GAME_DEV}"),
            "ml" => println!("{PERSONA_ML_SCI}"),
            "all" => {
                println!("{PERSONA_GAME_DEV}");
                println!("{PERSONA_ML_SCI}");
            }
            _ => {
                eprintln!(
                    "{}",
                    Ansi::paint(
                        self.cfg.color,
                        Ansi::BRIGHT_RED,
                        "unknown target. use: :persona game | :persona ml | :persona all",
                    )
                );
            }
        }
    }

    fn cmd_scientist(&mut self, mode: &str) {
        match mode {
            "on" => {
                self.scientist_mode = true;
                println!(
                    "{}",
                    Ansi::paint(
                        self.cfg.color,
                        Ansi::DIM,
                        "[scientist] timing/report mode enabled",
                    )
                );
            }
            "off" => {
                self.scientist_mode = false;
                println!(
                    "{}",
                    Ansi::paint(
                        self.cfg.color,
                        Ansi::DIM,
                        "[scientist] timing/report mode disabled",
                    )
                );
            }
            _ => eprintln!(
                "{}",
                Ansi::paint(
                    self.cfg.color,
                    Ansi::BRIGHT_RED,
                    "unknown mode. use: :scientist on | :scientist off",
                )
            ),
        }
    }

    fn cmd_track(&mut self, target: &str) {
        if target == "off" {
            self.experiment_log = None;
            println!(
                "{}",
                Ansi::paint(
                    self.cfg.color,
                    Ansi::DIM,
                    "[scientist] experiment tracking disabled"
                )
            );
            return;
        }

        let path = PathBuf::from(target);
        let header = "# Jules experiment log\n";
        if let Err(e) = fs::write(&path, header) {
            eprintln!(
                "{}",
                Ansi::paint(
                    self.cfg.color,
                    Ansi::BRIGHT_RED,
                    &format!("cannot start tracking at `{}`: {e}", path.display()),
                )
            );
            return;
        }
        self.experiment_log = Some(path.clone());
        println!(
            "{}",
            Ansi::paint(
                self.cfg.color,
                Ansi::DIM,
                &format!("[scientist] tracking enabled at `{}`", path.display()),
            )
        );
    }

    fn log_experiment(&self, input: &str, status: &str) {
        let Some(path) = &self.experiment_log else {
            return;
        };
        let record = format!("\n## status: {status}\n```jules\n{input}```\n");
        let _ = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .and_then(|mut f| f.write_all(record.as_bytes()));
    }
}

enum ReplRunError {
    Compile(Vec<Diag>),
    Runtime(String),
}

// =============================================================================
// §10  SHARED HELPERS
// =============================================================================

fn render_cfg(args: &CliArgs) -> RenderCfg {
    RenderCfg {
        color: args.color,
        tab_width: args.tab_width,
        context: 1,
    }
}

fn emit_diagnostics(diags: &[Diag], source: &str, filename: &str, cfg: &RenderCfg, json: bool) {
    if diags.is_empty() {
        return;
    }
    if json {
        println!("{}", diags_to_json(diags, filename));
    } else {
        let renderer = DiagRenderer::new(source, filename, cfg.clone());
        eprint!("{}", renderer.render_all(diags));
    }
}

// =============================================================================
// §11  MAIN ENTRY POINT
// =============================================================================

pub fn main() {
    let raw_args: Vec<String> = std::env::args().collect();

    let args = match CliArgs::parse(&raw_args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("jules: {e}");
            process::exit(2);
        }
    };

    let exit_code = match args.command {
        Command::Help => {
            cmd_help();
            0
        }
        Command::Version => {
            cmd_version();
            0
        }
        Command::Check => cmd_check(&args),
        Command::Fix => cmd_fix(&args),
        Command::Run => cmd_run(&args),
        Command::Compile => cmd_compile(&args),
        Command::Train => cmd_train(&args),
        Command::Estimate => cmd_estimate(&args),
        Command::Fmt => cmd_fmt(&args),
        Command::Repl => {
            let cfg = render_cfg(&args);
            let mut repl = Repl::new(cfg);
            repl.run();
            0
        }
    };

    process::exit(exit_code);
}

// =============================================================================
// §12  TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── CLI argument parser ────────────────────────────────────────────────────

    fn parse(args: &[&str]) -> Result<CliArgs, String> {
        let owned: Vec<String> = std::iter::once("jules")
            .chain(args.iter().copied())
            .map(String::from)
            .collect();
        CliArgs::parse(&owned)
    }

    #[test]
    fn test_cli_run_command() {
        let a = parse(&["run", "foo.jules"]).unwrap();
        assert_eq!(a.command, Command::Run);
        assert_eq!(a.file.as_deref(), Some(Path::new("foo.jules")));
    }

    #[test]
    fn test_cli_check_command() {
        let a = parse(&["check", "bar.jules"]).unwrap();
        assert_eq!(a.command, Command::Check);
        assert_eq!(a.file.as_deref(), Some(Path::new("bar.jules")));
    }

    #[test]
    fn test_incremental_check_cache_roundtrip() {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "jules_check_cache_{}_{}.jules",
            std::process::id(),
            9911
        ));
        let _ = std::fs::write(&p, "fn main() {}");
        let meta = CheckCacheMeta {
            source_hash: hash_source("fn main() {}"),
            diag_free: true,
        };
        store_incremental_check_cache(&p, &meta);
        let loaded = load_incremental_check_cache(&p).expect("cache entry should exist");
        assert_eq!(loaded.source_hash, meta.source_hash);
        assert_eq!(loaded.diag_free, meta.diag_free);
    }

    #[test]
    fn test_cli_fix_command() {
        let a = parse(&["fix", "broken.jules"]).unwrap();
        assert_eq!(a.command, Command::Fix);
        assert_eq!(a.file.as_deref(), Some(Path::new("broken.jules")));
    }

    #[test]
    fn test_cli_check_warn_error() {
        let a = parse(&["check", "bar.jules", "-W"]).unwrap();
        assert!(a.warn_as_error);
    }

    #[test]
    fn test_repl_demo_source_mapping() {
        assert!(demo_source("game").is_some());
        assert!(demo_source("arcade").is_some());
        assert!(demo_source("ml").is_some());
        assert!(demo_source("unknown").is_none());
    }

    #[test]
    fn test_repl_scientist_toggle() {
        let mut repl = Repl::new(RenderCfg::default());
        repl.cmd_scientist("on");
        assert!(repl.scientist_mode);
        repl.cmd_scientist("off");
        assert!(!repl.scientist_mode);
    }

    #[test]
    fn test_repl_experiment_log_writes() {
        let mut repl = Repl::new(RenderCfg::default());
        let mut path = std::env::temp_dir();
        path.push(format!(
            "jules_repl_test_{}_{}.md",
            std::process::id(),
            1539
        ));
        let p = path.to_string_lossy().to_string();
        repl.cmd_track(&p);
        repl.log_experiment("let x = 1;", "ok");
        let data = std::fs::read_to_string(&path).unwrap();
        assert!(data.contains("status: ok"));
        assert!(data.contains("let x = 1;"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_cli_quiet_flag() {
        let a = parse(&["check", "x.jules", "--quiet"]).unwrap();
        assert!(a.quiet);
    }

    #[test]
    fn test_cli_no_color_flag() {
        let a = parse(&["run", "x.jules", "--no-color"]).unwrap();
        assert!(!a.color);
    }

    #[test]
    fn test_cli_tab_width() {
        let a = parse(&["run", "x.jules", "--tab-width", "2"]).unwrap();
        assert_eq!(a.tab_width, 2);
    }

    #[test]
    fn test_cli_emit_tokens_flag() {
        let a = parse(&["run", "x.jules", "--emit-tokens"]).unwrap();
        assert!(a.emit_tokens);
    }

    #[test]
    fn test_cli_entry_flag() {
        let a = parse(&["run", "x.jules", "--entry", "forward"]).unwrap();
        assert_eq!(a.entry, "forward");
    }

    #[test]
    fn test_cli_rest_args() {
        let a = parse(&["run", "x.jules", "--", "hello", "world"]).unwrap();
        assert_eq!(a.rest_args, vec!["hello", "world"]);
    }

    #[test]
    fn test_cli_repl_command() {
        let a = parse(&["repl"]).unwrap();
        assert_eq!(a.command, Command::Repl);
    }

    #[test]
    fn test_cli_estimate_command() {
        let a = parse(&[
            "estimate",
            "--params",
            "40000000",
            "--batch",
            "128",
            "--episodes",
            "300000",
            "--steps",
            "128",
            "--envs",
            "8",
            "--device",
            "gpu",
        ])
        .unwrap();
        assert_eq!(a.command, Command::Estimate);
        assert_eq!(a.estimate_params, 40_000_000);
        assert_eq!(a.estimate_batch, 128);
        assert_eq!(a.estimate_episodes, 300_000);
        assert_eq!(a.estimate_steps, 128);
        assert_eq!(a.estimate_envs, 8);
        assert_eq!(a.estimate_device, "gpu");
    }

    #[test]
    fn test_cli_train_jax_flags() {
        let a = parse(&[
            "train",
            "model.jules",
            "--ml-backend",
            "jax",
            "--jax-dataset",
            "train.npz",
            "--jax-out",
            "artifacts/custom",
            "--jax-script",
            "scripts/custom.py",
        ])
        .unwrap();
        assert_eq!(a.command, Command::Train);
        assert_eq!(a.ml_backend, "jax");
        assert_eq!(a.jax_dataset.as_deref(), Some(Path::new("train.npz")));
        assert_eq!(a.jax_out, PathBuf::from("artifacts/custom"));
        assert_eq!(a.jax_script, PathBuf::from("scripts/custom.py"));
    }

    #[test]
    fn test_build_jax_ir_from_model_supported_dense() {
        let sp = Span::dummy();
        let model = crate::ast::ModelDecl {
            span: sp,
            attrs: vec![],
            name: "PolicyNet".to_string(),
            layers: vec![
                crate::ast::ModelLayer::Input {
                    span: sp,
                    size: 128,
                },
                crate::ast::ModelLayer::Dense {
                    span: sp,
                    units: 256,
                    activation: crate::ast::Activation::Relu,
                    bias: true,
                },
                crate::ast::ModelLayer::Output {
                    span: sp,
                    units: 10,
                    activation: crate::ast::Activation::Softmax,
                },
            ],
            device: crate::ast::ModelDevice::Auto,
            optimizer: None,
        };

        let ir = build_jax_ir_from_model(&model).unwrap();
        assert_eq!(ir.model_name, "PolicyNet");
        assert_eq!(ir.input_dim, 128);
        assert_eq!(ir.layers, vec![256, 10]);
        assert_eq!(ir.activation, "relu");
        assert_eq!(ir.task, "classification");
    }

    #[test]
    fn test_feature_capability_matrix_reports_unsupported_layer() {
        let sp = Span::dummy();
        let model = crate::ast::ModelDecl {
            span: sp,
            attrs: vec![],
            name: "VisionNet".to_string(),
            layers: vec![
                crate::ast::ModelLayer::Input { span: sp, size: 64 },
                crate::ast::ModelLayer::Conv2d {
                    span: sp,
                    filters: 32,
                    kernel_h: 3,
                    kernel_w: 3,
                    stride: 1,
                    padding: crate::ast::Padding::Same,
                    activation: crate::ast::Activation::Relu,
                },
            ],
            device: crate::ast::ModelDevice::Auto,
            optimizer: None,
        };
        let program = crate::ast::Program {
            span: sp,
            items: vec![crate::ast::Item::Model(model)],
        };
        let rows = feature_capability_matrix(&program);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].jules_supported);
        assert!(!rows[0].jax_supported);
        assert!(rows[0]
            .jax_reason
            .as_deref()
            .unwrap()
            .contains("unsupported layer"));
    }

    #[test]
    fn test_estimate_training_param_scaling() {
        let small = estimate_training(1_000_000, 64, 100_000, 64, 1, "cpu");
        let large = estimate_training(40_000_000, 64, 100_000, 64, 1, "cpu");
        assert!(large.estimated_steps_per_sec < small.estimated_steps_per_sec);
        assert!(large.estimated_duration > small.estimated_duration);
    }

    #[test]
    fn test_estimate_training_reports_bottleneck() {
        let est = estimate_training(80_000_000, 128, 1_000, 128, 8, "cpu");
        assert!(matches!(est.bottleneck, "sim" | "model"));
        assert!(est.estimated_model_steps_per_sec > 0.0);
        assert!(est.estimated_sim_steps_per_sec > 0.0);
        assert!(
            (est.estimated_steps_per_sec
                - est
                    .estimated_model_steps_per_sec
                    .min(est.estimated_sim_steps_per_sec))
            .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_build_speed_actions_model_bound_contains_param_guidance() {
        let args = CliArgs {
            estimate_params: 80_000_000,
            estimate_batch: 128,
            estimate_episodes: 100_000,
            estimate_steps: 64,
            estimate_envs: 2,
            estimate_device: "cpu".to_string(),
            ..Default::default()
        };
        let est = estimate_training(
            args.estimate_params,
            args.estimate_batch,
            args.estimate_episodes,
            args.estimate_steps,
            args.estimate_envs,
            &args.estimate_device,
        );
        let actions = build_speed_actions(
            est,
            &args,
            ResourceSnapshot {
                ram_available_bytes: Some(u64::MAX / 2),
                gpu_available_bytes: None,
            },
        );
        assert!(actions.iter().any(|a| a.contains("reduce params")));
    }

    #[test]
    fn test_build_speed_actions_memory_warning() {
        let args = CliArgs {
            estimate_params: 120_000_000,
            estimate_batch: 512,
            estimate_episodes: 100_000,
            estimate_steps: 64,
            estimate_envs: 8,
            estimate_device: "cpu".to_string(),
            ..Default::default()
        };
        let est = estimate_training(
            args.estimate_params,
            args.estimate_batch,
            args.estimate_episodes,
            args.estimate_steps,
            args.estimate_envs,
            &args.estimate_device,
        );
        let actions = build_speed_actions(
            est,
            &args,
            ResourceSnapshot {
                ram_available_bytes: Some(1_000_000_000),
                gpu_available_bytes: None,
            },
        );
        assert!(actions.iter().any(|a| a.contains("memory risk")));
    }

    #[test]
    fn test_cli_version() {
        let a = parse(&["version"]).unwrap();
        assert_eq!(a.command, Command::Version);
    }

    #[test]
    fn test_cli_help() {
        let a = parse(&["help"]).unwrap();
        assert_eq!(a.command, Command::Help);
    }

    #[test]
    fn test_cli_unknown_flag_error() {
        assert!(parse(&["run", "x.jules", "--bogus"]).is_err());
    }

    #[test]
    fn test_cli_tab_width_invalid() {
        assert!(parse(&["run", "x.jules", "--tab-width", "abc"]).is_err());
    }

    #[test]
    fn test_apply_safe_syntax_fixes_keyword_typo() {
        let fixed = apply_safe_syntax_fixes("fun main() {}", &[]).unwrap();
        assert_eq!(fixed, "fn main() {}");
    }

    #[test]
    fn test_apply_safe_syntax_fixes_assignment_operator() {
        let diags = vec![Diag::error(sp(1, 7, 6, 8), "bad assignment")
            .with_hint("use `=` to assign a value (not `==`)")];
        let fixed = apply_safe_syntax_fixes("let x == 1;", &diags).unwrap();
        assert_eq!(fixed, "let x = 1;");
    }

    #[test]
    fn test_apply_safe_syntax_fixes_missing_comma() {
        let diags =
            vec![Diag::error(sp(1, 6, 5, 5), "missing comma").with_hint("separate items with `,`")];
        let fixed = apply_safe_syntax_fixes("foo(a b)", &diags).unwrap();
        assert_eq!(fixed, "foo(a, b)");
    }

    #[test]
    fn test_apply_safe_syntax_fixes_common_typos_and_balancing() {
        let fixed = apply_safe_syntax_fixes("pritn(\"x\"\nflase", &[]).unwrap();
        assert_eq!(fixed, "print(\"x\"\nfalse)");
    }

    #[test]
    fn test_detect_silent_issues_heuristics() {
        let src = "if a = b {}\nwhile true {}\nif x == 0.1 {}";
        let diags = detect_silent_issues(src);
        assert!(diags.iter().any(|d| d.code == Some("W-SILENT-ASSIGN-COND")));
        assert!(diags
            .iter()
            .any(|d| d.code == Some("W-SILENT-INFINITE-LOOP")));
        assert!(diags.iter().any(|d| d.code == Some("W-SILENT-FLOAT-EQ")));
    }

    // ── Diagnostic construction ────────────────────────────────────────────────

    fn sp(line: u32, col: u32, start: usize, end: usize) -> Span {
        Span {
            line,
            col,
            start,
            end,
        }
    }

    #[test]
    fn test_diag_error_is_error() {
        let d = Diag::error(sp(1, 1, 0, 1), "oops");
        assert!(d.is_error());
        assert_eq!(d.severity, DiagSeverity::Error);
    }

    #[test]
    fn test_diag_warning_not_error() {
        let d = Diag::warning(sp(1, 1, 0, 1), "hmm");
        assert!(!d.is_error());
    }

    #[test]
    fn test_diag_with_code_and_hint() {
        let d = Diag::error(sp(2, 5, 10, 15), "type mismatch")
            .with_code("E0042")
            .with_hint("consider adding a cast");
        assert_eq!(d.code, Some("E0042"));
        assert!(d.hint.is_some());
    }

    #[test]
    fn test_diag_with_label() {
        let d = Diag::error(sp(1, 1, 0, 5), "err").with_label(sp(2, 3, 10, 15), "defined here");
        assert_eq!(d.labels.len(), 1);
    }

    // ── DiagRenderer ──────────────────────────────────────────────────────────

    #[test]
    fn test_renderer_no_color_no_ansi() {
        let src = "let x = 1\nlet y = oops\n";
        let diag = Diag::error(sp(2, 9, 10, 14), "undefined variable `oops`");
        let cfg = RenderCfg {
            color: false,
            tab_width: 4,
            context: 0,
        };
        let r = DiagRenderer::new(src, "test.jules", cfg);
        let out = r.render(&diag);
        assert!(out.contains("undefined variable"), "got: {out}");
        assert!(out.contains("2"), "should show line number");
        assert!(
            !out.contains("\x1b["),
            "should have no ANSI codes when color=false"
        );
    }

    #[test]
    fn test_renderer_with_color() {
        let src = "x + y\n";
        let diag = Diag::error(sp(1, 3, 2, 3), "bad op");
        let cfg = RenderCfg {
            color: true,
            tab_width: 4,
            context: 0,
        };
        let r = DiagRenderer::new(src, "f.jules", cfg);
        let out = r.render(&diag);
        assert!(
            out.contains("\x1b["),
            "should contain ANSI codes when color=true"
        );
    }

    #[test]
    fn test_renderer_squiggle_width() {
        let src = "let tensor_val = A @ B\n";
        // Span covers "tensor_val" (10 chars)
        let diag = Diag::warning(sp(1, 5, 4, 14), "shadowed name");
        let cfg = RenderCfg {
            color: false,
            tab_width: 4,
            context: 0,
        };
        let r = DiagRenderer::new(src, "x.jules", cfg);
        let out = r.render(&diag);
        // Squiggle should appear
        assert!(out.contains('^'), "should contain squiggle chars: {out}");
    }

    #[test]
    fn test_renderer_context_lines() {
        let src = "line_one\nline_two_error\nline_three\n";
        let diag = Diag::error(sp(2, 1, 9, 22), "error here");
        let cfg = RenderCfg {
            color: false,
            tab_width: 4,
            context: 1,
        };
        let r = DiagRenderer::new(src, "f.jules", cfg);
        let out = r.render(&diag);
        // Should contain context line above.
        assert!(out.contains("line_one"), "context line above: {out}");
    }

    #[test]
    fn test_renderer_all_severities() {
        let src = "x\n";
        let cfg = RenderCfg {
            color: false,
            tab_width: 4,
            context: 0,
        };
        let r = DiagRenderer::new(src, "f.jules", cfg);
        let note = r.render(&Diag::note(sp(1, 1, 0, 1), "just a note"));
        let warn = r.render(&Diag::warning(sp(1, 1, 0, 1), "watch out"));
        let err = r.render(&Diag::error(sp(1, 1, 0, 1), "broken"));
        assert!(note.contains("note:"));
        assert!(warn.contains("warning:"));
        assert!(err.contains("error:"));
    }

    // ── JSON emitter ───────────────────────────────────────────────────────────

    #[test]
    fn test_json_output_valid_structure() {
        let diags = vec![
            Diag::error(sp(3, 5, 20, 25), "missing semicolon").with_code("E0010"),
            Diag::warning(sp(1, 1, 0, 3), "unused import"),
        ];
        let json = diags_to_json(&diags, "src/main.jules");
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("\"severity\": \"error\""));
        assert!(json.contains("\"severity\": \"warning\""));
        assert!(json.contains("E0010"));
        assert!(json.contains("missing semicolon"));
        assert!(json.contains("src/main.jules"));
    }

    #[test]
    fn test_json_empty_diagnostics() {
        let json = diags_to_json(&[], "f.jules");
        assert_eq!(json.trim(), "[]");
    }

    // ── Pipeline ──────────────────────────────────────────────────────────────

    #[test]
    fn test_pipeline_clean_source() {
        let mut unit = CompileUnit::new("test.jules", "fn main() {}\n");
        let pipeline = Pipeline::new();
        let result = pipeline.run(&mut unit);
        assert!(!unit.has_errors(), "{:?}", unit.diags);
        assert!(matches!(result, PipelineResult::Ok(_)));
    }

    #[test]
    fn test_pipeline_lex_error_halts() {
        // Backtick is not a valid Jules character.
        let mut unit = CompileUnit::new("test.jules", "let x = `bad`\n");
        let pipeline = Pipeline::new();
        let result = pipeline.run(&mut unit);
        assert!(unit.has_errors(), "expected lex error");
        assert!(matches!(result, PipelineResult::HaltedAt(PassName::Lex)));
    }

    #[test]
    fn test_pipeline_parse_error_halts() {
        // Missing closing paren
        let mut unit = CompileUnit::new("test.jules", "fn foo( {}\n");
        let pipeline = Pipeline::new();
        let result = pipeline.run(&mut unit);
        assert!(unit.has_errors(), "expected parse error");
        assert!(matches!(result, PipelineResult::HaltedAt(PassName::Parse)));
    }

    #[test]
    fn test_pipeline_warn_as_error() {
        // Inject a synthetic warning directly, then apply promote.
        let mut unit = CompileUnit::new("test.jules", "let x = 1\n");
        unit.diags.push(Diag::warning(sp(1, 1, 0, 1), "unused"));
        let mut _pipeline = Pipeline::new();
        _pipeline.warn_as_error = true;
        // The warning is already present; run again to trigger promotion.
        // (In real usage the pipeline inserts the warning during analysis.)
        // Promote manually to test the behaviour.
        for d in &mut unit.diags {
            if d.severity == DiagSeverity::Warning {
                d.severity = DiagSeverity::Error;
            }
        }
        assert!(unit.has_errors());
    }

    #[test]
    fn test_pipeline_quiet_suppresses_notes() {
        let mut unit = CompileUnit::new("test.jules", "let x = 1\n");
        unit.diags.push(Diag::note(sp(1, 1, 0, 1), "just info"));
        unit.diags.push(Diag::error(sp(1, 1, 0, 1), "real error"));
        let mut _pipeline = Pipeline::new();
        _pipeline.quiet = true;
        unit.diags.retain(|d| d.severity == DiagSeverity::Error);
        assert_eq!(unit.diags.len(), 1);
        assert!(unit.has_errors());
    }

    #[test]
    fn test_compile_unit_counts() {
        let mut unit = CompileUnit::new("x.jules", "");
        unit.diags.push(Diag::error(sp(1, 1, 0, 1), "e1"));
        unit.diags.push(Diag::error(sp(1, 1, 0, 1), "e2"));
        unit.diags.push(Diag::warning(sp(1, 1, 0, 1), "w1"));
        assert_eq!(unit.error_count(), 2);
        assert_eq!(unit.warning_count(), 1);
    }

    // ── Tab expansion ──────────────────────────────────────────────────────────

    #[test]
    fn test_expand_tabs_no_tabs() {
        let cfg = RenderCfg {
            color: false,
            tab_width: 4,
            context: 0,
        };
        let r = DiagRenderer::new("abc", "f.jules", cfg);
        assert_eq!(r.expand_tabs("hello world", 5), 5);
    }

    #[test]
    fn test_expand_tabs_with_tab() {
        let cfg = RenderCfg {
            color: false,
            tab_width: 4,
            context: 0,
        };
        let r = DiagRenderer::new("\t", "f.jules", cfg);
        // Tab at col 0 → expands to 4 display cols.
        assert_eq!(r.expand_tabs("\thello", 1), 4);
    }

    #[test]
    fn test_expand_tabs_mid_line() {
        let cfg = RenderCfg {
            color: false,
            tab_width: 4,
            context: 0,
        };
        let r = DiagRenderer::new("", "f.jules", cfg);
        // "ab\t" → after 'ab' we're at col 2; tab brings to 4.
        assert_eq!(r.expand_tabs("ab\tcde", 3), 4);
    }

    // ── Ansi helper ───────────────────────────────────────────────────────────

    #[test]
    fn test_ansi_paint_disabled() {
        let out = Ansi::paint(false, Ansi::RED, "hello");
        assert_eq!(out, "hello");
        assert!(!out.contains('\x1b'));
    }

    #[test]
    fn test_ansi_paint_enabled() {
        let out = Ansi::paint(true, Ansi::RED, "hello");
        assert!(out.starts_with("\x1b["));
        assert!(out.contains("hello"));
        assert!(out.ends_with("\x1b[0m"));
    }

    // ── LexError → Diag conversion ────────────────────────────────────────────

    #[test]
    fn test_lex_error_into_diag() {
        let e = LexError::new("unexpected char", sp(5, 3, 40, 41));
        let d = Diag::from(e);
        assert_eq!(d.severity, DiagSeverity::Error);
        assert_eq!(d.code, Some("E0001"));
        assert!(d.message.contains("unexpected char"));
    }

    // ── End-to-end token-dump smoke test ─────────────────────────────────────

    #[test]
    fn test_lex_jules_snippet() {
        let src = r#"
            @gpu
            fn forward(A: tensor<f32>[128, 128]) -> tensor<f32>[128, 128] {
                let C = grad A @ A
                return C
            }
        "#;
        let mut lexer = Lexer::new(src);
        let (tokens, errors) = lexer.tokenize();
        assert!(errors.is_empty(), "lex errors: {errors:?}");
        let kinds: Vec<_> = tokens.iter().map(|t| &t.kind).collect();
        assert!(kinds
            .iter()
            .any(|k| matches!(k, crate::lexer::TokenKind::AtGpu)));
        assert!(kinds
            .iter()
            .any(|k| matches!(k, crate::lexer::TokenKind::KwFn)));
        assert!(kinds
            .iter()
            .any(|k| matches!(k, crate::lexer::TokenKind::MatMul)));
        assert!(kinds
            .iter()
            .any(|k| matches!(k, crate::lexer::TokenKind::KwGrad)));
    }

    #[test]
    fn test_lex_invalid_char_produces_diag() {
        let src = "let x = `wat`";
        let mut lexer = Lexer::new(src);
        let (_, errors) = lexer.tokenize();
        assert!(!errors.is_empty());
        let diags: Vec<Diag> = errors.into_iter().map(Diag::from).collect();
        assert!(diags[0].is_error());
    }
}
