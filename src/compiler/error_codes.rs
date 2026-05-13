// =============================================================================
// jules/src/compiler/error_codes.rs
//
// Comprehensive error-code system for the Jules compiler.
//
// This module defines a typed ErrorCode for every diagnostic category in the
// compiler pipeline, together with structured hints (DiagnosticHint) and
// category grouping (ErrorCodeCategory).  It integrates with the existing
// diagnostic infrastructure:
//
//   - `Diag` in `main.rs` with `code: Option<&'static str>` field
//   - `typeck::Diagnostic` with `severity, span, message, notes`
//   - `sema::Diagnostic` with `severity, span, message, labels`
//   - `borrowck::Diagnostic` with `severity, span, message, labels`
//   - `RuntimeError` in `interp.rs` with `message, span`
//
// Previously only lexer (E0001), parser (E0002), and runtime (E9000) had codes.
// This module fills in every range so that each pass can attach a precise,
// stable error code to its diagnostics.
//
// Usage in adapter functions (main.rs):
//
//   fn adapt_typeck_diag(d: typeck::Diagnostic) -> Diag {
//       let code = ErrorCode::E2001;          // pick the right variant
//       Diag::error(d.span, d.message).with_code(code.code())
//   }
//
// =============================================================================

use std::fmt;

// =============================================================================
// §1  ERROR CODE CATEGORY
// =============================================================================

/// Top-level grouping for error codes, mirroring the pipeline passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCodeCategory {
    Lexer,
    Parser,
    TypeCheck,
    Semantic,
    BorrowCheck,
    Effect,
    Ownership,
    IR,
    ShaderGpuMl,
    Runtime,
}

impl fmt::Display for ErrorCodeCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCodeCategory::Lexer => write!(f, "lexer"),
            ErrorCodeCategory::Parser => write!(f, "parser"),
            ErrorCodeCategory::TypeCheck => write!(f, "type checker"),
            ErrorCodeCategory::Semantic => write!(f, "semantic analysis"),
            ErrorCodeCategory::BorrowCheck => write!(f, "borrow checker"),
            ErrorCodeCategory::Effect => write!(f, "effect system"),
            ErrorCodeCategory::Ownership => write!(f, "ownership/region"),
            ErrorCodeCategory::IR => write!(f, "IR / compilation"),
            ErrorCodeCategory::ShaderGpuMl => write!(f, "shader / GPU / ML pipeline"),
            ErrorCodeCategory::Runtime => write!(f, "runtime"),
        }
    }
}

// =============================================================================
// §2  ERROR CODE ENUM
// =============================================================================

/// A typed error code for every diagnostic category in the Jules compiler.
///
/// Each variant maps to a stable `"Exxxx"` string (e.g. `"E2001"`) and
/// carries a human-readable description via [`ErrorCode::description`].
///
/// # Code ranges
///
/// | Range   | Category         |
/// |---------|------------------|
/// | E0xxx   | Lexer            |
/// | E1xxx   | Parser           |
/// | E2xxx   | Type Checker     |
/// | E3xxx   | Semantic Analysis|
/// | E4xxx   | Borrow/Ownership |
/// | E5xxx   | Effect System    |
/// | E6xxx   | Ownership/Region |
/// | E7xxx   | IR / Compilation |
/// | E9xxx   | Runtime          |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    // ── E0xxx — Lexer ───────────────────────────────────────────────────────
    /// E0001 — Invalid token encountered during lexing.
    E0001,
    /// E0002 — Unterminated string literal.
    E0002,
    /// E0003 — Invalid escape sequence in string literal.
    E0003,
    /// E0004 — Invalid number literal (malformed or out of range).
    E0004,
    /// E0005 — Unterminated block comment.
    E0005,

    // ── E1xxx — Parser ──────────────────────────────────────────────────────
    /// E1001 — Expected a specific token that was not found.
    E1001,
    /// E1002 — Unexpected token encountered.
    E1002,
    /// E1003 — Expected an identifier.
    E1003,
    /// E1004 — Expected a type.
    E1004,
    /// E1005 — Expected an expression.
    E1005,
    /// E1006 — Expected an item (fn, struct, etc.).
    E1006,
    /// E1007 — Expected a pattern.
    E1007,
    /// E1008 — Duplicate field in struct literal or pattern.
    E1008,
    /// E1009 — Invalid assignment target.
    E1009,

    // ── E2xxx — Type Checker ────────────────────────────────────────────────
    /// E2001 — Type mismatch between expected and actual types.
    E2001,
    /// E2002 — Unknown type name (no struct, component, or enum declared).
    E2002,
    /// E2003 — Undefined variable or name.
    E2003,
    /// E2004 — Binary operator type mismatch.
    E2004,
    /// E2005 — Unary operator type mismatch.
    E2005,
    /// E2006 — Cannot compare incompatible types.
    E2006,
    /// E2007 — Cannot index a non-indexable type.
    E2007,
    /// E2008 — No such field on a struct / component / enum.
    E2008,
    /// E2009 — Cannot assign to an immutable binding.
    E2009,
    /// E2010 — Call on a non-function type.
    E2010,
    /// E2011 — Wrong number of arguments in function call.
    E2011,
    /// E2012 — Argument type mismatch in function call.
    E2012,
    /// E2013 — Return type mismatch in function body.
    E2013,
    /// E2014 — Tensor shape mismatch (static shape mismatch).
    E2014,
    /// E2015 — Tensor element type mismatch.
    E2015,
    /// E2016 — Tensor rank mismatch.
    E2016,
    /// E2017 — Invalid tensor operation for the given types.
    E2017,
    /// E2018 — Array length is not a compile-time constant.
    E2018,
    /// E2019 — Function parameter missing type annotation.
    E2019,
    /// E2020 — System parameter requires explicit type annotation.
    E2020,
    /// E2021 — Operation cannot be applied to a non-tensor type.
    E2021,
    /// E2022 — @grad applied to a non-differentiable type.
    E2022,
    /// E2023 — Swizzle error (invalid swizzle mask or component count).
    E2023,
    /// E2024 — Model layer type or shape consistency error.
    E2024,
    /// E2025 — Train block semantic error.
    E2025,
    /// E2026 — Potential data race between systems.
    E2026,
    /// E2027 — Generic bounds not yet implemented.
    E2027,
    /// E2028 — Occurs check failed (infinite type).
    E2028,
    /// E2029 — Non-exhaustive match expression.
    E2029,

    // ── E3xxx — Semantic Analysis ───────────────────────────────────────────
    /// E3001 — Unused variable binding.
    E3001,
    /// E3002 — Variable shadowing an outer binding.
    E3002,
    /// E3003 — Undeclared component referenced in a query.
    E3003,
    /// E3004 — Duplicate component in a single query clause.
    E3004,
    /// E3005 — Component appears in both `with` and `without`.
    E3005,
    /// E3006 — Unconstrained parallel query (empty `with`).
    E3006,
    /// E3007 — Read-write aliasing between systems.
    E3007,
    /// E3008 — Duplicate struct / component field.
    E3008,
    /// E3009 — Agent declares no behaviours.
    E3009,
    /// E3010 — Duplicate behaviour priority within an agent.
    E3010,
    /// E3011 — Duplicate perception kind within an agent.
    E3011,
    /// E3012 — Invalid memory capacity (zero or negative).
    E3012,
    /// E3013 — Invalid learning rate (non-positive or NaN).
    E3013,
    /// E3014 — Invalid discount factor (outside [0, 1]).
    E3014,
    /// E3015 — Side effect in utility expression.
    E3015,
    /// E3016 — Model layer consistency error (e.g. missing input/output).
    E3016,
    /// E3017 — Agent with learning declared but no train block.
    E3017,
    /// E3018 — `break` used outside a loop.
    E3018,
    /// E3019 — `continue` used outside a loop.
    E3019,
    /// E3020 — `return` used outside a function.
    E3020,
    /// E3021 — `await` used outside an async function.
    E3021,
    /// E3022 — Nested `spawn` block.
    E3022,
    /// E3023 — Nested `sync` block.
    E3023,
    /// E3024 — Nested `atomic` block.
    E3024,

    // ── E4xxx — Borrow / Ownership ──────────────────────────────────────────
    /// E4001 — Use of a moved value.
    E4001,
    /// E4002 — Borrow of a moved value.
    E4002,
    /// E4003 — Mutable borrow inside a parallel / spawned block.
    E4003,
    /// E4004 — Borrow while the target is already borrowed.
    E4004,
    /// E4005 — Immutable borrow while the target is mutably borrowed.
    E4005,
    /// E4006 — Assignment to a moved value.
    E4006,
    /// E4007 — Assignment while the target is borrowed.
    E4007,
    /// E4008 — Move inside a parallel / spawned block.
    E4008,
    /// E4009 — Move while the target is borrowed.
    E4009,
    /// E4010 — Assignment through an immutable reference.
    E4010,
    /// E4011 — Dereference of a non-reference for assignment.
    E4011,
    /// E4012 — Unsupported dereference assignment target.
    E4012,

    // ── E5xxx — Effect System ───────────────────────────────────────────────
    /// E5001 — Pure code depends on IO effect.
    E5001,
    /// E5002 — IO effect not declared in function signature.
    E5002,
    /// E5003 — Effect boundary violation (crossing a purity boundary).
    E5003,
    /// E5004 — Mutation outside of a declared effect.
    E5004,
    /// E5005 — Side effect in a pure context.
    E5005,
    /// E5006 — Unsafe IO access without proper annotation.
    E5006,

    // ── E6xxx — Ownership / Region ──────────────────────────────────────────
    /// E6001 — Region lifetime exceeded.
    E6001,
    /// E6002 — Region borrow conflict.
    E6002,
    /// E6003 — Owned value used after transfer.
    E6003,
    /// E6004 — Copy of a non-copy type.
    E6004,
    /// E6005 — Mutable aliasing violation.
    E6005,
    /// E6006 — Thread ownership conflict.
    E6006,
    /// E6007 — Shared mutable access without atomic.
    E6007,

    // ── E7xxx — IR / Compilation ────────────────────────────────────────────
    /// E7001 — IR node kind not covered by the lowering pass.
    E7001,
    /// E7002 — IR lowering failed.
    E7002,
    /// E7003 — Invalid IR operation.
    E7003,
    /// E7004 — SSA validation failed.
    E7004,
    /// E7005 — Block terminator (branch/return) missing.
    E7005,
    /// E7006 — PHI node conflict.
    E7006,
    /// E7007 — Type annotation missing in IR.
    E7007,
    /// E7008 — Effect capability violation (operation requires unlisted capability).
    E7008,
    /// E7009 — Comptime evaluation failed.
    E7009,
    /// E7010 — Trait method not satisfied for this type.
    E7010,
    /// E7011 — Module visibility violation (accessing private item).
    E7011,
    /// E7012 — ECS system scheduling conflict (race detected).
    E7012,
    /// E7013 — IR capability set inconsistency.
    E7013,

    // ── E8xxx — Shader / GPU / ML Pipeline ───────────────────────────────────
    /// E8001 — Shader stage missing (shader must have at least one stage).
    E8001,
    /// E8002 — Shader resource binding conflict (duplicate group/binding).
    E8002,
    /// E8003 — Shader vertex stage missing required output.
    E8003,
    /// E8004 — Shader fragment stage missing required output.
    E8004,
    /// E8005 — Invalid shader workgroup size.
    E8005,
    /// E8006 — GPU dispatch with incompatible tensor layout.
    E8006,
    /// E8007 — ML graph mode conflict (eager op in static graph context).
    E8007,
    /// E8008 — ML autodiff on non-differentiable operation.
    E8008,
    /// E8009 — Tensor layout mismatch for kernel (expected NHWC, got NCHW).
    E8009,
    /// E8010 — Kernel fusion boundary violation.
    E8010,
    /// E8011 — GPU capability required but not declared in function signature.
    E8011,
    /// E8012 — SIMD capability required but not declared in function signature.
    E8012,

    // ── E9xxx — Runtime ─────────────────────────────────────────────────────
    /// E9001 — Type error at runtime (dynamic type check failure).
    E9001,
    /// E9002 — Index out of bounds.
    E9002,
    /// E9003 — Tensor shape mismatch at runtime.
    E9003,
    /// E9004 — Undefined function called at runtime.
    E9004,
    /// E9005 — Field not found on a runtime value.
    E9005,
    /// E9006 — Component not found in ECS world.
    E9006,
    /// E9007 — Division by zero.
    E9007,
    /// E9008 — Arithmetic overflow.
    E9008,
    /// E9009 — Null / None unwrap.
    E9009,
    /// E9010 — Task join error.
    E9010,
    /// E9011 — Region access after free.
    E9011,
    /// E9012 — Match exhaustiveness failure at runtime.
    E9012,
    /// E9999 — Unknown / unclassified runtime error.
    E9999,
}

impl ErrorCode {
    /// Return the stable string code for this error, e.g. `"E2001"`.
    pub fn code(self) -> &'static str {
        match self {
            // Lexer
            ErrorCode::E0001 => "E0001",
            ErrorCode::E0002 => "E0002",
            ErrorCode::E0003 => "E0003",
            ErrorCode::E0004 => "E0004",
            ErrorCode::E0005 => "E0005",
            // Parser
            ErrorCode::E1001 => "E1001",
            ErrorCode::E1002 => "E1002",
            ErrorCode::E1003 => "E1003",
            ErrorCode::E1004 => "E1004",
            ErrorCode::E1005 => "E1005",
            ErrorCode::E1006 => "E1006",
            ErrorCode::E1007 => "E1007",
            ErrorCode::E1008 => "E1008",
            ErrorCode::E1009 => "E1009",
            // Type Checker
            ErrorCode::E2001 => "E2001",
            ErrorCode::E2002 => "E2002",
            ErrorCode::E2003 => "E2003",
            ErrorCode::E2004 => "E2004",
            ErrorCode::E2005 => "E2005",
            ErrorCode::E2006 => "E2006",
            ErrorCode::E2007 => "E2007",
            ErrorCode::E2008 => "E2008",
            ErrorCode::E2009 => "E2009",
            ErrorCode::E2010 => "E2010",
            ErrorCode::E2011 => "E2011",
            ErrorCode::E2012 => "E2012",
            ErrorCode::E2013 => "E2013",
            ErrorCode::E2014 => "E2014",
            ErrorCode::E2015 => "E2015",
            ErrorCode::E2016 => "E2016",
            ErrorCode::E2017 => "E2017",
            ErrorCode::E2018 => "E2018",
            ErrorCode::E2019 => "E2019",
            ErrorCode::E2020 => "E2020",
            ErrorCode::E2021 => "E2021",
            ErrorCode::E2022 => "E2022",
            ErrorCode::E2023 => "E2023",
            ErrorCode::E2024 => "E2024",
            ErrorCode::E2025 => "E2025",
            ErrorCode::E2026 => "E2026",
            ErrorCode::E2027 => "E2027",
            ErrorCode::E2028 => "E2028",
            ErrorCode::E2029 => "E2029",
            // Semantic Analysis
            ErrorCode::E3001 => "E3001",
            ErrorCode::E3002 => "E3002",
            ErrorCode::E3003 => "E3003",
            ErrorCode::E3004 => "E3004",
            ErrorCode::E3005 => "E3005",
            ErrorCode::E3006 => "E3006",
            ErrorCode::E3007 => "E3007",
            ErrorCode::E3008 => "E3008",
            ErrorCode::E3009 => "E3009",
            ErrorCode::E3010 => "E3010",
            ErrorCode::E3011 => "E3011",
            ErrorCode::E3012 => "E3012",
            ErrorCode::E3013 => "E3013",
            ErrorCode::E3014 => "E3014",
            ErrorCode::E3015 => "E3015",
            ErrorCode::E3016 => "E3016",
            ErrorCode::E3017 => "E3017",
            ErrorCode::E3018 => "E3018",
            ErrorCode::E3019 => "E3019",
            ErrorCode::E3020 => "E3020",
            ErrorCode::E3021 => "E3021",
            ErrorCode::E3022 => "E3022",
            ErrorCode::E3023 => "E3023",
            ErrorCode::E3024 => "E3024",
            // Borrow / Ownership
            ErrorCode::E4001 => "E4001",
            ErrorCode::E4002 => "E4002",
            ErrorCode::E4003 => "E4003",
            ErrorCode::E4004 => "E4004",
            ErrorCode::E4005 => "E4005",
            ErrorCode::E4006 => "E4006",
            ErrorCode::E4007 => "E4007",
            ErrorCode::E4008 => "E4008",
            ErrorCode::E4009 => "E4009",
            ErrorCode::E4010 => "E4010",
            ErrorCode::E4011 => "E4011",
            ErrorCode::E4012 => "E4012",
            // Effect System
            ErrorCode::E5001 => "E5001",
            ErrorCode::E5002 => "E5002",
            ErrorCode::E5003 => "E5003",
            ErrorCode::E5004 => "E5004",
            ErrorCode::E5005 => "E5005",
            ErrorCode::E5006 => "E5006",
            // Ownership / Region
            ErrorCode::E6001 => "E6001",
            ErrorCode::E6002 => "E6002",
            ErrorCode::E6003 => "E6003",
            ErrorCode::E6004 => "E6004",
            ErrorCode::E6005 => "E6005",
            ErrorCode::E6006 => "E6006",
            ErrorCode::E6007 => "E6007",
            // IR / Compilation
            ErrorCode::E7001 => "E7001",
            ErrorCode::E7002 => "E7002",
            ErrorCode::E7003 => "E7003",
            ErrorCode::E7004 => "E7004",
            ErrorCode::E7005 => "E7005",
            ErrorCode::E7006 => "E7006",
            ErrorCode::E7007 => "E7007",
            ErrorCode::E7008 => "E7008",
            ErrorCode::E7009 => "E7009",
            ErrorCode::E7010 => "E7010",
            ErrorCode::E7011 => "E7011",
            ErrorCode::E7012 => "E7012",
            ErrorCode::E7013 => "E7013",
            // Shader / GPU / ML Pipeline
            ErrorCode::E8001 => "E8001",
            ErrorCode::E8002 => "E8002",
            ErrorCode::E8003 => "E8003",
            ErrorCode::E8004 => "E8004",
            ErrorCode::E8005 => "E8005",
            ErrorCode::E8006 => "E8006",
            ErrorCode::E8007 => "E8007",
            ErrorCode::E8008 => "E8008",
            ErrorCode::E8009 => "E8009",
            ErrorCode::E8010 => "E8010",
            ErrorCode::E8011 => "E8011",
            ErrorCode::E8012 => "E8012",
            // Runtime
            ErrorCode::E9001 => "E9001",
            ErrorCode::E9002 => "E9002",
            ErrorCode::E9003 => "E9003",
            ErrorCode::E9004 => "E9004",
            ErrorCode::E9005 => "E9005",
            ErrorCode::E9006 => "E9006",
            ErrorCode::E9007 => "E9007",
            ErrorCode::E9008 => "E9008",
            ErrorCode::E9009 => "E9009",
            ErrorCode::E9010 => "E9010",
            ErrorCode::E9011 => "E9011",
            ErrorCode::E9012 => "E9012",
            ErrorCode::E9999 => "E9999",
        }
    }

    /// Return a human-readable description of what this error code represents.
    pub fn description(self) -> &'static str {
        match self {
            // Lexer
            ErrorCode::E0001 => "invalid token",
            ErrorCode::E0002 => "unterminated string literal",
            ErrorCode::E0003 => "invalid escape sequence",
            ErrorCode::E0004 => "invalid number literal",
            ErrorCode::E0005 => "unterminated block comment",
            // Parser
            ErrorCode::E1001 => "expected token",
            ErrorCode::E1002 => "unexpected token",
            ErrorCode::E1003 => "expected identifier",
            ErrorCode::E1004 => "expected type",
            ErrorCode::E1005 => "expected expression",
            ErrorCode::E1006 => "expected item",
            ErrorCode::E1007 => "expected pattern",
            ErrorCode::E1008 => "duplicate field",
            ErrorCode::E1009 => "invalid assignment target",
            // Type Checker
            ErrorCode::E2001 => "type mismatch",
            ErrorCode::E2002 => "unknown type",
            ErrorCode::E2003 => "undefined variable",
            ErrorCode::E2004 => "binary operator type mismatch",
            ErrorCode::E2005 => "unary operator type mismatch",
            ErrorCode::E2006 => "cannot compare types",
            ErrorCode::E2007 => "cannot index type",
            ErrorCode::E2008 => "no such field",
            ErrorCode::E2009 => "cannot assign to immutable binding",
            ErrorCode::E2010 => "call on a non-function type",
            ErrorCode::E2011 => "wrong argument count",
            ErrorCode::E2012 => "argument type mismatch",
            ErrorCode::E2013 => "return type mismatch",
            ErrorCode::E2014 => "tensor shape mismatch",
            ErrorCode::E2015 => "tensor element type mismatch",
            ErrorCode::E2016 => "tensor rank mismatch",
            ErrorCode::E2017 => "invalid tensor operation",
            ErrorCode::E2018 => "array length is not a compile-time constant",
            ErrorCode::E2019 => "parameter missing type annotation",
            ErrorCode::E2020 => "system parameter requires explicit type",
            ErrorCode::E2021 => "operation cannot be applied to non-tensor type",
            ErrorCode::E2022 => "@grad on non-differentiable type",
            ErrorCode::E2023 => "swizzle error",
            ErrorCode::E2024 => "model layer error",
            ErrorCode::E2025 => "train block error",
            ErrorCode::E2026 => "potential data race between systems",
            ErrorCode::E2027 => "generic bounds not yet implemented",
            ErrorCode::E2028 => "occurs check failed (infinite type)",
            ErrorCode::E2029 => "non-exhaustive match",
            // Semantic Analysis
            ErrorCode::E3001 => "unused variable",
            ErrorCode::E3002 => "variable shadowing",
            ErrorCode::E3003 => "undeclared component",
            ErrorCode::E3004 => "duplicate component in query",
            ErrorCode::E3005 => "component in both with/without",
            ErrorCode::E3006 => "unconstrained parallel query",
            ErrorCode::E3007 => "read-write aliasing between systems",
            ErrorCode::E3008 => "duplicate struct or component field",
            ErrorCode::E3009 => "agent declares no behaviours",
            ErrorCode::E3010 => "duplicate behaviour priority",
            ErrorCode::E3011 => "duplicate perception kind",
            ErrorCode::E3012 => "invalid memory capacity",
            ErrorCode::E3013 => "invalid learning rate",
            ErrorCode::E3014 => "invalid discount factor",
            ErrorCode::E3015 => "side effect in utility expression",
            ErrorCode::E3016 => "model layer consistency error",
            ErrorCode::E3017 => "agent with learning but no train block",
            ErrorCode::E3018 => "break outside loop",
            ErrorCode::E3019 => "continue outside loop",
            ErrorCode::E3020 => "return outside function",
            ErrorCode::E3021 => "await outside async function",
            ErrorCode::E3022 => "nested spawn block",
            ErrorCode::E3023 => "nested sync block",
            ErrorCode::E3024 => "nested atomic block",
            // Borrow / Ownership
            ErrorCode::E4001 => "use after move",
            ErrorCode::E4002 => "borrow after move",
            ErrorCode::E4003 => "mutable borrow in parallel block",
            ErrorCode::E4004 => "borrow while borrowed",
            ErrorCode::E4005 => "immutable borrow while mutably borrowed",
            ErrorCode::E4006 => "assign to moved value",
            ErrorCode::E4007 => "assign while borrowed",
            ErrorCode::E4008 => "move in parallel block",
            ErrorCode::E4009 => "move while borrowed",
            ErrorCode::E4010 => "assign through immutable reference",
            ErrorCode::E4011 => "deref non-reference for assignment",
            ErrorCode::E4012 => "unsupported deref assignment target",
            // Effect System
            ErrorCode::E5001 => "pure code depends on IO",
            ErrorCode::E5002 => "IO effect not declared",
            ErrorCode::E5003 => "effect boundary violation",
            ErrorCode::E5004 => "mutation outside declared effect",
            ErrorCode::E5005 => "side effect in pure context",
            ErrorCode::E5006 => "unsafe IO access",
            // Ownership / Region
            ErrorCode::E6001 => "region lifetime exceeded",
            ErrorCode::E6002 => "region borrow conflict",
            ErrorCode::E6003 => "owned value used after transfer",
            ErrorCode::E6004 => "copy of non-copy type",
            ErrorCode::E6005 => "mutable aliasing violation",
            ErrorCode::E6006 => "thread ownership conflict",
            ErrorCode::E6007 => "shared mutable without atomic",
            // IR / Compilation
            ErrorCode::E7001 => "IR node not covered",
            ErrorCode::E7002 => "lowering failed",
            ErrorCode::E7003 => "invalid IR operation",
            ErrorCode::E7004 => "SSA validation failed",
            ErrorCode::E7005 => "block terminator missing",
            ErrorCode::E7006 => "PHI node conflict",
            ErrorCode::E7007 => "type annotation missing in IR",
            ErrorCode::E7008 => "effect capability violation",
            ErrorCode::E7009 => "comptime evaluation failed",
            ErrorCode::E7010 => "trait method not satisfied",
            ErrorCode::E7011 => "module visibility violation",
            ErrorCode::E7012 => "ECS system scheduling conflict",
            ErrorCode::E7013 => "capability set inconsistency",
            // Shader / GPU / ML Pipeline
            ErrorCode::E8001 => "shader stage missing",
            ErrorCode::E8002 => "shader resource binding conflict",
            ErrorCode::E8003 => "shader vertex output missing",
            ErrorCode::E8004 => "shader fragment output missing",
            ErrorCode::E8005 => "invalid shader workgroup size",
            ErrorCode::E8006 => "GPU dispatch tensor layout mismatch",
            ErrorCode::E8007 => "ML graph mode conflict",
            ErrorCode::E8008 => "autodiff on non-differentiable operation",
            ErrorCode::E8009 => "tensor layout mismatch for kernel",
            ErrorCode::E8010 => "kernel fusion boundary violation",
            ErrorCode::E8011 => "GPU capability not declared",
            ErrorCode::E8012 => "SIMD capability not declared",
            // Runtime
            ErrorCode::E9001 => "type error at runtime",
            ErrorCode::E9002 => "index out of bounds",
            ErrorCode::E9003 => "tensor shape mismatch at runtime",
            ErrorCode::E9004 => "undefined function at runtime",
            ErrorCode::E9005 => "field not found",
            ErrorCode::E9006 => "component not found",
            ErrorCode::E9007 => "division by zero",
            ErrorCode::E9008 => "arithmetic overflow",
            ErrorCode::E9009 => "null or None unwrap",
            ErrorCode::E9010 => "task join error",
            ErrorCode::E9011 => "region access after free",
            ErrorCode::E9012 => "match exhaustiveness failure",
            ErrorCode::E9999 => "unknown runtime error",
        }
    }

    /// Return the category for this error code.
    pub fn category(self) -> ErrorCodeCategory {
        match self {
            ErrorCode::E0001 | ErrorCode::E0002 | ErrorCode::E0003
            | ErrorCode::E0004 | ErrorCode::E0005 => ErrorCodeCategory::Lexer,

            ErrorCode::E1001 | ErrorCode::E1002 | ErrorCode::E1003
            | ErrorCode::E1004 | ErrorCode::E1005 | ErrorCode::E1006
            | ErrorCode::E1007 | ErrorCode::E1008 | ErrorCode::E1009 => ErrorCodeCategory::Parser,

            ErrorCode::E2001 | ErrorCode::E2002 | ErrorCode::E2003
            | ErrorCode::E2004 | ErrorCode::E2005 | ErrorCode::E2006
            | ErrorCode::E2007 | ErrorCode::E2008 | ErrorCode::E2009
            | ErrorCode::E2010 | ErrorCode::E2011 | ErrorCode::E2012
            | ErrorCode::E2013 | ErrorCode::E2014 | ErrorCode::E2015
            | ErrorCode::E2016 | ErrorCode::E2017 | ErrorCode::E2018
            | ErrorCode::E2019 | ErrorCode::E2020 | ErrorCode::E2021
            | ErrorCode::E2022 | ErrorCode::E2023 | ErrorCode::E2024
            | ErrorCode::E2025 | ErrorCode::E2026 | ErrorCode::E2027
            | ErrorCode::E2028 | ErrorCode::E2029 => ErrorCodeCategory::TypeCheck,

            ErrorCode::E3001 | ErrorCode::E3002 | ErrorCode::E3003
            | ErrorCode::E3004 | ErrorCode::E3005 | ErrorCode::E3006
            | ErrorCode::E3007 | ErrorCode::E3008 | ErrorCode::E3009
            | ErrorCode::E3010 | ErrorCode::E3011 | ErrorCode::E3012
            | ErrorCode::E3013 | ErrorCode::E3014 | ErrorCode::E3015
            | ErrorCode::E3016 | ErrorCode::E3017 | ErrorCode::E3018
            | ErrorCode::E3019 | ErrorCode::E3020 | ErrorCode::E3021
            | ErrorCode::E3022 | ErrorCode::E3023 | ErrorCode::E3024 => ErrorCodeCategory::Semantic,

            ErrorCode::E4001 | ErrorCode::E4002 | ErrorCode::E4003
            | ErrorCode::E4004 | ErrorCode::E4005 | ErrorCode::E4006
            | ErrorCode::E4007 | ErrorCode::E4008 | ErrorCode::E4009
            | ErrorCode::E4010 | ErrorCode::E4011 | ErrorCode::E4012 => ErrorCodeCategory::BorrowCheck,

            ErrorCode::E5001 | ErrorCode::E5002 | ErrorCode::E5003
            | ErrorCode::E5004 | ErrorCode::E5005 | ErrorCode::E5006 => ErrorCodeCategory::Effect,

            ErrorCode::E6001 | ErrorCode::E6002 | ErrorCode::E6003
            | ErrorCode::E6004 | ErrorCode::E6005 | ErrorCode::E6006
            | ErrorCode::E6007 => ErrorCodeCategory::Ownership,

            ErrorCode::E7001 | ErrorCode::E7002 | ErrorCode::E7003
            | ErrorCode::E7004 | ErrorCode::E7005 | ErrorCode::E7006
            | ErrorCode::E7007 | ErrorCode::E7008 | ErrorCode::E7009
            | ErrorCode::E7010 | ErrorCode::E7011 | ErrorCode::E7012
            | ErrorCode::E7013 => ErrorCodeCategory::IR,

            ErrorCode::E8001 | ErrorCode::E8002 | ErrorCode::E8003
            | ErrorCode::E8004 | ErrorCode::E8005 | ErrorCode::E8006
            | ErrorCode::E8007 | ErrorCode::E8008 | ErrorCode::E8009
            | ErrorCode::E8010 | ErrorCode::E8011 | ErrorCode::E8012 => ErrorCodeCategory::ShaderGpuMl,

            ErrorCode::E9001 | ErrorCode::E9002 | ErrorCode::E9003
            | ErrorCode::E9004 | ErrorCode::E9005 | ErrorCode::E9006
            | ErrorCode::E9007 | ErrorCode::E9008 | ErrorCode::E9009
            | ErrorCode::E9010 | ErrorCode::E9011 | ErrorCode::E9012
            | ErrorCode::E9999 => ErrorCodeCategory::Runtime,
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

// =============================================================================
// §3  DIAGNOSTIC HINT
// =============================================================================

/// Structured hint types that can be attached to any diagnostic.
///
/// These provide actionable suggestions that the diagnostic renderer can
/// display as `help:` lines.  Each variant carries the minimal context
/// needed to produce a human-readable suggestion.
#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticHint {
    /// Suggest adding a type annotation: `name: suggested_type`
    AddTypeAnnotation {
        name: String,
        suggested_type: String,
    },

    /// Suggest adding `mut` to a binding: `let mut name`
    AddMutableKeyword {
        name: String,
    },

    /// Suggest replacing `from` with `to`
    ReplaceWith {
        from: String,
        to: String,
    },

    /// Suggest declaring `name` before its first use
    DeclareBeforeUse {
        name: String,
    },

    /// Suggest adding an effect annotation: `#[effect(effect)]`
    AddEffectAnnotation {
        effect: String,
    },

    /// Suggest adding a region annotation: `region(region_name)`
    AddRegion {
        region_name: String,
    },

    /// Suggest using `.copy()` on `name`
    UseCopy {
        name: String,
    },

    /// Suggest using `.clone()` on `name`
    UseClone {
        name: String,
    },

    /// Suggest using a reference: `&name` or `&mut name`
    UseRef {
        name: String,
    },

    /// Suggest adding a match arm for `pattern`
    AddMatchArm {
        pattern: String,
    },

    /// Suggest propagating with `?`: `name?`
    PropagateWithTry {
        name: String,
    },

    /// Suggest an explicit cast: `from as to`
    UseExplicitCast {
        from: String,
        to: String,
    },

    /// Suggest removing mutation on `name`
    RemoveMutation {
        name: String,
    },

    /// Suggest adding a capability annotation: `effect(gpu)` or `effect(simd)`.
    AddCapability {
        capability: String,
    },

    /// Suggest adding a shader stage to the shader definition.
    AddShaderStage {
        stage: String,
    },

    /// Suggest declaring the correct graph mode for an ML function.
    AddGraphMode {
        mode: String,
    },

    /// Suggest adding ECS system read/write declarations.
    AddSystemAccess {
        component: String,
        access: String,
    },

    /// Suggest implementing a trait method.
    ImplementTraitMethod {
        trait_name: String,
        method: String,
    },

    /// Suggest declaring comptime for a compile-time expression.
    AddComptime {
        expression: String,
    },

    /// Suggest adding a visibility modifier.
    AddVisibility {
        item: String,
        visibility: String,
    },

    /// A free-form hint that doesn't fit any structured variant.
    Custom(String),
}

impl DiagnosticHint {
    /// Produce a human-readable hint string for display in diagnostic output.
    pub fn to_hint_string(&self) -> String {
        match self {
            DiagnosticHint::AddTypeAnnotation { name, suggested_type } => {
                format!("add a type annotation: `{name}: {suggested_type}`")
            }
            DiagnosticHint::AddMutableKeyword { name } => {
                format!("declare `{name}` as mutable with `let mut {name}`")
            }
            DiagnosticHint::ReplaceWith { from, to } => {
                format!("replace `{from}` with `{to}`")
            }
            DiagnosticHint::DeclareBeforeUse { name } => {
                format!("declare `{name}` before its first use")
            }
            DiagnosticHint::AddEffectAnnotation { effect } => {
                format!("add effect annotation: `#[effect({effect})]`")
            }
            DiagnosticHint::AddRegion { region_name } => {
                format!("add a region annotation: `region({region_name})`")
            }
            DiagnosticHint::UseCopy { name } => {
                format!("use `.copy()` on `{name}` if the type implements copy semantics")
            }
            DiagnosticHint::UseClone { name } => {
                format!("use `.clone()` on `{name}` to create an owned copy before the move")
            }
            DiagnosticHint::UseRef { name } => {
                format!("borrow `{name}` by reference with `&{name}` or `&mut {name}`")
            }
            DiagnosticHint::AddMatchArm { pattern } => {
                format!("add a match arm for the pattern `{pattern}`")
            }
            DiagnosticHint::PropagateWithTry { name } => {
                format!("propagate the error with `{name}?`")
            }
            DiagnosticHint::UseExplicitCast { from, to } => {
                format!("use an explicit cast: `{from} as {to}`")
            }
            DiagnosticHint::RemoveMutation { name } => {
                format!("remove the mutation of `{name}` or use an immutable reference")
            }
            DiagnosticHint::AddCapability { capability } => {
                format!("add capability annotation: `effect({capability})` to the function signature")
            }
            DiagnosticHint::AddShaderStage { stage } => {
                format!("add a `{stage}` stage to the shader definition")
            }
            DiagnosticHint::AddGraphMode { mode } => {
                format!("declare the function with `@{mode}` to set the execution mode")
            }
            DiagnosticHint::AddSystemAccess { component, access } => {
                format!("declare `{access}` access to component `{component}` in the system signature")
            }
            DiagnosticHint::ImplementTraitMethod { trait_name, method } => {
                format!("implement `{method}` from trait `{trait_name}` for this type")
            }
            DiagnosticHint::AddComptime { expression } => {
                format!("mark `{expression}` with `comptime` to evaluate at compile time")
            }
            DiagnosticHint::AddVisibility { item, visibility } => {
                format!("add `{visibility}` visibility to `{item}` to make it accessible from this scope")
            }
            DiagnosticHint::Custom(msg) => msg.clone(),
        }
    }
}

impl fmt::Display for DiagnosticHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hint_string())
    }
}

// =============================================================================
// §4  HELPER FUNCTIONS
// =============================================================================

/// Parse any `"Exxxx"` string into its category.
///
/// Returns `ErrorCodeCategory::Runtime` as a fallback for unrecognised codes.
///
/// # Examples
///
/// ```
/// use crate::compiler::error_codes::{error_code_category, ErrorCodeCategory};
/// assert_eq!(error_code_category("E2001"), ErrorCodeCategory::TypeCheck);
/// assert_eq!(error_code_category("E4003"), ErrorCodeCategory::BorrowCheck);
/// ```
pub fn error_code_category(code: &str) -> ErrorCodeCategory {
    if code.len() < 2 || !code.starts_with('E') {
        return ErrorCodeCategory::Runtime; // fallback
    }

    // Parse the numeric portion after 'E'.
    let num: u32 = code[1..].parse().unwrap_or(9999);

    // Determine category by range.
    match num {
        0..=999 => ErrorCodeCategory::Lexer,
        1000..=1999 => ErrorCodeCategory::Parser,
        2000..=2999 => ErrorCodeCategory::TypeCheck,
        3000..=3999 => ErrorCodeCategory::Semantic,
        4000..=4999 => ErrorCodeCategory::BorrowCheck,
        5000..=5999 => ErrorCodeCategory::Effect,
        6000..=6999 => ErrorCodeCategory::Ownership,
        7000..=7999 => ErrorCodeCategory::IR,
        8000..=8999 => ErrorCodeCategory::ShaderGpuMl,
        9000..=9999 => ErrorCodeCategory::Runtime,
        _ => ErrorCodeCategory::Runtime,
    }
}

/// Returns `true` if the given code string is an error code (starts with `E`).
///
/// This distinguishes error codes from warning codes (which start with `W`)
/// or other diagnostic codes.
pub fn is_error(code: &str) -> bool {
    code.starts_with('E')
}

/// Returns `true` if this code can be downgraded from an error to a warning
/// under certain lint configurations.
///
/// Certain diagnostics represent style or best-practice issues that the user
/// may want to treat as warnings rather than hard errors.  The following
/// categories are eligible:
///
/// - **E3xxx (Semantic)**: many are lints (unused variable, shadowing, etc.)
/// - **E5xxx (Effect)**: effect violations can be warnings in permissive mode
///
/// **IMPORTANT**: Ownership / borrow-check codes (E4xxx) are NEVER eligible
/// for downgrading.  Ownership violations are memory-safety issues and must
/// ALWAYS be hard errors.  This is non-negotiable.
///
/// All other categories are hard errors that cannot be downgraded.
pub fn is_warning_eligible(code: &str) -> bool {
    if !is_error(code) {
        return false;
    }

    let num: u32 = code[1..].parse().unwrap_or(0);

    // Semantic analysis codes (E3xxx) are often lints.
    if (3000..=3999).contains(&num) {
        // A few semantic codes are hard errors even within E3xxx:
        // E3003 (undeclared component), E3005 (component in both with/without),
        // E3018–E3024 (control-flow integrity) cannot be downgraded.
        return !matches!(
            num,
            3003 | 3005 | 3018 | 3019 | 3020 | 3021 | 3022 | 3023 | 3024
        );
    }

    // Effect system codes (E5xxx) can be warnings in permissive mode.
    if (5000..=5999).contains(&num) {
        return true;
    }

    // OWNERSHIP IS ALWAYS A HARD ERROR.
    // No borrow-check / ownership code (E4xxx) can ever be downgraded to a
    // warning.  E4003, E4004, E4005 were previously "advisory" aliasing
    // diagnostics, but ownership violations are memory-safety issues and
    // must always be hard errors.  This is non-negotiable.
    // if matches!(num, 4003 | 4004 | 4005) {
    //     return true;
    // }

    false
}

// =============================================================================
// §5  TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_code_returns_correct_string() {
        assert_eq!(ErrorCode::E0001.code(), "E0001");
        assert_eq!(ErrorCode::E2001.code(), "E2001");
        assert_eq!(ErrorCode::E4001.code(), "E4001");
        assert_eq!(ErrorCode::E9999.code(), "E9999");
    }

    #[test]
    fn error_code_display_impl() {
        assert_eq!(format!("{}", ErrorCode::E2001), "E2001");
    }

    #[test]
    fn error_code_description_is_nonempty() {
        // Every variant should have a non-empty description.
        let codes = all_error_codes();
        for code in codes {
            assert!(
                !code.description().is_empty(),
                "ErrorCode::{:?} has an empty description",
                code
            );
        }
    }

    #[test]
    fn category_matches_code_range() {
        assert_eq!(ErrorCode::E0001.category(), ErrorCodeCategory::Lexer);
        assert_eq!(ErrorCode::E1001.category(), ErrorCodeCategory::Parser);
        assert_eq!(ErrorCode::E2001.category(), ErrorCodeCategory::TypeCheck);
        assert_eq!(ErrorCode::E3001.category(), ErrorCodeCategory::Semantic);
        assert_eq!(ErrorCode::E4001.category(), ErrorCodeCategory::BorrowCheck);
        assert_eq!(ErrorCode::E5001.category(), ErrorCodeCategory::Effect);
        assert_eq!(ErrorCode::E6001.category(), ErrorCodeCategory::Ownership);
        assert_eq!(ErrorCode::E7001.category(), ErrorCodeCategory::IR);
        assert_eq!(ErrorCode::E9001.category(), ErrorCodeCategory::Runtime);
    }

    #[test]
    fn error_code_category_from_string() {
        assert_eq!(error_code_category("E0001"), ErrorCodeCategory::Lexer);
        assert_eq!(error_code_category("E1009"), ErrorCodeCategory::Parser);
        assert_eq!(error_code_category("E2029"), ErrorCodeCategory::TypeCheck);
        assert_eq!(error_code_category("E3024"), ErrorCodeCategory::Semantic);
        assert_eq!(error_code_category("E4012"), ErrorCodeCategory::BorrowCheck);
        assert_eq!(error_code_category("E5006"), ErrorCodeCategory::Effect);
        assert_eq!(error_code_category("E6007"), ErrorCodeCategory::Ownership);
        assert_eq!(error_code_category("E7007"), ErrorCodeCategory::IR);
        assert_eq!(error_code_category("E9999"), ErrorCodeCategory::Runtime);
    }

    #[test]
    fn error_code_category_fallback() {
        assert_eq!(error_code_category("W0001"), ErrorCodeCategory::Runtime);
        assert_eq!(error_code_category("garbage"), ErrorCodeCategory::Runtime);
        assert_eq!(error_code_category("E"), ErrorCodeCategory::Runtime);
    }

    #[test]
    fn is_error_predicate() {
        assert!(is_error("E0001"));
        assert!(is_error("E9999"));
        assert!(!is_error("W0001"));
        assert!(!is_error(""));
        assert!(!is_error("e0001")); // lowercase not accepted
    }

    #[test]
    fn is_warning_eligible_semantic_lints() {
        // E3001 (unused variable) and E3002 (shadowing) are warning-eligible.
        assert!(is_warning_eligible("E3001"));
        assert!(is_warning_eligible("E3002"));
        // E3003 (undeclared component) is a hard error.
        assert!(!is_warning_eligible("E3003"));
        // E3018 (break outside loop) is a hard error.
        assert!(!is_warning_eligible("E3018"));
    }

    #[test]
    fn is_warning_eligible_effects() {
        assert!(is_warning_eligible("E5001"));
        assert!(is_warning_eligible("E5006"));
    }

    #[test]
    fn is_warning_eligible_borrow_always_hard_error() {
        // Ownership/borrow-check codes are NEVER warning-eligible.
        // Ownership violations are memory-safety issues — always hard errors.
        assert!(!is_warning_eligible("E4001"), "E4001 (use-after-move) must be hard error");
        assert!(!is_warning_eligible("E4002"), "E4002 (borrow of moved value) must be hard error");
        assert!(!is_warning_eligible("E4003"), "E4003 (mut borrow in parallel) must be hard error");
        assert!(!is_warning_eligible("E4004"), "E4004 (borrow while borrowed) must be hard error");
        assert!(!is_warning_eligible("E4005"), "E4005 (imm borrow while mut borrowed) must be hard error");
        assert!(!is_warning_eligible("E4006"), "E4006 (assign to moved value) must be hard error");
    }

    #[test]
    fn is_warning_eligible_other_categories() {
        // Lexer, parser, typeck, IR, and runtime are not eligible.
        assert!(!is_warning_eligible("E0001"));
        assert!(!is_warning_eligible("E1001"));
        assert!(!is_warning_eligible("E2001"));
        assert!(!is_warning_eligible("E7001"));
        assert!(!is_warning_eligible("E9001"));
    }

    #[test]
    fn hint_strings_are_readable() {
        let hint = DiagnosticHint::AddTypeAnnotation {
            name: "x".into(),
            suggested_type: "f32".into(),
        };
        assert_eq!(hint.to_hint_string(), "add a type annotation: `x: f32`");

        let hint = DiagnosticHint::AddMutableKeyword { name: "x".into() };
        assert!(hint.to_hint_string().contains("let mut x"));

        let hint = DiagnosticHint::UseClone { name: "data".into() };
        assert!(hint.to_hint_string().contains(".clone()"));

        let hint = DiagnosticHint::AddMatchArm { pattern: "_".into() };
        assert!(hint.to_hint_string().contains("`_`"));

        let hint = DiagnosticHint::Custom("try something else".into());
        assert_eq!(hint.to_hint_string(), "try something else");
    }

    #[test]
    fn hint_display_impl() {
        let hint = DiagnosticHint::UseRef { name: "x".into() };
        assert!(format!("{}", hint).contains("&x"));
    }

    // ── Helper: enumerate all ErrorCode variants ────────────────────────────

    fn all_error_codes() -> Vec<ErrorCode> {
        vec![
            ErrorCode::E0001, ErrorCode::E0002, ErrorCode::E0003,
            ErrorCode::E0004, ErrorCode::E0005,
            ErrorCode::E1001, ErrorCode::E1002, ErrorCode::E1003,
            ErrorCode::E1004, ErrorCode::E1005, ErrorCode::E1006,
            ErrorCode::E1007, ErrorCode::E1008, ErrorCode::E1009,
            ErrorCode::E2001, ErrorCode::E2002, ErrorCode::E2003,
            ErrorCode::E2004, ErrorCode::E2005, ErrorCode::E2006,
            ErrorCode::E2007, ErrorCode::E2008, ErrorCode::E2009,
            ErrorCode::E2010, ErrorCode::E2011, ErrorCode::E2012,
            ErrorCode::E2013, ErrorCode::E2014, ErrorCode::E2015,
            ErrorCode::E2016, ErrorCode::E2017, ErrorCode::E2018,
            ErrorCode::E2019, ErrorCode::E2020, ErrorCode::E2021,
            ErrorCode::E2022, ErrorCode::E2023, ErrorCode::E2024,
            ErrorCode::E2025, ErrorCode::E2026, ErrorCode::E2027,
            ErrorCode::E2028, ErrorCode::E2029,
            ErrorCode::E3001, ErrorCode::E3002, ErrorCode::E3003,
            ErrorCode::E3004, ErrorCode::E3005, ErrorCode::E3006,
            ErrorCode::E3007, ErrorCode::E3008, ErrorCode::E3009,
            ErrorCode::E3010, ErrorCode::E3011, ErrorCode::E3012,
            ErrorCode::E3013, ErrorCode::E3014, ErrorCode::E3015,
            ErrorCode::E3016, ErrorCode::E3017, ErrorCode::E3018,
            ErrorCode::E3019, ErrorCode::E3020, ErrorCode::E3021,
            ErrorCode::E3022, ErrorCode::E3023, ErrorCode::E3024,
            ErrorCode::E4001, ErrorCode::E4002, ErrorCode::E4003,
            ErrorCode::E4004, ErrorCode::E4005, ErrorCode::E4006,
            ErrorCode::E4007, ErrorCode::E4008, ErrorCode::E4009,
            ErrorCode::E4010, ErrorCode::E4011, ErrorCode::E4012,
            ErrorCode::E5001, ErrorCode::E5002, ErrorCode::E5003,
            ErrorCode::E5004, ErrorCode::E5005, ErrorCode::E5006,
            ErrorCode::E6001, ErrorCode::E6002, ErrorCode::E6003,
            ErrorCode::E6004, ErrorCode::E6005, ErrorCode::E6006,
            ErrorCode::E6007,
            ErrorCode::E7001, ErrorCode::E7002, ErrorCode::E7003,
            ErrorCode::E7004, ErrorCode::E7005, ErrorCode::E7006,
            ErrorCode::E7007,
            ErrorCode::E9001, ErrorCode::E9002, ErrorCode::E9003,
            ErrorCode::E9004, ErrorCode::E9005, ErrorCode::E9006,
            ErrorCode::E9007, ErrorCode::E9008, ErrorCode::E9009,
            ErrorCode::E9010, ErrorCode::E9011, ErrorCode::E9012,
            ErrorCode::E9999,
        ]
    }
}
