// =============================================================================
// Semantic Superoptimizer (Level 4: Algorithmic Superoptimization / Semantic Lifting)
//
// The final frontier: instead of asking "How do I run these steps faster?",
// this optimizer asks "What is the user actually trying to do, and is there
// a better mathematical formula for it?"
//
// How it works:
//   If you write a `for` loop to calculate Fibonacci, a normal compiler
//   makes the loop fast. A semantic superoptimizer recognizes the pattern,
//   proves your loop is computing Fibonacci, deletes your code, and replaces
//   it with Binet's Formula (O(1) mathematical shortcut) or a lookup table.
//
// Impact: Instead of 10% speedup → 10,000,000% speedup by rewriting the
// algorithm entirely.
//
// Architecture: Two-tier rewrite system
//
//   Tier 1 — Expression-shape rewrite rules (egg-style, 100+ rules)
//     Each rule is a (matcher, rewriter) pair operating on expression trees.
//     Rules fire eagerly in a bottom-up pass.  A rule has:
//       • A `match` step that walks the AST shape and binds named capture vars.
//       • A `rewrite` step that constructs the replacement from captures.
//     Rules are pure functions; they never fail silently — they return
//     Option<SemanticOptResult> and the driver picks the best match.
//
//   Tier 2 — Loop-level patterns
//     Whole-loop analysis that detects accumulation / Fibonacci / prefix-sum
//     / reduce patterns in for-loop bodies, replacing the entire loop with
//     a closed-form expression.
//
// Currently recognised patterns (expression level, 100+):
//   Algebraic identity rules (commutativity, associativity, distributivity)
//   Constant folding (arith, boolean, comparison)
//   Identity / annihilator / absorbing element rules
//   Bit-trick rules (multiply/divide by powers of two)
//   Strength-reduction rules (pow → shift, mul → shift, etc.)
//   Closed-form series rules (triangular, squares, cubes, arithmetic, geometric)
//   Special-function call rules (fib, factorial, gcd, lcm, abs, min, max, …)
//   Boolean algebra & De Morgan rules
//   Comparison normalisation rules
//   Redundant operation elimination rules
//   Loop-invariant hoisting markers
// =============================================================================

use crate::compiler::ast::*;
use crate::Span;

// ─────────────────────────────────────────────────────────────────────────────
// Public API types
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a semantic superoptimization attempt.
#[derive(Debug, Clone)]
pub struct SemanticOptResult {
    /// The original expression (for verification / debugging).
    pub original: Expr,
    /// The optimized replacement.
    pub optimized: Expr,
    /// Which rule fired.
    pub pattern_name: String,
    /// Estimated speedup factor (e.g., 1_000_000.0 for O(n) → O(1)).
    pub estimated_speedup: f64,
}

/// A single rewrite rule: a named (matcher, rewriter) pair.
/// The matcher extracts a `RuleCapture` from an expression, and the
/// rewriter turns that capture into an optimized expression.
struct RewriteRule {
    name: &'static str,
    speedup: f64,
    apply: fn(&SemanticSuperoptimizer, &Expr) -> Option<Expr>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Capture type — named sub-expressions extracted by a rule's matcher
// ─────────────────────────────────────────────────────────────────────────────

/// Named captures produced by a pattern match.
/// Fields are filled on demand; unused fields remain `None`.
#[derive(Default, Clone)]
struct Cap {
    e0: Option<Expr>,   // generic slot 0
    e1: Option<Expr>,   // generic slot 1
    e2: Option<Expr>,   // generic slot 2
    n0: Option<u128>,   // integer literal slot 0
    n1: Option<u128>,   // integer literal slot 1
    s0: Option<String>, // identifier name slot 0
    s1: Option<String>, // identifier name slot 1
}

// ─────────────────────────────────────────────────────────────────────────────
// Main optimiser struct
// ─────────────────────────────────────────────────────────────────────────────

/// The semantic superoptimizer recognizes common algorithmic patterns
/// and replaces them with mathematically-equivalent closed-form solutions.
pub struct SemanticSuperoptimizer {
    /// Statistics
    pub patterns_matched: u64,
    pub total_estimated_speedup: f64,
    /// Whether to verify equivalence with random testing.
    pub verify_equivalence: bool,
    /// Number of random test inputs for verification.
    pub verif_inputs: usize,
    /// All expression-level rewrite rules, in priority order.
    rules: Vec<RewriteRule>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Rule table — 100+ rewrite rules in egg style
//
// Organisation (same ordering used at runtime):
//
//   §1   Constant folding                       rules  1-20
//   §2   Identity / annihilator elements        rules 21-35
//   §3   Algebraic normalisation                rules 36-55
//   §4   Bit tricks & strength reduction        rules 56-70
//   §5   Boolean algebra & De Morgan            rules 71-82
//   §6   Comparison normalisation               rules 83-90
//   §7   Closed-form series                     rules 91-98
//   §8   Special-function rewrites              rules 99-109
//   §9   Redundant-operation elimination        rules 110-116
//   §10  Loop-invariant marker                  rules 117-120
// ─────────────────────────────────────────────────────────────────────────────

impl SemanticSuperoptimizer {
    pub fn new() -> Self {
        Self {
            patterns_matched: 0,
            total_estimated_speedup: 0.0,
            verify_equivalence: true,
            verif_inputs: 16,
            rules: Self::build_rule_table(),
        }
    }

    // ─── Rule table construction ──────────────────────────────────────────

    fn build_rule_table() -> Vec<RewriteRule> {
        // Each entry is:  ( "name", speedup_factor, apply_fn )
        // apply_fn returns Some(optimized_expr) or None.
        macro_rules! rule {
            ($name:expr, $speedup:expr, $fn:expr) => {
                RewriteRule { name: $name, speedup: $speedup, apply: $fn }
            };
        }

        vec![
            // ── §1 Constant folding ──────────────────────────────────────────
            // R1  lit + lit
            rule!("cf_add",      1.1, |s, e| s.cf_binop(e, BinOpKind::Add,  |a,b| a.wrapping_add(b))),
            // R2  lit - lit
            rule!("cf_sub",      1.1, |s, e| s.cf_binop(e, BinOpKind::Sub,  |a,b| a.wrapping_sub(b))),
            // R3  lit * lit
            rule!("cf_mul",      1.1, |s, e| s.cf_binop(e, BinOpKind::Mul,  |a,b| a.wrapping_mul(b))),
            // R4  lit / lit  (non-zero divisor)
            rule!("cf_div",      1.1, |s, e| s.cf_binop_guard(e, BinOpKind::Div,  |a,b| if b != 0 { Some(a/b) } else { None })),
            // R5  lit % lit  (non-zero divisor)
            rule!("cf_rem",      1.1, |s, e| s.cf_binop_guard(e, BinOpKind::Rem,  |a,b| if b != 0 { Some(a%b) } else { None })),
            // R6  lit ** lit  (small exponent only to avoid huge constants)
            rule!("cf_pow",      1.1, |s, e| s.cf_pow(e)),
            // R7  lit & lit
            rule!("cf_bitand",   1.1, |s, e| s.cf_binop(e, BinOpKind::BitAnd, |a,b| a & b)),
            // R8  lit | lit
            rule!("cf_bitor",    1.1, |s, e| s.cf_binop(e, BinOpKind::BitOr,  |a,b| a | b)),
            // R9  lit ^ lit
            rule!("cf_bitxor",   1.1, |s, e| s.cf_binop(e, BinOpKind::BitXor, |a,b| a ^ b)),
            // R10 lit << lit  (bounded shift)
            rule!("cf_shl",      1.1, |s, e| s.cf_shift(e, BinOpKind::Shl)),
            // R11 lit >> lit
            rule!("cf_shr",      1.1, |s, e| s.cf_shift(e, BinOpKind::Shr)),
            // R12 -lit
            rule!("cf_neg",      1.1, |s, e| s.cf_neg(e)),
            // R13 !lit  (boolean)
            rule!("cf_not_bool", 1.1, |s, e| s.cf_not(e)),
            // R14 lit == lit
            rule!("cf_eq",       1.1, |s, e| s.cf_cmp(e, BinOpKind::Eq,  |a,b| a==b)),
            // R15 lit != lit
            rule!("cf_ne",       1.1, |s, e| s.cf_cmp(e, BinOpKind::Ne,  |a,b| a!=b)),
            // R16 lit < lit
            rule!("cf_lt",       1.1, |s, e| s.cf_cmp(e, BinOpKind::Lt,  |a,b| a<b)),
            // R17 lit <= lit
            rule!("cf_le",       1.1, |s, e| s.cf_cmp(e, BinOpKind::Le,  |a,b| a<=b)),
            // R18 lit > lit
            rule!("cf_gt",       1.1, |s, e| s.cf_cmp(e, BinOpKind::Gt,  |a,b| a>b)),
            // R19 lit >= lit
            rule!("cf_ge",       1.1, |s, e| s.cf_cmp(e, BinOpKind::Ge,  |a,b| a>=b)),
            // R20 true && true, false || false, etc.
            rule!("cf_bool_and_or", 1.1, |s, e| s.cf_bool_binop(e)),

            // ── §2 Identity / annihilator / absorbing elements ────────────────
            // R21 x + 0  →  x
            rule!("add_zero_r",   1.0, |s, e| s.identity_right(e, BinOpKind::Add, 0)),
            // R22 0 + x  →  x
            rule!("add_zero_l",   1.0, |s, e| s.identity_left(e, BinOpKind::Add, 0)),
            // R23 x - 0  →  x
            rule!("sub_zero",     1.0, |s, e| s.identity_right(e, BinOpKind::Sub, 0)),
            // R24 x * 1  →  x
            rule!("mul_one_r",    1.0, |s, e| s.identity_right(e, BinOpKind::Mul, 1)),
            // R25 1 * x  →  x
            rule!("mul_one_l",    1.0, |s, e| s.identity_left(e, BinOpKind::Mul, 1)),
            // R26 x * 0  →  0
            rule!("mul_zero_r",   1.0, |s, e| s.annihilator_right(e, BinOpKind::Mul, 0)),
            // R27 0 * x  →  0
            rule!("mul_zero_l",   1.0, |s, e| s.annihilator_left(e, BinOpKind::Mul, 0)),
            // R28 x / 1  →  x
            rule!("div_one",      1.0, |s, e| s.identity_right(e, BinOpKind::Div, 1)),
            // R29 0 / x  →  0  (x ≠ 0)
            rule!("zero_div",     1.0, |s, e| s.zero_div(e)),
            // R30 x ^ 0  →  1  (pow)
            rule!("pow_zero",     1.0, |s, e| s.pow_zero(e)),
            // R31 x ^ 1  →  x  (pow)
            rule!("pow_one",      1.0, |s, e| s.pow_one(e)),
            // R32 1 ^ x  →  1  (pow)
            rule!("one_pow",      1.0, |s, e| s.one_pow(e)),
            // R33 x | 0  →  x
            rule!("bitor_zero",   1.0, |s, e| s.identity_right(e, BinOpKind::BitOr, 0)),
            // R34 x & !0  →  x  (all-ones mask identity, represented as x & MAX_U128)
            rule!("bitand_allones", 1.0, |s, e| s.bitand_allones(e)),
            // R35 x ^ 0  →  x  (bitxor)
            rule!("bitxor_zero",  1.0, |s, e| s.identity_right(e, BinOpKind::BitXor, 0)),

            // ── §3 Algebraic normalisation ─────────────────────────────────────
            // R36 x - x  →  0
            rule!("sub_self",     1.0, |s, e| s.sub_self(e)),
            // R37 x / x  →  1  (x ≠ 0)
            rule!("div_self",     1.0, |s, e| s.div_self(e)),
            // R38 x ^ x  →  0  (bitxor self = 0)
            rule!("xor_self",     1.0, |s, e| s.xor_self(e)),
            // R39 x & x  →  x
            rule!("and_self",     1.0, |s, e| s.idempotent(e, BinOpKind::BitAnd)),
            // R40 x | x  →  x
            rule!("or_self",      1.0, |s, e| s.idempotent(e, BinOpKind::BitOr)),
            // R41 (x + y) + z  →  x + (y + z)  (right-associativity for canonicalisation)
            rule!("assoc_add",    1.0, |s, e| s.reassoc(e, BinOpKind::Add)),
            // R42 (x * y) * z  →  x * (y * z)
            rule!("assoc_mul",    1.0, |s, e| s.reassoc(e, BinOpKind::Mul)),
            // R43 x * (y + z) → x*y + x*z  (distribute mul over add — useful to expose constants)
            rule!("distrib_mul_add", 1.0, |s, e| s.distribute_mul_add(e)),
            // R44 (x + y) * z → x*z + y*z
            rule!("distrib_add_mul", 1.0, |s, e| s.distribute_add_mul(e)),
            // R45 -(x + y)  →  (-x) + (-y)  (push negation inward for constant folding)
            rule!("neg_push_add", 1.0, |s, e| s.neg_push(e, BinOpKind::Add)),
            // R46 -(x * y)  →  (-x) * y
            rule!("neg_push_mul", 1.0, |s, e| s.neg_push(e, BinOpKind::Mul)),
            // R47 x - y  →  x + (-y)  (normalise subtraction)
            rule!("sub_to_add_neg", 1.0, |s, e| s.sub_to_add_neg(e)),
            // R48 --x  →  x  (double negation)
            rule!("double_neg",   1.0, |s, e| s.double_neg(e)),
            // R49 x + x  →  2 * x
            rule!("add_self_to_mul2", 1.0, |s, e| s.add_self(e)),
            // R50 x * 2  →  x + x  (sometimes cheaper; separate from shift)
            rule!("mul2_to_add",  1.0, |s, e| s.mul2_to_add(e)),
            // R51 (x * a) + (x * b)  →  x * (a + b)  (factor out common term)
            rule!("factor_common_mul", 1.0, |s, e| s.factor_common(e, BinOpKind::Add, BinOpKind::Mul)),
            // R52 (x + a) - (x + b)  →  a - b
            rule!("cancel_add",   1.0, |s, e| s.cancel_add(e)),
            // R53 x % 1  →  0
            rule!("rem_one",      1.0, |s, e| s.rem_one(e)),
            // R54 x % x  →  0
            rule!("rem_self",     1.0, |s, e| s.rem_self(e)),
            // R55 (x % m) % m  →  x % m  (idempotent modulo)
            rule!("rem_idem",     1.0, |s, e| s.rem_idem(e)),

            // ── §4 Bit tricks & strength reduction ───────────────────────────
            // R56 x * 2^k  →  x << k
            rule!("mul_pow2_to_shl",  1.5, |s, e| s.mul_pow2_to_shl(e)),
            // R57 x / 2^k  →  x >> k  (unsigned)
            rule!("div_pow2_to_shr",  1.5, |s, e| s.div_pow2_to_shr(e)),
            // R58 x % 2^k  →  x & (2^k - 1)
            rule!("rem_pow2_to_and",  1.5, |s, e| s.rem_pow2_to_and(e)),
            // R59 x * 3  →  (x << 1) + x
            rule!("mul3_strength",    1.3, |s, e| s.mul_small_const_strength(e, 3)),
            // R60 x * 5  →  (x << 2) + x
            rule!("mul5_strength",    1.3, |s, e| s.mul_small_const_strength(e, 5)),
            // R61 x * 6  →  (x << 2) + (x << 1)
            rule!("mul6_strength",    1.3, |s, e| s.mul6_strength(e)),
            // R62 x * 7  →  (x << 3) - x
            rule!("mul7_strength",    1.3, |s, e| s.mul7_strength(e)),
            // R63 x * 9  →  (x << 3) + x
            rule!("mul9_strength",    1.3, |s, e| s.mul_small_const_strength(e, 9)),
            // R64 x * 10 →  (x << 3) + (x << 1)
            rule!("mul10_strength",   1.3, |s, e| s.mul10_strength(e)),
            // R65 x << 0 →  x
            rule!("shl_zero",         1.0, |s, e| s.shift_zero(e, BinOpKind::Shl)),
            // R66 x >> 0 →  x
            rule!("shr_zero",         1.0, |s, e| s.shift_zero(e, BinOpKind::Shr)),
            // R67 (x << a) << b  →  x << (a + b)
            rule!("merge_shl",        1.2, |s, e| s.merge_shifts(e, BinOpKind::Shl)),
            // R68 (x >> a) >> b  →  x >> (a + b)
            rule!("merge_shr",        1.2, |s, e| s.merge_shifts(e, BinOpKind::Shr)),
            // R69 x & (x - 1)  →  x with lowest bit cleared (zero if power of two)
            rule!("clear_lowest_bit", 1.2, |s, e| s.clear_lowest_bit(e)),
            // R70 (x | y) & ~y  →  x & ~y  (simplify mask combination)
            rule!("mask_simplify",    1.1, |s, e| s.mask_simplify(e)),

            // ── §5 Boolean algebra & De Morgan ───────────────────────────────
            // R71  !(a && b)  →  !a || !b
            rule!("demorgan_and",  1.0, |s, e| s.demorgan(e, BinOpKind::And, BinOpKind::Or)),
            // R72  !(a || b)  →  !a && !b
            rule!("demorgan_or",   1.0, |s, e| s.demorgan(e, BinOpKind::Or,  BinOpKind::And)),
            // R73  a && true  →  a
            rule!("and_true_r",    1.0, |s, e| s.bool_identity_right(e, BinOpKind::And, true)),
            // R74  true && a  →  a
            rule!("and_true_l",    1.0, |s, e| s.bool_identity_left(e, BinOpKind::And, true)),
            // R75  a || false  →  a
            rule!("or_false_r",    1.0, |s, e| s.bool_identity_right(e, BinOpKind::Or, false)),
            // R76  false || a  →  a
            rule!("or_false_l",    1.0, |s, e| s.bool_identity_left(e, BinOpKind::Or, false)),
            // R77  a && false  →  false
            rule!("and_false",     1.0, |s, e| s.bool_absorb(e, BinOpKind::And, false)),
            // R78  a || true   →  true
            rule!("or_true",       1.0, |s, e| s.bool_absorb(e, BinOpKind::Or,  true)),
            // R79  a && a  →  a  (idempotent)
            rule!("and_idem",      1.0, |s, e| s.bool_idem(e, BinOpKind::And)),
            // R80  a || a  →  a
            rule!("or_idem",       1.0, |s, e| s.bool_idem(e, BinOpKind::Or)),
            // R81  !!a  →  a
            rule!("double_not",    1.0, |s, e| s.double_not(e)),
            // R82  a && !a  →  false  (contradiction)
            rule!("contradiction", 1.0, |s, e| s.contradiction(e)),

            // ── §6 Comparison normalisation ──────────────────────────────────
            // R83  x < y  ≡  !(x >= y)  (but only simplify when one side is 0/1)
            rule!("cmp_zero_l",    1.0, |s, e| s.cmp_zero(e)),
            // R84  x <= y  →  x < y + 1  (if y is a literal; avoids fenceposts)
            rule!("le_to_lt",      1.0, |s, e| s.le_to_lt(e)),
            // R85  x == x  →  true
            rule!("eq_self",       1.0, |s, e| s.eq_self(e)),
            // R86  x != x  →  false
            rule!("ne_self",       1.0, |s, e| s.ne_self(e)),
            // R87  x < x  →  false
            rule!("lt_self",       1.0, |s, e| s.lt_self(e)),
            // R88  x > x  →  false
            rule!("gt_self",       1.0, |s, e| s.gt_self(e)),
            // R89  x >= x  →  true
            rule!("ge_self",       1.0, |s, e| s.ge_self(e)),
            // R90  x <= x  →  true
            rule!("le_self",       1.0, |s, e| s.le_self(e)),

            // ── §7 Closed-form series ─────────────────────────────────────────
            // R91  1 + 2 + … + n  (BinOp chain)  →  n*(n+1)/2
            rule!("triangular_sum",    1_000.0, |s, e| s.rule_triangular_sum(e)),
            // R92  1² + 2² + … + n²  →  n*(n+1)*(2n+1)/6
            rule!("sum_of_squares",    1_000.0, |s, e| s.rule_sum_of_squares(e)),
            // R93  1³ + 2³ + … + n³  →  (n*(n+1)/2)²
            rule!("sum_of_cubes",      1_000.0, |s, e| s.rule_sum_of_cubes(e)),
            // R94  a + (a+d) + (a+2d) + …  →  n*(2a+(n-1)d)/2
            rule!("arithmetic_series", 1_000.0, |s, e| s.rule_arithmetic_series(e)),
            // R95  1 + r + r² + …  →  (r^n - 1)/(r - 1)
            rule!("geometric_series",  1_000.0, |s, e| s.rule_geometric_series(e)),
            // R96  a₀ + a₁x + a₂x² + …  →  Horner form
            rule!("horner_method",     2.0,     |s, e| s.rule_horner(e)),
            // R97  (n*(n+1)/2) + n + 1  →  (n+1)*(n+2)/2  (extend triangular)
            rule!("extend_triangular", 1.5,     |s, e| s.rule_extend_triangular(e)),
            // R98  (n*(n+1)/2)^2 + (n+1)^3  →  ((n+1)*(n+2)/2)^2  (extend cubes)
            rule!("extend_cubes",      1.5,     |s, e| s.rule_extend_cubes(e)),

            // ── §8 Special-function call rewrites ────────────────────────────
            // R99   fib(n) / fibonacci(n)  →  fib_fast(n)
            rule!("fib_to_fast",       100_000.0, |s, e| s.rule_fib_call(e)),
            // R100  factorial(0)  →  1
            rule!("factorial_zero",    1.0,       |s, e| s.rule_factorial_const(e, 0, 1)),
            // R101  factorial(1)  →  1
            rule!("factorial_one",     1.0,       |s, e| s.rule_factorial_const(e, 1, 1)),
            // R102  factorial(n) where n < 13  →  lookup table constant
            rule!("factorial_small",   1_000.0,   |s, e| s.rule_factorial_small(e)),
            // R103  gcd(x, 0) / gcd(0, x)  →  x
            rule!("gcd_zero",          1.0,       |s, e| s.rule_gcd_zero(e)),
            // R104  gcd(x, x)  →  x
            rule!("gcd_self",          1.0,       |s, e| s.rule_gcd_self(e)),
            // R105  lcm(x, x)  →  x
            rule!("lcm_self",          1.0,       |s, e| s.rule_lcm_self(e)),
            // R106  abs(abs(x))  →  abs(x)
            rule!("abs_idem",          1.0,       |s, e| s.rule_abs_idem(e)),
            // R107  min(x, x)  →  x
            rule!("min_self",          1.0,       |s, e| s.rule_minmax_self(e, "min")),
            // R108  max(x, x)  →  x
            rule!("max_self",          1.0,       |s, e| s.rule_minmax_self(e, "max")),
            // R109  min(min(x,y),z) / max(max(x,y),z)  →  min/max(x,y,z)  (flatten)
            rule!("minmax_flatten",    1.2,       |s, e| s.rule_minmax_flatten(e)),

            // ── §9 Redundant-operation elimination ──────────────────────────
            // R110  (x + y) - y  →  x
            rule!("add_sub_cancel",    1.0, |s, e| s.rule_add_sub_cancel(e)),
            // R111  (x - y) + y  →  x
            rule!("sub_add_cancel",    1.0, |s, e| s.rule_sub_add_cancel(e)),
            // R112  (x * y) / y  →  x  (y ≠ 0)
            rule!("mul_div_cancel",    1.0, |s, e| s.rule_mul_div_cancel(e)),
            // R113  (x / y) * y  →  x  (only exact, flagged as annotation)
            rule!("div_mul_cancel",    1.0, |s, e| s.rule_div_mul_cancel(e)),
            // R114  (x << k) >> k  →  x & ((1<<k)-1)^MAX  (mask, unsigned)
            rule!("shl_shr_cancel",    1.1, |s, e| s.rule_shl_shr_cancel(e)),
            // R115  if true { e1 } else { e2 }  →  e1
            rule!("if_true",           1.0, |s, e| s.rule_if_const(e, true)),
            // R116  if false { e1 } else { e2 }  →  e2
            rule!("if_false",          1.0, |s, e| s.rule_if_const(e, false)),

            // ── §10 Loop-invariant & annotation markers ──────────────────────
            // R117  prefix_sum(arr) pattern detected in expression
            rule!("prefix_sum_expr",   8.0,  |s, e| s.rule_prefix_sum_expr(e)),
            // R118  reduce(f, acc, x) annotation
            rule!("reduce_annot",      4.0,  |s, e| s.rule_reduce_annot(e)),
            // R119  fib(n) add pattern (a+b where names suggest fib)
            rule!("fib_add_pattern",   10.0, |s, e| s.rule_fib_add(e)),
            // R120  pow(x, n) via exponentiation-by-squaring marker
            rule!("pow_exp_squaring",  8.0,  |s, e| s.rule_pow_by_squaring(e)),
        ]
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Top-level driver
    // ─────────────────────────────────────────────────────────────────────────

    /// Optimize a program by scanning for recognizable algorithmic patterns.
    pub fn optimize_program(&mut self, program: &mut Program) {
        for item in &mut program.items {
            if let Item::Fn(fn_decl) = item {
                if let Some(body) = &mut fn_decl.body {
                    self.optimize_block(body);
                }
            }
        }
    }

    fn optimize_block(&mut self, block: &mut Block) {
        for stmt in &mut block.stmts {
            self.optimize_stmt(stmt);
        }
        if let Some(tail) = &mut block.tail {
            let old = std::mem::replace(
                tail.as_mut(),
                Expr::IntLit { span: Span::dummy(), value: 0 },
            );
            **tail = self.optimize_expr(old);
        }
    }

    fn optimize_stmt(&mut self, stmt: &mut Stmt) {
        match stmt {
            Stmt::Let { init: Some(expr), .. } | Stmt::Expr { expr, .. } => {
                let old = std::mem::replace(
                    expr,
                    Expr::IntLit { span: Span::dummy(), value: 0 },
                );
                *expr = self.optimize_expr(old);
            }
            Stmt::ForIn { span, pattern, iter, body, .. } => {
                if let Some(replacement) =
                    self.optimize_for_loop(pattern, iter, body, *span)
                {
                    self.patterns_matched += 1;
                    *stmt = replacement;
                } else {
                    self.optimize_block(body);
                }
            }
            Stmt::While { body, .. } | Stmt::EntityFor { body, .. } => {
                self.optimize_block(body);
            }
            Stmt::If { cond, then, else_, .. } => {
                let old = std::mem::replace(
                    cond,
                    Expr::IntLit { span: Span::dummy(), value: 0 },
                );
                *cond = self.optimize_expr(old);
                self.optimize_block(then);
                if let Some(else_box) = else_ {
                    if let IfOrBlock::Block(b) = &mut **else_box {
                        self.optimize_block(b);
                    }
                }
            }
            Stmt::Return { value: Some(expr), .. } => {
                let old = std::mem::replace(
                    expr,
                    Expr::IntLit { span: Span::dummy(), value: 0 },
                );
                *expr = self.optimize_expr(old);
            }
            Stmt::Loop { body, .. } => {
                self.optimize_block(body);
            }
            _ => {}
        }
    }

    /// Saturating bottom-up expression optimizer.
    /// Re-runs the rule set until no more rules fire (fixpoint).
    fn optimize_expr(&mut self, expr: Expr) -> Expr {
        // Bottom-up: recurse into children first.
        let mut expr = self.recurse_into_children(expr);

        // Then apply rules in a fixpoint loop (max 32 iterations to be safe).
        let mut changed = true;
        let mut iters = 0;
        while changed && iters < 32 {
            changed = false;
            iters += 1;

            // We need to iterate over rules without borrowing self mutably,
            // so we collect (name, speedup, result) before updating counters.
            let mut best: Option<(usize, f64, Expr)> = None;
            for (i, rule) in self.rules.iter().enumerate() {
                if let Some(new_expr) = (rule.apply)(self, &expr) {
                    // Take the first matching rule (rules are in priority order).
                    if best.is_none() {
                        best = Some((i, rule.speedup, new_expr));
                        break; // one rule per fixpoint iteration keeps things simple
                    }
                }
            }
            if let Some((_i, speedup, new_expr)) = best {
                self.patterns_matched += 1;
                self.total_estimated_speedup += speedup;
                expr = new_expr;
                changed = true;
                // Recurse into children of the newly built expression.
                expr = self.recurse_into_children(expr);
            }
        }
        expr
    }

    fn recurse_into_children(&mut self, expr: Expr) -> Expr {
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
            Expr::IfExpr { span, cond, then, else_ } => Expr::IfExpr {
                span,
                cond: Box::new(self.optimize_expr(*cond)),
                then,
                else_,
            },
            Expr::Block(block) => {
                let mut b = *block;
                self.optimize_block(&mut b);
                Expr::Block(Box::new(b))
            }
            Expr::Pow { span, base, exp } => {
                let base = self.optimize_expr(*base);
                let exp  = self.optimize_expr(*exp);
                Expr::Pow { span, base: Box::new(base), exp: Box::new(exp) }
            }
            other => other,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §1 Constant-folding rule implementations
    // ─────────────────────────────────────────────────────────────────────────

    /// Fold a binary operation on two integer literals.
    fn cf_binop(&self, expr: &Expr, op: BinOpKind, f: fn(u128, u128) -> u128) -> Option<Expr> {
        if let Expr::BinOp { op: eop, lhs, rhs, span } = expr {
            if *eop == op {
                if let (Some(a), Some(b)) = (Self::as_int_lit(lhs), Self::as_int_lit(rhs)) {
                    return Some(Expr::IntLit { span: *span, value: f(a, b) });
                }
            }
        }
        None
    }

    /// Fold with a fallible function (e.g., division).
    fn cf_binop_guard(
        &self,
        expr: &Expr,
        op: BinOpKind,
        f: fn(u128, u128) -> Option<u128>,
    ) -> Option<Expr> {
        if let Expr::BinOp { op: eop, lhs, rhs, span } = expr {
            if *eop == op {
                if let (Some(a), Some(b)) = (Self::as_int_lit(lhs), Self::as_int_lit(rhs)) {
                    if let Some(result) = f(a, b) {
                        return Some(Expr::IntLit { span: *span, value: result });
                    }
                }
            }
        }
        None
    }

    /// Fold `x ** n` for small n (n ≤ 16) to avoid astronomically large constants.
    fn cf_pow(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::Pow { base, exp, span } = expr {
            if let (Some(b), Some(e)) = (Self::as_int_lit(base), Self::as_int_lit(exp)) {
                if e <= 16 {
                    let result = b.wrapping_pow(e as u32);
                    return Some(Expr::IntLit { span: *span, value: result });
                }
            }
        }
        None
    }

    /// Fold bounded shift operations.
    fn cf_shift(&self, expr: &Expr, op: BinOpKind) -> Option<Expr> {
        if let Expr::BinOp { op: eop, lhs, rhs, span } = expr {
            if *eop == op {
                if let (Some(a), Some(b)) = (Self::as_int_lit(lhs), Self::as_int_lit(rhs)) {
                    if b < 128 {
                        let result = match op {
                            BinOpKind::Shl => a.wrapping_shl(b as u32),
                            BinOpKind::Shr => a >> b,
                            _ => return None,
                        };
                        return Some(Expr::IntLit { span: *span, value: result });
                    }
                }
            }
        }
        None
    }

    /// Fold negation of an integer literal.
    fn cf_neg(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::UnOp { op: UnOpKind::Neg, expr: inner, span } = expr {
            if let Some(v) = Self::as_int_lit(inner) {
                return Some(Expr::IntLit { span: *span, value: v.wrapping_neg() });
            }
        }
        None
    }

    /// Fold boolean NOT.
    fn cf_not(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::UnOp { op: UnOpKind::Not, expr: inner, span } = expr {
            if let Some(b) = Self::as_bool_lit(inner) {
                return Some(Self::bool_lit(*span, !b));
            }
        }
        None
    }

    /// Fold a comparison of two integer literals.
    fn cf_cmp(&self, expr: &Expr, op: BinOpKind, f: fn(u128, u128) -> bool) -> Option<Expr> {
        if let Expr::BinOp { op: eop, lhs, rhs, span } = expr {
            if *eop == op {
                if let (Some(a), Some(b)) = (Self::as_int_lit(lhs), Self::as_int_lit(rhs)) {
                    return Some(Self::bool_lit(*span, f(a, b)));
                }
            }
        }
        None
    }

    /// Fold boolean && and ||.
    fn cf_bool_binop(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::BinOp { op, lhs, rhs, span } = expr {
            match op {
                BinOpKind::And => {
                    if let (Some(a), Some(b)) = (Self::as_bool_lit(lhs), Self::as_bool_lit(rhs)) {
                        return Some(Self::bool_lit(*span, a && b));
                    }
                }
                BinOpKind::Or => {
                    if let (Some(a), Some(b)) = (Self::as_bool_lit(lhs), Self::as_bool_lit(rhs)) {
                        return Some(Self::bool_lit(*span, a || b));
                    }
                }
                _ => {}
            }
        }
        None
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §2 Identity / annihilator rules
    // ─────────────────────────────────────────────────────────────────────────

    fn identity_right(&self, expr: &Expr, op: BinOpKind, identity: u128) -> Option<Expr> {
        if let Expr::BinOp { op: eop, lhs, rhs, .. } = expr {
            if *eop == op && Self::is_int_lit(rhs, identity) {
                return Some(*lhs.clone());
            }
        }
        None
    }

    fn identity_left(&self, expr: &Expr, op: BinOpKind, identity: u128) -> Option<Expr> {
        if let Expr::BinOp { op: eop, lhs, rhs, .. } = expr {
            if *eop == op && Self::is_int_lit(lhs, identity) {
                return Some(*rhs.clone());
            }
        }
        None
    }

    fn annihilator_right(&self, expr: &Expr, op: BinOpKind, ann: u128) -> Option<Expr> {
        if let Expr::BinOp { op: eop, rhs, span, .. } = expr {
            if *eop == op && Self::is_int_lit(rhs, ann) {
                return Some(Expr::IntLit { span: *span, value: ann });
            }
        }
        None
    }

    fn annihilator_left(&self, expr: &Expr, op: BinOpKind, ann: u128) -> Option<Expr> {
        if let Expr::BinOp { op: eop, lhs, span, .. } = expr {
            if *eop == op && Self::is_int_lit(lhs, ann) {
                return Some(Expr::IntLit { span: *span, value: ann });
            }
        }
        None
    }

    fn zero_div(&self, expr: &Expr) -> Option<Expr> {
        // 0 / x → 0 (x might be non-zero; we can't prove it, but it is safe for integers)
        if let Expr::BinOp { op: BinOpKind::Div, lhs, span, .. } = expr {
            if Self::is_int_lit(lhs, 0) {
                return Some(Expr::IntLit { span: *span, value: 0 });
            }
        }
        None
    }

    fn pow_zero(&self, expr: &Expr) -> Option<Expr> {
        // x ^ 0 → 1
        if let Expr::Pow { exp, span, .. } = expr {
            if Self::is_int_lit(exp, 0) {
                return Some(Expr::IntLit { span: *span, value: 1 });
            }
        }
        None
    }

    fn pow_one(&self, expr: &Expr) -> Option<Expr> {
        // x ^ 1 → x
        if let Expr::Pow { base, exp, .. } = expr {
            if Self::is_int_lit(exp, 1) {
                return Some(*base.clone());
            }
        }
        None
    }

    fn one_pow(&self, expr: &Expr) -> Option<Expr> {
        // 1 ^ x → 1
        if let Expr::Pow { base, span, .. } = expr {
            if Self::is_int_lit(base, 1) {
                return Some(Expr::IntLit { span: *span, value: 1 });
            }
        }
        None
    }

    fn bitand_allones(&self, expr: &Expr) -> Option<Expr> {
        // x & u128::MAX → x
        if let Expr::BinOp { op: BinOpKind::BitAnd, lhs, rhs, .. } = expr {
            if Self::is_int_lit(rhs, u128::MAX) {
                return Some(*lhs.clone());
            }
            if Self::is_int_lit(lhs, u128::MAX) {
                return Some(*rhs.clone());
            }
        }
        None
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §3 Algebraic normalisation
    // ─────────────────────────────────────────────────────────────────────────

    fn sub_self(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::BinOp { op: BinOpKind::Sub, lhs, rhs, span } = expr {
            if Self::exprs_equal(lhs, rhs) {
                return Some(Expr::IntLit { span: *span, value: 0 });
            }
        }
        None
    }

    fn div_self(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::BinOp { op: BinOpKind::Div, lhs, rhs, span } = expr {
            if Self::exprs_equal(lhs, rhs) && !Self::is_int_lit(lhs, 0) {
                return Some(Expr::IntLit { span: *span, value: 1 });
            }
        }
        None
    }

    fn xor_self(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::BinOp { op: BinOpKind::BitXor, lhs, rhs, span } = expr {
            if Self::exprs_equal(lhs, rhs) {
                return Some(Expr::IntLit { span: *span, value: 0 });
            }
        }
        None
    }

    fn idempotent(&self, expr: &Expr, op: BinOpKind) -> Option<Expr> {
        if let Expr::BinOp { op: eop, lhs, rhs, .. } = expr {
            if *eop == op && Self::exprs_equal(lhs, rhs) {
                return Some(*lhs.clone());
            }
        }
        None
    }

    /// (x op y) op z  →  x op (y op z)  (right-associate for canonical form)
    fn reassoc(&self, expr: &Expr, op: BinOpKind) -> Option<Expr> {
        if let Expr::BinOp { op: eop, lhs, rhs, span } = expr {
            if *eop == op {
                if let Expr::BinOp { op: inner_op, lhs: ll, rhs: lr, .. } = lhs.as_ref() {
                    if *inner_op == op {
                        let span = *span;
                        let new_inner = Expr::BinOp {
                            span,
                            op,
                            lhs: lr.clone(),
                            rhs: rhs.clone(),
                        };
                        return Some(Expr::BinOp {
                            span,
                            op,
                            lhs: ll.clone(),
                            rhs: Box::new(new_inner),
                        });
                    }
                }
            }
        }
        None
    }

    fn distribute_mul_add(&self, expr: &Expr) -> Option<Expr> {
        // x * (y + z) → x*y + x*z  — only if rhs is a literal on one side (expose const folding)
        if let Expr::BinOp { op: BinOpKind::Mul, lhs: x, rhs, span } = expr {
            if let Expr::BinOp { op: BinOpKind::Add, lhs: y, rhs: z, .. } = rhs.as_ref() {
                // Only distribute when y or z is a literal to avoid blowup.
                if Self::as_int_lit(y).is_some() || Self::as_int_lit(z).is_some() {
                    let span = *span;
                    return Some(Expr::BinOp {
                        span,
                        op: BinOpKind::Add,
                        lhs: Box::new(Expr::BinOp {
                            span,
                            op: BinOpKind::Mul,
                            lhs: x.clone(),
                            rhs: y.clone(),
                        }),
                        rhs: Box::new(Expr::BinOp {
                            span,
                            op: BinOpKind::Mul,
                            lhs: x.clone(),
                            rhs: z.clone(),
                        }),
                    });
                }
            }
        }
        None
    }

    fn distribute_add_mul(&self, expr: &Expr) -> Option<Expr> {
        // (x + y) * z → x*z + y*z  — only when z is a literal
        if let Expr::BinOp { op: BinOpKind::Mul, lhs, rhs: z, span } = expr {
            if let Expr::BinOp { op: BinOpKind::Add, lhs: x, rhs: y, .. } = lhs.as_ref() {
                if Self::as_int_lit(z).is_some() {
                    let span = *span;
                    return Some(Expr::BinOp {
                        span,
                        op: BinOpKind::Add,
                        lhs: Box::new(Expr::BinOp {
                            span,
                            op: BinOpKind::Mul,
                            lhs: x.clone(),
                            rhs: z.clone(),
                        }),
                        rhs: Box::new(Expr::BinOp {
                            span,
                            op: BinOpKind::Mul,
                            lhs: y.clone(),
                            rhs: z.clone(),
                        }),
                    });
                }
            }
        }
        None
    }

    fn neg_push(&self, expr: &Expr, inner_op: BinOpKind) -> Option<Expr> {
        // -(lhs op rhs)  →  (-lhs) op (-rhs)  for op ∈ {Add, Mul}
        if let Expr::UnOp { op: UnOpKind::Neg, expr: inner, span } = expr {
            if let Expr::BinOp { op, lhs, rhs, .. } = inner.as_ref() {
                if *op == inner_op {
                    let span = *span;
                    let neg_lhs = Expr::UnOp {
                        span,
                        op: UnOpKind::Neg,
                        expr: lhs.clone(),
                    };
                    let neg_rhs = Expr::UnOp {
                        span,
                        op: UnOpKind::Neg,
                        expr: rhs.clone(),
                    };
                    return Some(Expr::BinOp {
                        span,
                        op: inner_op,
                        lhs: Box::new(neg_lhs),
                        rhs: Box::new(neg_rhs),
                    });
                }
            }
        }
        None
    }

    fn sub_to_add_neg(&self, expr: &Expr) -> Option<Expr> {
        // x - y  →  x + (-y)  only when y is a complex expression (not a literal)
        // Avoids infinite loops with cf_sub.
        if let Expr::BinOp { op: BinOpKind::Sub, lhs, rhs, span } = expr {
            if Self::as_int_lit(lhs).is_none() && Self::as_int_lit(rhs).is_none() {
                // Only fire when rhs is a BinOp itself, to avoid trivial cycles.
                if matches!(rhs.as_ref(), Expr::BinOp { .. }) {
                    let span = *span;
                    let neg_rhs = Expr::UnOp { span, op: UnOpKind::Neg, expr: rhs.clone() };
                    return Some(Expr::BinOp {
                        span,
                        op: BinOpKind::Add,
                        lhs: lhs.clone(),
                        rhs: Box::new(neg_rhs),
                    });
                }
            }
        }
        None
    }

    fn double_neg(&self, expr: &Expr) -> Option<Expr> {
        // --x → x
        if let Expr::UnOp { op: UnOpKind::Neg, expr: inner, .. } = expr {
            if let Expr::UnOp { op: UnOpKind::Neg, expr: x, .. } = inner.as_ref() {
                return Some(*x.clone());
            }
        }
        None
    }

    fn add_self(&self, expr: &Expr) -> Option<Expr> {
        // x + x → 2 * x
        if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, span } = expr {
            if Self::exprs_equal(lhs, rhs) {
                let span = *span;
                return Some(Expr::BinOp {
                    span,
                    op: BinOpKind::Mul,
                    lhs: Box::new(Expr::IntLit { span, value: 2 }),
                    rhs: lhs.clone(),
                });
            }
        }
        None
    }

    fn mul2_to_add(&self, expr: &Expr) -> Option<Expr> {
        // x * 2 → x + x  (expose add-self pattern for further simplification)
        // Only fire when x is not already a literal.
        if let Expr::BinOp { op: BinOpKind::Mul, lhs, rhs, span } = expr {
            if Self::is_int_lit(rhs, 2) && Self::as_int_lit(lhs).is_none() {
                let span = *span;
                return Some(Expr::BinOp {
                    span,
                    op: BinOpKind::Add,
                    lhs: lhs.clone(),
                    rhs: lhs.clone(),
                });
            }
        }
        None
    }

    fn factor_common(&self, expr: &Expr, outer_op: BinOpKind, inner_op: BinOpKind) -> Option<Expr> {
        // (x op_inner a) outer_op (x op_inner b)  →  x op_inner (a outer_op b)
        if let Expr::BinOp { op: eop, lhs, rhs, span } = expr {
            if *eop == outer_op {
                if let (
                    Expr::BinOp { op: lop, lhs: ll, rhs: lr, .. },
                    Expr::BinOp { op: rop, lhs: rl, rhs: rr, .. },
                ) = (lhs.as_ref(), rhs.as_ref())
                {
                    if *lop == inner_op && *rop == inner_op && Self::exprs_equal(ll, rl) {
                        let span = *span;
                        return Some(Expr::BinOp {
                            span,
                            op: inner_op,
                            lhs: ll.clone(),
                            rhs: Box::new(Expr::BinOp {
                                span,
                                op: outer_op,
                                lhs: lr.clone(),
                                rhs: rr.clone(),
                            }),
                        });
                    }
                }
            }
        }
        None
    }

    fn cancel_add(&self, expr: &Expr) -> Option<Expr> {
        // (x + a) - (x + b)  →  a - b
        if let Expr::BinOp { op: BinOpKind::Sub, lhs, rhs, span } = expr {
            if let (
                Expr::BinOp { op: BinOpKind::Add, lhs: ll, rhs: lr, .. },
                Expr::BinOp { op: BinOpKind::Add, lhs: rl, rhs: rr, .. },
            ) = (lhs.as_ref(), rhs.as_ref())
            {
                if Self::exprs_equal(ll, rl) {
                    let span = *span;
                    return Some(Expr::BinOp {
                        span,
                        op: BinOpKind::Sub,
                        lhs: lr.clone(),
                        rhs: rr.clone(),
                    });
                }
            }
        }
        None
    }

    fn rem_one(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::BinOp { op: BinOpKind::Rem, rhs, span, .. } = expr {
            if Self::is_int_lit(rhs, 1) {
                return Some(Expr::IntLit { span: *span, value: 0 });
            }
        }
        None
    }

    fn rem_self(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::BinOp { op: BinOpKind::Rem, lhs, rhs, span } = expr {
            if Self::exprs_equal(lhs, rhs) {
                return Some(Expr::IntLit { span: *span, value: 0 });
            }
        }
        None
    }

    fn rem_idem(&self, expr: &Expr) -> Option<Expr> {
        // (x % m) % m → x % m
        if let Expr::BinOp { op: BinOpKind::Rem, lhs, rhs: outer_m, .. } = expr {
            if let Expr::BinOp { op: BinOpKind::Rem, rhs: inner_m, .. } = lhs.as_ref() {
                if Self::exprs_equal(outer_m, inner_m) {
                    return Some(*lhs.clone());
                }
            }
        }
        None
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §4 Bit tricks & strength reduction
    // ─────────────────────────────────────────────────────────────────────────

    fn mul_pow2_to_shl(&self, expr: &Expr) -> Option<Expr> {
        // x * 2^k → x << k
        if let Expr::BinOp { op: BinOpKind::Mul, lhs, rhs, span } = expr {
            for (factor, shift_side, other_side) in [
                (Self::as_int_lit(rhs), rhs, lhs),
                (Self::as_int_lit(lhs), lhs, rhs),
            ] {
                if let Some(v) = factor {
                    if v > 1 && v.is_power_of_two() {
                        let k = v.trailing_zeros() as u128;
                        let span = *span;
                        return Some(Expr::BinOp {
                            span,
                            op: BinOpKind::Shl,
                            lhs: Box::new(*other_side.clone()),
                            rhs: Box::new(Expr::IntLit { span, value: k }),
                        });
                    }
                }
            }
        }
        None
    }

    fn div_pow2_to_shr(&self, expr: &Expr) -> Option<Expr> {
        // x / 2^k → x >> k
        if let Expr::BinOp { op: BinOpKind::Div, lhs, rhs, span } = expr {
            if let Some(v) = Self::as_int_lit(rhs) {
                if v > 1 && v.is_power_of_two() {
                    let k = v.trailing_zeros() as u128;
                    let span = *span;
                    return Some(Expr::BinOp {
                        span,
                        op: BinOpKind::Shr,
                        lhs: lhs.clone(),
                        rhs: Box::new(Expr::IntLit { span, value: k }),
                    });
                }
            }
        }
        None
    }

    fn rem_pow2_to_and(&self, expr: &Expr) -> Option<Expr> {
        // x % 2^k → x & (2^k - 1)
        if let Expr::BinOp { op: BinOpKind::Rem, lhs, rhs, span } = expr {
            if let Some(v) = Self::as_int_lit(rhs) {
                if v > 1 && v.is_power_of_two() {
                    let mask = v - 1;
                    let span = *span;
                    return Some(Expr::BinOp {
                        span,
                        op: BinOpKind::BitAnd,
                        lhs: lhs.clone(),
                        rhs: Box::new(Expr::IntLit { span, value: mask }),
                    });
                }
            }
        }
        None
    }

    /// x * c → (x << log2(nearest_lower_pow2)) + x * remainder
    /// For small constants 3, 5, 9 that equal 2^k + 1.
    fn mul_small_const_strength(&self, expr: &Expr, c: u128) -> Option<Expr> {
        if let Expr::BinOp { op: BinOpKind::Mul, lhs, rhs, span } = expr {
            let (x, factor) = if Self::is_int_lit(rhs, c) {
                (lhs.as_ref(), c)
            } else if Self::is_int_lit(lhs, c) {
                (rhs.as_ref(), c)
            } else {
                return None;
            };
            // c = 2^k + 1  →  x << k + x
            let k_plus1 = (factor - 1).next_power_of_two();
            if factor == k_plus1 + 1 {
                let k = (k_plus1 as f64).log2() as u128;
                let span = *span;
                return Some(Expr::BinOp {
                    span,
                    op: BinOpKind::Add,
                    lhs: Box::new(Expr::BinOp {
                        span,
                        op: BinOpKind::Shl,
                        lhs: Box::new(x.clone()),
                        rhs: Box::new(Expr::IntLit { span, value: k }),
                    }),
                    rhs: Box::new(x.clone()),
                });
            }
        }
        None
    }

    fn mul6_strength(&self, expr: &Expr) -> Option<Expr> {
        // x * 6 → (x << 2) + (x << 1)
        if let Expr::BinOp { op: BinOpKind::Mul, lhs, rhs, span } = expr {
            let x = if Self::is_int_lit(rhs, 6) { lhs.as_ref() }
                    else if Self::is_int_lit(lhs, 6) { rhs.as_ref() }
                    else { return None; };
            let span = *span;
            return Some(Expr::BinOp {
                span,
                op: BinOpKind::Add,
                lhs: Box::new(Expr::BinOp { span, op: BinOpKind::Shl, lhs: Box::new(x.clone()), rhs: Box::new(Expr::IntLit { span, value: 2 }) }),
                rhs: Box::new(Expr::BinOp { span, op: BinOpKind::Shl, lhs: Box::new(x.clone()), rhs: Box::new(Expr::IntLit { span, value: 1 }) }),
            });
        }
        None
    }

    fn mul7_strength(&self, expr: &Expr) -> Option<Expr> {
        // x * 7 → (x << 3) - x
        if let Expr::BinOp { op: BinOpKind::Mul, lhs, rhs, span } = expr {
            let x = if Self::is_int_lit(rhs, 7) { lhs.as_ref() }
                    else if Self::is_int_lit(lhs, 7) { rhs.as_ref() }
                    else { return None; };
            let span = *span;
            return Some(Expr::BinOp {
                span,
                op: BinOpKind::Sub,
                lhs: Box::new(Expr::BinOp { span, op: BinOpKind::Shl, lhs: Box::new(x.clone()), rhs: Box::new(Expr::IntLit { span, value: 3 }) }),
                rhs: Box::new(x.clone()),
            });
        }
        None
    }

    fn mul10_strength(&self, expr: &Expr) -> Option<Expr> {
        // x * 10 → (x << 3) + (x << 1)
        if let Expr::BinOp { op: BinOpKind::Mul, lhs, rhs, span } = expr {
            let x = if Self::is_int_lit(rhs, 10) { lhs.as_ref() }
                    else if Self::is_int_lit(lhs, 10) { rhs.as_ref() }
                    else { return None; };
            let span = *span;
            return Some(Expr::BinOp {
                span,
                op: BinOpKind::Add,
                lhs: Box::new(Expr::BinOp { span, op: BinOpKind::Shl, lhs: Box::new(x.clone()), rhs: Box::new(Expr::IntLit { span, value: 3 }) }),
                rhs: Box::new(Expr::BinOp { span, op: BinOpKind::Shl, lhs: Box::new(x.clone()), rhs: Box::new(Expr::IntLit { span, value: 1 }) }),
            });
        }
        None
    }

    fn shift_zero(&self, expr: &Expr, op: BinOpKind) -> Option<Expr> {
        if let Expr::BinOp { op: eop, lhs, rhs, .. } = expr {
            if *eop == op && Self::is_int_lit(rhs, 0) {
                return Some(*lhs.clone());
            }
        }
        None
    }

    fn merge_shifts(&self, expr: &Expr, op: BinOpKind) -> Option<Expr> {
        // (x << a) << b → x << (a + b)
        if let Expr::BinOp { op: eop, lhs, rhs: b_expr, span } = expr {
            if *eop == op {
                if let Expr::BinOp { op: inner_op, lhs: x, rhs: a_expr, .. } = lhs.as_ref() {
                    if *inner_op == op {
                        if let (Some(a), Some(b)) = (Self::as_int_lit(a_expr), Self::as_int_lit(b_expr)) {
                            let span = *span;
                            return Some(Expr::BinOp {
                                span,
                                op,
                                lhs: x.clone(),
                                rhs: Box::new(Expr::IntLit { span, value: a + b }),
                            });
                        }
                    }
                }
            }
        }
        None
    }

    fn clear_lowest_bit(&self, expr: &Expr) -> Option<Expr> {
        // x & (x - 1) — return an annotation expr marking it as clear-lowest-bit
        // We don't restructure it further but mark it for backend.
        if let Expr::BinOp { op: BinOpKind::BitAnd, lhs: x, rhs, span } = expr {
            if let Expr::BinOp { op: BinOpKind::Sub, lhs: x2, rhs: one, .. } = rhs.as_ref() {
                if Self::exprs_equal(x, x2) && Self::is_int_lit(one, 1) {
                    let span = *span;
                    return Some(Expr::Call {
                        span,
                        func: Box::new(Expr::Ident { span, name: "clear_lowest_bit".to_string() }),
                        args: vec![*x.clone()],
                        named: vec![],
                    });
                }
            }
        }
        None
    }

    fn mask_simplify(&self, expr: &Expr) -> Option<Expr> {
        // (x | y) & ~y → x & ~y   (when ~y appears literally as bitwise NOT)
        if let Expr::BinOp { op: BinOpKind::BitAnd, lhs, rhs: not_y, span } = expr {
            if let Expr::UnOp { op: UnOpKind::Not, expr: y2, .. } = not_y.as_ref() {
                if let Expr::BinOp { op: BinOpKind::BitOr, lhs: x, rhs: y1, .. } = lhs.as_ref() {
                    if Self::exprs_equal(y1, y2) {
                        let span = *span;
                        return Some(Expr::BinOp {
                            span,
                            op: BinOpKind::BitAnd,
                            lhs: x.clone(),
                            rhs: not_y.clone().into(),
                        });
                    }
                }
            }
        }
        None
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §5 Boolean algebra & De Morgan
    // ─────────────────────────────────────────────────────────────────────────

    fn demorgan(&self, expr: &Expr, inner_op: BinOpKind, outer_op: BinOpKind) -> Option<Expr> {
        // !(a inner_op b)  →  (!a) outer_op (!b)
        if let Expr::UnOp { op: UnOpKind::Not, expr: inner, span } = expr {
            if let Expr::BinOp { op, lhs, rhs, .. } = inner.as_ref() {
                if *op == inner_op {
                    let span = *span;
                    return Some(Expr::BinOp {
                        span,
                        op: outer_op,
                        lhs: Box::new(Expr::UnOp { span, op: UnOpKind::Not, expr: lhs.clone() }),
                        rhs: Box::new(Expr::UnOp { span, op: UnOpKind::Not, expr: rhs.clone() }),
                    });
                }
            }
        }
        None
    }

    fn bool_identity_right(&self, expr: &Expr, op: BinOpKind, id: bool) -> Option<Expr> {
        if let Expr::BinOp { op: eop, lhs, rhs, .. } = expr {
            if *eop == op && Self::as_bool_lit(rhs) == Some(id) {
                return Some(*lhs.clone());
            }
        }
        None
    }

    fn bool_identity_left(&self, expr: &Expr, op: BinOpKind, id: bool) -> Option<Expr> {
        if let Expr::BinOp { op: eop, lhs, rhs, .. } = expr {
            if *eop == op && Self::as_bool_lit(lhs) == Some(id) {
                return Some(*rhs.clone());
            }
        }
        None
    }

    fn bool_absorb(&self, expr: &Expr, op: BinOpKind, absorber: bool) -> Option<Expr> {
        if let Expr::BinOp { op: eop, lhs, rhs, span } = expr {
            if *eop == op {
                if Self::as_bool_lit(lhs) == Some(absorber) || Self::as_bool_lit(rhs) == Some(absorber) {
                    return Some(Self::bool_lit(*span, absorber));
                }
            }
        }
        None
    }

    fn bool_idem(&self, expr: &Expr, op: BinOpKind) -> Option<Expr> {
        if let Expr::BinOp { op: eop, lhs, rhs, .. } = expr {
            if *eop == op && Self::exprs_equal(lhs, rhs) {
                return Some(*lhs.clone());
            }
        }
        None
    }

    fn double_not(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::UnOp { op: UnOpKind::Not, expr: inner, .. } = expr {
            if let Expr::UnOp { op: UnOpKind::Not, expr: x, .. } = inner.as_ref() {
                return Some(*x.clone());
            }
        }
        None
    }

    fn contradiction(&self, expr: &Expr) -> Option<Expr> {
        // a && !a → false
        if let Expr::BinOp { op: BinOpKind::And, lhs, rhs, span } = expr {
            if let Expr::UnOp { op: UnOpKind::Not, expr: not_inner, .. } = rhs.as_ref() {
                if Self::exprs_equal(lhs, not_inner) {
                    return Some(Self::bool_lit(*span, false));
                }
            }
            if let Expr::UnOp { op: UnOpKind::Not, expr: not_inner, .. } = lhs.as_ref() {
                if Self::exprs_equal(not_inner, rhs) {
                    return Some(Self::bool_lit(*span, false));
                }
            }
        }
        None
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §6 Comparison normalisation
    // ─────────────────────────────────────────────────────────────────────────

    fn cmp_zero(&self, expr: &Expr) -> Option<Expr> {
        // x < 0 is always false for unsigned; x >= 0 is always true.
        // (We only apply this if the rhs is literally 0.)
        if let Expr::BinOp { op, rhs, span, .. } = expr {
            if Self::is_int_lit(rhs, 0) {
                match op {
                    BinOpKind::Lt => return Some(Self::bool_lit(*span, false)), // x < 0 always false unsigned
                    BinOpKind::Ge => return Some(Self::bool_lit(*span, true)),  // x >= 0 always true unsigned
                    _ => {}
                }
            }
        }
        None
    }

    fn le_to_lt(&self, expr: &Expr) -> Option<Expr> {
        // x <= lit  →  x < lit + 1
        if let Expr::BinOp { op: BinOpKind::Le, lhs, rhs, span } = expr {
            if let Some(v) = Self::as_int_lit(rhs) {
                if let Some(v1) = v.checked_add(1) {
                    let span = *span;
                    return Some(Expr::BinOp {
                        span,
                        op: BinOpKind::Lt,
                        lhs: lhs.clone(),
                        rhs: Box::new(Expr::IntLit { span, value: v1 }),
                    });
                }
            }
        }
        None
    }

    fn eq_self(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::BinOp { op: BinOpKind::Eq, lhs, rhs, span } = expr {
            if Self::exprs_equal(lhs, rhs) { return Some(Self::bool_lit(*span, true)); }
        }
        None
    }

    fn ne_self(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::BinOp { op: BinOpKind::Ne, lhs, rhs, span } = expr {
            if Self::exprs_equal(lhs, rhs) { return Some(Self::bool_lit(*span, false)); }
        }
        None
    }

    fn lt_self(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::BinOp { op: BinOpKind::Lt, lhs, rhs, span } = expr {
            if Self::exprs_equal(lhs, rhs) { return Some(Self::bool_lit(*span, false)); }
        }
        None
    }

    fn gt_self(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::BinOp { op: BinOpKind::Gt, lhs, rhs, span } = expr {
            if Self::exprs_equal(lhs, rhs) { return Some(Self::bool_lit(*span, false)); }
        }
        None
    }

    fn ge_self(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::BinOp { op: BinOpKind::Ge, lhs, rhs, span } = expr {
            if Self::exprs_equal(lhs, rhs) { return Some(Self::bool_lit(*span, true)); }
        }
        None
    }

    fn le_self(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::BinOp { op: BinOpKind::Le, lhs, rhs, span } = expr {
            if Self::exprs_equal(lhs, rhs) { return Some(Self::bool_lit(*span, true)); }
        }
        None
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §7 Closed-form series rules (delegates to the existing helpers)
    // ─────────────────────────────────────────────────────────────────────────

    fn rule_triangular_sum(&self, expr: &Expr) -> Option<Expr> {
        self.try_triangular_sum(expr).map(|r| r.optimized)
    }

    fn rule_sum_of_squares(&self, expr: &Expr) -> Option<Expr> {
        self.try_sum_of_squares(expr).map(|r| r.optimized)
    }

    fn rule_sum_of_cubes(&self, expr: &Expr) -> Option<Expr> {
        self.try_sum_of_cubes(expr).map(|r| r.optimized)
    }

    fn rule_arithmetic_series(&self, expr: &Expr) -> Option<Expr> {
        self.try_arithmetic_series(expr).map(|r| r.optimized)
    }

    fn rule_geometric_series(&self, expr: &Expr) -> Option<Expr> {
        self.try_geometric_series(expr).map(|r| r.optimized)
    }

    fn rule_horner(&self, expr: &Expr) -> Option<Expr> {
        self.try_horner_pattern(expr).map(|r| r.optimized)
    }

    /// Recognize (n*(n+1)/2) + (n+1)  →  (n+1)*(n+2)/2
    fn rule_extend_triangular(&self, expr: &Expr) -> Option<Expr> {
        // LHS must be the triangular formula for some n; RHS must be n+1.
        if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, span } = expr {
            if let Some(n) = Self::extract_triangular_n(lhs) {
                // Check rhs == n + 1
                if let Expr::BinOp { op: BinOpKind::Add, lhs: rl, rhs: rr, .. } = rhs.as_ref() {
                    if Self::exprs_equal(rl, &n) && Self::is_int_lit(rr, 1) {
                        let span = *span;
                        let n1 = Expr::BinOp {
                            span,
                            op: BinOpKind::Add,
                            lhs: Box::new(n.clone()),
                            rhs: Box::new(Expr::IntLit { span, value: 1 }),
                        };
                        return Some(Self::build_triangular_formula(span, &n1));
                    }
                }
            }
        }
        None
    }

    /// Recognize sum-of-cubes formula extended by (n+1)^3.
    fn rule_extend_cubes(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, span } = expr {
            if let Some(n) = Self::extract_sum_of_cubes_n(lhs) {
                // rhs should be (n+1)^3
                if Self::is_cube_of_n_plus_one(rhs, &n) {
                    let span = *span;
                    let n1 = Expr::BinOp {
                        span,
                        op: BinOpKind::Add,
                        lhs: Box::new(n.clone()),
                        rhs: Box::new(Expr::IntLit { span, value: 1 }),
                    };
                    return Some(Self::build_sum_of_cubes_formula(span, &n1));
                }
            }
        }
        None
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §8 Special-function call rewrites
    // ─────────────────────────────────────────────────────────────────────────

    fn rule_fib_call(&self, expr: &Expr) -> Option<Expr> {
        self.try_matrix_power_pattern(expr).map(|r| r.optimized)
    }

    fn rule_factorial_const(&self, expr: &Expr, arg_val: u128, result: u128) -> Option<Expr> {
        if let Expr::Call { func, args, span, .. } = expr {
            if let Expr::Ident { name, .. } = func.as_ref() {
                if name == "factorial" && args.len() == 1 {
                    if Self::is_int_lit(&args[0], arg_val) {
                        return Some(Expr::IntLit { span: *span, value: result });
                    }
                }
            }
        }
        None
    }

    /// factorial(n) → const for n ∈ 0..=12
    fn rule_factorial_small(&self, expr: &Expr) -> Option<Expr> {
        const FACT: [u128; 13] = [
            1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880,
            3628800, 39916800, 479001600,
        ];
        if let Expr::Call { func, args, span, .. } = expr {
            if let Expr::Ident { name, .. } = func.as_ref() {
                if name == "factorial" && args.len() == 1 {
                    if let Some(n) = Self::as_int_lit(&args[0]) {
                        if (n as usize) < FACT.len() {
                            return Some(Expr::IntLit { span: *span, value: FACT[n as usize] });
                        }
                    }
                }
            }
        }
        None
    }

    fn rule_gcd_zero(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::Call { func, args, .. } = expr {
            if let Expr::Ident { name, .. } = func.as_ref() {
                if (name == "gcd" || name == "gcd_u64") && args.len() == 2 {
                    if Self::is_int_lit(&args[0], 0) { return Some(args[1].clone()); }
                    if Self::is_int_lit(&args[1], 0) { return Some(args[0].clone()); }
                }
            }
        }
        None
    }

    fn rule_gcd_self(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::Call { func, args, .. } = expr {
            if let Expr::Ident { name, .. } = func.as_ref() {
                if (name == "gcd" || name == "gcd_u64") && args.len() == 2 {
                    if Self::exprs_equal(&args[0], &args[1]) {
                        return Some(args[0].clone());
                    }
                }
            }
        }
        None
    }

    fn rule_lcm_self(&self, expr: &Expr) -> Option<Expr> {
        if let Expr::Call { func, args, .. } = expr {
            if let Expr::Ident { name, .. } = func.as_ref() {
                if name == "lcm" && args.len() == 2 {
                    if Self::exprs_equal(&args[0], &args[1]) {
                        return Some(args[0].clone());
                    }
                }
            }
        }
        None
    }

    fn rule_abs_idem(&self, expr: &Expr) -> Option<Expr> {
        // abs(abs(x)) → abs(x)
        if let Expr::Call { func, args, .. } = expr {
            if let Expr::Ident { name, .. } = func.as_ref() {
                if name == "abs" && args.len() == 1 {
                    if let Expr::Call { func: inner_func, args: inner_args, .. } = &args[0] {
                        if let Expr::Ident { name: iname, .. } = inner_func.as_ref() {
                            if iname == "abs" && inner_args.len() == 1 {
                                return Some(args[0].clone());
                            }
                        }
                    }
                }
            }
        }
        None
    }

    fn rule_minmax_self(&self, expr: &Expr, fn_name: &str) -> Option<Expr> {
        if let Expr::Call { func, args, .. } = expr {
            if let Expr::Ident { name, .. } = func.as_ref() {
                if name == fn_name && args.len() == 2 && Self::exprs_equal(&args[0], &args[1]) {
                    return Some(args[0].clone());
                }
            }
        }
        None
    }

    fn rule_minmax_flatten(&self, expr: &Expr) -> Option<Expr> {
        // min(min(x,y), z) → min(x, y, z)  (we call a variadic min3/max3 intrinsic)
        if let Expr::Call { func, args, span, .. } = expr {
            if let Expr::Ident { name, .. } = func.as_ref() {
                if (name == "min" || name == "max") && args.len() == 2 {
                    if let Expr::Call { func: inner_f, args: inner_args, .. } = &args[0] {
                        if let Expr::Ident { name: iname, .. } = inner_f.as_ref() {
                            if iname == name.as_str() && inner_args.len() == 2 {
                                let span = *span;
                                let fn3 = format!("{}3", name);
                                return Some(Expr::Call {
                                    span,
                                    func: Box::new(Expr::Ident { span, name: fn3 }),
                                    args: vec![
                                        inner_args[0].clone(),
                                        inner_args[1].clone(),
                                        args[1].clone(),
                                    ],
                                    named: vec![],
                                });
                            }
                        }
                    }
                }
            }
        }
        None
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §9 Redundant-operation elimination
    // ─────────────────────────────────────────────────────────────────────────

    fn rule_add_sub_cancel(&self, expr: &Expr) -> Option<Expr> {
        // (x + y) - y → x
        if let Expr::BinOp { op: BinOpKind::Sub, lhs, rhs: y_outer, .. } = expr {
            if let Expr::BinOp { op: BinOpKind::Add, lhs: x, rhs: y_inner, .. } = lhs.as_ref() {
                if Self::exprs_equal(y_inner, y_outer) { return Some(*x.clone()); }
            }
            if let Expr::BinOp { op: BinOpKind::Add, lhs: y_inner, rhs: x, .. } = lhs.as_ref() {
                if Self::exprs_equal(y_inner, y_outer) { return Some(*x.clone()); }
            }
        }
        None
    }

    fn rule_sub_add_cancel(&self, expr: &Expr) -> Option<Expr> {
        // (x - y) + y → x
        if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs: y_outer, .. } = expr {
            if let Expr::BinOp { op: BinOpKind::Sub, lhs: x, rhs: y_inner, .. } = lhs.as_ref() {
                if Self::exprs_equal(y_inner, y_outer) { return Some(*x.clone()); }
            }
        }
        None
    }

    fn rule_mul_div_cancel(&self, expr: &Expr) -> Option<Expr> {
        // (x * y) / y → x  (y ≠ 0, and y is a non-zero literal)
        if let Expr::BinOp { op: BinOpKind::Div, lhs, rhs: y_outer, .. } = expr {
            if let Expr::BinOp { op: BinOpKind::Mul, lhs: x, rhs: y_inner, .. } = lhs.as_ref() {
                if Self::exprs_equal(y_inner, y_outer) && !Self::is_int_lit(y_outer, 0) {
                    return Some(*x.clone());
                }
                if Self::exprs_equal(x, y_outer) && !Self::is_int_lit(y_outer, 0) {
                    return Some(*y_inner.clone());
                }
            }
        }
        None
    }

    fn rule_div_mul_cancel(&self, expr: &Expr) -> Option<Expr> {
        // (x / y) * y → x  (annotated as "exact division assumed")
        if let Expr::BinOp { op: BinOpKind::Mul, lhs, rhs: y_outer, span } = expr {
            if let Expr::BinOp { op: BinOpKind::Div, lhs: x, rhs: y_inner, .. } = lhs.as_ref() {
                if Self::exprs_equal(y_inner, y_outer) && !Self::is_int_lit(y_outer, 0) {
                    let span = *span;
                    // Wrap in an annotation call so the backend knows it assumed exactness.
                    return Some(Expr::Call {
                        span,
                        func: Box::new(Expr::Ident { span, name: "exact_div_restore".to_string() }),
                        args: vec![*x.clone()],
                        named: vec![],
                    });
                }
            }
        }
        None
    }

    fn rule_shl_shr_cancel(&self, expr: &Expr) -> Option<Expr> {
        // (x << k) >> k → x & ((1 << k) - 1)^complement  (mask high bits)
        // Simplified: just return x (valid for unsigned when no overflow).
        if let Expr::BinOp { op: BinOpKind::Shr, lhs, rhs: k_outer, .. } = expr {
            if let Expr::BinOp { op: BinOpKind::Shl, lhs: x, rhs: k_inner, .. } = lhs.as_ref() {
                if Self::exprs_equal(k_inner, k_outer) {
                    // Valid only if the shift is small enough that no bits were lost.
                    // We can only be sure if k is 0; otherwise annotate.
                    if Self::is_int_lit(k_outer, 0) {
                        return Some(*x.clone());
                    }
                }
            }
        }
        None
    }

    fn rule_if_const(&self, expr: &Expr, cond_val: bool) -> Option<Expr> {
        if let Expr::IfExpr { cond, then, else_, .. } = expr {
            if Self::as_bool_lit(cond) == Some(cond_val) {
                if cond_val {
                    // if true { e1 } else { e2 } → e1
                    if let Some(tail) = &then.tail {
                        return Some(*tail.clone());
                    }
                } else {
                    // if false { e1 } else { e2 } → e2
                    if let Some(else_block) = else_ {
                        // else_block is already &Block (the Stmt::If else_ field)
                        if let Some(tail) = &else_block.tail {
                            return Some(*tail.clone());
                        }
                    }
                }
            }
        }
        None
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §10 Annotation markers
    // ─────────────────────────────────────────────────────────────────────────

    fn rule_prefix_sum_expr(&self, expr: &Expr) -> Option<Expr> {
        self.try_prefix_sum_pattern(expr).map(|r| r.optimized)
    }

    fn rule_reduce_annot(&self, expr: &Expr) -> Option<Expr> {
        self.try_reduce_pattern(expr).map(|r| r.optimized)
    }

    fn rule_fib_add(&self, expr: &Expr) -> Option<Expr> {
        self.try_matrix_power_pattern(expr).map(|r| r.optimized)
    }

    fn rule_pow_by_squaring(&self, expr: &Expr) -> Option<Expr> {
        // Mark large integer powers for exponentiation-by-squaring in codegen.
        // x ** n where n ≥ 4 → pow_fast(x, n)
        if let Expr::Pow { base, exp, span } = expr {
            if let Some(n) = Self::as_int_lit(exp) {
                if n >= 4 {
                    let span = *span;
                    return Some(Expr::Call {
                        span,
                        func: Box::new(Expr::Ident { span, name: "pow_fast".to_string() }),
                        args: vec![*base.clone(), Expr::IntLit { span, value: n }],
                        named: vec![],
                    });
                }
            }
        }
        None
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Semantic-lift dispatcher (kept for backward compat; now calls rule table)
    // ─────────────────────────────────────────────────────────────────────────

    fn try_semantic_lift(&self, expr: &Expr) -> Option<SemanticOptResult> {
        for rule in &self.rules {
            if let Some(optimized) = (rule.apply)(self, expr) {
                return Some(SemanticOptResult {
                    original: expr.clone(),
                    optimized,
                    pattern_name: rule.name.to_string(),
                    estimated_speedup: rule.speedup,
                });
            }
        }
        None
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Expression-level pattern helpers (public for tests)
    // ─────────────────────────────────────────────────────────────────────────

    /// Pattern: Triangular number sum  1+2+…+n  →  n*(n+1)/2
    pub fn try_triangular_sum(&self, expr: &Expr) -> Option<SemanticOptResult> {
        if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, span } = expr {
            if let Some(n) = Self::extract_ident(rhs) {
                if Self::is_sequential_sum_to_n(lhs, &n).is_some() {
                    let span = *span;
                    let optimized = Self::build_triangular_formula(
                        span,
                        &Expr::Ident { span, name: n.clone() },
                    );
                    return Some(SemanticOptResult {
                        original: expr.clone(),
                        optimized,
                        pattern_name: "triangular_sum".to_string(),
                        estimated_speedup: 1000.0,
                    });
                }
            }
        }
        None
    }

    /// Pattern: Arithmetic series sum  a+(a+d)+…  →  n*(2a+(n-1)d)/2
    pub fn try_arithmetic_series(&self, expr: &Expr) -> Option<SemanticOptResult> {
        let terms = Self::collect_add_terms(expr);
        if terms.len() < 3 { return None; }

        let first  = terms[0];
        let second = terms[1];
        let (base, step) = Self::extract_arithmetic_base_step(first, second)?;

        for (k, term) in terms.iter().enumerate().skip(2) {
            if !Self::matches_arithmetic_term(term, &base, &step, k as u128) {
                return None;
            }
        }

        let n = terms.len() as u128;
        let span = expr.span();
        let optimized = Self::build_arithmetic_series_formula(span, &base, &step, n);

        Some(SemanticOptResult {
            original: expr.clone(),
            optimized,
            pattern_name: "arithmetic_series".to_string(),
            estimated_speedup: 1000.0 * terms.len() as f64,
        })
    }

    /// Pattern: Power computation   x*x*… → pow marker
    pub fn try_power_pattern(&self, expr: &Expr) -> Option<SemanticOptResult> {
        if let Expr::BinOp { op: BinOpKind::Mul, lhs, rhs, span } = expr {
            if let (Some(l), Some(r)) = (Self::extract_ident(lhs), Self::extract_ident(rhs)) {
                if l == r {
                    let span = *span;
                    let optimized = Expr::BinOp {
                        span,
                        op: BinOpKind::Mul,
                        lhs: Box::new(Expr::Ident { span, name: l.clone() }),
                        rhs: Box::new(Expr::Ident { span, name: l }),
                    };
                    return Some(SemanticOptResult {
                        original: expr.clone(),
                        optimized,
                        pattern_name: "power_squaring".to_string(),
                        estimated_speedup: 2.0,
                    });
                }
            }
        }
        None
    }

    /// Pattern: Polynomial → Horner's method
    pub fn try_horner_pattern(&self, expr: &Expr) -> Option<SemanticOptResult> {
        let terms = Self::collect_polynomial_terms(expr)?;
        if terms.len() < 3 { return None; }

        let span = expr.span();
        let x_name = terms.first()?.1.clone()?;

        let mut horner = Expr::IntLit { span, value: terms.last()?.0 };
        for (coeff, _var) in terms.iter().rev().skip(1) {
            horner = Expr::BinOp {
                span,
                op: BinOpKind::Add,
                lhs: Box::new(Expr::BinOp {
                    span,
                    op: BinOpKind::Mul,
                    lhs: Box::new(horner),
                    rhs: Box::new(Expr::Ident { span, name: x_name.clone() }),
                }),
                rhs: Box::new(Expr::IntLit { span, value: *coeff }),
            };
        }

        Some(SemanticOptResult {
            original: expr.clone(),
            optimized: horner,
            pattern_name: "horner_method".to_string(),
            estimated_speedup: (terms.len() as f64).sqrt(),
        })
    }

    /// Pattern: Matrix power / fib fast
    pub fn try_matrix_power_pattern(&self, expr: &Expr) -> Option<SemanticOptResult> {
        if let Expr::Call { span, func, args, named } = expr {
            if named.is_empty() && args.len() == 1 {
                if let Expr::Ident { name, .. } = func.as_ref() {
                    if matches!(name.as_str(), "fib" | "fibonacci") {
                        let span = *span;
                        let optimized = Expr::Call {
                            span,
                            func: Box::new(Expr::Ident { span, name: "fib_fast".to_string() }),
                            args: args.clone(),
                            named: vec![],
                        };
                        return Some(SemanticOptResult {
                            original: expr.clone(),
                            optimized,
                            pattern_name: "matrix_power_fibonacci".to_string(),
                            estimated_speedup: 100_000.0,
                        });
                    }
                }
            }
        }

        // Also catch `a + b` where names hint at a Fibonacci pair.
        if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, span } = expr {
            if let (Some(a), Some(b)) = (Self::extract_ident(lhs), Self::extract_ident(rhs)) {
                if (a == "a" && b == "b") || (a == "fib_n" && b == "fib_n_1") {
                    // Return the expression as-is (the real win comes at loop level).
                    let span = *span;
                    let optimized = Expr::BinOp {
                        span,
                        op: BinOpKind::Add,
                        lhs: Box::new(Expr::Ident { span, name: a }),
                        rhs: Box::new(Expr::Ident { span, name: b }),
                    };
                    return Some(SemanticOptResult {
                        original: expr.clone(),
                        optimized,
                        pattern_name: "fibonacci_accumulator".to_string(),
                        estimated_speedup: 10.0,
                    });
                }
            }
        }
        None
    }

    /// Pattern: Geometric series  1 + r + r² + …  →  (r^n - 1)/(r - 1)
    pub fn try_geometric_series(&self, expr: &Expr) -> Option<SemanticOptResult> {
        let terms = Self::collect_add_terms(expr);
        if terms.len() < 3 { return None; }
        if !Self::is_int_lit(terms[0], 1) { return None; }
        let r = Self::extract_ident(terms[1])?;
        for (k, term) in terms.iter().enumerate().skip(2) {
            if !Self::is_power_of_ident(term, &r, k as u32) { return None; }
        }

        let n   = terms.len() as u128;
        let span = expr.span();
        let optimized = Expr::BinOp {
            span,
            op: BinOpKind::Div,
            lhs: Box::new(Expr::BinOp {
                span,
                op: BinOpKind::Sub,
                lhs: Box::new(Expr::Pow {
                    span,
                    base: Box::new(Expr::Ident { span, name: r.clone() }),
                    exp:  Box::new(Expr::IntLit { span, value: n }),
                }),
                rhs: Box::new(Expr::IntLit { span, value: 1 }),
            }),
            rhs: Box::new(Expr::BinOp {
                span,
                op: BinOpKind::Sub,
                lhs: Box::new(Expr::Ident { span, name: r }),
                rhs: Box::new(Expr::IntLit { span, value: 1 }),
            }),
        };

        Some(SemanticOptResult {
            original: expr.clone(),
            optimized,
            pattern_name: "geometric_series".to_string(),
            estimated_speedup: 1000.0 * terms.len() as f64,
        })
    }

    /// Pattern: Sum of squares  1+4+9+…  →  n*(n+1)*(2n+1)/6
    pub fn try_sum_of_squares(&self, expr: &Expr) -> Option<SemanticOptResult> {
        let terms = Self::collect_add_terms(expr);
        if terms.len() < 3 { return None; }
        for (i, term) in terms.iter().enumerate() {
            let k = (i + 1) as u128;
            if !Self::is_int_lit(term, k * k) { return None; }
        }
        let n   = terms.len() as u128;
        let span = expr.span();
        let result = n * (n + 1) * (2 * n + 1) / 6;
        Some(SemanticOptResult {
            original: expr.clone(),
            optimized: Expr::IntLit { span, value: result },
            pattern_name: "sum_of_squares".to_string(),
            estimated_speedup: 1000.0,
        })
    }

    /// Pattern: Sum of cubes  1+8+27+…  →  (n*(n+1)/2)²
    pub fn try_sum_of_cubes(&self, expr: &Expr) -> Option<SemanticOptResult> {
        let terms = Self::collect_add_terms(expr);
        if terms.len() < 3 { return None; }
        for (i, term) in terms.iter().enumerate() {
            let k = (i + 1) as u128;
            if !Self::is_int_lit(term, k * k * k) { return None; }
        }
        let n         = terms.len() as u128;
        let span       = expr.span();
        let triangular = n * (n + 1) / 2;
        let result     = triangular * triangular;
        Some(SemanticOptResult {
            original: expr.clone(),
            optimized: Expr::IntLit { span, value: result },
            pattern_name: "sum_of_cubes".to_string(),
            estimated_speedup: 1000.0,
        })
    }

    /// Pattern: Prefix sum expression  arr[i] + arr[i-1]
    pub fn try_prefix_sum_pattern(&self, expr: &Expr) -> Option<SemanticOptResult> {
        if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, span } = expr {
            if Self::is_adjacent_index_access(lhs, rhs) {
                let span = *span;
                let optimized = Expr::Call {
                    span,
                    func: Box::new(Expr::Ident { span, name: "prefix_sum_simd".to_string() }),
                    args: vec![*lhs.clone(), *rhs.clone()],
                    named: vec![],
                };
                return Some(SemanticOptResult {
                    original: expr.clone(),
                    optimized,
                    pattern_name: "prefix_sum_annotation".to_string(),
                    estimated_speedup: 8.0,
                });
            }
        }
        None
    }

    /// Pattern: Reduce annotation  reduce(acc, x) / fold(acc, x)
    pub fn try_reduce_pattern(&self, expr: &Expr) -> Option<SemanticOptResult> {
        if let Expr::Call { span, func, args, named } = expr {
            if named.is_empty() && args.len() == 2 {
                if let Expr::Ident { name, .. } = func.as_ref() {
                    if matches!(name.as_str(), "reduce" | "fold") {
                        let span = *span;
                        let optimized = Expr::Call {
                            span,
                            func: Box::new(Expr::Ident { span, name: "parallel_reduce".to_string() }),
                            args: args.clone(),
                            named: vec![],
                        };
                        return Some(SemanticOptResult {
                            original: expr.clone(),
                            optimized,
                            pattern_name: "parallel_reduce_annotation".to_string(),
                            estimated_speedup: 4.0,
                        });
                    }
                }
            }
        }
        None
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Loop-level pattern recognition  (Tier 2)
    // ─────────────────────────────────────────────────────────────────────────

    fn optimize_for_loop(
        &mut self,
        pattern: &Pattern,
        iter: &Expr,
        body: &Block,
        loop_span: Span,
    ) -> Option<Stmt> {
        let loop_var          = Self::extract_pattern_name(pattern)?;
        let (lo_val, hi_expr) = Self::extract_range_info(iter)?;
        let n_expr            = hi_expr.clone();

        // Pattern A: Single accumulation statement
        if body.stmts.len() == 1 && body.tail.is_none() {
            if let Some(rep) = self.analyze_single_accumulation(
                &loop_var, &n_expr, lo_val, &body.stmts[0], loop_span,
            ) {
                self.total_estimated_speedup += 1000.0;
                return Some(rep);
            }
        }

        // Pattern B: Fibonacci swap
        if let Some(rep) = self.analyze_fibonacci_loop(&loop_var, &n_expr, body, loop_span) {
            self.total_estimated_speedup += 100_000.0;
            return Some(rep);
        }

        // Pattern C: Prefix sum
        if let Some(rep) = self.analyze_prefix_sum_loop(&loop_var, &n_expr, body, loop_span) {
            self.total_estimated_speedup += 8.0;
            return Some(rep);
        }

        // Pattern D: Generic reduce
        if let Some(rep) = self.analyze_reduce_loop(&loop_var, &n_expr, body, loop_span) {
            self.total_estimated_speedup += 4.0;
            return Some(rep);
        }

        None
    }

    fn analyze_single_accumulation(
        &self,
        loop_var: &str,
        n_expr: &Expr,
        lo_val: u128,
        stmt: &Stmt,
        span: Span,
    ) -> Option<Stmt> {
        let (acc_name, value_expr, assign_op) = Self::extract_accumulation(stmt)?;

        match assign_op {
            AssignOpKind::AddAssign => {
                self.classify_additive_accumulation(loop_var, n_expr, lo_val, &acc_name, value_expr, span)
            }
            AssignOpKind::Assign => {
                // acc = acc + expr
                if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, .. } = value_expr {
                    if Self::expr_is_ident(lhs, &acc_name) {
                        return self.classify_additive_accumulation(
                            loop_var, n_expr, lo_val, &acc_name, rhs, span,
                        );
                    }
                }
                // acc = acc * r + c  (Horner / geometric)
                if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs: c_rhs, .. } = value_expr {
                    if let Expr::BinOp { op: BinOpKind::Mul, lhs: ml, rhs: mr, .. } = lhs.as_ref() {
                        let (r_expr, ok) = if Self::expr_is_ident(ml, &acc_name) {
                            (mr.as_ref(), true)
                        } else if Self::expr_is_ident(mr, &acc_name) {
                            (ml.as_ref(), true)
                        } else {
                            (ml.as_ref(), false)
                        };
                        if ok {
                            return Some(self.build_geometric_horner_replacement(
                                span, &acc_name, r_expr, c_rhs.as_ref(), n_expr,
                            ));
                        }
                    }
                }
                None
            }
            AssignOpKind::MulAssign => {
                // acc *= r → acc = r^n
                let r_expr = value_expr;
                let optimized = Expr::Assign {
                    span,
                    op: AssignOpKind::Assign,
                    target: Box::new(Expr::Ident { span, name: acc_name }),
                    value: Box::new(Expr::Pow {
                        span,
                        base: Box::new(r_expr.clone()),
                        exp:  Box::new(n_expr.clone()),
                    }),
                };
                Some(Stmt::Expr { span, expr: optimized, has_semi: true })
            }
            _ => None,
        }
    }

    fn classify_additive_accumulation(
        &self,
        loop_var: &str,
        n_expr: &Expr,
        lo_val: u128,
        acc_name: &str,
        added_expr: &Expr,
        span: Span,
    ) -> Option<Stmt> {
        let assign = |value: Expr| Stmt::Expr {
            span,
            expr: Expr::Assign {
                span,
                op: AssignOpKind::Assign,
                target: Box::new(Expr::Ident { span, name: acc_name.to_string() }),
                value: Box::new(value),
            },
            has_semi: true,
        };

        // sum += i  → triangular
        if Self::expr_is_ident(added_expr, loop_var) {
            let n_adjusted = if lo_val == 0 {
                n_expr.clone()
            } else {
                Expr::BinOp {
                    span,
                    op: BinOpKind::Sub,
                    lhs: Box::new(n_expr.clone()),
                    rhs: Box::new(Expr::IntLit { span, value: 1 }),
                }
            };
            return Some(assign(Self::build_triangular_formula(span, &n_adjusted)));
        }

        // sum += i*i  → sum of squares
        if Self::is_mul_of_var(added_expr, loop_var, 2) {
            return Some(assign(Self::build_sum_of_squares_formula(span, n_expr)));
        }

        // sum += i*i*i  → sum of cubes
        if Self::is_mul_of_var(added_expr, loop_var, 3) {
            return Some(assign(Self::build_sum_of_cubes_formula(span, n_expr)));
        }

        // sum += a + i * d  → arithmetic series
        if let Some((a_expr, d_expr)) = Self::extract_arithmetic_term(added_expr, loop_var) {
            return Some(assign(Self::build_arithmetic_series_loop_formula(span, n_expr, &a_expr, &d_expr)));
        }

        // sum += r ** i  → geometric series
        if let Expr::Pow { base, exp, .. } = added_expr {
            if Self::expr_is_ident(exp, loop_var) {
                return Some(assign(Self::build_geometric_series_formula(span, base, n_expr)));
            }
        }

        None
    }

    fn analyze_fibonacci_loop(
        &self,
        _loop_var: &str,
        n_expr: &Expr,
        body: &Block,
        span: Span,
    ) -> Option<Stmt> {
        if body.tail.is_some() { return None; }

        // 3-statement pattern:  temp = a+b; a = b; b = temp
        if body.stmts.len() == 3 {
            if let (Some((temp, first_val)), Some((a2, b_val)), Some((b3, temp_val))) = (
                Self::extract_plain_assignment(&body.stmts[0]),
                Self::extract_plain_assignment(&body.stmts[1]),
                Self::extract_plain_assignment(&body.stmts[2]),
            ) {
                if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, .. } = first_val {
                    let a_name = Self::extract_ident(lhs)?;
                    let b_name = Self::extract_ident(rhs)?;
                    if Self::expr_is_ident(b_val, &b_name)
                        && a2 == a_name
                        && Self::expr_is_ident(temp_val, temp)
                        && b3 == b_name
                    {
                        return Some(self.build_fibonacci_replacement(span, &a_name, &b_name, n_expr));
                    }
                }
            }
        }

        // 2-statement pattern:  tuple destructure  (a, b) = (b, a+b)
        if body.stmts.len() == 2 || body.stmts.len() == 1 {
            if let Stmt::Expr { expr, .. } = &body.stmts[0] {
                if let Expr::Assign {
                    op: AssignOpKind::Assign,
                    target,
                    value,
                    ..
                } = expr
                {
                    if let (
                        Expr::Tuple { elems: targets, .. },
                        Expr::Tuple { elems: values, .. },
                    ) = (target.as_ref(), value.as_ref())
                    {
                        if targets.len() == 2 && values.len() == 2 {
                            let a = Self::extract_ident_from_expr(&targets[0])?;
                            let b = Self::extract_ident_from_expr(&targets[1])?;
                            if Self::expr_is_ident(&values[0], &b) {
                                if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, .. } = &values[1] {
                                    if Self::expr_is_ident(lhs, &a) && Self::expr_is_ident(rhs, &b) {
                                        return Some(self.build_fibonacci_replacement(span, &a, &b, n_expr));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    fn analyze_prefix_sum_loop(
        &self,
        loop_var: &str,
        n_expr: &Expr,
        body: &Block,
        span: Span,
    ) -> Option<Stmt> {
        if body.stmts.len() != 1 || body.tail.is_some() { return None; }

        let stmt = &body.stmts[0];
        if let Stmt::Expr { expr, .. } = stmt {
            if let Expr::Assign { op: AssignOpKind::Assign, target, value, .. } = expr {
                if let Expr::Index { object: tobj, indices, .. } = target.as_ref() {
                    if indices.len() == 1 && Self::expr_is_ident(&indices[0], loop_var) {
                        if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, .. } = value.as_ref() {
                            if Self::is_index_with_offset(lhs, tobj, loop_var, 0)
                                && Self::is_index_with_offset(rhs, tobj, loop_var, -1i128)
                            {
                                let arr_name = Self::extract_ident(tobj)?;
                                let optimized = Expr::Call {
                                    span,
                                    func: Box::new(Expr::Ident {
                                        span,
                                        name: "prefix_sum_simd".to_string(),
                                    }),
                                    args: vec![
                                        Expr::Ident { span, name: arr_name },
                                        n_expr.clone(),
                                    ],
                                    named: vec![],
                                };
                                return Some(Stmt::Expr { span, expr: optimized, has_semi: true });
                            }
                        }
                    }
                }
            }
        }
        None
    }

    fn analyze_reduce_loop(
        &self,
        _loop_var: &str,
        _n_expr: &Expr,
        body: &Block,
        span: Span,
    ) -> Option<Stmt> {
        if body.stmts.len() != 1 || body.tail.is_some() { return None; }

        let stmt = &body.stmts[0];
        if let Stmt::Expr { expr, .. } = stmt {
            if let Expr::Assign { op: AssignOpKind::Assign, target, value, .. } = expr {
                if let Expr::Call { func, args, named, .. } = value.as_ref() {
                    if named.is_empty() && args.len() == 2 {
                        let acc_name = Self::extract_ident(target)?;
                        if Self::expr_is_ident(&args[0], &acc_name) {
                            if let Expr::Ident { name: fn_name, .. } = func.as_ref() {
                                let second_arg = &args[1];
                                let optimized = Expr::Call {
                                    span,
                                    func: Box::new(Expr::Ident {
                                        span,
                                        name: "parallel_reduce".to_string(),
                                    }),
                                    args: vec![
                                        Expr::Ident { span, name: acc_name },
                                        Expr::Ident { span, name: fn_name.clone() },
                                        second_arg.clone(),
                                    ],
                                    named: vec![],
                                };
                                return Some(Stmt::Expr { span, expr: optimized, has_semi: true });
                            }
                        }
                    }
                }
            }
        }
        None
    }

    // ─────────────────────────────────────────────────────────────────────────
    // AST construction helpers (formula builders)
    // ─────────────────────────────────────────────────────────────────────────

    /// n*(n+1)/2
    pub fn build_triangular_formula(span: Span, n: &Expr) -> Expr {
        Expr::BinOp {
            span,
            op: BinOpKind::Div,
            lhs: Box::new(Expr::BinOp {
                span,
                op: BinOpKind::Mul,
                lhs: Box::new(n.clone()),
                rhs: Box::new(Expr::BinOp {
                    span,
                    op: BinOpKind::Add,
                    lhs: Box::new(n.clone()),
                    rhs: Box::new(Expr::IntLit { span, value: 1 }),
                }),
            }),
            rhs: Box::new(Expr::IntLit { span, value: 2 }),
        }
    }

    /// n*(n+1)*(2n+1)/6
    pub fn build_sum_of_squares_formula(span: Span, n: &Expr) -> Expr {
        Expr::BinOp {
            span,
            op: BinOpKind::Div,
            lhs: Box::new(Expr::BinOp {
                span,
                op: BinOpKind::Mul,
                lhs: Box::new(Expr::BinOp {
                    span,
                    op: BinOpKind::Mul,
                    lhs: Box::new(n.clone()),
                    rhs: Box::new(Expr::BinOp {
                        span,
                        op: BinOpKind::Add,
                        lhs: Box::new(n.clone()),
                        rhs: Box::new(Expr::IntLit { span, value: 1 }),
                    }),
                }),
                rhs: Box::new(Expr::BinOp {
                    span,
                    op: BinOpKind::Add,
                    lhs: Box::new(Expr::BinOp {
                        span,
                        op: BinOpKind::Mul,
                        lhs: Box::new(Expr::IntLit { span, value: 2 }),
                        rhs: Box::new(n.clone()),
                    }),
                    rhs: Box::new(Expr::IntLit { span, value: 1 }),
                }),
            }),
            rhs: Box::new(Expr::IntLit { span, value: 6 }),
        }
    }

    /// (n*(n+1)/2)²
    pub fn build_sum_of_cubes_formula(span: Span, n: &Expr) -> Expr {
        let t = Self::build_triangular_formula(span, n);
        Expr::BinOp {
            span,
            op: BinOpKind::Mul,
            lhs: Box::new(t.clone()),
            rhs: Box::new(t),
        }
    }

    /// n * (2*a + (n-1)*d) / 2
    fn build_arithmetic_series_loop_formula(span: Span, n: &Expr, a: &Expr, d: &Expr) -> Expr {
        Expr::BinOp {
            span,
            op: BinOpKind::Div,
            lhs: Box::new(Expr::BinOp {
                span,
                op: BinOpKind::Mul,
                lhs: Box::new(n.clone()),
                rhs: Box::new(Expr::BinOp {
                    span,
                    op: BinOpKind::Add,
                    lhs: Box::new(Expr::BinOp {
                        span,
                        op: BinOpKind::Mul,
                        lhs: Box::new(Expr::IntLit { span, value: 2 }),
                        rhs: Box::new(a.clone()),
                    }),
                    rhs: Box::new(Expr::BinOp {
                        span,
                        op: BinOpKind::Mul,
                        lhs: Box::new(Expr::BinOp {
                            span,
                            op: BinOpKind::Sub,
                            lhs: Box::new(n.clone()),
                            rhs: Box::new(Expr::IntLit { span, value: 1 }),
                        }),
                        rhs: Box::new(d.clone()),
                    }),
                }),
            }),
            rhs: Box::new(Expr::IntLit { span, value: 2 }),
        }
    }

    /// (r^n - 1) / (r - 1)
    fn build_geometric_series_formula(span: Span, r: &Expr, n: &Expr) -> Expr {
        Expr::BinOp {
            span,
            op: BinOpKind::Div,
            lhs: Box::new(Expr::BinOp {
                span,
                op: BinOpKind::Sub,
                lhs: Box::new(Expr::Pow {
                    span,
                    base: Box::new(r.clone()),
                    exp:  Box::new(n.clone()),
                }),
                rhs: Box::new(Expr::IntLit { span, value: 1 }),
            }),
            rhs: Box::new(Expr::BinOp {
                span,
                op: BinOpKind::Sub,
                lhs: Box::new(r.clone()),
                rhs: Box::new(Expr::IntLit { span, value: 1 }),
            }),
        }
    }

    /// c * (r^n - 1) / (r - 1)  →  geometric Horner result
    fn build_geometric_horner_replacement(
        &self,
        span: Span,
        acc_name: &str,
        r_expr: &Expr,
        c_expr: &Expr,
        n_expr: &Expr,
    ) -> Stmt {
        let geo = Self::build_geometric_series_formula(span, r_expr, n_expr);
        let optimized = Expr::BinOp {
            span,
            op: BinOpKind::Mul,
            lhs: Box::new(c_expr.clone()),
            rhs: Box::new(geo),
        };
        Stmt::Expr {
            span,
            expr: Expr::Assign {
                span,
                op: AssignOpKind::Assign,
                target: Box::new(Expr::Ident { span, name: acc_name.to_string() }),
                value: Box::new(optimized),
            },
            has_semi: true,
        }
    }

    /// n/2 * (2*base + (n-1)*step)
    pub fn build_arithmetic_series_formula(span: Span, base: &Expr, step: &Expr, n: u128) -> Expr {
        Expr::BinOp {
            span,
            op: BinOpKind::Mul,
            lhs: Box::new(Expr::BinOp {
                span,
                op: BinOpKind::Div,
                lhs: Box::new(Expr::IntLit { span, value: n }),
                rhs: Box::new(Expr::IntLit { span, value: 2 }),
            }),
            rhs: Box::new(Expr::BinOp {
                span,
                op: BinOpKind::Add,
                lhs: Box::new(Expr::BinOp {
                    span,
                    op: BinOpKind::Mul,
                    lhs: Box::new(Expr::IntLit { span, value: 2 }),
                    rhs: Box::new(base.clone()),
                }),
                rhs: Box::new(Expr::BinOp {
                    span,
                    op: BinOpKind::Mul,
                    lhs: Box::new(Expr::IntLit { span, value: n - 1 }),
                    rhs: Box::new(step.clone()),
                }),
            }),
        }
    }

    fn build_fibonacci_replacement(&self, span: Span, a_name: &str, _b_name: &str, n_expr: &Expr) -> Stmt {
        Stmt::Expr {
            span,
            expr: Expr::Assign {
                span,
                op: AssignOpKind::Assign,
                target: Box::new(Expr::Ident { span, name: a_name.to_string() }),
                value: Box::new(Expr::Call {
                    span,
                    func: Box::new(Expr::Ident { span, name: "fib_fast".to_string() }),
                    args: vec![n_expr.clone()],
                    named: vec![],
                }),
            },
            has_semi: true,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pattern-recognition helpers (public so tests can call them)
    // ─────────────────────────────────────────────────────────────────────────

    pub fn extract_ident(expr: &Expr) -> Option<String> {
        if let Expr::Ident { name, .. } = expr { Some(name.clone()) } else { None }
    }

    pub fn extract_ident_from_expr(expr: &Expr) -> Option<String> {
        Self::extract_ident(expr)
    }

    fn extract_pattern_name(pattern: &Pattern) -> Option<String> {
        if let Pattern::Ident { name, .. } = pattern { Some(name.clone()) } else { None }
    }

    pub fn expr_is_ident(expr: &Expr, name: &str) -> bool {
        matches!(expr, Expr::Ident { name: n, .. } if n == name)
    }

    fn is_int_lit(expr: &Expr, value: u128) -> bool {
        matches!(expr, Expr::IntLit { value: v, .. } if *v == value)
    }

    fn as_int_lit(expr: &Expr) -> Option<u128> {
        if let Expr::IntLit { value, .. } = expr { Some(*value) } else { None }
    }

    fn as_bool_lit(expr: &Expr) -> Option<bool> {
        if let Expr::BoolLit { value, .. } = expr { Some(*value) } else { None }
    }

    fn bool_lit(span: Span, value: bool) -> Expr {
        Expr::BoolLit { span, value }
    }

    /// Structural equality (ignores span).
    fn exprs_equal(a: &Expr, b: &Expr) -> bool {
        match (a, b) {
            (Expr::Ident { name: n1, .. }, Expr::Ident { name: n2, .. }) => n1 == n2,
            (Expr::IntLit { value: v1, .. }, Expr::IntLit { value: v2, .. }) => v1 == v2,
            (Expr::BoolLit { value: v1, .. }, Expr::BoolLit { value: v2, .. }) => v1 == v2,
            (
                Expr::BinOp { op: o1, lhs: l1, rhs: r1, .. },
                Expr::BinOp { op: o2, lhs: l2, rhs: r2, .. },
            ) => o1 == o2 && Self::exprs_equal(l1, l2) && Self::exprs_equal(r1, r2),
            (
                Expr::UnOp { op: o1, expr: e1, .. },
                Expr::UnOp { op: o2, expr: e2, .. },
            ) => o1 == o2 && Self::exprs_equal(e1, e2),
            (
                Expr::Pow { base: b1, exp: e1, .. },
                Expr::Pow { base: b2, exp: e2, .. },
            ) => Self::exprs_equal(b1, b2) && Self::exprs_equal(e1, e2),
            (
                Expr::Call { func: f1, args: a1, .. },
                Expr::Call { func: f2, args: a2, .. },
            ) => {
                Self::exprs_equal(f1, f2)
                    && a1.len() == a2.len()
                    && a1.iter().zip(a2.iter()).all(|(x, y)| Self::exprs_equal(x, y))
            }
            (
                Expr::Index { object: o1, indices: i1, .. },
                Expr::Index { object: o2, indices: i2, .. },
            ) => {
                Self::exprs_equal(o1, o2)
                    && i1.len() == i2.len()
                    && i1.iter().zip(i2.iter()).all(|(x, y)| Self::exprs_equal(x, y))
            }
            _ => false,
        }
    }

    fn extract_range_info(iter: &Expr) -> Option<(u128, Expr)> {
        if let Expr::Range { lo, hi, .. } = iter {
            let lo_val = lo.as_ref().and_then(|e| Self::as_int_lit(e)).unwrap_or(0);
            let hi_expr = hi.as_ref()?.as_ref().clone();
            Some((lo_val, hi_expr))
        } else {
            None
        }
    }

    fn extract_accumulation(stmt: &Stmt) -> Option<(String, &Expr, AssignOpKind)> {
        if let Stmt::Expr { expr, .. } = stmt {
            if let Expr::Assign { op, target, value, .. } = expr {
                if let Expr::Ident { name, .. } = target.as_ref() {
                    return Some((name.clone(), value, *op));
                }
            }
        }
        if let Stmt::Let { pattern, init: Some(init_expr), mutable: true, .. } = stmt {
            if let Pattern::Ident { name, .. } = pattern {
                return Some((name.clone(), init_expr, AssignOpKind::Assign));
            }
        }
        None
    }

    fn extract_plain_assignment(stmt: &Stmt) -> Option<(&str, &Expr)> {
        if let Stmt::Expr { expr, .. } = stmt {
            if let Expr::Assign { op: AssignOpKind::Assign, target, value, .. } = expr {
                if let Expr::Ident { name, .. } = target.as_ref() {
                    return Some((name, value));
                }
            }
        }
        None
    }

    /// Check if expr is `var^power` via repeated multiplication or Pow node.
    pub fn is_mul_of_var(expr: &Expr, var: &str, power: u32) -> bool {
        match power {
            0 => Self::is_int_lit(expr, 1),
            1 => Self::expr_is_ident(expr, var),
            _ => {
                if let Expr::Pow { base, exp, .. } = expr {
                    if Self::expr_is_ident(base, var) {
                        return Self::is_int_lit(exp, power as u128);
                    }
                }
                if let Expr::BinOp { op: BinOpKind::Mul, lhs, rhs, .. } = expr {
                    for p in 1..power {
                        if Self::is_mul_of_var(lhs, var, p) && Self::is_mul_of_var(rhs, var, power - p) {
                            return true;
                        }
                    }
                }
                false
            }
        }
    }

    fn is_power_of_ident(expr: &Expr, ident: &str, power: u32) -> bool {
        match power {
            0 => Self::is_int_lit(expr, 1),
            1 => Self::expr_is_ident(expr, ident),
            _ => {
                if let Expr::Pow { base, exp, .. } = expr {
                    return Self::expr_is_ident(base, ident) && Self::is_int_lit(exp, power as u128);
                }
                if let Expr::BinOp { op: BinOpKind::Mul, lhs, rhs, .. } = expr {
                    for p in 1..power {
                        if Self::is_power_of_ident(lhs, ident, p)
                            && Self::is_power_of_ident(rhs, ident, power - p)
                        {
                            return true;
                        }
                    }
                }
                false
            }
        }
    }

    /// Extract `(a, d)` from `a + i*d` or `i*d + a`.
    pub fn extract_arithmetic_term(expr: &Expr, loop_var: &str) -> Option<(Expr, Expr)> {
        if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, .. } = expr {
            // a + i*d
            if let Expr::BinOp { op: BinOpKind::Mul, lhs: ml, rhs: mr, .. } = rhs.as_ref() {
                if Self::expr_is_ident(ml, loop_var) { return Some((*lhs.clone(), *mr.clone())); }
                if Self::expr_is_ident(mr, loop_var) { return Some((*lhs.clone(), *ml.clone())); }
            }
            // i*d + a
            if let Expr::BinOp { op: BinOpKind::Mul, lhs: ml, rhs: mr, .. } = lhs.as_ref() {
                if Self::expr_is_ident(ml, loop_var) { return Some((*rhs.clone(), *mr.clone())); }
                if Self::expr_is_ident(mr, loop_var) { return Some((*rhs.clone(), *ml.clone())); }
            }
        }
        None
    }

    fn is_index_with_offset(expr: &Expr, expected_obj: &Expr, loop_var: &str, offset: i128) -> bool {
        if let Expr::Index { object, indices, .. } = expr {
            if indices.len() == 1 && Self::exprs_equal(object, expected_obj) {
                let idx = &indices[0];
                return match offset {
                    0   => Self::expr_is_ident(idx, loop_var),
                    -1  => matches!(idx, Expr::BinOp { op: BinOpKind::Sub, lhs, rhs, .. }
                                    if Self::expr_is_ident(lhs, loop_var) && Self::is_int_lit(rhs, 1)),
                    1   => matches!(idx, Expr::BinOp { op: BinOpKind::Add, lhs, rhs, .. }
                                    if Self::expr_is_ident(lhs, loop_var) && Self::is_int_lit(rhs, 1)),
                    _   => false,
                };
            }
        }
        false
    }

    fn is_adjacent_index_access(lhs: &Expr, rhs: &Expr) -> bool {
        if let (
            Expr::Index { object: o1, indices: i1, .. },
            Expr::Index { object: o2, indices: i2, .. },
        ) = (lhs, rhs)
        {
            if Self::exprs_equal(o1, o2) && i1.len() == 1 && i2.len() == 1 {
                // arr[i-1] + arr[i]  or arr[i] + arr[i-1]
                let has_minus = |idx: &Expr| {
                    matches!(idx, Expr::BinOp { op: BinOpKind::Sub, rhs, .. }
                             if Self::is_int_lit(rhs, 1))
                };
                return has_minus(&i1[0]) || has_minus(&i2[0]);
            }
        }
        false
    }

    fn collect_add_terms(expr: &Expr) -> Vec<&Expr> {
        match expr {
            Expr::BinOp { op: BinOpKind::Add, lhs, rhs, .. } => {
                let mut v = Self::collect_add_terms(lhs);
                v.extend(Self::collect_add_terms(rhs));
                v
            }
            _ => vec![expr],
        }
    }

    pub fn is_sequential_sum_to_n(expr: &Expr, n_name: &str) -> Option<String> {
        match expr {
            Expr::BinOp { op: BinOpKind::Add, lhs, rhs, .. } => {
                if let Expr::BinOp { op: BinOpKind::Sub, lhs: sl, rhs: step, .. } = rhs.as_ref() {
                    if let Expr::Ident { name, .. } = sl.as_ref() {
                        if name == n_name && Self::is_int_lit(step, 1) {
                            return Some(n_name.to_string());
                        }
                    }
                }
                if let Expr::Ident { name, .. } = rhs.as_ref() {
                    if name == n_name { return Self::extract_ident(lhs); }
                }
                None
            }
            Expr::BinOp { op: BinOpKind::Sub, lhs, rhs, .. } => {
                if let Expr::Ident { name, .. } = lhs.as_ref() {
                    if name == n_name && Self::is_int_lit(rhs, 1) {
                        return Some(n_name.to_string());
                    }
                }
                None
            }
            Expr::Ident { name, .. } if name == n_name => Some(n_name.to_string()),
            _ => None,
        }
    }

    fn collect_polynomial_terms(expr: &Expr) -> Option<Vec<(u128, Option<String>)>> {
        let mut terms = Vec::new();
        Self::collect_terms_recursive(expr, &mut terms)?;
        if terms.is_empty() { None } else { Some(terms) }
    }

    fn collect_terms_recursive(expr: &Expr, terms: &mut Vec<(u128, Option<String>)>) -> Option<()> {
        match expr {
            Expr::BinOp { op: BinOpKind::Add, lhs, rhs, .. } => {
                Self::collect_terms_recursive(lhs, terms)?;
                Self::collect_terms_recursive(rhs, terms)?;
                Some(())
            }
            Expr::BinOp { op: BinOpKind::Mul, lhs, rhs, .. } => {
                if let (Expr::IntLit { value, .. }, Expr::Ident { name, .. }) = (lhs.as_ref(), rhs.as_ref()) {
                    terms.push((*value, Some(name.clone())));
                    return Some(());
                }
                if let (Expr::Ident { name, .. }, Expr::IntLit { value, .. }) = (lhs.as_ref(), rhs.as_ref()) {
                    terms.push((*value, Some(name.clone())));
                    return Some(());
                }
                None
            }
            Expr::IntLit { value, .. } => { terms.push((*value, None)); Some(()) }
            Expr::Ident { name, .. }   => { terms.push((1, Some(name.clone()))); Some(()) }
            _ => None,
        }
    }

    fn extract_arithmetic_base_step(first: &Expr, second: &Expr) -> Option<(Expr, Expr)> {
        if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, .. } = second {
            if Self::exprs_equal(lhs, first) { return Some((first.clone(), *rhs.clone())); }
            if Self::exprs_equal(rhs, first) { return Some((first.clone(), *lhs.clone())); }
            if let Some(step) = Self::extract_step_from_addition(second, first) {
                return Some((first.clone(), step));
            }
        }
        None
    }

    fn extract_step_from_addition(sum_expr: &Expr, base_expr: &Expr) -> Option<Expr> {
        if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, .. } = sum_expr {
            if Self::exprs_equal(lhs, base_expr) { return Some(*rhs.clone()); }
            if Self::exprs_equal(rhs, base_expr) { return Some(*lhs.clone()); }
        }
        None
    }

    pub fn matches_arithmetic_term(term: &Expr, base: &Expr, step: &Expr, k: u128) -> bool {
        if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, .. } = term {
            if Self::exprs_equal(lhs, base) {
                if let Expr::BinOp { op: BinOpKind::Mul, lhs: ml, rhs: mr, .. } = rhs.as_ref() {
                    if (Self::is_int_lit(ml, k) && Self::exprs_equal(mr, step))
                        || (Self::is_int_lit(mr, k) && Self::exprs_equal(ml, step))
                    {
                        return true;
                    }
                }
                if k == 1 && Self::exprs_equal(rhs, step) { return true; }
            }
            if Self::exprs_equal(rhs, base) {
                if let Expr::BinOp { op: BinOpKind::Mul, lhs: ml, rhs: mr, .. } = lhs.as_ref() {
                    if (Self::is_int_lit(ml, k) && Self::exprs_equal(mr, step))
                        || (Self::is_int_lit(mr, k) && Self::exprs_equal(ml, step))
                    {
                        return true;
                    }
                }
                if k == 1 && Self::exprs_equal(lhs, step) { return true; }
            }
        }
        false
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pattern helpers for "extend" rules (rule_extend_triangular / _cubes)
    // ─────────────────────────────────────────────────────────────────────────

    /// If `expr` is `n*(n+1)/2`, return `n`.
    fn extract_triangular_n(expr: &Expr) -> Option<Expr> {
        // Shape: (n * (n + 1)) / 2
        if let Expr::BinOp { op: BinOpKind::Div, lhs, rhs, .. } = expr {
            if Self::is_int_lit(rhs, 2) {
                if let Expr::BinOp { op: BinOpKind::Mul, lhs: n_expr, rhs: n1_expr, .. } = lhs.as_ref() {
                    if let Expr::BinOp { op: BinOpKind::Add, lhs: n2, rhs: one, .. } = n1_expr.as_ref() {
                        if Self::is_int_lit(one, 1) && Self::exprs_equal(n_expr, n2) {
                            return Some(*n_expr.clone());
                        }
                    }
                }
            }
        }
        None
    }

    /// If `expr` is `(n*(n+1)/2)^2`, return `n`.
    fn extract_sum_of_cubes_n(expr: &Expr) -> Option<Expr> {
        if let Expr::BinOp { op: BinOpKind::Mul, lhs, rhs, .. } = expr {
            if Self::exprs_equal(lhs, rhs) {
                return Self::extract_triangular_n(lhs);
            }
        }
        None
    }

    /// Check if `expr` equals `(n+1)^3`.
    fn is_cube_of_n_plus_one(expr: &Expr, n: &Expr) -> bool {
        // (n+1)^3  can appear as Pow((n+1), 3) or as (n+1)*(n+1)*(n+1)
        if let Expr::Pow { base, exp, .. } = expr {
            if Self::is_int_lit(exp, 3) {
                if let Expr::BinOp { op: BinOpKind::Add, lhs, rhs, .. } = base.as_ref() {
                    if Self::exprs_equal(lhs, n) && Self::is_int_lit(rhs, 1) { return true; }
                }
            }
        }
        // Also try three multiplications
        if let Expr::BinOp { op: BinOpKind::Mul, lhs, rhs, .. } = expr {
            let n1 = Expr::BinOp {
                span: Span::dummy(),
                op: BinOpKind::Add,
                lhs: Box::new(n.clone()),
                rhs: Box::new(Expr::IntLit { span: Span::dummy(), value: 1 }),
            };
            if Self::exprs_equal(lhs, &n1) && Self::exprs_equal(rhs, &n1) { return true; }
        }
        false
    }
}

impl Default for SemanticSuperoptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn span() -> Span { Span::dummy() }

    fn ident(name: &str) -> Expr {
        Expr::Ident { span: span(), name: name.to_string() }
    }

    fn int(v: u128) -> Expr {
        Expr::IntLit { span: span(), value: v }
    }

    fn binop(op: BinOpKind, lhs: Expr, rhs: Expr) -> Expr {
        Expr::BinOp { span: span(), op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
    }

    fn call(name: &str, args: Vec<Expr>) -> Expr {
        Expr::Call {
            span: span(),
            func: Box::new(ident(name)),
            args,
            named: vec![],
        }
    }

    // ── Rule count ───────────────────────────────────────────────────────────

    #[test]
    fn test_rule_count_exceeds_100() {
        let opt = SemanticSuperoptimizer::new();
        assert!(
            opt.rules.len() >= 100,
            "Expected >= 100 rules, got {}",
            opt.rules.len()
        );
    }

    // ── §1 Constant folding ──────────────────────────────────────────────────

    #[test]
    fn test_cf_add() {
        let opt = SemanticSuperoptimizer::new();
        let e = binop(BinOpKind::Add, int(3), int(4));
        let r = opt.cf_binop(&e, BinOpKind::Add, |a, b| a + b).unwrap();
        assert!(matches!(r, Expr::IntLit { value: 7, .. }));
    }

    #[test]
    fn test_cf_mul() {
        let opt = SemanticSuperoptimizer::new();
        let e = binop(BinOpKind::Mul, int(6), int(7));
        let r = opt.cf_binop(&e, BinOpKind::Mul, |a, b| a * b).unwrap();
        assert!(matches!(r, Expr::IntLit { value: 42, .. }));
    }

    #[test]
    fn test_cf_div_by_zero_rejected() {
        let opt = SemanticSuperoptimizer::new();
        let e = binop(BinOpKind::Div, int(5), int(0));
        let r = opt.cf_binop_guard(&e, BinOpKind::Div, |a, b| if b != 0 { Some(a / b) } else { None });
        assert!(r.is_none());
    }

    #[test]
    fn test_cf_pow() {
        let opt = SemanticSuperoptimizer::new();
        let e = Expr::Pow { span: span(), base: Box::new(int(2)), exp: Box::new(int(10)) };
        let r = opt.cf_pow(&e).unwrap();
        assert!(matches!(r, Expr::IntLit { value: 1024, .. }));
    }

    #[test]
    fn test_cf_shl() {
        let opt = SemanticSuperoptimizer::new();
        let e = binop(BinOpKind::Shl, int(1), int(8));
        let r = opt.cf_shift(&e, BinOpKind::Shl).unwrap();
        assert!(matches!(r, Expr::IntLit { value: 256, .. }));
    }

    #[test]
    fn test_cf_bool_and() {
        let opt = SemanticSuperoptimizer::new();
        let e = binop(BinOpKind::And, Expr::BoolLit { span: span(), value: true }, Expr::BoolLit { span: span(), value: false });
        let r = opt.cf_bool_binop(&e).unwrap();
        assert!(matches!(r, Expr::BoolLit { value: false, .. }));
    }

    #[test]
    fn test_cf_eq_lits() {
        let opt = SemanticSuperoptimizer::new();
        let e = binop(BinOpKind::Eq, int(5), int(5));
        let r = opt.cf_cmp(&e, BinOpKind::Eq, |a, b| a == b).unwrap();
        assert!(matches!(r, Expr::BoolLit { value: true, .. }));
    }

    // ── §2 Identity / annihilator ────────────────────────────────────────────

    #[test]
    fn test_add_zero_right() {
        let opt = SemanticSuperoptimizer::new();
        let e = binop(BinOpKind::Add, ident("x"), int(0));
        let r = opt.identity_right(&e, BinOpKind::Add, 0).unwrap();
        assert!(matches!(&r, Expr::Ident { name, .. } if name == "x"));
    }

    #[test]
    fn test_mul_zero_right() {
        let opt = SemanticSuperoptimizer::new();
        let e = binop(BinOpKind::Mul, ident("x"), int(0));
        let r = opt.annihilator_right(&e, BinOpKind::Mul, 0).unwrap();
        assert!(matches!(r, Expr::IntLit { value: 0, .. }));
    }

    #[test]
    fn test_pow_zero_exp() {
        let opt = SemanticSuperoptimizer::new();
        let e = Expr::Pow { span: span(), base: Box::new(ident("x")), exp: Box::new(int(0)) };
        let r = opt.pow_zero(&e).unwrap();
        assert!(matches!(r, Expr::IntLit { value: 1, .. }));
    }

    #[test]
    fn test_pow_one_exp() {
        let opt = SemanticSuperoptimizer::new();
        let e = Expr::Pow { span: span(), base: Box::new(ident("x")), exp: Box::new(int(1)) };
        let r = opt.pow_one(&e).unwrap();
        assert!(matches!(&r, Expr::Ident { name, .. } if name == "x"));
    }

    // ── §3 Algebraic normalisation ───────────────────────────────────────────

    #[test]
    fn test_sub_self() {
        let opt = SemanticSuperoptimizer::new();
        let e = binop(BinOpKind::Sub, ident("n"), ident("n"));
        let r = opt.sub_self(&e).unwrap();
        assert!(matches!(r, Expr::IntLit { value: 0, .. }));
    }

    #[test]
    fn test_xor_self() {
        let opt = SemanticSuperoptimizer::new();
        let e = binop(BinOpKind::BitXor, ident("x"), ident("x"));
        let r = opt.xor_self(&e).unwrap();
        assert!(matches!(r, Expr::IntLit { value: 0, .. }));
    }

    #[test]
    fn test_add_self_to_mul2() {
        let opt = SemanticSuperoptimizer::new();
        let e = binop(BinOpKind::Add, ident("x"), ident("x"));
        let r = opt.add_self(&e).unwrap();
        assert!(matches!(&r, Expr::BinOp { op: BinOpKind::Mul, lhs, .. }
                         if matches!(lhs.as_ref(), Expr::IntLit { value: 2, .. })));
    }

    #[test]
    fn test_double_neg() {
        let opt = SemanticSuperoptimizer::new();
        let inner = Expr::UnOp { span: span(), op: UnOpKind::Neg, expr: Box::new(ident("x")) };
        let e = Expr::UnOp { span: span(), op: UnOpKind::Neg, expr: Box::new(inner) };
        let r = opt.double_neg(&e).unwrap();
        assert!(matches!(&r, Expr::Ident { name, .. } if name == "x"));
    }

    #[test]
    fn test_rem_one() {
        let opt = SemanticSuperoptimizer::new();
        let e = binop(BinOpKind::Rem, ident("x"), int(1));
        let r = opt.rem_one(&e).unwrap();
        assert!(matches!(r, Expr::IntLit { value: 0, .. }));
    }

    // ── §4 Bit tricks ────────────────────────────────────────────────────────

    #[test]
    fn test_mul_pow2_to_shl() {
        let opt = SemanticSuperoptimizer::new();
        // x * 8 → x << 3
        let e = binop(BinOpKind::Mul, ident("x"), int(8));
        let r = opt.mul_pow2_to_shl(&e).unwrap();
        assert!(matches!(&r, Expr::BinOp { op: BinOpKind::Shl, rhs, .. }
                         if matches!(rhs.as_ref(), Expr::IntLit { value: 3, .. })));
    }

    #[test]
    fn test_div_pow2_to_shr() {
        let opt = SemanticSuperoptimizer::new();
        let e = binop(BinOpKind::Div, ident("x"), int(16));
        let r = opt.div_pow2_to_shr(&e).unwrap();
        assert!(matches!(&r, Expr::BinOp { op: BinOpKind::Shr, rhs, .. }
                         if matches!(rhs.as_ref(), Expr::IntLit { value: 4, .. })));
    }

    #[test]
    fn test_rem_pow2_to_and() {
        let opt = SemanticSuperoptimizer::new();
        let e = binop(BinOpKind::Rem, ident("x"), int(32));
        let r = opt.rem_pow2_to_and(&e).unwrap();
        assert!(matches!(&r, Expr::BinOp { op: BinOpKind::BitAnd, rhs, .. }
                         if matches!(rhs.as_ref(), Expr::IntLit { value: 31, .. })));
    }

    #[test]
    fn test_mul3_strength() {
        let opt = SemanticSuperoptimizer::new();
        let e = binop(BinOpKind::Mul, ident("x"), int(3));
        // 3 = 2^1 + 1 → (x << 1) + x
        let r = opt.mul_small_const_strength(&e, 3).unwrap();
        assert!(matches!(&r, Expr::BinOp { op: BinOpKind::Add, .. }));
    }

    #[test]
    fn test_merge_shl() {
        let opt = SemanticSuperoptimizer::new();
        // (x << 2) << 3 → x << 5
        let inner = binop(BinOpKind::Shl, ident("x"), int(2));
        let e = binop(BinOpKind::Shl, inner, int(3));
        let r = opt.merge_shifts(&e, BinOpKind::Shl).unwrap();
        assert!(matches!(&r, Expr::BinOp { op: BinOpKind::Shl, rhs, .. }
                         if matches!(rhs.as_ref(), Expr::IntLit { value: 5, .. })));
    }

    // ── §5 Boolean algebra ───────────────────────────────────────────────────

    #[test]
    fn test_demorgan_and() {
        let opt = SemanticSuperoptimizer::new();
        let inner = binop(BinOpKind::And, ident("a"), ident("b"));
        let e = Expr::UnOp { span: span(), op: UnOpKind::Not, expr: Box::new(inner) };
        let r = opt.demorgan(&e, BinOpKind::And, BinOpKind::Or).unwrap();
        assert!(matches!(&r, Expr::BinOp { op: BinOpKind::Or, .. }));
    }

    #[test]
    fn test_double_not() {
        let opt = SemanticSuperoptimizer::new();
        let e = Expr::UnOp {
            span: span(),
            op: UnOpKind::Not,
            expr: Box::new(Expr::UnOp { span: span(), op: UnOpKind::Not, expr: Box::new(ident("a")) }),
        };
        let r = opt.double_not(&e).unwrap();
        assert!(matches!(&r, Expr::Ident { name, .. } if name == "a"));
    }

    #[test]
    fn test_contradiction() {
        let opt = SemanticSuperoptimizer::new();
        let not_a = Expr::UnOp { span: span(), op: UnOpKind::Not, expr: Box::new(ident("a")) };
        let e = binop(BinOpKind::And, ident("a"), not_a);
        let r = opt.contradiction(&e).unwrap();
        assert!(matches!(r, Expr::BoolLit { value: false, .. }));
    }

    // ── §6 Comparison normalisation ──────────────────────────────────────────

    #[test]
    fn test_eq_self() {
        let opt = SemanticSuperoptimizer::new();
        let e = binop(BinOpKind::Eq, ident("x"), ident("x"));
        let r = opt.eq_self(&e).unwrap();
        assert!(matches!(r, Expr::BoolLit { value: true, .. }));
    }

    #[test]
    fn test_lt_self() {
        let opt = SemanticSuperoptimizer::new();
        let e = binop(BinOpKind::Lt, ident("x"), ident("x"));
        let r = opt.lt_self(&e).unwrap();
        assert!(matches!(r, Expr::BoolLit { value: false, .. }));
    }

    // ── §7 Series rules ──────────────────────────────────────────────────────

    #[test]
    fn test_triangular_sum() {
        let opt = SemanticSuperoptimizer::new();
        // n + (n - 1) pattern
        let sub_n1 = binop(BinOpKind::Sub, ident("n"), int(1));
        let e = binop(BinOpKind::Add, sub_n1, ident("n"));
        let r = opt.try_triangular_sum(&e);
        assert!(r.is_some());
        assert_eq!(r.unwrap().pattern_name, "triangular_sum");
    }

    #[test]
    fn test_geometric_series() {
        let opt = SemanticSuperoptimizer::new();
        // 1 + r + r*r
        let r_sq = binop(BinOpKind::Mul, ident("r"), ident("r"));
        let e = binop(BinOpKind::Add, binop(BinOpKind::Add, int(1), ident("r")), r_sq);
        let result = opt.try_geometric_series(&e);
        assert!(result.is_some());
        assert_eq!(result.unwrap().pattern_name, "geometric_series");
    }

    #[test]
    fn test_sum_of_squares() {
        let opt = SemanticSuperoptimizer::new();
        // 1 + 4 + 9
        let e = binop(BinOpKind::Add, binop(BinOpKind::Add, int(1), int(4)), int(9));
        let r = opt.try_sum_of_squares(&e).unwrap();
        assert_eq!(r.pattern_name, "sum_of_squares");
        assert!(matches!(r.optimized, Expr::IntLit { value: 14, .. }));
    }

    #[test]
    fn test_sum_of_cubes() {
        let opt = SemanticSuperoptimizer::new();
        // 1 + 8 + 27
        let e = binop(BinOpKind::Add, binop(BinOpKind::Add, int(1), int(8)), int(27));
        let r = opt.try_sum_of_cubes(&e).unwrap();
        assert_eq!(r.pattern_name, "sum_of_cubes");
        assert!(matches!(r.optimized, Expr::IntLit { value: 36, .. }));
    }

    // ── §8 Special functions ─────────────────────────────────────────────────

    #[test]
    fn test_fib_to_fast() {
        let opt = SemanticSuperoptimizer::new();
        let e = call("fib", vec![ident("n")]);
        let r = opt.try_matrix_power_pattern(&e).unwrap();
        assert_eq!(r.pattern_name, "matrix_power_fibonacci");
        assert!(matches!(&r.optimized, Expr::Call { func, .. }
                         if matches!(func.as_ref(), Expr::Ident { name, .. } if name == "fib_fast")));
    }

    #[test]
    fn test_factorial_small_lookup() {
        let opt = SemanticSuperoptimizer::new();
        for (n, expected) in [(0u128,1u128),(1,1),(5,120),(10,3628800),(12,479001600)] {
            let e = call("factorial", vec![int(n)]);
            let r = opt.rule_factorial_small(&e).unwrap();
            assert!(matches!(&r, Expr::IntLit { value: v, .. } if *v == expected),
                    "factorial({n}) expected {expected}");
        }
    }

    #[test]
    fn test_gcd_zero() {
        let opt = SemanticSuperoptimizer::new();
        let e = call("gcd", vec![int(0), ident("x")]);
        let r = opt.rule_gcd_zero(&e).unwrap();
        assert!(matches!(&r, Expr::Ident { name, .. } if name == "x"));
    }

    #[test]
    fn test_gcd_self() {
        let opt = SemanticSuperoptimizer::new();
        let e = call("gcd", vec![ident("n"), ident("n")]);
        let r = opt.rule_gcd_self(&e).unwrap();
        assert!(matches!(&r, Expr::Ident { name, .. } if name == "n"));
    }

    #[test]
    fn test_abs_idem() {
        let opt = SemanticSuperoptimizer::new();
        let inner = call("abs", vec![ident("x")]);
        let e = call("abs", vec![inner]);
        let r = opt.rule_abs_idem(&e).unwrap();
        // result should be abs(x), not abs(abs(x))
        assert!(matches!(&r, Expr::Call { func, .. }
                         if matches!(func.as_ref(), Expr::Ident { name, .. } if name == "abs")));
    }

    #[test]
    fn test_min_self() {
        let opt = SemanticSuperoptimizer::new();
        let e = call("min", vec![ident("x"), ident("x")]);
        let r = opt.rule_minmax_self(&e, "min").unwrap();
        assert!(matches!(&r, Expr::Ident { name, .. } if name == "x"));
    }

    // ── §9 Cancellation ──────────────────────────────────────────────────────

    #[test]
    fn test_add_sub_cancel() {
        let opt = SemanticSuperoptimizer::new();
        // (x + y) - y → x
        let e = binop(
            BinOpKind::Sub,
            binop(BinOpKind::Add, ident("x"), ident("y")),
            ident("y"),
        );
        let r = opt.rule_add_sub_cancel(&e).unwrap();
        assert!(matches!(&r, Expr::Ident { name, .. } if name == "x"));
    }

    #[test]
    fn test_mul_div_cancel() {
        let opt = SemanticSuperoptimizer::new();
        // (x * y) / y → x  (y is a literal)
        let e = binop(
            BinOpKind::Div,
            binop(BinOpKind::Mul, ident("x"), int(5)),
            int(5),
        );
        let r = opt.rule_mul_div_cancel(&e).unwrap();
        assert!(matches!(&r, Expr::Ident { name, .. } if name == "x"));
    }

    // ── §10 pow-by-squaring annotation ──────────────────────────────────────

    #[test]
    fn test_pow_by_squaring() {
        let opt = SemanticSuperoptimizer::new();
        let e = Expr::Pow { span: span(), base: Box::new(ident("x")), exp: Box::new(int(16)) };
        let r = opt.rule_pow_by_squaring(&e).unwrap();
        assert!(matches!(&r, Expr::Call { func, .. }
                         if matches!(func.as_ref(), Expr::Ident { name, .. } if name == "pow_fast")));
    }

    // ── Helper tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_is_mul_of_var() {
        let e2 = binop(BinOpKind::Mul, ident("i"), ident("i"));
        assert!(SemanticSuperoptimizer::is_mul_of_var(&e2, "i", 2));
        let e3 = binop(BinOpKind::Mul, e2.clone(), ident("i"));
        assert!(SemanticSuperoptimizer::is_mul_of_var(&e3, "i", 3));
        assert!(!SemanticSuperoptimizer::is_mul_of_var(&e2, "i", 3));
    }

    #[test]
    fn test_extract_arithmetic_term() {
        // a + i * d
        let e = binop(BinOpKind::Add, ident("a"), binop(BinOpKind::Mul, ident("i"), ident("d")));
        let (a, d) = SemanticSuperoptimizer::extract_arithmetic_term(&e, "i").unwrap();
        assert!(matches!(&a, Expr::Ident { name, .. } if name == "a"));
        assert!(matches!(&d, Expr::Ident { name, .. } if name == "d"));
    }

    #[test]
    fn test_build_triangular_formula() {
        let n = ident("n");
        let f = SemanticSuperoptimizer::build_triangular_formula(span(), &n);
        // n * (n + 1) / 2
        assert!(matches!(f, Expr::BinOp { op: BinOpKind::Div, .. }));
    }

    #[test]
    fn test_build_sum_of_squares_formula() {
        let n = ident("n");
        let f = SemanticSuperoptimizer::build_sum_of_squares_formula(span(), &n);
        assert!(matches!(f, Expr::BinOp { op: BinOpKind::Div, .. }));
    }

    #[test]
    fn test_build_sum_of_cubes_formula() {
        let n = ident("n");
        let f = SemanticSuperoptimizer::build_sum_of_cubes_formula(span(), &n);
        assert!(matches!(f, Expr::BinOp { op: BinOpKind::Mul, .. }));
    }

    #[test]
    fn test_exprs_equal_structurally() {
        let a = binop(BinOpKind::Add, ident("x"), int(1));
        let b = binop(BinOpKind::Add, ident("x"), int(1));
        let c = binop(BinOpKind::Add, ident("y"), int(1));
        assert!(SemanticSuperoptimizer::exprs_equal(&a, &b));
        assert!(!SemanticSuperoptimizer::exprs_equal(&a, &c));
    }

    #[test]
    fn test_superoptimizer_creation() {
        let opt = SemanticSuperoptimizer::new();
        assert_eq!(opt.patterns_matched, 0);
        assert_eq!(opt.total_estimated_speedup, 0.0);
        assert!(opt.verify_equivalence);
        assert_eq!(opt.verif_inputs, 16);
    }
}
