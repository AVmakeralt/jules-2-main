// =============================================================================
// CEGIS (Counter-Example Guided Inductive Synthesis) Verification Layer
//
// SMT-backed equivalence verification for the Jules superoptimizer.
// This is the STOKE architecture: test-vector filter rejects 99.9% of
// candidates in nanoseconds; SMT only runs on the ~0.1% that pass all tests.
//
// Architecture:
//   1. Test-vector filter: evaluate src & candidate on N concrete inputs.
//      If any disagree → reject immediately (no SMT needed).
//   2. SMT verify: encode both programs as Z3 bitvector expressions.
//      Ask solver: ∃ input . src(input) ≠ candidate(input)?
//      - UNSAT → proven equivalent → accept
//      - SAT   → counterexample found → add to test vectors → loop
//   3. CEGIS loop: iterate steps 1-2 until UNSAT or max_iterations.
//
// When compiled without the `smt-verify` feature (which requires libclang
// for the z3-sys build), the SMT step is replaced with an Inconclusive
// result. The test-vector filter still works and catches ~99.9% of
// incorrect candidates — only formal proof is unavailable.
// =============================================================================

#![cfg(feature = "core-superopt")]

use std::collections::HashMap;

#[cfg(feature = "smt-verify")]
use std::time::Instant;

#[cfg(feature = "smt-verify")]
use z3::ast::{Ast, Bool, BV};
#[cfg(feature = "smt-verify")]
use z3::{Config, Context, SatResult, Solver};

use crate::compiler::ast::{BinOpKind, UnOpKind};
use crate::optimizer::mcts_superoptimizer::Instr;

// =============================================================================
// §1  VerifyResult
// =============================================================================

/// The outcome of an equivalence verification attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyResult {
    /// The two programs are provably equivalent for all inputs.
    Equivalent,
    /// A counterexample was found: the given input values produce
    /// different outputs. The vector length equals the number of
    /// program variables (in sorted order).
    CounterExample(Vec<u64>),
    /// The SMT solver timed out before reaching a conclusion.
    Timeout,
    /// Verification could not be completed for another reason
    /// (e.g., unsupported operation, solver error, or SMT unavailable).
    Inconclusive(String),
}

// =============================================================================
// §2  CegisStats
// =============================================================================

/// Statistics collected by the CEGIS verifier across all invocations.
#[derive(Debug, Clone, Default)]
pub struct CegisStats {
    /// Total number of `verify()` calls.
    pub total_verifications: u64,
    /// Number of verifications that returned `Equivalent`.
    pub equivalent_count: u64,
    /// Number of verifications that returned `CounterExample`.
    pub counterexample_count: u64,
    /// Number of verifications that returned `Timeout`.
    pub timeout_count: u64,
    /// Number of verifications that returned `Inconclusive`.
    pub inconclusive_count: u64,
    /// Number of candidates rejected by the fast test-vector filter
    /// (never reached SMT).
    pub filter_rejections: u64,
    /// Number of SMT solver invocations.
    pub smt_invocations: u64,
    /// Total wall-clock time spent in SMT solving (milliseconds).
    pub smt_time_ms: u64,
    /// Total number of CEGIS iterations across all verifications.
    pub cegis_iterations: u64,
    /// Number of counterexamples extracted from SMT that were added
    /// to the test-vector pool.
    pub counterexamples_added: u64,
}

// =============================================================================
// §3  CegisVerifier
// =============================================================================

/// Stateful CEGIS verifier that maintains a growing pool of test vectors
/// and accumulates statistics across multiple verification calls.
///
/// The test-vector pool grows over time as counterexamples are discovered
/// by the SMT solver. This makes subsequent verifications faster because
/// more candidates are rejected by the cheap test-vector filter before
/// reaching the expensive SMT solver.
pub struct CegisVerifier {
    /// Bit width for bitvector encoding (typically 64 for i64 operations).
    pub bitwidth: u32,
    /// Pool of test vectors. Each inner vector has one value per program
    /// variable, in sorted variable-name order.
    test_vectors: Vec<Vec<u64>>,
    /// Maximum number of CEGIS iterations before giving up.
    pub max_iterations: usize,
    /// Timeout for each Z3 solver call, in milliseconds.
    pub smt_timeout_ms: u64,
    /// Accumulated statistics.
    stats: CegisStats,
}

impl CegisVerifier {
    /// Create a new CEGIS verifier.
    pub fn new(bitwidth: u32, num_inputs: usize, max_iterations: usize) -> Self {
        let test_vectors = Self::generate_initial_vectors(bitwidth, num_inputs);
        Self {
            bitwidth,
            test_vectors,
            max_iterations,
            smt_timeout_ms: 5000,
            stats: CegisStats::default(),
        }
    }

    /// Verify that `candidate` is semantically equivalent to `src`.
    ///
    /// This runs the full CEGIS loop:
    /// 1. Fast test-vector filter
    /// 2. SMT verification (if z3 is linked) for candidates that pass the filter
    /// 3. Counterexample extraction and test-vector augmentation
    /// 4. Iterate until proven equivalent, timed out, or max_iterations reached
    pub fn verify(&mut self, src: &Instr, candidate: &Instr) -> VerifyResult {
        self.stats.total_verifications += 1;

        // Collect the union of variable names from both programs.
        let var_names = Self::collect_all_variables(src, candidate);

        // If there are no variables (both are constants), just evaluate directly.
        if var_names.is_empty() {
            let src_val = interpret(src, &[], &[]);
            let cand_val = interpret(candidate, &[], &[]);
            return match (src_val, cand_val) {
                (Some(s), Some(c)) if s == c => {
                    self.stats.equivalent_count += 1;
                    VerifyResult::Equivalent
                }
                (Some(_), Some(_)) => {
                    self.stats.counterexample_count += 1;
                    VerifyResult::CounterExample(vec![])
                }
                _ => {
                    self.stats.inconclusive_count += 1;
                    VerifyResult::Inconclusive(
                        "Cannot evaluate constant expressions".to_string(),
                    )
                }
            };
        }

        // CEGIS loop
        for _iteration in 0..self.max_iterations {
            self.stats.cegis_iterations += 1;

            // Step 1: Fast test-vector filter.
            let src_result = self.run_test_filter(src, &var_names);
            let cand_result = self.run_test_filter(candidate, &var_names);

            if src_result.len() != cand_result.len() {
                self.stats.inconclusive_count += 1;
                return VerifyResult::Inconclusive(
                    "One or both programs could not be evaluated on test vectors".to_string(),
                );
            }

            // Check for disagreement on any test vector.
            let mut all_match = true;
            for (sv, cv) in src_result.iter().zip(cand_result.iter()) {
                if sv != cv {
                    all_match = false;
                    break;
                }
            }

            if !all_match {
                self.stats.filter_rejections += 1;
                self.stats.counterexample_count += 1;
                for (i, (sv, cv)) in src_result.iter().zip(cand_result.iter()).enumerate() {
                    if sv != cv {
                        if i < self.test_vectors.len() {
                            return VerifyResult::CounterExample(self.test_vectors[i].clone());
                        }
                    }
                }
                return VerifyResult::CounterExample(
                    self.test_vectors.first().cloned().unwrap_or_default(),
                );
            }

            // Step 2: SMT verification (only if z3 is linked).
            #[cfg(feature = "smt-verify")]
            {
                self.stats.smt_invocations += 1;
                let start = Instant::now();
                let smt_result = verify_equivalence_smt(
                    src, candidate, self.bitwidth, &var_names, self.smt_timeout_ms,
                );
                self.stats.smt_time_ms += start.elapsed().as_millis() as u64;

                match smt_result {
                    VerifyResult::Equivalent => {
                        self.stats.equivalent_count += 1;
                        return VerifyResult::Equivalent;
                    }
                    VerifyResult::CounterExample(ce_values) => {
                        self.stats.counterexamples_added += 1;
                        self.add_test_vector(ce_values.clone());
                        if iteration + 1 >= self.max_iterations {
                            self.stats.counterexample_count += 1;
                            return VerifyResult::CounterExample(ce_values);
                        }
                        // Otherwise loop back with the enriched test set.
                    }
                    VerifyResult::Timeout => {
                        self.stats.timeout_count += 1;
                        return VerifyResult::Timeout;
                    }
                    VerifyResult::Inconclusive(msg) => {
                        self.stats.inconclusive_count += 1;
                        return VerifyResult::Inconclusive(msg);
                    }
                }
            }

            #[cfg(not(feature = "smt-verify"))]
            {
                // Without Z3, we can only rely on the test-vector filter.
                // Report Inconclusive since we cannot formally prove equivalence.
                self.stats.inconclusive_count += 1;
                return VerifyResult::Inconclusive(
                    "SMT verification unavailable (z3 not linked). \
                     Test-vector filter passed but formal proof not possible."
                        .to_string(),
                );
            }
        }

        self.stats.inconclusive_count += 1;
        VerifyResult::Inconclusive(format!(
            "CEGIS loop exhausted {} iterations without a conclusion",
            self.max_iterations
        ))
    }

    /// Add a test vector to the pool.
    pub fn add_test_vector(&mut self, vector: Vec<u64>) {
        self.test_vectors.push(vector);
    }

    /// Return a reference to the accumulated statistics.
    pub fn stats(&self) -> &CegisStats {
        &self.stats
    }

    /// Return the current set of test vectors.
    pub fn test_vectors(&self) -> &[Vec<u64>] {
        &self.test_vectors
    }

    /// Evaluate `instr` on all current test vectors.
    fn run_test_filter(&self, instr: &Instr, var_names: &[String]) -> Vec<Option<u64>> {
        self.test_vectors
            .iter()
            .map(|tv| interpret(instr, tv, var_names))
            .collect()
    }

    /// Collect the sorted union of variable names from both programs.
    pub(crate) fn collect_all_variables(src: &Instr, candidate: &Instr) -> Vec<String> {
        let mut vars: Vec<String> = Vec::new();
        for v in src.variables() {
            if !vars.contains(&v) {
                vars.push(v);
            }
        }
        for v in candidate.variables() {
            if !vars.contains(&v) {
                vars.push(v);
            }
        }
        vars.sort();
        vars
    }

    /// Generate an initial set of test vectors exercising edge cases.
    fn generate_initial_vectors(bitwidth: u32, num_inputs: usize) -> Vec<Vec<u64>> {
        if num_inputs == 0 {
            return vec![];
        }
        let mask = if bitwidth < 64 { (1u64 << bitwidth) - 1 } else { u64::MAX };
        let interesting: Vec<u64> = vec![
            0, 1, mask,
            1u64 << (bitwidth - 1),
            (1u64 << (bitwidth - 1)) - 1,
            42, 255, 256,
            0xDEADBEEF,
            0xAAAAAAAAAAAAAAAA & mask,
            0x5555555555555555 & mask,
            7, 13, 65535,
            0x7FFFFFFFFFFFFFFF,
        ];
        let mut vectors = Vec::new();
        for &val in &interesting {
            vectors.push(vec![val; num_inputs]);
        }
        if num_inputs > 1 && interesting.len() >= num_inputs {
            for start in 0..interesting.len().saturating_sub(num_inputs + 1) {
                let tv: Vec<u64> = (0..num_inputs)
                    .map(|i| interesting[(start + i) % interesting.len()])
                    .collect();
                vectors.push(tv);
            }
        }
        vectors
    }
}

// =============================================================================
// §4  Test-Vector Filter (standalone)
// =============================================================================

/// Fast test-vector filter: check whether `src` and `candidate` produce the
/// same output on every test vector.
///
/// Returns `true` if all test vectors agree (candidate passes the filter),
/// `false` if any test vector shows a difference (reject immediately).
pub fn test_vector_filter(
    src: &Instr,
    candidate: &Instr,
    test_vectors: &[Vec<u64>],
) -> bool {
    let var_names = CegisVerifier::collect_all_variables(src, candidate);
    for tv in test_vectors {
        let src_val = interpret(src, tv, &var_names);
        let cand_val = interpret(candidate, tv, &var_names);
        if src_val != cand_val {
            return false;
        }
    }
    true
}

// =============================================================================
// §5  Standalone verify_equivalence Function
// =============================================================================

/// Verify equivalence of two programs using the CEGIS loop.
pub fn verify_equivalence(
    src: &Instr,
    candidate: &Instr,
    bitwidth: u32,
    num_inputs: usize,
    timeout_ms: u64,
) -> VerifyResult {
    let mut verifier = CegisVerifier::new(bitwidth, num_inputs, 10);
    verifier.smt_timeout_ms = timeout_ms;
    verifier.verify(src, candidate)
}

// =============================================================================
// §6  Interpreter (concrete evaluation — always available)
// =============================================================================

/// Interpret an `Instr` tree given concrete input values.
fn interpret(instr: &Instr, inputs: &[u64], input_names: &[String]) -> Option<u64> {
    match instr {
        Instr::ConstInt(v) => Some(*v as u64),
        Instr::ConstFloat(bits) => Some(*bits),
        Instr::ConstBool(b) => Some(if *b { 1u64 } else { 0u64 }),
        Instr::Var(name) => input_names
            .iter()
            .position(|n| n == name)
            .map(|i| inputs.get(i).copied().unwrap_or(0)),
        Instr::BinOp { op, lhs, rhs } => {
            let l = interpret(lhs, inputs, input_names)?;
            let r = interpret(rhs, inputs, input_names)?;
            Some(eval_binop(*op, l, r))
        }
        Instr::UnOp { op, operand } => {
            let v = interpret(operand, inputs, input_names)?;
            Some(eval_unop(*op, v))
        }
    }
}

/// Evaluate a binary operation on concrete u64 values with wrapping semantics.
fn eval_binop(op: BinOpKind, l: u64, r: u64) -> u64 {
    match op {
        BinOpKind::Add => l.wrapping_add(r),
        BinOpKind::Sub => l.wrapping_sub(r),
        BinOpKind::Mul => l.wrapping_mul(r),
        BinOpKind::Div => {
            if r == 0 { 0 } else { (l as i64).wrapping_div(r as i64) as u64 }
        }
        BinOpKind::Rem => {
            if r == 0 { 0 } else { (l as i64).wrapping_rem(r as i64) as u64 }
        }
        BinOpKind::FloorDiv => {
            if r == 0 { 0 } else {
                let li = l as i64;
                let ri = r as i64;
                let d = li / ri;
                let result = if (li < 0) != (ri < 0) && li % ri != 0 { d - 1 } else { d };
                result as u64
            }
        }
        BinOpKind::BitAnd => l & r,
        BinOpKind::BitOr => l | r,
        BinOpKind::BitXor => l ^ r,
        BinOpKind::Shl => l.wrapping_shl(r as u32),
        BinOpKind::Shr => ((l as i64).wrapping_shr(r as u32)) as u64,
        BinOpKind::Eq => if l == r { 1 } else { 0 },
        BinOpKind::Ne => if l != r { 1 } else { 0 },
        BinOpKind::Lt => if (l as i64) < (r as i64) { 1 } else { 0 },
        BinOpKind::Le => if (l as i64) <= (r as i64) { 1 } else { 0 },
        BinOpKind::Gt => if (l as i64) > (r as i64) { 1 } else { 0 },
        BinOpKind::Ge => if (l as i64) >= (r as i64) { 1 } else { 0 },
        BinOpKind::And => if l != 0 && r != 0 { 1 } else { 0 },
        BinOpKind::Or => if l != 0 || r != 0 { 1 } else { 0 },
    }
}

/// Evaluate a unary operation on a concrete u64 value.
fn eval_unop(op: UnOpKind, v: u64) -> u64 {
    match op {
        UnOpKind::Neg => (v as i64).wrapping_neg() as u64,
        UnOpKind::Not => if v == 0 { 1 } else { 0 },
        UnOpKind::Deref | UnOpKind::Ref | UnOpKind::RefMut => 0,
    }
}

// =============================================================================
// §7  SMT Encoding Core  (only when smt-verify feature is enabled)
// =============================================================================

#[cfg(feature = "smt-verify")]
fn verify_equivalence_smt(
    src: &Instr,
    candidate: &Instr,
    bitwidth: u32,
    var_names: &[String],
    timeout_ms: u64,
) -> VerifyResult {
    let mut cfg = Config::new();
    cfg.set_timeout_msec(timeout_ms);
    let ctx = Context::new(&cfg);

    // Create symbolic bitvector variables for each input.
    let mut inputs: HashMap<String, BV<'_>> = HashMap::new();
    for name in var_names {
        let bv = BV::new_const(&ctx, name.as_str(), bitwidth);
        inputs.insert(name.clone(), bv);
    }

    // Encode both programs.
    let src_bv = encode_instr(&ctx, src, &inputs, bitwidth);
    let cand_bv = encode_instr(&ctx, candidate, &inputs, bitwidth);

    // Assert: src != candidate
    let neq = src_bv._eq(&cand_bv).not();
    let solver = Solver::new(&ctx);
    solver.assert(&neq);

    match solver.check() {
        SatResult::Unsat => VerifyResult::Equivalent,
        SatResult::Unknown => VerifyResult::Timeout,
        SatResult::Sat => {
            match solver.get_model() {
                Some(model) => {
                    let ce_values: Vec<u64> = var_names.iter()
                        .map(|name| {
                            inputs.get(name)
                                .and_then(|bv| model.eval(bv, true))
                                .and_then(|val| val.as_u64())
                                .unwrap_or(0)
                        })
                        .collect();
                    VerifyResult::CounterExample(ce_values)
                }
                None => VerifyResult::Inconclusive(
                    "SMT solver returned SAT but no model available".to_string(),
                ),
            }
        }
    }
}

/// Encode an `Instr` as a Z3 bitvector expression.
#[cfg(feature = "smt-verify")]
fn encode_instr<'ctx>(
    ctx: &'ctx Context,
    instr: &Instr,
    inputs: &HashMap<String, BV<'ctx>>,
    bitwidth: u32,
) -> BV<'ctx> {
    match instr {
        Instr::ConstInt(v) => BV::from_u64(ctx, *v as u64, bitwidth),
        Instr::ConstFloat(bits) => BV::from_u64(ctx, *bits, bitwidth),
        Instr::ConstBool(b) => BV::from_u64(ctx, if *b { 1 } else { 0 }, bitwidth),
        Instr::Var(name) => inputs
            .get(name)
            .cloned()
            .unwrap_or_else(|| BV::from_u64(ctx, 0, bitwidth)),
        Instr::BinOp { op, lhs, rhs } => {
            let l = encode_instr(ctx, lhs, inputs, bitwidth);
            let r = encode_instr(ctx, rhs, inputs, bitwidth);
            encode_binop(ctx, *op, &l, &r, bitwidth)
        }
        Instr::UnOp { op, operand } => {
            let v = encode_instr(ctx, operand, inputs, bitwidth);
            encode_unop(ctx, *op, &v, bitwidth)
        }
    }
}

/// Encode a binary operation as a Z3 bitvector expression.
#[cfg(feature = "smt-verify")]
fn encode_binop<'ctx>(
    ctx: &'ctx Context,
    op: BinOpKind,
    l: &BV<'ctx>,
    r: &BV<'ctx>,
    bitwidth: u32,
) -> BV<'ctx> {
    let one = || BV::from_u64(ctx, 1, bitwidth);
    let zero = || BV::from_u64(ctx, 0, bitwidth);

    match op {
        BinOpKind::Add => l.bvadd(r),
        BinOpKind::Sub => l.bvsub(r),
        BinOpKind::Mul => l.bvmul(r),
        BinOpKind::Div => {
            let z = zero();
            r._eq(&z).ite(&z, &l.bvsdiv(r))
        }
        BinOpKind::Rem => {
            let z = zero();
            r._eq(&z).ite(&z, &l.bvsrem(r))
        }
        BinOpKind::FloorDiv => {
            let z = zero();
            r._eq(&z).ite(&z, &l.bvsdiv(r))
        }
        BinOpKind::BitAnd => l.bvand(r),
        BinOpKind::BitOr => l.bvor(r),
        BinOpKind::BitXor => l.bvxor(r),
        BinOpKind::Shl => l.bvshl(r),
        BinOpKind::Shr => l.bvashr(r),
        BinOpKind::Eq => l._eq(r).ite(&one(), &zero()),
        BinOpKind::Ne => l._eq(r).not().ite(&one(), &zero()),
        BinOpKind::Lt => l.bvslt(r).ite(&one(), &zero()),
        BinOpKind::Le => l.bvsle(r).ite(&one(), &zero()),
        BinOpKind::Gt => l.bvsgt(r).ite(&one(), &zero()),
        BinOpKind::Ge => l.bvsge(r).ite(&one(), &zero()),
        BinOpKind::And => {
            let l_bool = l._eq(&zero()).not();
            let r_bool = r._eq(&zero()).not();
            l_bool.and(&[&r_bool]).ite(&one(), &zero())
        }
        BinOpKind::Or => {
            let l_bool = l._eq(&zero()).not();
            let r_bool = r._eq(&zero()).not();
            l_bool.or(&[&r_bool]).ite(&one(), &zero())
        }
    }
}

/// Encode a unary operation as a Z3 bitvector expression.
#[cfg(feature = "smt-verify")]
fn encode_unop<'ctx>(
    ctx: &'ctx Context,
    op: UnOpKind,
    v: &BV<'ctx>,
    bitwidth: u32,
) -> BV<'ctx> {
    let zero = || BV::from_u64(ctx, 0, bitwidth);
    let one = || BV::from_u64(ctx, 1, bitwidth);
    match op {
        UnOpKind::Neg => v.bvneg(),
        UnOpKind::Not => v._eq(&zero()).ite(&one(), &zero()),
        UnOpKind::Deref | UnOpKind::Ref | UnOpKind::RefMut => v.clone(),
    }
}

// =============================================================================
// §8  Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_x_times_2() -> Instr {
        Instr::BinOp {
            op: BinOpKind::Mul,
            lhs: Box::new(Instr::Var("x".to_string())),
            rhs: Box::new(Instr::ConstInt(2)),
        }
    }

    fn make_x_shl_1() -> Instr {
        Instr::BinOp {
            op: BinOpKind::Shl,
            lhs: Box::new(Instr::Var("x".to_string())),
            rhs: Box::new(Instr::ConstInt(1)),
        }
    }

    fn make_x_plus_0() -> Instr {
        Instr::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(Instr::Var("x".to_string())),
            rhs: Box::new(Instr::ConstInt(0)),
        }
    }

    fn make_x() -> Instr {
        Instr::Var("x".to_string())
    }

    fn make_x_plus_1() -> Instr {
        Instr::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(Instr::Var("x".to_string())),
            rhs: Box::new(Instr::ConstInt(1)),
        }
    }

    #[test]
    fn test_interpret_const_int() {
        assert_eq!(interpret(&Instr::ConstInt(42), &[], &[]), Some(42));
    }

    #[test]
    fn test_interpret_const_bool() {
        assert_eq!(interpret(&Instr::ConstBool(true), &[], &[]), Some(1));
        assert_eq!(interpret(&Instr::ConstBool(false), &[], &[]), Some(0));
    }

    #[test]
    fn test_interpret_var() {
        assert_eq!(
            interpret(&Instr::Var("x".to_string()), &[99], &["x".to_string()]),
            Some(99)
        );
    }

    #[test]
    fn test_interpret_add() {
        let instr = Instr::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(Instr::Var("x".to_string())),
            rhs: Box::new(Instr::ConstInt(10)),
        };
        assert_eq!(interpret(&instr, &[5], &["x".to_string()]), Some(15));
    }

    #[test]
    fn test_interpret_div_by_zero() {
        let instr = Instr::BinOp {
            op: BinOpKind::Div,
            lhs: Box::new(Instr::ConstInt(10)),
            rhs: Box::new(Instr::ConstInt(0)),
        };
        assert_eq!(interpret(&instr, &[], &[]), Some(0));
    }

    #[test]
    fn test_filter_x_times_2_eq_x_shl_1() {
        let src = make_x_times_2();
        let candidate = make_x_shl_1();
        let tvs: Vec<Vec<u64>> = vec![vec![0], vec![1], vec![42], vec![255], vec![u64::MAX]];
        assert!(test_vector_filter(&src, &candidate, &tvs));
    }

    #[test]
    fn test_filter_x_plus_0_eq_x() {
        let src = make_x_plus_0();
        let candidate = make_x();
        let tvs: Vec<Vec<u64>> = vec![vec![0], vec![1], vec![u64::MAX]];
        assert!(test_vector_filter(&src, &candidate, &tvs));
    }

    #[test]
    fn test_filter_x_plus_1_neq_x() {
        let src = make_x_plus_1();
        let candidate = make_x();
        let tvs: Vec<Vec<u64>> = vec![vec![0]];
        assert!(!test_vector_filter(&src, &candidate, &tvs));
    }

    #[test]
    fn test_cegis_verifier_counterexample() {
        let mut verifier = CegisVerifier::new(64, 1, 5);
        let src = make_x_plus_1();
        let candidate = make_x();
        let result = verifier.verify(&src, &candidate);
        match result {
            VerifyResult::CounterExample(_) => {}
            other => panic!("Expected CounterExample, got {:?}", other),
        }
    }

    // ── SMT tests (only when z3 is available) ──────────────────────────

    #[cfg(feature = "smt-verify")]
    #[test]
    fn test_smt_x_times_2_eq_x_shl_1() {
        let src = make_x_times_2();
        let candidate = make_x_shl_1();
        let result = verify_equivalence(&src, &candidate, 64, 1, 10000);
        assert_eq!(result, VerifyResult::Equivalent);
    }

    #[cfg(feature = "smt-verify")]
    #[test]
    fn test_smt_x_plus_0_eq_x() {
        let src = make_x_plus_0();
        let candidate = make_x();
        let result = verify_equivalence(&src, &candidate, 64, 1, 10000);
        assert_eq!(result, VerifyResult::Equivalent);
    }

    #[cfg(feature = "smt-verify")]
    #[test]
    fn test_smt_x_plus_1_neq_x() {
        let src = make_x_plus_1();
        let candidate = make_x();
        let result = verify_equivalence(&src, &candidate, 64, 1, 10000);
        match result {
            VerifyResult::CounterExample(vals) => {
                assert_eq!(vals.len(), 1);
                assert_ne!(vals[0].wrapping_add(1), vals[0]);
            }
            other => panic!("Expected CounterExample, got {:?}", other),
        }
    }

    #[cfg(feature = "smt-verify")]
    #[test]
    fn test_smt_constant_fold() {
        let src = Instr::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(Instr::ConstInt(2)),
            rhs: Box::new(Instr::ConstInt(3)),
        };
        let candidate = Instr::ConstInt(5);
        let result = verify_equivalence(&src, &candidate, 64, 0, 5000);
        assert_eq!(result, VerifyResult::Equivalent);
    }

    #[cfg(feature = "smt-verify")]
    #[test]
    fn test_cegis_verifier_equivalent() {
        let mut verifier = CegisVerifier::new(64, 1, 5);
        let src = make_x_times_2();
        let candidate = make_x_shl_1();
        let result = verifier.verify(&src, &candidate);
        assert_eq!(result, VerifyResult::Equivalent);
        assert_eq!(verifier.stats().equivalent_count, 1);
    }
}
