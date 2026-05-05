// =============================================================================
// Known-Bits Abstract Interpretation for Superoptimizer Search Pruning
//
// For each bit position, track whether it's Known0, Known1, or Unknown.
// This lets you prove two expressions CAN'T be equivalent without invoking SMT.
//
// This is the standard "known bits" analysis used in LLVM's ValueTracking.
// The transfer functions for AND/OR/XOR are exact; ADD/SUB use the
// LLVM-style ripple-carry simulation; MUL is conservative (trailing zeros only).
//
// The main entry point for the superoptimizer is `may_be_equivalent`, which
// returns false when known-bits analysis proves two expressions differ on at
// least one known bit. This rejects many candidates in nanoseconds without
// ever invoking Z3.
// =============================================================================

use rustc_hash::FxHashMap;

use crate::compiler::ast::{BinOpKind, UnOpKind};
use crate::optimizer::mcts_superoptimizer::Instr;

// =============================================================================
// §1  Core Type
// =============================================================================

/// For each bit position, track whether it's known to be 0, known to be 1, or unknown.
/// This is the standard "known bits" analysis used in LLVM's ValueTracking.
///
/// Invariant: zeros & ones == 0 (a bit can't be known-0 AND known-1 simultaneously)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnownBits {
    /// Bitmask: 1 = this bit is known to be 0
    pub zeros: u64,
    /// Bitmask: 1 = this bit is known to be 1
    pub ones: u64,
}

impl KnownBits {
    // ── Constructors ──────────────────────────────────────────────────────

    /// All bits unknown — the most conservative starting point.
    pub fn unknown() -> Self {
        Self { zeros: 0, ones: 0 }
    }

    /// All bits known from a concrete constant value.
    pub fn constant(value: u64) -> Self {
        Self {
            zeros: !value,
            ones: value,
        }
    }

    // ── Bit-level queries ─────────────────────────────────────────────────

    /// Query a single bit: returns `Some(true)` if known-1, `Some(false)` if
    /// known-0, `None` if unknown.
    pub fn is_known(&self, bit: u32) -> Option<bool> {
        if bit >= 64 {
            return None;
        }
        let mask = 1u64 << bit;
        if self.zeros & mask != 0 {
            Some(false)
        } else if self.ones & mask != 0 {
            Some(true)
        } else {
            None
        }
    }

    /// Returns true if bit `bit` is known to be 0.
    pub fn is_known_zero(&self, bit: u32) -> bool {
        bit < 64 && (self.zeros >> bit) & 1 != 0
    }

    /// Returns true if bit `bit` is known to be 1.
    pub fn is_known_one(&self, bit: u32) -> bool {
        bit < 64 && (self.ones >> bit) & 1 != 0
    }

    /// Returns true if bit `bit` is unknown.
    pub fn is_unknown(&self, bit: u32) -> bool {
        bit < 64 && !self.is_known_zero(bit) && !self.is_known_one(bit)
    }

    /// If all 64 bits are known, return the concrete value. Otherwise `None`.
    pub fn known_value(&self) -> Option<u64> {
        // All bits are known when zeros | ones == all-ones
        if self.zeros | self.ones == !0u64 {
            Some(self.ones)
        } else {
            None
        }
    }

    /// Check whether `self` and `other` contradict each other on any bit
    /// position (i.e. one says known-0 and the other says known-1 for the
    /// same bit). If so, they cannot both describe the same value.
    pub fn intersects(&self, other: &KnownBits) -> bool {
        (self.zeros & other.ones) != 0 || (self.ones & other.zeros) != 0
    }
}

// =============================================================================
// §2  Transfer Functions for Bitwise Operations
// =============================================================================

/// AND: if either input bit is known-0, output is 0.
/// Both must be known-1 for output to be 1.
pub fn from_and(a: &KnownBits, b: &KnownBits) -> KnownBits {
    KnownBits {
        zeros: a.zeros | b.zeros,
        ones: a.ones & b.ones,
    }
}

/// OR: if either input bit is known-1, output is 1.
/// Both must be known-0 for output to be 0.
pub fn from_or(a: &KnownBits, b: &KnownBits) -> KnownBits {
    KnownBits {
        zeros: a.zeros & b.zeros,
        ones: a.ones | b.ones,
    }
}

/// XOR: known bits propagate when the other input is known.
/// Output is known-0 when both inputs have the same known value.
/// Output is known-1 when inputs have different known values.
pub fn from_xor(a: &KnownBits, b: &KnownBits) -> KnownBits {
    KnownBits {
        zeros: (a.zeros & b.zeros) | (a.ones & b.ones),
        ones: (a.zeros & b.ones) | (a.ones & b.zeros),
    }
}

// =============================================================================
// §3  Transfer Functions for Arithmetic Operations
// =============================================================================

/// ADD: ripple-carry analysis (LLVM's KnownBits::computeForAddSub).
///
/// For each bit position, we simulate all possible combinations of input bits
/// and carry-in to determine which output values and carry-out values are
/// possible. This is O(64 * 8) and very fast.
///
/// Key insight: "the low 3 bits of (x << 3) + y are exactly the low 3 bits of y"
/// because x << 3 has known-0 for the low 3 bits, so carry never propagates
/// from y into the upper bits.
pub fn from_add(a: &KnownBits, b: &KnownBits) -> KnownBits {
    let mut zeros = 0u64;
    let mut ones = 0u64;

    // Carry-in to bit 0 is always 0.
    let mut carry_may_be_0 = true;
    let mut carry_may_be_1 = false;

    for i in 0..64 {
        // Determine possible values for each input at this bit position.
        // a_i can be 0 iff the ones-bit is NOT set (known-0 or unknown)
        // a_i can be 1 iff the zeros-bit is NOT set (known-1 or unknown)
        let a_may_be_0 = (a.ones >> i) & 1 == 0;
        let a_may_be_1 = (a.zeros >> i) & 1 == 0;
        let b_may_be_0 = (b.ones >> i) & 1 == 0;
        let b_may_be_1 = (b.zeros >> i) & 1 == 0;

        let mut result_may_be_0 = false;
        let mut result_may_be_1 = false;
        let mut next_carry_may_be_0 = false;
        let mut next_carry_may_be_1 = false;

        // Enumerate all valid (a_val, b_val, carry_val) triples.
        // At most 8 iterations per bit position.
        for a_val in [false, true] {
            if !a_val && !a_may_be_0 { continue; }
            if a_val && !a_may_be_1 { continue; }
            for b_val in [false, true] {
                if !b_val && !b_may_be_0 { continue; }
                if b_val && !b_may_be_1 { continue; }
                for carry_val in [false, true] {
                    if !carry_val && !carry_may_be_0 { continue; }
                    if carry_val && !carry_may_be_1 { continue; }

                    let av = a_val as u64;
                    let bv = b_val as u64;
                    let cv = carry_val as u64;

                    let sum = av ^ bv ^ cv;
                    let carry_out = (av & bv) | (av & cv) | (bv & cv);

                    if sum == 0 { result_may_be_0 = true; }
                    if sum == 1 { result_may_be_1 = true; }
                    if carry_out == 0 { next_carry_may_be_0 = true; }
                    if carry_out == 1 { next_carry_may_be_1 = true; }
                }
            }
        }

        // A result bit is known-0 if it can ONLY be 0.
        // A result bit is known-1 if it can ONLY be 1.
        if !result_may_be_1 { zeros |= 1u64 << i; }
        if !result_may_be_0 { ones |= 1u64 << i; }

        carry_may_be_0 = next_carry_may_be_0;
        carry_may_be_1 = next_carry_may_be_1;
    }

    KnownBits { zeros, ones }
}

/// SUB: a - b = a + NOT(b) + 1. Implemented with a borrow-ripple simulation
/// analogous to the add carry-ripple, but using the subtraction truth table.
///
/// result_i = a_i XOR b_i XOR borrow_i
/// borrow_{i+1} = (!a_i & b_i) | (!a_i & borrow_i) | (b_i & borrow_i)
pub fn from_sub(a: &KnownBits, b: &KnownBits) -> KnownBits {
    let mut zeros = 0u64;
    let mut ones = 0u64;

    // Borrow-in to bit 0 is always 0.
    let mut borrow_may_be_0 = true;
    let mut borrow_may_be_1 = false;

    for i in 0..64 {
        let a_may_be_0 = (a.ones >> i) & 1 == 0;
        let a_may_be_1 = (a.zeros >> i) & 1 == 0;
        let b_may_be_0 = (b.ones >> i) & 1 == 0;
        let b_may_be_1 = (b.zeros >> i) & 1 == 0;

        let mut result_may_be_0 = false;
        let mut result_may_be_1 = false;
        let mut next_borrow_may_be_0 = false;
        let mut next_borrow_may_be_1 = false;

        for a_val in [false, true] {
            if !a_val && !a_may_be_0 { continue; }
            if a_val && !a_may_be_1 { continue; }
            for b_val in [false, true] {
                if !b_val && !b_may_be_0 { continue; }
                if b_val && !b_may_be_1 { continue; }
                for borrow_val in [false, true] {
                    if !borrow_val && !borrow_may_be_0 { continue; }
                    if borrow_val && !borrow_may_be_1 { continue; }

                    let av = a_val as u64;
                    let bv = b_val as u64;
                    let brv = borrow_val as u64;

                    // Subtraction: result = a - b - borrow
                    // Equivalent to: result_i = a_i XOR b_i XOR borrow_i
                    let diff = av ^ bv ^ brv;
                    // Borrow out: borrow_{i+1} = majority(!a_i, b_i, borrow_i)
                    let na = 1 - av; // NOT a_i
                    let borrow_out = (na & bv) | (na & brv) | (bv & brv);

                    if diff == 0 { result_may_be_0 = true; }
                    if diff == 1 { result_may_be_1 = true; }
                    if borrow_out == 0 { next_borrow_may_be_0 = true; }
                    if borrow_out == 1 { next_borrow_may_be_1 = true; }
                }
            }
        }

        if !result_may_be_1 { zeros |= 1u64 << i; }
        if !result_may_be_0 { ones |= 1u64 << i; }

        borrow_may_be_0 = next_borrow_may_be_0;
        borrow_may_be_1 = next_borrow_may_be_1;
    }

    KnownBits { zeros, ones }
}

/// MUL: conservative — only known-zero when either input has trailing zeros.
///
/// If a has at least K trailing known-zero bits and b has at least L, then
/// a * b has at least K + L trailing known-zero bits. Beyond that, we don't
/// try to track known bits through multiplication (it would require
/// enumerating cross-product partial products).
pub fn from_mul(a: &KnownBits, b: &KnownBits) -> KnownBits {
    let tz_a = min_trailing_zeros(a);
    let tz_b = min_trailing_zeros(b);
    let tz_result = tz_a + tz_b;

    if tz_result == 0 {
        return KnownBits::unknown();
    }

    // If tz_result >= 64, the entire result is known to be 0.
    if tz_result >= 64 {
        return KnownBits::constant(0);
    }

    let zeros = (1u64 << tz_result) - 1;

    KnownBits { zeros, ones: 0 }
}

/// SHL: shift zeros in from the right.
///
/// If the shift amount is known (say S), then:
///   - The low S bits of the result are known-0
///   - The remaining bits are the input bits shifted left by S
///
/// If the shift amount is unknown, we can only conservatively say: the bits
/// that are known-0 in the input might be shifted anywhere, so we can only
/// track that known-0 bits that are present regardless of shift amount
/// (which is none in the general case). However, if the shift amount is
/// known-zero (shift by 0), the result is just the input.
pub fn from_shl(a: &KnownBits, shift: &KnownBits) -> KnownBits {
    // If the shift amount is fully known, we can compute exact results.
    if let Some(s) = shift.known_value() {
        let s = s as u32;
        if s == 0 {
            return *a;
        }
        if s >= 64 {
            // Shifting left by 64+ yields 0 (for u64).
            return KnownBits::constant(0);
        }
        let low_zeros = (1u64 << s) - 1; // low S bits are 0
        return KnownBits {
            zeros: (a.zeros << s) | low_zeros,
            ones: a.ones << s,
        };
    }

    // If shift amount is unknown but we know some trailing zeros in a,
    // those zeros will persist in some form. But it's hard to make strong
    // claims without knowing the shift amount, so we return unknown.
    //
    // One useful special case: if a is known to be 0, the result is 0
    // regardless of shift.
    if a.known_value() == Some(0) {
        return KnownBits::constant(0);
    }

    KnownBits::unknown()
}

/// SHR: shift zeros in from the left (logical right shift).
///
/// If the shift amount is known (say S), then:
///   - The top S bits of the result are known-0
///   - The remaining bits are the input bits shifted right by S
pub fn from_shr(a: &KnownBits, shift: &KnownBits) -> KnownBits {
    if let Some(s) = shift.known_value() {
        let s = s as u32;
        if s == 0 {
            return *a;
        }
        if s >= 64 {
            // Logical right shift by 64+ yields 0.
            return KnownBits::constant(0);
        }
        let high_zeros = !0u64 << (64 - s); // top S bits are 0
        return KnownBits {
            zeros: (a.zeros >> s) | high_zeros,
            ones: a.ones >> s,
        };
    }

    if a.known_value() == Some(0) {
        return KnownBits::constant(0);
    }

    KnownBits::unknown()
}

// =============================================================================
// §4  Transfer Functions for Unary Operations
// =============================================================================

/// NOT: swap zeros and ones (bitwise complement).
pub fn from_not(a: &KnownBits) -> KnownBits {
    KnownBits {
        zeros: a.ones,
        ones: a.zeros,
    }
}

/// NEG: -(x) = NOT(x) + 1.
pub fn from_neg(a: &KnownBits) -> KnownBits {
    let not_a = from_not(a);
    let one = KnownBits::constant(1);
    from_add(&not_a, &one)
}

// =============================================================================
// §5  Comparison Pruning
// =============================================================================

/// Check if two KnownBits are provably different (at least one bit position
/// is known-1 in one and known-0 in the other). If so, the expressions
/// CANNOT be equivalent and we can skip SMT verification entirely.
pub fn provably_different(a: &KnownBits, b: &KnownBits) -> bool {
    // A bit position where a is known-1 and b is known-0, or vice versa.
    (a.ones & b.zeros) != 0 || (a.zeros & b.ones) != 0
}

// =============================================================================
// §6  Trailing Zeros Analysis
// =============================================================================

/// Count the minimum number of trailing zeros in a KnownBits value.
/// If the low K bits are all known-zero, returns K.
///
/// This is useful for shift/multiply optimization: if `x` has at least 3
/// trailing known-zero bits, then `x * 8` can be replaced with `x << 3`
/// (and we can prove the low 3 bits of the result are 0).
pub fn min_trailing_zeros(kb: &KnownBits) -> u32 {
    // The low K bits are all known-zero iff (zeros & ((1 << K) - 1)) == ((1 << K) - 1)
    // i.e., the low K bits of the zeros mask are all set.
    let known_zero_low = kb.zeros;
    if known_zero_low == 0 {
        return 0;
    }
    known_zero_low.trailing_ones()
}

// =============================================================================
// §7  Demanded-Bits Analysis (Reverse Direction)
// =============================================================================

/// Result of demanded-bits analysis: which bits of the input expression
/// actually affect the demanded output bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DemandedResult {
    /// Bits of the overall input that matter for the demanded output bits.
    pub src_demanded: u64,
}

/// Given an instruction and the demanded bits of its output, compute which
/// bits of its inputs are demanded.
///
/// This is a reverse (backward) analysis: starting from which output bits
/// the consumer cares about, we determine which input bits the producer
/// must compute correctly.
pub fn demanded_bits(instr: &Instr, demanded: u64) -> DemandedResult {
    if demanded == 0 {
        return DemandedResult { src_demanded: 0 };
    }

    match instr {
        Instr::ConstInt(_) | Instr::ConstFloat(_) | Instr::ConstBool(_) => {
            // Constants don't depend on any input.
            DemandedResult { src_demanded: 0 }
        }
        Instr::Var(_) => {
            // The variable's demanded bits are exactly the output demanded bits.
            DemandedResult { src_demanded: demanded }
        }
        Instr::BinOp { op, lhs, rhs } => {
            let lhs_dem = demanded_bits(lhs, demanded).src_demanded;
            let rhs_dem = demanded_bits(rhs, demanded).src_demanded;

            let combined = match op {
                // Bitwise operations: each output bit depends only on the
                // corresponding input bits.
                BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor => {
                    lhs_dem | rhs_dem
                }
                // Addition/subtraction: due to carry propagation, if output
                // bit i is demanded, all input bits 0..=i are demanded.
                BinOpKind::Add | BinOpKind::Sub => {
                    // If any bit at position i or above is demanded, we need
                    // all bits from 0 to i (carry can propagate upward).
                    let highest_demanded = 63 - demanded.leading_zeros();
                    let all_below = if highest_demanded >= 63 {
                        !0u64
                    } else {
                        (1u64 << (highest_demanded + 1)) - 1
                    };
                    // Also include the directly demanded bits for both operands.
                    all_below | lhs_dem | rhs_dem
                }
                // Multiplication: conservative — demand all bits of both operands.
                BinOpKind::Mul => !0u64,
                // Division/remainder: demand all bits of both operands.
                BinOpKind::Div | BinOpKind::Rem | BinOpKind::FloorDiv => !0u64,
                // Shifts: demand the shifted bits of the value and the shift amount.
                BinOpKind::Shl | BinOpKind::Shr => {
                    lhs_dem | rhs_dem
                }
                // Comparisons: demand all bits (result depends on all input bits).
                BinOpKind::Eq | BinOpKind::Ne
                | BinOpKind::Lt | BinOpKind::Le
                | BinOpKind::Gt | BinOpKind::Ge => !0u64,
                // Logical: conservative
                BinOpKind::And | BinOpKind::Or => !0u64,
            };
            DemandedResult { src_demanded: combined }
        }
        Instr::UnOp { op, operand } => {
            let op_dem = demanded_bits(operand, demanded).src_demanded;
            let result = match op {
                UnOpKind::Not => op_dem,
                UnOpKind::Neg => {
                    // NEG = NOT + 1, so carry can propagate.
                    // Demand all bits up to the highest demanded output bit.
                    let highest = 63 - demanded.leading_zeros();
                    if highest >= 63 {
                        !0u64
                    } else {
                        let mask = (1u64 << (highest + 1)) - 1;
                        op_dem | mask
                    }
                }
                UnOpKind::Deref | UnOpKind::Ref | UnOpKind::RefMut => demanded,
            };
            DemandedResult { src_demanded: result }
        }
    }
}

// =============================================================================
// §8  Analysis Driver
// =============================================================================

/// Compute known bits for an instruction tree, given variable bindings.
/// Variables not in the environment are treated as fully unknown.
///
/// This recursively computes KnownBits for the entire expression tree using
/// the transfer functions defined above.
pub fn analyze(instr: &Instr, env: &FxHashMap<String, KnownBits>) -> KnownBits {
    match instr {
        Instr::ConstInt(v) => {
            // Truncate to u64 for known-bits analysis.
            KnownBits::constant(*v as u64)
        }
        Instr::ConstFloat(_) => {
            // Floats don't have meaningful known-bits in this integer analysis.
            KnownBits::unknown()
        }
        Instr::ConstBool(b) => {
            KnownBits::constant(if *b { 1 } else { 0 })
        }
        Instr::Var(name) => {
            env.get(name).copied().unwrap_or_else(KnownBits::unknown)
        }
        Instr::BinOp { op, lhs, rhs } => {
            let a = analyze(lhs, env);
            let b = analyze(rhs, env);
            analyze_binop(*op, &a, &b)
        }
        Instr::UnOp { op, operand } => {
            let a = analyze(operand, env);
            analyze_unop(*op, &a)
        }
    }
}

/// Apply the appropriate transfer function for a binary operation.
fn analyze_binop(op: BinOpKind, a: &KnownBits, b: &KnownBits) -> KnownBits {
    match op {
        BinOpKind::Add => from_add(a, b),
        BinOpKind::Sub => from_sub(a, b),
        BinOpKind::Mul => from_mul(a, b),
        BinOpKind::BitAnd => from_and(a, b),
        BinOpKind::BitOr => from_or(a, b),
        BinOpKind::BitXor => from_xor(a, b),
        BinOpKind::Shl => from_shl(a, b),
        BinOpKind::Shr => from_shr(a, b),
        // Division, remainder, comparisons, and logical ops are too complex
        // for known-bits analysis. Return unknown conservatively.
        BinOpKind::Div
        | BinOpKind::Rem
        | BinOpKind::FloorDiv
        | BinOpKind::Eq
        | BinOpKind::Ne
        | BinOpKind::Lt
        | BinOpKind::Le
        | BinOpKind::Gt
        | BinOpKind::Ge
        | BinOpKind::And
        | BinOpKind::Or => KnownBits::unknown(),
    }
}

/// Apply the appropriate transfer function for a unary operation.
fn analyze_unop(op: UnOpKind, a: &KnownBits) -> KnownBits {
    match op {
        UnOpKind::Neg => from_neg(a),
        UnOpKind::Not => from_not(a),
        UnOpKind::Deref | UnOpKind::Ref | UnOpKind::RefMut => KnownBits::unknown(),
    }
}

// =============================================================================
// §9  Equality Pre-Check
// =============================================================================

/// Quick pre-check: can these two expressions possibly be equivalent?
///
/// Returns `false` if known-bits analysis proves they differ on at least one
/// known bit. Returns `true` if they MIGHT be equivalent (unknown — need SMT
/// to confirm).
///
/// This is called BEFORE the expensive SMT verification step. It rejects
/// many candidates in nanoseconds without ever invoking Z3.
pub fn may_be_equivalent(
    src: &Instr,
    candidate: &Instr,
    env: &FxHashMap<String, KnownBits>,
) -> bool {
    let src_bits = analyze(src, env);
    let candidate_bits = analyze(candidate, env);

    // If the known bits are provably different, they cannot be equivalent.
    !provably_different(&src_bits, &candidate_bits)
}

// =============================================================================
// §10  Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── from_and with concrete values ────────────────────────────────────

    #[test]
    fn test_from_and_concrete() {
        let a = KnownBits::constant(0xFF00); // 0000_0000_0000_0000_1111_1111_0000_0000
        let b = KnownBits::constant(0x0FF0); // 0000_0000_0000_0000_0000_1111_1111_0000
        let result = from_and(&a, &b);
        assert_eq!(result.known_value(), Some(0x0F00));
    }

    #[test]
    fn test_from_and_partial() {
        // a: low 8 bits unknown, high 8 bits known-0
        let a = KnownBits { zeros: 0xFF00, ones: 0x0000 };
        // b: low 4 bits known-1, next 4 bits known-0, high 8 bits unknown
        let b = KnownBits { zeros: 0x00F0, ones: 0x000F };
        let result = from_and(&a, &b);
        // AND with a having unknown low 8 bits:
        // bits 0-3: unknown & known-1 = unknown
        // bits 4-7: unknown & known-0 = known-0
        // bits 8-15: known-0 & unknown = known-0
        assert!(result.is_unknown(0));
        assert!(result.is_unknown(1));
        assert!(result.is_known_zero(4));
        assert!(result.is_known_zero(7));
        assert!(result.is_known_zero(8));
        assert!(result.is_known_zero(15));
    }

    // ── from_add ripple carry: prove low bits of (x << 3) + y = low bits of y ──

    #[test]
    fn test_from_add_ripple_carry_shift3() {
        // x << 3 has low 3 bits known-0
        let x_shifted = KnownBits {
            zeros: 0b111,  // low 3 bits known-0
            ones: 0,       // upper bits unknown
        };
        // y is fully unknown
        let y = KnownBits::unknown();

        let result = from_add(&x_shifted, &y);

        // The low 3 bits of (x<<3 + y) should equal the low 3 bits of y.
        // Since y is unknown, the low 3 bits of the result should also be unknown.
        // But the KEY property is: the carry from bits 0-2 of y cannot affect
        // bit 3 of x_shifted (since x_shifted's bits 0-2 are 0, there's nothing
        // to carry into). Let's verify the low 3 bits are unknown (matching y).
        assert!(result.is_unknown(0));
        assert!(result.is_unknown(1));
        assert!(result.is_unknown(2));
    }

    #[test]
    fn test_from_add_known_shift_and_partial_y() {
        // x << 3: low 3 bits = 0
        let x_shifted = KnownBits {
            zeros: 0b111,
            ones: 0,
        };
        // y with bit 0 known-1
        let y = KnownBits {
            zeros: 0,
            ones: 0b001,
        };

        let result = from_add(&x_shifted, &y);
        // bit 0 of (x<<3 + y): 0 + 1 + carry(0) = 1, no carry
        assert!(result.is_known_one(0));
    }

    #[test]
    fn test_from_add_concrete() {
        let a = KnownBits::constant(10);
        let b = KnownBits::constant(20);
        let result = from_add(&a, &b);
        assert_eq!(result.known_value(), Some(30));
    }

    #[test]
    fn test_from_add_with_carry() {
        // a: bit 0 known-1, rest unknown
        let a = KnownBits { zeros: 0, ones: 1 };
        // b: bit 0 known-1, rest unknown
        let b = KnownBits { zeros: 0, ones: 1 };
        let result = from_add(&a, &b);
        // bit 0: 1 + 1 + 0 = 0, carry 1
        assert!(result.is_known_zero(0));
        // bit 1: unknown + unknown + carry(1) = unknown
        // (carry might be 0 or 1 depending on the unknown bits)
        // Actually: carry-in to bit 1 is always 1 (since bit 0 always produces carry).
        // Wait, carry from bit 0: 1+1+0 = 10 binary, so carry out = 1 always.
        // So carry into bit 1 is always 1.
        // But a_1 and b_1 are unknown. So result_1 = a_1 XOR b_ XOR 1.
        // a_1=0, b_1=0 → 0^0^1 = 1
        // a_1=0, b_1=1 → 0^1^1 = 0
        // a_1=1, b_1=0 → 1^0^1 = 0
        // a_1=1, b_1=1 → 1^1^1 = 1
        // So result_1 is unknown.
        assert!(result.is_unknown(1));
    }

    // ── provably_different catches obvious differences ──────────────────

    #[test]
    fn test_provably_different_constants() {
        let a = KnownBits::constant(42);
        let b = KnownBits::constant(99);
        assert!(provably_different(&a, &b));
    }

    #[test]
    fn test_provably_different_partial() {
        // a: bit 0 known-1
        let a = KnownBits { zeros: 0, ones: 1 };
        // b: bit 0 known-0
        let b = KnownBits { zeros: 1, ones: 0 };
        assert!(provably_different(&a, &b));
    }

    #[test]
    fn test_not_provably_different_unknowns() {
        let a = KnownBits::unknown();
        let b = KnownBits::unknown();
        assert!(!provably_different(&a, &b));
    }

    #[test]
    fn test_not_provably_different_same() {
        let a = KnownBits::constant(42);
        assert!(!provably_different(&a, &a));
    }

    // ── may_be_equivalent for x*2 vs x<<1 ──────────────────────────────

    #[test]
    fn test_may_be_equivalent_x2_vs_xshl1() {
        let mut env: FxHashMap<String, KnownBits> = FxHashMap::default();
        // x is fully unknown
        env.insert("x".to_string(), KnownBits::unknown());

        // x * 2
        let x_mul_2 = Instr::BinOp {
            op: BinOpKind::Mul,
            lhs: Box::new(Instr::Var("x".to_string())),
            rhs: Box::new(Instr::ConstInt(2)),
        };

        // x << 1
        let x_shl_1 = Instr::BinOp {
            op: BinOpKind::Shl,
            lhs: Box::new(Instr::Var("x".to_string())),
            rhs: Box::new(Instr::ConstInt(1)),
        };

        // Both should have the same known bits: bit 0 is known-0, rest unknown.
        let mul_bits = analyze(&x_mul_2, &env);
        let shl_bits = analyze(&x_shl_1, &env);

        // x * 2: min_trailing_zeros(x) + min_trailing_zeros(2) = 0 + 1 = 1
        // So bit 0 should be known-0
        assert!(mul_bits.is_known_zero(0));
        // x << 1: low 1 bit is 0
        assert!(shl_bits.is_known_zero(0));

        // They should not be provably different
        assert!(!provably_different(&mul_bits, &shl_bits));

        // may_be_equivalent should return true
        assert!(may_be_equivalent(&x_mul_2, &x_shl_1, &env));
    }

    #[test]
    fn test_may_be_equivalent_different_constants() {
        let env: FxHashMap<String, KnownBits> = FxHashMap::default();

        let a = Instr::ConstInt(42);
        let b = Instr::ConstInt(99);

        assert!(!may_be_equivalent(&a, &b, &env));
    }

    #[test]
    fn test_may_be_equivalent_same_constants() {
        let env: FxHashMap<String, KnownBits> = FxHashMap::default();

        let a = Instr::ConstInt(42);
        let b = Instr::ConstInt(42);

        assert!(may_be_equivalent(&a, &b, &env));
    }

    // ── min_trailing_zeros counts correctly ─────────────────────────────

    #[test]
    fn test_min_trailing_zeros_unknown() {
        let kb = KnownBits::unknown();
        assert_eq!(min_trailing_zeros(&kb), 0);
    }

    #[test]
    fn test_min_trailing_zeros_constant() {
        assert_eq!(min_trailing_zeros(&KnownBits::constant(0)), 64);
        assert_eq!(min_trailing_zeros(&KnownBits::constant(1)), 0);
        assert_eq!(min_trailing_zeros(&KnownBits::constant(2)), 1);
        assert_eq!(min_trailing_zeros(&KnownBits::constant(4)), 2);
        assert_eq!(min_trailing_zeros(&KnownBits::constant(8)), 3);
        assert_eq!(min_trailing_zeros(&KnownBits::constant(16)), 4);
    }

    #[test]
    fn test_min_trailing_zeros_partial() {
        // Low 3 bits known-0, rest unknown
        let kb = KnownBits { zeros: 0b111, ones: 0 };
        assert_eq!(min_trailing_zeros(&kb), 3);
    }

    #[test]
    fn test_min_trailing_zeros_mixed() {
        // Low 2 bits known-0, bit 2 known-1
        let kb = KnownBits { zeros: 0b011, ones: 0b100 };
        assert_eq!(min_trailing_zeros(&kb), 2);
    }

    // ── constant round-trips through known_value ────────────────────────

    #[test]
    fn test_constant_roundtrip() {
        for val in [0u64, 1, 42, 255, 256, 0xFFFF, u32::MAX as u64, u64::MAX] {
            let kb = KnownBits::constant(val);
            assert_eq!(kb.known_value(), Some(val), "roundtrip failed for {val}");
        }
    }

    #[test]
    fn test_constant_zeros_ones_invariant() {
        for val in [0u64, 1, 42, 255, 0xABCD, u64::MAX] {
            let kb = KnownBits::constant(val);
            assert_eq!(kb.zeros & kb.ones, 0, "invariant violated for {val}");
        }
    }

    // ── from_neg of known constant gives correct result ─────────────────

    #[test]
    fn test_from_neg_zero() {
        let zero = KnownBits::constant(0);
        let neg_zero = from_neg(&zero);
        // -0 = 0 in two's complement
        assert_eq!(neg_zero.known_value(), Some(0));
    }

    #[test]
    fn test_from_neg_one() {
        let one = KnownBits::constant(1);
        let neg_one = from_neg(&one);
        // -1 = NOT(1) + 1 = 0xFFFFFFFFFFFFFFFE + 1 = 0xFFFFFFFFFFFFFFFF
        assert_eq!(neg_one.known_value(), Some(!0u64));
    }

    #[test]
    fn test_from_neg_arbitrary() {
        let val = KnownBits::constant(42);
        let neg_val = from_neg(&val);
        // -42 in two's complement
        let expected = (0u64).wrapping_sub(42);
        assert_eq!(neg_val.known_value(), Some(expected));
    }

    #[test]
    fn test_from_neg_max() {
        let max = KnownBits::constant(u64::MAX);
        let neg_max = from_neg(&max);
        // -MAX = NOT(MAX) + 1 = 0 + 1 = 1
        assert_eq!(neg_max.known_value(), Some(1));
    }

    // ── Additional transfer function tests ──────────────────────────────

    #[test]
    fn test_from_or_concrete() {
        let a = KnownBits::constant(0xFF00);
        let b = KnownBits::constant(0x0FF0);
        let result = from_or(&a, &b);
        assert_eq!(result.known_value(), Some(0xFFF0));
    }

    #[test]
    fn test_from_xor_concrete() {
        let a = KnownBits::constant(0xFF00);
        let b = KnownBits::constant(0x0FF0);
        let result = from_xor(&a, &b);
        assert_eq!(result.known_value(), Some(0xF0F0));
    }

    #[test]
    fn test_from_sub_concrete() {
        let a = KnownBits::constant(100);
        let b = KnownBits::constant(30);
        let result = from_sub(&a, &b);
        assert_eq!(result.known_value(), Some(70));
    }

    #[test]
    fn test_from_not_swaps() {
        let val = KnownBits::constant(0xABCD);
        let not_val = from_not(&val);
        assert_eq!(not_val.known_value(), Some(!0xABCDu64));
    }

    #[test]
    fn test_from_mul_trailing_zeros() {
        // a: low 2 bits known-0
        let a = KnownBits { zeros: 0b11, ones: 0 };
        // b: low 3 bits known-0
        let b = KnownBits { zeros: 0b111, ones: 0 };
        let result = from_mul(&a, &b);
        // Should have at least 2 + 3 = 5 trailing known-zero bits
        assert!(result.is_known_zero(0));
        assert!(result.is_known_zero(1));
        assert!(result.is_known_zero(2));
        assert!(result.is_known_zero(3));
        assert!(result.is_known_zero(4));
    }

    #[test]
    fn test_from_shl_known_shift() {
        let a = KnownBits::constant(0xFF);
        let shift = KnownBits::constant(4);
        let result = from_shl(&a, &shift);
        assert_eq!(result.known_value(), Some(0xFF0));
    }

    #[test]
    fn test_from_shr_known_shift() {
        let a = KnownBits::constant(0xFF0);
        let shift = KnownBits::constant(4);
        let result = from_shr(&a, &shift);
        assert_eq!(result.known_value(), Some(0xFF));
    }

    #[test]
    fn test_from_add_all_zeros() {
        let a = KnownBits::constant(0);
        let b = KnownBits::constant(0);
        let result = from_add(&a, &b);
        assert_eq!(result.known_value(), Some(0));
    }

    #[test]
    fn test_invariant_preserved_by_transfer_functions() {
        // Verify that all transfer functions preserve the zeros & ones == 0 invariant
        let test_values = [
            KnownBits::unknown(),
            KnownBits::constant(0),
            KnownBits::constant(1),
            KnownBits::constant(u64::MAX),
            KnownBits::constant(0xAAAAAAAAAAAAAAAA),
            KnownBits { zeros: 0xFF00, ones: 0x00FF },
            KnownBits { zeros: 0x0F0F, ones: 0xF0F0 },
        ];

        for a in &test_values {
            assert_eq!(a.zeros & a.ones, 0, "input invariant violated");
            for b in &test_values {
                let and_result = from_and(a, b);
                assert_eq!(and_result.zeros & and_result.ones, 0, "AND invariant violated");

                let or_result = from_or(a, b);
                assert_eq!(or_result.zeros & or_result.ones, 0, "OR invariant violated");

                let xor_result = from_xor(a, b);
                assert_eq!(xor_result.zeros & xor_result.ones, 0, "XOR invariant violated");

                let add_result = from_add(a, b);
                assert_eq!(add_result.zeros & add_result.ones, 0, "ADD invariant violated");

                let sub_result = from_sub(a, b);
                assert_eq!(sub_result.zeros & sub_result.ones, 0, "SUB invariant violated");

                let mul_result = from_mul(a, b);
                assert_eq!(mul_result.zeros & mul_result.ones, 0, "MUL invariant violated");

                let not_result = from_not(a);
                assert_eq!(not_result.zeros & not_result.ones, 0, "NOT invariant violated");

                let neg_result = from_neg(a);
                assert_eq!(neg_result.zeros & neg_result.ones, 0, "NEG invariant violated");
            }
        }
    }

    #[test]
    fn test_intersects_method() {
        let a = KnownBits::constant(42);
        let b = KnownBits::constant(42);
        assert!(!a.intersects(&b)); // same value, no contradiction

        let c = KnownBits::constant(99);
        assert!(a.intersects(&c)); // different values, contradiction exists

        let unknown = KnownBits::unknown();
        assert!(!unknown.intersects(&a)); // unknown doesn't contradict anything
    }

    #[test]
    fn test_is_known_bit_queries() {
        let kb = KnownBits { zeros: 0b1010, ones: 0b0101 };
        // bit 0: known-1 (ones bit set)
        assert_eq!(kb.is_known(0), Some(true));
        assert!(kb.is_known_one(0));
        assert!(!kb.is_known_zero(0));
        assert!(!kb.is_unknown(0));

        // bit 1: known-0 (zeros bit set)
        assert_eq!(kb.is_known(1), Some(false));
        assert!(kb.is_known_zero(1));
        assert!(!kb.is_known_one(1));
        assert!(!kb.is_unknown(1));

        // bits 2-3: both set (invalid state, but shouldn't crash)
        // bits 4+: unknown
        assert_eq!(kb.is_known(4), None);
        assert!(!kb.is_known_zero(4));
        assert!(!kb.is_known_one(4));
        assert!(kb.is_unknown(4));
    }

    #[test]
    fn test_known_value_partial() {
        // Only some bits known
        let kb = KnownBits { zeros: 0x0F, ones: 0xF0 };
        // zeros | ones = 0xFF, not all 64 bits known
        assert_eq!(kb.known_value(), None);
    }
}
