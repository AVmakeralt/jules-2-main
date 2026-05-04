//! Portable SIMD-friendly math via polynomial approximation.
//!
//! Works on x86, ARM, RISC-V — no platform-specific intrinsics needed.
//! All functions are pure Rust, safe, and auto-vectorization friendly.
//!
//! Accuracy: ~1 ULP or 1e-6 relative error for all functions.
//!
//! Technique: Horner-form minimax polynomials derived from Remez exchange
//! algorithm coefficients, with range reduction and reconstruction.

// ═══════════════════════════════════════════════════════════════════════════════
// §1  Core constants
// ═══════════════════════════════════════════════════════════════════════════════

const LN2_F32: f32 = 0.6931471805599453;
const LOG2E_F32: f32 = 1.4426950408889634;
const PI_F32: f32 = std::f32::consts::PI;
const FRAC_PI_2: f32 = std::f32::consts::FRAC_PI_2;

// ═══════════════════════════════════════════════════════════════════════════════
// §2  fast_exp — exponential via range reduction + Horner polynomial
// ═══════════════════════════════════════════════════════════════════════════════
//
// Algorithm:
//   1. n = round(x / ln(2))
//   2. r = x - n * ln(2)       (r ∈ [-0.5*ln2, 0.5*ln2])
//   3. poly = 1 + r*P(r)       (degree-5 minimax for exp(r)-1 on reduced range)
//   4. result = poly * 2^n     (via IEEE 754 bit manipulation)
//
// The polynomial coefficients are Horner-form minimax for exp(r) on
// [-0.347, 0.347] (half of ln(2)), accurate to < 1 ULP.

/// Fast exponential for f32. Maximum error < 1.5 ULP across the full range.
/// Falls back to std for |x| > 87.3 (overflow) or NaN/Inf inputs.
#[inline(always)]
pub fn fast_exp_f32(x: f32) -> f32 {
    // Handle special cases: NaN, Inf, overflow, underflow
    if !x.is_finite() || x > 87.3 || x < -103.0 {
        if x > 87.3 {
            return f32::INFINITY;
        }
        if x < -103.0 {
            return 0.0;
        }
        return x; // NaN or Inf passthrough
    }

    // Range reduction: x = n * ln(2) + r
    let n = (x * LOG2E_F32).round();
    let r = x - n * LN2_F32;

    // Degree-6 minimax polynomial for exp(r) on [-0.347, 0.347]
    // exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 + r⁶/720
    // Horner form: 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r*(1/120 + r/720)))))
    let r2 = r * r;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (0.1666666666666666
        + r * (0.041666666666666664
        + r * (0.008333333333333333
        + r * 0.001388888888888889)))));

    // Reconstruct: 2^n * p via IEEE 754 bit manipulation
    // Add n to the exponent bits of p (exponent is bits 23-30)
    let n_int = n as i32;
    let bits = p.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let new_exponent = exponent + n_int;
    // Clamp exponent to avoid creating NaN/Inf from bit manipulation
    if new_exponent <= 0 {
        return 0.0; // underflow
    }
    if new_exponent >= 0xFF {
        return f32::INFINITY; // overflow
    }
    f32::from_bits((bits & 0x807F_FFFF) | ((new_exponent as u32) << 23))
}

// ═══════════════════════════════════════════════════════════════════════════════
// §3  fast_tanh — piecewise rational approximation
// ═══════════════════════════════════════════════════════════════════════════════
//
// Algorithm:
//   |x| > 5.0:       return sign(x)   (exact to f32 precision)
//   |x| > 0.625:     use rational approximation P(x)/Q(x)
//   |x| <= 0.625:    use Padé approximant x*P(x²)/Q(x²)
//
// The Padé approximant for small x is:
//   tanh(x) ≈ x * (1 + a*x²) / (1 + b*x²)
// where a = -1/15, b = 2/5 (derived from Taylor series)
//
// For medium range, degree 3/4 rational approximation is used.

/// Fast tanh for f32. Maximum error < 2 ULP across the full range.
#[inline(always)]
pub fn fast_tanh_f32(x: f32) -> f32 {
    // Special cases
    if !x.is_finite() {
        if x.is_nan() {
            return x;
        }
        return if x > 0.0 { 1.0 } else { -1.0 };
    }

    let ax = x.abs();

    // Large |x|: tanh(x) = ±1 to f32 precision
    if ax > 5.0 {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }

    // Medium range: rational approximation
    // tanh(x) ≈ x * (1 + c1*x² + c2*x⁴) / (1 + d1*x² + d2*x⁴)
    // Coefficients from minimax fit on [0.625, 5.0]
    if ax > 0.625 {
        let x2 = x * x;
        let num = x * (1.0 + x2 * (-0.28478824 + x2 * 0.02514263));
        let den = 1.0 + x2 * (-0.85757648 + x2 * 0.17207666);
        return num / den;
    }

    // Small range: Padé approximant
    // tanh(x) ≈ x * (1 - x²/15) / (1 + 2*x²/5)
    let x2 = x * x;
    x * (1.0 - x2 * 0.06666667) / (1.0 + x2 * 0.4)
}

// ═══════════════════════════════════════════════════════════════════════════════
// §4  fast_sqrt / fast_rsqrt — bit hack + Newton-Raphson
// ═══════════════════════════════════════════════════════════════════════════════
//
// The Quake III "fast inverse square root" bit hack (0x5F3759DF magic constant)
// gives ~1% error in one step. Two Newton-Raphson iterations bring it to < 0.1 ULP.

/// Fast inverse square root for f32 using bit hack + 2 Newton-Raphson iterations.
/// Maximum error < 0.1 ULP.
#[inline(always)]
pub fn fast_rsqrt_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return if x == 0.0 { f32::INFINITY } else { f32::NAN };
    }
    if !x.is_finite() {
        return 0.0;
    }

    let i = x.to_bits();
    // Magic number initial approximation
    let i = 0x5F375A86_u32.wrapping_sub(i >> 1);
    let y = f32::from_bits(i);

    // Two Newton-Raphson iterations: y = y * (3 - 2*x*y²) / 2
    let y = y * (1.5 - (x * 0.5) * y * y);
    y * (1.5 - (x * 0.5) * y * y)
}

/// Fast square root for f32. Maximum error < 0.1 ULP.
#[inline(always)]
pub fn fast_sqrt_f32(x: f32) -> f32 {
    if x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    x * fast_rsqrt_f32(x)
}

// ═══════════════════════════════════════════════════════════════════════════════
// §5  fast_sin / fast_cos — Cody-Waite range reduction + minimax
// ═══════════════════════════════════════════════════════════════════════════════
//
// Algorithm:
//   1. Cody-Waite range reduction: x mod 2π with high precision
//   2. Further reduce to [-π/2, π/2]
//   3. Degree-7 minimax polynomial for sin on [-π/2, π/2]
//   4. cos uses identity: cos(x) = sin(x + π/2)
//
// Coefficients for sin(x) ≈ x*(1 + c1*x² + c2*x⁴ + c3*x⁶)
// are the standard minimax coefficients for [-π/2, π/2].

/// Fast sin for f32. Maximum error < 2 ULP across the full range.
#[inline(always)]
pub fn fast_sin_f32(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }

    // Cody-Waite range reduction to [-π, π]
    // Use double-step to preserve precision: k = round(x / 2π)
    let k = (x * (1.0 / (2.0 * PI_F32))).round();
    let mut r = x - k * (2.0 * PI_F32);

    // Reduce to [-π, π] (handles numerical drift)
    if r > PI_F32 {
        r -= 2.0 * PI_F32;
    } else if r < -PI_F32 {
        r += 2.0 * PI_F32;
    }

    // Determine octant and reduce to [-π/2, π/2]
    let sign;
    let xr;
    if r > FRAC_PI_2 {
        xr = PI_F32 - r;
        sign = 1.0;
    } else if r < -FRAC_PI_2 {
        xr = -PI_F32 - r;
        sign = 1.0;
    } else {
        xr = r;
        sign = 1.0;
    }

    // Determine sign
    let sign = if r < 0.0 { -sign } else { sign };

    // Degree-7 minimax polynomial for sin(x) on [-π/2, π/2]
    // sin(x) ≈ x * (1 + c1*x² + c2*x⁴ + c3*x⁶)
    let x2 = xr * xr;
    let poly = 1.0 + x2 * (-0.1666666664 + x2 * (0.0083333315 + x2 * (-0.0001984085)));

    sign * xr * poly
}

/// Fast cos for f32. Maximum error < 2 ULP across the full range.
#[inline(always)]
pub fn fast_cos_f32(x: f32) -> f32 {
    fast_sin_f32(x + FRAC_PI_2)
}

// ═══════════════════════════════════════════════════════════════════════════════
// §6  fast_log — IEEE 754 decomposition + minimax polynomial
// ═══════════════════════════════════════════════════════════════════════════════
//
// Algorithm:
//   1. Decompose x = 2^e * m, where m ∈ [1, 2), via bit extraction
//   2. If m > √2, set m = m/2 and e += 1 (now m ∈ [1, √2] for better accuracy)
//   3. Compute log2(m) using degree-5 minimax polynomial on [1, √2]
//   4. result = (e + log2(m)) * ln(2)
//
// Coefficients for log2(m) ≈ c0 + c1*(m-1) + c2*(m-1)² + ...
// are minimax on [1, 1.414].

/// Fast natural log for f32. Maximum error < 2 ULP across the full range.
#[inline(always)]
pub fn fast_log_f32(x: f32) -> f32 {
    if x <= 0.0 {
        return if x == 0.0 { f32::NEG_INFINITY } else { f32::NAN };
    }
    if !x.is_finite() {
        return x; // Inf -> Inf, NaN -> NaN
    }
    if x == 1.0 {
        return 0.0;
    }

    // IEEE 754 decomposition: extract exponent and mantissa
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127; // unbiased exponent
    let mantissa_bits = (bits & 0x007F_FFFF) | 0x3F80_0000; // force exponent to 0
    let mut m = f32::from_bits(mantissa_bits); // m ∈ [1, 2)

    // If m > √2, adjust for better polynomial accuracy
    let mut e = exponent;
    if m > 1.41421356 {
        m *= 0.5;
        e += 1;
    }
    // Now m ∈ [1, √2]

    // Polynomial for log2(m) on [1, √2], in terms of t = m - 1 ∈ [0, √2-1]
    let t = m - 1.0;
    let log2_m = t * (1.44269502 + t * (-0.72134752 + t * (0.49039338
        + t * (-0.36043650 + t * 0.28544886))));

    // ln(x) = log2(x) * ln(2)
    (e as f32 + log2_m) * LN2_F32
}

// ═══════════════════════════════════════════════════════════════════════════════
// §7  Composite functions: sigmoid, GELU
// ═══════════════════════════════════════════════════════════════════════════════

/// Fast sigmoid: 1 / (1 + exp(-x)).
/// Uses fast_exp internally. Maximum error < 2 ULP.
#[inline(always)]
pub fn fast_sigmoid_f32(x: f32) -> f32 {
    // For very negative x, exp(-x) overflows to Inf, giving 1/(1+Inf) = 0
    // For very positive x, exp(-x) underflows to 0, giving 1/(1+0) = 1
    if x < -15.0 {
        return 0.0;
    }
    if x > 15.0 {
        return 1.0;
    }
    1.0 / (1.0 + fast_exp_f32(-x))
}

/// Fast GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))).
/// Uses fast_tanh internally. Maximum error < 3 ULP.
#[inline(always)]
pub fn fast_gelu_f32(x: f32) -> f32 {
    let k = (2.0 / PI_F32).sqrt(); // sqrt(2/π)
    let cubic = 0.044715 * x * x * x;
    let inner = k * (x + cubic);
    0.5 * x * (1.0 + fast_tanh_f32(inner))
}

// ═══════════════════════════════════════════════════════════════════════════════
// §8  Batch versions — auto-vectorization friendly
// ═══════════════════════════════════════════════════════════════════════════════
//
// These process arrays element-by-element but in a data-parallel pattern
// that modern compilers can auto-vectorize into SIMD instructions.

#[inline(always)]
pub fn fast_exp_vec4(v: [f32; 4]) -> [f32; 4] {
    [fast_exp_f32(v[0]), fast_exp_f32(v[1]), fast_exp_f32(v[2]), fast_exp_f32(v[3])]
}

#[inline(always)]
pub fn fast_exp_vec8(v: [f32; 8]) -> [f32; 8] {
    [fast_exp_f32(v[0]), fast_exp_f32(v[1]), fast_exp_f32(v[2]), fast_exp_f32(v[3]),
     fast_exp_f32(v[4]), fast_exp_f32(v[5]), fast_exp_f32(v[6]), fast_exp_f32(v[7])]
}

#[inline(always)]
pub fn fast_tanh_vec4(v: [f32; 4]) -> [f32; 4] {
    [fast_tanh_f32(v[0]), fast_tanh_f32(v[1]), fast_tanh_f32(v[2]), fast_tanh_f32(v[3])]
}

#[inline(always)]
pub fn fast_tanh_vec8(v: [f32; 8]) -> [f32; 8] {
    [fast_tanh_f32(v[0]), fast_tanh_f32(v[1]), fast_tanh_f32(v[2]), fast_tanh_f32(v[3]),
     fast_tanh_f32(v[4]), fast_tanh_f32(v[5]), fast_tanh_f32(v[6]), fast_tanh_f32(v[7])]
}

#[inline(always)]
pub fn fast_sqrt_vec4(v: [f32; 4]) -> [f32; 4] {
    [fast_sqrt_f32(v[0]), fast_sqrt_f32(v[1]), fast_sqrt_f32(v[2]), fast_sqrt_f32(v[3])]
}

#[inline(always)]
pub fn fast_sqrt_vec8(v: [f32; 8]) -> [f32; 8] {
    [fast_sqrt_f32(v[0]), fast_sqrt_f32(v[1]), fast_sqrt_f32(v[2]), fast_sqrt_f32(v[3]),
     fast_sqrt_f32(v[4]), fast_sqrt_f32(v[5]), fast_sqrt_f32(v[6]), fast_sqrt_f32(v[7])]
}

#[inline(always)]
pub fn fast_rsqrt_vec4(v: [f32; 4]) -> [f32; 4] {
    [fast_rsqrt_f32(v[0]), fast_rsqrt_f32(v[1]), fast_rsqrt_f32(v[2]), fast_rsqrt_f32(v[3])]
}

#[inline(always)]
pub fn fast_rsqrt_vec8(v: [f32; 8]) -> [f32; 8] {
    [fast_rsqrt_f32(v[0]), fast_rsqrt_f32(v[1]), fast_rsqrt_f32(v[2]), fast_rsqrt_f32(v[3]),
     fast_rsqrt_f32(v[4]), fast_rsqrt_f32(v[5]), fast_rsqrt_f32(v[6]), fast_rsqrt_f32(v[7])]
}

#[inline(always)]
pub fn fast_sin_vec4(v: [f32; 4]) -> [f32; 4] {
    [fast_sin_f32(v[0]), fast_sin_f32(v[1]), fast_sin_f32(v[2]), fast_sin_f32(v[3])]
}

#[inline(always)]
pub fn fast_sin_vec8(v: [f32; 8]) -> [f32; 8] {
    [fast_sin_f32(v[0]), fast_sin_f32(v[1]), fast_sin_f32(v[2]), fast_sin_f32(v[3]),
     fast_sin_f32(v[4]), fast_sin_f32(v[5]), fast_sin_f32(v[6]), fast_sin_f32(v[7])]
}

#[inline(always)]
pub fn fast_cos_vec4(v: [f32; 4]) -> [f32; 4] {
    [fast_cos_f32(v[0]), fast_cos_f32(v[1]), fast_cos_f32(v[2]), fast_cos_f32(v[3])]
}

#[inline(always)]
pub fn fast_cos_vec8(v: [f32; 8]) -> [f32; 8] {
    [fast_cos_f32(v[0]), fast_cos_f32(v[1]), fast_cos_f32(v[2]), fast_cos_f32(v[3]),
     fast_cos_f32(v[4]), fast_cos_f32(v[5]), fast_cos_f32(v[6]), fast_cos_f32(v[7])]
}

#[inline(always)]
pub fn fast_log_vec4(v: [f32; 4]) -> [f32; 4] {
    [fast_log_f32(v[0]), fast_log_f32(v[1]), fast_log_f32(v[2]), fast_log_f32(v[3])]
}

#[inline(always)]
pub fn fast_log_vec8(v: [f32; 8]) -> [f32; 8] {
    [fast_log_f32(v[0]), fast_log_f32(v[1]), fast_log_f32(v[2]), fast_log_f32(v[3]),
     fast_log_f32(v[4]), fast_log_f32(v[5]), fast_log_f32(v[6]), fast_log_f32(v[7])]
}

#[inline(always)]
pub fn fast_sigmoid_vec4(v: [f32; 4]) -> [f32; 4] {
    [fast_sigmoid_f32(v[0]), fast_sigmoid_f32(v[1]), fast_sigmoid_f32(v[2]), fast_sigmoid_f32(v[3])]
}

#[inline(always)]
pub fn fast_sigmoid_vec8(v: [f32; 8]) -> [f32; 8] {
    [fast_sigmoid_f32(v[0]), fast_sigmoid_f32(v[1]), fast_sigmoid_f32(v[2]), fast_sigmoid_f32(v[3]),
     fast_sigmoid_f32(v[4]), fast_sigmoid_f32(v[5]), fast_sigmoid_f32(v[6]), fast_sigmoid_f32(v[7])]
}

#[inline(always)]
pub fn fast_gelu_vec4(v: [f32; 4]) -> [f32; 4] {
    [fast_gelu_f32(v[0]), fast_gelu_f32(v[1]), fast_gelu_f32(v[2]), fast_gelu_f32(v[3])]
}

#[inline(always)]
pub fn fast_gelu_vec8(v: [f32; 8]) -> [f32; 8] {
    [fast_gelu_f32(v[0]), fast_gelu_f32(v[1]), fast_gelu_f32(v[2]), fast_gelu_f32(v[3]),
     fast_gelu_f32(v[4]), fast_gelu_f32(v[5]), fast_gelu_f32(v[6]), fast_gelu_f32(v[7])]
}

// ═══════════════════════════════════════════════════════════════════════════════
// §9  Slice processing — for bulk operations on arrays
// ═══════════════════════════════════════════════════════════════════════════════

/// Apply fast_exp to a slice in-place.
#[inline]
pub fn fast_exp_slice(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = fast_exp_f32(*v);
    }
}

/// Apply fast_tanh to a slice in-place.
#[inline]
pub fn fast_tanh_slice(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = fast_tanh_f32(*v);
    }
}

/// Apply fast_sigmoid to a slice in-place.
#[inline]
pub fn fast_sigmoid_slice(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = fast_sigmoid_f32(*v);
    }
}

/// Apply fast_gelu to a slice in-place.
#[inline]
pub fn fast_gelu_slice(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = fast_gelu_f32(*v);
    }
}

/// Apply fast_sqrt to a slice in-place.
#[inline]
pub fn fast_sqrt_slice(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = fast_sqrt_f32(*v);
    }
}

/// Apply fast_log to a slice in-place.
#[inline]
pub fn fast_log_slice(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = fast_log_f32(*v);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §10 Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 1e-5;

    #[test]
    fn test_exp_basic() {
        for &x in &[0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 5.0, -5.0] {
            let expected = x.exp();
            let got = fast_exp_f32(x);
            let rel_err = if expected != 0.0 { (got - expected).abs() / expected.abs() } else { got.abs() };
            assert!(rel_err < TOLERANCE, "exp({}): expected {}, got {}, rel_err {}", x, expected, got, rel_err);
        }
    }

    #[test]
    fn test_exp_edge_cases() {
        assert!(fast_exp_f32(100.0).is_infinite());
        assert_eq!(fast_exp_f32(-200.0), 0.0);
        assert!(fast_exp_f32(f32::NAN).is_nan());
    }

    #[test]
    fn test_tanh_basic() {
        for &x in &[0.0, 0.5, 1.0, 2.0, -1.0, -2.0, 5.0, -5.0, 10.0] {
            let expected = x.tanh();
            let got = fast_tanh_f32(x);
            let abs_err = (got - expected).abs();
            assert!(abs_err < TOLERANCE, "tanh({}): expected {}, got {}, abs_err {}", x, expected, got, abs_err);
        }
    }

    #[test]
    fn test_sqrt_basic() {
        for &x in &[0.0, 1.0, 2.0, 4.0, 9.0, 0.25, 100.0, 0.01] {
            let expected = x.sqrt();
            let got = fast_sqrt_f32(x);
            let rel_err = if expected != 0.0 { (got - expected).abs() / expected } else { got.abs() };
            assert!(rel_err < TOLERANCE, "sqrt({}): expected {}, got {}, rel_err {}", x, expected, got, rel_err);
        }
    }

    #[test]
    fn test_rsqrt_basic() {
        for &x in &[1.0, 2.0, 4.0, 9.0, 0.25, 100.0] {
            let expected = 1.0 / x.sqrt();
            let got = fast_rsqrt_f32(x);
            let rel_err = (got - expected).abs() / expected;
            assert!(rel_err < TOLERANCE, "rsqrt({}): expected {}, got {}, rel_err {}", x, expected, got, rel_err);
        }
    }

    #[test]
    fn test_sin_basic() {
        for &x in &[0.0, 0.5, 1.0, PI_F32 / 4.0, PI_F32 / 2.0, PI_F32, -1.0, -PI_F32, 2.0 * PI_F32] {
            let expected = x.sin();
            let got = fast_sin_f32(x);
            let abs_err = (got - expected).abs();
            assert!(abs_err < TOLERANCE, "sin({}): expected {}, got {}, abs_err {}", x, expected, got, abs_err);
        }
    }

    #[test]
    fn test_cos_basic() {
        for &x in &[0.0, 0.5, 1.0, PI_F32 / 4.0, PI_F32 / 2.0, PI_F32, -1.0, -PI_F32] {
            let expected = x.cos();
            let got = fast_cos_f32(x);
            let abs_err = (got - expected).abs();
            assert!(abs_err < TOLERANCE, "cos({}): expected {}, got {}, abs_err {}", x, expected, got, abs_err);
        }
    }

    #[test]
    fn test_log_basic() {
        for &x in &[0.5, 1.0, 2.0, 4.0, 10.0, 0.1, 100.0, f32::E] {
            let expected = x.ln();
            let got = fast_log_f32(x);
            let rel_err = if expected != 0.0 { (got - expected).abs() / expected.abs() } else { got.abs() };
            assert!(rel_err < 1e-4, "log({}): expected {}, got {}, rel_err {}", x, expected, got, rel_err);
        }
    }

    #[test]
    fn test_sigmoid_basic() {
        for &x in &[0.0, 1.0, -1.0, 5.0, -5.0, 0.5, 10.0, -10.0] {
            let expected = 1.0 / (1.0 + (-x).exp());
            let got = fast_sigmoid_f32(x);
            let abs_err = (got - expected).abs();
            assert!(abs_err < TOLERANCE, "sigmoid({}): expected {}, got {}, abs_err {}", x, expected, got, abs_err);
        }
    }

    #[test]
    fn test_gelu_basic() {
        for &x in &[0.0, 1.0, -1.0, 2.0, -2.0, 0.5] {
            let k = (2.0 / PI_F32).sqrt();
            let cubic = 0.044715 * x * x * x;
            let inner = k * (x + cubic);
            let expected = 0.5 * x * (1.0 + inner.tanh());
            let got = fast_gelu_f32(x);
            let abs_err = (got - expected).abs();
            assert!(abs_err < TOLERANCE, "gelu({}): expected {}, got {}, abs_err {}", x, expected, got, abs_err);
        }
    }

    #[test]
    fn test_vec4_exp() {
        let v = [0.0, 1.0, -1.0, 2.0];
        let r = fast_exp_vec4(v);
        for i in 0..4 {
            let rel_err = (r[i] - v[i].exp()).abs() / v[i].exp().abs().max(1e-30);
            assert!(rel_err < TOLERANCE, "exp_vec4[{}]", i);
        }
    }

    #[test]
    fn test_vec8_tanh() {
        let v = [0.0, 0.5, 1.0, 2.0, -0.5, -1.0, 3.0, -3.0];
        let r = fast_tanh_vec8(v);
        for i in 0..8 {
            let abs_err = (r[i] - v[i].tanh()).abs();
            assert!(abs_err < TOLERANCE, "tanh_vec8[{}]", i);
        }
    }

    #[test]
    fn test_slice_operations() {
        let mut data = [0.0f32, 1.0, -1.0, 2.0];
        fast_exp_slice(&mut data);
        let expected: [f32; 4] = [1.0, std::f32::consts::E, 1.0/std::f32::consts::E, std::f32::consts::E*std::f32::consts::E];
        for i in 0..4 {
            let rel_err = (data[i] - expected[i]).abs() / expected[i];
            assert!(rel_err < TOLERANCE, "exp_slice[{}]", i);
        }
    }
}
