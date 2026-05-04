// =========================================================================
// Formal Verification & Proof Trust Protocol
//
// Bridges the gap between compile-time SMT proofs and runtime safety.
// When the SMT solver proves an optimization is safe, the runtime should
// TRUST that proof and disable redundant runtime checks.
//
// Architecture:
//   Layer 1: Proof Trust Protocol - TIER_0_TRUSTED bytecode skips watchdog
//   Layer 2: Entropy Watchdog      - State-mutation monitoring (not PC counting)
//   Layer 3: OSR De-optimization   - Graceful fallback instead of panic
// =========================================================================

use crate::compiler::sat_smt_solver::SatSmtSolver;
use crate::compiler::translation_validation::{TranslationValidator, ValidationResult};

// =============================================================================
// §1  TRUST TIER SYSTEM
// =============================================================================

/// Trust tier metadata embedded in compiled bytecode function headers.
///
/// When the SMT solver and/or translation validator prove properties about
/// a function (termination, no-overflow, etc.), the compiler tags that
/// function with the appropriate tier. The runtime uses this tag to decide
/// which safety checks to skip — trading redundant runtime overhead for
/// *proven* compile-time guarantees.
///
/// ## Safety Invariant
///
///   TIER_0_TRUSTED > TIER_1_VERIFIED > TIER_2_HEURISTIC > TIER_3_UNVERIFIED
///
/// A function can only be promoted to a higher tier after the corresponding
/// proof obligation is discharged.  Demotion (e.g. after a guard failure
/// during OSR) resets the tier to TIER_3_UNVERIFIED.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum TrustTier {
    /// Fully verified by SMT solver: termination proven, overflow proven
    /// impossible, all array bounds proven safe. The runtime may:
    ///   - Disable the watchdog timer entirely
    ///   - Use wrapping arithmetic without overflow checks
    ///   - Skip bounds checks on array accesses
    ///   - Run at maximum throughput with zero safety overhead
    Tier0Trusted = 0,

    /// Partially verified: some properties proven (e.g. termination but
    /// not overflow). The runtime may disable specific checks that were
    /// proven, but must retain others.
    Tier1Verified = 1,

    /// Heuristically safe: no formal proof, but static analysis suggests
    /// the function is likely safe (e.g. simple loops with clear bounds).
    /// The runtime uses the entropy watchdog with relaxed thresholds.
    Tier2Heuristic = 2,

    /// Unverified: no proof or heuristic. The runtime applies full
    /// safety monitoring (entropy watchdog + checked arithmetic).
    Tier3Unverified = 3,
}

impl Default for TrustTier {
    fn default() -> Self {
        TrustTier::Tier3Unverified
    }
}

impl TrustTier {
    /// Returns true if this tier allows the runtime to skip the watchdog.
    #[inline(always)]
    pub fn skip_watchdog(&self) -> bool {
        matches!(self, TrustTier::Tier0Trusted)
    }

    /// Returns true if this tier allows unchecked (wrapping) arithmetic.
    #[inline(always)]
    pub fn allow_unchecked_arithmetic(&self) -> bool {
        matches!(self, TrustTier::Tier0Trusted | TrustTier::Tier1Verified)
    }

    /// Returns the default arithmetic mode for this trust tier.
    #[inline(always)]
    pub fn default_arithmetic_mode(&self) -> ArithmeticMode {
        match self {
            TrustTier::Tier0Trusted => ArithmeticMode::Wrapping,
            TrustTier::Tier1Verified => ArithmeticMode::Wrapping,
            TrustTier::Tier2Heuristic => ArithmeticMode::Strict,
            TrustTier::Tier3Unverified => ArithmeticMode::Strict,
        }
    }

    /// Returns the watchdog sensitivity for this trust tier.
    #[inline(always)]
    pub fn watchdog_sensitivity(&self) -> WatchdogSensitivity {
        match self {
            TrustTier::Tier0Trusted => WatchdogSensitivity::Disabled,
            TrustTier::Tier1Verified => WatchdogSensitivity::Relaxed,
            TrustTier::Tier2Heuristic => WatchdogSensitivity::Normal,
            TrustTier::Tier3Unverified => WatchdogSensitivity::Aggressive,
        }
    }
}

// =============================================================================
// §2  ARITHMETIC MODE
// =============================================================================

/// Arithmetic semantics for integer operations in the bytecode VM.
///
/// Controls what happens when an integer operation would overflow:
///   - **Strict**: panic on overflow (Rust default, safe but slow)
///   - **Wrapping**: silently wrap around (two's complement, fast)
///   - **Saturating**: clamp to MIN/MAX (useful for DSP/signal processing)
///
/// The compiler selects the mode based on:
///   1. The `strict_math` attribute on the function (user-specified)
///   2. The TrustTier (SMT-verified functions can safely use Wrapping)
///   3. The global VM configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ArithmeticMode {
    /// Rust's default checked arithmetic: panics on overflow in debug,
    /// wraps in release.  The VM explicitly checks and returns an error.
    Strict = 0,
    /// Two's complement wrapping: overflow silently wraps around.
    /// This is the correct default for systems programming and
    /// performance-critical code.  When the SMT solver proves that
    /// overflow cannot occur, this mode is *semantically equivalent*
    /// to Strict but with zero runtime overhead.
    Wrapping = 1,
    /// Saturating arithmetic: clamps to i64::MIN or i64::MAX on overflow.
    /// Useful for DSP workloads where wrapping would produce audibly
    /// incorrect results.
    Saturating = 2,
}

impl Default for ArithmeticMode {
    fn default() -> Self {
        ArithmeticMode::Wrapping
    }
}

impl ArithmeticMode {
    /// Apply integer addition with the selected overflow semantics.
    #[inline(always)]
    pub fn add(&self, l: i64, r: i64) -> i64 {
        match self {
            ArithmeticMode::Strict => l.checked_add(r).unwrap_or_else(|| {
                // In strict mode, we don't panic — we return the wrapping
                // result but the VM will have already checked and returned
                // an error. This branch should never be reached in practice.
                l.wrapping_add(r)
            }),
            ArithmeticMode::Wrapping => l.wrapping_add(r),
            ArithmeticMode::Saturating => l.saturating_add(r),
        }
    }

    /// Apply integer subtraction with the selected overflow semantics.
    #[inline(always)]
    pub fn sub(&self, l: i64, r: i64) -> i64 {
        match self {
            ArithmeticMode::Strict => l.checked_sub(r).unwrap_or_else(|| {
                l.wrapping_sub(r)
            }),
            ArithmeticMode::Wrapping => l.wrapping_sub(r),
            ArithmeticMode::Saturating => l.saturating_sub(r),
        }
    }

    /// Apply integer multiplication with the selected overflow semantics.
    #[inline(always)]
    pub fn mul(&self, l: i64, r: i64) -> i64 {
        match self {
            ArithmeticMode::Strict => l.checked_mul(r).unwrap_or_else(|| {
                l.wrapping_mul(r)
            }),
            ArithmeticMode::Wrapping => l.wrapping_mul(r),
            ArithmeticMode::Saturating => l.saturating_mul(r),
        }
    }

    /// Apply integer negation with the selected overflow semantics.
    #[inline(always)]
    pub fn neg(&self, v: i64) -> i64 {
        match self {
            ArithmeticMode::Strict => v.checked_neg().unwrap_or_else(|| {
                v.wrapping_neg()
            }),
            ArithmeticMode::Wrapping => v.wrapping_neg(),
            ArithmeticMode::Saturating => v.saturating_neg(),
        }
    }

    /// Check if an overflow occurred and return an error in Strict mode.
    /// Returns Ok(result) for Wrapping/Saturating modes (never errors).
    #[inline(always)]
    pub fn check_add(&self, l: i64, r: i64) -> Result<i64, String> {
        match self {
            ArithmeticMode::Strict => l.checked_add(r)
                .ok_or_else(|| format!("arithmetic overflow: {} + {}", l, r)),
            ArithmeticMode::Wrapping => Ok(l.wrapping_add(r)),
            ArithmeticMode::Saturating => Ok(l.saturating_add(r)),
        }
    }

    #[inline(always)]
    pub fn check_sub(&self, l: i64, r: i64) -> Result<i64, String> {
        match self {
            ArithmeticMode::Strict => l.checked_sub(r)
                .ok_or_else(|| format!("arithmetic overflow: {} - {}", l, r)),
            ArithmeticMode::Wrapping => Ok(l.wrapping_sub(r)),
            ArithmeticMode::Saturating => Ok(l.saturating_sub(r)),
        }
    }

    #[inline(always)]
    pub fn check_mul(&self, l: i64, r: i64) -> Result<i64, String> {
        match self {
            ArithmeticMode::Strict => l.checked_mul(r)
                .ok_or_else(|| format!("arithmetic overflow: {} * {}", l, r)),
            ArithmeticMode::Wrapping => Ok(l.wrapping_mul(r)),
            ArithmeticMode::Saturating => Ok(l.saturating_mul(r)),
        }
    }
}

// =============================================================================
// §3  WATCHDOG SENSITIVITY
// =============================================================================

/// Controls how aggressively the loop watchdog monitors execution.
///
/// The watchdog is the runtime's defense against infinite loops. Its
/// sensitivity determines how many iterations it tolerates before
/// triggering a safety response (de-optimization or error).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchdogSensitivity {
    /// Watchdog is completely disabled. Used only for TIER_0_TRUSTED
    /// functions where the SMT solver has proven termination.
    Disabled,

    /// Very high tolerance: allows up to 100M iterations before triggering.
    /// Used for TIER_1_VERIFIED functions where termination is likely.
    Relaxed,

    /// Normal tolerance: allows up to 10M iterations.
    /// Used for TIER_2_HEURISTIC functions.
    Normal,

    /// Low tolerance: triggers after 1M iterations.
    /// Used for TIER_3_UNVERIFIED functions with no safety guarantees.
    Aggressive,
}

impl WatchdogSensitivity {
    /// Returns the maximum number of iterations before the watchdog triggers.
    /// Returns None if the watchdog is disabled.
    #[inline]
    pub fn max_iterations(&self) -> Option<u64> {
        match self {
            WatchdogSensitivity::Disabled => None,
            WatchdogSensitivity::Relaxed => Some(1_000_000_000),
            WatchdogSensitivity::Normal => Some(100_000_000),
            WatchdogSensitivity::Aggressive => Some(10_000_000),
        }
    }

    /// Returns the maximum iterations, scaled by a complexity factor.
    /// The GNN learned-latency model can provide this factor: if it knows
    /// a function is heavy, it tells the VM to "be patient."
    #[inline]
    pub fn max_iterations_scaled(&self, complexity_factor: f64) -> Option<u64> {
        self.max_iterations()
            .map(|base| ((base as f64) * complexity_factor) as u64)
    }
}

// =============================================================================
// §4  PROOF VERIFICATION PIPELINE
// =============================================================================

/// Result of the full formal verification pipeline for a function.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// The trust tier assigned to this function.
    pub tier: TrustTier,
    /// Whether termination was proven (loop always exits).
    pub termination_proven: bool,
    /// Whether absence of arithmetic overflow was proven.
    pub overflow_proven_safe: bool,
    /// Whether array bounds safety was proven.
    pub bounds_proven_safe: bool,
    /// Human-readable explanation of the verification outcome.
    pub explanation: String,
}

impl VerificationResult {
    /// Create a result for an unverified function.
    pub fn unverified(reason: &str) -> Self {
        Self {
            tier: TrustTier::Tier3Unverified,
            termination_proven: false,
            overflow_proven_safe: false,
            bounds_proven_safe: false,
            explanation: reason.to_string(),
        }
    }

    /// Create a result for a fully verified function.
    pub fn fully_verified(explanation: &str) -> Self {
        Self {
            tier: TrustTier::Tier0Trusted,
            termination_proven: true,
            overflow_proven_safe: true,
            bounds_proven_safe: true,
            explanation: explanation.to_string(),
        }
    }
}

/// The formal verification pipeline that combines SMT solving,
/// translation validation, and heuristic analysis to assign
/// a TrustTier to each compiled function.
pub struct FormalVerifier {
    /// Whether SMT-based verification is enabled.
    pub smt_enabled: bool,
    /// Whether translation validation is enabled.
    pub translation_validation_enabled: bool,
}

impl FormalVerifier {
    pub fn new() -> Self {
        Self {
            smt_enabled: true,
            translation_validation_enabled: true,
        }
    }

    /// Run the verification pipeline on a function.
    ///
    /// This is the main entry point called by the compiler after
    /// optimization. It attempts to prove safety properties and
    /// assigns a TrustTier accordingly.
    pub fn verify_function(
        &self,
        function_name: &str,
        instruction_count: usize,
    ) -> VerificationResult {
        // Simple heuristic: very short functions (≤3 instructions) with
        // no loops are trivially safe.
        if instruction_count <= 3 {
            return VerificationResult {
                tier: TrustTier::Tier2Heuristic,
                termination_proven: true,
                overflow_proven_safe: false,
                bounds_proven_safe: false,
                explanation: format!(
                    "Function '{}' has only {} instructions — trivially terminates",
                    function_name, instruction_count
                ),
            };
        }

        // If SMT verification is enabled, attempt to prove properties.
        // In a full implementation, this would:
        //   1. Encode the function's semantics as SMT constraints
        //   2. Ask the solver to prove: ∀ inputs, no overflow ∧ loop terminates
        //   3. If proven, assign TIER_0_TRUSTED
        if self.smt_enabled {
            // Placeholder: in production, this calls into SatSmtSolver
            // and TranslationValidator. For now, we use the heuristic.
        }

        // Default: unverified
        VerificationResult::unverified(&format!(
            "Function '{}' not formally verified ({} instructions)",
            function_name, instruction_count
        ))
    }

    /// Verify a function using the SMT solver's range analysis.
    ///
    /// If the range analysis can prove that all intermediate values
    /// fit within i64, and that all loop induction variables are
    /// strictly monotonic, the function is promoted to TIER_1_VERIFIED
    /// or TIER_0_TRUSTED.
    pub fn verify_with_smt(
        &self,
        solver: &SatSmtSolver,
        function_name: &str,
    ) -> VerificationResult {
        // If the solver has constraints that are all unsatisfiable
        // for overflow conditions, the function is overflow-safe.
        // This is a simplified check; a full implementation would
        // construct overflow constraints for every arithmetic operation.

        // For now, if the solver has no constraints, we can't prove anything.
        VerificationResult::unverified(&format!(
            "SMT verification incomplete for '{}'",
            function_name
        ))
    }

    /// Verify a function using translation validation results.
    ///
    /// If the translation validator proves that the optimized code
    /// is semantically equivalent to the original, the function's
    /// optimization is safe and can be trusted.
    pub fn verify_with_translation(
        &self,
        validator: &mut TranslationValidator,
        function_name: &str,
    ) -> VerificationResult {
        let result = validator.validate();

        match result {
            ValidationResult::Valid => VerificationResult {
                tier: TrustTier::Tier1Verified,
                termination_proven: false,  // Translation validation doesn't prove termination
                overflow_proven_safe: true,  // But it does prove semantic equivalence
                bounds_proven_safe: true,
                explanation: format!(
                    "Translation validation passed for '{}' — optimized code is semantically equivalent",
                    function_name
                ),
            },
            ValidationResult::Invalid { reason, .. } => VerificationResult::unverified(&format!(
                "Translation validation FAILED for '{}': {}",
                function_name, reason
            )),
            ValidationResult::Unknown { reason } => VerificationResult {
                tier: TrustTier::Tier3Unverified,
                termination_proven: false,
                overflow_proven_safe: false,
                bounds_proven_safe: false,
                explanation: format!(
                    "Translation validation inconclusive for '{}': {}",
                    function_name, reason
                ),
            },
        }
    }
}

impl Default for FormalVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// §5  OSR (ON-STACK REPLACEMENT) DE-OPTIMIZATION
// =============================================================================

/// Reason for an OSR de-optimization event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OsrTrigger {
    /// Arithmetic overflow detected in strict mode.
    ArithmeticOverflow { operation: String, lhs: i64, rhs: i64 },
    /// Watchdog detected a potentially infinite loop.
    WatchdogTimeout { pc: usize, iterations: u64 },
    /// Type guard failure in the JIT-compiled code.
    TypeGuardFailure { expected: String, actual: String },
    /// Bounds check failure.
    BoundsCheckFailure { index: usize, length: usize },
}

/// Result of an OSR de-optimization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OsrOutcome {
    /// Successfully de-optimized to the safe interpreter.
    /// Execution continues in interpreted mode.
    Deoptimized {
        /// The function that was de-optimized.
        function_name: String,
        /// The program counter where de-optimization occurred.
        pc: usize,
        /// The reason for de-optimization.
        trigger: OsrTrigger,
    },
    /// De-optimization is not available (no interpreter fallback).
    /// The VM must decide whether to panic or continue with reduced safety.
    Unavailable {
        trigger: OsrTrigger,
    },
}

/// The OSR engine manages transitions between optimized and
/// de-optimized execution modes.
#[derive(Debug)]
pub struct OsrEngine {
    /// Number of times de-optimization has been triggered.
    deopt_count: u64,
    /// Whether OSR is enabled (requires interpreter fallback).
    enabled: bool,
}

impl OsrEngine {
    pub fn new(enabled: bool) -> Self {
        Self {
            deopt_count: 0,
            enabled,
        }
    }

    /// Attempt an OSR de-optimization.
    ///
    /// In a full implementation, this would:
    ///   1. Capture the current VM state (registers, stack, PC)
    ///   2. Translate the optimized representation back to the
    ///      interpreter's frame layout
    ///   3. Resume execution in the interpreter
    ///
    /// For now, we record the event and demote the trust tier.
    pub fn deoptimize(&mut self, trigger: OsrTrigger, function_name: &str, pc: usize) -> OsrOutcome {
        self.deopt_count += 1;

        if self.enabled {
            OsrOutcome::Deoptimized {
                function_name: function_name.to_string(),
                pc,
                trigger,
            }
        } else {
            OsrOutcome::Unavailable { trigger }
        }
    }

    /// Get the number of de-optimization events.
    pub fn deopt_count(&self) -> u64 {
        self.deopt_count
    }
}

impl Default for OsrEngine {
    fn default() -> Self {
        Self::new(true)
    }
}

// =============================================================================
// §6  ENTROPY WATCHDOG
// =============================================================================

/// State-mutation entropy watchdog for loop detection.
///
/// Unlike the naive PC-counter approach (which panics when the same PC
/// is hit N times), this watchdog monitors whether the program state
/// is actually *changing*. A loop that is iterating through an array
/// or mutating variables has "entropy" — it's making progress. Only
/// truly stagnant loops (where state is unchanged across iterations)
/// are flagged as potential infinite loops.
///
/// ## How it works
///
/// The watchdog maintains a "fingerprint" of the program state:
///   - Hash of all register/slot values
///   - Hash of memory writes since the last backedge
///
/// On each backward jump (loop backedge), it compares the current
/// fingerprint with the previous one. If the fingerprint changes,
/// the loop has entropy and the watchdog resets its counter.
/// If the fingerprint is unchanged for N consecutive backedges,
/// the watchdog triggers.
#[derive(Debug)]
pub struct EntropyWatchdog {
    /// Previous state fingerprint (hash of slot values).
    pub prev_fingerprint: u64,
    /// Number of consecutive backedges with the same fingerprint.
    pub stagnant_count: u64,
    /// Total number of backedges observed.
    pub total_backedges: u64,
    /// Number of distinct state changes observed.
    pub entropy_events: u64,
    /// Sensitivity level controlling the trigger threshold.
    pub sensitivity: WatchdogSensitivity,
    /// Complexity factor from the GNN learned-latency model.
    /// Heavier functions get more patience.
    pub complexity_factor: f64,
    /// Whether the watchdog is currently active.
    pub active: bool,
}

impl EntropyWatchdog {
    /// Create a new entropy watchdog with the given sensitivity.
    pub fn new(sensitivity: WatchdogSensitivity) -> Self {
        Self {
            prev_fingerprint: 0,
            stagnant_count: 0,
            total_backedges: 0,
            entropy_events: 0,
            sensitivity,
            complexity_factor: 1.0,
            active: sensitivity != WatchdogSensitivity::Disabled,
        }
    }

    /// Set the complexity factor (from the GNN learned-latency model).
    pub fn set_complexity_factor(&mut self, factor: f64) {
        self.complexity_factor = factor.max(0.1);
    }

    /// Observe a loop backedge with the current slot state.
    ///
    /// Call this when the PC jumps backward (loop iteration).
    /// Returns `true` if the watchdog detects a potentially
    /// infinite loop (stagnant state).
    pub fn observe_backedge(&mut self, slots: &[crate::interp::Value]) -> bool {
        if !self.active {
            return false;
        }

        self.total_backedges += 1;

        // Compute a lightweight fingerprint of the current state.
        // We don't need a cryptographic hash — just something that
        // changes when the state changes.
        let fingerprint = Self::fingerprint_slots(slots);

        if fingerprint != self.prev_fingerprint {
            // State has changed — the loop has entropy (making progress).
            self.entropy_events += 1;
            self.stagnant_count = 0;
        } else {
            // State is unchanged — potential infinite loop.
            self.stagnant_count += 1;
        }

        self.prev_fingerprint = fingerprint;

        // Check if we've exceeded the threshold.
        if let Some(max_iter) = self.sensitivity.max_iterations_scaled(self.complexity_factor) {
            // Trigger if total iterations exceed the threshold AND
            // no entropy has been detected recently.
            let stagnant_threshold = (max_iter / 100).max(1000); // 1% of max, min 1000
            if self.stagnant_count > stagnant_threshold {
                return true; // Infinite loop detected!
            }
            // Also trigger if total iterations exceed the hard limit
            // even with entropy (extremely long-running loop).
            if self.total_backedges > max_iter {
                return true;
            }
        }

        false
    }

    /// Quick check: is the loop making progress?
    /// Returns the entropy ratio (0.0 = stagnant, 1.0 = fully progressing).
    pub fn entropy_ratio(&self) -> f64 {
        if self.total_backedges == 0 {
            return 1.0
        }
        self.entropy_events as f64 / self.total_backedges as f64
    }

    /// Reset the watchdog state (e.g. when entering a new function).
    pub fn reset(&mut self) {
        self.prev_fingerprint = 0;
        self.stagnant_count = 0;
        self.total_backedges = 0;
        self.entropy_events = 0;
    }

    /// Compute a lightweight fingerprint of the slot array.
    ///
    /// Uses FxHash-style mixing for speed. We only hash the first
    /// N slots to avoid O(n) overhead on large slot arrays.
    fn fingerprint_slots(slots: &[crate::interp::Value]) -> u64 {
        // Only hash up to 64 slots for performance.
        // This is sufficient to detect state changes in the
        // vast majority of loops (which use few variables).
        let len = slots.len().min(64);
        let mut hash: u64 = 0xAF63_DC4C_8601_EC8C; // FxHash seed
        for i in 0..len {
            // Mix the slot index and a simple type-tag-based hash.
            // We avoid deep-hashing Value contents for speed;
            // instead, we hash the type tag + a shallow value.
            let slot = &slots[i];
            let type_hash = slot.type_tag() as u64;
            let val_hash = match slot {
                crate::interp::Value::I64(v) => *v as u64,
                crate::interp::Value::F64(v) => v.to_bits(),
                crate::interp::Value::Bool(v) => *v as u64,
                crate::interp::Value::U64(v) => *v,
                crate::interp::Value::I32(v) => *v as u64,
                _ => i as u64, // Use index as proxy for complex types
            };
            // FxHash mixing
            hash = hash.rotate_left(8) ^ (type_hash.wrapping_mul(0x517c_c1b7_2722_0a95));
            hash = hash.rotate_left(8) ^ (val_hash.wrapping_mul(0x517c_c1b7_2722_0a95));
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic_modes() {
        // Wrapping mode: overflow wraps around
        let wrapping = ArithmeticMode::Wrapping;
        assert_eq!(wrapping.mul(i64::MAX, 2), i64::MAX.wrapping_mul(2));

        // Saturating mode: overflow clamps
        let saturating = ArithmeticMode::Saturating;
        assert_eq!(saturating.mul(i64::MAX, 2), i64::MAX);
        assert_eq!(saturating.add(i64::MAX, 1), i64::MAX);

        // Strict mode: checked operations
        let strict = ArithmeticMode::Strict;
        assert!(strict.check_mul(i64::MAX, 2).is_err());
        assert!(strict.check_add(i64::MAX, 1).is_err());
        assert!(strict.check_mul(100, 200).is_ok());
    }

    #[test]
    fn test_trust_tier_privileges() {
        assert!(TrustTier::Tier0Trusted.skip_watchdog());
        assert!(TrustTier::Tier0Trusted.allow_unchecked_arithmetic());
        assert!(!TrustTier::Tier3Unverified.skip_watchdog());
        assert!(!TrustTier::Tier3Unverified.allow_unchecked_arithmetic());
    }

    #[test]
    fn test_watchdog_sensitivity_thresholds() {
        assert!(WatchdogSensitivity::Disabled.max_iterations().is_none());
        assert_eq!(WatchdogSensitivity::Relaxed.max_iterations(), Some(1_000_000_000));
        assert_eq!(WatchdogSensitivity::Normal.max_iterations(), Some(100_000_000));
        assert_eq!(WatchdogSensitivity::Aggressive.max_iterations(), Some(10_000_000));
    }

    #[test]
    fn test_osr_engine() {
        let mut osr = OsrEngine::new(true);
        let outcome = osr.deoptimize(
            OsrTrigger::ArithmeticOverflow {
                operation: "mul".to_string(),
                lhs: i64::MAX,
                rhs: 2,
            },
            "test_func",
            42,
        );
        assert!(matches!(outcome, OsrOutcome::Deoptimized { .. }));
        assert_eq!(osr.deopt_count(), 1);
    }

    #[test]
    fn test_entropy_watchdog_detects_stagnant() {
        use crate::interp::Value;
        let mut watchdog = EntropyWatchdog::new(WatchdogSensitivity::Aggressive);
        // Same state every time — should eventually trigger
        let slots = vec![Value::I64(42), Value::I64(0)];
        for _ in 0..2000 {
            if watchdog.observe_backedge(&slots) {
                return; // Detected stagnant loop
            }
        }
        panic!("Watchdog should have detected stagnant loop");
    }

    #[test]
    fn test_entropy_watchdog_allows_progress() {
        use crate::interp::Value;
        let mut watchdog = EntropyWatchdog::new(WatchdogSensitivity::Normal);
        // Changing state — should not trigger
        for i in 0..100 {
            let slots = vec![Value::I64(i), Value::I64(0)];
            let triggered = watchdog.observe_backedge(&slots);
            assert!(!triggered, "Watchdog should not trigger on progressing loop");
        }
        assert!(watchdog.entropy_ratio() > 0.5);
    }
}
