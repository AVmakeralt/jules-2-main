// =============================================================================
// AutoVectorizer Bridge — connects the autovec module to the JIT pipeline
//
// This module translates the JIT interpreter's instruction stream into the
// loop-level representation expected by `AutoVectorizer`, then feeds
// vectorization hints back into the compilation pipeline.
//
// Responsibilities:
//   1. Scan the instruction stream for loops (backwards jumps).
//   2. Detect induction variables and loop-body patterns.
//   3. Build `LoopInfo` descriptors for each detected loop.
//   4. Invoke `AutoVectorizer::analyze_loop` on each descriptor.
//   5. Return `SimdHint`s for loops that are safe and beneficial to vectorize.
// =============================================================================

use crate::compiler::ast::BinOpKind;
use crate::interp::Instr;
#[allow(unused_imports)]
use crate::optimizer::autovec::{
    AutoVectorizer, LoopInfo, VectorOp, VectorPattern, VectorWidth, VerificationResult,
};

// ---------------------------------------------------------------------------
// SimdHint — a vectorization recommendation that the JIT can consume
// ---------------------------------------------------------------------------

/// A SIMD vectorization hint emitted for a loop that the auto-vectorizer
/// has determined is safe and beneficial to vectorize.
#[derive(Debug, Clone)]
pub struct SimdHint {
    /// Program-counter index of the first instruction in the loop header.
    pub loop_start_pc: usize,
    /// Program-counter index of the backward-jump that closes the loop.
    pub loop_end_pc: usize,
    /// Recommended SIMD vector width.
    pub width: VectorWidth,
    /// Detected vector pattern (map / reduce / zip / …).
    pub pattern: VectorPattern,
    /// Estimated speed-up factor (e.g. 3.8×).
    pub speedup: f64,
    /// Slot index of the primary induction variable.
    pub induction_var: u16,
}

// ---------------------------------------------------------------------------
// Loop detection helpers
// ---------------------------------------------------------------------------

/// A lightweight description of a loop found in the instruction stream.
struct RawLoop {
    /// PC of the first instruction inside the loop body.
    start_pc: usize,
    /// PC of the backward branch that closes the loop.
    end_pc: usize,
    /// Slot that appears to be the induction variable (incremented each iter).
    induction_slot: Option<u16>,
    /// Arithmetic operations observed inside the loop body.
    body_ops: Vec<VectorOp>,
    /// Whether the loop body contains array-index operations.
    has_indexed_access: bool,
    /// Whether the loop body looks element-wise (load-compute-store per iter).
    is_element_wise: bool,
    /// Whether the loop appears to accumulate into a single value.
    is_reduction: bool,
    /// Best-guess element type derived from the operations inside the loop.
    element_type: String,
}

/// Classify a `BinOpKind` into a `VectorOp`, if applicable.
#[allow(dead_code)]
fn binop_to_vectorop(kind: BinOpKind) -> Option<VectorOp> {
    match kind {
        BinOpKind::Add => Some(VectorOp::Add),
        BinOpKind::Sub => Some(VectorOp::Sub),
        BinOpKind::Mul => Some(VectorOp::Mul),
        BinOpKind::Div => Some(VectorOp::Div),
        _ => None,
    }
}

/// Infer the element type string from the instructions in a loop body.
fn infer_element_type(instrs: &[Instr], start: usize, end: usize) -> String {
    for instr in &instrs[start..=end.min(instrs.len() - 1)] {
        match instr {
            Instr::LoadF64(..) | Instr::BinOp(_, BinOpKind::Div, _, _) => return "f64".to_string(),
            Instr::LoadF32(..) => return "f32".to_string(),
            Instr::LoadI64(..) => return "i64".to_string(),
            _ => {}
        }
    }
    // Default to i32 — the most common integer width.
    "i32".to_string()
}

/// Scan the instruction stream and return descriptors for every loop
/// (i.e. every backwards conditional or unconditional jump).
fn detect_loops(instrs: &[Instr]) -> Vec<RawLoop> {
    let mut loops = Vec::new();

    for (pc, instr) in instrs.iter().enumerate() {
        let offset = match instr {
            Instr::Jump(off) => *off,
            Instr::JumpFalse(_, off) => *off,
            Instr::JumpTrue(_, off) => *off,
            _ => continue,
        };

        // A negative offset means we are jumping backwards → loop.
        if offset >= 0 {
            continue;
        }

        let target_pc = (pc as i64 + offset as i64) as usize;
        if target_pc >= pc {
            // Shouldn't happen for negative offsets, but be safe.
            continue;
        }

        let start_pc = target_pc;
        let end_pc = pc;

        // --- Analyse the body between start_pc and end_pc -------------------
        let mut induction_slot: Option<u16> = None;
        let mut body_ops: Vec<VectorOp> = Vec::new();
        let mut has_indexed_access = false;
        let mut is_element_wise = false;
        let mut is_reduction = false;

        // Track stores to detect an induction-style increment.
        let mut stored_slots = Vec::new();
        // Track loads to detect element-wise load-compute-store.
        let mut load_count = 0usize;
        let mut store_count = 0usize;

        for i in start_pc..=end_pc.min(instrs.len() - 1) {
            match &instrs[i] {
                Instr::Store(slot, _reg) => {
                    stored_slots.push(*slot);
                    store_count += 1;
                }
                Instr::Load(_reg, _slot) => {
                    load_count += 1;
                }
                Instr::BinOp(_dst, kind, _lhs, _rhs) => {
                    if let Some(vop) = binop_to_vectorop(*kind) {
                        body_ops.push(vop);
                    }
                }
                Instr::ArrayGet(..) | Instr::IndexGet(..) => {
                    has_indexed_access = true;
                    load_count += 1;
                }
                Instr::ArraySet(..) | Instr::IndexSet(..) => {
                    store_count += 1;
                }
                _ => {}
            }
        }

        // Heuristic: if a slot is stored and then used as an add operand,
        // treat it as the induction variable.
        for &slot in &stored_slots {
            for i in start_pc..=end_pc.min(instrs.len() - 1) {
                if let Instr::BinOp(_dst, BinOpKind::Add, _lhs, _rhs) = &instrs[i] {
                    // Crude: first stored slot that participates in an add
                    // is treated as the induction variable.
                    induction_slot = Some(slot);
                    break;
                }
            }
            if induction_slot.is_some() {
                break;
            }
        }

        // If there is roughly one load and one store per iteration, call it
        // element-wise.  A single store with many loads suggests a reduction.
        if store_count == 1 && load_count > 1 {
            is_reduction = true;
        } else if load_count > 0 && store_count > 0 {
            is_element_wise = true;
        }

        let element_type = infer_element_type(instrs, start_pc, end_pc);

        loops.push(RawLoop {
            start_pc,
            end_pc,
            induction_slot,
            body_ops,
            has_indexed_access,
            is_element_wise,
            is_reduction,
            element_type,
        });
    }

    loops
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Analyse the instruction stream and return SIMD hints for every loop that
/// the auto-vectorizer judges safe and beneficial to vectorize.
///
/// Returns `None` when no loops are detected or none can be safely vectorized.
#[allow(dead_code)]
pub fn try_vectorize_instructions(instrs: &[Instr], slot_count: u16) -> Option<Vec<SimdHint>> {
    let raw_loops = detect_loops(instrs);
    if raw_loops.is_empty() {
        return None;
    }

    let mut vectorizer = AutoVectorizer::new();
    let mut hints: Vec<SimdHint> = Vec::new();

    for raw in &raw_loops {
        // Estimate the iteration count from the slot range.
        // This is a rough heuristic: we use the induction variable's slot
        // index relative to the total slot count as a proxy.  When we cannot
        // determine an induction variable we fall back to a conservative
        // estimate.
        let iteration_count = raw
            .induction_slot
            .map(|s| {
                // Assume the loop runs over a range proportional to the slot
                // space.  In practice the JIT would pass in real profile data.
                (slot_count as usize).saturating_sub(s as usize).max(32)
            })
            .unwrap_or(32);

        let loop_info = LoopInfo {
            iteration_count,
            element_type: raw.element_type.clone(),
            is_reduction: raw.is_reduction,
            has_indexed_access: raw.has_indexed_access,
            is_element_wise: raw.is_element_wise,
            operations: raw
                .body_ops
                .iter()
                .map(|op| format!("{:?}", op).to_lowercase())
                .collect(),
        };

        let location = format!("pc_{}..{}", raw.start_pc, raw.end_pc);

        // Feed the loop to the auto-vectorizer.
        vectorizer.analyze_loop(location, loop_info);
    }

    // Collect only the candidates that the vectorizer marked as safe.
    for candidate in vectorizer.safe_candidates() {
        // Recover the RawLoop that corresponds to this candidate.
        let raw = raw_loops
            .iter()
            .find(|r| format!("pc_{}..{}", r.start_pc, r.end_pc) == candidate.location);

        if let Some(raw) = raw {
            hints.push(SimdHint {
                loop_start_pc: raw.start_pc,
                loop_end_pc: raw.end_pc,
                width: candidate.width,
                pattern: candidate.pattern.clone(),
                speedup: candidate.speedup,
                induction_var: raw.induction_slot.unwrap_or(0),
            });
        }
    }

    if hints.is_empty() {
        None
    } else {
        Some(hints)
    }
}

/// Decide whether a load from `slot` with the given `stride` (in bytes)
/// would benefit from a software prefetch instruction.
///
/// Prefetching is worthwhile for sequential or small-stride access patterns
/// that will be traversed repeatedly — typical of vectorised loops.
#[allow(dead_code)]
pub fn should_emit_prefetch(slot: u16, stride: i64) -> bool {
    // Skip slot 0 (usually a special/void slot).
    if slot == 0 {
        return false;
    }

    // Positive, moderate strides (4 B, 8 B, 16 B, 32 B …) correspond to
    // iterating over contiguous arrays and benefit from prefetching.
    if stride > 0 && stride <= 64 {
        return true;
    }

    // A stride of exactly the cache-line size (64 B on most x86) is the
    // sweet spot for prefetching.
    if stride == 64 {
        return true;
    }

    // Strides up to one page (4 KiB) can still benefit from prefetch on
    // modern CPUs, but with diminishing returns.
    if stride > 64 && stride <= 4096 && stride % 64 == 0 {
        return true;
    }

    // Negative strides are rare in generated code; skip them.
    // Very large strides (>4 KiB) are unlikely to benefit from prefetching
    // because the hardware prefetcher does not track them well.
    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_emit_prefetch_sequential() {
        // 4-byte stride (i32 array) → definitely prefetch.
        assert!(should_emit_prefetch(1, 4));
        // 8-byte stride (f64/i64 array) → definitely prefetch.
        assert!(should_emit_prefetch(1, 8));
        // Cache-line stride → definitely prefetch.
        assert!(should_emit_prefetch(1, 64));
    }

    #[test]
    fn test_should_emit_prefetch_negative_or_zero() {
        // Slot 0 is special — no prefetch.
        assert!(!should_emit_prefetch(0, 8));
        // Negative strides — no prefetch.
        assert!(!should_emit_prefetch(1, -8));
    }

    #[test]
    fn test_should_emit_prefetch_large_stride() {
        // 128 B stride, cache-line aligned → still beneficial.
        assert!(should_emit_prefetch(1, 128));
        // 4 KiB stride, cache-line aligned → borderline but allowed.
        assert!(should_emit_prefetch(1, 4096));
        // 8 KiB stride → too large, skip.
        assert!(!should_emit_prefetch(1, 8192));
        // 100 B stride — not cache-line aligned → skip.
        assert!(!should_emit_prefetch(1, 100));
    }

    #[test]
    fn test_try_vectorize_no_loops() {
        // A flat instruction stream with no backward jumps → None.
        let instrs = vec![
            Instr::LoadI32(0, 1),
            Instr::LoadI32(1, 2),
            Instr::BinOp(2, BinOpKind::Add, 0, 1),
            Instr::Return(2),
        ];
        assert!(try_vectorize_instructions(&instrs, 4).is_none());
    }

    #[test]
    fn test_simd_hint_fields() {
        let hint = SimdHint {
            loop_start_pc: 5,
            loop_end_pc: 20,
            width: VectorWidth::W256,
            pattern: VectorPattern::Map {
                op: VectorOp::Add,
                element_type: "f64".to_string(),
            },
            speedup: 3.5,
            induction_var: 2,
        };
        assert_eq!(hint.loop_start_pc, 5);
        assert_eq!(hint.loop_end_pc, 20);
        assert_eq!(hint.width, VectorWidth::W256);
        assert!((hint.speedup - 3.5).abs() < f64::EPSILON);
        assert_eq!(hint.induction_var, 2);
    }

    #[test]
    fn test_binop_to_vectorop() {
        assert_eq!(binop_to_vectorop(BinOpKind::Add), Some(VectorOp::Add));
        assert_eq!(binop_to_vectorop(BinOpKind::Sub), Some(VectorOp::Sub));
        assert_eq!(binop_to_vectorop(BinOpKind::Mul), Some(VectorOp::Mul));
        assert_eq!(binop_to_vectorop(BinOpKind::Div), Some(VectorOp::Div));
        assert_eq!(binop_to_vectorop(BinOpKind::Eq), None);
        assert_eq!(binop_to_vectorop(BinOpKind::And), None);
    }
}
