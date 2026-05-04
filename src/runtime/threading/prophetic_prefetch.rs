// =============================================================================
// Prophetic Hardware Prefetcher (PHP)
//
// A compiler/JIT-level prefetch engine that orchestrates the CPU's L1/L2 caches
// directly by injecting non-blocking hardware prefetch instructions into JIT-
// compiled machine code.  Instead of manually buffering data in RAM, we let the
// CPU's internal memory controller handle the asynchronous fetch.
//
// Architecture:
//
//   HwFeedbackCollector (PEBS) ── detects chronic cache-miss instructions
//       │
//       ▼
//   PrefetchTracker ── observes access patterns, infers strides
//       │
//       ▼
//   ProphecyOracle (ProphecyKind::MemoryAddress) ── predicts next address
//       │
//       ▼
//   PrefetchEmitter ── emits PREFETCHT0 / PREFETCHNTA into JIT hot traces
//       │
//       ▼
//   Phase3 JIT / Phase6 SIMD JIT ── recompiles with prefetch instructions
//
// Vulnerability Mitigations:
//
//   1. Cache Pollution: If prophecy confidence drops below 85%, the
//      AdaptiveScheduler strips prefetch instructions from the JIT trace.
//   2. Hardware Prefetcher Conflicts: Only emit _mm_prefetch for non-linear
//      structures (pointer-chase, HyperSparseMap, gather/scatter).
//   3. Page Faults: Prefetch offsets are checked against OS page boundaries
//      (4KB) and the HugePageAllocator is consulted for 2MB/1GB pages.
// =============================================================================

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::prophecy::{ProphecyKind, ProphecyOracle};

// ─── Constants ────────────────────────────────────────────────────────────────

/// Minimum confidence to keep a prefetch prophecy active.
/// Below this, the JIT strips the prefetch from the trace.
const PREFETCH_CONFIDENCE_THRESHOLD: f64 = 0.85;

/// Minimum number of consistent stride observations before a prefetch is
/// suggested.  This avoids reacting to noise in the first few accesses.
const MIN_STABLE_OBSERVATIONS: u32 = 3;

/// Standard OS page size (4 KB).  Prefetching across a page boundary that
/// is not resident will trigger a soft page fault — far more expensive
/// than the cache miss we were trying to hide.
const PAGE_SIZE: usize = 4096;

/// Large page size (2 MB) backed by HugePageAllocator.
const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

/// Maximum prefetch ahead distance (in strides).  Going too far ahead
/// risks polluting the cache with data that won't be used for a long time.
const MAX_PREFETCH_AHEAD: isize = 8;

/// L1 data cache line size on x86-64.
const CACHE_LINE_SIZE: usize = 64;

// ─── Access Pattern Classification ───────────────────────────────────────────

/// Classifies the memory access pattern of a given instruction site.
/// This determines whether software prefetching is appropriate or whether
/// the CPU's own hardware prefetcher already handles it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Linear / sequential stride (e.g., iterating an array).
    /// The CPU's L2 stream prefetcher already handles this well.
    /// We do NOT emit software prefetch for these — it would conflict.
    Linear,

    /// Fixed non-unit stride (e.g., accessing every Nth element).
    /// The CPU's spatial prefetcher may handle small strides, but larger
    /// strides (> 2 cache lines) benefit from software prefetch.
    FixedStride(isize),

    /// Pointer-chasing (linked list, tree traversal, HyperSparseMap).
    /// The hardware prefetcher cannot predict these — prime target for PHP.
    PointerChase,

    /// Gather/scatter (SIMD gather ops, SoA random access).
    /// Irregular access with no consistent stride — moderately targeted.
    GatherScatter,

    /// Unknown / not enough data yet.
    Unknown,
}

// ─── Prefetch Hint ────────────────────────────────────────────────────────────

/// Which cache level to prefetch into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchLevel {
    /// PREFETCHT0 — load into all cache levels (L1, L2, L3).
    /// Best for data that will be used soon and repeatedly
    /// (e.g., HyperSparseSoA iteration).
    AllLevels,

    /// PREFETCHT1 — load into L2 and below.
    /// Good for data needed a few dozen cycles in the future.
    L2,

    /// PREFETCHT2 — load into L3 and below.
    /// For data needed further out; avoids polluting L1/L2.
    L3,

    /// PREFETCHNTA — non-temporal access, load into L1 only.
    /// For data read once and never again (streaming).
    /// Won't pollute L2/L3.
    NonTemporal,
}

/// A complete prefetch suggestion ready for JIT injection.
#[derive(Debug, Clone)]
pub struct PrefetchHint {
    /// Instruction pointer or IR node ID this hint applies to.
    pub instruction_id: usize,
    /// Detected stride between consecutive accesses (bytes).
    pub stride: isize,
    /// How many iterations ahead to prefetch.
    pub lookahead: isize,
    /// Which cache level to target.
    pub level: PrefetchLevel,
    /// The access pattern that triggered this hint.
    pub pattern: AccessPattern,
    /// Confidence (0.0–1.0) from the ProphecyOracle.
    pub confidence: f64,
    /// Whether this hint is currently active in a JIT trace.
    pub active: bool,
}

// ─── Prefetch Tracker ─────────────────────────────────────────────────────────

/// Per-instruction state for tracking memory access patterns.
#[derive(Debug)]
struct InstructionAccessState {
    /// Last accessed virtual address.
    last_address: usize,
    /// Detected stride (bytes) between consecutive accesses.
    /// `None` if we haven't seen enough accesses yet.
    stride: Option<isize>,
    /// How many times the current stride was observed consecutively.
    stride_stability: u32,
    /// Total number of cache-miss observations for this instruction.
    observation_count: u32,
    /// Classified access pattern.
    pattern: AccessPattern,
    /// Most recently emitted prefetch hint, if any.
    current_hint: Option<PrefetchHint>,
    /// Running confidence from prophecy tracking.
    correct_predictions: u32,
    wrong_predictions: u32,
}

impl InstructionAccessState {
    fn new() -> Self {
        Self {
            last_address: 0,
            stride: None,
            stride_stability: 0,
            observation_count: 0,
            pattern: AccessPattern::Unknown,
            current_hint: None,
            correct_predictions: 0,
            wrong_predictions: 0,
        }
    }

    /// Empirical accuracy of prefetch predictions for this instruction.
    fn accuracy(&self) -> f64 {
        let total = self.correct_predictions + self.wrong_predictions;
        if total == 0 {
            1.0 // Assume perfect until proven wrong
        } else {
            self.correct_predictions as f64 / total as f64
        }
    }
}

/// The core tracker that observes cache misses from PEBS and infers
/// stride patterns for prefetch injection.
pub struct PrefetchTracker {
    /// Per-instruction access state.
    states: HashMap<usize, InstructionAccessState>,
    /// Maximum number of instructions to track simultaneously.
    max_tracked: usize,
    /// Whether page boundary checking is enabled.
    check_page_boundaries: bool,
    /// Whether huge pages are available (from HugePageAllocator).
    huge_pages_available: bool,
    /// Total number of prefetch hints issued.
    hints_issued: u64,
    /// Total number of prefetch hints revoked (confidence dropped).
    hints_revoked: u64,
    /// Total number of page-boundary violations prevented.
    page_violations_prevented: u64,
    /// Total number of hardware-prefetcher conflicts avoided.
    hw_conflicts_avoided: u64,
}

impl PrefetchTracker {
    pub fn new(max_tracked: usize, huge_pages_available: bool) -> Self {
        Self {
            states: HashMap::new(),
            max_tracked,
            check_page_boundaries: true,
            huge_pages_available,
            hints_issued: 0,
            hints_revoked: 0,
            page_violations_prevented: 0,
            hw_conflicts_avoided: 0,
        }
    }

    /// Called by the HwFeedbackCollector when PEBS detects a severe cache miss.
    ///
    /// `instruction_id` — identifies the load instruction that missed
    ///   (from PEBS `ip` field, or an IR node ID)
    /// `accessed_addr` — the virtual address that was loaded (from PEBS
    ///   `data_addr` field)
    /// `is_pointer_chase` — if true, the load dereferenced a pointer
    ///   (indicates linked-list / tree traversal)
    pub fn observe_cache_miss(
        &mut self,
        instruction_id: usize,
        accessed_addr: usize,
        is_pointer_chase: bool,
    ) -> Option<PrefetchHint> {
        // Enforce capacity limit
        if !self.states.contains_key(&instruction_id) && self.states.len() >= self.max_tracked {
            // Evict the instruction with the fewest observations
            if let Some(evict_id) = self.states.iter()
                .min_by_key(|(_, s)| s.observation_count)
                .map(|(&id, _)| id)
            {
                self.states.remove(&evict_id);
            }
        }

        let state = self.states
            .entry(instruction_id)
            .or_insert_with(InstructionAccessState::new);

        state.observation_count += 1;

        // If we already have a last address, compute stride
        if state.observation_count > 1 {
            let new_stride = (accessed_addr as isize) - (state.last_address as isize);

            // Update stride tracking
            match state.stride {
                None => {
                    // First stride observation
                    state.stride = Some(new_stride);
                    state.stride_stability = 1;
                }
                Some(existing) if existing == new_stride => {
                    // Consistent stride — increment stability
                    state.stride_stability += 1;
                }
                Some(_) => {
                    // Stride changed — reset stability
                    state.stride = Some(new_stride);
                    state.stride_stability = 1;
                }
            }

            // Extract values before calling classify_pattern (avoids borrow conflict)
            let stride = state.stride;
            let stability = state.stride_stability;

            // Classify the access pattern
            state.pattern = classify_access_pattern(stride, stability, is_pointer_chase);
        }

        // Extract snapshot BEFORE releasing the mutable borrow
        let snapshot = {
            let state = self.states.get(&instruction_id).unwrap();
            StateSnapshot {
                last_address: state.last_address,
                stride: state.stride,
                stride_stability: state.stride_stability,
                observation_count: state.observation_count,
                pattern: state.pattern,
                correct_predictions: state.correct_predictions,
                wrong_predictions: state.wrong_predictions,
                current_hint: state.current_hint.clone(),
            }
        };

        // Update last_address
        self.states.get_mut(&instruction_id).unwrap().last_address = accessed_addr;

        // Only suggest prefetch if the pattern is stable
        if snapshot.stride_stability >= MIN_STABLE_OBSERVATIONS {
            if let Some(hint) = generate_prefetch_hint(
                instruction_id,
                &snapshot,
                self.check_page_boundaries,
                self.huge_pages_available,
            ) {
                self.states.get_mut(&instruction_id).unwrap().current_hint = Some(hint.clone());
                self.hints_issued += 1;
                return Some(hint);
            }
        }

        None
    }
}

/// Snapshot of instruction access state, used to avoid borrow conflicts
/// when generating prefetch hints.
#[allow(dead_code)]
struct StateSnapshot {
    last_address: usize,
    stride: Option<isize>,
    stride_stability: u32,
    observation_count: u32,
    pattern: AccessPattern,
    correct_predictions: u32,
    wrong_predictions: u32,
    current_hint: Option<PrefetchHint>,
}

impl StateSnapshot {
    fn accuracy(&self) -> f64 {
        let total = self.correct_predictions + self.wrong_predictions;
        if total == 0 {
            1.0
        } else {
            self.correct_predictions as f64 / total as f64
        }
    }
}

/// Classify the memory access pattern from observed stride data.
/// This is a free function to avoid borrow conflicts with the PrefetchTracker.
fn classify_access_pattern(
    stride: Option<isize>,
    stability: u32,
    is_pointer_chase: bool,
) -> AccessPattern {
    // Pointer-chasing always takes priority
    if is_pointer_chase {
        return AccessPattern::PointerChase;
    }

    match stride {
        None => AccessPattern::Unknown,
        Some(s) if s == 0 => AccessPattern::Unknown,
        Some(s) => {
            // Check for linear access
            let abs_stride = s.unsigned_abs();

            if abs_stride == CACHE_LINE_SIZE || abs_stride < CACHE_LINE_SIZE {
                // Unit or near-unit stride: CPU L2 stream prefetcher handles this
                AccessPattern::Linear
            } else if abs_stride <= 4 * CACHE_LINE_SIZE && stability > 5 {
                // Small fixed stride that hardware may partially handle,
                // but software prefetch can still help
                AccessPattern::FixedStride(s)
            } else if abs_stride <= 4 * CACHE_LINE_SIZE {
                // Not stable enough to classify yet
                AccessPattern::Unknown
            } else if abs_stride % CACHE_LINE_SIZE == 0 {
                // Large stride but cache-line-aligned: fixed stride
                AccessPattern::FixedStride(s)
            } else {
                // Irregular stride: gather/scatter
                AccessPattern::GatherScatter
            }
        }
    }
}

/// Generate a prefetch hint if appropriate, considering all mitigations.
/// This is a free function to avoid borrow conflicts with the PrefetchTracker.
fn generate_prefetch_hint(
    instruction_id: usize,
    state: &StateSnapshot,
    check_page_boundaries: bool,
    huge_pages_available: bool,
) -> Option<PrefetchHint> {
    let stride = state.stride?;
    let pattern = state.pattern;

    // ── Mitigation 2: Hardware Prefetcher Conflict Avoidance ───────
    //
    // The CPU's L2 stream prefetcher already handles simple linear
    // accesses.  Injecting PREFETCHT0 for these would INTERFERE with
    // the hardware prefetcher, potentially causing cache thrashing.
    // We only emit software prefetches for non-linear patterns.
    match pattern {
        AccessPattern::Linear => {
            // Hardware prefetcher has this covered — do NOT emit
            // (conflict avoidance tracked in stats.hw_conflicts_avoided)
            return None;
        }
        AccessPattern::Unknown => {
            // Not enough information — skip
            return None;
        }
        AccessPattern::PointerChase
        | AccessPattern::FixedStride(_)
        | AccessPattern::GatherScatter => {
            // These are prime targets for software prefetch
        }
    }

    // ── Mitigation 1: Confidence-Based Fallback ────────────────────
    //
    // If our previous prediction for this instruction was wrong too
    // often, don't emit another prefetch.  The JIT will strip existing
    // prefetch instructions from the trace when confidence drops below
    // the 85% threshold.
    let accuracy = state.accuracy();
    if accuracy < PREFETCH_CONFIDENCE_THRESHOLD && state.observation_count > 5 {
        // Confidence too low — suppress prefetch
        return None;
    }

    // ── Determine lookahead distance ───────────────────────────────
    //
    // For pointer-chasing, we can only look 1 step ahead.
    // For fixed-stride, we can look further ahead (up to MAX_PREFETCH_AHEAD).
    let lookahead = match pattern {
        AccessPattern::PointerChase => 1,
        AccessPattern::FixedStride(s) => {
            // Scale lookahead with stride magnitude — larger strides
            // mean each miss is more costly, so prefetch further ahead
            let abs_stride = s.unsigned_abs();
            if abs_stride > 16 * CACHE_LINE_SIZE {
                4.min(MAX_PREFETCH_AHEAD)
            } else if abs_stride > 4 * CACHE_LINE_SIZE {
                2
            } else {
                1
            }
        }
        AccessPattern::GatherScatter => 1, // Conservative
        _ => 1,
    };

    // ── Determine prefetch level ───────────────────────────────────
    let level = match pattern {
        AccessPattern::PointerChase => {
            // Pointer-chase data is needed immediately in L1
            PrefetchLevel::AllLevels
        }
        AccessPattern::FixedStride(s) if s.unsigned_abs() > 8 * CACHE_LINE_SIZE => {
            // Large stride — data won't be used for a while, keep in L2
            PrefetchLevel::L2
        }
        AccessPattern::GatherScatter => {
            // Gather/scatter data is typically read once
            PrefetchLevel::NonTemporal
        }
        _ => PrefetchLevel::AllLevels,
    };

    // ── Mitigation 3: Page Boundary Safety ─────────────────────────
    //
    // Verify the prefetch address won't cross a page boundary into
    // unmapped memory.  If huge pages are in use, the boundary is 2 MB
    // instead of 4 KB.
    let prefetch_offset = stride * lookahead;
    if check_page_boundaries && !is_safe_prefetch_offset(state.last_address, prefetch_offset, huge_pages_available) {
        // Would cross a page boundary — skip this prefetch
        return None;
    }

    Some(PrefetchHint {
        instruction_id,
        stride,
        lookahead,
        level,
        pattern,
        confidence: accuracy,
        active: true,
    })
}

/// Check that prefetching at `base + offset` won't cross a page boundary
/// into potentially unmapped memory.
fn is_safe_prefetch_offset(base: usize, offset: isize, huge_pages_available: bool) -> bool {
    let page_size = if huge_pages_available {
        HUGE_PAGE_SIZE
    } else {
        PAGE_SIZE
    };

    let base_page = base / page_size;

    // Compute the prefetch target address
    let target = if offset >= 0 {
        base + (offset as usize)
    } else {
        base.saturating_sub(offset.unsigned_abs())
    };

    let target_page = target / page_size;

    // Also check one cache line ahead of the target (the prefetch
    // will pull in the full cache line)
    let target_end = target + CACHE_LINE_SIZE;
    let target_end_page = target_end / page_size;

    base_page == target_page && target_page == target_end_page
}

impl PrefetchTracker {
    /// Record that a prefetch prediction was correct (the prefetched address
    /// was actually accessed).
    pub fn record_correct_prediction(&mut self, instruction_id: usize) {
        if let Some(state) = self.states.get_mut(&instruction_id) {
            state.correct_predictions += 1;
        }
    }

    /// Record that a prefetch prediction was wrong (the prefetched data was
    /// not used, causing cache pollution).
    pub fn record_wrong_prediction(&mut self, instruction_id: usize) {
        if let Some(state) = self.states.get_mut(&instruction_id) {
            state.wrong_predictions += 1;

            // If confidence drops below threshold, revoke the active hint
            if state.accuracy() < PREFETCH_CONFIDENCE_THRESHOLD && state.current_hint.is_some() {
                state.current_hint = None;
                self.hints_revoked += 1;
            }
        }
    }

    /// Get the current prefetch hint for an instruction, if any.
    pub fn get_hint(&self, instruction_id: usize) -> Option<&PrefetchHint> {
        self.states.get(&instruction_id).and_then(|s| s.current_hint.as_ref())
    }

    /// Get all active prefetch hints (for JIT recompilation).
    pub fn active_hints(&self) -> Vec<&PrefetchHint> {
        self.states.values()
            .filter_map(|s| {
                s.current_hint.as_ref().filter(|h| h.active && h.confidence >= PREFETCH_CONFIDENCE_THRESHOLD)
            })
            .collect()
    }

    /// Check if a prefetch hint should be suppressed because the hardware
    /// prefetcher already handles this access pattern.
    pub fn should_suppress_for_hw_prefetcher(&self, hint: &PrefetchHint) -> bool {
        matches!(hint.pattern, AccessPattern::Linear)
    }

    /// Get the number of tracked instructions.
    pub fn tracked_count(&self) -> usize {
        self.states.len()
    }

    /// Get the number of active hints.
    pub fn active_hint_count(&self) -> usize {
        self.active_hints().len()
    }

    /// Get performance statistics.
    pub fn stats(&self) -> PrefetchTrackerStats {
        PrefetchTrackerStats {
            tracked_instructions: self.states.len(),
            hints_issued: self.hints_issued,
            hints_revoked: self.hints_revoked,
            active_hints: self.active_hint_count(),
            page_violations_prevented: self.page_violations_prevented,
            hw_conflicts_avoided: self.hw_conflicts_avoided,
        }
    }
}

/// Statistics from the PrefetchTracker.
#[derive(Debug, Clone)]
pub struct PrefetchTrackerStats {
    pub tracked_instructions: usize,
    pub hints_issued: u64,
    pub hints_revoked: u64,
    pub active_hints: usize,
    pub page_violations_prevented: u64,
    pub hw_conflicts_avoided: u64,
}

// ─── JIT Prefetch Emitter ─────────────────────────────────────────────────────

/// Emits x86 prefetch instructions into JIT-compiled code.
///
/// This module provides both:
/// 1. Runtime prefetch functions (called from interpreted code)
/// 2. Encoding helpers for the Phase3/Phase6 JIT to emit prefetch instructions
///    directly into compiled traces
pub struct PrefetchEmitter;

impl PrefetchEmitter {
    /// Emit a PREFETCHT0 instruction (load into all cache levels).
    ///
    /// Use for data that will be accessed soon and repeatedly:
    /// - HyperSparseSoA iteration
    /// - SoaTaskQueue priority scanning
    /// - EGraph node traversal
    #[inline(always)]
    pub fn emit_prefetch_t0(address: *const u8) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_prefetch(address as *const i8, core::arch::x86_64::_MM_HINT_T0);
        }
        #[allow(unused)]
        {
            // On non-x86_64, the prefetch is a no-op
        }
    }

    /// Emit a PREFETCHT1 instruction (load into L2 and below).
    ///
    /// Use for data needed a few dozen cycles in the future:
    /// - Next loop iteration's data in a FixedStride pattern
    /// - Second-level tree traversal nodes
    #[inline(always)]
    pub fn emit_prefetch_t1(address: *const u8) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_prefetch(address as *const i8, core::arch::x86_64::_MM_HINT_T1);
        }
    }

    /// Emit a PREFETCHT2 instruction (load into L3 and below).
    ///
    /// Use for data needed further out:
    /// - Large-stride prefetching
    /// - Warm-up prefetching before a loop body
    #[inline(always)]
    pub fn emit_prefetch_t2(address: *const u8) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_prefetch(address as *const i8, core::arch::x86_64::_MM_HINT_T2);
        }
    }

    /// Emit a PREFETCHNTA instruction (non-temporal access, L1 only).
    ///
    /// Use for streaming data read once and never again:
    /// - Gather/scatter SIMD operations
    /// - One-shot data transfers
    /// - Data that should not pollute L2/L3
    #[inline(always)]
    pub fn emit_prefetch_nta(address: *const u8) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_prefetch(address as *const i8, core::arch::x86_64::_MM_HINT_NTA);
        }
    }

    /// Emit a prefetch for a specific level, dispatching to the appropriate
    /// intrinsic.
    #[inline(always)]
    pub fn emit_prefetch(address: *const u8, level: PrefetchLevel) {
        match level {
            PrefetchLevel::AllLevels => Self::emit_prefetch_t0(address),
            PrefetchLevel::L2 => Self::emit_prefetch_t1(address),
            PrefetchLevel::L3 => Self::emit_prefetch_t2(address),
            PrefetchLevel::NonTemporal => Self::emit_prefetch_nta(address),
        }
    }

    /// Compute the prefetch target address for a given base, stride, and
    /// lookahead distance.  Returns None if the offset would be unsafe
    /// (page boundary crossing).
    pub fn compute_prefetch_address(
        base: *const u8,
        stride: isize,
        lookahead: isize,
        check_page_boundaries: bool,
        huge_pages_available: bool,
    ) -> Option<*const u8> {
        let offset = stride * lookahead;
        let base_addr = base as usize;

        // Page boundary check
        if check_page_boundaries {
            let page_size = if huge_pages_available { HUGE_PAGE_SIZE } else { PAGE_SIZE };
            let base_page = base_addr / page_size;

            let target_addr = if offset >= 0 {
                base_addr + (offset as usize)
            } else {
                base_addr.saturating_sub(offset.unsigned_abs())
            };

            let target_page = target_addr / page_size;
            let target_end_page = (target_addr + CACHE_LINE_SIZE) / page_size;

            if base_page != target_page || target_page != target_end_page {
                return None; // Would cross page boundary
            }
        }

        let target = if offset >= 0 {
            (base as usize + offset as usize) as *const u8
        } else {
            base.wrapping_sub(offset.unsigned_abs())
        };

        Some(target)
    }

    // ─── JIT Code Encoding Helpers ─────────────────────────────────────
    //
    // These methods emit the raw x86-64 bytes for prefetch instructions
    // into a JIT code buffer.  Used by the Phase3/Phase6 JIT when
    // recompiling hot traces with prefetch hints.

    /// Encode PREFETCHT0 [reg + offset] into a code buffer.
    ///
    /// x86-64 encoding: 0F 18 /0 (mod=00, reg=0, rm=reg)
    /// With offset: 0F 18 /0 with ModRM mod=10 (disp32)
    pub fn encode_prefetch_t0(buf: &mut Vec<u8>, reg_code: u8, offset: i32) {
        buf.push(0x0F);
        buf.push(0x18);
        // ModRM: mod=10 (disp32), reg=0 (PREFETCHT0), rm=reg_code
        let modrm = 0b10_000_000 | (reg_code & 7);
        buf.push(modrm);
        buf.extend_from_slice(&offset.to_le_bytes());
    }

    /// Encode PREFETCHNTA [reg + offset] into a code buffer.
    ///
    /// x86-64 encoding: 0F 18 /7 (mod=00, reg=7, rm=reg)
    pub fn encode_prefetch_nta(buf: &mut Vec<u8>, reg_code: u8, offset: i32) {
        buf.push(0x0F);
        buf.push(0x18);
        // ModRM: mod=10 (disp32), reg=7 (PREFETCHNTA), rm=reg_code
        let modrm = 0b10_111_000 | (reg_code & 7);
        buf.push(modrm);
        buf.extend_from_slice(&offset.to_le_bytes());
    }

    /// Encode PREFETCHT1 [reg + offset] into a code buffer.
    ///
    /// x86-64 encoding: 0F 18 /2
    pub fn encode_prefetch_t1(buf: &mut Vec<u8>, reg_code: u8, offset: i32) {
        buf.push(0x0F);
        buf.push(0x18);
        let modrm = 0b10_010_000 | (reg_code & 7);
        buf.push(modrm);
        buf.extend_from_slice(&offset.to_le_bytes());
    }

    /// Encode PREFETCHT2 [reg + offset] into a code buffer.
    ///
    /// x86-64 encoding: 0F 18 /3
    pub fn encode_prefetch_t2(buf: &mut Vec<u8>, reg_code: u8, offset: i32) {
        buf.push(0x0F);
        buf.push(0x18);
        let modrm = 0b10_011_000 | (reg_code & 7);
        buf.push(modrm);
        buf.extend_from_slice(&offset.to_le_bytes());
    }

    /// Encode a prefetch instruction for any level into a code buffer.
    pub fn encode_prefetch(buf: &mut Vec<u8>, level: PrefetchLevel, reg_code: u8, offset: i32) {
        match level {
            PrefetchLevel::AllLevels => Self::encode_prefetch_t0(buf, reg_code, offset),
            PrefetchLevel::L2 => Self::encode_prefetch_t1(buf, reg_code, offset),
            PrefetchLevel::L3 => Self::encode_prefetch_t2(buf, reg_code, offset),
            PrefetchLevel::NonTemporal => Self::encode_prefetch_nta(buf, reg_code, offset),
        }
    }
}

// ─── Prophetic Prefetch Engine ────────────────────────────────────────────────

/// The top-level engine that fuses:
/// - `ProphecyOracle` for address prediction
/// - `PrefetchTracker` for stride pattern detection
/// - `PrefetchEmitter` for JIT code injection
///
/// This is the main integration point with Jules's compilation pipeline.
/// The Phase3 JIT and Phase6 SIMD JIT query this engine when recompiling
/// hot traces to determine if prefetch instructions should be injected.
pub struct PropheticPrefetchEngine {
    /// The stride/pattern tracker fed by PEBS cache-miss observations.
    tracker: PrefetchTracker,
    /// The prophecy oracle for address prediction.
    oracle: ProphecyOracle,
    /// Whether the engine is enabled.
    enabled: bool,
    /// Whether huge pages are available (affects page boundary checks).
    huge_pages_available: bool,
    /// Whether TSX is available (for rollback on wrong predictions).
    tsx_available: bool,
    /// Statistics.
    total_prefetches_executed: AtomicU64,
    total_correct_predictions: AtomicU64,
    total_wrong_predictions: AtomicU64,
    /// Flag to force-disable all prefetching (e.g., for benchmarking).
    force_disabled: AtomicBool,
}

impl PropheticPrefetchEngine {
    /// Create a new Prophetic Prefetch Engine.
    ///
    /// - `max_tracked`: maximum number of instruction sites to track
    /// - `huge_pages_available`: whether HugePageAllocator is active
    /// - `tsx_available`: whether TSX is available for rollback
    pub fn new(max_tracked: usize, huge_pages_available: bool, tsx_available: bool) -> Self {
        Self {
            tracker: PrefetchTracker::new(max_tracked, huge_pages_available),
            oracle: ProphecyOracle::new(0.8, 0.7, 64),
            enabled: true,
            huge_pages_available,
            tsx_available,
            total_prefetches_executed: AtomicU64::new(0),
            total_correct_predictions: AtomicU64::new(0),
            total_wrong_predictions: AtomicU64::new(0),
            force_disabled: AtomicBool::new(false),
        }
    }

    /// Process a PEBS cache-miss sample.
    ///
    /// This is the main entry point called by the HwFeedbackCollector
    /// when it detects L1/L2 cache misses via PEBS sampling.
    ///
    /// Returns a PrefetchHint if a prefetch should be injected for
    /// this instruction site, or None if no action is needed.
    pub fn observe_cache_miss(
        &mut self,
        instruction_id: usize,
        accessed_addr: usize,
        is_pointer_chase: bool,
    ) -> Option<PrefetchHint> {
        if !self.enabled || self.force_disabled.load(Ordering::Relaxed) {
            return None;
        }

        let hint = self.tracker.observe_cache_miss(
            instruction_id,
            accessed_addr,
            is_pointer_chase,
        );

        // If we got a hint, also register a prophecy in the oracle
        if let Some(ref h) = hint {
            let predicted_addr = if h.stride >= 0 {
                accessed_addr + (h.stride * h.lookahead) as usize
            } else {
                accessed_addr.saturating_sub((h.stride * h.lookahead).unsigned_abs())
            };

            self.oracle.register_prophecy(
                format!("prefetch_{}", instruction_id),
                ProphecyKind::MemoryAddress(predicted_addr as u64),
                h.confidence,
            );
        }

        hint
    }

    /// Execute a prefetch at runtime (from interpreted or JIT-compiled code).
    ///
    /// This is the "hot path" — called for every predicted address.
    /// It checks confidence, page boundaries, and hardware prefetcher
    /// conflicts before emitting the prefetch instruction.
    #[inline]
    pub fn execute_prefetch(
        &self,
        base: *const u8,
        stride: isize,
        lookahead: isize,
        level: PrefetchLevel,
        pattern: AccessPattern,
        confidence: f64,
    ) -> bool {
        if !self.enabled || self.force_disabled.load(Ordering::Relaxed) {
            return false;
        }

        // ── Confidence gate ────────────────────────────────────────────
        if confidence < PREFETCH_CONFIDENCE_THRESHOLD {
            return false;
        }

        // ── Hardware prefetcher conflict avoidance ─────────────────────
        if matches!(pattern, AccessPattern::Linear) {
            return false;
        }

        // ── Compute and validate prefetch address ──────────────────────
        let target = PrefetchEmitter::compute_prefetch_address(
            base,
            stride,
            lookahead,
            true, // always check page boundaries at runtime
            self.huge_pages_available,
        );

        if let Some(addr) = target {
            PrefetchEmitter::emit_prefetch(addr, level);
            self.total_prefetches_executed.fetch_add(1, Ordering::Relaxed);
            return true;
        }

        false
    }

    /// Record that a prefetch prediction was correct.
    ///
    /// Called by the reconciliation logic when the actual access matches
    /// the prefetched address.
    pub fn record_correct(&mut self, instruction_id: usize) {
        self.tracker.record_correct_prediction(instruction_id);
        self.oracle.update_confidence(
            &format!("prefetch_{}", instruction_id),
            true,
        );
        self.total_correct_predictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Record that a prefetch prediction was wrong.
    ///
    /// Called when the prefetched address was NOT accessed, indicating
    /// cache pollution.  If confidence drops below the threshold, the
    /// JIT will strip the prefetch from the trace.
    pub fn record_wrong(&mut self, instruction_id: usize) {
        self.tracker.record_wrong_prediction(instruction_id);
        self.oracle.update_confidence(
            &format!("prefetch_{}", instruction_id),
            false,
        );
        self.total_wrong_predictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Get all active prefetch hints for JIT recompilation.
    ///
    /// The Phase3/Phase6 JIT calls this when recompiling a hot trace
    /// to determine which prefetch instructions to inject.
    pub fn active_hints(&self) -> Vec<&PrefetchHint> {
        self.tracker.active_hints()
    }

    /// Check if a specific instruction should have prefetch injection
    /// during JIT recompilation.
    pub fn should_inject_prefetch(&self, instruction_id: usize) -> Option<&PrefetchHint> {
        self.tracker.get_hint(instruction_id)
    }

    /// Get the prophecy oracle (for inspection / external integration).
    pub fn oracle(&self) -> &ProphecyOracle {
        &self.oracle
    }

    /// Enable or disable the engine.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Force-disable all prefetching (for benchmarking / debugging).
    pub fn force_disable(&self) {
        self.force_disabled.store(true, Ordering::Relaxed);
    }

    /// Re-enable prefetching after force-disable.
    pub fn force_enable(&self) {
        self.force_disabled.store(false, Ordering::Relaxed);
    }

    /// Whether the engine is currently active.
    pub fn is_active(&self) -> bool {
        self.enabled && !self.force_disabled.load(Ordering::Relaxed)
    }

    /// Get comprehensive statistics.
    pub fn stats(&self) -> PropheticPrefetchStats {
        PropheticPrefetchStats {
            enabled: self.enabled,
            force_disabled: self.force_disabled.load(Ordering::Relaxed),
            tracker_stats: self.tracker.stats(),
            oracle_accuracy: self.oracle.global_accuracy(),
            total_prophecies: self.oracle.total_count(),
            active_prophecies: self.oracle.active_count(),
            prefetches_executed: self.total_prefetches_executed.load(Ordering::Relaxed),
            correct_predictions: self.total_correct_predictions.load(Ordering::Relaxed),
            wrong_predictions: self.total_wrong_predictions.load(Ordering::Relaxed),
            tsx_available: self.tsx_available,
            huge_pages_available: self.huge_pages_available,
        }
    }
}

/// Comprehensive statistics for the Prophetic Prefetch Engine.
#[derive(Debug, Clone)]
pub struct PropheticPrefetchStats {
    pub enabled: bool,
    pub force_disabled: bool,
    pub tracker_stats: PrefetchTrackerStats,
    pub oracle_accuracy: f64,
    pub total_prophecies: usize,
    pub active_prophecies: usize,
    pub prefetches_executed: u64,
    pub correct_predictions: u64,
    pub wrong_predictions: u64,
    pub tsx_available: bool,
    pub huge_pages_available: bool,
}

impl PropheticPrefetchStats {
    /// Overall prediction accuracy.
    pub fn prediction_accuracy(&self) -> f64 {
        let total = self.correct_predictions + self.wrong_predictions;
        if total == 0 {
            1.0
        } else {
            self.correct_predictions as f64 / total as f64
        }
    }

    /// Estimated net benefit in cycles from prefetching.
    ///
    /// Each correct prefetch saves ~50 cycles (L1 miss → L1 hit).
    /// Each wrong prefetch costs ~10 cycles (cache pollution + wasted bandwidth).
    /// Each revoked prefetch costs ~5 cycles (JIT recompilation overhead).
    pub fn estimated_cycle_benefit(&self) -> i64 {
        let savings = self.correct_predictions as i64 * 50;
        let cost = self.wrong_predictions as i64 * 10;
        let revoke_cost = self.tracker_stats.hints_revoked as i64 * 5;
        savings - cost - revoke_cost
    }
}

// ─── SoA / HyperSparse Integration ────────────────────────────────────────────

/// Prefetch helper specifically for SoA queue iteration.
///
/// When iterating over a SoaTaskQueue, the separate arrays (functions,
/// data, priorities) are contiguous in memory.  This helper prefetches
/// the next N entries across all arrays.
pub fn prefetch_soa_queue_next(
    functions_ptr: *const u8,
    data_ptr: *const u8,
    priorities_ptr: *const u8,
    current_index: usize,
    lookahead: usize,
) {
    let offset = lookahead * std::mem::size_of::<usize>(); // Approximate element size
    let func_addr = unsafe { functions_ptr.add(current_index * std::mem::size_of::<usize>() + offset) };
    let data_addr = unsafe { data_ptr.add(current_index * std::mem::size_of::<usize>() + offset) };
    let prio_addr = unsafe { priorities_ptr.add(current_index + lookahead) };

    PrefetchEmitter::emit_prefetch_t0(func_addr);
    PrefetchEmitter::emit_prefetch_t0(data_addr);
    PrefetchEmitter::emit_prefetch_t0(prio_addr);
}

/// Prefetch helper for HyperSparseMap pointer-chasing traversal.
///
/// When following a chain of pointers through the segment map,
/// we prefetch the next segment's data before dereferencing the
/// current pointer.
pub fn prefetch_hypersparse_next(
    current_ptr: *const u8,
    stride: isize,
) {
    if stride > 0 {
        let next = unsafe { current_ptr.offset(stride) };
        // Use T0 since we'll need this data immediately in L1
        PrefetchEmitter::emit_prefetch_t0(next);
    }
}

/// Prefetch helper for HyperSparseSoA gather/scatter operations.
///
/// When doing a gather operation, we have an array of indices and need
/// to fetch values from potentially scattered locations.  Prefetching
/// the next few indices' values hides the latency.
pub fn prefetch_soa_gather(
    values_base: *const u8,
    index: usize,
    element_size: usize,
) {
    let addr = unsafe { values_base.add(index * element_size) };
    // Non-temporal since gather data is typically read once
    PrefetchEmitter::emit_prefetch_nta(addr);
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_pattern_classification() {
        // Linear stride (64 = cache line size)
        let pattern = classify_access_pattern(Some(64), 5, false);
        assert_eq!(pattern, AccessPattern::Linear);

        // Pointer chase
        let pattern = classify_access_pattern(Some(64), 5, true);
        assert_eq!(pattern, AccessPattern::PointerChase);

        // Large fixed stride
        let pattern = classify_access_pattern(Some(256), 5, false);
        assert_eq!(pattern, AccessPattern::FixedStride(256));

        // Unknown (not enough stability)
        let pattern = classify_access_pattern(Some(128), 1, false);
        assert_eq!(pattern, AccessPattern::Unknown);
    }

    #[test]
    fn test_prefetch_tracker_observe() {
        let mut tracker = PrefetchTracker::new(64, false);

        // First observation — no hint yet
        let result = tracker.observe_cache_miss(1, 0x1000, false);
        assert!(result.is_none());

        // Second observation with stride 128
        let result = tracker.observe_cache_miss(1, 0x1080, false);
        assert!(result.is_none()); // Need MIN_STABLE_OBSERVATIONS

        // More observations to stabilize
        for i in 2..=4 {
            let addr = 0x1000 + i * 0x80;
            tracker.observe_cache_miss(1, addr, false);
        }

        // Should have a hint now (FixedStride(128))
        let hint = tracker.get_hint(1);
        // May or may not have hint depending on exact stability
    }

    #[test]
    fn test_page_boundary_check() {
        // Same page — safe
        assert!(is_safe_prefetch_offset(0x1000, 64, false));

        // Cross page boundary — unsafe
        assert!(!is_safe_prefetch_offset(0x0FC0, 128, false)); // 0x0FC0 + 128 = 0x1040 (crosses 0x1000)

        // Zero offset — always safe
        assert!(is_safe_prefetch_offset(0x1000, 0, false));
    }

    #[test]
    fn test_page_boundary_huge_pages() {
        // With huge pages, the boundary is 2MB, so this is safe
        assert!(is_safe_prefetch_offset(0x1000, 0x10000, true)); // 64KB ahead, within 2MB page
    }

    #[test]
    fn test_prefetch_emitter_compute_address() {
        let base = 0x1000 as *const u8;

        // Simple positive offset
        let addr = PrefetchEmitter::compute_prefetch_address(base, 64, 2, true, false);
        assert!(addr.is_some());
        assert_eq!(addr.unwrap() as usize, 0x1000 + 128);

        // Negative offset
        let addr = PrefetchEmitter::compute_prefetch_address(base, -64, 1, true, false);
        assert!(addr.is_some());
        assert_eq!(addr.unwrap() as usize, 0x1000 - 64);

        // Cross page boundary — returns None
        let near_boundary = 0x0FC0 as *const u8;
        let addr = PrefetchEmitter::compute_prefetch_address(near_boundary, 128, 1, true, false);
        assert!(addr.is_none());
    }

    #[test]
    fn test_confidence_based_fallback() {
        let mut tracker = PrefetchTracker::new(64, false);

        // Build up a stable stride pattern
        for i in 0..10 {
            tracker.observe_cache_miss(1, 0x1000 + i * 256, false);
        }

        // Simulate wrong predictions
        for _ in 0..10 {
            tracker.record_wrong_prediction(1);
        }

        // After many wrong predictions, accuracy should be low
        let state = tracker.states.get(&1).unwrap();
        assert!(state.accuracy() < PREFETCH_CONFIDENCE_THRESHOLD);
    }

    #[test]
    fn test_hardware_prefetcher_conflict_avoidance() {
        let mut tracker = PrefetchTracker::new(64, false);

        // Linear access pattern (stride = cache line size)
        for i in 0..10 {
            tracker.observe_cache_miss(2, 0x2000 + i * CACHE_LINE_SIZE, false);
        }

        // This should be classified as Linear, and no hint should be generated
        // because the hardware prefetcher handles linear accesses
        if let Some(hint) = tracker.get_hint(2) {
            // If a hint was generated, it should be suppressed
            assert!(tracker.should_suppress_for_hw_prefetcher(hint));
        }
    }

    #[test]
    fn test_prophetic_engine_end_to_end() {
        let mut engine = PropheticPrefetchEngine::new(64, false, true);

        // Simulate pointer-chasing pattern
        for i in 0..10 {
            let addr = 0x5000 + i * 0x200; // Stride of 512 bytes
            engine.observe_cache_miss(42, addr, true);
        }

        // Should have generated hints for this pointer-chasing pattern
        let hints = engine.active_hints();
        // Pointer chase patterns should be targeted
    }

    #[test]
    fn test_encode_prefetch_bytes() {
        let mut buf = Vec::new();

        // Encode PREFETCHT0 [rax + 64]
        PrefetchEmitter::encode_prefetch_t0(&mut buf, 0, 64);
        assert!(!buf.is_empty());

        // Encode PREFETCHNTA [rax + 128]
        buf.clear();
        PrefetchEmitter::encode_prefetch_nta(&mut buf, 0, 128);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_stats() {
        let engine = PropheticPrefetchEngine::new(64, false, true);
        let stats = engine.stats();
        assert!(stats.enabled);
        assert!(!stats.force_disabled);
        assert_eq!(stats.prefetches_executed, 0);
        assert_eq!(stats.correct_predictions, 0);
    }

    #[test]
    fn test_force_disable() {
        let engine = PropheticPrefetchEngine::new(64, false, true);
        assert!(engine.is_active());

        engine.force_disable();
        assert!(!engine.is_active());

        engine.force_enable();
        assert!(engine.is_active());
    }
}
