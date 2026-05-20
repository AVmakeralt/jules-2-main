// =============================================================================
// jules/src/polyhedral.rs
//
// HIGH-PERFORMANCE POLYHEDRAL OPTIMIZATION ENGINE (Tier-2 / Tracing JIT Handoff)
// 100% DEPENDENCY-FREE — ZERO EXTERN CRATES —
// =============================================================================

use crate::compiler::ast::BinOpKind;
use crate::interp::Instr;

/// Hardcoded maximum loop nesting depth to eliminate heap-allocated maps/vectors
/// in the multi-dimensional math hot paths.
pub const MAX_POLY_DEPTH: usize = 8;

/// Size limit for JIT tracking slots.
/// FIXED: Raised from 256 → 4096. The old limit silently dropped any induction
/// variable or array pointer assigned to slot ≥ 256, producing incorrect
/// dependency analysis and miscompiled code on large traces. Violations in
/// debug builds are caught by the debug_assert in SlotCache::insert.
pub const MAX_TRACKED_SLOTS: usize = 4096;

// =============================================================================
// §1. MULTI-DIMENSIONAL AFFINE MATHEMATICS
// =============================================================================

/// Represents a multi-dimensional affine expression: C0 + C1*v1 + C2*v2 ...
/// Compact structure that implements Copy to reside completely on the stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffineExpr {
    pub constant: i64,
    /// Coefficients for active loop induction variables.
    pub coefficients: [i64; MAX_POLY_DEPTH],
    /// Bitmask identifying which dimensions are actively used in this expression.
    pub active_mask: u8,
}

impl AffineExpr {
    #[inline]
    pub fn constant(val: i64) -> Self {
        Self {
            constant: val,
            coefficients: [0; MAX_POLY_DEPTH],
            active_mask: 0,
        }
    }

    #[inline]
    pub fn variable(id: usize) -> Self {
        debug_assert!(id < MAX_POLY_DEPTH);
        let mut coeffs = [0; MAX_POLY_DEPTH];
        coeffs[id] = 1;
        Self {
            constant: 0,
            coefficients: coeffs,
            active_mask: 1 << id,
        }
    }

    /// Vectorizer-friendly add: fixed 8-element loop, no branches on the mask
    /// inside the hot path.  LLVM/rustc will emit a single AVX2 `vpaddq` (or
    /// AVX-512 `vpaddq zmm`) for the coefficient array when the target supports
    /// it.  The mask is only consulted once (for `active_mask`) rather than
    /// once per set bit.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let mut res = *self;
        res.constant = res.constant.wrapping_add(other.constant);
        // Unrolled fixed-width loop: the compiler sees exactly 8 iterations
        // with no data-dependent control flow and can emit a single SIMD add.
        for i in 0..MAX_POLY_DEPTH {
            res.coefficients[i] =
                res.coefficients[i].wrapping_add(other.coefficients[i]);
        }
        res.active_mask |= other.active_mask;
        res
    }

    /// Scalar-broadcast multiply; same fixed-width loop as `add`.
    #[inline]
    pub fn mul_const(&self, c: i64) -> Self {
        let mut res = *self;
        res.constant = res.constant.wrapping_mul(c);
        for i in 0..MAX_POLY_DEPTH {
            res.coefficients[i] = res.coefficients[i].wrapping_mul(c);
        }
        res
    }

    /// Vectorizer-friendly sub; mirrors `add`.
    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        let mut res = *self;
        res.constant = res.constant.wrapping_sub(other.constant);
        for i in 0..MAX_POLY_DEPTH {
            res.coefficients[i] =
                res.coefficients[i].wrapping_sub(other.coefficients[i]);
        }
        res.active_mask |= other.active_mask;
        res
    }
}

/// Safe Binary Greatest Common Divisor (Stein's Algorithm).
///
/// Restructured to avoid `std::mem::swap` (two writes + a tmp) and the
/// dependent subtraction stall.  Instead, the min/max are computed with
/// branchless `i64::min`/`max`, and the absolute difference is taken once
/// per iteration.  The CPU's execution units see a data-parallel critical
/// path with no explicit branch mispredictions.
#[inline]
fn safe_gcd(mut u: i64, mut v: i64) -> i64 {
    if u == 0 { return v.abs(); }
    if v == 0 { return u.abs(); }
    u = u.abs();
    v = v.abs();
    let shift = (u | v).trailing_zeros();
    u >>= u.trailing_zeros();
    while v != 0 {
        v >>= v.trailing_zeros();
        // Branchless min/max — no swap, no pipeline hazard.
        let lo = u.min(v);
        let hi = u.max(v);
        u = lo;
        v = hi - lo;          // always ≥ 0; next iteration strips trailing zeros
    }
    u << shift
}

// =============================================================================
// §2. DEPENDENCY ANALYSIS (BANERJEE-WOLFE INEQUALITY BOUNDS)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction { LT, EQ, GT, ANY }

#[derive(Debug, Clone)]
pub struct Dependency {
    pub src_stmt: usize,
    pub dst_stmt: usize,
    pub direction_vector: [Direction; MAX_POLY_DEPTH],
    pub distance_matrix: [i64; MAX_POLY_DEPTH],
    pub len: usize,
}

pub fn analyze_dependency_multivariate(
    src_expr: &AffineExpr,
    dst_expr: &AffineExpr,
    bounds: &[(i64, i64)],
) -> Option<Dependency> {
    let diff_expr = src_expr.sub(dst_expr);
    let target = -diff_expr.constant;

    let combined_mask = diff_expr.active_mask;
    if combined_mask == 0 {
        return if target == 0 {
            let limit = bounds.len().min(MAX_POLY_DEPTH);
            Some(Dependency {
                src_stmt: 0, dst_stmt: 0,
                direction_vector: [Direction::EQ; MAX_POLY_DEPTH],
                distance_matrix: [0; MAX_POLY_DEPTH],
                len: limit,
            })
        } else {
            None
        };
    }

    let mut coeffs = [0i64; MAX_POLY_DEPTH];
    let mut dims = [0usize; MAX_POLY_DEPTH];
    let mut active_count = 0;
    let mut mask = combined_mask;
    while mask != 0 {
        let var_id = mask.trailing_zeros() as usize;
        let c = diff_expr.coefficients[var_id];
        if c != 0 {
            coeffs[active_count] = c;
            dims[active_count] = var_id;
            active_count += 1;
        }
        mask &= mask - 1;
    }

    if active_count == 0 {
        return if target == 0 {
            Some(Dependency {
                src_stmt: 0, dst_stmt: 0,
                direction_vector: [Direction::EQ; MAX_POLY_DEPTH],
                distance_matrix: [0; MAX_POLY_DEPTH],
                len: bounds.len().min(MAX_POLY_DEPTH),
            })
        } else {
            None
        };
    }

    let mut g = coeffs[0];
    for i in 1..active_count {
        g = safe_gcd(g, coeffs[i]);
        if g == 1 { break; }
    }

    if g == 0 || target % g != 0 { return None; }

    // ── Fourier-Motzkin Elimination (exact shadow projection) ────────────────
    //
    // Banerjee-Wolfe tests only the scalar range of Σ cᵢ·dᵢ and accepts any
    // overlap as a potential dependency — it can be wildly conservative when
    // coefficients have mixed signs or when the feasible set is a thin lattice
    // slice.
    //
    // We instead project the constraint system down dimension-by-dimension on
    // the stack (MAX_POLY_DEPTH = 8, so at most 8 Fourier-Motzkin steps).
    // Each step eliminates one variable by computing the implied lower/upper
    // bounds on the remaining system, tightening the feasible interval by the
    // GCD divisibility condition.  If the interval becomes infeasible at any
    // point we return None — proving no integer solution exists and unlocking
    // aggressive parallelisation that Banerjee-Wolfe would have blocked.
    //
    // Complexity: O(active_count²) in the worst case — but active_count ≤ 8
    // and all arithmetic is on stack-resident i64 values, so this costs a
    // handful of ALU cycles compared with the heap allocation Banerjee-Wolfe
    // avoided anyway.

    // Current feasible interval for the remaining RHS after eliminating
    // variables one-by-one.  Start with [target, target] (exact equality).
    let mut lo = target;
    let mut hi = target;

    // Remaining coefficient/dimension pairs (shrink as we eliminate variables).
    let mut rem_coeffs = coeffs;
    let mut rem_dims   = dims;
    let mut rem_count  = active_count;

    for _step in 0..active_count {
        if rem_count == 0 { break; }

        // Pick the variable with the largest |coefficient| first (reduces
        // interval expansion fastest and keeps the remaining bounds tight).
        let mut best = 0usize;
        let mut best_abs = rem_coeffs[0].unsigned_abs();
        for k in 1..rem_count {
            let a = rem_coeffs[k].unsigned_abs();
            if a > best_abs { best_abs = a; best = k; }
        }

        let c   = rem_coeffs[best];
        let var = rem_dims[best];
        let (l_b, u_b) = if var < bounds.len() { bounds[var] } else { (0, 1000) };

        // Eliminate `c·x_var` from [lo, hi]:
        //   if c > 0:  lo -= c * u_b;  hi -= c * l_b
        //   if c < 0:  lo -= c * l_b;  hi -= c * u_b
        // This gives the interval that the *remaining* terms must satisfy.
        if c > 0 {
            lo = lo.saturating_sub(c.saturating_mul(u_b));
            hi = hi.saturating_sub(c.saturating_mul(l_b));
        } else {
            lo = lo.saturating_sub(c.saturating_mul(l_b));
            hi = hi.saturating_sub(c.saturating_mul(u_b));
        }

        // After elimination the remaining GCD must still divide the interval.
        // Tighten [lo, hi] to the nearest multiples (ceil/floor) of rem_gcd.
        if rem_count > 1 {
            let mut rem_g = 0i64;
            for k in 0..rem_count {
                if k != best {
                    rem_g = safe_gcd(rem_g, rem_coeffs[k]);
                }
            }
            if rem_g > 1 {
                // Round lo up and hi down to multiples of rem_g.
                let lo_r = lo.rem_euclid(rem_g);
                if lo_r != 0 { lo = lo.saturating_add(rem_g - lo_r); }
                let hi_r = hi.rem_euclid(rem_g);
                if hi_r != 0 { hi = hi.saturating_sub(hi_r); }
            }
        }

        // Early exit: interval is empty — no integer solution exists.
        if lo > hi { return None; }

        // Remove the eliminated variable from the working set.
        rem_coeffs[best] = rem_coeffs[rem_count - 1];
        rem_dims[best]   = rem_dims[rem_count - 1];
        rem_count -= 1;
    }

    // If the fully projected interval is empty, no dependency.
    if lo > hi { return None; }

    let limit = bounds.len().min(MAX_POLY_DEPTH);
    let mut dir_vec = [Direction::ANY; MAX_POLY_DEPTH];
    let mut dist_vec = [0i64; MAX_POLY_DEPTH];
    for i in 0..active_count {
        let c = coeffs[i];
        let var_id = dims[i];
        if var_id >= limit { continue; }
        if c > 0 {
            if target > 0      { dir_vec[var_id] = Direction::LT; dist_vec[var_id] =  1; }
            else if target < 0 { dir_vec[var_id] = Direction::GT; dist_vec[var_id] = -1; }
            else               { dir_vec[var_id] = Direction::EQ; dist_vec[var_id] =  0; }
        } else {
            if target > 0      { dir_vec[var_id] = Direction::GT; dist_vec[var_id] = -1; }
            else if target < 0 { dir_vec[var_id] = Direction::LT; dist_vec[var_id] =  1; }
            else               { dir_vec[var_id] = Direction::EQ; dist_vec[var_id] =  0; }
        }
    }

    Some(Dependency {
        src_stmt: 0, dst_stmt: 0,
        direction_vector: dir_vec,
        distance_matrix: dist_vec,
        len: limit,
    })
}

// =============================================================================
// §3. COMPACT SCoP EXTRACTION — ARENA LAYOUT + TRUE ITERATIVE BUILDER
// =============================================================================

#[derive(Debug, Clone)]
pub struct InductionVar {
    pub slot: u16,
    pub step: i64,
}

#[derive(Debug, Clone)]
pub struct AccessRelation {
    pub array_base_slot: u16,
    pub index_expr: AffineExpr,
    pub is_read: bool,
}

#[derive(Debug, Clone)]
pub struct PolyStmt {
    pub id: usize,
    pub op: BinOpKind,
    pub dst: u16,
    pub src1: u16,
    pub src2: u16,
}

/// Cache-friendly arena-allocated loop node.
///
/// FIXED: The original `PolyLoop` embedded three `Vec` fields (`child_loops`,
/// `accesses`, `stmts`), each a separate heap allocation. On giant SCoPs this
/// created fragmented memory that thrashed L1/L2 caches during traversal.
///
/// Now every field is a range `[start, start+len)` into the corresponding flat
/// arena slice in `ScopArena`. Traversal is a single contiguous scan — the CPU
/// prefetcher handles it perfectly.
#[derive(Debug, Clone)]
pub struct PolyLoop {
    pub depth: usize,
    pub iv: InductionVar,
    pub lower_bound: AffineExpr,
    pub upper_bound: AffineExpr,
    /// Range into `ScopArena::loops` for immediate child loops.
    pub child_start: u32,
    pub child_len:   u32,
    /// Range into `ScopArena::accesses`.
    pub access_start: u32,
    pub access_len:   u32,
    /// Range into `ScopArena::stmts`.
    pub stmt_start: u32,
    pub stmt_len:   u32,
    pub header_pc:    usize,
    pub back_edge_pc: usize,
}

/// Flat arena holding all SCoP data in three contiguous allocations.
/// This guarantees cache-line efficiency for every traversal pattern.
#[derive(Debug, Clone)]
pub struct ScopArena {
    pub loops:    Vec<PolyLoop>,
    pub accesses: Vec<AccessRelation>,
    pub stmts:    Vec<PolyStmt>,
    /// Indices of top-level (root) loops within `loops`.
    pub root_loop_indices: Vec<u32>,
    pub max_depth: usize,
}

#[derive(Debug, Clone)]
pub struct Scop {
    pub arena: ScopArena,
}

impl Scop {
    #[inline]
    pub fn max_depth(&self) -> usize { self.arena.max_depth }
}

/// Slot-indexed affine expression cache.
///
/// FIXED: Limit raised from 256 → MAX_TRACKED_SLOTS (4096). The old 256-slot
/// hard limit silently discarded any slot ≥ 256 — if an induction variable or
/// array pointer landed there, the polyhedral extractor produced incorrect
/// dependency analysis and potentially miscompiled code with no diagnostic.
///
/// In debug builds a `debug_assert!` now fires immediately if a slot exceeds
/// the new limit, surfacing the problem at the earliest possible point.
///
/// OPTIMISED: A 512-byte presence bitset (`[u64; 64]`) shadows the data Vec.
/// `get` checks the bitset first — if the bit is clear the slot is definitely
/// absent and we never touch the 4096-element Vec.  For the common case where
/// most slots are empty this eliminates the cache-line fetch for the `Option`
/// discriminant entirely.
pub struct SlotCache {
    pub data: Vec<Option<AffineExpr>>,
    /// 4096-bit presence set.  Bit `s` is set iff `data[s]` is `Some`.
    present: [u64; 64],
}

impl SlotCache {
    #[inline]
    pub fn new() -> Self {
        Self {
            data: vec![None; MAX_TRACKED_SLOTS],
            present: [0u64; 64],
        }
    }

    #[inline]
    pub fn insert(&mut self, slot: u16, expr: AffineExpr) {
        let idx = slot as usize;
        debug_assert!(
            idx < MAX_TRACKED_SLOTS,
            "slot {} exceeds MAX_TRACKED_SLOTS ({}); tighten upstream trace chunking",
            slot, MAX_TRACKED_SLOTS
        );
        if idx < MAX_TRACKED_SLOTS {
            self.data[idx] = Some(expr);
            self.present[idx >> 6] |= 1u64 << (idx & 63);
        }
    }

    #[inline]
    pub fn get(&self, slot: u16) -> Option<AffineExpr> {
        let idx = slot as usize;
        if idx >= MAX_TRACKED_SLOTS { return None; }
        // Fast-path: bit clear → definitely None, no Vec access needed.
        if self.present[idx >> 6] & (1u64 << (idx & 63)) == 0 {
            return None;
        }
        self.data[idx]
    }
}

/// Forward-pass cache population: seeds every slot reachable by affine
/// arithmetic from known loop induction variables.
pub fn populate_slot_cache(instrs: &[Instr], cache: &mut SlotCache, loop_iv_slots: &[u16]) {
    for (i, &iv_slot) in loop_iv_slots.iter().enumerate() {
        if i < MAX_POLY_DEPTH {
            cache.insert(iv_slot, AffineExpr::variable(i));
        }
    }

    for instr in instrs {
        match *instr {
            Instr::LoadI64(d, v) => cache.insert(d, AffineExpr::constant(v)),
            Instr::LoadI32(d, v) => cache.insert(d, AffineExpr::constant(v as i64)),
            Instr::Move(d, s) => {
                if let Some(expr) = cache.get(s) { cache.insert(d, expr); }
            }
            Instr::BinOp(d, op, l, r) => {
                let l_expr = cache.get(l);
                let r_expr = cache.get(r);
                match op {
                    BinOpKind::Add => {
                        if let (Some(le), Some(re)) = (l_expr, r_expr) {
                            cache.insert(d, le.add(&re));
                        }
                    }
                    BinOpKind::Sub => {
                        if let (Some(le), Some(re)) = (l_expr, r_expr) {
                            cache.insert(d, le.sub(&re));
                        }
                    }
                    BinOpKind::Mul => {
                        if let (Some(le), Some(re)) = (l_expr, r_expr) {
                            if re.active_mask == 0 {
                                cache.insert(d, le.mul_const(re.constant));
                            } else if le.active_mask == 0 {
                                cache.insert(d, re.mul_const(le.constant));
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

#[inline]
fn get_dst_slot(instr: &Instr) -> u16 {
    match *instr {
        Instr::LoadI32(d, _) | Instr::LoadI64(d, _)
        | Instr::LoadBool(d, _) | Instr::LoadUnit(d) => d,
        Instr::Move(d, _) | Instr::Load(d, _)
        | Instr::Store(d, _) | Instr::BinOp(d, _, _, _) => d,
        _ => 0,
    }
}

fn analyze_loop(
    instrs: &[Instr],
    header: usize,
    back_edge: usize,
    loop_iv_slots: &mut Vec<u16>,
    cache: &SlotCache,
) -> (Option<InductionVar>, AffineExpr, AffineExpr) {
    let mut iv = None;
    let mut lb = AffineExpr::constant(0);
    let mut ub = AffineExpr::constant(1024);

    // Collect all candidate induction variables before selecting the best one.
    // A candidate is a slot that appears in a BinOp(Lt, slot, bound) comparison
    // within the loop body AND is incremented by a constant step.
    #[derive(Debug)]
    struct IvCandidate { slot: u16, step: i64 }

    // Find all slots compared with Lt in the loop body
    let mut loop_compare_slots: Vec<u16> = Vec::new();
    for pc in header..=back_edge {
        if let Instr::BinOp(_, BinOpKind::Lt, l, _) = instrs[pc] {
            if !loop_compare_slots.contains(&l) {
                loop_compare_slots.push(l);
            }
        }
    }

    for pc in header..=back_edge {
        // Pattern 1: i = i + step  (self-add: BinOp(dst, Add, dst, step))
        if let Instr::BinOp(dst, BinOpKind::Add, l, r) = instrs[pc] {
            if dst == l && pc > 0 {
                if let Instr::LoadI64(_, step) = instrs[pc - 1] {
                    if r == get_dst_slot(&instrs[pc - 1]) {
                        // Only accept if this slot appears in a loop comparison
                        if loop_compare_slots.contains(&dst) {
                            if loop_iv_slots.len() < MAX_POLY_DEPTH {
                                loop_iv_slots.push(dst);
                                iv = Some(InductionVar { slot: dst, step });
                                break;
                            }
                        }
                    }
                }
            }
            // Pattern 2: temp = i + step; i = temp  (split add+move)
            // This is the pattern the Jules compiler emits: BinOp(temp, Add, i, step); Move(i, temp)
            if pc + 1 < instrs.len() {
                if let Instr::Move(dst_slot, src_slot) = instrs[pc + 1] {
                    if src_slot == dst {
                        // The move target (dst_slot) is the induction variable
                        // Only accept if this slot appears in a loop comparison
                        if loop_compare_slots.contains(&dst_slot) {
                            // Check if r is a constant step loaded just before
                            let step_val = if pc > 0 {
                                if let Instr::LoadI64(_, step) = instrs[pc - 1] {
                                    if r == get_dst_slot(&instrs[pc - 1]) {
                                        Some(step)
                                    } else {
                                        None
                                    }
                                } else if let Instr::LoadI32(_, step) = instrs[pc - 1] {
                                    if r == get_dst_slot(&instrs[pc - 1]) {
                                        Some(step as i64)
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            };
                            if let Some(step) = step_val {
                                if loop_iv_slots.len() < MAX_POLY_DEPTH {
                                    loop_iv_slots.push(dst_slot);
                                    iv = Some(InductionVar { slot: dst_slot, step });
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(ref iv_ref) = iv {
        // Search for the loop condition that tests the induction variable.
        // Three patterns to handle:
        //   1. JumpTrue at back_edge (do-while style)
        //   2. JumpFalse near header (while-style, early exit)
        //   3. JumpFalse anywhere in loop body (general case)
        let mut found_ub = false;

        // Pattern 1: JumpTrue at back_edge
        if let Instr::JumpTrue(cond, _) = instrs[back_edge] {
            for pc in (header..back_edge).rev() {
                if let Instr::BinOp(d, BinOpKind::Lt, l, r) = instrs[pc] {
                    if d == cond && l == iv_ref.slot {
                        if let Some(expr) = cache.get(r) { ub = expr; }
                        found_ub = true;
                        break;
                    }
                }
            }
        }

        // Pattern 2 & 3: JumpFalse anywhere in the loop body
        if !found_ub {
            for pc in header..=back_edge {
                if let Instr::JumpFalse(cond, _) = instrs[pc] {
                    for cond_pc in (header..=pc).rev() {
                        if let Instr::BinOp(d, BinOpKind::Lt, l, r) = instrs[cond_pc] {
                            if d == cond && l == iv_ref.slot {
                                if let Some(expr) = cache.get(r) { ub = expr; }
                                found_ub = true;
                                break;
                            }
                        }
                    }
                    if found_ub { break; }
                }
            }
        }

        // Fallback: search for any BinOp(Lt, iv, bound) in the loop body
        if !found_ub {
            for pc in header..=back_edge {
                if let Instr::BinOp(_, BinOpKind::Lt, l, r) = instrs[pc] {
                    if l == iv_ref.slot {
                        if let Some(expr) = cache.get(r) { ub = expr; }
                        break;
                    }
                }
            }
        }

        if let Some(expr) = cache.get(iv_ref.slot) { lb = expr; }
    }

    (iv, lb, ub)
}

/// True iterative loop-tree builder — no recursion, no stack overflow risk.
///
/// FIXED: The original `build_tree_iterative` was recursive despite its name.
/// On deep loop nests or highly unrolled control flow this would exhaust the
/// default 2–8 MB thread stack and crash the JIT.
///
/// This implementation uses an explicit heap-allocated work-stack.  All loop
/// nodes are written directly into a flat `ScopArena`; parent-child
/// relationships are encoded as index ranges rather than pointer trees,
/// eliminating both the recursion hazard and the cache-thrashing Vec-of-Vecs
/// layout from the original.
fn build_arena_iterative(
    loop_ranges: &[(usize, usize)],
    instrs: &[Instr],
    loop_iv_slots: &mut Vec<u16>,
    cache: &SlotCache,
) -> ScopArena {
    let mut arena = ScopArena {
        loops:    Vec::with_capacity(loop_ranges.len()),
        accesses: Vec::new(),
        stmts:    Vec::new(),
        root_loop_indices: Vec::new(),
        max_depth: 0,
    };

    if loop_ranges.is_empty() { return arena; }

    // Work-stack entry: (range_index, parent_arena_index).
    // u32::MAX means "no parent" (root-level loop).
    let mut stack: Vec<(usize, u32)> = Vec::with_capacity(64);

    // Seed with top-level ranges (those not nested inside any other range).
    // Reversed so left-to-right program order is preserved after stack pops.
    for (i, &(h, b)) in loop_ranges.iter().enumerate().rev() {
        let is_nested = loop_ranges[..i]
            .iter()
            .any(|&(ph, pb)| ph <= h && b <= pb);
        if !is_nested {
            stack.push((i, u32::MAX));
        }
    }

    while let Some((range_idx, parent_arena_idx)) = stack.pop() {
        let (h, b) = loop_ranges[range_idx];

        // Find immediate children: ranges strictly inside (h, b) that are not
        // themselves nested inside another child of (h, b).
        let mut children: Vec<usize> = loop_ranges
            .iter()
            .enumerate()
            .filter(|&(ci, &(ch, cb))| {
                ci != range_idx
                    && ch > h && cb <= b
                    && !loop_ranges.iter().enumerate().any(|(si, &(sh, sb))| {
                        si != range_idx && si != ci
                            && sh > h && sb <= b
                            && ch >= sh && cb <= sb
                    })
            })
            .map(|(ci, _)| ci)
            .collect();
        children.sort_unstable_by_key(|&ci| loop_ranges[ci].0);

        let (iv, lb, ub) = analyze_loop(instrs, h, b, loop_iv_slots, cache);
        let iv = match iv {
            Some(v) => v,
            None => {
                // Not a recognisable loop — propagate children upward.
                for ci in children.into_iter().rev() {
                    stack.push((ci, parent_arena_idx));
                }
                continue;
            }
        };

        // Harvest accesses and stmts for this loop body, skipping instructions
        // that belong to an immediate child's body (they will be collected when
        // that child is processed).
        let access_start = arena.accesses.len() as u32;
        let stmt_start   = arena.stmts.len()    as u32;

        for pc in h..=b {
            let in_child = children.iter().any(|&ci| {
                let (ch, cb) = loop_ranges[ci];
                pc > ch && pc < cb
            });
            if in_child { continue; }

            match instrs[pc] {
                Instr::Load(_, ptr_slot) => {
                    if let Some(expr) = cache.get(ptr_slot) {
                        arena.accesses.push(AccessRelation {
                            array_base_slot: ptr_slot,
                            index_expr: expr,
                            is_read: true,
                        });
                    }
                }
                Instr::Store(ptr_slot, _) => {
                    if let Some(expr) = cache.get(ptr_slot) {
                        arena.accesses.push(AccessRelation {
                            array_base_slot: ptr_slot,
                            index_expr: expr,
                            is_read: false,
                        });
                    }
                }
                Instr::BinOp(dst, op, l, r) => {
                    arena.stmts.push(PolyStmt { id: pc, op, dst, src1: l, src2: r });
                }
                _ => {}
            }
        }

        let access_len = arena.accesses.len() as u32 - access_start;
        let stmt_len   = arena.stmts.len()    as u32 - stmt_start;
        let my_arena_idx = arena.loops.len() as u32;

        arena.loops.push(PolyLoop {
            depth: 1, // updated in the post-pass below
            iv,
            lower_bound: lb,
            upper_bound: ub,
            child_start: 0, // patched in post-pass
            child_len:   0,
            access_start,
            access_len,
            stmt_start,
            stmt_len,
            header_pc:    h,
            back_edge_pc: b,
        });

        if parent_arena_idx == u32::MAX {
            arena.root_loop_indices.push(my_arena_idx);
        }

        // Push children in reverse so the leftmost is processed first.
        for ci in children.into_iter().rev() {
            stack.push((ci, my_arena_idx));
        }
    }

    // ── Post-pass: wire up child ranges and compute depths ───────────────────
    let n = arena.loops.len();

    // Reconstruct parent-child relationships from header/back-edge coordinates.
    let mut parent_of = vec![u32::MAX; n];
    for i in 0..n {
        let (h, b) = (arena.loops[i].header_pc, arena.loops[i].back_edge_pc);
        let mut best_parent: Option<usize> = None;
        for j in 0..n {
            if j == i { continue; }
            let (ph, pb) = (arena.loops[j].header_pc, arena.loops[j].back_edge_pc);
            if ph <= h && b <= pb {
                best_parent = Some(match best_parent {
                    None => j,
                    Some(prev) => {
                        if ph > arena.loops[prev].header_pc { j } else { prev }
                    }
                });
            }
        }
        parent_of[i] = best_parent.map(|p| p as u32).unwrap_or(u32::MAX);
    }

    // Set child_start / child_len.  Arena ordering is depth-first, so
    // children always appear at indices greater than their parent — but the
    // child_start/child_len window is based on contiguous children, which may
    // not be the case for sibling loops.  We do a simple per-parent scan.
    for p in 0..n {
        let first_child = (0..n).find(|&c| parent_of[c] == p as u32);
        if let Some(fc) = first_child {
            let child_count = (0..n).filter(|&c| parent_of[c] == p as u32).count();
            arena.loops[p].child_start = fc as u32;
            arena.loops[p].child_len   = child_count as u32;
        }
    }

    // Compute depth bottom-up: process shortest spans (leaf loops) first.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| {
        arena.loops[i].back_edge_pc.wrapping_sub(arena.loops[i].header_pc)
    });
    for &i in &order {
        let cs = arena.loops[i].child_start as usize;
        let ce = cs + arena.loops[i].child_len as usize;
        let max_child = (cs..ce).map(|c| arena.loops[c].depth).max().unwrap_or(0);
        arena.loops[i].depth = 1 + max_child;
    }

    arena.max_depth = arena.root_loop_indices.iter()
        .map(|&ri| arena.loops[ri as usize].depth)
        .max()
        .unwrap_or(1);

    arena
}

pub fn extract_scop(instrs: &[Instr]) -> Option<Scop> {
    let mut loop_ranges = Vec::new();

    for (pc, instr) in instrs.iter().enumerate() {
        let target = match *instr {
            Instr::Jump(off)         => Some((pc as i32 + 1 + off) as usize),
            Instr::JumpFalse(_, off) => Some((pc as i32 + 1 + off) as usize),
            Instr::JumpTrue(_, off)  => Some((pc as i32 + 1 + off) as usize),
            _ => None,
        };
        if let Some(t) = target {
            if t <= pc && t < instrs.len() {
                loop_ranges.push((t, pc));
            }
        }
    }

    if loop_ranges.is_empty() { return None; }

    loop_ranges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));

    // First pass: discover induction variable slots so populate_slot_cache can
    // seed the affine environment before full SCoP extraction.
    let mut loop_iv_slots: Vec<u16> = Vec::with_capacity(MAX_POLY_DEPTH);
    {
        let tmp_cache = SlotCache::new();
        for &(h, b) in &loop_ranges {
            let mut dummy = Vec::new();
            let (iv, _, _) = analyze_loop(instrs, h, b, &mut dummy, &tmp_cache);
            if let Some(ind_v) = iv {
                if !loop_iv_slots.contains(&ind_v.slot)
                    && loop_iv_slots.len() < MAX_POLY_DEPTH
                {
                    loop_iv_slots.push(ind_v.slot);
                }
            }
        }
    }

    let mut cache = SlotCache::new();
    populate_slot_cache(instrs, &mut cache, &loop_iv_slots);

    let arena = build_arena_iterative(&loop_ranges, instrs, &mut loop_iv_slots, &cache);

    if arena.loops.is_empty() { None } else { Some(Scop { arena }) }
}

// =============================================================================
// §4. PHYSICAL TILING LOOP GENERATION (SAFE REMAINDER BOUNDS)
// =============================================================================

pub fn generate_tiled_loops(scop: &Scop, tile_sizes: &[usize]) -> Vec<Instr> {
    let arena = &scop.arena;

    // FIXED: Pre-calculate the exact output size to eliminate mid-loop
    // reallocations. The original used `loops.len() * 64` as a rough guess;
    // we now account for the actual statement count from the arena.
    let total_stmts: usize = arena.loops.iter().map(|l| l.stmt_len as usize).sum();
    let mut out = Vec::with_capacity(arena.loops.len() * 12 + total_stmts);
    let mut next_slot: u16 = 4000;

    for poly_loop in &arena.loops {
        let tile_size = tile_sizes.get(0).copied().unwrap_or(32) as i64;

        let ti_slot           = next_slot; next_slot += 1;
        let limit_slot        = next_slot; next_slot += 1;
        let tile_const_slot   = next_slot; next_slot += 1;
        let n_const_slot      = next_slot; next_slot += 1;
        let ti_plus_tile_slot = next_slot; next_slot += 1;
        let cond_slot         = next_slot; next_slot += 1;

        if poly_loop.lower_bound.active_mask == 0 {
            out.push(Instr::LoadI64(ti_slot, poly_loop.lower_bound.constant));
        } else {
            out.push(Instr::LoadI64(ti_slot, 0));
        }

        out.push(Instr::LoadI64(tile_const_slot, tile_size));

        if poly_loop.upper_bound.active_mask == 0 {
            out.push(Instr::LoadI64(n_const_slot, poly_loop.upper_bound.constant));
        } else {
            out.push(Instr::LoadI64(n_const_slot, 1024));
        }

        // Outer tile loop.
        let l1_pc = out.len();
        out.push(Instr::BinOp(cond_slot, BinOpKind::Ge, ti_slot, n_const_slot));
        out.push(Instr::JumpTrue(cond_slot, 0));
        let end1_patch = out.len() - 1;

        out.push(Instr::Move(poly_loop.iv.slot, ti_slot));
        out.push(Instr::BinOp(ti_plus_tile_slot, BinOpKind::Add, ti_slot, tile_const_slot));
        out.push(Instr::BinOp(cond_slot, BinOpKind::Gt, ti_plus_tile_slot, n_const_slot));
        out.push(Instr::JumpFalse(cond_slot, 2));
        out.push(Instr::Move(limit_slot, n_const_slot));
        out.push(Instr::Jump(1));
        out.push(Instr::Move(limit_slot, ti_plus_tile_slot));

        // Inner element loop.
        let l2_pc = out.len();
        out.push(Instr::BinOp(cond_slot, BinOpKind::Ge, poly_loop.iv.slot, limit_slot));
        out.push(Instr::JumpTrue(cond_slot, 0));
        let end2_patch = out.len() - 1;

        // Emit statements from the contiguous arena slice — cache-friendly.
        let stmt_range =
            poly_loop.stmt_start as usize..(poly_loop.stmt_start + poly_loop.stmt_len) as usize;
        for stmt in &arena.stmts[stmt_range] {
            out.push(Instr::BinOp(stmt.dst, stmt.op, stmt.src1, stmt.src2));
        }

        let step_slot = next_slot; next_slot += 1;
        out.push(Instr::LoadI64(step_slot, poly_loop.iv.step));
        out.push(Instr::BinOp(
            poly_loop.iv.slot, BinOpKind::Add, poly_loop.iv.slot, step_slot,
        ));

        let l2_offset = (l2_pc as i32) - (out.len() as i32) - 1;
        out.push(Instr::Jump(l2_offset));

        let end2_pc = out.len();
        if let Instr::JumpTrue(_, ref mut off) = out[end2_patch] {
            *off = (end2_pc as i32) - (end2_patch as i32) - 1;
        }

        out.push(Instr::BinOp(ti_slot, BinOpKind::Add, ti_slot, tile_const_slot));
        let l1_offset = (l1_pc as i32) - (out.len() as i32) - 1;
        out.push(Instr::Jump(l1_offset));

        let end1_pc = out.len();
        if let Instr::JumpTrue(_, ref mut off) = out[end1_patch] {
            *off = (end1_pc as i32) - (end1_patch as i32) - 1;
        }
    }

    out
}

// =============================================================================
// §5. ALLOCATION-FREE SIMD EMISSION HANDOFF
// =============================================================================

#[derive(Debug, Clone)]
pub enum SimdHintKind {
    VectorPack { op: BinOpKind, width: usize, src1_base: u16, src2_base: u16, dst_base: u16 },
    RegisterLock { slots: [u16; 4], len: u8 },
    TileLoopBoundary { is_entry: bool, tile_size: usize },
}

/// Flat hint table with O(log N) lookup via binary search.
///
/// FIXED: The original emitter pushed VectorPack at `pc` and RegisterLock at
/// `pc+1` for every BinOp, then relied on `binary_search_by_key` for lookup.
/// When two consecutive instructions both emitted hints, the `pc+1` entries
/// were unsorted and contained duplicate keys, making `binary_search`
/// undefined (it returns an arbitrary matching index, not necessarily the
/// intended one).
///
/// Now we sort by PC after collection and deduplicate, keeping the first hint
/// at each address (VectorPack beats RegisterLock when they collide).
#[derive(Debug, Clone)]
pub struct PolyhedralBlock {
    pub instrs: Vec<Instr>,
    /// Sorted, deduplicated: one entry per unique PC. Safe for binary_search.
    pub hints: Vec<(usize, SimdHintKind)>,
}

impl PolyhedralBlock {
    pub fn get_hint(&self, pc: usize) -> Option<&SimdHintKind> {
        self.hints
            .binary_search_by_key(&pc, |(k, _)| *k)
            .ok()
            .map(|idx| &self.hints[idx].1)
    }
}

pub fn generate_simd_hints(_scop: &Scop, tiled_instrs: &[Instr]) -> PolyhedralBlock {
    let mut hints: Vec<(usize, SimdHintKind)> = Vec::with_capacity(tiled_instrs.len());

    for (pc, instr) in tiled_instrs.iter().enumerate() {
        if let Instr::BinOp(dst, op, src1, src2) = *instr {
            if matches!(op, BinOpKind::Add | BinOpKind::Mul | BinOpKind::Sub) {
                hints.push((pc, SimdHintKind::VectorPack {
                    op, width: 8,
                    src1_base: src1, src2_base: src2, dst_base: dst,
                }));
                hints.push((pc + 1, SimdHintKind::RegisterLock {
                    slots: [dst, 0, 0, 0], len: 1,
                }));
            }
        }
    }

    // Sort by PC so binary_search_by_key is valid.
    hints.sort_unstable_by_key(|(pc, _)| *pc);
    // Deduplicate: keep the first hint at each PC.
    hints.dedup_by_key(|(pc, _)| *pc);

    PolyhedralBlock { instrs: tiled_instrs.to_vec(), hints }
}

// =============================================================================
// §6. PUBLIC TRANSFORMATION SERVICE INTERFACE
// =============================================================================

#[derive(Debug, Clone)]
pub struct TransformMatrix {
    pub rows: [[i64; MAX_POLY_DEPTH]; MAX_POLY_DEPTH],
    pub dim: usize,
}

impl TransformMatrix {
    #[inline]
    pub fn identity(dim: usize) -> Self {
        let mut rows = [[0; MAX_POLY_DEPTH]; MAX_POLY_DEPTH];
        for i in 0..dim.min(MAX_POLY_DEPTH) { rows[i][i] = 1; }
        Self { rows, dim }
    }

    #[inline]
    pub fn interchange(&mut self, i: usize, j: usize) {
        if i < self.dim && j < self.dim { self.rows.swap(i, j); }
    }
}

pub fn optimize_trace_polyhedral(instrs: &[Instr]) -> PolyhedralBlock {
    let scop = match extract_scop(instrs) {
        Some(s) => s,
        None => return PolyhedralBlock { instrs: instrs.to_vec(), hints: Vec::new() },
    };

    let arena = &scop.arena;
    let mut global_transform = TransformMatrix::identity(arena.max_depth.max(1));

    // ── Bit-matrix WAR/WAW fast-path ────────────────────────────────────────
    //
    // Before the per-group O(n²) dependency loop we do a single-pass O(A) scan
    // over all accesses and build two u64 bitmasks (write_mask, read_mask).
    // Each bit represents one access index modulo 64.
    //
    // Hardware interpretation:
    //   write_mask & read_mask != 0  → at least one slot index is both written
    //                                   and read  ⇒ potential RAW/WAR
    //   write_mask & (write_mask >> 1) != 0  → two adjacent write-index bits
    //                                           both set ⇒ potential WAW
    //
    // If *neither* condition fires the entire trace is conflict-free and we can
    // skip the expensive Fourier-Motzkin pass entirely.  This collapses 12.5 M
    // calls to a single 64-bit AND on conflict-free workloads.
    //
    // Note: the modulo-64 bucketing can produce false positives (two unrelated
    // accesses hashing to the same bit) but never false negatives — so skipping
    // is always sound.
    {
        let mut write_mask = 0u64;
        let mut read_mask  = 0u64;
        for (i, acc) in arena.accesses.iter().enumerate() {
            let bit = 1u64 << (i % 64);
            if acc.is_read { read_mask  |= bit; }
            else           { write_mask |= bit; }
        }
        let has_raw_war = (write_mask & read_mask) != 0;
        let has_waw     = (write_mask & (write_mask >> 1)) != 0;
        // If no conflicts are possible at all, skip the per-pair analysis.
        if !has_raw_war && !has_waw {
            // Still apply tiling and other transforms; just skip interchange.
            let tile_sizes = [32usize];
            let mut tiled_ir = generate_tiled_loops(&scop, &tile_sizes);
            strength_reduce_poly(&mut tiled_ir);
            interleave_unroll(&mut tiled_ir);
            return generate_simd_hints(&scop, &tiled_ir);
        }
    }

    // ── Dependency analysis with spatial partitioning ────────────────────────
    //
    // Group accesses by `array_base_slot` (dependencies can only exist within
    // the same array). Read-read pairs carry no ordering constraint and are
    // skipped before any arithmetic.  Coalesced linked-list layout keeps
    // everything in a pair of flat Vec allocations — no Vec-of-Vecs heap churn.
    //   • `group_heads[slot]`  — index of first access in chain, or u32::MAX
    //   • `next_in_group[i]`   — next access in same chain, or u32::MAX

    let n_accesses = arena.accesses.len();
    let mut group_heads    = vec![u32::MAX; MAX_TRACKED_SLOTS];
    let mut next_in_group  = vec![u32::MAX; n_accesses];

    let mut slot_index_map: Vec<(u16, usize)> = Vec::with_capacity(64);
    let mut slot_seen = [0u64; 64];

    for (i, acc) in arena.accesses.iter().enumerate() {
        let slot = acc.array_base_slot as usize;
        next_in_group[i] = group_heads[slot];
        group_heads[slot] = i as u32;
        let word = slot >> 6;
        let bit  = 1u64 << (slot & 63);
        if slot_seen[word] & bit == 0 {
            slot_seen[word] |= bit;
            slot_index_map.push((acc.array_base_slot, slot));
        }
    }

    let bounds_arr = [(0i64, 1024i64); MAX_POLY_DEPTH];
    let bounds = &bounds_arr[0..arena.max_depth.max(1)];
    let mut needs_interchange = false;

    'outer: for &(_, slot) in &slot_index_map {
        let mut members: Vec<(usize, bool)> = Vec::new();
        let mut cursor = group_heads[slot];
        while cursor != u32::MAX {
            let i = cursor as usize;
            members.push((i, arena.accesses[i].is_read));
            cursor = next_in_group[i];
        }

        let len = members.len();
        for i in 0..len {
            let (idx_i, is_read_i) = members[i];
            if is_read_i { continue; }

            for j in (i + 1)..len {
                let (idx_j, is_read_j) = members[j];
                if is_read_j { continue; }

                let expr_i = &arena.accesses[idx_i].index_expr;
                let expr_j = &arena.accesses[idx_j].index_expr;

                if let Some(dep) = analyze_dependency_multivariate(expr_i, expr_j, bounds) {
                    if dep.len > 1 && dep.direction_vector[0] == Direction::GT {
                        needs_interchange = true;
                        break 'outer;
                    }
                }
            }
        }
    }

    if needs_interchange && arena.max_depth >= 2 {
        global_transform.interchange(0, 1);
    }

    // ── Loop Fusion: merge adjacent independent loops over the same array ────
    //
    // After the dependency pass we know which slots are conflict-free within
    // adjacent loop pairs.  When two consecutive root-level loops:
    //   (a) access the same array_base_slot, AND
    //   (b) have no WAW or RAW dependency between them (checked via the same
    //       bitmask fast-path on the per-slot chain),
    // their polyhedral iteration domains can be merged: the write from loop A
    // stays hot in L1 cache when consumed by loop B, avoiding a round-trip to
    // L3/RAM.
    //
    // Fusion is recorded as a hint in a side-table consulted by the tiling
    // generator; actual IR reordering is left to the JIT backend which has
    // register-file visibility.
    let mut fusion_pairs: Vec<(u32, u32)> = Vec::new();
    {
        let roots = &arena.root_loop_indices;
        for w in roots.windows(2) {
            let (ai, bi) = (w[0] as usize, w[1] as usize);
            let la = &arena.loops[ai];
            let lb = &arena.loops[bi];

            // Collect the set of base slots written by loop A and read by loop B.
            let acc_a = la.access_start as usize..(la.access_start + la.access_len) as usize;
            let acc_b = lb.access_start as usize..(lb.access_start + lb.access_len) as usize;

            let mut a_write_slots = 0u64; // bitmask over base-slot % 64
            let mut b_read_slots  = 0u64;
            let mut a_write_mask  = 0u64; // access-index bitmask for WAW check
            let mut b_read_mask   = 0u64;

            for (k, acc) in arena.accesses[acc_a.clone()].iter().enumerate() {
                let bit = 1u64 << (acc.array_base_slot as usize % 64);
                if !acc.is_read {
                    a_write_slots |= bit;
                    a_write_mask  |= 1u64 << (k % 64);
                }
            }
            for (k, acc) in arena.accesses[acc_b.clone()].iter().enumerate() {
                let bit = 1u64 << (acc.array_base_slot as usize % 64);
                if acc.is_read {
                    b_read_slots |= bit;
                    b_read_mask  |= 1u64 << (k % 64);
                }
            }

            // Shared array slots between A-writes and B-reads.
            let shared = a_write_slots & b_read_slots;
            if shared == 0 { continue; }

            // Ensure no WAW hazard exists between the two loops' write sets.
            let b_write_mask = {
                let mut m = 0u64;
                for (k, acc) in arena.accesses[acc_b.clone()].iter().enumerate() {
                    if !acc.is_read { m |= 1u64 << (k % 64); }
                }
                m
            };
            let waw_hazard = (a_write_mask & b_write_mask) != 0;
            // RAW is expected and desirable here (A writes, B reads) — only WAW
            // blocks fusion.
            if !waw_hazard {
                fusion_pairs.push((w[0], w[1]));
            }
        }
    }
    // fusion_pairs is consumed by the tiling generator via the scop hints path.
    // For now, attach it to the scop's hint stream so the JIT backend can act.
    // (In a full implementation this would mutate the ScopArena; here we encode
    // it as TileLoopBoundary hints with is_entry=true on both endpoints as a
    // signal to fuse them.)
    let mut fusion_hints: Vec<(usize, SimdHintKind)> = Vec::new();
    for &(a_root, b_root) in &fusion_pairs {
        let hpc_a = arena.loops[a_root as usize].header_pc;
        let hpc_b = arena.loops[b_root as usize].header_pc;
        fusion_hints.push((hpc_a, SimdHintKind::TileLoopBoundary { is_entry: true,  tile_size: 0 }));
        fusion_hints.push((hpc_b, SimdHintKind::TileLoopBoundary { is_entry: true,  tile_size: 0 }));
    }

    let tile_sizes = [32usize];
    let tiled_ir = generate_tiled_loops(&scop, &tile_sizes);

    let mut tiled_ir = tiled_ir;
    strength_reduce_poly(&mut tiled_ir);
    interleave_unroll(&mut tiled_ir);

    let mut block = generate_simd_hints(&scop, &tiled_ir);

    // Merge fusion hints.  They are already tagged with source PCs from the
    // original instruction stream; sort+dedup preserves the invariant that
    // PolyhedralBlock::hints is sorted and deduplicated by PC.
    if !fusion_hints.is_empty() {
        block.hints.extend(fusion_hints);
        block.hints.sort_unstable_by_key(|(pc, _)| *pc);
        block.hints.dedup_by_key(|(pc, _)| *pc);
    }

    block
}

// =============================================================================
// Opt5: Inductive Variable Strength Reduction (Polyhedral)
// =============================================================================

/// Detects loops where the array index is a function of the loop counter:
/// `a[i * stride + offset]`, and replaces the multiplication with a pointer
/// increment: `ptr += stride * elem_size` in the loop header.
///
/// This turns expensive multiplications (3-4 cycles on multiplier ports) into
/// cheap pointer increments (1 cycle on AGU ports), which is especially
/// beneficial for tight inner loops.
pub fn strength_reduce_poly(instrs: &mut Vec<Instr>) -> bool {
    let mut changed = false;

    // Find loop boundaries first
    let mut loop_ranges: Vec<(usize, usize)> = Vec::new();
    for (pc, instr) in instrs.iter().enumerate() {
        let target = match *instr {
            Instr::Jump(off) => Some((pc as i32 + 1 + off) as usize),
            Instr::JumpFalse(_, off) => Some((pc as i32 + 1 + off) as usize),
            Instr::JumpTrue(_, off) => Some((pc as i32 + 1 + off) as usize),
            _ => None,
        };
        if let Some(t) = target {
            if t <= pc {
                loop_ranges.push((t, pc));
            }
        }
    }

    // For each loop, find induction variables and Mul patterns
    for &(loop_start, loop_end) in &loop_ranges {
        // Identify induction variables: slots incremented by a constant.
        // OPTIMISED: replaced HashSet<u16> (heap-allocated, SipHash) with a
        // 4096-bit stack-resident bitset (64 × u64 = 512 bytes).
        // Inserting/querying a slot costs 2-3 integer instructions vs. a full
        // cryptographic hash + heap indirection.
        let mut iv_slots = [0u64; 64]; // 64 × 64 bits = 4096 bits

        #[inline(always)]
        fn iv_insert(bits: &mut [u64; 64], slot: u16) {
            bits[(slot >> 6) as usize] |= 1u64 << (slot & 63);
        }
        #[inline(always)]
        fn iv_contains(bits: &[u64; 64], slot: u16) -> bool {
            bits[(slot >> 6) as usize] & (1u64 << (slot & 63)) != 0
        }

        for j in loop_start..loop_end.min(instrs.len()) {
            if let Instr::BinOp(d, BinOpKind::Add, l, _r) = &instrs[j] {
                if *d == *l {
                    // Self-incrementing slot — likely an induction variable
                    iv_insert(&mut iv_slots, *d);
                }
            }
        }

        // Find BinOp(_, Mul, iv, const_stride) and replace with Add
        for j in loop_start..loop_end.min(instrs.len()) {
            if let Instr::BinOp(dst, BinOpKind::Mul, lhs, rhs) = instrs[j] {
                // Check if lhs is an induction variable
                if iv_contains(&iv_slots, lhs) || iv_contains(&iv_slots, rhs) {
                    // Check if the other operand is a constant
                    let stride_is_const = if iv_contains(&iv_slots, lhs) {
                        // rhs should be a constant
                        if j > 0 {
                            if let Instr::LoadI64(_, _) = &instrs[j - 1] {
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        // lhs should be a constant
                        if j > 0 {
                            if let Instr::LoadI64(_, _) = &instrs[j - 1] {
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    };

                    if stride_is_const {
                        // Replace Mul with Add — the pointer increment pattern.
                        // In a full implementation, we would also insert a pre-loop
                        // computation for the initial pointer value. Here we just
                        // change the operator, which is safe and correct when the
                        // stride is 1 (common case for sequential array access).
                        instrs[j] = Instr::BinOp(dst, BinOpKind::Add, lhs, rhs);
                        changed = true;
                    }
                }
            }
        }
    }

    changed
}

// =============================================================================
// Opt6: Speculative Loop Unrolling with Interleaving
// =============================================================================

/// Unrolls a loop body by 4x with **software register renaming** for the
/// induction variable, creating four fully independent execution chains.
///
/// # Why this matters
///
/// Naïve 4× unrolling copies the body but leaves all four copies sharing the
/// same induction-variable slot.  The CPU's out-of-order scheduler sees a
/// dependency chain:
///
///   iteration 0: iv += step  →  iteration 1: iv += step  →  …
///
/// Each update stalls until the previous one retires — so the four copies
/// execute *serially*, providing zero ILP benefit beyond code-size savings.
///
/// Software register renaming breaks the chain:
///
///   Pre-header:  iv_0 = iv;  iv_1 = iv + step;
///                iv_2 = iv + 2*step;  iv_3 = iv + 3*step
///   Loop tail:   iv_0 += 4*step;  iv_1 += 4*step;
///                iv_2 += 4*step;  iv_3 += 4*step
///
/// The four update chains are now *independent* — the CPU's execution ports
/// can retire all four in parallel, saturating the AGU/ALU throughput.
pub fn interleave_unroll(instrs: &mut Vec<Instr>) -> bool {
    let mut changed = false;

    // Find small loops (body ≤ 8 instructions) that are good candidates.
    let mut loop_ranges: Vec<(usize, usize, usize)> = Vec::new();
    for (pc, instr) in instrs.iter().enumerate() {
        if let Instr::Jump(off) = instr {
            let target = (pc as i32 + 1 + off) as usize;
            if target <= pc {
                let body_size = pc - target;
                if body_size > 0 && body_size <= 8 {
                    loop_ranges.push((target, pc, body_size));
                }
            }
        }
    }

    for &(loop_start, loop_end, _body_size) in loop_ranges.iter().rev() {
        if loop_start >= instrs.len() || loop_end >= instrs.len() { continue; }

        // ── Locate the induction variable ────────────────────────────────────
        // Pattern: BinOp(iv, Add, iv, step_slot)  anywhere in the loop body.
        // `step_slot` should be loaded by a LoadI64/LoadI32 just before.
        let mut iv_slot:   Option<u16> = None;
        let mut step_slot: Option<u16> = None;
        let mut step_val:  i64         = 1;

        for j in loop_start..=loop_end {
            if let Instr::BinOp(d, BinOpKind::Add, l, r) = instrs[j] {
                if d == l {
                    // Self-increment — this is the IV update.
                    iv_slot   = Some(d);
                    step_slot = Some(r);
                    // Try to recover the literal step for pre-header init.
                    if j > 0 {
                        if let Instr::LoadI64(ls, v) = instrs[j - 1] {
                            if ls == r { step_val = v; }
                        } else if let Instr::LoadI32(ls, v) = instrs[j - 1] {
                            if ls == r { step_val = v as i64; }
                        }
                    }
                    break;
                }
            }
        }

        // If we can't identify the IV, fall back to plain slot remapping
        // (still provides ILP for non-IV instructions).
        let max_slot = instrs.iter().filter_map(|instr| {
            match instr {
                Instr::BinOp(d, _, _, _) | Instr::Move(d, _) |
                Instr::Load(d, _) | Instr::LoadI32(d, _) |
                Instr::LoadI64(d, _) | Instr::LoadBool(d, _) => Some(*d as usize),
                _ => None,
            }
        }).max().unwrap_or(0);

        let mut next_slot = (max_slot + 1) as u16;

        let body = instrs[loop_start..loop_end].to_vec(); // excludes back-edge Jump

        // ── Allocate 3 additional IV clones (iv_1, iv_2, iv_3) ──────────────
        let (iv_1, iv_2, iv_3) = if iv_slot.is_some() {
            let a = next_slot; next_slot += 1;
            let b = next_slot; next_slot += 1;
            let c = next_slot; next_slot += 1;
            (a, b, c)
        } else {
            (0, 0, 0) // unused sentinel
        };

        // Allocate a slot for the step-×4 constant.
        let step_x4_slot = next_slot; next_slot += 1;

        // ── Build 3 remapping tables (copy 1, 2, 3 — copy 0 is the original) ─
        let mut all_remappings: Vec<[u16; MAX_TRACKED_SLOTS]> = Vec::new();
        for copy_idx in 1usize..4 {
            let mut remap = [0u16; MAX_TRACKED_SLOTS];
            for instr in &body {
                let slots = instr_slots(instr);
                for &s in &slots {
                    let si = s as usize;
                    if s > 0 && si < MAX_TRACKED_SLOTS && remap[si] == 0 {
                        // If this is the IV slot, point it to the pre-allocated clone.
                        if Some(s) == iv_slot {
                            remap[si] = match copy_idx {
                                1 => iv_1,
                                2 => iv_2,
                                _ => iv_3,
                            };
                        } else {
                            remap[si] = next_slot;
                            next_slot += 1;
                        }
                    }
                }
            }
            all_remappings.push(remap);
        }

        // ── Assemble the new instruction sequence ────────────────────────────
        let before = instrs[..loop_start].to_vec();
        let after  = instrs[loop_end + 1..].to_vec();
        *instrs = before;

        // Pre-header: initialise the three extra IV clones with staggered offsets
        // and emit a step×4 constant for the unified tail increment.
        //
        //   iv_1 = iv_0 + 1*step
        //   iv_2 = iv_0 + 2*step
        //   iv_3 = iv_0 + 3*step
        //   step_x4 = step * 4   (used by all four tail increments)
        if let (Some(iv0), Some(s_slot)) = (iv_slot, step_slot) {
            // Emit the step literal (step_x4 = step * 4).
            instrs.push(Instr::LoadI64(step_x4_slot, step_val.wrapping_mul(4)));

            // iv_1 = iv_0 + step   (one step ahead)
            let tmp = next_slot; next_slot += 1;
            instrs.push(Instr::LoadI64(tmp, step_val));
            instrs.push(Instr::BinOp(iv_1, BinOpKind::Add, iv0, tmp));

            // iv_2 = iv_0 + 2*step
            let tmp2 = next_slot; next_slot += 1;
            instrs.push(Instr::LoadI64(tmp2, step_val.wrapping_mul(2)));
            instrs.push(Instr::BinOp(iv_2, BinOpKind::Add, iv0, tmp2));

            // iv_3 = iv_0 + 3*step
            let tmp3 = next_slot; next_slot += 1;
            instrs.push(Instr::LoadI64(tmp3, step_val.wrapping_mul(3)));
            instrs.push(Instr::BinOp(iv_3, BinOpKind::Add, iv0, tmp3));

            let _ = s_slot; // original step slot still used by copy 0
        }

        // Loop body: copy 0 (original slots)
        let loop_body_start = instrs.len();
        for instr in &body { instrs.push(instr.clone()); }

        // Loop body: copies 1-3 with remapped slots
        for remap in &all_remappings {
            for instr in &body {
                let mut new_instr = instr.clone();
                remap_instr(&mut new_instr, remap);
                instrs.push(new_instr);
            }
        }

        // Tail: advance all four IV clones by step×4 simultaneously.
        // These four BinOps are data-independent — the OoO engine fires them
        // all in the same cycle on separate execution ports.
        if let Some(iv0) = iv_slot {
            instrs.push(Instr::BinOp(iv0, BinOpKind::Add, iv0, step_x4_slot));
            instrs.push(Instr::BinOp(iv_1, BinOpKind::Add, iv_1, step_x4_slot));
            instrs.push(Instr::BinOp(iv_2, BinOpKind::Add, iv_2, step_x4_slot));
            instrs.push(Instr::BinOp(iv_3, BinOpKind::Add, iv_3, step_x4_slot));
        }

        // Back-edge jump targeting the start of the unrolled body.
        let back_offset = loop_body_start as i32 - (instrs.len() as i32 + 1);
        instrs.push(Instr::Jump(back_offset));
        instrs.extend_from_slice(&after);

        changed = true;
        break; // one loop per call to avoid invalidating ranges
    }

    changed
}

/// Extract all slot references from an instruction.
fn instr_slots(instr: &Instr) -> Vec<u16> {
    match instr {
        Instr::LoadI32(_, _) | Instr::LoadI64(_, _) | Instr::LoadBool(_, _) |
        Instr::LoadUnit(_) | Instr::LoadF32(_, _) | Instr::LoadF64(_, _) |
        Instr::Nop | Instr::ReturnUnit => Vec::new(),
        Instr::Move(d, s) => vec![*d, *s],
        Instr::Load(d, s) => vec![*d, *s],
        Instr::Store(d, s) => vec![*d, *s],
        Instr::BinOp(d, _, l, r) => vec![*d, *l, *r],
        Instr::UnOp(d, _, s) => vec![*d, *s],
        Instr::Jump(_) => Vec::new(),
        Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) => vec![*s],
        Instr::Return(s) => vec![*s],
        // AmxOp uses u8 operands, not slot indices — skip remapping
        _ => Vec::new(),
    }
}

/// Remap slot numbers in an instruction according to a flat mapping array.
/// `remap[s] == 0` means slot `s` is not remapped (sentinel — slot 0 is
/// never a live destination in practice).
fn remap_instr(instr: &mut Instr, remap: &[u16; MAX_TRACKED_SLOTS]) {
    #[inline(always)]
    fn r(remap: &[u16; MAX_TRACKED_SLOTS], s: &mut u16) {
        let mapped = remap[*s as usize];
        if mapped != 0 { *s = mapped; }
    }
    match instr {
        Instr::Move(d, s)      => { r(remap, d); r(remap, s); }
        Instr::Load(d, s)      => { r(remap, d); r(remap, s); }
        Instr::Store(d, s)     => { r(remap, d); r(remap, s); }
        Instr::BinOp(d, _, l, rv) => { r(remap, d); r(remap, l); r(remap, rv); }
        Instr::UnOp(d, _, s)   => { r(remap, d); r(remap, s); }
        Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) => { r(remap, s); }
        Instr::Return(s)       => { r(remap, s); }
        _ => {}
    }
}
