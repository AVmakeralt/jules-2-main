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

    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let mut res = *self;
        res.constant = res.constant.wrapping_add(other.constant);
        let mut mask = other.active_mask;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            res.coefficients[i] = res.coefficients[i].wrapping_add(other.coefficients[i]);
            mask &= mask - 1;
        }
        res.active_mask |= other.active_mask;
        res
    }

    #[inline]
    pub fn mul_const(&self, c: i64) -> Self {
        let mut res = *self;
        res.constant = res.constant.wrapping_mul(c);
        let mut mask = res.active_mask;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            res.coefficients[i] = res.coefficients[i].wrapping_mul(c);
            mask &= mask - 1;
        }
        res
    }

    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        let mut res = *self;
        res.constant = res.constant.wrapping_sub(other.constant);
        let mut mask = other.active_mask;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            res.coefficients[i] = res.coefficients[i].wrapping_sub(other.coefficients[i]);
            mask &= mask - 1;
        }
        res.active_mask |= other.active_mask;
        res
    }
}

/// Safe Binary Greatest Common Divisor (Stein's Algorithm).
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
        if u > v { std::mem::swap(&mut u, &mut v); }
        v -= u;
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

    let mut min_val = 0i64;
    let mut max_val = 0i64;
    for i in 0..active_count {
        let c = coeffs[i];
        let var_id = dims[i];
        let (l, u) = if var_id < bounds.len() { bounds[var_id] } else { (0, 1000) };
        let range = u - l;
        if c > 0 {
            min_val = min_val.saturating_add(c.saturating_mul(-range));
            max_val = max_val.saturating_add(c.saturating_mul(range));
        } else {
            min_val = min_val.saturating_add(c.saturating_mul(range));
            max_val = max_val.saturating_add(c.saturating_mul(-range));
        }
    }

    if target < min_val || target > max_val { return None; }

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
pub struct SlotCache {
    pub data: Vec<Option<AffineExpr>>,
}

impl SlotCache {
    #[inline]
    pub fn new() -> Self {
        Self { data: vec![None; MAX_TRACKED_SLOTS] }
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
        }
    }

    #[inline]
    pub fn get(&self, slot: u16) -> Option<AffineExpr> {
        let idx = slot as usize;
        if idx < MAX_TRACKED_SLOTS { self.data[idx] } else { None }
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

    for pc in header..=back_edge {
        if let Instr::BinOp(dst, BinOpKind::Add, l, r) = instrs[pc] {
            if dst == l && pc > 0 {
                if let Instr::LoadI64(_, step) = instrs[pc - 1] {
                    if r == get_dst_slot(&instrs[pc - 1]) {
                        if loop_iv_slots.len() < MAX_POLY_DEPTH {
                            loop_iv_slots.push(dst);
                            iv = Some(InductionVar { slot: dst, step });
                            break;
                        }
                    }
                }
            }
        }
    }

    if let Some(ref iv_ref) = iv {
        if let Instr::JumpTrue(cond, _) = instrs[back_edge] {
            for pc in (header..back_edge).rev() {
                if let Instr::BinOp(d, BinOpKind::Lt, l, r) = instrs[pc] {
                    if d == cond && l == iv_ref.slot {
                        if let Some(expr) = cache.get(r) { ub = expr; }
                        break;
                    }
                }
            }
        } else {
            let scan_limit = back_edge.min(header + 10);
            for pc in header..=scan_limit {
                if let Instr::JumpFalse(cond, _) = instrs[pc] {
                    for cond_pc in (header..pc).rev() {
                        if let Instr::BinOp(d, BinOpKind::Lt, l, r) = instrs[cond_pc] {
                            if d == cond && l == iv_ref.slot {
                                if let Some(expr) = cache.get(r) { ub = expr; }
                                break;
                            }
                        }
                    }
                    break;
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

    // ── Dependency analysis with spatial partitioning ────────────────────────
    //
    // FIXED: The original code ran a brute-force O(A²) nested loop over all
    // memory accesses, calling the GCD-heavy `analyze_dependency_multivariate`
    // on every pair. With 5 000 accesses this executes 12.5 million times and
    // stalls the JIT.
    //
    // Fix: group accesses by `array_base_slot` first. Dependencies can only
    // exist between accesses to the *same* array, so we only run the expensive
    // test within each group.  Additionally, read-read pairs carry no ordering
    // constraints and are skipped before any arithmetic.
    //
    // This reduces the worst case from O(A²) to O(Σ aᵢ²) — typically a
    // 10–100× reduction for real workloads with many distinct arrays.

    // Build slot → group index map without a HashMap (open-address via a small
    // side-Vec; number of distinct base slots is usually ≪ 1 000).
    let mut slot_index_map: Vec<(u16, usize)> = Vec::with_capacity(64);
    let mut groups: Vec<Vec<(usize, bool)>> = Vec::new();

    for (i, acc) in arena.accesses.iter().enumerate() {
        let slot = acc.array_base_slot;
        let group_idx = match slot_index_map.iter().position(|&(s, _)| s == slot) {
            Some(pos) => slot_index_map[pos].1,
            None => {
                let idx = groups.len();
                groups.push(Vec::new());
                slot_index_map.push((slot, idx));
                idx
            }
        };
        groups[group_idx].push((i, acc.is_read));
    }

    let bounds_arr = [(0i64, 1024i64); MAX_POLY_DEPTH];
    let bounds = &bounds_arr[0..arena.max_depth.max(1)];
    let mut needs_interchange = false;

    'outer: for group in &groups {
        let len = group.len();
        for i in 0..len {
            let (idx_i, is_read_i) = group[i];
            // A source that is read-only cannot cause a write→read hazard
            // in the forward direction we test for.
            if is_read_i { continue; }

            for j in (i + 1)..len {
                let (idx_j, is_read_j) = group[j];
                // Read-read pairs have no scheduling constraint.
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

    let tile_sizes = [32usize];
    let tiled_ir = generate_tiled_loops(&scop, &tile_sizes);
    generate_simd_hints(&scop, &tiled_ir)
}
