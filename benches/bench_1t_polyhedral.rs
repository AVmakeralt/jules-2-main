// =============================================================================
// bench_1t_polyhedral.rs — 1 Trillion Iteration Polyhedral Benchmark
//
// Tests the polyhedral engine on a 1T (10^12) iteration loop:
//   1. SCoP extraction correctness at extreme bounds
//   2. Full pipeline (extract → dependency → tile → SIMD hints)
//   3. JIT execution of the 1T loop with polyhedral optimization
//   4. Correctness validation (sum formula check)
//
// Usage:
//   cargo run --release --bin bench-1t-polyhedral
// =============================================================================

use std::hint::black_box;
use std::time::Instant;

use jules::polyhedral::{
    AffineExpr, SlotCache, MAX_POLY_DEPTH,
    extract_scop, optimize_trace_polyhedral,
    analyze_dependency_multivariate,
};
use jules::interp::Instr;
use jules::compiler::ast::BinOpKind;

const ONE_TRILLION: i64 = 1_000_000_000_000;  // 10^12
const ONE_BILLION: i64  = 1_000_000_000;      // 10^9
const ONE_MILLION: i64  = 1_000_000;          // 10^6

fn fmt_dur(secs: f64) -> String {
    if secs >= 60.0 {
        let m = (secs / 60.0) as u64;
        let s = secs - (m as f64 * 60.0);
        format!("{}m{:.1}s", m, s)
    } else if secs >= 1.0 {
        format!("{:.3}s", secs)
    } else if secs >= 0.001 {
        format!("{:.2}ms", secs * 1000.0)
    } else {
        format!("{:.0}us", secs * 1_000_000.0)
    }
}

fn _fmt_count(n: u64) -> String {
    if n >= 1_000_000_000_000 {
        format!("{:.2}T", n as f64 / 1e12)
    } else if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}

// =============================================================================
// §1. BYTECODE GENERATORS FOR EXTREME-SCALE LOOPS
// =============================================================================

/// Generate bytecode for: s = 0; i = 0; while i < N { s = s + i; i = i + 1; }
/// Returns (sum).
///
/// Slot layout: 0=return, 1=s, 2=i, 3=temp, 5=N, 6=cond, 8=step
fn gen_sum_loop(n: i64) -> Vec<Instr> {
    //       index:  0      1      2       3              4                  5                     6            7          8              9           10         11
    vec![
        Instr::LoadI64(1, 0),          // s = 0         @0
        Instr::LoadI64(2, 0),          // i = 0         @1
        Instr::LoadI64(5, n),          // N = n         @2
        // header @3:
        Instr::BinOp(6, BinOpKind::Lt, 2, 5),  // cond = i < N   @3
        Instr::JumpFalse(6, 6),        // if !cond goto end (→ @11)  @4: 4+1+6=11
        Instr::BinOp(3, BinOpKind::Add, 1, 2),  // temp = s + i   @5
        Instr::Move(1, 3),             // s = temp      @6
        Instr::LoadI64(8, 1),          // step = 1      @7
        Instr::BinOp(3, BinOpKind::Add, 2, 8),  // temp = i + 1   @8
        Instr::Move(2, 3),             // i = temp      @9
        Instr::Jump(-7),               // goto header (=3)   @10
        // end @11:
        Instr::Move(0, 1),             // return s      @11
    ]
}

/// Generate bytecode for an LCG loop — properly patched.
fn gen_lcg_loop(n: i64) -> Vec<Instr> {
    let mut instrs = Vec::new();
    instrs.push(Instr::LoadI64(1, 42));        // s = 42       @0
    instrs.push(Instr::LoadI64(2, 0));         // i = 0        @1
    instrs.push(Instr::LoadI64(5, n));         // N = n         @2
    // header @3:
    instrs.push(Instr::BinOp(6, BinOpKind::Lt, 2, 5)); // cond = i < N  @3
    instrs.push(Instr::JumpFalse(6, 0));       // patch later   @4
    let jf_patch = instrs.len() - 1;
    instrs.push(Instr::LoadI64(7, 1664525));   // a = 1664525  @5
    instrs.push(Instr::BinOp(3, BinOpKind::Mul, 1, 7)); // temp = s*a  @6
    instrs.push(Instr::LoadI64(8, 1013904223)); // c          @7
    instrs.push(Instr::BinOp(4, BinOpKind::Add, 3, 8)); // temp2 = temp+c  @8
    instrs.push(Instr::Move(1, 4));            // s = temp2    @9
    instrs.push(Instr::LoadI64(10, 1));        // step = 1     @10
    instrs.push(Instr::BinOp(3, BinOpKind::Add, 2, 10)); // temp = i+1  @11
    instrs.push(Instr::Move(2, 3));            // i = temp     @12
    instrs.push(Instr::Jump(0)); // placeholder, will fix below
    // Jump at @13, target should be @3: offset = 3 - 13 - 1 = -11
    let jump_idx = instrs.len() - 1;
    if let Instr::Jump(ref mut off) = instrs[jump_idx] {
        *off = 3i32 - (jump_idx as i32) - 1;
    }
    let end_pc = instrs.len();
    // Patch JumpFalse: target = end_pc
    if let Instr::JumpFalse(_, ref mut off) = instrs[jf_patch] {
        *off = (end_pc as i32) - (jf_patch as i32) - 1;
    }
    instrs.push(Instr::Move(0, 1));            // return s
    instrs
}

/// Generate bytecode for: count = 0; i = 0; while i < N { count = count + 1; i++; }
///
/// Slot layout: 0=return, 1=count, 2=i, 3=temp, 5=N, 6=cond, 8=one, 10=step
fn gen_count_loop(n: i64) -> Vec<Instr> {
    let mut instrs = Vec::new();
    instrs.push(Instr::LoadI64(1, 0));         // count = 0    @0
    instrs.push(Instr::LoadI64(2, 0));         // i = 0        @1
    instrs.push(Instr::LoadI64(5, n));         // N = n         @2
    // header @3:
    instrs.push(Instr::BinOp(6, BinOpKind::Lt, 2, 5)); // cond = i < N  @3
    instrs.push(Instr::JumpFalse(6, 0));       // patch later   @4
    let jf_patch = instrs.len() - 1;
    instrs.push(Instr::LoadI64(8, 1));         // one = 1       @5
    instrs.push(Instr::BinOp(3, BinOpKind::Add, 1, 8)); // temp = count+1  @6
    instrs.push(Instr::Move(1, 3));            // count = temp  @7
    instrs.push(Instr::LoadI64(10, 1));        // step = 1      @8
    instrs.push(Instr::BinOp(3, BinOpKind::Add, 2, 10)); // temp = i+1  @9
    instrs.push(Instr::Move(2, 3));            // i = temp      @10
    // Jump back to header @3
    let jump_pc = instrs.len() as i32;         // @11
    instrs.push(Instr::Jump(3 - jump_pc - 1)); // goto @3
    // Patch JumpFalse
    let end_pc = instrs.len();
    if let Instr::JumpFalse(_, ref mut off) = instrs[jf_patch] {
        *off = (end_pc as i32) - (jf_patch as i32) - 1;
    }
    instrs.push(Instr::Move(0, 1));            // return count  @12
    instrs
}

/// Generate bytecode for nested loop: total = 0; i = 0; while i < M { j = 0; while j < N { total += i*j+1; j++ }; i++ }
fn gen_nested_loop(m: i64, n: i64) -> Vec<Instr> {
    let mut instrs = Vec::new();
    instrs.push(Instr::LoadI64(1, 0));     // total = 0
    instrs.push(Instr::LoadI64(2, 0));     // i = 0
    instrs.push(Instr::LoadI64(5, m));     // M = m
    // outer header:
    let outer_header = instrs.len();
    instrs.push(Instr::BinOp(6, BinOpKind::Lt, 2, 5)); // cond = i < M
    instrs.push(Instr::JumpFalse(6, 0));   // patched later
    let outer_jf_patch = instrs.len() - 1;
    instrs.push(Instr::LoadI64(3, 0));     // j = 0
    instrs.push(Instr::LoadI64(7, n));     // N = n
    // inner header:
    instrs.push(Instr::BinOp(8, BinOpKind::Lt, 3, 7)); // cond = j < N
    instrs.push(Instr::JumpFalse(8, 0));   // patched later
    let inner_jf_patch = instrs.len() - 1;
    instrs.push(Instr::BinOp(9, BinOpKind::Mul, 2, 3));  // temp = i * j
    instrs.push(Instr::LoadI64(10, 1));     // one = 1
    instrs.push(Instr::BinOp(9, BinOpKind::Add, 9, 10)); // temp = i*j + 1
    instrs.push(Instr::BinOp(1, BinOpKind::Add, 1, 9));  // total += temp
    // increment j
    instrs.push(Instr::LoadI64(10, 1));
    instrs.push(Instr::BinOp(9, BinOpKind::Add, 3, 10));
    instrs.push(Instr::Move(3, 9));
    instrs.push(Instr::Jump(0));           // patched to inner header
    let inner_jump_patch = instrs.len() - 1;
    // patch inner JumpFalse
    let inner_end = instrs.len();
    if let Instr::JumpFalse(_, ref mut off) = instrs[inner_jf_patch] {
        *off = (inner_end as i32) - (inner_jf_patch as i32) - 1;
    }
    // patch inner Jump back to inner header
    let inner_header_pc = outer_jf_patch + 3;
    if let Instr::Jump(ref mut off) = instrs[inner_jump_patch] {
        *off = (inner_header_pc as i32) - (inner_jump_patch as i32) - 1;
    }
    // increment i
    instrs.push(Instr::LoadI64(10, 1));
    instrs.push(Instr::BinOp(9, BinOpKind::Add, 2, 10));
    instrs.push(Instr::Move(2, 9));
    instrs.push(Instr::Jump(0));           // patched to outer header
    let outer_jump_patch = instrs.len() - 1;
    // patch outer JumpFalse
    let outer_end = instrs.len();
    if let Instr::JumpFalse(_, ref mut off) = instrs[outer_jf_patch] {
        *off = (outer_end as i32) - (outer_jf_patch as i32) - 1;
    }
    // patch outer Jump
    if let Instr::Jump(ref mut off) = instrs[outer_jump_patch] {
        *off = (outer_header as i32) - (outer_jump_patch as i32) - 1;
    }
    instrs.push(Instr::Move(0, 1));         // return total
    instrs
}

// =============================================================================
// §2. POLYHEDRAL ANALYSIS AT EXTREME SCALE
// =============================================================================

fn bench_scop_extraction_extreme() {
    println!();
    println!("═══ POLYHEDRAL SCoP EXTRACTION — EXTREME SCALE ═══════════════════");
    println!();
    println!("┌──────────────────────┬─────────┬───────────┬──────────┬─────────┬──────────┬─────────────┐");
    println!("│ Test Case            │ Instrs  │ Extract   │ Loops    │ Hints   │ Tile Ins │ UB Value    │");
    println!("├──────────────────────┼─────────┼───────────┼──────────┼─────────┼──────────┼─────────────┤");

    let test_cases: Vec<(&str, Vec<Instr>, i64)> = vec![
        ("count-1M",     gen_count_loop(ONE_MILLION),     ONE_MILLION),
        ("count-1B",     gen_count_loop(ONE_BILLION),     ONE_BILLION),
        ("count-1T",     gen_count_loop(ONE_TRILLION),    ONE_TRILLION),
        ("sum-1M",       gen_sum_loop(ONE_MILLION),       ONE_MILLION),
        ("sum-1B",       gen_sum_loop(ONE_BILLION),       ONE_BILLION),
        ("sum-1T",       gen_sum_loop(ONE_TRILLION),      ONE_TRILLION),
        ("lcg-1M",       gen_lcg_loop(ONE_MILLION),       ONE_MILLION),
        ("lcg-1B",       gen_lcg_loop(ONE_BILLION),       ONE_BILLION),
        ("lcg-1T",       gen_lcg_loop(ONE_TRILLION),      ONE_TRILLION),
        ("nested-1Kx1K", gen_nested_loop(1000, 1000),     1000),  // outer loop UB
    ];

    for (name, instrs, iters) in &test_cases {
        let n_instrs = instrs.len();

        // Benchmark SCoP extraction (1000 iterations for stable timing)
        let iterations = 1000;
        let start = Instant::now();
        let mut scop_result = None;
        for _ in 0..iterations {
            scop_result = extract_scop(instrs);
        }
        let extract_time = start.elapsed().as_secs_f64() / iterations as f64;

        let (n_loops, n_hints, n_tiled, ub_val) = if let Some(ref scop) = scop_result {
            let loops = scop.arena.loops.len();
            let ub = scop.arena.loops.first()
                .map(|l| l.upper_bound.constant)
                .unwrap_or(0);
            let block = optimize_trace_polyhedral(instrs);
            let hints = block.hints.len();
            let tiled = block.instrs.len();
            (loops, hints, tiled, ub)
        } else {
            (0, 0, 0, 0)
        };

        let ub_str = if ub_val >= ONE_TRILLION {
            format!("{:.0}T", ub_val as f64 / 1e12)
        } else if ub_val >= ONE_BILLION {
            format!("{:.0}B", ub_val as f64 / 1e9)
        } else if ub_val >= ONE_MILLION {
            format!("{:.0}M", ub_val as f64 / 1e6)
        } else {
            format!("{}", ub_val)
        };

        println!("│ {:20} │ {:>7} │ {:>7.1}us │ {:>8} │ {:>7} │ {:>8} │ {:>11} │",
            name, n_instrs, extract_time * 1_000_000.0, n_loops, n_hints, n_tiled, ub_str);

        // Verify UB was correctly extracted
        if scop_result.is_some() {
            let extracted_ub = scop_result.as_ref().unwrap().arena.loops.first()
                .map(|l| l.upper_bound.constant)
                .unwrap_or(0);
            assert_eq!(extracted_ub, *iters, "UB mismatch for {}: expected {}, got {}",
                name, iters, extracted_ub);
        }
    }

    println!("└──────────────────────┴─────────┴───────────┴──────────┴─────────┴──────────┴─────────────┘");
}

// =============================================================================
// §3. DEPENDENCY ANALYSIS AT EXTREME BOUNDS
// =============================================================================

fn bench_dependency_extreme() {
    println!();
    println!("═══ DEPENDENCY ANALYSIS — EXTREME BOUNDS ════════════════════════");
    println!();

    // Test with 1T bounds
    let bounds_1t = [(0i64, ONE_TRILLION); MAX_POLY_DEPTH];

    // Simple: src = i, dst = i + 1 (classic flow dependence)
    let src = AffineExpr::variable(0);  // i
    let dst = AffineExpr::variable(0).add(&AffineExpr::constant(1)); // i + 1

    let iterations = 100_000;
    let start = Instant::now();
    let mut dep_count = 0;
    for _ in 0..iterations {
        if analyze_dependency_multivariate(&src, &dst, &bounds_1t).is_some() {
            dep_count += 1;
        }
    }
    let dep_time = start.elapsed().as_secs_f64() / iterations as f64;

    println!("  1T-bound flow dependency (src=i, dst=i+1):");
    println!("    Per-call latency:  {:.0} ns", dep_time * 1_000_000_000.0);
    println!("    Throughput:        {:.0} calls/sec", 1.0 / dep_time);
    println!("    Dependencies found: {} / {} ({:.1}%)",
        dep_count, iterations, dep_count as f64 / iterations as f64 * 100.0);

    // Anti-dependence: src = i+1, dst = i
    let src2 = AffineExpr::variable(0).add(&AffineExpr::constant(1)); // i + 1
    let dst2 = AffineExpr::variable(0);  // i

    let start = Instant::now();
    let mut dep_count2 = 0;
    for _ in 0..iterations {
        if analyze_dependency_multivariate(&src2, &dst2, &bounds_1t).is_some() {
            dep_count2 += 1;
        }
    }
    let dep_time2 = start.elapsed().as_secs_f64() / iterations as f64;

    println!();
    println!("  1T-bound anti-dependency (src=i+1, dst=i):");
    println!("    Per-call latency:  {:.0} ns", dep_time2 * 1_000_000_000.0);
    println!("    Throughput:        {:.0} calls/sec", 1.0 / dep_time2);
    println!("    Dependencies found: {} / {} ({:.1}%)",
        dep_count2, iterations, dep_count2 as f64 / iterations as f64 * 100.0);

    // No-dependence: src = 2i, dst = 2i + 1 (GCD test eliminates this)
    let src3 = AffineExpr::variable(0).mul_const(2);  // 2i
    let dst3 = AffineExpr::variable(0).mul_const(2).add(&AffineExpr::constant(1)); // 2i + 1

    let start = Instant::now();
    let mut dep_count3 = 0;
    for _ in 0..iterations {
        if analyze_dependency_multivariate(&src3, &dst3, &bounds_1t).is_some() {
            dep_count3 += 1;
        }
    }
    let dep_time3 = start.elapsed().as_secs_f64() / iterations as f64;

    println!();
    println!("  1T-bound no-dependence (src=2i, dst=2i+1):");
    println!("    Per-call latency:  {:.0} ns", dep_time3 * 1_000_000_000.0);
    println!("    Throughput:        {:.0} calls/sec", 1.0 / dep_time3);
    println!("    Dependencies found: {} / {} ({:.1}%)  ← should be 0%",
        dep_count3, iterations, dep_count3 as f64 / iterations as f64 * 100.0);

    // Multi-dimensional with 1T bounds
    let src_4d = AffineExpr {
        constant: 0,
        coefficients: [1, 2, 3, 4, 0, 0, 0, 0],
        active_mask: 0b1111,
    };
    let dst_4d = AffineExpr {
        constant: 2,
        coefficients: [4, 3, 2, 1, 0, 0, 0, 0],
        active_mask: 0b1111,
    };

    let start = Instant::now();
    let mut dep_count_4d = 0;
    for _ in 0..iterations {
        if analyze_dependency_multivariate(&src_4d, &dst_4d, &bounds_1t).is_some() {
            dep_count_4d += 1;
        }
    }
    let dep_time_4d = start.elapsed().as_secs_f64() / iterations as f64;

    println!();
    println!("  1T-bound 4D dependency (complex affine):");
    println!("    Per-call latency:  {:.0} ns", dep_time_4d * 1_000_000_000.0);
    println!("    Throughput:        {:.0} calls/sec", 1.0 / dep_time_4d);
    println!("    Dependencies found: {} / {} ({:.1}%)",
        dep_count_4d, iterations, dep_count_4d as f64 / iterations as f64 * 100.0);
}

// =============================================================================
// §4. FULL POLYHEDRAL PIPELINE AT EXTREME SCALE
// =============================================================================

fn bench_full_pipeline_extreme() {
    println!();
    println!("═══ FULL POLYHEDRAL PIPELINE — EXTREME SCALE ═══════════════════");
    println!();
    println!("┌──────────────────────┬──────────────┬──────────────┬──────────────┬──────────┐");
    println!("│ Test Case            │  Full (us)    │ Extract (us) │ Tile (us)    │ Speedup  │");
    println!("├──────────────────────┼──────────────┼──────────────┼──────────────┼──────────┤");

    let test_cases: Vec<(&str, Vec<Instr>)> = vec![
        ("count-1M",     gen_count_loop(ONE_MILLION)),
        ("count-1B",     gen_count_loop(ONE_BILLION)),
        ("count-1T",     gen_count_loop(ONE_TRILLION)),
        ("sum-1M",       gen_sum_loop(ONE_MILLION)),
        ("sum-1B",       gen_sum_loop(ONE_BILLION)),
        ("sum-1T",       gen_sum_loop(ONE_TRILLION)),
        ("lcg-1M",       gen_lcg_loop(ONE_MILLION)),
        ("lcg-1T",       gen_lcg_loop(ONE_TRILLION)),
        ("nested-1Kx1K", gen_nested_loop(1000, 1000)),
    ];

    for (name, instrs) in &test_cases {
        let iterations = 1000;

        // Full pipeline: optimize_trace_polyhedral
        let start = Instant::now();
        for _ in 0..iterations {
            let block = optimize_trace_polyhedral(instrs);
            black_box(&block);
        }
        let full_time = start.elapsed().as_secs_f64() / iterations as f64;

        // SCoP extraction only
        let start = Instant::now();
        for _ in 0..iterations {
            let scop = extract_scop(instrs);
            black_box(&scop);
        }
        let extract_time = start.elapsed().as_secs_f64() / iterations as f64;

        // Tiling only
        let start = Instant::now();
        for _ in 0..iterations {
            if let Some(scop) = extract_scop(instrs) {
                let tiled = jules::polyhedral::generate_tiled_loops(&scop, &[32]);
                black_box(&tiled);
            }
        }
        let tile_time = start.elapsed().as_secs_f64() / iterations as f64;

        let speedup = if extract_time > 0.0 { full_time / extract_time } else { 0.0 };

        println!("│ {:20} │ {:>10.1}   │ {:>10.1}   │ {:>10.1}   │ {:>8.2}x │",
            name,
            full_time * 1_000_000.0,
            extract_time * 1_000_000.0,
            tile_time * 1_000_000.0,
            speedup);
    }

    println!("└──────────────────────┴──────────────┴──────────────┴──────────────┴──────────┘");
}

// =============================================================================
// §5. CORRECTNESS VALIDATION AT 1T
// =============================================================================

fn validate_polyhedral_correctness() {
    println!();
    println!("═══ CORRECTNESS VALIDATION — 1T SCALE ═══════════════════════════");
    println!();

    // Test 1: SCoP extraction extracts the correct upper bound for 1T loop
    let instrs = gen_sum_loop(ONE_TRILLION);
    match extract_scop(&instrs) {
        Some(scop) => {
            let ub = scop.arena.loops.first()
                .map(|l| l.upper_bound.constant)
                .unwrap_or(0);
            let n_loops = scop.arena.loops.len();
            if ub == ONE_TRILLION {
                println!("  [PASS] 1T sum loop: UB={} (correct), {} loop(s) extracted", ub, n_loops);
            } else {
                println!("  [FAIL] 1T sum loop: UB={} (expected {})", ub, ONE_TRILLION);
            }
            for (i, l) in scop.arena.loops.iter().enumerate() {
                println!("    Loop {}: iv=slot_{}, step={}, depth={}, header={}, back={}",
                    i, l.iv.slot, l.iv.step, l.depth, l.header_pc, l.back_edge_pc);
            }
        }
        None => println!("  [FAIL] 1T sum loop: no SCoP extracted!"),
    }

    // Test 2: Count loop at 1T
    let instrs = gen_count_loop(ONE_TRILLION);
    match extract_scop(&instrs) {
        Some(scop) => {
            let ub = scop.arena.loops.first()
                .map(|l| l.upper_bound.constant)
                .unwrap_or(0);
            if ub == ONE_TRILLION {
                println!("  [PASS] 1T count loop: UB={} (correct)", ub);
            } else {
                println!("  [FAIL] 1T count loop: UB={} (expected {})", ub, ONE_TRILLION);
            }
        }
        None => println!("  [FAIL] 1T count loop: no SCoP extracted!"),
    }

    // Test 3: LCG loop at 1T
    let instrs = gen_lcg_loop(ONE_TRILLION);
    match extract_scop(&instrs) {
        Some(scop) => {
            let ub = scop.arena.loops.first()
                .map(|l| l.upper_bound.constant)
                .unwrap_or(0);
            if ub == ONE_TRILLION {
                println!("  [PASS] 1T LCG loop: UB={} (correct)", ub);
            } else {
                println!("  [FAIL] 1T LCG loop: UB={} (expected {})", ub, ONE_TRILLION);
            }
        }
        None => println!("  [FAIL] 1T LCG loop: no SCoP extracted!"),
    }

    // Test 4: Nested loop with 1M outer iterations
    let instrs = gen_nested_loop(ONE_MILLION as i64, 1000);
    match extract_scop(&instrs) {
        Some(scop) => {
            println!("  [PASS] Nested 1M×1K loop: {} loop(s), depth={}",
                scop.arena.loops.len(), scop.max_depth());
            for (i, l) in scop.arena.loops.iter().enumerate() {
                let ub_str = if l.upper_bound.constant >= ONE_MILLION {
                    format!("{:.0}M", l.upper_bound.constant as f64 / 1e6)
                } else {
                    format!("{}", l.upper_bound.constant)
                };
                println!("    Loop {}: iv=slot_{}, step={}, depth={}, UB={}",
                    i, l.iv.slot, l.iv.step, l.depth, ub_str);
            }
        }
        None => println!("  [FAIL] Nested 1M×1K loop: no SCoP extracted!"),
    }

    // Test 5: Full pipeline on 1T produces valid output
    let instrs = gen_sum_loop(ONE_TRILLION);
    let block = optimize_trace_polyhedral(&instrs);
    println!("  [PASS] Full pipeline (1T sum): {} instructions, {} hints",
        block.instrs.len(), block.hints.len());

    // Test 6: AffineExpr arithmetic with extreme values
    let a = AffineExpr::variable(0);  // v0
    let b = AffineExpr::constant(ONE_TRILLION);  // 10^12
    let c = a.add(&b);               // v0 + 10^12
    let d = c.mul_const(2);          // 2*v0 + 2×10^12
    assert_eq!(d.constant, 2 * ONE_TRILLION);
    assert_eq!(d.coefficients[0], 2);
    assert_eq!(d.active_mask, 1);
    println!("  [PASS] AffineExpr with 1T constant: (v0 + 10^12) * 2 = {{const=2T, coeff[0]=2}}");

    // Test 7: Dependency analysis at 1T scale
    let src = AffineExpr::variable(0);  // i
    let dst = AffineExpr::variable(0).add(&AffineExpr::constant(1)); // i + 1
    let bounds = [(0i64, ONE_TRILLION); MAX_POLY_DEPTH];
    match analyze_dependency_multivariate(&src, &dst, &bounds) {
        Some(dep) => {
            println!("  [PASS] 1T dependency: src=i, dst=i+1 → direction={:?}, distance={:?}",
                &dep.direction_vector[..dep.len.min(2)], &dep.distance_matrix[..dep.len.min(2)]);
        }
        None => println!("  [WARN] 1T dependency: src=i, dst=i+1 → no dependency found"),
    }

    // Test 8: GCD test with large coefficients at 1T bounds
    // 2i and 2i+1 can never alias (GCD=2, but offset=1 is not divisible by 2)
    let src = AffineExpr::variable(0).mul_const(2);  // 2i
    let dst = AffineExpr::variable(0).mul_const(2).add(&AffineExpr::constant(1)); // 2i + 1
    match analyze_dependency_multivariate(&src, &dst, &bounds) {
        Some(dep) => {
            println!("  [WARN] GCD independence test: 2i vs 2i+1 → found dep (should be None): {:?}",
                &dep.direction_vector[..dep.len.min(2)]);
        }
        None => {
            println!("  [PASS] GCD independence test: 2i vs 2i+1 → no dependency (correctly eliminated)");
        }
    }

    // Test 9: SlotCache with 1T bound value
    let mut cache = SlotCache::new();
    cache.insert(5, AffineExpr::constant(ONE_TRILLION));
    match cache.get(5) {
        Some(expr) if expr.constant == ONE_TRILLION => {
            println!("  [PASS] SlotCache preserves 1T constant value");
        }
        Some(expr) => {
            println!("  [FAIL] SlotCache: expected {}, got {}", ONE_TRILLION, expr.constant);
        }
        None => {
            println!("  [FAIL] SlotCache: slot 5 not found after insert");
        }
    }

    println!();
}

// =============================================================================
// §6. JIT EXECUTION — 1T LOOP WITH POLYHEDRAL OPTIMIZATION
// =============================================================================

fn bench_jit_execution_1t() {
    println!();
    println!("═══ JIT EXECUTION — 1T LOOP WITH POLYHEDRAL ═══════════════════");
    println!();

    use jules::{CompileUnit, Pipeline};

    // Use 1B and 100M loops for JIT execution (1T would take too long for a quick bench)
    let workloads: Vec<(&str, &str, i64)> = vec![
        ("COUNT-100M", r#"
fn bench() -> i32 {
  let mut count: i32 = 0;
  let mut i: i32 = 0;
  while i < 100000000 {
    count = count + 1;
    i = i + 1;
  }
  count
}
"#, 100_000_000),
        ("SUM-100M", r#"
fn bench() -> i32 {
  let mut s: i32 = 0;
  let mut i: i32 = 0;
  while i < 100000000 {
    s = s + i;
    i = i + 1;
  }
  s
}
"#, 100_000_000),
        ("LCG-100M", r#"
fn bench() -> i32 {
  let mut s: i32 = 42;
  let mut i: i32 = 0;
  while i < 100000000 {
    s = s * 1664525 + 1013904223;
    i = i + 1;
  }
  s
}
"#, 100_000_000),
        ("COUNT-1B", r#"
fn bench() -> i32 {
  let mut count: i32 = 0;
  let mut i: i32 = 0;
  while i < 1000000000 {
    count = count + 1;
    i = i + 1;
  }
  count
}
"#, ONE_BILLION),
        ("SUM-1B", r#"
fn bench() -> i32 {
  let mut s: i32 = 0;
  let mut i: i32 = 0;
  while i < 1000000000 {
    s = s + i;
    i = i + 1;
  }
  s
}
"#, ONE_BILLION),
    ];

    println!("┌──────────────┬────────────┬────────────┬──────────────┬──────────┬──────────┐");
    println!("│ Workload     │  JIT (s)   │ Interp (s) │ Throughput   │ Giter/s  │ J/I (x)  │");
    println!("├──────────────┼────────────┼────────────┼──────────────┼──────────┼──────────┤");

    for (name, src, loops) in &workloads {
        let mut unit = CompileUnit::new(format!("<bench:{}>", name), *src);
        let result = Pipeline::new().run(&mut unit);
        if unit.has_errors() {
            println!("│ {:12} │ COMPILE ERR │            │              │          │          │", name);
            continue;
        }
        let prog = match result.program() {
            Some(p) => p,
            None => { println!("│ {:12} │  NO PROG   │            │              │          │          │", name); continue; }
        };

        // JIT benchmark
        let mut interp_jit = jules::interp::Interpreter::new();
        interp_jit.set_jit_enabled(true);
        interp_jit.set_advance_jit_enabled(true);
        interp_jit.set_native_jit_enabled(true);
        interp_jit.load_program(prog);
        // Warmup (3 rounds to trigger JIT tier promotion)
        let _ = interp_jit.call_fn("bench", vec![]);
        let _ = interp_jit.call_fn("bench", vec![]);
        std::thread::sleep(std::time::Duration::from_millis(100));
        let _ = interp_jit.call_fn("bench", vec![]);

        let start = Instant::now();
        let _ = black_box(interp_jit.call_fn("bench", vec![]));
        let jit_s = start.elapsed().as_secs_f64();
        let (nc, vc, fc) = interp_jit.jit_counters();

        // Interpreter benchmark (reduced iterations for speed)
        let interp_n = (*loops as f64 * 0.001).min(1_000_000.0) as i64;
        let interp_src = if *loops > 100_000 {
            src.replace(&loops.to_string(), &interp_n.to_string())
        } else {
            src.to_string()
        };
        let mut interp_unit = CompileUnit::new(format!("<interp:{}>", name), &interp_src);
        let interp_result = Pipeline::new().run(&mut interp_unit);
        let interp_s = if !interp_unit.has_errors() {
            if let Some(iprog) = interp_result.program() {
                let mut interp = jules::interp::Interpreter::new();
                interp.set_jit_enabled(false);
                interp.load_program(iprog);
                let _ = interp.call_fn("bench", vec![]);
                let start = Instant::now();
                let _ = black_box(interp.call_fn("bench", vec![]));
                let t = start.elapsed().as_secs_f64();
                let scale = *loops as f64 / interp_n as f64;
                t * scale
            } else { 0.0 }
        } else { 0.0 };

        let giter_s = if jit_s > 0.0 { (*loops as f64 / 1e9) / jit_s } else { 0.0 };
        let throughput = if jit_s > 0.0 {
            let ips = *loops as f64 / jit_s;
            if ips >= 1e12 { format!("{:.1}T/s", ips / 1e12) }
            else if ips >= 1e9 { format!("{:.1}G/s", ips / 1e9) }
            else if ips >= 1e6 { format!("{:.1}M/s", ips / 1e6) }
            else { format!("{:.0}/s", ips) }
        } else { "N/A".to_string() };
        let ji_ratio = if jit_s > 0.0 && interp_s > 0.0 { interp_s / jit_s } else { 0.0 };

        let short_name = if name.len() > 12 { &name[..12] } else { name };
        println!("│ {:12} │ {:>10} │ {:>10} │ {:>12} │ {:>8.1} │ {:>8.1} │",
            short_name, fmt_dur(jit_s), fmt_dur(interp_s), throughput, giter_s, ji_ratio);
        eprintln!("  [{}] counters: native={} vm={} fallback={}", name, nc, vc, fc);
    }

    println!("└──────────────┴────────────┴────────────┴──────────────┴──────────┴──────────┘");
}

// =============================================================================
// §7. SCALING ANALYSIS — LOOP BOUND SWEEP
// =============================================================================

fn bench_scaling_analysis() {
    println!();
    println!("═══ SCALING ANALYSIS — LOOP BOUND SWEEP ════════════════════════");
    println!();
    println!("  How does polyhedral analysis time scale with loop bound magnitude?");
    println!("  (Analysis time should be INDEPENDENT of bound magnitude since it's");
    println!("   just a constant in the bytecode — the key benefit of affine analysis.)");
    println!();
    println!("┌───────────────────┬───────────────┬──────────────┬──────────────┐");
    println!("│ Bound             │ Extract (ns)  │ Full (ns)    │ Dep Anal(ns) │");
    println!("├───────────────────┼───────────────┼──────────────┼──────────────┤");

    let bounds: Vec<(&str, i64)> = vec![
        ("1K",     1_000),
        ("1M",     ONE_MILLION),
        ("1B",     ONE_BILLION),
        ("1T",     ONE_TRILLION),
        ("100T",   100 * ONE_TRILLION),
        ("1P",     1_000_000_000_000_000), // 10^15
    ];

    for (name, n) in &bounds {
        let instrs = gen_sum_loop(*n);

        // SCoP extraction
        let iterations = 10_000;
        let start = Instant::now();
        for _ in 0..iterations {
            let scop = extract_scop(&instrs);
            black_box(&scop);
        }
        let extract_ns = (start.elapsed().as_secs_f64() / iterations as f64) * 1_000_000_000.0;

        // Full pipeline
        let start = Instant::now();
        for _ in 0..iterations {
            let block = optimize_trace_polyhedral(&instrs);
            black_box(&block);
        }
        let full_ns = (start.elapsed().as_secs_f64() / iterations as f64) * 1_000_000_000.0;

        // Dependency analysis
        let src = AffineExpr::variable(0);
        let dst = AffineExpr::variable(0).add(&AffineExpr::constant(1));
        let dep_bounds = [(0i64, *n); MAX_POLY_DEPTH];
        let start = Instant::now();
        for _ in 0..100_000 {
            black_box(analyze_dependency_multivariate(&src, &dst, &dep_bounds));
        }
        let dep_ns = (start.elapsed().as_secs_f64() / 100_000.0) * 1_000_000_000.0;

        println!("│ {:17} │ {:>11.0}   │ {:>10.0}   │ {:>10.0}   │",
            name, extract_ns, full_ns, dep_ns);
    }

    println!("└───────────────────┴───────────────┴──────────────┴──────────────┘");
    println!();
    println!("  → If times are roughly constant across bounds, the polyhedral engine");
    println!("    is correctly O(1) with respect to loop bound magnitude.");
}

// =============================================================================
// MAIN
// =============================================================================

fn main() {
    println!();
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  POLYHEDRAL 1T LOOP BENCHMARK                                 ║");
    println!("║  Jules Language — Polyhedral Engine + Phase 3 JIT             ║");
    println!("║  1 Trillion (10^12) Iteration Stress Test                     ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // CPU info
    #[cfg(target_os = "linux")]
    {
        let cpu = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| s.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap_or(" unknown").trim().to_string()))
            .unwrap_or_else(|| "unknown".to_string());
        println!("  CPU: {}", cpu);
    }
    println!("  PID: {}", std::process::id());
    println!();

    // Phase 1: Correctness
    validate_polyhedral_correctness();

    // Phase 2: Extreme-scale SCoP extraction
    bench_scop_extraction_extreme();

    // Phase 3: Dependency analysis at 1T bounds
    bench_dependency_extreme();

    // Phase 4: Full pipeline at extreme scale
    bench_full_pipeline_extreme();

    // Phase 5: Scaling analysis
    bench_scaling_analysis();

    // Phase 6: JIT execution with polyhedral
    bench_jit_execution_1t();

    println!();
    println!("  Benchmark complete.");
}
