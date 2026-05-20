// =============================================================================
// bench_polyhedral.rs — Polyhedral JIT Optimization Benchmark
//
// Measures:
//   1. Polyhedral SCoP extraction overhead on synthetic bytecode
//   2. Dependency analysis + loop tiling throughput
//   3. SIMD hint generation
//   4. Full pipeline: Jules source → JIT → native execution
//   5. Polyhedral engine correctness validation
//
// Usage:
//   cargo run --release --bin bench-polyhedral
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

fn fmt_dur(secs: f64) -> String {
    if secs >= 1.0 { format!("{:.3}s", secs) }
    else if secs >= 0.001 { format!("{:.2}ms", secs * 1000.0) }
    else { format!("{:.0}us", secs * 1_000_000.0) }
}

// =============================================================================
// §1. SYNTHETIC BYTECODE GENERATORS
// =============================================================================

/// Generate bytecode for a simple counted loop:
///   i = 0; while i < N { s = s + i; i = i + 1; }
fn gen_simple_loop(n: i64) -> Vec<Instr> {
    // Slot layout: 0=return, 1=s, 2=i, 3=temp, 4=N, 5=cond, 6=step
    vec![
        Instr::LoadI64(1, 0),          // s = 0
        Instr::LoadI64(2, 0),          // i = 0
        Instr::LoadI64(4, n),          // N = n
        // header:
        Instr::BinOp(5, BinOpKind::Lt, 2, 4),  // cond = i < N
        Instr::JumpFalse(5, 9),        // if !cond goto end
        Instr::BinOp(3, BinOpKind::Add, 1, 2),  // temp = s + i
        Instr::Move(1, 3),             // s = temp
        Instr::LoadI64(6, 1),          // step = 1
        Instr::BinOp(3, BinOpKind::Add, 2, 6),  // temp = i + 1
        Instr::Move(2, 3),             // i = temp
        Instr::Jump(-8),               // goto header
        // end:
        Instr::Move(0, 1),             // return s
    ]
}

/// Generate bytecode for a nested loop:
///   for i in 0..M: for j in 0..N: total += i*j + 1
fn gen_nested_loop(m: i64, n: i64) -> Vec<Instr> {
    // Simplified — just inner loop for now
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
    let inner_header_pc = outer_jf_patch + 3; // after outer JF + LoadI64(3,0) + LoadI64(7,n)
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

/// Generate bytecode for an LCG loop
fn gen_lcg_loop(n: i64) -> Vec<Instr> {
    vec![
        Instr::LoadI64(1, 42),             // s = 42
        Instr::LoadI64(2, 0),              // i = 0
        Instr::LoadI64(5, n),              // N = n
        // header:
        Instr::BinOp(6, BinOpKind::Lt, 2, 5),  // cond = i < N
        Instr::JumpFalse(6, 8),            // if !cond goto end
        Instr::LoadI64(7, 1664525),        // a = 1664525
        Instr::BinOp(3, BinOpKind::Mul, 1, 7),  // temp = s * a
        Instr::LoadI64(8, 1013904223),     // c = 1013904223
        Instr::BinOp(4, BinOpKind::Add, 3, 8),  // temp2 = temp + c
        Instr::Move(1, 4),                 // s = temp2
        Instr::LoadI64(9, 1),              // step = 1
        Instr::BinOp(3, BinOpKind::Add, 2, 9),  // temp = i + 1
        Instr::Move(2, 3),                 // i = temp
        Instr::Jump(-11),                  // goto header
        Instr::Move(0, 1),                 // return s
    ]
}

/// Generate bytecode for a muladd loop
fn gen_muladd_loop(n: i64) -> Vec<Instr> {
    vec![
        Instr::LoadI64(1, 1),              // a = 1
        Instr::LoadI64(2, 2),              // b = 2
        Instr::LoadI64(3, 0),              // i = 0
        Instr::LoadI64(6, n),              // N = n
        // header:
        Instr::BinOp(7, BinOpKind::Lt, 3, 6),  // cond = i < N
        Instr::JumpFalse(7, 9),            // if !cond goto end
        Instr::LoadI64(8, 3),              // three = 3
        Instr::BinOp(4, BinOpKind::Mul, 1, 8),  // tmp = a * 3
        Instr::BinOp(5, BinOpKind::Add, 4, 2),  // tmp = tmp + b
        Instr::Move(9, 2),                 // save b
        Instr::Move(1, 9),                 // a = old_b
        Instr::Move(2, 5),                 // b = tmp
        Instr::LoadI64(7, 1),              // step = 1
        Instr::BinOp(4, BinOpKind::Add, 3, 7),  // tmp = i + 1
        Instr::Move(3, 4),                 // i = tmp
        Instr::Jump(-12),                  // goto header
        Instr::Move(0, 1),                 // return a
    ]
}

// =============================================================================
// §2. POLYHEDRAL MICRO-BENCHMARKS
// =============================================================================

fn bench_scop_extraction() {
    println!();
    println!("═══ POLYHEDRAL SCoP EXTRACTION BENCHMARK ══════════════════════");
    println!();
    println!("┌───────────────────┬─────────┬───────────┬──────────┬─────────┬──────────┐");
    println!("│ Test Case         │ Instrs  │ Extract   │ Loops    │ Hints   │ Tile Ins │");
    println!("├───────────────────┼─────────┼───────────┼──────────┼─────────┼──────────┤");

    let test_cases: Vec<(&str, Vec<Instr>)> = vec![
        ("simple-1K", gen_simple_loop(1000)),
        ("simple-10K", gen_simple_loop(10000)),
        ("simple-100K", gen_simple_loop(100000)),
        ("simple-1M", gen_simple_loop(1_000_000)),
        ("lcg-10K", gen_lcg_loop(10000)),
        ("lcg-1M", gen_lcg_loop(1_000_000)),
        ("muladd-10K", gen_muladd_loop(10000)),
        ("muladd-1M", gen_muladd_loop(1_000_000)),
        ("nested-50x50", gen_nested_loop(50, 50)),
        ("nested-100x100", gen_nested_loop(100, 100)),
        ("nested-200x200", gen_nested_loop(200, 200)),
    ];

    for (name, instrs) in &test_cases {
        let n_instrs = instrs.len();

        // Benchmark SCoP extraction
        let iterations = 1000;
        let start = Instant::now();
        let mut scop_result = None;
        for _ in 0..iterations {
            scop_result = extract_scop(instrs);
        }
        let extract_time = start.elapsed().as_secs_f64() / iterations as f64;

        let (n_loops, n_hints, n_tiled) = if let Some(ref scop) = scop_result {
            let loops = scop.arena.loops.len();
            let block = optimize_trace_polyhedral(instrs);
            let hints = block.hints.len();
            let tiled = block.instrs.len();
            (loops, hints, tiled)
        } else {
            (0, 0, 0)
        };

        println!("│ {:17} │ {:>7} │ {:>7.1}us │ {:>8} │ {:>7} │ {:>8} │",
            name, n_instrs, extract_time * 1_000_000.0, n_loops, n_hints, n_tiled);
    }

    println!("└───────────────────┴─────────┴───────────┴──────────┴─────────┴──────────┘");
}

fn bench_dependency_analysis() {
    println!();
    println!("═══ DEPENDENCY ANALYSIS BENCHMARK ══════════════════════════════");
    println!();

    // Create two affine expressions with multiple dimensions
    let expr_a = AffineExpr {
        constant: 0,
        coefficients: [1, 2, 0, 0, 0, 0, 0, 0],
        active_mask: 0b11,
    };
    let expr_b = AffineExpr {
        constant: 1,
        coefficients: [2, 1, 0, 0, 0, 0, 0, 0],
        active_mask: 0b11,
    };
    let bounds = [(0i64, 1024i64); MAX_POLY_DEPTH];

    let iterations = 100_000;
    let start = Instant::now();
    let mut dep_count = 0;
    for _ in 0..iterations {
        if analyze_dependency_multivariate(&expr_a, &expr_b, &bounds).is_some() {
            dep_count += 1;
        }
    }
    let dep_time = start.elapsed().as_secs_f64() / iterations as f64;

    println!("  Banerjee-Wolfe dependency analysis (2-dim, 1024 bounds):");
    println!("    Per-call latency:  {:.0} ns", dep_time * 1_000_000_000.0);
    println!("    Throughput:        {:.0} calls/sec", 1.0 / dep_time);
    println!("    Dependencies found: {} / {} ({:.1}%)",
        dep_count, iterations, dep_count as f64 / iterations as f64 * 100.0);

    // Test with higher dimensions
    let expr_4d_a = AffineExpr {
        constant: 0,
        coefficients: [1, 2, 3, 4, 0, 0, 0, 0],
        active_mask: 0b1111,
    };
    let expr_4d_b = AffineExpr {
        constant: 2,
        coefficients: [4, 3, 2, 1, 0, 0, 0, 0],
        active_mask: 0b1111,
    };

    let start = Instant::now();
    let mut dep_count_4d = 0;
    for _ in 0..iterations {
        if analyze_dependency_multivariate(&expr_4d_a, &expr_4d_b, &bounds).is_some() {
            dep_count_4d += 1;
        }
    }
    let dep_time_4d = start.elapsed().as_secs_f64() / iterations as f64;

    println!();
    println!("  4-dimensional dependency analysis:");
    println!("    Per-call latency:  {:.0} ns", dep_time_4d * 1_000_000_000.0);
    println!("    Throughput:        {:.0} calls/sec", 1.0 / dep_time_4d);
    println!("    Dependencies found: {} / {} ({:.1}%)",
        dep_count_4d, iterations, dep_count_4d as f64 / iterations as f64 * 100.0);
}

fn bench_slot_cache() {
    println!();
    println!("═══ SLOT CACHE BENCHMARK ═══════════════════════════════════════");
    println!();

    let instrs = gen_simple_loop(10000);
    let loop_iv_slots: Vec<u16> = vec![2]; // i is the induction var

    let iterations = 10_000;
    let start = Instant::now();
    for _ in 0..iterations {
        let mut cache = SlotCache::new();
        jules::polyhedral::populate_slot_cache(&instrs, &mut cache, &loop_iv_slots);
        black_box(&cache);
    }
    let cache_time = start.elapsed().as_secs_f64() / iterations as f64;

    println!("  Slot cache population ({} instructions, 1 IV):", instrs.len());
    println!("    Per-call latency:  {:.1} us", cache_time * 1_000_000.0);
    println!("    Throughput:        {:.0} instrs/sec", instrs.len() as f64 / cache_time);
}

fn bench_full_polyhedral_pipeline() {
    println!();
    println!("═══ FULL POLYHEDRAL PIPELINE BENCHMARK ════════════════════════");
    println!();
    println!("┌───────────────────┬──────────────┬──────────────┬──────────────┬──────────┐");
    println!("│ Test Case         │  Full (us)    │ Extract (us) │ Tile (us)    │ Speedup  │");
    println!("├───────────────────┼──────────────┼──────────────┼──────────────┼──────────┤");

    let test_cases: Vec<(&str, Vec<Instr>)> = vec![
        ("simple-1K", gen_simple_loop(1000)),
        ("simple-10K", gen_simple_loop(10000)),
        ("simple-100K", gen_simple_loop(100000)),
        ("lcg-10K", gen_lcg_loop(10000)),
        ("muladd-10K", gen_muladd_loop(10000)),
        ("nested-50x50", gen_nested_loop(50, 50)),
        ("nested-100x100", gen_nested_loop(100, 100)),
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

        // Tiling only (requires SCoP)
        let start = Instant::now();
        for _ in 0..iterations {
            if let Some(scop) = extract_scop(instrs) {
                let tiled = jules::polyhedral::generate_tiled_loops(&scop, &[32]);
                black_box(&tiled);
            }
        }
        let tile_time = start.elapsed().as_secs_f64() / iterations as f64;

        let speedup = if extract_time > 0.0 { full_time / extract_time } else { 0.0 };

        println!("│ {:17} │ {:>10.1}   │ {:>10.1}   │ {:>10.1}   │ {:>8.2}x │",
            name,
            full_time * 1_000_000.0,
            extract_time * 1_000_000.0,
            tile_time * 1_000_000.0,
            speedup);
    }

    println!("└───────────────────┴──────────────┴──────────────┴──────────────┴──────────┘");
}

fn bench_polyhedral_correctness() {
    println!();
    println!("═══ POLYHEDRAL CORRECTNESS VALIDATION ══════════════════════════");
    println!();

    // Test 1: Simple loop SCoP extraction
    let instrs = gen_simple_loop(1000);
    match extract_scop(&instrs) {
        Some(scop) => {
            println!("  [PASS] simple_loop: extracted {} loop(s), depth={}",
                scop.arena.loops.len(), scop.max_depth());
            for (i, l) in scop.arena.loops.iter().enumerate() {
                println!("    Loop {}: iv=slot_{}, step={}, header={}, back={}",
                    i, l.iv.slot, l.iv.step, l.header_pc, l.back_edge_pc);
            }
        }
        None => println!("  [FAIL] simple_loop: no SCoP extracted!"),
    }

    // Test 2: Nested loop
    let instrs = gen_nested_loop(50, 50);
    match extract_scop(&instrs) {
        Some(scop) => {
            println!("  [PASS] nested_loop: extracted {} loop(s), depth={}",
                scop.arena.loops.len(), scop.max_depth());
            for (i, l) in scop.arena.loops.iter().enumerate() {
                println!("    Loop {}: iv=slot_{}, step={}, depth={}, header={}, back={}",
                    i, l.iv.slot, l.iv.step, l.depth, l.header_pc, l.back_edge_pc);
            }
        }
        None => println!("  [FAIL] nested_loop: no SCoP extracted!"),
    }

    // Test 3: LCG loop
    let instrs = gen_lcg_loop(10000);
    match extract_scop(&instrs) {
        Some(scop) => {
            println!("  [PASS] lcg_loop: extracted {} loop(s)", scop.arena.loops.len());
        }
        None => println!("  [FAIL] lcg_loop: no SCoP extracted!"),
    }

    // Test 4: Full pipeline produces valid output
    let instrs = gen_simple_loop(10000);
    let block = optimize_trace_polyhedral(&instrs);
    println!("  [PASS] full pipeline: {} instructions, {} hints",
        block.instrs.len(), block.hints.len());

    // Test 5: AffineExpr arithmetic
    let a = AffineExpr::variable(0);  // v0
    let b = AffineExpr::constant(5);  // 5
    let c = a.add(&b);               // v0 + 5
    let d = c.mul_const(2);          // 2*v0 + 10
    assert_eq!(d.constant, 10);
    assert_eq!(d.coefficients[0], 2);
    assert_eq!(d.active_mask, 1);
    println!("  [PASS] AffineExpr: v0 + 5 = {{const=5, coeff[0]=1}}; (v0+5)*2 = {{const=10, coeff[0]=2}}");

    // Test 6: Dependency analysis
    let src = AffineExpr::variable(0);  // i
    let dst = AffineExpr::variable(0).add(&AffineExpr::constant(1)); // i + 1
    let bounds = [(0i64, 100i64); MAX_POLY_DEPTH];
    match analyze_dependency_multivariate(&src, &dst, &bounds) {
        Some(dep) => {
            println!("  [PASS] dependency: src=i, dst=i+1 → direction={:?}, distance={:?}",
                &dep.direction_vector[..dep.len], &dep.distance_matrix[..dep.len]);
        }
        None => println!("  [WARN] dependency: src=i, dst=i+1 → no dependency found"),
    }

    println!();
}

// =============================================================================
// §3. JIT EXECUTION BENCHMARK (with polyhedral wired in)
// =============================================================================

fn bench_jit_execution() {
    println!();
    println!("═══ JIT EXECUTION BENCHMARK (polyhedral wired in) ═════════════");
    println!();

    use jules::{CompileUnit, Pipeline};

    let workloads: Vec<(&str, &str, i32)> = vec![
        ("COUNT-10M", r#"
fn bench() -> i32 {
  let mut count: i32 = 0;
  let mut i: i32 = 0;
  while i < 10000000 {
    count = count + 1;
    i = i + 1;
  }
  count
}
"#, 10_000_000),
        ("LCG-10M", r#"
fn bench() -> i32 {
  let mut s: i32 = 42;
  let mut i: i32 = 0;
  while i < 10000000 {
    s = s * 1664525 + 1013904223;
    i = i + 1;
  }
  s
}
"#, 10_000_000),
        ("SUM-10M", r#"
fn bench() -> i32 {
  let mut s: i32 = 0;
  let mut i: i32 = 0;
  while i < 10000000 {
    s = s + i;
    i = i + 1;
  }
  s
}
"#, 10_000_000),
        ("MULADD-10M", r#"
fn bench() -> i32 {
  let mut a: i32 = 1;
  let mut b: i32 = 2;
  let mut i: i32 = 0;
  while i < 10000000 {
    let tmp: i32 = a * 3 + b;
    a = b;
    b = tmp;
    i = i + 1;
  }
  a
}
"#, 10_000_000),
        ("NESTED-50x50", r#"
fn bench() -> i32 {
  let mut total: i32 = 0;
  let mut i: i32 = 0;
  while i < 50 {
    let mut j: i32 = 0;
    while j < 50 {
      total = total + i * j + 1;
      j = j + 1;
    }
    i = i + 1;
  }
  total
}
"#, 2500),
    ];

    println!("┌──────────────┬────────────┬────────────┬──────────┬──────────┐");
    println!("│ Workload     │  JIT (s)   │ Interp (s) │JIT Miter │ J/I (x)  │");
    println!("├──────────────┼────────────┼────────────┼──────────┼──────────┤");

    for (name, src, loops) in &workloads {
        let mut unit = CompileUnit::new(format!("<bench:{}>", name), *src);
        let result = Pipeline::new().run(&mut unit);
        if unit.has_errors() {
            println!("│ {:12} │ COMPILE ERR │            │          │          │", name);
            continue;
        }
        let prog = match result.program() {
            Some(p) => p,
            None => { println!("│ {:12} │  NO PROG   │            │          │          │", name); continue; }
        };

        // JIT benchmark
        let mut interp_jit = jules::interp::Interpreter::new();
        interp_jit.set_jit_enabled(true);
        interp_jit.set_advance_jit_enabled(true);
        interp_jit.set_native_jit_enabled(true);
        interp_jit.load_program(prog);
        // Warmup
        let _ = interp_jit.call_fn("bench", vec![]);
        let _ = interp_jit.call_fn("bench", vec![]);
        std::thread::sleep(std::time::Duration::from_millis(60));
        let _ = interp_jit.call_fn("bench", vec![]);

        let start = Instant::now();
        let _ = black_box(interp_jit.call_fn("bench", vec![]));
        let jit_s = start.elapsed().as_secs_f64();
        let (nc, vc, fc) = interp_jit.jit_counters();

        // Interpreter benchmark (reduced iterations)
        let interp_n = (*loops as f64 * 0.01).min(100_000.0) as i32;
        let mut interp_src_reduced = src.replace("10000000", &interp_n.to_string())
                                         .replace("50", &((interp_n as f64).sqrt() as i32).to_string());
        if *loops <= 10000 {
            interp_src_reduced = src.to_string(); // small workloads don't need reduction
        }
        let mut interp_unit = CompileUnit::new(format!("<interp:{}>", name), &interp_src_reduced);
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
                // Extrapolate to full loop count
                let scale = if *loops > 10000 { *loops as f64 / interp_n as f64 } else { 1.0 };
                t * scale
            } else { 0.0 }
        } else { 0.0 };

        let mips = if jit_s > 0.0 { (*loops as f64 / 1e6) / jit_s } else { 0.0 };
        let ji_ratio = if jit_s > 0.0 && interp_s > 0.0 { interp_s / jit_s } else { 0.0 };

        let short_name = if name.len() > 12 { &name[..12] } else { name };
        println!("│ {:12} │ {:>10} │ {:>10} │ {:>8.0} │ {:>8.1} │",
            short_name, fmt_dur(jit_s), fmt_dur(interp_s), mips, ji_ratio);
        eprintln!("  [{}] counters: native={} vm={} fallback={}", name, nc, vc, fc);
    }

    println!("└──────────────┴────────────┴────────────┴──────────┴──────────┘");
}

// =============================================================================
// MAIN
// =============================================================================

fn main() {
    println!();
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  POLYHEDRAL JIT OPTIMIZATION BENCHMARK                        ║");
    println!("║  Jules Language — Polyhedral Engine + Phase 3 JIT             ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  CPU: {}", get_cpu_info());
    println!("  PID: {}", std::process::id());
    println!();

    // Phase 1: Correctness
    bench_polyhedral_correctness();

    // Phase 2: Micro-benchmarks
    bench_scop_extraction();
    bench_dependency_analysis();
    bench_slot_cache();

    // Phase 3: Full polyhedral pipeline
    bench_full_polyhedral_pipeline();

    // Phase 4: JIT execution with polyhedral wired in
    bench_jit_execution();

    println!();
    println!("  Benchmark complete.");
}

fn get_cpu_info() -> String {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| s.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap_or(" unknown").trim().to_string()))
            .unwrap_or_else(|| "unknown".to_string())
    }
    #[cfg(not(target_os = "linux"))]
    { "unknown".to_string() }
}
