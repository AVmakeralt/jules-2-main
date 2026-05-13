# Global Performance Benchmarks & Golden Standards

This document serves as the **Single Source of Truth** for the best-ever recorded performance metrics across all modules of the Jules programming language runtime.

## The Golden Rule

**Never compare your local run only against your previous run.** Every benchmark must be measured against the "All-Time Best" (ATB) recorded in this file. This prevents **Performance Drift**, where small regressions are overlooked because they are compared to a recently degraded baseline rather than the historical peak.

---

## Benchmark Environment

| Parameter | Value |
| :--- | :--- |
| Architecture | x86_64 |
| CPU | Intel Xeon Processor (4 vCPU, 2800 MHz) |
| CPU Features | AVX-512, AMX, FMA, SSE4.2, AES-NI, TSX, PKU, AVX2, BMI1/2 |
| L1d Cache | 96 KiB |
| L1i Cache | 128 KiB |
| L3 Cache | 516096 KB |
| RAM | 7936 MB |
| OS | Linux 5.10.134 (containerized, KVM) |
| Rust Profile | `release` (opt-level=3, LTO=fat, codegen-units=1, panic=abort) |
| Rust Version | rustc 1.95.0 (2026-04-14) |
| Commit | `a1719ba` |

---

## Benchmark Registry

### 1. Core Compiler Pipeline (micro-benchmark)

Measures the end-to-end compilation pipeline from source string to ready-to-run program. 30 iterations per sample.

| Metric | Unit | All-Time Best (ATB) | Date Recorded | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Compile: tiny-main (p50) | us | 9.7 | 2026-05-14 | `fn main() {}` | NEW ATB |
| Compile: tiny-main (p95) | us | 26.2 | 2026-05-14 | NEW ATB |
| Compile: stress-lets 400 vars (p50) | us | 523.1 | 2026-05-14 | 400 `let` bindings | NEW ATB |
| Compile: stress-lets 400 vars (p95) | ms | 1.572 | 2026-05-14 | NEW ATB |
| Compile: ml-kernel 64-wide (p50) | us | 150.6 | 2026-05-14 | ML kernel with 64 vars | NEW ATB |
| Compile: ml-kernel 64-wide (p95) | us | 189.6 | 2026-05-14 | NEW ATB |
| Compile+Run: prime-sieve (p50) | us | 878.8 | 2026-05-14 | Sieve to 1000 |
| Compile+Run: prime-sieve runtime (p50) | ms | 9.641 | 2026-05-14 | Interpreter execution | NEW ATB |
| Compile+Run: prime-check (p50) | ms | 4.161 | 2026-05-14 | 10 prime checks | NEW ATB |
| Compile+Run: prime-check runtime (p50) | us | 24.4 | 2026-05-14 | Interpreter execution | NEW ATB |
| RSS overhead (tiny program) | MiB | 2.88 | 2026-05-14 | Delta from baseline |

### 2. BytecodeVM Execution (quick-inferno)

Runtime execution performance via the BytecodeVM backend. 10 iterations per benchmark.

| Benchmark | Unit | All-Time Best (ATB) | Date Recorded | Correctness |
| :--- | :--- | :--- | :--- | :--- |
| deep-arith-chain | us/iter | N/A | 2026-05-14 | FAIL (pipeline: borrow-check move error) |
| fibonacci-30 | us/iter | N/A | 2026-05-14 | FAIL (pipeline: borrow-check move error) |
| const-fold-heavy | us/iter | N/A | 2026-05-14 | FAIL (pipeline: borrow-check move error) |
| sum-to-1000 | us/iter | N/A | 2026-05-14 | FAIL (pipeline: borrow-check move error) |
| prime-sieve-500 | us/iter | N/A | 2026-05-14 | FAIL (pipeline: borrow-check move error) |
| collatz-200 | us/iter | N/A | 2026-05-14 | FAIL ([ir-typeck] return type () does not match function return type i32) |
| Compile overhead (total, 7 benches) | us | N/A | 2026-05-14 | All benchmarks fail at IR validation |

### 3. Inferno Suite (bench-inferno, 3 iterations)

Full stress-test suite covering arithmetic, loops, recursion, function calls, and control flow.

| Benchmark | Engine | Time (s/iter) | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| wrap-overflow-chain | VM | 0.0005 | PASS | 200 iterations wrapping mul/add |
| deep-arith-chain | VM | 0.0000 | PASS | 50-term constant fold |
| mixed-int-ops | VM | 0.0002 | PASS | 100 iterations add/sub/mul/div/rem |
| nested-5-deep | VM | 0.0033 | PASS | 4^5 = 1024 iterations |
| triangular-loop | VM | 0.0032 | PASS | Triangular double loop to 50 |
| loop-continue | VM | 0.0005 | PASS | 200 iterations with continue |
| loop-break-early | VM | 0.0001 | PASS | Early exit at i=71 |
| fibonacci-30 | VM | 0.0000 | FAIL | Off-by-one: got 832040, expected 514229 |
| fib-recursive-10 | - | 0.0000 | FAIL | Compile error: use of moved value `n` |
| collatz-1000 | - | 0.0000 | FAIL | VM runtime: cannot add () and i64 |
| call-overhead-1m | - | 0.0000 | FAIL | VM runtime: cannot add () and i64 |
| multi-arg-calls | - | 0.0000 | FAIL | Compile error: use of moved value `s` |
| deep-if-else | - | 0.0001 | FAIL | Compile error: return type mismatch |
| bool-logic-maze | VM | 0.0005 | PASS | 200 iterations with compound conditions |
| trial-div-1000 | - | 0.0000 | FAIL | VM runtime: cannot add () and i64 |
| many-functions-10 | - | 0.0000 | FAIL | Runtime: undefined variable `{name}` |
| long-function-500 | VM | 0.0000 | PASS | 100 let bindings |
| const-fold-heavy | VM | 0.0000 | PASS | Deep constant folding |
| dead-code-heavy | VM | 0.0005 | FAIL | Expected 1000, got 200 |
| gcd-euclidean | - | 0.0001 | FAIL | Compile error: cannot assign to `y`: value has been moved |
| mandelbrot-50x50 | - | 14.5840 | FAIL | Watchdog: infinite loop detected (OSR de-optimization) |
| mut-var-not-folded | VM | 0.0000 | PASS | Constant folding correctness |
| var-after-mutation | VM | 0.0000 | PASS | |
| multi-mutation | VM | 0.0000 | PASS | |
| cond-mutation | VM | 0.0000 | PASS | |
| self-mutation | VM | 0.0000 | PASS | |
| grand-finale | - | 0.0001 | FAIL | Compile error: cannot assign to `y`: value has been moved |
| **vm-vs-interp-100k** | - | 0.0810 | PASS | interp=4.6ms, vm=76.4ms, speedup=0.06x |

**Inferno Summary:** 16 passed, 12 failed out of 28 benchmarks. Known issues: ownership/move semantics in `let mut` + reassignment patterns, `()` type leaking from if-else branches, and Mandelbrot integer-overflow infinite loop.

### 4. VM vs Interpreter Comparison

Head-to-head performance comparison of the tree-walking interpreter vs the BytecodeVM.

| Metric | Interpreter | BytecodeVM | Speedup | Notes |
| :--- | :--- | :--- | :--- | :--- |
| wrap-overflow-chain (100k iters) | 4.6 ms | 76.4 ms | 0.06x | VM currently slower than interpreter for this workload |

**Note:** The BytecodeVM is currently in early development. The interpreter benefits from direct AST traversal with minimal overhead for simple loop bodies, while the VM incurs dispatch overhead. The VM is expected to outperform the interpreter for larger, more complex programs as the backend matures.

### 5. While-Loop Correctness (bench-speed)

Verification that assignment and control flow work correctly in loops.

| Test | Time (us) | Status |
| :--- | :--- | :--- |
| assign-in-if | 54.8 | PASS (result=1) |
| while-break | 26.3 | PASS (result=3) |
| while-cond | 4.4 | PASS (result=3) |
| while-100 | 17.4 | PASS (result=4950) |
| while-nested | 10.1 | PASS (result=25) |
| while-continue | 7.1 | PASS (result=45) |
| while-break-nested | 7.5 | PASS (result=3) |

### 6. ECS (Entity Component System) Performance

Benchmarked with 20,000 entities, 20 steps, dt=0.016. Each entity has pos (Vec3), vel (Vec3), health (F32), damage (F32).

| Execution Mode | Steps/s | Notes |
| :--- | :--- | :--- |
| baseline (query+update per step) | 269.4 | Naive per-entity query and update |
| soa-linear (SoA traversal) | 8,846.7 | Structure-of-Arrays linear scan |
| fused-linear (integrate_vec3_fused) | 10,481.9 | Fused position+velocity update |
| chunked-fused (chunk=64) | 4,682.8 | Chunked gather/compute/scatter |
| superoptimizer (integrate_vec3_superoptimizer) | 19,896.6 | Superoptimized path (best Jules) |
| aot-hash (AOT cached layout) | 175.4 | Pre-computed layout with hash validation |
| **Rust native (baseline reference)** | **127,759.0** | Pure Rust Vec-of-structs |

| Jules Mode vs Rust | Ratio (sec/step) |
| :--- | :--- |
| superoptimizer / Rust | 6.42x |
| fused-linear / Rust | 12.19x |
| chunked-fused / Rust | 27.28x |
| aot-hash / Rust | 728.24x |

**AOT hotspot profiling:** query=1.0%, fetch=41.7%, math=14.6%, write=42.8%

### 7. JHAL (Hardware Abstraction Layer) Microbenchmarks

All measurements use `black_box()` to prevent dead code elimination. 100,000 iterations unless noted.

| Operation | ns/iter | Throughput (ops/s) | Assessment |
| :--- | :--- | :--- | :--- |
| Ring buffer enqueue+dequeue | 1.1 | 898,077,217 | Fast (register ops) |
| Ring buffer bulk 255 | 0.4 | 2,293,577,982 | Trivial (1-2 instructions) |
| PCI BDF construction | 1.4 | 706,998,579 | Fast (register ops) |
| PCI BDF bounds check | 1.7 | 582,360,306 | Fast (register ops) |
| Device registry register | 0.6 | 1,655,629,139 | Fast (register ops) |
| Device registry find_by_class | 28.4 | 35,190,203 | Moderate (atomics/branches) |
| Zero-heap formatting (decimal) | 1.8 | 557,289,345 | Fast (register ops) |
| Zero-heap formatting (hex) | 0.3 | 3,145,346,460 | Trivial (1-2 instructions) |
| APIC timer config construction | 0.7 | 1,350,128,937 | Fast (register ops) |
| APIC register offset computation | 0.9 | 1,099,468,956 | Fast (register ops) |
| Console write buffered | 8.9 | 112,253,981 | Normal (ALU ops) |
| SFI config creation | 0.9 | 1,137,307,084 | Fast (register ops) |
| SFI pointer masking | 0.5 | 2,139,448,306 | Trivial (1-2 instructions) |
| SFI invariant verification | 0.3 | 3,072,574,203 | Trivial (1-2 instructions) |
| TSX status construction | 1.1 | 942,658,107 | Fast (register ops) |
| TSX transaction bound proof | 0.5 | 2,069,579,255 | Trivial (1-2 instructions) |
| AMX scheduler matmul 4x4 | 0.0 | 208,333,333,333 | DCE-DETECTED (bench artifact) |
| IRQ register partition proof | 5.9 | 170,460,550 | Normal (ALU ops) |
| IDT entry construction | 0.6 | 1,799,888,407 | Fast (register ops) |
| IDT full table build | 0.1 | 11,363,636,364 | DCE-DETECTED (bench artifact) |
| IRQ predictor record | 17.4 | 57,478,069 | Moderate (atomics/branches) |
| IRQ predictor predict | 91.4 | 10,938,797 | Moderate (atomics/branches) |
| Huge page allocator | 0.0 | 384,615,384,615 | DCE-DETECTED (bench artifact) |
| IOMMU DMA check | 0.7 | 1,516,392,200 | Fast (register ops) |
| CFI jump table lookup | 6.0 | 166,707,232 | Normal (ALU ops) |
| Identity map 1GB mapping | 210.6 | 4,748,203 | Slow (memory/cache) |

**Note:** Three JHAL benchmarks flagged `DCE-DETECTED` because the compiler can fully eliminate the measured work (AMX matmul fallback, IDT table build, huge page alloc). These need `black_box()` added to the output path.

### 8. Test Suite Summary

| Category | Total | Passed | Failed |
| :--- | :--- | :--- | :--- |
| Library tests (`cargo test --lib`) | 784 | 752 | 32 (+1 stack overflow abort) |

---

## Known Issues Impacting Benchmarks

1. **Ownership/Move Errors (IR Borrow Checker):** The IR borrow checker is now authoritative and catches all ownership violations as hard errors. Previously, these were silently allowed through the AST path. This causes many inferno benchmarks to fail at compilation. The benchmarks need their Jules source code updated to comply with ownership rules (e.g., using `copy` for integers, avoiding re-assignment of moved values). Affected: `wrap-overflow-chain`, `deep-arith-chain`, `mixed-int-ops`, `fibonacci-30`, `sum-to-1000`, `call-overhead-1m`, `multi-arg-calls`, `trial-div-1000`, `many-functions-10`, `long-function-500`, `const-fold-heavy`, `dead-code-heavy`, `mandelbrot-50x50`, `mut-var-not-folded`, `var-after-mutation`, `multi-mutation`, `cond-mutation`, `self-mutation`.

2. **Unit Type Leak (IR Type Checker):** The IR type checker now catches `()` type mismatches that were previously runtime errors. If-else branches and assignment statements that produce `()` instead of the intended value are now compile-time errors. This affects `gcd-euclidean`, `collatz-200`, `bool-logic-maze`, `deep-if-else`, `grand-finale`, `while-break-nested`. Fix: the lowering needs to properly propagate expression values from if-else blocks and assignment chains.

3. **Undefined Values (IR Lowering):** Some benchmarks produce `undefined value` errors in the IR type checker, indicating the lowering doesn't define all values needed in subsequent instructions (phi nodes, control flow). Affected: `nested-5-deep`, `triangular-loop`, `while-nested`, `while-continue`.

4. **i64/i32 Type Consistency (FIXED):** The lowerer, type checker, and `Int`/`int` type alias now all consistently default to `i32`. Previously, the lowerer defaulted to `i64`, the type checker to `i32`, and `Int` mapped to `i64`, causing type mismatches across the pipeline. This fix improved test suite results from 42 failures to 32 failures.

5. **Phi Node State Merging (FIXED):** The IR borrow checker now correctly propagates `Moved` state through phi nodes. Previously, `define_value()` in `check_instr()` overwrote the merged state from `merge_phi_states()`, making use-after-move detection impossible across control flow merges.

6. **Mandelbrot Infinite Loop:** The integer Mandelbrot benchmark triggers the watchdog timer due to integer overflow causing the escape condition to never be met. This is a language-level integer overflow issue, not a runtime bug.

7. **BytecodeVM vs Interpreter:** The VM currently runs slower than the interpreter for simple workloads. The interpreter's direct AST traversal has lower dispatch overhead than the VM's instruction decode loop for trivial programs.

8. **DCE in JHAL Benchmarks:** Three JHAL operations are optimized away by the Rust compiler despite `black_box()` on inputs. The output values also need to be consumed via `black_box()`.

9. **Identity Map Stack Overflow:** `test_identity_map_1gb` overflows the default stack during tests. Run with `RUST_MIN_STACK=16777216` to mitigate.

---

## How to Update This File

1. **Run Benchmarks:** Execute the standard performance suite in a controlled environment:
   ```bash
   cargo run --release --bin micro-benchmark 30
   cargo run --release --bin quick-inferno
   cargo run --release --bin bench-inferno 3
   cargo run --release --bin bench-jhal
   cargo run --release --bin bench-ecs -- 20000 20
   cargo run --release --bin bench-speed
   ```

2. **Comparison:** If your result is **better** than the ATB listed above, you must update this file in your PR.

3. **Verification:** Provide the environment specs and a log snippet proving the new record.

4. **Regression Check:** If your code is slower than the ATB, it is considered a **regression**, even if it is faster than the current `main` branch. You must optimize further or provide a technical justification (e.g., a necessary security trade-off).

> **Warning:** Performance regressions disguised as "slight variations" lead to technical debt. Respect the Golden Standard.
