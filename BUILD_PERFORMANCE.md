# Build Performance Guide

## Why Was Build Slow?

The original configuration had these settings that made builds extremely slow:

| Setting | Value | Impact on Compile Time |
|---------|-------|------------------------|
| `codegen-units` | 1 | **Prevents parallel codegen** - single-threaded LLVM compilation |
| `lto` | "fat" | **Full LTO** - analyzes entire crate graph (very expensive) |
| `incremental` | false | **No incremental compilation** - rebuilds everything from scratch |
| LLVM unroll threshold | 512 | **Aggressive unrolling** - increases code size and compile time |
| LLVM vectorize flags | forced 32-wide | **Forces vectorization** - extra compile-time analysis |

### Before (Slow Build)
```toml
[profile.release]
opt-level = 3
lto = "fat"              # Full LTO: 2-5 minutes
codegen-units = 1        # Single-threaded: 3-10 minutes
incremental = false      # No caching: always rebuilds all
```

**Total build time: 5-15 minutes**

### After (Fast Build)

**Development** (`cargo build`):
```toml
opt-level = 1
codegen-units = 16       # Parallel codegen
incremental = true       # Fast rebuilds
```
**Build time: 30-60 seconds**

**Release** (`cargo build --release`):
```toml
opt-level = 3
lto = "thin"             # Thin LTO (much faster)
codegen-units = 4        # 4 parallel units
```
**Build time: 1-3 minutes**

**Max Performance** (`cargo build --profile max-perf`):
```toml
opt-level = 3
lto = "fat"              # Full LTO
codegen-units = 1        # Single unit
```
**Build time: 5-15 minutes (only for final builds)**

---

## Build Profiles Explained

### 1. Development (default)
```bash
cargo build
```
- **Compile time**: 30-60 seconds
- **Runtime speed**: Good enough for testing
- **Use case**: Frequent rebuilds during development

### 2. Release (production)
```bash
cargo build --release
```
- **Compile time**: 1-3 minutes
- **Runtime speed**: Fast (95% of max-perf)
- **Use case**: CI/CD, production deployments

### 3. Max Performance (benchmarking)
```bash
cargo build --profile max-perf
```
- **Compile time**: 5-15 minutes
- **Runtime speed**: Maximum (100%)
- **Use case**: Benchmarks, performance testing, final releases

---

## Runtime vs Compile Time Trade-offs

### What Affects Compile Time?
- `codegen-units` (lower = slower)
- `lto` (fat > thin > off)
- `opt-level` (3 > 2 > 1 > 0)
- `incremental` (true = faster rebuilds)
- LLVM optimization flags (unroll, vectorize, etc.)

### What Affects Runtime Performance?
- `opt-level` (3 = fastest)
- `target-cpu=native` (use all CPU features)
- `target-feature=+avx2,+fma,...` (SIMD instructions)
- LTO (better cross-crate inlining)
- `panic = "abort"` (smaller binaries)

### Key Insight
**CPU feature flags do NOT slow down compilation!**
- `-C target-cpu=native` - detected at compile time, zero cost
- `-C target-feature=+avx2,...` - enables instructions, zero compile overhead
- **These are in .cargo/config.toml for ALL builds**

**What DOES slow down compilation:**
- `codegen-units = 1` - forces single-threaded LLVM
- `lto = "fat"` - full crate graph analysis
- LLVM unroll/vectorize args - extra optimization passes

---

## Recommended Workflow

### Daily Development
```bash
# Fast build for testing
cargo build

# Run tests
cargo test

# Run your program
cargo run -- my_program.jules
```

### Pre-release Testing
```bash
# Build with release optimizations
cargo build --release

# Run benchmarks
cargo run --bin micro-benchmark --release

# Test with release binary
./target/release/jules --jit my_program.jules
```

### Final Release / Benchmarking
```bash
# Build with maximum optimizations
cargo build --profile max-perf --features ultra-fast

# Run final benchmarks
./target/max-perf/jules --bench

# This binary is the FASTEST possible
```

---

## Build Time Comparison

| Profile | First Build | Rebuild (cached) | Runtime Speed |
|---------|-------------|------------------|---------------|
| dev | 30-60s | 5-10s | 1.0x (baseline) |
| release | 1-3min | 30-60s | 2-5x faster |
| max-perf | 5-15min | 1-3min | 5-10x fastest |

---

## Tips for Faster Builds

### 1. Use `cargo check` for quick feedback
```bash
cargo check              # Type-check only (5-10 seconds)
cargo check --release    # With release optimizations
```

### 2. Incremental compilation
Already enabled for dev profile. For faster release builds:
```toml
[profile.release]
incremental = true       # Trade 10% binary size for 50% faster rebuilds
```

### 3. Parallel compilation
Already configured in `.cargo/config.toml`:
```toml
[build]
jobs = 8                 # Use 8 parallel compilations
```

### 4. Reduce dependencies
- `pyo3` + `numpy` (Python bindings) add 30-60s to builds
- Build without Python unless needed:
  ```bash
  cargo build --release --no-default-features --features ffi-c,phase3-jit
  ```

### 5. Use `sccache` for caching
```bash
cargo install sccache
export RUSTC_WRAPPER=sccache
cargo build --release    # Subsequent builds are instant
```

---

## Troubleshooting

### Build still slow?

Check what's taking time:
```bash
cargo build --release --timings
# Opens timings.html with detailed breakdown
```

### Common issues:

1. **Cranelift codegen** (optional, adds 1-2 minutes):
   ```bash
   cargo build --release --no-default-features
   ```

2. **LLVM remarks** (verbose output, slows terminal):
   Remove `-C llvm-args=...` from rustflags

3. **Too many dependencies**:
   ```bash
   cargo tree | head -50  # See dependency tree
   ```

### Memory usage during build

Release builds use 4-8 GB RAM. If you're memory-constrained:
```toml
[profile.release]
codegen-units = 1        # Less memory, but slower
```

---

## Summary

**Fast builds = Smart profile choices**

- `cargo build` → Development (fast)
- `cargo build --release` → Production (balanced)
- `cargo build --profile max-perf` → Maximum speed (slow compile)

**The optimizations that matter for runtime (CPU features, SIMD) don't slow down compilation!**

Only LTO and codegen-units affect compile time, and they're now isolated to the max-perf profile.
