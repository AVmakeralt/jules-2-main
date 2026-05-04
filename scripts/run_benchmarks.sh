#!/bin/bash
# Jules Performance Benchmark Suite
# Run this from the jules-2-main directory

set -e

echo "=========================================="
echo "  Jules v3.0 Performance Benchmarks"
echo "=========================================="
echo ""

# Check if Rust toolchain is available
if ! command -v cargo &> /dev/null; then
    echo "ERROR: Rust toolchain not found. Install with:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "  source ~/.cargo/env"
    exit 1
fi

# Build release with maximum optimizations
echo "[1/5] Building Jules with max-perf profile..."
cargo build --release --profile=max-perf --features=full 2>&1 | tail -5
echo ""

# Build benchmarks
echo "[2/5] Building benchmarks..."
cargo build --release --features=full 2>&1 | tail -3
echo ""

# Run micro-benchmarks (compile speed)
echo "[3/5] Running micro-benchmarks (compile speed)..."
echo "----------------------------------------------"
./target/release/micro-benchmark 10
echo ""

# Run speed benchmarks (runtime speed)
echo "[4/5] Running speed benchmarks (runtime)..."
echo "----------------------------------------------"
./target/release/bench-speed 100
echo ""

# Run ECS benchmarks
echo "[5/5] Running ECS benchmarks..."
echo "----------------------------------------------"
./target/release/bench-ecs 1000
echo ""

echo "=========================================="
echo "  Benchmark Complete!"
echo "=========================================="
echo ""
echo "To compare with JIT:"
echo "  ./target/release/jules run --jit examples/benchmark.jules"
echo ""
echo "To compare with AOT (-O3 Thorough):"
echo "  ./target/release/jules build --opt=3 -o out.o source.jules"
echo "  chmod +x out.o && ./out.o"
echo ""

# Print hardware info if available
if command -v lscpu &> /dev/null; then
    echo "Hardware:"
    lscpu | grep -E "Model name|CPU\(s\)|Thread|Core"
fi
