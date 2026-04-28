#!/bin/bash
# BOLT (Binary Optimization and Layout Tool) build script for Jules
# BOLT optimizes the binary layout for better instruction cache performance
#
# Usage: ./scripts/bolt-optimize.sh
# Requires: llvm-bolt, merge-fdata (from LLVM tools)

set -e

BINARY="target/release/jules"
BOLT_DIR="/tmp/jules-bolt"

echo "=== Jules BOLT Optimization Script ==="
echo ""

# Check requirements
if ! command -v llvm-bolt &> /dev/null; then
    echo "Error: llvm-bolt not found. Install LLVM tools."
    echo "On Ubuntu: sudo apt install llvm-16-tools"
    echo "Or build from source: https://github.com/llvm/llvm-project"
    exit 1
fi

# Build a normal release binary first
echo "[1/4] Building release binary with perf data..."
cargo build --release --features full 2>&1 | tail -5

# Build with -fdata-sections for better BOLT results
echo "[2/4] Building with perf-friendly flags..."
RUSTFLAGS="-C link-arg=-Wl,--emit-relocs -C force-frame-pointers=no" \
    cargo build --release --features full 2>&1 | tail -5

# Create perf data directory
mkdir -p "$BOLT_DIR"

# Generate perf data by running workloads
echo "[3/4] Generating perf data..."
perf record -e branches:u -j any,u -o "$BOLT_DIR/perf.data" \
    ./$BINARY run examples/benchmark.jules 2>/dev/null || {
    echo "Warning: perf record failed. Running without perf data."
}

# Run BOLT optimization
echo "[4/4] Running BOLT optimization..."
llvm-bolt $BINARY \
    --perf-data="$BOLT_DIR/perf.data" \
    -o="${BINARY}.bolt" \
    --reorder-blocks=cache+ \
    --reorder-functions=cache+ \
    --split-functions=2 \
    --split-all-cold \
    --dyno-stats \
    --use-gnu-stack \
    2>&1 | tail -10 || {
    echo "Warning: BOLT optimization failed."
    echo "The regular release build is still available."
    exit 0
}

# Replace original with BOLT-optimized binary
mv "${BINARY}.bolt" "${BINARY}.bolt-optimized"
echo ""
echo "=== BOLT Optimization Complete ==="
echo "Optimized binary: ${BINARY}.bolt-optimized"
echo "Run with: ./${BINARY}.bolt-optimized <command>"
