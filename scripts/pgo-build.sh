#!/bin/bash
# PGO (Profile-Guided Optimization) build script for Jules
# This script:
#   1. Builds with instrumentation
#   2. Runs representative workloads to collect profiles
#   3. Rebuilds using the collected profile data
#
# Usage: ./scripts/pgo-build.sh

set -e

echo "=== Jules PGO Build Script ==="
echo ""

# Clean previous builds
echo "[1/5] Cleaning previous builds..."
cargo clean

# Step 1: Build with PGO instrumentation
echo "[2/5] Building with PGO instrumentation..."
export RUSTFLAGS="-Cprofile-generate=/tmp/jules-pgo-profiles"
cargo build --release --features full 2>&1 | tail -20

# Step 2: Run workloads to generate profile data
echo "[3/5] Running workloads to generate profile data..."

# Run a representative set of workloads
echo "  - Running ECS benchmark..."
./target/release/bench-ecs 2>/dev/null || true

echo "  - Running interpreter benchmark..."
./target/release/micro-benchmark 2>/dev/null || true

echo "  - Running chess ML benchmark..."
./target/release/bench-chess-ml 2>/dev/null || true

echo "  - Running sample Jules programs..."
if [ -f "examples/hello.jules" ]; then
    ./target/release/jules run examples/hello.jules 2>/dev/null || true
fi

# Merge PGO profiles
echo "[4/5] Merging PGO profiles..."
llvm-profdata merge -o /tmp/jules-pgo-profiles/merged.profdata /tmp/jules-pgo-profiles/*.profraw 2>/dev/null || {
    echo "Warning: Profile merge failed. PGO may not be fully effective."
}

# Step 3: Rebuild with PGO profile data
echo "[5/5] Rebuilding with PGO profile data..."
export RUSTFLAGS="-Cprofile-use=/tmp/jules-pgo-profiles/merged.profdata -Cllvm-args=-pgo-warn-missing-function"
cargo build --release --features full 2>&1 | tail -20

echo ""
echo "=== PGO Build Complete ==="
echo "Optimized binary: target/release/jules"
echo "Profile data: /tmp/jules-pgo-profiles/"
