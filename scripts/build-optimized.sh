#!/bin/bash
# Fast build script for maximum performance Jules binary
# Uses all available optimizations except PGO/BOLT

set -e

echo "=== Building Maximum Performance Jules Binary ==="
echo ""

# Detect CPU features
echo "CPU Features Detected:"
echo "  AVX2: $(grep -q avx2 /proc/cpuinfo 2>/dev/null && echo 'YES' || echo 'NO')"
echo "  FMA: $(grep -q fma /proc/cpuinfo 2>/dev/null && echo 'YES' || echo 'NO')"
echo "  BMI2: $(grep -q bmi2 /proc/cpuinfo 2>/dev/null && echo 'YES' || echo 'NO')"
echo "  AVX-512: $(grep -q avx512 /proc/cpuinfo 2>/dev/null && echo 'YES' || echo 'NO')"
echo ""

# Clean and build
echo "Building with maximum optimizations..."
cargo build --release --features full 2>&1 | tail -20

BINARY="target/release/jules"
if [ -f "$BINARY" ]; then
    SIZE=$(stat -c%s "$BINARY" 2>/dev/null || stat -f%z "$BINARY" 2>/dev/null)
    echo ""
    echo "=== Build Complete ==="
    echo "Binary: $BINARY"
    echo "Size: $(( SIZE / 1024 )) KB"
    echo ""
    echo "Run with: ./$BINARY <command>"
    echo ""
    echo "For even more performance, consider:"
    echo "  - PGO: ./scripts/pgo-build.sh"
    echo "  - BOLT: ./scripts/bolt-optimize.sh"
else
    echo "Build failed!"
    exit 1
fi
