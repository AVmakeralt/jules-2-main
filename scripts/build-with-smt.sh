#!/usr/bin/env bash
# =============================================================================
# Build Jules with smt-verify feature enabled
# =============================================================================
# This script sets up the required environment variables for building with
# the Z3 SMT solver (needed for CEGIS verification in the superoptimizer).
#
# Prerequisites:
#   1. libclang.so — install via: apt install libclang-dev
#      Or download to: /home/z/.local/lib/libclang.so
#   2. Z3 library + headers — install via: apt install libz3-dev
#      Or download prebuilt Z3 to: /home/z/.local/{lib,include}
#   3. GCC headers (for stdbool.h etc.) — usually already present
#
# The script auto-detects libclang and Z3 in common locations.
# Override detection by setting env vars before calling this script.
# =============================================================================

set -euo pipefail

# --- Auto-detect libclang ---
if [ -z "${LIBCLANG_PATH:-}" ]; then
    for dir in /home/z/.local/lib /usr/lib/llvm-*/lib /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib; do
        if [ -f "$dir/libclang.so" ] || [ -f "$dir/libclang.so.17" ] || [ -f "$dir/libclang-17.so" ]; then
            export LIBCLANG_PATH="$dir"
            echo "[smt-build] Found libclang at: $LIBCLANG_PATH"
            break
        fi
    done
    if [ -z "${LIBCLANG_PATH:-}" ]; then
        echo "[smt-build] ERROR: libclang.so not found. Install libclang-dev or set LIBCLANG_PATH." >&2
        exit 1
    fi
fi

# --- Auto-detect Z3 headers ---
if [ -z "${Z3_SYS_Z3_HEADER:-}" ]; then
    for path in /home/z/.local/include/z3.h /usr/include/z3.h /usr/local/include/z3.h; do
        if [ -f "$path" ]; then
            export Z3_SYS_Z3_HEADER="$path"
            echo "[smt-build] Found z3.h at: $Z3_SYS_Z3_HEADER"
            break
        fi
    done
    if [ -z "${Z3_SYS_Z3_HEADER:-}" ]; then
        echo "[smt-build] ERROR: z3.h not found. Install libz3-dev or set Z3_SYS_Z3_HEADER." >&2
        exit 1
    fi
fi

# --- Auto-detect GCC include path (for stdbool.h etc.) ---
if [ -z "${BINDGEN_EXTRA_CLANG_ARGS:-}" ]; then
    GCC_INCLUDE=$(find /usr/lib/gcc/x86_64-linux-gnu -maxdepth 2 -name "stdbool.h" -print -quit 2>/dev/null | xargs dirname 2>/dev/null || true)
    Z3_INCLUDE=$(dirname "$Z3_SYS_Z3_HEADER" 2>/dev/null || true)
    if [ -n "$GCC_INCLUDE" ] || [ -n "$Z3_INCLUDE" ]; then
        export BINDGEN_EXTRA_CLANG_ARGS=""
        [ -n "$GCC_INCLUDE" ] && BINDGEN_EXTRA_CLANG_ARGS="-I$GCC_INCLUDE"
        [ -n "$Z3_INCLUDE" ] && BINDGEN_EXTRA_CLANG_ARGS="$BINDGEN_EXTRA_CLANG_ARGS -I$Z3_INCLUDE"
        echo "[smt-build] BINDGEN_EXTRA_CLANG_ARGS=$BINDGEN_EXTRA_CLANG_ARGS"
    fi
fi

# --- Set library path ---
Z3_LIB_DIR=$(dirname "$(find /home/z/.local/lib /usr/lib /usr/local/lib -name 'libz3.so' -print -quit 2>/dev/null || echo '/usr/lib/libz3.so')")
export LD_LIBRARY_PATH="${Z3_LIB_DIR:-/usr/lib}:${LD_LIBRARY_PATH:-}"

echo "[smt-build] Building with smt-verify feature..."
echo "[smt-build]   LIBCLANG_PATH=$LIBCLANG_PATH"
echo "[smt-build]   Z3_SYS_Z3_HEADER=$Z3_SYS_Z3_HEADER"
echo "[smt-build]   LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# --- Run cargo ---
cargo build --features "core-superopt,smt-verify" "$@"
