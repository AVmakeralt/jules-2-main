// =============================================================================
// Build script for the Jules compiler
// =============================================================================
// Handles link search paths for optional dependencies.
//
// For the smt-verify feature, you need to set these environment variables
// before running cargo:
//   export LIBCLANG_PATH="/home/z/.local/lib"
//   export Z3_SYS_Z3_HEADER="/home/z/.local/include/z3.h"
//   export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-linux-gnu/14/include -I/home/z/.local/include"
//   export LD_LIBRARY_PATH="/home/z/.local/lib:$LD_LIBRARY_PATH"
//
// Or use the convenience script: ./scripts/build_with_smt.sh
// =============================================================================

fn main() {
    // Tell cargo where to find libz3.so at link time
    #[cfg(feature = "smt-verify")]
    {
        let lib_dirs = [
            "/home/z/.local/lib",
            "/usr/lib",
            "/usr/local/lib",
        ];
        for dir in &lib_dirs {
            if std::path::Path::new(dir).exists() {
                println!("cargo:rustc-link-search=native={}", dir);
            }
        }
        // Link libz3
        println!("cargo:rustc-link-lib=z3");
    }

    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    println!("cargo:rerun-if-env-changed=Z3_SYS_Z3_HEADER");
    println!("cargo:rerun-if-changed=build.rs");
}
