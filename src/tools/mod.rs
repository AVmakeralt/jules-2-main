pub mod asset_importer;
pub mod bench_optimizations;
pub mod ffi;
pub mod frame_debugger;
pub mod hot_reload;
pub mod profiling_tools;
pub mod shader_tooling;
pub mod string_intern;
pub mod throughput_sanity;
// train_gnn is a binary-only target — it has fn main() and uses jules:: imports.
// Do NOT include it as a lib module; compile it via `cargo run --bin train-gnn`.
