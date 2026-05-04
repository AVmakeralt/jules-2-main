pub mod bytecode_vm;
pub mod gpu_backend;
pub mod interp;
pub mod io_uring;
/// PrefetchEngine with CPU topology detection, stride prediction, etc.
/// Not actual memory management — feature-gated to reduce default build size.
#[cfg(feature = "gnn-optimizer")]
pub mod memory_management;
pub mod networking;
pub mod symbol;
pub mod threading;

// ── Jules Hardware Abstraction Layer (JHAL) ──────────────────────────────────
// Low-level drivers for bare-metal boot: Local APIC, UART 16550, PCIe.
// Uses "Provable Hardware Modeling" — every bit named, Acquire/Release
// ordering, zero heap allocation, TSX-safe.
pub mod jhal;
