#![allow(dead_code)]
//! Ultra-high-performance prefetch engine for bytecode VM execution.
//!
//! **Goal**: keep every CPU execution unit fed at all times by issuing the
//! right cache hint, at the right level, at the right distance — all derived
//! from measured hardware characteristics rather than guesswork.
//!
//! ## Architecture
//!
//! ```text
//!  ┌──────────────────────────── PrefetchEngine ──────────────────────────────┐
//!  │                                                                          │
//!  │  Cache line 0 — HOT PATH                                                │
//!  │  ┌──────────────────┐  ┌────────────────────┐  ┌──────────────────────┐ │
//!  │  │  DistanceTable   │  │  BandwidthThrottle │  │  Epoch / Density     │ │
//!  │  │  (6 × u8)        │  │  (budget u32)      │  │  (u32, u8)           │ │
//!  │  └──────────────────┘  └────────────────────┘  └──────────────────────┘ │
//!  │                                                                          │
//!  │  Cache lines 1-2 — WARM                                                 │
//!  │  ┌──────────────────────────────────────────────────────────────────┐   │
//!  │  │  StridePredictor   (4 streams × last_addr/stride/confidence)     │   │
//!  │  └──────────────────────────────────────────────────────────────────┘   │
//!  │                                                                          │
//!  │  Cache lines 3-4 — COLD (written infrequently)                         │
//!  │  ┌──────────────────┐  ┌──────────────────────────────────────────────┐ │
//!  │  │   CpuTopology    │  │         LatencyCalibrator                    │ │
//!  │  │   (CPUID result) │  │  pointer-chase chain + cached measurements   │ │
//!  │  └──────────────────┘  └──────────────────────────────────────────────┘ │
//!  │                                                                          │
//!  │  AtomicU32 feedback channels (written from perf-counter callbacks)      │
//!  │  ┌──────────────┐  ┌──────────────┐                                    │
//!  │  │  miss_rate   │  │   ipc_x10    │                                    │
//!  └──┴──────────────┴──┴──────────────┴────────────────────────────────────┘
//! ```
//!
//! ## Layered prefetch strategy
//!
//! ```text
//!   PC ──► L1 hint (insn_l1 ahead)  ← hide L1-miss latency (~4-12 cycles)
//!      ──► L2 hint (insn_l2 ahead)  ← hide L2-miss latency (~40 cycles)
//!      ──► L3 hint (insn_l3 ahead)  ← build L3 pipeline    (~200 cycles)
//!
//!   All three fire in parallel — modern CPUs process prefetch hints
//!   out-of-order so this costs ≈0 extra cycles on the hot dispatch path.
//! ```

use core::sync::atomic::{AtomicU32, Ordering};

// ── Top-level constants ───────────────────────────────────────────────────────

/// Number of independent stride streams to track simultaneously.
const STRIDE_STREAMS: usize = 4;

/// Minimum successive confirmations before acting on a predicted stride.
const STRIDE_CONFIDENCE_THRESHOLD: u8 = 3;

/// TSC cycles between automatic latency re-calibrations (~16 ms @ 3 GHz).
const RECALIBRATE_INTERVAL: u64 = 50_000_000;

/// Fallback latency constants when measurement is unavailable.
const DEFAULT_RAM_LATENCY: u32 = 200;
const DEFAULT_L2_LATENCY: u32 = 40;

/// Maximum prefetch hints issued per epoch to avoid saturating MSHRs.
const EPOCH_BUDGET: u32 = 192;

/// PREFETCHW hint constant — exclusive (write-intent), temporal, L1.
/// Acquires the cache line in Modified state, eliminating the RFO stall
/// that a cold write would otherwise cause.
#[cfg(target_arch = "x86_64")]
const MM_HINT_ET0: i32 = 7;

// ── CpuTopology ───────────────────────────────────────────────────────────────

/// Detected CPU cache hierarchy, filled once at startup via CPUID / CTR_EL0.
#[derive(Debug, Clone, Copy)]
pub struct CpuTopology {
    /// Physical cache line size in bytes.
    pub cache_line_bytes: usize,
    /// L1 data cache capacity in bytes.
    pub l1d_bytes: usize,
    /// L2 (unified) cache capacity in bytes.
    pub l2_bytes: usize,
    /// L3 (LLC) capacity in bytes.
    pub l3_bytes: usize,
    /// Approximate MSHR (Miss Status Holding Register) count.
    /// Bounds how many outstanding cache misses the CPU can track at once.
    pub mshr_count: usize,
}

impl CpuTopology {
    /// Probe the hardware at runtime and return the detected topology.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        { Self::detect_x86() }
        #[cfg(target_arch = "aarch64")]
        { Self::detect_aarch64() }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        { Self::fallback() }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86() -> Self {
        // Cache line size from CPUID leaf 1, EBX[15:8] × 8.
        let cache_line_bytes = unsafe {
            let r = core::arch::x86_64::__cpuid(1);
            let sz = ((r.ebx >> 8) & 0xFF) as usize * 8;
            if sz > 0 { sz } else { 64 }
        };

        let (mut l1d, mut l2, mut l3) = (32 << 10, 256 << 10, 8 << 20);

        // CPUID leaf 4: Intel Deterministic Cache Parameters.
        // AMD mirrors this in leaf 0x8000001D; leaf 4 works on both.
        unsafe {
            for sub in 0u32..16 {
                let r = core::arch::x86_64::__cpuid_count(4, sub);
                let cache_type = r.eax & 0x1F;
                if cache_type == 0 { break; } // terminator entry
                let level = (r.eax >> 5) & 0x7;
                // type 1 = data, type 3 = unified (both hold data).
                if cache_type == 1 || cache_type == 3 {
                    let ways       = ((r.ebx >> 22) & 0x3FF) as usize + 1;
                    let partitions = ((r.ebx >> 12) & 0x3FF) as usize + 1;
                    let line       = (r.ebx & 0xFFF) as usize + 1;
                    let sets        = r.ecx as usize + 1;
                    let size = ways * partitions * line * sets;
                    match (level, cache_type) {
                        (1, 1) => l1d = size,
                        (2, _) => l2  = size,
                        (3, _) => l3  = size,
                        _      => {}
                    }
                }
            }
        }

        let mshr_count = Self::x86_mshr_estimate();
        Self { cache_line_bytes, l1d_bytes: l1d, l2_bytes: l2, l3_bytes: l3, mshr_count }
    }

    #[cfg(target_arch = "x86_64")]
    fn x86_mshr_estimate() -> usize {
        // Max basic CPUID leaf >= 0x16 implies a recent micro-arch with more MSHRs.
        let max_leaf = unsafe { core::arch::x86_64::__cpuid(0).eax };
        if max_leaf >= 0x16 { 16 } else { 10 }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_aarch64() -> Self {
        // CTR_EL0 is readable from EL0 when SCTLR_EL1.UCT=1 (Linux sets this).
        // DminLine = bits[19:16] = log2(min D-cache line size in words).
        let ctr: u64;
        unsafe {
            core::arch::asm!(
                "mrs {r}, ctr_el0",
                r = out(reg) ctr,
                options(nostack, readonly, preserves_flags)
            );
        }
        let d_log2_words = ((ctr >> 16) & 0xF) as usize;
        let cache_line_bytes = (4usize << d_log2_words).clamp(16, 256);
        Self { cache_line_bytes, l1d_bytes: 64 << 10, l2_bytes: 512 << 10,
               l3_bytes: 8 << 20, mshr_count: 10 }
    }

    fn fallback() -> Self {
        Self { cache_line_bytes: 64, l1d_bytes: 32 << 10,
               l2_bytes: 256 << 10, l3_bytes: 8 << 20, mshr_count: 8 }
    }
}

// ── StridePredictor ───────────────────────────────────────────────────────────

/// Tracks up to `STRIDE_STREAMS` independent access streams and predicts
/// the next address from a confirmed stride pattern.
///
/// Algorithm: each observed address is matched against `(last_addr, stride)`
/// pairs. If the delta matches the stored stride, confidence increments.
/// At `STRIDE_CONFIDENCE_THRESHOLD`, a prefetch address is emitted.
#[derive(Debug, Clone)]
struct StridePredictor {
    last_addr:  [usize; STRIDE_STREAMS],
    stride:     [isize; STRIDE_STREAMS],
    confidence: [u8;    STRIDE_STREAMS],
    evict_ptr:  usize, // round-robin eviction cursor
}

impl StridePredictor {
    const fn new() -> Self {
        Self { last_addr: [0; STRIDE_STREAMS], stride: [0; STRIDE_STREAMS],
               confidence: [0; STRIDE_STREAMS], evict_ptr: 0 }
    }

    /// Observe a load from `addr`. Returns `Some(prefetch_target)` when a
    /// confirmed stride is detected. `lookahead` = strides to predict ahead.
    #[inline(always)]
    fn observe(&mut self, addr: usize, lookahead: usize) -> Option<usize> {
        // Search existing streams for a match.
        for i in 0..STRIDE_STREAMS {
            if self.last_addr[i] == 0 { continue; }

            let delta = (addr as isize).wrapping_sub(self.last_addr[i] as isize);

            if delta == self.stride[i] && delta != 0 {
                // Stride confirmed — increase confidence, maybe emit prefetch.
                self.confidence[i] = self.confidence[i].saturating_add(1);
                self.last_addr[i]  = addr;
                return if self.confidence[i] >= STRIDE_CONFIDENCE_THRESHOLD {
                    let target = (addr as isize)
                        .wrapping_add(delta * lookahead as isize) as usize;
                    Some(target)
                } else {
                    None
                };
            }

            // Same region, different delta → update stride, reset confidence.
            if self.last_addr[i].abs_diff(addr) < 4096 {
                self.stride[i]     = delta;
                self.confidence[i] = 1;
                self.last_addr[i]  = addr;
                return None;
            }
        }

        // No matching stream → evict the oldest slot (round-robin).
        let slot = self.evict_ptr;
        self.evict_ptr = (self.evict_ptr + 1) % STRIDE_STREAMS;
        self.last_addr[slot]  = addr;
        self.stride[slot]     = 0;
        self.confidence[slot] = 0;
        None
    }

    /// Discard all stream state (call at function returns, indirect jumps).
    #[inline(always)]
    fn invalidate(&mut self) {
        self.last_addr  = [0; STRIDE_STREAMS];
        self.confidence = [0; STRIDE_STREAMS];
    }
}

// ── DistanceTable ─────────────────────────────────────────────────────────────

/// Prefetch lookahead distances per cache level, per stream type.
/// Stored as `u8` — distances > 255 are unrealistic given cache sizes.
#[derive(Debug, Clone, Copy)]
struct DistanceTable {
    insn_l1: u8, insn_l2: u8, insn_l3: u8,
    slot_l1: u8, slot_l2: u8, slot_l3: u8,
}

impl Default for DistanceTable {
    fn default() -> Self {
        Self { insn_l1: 8, insn_l2: 18, insn_l3: 36,
               slot_l1: 4, slot_l2: 10, slot_l3: 20 }
    }
}

impl DistanceTable {
    /// Recompute all distances from measured latencies and runtime signals.
    ///
    /// The core identity:
    ///   `prefetch_distance = memory_latency_cycles × IPC`
    ///
    /// so that by the time PC reaches `pc + distance`, the data issued at
    /// `pc` has had exactly enough cycles to arrive from the memory subsystem.
    ///
    /// * `ram_lat`        — measured RAM latency in cycles.
    /// * `l2_lat`         — measured L2-miss latency in cycles.
    /// * `ipc_x10`        — IPC × 10 (0 → assume 2.0 IPC).
    /// * `branch_density` — branches per 8 instructions (more = tighter window
    ///                      to avoid crossing mispredicted branch targets).
    /// * `miss_rate`      — misses per 1 000 ops; higher = wider window.
    fn calibrate(&mut self, ram_lat: u32, l2_lat: u32, ipc_x10: u32,
                 branch_density: u8, miss_rate: u32) {
        let ipc = ipc_x10.max(5); // floor at 0.5 IPC

        let lat_to_dist = |lat: u32| -> u8 {
            ((lat * ipc / 10).clamp(1, 255)) as u8
        };

        let l1_lat = (l2_lat / 4).max(4);
        let l3_lat = (ram_lat * 3 / 4).max(l2_lat + 10);

        // Branch pressure scale factor (in tenths of unity).
        let bscale: u32 = if branch_density <= 2 { 14 }
                          else if branch_density <= 5 { 10 }
                          else { 6 };

        // Miss-rate bonus: sustained cache pressure → be more aggressive.
        let miss_bonus = (miss_rate.min(200) / 25) as u8; // 0–8

        let scale = |base: u8| -> u8 {
            ((base as u32 * bscale / 10) as u8).saturating_add(miss_bonus).max(1)
        };

        let i_l1 = lat_to_dist(l1_lat);
        let i_l2 = lat_to_dist(l2_lat).max(i_l1 + 4);
        let i_l3 = lat_to_dist(l3_lat).max(i_l2 + 8);

        self.insn_l1 = scale(i_l1);
        self.insn_l2 = scale(i_l2).max(self.insn_l1 + 2);
        self.insn_l3 = scale(i_l3).max(self.insn_l2 + 4);

        // Slots are accessed more densely → use ~half the instruction distances.
        self.slot_l1 = (self.insn_l1 / 2).max(1);
        self.slot_l2 = (self.insn_l2 / 2).max(self.slot_l1 + 2);
        self.slot_l3 = (self.insn_l3 / 2).max(self.slot_l2 + 3);
    }
}

// ── LatencyCalibrator ─────────────────────────────────────────────────────────

/// Measures true main-memory read latency using a pointer-chase technique.
///
/// The chain is a Lehmer-shuffled permutation where `chain[i]` is the index
/// of the next element to load, creating a serial dependency that forces the
/// CPU to wait for each load before issuing the next. We CLFLUSH the entire
/// chain before timing to guarantee cold-cache conditions.
///
/// The random permutation defeats the hardware stride prefetcher — if it were
/// sequential, the HW prefetcher would service it from L1 and we'd measure
/// L1 latency, not RAM latency.
struct LatencyCalibrator {
    /// 64-entry Lehmer-shuffled pointer-chase (stored as u32 indices,
    /// not raw pointers, so the struct is self-contained and moveable).
    chain: [u32; 64],
    /// Best measured RAM latency from the last calibration (cycles/access).
    pub cached_ram_latency: u32,
    /// Inferred L2 latency (cycles/access).
    pub cached_l2_latency: u32,
    /// TSC value at last calibration.
    last_tsc: u64,
    /// Calibration run count (for diagnostics).
    pub calibration_count: u32,
}

impl LatencyCalibrator {
    const fn new() -> Self {
        Self { chain: [0u32; 64], cached_ram_latency: DEFAULT_RAM_LATENCY,
               cached_l2_latency: DEFAULT_L2_LATENCY, last_tsc: 0,
               calibration_count: 0 }
    }

    /// Build a randomised pointer-chase permutation through the chain.
    /// Uses a simple LCG to produce non-sequential indices, ensuring the
    /// hardware prefetcher cannot predict the access sequence.
    fn init_chain(&mut self) {
        const N: usize = 64;
        let mut visited = [false; N];
        let mut cur: usize = 7; // arbitrary start, != 0
        for _ in 0..(N - 1) {
            visited[cur] = true;
            let mut nxt = (cur.wrapping_mul(37).wrapping_add(17)) % N;
            while visited[nxt] { nxt = (nxt + 1) % N; } // linear probe
            self.chain[cur] = nxt as u32;
            cur = nxt;
        }
        self.chain[cur] = 7u32; // close the loop
    }

    /// Run 3 trials of the pointer-chase measurement and return the best
    /// `(ram_cycles_per_access, l2_cycles_per_access)`.
    #[cfg(target_arch = "x86_64")]
    fn measure(&self) -> (u32, u32) {
        use core::arch::x86_64::{_mm_clflush, _mm_lfence, _mm_mfence, _rdtsc};
        const STEPS: u64 = 32;
        let mut best = u64::MAX;

        for _ in 0u32..3 {
            unsafe {
                // Flush the entire chain from all cache levels.
                for entry in &self.chain {
                    _mm_clflush((entry as *const u32).cast::<u8>());
                }
                _mm_mfence(); // all flushes must retire before we start

                _mm_lfence(); // serialise: no later instruction may issue yet
                let t0 = _rdtsc();
                _mm_lfence();

                // Serial load chain — each address depends on the previous value,
                // forcing the CPU to wait for the full memory latency each time.
                let mut idx = 7usize;
                for _ in 0..STEPS {
                    // SAFETY: all indices are in [0, 64) by chain construction.
                    idx = *self.chain.get_unchecked(idx) as usize;
                    // black_box prevents the compiler from eliminating this load.
                    core::hint::black_box(idx);
                }

                _mm_lfence();
                let t1 = _rdtsc();

                let elapsed = t1.wrapping_sub(t0);
                if elapsed < best { best = elapsed; }
                let _ = core::hint::black_box(idx);
            }
        }

        let per_access = ((best / STEPS) as u32).clamp(10, 1200);
        let l2 = (per_access / 5).max(15); // L2 ≈ 1/5 of RAM on modern x86
        (per_access, l2)
    }

    #[cfg(target_arch = "aarch64")]
    fn measure(&self) -> (u32, u32) {
        const STEPS: u64 = 32;
        let mut best = u64::MAX;
        for _ in 0u32..3 {
            unsafe {
                for entry in &self.chain {
                    core::arch::asm!("dc civac, {p}",
                        p = in(reg) (entry as *const u32 as usize),
                        options(nostack, preserves_flags));
                }
                core::arch::asm!("dsb sy; isb", options(nostack, preserves_flags));
                let t0: u64;
                core::arch::asm!("mrs {t}, cntvct_el0", t = out(reg) t0,
                                 options(nostack, readonly, preserves_flags));
                let mut idx = 7usize;
                for _ in 0..STEPS {
                    idx = *self.chain.get_unchecked(idx) as usize;
                    core::hint::black_box(idx);
                }
                let t1: u64;
                core::arch::asm!("mrs {t}, cntvct_el0", t = out(reg) t1,
                                 options(nostack, readonly, preserves_flags));
                let elapsed = t1.wrapping_sub(t0);
                if elapsed < best { best = elapsed; }
                let _ = core::hint::black_box(idx);
            }
        }
        let per_access = ((best / STEPS) as u32).clamp(10, 1200);
        (per_access, (per_access / 5).max(15))
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn measure(&self) -> (u32, u32) { (DEFAULT_RAM_LATENCY, DEFAULT_L2_LATENCY) }

    #[inline(always)]
    fn tsc() -> u64 {
        #[cfg(target_arch = "x86_64")]
        unsafe { core::arch::x86_64::_rdtsc() }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let v: u64;
            core::arch::asm!("mrs {v}, cntvct_el0", v = out(reg) v,
                             options(nostack, readonly, preserves_flags));
            v
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        0u64
    }

    #[inline(always)]
    fn stale(&self) -> bool {
        Self::tsc().wrapping_sub(self.last_tsc) >= RECALIBRATE_INTERVAL
    }

    #[cold]
    fn recalibrate(&mut self) {
        let (ram, l2) = self.measure();
        self.cached_ram_latency = ram;
        self.cached_l2_latency  = l2;
        self.last_tsc            = Self::tsc();
        self.calibration_count  += 1;
    }
}

// ── PrefetchEngine ────────────────────────────────────────────────────────────

/// The main prefetch engine — embed exactly one in your interpreter struct.
///
/// # Memory layout
///
/// `repr(align(64))` guarantees the struct starts on a fresh cache line.
/// The hottest fields (distance table + budget) are placed first, ensuring
/// they land in the same 64-byte line as the struct base address.
///
/// # Usage pattern
///
/// ```rust,ignore
/// let mut engine = PrefetchEngine::new();
///
/// loop {
///     // Once per epoch (~256 instructions):
///     engine.tick(measured_branch_density);
///
///     // In the hot dispatch loop:
///     engine.prefetch_dual(
///         insns.as_ptr(), pc,   insns.len(),
///         slots.as_ptr(), sp,   slots.len(),
///     );
///
///     // From a perf-counter callback (any thread):
///     engine.record_miss_rate(hw_misses_per_kop);
///     engine.record_ipc(measured_ipc_x10);
/// }
/// ```
#[repr(align(64))]
pub struct PrefetchEngine {
    // ── Cache line 0: hot path ───────────────────────────────────────────────
    dist:            DistanceTable, // 6 B
    throttle_budget: u32,           // 4 B — remaining hints for this epoch
    epoch_counter:   u32,           // 4 B
    branch_density:  u8,            // 1 B
    _pad0:           [u8; 49],      // align remainder to 64 B total

    // ── Cache lines 1-2: warm — stride predictor ~88 B ───────────────────────
    stride: StridePredictor,

    // ── Cache lines 3+: cold — written infrequently ──────────────────────────
    topology:   CpuTopology,
    calibrator: LatencyCalibrator,

    // Feedback channels — can be written from any thread via perf callbacks.
    // Only read at epoch boundaries (every ~4096 instructions).
    miss_rate: AtomicU32, // cache misses per 1 000 ops, 0–255
    ipc_x10:   AtomicU32, // IPC × 10 (e.g. 25 = 2.5 IPC)
}

unsafe impl Send for PrefetchEngine {}
unsafe impl Sync for PrefetchEngine {}

impl Default for PrefetchEngine {
    fn default() -> Self { Self::new() }
}

impl PrefetchEngine {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Create and initialise a new engine, detecting CPU topology and running
    /// an initial latency calibration pass.
    pub fn new() -> Self {
        let topology = CpuTopology::detect();
        let mut engine = Self {
            dist:            DistanceTable::default(),
            throttle_budget: EPOCH_BUDGET,
            epoch_counter:   0,
            branch_density:  3,
            _pad0:           [0u8; 49],
            stride:          StridePredictor::new(),
            topology,
            calibrator:      LatencyCalibrator::new(),
            miss_rate:       AtomicU32::new(0),
            ipc_x10:         AtomicU32::new(20),
        };
        engine.calibrator.init_chain();
        engine.calibrator.recalibrate();
        engine.retune();
        engine
    }

    // ── Feedback setters — callable from any thread ───────────────────────────

    /// Report cache misses per 1 000 operations.
    /// 0 = perfect; 255 = severe miss rate → engine will widen prefetch windows.
    #[inline(always)]
    pub fn record_miss_rate(&self, misses_per_kop: u8) {
        self.miss_rate.store(u32::from(misses_per_kop), Ordering::Relaxed);
    }

    /// Report measured IPC × 10 (e.g. 25 = 2.5 IPC).
    /// Used to derive element distances from measured cycle latencies.
    #[inline(always)]
    pub fn record_ipc(&self, ipc_x10: u8) {
        self.ipc_x10.store(u32::from(ipc_x10.max(1)), Ordering::Relaxed);
    }

    // ── Epoch management ──────────────────────────────────────────────────────

    /// Advance one epoch (~256 VM instructions).
    ///
    /// * Refills the bandwidth budget.
    /// * Re-tunes distances every 16 epochs (~4 096 instructions).
    /// * May trigger a latency re-calibration if the TSC interval elapsed.
    #[inline(always)]
    pub fn tick(&mut self, branch_density: u8) {
        self.throttle_budget = EPOCH_BUDGET;
        self.branch_density  = branch_density;
        self.epoch_counter   = self.epoch_counter.wrapping_add(1);
        if self.epoch_counter & 0xF == 0 {
            self.cold_tick();
        }
    }

    /// Notify the engine that a function return or indirect jump occurred.
    /// Clears stride predictor state to avoid false predictions across calls.
    #[inline(always)]
    pub fn notify_control_transfer(&mut self) {
        self.stride.invalidate();
    }

    #[cold]
    #[inline(never)]
    fn cold_tick(&mut self) {
        if self.calibrator.stale() { self.calibrator.recalibrate(); }
        self.retune();
    }

    fn retune(&mut self) {
        self.dist.calibrate(
            self.calibrator.cached_ram_latency,
            self.calibrator.cached_l2_latency,
            self.ipc_x10.load(Ordering::Relaxed),
            self.branch_density,
            self.miss_rate.load(Ordering::Relaxed),
        );
    }

    // ── Internal throttle ─────────────────────────────────────────────────────

    /// Attempt to claim `n` hint budget slots. Returns `true` on success.
    #[inline(always)]
    fn claim(&mut self, n: u32) -> bool {
        if self.throttle_budget >= n { self.throttle_budget -= n; true } else { false }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public prefetch API
    // ─────────────────────────────────────────────────────────────────────────

    /// **Instruction-stream prefetch** — three levels simultaneously.
    ///
    /// Issues L1/L2/L3 hints for `pc + l1`, `pc + l2`, `pc + l3`.
    /// All three fire as independent hardware operations; the CPU's
    /// out-of-order miss queue handles them in parallel at ≈0 dispatch cost.
    #[inline(always)]
    pub fn prefetch_insn<T>(&mut self, base: *const T, pc: usize, len: usize) {
        if !self.claim(3) { return; }
        let (l1, l2, l3) = (
            pc.saturating_add(self.dist.insn_l1 as usize),
            pc.saturating_add(self.dist.insn_l2 as usize),
            pc.saturating_add(self.dist.insn_l3 as usize),
        );
        // SAFETY: `n < len` guarantees `base.add(n)` is within the slice.
        unsafe {
            if l1 < len { prefetch_l1(base.add(l1).cast()) }
            if l2 < len { prefetch_l2(base.add(l2).cast()) }
            if l3 < len { prefetch_l3(base.add(l3).cast()) }
        }
    }

    /// **Slot prefetch** — value-stack / register file read (L1 + L2).
    #[inline(always)]
    pub fn prefetch_slot<T>(&mut self, base: *const T, slot: usize, len: usize) {
        if !self.claim(2) { return; }
        let (l1, l2) = (
            slot.saturating_add(self.dist.slot_l1 as usize),
            slot.saturating_add(self.dist.slot_l2 as usize),
        );
        unsafe {
            if l1 < len { prefetch_l1(base.add(l1).cast()) }
            if l2 < len { prefetch_l2(base.add(l2).cast()) }
        }
    }

    /// **Write-intent slot prefetch** — upcoming store path.
    ///
    /// Issues `PREFETCHW` (x86_64) / `PRFM PSTL1KEEP` (AArch64) so the
    /// cache line is acquired in Modified state before the write, eliminating
    /// the Read-For-Ownership round-trip to L3 / memory controller.
    #[inline(always)]
    pub fn prefetch_slot_write<T>(&mut self, base: *mut T, slot: usize, len: usize) {
        if !self.claim(1) { return; }
        let l1 = slot.saturating_add(self.dist.slot_l1 as usize);
        if l1 < len { unsafe { prefetch_write_l1(base.add(l1).cast()) } }
    }

    /// **Copy prefetch** — warm source (read) and destination (write) together.
    ///
    /// Use before `MOV slot_a → slot_b` to pre-acquire both endpoints.
    #[inline(always)]
    pub fn prefetch_copy<T>(
        &mut self,
        src: *const T, src_slot: usize, src_len: usize,
        dst: *mut   T, dst_slot: usize, dst_len: usize,
    ) {
        if !self.claim(2) { return; }
        let s = src_slot.saturating_add(self.dist.slot_l1 as usize);
        let d = dst_slot.saturating_add(self.dist.slot_l1 as usize);
        unsafe {
            if s < src_len { prefetch_l1(src.add(s).cast()) }
            if d < dst_len { prefetch_write_l1(dst.add(d).cast()) }
        }
    }

    /// **Dual-stream prefetch** — instruction + operand stream in one call.
    ///
    /// Issues up to 5 parallel cache hints. Prefer this over separate
    /// `prefetch_insn` + `prefetch_slot` calls in the main dispatch loop
    /// to halve the call overhead and let all hints retire together.
    #[inline(always)]
    pub fn prefetch_dual<I, S>(
        &mut self,
        insn_base: *const I, pc:   usize, insn_len: usize,
        slot_base: *const S, slot: usize, slot_len: usize,
    ) {
        if !self.claim(5) { return; }
        let (il1, il2, il3) = (
            pc.saturating_add(self.dist.insn_l1 as usize),
            pc.saturating_add(self.dist.insn_l2 as usize),
            pc.saturating_add(self.dist.insn_l3 as usize),
        );
        let (sl1, sl2) = (
            slot.saturating_add(self.dist.slot_l1 as usize),
            slot.saturating_add(self.dist.slot_l2 as usize),
        );
        unsafe {
            if il1 < insn_len { prefetch_l1(insn_base.add(il1).cast()) }
            if il2 < insn_len { prefetch_l2(insn_base.add(il2).cast()) }
            if il3 < insn_len { prefetch_l3(insn_base.add(il3).cast()) }
            if sl1 < slot_len { prefetch_l1(slot_base.add(sl1).cast()) }
            if sl2 < slot_len { prefetch_l2(slot_base.add(sl2).cast()) }
        }
    }

    /// **Triple-stream prefetch** — instruction + two independent slot streams.
    ///
    /// Use when an opcode reads two separate register files
    /// (e.g. general-purpose + floating-point registers).
    #[inline(always)]
    pub fn prefetch_triple<I, A, B>(
        &mut self,
        insn_base: *const I, pc:    usize, insn_len: usize,
        a_base:    *const A, a_idx: usize, a_len:    usize,
        b_base:    *const B, b_idx: usize, b_len:    usize,
    ) {
        if !self.claim(6) { return; }
        let (il1, il2) = (
            pc.saturating_add(self.dist.insn_l1 as usize),
            pc.saturating_add(self.dist.insn_l2 as usize),
        );
        let (al1, al2) = (
            a_idx.saturating_add(self.dist.slot_l1 as usize),
            a_idx.saturating_add(self.dist.slot_l2 as usize),
        );
        let (bl1, bl2) = (
            b_idx.saturating_add(self.dist.slot_l1 as usize),
            b_idx.saturating_add(self.dist.slot_l2 as usize),
        );
        unsafe {
            if il1 < insn_len { prefetch_l1(insn_base.add(il1).cast()) }
            if il2 < insn_len { prefetch_l2(insn_base.add(il2).cast()) }
            if al1 < a_len   { prefetch_l1(a_base.add(al1).cast())    }
            if al2 < a_len   { prefetch_l2(a_base.add(al2).cast())    }
            if bl1 < b_len   { prefetch_l1(b_base.add(bl1).cast())    }
            if bl2 < b_len   { prefetch_l2(b_base.add(bl2).cast())    }
        }
    }

    /// **Branch-aware prefetch** — warm both sides of a conditional branch.
    ///
    /// The more-likely target gets an L1 hint; the less-likely gets L2.
    /// This mirrors how a real branch predictor allocates fetch bandwidth —
    /// both paths are warmed but the hot path is prioritised.
    ///
    /// `taken_prob` — 0–255; 128 = 50/50, 255 = always taken.
    #[inline(always)]
    pub fn prefetch_branch<T>(
        &mut self,
        taken:      *const T,
        not_taken:  *const T,
        taken_prob: u8,
    ) {
        if !self.claim(2) { return; }
        unsafe {
            if taken_prob >= 128 {
                prefetch_l1(taken.cast());
                prefetch_l2(not_taken.cast());
            } else {
                prefetch_l2(taken.cast());
                prefetch_l1(not_taken.cast());
            }
        }
    }

    /// **Stride-predicted slot prefetch** — detects access patterns and
    /// emits speculative prefetches when a stride is confirmed.
    ///
    /// Returns the predicted target address (`Some`) or `None` if confidence
    /// hasn't yet reached the threshold.
    #[inline(always)]
    pub fn prefetch_slot_strided<T>(
        &mut self,
        base: *const T,
        slot: usize,
        len:  usize,
    ) -> Option<usize> {
        let addr      = unsafe { base.add(slot) as usize };
        let lookahead = self.dist.slot_l1 as usize;
        let predicted = self.stride.observe(addr, lookahead);

        if let Some(target) = predicted {
            let base_addr = base as usize;
            let end_addr  = unsafe { base.add(len) as usize };
            if target >= base_addr && target < end_addr && self.claim(1) {
                unsafe { prefetch_l1(target as *const u8) }
            }
        }
        predicted
    }

    /// **Loop body warming** — brings an entire instruction range into L2/L3.
    ///
    /// Call when the VM detects a backward branch (entering a hot loop).
    /// Issues one L2 hint per instruction in `[start, end)`, plus L3 hints
    /// for large bodies to build a deep prefetch pipeline.
    #[inline(always)]
    pub fn warm_loop_body<T>(&mut self, base: *const T, start: usize, end: usize, len: usize) {
        let end = end.min(len);
        if start >= end { return; }
        let count = (end - start) as u32;
        let hints = if count > 32 { count * 2 } else { count };
        if !self.claim(hints) { return; }
        for pc in start..end {
            unsafe {
                let ptr = base.add(pc).cast::<u8>();
                prefetch_l2(ptr);
                if count > 32 { prefetch_l3(ptr); }
            }
        }
    }

    /// **Region warming** — stride through a block issuing L2 hints per
    /// cache line.
    ///
    /// Use at function entry to pre-warm the local register frame before the
    /// dispatch loop. Clamped to 16 KiB to bound cost.
    #[inline(always)]
    pub fn warm_region<T>(&mut self, ptr: *const T, bytes: usize) {
        let clamped = bytes.min(16 << 10);
        let stride  = self.topology.cache_line_bytes;
        let n_lines = ((clamped + stride - 1) / stride) as u32;
        if !self.claim(n_lines) { return; }
        let base = ptr.cast::<u8>();
        let mut offset = 0;
        while offset < clamped {
            unsafe { prefetch_l2(base.add(offset)) }
            offset += stride;
        }
    }

    /// **Write-combining region warm** — issue write-intent prefetches for
    /// an entire block.
    ///
    /// Use before zeroing or bulk-initialising a slot array so every line is
    /// acquired in Modified state before the stores start. Capped at 4 KiB.
    #[inline(always)]
    pub fn prefetch_write_region<T>(&mut self, ptr: *mut T, bytes: usize) {
        let clamped = bytes.min(4 << 10);
        let stride  = self.topology.cache_line_bytes;
        let n_lines = ((clamped + stride - 1) / stride) as u32;
        if !self.claim(n_lines) { return; }
        let base = ptr.cast::<u8>();
        let mut offset = 0;
        while offset < clamped {
            unsafe { prefetch_write_l1(base.add(offset)) }
            offset += stride;
        }
    }

    /// **Non-temporal / streaming prefetch** — brings data into a fill buffer
    /// without allocating an L1/L2 cache line.
    ///
    /// Use for large read-once data (constant pools, debug tables, etc.)
    /// that would otherwise pollute the instruction and slot caches.
    #[inline(always)]
    pub fn prefetch_stream<T>(&mut self, base: *const T, idx: usize, len: usize) {
        let ahead = idx.saturating_add(self.dist.insn_l3 as usize);
        if ahead < len && self.claim(1) {
            unsafe { prefetch_nta(base.add(ahead).cast()) }
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    #[inline(always)] pub fn topology(&self)            -> &CpuTopology { &self.topology }
    #[inline(always)] pub fn insn_distance_l1(&self)   -> usize { self.dist.insn_l1 as usize }
    #[inline(always)] pub fn insn_distance_l2(&self)   -> usize { self.dist.insn_l2 as usize }
    #[inline(always)] pub fn insn_distance_l3(&self)   -> usize { self.dist.insn_l3 as usize }
    #[inline(always)] pub fn slot_distance_l1(&self)   -> usize { self.dist.slot_l1 as usize }
    #[inline(always)] pub fn measured_ram_latency(&self)-> u32  { self.calibrator.cached_ram_latency }
    #[inline(always)] pub fn calibration_count(&self)  -> u32  { self.calibrator.calibration_count }
    #[inline(always)] pub fn remaining_budget(&self)   -> u32  { self.throttle_budget }
}

// ── ISA-level prefetch primitives ─────────────────────────────────────────────
//
// Each function maps to a single instruction on supported arches and to a
// zero-cost no-op elsewhere. All are `unsafe` (raw pointer arithmetic) and
// `#[inline(always)]` (they must never add call overhead).
//
// x86_64 mapping:
//   prefetch_l1       → PREFETCHT0   (T0)  → L1d
//   prefetch_l2       → PREFETCHT1   (T1)  → L2
//   prefetch_l3       → PREFETCHT2   (T2)  → L3
//   prefetch_write_l1 → PREFETCHW    (ET0) → L1d, exclusive (write-intent)
//   prefetch_nta      → PREFETCHNTA  (NTA) → fill buffer, bypasses L1/L2
//
// AArch64 mapping:
//   prefetch_l1       → PRFM PLDL1KEEP  → L1
//   prefetch_l2       → PRFM PLDL2KEEP  → L2
//   prefetch_l3       → PRFM PLDL3KEEP  → L3
//   prefetch_write_l1 → PRFM PSTL1KEEP  → L1, store-intent
//   prefetch_nta      → PRFM PLDL3STRM  → streaming (non-allocating)

#[inline(always)]
unsafe fn prefetch_l1(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    { use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
      _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T0); }
    #[cfg(target_arch = "aarch64")]
    core::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) ptr,
                     options(nostack, readonly, preserves_flags));
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { let _ = ptr; }
}

#[inline(always)]
unsafe fn prefetch_l2(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    { use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T1};
      _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T1); }
    #[cfg(target_arch = "aarch64")]
    core::arch::asm!("prfm pldl2keep, [{p}]", p = in(reg) ptr,
                     options(nostack, readonly, preserves_flags));
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { let _ = ptr; }
}

#[inline(always)]
unsafe fn prefetch_l3(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    { use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T2};
      _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_T2); }
    #[cfg(target_arch = "aarch64")]
    core::arch::asm!("prfm pldl3keep, [{p}]", p = in(reg) ptr,
                     options(nostack, readonly, preserves_flags));
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { let _ = ptr; }
}

#[inline(always)]
unsafe fn prefetch_write_l1(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    { use core::arch::x86_64::_mm_prefetch;
      _mm_prefetch(ptr.cast::<i8>(), MM_HINT_ET0); }
    #[cfg(target_arch = "aarch64")]
    core::arch::asm!("prfm pstl1keep, [{p}]", p = in(reg) ptr,
                     options(nostack, preserves_flags));
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { let _ = ptr; }
}

#[inline(always)]
unsafe fn prefetch_nta(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    { use core::arch::x86_64::{_mm_prefetch, _MM_HINT_NTA};
      _mm_prefetch(ptr.cast::<i8>(), _MM_HINT_NTA); }
    #[cfg(target_arch = "aarch64")]
    core::arch::asm!("prfm pldl3strm, [{p}]", p = in(reg) ptr,
                     options(nostack, readonly, preserves_flags));
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { let _ = ptr; }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> PrefetchEngine { PrefetchEngine::new() }

    // ── Topology ──────────────────────────────────────────────────────────────

    #[test]
    fn topology_sane_values() {
        let t = CpuTopology::detect();
        assert!(t.cache_line_bytes >= 16 && t.cache_line_bytes <= 256,
                "cache_line_bytes={}", t.cache_line_bytes);
        assert!(t.l1d_bytes >= 4 << 10,      "l1d too small");
        assert!(t.l2_bytes  >= t.l1d_bytes,   "l2 < l1");
        assert!(t.l3_bytes  >= t.l2_bytes,    "l3 < l2");
        assert!(t.mshr_count >= 4 && t.mshr_count <= 128);
    }

    // ── Distance calibration ──────────────────────────────────────────────────

    #[test]
    fn distance_ordering_invariant() {
        let e = engine();
        assert!(e.dist.insn_l1 < e.dist.insn_l2, "insn L1 >= L2");
        assert!(e.dist.insn_l2 < e.dist.insn_l3, "insn L2 >= L3");
        assert!(e.dist.slot_l1 < e.dist.slot_l2, "slot L1 >= L2");
        assert!(e.dist.slot_l2 < e.dist.slot_l3, "slot L2 >= L3");
        assert!(e.dist.insn_l1 >= 1);
    }

    #[test]
    fn calibrate_extreme_inputs() {
        let mut d = DistanceTable::default();
        // Very fast memory, high IPC, few branches.
        d.calibrate(50, 12, 40, 0, 0);
        assert!(d.insn_l1 >= 1 && d.insn_l3 <= 255);
        assert!(d.insn_l1 < d.insn_l2 && d.insn_l2 < d.insn_l3);
        // Very slow memory, low IPC, many branches, many misses.
        d.calibrate(800, 200, 5, 255, 200);
        assert!(d.insn_l1 >= 1 && d.insn_l3 <= 255);
        assert!(d.insn_l1 < d.insn_l2 && d.insn_l2 < d.insn_l3);
    }

    #[test]
    fn high_miss_rate_widens_window() {
        let mut lo = DistanceTable::default();
        let mut hi = DistanceTable::default();
        lo.calibrate(200, 40, 20, 3, 0);
        hi.calibrate(200, 40, 20, 3, 200);
        assert!(hi.insn_l1 >= lo.insn_l1, "high miss rate must widen window");
    }

    // ── Throttle ──────────────────────────────────────────────────────────────

    #[test]
    fn budget_exhausts_exactly() {
        let mut e = engine();
        let mut granted = 0u32;
        for _ in 0..10_000 { if e.claim(1) { granted += 1; } }
        assert_eq!(granted, EPOCH_BUDGET);
    }

    #[test]
    fn tick_refills_budget() {
        let mut e = engine();
        for _ in 0..10_000 { e.claim(1); }
        assert!(!e.claim(1), "budget not exhausted");
        e.tick(3);
        assert!(e.claim(1), "budget not refilled after tick");
    }

    // ── Latency calibration ───────────────────────────────────────────────────

    #[test]
    fn chain_is_valid_permutation() {
        let mut c = LatencyCalibrator::new();
        c.init_chain();
        let mut visited = [false; 64];
        let mut idx = 7usize;
        for step in 0..64 {
            assert!(!visited[idx], "cycle detected at step {step}");
            visited[idx] = true;
            idx = c.chain[idx] as usize;
        }
        assert_eq!(idx, 7, "chain does not close at start");
        assert!(visited.iter().all(|&v| v), "not all 64 entries reachable");
    }

    #[test]
    fn calibration_nonzero() {
        let mut c = LatencyCalibrator::new();
        c.init_chain();
        let (ram, l2) = c.measure();
        assert!(ram > 0, "RAM latency is zero");
        assert!(l2  > 0, "L2 latency is zero");
        assert!(ram >= l2, "RAM latency < L2 (impossible)");
    }

    // ── Stride predictor ──────────────────────────────────────────────────────

    #[test]
    fn stride_detects_unit_stride() {
        let mut sp = StridePredictor::new();
        for i in 0usize..12 { sp.observe(i * 8, 4); }
        assert!(sp.observe(12 * 8, 4).is_some(), "unit stride not detected");
    }

    #[test]
    fn stride_no_predict_before_threshold() {
        let mut sp = StridePredictor::new();
        // Feed exactly threshold-1 confirmations.
        for i in 0usize..(STRIDE_CONFIDENCE_THRESHOLD as usize) {
            let r = sp.observe(i * 8, 4);
            assert!(r.is_none(), "predicted before threshold at step {i}");
        }
    }

    #[test]
    fn stride_invalidate_resets_state() {
        let mut sp = StridePredictor::new();
        for i in 0usize..10 { sp.observe(i * 8, 4); }
        sp.invalidate();
        assert!(sp.last_addr.iter().all(|&a| a == 0));
        assert!(sp.confidence.iter().all(|&c| c == 0));
    }

    // ── Safety: out-of-bounds ─────────────────────────────────────────────────

    #[test]
    fn oob_prefetch_is_safe() {
        let mut e = engine();
        let tiny = [0u8; 2];
        // All distances > 2 — must never read past the end of the slice.
        e.prefetch_insn(tiny.as_ptr(), 0, tiny.len());
        e.prefetch_slot(tiny.as_ptr(), 0, tiny.len());
        e.prefetch_stream(tiny.as_ptr(), 0, tiny.len());
    }

    // ── Smoke tests for every public method ──────────────────────────────────

    #[test]
    fn dual_stream_smoke() {
        let mut e = engine();
        let insns = [0u32; 128];
        let slots  = [0u64; 64];
        e.prefetch_dual(insns.as_ptr(), 0, insns.len(), slots.as_ptr(), 0, slots.len());
    }

    #[test]
    fn triple_stream_smoke() {
        let mut e = engine();
        let insns = [0u32; 128];
        let gp    = [0u64; 64];
        let fp    = [0f64; 64];
        e.prefetch_triple(insns.as_ptr(), 0, insns.len(),
                          gp.as_ptr(), 0, gp.len(), fp.as_ptr(), 0, fp.len());
    }

    #[test]
    fn branch_prefetch_smoke() {
        let mut e = engine();
        let insns = [0u32; 64];
        e.prefetch_branch(insns.as_ptr(), unsafe { insns.as_ptr().add(4) }, 220);
        e.prefetch_branch(insns.as_ptr(), unsafe { insns.as_ptr().add(4) }, 30);
    }

    #[test]
    fn warm_loop_body_smoke() {
        let mut e = engine();
        let insns = [0u32; 256];
        e.warm_loop_body(insns.as_ptr(), 10, 50, insns.len());
        e.warm_loop_body(insns.as_ptr(), 50, 10, insns.len()); // start >= end → no-op
    }

    #[test]
    fn warm_region_smoke() {
        let mut e = engine();
        let buf = vec![0u8; 32 << 10];
        e.warm_region(buf.as_ptr(), buf.len()); // clamped to 16 KiB internally
    }

    #[test]
    fn write_region_smoke() {
        let mut e = engine();
        let mut buf = vec![0u8; 8 << 10];
        e.prefetch_write_region(buf.as_mut_ptr(), buf.len());
    }

    #[test]
    fn feedback_setters_smoke() {
        let e = engine();
        e.record_miss_rate(0);
        e.record_miss_rate(255);
        e.record_ipc(1);
        e.record_ipc(40);
    }

    #[test]
    fn tick_does_not_panic_across_many_epochs() {
        let mut e = engine();
        for density in [0u8, 3, 10, 255] {
            for _ in 0..20 { e.tick(density); }
        }
    }

    #[test]
    fn notify_control_transfer_clears_stride() {
        let mut e = engine();
        let slots = [0u64; 64];
        for i in 0usize..12 { e.prefetch_slot_strided(slots.as_ptr(), i, slots.len()); }
        e.notify_control_transfer();
        assert!(e.stride.confidence.iter().all(|&c| c == 0));
    }

    // ── Layout guarantees ─────────────────────────────────────────────────────

    #[test]
    fn engine_is_cache_line_aligned() {
        let e = engine();
        let addr = &e as *const PrefetchEngine as usize;
        assert_eq!(addr % 64, 0, "address {addr:#x} not 64-byte aligned");
    }

    #[test]
    fn hot_fields_in_first_cache_line() {
        let e = engine();
        let base  = &e            as *const PrefetchEngine as usize;
        let field = &e.dist       as *const DistanceTable  as usize;
        assert!(field - base < 64, "DistanceTable is not in the first cache line");
        let field2 = &e.throttle_budget as *const u32 as usize;
        assert!(field2 - base < 64, "throttle_budget is not in the first cache line");
    }
}
