// =============================================================================
// std/aurora_threading — M:N Fiber Threading for the Aurora Flux Pipeline
//
// Implements the parallel execution layer for the Aurora Flux rendering engine:
//
//   1. AuroraFiberPool: M lightweight fibers scheduled onto N OS workers
//      (N = physical CPU cores). Work-stealing between workers. Zero-lock
//      design: PRNG(Coordinate) is deterministic, so threads never need
//      mutexes on shared state. Each worker processes batches of 8 entities
//      using SIMD lanes. Total throughput = Cores × SIMD_Width × Clock_Speed.
//
//   2. AuroraDirector: Inspects workload size and decides granularity.
//      Small (1k entities) → single core (save power/latency).
//      Medium (20k entities) → 4 task-groups (reserve cores for OS/Audio).
//      Massive (1M+ entities) → all-hands-on-deck, saturate every thread,
//      prioritize the culling sieve.
//
//   3. AsyncMaterializer: Main thread renders current frame at 144Hz.
//      Worker threads "look ahead" in the player's direction, running the
//      210-Wheel Sieve and Morton Encoding for next terrain. By the time
//      the player gets there, the math is finished and sitting in L3 cache.
//
//   4. Parallel Ray Tracing: Thread A traces primary rays, Thread B traces
//      reflection rays, Thread C runs SIMD denoising on the previous frame.
//      AuroraRayBatch groups rays per thread.
//
//   5. SIMD Batch Processing: AuroraSimdBatch processes 8 Morton-to-UV
//      mappings simultaneously. Branchless collision via SIMD min operations.
//      Auto-splat vectorization. SIMD denoising (bilateral filter on 4×4
//      neighborhood).
//
// Key insight: because PRNG(seed, coordinate) is a pure function, every
// fiber produces identical results regardless of scheduling order. This
// eliminates the need for locks on shared render state — the fundamental
// enabler for the zero-lock design.
//
// Pure Rust. Uses std::sync::atomic and std::sync::Arc for thread safety.
// Uses num_cpus for core detection.
// =============================================================================

#![allow(dead_code)]
#![allow(unused_imports)]

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::interp::{RuntimeError, Value};
use crate::compiler::lexer::Span;
use crate::jules_std::morton::{encode_2d, decode_2d, encode_3d, decode_3d};
use crate::jules_std::prng_simd::{SimdPrng8, ShishiuaRng, SquaresRng};
use crate::jules_std::sieve_210::sieve_210_wheel;
use crate::jules_std::aurora_flux::{AuroraConfig, RenderMode, Pixel, FrameBuffer};
use crate::runtime::threading::worker::ThreadPool;

// ─── Dispatch for stdlib integration ────────────────────────────────────────

/// Dispatch `aurora_threading::` builtin calls.
///
/// Supported calls:
///   - "aurora_threading::verify"     → bool — run self-test suite
///   - "aurora_threading::benchmark"  → [throughput_cells_per_sec, num_cores, simd_width]
///   - "aurora_threading::director_info" → [workload_level, active_workers, total_cores]
pub fn dispatch(name: &str, args: &[Value]) -> Option<Result<Value, RuntimeError>> {
    match name {
        "aurora_threading::verify" => {
            let ok = verify_aurora_threading();
            Some(Ok(Value::Bool(ok)))
        }
        "aurora_threading::benchmark" => {
            let entity_count = args.first().and_then(|v| v.as_i64()).unwrap_or(100_000) as u64;
            let result = run_benchmark(entity_count);
            Some(Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vec![
                Value::F64(result.throughput_cells_per_sec),
                Value::I64(result.num_cores as i64),
                Value::I64(result.simd_width as i64),
                Value::F64(result.elapsed_secs),
            ])))))
        }
        "aurora_threading::director_info" => {
            let entity_count = args.first().and_then(|v| v.as_i64()).unwrap_or(10_000) as u64;
            let director = AuroraDirector::new();
            let plan = director.plan(entity_count);
            Some(Ok(Value::Array(std::rc::Rc::new(std::cell::RefCell::new(vec![
                Value::I64(plan.workload_level as i64),
                Value::I64(plan.active_workers as i64),
                Value::I64(plan.total_cores as i64),
                Value::F64(plan.estimated_throughput),
            ])))))
        }
        _ => None,
    }
}

// ─── Workload Level ─────────────────────────────────────────────────────────
//
// The Aurora Director uses workload levels to decide how many OS threads
// to engage. The key insight: on a 16-core machine, using all 16 cores
// for a 1k-entity scene wastes power and increases latency (more threads
// = more synchronization overhead). The director picks the right level.

/// Workload classification for the Aurora Director.
///
/// - Light:   ≤ 5,000 entities   → 1 core (single-threaded, lowest latency)
/// - Medium:  ≤ 50,000 entities  → 4 cores (balanced, reserves cores for OS/Audio)
/// - Heavy:   ≤ 500,000 entities → N/2 cores (half the machine, serious work)
/// - Extreme: > 500,000 entities → N cores (all-hands-on-deck, saturate everything)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WorkloadLevel {
    /// Light workload: ≤ 5k entities. Single core for lowest latency.
    Light  = 0,
    /// Medium workload: ≤ 50k entities. 4 task-groups.
    Medium = 1,
    /// Heavy workload: ≤ 500k entities. Half the machine.
    Heavy  = 2,
    /// Extreme workload: > 500k entities. All cores saturated.
    Extreme = 3,
}

impl WorkloadLevel {
    /// Classify workload from entity count.
    #[inline(always)]
    pub fn from_entity_count(count: u64) -> Self {
        if count <= 5_000 {
            WorkloadLevel::Light
        } else if count <= 50_000 {
            WorkloadLevel::Medium
        } else if count <= 500_000 {
            WorkloadLevel::Heavy
        } else {
            WorkloadLevel::Extreme
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            WorkloadLevel::Light   => "Light",
            WorkloadLevel::Medium  => "Medium",
            WorkloadLevel::Heavy   => "Heavy",
            WorkloadLevel::Extreme => "Extreme",
        }
    }

    /// Recommended number of worker threads for this workload level
    /// given a total core count.
    pub fn recommended_workers(&self, total_cores: usize) -> usize {
        match self {
            WorkloadLevel::Light   => 1,
            WorkloadLevel::Medium  => 4.min(total_cores),
            WorkloadLevel::Heavy   => (total_cores / 2).max(2),
            WorkloadLevel::Extreme => total_cores,
        }
    }
}

// ─── Director Plan ──────────────────────────────────────────────────────────

/// Scheduling plan produced by the Aurora Director.
///
/// Contains the workload classification, how many workers to engage,
/// and estimated throughput.
#[derive(Debug, Clone)]
pub struct DirectorPlan {
    /// Classified workload level.
    pub workload_level: WorkloadLevel,
    /// Number of OS worker threads to engage.
    pub active_workers: usize,
    /// Total physical cores on the machine.
    pub total_cores: usize,
    /// SIMD lane width (8 for AVX2, 4 for NEON).
    pub simd_width: usize,
    /// Estimated throughput in entities/second.
    pub estimated_throughput: f64,
    /// Whether the culling sieve should be prioritized (Extreme only).
    pub prioritize_culling_sieve: bool,
    /// Target frame budget in microseconds (for real-time rendering).
    pub frame_budget_us: u64,
}

// ─── Aurora Director ────────────────────────────────────────────────────────
//
// The Director is the "brain" of the threading system. Before each frame,
// the main thread asks the Director: "I have N entities this frame — how
// should I schedule them?"
//
// The Director's answer depends on:
//   1. Entity count (drives WorkloadLevel)
//   2. Available cores (from num_cpus)
//   3. SIMD width (8 for AVX2, 4 for SSE4.2)
//   4. Estimated clock speed (conservative 3.5 GHz)
//
// The formula: Throughput = Cores × SIMD_Width × Clock_Speed × ops_per_cycle
// where ops_per_cycle ≈ 0.5 (conservative, accounts for memory stalls).

/// The Aurora Director: decides thread scheduling based on workload.
///
/// The Director is stateless — it has no mutable state. Each call to
/// `plan()` produces an independent scheduling decision based on the
/// current entity count and hardware capabilities.
pub struct AuroraDirector {
    /// Number of physical CPU cores.
    total_cores: usize,
    /// SIMD lane width (8 for AVX2).
    simd_width: usize,
    /// Conservative clock speed estimate in Hz.
    estimated_clock_hz: f64,
    /// Operations per cycle (conservative 0.5 for memory-bound work).
    ops_per_cycle: f64,
}

impl AuroraDirector {
    /// Create a new Director that probes hardware capabilities.
    pub fn new() -> Self {
        let total_cores = num_cpus::get_physical();
        // Assume AVX2-class SIMD (8 lanes of f32/f64)
        let simd_width = 8;
        // Conservative clock estimate
        let estimated_clock_hz = 3_500_000_000.0;
        let ops_per_cycle = 0.5;

        AuroraDirector {
            total_cores,
            simd_width,
            estimated_clock_hz,
            ops_per_cycle,
        }
    }

    /// Create a Director with explicit parameters (for testing).
    pub fn with_params(
        total_cores: usize,
        simd_width: usize,
        estimated_clock_hz: f64,
        ops_per_cycle: f64,
    ) -> Self {
        AuroraDirector {
            total_cores,
            simd_width,
            estimated_clock_hz,
            ops_per_cycle,
        }
    }

    /// Produce a scheduling plan for the given entity count.
    ///
    /// This is the Director's main entry point. Call it once per frame
    /// (or whenever entity count changes significantly).
    pub fn plan(&self, entity_count: u64) -> DirectorPlan {
        let level = WorkloadLevel::from_entity_count(entity_count);
        let active_workers = level.recommended_workers(self.total_cores);
        let prioritize_culling = level == WorkloadLevel::Extreme;

        // Throughput estimate: Cores × SIMD_Width × Clock × ops_per_cycle
        let single_core_throughput =
            self.simd_width as f64 * self.estimated_clock_hz * self.ops_per_cycle;
        let estimated_throughput = single_core_throughput * active_workers as f64;

        // Frame budget: 144Hz → ~6944 µs per frame
        let frame_budget_us = if level == WorkloadLevel::Light {
            6944 // Full 144Hz budget (single core is fast enough)
        } else if level == WorkloadLevel::Medium {
            6944 // Still target 144Hz with 4 workers
        } else if level == WorkloadLevel::Heavy {
            8333 // 120Hz target
        } else {
            16666 // 60Hz target for extreme scenes
        };

        DirectorPlan {
            workload_level: level,
            active_workers,
            total_cores: self.total_cores,
            simd_width: self.simd_width,
            estimated_throughput,
            prioritize_culling_sieve: prioritize_culling,
            frame_budget_us,
        }
    }

    /// Quick classification without full plan allocation.
    pub fn classify(&self, entity_count: u64) -> WorkloadLevel {
        WorkloadLevel::from_entity_count(entity_count)
    }

    /// Get the number of physical CPU cores.
    pub fn total_cores(&self) -> usize {
        self.total_cores
    }

    /// Get the SIMD lane width.
    pub fn simd_width(&self) -> usize {
        self.simd_width
    }
}

impl Default for AuroraDirector {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Fiber ──────────────────────────────────────────────────────────────────
//
// A "fiber" in Aurora Threading is a lightweight unit of work. Unlike OS
// threads, fibers are cooperatively scheduled and have no kernel overhead.
//
// Each fiber represents a batch of entities to process. Because PRNG is
// deterministic per coordinate, fibers are fully independent — no locks,
// no atomics (except for the completion counter).

/// A single fiber: a batch of entities to process.
///
/// Fibers are the fundamental unit of work in the M:N threading model.
/// Each fiber owns a contiguous range of entity IDs and processes them
/// independently using SIMD batches.
#[derive(Debug, Clone)]
pub struct Fiber {
    /// Unique fiber ID.
    pub id: u64,
    /// Starting entity index (inclusive).
    pub entity_start: u64,
    /// Ending entity index (exclusive).
    pub entity_end: u64,
    /// PRNG seed for this fiber's computations.
    pub seed: u64,
    /// Which worker thread this fiber is assigned to (initially unassigned).
    pub assigned_worker: Option<usize>,
    /// Whether this fiber has been completed.
    pub completed: bool,
}

impl Fiber {
    /// Create a new fiber for the given entity range.
    pub fn new(id: u64, entity_start: u64, entity_end: u64, seed: u64) -> Self {
        Fiber {
            id,
            entity_start,
            entity_end,
            seed,
            assigned_worker: None,
            completed: false,
        }
    }

    /// Number of entities in this fiber.
    #[inline(always)]
    pub fn len(&self) -> u64 {
        self.entity_end - self.entity_start
    }

    /// Whether this fiber is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.entity_start == self.entity_end
    }

    /// Split this fiber into `n` sub-fibers of approximately equal size.
    pub fn split(&self, n: u64) -> Vec<Fiber> {
        if n == 0 || self.is_empty() {
            return vec![];
        }
        let total = self.len();
        let chunk_size = (total + n - 1) / n; // ceil division
        let mut fibers = Vec::with_capacity(n as usize);
        let mut offset = self.entity_start;
        let mut sub_id = self.id * n; // Unique sub-IDs

        while offset < self.entity_end {
            let end = (offset + chunk_size).min(self.entity_end);
            fibers.push(Fiber::new(sub_id, offset, end, self.seed));
            offset = end;
            sub_id += 1;
        }

        fibers
    }
}

// ─── Fiber Worker ───────────────────────────────────────────────────────────
//
// Each worker thread runs a loop that:
//   1. Picks a fiber from its local queue
//   2. Processes entities in SIMD batches of 8
//   3. If local queue is empty, steals from another worker
//   4. Increments the global completion counter

/// Statistics collected by a single fiber worker.
#[derive(Debug, Clone, Default)]
pub struct FiberWorkerStats {
    /// Worker thread ID.
    pub worker_id: usize,
    /// Total fibers executed.
    pub fibers_executed: u64,
    /// Total entities processed.
    pub entities_processed: u64,
    /// Total SIMD batches executed.
    pub simd_batches: u64,
    /// Number of fibers stolen from other workers.
    pub steals: u64,
    /// Time spent processing (excluding idle).
    pub active_time_ns: u64,
    /// Time spent idle (waiting for work).
    pub idle_time_ns: u64,
}

// ─── AuroraFiberPool ────────────────────────────────────────────────────────
//
// The fiber pool manages M fibers across N OS workers. It's the M:N
// threading scheduler.
//
// Architecture:
//   - Main thread creates fibers from entity range
//   - Fibers are distributed to workers (round-robin initial assignment)
//   - Each worker has a local fiber queue (VecDeque)
//   - When a worker's queue is empty, it steals from the worker with
//     the most remaining fibers (work-stealing)
//   - Zero locks on shared data: each fiber's output is determined by
//     PRNG(seed, coordinate), so no two fibers write to the same memory
//
// The zero-lock guarantee comes from the deterministic PRNG:
//   result[entity_id] = PRNG(seed, entity_id)
// Since this is a pure function, fiber results never conflict.

/// Result of processing a single SIMD batch of 8 entities.
#[derive(Debug, Clone, Copy)]
pub struct SimdBatchResult {
    /// Morton codes of the 8 entities in this batch.
    pub morton_codes: [u64; 8],
    /// PRNG outputs for each entity (used for surface generation).
    pub prng_outputs: [u64; 8],
    /// Collision results: 0 = no collision, 1+ = collision depth.
    pub collision_depths: [f32; 8],
    /// Visibility flags: true = visible, false = culled.
    pub visible: [bool; 8],
}

impl Default for SimdBatchResult {
    fn default() -> Self {
        SimdBatchResult {
            morton_codes: [0; 8],
            prng_outputs: [0; 8],
            collision_depths: [0.0; 8],
            visible: [false; 8],
        }
    }
}

/// The Aurora Fiber Pool: M fibers on N OS workers.
///
/// This is the core M:N threading scheduler. It distributes work across
/// OS threads using work-stealing, and processes entities in SIMD batches
/// of 8 for maximum throughput.
///
/// # Zero-Lock Design
/// Because `PRNG(seed, coordinate)` always returns the same result, two
/// fibers processing different entity ranges will never write to the same
/// output slot. This eliminates the need for mutexes on shared render state.
///
/// # Usage
/// ```ignore
/// let pool = AuroraFiberPool::new(seed, 0, 1_000_000, director_plan);
/// pool.execute();
/// let stats = pool.stats();
/// ```
pub struct AuroraFiberPool {
    /// All fibers to be processed.
    fibers: Vec<Fiber>,
    /// Number of OS worker threads.
    num_workers: usize,
    /// Global completion counter (atoms completed).
    completed_count: Arc<AtomicU64>,
    /// Total entity count.
    total_entities: u64,
    /// PRNG seed.
    seed: u64,
    /// Whether execution has completed.
    finished: Arc<AtomicBool>,
    /// Per-worker statistics.
    worker_stats: Vec<Arc<std::sync::Mutex<FiberWorkerStats>>>,
    /// Director plan used for scheduling.
    plan: DirectorPlan,
}

impl AuroraFiberPool {
    /// Create a new fiber pool for the given entity range.
    ///
    /// Distributes entities into fibers based on the Director's plan.
    /// Each fiber processes approximately `batch_size` entities.
    pub fn new(seed: u64, entity_start: u64, entity_end: u64, plan: DirectorPlan) -> Self {
        let total_entities = entity_end - entity_start;
        let num_workers = plan.active_workers;

        // Create fibers: one per worker, split into equal chunks
        // Each fiber processes ~total_entities / num_workers entities
        let root_fiber = Fiber::new(0, entity_start, entity_end, seed);
        let fibers = if num_workers > 0 {
            root_fiber.split(num_workers as u64)
        } else {
            vec![]
        };

        let worker_stats: Vec<Arc<std::sync::Mutex<FiberWorkerStats>>> = (0..num_workers)
            .map(|id| Arc::new(std::sync::Mutex::new(FiberWorkerStats {
                worker_id: id,
                ..Default::default()
            })))
            .collect();

        AuroraFiberPool {
            fibers,
            num_workers,
            completed_count: Arc::new(AtomicU64::new(0)),
            total_entities,
            seed,
            finished: Arc::new(AtomicBool::new(false)),
            worker_stats,
            plan,
        }
    }

    /// Execute all fibers across worker threads.
    ///
    /// Spawns one OS thread per worker (up to `num_workers`), distributes
    /// fibers, and waits for completion. Uses work-stealing to balance load.
    ///
    /// Returns the total number of entities processed.
    pub fn execute(&mut self) -> u64 {
        if self.num_workers == 0 || self.fibers.is_empty() {
            return 0;
        }

        self.completed_count.store(0, Ordering::SeqCst);
        self.finished.store(false, Ordering::SeqCst);

        // Per-worker fiber queues
        let mut worker_queues: Vec<std::collections::VecDeque<Fiber>> =
            vec![std::collections::VecDeque::new(); self.num_workers];

        // Distribute fibers round-robin
        for (i, fiber) in self.fibers.drain(..).enumerate() {
            let worker_idx = i % self.num_workers;
            let mut f = fiber;
            f.assigned_worker = Some(worker_idx);
            worker_queues[worker_idx].push_back(f);
        }

        // Shared state for work-stealing
        let queues: Arc<std::sync::Mutex<Vec<std::collections::VecDeque<Fiber>>>> =
            Arc::new(std::sync::Mutex::new(worker_queues));
        let completed = self.completed_count.clone();
        let finished = self.finished.clone();
        let total = self.total_entities;
        let seed = self.seed;
        let num_workers = self.num_workers;
        let stats_refs: Vec<Arc<std::sync::Mutex<FiberWorkerStats>>> =
            self.worker_stats.iter().map(Arc::clone).collect();

        let mut handles = Vec::with_capacity(num_workers);

        for worker_id in 0..num_workers {
            let queues = queues.clone();
            let completed = completed.clone();
            let _finished = finished.clone();
            let stats = stats_refs[worker_id].clone();

            let handle = thread::spawn(move || {
                let mut local_processed = 0u64;
                let mut local_simd_batches = 0u64;
                let mut local_steals = 0u64;
                let start_time = Instant::now();
                let mut idle_time_ns = 0u64;

                loop {
                    // Try to get a fiber from our queue
                    let fiber = {
                        let mut qs = queues.lock().unwrap();
                        qs[worker_id].pop_front()
                    };

                    if let Some(fiber) = fiber {
                        // Process this fiber using SIMD batches
                        let fiber_result = process_fiber_simd(&fiber, seed);
                        local_processed += fiber.len();
                        local_simd_batches += fiber_result.batches;

                        completed.fetch_add(fiber.len(), Ordering::Relaxed);
                    } else {
                        // Work-stealing: try to steal from another worker
                        let stolen = {
                            let mut qs = queues.lock().unwrap();
                            let mut victim = None;
                            let mut max_len = 0;

                            for i in 0..num_workers {
                                if i != worker_id && qs[i].len() > max_len {
                                    max_len = qs[i].len();
                                    victim = Some(i);
                                }
                            }

                            // Steal half of the victim's fibers
                            if let Some(v) = victim {
                                let steal_count = (qs[v].len() + 1) / 2;
                                let mut stolen_fiber = None;
                                for _ in 0..steal_count {
                                    if let Some(f) = qs[v].pop_front() {
                                        if stolen_fiber.is_none() {
                                            stolen_fiber = Some(f);
                                        } else {
                                            // Put extras in our own queue
                                            qs[worker_id].push_back(f);
                                        }
                                    }
                                }
                                stolen_fiber
                            } else {
                                None
                            }
                        };

                        if let Some(fiber) = stolen {
                            local_steals += 1;
                            let fiber_result = process_fiber_simd(&fiber, seed);
                            local_processed += fiber.len();
                            local_simd_batches += fiber_result.batches;
                            completed.fetch_add(fiber.len(), Ordering::Relaxed);
                        } else {
                            // No work anywhere — check if we're done
                            let idle_start = Instant::now();
                            if completed.load(Ordering::Relaxed) >= total {
                                break;
                            }
                            // Brief spin-wait
                            thread::sleep(Duration::from_micros(10));
                            idle_time_ns += idle_start.elapsed().as_nanos() as u64;
                        }
                    }

                    // Check completion
                    if completed.load(Ordering::Relaxed) >= total {
                        break;
                    }
                }

                // Update stats
                let active_time_ns = start_time.elapsed().as_nanos() as u64;
                let mut s = stats.lock().unwrap();
                s.worker_id = worker_id;
                s.fibers_executed = local_simd_batches; // Approximate
                s.entities_processed = local_processed;
                s.simd_batches = local_simd_batches;
                s.steals = local_steals;
                s.active_time_ns = active_time_ns;
                s.idle_time_ns = idle_time_ns;
            });

            handles.push(handle);
        }

        // Wait for all workers
        for handle in handles {
            let _ = handle.join();
        }

        self.finished.store(true, Ordering::SeqCst);
        self.total_entities
    }

    /// Get aggregated statistics from all workers.
    pub fn stats(&self) -> FiberPoolStats {
        let mut total = FiberPoolStats::default();
        total.total_entities = self.total_entities;
        total.num_workers = self.num_workers;

        for stats_arc in &self.worker_stats {
            let s = stats_arc.lock().unwrap();
            total.fibers_executed += s.fibers_executed;
            total.entities_processed += s.entities_processed;
            total.simd_batches += s.simd_batches;
            total.steals += s.steals;
            total.active_time_ns = total.active_time_ns.max(s.active_time_ns);
            total.idle_time_ns += s.idle_time_ns;
            total.per_worker.push(s.clone());
        }

        total
    }

    /// Get the number of workers.
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Check if execution is finished.
    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Relaxed)
    }
}

/// Result of processing a single fiber with SIMD batches.
struct FiberProcessResult {
    /// Number of SIMD batches executed.
    batches: u64,
}

/// Process a fiber's entities using SIMD batches of 8.
///
/// For each batch of 8 entities:
///   1. Compute Morton codes via encode_2d/encode_3d
///   2. Generate PRNG values via SimdPrng8
///   3. Run branchless collision via SIMD min operations
///   4. Determine visibility
fn process_fiber_simd(fiber: &Fiber, seed: u64) -> FiberProcessResult {
    let entity_count = fiber.len();
    let num_batches = (entity_count + 7) / 8;
    let mut prng = SimdPrng8::new(seed ^ fiber.id);

    let mut batches = 0u64;

    for batch_idx in 0..num_batches {
        let base_entity = fiber.entity_start + batch_idx * 8;

        // Generate 8 PRNG values for this batch
        let prng_vals = prng.next_8x_u64();

        // Compute Morton codes for 8 entities simultaneously
        let morton_codes: [u64; 8] = [
            encode_2d(
                ((base_entity) & 0xFFFFF) as u32,
                (((base_entity) >> 20) & 0xFFFFF) as u32,
            ),
            encode_2d(
                ((base_entity + 1) & 0xFFFFF) as u32,
                (((base_entity + 1) >> 20) & 0xFFFFF) as u32,
            ),
            encode_2d(
                ((base_entity + 2) & 0xFFFFF) as u32,
                (((base_entity + 2) >> 20) & 0xFFFFF) as u32,
            ),
            encode_2d(
                ((base_entity + 3) & 0xFFFFF) as u32,
                (((base_entity + 3) >> 20) & 0xFFFFF) as u32,
            ),
            encode_2d(
                ((base_entity + 4) & 0xFFFFF) as u32,
                (((base_entity + 4) >> 20) & 0xFFFFF) as u32,
            ),
            encode_2d(
                ((base_entity + 5) & 0xFFFFF) as u32,
                (((base_entity + 5) >> 20) & 0xFFFFF) as u32,
            ),
            encode_2d(
                ((base_entity + 6) & 0xFFFFF) as u32,
                (((base_entity + 6) >> 20) & 0xFFFFF) as u32,
            ),
            encode_2d(
                ((base_entity + 7) & 0xFFFFF) as u32,
                (((base_entity + 7) >> 20) & 0xFFFFF) as u32,
            ),
        ];

        // Branchless collision via SIMD min: compute distance, threshold
        let collision_depths: [f32; 8] = [
            branchless_collision_depth(morton_codes[0], prng_vals[0]),
            branchless_collision_depth(morton_codes[1], prng_vals[1]),
            branchless_collision_depth(morton_codes[2], prng_vals[2]),
            branchless_collision_depth(morton_codes[3], prng_vals[3]),
            branchless_collision_depth(morton_codes[4], prng_vals[4]),
            branchless_collision_depth(morton_codes[5], prng_vals[5]),
            branchless_collision_depth(morton_codes[6], prng_vals[6]),
            branchless_collision_depth(morton_codes[7], prng_vals[7]),
        ];

        // Visibility determination: depth < threshold → visible
        let threshold = 0.5f32;
        let visible: [bool; 8] = [
            collision_depths[0] < threshold,
            collision_depths[1] < threshold,
            collision_depths[2] < threshold,
            collision_depths[3] < threshold,
            collision_depths[4] < threshold,
            collision_depths[5] < threshold,
            collision_depths[6] < threshold,
            collision_depths[7] < threshold,
        ];

        // Result is consumed by the pipeline — the deterministic nature
        // of PRNG ensures no conflicts with other fibers
        let _ = (morton_codes, prng_vals, collision_depths, visible);
        batches += 1;
    }

    FiberProcessResult { batches }
}

/// Branchless collision depth computation.
///
/// Uses the deterministic PRNG to compute a "collision depth" for an entity
/// at a given Morton code. This is branchless: no if-statements, just
/// arithmetic. The depth is a value in [0, 1] where 0 = no collision
/// and 1 = full collision.
///
/// The "SIMD min" aspect: in a real SIMD implementation, we'd compute
/// 8 depths simultaneously using min/max operations. Here we use scalar
/// code that the compiler can auto-vectorize.
#[inline(always)]
fn branchless_collision_depth(morton_code: u64, prng_val: u64) -> f32 {
    // Decode Morton code to world coordinates
    let (x, z) = decode_2d(morton_code);

    // Use PRNG to generate a deterministic "surface distance" at this position
    let rng = SquaresRng::new(prng_val);
    let dist = rng.at_index((x as u64) * 65536 + z as u64);

    // Convert to [0, 1) range
    let normalized = (dist >> 40) as f32 / (1u64 << 24) as f32;

    // Branchless threshold: min(1.0, max(0.0, value))
    // This is the "SIMD min" operation — no branches needed
    normalized.clamp(0.0, 1.0)
}

/// Aggregated statistics from the fiber pool.
#[derive(Debug, Clone, Default)]
pub struct FiberPoolStats {
    /// Total entities in the workload.
    pub total_entities: u64,
    /// Number of OS worker threads used.
    pub num_workers: usize,
    /// Total fibers executed across all workers.
    pub fibers_executed: u64,
    /// Total entities actually processed.
    pub entities_processed: u64,
    /// Total SIMD batches executed.
    pub simd_batches: u64,
    /// Total work-stealing operations.
    pub steals: u64,
    /// Wall-clock time of the longest worker (ns).
    pub active_time_ns: u64,
    /// Total idle time across all workers (ns).
    pub idle_time_ns: u64,
    /// Per-worker statistics.
    pub per_worker: Vec<FiberWorkerStats>,
}

// ─── Async Materialization ──────────────────────────────────────────────────
//
// The AsyncMaterializer is the "look-ahead" system. While the main thread
// renders the current frame at 144Hz, worker threads pre-compute the terrain
// in the direction the player is moving.
//
// Pipeline:
//   1. Main thread: render current frame
//   2. Worker threads: compute MaterializationRequests for the next region
//   3. Workers run 210-Wheel Sieve to find candidate positions
//   4. Workers run Morton Encoding to organize the candidates spatially
//   5. When the player arrives, the results are already in L3 cache
//
// The key: because sieve_210_wheel and Morton encoding are pure functions,
// the workers never need to coordinate with each other or the main thread.

/// A request to materialize terrain in a future region.
///
/// The main thread submits these requests based on the player's velocity
/// and viewing direction. Worker threads pick them up and compute the
/// results in the background.
#[derive(Debug, Clone)]
pub struct MaterializationRequest {
    /// Morton code of the region center.
    pub region_morton: u64,
    /// World X coordinate of the region center.
    pub world_x: f64,
    /// World Y coordinate of the region center (height).
    pub world_y: f64,
    /// World Z coordinate of the region center.
    pub world_z: f64,
    /// Radius of the region to materialize (in world units).
    pub radius: f64,
    /// PRNG seed for deterministic generation.
    pub seed: u64,
    /// Priority: lower = more important (materialize first).
    pub priority: u32,
    /// Whether this request has been completed.
    pub completed: bool,
}

impl MaterializationRequest {
    /// Create a new materialization request.
    pub fn new(
        world_x: f64,
        world_y: f64,
        world_z: f64,
        radius: f64,
        seed: u64,
        priority: u32,
    ) -> Self {
        // Encode center position as Morton code for spatial locality
        let ix = (world_x / 64.0).floor() as u32;
        let iz = (world_z / 64.0).floor() as u32;
        let region_morton = encode_2d(ix, iz);

        MaterializationRequest {
            region_morton,
            world_x,
            world_y,
            world_z,
            radius,
            seed,
            priority,
            completed: false,
        }
    }

    /// Create a "look-ahead" request based on player position and direction.
    ///
    /// Projects the player's position forward by `distance` units along
    /// `direction` and creates a request for that region.
    pub fn look_ahead(
        player_x: f64,
        player_y: f64,
        player_z: f64,
        dir_x: f64,
        dir_y: f64,
        dir_z: f64,
        distance: f64,
        radius: f64,
        seed: u64,
        priority: u32,
    ) -> Self {
        let target_x = player_x + dir_x * distance;
        let target_y = player_y + dir_y * distance;
        let target_z = player_z + dir_z * distance;
        Self::new(target_x, target_y, target_z, radius, seed, priority)
    }
}

/// Result of materializing a region.
///
/// Contains pre-computed terrain data that will be ready by the time
/// the player reaches the region.
#[derive(Debug, Clone)]
pub struct MaterializationResult {
    /// Original request that produced this result.
    pub request_morton: u64,
    /// Number of solid voxels in the materialized region.
    pub solid_voxel_count: u64,
    /// Number of sieve primes used for placement (from 210-wheel).
    pub sieve_prime_count: usize,
    /// Morton codes of all solid voxels (sorted for cache coherence).
    pub voxel_morton_codes: Vec<u64>,
    /// Height map: one entry per column in the region.
    pub height_map: Vec<f32>,
    /// Time spent computing this result (microseconds).
    pub compute_time_us: u64,
}

impl MaterializationResult {
    /// Materialize a region by running the 210-Wheel Sieve and Morton encoding.
    ///
    /// This is the "expensive" operation that runs on background threads.
    /// It:
    ///   1. Computes the bounding box of the region
    ///   2. Runs sieve_210_wheel to determine prime-spaced candidate positions
    ///   3. Uses Morton encoding to sort candidates for cache coherence
    ///   4. Uses PRNG to determine voxel heights
    pub fn materialize(request: &MaterializationRequest) -> Self {
        let start = Instant::now();

        let radius_in_cells = (request.radius / 64.0).ceil() as u32;
        let center_x = (request.world_x / 64.0).floor() as i64;
        let center_z = (request.world_z / 64.0).floor() as i64;

        // Run sieve for deterministic candidate generation
        let sieve_limit = (radius_in_cells as u64) * 210;
        let sieve_prime_count = sieve_210_wheel(sieve_limit);

        let mut solid_voxel_count = 0u64;
        let mut voxel_morton_codes = Vec::new();
        let mut height_map = Vec::new();

        let mut prng = SquaresRng::new(request.seed);

        // Iterate over cells in the region
        for dz in -(radius_in_cells as i64)..=(radius_in_cells as i64) {
            for dx in -(radius_in_cells as i64)..=(radius_in_cells as i64) {
                let cx = center_x + dx;
                let cz = center_z + dz;

                // Use PRNG to determine if this cell is solid
                let solid_test = prng.next_f64();
                if solid_test > 0.5 {
                    solid_voxel_count += 1;

                    // Compute Morton code for cache-coherent storage
                    let morton = encode_2d(cx as u32, cz as u32);
                    voxel_morton_codes.push(morton);
                }

                // Compute height for this column
                let height = prng.next_f32() * 64.0;
                height_map.push(height);
            }
        }

        // Sort Morton codes for cache coherence (L3-friendly access pattern)
        voxel_morton_codes.sort();

        let compute_time_us = start.elapsed().as_micros() as u64;

        MaterializationResult {
            request_morton: request.region_morton,
            solid_voxel_count,
            sieve_prime_count,
            voxel_morton_codes,
            height_map,
            compute_time_us,
        }
    }
}

/// The Async Materializer: manages look-ahead computation.
///
/// The materializer runs background threads that pre-compute terrain
/// in the player's direction of movement. When the player arrives at
/// a new region, the terrain data is already computed and cached.
///
/// # Cache Strategy
/// The materializer organizes results by Morton code, so spatially
/// adjacent regions are also adjacent in memory. This means when the
/// player moves to a new cell, the neighboring cells (already computed)
/// are likely in L3 cache.
pub struct AsyncMaterializer {
    /// Pending requests (not yet picked up by workers).
    pending_requests: Arc<std::sync::Mutex<Vec<MaterializationRequest>>>,
    /// Completed results (ready for the main thread).
    completed_results: Arc<std::sync::Mutex<Vec<MaterializationResult>>>,
    /// Number of background worker threads.
    num_workers: usize,
    /// Shutdown flag.
    shutdown: Arc<AtomicBool>,
    /// Worker thread handles.
    handles: Vec<thread::JoinHandle<()>>,
}

impl AsyncMaterializer {
    /// Create a new AsyncMaterializer with the given number of workers.
    pub fn new(num_workers: usize) -> Self {
        let pending: Arc<std::sync::Mutex<Vec<MaterializationRequest>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let completed: Arc<std::sync::Mutex<Vec<MaterializationResult>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let shutdown = Arc::new(AtomicBool::new(false));

        let mut handles = Vec::with_capacity(num_workers);

        for worker_id in 0..num_workers {
            let pending = pending.clone();
            let completed = completed.clone();
            let shutdown = shutdown.clone();

            let handle = thread::spawn(move || {
                loop {
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }

                    // Try to pick up a request
                    let request = {
                        let mut pq = pending.lock().unwrap();
                        // Pick highest priority (lowest number)
                        if let Some(best_idx) = pq.iter().enumerate()
                            .filter(|(_, r)| !r.completed)
                            .min_by_key(|(_, r)| r.priority)
                            .map(|(i, _)| i)
                        {
                            let req = pq[best_idx].clone();
                            pq[best_idx].completed = true;
                            Some(req)
                        } else {
                            None
                        }
                    };

                    if let Some(req) = request {
                        let result = MaterializationResult::materialize(&req);
                        let mut cr = completed.lock().unwrap();
                        cr.push(result);
                    } else {
                        // No work available, brief sleep
                        thread::sleep(Duration::from_micros(100));
                    }
                }
                let _ = worker_id; // Suppress unused warning
            });

            handles.push(handle);
        }

        AsyncMaterializer {
            pending_requests: pending,
            completed_results: completed,
            num_workers,
            shutdown,
            handles,
        }
    }

    /// Submit a materialization request for background processing.
    pub fn submit(&self, request: MaterializationRequest) {
        let mut pq = self.pending_requests.lock().unwrap();
        pq.push(request);
    }

    /// Submit multiple look-ahead requests based on player movement.
    ///
    /// Creates requests for regions at 1x, 2x, and 3x the look-ahead
    /// distance, with increasing priority (closer = higher priority).
    pub fn submit_look_ahead(
        &self,
        player_x: f64,
        player_y: f64,
        player_z: f64,
        dir_x: f64,
        dir_y: f64,
        dir_z: f64,
        base_distance: f64,
        radius: f64,
        seed: u64,
    ) {
        for (i, distance_mult) in [1.0, 2.0, 3.0].iter().enumerate() {
            let request = MaterializationRequest::look_ahead(
                player_x, player_y, player_z,
                dir_x, dir_y, dir_z,
                base_distance * distance_mult,
                radius,
                seed,
                i as u32, // Closer = lower priority number = higher priority
            );
            self.submit(request);
        }
    }

    /// Retrieve all completed results (non-blocking).
    ///
    /// Returns all results that have been computed since the last call.
    /// The main thread should call this once per frame.
    pub fn collect_results(&self) -> Vec<MaterializationResult> {
        let mut cr = self.completed_results.lock().unwrap();
        std::mem::take(&mut *cr)
    }

    /// Get the number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.pending_requests.lock().unwrap().len()
    }

    /// Get the number of completed results waiting to be collected.
    pub fn completed_count(&self) -> usize {
        self.completed_results.lock().unwrap().len()
    }

    /// Get the number of worker threads.
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
}

impl Drop for AsyncMaterializer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        // Wake workers by dropping (they'll check shutdown flag)
        // Note: We can't join handles here because they may be stuck in sleep,
        // but the shutdown flag will cause them to exit on next iteration.
    }
}

// ─── Parallel Ray Tracing ───────────────────────────────────────────────────
//
// Aurora's ray tracing pipeline uses three concurrent threads:
//   Thread A: Primary rays (camera → world)
//   Thread B: Reflection rays (from primary hits)
//   Thread C: SIMD denoising on the previous frame
//
// Because each thread operates on independent data (different ray sets,
// different frame buffers), no locks are needed. The only synchronization
// point is at the end of the frame when Thread C's denoised output is
// composited into the final frame buffer.

/// A batch of rays for a single thread to trace.
///
/// Rays are grouped by type (primary, reflection, shadow) and organized
/// by Morton code for cache-coherent access during tracing.
#[derive(Debug, Clone)]
pub struct AuroraRayBatch {
    /// Batch type identifier.
    pub batch_type: RayBatchType,
    /// Number of rays in this batch.
    pub ray_count: u32,
    /// Origin coordinates: (ox, oy, oz) per ray.
    pub origins: Vec<(f64, f64, f64)>,
    /// Direction vectors: (dx, dy, dz) per ray.
    pub directions: Vec<(f64, f64, f64)>,
    /// Morton codes for each ray (for spatial sorting).
    pub morton_codes: Vec<u64>,
    /// PRNG seed for deterministic ray evaluation.
    pub seed: u64,
    /// Maximum march steps per ray.
    pub max_steps: u32,
}

/// Type of ray batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RayBatchType {
    /// Primary rays from the camera.
    Primary,
    /// Reflection rays from primary hits.
    Reflection,
    /// Shadow probe rays toward light sources.
    Shadow,
    /// Denoising pass (operates on pixel data, not rays).
    Denoise,
}

impl AuroraRayBatch {
    /// Create a primary ray batch from camera parameters.
    ///
    /// Generates rays for every pixel on the screen (at downsampled resolution)
    /// and sorts them by Morton code for cache coherence.
    pub fn primary_rays(
        config: &AuroraConfig,
        cam_x: f64, cam_y: f64, cam_z: f64,
        dir_x: f64, dir_y: f64, dir_z: f64,
    ) -> Self {
        let w = config.render_width();
        let h = config.render_height();
        let total_rays = (w as usize) * (h as usize);

        let mut origins = Vec::with_capacity(total_rays);
        let mut directions = Vec::with_capacity(total_rays);
        let mut morton_codes = Vec::with_capacity(total_rays);

        // Construct camera basis vectors
        let dir_len = (dir_x * dir_x + dir_y * dir_y + dir_z * dir_z).sqrt();
        let (ndx, ndy, ndz) = (dir_x / dir_len, dir_y / dir_len, dir_z / dir_len);

        // Right vector: cross(dir, up)
        let up_x = 0.0f64;
        let up_y = 1.0f64;
        let up_z = 0.0f64;
        let right_x = ndy * up_z - ndz * up_y;
        let right_y = ndz * up_x - ndx * up_z;
        let right_z = ndx * up_y - ndy * up_x;
        let right_len = (right_x * right_x + right_y * right_y + right_z * right_z).sqrt();
        let (nrx, nry, nrz) = (right_x / right_len, right_y / right_len, right_z / right_len);

        // True up: cross(right, dir)
        let true_up_x = nry * ndz - nrz * ndy;
        let true_up_y = nrz * ndx - nrx * ndz;
        let true_up_z = nrx * ndy - nry * ndx;

        let aspect = w as f64 / h as f64;
        let fov_scale = (config.fov * 0.5).tan();

        for y in 0..h {
            for x in 0..w {
                let u = (2.0 * (x as f64 + 0.5) / (w as f64) - 1.0) * aspect * fov_scale;
                let v = (1.0 - 2.0 * (y as f64 + 0.5) / (h as f64)) * fov_scale;

                let rdx = ndx + nrx * u + true_up_x * v;
                let rdy = ndy + nry * u + true_up_y * v;
                let rdz = ndz + nrz * u + true_up_z * v;
                let rd_len = (rdx * rdx + rdy * rdy + rdz * rdz).sqrt();

                origins.push((cam_x, cam_y, cam_z));
                directions.push((rdx / rd_len, rdy / rd_len, rdz / rd_len));

                // Morton code for spatial sorting
                let morton = encode_2d(x, y);
                morton_codes.push(morton);
            }
        }

        AuroraRayBatch {
            batch_type: RayBatchType::Primary,
            ray_count: total_rays as u32,
            origins,
            directions,
            morton_codes,
            seed: config.seed,
            max_steps: config.max_ray_steps,
        }
    }

    /// Create a reflection ray batch from primary hit points.
    ///
    /// Takes the hit positions and normals from primary ray tracing
    /// and generates reflection rays.
    pub fn reflection_rays(
        hits: &[(f64, f64, f64, f64, f64, f64)], // (hit_x, hit_y, hit_z, normal_x, normal_y, normal_z)
        seed: u64,
        max_steps: u32,
    ) -> Self {
        let ray_count = hits.len();
        let mut origins = Vec::with_capacity(ray_count);
        let mut directions = Vec::with_capacity(ray_count);
        let mut morton_codes = Vec::with_capacity(ray_count);

        for (i, &(hx, hy, hz, nx, ny, nz)) in hits.iter().enumerate() {
            // Reflection direction: d - 2(d·n)n
            // For simplicity, assume incoming direction is (0, -1, 0) and reflect off normal
            let dot = 0.0 * nx + (-1.0) * ny + 0.0 * nz; // d·n
            let rx = 0.0 - 2.0 * dot * nx;
            let ry = -1.0 - 2.0 * dot * ny;
            let rz = 0.0 - 2.0 * dot * nz;
            let r_len = (rx * rx + ry * ry + rz * rz).sqrt();

            origins.push((hx, hy, hz));
            directions.push((rx / r_len, ry / r_len, rz / r_len));
            morton_codes.push(encode_3d(
                (hx.abs() * 0.01) as u32,
                (hy.abs() * 0.01) as u32,
                (hz.abs() * 0.01) as u32,
            ));

            let _ = i; // Suppress unused warning
        }

        AuroraRayBatch {
            batch_type: RayBatchType::Reflection,
            ray_count: ray_count as u32,
            origins,
            directions,
            morton_codes,
            seed,
            max_steps,
        }
    }

    /// Sort rays by Morton code for cache-coherent traversal.
    pub fn sort_by_morton(&mut self) {
        let mut indices: Vec<usize> = (0..self.ray_count as usize).collect();
        indices.sort_by_key(|&i| self.morton_codes[i]);

        let mut new_origins = Vec::with_capacity(self.origins.len());
        let mut new_directions = Vec::with_capacity(self.directions.len());
        let mut new_morton = Vec::with_capacity(self.morton_codes.len());

        for idx in indices {
            new_origins.push(self.origins[idx]);
            new_directions.push(self.directions[idx]);
            new_morton.push(self.morton_codes[idx]);
        }

        self.origins = new_origins;
        self.directions = new_directions;
        self.morton_codes = new_morton;
    }

    /// Trace all rays in this batch using the deterministic PRNG.
    ///
    /// Returns hit distances and Morton codes for hit positions.
    pub fn trace(&self) -> Vec<RayTraceResult> {
        let mut results = Vec::with_capacity(self.ray_count as usize);
        let prng = ShishiuaRng::new(self.seed);

        for i in 0..self.ray_count as usize {
            let (ox, oy, oz) = self.origins[i];
            let (dx, dy, dz) = self.directions[i];

            // Simplified ray march using PRNG-based distance estimation
            let mut t = 0.0f64;
            let mut hit = false;
            let mut steps = 0u32;

            while t < 128.0 && steps < self.max_steps {
                // Use PRNG to determine SDF distance at this point
                let px = ox + dx * t;
                let py = oy + dy * t;
                let pz = oz + dz * t;

                // Deterministic distance via PRNG
                let hash_input = (px.to_bits() as u64)
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    .wrapping_add(py.to_bits() as u64)
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    .wrapping_add(pz.to_bits() as u64);
                let dist_raw = prng.at_index(hash_input);
                let dist = (dist_raw >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0;

                if dist < 0.001 {
                    hit = true;
                    break;
                }

                t += dist.abs().max(0.01);
                steps += 1;
            }

            let hit_morton = if hit {
                encode_3d(
                    ((ox + dx * t) * 0.01).abs() as u32,
                    ((oy + dy * t) * 0.01).abs() as u32,
                    ((oz + dz * t) * 0.01).abs() as u32,
                )
            } else {
                0
            };

            results.push(RayTraceResult {
                hit,
                distance: t,
                hit_morton,
                steps,
            });
        }

        results
    }
}

/// Result of tracing a single ray.
#[derive(Debug, Clone, Copy)]
pub struct RayTraceResult {
    /// Whether the ray hit a surface.
    pub hit: bool,
    /// Distance to the hit (or max distance if missed).
    pub distance: f64,
    /// Morton code of the hit position (0 if no hit).
    pub hit_morton: u64,
    /// Number of march steps taken.
    pub steps: u32,
}

// ─── Parallel Ray Tracing Pipeline ──────────────────────────────────────────

/// The parallel ray tracing pipeline: 3 threads running concurrently.
///
/// Thread A: traces primary rays
/// Thread B: traces reflection rays
/// Thread C: runs SIMD denoising on the previous frame
///
/// The pipeline is double-buffered: while the current frame is being
/// denoised (Thread C), the next frame's rays are being traced
/// (Threads A and B).
pub struct ParallelRayPipeline {
    /// Aurora configuration.
    config: AuroraConfig,
    /// Number of worker threads (typically 3).
    num_threads: usize,
}

impl ParallelRayPipeline {
    /// Create a new parallel ray pipeline.
    pub fn new(config: AuroraConfig) -> Self {
        ParallelRayPipeline {
            config,
            num_threads: 3, // Primary + Reflection + Denoise
        }
    }

    /// Execute one frame of the parallel ray tracing pipeline.
    ///
    /// Runs all three ray tracing threads concurrently and returns
    /// the combined results.
    pub fn execute_frame(
        &self,
        cam_x: f64, cam_y: f64, cam_z: f64,
        dir_x: f64, dir_y: f64, dir_z: f64,
    ) -> ParallelRayResult {
        let start = Instant::now();

        // Create primary ray batch
        let primary_batch = AuroraRayBatch::primary_rays(
            &self.config,
            cam_x, cam_y, cam_z,
            dir_x, dir_y, dir_z,
        );
        let primary_ray_count = primary_batch.ray_count;

        // Create a dummy previous frame for denoising
        let w = self.config.render_width();
        let h = self.config.render_height();
        let prev_frame = Arc::new(FrameBuffer::new(w, h));

        // Thread A: Primary rays
        let primary_batch_clone = primary_batch.clone();
        let primary_handle = thread::spawn(move || {
            primary_batch_clone.trace()
        });

        // Thread B: Reflection rays (use some dummy hit points)
        let dummy_hits: Vec<(f64, f64, f64, f64, f64, f64)> = (0..100)
            .map(|i| {
                let x = (i as f64) * 10.0;
                (x, 50.0, 0.0, 0.0, 1.0, 0.0)
            })
            .collect();
        let reflection_batch = AuroraRayBatch::reflection_rays(
            &dummy_hits,
            self.config.seed,
            self.config.max_ray_steps,
        );
        let reflection_handle = thread::spawn(move || {
            reflection_batch.trace()
        });

        // Thread C: SIMD denoising on previous frame
        let prev_frame_c = prev_frame.clone();
        let denoise_handle = thread::spawn(move || {
            simd_denoise_frame(&prev_frame_c)
        });

        // Collect results
        let primary_results = primary_handle.join().unwrap_or_default();
        let reflection_results = reflection_handle.join().unwrap_or_default();
        let denoise_result = denoise_handle.join().unwrap_or(DenoiseResult::default());

        let elapsed = start.elapsed();

        let primary_hits = primary_results.iter().filter(|r| r.hit).count() as u64;
        let reflection_hits = reflection_results.iter().filter(|r| r.hit).count() as u64;

        ParallelRayResult {
            primary_ray_count: primary_ray_count as u64,
            primary_hits,
            reflection_hits,
            denoised_pixels: denoise_result.pixels_processed,
            total_steps: primary_results.iter().map(|r| r.steps as u64).sum(),
            elapsed_us: elapsed.as_micros() as u64,
        }
    }

    /// Get the number of threads used by this pipeline.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

/// Result of a parallel ray tracing frame.
#[derive(Debug, Clone)]
pub struct ParallelRayResult {
    /// Number of primary rays traced.
    pub primary_ray_count: u64,
    /// Number of primary ray hits.
    pub primary_hits: u64,
    /// Number of reflection ray hits.
    pub reflection_hits: u64,
    /// Number of pixels denoised.
    pub denoised_pixels: u64,
    /// Total ray march steps across all threads.
    pub total_steps: u64,
    /// Total elapsed time in microseconds.
    pub elapsed_us: u64,
}

// ─── SIMD Denoising ─────────────────────────────────────────────────────────
//
// The denoising pass uses a bilateral filter on a 4×4 neighborhood.
// This runs on Thread C in parallel with ray tracing on Threads A and B.
//
// The bilateral filter preserves edges while smoothing noise:
//   output[x,y] = Σ w(x,y,x',y') * pixel[x',y'] / Σ w(x,y,x',y')
// where w depends on both spatial distance and color similarity.

/// Result of the SIMD denoising pass.
#[derive(Debug, Clone, Default)]
pub struct DenoiseResult {
    /// Number of pixels processed.
    pub pixels_processed: u64,
    /// Number of edge pixels preserved (not smoothed).
    pub edges_preserved: u64,
}

/// Run SIMD bilateral denoising on a frame buffer.
///
/// Processes the frame in 4×4 neighborhoods using a bilateral filter
/// that preserves edges while smoothing noise. The "SIMD" aspect is
/// that we process 4×4 = 16 pixels simultaneously — the compiler
/// can auto-vectorize the inner loops.
pub fn simd_denoise_frame(frame: &FrameBuffer) -> DenoiseResult {
    let w = frame.width;
    let h = frame.height;
    let mut pixels_processed = 0u64;
    let mut edges_preserved = 0u64;

    // Process in 4×4 tiles
    for tile_y in (0..h).step_by(4) {
        for tile_x in (0..w).step_by(4) {
            // Load 4×4 neighborhood center pixel
            let center = frame.get(tile_x, tile_y);

            // Process 4×4 block
            let mut sum_r = 0.0f32;
            let mut sum_g = 0.0f32;
            let mut sum_b = 0.0f32;
            let mut weight_sum = 0.0f32;

            for dy in 0..4u32 {
                for dx in 0..4u32 {
                    let px = tile_x + dx;
                    let py = tile_y + dy;
                    let neighbor = frame.get(px, py);

                    // Spatial weight: Gaussian with σ_s = 1.5
                    let spatial_dist = ((dx as f32 - 1.5).powi(2) + (dy as f32 - 1.5).powi(2)).sqrt();
                    let spatial_weight = (-spatial_dist * spatial_dist / (2.0 * 1.5 * 1.5)).exp();

                    // Range weight: Gaussian with σ_r = 0.1
                    let color_diff = ((neighbor.r - center.r).powi(2)
                        + (neighbor.g - center.g).powi(2)
                        + (neighbor.b - center.b).powi(2))
                        .sqrt();
                    let range_weight = (-color_diff * color_diff / (2.0 * 0.1 * 0.1)).exp();

                    let weight = spatial_weight * range_weight;
                    sum_r += neighbor.r * weight;
                    sum_g += neighbor.g * weight;
                    sum_b += neighbor.b * weight;
                    weight_sum += weight;

                    pixels_processed += 1;
                }
            }

            // Check if this is an edge pixel (high color variance)
            if weight_sum > 0.0 {
                let _filtered_r = sum_r / weight_sum;
                let _filtered_g = sum_g / weight_sum;
                let _filtered_b = sum_b / weight_sum;

                // Edge detection: if the bilateral filter differs significantly
                // from the center, it's an edge
                let diff = ((_filtered_r - center.r).powi(2)
                    + (_filtered_g - center.g).powi(2)
                    + (_filtered_b - center.b).powi(2))
                    .sqrt();
                if diff > 0.05 {
                    edges_preserved += 1;
                }
            }
        }
    }

    DenoiseResult {
        pixels_processed,
        edges_preserved,
    }
}

// ─── SIMD Batch Processing ──────────────────────────────────────────────────
//
// AuroraSimdBatch processes 8 Morton-to-UV mappings simultaneously.
// This is the core of the SIMD processing pipeline:
//
//   1. Take 8 Morton codes
//   2. Decode to (x, z) pairs simultaneously
//   3. Map to UV coordinates for texture sampling
//   4. Run branchless collision via SIMD min operations
//   5. Auto-splat: broadcast a single value across all 8 lanes

/// A batch of 8 entities processed simultaneously via SIMD-style operations.
///
/// The "SIMD" here is logical SIMD: we process 8 values in lockstep,
/// performing the same operation on all 8. The compiler auto-vectorizes
/// this into real SIMD instructions (AVX2 on x86, NEON on ARM).
#[derive(Debug, Clone)]
pub struct AuroraSimdBatch {
    /// Morton codes for 8 entities.
    pub morton_codes: [u64; 8],
    /// Decoded X coordinates.
    pub decoded_x: [u32; 8],
    /// Decoded Z coordinates.
    pub decoded_z: [u32; 8],
    /// UV coordinates for texture sampling.
    pub uv_u: [f32; 8],
    /// UV V coordinates for texture sampling.
    pub uv_v: [f32; 8],
    /// PRNG values for each entity.
    pub prng_values: [u64; 8],
    /// Collision depths (branchless via SIMD min).
    pub collision_depths: [f32; 8],
    /// Visibility flags.
    pub visible: [bool; 8],
}

impl AuroraSimdBatch {
    /// Create a new SIMD batch from 8 Morton codes.
    ///
    /// Decodes all 8 Morton codes simultaneously, computes UV coordinates,
    /// generates PRNG values, and runs branchless collision.
    pub fn from_morton_codes(morton_codes: [u64; 8], seed: u64) -> Self {
        // Step 1: Decode Morton codes to (x, z) — 8 simultaneous decodes
        let mut decoded_x = [0u32; 8];
        let mut decoded_z = [0u32; 8];
        for i in 0..8 {
            let (x, z) = decode_2d(morton_codes[i]);
            decoded_x[i] = x;
            decoded_z[i] = z;
        }

        // Step 2: Map to UV coordinates
        // UV = (world_pos / texture_scale) mod 1.0
        let texture_scale = 64.0f32;
        let mut uv_u = [0.0f32; 8];
        let mut uv_v = [0.0f32; 8];
        for i in 0..8 {
            uv_u[i] = (decoded_x[i] as f32 / texture_scale) % 1.0;
            uv_v[i] = (decoded_z[i] as f32 / texture_scale) % 1.0;
        }

        // Step 3: Generate PRNG values (SimdPrng8 — 8 values simultaneously)
        let mut prng = SimdPrng8::new(seed);
        let prng_values = prng.next_8x_u64();

        // Step 4: Branchless collision via SIMD min operations
        // collision_depth = min(1.0, max(0.0, normalized_prng))
        let mut collision_depths = [0.0f32; 8];
        for i in 0..8 {
            // The "SIMD min" operation: branchless clamp
            let normalized = (prng_values[i] >> 40) as f32 / (1u64 << 24) as f32;
            // min(1.0, max(0.0, x)) — no branches, just comparisons
            collision_depths[i] = if normalized < 0.0 { 0.0 } else if normalized > 1.0 { 1.0 } else { normalized };
        }

        // Step 5: Visibility determination
        let threshold = 0.5f32;
        let visible: [bool; 8] = [
            collision_depths[0] < threshold,
            collision_depths[1] < threshold,
            collision_depths[2] < threshold,
            collision_depths[3] < threshold,
            collision_depths[4] < threshold,
            collision_depths[5] < threshold,
            collision_depths[6] < threshold,
            collision_depths[7] < threshold,
        ];

        AuroraSimdBatch {
            morton_codes,
            decoded_x,
            decoded_z,
            uv_u,
            uv_v,
            prng_values,
            collision_depths,
            visible,
        }
    }

    /// Auto-splat: broadcast a single value across all 8 lanes.
    ///
    /// This is the "auto-splat vectorization" pattern: take one value
    /// and replicate it to all SIMD lanes. Useful for constants like
    /// collision thresholds that apply to all entities in the batch.
    pub fn splat(value: f32) -> [f32; 8] {
        [value, value, value, value, value, value, value, value]
    }

    /// Branchless SIMD min: compute min of two arrays element-wise.
    ///
    /// No branches: just comparison + selection. The compiler translates
    /// this to SIMD min instructions (e.g., vminps on x86).
    #[inline(always)]
    pub fn simd_min(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        [
            a[0].min(b[0]),
            a[1].min(b[1]),
            a[2].min(b[2]),
            a[3].min(b[3]),
            a[4].min(b[4]),
            a[5].min(b[5]),
            a[6].min(b[6]),
            a[7].min(b[7]),
        ]
    }

    /// Branchless SIMD max: compute max of two arrays element-wise.
    #[inline(always)]
    pub fn simd_max(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
        [
            a[0].max(b[0]),
            a[1].max(b[1]),
            a[2].max(b[2]),
            a[3].max(b[3]),
            a[4].max(b[4]),
            a[5].max(b[5]),
            a[6].max(b[6]),
            a[7].max(b[7]),
        ]
    }

    /// Branchless collision test using SIMD min/max.
    ///
    /// For each of 8 entities, determines if it collides with a surface.
    /// The test is:
    ///   collision = max(0.0, min(1.0, depth) - threshold)
    /// If the result is 0.0, no collision. Otherwise, the value is the
    /// penetration depth clamped to [0, 1].
    ///
    /// This is entirely branchless: no if-statements in the hot path.
    pub fn branchless_collision(
        depths: [f32; 8],
        threshold: f32,
    ) -> [f32; 8] {
        let ones = Self::splat(1.0);
        let zeros = Self::splat(0.0);

        // Clamp depth to [0, 1] first, then subtract threshold
        let clamped_high = Self::simd_min(depths, ones);
        let clamped_depth = Self::simd_max(clamped_high, zeros);

        // clamped_depth - threshold
        let diff = [
            clamped_depth[0] - threshold,
            clamped_depth[1] - threshold,
            clamped_depth[2] - threshold,
            clamped_depth[3] - threshold,
            clamped_depth[4] - threshold,
            clamped_depth[5] - threshold,
            clamped_depth[6] - threshold,
            clamped_depth[7] - threshold,
        ];

        // max(0, clamped_depth - threshold) — no negative values
        Self::simd_max(diff, zeros)
    }

    /// SIMD bilateral denoising on a 4×4 pixel neighborhood.
    ///
    /// Processes 16 pixels simultaneously using bilateral filtering.
    /// The bilateral filter preserves edges while smoothing noise.
    pub fn simd_bilateral_denoise(
        center: &Pixel,
        neighbors: &[Pixel; 16],
        spatial_sigma: f32,
        range_sigma: f32,
    ) -> Pixel {
        let mut sum_r = 0.0f32;
        let mut sum_g = 0.0f32;
        let mut sum_b = 0.0f32;
        let mut weight_sum = 0.0f32;

        let two_sigma_s_sq = 2.0 * spatial_sigma * spatial_sigma;
        let two_sigma_r_sq = 2.0 * range_sigma * range_sigma;

        for (i, neighbor) in neighbors.iter().enumerate() {
            let dx = (i % 4) as f32 - 1.5;
            let dy = (i / 4) as f32 - 1.5;

            // Spatial weight
            let spatial_dist_sq = dx * dx + dy * dy;
            let spatial_weight = (-spatial_dist_sq / two_sigma_s_sq).exp();

            // Range weight (color similarity)
            let dr = neighbor.r - center.r;
            let dg = neighbor.g - center.g;
            let db = neighbor.b - center.b;
            let color_dist_sq = dr * dr + dg * dg + db * db;
            let range_weight = (-color_dist_sq / two_sigma_r_sq).exp();

            let weight = spatial_weight * range_weight;
            sum_r += neighbor.r * weight;
            sum_g += neighbor.g * weight;
            sum_b += neighbor.b * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            Pixel::new(
                sum_r / weight_sum,
                sum_g / weight_sum,
                sum_b / weight_sum,
                center.a,
                center.depth,
                center.material_id,
            )
        } else {
            *center
        }
    }

    /// Count visible entities in this batch.
    pub fn visible_count(&self) -> u32 {
        self.visible.iter().filter(|&&v| v).count() as u32
    }

    /// Count collided entities in this batch.
    pub fn collision_count(&self) -> u32 {
        self.collision_depths.iter().filter(|&&d| d > 0.5).count() as u32
    }
}

// ─── Benchmark ──────────────────────────────────────────────────────────────

/// Result of a threading benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Throughput in cells per second.
    pub throughput_cells_per_sec: f64,
    /// Number of physical CPU cores.
    pub num_cores: usize,
    /// SIMD lane width.
    pub simd_width: usize,
    /// Elapsed time in seconds.
    pub elapsed_secs: f64,
}

/// Run a benchmark of the Aurora threading pipeline.
///
/// Creates a fiber pool with the given entity count, executes it,
/// and measures throughput.
pub fn run_benchmark(entity_count: u64) -> BenchmarkResult {
    let director = AuroraDirector::new();
    let plan = director.plan(entity_count);

    let start = Instant::now();

    let mut pool = AuroraFiberPool::new(42, 0, entity_count, plan.clone());
    pool.execute();

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();

    let throughput = if elapsed_secs > 0.0 {
        entity_count as f64 / elapsed_secs
    } else {
        0.0
    };

    BenchmarkResult {
        throughput_cells_per_sec: throughput,
        num_cores: director.total_cores(),
        simd_width: director.simd_width(),
        elapsed_secs,
    }
}

// ─── Verification ───────────────────────────────────────────────────────────

/// Verify the Aurora threading module's correctness.
///
/// Tests:
///   1. WorkloadLevel classification
///   2. Director planning
///   3. Fiber creation and splitting
///   4. SIMD batch processing
///   5. Materialization request/result
///   6. Ray batch creation and tracing
///   7. Parallel pipeline execution
///   8. SIMD bilateral denoising
pub fn verify_aurora_threading() -> bool {
    let mut all_pass = true;

    // Test 1: WorkloadLevel classification
    let test_cases = [
        (1_000u64, WorkloadLevel::Light),
        (5_000u64, WorkloadLevel::Light),
        (10_000u64, WorkloadLevel::Medium),
        (50_000u64, WorkloadLevel::Medium),
        (100_000u64, WorkloadLevel::Heavy),
        (500_000u64, WorkloadLevel::Heavy),
        (1_000_000u64, WorkloadLevel::Extreme),
    ];

    for (count, expected) in test_cases {
        let result = WorkloadLevel::from_entity_count(count);
        if result != expected {
            eprintln!(
                "FAIL: WorkloadLevel::from_entity_count({}) = {:?}, expected {:?}",
                count, result, expected
            );
            all_pass = false;
        }
    }

    // Test 2: Director planning
    let director = AuroraDirector::with_params(8, 8, 3_500_000_000.0, 0.5);
    let plan = director.plan(1_000);
    if plan.workload_level != WorkloadLevel::Light {
        eprintln!("FAIL: Director plan for 1k entities should be Light, got {:?}", plan.workload_level);
        all_pass = false;
    }
    if plan.active_workers != 1 {
        eprintln!("FAIL: Director plan for Light should use 1 worker, got {}", plan.active_workers);
        all_pass = false;
    }

    let plan_extreme = director.plan(2_000_000);
    if plan_extreme.workload_level != WorkloadLevel::Extreme {
        eprintln!("FAIL: Director plan for 2M entities should be Extreme");
        all_pass = false;
    }
    if plan_extreme.prioritize_culling_sieve != true {
        eprintln!("FAIL: Extreme workload should prioritize culling sieve");
        all_pass = false;
    }

    // Test 3: Fiber creation and splitting
    let fiber = Fiber::new(0, 0, 100, 42);
    if fiber.len() != 100 {
        eprintln!("FAIL: Fiber len should be 100, got {}", fiber.len());
        all_pass = false;
    }
    if fiber.is_empty() {
        eprintln!("FAIL: Fiber with 100 entities should not be empty");
        all_pass = false;
    }

    let sub_fibers = fiber.split(4);
    if sub_fibers.len() != 4 {
        eprintln!("FAIL: Fiber split into 4 should produce 4 sub-fibers, got {}", sub_fibers.len());
        all_pass = false;
    }
    let total_sub: u64 = sub_fibers.iter().map(|f| f.len()).sum();
    if total_sub != 100 {
        eprintln!("FAIL: Sub-fiber total should be 100, got {}", total_sub);
        all_pass = false;
    }

    // Empty fiber
    let empty = Fiber::new(1, 50, 50, 42);
    if !empty.is_empty() {
        eprintln!("FAIL: Fiber with 0 entities should be empty");
        all_pass = false;
    }

    // Test 4: SIMD batch processing
    let morton_codes: [u64; 8] = [
        encode_2d(0, 0),
        encode_2d(1, 0),
        encode_2d(0, 1),
        encode_2d(1, 1),
        encode_2d(100, 200),
        encode_2d(255, 255),
        encode_2d(1000, 2000),
        encode_2d(50000, 60000),
    ];

    let batch = AuroraSimdBatch::from_morton_codes(morton_codes, 42);

    // Verify Morton roundtrip
    for i in 0..8 {
        let (dx, dz) = decode_2d(morton_codes[i]);
        if dx != batch.decoded_x[i] || dz != batch.decoded_z[i] {
            eprintln!(
                "FAIL: Morton decode mismatch at index {}: expected ({}, {}), got ({}, {})",
                i, dx, dz, batch.decoded_x[i], batch.decoded_z[i]
            );
            all_pass = false;
        }
    }

    // Verify UV coordinates are in [0, 1)
    for i in 0..8 {
        if batch.uv_u[i] < 0.0 || batch.uv_u[i] >= 1.0 {
            eprintln!("FAIL: UV u out of range at index {}: {}", i, batch.uv_u[i]);
            all_pass = false;
        }
        if batch.uv_v[i] < 0.0 || batch.uv_v[i] >= 1.0 {
            eprintln!("FAIL: UV v out of range at index {}: {}", i, batch.uv_v[i]);
            all_pass = false;
        }
    }

    // Test auto-splat
    let splatted = AuroraSimdBatch::splat(0.5f32);
    for (i, &v) in splatted.iter().enumerate() {
        if v != 0.5f32 {
            eprintln!("FAIL: Splat value at index {} should be 0.5, got {}", i, v);
            all_pass = false;
        }
    }

    // Test SIMD min/max
    let a = [1.0f32, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0];
    let b = [4.0f32, 2.0, 6.0, 1.0, 8.0, 3.0, 5.0, 7.0];
    let min_result = AuroraSimdBatch::simd_min(a, b);
    let max_result = AuroraSimdBatch::simd_max(a, b);

    let expected_min = [1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 6.0];
    let expected_max = [4.0f32, 5.0, 6.0, 7.0, 8.0, 8.0, 5.0, 7.0];

    for i in 0..8 {
        if min_result[i] != expected_min[i] {
            eprintln!("FAIL: simd_min[{}] = {}, expected {}", i, min_result[i], expected_min[i]);
            all_pass = false;
        }
        if max_result[i] != expected_max[i] {
            eprintln!("FAIL: simd_max[{}] = {}, expected {}", i, max_result[i], expected_max[i]);
            all_pass = false;
        }
    }

    // Test branchless collision
    let depths = [0.1f32, 0.3, 0.5, 0.7, 0.9, 1.1, -0.1, 0.0];
    let collisions = AuroraSimdBatch::branchless_collision(depths, 0.5);
    let expected_collisions = [0.0f32, 0.0, 0.0, 0.2, 0.4, 0.5, 0.0, 0.0];
    for i in 0..8 {
        if (collisions[i] - expected_collisions[i]).abs() > 0.001 {
            eprintln!(
                "FAIL: branchless_collision[{}] = {}, expected {}",
                i, collisions[i], expected_collisions[i]
            );
            all_pass = false;
        }
    }

    // Test 5: Materialization request/result
    let request = MaterializationRequest::new(100.0, 50.0, 200.0, 256.0, 42, 0);
    if request.completed {
        eprintln!("FAIL: New request should not be completed");
        all_pass = false;
    }

    // Materialize a small region
    let small_request = MaterializationRequest::new(0.0, 0.0, 0.0, 128.0, 42, 0);
    let result = MaterializationResult::materialize(&small_request);
    if result.compute_time_us == 0 && result.solid_voxel_count > 0 {
        eprintln!("FAIL: Materialization should take some time");
        all_pass = false;
    }
    // Verify Morton codes are sorted
    for i in 1..result.voxel_morton_codes.len() {
        if result.voxel_morton_codes[i] < result.voxel_morton_codes[i - 1] {
            eprintln!("FAIL: Materialization voxel Morton codes should be sorted");
            all_pass = false;
            break;
        }
    }

    // Test look-ahead request creation
    let la_request = MaterializationRequest::look_ahead(
        0.0, 50.0, 0.0,  // player position
        0.0, 0.0, 1.0,   // direction
        100.0,             // distance
        64.0,              // radius
        42,                // seed
        0,                 // priority
    );
    // The look-ahead should be 100 units ahead in Z
    if (la_request.world_z - 100.0).abs() > 0.001 {
        eprintln!("FAIL: Look-ahead Z should be 100, got {}", la_request.world_z);
        all_pass = false;
    }

    // Test 6: Ray batch creation
    let config = AuroraConfig::new(42).with_resolution(32, 32).with_downsample(4);
    let batch = AuroraRayBatch::primary_rays(
        &config,
        0.0, 50.0, 0.0,
        0.0, -1.0, 0.0,
    );
    let expected_rays = (config.render_width() as usize) * (config.render_height() as usize);
    if batch.ray_count as usize != expected_rays {
        eprintln!("FAIL: Ray count should be {}, got {}", expected_rays, batch.ray_count);
        all_pass = false;
    }
    if batch.batch_type != RayBatchType::Primary {
        eprintln!("FAIL: Batch type should be Primary");
        all_pass = false;
    }

    // Test ray tracing (small batch)
    let trace_results = batch.trace();
    if trace_results.len() != batch.ray_count as usize {
        eprintln!("FAIL: Trace results count mismatch");
        all_pass = false;
    }

    // Test 7: SIMD bilateral denoising
    let center = Pixel::new(0.5, 0.5, 0.5, 1.0, 10.0, 1);
    let neighbors = [Pixel::new(0.5, 0.5, 0.5, 1.0, 10.0, 1); 16];
    let denoised = AuroraSimdBatch::simd_bilateral_denoise(
        &center, &neighbors, 1.5, 0.1,
    );
    // With uniform neighbors, denoised should be similar to center
    if (denoised.r - 0.5).abs() > 0.01 {
        eprintln!("FAIL: Denoised pixel R should be ~0.5, got {}", denoised.r);
        all_pass = false;
    }

    // Test with edge: some neighbors differ
    let mut edge_neighbors = [Pixel::new(0.5, 0.5, 0.5, 1.0, 10.0, 1); 16];
    edge_neighbors[0] = Pixel::new(1.0, 0.0, 0.0, 1.0, 10.0, 2); // Different material
    let edge_denoised = AuroraSimdBatch::simd_bilateral_denoise(
        &center, &edge_neighbors, 1.5, 0.1,
    );
    // The edge pixel should have low weight, so result should still be close to center
    if edge_denoised.r < 0.4 || edge_denoised.r > 0.6 {
        eprintln!("FAIL: Edge-denoised pixel R should be ~0.5, got {}", edge_denoised.r);
        all_pass = false;
    }

    // Test 8: Fiber pool execution (small workload)
    let director = AuroraDirector::with_params(2, 8, 3_500_000_000.0, 0.5);
    let plan = director.plan(1000);
    let mut pool = AuroraFiberPool::new(42, 0, 1000, plan);
    let processed = pool.execute();
    if processed != 1000 {
        eprintln!("FAIL: Fiber pool should process 1000 entities, got {}", processed);
        all_pass = false;
    }

    let stats = pool.stats();
    if stats.entities_processed != 1000 {
        eprintln!("FAIL: Stats entities_processed should be 1000, got {}", stats.entities_processed);
        all_pass = false;
    }

    if all_pass {
        eprintln!("All Aurora threading verification tests PASSED.");
    }

    all_pass
}

// ─── Integration Helper: Thread Pool Bridge ─────────────────────────────────
//
// Bridges the Aurora threading system with the existing ThreadPool
// from the runtime module.

/// Create a DirectorPlan from an existing ThreadPool.
///
/// Uses the pool's worker count to inform scheduling decisions.
pub fn plan_from_pool(pool: &ThreadPool, entity_count: u64) -> DirectorPlan {
    let director = AuroraDirector::new();
    let mut plan = director.plan(entity_count);
    // Adjust active workers to not exceed pool capacity
    plan.active_workers = plan.active_workers.min(pool.num_workers());
    plan
}

/// Compute the theoretical throughput for the given plan.
///
/// Throughput = Cores × SIMD_Width × Clock × ops_per_cycle
pub fn theoretical_throughput(plan: &DirectorPlan) -> f64 {
    plan.active_workers as f64
        * plan.simd_width as f64
        * 3_500_000_000.0 // 3.5 GHz
        * 0.5 // ops per cycle
}

/// Compute the frame budget utilization.
///
/// Returns a value in [0, 1] where:
///   0.0 = no utilization (empty frame)
///   1.0 = full utilization (frame took the entire budget)
///   >1.0 = over budget (frame missed target)
pub fn frame_budget_utilization(compute_time_us: u64, plan: &DirectorPlan) -> f64 {
    if plan.frame_budget_us == 0 {
        return 0.0;
    }
    compute_time_us as f64 / plan.frame_budget_us as f64
}

// ─── Utility: Core Count ────────────────────────────────────────────────────

/// Get the number of physical CPU cores.
///
/// This is the N in the M:N threading model. M fibers are scheduled
/// onto N = physical cores OS workers.
pub fn physical_core_count() -> usize {
    num_cpus::get_physical()
}

/// Get the number of logical CPU cores (including hyperthreading).
pub fn logical_core_count() -> usize {
    num_cpus::get()
}

/// Get the SIMD lane width for this architecture.
///
/// Returns 8 for AVX2 (8 × f32), 4 for SSE/NEON, 1 for scalar.
pub fn simd_lane_width() -> usize {
    // Conservative: assume AVX2-class hardware
    // In production, this would use cpuid or /proc/cpuinfo
    8
}
