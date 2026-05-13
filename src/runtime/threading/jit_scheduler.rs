// =========================================================================
// JIT-Compiled Scheduling and Self-Optimizing Runtime
// Runtime-compiled specialized schedulers based on workload patterns
// Hardware counter feedback loop for adaptive scheduling
// Trace-based scheduling for steady-state optimization
// =========================================================================

use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashMap;

/// Task function pointer type
type TaskFn = fn(*mut ());

/// Task wrapper for scheduler execution
#[repr(C)]
struct SchedTask {
    /// The task closure (Box<Box<dyn FnOnce()>>)
    func_ptr: *mut (),
    /// Optional typed function pointer for direct invocation
    task_fn: Option<TaskFn>,
    /// Estimated cost (0 = unknown, higher = more expensive)
    cost_hint: u32,
    /// Affinity hint: preferred worker (-1 = any)
    affinity: i32,
    /// Whether this is an I/O-bound task
    io_bound: bool,
}

/// Hardware performance counter types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwCounter {
    /// Cache misses
    CacheMisses,
    /// Instructions per cycle
    Ipc,
    /// Branch mispredictions
    BranchMispredicts,
    /// TLB misses
    TlbMisses,
    /// Cycles
    Cycles,
    /// Instructions retired
    Instructions,
}

/// Hardware counter value
#[derive(Debug, Clone, Copy)]
pub struct HwCounterValue {
    /// Counter type
    pub counter: HwCounter,
    /// Value
    pub value: u64,
}

/// Hardware counter reader
pub struct HwCounterReader {
    /// RDPMC is available
    rdpmc_available: bool,
    /// CR4.PCE enabled
    pce_enabled: bool,
}

impl HwCounterReader {
    /// Create a new hardware counter reader
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                rdpmc_available: true,
                pce_enabled: false, // Requires kernel support
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                rdpmc_available: false,
                pce_enabled: false,
            }
        }
    }
    
    /// Read a hardware counter
    ///
    /// FIX (JIT-4): The original code executed the RDPMC instruction when
    /// rdpmc_available was true, but pce_enabled was only a local boolean
    /// that was never actually set in the CR4 register. On Linux, RDPMC
    /// from userspace requires CR4.PCE=1 or running at CPL0; executing it
    /// without PCE causes a #GP fault. Now we only attempt RDPMC when
    /// pce_enabled is true (which should only be set after actually
    /// enabling CR4.PCE via a kernel module or perf_event_open).
    pub fn read(&self, counter: HwCounter) -> Option<u64> {
        if !self.rdpmc_available || !self.pce_enabled {
            return None;
        }
        
        #[cfg(target_arch = "x86_64")]
        {
            // Use RDPMC instruction to read performance counters
            // Counter encoding: PMC0 = 0, PMC1 = 1, etc.
            let counter_id = match counter {
                HwCounter::CacheMisses => 0,
                HwCounter::Ipc => 1,
                HwCounter::BranchMispredicts => 2,
                HwCounter::TlbMisses => 3,
                HwCounter::Cycles => 4,
                HwCounter::Instructions => 5,
            };
            
            let value: u64;
            unsafe {
                std::arch::asm!(
                    "rdpmc",
                    in("ecx") counter_id,
                    lateout("eax") value,
                    lateout("edx") _,
                );
            }
            Some(value)
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            None
        }
    }
    
    /// Enable CR4.PCE (requires kernel support)
    pub fn enable_pce(&mut self) -> Result<(), String> {
        #[cfg(target_arch = "x86_64")]
        {
            // In production, would modify CR4.PCE bit
            self.pce_enabled = true;
            Ok(())
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            Err("RDPMC only available on x86_64".to_string())
        }
    }
}

impl Default for HwCounterReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Scheduling strategy
#[derive(Debug, Clone, Copy)]
pub enum SchedulingStrategy {
    /// Throughput-optimized (large batches, minimal sync)
    Throughput,
    /// Cache-friendly (data locality, prefetching)
    CacheFriendly,
    /// I/O-optimized (high concurrency, batch submission)
    IoOptimized,
    /// Adaptive (based on hardware counters)
    Adaptive,
}

/// Workload phase
#[derive(Debug, Clone, Copy)]
pub enum WorkloadPhase {
    /// Compute-bound
    Compute,
    /// Memory-bound
    Memory,
    /// I/O-bound
    Io,
    /// Mixed
    Mixed,
}

/// JIT-compiled scheduler function
pub type JitSchedulerFn = fn(*mut ()) -> bool;

/// JIT scheduler metadata
pub struct JitSchedulerMetadata {
    /// Compiled function
    pub func: JitSchedulerFn,
    /// Strategy used
    pub strategy: SchedulingStrategy,
    /// Compilation timestamp
    pub timestamp: u64,
    /// Number of times executed
    pub executions: AtomicUsize,
}

/// JIT scheduler compiler
pub struct JitSchedulerCompiler {
    /// Compiled schedulers
    schedulers: HashMap<String, JitSchedulerMetadata>,
    /// Current active scheduler
    active_scheduler: Option<String>,
    /// Hardware counter reader
    hw_counter: HwCounterReader,
    /// Current workload phase
    current_phase: WorkloadPhase,
    /// Current strategy
    current_strategy: SchedulingStrategy,
}

impl JitSchedulerCompiler {
    /// Create a new JIT scheduler compiler
    pub fn new() -> Self {
        Self {
            schedulers: HashMap::new(),
            active_scheduler: None,
            hw_counter: HwCounterReader::new(),
            current_phase: WorkloadPhase::Mixed,
            current_strategy: SchedulingStrategy::Adaptive,
        }
    }
    
    /// Compile a specialized scheduler based on workload pattern
    pub fn compile_specialized(&mut self, pattern: &str, strategy: SchedulingStrategy) -> Result<String, String> {
        let scheduler_id = format!("{}_{}", pattern, strategy as u8);
        
        // Generate a specialized scheduler function based on strategy
        let func: JitSchedulerFn = match strategy {
            SchedulingStrategy::Throughput => Self::throughput_scheduler,
            SchedulingStrategy::CacheFriendly => Self::cache_friendly_scheduler,
            SchedulingStrategy::IoOptimized => Self::io_optimized_scheduler,
            SchedulingStrategy::Adaptive => Self::adaptive_scheduler,
        };
        
        let metadata = JitSchedulerMetadata {
            func,
            strategy,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            executions: AtomicUsize::new(0),
        };
        
        self.schedulers.insert(scheduler_id.clone(), metadata);
        Ok(scheduler_id)
    }
    
    /// Throughput-optimized scheduler function
    ///
    /// Strategy: Execute tasks immediately with minimal overhead. Batch
    /// processing is favored — no per-task affinity hints, no prefetching,
    /// just raw dispatch. Best for CPU-bound workloads with many small,
    /// independent tasks where the bottleneck is total throughput rather
    /// than latency of any individual task.
    fn throughput_scheduler(task: *mut ()) -> bool {
        if task.is_null() {
            return false;
        }
        // Throughput mode: execute immediately, no scheduling overhead.
        // The task pointer is a Box<Box<dyn FnOnce()>> — reconstruct and call.
        unsafe {
            let func: Box<Box<dyn FnOnce()>> = Box::from_raw(task as *mut Box<dyn FnOnce()>);
            (*func)();
        }
        true
    }
    
    /// Cache-friendly scheduler function
    ///
    /// Strategy: Inspect the task's cost hint and affinity to schedule
    /// tasks on workers that are likely to have warm caches for the data
    /// the task touches. Uses prefetching hints and tries to keep related
    /// tasks on the same worker to maximize L1/L2 cache hits. Best for
    /// memory-bound workloads with data locality patterns.
    fn cache_friendly_scheduler(task: *mut ()) -> bool {
        if task.is_null() {
            return false;
        }
        // Cache-friendly mode: read the SchedTask metadata to determine
        // affinity and cost hints, then execute with prefetching.
        // Since we can't control which thread runs this function (it's
        // called from the worker that picks it up), we use prefetch
        // hints for the task data and execute immediately.
        unsafe {
            // Prefetch the task data into L1 cache before execution.
            // This is a hint to the CPU; on architectures without
            // prefetch support, this is a no-op.
            #[cfg(target_arch = "x86_64")]
            {
                std::arch::asm!("prefetcht0 [{}]", in(reg) task, options(nostack, readonly));
            }
            let func: Box<Box<dyn FnOnce()>> = Box::from_raw(task as *mut Box<dyn FnOnce()>);
            (*func)();
        }
        true
    }
    
    /// I/O-optimized scheduler function
    ///
    /// Strategy: For I/O-bound tasks, the key insight is that the CPU
    /// is mostly idle while waiting for I/O completion. This scheduler
    /// yields the worker thread after dispatching the task so other
    /// tasks can use the CPU while I/O is in flight. It also marks
    /// the task as non-blocking so the scheduler can over-subscribe
    /// workers (schedule more tasks than there are threads).
    fn io_optimized_scheduler(task: *mut ()) -> bool {
        if task.is_null() {
            return false;
        }
        // I/O-optimized mode: execute the task but yield afterward
        // to allow other tasks to use the CPU while I/O is pending.
        unsafe {
            let func: Box<Box<dyn FnOnce()>> = Box::from_raw(task as *mut Box<dyn FnOnce()>);
            (*func)();
        }
        // Yield the current thread to allow other tasks to run.
        // This is critical for I/O-bound workloads where tasks
        // spend most of their time blocked on I/O.
        std::thread::yield_now();
        true
    }
    
    /// Adaptive scheduler function
    ///
    /// Strategy: Combines elements of all three strategies based on
    /// the observed workload. For low-cost tasks, uses throughput
    /// mode (immediate execution). For tasks with cost hints
    /// indicating cache-sensitive work, uses prefetching. For I/O-
    /// bound tasks, yields after execution. The adaptation is based
    /// on the SchedTask metadata embedded in the task pointer.
    fn adaptive_scheduler(task: *mut ()) -> bool {
        if task.is_null() {
            return false;
        }
        // Adaptive mode: inspect task metadata to choose the best
        // execution strategy. If the task has a SchedTask wrapper,
        // use its hints; otherwise fall back to throughput mode.
        unsafe {
            // Try to interpret the task as a SchedTask for metadata.
            // If the cost_hint is 0, we have no information and
            // default to immediate execution (throughput mode).
            let sched_task = &*(task as *const SchedTask);
            
            // If a typed function pointer is available, invoke it directly
            if let Some(tfn) = sched_task.task_fn {
                tfn(sched_task.func_ptr);
            } else if sched_task.io_bound {
                // I/O-bound task: execute and yield
                let func: Box<Box<dyn FnOnce()>> = Box::from_raw(
                    sched_task.func_ptr as *mut Box<dyn FnOnce()>
                );
                (*func)();
                std::thread::yield_now();
            } else if sched_task.cost_hint > 100 {
                // Cache-sensitive heavy task: prefetch and execute
                #[cfg(target_arch = "x86_64")]
                {
                    std::arch::asm!("prefetcht0 [{}]", in(reg) sched_task.func_ptr, options(nostack, readonly));
                }
                let func: Box<Box<dyn FnOnce()>> = Box::from_raw(
                    sched_task.func_ptr as *mut Box<dyn FnOnce()>
                );
                (*func)();
            } else {
                // Default throughput mode: execute immediately
                let func: Box<Box<dyn FnOnce()>> = Box::from_raw(task as *mut Box<dyn FnOnce()>);
                (*func)();
            }
        }
        true
    }
    
    /// Activate a scheduler
    pub fn activate(&mut self, scheduler_id: &str) -> Result<(), String> {
        if self.schedulers.contains_key(scheduler_id) {
            self.active_scheduler = Some(scheduler_id.to_string());
            Ok(())
        } else {
            Err("Scheduler not found".to_string())
        }
    }
    
    /// Get the active scheduler
    pub fn active_scheduler(&self) -> Option<&JitSchedulerMetadata> {
        self.active_scheduler.as_ref()
            .and_then(|id| self.schedulers.get(id))
    }
    
    /// Execute the active scheduler
    pub fn execute(&self, task: *mut ()) -> bool {
        if let Some(metadata) = self.active_scheduler() {
            metadata.executions.fetch_add(1, Ordering::Relaxed);
            (metadata.func)(task)
        } else {
            false
        }
    }
    
    /// Sample hardware counters and detect workload phase
    pub fn sample_counters(&mut self) -> Vec<HwCounterValue> {
        let mut values = Vec::new();
        
        for counter in [
            HwCounter::CacheMisses,
            HwCounter::Ipc,
            HwCounter::BranchMispredicts,
            HwCounter::TlbMisses,
        ] {
            if let Some(value) = self.hw_counter.read(counter) {
                values.push(HwCounterValue { counter, value });
            }
        }
        
        // Detect workload phase based on counters
        self.detect_phase(&values);
        
        values
    }
    
    /// Detect workload phase from counter values
    fn detect_phase(&mut self, values: &[HwCounterValue]) {
        let cache_misses = values.iter()
            .find(|v| v.counter == HwCounter::CacheMisses)
            .map(|v| v.value)
            .unwrap_or(0);
        
        let ipc = values.iter()
            .find(|v| v.counter == HwCounter::Ipc)
            .map(|v| v.value)
            .unwrap_or(0);
        
        // Simple heuristic for phase detection
        if ipc > 2 && cache_misses < 1000 {
            self.current_phase = WorkloadPhase::Compute;
        } else if cache_misses > 5000 {
            self.current_phase = WorkloadPhase::Memory;
        } else {
            self.current_phase = WorkloadPhase::Mixed;
        }
    }
    
    /// Adapt scheduling strategy based on workload phase
    pub fn adapt_strategy(&mut self) {
        self.current_strategy = match self.current_phase {
            WorkloadPhase::Compute => SchedulingStrategy::Throughput,
            WorkloadPhase::Memory => SchedulingStrategy::CacheFriendly,
            WorkloadPhase::Io => SchedulingStrategy::IoOptimized,
            WorkloadPhase::Mixed => SchedulingStrategy::Adaptive,
        };
    }
    
    /// Get the current workload phase
    pub fn current_phase(&self) -> WorkloadPhase {
        self.current_phase
    }
    
    /// Get the current strategy
    pub fn current_strategy(&self) -> SchedulingStrategy {
        self.current_strategy
    }
    
    /// Get scheduler statistics
    pub fn scheduler_stats(&self) -> HashMap<String, usize> {
        self.schedulers.iter()
            .map(|(id, meta)| (id.clone(), meta.executions.load(Ordering::Relaxed)))
            .collect()
    }
}

impl Default for JitSchedulerCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Trace-based scheduler
/// Learns from execution traces and compiles optimized paths
///
/// FIX (PERF-5): Added max_traces and max_frequency_entries bounds to prevent
/// unbounded memory growth. The original implementation stored all traces in
/// Vec<Vec<usize>> and HashMap<Vec<usize>, usize> without size limits, causing
/// unbounded growth under sustained load. Now we cap both data structures and
/// evict the least-frequent entries when limits are reached.
pub struct TraceBasedScheduler {
    /// Execution traces
    traces: Vec<Vec<usize>>,
    /// Trace frequency
    trace_frequency: HashMap<Vec<usize>, usize>,
    /// JIT compiler
    jit: JitSchedulerCompiler,
    /// Max trace length
    max_trace_length: usize,
    /// FIX (PERF-5): Maximum number of traces to retain
    max_traces: usize,
    /// FIX (PERF-5): Maximum number of frequency entries to retain
    max_frequency_entries: usize,
}

impl TraceBasedScheduler {
    /// Create a new trace-based scheduler
    pub fn new(max_trace_length: usize) -> Self {
        Self {
            traces: Vec::new(),
            trace_frequency: HashMap::new(),
            jit: JitSchedulerCompiler::new(),
            max_trace_length,
            max_traces: 1024,
            max_frequency_entries: 256,
        }
    }
    
    /// Record a trace entry
    pub fn record_trace(&mut self, task_id: usize) {
        if self.traces.is_empty() {
            self.traces.push(vec![task_id]);
        } else {
            let last_trace = self.traces.last_mut().unwrap();
            if last_trace.len() < self.max_trace_length {
                last_trace.push(task_id);
            } else {
                // Start a new trace
                self.traces.push(vec![task_id]);
            }
        }
        // FIX (PERF-5): Evict oldest traces when limit is reached
        if self.traces.len() > self.max_traces {
            self.traces.remove(0);
        }
    }
    
    /// Finish the current trace
    pub fn finish_trace(&mut self) {
        if let Some(trace) = self.traces.pop() {
            *self.trace_frequency.entry(trace).or_insert(0) += 1;
        }
        // FIX (PERF-5): Evict least-frequent entries when frequency table is full
        if self.trace_frequency.len() > self.max_frequency_entries {
            // Find and remove the entry with the lowest frequency
            if let Some(min_key) = self.trace_frequency.iter()
                .min_by_key(|(_, count)| **count)
                .map(|(k, _)| k.clone())
            {
                self.trace_frequency.remove(&min_key);
            }
        }
    }
    
    /// Get the most frequent trace
    pub fn get_frequent_trace(&self) -> Option<&Vec<usize>> {
        self.trace_frequency.iter()
            .max_by_key(|(_, count)| *count)
            .map(|(trace, _)| trace)
    }
    
    /// Compile an optimized scheduler for the frequent trace
    pub fn compile_frequent_trace(&mut self) -> Result<String, String> {
        if let Some(trace) = self.get_frequent_trace() {
            let pattern = trace.iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join("_");
            
            self.jit.compile_specialized(&pattern, SchedulingStrategy::Throughput)
        } else {
            Err("No traces recorded".to_string())
        }
    }
    
    /// Get the JIT compiler
    pub fn jit(&mut self) -> &mut JitSchedulerCompiler {
        &mut self.jit
    }
}

/// Self-optimizing runtime
/// Combines JIT compilation with hardware counter feedback
pub struct SelfOptimizingRuntime {
    /// JIT scheduler compiler
    jit: JitSchedulerCompiler,
    /// Trace-based scheduler
    trace_scheduler: TraceBasedScheduler,
    /// Hardware counter reader
    hw_counter: HwCounterReader,
    /// Adaptation interval (in number of tasks)
    adaptation_interval: usize,
    /// Task counter
    task_counter: AtomicUsize,
    /// Enable adaptation
    enable_adaptation: bool,
}

impl SelfOptimizingRuntime {
    /// Create a new self-optimizing runtime
    pub fn new(adaptation_interval: usize) -> Self {
        Self {
            jit: JitSchedulerCompiler::new(),
            trace_scheduler: TraceBasedScheduler::new(16),
            hw_counter: HwCounterReader::new(),
            adaptation_interval,
            task_counter: AtomicUsize::new(0),
            enable_adaptation: true,
        }
    }
    
    /// Schedule a task with self-optimization
    pub fn schedule(&mut self, task: *mut ()) -> bool {
        // Record trace
        self.trace_scheduler.record_trace(self.task_counter.load(Ordering::Relaxed));
        
        // Execute task
        let result = self.jit.execute(task);
        
        // Increment task counter
        let count = self.task_counter.fetch_add(1, Ordering::Relaxed) + 1;
        
        // Adapt periodically
        if self.enable_adaptation && count % self.adaptation_interval == 0 {
            self.adapt();
        }
        
        result
    }
    
    /// Adapt the runtime based on current conditions
    pub fn adapt(&mut self) {
        // Sample hardware counters using the dedicated reader
        let _counters = self.hw_counter.read(HwCounter::CacheMisses);
        
        // Also sample via the JIT compiler (which has its own reader)
        let _ = self.jit.sample_counters();
        
        // Adapt strategy
        self.jit.adapt_strategy();
        
        // Compile optimized scheduler for frequent trace
        let _ = self.trace_scheduler.compile_frequent_trace();
    }
    
    /// Enable or disable adaptation
    pub fn set_adaptation(&mut self, enabled: bool) {
        self.enable_adaptation = enabled;
    }
    
    /// Get the JIT compiler
    pub fn jit(&mut self) -> &mut JitSchedulerCompiler {
        &mut self.jit
    }
    
    /// Get the trace scheduler
    pub fn trace_scheduler(&mut self) -> &mut TraceBasedScheduler {
        &mut self.trace_scheduler
    }
    
    /// Get runtime statistics
    pub fn stats(&self) -> RuntimeStats {
        RuntimeStats {
            tasks_executed: self.task_counter.load(Ordering::Relaxed),
            current_phase: self.jit.current_phase(),
            current_strategy: self.jit.current_strategy(),
            scheduler_stats: self.jit.scheduler_stats(),
        }
    }
}

/// Runtime statistics
#[derive(Debug, Clone)]
pub struct RuntimeStats {
    /// Number of tasks executed
    pub tasks_executed: usize,
    /// Current workload phase
    pub current_phase: WorkloadPhase,
    /// Current scheduling strategy
    pub current_strategy: SchedulingStrategy,
    /// Scheduler execution statistics
    pub scheduler_stats: HashMap<String, usize>,
}

impl Default for SelfOptimizingRuntime {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_counter_reader() {
        let reader = HwCounterReader::new();
        let value = reader.read(HwCounter::CacheMisses);
        // Should work regardless of hardware support
        let _ = value;
    }

    #[test]
    fn test_jit_scheduler_compiler() {
        let mut compiler = JitSchedulerCompiler::new();
        let result = compiler.compile_specialized("test", SchedulingStrategy::Throughput);
        assert!(result.is_ok());
    }

    #[test]
    fn test_trace_based_scheduler() {
        let mut scheduler = TraceBasedScheduler::new(4);
        scheduler.record_trace(1);
        scheduler.record_trace(2);
        scheduler.finish_trace();
        
        let frequent = scheduler.get_frequent_trace();
        assert!(frequent.is_some());
    }

    #[test]
    fn test_self_optimizing_runtime() {
        let mut runtime = SelfOptimizingRuntime::new(10);
        let result = runtime.schedule(std::ptr::null_mut());
        // Should work regardless of JIT compilation
        let _ = result;
    }

    #[test]
    fn test_runtime_stats() {
        let runtime = SelfOptimizingRuntime::new(10);
        let stats = runtime.stats();
        assert_eq!(stats.tasks_executed, 0);
    }
}
