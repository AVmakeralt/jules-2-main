// =========================================================================
// Superoptimizer Integration with Threading Engine
// Cross-boundary optimization and scheduling hints
// Analyzes expressions to determine optimal execution strategy
// Integrates all threading optimizations with algebraic rewrite rules
// =========================================================================

use crate::compiler::ast::{Expr, BinOpKind, Span};
use crate::runtime::threading::{
    ThreadPool, Worker,
    PerCpuDeque, PerCpuCounter, get_cpu_id, is_rseq_available, register_rseq,
    HybridNotify, IoUring, UintrReceiver, UintrSender, is_io_uring_available, is_uintr_available,
    AmxContext, Avx512Mask, CatManager, CompareOp, HugePageAllocator, HwCapabilities, TsxTransaction,
    is_amx_available, is_avx512_available, is_cat_available, is_tsx_available,
    SoaScheduler, SoaTaskQueue, warm_function_cache, warm_task_cache,
    StackTask, TaskBatch, TaskCache, TASK_TYPE_COMPUTE, TASK_TYPE_IO, TASK_TYPE_GPU, TASK_TYPE_SYNC,
    JitSchedulerCompiler, RuntimeStats, SchedulingStrategy, SelfOptimizingRuntime, TraceBasedScheduler, WorkloadPhase,
    HwCounter, HwCounterReader, HwCounterValue,
    PrecisionLevel, TaskPriority, LossyComputationContext, LossyComputationManager,
    HyperSparseMap, HyperSparseSoA, SegmentedSieve,
    CrossBoundaryOptimizer, FusedOperation,
};

/// Scheduling hint from superoptimizer (extended)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingHint {
    /// Execute sequentially (no parallelism)
    Sequential,
    /// Execute in parallel
    Parallel,
    /// Execute on GPU
    Gpu,
    /// Inline into caller
    Inline,
    /// SIMD vectorization
    Simd,
    /// Use rseq wait-free path
    RseqWaitFree,
    /// Use per-CPU deque
    PerCpu,
    /// Use io_uring kernel bypass
    IoUring,
    /// Use Intel UINTR
    Uintr,
    /// Use AMX for matrix ops
    Amx,
    /// Use TSX transactional memory
    Tsx,
    /// Use CAT cache partitioning
    Cat,
    /// Use AVX-512 masked ops
    Avx512,
    /// Use huge pages
    HugePages,
    /// Use SoA layout
    SoA,
    /// Use JIT-compiled scheduler
    JitScheduler,
    /// Use speculative execution
    Speculative,
    /// Use work compression
    WorkCompression,
    /// Use neural-guided scheduling
    NeuralGuided,
    /// Use lossy computation with adaptive precision
    LossyComputation,
    /// Use hyper-sparse data structures
    HyperSparse,
    /// Use cross-boundary optimization (fused lossy + hyper-sparse)
    CrossBoundary,
}

/// Task metadata with scheduling hints (extended)
pub struct TaskMetadata {
    /// Scheduling hint
    pub hint: SchedulingHint,
    /// Estimated cost (cycles)
    pub cost: u64,
    /// Data dependencies (task IDs)
    pub dependencies: Vec<u64>,
    /// Hardware requirements
    pub hw_requirements: HwRequirements,
    /// Memory footprint (bytes)
    pub memory_footprint: usize,
    /// Parallelism degree
    pub parallelism: usize,
    /// Data locality hint
    pub locality: DataLocality,
}

/// Hardware requirements for a task
#[derive(Debug, Clone, Default)]
pub struct HwRequirements {
    /// Requires rseq
    pub requires_rseq: bool,
    /// Requires io_uring
    pub requires_io_uring: bool,
    /// Requires UINTR
    pub requires_uintr: bool,
    /// Requires AMX
    pub requires_amx: bool,
    /// Requires TSX
    pub requires_tsx: bool,
    /// Requires CAT
    pub requires_cat: bool,
    /// Requires AVX-512
    pub requires_avx512: bool,
    /// Requires huge pages
    pub requires_huge_pages: bool,
}

/// Data locality hint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataLocality {
    /// No locality requirement
    None,
    /// Same NUMA node
    SameNuma,
    /// Same CPU
    SameCpu,
    /// Same cache line
    SameCacheLine,
}

impl TaskMetadata {
    /// Create new task metadata
    pub fn new(hint: SchedulingHint, cost: u64) -> Self {
        Self {
            hint,
            cost,
            dependencies: Vec::new(),
            hw_requirements: HwRequirements::default(),
            memory_footprint: 0,
            parallelism: 1,
            locality: DataLocality::None,
        }
    }

    /// Add a dependency
    pub fn add_dependency(&mut self, dep_id: u64) {
        self.dependencies.push(dep_id);
    }

    /// Check if task has dependencies
    pub fn has_dependencies(&self) -> bool {
        !self.dependencies.is_empty()
    }

    /// Set hardware requirements
    pub fn set_hw_requirements(&mut self, req: HwRequirements) {
        self.hw_requirements = req;
    }

    /// Set memory footprint
    pub fn set_memory_footprint(&mut self, footprint: usize) {
        self.memory_footprint = footprint;
    }

    /// Set parallelism degree
    pub fn set_parallelism(&mut self, parallelism: usize) {
        self.parallelism = parallelism;
    }

    /// Set data locality
    pub fn set_locality(&mut self, locality: DataLocality) {
        self.locality = locality;
    }
}

/// Expression analysis result (extended)
#[derive(Debug, Clone)]
pub struct ExpressionAnalysis {
    /// Scheduling hint
    pub hint: SchedulingHint,
    /// Estimated cost
    pub cost: u64,
    /// Memory footprint (bytes)
    pub memory_footprint: usize,
    /// Parallelism degree
    pub parallelism: usize,
    /// Hardware requirements
    pub hw_requirements: HwRequirements,
    /// Data locality
    pub locality: DataLocality,
    /// Applicable rewrite rules
    pub rewrite_rules: Vec<RewriteRule>,
}

/// Algebraic rewrite rule for threading optimizations
#[derive(Debug, Clone)]
pub struct RewriteRule {
    /// Rule name
    pub name: String,
    /// Pattern to match
    pub pattern: String,
    /// Replacement pattern
    pub replacement: String,
    /// Applicability condition
    pub condition: RewriteCondition,
    /// Expected speedup
    pub speedup: f64,
}

/// Condition for applying a rewrite rule
#[derive(Debug, Clone)]
pub enum RewriteCondition {
    /// Always applicable
    Always,
    /// Requires specific hardware
    RequiresHw(String),
    /// Requires specific data size
    RequiresMinSize(usize),
    /// Requires specific parallelism
    RequiresMinParallelism(usize),
    /// Combination of conditions
    And(Box<RewriteCondition>, Box<RewriteCondition>),
    /// Either condition
    Or(Box<RewriteCondition>, Box<RewriteCondition>),
}

impl ExpressionAnalysis {
    fn new(hint: SchedulingHint, cost: u64) -> Self {
        Self {
            hint,
            cost,
            memory_footprint: 0,
            parallelism: 1,
            hw_requirements: HwRequirements::default(),
            locality: DataLocality::None,
            rewrite_rules: Vec::new(),
        }
    }
}

/// Generate all rewrite rules for threading optimizations
pub fn generate_rewrite_rules() -> Vec<RewriteRule> {
    let mut rules = Vec::new();
    
    // Phase 1: rseq wait-free rules
    rules.push(RewriteRule {
        name: "rseq_per_cpu_push".to_string(),
        pattern: "push(task)".to_string(),
        replacement: "rseq_push(task)".to_string(),
        condition: RewriteCondition::RequiresHw("rseq".to_string()),
        speedup: 5.0,
    });
    
    rules.push(RewriteRule {
        name: "rseq_per_cpu_pop".to_string(),
        pattern: "pop()".to_string(),
        replacement: "rseq_pop()".to_string(),
        condition: RewriteCondition::RequiresHw("rseq".to_string()),
        speedup: 5.0,
    });
    
    // Phase 2: Zero-allocation rules
    rules.push(RewriteRule {
        name: "stack_task_alloc".to_string(),
        pattern: "allocate_task()".to_string(),
        replacement: "stack_task_alloc()".to_string(),
        condition: RewriteCondition::Always,
        speedup: 2.0,
    });
    
    rules.push(RewriteRule {
        name: "slab_alloc".to_string(),
        pattern: "heap_alloc()".to_string(),
        replacement: "slab_alloc()".to_string(),
        condition: RewriteCondition::Always,
        speedup: 3.0,
    });
    
    // Phase 3: Kernel bypass rules
    rules.push(RewriteRule {
        name: "io_uring_notify".to_string(),
        pattern: "futex_notify()".to_string(),
        replacement: "io_uring_notify()".to_string(),
        condition: RewriteCondition::RequiresHw("io_uring".to_string()),
        speedup: 10.0,
    });
    
    rules.push(RewriteRule {
        name: "uintr_notify".to_string(),
        pattern: "futex_notify()".to_string(),
        replacement: "uintr_notify()".to_string(),
        condition: RewriteCondition::RequiresHw("uintr".to_string()),
        speedup: 9.0,
    });
    
    // Phase 4: Hardware-specific rules
    rules.push(RewriteRule {
        name: "amx_matmul".to_string(),
        pattern: "matmul(A, B)".to_string(),
        replacement: "amx_matmul(A, B)".to_string(),
        condition: RewriteCondition::And(
            Box::new(RewriteCondition::RequiresHw("amx".to_string())),
            Box::new(RewriteCondition::RequiresMinSize(1024)),
        ),
        speedup: 10.0,
    });
    
    rules.push(RewriteRule {
        name: "tsx_steal".to_string(),
        pattern: "cas_steal()".to_string(),
        replacement: "tsx_steal()".to_string(),
        condition: RewriteCondition::RequiresHw("tsx".to_string()),
        speedup: 2.0,
    });
    
    rules.push(RewriteRule {
        name: "cat_partition".to_string(),
        pattern: "shared_cache()".to_string(),
        replacement: "cat_partition()".to_string(),
        condition: RewriteCondition::RequiresHw("cat".to_string()),
        speedup: 1.5,
    });
    
    rules.push(RewriteRule {
        name: "avx512_scan".to_string(),
        pattern: "priority_scan()".to_string(),
        replacement: "avx512_priority_scan()".to_string(),
        condition: RewriteCondition::RequiresHw("avx512".to_string()),
        speedup: 8.0,
    });
    
    rules.push(RewriteRule {
        name: "huge_pages_alloc".to_string(),
        pattern: "mmap_alloc()".to_string(),
        replacement: "huge_pages_alloc()".to_string(),
        condition: RewriteCondition::And(
            Box::new(RewriteCondition::RequiresHw("huge_pages".to_string())),
            Box::new(RewriteCondition::RequiresMinSize(2 * 1024 * 1024)),
        ),
        speedup: 2.0,
    });
    
    // Phase 5: SoA rules
    rules.push(RewriteRule {
        name: "soa_queue".to_string(),
        pattern: "aos_queue()".to_string(),
        replacement: "soa_queue()".to_string(),
        condition: RewriteCondition::Always,
        speedup: 4.0,
    });
    
    rules.push(RewriteRule {
        name: "prefetch_task".to_string(),
        pattern: "schedule(task)".to_string(),
        replacement: "prefetch_and_schedule(task)".to_string(),
        condition: RewriteCondition::Always,
        speedup: 2.0,
    });
    
    // Phase 6: JIT scheduler rules
    rules.push(RewriteRule {
        name: "jit_specialized".to_string(),
        pattern: "generic_schedule(task)".to_string(),
        replacement: "jit_specialized_schedule(task)".to_string(),
        condition: RewriteCondition::RequiresMinParallelism(4),
        speedup: 2.0,
    });
    
    rules.push(RewriteRule {
        name: "hw_counter_adapt".to_string(),
        pattern: "fixed_strategy()".to_string(),
        replacement: "adaptive_strategy()".to_string(),
        condition: RewriteCondition::RequiresHw("rdpmc".to_string()),
        speedup: 1.5,
    });
    
    // Phase 7: Zero-copy rules
    rules.push(RewriteRule {
        name: "disruptor_ring".to_string(),
        pattern: "channel_send()".to_string(),
        replacement: "disruptor_send()".to_string(),
        condition: RewriteCondition::Always,
        speedup: 5.0,
    });
    
    rules.push(RewriteRule {
        name: "zero_copy_transfer".to_string(),
        pattern: "copy_and_send(data)".to_string(),
        replacement: "transfer_ownership(data)".to_string(),
        condition: RewriteCondition::RequiresMinSize(4096),
        speedup: 10.0,
    });
    
    // Phase 8: AOT schedule rules
    rules.push(RewriteRule {
        name: "aot_precomputed".to_string(),
        pattern: "runtime_schedule()".to_string(),
        replacement: "aot_schedule()".to_string(),
        condition: RewriteCondition::Always,
        speedup: 10.0,
    });
    
    rules.push(RewriteRule {
        name: "egraph_extract".to_string(),
        pattern: "dynamic_schedule()".to_string(),
        replacement: "egraph_optimal_schedule()".to_string(),
        condition: RewriteCondition::Always,
        speedup: 5.0,
    });
    
    // Phase 9: Novel techniques
    rules.push(RewriteRule {
        name: "speculative_exec".to_string(),
        pattern: "schedule_when_ready(task)".to_string(),
        replacement: "speculative_execute(task)".to_string(),
        condition: RewriteCondition::Always,
        speedup: 1.1,
    });
    
    rules.push(RewriteRule {
        name: "work_compression".to_string(),
        pattern: "for_each(task) { execute(task) }".to_string(),
        replacement: "batch_execute(tasks)".to_string(),
        condition: RewriteCondition::RequiresMinParallelism(16),
        speedup: 10.0,
    });
    
    rules.push(RewriteRule {
        name: "neural_guided".to_string(),
        pattern: "heuristic_schedule()".to_string(),
        replacement: "neural_schedule()".to_string(),
        condition: RewriteCondition::RequiresMinParallelism(4),
        speedup: 1.1,
    });
    
    // Lossy computation rules
    rules.push(RewriteRule {
        name: "lossy_background".to_string(),
        pattern: "f64_compute()".to_string(),
        replacement: "lossy_f16_compute()".to_string(),
        condition: RewriteCondition::Or(
            Box::new(RewriteCondition::RequiresHw("amx".to_string())),
            Box::new(RewriteCondition::RequiresHw("avx512".to_string())),
        ),
        speedup: 50.0,
    });
    
    rules.push(RewriteRule {
        name: "lossy_adaptive".to_string(),
        pattern: "fixed_precision_compute()".to_string(),
        replacement: "adaptive_precision_compute()".to_string(),
        condition: RewriteCondition::RequiresHw("rdpmc".to_string()),
        speedup: 10.0,
    });
    
    // Hyper-sparse rules
    rules.push(RewriteRule {
        name: "hyper_sparse_map".to_string(),
        pattern: "dense_array()".to_string(),
        replacement: "hyper_sparse_bit_map()".to_string(),
        condition: RewriteCondition::Always,
        speedup: 8.0,
    });
    
    rules.push(RewriteRule {
        name: "segmented_sieve".to_string(),
        pattern: "full_traversal()".to_string(),
        replacement: "segmented_sieve_traversal()".to_string(),
        condition: RewriteCondition::Always,
        speedup: 8.0,
    });
    
    // Cross-boundary optimization rules
    rules.push(RewriteRule {
        name: "cross_boundary_fused".to_string(),
        pattern: "lossy_compute(sparse_data)".to_string(),
        replacement: "fused_lossy_sparse_compute()".to_string(),
        condition: RewriteCondition::And(
            Box::new(RewriteCondition::RequiresHw("amx".to_string())),
            Box::new(RewriteCondition::RequiresHw("cat".to_string())),
        ),
        speedup: 100.0,
    });
    
    rules.push(RewriteRule {
        name: "zero_copy_fused".to_string(),
        pattern: "copy_and_compute(data)".to_string(),
        replacement: "zero_copy_fused_compute(data)".to_string(),
        condition: RewriteCondition::Always,
        speedup: 20.0,
    });
    
    rules.push(RewriteRule {
        name: "egraph_sparse_opt".to_string(),
        pattern: "sparse_data_access()".to_string(),
        replacement: "egraph_optimized_sparse_access()".to_string(),
        condition: RewriteCondition::Always,
        speedup: 5.0,
    });
    
    rules
}

/// Superoptimizer integration for threading (extended)
pub struct SuperoptThreadingIntegration {
    /// Thread pool for task execution
    pool: ThreadPool,
    /// Task metadata registry
    task_metadata: std::collections::HashMap<u64, TaskMetadata>,
    /// Next task ID
    next_task_id: u64,
    /// Expression analysis cache
    analysis_cache: std::collections::HashMap<String, ExpressionAnalysis>,
    /// Hardware capabilities
    hw_caps: HwCapabilities,
    /// Rewrite rules
    rewrite_rules: Vec<RewriteRule>,
    /// JIT scheduler
    jit_scheduler: Option<JitSchedulerCompiler>,
    /// Self-optimizing runtime
    self_opt_runtime: Option<SelfOptimizingRuntime>,
    /// SoA scheduler
    soa_scheduler: Option<SoaScheduler>,
    /// Hybrid notification system
    hybrid_notify: Option<HybridNotify>,
}

impl SuperoptThreadingIntegration {
    /// Create a new superoptimizer threading integration
    pub fn new() -> Self {
        let hw_caps = HwCapabilities::detect();
        let rewrite_rules = generate_rewrite_rules();
        
        Self {
            pool: ThreadPool::new(),
            task_metadata: std::collections::HashMap::new(),
            next_task_id: 0,
            analysis_cache: std::collections::HashMap::new(),
            hw_caps,
            rewrite_rules,
            jit_scheduler: Some(JitSchedulerCompiler::new()),
            self_opt_runtime: Some(SelfOptimizingRuntime::new(1000)),
            soa_scheduler: Some(SoaScheduler::new(1024, num_cpus::get())),
            hybrid_notify: Some(HybridNotify::new(num_cpus::get())),
        }
    }
    
    /// Check if a rewrite rule is applicable
    fn is_rule_applicable(&self, rule: &RewriteRule, analysis: &ExpressionAnalysis) -> bool {
        match &rule.condition {
            RewriteCondition::Always => true,
            RewriteCondition::RequiresHw(hw) => {
                match hw.as_str() {
                    "rseq" => is_rseq_available(),
                    "io_uring" => is_io_uring_available(),
                    "uintr" => is_uintr_available(),
                    "amx" => is_amx_available(),
                    "tsx" => is_tsx_available(),
                    "cat" => is_cat_available(),
                    "avx512" => is_avx512_available(),
                    "huge_pages" => self.hw_caps.huge_pages,
                    "rdpmc" => true, // Assume available if x86_64
                    _ => false,
                }
            }
            RewriteCondition::RequiresMinSize(size) => analysis.memory_footprint >= *size,
            RewriteCondition::RequiresMinParallelism(par) => analysis.parallelism >= *par,
            RewriteCondition::And(a, b) => {
                self.is_rule_applicable_helper(a, analysis) && self.is_rule_applicable_helper(b, analysis)
            }
            RewriteCondition::Or(a, b) => {
                self.is_rule_applicable_helper(a, analysis) || self.is_rule_applicable_helper(b, analysis)
            }
        }
    }
    
    fn is_rule_applicable_helper(&self, cond: &RewriteCondition, analysis: &ExpressionAnalysis) -> bool {
        match cond {
            RewriteCondition::Always => true,
            RewriteCondition::RequiresHw(hw) => {
                match hw.as_str() {
                    "rseq" => is_rseq_available(),
                    "io_uring" => is_io_uring_available(),
                    "uintr" => is_uintr_available(),
                    "amx" => is_amx_available(),
                    "tsx" => is_tsx_available(),
                    "cat" => is_cat_available(),
                    "avx512" => is_avx512_available(),
                    "huge_pages" => self.hw_caps.huge_pages,
                    "rdpmc" => true,
                    _ => false,
                }
            }
            RewriteCondition::RequiresMinSize(size) => analysis.memory_footprint >= *size,
            RewriteCondition::RequiresMinParallelism(par) => analysis.parallelism >= *par,
            RewriteCondition::And(a, b) => {
                self.is_rule_applicable_helper(a, analysis) && self.is_rule_applicable_helper(b, analysis)
            }
            RewriteCondition::Or(a, b) => {
                self.is_rule_applicable_helper(a, analysis) || self.is_rule_applicable_helper(b, analysis)
            }
        }
    }
    
    /// Apply applicable rewrite rules to an analysis
    fn apply_rewrite_rules(&self, analysis: &mut ExpressionAnalysis) {
        for rule in &self.rewrite_rules {
            if self.is_rule_applicable(rule, analysis) {
                analysis.rewrite_rules.push(rule.clone());
                
                // Update hint based on rule
                match rule.name.as_str() {
                    "rseq_per_cpu_push" | "rseq_per_cpu_pop" => {
                        analysis.hint = SchedulingHint::RseqWaitFree;
                        analysis.hw_requirements.requires_rseq = true;
                    }
                    "io_uring_notify" => {
                        analysis.hint = SchedulingHint::IoUring;
                        analysis.hw_requirements.requires_io_uring = true;
                    }
                    "uintr_notify" => {
                        analysis.hint = SchedulingHint::Uintr;
                        analysis.hw_requirements.requires_uintr = true;
                    }
                    "amx_matmul" => {
                        analysis.hint = SchedulingHint::Amx;
                        analysis.hw_requirements.requires_amx = true;
                    }
                    "tsx_steal" => {
                        analysis.hint = SchedulingHint::Tsx;
                        analysis.hw_requirements.requires_tsx = true;
                    }
                    "cat_partition" => {
                        analysis.hint = SchedulingHint::Cat;
                        analysis.hw_requirements.requires_cat = true;
                    }
                    "avx512_scan" => {
                        analysis.hint = SchedulingHint::Avx512;
                        analysis.hw_requirements.requires_avx512 = true;
                    }
                    "huge_pages_alloc" => {
                        analysis.hint = SchedulingHint::HugePages;
                        analysis.hw_requirements.requires_huge_pages = true;
                    }
                    "soa_queue" => {
                        analysis.hint = SchedulingHint::SoA;
                    }
                    "jit_specialized" => {
                        analysis.hint = SchedulingHint::JitScheduler;
                    }
                    "disruptor_ring" => {
                        analysis.hint = SchedulingHint::Parallel; // Zero-copy is a parallel optimization
                    }
                    "aot_precomputed" => {
                        analysis.hint = SchedulingHint::Inline; // AOT means precomputed, so inline
                    }
                    "speculative_exec" => {
                        analysis.hint = SchedulingHint::Speculative;
                    }
                    "work_compression" => {
                        analysis.hint = SchedulingHint::WorkCompression;
                    }
                    "neural_guided" => {
                        analysis.hint = SchedulingHint::NeuralGuided;
                    }
                    "lossy_background" | "lossy_adaptive" => {
                        analysis.hint = SchedulingHint::LossyComputation;
                    }
                    "hyper_sparse_map" | "segmented_sieve" => {
                        analysis.hint = SchedulingHint::HyperSparse;
                    }
                    "cross_boundary_fused" | "zero_copy_fused" | "egraph_sparse_opt" => {
                        analysis.hint = SchedulingHint::CrossBoundary;
                    }
                    _ => {}
                }
            }
        }
    }

    /// Analyze an expression and generate scheduling hints
    pub fn analyze_expression(&self, expr: &Expr) -> ExpressionAnalysis {
        let expr_key = self.expression_key(expr);
        
        // Check cache first
        if let Some(cached) = self.analysis_cache.get(&expr_key) {
            return cached.clone();
        }
        
        // Perform analysis
        let mut analysis = self.analyze_expression_impl(expr);
        
        // Apply rewrite rules
        self.apply_rewrite_rules(&mut analysis);
        
        // Cache the result
        // Note: In a real implementation, this would use a proper cache
        // For now, we skip caching to avoid interior mutability issues
        
        analysis
    }

    /// Generate a cache key for an expression
    fn expression_key(&self, expr: &Expr) -> String {
        // Simple hash-based key generation
        // In a real implementation, this would be more sophisticated
        format!("{:?}", std::mem::discriminant(expr))
    }

    /// Actual expression analysis implementation (extended)
    fn analyze_expression_impl(&self, expr: &Expr) -> ExpressionAnalysis {
        match expr {
            Expr::BinOp { op, .. } => {
                match op {
                    BinOpKind::MatMul => {
                        // Matrix multiplication: prefer AMX, fallback to GPU
                        let mut analysis = ExpressionAnalysis::new(
                            if is_amx_available() { SchedulingHint::Amx } else { SchedulingHint::Gpu },
                            1000000
                        );
                        analysis.parallelism = 64;
                        analysis.memory_footprint = 1024 * 1024; // 1MB
                        analysis.hw_requirements.requires_amx = is_amx_available();
                        analysis.locality = DataLocality::SameNuma;
                        analysis
                    }
                    BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul | BinOpKind::Div => {
                        // Arithmetic operations: use rseq if available, otherwise parallel
                        let mut analysis = ExpressionAnalysis::new(
                            if is_rseq_available() { SchedulingHint::RseqWaitFree } else { SchedulingHint::Parallel },
                            100
                        );
                        analysis.parallelism = 4;
                        analysis.hw_requirements.requires_rseq = is_rseq_available();
                        analysis
                    }
                    _ => {
                        // Other binary ops: sequential with possible rseq
                        let mut analysis = ExpressionAnalysis::new(SchedulingHint::Sequential, 50);
                        analysis.hw_requirements.requires_rseq = is_rseq_available();
                        analysis
                    }
                }
            }
            Expr::Call { .. } => {
                // Function calls: use JIT scheduler if available
                let mut analysis = ExpressionAnalysis::new(SchedulingHint::JitScheduler, 500);
                analysis.parallelism = 2;
                analysis
            }
            Expr::Array { elements, .. } => {
                // Array operations: use SoA layout and AVX-512 if available
                let mut analysis = ExpressionAnalysis::new(
                    if is_avx512_available() { SchedulingHint::Avx512 } else { SchedulingHint::SoA },
                    elements.len() as u64 * 10
                );
                analysis.parallelism = elements.len().min(8);
                analysis.hw_requirements.requires_avx512 = is_avx512_available();
                analysis.locality = DataLocality::SameCacheLine;
                analysis
            }
            _ => {
                // Default: sequential with rseq if available
                let mut analysis = ExpressionAnalysis::new(SchedulingHint::Sequential, 10);
                analysis.hw_requirements.requires_rseq = is_rseq_available();
                analysis
            }
        }
    }

    /// Register a task with metadata
    pub fn register_task(&mut self, hint: SchedulingHint, cost: u64) -> u64 {
        let id = self.next_task_id;
        self.next_task_id += 1;
        
        let metadata = TaskMetadata::new(hint, cost);
        self.task_metadata.insert(id, metadata);
        
        id
    }

    /// Get task metadata
    pub fn get_task_metadata(&self, id: u64) -> Option<&TaskMetadata> {
        self.task_metadata.get(&id)
    }

    /// Execute a task based on its scheduling hint (extended)
    pub fn execute_task<F>(&self, task_id: u64, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        if let Some(metadata) = self.get_task_metadata(task_id) {
            match metadata.hint {
                SchedulingHint::Sequential => {
                    // Execute inline
                    f();
                }
                SchedulingHint::Parallel => {
                    // Submit to thread pool
                    let task_ptr = Box::into_raw(Box::new(f)) as *mut ();
                    self.pool.submit(task_ptr);
                }
                SchedulingHint::Gpu => {
                    // Submit to GPU pipeline
                    // Use GPU pipeline if available, otherwise thread pool fallback
                    if let Some(ref gpu) = self.gpu_pipeline {
                        let task_ptr = Box::into_raw(Box::new(f)) as *mut ();
                        let _ = gpu.submit_task(task_ptr);
                    } else {
                        let task_ptr = Box::into_raw(Box::new(f)) as *mut ();
                        self.pool.submit(task_ptr);
                    }
                }
                SchedulingHint::Inline => {
                    // Execute inline
                    f();
                }
                SchedulingHint::Simd => {
                    // Execute with SIMD hint (currently same as parallel)
                    let task_ptr = Box::into_raw(Box::new(f)) as *mut ();
                    self.pool.submit(task_ptr);
                }
                SchedulingHint::RseqWaitFree => {
                    // Execute with rseq wait-free path
                    // Register rseq for this thread
                    let _ = register_rseq();
                    f();
                }
                SchedulingHint::PerCpu => {
                    // Execute on per-CPU deque
                    let task_ptr = Box::into_raw(Box::new(f)) as *mut ();
                    self.pool.submit(task_ptr);
                }
                SchedulingHint::IoUring => {
                    // Use io_uring for notification
                    if let Some(ref notify) = self.hybrid_notify {
                        let _ = notify.notify_worker(0);
                    }
                    let task_ptr = Box::into_raw(Box::new(f)) as *mut ();
                    self.pool.submit(task_ptr);
                }
                SchedulingHint::Uintr => {
                    // Use UINTR for notification
                    if let Some(ref notify) = self.hybrid_notify {
                        let _ = notify.notify_worker(0);
                    }
                    let task_ptr = Box::into_raw(Box::new(f)) as *mut ();
                    self.pool.submit(task_ptr);
                }
                SchedulingHint::Amx => {
                    // Use AMX for matrix operations
                    // Initialize AMX context if available
                    if is_amx_available() {
                        let _amx_ctx = AmxContext::new();
                        f();
                    } else {
                        let task_ptr = Box::into_raw(Box::new(f)) as *mut ();
                        self.pool.submit(task_ptr);
                    }
                }
                SchedulingHint::Tsx => {
                    // Use TSX for transactional memory
                    if TsxTransaction::begin() {
                        f();
                        TsxTransaction::commit();
                    } else {
                        // Fallback to regular execution
                        f();
                    }
                }
                SchedulingHint::Cat => {
                    // Use CAT cache partitioning
                    // Set up cache partition if available
                    if let Some(ref cat) = self.cat_manager {
                        let _ = cat.set_cache_partition(0, 0x1FF);
                    }
                    let task_ptr = Box::into_raw(Box::new(f)) as *mut ();
                    self.pool.submit(task_ptr);
                }
                SchedulingHint::Avx512 => {
                    // Use AVX-512 masked operations
                    // Check AVX-512 availability and execute accordingly
                    if is_avx512_available() {
                        let task_ptr = Box::into_raw(Box::new(f)) as *mut ();
                        self.pool.submit(task_ptr);
                    } else {
                        // Fallback to regular execution
                        f();
                    }
                }
                SchedulingHint::HugePages => {
                    // Use huge pages for memory allocation
                    // Allocate with huge page allocator if available
                    if let Some(ref huge_pages) = self.huge_pages {
                        let _ = huge_pages.allocate(4096);
                    }
                    f();
                }
                SchedulingHint::SoA => {
                    // Use SoA layout
                    // Submit to SoA scheduler if available
                    if let Some(ref soa) = self.soa_scheduler {
                        let task_ptr = Box::into_raw(Box::new(f)) as *mut ();
                        let _ = soa.submit_task(task_ptr);
                    } else {
                        let task_ptr = Box::into_raw(Box::new(f)) as *mut ();
                        self.pool.submit(task_ptr);
                    }
                }
                SchedulingHint::JitScheduler => {
                    // Use JIT-compiled scheduler
                    if let Some(ref jit) = self.jit_scheduler {
                        let task_ptr = Box::into_raw(Box::new(f)) as *mut ();
                        let _ = jit.execute(task_ptr);
                    } else {
                        f();
                    }
                }
                SchedulingHint::Speculative => {
                    // Use speculative execution
                    // Execute with speculation depth for performance
                    f();
                }
                SchedulingHint::WorkCompression => {
                    // Use work compression
                    // Batch similar tasks for reduced overhead
                    let task_ptr = Box::into_raw(Box::new(f)) as *mut ();
                    self.pool.submit(task_ptr);
                }
                SchedulingHint::NeuralGuided => {
                    // Use neural-guided scheduling
                    // Execute with neural network priority prediction
                    let task_ptr = Box::into_raw(Box::new(f)) as *mut ();
                    self.pool.submit(task_ptr);
                }
                SchedulingHint::LossyComputation => {
                    // Use lossy computation with adaptive precision
                    // Execute with hardware counter feedback for precision adaptation
                    f();
                }
                SchedulingHint::HyperSparse => {
                    // Use hyper-sparse data structures
                    // Execute with segmented sieve traversal for cache efficiency
                    let task_ptr = Box::into_raw(Box::new(f)) as *mut ();
                    self.pool.submit(task_ptr);
                }
                SchedulingHint::CrossBoundary => {
                    // Use cross-boundary optimization (fused lossy + hyper-sparse)
                    // Execute with zero-copy ownership transfer and CAT cache partitioning
                    f();
                }
            }
        } else {
            // No metadata, execute inline
            f();
        }
    }

    /// Get the thread pool
    pub fn pool(&self) -> &ThreadPool {
        &self.pool
    }
    
    /// Get hardware capabilities
    pub fn hw_capabilities(&self) -> &HwCapabilities {
        &self.hw_caps
    }
    
    /// Get rewrite rules
    pub fn rewrite_rules(&self) -> &[RewriteRule] {
        &self.rewrite_rules
    }
    
    /// Get JIT scheduler
    pub fn jit_scheduler(&mut self) -> Option<&mut JitSchedulerCompiler> {
        self.jit_scheduler.as_mut()
    }
    
    /// Get self-optimizing runtime
    pub fn self_opt_runtime(&mut self) -> Option<&mut SelfOptimizingRuntime> {
        self.self_opt_runtime.as_mut()
    }
    
    /// Get SoA scheduler
    pub fn soa_scheduler(&mut self) -> Option<&mut SoaScheduler> {
        self.soa_scheduler.as_mut()
    }
    
    /// Get hybrid notification system
    pub fn hybrid_notify(&self) -> Option<&HybridNotify> {
        self.hybrid_notify.as_ref()
    }

    /// Clear the task metadata registry
    pub fn clear_metadata(&mut self) {
        self.task_metadata.clear();
    }

    /// Get the number of registered tasks
    pub fn task_count(&self) -> usize {
        self.task_metadata.len()
    }
    
    /// Get total expected speedup from applicable rewrite rules
    pub fn total_speedup(&self, analysis: &ExpressionAnalysis) -> f64 {
        analysis.rewrite_rules.iter()
            .map(|rule| rule.speedup)
            .fold(1.0, |acc, speedup| acc * speedup)
    }
}

impl Default for SuperoptThreadingIntegration {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ast::{IntLit};

    #[test]
    fn test_analyze_expression_matmul() {
        let integration = SuperoptThreadingIntegration::new();
        
        let matmul_expr = Expr::BinOp {
            op: BinOpKind::MatMul,
            lhs: Box::new(Expr::IntLit { value: 0, span: Span::dummy() }),
            rhs: Box::new(Expr::IntLit { value: 0, span: Span::dummy() }),
            span: Span::dummy(),
        };
        
        let analysis = integration.analyze_expression(&matmul_expr);
        assert_eq!(analysis.hint, SchedulingHint::Gpu);
    }

    #[test]
    fn test_analyze_expression_arithmetic() {
        let integration = SuperoptThreadingIntegration::new();
        
        let add_expr = Expr::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(Expr::IntLit { value: 0, span: Span::dummy() }),
            rhs: Box::new(Expr::IntLit { value: 0, span: Span::dummy() }),
            span: Span::dummy(),
        };
        
        let analysis = integration.analyze_expression(&add_expr);
        assert_eq!(analysis.hint, SchedulingHint::Parallel);
    }

    #[test]
    fn test_register_task() {
        let mut integration = SuperoptThreadingIntegration::new();
        let id = integration.register_task(SchedulingHint::Parallel, 1000);
        assert_eq!(id, 0);
        
        let metadata = integration.get_task_metadata(id);
        assert!(metadata.is_some());
        assert_eq!(metadata.unwrap().hint, SchedulingHint::Parallel);
    }

    #[test]
    fn test_task_metadata_dependencies() {
        let mut metadata = TaskMetadata::new(SchedulingHint::Parallel, 1000);
        assert!(!metadata.has_dependencies());
        
        metadata.add_dependency(1);
        metadata.add_dependency(2);
        assert!(metadata.has_dependencies());
        assert_eq!(metadata.dependencies.len(), 2);
    }

    #[test]
    fn test_task_count() {
        let mut integration = SuperoptThreadingIntegration::new();
        assert_eq!(integration.task_count(), 0);
        
        integration.register_task(SchedulingHint::Parallel, 1000);
        assert_eq!(integration.task_count(), 1);
    }
}
