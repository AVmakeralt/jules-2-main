// =========================================================================
// Custom Multi-Thread Engine for Jules
// NUMA-aware work-stealing scheduler with zero-allocation tasks
// =========================================================================

pub mod affinity;
pub mod cross_boundary;
pub mod deque;
pub mod disruptor;
pub mod ecs_lockfree;
pub mod egraph_schedule;
pub mod epoch;
pub mod green;
pub mod gpu_pipeline;
pub mod hyper_sparse;
pub mod novel_scheduling;
pub mod numa;
pub mod hw_optimizations;
pub mod jit_scheduler;
pub mod join;
pub mod kernel_bypass;
pub mod lossy_computation;
pub mod rseq;
pub mod rseq_integration;
pub mod worker;
pub mod superopt_integration;
pub mod slab;
pub mod percpu_deque;
pub mod soa_queue;
pub mod stack_task;
pub mod worker;

pub use affinity::{set_thread_affinity, set_thread_affinity_for_thread};
pub use cross_boundary::{CrossBoundaryOptimizer, FusedOperation, FusedOperationBuilder, OptimizationResult, ZeroCopyTransfer};
pub use deque::WorkStealingDeque;
pub use disruptor::{DisruptorRing, OwnedData, WorkerDisruptor, ZeroCopyMessaging};
pub use ecs_lockfree::{ComponentStorageData, ComponentStorageTrait, EntityId, LockFreeComponentStorage, SparseSet};
pub use egraph_schedule::{AotScheduler, EGraph, EGraphBuilder, EGraphExtractor, EGraphNode, PrecomputedSchedule};
pub use green::{GreenContext, GreenScheduler, GreenThreadId};
pub use gpu_pipeline::{GpuPipeline, GpuTaskHandle};
pub use hw_optimizations::{AmxContext, Avx512Mask, CatManager, CompareOp, HugePageAllocator, HwCapabilities, TsxTransaction, avx512_conflict_detection, is_amx_available, is_avx512_available, is_cat_available, is_tsx_available};
pub use hyper_sparse::{BitSegment, HyperSparseMap, HyperSparseSoA, SegmentedSieve};
pub use jit_scheduler::{HwCounter, HwCounterReader, HwCounterValue, JitSchedulerCompiler, RuntimeStats, SchedulingStrategy, SelfOptimizingRuntime, TraceBasedScheduler, WorkloadPhase};
pub use join::{join, JoinHandle};
pub use kernel_bypass::{HybridNotify, IoUring, UintrReceiver, UintrSender, is_io_uring_available, is_uintr_available};
pub use lossy_computation::{HwCounterFeedback, LossyComputationContext, LossyComputationManager, LossyFastPath, PrecisionLevel, TaskPriority};
pub use novel_scheduling::{CompressedTask, NeuralScheduler, NovelSchedulingEngine, SpeculativeExecutor, SpeculativeTask, TaskFeatures, TrainingSample, WorkCompressor};
pub use numa::{NumaNode, NumaTopology};
pub use percpu_deque::PerCpuDeque;
pub use rseq::{PerCpu, PerCpuCounter, get_cpu_id, is_rseq_available, register_rseq};
pub use rseq_integration::{RseqAtomicCounter, RseqManager, RseqMutex, RseqRingBuffer, RseqStats, global_rseq_manager, init_current_thread};
pub use slab::SlabAllocator;
pub use soa_queue::{DataAffinityMap, SoaScheduler, SoaTaskQueue, warm_function_cache, warm_task_cache};
pub use stack_task::{StackTask, TaskBatch, TaskCache, TASK_TYPE_COMPUTE, TASK_TYPE_IO, TASK_TYPE_GPU, TASK_TYPE_SYNC};
pub use superopt_integration::{DataLocality, ExpressionAnalysis, HwRequirements, RewriteCondition, RewriteRule, SchedulingHint, SuperoptThreadingIntegration, TaskMetadata, generate_rewrite_rules};
pub use worker::{ThreadPool, Worker};

/// Result type for threading operations
pub type ThreadResult<T> = Result<T, String>;
