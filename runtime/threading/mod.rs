// =========================================================================
// Custom Multi-Thread Engine for Jules
// NUMA-aware work-stealing scheduler with zero-allocation tasks
// =========================================================================

pub mod affinity;
pub mod deque;
pub mod ecs_lockfree;
pub mod epoch;
pub mod green;
pub mod gpu_pipeline;
pub mod join;
pub mod numa;
pub mod slab;
pub mod superopt_integration;
pub mod worker;

pub use affinity::{set_thread_affinity, set_thread_affinity_for_thread};
pub use deque::WorkStealingDeque;
pub use ecs_lockfree::{ComponentStorage, EntityId, LockFreeComponentStorage, SparseSet};
pub use green::{GreenContext, GreenScheduler, GreenThreadId};
pub use gpu_pipeline::{GpuPipeline, GpuTaskHandle};
pub use join::{join, JoinHandle};
pub use numa::{NumaNode, NumaTopology};
pub use slab::SlabAllocator;
pub use superopt_integration::{SchedulingHint, SuperoptThreadingIntegration, TaskMetadata};
pub use worker::{ThreadPool, Worker};

/// Result type for threading operations
pub type ThreadResult<T> = Result<T, String>;
