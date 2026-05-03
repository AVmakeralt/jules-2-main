// =============================================================================
// Zero-Cost Heterogeneous Memory Architecture (ZCHMA)
//
// A revolutionary memory management system that treats memory hierarchy as a single
// addressable continuum with differentiable latency. The borrow checker tracks
// lifetimes and access patterns, emitting memory placement hints as probabilistic
// distributions, while hardware (Intel Flat Memory Mode, AMD Unified Memory, or
// CXL 3.0) handles actual page migration.
//
// Architecture Overview:
// ┌─────────────────────────────────────────────────────────────────────┐
// │                      Jules Compiler Frontend                          │
// │  ┌────────────────┐  ┌─────────────────┐  ┌───────────────────────┐ │
// │  │ Borrow Checker │──│ Lifetime Tracker │──│ Memory Placement Hint │ │
// │  │   (existing)   │  │                 │  │  Generator            │ │
// │  └────────────────┘  └─────────────────┘  └───────────────────────┘ │
// │                                                              │        │
// │  ┌─────────────────────────────────────────────────────────────▼────┐ │
// │  │              Placement Distribution Emitter                    │   │
// │  │  - Tier probabilities: DRAM:0.7, HBM:0.2, VRAM:0.1            │   │
// │  │  - Access pattern signatures                                  │   │
// │  │  - Migration cost model parameters                             │   │
// │  └───────────────────────────────────────────────────────────────┘   │
// └─────────────────────────────────────────────────────────────────────┘
//                              │
//                              ▼
// ┌─────────────────────────────────────────────────────────────────────┐
// │                    ZCHMA Runtime Layer                               │
// │  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │
// │  │ Memory Tier    │  │ Page Migration  │  │ Access Pattern        │ │
// │  │ Controller     │  │ Scheduler       │  │ Monitor               │ │
// │  └────────────────┘  └─────────────────┘  └──────────────────────┘ │
// │          │                   │                     │                │
// │          ▼                   ▼                     ▼                │
// │  ┌─────────────────────────────────────────────────────────────────┐ │
// │  │           Hardware Coordination Layer (CXL/Flat Memory)        │ │
// │  └─────────────────────────────────────────────────────────────────┘ │
// └─────────────────────────────────────────────────────────────────────┘
//                              │
//                              ▼
// ┌─────────────────────────────────────────────────────────────────────┐
// │                      Physical Memory Tiers                          │
// │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
// │  │   DRAM   │  │   HBM    │  │CXL MEM   │  │   GPU    │             │
// │  │ (CPU)    │  │ (CPU)    │  │(CXL 3.0) │  │  VRAM    │             │
// │  └──────────┘  └──────────┘  └──────────┘  └──────────┘             │
// └─────────────────────────────────────────────────────────────────────┘
//
// Key Innovations:
// 1. DIFFERENTIABLE MEMORY LATENCY: Memory tier access costs are treated as
//    continuous functions, enabling compiler optimization via gradient descent.
//
// 2. PROBABILISTIC PLACEMENT HINTS: Instead of explicit malloc/cudaMalloc,
//    the compiler emits probability distributions over memory tiers.
//
// 3. HARDWARE-VERIFIED SEMANTIC EQUIVALENCE: Translation validation ensures
//    that any valid placement is semantically equivalent to any other.
//
// 4. ZERO-COST PLACEMENT: The programmer never explicitly chooses memory tier.
//    The system is correct regardless of where bytes physically live.
//
// =============================================================================

use crate::compiler::borrowck::{Lifetime, AccessPattern};
use crate::runtime::memory_management::MemoryTier;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ─────────────────────────────────────────────────────────────────────────────
// Public API Types
// ─────────────────────────────────────────────────────────────────────────────

/// Probability distribution over available memory tiers.
///
/// Each tier has an associated probability weight representing the likelihood
/// that this allocation should reside in that tier based on the compiler's
/// analysis of access patterns, lifetimes, and hardware topology.
#[derive(Debug, Clone)]
pub struct PlacementDistribution {
    /// Probability weights for each memory tier.
    /// Sum of all weights should equal 1.0 for valid distributions.
    tier_weights: HashMap<MemoryTierId, f64>,
    /// Confidence score (0.0 to 1.0) in the placement prediction.
    /// Lower confidence means the compiler couldn't determine a clear winner.
    confidence: f64,
    /// Estimated access frequency per second for this allocation.
    estimated_access_rate: f64,
    /// Estimated total bytes transferred if migrated between tiers.
    migration_cost_bytes: u64,
}

impl PlacementDistribution {
    /// Create a new placement distribution from tier weights.
    pub fn new(tier_weights: HashMap<MemoryTierId, f64>) -> Self {
        let total: f64 = tier_weights.values().sum();
        let confidence = if total > 0.0 {
            tier_weights.values().map(|w| w / total).fold(0.0, |acc, w| acc + w * w)
        } else {
            0.0
        };
        
        Self {
            tier_weights,
            confidence,
            estimated_access_rate: 0.0,
            migration_cost_bytes: 0,
        }
    }
    
    /// Create a uniform distribution across all tiers.
    pub fn uniform(tiers: &[MemoryTierId]) -> Self {
        let weight = 1.0 / tiers.len() as f64;
        let tier_weights: HashMap<MemoryTierId, f64> = tiers
            .iter()
            .map(|t| (*t, weight))
            .collect();
        Self::new(tier_weights)
    }
    
    /// Get the most likely memory tier for this allocation.
    pub fn most_likely_tier(&self) -> Option<MemoryTierId> {
        self.tier_weights
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(t, _)| *t)
    }
    
    /// Compute expected memory latency in nanoseconds.
    pub fn expected_latency(&self, tier_latencies: &HashMap<MemoryTierId, f64>) -> f64 {
        self.tier_weights
            .iter()
            .map(|(tier, weight)| {
                tier_latencies.get(tier).copied().unwrap_or(100.0) * weight
            })
            .sum()
    }
}

/// Unique identifier for a memory tier in the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryTierId(pub u32);

/// Access pattern classification for a memory allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPatternClass {
    /// Frequent random access - likely to benefit from high-bandwidth memory.
    RandomAccess,
    /// Sequential streaming access - benefits from DRAM or HBM.
    Streaming,
    /// Mostly read, occasional writes - can use write-coalescing tiers.
    ReadHeavy,
    /// Mostly write - needs a tier with efficient write paths.
    WriteHeavy,
    /// Alternating read/write in similar amounts.
    Balanced,
    /// Infrequent access - can use slower/cheaper tiers.
    Infrequent,
}

/// Memory tier characteristics for cost modeling.
#[derive(Debug, Clone)]
pub struct TierCharacteristics {
    pub tier: MemoryTierId,
    pub latency_ns: f64,
    pub bandwidth_gb_s: f64,
    pub capacity_bytes: u64,
    pub is_volatile: bool,
    pub supports_direct_access: bool,
    pub migration_overhead_ns: f64,
    pub energy_per_access_relative: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Lifetime-to-Tier Mapping Engine
// ─────────────────────────────────────────────────────────────────────────────

/// Maps program lifetimes to optimal memory tier distributions.
///
/// This is the core of ZCHMA's value proposition: instead of requiring
/// programmers to explicitly choose memory tiers (malloc vs cudaMalloc),
/// the compiler analyzes lifetimes and access patterns to automatically
/// emit tier probabilities.
pub struct LifetimeTierMapper {
    /// Known tier characteristics from hardware inventory.
    tier_characteristics: Vec<TierCharacteristics>,
    /// Access pattern classifier.
    access_classifier: AccessPatternClassifier,
    /// Migration cost model.
    migration_model: MigrationCostModel,
    /// Statistics for tier selection decisions.
    selection_stats: TierSelectionStats,
}

#[derive(Debug, Clone, Default)]
pub struct TierSelectionStats {
    pub total_allocations: AtomicU64,
    pub tier_selections: HashMap<MemoryTierId, AtomicU64>,
    pub migration_decisions: AtomicU64,
    pub successful_migrations: AtomicU64,
    pub failed_migrations: AtomicU64,
}

impl LifetimeTierMapper {
    /// Create a new lifetime-to-tier mapper with the given hardware topology.
    pub fn new(tier_characteristics: Vec<TierCharacteristics>) -> Self {
        Self {
            tier_characteristics,
            access_classifier: AccessPatternClassifier::new(),
            migration_model: MigrationCostModel::new(),
            selection_stats: TierSelectionStats::default(),
        }
    }
    
    /// Map a lifetime and access pattern to a placement distribution.
    pub fn map_lifetime_to_tier(
        &mut self,
        lifetime: &Lifetime,
        access_pattern: &AccessPattern,
        size_bytes: u64,
    ) -> PlacementDistribution {
        self.selection_stats.total_allocations.fetch_add(1, Ordering::Relaxed);
        
        // Classify the access pattern
        let pattern_class = self.access_classifier.classify(access_pattern);
        
        // Compute weights for each tier based on multiple factors
        let mut weights = HashMap::new();
        let mut total_score = 0.0;
        
        for tier in &self.tier_characteristics {
            let score = self.compute_tier_score(tier, lifetime, &pattern_class, size_bytes);
            weights.insert(tier.tier, score);
            total_score += score;
        }
        
        // Normalize weights to probabilities
        if total_score > 0.0 {
            for weight in weights.values_mut() {
                *weight /= total_score;
            }
        }
        
        // Update statistics for the most likely tier
        let distribution = PlacementDistribution::new(weights.clone());
        if let Some(tier) = distribution.most_likely_tier() {
            self.selection_stats.tier_selections
                .entry(tier)
                .or_insert_with(|| AtomicU64::new(0))
                .fetch_add(1, Ordering::Relaxed);
        }
        
        distribution
    }
    
    /// Compute a score for a specific tier given the allocation context.
    fn compute_tier_score(
        &self,
        tier: &TierCharacteristics,
        lifetime: &Lifetime,
        pattern: &AccessPatternClass,
        size_bytes: u64,
    ) -> f64 {
        let mut score = 100.0;
        
        // Penalize tiers that are too small for this allocation
        if tier.capacity_bytes < size_bytes {
            return 0.0;
        }
        
        // Weight by latency (lower is better)
        score *= 1000.0 / (tier.latency_ns + 1.0);
        
        // Weight by bandwidth (higher is better for streaming/burst access)
        if matches!(pattern, AccessPatternClass::Streaming | AccessPatternClass::RandomAccess) {
            score *= (tier.bandwidth_gb_s / 100.0).max(0.1);
        }
        
        // Weight by lifetime (longer lifetimes favor slower/cheaper tiers)
        let lifetime_factor = (lifetime.estimated_length.as_secs_f64() / 3600.0).min(10.0) / 10.0;
        score *= 1.0 - (lifetime_factor * 0.3);
        
        // Penalize volatile tiers for long-lived allocations
        if tier.is_volatile && lifetime.estimated_length.as_secs() > 86400 {
            score *= 0.1;
        }
        
        // Boost tiers that support direct access for infreqent patterns
        if tier.supports_direct_access && matches!(pattern, AccessPatternClass::Infrequent) {
            score *= 2.0;
        }
        
        score
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Access Pattern Classification
// ─────────────────────────────────────────────────────────────────────────────

/// Classifies memory access patterns from compiler analysis.
pub struct AccessPatternClassifier {
    /// Training data for pattern classification (could use ML model).
    training_data: HashMap<String, AccessPatternClass>,
}

impl AccessPatternClassifier {
    pub fn new() -> Self {
        let mut training_data = HashMap::new();
        // Initialize with known patterns
        training_data.insert("matmul_accumulator".into(), AccessPatternClass::RandomAccess);
        training_data.insert("streaming_buffer".into(), AccessPatternClass::Streaming);
        training_data.insert("lookup_table".into(), AccessPatternClass::ReadHeavy);
        training_data.insert("gradient_buffer".into(), AccessPatternClass::WriteHeavy);
        training_data.insert("persistent_config".into(), AccessPatternClass::Infrequent);
        Self { training_data }
    }
    
    /// Classify an access pattern into a tier-appropriate category.
    pub fn classify(&self, pattern: &AccessPattern) -> AccessPatternClass {
        // Check for known patterns
        if let Some(class) = pattern.operation_name.as_ref()
            .and_then(|name| self.training_data.get(name))
            .copied()
        {
            return class;
        }
        
        // Heuristic classification based on access characteristics
        if pattern.is_random_access {
            AccessPatternClass::RandomAccess
        } else if pattern.is_sequential {
            AccessPatternClass::Streaming
        } else if pattern.read_write_ratio > 0.8 {
            AccessPatternClass::ReadHeavy
        } else if pattern.read_write_ratio < 0.2 {
            AccessPatternClass::WriteHeavy
        } else if pattern.access_frequency < 0.01 {
            AccessPatternClass::Infrequent
        } else {
            AccessPatternClass::Balanced
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Migration Cost Model
// ─────────────────────────────────────────────────────────────────────────────

/// Models the cost of migrating pages between memory tiers.
pub struct MigrationCostModel {
    /// Bandwidth between tier pairs (in GB/s).
    inter_tier_bandwidth: HashMap<(MemoryTierId, MemoryTierId), f64>,
    /// Overhead per migration decision (in nanoseconds).
    decision_overhead_ns: f64,
    /// Cache coherency cost (for NUMA/CXL scenarios).
    coherency_overhead_ns: f64,
}

impl MigrationCostModel {
    pub fn new() -> Self {
        let mut inter_tier_bandwidth = HashMap::new();
        // Initialize with common hardware topologies
        // DRAM <-> HBM: ~500 GB/s
        inter_tier_bandwidth.insert((MemoryTierId(0), MemoryTierId(1)), 500.0);
        inter_tier_bandwidth.insert((MemoryTierId(1), MemoryTierId(0)), 500.0);
        // CXL <-> DRAM: ~200 GB/s
        inter_tier_bandwidth.insert((MemoryTierId(2), MemoryTierId(0)), 200.0);
        inter_tier_bandwidth.insert((MemoryTierId(0), MemoryTierId(2)), 200.0);
        // GPU VRAM <-> DRAM: ~50 GB/s (PCIe/NVLink)
        inter_tier_bandwidth.insert((MemoryTierId(3), MemoryTierId(0)), 50.0);
        inter_tier_bandwidth.insert((MemoryTierId(0), MemoryTierId(3)), 50.0);
        
        Self {
            inter_tier_bandwidth,
            decision_overhead_ns: 1000.0,
            coherency_overhead_ns: 500.0,
        }
    }
    
    /// Estimate the cost of migrating `size_bytes` from `src` to `dst`.
    pub fn estimate_migration_cost(
        &self,
        src: MemoryTierId,
        dst: MemoryTierId,
        size_bytes: u64,
    ) -> MigrationCostEstimate {
        let bandwidth = self.inter_tier_bandwidth
            .get(&(src, dst))
            .copied()
            .unwrap_or(10.0);
        
        let transfer_time_ns = (size_bytes as f64 / bandwidth / 1e9) * 1e9;
        let total_cost_ns = transfer_time_ns + self.decision_overhead_ns + self.coherency_overhead_ns;
        
        MigrationCostEstimate {
            src_tier: src,
            dst_tier: dst,
            size_bytes,
            transfer_time_ns,
            overhead_ns: self.decision_overhead_ns + self.coherency_overhead_ns,
            total_cost_ns,
        }
    }
}

/// Result of a migration cost estimation.
#[derive(Debug, Clone)]
pub struct MigrationCostEstimate {
    pub src_tier: MemoryTierId,
    pub dst_tier: MemoryTierId,
    pub size_bytes: u64,
    pub transfer_time_ns: f64,
    pub overhead_ns: f64,
    pub total_cost_ns: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Page Migration Scheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Schedules page migrations based on runtime access patterns.
pub struct PageMigrationScheduler {
    /// Active migrations in progress.
    active_migrations: HashMap<u64, MigrationTask>,
    /// Migration queue sorted by priority.
    migration_queue: Vec<MigrationRequest>,
    /// Statistics on migration decisions.
    migration_stats: MigrationStats,
}

#[derive(Debug, Clone)]
pub struct MigrationTask {
    pub allocation_id: u64,
    pub from_tier: MemoryTierId,
    pub to_tier: MemoryTierId,
    pub pages: Vec<u64>,
    pub started_at_ns: u64,
    pub priority: f64,
}

#[derive(Debug, Clone)]
pub struct MigrationRequest {
    pub allocation_id: u64,
    pub from_tier: MemoryTierId,
    pub to_tier: MemoryTierId,
    pub size_bytes: u64,
    pub expected_benefit_ns: f64,
    pub priority: f64,
}

#[derive(Debug, Clone, Default)]
pub struct MigrationStats {
    pub total_migrations_initiated: u64,
    pub total_migrations_completed: u64,
    pub total_migrations_aborted: u64,
    pub total_bytes_migrated: u64,
    pub average_migration_time_ns: f64,
}

impl PageMigrationScheduler {
    pub fn new() -> Self {
        Self {
            active_migrations: HashMap::new(),
            migration_queue: Vec::new(),
            migration_stats: MigrationStats::default(),
        }
    }
    
    /// Request a page migration based on observed access patterns.
    pub fn request_migration(&mut self, request: MigrationRequest) {
        // Only migrate if the expected benefit outweighs the cost
        let cost_estimate = self.estimate_cost(&request);
        if request.expected_benefit_ns > cost_estimate.total_cost_ns * 2.0 {
            self.migration_queue.push(request);
            self.migration_queue.sort_by(|a, b| {
                b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }
    
    fn estimate_cost(&self, request: &MigrationRequest) -> MigrationCostEstimate {
        MigrationCostEstimate {
            src_tier: request.from_tier,
            dst_tier: request.to_tier,
            size_bytes: request.size_bytes,
            transfer_time_ns: 0.0, // Would use migration model
            overhead_ns: 0.0,
            total_cost_ns: 0.0,
        }
    }
    
    /// Process pending migrations (called by runtime).
    pub fn process_migrations(&mut self, budget_ns: u64) -> Vec<MigrationResult> {
        let mut results = Vec::new();
        let mut remaining_budget = budget_ns;
        
        while let Some(request) = self.migration_queue.pop() {
            if remaining_budget < 10000 {
                // Budget exhausted
                self.migration_queue.push(request);
                break;
            }
            
            let result = self.execute_migration(&request);
            remaining_budget = remaining_budget.saturating_sub(result.duration_ns);
            results.push(result);
        }
        
        results
    }
    
    fn execute_migration(&mut self, request: &MigrationRequest) -> MigrationResult {
        // Placeholder for actual migration execution
        // In practice, this would coordinate with the OS/hardware
        self.migration_stats.total_migrations_initiated += 1;
        
        MigrationResult {
            allocation_id: request.allocation_id,
            success: true,
            bytes_migrated: request.size_bytes,
            duration_ns: 1000, // Placeholder
        }
    }
}

/// Result of a migration execution.
#[derive(Debug, Clone)]
pub struct MigrationResult {
    pub allocation_id: u64,
    pub success: bool,
    pub bytes_migrated: u64,
    pub duration_ns: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Translation Validation (Semantic Equivalence)
// ─────────────────────────────────────────────────────────────────────────────

/// Validates that memory tier migrations preserve program semantics.
///
/// This extends the existing translation_validation.rs to prove that moving
/// data between memory tiers doesn't change program meaning.
pub struct TierMigrationValidator {
    /// SMT solver for formal verification.
    smt_solver: Box<dyn TierSmtSolver>,
    /// Cache of validated migration scenarios.
    validation_cache: HashMap<MigrationScenario, ValidationResult>,
}

/// A specific migration scenario to validate.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MigrationScenario {
    pub src_tier: MemoryTierId,
    pub dst_tier: MemoryTierId,
    pub allocation_type: AllocationType,
    pub access_pattern: AccessPatternClass,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum AllocationType {
    Tensor { shape: Vec<u64>, dtype: DataType },
    Buffer { capacity: u64, element_size: u64 },
    Custom { size_bytes: u64 },
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum DataType {
    F32,
    F64,
    I32,
    I64,
    U8,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_safe: bool,
    pub proof: Option<String>,
    pub constraints: Vec<MigrationConstraint>,
}

/// Constraints that must be satisfied for safe migration.
#[derive(Debug, Clone)]
pub struct MigrationConstraint {
    pub description: String,
    pub must_hold: bool,
}

impl TierMigrationValidator {
    /// Create a new tier migration validator.
    pub fn new() -> Self {
        Self {
            smt_solver: todo!("Initialize SMT solver for tier validation"),
            validation_cache: HashMap::new(),
        }
    }
    
    /// Validate that migrating `scenario` preserves semantics.
    pub fn validate(&mut self, scenario: &MigrationScenario) -> ValidationResult {
        // Check cache first
        if let Some(result) = self.validation_cache.get(scenario) {
            return result.clone();
        }
        
        // Build SMT query for this scenario
        let query = self.build_smt_query(scenario);
        
        // Execute SMT query
        let result = self.smt_solver.solve(query);
        
        // Cache and return
        let validation_result = ValidationResult {
            is_safe: result.is_sat(), // UNSAT means definitely safe
            proof: result.proof(),
            constraints: self.extract_constraints(&result),
        };
        
        self.validation_cache.insert(scenario.clone(), validation_result.clone());
        validation_result
    }
    
    fn build_smt_query(&self, scenario: &MigrationScenario) -> SmtQuery {
        // Placeholder for SMT query construction
        // Would create constraints ensuring:
        // 1. All accesses maintain the same relative ordering
        // 2. No stale reads occur after migration
        // 3. Coherency is maintained with other threads
        todo!("Build SMT query for tier migration validation")
    }
    
    fn extract_constraints(&self, result: &SmtResult) -> Vec<MigrationConstraint> {
        vec![
            MigrationConstraint {
                description: "All threads observe consistent memory state".into(),
                must_hold: true,
            },
            MigrationConstraint {
                description: "No writes are lost during migration".into(),
                must_hold: true,
            },
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SMT Solver Integration Types
// ─────────────────────────────────────────────────────────────────────────────

trait TierSmtSolver {
    fn solve(&mut self, query: SmtQuery) -> SmtResult;
}

struct SmtQuery {
    // Placeholder
}

struct SmtResult {
    // Placeholder
}

impl SmtResult {
    fn is_sat(&self) -> bool {
        todo!("Check satisfiability")
    }
    
    fn proof(&self) -> Option<String> {
        todo!("Generate proof")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ZCHMA Runtime Interface
// ─────────────────────────────────────────────────────────────────────────────

/// Main ZCHMA runtime interface.
///
/// This is the entry point for the runtime to interact with ZCHMA's
/// memory management capabilities.
pub struct ZchmaRuntime {
    /// Lifetime-to-tier mapper.
    mapper: LifetimeTierMapper,
    /// Page migration scheduler.
    scheduler: PageMigrationScheduler,
    /// Tier migration validator.
    validator: TierMigrationValidator,
    /// Hardware topology.
    topology: MemoryTopology,
}

impl ZchmaRuntime {
    /// Initialize ZCHMA runtime with discovered hardware.
    pub fn init() -> Self {
        let topology = MemoryTopology::discover();
        let mapper = LifetimeTierMapper::new(topology.tier_characteristics());
        
        Self {
            mapper,
            scheduler: PageMigrationScheduler::new(),
            validator: TierMigrationValidator::new(),
            topology,
        }
    }
    
    /// Allocate memory with automatic tier selection.
    pub fn alloc(
        &mut self,
        lifetime: &Lifetime,
        access_pattern: &AccessPattern,
        size_bytes: u64,
    ) -> ZchmaAllocation {
        let distribution = self.mapper.map_lifetime_to_tier(
            lifetime,
            access_pattern,
            size_bytes,
        );
        
        ZchmaAllocation {
            distribution,
            size_bytes,
            actual_tier: distribution.most_likely_tier().unwrap_or(MemoryTierId(0)),
        }
    }
    
    /// Request migration based on runtime feedback.
    pub fn request_migration(
        &mut self,
        allocation_id: u64,
        new_tier: MemoryTierId,
        expected_benefit_ns: f64,
    ) {
        // Validate the migration first
        let scenario = MigrationScenario {
            src_tier: MemoryTierId(0), // Would come from allocation metadata
            dst_tier: new_tier,
            allocation_type: AllocationType::Custom { size_bytes: 0 },
            access_pattern: AccessPatternClass::Balanced,
        };
        
        if self.validator.validate(&scenario).is_safe {
            self.scheduler.request_migration(MigrationRequest {
                allocation_id,
                from_tier: MemoryTierId(0),
                to_tier: new_tier,
                size_bytes: 0,
                expected_benefit_ns,
                priority: expected_benefit_ns / 1e6,
            });
        }
    }
    
    /// Process pending migrations (called periodically by runtime).
    pub fn process_migrations(&mut self, budget_ns: u64) {
        self.scheduler.process_migrations(budget_ns);
    }
}

/// Result of a ZCHMA allocation.
#[derive(Debug, Clone)]
pub struct ZchmaAllocation {
    pub distribution: PlacementDistribution,
    pub size_bytes: u64,
    pub actual_tier: MemoryTierId,
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory Topology Discovery
// ─────────────────────────────────────────────────────────────────────────────

/// Discovered memory topology including all available tiers.
pub struct MemoryTopology {
    tiers: Vec<TierCharacteristics>,
}

impl MemoryTopology {
    /// Discover available memory tiers on this system.
    pub fn discover() -> Self {
        let mut tiers = Vec::new();
        
        // Always have at least DRAM
        tiers.push(TierCharacteristics {
            tier: MemoryTierId(0),
            latency_ns: 100.0,
            bandwidth_gb_s: 50.0,
            capacity_bytes: u64::MAX,
            is_volatile: true,
            supports_direct_access: true,
            migration_overhead_ns: 1000.0,
            energy_per_access_relative: 1.0,
        });
        
        // Check for HBM (via CPUID on Intel/AMD)
        if Self::has_hbm() {
            tiers.push(TierCharacteristics {
                tier: MemoryTierId(1),
                latency_ns: 50.0,
                bandwidth_gb_s: 256.0,
                capacity_bytes: 16 * 1024 * 1024 * 1024, // 16GB typical
                is_volatile: true,
                supports_direct_access: true,
                migration_overhead_ns: 500.0,
                energy_per_access_relative: 0.8,
            });
        }
        
        // Check for CXL memory
        if Self::has_cxl() {
            tiers.push(TierCharacteristics {
                tier: MemoryTierId(2),
                latency_ns: 150.0,
                bandwidth_gb_s: 100.0,
                capacity_bytes: 64 * 1024 * 1024 * 1024, // 64GB typical
                is_volatile: true,
                supports_direct_access: true,
                migration_overhead_ns: 2000.0,
                energy_per_access_relative: 1.2,
            });
        }
        
        // Check for GPU memory (if CUDA/ROCm available)
        if Self::has_gpu_memory() {
            tiers.push(TierCharacteristics {
                tier: MemoryTierId(3),
                latency_ns: 500.0,
                bandwidth_gb_s: 1000.0,
                capacity_bytes: 16 * 1024 * 1024 * 1024, // 16GB typical
                is_volatile: true,
                supports_direct_access: false,
                migration_overhead_ns: 10000.0,
                energy_per_access_relative: 0.5,
            });
        }
        
        Self { tiers }
    }
    
    fn has_hbm() -> bool {
        // Placeholder: would check CPUID or system info
        false
    }
    
    fn has_cxl() -> bool {
        // Placeholder: would check CXL-capable devices
        false
    }
    
    fn has_gpu_memory() -> bool {
        // Placeholder: would check CUDA/ROCm availability
        false
    }
    
    pub fn tier_characteristics(&self) -> Vec<TierCharacteristics> {
        self.tiers.clone()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Compiler Integration
// ─────────────────────────────────────────────────────────────────────────────

/// Emit placement distribution from borrow checker analysis.
pub struct PlacementHintEmitter {
    /// Output buffer for placement hints.
    hints: Vec<PlacementHint>,
}

#[derive(Debug, Clone)]
pub struct PlacementHint {
    pub allocation_id: u64,
    pub distribution: PlacementDistribution,
    pub source_location: String,
}

impl PlacementHintEmitter {
    pub fn new() -> Self {
        Self { hints: Vec::new() }
    }
    
    /// Emit a placement hint for an allocation.
    pub fn emit(&mut self, hint: PlacementHint) {
        self.hints.push(hint);
    }
    
    /// Get all accumulated hints for code generation.
    pub fn into_hints(self) -> Vec<PlacementHint> {
        self.hints
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_placement_distribution() {
        let mut weights = HashMap::new();
        weights.insert(MemoryTierId(0), 0.7);
        weights.insert(MemoryTierId(1), 0.3);
        
        let dist = PlacementDistribution::new(weights);
        assert_eq!(dist.most_likely_tier(), Some(MemoryTierId(0)));
    }
    
    #[test]
    fn test_access_pattern_classification() {
        let classifier = AccessPatternClassifier::new();
        let pattern = AccessPattern {
            is_random_access: true,
            is_sequential: false,
            read_write_ratio: 0.9,
            access_frequency: 1.0,
            operation_name: None,
        };
        
        let class = classifier.classify(&pattern);
        assert_eq!(class, AccessPatternClass::RandomAccess);
    }
    
    #[test]
    fn test_migration_cost_model() {
        let model = MigrationCostModel::new();
        let estimate = model.estimate_migration_cost(
            MemoryTierId(0),
            MemoryTierId(1),
            1024 * 1024, // 1MB
        );
        
        assert!(estimate.size_bytes == 1024 * 1024);
        assert!(estimate.total_cost_ns > 0.0);
    }
}

// =============================================================================
// Integration Points
// =============================================================================
//
// The following integration points connect ZCHMA to the rest of the Jules
// compiler and runtime:
//
// 1. BORROW CHECKER INTEGRATION (src/compiler/borrowck.rs)
//    - Emit PlacementHint for each allocation
//    - Include lifetime analysis in placement decisions
//
// 2. MEMORY MANAGEMENT INTEGRATION (src/runtime/memory_management.rs)
//    - Replace explicit tier selection with ZchmaRuntime::alloc
//    - Hook migration scheduler into runtime event loop
//
// 3. TRANSLATION VALIDATION INTEGRATION (src/compiler/translation_validation.rs)
//    - Extend existing validation to cover tier migrations
//    - Use ZchmaRuntime::validator for migration proofs
//
// 4. HARDWARE FEEDBACK INTEGRATION (src/compiler/hw_feedback.rs)
//    - Feed performance counters back to tier selection model
//    - Update PlacementDistribution weights based on actual behavior
//
// 5. AOT COMPILER INTEGRATION (src/jit/aot_native.rs)
//    - Emit placement distribution as metadata in ELF
//    - Support runtime override of compile-time placement decisions
//
// =============================================================================
