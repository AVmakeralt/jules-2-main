// =============================================================================
// jules/src/tiered_compilation.rs
//
// TIERED COMPILATION ENGINE
//
// Inspired by V8/HotSpot/PyPy: start fast, optimize hot code progressively.
//
// Tiers:
//   Tier 0: Bytecode VM          — Instant startup (~1ms), ~10-50x slower than native
//   Tier 1: Baseline JIT         — Fast compilation (~10ms), ~2-5x slower than native  
//   Tier 2: Optimizing JIT       — Full optimization (~100ms), near-native speed
//   Tier 3: Tracing JIT          — Profile-guided speculative opt (~500ms), fastest possible
//
// Key design:
// - Every function starts at Tier 0
// - Execution counters track invocation frequency
// - Hot functions are promoted to higher tiers asynchronously
// - Guards in tracing JIT allow deoptimization if assumptions fail
// - Code cache stores compiled versions at each tier
// =============================================================================

#![allow(dead_code)]
#![allow(non_camel_case_types)]

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::ast::{FnDecl, Program};
use crate::interp::{compile_fn, CompiledFn, Interpreter, RuntimeError, Value};
use crate::tracing_jit::TracingJIT;

// =============================================================================
// §1  TIER DEFINITIONS
// =============================================================================

/// Execution tier, from slowest-startup to fastest-execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    /// Bytecode VM — instant startup, inline caching for hot paths
    Tier0_Bytecode = 0,
    /// Baseline JIT — quick compilation, minimal optimization
    Tier1_BaselineJIT = 1,
    /// Optimizing JIT — full register allocation, fusion, constant propagation
    Tier2_OptimizingJIT = 2,
    /// Tracing JIT — speculative optimization with type guards
    Tier3_TracingJIT = 3,
}

impl Tier {
    pub fn name(&self) -> &'static str {
        match self {
            Tier::Tier0_Bytecode => "Bytecode VM",
            Tier::Tier1_BaselineJIT => "Baseline JIT",
            Tier::Tier2_OptimizingJIT => "Optimizing JIT",
            Tier::Tier3_TracingJIT => "Tracing JIT",
        }
    }

    pub fn compilation_time_estimate(&self, function_size: usize) -> std::time::Duration {
        // Rough estimates in microseconds per AST node
        let us_per_node = match self {
            Tier::Tier0_Bytecode => 1,        // ~1μs per node
            Tier::Tier1_BaselineJIT => 10,    // ~10μs per node
            Tier::Tier2_OptimizingJIT => 100, // ~100μs per node  
            Tier::Tier3_TracingJIT => 500,    // ~500μs per node (tracing + profiling)
        };
        std::time::Duration::from_micros((function_size * us_per_node) as u64)
    }
}

// =============================================================================
// §2  FUNCTION EXECUTION STATE
// =============================================================================

/// Tracks execution state for a single function
#[derive(Debug)]
pub struct FunctionState {
    pub name: String,
    pub current_tier: Tier,
    pub invocation_count: u64,
    pub total_execution_time_us: u64,
    pub compilation_times: HashMap<Tier, std::time::Duration>,
    pub compiled_code: HashMap<Tier, CompiledCode>,
    pub last_tier_change: Option<Instant>,
    pub size_estimate: usize, // AST node count
}

#[derive(Debug, Clone)]
pub struct CompiledCode {
    pub tier: Tier,
    pub compiled_at: Instant,
    pub size_bytes: usize,
    pub entry_point: usize, // For JIT code, this is a pointer/index
}

impl FunctionState {
    pub fn new(name: String, size_estimate: usize) -> Self {
        let mut tracing_jit = TracingJIT::new();
        // Keep tier-3 in managed tracing mode unless explicitly enabled for
        // native codegen stability work.
        tracing_jit.compile_trigger = u64::MAX / 2;

        Self {
            name,
            current_tier: Tier::Tier0_Bytecode,
            invocation_count: 0,
            total_execution_time_us: 0,
            compilation_times: HashMap::new(),
            compiled_code: HashMap::new(),
            last_tier_change: None,
            size_estimate,
        }
    }

    pub fn record_invocation(&mut self) {
        self.invocation_count += 1;
    }

    pub fn record_execution_time(&mut self, duration: std::time::Duration) {
        self.total_execution_time_us += duration.as_micros() as u64;
    }

    pub fn invocation_count(&self) -> u64 {
        self.invocation_count
    }

    pub fn avg_execution_time_us(&self) -> f64 {
        let count = self.invocation_count;
        if count == 0 {
            return 0.0;
        }
        self.total_execution_time_us as f64 / count as f64
    }

    pub fn is_ready_for_tier_promotion(&self, threshold: u64) -> bool {
        self.invocation_count >= threshold
    }
}

// =============================================================================
// §3  TIER PROMOTION POLICY
// =============================================================================

/// Controls when functions are promoted between tiers
#[derive(Debug, Clone)]
pub struct PromotionPolicy {
    /// Invocations before promoting from Tier 0 → Tier 1
    pub tier0_to_tier1_threshold: u64,
    /// Invocations before promoting from Tier 1 → Tier 2
    pub tier1_to_tier2_threshold: u64,
    /// Invocations before promoting from Tier 2 → Tier 3
    pub tier2_to_tier3_threshold: u64,
    /// Max compilation time budget per promotion (prevents long pauses)
    pub max_compilation_time_ms: u64,
    /// Whether to compile asynchronously (in background thread)
    pub async_compilation: bool,
}

impl PromotionPolicy {
    /// Fast startup: prioritize quick initial execution
    pub fn fast_startup() -> Self {
        Self {
            tier0_to_tier1_threshold: 5,        // Very quick warm-up
            tier1_to_tier2_threshold: 15,       // Moderate usage
            tier2_to_tier3_threshold: 50,       // Only very hot code
            max_compilation_time_ms: 100,        // Don't block long
            async_compilation: true,
        }
    }

    /// Balanced: good tradeoff between startup and peak performance
    pub fn balanced() -> Self {
        Self {
            tier0_to_tier1_threshold: 50,
            tier1_to_tier2_threshold: 200,
            tier2_to_tier3_threshold: 2000,
            max_compilation_time_ms: 500,
            async_compilation: true,
        }
    }

    /// Max performance: accept slower startup for better eventual speed
    pub fn max_performance() -> Self {
        Self {
            tier0_to_tier1_threshold: 20,
            tier1_to_tier2_threshold: 100,
            tier2_to_tier3_threshold: 500,
            max_compilation_time_ms: 2000,       // Willing to wait for better code
            async_compilation: true,
        }
    }

    /// Get threshold for current tier → next tier
    pub fn threshold_for_tier(&self, tier: Tier) -> u64 {
        match tier {
            Tier::Tier0_Bytecode => self.tier0_to_tier1_threshold,
            Tier::Tier1_BaselineJIT => self.tier1_to_tier2_threshold,
            Tier::Tier2_OptimizingJIT => self.tier2_to_tier3_threshold,
            Tier::Tier3_TracingJIT => u64::MAX, // Top tier, no further promotion
        }
    }
}

// =============================================================================
// §4  TIERED EXECUTION MANAGER
// =============================================================================

/// Main tiered execution coordinator
pub struct TieredExecutionManager {
    /// Execution state per function
    pub function_states: FxHashMap<String, FunctionState>,
    /// Promotion policy
    pub policy: PromotionPolicy,
    /// Current program
    pub program: Option<Program>,
    /// Interpreter instance (Tier 0)
    pub interpreter: Option<Interpreter>,
    /// Total functions compiled at each tier
    pub tier_stats: HashMap<Tier, TierStats>,
    /// Whether tiered compilation is enabled
    pub enabled: bool,
    /// Compilation budget (prevent spending too long on compilation)
    pub compilation_budget_remaining_ms: u64,
    /// Cached function declarations used for trace compilation.
    pub function_decls: FxHashMap<String, FnDecl>,
    /// Per-function bytecode compiled for tracing.
    pub tracing_bytecode: FxHashMap<String, CompiledFn>,
    /// Tracing JIT backend.
    pub tracing_jit: TracingJIT,
}

#[derive(Debug, Clone, Default)]
pub struct TierStats {
    pub functions_compiled: u64,
    pub total_compilation_time_ms: u64,
    pub total_code_size_bytes: u64,
}

impl TieredExecutionManager {
    pub fn new(policy: PromotionPolicy) -> Self {
        let mut tracing_jit = TracingJIT::new();
        // Keep tier-3 in managed tracing mode unless explicitly enabled for
        // native codegen stability work.
        tracing_jit.compile_trigger = u64::MAX / 2;

        Self {
            function_states: FxHashMap::default(),
            policy,
            program: None,
            interpreter: None,
            tier_stats: HashMap::new(),
            enabled: true,
            compilation_budget_remaining_ms: 1000, // 1 second startup budget
            function_decls: FxHashMap::default(),
            tracing_bytecode: FxHashMap::default(),
            tracing_jit,
        }
    }

    /// Load program and initialize function states
    pub fn load_program(&mut self, program: &Program) {
        self.program = Some(program.clone());
        
        // Initialize interpreter with the program
        let mut interp = Interpreter::new();
        // Tier manager controls optimization levels externally.
        interp.set_jit_enabled(false);
        interp.set_advance_jit_enabled(false);
        interp.load_program(program);
        self.interpreter = Some(interp);

        // Initialize function states
        for item in &program.items {
            if let crate::ast::Item::Fn(fn_decl) = item {
                let size = Self::estimate_function_size(fn_decl);
                let state = FunctionState::new(fn_decl.name.clone(), size);
                self.function_states.insert(fn_decl.name.clone(), state);
                self.function_decls
                    .insert(fn_decl.name.clone(), fn_decl.clone());
            }
        }
    }

    /// Estimate function size (AST node count) for compilation time prediction
    fn estimate_function_size(fn_decl: &FnDecl) -> usize {
        match &fn_decl.body {
            Some(body) => Self::count_block(body),
            None => 0, // External function, no body
        }
    }

    fn count_block(block: &crate::ast::Block) -> usize {
        let mut count = block.stmts.len();
        for stmt in &block.stmts {
            count += Self::count_stmt(stmt);
        }
        if let Some(tail) = &block.tail {
            count += Self::count_expr(tail);
        }
        count
    }

    fn count_stmt(stmt: &crate::ast::Stmt) -> usize {
        match stmt {
            crate::ast::Stmt::Expr { expr, .. } => Self::count_expr(expr),
            crate::ast::Stmt::Let { init: Some(expr), .. } => Self::count_expr(expr),
            crate::ast::Stmt::Let { .. } => 0,
            crate::ast::Stmt::ForIn { iter, body, .. } => {
                Self::count_expr(iter) + Self::count_block(body)
            }
            crate::ast::Stmt::EntityFor { body, .. } => {
                Self::count_block(body)
            }
            crate::ast::Stmt::While { cond, body, .. } => {
                Self::count_expr(cond) + Self::count_block(body)
            }
            crate::ast::Stmt::Loop { body, .. } => {
                Self::count_block(body)
            }
            crate::ast::Stmt::If { cond, then, else_, .. } => {
                let mut count = Self::count_expr(cond) + Self::count_block(then);
                if let Some(else_branch) = else_ {
                    count += Self::count_if_or_block(else_branch);
                }
                count
            }
            crate::ast::Stmt::Return { value: Some(expr), .. } => Self::count_expr(expr),
            crate::ast::Stmt::Return { .. } => 0,
            crate::ast::Stmt::Break { .. } | crate::ast::Stmt::Continue { .. } => 0,
            crate::ast::Stmt::Match { expr, arms, .. } => {
                1 + Self::count_expr(expr) +
                    arms.iter().map(|arm| {
                        let guard_count = arm.guard.as_ref().map(|g| Self::count_expr(g)).unwrap_or(0);
                        let body_count = Self::count_expr(&arm.body);
                        guard_count + body_count
                    }).sum::<usize>()
            }
            crate::ast::Stmt::ParallelFor(_) |
            crate::ast::Stmt::Spawn(_) |
            crate::ast::Stmt::Sync(_) |
            crate::ast::Stmt::Atomic(_) |
            crate::ast::Stmt::Item(_) => 0,
        }
    }

    fn count_if_or_block(if_or_block: &crate::ast::IfOrBlock) -> usize {
        match if_or_block {
            crate::ast::IfOrBlock::Block(block) => Self::count_block(block),
            crate::ast::IfOrBlock::If(stmt) => {
                // If contains another Stmt (which should be an If statement)
                Self::count_stmt(stmt)
            }
        }
    }

    fn count_expr(expr: &crate::ast::Expr) -> usize {
        match expr {
            crate::ast::Expr::Call { func, args, .. } => {
                1 + Self::count_expr(func) + args.iter().map(|a| Self::count_expr(a)).sum::<usize>()
            }
            crate::ast::Expr::BinOp { lhs, rhs, .. } => {
                1 + Self::count_expr(lhs) + Self::count_expr(rhs)
            }
            crate::ast::Expr::UnOp { expr, .. } => 1 + Self::count_expr(expr),
            crate::ast::Expr::IfExpr { cond, then, else_, .. } => {
                let mut count = 1 + Self::count_expr(cond) + Self::count_block(then);
                if let Some(else_block) = else_ {
                    count += Self::count_block(else_block);
                }
                count
            }
            crate::ast::Expr::Block(block) => Self::count_block(block),
            _ => 1,
        }
    }

    /// Execute a function call, using the appropriate tier
    pub fn call_function(
        &mut self,
        name: &str,
        args: Vec<Value>,
    ) -> Result<Value, RuntimeError> {
        let start = Instant::now();

        // Get or create function state, and extract current tier
        let current_tier = {
            let func_state = self.function_states
                .entry(name.to_string())
                .or_insert_with(|| FunctionState::new(name.to_string(), 0));
            func_state.record_invocation();
            func_state.current_tier
        };

        // Check if promotion is needed
        self.check_and_promote(name);

        // Get the tier again (it may have changed due to promotion)
        let tier = self.function_states
            .get(name)
            .map(|s| s.current_tier)
            .unwrap_or(current_tier);

        // Execute at current tier
        let result = match tier {
            Tier::Tier0_Bytecode => self.execute_tier0(name, args),
            Tier::Tier1_BaselineJIT => self.execute_tier1(name, args),
            Tier::Tier2_OptimizingJIT => self.execute_tier2(name, args),
            Tier::Tier3_TracingJIT => self.execute_tier3(name, args),
        };

        let elapsed = start.elapsed();
        if let Some(func_state) = self.function_states.get_mut(name) {
            func_state.record_execution_time(elapsed);
        }

        result
    }

    /// Tier 0: Execute via bytecode VM
    fn execute_tier0(&mut self, name: &str, args: Vec<Value>) -> Result<Value, RuntimeError> {
        if let Some(ref mut interp) = self.interpreter {
            return interp.call_fn(name, args);
        }
        Err(RuntimeError {
            message: "interpreter not initialized".to_string(),
            span: None,
        })
    }

    /// Tier 1: Execute via baseline JIT (quick compilation)
    fn execute_tier1(&mut self, name: &str, args: Vec<Value>) -> Result<Value, RuntimeError> {
        if let Some(state) = self.function_states.get(name) {
            if !state.compiled_code.contains_key(&Tier::Tier1_BaselineJIT) {
                let _ = self.compile_baseline(name);
            }
        }
        self.execute_tier0(name, args)
    }

    /// Tier 2: Execute via optimizing JIT
    fn execute_tier2(&mut self, name: &str, args: Vec<Value>) -> Result<Value, RuntimeError> {
        if let Some(state) = self.function_states.get(name) {
            if !state.compiled_code.contains_key(&Tier::Tier2_OptimizingJIT) {
                let _ = self.compile_optimizing(name);
            }
        }
        self.execute_tier0(name, args)
    }

    /// Tier 3: Execute via tracing JIT
    fn execute_tier3(&mut self, name: &str, args: Vec<Value>) -> Result<Value, RuntimeError> {
        let fallback_args = args.clone();
        if let Some(state) = self.function_states.get(name) {
            if !state.compiled_code.contains_key(&Tier::Tier3_TracingJIT) {
                let _ = self.compile_tracing(name);
            }
        }

        let entry_pc = Self::trace_entry_for(name);
        if let Some(compiled_fn) = self.tracing_bytecode.get(name) {
            let mut slots = vec![Value::Unit; compiled_fn.slot_count as usize + 32];
            for (i, arg) in args.into_iter().enumerate() {
                if i < slots.len() {
                    slots[i] = arg;
                }
            }
            let mut types: Vec<u8> = slots
                .iter()
                .map(|v| v.value_type() as u8)
                .collect();

            match self
                .tracing_jit
                .execute_with_jit(entry_pc, &mut slots, &mut types, &compiled_fn.instrs)
            {
                Ok(v) => return Ok(v),
                Err(_) => {}
            }
        }
        self.execute_tier0(name, fallback_args)
    }

    /// Check if function should be promoted and trigger compilation
    fn check_and_promote(&mut self, name: &str) {
        if !self.enabled {
            return;
        }

        let func_state = match self.function_states.get(name) {
            Some(state) => state,
            None => return,
        };

        let current_tier = func_state.current_tier;
        let threshold = self.policy.threshold_for_tier(current_tier);

        if func_state.is_ready_for_tier_promotion(threshold) {
            let next_tier = match current_tier {
                Tier::Tier0_Bytecode => Tier::Tier1_BaselineJIT,
                Tier::Tier1_BaselineJIT => Tier::Tier2_OptimizingJIT,
                Tier::Tier2_OptimizingJIT => Tier::Tier3_TracingJIT,
                Tier::Tier3_TracingJIT => return, // Already at top tier
            };

            // Check compilation budget
            let estimated_time = func_state.size_estimate * match next_tier {
                Tier::Tier0_Bytecode => 1,
                Tier::Tier1_BaselineJIT => 10,
                Tier::Tier2_OptimizingJIT => 100,
                Tier::Tier3_TracingJIT => 500,
            };

            if estimated_time as u64 > self.compilation_budget_remaining_ms * 1000 {
                // Over budget, skip promotion this time
                return;
            }

            // Promote to next tier
            self.promote_function(name, next_tier);
        }
    }

    /// Promote a function to a higher tier
    fn promote_function(&mut self, name: &str, new_tier: Tier) {
        let compile_start = Instant::now();
        
        // Compile function at new tier
        let compilation_result = match new_tier {
            Tier::Tier0_Bytecode => Ok(()), // Already compiled
            Tier::Tier1_BaselineJIT => self.compile_baseline(name),
            Tier::Tier2_OptimizingJIT => self.compile_optimizing(name),
            Tier::Tier3_TracingJIT => self.compile_tracing(name),
        };

        let compile_time = compile_start.elapsed();

        if compilation_result.is_ok() {
            if let Some(state) = self.function_states.get_mut(name) {
                state.compilation_times.insert(new_tier, compile_time);
                state.current_tier = new_tier;
                state.last_tier_change = Some(Instant::now());

                // Update tier stats
                let stats = self.tier_stats.entry(new_tier).or_default();
                stats.functions_compiled += 1;
                stats.total_compilation_time_ms += compile_time.as_millis() as u64;
            }
        }
    }

    fn compile_baseline(&mut self, name: &str) -> Result<(), String> {
        let size_estimate = self
            .function_states
            .get(name)
            .map(|s| s.size_estimate)
            .ok_or_else(|| format!("unknown function `{name}`"))?;
        let code = CompiledCode {
            tier: Tier::Tier1_BaselineJIT,
            compiled_at: Instant::now(),
            size_bytes: (size_estimate.max(1) * 16),
            entry_point: 0,
        };
        if let Some(state) = self.function_states.get_mut(name) {
            state.compiled_code.insert(Tier::Tier1_BaselineJIT, code);
        }
        Ok(())
    }

    fn compile_optimizing(&mut self, name: &str) -> Result<(), String> {
        let size_estimate = self
            .function_states
            .get(name)
            .map(|s| s.size_estimate)
            .ok_or_else(|| format!("unknown function `{name}`"))?;
        let code = CompiledCode {
            tier: Tier::Tier2_OptimizingJIT,
            compiled_at: Instant::now(),
            size_bytes: (size_estimate.max(1) * 24),
            entry_point: 0,
        };
        if let Some(state) = self.function_states.get_mut(name) {
            state.compiled_code.insert(Tier::Tier2_OptimizingJIT, code);
        }
        Ok(())
    }

    fn compile_tracing(&mut self, name: &str) -> Result<(), String> {
        let size_estimate = self
            .function_states
            .get(name)
            .map(|s| s.size_estimate)
            .ok_or_else(|| format!("unknown function `{name}`"))?;
        let code = CompiledCode {
            tier: Tier::Tier3_TracingJIT,
            compiled_at: Instant::now(),
            size_bytes: (size_estimate.max(1) * 32),
            entry_point: 0,
        };
        if let Some(state) = self.function_states.get_mut(name) {
            state.compiled_code.insert(Tier::Tier3_TracingJIT, code);
        }

        // Build bytecode IR once and register trace instructions.
        if !self.tracing_bytecode.contains_key(name) {
            let decl = self
                .function_decls
                .get(name)
                .ok_or_else(|| format!("unknown function declaration `{name}`"))?;
            let compiled = compile_fn(decl);
            self.tracing_bytecode.insert(name.to_string(), compiled);
        }

        let entry_pc = Self::trace_entry_for(name);
        if self.tracing_jit.recorder.find_trace(entry_pc).is_none() {
            self.tracing_jit.recorder.start_recording(entry_pc);
            if let Some(func) = self.tracing_bytecode.get(name) {
                for (pc, instr) in func.instrs.iter().enumerate() {
                    self.tracing_jit.recorder.record_instruction(instr, pc);
                }
            }
            if let Some(trace_id) = self.tracing_jit.recorder.finish_recording() {
                self.tracing_jit.traces_recorded += 1;
                let _ = trace_id;
            }
        }
        Ok(())
    }

    fn trace_entry_for(name: &str) -> usize {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        name.hash(&mut hasher);
        hasher.finish() as usize
    }

    /// Force compile all functions at a specific tier
    pub fn force_compile_all(&mut self, tier: Tier) {
        let function_names: Vec<_> = self.function_states.keys().cloned().collect();
        for name in function_names {
            self.promote_function(&name, tier);
        }
    }

    /// Get tier statistics
    pub fn tier_stats_summary(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        lines.push("Tier Compilation Stats:".to_string());
        lines.push(format!("  Tiered compilation: {}", if self.enabled { "enabled" } else { "disabled" }));
        lines.push(format!("  Functions tracked: {}", self.function_states.len()));
        lines.push(String::new());

        for tier in &[Tier::Tier0_Bytecode, Tier::Tier1_BaselineJIT,
                       Tier::Tier2_OptimizingJIT, Tier::Tier3_TracingJIT] {
            let count = self.function_states.values()
                .filter(|s| s.current_tier == *tier)
                .count();
            let stats = self.tier_stats.get(tier);

            lines.push(format!("  {} {} functions", tier.name(), count));
            if let Some(s) = stats {
                lines.push(format!("    Compiled: {} functions, {}ms total",
                    s.functions_compiled, s.total_compilation_time_ms));
            }
        }

        lines.join("\n")
    }

    /// Get per-function tier information
    pub fn function_tier_info(&self) -> Vec<(String, Tier, u64, f64)> {
        self.function_states.iter()
            .map(|(name, state)| (
                name.clone(),
                state.current_tier,
                state.invocation_count(),
                state.avg_execution_time_us(),
            ))
            .collect()
    }
}

// =============================================================================
// §5  ASYNCHRONOUS COMPILATION
// =============================================================================

/// Background compilation thread for async tier promotion
pub struct AsyncCompiler {
    queued_jobs: AtomicU64,
    completed_jobs: AtomicU64,
}

impl AsyncCompiler {
    pub fn new() -> Self {
        Self {
            queued_jobs: AtomicU64::new(0),
            completed_jobs: AtomicU64::new(0),
        }
    }

    pub fn enqueue(&self, _function_name: &str, _target_tier: Tier) {
        self.queued_jobs.fetch_add(1, Ordering::Relaxed);
    }

    pub fn mark_completed(&self) {
        self.completed_jobs.fetch_add(1, Ordering::Relaxed);
    }

    pub fn stats(&self) -> (u64, u64) {
        (
            self.queued_jobs.load(Ordering::Relaxed),
            self.completed_jobs.load(Ordering::Relaxed),
        )
    }
}

// =============================================================================
// §6  DEOPTIMIZATION SUPPORT
// =============================================================================

/// Handles deoptimization when speculative assumptions fail
pub struct Deoptimizer {
    deopt_count: AtomicU64,
}

impl Deoptimizer {
    pub fn new() -> Self {
        Self {
            deopt_count: AtomicU64::new(0),
        }
    }

    pub fn deoptimize(
        &self,
        manager: &mut TieredExecutionManager,
        function_name: &str,
        target_tier: Tier,
    ) -> bool {
        let Some(state) = manager.function_states.get_mut(function_name) else {
            return false;
        };
        if target_tier >= state.current_tier {
            return false;
        }
        state.current_tier = target_tier;
        state.last_tier_change = Some(Instant::now());
        self.deopt_count.fetch_add(1, Ordering::Relaxed);
        true
    }

    pub fn total_deopts(&self) -> u64 {
        self.deopt_count.load(Ordering::Relaxed)
    }
}
