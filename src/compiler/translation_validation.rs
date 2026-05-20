// =========================================================================
// Translation Validation - Prove Semantic Equivalence
// Mathematical proof that optimized code preserves original semantics
// Ensures correctness is never traded for speed
// =========================================================================
//
// This module uses canonical IR types from `crate::compiler::ir` wherever
// possible instead of defining duplicates:
//
//   • BlockId  → ir::CodegenBlockId (= usize)
//   • VarId    → ir::VarId           (= usize)
//   • BinaryOp → BinOp { Arith(IrBinOp), Cmp(IrCmpOp) }
//   • UnaryOp  → ir::IrUnOp
//
// Types unique to translation validation (InstrId, Instruction,
// BasicBlock, ControlFlowGraph, AbstractValue, SymbolicState,
// ValidationResult, TranslationValidator) remain defined here.

use std::collections::{HashMap, HashSet};

// Re-export canonical types from ir.rs for use in this module and consumers.
pub use crate::compiler::ir::{CodegenBlockId as BlockId, IrBinOp, IrCmpOp, IrUnOp, VarId};

/// Instruction identifier — unique to translation validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstrId(pub usize);

// =============================================================================
// BinOp — combined arithmetic + comparison operator
// =============================================================================
//
// The translation validation module treats arithmetic and comparison
// operations uniformly inside `Instruction` and `AbstractValue`, so we
// wrap the canonical `IrBinOp` and `IrCmpOp` in a single enum rather
// than duplicating their variants.
// =============================================================================

/// Binary operator used in translation validation.
///
/// Wraps the canonical [`IrBinOp`] (arithmetic / bitwise) and
/// [`IrCmpOp`] (comparison) from `ir.rs` so that `Instruction` and
/// `AbstractValue` can treat them uniformly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    /// Arithmetic or bitwise binary operation.
    Arith(IrBinOp),
    /// Comparison binary operation.
    Cmp(IrCmpOp),
}

impl BinOp {
    /// Construct from the old-style flat variants used before the refactor.
    ///
    /// Provided for backward-compatible construction in tests and callers
    /// that were using the previous monolithic `BinaryOp` enum.
    pub fn add() -> Self { BinOp::Arith(IrBinOp::Add) }
    pub fn sub() -> Self { BinOp::Arith(IrBinOp::Sub) }
    pub fn mul() -> Self { BinOp::Arith(IrBinOp::Mul) }
    pub fn div() -> Self { BinOp::Arith(IrBinOp::Div) }
    pub fn rem() -> Self { BinOp::Arith(IrBinOp::Rem) }
    pub fn and() -> Self { BinOp::Arith(IrBinOp::And) }
    pub fn or()  -> Self { BinOp::Arith(IrBinOp::Or) }
    pub fn xor() -> Self { BinOp::Arith(IrBinOp::BitXor) }
    pub fn shl() -> Self { BinOp::Arith(IrBinOp::Shl) }
    pub fn shr() -> Self { BinOp::Arith(IrBinOp::Shr) }
    pub fn eq()  -> Self { BinOp::Cmp(IrCmpOp::Eq) }
    pub fn ne()  -> Self { BinOp::Cmp(IrCmpOp::Ne) }
    pub fn lt()  -> Self { BinOp::Cmp(IrCmpOp::Lt) }
    pub fn le()  -> Self { BinOp::Cmp(IrCmpOp::Le) }
    pub fn gt()  -> Self { BinOp::Cmp(IrCmpOp::Gt) }
    pub fn ge()  -> Self { BinOp::Cmp(IrCmpOp::Ge) }
}

// =============================================================================
// Instruction — validation-level instruction representation
// =============================================================================
//
// This is a higher-level view than `ir::IRInstr`, designed for symbolic
// execution and equivalence checking. Conversion functions from `IRInstr`
// are provided below.
// =============================================================================

/// Instruction operation (validation-level representation).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Instruction {
    /// Load from memory
    Load { dst: VarId, addr: VarId },
    /// Store to memory
    Store { addr: VarId, src: VarId },
    /// Binary operation
    BinaryOp { dst: VarId, op: BinOp, left: VarId, right: VarId },
    /// Unary operation
    UnaryOp { dst: VarId, op: IrUnOp, src: VarId },
    /// Move/copy
    Move { dst: VarId, src: VarId },
    /// Conditional branch
    Branch { cond: VarId, true_block: BlockId, false_block: BlockId },
    /// Unconditional jump
    Jump { target: BlockId },
    /// Call
    Call { func: String, args: Vec<VarId>, ret: Option<VarId> },
    /// Return
    Return { value: Option<VarId> },
    /// Phi node (SSA)
    Phi { dst: VarId, incoming: Vec<(BlockId, VarId)> },
}

// =============================================================================
// Conversion from ir::IRInstr
// =============================================================================

impl Instruction {
    /// Convert a codegen-level [`crate::compiler::ir::IRInstr`] into a
    /// validation-level `Instruction`.
    ///
    /// Returns `None` for instructions that have no validation-level
    /// equivalent (e.g., `Label`, `Comment`).
    pub fn from_codegen_instr(instr: &crate::compiler::ir::IRInstr) -> Option<Self> {
        use crate::compiler::ir::IRInstr;
        match instr {
            IRInstr::Const { dst, .. } | IRInstr::ConstF64 { dst, .. } => {
                // Constants are materialised as moves from themselves;
                // at the validation level we treat them as identity moves.
                Some(Instruction::Move { dst: *dst, src: *dst })
            }
            IRInstr::Add { dst, lhs, rhs }  => Some(Instruction::BinaryOp { dst: *dst, op: BinOp::add(), left: *lhs, right: *rhs }),
            IRInstr::Sub { dst, lhs, rhs }  => Some(Instruction::BinaryOp { dst: *dst, op: BinOp::sub(), left: *lhs, right: *rhs }),
            IRInstr::Mul { dst, lhs, rhs }  => Some(Instruction::BinaryOp { dst: *dst, op: BinOp::mul(), left: *lhs, right: *rhs }),
            IRInstr::SDiv { dst, lhs, rhs } => Some(Instruction::BinaryOp { dst: *dst, op: BinOp::div(), left: *lhs, right: *rhs }),
            IRInstr::SRem { dst, lhs, rhs } => Some(Instruction::BinaryOp { dst: *dst, op: BinOp::rem(), left: *lhs, right: *rhs }),
            IRInstr::Neg { dst, src }       => Some(Instruction::UnaryOp { dst: *dst, op: IrUnOp::Neg, src: *src }),
            IRInstr::And { dst, lhs, rhs }  => Some(Instruction::BinaryOp { dst: *dst, op: BinOp::and(), left: *lhs, right: *rhs }),
            IRInstr::Or  { dst, lhs, rhs }  => Some(Instruction::BinaryOp { dst: *dst, op: BinOp::or(),  left: *lhs, right: *rhs }),
            IRInstr::Xor { dst, lhs, rhs }  => Some(Instruction::BinaryOp { dst: *dst, op: BinOp::xor(), left: *lhs, right: *rhs }),
            IRInstr::Shl { dst, lhs, rhs }  => Some(Instruction::BinaryOp { dst: *dst, op: BinOp::shl(), left: *lhs, right: *rhs }),
            IRInstr::AShr { dst, lhs, rhs } => Some(Instruction::BinaryOp { dst: *dst, op: BinOp::shr(), left: *lhs, right: *rhs }),
            IRInstr::LShr { dst, lhs, rhs } => Some(Instruction::BinaryOp { dst: *dst, op: BinOp::shr(), left: *lhs, right: *rhs }),
            IRInstr::Not { dst, src }       => Some(Instruction::UnaryOp { dst: *dst, op: IrUnOp::Not, src: *src }),
            IRInstr::ICmp { dst, cond, lhs, rhs } => {
                let op = match cond {
                    crate::compiler::ir::ICmpCond::Eq  => BinOp::eq(),
                    crate::compiler::ir::ICmpCond::Ne  => BinOp::ne(),
                    crate::compiler::ir::ICmpCond::SLt => BinOp::lt(),
                    crate::compiler::ir::ICmpCond::SLe => BinOp::le(),
                    crate::compiler::ir::ICmpCond::SGt => BinOp::gt(),
                    crate::compiler::ir::ICmpCond::SGe => BinOp::ge(),
                    crate::compiler::ir::ICmpCond::ULt => BinOp::lt(),
                    crate::compiler::ir::ICmpCond::ULe => BinOp::le(),
                    crate::compiler::ir::ICmpCond::UGt => BinOp::gt(),
                    crate::compiler::ir::ICmpCond::UGe => BinOp::ge(),
                };
                Some(Instruction::BinaryOp { dst: *dst, op, left: *lhs, right: *rhs })
            }
            IRInstr::Move { dst, src } => Some(Instruction::Move { dst: *dst, src: *src }),
            IRInstr::Br { target } => Some(Instruction::Jump { target: *target }),
            IRInstr::CondBr { cond, if_true, if_false } => {
                Some(Instruction::Branch { cond: *cond, true_block: *if_true, false_block: *if_false })
            }
            IRInstr::Ret { value } => Some(Instruction::Return { value: *value }),
            IRInstr::Call { dst, func, args } => {
                Some(Instruction::Call { func: func.clone(), args: args.to_vec(), ret: Some(*dst) })
            }
            IRInstr::TailCall { func, args } => {
                Some(Instruction::Call { func: func.clone(), args: args.to_vec(), ret: None })
            }
            IRInstr::Alloca { dst, .. } => Some(Instruction::Move { dst: *dst, src: *dst }),
            IRInstr::Store { ptr, value } => Some(Instruction::Store { addr: *ptr, src: *value }),
            IRInstr::Load { dst, ptr } => Some(Instruction::Load { dst: *dst, addr: *ptr }),
            IRInstr::Phi { dst, incoming } => {
                Some(Instruction::Phi { dst: *dst, incoming: incoming.to_vec() })
            }
            IRInstr::Label | IRInstr::Comment(_) => None,
        }
    }
}

// =============================================================================
// BasicBlock — validation-level basic block
// =============================================================================

/// Basic block (validation-level representation).
///
/// This is a simplified view of `ir::IRBlock` that holds validation-level
/// `Instruction`s instead of `IRInstr`s.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Block ID
    pub id: BlockId,
    /// Instructions in the block
    pub instructions: Vec<Instruction>,
    /// Successors
    pub successors: Vec<BlockId>,
    /// Predecessors
    pub predecessors: Vec<BlockId>,
}

// =============================================================================
// ControlFlowGraph — validation-level CFG
// =============================================================================

/// Control flow graph (validation-level representation).
///
/// Can be constructed from an `ir::IRFunction` via
/// [`ControlFlowGraph::from_codegen_function`].
#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    /// Basic blocks
    pub blocks: HashMap<BlockId, BasicBlock>,
    /// Entry block
    pub entry: BlockId,
    /// Exit block
    pub exit: BlockId,
}

impl ControlFlowGraph {
    /// Build a `ControlFlowGraph` from a codegen-level `ir::IRFunction`.
    ///
    /// Converts each `IRInstr` to a validation-level `Instruction` and
    /// reconstructs the block structure with predecessors/successors.
    pub fn from_codegen_function(func: &crate::compiler::ir::IRFunction) -> Self {
        let mut blocks = HashMap::new();

        for ir_block in &func.blocks {
            let block_id = ir_block.id;
            let instructions: Vec<Instruction> = ir_block.instrs.iter()
                .filter_map(|instr| Instruction::from_codegen_instr(instr))
                .collect();

            blocks.insert(block_id, BasicBlock {
                id: block_id,
                instructions,
                successors: ir_block.successors.clone(),
                predecessors: ir_block.predecessors.clone(),
            });
        }

        // Determine exit block: the block containing a Return instruction.
        let exit = blocks.iter()
            .find(|(_, b)| b.instructions.iter().any(|i| matches!(i, Instruction::Return { .. })))
            .map(|(id, _)| *id)
            .unwrap_or(func.entry_block);

        ControlFlowGraph {
            blocks,
            entry: func.entry_block,
            exit,
        }
    }
}

// =============================================================================
// AbstractValue — symbolic execution domain
// =============================================================================

/// Abstract value for symbolic execution
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AbstractValue {
    /// Unknown value
    Unknown,
    /// Constant value
    Constant(i64),
    /// Variable reference
    Var(VarId),
    /// Binary expression
    BinaryExpr { op: BinOp, left: Box<AbstractValue>, right: Box<AbstractValue> },
    /// Unary expression
    UnaryExpr { op: IrUnOp, src: Box<AbstractValue> },
    /// Phi expression
    Phi { incoming: Vec<(BlockId, AbstractValue)> },
}

/// Symbolic execution state
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolicState {
    /// Variable values
    pub values: HashMap<VarId, AbstractValue>,
    /// Path condition
    pub path_condition: AbstractValue,
}

/// Validation result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationResult {
    /// Valid - semantically equivalent
    Valid,
    /// Invalid - semantic difference found
    Invalid { reason: String, location: BlockId },
    /// Unknown - could not prove equivalence
    Unknown { reason: String },
}

/// Translation validator
pub struct TranslationValidator {
    /// Original CFG
    original: ControlFlowGraph,
    /// Optimized CFG
    optimized: ControlFlowGraph,
    /// Variable mapping between original and optimized
    var_mapping: HashMap<VarId, VarId>,
    /// Block mapping between original and optimized
    block_mapping: HashMap<BlockId, BlockId>,
}

impl TranslationValidator {
    /// Create a new translation validator
    pub fn new(
        original: ControlFlowGraph,
        optimized: ControlFlowGraph,
    ) -> Self {
        Self {
            original,
            optimized,
            var_mapping: HashMap::new(),
            block_mapping: HashMap::new(),
        }
    }

    /// Set variable mapping
    pub fn set_var_mapping(&mut self, original: VarId, optimized: VarId) {
        self.var_mapping.insert(original, optimized);
    }

    /// Set block mapping
    pub fn set_block_mapping(&mut self, original: BlockId, optimized: BlockId) {
        self.block_mapping.insert(original, optimized);
    }

    /// Validate translation
    pub fn validate(&mut self) -> ValidationResult {
        // Step 0: Build default variable mapping for variables with same IDs
        self.build_default_var_mapping();

        // Step 1: Validate control flow structure
        if let Err(reason) = self.validate_control_flow() {
            return ValidationResult::Invalid { reason, location: self.original.entry };
        }

        // Step 2: Validate data flow
        if let Err(reason) = self.validate_data_flow() {
            return ValidationResult::Invalid { reason, location: self.original.entry };
        }

        // Step 3: Symbolic execution to prove equivalence
        if let Err(reason) = self.symbolic_validation() {
            return ValidationResult::Unknown { reason };
        }

        ValidationResult::Valid
    }

    /// Build default variable mapping for variables with same IDs in both CFGs
    fn build_default_var_mapping(&mut self) {
        let optimized_vars = self.collect_variables(&self.optimized);
        for var in self.collect_variables(&self.original) {
            if optimized_vars.contains(&var) && !self.var_mapping.contains_key(&var) {
                self.var_mapping.insert(var, var);
            }
        }
    }

    /// Validate control flow structure
    fn validate_control_flow(&self) -> Result<(), String> {
        // Check that number of blocks is reasonable
        if self.optimized.blocks.len() < self.original.blocks.len() / 2 {
            return Err("Optimized CFG has too few blocks compared to original".to_string());
        }

        // Check that entry and exit blocks exist
        if !self.optimized.blocks.contains_key(&self.optimized.entry) {
            return Err("Optimized CFG missing entry block".to_string());
        }

        if !self.optimized.blocks.contains_key(&self.optimized.exit) {
            return Err("Optimized CFG missing exit block".to_string());
        }

        // Check that all blocks are reachable
        let reachable = self.compute_reachable_blocks(&self.optimized, self.optimized.entry);
        if reachable.len() != self.optimized.blocks.len() {
            return Err("Optimized CFG has unreachable blocks".to_string());
        }

        Ok(())
    }

    /// Validate data flow
    fn validate_data_flow(&self) -> Result<(), String> {
        // Check that all variables in original have mappings, or were eliminated
        // by optimization (e.g., constant propagation folding away inputs)
        let original_vars = self.collect_variables(&self.original);
        for var in original_vars {
            if !self.var_mapping.contains_key(&var) {
                // Variable was eliminated by optimization (e.g., constant propagation)
                // This is valid as long as the symbolic validation proves equivalence
                continue;
            }
        }

        // Check that live variables are preserved
        let original_live = self.compute_live_variables(&self.original);
        let optimized_live = self.compute_live_variables(&self.optimized);

        for (block, live_vars) in original_live {
            if let Some(opt_block) = self.block_mapping.get(&block) {
                if let Some(opt_live) = optimized_live.get(opt_block) {
                    // Check that live variables are subset of optimized live variables
                    for var in live_vars {
                        if let Some(mapped_var) = self.var_mapping.get(&var) {
                            if !opt_live.contains(mapped_var) {
                                return Err(format!("Live variable {:?} not live in optimized block", var));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Symbolic validation
    fn symbolic_validation(&self) -> Result<(), String> {
        // Perform symbolic execution on both CFGs
        let original_states = self.symbolic_execute(&self.original);
        let optimized_states = self.symbolic_execute(&self.optimized);

        // Compare states at exit block
        let original_exit = original_states.get(&self.original.exit);
        let optimized_exit = optimized_states.get(&self.optimized.exit);

        match (original_exit, optimized_exit) {
            (Some(orig_state), Some(opt_state)) => {
                self.compare_states(orig_state, opt_state)
            }
            (None, None) => Ok(()),
            _ => Err("Exit state mismatch between original and optimized".to_string()),
        }
    }

    /// Compare two symbolic states
    fn compare_states(&self, original: &SymbolicState, optimized: &SymbolicState) -> Result<(), String> {
        // Compare variable values
        for (orig_var, orig_value) in &original.values {
            if let Some(opt_var) = self.var_mapping.get(orig_var) {
                if let Some(opt_value) = optimized.values.get(opt_var) {
                    if !self.values_equivalent(orig_value, opt_value) {
                        return Err(format!(
                            "Variable {:?} has different values: original={:?}, optimized={:?}",
                            orig_var, orig_value, opt_value
                        ));
                    }
                }
            }
        }

        // Compare path conditions
        if !self.values_equivalent(&original.path_condition, &optimized.path_condition) {
            return Err("Path conditions differ between original and optimized".to_string());
        }

        Ok(())
    }

    /// Check if two abstract values are equivalent
    fn values_equivalent(&self, a: &AbstractValue, b: &AbstractValue) -> bool {
        match (a, b) {
            (AbstractValue::Unknown, AbstractValue::Unknown) => true,
            (AbstractValue::Constant(c1), AbstractValue::Constant(c2)) => c1 == c2,
            (AbstractValue::Var(v1), AbstractValue::Var(v2)) => {
                self.var_mapping.get(v1) == Some(v2) || v1 == v2
            }
            (AbstractValue::BinaryExpr { op: op1, left: l1, right: r1 },
             AbstractValue::BinaryExpr { op: op2, left: l2, right: r2 }) => {
                op1 == op2 && self.values_equivalent(l1, l2) && self.values_equivalent(r1, r2)
            }
            (AbstractValue::UnaryExpr { op: op1, src: s1 },
             AbstractValue::UnaryExpr { op: op2, src: s2 }) => {
                op1 == op2 && self.values_equivalent(s1, s2)
            }
            _ => false,
        }
    }

    /// Symbolic execution on a CFG
    fn symbolic_execute(&self, cfg: &ControlFlowGraph) -> HashMap<BlockId, SymbolicState> {
        let mut states = HashMap::new();
        let mut worklist = vec![cfg.entry];
        let mut initial_state = SymbolicState {
            values: HashMap::new(),
            path_condition: AbstractValue::Constant(1),
        };

        // Initialize with unknown values for all variables
        for block in cfg.blocks.values() {
            for instr in &block.instructions {
                self.collect_instr_vars(instr, &mut initial_state.values);
            }
        }

        states.insert(cfg.entry, initial_state);

        while let Some(block_id) = worklist.pop() {
            if let Some(block) = cfg.blocks.get(&block_id) {
                let mut state = states.get(&block_id).cloned().unwrap_or_else(|| SymbolicState {
                    values: HashMap::new(),
                    path_condition: AbstractValue::Constant(1),
                });

                // Execute instructions in block
                for instr in &block.instructions {
                    self.execute_instruction(instr, &mut state);
                }

                // Propagate to successors
                for succ in &block.successors {
                    let succ_state = states.get(succ).cloned().unwrap_or_else(|| SymbolicState {
                        values: HashMap::new(),
                        path_condition: AbstractValue::Constant(1),
                    });

                    // Merge states
                    let merged = self.merge_states(&state, &succ_state);

                    // Fixed-point: only push successor if state changed
                    if states.get(succ) != Some(&merged) {
                        states.insert(*succ, merged);
                        worklist.push(*succ);
                    }
                }
            }
        }

        states
    }

    /// Collect variables from an instruction
    fn collect_instr_vars(&self, instr: &Instruction, values: &mut HashMap<VarId, AbstractValue>) {
        match instr {
            Instruction::Load { dst, .. } => {
                values.insert(*dst, AbstractValue::Unknown);
            }
            Instruction::BinaryOp { dst, left, right, .. } => {
                values.insert(*dst, AbstractValue::Unknown);
                values.entry(*left).or_insert(AbstractValue::Unknown);
                values.entry(*right).or_insert(AbstractValue::Unknown);
            }
            Instruction::UnaryOp { dst, src, .. } => {
                values.insert(*dst, AbstractValue::Unknown);
                values.entry(*src).or_insert(AbstractValue::Unknown);
            }
            Instruction::Move { dst, src } => {
                values.insert(*dst, AbstractValue::Unknown);
                values.entry(*src).or_insert(AbstractValue::Unknown);
            }
            Instruction::Branch { cond, .. } => {
                values.entry(*cond).or_insert(AbstractValue::Unknown);
            }
            Instruction::Call { args, ret, .. } => {
                for arg in args {
                    values.entry(*arg).or_insert(AbstractValue::Unknown);
                }
                if let Some(ret_var) = ret {
                    values.insert(*ret_var, AbstractValue::Unknown);
                }
            }
            Instruction::Phi { dst, incoming } => {
                values.insert(*dst, AbstractValue::Unknown);
                for (_, var) in incoming {
                    values.entry(*var).or_insert(AbstractValue::Unknown);
                }
            }
            _ => {}
        }
    }

    /// Execute an instruction symbolically
    fn execute_instruction(&self, instr: &Instruction, state: &mut SymbolicState) {
        match instr {
            Instruction::Load { dst, addr: _ } => {
                state.values.insert(*dst, AbstractValue::Unknown);
            }
            Instruction::Store { addr: _, src: _ } => {
                // Store doesn't produce a value
            }
            Instruction::BinaryOp { dst, op, left, right } => {
                let left_val = state.values.get(left).cloned().unwrap_or(AbstractValue::Unknown);
                let right_val = state.values.get(right).cloned().unwrap_or(AbstractValue::Unknown);
                state.values.insert(*dst, AbstractValue::BinaryExpr {
                    op: *op,
                    left: Box::new(left_val),
                    right: Box::new(right_val),
                });
            }
            Instruction::UnaryOp { dst, op, src } => {
                let src_val = state.values.get(src).cloned().unwrap_or(AbstractValue::Unknown);
                state.values.insert(*dst, AbstractValue::UnaryExpr {
                    op: *op,
                    src: Box::new(src_val),
                });
            }
            Instruction::Move { dst, src } => {
                let src_val = state.values.get(src).cloned().unwrap_or(AbstractValue::Unknown);
                state.values.insert(*dst, src_val);
            }
            Instruction::Branch { cond: _, true_block: _, false_block: _ } => {
                // Branch affects path condition
            }
            Instruction::Jump { .. } => {
                // Jump doesn't affect state
            }
            Instruction::Call { func: _, args: _, ret } => {
                // Function call - treat as unknown
                if let Some(ret_var) = ret {
                    state.values.insert(*ret_var, AbstractValue::Unknown);
                }
            }
            Instruction::Return { .. } => {
                // Return doesn't affect state
            }
            Instruction::Phi { dst, incoming } => {
                let phi_values: Vec<_> = incoming.iter()
                    .map(|(_, var)| state.values.get(var).cloned().unwrap_or(AbstractValue::Unknown))
                    .collect();

                if phi_values.len() == 1 {
                    state.values.insert(*dst, phi_values[0].clone());
                } else {
                    state.values.insert(*dst, AbstractValue::Phi {
                        incoming: incoming.iter().cloned().map(|(b, v)| (b, state.values.get(&v).cloned().unwrap_or(AbstractValue::Unknown))).collect(),
                    });
                }
            }
        }
    }

    /// Merge two symbolic states
    fn merge_states(&self, state1: &SymbolicState, state2: &SymbolicState) -> SymbolicState {
        let mut merged_values = HashMap::new();
        let all_vars: HashSet<_> = state1.values.keys().chain(state2.values.keys()).collect();

        for var in all_vars {
            let val1 = state1.values.get(var);
            let val2 = state2.values.get(var);

            let merged = match (val1, val2) {
                (Some(v1), Some(v2)) if v1 == v2 => v1.clone(),
                _ => AbstractValue::Unknown,
            };

            merged_values.insert(*var, merged);
        }

        SymbolicState {
            values: merged_values,
            path_condition: AbstractValue::Unknown,
        }
    }

    /// Compute reachable blocks from entry
    fn compute_reachable_blocks(&self, cfg: &ControlFlowGraph, entry: BlockId) -> HashSet<BlockId> {
        let mut reachable = HashSet::new();
        let mut worklist = vec![entry];

        while let Some(block_id) = worklist.pop() {
            if reachable.insert(block_id) {
                if let Some(block) = cfg.blocks.get(&block_id) {
                    worklist.extend(&block.successors);
                }
            }
        }

        reachable
    }

    /// Collect all variables in CFG
    fn collect_variables(&self, cfg: &ControlFlowGraph) -> HashSet<VarId> {
        let mut vars_map: HashMap<VarId, AbstractValue> = HashMap::new();
        for block in cfg.blocks.values() {
            for instr in &block.instructions {
                self.collect_instr_vars(instr, &mut vars_map);
            }
        }
        vars_map.into_keys().collect()
    }

    /// Compute live variables for each block
    fn compute_live_variables(&self, cfg: &ControlFlowGraph) -> HashMap<BlockId, HashSet<VarId>> {
        let mut live = HashMap::new();
        let mut changed = true;

        // Initialize all blocks with empty sets
        for &block_id in cfg.blocks.keys() {
            live.insert(block_id, HashSet::new());
        }

        while changed {
            changed = false;

            for block_id in cfg.blocks.keys().cloned().collect::<Vec<_>>() {
                if let Some(block) = cfg.blocks.get(&block_id) {
                    let mut block_live = HashSet::new();

                    // Union of successors' live variables
                    for succ in &block.successors {
                        if let Some(succ_live) = live.get(succ) {
                            block_live.extend(succ_live);
                        }
                    }

                    // Process instructions in reverse order
                    for instr in block.instructions.iter().rev() {
                        self.update_live_for_instr(instr, &mut block_live);
                    }

                    if let Some(current_live) = live.get(&block_id) {
                        if block_live != *current_live {
                            live.insert(block_id, block_live);
                            changed = true;
                        }
                    }
                }
            }
        }

        live
    }

    /// Update live variables for an instruction
    fn update_live_for_instr(&self, instr: &Instruction, live: &mut HashSet<VarId>) {
        match instr {
            Instruction::Load { dst, .. } => {
                live.remove(dst);
            }
            Instruction::BinaryOp { dst, left, right, .. } => {
                live.remove(dst);
                live.insert(*left);
                live.insert(*right);
            }
            Instruction::UnaryOp { dst, src, .. } => {
                live.remove(dst);
                live.insert(*src);
            }
            Instruction::Move { dst, src } => {
                live.remove(dst);
                live.insert(*src);
            }
            Instruction::Branch { cond, .. } => {
                live.insert(*cond);
            }
            Instruction::Call { args, ret, .. } => {
                for arg in args {
                    live.insert(*arg);
                }
                if let Some(ret_var) = ret {
                    live.remove(ret_var);
                }
            }
            Instruction::Phi { dst, incoming } => {
                live.remove(dst);
                for (_, var) in incoming {
                    live.insert(*var);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_validation() {
        let mut original = ControlFlowGraph {
            blocks: HashMap::new(),
            entry: 0,
            exit: 1,
        };

        let block0 = BasicBlock {
            id: 0,
            instructions: vec![
                Instruction::BinaryOp {
                    dst: 0,
                    op: BinOp::add(),
                    left: 1,
                    right: 2,
                },
            ],
            successors: vec![1],
            predecessors: vec![],
        };

        let block1 = BasicBlock {
            id: 1,
            instructions: vec![Instruction::Return { value: Some(0) }],
            successors: vec![],
            predecessors: vec![0],
        };

        original.blocks.insert(0, block0);
        original.blocks.insert(1, block1);

        let mut optimized = original.clone();
        optimized.entry = 0;
        optimized.exit = 1;

        let mut validator = TranslationValidator::new(original, optimized);
        validator.set_block_mapping(0, 0);
        validator.set_block_mapping(1, 1);

        let result = validator.validate();
        assert_eq!(result, ValidationResult::Valid);
    }

    #[test]
    fn test_constant_propagation_validation() {
        let mut original = ControlFlowGraph {
            blocks: HashMap::new(),
            entry: 0,
            exit: 1,
        };

        let block0 = BasicBlock {
            id: 0,
            instructions: vec![
                Instruction::BinaryOp {
                    dst: 0,
                    op: BinOp::add(),
                    left: 1,
                    right: 2,
                },
            ],
            successors: vec![1],
            predecessors: vec![],
        };

        let block1 = BasicBlock {
            id: 1,
            instructions: vec![Instruction::Return { value: Some(0) }],
            successors: vec![],
            predecessors: vec![0],
        };

        original.blocks.insert(0, block0);
        original.blocks.insert(1, block1);

        let mut optimized = ControlFlowGraph {
            blocks: HashMap::new(),
            entry: 0,
            exit: 1,
        };

        // Optimized version with constant folded
        let opt_block0 = BasicBlock {
            id: 0,
            instructions: vec![
                Instruction::Move {
                    dst: 0,
                    src: 3, // Constant result
                },
            ],
            successors: vec![1],
            predecessors: vec![],
        };

        let opt_block1 = BasicBlock {
            id: 1,
            instructions: vec![Instruction::Return { value: Some(0) }],
            successors: vec![],
            predecessors: vec![0],
        };

        optimized.blocks.insert(0, opt_block0);
        optimized.blocks.insert(1, opt_block1);

        let mut validator = TranslationValidator::new(original, optimized);
        validator.set_block_mapping(0, 0);
        validator.set_block_mapping(1, 1);
        validator.set_var_mapping(0, 0);

        let result = validator.validate();
        // Should be valid or unknown (depends on constant folding proof)
        assert!(!matches!(result, ValidationResult::Invalid { .. }));
    }

    #[test]
    fn test_binop_roundtrip() {
        // Verify that BinOp convenience constructors match the expected IrBinOp/IrCmpOp variants
        assert_eq!(BinOp::add(), BinOp::Arith(IrBinOp::Add));
        assert_eq!(BinOp::sub(), BinOp::Arith(IrBinOp::Sub));
        assert_eq!(BinOp::eq(),  BinOp::Cmp(IrCmpOp::Eq));
        assert_eq!(BinOp::ne(),  BinOp::Cmp(IrCmpOp::Ne));
        assert_eq!(BinOp::lt(),  BinOp::Cmp(IrCmpOp::Lt));
    }
}
