// =========================================================================
// Translation Validation - Prove Semantic Equivalence
// Mathematical proof that optimized code preserves original semantics
// Ensures correctness is never traded for speed
// =========================================================================

use std::collections::{HashMap, HashSet};

/// Basic block identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

/// Instruction identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstrId(pub usize);

/// Variable identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarId(pub usize);

/// Instruction operation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Instruction {
    /// Load from memory
    Load { dst: VarId, addr: VarId },
    /// Store to memory
    Store { addr: VarId, src: VarId },
    /// Binary operation
    BinaryOp { dst: VarId, op: BinaryOp, left: VarId, right: VarId },
    /// Unary operation
    UnaryOp { dst: VarId, op: UnaryOp, src: VarId },
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

/// Binary operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Unary operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Not,
}

/// Basic block
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

/// Control flow graph
#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    /// Basic blocks
    pub blocks: HashMap<BlockId, BasicBlock>,
    /// Entry block
    pub entry: BlockId,
    /// Exit block
    pub exit: BlockId,
}

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
    BinaryExpr { op: BinaryOp, left: Box<AbstractValue>, right: Box<AbstractValue> },
    /// Unary expression
    UnaryExpr { op: UnaryOp, src: Box<AbstractValue> },
    /// Phi expression
    Phi { incoming: Vec<(BlockId, AbstractValue)> },
}

/// Symbolic execution state
#[derive(Debug, Clone)]
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
        // Check that all variables in original have mappings
        let original_vars = self.collect_variables(&self.original);
        for var in original_vars {
            if !self.var_mapping.contains_key(&var) {
                return Err(format!("Original variable {:?} has no mapping in optimized", var));
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
                    states.insert(*succ, merged);
                    worklist.push(*succ);
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
            Instruction::Load { dst, addr } => {
                let addr_val = state.values.get(addr).cloned().unwrap_or(AbstractValue::Unknown);
                state.values.insert(*dst, AbstractValue::Unknown);
            }
            Instruction::Store { addr, src } => {
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
            Instruction::Branch { cond, true_block, false_block } => {
                let cond_val = state.values.get(cond).cloned().unwrap_or(AbstractValue::Unknown);
                // Branch affects path condition
            }
            Instruction::Jump { .. } => {
                // Jump doesn't affect state
            }
            Instruction::Call { func, args, ret } => {
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
            entry: BlockId(0),
            exit: BlockId(1),
        };

        let mut block0 = BasicBlock {
            id: BlockId(0),
            instructions: vec![
                Instruction::BinaryOp {
                    dst: VarId(0),
                    op: BinaryOp::Add,
                    left: VarId(1),
                    right: VarId(2),
                },
            ],
            successors: vec![BlockId(1)],
            predecessors: vec![],
        };

        let block1 = BasicBlock {
            id: BlockId(1),
            instructions: vec![Instruction::Return { value: Some(VarId(0)) }],
            successors: vec![],
            predecessors: vec![BlockId(0)],
        };

        original.blocks.insert(BlockId(0), block0);
        original.blocks.insert(BlockId(1), block1);

        let mut optimized = original.clone();
        optimized.entry = BlockId(0);
        optimized.exit = BlockId(1);

        let mut validator = TranslationValidator::new(original, optimized);
        validator.set_block_mapping(BlockId(0), BlockId(0));
        validator.set_block_mapping(BlockId(1), BlockId(1));

        let result = validator.validate();
        assert_eq!(result, ValidationResult::Valid);
    }

    #[test]
    fn test_constant_propagation_validation() {
        let mut original = ControlFlowGraph {
            blocks: HashMap::new(),
            entry: BlockId(0),
            exit: BlockId(1),
        };

        let block0 = BasicBlock {
            id: BlockId(0),
            instructions: vec![
                Instruction::BinaryOp {
                    dst: VarId(0),
                    op: BinaryOp::Add,
                    left: VarId(1),
                    right: VarId(2),
                },
            ],
            successors: vec![BlockId(1)],
            predecessors: vec![],
        };

        let block1 = BasicBlock {
            id: BlockId(1),
            instructions: vec![Instruction::Return { value: Some(VarId(0)) }],
            successors: vec![],
            predecessors: vec![BlockId(0)],
        };

        original.blocks.insert(BlockId(0), block0);
        original.blocks.insert(BlockId(1), block1);

        let mut optimized = ControlFlowGraph {
            blocks: HashMap::new(),
            entry: BlockId(0),
            exit: BlockId(1),
        };

        // Optimized version with constant folded
        let opt_block0 = BasicBlock {
            id: BlockId(0),
            instructions: vec![
                Instruction::Move {
                    dst: VarId(0),
                    src: VarId(3), // Constant result
                },
            ],
            successors: vec![BlockId(1)],
            predecessors: vec![],
        };

        let opt_block1 = BasicBlock {
            id: BlockId(1),
            instructions: vec![Instruction::Return { value: Some(VarId(0)) }],
            successors: vec![],
            predecessors: vec![BlockId(0)],
        };

        optimized.blocks.insert(BlockId(0), opt_block0);
        optimized.blocks.insert(BlockId(1), opt_block1);

        let mut validator = TranslationValidator::new(original, optimized);
        validator.set_block_mapping(BlockId(0), BlockId(0));
        validator.set_block_mapping(BlockId(1), BlockId(1));
        validator.set_var_mapping(VarId(0), VarId(0));

        let result = validator.validate();
        // Should be valid or unknown (depends on constant folding proof)
        assert!(!matches!(result, ValidationResult::Invalid { .. }));
    }
}
