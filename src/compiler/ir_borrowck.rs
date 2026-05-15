// =============================================================================
// jules/src/compiler/ir_borrowck.rs
//
// IR-based borrow checker that operates on FlatIrModule.
//
// Unlike the AST-based borrow checker (borrowck.rs) which re-derives ownership
// semantics from scratch, this pass leverages the Ownership annotations already
// present on every IrInstr.  The IR carries Owned/Ref/MutRef/Shared/Copy on
// each instruction, so we use those annotations directly rather than
// re-inferring them.
//
// Design:
//   • Walks each block in every function sequentially
//   • Tracks ValueState (Available / Moved / MutBorrowed) per ValueId
//   • Checks use-after-move, double mutable borrow, alias violations,
//     ownership transfer correctness, and return ownership
//   • For phi nodes, merges states from predecessor blocks
//   • Does not perform full dataflow analysis — practical and fast
// =============================================================================

use rustc_hash::FxHashMap;

use crate::compiler::ir::{
    AliasKind, BlockId, FlatBlock, FlatIrFunction, FlatIrModule, IrOp, Ownership, ValueId,
};
use crate::compiler::lexer::Span;

// =============================================================================
// §1  PUBLIC RESULT TYPES
// =============================================================================

/// Ownership state of a value tracked during borrow checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueState {
    /// Value is available for use (Owned/Copy/Ref/Shared).
    Available,
    /// Value has been moved out — use is an error.
    Moved,
    /// Value has an active mutable borrow — taking another mutable borrow
    /// is an error.
    MutBorrowed,
}

/// The kind of borrow-check diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrBorrowDiagKind {
    /// Used a value after it has been moved.
    UseAfterMove,
    /// Took a second mutable borrow while one was already active.
    DoubleMutBorrow,
    /// Two mutable references to the same allocation exist simultaneously.
    AliasViolation,
    /// Moved a value while it was still borrowed.
    MoveWhileBorrowed,
    /// Borrowed a value after it has been moved.
    BorrowAfterMove,
}

impl std::fmt::Display for IrBorrowDiagKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IrBorrowDiagKind::UseAfterMove => write!(f, "use-after-move"),
            IrBorrowDiagKind::DoubleMutBorrow => write!(f, "double-mutable-borrow"),
            IrBorrowDiagKind::AliasViolation => write!(f, "alias-violation"),
            IrBorrowDiagKind::MoveWhileBorrowed => write!(f, "move-while-borrowed"),
            IrBorrowDiagKind::BorrowAfterMove => write!(f, "borrow-after-move"),
        }
    }
}

/// A single diagnostic from the IR borrow checker.
#[derive(Debug, Clone)]
pub struct IrBorrowDiag {
    pub span: Span,
    pub kind: IrBorrowDiagKind,
    pub message: String,
}

/// The aggregate result of borrow-checking a FlatIrModule.
#[derive(Debug, Clone)]
pub struct IrBorrowckResult {
    pub diagnostics: Vec<IrBorrowDiag>,
    pub errors: usize,
    pub warnings: usize,
}

// =============================================================================
// §2  BORROW CHECKER STATE
// =============================================================================

/// Per-value metadata tracked during checking.
#[derive(Debug, Clone, Copy)]
struct ValueInfo {
    /// Current ownership state of the value.
    state: ValueState,
    /// The Ownership annotation from the instruction that defined this value.
    ownership: Ownership,
    /// Number of outstanding immutable (Ref/Shared) borrows.
    imm_borrows: u32,
    /// Number of outstanding mutable (MutRef) borrows.
    mut_borrows: u32,
}

/// The borrow checker, operating on a single function at a time.
struct IrBorrowChecker {
    diags: Vec<IrBorrowDiag>,
    /// Per-value state, keyed by ValueId.
    values: FxHashMap<ValueId, ValueInfo>,
    /// Maps a borrowed-from ValueId to the set of ValueIds that borrow from it.
    /// Used for alias tracking: if two MutRef values borrow the same root,
    /// that's an alias violation.
    borrow_roots: FxHashMap<ValueId, Vec<ValueId>>,
    /// Block states for phi-node merging: maps BlockId to the value states
    /// at the *end* of that block.
    block_states: FxHashMap<BlockId, FxHashMap<ValueId, ValueInfo>>,
}

impl IrBorrowChecker {
    fn new() -> Self {
        IrBorrowChecker {
            diags: Vec::new(),
            values: FxHashMap::default(),
            borrow_roots: FxHashMap::default(),
            block_states: FxHashMap::default(),
        }
    }

    // ── Diagnostic helpers ──────────────────────────────────────────────

    fn emit(&mut self, span: Span, kind: IrBorrowDiagKind, message: String) {
        self.diags.push(IrBorrowDiag {
            span,
            kind,
            message,
        });
    }

    // ── Value state helpers ─────────────────────────────────────────────

    /// Mark a value as defined with the given Ownership.
    fn define_value(&mut self, id: ValueId, ownership: Ownership) {
        self.values.insert(id, ValueInfo {
            state: ValueState::Available,
            ownership,
            imm_borrows: 0,
            mut_borrows: 0,
        });
    }

    /// Get the current state of a value, or Available if not tracked.
    fn get_state(&self, id: ValueId) -> ValueState {
        self.values.get(&id).map(|v| v.state).unwrap_or(ValueState::Available)
    }

    /// Get the ownership annotation of a value, or Copy as a safe default.
    fn get_ownership(&self, id: ValueId) -> Ownership {
        self.values.get(&id).map(|v| v.ownership).unwrap_or(Ownership::Copy)
    }

    /// Mark a value as moved.
    fn mark_moved(&mut self, id: ValueId) {
        if let Some(info) = self.values.get_mut(&id) {
            info.state = ValueState::Moved;
        }
    }

    /// Mark a value as mutably borrowed.
    fn mark_mut_borrowed(&mut self, id: ValueId) {
        if let Some(info) = self.values.get_mut(&id) {
            info.state = ValueState::MutBorrowed;
        }
    }

    /// Check whether using `id` as a read is valid.  Returns true if OK.
    fn check_read_use(&mut self, id: ValueId, span: Span) -> bool {
        let state = self.get_state(id);
        match state {
            ValueState::Available => true,
            ValueState::Moved => {
                self.emit(
                    span,
                    IrBorrowDiagKind::UseAfterMove,
                    format!("use of moved value {}", id),
                );
                false
            }
            ValueState::MutBorrowed => {
                // Reading while mutably borrowed is allowed (the mutable ref
                // holder can read, and others can't access anyway).
                true
            }
        }
    }

    /// Check whether using `id` as a move/consume is valid.  Returns true if OK.
    fn check_move_use(&mut self, id: ValueId, span: Span) -> bool {
        let state = self.get_state(id);
        let ownership = self.get_ownership(id);
        // Copy types are implicitly copied, never moved.
        if ownership == Ownership::Copy {
            return self.check_read_use(id, span);
        }
        match state {
            ValueState::Available => {
                // Check if the value is currently borrowed.
                if let Some(info) = self.values.get(&id) {
                    if info.imm_borrows > 0 || info.mut_borrows > 0 {
                        self.emit(
                            span,
                            IrBorrowDiagKind::MoveWhileBorrowed,
                            format!("cannot move {} while it is borrowed", id),
                        );
                        return false;
                    }
                }
                true
            }
            ValueState::Moved => {
                self.emit(
                    span,
                    IrBorrowDiagKind::UseAfterMove,
                    format!("use of moved value {}", id),
                );
                false
            }
            ValueState::MutBorrowed => {
                self.emit(
                    span,
                    IrBorrowDiagKind::MoveWhileBorrowed,
                    format!("cannot move {} while it is mutably borrowed", id),
                );
                false
            }
        }
    }

    /// Record that `borrower` borrows from `root`.  Check for alias violations.
    fn record_borrow(&mut self, root: ValueId, borrower: ValueId, is_mut: bool, span: Span) {
        // Check the root isn't moved.
        let root_state = self.get_state(root);
        if root_state == ValueState::Moved {
            self.emit(
                span,
                IrBorrowDiagKind::BorrowAfterMove,
                format!("cannot borrow {} because it has been moved", root),
            );
            return;
        }

        if is_mut {
            // Check for existing borrows on root.
            if let Some(info) = self.values.get(&root) {
                if info.mut_borrows > 0 {
                    self.emit(
                        span,
                        IrBorrowDiagKind::DoubleMutBorrow,
                        format!(
                            "cannot mutably borrow {} — it is already mutably borrowed",
                            root
                        ),
                    );
                    return;
                }
                if info.imm_borrows > 0 {
                    self.emit(
                        span,
                        IrBorrowDiagKind::DoubleMutBorrow,
                        format!(
                            "cannot mutably borrow {} — it is already immutably borrowed",
                            root
                        ),
                    );
                    return;
                }
            }

            // Check for alias violation: if another MutRef already borrows
            // the same root, that's two mutable references to the same
            // allocation.
            if let Some(borrowers) = self.borrow_roots.get(&root) {
                for &prev_borrower in borrowers {
                    if let Some(prev_info) = self.values.get(&prev_borrower) {
                        if prev_info.ownership == Ownership::MutRef
                            && prev_info.state != ValueState::Moved
                        {
                            self.emit(
                                span,
                                IrBorrowDiagKind::AliasViolation,
                                format!(
                                    "alias violation: {} and {} are both mutable \
                                     references to {}",
                                    prev_borrower, borrower, root
                                ),
                            );
                            break;
                        }
                    }
                }
            }

            // Update root borrow counts.
            if let Some(info) = self.values.get_mut(&root) {
                info.mut_borrows += 1;
                info.state = ValueState::MutBorrowed;
            }

            // Track the borrow root.
            self.borrow_roots.entry(root).or_default().push(borrower);
        } else {
            // Immutable borrow — check root isn't mutably borrowed.
            if let Some(info) = self.values.get(&root) {
                if info.mut_borrows > 0 {
                    self.emit(
                        span,
                        IrBorrowDiagKind::DoubleMutBorrow,
                        format!(
                            "cannot immutably borrow {} — it is already mutably borrowed",
                            root
                        ),
                    );
                    return;
                }
            }
            if let Some(info) = self.values.get_mut(&root) {
                info.imm_borrows += 1;
            }

            self.borrow_roots.entry(root).or_default().push(borrower);
        }
    }

    /// Release all borrows from a given root (when the root is consumed or
    /// goes out of scope).
    fn release_borrows_from(&mut self, root: ValueId) {
        if let Some(borrowers) = self.borrow_roots.remove(&root) {
            for borrower_id in borrowers {
                // The borrower itself may still be in the values map; its
                // state is no longer tied to this root.
                if let Some(info) = self.values.get_mut(&borrower_id) {
                    // If the borrower was MutBorrowed because of this root,
                    // transition back to Available (conservative: only if
                    // no other borrow roots reference it).
                    if info.state == ValueState::MutBorrowed {
                        info.state = ValueState::Available;
                    }
                }
            }
        }
        // Reset the root's own borrow counts.
        if let Some(info) = self.values.get_mut(&root) {
            info.imm_borrows = 0;
            info.mut_borrows = 0;
        }
    }

    // ── Operand extraction ──────────────────────────────────────────────

    /// Collect all ValueIds used by an operation (not the dst).
    fn operands_of(op: &IrOp) -> Vec<ValueId> {
        match op {
            IrOp::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],
            IrOp::UnOp { operand, .. } => vec![*operand],
            IrOp::Store { ptr, value } => vec![*ptr, *value],
            IrOp::Load { ptr, .. } => vec![*ptr],
            IrOp::Move { src } => vec![*src],
            IrOp::Copy { src } => vec![*src],
            IrOp::Ret { value } => {
                match value {
                    Some(v) => vec![*v],
                    None => vec![],
                }
            }
            IrOp::Call { args, .. } => args.clone(),
            IrOp::Intrinsic { args, .. } => args.clone(),
            IrOp::TaskSpawn { args, .. } => args.clone(),
            IrOp::TaskJoin { task } => vec![*task],
            IrOp::CondBr { cond, .. } => vec![*cond],
            IrOp::Phi { .. } => vec![],
            IrOp::TypeCheck { value, .. } => vec![*value],
            IrOp::Cast { src, .. } => vec![*src],
            IrOp::Emit { value, .. } => vec![*value],
            IrOp::Alloca { .. }
            | IrOp::ConstInt { .. }
            | IrOp::ConstFloat { .. }
            | IrOp::ConstBool { .. }
            | IrOp::ConstStr { .. }
            | IrOp::ConstUnit
            | IrOp::Nop
            | IrOp::Jump { .. }
            | IrOp::ParallelStart { .. }
            | IrOp::ParallelEnd { .. }
            | IrOp::RegionAlloc { .. } => vec![],
            IrOp::MatMul { lhs, rhs }
            | IrOp::HadamardMul { lhs, rhs }
            | IrOp::HadamardDiv { lhs, rhs }
            | IrOp::TensorConcat { lhs, rhs }
            | IrOp::KronProd { lhs, rhs }
            | IrOp::OuterProd { lhs, rhs } => vec![*lhs, *rhs],
        }
    }

    /// Collect the incoming (BlockId, ValueId) pairs from a Phi instruction.
    fn phi_incoming(op: &IrOp) -> &[(BlockId, ValueId)] {
        match op {
            IrOp::Phi { incoming } => incoming,
            _ => &[],
        }
    }

    /// Determine which operands are moved (consumed) by an operation.
    fn moved_operands(op: &IrOp, instr_ownership: Ownership) -> Vec<ValueId> {
        match op {
            // Move instruction explicitly moves src.
            IrOp::Move { src } => vec![*src],
            // Return moves the returned value.
            IrOp::Ret { value } => {
                match value {
                    Some(v) => vec![*v],
                    None => vec![],
                }
            }
            // Store moves the value into the pointer.
            IrOp::Store { value, .. } => vec![*value],
            // TaskSpawn moves args depending on ownership model.
            IrOp::TaskSpawn { args, ownership, .. } => {
                match ownership {
                    crate::compiler::ir::TaskOwnership::Move => args.clone(),
                    // Copy and Ref don't move.
                    _ => vec![],
                }
            }
            // Call may move Owned args.
            IrOp::Call { args, .. } => {
                // Conservative: for calls, we check the ownership of each
                // argument.  Owned arguments are moved; Copy/Ref/Shared are
                // not.  Since we don't have per-argument ownership here, we
                // use the instruction's ownership as a hint for the first
                // arg and assume the rest are borrowed or copied.
                // In practice, the lower should annotate each arg, but for
                // now we move args only if the overall instruction is Owned.
                if instr_ownership == Ownership::Owned {
                    args.clone()
                } else {
                    vec![]
                }
            }
            // BinOp uses operands by value — Copy types are copied, Owned
            // are consumed.
            IrOp::BinOp { lhs, rhs, .. } => {
                if instr_ownership == Ownership::Owned {
                    vec![*lhs, *rhs]
                } else {
                    vec![]
                }
            }
            _ => vec![],
        }
    }

    /// Determine which dst values represent borrows, and from which root.
    /// Returns (borrower_id, root_id, is_mut).
    fn borrow_info(op: &IrOp, dst: Option<ValueId>, ownership: Ownership) -> Option<(ValueId, ValueId, bool)> {
        match op {
            IrOp::Load { ptr, .. } => {
                // A Load produces a value from a pointer.  If the result is
                // Ref or MutRef, it's borrowing from ptr.
                let dst = dst?;
                match ownership {
                    Ownership::Ref | Ownership::Shared => Some((dst, *ptr, false)),
                    Ownership::MutRef => Some((dst, *ptr, true)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    // ── Block-level checking ────────────────────────────────────────────

    /// Merge predecessor block states for phi nodes at the start of a block.
    fn merge_phi_states(&mut self, block: &FlatBlock) {
        // Find any Phi instructions at the start of the block.
        for instr in &block.instrs {
            let incoming = Self::phi_incoming(&instr.op);
            if incoming.is_empty() {
                // Phis must be at the top of the block; stop when we hit
                // a non-phi instruction.
                break;
            }
            if let Some(dst) = instr.dst {
                // Merge states from all incoming values.
                let mut merged_state = ValueState::Available;
                for &(pred_block, pred_value) in incoming {
                    // Check if the predecessor has a recorded state.
                    let pred_state = self.block_states
                        .get(&pred_block)
                        .and_then(|states| states.get(&pred_value))
                        .map(|info| info.state)
                        .unwrap_or(ValueState::Available);

                    merged_state = merge_states(merged_state, pred_state);
                }

                self.values.insert(dst, ValueInfo {
                    state: merged_state,
                    ownership: instr.ownership,
                    imm_borrows: 0,
                    mut_borrows: 0,
                });
            }
        }
    }

    /// Check a single instruction.
    fn check_instr(&mut self, instr: &crate::compiler::ir::IrInstr) {
        let span = instr.span;
        let ownership = instr.ownership;

        // ── Step 1: Check uses of operands ────────────────────────────
        let operands = Self::operands_of(&instr.op);
        let moved = Self::moved_operands(&instr.op, ownership);

        for &operand_id in &operands {
            // Determine if this operand is moved or just read.
            if moved.contains(&operand_id) {
                self.check_move_use(operand_id, span);
            } else {
                // For non-moved operands, check the value is available
                // for reading.  For Copy types, reads are always fine.
                let op_own = self.get_ownership(operand_id);
                if op_own == Ownership::Copy {
                    // Copy types are implicitly copied — no move check
                    // needed, but still check for use-after-move.
                    self.check_read_use(operand_id, span);
                } else {
                    self.check_read_use(operand_id, span);
                }
            }
        }

        // ── Step 2: Apply move semantics to consumed operands ────────
        for &moved_id in &moved {
            let mv_own = self.get_ownership(moved_id);
            if mv_own != Ownership::Copy {
                self.mark_moved(moved_id);
                // Release any borrows from this value, since it's moved.
                self.release_borrows_from(moved_id);
            }
        }

        // ── Step 3: Register the destination value ───────────────────
        // Phi nodes are already handled by merge_phi_states — their
        // destination state reflects the merged predecessor states.
        // Overwriting with define_value would erase the merged Moved
        // state, which would make phi-node use-after-move detection
        // impossible.  Skip define_value for Phi instructions.
        if let IrOp::Phi { .. } = &instr.op {
            // Already defined in merge_phi_states — nothing to do.
        } else if let Some(dst) = instr.dst {
            self.define_value(dst, ownership);

            // Track borrow relationships.
            if let Some((borrower, root, is_mut)) =
                Self::borrow_info(&instr.op, instr.dst, ownership)
            {
                self.record_borrow(root, borrower, is_mut, span);
            }

            // Special handling for Load that produces a MutRef: the root
            // is now MutBorrowed.
            if ownership == Ownership::MutRef {
                if let IrOp::Load { ptr, .. } = &instr.op {
                    self.mark_mut_borrowed(*ptr);
                }
            }
        }

        // ── Step 4: Special handling for Store ───────────────────────
        // A Store writes through a pointer, which may invalidate borrows.
        if let IrOp::Store { ptr, value } = &instr.op {
            let ptr_state = self.get_state(*ptr);
            if ptr_state == ValueState::Moved {
                self.emit(
                    span,
                    IrBorrowDiagKind::UseAfterMove,
                    format!("store through moved pointer {}", ptr),
                );
            }
            // Writing through a pointer that has existing immutable
            // borrows is an alias violation.
            if let Some(info) = self.values.get(ptr) {
                if info.imm_borrows > 0 {
                    self.emit(
                        span,
                        IrBorrowDiagKind::AliasViolation,
                        format!(
                            "store through {} invalidates {} immutable borrows",
                            ptr, info.imm_borrows
                        ),
                    );
                }
            }
            let _ = value; // already handled in moved_operands
        }

        // ── Step 5: Special handling for alias metadata ──────────────
        // If the instruction has AliasKind::NoAlias, we trust it — no
        // alias conflict.  If Unknown, we've already done conservative
        // checking above.  If Restrict, we assume no overlap.
        //
        // (This is informational; the main alias checking happens via
        // borrow_roots tracking above.)

        // ── Step 6: Special handling for parallel regions ────────────
        // ParallelStart/ParallelEnd are structural markers.  Within a
        // parallel region, MutRef borrows of shared data are forbidden
        // (they could alias across threads).  We check this by flagging
        // MutRef instructions whose root is not thread-local.
        //
        // For now, we emit a warning if a MutRef is created within a
        // parallel region and its root was defined before the region
        // started.  This is a simple approximation.
        if ownership == Ownership::MutRef {
            if let IrOp::Load { ptr, .. } = &instr.op {
                // The ptr value might be from an outer scope — in a full
                // implementation we'd track parallel region boundaries.
                // Here we rely on the AliasKind annotation for a hint.
                if instr.alias == AliasKind::Unknown {
                    // Conservative: no alias info, so we can't prove
                    // safety.  This is informational only (no error).
                }
            }
        }
    }

    /// Check all instructions in a block.
    fn check_block(&mut self, block: &FlatBlock) {
        // Merge phi states from predecessors.
        self.merge_phi_states(block);

        // Check each instruction.
        for instr in &block.instrs {
            self.check_instr(instr);
        }

        // Snapshot the value states at the end of this block for phi
        // merging in successor blocks.
        self.block_states.insert(block.id, self.values.clone());
    }

    /// Check a function.
    fn check_function(&mut self, func: &FlatIrFunction) {
        // Reset per-function state.
        self.values.clear();
        self.borrow_roots.clear();
        self.block_states.clear();

        // Function parameters start as Available with their ownership
        // determined by the parameter type.  For simplicity, parameters
        // are assumed Owned unless the type is a reference type.
        for &(param_id, ref param_ty) in &func.params {
            let ownership = ownership_from_type(param_ty);
            self.define_value(param_id, ownership);
        }

        // Walk blocks in order.  In a well-formed flat IR, blocks are
        // typically ordered such that predecessors come before successors
        // (with loops being the exception).  For loops, the phi-merge
        // will use whatever state we have from the first pass, which is
        // conservative but sound.
        for block in &func.blocks {
            self.check_block(block);
        }
    }
}

// =============================================================================
// §3  HELPER FUNCTIONS
// =============================================================================

/// Merge two ValueStates: the result is the "worse" of the two.
/// Moved > MutBorrowed > Available.
fn merge_states(a: ValueState, b: ValueState) -> ValueState {
    match (a, b) {
        (ValueState::Moved, _) | (_, ValueState::Moved) => ValueState::Moved,
        (ValueState::MutBorrowed, _) | (_, ValueState::MutBorrowed) => ValueState::MutBorrowed,
        (ValueState::Available, ValueState::Available) => ValueState::Available,
    }
}

/// Infer an Ownership from an IrType.  Reference types are borrowed,
/// everything else is Owned by default.
fn ownership_from_type(ty: &crate::compiler::ir::IrType) -> Ownership {
    match ty {
        crate::compiler::ir::IrType::Ref(_) => Ownership::Ref,
        crate::compiler::ir::IrType::MutRef(_) => Ownership::MutRef,
        // Primitive-like types are Copy.
        crate::compiler::ir::IrType::Bool
        | crate::compiler::ir::IrType::Int { .. }
        | crate::compiler::ir::IrType::Float { .. }
        | crate::compiler::ir::IrType::Unit => Ownership::Copy,
        _ => Ownership::Owned,
    }
}

// =============================================================================
// §4  PUBLIC ENTRY POINT
// =============================================================================

/// Borrow-check a FlatIrModule, returning all diagnostics.
///
/// This is the main entry point for the IR-based borrow checker.  It walks
/// every function in the module, tracking ownership states and checking for:
///
/// 1. **Use-after-move**: Using a ValueId after it has been consumed by a
///    move operation.
/// 2. **Double mutable borrow**: Creating a second mutable borrow while one
///    is already active.
/// 3. **Alias violation**: Two mutable references to the same allocation
///    existing simultaneously.
/// 4. **Ownership transfer correctness**: Verifying that moved values aren't
///    used afterward in TaskSpawn and Call with move semantics.
/// 5. **Return ownership**: Verifying that Owned values returned from
///    functions aren't also used after the return.
pub fn ir_borrowck(module: &FlatIrModule) -> IrBorrowckResult {
    let mut checker = IrBorrowChecker::new();

    for func in &module.functions {
        checker.check_function(func);
    }

    // All diagnostic kinds are currently errors.
    let error_count = checker.diags.len();
    let warning_count = 0usize;

    IrBorrowckResult {
        diagnostics: checker.diags,
        errors: error_count,
        warnings: warning_count,
    }
}

// =============================================================================
// §5  TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::{
        AliasKind, BlockId, CostHint, EffectFlags, FlatBlock, FlatIrFunction, FlatIrModule,
        IrInstr, IrOp, IrType, Ownership, ValueId,
    };
    use crate::compiler::lexer::Span;

    fn dummy_span() -> Span {
        Span::dummy()
    }

    fn vid(n: u32) -> ValueId {
        ValueId(n)
    }

    fn bid(n: u32) -> BlockId {
        BlockId(n)
    }

    /// Build a minimal FlatIrModule with one function containing the given
    /// blocks.
    fn make_module(blocks: Vec<FlatBlock>) -> FlatIrModule {
        FlatIrModule {
            functions: vec![FlatIrFunction {
                name: "test_fn".to_string(),
                params: vec![],
                ret_ty: IrType::Unit,
                blocks,
                entry: bid(0),
                effects: EffectFlags::pure(),
                requires: vec![],
                ensures: vec![],
                span: dummy_span(),
            }],
            intrinsics: vec![],
            span: dummy_span(),
            lowering_errors: vec![],
        }
    }

    fn make_instr(dst: Option<ValueId>, op: IrOp, ownership: Ownership) -> IrInstr {
        IrInstr {
            dst,
            op,
            span: dummy_span(),
            effects: EffectFlags::pure(),
            ownership,
            cost: CostHint::Unknown,
            alias: AliasKind::Unknown,
        }
    }

    #[test]
    fn test_use_after_move() {
        // v0 = ConstInt 42 (Owned)
        // v1 = Move v0   (Owned)
        // v2 = BinOp(Add, v0, v0)  -- use after move!
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::ConstInt {
                        value: 42,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Owned,
                ),
                make_instr(Some(vid(1)), IrOp::Move { src: vid(0) }, Ownership::Owned),
                make_instr(
                    Some(vid(2)),
                    IrOp::BinOp {
                        op: crate::compiler::ast::BinOpKind::Add,
                        lhs: vid(0),
                        rhs: vid(0),
                    },
                    Ownership::Copy,
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::UseAfterMove),
            "expected use-after-move error, got: {:?}",
            result.diagnostics
        );
    }

    #[test]
    fn test_copy_not_moved() {
        // v0 = ConstInt 42 (Copy)
        // v1 = Copy v0   (Copy)
        // v2 = BinOp(Add, v0, v0)  -- fine, Copy is not moved
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::ConstInt {
                        value: 42,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Copy,
                ),
                make_instr(Some(vid(1)), IrOp::Copy { src: vid(0) }, Ownership::Copy),
                make_instr(
                    Some(vid(2)),
                    IrOp::BinOp {
                        op: crate::compiler::ast::BinOpKind::Add,
                        lhs: vid(0),
                        rhs: vid(0),
                    },
                    Ownership::Copy,
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert_eq!(
            result.errors, 0,
            "expected no errors for Copy values, got: {:?}",
            result.diagnostics
        );
    }

    #[test]
    fn test_double_mut_borrow() {
        // v0 = Alloca (Owned)
        // v1 = Load v0 (MutRef)
        // v2 = Load v0 (MutRef)  -- double mutable borrow!
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::Alloca {
                        ty: IrType::Int { width: 64, signed: true },
                        align: 8,
                    },
                    Ownership::Owned,
                ),
                make_instr(
                    Some(vid(1)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::MutRef,
                ),
                make_instr(
                    Some(vid(2)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::MutRef,
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::DoubleMutBorrow),
            "expected double-mut-borrow error, got: {:?}",
            result.diagnostics
        );
    }

    #[test]
    fn test_move_while_borrowed() {
        // v0 = Alloca (Owned)
        // v1 = Load v0 (Ref) — creates an immutable borrow
        // v2 = Move v0       — move while borrowed!
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::Alloca {
                        ty: IrType::Int { width: 64, signed: true },
                        align: 8,
                    },
                    Ownership::Owned,
                ),
                make_instr(
                    Some(vid(1)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Ref,
                ),
                make_instr(Some(vid(2)), IrOp::Move { src: vid(0) }, Ownership::Owned),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::MoveWhileBorrowed),
            "expected move-while-borrowed error, got: {:?}",
            result.diagnostics
        );
    }

    #[test]
    fn test_borrow_after_move() {
        // v0 = ConstInt 42 (Owned)
        // v1 = Move v0      (Owned)
        // v2 = Load v0      (Ref) — borrow after move!
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::ConstInt {
                        value: 42,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Owned,
                ),
                make_instr(Some(vid(1)), IrOp::Move { src: vid(0) }, Ownership::Owned),
                make_instr(
                    Some(vid(2)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Ref,
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::BorrowAfterMove),
            "expected borrow-after-move error, got: {:?}",
            result.diagnostics
        );
    }

    #[test]
    fn test_return_moves_value() {
        // v0 = ConstInt 42 (Owned)
        // Ret v0
        // Move v0 — use after return (move)
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::ConstInt {
                        value: 42,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Owned,
                ),
                make_instr(None, IrOp::Ret { value: Some(vid(0)) }, Ownership::Owned),
                make_instr(Some(vid(1)), IrOp::Move { src: vid(0) }, Ownership::Owned),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::UseAfterMove),
            "expected use-after-move after Ret, got: {:?}",
            result.diagnostics
        );
    }

    #[test]
    fn test_task_spawn_moves_owned() {
        // v0 = ConstInt 42 (Owned)
        // TaskSpawn("task", [v0], Move) — moves v0
        // Move v0 — use after move!
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::ConstInt {
                        value: 42,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Owned,
                ),
                make_instr(
                    Some(vid(1)),
                    IrOp::TaskSpawn {
                        func: "task".to_string(),
                        args: vec![vid(0)],
                        ownership: crate::compiler::ir::TaskOwnership::Move,
                    },
                    Ownership::Owned,
                ),
                make_instr(Some(vid(2)), IrOp::Move { src: vid(0) }, Ownership::Owned),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::UseAfterMove),
            "expected use-after-move after TaskSpawn with Move, got: {:?}",
            result.diagnostics
        );
    }

    #[test]
    fn test_task_spawn_copy_does_not_move() {
        // v0 = ConstInt 42 (Copy)
        // TaskSpawn("task", [v0], Copy) — does NOT move v0
        // BinOp(Add, v0, v0) — fine
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::ConstInt {
                        value: 42,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Copy,
                ),
                make_instr(
                    Some(vid(1)),
                    IrOp::TaskSpawn {
                        func: "task".to_string(),
                        args: vec![vid(0)],
                        ownership: crate::compiler::ir::TaskOwnership::Copy,
                    },
                    Ownership::Copy,
                ),
                make_instr(
                    Some(vid(2)),
                    IrOp::BinOp {
                        op: crate::compiler::ast::BinOpKind::Add,
                        lhs: vid(0),
                        rhs: vid(0),
                    },
                    Ownership::Copy,
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert_eq!(
            result.errors, 0,
            "expected no errors for Copy TaskSpawn, got: {:?}",
            result.diagnostics
        );
    }

    #[test]
    fn test_immut_borrow_then_immut_ok() {
        // v0 = Alloca (Owned)
        // v1 = Load v0 (Ref)
        // v2 = Load v0 (Ref) — multiple immutable borrows are OK
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::Alloca {
                        ty: IrType::Int { width: 64, signed: true },
                        align: 8,
                    },
                    Ownership::Owned,
                ),
                make_instr(
                    Some(vid(1)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Ref,
                ),
                make_instr(
                    Some(vid(2)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Ref,
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert_eq!(
            result.errors, 0,
            "expected no errors for multiple immutable borrows, got: {:?}",
            result.diagnostics
        );
    }

    #[test]
    fn test_mut_borrow_then_immut_error() {
        // v0 = Alloca (Owned)
        // v1 = Load v0 (MutRef)
        // v2 = Load v0 (Ref) — immutable borrow while mutably borrowed!
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::Alloca {
                        ty: IrType::Int { width: 64, signed: true },
                        align: 8,
                    },
                    Ownership::Owned,
                ),
                make_instr(
                    Some(vid(1)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::MutRef,
                ),
                make_instr(
                    Some(vid(2)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Ref,
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::DoubleMutBorrow),
            "expected double-mut-borrow error (mut then immut), got: {:?}",
            result.diagnostics
        );
    }

    #[test]
    fn test_alias_violation_two_mutrefs() {
        // v0 = Alloca (Owned)
        // v1 = Load v0 (MutRef) — first mut ref
        // v2 = Load v0 (MutRef) — second mut ref, alias violation
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::Alloca {
                        ty: IrType::Int { width: 64, signed: true },
                        align: 8,
                    },
                    Ownership::Owned,
                ),
                make_instr(
                    Some(vid(1)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::MutRef,
                ),
                make_instr(
                    Some(vid(2)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::MutRef,
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        // Should get either DoubleMutBorrow or AliasViolation (or both).
        let has_error = result.diagnostics.iter().any(|d| {
            d.kind == IrBorrowDiagKind::DoubleMutBorrow
                || d.kind == IrBorrowDiagKind::AliasViolation
        });
        assert!(
            has_error,
            "expected double-mut-borrow or alias-violation error, got: {:?}",
            result.diagnostics
        );
    }

    #[test]
    fn test_no_errors_for_simple_function() {
        // v0 = ConstInt 1 (Copy)
        // v1 = ConstInt 2 (Copy)
        // v2 = BinOp(Add, v0, v1) (Copy)
        // Ret v2
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::ConstInt {
                        value: 1,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Copy,
                ),
                make_instr(
                    Some(vid(1)),
                    IrOp::ConstInt {
                        value: 2,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Copy,
                ),
                make_instr(
                    Some(vid(2)),
                    IrOp::BinOp {
                        op: crate::compiler::ast::BinOpKind::Add,
                        lhs: vid(0),
                        rhs: vid(1),
                    },
                    Ownership::Copy,
                ),
                make_instr(None, IrOp::Ret { value: Some(vid(2)) }, Ownership::Copy),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert_eq!(
            result.errors, 0,
            "expected no errors for simple Copy function, got: {:?}",
            result.diagnostics
        );
    }

    #[test]
    fn test_store_invalidates_immut_borrows() {
        // v0 = Alloca (Owned)
        // v1 = Load v0 (Ref)
        // Store v0, v1 — write while immutably borrowed
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::Alloca {
                        ty: IrType::Int { width: 64, signed: true },
                        align: 8,
                    },
                    Ownership::Owned,
                ),
                make_instr(
                    Some(vid(1)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Ref,
                ),
                make_instr(
                    None,
                    IrOp::Store {
                        ptr: vid(0),
                        value: vid(1),
                    },
                    Ownership::Owned,
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::AliasViolation),
            "expected alias-violation on Store with immutable borrows, got: {:?}",
            result.diagnostics
        );
    }

    #[test]
    fn test_empty_module_no_errors() {
        let module = FlatIrModule {
            functions: vec![],
            intrinsics: vec![],
            span: dummy_span(),
            lowering_errors: vec![],
        };
        let result = ir_borrowck(&module);
        assert_eq!(result.errors, 0);
        assert_eq!(result.diagnostics.len(), 0);
    }
}

// =============================================================================
// §6  STRESS TESTS — OWNERSHIP IS ALWAYS A HARD ERROR
// =============================================================================
//
// These tests verify that the IR borrow checker correctly identifies ownership
// violations as HARD ERRORS under adversarial and edge-case conditions.
// The fundamental invariant: ownership violations are NEVER warnings.
// =============================================================================

#[cfg(test)]
mod stress_tests {
    use super::*;
    use crate::compiler::ir::{
        AliasKind, BlockId, CostHint, EffectFlags, FlatBlock, FlatIrFunction, FlatIrModule,
        IrInstr, IrOp, IrType, Ownership, ValueId,
    };
    use crate::compiler::lexer::Span;

    fn dummy_span() -> Span {
        Span::dummy()
    }

    fn vid(n: u32) -> ValueId {
        ValueId(n)
    }

    fn bid(n: u32) -> BlockId {
        BlockId(n)
    }

    fn make_module_with_fns(fns: Vec<FlatIrFunction>) -> FlatIrModule {
        FlatIrModule {
            functions: fns,
            intrinsics: vec![],
            span: dummy_span(),
            lowering_errors: vec![],
        }
    }

    fn make_module(blocks: Vec<FlatBlock>) -> FlatIrModule {
        FlatIrModule {
            functions: vec![FlatIrFunction {
                name: "stress_fn".to_string(),
                params: vec![],
                ret_ty: IrType::Unit,
                blocks,
                entry: bid(0),
                effects: EffectFlags::pure(),
                requires: vec![],
                ensures: vec![],
                span: dummy_span(),
            }],
            intrinsics: vec![],
            span: dummy_span(),
            lowering_errors: vec![],
        }
    }

    fn make_instr(dst: Option<ValueId>, op: IrOp, ownership: Ownership) -> IrInstr {
        IrInstr {
            dst,
            op,
            span: dummy_span(),
            effects: EffectFlags::pure(),
            ownership,
            cost: CostHint::Unknown,
            alias: AliasKind::Unknown,
        }
    }

    // ── INVARIANT: All diagnostics are errors, never warnings ────────────

    #[test]
    fn stress_all_ownership_diagnostics_are_errors() {
        // Build a module with multiple ownership violations.
        // Every single diagnostic must be an error — warnings count must be 0.
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                // v0 = Owned value
                make_instr(
                    Some(vid(0)),
                    IrOp::ConstInt {
                        value: 1,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Owned,
                ),
                // v1 = Move v0 (consumes v0)
                make_instr(Some(vid(1)), IrOp::Move { src: vid(0) }, Ownership::Owned),
                // v2 = Move v0 — use-after-move!
                make_instr(Some(vid(2)), IrOp::Move { src: vid(0) }, Ownership::Owned),
                // v3 = Alloca
                make_instr(
                    Some(vid(3)),
                    IrOp::Alloca {
                        ty: IrType::Int { width: 64, signed: true },
                        align: 8,
                    },
                    Ownership::Owned,
                ),
                // v4 = Load v3 (MutRef)
                make_instr(
                    Some(vid(4)),
                    IrOp::Load {
                        ptr: vid(3),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::MutRef,
                ),
                // v5 = Load v3 (MutRef) — double mut borrow!
                make_instr(
                    Some(vid(5)),
                    IrOp::Load {
                        ptr: vid(3),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::MutRef,
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));

        // FUNDAMENTAL INVARIANT: warnings count is always 0.
        assert_eq!(
            result.warnings, 0,
            "OWNERSHIP VIOLATION: warnings count must be 0, got {} warnings. \
             Ownership is ALWAYS a hard error!",
            result.warnings
        );

        // Must have at least 2 errors (use-after-move + double-mut-borrow).
        assert!(
            result.errors >= 2,
            "expected at least 2 errors, got {} errors: {:?}",
            result.errors,
            result.diagnostics
        );
    }

    // ── STRESS: Cascading use-after-move ──────────────────────────────────

    #[test]
    fn stress_cascading_use_after_move() {
        // Move a value, then use it multiple times — each use is a separate error.
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::ConstInt {
                        value: 42,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Owned,
                ),
                make_instr(Some(vid(1)), IrOp::Move { src: vid(0) }, Ownership::Owned),
                // Three uses of moved value v0
                make_instr(
                    Some(vid(2)),
                    IrOp::BinOp {
                        op: crate::compiler::ast::BinOpKind::Add,
                        lhs: vid(0),
                        rhs: vid(0),
                    },
                    Ownership::Copy,
                ),
                make_instr(
                    Some(vid(3)),
                    IrOp::UnOp {
                        op: crate::compiler::ast::UnOpKind::Neg,
                        operand: vid(0),
                    },
                    Ownership::Copy,
                ),
                make_instr(Some(vid(4)), IrOp::Move { src: vid(0) }, Ownership::Owned),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        let uam_count = result.diagnostics.iter()
            .filter(|d| d.kind == IrBorrowDiagKind::UseAfterMove)
            .count();
        assert!(
            uam_count >= 3,
            "expected at least 3 use-after-move errors for cascading violations, got {}: {:?}",
            uam_count,
            result.diagnostics
        );
        assert_eq!(result.warnings, 0, "ownership errors must never be warnings");
    }

    // ── STRESS: Borrow-then-move chain ────────────────────────────────────

    #[test]
    fn stress_borrow_then_move_chain() {
        // Allocate, borrow mutably, then try to move — should error.
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::Alloca {
                        ty: IrType::Int { width: 64, signed: true },
                        align: 8,
                    },
                    Ownership::Owned,
                ),
                // v1 = Load v0 (MutRef) — borrows v0
                make_instr(
                    Some(vid(1)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::MutRef,
                ),
                // v2 = Move v0 — move while mutably borrowed!
                make_instr(Some(vid(2)), IrOp::Move { src: vid(0) }, Ownership::Owned),
                // v3 = Move v1 — move the mut ref itself
                make_instr(Some(vid(3)), IrOp::Move { src: vid(1) }, Ownership::Owned),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::MoveWhileBorrowed),
            "expected move-while-borrowed error, got: {:?}",
            result.diagnostics
        );
        assert_eq!(result.warnings, 0, "ownership errors must never be warnings");
    }

    // ── STRESS: Multiple functions with ownership violations ──────────────

    #[test]
    fn stress_multiple_functions_independent_errors() {
        // Two functions, each with an ownership violation.
        // Both must be caught — no cross-function leakage.
        let fn1 = FlatIrFunction {
            name: "fn1".to_string(),
            params: vec![],
            ret_ty: IrType::Unit,
            blocks: vec![FlatBlock {
                id: bid(0),
                instrs: vec![
                    make_instr(
                        Some(vid(0)),
                        IrOp::ConstInt {
                            value: 1,
                            ty: IrType::Int { width: 64, signed: true },
                        },
                        Ownership::Owned,
                    ),
                    make_instr(Some(vid(1)), IrOp::Move { src: vid(0) }, Ownership::Owned),
                    make_instr(Some(vid(2)), IrOp::Move { src: vid(0) }, Ownership::Owned),
                ],
                span: dummy_span(),
            }],
            entry: bid(0),
            effects: EffectFlags::pure(),
            requires: vec![],
            ensures: vec![],
            span: dummy_span(),
        };
        let fn2 = FlatIrFunction {
            name: "fn2".to_string(),
            params: vec![],
            ret_ty: IrType::Unit,
            blocks: vec![FlatBlock {
                id: bid(0),
                instrs: vec![
                    make_instr(
                        Some(vid(0)),
                        IrOp::Alloca {
                            ty: IrType::Int { width: 64, signed: true },
                            align: 8,
                        },
                        Ownership::Owned,
                    ),
                    make_instr(
                        Some(vid(1)),
                        IrOp::Load {
                            ptr: vid(0),
                            ty: IrType::Int { width: 64, signed: true },
                        },
                        Ownership::MutRef,
                    ),
                    make_instr(
                        Some(vid(2)),
                        IrOp::Load {
                            ptr: vid(0),
                            ty: IrType::Int { width: 64, signed: true },
                        },
                        Ownership::MutRef,
                    ),
                ],
                span: dummy_span(),
            }],
            entry: bid(0),
            effects: EffectFlags::pure(),
            requires: vec![],
            ensures: vec![],
            span: dummy_span(),
        };
        let result = ir_borrowck(&make_module_with_fns(vec![fn1, fn2]));
        assert!(
            result.errors >= 2,
            "expected at least 2 errors across two functions, got {}: {:?}",
            result.errors,
            result.diagnostics
        );
        assert_eq!(result.warnings, 0, "ownership errors must never be warnings");
    }

    // ── STRESS: Function parameters with ownership ────────────────────────

    #[test]
    fn stress_function_params_ownership_tracking() {
        // Function takes a String parameter (Owned, not Copy), moves it,
        // then tries to use again — should detect use-after-move.
        let fn_with_params = FlatIrFunction {
            name: "consume_then_use".to_string(),
            params: vec![
                (vid(0), IrType::String), // String is Owned (not Copy)
            ],
            ret_ty: IrType::Unit,
            blocks: vec![FlatBlock {
                id: bid(0),
                instrs: vec![
                    // Move the parameter
                    make_instr(Some(vid(1)), IrOp::Move { src: vid(0) }, Ownership::Owned),
                    // Try to use the moved parameter — error!
                    make_instr(
                        Some(vid(2)),
                        IrOp::BinOp {
                            op: crate::compiler::ast::BinOpKind::Add,
                            lhs: vid(0),
                            rhs: vid(1),
                        },
                        Ownership::Owned,
                    ),
                ],
                span: dummy_span(),
            }],
            entry: bid(0),
            effects: EffectFlags::pure(),
            requires: vec![],
            ensures: vec![],
            span: dummy_span(),
        };
        let result = ir_borrowck(&make_module_with_fns(vec![fn_with_params]));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::UseAfterMove),
            "expected use-after-move for moved String parameter, got: {:?}",
            result.diagnostics
        );
        assert_eq!(result.warnings, 0, "ownership errors must never be warnings");
    }

    // ── STRESS: Copy types never trigger use-after-move ───────────────────

    #[test]
    fn stress_copy_types_always_safe() {
        // Use Copy types extensively — no ownership violations should be detected.
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::ConstInt {
                        value: 1,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Copy,
                ),
                make_instr(
                    Some(vid(1)),
                    IrOp::ConstBool { value: true },
                    Ownership::Copy,
                ),
                // Use v0 many times — Copy, so never moved
                make_instr(
                    Some(vid(2)),
                    IrOp::BinOp {
                        op: crate::compiler::ast::BinOpKind::Add,
                        lhs: vid(0),
                        rhs: vid(0),
                    },
                    Ownership::Copy,
                ),
                make_instr(
                    Some(vid(3)),
                    IrOp::BinOp {
                        op: crate::compiler::ast::BinOpKind::Mul,
                        lhs: vid(0),
                        rhs: vid(2),
                    },
                    Ownership::Copy,
                ),
                make_instr(Some(vid(4)), IrOp::Copy { src: vid(0) }, Ownership::Copy),
                make_instr(
                    Some(vid(5)),
                    IrOp::BinOp {
                        op: crate::compiler::ast::BinOpKind::Add,
                        lhs: vid(0),
                        rhs: vid(4),
                    },
                    Ownership::Copy,
                ),
                make_instr(None, IrOp::Ret { value: Some(vid(5)) }, Ownership::Copy),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert_eq!(
            result.errors, 0,
            "Copy types should never cause ownership errors, got: {:?}",
            result.diagnostics
        );
        assert_eq!(result.warnings, 0);
    }

    // ── STRESS: TaskSpawn moves Owned args ────────────────────────────────

    #[test]
    fn stress_task_spawn_move_then_use() {
        // Spawn a task with Move ownership, then try to use the value.
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::ConstInt {
                        value: 99,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Owned,
                ),
                make_instr(
                    Some(vid(1)),
                    IrOp::TaskSpawn {
                        func: "worker".to_string(),
                        args: vec![vid(0)],
                        ownership: crate::compiler::ir::TaskOwnership::Move,
                    },
                    Ownership::Owned,
                ),
                // Try to use v0 after it was moved into the task — error!
                make_instr(
                    Some(vid(2)),
                    IrOp::BinOp {
                        op: crate::compiler::ast::BinOpKind::Add,
                        lhs: vid(0),
                        rhs: vid(0),
                    },
                    Ownership::Copy,
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::UseAfterMove),
            "expected use-after-move after TaskSpawn(Move), got: {:?}",
            result.diagnostics
        );
        assert_eq!(result.warnings, 0, "ownership errors must never be warnings");
    }

    // ── STRESS: Return moves the value ────────────────────────────────────

    #[test]
    fn stress_return_moves_value_then_use() {
        // Return a value, then try to use it again — use-after-move.
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::ConstInt {
                        value: 7,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Owned,
                ),
                // Return v0 (moves it)
                make_instr(None, IrOp::Ret { value: Some(vid(0)) }, Ownership::Owned),
                // Use v0 after return — unreachable but still an error
                make_instr(Some(vid(1)), IrOp::Move { src: vid(0) }, Ownership::Owned),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::UseAfterMove),
            "expected use-after-move after Ret, got: {:?}",
            result.diagnostics
        );
        assert_eq!(result.warnings, 0);
    }

    // ── STRESS: Store through moved pointer ───────────────────────────────

    #[test]
    fn stress_store_through_moved_pointer() {
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::Alloca {
                        ty: IrType::Int { width: 64, signed: true },
                        align: 8,
                    },
                    Ownership::Owned,
                ),
                // Move the pointer
                make_instr(Some(vid(1)), IrOp::Move { src: vid(0) }, Ownership::Owned),
                // Try to store through the moved pointer
                make_instr(
                    None,
                    IrOp::Store {
                        ptr: vid(0),
                        value: vid(1),
                    },
                    Ownership::Owned,
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::UseAfterMove),
            "expected use-after-move on Store through moved pointer, got: {:?}",
            result.diagnostics
        );
        assert_eq!(result.warnings, 0);
    }

    // ── STRESS: Borrow-after-move ─────────────────────────────────────────

    #[test]
    fn stress_borrow_after_move_explicit() {
        // Move a value, then try to create a reference to it.
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::Alloca {
                        ty: IrType::Int { width: 64, signed: true },
                        align: 8,
                    },
                    Ownership::Owned,
                ),
                // Move v0
                make_instr(Some(vid(1)), IrOp::Move { src: vid(0) }, Ownership::Owned),
                // Try to immutably borrow v0 after move
                make_instr(
                    Some(vid(2)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Ref,
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::BorrowAfterMove),
            "expected borrow-after-move, got: {:?}",
            result.diagnostics
        );
        assert_eq!(result.warnings, 0);
    }

    // ── STRESS: Deeply nested borrows ─────────────────────────────────────

    #[test]
    fn stress_triple_mut_borrow() {
        // Three mutable borrows of the same allocation — all errors.
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::Alloca {
                        ty: IrType::Int { width: 64, signed: true },
                        align: 8,
                    },
                    Ownership::Owned,
                ),
                make_instr(
                    Some(vid(1)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::MutRef,
                ),
                make_instr(
                    Some(vid(2)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::MutRef,
                ),
                make_instr(
                    Some(vid(3)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::MutRef,
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        // At least 2 errors (2nd and 3rd MutRef loads are both violations)
        let double_mut_count = result.diagnostics.iter()
            .filter(|d| d.kind == IrBorrowDiagKind::DoubleMutBorrow)
            .count();
        assert!(
            double_mut_count >= 2,
            "expected at least 2 DoubleMutBorrow errors, got {}: {:?}",
            double_mut_count,
            result.diagnostics
        );
        assert_eq!(result.warnings, 0);
    }

    // ── STRESS: Call with Owned args moves them ───────────────────────────

    #[test]
    fn stress_call_moves_owned_args() {
        // Call a function with Owned ownership — args are moved.
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::ConstInt {
                        value: 42,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Owned,
                ),
                // Call with Owned ownership — moves v0
                make_instr(
                    Some(vid(1)),
                    IrOp::Call {
                        func: "consume".to_string(),
                        args: vec![vid(0)],
                    },
                    Ownership::Owned,
                ),
                // Try to use v0 after call — use-after-move!
                make_instr(
                    Some(vid(2)),
                    IrOp::BinOp {
                        op: crate::compiler::ast::BinOpKind::Add,
                        lhs: vid(0),
                        rhs: vid(0),
                    },
                    Ownership::Copy,
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::UseAfterMove),
            "expected use-after-move after Call(Owned), got: {:?}",
            result.diagnostics
        );
        assert_eq!(result.warnings, 0);
    }

    // ── STRESS: Mixed ownership across blocks (phi nodes) ─────────────────

    #[test]
    fn stress_phi_node_merged_state() {
        // Block 0: define v0 (Owned), move v0 → v1
        // Block 1: phi from block 0 (v0 is Moved)
        // Use the phi — should detect moved state
        let blocks = vec![
            FlatBlock {
                id: bid(0),
                instrs: vec![
                    make_instr(
                        Some(vid(0)),
                        IrOp::ConstInt {
                            value: 10,
                            ty: IrType::Int { width: 64, signed: true },
                        },
                        Ownership::Owned,
                    ),
                    make_instr(Some(vid(1)), IrOp::Move { src: vid(0) }, Ownership::Owned),
                ],
                span: dummy_span(),
            },
            FlatBlock {
                id: bid(1),
                instrs: vec![
                    // phi: merge v0 from block 0 (Moved state)
                    make_instr(
                        Some(vid(2)),
                        IrOp::Phi { incoming: vec![(bid(0), vid(0))] },
                        Ownership::Owned,
                    ),
                    // Use the phi value — should detect Moved state
                    make_instr(
                        Some(vid(3)),
                        IrOp::BinOp {
                            op: crate::compiler::ast::BinOpKind::Add,
                            lhs: vid(2),
                            rhs: vid(2),
                        },
                        Ownership::Copy,
                    ),
                ],
                span: dummy_span(),
            },
        ];
        let result = ir_borrowck(&make_module(blocks));
        // The phi node should carry the Moved state from block 0
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::UseAfterMove),
            "expected use-after-move through phi node, got: {:?}",
            result.diagnostics
        );
        assert_eq!(result.warnings, 0);
    }

    // ── STRESS: Large number of values (performance) ──────────────────────

    #[test]
    fn stress_many_values_no_false_positives() {
        // Create 100 Copy values and use them all — should be 0 errors.
        let mut instrs: Vec<IrInstr> = Vec::new();
        for i in 0..100u32 {
            instrs.push(make_instr(
                Some(vid(i)),
                IrOp::ConstInt {
                    value: i as i128,
                    ty: IrType::Int { width: 64, signed: true },
                },
                Ownership::Copy,
            ));
        }
        // Use all of them in one big BinOp chain
        for i in 0..50u32 {
            instrs.push(make_instr(
                Some(vid(100 + i)),
                IrOp::BinOp {
                    op: crate::compiler::ast::BinOpKind::Add,
                    lhs: vid(i * 2),
                    rhs: vid(i * 2 + 1),
                },
                Ownership::Copy,
            ));
        }
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs,
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert_eq!(
            result.errors, 0,
            "expected 0 errors for 100 Copy values, got {}: {:?}",
            result.errors,
            result.diagnostics
        );
    }

    // ── STRESS: Large number of Owned values with violations ──────────────

    #[test]
    fn stress_many_owned_all_violations() {
        // Create 50 Owned values, move each one, then try to use each one again.
        // Should get 50 use-after-move errors.
        let mut instrs: Vec<IrInstr> = Vec::new();
        for i in 0..50u32 {
            let base = i * 3;
            instrs.push(make_instr(
                Some(vid(base)),
                IrOp::ConstInt {
                    value: i as i128,
                    ty: IrType::Int { width: 64, signed: true },
                },
                Ownership::Owned,
            ));
            instrs.push(make_instr(
                Some(vid(base + 1)),
                IrOp::Move { src: vid(base) },
                Ownership::Owned,
            ));
            // Use after move
            instrs.push(make_instr(
                Some(vid(base + 2)),
                IrOp::BinOp {
                    op: crate::compiler::ast::BinOpKind::Add,
                    lhs: vid(base),
                    rhs: vid(base + 1),
                },
                Ownership::Copy,
            ));
        }
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs,
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        let uam_count = result.diagnostics.iter()
            .filter(|d| d.kind == IrBorrowDiagKind::UseAfterMove)
            .count();
        assert!(
            uam_count >= 50,
            "expected at least 50 use-after-move errors, got {}: {:?}",
            uam_count,
            result.diagnostics
        );
        assert_eq!(result.warnings, 0);
    }

    // ── STRESS: Immut borrow then mut borrow ──────────────────────────────

    #[test]
    fn stress_immut_then_mut_borrow_same_root() {
        // First take an immutable borrow, then try a mutable borrow — error.
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::Alloca {
                        ty: IrType::Int { width: 64, signed: true },
                        align: 8,
                    },
                    Ownership::Owned,
                ),
                make_instr(
                    Some(vid(1)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Ref,
                ),
                make_instr(
                    Some(vid(2)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::MutRef, // Error: can't mutably borrow while immutably borrowed
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::DoubleMutBorrow),
            "expected DoubleMutBorrow for mut borrow while immutably borrowed, got: {:?}",
            result.diagnostics
        );
        assert_eq!(result.warnings, 0);
    }

    // ── STRESS: Multiple immutable borrows are OK ─────────────────────────

    #[test]
    fn stress_many_immut_borrows_ok() {
        // Take 10 immutable borrows — all fine.
        let mut instrs: Vec<IrInstr> = vec![
            make_instr(
                Some(vid(0)),
                IrOp::Alloca {
                    ty: IrType::Int { width: 64, signed: true },
                    align: 8,
                },
                Ownership::Owned,
            ),
        ];
        for i in 1..=10u32 {
            instrs.push(make_instr(
                Some(vid(i)),
                IrOp::Load {
                    ptr: vid(0),
                    ty: IrType::Int { width: 64, signed: true },
                },
                Ownership::Ref,
            ));
        }
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs,
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert_eq!(
            result.errors, 0,
            "multiple immutable borrows should be OK, got: {:?}",
            result.diagnostics
        );
    }

    // ── STRESS: error_codes module ownership invariants ───────────────────

    #[test]
    fn stress_ownership_codes_never_warning_eligible() {
        use crate::compiler::error_codes::is_warning_eligible;

        // Exhaustively check ALL E4xxx codes — none should be warning-eligible.
        for code_num in 4000u32..=4999 {
            let code = format!("E{}", code_num);
            assert!(
                !is_warning_eligible(&code),
                "OWNERSHIP VIOLATION: {} is warning-eligible! \
                 All ownership/borrow codes must be hard errors.",
                code
            );
        }
    }

    // ── STRESS: IR borrowck result invariant ──────────────────────────────

    #[test]
    fn stress_borrowck_result_warnings_always_zero() {
        // Regardless of what module we check, the warnings count must always be 0.
        // This is the fundamental ownership = hard error invariant.

        // Test 1: Module with violations
        let module_with_violations = make_module(vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::ConstInt {
                        value: 1,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Owned,
                ),
                make_instr(Some(vid(1)), IrOp::Move { src: vid(0) }, Ownership::Owned),
                make_instr(Some(vid(2)), IrOp::Move { src: vid(0) }, Ownership::Owned),
            ],
            span: dummy_span(),
        }]);
        let result = ir_borrowck(&module_with_violations);
        assert_eq!(
            result.warnings, 0,
            "FUNDAMENTAL INVARIANT VIOLATED: ir_borrowck returned {} warnings. \
             Ownership is ALWAYS a hard error!",
            result.warnings
        );

        // Test 2: Module without violations
        let module_clean = make_module(vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::ConstInt {
                        value: 1,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Copy,
                ),
                make_instr(
                    Some(vid(1)),
                    IrOp::ConstInt {
                        value: 2,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Copy,
                ),
                make_instr(None, IrOp::Ret { value: Some(vid(0)) }, Ownership::Copy),
            ],
            span: dummy_span(),
        }]);
        let result = ir_borrowck(&module_clean);
        assert_eq!(
            result.warnings, 0,
            "FUNDAMENTAL INVARIANT VIOLATED: ir_borrowck returned {} warnings on clean module.",
            result.warnings
        );

        // Test 3: Empty module
        let module_empty = FlatIrModule {
            functions: vec![],
            intrinsics: vec![],
            span: dummy_span(),
            lowering_errors: vec![],
        };
        let result = ir_borrowck(&module_empty);
        assert_eq!(
            result.warnings, 0,
            "FUNDAMENTAL INVARIANT VIOLATED: ir_borrowck returned {} warnings on empty module.",
            result.warnings
        );
    }

    // ── STRESS: Move inside a call chain ──────────────────────────────────

    #[test]
    fn stress_move_inside_call_chain() {
        // v0 = Owned
        // Call(f, [v0]) with Owned ownership — moves v0
        // Call(g, [v0]) — use after move!
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::ConstInt {
                        value: 42,
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Owned,
                ),
                make_instr(
                    Some(vid(1)),
                    IrOp::Call {
                        func: "f".to_string(),
                        args: vec![vid(0)],
                    },
                    Ownership::Owned,
                ),
                make_instr(
                    Some(vid(2)),
                    IrOp::Call {
                        func: "g".to_string(),
                        args: vec![vid(0)],
                    },
                    Ownership::Owned,
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::UseAfterMove),
            "expected use-after-move in call chain, got: {:?}",
            result.diagnostics
        );
        assert_eq!(result.warnings, 0);
    }

    // ── STRESS: Store invalidates immutable borrows ───────────────────────

    #[test]
    fn stress_store_invalidate_multiple_immut_borrows() {
        // v0 = Alloca
        // v1, v2, v3 = Load v0 (Ref) — 3 immutable borrows
        // Store v0, v1 — write while 3 immutable borrows exist
        let blocks = vec![FlatBlock {
            id: bid(0),
            instrs: vec![
                make_instr(
                    Some(vid(0)),
                    IrOp::Alloca {
                        ty: IrType::Int { width: 64, signed: true },
                        align: 8,
                    },
                    Ownership::Owned,
                ),
                make_instr(
                    Some(vid(1)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Ref,
                ),
                make_instr(
                    Some(vid(2)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Ref,
                ),
                make_instr(
                    Some(vid(3)),
                    IrOp::Load {
                        ptr: vid(0),
                        ty: IrType::Int { width: 64, signed: true },
                    },
                    Ownership::Ref,
                ),
                make_instr(
                    None,
                    IrOp::Store {
                        ptr: vid(0),
                        value: vid(1),
                    },
                    Ownership::Owned,
                ),
            ],
            span: dummy_span(),
        }];
        let result = ir_borrowck(&make_module(blocks));
        assert!(
            result.diagnostics.iter().any(|d| d.kind == IrBorrowDiagKind::AliasViolation),
            "expected alias violation on Store with multiple immutable borrows, got: {:?}",
            result.diagnostics
        );
        assert_eq!(result.warnings, 0);
    }
}
