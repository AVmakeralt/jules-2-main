// =============================================================================
// jules/src/compiler/ir_to_bytecode.rs
//
// IR-TO-BYTECODE COMPILER
//
// Converts FlatIrModule (flat SSA instruction IR) into BytecodeVM instructions.
// This replaces the AST-to-bytecode compilation path, making the IR the single
// source of truth for the VM.
//
// Architecture:
//   FlatIrModule → IrBytecodeResult (Vec<BytecodeFunction>)
//   BytecodeVM::load_functions(result.functions) → execution
//
// Key design decisions:
//   • SSA ValueIds are mapped 1:1 to u16 local variable slots via a HashMap.
//   • Block IDs are resolved to instruction offsets in a patching pass.
//   • Phi nodes are lowered to Move instructions at predecessor block ends.
//   • The VM's Call instruction accepts Value::Str(fn_name) for named calls.
// =============================================================================

use std::collections::HashMap;

use crate::compiler::ast::{BinOpKind, UnOpKind};
use crate::compiler::ir::{FlatIrFunction, FlatIrModule, IrOp, ValueId, BlockId};
use crate::runtime::bytecode_vm::{BytecodeFunction, Instr};
use crate::runtime::interp::Value;

// =============================================================================
// §1  RESULT TYPE
// =============================================================================

/// Result of compiling a FlatIrModule to bytecode.
///
/// Contains all compiled BytecodeFunctions (ready for `BytecodeVM::load_functions`)
/// and any non-fatal errors encountered during compilation.
#[derive(Debug)]
pub struct IrBytecodeResult {
    /// Compiled bytecode functions, in the same order as the source IR functions.
    pub functions: Vec<BytecodeFunction>,
    /// Non-fatal errors (e.g. unsupported IR ops). The function is still emitted
    /// with a Nop placeholder for the unsupported instruction.
    pub errors: Vec<String>,
}

// =============================================================================
// §2  INTERNAL TYPES
// =============================================================================

/// Placeholder for a jump whose target offset is not yet known.
struct PendingJump {
    /// Instruction index of the Jump/JumpFalse/JumpTrue in the bytecode stream.
    instr_idx: usize,
    /// The target BlockId that needs to be resolved.
    target_block: BlockId,
}

/// Phi-node information collected during the first walk.
#[derive(Clone)]
struct PhiInfo {
    /// The destination slot for the phi result.
    dst_slot: u16,
    /// Incoming values: (predecessor BlockId, ValueId of the incoming value).
    incoming: Vec<(BlockId, ValueId)>,
}

// =============================================================================
// §3  PER-FUNCTION COMPILER
// =============================================================================

struct FunctionCompiler {
    /// Maps SSA ValueId → u16 local slot index.
    value_map: HashMap<u32, u16>,
    /// Next available local slot.
    next_slot: u16,
    /// Number of parameters.
    num_params: u16,
    /// The bytecode function being built.
    output: BytecodeFunction,
    /// Maps BlockId → instruction offset in the bytecode stream.
    block_offsets: HashMap<u32, usize>,
    /// Jumps that need their offsets patched once all blocks are emitted.
    pending_jumps: Vec<PendingJump>,
    /// Phi nodes collected from all blocks, keyed by the block they belong to.
    /// The Vec<PhiInfo> for a block contains all phis in that block.
    phis_by_block: HashMap<u32, Vec<PhiInfo>>,
    /// Non-fatal errors.
    errors: Vec<String>,
    /// Function name → index mapping (from the module level).
    func_name_to_idx: HashMap<String, usize>,
}

impl FunctionCompiler {
    fn new(ir_func: &FlatIrFunction, func_name_to_idx: &HashMap<String, usize>) -> Self {
        let mut fc = FunctionCompiler {
            value_map: HashMap::new(),
            next_slot: 0,
            num_params: ir_func.params.len() as u16,
            output: BytecodeFunction::new(ir_func.name.clone()),
            block_offsets: HashMap::new(),
            pending_jumps: Vec::new(),
            phis_by_block: HashMap::new(),
            errors: Vec::new(),
            func_name_to_idx: func_name_to_idx.clone(),
        };

        // Assign slots for parameters (slot 0, 1, ..., N-1).
        for (vid, _ty) in &ir_func.params {
            fc.assign_slot(*vid);
        }

        fc.output.num_params = ir_func.params.len() as u16;
        fc
    }

    /// Get or create a slot for a ValueId.
    fn slot_for(&mut self, vid: ValueId) -> u16 {
        if let Some(&slot) = self.value_map.get(&vid.0) {
            slot
        } else {
            self.assign_slot(vid)
        }
    }

    /// Assign a new slot for a ValueId. Panics if we've exhausted u16 space.
    fn assign_slot(&mut self, vid: ValueId) -> u16 {
        let slot = self.next_slot;
        self.next_slot = self.next_slot.saturating_add(1);
        if self.next_slot == 0 {
            panic!("ir_to_bytecode: slot overflow — more than 65535 values in function");
        }
        self.value_map.insert(vid.0, slot);
        slot
    }

    /// Allocate a fresh temporary slot (not tied to a ValueId).
    fn alloc_temp(&mut self) -> u16 {
        let slot = self.next_slot;
        self.next_slot = self.next_slot.saturating_add(1);
        if self.next_slot == 0 {
            panic!("ir_to_bytecode: slot overflow — more than 65535 values in function");
        }
        slot
    }

    /// Emit an instruction and return its index.
    fn emit(&mut self, instr: Instr) -> usize {
        let idx = self.output.instructions.len();
        self.output.instructions.push(instr);
        idx
    }

    // ─── Main compilation entry point ──────────────────────────────────────

    fn compile(mut self, ir_func: &FlatIrFunction) -> (BytecodeFunction, Vec<String>) {
        // ── Phase 1: Collect phi nodes and assign slots for all values ──
        self.collect_phis_and_slots(ir_func);

        // ── Phase 2: Emit bytecode for all blocks ──
        self.emit_blocks(ir_func);

        // ── Phase 3: Patch all pending jumps ──
        self.patch_jumps();

        // Finalize
        self.output.num_locals = self.next_slot;
        (self.output, self.errors)
    }

    /// Walk all blocks and instructions to:
    ///   1. Assign slots for every ValueId that appears as a destination.
    ///   2. Collect phi nodes for later lowering.
    fn collect_phis_and_slots(&mut self, ir_func: &FlatIrFunction) {
        for block in &ir_func.blocks {
            let mut block_phis = Vec::new();
            for instr in &block.instrs {
                // Assign a slot for the destination ValueId.
                if let Some(dst) = instr.dst {
                    self.slot_for(dst);
                }

                // Collect phi nodes.
                if let IrOp::Phi { incoming } = &instr.op {
                    let dst_slot = self.slot_for(instr.dst.unwrap());
                    block_phis.push(PhiInfo {
                        dst_slot,
                        incoming: incoming.clone(),
                    });
                }

                // For Call, we need temp slots for args.
                if let IrOp::Call { args, .. } = &instr.op {
                    for arg in args {
                        self.slot_for(*arg);
                    }
                }
            }
            if !block_phis.is_empty() {
                self.phis_by_block.insert(block.id.0, block_phis);
            }
        }
    }

    /// Emit bytecode for all blocks in the function.
    fn emit_blocks(&mut self, ir_func: &FlatIrFunction) {
        for block in &ir_func.blocks {
            // Record the start offset of this block.
            self.block_offsets.insert(block.id.0, self.output.instructions.len());

            // NOTE: We do NOT emit phi-resolution Moves at the top of the block.
            // Phi resolution is handled entirely by predecessor-inserted Move
            // instructions (see emit_phi_moves_for_predecessor). Emitting
            // defensive Moves here would OVERWRITE the correct values set by
            // predecessor blocks at runtime, because these defensive Moves
            // execute after the jump transfers control to this block. This was
            // the cause of the "unit type leak" bug where loop header phis
            // always received the initial (pre-loop) value instead of the
            // back-edge (accumulated) value.

            // Emit instructions for this block.
            for instr in &block.instrs {
                match &instr.op {
                    IrOp::Nop => {
                        self.emit(Instr::Nop);
                    }

                    // ── Constants ──────────────────────────────────────────
                    IrOp::ConstInt { value, .. } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        self.emit(Instr::LoadConstInt { dst, value: *value as i64 });
                    }

                    IrOp::ConstFloat { bits, .. } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        self.emit(Instr::LoadConstFloat { dst, value: f64::from_bits(*bits) });
                    }

                    IrOp::ConstBool { value } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        self.emit(Instr::LoadConstBool { dst, value: *value });
                    }

                    IrOp::ConstStr { idx } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        // Use LoadConst with constant pool index.
                        // We store the string index as the constant pool entry.
                        let const_idx = *idx;
                        self.emit(Instr::LoadConst { dst, idx: const_idx });
                    }

                    IrOp::ConstUnit => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        self.emit(Instr::LoadConstUnit { dst });
                    }

                    // ── Binary operations ──────────────────────────────────
                    IrOp::BinOp { op, lhs, rhs } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        let lhs_slot = self.slot_for(*lhs);
                        let rhs_slot = self.slot_for(*rhs);
                        let instr = match op {
                            BinOpKind::Add => Instr::Add { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::Sub => Instr::Sub { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::Mul => Instr::Mul { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::Div => Instr::Div { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::Rem => Instr::Rem { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::FloorDiv => Instr::Div { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::Eq  => Instr::Eq  { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::Ne  => Instr::Ne  { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::Lt  => Instr::Lt  { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::Le  => Instr::Le  { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::Gt  => Instr::Gt  { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::Ge  => Instr::Ge  { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::And => Instr::BitAnd { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::Or  => Instr::BitOr  { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::BitAnd => Instr::BitAnd { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::BitOr  => Instr::BitOr  { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::BitXor => Instr::BitXor { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::Shl => Instr::Shl { dst, lhs: lhs_slot, rhs: rhs_slot },
                            BinOpKind::Shr => Instr::Shr { dst, lhs: lhs_slot, rhs: rhs_slot },
                        };
                        self.emit(instr);
                    }

                    // ── Unary operations ───────────────────────────────────
                    IrOp::UnOp { op, operand } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        let src = self.slot_for(*operand);
                        let bc_instr = match op {
                            UnOpKind::Neg => Instr::Neg { dst, src },
                            UnOpKind::Not => Instr::Not { dst, src },
                            UnOpKind::Deref | UnOpKind::Ref | UnOpKind::RefMut => {
                                // Deref/Ref are identity at the bytecode level
                                // (the VM doesn't enforce borrow checking).
                                Instr::Move { dst, src }
                            }
                        };
                        self.emit(bc_instr);
                    }

                    // ── Move ───────────────────────────────────────────────
                    IrOp::Move { src } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        let src_slot = self.slot_for(*src);
                        self.emit(Instr::Move { dst, src: src_slot });
                    }

                    // ── Copy ───────────────────────────────────────────────
                    IrOp::Copy { src } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        let src_slot = self.slot_for(*src);
                        self.emit(Instr::Move { dst, src: src_slot });
                    }

                    // ── Return ─────────────────────────────────────────────
                    IrOp::Ret { value } => {
                        match value {
                            Some(vid) => {
                                let ret_slot = self.slot_for(*vid);
                                // Move return value to slot 0 (VM convention).
                                if ret_slot != 0 {
                                    self.emit(Instr::Move { dst: 0, src: ret_slot });
                                }
                                self.emit(Instr::Return { value: 0 });
                            }
                            None => {
                                self.emit(Instr::LoadConstUnit { dst: 0 });
                                self.emit(Instr::Return { value: 0 });
                            }
                        }
                    }

                    // ── Conditional branch ─────────────────────────────────
                    IrOp::CondBr { cond, if_true, if_false } => {
                        let cond_slot = self.slot_for(*cond);

                        // Phi resolution for the FALSE branch: emit moves from
                        // the current block to the if_false target BEFORE the
                        // JumpFalse instruction. At runtime, when the condition
                        // is false, the JumpFalse transfers control to if_false,
                        // and these moves must have already executed.
                        self.emit_phi_moves_for_predecessor(block.id, *if_false);

                        // JumpFalse to the false block; fall through to the true block.
                        let jf_pos = self.emit(Instr::JumpFalse {
                            cond: cond_slot,
                            offset: 0, // placeholder
                        });
                        self.pending_jumps.push(PendingJump {
                            instr_idx: jf_pos,
                            target_block: *if_false,
                        });

                        // Phi resolution for the TRUE branch: emit moves from
                        // the current block to the if_true target. These only
                        // execute when the condition is true (fall-through path),
                        // before the unconditional jump to if_true.
                        self.emit_phi_moves_for_predecessor(block.id, *if_true);

                        // Jump to the true block.
                        let j_pos = self.emit(Instr::Jump { offset: 0 });
                        self.pending_jumps.push(PendingJump {
                            instr_idx: j_pos,
                            target_block: *if_true,
                        });
                    }

                    // ── Unconditional jump ─────────────────────────────────
                    IrOp::Jump { target } => {
                        // Before emitting the jump, insert phi-resolution Moves
                        // for the target block's phi nodes. The incoming values
                        // come from the current block's definitions.
                        self.emit_phi_moves_for_predecessor(block.id, *target);

                        let j_pos = self.emit(Instr::Jump { offset: 0 });
                        self.pending_jumps.push(PendingJump {
                            instr_idx: j_pos,
                            target_block: *target,
                        });
                    }

                    // ── Phi node ───────────────────────────────────────────
                    // Phi nodes are handled via the predecessor-inserted Move
                    // instructions. At the point where a phi appears in the
                    // block, its destination slot already has the correct value
                    // (set by Moves inserted at predecessor terminators).
                    // We do not emit any bytecode here — the Moves were emitted
                    // before the Jump/CondBr in predecessor blocks.
                    IrOp::Phi { .. } => {
                        // No bytecode emitted; phi resolution is handled by
                        // predecessor-inserted Moves.
                    }

                    // ── Function call ──────────────────────────────────────
                    IrOp::Call { func, args } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        let argc = args.len() as u16;

                        // Allocate contiguous slots for arguments starting at arg_start.
                        let arg_start = self.alloc_temp();
                        for _ in 1..argc {
                            self.alloc_temp();
                        }

                        // Copy each argument value into its slot.
                        for (i, arg_vid) in args.iter().enumerate() {
                            let arg_slot = self.slot_for(*arg_vid);
                            let target = arg_start + i as u16;
                            if arg_slot != target {
                                self.emit(Instr::Move { dst: target, src: arg_slot });
                            }
                        }

                        // Store the function name as a string constant so the VM
                        // can look up the callee by name (matches the BytecodeCompiler
                        // convention of using Value::Str(fn_name) for named calls).
                        let func_slot = self.alloc_temp();
                        let const_idx = self.output.add_constant(Value::Str(func.clone()));
                        self.emit(Instr::LoadConst { dst: func_slot, idx: const_idx });

                        self.emit(Instr::Call {
                            dst,
                            func: func_slot,
                            argc,
                            start: arg_start,
                        });
                    }

                    // ── Intrinsic call ─────────────────────────────────────
                    IrOp::Intrinsic { name, args } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        let argc = args.len() as u16;

                        let arg_start = self.alloc_temp();
                        for _ in 1..argc {
                            self.alloc_temp();
                        }

                        for (i, arg_vid) in args.iter().enumerate() {
                            let arg_slot = self.slot_for(*arg_vid);
                            let target = arg_start + i as u16;
                            if arg_slot != target {
                                self.emit(Instr::Move { dst: target, src: arg_slot });
                            }
                        }

                        // Store the intrinsic name as a string constant.
                        let func_slot = self.alloc_temp();
                        let const_idx = self.output.add_constant(Value::Str(format!("__intrinsic_{name}")));
                        self.emit(Instr::LoadConst { dst: func_slot, idx: const_idx });

                        self.emit(Instr::Call {
                            dst,
                            func: func_slot,
                            argc,
                            start: arg_start,
                        });

                        self.errors.push(format!(
                            "intrinsic `{name}` emitted as named call — may fail at runtime"
                        ));
                    }

                    // ── Memory: Alloca ─────────────────────────────────────
                    IrOp::Alloca { .. } => {
                        // Alloca just reserves a slot (already done in
                        // collect_phis_and_slots via the dst ValueId).
                        // Emit a LoadConstUnit to initialize the slot.
                        let dst = self.slot_for(instr.dst.unwrap());
                        self.emit(Instr::LoadConstUnit { dst });
                    }

                    // ── Memory: Store ──────────────────────────────────────
                    IrOp::Store { ptr, value } => {
                        let ptr_slot = self.slot_for(*ptr);
                        let val_slot = self.slot_for(*value);
                        // At the bytecode level, Store(ptr, value) is a Move
                        // into the slot that ptr points to. Since the VM uses
                        // a flat slot array (not pointer dereferencing), we
                        // treat ptr as a direct slot index.
                        self.emit(Instr::Move { dst: ptr_slot, src: val_slot });
                    }

                    // ── Memory: Load ───────────────────────────────────────
                    IrOp::Load { ptr, .. } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        let src_slot = self.slot_for(*ptr);
                        // Load from a mutable location: at the bytecode level,
                        // this is a Move from the source slot.
                        self.emit(Instr::Move { dst, src: src_slot });
                    }

                    // ── Type operations ────────────────────────────────────
                    IrOp::TypeCheck { value, .. } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        let src = self.slot_for(*value);
                        // Emit a type check — for now, assume the check passes
                        // and just copy the value.
                        self.emit(Instr::Move { dst, src });
                        self.errors.push(
                            "IrOp::TypeCheck emitted as Move — runtime type check not implemented".to_string()
                        );
                    }

                    IrOp::Cast { src, .. } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        let src_slot = self.slot_for(*src);
                        // Cast at the bytecode level is a Move; the VM's
                        // register-based representation is untyped (all Values).
                        self.emit(Instr::Move { dst, src: src_slot });
                    }

                    // ── Tensor operations ──────────────────────────────────
                    IrOp::MatMul { lhs, rhs } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        let lhs_slot = self.slot_for(*lhs);
                        let rhs_slot = self.slot_for(*rhs);
                        self.emit(Instr::MatMul { dst, lhs: lhs_slot, rhs: rhs_slot });
                    }

                    IrOp::HadamardMul { lhs, rhs } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        let lhs_slot = self.slot_for(*lhs);
                        let rhs_slot = self.slot_for(*rhs);
                        self.emit(Instr::HadamardMul { dst, lhs: lhs_slot, rhs: rhs_slot });
                    }

                    IrOp::HadamardDiv { lhs, rhs } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        let lhs_slot = self.slot_for(*lhs);
                        let rhs_slot = self.slot_for(*rhs);
                        // No HadamardDiv in the VM; lower as Div.
                        self.emit(Instr::Div { dst, lhs: lhs_slot, rhs: rhs_slot });
                        self.errors.push(
                            "IrOp::HadamardDiv lowered as Div — element-wise semantics lost".to_string()
                        );
                    }

                    IrOp::TensorConcat { lhs, rhs } => {
                        // No TensorConcat in the VM; emit as Nop with error.
                        self.emit(Instr::Nop);
                        self.errors.push(
                            "IrOp::TensorConcat not supported — emitted Nop".to_string()
                        );
                    }

                    IrOp::KronProd { lhs, rhs } => {
                        self.emit(Instr::Nop);
                        self.errors.push(
                            "IrOp::KronProd not supported — emitted Nop".to_string()
                        );
                    }

                    IrOp::OuterProd { lhs, rhs } => {
                        self.emit(Instr::Nop);
                        self.errors.push(
                            "IrOp::OuterProd not supported — emitted Nop".to_string()
                        );
                    }

                    // ── Parallelism ────────────────────────────────────────
                    IrOp::ParallelStart { region_id } => {
                        self.emit(Instr::Nop);
                        self.errors.push(format!(
                            "IrOp::ParallelStart({region_id}) not supported — emitted Nop"
                        ));
                    }

                    IrOp::ParallelEnd { region_id } => {
                        self.emit(Instr::Nop);
                        self.errors.push(format!(
                            "IrOp::ParallelEnd({region_id}) not supported — emitted Nop"
                        ));
                    }

                    // ── Region ──────────────────────────────────────────────
                    IrOp::RegionAlloc { .. } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        self.emit(Instr::LoadConstUnit { dst });
                        self.errors.push(
                            "IrOp::RegionAlloc lowered to LoadConstUnit — no region support".to_string()
                        );
                    }

                    // ── Tasks ──────────────────────────────────────────────
                    IrOp::TaskSpawn { func, args, .. } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        // Lower as a regular function call for now.
                        let argc = args.len() as u16;
                        let arg_start = self.alloc_temp();
                        for _ in 1..argc {
                            self.alloc_temp();
                        }
                        for (i, arg_vid) in args.iter().enumerate() {
                            let arg_slot = self.slot_for(*arg_vid);
                            let target = arg_start + i as u16;
                            if arg_slot != target {
                                self.emit(Instr::Move { dst: target, src: arg_slot });
                            }
                        }
                        let func_slot = self.alloc_temp();
                        let const_idx = self.output.add_constant(Value::Str(func.clone()));
                        self.emit(Instr::LoadConst { dst: func_slot, idx: const_idx });
                        self.emit(Instr::Call { dst, func: func_slot, argc, start: arg_start });
                        self.errors.push(format!(
                            "IrOp::TaskSpawn({func}) lowered to synchronous Call"
                        ));
                    }

                    IrOp::TaskJoin { task } => {
                        let dst = self.slot_for(instr.dst.unwrap());
                        let src = self.slot_for(*task);
                        self.emit(Instr::Move { dst, src });
                        self.errors.push(
                            "IrOp::TaskJoin lowered to Move — no async support".to_string()
                        );
                    }

                    // ── Effect emission ────────────────────────────────────
                    IrOp::Emit { value, effect } => {
                        // Emit the value to stdout (Print) and note the effect.
                        let src = self.slot_for(*value);
                        self.emit(Instr::Print { src });
                        let _ = effect; // effect name is metadata only at this level
                    }
                }
            }
        }
    }

    /// Emit Move instructions to resolve phi nodes for `target_block` before
    /// a Jump/CondBr transfers control there. This is called from the
    /// *predecessor* block, just before the jump instruction is emitted.
    ///
    /// For each phi in the target block, we copy the incoming value from this
    /// predecessor into the phi's destination slot.
    /// Emit Move instructions to resolve phi nodes for `target_block` before
    /// a Jump/CondBr transfers control there from `source_block`.
    ///
    /// For each phi in the target block, we copy ONLY the incoming value from
    /// this specific predecessor into the phi's destination slot.
    ///
    /// IMPORTANT: We must only emit the Move for the current predecessor,
    /// NOT for all incoming values. If we emit moves for all predecessors,
    /// the last move always wins, making the phi always take the else-branch
    /// value regardless of which branch was actually taken.
    fn emit_phi_moves_for_predecessor(&mut self, source_block: BlockId, target_block: BlockId) {
        // Clone the phi data to avoid borrowing self.phis_by_block while
        // we mutably borrow self for slot_for() and emit().
        let phis = match self.phis_by_block.get(&target_block.0) {
            Some(p) => p.clone(),
            None => return,
        };
        for phi in &phis {
            // Only emit the Move for the incoming value from THIS predecessor.
            // This is critical: each predecessor edge must only write its own
            // incoming value to the phi destination slot. Writing all incoming
            // values would cause the last-written value to always win,
            // regardless of which branch was taken at runtime.
            for (pred_block, incoming_vid) in &phi.incoming {
                if *pred_block == source_block {
                    let src_slot = self.slot_for(*incoming_vid);
                    self.emit(Instr::Move { dst: phi.dst_slot, src: src_slot });
                    break; // Only one incoming value per predecessor
                }
            }
        }
    }

    /// Patch all pending jump instructions with their resolved target offsets.
    fn patch_jumps(&mut self) {
        // Take ownership of pending_jumps to avoid borrowing self while mutating.
        let pending_jumps = std::mem::take(&mut self.pending_jumps);
        for pending in &pending_jumps {
            let target_offset = match self.block_offsets.get(&pending.target_block.0) {
                Some(&offset) => offset as i32,
                None => {
                    self.errors.push(format!(
                        "ir_to_bytecode: block bb{} not found — jump at instr {} targets unknown block",
                        pending.target_block.0, pending.instr_idx
                    ));
                    continue;
                }
            };
            let current_offset = pending.instr_idx as i32;
            let relative_offset = target_offset - current_offset;
            match &mut self.output.instructions[pending.instr_idx] {
                Instr::Jump { offset } => *offset = relative_offset,
                Instr::JumpFalse { offset, .. } => *offset = relative_offset,
                Instr::JumpTrue { offset, .. } => *offset = relative_offset,
                other => {
                    self.errors.push(format!(
                        "ir_to_bytecode: expected jump instruction at idx {}, found {:?}",
                        pending.instr_idx, other
                    ));
                }
            }
        }
    }
}

// =============================================================================
// §4  MODULE-LEVEL COMPILATION
// =============================================================================

/// Compile a FlatIrModule into bytecode that the BytecodeVM can execute.
///
/// # Example
///
/// ```rust,ignore
/// let ir_module = lower_program(&ast);
/// let result = compile_ir_module(&ir_module);
/// if !result.errors.is_empty() {
///     for err in &result.errors {
///         eprintln!("warning: {err}");
///     }
/// }
/// let mut vm = BytecodeVM::new();
/// vm.load_functions(result.functions);
/// vm.call_fn("main", vec![]).unwrap();
/// ```
pub fn compile_ir_module(module: &FlatIrModule) -> IrBytecodeResult {
    let mut errors = Vec::new();

    // Build function name → index mapping (for cross-function calls).
    let func_name_to_idx: HashMap<String, usize> = module
        .functions
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name.clone(), i))
        .collect();

    // Compile each function.
    let mut functions = Vec::with_capacity(module.functions.len());
    for ir_func in &module.functions {
        let fc = FunctionCompiler::new(ir_func, &func_name_to_idx);
        let (bytecode_func, fc_errors) = fc.compile(ir_func);
        errors.extend(fc_errors);
        functions.push(bytecode_func);
    }

    IrBytecodeResult { functions, errors }
}

// =============================================================================
// §5  VM INTEGRATION HELPER
// =============================================================================

impl IrBytecodeResult {
    /// Returns true if there were any errors during compilation.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Returns true if there were any fatal errors (errors that would prevent
    /// correct execution). Fatal errors include unknown block references,
    /// undefined values, and other structural problems. Soft errors (unsupported
    /// ops lowered to Nop/Move) are not considered fatal since the program
    /// can still execute with degraded semantics.
    pub fn has_fatal_errors(&self) -> bool {
        self.errors.iter().any(|e| {
            e.contains("block not found")
                || e.contains("undefined value")
                || e.contains("function not found")
                || e.contains("parameter index out of range")
        })
    }
}

// =============================================================================
// §6  TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::{FlatBlock, FlatIrFunction, IrInstr, IrOp, IrType, ValueId, BlockId, EffectFlags, Ownership, CostHint, AliasKind};
    use crate::compiler::lexer::Span;

    fn dummy_span() -> Span {
        Span::dummy()
    }

    fn make_instr(dst: Option<ValueId>, op: IrOp) -> IrInstr {
        IrInstr {
            dst,
            op,
            span: dummy_span(),
            effects: EffectFlags::pure(),
            ownership: Ownership::Copy,
            cost: CostHint::Unknown,
            alias: AliasKind::default(),
        }
    }

    #[test]
    fn test_compile_simple_function() {
        // fn add(a: i64, b: i64) -> i64 {
        //     v2 = a + b
        //     return v2
        // }
        let v0 = ValueId(0);
        let v1 = ValueId(1);
        let v2 = ValueId(2);

        let entry_block = FlatBlock {
            id: BlockId(0),
            instrs: vec![
                make_instr(Some(v2), IrOp::BinOp {
                    op: BinOpKind::Add,
                    lhs: v0,
                    rhs: v1,
                }),
                make_instr(None, IrOp::Ret { value: Some(v2) }),
            ],
            span: dummy_span(),
        };

        let ir_func = FlatIrFunction {
            name: "add".to_string(),
            params: vec![(v0, IrType::Int { width: 64, signed: true }), (v1, IrType::Int { width: 64, signed: true })],
            ret_ty: IrType::Int { width: 64, signed: true },
            blocks: vec![entry_block],
            entry: BlockId(0),
            effects: EffectFlags::pure(),
            requires: vec![],
            ensures: vec![],
            span: dummy_span(),
        };

        let module = FlatIrModule {
            functions: vec![ir_func],
            intrinsics: vec![],
            span: dummy_span(),
            lowering_errors: vec![],
        };

        let result = compile_ir_module(&module);
        assert!(!result.has_fatal_errors());
        assert_eq!(result.functions.len(), 1);

        let func = &result.functions[0];
        assert_eq!(func.name, "add");
        assert_eq!(func.num_params, 2);

        // Check instructions:
        //   LoadConstInt is NOT emitted here — params are in slots 0, 1
        //   Add { dst: 2, lhs: 0, rhs: 1 }
        //   Move { dst: 0, src: 2 }
        //   Return { value: 0 }
        let instrs = &func.instructions;
        assert!(instrs.len() >= 2, "expected at least 2 instructions, got {}", instrs.len());

        // Find the Add instruction
        let has_add = instrs.iter().any(|i| matches!(i, Instr::Add { dst: 2, lhs: 0, rhs: 1 }));
        assert!(has_add, "expected Add {{ dst: 2, lhs: 0, rhs: 1 }}, got {:?}", instrs);

        // Find the Return instruction
        let has_ret = instrs.iter().any(|i| matches!(i, Instr::Return { value: 0 }));
        assert!(has_ret, "expected Return {{ value: 0 }}, got {:?}", instrs);
    }

    #[test]
    fn test_compile_conditional() {
        // fn abs(x: i64) -> i64 {
        //     v1 = x < 0
        //     if v1 { return -x } else { return x }
        // }
        let v0 = ValueId(0); // param x
        let v1 = ValueId(1); // x < 0
        let v2 = ValueId(2); // -x

        let entry_block = FlatBlock {
            id: BlockId(0),
            instrs: vec![
                make_instr(Some(v1), IrOp::BinOp {
                    op: BinOpKind::Lt,
                    lhs: v0,
                    rhs: ValueId(100), // const 0
                }),
                make_instr(None, IrOp::CondBr {
                    cond: v1,
                    if_true: BlockId(1),
                    if_false: BlockId(2),
                }),
            ],
            span: dummy_span(),
        };

        let true_block = FlatBlock {
            id: BlockId(1),
            instrs: vec![
                make_instr(Some(v2), IrOp::UnOp {
                    op: UnOpKind::Neg,
                    operand: v0,
                }),
                make_instr(None, IrOp::Ret { value: Some(v2) }),
            ],
            span: dummy_span(),
        };

        let false_block = FlatBlock {
            id: BlockId(2),
            instrs: vec![
                make_instr(None, IrOp::Ret { value: Some(v0) }),
            ],
            span: dummy_span(),
        };

        // Need a constant for 0
        let v100 = ValueId(100);
        // Insert the const into entry block
        let mut entry_instrs = vec![make_instr(Some(v100), IrOp::ConstInt { value: 0, ty: IrType::Int { width: 64, signed: true } })];
        entry_instrs.extend(entry_block.instrs);

        let ir_func = FlatIrFunction {
            name: "abs".to_string(),
            params: vec![(v0, IrType::Int { width: 64, signed: true })],
            ret_ty: IrType::Int { width: 64, signed: true },
            blocks: vec![
                FlatBlock { id: BlockId(0), instrs: entry_instrs, span: dummy_span() },
                true_block,
                false_block,
            ],
            entry: BlockId(0),
            effects: EffectFlags::pure(),
            requires: vec![],
            ensures: vec![],
            span: dummy_span(),
        };

        let module = FlatIrModule {
            functions: vec![ir_func],
            intrinsics: vec![],
            span: dummy_span(),
            lowering_errors: vec![],
        };

        let result = compile_ir_module(&module);
        assert!(!result.has_fatal_errors());
        assert_eq!(result.functions.len(), 1);

        let func = &result.functions[0];
        // Should have: LoadConstInt, Lt, JumpFalse, Jump, Neg, Move, Return, Move, Return
        assert!(func.instructions.len() >= 5, "expected at least 5 instructions, got {}", func.instructions.len());
    }

    #[test]
    fn test_compile_constants() {
        let v0 = ValueId(0);

        let entry_block = FlatBlock {
            id: BlockId(0),
            instrs: vec![
                make_instr(Some(v0), IrOp::ConstInt { value: 42, ty: IrType::Int { width: 64, signed: true } }),
                make_instr(None, IrOp::Ret { value: Some(v0) }),
            ],
            span: dummy_span(),
        };

        let ir_func = FlatIrFunction {
            name: "const42".to_string(),
            params: vec![],
            ret_ty: IrType::Int { width: 64, signed: true },
            blocks: vec![entry_block],
            entry: BlockId(0),
            effects: EffectFlags::pure(),
            requires: vec![],
            ensures: vec![],
            span: dummy_span(),
        };

        let module = FlatIrModule {
            functions: vec![ir_func],
            intrinsics: vec![],
            span: dummy_span(),
            lowering_errors: vec![],
        };

        let result = compile_ir_module(&module);
        assert_eq!(result.functions.len(), 1);

        let func = &result.functions[0];
        // LoadConstInt { dst: 0, value: 42 }
        // Move { dst: 0, src: 0 } — no-op move (dst == src)
        // Return { value: 0 }
        let has_load = func.instructions.iter().any(|i| {
            matches!(i, Instr::LoadConstInt { dst: 0, value: 42 })
        });
        assert!(has_load, "expected LoadConstInt {{ dst: 0, value: 42 }}, got {:?}", func.instructions);
    }
}
