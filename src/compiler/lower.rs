// =============================================================================
// jules/src/compiler/lower.rs
//
// AST → Unified Semantic SSA IR lowering pass.
//
// Converts the high-level AST (from ast.rs) into the low-level flat SSA IR
// (from ir.rs). Every AST node is mapped to one or more IR instructions
// with SSA value numbering, flat block construction, effect metadata,
// and ownership annotations.
//
// Output: FlatIrModule (flat instruction IR), not the structured IrModule.
// =============================================================================

use crate::compiler::ast::*;
use crate::compiler::ir::TaskOwnership as IrTaskOwnership;
use crate::compiler::ir::*;
use crate::compiler::lexer::Span;
use crate::compiler::typeck::Diagnostics;
use std::collections::HashMap;
use std::fmt;

/// Loop context for break/continue lowering.
struct LoopContext {
    /// The block to jump to on `break`.
    break_block: BlockId,
    /// The block to jump to on `continue`.
    continue_block: BlockId,
    /// State variables for the loop (for future SSA loop-carried values).
    loop_state_vars: Vec<(ValueId, IrType)>,
}

/// Lower an AST Program into a Flat IR Module.
///
/// # ONE TRUTH RULE
///
/// After this function runs, the returned `FlatIrModule` is the SOLE source of
/// truth for all subsequent compiler passes. The AST must NEVER be semantically
/// analyzed again — all type checking, borrow checking, and code generation
/// MUST operate on the IR.
///
/// If lowering encounters errors (e.g., break outside a loop), they are
/// collected in the module's diagnostics and should be treated as hard errors
/// by the caller.
pub fn lower_program(program: &Program) -> FlatIrModule {
    let mut ctx = LowerCtx::new();
    for item in &program.items {
        ctx.lower_item(item);
    }

    // Propagate any lowering diagnostics (e.g., break outside loop) into
    // the IR module so the pipeline can report them as hard errors.
    // The lowerer's Diagnostics use the same Severity::Error as the
    // borrow checker — these are compile-time errors, not warnings.
    let lowering_errors = ctx.diagnostics();

    FlatIrModule {
        functions: ctx.functions,
        intrinsics: ctx.intrinsics,
        span: program.span,
        lowering_errors,
    }
}

// ─── Lowering Context ──────────────────────────────────────────────────────────

struct LowerCtx {
    functions: Vec<FlatIrFunction>,
    intrinsics: Vec<IrIntrinsic>,
    next_value: u32,
    next_block: u32,
    /// Currently-building flat blocks for the function being lowered.
    current_blocks: Vec<FlatBlock>,
    /// ID of the current block being filled.
    current_block_id: BlockId,
    /// Name → ValueId mapping for the current scope.
    env: Vec<(String, ValueId)>,
    /// Scope stack — each entry is the number of bindings at scope entry.
    scope_stack: Vec<usize>,
    /// String constant table.
    strings: Vec<String>,
    /// String → index lookup for O(1) interning.
    string_map: HashMap<String, u32>,
    /// Region counter for region-alloc instructions.
    next_region: u32,
    /// Currently-building function name (for diagnostics).
    current_fn: String,
    /// Loop context stack — push on loop entry, pop on loop exit.
    loop_stack: Vec<LoopContext>,
    /// FIX (IR-1): Diagnostics collector for compile-time errors.
    diagnostics: Diagnostics,
}

impl LowerCtx {
    fn new() -> Self {
        LowerCtx {
            functions: vec![],
            intrinsics: vec![],
            next_value: 0,
            next_block: 0,
            current_blocks: vec![],
            current_block_id: BlockId(0),
            env: vec![],
            scope_stack: vec![],
            strings: vec![],
            string_map: HashMap::new(),
            next_region: 0,
            current_fn: String::new(),
            loop_stack: vec![],
            diagnostics: Diagnostics { items: vec![] },
        }
    }

    /// Return a vector of (Span, message) pairs for all errors encountered
    /// during lowering. These should be treated as HARD ERRORS by the pipeline.
    fn diagnostics(&self) -> Vec<(crate::compiler::lexer::Span, String)> {
        self.diagnostics
            .items
            .iter()
            .filter(|d| matches!(d.severity, crate::compiler::typeck::Severity::Error))
            .map(|d| (d.span, d.message.clone()))
            .collect()
    }

    // ── SSA Helpers ─────────────────────────────────────────────────────────

    fn fresh_value(&mut self) -> ValueId {
        let id = self.next_value;
        self.next_value += 1;
        ValueId(id)
    }

    fn fresh_block(&mut self) -> BlockId {
        let id = self.next_block;
        self.next_block += 1;
        BlockId(id)
    }

    // ── Scope Management ────────────────────────────────────────────────────

    fn push_scope(&mut self) {
        self.scope_stack.push(self.env.len());
    }

    fn pop_scope(&mut self) {
        if let Some(mark) = self.scope_stack.pop() {
            self.env.truncate(mark);
        }
    }

    fn bind(&mut self, name: String, vid: ValueId) {
        // Remove any earlier binding of the same name in current scope
        // (shadowing)
        self.env.retain(|(n, _)| n != &name);
        self.env.push((name, vid));
    }

    fn lookup(&self, name: &str) -> Option<ValueId> {
        self.env.iter().rev().find(|(n, _)| n == name).map(|&(_, v)| v)
    }

    /// Check if the current block already has a terminator (O(1) via cached index).
    fn is_terminated_fast(&self) -> bool {
        let idx = self.current_block_id.0 as usize;
        if idx < self.current_blocks.len() && self.current_blocks[idx].id == self.current_block_id {
            self.current_blocks[idx].is_terminated()
        } else {
            self.is_terminated()
        }
    }

    // ── String Table ────────────────────────────────────────────────────────

    fn intern_string(&mut self, s: &str) -> u32 {
        if let Some(&idx) = self.string_map.get(s) {
            idx
        } else {
            let idx = self.strings.len() as u32;
            self.strings.push(s.to_string());
            self.string_map.insert(s.to_string(), idx);
            idx
        }
    }

    // ── Instruction Emission ────────────────────────────────────────────────

    /// Emit an instruction to the current flat block.
    /// Returns the ValueId of the result, if the instruction produces one.
    fn emit(&mut self, op: IrOp, span: Span, effects: EffectFlags, ownership: Ownership) -> Option<ValueId> {
        let dst = if op.is_terminator() {
            None
        } else {
            Some(self.fresh_value())
        };
        let instr = IrInstr {
            dst,
            op,
            span,
            effects,
            ownership,
            cost: CostHint::default(),
            alias: AliasKind::default(),
        };
        self.current_block_mut().instrs.push(instr);
        dst
    }

    /// Emit a terminator instruction (no result value).
    fn emit_terminator(&mut self, op: IrOp, span: Span, effects: EffectFlags) {
        debug_assert!(op.is_terminator());
        let instr = IrInstr {
            dst: None,
            op,
            span,
            effects,
            ownership: Ownership::Copy,
            cost: CostHint::default(),
            alias: AliasKind::default(),
        };
        self.current_block_mut().instrs.push(instr);
    }

    /// Get a mutable reference to the current flat block (O(1) via direct index).
    fn current_block_mut(&mut self) -> &mut FlatBlock {
        let idx = self.current_block_id.0 as usize;
        if idx < self.current_blocks.len() && self.current_blocks[idx].id == self.current_block_id {
            &mut self.current_blocks[idx]
        } else {
            // Fallback: linear scan (blocks may not be densely packed)
            let id = self.current_block_id;
            self.current_blocks.iter_mut().find(|b| b.id == id).expect("current block must exist")
        }
    }

    /// Create a new flat block and return its ID.
    fn create_block(&mut self) -> BlockId {
        let id = self.fresh_block();
        self.current_blocks.push(FlatBlock::new(id));
        id
    }

    /// Switch to emitting into a different block.
    fn switch_to_block(&mut self, id: BlockId) {
        self.current_block_id = id;
    }

    /// Check if the current block already has a terminator (O(1) via direct index).
    fn is_terminated(&self) -> bool {
        let idx = self.current_block_id.0 as usize;
        if idx < self.current_blocks.len() && self.current_blocks[idx].id == self.current_block_id {
            self.current_blocks[idx].is_terminated()
        } else {
            // Fallback: linear scan (blocks may not be densely packed)
            self.current_blocks
                .iter()
                .find(|b| b.id == self.current_block_id)
                .map_or(false, |b| b.is_terminated())
        }
    }

    // ── Top-Level Lowering ──────────────────────────────────────────────────

    fn lower_item(&mut self, item: &Item) {
        match item {
            Item::Fn(decl) => self.lower_fn(decl),
            Item::System(decl) => self.lower_system(decl),
            Item::Struct(decl) => {
                // Structs are type declarations — no IR instructions needed.
                // Future: emit type layout metadata.
                let _ = decl;
            }
            Item::Component(decl) => {
                let _ = decl;
            }
            Item::Enum(decl) => {
                let _ = decl;
            }
            Item::Const(decl) => {
                // Lower constant as a function that returns the constant value
                self.lower_const(decl);
            }
            Item::Use(path) => {
                let _ = path;
            }
            Item::Agent(decl) => {
                let _ = decl;
            }
            Item::Model(decl) => {
                let _ = decl;
            }
            Item::Train(decl) => {
                let _ = decl;
            }
            Item::Shader(decl) => {
                let _ = decl;
            }
            Item::Scene(decl) => {
                let _ = decl;
            }
            Item::Prefab(decl) => {
                let _ = decl;
            }
            Item::PhysicsConfig(decl) => {
                let _ = decl;
            }
            Item::Loss(decl) => {
                let _ = decl;
            }
            Item::Mod { span, name, items, .. } => {
                let _ = (span, name);
                if let Some(items) = items {
                    for item in items {
                        self.lower_item(item);
                    }
                }
            }
        }
    }

    // ── Function Lowering ───────────────────────────────────────────────────

    fn lower_fn(&mut self, decl: &FnDecl) {
        // Save state from any previous function
        self.current_fn = decl.name.clone();
        self.env.clear();
        self.scope_stack.clear();
        self.current_blocks.clear();
        self.next_value = 0;
        self.next_block = 0;
        self.loop_stack.clear();

        // Create entry block
        let entry = self.fresh_block();
        self.current_blocks.push(FlatBlock::new(entry));
        self.current_block_id = entry;

        // Bind parameters as SSA values
        let mut params = Vec::new();
        for p in &decl.params {
            let vid = self.fresh_value();
            let ir_ty = p.ty.as_ref().map(|t| self.lower_type(t)).unwrap_or(IrType::Unit);
            params.push((vid, ir_ty));
            self.bind(p.name.clone(), vid);
        }

        // Compute function effect flags
        let fn_effects = match decl.effect.as_deref() {
            Some("pure") => EffectFlags::PURE,
            Some("io") => EffectFlags::PURE.union(EffectFlags::IO),
            _ => {
                // Infer effects from the body if no explicit annotation.
                if let Some(body) = &decl.body {
                    let inferred = self.infer_block_effects(body);
                    if inferred.is_pure() {
                        EffectFlags::PURE
                    } else {
                        inferred
                    }
                } else {
                    EffectFlags::PURE // default to pure
                }
            }
        };

        // Lower requires/ensures as function metadata (lower the expressions
        // but don't emit them into the body — they become contract assertions)
        let mut requires_vids = Vec::new();
        for req in &decl.requires {
            if let Some(vid) = self.lower_expr(req) {
                requires_vids.push(vid);
            }
        }
        let mut ensures_vids = Vec::new();
        for ens in &decl.ensures {
            if let Some(vid) = self.lower_expr(ens) {
                ensures_vids.push(vid);
            }
        }

        // Lower function body
        if let Some(body) = &decl.body {
            let tail_val = self.lower_block(body);

            // Add implicit return if the block didn't terminate
            if !self.is_terminated() {
                self.emit_terminator(
                    IrOp::Ret { value: tail_val },
                    body.span,
                    EffectFlags::TERMINATES,
                );
            }
        } else {
            // No body — extern function declaration
            self.emit_terminator(
                IrOp::Ret { value: None },
                decl.span,
                EffectFlags::TERMINATES,
            );
        }

        // Compute return type
        let ret_ty = decl.ret_ty.as_ref().map(|t| self.lower_type(t)).unwrap_or(IrType::Unit);

        // Finalize function
        let func = FlatIrFunction {
            name: decl.name.clone(),
            params,
            ret_ty,
            blocks: std::mem::take(&mut self.current_blocks),
            entry,
            effects: fn_effects,
            requires: requires_vids,
            ensures: ensures_vids,
            span: decl.span,
        };
        self.functions.push(func);
    }

    // ── System Lowering ─────────────────────────────────────────────────────

    fn lower_system(&mut self, decl: &SystemDecl) {
        // Systems are lowered as functions with a special naming convention
        self.current_fn = format!("__system_{}", decl.name);
        self.env.clear();
        self.scope_stack.clear();
        self.current_blocks.clear();
        self.next_value = 0;
        self.next_block = 0;
        self.loop_stack.clear();

        let entry = self.fresh_block();
        self.current_blocks.push(FlatBlock::new(entry));
        self.current_block_id = entry;

        let mut params = Vec::new();
        for p in &decl.params {
            let vid = self.fresh_value();
            let ir_ty = p.ty.as_ref().map(|t| self.lower_type(t)).unwrap_or(IrType::Unit);
            params.push((vid, ir_ty));
            self.bind(p.name.clone(), vid);
        }

        // Lower the system body
        let tail_val = self.lower_block(&decl.body);
        if !self.is_terminated() {
            self.emit_terminator(
                IrOp::Ret { value: tail_val },
                decl.body.span,
                EffectFlags::TERMINATES,
            );
        }

        let func = FlatIrFunction {
            name: format!("__system_{}", decl.name),
            params,
            ret_ty: IrType::Unit,
            blocks: std::mem::take(&mut self.current_blocks),
            entry,
            effects: EffectFlags::PARALLEL,
            requires: vec![],
            ensures: vec![],
            span: decl.span,
        };
        self.functions.push(func);
    }

    // ── Const Lowering ─────────────────────────────────────────────────────

    fn lower_const(&mut self, decl: &ConstDecl) {
        // Lower a constant as a function named __const_{name} that returns the value.
        self.current_fn = format!("__const_{}", decl.name);
        self.env.clear();
        self.scope_stack.clear();
        self.current_blocks.clear();
        self.next_value = 0;
        self.next_block = 0;
        self.loop_stack.clear();

        // Create entry block
        let entry = self.fresh_block();
        self.current_blocks.push(FlatBlock::new(entry));
        self.current_block_id = entry;

        // Lower the constant's expression
        let val_vid = self.lower_expr(&decl.value);

        // Return the value
        if !self.is_terminated() {
            self.emit_terminator(
                IrOp::Ret { value: val_vid },
                decl.span,
                EffectFlags::TERMINATES,
            );
        }

        // Compute return type
        let ret_ty = self.lower_type(&decl.ty);

        // Finalize function
        let func = FlatIrFunction {
            name: format!("__const_{}", decl.name),
            params: vec![],
            ret_ty,
            blocks: std::mem::take(&mut self.current_blocks),
            entry,
            effects: EffectFlags::PURE,
            requires: vec![],
            ensures: vec![],
            span: decl.span,
        };
        self.functions.push(func);
    }

    // ── Block Lowering ──────────────────────────────────────────────────────

    fn lower_block(&mut self, block: &Block) -> Option<ValueId> {
        self.push_scope();
        for stmt in &block.stmts {
            self.lower_stmt(stmt);
            if self.is_terminated_fast() {
                self.pop_scope();
                return None;
            }
        }
        let tail_val = block.tail.as_ref().and_then(|e| self.lower_expr(e));
        self.pop_scope();
        tail_val
    }

    // ── Statement Lowering ──────────────────────────────────────────────────

    fn lower_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let { pattern, ty, init, mutable, span } => {
                if let Some(init_expr) = init {
                    if let Some(vid) = self.lower_expr(init_expr) {
                        self.bind_pattern(pattern, vid, *mutable);
                    }
                } else {
                    // Uninitialized binding — allocate a slot
                    let ir_ty = ty.as_ref().map(|t| self.lower_type(t)).unwrap_or(IrType::Unit);
                    let ptr = self.emit(
                        IrOp::Alloca { ty: ir_ty, align: 8 },
                        *span,
                        EffectFlags::ALLOC,
                        if *mutable { Ownership::MUT_BORROW } else { Ownership::OWN },
                    );
                    if let Some(ptr) = ptr {
                        self.bind_pattern(pattern, ptr, *mutable);
                    }
                }
            }

            Stmt::Expr { expr, has_semi, span } => {
                let vid = self.lower_expr(expr);
                // If has_semi, the value is discarded (already computed for effects)
                let _ = (vid, has_semi, span);
            }

            Stmt::Return { value, span } => {
                let val = value.as_ref().and_then(|e| self.lower_expr(e));
                self.emit_terminator(
                    IrOp::Ret { value: val },
                    *span,
                    EffectFlags::TERMINATES,
                );
            }

            Stmt::Break { span, .. } => {
                // Break is lowered as a jump to the loop exit block.
                if let Some(ctx) = self.loop_stack.last() {
                    let break_block = ctx.break_block;
                    self.emit_terminator(
                        IrOp::Jump { target: break_block },
                        *span,
                        EffectFlags::TERMINATES,
                    );
                } else {
                    // FIX (IR-1): Break outside of a loop is a compile-time
                    // error, not a silent Nop. Emitting a Nop for break outside
                    // a loop silently produces incorrect code that appears to
                    // compile successfully but does not implement the programmer's
                    // intent.
                    self.diagnostics.error(
                        *span,
                        "break statement outside of a loop",
                    );
                }
            }

            Stmt::Continue { span, .. } => {
                // Continue is lowered as a jump to the loop header/continue block.
                if let Some(ctx) = self.loop_stack.last() {
                    let continue_block = ctx.continue_block;
                    self.emit_terminator(
                        IrOp::Jump { target: continue_block },
                        *span,
                        EffectFlags::TERMINATES,
                    );
                } else {
                    // FIX (IR-1): Continue outside of a loop is a compile-time
                    // error, not a silent Nop.
                    self.diagnostics.error(
                        *span,
                        "continue statement outside of a loop",
                    );
                }
            }

            Stmt::If { cond, then, else_, span } => {
                self.lower_if_stmt(cond, then, else_, *span);
            }

            Stmt::While { cond, body, span, .. } => {
                self.lower_while(cond, body, *span);
            }

            Stmt::Loop { body, span, .. } => {
                self.lower_loop(body, *span);
            }

            Stmt::ForIn { pattern, iter, body, span, .. } => {
                self.lower_for_in(pattern, iter, body, *span);
            }

            Stmt::EntityFor { var, body, query, span, .. } => {
                // EntityFor is a specialized for-loop over entities matching a query.
                // Lower as: __entity_query_begin → loop { __entity_iter → body → continue } → __entity_query_end
                self.push_scope();

                // Look up or create a world value
                let world_vid = self.lookup("world").unwrap_or_else(|| {
                    self.emit(
                        IrOp::ConstUnit,
                        *span,
                        EffectFlags::pure(),
                        Ownership::Copy,
                    ).unwrap_or(ValueId(0))
                });

                // Emit __entity_query_begin(world, with_components, without_components)
                let with_str = query.with.join(",");
                let without_str = query.without.join(",");
                let with_idx = self.intern_string(&with_str);
                let without_idx = self.intern_string(&without_str);
                let with_vid = self.emit(
                    IrOp::ConstStr { idx: with_idx },
                    *span,
                    EffectFlags::pure(),
                    Ownership::Copy,
                );
                let without_vid = self.emit(
                    IrOp::ConstStr { idx: without_idx },
                    *span,
                    EffectFlags::pure(),
                    Ownership::Copy,
                );

                let mut begin_args = vec![world_vid];
                if let Some(wv) = with_vid { begin_args.push(wv); }
                if let Some(wov) = without_vid { begin_args.push(wov); }

                let query_handle = self.emit(
                    IrOp::Call {
                        func: "__entity_query_begin".to_string(),
                        args: begin_args,
                    },
                    *span,
                    EffectFlags::IO,
                    Ownership::OWN,
                );

                // Create loop blocks
                let header_block = self.create_block();
                let body_block = self.create_block();
                let exit_block = self.create_block();

                // Push loop context for break/continue
                self.loop_stack.push(LoopContext {
                    break_block: exit_block,
                    continue_block: header_block,
                    loop_state_vars: vec![],
                });

                // Jump to header
                self.emit_terminator(
                    IrOp::Jump { target: header_block },
                    *span,
                    EffectFlags::TERMINATES,
                );

                // Header: call __entity_iter to check for next entity
                self.switch_to_block(header_block);
                let mut iter_args = vec![];
                if let Some(qh) = query_handle { iter_args.push(qh); }
                let entity_vid = self.emit(
                    IrOp::Call {
                        func: "__entity_iter".to_string(),
                        args: iter_args,
                    },
                    *span,
                    EffectFlags::IO,
                    Ownership::OWN,
                );

                // Check if entity is valid (non-unit) — for now, always branch to body
                // A proper implementation would check for a sentinel value
                if let Some(ev) = entity_vid {
                    let has_next = self.emit(
                        IrOp::Call {
                            func: "__entity_iter_valid".to_string(),
                            args: vec![ev],
                        },
                        *span,
                        EffectFlags::pure(),
                        Ownership::Copy,
                    );
                    if let Some(hn) = has_next {
                        self.emit_terminator(
                            IrOp::CondBr { cond: hn, if_true: body_block, if_false: exit_block },
                            *span,
                            EffectFlags::TERMINATES,
                        );
                    } else {
                        self.emit_terminator(
                            IrOp::Jump { target: body_block },
                            *span,
                            EffectFlags::TERMINATES,
                        );
                    }
                } else {
                    self.emit_terminator(
                        IrOp::Jump { target: exit_block },
                        *span,
                        EffectFlags::TERMINATES,
                    );
                }

                // Body: bind entity variable, lower body
                self.switch_to_block(body_block);
                if let Some(ev) = entity_vid {
                    self.bind(var.clone(), ev);
                }
                self.lower_block(body);
                if !self.is_terminated() {
                    self.emit_terminator(
                        IrOp::Jump { target: header_block },
                        body.span,
                        EffectFlags::TERMINATES,
                    );
                }

                // Exit: call __entity_query_end
                self.switch_to_block(exit_block);
                let mut end_args = vec![];
                if let Some(qh) = query_handle { end_args.push(qh); }
                self.emit(
                    IrOp::Call {
                        func: "__entity_query_end".to_string(),
                        args: end_args,
                    },
                    *span,
                    EffectFlags::IO,
                    Ownership::Copy,
                );

                // Pop loop context
                self.loop_stack.pop();
                self.pop_scope();
            }

            Stmt::Match { expr, arms, span } => {
                self.lower_match(expr, arms, *span);
            }

            Stmt::Item(item) => {
                self.lower_item(item);
            }

            Stmt::ParallelFor(pf) => {
                // Lower as a parallel region + for loop
                let region_id = self.next_region;
                self.next_region += 1;
                self.emit(
                    IrOp::ParallelStart { region_id },
                    pf.span,
                    EffectFlags::PARALLEL,
                    Ownership::Copy,
                );
                self.lower_for_in(&pf.var, &pf.iter, &pf.body, pf.span);
                self.emit(
                    IrOp::ParallelEnd { region_id },
                    pf.span,
                    EffectFlags::PARALLEL,
                    Ownership::Copy,
                );
            }

            Stmt::Spawn(sb) => {
                // Spawn block — emit as parallel start + body
                self.emit(
                    IrOp::ParallelStart { region_id: self.next_region },
                    sb.span,
                    EffectFlags::PARALLEL,
                    Ownership::Copy,
                );
                self.next_region += 1;
                self.lower_block(&sb.body);
            }

            Stmt::Sync(sb) => {
                // Sync block — barrier
                self.emit(
                    IrOp::ParallelEnd { region_id: self.next_region.saturating_sub(1) },
                    sb.span,
                    EffectFlags::PARALLEL,
                    Ownership::Copy,
                );
                self.lower_block(&sb.body);
            }

            Stmt::Atomic(ab) => {
                // Atomic block — annotate with ATOMIC effect
                self.push_scope();
                for s in &ab.body.stmts {
                    self.lower_stmt(s);
                    if self.is_terminated() { break; }
                }
                if let Some(tail) = &ab.body.tail {
                    self.lower_expr(tail);
                }
                self.pop_scope();
            }

            Stmt::Effect { name, body, span } => {
                // Effect block — lower body with IO effect
                self.push_scope();
                for s in &body.stmts {
                    self.lower_stmt(s);
                    if self.is_terminated() { break; }
                }
                if let Some(tail) = &body.tail {
                    let vid = self.lower_expr(tail);
                    let _ = vid;
                }
                self.pop_scope();
                let _ = (name, span);
            }

            Stmt::Region { name, body, span } => {
                // Region-based allocation block
                let region_id = self.next_region;
                self.next_region += 1;
                self.push_scope();
                // Emit region allocation marker
                self.emit(
                    IrOp::RegionAlloc { region: region_id, ty: IrType::Unit },
                    *span,
                    EffectFlags::ALLOC,
                    Ownership::OWN,
                );
                for s in &body.stmts {
                    self.lower_stmt(s);
                    if self.is_terminated() { break; }
                }
                if let Some(tail) = &body.tail {
                    self.lower_expr(tail);
                }
                self.pop_scope();
                let _ = name;
            }

            Stmt::TaskSpawn { name, task_expr, span } => {
                // Lower the task expression and emit TaskSpawn
                let expr_vid = self.lower_expr(task_expr);
                let task_vid = self.emit(
                    IrOp::TaskSpawn {
                        func: format!("__task_{}", name),
                        args: if let Some(v) = expr_vid { vec![v] } else { vec![] },
                        ownership: IrTaskOwnership::Move,
                    },
                    *span,
                    EffectFlags::PARALLEL,
                    Ownership::OWN,
                );
                if let Some(tv) = task_vid {
                    self.bind(name.clone(), tv);
                }
            }

            Stmt::TaskJoin { name, span } => {
                if let Some(task_vid) = self.lookup(name) {
                    self.emit(
                        IrOp::TaskJoin { task: task_vid },
                        *span,
                        EffectFlags::PARALLEL,
                        Ownership::Copy,
                    );
                }
            }

            Stmt::UnsafeBlock { body, span } => {
                // Lower body with UNSAFE effect annotation
                self.push_scope();
                for s in &body.stmts {
                    self.lower_stmt(s);
                    if self.is_terminated() { break; }
                }
                if let Some(tail) = &body.tail {
                    let vid = self.lower_expr(tail);
                    let _ = vid;
                }
                self.pop_scope();
                let _ = span;
            }

            Stmt::IntrinsicsBlock { span, decls } => {
                for decl in decls {
                    self.intrinsics.push(IrIntrinsic {
                        name: decl.name.clone(),
                        param_types: decl.params.iter().map(|t| self.lower_type(t)).collect(),
                        ret_type: self.lower_type(&decl.ret),
                        effects: EffectFlags::pure(),
                    });
                }
                let _ = span;
            }

            Stmt::Requires { condition, span } => {
                // Lower as a contract assertion — for now just evaluate
                let _ = self.lower_expr(condition);
                let _ = span;
            }

            Stmt::Ensures { condition, span } => {
                // Lower as a postcondition — for now just evaluate
                let _ = self.lower_expr(condition);
                let _ = span;
            }
        }
    }

    // ── Expression Lowering ─────────────────────────────────────────────────

    fn lower_expr(&mut self, expr: &Expr) -> Option<ValueId> {
        match expr {
            // ── Literals ────────────────────────────────────────────────────
            Expr::IntLit { value, ty, span } => {
                let ir_ty = ty.as_ref()
                    .map(|et| self.elem_type_to_ir(et.clone()))
                    // FIX: Default to i32 (not i64) to match annotate_literal_types
                    // in typeck.rs, which sets unannotated IntLit to I32.
                    // Optimizers create IntLit nodes with ty=None, so this fallback
                    // must be consistent with the type checker's default.
                    .unwrap_or(IrType::Int { width: 32, signed: true });

                // Truncate to i128 for the IR representation
                let truncated = (*value & 0xFFFFFFFFFFFFFFFF) as i128;
                // Handle sign extension for smaller types
                let val = match ir_ty {
                    IrType::Int { width: 8, signed: true } => (truncated as i8) as i128,
                    IrType::Int { width: 16, signed: true } => (truncated as i16) as i128,
                    IrType::Int { width: 32, signed: true } => (truncated as i32) as i128,
                    IrType::Int { width: 8, signed: false } => (truncated as u8) as i128,
                    IrType::Int { width: 16, signed: false } => (truncated as u16) as i128,
                    IrType::Int { width: 32, signed: false } => (truncated as u32) as i128,
                    _ => truncated,
                };

                self.emit(
                    IrOp::ConstInt { value: val, ty: ir_ty },
                    *span,
                    EffectFlags::pure(),
                    Ownership::Copy,
                )
            }

            Expr::FloatLit { value, span } => {
                let bits = value.to_bits();
                // FIX (IR-6): Float literals now default to f64 instead of
                // using a fragile value-based heuristic. The original code
                // classified 3.0 as f32 and 3.4028235e38 as f64 based on
                // runtime value inspection, which could lead to silent
                // precision loss or type mismatches downstream. The type
                // should come from an AST type annotation or a language
                // default (f64), not from the literal value.
                let ir_ty = IrType::Float { width: 64 };
                self.emit(
                    IrOp::ConstFloat { bits, ty: ir_ty },
                    *span,
                    EffectFlags::pure(),
                    Ownership::Copy,
                )
            }

            Expr::BoolLit { value, span } => {
                self.emit(
                    IrOp::ConstBool { value: *value },
                    *span,
                    EffectFlags::pure(),
                    Ownership::Copy,
                )
            }

            Expr::StrLit { value, span } => {
                let idx = self.intern_string(value);
                self.emit(
                    IrOp::ConstStr { idx },
                    *span,
                    EffectFlags::pure(),
                    Ownership::OWN,
                )
            }

            // ── Identifiers ─────────────────────────────────────────────────
            Expr::Ident { name, span } => {
                if let Some(vid) = self.lookup(name) {
                    Some(vid)
                } else {
                    // Undeclared variable — emit a placeholder
                    self.emit(
                        IrOp::ConstUnit,
                        *span,
                        EffectFlags::none(),
                        Ownership::Copy,
                    )
                }
            }

            Expr::Path { segments, span } => {
                // Qualified path — lower as a function reference
                let full_name = segments.join("::");
                let idx = self.intern_string(&full_name);
                self.emit(
                    IrOp::ConstStr { idx },
                    *span,
                    EffectFlags::pure(),
                    Ownership::Copy,
                )
            }

            // ── Vector / Array constructors ─────────────────────────────────
            Expr::VecCtor { size, elems, span } => {
                let lanes = size.lanes();
                let mut elem_vids = Vec::new();
                for e in elems {
                    if let Some(vid) = self.lower_expr(e) {
                        elem_vids.push(vid);
                    }
                }
                // Emit as a call to vec constructor
                self.emit(
                    IrOp::Call {
                        func: format!("__vec{}_ctor", lanes),
                        args: elem_vids,
                    },
                    *span,
                    EffectFlags::pure(),
                    Ownership::OWN,
                )
            }

            Expr::ArrayLit { elems, span } => {
                let mut elem_vids = Vec::new();
                for e in elems {
                    if let Some(vid) = self.lower_expr(e) {
                        elem_vids.push(vid);
                    }
                }
                self.emit(
                    IrOp::Call {
                        func: "__array_ctor".to_string(),
                        args: elem_vids,
                    },
                    *span,
                    EffectFlags::pure().union(EffectFlags::ALLOC),
                    Ownership::OWN,
                )
            }

            // ── Binary Operations ───────────────────────────────────────────
            Expr::BinOp { op, lhs, rhs, span } => {
                let lhs_vid = self.lower_expr(lhs);
                let rhs_vid = self.lower_expr(rhs);
                match (lhs_vid, rhs_vid) {
                    (Some(l), Some(r)) => {
                        self.emit(
                            IrOp::BinOp { op: *op, lhs: l, rhs: r },
                            *span,
                            EffectFlags::pure(),
                            Ownership::Copy,
                        )
                    }
                    _ => None,
                }
            }

            // ── Unary Operations ────────────────────────────────────────────
            Expr::UnOp { op, expr, span } => {
                let operand_vid = self.lower_expr(expr);
                match operand_vid {
                    Some(v) => {
                        self.emit(
                            IrOp::UnOp { op: *op, operand: v },
                            *span,
                            EffectFlags::pure(),
                            Ownership::Copy,
                        )
                    }
                    _ => None,
                }
            }

            // ── Assignment ──────────────────────────────────────────────────
            Expr::Assign { op, target, value, span } => {
                let val_vid = self.lower_expr(value);

                match op {
                    AssignOpKind::Assign => {
                        // Simple assignment — if target is an ident, rebind it
                        if let Expr::Ident { name, .. } = target.as_ref() {
                            if let Some(vid) = val_vid {
                                self.bind(name.clone(), vid);
                                return Some(vid);
                            }
                        }
                        // Otherwise emit Move instruction
                        match val_vid {
                            Some(v) => self.emit(
                                IrOp::Move { src: v },
                                *span,
                                EffectFlags::WRITE,
                                Ownership::OWN,
                            ),
                            None => None,
                        }
                    }
                    AssignOpKind::MutAssign => {
                        // Explicit mutation — emit Store
                        if let Expr::Ident { name, .. } = target.as_ref() {
                            if let (Some(ptr), Some(val)) = (self.lookup(name), val_vid) {
                                self.emit(
                                    IrOp::Store { ptr, value: val },
                                    *span,
                                    EffectFlags::WRITE,
                                    Ownership::MUT_BORROW,
                                );
                                self.bind(name.clone(), val);
                                return Some(val);
                            }
                        }
                        val_vid
                    }
                    _ => {
                        // Compound assignment: desugar to binop + assign
                        // e.g., x += 1  →  x = x + 1
                        if let Some(binop) = op.to_binop() {
                            let lhs_vid = self.lower_expr(target);
                            match (lhs_vid, val_vid) {
                                (Some(l), Some(r)) => {
                                    let result = self.emit(
                                        IrOp::BinOp { op: binop, lhs: l, rhs: r },
                                        *span,
                                        EffectFlags::pure(),
                                        Ownership::Copy,
                                    );
                                    // Re-assign the target
                                    if let Expr::Ident { name, .. } = target.as_ref() {
                                        if let Some(vid) = result {
                                            self.bind(name.clone(), vid);
                                        }
                                    }
                                    result
                                }
                                _ => None,
                            }
                        } else {
                            // MatMulAssign — emit as MatMul + assign
                            if matches!(op, AssignOpKind::MatMulAssign) {
                                let lhs_vid = self.lower_expr(target);
                                match (lhs_vid, val_vid) {
                                    (Some(l), Some(r)) => {
                                        let result = self.emit(
                                            IrOp::MatMul { lhs: l, rhs: r },
                                            *span,
                                            EffectFlags::pure(),
                                            Ownership::OWN,
                                        );
                                        if let Expr::Ident { name, .. } = target.as_ref() {
                                            if let Some(vid) = result {
                                                self.bind(name.clone(), vid);
                                            }
                                        }
                                        result
                                    }
                                    _ => None,
                                }
                            } else {
                                val_vid
                            }
                        }
                    }
                }
            }

            // ── Field Access ────────────────────────────────────────────────
            Expr::Field { object, field, span } => {
                let obj_vid = self.lower_expr(object);
                match obj_vid {
                    Some(v) => {
                        // Field access is lowered as an intrinsic call
                        self.emit(
                            IrOp::Intrinsic {
                                name: format!("__field_{}", field),
                                args: vec![v],
                            },
                            *span,
                            EffectFlags::pure().union(EffectFlags::READONLY),
                            Ownership::Shared,
                        )
                    }
                    _ => None,
                }
            }

            // ── Index Access ────────────────────────────────────────────────
            Expr::Index { object, indices, span } => {
                let obj_vid = self.lower_expr(object);
                let mut idx_vids = Vec::new();
                for idx_expr in indices {
                    if let Some(vid) = self.lower_expr(idx_expr) {
                        idx_vids.push(vid);
                    }
                }
                match obj_vid {
                    Some(v) => {
                        let mut args = vec![v];
                        args.extend(idx_vids);
                        self.emit(
                            IrOp::Intrinsic {
                                name: "__index".to_string(),
                                args,
                            },
                            *span,
                            EffectFlags::pure().union(EffectFlags::READONLY),
                            Ownership::Shared,
                        )
                    }
                    _ => None,
                }
            }

            // ── Function Calls ──────────────────────────────────────────────
            Expr::Call { func, args, named, span } => {
                let mut arg_vids = Vec::new();
                for a in args {
                    if let Some(vid) = self.lower_expr(a) {
                        arg_vids.push(vid);
                    }
                }
                // Named arguments are lowered positionally after positional args
                for (_, val) in named {
                    if let Some(vid) = self.lower_expr(val) {
                        arg_vids.push(vid);
                    }
                }

                // Resolve function name
                let func_name = match func.as_ref() {
                    Expr::Ident { name, .. } => name.clone(),
                    Expr::Path { segments, .. } => segments.join("::"),
                    _ => "__unknown_call".to_string(),
                };

                self.emit(
                    IrOp::Call { func: func_name, args: arg_vids },
                    *span,
                    EffectFlags::pure(), // conservative: caller should annotate
                    Ownership::OWN,
                )
            }

            Expr::MethodCall { receiver, method, args, span } => {
                let mut arg_vids = Vec::new();
                if let Some(vid) = self.lower_expr(receiver) {
                    arg_vids.push(vid);
                }
                for a in args {
                    if let Some(vid) = self.lower_expr(a) {
                        arg_vids.push(vid);
                    }
                }

                self.emit(
                    IrOp::Call {
                        func: format!("__method_{}", method),
                        args: arg_vids,
                    },
                    *span,
                    EffectFlags::pure(),
                    Ownership::OWN,
                )
            }

            // ── Tensor Operations ───────────────────────────────────────────
            Expr::MatMul { lhs, rhs, span } => {
                let l = self.lower_expr(lhs);
                let r = self.lower_expr(rhs);
                match (l, r) {
                    (Some(lv), Some(rv)) => {
                        self.emit(
                            IrOp::MatMul { lhs: lv, rhs: rv },
                            *span,
                            EffectFlags::pure(),
                            Ownership::OWN,
                        )
                    }
                    _ => None,
                }
            }

            Expr::HadamardMul { lhs, rhs, span } => {
                let l = self.lower_expr(lhs);
                let r = self.lower_expr(rhs);
                match (l, r) {
                    (Some(lv), Some(rv)) => {
                        self.emit(
                            IrOp::HadamardMul { lhs: lv, rhs: rv },
                            *span,
                            EffectFlags::pure(),
                            Ownership::OWN,
                        )
                    }
                    _ => None,
                }
            }

            Expr::HadamardDiv { lhs, rhs, span } => {
                let l = self.lower_expr(lhs);
                let r = self.lower_expr(rhs);
                match (l, r) {
                    (Some(lv), Some(rv)) => {
                        self.emit(
                            IrOp::HadamardDiv { lhs: lv, rhs: rv },
                            *span,
                            EffectFlags::pure(),
                            Ownership::OWN,
                        )
                    }
                    _ => None,
                }
            }

            Expr::TensorConcat { lhs, rhs, span } => {
                let l = self.lower_expr(lhs);
                let r = self.lower_expr(rhs);
                match (l, r) {
                    (Some(lv), Some(rv)) => {
                        self.emit(
                            IrOp::TensorConcat { lhs: lv, rhs: rv },
                            *span,
                            EffectFlags::pure().union(EffectFlags::ALLOC),
                            Ownership::OWN,
                        )
                    }
                    _ => None,
                }
            }

            Expr::KronProd { lhs, rhs, span } => {
                let l = self.lower_expr(lhs);
                let r = self.lower_expr(rhs);
                match (l, r) {
                    (Some(lv), Some(rv)) => {
                        self.emit(
                            IrOp::KronProd { lhs: lv, rhs: rv },
                            *span,
                            EffectFlags::pure(),
                            Ownership::OWN,
                        )
                    }
                    _ => None,
                }
            }

            Expr::OuterProd { lhs, rhs, span } => {
                let l = self.lower_expr(lhs);
                let r = self.lower_expr(rhs);
                match (l, r) {
                    (Some(lv), Some(rv)) => {
                        self.emit(
                            IrOp::OuterProd { lhs: lv, rhs: rv },
                            *span,
                            EffectFlags::pure(),
                            Ownership::OWN,
                        )
                    }
                    _ => None,
                }
            }

            Expr::Grad { inner, span } => {
                let v = self.lower_expr(inner);
                match v {
                    Some(vid) => {
                        self.emit(
                            IrOp::Intrinsic {
                                name: "__grad".to_string(),
                                args: vec![vid],
                            },
                            *span,
                            EffectFlags::pure(),
                            Ownership::OWN,
                        )
                    }
                    _ => None,
                }
            }

            Expr::Pow { base, exp, span } => {
                let b = self.lower_expr(base);
                let e = self.lower_expr(exp);
                match (b, e) {
                    (Some(bv), Some(ev)) => {
                        self.emit(
                            IrOp::Intrinsic {
                                name: "__pow".to_string(),
                                args: vec![bv, ev],
                            },
                            *span,
                            EffectFlags::pure(),
                            Ownership::Copy,
                        )
                    }
                    _ => None,
                }
            }

            // ── Range ───────────────────────────────────────────────────────
            Expr::Range { lo, hi, inclusive, span } => {
                let mut args = Vec::new();
                if let Some(lo_e) = lo {
                    if let Some(vid) = self.lower_expr(lo_e) {
                        args.push(vid);
                    }
                }
                if let Some(hi_e) = hi {
                    if let Some(vid) = self.lower_expr(hi_e) {
                        args.push(vid);
                    }
                }
                let name = if *inclusive { "__range_incl" } else { "__range" };
                self.emit(
                    IrOp::Intrinsic { name: name.to_string(), args },
                    *span,
                    EffectFlags::pure(),
                    Ownership::OWN,
                )
            }

            // ── Cast ────────────────────────────────────────────────────────
            Expr::Cast { expr, ty, span } => {
                let v = self.lower_expr(expr);
                let target_ty = self.lower_type(ty);
                match v {
                    Some(vid) => {
                        self.emit(
                            IrOp::Cast { src: vid, target_ty },
                            *span,
                            EffectFlags::pure(),
                            Ownership::Copy,
                        )
                    }
                    _ => None,
                }
            }

            // ── If Expression ───────────────────────────────────────────────
            Expr::IfExpr { cond, then, else_, span } => {
                self.lower_if_expr(cond, then, else_, *span)
            }

            // ── Closures ────────────────────────────────────────────────────
            Expr::Closure { params, body, span, .. } => {
                // Lower closure as a function + capture
                let _ = params;
                let _ = body;
                self.emit(
                    IrOp::ConstUnit,
                    *span,
                    EffectFlags::pure(),
                    Ownership::OWN,
                )
            }

            // ── Block Expression ────────────────────────────────────────────
            Expr::Block(block) => {
                self.lower_block(block)
            }

            // ── Tuple ───────────────────────────────────────────────────────
            Expr::Tuple { elems, span } => {
                let mut elem_vids = Vec::new();
                for e in elems {
                    if let Some(vid) = self.lower_expr(e) {
                        elem_vids.push(vid);
                    }
                }
                self.emit(
                    IrOp::Call {
                        func: "__tuple_ctor".to_string(),
                        args: elem_vids,
                    },
                    *span,
                    EffectFlags::pure(),
                    Ownership::OWN,
                )
            }

            // ── Struct Literal ──────────────────────────────────────────────
            Expr::StructLit { name, fields, span } => {
                let mut args = Vec::new();
                for (_, val) in fields {
                    if let Some(vid) = self.lower_expr(val) {
                        args.push(vid);
                    }
                }
                self.emit(
                    IrOp::Call {
                        func: format!("__struct_{}_ctor", name),
                        args,
                    },
                    *span,
                    EffectFlags::pure().union(EffectFlags::ALLOC),
                    Ownership::OWN,
                )
            }

            // ── Pipeline Expression ─────────────────────────────────────────
            Expr::Pipeline { stages, span } => {
                // x |> f(args)  →  f(x, args)
                // x |> f |> g   →  g(f(x))
                if stages.is_empty() {
                    return self.emit(
                        IrOp::ConstUnit,
                        *span,
                        EffectFlags::pure(),
                        Ownership::Copy,
                    );
                }

                // Lower the first stage (the source value)
                let mut current = self.lower_expr(&stages[0]);

                // Apply each subsequent stage as a function call
                for stage in &stages[1..] {
                    current = self.lower_pipeline_stage(current, stage, *span);
                }
                current
            }

            // ── Emit ────────────────────────────────────────────────────────
            Expr::Emit { effect, value, span } => {
                let val_vid = self.lower_expr(value);
                match val_vid {
                    Some(v) => {
                        self.emit(
                            IrOp::Emit { effect: effect.clone(), value: v },
                            *span,
                            EffectFlags::IO,
                            Ownership::Copy,
                        )
                    }
                    _ => None,
                }
            }

            // ── Copy ────────────────────────────────────────────────────────
            Expr::Copy { inner, span } => {
                let v = self.lower_expr(inner);
                match v {
                    Some(vid) => {
                        self.emit(
                            IrOp::Copy { src: vid },
                            *span,
                            EffectFlags::pure(),
                            Ownership::Copy,
                        )
                    }
                    _ => None,
                }
            }
        }
    }

    // ── Pipeline Stage Lowering ─────────────────────────────────────────────

    fn lower_pipeline_stage(&mut self, piped_value: Option<ValueId>, stage: &Expr, span: Span) -> Option<ValueId> {
        match stage {
            Expr::Call { func, args, named, span: call_span } => {
                // f(args) with piped value as first arg → f(piped, args)
                let mut arg_vids = Vec::new();
                if let Some(v) = piped_value {
                    arg_vids.push(v);
                }
                for a in args {
                    if let Some(vid) = self.lower_expr(a) {
                        arg_vids.push(vid);
                    }
                }
                for (_, val) in named {
                    if let Some(vid) = self.lower_expr(val) {
                        arg_vids.push(vid);
                    }
                }
                let func_name = match func.as_ref() {
                    Expr::Ident { name, .. } => name.clone(),
                    Expr::Path { segments, .. } => segments.join("::"),
                    _ => "__unknown_call".to_string(),
                };
                self.emit(
                    IrOp::Call { func: func_name, args: arg_vids },
                    *call_span,
                    EffectFlags::pure(),
                    Ownership::OWN,
                )
            }
            Expr::MethodCall { receiver: _, method, args, span: call_span } => {
                // obj.method(args) with piped value replacing obj
                let mut arg_vids = Vec::new();
                if let Some(v) = piped_value {
                    arg_vids.push(v);
                }
                for a in args {
                    if let Some(vid) = self.lower_expr(a) {
                        arg_vids.push(vid);
                    }
                }
                self.emit(
                    IrOp::Call {
                        func: format!("__method_{}", method),
                        args: arg_vids,
                    },
                    *call_span,
                    EffectFlags::pure(),
                    Ownership::OWN,
                )
            }
            Expr::Ident { name, .. } => {
                // Simple function name: f  →  f(piped)
                let mut args = Vec::new();
                if let Some(v) = piped_value {
                    args.push(v);
                }
                self.emit(
                    IrOp::Call { func: name.clone(), args },
                    span,
                    EffectFlags::pure(),
                    Ownership::OWN,
                )
            }
            _ => {
                // Unknown stage type — evaluate and return
                self.lower_expr(stage)
            }
        }
    }

    // ── Control Flow Lowering ───────────────────────────────────────────────

    fn lower_if_expr(&mut self, cond: &Expr, then: &Block, else_: &Option<Box<Block>>, span: Span) -> Option<ValueId> {
        let cond_vid = self.lower_expr(cond);

        // Create blocks: then, else, merge
        let then_block = self.create_block();
        let else_block = self.create_block();
        let merge_block = self.create_block();

        // Emit CondBr in current block
        if let Some(cv) = cond_vid {
            self.emit_terminator(
                IrOp::CondBr { cond: cv, if_true: then_block, if_false: else_block },
                span,
                EffectFlags::TERMINATES,
            );
        }

        // Lower then block
        self.switch_to_block(then_block);
        let then_val = self.lower_block(then);
        if !self.is_terminated() {
            self.emit_terminator(
                IrOp::Jump { target: merge_block },
                then.span,
                EffectFlags::TERMINATES,
            );
        }

        // Lower else block
        self.switch_to_block(else_block);
        let else_val = else_.as_ref().and_then(|b| self.lower_block(b));
        if !self.is_terminated() {
            self.emit_terminator(
                IrOp::Jump { target: merge_block },
                else_.as_ref().map(|b| b.span).unwrap_or(span),
                EffectFlags::TERMINATES,
            );
        }

        // Merge block with Phi node
        self.switch_to_block(merge_block);

        match (then_val, else_val) {
            (Some(tv), Some(ev)) => {
                // Emit Phi node
                self.emit(
                    IrOp::Phi { incoming: vec![(then_block, tv), (else_block, ev)] },
                    span,
                    EffectFlags::pure(),
                    Ownership::Copy,
                )
            }
            _ => None,
        }
    }

    fn lower_if_stmt(&mut self, cond: &Expr, then: &Block, else_: &Option<Box<IfOrBlock>>, span: Span) {
        let cond_vid = self.lower_expr(cond);

        let then_block = self.create_block();
        let else_block = self.create_block();
        let merge_block = self.create_block();

        if let Some(cv) = cond_vid {
            self.emit_terminator(
                IrOp::CondBr { cond: cv, if_true: then_block, if_false: else_block },
                span,
                EffectFlags::TERMINATES,
            );
        }

        // Lower then
        self.switch_to_block(then_block);
        self.lower_block(then);
        if !self.is_terminated() {
            self.emit_terminator(
                IrOp::Jump { target: merge_block },
                then.span,
                EffectFlags::TERMINATES,
            );
        }

        // Lower else
        self.switch_to_block(else_block);
        if let Some(else_branch) = else_ {
            match else_branch.as_ref() {
                IfOrBlock::If(if_stmt) => {
                    if let Stmt::If { cond, then, else_, span } = if_stmt {
                        self.lower_if_stmt(cond, then, else_, *span);
                    }
                }
                IfOrBlock::Block(block) => {
                    self.lower_block(block);
                }
            }
        }
        if !self.is_terminated() {
            self.emit_terminator(
                IrOp::Jump { target: merge_block },
                span,
                EffectFlags::TERMINATES,
            );
        }

        self.switch_to_block(merge_block);
    }

    fn lower_while(&mut self, cond: &Expr, body: &Block, span: Span) {
        let header_block = self.create_block();
        let body_block = self.create_block();
        let exit_block = self.create_block();

        // Push loop context for break/continue
        self.loop_stack.push(LoopContext {
            break_block: exit_block,
            continue_block: header_block,
            loop_state_vars: vec![],
        });

        // Jump from current block to header
        self.emit_terminator(
            IrOp::Jump { target: header_block },
            span,
            EffectFlags::TERMINATES,
        );

        // Header: evaluate condition
        self.switch_to_block(header_block);
        let cond_vid = self.lower_expr(cond);
        if let Some(cv) = cond_vid {
            self.emit_terminator(
                IrOp::CondBr { cond: cv, if_true: body_block, if_false: exit_block },
                span,
                EffectFlags::TERMINATES,
            );
        }

        // Body: lower body, then jump back to header
        self.switch_to_block(body_block);
        self.lower_block(body);
        if !self.is_terminated() {
            self.emit_terminator(
                IrOp::Jump { target: header_block },
                body.span,
                EffectFlags::TERMINATES,
            );
        }

        // Continue after loop
        self.switch_to_block(exit_block);

        // Pop loop context
        self.loop_stack.pop();
    }

    fn lower_loop(&mut self, body: &Block, span: Span) {
        let header_block = self.create_block();
        let exit_block = self.create_block();

        // Push loop context for break/continue
        self.loop_stack.push(LoopContext {
            break_block: exit_block,
            continue_block: header_block,
            loop_state_vars: vec![],
        });

        self.emit_terminator(
            IrOp::Jump { target: header_block },
            span,
            EffectFlags::TERMINATES,
        );

        self.switch_to_block(header_block);
        self.lower_block(body);
        if !self.is_terminated() {
            self.emit_terminator(
                IrOp::Jump { target: header_block },
                body.span,
                EffectFlags::TERMINATES,
            );
        }

        self.switch_to_block(exit_block);

        // Pop loop context
        self.loop_stack.pop();
    }

    fn lower_for_in(&mut self, pattern: &Pattern, iter: &Expr, body: &Block, span: Span) {
        // Lower as: get iterator → loop { has_next? → body → next }
        let iter_vid = self.lower_expr(iter);

        let header_block = self.create_block();
        let body_block = self.create_block();
        let exit_block = self.create_block();

        // Push loop context for break/continue
        self.loop_stack.push(LoopContext {
            break_block: exit_block,
            continue_block: header_block,
            loop_state_vars: vec![],
        });

        // Jump to header
        self.emit_terminator(
            IrOp::Jump { target: header_block },
            span,
            EffectFlags::TERMINATES,
        );

        // Header: check has_next
        self.switch_to_block(header_block);
        let has_next = match iter_vid {
            Some(iv) => self.emit(
                IrOp::Call {
                    func: "__iter_has_next".to_string(),
                    args: vec![iv],
                },
                span,
                EffectFlags::pure(),
                Ownership::Copy,
            ),
            None => None,
        };
        if let Some(hn) = has_next {
            self.emit_terminator(
                IrOp::CondBr { cond: hn, if_true: body_block, if_false: exit_block },
                span,
                EffectFlags::TERMINATES,
            );
        } else {
            // No iterator — just exit
            self.emit_terminator(
                IrOp::Jump { target: exit_block },
                span,
                EffectFlags::TERMINATES,
            );
        }

        // Body: get next value, bind pattern, lower body
        self.switch_to_block(body_block);
        let next_val = match iter_vid {
            Some(iv) => self.emit(
                IrOp::Call {
                    func: "__iter_next".to_string(),
                    args: vec![iv],
                },
                span,
                EffectFlags::pure(),
                Ownership::OWN,
            ),
            None => None,
        };
        if let Some(nv) = next_val {
            self.bind_pattern(pattern, nv, false);
        }
        self.lower_block(body);
        if !self.is_terminated() {
            self.emit_terminator(
                IrOp::Jump { target: header_block },
                body.span,
                EffectFlags::TERMINATES,
            );
        }

        self.switch_to_block(exit_block);

        // Pop loop context — emit loop-carried state variable metadata
        // so the optimizer can reason about loop induction variables.
        if let Some(lctx) = self.loop_stack.pop() {
            for (vid, ir_ty) in &lctx.loop_state_vars {
                self.emit(
                    IrOp::Move { src: *vid },
                    span,
                    EffectFlags::pure(),
                    self.infer_ownership(ir_ty),
                );
            }
        }
    }

    fn lower_match(&mut self, expr: &Expr, arms: &[MatchArm], span: Span) {
        let scrutinee = self.lower_expr(expr);

        // Lower as a chain of CondBr blocks
        let exit_block = self.create_block();

        for arm in arms {
            let arm_block = self.create_block();
            let next_arm_block = self.create_block();

            // Emit type check for pattern (simplified — just use the arm's span)
            // For now, we emit a placeholder comparison
            if let Some(sv) = scrutinee {
                let matches = self.emit(
                    IrOp::TypeCheck {
                        value: sv,
                        expected: IrType::Unit, // placeholder
                    },
                    arm.span,
                    EffectFlags::pure(),
                    Ownership::Copy,
                );
                if let Some(mv) = matches {
                    self.emit_terminator(
                        IrOp::CondBr { cond: mv, if_true: arm_block, if_false: next_arm_block },
                        arm.span,
                        EffectFlags::TERMINATES,
                    );
                }
            }

            // Lower arm body
            self.switch_to_block(arm_block);
            self.lower_expr(&arm.body);
            if !self.is_terminated() {
                self.emit_terminator(
                    IrOp::Jump { target: exit_block },
                    arm.span,
                    EffectFlags::TERMINATES,
                );
            }

            self.switch_to_block(next_arm_block);
        }

        // If no arm matched, jump to exit
        if !self.is_terminated() {
            self.emit_terminator(
                IrOp::Jump { target: exit_block },
                span,
                EffectFlags::TERMINATES,
            );
        }

        self.switch_to_block(exit_block);
    }

    // ── Pattern Binding ─────────────────────────────────────────────────────

    fn bind_pattern(&mut self, pattern: &Pattern, vid: ValueId, mutable: bool) {
        match pattern {
            Pattern::Ident { name, .. } => {
                self.bind(name.clone(), vid);
                let _ = mutable;
            }
            Pattern::Wildcard(_) => {
                // Value is discarded
            }
            Pattern::Tuple { elems, .. } => {
                // For tuple destructuring, we'd emit extract operations
                // For now, bind each element as the same value (placeholder)
                for (i, elem) in elems.iter().enumerate() {
                    let field_vid = self.emit(
                        IrOp::Intrinsic {
                            name: format!("__tuple_get_{}", i),
                            args: vec![vid],
                        },
                        pattern.span(),
                        EffectFlags::pure(),
                        Ownership::Copy,
                    );
                    if let Some(fv) = field_vid {
                        self.bind_pattern(elem, fv, mutable);
                    }
                }
            }
            Pattern::Struct { fields, .. } => {
                for (field_name, sub_pattern) in fields {
                    let field_vid = self.emit(
                        IrOp::Intrinsic {
                            name: format!("__field_{}", field_name),
                            args: vec![vid],
                        },
                        pattern.span(),
                        EffectFlags::pure(),
                        Ownership::Copy,
                    );
                    if let Some(fv) = field_vid {
                        if let Some(sp) = sub_pattern {
                            self.bind_pattern(sp, fv, mutable);
                        }
                    }
                }
            }
            _ => {
                // Other patterns — just bind as-is (placeholder)
            }
        }
    }

    // ── Type Lowering ───────────────────────────────────────────────────────

    fn lower_type(&self, ty: &Type) -> IrType {
        match ty {
            Type::Scalar(elem) => self.elem_type_to_ir(elem.clone()),

            Type::Tensor { elem, shape } => {
                IrType::Tensor {
                    elem: Box::new(self.elem_type_to_ir(elem.clone())),
                    shape: shape.iter().map(|d| self.lower_dim_to_u64(d)).collect(),
                }
            }

            Type::Vec { size, family } => {
                let elem = match family {
                    VecFamily::Float => IrType::Float { width: 32 },
                    VecFamily::Int => IrType::Int { width: 32, signed: true },
                    VecFamily::UInt => IrType::Int { width: 32, signed: false },
                };
                let len = size.lanes() as u64;
                IrType::Array { elem: Box::new(elem), len: Some(len) }
            }

            Type::Mat { size } => {
                let lanes = (size.lanes() * size.lanes()) as u64;
                IrType::Array { elem: Box::new(IrType::Float { width: 32 }), len: Some(lanes) }
            }

            Type::Quat => {
                IrType::Array { elem: Box::new(IrType::Float { width: 32 }), len: Some(4) }
            }

            Type::Named(name) => IrType::Unknown, // opaque named type

            Type::Tuple(elems) => {
                IrType::Struct {
                    name: "__tuple".to_string(),
                    fields: elems.iter().enumerate()
                        .map(|(i, t)| (format!("_{}", i), self.lower_type(t)))
                        .collect(),
                }
            }

            Type::Array { elem, len: _ } => {
                IrType::Array { elem: Box::new(self.lower_type(elem)), len: None }
            }

            Type::Slice(elem) => {
                IrType::Slice(Box::new(self.lower_type(elem)))
            }

            Type::Ref { mutable: false, inner } => {
                IrType::Ref(Box::new(self.lower_type(inner)))
            }

            Type::Ref { mutable: true, inner } => {
                IrType::MutRef(Box::new(self.lower_type(inner)))
            }

            Type::Option(inner) => {
                IrType::Enum {
                    name: format!("Option_{}", self.type_summary(inner)),
                    variants: vec![
                        ("None".to_string(), vec![]),
                        ("Some".to_string(), vec![self.lower_type(inner)]),
                    ],
                }
            }

            Type::Result { ok, err } => {
                IrType::Enum {
                    name: format!("Result_{}_{}", self.type_summary(ok), self.type_summary(err)),
                    variants: vec![
                        ("Ok".to_string(), vec![self.lower_type(ok)]),
                        ("Err".to_string(), vec![self.lower_type(err)]),
                    ],
                }
            }

            Type::FnPtr { params, ret } => {
                IrType::FnPtr {
                    params: params.iter().map(|t| self.lower_type(t)).collect(),
                    ret: Box::new(self.lower_type(ret)),
                }
            }

            Type::Never => IrType::Never,

            Type::Infer => IrType::Unit, // inference placeholder

            Type::Own(inner) => {
                // Owned value — just the value itself in the flat IR
                self.lower_type(inner)
            }

            Type::Unique(inner) => {
                // Unique reference — immutable in the flat IR
                IrType::Ref(Box::new(self.lower_type(inner)))
            }

            Type::Shared(inner) => {
                IrType::Ref(Box::new(self.lower_type(inner)))
            }
        }
    }

    /// Lower a DimExpr to a u64 shape dimension for the flat IrType::Tensor.
    fn lower_dim_to_u64(&self, dim: &DimExpr) -> u64 {
        match dim {
            DimExpr::Lit(n) => *n,
            DimExpr::Named(_) | DimExpr::Dynamic | DimExpr::Expr(_) => 0, // 0 = dynamic/unknown dimension
        }
    }

    fn elem_type_to_ir(&self, elem: ElemType) -> IrType {
        match elem {
            ElemType::I8   => IrType::Int { width: 8,  signed: true },
            ElemType::I16  => IrType::Int { width: 16, signed: true },
            ElemType::I32  => IrType::Int { width: 32, signed: true },
            ElemType::I64  => IrType::Int { width: 64, signed: true },
            ElemType::U8   => IrType::Int { width: 8,  signed: false },
            ElemType::U16  => IrType::Int { width: 16, signed: false },
            ElemType::U32  => IrType::Int { width: 32, signed: false },
            ElemType::U64  => IrType::Int { width: 64, signed: false },
            ElemType::F16  => IrType::Float { width: 16 },
            ElemType::F32  => IrType::Float { width: 32 },
            ElemType::F64  => IrType::Float { width: 64 },
            ElemType::Bf16 => IrType::Float { width: 16 },
            ElemType::Bool => IrType::Bool,
            ElemType::Usize => IrType::Int { width: 64, signed: false },
        }
    }

    fn type_summary(&self, ty: &Type) -> String {
        match ty {
            Type::Scalar(e) => format!("{:?}", e).to_lowercase(),
            Type::Named(n) => n.clone(),
            Type::Tuple(elems) => {
                let parts: Vec<String> = elems.iter().map(|t| self.type_summary(t)).collect();
                parts.join("_")
            }
            _ => "unknown".to_string(),
        }
    }

    // ── Effect Inference ────────────────────────────────────────────────────

    fn infer_effects(&self, expr: &Expr) -> EffectFlags {
        match expr {
            Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::BoolLit { .. } | Expr::StrLit { .. } => {
                EffectFlags::pure()
            }
            Expr::Ident { .. } | Expr::Path { .. } => EffectFlags::pure(),
            Expr::BinOp { .. } | Expr::UnOp { .. } => EffectFlags::pure(),
            Expr::Assign { .. } => EffectFlags::WRITE,
            Expr::Call { .. } | Expr::MethodCall { .. } => {
                // Conservative: calls may have any effect
                EffectFlags::PURE.union(EffectFlags::READONLY).union(EffectFlags::WRITE).union(EffectFlags::IO)
            }
            Expr::Emit { .. } => EffectFlags::IO,
            Expr::Copy { .. } => EffectFlags::pure(),
            Expr::Field { .. } | Expr::Index { .. } => EffectFlags::pure().union(EffectFlags::READONLY),
            Expr::IfExpr { cond, then, else_, .. } => {
                let mut effects = self.infer_effects(cond);
                effects = effects.union(self.infer_block_effects(then));
                if let Some(e) = else_ {
                    effects = effects.union(self.infer_block_effects(e));
                }
                effects
            }
            Expr::Pipeline { stages, .. } => {
                let mut effects = EffectFlags::pure();
                for stage in stages {
                    effects = effects.union(self.infer_effects(stage));
                }
                effects
            }
            _ => EffectFlags::pure(),
        }
    }

    fn infer_block_effects(&self, block: &Block) -> EffectFlags {
        let mut effects = EffectFlags::pure();
        for stmt in &block.stmts {
            effects = effects.union(self.infer_stmt_effects(stmt));
        }
        if let Some(tail) = &block.tail {
            effects = effects.union(self.infer_effects(tail));
        }
        effects
    }

    fn infer_stmt_effects(&self, stmt: &Stmt) -> EffectFlags {
        match stmt {
            Stmt::Let { init, .. } => {
                init.as_ref().map_or(EffectFlags::pure(), |e| self.infer_effects(e))
            }
            Stmt::Expr { expr, .. } => self.infer_effects(expr),
            Stmt::Return { .. } => EffectFlags::TERMINATES,
            Stmt::If { cond, then, else_, .. } => {
                let mut e = self.infer_effects(cond);
                e = e.union(self.infer_block_effects(then));
                if let Some(el) = else_ {
                    match el.as_ref() {
                        IfOrBlock::If(s) => e = e.union(self.infer_stmt_effects(s)),
                        IfOrBlock::Block(b) => e = e.union(self.infer_block_effects(b)),
                    }
                }
                e
            }
            Stmt::While { cond, body, .. } => {
                self.infer_effects(cond).union(self.infer_block_effects(body))
            }
            Stmt::ForIn { iter, body, .. } => {
                self.infer_effects(iter).union(self.infer_block_effects(body))
            }
            Stmt::Effect { .. } => EffectFlags::IO,
            Stmt::Region { .. } => EffectFlags::ALLOC,
            Stmt::TaskSpawn { .. } => EffectFlags::PARALLEL,
            Stmt::TaskJoin { .. } => EffectFlags::PARALLEL,
            Stmt::UnsafeBlock { .. } => EffectFlags::WRITE,
            _ => EffectFlags::pure(),
        }
    }

    // ── Ownership Inference ─────────────────────────────────────────────────

    fn infer_ownership(&self, ty: &IrType) -> Ownership {
        match ty {
            IrType::Int { .. } | IrType::Float { .. } | IrType::Bool | IrType::Unit => Ownership::Copy,
            IrType::String => Ownership::OWN,
            IrType::Tensor { .. } => Ownership::OWN,
            IrType::Array { .. } => Ownership::OWN,
            IrType::Slice(_) => Ownership::Shared,
            IrType::Ref(_) => Ownership::Shared,
            IrType::MutRef(_) => Ownership::MUT_BORROW,
            IrType::FnPtr { .. } => Ownership::Copy,
            IrType::Struct { .. } => Ownership::OWN,
            IrType::Enum { .. } => Ownership::Copy,
            IrType::Never => Ownership::Copy,
            IrType::Unknown => Ownership::OWN,
        }
    }
}

// =============================================================================
// IR Display / Debug (for emit_ir flag)
// =============================================================================

impl fmt::Display for FlatIrModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "; === Jules Flat IR Module ===")?;
        for func in &self.functions {
            writeln!(f, "\n{} {{", func.name)?;
            writeln!(f, "  entry: {}", func.entry)?;
            for (vid, ty) in &func.params {
                writeln!(f, "  param {} : {}", vid, ty)?;
            }
            writeln!(f, "  returns {}", func.ret_ty)?;
            writeln!(f, "  effects: {}", func.effects)?;
            for block in &func.blocks {
                writeln!(f, "\n{}:", block.id)?;
                for instr in &block.instrs {
                    if let Some(dst) = instr.dst {
                        write!(f, "    {} = ", dst)?;
                    } else {
                        write!(f, "    ")?;
                    }
                    write!(f, "{:?}", instr.op)?;
                    if !instr.effects.is_pure() {
                        write!(f, "  ; effects={}", instr.effects)?;
                    }
                    if instr.ownership != Ownership::Copy {
                        write!(f, "  ; own={:?}", instr.ownership)?;
                    }
                    writeln!(f)?;
                }
            }
            writeln!(f, "}}")?;
        }
        for intr in &self.intrinsics {
            writeln!(f, "\nintrinsic {}({:?}) -> {}  ; effects={}",
                intr.name, intr.param_types, intr.ret_type, intr.effects)?;
        }
        Ok(())
    }
}
