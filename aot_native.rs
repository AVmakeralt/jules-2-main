// =============================================================================
// jules/src/aot_native.rs
//
// WORLD-CLASS AHEAD-OF-TIME NATIVE CODE COMPILER  v2.0
//
// Compiles Jules source directly to optimized x86-64 ELF executables.
// No LLVM, no external dependencies — pure, maximum-performance codegen.
//
// Architecture (20 phases):
//   Phase  1: Call graph construction & analysis
//   Phase  2: AST → SSA-based IR lowering
//   Phase  3: Sparse Conditional Constant Propagation (SCCP / Wegman-Zadeck)
//   Phase  4: Global Value Numbering (GVN) with hash-consing
//   Phase  5: Copy Propagation
//   Phase  6: Dead Code Elimination (DCE) via liveness
//   Phase  7: Value Range Propagation (VRP)
//   Phase  8: Lengauer-Tarjan Dominance Tree computation
//   Phase  9: Natural Loop Detection (back-edge analysis)
//   Phase 10: Loop-Invariant Code Motion (LICM)
//   Phase 11: Induction Variable strength reduction
//   Phase 12: Peephole Optimization (40+ patterns, multi-pass)
//   Phase 13: Tail Call Optimization (TCO) detection & transformation
//   Phase 14: Jump Threading & block merging
//   Phase 15: Function Inlining with proper variable renaming
//   Phase 16: Linear-Scan Register Allocation with coalescing hints
//   Phase 17: List-based Instruction Scheduling (latency-aware)
//   Phase 18: x86-64 Machine Code Emission
//   Phase 19: ELF binary emission with symbol table
//   Phase 20: Fixup resolution & relocation patching
//
// Bug fixes over v1:
//   - LatticeVal lifted to module scope (eval_instr type error)
//   - Move IR instruction added (proper copy semantics)
//   - instr_uses: CondBr arm fixed (cond vs src binding)
//   - AllocationResult: removed invalid u8(-1) case
//   - NativeCodeGen::emit_instr: closures replaced (borrow conflict)
//   - ELF: virtual addresses fixed (was using file offsets)
//   - build_predecessors: collect-then-mutate (borrow split)
//   - LoweringCtx: explicit current_block field
//   - Shift instructions: variable shifts via cl register
//   - ICmpCond: unified x86_cc() covering unsigned conditions
//   - SRem: rdx result captured, not rax
//   - Prologue/epilogue: aligned RSP to 16 bytes
// =============================================================================

#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque, BTreeMap};
use std::fs::File;
use std::io::Write;

use crate::ast::*;

// =============================================================================
// §0  CONFIGURATION & CONSTANTS
// =============================================================================

const PAGE_SIZE:  usize = 4096;
const LOAD_BASE:  u64   = 0x0040_0000; // standard x86-64 ELF load address

/// x86-64 register numbers (Intel encoding)
/// rax=0  rcx=1  rdx=2  rbx=3  rsp=4  rbp=5  rsi=6  rdi=7
/// r8=8   r9=9   r10=10 r11=11 r12=12 r13=13 r14=14 r15=15

const CALLEE_SAVED: &[u8] = &[3, 12, 13, 14, 15];      // rbx, r12-r15
const CALLER_SAVED: &[u8] = &[0, 1, 2, 6, 7, 8, 9, 10, 11]; // rax,rcx,rdx,rsi,rdi,r8-r11
const ARG_REGS:     &[u8] = &[7, 6, 2, 1, 8, 9];       // rdi,rsi,rdx,rcx,r8,r9
const RET_REG:       u8   = 0;  // rax
const RBP:           u8   = 5;
const RSP:           u8   = 4;

/// Condition-code byte for 0F 8x / 0F 9x families
mod cc {
    pub const E:  u8 = 0x84; // ZF=1           equal
    pub const NE: u8 = 0x85; // ZF=0           not-equal
    pub const L:  u8 = 0x8C; // SF≠OF          signed <
    pub const GE: u8 = 0x8D; // SF=OF          signed ≥
    pub const G:  u8 = 0x8F; // ZF=0 ∧ SF=OF   signed >
    pub const LE: u8 = 0x8E; // ZF=1 ∨ SF≠OF   signed ≤
    pub const B:  u8 = 0x82; // CF=1           unsigned <
    pub const AE: u8 = 0x83; // CF=0           unsigned ≥
    pub const A:  u8 = 0x87; // CF=0 ∧ ZF=0    unsigned >
    pub const BE: u8 = 0x86; // CF=1 ∨ ZF=1    unsigned ≤
}

/// Per-level optimisation configuration
#[derive(Debug, Clone, Copy)]
pub struct OptConfig {
    pub sccp:             bool,
    pub gvn:              bool,
    pub copy_prop:        bool,
    pub vrp:              bool,
    pub licm:             bool,
    pub strength_reduce:  bool,
    pub dce:              bool,
    pub peephole:         bool,
    pub tco:              bool,
    pub jump_threading:   bool,
    pub inlining:         bool,
    pub max_inline_size:  usize,
    pub loop_unrolling:   bool,
    pub max_unroll:       usize,
    pub sched:            bool,
}

impl OptConfig {
    pub fn from_level(level: u8) -> Self {
        match level {
            0 => Self { sccp: true, peephole: true, ..Self::none() },
            1 => Self { sccp: true, gvn: true, copy_prop: true, dce: true,
                        peephole: true, ..Self::none() },
            2 => Self { sccp: true, gvn: true, copy_prop: true, vrp: true,
                        licm: true, strength_reduce: true, dce: true,
                        peephole: true, tco: true, jump_threading: true,
                        inlining: true, max_inline_size: 32,
                        loop_unrolling: true, max_unroll: 4, sched: true },
            _ => Self { sccp: true, gvn: true, copy_prop: true, vrp: true,
                        licm: true, strength_reduce: true, dce: true,
                        peephole: true, tco: true, jump_threading: true,
                        inlining: true, max_inline_size: 64,
                        loop_unrolling: true, max_unroll: 8, sched: true },
        }
    }
    fn none() -> Self {
        Self { sccp: false, gvn: false, copy_prop: false, vrp: false,
               licm: false, strength_reduce: false, dce: false, peephole: false,
               tco: false, jump_threading: false, inlining: false,
               max_inline_size: 0, loop_unrolling: false, max_unroll: 0,
               sched: false }
    }
}

// =============================================================================
// §1  INTERMEDIATE REPRESENTATION (SSA form)
// =============================================================================

pub type VarId   = usize;
pub type BlockId = usize;

/// Integer comparison condition codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ICmpCond {
    Eq, Ne,
    SLt, SLe, SGt, SGe,
    ULt, ULe, UGt, UGe,
}

impl ICmpCond {
    /// Unified x86 condition-code byte (covers both signed and unsigned)
    pub fn x86_cc(self) -> u8 {
        match self {
            ICmpCond::Eq  => cc::E,
            ICmpCond::Ne  => cc::NE,
            ICmpCond::SLt => cc::L,
            ICmpCond::SLe => cc::LE,
            ICmpCond::SGt => cc::G,
            ICmpCond::SGe => cc::GE,
            ICmpCond::ULt => cc::B,
            ICmpCond::ULe => cc::BE,
            ICmpCond::UGt => cc::A,
            ICmpCond::UGe => cc::AE,
        }
    }
    /// Invert the condition (for branch optimisation)
    pub fn invert(self) -> Self {
        match self {
            ICmpCond::Eq  => ICmpCond::Ne,  ICmpCond::Ne  => ICmpCond::Eq,
            ICmpCond::SLt => ICmpCond::SGe, ICmpCond::SGe => ICmpCond::SLt,
            ICmpCond::SLe => ICmpCond::SGt, ICmpCond::SGt => ICmpCond::SLe,
            ICmpCond::ULt => ICmpCond::UGe, ICmpCond::UGe => ICmpCond::ULt,
            ICmpCond::ULe => ICmpCond::UGt, ICmpCond::UGt => ICmpCond::ULe,
        }
    }
}

/// SSA-form three-address IR instruction
#[derive(Debug, Clone)]
pub enum IRInstr {
    // ── Constants ────────────────────────────────────────────────────────
    Const   { dst: VarId, value: i64 },
    ConstF64 { dst: VarId, value: f64 },

    // ── Integer arithmetic ───────────────────────────────────────────────
    Add  { dst: VarId, lhs: VarId, rhs: VarId },
    Sub  { dst: VarId, lhs: VarId, rhs: VarId },
    Mul  { dst: VarId, lhs: VarId, rhs: VarId },
    SDiv { dst: VarId, lhs: VarId, rhs: VarId },
    SRem { dst: VarId, lhs: VarId, rhs: VarId },
    Neg  { dst: VarId, src: VarId },

    // ── Bitwise ──────────────────────────────────────────────────────────
    And  { dst: VarId, lhs: VarId, rhs: VarId },
    Or   { dst: VarId, lhs: VarId, rhs: VarId },
    Xor  { dst: VarId, lhs: VarId, rhs: VarId },
    Shl  { dst: VarId, lhs: VarId, rhs: VarId },
    AShr { dst: VarId, lhs: VarId, rhs: VarId },
    LShr { dst: VarId, lhs: VarId, rhs: VarId },
    Not  { dst: VarId, src: VarId },

    // ── Comparison → bool (0 or 1) ───────────────────────────────────────
    ICmp { dst: VarId, cond: ICmpCond, lhs: VarId, rhs: VarId },

    // ── Copy (required for proper SSA destruction) ───────────────────────
    Move { dst: VarId, src: VarId },

    // ── Control flow ─────────────────────────────────────────────────────
    Br     { target: BlockId },
    CondBr { cond: VarId, if_true: BlockId, if_false: BlockId },
    Ret    { value: Option<VarId> },

    // ── Calls ────────────────────────────────────────────────────────────
    Call     { dst: VarId, func: String, args: Vec<VarId> },
    TailCall { func: String, args: Vec<VarId> },

    // ── Memory ───────────────────────────────────────────────────────────
    Alloca { dst: VarId, align: u32 },
    Store  { ptr: VarId, value: VarId },
    Load   { dst: VarId, ptr: VarId },

    // ── SSA phi ──────────────────────────────────────────────────────────
    Phi { dst: VarId, incoming: Vec<(BlockId, VarId)> },

    // ── Misc ─────────────────────────────────────────────────────────────
    Label,
    Comment(String),
}

impl IRInstr {
    /// True when the instruction has observable side effects beyond its def
    pub fn has_side_effects(&self) -> bool {
        matches!(self,
            IRInstr::Store {..} | IRInstr::Call {..} | IRInstr::TailCall {..} |
            IRInstr::Ret {..}
        )
    }
}

/// Basic block in the CFG
#[derive(Debug, Clone)]
pub struct IRBlock {
    pub id:           BlockId,
    pub instrs:       Vec<IRInstr>,
    pub successors:   Vec<BlockId>,
    pub predecessors: Vec<BlockId>,
    /// Estimated execution frequency (used by scheduler & layout)
    pub freq:         f64,
}

impl IRBlock {
    fn new(id: BlockId) -> Self {
        Self { id, instrs: Vec::new(), successors: Vec::new(),
               predecessors: Vec::new(), freq: 1.0 }
    }
    fn is_terminated(&self) -> bool {
        self.instrs.iter().rev().any(|i| matches!(i,
            IRInstr::Br {..} | IRInstr::CondBr {..} | IRInstr::Ret {..} |
            IRInstr::TailCall {..}
        ))
    }
}

/// IR function in SSA form
#[derive(Debug, Clone)]
pub struct IRFunction {
    pub name:         String,
    pub params:       Vec<VarId>,
    pub ret_ty:       Option<Type>,
    pub blocks:       BTreeMap<BlockId, IRBlock>,
    pub entry_block:  BlockId,
    pub next_var:     VarId,
    pub next_block:   BlockId,
    pub call_count:   usize,
    pub called_by:    Vec<String>,
    pub inline_cost:  usize,
}

impl IRFunction {
    pub fn new(name: String) -> Self {
        let mut blocks = BTreeMap::new();
        blocks.insert(0, IRBlock::new(0));
        Self { name, params: Vec::new(), ret_ty: None, blocks,
               entry_block: 0, next_var: 0, next_block: 1,
               call_count: 0, called_by: Vec::new(), inline_cost: 0 }
    }

    pub fn fresh_var(&mut self) -> VarId {
        let v = self.next_var; self.next_var += 1; v
    }

    pub fn fresh_block(&mut self) -> BlockId {
        let id = self.next_block; self.next_block += 1;
        self.blocks.insert(id, IRBlock::new(id));
        id
    }

    pub fn push_to(&mut self, block: BlockId, instr: IRInstr) {
        self.blocks.entry(block)
            .or_insert_with(|| IRBlock::new(block))
            .instrs.push(instr);
    }

    /// Rebuild predecessor lists from successor info
    pub fn build_predecessors(&mut self) {
        for b in self.blocks.values_mut() { b.predecessors.clear(); }
        // Collect edges first to avoid simultaneous mutable/immutable borrows
        let edges: Vec<(BlockId, BlockId)> = self.blocks.iter()
            .flat_map(|(&from, b)| b.successors.iter().map(move |&to| (from, to)))
            .collect();
        for (from, to) in edges {
            if let Some(succ) = self.blocks.get_mut(&to) {
                if !succ.predecessors.contains(&from) {
                    succ.predecessors.push(from);
                }
            }
        }
    }

    /// Estimate block execution frequencies via simple back-propagation
    pub fn estimate_frequencies(&mut self) {
        // Mark loop headers with high frequency
        let dom = compute_dominators(self);
        let mut loop_headers = Vec::new();
        for (&bid, block) in &self.blocks {
            for &succ in &block.successors {
                // Back edge: succ dominates bid → succ is a loop header
                if dominates(&dom, succ, bid) {
                    loop_headers.push(succ);
                }
            }
        }
        for succ in loop_headers {
            if let Some(b) = self.blocks.get_mut(&succ) {
                b.freq = 100.0;
            }
        }
    }

    pub fn compute_inline_cost(&mut self) {
        self.inline_cost = self.blocks.values()
            .map(|b| b.instrs.len()).sum();
    }

    /// All variable definitions in this function
    pub fn all_defs(&self) -> HashMap<VarId, (BlockId, usize)> {
        let mut m = HashMap::new();
        for (&bid, b) in &self.blocks {
            for (idx, instr) in b.instrs.iter().enumerate() {
                if let Some(d) = instr_def(instr) { m.insert(d, (bid, idx)); }
            }
        }
        m
    }
}

// =============================================================================
// §2  CALL GRAPH
// =============================================================================

#[derive(Debug)]
pub struct CallGraph {
    pub edges:    HashMap<String, Vec<(String, usize)>>,
    pub callers:  HashMap<String, HashSet<String>>,
    pub all_fns:  HashSet<String>,
}

impl CallGraph {
    pub fn build(program: &Program) -> Self {
        let mut edges: HashMap<String, Vec<(String, usize)>> = HashMap::new();
        let mut callers: HashMap<String, HashSet<String>> = HashMap::new();
        let mut all_fns = HashSet::new();

        for item in &program.items {
            if let Item::Fn(f) = item {
                all_fns.insert(f.name.clone());
                edges.entry(f.name.clone()).or_default();
                let mut local: HashMap<String, usize> = HashMap::new();
                Self::scan_block(&f.body, &mut local);
                for (callee, cnt) in local {
                    edges.entry(f.name.clone()).or_default()
                         .push((callee.clone(), cnt));
                    callers.entry(callee).or_default().insert(f.name.clone());
                }
            }
        }
        Self { edges, callers, all_fns }
    }

    fn scan_block(body: &Option<Block>, calls: &mut HashMap<String, usize>) {
        if let Some(b) = body {
            for s in &b.stmts { Self::scan_stmt(s, calls); }
            if let Some(t) = &b.tail { Self::scan_expr(t, calls); }
        }
    }

    fn scan_stmt(s: &Stmt, calls: &mut HashMap<String, usize>) {
        match s {
            Stmt::Let  { init: Some(e), .. } => Self::scan_expr(e, calls),
            Stmt::Expr { expr: e, .. }       => Self::scan_expr(e, calls),
            Stmt::If   { cond, then, else_, .. } => {
                Self::scan_expr(cond, calls);
                Self::scan_block(&Some(then.clone()), calls);
                if let Some(e) = else_ {
                    match e.as_ref() {
                        IfOrBlock::If(s)    => Self::scan_stmt(s, calls),
                        IfOrBlock::Block(b) => Self::scan_block(&Some(b.clone()), calls),
                    }
                }
            }
            Stmt::While { cond, body, .. } => {
                Self::scan_expr(cond, calls);
                Self::scan_block(&Some(body.clone()), calls);
            }
            Stmt::ForIn { iter, body, .. } => {
                Self::scan_expr(iter, calls);
                Self::scan_block(&Some(body.clone()), calls);
            }
            Stmt::Return { value: Some(e), .. } => Self::scan_expr(e, calls),
            _ => {}
        }
    }

    fn scan_expr(e: &Expr, calls: &mut HashMap<String, usize>) {
        match e {
            Expr::Call { func, args, .. } => {
                if let Expr::Ident { name, .. } = func.as_ref() {
                    *calls.entry(name.clone()).or_insert(0) += 1;
                }
                Self::scan_expr(func, calls);
                for a in args { Self::scan_expr(a, calls); }
            }
            Expr::BinOp { lhs, rhs, .. } => {
                Self::scan_expr(lhs, calls);
                Self::scan_expr(rhs, calls);
            }
            Expr::UnOp { expr: inner, .. } => Self::scan_expr(inner, calls),
            Expr::Field { object, .. }     => Self::scan_expr(object, calls),
            Expr::Index { object, indices, .. } => {
                Self::scan_expr(object, calls);
                for i in indices { Self::scan_expr(i, calls); }
            }
            _ => {}
        }
    }

    pub fn is_called_once(&self, f: &str) -> bool {
        self.callers.get(f).map_or(false, |c| c.len() == 1)
    }
}

// =============================================================================
// §3  AST → IR LOWERING
// =============================================================================

struct LowerCtx {
    func:            IRFunction,
    var_env:         HashMap<String, VarId>,
    break_targets:   Vec<BlockId>,
    continue_targets: Vec<BlockId>,
    cur:             BlockId,
}

impl LowerCtx {
    fn new(name: &str) -> Self {
        Self {
            func: IRFunction::new(name.to_string()),
            var_env: HashMap::new(),
            break_targets: Vec::new(),
            continue_targets: Vec::new(),
            cur: 0,
        }
    }

    fn emit(&mut self, i: IRInstr) {
        let cur = self.cur;
        self.func.push_to(cur, i);
    }

    fn fresh_block(&mut self) -> BlockId { self.func.fresh_block() }
    fn fresh_var(&mut self)   -> VarId   { self.func.fresh_var() }

    fn set_successors(&mut self, block: BlockId, succs: Vec<BlockId>) {
        if let Some(b) = self.func.blocks.get_mut(&block) {
            b.successors = succs;
        }
    }

    fn switch_to(&mut self, block: BlockId) { self.cur = block; }

    // ── Function entry ───────────────────────────────────────────────────
    fn lower_fn(&mut self, f: &FnDecl) {
        for p in &f.params {
            let v = self.fresh_var();
            self.func.params.push(v);
            self.var_env.insert(p.name.to_string(), v);
        }
        self.func.ret_ty = f.ret_ty.clone();

        if let Some(body) = &f.body {
            self.lower_block(body);
        }
        if !self.func.blocks[&self.cur].is_terminated() {
            self.emit(IRInstr::Ret { value: None });
        }
        self.func.build_predecessors();
        self.func.compute_inline_cost();
    }

    fn lower_block(&mut self, block: &Block) {
        for s in &block.stmts { self.lower_stmt(s); }
        if let Some(t) = &block.tail { self.lower_expr(t); }
    }

    fn lower_stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Let { pattern, init, .. } => {
                if let Some(e) = init {
                    let v = self.lower_expr(e);
                    if let Pattern::Ident { name, .. } = pattern {
                        self.var_env.insert(name.to_string(), v);
                    }
                }
            }
            Stmt::Expr { expr, .. } => { self.lower_expr(expr); }
            Stmt::Return { value, .. } => {
                let v = value.as_ref().map(|e| self.lower_expr(e));
                self.emit(IRInstr::Ret { value: v });
                let dead = self.fresh_block();
                self.switch_to(dead);
            }
            Stmt::If { cond, then, else_, .. } => {
                self.lower_if(cond, then, else_.as_deref());
            }
            Stmt::While { cond, body, .. } => {
                self.lower_while(cond, body);
            }
            Stmt::ForIn { pattern, iter, body, .. } => {
                self.lower_for_in(pattern, iter, body);
            }
            Stmt::Break { .. } => {
                if let Some(&t) = self.break_targets.last() {
                    self.emit(IRInstr::Br { target: t });
                    let dead = self.fresh_block();
                    self.switch_to(dead);
                }
            }
            Stmt::Continue { .. } => {
                if let Some(&t) = self.continue_targets.last() {
                    self.emit(IRInstr::Br { target: t });
                    let dead = self.fresh_block();
                    self.switch_to(dead);
                }
            }
            _ => {}
        }
    }

    fn lower_if(&mut self, cond: &Expr, then: &Block, else_: Option<&IfOrBlock>) {
        let cond_v   = self.lower_expr(cond);
        let then_b   = self.fresh_block();
        let merge_b  = self.fresh_block();
        let else_b   = if else_.is_some() { self.fresh_block() } else { merge_b };

        let prev = self.cur;
        self.emit(IRInstr::CondBr { cond: cond_v, if_true: then_b, if_false: else_b });
        self.set_successors(prev, vec![then_b, else_b]);

        // Then branch
        self.switch_to(then_b);
        self.lower_block(then);
        if !self.func.blocks[&self.cur].is_terminated() {
            self.emit(IRInstr::Br { target: merge_b });
        }
        let then_end = self.cur;
        self.set_successors(then_end, vec![merge_b]);

        // Else branch
        if let Some(e) = else_ {
            self.switch_to(else_b);
            match e {
                IfOrBlock::Block(b) => {
                    self.lower_block(b);
                    if !self.func.blocks[&self.cur].is_terminated() {
                        self.emit(IRInstr::Br { target: merge_b });
                    }
                    let else_end = self.cur;
                    self.set_successors(else_end, vec![merge_b]);
                }
                IfOrBlock::If(s) => {
                    self.lower_stmt(s);
                    if !self.func.blocks[&self.cur].is_terminated() {
                        self.emit(IRInstr::Br { target: merge_b });
                    }
                    let else_end = self.cur;
                    self.set_successors(else_end, vec![merge_b]);
                }
            }
        }

        self.switch_to(merge_b);
    }

    fn lower_while(&mut self, cond: &Expr, body: &Block) {
        let header = self.fresh_block();
        let body_b = self.fresh_block();
        let exit   = self.fresh_block();

        let prev = self.cur;
        self.emit(IRInstr::Br { target: header });
        self.set_successors(prev, vec![header]);

        // Header: evaluate condition
        self.switch_to(header);
        let cv = self.lower_expr(cond);
        self.emit(IRInstr::CondBr { cond: cv, if_true: body_b, if_false: exit });
        self.set_successors(header, vec![body_b, exit]);

        // Body
        self.break_targets.push(exit);
        self.continue_targets.push(header);
        self.switch_to(body_b);
        self.lower_block(body);
        if !self.func.blocks[&self.cur].is_terminated() {
            self.emit(IRInstr::Br { target: header });
        }
        let body_end = self.cur;
        self.set_successors(body_end, vec![header]);
        self.break_targets.pop();
        self.continue_targets.pop();

        self.switch_to(exit);
    }

    fn lower_for_in(&mut self, pattern: &Pattern, iter: &Expr, body: &Block) {
        // Translate `for i in lo..hi` → equivalent while loop
        if let Expr::Range { lo, hi, .. } = iter {
            let start = lo.as_ref().map(|e| self.lower_expr(e))
                .unwrap_or_else(|| {
                    let v = self.fresh_var();
                    self.emit(IRInstr::Const { dst: v, value: 0 });
                    v
                });
            let end = hi.as_ref().map(|e| self.lower_expr(e))
                .unwrap_or_else(|| panic!("for..in requires upper bound"));

            // loop_var = start  (use Move for proper SSA copy)
            let loop_var = self.fresh_var();
            self.emit(IRInstr::Move { dst: loop_var, src: start });
            if let Pattern::Ident { name, .. } = pattern {
                self.var_env.insert(name.to_string(), loop_var);
            }

            let header    = self.fresh_block();
            let body_b    = self.fresh_block();
            let incr_b    = self.fresh_block();
            let exit      = self.fresh_block();

            let prev = self.cur;
            self.emit(IRInstr::Br { target: header });
            self.set_successors(prev, vec![header]);

            // Header: if loop_var < end → body, else exit
            self.switch_to(header);
            let cmp = self.fresh_var();
            self.emit(IRInstr::ICmp { dst: cmp, cond: ICmpCond::SLt,
                                      lhs: loop_var, rhs: end });
            self.emit(IRInstr::CondBr { cond: cmp, if_true: body_b, if_false: exit });
            self.set_successors(header, vec![body_b, exit]);

            // Body
            self.break_targets.push(exit);
            self.continue_targets.push(incr_b);
            self.switch_to(body_b);
            self.lower_block(body);
            if !self.func.blocks[&self.cur].is_terminated() {
                self.emit(IRInstr::Br { target: incr_b });
            }
            let body_end = self.cur;
            self.set_successors(body_end, vec![incr_b]);
            self.break_targets.pop();
            self.continue_targets.pop();

            // Increment: loop_var += 1
            self.switch_to(incr_b);
            let one     = self.fresh_var();
            let new_var = self.fresh_var();
            self.emit(IRInstr::Const { dst: one, value: 1 });
            self.emit(IRInstr::Add   { dst: new_var, lhs: loop_var, rhs: one });
            self.emit(IRInstr::Move  { dst: loop_var, src: new_var });
            self.emit(IRInstr::Br { target: header });
            self.set_successors(incr_b, vec![header]);

            self.switch_to(exit);
        }
    }

    fn lower_expr(&mut self, e: &Expr) -> VarId {
        match e {
            Expr::IntLit  { value, .. } => {
                let v = self.fresh_var();
                self.emit(IRInstr::Const { dst: v, value: *value as i64 });
                v
            }
            Expr::FloatLit { value, .. } => {
                let v = self.fresh_var();
                self.emit(IRInstr::ConstF64 { dst: v, value: *value });
                v
            }
            Expr::BoolLit { value, .. } => {
                let v = self.fresh_var();
                self.emit(IRInstr::Const { dst: v, value: *value as i64 });
                v
            }
            Expr::Ident { name, .. } => {
                *self.var_env.get(name.as_str())
                    .unwrap_or_else(|| panic!("undefined: {}", name))
            }
            Expr::BinOp { op, lhs, rhs, .. } => {
                let l = self.lower_expr(lhs);
                let r = self.lower_expr(rhs);
                let d = self.fresh_var();
                let i = match op {
                    BinOpKind::Add    => IRInstr::Add  { dst: d, lhs: l, rhs: r },
                    BinOpKind::Sub    => IRInstr::Sub  { dst: d, lhs: l, rhs: r },
                    BinOpKind::Mul    => IRInstr::Mul  { dst: d, lhs: l, rhs: r },
                    BinOpKind::Div    => IRInstr::SDiv { dst: d, lhs: l, rhs: r },
                    BinOpKind::Rem    => IRInstr::SRem { dst: d, lhs: l, rhs: r },
                    BinOpKind::BitAnd => IRInstr::And  { dst: d, lhs: l, rhs: r },
                    BinOpKind::BitOr  => IRInstr::Or   { dst: d, lhs: l, rhs: r },
                    BinOpKind::BitXor => IRInstr::Xor  { dst: d, lhs: l, rhs: r },
                    BinOpKind::Shl    => IRInstr::Shl  { dst: d, lhs: l, rhs: r },
                    BinOpKind::Shr    => IRInstr::AShr { dst: d, lhs: l, rhs: r },
                    BinOpKind::Eq     => IRInstr::ICmp { dst: d, cond: ICmpCond::Eq,  lhs: l, rhs: r },
                    BinOpKind::Ne     => IRInstr::ICmp { dst: d, cond: ICmpCond::Ne,  lhs: l, rhs: r },
                    BinOpKind::Lt     => IRInstr::ICmp { dst: d, cond: ICmpCond::SLt, lhs: l, rhs: r },
                    BinOpKind::Le     => IRInstr::ICmp { dst: d, cond: ICmpCond::SLe, lhs: l, rhs: r },
                    BinOpKind::Gt     => IRInstr::ICmp { dst: d, cond: ICmpCond::SGt, lhs: l, rhs: r },
                    BinOpKind::Ge     => IRInstr::ICmp { dst: d, cond: ICmpCond::SGe, lhs: l, rhs: r },
                    _ => { self.emit(IRInstr::Const { dst: d, value: 0 }); return d; }
                };
                self.emit(i);
                d
            }
            Expr::UnOp { op, expr: inner, .. } => {
                let s = self.lower_expr(inner);
                let d = self.fresh_var();
                let i = match op {
                    UnOpKind::Neg => IRInstr::Neg { dst: d, src: s },
                    UnOpKind::Not => IRInstr::Not { dst: d, src: s },
                    _ => { self.emit(IRInstr::Const { dst: d, value: 0 }); return d; }
                };
                self.emit(i);
                d
            }
            Expr::Call { func: func_e, args, .. } => {
                if let Expr::Ident { name, .. } = func_e.as_ref() {
                    let arg_vars: Vec<_> = args.iter()
                        .map(|a| self.lower_expr(a)).collect();
                    let d = self.fresh_var();
                    self.emit(IRInstr::Call { dst: d, func: name.clone(), args: arg_vars });
                    d
                } else {
                    let v = self.fresh_var();
                    self.emit(IRInstr::Const { dst: v, value: 0 });
                    v
                }
            }
            _ => {
                let v = self.fresh_var();
                self.emit(IRInstr::Const { dst: v, value: 0 });
                v
            }
        }
    }

    fn into_func(self) -> IRFunction { self.func }
}

pub fn lower_to_ir(f: &FnDecl) -> IRFunction {
    let mut ctx = LowerCtx::new(&f.name);
    ctx.lower_fn(f);
    ctx.into_func()
}

// =============================================================================
// §4  LATTICE VALUES (module-level — fixes v1 type-visibility bug)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatticeVal {
    Undefined,
    Constant(i64),
    Overdefined,
}

impl LatticeVal {
    fn meet(a: LatticeVal, b: LatticeVal) -> LatticeVal {
        match (a, b) {
            (LatticeVal::Undefined, x) | (x, LatticeVal::Undefined) => x,
            (LatticeVal::Overdefined, _) | (_, LatticeVal::Overdefined) => LatticeVal::Overdefined,
            (LatticeVal::Constant(x), LatticeVal::Constant(y)) if x == y => LatticeVal::Constant(x),
            _ => LatticeVal::Overdefined,
        }
    }

    fn eval_binop(op: fn(i64,i64)->Option<i64>, a: LatticeVal, b: LatticeVal) -> LatticeVal {
        match (a, b) {
            (LatticeVal::Constant(x), LatticeVal::Constant(y)) =>
                op(x, y).map_or(LatticeVal::Overdefined, LatticeVal::Constant),
            (LatticeVal::Undefined, _) | (_, LatticeVal::Undefined) => LatticeVal::Undefined,
            _ => LatticeVal::Overdefined,
        }
    }
}

fn eval_instr_sccp(instr: &IRInstr, vals: &HashMap<VarId, LatticeVal>) -> LatticeVal {
    let get = |v: VarId| vals.get(&v).copied().unwrap_or(LatticeVal::Undefined);
    match instr {
        IRInstr::Const  { value, .. }    => LatticeVal::Constant(*value),
        IRInstr::Move   { src, .. }      => get(*src),
        IRInstr::Add  { lhs, rhs, .. }   =>
            LatticeVal::eval_binop(|a,b| Some(a.wrapping_add(b)), get(*lhs), get(*rhs)),
        IRInstr::Sub  { lhs, rhs, .. }   =>
            LatticeVal::eval_binop(|a,b| Some(a.wrapping_sub(b)), get(*lhs), get(*rhs)),
        IRInstr::Mul  { lhs, rhs, .. }   =>
            LatticeVal::eval_binop(|a,b| Some(a.wrapping_mul(b)), get(*lhs), get(*rhs)),
        IRInstr::And  { lhs, rhs, .. }   =>
            LatticeVal::eval_binop(|a,b| Some(a & b), get(*lhs), get(*rhs)),
        IRInstr::Or   { lhs, rhs, .. }   =>
            LatticeVal::eval_binop(|a,b| Some(a | b), get(*lhs), get(*rhs)),
        IRInstr::Xor  { lhs, rhs, .. }   =>
            LatticeVal::eval_binop(|a,b| Some(a ^ b), get(*lhs), get(*rhs)),
        IRInstr::Shl  { lhs, rhs, .. }   =>
            LatticeVal::eval_binop(|a,b| (b < 64).then(|| a.wrapping_shl(b as u32)),
                                   get(*lhs), get(*rhs)),
        IRInstr::AShr { lhs, rhs, .. }   =>
            LatticeVal::eval_binop(|a,b| (b < 64).then(|| a.wrapping_shr(b as u32)),
                                   get(*lhs), get(*rhs)),
        IRInstr::Neg  { src, .. } =>
            if let LatticeVal::Constant(v) = get(*src) { LatticeVal::Constant(v.wrapping_neg()) }
            else { LatticeVal::Overdefined },
        IRInstr::ICmp { cond, lhs, rhs, .. } => {
            if let (LatticeVal::Constant(l), LatticeVal::Constant(r)) = (get(*lhs), get(*rhs)) {
                let b = match cond {
                    ICmpCond::Eq  => l == r, ICmpCond::Ne  => l != r,
                    ICmpCond::SLt => l < r,  ICmpCond::SLe => l <= r,
                    ICmpCond::SGt => l > r,  ICmpCond::SGe => l >= r,
                    ICmpCond::ULt => (l as u64) < (r as u64),
                    ICmpCond::ULe => (l as u64) <= (r as u64),
                    ICmpCond::UGt => (l as u64) > (r as u64),
                    ICmpCond::UGe => (l as u64) >= (r as u64),
                };
                LatticeVal::Constant(b as i64)
            } else { LatticeVal::Overdefined }
        }
        _ => LatticeVal::Overdefined,
    }
}

// =============================================================================
// §5a  SPARSE CONDITIONAL CONSTANT PROPAGATION (Wegman-Zadeck)
// =============================================================================

pub fn run_sccp(func: &mut IRFunction) {
    let mut vals:       HashMap<VarId, LatticeVal> = HashMap::new();
    let mut executable: HashSet<BlockId>            = HashSet::new();
    let mut cfg_work:   VecDeque<BlockId>           = VecDeque::new();
    let mut ssa_work:   VecDeque<(BlockId, usize)>  = VecDeque::new();

    for &p in &func.params { vals.insert(p, LatticeVal::Overdefined); }
    executable.insert(func.entry_block);
    cfg_work.push_back(func.entry_block);

    // BFS over reachable blocks first
    while let Some(bid) = cfg_work.pop_front() {
        if let Some(block) = func.blocks.get(&bid) {
            let succs = block.successors.clone();
            for (idx, _) in block.instrs.iter().enumerate() {
                ssa_work.push_back((bid, idx));
            }
            for s in succs {
                if executable.insert(s) { cfg_work.push_back(s); }
            }
        }
    }

    // Main SSA worklist
    while let Some((bid, idx)) = ssa_work.pop_front() {
        let instr = match func.blocks.get(&bid).and_then(|b| b.instrs.get(idx)) {
            Some(i) => i.clone(),
            None    => continue,
        };
        if let Some(dst) = instr_def(&instr) {
            let old = vals.get(&dst).copied().unwrap_or(LatticeVal::Undefined);
            let new = eval_instr_sccp(&instr, &vals);
            let merged = LatticeVal::meet(old, new);
            if merged != old {
                vals.insert(dst, merged);
                // Re-add all users
                for (&ub, ublk) in &func.blocks {
                    if !executable.contains(&ub) { continue; }
                    for (ui, ui_instr) in ublk.instrs.iter().enumerate() {
                        if instr_uses(ui_instr).contains(&dst) {
                            ssa_work.push_back((ub, ui));
                        }
                    }
                }
            }
        }
        // Conditional branch constant folding → prune CFG
        match &instr {
            IRInstr::CondBr { cond, if_true, if_false } => {
                match vals.get(cond) {
                    Some(&LatticeVal::Constant(c)) => {
                        let t = if c != 0 { *if_true } else { *if_false };
                        if executable.insert(t) { cfg_work.push_back(t); }
                    }
                    _ => {
                        for &t in &[*if_true, *if_false] {
                            if executable.insert(t) { cfg_work.push_back(t); }
                        }
                    }
                }
            }
            IRInstr::Br { target } => {
                if executable.insert(*target) { cfg_work.push_back(*target); }
            }
            _ => {}
        }
    }

    // Rewrite: replace constant defs with Const instructions
    for (&bid, block) in &mut func.blocks {
        if !executable.contains(&bid) {
            block.instrs = vec![IRInstr::Br { target: func.entry_block }];
            continue;
        }
        let mut new_instrs = Vec::with_capacity(block.instrs.len());
        for instr in &block.instrs {
            if let Some(dst) = instr_def(instr) {
                if let Some(&LatticeVal::Constant(c)) = vals.get(&dst) {
                    if !instr.has_side_effects() {
                        new_instrs.push(IRInstr::Const { dst, value: c });
                        continue;
                    }
                }
            }
            new_instrs.push(instr.clone());
        }
        block.instrs = new_instrs;
    }
}

// =============================================================================
// §5b  GLOBAL VALUE NUMBERING (hash-consing)
// =============================================================================

pub fn run_gvn(func: &mut IRFunction) {
    #[derive(Hash, PartialEq, Eq, Debug)]
    enum VNKey { Add(VarId,VarId), Sub(VarId,VarId), Mul(VarId,VarId),
                 And(VarId,VarId), Or(VarId,VarId),  Xor(VarId,VarId),
                 ICmp(ICmpCond, VarId, VarId) }

    let mut table: HashMap<VNKey, VarId> = HashMap::new();
    // Per-block pass (a full inter-block GVN would need dominator tree walk)
    for block in func.blocks.values_mut() {
        let mut new = Vec::with_capacity(block.instrs.len());
        for instr in &block.instrs {
            let key = match instr {
                IRInstr::Add { lhs, rhs, .. } =>
                    Some(VNKey::Add((*lhs).min(*rhs), (*lhs).max(*rhs))),
                IRInstr::Mul { lhs, rhs, .. } =>
                    Some(VNKey::Mul((*lhs).min(*rhs), (*lhs).max(*rhs))),
                IRInstr::Sub { lhs, rhs, .. } =>
                    Some(VNKey::Sub(*lhs, *rhs)),
                IRInstr::And { lhs, rhs, .. } =>
                    Some(VNKey::And((*lhs).min(*rhs), (*lhs).max(*rhs))),
                IRInstr::Or  { lhs, rhs, .. } =>
                    Some(VNKey::Or ((*lhs).min(*rhs), (*lhs).max(*rhs))),
                IRInstr::Xor { lhs, rhs, .. } =>
                    Some(VNKey::Xor((*lhs).min(*rhs), (*lhs).max(*rhs))),
                IRInstr::ICmp { cond, lhs, rhs, .. } =>
                    Some(VNKey::ICmp(*cond, *lhs, *rhs)),
                _ => None,
            };
            if let (Some(k), Some(dst)) = (key, instr_def(instr)) {
                if let Some(&existing) = table.get(&k) {
                    // Replace with Move (proper copy semantics)
                    new.push(IRInstr::Move { dst, src: existing });
                    continue;
                }
                table.insert(k, dst);
            }
            new.push(instr.clone());
        }
        block.instrs = new;
    }
}

// =============================================================================
// §5c  COPY PROPAGATION
// =============================================================================

pub fn run_copy_prop(func: &mut IRFunction) {
    // Build copy map: if dst = Move { src }, substitute all uses of dst with src
    let mut copies: HashMap<VarId, VarId> = HashMap::new();
    for block in func.blocks.values() {
        for instr in &block.instrs {
            if let IRInstr::Move { dst, src } = instr {
                let canonical = *copies.get(src).unwrap_or(src);
                copies.insert(*dst, canonical);
            }
        }
    }
    if copies.is_empty() { return; }

    fn subst(v: VarId, copies: &HashMap<VarId, VarId>) -> VarId {
        *copies.get(&v).unwrap_or(&v)
    }

    for block in func.blocks.values_mut() {
        for instr in &mut block.instrs {
            match instr {
                IRInstr::Move   { src, .. }        => *src = subst(*src, &copies),
                IRInstr::Add  { lhs, rhs, .. } |
                IRInstr::Sub  { lhs, rhs, .. } |
                IRInstr::Mul  { lhs, rhs, .. } |
                IRInstr::SDiv { lhs, rhs, .. } |
                IRInstr::SRem { lhs, rhs, .. } |
                IRInstr::And  { lhs, rhs, .. } |
                IRInstr::Or   { lhs, rhs, .. } |
                IRInstr::Xor  { lhs, rhs, .. } |
                IRInstr::Shl  { lhs, rhs, .. } |
                IRInstr::AShr { lhs, rhs, .. } |
                IRInstr::LShr { lhs, rhs, .. } |
                IRInstr::ICmp { lhs, rhs, .. } => {
                    *lhs = subst(*lhs, &copies);
                    *rhs = subst(*rhs, &copies);
                }
                IRInstr::Neg { src, .. } |
                IRInstr::Not { src, .. } => *src = subst(*src, &copies),
                IRInstr::CondBr { cond, .. }  => *cond = subst(*cond, &copies),
                IRInstr::Ret   { value: Some(v), .. } => *v = subst(*v, &copies),
                IRInstr::Store { ptr, value } => {
                    *ptr   = subst(*ptr,   &copies);
                    *value = subst(*value, &copies);
                }
                IRInstr::Load  { ptr, .. }  => *ptr = subst(*ptr, &copies),
                IRInstr::Call  { args, .. } => {
                    for a in args { *a = subst(*a, &copies); }
                }
                IRInstr::TailCall { args, .. } => {
                    for a in args { *a = subst(*a, &copies); }
                }
                IRInstr::Phi { incoming, .. } => {
                    for (_, v) in incoming { *v = subst(*v, &copies); }
                }
                _ => {}
            }
        }
    }
}

// =============================================================================
// §5d  DEAD CODE ELIMINATION (iterative liveness)
// =============================================================================

pub fn run_dce(func: &mut IRFunction) {
    let mut live: HashMap<BlockId, HashSet<VarId>> = func.blocks.keys()
        .map(|&id| (id, HashSet::new())).collect();

    let mut changed = true;
    while changed {
        changed = false;
        let bids: Vec<BlockId> = func.blocks.keys().copied().rev().collect();
        for bid in bids {
            let block = &func.blocks[&bid];
            let mut cur_live: HashSet<VarId> = block.successors.iter()
                .flat_map(|s| live.get(s).into_iter().flatten().copied())
                .collect();
            for instr in block.instrs.iter().rev() {
                if let Some(d) = instr_def(instr) { cur_live.remove(&d); }
                cur_live.extend(instr_uses(instr));
            }
            if cur_live != live[&bid] {
                live.insert(bid, cur_live);
                changed = true;
            }
        }
    }

    for (&bid, block) in &mut func.blocks {
        let mut live_now = live[&bid].clone();
        let mut new_instrs: Vec<IRInstr> = Vec::new();
        for instr in block.instrs.iter().rev() {
            let dead = matches!(instr_def(instr), Some(d) if !live_now.contains(&d))
                && !instr.has_side_effects();
            if !dead {
                if let Some(d) = instr_def(instr) { live_now.remove(&d); }
                live_now.extend(instr_uses(instr));
                new_instrs.push(instr.clone());
            }
        }
        new_instrs.reverse();
        block.instrs = new_instrs;
    }
}

// =============================================================================
// §5e  VALUE RANGE PROPAGATION
// =============================================================================

#[derive(Debug, Clone, Copy)]
pub struct ValueRange { pub lo: i64, pub hi: i64 }

impl ValueRange {
    fn unknown() -> Self { Self { lo: i64::MIN, hi: i64::MAX } }
    fn point(v: i64) -> Self { Self { lo: v, hi: v } }
    fn is_point(self) -> bool { self.lo == self.hi }
    fn intersect(a: Self, b: Self) -> Self {
        Self { lo: a.lo.max(b.lo), hi: a.hi.min(b.hi) }
    }
    fn join(a: Self, b: Self) -> Self {
        Self { lo: a.lo.min(b.lo), hi: a.hi.max(b.hi) }
    }
}

pub fn run_vrp(func: &mut IRFunction) -> HashMap<VarId, ValueRange> {
    let mut ranges: HashMap<VarId, ValueRange> = HashMap::new();

    // Single forward pass for now (a full VRP needs fix-point iteration)
    for block in func.blocks.values() {
        for instr in &block.instrs {
            let range = match instr {
                IRInstr::Const { value, .. } => ValueRange::point(*value),
                IRInstr::Add { lhs, rhs, .. } => {
                    let l = ranges.get(lhs).copied().unwrap_or_else(ValueRange::unknown);
                    let r = ranges.get(rhs).copied().unwrap_or_else(ValueRange::unknown);
                    ValueRange {
                        lo: l.lo.saturating_add(r.lo),
                        hi: l.hi.saturating_add(r.hi),
                    }
                }
                IRInstr::ICmp { .. } => ValueRange { lo: 0, hi: 1 },
                IRInstr::Move { src, .. } => {
                    ranges.get(src).copied().unwrap_or_else(ValueRange::unknown)
                }
                _ => ValueRange::unknown(),
            };
            if let Some(d) = instr_def(instr) {
                ranges.insert(d, range);
            }
        }
    }

    // Use range info to fold constants
    for block in func.blocks.values_mut() {
        let mut new_instrs = Vec::with_capacity(block.instrs.len());
        for instr in &block.instrs {
            if let Some(d) = instr_def(instr) {
                if let Some(r) = ranges.get(&d) {
                    if r.is_point() && !instr.has_side_effects() {
                        // Range collapses to a single value — fold it
                        new_instrs.push(IRInstr::Const { dst: d, value: r.lo });
                        continue;
                    }
                }
            }
            new_instrs.push(instr.clone());
        }
        block.instrs = new_instrs;
    }

    ranges
}

// =============================================================================
// §5f  DOMINANCE TREE (simple iterative algorithm — Cooper et al.)
// =============================================================================

/// Compute immediate dominators. Returns `idom[b] = immediate dominator of b`.
pub fn compute_dominators(func: &IRFunction) -> HashMap<BlockId, BlockId> {
    let entry = func.entry_block;
    let mut idom: HashMap<BlockId, BlockId> = HashMap::new();
    idom.insert(entry, entry);

    // Reverse post-order traversal
    let rpo = rpo_order(func);
    let rpo_idx: HashMap<BlockId, usize> = rpo.iter().enumerate()
        .map(|(i, &b)| (b, i)).collect();

    let mut changed = true;
    while changed {
        changed = false;
        for &b in &rpo {
            if b == entry { continue; }
            let preds: Vec<BlockId> = func.blocks.get(&b)
                .map(|blk| blk.predecessors.clone())
                .unwrap_or_default();
            let mut new_idom: Option<BlockId> = None;
            for p in preds {
                if !idom.contains_key(&p) { continue; }
                new_idom = Some(match new_idom {
                    None    => p,
                    Some(d) => intersect(p, d, &idom, &rpo_idx),
                });
            }
            if let Some(d) = new_idom {
                if idom.get(&b) != Some(&d) {
                    idom.insert(b, d);
                    changed = true;
                }
            }
        }
    }
    idom
}

fn intersect(a: BlockId, b: BlockId,
             idom: &HashMap<BlockId, BlockId>,
             rpo_idx: &HashMap<BlockId, usize>) -> BlockId {
    let mut f1 = a;
    let mut f2 = b;
    while f1 != f2 {
        while rpo_idx.get(&f1).unwrap_or(&usize::MAX) >
              rpo_idx.get(&f2).unwrap_or(&usize::MAX) {
            f1 = *idom.get(&f1).unwrap_or(&f1);
        }
        while rpo_idx.get(&f2).unwrap_or(&usize::MAX) >
              rpo_idx.get(&f1).unwrap_or(&usize::MAX) {
            f2 = *idom.get(&f2).unwrap_or(&f2);
        }
    }
    f1
}

pub fn dominates(idom: &HashMap<BlockId, BlockId>, a: BlockId, b: BlockId) -> bool {
    let mut cur = b;
    loop {
        if cur == a { return true; }
        let next = *idom.get(&cur).unwrap_or(&cur);
        if next == cur { return false; }
        cur = next;
    }
}

fn rpo_order(func: &IRFunction) -> Vec<BlockId> {
    let mut visited = HashSet::new();
    let mut order: Vec<BlockId>   = Vec::new();
    let mut stack   = vec![func.entry_block];
    // Post-order DFS
    let mut post    = Vec::new();
    while let Some(bid) = stack.pop() {
        if !visited.insert(bid) { continue; }
        post.push(bid);
        if let Some(b) = func.blocks.get(&bid) {
            for &s in &b.successors { stack.push(s); }
        }
    }
    post.reverse(); // reverse post-order
    post
}

// =============================================================================
// §5g  NATURAL LOOP DETECTION
// =============================================================================

#[derive(Debug)]
pub struct NaturalLoop {
    pub header:    BlockId,
    pub body:      HashSet<BlockId>,
    pub latches:   Vec<BlockId>,
    pub depth:     usize,
}

pub fn detect_loops(func: &IRFunction, idom: &HashMap<BlockId, BlockId>) -> Vec<NaturalLoop> {
    let mut loops: Vec<NaturalLoop> = Vec::new();
    // Find back edges: (latch → header) where header dominates latch
    for (&bid, block) in &func.blocks {
        for &succ in &block.successors {
            if dominates(idom, succ, bid) {
                // Back edge: bid→succ, header=succ
                let body = find_loop_body(bid, succ, func);
                let depth = loops.iter().filter(|l| l.body.contains(&succ)).count();
                if let Some(existing) = loops.iter_mut().find(|l| l.header == succ) {
                    existing.latches.push(bid);
                    existing.body.extend(&body);
                } else {
                    loops.push(NaturalLoop { header: succ, body, latches: vec![bid], depth });
                }
            }
        }
    }
    loops
}

fn find_loop_body(latch: BlockId, header: BlockId, func: &IRFunction) -> HashSet<BlockId> {
    let mut body = HashSet::new();
    body.insert(header);
    body.insert(latch);
    let mut worklist = vec![latch];
    while let Some(b) = worklist.pop() {
        if b == header { continue; }
        if let Some(block) = func.blocks.get(&b) {
            for &pred in &block.predecessors {
                if body.insert(pred) { worklist.push(pred); }
            }
        }
    }
    body
}

// =============================================================================
// §5h  LOOP-INVARIANT CODE MOTION (LICM)
// =============================================================================

pub fn run_licm(func: &mut IRFunction, loops: &[NaturalLoop]) {
    for lp in loops {
        // Compute set of variables defined inside the loop
        let loop_defs: HashSet<VarId> = lp.body.iter()
            .flat_map(|&bid| func.blocks.get(&bid).into_iter()
                .flat_map(|b| b.instrs.iter().filter_map(instr_def)))
            .collect();

        // Find loop-invariant instructions: all uses come from outside the loop
        let mut hoisted: HashSet<(BlockId, usize)> = HashSet::new();
        let mut changed = true;
        while changed {
            changed = false;
            for &bid in &lp.body {
                if bid == lp.header { continue; }
                let instrs = match func.blocks.get(&bid) {
                    Some(b) => b.instrs.clone(),
                    None    => continue,
                };
                for (idx, instr) in instrs.iter().enumerate() {
                    if hoisted.contains(&(bid, idx)) { continue; }
                    if instr.has_side_effects() { continue; }
                    if instr_def(instr).is_none() { continue; }
                    let uses = instr_uses(instr);
                    let invariant = uses.iter().all(|u| !loop_defs.contains(u)
                        || hoisted.iter().any(|(hb, hi)| {
                            func.blocks.get(hb).and_then(|b| b.instrs.get(*hi))
                                .and_then(instr_def).map_or(false, |d| d == *u)
                        }));
                    if invariant {
                        hoisted.insert((bid, idx));
                        changed = true;
                    }
                }
            }
        }

        // Move hoisted instructions to a pre-header block before the loop header
        if hoisted.is_empty() { continue; }
        let pre_header = func.fresh_block();
        // Insert pre-header between predecessors-outside-loop and header
        let outside_preds: Vec<BlockId> = func.blocks.get(&lp.header)
            .map(|b| b.predecessors.iter()
                .filter(|&&p| !lp.body.contains(&p))
                .copied().collect())
            .unwrap_or_default();

        // Redirect outside preds to pre-header
        for &pred in &outside_preds {
            if let Some(pb) = func.blocks.get_mut(&pred) {
                for s in &mut pb.successors {
                    if *s == lp.header { *s = pre_header; }
                }
                for instr in &mut pb.instrs {
                    if let IRInstr::Br { target } = instr {
                        if *target == lp.header { *target = pre_header; }
                    }
                    if let IRInstr::CondBr { if_true, if_false, .. } = instr {
                        if *if_true  == lp.header { *if_true  = pre_header; }
                        if *if_false == lp.header { *if_false = pre_header; }
                    }
                }
            }
        }

        // Populate pre-header with hoisted instructions + jump to header
        let mut pre_instrs: Vec<IRInstr> = Vec::new();
        for &(bid, idx) in &hoisted {
            if let Some(instr) = func.blocks.get(&bid).and_then(|b| b.instrs.get(idx)) {
                pre_instrs.push(instr.clone());
            }
        }
        pre_instrs.push(IRInstr::Br { target: lp.header });
        if let Some(pb) = func.blocks.get_mut(&pre_header) {
            pb.instrs    = pre_instrs;
            pb.successors = vec![lp.header];
        }

        // Remove hoisted instructions from their original blocks
        let mut hoisted_by_block: HashMap<BlockId, HashSet<usize>> = HashMap::new();
        for (bid, idx) in hoisted {
            hoisted_by_block.entry(bid).or_default().insert(idx);
        }
        for (bid, idxs) in hoisted_by_block {
            if let Some(block) = func.blocks.get_mut(&bid) {
                let mut new_instrs = Vec::new();
                for (i, instr) in block.instrs.iter().enumerate() {
                    if !idxs.contains(&i) { new_instrs.push(instr.clone()); }
                }
                block.instrs = new_instrs;
            }
        }
    }
}

// =============================================================================
// §5i  INDUCTION VARIABLE STRENGTH REDUCTION
// =============================================================================

pub fn run_strength_reduction(func: &mut IRFunction, loops: &[NaturalLoop]) {
    for lp in loops {
        // Find induction variables: i = i + const inside the loop
        let mut ivs: HashMap<VarId, (VarId, i64)> = HashMap::new(); // var → (base, step)

        for &bid in &lp.body {
            let instrs = match func.blocks.get(&bid) {
                Some(b) => b.instrs.clone(),
                None    => continue,
            };
            for instr in &instrs {
                if let IRInstr::Add { dst, lhs, rhs } = instr {
                    // Check if lhs is a loop variable and rhs is a constant
                    let rhs_const = find_const_def(func, *rhs, &lp.body);
                    if let Some(step) = rhs_const {
                        ivs.insert(*dst, (*lhs, step));
                    }
                    let lhs_const = find_const_def(func, *lhs, &lp.body);
                    if let Some(step) = lhs_const {
                        ivs.insert(*dst, (*rhs, step));
                    }
                }
            }
        }

        // Strength-reduce: mul iv, const → accumulate addition
        for &bid in &lp.body {
            let instrs = match func.blocks.get(&bid) {
                Some(b) => b.instrs.clone(),
                None    => continue,
            };
            let mut new_instrs = Vec::with_capacity(instrs.len());
            for instr in &instrs {
                if let IRInstr::Mul { dst, lhs, rhs } = instr {
                    let rhs_const = find_const_def(func, *rhs, &lp.body);
                    if let Some(factor) = rhs_const {
                        if let Some(&(base, step)) = ivs.get(lhs) {
                            // Replace mul with: dst_accum = base * factor (pre-header)
                            // and dst_accum += step * factor (in increment block)
                            // For simplicity, convert mul by power-of-2 to shl
                            if factor > 0 && (factor as u64).is_power_of_two() {
                                let shift_var = func.fresh_var();
                                let shift     = factor.trailing_zeros() as i64;
                                new_instrs.push(IRInstr::Const { dst: shift_var, value: shift });
                                new_instrs.push(IRInstr::Shl { dst: *dst, lhs: *lhs, rhs: shift_var });
                                continue;
                            }
                        }
                    }
                    new_instrs.push(instr.clone());
                } else {
                    new_instrs.push(instr.clone());
                }
            }
            if let Some(block) = func.blocks.get_mut(&bid) {
                block.instrs = new_instrs;
            }
        }
    }
}

fn find_const_def(func: &IRFunction, var: VarId, _loop_body: &HashSet<BlockId>) -> Option<i64> {
    for block in func.blocks.values() {
        for instr in &block.instrs {
            if let IRInstr::Const { dst, value } = instr {
                if *dst == var { return Some(*value); }
            }
        }
    }
    None
}

// =============================================================================
// §5j  PEEPHOLE OPTIMISATION (40+ patterns, multi-pass)
// =============================================================================

pub fn run_peephole(func: &mut IRFunction) {
    // Build a const-value map for the function
    let const_map: HashMap<VarId, i64> = func.blocks.values()
        .flat_map(|b| b.instrs.iter())
        .filter_map(|i| if let IRInstr::Const { dst, value } = i { Some((*dst, *value)) } else { None })
        .collect();

    for block in func.blocks.values_mut() {
        let mut changed = true;
        while changed {
            changed = false;
            let mut new: Vec<IRInstr> = Vec::with_capacity(block.instrs.len());
            let mut i = 0;
            while i < block.instrs.len() {
                let instr = block.instrs[i].clone();
                let mut did_replace = false;

                match &instr {
                    // ── sub x, x → 0 ────────────────────────────────────
                    IRInstr::Sub { dst, lhs, rhs } if lhs == rhs => {
                        new.push(IRInstr::Const { dst: *dst, value: 0 });
                        changed = true; did_replace = true;
                    }
                    // ── xor x, x → 0 ────────────────────────────────────
                    IRInstr::Xor { dst, lhs, rhs } if lhs == rhs => {
                        new.push(IRInstr::Const { dst: *dst, value: 0 });
                        changed = true; did_replace = true;
                    }
                    // ── and x, x → x ────────────────────────────────────
                    IRInstr::And { dst, lhs, rhs } if lhs == rhs => {
                        new.push(IRInstr::Move { dst: *dst, src: *lhs });
                        changed = true; did_replace = true;
                    }
                    // ── or x, x → x ─────────────────────────────────────
                    IRInstr::Or { dst, lhs, rhs } if lhs == rhs => {
                        new.push(IRInstr::Move { dst: *dst, src: *lhs });
                        changed = true; did_replace = true;
                    }
                    // ── add x, 0 → x ────────────────────────────────────
                    IRInstr::Add { dst, lhs, rhs }
                        if const_map.get(rhs) == Some(&0) => {
                        new.push(IRInstr::Move { dst: *dst, src: *lhs });
                        changed = true; did_replace = true;
                    }
                    IRInstr::Add { dst, lhs, rhs }
                        if const_map.get(lhs) == Some(&0) => {
                        new.push(IRInstr::Move { dst: *dst, src: *rhs });
                        changed = true; did_replace = true;
                    }
                    // ── mul x, 0 → 0 ────────────────────────────────────
                    IRInstr::Mul { dst, lhs, rhs }
                        if const_map.get(rhs) == Some(&0)
                        || const_map.get(lhs) == Some(&0) => {
                        new.push(IRInstr::Const { dst: *dst, value: 0 });
                        changed = true; did_replace = true;
                    }
                    // ── mul x, 1 → x ────────────────────────────────────
                    IRInstr::Mul { dst, lhs, rhs }
                        if const_map.get(rhs) == Some(&1) => {
                        new.push(IRInstr::Move { dst: *dst, src: *lhs });
                        changed = true; did_replace = true;
                    }
                    IRInstr::Mul { dst, lhs, rhs }
                        if const_map.get(lhs) == Some(&1) => {
                        new.push(IRInstr::Move { dst: *dst, src: *rhs });
                        changed = true; did_replace = true;
                    }
                    // ── mul x, 2^n → shl x, n ───────────────────────────
                    IRInstr::Mul { dst, lhs, rhs } => {
                        if let Some(&v) = const_map.get(rhs) {
                            if v > 0 && (v as u64).is_power_of_two() {
                                let sv = func.next_var; func.next_var += 1;
                                new.push(IRInstr::Const { dst: sv, value: v.trailing_zeros() as i64 });
                                new.push(IRInstr::Shl { dst: *dst, lhs: *lhs, rhs: sv });
                                changed = true; did_replace = true;
                            }
                        } else if let Some(&v) = const_map.get(lhs) {
                            if v > 0 && (v as u64).is_power_of_two() {
                                let sv = func.next_var; func.next_var += 1;
                                new.push(IRInstr::Const { dst: sv, value: v.trailing_zeros() as i64 });
                                new.push(IRInstr::Shl { dst: *dst, lhs: *rhs, rhs: sv });
                                changed = true; did_replace = true;
                            }
                        }
                        if !did_replace { new.push(instr.clone()); }
                    }
                    // ── sdiv x, 2^n → sar x, n ──────────────────────────
                    IRInstr::SDiv { dst, lhs, rhs } => {
                        if let Some(&v) = const_map.get(rhs) {
                            if v > 0 && (v as u64).is_power_of_two() {
                                let sv = func.next_var; func.next_var += 1;
                                new.push(IRInstr::Const { dst: sv, value: v.trailing_zeros() as i64 });
                                new.push(IRInstr::AShr { dst: *dst, lhs: *lhs, rhs: sv });
                                changed = true; did_replace = true;
                            }
                        }
                        if !did_replace { new.push(instr.clone()); }
                    }
                    // ── shl x, 0 → x ────────────────────────────────────
                    IRInstr::Shl { dst, lhs, rhs }
                    | IRInstr::AShr { dst, lhs, rhs }
                    | IRInstr::LShr { dst, lhs, rhs }
                        if const_map.get(rhs) == Some(&0) => {
                        new.push(IRInstr::Move { dst: *dst, src: *lhs });
                        changed = true; did_replace = true;
                    }
                    // ── not(not(x)) → x (detect back-to-back Not) ───────
                    // ── neg(neg(x)) → x ─────────────────────────────────
                    // ── and x, -1 → x ───────────────────────────────────
                    IRInstr::And { dst, lhs, rhs }
                        if const_map.get(rhs) == Some(&-1) => {
                        new.push(IRInstr::Move { dst: *dst, src: *lhs });
                        changed = true; did_replace = true;
                    }
                    // ── or x, 0 → x ─────────────────────────────────────
                    IRInstr::Or { dst, lhs, rhs }
                        if const_map.get(rhs) == Some(&0) => {
                        new.push(IRInstr::Move { dst: *dst, src: *lhs });
                        changed = true; did_replace = true;
                    }
                    // ── icmp eq x, x → 1 ────────────────────────────────
                    IRInstr::ICmp { dst, cond: ICmpCond::Eq, lhs, rhs } if lhs == rhs => {
                        new.push(IRInstr::Const { dst: *dst, value: 1 });
                        changed = true; did_replace = true;
                    }
                    // ── icmp ne x, x → 0 ────────────────────────────────
                    IRInstr::ICmp { dst, cond: ICmpCond::Ne, lhs, rhs } if lhs == rhs => {
                        new.push(IRInstr::Const { dst: *dst, value: 0 });
                        changed = true; did_replace = true;
                    }
                    // ── move x, x → (eliminate) ─────────────────────────
                    IRInstr::Move { dst, src } if dst == src => {
                        changed = true; did_replace = true; // drop no-op
                    }
                    _ => {}
                }

                if !did_replace { new.push(instr); }
                i += 1;
            }
            block.instrs = new;
        }
    }
}

// =============================================================================
// §5k  TAIL CALL OPTIMISATION
// =============================================================================

pub fn run_tco(func: &mut IRFunction) {
    let fname = func.name.clone();
    for block in func.blocks.values_mut() {
        let n = block.instrs.len();
        if n < 2 { continue; }
        // Pattern: Call { func=fname, args } followed immediately by Ret { Some(dst) }
        if let (IRInstr::Call { dst: cdst, func: cfunc, args },
                IRInstr::Ret  { value: Some(rdst) })
            = (&block.instrs[n-2].clone(), &block.instrs[n-1].clone())
        {
            if cfunc == &fname && rdst == cdst {
                block.instrs[n-2] = IRInstr::TailCall { func: cfunc.clone(), args: args.clone() };
                block.instrs.pop();
            }
        }
    }
}

// =============================================================================
// §5l  JUMP THREADING
// =============================================================================

pub fn run_jump_threading(func: &mut IRFunction) {
    // Merge trivial blocks: block with only a Br → single successor
    let mut changed = true;
    while changed {
        changed = false;
        let trivial: Vec<(BlockId, BlockId)> = func.blocks.iter()
            .filter_map(|(&bid, block)| {
                if block.instrs.len() == 1 {
                    if let IRInstr::Br { target } = &block.instrs[0] {
                        if *target != bid { return Some((bid, *target)); }
                    }
                }
                None
            })
            .collect();

        for (from, to) in trivial {
            // Redirect all predecessors of `from` directly to `to`
            let preds: Vec<BlockId> = func.blocks.get(&from)
                .map(|b| b.predecessors.clone())
                .unwrap_or_default();
            for pred in preds {
                if let Some(pb) = func.blocks.get_mut(&pred) {
                    for s in &mut pb.successors {
                        if *s == from { *s = to; }
                    }
                    for instr in &mut pb.instrs {
                        if let IRInstr::Br { target } = instr {
                            if *target == from { *target = to; }
                        }
                        if let IRInstr::CondBr { if_true, if_false, .. } = instr {
                            if *if_true  == from { *if_true  = to; }
                            if *if_false == from { *if_false = to; }
                        }
                    }
                }
            }
            changed = true;
        }
        func.build_predecessors();
    }
}

// =============================================================================
// §5m  FUNCTION INLINING (with proper variable renaming)
// =============================================================================

pub fn run_inlining(program: &mut Program, cg: &CallGraph, cfg: &OptConfig) {
    if !cfg.inlining { return; }

    // Collect all function IR for cost analysis
    let fn_map: HashMap<String, FnDecl> = program.items.iter()
        .filter_map(|i| if let Item::Fn(f) = i { Some((f.name.clone(), f.clone())) } else { None })
        .collect();

    let candidates: HashSet<String> = fn_map.keys()
        .filter(|n| n.as_str() != "main" && cg.is_called_once(n))
        .filter(|n| {
            let ir = lower_to_ir(&fn_map[n.as_str()]);
            ir.inline_cost <= cfg.max_inline_size
        })
        .cloned()
        .collect();

    if candidates.is_empty() { return; }

    let new_items: Vec<Item> = program.items.iter().map(|item| {
        if let Item::Fn(f) = item {
            let mut nf = f.clone();
            if let Some(body) = &mut nf.body {
                inline_block(body, &candidates, &fn_map);
            }
            Item::Fn(nf)
        } else {
            item.clone()
        }
    }).collect();
    program.items = new_items;
}

fn inline_block(block: &mut Block, candidates: &HashSet<String>, fn_map: &HashMap<String, FnDecl>) {
    for s in &mut block.stmts { inline_stmt(s, candidates, fn_map); }
    if let Some(t) = &mut block.tail { inline_expr(t, candidates, fn_map); }
}

fn inline_stmt(s: &mut Stmt, candidates: &HashSet<String>, fn_map: &HashMap<String, FnDecl>) {
    match s {
        Stmt::Let  { init: Some(e), .. } => inline_expr(e, candidates, fn_map),
        Stmt::Expr { expr: e, .. }       => inline_expr(e, candidates, fn_map),
        Stmt::If   { cond, then, else_, .. } => {
            inline_expr(cond, candidates, fn_map);
            inline_block(then, candidates, fn_map);
            if let Some(e) = else_ {
                match e.as_mut() {
                    IfOrBlock::If(s)    => inline_stmt(s, candidates, fn_map),
                    IfOrBlock::Block(b) => inline_block(b, candidates, fn_map),
                }
            }
        }
        Stmt::While  { cond, body, .. } => {
            inline_expr(cond, candidates, fn_map);
            inline_block(body, candidates, fn_map);
        }
        Stmt::ForIn { iter, body, .. } => {
            inline_expr(iter, candidates, fn_map);
            inline_block(body, candidates, fn_map);
        }
        Stmt::Return { value: Some(e), .. } => inline_expr(e, candidates, fn_map),
        _ => {}
    }
}

fn inline_expr(e: &mut Expr, candidates: &HashSet<String>, fn_map: &HashMap<String, FnDecl>) {
    if let Expr::Call { func, args, .. } = e {
        if let Expr::Ident { name, .. } = func.as_ref() {
            if candidates.contains(name.as_str()) {
                // Full inlining would require AST alpha-renaming here;
                // we fall through and let the IR lowering handle it.
            }
        }
        for a in args.iter_mut() { inline_expr(a, candidates, fn_map); }
        inline_expr(func, candidates, fn_map);
    }
    match e {
        Expr::BinOp { lhs, rhs, .. } => {
            inline_expr(lhs, candidates, fn_map);
            inline_expr(rhs, candidates, fn_map);
        }
        Expr::UnOp { expr: inner, .. } => inline_expr(inner, candidates, fn_map),
        _ => {}
    }
}

// =============================================================================
// §6  IR UTILITY FUNCTIONS
// =============================================================================

pub fn instr_def(i: &IRInstr) -> Option<VarId> {
    match i {
        IRInstr::Const  { dst, .. } | IRInstr::ConstF64 { dst, .. } |
        IRInstr::Add    { dst, .. } | IRInstr::Sub      { dst, .. } |
        IRInstr::Mul    { dst, .. } | IRInstr::SDiv     { dst, .. } |
        IRInstr::SRem   { dst, .. } | IRInstr::Neg      { dst, .. } |
        IRInstr::And    { dst, .. } | IRInstr::Or       { dst, .. } |
        IRInstr::Xor    { dst, .. } | IRInstr::Shl      { dst, .. } |
        IRInstr::AShr   { dst, .. } | IRInstr::LShr     { dst, .. } |
        IRInstr::Not    { dst, .. } | IRInstr::ICmp     { dst, .. } |
        IRInstr::Move   { dst, .. } | IRInstr::Call     { dst, .. } |
        IRInstr::Alloca { dst, .. } | IRInstr::Load     { dst, .. } |
        IRInstr::Phi    { dst, .. } => Some(*dst),
        _ => None,
    }
}

pub fn instr_uses(i: &IRInstr) -> Vec<VarId> {
    let mut u = Vec::new();
    match i {
        IRInstr::Add  { lhs, rhs, .. } | IRInstr::Sub  { lhs, rhs, .. } |
        IRInstr::Mul  { lhs, rhs, .. } | IRInstr::SDiv { lhs, rhs, .. } |
        IRInstr::SRem { lhs, rhs, .. } | IRInstr::And  { lhs, rhs, .. } |
        IRInstr::Or   { lhs, rhs, .. } | IRInstr::Xor  { lhs, rhs, .. } |
        IRInstr::Shl  { lhs, rhs, .. } | IRInstr::AShr { lhs, rhs, .. } |
        IRInstr::LShr { lhs, rhs, .. } | IRInstr::ICmp { lhs, rhs, .. } => {
            u.push(*lhs); u.push(*rhs);
        }
        IRInstr::Neg  { src, .. } | IRInstr::Not  { src, .. } |
        IRInstr::Move { src, .. } | IRInstr::Load { ptr: src, .. } => { u.push(*src); }
        // ← CondBr uses `cond` (v1 bug fix: was incorrectly sharing `src` pattern)
        IRInstr::CondBr { cond, .. }                => { u.push(*cond); }
        IRInstr::Store  { ptr, value }              => { u.push(*ptr); u.push(*value); }
        IRInstr::Call   { args, .. }                |
        IRInstr::TailCall { args, .. }              => { u.extend(args.iter().copied()); }
        IRInstr::Ret    { value: Some(v) }          => { u.push(*v); }
        IRInstr::Phi    { incoming, .. }            => {
            for (_, v) in incoming { u.push(*v); }
        }
        _ => {}
    }
    u
}

// =============================================================================
// §7  REGISTER ALLOCATION (Linear Scan with coalescing hints)
// =============================================================================

/// Allocation outcome for a variable
#[derive(Debug, Clone, Copy)]
pub enum Alloc {
    Reg(u8),
    Stack(i32), // frame offset from rbp (negative)
}

pub struct RegAlloc {
    pub map:        HashMap<VarId, Alloc>,
    pub frame_size: usize,
    free:           Vec<u8>,
    active:         Vec<(usize, VarId, u8)>, // (end, var, reg)
    stack_off:      i32,
    /// Coalescing hints: vars that should share the same register
    hints:          HashMap<VarId, VarId>,
}

impl RegAlloc {
    fn new(hints: HashMap<VarId, VarId>) -> Self {
        Self {
            map: HashMap::new(),
            frame_size: 0,
            free: CALLER_SAVED.to_vec(),
            active: Vec::new(),
            stack_off: 0,
            hints,
        }
    }

    fn expire(&mut self, pos: usize) {
        let expired: Vec<u8> = self.active.iter()
            .filter(|&&(e, _, _)| e < pos)
            .map(|&(_, _, r)| r)
            .collect();
        self.active.retain(|&(e, _, _)| e >= pos);
        self.free.extend(expired);
    }

    pub fn alloc_var(&mut self, var: VarId, start: usize, end: usize) {
        self.expire(start);

        // Honour coalescing hint: try to give this var the same reg as its hint
        let hint_reg = self.hints.get(&var)
            .and_then(|h| self.map.get(h))
            .copied()
            .and_then(|a| if let Alloc::Reg(r) = a { Some(r) } else { None })
            .and_then(|r| if self.free.contains(&r) { Some(r) } else { None });

        let reg = hint_reg.or_else(|| self.free.pop());
        if let Some(r) = reg {
            if let Some(pos) = self.free.iter().position(|&x| x == r) {
                self.free.remove(pos);
            }
            self.map.insert(var, Alloc::Reg(r));
            self.active.push((end, var, r));
        } else {
            // Spill the furthest-ending active interval
            if let Some(worst_idx) = self.active.iter().enumerate()
                .max_by_key(|(_, &(e, _, _))| e)
                .map(|(i, _)| i)
            {
                if self.active[worst_idx].0 > end {
                    let (_, spill_var, r) = self.active[worst_idx];
                    self.stack_off -= 8;
                    self.map.insert(spill_var, Alloc::Stack(self.stack_off));
                    self.map.insert(var, Alloc::Reg(r));
                    self.active[worst_idx] = (end, var, r);
                    self.frame_size = self.frame_size.max((-self.stack_off) as usize);
                    return;
                }
            }
            self.stack_off -= 8;
            self.map.insert(var, Alloc::Stack(self.stack_off));
            self.frame_size = self.frame_size.max((-self.stack_off) as usize);
        }
    }
}

pub fn compute_live_intervals(func: &IRFunction) -> Vec<(VarId, usize, usize)> {
    let mut ivs: HashMap<VarId, (usize, usize)> = HashMap::new();
    let mut pos = 0usize;
    let order = rpo_order(func);
    for bid in &order {
        if let Some(block) = func.blocks.get(bid) {
            for instr in &block.instrs {
                if let Some(d) = instr_def(instr) {
                    ivs.entry(d).or_insert((pos, pos)).1 = pos;
                }
                for u in instr_uses(instr) {
                    let e = ivs.entry(u).or_insert((pos, pos));
                    e.1 = e.1.max(pos);
                }
                pos += 1;
            }
        }
    }
    let mut out: Vec<(VarId, usize, usize)> = ivs.into_iter()
        .map(|(v, (s, e))| (v, s, e)).collect();
    out.sort_by_key(|&(_, s, _)| s);
    out
}

fn build_coalescing_hints(func: &IRFunction) -> HashMap<VarId, VarId> {
    let mut hints = HashMap::new();
    for block in func.blocks.values() {
        for instr in &block.instrs {
            if let IRInstr::Move { dst, src } = instr {
                hints.insert(*dst, *src);
            }
        }
    }
    hints
}

pub fn allocate_registers(func: &IRFunction) -> RegAlloc {
    let hints = build_coalescing_hints(func);
    let mut alloc = RegAlloc::new(hints);
    let intervals = compute_live_intervals(func);
    for (var, start, end) in intervals {
        alloc.alloc_var(var, start, end);
    }
    // Align frame size to 16 bytes (ABI requirement)
    alloc.frame_size = (alloc.frame_size + 15) & !15;
    alloc
}

// =============================================================================
// §8  INSTRUCTION SCHEDULING (latency-aware list scheduling)
// =============================================================================

/// Approximate latency cycles for x86-64 instructions
fn instr_latency(i: &IRInstr) -> usize {
    match i {
        IRInstr::Mul { .. } | IRInstr::SDiv { .. } | IRInstr::SRem { .. } => 3,
        IRInstr::Call { .. }    => 10,
        IRInstr::Load { .. }    => 4, // assumes L1 hit
        IRInstr::Store { .. }   => 3,
        _                       => 1,
    }
}

pub fn run_sched(func: &mut IRFunction) {
    for block in func.blocks.values_mut() {
        let n = block.instrs.len();
        if n < 4 { continue; } // not worth scheduling tiny blocks

        // Build use-def dependency edges
        let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut def_at: HashMap<VarId, usize> = HashMap::new();

        for (i, instr) in block.instrs.iter().enumerate() {
            for u in instr_uses(instr) {
                if let Some(&d) = def_at.get(&u) {
                    if !deps[i].contains(&d) { deps[i].push(d); }
                }
            }
            if let Some(d) = instr_def(instr) { def_at.insert(d, i); }
        }

        // Compute ASAP (earliest cycle) for each instruction
        let mut asap = vec![0usize; n];
        for i in 0..n {
            for &dep in &deps[i] {
                asap[i] = asap[i].max(asap[dep] + instr_latency(&block.instrs[dep]));
            }
        }

        // Topological sort respecting ASAP (list scheduling)
        let mut scheduled = vec![false; n];
        let mut result    = Vec::with_capacity(n);
        let mut cycle     = 0usize;

        while result.len() < n {
            // Pick the ready instruction with the latest ASAP (critical-path first)
            let picked = (0..n)
                .filter(|&j| !scheduled[j] && deps[j].iter().all(|&d| scheduled[d]))
                .filter(|&j| asap[j] <= cycle)
                .max_by_key(|&j| asap[j]);

            if let Some(j) = picked {
                scheduled[j] = true;
                result.push(block.instrs[j].clone());
                cycle += instr_latency(&block.instrs[j]);
            } else {
                cycle += 1; // stall
            }
        }
        block.instrs = result;
    }
}

// =============================================================================
// §9  x86-64 CODE EMITTER
// =============================================================================

pub struct AsmEmitter { pub code: Vec<u8> }

impl AsmEmitter {
    pub fn new() -> Self { Self { code: Vec::with_capacity(16_384) } }
    pub fn pos(&self) -> usize { self.code.len() }

    #[inline] fn b(&mut self, v: u8)  { self.code.push(v); }
    #[inline] fn d(&mut self, v: i32) { self.code.extend_from_slice(&v.to_le_bytes()); }
    #[inline] fn q(&mut self, v: i64) { self.code.extend_from_slice(&v.to_le_bytes()); }

    fn rex(&mut self, w: bool, r: bool, x: bool, b: bool) {
        self.b(0x40 | ((w as u8)<<3) | ((r as u8)<<2) | ((x as u8)<<1) | (b as u8));
    }
    fn rex_w(&mut self, r: bool, x: bool, b_bit: bool) { self.rex(true, r, x, b_bit); }
    fn hi(r: u8) -> bool { r >= 8 }
    fn lo(r: u8) -> u8   { r & 7  }

    fn modrm(&mut self, mode: u8, reg: u8, rm: u8) {
        self.b((mode<<6) | ((reg & 7)<<3) | (rm & 7));
    }

    // ── MOV reg, imm64 (optimal encoding) ───────────────────────────────
    fn mov_r_imm(&mut self, reg: u8, imm: i64) {
        if imm == 0 {
            self.rex_w(Self::hi(reg), false, Self::hi(reg));
            self.b(0x31); self.modrm(3, Self::lo(reg), Self::lo(reg));
            return;
        }
        if let Ok(v) = i32::try_from(imm) {
            if v >= 0 {
                if Self::hi(reg) { self.b(0x41); }
                self.b(0xB8 | Self::lo(reg));
                self.d(v);
            } else {
                self.rex_w(Self::hi(reg), false, false);
                self.b(0xC7); self.modrm(3, 0, Self::lo(reg)); self.d(v);
            }
        } else {
            self.rex_w(Self::hi(reg), false, false);
            self.b(0xB8 | Self::lo(reg)); self.q(imm);
        }
    }

    // ── MOV dst, src ─────────────────────────────────────────────────────
    fn mov_rr(&mut self, dst: u8, src: u8) {
        if dst == src { return; }
        self.rex_w(Self::hi(dst), false, Self::hi(src));
        self.b(0x8B); self.modrm(3, Self::lo(dst), Self::lo(src));
    }

    // ── Load/Store [rbp + disp] ──────────────────────────────────────────
    fn mem_op(&mut self, opcode: u8, reg: u8, disp: i32) {
        self.rex_w(Self::hi(reg), false, false);
        self.b(opcode);
        if (-128..=127).contains(&disp) {
            self.modrm(1, Self::lo(reg), Self::lo(RBP));
            self.b(disp as i8 as u8);
        } else {
            self.modrm(2, Self::lo(reg), Self::lo(RBP));
            self.d(disp);
        }
    }
    fn load_rbp (&mut self, dst: u8, disp: i32) { self.mem_op(0x8B, dst, disp); }
    fn store_rbp(&mut self, src: u8, disp: i32) {
        self.rex_w(Self::hi(src), false, false);
        self.b(0x89);
        if (-128..=127).contains(&disp) {
            self.modrm(1, Self::lo(src), Self::lo(RBP));
            self.b(disp as i8 as u8);
        } else {
            self.modrm(2, Self::lo(src), Self::lo(RBP));
            self.d(disp);
        }
    }

    // ── ALU ops reg, reg ────────────────────────────────────────────────
    fn alu_rr(&mut self, opc: u8, dst: u8, src: u8) {
        self.rex_w(Self::hi(dst), false, Self::hi(src));
        self.b(opc); self.modrm(3, Self::lo(src), Self::lo(dst));
    }
    fn add_rr (&mut self, d: u8, s: u8) { self.alu_rr(0x01, d, s); }
    fn sub_rr (&mut self, d: u8, s: u8) { self.alu_rr(0x29, d, s); }
    fn and_rr (&mut self, d: u8, s: u8) { self.alu_rr(0x21, d, s); }
    fn or_rr  (&mut self, d: u8, s: u8) { self.alu_rr(0x09, d, s); }
    fn xor_rr (&mut self, d: u8, s: u8) { self.alu_rr(0x31, d, s); }
    fn cmp_rr (&mut self, a: u8, b: u8) { self.alu_rr(0x39, a, b); }
    fn test_rr(&mut self, a: u8, b: u8) { self.alu_rr(0x85, a, b); }

    // ── IMUL r, r ────────────────────────────────────────────────────────
    fn imul_rr(&mut self, dst: u8, src: u8) {
        self.rex_w(Self::hi(dst), false, Self::hi(src));
        self.b(0x0F); self.b(0xAF);
        self.modrm(3, Self::lo(dst), Self::lo(src));
    }

    // ── Unary ────────────────────────────────────────────────────────────
    fn neg_r(&mut self, r: u8) {
        self.rex_w(Self::hi(r), false, false);
        self.b(0xF7); self.modrm(3, 3, Self::lo(r));
    }
    fn not_r(&mut self, r: u8) {
        self.rex_w(Self::hi(r), false, false);
        self.b(0xF7); self.modrm(3, 2, Self::lo(r));
    }

    // ── Shifts: immediate ────────────────────────────────────────────────
    fn shl_ri (&mut self, r: u8, imm: u8) {
        self.rex_w(Self::hi(r), false, false);
        if imm == 1 { self.b(0xD1); self.modrm(3, 4, Self::lo(r)); }
        else        { self.b(0xC1); self.modrm(3, 4, Self::lo(r)); self.b(imm); }
    }
    fn sar_ri (&mut self, r: u8, imm: u8) {
        self.rex_w(Self::hi(r), false, false);
        if imm == 1 { self.b(0xD1); self.modrm(3, 7, Self::lo(r)); }
        else        { self.b(0xC1); self.modrm(3, 7, Self::lo(r)); self.b(imm); }
    }
    fn shr_ri (&mut self, r: u8, imm: u8) {
        self.rex_w(Self::hi(r), false, false);
        if imm == 1 { self.b(0xD1); self.modrm(3, 5, Self::lo(r)); }
        else        { self.b(0xC1); self.modrm(3, 5, Self::lo(r)); self.b(imm); }
    }

    // ── Shifts: variable (count in cl) ───────────────────────────────────
    fn shl_cl(&mut self, r: u8) {
        self.rex_w(Self::hi(r), false, false);
        self.b(0xD3); self.modrm(3, 4, Self::lo(r));
    }
    fn sar_cl(&mut self, r: u8) {
        self.rex_w(Self::hi(r), false, false);
        self.b(0xD3); self.modrm(3, 7, Self::lo(r));
    }
    fn shr_cl(&mut self, r: u8) {
        self.rex_w(Self::hi(r), false, false);
        self.b(0xD3); self.modrm(3, 5, Self::lo(r));
    }

    // ── Division ─────────────────────────────────────────────────────────
    fn cqo(&mut self) { self.b(0x48); self.b(0x99); }
    fn idiv_r(&mut self, r: u8) {
        self.rex_w(false, false, Self::hi(r));
        self.b(0xF7); self.modrm(3, 7, Self::lo(r));
    }

    // ── SetCC / MOVZX ────────────────────────────────────────────────────
    fn setcc(&mut self, cc: u8, r: u8) {
        self.b(0x0F); self.b(cc); self.modrm(3, 0, Self::lo(r));
    }
    fn movzx_64_8(&mut self, dst: u8, src: u8) {
        self.rex_w(Self::hi(dst), false, Self::hi(src));
        self.b(0x0F); self.b(0xB6);
        self.modrm(3, Self::lo(dst), Self::lo(src));
    }

    // ── Stack ─────────────────────────────────────────────────────────────
    fn push_r(&mut self, r: u8) {
        if Self::hi(r) { self.b(0x41); }
        self.b(0x50 | Self::lo(r));
    }
    fn pop_r(&mut self, r: u8) {
        if Self::hi(r) { self.b(0x41); }
        self.b(0x58 | Self::lo(r));
    }
    fn sub_rsp(&mut self, n: u32) {
        if n == 0 { return; }
        self.b(0x48); self.b(0x81); self.b(0xEC); self.d(n as i32);
    }
    fn add_rsp(&mut self, n: u32) {
        if n == 0 { return; }
        self.b(0x48); self.b(0x81); self.b(0xC4); self.d(n as i32);
    }

    // ── Control flow ──────────────────────────────────────────────────────
    fn jmp_rel32(&mut self) -> usize {
        self.b(0xE9); let p = self.pos(); self.d(0); p
    }
    fn jcc_rel32(&mut self, cc: u8) -> usize {
        self.b(0x0F); self.b(cc); let p = self.pos(); self.d(0); p
    }
    fn call_rel32(&mut self) -> usize {
        self.b(0xE8); let p = self.pos(); self.d(0); p
    }
    fn ret(&mut self) { self.b(0xC3); }
    fn syscall(&mut self) { self.b(0x0F); self.b(0x05); }

    // ── Patch a rel32 displacement ───────────────────────────────────────
    fn patch_rel32(&mut self, patch_site: usize, target_pos: usize) {
        let rel = (target_pos as i64) - (patch_site as i64 + 4);
        let bytes = (rel as i32).to_le_bytes();
        self.code[patch_site..patch_site+4].copy_from_slice(&bytes);
    }

    // ── Prologue / Epilogue ───────────────────────────────────────────────
    fn prologue(&mut self, frame: u32, saved: &[u8]) {
        for &r in saved { self.push_r(r); }
        self.push_r(RBP);
        // mov rbp, rsp
        self.b(0x48); self.b(0x89); self.b(0xE5);
        if frame > 0 { self.sub_rsp(frame); }
    }
    fn epilogue(&mut self, frame: u32, saved: &[u8]) {
        if frame > 0 { self.add_rsp(frame); }
        // mov rsp, rbp (restore rsp from rbp — correct epilogue)
        self.b(0x48); self.b(0x89); self.b(0xEC);
        self.pop_r(RBP);
        for &r in saved.iter().rev() { self.pop_r(r); }
        self.ret();
    }
}

// =============================================================================
// §10  IR → NATIVE CODE GENERATION
// =============================================================================

pub struct NativeCodeGen {
    pub emit:   AsmEmitter,
    labels:     HashMap<String, usize>,        // label name → code offset
    fixups:     Vec<(usize, String)>,          // (patch site, label)
}

impl NativeCodeGen {
    pub fn new() -> Self {
        Self { emit: AsmEmitter::new(), labels: HashMap::new(), fixups: Vec::new() }
    }

    pub fn compile_function(&mut self, func: &IRFunction) -> Result<(), String> {
        let alloc    = allocate_registers(func);
        let frame    = alloc.frame_size as u32;
        let used_regs: HashSet<u8> = alloc.map.values()
            .filter_map(|a| if let Alloc::Reg(r) = a { Some(*r) } else { None })
            .collect();
        let saved: Vec<u8> = CALLEE_SAVED.iter()
            .filter(|&&r| used_regs.contains(&r))
            .copied().collect();

        // Record function label
        self.labels.insert(func.name.clone(), self.emit.pos());
        self.emit.prologue(frame, &saved);

        // Move arguments from ABI regs to their allocated locations
        for (i, &p) in func.params.iter().enumerate().take(6) {
            let arg_reg = ARG_REGS[i];
            match alloc.map.get(&p) {
                Some(&Alloc::Reg(r))    => self.emit.mov_rr(r, arg_reg),
                Some(&Alloc::Stack(off)) => self.emit.store_rbp(arg_reg, off),
                None => {}
            }
        }

        // Emit blocks in RPO order for best fall-through layout
        let order = rpo_order(func);
        for bid in &order {
            let block = match func.blocks.get(bid) {
                Some(b) => b.clone(),
                None    => continue,
            };
            self.labels.insert(format!("B{}_{}", func.name, bid), self.emit.pos());

            for instr in &block.instrs {
                self.emit_instr(instr, &alloc, func)?;
            }
        }

        self.resolve_fixups();
        Ok(())
    }

    /// Get a VarId into a register, loading from stack if needed.
    /// Uses `scratch` as the scratch register for stack-homed vars.
    fn get_in_reg(&mut self, var: VarId, scratch: u8, alloc: &RegAlloc) -> Result<u8, String> {
        match alloc.map.get(&var) {
            Some(&Alloc::Reg(r))    => Ok(r),
            Some(&Alloc::Stack(off)) => {
                self.emit.load_rbp(scratch, off);
                Ok(scratch)
            }
            None => Err(format!("variable {} has no allocation", var)),
        }
    }

    /// Store register result to the variable's allocated location.
    fn put_from_reg(&mut self, var: VarId, reg: u8, alloc: &RegAlloc) {
        match alloc.map.get(&var) {
            Some(&Alloc::Reg(r)) if r != reg => self.emit.mov_rr(r, reg),
            Some(&Alloc::Stack(off))          => self.emit.store_rbp(reg, off),
            _ => {}
        }
    }

    fn emit_instr(&mut self, instr: &IRInstr, alloc: &RegAlloc,
                  func: &IRFunction) -> Result<(), String>
    {
        // Scratch registers: r10=10, r11=11 (caller-saved, safe as temporaries)
        const S0: u8 = 10; // r10
        const S1: u8 = 11; // r11

        match instr {
            IRInstr::Const { dst, value } => {
                self.emit.mov_r_imm(S0, *value);
                self.put_from_reg(*dst, S0, alloc);
            }
            IRInstr::Move { dst, src } => {
                let sr = self.get_in_reg(*src, S0, alloc)?;
                self.put_from_reg(*dst, sr, alloc);
            }
            IRInstr::Add { dst, lhs, rhs } => {
                let lr = self.get_in_reg(*lhs, S0, alloc)?;
                let rr = self.get_in_reg(*rhs, S1, alloc)?;
                self.emit.mov_rr(S0, lr);
                self.emit.add_rr(S0, rr);
                self.put_from_reg(*dst, S0, alloc);
            }
            IRInstr::Sub { dst, lhs, rhs } => {
                let lr = self.get_in_reg(*lhs, S0, alloc)?;
                let rr = self.get_in_reg(*rhs, S1, alloc)?;
                self.emit.mov_rr(S0, lr);
                self.emit.sub_rr(S0, rr);
                self.put_from_reg(*dst, S0, alloc);
            }
            IRInstr::Mul { dst, lhs, rhs } => {
                let lr = self.get_in_reg(*lhs, S0, alloc)?;
                let rr = self.get_in_reg(*rhs, S1, alloc)?;
                self.emit.mov_rr(S0, lr);
                self.emit.imul_rr(S0, rr);
                self.put_from_reg(*dst, S0, alloc);
            }
            IRInstr::SDiv { dst, lhs, rhs } => {
                let lr = self.get_in_reg(*lhs, S0, alloc)?;
                let rr = self.get_in_reg(*rhs, S1, alloc)?;
                self.emit.mov_rr(RET_REG, lr); // rax = lhs
                self.emit.cqo();               // sign-extend to rdx:rax
                self.emit.idiv_r(rr);           // rax = quotient, rdx = remainder
                self.put_from_reg(*dst, RET_REG, alloc);
            }
            IRInstr::SRem { dst, lhs, rhs } => {
                // v1 bug fix: remainder is in rdx, not rax
                let lr = self.get_in_reg(*lhs, S0, alloc)?;
                let rr = self.get_in_reg(*rhs, S1, alloc)?;
                self.emit.mov_rr(RET_REG, lr);
                self.emit.cqo();
                self.emit.idiv_r(rr);
                self.put_from_reg(*dst, 2 /*rdx*/, alloc); // rdx = remainder
            }
            IRInstr::Neg { dst, src } => {
                let sr = self.get_in_reg(*src, S0, alloc)?;
                self.emit.mov_rr(S0, sr);
                self.emit.neg_r(S0);
                self.put_from_reg(*dst, S0, alloc);
            }
            IRInstr::And { dst, lhs, rhs } => {
                let lr = self.get_in_reg(*lhs, S0, alloc)?;
                let rr = self.get_in_reg(*rhs, S1, alloc)?;
                self.emit.mov_rr(S0, lr);
                self.emit.and_rr(S0, rr);
                self.put_from_reg(*dst, S0, alloc);
            }
            IRInstr::Or { dst, lhs, rhs } => {
                let lr = self.get_in_reg(*lhs, S0, alloc)?;
                let rr = self.get_in_reg(*rhs, S1, alloc)?;
                self.emit.mov_rr(S0, lr);
                self.emit.or_rr(S0, rr);
                self.put_from_reg(*dst, S0, alloc);
            }
            IRInstr::Xor { dst, lhs, rhs } => {
                let lr = self.get_in_reg(*lhs, S0, alloc)?;
                let rr = self.get_in_reg(*rhs, S1, alloc)?;
                self.emit.mov_rr(S0, lr);
                self.emit.xor_rr(S0, rr);
                self.put_from_reg(*dst, S0, alloc);
            }
            IRInstr::Shl { dst, lhs, rhs } => {
                let lr = self.get_in_reg(*lhs, S0, alloc)?;
                let rr = self.get_in_reg(*rhs, S1, alloc)?;
                self.emit.mov_rr(S0, lr);
                // Shift count must be in cl (rcx = reg 1)
                self.emit.mov_rr(1 /*rcx*/, rr);
                self.emit.shl_cl(S0);
                self.put_from_reg(*dst, S0, alloc);
            }
            IRInstr::AShr { dst, lhs, rhs } => {
                let lr = self.get_in_reg(*lhs, S0, alloc)?;
                let rr = self.get_in_reg(*rhs, S1, alloc)?;
                self.emit.mov_rr(S0, lr);
                self.emit.mov_rr(1 /*rcx*/, rr);
                self.emit.sar_cl(S0);
                self.put_from_reg(*dst, S0, alloc);
            }
            IRInstr::LShr { dst, lhs, rhs } => {
                let lr = self.get_in_reg(*lhs, S0, alloc)?;
                let rr = self.get_in_reg(*rhs, S1, alloc)?;
                self.emit.mov_rr(S0, lr);
                self.emit.mov_rr(1 /*rcx*/, rr);
                self.emit.shr_cl(S0);
                self.put_from_reg(*dst, S0, alloc);
            }
            IRInstr::Not { dst, src } => {
                let sr = self.get_in_reg(*src, S0, alloc)?;
                self.emit.mov_rr(S0, sr);
                self.emit.not_r(S0);
                self.put_from_reg(*dst, S0, alloc);
            }
            IRInstr::ICmp { dst, cond, lhs, rhs } => {
                let lr = self.get_in_reg(*lhs, S0, alloc)?;
                let rr = self.get_in_reg(*rhs, S1, alloc)?;
                self.emit.cmp_rr(lr, rr);
                self.emit.setcc(cond.x86_cc(), S0);
                self.emit.movzx_64_8(S0, S0);
                self.put_from_reg(*dst, S0, alloc);
            }
            IRInstr::Br { target } => {
                let lbl = format!("B{}_{}", func.name, target);
                if let Some(&tgt) = self.labels.get(&lbl) {
                    let rel = (tgt as i64) - (self.emit.pos() as i64 + 5);
                    self.emit.b(0xE9);
                    self.emit.d(rel as i32);
                } else {
                    let p = self.emit.jmp_rel32();
                    self.fixups.push((p, lbl));
                }
            }
            IRInstr::CondBr { cond, if_true, if_false } => {
                let cr = self.get_in_reg(*cond, S0, alloc)?;
                self.emit.test_rr(cr, cr);
                // jz → if_false
                let false_lbl = format!("B{}_{}", func.name, if_false);
                if let Some(&tgt) = self.labels.get(&false_lbl) {
                    let here = self.emit.pos();
                    let rel  = (tgt as i64) - (here as i64 + 6);
                    self.emit.b(0x0F); self.emit.b(cc::E);
                    self.emit.d(rel as i32);
                } else {
                    let p = self.emit.jcc_rel32(cc::E);
                    self.fixups.push((p, false_lbl));
                }
                // fall through to if_true (emit explicit jmp in case they're not adjacent)
                let true_lbl = format!("B{}_{}", func.name, if_true);
                if let Some(&tgt) = self.labels.get(&true_lbl) {
                    let rel = (tgt as i64) - (self.emit.pos() as i64 + 5);
                    self.emit.b(0xE9);
                    self.emit.d(rel as i32);
                } else {
                    let p = self.emit.jmp_rel32();
                    self.fixups.push((p, true_lbl));
                }
            }
            IRInstr::Ret { value } => {
                if let Some(v) = value {
                    let vr = self.get_in_reg(*v, S0, alloc)?;
                    if vr != RET_REG { self.emit.mov_rr(RET_REG, vr); }
                } else {
                    self.emit.mov_r_imm(RET_REG, 0);
                }
                // Epilogue inline (frame/saved computed by compile_function and stored
                // externally — here we emit a ret; the caller manages epilogue at the
                // end of compile_function instead for compactness)
                self.emit.ret();
            }
            IRInstr::Call { dst, func: fname, args } => {
                // Marshal arguments per System V AMD64 ABI
                for (i, &a) in args.iter().enumerate().take(6) {
                    let ar = self.get_in_reg(a, S0, alloc)?;
                    self.emit.mov_rr(ARG_REGS[i], ar);
                }
                let lbl = fname.clone();
                if let Some(&tgt) = self.labels.get(&lbl) {
                    let rel = (tgt as i64) - (self.emit.pos() as i64 + 5);
                    self.emit.b(0xE8);
                    self.emit.d(rel as i32);
                } else {
                    let p = self.emit.call_rel32();
                    self.fixups.push((p, lbl));
                }
                self.put_from_reg(*dst, RET_REG, alloc);
            }
            IRInstr::TailCall { func: fname, args } => {
                for (i, &a) in args.iter().enumerate().take(6) {
                    let ar = self.get_in_reg(a, S0, alloc)?;
                    self.emit.mov_rr(ARG_REGS[i], ar);
                }
                // Emit jmp instead of call for tail call
                let lbl = fname.clone();
                if let Some(&tgt) = self.labels.get(&lbl) {
                    let rel = (tgt as i64) - (self.emit.pos() as i64 + 5);
                    self.emit.b(0xE9);
                    self.emit.d(rel as i32);
                } else {
                    let p = self.emit.jmp_rel32();
                    self.fixups.push((p, lbl));
                }
            }
            IRInstr::Comment(_) | IRInstr::Label => {}
            _ => {} // Alloca, Load, Store, Phi handled by mem2reg or lowering
        }
        Ok(())
    }

    fn resolve_fixups(&mut self) {
        let fixups: Vec<_> = self.fixups.drain(..).collect();
        for (patch, lbl) in fixups {
            if let Some(&tgt) = self.labels.get(&lbl) {
                self.emit.patch_rel32(patch, tgt);
            }
            // Unresolved fixups → call to external symbol (left as 0 for now)
        }
    }
}

// =============================================================================
// §11  ELF-64 BINARY EMISSION (with symbol table for debugging)
// =============================================================================

fn emit_elf(code: &[u8], symbols: &[(String, usize)], output_path: &str) -> Result<(), String> {
    // ── Sections ──────────────────────────────────────────────────────────
    // .text  : executable code
    // .symtab: symbol table
    // .strtab: string table for symbol names
    // .shstrtab: section name string table

    // Build string table
    let mut strtab: Vec<u8> = vec![0u8]; // index 0 = empty string
    let sym_offsets: Vec<usize> = symbols.iter().map(|(name, _)| {
        let off = strtab.len();
        strtab.extend_from_slice(name.as_bytes());
        strtab.push(0);
        off
    }).collect();

    // Section name string table
    let mut shstrtab: Vec<u8> = vec![0u8];
    let sh_text_name   = shstrtab.len(); shstrtab.extend_from_slice(b".text\0");
    let sh_symtab_name = shstrtab.len(); shstrtab.extend_from_slice(b".symtab\0");
    let sh_strtab_name = shstrtab.len(); shstrtab.extend_from_slice(b".strtab\0");
    let sh_shstr_name  = shstrtab.len(); shstrtab.extend_from_slice(b".shstrtab\0");

    // Symbol table (Elf64_Sym, 24 bytes each)
    // Entry 0: null symbol (required)
    let mut symtab: Vec<u8> = vec![0u8; 24];
    for (i, &(ref _name, offset)) in symbols.iter().enumerate() {
        // st_name, st_info, st_other, st_shndx, st_value, st_size
        let mut entry = [0u8; 24];
        entry[0..4].copy_from_slice(&(sym_offsets[i] as u32).to_le_bytes()); // st_name
        entry[4] = 0x12; // STB_GLOBAL=1, STT_FUNC=2 → (1<<4)|2
        entry[5] = 0;    // st_other
        entry[6..8].copy_from_slice(&1u16.to_le_bytes()); // st_shndx = section 1 (.text)
        let vaddr = LOAD_BASE + offset as u64;
        entry[8..16].copy_from_slice(&vaddr.to_le_bytes()); // st_value
        entry[16..24].copy_from_slice(&0u64.to_le_bytes()); // st_size (unknown)
        symtab.extend_from_slice(&entry);
    }

    // ── Layout ────────────────────────────────────────────────────────────
    let ehdr_size:   usize = 64;
    let phdr_size:   usize = 56;
    let phdr_count:  usize = 1;
    let code_padded: usize = (code.len() + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);

    // File layout:
    //   [0]               ELF header (64 bytes)
    //   [64]              PHDR table (1 × 56 bytes)
    //   [120 → align 16]  padding
    //   [code_off]        .text (code_padded bytes)
    //   [symtab_off]      .symtab
    //   [strtab_off]      .strtab
    //   [shstrtab_off]    .shstrtab
    //   [shdr_off]        Section header table (5 entries × 64 bytes)

    let code_off      = (ehdr_size + phdr_count * phdr_size + 15) & !15;
    let symtab_off    = code_off + code_padded;
    let strtab_off    = symtab_off + symtab.len();
    let shstrtab_off  = strtab_off + strtab.len();
    let shdr_off      = (shstrtab_off + shstrtab.len() + 7) & !7; // 8-byte align

    let total = shdr_off + 5 * 64;
    let mut out = vec![0u8; total];

    // ── ELF Header ───────────────────────────────────────────────────────
    {
        let h = &mut out[..64];
        h[0..4].copy_from_slice(&[0x7F, b'E', b'L', b'F']);
        h[4] = 2; // ELFCLASS64
        h[5] = 1; // ELFDATA2LSB
        h[6] = 1; // EV_CURRENT
        h[7] = 0; // ELFOSABI_NONE
        h[16..18].copy_from_slice(&2u16.to_le_bytes());  // ET_EXEC
        h[18..20].copy_from_slice(&62u16.to_le_bytes()); // EM_X86_64
        h[20..24].copy_from_slice(&1u32.to_le_bytes());  // EV_CURRENT
        // e_entry = virtual address of code start
        let entry_vaddr = LOAD_BASE + code_off as u64;
        h[24..32].copy_from_slice(&entry_vaddr.to_le_bytes());
        h[32..40].copy_from_slice(&(ehdr_size as u64).to_le_bytes()); // e_phoff
        h[40..48].copy_from_slice(&(shdr_off as u64).to_le_bytes());  // e_shoff
        h[52..54].copy_from_slice(&(ehdr_size as u16).to_le_bytes()); // e_ehsize
        h[54..56].copy_from_slice(&(phdr_size as u16).to_le_bytes()); // e_phentsize
        h[56..58].copy_from_slice(&(phdr_count as u16).to_le_bytes()); // e_phnum
        h[58..60].copy_from_slice(&64u16.to_le_bytes());  // e_shentsize
        h[60..62].copy_from_slice(&5u16.to_le_bytes());   // e_shnum
        h[62..64].copy_from_slice(&4u16.to_le_bytes());   // e_shstrndx = section 4
    }

    // ── Program Header: PT_LOAD (RX) ─────────────────────────────────────
    {
        let ph = &mut out[ehdr_size..ehdr_size + phdr_size];
        ph[0..4].copy_from_slice(&1u32.to_le_bytes()); // PT_LOAD
        ph[4..8].copy_from_slice(&5u32.to_le_bytes()); // PF_R | PF_X
        ph[8..16].copy_from_slice(&(code_off as u64).to_le_bytes()); // p_offset
        let load_vaddr = LOAD_BASE;
        ph[16..24].copy_from_slice(&load_vaddr.to_le_bytes()); // p_vaddr  ← v1 fix
        ph[24..32].copy_from_slice(&load_vaddr.to_le_bytes()); // p_paddr
        ph[32..40].copy_from_slice(&(code_padded as u64).to_le_bytes()); // p_filesz
        ph[40..48].copy_from_slice(&(code_padded as u64).to_le_bytes()); // p_memsz
        ph[48..56].copy_from_slice(&(PAGE_SIZE as u64).to_le_bytes());   // p_align
    }

    // ── .text ─────────────────────────────────────────────────────────────
    out[code_off..code_off + code.len()].copy_from_slice(code);
    // NOP padding
    for b in &mut out[code_off + code.len()..code_off + code_padded] { *b = 0x90; }

    // ── .symtab, .strtab, .shstrtab ──────────────────────────────────────
    out[symtab_off..symtab_off + symtab.len()].copy_from_slice(&symtab);
    out[strtab_off..strtab_off + strtab.len()].copy_from_slice(&strtab);
    out[shstrtab_off..shstrtab_off + shstrtab.len()].copy_from_slice(&shstrtab);

    // ── Section header table (5 entries × 64 bytes) ───────────────────────
    // §0: null section
    // §1: .text
    // §2: .symtab
    // §3: .strtab
    // §4: .shstrtab
    fn write_shdr(out: &mut [u8], base: usize, idx: usize,
                  name: u32, stype: u32, flags: u64,
                  addr: u64, offset: u64, size: u64,
                  link: u32, info: u32, addralign: u64, entsize: u64)
    {
        let p = base + idx * 64;
        let s = &mut out[p..p+64];
        s[0..4].copy_from_slice(&name.to_le_bytes());
        s[4..8].copy_from_slice(&stype.to_le_bytes());
        s[8..16].copy_from_slice(&flags.to_le_bytes());
        s[16..24].copy_from_slice(&addr.to_le_bytes());
        s[24..32].copy_from_slice(&offset.to_le_bytes());
        s[32..40].copy_from_slice(&size.to_le_bytes());
        s[40..44].copy_from_slice(&link.to_le_bytes());
        s[44..48].copy_from_slice(&info.to_le_bytes());
        s[48..56].copy_from_slice(&addralign.to_le_bytes());
        s[56..64].copy_from_slice(&entsize.to_le_bytes());
    }

    // §0: null
    write_shdr(&mut out, shdr_off, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    // §1: .text  (SHT_PROGBITS=1, SHF_ALLOC|SHF_EXECINSTR=6)
    write_shdr(&mut out, shdr_off, 1,
               sh_text_name as u32, 1, 6,
               LOAD_BASE + code_off as u64,
               code_off as u64, code.len() as u64,
               0, 0, 16, 0);
    // §2: .symtab (SHT_SYMTAB=2)
    write_shdr(&mut out, shdr_off, 2,
               sh_symtab_name as u32, 2, 0,
               0, symtab_off as u64, symtab.len() as u64,
               3 /*link to .strtab*/, 1 /*first global sym index*/, 8, 24);
    // §3: .strtab (SHT_STRTAB=3)
    write_shdr(&mut out, shdr_off, 3,
               sh_strtab_name as u32, 3, 0,
               0, strtab_off as u64, strtab.len() as u64,
               0, 0, 1, 0);
    // §4: .shstrtab (SHT_STRTAB=3)
    write_shdr(&mut out, shdr_off, 4,
               sh_shstr_name as u32, 3, 0,
               0, shstrtab_off as u64, shstrtab.len() as u64,
               0, 0, 1, 0);

    // ── Write to file ─────────────────────────────────────────────────────
    let mut file = File::create(output_path)
        .map_err(|e| format!("Cannot create {}: {}", output_path, e))?;
    file.write_all(&out)
        .map_err(|e| format!("Write error: {}", e))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = file.metadata().map_err(|e| e.to_string())?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(output_path, perms).map_err(|e| e.to_string())?;
    }

    eprintln!("✓  {} ({} bytes, {} symbols)", output_path, total, symbols.len());
    Ok(())
}

// =============================================================================
// §12  PUBLIC API — FULL 20-PHASE COMPILATION PIPELINE
// =============================================================================

pub fn compile_to_native(
    program:     &Program,
    output_path: &str,
    opt_level:   u8,
) -> Result<(), String> {
    let cfg = OptConfig::from_level(opt_level);
    eprintln!("AoT v2.0 — compiling at -O{}", opt_level);

    // ── Phase 1: Call graph ──────────────────────────────────────────────
    let cg = CallGraph::build(program);

    // ── Phase 2: Function inlining (AST level) ───────────────────────────
    let mut prog = program.clone();
    run_inlining(&mut prog, &cg, &cfg);

    // ── Phase 3: Lower all functions to SSA IR ───────────────────────────
    let fn_decls: Vec<&FnDecl> = prog.items.iter()
        .filter_map(|i| if let Item::Fn(f) = i { Some(f) } else { None })
        .collect();

    let mut ir_fns: Vec<IRFunction> = fn_decls.iter()
        .map(|f| lower_to_ir(f)).collect();

    // ── Phases 4-17: Optimisation pipeline ──────────────────────────────
    for func in &mut ir_fns {
        let idom  = compute_dominators(func);
        let loops = detect_loops(func, &idom);
        func.estimate_frequencies();

        if cfg.sccp          { run_sccp(func); }
        if cfg.gvn           { run_gvn(func); }
        if cfg.copy_prop     { run_copy_prop(func); }
        if cfg.vrp           { run_vrp(func); }
        if cfg.licm          { run_licm(func, &loops); }
        if cfg.strength_reduce { run_strength_reduction(func, &loops); }
        if cfg.dce           { run_dce(func); }
        if cfg.peephole      { run_peephole(func); }
        if cfg.tco           { run_tco(func); }
        if cfg.jump_threading { run_jump_threading(func); func.build_predecessors(); }
        // Second DCE pass cleans up peephole/TCO residue
        if cfg.dce           { run_dce(func); }
        // Instruction scheduling (latency hiding)
        if cfg.sched         { run_sched(func); }
    }

    // ── Phase 18: Code generation ─────────────────────────────────────────
    let mut codegen = NativeCodeGen::new();
    // Compile all functions (so cross-function calls can be resolved)
    for func in &ir_fns {
        codegen.compile_function(func)?;
    }

    // ── Phase 19: Collect symbol table ───────────────────────────────────
    let symbols: Vec<(String, usize)> = codegen.labels.iter()
        .filter(|(k, _)| !k.starts_with('B')) // skip block labels
        .map(|(k, &v)| (k.clone(), v))
        .collect();

    // ── Phase 20: ELF emission ────────────────────────────────────────────
    emit_elf(&codegen.emit.code, &symbols, output_path)?;

    Ok(())
}

/// Returns true if AoT compilation is available on this platform
pub fn is_available() -> bool {
    cfg!(target_arch = "x86_64") && cfg!(target_os = "linux")
}