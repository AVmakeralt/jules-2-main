// =============================================================================
// jules/src/compiler/ir_typeck.rs
//
// IR-based Type Checker — validates type correctness on FlatIrModule.
//
// This type checker operates on the flat instruction IR, validating that the
// IR's type annotations are internally consistent. It catches issues like
// "cannot add () and i64" BEFORE execution.
//
// Architecture
// ────────────
//   1. Build a type table:  ValueId → IrType   (from params, dsts, constants)
//   2. Walk every instruction and validate operand types against rules
//   3. Accumulate diagnostics (errors + warnings) into IrTypeckResult
//
// The checker is deliberately conservative: unknown types (IrType::Unknown)
// are treated as compatible with everything to avoid false positives on
// code that hasn't been fully typed yet.
// =============================================================================

use std::collections::HashMap;

use crate::compiler::ast::{BinOpKind, UnOpKind};
use crate::compiler::ir::{FlatIrFunction, FlatIrModule, IrOp, IrType, ValueId};
use crate::compiler::lexer::Span;

// =============================================================================
// PUBLIC RESULT TYPES
// =============================================================================

/// The result of IR type checking.
#[derive(Debug)]
pub struct IrTypeckResult {
    /// All diagnostics (errors and warnings) emitted during checking.
    pub diagnostics: Vec<IrTypeDiag>,
    /// Maps ValueId (as u32) to its inferred / annotated IrType.
    pub type_table: HashMap<u32, IrType>,
    /// Number of error diagnostics.
    pub errors: usize,
    /// Number of warning diagnostics.
    pub warnings: usize,
}

impl IrTypeckResult {
    /// Returns true if any errors were found.
    pub fn has_errors(&self) -> bool {
        self.errors > 0
    }
}

/// A single IR type-check diagnostic.
#[derive(Debug)]
pub struct IrTypeDiag {
    /// Source span where the issue was detected.
    pub span: Span,
    /// Category of the diagnostic.
    pub kind: IrTypeDiagKind,
    /// Human-readable error message.
    pub message: String,
}

/// Category of IR type-check diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrTypeDiagKind {
    /// Two types that should match don't.
    TypeMismatch,
    /// A referenced ValueId has no known type.
    UndefinedValue,
    /// A cast between incompatible types.
    InvalidCast,
    /// A non-bool value used as a branch condition.
    NonBoolCondition,
    /// A return value type doesn't match the function return type.
    ReturnTypeMismatch,
    /// Wrong number of arguments in a call.
    ArgCountMismatch,
    /// Phi node incoming values have different types.
    PhiTypeMismatch,
    /// A constant value doesn't fit in its declared type.
    ConstOverflow,
    /// A unary operator applied to a non-numeric type.
    InvalidUnaryOperand,
    /// A binary operator applied to incompatible types.
    InvalidBinOp,
}

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

/// Validate type correctness on a FlatIrModule.
///
/// Walks all functions and their instructions, builds a ValueId → IrType table,
/// and checks that every operation's type annotations are internally consistent.
pub fn ir_typeck(module: &FlatIrModule) -> IrTypeckResult {
    let mut ctx = TypeckCtx::new();

    // Build intrinsic signature table for call validation.
    for intrinsic in &module.intrinsics {
        ctx.intrinsic_sigs.insert(
            intrinsic.name.clone(),
            IntrinsicSig {
                param_types: intrinsic.param_types.clone(),
                ret_type: intrinsic.ret_type.clone(),
            },
        );
    }

    // Check each function.
    for func in &module.functions {
        ctx.check_function(func);
    }

    IrTypeckResult {
        diagnostics: ctx.diagnostics,
        type_table: ctx.type_table,
        errors: ctx.errors,
        warnings: ctx.warnings,
    }
}

// =============================================================================
// INTERNAL: Intrinsics signature table
// =============================================================================

/// Stored signature for an intrinsic function.
#[derive(Debug, Clone)]
struct IntrinsicSig {
    param_types: Vec<IrType>,
    ret_type: IrType,
}

// =============================================================================
// INTERNAL: TYPE-CHECK CONTEXT
// =============================================================================

struct TypeckCtx {
    /// ValueId → IrType mapping built up during checking.
    type_table: HashMap<u32, IrType>,
    /// Accumulated diagnostics.
    diagnostics: Vec<IrTypeDiag>,
    /// Error count.
    errors: usize,
    /// Warning count.
    warnings: usize,
    /// Current function return type (set during function checking).
    current_ret_ty: Option<IrType>,
    /// Intrinsic signatures for call validation.
    intrinsic_sigs: HashMap<String, IntrinsicSig>,
    /// Function signatures by name (for Call validation).
    func_sigs: HashMap<String, FuncSig>,
}

/// Stored signature for a flat IR function.
#[derive(Debug, Clone)]
struct FuncSig {
    param_types: Vec<IrType>,
    ret_type: IrType,
}

impl TypeckCtx {
    fn new() -> Self {
        TypeckCtx {
            type_table: HashMap::new(),
            diagnostics: Vec::new(),
            errors: 0,
            warnings: 0,
            current_ret_ty: None,
            intrinsic_sigs: HashMap::new(),
            func_sigs: HashMap::new(),
        }
    }

    // ── Diagnostic helpers ─────────────────────────────────────────────

    fn emit_error(&mut self, span: Span, kind: IrTypeDiagKind, message: String) {
        self.diagnostics.push(IrTypeDiag { span, kind, message });
        self.errors += 1;
    }

    fn emit_warning(&mut self, span: Span, kind: IrTypeDiagKind, message: String) {
        self.diagnostics.push(IrTypeDiag { span, kind, message });
        self.warnings += 1;
    }

    // ── Type table helpers ─────────────────────────────────────────────

    /// Record the type of a value.
    fn record_type(&mut self, id: ValueId, ty: IrType) {
        self.type_table.insert(id.0, ty);
    }

    /// Look up the type of a value. Returns None if unknown.
    fn lookup_type(&self, id: ValueId) -> Option<&IrType> {
        self.type_table.get(&id.0)
    }

    /// Look up the type of a value, or IrType::Unknown if not found.
    /// Also emits an UndefinedValue error if not found.
    fn require_type(&mut self, id: ValueId, span: Span) -> IrType {
        match self.type_table.get(&id.0) {
            Some(ty) => ty.clone(),
            None => {
                self.emit_error(
                    span,
                    IrTypeDiagKind::UndefinedValue,
                    format!("undefined value v{}", id.0),
                );
                IrType::Unknown
            }
        }
    }

    // ── Function checking ──────────────────────────────────────────────

    fn check_function(&mut self, func: &FlatIrFunction) {
        // Set the return type for Ret validation.
        self.current_ret_ty = Some(func.ret_ty.clone());

        // Register function params in the type table.
        for (vid, ty) in &func.params {
            self.record_type(*vid, ty.clone());
        }

        // Build a function signature table for call validation.
        let sig = FuncSig {
            param_types: func.params.iter().map(|(_, ty)| ty.clone()).collect(),
            ret_type: func.ret_ty.clone(),
        };
        self.func_sigs.insert(func.name.clone(), sig);

        // Walk all blocks and instructions.
        for block in &func.blocks {
            self.check_block(block);
        }

        self.current_ret_ty = None;
    }

    fn check_block(&mut self, block: &crate::compiler::ir::FlatBlock) {
        for instr in &block.instrs {
            self.check_instr(instr);
        }
    }

    // ── Instruction checking ───────────────────────────────────────────

    fn check_instr(&mut self, instr: &crate::compiler::ir::IrInstr) {
        let span = instr.span;

        // If the instruction defines a dst value, we'll record its type.
        let result_ty = self.check_op(&instr.op, span);

        // Record the result type for the destination value.
        if let Some(dst) = instr.dst {
            if let Some(ty) = result_ty {
                self.record_type(dst, ty);
            }
        }
    }

    /// Check an IrOp and return the result type (if determinable).
    fn check_op(&mut self, op: &IrOp, span: Span) -> Option<IrType> {
        match op {
            // ── Constants ──────────────────────────────────────────────
            IrOp::ConstInt { value, ty } => {
                self.check_const_int(*value, ty, span);
                Some(ty.clone())
            }
            IrOp::ConstFloat { bits: _, ty } => {
                self.check_const_float(ty, span);
                Some(ty.clone())
            }
            IrOp::ConstBool { value: _ } => Some(IrType::Bool),
            IrOp::ConstStr { idx: _ } => Some(IrType::String),
            IrOp::ConstUnit => Some(IrType::Unit),

            // ── Binary operators ──────────────────────────────────────
            IrOp::BinOp { op: binop, lhs, rhs } => {
                Some(self.check_binop(*binop, *lhs, *rhs, span))
            }

            // ── Unary operators ───────────────────────────────────────
            IrOp::UnOp { op: unop, operand } => {
                Some(self.check_unop(*unop, *operand, span))
            }

            // ── Memory operations ─────────────────────────────────────
            IrOp::Alloca { ty, align: _ } => {
                // Alloca returns a pointer (Ref) to the allocated type.
                Some(IrType::Ref(Box::new(ty.clone())))
            }
            IrOp::Store { ptr, value } => {
                self.check_store(*ptr, *value, span);
                None // Store doesn't produce a value.
            }
            IrOp::Load { ptr, ty } => {
                self.check_load(*ptr, ty, span);
                Some(ty.clone())
            }
            IrOp::Move { src } => {
                // Move produces the same type as the source.
                Some(self.require_type(*src, span))
            }
            IrOp::Copy { src } => {
                // Copy produces the same type as the source.
                Some(self.require_type(*src, span))
            }

            // ── Control flow ──────────────────────────────────────────
            IrOp::Ret { value } => {
                self.check_ret(value, span);
                None
            }
            IrOp::Nop => None,
            IrOp::CondBr { cond, if_true: _, if_false: _ } => {
                self.check_condbr(*cond, span);
                None
            }
            IrOp::Jump { target: _ } => None,

            // ── Phi ───────────────────────────────────────────────────
            IrOp::Phi { incoming } => {
                Some(self.check_phi(incoming, span))
            }

            // ── Tensor operations ─────────────────────────────────────
            IrOp::MatMul { lhs, rhs }
            | IrOp::HadamardMul { lhs, rhs }
            | IrOp::HadamardDiv { lhs, rhs }
            | IrOp::TensorConcat { lhs, rhs }
            | IrOp::KronProd { lhs, rhs }
            | IrOp::OuterProd { lhs, rhs } => {
                // Tensor ops: both operands should be tensor-like.
                // We don't enforce strict shape checking here (that's the
                // AST type checker's job), but we verify both operands exist.
                let _lhs_ty = self.require_type(*lhs, span);
                let _rhs_ty = self.require_type(*rhs, span);
                // Result type is unknown without deeper shape analysis.
                Some(IrType::Unknown)
            }

            // ── Calls ─────────────────────────────────────────────────
            IrOp::Call { func, args } => {
                Some(self.check_call(func, args, span))
            }
            IrOp::Intrinsic { name, args } => {
                Some(self.check_intrinsic_call(name, args, span))
            }

            // ── Parallelism ──────────────────────────────────────────
            IrOp::ParallelStart { region_id: _ } => None,
            IrOp::ParallelEnd { region_id: _ } => None,

            // ── Region ────────────────────────────────────────────────
            IrOp::RegionAlloc { region: _, ty } => {
                Some(IrType::Ref(Box::new(ty.clone())))
            }

            // ── Tasks ─────────────────────────────────────────────────
            IrOp::TaskSpawn { func, args, ownership: _ } => {
                // Validate args exist, return type from function sig if known.
                for arg in args {
                    self.require_type(*arg, span);
                }
                match self.func_sigs.get(func) {
                    Some(sig) => Some(sig.ret_type.clone()),
                    None => Some(IrType::Unknown),
                }
            }
            IrOp::TaskJoin { task: _ } => {
                // TaskJoin result type is unknown without deeper analysis.
                Some(IrType::Unknown)
            }

            // ── Type operations ───────────────────────────────────────
            IrOp::TypeCheck { value: _, expected: _ } => {
                // TypeCheck returns Bool.
                Some(IrType::Bool)
            }
            IrOp::Cast { src, target_ty } => {
                self.check_cast(*src, target_ty, span);
                Some(target_ty.clone())
            }

            // ── Effect emission ───────────────────────────────────────
            IrOp::Emit { effect: _, value } => {
                self.require_type(*value, span);
                Some(IrType::Unit)
            }
        }
    }

    // ── Individual check implementations ────────────────────────────────

    /// Check that a ConstInt value fits in its declared type.
    fn check_const_int(&mut self, value: i128, ty: &IrType, span: Span) {
        match ty {
            IrType::Int { width, signed } => {
                let bits = *width as u32;
                if bits == 0 {
                    self.emit_error(
                        span,
                        IrTypeDiagKind::ConstOverflow,
                        format!("integer type i{}/u{} has zero width", width, width),
                    );
                    return;
                }
                if bits > 128 {
                    // Can't represent > 128 bits in i128, but we allow it
                    // since the value is stored as i128.
                    return;
                }
                if *signed {
                    // Signed: range is [-2^(bits-1), 2^(bits-1) - 1]
                    let min = -(1i128 << (bits - 1));
                    let max = (1i128 << (bits - 1)) - 1;
                    if value < min || value > max {
                        self.emit_error(
                            span,
                            IrTypeDiagKind::ConstOverflow,
                            format!(
                                "constant value {} does not fit in i{} (range {}..={})",
                                value, width, min, max
                            ),
                        );
                    }
                } else {
                    // Unsigned: range is [0, 2^bits - 1]
                    if value < 0 {
                        self.emit_error(
                            span,
                            IrTypeDiagKind::ConstOverflow,
                            format!(
                                "negative constant {} cannot fit in unsigned type u{}",
                                value, width
                            ),
                        );
                    } else {
                        let max = (1u128 << bits) - 1;
                        if (value as u128) > max {
                            self.emit_error(
                                span,
                                IrTypeDiagKind::ConstOverflow,
                                format!(
                                    "constant value {} does not fit in u{} (range 0..={})",
                                    value, width, max
                                ),
                            );
                        }
                    }
                }
            }
            IrType::Unknown => {
                // Allow unknown types — they'll be resolved later.
            }
            other => {
                self.emit_error(
                    span,
                    IrTypeDiagKind::TypeMismatch,
                    format!("ConstInt has non-integer type {}", other),
                );
            }
        }
    }

    /// Check that a ConstFloat has a Float type.
    fn check_const_float(&mut self, ty: &IrType, span: Span) {
        match ty {
            IrType::Float { width: _ } => {}
            IrType::Unknown => {}
            other => {
                self.emit_error(
                    span,
                    IrTypeDiagKind::TypeMismatch,
                    format!("ConstFloat has non-float type {}", other),
                );
            }
        }
    }

    /// Check a binary operation: both sides must be same type (Int/Int or Float/Float), no mixing.
    fn check_binop(&mut self, op: BinOpKind, lhs: ValueId, rhs: ValueId, span: Span) -> IrType {
        let lhs_ty = self.require_type(lhs, span);
        let rhs_ty = self.require_type(rhs, span);

        // Logical And/Or require Bool operands.
        match op {
            BinOpKind::And | BinOpKind::Or => {
                if !is_bool_or_unknown(&lhs_ty) {
                    self.emit_error(
                        span,
                        IrTypeDiagKind::InvalidBinOp,
                        format!(
                            "logical {:?} requires bool operands, got {}",
                            op, lhs_ty
                        ),
                    );
                }
                if !is_bool_or_unknown(&rhs_ty) {
                    self.emit_error(
                        span,
                        IrTypeDiagKind::InvalidBinOp,
                        format!(
                            "logical {:?} requires bool operands, got {}",
                            op, rhs_ty
                        ),
                    );
                }
                return IrType::Bool;
            }
            _ => {}
        }

        // Comparison operators require same comparable type, result is Bool.
        match op {
            BinOpKind::Eq
            | BinOpKind::Ne
            | BinOpKind::Lt
            | BinOpKind::Le
            | BinOpKind::Gt
            | BinOpKind::Ge => {
                if !types_compatible(&lhs_ty, &rhs_ty) {
                    self.emit_error(
                        span,
                        IrTypeDiagKind::TypeMismatch,
                        format!(
                            "comparison {:?} requires same types, got {} and {}",
                            op, lhs_ty, rhs_ty
                        ),
                    );
                }
                if !is_comparable(&lhs_ty) {
                    self.emit_error(
                        span,
                        IrTypeDiagKind::InvalidBinOp,
                        format!(
                            "comparison {:?} on non-comparable type {}",
                            op, lhs_ty
                        ),
                    );
                }
                return IrType::Bool;
            }
            _ => {}
        }

        // Arithmetic/bitwise operators: both sides must be same numeric type.
        if !types_compatible(&lhs_ty, &rhs_ty) {
            self.emit_error(
                span,
                IrTypeDiagKind::TypeMismatch,
                format!(
                    "binary {:?} requires same types, got {} and {}",
                    op, lhs_ty, rhs_ty
                ),
            );
        }

        // Check that the operands are numeric (Int or Float).
        // Bitwise ops require Int.
        match op {
            BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor
            | BinOpKind::Shl | BinOpKind::Shr => {
                if !is_int_or_unknown(&lhs_ty) {
                    self.emit_error(
                        span,
                        IrTypeDiagKind::InvalidBinOp,
                        format!(
                            "bitwise {:?} requires integer operands, got {}",
                            op, lhs_ty
                        ),
                    );
                }
            }
            BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul
            | BinOpKind::Div | BinOpKind::Rem | BinOpKind::FloorDiv => {
                if !is_numeric_or_unknown(&lhs_ty) {
                    self.emit_error(
                        span,
                        IrTypeDiagKind::InvalidBinOp,
                        format!(
                            "arithmetic {:?} requires numeric operands, got {}",
                            op, lhs_ty
                        ),
                    );
                }
            }
            _ => {}
        }

        // Result type is the same as the operand type.
        lhs_ty
    }

    /// Check a unary operation: operand must be numeric for Neg, Bool for Not.
    fn check_unop(&mut self, op: UnOpKind, operand: ValueId, span: Span) -> IrType {
        let operand_ty = self.require_type(operand, span);

        match op {
            UnOpKind::Neg => {
                if !is_numeric_or_unknown(&operand_ty) {
                    self.emit_error(
                        span,
                        IrTypeDiagKind::InvalidUnaryOperand,
                        format!("negation requires numeric operand, got {}", operand_ty),
                    );
                }
                operand_ty
            }
            UnOpKind::Not => {
                if !is_bool_or_unknown(&operand_ty) && !is_int_or_unknown(&operand_ty) {
                    self.emit_error(
                        span,
                        IrTypeDiagKind::InvalidUnaryOperand,
                        format!(
                            "logical/bitwise not requires bool or integer operand, got {}",
                            operand_ty
                        ),
                    );
                }
                operand_ty
            }
            UnOpKind::Deref => {
                // Deref: operand should be Ref or MutRef.
                match &operand_ty {
                    IrType::Ref(inner) | IrType::MutRef(inner) => {
                        // Deref produces the inner type of the reference.
                        // If the inner type is itself a reference, deref peels
                        // one layer to get the pointed-to value type.
                        let inner_ty = *inner.clone();
                        match inner_ty {
                            IrType::Ref(inner2) | IrType::MutRef(inner2) => *inner2,
                            other => other,
                        }
                    }
                    IrType::Unknown => IrType::Unknown,
                    other => {
                        self.emit_error(
                            span,
                            IrTypeDiagKind::InvalidUnaryOperand,
                            format!("deref requires reference operand, got {}", other),
                        );
                        IrType::Unknown
                    }
                }
            }
            UnOpKind::Ref => {
                // Ref: produces &T
                IrType::Ref(Box::new(operand_ty))
            }
            UnOpKind::RefMut => {
                // RefMut: produces &mut T
                IrType::MutRef(Box::new(operand_ty))
            }
        }
    }

    /// Check Store: value type must match ptr's target type.
    fn check_store(&mut self, ptr: ValueId, value: ValueId, span: Span) {
        let ptr_ty = self.require_type(ptr, span);
        let value_ty = self.require_type(value, span);

        match &ptr_ty {
            IrType::Ref(inner) | IrType::MutRef(inner) => {
                if !types_compatible(inner, &value_ty) {
                    self.emit_error(
                        span,
                        IrTypeDiagKind::TypeMismatch,
                        format!(
                            "store: value type {} does not match ptr target type {}",
                            value_ty, inner
                        ),
                    );
                }
            }
            IrType::Unknown => {
                // Allow unknown ptr types — may be resolved later.
            }
            other => {
                self.emit_error(
                    span,
                    IrTypeDiagKind::TypeMismatch,
                    format!("store: ptr must be a reference type, got {}", other),
                );
            }
        }
    }

    /// Check Load: result type must match ptr's target type.
    fn check_load(&mut self, ptr: ValueId, expected_ty: &IrType, span: Span) {
        let ptr_ty = self.require_type(ptr, span);

        match &ptr_ty {
            IrType::Ref(inner) | IrType::MutRef(inner) => {
                if !types_compatible(inner, expected_ty) {
                    self.emit_error(
                        span,
                        IrTypeDiagKind::TypeMismatch,
                        format!(
                            "load: result type {} does not match ptr target type {}",
                            expected_ty, inner
                        ),
                    );
                }
            }
            IrType::Unknown => {
                // Allow unknown ptr types.
            }
            other => {
                self.emit_error(
                    span,
                    IrTypeDiagKind::TypeMismatch,
                    format!("load: ptr must be a reference type, got {}", other),
                );
            }
        }
    }

    /// Check Ret: value type must match function return type.
    fn check_ret(&mut self, value: &Option<ValueId>, span: Span) {
        let ret_ty = match &self.current_ret_ty {
            Some(ty) => ty.clone(),
            None => return, // No function context — can't check.
        };

        match value {
            Some(vid) => {
                let value_ty = self.require_type(*vid, span);
                if !types_compatible(&value_ty, &ret_ty) {
                    self.emit_error(
                        span,
                        IrTypeDiagKind::ReturnTypeMismatch,
                        format!(
                            "return type {} does not match function return type {}",
                            value_ty, ret_ty
                        ),
                    );
                }
            }
            None => {
                // Returning nothing — function must return Unit.
                if !is_unit_or_unknown(&ret_ty) {
                    self.emit_error(
                        span,
                        IrTypeDiagKind::ReturnTypeMismatch,
                        format!(
                            "empty return in function with non-unit return type {}",
                            ret_ty
                        ),
                    );
                }
            }
        }
    }

    /// Check CondBr: condition must be Bool.
    fn check_condbr(&mut self, cond: ValueId, span: Span) {
        let cond_ty = self.require_type(cond, span);
        if !is_bool_or_unknown(&cond_ty) {
            self.emit_error(
                span,
                IrTypeDiagKind::NonBoolCondition,
                format!("conditional branch requires bool condition, got {}", cond_ty),
            );
        }
    }

    /// Check Phi: all incoming values must have the same type.
    fn check_phi(&mut self, incoming: &[(crate::compiler::ir::BlockId, ValueId)], span: Span) -> IrType {
        if incoming.is_empty() {
            return IrType::Unknown;
        }

        let mut first_ty: Option<IrType> = None;

        for (_block, vid) in incoming {
            let ty = self.require_type(*vid, span);
            match &first_ty {
                None => first_ty = Some(ty),
                Some(expected) => {
                    if !types_compatible(expected, &ty) {
                        self.emit_error(
                            span,
                            IrTypeDiagKind::PhiTypeMismatch,
                            format!(
                                "phi node has incoming values of different types: {} and {}",
                                expected, ty
                            ),
                        );
                    }
                }
            }
        }

        first_ty.unwrap_or(IrType::Unknown)
    }

    /// Check Call: args must match function signature.
    fn check_call(&mut self, func_name: &str, args: &[ValueId], span: Span) -> IrType {
        // Look up the function signature.
        let sig = self.func_sigs.get(func_name).cloned();

        // Also check intrinsics.
        let intrinsic_sig = self.intrinsic_sigs.get(func_name).cloned();

        match (sig, intrinsic_sig) {
            (Some(sig), _) => {
                // Check argument count.
                if args.len() != sig.param_types.len() {
                    self.emit_error(
                        span,
                        IrTypeDiagKind::ArgCountMismatch,
                        format!(
                            "call to '{}' expects {} arguments, got {}",
                            func_name,
                            sig.param_types.len(),
                            args.len()
                        ),
                    );
                }

                // Check argument types.
                for (i, arg) in args.iter().enumerate() {
                    let arg_ty = self.require_type(*arg, span);
                    if let Some(expected) = sig.param_types.get(i) {
                        if !types_compatible(expected, &arg_ty) {
                            self.emit_error(
                                span,
                                IrTypeDiagKind::TypeMismatch,
                                format!(
                                    "argument {} of call to '{}': expected {}, got {}",
                                    i, func_name, expected, arg_ty
                                ),
                            );
                        }
                    }
                }

                sig.ret_type
            }
            (None, Some(intrinsic_sig)) => {
                // Check argument count.
                if args.len() != intrinsic_sig.param_types.len() {
                    self.emit_error(
                        span,
                        IrTypeDiagKind::ArgCountMismatch,
                        format!(
                            "call to intrinsic '{}' expects {} arguments, got {}",
                            func_name,
                            intrinsic_sig.param_types.len(),
                            args.len()
                        ),
                    );
                }

                // Check argument types.
                for (i, arg) in args.iter().enumerate() {
                    let arg_ty = self.require_type(*arg, span);
                    if let Some(expected) = intrinsic_sig.param_types.get(i) {
                        if !types_compatible(expected, &arg_ty) {
                            self.emit_error(
                                span,
                                IrTypeDiagKind::TypeMismatch,
                                format!(
                                    "argument {} of intrinsic '{}': expected {}, got {}",
                                    i, func_name, expected, arg_ty
                                ),
                            );
                        }
                    }
                }

                intrinsic_sig.ret_type
            }
            (None, None) => {
                // Unknown function — validate args exist but don't check types.
                for arg in args {
                    self.require_type(*arg, span);
                }
                IrType::Unknown
            }
        }
    }

    /// Check Intrinsic call: args must match intrinsic signature.
    fn check_intrinsic_call(&mut self, name: &str, args: &[ValueId], span: Span) -> IrType {
        // Clone the signature to avoid borrowing self while emitting errors.
        let sig = self.intrinsic_sigs.get(name).cloned();
        match sig {
            Some(sig) => {
                if args.len() != sig.param_types.len() {
                    self.emit_error(
                        span,
                        IrTypeDiagKind::ArgCountMismatch,
                        format!(
                            "intrinsic '{}' expects {} arguments, got {}",
                            name,
                            sig.param_types.len(),
                            args.len()
                        ),
                    );
                }

                for (i, arg) in args.iter().enumerate() {
                    let arg_ty = self.require_type(*arg, span);
                    if let Some(expected) = sig.param_types.get(i) {
                        if !types_compatible(expected, &arg_ty) {
                            self.emit_error(
                                span,
                                IrTypeDiagKind::TypeMismatch,
                                format!(
                                    "argument {} of intrinsic '{}': expected {}, got {}",
                                    i, name, expected, arg_ty
                                ),
                            );
                        }
                    }
                }

                sig.ret_type
            }
            None => {
                // Unknown intrinsic — just validate args exist.
                for arg in args {
                    self.require_type(*arg, span);
                }
                IrType::Unknown
            }
        }
    }

    /// Check Cast: types must be compatible for casting.
    fn check_cast(&mut self, src: ValueId, target_ty: &IrType, span: Span) {
        let src_ty = self.require_type(src, span);

        // Allow casts between numeric types (int↔int, int↔float, float↔float).
        // Allow casts between Ref/MutRef variants.
        // Disallow nonsensical casts like Bool → Tensor.
        if types_cast_compatible(&src_ty, target_ty) {
            return;
        }

        self.emit_error(
            span,
            IrTypeDiagKind::InvalidCast,
            format!("cannot cast from {} to {}", src_ty, target_ty),
        );
    }
}

// =============================================================================
// TYPE COMPATIBILITY HELPERS
// =============================================================================

/// Returns true if two types are compatible (same type, or either is Unknown).
fn types_compatible(a: &IrType, b: &IrType) -> bool {
    if matches!(a, IrType::Unknown) || matches!(b, IrType::Unknown) {
        return true;
    }
    a == b
}

/// Returns true if a cast from `src` to `target` is valid.
fn types_cast_compatible(src: &IrType, target: &IrType) -> bool {
    // Unknown is always allowed.
    if matches!(src, IrType::Unknown) || matches!(target, IrType::Unknown) {
        return true;
    }

    // Same type → always valid.
    if src == target {
        return true;
    }

    // Numeric to numeric.
    if is_numeric(src) && is_numeric(target) {
        return true;
    }

    // Int to Int (different widths).
    if matches!(src, IrType::Int { .. }) && matches!(target, IrType::Int { .. }) {
        return true;
    }

    // Float to Float (different widths).
    if matches!(src, IrType::Float { .. }) && matches!(target, IrType::Float { .. }) {
        return true;
    }

    // Int to Float or Float to Int.
    if matches!(src, IrType::Int { .. }) && matches!(target, IrType::Float { .. }) {
        return true;
    }
    if matches!(src, IrType::Float { .. }) && matches!(target, IrType::Int { .. }) {
        return true;
    }

    // Ref to MutRef and vice versa (same inner type).
    match (src, target) {
        (IrType::Ref(inner), IrType::MutRef(inner2))
        | (IrType::MutRef(inner), IrType::Ref(inner2)) => {
            return types_compatible(inner, inner2);
        }
        _ => {}
    }

    // Ptr-like types: Ref/MutRef to Int (pointer address).
    if matches!(src, IrType::Ref(_) | IrType::MutRef(_))
        && matches!(target, IrType::Int { .. })
    {
        return true;
    }

    // Int to Ref/MutRef (pointer from address).
    if matches!(src, IrType::Int { .. })
        && matches!(target, IrType::Ref(_) | IrType::MutRef(_))
    {
        return true;
    }

    false
}

/// Returns true if the type is numeric (Int or Float).
fn is_numeric(ty: &IrType) -> bool {
    matches!(ty, IrType::Int { .. } | IrType::Float { .. })
}

/// Returns true if the type is numeric or Unknown.
fn is_numeric_or_unknown(ty: &IrType) -> bool {
    matches!(ty, IrType::Int { .. } | IrType::Float { .. } | IrType::Unknown)
}

/// Returns true if the type is Int or Unknown.
fn is_int_or_unknown(ty: &IrType) -> bool {
    matches!(ty, IrType::Int { .. } | IrType::Unknown)
}

/// Returns true if the type is Bool or Unknown.
fn is_bool_or_unknown(ty: &IrType) -> bool {
    matches!(ty, IrType::Bool | IrType::Unknown)
}

/// Returns true if the type is Unit or Unknown.
fn is_unit_or_unknown(ty: &IrType) -> bool {
    matches!(ty, IrType::Unit | IrType::Unknown)
}

/// Returns true if the type is comparable (can be used in Eq/Lt/etc.).
fn is_comparable(ty: &IrType) -> bool {
    matches!(
        ty,
        IrType::Int { .. }
            | IrType::Float { .. }
            | IrType::Bool
            | IrType::String
            | IrType::Unknown
    )
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ir::{BlockId, EffectFlags, FlatBlock, IrInstr, Ownership};
    use crate::compiler::lexer::Span;

    fn dummy_span() -> Span {
        Span::dummy()
    }

    /// Helper: build a minimal FlatIrModule with one function.
    fn make_module(func: FlatIrFunction) -> FlatIrModule {
        FlatIrModule {
            functions: vec![func],
            intrinsics: vec![],
            span: dummy_span(),
            lowering_errors: vec![],
        }
    }

    /// Helper: build a FlatIrFunction from blocks.
    fn make_func(name: &str, params: Vec<(ValueId, IrType)>, ret_ty: IrType, blocks: Vec<FlatBlock>) -> FlatIrFunction {
        FlatIrFunction {
            name: name.to_string(),
            params,
            ret_ty,
            blocks,
            entry: BlockId(0),
            effects: EffectFlags::pure(),
            requires: vec![],
            ensures: vec![],
            span: dummy_span(),
        }
    }

    fn make_block(id: u32, instrs: Vec<IrInstr>) -> FlatBlock {
        FlatBlock {
            id: BlockId(id),
            instrs,
            span: dummy_span(),
        }
    }

    fn make_instr(dst: Option<ValueId>, op: IrOp) -> IrInstr {
        IrInstr {
            dst,
            op,
            span: dummy_span(),
            effects: EffectFlags::pure(),
            ownership: Ownership::Copy,
            cost: crate::compiler::ir::CostHint::Cheap,
            alias: crate::compiler::ir::AliasKind::Unknown,
        }
    }

    #[test]
    fn test_empty_module_no_errors() {
        let module = FlatIrModule {
            functions: vec![],
            intrinsics: vec![],
            span: dummy_span(),
            lowering_errors: vec![],
        };
        let result = ir_typeck(&module);
        assert!(!result.has_errors());
        assert_eq!(result.errors, 0);
    }

    #[test]
    fn test_const_int_valid() {
        let func = make_func(
            "test",
            vec![],
            IrType::Unit,
            vec![make_block(0, vec![
                make_instr(
                    Some(ValueId(0)),
                    IrOp::ConstInt { value: 42, ty: IrType::Int { width: 32, signed: true } },
                ),
                make_instr(None, IrOp::Ret { value: None }),
            ])],
        );
        let result = ir_typeck(&make_module(func));
        assert!(!result.has_errors(), "errors: {:?}", result.diagnostics);
    }

    #[test]
    fn test_const_int_overflow() {
        let func = make_func(
            "test",
            vec![],
            IrType::Unit,
            vec![make_block(0, vec![
                make_instr(
                    Some(ValueId(0)),
                    IrOp::ConstInt { value: 300, ty: IrType::Int { width: 8, signed: true } },
                ),
                make_instr(None, IrOp::Ret { value: None }),
            ])],
        );
        let result = ir_typeck(&make_module(func));
        assert!(result.has_errors());
        assert_eq!(result.diagnostics[0].kind, IrTypeDiagKind::ConstOverflow);
    }

    #[test]
    fn test_binop_type_mismatch() {
        // v0: i32 = 1
        // v1: f64 = 2.0
        // v2 = v0 + v1  ← type mismatch
        let func = make_func(
            "test",
            vec![],
            IrType::Unit,
            vec![make_block(0, vec![
                make_instr(
                    Some(ValueId(0)),
                    IrOp::ConstInt { value: 1, ty: IrType::Int { width: 32, signed: true } },
                ),
                make_instr(
                    Some(ValueId(1)),
                    IrOp::ConstFloat { bits: 0, ty: IrType::Float { width: 64 } },
                ),
                make_instr(
                    Some(ValueId(2)),
                    IrOp::BinOp { op: BinOpKind::Add, lhs: ValueId(0), rhs: ValueId(1) },
                ),
                make_instr(None, IrOp::Ret { value: None }),
            ])],
        );
        let result = ir_typeck(&make_module(func));
        assert!(result.has_errors());
    }

    #[test]
    fn test_condbr_non_bool() {
        // v0: i32 = 1
        // CondBr v0 ← non-bool condition
        let func = make_func(
            "test",
            vec![],
            IrType::Unit,
            vec![make_block(0, vec![
                make_instr(
                    Some(ValueId(0)),
                    IrOp::ConstInt { value: 1, ty: IrType::Int { width: 32, signed: true } },
                ),
                make_instr(
                    None,
                    IrOp::CondBr { cond: ValueId(0), if_true: BlockId(1), if_false: BlockId(2) },
                ),
            ])],
        );
        let result = ir_typeck(&make_module(func));
        assert!(result.has_errors());
        assert_eq!(result.diagnostics[0].kind, IrTypeDiagKind::NonBoolCondition);
    }

    #[test]
    fn test_return_type_mismatch() {
        // fn test() -> i32 { return () }
        let func = make_func(
            "test",
            vec![],
            IrType::Int { width: 32, signed: true },
            vec![make_block(0, vec![
                make_instr(
                    None,
                    IrOp::Ret { value: None }, // returning () from i32 function
                ),
            ])],
        );
        let result = ir_typeck(&make_module(func));
        assert!(result.has_errors());
        assert_eq!(result.diagnostics[0].kind, IrTypeDiagKind::ReturnTypeMismatch);
    }

    #[test]
    fn test_phi_type_mismatch() {
        // v0: i32 = 1
        // v1: f64 = 2.0
        // v2 = phi [bb1: v0, bb2: v1]  ← mismatched types
        let func = make_func(
            "test",
            vec![],
            IrType::Unit,
            vec![make_block(0, vec![
                make_instr(
                    Some(ValueId(0)),
                    IrOp::ConstInt { value: 1, ty: IrType::Int { width: 32, signed: true } },
                ),
                make_instr(
                    Some(ValueId(1)),
                    IrOp::ConstFloat { bits: 0, ty: IrType::Float { width: 64 } },
                ),
                make_instr(
                    Some(ValueId(2)),
                    IrOp::Phi { incoming: vec![(BlockId(1), ValueId(0)), (BlockId(2), ValueId(1))] },
                ),
                make_instr(None, IrOp::Ret { value: None }),
            ])],
        );
        let result = ir_typeck(&make_module(func));
        assert!(result.has_errors());
        assert_eq!(result.diagnostics[0].kind, IrTypeDiagKind::PhiTypeMismatch);
    }

    #[test]
    fn test_undefined_value() {
        // Use v99 without defining it
        let func = make_func(
            "test",
            vec![],
            IrType::Unit,
            vec![make_block(0, vec![
                make_instr(
                    None,
                    IrOp::Ret { value: Some(ValueId(99)) },
                ),
            ])],
        );
        let result = ir_typeck(&make_module(func));
        assert!(result.has_errors());
        assert_eq!(result.diagnostics[0].kind, IrTypeDiagKind::UndefinedValue);
    }

    #[test]
    fn test_valid_binop() {
        let func = make_func(
            "test",
            vec![],
            IrType::Unit,
            vec![make_block(0, vec![
                make_instr(
                    Some(ValueId(0)),
                    IrOp::ConstInt { value: 1, ty: IrType::Int { width: 32, signed: true } },
                ),
                make_instr(
                    Some(ValueId(1)),
                    IrOp::ConstInt { value: 2, ty: IrType::Int { width: 32, signed: true } },
                ),
                make_instr(
                    Some(ValueId(2)),
                    IrOp::BinOp { op: BinOpKind::Add, lhs: ValueId(0), rhs: ValueId(1) },
                ),
                make_instr(None, IrOp::Ret { value: None }),
            ])],
        );
        let result = ir_typeck(&make_module(func));
        assert!(!result.has_errors(), "errors: {:?}", result.diagnostics);
    }

    #[test]
    fn test_valid_call() {
        // Define a function that takes i32 and returns i32, then call it.
        let callee = make_func(
            "add1",
            vec![(ValueId(0), IrType::Int { width: 32, signed: true })],
            IrType::Int { width: 32, signed: true },
            vec![make_block(0, vec![
                make_instr(None, IrOp::Ret { value: Some(ValueId(0)) }),
            ])],
        );
        let caller = make_func(
            "main",
            vec![],
            IrType::Unit,
            vec![make_block(0, vec![
                make_instr(
                    Some(ValueId(0)),
                    IrOp::ConstInt { value: 5, ty: IrType::Int { width: 32, signed: true } },
                ),
                make_instr(
                    Some(ValueId(1)),
                    IrOp::Call { func: "add1".to_string(), args: vec![ValueId(0)] },
                ),
                make_instr(None, IrOp::Ret { value: None }),
            ])],
        );
        let module = FlatIrModule {
            functions: vec![callee, caller],
            intrinsics: vec![],
            span: dummy_span(),
            lowering_errors: vec![],
        };
        let result = ir_typeck(&module);
        assert!(!result.has_errors(), "errors: {:?}", result.diagnostics);
    }

    #[test]
    fn test_call_arg_count_mismatch() {
        let callee = make_func(
            "add1",
            vec![(ValueId(0), IrType::Int { width: 32, signed: true })],
            IrType::Int { width: 32, signed: true },
            vec![make_block(0, vec![
                make_instr(None, IrOp::Ret { value: Some(ValueId(0)) }),
            ])],
        );
        let caller = make_func(
            "main",
            vec![],
            IrType::Unit,
            vec![make_block(0, vec![
                make_instr(
                    Some(ValueId(1)),
                    IrOp::Call { func: "add1".to_string(), args: vec![] }, // no args!
                ),
                make_instr(None, IrOp::Ret { value: None }),
            ])],
        );
        let module = FlatIrModule {
            functions: vec![callee, caller],
            intrinsics: vec![],
            span: dummy_span(),
            lowering_errors: vec![],
        };
        let result = ir_typeck(&module);
        assert!(result.has_errors());
        assert_eq!(result.diagnostics[0].kind, IrTypeDiagKind::ArgCountMismatch);
    }

    #[test]
    fn test_const_int_wrong_type() {
        // ConstInt with Bool type → type mismatch
        let func = make_func(
            "test",
            vec![],
            IrType::Unit,
            vec![make_block(0, vec![
                make_instr(
                    Some(ValueId(0)),
                    IrOp::ConstInt { value: 1, ty: IrType::Bool },
                ),
                make_instr(None, IrOp::Ret { value: None }),
            ])],
        );
        let result = ir_typeck(&make_module(func));
        assert!(result.has_errors());
        assert_eq!(result.diagnostics[0].kind, IrTypeDiagKind::TypeMismatch);
    }

    #[test]
    fn test_valid_store_load() {
        // alloca i32, store i32, load i32
        let func = make_func(
            "test",
            vec![],
            IrType::Unit,
            vec![make_block(0, vec![
                make_instr(
                    Some(ValueId(0)),
                    IrOp::Alloca { ty: IrType::Int { width: 32, signed: true }, align: 4 },
                ),
                make_instr(
                    Some(ValueId(1)),
                    IrOp::ConstInt { value: 42, ty: IrType::Int { width: 32, signed: true } },
                ),
                make_instr(
                    None,
                    IrOp::Store { ptr: ValueId(0), value: ValueId(1) },
                ),
                make_instr(
                    Some(ValueId(2)),
                    IrOp::Load { ptr: ValueId(0), ty: IrType::Int { width: 32, signed: true } },
                ),
                make_instr(None, IrOp::Ret { value: None }),
            ])],
        );
        let result = ir_typeck(&make_module(func));
        assert!(!result.has_errors(), "errors: {:?}", result.diagnostics);
    }

    #[test]
    fn test_store_type_mismatch() {
        // alloca i32, store f64 → mismatch
        let func = make_func(
            "test",
            vec![],
            IrType::Unit,
            vec![make_block(0, vec![
                make_instr(
                    Some(ValueId(0)),
                    IrOp::Alloca { ty: IrType::Int { width: 32, signed: true }, align: 4 },
                ),
                make_instr(
                    Some(ValueId(1)),
                    IrOp::ConstFloat { bits: 0, ty: IrType::Float { width: 64 } },
                ),
                make_instr(
                    None,
                    IrOp::Store { ptr: ValueId(0), value: ValueId(1) },
                ),
                make_instr(None, IrOp::Ret { value: None }),
            ])],
        );
        let result = ir_typeck(&make_module(func));
        assert!(result.has_errors());
    }

    #[test]
    fn test_valid_cast_int_to_float() {
        let func = make_func(
            "test",
            vec![],
            IrType::Unit,
            vec![make_block(0, vec![
                make_instr(
                    Some(ValueId(0)),
                    IrOp::ConstInt { value: 1, ty: IrType::Int { width: 32, signed: true } },
                ),
                make_instr(
                    Some(ValueId(1)),
                    IrOp::Cast { src: ValueId(0), target_ty: IrType::Float { width: 64 } },
                ),
                make_instr(None, IrOp::Ret { value: None }),
            ])],
        );
        let result = ir_typeck(&make_module(func));
        assert!(!result.has_errors(), "errors: {:?}", result.diagnostics);
    }

    #[test]
    fn test_invalid_cast_bool_to_tensor() {
        let func = make_func(
            "test",
            vec![],
            IrType::Unit,
            vec![make_block(0, vec![
                make_instr(
                    Some(ValueId(0)),
                    IrOp::ConstBool { value: true },
                ),
                make_instr(
                    Some(ValueId(1)),
                    IrOp::Cast {
                        src: ValueId(0),
                        target_ty: IrType::Tensor {
                            elem: Box::new(IrType::Float { width: 32 }),
                            shape: vec![4, 4],
                        },
                    },
                ),
                make_instr(None, IrOp::Ret { value: None }),
            ])],
        );
        let result = ir_typeck(&make_module(func));
        assert!(result.has_errors());
        assert_eq!(result.diagnostics[0].kind, IrTypeDiagKind::InvalidCast);
    }

    #[test]
    fn test_type_table_populated() {
        let func = make_func(
            "test",
            vec![(ValueId(10), IrType::Int { width: 64, signed: false })],
            IrType::Unit,
            vec![make_block(0, vec![
                make_instr(
                    Some(ValueId(0)),
                    IrOp::ConstInt { value: 42, ty: IrType::Int { width: 32, signed: true } },
                ),
                make_instr(None, IrOp::Ret { value: None }),
            ])],
        );
        let result = ir_typeck(&make_module(func));
        assert!(!result.has_errors());
        // Check that param and const types are in the type table.
        assert_eq!(
            result.type_table.get(&0),
            Some(&IrType::Int { width: 32, signed: true })
        );
        assert_eq!(
            result.type_table.get(&10),
            Some(&IrType::Int { width: 64, signed: false })
        );
    }
}
