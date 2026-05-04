// =========================================================================
// SAT/SMT Solvers - Range Analysis and Optimal Instruction Selection
// Mathematical proving of program logic soundness
// Range analysis for variable bounds and optimal instruction selection
// =========================================================================

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Variable identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarId(pub usize);

/// Value range for a variable
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueRange {
    /// Minimum value
    pub min: i64,
    /// Maximum value
    pub max: i64,
    /// Whether the range is known
    pub known: bool,
}

impl ValueRange {
    /// Create a new value range
    pub fn new(min: i64, max: i64) -> Self {
        Self {
            min,
            max,
            known: true,
        }
    }

    /// Create an unknown range
    pub fn unknown() -> Self {
        Self {
            min: i64::MIN,
            max: i64::MAX,
            known: false,
        }
    }

    /// Check if value is in range
    pub fn contains(&self, value: i64) -> bool {
        if !self.known {
            return true;
        }
        value >= self.min && value <= self.max
    }

    /// Intersect two ranges
    pub fn intersect(&self, other: &ValueRange) -> ValueRange {
        if !self.known || !other.known {
            return ValueRange::unknown();
        }
        ValueRange::new(self.min.max(other.min), self.max.min(other.max))
    }

    /// Union two ranges
    pub fn union(&self, other: &ValueRange) -> ValueRange {
        if !self.known {
            return *other;
        }
        if !other.known {
            return *self;
        }
        ValueRange::new(self.min.min(other.min), self.max.max(other.max))
    }

    /// Check if range is a single value
    pub fn is_constant(&self) -> bool {
        self.known && self.min == self.max
    }

    /// Get the constant value if range is a single value
    pub fn constant_value(&self) -> Option<i64> {
        if self.is_constant() {
            Some(self.min)
        } else {
            None
        }
    }
}

/// Boolean expression for SAT solver
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BoolExpr {
    /// Variable
    Var(VarId),
    /// Not
    Not(Box<BoolExpr>),
    /// And
    And(Box<BoolExpr>, Box<BoolExpr>),
    /// Or
    Or(Box<BoolExpr>, Box<BoolExpr>),
    /// Implies
    Implies(Box<BoolExpr>, Box<BoolExpr>),
    /// True
    True,
    /// False,
    False,
}

/// Arithmetic expression for SMT solver
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArithExpr {
    /// Constant
    Const(i64),
    /// Variable (by VarId)
    Var(VarId),
    /// Named variable (by string name — used by DCE)
    NamedVar(String),
    /// Add
    Add(Box<ArithExpr>, Box<ArithExpr>),
    /// Subtract
    Sub(Box<ArithExpr>, Box<ArithExpr>),
    /// Multiply
    Mul(Box<ArithExpr>, Box<ArithExpr>),
    /// Divide
    Div(Box<ArithExpr>, Box<ArithExpr>),
    /// Modulo
    Mod(Box<ArithExpr>, Box<ArithExpr>),
    /// Negate
    Neg(Box<ArithExpr>),
}

/// Comparison operator for convenience constraint construction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// SMT constraint
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Constraint {
    /// Less than
    Lt(Box<ArithExpr>, Box<ArithExpr>),
    /// Less than or equal
    Le(Box<ArithExpr>, Box<ArithExpr>),
    /// Greater than
    Gt(Box<ArithExpr>, Box<ArithExpr>),
    /// Greater than or equal
    Ge(Box<ArithExpr>, Box<ArithExpr>),
    /// Equal
    Eq(Box<ArithExpr>, Box<ArithExpr>),
    /// Not equal
    Ne(Box<ArithExpr>, Box<ArithExpr>),
    /// Boolean constraint
    Bool(BoolExpr),
}

impl Constraint {
    /// Convenience constructor: compare a named variable against a constant.
    /// Used by the DCE symbolic executor to build constraints from AST conditions.
    pub fn comparison(var_name: String, value: i64, op: ComparisonOp) -> Self {
        let var = ArithExpr::NamedVar(var_name);
        let val = ArithExpr::Const(value);
        match op {
            ComparisonOp::Eq => Constraint::Eq(Box::new(var), Box::new(val)),
            ComparisonOp::Ne => Constraint::Ne(Box::new(var), Box::new(val)),
            ComparisonOp::Lt => Constraint::Lt(Box::new(var), Box::new(val)),
            ComparisonOp::Le => Constraint::Le(Box::new(var), Box::new(val)),
            ComparisonOp::Gt => Constraint::Gt(Box::new(var), Box::new(val)),
            ComparisonOp::Ge => Constraint::Ge(Box::new(var), Box::new(val)),
        }
    }
}

/// SMT solver result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverResult {
    /// Satisfiable with assignment
    Sat(HashMap<VarId, bool>),
    /// Unsatisfiable
    Unsat,
    /// Unknown
    Unknown,
}

/// Instruction candidate for selection
#[derive(Debug, Clone)]
pub struct InstructionCandidate {
    /// Instruction mnemonic
    pub mnemonic: String,
    /// Latency in cycles
    pub latency: u32,
    /// Throughput (cycles per operation)
    pub throughput: f32,
    /// Port usage
    pub ports: Vec<u32>,
    /// Size in bytes
    pub size: u32,
    /// Whether it requires specific CPU features
    pub required_features: HashSet<String>,
}

/// SAT/SMT solver
pub struct SatSmtSolver {
    /// Variable ranges
    ranges: HashMap<VarId, ValueRange>,
    /// Constraints
    constraints: Vec<Constraint>,
    /// Next variable ID
    next_var_id: usize,
}

impl SatSmtSolver {
    /// Create a new SAT/SMT solver
    pub fn new() -> Self {
        Self {
            ranges: HashMap::new(),
            constraints: Vec::new(),
            next_var_id: 0,
        }
    }

    /// Create a new variable
    pub fn new_var(&mut self) -> VarId {
        let id = self.next_var_id;
        self.next_var_id += 1;
        self.ranges.insert(VarId(id), ValueRange::unknown());
        VarId(id)
    }

    /// Set range for a variable
    pub fn set_range(&mut self, var: VarId, range: ValueRange) {
        self.ranges.insert(var, range);
    }

    /// Get range for a variable
    pub fn get_range(&self, var: VarId) -> ValueRange {
        self.ranges.get(&var).copied().unwrap_or(ValueRange::unknown())
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Perform range analysis
    pub fn range_analysis(&mut self) -> HashMap<VarId, ValueRange> {
        let mut changed = true;
        let mut iteration = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iteration < MAX_ITERATIONS {
            changed = false;
            iteration += 1;

            for constraint in self.constraints.clone() {
                if let Some(new_ranges) = self.propagate_constraint(&constraint) {
                    for (var, new_range) in new_ranges {
                        let old_range = self.get_range(var);
                        let merged = old_range.intersect(&new_range);
                        if merged != old_range {
                            self.set_range(var, merged);
                            changed = true;
                        }
                    }
                }
            }
        }

        self.ranges.clone()
    }

    /// Propagate a constraint to update ranges
    fn propagate_constraint(&self, constraint: &Constraint) -> Option<HashMap<VarId, ValueRange>> {
        match constraint {
            Constraint::Lt(left, right) => self.propagate_comparison(left, right, |a, b| a < b),
            Constraint::Le(left, right) => self.propagate_comparison(left, right, |a, b| a <= b),
            Constraint::Gt(left, right) => self.propagate_comparison(left, right, |a, b| a > b),
            Constraint::Ge(left, right) => self.propagate_comparison(left, right, |a, b| a >= b),
            Constraint::Eq(left, right) => self.propagate_equality(left, right),
            Constraint::Ne(left, right) => self.propagate_inequality(left, right),
            Constraint::Bool(_) => None,
        }
    }

    /// Propagate comparison constraint
    fn propagate_comparison(
        &self,
        left: &ArithExpr,
        right: &ArithExpr,
        compare: fn(i64, i64) -> bool,
    ) -> Option<HashMap<VarId, ValueRange>> {
        let mut new_ranges = HashMap::new();

        if let (ArithExpr::Var(left_var), ArithExpr::Const(right_const)) = (left, right) {
            let left_range = self.get_range(*left_var);
            if left_range.known {
                // If left < right_const, then left.max < right_const
                let new_max = if compare(left_range.max, *right_const) {
                    left_range.max
                } else {
                    *right_const - 1
                };
                new_ranges.insert(*left_var, ValueRange::new(left_range.min, new_max));
            }
        }

        if let (ArithExpr::Const(left_const), ArithExpr::Var(right_var)) = (left, right) {
            let right_range = self.get_range(*right_var);
            if right_range.known {
                // If left_const < right, then right.min > left_const
                let new_min = if compare(*left_const, right_range.min) {
                    right_range.min
                } else {
                    *left_const + 1
                };
                new_ranges.insert(*right_var, ValueRange::new(new_min, right_range.max));
            }
        }

        if new_ranges.is_empty() {
            None
        } else {
            Some(new_ranges)
        }
    }

    /// Propagate equality constraint
    fn propagate_equality(&self, left: &ArithExpr, right: &ArithExpr) -> Option<HashMap<VarId, ValueRange>> {
        let mut new_ranges = HashMap::new();

        if let (ArithExpr::Var(left_var), ArithExpr::Var(right_var)) = (left, right) {
            let left_range = self.get_range(*left_var);
            let right_range = self.get_range(*right_var);
            
            if left_range.known && right_range.known {
                let intersected = left_range.intersect(&right_range);
                new_ranges.insert(*left_var, intersected);
                new_ranges.insert(*right_var, intersected);
            }
        }

        if let (ArithExpr::Var(var), ArithExpr::Const(const_val)) = (left, right) {
            new_ranges.insert(*var, ValueRange::new(*const_val, *const_val));
        }

        if let (ArithExpr::Const(const_val), ArithExpr::Var(var)) = (left, right) {
            new_ranges.insert(*var, ValueRange::new(*const_val, *const_val));
        }

        if new_ranges.is_empty() {
            None
        } else {
            Some(new_ranges)
        }
    }

    /// Propagate inequality constraint
    fn propagate_inequality(&self, left: &ArithExpr, right: &ArithExpr) -> Option<HashMap<VarId, ValueRange>> {
        let mut new_ranges = HashMap::new();

        if let (ArithExpr::Var(var), ArithExpr::Const(const_val)) = (left, right) {
            let range = self.get_range(*var);
            if range.known && range.is_constant() && range.min == *const_val {
                // If var == const_val, this constraint is unsatisfiable
                // Return empty map to indicate conflict
                return Some(HashMap::new());
            }
        }

        if new_ranges.is_empty() {
            None
        } else {
            Some(new_ranges)
        }
    }

    /// Simplify expression based on known ranges
    pub fn simplify_expr(&self, expr: &ArithExpr) -> ArithExpr {
        match expr {
            ArithExpr::Const(c) => ArithExpr::Const(*c),
            ArithExpr::Var(v) => {
                let range = self.get_range(*v);
                if let Some(const_val) = range.constant_value() {
                    ArithExpr::Const(const_val)
                } else {
                    ArithExpr::Var(*v)
                }
            }
            ArithExpr::NamedVar(_) => expr.clone(), // Named variables pass through unchanged
            ArithExpr::Add(a, b) => {
                let a_simplified = self.simplify_expr(a);
                let b_simplified = self.simplify_expr(b);
                if let (ArithExpr::Const(a_val), ArithExpr::Const(b_val)) = (&a_simplified, &b_simplified) {
                    ArithExpr::Const(a_val + b_val)
                } else {
                    ArithExpr::Add(Box::new(a_simplified), Box::new(b_simplified))
                }
            }
            ArithExpr::Sub(a, b) => {
                let a_simplified = self.simplify_expr(a);
                let b_simplified = self.simplify_expr(b);
                if let (ArithExpr::Const(a_val), ArithExpr::Const(b_val)) = (&a_simplified, &b_simplified) {
                    ArithExpr::Const(a_val - b_val)
                } else {
                    ArithExpr::Sub(Box::new(a_simplified), Box::new(b_simplified))
                }
            }
            ArithExpr::Mul(a, b) => {
                let a_simplified = self.simplify_expr(a);
                let b_simplified = self.simplify_expr(b);
                if let (ArithExpr::Const(a_val), ArithExpr::Const(b_val)) = (&a_simplified, &b_simplified) {
                    ArithExpr::Const(a_val * b_val)
                } else {
                    ArithExpr::Mul(Box::new(a_simplified), Box::new(b_simplified))
                }
            }
            ArithExpr::Div(a, b) => {
                let a_simplified = self.simplify_expr(a);
                let b_simplified = self.simplify_expr(b);
                if let (ArithExpr::Const(a_val), ArithExpr::Const(b_val)) = (&a_simplified, &b_simplified) {
                    if *b_val != 0 {
                        ArithExpr::Const(a_val / b_val)
                    } else {
                        ArithExpr::Div(Box::new(a_simplified), Box::new(b_simplified))
                    }
                } else {
                    ArithExpr::Div(Box::new(a_simplified), Box::new(b_simplified))
                }
            }
            ArithExpr::Mod(a, b) => {
                let a_simplified = self.simplify_expr(a);
                let b_simplified = self.simplify_expr(b);
                if let (ArithExpr::Const(a_val), ArithExpr::Const(b_val)) = (&a_simplified, &b_simplified) {
                    if *b_val != 0 {
                        ArithExpr::Const(a_val % b_val)
                    } else {
                        ArithExpr::Mod(Box::new(a_simplified), Box::new(b_simplified))
                    }
                } else {
                    ArithExpr::Mod(Box::new(a_simplified), Box::new(b_simplified))
                }
            }
            ArithExpr::Neg(a) => {
                let a_simplified = self.simplify_expr(a);
                if let ArithExpr::Const(a_val) = a_simplified {
                    ArithExpr::Const(-a_val)
                } else {
                    ArithExpr::Neg(Box::new(a_simplified))
                }
            }
        }
    }

    /// Check if a condition is always true based on ranges
    pub fn is_always_true(&self, constraint: &Constraint) -> bool {
        match constraint {
            Constraint::Lt(left, right) => {
                let left_range = self.eval_range(left);
                let right_range = self.eval_range(right);
                left_range.known && right_range.known && left_range.max < right_range.min
            }
            Constraint::Le(left, right) => {
                let left_range = self.eval_range(left);
                let right_range = self.eval_range(right);
                left_range.known && right_range.known && left_range.max <= right_range.min
            }
            Constraint::Gt(left, right) => {
                let left_range = self.eval_range(left);
                let right_range = self.eval_range(right);
                left_range.known && right_range.known && left_range.min > right_range.max
            }
            Constraint::Ge(left, right) => {
                let left_range = self.eval_range(left);
                let right_range = self.eval_range(right);
                left_range.known && right_range.known && left_range.min >= right_range.max
            }
            Constraint::Eq(left, right) => {
                let left_range = self.eval_range(left);
                let right_range = self.eval_range(right);
                left_range.known && right_range.known && left_range == right_range && left_range.is_constant()
            }
            Constraint::Ne(left, right) => {
                let left_range = self.eval_range(left);
                let right_range = self.eval_range(right);
                left_range.known && right_range.known && (left_range.max < right_range.min || left_range.min > right_range.max)
            }
            Constraint::Bool(BoolExpr::True) => true,
            Constraint::Bool(BoolExpr::False) => false,
            Constraint::Bool(_) => false,
        }
    }

    /// Check if a condition is always false based on ranges
    pub fn is_always_false(&self, constraint: &Constraint) -> bool {
        match constraint {
            Constraint::Lt(left, right) => {
                let left_range = self.eval_range(left);
                let right_range = self.eval_range(right);
                left_range.known && right_range.known && left_range.min >= right_range.max
            }
            Constraint::Le(left, right) => {
                let left_range = self.eval_range(left);
                let right_range = self.eval_range(right);
                left_range.known && right_range.known && left_range.min > right_range.max
            }
            Constraint::Gt(left, right) => {
                let left_range = self.eval_range(left);
                let right_range = self.eval_range(right);
                left_range.known && right_range.known && left_range.max <= right_range.min
            }
            Constraint::Ge(left, right) => {
                let left_range = self.eval_range(left);
                let right_range = self.eval_range(right);
                left_range.known && right_range.known && left_range.max < right_range.min
            }
            Constraint::Eq(left, right) => {
                let left_range = self.eval_range(left);
                let right_range = self.eval_range(right);
                left_range.known && right_range.known && (left_range.max < right_range.min || left_range.min > right_range.max)
            }
            Constraint::Ne(left, right) => {
                let left_range = self.eval_range(left);
                let right_range = self.eval_range(right);
                left_range.known && right_range.known && left_range == right_range && left_range.is_constant()
            }
            Constraint::Bool(BoolExpr::True) => false,
            Constraint::Bool(BoolExpr::False) => true,
            Constraint::Bool(_) => false,
        }
    }

    /// Evaluate range of an arithmetic expression
    fn eval_range(&self, expr: &ArithExpr) -> ValueRange {
        match expr {
            ArithExpr::Const(c) => ValueRange::new(*c, *c),
            ArithExpr::Var(v) => self.get_range(*v),
            ArithExpr::NamedVar(_) => ValueRange::unknown(), // Named vars have unknown ranges
            ArithExpr::Add(a, b) => {
                let a_range = self.eval_range(a);
                let b_range = self.eval_range(b);
                if a_range.known && b_range.known {
                    ValueRange::new(a_range.min + b_range.min, a_range.max + b_range.max)
                } else {
                    ValueRange::unknown()
                }
            }
            ArithExpr::Sub(a, b) => {
                let a_range = self.eval_range(a);
                let b_range = self.eval_range(b);
                if a_range.known && b_range.known {
                    ValueRange::new(a_range.min - b_range.max, a_range.max - b_range.min)
                } else {
                    ValueRange::unknown()
                }
            }
            ArithExpr::Mul(a, b) => {
                let a_range = self.eval_range(a);
                let b_range = self.eval_range(b);
                if a_range.known && b_range.known {
                    let candidates = [
                        a_range.min * b_range.min,
                        a_range.min * b_range.max,
                        a_range.max * b_range.min,
                        a_range.max * b_range.max,
                    ];
                    ValueRange::new(*candidates.iter().min().unwrap(), *candidates.iter().max().unwrap())
                } else {
                    ValueRange::unknown()
                }
            }
            _ => ValueRange::unknown(),
        }
    }

    /// Solve SAT problem (simplified DPLL)
    pub fn solve_sat(&self, constraints: &[BoolExpr]) -> SolverResult {
        // Simplified SAT solver using DPLL algorithm
        let mut assignment = HashMap::new();
        
        if self.dpll(constraints, &mut assignment, 0) {
            SolverResult::Sat(assignment)
        } else {
            SolverResult::Unsat
        }
    }

    /// DPLL algorithm for SAT solving
    fn dpll(&self, constraints: &[BoolExpr], assignment: &mut HashMap<VarId, bool>, depth: usize) -> bool {
        if depth > 1000 {
            return false;
        }

        // Check if all constraints are satisfied
        let mut all_satisfied = true;
        for constraint in constraints {
            if !self.eval_bool(constraint, assignment) {
                all_satisfied = false;
                break;
            }
        }
        if all_satisfied {
            return true;
        }

        // Find an unassigned variable
        let unassigned_var = self.find_unassigned_var(constraints, assignment);
        if unassigned_var.is_none() {
            return false;
        }
        let var = unassigned_var.unwrap();

        // Try assigning true
        assignment.insert(var, true);
        if self.dpll(constraints, assignment, depth + 1) {
            return true;
        }

        // Try assigning false
        assignment.insert(var, false);
        if self.dpll(constraints, assignment, depth + 1) {
            return true;
        }

        // Backtrack
        assignment.remove(&var);
        false
    }

    /// Find an unassigned variable
    fn find_unassigned_var(&self, constraints: &[BoolExpr], assignment: &HashMap<VarId, bool>) -> Option<VarId> {
        for constraint in constraints {
            if let Some(var) = self.find_var_in_bool(constraint, assignment) {
                return Some(var);
            }
        }
        None
    }

    /// Find variable in boolean expression
    fn find_var_in_bool(&self, expr: &BoolExpr, assignment: &HashMap<VarId, bool>) -> Option<VarId> {
        match expr {
            BoolExpr::Var(v) if !assignment.contains_key(v) => Some(*v),
            BoolExpr::Not(e) => self.find_var_in_bool(e, assignment),
            BoolExpr::And(a, b) => self.find_var_in_bool(a, assignment).or_else(|| self.find_var_in_bool(b, assignment)),
            BoolExpr::Or(a, b) => self.find_var_in_bool(a, assignment).or_else(|| self.find_var_in_bool(b, assignment)),
            BoolExpr::Implies(a, b) => self.find_var_in_bool(a, assignment).or_else(|| self.find_var_in_bool(b, assignment)),
            _ => None,
        }
    }

    /// Evaluate boolean expression with assignment
    fn eval_bool(&self, expr: &BoolExpr, assignment: &HashMap<VarId, bool>) -> bool {
        match expr {
            BoolExpr::True => true,
            BoolExpr::False => false,
            BoolExpr::Var(v) => *assignment.get(v).unwrap_or(&false),
            BoolExpr::Not(e) => !self.eval_bool(e, assignment),
            BoolExpr::And(a, b) => self.eval_bool(a, assignment) && self.eval_bool(b, assignment),
            BoolExpr::Or(a, b) => self.eval_bool(a, assignment) || self.eval_bool(b, assignment),
            BoolExpr::Implies(a, b) => !self.eval_bool(a, assignment) || self.eval_bool(b, assignment),
        }
    }
}

impl Default for SatSmtSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Instruction selector for optimal instruction selection
pub struct InstructionSelector {
    /// Available instructions
    instructions: Vec<InstructionCandidate>,
    /// CPU features available
    available_features: HashSet<String>,
}

impl InstructionSelector {
    /// Create a new instruction selector
    pub fn new(available_features: HashSet<String>) -> Self {
        Self {
            instructions: Self::get_instruction_set(),
            available_features,
        }
    }

    /// Get the instruction set for x86_64
    fn get_instruction_set() -> Vec<InstructionCandidate> {
        vec![
            InstructionCandidate {
                mnemonic: "add".to_string(),
                latency: 1,
                throughput: 0.5,
                ports: vec![0, 1, 5],
                size: 3,
                required_features: HashSet::new(),
            },
            InstructionCandidate {
                mnemonic: "lea".to_string(),
                latency: 1,
                throughput: 0.5,
                ports: vec![1, 5],
                size: 4,
                required_features: HashSet::new(),
            },
            InstructionCandidate {
                mnemonic: "imul".to_string(),
                latency: 3,
                throughput: 1.0,
                ports: vec![1],
                size: 4,
                required_features: HashSet::new(),
            },
            InstructionCandidate {
                mnemonic: "imul".to_string(),
                latency: 3,
                throughput: 1.0,
                ports: vec![1],
                size: 3,
                required_features: {
                    let mut set = HashSet::new();
                    set.insert("BMI2".to_string());
                    set
                },
            },
            InstructionCandidate {
                mnemonic: "shlx".to_string(),
                latency: 1,
                throughput: 0.5,
                ports: vec![1, 5],
                size: 4,
                required_features: {
                    let mut set = HashSet::new();
                    set.insert("BMI2".to_string());
                    set
                },
            },
            InstructionCandidate {
                mnemonic: "vaddps".to_string(),
                latency: 3,
                throughput: 0.5,
                ports: vec![0, 1],
                size: 4,
                required_features: {
                    let mut set = HashSet::new();
                    set.insert("AVX".to_string());
                    set
                },
            },
            InstructionCandidate {
                mnemonic: "vaddpd".to_string(),
                latency: 3,
                throughput: 0.5,
                ports: vec![0, 1],
                size: 4,
                required_features: {
                    let mut set = HashSet::new();
                    set.insert("AVX".to_string());
                    set
                },
            },
            InstructionCandidate {
                mnemonic: "vfmadd231ps".to_string(),
                latency: 4,
                throughput: 0.5,
                ports: vec![0, 1],
                size: 5,
                required_features: {
                    let mut set = HashSet::new();
                    set.insert("FMA".to_string());
                    set
                },
            },
        ]
    }

    /// Select optimal instruction for an operation
    pub fn select_instruction(&self, operation: &str, operands: usize) -> Option<&InstructionCandidate> {
        let mut candidates: Vec<_> = self.instructions
            .iter()
            .filter(|inst| {
                inst.mnemonic.starts_with(operation) && 
                self.features_available(&inst.required_features)
            })
            .collect();

        // Sort by latency, then throughput, then size
        candidates.sort_by(|a, b| {
            a.latency
                .cmp(&b.latency)
                .then_with(|| a.throughput.partial_cmp(&b.throughput).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.size.cmp(&b.size))
        });

        candidates.first().copied()
    }

    /// Check if required features are available
    fn features_available(&self, required: &HashSet<String>) -> bool {
        required.iter().all(|f| self.available_features.contains(f))
    }

    /// Prove that instruction X is optimal for given operation
    pub fn prove_optimal(&self, operation: &str, selected: &InstructionCandidate) -> bool {
        // Get all candidates for this operation
        let candidates: Vec<_> = self.instructions
            .iter()
            .filter(|inst| {
                inst.mnemonic.starts_with(operation) && 
                self.features_available(&inst.required_features)
            })
            .collect();

        // Check if selected is Pareto-optimal (no other instruction is strictly better)
        for candidate in &candidates {
            if candidate.mnemonic == selected.mnemonic {
                continue;
            }

            // Check if candidate is strictly better
            let strictly_better = candidate.latency < selected.latency
                && candidate.throughput <= selected.throughput
                && candidate.size <= selected.size;

            if strictly_better {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_analysis() {
        let mut solver = SatSmtSolver::new();
        let x = solver.new_var();
        let y = solver.new_var();

        solver.set_range(x, ValueRange::new(0, 10));
        solver.set_range(y, ValueRange::new(5, 15));

        solver.add_constraint(Constraint::Lt(
            Box::new(ArithExpr::Var(x)),
            Box::new(ArithExpr::Var(y)),
        ));

        let ranges = solver.range_analysis();
        assert!(ranges.get(&x).unwrap().max < ranges.get(&y).unwrap().min);
    }

    #[test]
    fn test_constant_propagation() {
        let mut solver = SatSmtSolver::new();
        let x = solver.new_var();

        solver.set_range(x, ValueRange::new(5, 5)); // Constant

        let expr = ArithExpr::Add(
            Box::new(ArithExpr::Var(x)),
            Box::new(ArithExpr::Const(3)),
        );

        let simplified = solver.simplify_expr(&expr);
        assert_eq!(simplified, ArithExpr::Const(8));
    }

    #[test]
    fn test_always_true_condition() {
        let mut solver = SatSmtSolver::new();
        let x = solver.new_var();
        let y = solver.new_var();

        solver.set_range(x, ValueRange::new(0, 5));
        solver.set_range(y, ValueRange::new(10, 20));

        let constraint = Constraint::Lt(
            Box::new(ArithExpr::Var(x)),
            Box::new(ArithExpr::Var(y)),
        );

        assert!(solver.is_always_true(&constraint));
    }

    #[test]
    fn test_instruction_selection() {
        let mut features = HashSet::new();
        features.insert("AVX".to_string());
        features.insert("FMA".to_string());

        let selector = InstructionSelector::new(features);
        let selected = selector.select_instruction("vadd", 2);

        assert!(selected.is_some());
        assert!(selected.unwrap().mnemonic.starts_with("vadd"));
    }
}
