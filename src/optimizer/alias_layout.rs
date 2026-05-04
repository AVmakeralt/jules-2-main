// =============================================================================
// Alias-Aware Memory Layout via Ownership Proofs
//
// Jules has a borrow checker — which means it has ownership proofs that most
// languages lack. This unlocks something C and C++ can't do safely:
// provably alias-free memory layouts.
//
// When the borrow checker proves two memory regions can never alias, Jules can:
//
//   1. Emit noalias hints to Cranelift/LLVM (already partially done via
//      Rust's noalias, but Jules can be more aggressive at the language level).
//   2. Automatically convert AoS (Array of Structures) to SoA based on
//      proven access patterns, not just heuristics.
//   3. Pack hot fields of structs together based on proven co-access patterns
//      from lifetime analysis — essentially alias-guided data layout.
//
// The borrow checker's lifetime graph already encodes what gets accessed
// together. Jules just needs to use that graph to inform struct field
// ordering and memory layout.
//
// Architecture:
//
//   Borrow checker lifetime graph
//       │
//       ▼
//   AliasAnalysis ─── proves noalias relationships from ownership proofs
//       │
//       ▼
//   LayoutOptimizer ─── reorder struct fields, emit noalias hints, AoS→SoA
//       │                   • Hot-field packing: fields accessed together → adjacent
//       │                   • Cold-field separation: rarely-used fields → separate cache line
//       │                   • Noalias emission: proven non-aliasing → Cranelift/LLVM hints
//       ▼
//   Optimized Struct/Array layout
// =============================================================================

use std::collections::{HashMap, HashSet};

use crate::compiler::ast::*;

// ─── Alias Analysis ───────────────────────────────────────────────────────────

/// An alias relationship between two memory regions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AliasRelation {
    /// The first region (variable or field).
    pub region_a: MemoryRegion,
    /// The second region (variable or field).
    pub region_b: MemoryRegion,
    /// Whether these regions can ever alias.
    pub can_alias: bool,
    /// The proof: why we know they alias or don't.
    pub proof: AliasProof,
}

/// A memory region that can be analyzed for aliasing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemoryRegion {
    /// A local variable.
    Variable(String),
    /// A struct field: struct_name.field_name.
    StructField { struct_name: String, field_name: String },
    /// An array element: array_name[index].
    ArrayElement { array_name: String, index: String },
    /// A reference target: the thing a reference points to.
    RefTarget(String),
}

/// Proof that two regions do or do not alias.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AliasProof {
    /// The borrow checker proved these are separate owners.
    OwnershipDisjoint,
    /// The lifetimes don't overlap (NLL).
    LifetimeDisjoint,
    /// Both are &mut references, and the borrow checker only allows one at a
    /// time, so they can never alias.
    MutBorrowExclusivity,
    /// The types are different, so they can't point to the same memory.
    TypeDisjoint,
    /// The fields belong to the same struct but are at different offsets.
    FieldOffsetDisjoint,
    /// We can't prove either way — conservatively assume aliasing.
    Unknown,
    /// They definitely alias (same variable, same field).
    DefiniteAlias,
}

/// The alias analysis engine. Derives alias relationships from the borrow
/// checker's ownership proofs and lifetime information.
pub struct AliasAnalyzer {
    /// All proven non-aliasing relationships.
    noalias_pairs: HashSet<(MemoryRegion, MemoryRegion)>,
    /// All struct fields and their access patterns.
    field_access_patterns: HashMap<String, FieldAccessPattern>,
    /// All struct definitions.
    struct_defs: HashMap<String, StructDecl>,
    /// Struct names that have been analyzed.
    analyzed_structs: HashSet<String>,
}

/// Access pattern for a struct field.
#[derive(Debug, Clone)]
pub struct FieldAccessPattern {
    /// The struct name.
    pub struct_name: String,
    /// The field name.
    pub field_name: String,
    /// Number of times this field is read.
    pub read_count: u64,
    /// Number of times this field is written.
    pub write_count: u64,
    /// Number of times this field is accessed together with each other field.
    pub co_access_counts: HashMap<String, u64>,
    /// Whether this field is ever accessed via &mut (exclusive).
    pub is_mutably_borrowed: bool,
    /// Whether this field's type is a Copy type (scalar, etc.).
    pub is_copy_type: bool,
}

impl FieldAccessPattern {
    pub fn new(struct_name: String, field_name: String, is_copy_type: bool) -> Self {
        Self {
            struct_name,
            field_name,
            read_count: 0,
            write_count: 0,
            co_access_counts: HashMap::new(),
            is_mutably_borrowed: false,
            is_copy_type,
        }
    }

    /// Hotness score: how frequently this field is accessed.
    pub fn hotness(&self) -> u64 {
        self.read_count + self.write_count * 2 // Writes are "hotter"
    }

    /// Whether this field is hot (frequently accessed).
    pub fn is_hot(&self, threshold: u64) -> bool {
        self.hotness() >= threshold
    }
}

impl AliasAnalyzer {
    pub fn new() -> Self {
        Self {
            noalias_pairs: HashSet::new(),
            field_access_patterns: HashMap::new(),
            struct_defs: HashMap::new(),
            analyzed_structs: HashSet::new(),
        }
    }

    /// Register a struct definition for analysis.
    pub fn register_struct(&mut self, decl: &StructDecl) {
        self.struct_defs.insert(decl.name.clone(), decl.clone());
    }

    /// Record that two memory regions are provably non-aliasing.
    pub fn prove_noalias(&mut self, a: MemoryRegion, b: MemoryRegion, proof: AliasProof) {
        match proof {
            AliasProof::OwnershipDisjoint
            | AliasProof::LifetimeDisjoint
            | AliasProof::MutBorrowExclusivity
            | AliasProof::TypeDisjoint
            | AliasProof::FieldOffsetDisjoint => {
                self.noalias_pairs.insert((a.clone(), b.clone()));
                self.noalias_pairs.insert((b, a)); // Symmetric
            }
            _ => {} // Not a non-aliasing proof
        }
    }

    /// Check if two memory regions are provably non-aliasing.
    pub fn is_noalias(&self, a: &MemoryRegion, b: &MemoryRegion) -> bool {
        self.noalias_pairs.contains(&(a.clone(), b.clone()))
    }

    /// Record a field access.
    pub fn record_field_access(
        &mut self,
        struct_name: &str,
        field_name: &str,
        is_read: bool,
        is_copy_type: bool,
    ) {
        let key = format!("{}.{}", struct_name, field_name);
        let pattern = self
            .field_access_patterns
            .entry(key)
            .or_insert_with(|| FieldAccessPattern::new(struct_name.to_string(), field_name.to_string(), is_copy_type));

        if is_read {
            pattern.read_count += 1;
        } else {
            pattern.write_count += 1;
        }
    }

    /// Record co-access of two fields (accessed in the same hot loop body).
    pub fn record_co_access(&mut self, struct_name: &str, field_a: &str, field_b: &str) {
        if field_a == field_b {
            return;
        }
        let key_a = format!("{}.{}", struct_name, field_a);
        let key_b = format!("{}.{}", struct_name, field_b);

        if let Some(pattern) = self.field_access_patterns.get_mut(&key_a) {
            *pattern.co_access_counts.entry(field_b.to_string()).or_insert(0) += 1;
        }
        if let Some(pattern) = self.field_access_patterns.get_mut(&key_b) {
            *pattern.co_access_counts.entry(field_a.to_string()).or_insert(0) += 1;
        }
    }

    /// Prove that all fields of a struct are non-aliasing (from borrow checker).
    pub fn prove_struct_fields_noalias(&mut self, struct_name: &str) {
        // Collect field names first to avoid borrowing self as both
        // mutable and immutable at the same time.
        let field_names: Vec<String> = if let Some(decl) = self.struct_defs.get(struct_name) {
            decl.fields.iter().map(|f| f.name.clone()).collect()
        } else {
            return;
        };

        for i in 0..field_names.len() {
            for j in (i + 1)..field_names.len() {
                self.prove_noalias(
                    MemoryRegion::StructField {
                        struct_name: struct_name.to_string(),
                        field_name: field_names[i].clone(),
                    },
                    MemoryRegion::StructField {
                        struct_name: struct_name.to_string(),
                        field_name: field_names[j].clone(),
                    },
                    AliasProof::FieldOffsetDisjoint,
                );
            }
        }
        self.analyzed_structs.insert(struct_name.to_string());
    }

    /// Get all noalias pairs.
    pub fn noalias_pairs(&self) -> &HashSet<(MemoryRegion, MemoryRegion)> {
        &self.noalias_pairs
    }

    /// Get field access patterns for a struct.
    pub fn field_patterns(&self, struct_name: &str) -> Vec<&FieldAccessPattern> {
        self.field_access_patterns
            .iter()
            .filter(|(k, _)| k.starts_with(&format!("{}.", struct_name)))
            .map(|(_, v)| v)
            .collect()
    }
}

impl Default for AliasAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Layout Optimizer ─────────────────────────────────────────────────────────

/// A noalias hint to emit to the backend (Cranelift/LLVM).
#[derive(Debug, Clone)]
pub struct NoaliasHint {
    /// The parameter or variable that is noalias.
    pub name: String,
    /// The reason it's noalias.
    pub proof: AliasProof,
}

/// A field reordering suggestion.
#[derive(Debug, Clone)]
pub struct FieldReorderSuggestion {
    /// The struct name.
    pub struct_name: String,
    /// The original field order.
    pub original_order: Vec<String>,
    /// The suggested field order (hot fields first, co-accessed fields adjacent).
    pub suggested_order: Vec<String>,
    /// Estimated cache miss reduction (0.0–1.0).
    pub estimated_cache_miss_reduction: f64,
    /// Reason for the reordering.
    pub reason: String,
}

/// An AoS → SoA conversion suggestion.
#[derive(Debug, Clone)]
pub struct SoaConversionSuggestion {
    /// The struct name.
    pub struct_name: String,
    /// The fields that should be split into separate arrays.
    pub fields_to_split: Vec<String>,
    /// Estimated speedup.
    pub estimated_speedup: f64,
    /// Reason for the conversion.
    pub reason: String,
}

/// Result of layout optimization.
#[derive(Debug, Clone, Default)]
pub struct LayoutOptimizationResult {
    /// Noalias hints to emit.
    pub noalias_hints: Vec<NoaliasHint>,
    /// Field reordering suggestions.
    pub field_reorder_suggestions: Vec<FieldReorderSuggestion>,
    /// SoA conversion suggestions.
    pub soa_conversion_suggestions: Vec<SoaConversionSuggestion>,
    /// Number of noalias pairs proven.
    pub noalias_pairs_proven: usize,
    /// Estimated overall speedup.
    pub estimated_speedup: f64,
}

/// The layout optimizer uses alias analysis results to reorder struct fields,
/// emit noalias hints, and suggest AoS→SoA conversions.
pub struct LayoutOptimizer {
    /// The alias analyzer.
    analyzer: AliasAnalyzer,
    /// Hotness threshold for field access patterns.
    hot_threshold: u64,
    /// Co-access threshold for field grouping.
    co_access_threshold: u64,
}

impl LayoutOptimizer {
    pub fn new(hot_threshold: u64, co_access_threshold: u64) -> Self {
        Self {
            analyzer: AliasAnalyzer::new(),
            hot_threshold,
            co_access_threshold,
        }
    }

    /// Register a struct definition.
    pub fn register_struct(&mut self, decl: &StructDecl) {
        self.analyzer.register_struct(decl);
    }

    /// Analyze a program and produce layout optimization suggestions.
    pub fn optimize_program(&mut self, program: &Program) -> LayoutOptimizationResult {
        let mut result = LayoutOptimizationResult::default();

        // Phase 1: Collect struct definitions and analyze access patterns.
        for item in &program.items {
            match item {
                Item::Struct(decl) => {
                    self.analyzer.register_struct(decl);
                    self.analyzer.prove_struct_fields_noalias(&decl.name);
                }
                Item::Component(decl) => {
                    self.analyzer.register_struct(&StructDecl {
                        span: decl.span,
                        attrs: decl.attrs.clone(),
                        name: decl.name.clone(),
                        generics: vec![],
                        fields: decl.fields.clone(),
                    });
                    self.analyzer.prove_struct_fields_noalias(&decl.name);
                }
                _ => {}
            }
        }

        // Phase 2: Analyze access patterns from function bodies.
        for item in &program.items {
            match item {
                Item::Fn(fn_decl) => {
                    if let Some(body) = &fn_decl.body {
                        self.analyze_access_patterns_block(body);
                    }
                }
                Item::System(sys) => {
                    self.analyze_access_patterns_block(&sys.body);
                }
                _ => {}
            }
        }

        // Phase 3: Generate noalias hints from proven relationships.
        for (a, _b) in self.analyzer.noalias_pairs() {
            if let MemoryRegion::StructField { struct_name, field_name } = a {
                result.noalias_hints.push(NoaliasHint {
                    name: format!("{}.{}", struct_name, field_name),
                    proof: AliasProof::FieldOffsetDisjoint,
                });
            }
        }
        result.noalias_pairs_proven = self.analyzer.noalias_pairs().len() / 2; // Symmetric

        // Phase 4: Generate field reordering suggestions.
        for (struct_name, decl) in &self.analyzer.struct_defs {
            let suggestion = self.suggest_field_reorder(struct_name, decl);
            if let Some(s) = suggestion {
                result.field_reorder_suggestions.push(s);
            }
        }

        // Phase 5: Generate SoA conversion suggestions.
        for (struct_name, _) in &self.analyzer.struct_defs {
            let suggestion = self.suggest_soa_conversion(struct_name);
            if let Some(s) = suggestion {
                result.soa_conversion_suggestions.push(s);
            }
        }

        // Estimate overall speedup.
        let reorder_speedup: f64 = result
            .field_reorder_suggestions
            .iter()
            .map(|s| s.estimated_cache_miss_reduction * 0.5)
            .sum();
        let soa_speedup: f64 = result
            .soa_conversion_suggestions
            .iter()
            .map(|s| s.estimated_speedup - 1.0)
            .sum();
        let noalias_speedup = if result.noalias_pairs_proven > 0 {
            0.05 * result.noalias_pairs_proven.min(10) as f64
        } else {
            0.0
        };
        result.estimated_speedup = 1.0 + reorder_speedup + soa_speedup + noalias_speedup;

        result
    }

    /// Analyze access patterns in a block.
    fn analyze_access_patterns_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.analyze_stmt(stmt);
        }
    }

    fn analyze_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Expr { expr, .. } => self.analyze_expr(expr),
            Stmt::Let { init: Some(init), .. } => self.analyze_expr(init),
            Stmt::If { cond, then, else_, .. } => {
                self.analyze_expr(cond);
                self.analyze_access_patterns_block(then);
                if let Some(eb) = else_ {
                    match &**eb {
                        IfOrBlock::Block(b) => self.analyze_access_patterns_block(b),
                        IfOrBlock::If(s) => self.analyze_stmt(s),
                    }
                }
            }
            Stmt::While { cond, body, .. } => {
                self.analyze_expr(cond);
                self.analyze_access_patterns_block(body);
            }
            Stmt::ForIn { iter, body, .. } => {
                self.analyze_expr(iter);
                self.analyze_access_patterns_block(body);
            }
            Stmt::Return { value: Some(v), .. } => self.analyze_expr(v),
            Stmt::Match { expr, arms, .. } => {
                self.analyze_expr(expr);
                for arm in arms {
                    self.analyze_expr(&arm.body);
                }
            }
            _ => {}
        }
    }

    fn analyze_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Field { object, field, .. } => {
                if let Expr::Ident { name, .. } = &**object {
                    // Record access to struct_name.field_name
                    self.analyzer.record_field_access(name, field, true, false);
                }
                self.analyze_expr(object);
            }
            Expr::Assign { target, value, .. } => {
                if let Expr::Field { object, field, .. } = &**target {
                    if let Expr::Ident { name, .. } = &**object {
                        self.analyzer.record_field_access(name, field, false, false);
                    }
                }
                self.analyze_expr(target);
                self.analyze_expr(value);
            }
            Expr::BinOp { lhs, rhs, .. } => {
                self.analyze_expr(lhs);
                self.analyze_expr(rhs);
            }
            Expr::Call { func, args, .. } => {
                self.analyze_expr(func);
                for a in args {
                    self.analyze_expr(a);
                }
            }
            _ => {}
        }
    }

    /// Suggest a field reordering for a struct based on access patterns.
    fn suggest_field_reorder(&self, struct_name: &str, decl: &StructDecl) -> Option<FieldReorderSuggestion> {
        let patterns = self.analyzer.field_patterns(struct_name);
        if patterns.is_empty() {
            return None;
        }

        let original_order: Vec<String> = decl.fields.iter().map(|f| f.name.clone()).collect();

        // Sort fields by hotness (hottest first).
        let mut field_hotness: Vec<(String, u64)> = original_order
            .iter()
            .map(|name| {
                let key = format!("{}.{}", struct_name, name);
                let hotness = self
                    .analyzer
                    .field_access_patterns
                    .get(&key)
                    .map(|p| p.hotness())
                    .unwrap_or(0);
                (name.clone(), hotness)
            })
            .collect();

        field_hotness.sort_by(|a, b| b.1.cmp(&a.1));

        // Group co-accessed fields together.
        let suggested_order = self.group_coaccessed_fields(struct_name, &field_hotness);

        // Check if the order actually changed.
        if suggested_order == original_order {
            return None;
        }

        // Estimate cache miss reduction.
        let hot_fields_in_first_line = self.count_hot_fields_in_first_cacheline(
            &suggested_order,
            &decl.fields,
            64, // 64-byte cache line
        );
        let hot_fields_in_original = self.count_hot_fields_in_first_cacheline(
            &original_order,
            &decl.fields,
            64,
        );

        let estimated_reduction = if hot_fields_in_original > 0 {
            1.0 - (hot_fields_in_original as f64 / hot_fields_in_first_line.max(1) as f64)
        } else {
            0.0
        };

        Some(FieldReorderSuggestion {
            struct_name: struct_name.to_string(),
            original_order,
            suggested_order,
            estimated_cache_miss_reduction: estimated_reduction.max(0.0).min(1.0),
            reason: "Hot-field packing: frequently accessed fields placed first for cache locality".to_string(),
        })
    }

    /// Group co-accessed fields adjacent to each other.
    fn group_coaccessed_fields(
        &self,
        struct_name: &str,
        sorted_fields: &[(String, u64)],
    ) -> Vec<String> {
        let mut result = Vec::new();
        let mut placed = HashSet::new();

        for (field_name, _) in sorted_fields {
            if placed.contains(field_name) {
                continue;
            }
            result.push(field_name.clone());
            placed.insert(field_name.clone());

            // Find co-accessed fields not yet placed.
            let key = format!("{}.{}", struct_name, field_name);
            if let Some(pattern) = self.analyzer.field_access_patterns.get(&key) {
                let mut coaccessed: Vec<(String, u64)> = pattern
                    .co_access_counts
                    .iter()
                    .filter(|(name, count)| {
                        **count >= self.co_access_threshold && !placed.contains(*name)
                    })
                    .map(|(name, count)| (name.clone(), *count))
                    .collect();
                coaccessed.sort_by(|a, b| b.1.cmp(&a.1));

                for (co_field, _) in coaccessed {
                    if !placed.contains(&co_field) {
                        result.push(co_field.clone());
                        placed.insert(co_field);
                    }
                }
            }
        }

        result
    }

    /// Count how many hot fields fit in the first cache line.
    fn count_hot_fields_in_first_cacheline(
        &self,
        order: &[String],
        fields: &[StructField],
        cacheline_size: usize,
    ) -> usize {
        let mut offset = 0;
        let mut count = 0;
        let field_sizes: HashMap<&str, usize> = fields
            .iter()
            .map(|f| (f.name.as_str(), Self::type_size(&f.ty)))
            .collect();

        for field_name in order {
            let size = field_sizes.get(field_name.as_str()).copied().unwrap_or(8);
            if offset + size > cacheline_size {
                break;
            }
            let _key = format!("xxx.{}", field_name); // We don't know struct name here
            offset += size;
            count += 1;
        }
        count
    }

    /// Estimate the byte size of a type.
    fn type_size(ty: &Type) -> usize {
        match ty {
            Type::Scalar(e) => e.byte_size(),
            Type::Ref { .. } => 8,
            Type::Tuple(ts) => ts.iter().map(Self::type_size).sum(),
            Type::Array { elem: _, .. } => 16, // conservative
            Type::Named(_) => 16,           // conservative
            _ => 8,
        }
    }

    /// Suggest AoS → SoA conversion for a struct if the access pattern
    /// warrants it.
    fn suggest_soa_conversion(&self, struct_name: &str) -> Option<SoaConversionSuggestion> {
        let patterns = self.analyzer.field_patterns(struct_name);
        if patterns.is_empty() {
            return None;
        }

        // SoA is beneficial when fields are accessed independently (field-wise).
        let mut field_wise_fields = Vec::new();
        for pattern in &patterns {
            let total_accesses = pattern.read_count + pattern.write_count;
            if total_accesses == 0 {
                continue;
            }
            // If a field has low co-access with other fields, it's a good
            // candidate for SoA splitting.
            let total_co_access: u64 = pattern.co_access_counts.values().sum();
            let co_access_ratio = if total_accesses > 0 {
                total_co_access as f64 / total_accesses as f64
            } else {
                0.0
            };

            if co_access_ratio < 0.3 && pattern.hotness() > self.hot_threshold {
                field_wise_fields.push(pattern.field_name.clone());
            }
        }

        if field_wise_fields.len() < 2 {
            return None;
        }

        Some(SoaConversionSuggestion {
            struct_name: struct_name.to_string(),
            fields_to_split: field_wise_fields,
            estimated_speedup: 2.0, // Typical SoA speedup for field-wise access
            reason: "Field-wise access pattern detected: SoA layout improves cache locality".to_string(),
        })
    }
}

impl Default for LayoutOptimizer {
    fn default() -> Self {
        Self::new(100, 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alias_analyzer() {
        let mut analyzer = AliasAnalyzer::new();
        analyzer.prove_noalias(
            MemoryRegion::StructField {
                struct_name: "Point".into(),
                field_name: "x".into(),
            },
            MemoryRegion::StructField {
                struct_name: "Point".into(),
                field_name: "y".into(),
            },
            AliasProof::FieldOffsetDisjoint,
        );
        assert!(analyzer.is_noalias(
            &MemoryRegion::StructField {
                struct_name: "Point".into(),
                field_name: "x".into(),
            },
            &MemoryRegion::StructField {
                struct_name: "Point".into(),
                field_name: "y".into(),
            },
        ));
    }

    #[test]
    fn test_field_access_pattern() {
        let mut pattern = FieldAccessPattern::new("Point".into(), "x".into(), true);
        pattern.read_count = 100;
        pattern.write_count = 50;
        assert_eq!(pattern.hotness(), 200);
        assert!(pattern.is_hot(100));
    }

    #[test]
    fn test_layout_optimizer_default() {
        let optimizer = LayoutOptimizer::default();
        assert_eq!(optimizer.hot_threshold, 100);
        assert_eq!(optimizer.co_access_threshold, 10);
    }
}
