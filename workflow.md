# Jules Development Workflow

This document defines the mandatory operational procedures for the Jules project. All development, refactoring, and hardware-software co-design must adhere to these steps.

## 1. Implementation Integrity (The No-Stub Rule)
Jules is a zero-tolerance system for runtime failure.
- **No Placeholders:** `todo!()`, `unimplemented!()`, or `/* implementation here */` are strictly forbidden. 
- **Full Definition:** Every function, struct, and IR instruction must be fully implemented with correct types and safety documentation.
- **Partial Work:** If a task is too large for one turn, you must commit a functional sub-module that passes `cargo check` and contains no stubs.

## 2. The Deletion & Refactor Protocol (Consent First)
Because Jules has complex dependencies between the JIT, IR, and JHAL, silent deletions are catastrophic.
- **Deletions:** You must not delete existing logic, files, or IR instructions without an **Impact Report**.
- **Impact Report Requirements:**
    1. **Target:** What is being removed?
    2. **Justification:** Why is it obsolete or harmful?
    3. **Dependency Check:** List every module currently importing this target.
    4. **Replacement:** Point to the new logic that satisfies the original requirement.
- **Consent:** You must wait for the user to state "Proceed" or "Delete Confirmed" before executing the change.

## 3. The "Spec-First" Development Cycle
Before writing any code for a new feature or optimization:
1. **Reference the Spec:** Identify the section in `jules-ir-specification.pdf` or `JHAL Architecture` that governs this area.
2. **Technical Plan:** Outline the lowering path (Source -> IR -> Machine Code) or the register-level impact (JHAL).
3. **Verify Unified IR:** Explain how this feature fits into the existing unified instruction set without introducing "Compiler Schizophrenia."

## 4. Mandatory Verification Loop
Before declaring a task "Completed," the agent must:
1. **Syntactic Check:** Run `cargo check`.
2. **Invariant Check:** Run `cargo test --lib validators` (SSA, Type, Effect, and Ownership).
3. **Performance Check:** If modifying the Prophetic Prefetch Engine, verify that `estimated_cycle_benefit()` remains positive against existing traces.
4. **Safety Check:** If in `src/jhal`, audit for any accidental `std` or `alloc` usage.

## 5. Worklog Discipline
Every session MUST conclude with an update to `worklog.md`. 
- **Save Point:** Document the exact file and line number where the next session should begin.
- **Audit:** Explicitly state: "Verified: No stubs remain and no unauthorized deletions occurred."
