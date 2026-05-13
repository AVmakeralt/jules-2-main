# Jules Development Worklog

> **Protocol:** > 1. **New Entry:** Append to the TOP of this file before every task.
> 2. **Check-in:** Summarize the current "Ununified State" (what is broken/in-progress).
> 3. **Validation:** Explicitly confirm no `todo!()` or `unimplemented!()` stubs remain.
> 4. **Consent:** Record any major deletions approved by the developer.

---

## [YYYY-MM-DD] - [Title of Current Task]
**Status:** 🟡 In Progress | 🟢 Completed | 🔴 Blocked
**System Layer:** [e.g., JHAL / Unified-IR / JIT / Prefetch]

### 1. The Mission
- [ ] Primary Goal (e.g., Implement `APIC` timer frequency scaling)
- [ ] Secondary Goal (e.g., Add `cfg_validator` check for unreachable blocks)

### 2. Invariants & Guardrails
- **Invariants Touched:** [e.g., SSA Form, Hardware Interrupt Latency]
- **Safety Check:** [e.g., Verified zero-heap in `src/jhal`]

### 3. Execution Log
- **Change:** Describe specific logic modification.
- **Why:** The architectural justification for this path.
- **Deletions:** [List any files/structs removed and reference dev consent].

### 4. Current Save Point (Context for Next Session)
- **Current State:** [e.g., "The IR lowering works, but the emitter produces invalid x86_64 for `VectorAdd`."]
- **Next Immediate Step:** [e.g., Debug `src/jit/emitter.rs` line 442.]
- **Warnings:** [e.g., "Do not touch `local_apic.rs` until the timer overflow bug is mapped."]

---

## [2026-05-12] - Project Initialization
**Status:** 🟢 Completed
**System Layer:** Project Governance

### 1. The Mission
- [x] Establish `agent.md` governance.
- [x] Create `workflow.md` for strict stub/deletion rules.
- [x] Initialize `worklog.md` for state tracking.

### 2. Invariants & Guardrails
- **Invariants Touched:** Project-wide development standards.
- **Safety Check:** All new markdown files follow the Jules "Context-First" philosophy.

### 3. Execution Log
- **Change:** Created root governance files.
- **Why:** To prevent agent hallucinations and ensure implementation completeness in a complex Rust/Bare-metal environment.
- **Deletions:** None.

### 4. Current Save Point
- **Current State:** The "Constitution" for Jules development is set. The agent is now aware of the `jules-ir-specification.pdf` and JHAL architecture requirements.
- **Next Immediate Step:** Begin implementation or refactor of specific crates using the new Workflow.
- **Warnings:** Ensure the agent always reads the last entry of this log before starting.
