Maintainer Practices

This document defines the operational practices for maintainers of the Jules compiler and runtime.

Jules is a compiler system with multiple execution backends (AST, IR, VM, JIT) and hardware-aware optimization layers. Mistakes in maintenance can propagate into generated machine code.

Maintain responsibly.

1. Core Responsibility

Maintainers are responsible for preserving:

Semantic correctness of the IR pipeline
Determinism of execution under identical inputs
Safety of optimization and JIT transformations
Stability of compiler output across versions
Integrity of the dependency and build system

Performance improvements must never compromise correctness.

2. Code Modification Rules
2.1 No Silent Behavioral Changes

Any change that affects:

IR lowering
optimizer behavior
type system rules
execution semantics
JIT code generation

must be explicitly documented.

If behavior changes, even if “obviously correct,” it is still a breaking change.

2.2 No Undocumented Deletions

Do not remove:

IR instructions
optimizer rules
AST constructs
runtime behaviors
benchmark definitions

without:

a clear justification
dependency impact analysis
migration or replacement path
2.3 Prefer Refactoring Over Rewriting

Rewrites are high-risk in a multi-layer compiler.

Preferred order:

Local refactor
Pass-level adjustment
Pipeline reordering
Full subsystem rewrite (last resort)
3. Compiler Invariants

The following invariants must always hold:

IR must remain well-formed SSA
All values must have a single definition point
Optimizations must preserve observable semantics
Execution backends must agree on program meaning
Undefined behavior must not propagate between layers

If an invariant cannot be preserved:

stop the change
isolate the issue
propose architectural adjustment instead
4. Testing Requirements

All changes must pass:

cargo test --lib
IR validation suite
optimizer correctness tests
deterministic execution checks (where applicable)

For compiler-core changes:

run full benchmark suite (if performance-relevant)
verify interpreter vs IR vs JIT agreement on outputs

If outputs differ:

the bug is in the compiler, not the test
5. Performance Changes

Performance improvements must be:

measurable
reproducible
benchmark-backed

Do not accept:

micro-optimizations without profiling evidence
“expected speedups”
theoretical improvements without runtime confirmation

If performance increases but correctness becomes uncertain:

reject the change

6. Optimization Safety Rules

Optimizers must:

preserve semantics exactly
never assume undefined behavior unless explicitly modeled
avoid cross-pass state leakage
be deterministic given identical IR input

Forbidden:

speculative rewrites without validation
hidden runtime-dependent transformations
non-replayable stochastic optimizations without seed control
7. JIT and Code Generation

JIT compilation must:

be reproducible given same IR input
respect memory and type safety contracts
not introduce observable race conditions
be isolated from optimizer heuristics

Any deviation between:

interpreter output
IR interpreter output
JIT output

must be treated as a critical defect.

8. Dependency Management

Before adding a dependency:

evaluate necessity
inspect transitive dependency graph
verify maintenance status
ensure no runtime code execution during build
prefer internal implementations for compiler-critical logic

If a dependency introduces ambiguity in behavior:

do not add it

9. Review Standards

All code reviews must verify:

correctness of logic, not just style
IR consistency impact
optimizer safety implications
memory model correctness
absence of hidden side effects

Approval requires at least:

1 maintainer familiar with IR pipeline
1 maintainer familiar with runtime/JIT layer (for cross-cutting changes)
10. Release Discipline

Before release:

verify full test suite pass
verify benchmark regression check
confirm IR / JIT / interpreter agreement
audit recent optimizer changes
validate no accidental semantic drift

If uncertainty exists:

delay release

11. Incident Handling

If a critical issue is discovered:

freeze affected pipeline layer (AST / IR / JIT / optimizer)
reproduce minimal failing case
identify semantic boundary violation
patch with smallest safe change
add regression test
document root cause

Do not patch symptoms in multiple layers independently.

12. Maintainer Mindset

Maintain a compiler, not a codebase.

Assume:

inputs are adversarial
optimizations can be wrong
performance changes can hide correctness bugs
small changes can have system-wide effects

If a change feels “local,” verify it is not global.

13. Final Rule

If a change makes the system faster but less certain,

it is not an optimization.

It is a risk.
