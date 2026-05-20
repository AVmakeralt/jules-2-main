# deslop.md

## Purpose

This document defines the standards required to convert unstable, bloated, AI-generated, overengineered, or otherwise "slop" code into production-grade software.

"Works on my machine" is not a success condition.

The objective is:
- deterministic behavior
- maintainable architecture
- measurable performance
- controlled complexity
- graceful failure
- minimal hidden behavior

If the CPU spikes by 3% and the application enters a death spiral, the code is slop.

---

# 1. Core Principles

## 1.1 Simplicity Is a Performance Feature

Prefer:
- explicit logic
- linear control flow
- minimal abstraction depth
- predictable memory behavior

Avoid:
- abstraction pyramids
- dependency nesting hell
- magical frameworks
- runtime reflection abuse
- "smart" helper layers nobody understands

Every layer must justify its existence.

---

## 1.2 Determinism Over Cleverness

Code should behave:
- consistently
- predictably
- debuggably

Ban:
- hidden mutations
- implicit global state
- timing-sensitive logic
- random retry loops
- side effects disguised as helpers

If execution order matters, make it explicit.

---

## 1.3 Failure Must Be Contained

Good systems degrade gracefully.

Bad systems:
- panic from one malformed packet
- deadlock under load
- recursively retry until the machine cries
- allocate memory until the OS intervenes like an exhausted parent

Every subsystem must define:
- failure boundaries
- timeout behavior
- retry limits
- memory ceilings
- recovery strategy

---

# 2. Slop Detection

## Immediate Red Flags

### Architecture
- massive god classes/files
- circular dependencies
- deeply nested inheritance
- framework-driven design
- business logic inside UI code
- unnecessary microservices

### Performance
- allocations inside hot loops
- excessive copying
- string-heavy processing
- cache-unfriendly layouts
- blocking I/O in critical paths
- polling instead of signaling

### Reliability
- swallowed exceptions
- global mutable state
- race conditions
- uncontrolled thread spawning
- infinite retries
- no backpressure handling

### AI Slop Indicators
- repeated helper wrappers
- meaningless abstractions
- duplicate utility functions
- inconsistent naming
- giant functions with comment-generated structure
- unnecessary async usage
- excessive dependency usage for trivial tasks

If removing 40% of the code improves readability instantly:
the codebase is contaminated.

---

# 3. Deslopification Process

## Step 1: Establish Ground Truth

Before changing anything:
- define expected behavior
- create reproducible tests
- collect benchmarks
- profile CPU/memory usage
- identify hot paths

Do not optimize based on vibes.

Humans love doing this. Humans are wrong constantly.

---

## Step 2: Remove Architectural Noise

Delete:
- dead abstractions
- wrapper layers
- speculative extensibility
- unnecessary interfaces
- duplicated logic
- fake generic systems

A smaller system is easier to optimize.

---

## Step 3: Flatten Critical Paths

Critical execution paths should:
- minimize branching
- minimize allocations
- avoid virtual dispatch where possible
- avoid unnecessary synchronization
- maximize cache locality

Hot code must look boring.

Boring code scales.

---

## Step 4: Control Memory Behavior

Memory instability destroys performance.

Prefer:
- stack allocation where possible
- object pooling for hot allocations
- contiguous memory layouts
- arena allocators for predictable lifetimes
- explicit ownership rules

Avoid:
- allocation churn
- hidden copies
- fragmented object graphs
- recursive allocation patterns

Garbage collectors are not magical cleanup fairies.

---

## Step 5: Remove Latency Multipliers

Latency compounds exponentially.

Audit:
- blocking operations
- lock contention
- synchronous disk/network access
- excessive serialization
- retry storms

One slow subsystem can poison the entire runtime.

---

# 4. Performance Standards

## Required Metrics

Measure:
- throughput
- average latency
- p95/p99 latency
- memory usage
- allocation rate
- startup time
- tail behavior under load

Average latency alone is propaganda.

---

## Benchmark Rules

Benchmarks must:
- warm up the runtime
- isolate variables
- use realistic workloads
- run repeatedly
- record variance

Never trust:
- single-run benchmarks
- debug builds
- synthetic nonsense detached from production behavior

---

# 5. Concurrency Rules

Threads are expensive.
Locks are expensive.
Deadlocks are forever.

Prefer:
- immutable data
- message passing
- bounded queues
- lock-free structures when justified
- work-stealing schedulers

Avoid:
- shared mutable state
- giant mutexes
- thread-per-task designs
- uncontrolled async spawning

Concurrency is not free performance.
It is distributed debugging trauma.

---

# 6. Dependency Policy

Every dependency:
- increases attack surface
- increases compile time
- increases maintenance burden
- increases instability risk

Before adding a dependency:
1. Is it necessary?
2. Can the functionality be implemented simply in-house?
3. Is the dependency maintained?
4. Is the dependency performant?
5. Does it pull 47 transitive packages to reverse a string?

Modern software ecosystems have normalized absurdity.

Resist.

---

# 7. Code Review Requirements

Code reviews must reject:
- unnecessary complexity
- hidden behavior
- unmeasured optimization claims
- architecture without justification
- premature abstraction
- performance claims without profiling

"Cleaner" does not mean faster.
"More generic" does not mean better.

---

# 8. Definition of Done

Code is considered deslopped when:
- behavior is deterministic
- benchmarks are reproducible
- failures are contained
- architecture is understandable
- hot paths are optimized
- memory behavior is controlled
- profiling confirms improvements
- maintenance cost is reduced

If future developers can understand it without archaeological excavation:
success.

---

# 9. Final Rule

The machine is literal.

It does not care:
- how elegant the abstraction is
- how modern the framework is
- how many patterns were used
- how clever the architecture diagram looked

It executes instructions.

Optimize accordingly.
