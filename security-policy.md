# security-policy.md

# Purpose

This document defines the dependency and supply-chain security rules for the compiler project.

Compilers are high-trust software.

A compromised compiler can:
- inject malicious code
- corrupt binaries
- leak source code
- create non-deterministic behavior
- undermine the integrity of the entire ecosystem

Security is not optional infrastructure.
Security is part of correctness.

---

# 1. Security Philosophy

## 1.1 Assume External Code Is Hostile

Every dependency:
- expands attack surface
- increases maintenance burden
- introduces supply-chain risk
- creates hidden execution paths

No dependency is trusted automatically.

Popularity is not evidence of safety.

---

## 1.2 Minimize Total Dependencies

The best dependency:
- does not exist

Prefer:
- standard library implementations
- small audited libraries
- internal implementations for critical systems

Avoid:
- massive frameworks
- deep dependency trees
- unnecessary abstraction libraries
- runtime package resolution

Complex dependency graphs reduce auditability.

---

## 1.3 Reproducibility Is Mandatory

Builds must be:
- deterministic
- reproducible
- version-locked
- environment-independent

Two identical source trees must produce identical binaries.

If builds are non-deterministic:
the toolchain cannot be trusted.

---

# 2. Dependency Admission Policy

A dependency may only be added if ALL conditions are met:

## Required Criteria

- actively maintained
- minimal transitive dependencies
- compatible license
- reviewed source code
- stable release history
- deterministic behavior
- no unnecessary runtime execution

---

## Automatic Rejection Criteria

Dependencies are rejected if they:
- execute install-time scripts
- download remote code during build
- require telemetry
- depend on abandoned projects
- introduce large dependency trees
- rely on dynamic runtime patching
- use obfuscation
- hide unsafe/native behavior
- perform undocumented network access

If the dependency behaves like malware:
treat it like malware.

---

# 3. Approved Dependency Categories

Allowed:
- parsing utilities
- cryptographic primitives from trusted sources
- compression libraries
- deterministic serialization libraries
- low-level systems utilities

Restricted:
- networking frameworks
- package managers
- scripting runtimes
- reflection-heavy systems
- automatic code generators

Forbidden:
- analytics SDKs
- telemetry frameworks
- ad/tracking systems
- auto-updating libraries
- cloud-coupled infrastructure

A compiler is not a social media platform.

---

# 4. Build System Security

## Rules

Build systems must:
- avoid remote execution
- avoid downloading dependencies during compilation
- pin exact versions
- support offline builds
- verify hashes/signatures where possible

Build scripts must remain:
- readable
- deterministic
- auditable

---

## Forbidden Build Behavior

Never:
- execute arbitrary shell pipelines from dependencies
- fetch latest-version packages automatically
- allow silent dependency upgrades
- compile unverified native code

"Latest" is not a version strategy.

---

# 5. Compiler Trust Boundaries

The compiler must assume:
- source files may be malicious
- build inputs may be malformed
- plugins may be hostile
- generated ASTs may be corrupted
- optimization passes may encounter invalid state

The compiler must:
- fail safely
- validate aggressively
- isolate unsafe operations
- avoid undefined internal states

A malformed file should produce an error.
Not summon undefined behavior demons from the heap.

---

# 6. Unsafe Code Policy

Unsafe/native code:
- must be isolated
- must be documented
- must justify existence
- must include bounds validation
- must minimize pointer aliasing risks

Unsafe code paths require:
- profiling justification
- security review
- memory validation strategy

Performance gains do not excuse memory corruption.

---

# 7. Third-Party Code Review Requirements

Before adoption:
- inspect repository history
- inspect maintainer activity
- inspect release cadence
- inspect issue tracker
- inspect dependency graph
- inspect unsafe/native sections

High-risk indicators:
- sudden ownership transfer
- unexplained rewrites
- mass minified commits
- suspicious release timing
- dependency explosions

Supply-chain attacks often look normal until they are not.

---

# 8. Dependency Update Policy

Dependency updates:
- must be deliberate
- must be reviewed
- must be benchmarked
- must pass reproducibility checks

Never:
- auto-merge dependency updates
- blindly trust patch versions
- upgrade during release stabilization

Small version numbers can contain enormous mistakes.

---

# 9. Security Validation

Security checks should include:
- static analysis
- fuzz testing
- malformed input testing
- memory validation
- reproducibility verification
- dependency auditing

Critical systems should be fuzzed continuously.

Users are extremely creative at generating invalid inputs accidentally.

Attackers are even more creative on purpose.

---

# 10. Incident Response

If a dependency is compromised:
1. freeze releases
2. isolate affected components
3. identify exposure scope
4. revoke compromised artifacts
5. rebuild from verified sources
6. document the incident publicly

Security failures hidden silently become future disasters.

---

# 11. Final Rule

Convenience is temporary.

Compromise is permanent.

The compiler is part of the trusted computing chain.
Treat it accordingly.
