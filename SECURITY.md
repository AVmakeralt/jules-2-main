# Security Policy: Jules Programming Language

## 1. Supported Versions
Security updates are provided only for actively maintained branches. We recommend all users migrate to the latest stable release.

| Version | Status | Security Updates |
| :--- | :--- | :--- |
| **6.x (main)** |  Supported | Full updates (Compiler, IR, Runtime, Tooling) |
| **5.x** |  Maintenance | Critical-severity fixes only |
| **4.x** |  Maintenance | High-severity fixes only |
| **< 4.0** | End of Life | Unsupported |

---

## 2. Reporting a Vulnerability
We treat security issues in the Jules compiler as **correctness and trust-chain failures**. If the compiler produces code that deviates from defined semantics, it is a security risk.

### How to Report
Please report security issues **privately** to prevent "0-day" exploits:
* **Email:** `security@jules-lang.org` (Replace with actual contact)
* **Subject:** `SECURITY: [Short description of the issue]`
* **Requirement:** Do not open public issues until a fix has been released or the disclosure window has passed.

### Required Information
To help us triage, please include:
1. **Reproduction:** A minimal Jules source code example (`.jules`).
2. **Environment:** Compiler version/commit hash and Execution Mode (AST, IR, VM, or JIT).
3. **Behavior:** Detailed description of "Expected" vs "Actual" behavior.
4. **Context:** Is the issue deterministic, or does it depend on specific optimization flags (e.g., `-O3`)?

---

## 3. Scope & Classification
We evaluate vulnerabilities based on where they occur in the transformation pipeline.

### In-Scope Components
* **Frontend:** Parser, Type System, AST validation.
* **Intermediate Representation (IR):** SSA lowering, IR verification passes.
* **Optimizers:** E-graphs, superoptimizer, and ML-driven passes.
* **Execution:** Bytecode VM, tiered JIT compiler, and the Hardware Abstraction Layer (JHAL).
* **Memory:** Allocation strategies and Garbage Collection (GC) safety.

### Out-of-Scope
* Third-party dependencies not modified by the Jules project.
* User-written code explicitly marked as `unsafe`.
* Experimental or "unstable" feature branches.

---

## 4. Response & Disclosure

### Triage Timeline
| Severity | Initial Response | Resolution Goal |
| :--- | :--- | :--- |
| **Critical** | 48 Hours | Memory safety, RCE, or IR corruption. |
| **High** | 5 Business Days | Optimizer unsoundness or JIT miscompilation. |
| **Medium** | 10 Business Days | Compiler crashes (DoS) or malformed input handling. |
| **Low** | Best Effort | Minor edge cases with no path to exploitation. |

### Disclosure Policy
We follow **coordinated disclosure**. Vulnerabilities will be disclosed publicly after a fix is released or **90 days** have passed since the initial report. Extensions may be requested for complex architectural fixes.

---

## 5. Security Principles
The Jules compiler is governed by the following mandates:
1. **Untrusted Input:** The compiler treats all source code as untrusted.
2. **Semantic Preservation:** Transformation passes must never change program meaning.
3. **Safety First:** Performance optimizations are **invalid** if they compromise type or memory safety.
4. **Determinism:** Identical inputs and configurations must produce bit-identical output.

> **Final Note:** If a compiled program behaves differently than its defined semantics due to the compiler, that is considered a security issue.
