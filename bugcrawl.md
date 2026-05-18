# bugcrawl.md

## Purpose

This document defines a debugging protocol for agents working in large, complex repositories.

Goal:
Before any change is committed, the agent must determine:
- What breaks if this change is applied?
- What depends on the modified code?
- How can we validate the fix safely?
- What hidden side effects exist?

This is not optional. Large repos are not forgiving environments.

---

## 1. Pre-Change Analysis (READ BEFORE EDITING ANYTHING)

Before modifying any file, the agent MUST:

### 1.1 Identify Scope
- What file(s) are being changed?
- What modules depend on this code?
- Is this part of a public API, shared utility, or internal logic?

### 1.2 Dependency Crawl
Ask:
- What imports this?
- What calls this function/class?
- What downstream systems rely on this output?

If unsure:
- Search repo references (grep/ripgrep style reasoning)
- Assume more dependencies exist than you found

Rule:
> If you think you found all dependencies, you probably didn’t.

---

## 2. Change Impact Simulation

Before writing code, simulate mentally:

### Ask:
- If I change this behavior, what breaks immediately?
- What breaks silently (worse category)?
- What edge cases rely on the old behavior?
- What tests might fail?
- What production flows might degrade?

### Special attention:
- Authentication / security logic
- Data models / schemas
- Shared utilities
- Async flows
- Caching layers
- Error handling paths

---

## 3. Safe Modification Rules

When editing code:

### 3.1 Minimal Change Principle
- Change the smallest possible surface area
- Avoid “cleanup while I’m here” edits unless explicitly required

### 3.2 Preserve Contracts
Do NOT change:
- Function signatures (unless necessary)
- Return types
- Public behavior
- Error semantics

Unless you explicitly verify all callers.

---

## 4. Debugging Protocol

If a bug exists:

### Step 1: Reproduce mentally
- Trace execution path
- Identify failure point
- Determine expected vs actual behavior

### Step 2: Isolate cause
Ask:
- Is this logic wrong?
- Is input unexpected?
- Is state corrupted?
- Is async timing involved?

### Step 3: Validate fix
Before claiming success:
- Does this fix the root cause or just the symptom?
- Does it introduce new edge cases?
- Does it rely on hidden assumptions?

---

## 5. Pre-Commit Checklist (MANDATORY)

Before any commit is considered “safe”:

- [ ] I understand what this change affects
- [ ] I checked dependencies (direct and indirect)
- [ ] I considered edge cases
- [ ] I did not break public interfaces
- [ ] I validated failure modes
- [ ] I did not introduce silent behavior changes
- [ ] I confirmed fix does not shift bug elsewhere

If any box is unchecked:
DO NOT COMMIT.

---

## 6. Risk Classification

Classify every change:

### LOW RISK
- Local refactor
- No API changes
- No shared utilities touched

### MEDIUM RISK
- Internal logic change
- Shared helper modified
- Minor behavioral changes

### HIGH RISK
- Auth, payments, data integrity
- Public APIs
- Core services
- Anything async + shared state

High risk changes require extra review, even for agents.

---

## 7. “Will This Break Anything?” Heuristic

Ask this repeatedly:

> If this function behaves differently, who notices first?

If answer is unclear:
ASSUME something breaks.

Because it will. Quietly. In production. On a Friday.

---

## 8. Debugging Mindset

- Assume hidden coupling exists
- Assume someone relied on the bug
- Assume tests are incomplete
- Assume logs are lying by omission

You are not just fixing code.
You are negotiating with a fragile system held together by history and optimism.

---

## 9. Final Rule

If uncertain:

> Prefer analysis over modification.

A correct diagnosis is better than a rushed patch that spreads the problem.
