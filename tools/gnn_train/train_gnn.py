#!/usr/bin/env python3
"""
GNN E-Graph Superoptimizer — Training Pipeline

Trains a Graph Neural Network to predict optimal rewrite actions for e-graph
optimization. The model learns patterns like:
  - x+0 -> x (additive identity)
  - x*1 -> x (multiplicative identity)
  - x/2^k -> x>>k (strength reduction)
  - x-x -> 0 (self-subtraction)
  - --x -> x (double negation)

Architecture: AlphaGo-style GNN with:
  - Message-passing GNN layers (GCN-style)
  - Policy head: predicts best rewrite action
  - Value head: predicts expected improvement
  - ~100k parameters

After training, exports weights as a Rust source file with baked-in weights.
"""

import numpy as np
import sys
import os
import json
import time
from collections import defaultdict

# =====================================================================
# 1. Expression AST (mirrors Jules-2 Rust AST)
# =====================================================================

class Expr:
    pass

class VarExpr(Expr):
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name

class IntLitExpr(Expr):
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return str(self.value)

class BinOpExpr(Expr):
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs
    def __repr__(self):
        return f"({self.lhs} {self.op} {self.rhs})"

class UnOpExpr(Expr):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand
    def __repr__(self):
        return f"{self.op}({self.operand})"

# =====================================================================
# 2. Cost Model (mirrors Rust hardware cost model - Skylake)
# =====================================================================

COST_TABLE = {
    'add': 1, 'sub': 1, 'and': 1, 'or': 1, 'xor': 1,
    'shl': 1, 'shr': 1, 'bitand': 1, 'bitor': 1, 'bitxor': 1,
    'mul': 3, 'div': 35, 'rem': 35,
    'neg': 1, 'not': 1,
    'const': 0, 'var': 1, 'floordiv': 35,
    'eq': 1, 'ne': 1, 'lt': 1, 'le': 1, 'gt': 1, 'ge': 1,
}

def estimate_cost(expr):
    """Estimate cycle cost of an expression"""
    if isinstance(expr, IntLitExpr):
        return COST_TABLE['const']
    elif isinstance(expr, VarExpr):
        return COST_TABLE['var']
    elif isinstance(expr, BinOpExpr):
        lcost = estimate_cost(expr.lhs)
        rcost = estimate_cost(expr.rhs)
        op_cost = COST_TABLE.get(expr.op, 1)
        return lcost + rcost + op_cost
    elif isinstance(expr, UnOpExpr):
        ocost = estimate_cost(expr.operand)
        op_cost = COST_TABLE.get(expr.op, 1)
        return ocost + op_cost
    return 1

# =====================================================================
# 3. Rewrite Rules (mirrors Rust RewriteAction)
# =====================================================================

REWRITE_ACTIONS = [
    'Commute',          # 0: swap lhs/rhs for commutative ops
    'IdentityRight',    # 1: x+0->x, x*1->x, x/1->x
    'IdentityLeft',     # 2: 0+x->x, 1*x->x
    'AnnihilateRight',  # 3: x*0->0, x&0->0
    'AnnihilateLeft',   # 4: 0*x->0, 0&x->0
    'ConstantFold',     # 5: fold constant expressions
    'StrengthReduce',   # 6: x*2^k -> x<<k, x/2^k -> x>>k
    'Distribute',       # 7: x*(y+z) -> x*y+x*z
    'Factor',           # 8: x*y+x*z -> x*(y+z)
    'DoubleNegate',     # 9: --x -> x, !!x -> x
    'SelfIdentity',     # 10: x-x->0, x^x->0
    'Absorb',           # 11: x&(x|y)->x
]

COMMUTATIVE_OPS = {'add', 'mul', 'bitand', 'bitor', 'bitxor'}

def is_applicable(action_idx, expr):
    """Check if a rewrite action is applicable to an expression"""
    action = REWRITE_ACTIONS[action_idx]
    
    if not isinstance(expr, BinOpExpr) and not isinstance(expr, UnOpExpr):
        return action in ['ConstantFold']  # Can always try folding
    
    if action == 'Commute':
        return isinstance(expr, BinOpExpr) and expr.op in COMMUTATIVE_OPS
    
    elif action == 'IdentityRight':
        if isinstance(expr, BinOpExpr):
            if expr.op in ('add', 'sub') and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 0:
                return True
            if expr.op == 'mul' and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 1:
                return True
            if expr.op == 'div' and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 1:
                return True
        return False
    
    elif action == 'IdentityLeft':
        if isinstance(expr, BinOpExpr):
            if expr.op == 'add' and isinstance(expr.lhs, IntLitExpr) and expr.lhs.value == 0:
                return True
            if expr.op == 'mul' and isinstance(expr.lhs, IntLitExpr) and expr.lhs.value == 1:
                return True
        return False
    
    elif action == 'AnnihilateRight':
        if isinstance(expr, BinOpExpr):
            if expr.op == 'mul' and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 0:
                return True
            if expr.op == 'bitand' and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 0:
                return True
        return False
    
    elif action == 'AnnihilateLeft':
        if isinstance(expr, BinOpExpr):
            if expr.op == 'mul' and isinstance(expr.lhs, IntLitExpr) and expr.lhs.value == 0:
                return True
            if expr.op == 'bitand' and isinstance(expr.lhs, IntLitExpr) and expr.lhs.value == 0:
                return True
        return False
    
    elif action == 'ConstantFold':
        if isinstance(expr, BinOpExpr) and isinstance(expr.lhs, IntLitExpr) and isinstance(expr.rhs, IntLitExpr):
            return True
        if isinstance(expr, UnOpExpr) and isinstance(expr.operand, IntLitExpr):
            return True
        return False
    
    elif action == 'StrengthReduce':
        if isinstance(expr, BinOpExpr) and isinstance(expr.rhs, IntLitExpr):
            v = expr.rhs.value
            if v > 1 and (v & (v - 1)) == 0 and expr.op in ('mul', 'div'):
                return True
        return False
    
    elif action == 'Distribute':
        if isinstance(expr, BinOpExpr) and expr.op == 'mul':
            if isinstance(expr.rhs, BinOpExpr) and expr.rhs.op == 'add':
                return True
        return False
    
    elif action == 'Factor':
        if isinstance(expr, BinOpExpr) and expr.op == 'add':
            if isinstance(expr.lhs, BinOpExpr) and isinstance(expr.rhs, BinOpExpr):
                if expr.lhs.op == 'mul' and expr.rhs.op == 'mul':
                    # Check if common factor
                    if repr(expr.lhs.lhs) == repr(expr.rhs.lhs):
                        return True
        return False
    
    elif action == 'DoubleNegate':
        if isinstance(expr, UnOpExpr) and isinstance(expr.operand, UnOpExpr):
            if expr.op == expr.operand.op and expr.op in ('neg', 'not'):
                return True
        return False
    
    elif action == 'SelfIdentity':
        if isinstance(expr, BinOpExpr):
            if repr(expr.lhs) == repr(expr.rhs):
                if expr.op in ('sub', 'bitxor', 'eq', 'ne'):
                    return True
        return False
    
    elif action == 'Absorb':
        return False  # Complex, rarely applicable
    
    return False

def apply_rewrite(action_idx, expr):
    """Apply a rewrite action, return (new_expr, cost_delta) or None"""
    action = REWRITE_ACTIONS[action_idx]
    
    if action == 'Commute':
        if isinstance(expr, BinOpExpr) and expr.op in COMMUTATIVE_OPS:
            new_expr = BinOpExpr(expr.op, expr.rhs, expr.lhs)
            return new_expr, 0  # Commuting doesn't change cost
    
    elif action == 'IdentityRight':
        if isinstance(expr, BinOpExpr):
            if expr.op in ('add', 'sub') and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 0:
                return expr.lhs, -1  # Remove one op
            if expr.op == 'mul' and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 1:
                return expr.lhs, -1
            if expr.op == 'div' and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 1:
                return expr.lhs, -1  # div is expensive to remove
    
    elif action == 'IdentityLeft':
        if isinstance(expr, BinOpExpr):
            if expr.op == 'add' and isinstance(expr.lhs, IntLitExpr) and expr.lhs.value == 0:
                return expr.rhs, -1
            if expr.op == 'mul' and isinstance(expr.lhs, IntLitExpr) and expr.lhs.value == 1:
                return expr.rhs, -1
    
    elif action == 'AnnihilateRight':
        if isinstance(expr, BinOpExpr):
            if expr.op == 'mul' and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 0:
                return IntLitExpr(0), -estimate_cost(expr)
            if expr.op == 'bitand' and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 0:
                return IntLitExpr(0), -estimate_cost(expr)
    
    elif action == 'AnnihilateLeft':
        if isinstance(expr, BinOpExpr):
            if expr.op == 'mul' and isinstance(expr.lhs, IntLitExpr) and expr.lhs.value == 0:
                return IntLitExpr(0), -estimate_cost(expr)
            if expr.op == 'bitand' and isinstance(expr.lhs, IntLitExpr) and expr.lhs.value == 0:
                return IntLitExpr(0), -estimate_cost(expr)
    
    elif action == 'ConstantFold':
        if isinstance(expr, BinOpExpr) and isinstance(expr.lhs, IntLitExpr) and isinstance(expr.rhs, IntLitExpr):
            l, r = expr.lhs.value, expr.rhs.value
            result = None
            if expr.op == 'add': result = l + r
            elif expr.op == 'sub': result = l - r
            elif expr.op == 'mul': result = l * r
            elif expr.op == 'div' and r != 0: result = l // r
            elif expr.op == 'rem' and r != 0: result = l % r
            elif expr.op == 'bitand': result = l & r
            elif expr.op == 'bitor': result = l | r
            elif expr.op == 'bitxor': result = l ^ r
            elif expr.op == 'shl': result = l << r
            elif expr.op == 'shr': result = l >> r
            if result is not None:
                new_expr = IntLitExpr(result & ((1 << 128) - 1))
                old_cost = estimate_cost(expr)
                return new_expr, -old_cost
        if isinstance(expr, UnOpExpr) and isinstance(expr.operand, IntLitExpr):
            v = expr.operand.value
            if expr.op == 'neg':
                return IntLitExpr((-v) & ((1 << 128) - 1)), -1
            elif expr.op == 'not':
                return IntLitExpr((~v) & ((1 << 128) - 1)), -1
    
    elif action == 'StrengthReduce':
        if isinstance(expr, BinOpExpr) and isinstance(expr.rhs, IntLitExpr):
            v = expr.rhs.value
            if v > 1 and (v & (v - 1)) == 0:
                shift = int(np.log2(v))
                new_op = 'shl' if expr.op == 'mul' else 'shr'
                new_expr = BinOpExpr(new_op, expr.lhs, IntLitExpr(shift))
                old_cost = COST_TABLE.get(expr.op, 1)
                new_cost = COST_TABLE.get(new_op, 1)
                return new_expr, new_cost - old_cost
    
    elif action == 'Distribute':
        if isinstance(expr, BinOpExpr) and expr.op == 'mul':
            if isinstance(expr.rhs, BinOpExpr) and expr.rhs.op == 'add':
                a, b, c = expr.lhs, expr.rhs.lhs, expr.rhs.rhs
                new_expr = BinOpExpr('add', BinOpExpr('mul', a, b), BinOpExpr('mul', a, c))
                # Distribute adds 1 mul but removes 1 add - usually worse for cost
                return new_expr, 1  # Usually negative for speed
    
    elif action == 'Factor':
        if isinstance(expr, BinOpExpr) and expr.op == 'add':
            if isinstance(expr.lhs, BinOpExpr) and isinstance(expr.rhs, BinOpExpr):
                if expr.lhs.op == 'mul' and expr.rhs.op == 'mul':
                    if repr(expr.lhs.lhs) == repr(expr.rhs.lhs):
                        a, b, c = expr.lhs.lhs, expr.lhs.rhs, expr.rhs.rhs
                        new_expr = BinOpExpr('mul', a, BinOpExpr('add', b, c))
                        # Factoring removes 1 mul - usually better
                        return new_expr, -2
    
    elif action == 'DoubleNegate':
        if isinstance(expr, UnOpExpr) and isinstance(expr.operand, UnOpExpr):
            if expr.op == expr.operand.op and expr.op in ('neg', 'not'):
                return expr.operand.operand, -1
    
    elif action == 'SelfIdentity':
        if isinstance(expr, BinOpExpr) and repr(expr.lhs) == repr(expr.rhs):
            if expr.op == 'sub' or expr.op == 'bitxor':
                return IntLitExpr(0), -estimate_cost(expr)
    
    return None

# =====================================================================
# 4. Expression to Graph (mirrors Rust ProgramGraph)
# =====================================================================

# 29 op types + depth + num_children + is_constant + has_variable = 33
OP_TO_IDX = {
    'const_int': 0, 'const_float': 1, 'const_bool': 2,
    'var': 3,
    'add': 4, 'sub': 5, 'mul': 6, 'div': 7, 'rem': 8, 'floordiv': 9,
    'eq': 10, 'ne': 11, 'lt': 12, 'le': 13, 'gt': 14, 'ge': 15,
    'and': 16, 'or': 17,
    'bitand': 18, 'bitor': 19, 'bitxor': 20,
    'shl': 21, 'shr': 22,
    'neg': 23, 'not': 24,
    'deref': 25, 'ref': 26, 'refmut': 27,
    'other': 28,
}

FEATURE_DIM = 33

def expr_to_graph(expr, max_nodes=32):
    """Convert expression to graph features and adjacency"""
    nodes = []
    edges_src = []
    edges_dst = []
    action_masks = []
    
    def build(e, depth):
        idx = len(nodes)
        features = np.zeros(FEATURE_DIM, dtype=np.float64)
        
        if isinstance(e, IntLitExpr):
            features[0] = 1.0
            features[31] = 1.0  # is_constant
        elif isinstance(e, VarExpr):
            features[3] = 1.0
            features[32] = 1.0  # has_variable
        elif isinstance(e, BinOpExpr):
            op_idx = OP_TO_IDX.get(e.op, 28)
            features[op_idx] = 1.0
            features[30] = 2.0 / 2.0  # num_children normalized
        elif isinstance(e, UnOpExpr):
            op_idx = OP_TO_IDX.get(e.op, 28)
            features[op_idx] = 1.0
            features[30] = 1.0 / 2.0
        else:
            features[28] = 1.0
        
        features[29] = np.log1p(depth) / 5.0  # log depth normalized
        
        nodes.append(features)
        
        # Compute action mask
        mask = np.zeros(len(REWRITE_ACTIONS), dtype=np.float64)
        for a_idx in range(len(REWRITE_ACTIONS)):
            mask[a_idx] = 1.0 if is_applicable(a_idx, e) else 0.0
        action_masks.append(mask)
        
        if isinstance(e, BinOpExpr):
            l_idx = build(e.lhs, depth + 1)
            r_idx = build(e.rhs, depth + 1)
            edges_src.extend([idx, idx])
            edges_dst.extend([l_idx, r_idx])
        elif isinstance(e, UnOpExpr):
            o_idx = build(e.operand, depth + 1)
            edges_src.append(idx)
            edges_dst.append(o_idx)
        
        return idx
    
    build(expr, 0)
    
    n = len(nodes)
    if n == 0:
        n = 1
        nodes.append(np.zeros(FEATURE_DIM, dtype=np.float64))
        action_masks.append(np.zeros(len(REWRITE_ACTIONS), dtype=np.float64))
    
    # Pad to max_nodes
    while len(nodes) < max_nodes:
        nodes.append(np.zeros(FEATURE_DIM, dtype=np.float64))
        action_masks.append(np.zeros(len(REWRITE_ACTIONS), dtype=np.float64))
    
    # Build adjacency with self-loops
    adj = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    for s, d in zip(edges_src, edges_dst):
        if s < max_nodes and d < max_nodes:
            adj[s][d] = 1.0
            adj[d][s] = 1.0  # Undirected (add reverse)
    for i in range(n):
        adj[i][i] = 1.0  # Self-loops
    
    # Normalize: D^{-1/2} A D^{-1/2}
    degree = adj.sum(axis=1)
    degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    adj_norm = adj * np.outer(degree_inv_sqrt, degree_inv_sqrt)
    
    node_features = np.array(nodes[:max_nodes], dtype=np.float64)
    action_mask = np.array(action_masks[:max_nodes], dtype=np.float64)
    
    return node_features, adj_norm, action_mask, n  # n = actual number of nodes

# =====================================================================
# 5. Training Data Generation
# =====================================================================

def generate_expressions():
    """Generate a comprehensive set of training expressions"""
    exprs = []
    labels = []
    
    # x + 0 -> x (IdentityRight)
    exprs.append(BinOpExpr('add', VarExpr('x'), IntLitExpr(0)))
    labels.append(1)  # IdentityRight
    
    # x * 1 -> x (IdentityRight)
    exprs.append(BinOpExpr('mul', VarExpr('x'), IntLitExpr(1)))
    labels.append(1)
    
    # x / 1 -> x (IdentityRight)
    exprs.append(BinOpExpr('div', VarExpr('x'), IntLitExpr(1)))
    labels.append(1)
    
    # x - 0 -> x (IdentityRight)
    exprs.append(BinOpExpr('sub', VarExpr('x'), IntLitExpr(0)))
    labels.append(1)
    
    # 0 + x -> x (IdentityLeft)
    exprs.append(BinOpExpr('add', IntLitExpr(0), VarExpr('x')))
    labels.append(2)
    
    # 1 * x -> x (IdentityLeft)
    exprs.append(BinOpExpr('mul', IntLitExpr(1), VarExpr('x')))
    labels.append(2)
    
    # x * 0 -> 0 (AnnihilateRight)
    exprs.append(BinOpExpr('mul', VarExpr('x'), IntLitExpr(0)))
    labels.append(3)
    
    # x & 0 -> 0 (AnnihilateRight)
    exprs.append(BinOpExpr('bitand', VarExpr('x'), IntLitExpr(0)))
    labels.append(3)
    
    # 0 * x -> 0 (AnnihilateLeft)
    exprs.append(BinOpExpr('mul', IntLitExpr(0), VarExpr('x')))
    labels.append(4)
    
    # 3 + 5 -> 8 (ConstantFold)
    exprs.append(BinOpExpr('add', IntLitExpr(3), IntLitExpr(5)))
    labels.append(5)
    
    # 10 - 3 -> 7 (ConstantFold)
    exprs.append(BinOpExpr('sub', IntLitExpr(10), IntLitExpr(3)))
    labels.append(5)
    
    # 4 * 7 -> 28 (ConstantFold)
    exprs.append(BinOpExpr('mul', IntLitExpr(4), IntLitExpr(7)))
    labels.append(5)
    
    # x * 8 -> x << 3 (StrengthReduce)
    exprs.append(BinOpExpr('mul', VarExpr('x'), IntLitExpr(8)))
    labels.append(6)
    
    # x * 4 -> x << 2 (StrengthReduce)
    exprs.append(BinOpExpr('mul', VarExpr('x'), IntLitExpr(4)))
    labels.append(6)
    
    # x / 16 -> x >> 4 (StrengthReduce)
    exprs.append(BinOpExpr('div', VarExpr('x'), IntLitExpr(16)))
    labels.append(6)
    
    # x / 32 -> x >> 5 (StrengthReduce)
    exprs.append(BinOpExpr('div', VarExpr('x'), IntLitExpr(32)))
    labels.append(6)
    
    # x * 64 -> x << 6 (StrengthReduce)
    exprs.append(BinOpExpr('mul', VarExpr('x'), IntLitExpr(64)))
    labels.append(6)
    
    # x - x -> 0 (SelfIdentity)
    exprs.append(BinOpExpr('sub', VarExpr('x'), VarExpr('x')))
    labels.append(10)
    
    # x ^ x -> 0 (SelfIdentity)
    exprs.append(BinOpExpr('bitxor', VarExpr('x'), VarExpr('x')))
    labels.append(10)
    
    # --x -> x (DoubleNegate)
    exprs.append(UnOpExpr('neg', UnOpExpr('neg', VarExpr('x'))))
    labels.append(9)
    
    # !!x -> x (DoubleNegate)
    exprs.append(UnOpExpr('not', UnOpExpr('not', VarExpr('x'))))
    labels.append(9)
    
    # x * (y + z) -> x*y + x*z (Distribute)
    exprs.append(BinOpExpr('mul', VarExpr('x'), BinOpExpr('add', VarExpr('y'), VarExpr('z'))))
    labels.append(7)
    
    # x*y + x*z -> x*(y+z) (Factor)
    exprs.append(BinOpExpr('add', BinOpExpr('mul', VarExpr('x'), VarExpr('y')), BinOpExpr('mul', VarExpr('x'), VarExpr('z'))))
    labels.append(8)
    
    # (x+0) * 1 -> x (IdentityRight then IdentityRight)
    exprs.append(BinOpExpr('mul', BinOpExpr('add', VarExpr('x'), IntLitExpr(0)), IntLitExpr(1)))
    labels.append(1)  # First action: IdentityRight on the mul
    
    # ((x*1)+0)*(y+0) -> x*y
    exprs.append(BinOpExpr('mul',
        BinOpExpr('add', BinOpExpr('mul', VarExpr('x'), IntLitExpr(1)), IntLitExpr(0)),
        BinOpExpr('add', VarExpr('y'), IntLitExpr(0))))
    labels.append(1)
    
    # x + y (no improvement possible - negative example)
    exprs.append(BinOpExpr('add', VarExpr('x'), VarExpr('y')))
    labels.append(-1)  # No good action
    
    # x * y (no improvement)
    exprs.append(BinOpExpr('mul', VarExpr('x'), VarExpr('y')))
    labels.append(-1)
    
    # Commute: a + b -> b + a (applicable but no cost improvement)
    exprs.append(BinOpExpr('add', VarExpr('a'), VarExpr('b')))
    labels.append(-1)
    
    # x | 0 -> x (IdentityRight for bitor)
    exprs.append(BinOpExpr('bitor', VarExpr('x'), IntLitExpr(0)))
    labels.append(1)
    
    # x ^ 0 -> x (IdentityRight for bitxor)
    exprs.append(BinOpExpr('bitxor', VarExpr('x'), IntLitExpr(0)))
    labels.append(1)
    
    # Complex: (a+0)*(b*1)+(x-x)
    exprs.append(BinOpExpr('add',
        BinOpExpr('mul',
            BinOpExpr('add', VarExpr('a'), IntLitExpr(0)),
            BinOpExpr('mul', VarExpr('b'), IntLitExpr(1))),
        BinOpExpr('sub', VarExpr('x'), VarExpr('x'))))
    labels.append(1)
    
    # (((x+0)*1)+(y*0))*((z/1)+(w-w))
    exprs.append(BinOpExpr('mul',
        BinOpExpr('add',
            BinOpExpr('mul',
                BinOpExpr('add', VarExpr('x'), IntLitExpr(0)),
                IntLitExpr(1)),
            BinOpExpr('mul', VarExpr('y'), IntLitExpr(0))),
        BinOpExpr('add',
            BinOpExpr('div', VarExpr('z'), IntLitExpr(1)),
            BinOpExpr('sub', VarExpr('w'), VarExpr('w')))))
    labels.append(1)
    
    # x * 2 -> x << 1 (StrengthReduce with small power)
    exprs.append(BinOpExpr('mul', VarExpr('x'), IntLitExpr(2)))
    labels.append(6)
    
    # x / 2 -> x >> 1 (StrengthReduce with small power)
    exprs.append(BinOpExpr('div', VarExpr('x'), IntLitExpr(2)))
    labels.append(6)
    
    # x / 4 -> x >> 2
    exprs.append(BinOpExpr('div', VarExpr('x'), IntLitExpr(4)))
    labels.append(6)
    
    # x / 8 -> x >> 3
    exprs.append(BinOpExpr('div', VarExpr('x'), IntLitExpr(8)))
    labels.append(6)
    
    # x * 16 -> x << 4
    exprs.append(BinOpExpr('mul', VarExpr('x'), IntLitExpr(16)))
    labels.append(6)
    
    # x * 128 -> x << 7
    exprs.append(BinOpExpr('mul', VarExpr('x'), IntLitExpr(128)))
    labels.append(6)
    
    # 100 / 10 -> 10 (ConstantFold)
    exprs.append(BinOpExpr('div', IntLitExpr(100), IntLitExpr(10)))
    labels.append(5)
    
    # x & x -> x (SelfIdentity - bitand with self)
    exprs.append(BinOpExpr('bitand', VarExpr('x'), VarExpr('x')))
    labels.append(10)
    
    # x | x -> x (SelfIdentity - bitor with self)
    exprs.append(BinOpExpr('bitor', VarExpr('x'), VarExpr('x')))
    labels.append(10)
    
    # -(-y) -> y (DoubleNegate with variable)
    exprs.append(UnOpExpr('neg', UnOpExpr('neg', VarExpr('y'))))
    labels.append(9)
    
    # x * 256 -> x << 8 (StrengthReduce large)
    exprs.append(BinOpExpr('mul', VarExpr('x'), IntLitExpr(256)))
    labels.append(6)
    
    # x / 256 -> x >> 8
    exprs.append(BinOpExpr('div', VarExpr('x'), IntLitExpr(256)))
    labels.append(6)
    
    # More complex nested: (x+0)/(8*1) -> x>>3
    exprs.append(BinOpExpr('div',
        BinOpExpr('add', VarExpr('x'), IntLitExpr(0)),
        IntLitExpr(8)))
    labels.append(6)  # Best first action is strength reduce on div
    
    # (x*1)+0 -> x (identity cascades)
    exprs.append(BinOpExpr('add',
        BinOpExpr('mul', VarExpr('x'), IntLitExpr(1)),
        IntLitExpr(0)))
    labels.append(1)
    
    return exprs, labels

# =====================================================================
# 6. GNN Model (PyTorch, mirrors Rust architecture)
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    """GNN message-passing layer with optional attention"""
    def __init__(self, d_in, d_out, num_heads=4, use_attention=True):
        super().__init__()
        self.w_self = nn.Linear(d_in, d_out)
        self.w_neigh = nn.Linear(d_in, d_out)
        self.layer_norm = nn.LayerNorm(d_out)
        self.use_attention = use_attention
        if use_attention:
            self.attn_key = nn.Linear(d_in, num_heads, bias=False)
            self.attn_query = nn.Linear(d_in, num_heads, bias=False)
    
    def forward(self, h, adj):
        # h: (batch, N, d_in), adj: (batch, N, N)
        h_self = self.w_self(h)
        h_neigh = torch.bmm(adj, h)  # (batch, N, d_in)
        h_neigh = self.w_neigh(h_neigh)
        
        h_combined = h_self + h_neigh
        h_normed = self.layer_norm(h_combined)
        return F.relu(h_normed)

class GraphReadout(nn.Module):
    """Weighted mean pooling over nodes"""
    def __init__(self, d_h):
        super().__init__()
        self.w = nn.Linear(d_h, d_h)
    
    def forward(self, node_embeddings):
        # node_embeddings: (batch, N, d_h)
        scores = self.w(node_embeddings)  # (batch, N, d_h)
        weights = F.relu(scores)  # Non-negative weights
        weight_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (batch, 1, d_h)
        normalized_weights = weights / weight_sum  # (batch, N, d_h)
        graph_repr = (normalized_weights * node_embeddings).sum(dim=1)  # (batch, d_h)
        return graph_repr

class PolicyHead(nn.Module):
    """Predicts which rewrite action to apply"""
    def __init__(self, d_h, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(d_h, d_h)
        self.fc2 = nn.Linear(d_h, num_actions)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return F.log_softmax(logits, dim=-1)

class ValueHead(nn.Module):
    """Predicts expected improvement"""
    def __init__(self, d_h):
        super().__init__()
        self.fc1 = nn.Linear(d_h, d_h)
        self.fc2 = nn.Linear(d_h, 1)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(h))

class GNNEgraphModel(nn.Module):
    """Full GNN E-Graph Optimizer model"""
    def __init__(self, node_feature_dim=33, hidden_dim=96, num_layers=4,
                 num_heads=4, num_actions=12, use_attention=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        self.embedding = nn.Linear(node_feature_dim, hidden_dim)
        
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(GNNLayer(
                hidden_dim, hidden_dim, num_heads, use_attention
            ))
        
        self.readout = GraphReadout(hidden_dim)
        self.policy_head = PolicyHead(hidden_dim, num_actions)
        self.value_head = ValueHead(hidden_dim)
    
    def forward(self, node_features, adj, action_mask):
        """
        node_features: (batch, N, F)
        adj: (batch, N, N) normalized adjacency
        action_mask: (batch, N, num_actions) - which actions are applicable per node
        """
        batch_size = node_features.size(0)
        
        # Embed input features
        h = F.relu(self.embedding(node_features))
        
        # GNN message passing
        for layer in self.gnn_layers:
            h = layer(h, adj)
        
        # Graph readout
        graph_repr = self.readout(h)  # (batch, d_h)
        
        # Policy head
        log_policy = self.policy_head(graph_repr)  # (batch, num_actions)
        
        # Apply action mask (root node = node 0)
        # Use the action mask from the root node
        root_mask = action_mask[:, 0, :]  # (batch, num_actions)
        # Mask invalid actions with very negative logit
        masked_logits = log_policy.clone()
        masked_logits[root_mask == 0] = -1e9
        log_policy_masked = F.log_softmax(masked_logits, dim=-1)
        
        # Value head
        value = self.value_head(graph_repr)  # (batch, 1)
        
        return log_policy_masked, value.squeeze(-1)

def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =====================================================================
# 7. Training Loop
# =====================================================================

def train_model():
    print("=" * 70)
    print("  GNN E-Graph Superoptimizer — Training Pipeline")
    print("=" * 70)
    print()
    
    # Model config: ~100k parameters
    # hidden_dim=96, num_layers=4, num_heads=4
    # Params: embedding(33*96+96=3264) + 4 GNN layers(4*(96*96+96+96*96+96+96*4+96*4)=4*(9216+96+9216+96+384+384)=4*19392=77568)
    # + readout(96*96+96=9312) + policy(96*96+96+96*12+12=10572) + value(96*96+96+96+1=10401)
    # = 3264+77568+9312+10572+10401 = 111,117
    
    HIDDEN_DIM = 96
    NUM_LAYERS = 4
    NUM_HEADS = 4
    NUM_ACTIONS = 12
    FEATURE_DIM = 33
    MAX_NODES = 32
    BATCH_SIZE = 32
    NUM_EPOCHS = 200
    LR = 0.001
    GAMMA = 0.99
    VALUE_LOSS_COEF = 0.5
    ENTROPY_COEF = 0.01
    
    device = torch.device('cpu')
    
    # Create model
    model = GNNEgraphModel(
        node_feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_actions=NUM_ACTIONS,
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"  Model parameters: {num_params:,}")
    print(f"  Architecture: hidden={HIDDEN_DIM}, layers={NUM_LAYERS}, heads={NUM_HEADS}")
    print()
    
    # Generate training data
    print("  Generating training data...")
    exprs, labels = generate_expressions()
    print(f"  Training expressions: {len(exprs)}")
    
    # Convert to graph data
    all_features = []
    all_adj = []
    all_masks = []
    all_labels = []
    all_rewards = []
    
    for expr, label in zip(exprs, labels):
        features, adj, mask, n = expr_to_graph(expr, MAX_NODES)
        
        # Compute reward for each action
        best_reward = 0.0
        best_action = -1
        for a_idx in range(NUM_ACTIONS):
            if is_applicable(a_idx, expr):
                result = apply_rewrite(a_idx, expr)
                if result is not None:
                    new_expr, cost_delta = result
                    orig_cost = estimate_cost(expr)
                    new_cost = estimate_cost(new_expr)
                    reward = (orig_cost - new_cost) / max(orig_cost, 1)
                    if reward > best_reward:
                        best_reward = reward
                        best_action = a_idx
        
        # Use the label as ground truth if available
        if label >= 0:
            target_action = label
        elif best_action >= 0:
            target_action = best_action
        else:
            target_action = 0  # Default
        
        # Compute reward for target action
        target_reward = best_reward
        
        all_features.append(features)
        all_adj.append(adj)
        all_masks.append(mask)
        all_labels.append(target_action)
        all_rewards.append(target_reward)
    
    features_t = torch.tensor(np.array(all_features), dtype=torch.float32).to(device)
    adj_t = torch.tensor(np.array(all_adj), dtype=torch.float32).to(device)
    masks_t = torch.tensor(np.array(all_masks), dtype=torch.float32).to(device)
    labels_t = torch.tensor(all_labels, dtype=torch.long).to(device)
    rewards_t = torch.tensor(all_rewards, dtype=torch.float32).to(device)
    
    # Data augmentation: create variations by swapping variable names
    aug_features = [features_t]
    aug_adj = [adj_t]
    aug_masks = [masks_t]
    aug_labels = [labels_t]
    aug_rewards = [rewards_t]
    
    for _ in range(3):  # 3x augmentation
        # Randomly perturb some constant values
        noise = torch.randn_like(features_t) * 0.05
        noisy_features = features_t.clone()
        # Only add noise to continuous features (29-32)
        noisy_features[:, :, 29:33] += noise[:, :, 29:33]
        aug_features.append(noisy_features)
        aug_adj.append(adj_t)
        aug_masks.append(masks_t)
        aug_labels.append(labels_t)
        aug_rewards.append(rewards_t)
    
    features_t = torch.cat(aug_features, dim=0)
    adj_t = torch.cat(aug_adj, dim=0)
    masks_t = torch.cat(aug_masks, dim=0)
    labels_t = torch.cat(aug_labels, dim=0)
    rewards_t = torch.cat(aug_rewards, dim=0)
    
    dataset_size = features_t.size(0)
    print(f"  Dataset size (with augmentation): {dataset_size}")
    print()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training
    print("  Starting training...")
    print()
    
    best_loss = float('inf')
    best_accuracy = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        
        # Shuffle
        perm = torch.randperm(dataset_size)
        features_shuffled = features_t[perm]
        adj_shuffled = adj_t[perm]
        masks_shuffled = masks_t[perm]
        labels_shuffled = labels_t[perm]
        rewards_shuffled = rewards_t[perm]
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_start in range(0, dataset_size, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, dataset_size)
            
            bf = features_shuffled[batch_start:batch_end]
            ba = adj_shuffled[batch_start:batch_end]
            bm = masks_shuffled[batch_start:batch_end]
            bl = labels_shuffled[batch_start:batch_end]
            br = rewards_shuffled[batch_start:batch_end]
            
            optimizer.zero_grad()
            
            # Forward pass
            log_policy, value_pred = model(bf, ba, bm)
            
            # Policy loss: cross-entropy with ground truth actions
            policy_loss = F.nll_loss(log_policy, bl)
            
            # Value loss: MSE between predicted and actual improvement
            value_target = br.unsqueeze(0) if br.dim() == 0 else br
            value_loss = F.mse_loss(value_pred, value_target)
            
            # Entropy bonus for exploration
            entropy = -(log_policy * log_policy.exp()).sum(dim=-1).mean()
            
            # Total loss
            loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * (batch_end - batch_start)
            
            # Accuracy
            predicted = log_policy.argmax(dim=-1)
            total_correct += (predicted == bl).sum().item()
            total_samples += (batch_end - batch_start)
        
        scheduler.step()
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100.0
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%, best_acc={best_accuracy:.1f}%")
    
    print()
    print(f"  Training complete: best_loss={best_loss:.4f}, best_accuracy={best_accuracy:.1f}%")
    print()
    
    # =====================================================================
    # 8. Evaluate the model
    # =====================================================================
    
    print("  Evaluating trained model...")
    model.eval()
    
    with torch.no_grad():
        log_policy, value_pred = model(features_t[:len(exprs)], adj_t[:len(exprs)], masks_t[:len(exprs)])
        predicted = log_policy.argmax(dim=-1)
    
    correct = 0
    improved = 0
    for i, (expr, label) in enumerate(zip(exprs, labels)):
        pred_action = predicted[i].item()
        pred_name = REWRITE_ACTIONS[pred_action] if pred_action < len(REWRITE_ACTIONS) else "Invalid"
        
        # Check if prediction actually improves the expression
        orig_cost = estimate_cost(expr)
        if is_applicable(pred_action, expr):
            result = apply_rewrite(pred_action, expr)
            if result is not None:
                new_expr, _ = result
                new_cost = estimate_cost(new_expr)
                if new_cost < orig_cost:
                    improved += 1
        
        if label >= 0 and pred_action == label:
            correct += 1
        
        print(f"    {str(expr)[:40]:40s}  target={REWRITE_ACTIONS[label] if label >= 0 else 'none':15s}  "
              f"pred={pred_name:15s}  value={value_pred[i].item():.3f}")
    
    print()
    print(f"  Label accuracy: {correct}/{len(exprs)} ({correct/len(exprs)*100:.1f}%)")
    print(f"  Cost improvement rate: {improved}/{len(exprs)} ({improved/len(exprs)*100:.1f}%)")
    
    # =====================================================================
    # 9. Export weights as Rust source file
    # =====================================================================
    
    print()
    print("  Exporting trained weights to Rust source file...")
    
    weights = export_weights_as_rust(model, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, NUM_ACTIONS)
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                'src', 'optimizer', 'gnn_trained_weights.rs')
    
    with open(output_path, 'w') as f:
        f.write(weights)
    
    print(f"  Weights exported to: {output_path}")
    print(f"  Total parameters: {num_params:,}")
    
    # Also save as JSON for verification
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gnn_weights.json')
    weights_dict = {}
    for name, param in model.named_parameters():
        weights_dict[name] = param.detach().numpy().tolist()
    with open(json_path, 'w') as f:
        json.dump(weights_dict, f)
    print(f"  Weights JSON saved to: {json_path}")
    
    return model

def export_weights_as_rust(model, hidden_dim, num_layers, num_heads, num_actions):
    """Export model weights as a Rust source file with baked-in weights"""
    
    def tensor_to_rust_array(tensor, name, indent=4):
        """Convert a PyTorch tensor to a Rust const array"""
        data = tensor.detach().numpy().flatten()
        lines = []
        indent_str = ' ' * indent
        
        if len(data) <= 16:
            # Single line
            values = ', '.join(f'{v:.10}' for v in data)
            lines.append(f'{indent_str}pub const {name}: [f64; {len(data)}] = [{values}];')
        else:
            # Multi-line, 4 values per line
            lines.append(f'{indent_str}pub const {name}: [f64; {len(data)}] = [')
            for i in range(0, len(data), 4):
                chunk = data[i:i+4]
                values = ', '.join(f'{v:.10}' for v in chunk)
                comma = ',' if i + 4 < len(data) else ','
                lines.append(f'{indent_str}    {values}{comma}')
            lines.append(f'{indent_str}];')
        
        return '\n'.join(lines)
    
    # Collect all weights
    state_dict = model.state_dict()
    
    rust_code = []
    rust_code.append('// =============================================================================')
    rust_code.append('// GNN E-Graph Superoptimizer — Pre-Trained Weights')
    rust_code.append('//')
    rust_code.append('// AUTO-GENERATED by tools/gnn_train/train_gnn.py')
    rust_code.append('// DO NOT EDIT MANUALLY')
    rust_code.append('//')
    rust_code.append(f'// Model: hidden_dim={hidden_dim}, num_layers={num_layers},')
    rust_code.append(f'//        num_heads={num_heads}, num_actions={num_actions}')
    rust_code.append(f'// Total parameters: {count_parameters(model):,}')
    rust_code.append('// =============================================================================')
    rust_code.append('')
    rust_code.append('use super::{Tensor, GnnConfig, GnnEgraphModel, GnnLayer, LayerNorm, PolicyHead,')
    rust_code.append('            ValueHead, GraphReadout, SimpleRng};')
    rust_code.append('')
    rust_code.append(f'/// Hidden dimension for the trained model')
    rust_code.append(f'pub const TRAINED_HIDDEN_DIM: usize = {hidden_dim};')
    rust_code.append(f'/// Number of GNN layers for the trained model')
    rust_code.append(f'pub const TRAINED_NUM_LAYERS: usize = {num_layers};')
    rust_code.append(f'/// Number of attention heads for the trained model')
    rust_code.append(f'pub const TRAINED_NUM_HEADS: usize = {num_heads};')
    rust_code.append(f'/// Number of actions for the trained model')
    rust_code.append(f'pub const TRAINED_NUM_ACTIONS: usize = {num_actions};')
    rust_code.append(f'/// Node feature dimension for the trained model')
    rust_code.append(f'pub const TRAINED_FEATURE_DIM: usize = 33;')
    rust_code.append('')
    
    # Embedding layer
    rust_code.append('// ── Embedding Layer ──')
    embed_weight = state_dict['embedding.weight'].numpy()  # (hidden_dim, 33)
    embed_bias = state_dict['embedding.bias'].numpy()  # (hidden_dim,)
    
    # Note: PyTorch Linear stores weight as (out, in), Rust stores as (in, out)
    # So we need to transpose
    embed_weight_t = embed_weight.T  # Now (33, hidden_dim) matching Rust's (node_feature_dim, hidden_dim)
    
    rust_code.append(tensor_to_rust_array(
        torch.tensor(embed_weight_t.flatten()), 'EMBEDDING_WEIGHT'))
    rust_code.append(tensor_to_rust_array(
        torch.tensor(embed_bias.flatten()), 'EMBEDDING_BIAS'))
    rust_code.append('')
    
    # GNN layers
    for layer_idx in range(num_layers):
        rust_code.append(f'// ── GNN Layer {layer_idx} ──')
        
        # w_self: Rust (hidden_dim, hidden_dim), PyTorch (hidden_dim, hidden_dim)
        w_self = state_dict[f'gnn_layers.{layer_idx}.w_self.weight'].numpy().T
        b_self = state_dict[f'gnn_layers.{layer_idx}.w_self.bias'].numpy()
        
        # w_neigh: same
        w_neigh = state_dict[f'gnn_layers.{layer_idx}.w_neigh.weight'].numpy().T
        b_neigh = state_dict[f'gnn_layers.{layer_idx}.w_neigh.bias'].numpy()
        
        # Layer norm
        ln_gamma = state_dict[f'gnn_layers.{layer_idx}.layer_norm.weight'].numpy()
        ln_beta = state_dict[f'gnn_layers.{layer_idx}.layer_norm.bias'].numpy()
        
        # Attention (if present)
        has_attn = f'gnn_layers.{layer_idx}.attn_key.weight' in state_dict
        
        prefix = f'GNN{layer_idx}'
        rust_code.append(tensor_to_rust_array(
            torch.tensor(w_self.flatten()), f'{prefix}_W_SELF'))
        rust_code.append(tensor_to_rust_array(
            torch.tensor(b_self.flatten()), f'{prefix}_B_SELF'))
        rust_code.append(tensor_to_rust_array(
            torch.tensor(w_neigh.flatten()), f'{prefix}_W_NEIGH'))
        rust_code.append(tensor_to_rust_array(
            torch.tensor(b_neigh.flatten()), f'{prefix}_B_NEIGH'))
        rust_code.append(tensor_to_rust_array(
            torch.tensor(ln_gamma.flatten()), f'{prefix}_LN_GAMMA'))
        rust_code.append(tensor_to_rust_array(
            torch.tensor(ln_beta.flatten()), f'{prefix}_LN_BETA'))
        
        if has_attn:
            attn_k = state_dict[f'gnn_layers.{layer_idx}.attn_key.weight'].numpy().T
            attn_q = state_dict[f'gnn_layers.{layer_idx}.attn_query.weight'].numpy().T
            rust_code.append(tensor_to_rust_array(
                torch.tensor(attn_k.flatten()), f'{prefix}_ATTN_KEY'))
            rust_code.append(tensor_to_rust_array(
                torch.tensor(attn_q.flatten()), f'{prefix}_ATTN_QUERY'))
        
        rust_code.append('')
    
    # Graph readout
    rust_code.append('// ── Graph Readout ──')
    readout_w = state_dict['readout.w.weight'].numpy().T
    readout_b = state_dict['readout.w.bias'].numpy()
    rust_code.append(tensor_to_rust_array(
        torch.tensor(readout_w.flatten()), 'READOUT_W'))
    rust_code.append(tensor_to_rust_array(
        torch.tensor(readout_b.flatten()), 'READOUT_B'))
    rust_code.append('')
    
    # Policy head
    rust_code.append('// ── Policy Head ──')
    policy_w1 = state_dict['policy_head.fc1.weight'].numpy().T
    policy_b1 = state_dict['policy_head.fc1.bias'].numpy()
    policy_w2 = state_dict['policy_head.fc2.weight'].numpy().T
    policy_b2 = state_dict['policy_head.fc2.bias'].numpy()
    rust_code.append(tensor_to_rust_array(
        torch.tensor(policy_w1.flatten()), 'POLICY_W1'))
    rust_code.append(tensor_to_rust_array(
        torch.tensor(policy_b1.flatten()), 'POLICY_B1'))
    rust_code.append(tensor_to_rust_array(
        torch.tensor(policy_w2.flatten()), 'POLICY_W2'))
    rust_code.append(tensor_to_rust_array(
        torch.tensor(policy_b2.flatten()), 'POLICY_B2'))
    rust_code.append('')
    
    # Value head
    rust_code.append('// ── Value Head ──')
    value_w1 = state_dict['value_head.fc1.weight'].numpy().T
    value_b1 = state_dict['value_head.fc1.bias'].numpy()
    value_w2 = state_dict['value_head.fc2.weight'].numpy().T
    value_b2 = state_dict['value_head.fc2.bias'].numpy()
    rust_code.append(tensor_to_rust_array(
        torch.tensor(value_w1.flatten()), 'VALUE_W1'))
    rust_code.append(tensor_to_rust_array(
        torch.tensor(value_b1.flatten()), 'VALUE_B1'))
    rust_code.append(tensor_to_rust_array(
        torch.tensor(value_w2.flatten()), 'VALUE_W2'))
    rust_code.append(tensor_to_rust_array(
        torch.tensor(value_b2.flatten()), 'VALUE_B2'))
    rust_code.append('')
    
    # Builder function
    rust_code.append('''/// Load the pre-trained GNN model with baked-in weights
pub fn load_pretrained_gnn() -> GnnEgraphModel {
    let config = GnnConfig {
        hidden_dim: TRAINED_HIDDEN_DIM,
        num_layers: TRAINED_NUM_LAYERS,
        num_heads: TRAINED_NUM_HEADS,
        num_actions: TRAINED_NUM_ACTIONS,
        node_feature_dim: TRAINED_FEATURE_DIM,
        learning_rate: 0.001,
        gamma: 0.99,
        num_episodes: 0,  // Already trained
        batch_size: 32,
        use_attention: true,
        value_loss_coef: 0.5,
        entropy_coef: 0.01,
    };
    
    // Build the model with pre-trained weights
    let embedding = Tensor::from_vec(
        EMBEDDING_WEIGHT.to_vec(), TRAINED_FEATURE_DIM, TRAINED_HIDDEN_DIM);
    let embedding_bias = Tensor::from_vec(
        EMBEDDING_BIAS.to_vec(), 1, TRAINED_HIDDEN_DIM);
    
    let mut gnn_layers = Vec::new();''')
    
    for layer_idx in range(num_layers):
        prefix = f'GNN{layer_idx}'
        has_attn_str = 'true' if f'gnn_layers.{layer_idx}.attn_key.weight' in state_dict else 'false'
        
        rust_code.append(f'''
    {{
        let w_self = Tensor::from_vec(
            {prefix}_W_SELF.to_vec(), TRAINED_HIDDEN_DIM, TRAINED_HIDDEN_DIM);
        let b_self = Tensor::from_vec(
            {prefix}_B_SELF.to_vec(), 1, TRAINED_HIDDEN_DIM);
        let w_neigh = Tensor::from_vec(
            {prefix}_W_NEIGH.to_vec(), TRAINED_HIDDEN_DIM, TRAINED_HIDDEN_DIM);
        let b_neigh = Tensor::from_vec(
            {prefix}_B_NEIGH.to_vec(), 1, TRAINED_HIDDEN_DIM);
        let ln_gamma = Tensor::from_vec(
            {prefix}_LN_GAMMA.to_vec(), 1, TRAINED_HIDDEN_DIM);
        let ln_beta = Tensor::from_vec(
            {prefix}_LN_BETA.to_vec(), 1, TRAINED_HIDDEN_DIM);
        let attn_key = if {has_attn_str} {{
            Some(Tensor::from_vec(
                {prefix}_ATTN_KEY.to_vec(), TRAINED_HIDDEN_DIM, TRAINED_NUM_HEADS))
        }} else {{ None }};
        let attn_query = if {has_attn_str} {{
            Some(Tensor::from_vec(
                {prefix}_ATTN_QUERY.to_vec(), TRAINED_HIDDEN_DIM, TRAINED_NUM_HEADS))
        }} else {{ None }};
        
        gnn_layers.push(GnnLayer {{
            w_self,
            w_neigh,
            bias: b_self.add(&b_neigh),  // Combine biases
            layer_norm: LayerNorm {{
                gamma: ln_gamma,
                beta: ln_beta,
                eps: 1e-5,
            }},
            attn_key,
            attn_query,
        }});
    }}''')
    
    rust_code.append('''
    let readout = GraphReadout {
        w: Tensor::from_vec(READOUT_W.to_vec(), TRAINED_HIDDEN_DIM, TRAINED_HIDDEN_DIM),
        b: Tensor::from_vec(READOUT_B.to_vec(), 1, TRAINED_HIDDEN_DIM),
    };
    
    let policy_head = PolicyHead {
        w1: Tensor::from_vec(POLICY_W1.to_vec(), TRAINED_HIDDEN_DIM, TRAINED_HIDDEN_DIM),
        b1: Tensor::from_vec(POLICY_B1.to_vec(), 1, TRAINED_HIDDEN_DIM),
        w2: Tensor::from_vec(POLICY_W2.to_vec(), TRAINED_HIDDEN_DIM, TRAINED_NUM_ACTIONS),
        b2: Tensor::from_vec(POLICY_B2.to_vec(), 1, TRAINED_NUM_ACTIONS),
    };
    
    let value_head = ValueHead {
        w1: Tensor::from_vec(VALUE_W1.to_vec(), TRAINED_HIDDEN_DIM, TRAINED_HIDDEN_DIM),
        b1: Tensor::from_vec(VALUE_B1.to_vec(), 1, TRAINED_HIDDEN_DIM),
        w2: Tensor::from_vec(VALUE_W2.to_vec(), TRAINED_HIDDEN_DIM, 1),
        b2: Tensor::from_vec(VALUE_B2.to_vec(), 1, 1),
    };
    
    GnnEgraphModel {
        embedding,
        embedding_bias,
        gnn_layers,
        readout,
        policy_head,
        value_head,
        config,
    }
}''')
    
    return '\n'.join(rust_code)

# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    model = train_model()
    print()
    print("=" * 70)
    print("  GNN Training Pipeline Complete!")
    print("=" * 70)
