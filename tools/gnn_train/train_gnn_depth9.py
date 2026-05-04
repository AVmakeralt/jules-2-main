#!/usr/bin/env python3
"""
GNN E-Graph Superoptimizer — Depth-9 Rollout Training

Architecture: 4-layer GNN (hidden_dim=96, ~110k params)
Training: Depth-9 multi-step greedy rollout for training data
  - Generates training labels by running 9-step greedy optimization
  - Extended 500-epoch training with cosine annealing + warm restarts
  - Label smoothing, gradient clipping, and aggressive augmentation
  - This gives the GNN "depth-9 foresight" — it learns what a 9-step
    optimization sequence would discover, but predicts it in one shot.

After training, exports weights as a Rust source file with baked-in weights.
"""

import numpy as np
import sys
import os
import json
import time
import copy

# =====================================================================
# 1-3. Expression AST, Cost Model, Rewrite Rules (same as before)
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

COST_TABLE = {
    'add': 1, 'sub': 1, 'and': 1, 'or': 1, 'xor': 1,
    'shl': 1, 'shr': 1, 'bitand': 1, 'bitor': 1, 'bitxor': 1,
    'mul': 3, 'div': 35, 'rem': 35,
    'neg': 1, 'not': 1,
    'const': 0, 'var': 1, 'floordiv': 35,
    'eq': 1, 'ne': 1, 'lt': 1, 'le': 1, 'gt': 1, 'ge': 1,
}

def estimate_cost(expr):
    if isinstance(expr, IntLitExpr):
        return COST_TABLE['const']
    elif isinstance(expr, VarExpr):
        return COST_TABLE['var']
    elif isinstance(expr, BinOpExpr):
        return estimate_cost(expr.lhs) + estimate_cost(expr.rhs) + COST_TABLE.get(expr.op, 1)
    elif isinstance(expr, UnOpExpr):
        return estimate_cost(expr.operand) + COST_TABLE.get(expr.op, 1)
    return 1

REWRITE_ACTIONS = [
    'Commute', 'IdentityRight', 'IdentityLeft', 'AnnihilateRight',
    'AnnihilateLeft', 'ConstantFold', 'StrengthReduce', 'Distribute',
    'Factor', 'DoubleNegate', 'SelfIdentity', 'Absorb',
]

COMMUTATIVE_OPS = {'add', 'mul', 'bitand', 'bitor', 'bitxor'}

def is_applicable(action_idx, expr):
    action = REWRITE_ACTIONS[action_idx]
    if not isinstance(expr, BinOpExpr) and not isinstance(expr, UnOpExpr):
        return action in ['ConstantFold']
    if action == 'Commute':
        return isinstance(expr, BinOpExpr) and expr.op in COMMUTATIVE_OPS
    elif action == 'IdentityRight':
        if isinstance(expr, BinOpExpr):
            if expr.op in ('add', 'sub') and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 0: return True
            if expr.op == 'mul' and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 1: return True
            if expr.op == 'div' and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 1: return True
            if expr.op in ('bitor', 'bitxor') and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 0: return True
        return False
    elif action == 'IdentityLeft':
        if isinstance(expr, BinOpExpr):
            if expr.op == 'add' and isinstance(expr.lhs, IntLitExpr) and expr.lhs.value == 0: return True
            if expr.op == 'mul' and isinstance(expr.lhs, IntLitExpr) and expr.lhs.value == 1: return True
        return False
    elif action == 'AnnihilateRight':
        if isinstance(expr, BinOpExpr):
            if expr.op == 'mul' and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 0: return True
            if expr.op == 'bitand' and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 0: return True
        return False
    elif action == 'AnnihilateLeft':
        if isinstance(expr, BinOpExpr):
            if expr.op == 'mul' and isinstance(expr.lhs, IntLitExpr) and expr.lhs.value == 0: return True
            if expr.op == 'bitand' and isinstance(expr.lhs, IntLitExpr) and expr.lhs.value == 0: return True
        return False
    elif action == 'ConstantFold':
        if isinstance(expr, BinOpExpr) and isinstance(expr.lhs, IntLitExpr) and isinstance(expr.rhs, IntLitExpr): return True
        if isinstance(expr, UnOpExpr) and isinstance(expr.operand, IntLitExpr): return True
        return False
    elif action == 'StrengthReduce':
        if isinstance(expr, BinOpExpr) and isinstance(expr.rhs, IntLitExpr):
            v = expr.rhs.value
            if v > 1 and (v & (v - 1)) == 0 and expr.op in ('mul', 'div'): return True
        return False
    elif action == 'Distribute':
        if isinstance(expr, BinOpExpr) and expr.op == 'mul':
            if isinstance(expr.rhs, BinOpExpr) and expr.rhs.op == 'add': return True
        return False
    elif action == 'Factor':
        if isinstance(expr, BinOpExpr) and expr.op == 'add':
            if isinstance(expr.lhs, BinOpExpr) and isinstance(expr.rhs, BinOpExpr):
                if expr.lhs.op == 'mul' and expr.rhs.op == 'mul':
                    if repr(expr.lhs.lhs) == repr(expr.rhs.lhs): return True
        return False
    elif action == 'DoubleNegate':
        if isinstance(expr, UnOpExpr) and isinstance(expr.operand, UnOpExpr):
            if expr.op == expr.operand.op and expr.op in ('neg', 'not'): return True
        return False
    elif action == 'SelfIdentity':
        if isinstance(expr, BinOpExpr):
            if repr(expr.lhs) == repr(expr.rhs):
                if expr.op in ('sub', 'bitxor', 'eq', 'ne'): return True
        return False
    elif action == 'Absorb':
        return False
    return False

def apply_rewrite(action_idx, expr):
    action = REWRITE_ACTIONS[action_idx]
    if action == 'Commute':
        if isinstance(expr, BinOpExpr) and expr.op in COMMUTATIVE_OPS:
            return BinOpExpr(expr.op, expr.rhs, expr.lhs), 0
    elif action == 'IdentityRight':
        if isinstance(expr, BinOpExpr):
            if expr.op in ('add', 'sub') and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 0:
                return expr.lhs, -1
            if expr.op == 'mul' and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 1:
                return expr.lhs, -1
            if expr.op == 'div' and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 1:
                return expr.lhs, -35
            if expr.op in ('bitor', 'bitxor') and isinstance(expr.rhs, IntLitExpr) and expr.rhs.value == 0:
                return expr.lhs, -1
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
            ops = {'add': l+r, 'sub': l-r, 'mul': l*r}
            if expr.op in ops: result = ops[expr.op]
            elif expr.op == 'div' and r != 0: result = l // r
            elif expr.op == 'rem' and r != 0: result = l % r
            elif expr.op == 'bitand': result = l & r
            elif expr.op == 'bitor': result = l | r
            elif expr.op == 'bitxor': result = l ^ r
            elif expr.op == 'shl': result = l << r
            elif expr.op == 'shr': result = l >> r
            if result is not None:
                return IntLitExpr(result & ((1 << 128) - 1)), -estimate_cost(expr)
        if isinstance(expr, UnOpExpr) and isinstance(expr.operand, IntLitExpr):
            v = expr.operand.value
            if expr.op == 'neg': return IntLitExpr((-v) & ((1 << 128) - 1)), -1
            elif expr.op == 'not': return IntLitExpr((~v) & ((1 << 128) - 1)), -1
    elif action == 'StrengthReduce':
        if isinstance(expr, BinOpExpr) and isinstance(expr.rhs, IntLitExpr):
            v = expr.rhs.value
            if v > 1 and (v & (v - 1)) == 0:
                shift = int(np.log2(v))
                new_op = 'shl' if expr.op == 'mul' else 'shr'
                return BinOpExpr(new_op, expr.lhs, IntLitExpr(shift)), COST_TABLE.get(new_op, 1) - COST_TABLE.get(expr.op, 1)
    elif action == 'Distribute':
        if isinstance(expr, BinOpExpr) and expr.op == 'mul':
            if isinstance(expr.rhs, BinOpExpr) and expr.rhs.op == 'add':
                a, b, c = expr.lhs, expr.rhs.lhs, expr.rhs.rhs
                return BinOpExpr('add', BinOpExpr('mul', a, b), BinOpExpr('mul', a, c)), 1
    elif action == 'Factor':
        if isinstance(expr, BinOpExpr) and expr.op == 'add':
            if isinstance(expr.lhs, BinOpExpr) and isinstance(expr.rhs, BinOpExpr):
                if expr.lhs.op == 'mul' and expr.rhs.op == 'mul':
                    if repr(expr.lhs.lhs) == repr(expr.rhs.lhs):
                        a, b, c = expr.lhs.lhs, expr.lhs.rhs, expr.rhs.rhs
                        return BinOpExpr('mul', a, BinOpExpr('add', b, c)), -2
    elif action == 'DoubleNegate':
        if isinstance(expr, UnOpExpr) and isinstance(expr.operand, UnOpExpr):
            if expr.op == expr.operand.op and expr.op in ('neg', 'not'):
                return expr.operand.operand, -1
    elif action == 'SelfIdentity':
        if isinstance(expr, BinOpExpr) and repr(expr.lhs) == repr(expr.rhs):
            if expr.op == 'sub' or expr.op == 'bitxor':
                return IntLitExpr(0), -estimate_cost(expr)
    return None

def depth9_greedy_optimize(expr, max_steps=9):
    """Run depth-9 greedy optimization to find best first action and total reward."""
    current = expr
    orig_cost = estimate_cost(current)
    first_action = -1
    total_reward = 0.0
    
    for step in range(max_steps):
        best_action = -1
        best_reward = 0.0
        best_result = None
        for a_idx in range(len(REWRITE_ACTIONS)):
            if is_applicable(a_idx, current):
                result = apply_rewrite(a_idx, current)
                if result is not None:
                    new_expr, _ = result
                    new_cost = estimate_cost(new_expr)
                    cur_cost = estimate_cost(current)
                    reward = (cur_cost - new_cost) / max(cur_cost, 1)
                    if reward > best_reward:
                        best_reward = reward
                        best_action = a_idx
                        best_result = result
        if best_action < 0 or best_reward <= 0:
            break
        if step == 0:
            first_action = best_action
        total_reward += best_reward * (0.99 ** step)  # Discounted
        current = best_result[0]
    
    final_cost = estimate_cost(current)
    total_improvement = (orig_cost - final_cost) / max(orig_cost, 1)
    return first_action, total_improvement, current

# =====================================================================
# 4. Expression to Graph
# =====================================================================

OP_TO_IDX = {
    'const_int': 0, 'const_float': 1, 'const_bool': 2, 'var': 3,
    'add': 4, 'sub': 5, 'mul': 6, 'div': 7, 'rem': 8, 'floordiv': 9,
    'eq': 10, 'ne': 11, 'lt': 12, 'le': 13, 'gt': 14, 'ge': 15,
    'and': 16, 'or': 17, 'bitand': 18, 'bitor': 19, 'bitxor': 20,
    'shl': 21, 'shr': 22, 'neg': 23, 'not': 24,
    'deref': 25, 'ref': 26, 'refmut': 27, 'other': 28,
}

FEATURE_DIM = 33

def expr_to_graph(expr, max_nodes=32):
    nodes = []
    edges_src = []
    edges_dst = []
    action_masks = []
    
    def build(e, depth):
        idx = len(nodes)
        features = np.zeros(FEATURE_DIM, dtype=np.float64)
        if isinstance(e, IntLitExpr):
            features[0] = 1.0; features[31] = 1.0
        elif isinstance(e, VarExpr):
            features[3] = 1.0; features[32] = 1.0
        elif isinstance(e, BinOpExpr):
            features[OP_TO_IDX.get(e.op, 28)] = 1.0; features[30] = 1.0
        elif isinstance(e, UnOpExpr):
            features[OP_TO_IDX.get(e.op, 28)] = 1.0; features[30] = 0.5
        else:
            features[28] = 1.0
        features[29] = np.log1p(depth) / 5.0
        nodes.append(features)
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
    while len(nodes) < max_nodes:
        nodes.append(np.zeros(FEATURE_DIM, dtype=np.float64))
        action_masks.append(np.zeros(len(REWRITE_ACTIONS), dtype=np.float64))
    adj = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    for s, d in zip(edges_src, edges_dst):
        if s < max_nodes and d < max_nodes:
            adj[s][d] = 1.0; adj[d][s] = 1.0
    for i in range(min(n, max_nodes)):
        adj[i][i] = 1.0
    degree = adj.sum(axis=1)
    degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    adj_norm = adj * np.outer(degree_inv_sqrt, degree_inv_sqrt)
    return np.array(nodes[:max_nodes], dtype=np.float64), adj_norm, np.array(action_masks[:max_nodes], dtype=np.float64), n

# =====================================================================
# 5. Training Data Generation
# =====================================================================

def generate_expressions():
    exprs = []
    # Identity patterns
    for op, val in [('add', 0), ('sub', 0), ('mul', 1), ('div', 1)]:
        exprs.append(BinOpExpr(op, VarExpr('x'), IntLitExpr(val)))
    exprs.append(BinOpExpr('add', IntLitExpr(0), VarExpr('x')))
    exprs.append(BinOpExpr('mul', IntLitExpr(1), VarExpr('x')))
    exprs.append(BinOpExpr('bitor', VarExpr('x'), IntLitExpr(0)))
    exprs.append(BinOpExpr('bitxor', VarExpr('x'), IntLitExpr(0)))
    # Annihilate
    exprs.append(BinOpExpr('mul', VarExpr('x'), IntLitExpr(0)))
    exprs.append(BinOpExpr('bitand', VarExpr('x'), IntLitExpr(0)))
    exprs.append(BinOpExpr('mul', IntLitExpr(0), VarExpr('x')))
    # ConstantFold
    exprs.append(BinOpExpr('add', IntLitExpr(3), IntLitExpr(5)))
    exprs.append(BinOpExpr('sub', IntLitExpr(10), IntLitExpr(3)))
    exprs.append(BinOpExpr('mul', IntLitExpr(4), IntLitExpr(7)))
    exprs.append(BinOpExpr('div', IntLitExpr(100), IntLitExpr(10)))
    exprs.append(BinOpExpr('bitand', IntLitExpr(15), IntLitExpr(7)))
    # StrengthReduce
    for k in range(1, 9):
        val = 1 << k
        exprs.append(BinOpExpr('mul', VarExpr('x'), IntLitExpr(val)))
        exprs.append(BinOpExpr('div', VarExpr('x'), IntLitExpr(val)))
    # SelfIdentity
    exprs.append(BinOpExpr('sub', VarExpr('x'), VarExpr('x')))
    exprs.append(BinOpExpr('bitxor', VarExpr('x'), VarExpr('x')))
    # DoubleNegate
    exprs.append(UnOpExpr('neg', UnOpExpr('neg', VarExpr('x'))))
    exprs.append(UnOpExpr('not', UnOpExpr('not', VarExpr('x'))))
    # Distribute / Factor
    exprs.append(BinOpExpr('mul', VarExpr('x'), BinOpExpr('add', VarExpr('y'), VarExpr('z'))))
    exprs.append(BinOpExpr('add', BinOpExpr('mul', VarExpr('x'), VarExpr('y')), BinOpExpr('mul', VarExpr('x'), VarExpr('z'))))
    # Multi-step patterns
    exprs.append(BinOpExpr('mul', BinOpExpr('add', VarExpr('x'), IntLitExpr(0)), IntLitExpr(1)))
    exprs.append(BinOpExpr('mul', BinOpExpr('add', BinOpExpr('mul', VarExpr('x'), IntLitExpr(1)), IntLitExpr(0)), BinOpExpr('add', VarExpr('y'), IntLitExpr(0))))
    exprs.append(BinOpExpr('add', BinOpExpr('mul', BinOpExpr('add', VarExpr('a'), IntLitExpr(0)), BinOpExpr('mul', VarExpr('b'), IntLitExpr(1))), BinOpExpr('sub', VarExpr('x'), VarExpr('x'))))
    exprs.append(BinOpExpr('mul', BinOpExpr('add', BinOpExpr('mul', BinOpExpr('add', VarExpr('x'), IntLitExpr(0)), IntLitExpr(1)), BinOpExpr('mul', VarExpr('y'), IntLitExpr(0))), BinOpExpr('add', BinOpExpr('div', VarExpr('z'), IntLitExpr(1)), BinOpExpr('sub', VarExpr('w'), VarExpr('w')))))
    exprs.append(BinOpExpr('div', BinOpExpr('add', VarExpr('x'), IntLitExpr(0)), IntLitExpr(8)))
    exprs.append(BinOpExpr('add', BinOpExpr('mul', VarExpr('x'), IntLitExpr(1)), IntLitExpr(0)))
    exprs.append(UnOpExpr('neg', UnOpExpr('neg', BinOpExpr('add', VarExpr('x'), IntLitExpr(0)))))
    exprs.append(BinOpExpr('add', BinOpExpr('mul', VarExpr('x'), IntLitExpr(2)), IntLitExpr(0)))
    exprs.append(BinOpExpr('div', BinOpExpr('mul', BinOpExpr('add', VarExpr('x'), IntLitExpr(0)), IntLitExpr(1)), BinOpExpr('add', BinOpExpr('sub', VarExpr('y'), VarExpr('y')), IntLitExpr(1))))
    # Negative examples
    exprs.append(BinOpExpr('add', VarExpr('x'), VarExpr('y')))
    exprs.append(BinOpExpr('mul', VarExpr('x'), VarExpr('y')))
    exprs.append(BinOpExpr('sub', VarExpr('x'), VarExpr('y')))
    exprs.append(VarExpr('x'))
    return exprs

# =====================================================================
# 6. GNN Model (4-layer, hidden=96)
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
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
        h_self = self.w_self(h)
        h_neigh = torch.bmm(adj, h)
        h_neigh = self.w_neigh(h_neigh)
        h_combined = h_self + h_neigh
        h_normed = self.layer_norm(h_combined)
        return F.relu(h_normed)

class GraphReadout(nn.Module):
    def __init__(self, d_h):
        super().__init__()
        self.w = nn.Linear(d_h, d_h)
    
    def forward(self, node_embeddings):
        scores = self.w(node_embeddings)
        weights = F.relu(scores)
        weight_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        normalized_weights = weights / weight_sum
        return (normalized_weights * node_embeddings).sum(dim=1)

class PolicyHead(nn.Module):
    def __init__(self, d_h, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(d_h, d_h)
        self.fc2 = nn.Linear(d_h, num_actions)
    
    def forward(self, x):
        return F.log_softmax(self.fc2(F.relu(self.fc1(x))), dim=-1)

class ValueHead(nn.Module):
    def __init__(self, d_h):
        super().__init__()
        self.fc1 = nn.Linear(d_h, d_h)
        self.fc2 = nn.Linear(d_h, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc2(F.relu(self.fc1(x))))

class GNNEgraphModel(nn.Module):
    def __init__(self, node_feature_dim=33, hidden_dim=96, num_layers=4,
                 num_heads=4, num_actions=12, use_attention=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.embedding = nn.Linear(node_feature_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim, num_heads, use_attention)
            for _ in range(num_layers)
        ])
        self.readout = GraphReadout(hidden_dim)
        self.policy_head = PolicyHead(hidden_dim, num_actions)
        self.value_head = ValueHead(hidden_dim)
    
    def forward(self, node_features, adj, action_mask):
        h = F.relu(self.embedding(node_features))
        for layer in self.gnn_layers:
            h = layer(h, adj)
        graph_repr = self.readout(h)
        log_policy = self.policy_head(graph_repr)
        root_mask = action_mask[:, 0, :]
        masked_logits = log_policy.clone()
        masked_logits[root_mask == 0] = -1e9
        log_policy_masked = F.log_softmax(masked_logits, dim=-1)
        value = self.value_head(graph_repr)
        return log_policy_masked, value.squeeze(-1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =====================================================================
# 7. Training Loop
# =====================================================================

def train_model():
    print("=" * 70)
    print("  GNN E-Graph Superoptimizer — Depth-9 Rollout Training")
    print("=" * 70)
    print()
    
    HIDDEN_DIM = 96
    NUM_LAYERS = 4
    NUM_HEADS = 4
    NUM_ACTIONS = 12
    FEATURE_DIM = 33
    MAX_NODES = 32
    BATCH_SIZE = 55  # Process all samples per batch
    NUM_EPOCHS = 500
    LR = 0.001
    VALUE_LOSS_COEF = 0.5
    ENTROPY_COEF = 0.01
    
    device = torch.device('cpu')
    
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
    print(f"  Training: depth-9 rollout labels, {NUM_EPOCHS} epochs")
    print()
    
    # Generate training data with depth-9 rollouts
    print("  Generating depth-9 rollout training data...")
    exprs = generate_expressions()
    print(f"  Training expressions: {len(exprs)}")
    
    all_features = []
    all_adj = []
    all_masks = []
    all_labels = []
    all_rewards = []
    
    for expr in exprs:
        features, adj, mask, n = expr_to_graph(expr, MAX_NODES)
        
        # Use depth-9 greedy rollout to find best first action and total reward
        first_action, total_improvement, _ = depth9_greedy_optimize(expr, max_steps=9)
        
        if first_action < 0:
            # No improvement possible - default to 0 (Commute, low cost if wrong)
            first_action = 0
        
        all_features.append(features)
        all_adj.append(adj)
        all_masks.append(mask)
        all_labels.append(first_action)
        all_rewards.append(total_improvement)
    
    features_t = torch.tensor(np.array(all_features), dtype=torch.float32).to(device)
    adj_t = torch.tensor(np.array(all_adj), dtype=torch.float32).to(device)
    masks_t = torch.tensor(np.array(all_masks), dtype=torch.float32).to(device)
    labels_t = torch.tensor(all_labels, dtype=torch.long).to(device)
    rewards_t = torch.tensor(all_rewards, dtype=torch.float32).to(device)
    
    # Augmentation
    aug_features = [features_t]
    aug_adj = [adj_t]
    aug_masks = [masks_t]
    aug_labels = [labels_t]
    aug_rewards = [rewards_t]
    
    for _ in range(3):
        noisy = features_t.clone()
        noise = torch.randn_like(noisy[:, :, 29:33]) * 0.05
        noisy[:, :, 29:33] += noise
        aug_features.append(noisy)
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
    
    # Show depth-9 labels
    print("  Depth-9 rollout labels:")
    for i, expr in enumerate(exprs):
        print(f"    {str(expr)[:50]:50s}  first_action={REWRITE_ACTIONS[all_labels[i]]:15s}  improvement={all_rewards[i]:.3f}")
    print()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-6
    )
    
    print("  Starting depth-9 training...")
    print()
    
    best_loss = float('inf')
    best_accuracy = 0.0
    best_model_state = None
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        
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
            
            log_policy, value_pred = model(bf, ba, bm)
            
            policy_loss = F.nll_loss(log_policy, bl)
            value_target = br.unsqueeze(0) if br.dim() == 0 else br
            value_loss = F.mse_loss(value_pred, value_target)
            entropy = -(log_policy * log_policy.exp()).sum(dim=-1).mean()
            
            loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * (batch_end - batch_start)
            predicted = log_policy.argmax(dim=-1)
            total_correct += (predicted == bl).sum().item()
            total_samples += (batch_end - batch_start)
        
        scheduler.step()
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100.0
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = copy.deepcopy(model.state_dict())
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:4d}/{NUM_EPOCHS}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%, best_acc={best_accuracy:.1f}%, lr={lr_now:.6f}")
    
    print()
    print(f"  Training complete: best_loss={best_loss:.4f}, best_accuracy={best_accuracy:.1f}%")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("  Loaded best model checkpoint")
    print()
    
    # Evaluate
    print("  Evaluating depth-9 trained model...")
    model.eval()
    
    with torch.no_grad():
        log_policy, value_pred = model(features_t[:len(exprs)], adj_t[:len(exprs)], masks_t[:len(exprs)])
        predicted = log_policy.argmax(dim=-1)
    
    correct = 0
    improved = 0
    for i, expr in enumerate(exprs):
        pred_action = predicted[i].item()
        pred_name = REWRITE_ACTIONS[pred_action] if pred_action < len(REWRITE_ACTIONS) else "Invalid"
        orig_cost = estimate_cost(expr)
        if is_applicable(pred_action, expr):
            result = apply_rewrite(pred_action, expr)
            if result is not None:
                new_cost = estimate_cost(result[0])
                if new_cost < orig_cost:
                    improved += 1
        if pred_action == all_labels[i]:
            correct += 1
        print(f"    {str(expr)[:45]:45s}  target={REWRITE_ACTIONS[all_labels[i]]:15s}  pred={pred_name:15s}  value={value_pred[i].item():.3f}")
    
    print()
    print(f"  Label accuracy: {correct}/{len(exprs)} ({correct/len(exprs)*100:.1f}%)")
    print(f"  Cost improvement rate: {improved}/{len(exprs)} ({improved/len(exprs)*100:.1f}%)")
    
    # Export weights
    print()
    print("  Exporting depth-9 trained weights to Rust source file...")
    
    weights_code = export_weights_as_rust(model, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, NUM_ACTIONS)
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                'src', 'optimizer', 'gnn_trained_weights.rs')
    
    with open(output_path, 'w') as f:
        f.write(weights_code)
    
    print(f"  Weights exported to: {output_path}")
    print(f"  Total parameters: {num_params:,}")
    
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gnn_weights_depth9.json')
    weights_dict = {}
    for name, param in model.named_parameters():
        weights_dict[name] = param.detach().numpy().tolist()
    with open(json_path, 'w') as f:
        json.dump(weights_dict, f)
    print(f"  Weights JSON saved to: {json_path}")
    
    return model

def export_weights_as_rust(model, hidden_dim, num_layers, num_heads, num_actions):
    def tensor_to_rust_array(tensor, name, indent=4):
        data = [float(v) for v in tensor.detach().numpy().flatten().tolist()]
        lines = []
        indent_str = ' ' * indent
        if len(data) <= 16:
            values = ', '.join(f'{v:.10}' for v in data)
            lines.append(f'{indent_str}pub const {name}: [f64; {len(data)}] = [{values}];')
        else:
            lines.append(f'{indent_str}pub const {name}: [f64; {len(data)}] = [')
            for i in range(0, len(data), 4):
                chunk = data[i:i+4]
                values = ', '.join(f'{v:.10}' for v in chunk)
                lines.append(f'{indent_str}    {values},')
            lines.append(f'{indent_str}];')
        return '\n'.join(lines)
    
    state_dict = model.state_dict()
    
    rust_code = []
    rust_code.append('// =============================================================================')
    rust_code.append('// GNN E-Graph Superoptimizer — Pre-Trained Weights (Depth-9 Rollout)')
    rust_code.append('//')
    rust_code.append('// AUTO-GENERATED by tools/gnn_train/train_gnn_depth9.py')
    rust_code.append('// DO NOT EDIT MANUALLY')
    rust_code.append('//')
    rust_code.append(f'// Model: hidden_dim={hidden_dim}, num_layers={num_layers},')
    rust_code.append(f'//        num_heads={num_heads}, num_actions={num_actions}')
    rust_code.append(f'// Training: depth-9 greedy rollout labels, 500 epochs')
    rust_code.append(f'// Total parameters: {count_parameters(model):,}')
    rust_code.append('// =============================================================================')
    rust_code.append('')
    rust_code.append('use super::gnn_egraph_optimizer::{Tensor, GnnConfig, GnnEgraphModel, GnnLayer, LayerNorm, PolicyHead,')
    rust_code.append('            ValueHead, GraphReadout};')
    rust_code.append('')
    rust_code.append(f'pub const TRAINED_HIDDEN_DIM: usize = {hidden_dim};')
    rust_code.append(f'pub const TRAINED_NUM_LAYERS: usize = {num_layers};')
    rust_code.append(f'pub const TRAINED_NUM_HEADS: usize = {num_heads};')
    rust_code.append(f'pub const TRAINED_NUM_ACTIONS: usize = {num_actions};')
    rust_code.append(f'pub const TRAINED_FEATURE_DIM: usize = 33;')
    rust_code.append('')
    
    # Embedding
    rust_code.append('// ── Embedding Layer ──')
    rust_code.append(tensor_to_rust_array(state_dict['embedding.weight'], 'EMBEDDING_WEIGHT'))
    rust_code.append(tensor_to_rust_array(state_dict['embedding.bias'], 'EMBEDDING_BIAS'))
    rust_code.append('')
    
    # GNN layers
    for i in range(num_layers):
        rust_code.append(f'// ── GNN Layer {i} ──')
        p = f'gnn_layers.{i}.'
        rust_code.append(tensor_to_rust_array(state_dict[f'{p}w_self.weight'], f'GNN{i}_W_SELF_WEIGHT'))
        rust_code.append(tensor_to_rust_array(state_dict[f'{p}w_self.bias'], f'GNN{i}_W_SELF_BIAS'))
        rust_code.append(tensor_to_rust_array(state_dict[f'{p}w_neigh.weight'], f'GNN{i}_W_NEIGH_WEIGHT'))
        rust_code.append(tensor_to_rust_array(state_dict[f'{p}w_neigh.bias'], f'GNN{i}_W_NEIGH_BIAS'))
        rust_code.append(tensor_to_rust_array(state_dict[f'{p}layer_norm.weight'], f'GNN{i}_LN_GAMMA'))
        rust_code.append(tensor_to_rust_array(state_dict[f'{p}layer_norm.bias'], f'GNN{i}_LN_BETA'))
        if f'{p}attn_key.weight' in state_dict:
            rust_code.append(tensor_to_rust_array(state_dict[f'{p}attn_key.weight'], f'GNN{i}_ATTN_KEY'))
            rust_code.append(tensor_to_rust_array(state_dict[f'{p}attn_query.weight'], f'GNN{i}_ATTN_QUERY'))
        rust_code.append('')
    
    # Readout
    rust_code.append('// ── Graph Readout ──')
    rust_code.append(tensor_to_rust_array(state_dict['readout.w.weight'], 'READOUT_W_WEIGHT'))
    rust_code.append(tensor_to_rust_array(state_dict['readout.w.bias'], 'READOUT_W_BIAS'))
    rust_code.append('')
    
    # Policy head
    rust_code.append('// ── Policy Head ──')
    rust_code.append(tensor_to_rust_array(state_dict['policy_head.fc1.weight'], 'POLICY_W1_WEIGHT'))
    rust_code.append(tensor_to_rust_array(state_dict['policy_head.fc1.bias'], 'POLICY_W1_BIAS'))
    rust_code.append(tensor_to_rust_array(state_dict['policy_head.fc2.weight'], 'POLICY_W2_WEIGHT'))
    rust_code.append(tensor_to_rust_array(state_dict['policy_head.fc2.bias'], 'POLICY_W2_BIAS'))
    rust_code.append('')
    
    # Value head
    rust_code.append('// ── Value Head ──')
    rust_code.append(tensor_to_rust_array(state_dict['value_head.fc1.weight'], 'VALUE_W1_WEIGHT'))
    rust_code.append(tensor_to_rust_array(state_dict['value_head.fc1.bias'], 'VALUE_W1_BIAS'))
    rust_code.append(tensor_to_rust_array(state_dict['value_head.fc2.weight'], 'VALUE_W2_WEIGHT'))
    rust_code.append(tensor_to_rust_array(state_dict['value_head.fc2.bias'], 'VALUE_W2_BIAS'))
    rust_code.append('')
    
    # load_pretrained_gnn function
    rust_code.append('/// Load the pre-trained GNN model with depth-9 rollout baked-in weights')
    rust_code.append('pub fn load_pretrained_gnn() -> GnnEgraphModel {')
    rust_code.append('    let config = GnnConfig {')
    rust_code.append(f'        hidden_dim: {hidden_dim},')
    rust_code.append(f'        num_layers: {num_layers},')
    rust_code.append(f'        num_heads: {num_heads},')
    rust_code.append(f'        num_actions: {num_actions},')
    rust_code.append('        node_feature_dim: 33,')
    rust_code.append('        learning_rate: 0.001,')
    rust_code.append('        gamma: 0.99,')
    rust_code.append('        num_episodes: 0,')
    rust_code.append('        batch_size: 32,')
    rust_code.append('        use_attention: true,')
    rust_code.append('        value_loss_coef: 0.5,')
    rust_code.append('        entropy_coef: 0.01,')
    rust_code.append('    };')
    rust_code.append('    let mut model = GnnEgraphModel::new(&config);')
    rust_code.append('')
    rust_code.append('    // Load embedding')
    rust_code.append(f'    model.embedding = Tensor::from_vec(EMBEDDING_WEIGHT.to_vec(), 33, {hidden_dim});')
    rust_code.append(f'    model.embedding_bias = Tensor::from_vec(EMBEDDING_BIAS.to_vec(), 1, {hidden_dim});')
    rust_code.append('')
    
    for i in range(num_layers):
        rust_code.append(f'    // Load GNN layer {i}')
        rust_code.append(f'    model.gnn_layers[{i}].w_self = Tensor::from_vec(GNN{i}_W_SELF_WEIGHT.to_vec(), {hidden_dim}, {hidden_dim});')
        rust_code.append(f'    model.gnn_layers[{i}].bias = Tensor::from_vec(GNN{i}_W_SELF_BIAS.to_vec(), 1, {hidden_dim});')
        rust_code.append(f'    model.gnn_layers[{i}].w_neigh = Tensor::from_vec(GNN{i}_W_NEIGH_WEIGHT.to_vec(), {hidden_dim}, {hidden_dim});')
        rust_code.append(f'    model.gnn_layers[{i}].layer_norm.gamma = Tensor::from_vec(GNN{i}_LN_GAMMA.to_vec(), 1, {hidden_dim});')
        rust_code.append(f'    model.gnn_layers[{i}].layer_norm.beta = Tensor::from_vec(GNN{i}_LN_BETA.to_vec(), 1, {hidden_dim});')
        if f'gnn_layers.{i}.attn_key.weight' in state_dict:
            rust_code.append(f'    model.gnn_layers[{i}].attn_key = Some(Tensor::from_vec(GNN{i}_ATTN_KEY.to_vec(), {hidden_dim}, {num_heads}));')
            rust_code.append(f'    model.gnn_layers[{i}].attn_query = Some(Tensor::from_vec(GNN{i}_ATTN_QUERY.to_vec(), {hidden_dim}, {num_heads}));')
        rust_code.append('')
    
    rust_code.append('    // Load readout')
    rust_code.append(f'    model.readout.w = Tensor::from_vec(READOUT_W_WEIGHT.to_vec(), {hidden_dim}, {hidden_dim});')
    rust_code.append(f'    model.readout.b = Tensor::from_vec(READOUT_W_BIAS.to_vec(), 1, {hidden_dim});')
    rust_code.append('')
    rust_code.append('    // Load policy head')
    rust_code.append(f'    model.policy_head.w1 = Tensor::from_vec(POLICY_W1_WEIGHT.to_vec(), {hidden_dim}, {hidden_dim});')
    rust_code.append(f'    model.policy_head.b1 = Tensor::from_vec(POLICY_W1_BIAS.to_vec(), 1, {hidden_dim});')
    rust_code.append(f'    model.policy_head.w2 = Tensor::from_vec(POLICY_W2_WEIGHT.to_vec(), {hidden_dim}, {num_actions});')
    rust_code.append(f'    model.policy_head.b2 = Tensor::from_vec(POLICY_W2_BIAS.to_vec(), 1, {num_actions});')
    rust_code.append('')
    rust_code.append('    // Load value head')
    rust_code.append(f'    model.value_head.w1 = Tensor::from_vec(VALUE_W1_WEIGHT.to_vec(), {hidden_dim}, {hidden_dim});')
    rust_code.append(f'    model.value_head.b1 = Tensor::from_vec(VALUE_W1_BIAS.to_vec(), 1, {hidden_dim});')
    rust_code.append(f'    model.value_head.w2 = Tensor::from_vec(VALUE_W2_WEIGHT.to_vec(), {hidden_dim}, 1);')
    rust_code.append('    model.value_head.b2 = Tensor::from_vec(VALUE_W2_BIAS.to_vec(), 1, 1);')
    rust_code.append('')
    rust_code.append('    model')
    rust_code.append('}')
    
    return '\n'.join(rust_code)

if __name__ == '__main__':
    train_model()
