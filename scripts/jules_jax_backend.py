#!/usr/bin/env python3
"""Jules JAX backend bridge.

This script gives Jules a high-performance ML backend path by:
1) Loading a Jules-exported IR JSON for dense/MLP models.
2) Training with JAX + XLA (jit-compiled step function).
3) Exporting fp32 and INT8-per-output-channel weights that can be
   consumed by Jules' low-byte inference path.

Expected IR format (JSON):
{
  "schema_version": 1,
  "model_name": "PolicyNet",
  "input_dim": 128,
  "layers": [256, 256, 10],
  "activation": "gelu",
  "task": "classification"
}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
try:
    import jax
    import jax.numpy as jnp
    import optax
except Exception as exc:  # dependency/setup failure should be user-facing and explicit
    raise SystemExit(
        "JAX backend dependencies are missing or broken.\n"
        "Install with:\n"
        "  pip install --upgrade pip\n"
        "  pip install jax optax numpy\n"
        f"Original import error: {exc}"
    )


Array = jax.Array
Params = List[Dict[str, Array]]


@dataclass(frozen=True)
class JulesModelIR:
    schema_version: int
    model_name: str
    input_dim: int
    layers: List[int]
    activation: str = "gelu"
    task: str = "classification"


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 40
    batch_size: int = 256
    label_smoothing: float = 0.0


REQUIRED_IR_KEYS = {
    "schema_version",
    "model_name",
    "input_dim",
    "layers",
    "activation",
    "task",
}
LEGACY_OPTIONAL_KEYS = {"input_dim", "layers", "activation", "task"}
ALLOWED_IR_KEYS = REQUIRED_IR_KEYS | LEGACY_OPTIONAL_KEYS
ALLOWED_ACTIVATIONS = {"gelu", "relu", "silu", "tanh"}
ALLOWED_TASKS = {"classification", "regression"}


def _normalize_ir_dict(raw: Dict[str, Any], source: Path) -> Dict[str, Any]:
    """Accept beginner-friendly minimal IR and normalize to strict schema."""
    normalized = dict(raw)
    normalized.setdefault("schema_version", 1)
    normalized.setdefault("model_name", source.stem or "model")
    normalized.setdefault("activation", "gelu")
    normalized.setdefault("task", "classification")
    return normalized


def _validate_ir_dict(raw: Dict[str, Any], source: Path) -> None:
    unknown = sorted(set(raw.keys()) - ALLOWED_IR_KEYS)
    missing = sorted(REQUIRED_IR_KEYS - set(raw.keys()))
    if unknown:
        raise ValueError(f"IR validation failed ({source}): unknown keys: {unknown}")
    if missing:
        raise ValueError(f"IR validation failed ({source}): missing keys: {missing}")

    if not isinstance(raw["schema_version"], int) or raw["schema_version"] != 1:
        raise ValueError(
            f"IR validation failed ({source}): schema_version must be integer 1"
        )
    if not isinstance(raw["model_name"], str) or not raw["model_name"].strip():
        raise ValueError(f"IR validation failed ({source}): model_name must be non-empty string")
    if not isinstance(raw["input_dim"], int) or raw["input_dim"] <= 0:
        raise ValueError(f"IR validation failed ({source}): input_dim must be a positive integer")
    if not isinstance(raw["layers"], list) or len(raw["layers"]) == 0:
        raise ValueError(f"IR validation failed ({source}): layers must be a non-empty integer list")
    if any((not isinstance(v, int) or v <= 0) for v in raw["layers"]):
        raise ValueError(
            f"IR validation failed ({source}): all layers entries must be positive integers"
        )
    if raw["activation"] not in ALLOWED_ACTIVATIONS:
        raise ValueError(
            f"IR validation failed ({source}): activation must be one of {sorted(ALLOWED_ACTIVATIONS)}"
        )
    if raw["task"] not in ALLOWED_TASKS:
        raise ValueError(
            f"IR validation failed ({source}): task must be one of {sorted(ALLOWED_TASKS)}"
        )


def load_ir(path: Path) -> JulesModelIR:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"IR validation failed ({path}): top-level JSON must be an object")
    raw = _normalize_ir_dict(raw, path)
    _validate_ir_dict(raw, path)
    return JulesModelIR(
        schema_version=raw["schema_version"],
        model_name=str(raw["model_name"]),
        input_dim=int(raw["input_dim"]),
        layers=[int(x) for x in raw["layers"]],
        activation=str(raw["activation"]).lower(),
        task=str(raw["task"]).lower(),
    )


def _activation(name: str):
    if name == "relu":
        return jax.nn.relu
    if name == "silu":
        return jax.nn.silu
    if name == "tanh":
        return jnp.tanh
    return jax.nn.gelu


def init_params(ir: JulesModelIR, rng: Array) -> Params:
    dims = [ir.input_dim, *ir.layers]
    params: Params = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        rng, k_w, k_b = jax.random.split(rng, 3)
        limit = jnp.sqrt(6.0 / (in_dim + out_dim))
        w = jax.random.uniform(k_w, (in_dim, out_dim), minval=-limit, maxval=limit)
        b = jnp.zeros((out_dim,), dtype=jnp.float32)
        params.append({"w": w.astype(jnp.float32), "b": b})
    return params


def forward(params: Params, x: Array, activation_name: str) -> Array:
    act = _activation(activation_name)
    h = x
    for i, layer in enumerate(params):
        h = h @ layer["w"] + layer["b"]
        if i < len(params) - 1:
            h = act(h)
    return h


def cross_entropy_loss(logits: Array, y: Array, label_smoothing: float) -> Array:
    num_classes = logits.shape[-1]
    y_onehot = jax.nn.one_hot(y, num_classes)
    if label_smoothing > 0:
        y_onehot = y_onehot * (1.0 - label_smoothing) + label_smoothing / num_classes
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(y_onehot * log_probs, axis=-1))


def accuracy(logits: Array, y: Array) -> Array:
    return jnp.mean((jnp.argmax(logits, axis=-1) == y).astype(jnp.float32))


def make_step_fn(ir: JulesModelIR, optimizer: optax.GradientTransformation, label_smoothing: float):
    @jax.jit
    def step(params: Params, opt_state: optax.OptState, x: Array, y: Array):
        def loss_fn(p):
            logits = forward(p, x, ir.activation)
            return cross_entropy_loss(logits, y, label_smoothing), logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_next = optimizer.update(grads, opt_state, params)
        params_next = optax.apply_updates(params, updates)
        acc = accuracy(logits, y)
        return params_next, opt_state_next, loss, acc

    return step


def batches(x: np.ndarray, y: np.ndarray, batch_size: int):
    n = x.shape[0]
    for i in range(0, n, batch_size):
        yield x[i : i + batch_size], y[i : i + batch_size]


def train(ir: JulesModelIR, x: np.ndarray, y: np.ndarray, cfg: TrainConfig):
    rng = jax.random.PRNGKey(cfg.seed)
    params = init_params(ir, rng)
    optimizer = optax.adamw(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = optimizer.init(params)
    step = make_step_fn(ir, optimizer, cfg.label_smoothing)

    for epoch in range(cfg.epochs):
        perm = np.random.permutation(x.shape[0])
        x_epoch = x[perm]
        y_epoch = y[perm]

        losses = []
        accs = []
        for xb, yb in batches(x_epoch, y_epoch, cfg.batch_size):
            xb_j = jnp.asarray(xb, dtype=jnp.float32)
            yb_j = jnp.asarray(yb, dtype=jnp.int32)
            params, opt_state, loss, acc = step(params, opt_state, xb_j, yb_j)
            losses.append(float(loss))
            accs.append(float(acc))

        print(
            f"epoch={epoch + 1:03d} loss={np.mean(losses):.6f} "
            f"acc={np.mean(accs):.4f} batches={len(losses)}"
        )

    return params


def quantize_per_output_channel(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """INT8 quantization matching Jules' per-output-channel scale style.

    Input shape: [in_dim, out_dim]
    Output:
      qweights: int8 array [in_dim, out_dim]
      scales: float32 array [out_dim]
    """
    max_abs = np.max(np.abs(weights), axis=0)
    scales = np.where(max_abs > 0.0, max_abs / 127.0, 1.0).astype(np.float32)
    q = np.round(weights / scales[None, :]).clip(-127, 127).astype(np.int8)
    return q, scales


def export_params(params: Params, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # FP32 checkpoint
    fp32 = {}
    for i, layer in enumerate(params):
        fp32[f"layer_{i}.w"] = np.asarray(layer["w"], dtype=np.float32)
        fp32[f"layer_{i}.b"] = np.asarray(layer["b"], dtype=np.float32)
    np.savez(out_dir / "jules_jax_fp32.npz", **fp32)

    # Jules INT8 export JSON
    int8_payload = {"layers": []}
    for i, layer in enumerate(params):
        w = np.asarray(layer["w"], dtype=np.float32)
        b = np.asarray(layer["b"], dtype=np.float32)
        q, s = quantize_per_output_channel(w)
        int8_payload["layers"].append(
            {
                "name": f"layer_{i}",
                "in_dim": int(w.shape[0]),
                "out_dim": int(w.shape[1]),
                "qweights": q.reshape(-1).tolist(),
                "scales": s.tolist(),
                "bias": b.tolist(),
            }
        )

    (out_dir / "jules_int8_weights.json").write_text(json.dumps(int8_payload))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Jules MLP models with JAX backend")
    p.add_argument("--ir", type=Path, required=True, help="Path to Jules model IR JSON")
    p.add_argument("--dataset", type=Path, required=True, help="NPZ with arrays x_train, y_train")
    p.add_argument("--out", type=Path, default=Path("artifacts/jax"), help="Output directory")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ir = load_ir(args.ir)

    data = np.load(args.dataset)
    x = np.asarray(data["x_train"], dtype=np.float32)
    y = np.asarray(data["y_train"], dtype=np.int32)

    if x.ndim != 2:
        raise ValueError(f"x_train must be rank-2 [N, D], got shape {x.shape}")
    if x.shape[1] != ir.input_dim:
        raise ValueError(f"input_dim mismatch: IR={ir.input_dim} dataset={x.shape[1]}")

    cfg = TrainConfig(
        seed=args.seed,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        label_smoothing=args.label_smoothing,
    )

    print(f"IR: schema=v{ir.schema_version} model={ir.model_name} task={ir.task}")
    print(f"JAX devices: {[d.device_kind for d in jax.devices()]}")
    params = train(ir, x, y, cfg)
    export_params(params, args.out)
    print(f"Exported artifacts to: {args.out}")


if __name__ == "__main__":
    main()
