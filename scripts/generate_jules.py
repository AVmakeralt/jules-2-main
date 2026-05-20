#!/usr/bin/env python3
"""Jules corpus generator.

Generates syntactically valid Jules source files suitable for parser/LLM training.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

INT_LITS = list(range(0, 512))
FLOAT_LITS = [round(i * 0.125, 3) for i in range(0, 256)]


def lit(is_float: bool) -> str:
    return f"{random.choice(FLOAT_LITS):.3f}" if is_float else str(random.choice(INT_LITS))


def var_name(fn_i: int, v_i: int) -> str:
    return f"v_{fn_i}_{v_i}"


def gen_assign(name: str, is_float: bool) -> str:
    op = random.choice(["+", "-", "*"])
    return f"{name} = {name} {op} {lit(is_float)};"


def gen_if(name: str, is_float: bool) -> str:
    return "\n".join(
        [
            f"if {name} > {lit(is_float)} {{",
            f"    {name} = {name} + {lit(is_float)};",
            "} else {",
            f"    {name} = {name} - {lit(is_float)};",
            "}",
        ]
    )


def gen_loop(name: str, is_float: bool) -> str:
    bound = random.randint(2, 24)
    if is_float:
        body = f"{name} = {name} + {lit(True)};"
    else:
        body = f"{name} = {name} + i + {lit(False)};"
    return f"for i in 0..{bound} {{\n    {body}\n}}"


def gen_function(index: int, min_stmts: int, max_stmts: int) -> str:
    fname = f"f_{index:06d}"
    stmts: List[str] = []

    vars_meta = []
    for vi in range(random.randint(1, 5)):
        mutable = random.random() < 0.8
        is_float = random.random() < 0.35
        v = var_name(index, vi)
        prefix = "let mut" if mutable else "let"
        stmts.append(f"{prefix} {v} = {lit(is_float)};")
        vars_meta.append((v, is_float, mutable))

    for _ in range(random.randint(min_stmts, max_stmts)):
        v, is_float, mutable = random.choice(vars_meta)
        kind = random.choices(["assign", "if", "loop", "print"], weights=[0.45, 0.2, 0.25, 0.1])[0]

        if not mutable:
            tmp = f"tmp_{v}"
            stmts.append(f"let mut {tmp} = {v};")
            v = tmp
            mutable = True

        if kind == "assign":
            stmts.append(gen_assign(v, is_float))
        elif kind == "if":
            stmts.append(gen_if(v, is_float))
        elif kind == "loop":
            stmts.append(gen_loop(v, is_float))
        else:
            stmts.append(f"let _observe_{index} = {v};")

    out = random.choice(vars_meta)[0]
    stmts.append(f"let _result_{index} = {out};")

    indented = "\n".join(f"    {line}" for block in stmts for line in block.splitlines())
    return f"fn {fname}() {{\n{indented}\n}}\n"


def build_file(count: int, seed: int | None, call_main: int, min_stmts: int, max_stmts: int) -> str:
    if seed is not None:
        random.seed(seed)

    parts = [f"// Generated Jules corpus: {count} functions (seed={seed})", ""]
    for i in range(1, count + 1):
        parts.append(gen_function(i, min_stmts, max_stmts))

    if call_main > 0:
        lines = ["    let mut dispatch = 0;"]
        for i in range(1, min(count, call_main) + 1):
            lines.append(f"    dispatch = dispatch + {i};")
        lines.append("    let _dispatch_done = dispatch;")
        main_body = "\n".join(lines)
        parts.append(f"fn main() {{\n{main_body}\n}}")

    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate parser-valid Jules source files")
    ap.add_argument("-n", "--num", type=int, default=1000, help="number of functions")
    ap.add_argument("-o", "--out", default="generated.jules", help="output path")
    ap.add_argument("--seed", type=int, default=None, help="PRNG seed")
    ap.add_argument("--call-main", type=int, default=0, help="emit a main that calls the first N functions")
    ap.add_argument("--min-stmts", type=int, default=3)
    ap.add_argument("--max-stmts", type=int, default=8)
    args = ap.parse_args()

    if args.num < 1:
        raise SystemExit("--num must be >= 1")
    if args.min_stmts < 1 or args.max_stmts < args.min_stmts:
        raise SystemExit("invalid statement bounds")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_file(
            count=args.num,
            seed=args.seed,
            call_main=args.call_main,
            min_stmts=args.min_stmts,
            max_stmts=args.max_stmts,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.num} functions to {output}")


if __name__ == "__main__":
    main()
