"""Exhaustive best-chain search: the ceiling this toolbox can reach on a task.

An agent's score is only readable against a ceiling. `restore run` is
deterministic and the tool set is small, so the best chain up to a given depth
is computable outright: this walks every tool sequence of length <= --depth
depth-first, scoring each prefix, and reports the best one per task.

That number is the oracle - what a perfect planner with this toolbox would get.
The gap between it and an episode's score is the part harness evolution could
in principle recover; the gap between it and 1.0 is what needs better tools, not
a better strategy. Both are worth stating before claiming a loop "works".

    uv run python tasks/oracle_chains.py --depth 3 --out work/oracle.json

Pure CPU, no reef and no model: it calls toolbox.builtin directly and memoises
on the chain prefix, so a depth-3 sweep over 6 tools is 258 image ops per task.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def _load(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _score(out: np.ndarray, ref: np.ndarray, psnr_in: float) -> float:
    """判据**直接调 `restore score` 用的那两个函数**,不在这里重推。

    天花板必须用闸门自己的度量来量,否则量的是自己的假设(判据自写是复现过的坑)。
    """
    from toolbox.cli import _psnr, reference_score

    return reference_score(_psnr(out, ref), psnr_in)


def search(image: np.ndarray, ref: np.ndarray, tools: dict, depth: int) -> tuple[list[str], float, int, dict]:
    """(best chain, best score, chains evaluated, score of every depth-1 tool)."""
    from toolbox.cli import _psnr

    psnr_in = _psnr(image, ref)      # every score is a gain over this
    best_chain: list[str] = []
    best_score = _score(image, ref, psnr_in)  # the identity chain: exactly 0.0
    evaluated = 1
    singles: dict[str, float] = {}

    def walk(current: np.ndarray, chain: list[str]) -> None:
        nonlocal best_chain, best_score, evaluated
        if len(chain) >= depth:
            return
        for name, fn in tools.items():
            try:
                result = fn(current)
            except Exception as exc:  # noqa: BLE001 - a tool that throws is not a chain, and must not
                print(f"  ! {'>'.join([*chain, name])} raised {exc!r}")  # be silently skipped as if it scored 0
                continue
            score = _score(result, ref, psnr_in)
            evaluated += 1
            if not chain:
                singles[name] = round(score, 4)
            if score > best_score:
                best_chain, best_score = [*chain, name], score
            walk(result, [*chain, name])

    walk(image, [])
    return best_chain, best_score, evaluated, singles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--manifest", default=str(ROOT / "tasks" / "data" / "manifest.json"))
    parser.add_argument("--out", default=str(ROOT / "work" / "oracle.json"))
    parser.add_argument("--task", action="append", help="restrict to these task ids (repeatable)")
    args = parser.parse_args()

    from toolbox import builtin

    registry = yaml.safe_load((ROOT / "toolbox" / "registry.yaml").read_text())["tools"]
    tools = {
        name: getattr(builtin, spec["backend"].split(":", 1)[1])
        for name, spec in registry.items()
        if spec["backend"].startswith("builtin:")
    }
    print(f"tools: {', '.join(sorted(tools))} (depth <= {args.depth})")

    manifest = json.loads(Path(args.manifest).read_text())
    wanted = set(args.task) if args.task else None
    report = {"depth": args.depth, "tools": sorted(tools), "tasks": {}}
    for task in manifest:
        tid = task["task_id"]
        if wanted is not None and tid not in wanted:
            continue
        frame = task["frames"][0]
        t0 = time.monotonic()
        chain, score, evaluated, singles = search(_load(frame["input"]), _load(frame["reference"]), tools, args.depth)
        report["tasks"][tid] = {
            "identity": 0.0,  # 改善量刻度下,什么都不做恒为 0
            "best_chain": chain,
            "best_score": round(score, 4),
            "chains_evaluated": evaluated,
            "single_tool_scores": dict(sorted(singles.items(), key=lambda kv: -kv[1])),
        }
        print(f"{tid}: best {score:.4f} via {'>'.join(chain) or '(identity)'} "
              f"({evaluated} chains, {time.monotonic() - t0:.1f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
