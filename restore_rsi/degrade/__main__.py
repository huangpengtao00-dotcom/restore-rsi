"""CLI:

  python -m restore_rsi.degrade make --families night,fog,night+fog --severities 0,1,2,3,4 --n 20 --seed 0 --out tasks/data_v1
      [--src DIR]            干净图目录(默认用 judge-lab gen_test_image 造合成底图)
      [--depth auto|synthetic|depth_anything]   雾的深度来源;auto=有 Depth-Anything-V2-Small 就用,否则合成并在 manifest 标明
      [--frames K] [--max-side 512]
  python -m restore_rsi.degrade replay --manifest tasks/data_v1/manifest.json      逐条重放并核对位级一致
  python -m restore_rsi.degrade strip --src DIR --n 5 --families night,fog --out work/probe_strip.png   五档对比条

国内网络下载 Depth-Anything 权重:HF_ENDPOINT=https://hf-mirror.com(权重进 ~/.cache/huggingface,不进仓)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from . import FAMILIES
from .base import SEVERITIES
from .depth import resolve_backend, synthetic_depth
from .pipeline import (
    degrade_frame,
    derive_seed,
    list_source_images,
    load_image,
    load_manifest,
    make_dataset,
    parse_chain,
    quantize_depth,
    replay_task,
    sample_params,
)


def _csv_int(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def cmd_make(a) -> int:
    mpath = make_dataset(
        out=Path(a.out),
        chains=[c for c in a.families.split(",") if c.strip()],
        severities=_csv_int(a.severities),
        n=a.n,
        seed=a.seed,
        src=Path(a.src).expanduser() if a.src else None,
        depth_mode=a.depth,
        n_frames=a.frames,
        max_side=a.max_side or None,
    )
    tasks = load_manifest(mpath)
    srcs = sorted({t["gt"]["depth_source"] for t in tasks if t["gt"]["depth_source"]})
    print(f"{mpath}  tasks={len(tasks)}  depth_source={srcs or 'n/a'}")
    return 0


def cmd_replay(a) -> int:
    tasks = load_manifest(Path(a.manifest))
    bad = 0
    for t in tasks:
        for frame, regen in zip(t["frames"], replay_task(t), strict=True):
            stored = np.asarray(Image.open(frame["input"]).convert("RGB"))
            if stored.shape != regen.shape or not np.array_equal(stored, regen):
                bad += 1
                print(f"MISMATCH {t['task_id']} {frame['input']}", file=sys.stderr)
    print(f"replayed {len(tasks)} tasks, mismatches={bad}")
    return 1 if bad else 0


def cmd_strip(a) -> int:
    """每张底图一行组:每个族一行,列 = severity 0..4(第 0 列即原图)。"""
    chains = [parse_chain(c) for c in a.families.split(",") if c.strip()]
    files = list_source_images(Path(a.src).expanduser())
    step = max(1, len(files) // a.n)
    picks = files[:: step][: a.n]
    needs_depth = any(FAMILIES[f].NEEDS_DEPTH for ch in chains for f in ch)
    est, dsrc, _ = resolve_backend(a.depth) if needs_depth else (None, None, {})
    tile = a.tile
    rows = []
    for i, p in enumerate(picks):
        clean = load_image(p, tile)
        depth = None
        if needs_depth:
            depth = quantize_depth(est(clean) if est else synthetic_depth(clean.shape[:2], np.random.default_rng(derive_seed(a.seed, 3, i))))
        for ci, chain in enumerate(chains):
            row = []
            for sev in SEVERITIES:
                params = sample_params(chain, sev, derive_seed(a.seed, 1, ci, i))
                seeds = {name: derive_seed(a.seed, 2, ci, sev, i, 0, k) for k, name in enumerate(chain)}
                row.append(_fit(degrade_frame(clean, chain, params, seeds, depth), tile))
            rows.append(np.concatenate(row, axis=1))
    width = max(r.shape[1] for r in rows)
    canvas = np.concatenate([np.pad(r, ((0, 4), (0, width - r.shape[1]), (0, 0))) for r in rows], axis=0)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out)
    print(f"{out}  {canvas.shape[1]}x{canvas.shape[0]}  rows={[p.name for p in picks]} families={a.families} depth_source={dsrc}")
    return 0


def _fit(img: np.ndarray, tile: int) -> np.ndarray:
    """横向补到 tile 宽(高度保持),各列等宽对齐。"""
    h, w = img.shape[:2]
    return np.pad(img, ((0, 0), (0, max(0, tile - w) + 2), (0, 0)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m restore_rsi.degrade", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make")
    m.add_argument("--src", default=None)
    m.add_argument("--families", default="night,fog", help=f"逗号分隔的链;链内用 + 叠加。族:{sorted(FAMILIES)}")
    m.add_argument("--severities", default="0,1,2,3,4")
    m.add_argument("--n", type=int, default=20)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", default="tasks/data_v1")
    m.add_argument("--depth", default="auto", choices=["auto", "synthetic", "depth_anything"])
    m.add_argument("--frames", type=int, default=1)
    m.add_argument("--max-side", type=int, default=512)
    m.set_defaults(fn=cmd_make)

    r = sub.add_parser("replay")
    r.add_argument("--manifest", required=True)
    r.set_defaults(fn=cmd_replay)

    s = sub.add_parser("strip")
    s.add_argument("--src", required=True)
    s.add_argument("--n", type=int, default=5)
    s.add_argument("--families", default="night,fog")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--tile", type=int, default=256)
    s.add_argument("--depth", default="auto", choices=["auto", "synthetic", "depth_anything"])
    s.add_argument("--out", default="work/probe_strip.png")
    s.set_defaults(fn=cmd_strip)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
