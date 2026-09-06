"""把退化族串成任务集:底图 → 链式叠加(顺序显式)→ 落盘 + manifest;并提供逐条重放。

manifest 契约(与 tasks/make_tasks.py 同构,按帧序列设计,单图 = 长度 1):
{
  "task_id": "night+fog_s3_007",
  "frames": [{"input": ..., "reference": ..., "depth": ...|null}],     # 每帧一项
  "gt": {
    "applied_order": ["night", "fog"],                                   # 施加顺序(复原通常逆序)
    "severity":  {"night": 3, "fog": 3},
    "params":    {"night": {...全部数值...}, "fog": {...}},             # 序列内各帧共用(同一场景)
    "seeds":     [{"night": 123, "fog": 456, "depth": 789}],            # 每帧一项:各族噪声 seed + 深度 seed
    "depth_source": "depth_anything_v2_small" | "synthetic" | null,
    "depth_info":  {...模型 id 或回退原因...},
    "families":  {"night": {"surrogate": false, "model": "..."}, ...},
    "source": {"kind": "synthetic:gen_test_image" | "dir", "name": ...}
  }
}
重放:replay_task(entry) 用 reference + params + seeds(+ 已存深度图)重建每帧 input,位级一致。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from . import FAMILIES
from .base import check_severity
from .depth import SOURCE_SYN, depth_from_u16, depth_to_u16, resolve_backend, synthetic_depth

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ---------------- seeds ----------------

def derive_seed(master: int, *path: int) -> int:
    """从 master seed 和整数路径(任务序号/帧/阶段)确定性派生一个 32 位 seed。"""
    return int(np.random.SeedSequence([int(master), *[int(p) for p in path]]).generate_state(1, dtype=np.uint32)[0])


def parse_chain(spec: str) -> list[str]:
    chain = [s.strip() for s in spec.split("+") if s.strip()]
    if not chain:
        raise ValueError(f"empty chain spec {spec!r}")
    for name in chain:
        if name not in FAMILIES:
            raise ValueError(f"unknown family {name!r}; known: {sorted(FAMILIES)}")
    return chain


# ---------------- core ----------------

def quantize_depth(dist: np.ndarray) -> np.ndarray:
    """深度过一遍 u16 量化:生成与重放都用这份,保证存盘深度图 == 实际用的深度。"""
    return depth_from_u16(depth_to_u16(dist))


def degrade_frame(clean_u8: np.ndarray, chain: list[str], params: dict, seeds: dict, depth: np.ndarray | None) -> np.ndarray:
    """按 chain 顺序逐族施加。每族一个独立 rng(seed 来自 seeds[name]);需要深度的族拿传入的 depth。"""
    x = clean_u8
    for name in chain:
        fam = FAMILIES[name]
        rng = np.random.default_rng(int(seeds[name]))
        if fam.NEEDS_DEPTH:
            if depth is None:
                raise ValueError(f"family {name} needs depth but none given")
            x = fam.apply(x, params[name], rng, depth=depth)
        else:
            x = fam.apply(x, params[name], rng)
    return x


def sample_params(chain: list[str], severity: int, param_seed: int) -> dict:
    """各族参数从同一个 seed 派生的独立 rng 抽;同 seed 不同 severity → 同一组抖动(单调扫描的前提)。"""
    check_severity(severity)
    return {name: FAMILIES[name].params_for_severity(severity, np.random.default_rng(derive_seed(param_seed, i))) for i, name in enumerate(chain)}


# ---------------- sources ----------------

def list_source_images(src: Path) -> list[Path]:
    files = sorted(p for p in Path(src).iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"no images under {src}")
    return files


def load_image(path: Path, max_side: int | None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.LANCZOS)
    return np.asarray(im)


def gen_synthetic_base(rng: np.random.Generator, size: int) -> np.ndarray:
    """合成底图。实现在 restore_rsi.synth —— 全仓唯一一份,不再 sys.path 注入仓外路径。"""
    from restore_rsi.synth import gen_test_image

    return gen_test_image(rng, size=size)


# ---------------- dataset ----------------

def make_dataset(
    out: Path,
    chains: list[str],
    severities: list[int],
    n: int,
    seed: int,
    src: Path | None = None,
    depth_mode: str = "auto",
    n_frames: int = 1,
    max_side: int | None = 512,
    base_size: int = 256,
) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    parsed = [parse_chain(c) for c in chains]
    severities = [check_severity(s) for s in severities]
    needs_depth = any(FAMILIES[f].NEEDS_DEPTH for ch in parsed for f in ch)
    estimator, depth_source, depth_info = resolve_backend(depth_mode) if needs_depth else (None, None, {})
    src_files = list_source_images(src) if src else None

    manifest = []
    for ci, chain in enumerate(parsed):
        for si, sev in enumerate(severities):
            for i in range(n):
                # 底图:同 (ci, i) 在不同 severity 下用同一张底图,便于横向对比
                base_seed = derive_seed(seed, 0, ci, i)
                if src_files:
                    p = src_files[i % len(src_files)]
                    clean = load_image(p, max_side)
                    source = {"kind": "dir", "name": p.name, "path": str(p)}
                else:
                    clean = gen_synthetic_base(np.random.default_rng(base_seed), base_size)
                    source = {"kind": "synthetic:gen_test_image", "name": f"synthetic_{base_seed}"}
                param_seed = derive_seed(seed, 1, ci, i)
                params = sample_params(chain, sev, param_seed)

                tid = f"{'+'.join(chain)}_s{sev}_{i:03d}"
                tdir = out / tid
                tdir.mkdir(exist_ok=True)
                frames, seeds_per_frame = [], []
                depth = None
                task_depth_source = None
                if needs_depth and any(FAMILIES[f].NEEDS_DEPTH for f in chain):
                    task_depth_source = depth_source
                for f in range(n_frames):
                    seeds = {name: derive_seed(seed, 2, ci, si, i, f, k) for k, name in enumerate(chain)}
                    seeds["depth"] = derive_seed(seed, 3, ci, i)  # 深度是场景属性:序列内各帧共用
                    depth_path = None
                    if task_depth_source is not None:
                        if depth is None:  # 每任务算一次
                            raw = estimator(clean) if estimator else synthetic_depth(clean.shape[:2], np.random.default_rng(seeds["depth"]))
                            depth = quantize_depth(raw)  # 用的就是存盘的那份(u16),重放才能位级一致
                        depth_path = tdir / f"depth_{f:03d}.png"
                        Image.fromarray(depth_to_u16(depth)).save(depth_path)
                    degraded = degrade_frame(clean, chain, params, seeds, depth)
                    ref_path, in_path = tdir / f"clean_{f:03d}.png", tdir / f"input_{f:03d}.png"
                    Image.fromarray(clean).save(ref_path)
                    Image.fromarray(degraded).save(in_path)
                    frames.append({"input": str(in_path), "reference": str(ref_path), "depth": str(depth_path) if depth_path else None})
                    seeds_per_frame.append(seeds)
                manifest.append(
                    {
                        "task_id": tid,
                        "frames": frames,
                        "gt": {
                            "applied_order": list(chain),
                            "severity": {name: sev for name in chain},
                            "params": params,
                            "seeds": seeds_per_frame,
                            "depth_source": task_depth_source,
                            "depth_info": depth_info if task_depth_source else {},
                            "families": {name: {"surrogate": bool(FAMILIES[name].SURROGATE), "model": FAMILIES[name].MODEL} for name in chain},
                            "source": source,
                        },
                    }
                )
    mpath = out / "manifest.json"
    mpath.write_text(json.dumps({"version": 1, "seed": seed, "tasks": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    return mpath


# ---------------- replay ----------------

def replay_task(entry: dict) -> list[np.ndarray]:
    """用 manifest 条目重建每帧 input(不依赖已存的 input 图)。深度:synthetic 从 seed 重算,否则读已存深度图。"""
    gt = entry["gt"]
    chain = gt["applied_order"]
    outs = []
    for frame, seeds in zip(entry["frames"], gt["seeds"], strict=True):
        clean = np.asarray(Image.open(frame["reference"]).convert("RGB"))
        depth = None
        if gt.get("depth_source") is not None:
            if gt["depth_source"] == SOURCE_SYN:
                depth = quantize_depth(synthetic_depth(clean.shape[:2], np.random.default_rng(int(seeds["depth"]))))
            else:
                if not frame.get("depth"):
                    raise FileNotFoundError(f"{entry['task_id']}: depth_source={gt['depth_source']} but no depth file recorded")
                depth = depth_from_u16(np.asarray(Image.open(frame["depth"])))
        outs.append(degrade_frame(clean, chain, gt["params"], seeds, depth))
    return outs


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["tasks"] if isinstance(data, dict) else data
