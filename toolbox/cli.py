"""`restore` —— agent 在 episode 里唯一能碰的复原接口。四个子命令:

  restore catalog                       列出工具(来自 registry.yaml)
  restore diagnose IN                   退化诊断 JSON(现在是启发式;以后换 VLM/IQA,接口不变)
  restore run TOOL IN OUT               跑一个工具;按 (tool, 输入内容哈希) 缓存,同序列不重跑
  restore score IN OUT [--ref REF]      打分并写 verifier 记录;stdout 最后一行 `REEF_SCORE=<float>`

设计约束(与 reef episode 机制对齐):
- episode 目录跑完即删,evaluate() 只能拿到 stdout/trajectory —— 所以 `score` 把结果、证据、
  工具链写到 RESTORE_RESULTS_DIR(外部持久目录),并把分数打到 stdout 最后一行。
- 每一步都落 JSONL 轨迹(工具、输入/输出哈希、耗时、状态),这是可观测与失败归因的原料。
- 不静默降级:工具崩 → 非零退出 + 显式 status,绝不悄悄把输入当输出。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).parent
RESULTS_DIR = Path(os.environ.get("RESTORE_RESULTS_DIR", HERE.parent / "work" / "results"))
CACHE_DIR = Path(os.environ.get("RESTORE_CACHE_DIR", HERE.parent / "work" / "cache"))
TRACE_PATH = Path(os.environ.get("RESTORE_TRACE", RESULTS_DIR / "trace.jsonl"))
EPISODE_ID = os.environ.get("RESTORE_EPISODE_ID", "adhoc")


def _registry() -> dict:
    return yaml.safe_load((HERE / "registry.yaml").read_text())["tools"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _trace(event: dict) -> None:
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": time.time(), "episode": EPISODE_ID, **event}
    with TRACE_PATH.open("a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


# ---------------- catalog ----------------

def cmd_catalog() -> int:
    reg = _registry()
    by_task: dict[str, list] = {}
    for name, spec in reg.items():
        by_task.setdefault(spec["task"], []).append({"tool": name, "cost": spec.get("cost", 1), "note": spec.get("note", "")})
    print(json.dumps(by_task, ensure_ascii=False, indent=2))
    return 0


# ---------------- diagnose ----------------

def diagnose_array(img: np.ndarray) -> dict:
    """启发式退化诊断(替身)。输出稠密强度 0-1,不做硬分类——让 agent 自己决策。
    以后换 VLM/专用诊断模型时保持同一 JSON 形状。"""
    x = img.astype(np.float64) / 255.0
    lum = x @ np.array([0.299, 0.587, 0.114])
    dark = x.min(axis=2)
    # 雾:暗通道整体偏高 + 全局对比度低
    haze = float(np.clip((dark.mean() - 0.15) / 0.5, 0, 1)) * float(np.clip(1.0 - lum.std() / 0.25, 0, 1))
    # 低光:亮度均值低
    low_light = float(np.clip((0.35 - lum.mean()) / 0.35, 0, 1))
    # 噪声:拉普拉斯高频能量(与锐度共享,故用中值残差)
    med = np.asarray(Image.fromarray(img).filter(__import__("PIL.ImageFilter", fromlist=["MedianFilter"]).MedianFilter(3))).astype(np.float64) / 255.0
    noise = float(np.clip(np.abs(x - med).mean() / 0.04, 0, 1))
    # 模糊:梯度能量低
    gy, gx = np.gradient(lum)
    blur = float(np.clip(1.0 - np.sqrt(gx**2 + gy**2).mean() / 0.05, 0, 1))
    return {
        "degradations": {"haze": round(haze, 3), "low_light": round(low_light, 3), "noise": round(noise, 3), "blur": round(blur, 3)},
        "stats": {"lum_mean": round(float(lum.mean()), 3), "lum_std": round(float(lum.std()), 3), "dark_channel_mean": round(float(dark.mean()), 3)},
        "diagnoser": "heuristic-v0",
    }


def cmd_diagnose(inp: Path) -> int:
    d = diagnose_array(_load(inp))
    d["input_sha"] = _sha(inp)
    _trace({"kind": "diagnose", "input_sha": d["input_sha"], "result": d["degradations"]})
    print(json.dumps(d, ensure_ascii=False, indent=2))
    return 0


# ---------------- run ----------------

def cmd_run(tool: str, inp: Path, out: Path) -> int:
    reg = _registry()
    if tool not in reg:
        print(json.dumps({"status": "unknown_tool", "tool": tool, "known": sorted(reg)}), file=sys.stderr)
        return 2
    in_sha = _sha(inp)
    cache_key = f"{tool}-{in_sha}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{cache_key}.png"
    t0 = time.time()
    if cached.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(cached.read_bytes())
        _trace({"kind": "run", "tool": tool, "input_sha": in_sha, "output_sha": _sha(out), "status": "cache_hit", "ms": 0})
        print(json.dumps({"status": "cache_hit", "tool": tool, "output": str(out)}))
        return 0
    backend = reg[tool]["backend"]
    try:
        if backend.startswith("builtin:"):
            from . import builtin  # type: ignore

            fn = getattr(builtin, backend.split(":", 1)[1])
            result = fn(_load(inp))
        elif backend.startswith("docker:"):
            raise NotImplementedError("docker backend: 接真 JarvisIR 工具时实现,接口 in.png out.png 不变")
        else:
            raise ValueError(f"bad backend spec: {backend}")
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result).save(out)
        cached.write_bytes(out.read_bytes())
    except Exception as e:
        _trace({"kind": "run", "tool": tool, "input_sha": in_sha, "status": "failed", "error": repr(e), "ms": int((time.time() - t0) * 1000)})
        print(json.dumps({"status": "failed", "tool": tool, "error": repr(e)}), file=sys.stderr)
        return 1
    ms = int((time.time() - t0) * 1000)
    _trace({"kind": "run", "tool": tool, "input_sha": in_sha, "output_sha": _sha(out), "status": "ok", "ms": ms})
    print(json.dumps({"status": "ok", "tool": tool, "output": str(out), "ms": ms}))
    return 0


# ---------------- score ----------------

def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 99.0 if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def cmd_score(inp: Path, out: Path, ref: Path | None) -> int:
    """有参考(合成任务,答案已知)→ PSNR 归一到 0-1;无参考 → 诊断分改善量。
    两者都记进 verifier 记录;无参考分数是已知不可靠的(见 judge-lab 结论),标 `reliability: low`。"""
    if not out.exists():
        _trace({"kind": "score", "status": "no_output"})
        print(json.dumps({"status": "no_output"}), file=sys.stderr)
        print("REEF_SCORE=0.0")
        return 1
    o = _load(out)
    record: dict = {"episode": EPISODE_ID, "input_sha": _sha(inp), "output_sha": _sha(out)}
    if ref is not None and ref.exists():
        r = _load(ref)
        if r.shape != o.shape:
            o_img = Image.fromarray(o).resize((r.shape[1], r.shape[0]))
            o = np.asarray(o_img)
        psnr = _psnr(o, r)
        score = float(np.clip((psnr - 15.0) / 20.0, 0, 1))  # 15dB→0, 35dB→1
        record.update({"mode": "reference", "psnr": round(psnr, 3), "score": round(score, 4), "reliability": "high"})
    else:
        before, after = diagnose_array(_load(inp))["degradations"], diagnose_array(o)["degradations"]
        improve = float(np.mean([before[k] - after[k] for k in before]))
        score = float(np.clip(0.5 + improve, 0, 1))
        record.update({"mode": "no_reference", "before": before, "after": after, "score": round(score, 4), "reliability": "low"})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{EPISODE_ID}.verdict.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    (RESULTS_DIR / f"{EPISODE_ID}.output.png").write_bytes(out.read_bytes())
    _trace({"kind": "score", **{k: v for k, v in record.items() if k in ("mode", "score", "psnr")}})
    print(json.dumps(record, ensure_ascii=False))
    print(f"REEF_SCORE={score:.4f}")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd, args = argv[0], argv[1:]
    if cmd == "catalog":
        return cmd_catalog()
    if cmd == "diagnose" and len(args) == 1:
        return cmd_diagnose(Path(args[0]))
    if cmd == "run" and len(args) == 3:
        return cmd_run(args[0], Path(args[1]), Path(args[2]))
    if cmd == "score" and len(args) >= 2:
        ref = Path(args[args.index("--ref") + 1]) if "--ref" in args else None
        return cmd_score(Path(args[0]), Path(args[1]), ref)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


def _entry() -> None:  # console_scripts 入口
    sys.exit(main(sys.argv[1:]))
