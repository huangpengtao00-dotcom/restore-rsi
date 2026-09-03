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

#: diagnose 的标定常数。来源写在 diagnose_array 的 docstring 里,
#: 由 tests/test_diagnose.py 的契约测试守住(干净图四项须 ~0、同底图须单调、模糊须能测出)。
_DIAG = {
    "low_light_zero": 0.32,   # 去雾后亮度均值高于此视为不暗(12 题实测:不暗 [0.324, 0.499])
    "low_light_span": 0.30,
    "haze_zero": 0.30,        # 亮度归一后的暗通道地板/大气光,干净图 0.281、无雾最高 0.281
    "haze_span": 0.45,
    "noise_span": 0.20,       # 强噪声实测中值残差 0.107–0.179,除以 0.20 不饱和
    # 模糊:二阶导能量 / 一阶导能量。零点取 12 张干净图里最低的那个(0.743),
    # 于是干净图与本任务集的退化图全读 0(它们都没加过模糊);高斯模糊 r=1 起开始读到。
    # 注意:grad/std 这类"自归一梯度能量"测不出模糊 —— 模糊把一阶导和 std 同比缩小,
    # 比值几乎不变(实测 r=0→3 只从 0.0548 降到 0.0473)。必须用二阶/一阶之比。
    "blur_zero": 0.74,
    "blur_span": 0.45,
}


def _luminance(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float64) / 255.0 @ np.array([0.299, 0.587, 0.114])


def diagnose_array(img: np.ndarray) -> dict:
    """退化诊断:四个 0–1 强度,不做硬分类——让 agent 自己决策。
    以后换 VLM/专用诊断模型时保持同一 JSON 形状。

    **每个读数都在"其他退化已被抵消"的图上测**,这是 v1 相对 v0 的全部区别。
    v0 直接在原图上测四个统计量,结果三个读数互相污染(2026-09-03 实测,同一底图、
    同一雾强度 t=0.7,只是加了暗光,雾读数就从 0.725 掉到 0.065,掉 91%);而 `blur`
    在**干净参考图上就读 0.912** —— 它测的其实是"梯度能量低",在任何不带噪声的图上
    都成立,所以它是个反噪声读数,不是模糊读数。agent 唯一的观测是这个,于是把从不
    出现在最优链里的锐化用成了第二常用工具。

    v1 的解耦:
    - `low_light` 在**去雾后**的图上测亮度均值 —— 雾的大气光会把原图提亮;
    - `haze` 在**亮度归一后**的图上测「暗通道 5% 分位 / 大气光」 —— 暗光会压缩暗通道,
      且除以 A 抵消整体明暗。用暗通道地板而不是均值,是因为均值被彩色高饱和区抬高;
    - `noise` / `blur` 在**去雾且亮度归一后**的图上测,`blur` 用 grad/std 自归一,
      因此不再随明暗与对比度漂移。

    残余掩盖:同底图同雾强度加暗光,雾读数仍掉 22%(v0 是 91%)。已知未消除。

    标定常数在 `_DIAG`,由 `tests/test_diagnose.py` 守住。
    """
    from PIL import ImageFilter

    from .builtin import dcp_estimate, dehaze_dcp, lowlight_gamma

    x = img.astype(np.float64) / 255.0

    dehazed = dehaze_dcp(img)                       # 抵消雾,再量暗
    lum_dehazed = float(_luminance(dehazed).mean())
    low_light = float(np.clip((_DIAG["low_light_zero"] - lum_dehazed) / _DIAG["low_light_span"], 0, 1))

    brightened = lowlight_gamma(img)                # 抵消暗,再量雾(雾仍在)
    airlight, _ = dcp_estimate(brightened)
    dark_floor = float(np.quantile((brightened.astype(np.float64) / 255.0).min(axis=2), 0.05))
    haze_feat = dark_floor / max(float(np.mean(airlight)), 1e-3)
    haze = float(np.clip((haze_feat - _DIAG["haze_zero"]) / _DIAG["haze_span"], 0, 1))

    normalised = lowlight_gamma(dehazed)            # 雾与暗都抵消,再量噪声与模糊
    n = normalised.astype(np.float64) / 255.0
    smoothed = np.asarray(Image.fromarray(normalised).filter(ImageFilter.MedianFilter(3)))
    med = smoothed.astype(np.float64) / 255.0
    noise = float(np.clip(np.abs(n - med).mean() / _DIAG["noise_span"], 0, 1))

    # 模糊在**中值滤波后**的图上量:噪声的高频会假装"锐利",不去掉就测不出模糊。
    lum_s = _luminance(smoothed)
    gy, gx = np.gradient(lum_s)
    grad = float(np.sqrt(gx**2 + gy**2).mean())
    lap = float(np.abs(np.gradient(gx, axis=1) + np.gradient(gy, axis=0)).mean())
    hf_ratio = lap / max(grad, 1e-9)
    blur = float(np.clip((_DIAG["blur_zero"] - hf_ratio) / _DIAG["blur_span"], 0, 1))

    return {
        "degradations": {"haze": round(haze, 3), "low_light": round(low_light, 3), "noise": round(noise, 3), "blur": round(blur, 3)},
        "stats": {
            "lum_mean_dehazed": round(lum_dehazed, 3),
            "dark_floor_over_airlight": round(haze_feat, 3),
            "high_freq_ratio": round(hf_ratio, 4),
        },
        "diagnoser": "heuristic-v1-decoupled",
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


#: 有参考时的分数刻度:相对退化输入的 PSNR 改善量,除以 IMPROVEMENT_SPAN 归一。
#:
#: v0 用的是绝对刻度 `(PSNR-15)/20`,下限 15 dB **高于任务起点** —— 12 个任务里 8 个
#: 的退化图本身就低于 15 dB(三个评测任务是 13.19 / 12.76 / 8.59 dB)。于是 8.6 dB 和
#: 14.9 dB 都记 0.0:一整段真实差异对闸门不可见。实测代价:18 个 episode 分数里 7 个
#: (39%)钉死在 0,9 次逐题比较里 5 次至少一边是 0 —— 那些"平局"不是平局。而且单个
#: 工具用一次几乎全得 0.0000,agent 的第一步永远拿不到反馈。
#:
#: 改成改善量之后:identity(什么都不做)恒等于 0.0,而任何真实改善都有梯度。
#: 逐题看这只是减掉一个常数(单调变换),所以**不改变闸门的逐题胜负**,只是把被地板
#: 压平的那段还原出来。SPAN=15 dB 的依据:各步数预算下最优链的实测改善量最大约 14 dB。
IMPROVEMENT_SPAN_DB = 15.0


def reference_score(psnr_out: float, psnr_in: float) -> float:
    """改善量归一到 0–1。判据只在这里定义,oracle 与 judge 都调它,不各自重推。"""
    return float(np.clip((psnr_out - psnr_in) / IMPROVEMENT_SPAN_DB, 0, 1))


def cmd_score(inp: Path, out: Path, ref: Path | None) -> int:
    """有参考(合成任务,答案已知)→ 相对输入的 PSNR 改善量归一到 0-1;无参考 → 诊断分改善量。
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
        i = _load(inp)
        if r.shape != i.shape:
            i = np.asarray(Image.fromarray(i).resize((r.shape[1], r.shape[0])))
        psnr, psnr_in = _psnr(o, r), _psnr(i, r)
        score = reference_score(psnr, psnr_in)
        record.update({
            "mode": "reference",
            "psnr": round(psnr, 3),
            "psnr_input": round(psnr_in, 3),
            "gain_db": round(psnr - psnr_in, 3),
            "score": round(score, 4),
            "reliability": "high",
        })
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
