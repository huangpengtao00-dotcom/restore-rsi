"""造"答案已知"的复原任务集:干净图 → 程序化退化(雾/低光/噪声,可叠加,强度记录)→ 落盘 manifest。

每个任务 = 一个 episode 的 prompt + 一组文件;**按帧序列设计**(frames 列表),单图是长度 1 的特例,
以后多视角/视频任务不改 schema(见 21 号文档:几何一致性 reward 需要序列)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def gen_test_image(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """合成底图:梯度背景 + 8 个随机色块,直方图够丰富,四个诊断读数都测得动。

    2026-09-06 从 judge-lab 内联进来(原 `judgelab.domains.color.synth.gen_test_image`)。
    一起去掉的是 `sys.path.insert(0, Path.home() / "research" / "judge-lab")`:那条硬编码的
    跨仓路径让这个仓**在别人的机器上和 CI 里都造不出数据**,而 `tasks/data/` 是 gitignored
    的,依赖 manifest 的测试于是静默 skip 掉 —— 不是变红,是消失,更难发现。

    内联版与原实现**逐字节一致**已验证:6 组 (seed x size) 加同一个 rng 上连续 5 次调用,
    `np.array_equal` 全部 True。所以 2026-09-03~09-04 那批实测数字与新生成的数据仍可比。
    真数据接 UIEB/CleanBench 时替换这一个函数即可,manifest schema 不变。
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) / size
    base = np.stack(
        [0.25 + 0.5 * xx, 0.25 + 0.5 * yy, 0.35 + 0.3 * (xx + yy) / 2], axis=-1
    )
    for _ in range(8):
        cx, cy = rng.uniform(0, size, 2)
        r = rng.uniform(size * 0.05, size * 0.22)
        color = rng.uniform(0.05, 0.95, 3)
        mask = (yy * size - cy) ** 2 + (xx * size - cx) ** 2 < r**2
        base[mask] = color
    return (base * 255).astype(np.uint8)


OUT = Path(__file__).parent / "data"


def add_haze(x: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
    """大气散射简化:I = J*trans + A*(1-trans),trans 随强度 t 降。"""
    A = rng.uniform(0.75, 0.95)
    trans = 1.0 - 0.75 * t
    return x * trans + A * (1.0 - trans)


def add_lowlight(x: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
    gamma = 1.0 + 2.2 * t
    return np.clip(x, 0, 1) ** gamma * (1.0 - 0.4 * t)


def add_noise(x: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
    return x + rng.normal(0, 0.12 * t, x.shape)


DEGRADERS = {"haze": add_haze, "low_light": add_lowlight, "noise": add_noise}


def make(n_per_combo: int = 2, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    combos = [("haze",), ("low_light",), ("noise",), ("haze", "low_light"), ("low_light", "noise"), ("haze", "noise")]
    manifest = []
    for combo in combos:
        for i in range(n_per_combo):
            clean = gen_test_image(rng, size=256)
            x = clean.astype(np.float64) / 255.0
            strengths = {}
            for d in combo:  # 叠加顺序即施加顺序,记录下来(最优复原顺序通常是逆序——agent 要学的东西)
                t = float(rng.uniform(0.4, 0.9))
                x = DEGRADERS[d](x, t, rng)
                strengths[d] = round(t, 3)
            degraded = (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
            tid = f"{'_'.join(combo)}_{i:02d}"
            tdir = OUT / tid
            tdir.mkdir(exist_ok=True)
            Image.fromarray(clean).save(tdir / "clean_000.png")
            Image.fromarray(degraded).save(tdir / "input_000.png")
            manifest.append(
                {
                    "task_id": tid,
                    "frames": [{"input": str(tdir / "input_000.png"), "reference": str(tdir / "clean_000.png")}],
                    "gt": {"applied_order": list(combo), "strengths": strengths},
                }
            )
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return OUT / "manifest.json"


if __name__ == "__main__":
    p = make()
    print(p, len(json.loads(p.read_text())), "tasks")
