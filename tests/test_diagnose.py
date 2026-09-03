"""诊断读数的契约。

这是 agent 在 episode 里唯一的观测,它有多准就决定策略能有多好。v0 的四个读数
互相污染,而且 `blur` 在干净参考图上就读 0.912 —— 实测后果:锐化成了 agent 第二
常用的工具(占 23.7% 的调用),而它在任何最优链里都不出现。所以这里把"读数必须
满足什么"写成测试,而不是靠看一眼觉得差不多。

四条契约:
1. **干净图四项都读 ~0** —— 没有退化就不该报退化;
2. **在场/不在场分得开** —— 读数得能回答"这个退化在不在";
3. **同一底图上对强度单调** —— 跨图比绝对值会被内容混淆(v0 审查时我就在这上面
   得出过错结论),所以单调性必须固定底图来测;
4. **`blur` 测得出真模糊** —— 否则"12 题都读 0"可能只是因为它恒为 0。

标定常数在 `toolbox.cli._DIAG`;改动它就要让这些测试重新通过。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from toolbox.cli import diagnose_array

MANIFEST = Path(__file__).resolve().parent.parent / "tasks" / "data" / "manifest.json"
FAMILIES = ("haze", "low_light", "noise")


def _load(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


@pytest.fixture(scope="module")
def tasks():
    if not MANIFEST.exists():
        pytest.skip("tasks/data 未生成:先跑 tasks/make_tasks.py")
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="module")
def readings(tasks):
    """(退化图读数, 干净图读数) —— 每张图只诊断一次。"""
    degraded = [(t["gt"]["applied_order"], diagnose_array(_load(t["frames"][0]["input"]))["degradations"]) for t in tasks]
    clean = [diagnose_array(_load(t["frames"][0]["reference"]))["degradations"] for t in tasks]
    return degraded, clean


# ---------------- 契约 1:干净图不报退化 ----------------

@pytest.mark.parametrize("key", ("haze", "low_light", "noise", "blur"))
def test_clean_images_read_near_zero(readings, key):
    _, clean = readings
    worst = max(c[key] for c in clean)
    assert worst < 0.15, f"干净参考图的 {key} 读到 {worst:.3f}(v0 的 blur 在这里读 0.912)"


# ---------------- 契约 2:在场/不在场分得开 ----------------

@pytest.mark.parametrize("family", FAMILIES)
def test_reading_separates_presence(readings, family):
    degraded, _ = readings
    present = [d[family] for chain, d in degraded if family in chain]
    absent = [d[family] for chain, d in degraded if family not in chain]
    assert present and absent
    assert min(present) > max(absent), (
        f"{family} 读数区间重叠:在场 [{min(present):.3f}, {max(present):.3f}] "
        f"vs 不在场 [{min(absent):.3f}, {max(absent):.3f}] —— 这个读数判断不了退化在不在"
    )


def test_blur_never_fires_on_a_task_set_without_blur(readings, tasks):
    degraded, _ = readings
    assert not any("blur" in t["gt"]["applied_order"] for t in tasks), "任务集里出现了模糊族,本测试的前提变了"
    worst = max(d["blur"] for _, d in degraded)
    assert worst < 0.15, f"没有任务加过模糊,blur 却读到 {worst:.3f}"


# ---------------- 契约 3:同底图上对强度单调 ----------------

def _base_image() -> np.ndarray:
    import sys

    sys.path.insert(0, str(Path.home() / "research" / "judge-lab"))
    from judgelab.domains.color.synth import gen_test_image

    return gen_test_image(np.random.default_rng(0), size=256)


def _u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


#: 与 tasks/make_tasks.py 的退化器同式(原样抄口径,不自己重推)。
def _add_haze(x, t, airlight=0.85):
    trans = 1.0 - 0.75 * t
    return x * trans + airlight * (1.0 - trans)


def _add_lowlight(x, t):
    return np.clip(x, 0, 1) ** (1.0 + 2.2 * t) * (1.0 - 0.4 * t)


def _add_noise(x, t, seed=1):
    return x + np.random.default_rng(seed).normal(0, 0.12 * t, x.shape)


STRENGTHS = (0.2, 0.4, 0.6, 0.8)


@pytest.mark.parametrize(
    "family,degrader",
    [("haze", _add_haze), ("low_light", _add_lowlight), ("noise", _add_noise)],
)
def test_reading_is_monotone_in_strength_on_a_fixed_base(family, degrader):
    """固定底图扫强度。跨图比绝对值会被内容混淆——审查时我在这上面判错过一次。"""
    base = _base_image().astype(np.float64) / 255.0
    values = [diagnose_array(_u8(degrader(base, t)))["degradations"][family] for t in STRENGTHS]
    assert all(a < b for a, b in zip(values, values[1:])), f"{family} 读数对强度不单调:{values}"


# ---------------- 契约 4:blur 测得出真模糊 ----------------

def test_blur_detects_real_blur():
    """否则"12 题都读 0"可能只是因为这个读数恒为 0。

    注意 grad/std 这类自归一梯度能量**测不出**模糊:模糊把一阶导和 std 同比缩小,
    比值几乎不动(实测 r=0→3 只从 0.0548 降到 0.0473)。现在用二阶/一阶导之比。
    """
    sharp = _base_image()
    values = [diagnose_array(sharp)["degradations"]["blur"]]
    for radius in (1.0, 2.0, 3.0):
        blurred = np.asarray(Image.fromarray(sharp).filter(ImageFilter.GaussianBlur(radius)))
        values.append(diagnose_array(blurred)["degradations"]["blur"])
    assert values[0] < 0.15, f"清晰图 blur 读 {values[0]:.3f}"
    assert all(a < b for a, b in zip(values, values[1:])), f"blur 对模糊半径不单调:{values}"
    assert values[-1] > 0.5, f"半径 3 的模糊只读到 {values[-1]:.3f}"
