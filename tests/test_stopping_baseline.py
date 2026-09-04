"""贪心基线的契约:它是 LLM agent 必须被对照的那条线。

写这个测试的理由:2026-09-04 实测发现,一条只用 agent 自己看得见的信息的确定性
规则——每步试走全部 6 个工具、选诊断总分降幅最大的、降不动就停——在三个评测任务
上拿到 0.607 / 0.267 / 0.649,**比 LLM agent 的典型表现好,而且从不归零**(agent
的 episode 分数在 0.0 到 0.57 之间跳)。

也就是说:自进化循环现在想学的东西,一条十行的规则已经做得更好。任何"循环学会了
什么"的说法,都要先跨过这条线。

这个测试把基线钉住,防止工具或诊断的改动悄悄挪动它 —— 挪动了就说明对照失效,
所有"agent 更好了"的结论都要重算。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tasks" / "data" / "manifest.json"

#: 2026-09-04 实测值。容差 0.02 —— 工具确定性,这里的浮动只来自诊断标定的微调。
BASELINE = {
    "haze_low_light_00": 0.6071,
    "haze_low_light_01": 0.2670,
    "low_light_noise_00": 0.6489,
}
MAX_STEPS = 8


@pytest.fixture(scope="module")
def tools():
    sys.path.insert(0, str(ROOT))
    from toolbox import builtin

    registry = yaml.safe_load((ROOT / "toolbox" / "registry.yaml").read_text())["tools"]
    return {
        name: getattr(builtin, spec["backend"].split(":", 1)[1])
        for name, spec in registry.items()
        if spec["backend"].startswith("builtin:")
    }


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST.exists():
        pytest.skip("tasks/data 未生成:先跑 tasks/make_tasks.py")
    return {task["task_id"]: task for task in json.loads(MANIFEST.read_text())}


def _greedy(task: dict, tools: dict) -> tuple[float, float]:
    """(停手时的分数, 轨迹上出现过的最高分)。只用盲信息决定走不走。"""
    from toolbox.cli import _psnr, diagnose_array, reference_score

    frame = task["frames"][0]
    load = lambda p: np.asarray(Image.open(p).convert("RGB"))  # noqa: E731
    img, ref = load(frame["input"]), load(frame["reference"])
    psnr_in = _psnr(img, ref)
    score = best = 0.0
    for _ in range(MAX_STEPS):
        current = sum(diagnose_array(img)["degradations"].values())
        pick = None
        for fn in tools.values():
            try:
                nxt = fn(img)
            except Exception:  # noqa: BLE001 - 抛异常的工具不是候选,但也不能当成 0 分
                continue
            drop = current - sum(diagnose_array(nxt)["degradations"].values())
            if pick is None or drop > pick[0]:
                pick = (drop, nxt)
        if pick is None or pick[0] <= 0.0:
            break
        img = pick[1]
        score = reference_score(_psnr(img, ref), psnr_in)
        best = max(best, score)
    return score, best


@pytest.mark.parametrize("task_id,expected", BASELINE.items())
def test_greedy_baseline_holds(manifest, tools, task_id, expected):
    score, _ = _greedy(manifest[task_id], tools)
    assert score == pytest.approx(expected, abs=0.02), (
        f"{task_id} 的贪心基线从 {expected} 变成 {score:.4f} —— 对照线动了,"
        f"所有拿 agent 和它比的结论都要重算"
    )


def test_greedy_never_collapses_to_zero(manifest, tools):
    """agent 的 episode 会归零(多走一步毁掉),贪心不会 —— 这正是它值得当基线的原因。"""
    for task_id in BASELINE:
        score, _ = _greedy(manifest[task_id], tools)
        assert score > 0.2, f"{task_id} 贪心也归零了({score:.4f}),基线的意义就没了"
