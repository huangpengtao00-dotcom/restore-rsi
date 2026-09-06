"""消融开关:每一个都必须真的改变行为,而且默认必须是修好的那一侧。

这套开关不是配置项,是**实验装置**。文献核查(Obsidian 31 号)显示"自进化闸门
不可信"在文本域已经有人做了(PACE arXiv 2606.08106 用 anytime-valid e-process,
量出 30–42% false commits;HarnessFix 2606.06324 做了 harness 缺陷七层分类)。
它们报的是**错误率**,报不了**代价** —— GSM8K/SWE-Bench/Atari 上没有"这题最好能
做到多少"这个量。

复原任务有:工具确定性、动作空间 6 个、真值由构造给出,天花板可以穷举。所以问题
能从"这个缺陷错多少次"变成"**这个缺陷值多少可达分数**"。要回答它,缺陷必须能被
打开 —— 这就是这些开关存在的全部理由。

于是每个开关都要满足两条,少一条这装置就是坏的:
  1. 默认是修好的那一侧(打开缺陷必须是显式动作,不能是忘了设)
  2. 打开之后行为**确实不同**(否则这一臂等于没做,而那不会让任何测试变红)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reef_example"))

from harness.ablation import CHOICES, Ablation  # noqa: E402
from harness.scoring import MeasurementFailure, attained, encode_measurement_failure  # noqa: E402


# ------------------------------------------------- 配置层本身

def test_default_is_every_defect_repaired():
    a = Ablation()
    assert a.is_baseline and a.label() == "baseline"
    for name, choices in CHOICES.items():
        assert getattr(a, name) == choices[0], f"{name} 的默认值必须是修好的那一侧"


@pytest.mark.parametrize("spec", ["gate=nonsense", "typo=1", "noequals", "gate="])
def test_a_typo_is_refused_not_silently_defaulted(spec):
    """静默忽略一个拼错的开关,意味着实验标签说打开了缺陷而实际没打开。
    那种数据看起来完全正常,且事后无法分辨 —— 比跑失败糟得多。"""
    with pytest.raises(ValueError):
        Ablation.parse(spec)


def test_label_names_only_what_was_switched_on():
    a = Ablation.parse("measurement_failure=zero,gate=wins_over_losses")
    assert "measurement_failure=zero" in a.label() and "gate=wins_over_losses" in a.label()
    assert "batch_criterion" not in a.label(), "没动的开关不该出现在标签里"


# ------------------------------------------------- measurement_failure:四种编码

def test_default_raises_rather_than_scoring():
    with pytest.raises(MeasurementFailure):
        encode_measurement_failure("judge crashed", "haze_00")


@pytest.mark.parametrize(
    "mode,check,why",
    [
        ("zero", lambda v: v == 0.0, "JarvisArt 的 28 处:悲观偏差"),
        ("exclude", lambda v: math.isnan(v), "JarvisEvo 从分母消失:乐观偏差"),
        ("guess", lambda v: 0.0 <= v <= 1.0, "VIEScore 猜一个分:纯噪声"),
    ],
)
def test_each_encoding_reproduces_a_published_behaviour(mode, check, why, monkeypatch):
    monkeypatch.setenv("RESTORE_ABLATION", f"measurement_failure={mode}")
    assert check(encode_measurement_failure("judge crashed", "haze_00")), why


def test_exclude_really_disappears_from_the_gate(monkeypatch):
    """NaN 必须在闸门里既不算胜也不算负 —— 那才是"从分母消失",不是"记 0 分"。

    这两件事方向相反:记 0 压低分数(保守),从分母消失抬高均值(乐观)。如果 NaN
    被当成 0 处理,这一臂测的就不是它要测的东西了。
    """
    from harness.selection import _tally

    wins, losses = _tally((0.9, float("nan"), 0.2), (0.5, 0.5, 0.5))
    assert (wins, losses) == (1, 1), "NaN 应当既不计胜也不计负"


def test_guess_is_reproducible(monkeypatch):
    """不可复现的对照组等于没有对照组。"""
    monkeypatch.setenv("RESTORE_ABLATION", "measurement_failure=guess")
    a = encode_measurement_failure("boom", "haze_00", "ep1")
    b = encode_measurement_failure("boom", "haze_00", "ep1")
    c = encode_measurement_failure("boom", "haze_01", "ep1")
    assert a == b, "同样的输入必须给同样的分"
    assert a != c, "不同任务不该拿到同一个猜测值"


# ------------------------------------------------- batch_criterion

def test_the_two_batch_criteria_disagree_on_a_real_task(monkeypatch):
    """haze_noise_00:上界 0.2019,贪心拿 0.2281。

    相对判据说它达成 113%(不进 batch);绝对判据说 0.2281 <= 0.5(进 batch,
    于是 propose 被要求为一道已经做到头的题找改进)。这一臂要测的就是这个差。
    """
    frame = {"ceiling": 0.2019, "ceiling_depth": 3}
    monkeypatch.setenv("RESTORE_ABLATION", "batch_criterion=attained")
    rel, _ = attained(0.2281, frame)
    monkeypatch.setenv("RESTORE_ABLATION", "batch_criterion=absolute")
    ab, note = attained(0.2281, frame)
    assert rel > 1.0 and ab == pytest.approx(0.2281)
    assert (rel <= 0.5) != (ab <= 0.5), "两个判据在这道题上必须判得不一样"
    assert "ablation:absolute" in note, "口径要写在 feedback 里,否则事后分不清"


# ------------------------------------------------- diagnosis

def _img(name: str) -> np.ndarray:
    return np.asarray(Image.open(ROOT / "tasks" / "data" / "haze_low_light_00" / name).convert("RGB"))


@pytest.mark.skipif(
    not (ROOT / "tasks" / "data" / "haze_low_light_00" / "clean_000.png").exists(),
    reason="tasks/data 未生成:先跑 tasks/make_tasks.py",
)
def test_v0_diagnosis_is_actually_broken_in_the_documented_way(monkeypatch):
    """v0 臂必须复现出它当初的病,否则这一臂测的不是那个缺陷。

    两条症状,都在 2026-09-03 的审计里记过:
      * blur 在**干净参考图上**就读得很高(它其实是个反噪声读数)
      * haze 在干净图上读得**不比退化图低** —— 也就是完全不可判别
    """
    from toolbox.cli import diagnose_array

    monkeypatch.setenv("RESTORE_ABLATION", "diagnosis=coupled_v0")
    degraded = diagnose_array(_img("input_000.png"))["degradations"]
    clean = diagnose_array(_img("clean_000.png"))["degradations"]
    assert clean["blur"] > 0.5, f"v0 的 blur 在干净图上应当虚高,实际 {clean['blur']}"
    assert clean["haze"] >= degraded["haze"], (
        f"v0 的 haze 应当不可判别(干净 {clean['haze']} vs 退化 {degraded['haze']})"
    )


@pytest.mark.skipif(
    not (ROOT / "tasks" / "data" / "haze_low_light_00" / "clean_000.png").exists(),
    reason="tasks/data 未生成",
)
def test_v1_diagnosis_does_not_have_those_symptoms():
    """反向:默认那一臂必须没有上面两条症状。少了这条,上一条测的可能只是"两版不同"。"""
    from toolbox.cli import diagnose_array

    degraded = diagnose_array(_img("input_000.png"))["degradations"]
    clean = diagnose_array(_img("clean_000.png"))["degradations"]
    assert clean["blur"] == 0.0, f"v1 的 blur 在干净图上应当是 0,实际 {clean['blur']}"
    assert degraded["haze"] > clean["haze"], "v1 的 haze 必须可判别"


# ------------------------------------------------- gate

@pytest.mark.parametrize("mode,cls", [("sign_test", "SignTestSelector"), ("wins_over_losses", "WinsOverLossesSelector")])
def test_gate_arm_selects_the_right_selector(mode, cls, monkeypatch):
    monkeypatch.setenv("RESTORE_ABLATION", f"gate={mode}")
    from harness.selection import build_selector

    assert type(build_selector()).__name__ == cls


def test_the_two_gates_disagree_on_the_measured_step():
    """2026-09-03 真实发生的那一步:3 胜 0 负。

    旧规则发布(3>0),符号检验拒绝(n=3 时 p=0.125 > alpha=0.10)。这个差就是
    "闸门值多少可达分数"那一行要量的东西。
    """
    from harness.selection import p_value, sign_test_threshold

    wins, losses = 3, 0
    assert wins > losses, "旧规则会发布"
    assert wins < sign_test_threshold(wins + losses, 0.10), "符号检验应当拒绝"
    assert p_value(3, 3) == pytest.approx(0.125)
