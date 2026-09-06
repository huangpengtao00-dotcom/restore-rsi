"""「失败」的定义必须相对这道题能达到什么,而不是一把绝对的尺子。

`data.max_score` 是绝对阈值(0.5)。这 12 道题穷举出来的上界差 **46.5 倍**
(0.0215 ~ 1.0000),实测 12 题里 4 题判反:

    任务             上界    贪心拿到   达成率    绝对阈说
    low_light_00    1.0000   0.6564    66%      成功  -> 永不进 batch
    low_light_01    0.9650   0.6695    69%      成功  -> 永不进 batch
    haze_noise_00   0.2019   0.2281   113%      失败  -> 进 batch
    haze_noise_01   0.4071   0.4766   117%      失败  -> 进 batch

前两道有巨大改进空间却 propose 从没见过,后两道已经做到头却让 propose 去找不存在
的改进。**batch 的组成本身是偏的**,而挪阈值救不了 —— 一把尺子量不了 46 倍量程。

这里钉住三件事:换算正确、没有上界时不静默、以及口径永远写在 feedback 里。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reef_example"))

from harness.scoring import attained  # noqa: E402

#: 真实测出来的四组数(见上表)。用真数据而不是编的,因为这条判据的全部意义
#: 就是"绝对阈和相对阈在这批任务上判得不一样"。
REAL = [
    ("low_light_00", 1.0000, 0.6564, 0.656),
    ("low_light_01", 0.9650, 0.6695, 0.694),
    ("haze_noise_00", 0.2019, 0.2281, 1.130),
    ("haze_noise_01", 0.4071, 0.4766, 1.171),
    ("haze_01", 0.5180, 0.5162, 0.997),
    ("noise_01", 1.0000, 0.0483, 0.048),
]


@pytest.mark.parametrize("tid,ceiling,score,expected", REAL, ids=[r[0] for r in REAL])
def test_attained_is_the_score_over_the_ceiling(tid, ceiling, score, expected):
    reported, note = attained(score, {"ceiling": ceiling, "ceiling_depth": 3})
    assert reported == pytest.approx(expected, abs=0.001)
    # 绝对分和上界都要留在说明里,否则这个换算不可还原,历史数字也对不上
    assert f"{score:.4f}" in note and f"{ceiling:.4f}" in note


@pytest.mark.parametrize("tid,ceiling,score,expected", REAL, ids=[r[0] for r in REAL])
def test_the_two_rules_disagree_exactly_where_measured(tid, ceiling, score, expected):
    """绝对阈与相对阈的判定差异,必须和实测那 4/12 一致。

    这条不是重复上一条:上一条测算术,这条测**这个改动确实改变了 batch 的组成**。
    如果两把尺子判得一样,整个改动就没有意义,而那种"没意义"不会让别的测试变红。
    """
    absolute_says_failed = score <= 0.5
    relative_says_failed = expected <= 0.8
    known_disagreement = tid in {"low_light_00", "low_light_01", "haze_noise_00", "haze_noise_01"}
    assert (absolute_says_failed != relative_says_failed) == known_disagreement, (
        f"{tid}: 绝对阈判{'失败' if absolute_says_failed else '成功'},"
        f"相对阈判{'失败' if relative_says_failed else '成功'};与实测记录不符"
    )


def test_missing_ceiling_falls_back_but_says_so():
    """没有上界就报绝对分 —— 但必须在说明里写出来。

    同一批数字里混着两种口径而没人看得出来,比用错口径更糟:它让"这些分能不能
    放一起比"这个问题变得无法回答,而那正是这个仓反复栽过的那类错误。
    """
    reported, note = attained(0.42, {})
    assert reported == 0.42
    assert "ceiling=none" in note and "absolute" in note


@pytest.mark.parametrize("bad", [0.0, None, -1.0], ids=["零", "null", "负数"])
def test_a_useless_ceiling_is_treated_as_absent(bad):
    """上界为 0 或负数时不能拿去做除数,也不能假装换算过了。"""
    reported, note = attained(0.3, {"ceiling": bad})
    assert reported == 0.3
    assert "ceiling=none" in note
