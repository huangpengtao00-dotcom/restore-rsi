"""闸门判据的契约:发布这件事必须有已知的错误率。

reef 自带的 `wins > losses` 在这套系统上假阳性率 50%,与重复次数无关 —— 候选与
现版无差别时每次逐题比较就是抛硬币。实测代价:每题评估 1 次时闸门连续两步以
2 胜 1 负发布,而同一棵树的重复评估显示那些差(0.01–0.19)远小于噪声(0.15–0.65)。

这里钉住三件事:临界值确实控制住了 alpha、平局被剔除、以及真实数据上的判决。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reef_example"))

from harness.selection import SignTestSelector, p_value, sign_test_threshold  # noqa: E402


@pytest.mark.parametrize("n", range(1, 31))
@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_threshold_actually_controls_alpha(n, alpha):
    """临界值处的假阳性率必须 <= alpha,而低一档必须 > alpha(否则是过严)。"""
    k = sign_test_threshold(n, alpha)
    assert p_value(k, n) <= alpha, f"n={n} alpha={alpha}: 阈值 {k} 的假阳性率 {p_value(k, n):.3f} 超了"
    if k - 1 >= 0:
        assert p_value(k - 1, n) > alpha, f"n={n} alpha={alpha}: 阈值可以更低,{k} 过严"


def test_old_rule_would_be_a_coin_flip():
    """记录旧规则的错误率,这是换判据的全部理由。"""
    for n in (3, 9, 15):
        assert p_value(n // 2 + 1, n) == pytest.approx(0.5, abs=0.02), "wins>losses 的假阳性率就是 50%"


class _Evaluation:
    def __init__(self, candidate, current):
        self.metrics = {"candidate_scores": candidate, "current_scores": current}


def _decide(candidate, current, alpha=0.10):
    pytest.importorskip("reef.train.evaluation.contracts")
    return SignTestSelector(alpha=alpha).decide(None, _Evaluation(candidate, current))


def test_the_measured_step_is_rejected():
    """2026-09-04 实测的那一步:9 次比较 4 胜 5 负 —— 零假设下最可能的结果。"""
    current = [0.5722, 0.3999, 0.0, 0.2165, 0.2165, 0.0, 0.5791, 0.3517, 0.6198]
    candidate = [0.0, 0.44, 0.5722, 0.0, 0.0, 0.2165, 0.5209, 0.5645, 0.6054]
    decision = _decide(candidate, current)
    assert decision.outcome == "reject"
    assert decision.metrics["wins"] == 4 and decision.metrics["losses"] == 5


def test_a_two_of_three_split_no_longer_publishes():
    """旧规则下 2 胜 1 负会发布(实测发生过两次),符号检验下不会。"""
    decision = _decide([0.5, 0.5, 0.0], [0.4, 0.4, 0.1])
    assert decision.metrics["wins"] == 2 and decision.metrics["losses"] == 1
    assert decision.outcome == "reject"


def test_a_clean_sweep_publishes():
    decision = _decide([0.5] * 9, [0.1] * 9)
    assert decision.outcome == "select"
    assert decision.metrics["p_value"] < 0.10


def test_ties_are_dropped_not_counted_as_evidence():
    """平局对方向没有信息:6 个平局不应该稀释 3 胜 0 负。"""
    decision = _decide([0.5, 0.5, 0.5] + [0.2] * 6, [0.1, 0.1, 0.1] + [0.2] * 6)
    assert decision.metrics["ties"] == 6
    assert decision.metrics["n"] == 3, "平局不能进 n"


# ------------------------------------------------- 测不出来 ≠ 表现最差

def test_a_failed_episode_is_excluded_not_counted_as_a_loss():
    """`None` 是 episode 根本没跑起来,不是"它比对面差"。

    reef 的口径把 None 当 -inf,于是容器没起来、pi 崩了、机器卡了,都会变成闸门的
    证据 —— 白送对面一个 loss。照抄那个口径而没问它对不对,就是把"基础设施失效
    伪装成算法效果"这个坑抄进自己家(用了三天才发现)。

    正确的做法是排除:证据少了就该更难发布,而不是拿故障当成绩。
    """
    from harness.selection import _tally

    # 候选第二个 episode 没跑起来,另外两个一胜一负
    wins, losses, excluded = _tally((0.9, None, 0.2), (0.5, 0.5, 0.5))
    assert (wins, losses, excluded) == (1, 1, 1)
    # 旧口径会得到 (1, 2):那个 None 变成了一个 loss
    assert losses == 1, "没跑起来的 episode 不该算输"


def test_excluding_makes_the_gate_harder_not_easier():
    """排除之后 n 变小,符号检验自动更严 —— 这是对的,不是副作用。

    少了证据还想发布,就该更难。n=2 时临界值是 3,怎么都不显著。
    """
    from harness.selection import sign_test_threshold

    assert sign_test_threshold(9, 0.10) == 7
    assert sign_test_threshold(2, 0.10) > 2, "n=2 时应当无法达到临界值"
    assert sign_test_threshold(0, 0.10) == 1, "没有可比的对子就不可能发布"


def test_both_sides_unmeasurable_yields_no_evidence():
    from harness.selection import _tally

    assert _tally((None, None), (None, None)) == (0, 0, 2)
