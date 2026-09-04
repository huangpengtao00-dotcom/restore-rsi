"""闸门判据:符号检验,而不是「胜数多于负数」。

reef 自带的 `score_comparison` 用 `wins > losses`。在这套系统上它**假阳性率
50%** —— 与跑多少次重复无关:候选和现版毫无差别时,每一次逐题比较就是一次抛
硬币,而「正面比反面多」的概率恰好是 1/2。实测两次:每题评估 1 次时闸门连续两
步以 2 胜 1 负发布,而事后同一棵树的重复评估显示那些差(0.01–0.19)远小于噪声
(0.15–0.65)。

改用符号检验:把 n 次逐题比较当作 n 次伯努利试验,零假设 p=0.5,只有当胜数达到
`alpha` 对应的临界值时才发布。n=9(3 任务 × 3 重复)时:

    阈值      零假设下达到该值的概率(单侧)
    ≥5(旧规则)      50.0%
    ≥6               25.4%
    ≥7                9.0%
    ≥8                2.0%

默认 alpha=0.10 → n=9 需要 ≥7 胜。

**这不是把闸门调严,是把它的错误率变成已知量。** 旧规则不是「宽松」,是没有定义
过错误率;换成符号检验之后,「发布」这件事第一次有了可以写进论文的含义:在零假设
下这一步被误判的概率 ≤ alpha。

平局(两边分数相同)按符号检验的惯例剔除,不计入 n —— 它们对方向没有信息。
"""

from __future__ import annotations

import logging
from math import comb

log = logging.getLogger("restore.selection")


def _tally(candidate, current) -> tuple[int, int]:
    """(wins, losses),口径与 reef 的 `_score_comparison_tally` 一致:None 视为 -inf。"""
    wins = losses = 0
    for cand, cur in zip(candidate, current, strict=True):
        cand_rank = cand if cand is not None else float("-inf")
        cur_rank = cur if cur is not None else float("-inf")
        if cand_rank > cur_rank:
            wins += 1
        elif cand_rank < cur_rank:
            losses += 1
    return wins, losses


def sign_test_threshold(n: int, alpha: float) -> int:
    """零假设 p=0.5 下,使单侧 P(X >= k) <= alpha 的最小 k。

    n=0 时返回 1,即「没有可比的配对就不可能发布」。
    """
    if n <= 0:
        return 1
    for k in range(n, -1, -1):
        if sum(comb(n, i) for i in range(k, n + 1)) / 2**n > alpha:
            return k + 1
    return 0


def p_value(wins: int, n: int) -> float:
    """单侧 P(X >= wins),零假设 p=0.5。"""
    if n <= 0:
        return 1.0
    return sum(comb(n, i) for i in range(wins, n + 1)) / 2**n


class SignTestSelector:
    """发布当且仅当胜数在符号检验下显著。

    `alpha` 是这一步的假阳性率上界。它是这个类存在的全部理由:让「发布」有一个
    已知的错误率,而不是一个未定义的错误率。
    """

    def __init__(self, alpha: float = 0.10) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        self.alpha = alpha

    def decide(self, candidate, evaluation):
        from reef.train.evaluation.contracts import SelectionDecision  # lazy: 保持 harness 可脱离 reef 导入

        candidate_scores = tuple(evaluation.metrics.get("candidate_scores", ()))
        current_scores = tuple(evaluation.metrics.get("current_scores", ()))
        wins, losses = _tally(candidate_scores, current_scores)
        ties = len(candidate_scores) - wins - losses
        n = wins + losses  # 平局不带方向信息,按惯例剔除
        threshold = sign_test_threshold(n, self.alpha)
        selected = wins >= threshold
        p = p_value(wins, n)
        reason = (
            f"sign test: {wins}/{n} wins (ties {ties} dropped), threshold {threshold} at alpha={self.alpha}, "
            f"one-sided p={p:.3f}"
        )
        log.info("gate: %s -> %s", reason, "select" if selected else "reject")
        return SelectionDecision(
            outcome="select" if selected else "reject",
            policy="sign_test",
            policy_version="1",
            reason=reason,
            evaluation=evaluation,
            metrics={"wins": wins, "losses": losses, "ties": ties, "n": n, "threshold": threshold, "p_value": p},
        )


#: serve.yaml 的 `evolution.selection` 指向它。
selector = SignTestSelector(alpha=0.10)
