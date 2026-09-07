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


def _measurable(value) -> bool:
    """这一次评估到底测出东西了没有。

    `None` 是 reef 在 episode 根本没跑起来时给的(`_run_and_score` 捕获 EpisodeError /
    TrajectoryError 的那两支),`NaN` 是消融的 exclude 臂。两者都不是成绩。
    """
    import math

    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _tally(candidate, current) -> tuple[int, int, int]:
    """(wins, losses, excluded)。**测不出来的对子不参与胜负。**

    这里和 reef 的 `_score_comparison_tally` 口径**不同**,是有意的。reef 把 `None`
    当 `-inf`,于是"这个 episode 没启动起来"被算成"它比对面差",白送对面一个 loss。
    那跟策略好坏毫无关系 —— 容器没起来、pi 崩了、机器卡了,都会变成闸门的证据。

    这正是这个仓一直在追的那个病:基础设施的失效伪装成算法效果。照抄别人的口径
    而没问它对不对,就把同一个坑抄进来了(2026-09-06 才发现,当时已经用了三天)。

    排除之后 n 变小,符号检验会自动变严 —— n=2 时临界值是 3,怎么都不显著。
    那是对的:证据少了,就该更难发布,而不是拿剩下的几个凑一个结论。
    """
    wins = losses = excluded = 0
    for cand, cur in zip(candidate, current, strict=True):
        if not (_measurable(cand) and _measurable(cur)):
            excluded += 1
            continue
        if cand > cur:
            wins += 1
        elif cand < cur:
            losses += 1
    return wins, losses, excluded


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
        wins, losses, excluded = _tally(candidate_scores, current_scores)
        ties = len(candidate_scores) - wins - losses - excluded
        n = wins + losses  # 平局不带方向信息,按惯例剔除
        threshold = sign_test_threshold(n, self.alpha)
        selected = wins >= threshold
        p = p_value(wins, n)
        reason = (
            f"sign test: {wins}/{n} wins (ties {ties} dropped, {excluded} unmeasurable), "
            f"threshold {threshold} at alpha={self.alpha}, one-sided p={p:.3f}"
        )
        if excluded:
            # 一定要说出来:排除了几对,直接决定这一步的证据有多少
            log.warning("gate: %d/%d 对无法测量(episode 没跑起来,或消融的 exclude 臂)",
                        excluded, len(candidate_scores))
        log.info("gate: %s -> %s", reason, "select" if selected else "reject")
        return SelectionDecision(
            outcome="select" if selected else "reject",
            policy="sign_test",
            policy_version="1",
            reason=reason,
            evaluation=evaluation,
            metrics={"wins": wins, "losses": losses, "ties": ties, "excluded": excluded,
                     "n": n, "threshold": threshold, "p_value": p},
        )


class EProcessSelector:
    """PACE 的 testing-by-betting 闸门(arXiv 2606.08106),按原文实现。

    **零假设与符号检验完全相同**:配对比较,平局丢弃,不一致对在 H0 下各占一半
    (McNemar 式)。差别在判定方式 —— 符号检验是固定 n 的批量判定,e-process 是
    序贯的:每来一个不一致对就下一次注,财富越过 1/alpha 就提交。

        E_0 = 1
        E  <- E * (1 + lambda * (2*w_i - 1))          原文式 (2),lambda in [0,1)
        commit  当  E >= 1/alpha

    H0 下 E[2w-1] = 0,E 是非负上鞅且 E[E_i] <= 1,由 Ville 不等式
    Pr[sup_t E_t >= 1/alpha] <= alpha —— **在任意停时都成立**,这正是"anytime-valid"
    的含义:可以边评边看,证据够了立刻停,而不必预先定 n。原文默认 alpha=0.05、
    lambda=0.5。

    **一个诚实的范围限制**(原文自己写明):保证是 per-candidate 的,不是 run 级的
    familywise —— 跑很多个不改进的候选,期望假提交数按每候选 alpha 累加。

    **在这个仓里的实现差异,必须说清楚**:reef 是批量跑完所有 episode 再判,所以
    这里只能**按评估顺序事后重放**那个序贯过程。错误率保证照样成立(Ville 不等式
    对任何停时都成立),但 anytime-valid 最大的实际好处 —— 提前停下来省评估 ——
    在这个架构里拿不到。要拿到得让 evaluate 能增量返回,那是 reef 侧的改动。
    """

    def __init__(self, alpha: float = 0.05, bet: float = 0.5) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if not 0.0 <= bet < 1.0:
            raise ValueError(f"lambda must be in [0, 1), got {bet!r}")
        self.alpha = alpha
        self.bet = bet

    def wealth_path(self, candidate, current) -> tuple[float, list[float], int]:
        """(最终财富, 每步财富, 首次越过阈值的位置或 -1)。

        返回整条路径而不只是终值:anytime-valid 的判定是 `sup_t E_t >= 1/alpha`,
        也就是**只要中途越过就算**,不是看最后停在哪 —— 财富可以先冲高再跌回来。
        """
        threshold = 1.0 / self.alpha
        wealth, path, crossed = 1.0, [], -1
        index = 0
        for cand, cur in zip(candidate, current, strict=True):
            if not (_measurable(cand) and _measurable(cur)):
                continue          # 测不出来的不参与,同 _tally
            if cand == cur:
                continue          # 平局丢弃(McNemar)
            w = 1 if cand > cur else 0
            wealth *= 1.0 + self.bet * (2 * w - 1)
            path.append(wealth)
            index += 1
            if crossed < 0 and wealth >= threshold:
                crossed = index
        return wealth, path, crossed

    def decide(self, candidate, evaluation):
        from reef.train.evaluation.contracts import SelectionDecision  # lazy

        candidate_scores = tuple(evaluation.metrics.get("candidate_scores", ()))
        current_scores = tuple(evaluation.metrics.get("current_scores", ()))
        final, path, crossed = self.wealth_path(candidate_scores, current_scores)
        peak = max(path, default=1.0)
        threshold = 1.0 / self.alpha
        selected = crossed > 0
        wins, losses, excluded = _tally(candidate_scores, current_scores)
        reason = (
            f"e-process (PACE): wealth peak {peak:.3f} vs threshold {threshold:.1f} "
            f"(lambda={self.bet}, alpha={self.alpha}); {wins}/{wins + losses} discordant wins, "
            f"{excluded} unmeasurable; " + (f"crossed at pair {crossed}" if crossed > 0 else "never crossed")
        )
        log.info("gate: %s -> %s", reason, "select" if selected else "reject")
        return SelectionDecision(
            outcome="select" if selected else "reject",
            policy="e_process",
            policy_version="pace-2606.08106",
            reason=reason,
            evaluation=evaluation,
            metrics={
                "wins": wins, "losses": losses, "excluded": excluded, "n": wins + losses,
                "wealth_final": final, "wealth_peak": peak, "threshold": threshold,
                "crossed_at": crossed,
                # 同一批数据在符号检验下的结论,便于直接对照两种判据
                "p_value": p_value(wins, wins + losses),
            },
        )


class WinsOverLossesSelector:
    """reef 自带的规则:胜数 > 负数就发布。**只为消融对照存在。**

    候选与现版毫无差别时,每次逐题比较就是一次抛硬币,而"正面比反面多"的概率
    恰好是 1/2 —— 所以这条规则的假阳性率是 50%,与重复多少次无关。它不是"宽松",
    是从未定义过错误率。

    独立实现一份而不是去 import reef 的类,有两个理由:口径已经在 `_tally` 里
    复刻过(None 视为 -inf),而且这样测试不需要装 reef 就能跑对照。
    """

    def decide(self, candidate, evaluation):
        from reef.train.evaluation.contracts import SelectionDecision  # lazy

        candidate_scores = tuple(evaluation.metrics.get("candidate_scores", ()))
        current_scores = tuple(evaluation.metrics.get("current_scores", ()))
        wins, losses, excluded = _tally(candidate_scores, current_scores)
        selected = wins > losses
        reason = f"wins>losses: {wins} vs {losses} (ablation arm; false-positive rate 50% by construction)"
        log.info("gate[ablation]: %s -> %s", reason, "select" if selected else "reject")
        return SelectionDecision(
            outcome="select" if selected else "reject",
            policy="wins_over_losses",
            policy_version="ablation",
            reason=reason,
            evaluation=evaluation,
            # 刻意也报 p 值:同一份数据在受控判据下**本来**是什么结论,要能直接对照。
            metrics={
                "wins": wins, "losses": losses, "excluded": excluded, "n": wins + losses,
                "p_value": p_value(wins, wins + losses),
            },
        )


def build_selector():
    """按消融配置选闸门。默认 sign_test。

    在 import 时求值一次(serve.yaml 指向下面的 `selector`),所以一次实验只有一个
    配置 —— 那正是想要的:一份结果对应一个明确的闸门,不会跑到一半换了判据。
    """
    from .ablation import current

    gate = current().gate
    if gate == "wins_over_losses":
        log.warning("gate: 消融臂 wins_over_losses(假阳性率 50%),不是默认判据")
        return WinsOverLossesSelector()
    if gate == "e_process":
        log.info("gate: e-process(PACE 2606.08106),alpha=0.05 lambda=0.5")
        return EProcessSelector()
    return SignTestSelector(alpha=0.10)


#: serve.yaml 的 `evolution.selection` 指向它。
selector = build_selector()
