"""闸门可行域:一个判据能不能提交,和它的保证是否成立,是两件事。

    uv run python tasks/gate_feasibility.py

背景:PACE(arXiv 2606.08106)把 greedy acceptance 定性为未受控的自适应多重检验,
用 testing-by-betting 的 e-process 取代它 —— E<-E*(1+λ(2w-1)),E>=1/α 即提交,
由 Ville 不等式在**任意停时**控制假提交率。零假设和这个仓用的符号检验完全一样
(配对、丢平局、H0 下不一致对各半),差别只在序贯 vs 固定 n。

这个脚本回答一个 PACE 自己回答不了的问题:**在小样本的自进化循环里,那个闸门
还能不能触发?** α=0.05、λ=0.5 时阈值是 20,而 1.5^8=25.6 —— 必须先连赢 8 次。
这个仓每步只有 7~9 个不一致对。

它不是 PACE 的缺陷:原文实验跑在 GSM8K/SVAMP/ARC 上,dev set 够大,不一致对能
攒到几十上百。它说明的是**保证成立 ≠ 在你的规模上能用**,而这一点只有把闸门放到
一个小循环里真跑才看得见。

纯计算,不需要跑 episode。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reef_example"))

from harness.selection import EProcessSelector, sign_test_threshold  # noqa: E402


def min_consecutive_wins(alpha: float, bet: float, cap: int = 500) -> int | None:
    """全胜情况下,最少要几个不一致对才能越过 1/alpha。"""
    sel = EProcessSelector(alpha=alpha, bet=bet)
    for n in range(1, cap + 1):
        _, _, crossed = sel.wealth_path((1,) * n, (0,) * n)
        if crossed > 0:
            return crossed
    return None


def min_wins_at(n: int, alpha: float, bet: float) -> int | None:
    """n 个不一致对时,最少要几胜才可能提交(取最有利的顺序:先赢后输)。

    取最有利顺序是因为 anytime-valid 判的是 `sup_t E_t`,只要中途越过就算 ——
    所以"能不能提交"要按最好情况算,不是按终值。
    """
    sel = EProcessSelector(alpha=alpha, bet=bet)
    for k in range(n + 1):
        _, _, crossed = sel.wealth_path((1,) * k + (0,) * (n - k), (0,) * k + (1,) * (n - k))
        if crossed > 0:
            return k
    return None


def main() -> None:
    print("一、λ 决定最少连胜数(α=0.05,阈值 20)\n")
    print(f"{'λ':>6}{'单次赢 ×':>10}{'最少连胜':>10}")
    for bet in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        need = min_consecutive_wins(0.05, bet)
        note = "  <- PACE 原文默认" if bet == 0.5 else ""
        print(f"{bet:>6}{1 + bet:>10.2f}{need if need else '>500':>10}{note}")

    print("\n二、两种闸门的严格程度随 n **反转**\n")
    print(f"{'不一致对 n':>12}{'符号检验 α=0.10':>18}{'e-process α=0.05 λ=0.5':>26}{'谁更严':>10}")
    for n in (5, 7, 8, 9, 10, 12, 15, 18, 30):
        st = sign_test_threshold(n, 0.10)
        st_txt = f">={st}" if st <= n else "不可能"
        ep = min_wins_at(n, 0.05, 0.5)
        ep_txt = f">={ep}" if ep is not None else "不可能"
        if ep is None:
            who = "e-process"
        elif st > n:
            who = "符号检验"
        else:
            who = "e-process" if ep > st else ("符号检验" if st > ep else "相同")
        print(f"{n:>12}{st_txt:>18}{ep_txt:>26}{who:>10}")

    print("""
读法:交叉点在 n≈10–12。

  小循环(n<10)   e-process 几乎不可能提交 —— 阈值要 8 连胜,而总共只有 7~9 对
  大 dev set(n>12) e-process 反而更松 —— 它只要 8 连胜,不管总共评了多少对

后者正是 anytime-valid 的设计意图(证据够了就停,别浪费评估),但在小循环里
它变成了一道几乎迈不过的门槛。**保证成立和在你的规模上能用,是两件事。**

要选 λ 就得知道真实效应有多大 —— 而"真实效应"需要天花板才能量,那是这个域
能做、文本域做不到的事。""")


if __name__ == "__main__":
    main()
