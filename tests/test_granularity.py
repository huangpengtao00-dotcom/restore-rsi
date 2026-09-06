"""决策粒度:同一个信号在两个粒度上回答的不是同一个问题。

2026-09-06 的修正。此前报告的「最好的盲信号 AUC 0.789」是在**逐行**数据上算的 ——
每个 (状态, 候选工具) 一行,问「这个候选会不会改善」。那道题里最差的工具只有 14.5%
会改善、最好的有 68.8%,跨度极大,信号轻松把两群分开:它测的是**选择**(哪个工具好),
不是**停手**(该不该再走一步)。

真实的停手决策发生在选完之后。同样的规则选出最佳工具,再问「走它会不会改善」,
同一个信号从 0.789 掉到 0.545,而基线就有 68.8%。

所以结论要改口径:不是「有个 AUC 0.789 的信号但还不够好」,而是**在正确的决策粒度上,
现有的无参考信号全都接近随机**。这些数字钉在这里,因为它们是论文里要写的数,而且
其中一个已经被我自己引用错过一次。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

DATA = ROOT / "work" / "stopping_d3.json"
pytestmark = pytest.mark.skipif(
    not DATA.exists(),
    reason="work/stopping_d3.json 未生成:先跑 tasks/stopping_signal.py --depth 3",
)


def _rows():
    return json.loads(DATA.read_text(encoding="utf-8"))["rows"]


def test_the_two_granularities_are_different_questions():
    """逐行和组级的差必须存在且明显 —— 这个差本身就是那条方法学结论。"""
    from stopping_analysis import by_decision_group, severity_drop
    from stopping_signal import auc

    rows = _rows()
    per_row = auc([severity_drop(r) for r in rows], [r["improved"] for r in rows])
    best = [max(m, key=severity_drop) for m in by_decision_group(rows).values()]
    per_group = auc([severity_drop(r) for r in best], [r["improved"] for r in best])

    assert per_row == pytest.approx(0.789, abs=0.01), f"逐行 AUC 变了:{per_row:.3f}"
    assert per_group == pytest.approx(0.545, abs=0.02), f"组级 AUC 变了:{per_group:.3f}"
    assert per_row - per_group > 0.2, "两个粒度的差消失了,那条方法学结论就不成立了"


def test_the_stopping_signal_is_close_to_a_coin_flip():
    """停手粒度上的判别力必须被如实描述为「接近随机」。

    这条存在的理由是防止把 0.789 再写回停手的语境里 —— 我自己已经写错过一次。
    """
    from stopping_analysis import by_decision_group, severity_drop
    from stopping_signal import auc

    best = [max(m, key=severity_drop) for m in by_decision_group(_rows()).values()]
    per_group = auc([severity_drop(r) for r in best], [r["improved"] for r in best])
    assert per_group < 0.60, f"停手信号 AUC {per_group:.3f};高于 0.60 就该重新描述结论"


def test_every_decision_group_has_all_six_tools():
    """组级判据的前提:每组都是完整的六个候选。少了就不是「选最好的」那个决策。"""
    from stopping_analysis import by_decision_group

    sizes = {len(m) for m in by_decision_group(_rows()).values()}
    assert sizes == {6}, f"组大小不一致:{sizes}"
