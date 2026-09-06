"""量每个可观测信号对「下一步会不会变好」的判别力。

输入是 stopping_signal.py 采的决策点。对每个候选信号算 AUC(1.0=完美,0.5=抛硬币),
并把当前实际在用的规则(诊断读数下降就继续)单独拎出来算它的准确率 —— 因为那正是
被发布的那条 skill 教给 agent 的做法。

    uv run python tasks/stopping_analysis.py work/stopping_d3.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stopping_signal import auc  # noqa: E402  判据只有一份实现

READINGS = ("haze", "low_light", "noise", "blur")


def signals(row: dict) -> dict[str, float]:
    """决策时刻(以及走完这一步之后)能看到的量,全部不含参考图。

    `*_after_*` 这一类是「先走一步,看看观测怎么变,再决定要不要保留」——盲策略
    确实可以这么做:工具是确定性的,走一步再回退不花钱。
    """
    before, after = row["obs"], row["obs_after"]
    return {
        # 决策前就能算的
        "severity_sum(before)": before["severity_sum"],
        "depth": float(row["depth"]),
        "drift_from_input(before)": before["drift_from_input_db"],
        # 试走一步之后能算的
        "severity_sum 下降量": before["severity_sum"] - after["severity_sum"],
        "haze 下降量": before["haze"] - after["haze"],
        "low_light 下降量": before["low_light"] - after["low_light"],
        "noise 下降量": before["noise"] - after["noise"],
        "blur 下降量": before["blur"] - after["blur"],
        "drift 增量(dB)": after["drift_from_input_db"] - before["drift_from_input_db"],
        "severity_sum(after)": -after["severity_sum"],  # 取负:越小越好 → 越大越"预测变好"
    }


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "work/stopping_d3.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    labels = [r["improved"] for r in rows]
    base = sum(labels) / len(labels)
    print(f"决策点 {len(rows)} 个,其中「下一步变好」{sum(labels)} 个({base*100:.1f}%)\n")

    names = list(signals(rows[0]))
    print(f"{'信号':28s}{'AUC':>8s}{'判别力':>10s}")
    scored = []
    for name in names:
        values = [signals(r)[name] for r in rows]
        a = auc(values, labels)
        scored.append((a, name))
    for a, name in sorted(scored, reverse=True):
        verdict = "强" if a >= 0.75 else ("有" if a >= 0.6 else ("弱" if a >= 0.55 else "无(≈抛硬币)"))
        print(f"{name:28s}{a:8.3f}{verdict:>10s}")

    print("\n=== agent 当前实际在用的规则 ===")
    print("被发布的那条 skill 教的是:诊断读数还高就继续修,读数都低于阈值就停。")
    rule = [signals(r)["severity_sum 下降量"] > 0 for r in rows]
    tp = sum(1 for p, y in zip(rule, labels) if p and y)
    fp = sum(1 for p, y in zip(rule, labels) if p and not y)
    fn = sum(1 for p, y in zip(rule, labels) if not p and y)
    tn = sum(1 for p, y in zip(rule, labels) if not p and not y)
    print(f"  「读数下降 → 认为这步是对的」:")
    print(f"     判对 {tp+tn}/{len(rows)} = {(tp+tn)/len(rows)*100:.1f}%(全猜「变好」的基线 {base*100:.1f}%)")
    print(f"     它说该走、实际变差:{fp} 次  ← 这就是把图弄坏的那些步")
    print(f"     它说别走、实际变好:{fn} 次")

    granularity_report(rows)

    print("\n=== 分链长看:越往后越容易走错吗 ===")
    print(f"{'链长':>5s}{'决策点':>8s}{'实际变好占比':>14s}{'读数规则准确率':>16s}")
    for d in sorted({r["depth"] for r in rows}):
        sub = [(r, p) for r, p in zip(rows, rule) if r["depth"] == d]
        ys = [r["improved"] for r, _ in sub]
        acc = sum(1 for (r, p) in sub if p == bool(r["improved"])) / len(sub)
        print(f"{d:5d}{len(sub):8d}{sum(ys)/len(ys)*100:13.1f}%{acc*100:15.1f}%")


def threshold_sweep(rows) -> None:
    """把「读数降了就走」换成「降幅超过 τ 才走」,扫 τ 看精确率能到多少。"""
    labels = [r["improved"] for r in rows]
    drops = [signals(r)["severity_sum 下降量"] for r in rows]
    print("\n=== 把规则改成「诊断总分降幅 > τ 才走」 ===")
    print(f"{'τ':>7s}{'说该走的次数':>13s}{'其中真变好':>12s}{'精确率':>9s}{'召回':>8s}")
    for tau in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        go = [d > tau for d in drops]
        tp = sum(1 for g, y in zip(go, labels) if g and y)
        n_go = sum(go)
        if n_go == 0:
            print(f"{tau:7.2f}{0:13d}{'-':>12s}{'-':>9s}{'-':>8s}")
            continue
        print(f"{tau:7.2f}{n_go:13d}{tp:12d}{tp/n_go*100:8.1f}%{tp/sum(labels)*100:7.1f}%")
    print("  (τ=0 就是当前规则)")


def greedy_simulation(depth: int, tau_values, tasks=None) -> None:
    """端到端:贪心策略在不同 τ 下能拿到多少分,对照完美停手(oracle)。

    策略:每一步试走全部 6 个工具,选诊断总分降幅最大的那个;降幅不超过 τ 就停手。
    这正是被发布的那条 skill 描述的做法(τ=0),只是把「读数降了就走」参数化了。
    """
    import yaml
    from PIL import Image
    import numpy as np
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from toolbox import builtin
    from toolbox.cli import _psnr, diagnose_array, reference_score

    registry = yaml.safe_load((root / "toolbox" / "registry.yaml").read_text(encoding="utf-8"))["tools"]
    tools = {n: getattr(builtin, sp["backend"].split(":", 1)[1])
             for n, sp in registry.items() if sp["backend"].startswith("builtin:")}
    manifest = json.loads((root / "tasks" / "data" / "manifest.json").read_text(encoding="utf-8"))
    if tasks:
        manifest = [t for t in manifest if t["task_id"] in tasks]

    def load(p_): return np.asarray(Image.open(p_).convert("RGB"))

    print(f"\n=== 端到端:贪心策略(试走 6 个工具,选诊断降幅最大的;降幅 <= τ 就停)===")
    header = f"{'任务':22s}" + "".join(f"{'τ=' + str(t):>10s}" for t in tau_values) + f"{'完美停手':>10s}"
    print(header)
    totals = {t: 0.0 for t in tau_values}; oracle_total = 0.0
    for task in manifest:
        frame = task["frames"][0]
        inp, ref = load(frame["input"]), load(frame["reference"])
        psnr_in = _psnr(inp, ref)
        row = f"{task['task_id']:22s}"
        best_seen = 0.0
        for tau in tau_values:
            img, score = inp, 0.0
            for _ in range(depth):
                cur_sum = sum(diagnose_array(img)["degradations"].values())
                best = None
                for name, fn in tools.items():
                    try: nxt = fn(img)
                    except Exception: continue
                    drop = cur_sum - sum(diagnose_array(nxt)["degradations"].values())
                    if best is None or drop > best[0]: best = (drop, nxt)
                if best is None or best[0] <= tau: break
                img = best[1]
                score = reference_score(_psnr(img, ref), psnr_in)
                best_seen = max(best_seen, score)
            totals[tau] += score
            row += f"{score:10.4f}"
        # 完美停手 = 同一条贪心轨迹上出现过的最高分(agent 走过但不知道该在哪停)
        oracle_total += best_seen
        row += f"{best_seen:10.4f}"
        print(row)
    print(f"{'合计':22s}" + "".join(f"{totals[t]:10.4f}" for t in tau_values) + f"{oracle_total:10.4f}")
    print("  「完美停手」= 同一条贪心轨迹上出现过的最高分。它和 τ 那几列的差,")
    print("  就是「走过最优点却不知道该停」白丢的分 —— 不需要更好的工具,只需要知道何时收手。")





# --------------------------------------------------------------- 决策粒度

READINGS_4 = ("haze", "low_light", "noise", "blur")


def severity_drop(row: dict) -> float:
    """走这一步之后,诊断总分降了多少(agent 看得见的那个量)。"""
    before = sum(row["obs"][k] for k in READINGS_4)
    after = sum(row["obs_after"][k] for k in READINGS_4)
    return before - after


def by_decision_group(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """按 (任务, 当前链) 分组。一组 = 一次真实决策,组内是六个候选工具。"""
    from collections import defaultdict

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["task_id"], row["chain"])].append(row)
    return groups


def granularity_report(rows: list[dict]) -> None:
    """同一个信号在两个粒度上差多少 —— 以及为什么只有一个是「停手」。

    2026-09-06 的修正。此前报告的「最好的盲信号 AUC 0.789」是在**逐行**数据上算的:
    每个 (状态, 候选工具) 一行,问「这个候选会不会改善」。那道题里最差的工具只有
    14.5% 会改善、最好的有 68.8%,跨度极大,信号轻松把两群分开 —— 它测的是
    **选择**(哪个工具好),不是**停手**(该不该再走一步)。

    真实的停手决策发生在选完之后。按同样的规则选出最佳工具再问「走它会不会改善」,
    同一个信号从 0.789 掉到 0.545,而基线是 68.8% —— 几乎没有预测能力。

    结论要改口径:**不是「有一个 AUC 0.789 的信号但还不够好」,而是「在正确的决策
    粒度上,现有的无参考信号全都接近随机」**。停手问题比原先描述的更难,那 23% 的
    可达分数也更难拿回来。

    方法学教训:换一个决策粒度,同一个数就在回答另一个问题。报 AUC 之前先说清楚
    判别的是什么。
    """
    all_labels = [r["improved"] for r in rows]
    per_row = auc([severity_drop(r) for r in rows], all_labels)

    groups = by_decision_group(rows)
    best = [max(members, key=severity_drop) for members in groups.values()]
    worst = [min(members, key=severity_drop) for members in groups.values()]
    per_group = auc([severity_drop(r) for r in best], [r["improved"] for r in best])
    worst_auc = auc([severity_drop(r) for r in worst], [r["improved"] for r in worst])

    print("\n=== 决策粒度:同一个信号,两个问题 ===")
    print(f"{'问题':46s}{'样本':>7s}{'基线':>8s}{'AUC':>8s}")
    for label, sample, a in (
        ("A 任意一个候选工具会不会改善(选择问题)", rows, per_row),
        ("B 已选出的最佳工具,走不走(停手问题)", best, per_group),
        ("  对照:每组最差的工具", worst, worst_auc),
    ):
        base = sum(r["improved"] for r in sample) / len(sample)
        print(f"{label:46s}{len(sample):>7d}{base*100:>7.1f}%{a:>8.3f}")
    print("  A 的高 AUC 来自把好工具和坏工具分开,那不是停手决策要回答的问题。")
    print("  真实决策在选完之后,而在那个粒度上信号接近抛硬币。")


if __name__ == "__main__":
    main()
