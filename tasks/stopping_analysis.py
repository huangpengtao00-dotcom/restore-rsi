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



if __name__ == "__main__":
    main()
