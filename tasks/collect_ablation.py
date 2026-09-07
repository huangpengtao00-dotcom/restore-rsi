"""把各消融臂的闸门判决收成一张表:每个缺陷值多少可达分数。

    uv run python tasks/collect_ablation.py <臂目录> [<臂目录> ...]

每个臂目录下要有 agent-record/*.commits.jsonl —— reef 每跑完一个 evolve step
就往里追加一行,里面有这一步的胜负平、p 值、两边每道题的得分。

**不改 reef 写下的数,只做换算。** 判据(sign_test / wins_over_losses)原样从
harness.selection 里调,不在这里重推 —— 自写判据等于验自己的假设。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reef_example"))

from harness.selection import p_value, sign_test_threshold  # noqa: E402


def load_arm(work: Path) -> dict:
    records = sorted(work.glob("agent-record/*.commits.jsonl"))
    steps = []
    for path in records:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            m = json.loads(line)["metrics"]
            ev = m.get("selection", {}).get("evaluation", {}).get("metrics", {})
            steps.append(
                {
                    "wins": m["wins"],
                    "losses": m["losses"],
                    "excluded": m.get("excluded", 0),
                    "n": m["n"],
                    "p_value": m.get("p_value"),
                    "selected": m["selected"],
                    "published": m.get("published", False),
                    "candidate_score": m["candidate_score"],
                    "current_score": m["current_score"],
                    "candidate_scores": ev.get("candidate_scores", []),
                    "current_scores": ev.get("current_scores", []),
                }
            )
    return {"name": work.name, "steps": steps}


def counterfactual(step: dict) -> dict:
    """同一份胜负,两个闸门各自会怎么判。

    reef 记下的 wins/losses 已经是这个臂自己那把闸门数出来的 —— wins_over_losses
    不丢平局,所以它的 n 里含平局,两边的 n 本来就不可比。这里只报**在这份数上**
    各自的判决,不假装能互换。
    """
    wins, losses = step["wins"], step["losses"]
    n = wins + losses
    return {
        "wins_over_losses": wins > losses,
        "sign_test": n > 0 and wins >= sign_test_threshold(n, 0.10),
        "p": p_value(wins, n) if n else None,
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    arms = [load_arm(Path(a)) for a in sys.argv[1:]]

    print(f"{'臂':<12} {'步':>3} {'发布':>4} {'胜-负-弃':>10} {'p':>7} {'旧闸':>5} {'符号':>5}  候选/现版总分")
    for arm in arms:
        for i, s in enumerate(arm["steps"], 1):
            cf = counterfactual(s)
            print(
                f"{arm['name'] if i == 1 else '':<12} {i:>3} "
                f"{'是' if s['published'] else '否':>4} "
                f"{s['wins']}-{s['losses']}-{s['excluded']:<6} "
                f"{(s['p_value'] if s['p_value'] is not None else float('nan')):>7.3f} "
                f"{'发' if cf['wins_over_losses'] else '拒':>5} "
                f"{'发' if cf['sign_test'] else '拒':>5}  "
                f"{s['candidate_score']:.3f}/{s['current_score']:.3f}"
            )

    print("\n=== 两把闸门在同一份数上分歧几次 ===")
    disagree = total = 0
    for arm in arms:
        for s in arm["steps"]:
            cf = counterfactual(s)
            total += 1
            disagree += cf["wins_over_losses"] != cf["sign_test"]
    print(f"{total} 步中分歧 {disagree} 步")

    print("\n=== 每题得分分布(候选 vs 现版,合并所有步) ===")
    for arm in arms:
        cand = [x for s in arm["steps"] for x in s["candidate_scores"]]
        curr = [x for s in arm["steps"] for x in s["current_scores"]]
        if not cand:
            continue
        zeros_c = sum(1 for x in cand if x == 0.0)
        zeros_k = sum(1 for x in curr if x == 0.0)
        print(
            f"{arm['name']:<12} n={len(cand):>3}  候选均值 {sum(cand)/len(cand):.4f} "
            f"(0分 {zeros_c})   现版均值 {sum(curr)/len(curr):.4f} (0分 {zeros_k})"
        )


if __name__ == "__main__":
    main()
