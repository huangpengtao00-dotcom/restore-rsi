"""盲策略靠什么信号知道该收手 —— 有真值的判定实验。

问题是实测出来的:盲评协议把参考图和打分收到判分方,堵住了泄题,但 agent 因此
无法知道自己上一步把图弄坏了。实测同一棵树上

    dehaze_dcp > lowlight_gamma                    ->  0.2165 / 0.4158
    dehaze_dcp > lowlight_gamma > lowlight_clahe   ->  0.0000

多走一步就归零,而 agent 手上只有诊断读数。所以要问的是一个可以判定的问题:

    **给定 agent 在这一步能看到的东西,能不能预测「下一步会让真实分数变好还是变坏」?**

这个问题有真值(合成任务的干净原图),动作空间可枚举(6 个确定性工具),所以可以
穷举所有链的每一步,把「观测」和「真实结果」配对,直接量每个候选信号的判别力。

对每个链前缀 s 和每个可选工具 a,记录:
  观测(agent 看得见)  诊断四读数、它们的和、链长、已偏离输入多远(PSNR vs 输入)
  真值(agent 看不见)  score(s·a) - score(s) > 0 ?

然后对每个候选信号算 AUC:1.0 = 完美预测,0.5 = 和抛硬币一样。

    uv run python tasks/stopping_signal.py --depth 3 --out work/stopping.json

纯 CPU,不碰 reef 也不用模型。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def _load(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def auc(scores: list[float], labels: list[int]) -> float:
    """P(随机一个正例的分数 > 随机一个负例的分数),平局算半个。Mann-Whitney U。"""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):  # 平均秩处理并列
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def collect(task: dict, tools: dict, depth: int) -> list[dict]:
    """穷举 ≤depth 的链,对每个 (前缀, 下一步) 记录观测与真实结果。"""
    from toolbox.cli import _psnr, diagnose_array, reference_score

    frame = task["frames"][0]
    inp, ref = _load(frame["input"]), _load(frame["reference"])
    psnr_in = _psnr(inp, ref)
    rows: list[dict] = []

    def observe(img: np.ndarray) -> dict:
        d = diagnose_array(img)["degradations"]
        return {
            **d,
            "severity_sum": round(sum(d.values()), 4),
            # 已经偏离输入多远。过度处理的直接症状就是走得太远,而这个量
            # agent 完全看得见 —— 它有输入图,也有当前图。
            "drift_from_input_db": round(_psnr(img, inp), 3),
        }

    def walk(img: np.ndarray, chain: list[str], obs: dict, score: float) -> None:
        if len(chain) >= depth:
            return
        for name, fn in tools.items():
            try:
                nxt = fn(img)
            except Exception as exc:  # noqa: BLE001 - 工具抛异常不是"链",但不能当成 0 分静默跳过
                print(f"  ! {'>'.join([*chain, name])} raised {exc!r}")
                continue
            nxt_score = reference_score(_psnr(nxt, ref), psnr_in)
            nxt_obs = observe(nxt)
            rows.append(
                {
                    "task_id": task["task_id"],
                    "chain": ">".join(chain) or "(input)",
                    "depth": len(chain),
                    "next_tool": name,
                    "obs": obs,                       # 决策时刻看得见的
                    "obs_after": nxt_obs,             # 走完这一步之后看得见的
                    "score": round(score, 4),
                    "next_score": round(nxt_score, 4),
                    "delta": round(nxt_score - score, 4),
                    "improved": int(nxt_score > score),
                }
            )
            walk(nxt, [*chain, name], nxt_obs, nxt_score)

    walk(inp, [], observe(inp), reference_score(psnr_in, psnr_in))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--manifest", default=str(ROOT / "tasks" / "data" / "manifest.json"))
    parser.add_argument("--out", default=str(ROOT / "work" / "stopping.json"))
    parser.add_argument("--task", action="append")
    args = parser.parse_args()

    from toolbox import builtin

    registry = yaml.safe_load((ROOT / "toolbox" / "registry.yaml").read_text())["tools"]
    tools = {
        name: getattr(builtin, spec["backend"].split(":", 1)[1])
        for name, spec in registry.items()
        if spec["backend"].startswith("builtin:")
    }

    manifest = json.loads(Path(args.manifest).read_text())
    wanted = set(args.task) if args.task else None
    rows: list[dict] = []
    for task in manifest:
        if wanted is not None and task["task_id"] not in wanted:
            continue
        t0 = time.monotonic()
        got = collect(task, tools, args.depth)
        rows.extend(got)
        print(f"{task['task_id']:22s} {len(got):6d} 个决策点  ({time.monotonic() - t0:.0f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"depth": args.depth, "rows": rows}, indent=1) + "\n")
    print(f"\n共 {len(rows)} 个决策点,{sum(r['improved'] for r in rows)} 个是「下一步变好」-> {out}")


if __name__ == "__main__":
    main()
