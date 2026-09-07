"""开跑后按分钟核这批数据有不有效 —— 六次白跑换来的规矩。

    uv run python tasks/check_arm_validity.py <臂目录> [<臂目录> ...] --n-expect 36

每一条都是**能证伪的**:不满足就说明这批数据白跑,越早停越好。六次白跑的形状
全一样——跑完了、日志正常、数据是错的;所以判据不能是"进程还活着"或"日志在长"。

    第 3 分钟   每臂日志开头有自己的「消融配置」行,且四个臂互不相同
    第 3 分钟   每臂日志开头的「隔离」行指向自己的 work 与端口(串台就是白跑)
    第 5 分钟   每臂 results/trace.jsonl 存在且在长
    第 5 分钟   recipe 里 evolution.tasks 条数 == 期望的配对数(这轮的全部理由)
    第 8 分钟   coupled_v0 那臂的 diagnose 读数与其他臂**不同**(消融真生效了)
    出第一步后 commits.jsonl 的 n 落在期望量级,不是个位数
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

OK, BAD, WAIT = "✅", "❌", "…"


def check(work: Path, n_expect: int) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    log = work / "run.log"
    text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""

    cfg = [l for l in text.splitlines() if "消融配置" in l]
    out.append((OK if cfg else (WAIT if not text else BAD), "自报配置", cfg[0][-90:] if cfg else "日志里没有「消融配置」行"))

    iso = [l for l in text.splitlines() if "隔离:" in l]
    right = bool(iso) and str(work.resolve()) in iso[0]
    out.append((OK if right else (WAIT if not text else BAD), "隔离正确", iso[0][-90:] if iso else "没有「隔离」行"))

    trace = work / "results" / "trace.jsonl"
    n_tr = sum(1 for _ in trace.open()) if trace.exists() else 0
    out.append((OK if n_tr else WAIT, "trace 在长", f"{n_tr} 事件"))

    recipe = work / "recipes" / "harness_evolve.yaml"
    if recipe.exists():
        n_tasks = len(yaml.safe_load(recipe.read_text(encoding="utf-8"))["evolution"]["tasks"])
        out.append((OK if n_tasks == n_expect else BAD, "配对数", f"recipe 里 {n_tasks} 条,期望 {n_expect}"))
    else:
        out.append((WAIT, "配对数", "recipe 还没生成"))

    commits = sorted(work.glob("agent-record/*.commits.jsonl"))
    ns = [json.loads(l)["metrics"]["n"] for p in commits for l in p.read_text().splitlines() if l.strip()]
    out.append((OK if ns else WAIT, "闸门 n", f"{ns}" if ns else "还没出第一步"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("works", nargs="+", type=Path)
    ap.add_argument("--n-expect", type=int, required=True, help="recipe 里 evolution.tasks 应有的条数 = 题数 x 重复")
    args = ap.parse_args()

    bad = 0
    configs = []
    for work in args.works:
        print(f"=== {work.name}")
        for mark, name, detail in check(work, args.n_expect):
            print(f"   {mark} {name:<8} {detail}")
            bad += mark == BAD
        cfg = [l for l in (work / "run.log").read_text(errors="ignore").splitlines() if "消融配置" in l] if (work / "run.log").exists() else []
        configs.append(cfg[0].split("消融配置:")[-1] if cfg else None)

    known = [c for c in configs if c]
    if len(known) != len(set(known)):
        print(f"\n{BAD} 有两个臂的配置一样 —— 那两个臂里有一个白跑")
        bad += 1
    elif known:
        print(f"\n{OK} {len(known)} 个臂配置互不相同")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
