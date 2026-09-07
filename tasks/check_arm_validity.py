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
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

OK, BAD, WAIT = "✅", "❌", "…"


def sleep_during(work: Path) -> tuple[int, int, str]:
    """(睡眠秒数, 次数, 说明) —— 这一臂跑的那段时间里,这台机器睡了多久。

    2026-09-07 实测:n=36 那轮 148 分钟里机器睡了 73 分钟(占 50%),其中最长
    一段 989s 是 Clamshell Sleep(合上盖子)。睡眠期间进程冻结、不发请求、
    **也就不会在任何日志里留下痕迹** —— 时间白流,而 run.log 只是看起来慢。

    真正的伤害不在慢:睡下去的那一刻有在途请求就会断。同一轮里 propose 因此
    连挂三次,落盘成 {"skipped": "no proposal"}(见 evidence/2026-09-07-dns-blip-as-no-proposal)。

    所以每份有效性报告都要带这个数。判据取自 pmset,不是猜:`Entering Sleep state`
    行末尾的 `N secs` 就是该次睡眠时长(用 19:03:41 的 15s、19:04:41 的 65s 与
    随后的 DarkWake 时间戳交叉核对过)。
    """
    log = work / "run.log"
    if not log.exists():
        return 0, 0, "没有 run.log"
    lines = [l for l in log.read_text(encoding="utf-8", errors="ignore").splitlines() if l[:2].isdigit()]
    if not lines:
        return 0, 0, "run.log 里没有带时间戳的行"
    day = datetime.date.fromtimestamp(log.stat().st_mtime)
    def stamp(line: str) -> datetime.datetime:
        return datetime.datetime.combine(day, datetime.time.fromisoformat(line[:8]))
    lo, hi = stamp(lines[0]), stamp(lines[-1])
    if hi < lo:                       # 跨午夜
        hi += datetime.timedelta(days=1)
    try:
        out = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return 0, 0, f"读不到 pmset:{exc}"     # 探测失败不等于「没睡」,如实说
    total = count = 0
    for line in out.splitlines():
        m = re.match(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \S+ Sleep\s+Entering Sleep state.*?(\d+) secs\s*$", line)
        if not m:
            continue
        when = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        if lo <= when <= hi:
            total += int(m.group(2))
            count += 1
    span = max(1, int((hi - lo).total_seconds()))
    return total, count, f"{total}s / 跨度 {span}s = {total / span * 100:.0f}%"


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
    n_tr = sum(1 for _ in trace.open(encoding="utf-8")) if trace.exists() else 0
    out.append((OK if n_tr else WAIT, "trace 在长", f"{n_tr} 事件"))

    recipe = work / "recipes" / "harness_evolve.yaml"
    if recipe.exists():
        n_tasks = len(yaml.safe_load(recipe.read_text(encoding="utf-8"))["evolution"]["tasks"])
        out.append((OK if n_tasks == n_expect else BAD, "配对数", f"recipe 里 {n_tasks} 条,期望 {n_expect}"))
    else:
        out.append((WAIT, "配对数", "recipe 还没生成"))

    commits = sorted(work.glob("agent-record/*.commits.jsonl"))
    steps = [json.loads(l)["metrics"] for p in commits for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    # 一次 DNS 抽风把 n36-mf 三步全烧了,每步落盘成 {"skipped": "no proposal"} ——
    # 步数照涨、日志无异常、读起来像"模型这轮没想到改动"这个合法结果。
    # episode 的推理调用有 4 次重试,propose 没有,一次失败就跳过。
    # 所以 skipped 必须单独报,不能混进步数里。(2026-09-07 实测)
    skipped = [m["skipped"] for m in steps if "skipped" in m]
    if skipped:
        out.append((BAD, "跳过的步", f"{len(skipped)} 步没有候选:{sorted(set(skipped))} —— 查 reef.log 的 propose"))

    ns = [m["n"] for m in steps if "n" in m]
    out.append((OK if ns else WAIT, "闸门 n", f"{ns}" if ns else "还没出第一步"))

    slept, times, detail = sleep_during(work)
    mark = OK if slept < 60 else BAD
    out.append((mark, "机器睡眠", f"{times} 次,{detail}" + ("" if slept < 60 else " —— 在途请求会断,时间也白流")))
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
        cfg = [l for l in (work / "run.log").read_text(encoding="utf-8", errors="ignore").splitlines() if "消融配置" in l] if (work / "run.log").exists() else []
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
