#!/usr/bin/env python3
"""闭环判据:测试不是只会变红,它还会**消失**。

`pytest -q` 在干净 clone 上打印 `144 passed, 27 skipped` 并**退出码 0**。绿灯,
覆盖面少了 27 项 —— 这个仓真实发生过:`tasks/data/` 是 gitignored 的生成物,
而 `tasks/make_tasks.py` 曾经 `sys.path.insert` 到 `~/research/judge-lab`,于是
在任何别人的机器上都造不出数据,依赖 manifest 的测试静默 skip。

所以 CI 的判据不能只是"退出码为 0",必须**把允许的 skip 数钉死**。多出来一个就红,
并打印它的理由 —— 让"测试消失"和"测试失败"一样响。

    uv run python scripts/check_skips.py report.xml --max-skipped 0

退出码:0 = 通过;1 = 有失败/错误,或 skip 超出允许值;2 = 报告本身读不了。
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="pytest --junit-xml 产出的 xml")
    parser.add_argument(
        "--max-skipped", type=int, default=0,
        help="允许的 skip 数上限(默认 0)。装齐依赖后应当是 0;"
             "刻意不装某个可选依赖时写出那个确切的数,别写一个宽松的上界。",
    )
    args = parser.parse_args()

    try:
        root = ET.parse(args.report).getroot()
    except (OSError, ET.ParseError) as exc:
        print(f"读不了测试报告 {args.report}: {exc!r}", file=sys.stderr)
        return 2

    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        print(f"{args.report} 里没有 testsuite —— pytest 没跑起来?", file=sys.stderr)
        return 2

    total = sum(int(s.get("tests", 0)) for s in suites)
    failures = sum(int(s.get("failures", 0)) for s in suites)
    errors = sum(int(s.get("errors", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    passed = total - failures - errors - skipped

    print(f"收集 {total} 项:通过 {passed} / 失败 {failures} / 错误 {errors} / 跳过 {skipped}")

    if skipped:
        reasons = Counter()
        for case in root.iter("testcase"):
            for skip in case.iter("skipped"):
                reasons[(skip.get("message") or "(无理由)").strip()[:160]] += 1
        print("跳过的理由:")
        for reason, count in reasons.most_common():
            print(f"  [{count:3d}] {reason}")

    problems = []
    if failures or errors:
        problems.append(f"{failures} 个失败 + {errors} 个错误")
    if skipped > args.max_skipped:
        problems.append(
            f"跳过 {skipped} 项,超出允许的 {args.max_skipped} —— "
            f"测试消失和测试失败一样是坏消息,不要靠放宽这个数来变绿"
        )
    if problems:
        print("闸门不通过:" + ";".join(problems), file=sys.stderr)
        return 1

    print(f"闸门通过:{passed} 项通过,跳过 {skipped} 项(允许 {args.max_skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
