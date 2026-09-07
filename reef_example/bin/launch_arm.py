#!/usr/bin/env python3
"""发车一个消融臂,**脱离当前会话组**。

    launch_arm.py <臂名> <端口> <RESTORE_ABLATION 或 baseline> [max_score]

`nohup cmd &` 在 macOS 上不够:进程仍在同一个会话组里,主控会话一中断就连坐。
2026-09-07 实测——四个臂 record 段全跑完、evolve 正在跑,一次上下文压缩全部被杀,
每臂只留下 1–2 个 step 的落盘数据。日志尾部是正常的 "evolve step still running",
没有任何报错,所以从日志完全看不出它们是被杀的。

`start_new_session=True` 让子进程 setsid,pid 即新会话组的 pgid,主控怎么中断都不连坐。
同时把 pid 写进臂目录,收工/排障时按 pid 停(别用 pkill -f 模式,会误杀同类)。
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    arm, port, ablation = sys.argv[1], sys.argv[2], sys.argv[3]
    max_score = sys.argv[4] if len(sys.argv) > 4 else "0.9"

    work = f"work/{arm}"
    env = {**os.environ, "REEF_PORT": port, "RESTORE_WORK": work, "RESTORE_MAX_SCORE": max_score}
    if ablation and ablation != "baseline":
        env["RESTORE_ABLATION"] = ablation
    else:
        env.pop("RESTORE_ABLATION", None)

    log = Path(f"/tmp/abl-{arm}.log")
    with log.open("w") as handle:
        proc = subprocess.Popen(
            ["./run.sh"], cwd=HERE, env=env,
            stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,          # setsid:pid == pgid,主控中断不连坐
        )
    pidfile = HERE / work / "run.pid"
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(f"{proc.pid}\n", encoding="utf-8")
    print(f"{arm}: pid={proc.pid} pgid={proc.pid} port={port} ablation={ablation} log={log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
