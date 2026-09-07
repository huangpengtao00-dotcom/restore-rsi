#!/usr/bin/env python3
"""发车一个消融臂,**脱离当前会话组**。

    launch_arm.py <臂名> <端口> <RESTORE_ABLATION 或 baseline> [max_score] [run.py 的参数...]

环境变量原样传下去(RESTORE_TASK_IDS / RESTORE_ORACLE 等),所以"评估用几道题"
和"录制跑几道题"可以分开设:前者进 recipe,后者用 `--limit N`。

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

#: reef_example 目录 —— 这个脚本在它的 bin/ 下,所以要往上一级。
#: (第一版写成 `.parent` 直接指到 bin/,四次发车全是 FileNotFoundError: './run.sh';
#: 而当时的测试只检查源码里有没有 start_new_session=True,静态过了、实际起不来。)
HERE = Path(__file__).resolve().parent.parent


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    arm, port, ablation = sys.argv[1], sys.argv[2], sys.argv[3]
    max_score = sys.argv[4] if len(sys.argv) > 4 else "0.9"
    extra = sys.argv[5:]          # 透传给 run.sh -> run.py

    runner = HERE / "run.sh"
    if not runner.is_file():
        # 说清楚是路径不对,而不是抛一个光秃秃的 FileNotFoundError 堆栈
        print(f"找不到 {runner} —— HERE 应当指向含 run.sh 的 reef_example 目录", file=sys.stderr)
        return 2

    work = f"work/{arm}"
    env = {**os.environ, "REEF_PORT": port, "RESTORE_WORK": work, "RESTORE_MAX_SCORE": max_score}
    if ablation and ablation != "baseline":
        env["RESTORE_ABLATION"] = ablation
    else:
        env.pop("RESTORE_ABLATION", None)

    log = Path(f"/tmp/abl-{arm}.log")
    with log.open("w") as handle:
        proc = subprocess.Popen(
            ["./run.sh", *extra], cwd=HERE, env=env,
            stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,          # setsid:pid == pgid,主控中断不连坐
        )
    pidfile = HERE / work / "run.pid"
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(f"{proc.pid}\n", encoding="utf-8")
    print(
        f"{arm}: pid={proc.pid} pgid={proc.pid} port={port} ablation={ablation} "
        f"tasks={os.environ.get('RESTORE_TASK_IDS', 'serve.yaml 默认')} args={extra or '无'} log={log}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
