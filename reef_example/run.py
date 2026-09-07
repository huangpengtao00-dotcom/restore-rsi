"""The harness evolution loop for restore-rsi, in the open (mirrors reef's tutorial run.py).

One pass:

    record - each task prompt runs as a short relay: reef inference serves the
             reply, run.py executes the `restore` commands it names through
             bin/restore and feeds their real stdout back as the next turn, up
             to MAX_TURNS or until the reply names no command. Every turn is
             recorded against its own receipt
    report - run.py scores the relay's final image with `restore score --ref`
             (the reference stays on this side, never in the transcript) and
             reports the REEF_SCORE against the turn receipts; every report at
             or below data.max_score (0.5)
             batches and triggers one gated evolve step: propose() asks the
             served model for a skill mutation over the failures, real pi
             episodes score current vs candidate on all tasks, a win publishes
    pull   - GET /reef/harness returns the winning composition; the evolved
             skill files are printed and saved to work/evolved_skills/

Start this through ./run.sh: it materializes work/tasks.json and starts the Reef
these constants point at. `python run.py --limit 1` records only the first task.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from reef_client import ReefClient, ReefClientError

from harness.ablation import current as current_ablation
from harness.evolution import parse_score, task_id
from harness.scoring import attained, batch_threshold

HERE = Path(__file__).resolve().parent
#: 本次配置的状态目录与端口。同一台机器上并行跑多个消融臂时,每个臂必须有自己的一份 ——
#: 共用会让 trace、缓存、task_refs 互相覆盖,而那种污染事后完全看不出来。
WORK = HERE / os.environ.get("RESTORE_WORK", "work")
SERVICE_URL = f"http://127.0.0.1:{os.environ.get('REEF_PORT', '8900')}"  # the Reef run.sh started
SCENARIO = os.environ.get("REEF_SCENARIO", "restore-evolve")  # this workload's isolated lane
TOKEN = "reef-local"  # matches serve.yaml
MODEL = "litellm/DeepSeek-V4-Flash"  # matches serve.yaml's upstream_model
PULL_TIMEOUT_S = float(os.environ.get("PULL_TIMEOUT_S", "3600"))

TASKS_FILE = WORK / "tasks.json"
REFS_FILE = WORK / "task_refs.json"
REGISTRY_FILE = HERE.parent / "toolbox" / "registry.yaml"
RESTORE = HERE / "bin" / "restore"

#: A `restore diagnose|run` command in the reply, in *command position*.
#:
#: Two failure modes had to be avoided at once. Anchoring to the line start
#: matches nothing, because DeepSeek-V4-Flash wraps every command in its own
#: tool-call markup (`<||DSML||parameter name="command" ...>cd /repo && restore
#: diagnose /path</...>`) - that is why the first full run scored 0 on every
#: recorded task. But matching the bare phrase anywhere executes the model's
#: own prose: "I should NOT restore run sharpen_unsharp" ran sharpen_unsharp
#: (found in the 2026-09-03 audit). So the command must be preceded by
#: something that makes it a command and not a mention: start of line, a shell
#: prompt or `&&`, a backtick, or the `>` that closes a markup tag.
COMMAND = re.compile(r"(?:^|[`>$]\s*|&&\s*)restore\s+(diagnose|run)\s+([^\n<`&;|]*)", re.M)

MAX_TURNS = int(os.environ.get("RECORD_MAX_TURNS", "6"))

FOLLOW_UP = (
    "Continue: issue the next `restore run TOOL IN OUT` command, or finish by replying with the path of "
    "your final image alone on the last line and no further commands."
)

#: A reply with no `restore` command usually means the model wandered off (the
#: first 3-task run lost a whole task to a reply of `pwd`), not that it is done.
#: One nudge is sent before the relay accepts the reply as final; two such
#: replies in a row end it.
NUDGE = (
    "That reply contained no `restore` command. Only `restore diagnose IN` and `restore run TOOL IN OUT` are "
    "available - no shell of your own. Reply with exactly one such command, or, if the restoration is finished, "
    "with the path of your final image alone on the last line."
)


def _tool_names() -> set[str]:
    """Tool names `restore run` accepts, so an echoed `restore run TOOL IN OUT`
    from the prompt's own catalog is reported back as unknown instead of being
    executed as a tool literally named TOOL."""
    import yaml

    return set(yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))["tools"])


def _restore(args: list[str], cwd: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run([str(RESTORE), *args], cwd=cwd, env=env, capture_output=True, text=True)


def _clip(proc: subprocess.CompletedProcess, limit: int = 400) -> str:
    return (proc.stdout or proc.stderr or "").strip()[:limit] or "(no output)"


def _infer(client: ReefClient, messages: list[dict], log, attempts: int = 4) -> tuple[dict, str] | None:
    """One relay turn, retried past transient upstream failures.

    Without this a single hiccup kills the whole run: reef answers 503
    "inference retry deadline exceeded (300s)" when the upstream stalls, the
    exception propagates out of the relay, and run.sh's EXIT trap takes reef
    down with it. That cost a re-roll of the noise-floor experiment on
    2026-09-03 - the upstream answered a direct probe in 1.5 s a minute later,
    so the failure was transient and the run was lost to the absence of a
    retry, not to the outage. Returns None when every attempt fails; the caller
    then closes the relay and scores the chain it already has.
    """
    for attempt in range(1, attempts + 1):
        try:
            return client.inference_with_record(SCENARIO, "/v1/chat/completions", {"model": MODEL, "messages": messages})
        except (ReefClientError, TimeoutError, OSError) as exc:
            status = getattr(exc, "status", None)
            if isinstance(exc, ReefClientError) and status is not None and status < 500:
                raise  # 4xx is our own bug (bad scenario, bad payload) - do not paper over it
            if attempt == attempts:
                log(f"  inference failed {attempts}x ({type(exc).__name__} {status}); closing the relay here")
                return None
            backoff = 5.0 * attempt
            log(f"  inference attempt {attempt}/{attempts} failed ({type(exc).__name__} {status}); retrying in {backoff:.0f}s")
            time.sleep(backoff)
    return None


def record_episode(tid: str, prompt: str, frame: dict, client: ReefClient, log) -> tuple[float, str, list[str]]:
    """Relay one task: served reply -> real `restore` commands -> stdout back.

    Returns (score, feedback, receipts). The relay runs in a scratch directory
    with RESTORE_EPISODE_ID=record-<task>, so its outputs and trace events land
    in work/results beside the episodes'. Whatever paths the model names are
    ignored: the chain starts at the task input and each successful step's
    output becomes the next step's input, which is also what keeps the reply
    from being able to point `restore` at a file of its choosing. Scoring
    happens after the relay closes, on this side, with the reference the
    transcript never carried; no tool ran -> 0.0.
    """
    tools = _tool_names()
    messages = [{"role": "user", "content": prompt}]
    receipts: list[str] = []
    chain: list[str] = []
    episode_id = f"record-{tid}-{int(time.time())}"
    env = {**os.environ, "RESTORE_EPISODE_ID": episode_id}
    nudged = False
    with tempfile.TemporaryDirectory(prefix=f"restore-record-{tid}-") as scratch:
        cwd = Path(scratch)
        current = Path(frame["input"])
        truncated = None
        for turn in range(1, MAX_TURNS + 1):
            answered = _infer(client, messages, log)
            if answered is None:
                truncated = turn
                break
            body, receipt = answered
            reply = body["choices"][0]["message"]["content"] or ""
            receipts.append(receipt)
            messages.append({"role": "assistant", "content": reply})
            commands = COMMAND.findall(reply)
            log(f"  turn {turn}: {len(commands)} command(s) {[verb for verb, _ in commands]} reply {reply.strip()[:160]!r}")
            if not commands:
                if nudged:
                    break
                nudged = True
                messages.append({"role": "user", "content": NUDGE})
                continue
            nudged = False
            observations = []
            for verb, rest in commands:
                if verb == "diagnose":
                    proc = _restore(["diagnose", str(current)], cwd, env)
                    observations.append(f"$ restore diagnose {current.name}\n{_clip(proc)}")
                    continue
                tool = (rest.split() or [""])[0]
                if tool not in tools:
                    observations.append(f"$ restore run {tool}\nunknown tool {tool!r}; available: {', '.join(sorted(tools))}")
                    continue
                out = cwd / f"step{len(chain) + 1}_{tool}.png"
                proc = _restore(["run", tool, str(current), str(out)], cwd, env)
                if proc.returncode != 0:
                    observations.append(f"$ restore run {tool}\nFAILED: {_clip(proc, 200)}")
                    continue
                current = out
                chain.append(tool)
                observations.append(f"$ restore run {tool} {out.name}\n{_clip(proc, 200)}")
            messages.append({"role": "user", "content": "\n\n".join(observations) + "\n\n" + FOLLOW_UP})
        if not chain:
            reason = f"upstream failed at turn {truncated}" if truncated else f"no tool ran in {len(receipts)} turn(s)"
            # 0 在两种口径下都是 0,但把口径写出来,免得后面对数时要猜
            return 0.0, f"episode={episode_id} {reason} (nothing ran; 0 in either scale)", receipts
        score_run = _restore(["score", frame["input"], str(current), "--ref", frame["reference"]], cwd, env)
    score = parse_score(score_run.stdout)
    if score is None:
        # Third of the same kind (see harness/evolution.py:_score_output): a
        # scoring failure on *our* side is not the relay's performance. A 0.0
        # here reports as a failing task, batches, and triggers a gated evolve
        # step over a failure that never happened - the loop would then be
        # learning from a broken measurement.
        raise RuntimeError(
            f"record[{tid}]: scoring the relay's final image failed "
            f"(exit={score_run.returncode}, no REEF_SCORE line): {_clip(score_run, 300)!r}"
        )
    note = f" truncated@{truncated}" if truncated else ""
    # 报给 reef 的是**达成率**(相对这道题穷举出来的上界),不是绝对分。
    #
    # `data.max_score` 是一把绝对的尺子,而这 12 道题的上界差 46.5 倍。实测 4/12
    # 判反:low_light_00 上界 1.000 只拿到 66%,绝对阈说它"成功"于是永不进 batch;
    # haze_noise_00 上界 0.202 拿到 113%,绝对阈说它"失败"于是进 batch,让 propose
    # 去为一道已经做到头的题找不存在的改进。batch 的组成因此系统性地偏了。
    #
    # 绝对分和上界都留在 feedback 里,所以这个换算随时可以还原,任何历史数字也还能对上。
    reported, scale = attained(score, frame)
    # episode id 显式写进 feedback,而不是让 propose 按 task 前缀去猜最近的那个 ——
    # 猜会在重跑同一任务时静默取错一轮,而取错的症状是"策略建议看起来莫名其妙"。
    return reported, f"episode={episode_id} turns={len(receipts)}{scale} chain=" + ">".join(chain) + note, receipts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="record only the first N tasks")
    args = parser.parse_args()

    tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))[: args.limit]
    refs = json.loads(REFS_FILE.read_text(encoding="utf-8"))
    client = ReefClient(SERVICE_URL, token=TOKEN, timeout_s=600.0)
    log_path = WORK / "run.log"
    log_path.parent.mkdir(exist_ok=True)

    def log(line: str) -> None:
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(f"{time.strftime('%H:%M:%S')} {line}\n")

    log(f"== record+report: {len(tasks)} task(s), scenario {SCENARIO}")
    # 这一臂到底在跑什么配置,必须写进它自己的日志。六次白跑全是"跑完了、日志正常、
    # 数据是错的";事后靠决策记录的字段形状反推能核出来,但那要人想到去核。
    # 打印的是 **解析后的** 四个字段(不是环境变量原文),所以拼错的开关会在这里
    # 报错、而不是静默退回默认值。
    ablation = current_ablation()
    log(f"== 消融配置:{ablation.label()} | 全字段 {ablation.as_dict()}")
    log(f"== 隔离:work={WORK} service={SERVICE_URL}")
    # 等待上限也要写进日志:它到点就 sys.exit(2),而 run.sh 的 EXIT trap 会把 reef
    # 一起杀掉 —— 正在跑的那一步于是没有落盘,只留下前面几步。步数变多、每步
    # episode 变多时(RESTORE_TASK_IDS 一放开就是 4 倍),3600s 的默认值不够,
    # 而超时退出的日志长得和"跑完了没人赢"一模一样。
    log(f"== 等待上限:{PULL_TIMEOUT_S:.0f}s(PULL_TIMEOUT_S);到点即退出并停掉 reef")
    threshold = batch_threshold()
    log(f"== 失败判据:reported <= {threshold} 进 batch(读自 recipe 的 data.max_score)")
    failures = 0
    for index, task in enumerate(tasks, start=1):
        tid = task_id(task) or f"task{index}"
        t0 = time.monotonic()
        log(f"task {index} {tid}: relaying (up to {MAX_TURNS} turns)")
        score, feedback, receipts = record_episode(tid, task, refs[tid], client, log)
        # Exactly one reference, and it must be the first turn's: CordisProcessor
        # drops any report claiming more than a single request (processor.py:48,
        # `len(context.references) != 1 -> NEVER`) *silently* - a whole 3-task run
        # ended with batch_ready false and no error anywhere. And propose() reads
        # the task prompt off `messages[-1]` of the referenced payload, which is
        # the clean `[task_id] ...` prompt only on turn 1; a later turn's payload
        # ends with a tool observation, so it would batch under task id "?".
        client.report(
            SCENARIO,
            {"agent_record_id": f"restore-evolve-{tid}", "score": score, "feedback": f"{tid} {feedback}"},
            references=receipts[:1],
        )
        if score <= threshold:
            failures += 1
        log(f"task {index} {tid}: reported {score:.4f} ({feedback}) receipts {receipts} {time.monotonic() - t0:.0f}s")

    if failures == 0:
        log("every task passed: nothing batched, no evolve step runs")
        return
    log(f"{failures} failing report(s) batched; each triggers one gated evolve step (episodes take minutes)")

    # pull: GET /reef/harness 404s until a winning step publishes its tree.
    manifest = None
    deadline = time.monotonic() + PULL_TIMEOUT_S
    last_status = None
    while manifest is None and time.monotonic() < deadline:
        try:
            manifest = client.get("/reef/harness", extra_headers={"x-reef-scenario": SCENARIO})
        except (TimeoutError, OSError) as exc:  # noqa: PERF203
            # reef runs evolve episodes synchronously and stops answering HTTP
            # meanwhile; a socket timeout here means "still evolving", not failure.
            # Letting it propagate killed run.py, whose EXIT trap in run.sh then
            # killed reef mid-episode - the reason evolve never seemed to finish.
            log(f"  service busy ({type(exc).__name__}); evolve step still running")
            time.sleep(10.0)
            continue
        except ReefClientError as exc:  # noqa: PERF203 - publish poll
            if exc.status != 404:  # 404 only means nothing has published yet
                raise
            try:
                status = client.get("/reef/status")
            except (TimeoutError, OSError):
                time.sleep(10.0)
                continue
            if error := status.get("error"):
                raise SystemExit(f"evolve step failed: {error}; check work/stack/reef.log") from exc
            brief = json.dumps({k: v for k, v in status.items() if k not in ("error",)}, sort_keys=True, default=str)
            if brief != last_status:
                log(f"  status: {brief[:300]}")
                last_status = brief
            time.sleep(5.0)
    if manifest is None:
        log(f"no skill mutation won a gate within {PULL_TIMEOUT_S:.0f}s; see work/agent-record/*.commits.jsonl")
        sys.exit(2)

    log(f"published: artifact {manifest['release_id']} (parent {manifest['parent_release_id']})")
    log("gate metrics (the evolve step that published this artifact):")
    log(json.dumps(manifest["gate"], indent=2, sort_keys=True, default=str))
    out_dir = WORK / "evolved_skills"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log("evolved skill files:")
    for path, text in sorted(manifest["files"].items()):
        if "/skills/" in path:
            log(f"--- {path} ---")
            log(text)
            target = out_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
