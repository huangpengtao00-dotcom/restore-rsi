"""The harness evolution loop for restore-rsi, in the open (mirrors reef's tutorial run.py).

One pass:

    record - each task prompt goes once through reef inference (a plain chat
             completion: no shell, so the model answers with the `restore run`
             commands it would issue); reef serves the reply and records the
             exchange against a receipt
    report - run.py executes that plan for real through bin/restore, scores the
             output with `restore score --ref` and reports the REEF_SCORE
             against the receipt; every report at or below data.max_score (0.5)
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

from harness.evolution import parse_score, task_id

HERE = Path(__file__).resolve().parent
SERVICE_URL = "http://127.0.0.1:8900"  # the Reef run.sh started
SCENARIO = os.environ.get("REEF_SCENARIO", "restore-evolve")  # this workload's isolated lane
TOKEN = "reef-local"  # matches serve.yaml
MODEL = "litellm/DeepSeek-V4-Flash"  # matches serve.yaml's upstream_model
PULL_TIMEOUT_S = float(os.environ.get("PULL_TIMEOUT_S", "3600"))

TASKS_FILE = HERE / "work" / "tasks.json"
REFS_FILE = HERE / "work" / "task_refs.json"
RESTORE = HERE / "bin" / "restore"
RUN_LINE = re.compile(r"restore\s+run\s+([A-Za-z0-9_]+)\s+(\S+)\s+(\S+)")


def execute_plan(tid: str, reply: str, frame: dict) -> tuple[float, str]:
    """Run the `restore run` chain the reply names, score the final output.

    Returns (score, feedback). The chain runs in a scratch directory with
    RESTORE_EPISODE_ID=record-<task>, so its verdict/output/trace land in
    work/results beside the episodes'. No commands -> 0.0 ("no plan").
    """
    steps = RUN_LINE.findall(reply)
    if not steps:
        return 0.0, "no plan"
    env = {**os.environ, "RESTORE_EPISODE_ID": f"record-{tid}-{int(time.time())}"}
    with tempfile.TemporaryDirectory(prefix=f"restore-record-{tid}-") as scratch:
        cwd = Path(scratch)
        # Whatever names the model used, the chain starts at the task input and
        # each step's output feeds the next.
        current = Path(frame["input"])
        for index, (tool, _in, _out) in enumerate(steps, start=1):
            out = cwd / f"step{index}_{tool}.png"
            run = subprocess.run([str(RESTORE), "run", tool, str(current), str(out)], cwd=cwd, env=env, capture_output=True, text=True)
            if run.returncode != 0:
                return 0.0, f"step {index} {tool} failed: {(run.stderr or run.stdout).strip()[:200]}"
            current = out
        score_run = subprocess.run(
            [str(RESTORE), "score", frame["input"], str(current), "--ref", frame["reference"]],
            cwd=cwd, env=env, capture_output=True, text=True,
        )
    score = parse_score(score_run.stdout)
    if score is None:
        return 0.0, f"score failed: {(score_run.stderr or score_run.stdout).strip()[:200]}"
    return score, "chain=" + ">".join(tool for tool, _, _ in steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="record only the first N tasks")
    args = parser.parse_args()

    tasks = json.loads(TASKS_FILE.read_text())[: args.limit]
    refs = json.loads(REFS_FILE.read_text())
    client = ReefClient(SERVICE_URL, token=TOKEN, timeout_s=600.0)
    log_path = HERE / "work" / "run.log"
    log_path.parent.mkdir(exist_ok=True)

    def log(line: str) -> None:
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(f"{time.strftime('%H:%M:%S')} {line}\n")

    log(f"== record+report: {len(tasks)} task(s), scenario {SCENARIO}")
    failures = 0
    for index, task in enumerate(tasks, start=1):
        tid = task_id(task) or f"task{index}"
        t0 = time.monotonic()
        body, receipt = client.inference_with_record(
            SCENARIO,
            "/v1/chat/completions",
            {"model": MODEL, "messages": [{"role": "user", "content": task}]},
        )
        reply = body["choices"][0]["message"]["content"] or ""
        score, feedback = execute_plan(tid, reply, refs[tid])
        client.report(
            SCENARIO,
            {"agent_record_id": f"restore-evolve-{tid}", "score": score, "feedback": f"{tid} {feedback}"},
            references=[receipt],
        )
        if score <= 0.5:
            failures += 1
        log(f"task {index} {tid}: score {score:.4f} ({feedback}) receipt {receipt} {time.monotonic() - t0:.0f}s")
        log(f"  reply: {reply.strip()[:400]!r}")

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
        except ReefClientError as exc:  # noqa: PERF203 - publish poll
            if exc.status != 404:  # 404 only means nothing has published yet
                raise
            status = client.get("/reef/status")
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
    out_dir = HERE / "work" / "evolved_skills"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log("evolved skill files:")
    for path, text in sorted(manifest["files"].items()):
        if "/skills/" in path:
            log(f"--- {path} ---")
            log(text)
            target = out_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)


if __name__ == "__main__":
    main()
