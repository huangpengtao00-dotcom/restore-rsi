"""Skill evolution over image-restoration episodes: the method module serve.yaml names.

``propose`` is the self proposer from reef's harness_evolve tutorial, re-aimed:
the served model reads the current skill nodes and the batched failing tasks
(prompt + the score each earned) and proposes one mutation on a skill node - a
restoration *strategy* (which tools, in what order, for which diagnosis), never
a provider or a tool implementation.

``evaluate`` differs from the tutorial's exact-match grader in where the score
comes from: nothing the episode says is trusted. The episode has ``restore
diagnose`` and ``restore run`` only - no reference path, no score command - and
the judge here recovers the hash of its last successful ``restore run`` output
from the toolbox trace, pulls that image out of the content-addressed cache and
scores it against the reference itself. So a number in the reply is not a score,
and an episode cannot hill-climb on ground truth it never holds.

That protocol is a correction, not a design choice made up front. The first
version handed the episode the reference path and the ``restore score`` command
and read the number it printed; the agent stopped restoring and started
searching against the reference. Same model, same tools, same three tasks, the
only change being the reference's visibility: 0.858 leaky vs 0.394 blind -
roughly half the score was the leak (README, 实测记录). A shortcut - own image
tooling, or naming the reference / manifest / degradation code - still zeroes
the episode: a coding agent with a shell finds those files if they are reachable
(the 2026-09-02 smoke run read tasks/data/manifest.json to invert the exact
degradation), so the judge keeps watching for it.

The model ``propose`` asks is ``models.served``, the binding reef hands it, so
this module never names an endpoint or holds a credential. ``run.py`` imports
only the pure helpers (``parse_score``, ``task_id``), so reef stays a lazy import.
"""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger("restore.evolution")

#: Score parsing: the verifier prints this as its last stdout line.
SCORE_LINE = re.compile(r"REEF_SCORE=\s*(-?\d+(?:\.\d+)?)")

#: Task prompts start with ``[task_id]``.
TASK_ID = re.compile(r"^\[([A-Za-z0-9_.-]+)\]")

#: Entry ids and skill names become path segments (skills/<name>/SKILL.md).
_ENTRY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

#: Shell commands that mean the episode bypassed the toolbox. Any bash tool
#: call matching one of these (outside the sanctioned `restore score ... --ref`)
#: zeroes the episode.
_SHORTCUT_PATTERNS = (
    re.compile(r"clean_\d+\.png"),  # the reference image, by name
    re.compile(r"--ref\b"),  # scoring against the reference: blind episodes have no such command
    re.compile(r"manifest\.json"),  # ground-truth degradation parameters
    re.compile(r"restore_rsi/degrade|/degrade/|make_tasks\.py"),  # the degradation code
    # Hand-rolled image code instead of the toolbox (the smoke run inverted the
    # degradation in numpy): python / PIL / numpy / imagemagick in a shell call.
    re.compile(r"\bpython3?\b|\bPIL\b|\bnumpy\b|\bnp\.|\bmagick\b|\bconvert\s+\S+\.png"),
)


# --------------------------------------------------------------------------- propose


def propose(nodes, samples, models):
    """Ask the served model for one skill improvement over its own failures.

    ``nodes`` are the composition's (kind, config) pairs and ``samples`` the
    batched failing requests (each carries the recorded chat payload and the
    reported score). Any endpoint or parse failure returns ``None`` - a skipped
    step, never a crash.
    """
    if not samples:
        return None
    skills = [dict(config) for name, config in nodes if name == "skill"]
    failures = []
    for sample in samples:
        payload = sample.payload
        messages = payload.get("messages") if isinstance(payload, dict) else None
        prompt = messages[-1].get("content") if messages else str(payload)
        failures.append(
            {
                "task": task_id(prompt) or "?",
                "prompt": prompt,
                "score": getattr(sample, "score", None),
                "feedback": getattr(sample, "feedback", None),
            }
        )
    prompt = (
        "You are improving the skill file of an image-restoration agent. The agent restores a degraded "
        "photo by chaining classical tools through a CLI - `restore diagnose IN` and `restore run TOOL IN OUT`, "
        "and nothing else. It never sees the clean original and never computes a score: afterwards a judge "
        "compares its final image with the original at PSNR, normalised to 0..1 (0.0 = 15 dB or worse, "
        "1.0 = 35 dB). The tasks below scored at or below 0.5 (failed).\n\n"
        f"Failed tasks (prompt, score):\n{json.dumps(failures, indent=2, ensure_ascii=False, default=str)}\n\n"
        f"Current skills:\n{json.dumps(skills, indent=2, ensure_ascii=False)}\n\n"
        "Write ONE improved or new skill (SKILL.md markdown) that gives the agent a concrete restoration "
        "strategy: how to read the diagnosis numbers, which tool sequence to use for combined haze+low-light "
        "and low-light+noise, when to apply a tool twice, what to avoid (e.g. dehazing a dark image first "
        "makes it darker; sharpening amplifies noise), and how to decide it is done from the diagnosis alone. "
        "Rules the skill must state: only the `restore` commands touch images; the last `restore run` output "
        "is what gets judged, so finish by naming its path; never look for the clean original, the task "
        "manifest or the degradation code, and never write a score of your own - both score 0.\n"
        "Respond with exactly one JSON object and nothing else:\n"
        '{"id": "<skill name>", "name": "skill", "config": {"name": "<same skill name>", '
        '"text": "<the full SKILL.md markdown>"}}\n'
        "Reuse an existing skill's name to update it (prefer improving 'restore-strategy'); "
        "use a new lowercase-hyphen name to add one."
    )
    try:
        # DeepSeek-V4-Flash reasons before answering; give it room (the
        # training thread blocks on this call, so still bounded).
        reply = models.served.chat([{"role": "user", "content": prompt}], timeout_s=240.0)
    except Exception as exc:  # noqa: BLE001 - a skipped step, logged
        log.warning("propose: served model call failed: %r", exc)
        return None
    proposal = _parse_proposal(reply)
    if proposal is None:
        log.warning("propose: no usable proposal in reply: %.300r", reply)
        return None
    entry_id, config = proposal
    from reef.train.cordis_backend import Mutation  # lazy: keeps run.py reef-free

    op = "update" if any(skill.get("name") == entry_id for skill in skills) else "create"
    log.info("propose: %s %s (%d chars)", op, entry_id, len(config["text"]))
    # reef 只在 commits.jsonl 里留 op/id,被门拒掉的候选文本会丢;原样落盘以便事后审
    _log_proposal({"op": op, "entry_id": entry_id, "reply": reply, "config": config})
    return Mutation(op, entry_id, {"name": "skill", "config": config})


def _log_proposal(record: dict) -> None:
    import json, os, time
    from pathlib import Path

    out = Path(os.environ.get("RESTORE_RESULTS_DIR", "work/results")) / "proposals.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps({"ts": time.time(), **record}, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- evaluate


def evaluate(task: str, result) -> float:
    """Score one episode: the judge scores the episode's final image itself.

    Blind protocol (2026-09-03): the episode never sees the reference or any
    score - it only gets `restore diagnose` and `restore run`. The reference
    lives here, in the judge, so the gate measures restoration rather than the
    agent's ability to hill-climb on ground truth (see README, reference leak).
    A shortcut (own image tooling, or touching the reference/manifest/degradation
    code) zeroes the episode regardless.
    """
    tid = task_id(task) or "?"
    shortcut = _shortcut_taken(result.trajectory)
    if shortcut is not None:
        log.warning("evaluate[%s]: shortcut -> 0.0: %s", tid, shortcut)
        return 0.0
    episode = _episode_id(result)
    if episode is None:
        # "I could not measure this episode" is not "this episode restored
        # nothing", and reporting the second for the first is a silent
        # downgrade: reef coerces the return to float, so a 0.0 here enters the
        # gate as a real score and decides wins and losses. Both `select`
        # verdicts on 2026-09-03 turned on a task that scored exactly 0.0, and
        # at least one of those was this failure rather than a bad chain.
        # bin/restore now echoes the id on every call, so a missing id means
        # the episode ran no `restore` command at all - which the next branch
        # scores 0.0 correctly - or the wiring is broken, which must be loud.
        raise RuntimeError(
            f"evaluate[{tid}]: no reef-episode marker in the trajectory. bin/restore echoes it on every "
            f"call, so either no `restore` command ran (then there is nothing to score and the trace is "
            f"empty for this episode) or the wrapper is not on PATH. exit={result.exit_code} "
            f"messages={len(list(_messages(result.trajectory)))}"
        )
    output_sha = _final_output_sha(episode)
    image = _cached_image(output_sha) if output_sha else None
    if image is None:
        # Reached only when the episode really produced no successful
        # `restore run` - a legitimate 0.0 under the gain-over-input scale.
        log.warning("evaluate[%s]: episode %s ran no successful `restore run` -> 0.0", tid, episode)
        return 0.0
    score = _score_output(tid, image)
    log.info(
        "evaluate[%s]: score=%.4f sha=%s exit=%s residue=%d",
        tid, score, output_sha, result.exit_code, len(result.residue),
    )
    return float(score)


def parse_score(text: str | None) -> float | None:
    """The last ``REEF_SCORE=<float>`` in ``text``, clipped to 0..1, or None.

    Used by the judge here and by run.py, which scores the recorded pass the
    same way; the episode itself never runs `restore score`.
    """
    if not text:
        return None
    found = SCORE_LINE.findall(text)
    if not found:
        return None
    return min(1.0, max(0.0, float(found[-1])))


def _episode_id(result) -> str | None:
    """The RESTORE_EPISODE_ID every `bin/restore` call echoes, taken from the
    trajectory.

    All markers in one episode must agree - the wrapper derives the id from the
    episode root, so a disagreement means the derivation fell back mid-episode
    and the trace is split across ids. That is not a detail: taking the *first*
    marker used to lock the judge onto a phantom id whenever the agent's first
    command ran from an odd cwd, and it then found no `restore run` under it and
    scored a working episode 0.0 (three times on 2026-09-03). Raise instead of
    guessing which id is the real one.
    """
    import re as _re

    seen: list[str] = []
    for message in _messages(result.trajectory):
        for match in _re.finditer(r"reef-episode-([A-Za-z0-9_-]+)", _text_parts(message.get("content")) or ""):
            if match.group(1) not in seen:
                seen.append(match.group(1))
    if len(seen) > 1:
        raise RuntimeError(
            f"episode markers disagree: {seen}. bin/restore should derive one id per episode; "
            f"a split means its derivation fell back part-way and the trace cannot be joined."
        )
    return seen[0] if seen else None


def _final_output_sha(episode_id: str) -> str | None:
    """The output hash of the last successful `restore run` of this episode.

    The toolbox's trace is the record of what actually ran (the episode root is
    already deleted by the time evaluate is called), and every run's output is
    kept in the content-addressed cache, so the hash is enough to score it.
    """
    import json as _json

    trace = _results_dir() / "trace.jsonl"
    last = None
    try:
        lines = trace.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            event = _json.loads(line)
        except ValueError:
            continue
        if event.get("episode") != episode_id or event.get("kind") != "run":
            continue
        if event.get("status") in {"ok", "cache_hit"} and event.get("output_sha"):
            last = event["output_sha"]
    return last


def _results_dir():
    import os
    from pathlib import Path

    return Path(os.environ.get("RESTORE_RESULTS_DIR", Path(__file__).resolve().parent.parent / "work" / "results"))


def _cache_dir():
    import os
    from pathlib import Path

    return Path(os.environ.get("RESTORE_CACHE_DIR", Path(__file__).resolve().parent.parent / "work" / "cache"))


def _cached_image(output_sha: str):
    """The cache file whose *content* hashes to output_sha (cache keys are
    `<tool>-<input_sha>.png`, so the output hash needs a scan)."""
    import hashlib

    for path in _cache_dir().glob("*.png"):
        if hashlib.sha256(path.read_bytes()).hexdigest()[:16] == output_sha:
            return path
    return None


def _score_output(tid: str, image_path) -> float:
    """Run `restore score` on the episode's final image against the reference.

    Reference paths were written by materialize.py and are read *here*, in the
    judge - nothing in the episode environment carries them.
    """
    import json as _json
    import subprocess
    from pathlib import Path

    refs_file = Path(__file__).resolve().parent.parent / "work" / "task_refs.json"
    try:
        frame = _json.loads(refs_file.read_text())[tid]
    except (OSError, KeyError, ValueError) as exc:
        # Same class of bug as the phantom episode id (see _episode_id): "I have
        # no reference for this task" is a wiring failure, not "this episode
        # restored nothing". Returning 0.0 hands the gate a real score, and a
        # score of 0.0 decides wins and losses - on 2026-09-03 both `select`
        # verdicts turned on a task that scored exactly 0.0. materialize.py
        # writes this file before any episode runs, so a miss means the file was
        # not materialized, the task id drifted, or the json is corrupt. All
        # three are loud problems.
        raise RuntimeError(
            f"evaluate[{tid}]: no reference for this task in {refs_file} ({exc!r}). "
            f"materialize.py writes it before the run; a miss means it was not materialized, "
            f"the task id drifted, or the file is corrupt - none of which is a score of 0.0."
        ) from exc
    proc = subprocess.run(
        ["restore", "score", frame["input"], str(image_path), "--ref", frame["reference"]],
        capture_output=True, text=True, timeout=120, check=False,
        env={**__import__("os").environ, "RESTORE_EPISODE_ID": f"judge-{tid}"},
    )
    score = parse_score(proc.stdout)
    if score is None:
        # The judge's own scoring subprocess failed or printed no REEF_SCORE
        # line. That is a measurement failure on *this* side of the protocol,
        # so it must not be reported as the episode's performance: the gate
        # cannot tell "the judge broke" from "the agent achieved nothing", and
        # it will publish or reject on the difference. reef keeps the exception
        # loud (`_score_episode` sits outside `_run_and_score`'s try, and the
        # trainer re-raises after `abort_step`), so raising aborts the step
        # instead of silently poisoning it.
        raise RuntimeError(
            f"evaluate[{tid}]: judge scoring failed (exit={proc.returncode}, no REEF_SCORE line). "
            f"stderr/stdout: {(proc.stderr or proc.stdout)[:300]!r}"
        )
    return score


def task_id(prompt: str | None) -> str | None:
    match = TASK_ID.match(prompt or "")
    return match.group(1) if match else None


# --------------------------------------------------------------------------- helpers


def _parse_proposal(reply: str):
    """The strict proposal object dug out of the model's text, or None."""
    try:
        parsed = json.loads(reply[reply.find("{") : reply.rfind("}") + 1])
    except ValueError:
        return None
    entry_id, config = parsed.get("id"), parsed.get("config")
    if parsed.get("name") != "skill" or not isinstance(entry_id, str) or not _ENTRY_NAME.fullmatch(entry_id):
        return None
    if not isinstance(config, dict) or config.get("name") != entry_id:
        return None
    text = config.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return entry_id, {"name": entry_id, "text": text}


def _messages(trajectory):
    """pi session events carry the message under ``message``; flat events are
    tolerated too."""
    for event in trajectory:
        message = event.get("message") or event
        if isinstance(message, dict) and message.get("role"):
            yield message


def _text_parts(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    return ""


def _shortcut_taken(trajectory) -> str | None:
    """The first tool call that bypassed the toolbox, or None."""
    for message in _messages(trajectory):
        if message.get("role") != "assistant" or not isinstance(message.get("content"), list):
            continue
        for part in message["content"]:
            if not isinstance(part, dict) or part.get("type") != "toolCall":
                continue
            args = part.get("arguments") or {}
            command = " ".join(str(value) for value in args.values()) if isinstance(args, dict) else str(args)
            # No exemption: under the blind protocol nothing in an episode has a
            # legitimate reason to name the reference or score against it. (The
            # judge's own `restore score --ref` runs in a separate subprocess and
            # never appears in an episode trajectory.)
            for pattern in _SHORTCUT_PATTERNS:
                if pattern.search(command):
                    return f"{part.get('name')}: {command[:200]}"
    return None
