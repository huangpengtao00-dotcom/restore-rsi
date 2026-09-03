"""Skill evolution over image-restoration episodes: the method module serve.yaml names.

``propose`` is the self proposer from reef's harness_evolve tutorial, re-aimed:
the served model reads the current skill nodes and the batched failing tasks
(prompt + the score each earned) and proposes one mutation on a skill node - a
restoration *strategy* (which tools, in what order, for which diagnosis), never
a provider or a tool implementation.

``evaluate`` differs from the tutorial's exact-match grader: an episode's score
is the ``REEF_SCORE=<float>`` line that ``restore score`` printed. It is read
from the *tool result* in the trajectory first (the verifier's own stdout) and
only then from the agent's final text, so a fabricated number in the reply
cannot beat a real one. Episodes that took a shortcut - touching the reference
image, the task manifest or the degradation code instead of restoring through
the ``restore`` tools - score 0.0 with the reason logged: the reference is
visible to the agent (``--ref`` needs it) and a coding agent with a shell will
find it, so the verifier has to watch for it (this showed up in the first
manual pi smoke run, 2026-09-02: the agent read tasks/data/manifest.json to
invert the exact degradation).

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
        "photo by chaining classical tools through a CLI (`restore diagnose`, `restore run TOOL IN OUT`, "
        "`restore score IN OUT --ref REF`); the score is PSNR against the clean reference normalised to "
        "0..1 (0.0 = 15 dB or worse, 1.0 = 35 dB). The tasks below scored at or below 0.5 (failed).\n\n"
        f"Failed tasks (prompt, score):\n{json.dumps(failures, indent=2, ensure_ascii=False, default=str)}\n\n"
        f"Current skills:\n{json.dumps(skills, indent=2, ensure_ascii=False)}\n\n"
        "Write ONE improved or new skill (SKILL.md markdown) that gives the agent a concrete restoration "
        "strategy: how to read the diagnosis numbers, which tool sequence to use for combined haze+low-light "
        "and low-light+noise, when to apply a tool twice, what to avoid (e.g. dehazing a dark image first "
        "makes it darker; sharpening amplifies noise), and to ALWAYS end with `restore score` and the "
        "REEF_SCORE line. Rules the skill must state: only the `restore` commands touch images; never open "
        "the reference image, manifest or degradation code - that scores 0.\n"
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
    """Score one episode: the REEF_SCORE the verifier printed, or 0.0.

    Order of trust: the score inside a bash tool result (real `restore score`
    stdout) > the score in the final assistant text. A shortcut (reference /
    manifest / degradation code touched) zeroes the episode regardless.
    """
    tid = task_id(task) or "?"
    shortcut = _shortcut_taken(result.trajectory)
    tool_score = _score_from_tool_results(result.trajectory)
    text_score = parse_score(_final_assistant_text(result.trajectory))
    if shortcut is not None:
        log.warning("evaluate[%s]: shortcut -> 0.0: %s", tid, shortcut)
        score = 0.0
    elif tool_score is not None:
        score = tool_score
    elif text_score is not None:
        log.warning("evaluate[%s]: REEF_SCORE only in assistant text (no tool result) -> not trusted", tid)
        score = 0.0
    else:
        score = 0.0
    log.info(
        "evaluate[%s]: score=%.4f tool=%s text=%s exit=%s residue=%d",
        tid, score, tool_score, text_score, result.exit_code, len(result.residue),
    )
    return float(score)


def parse_score(text: str | None) -> float | None:
    """The last ``REEF_SCORE=<float>`` in ``text``, clipped to 0..1, or None."""
    if not text:
        return None
    found = SCORE_LINE.findall(text)
    if not found:
        return None
    return min(1.0, max(0.0, float(found[-1])))


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


def _final_assistant_text(trajectory) -> str | None:
    for message in reversed(list(_messages(trajectory))):
        if message.get("role") == "assistant":
            text = _text_parts(message.get("content"))
            if text.strip():
                return text
    return None


def _score_from_tool_results(trajectory) -> float | None:
    """The last REEF_SCORE printed by a tool (pi records tool output as
    ``role: toolResult`` messages)."""
    last = None
    for message in _messages(trajectory):
        if message.get("role") != "toolResult":
            continue
        score = parse_score(_text_parts(message.get("content")))
        if score is not None:
            last = score
    return last


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
            # `restore score IN OUT --ref .../clean_000.png` is the sanctioned use of the reference.
            sanitized = re.sub(r"restore\s+score\s+\S+\s+\S+\s+--ref\s+\S+", "restore score", command)
            for pattern in _SHORTCUT_PATTERNS:
                if pattern.search(sanitized):
                    return f"{part.get('name')}: {command[:200]}"
    return None
