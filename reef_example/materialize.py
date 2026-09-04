"""Copy serve.yaml's recipe sections where the recipe registry reads them, with
the task prompts generated from tasks/data/manifest.json.

serve.yaml carries the deployment (`reef:`, `services:`) and the harness_evolve
recipe. The registry loads named recipes from REEF_RECIPE_CONFIG_DIR, so run.sh
drops those four sections there before starting Reef - after filling
`evolution.tasks` from `restore.task_ids` - and writes the task list beside
them (work/tasks.json) for run.py. The same prompt text is what reef records on
the pass through inference and what pi gets as the episode prompt, so the two
never drift.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RECIPE_SECTIONS = ("implementation", "model", "evolution", "data")


def tool_catalog() -> str:
    """One line per tool from toolbox/registry.yaml (what `restore catalog` shows)."""
    registry = yaml.safe_load((HERE.parent / "toolbox" / "registry.yaml").read_text())["tools"]
    return "\n".join(f"  - {name} ({spec['task']}): {spec.get('note', '')}" for name, spec in registry.items())


def opaque_id(task_id: str) -> str:
    """The id the episode sees: `t<8 hex>` instead of `haze_low_light_00`.

    The readable id leaked the answer. `[haze_low_light_00]` plus an input path
    under `tasks/data/haze_low_light_00/` tells the agent which degradations are
    present without diagnosing anything - and it used that: across 207 tool
    calls it applied dehaze on the haze tasks and never on the one without,
    while its own haze reading sat at 0.073 (2026-09-03 audit). Any claim about
    how well the agent reads a diagnosis needs that channel closed.
    """
    return "t" + hashlib.sha256(task_id.encode()).hexdigest()[:8]


def task_prompt(catalog: str, prompt_id: str, input_path: Path) -> str:
    """The episode prompt for one manifest task.

    Starts with the stable `[<opaque id>]` prefix (evaluate/report key on it),
    names the input, lists the two `restore` commands the episode may use, and
    pins the reply format: the final image's path, alone on the last line.

    Blind by construction, in two ways: no reference path and no `restore score`
    (the judge in harness.evolution holds both), and no readable task id or
    input path (see `opaque_id`) - so neither the answer nor the *question* is
    in the prompt.

    The last paragraph covers the recorded relay, where the model has no shell
    of its own: run.py executes each command it names and feeds the real stdout
    back as the next turn, so one-command-per-reply is the honest instruction in
    both environments.
    """
    return (
        f"[{prompt_id}] Restore the degraded image {input_path} as close to its clean original as you can. "
        f"Work in the current directory (write intermediate/final PNGs here). Commands available in the shell:\n"
        f"  restore diagnose IN            -> JSON with degradation estimates (haze, low_light, noise, blur, 0..1)\n"
        f"  restore run TOOL IN OUT        -> apply one tool; chain tools by feeding OUT into the next IN\n"
        f"Tools (TOOL):\n{catalog}\n"
        f"Plan the tool sequence from the diagnosis and apply it. Your last `restore run` output is what gets judged - "
        f"there is no score command and no reference image in this episode: a judge compares your final image with the "
        f"clean original afterwards. Finish your reply with the exact path of your final image, alone on the last "
        f"line.\n"
        f"Rules (violations score 0): only `restore run` may create or modify images - no python/PIL/numpy/imagemagick "
        f"of your own; never open, copy or inspect the task manifest or the degradation code.\n"
        f"If you have no shell of your own, just name the command you want to run: it is executed for you and "
        f"its real output comes back, so work one command per reply and read the result before deciding the next."
    )


def main() -> None:
    config = yaml.safe_load((HERE / "serve.yaml").read_text())
    restore = config["restore"]
    manifest_path = (HERE / restore["manifest"]).resolve()
    if not manifest_path.exists():
        sys.exit(f"missing {manifest_path}: run `uv run python tasks/make_tasks.py` first")
    by_id = {task["task_id"]: task for task in json.loads(manifest_path.read_text())}
    missing = [task_id for task_id in restore["task_ids"] if task_id not in by_id]
    if missing:
        sys.exit(f"task ids not in manifest: {missing}")
    catalog = tool_catalog()
    work = HERE / "work"
    inputs = work / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    # The episode only ever sees work/inputs/<opaque>.png, so neither the id nor
    # the path names the degradations. task_refs.json keys on the same opaque id
    # (the judge reads it); task_map.json is the reverse map, for analysis only.
    repeats = int(restore.get("repeats", 1))
    tasks, refs, mapping = [], {}, {}
    for task_id in restore["task_ids"]:
        frame = by_id[task_id]["frames"][0]
        prompt_id = opaque_id(task_id)
        episode_input = inputs / f"{prompt_id}.png"
        shutil.copyfile(frame["input"], episode_input)
        prompt = task_prompt(catalog, prompt_id, episode_input)
        # The same prompt, `repeats` times. reef runs one episode per entry per
        # side, so a task's repeats are independent samples of the same
        # configuration - the gate votes over them, and their disagreement is
        # the step's own reliability read-out.
        tasks.extend([prompt] * repeats)
        refs[prompt_id] = {"input": str(episode_input), "reference": frame["reference"]}
        mapping[prompt_id] = task_id

    recipe = {key: config[key] for key in RECIPE_SECTIONS}
    recipe["evolution"] = {**recipe["evolution"], "tasks": tasks}
    # Measurement override. The noise floor needs one evolve step per run
    # regardless of how the recorded task scored, and `data.max_score` (0.5)
    # normally suppresses the step when the task did *not* fail - which silently
    # produced no sample when a relay happened to score 0.5021.
    if (override := os.environ.get("RESTORE_MAX_SCORE")) is not None:
        recipe["data"] = {**recipe["data"], "max_score": float(override)}
        print(f"  data.max_score overridden to {override} (RESTORE_MAX_SCORE)")
    (work / "recipes").mkdir(parents=True, exist_ok=True)
    (work / "recipes" / "harness_evolve.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True))
    # Two lists, deliberately different. `evolution.tasks` (in the recipe) carries
    # the repeats, because that is the evaluation. run.py's record pass wants each
    # task once - it is generating failing traffic to batch on, not measuring - so
    # tasks.json stays unique. Feeding it the repeated list would fire `repeats`
    # times as many evolve steps for no added information.
    unique = tasks[:: max(1, repeats)]
    (work / "tasks.json").write_text(json.dumps(unique, indent=2, ensure_ascii=False) + "\n")
    (work / "task_refs.json").write_text(json.dumps(refs, indent=2) + "\n")
    (work / "task_map.json").write_text(json.dumps(mapping, indent=2) + "\n")
    print(f"materialized {len(tasks)} entries ({len(mapping)} tasks x {repeats} repeats) -> {work / 'recipes' / 'harness_evolve.yaml'}, {work / 'tasks.json'}")
    for prompt_id, task_id in mapping.items():
        print(f"  {prompt_id} = {task_id}")


if __name__ == "__main__":
    main()
