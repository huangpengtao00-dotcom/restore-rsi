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

import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RECIPE_SECTIONS = ("implementation", "model", "evolution", "data")


def tool_catalog() -> str:
    """One line per tool from toolbox/registry.yaml (what `restore catalog` shows)."""
    registry = yaml.safe_load((HERE.parent / "toolbox" / "registry.yaml").read_text())["tools"]
    return "\n".join(f"  - {name} ({spec['task']}): {spec.get('note', '')}" for name, spec in registry.items())


def task_prompt(task: dict, catalog: str) -> str:
    """The episode prompt for one manifest task.

    Starts with the stable `[task_id]` prefix (evaluate/report key on it),
    names the input and the reference, lists the `restore` commands, and pins
    the reply format: the `REEF_SCORE=` line `restore score` printed, alone on
    the last line. The last paragraph is for the recorded pass, where the same
    prompt is answered without a shell: run.py executes the listed commands
    itself and scores the real output.
    """
    frame = task["frames"][0]
    return (
        f"[{task['task_id']}] Restore the degraded image {frame['input']} as close to its clean original as you can. "
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
        f"If you cannot execute commands, reply instead with the exact `restore run ...` commands you would run, "
        f"in order, one per line (use in.png for the input, chain via out1.png, out2.png, ...), and nothing else."
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
    tasks = [task_prompt(by_id[task_id], catalog) for task_id in restore["task_ids"]]

    recipe = {key: config[key] for key in RECIPE_SECTIONS}
    recipe["evolution"] = {**recipe["evolution"], "tasks": tasks}
    work = HERE / "work"
    (work / "recipes").mkdir(parents=True, exist_ok=True)
    (work / "recipes" / "harness_evolve.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True))
    (work / "tasks.json").write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n")
    (work / "task_refs.json").write_text(
        json.dumps({task_id: by_id[task_id]["frames"][0] for task_id in restore["task_ids"]}, indent=2) + "\n"
    )
    print(f"materialized {len(tasks)} task(s) -> {work / 'recipes' / 'harness_evolve.yaml'}, {work / 'tasks.json'}")


if __name__ == "__main__":
    main()
