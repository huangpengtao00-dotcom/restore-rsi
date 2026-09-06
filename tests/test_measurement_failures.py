"""测量失败不是分数:judge 自己坏了必须响,而真实的 0 分必须仍然是 0 分。

三处「返回 0.0」在 2026-09-06 之前是静默降级,和 09-04 修掉的 phantom episode id
是同一个病:闸门分不出「我没测出来」和「它什么都没做成」,而 0.0 是会决定胜负的
真分数 —— 09-03 两次 `select` 判决都压在一道恰好 0.0 的题上。

  harness/evolution.py:_score_output  参考图读不到 / judge 打分无 REEF_SCORE
  run.py:record_episode               记录段给最终图打分失败

改成 raise 之后,这里同时钉住反向判据:**合法的 0.0 仍然要是 0.0**。只测「坏了会
响」是不够的 —— 那样把 shortcut 惩罚或「一个工具都没跑成」也改成异常同样能过测,
而那才是真的把闸门弄坏(判据要能区分候选)。
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reef_example"))

from harness import evolution  # noqa: E402

REFS = ROOT / "reef_example" / "work" / "task_refs.json"


def _result(trajectory, exit_code=0):
    """最小的 EpisodeResultLike:evaluate 只用 trajectory / exit_code / residue。"""
    return types.SimpleNamespace(trajectory=list(trajectory), exit_code=exit_code, residue=())


def _assistant_tool_call(name: str, command: str) -> dict:
    return {
        "message": {
            "role": "assistant",
            "content": [{"type": "toolCall", "name": name, "arguments": {"command": command}}],
        }
    }


# --------------------------------------------------------------- 坏了必须响

def test_missing_reference_raises_instead_of_scoring_zero():
    """task_refs.json 里没有这个 task id → 抛错,不是 0.0。

    materialize.py 在任何 episode 之前写这个文件,所以查不到只有三种可能:没
    materialize、task id 漂了、文件坏了。三种都是响亮的问题,不是「复原了个 0 分」。
    """
    with pytest.raises(RuntimeError, match="no reference for this task"):
        evolution._score_output("no-such-task-id-ever", ROOT / "tasks" / "data" / "manifest.json")


def test_judge_scoring_failure_raises_instead_of_scoring_zero(monkeypatch, tmp_path):
    """`restore score` 跑了但没打出 REEF_SCORE → 抛错,不是 0.0。

    这是 *我们这一侧* 的测量失败。报成 0.0 的话,闸门会拿「judge 坏了」当「agent
    什么都没做成」,并据此发布或拒绝一条经验。
    """
    refs = {"fake-task": {"input": "in.png", "reference": "ref.png"}}
    REFS.parent.mkdir(parents=True, exist_ok=True)
    original = REFS.read_text() if REFS.exists() else None
    REFS.write_text(json.dumps({**(json.loads(original) if original else {}), **refs}))
    try:
        def broken_run(*args, **kwargs):
            return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="boom: no such file")

        monkeypatch.setattr(subprocess, "run", broken_run)
        with pytest.raises(RuntimeError, match="judge scoring failed"):
            evolution._score_output("fake-task", tmp_path / "out.png")
    finally:
        if original is None:
            REFS.unlink(missing_ok=True)
        else:
            REFS.write_text(original)


# ------------------------------------------------- 真的 0 分仍然必须是 0 分
# 少了这两条,「把所有 0.0 都改成异常」也能让上面两条通过 —— 而那会把闸门彻底弄坏。

def test_shortcut_still_scores_zero_not_raises():
    """碰参考图/自造图像工具是惩罚性 0 分,是判决而不是故障,必须仍然返回 0.0。"""
    trajectory = [_assistant_tool_call("bash", "cat tasks/data/haze_00/clean_000.png")]
    assert evolution.evaluate("[haze_00] restore it", _result(trajectory)) == 0.0


def test_episode_that_ran_nothing_scores_zero_not_raises(monkeypatch):
    """跑了 restore 但没有一次成功的 `restore run` → 改善量刻度下就是 0.0,不是故障。"""
    trajectory = [
        {"message": {"role": "assistant", "content": [
            {"type": "toolCall", "name": "bash", "arguments": {"command": "restore diagnose in.png"}}]}},
        {"message": {"role": "user", "content": "reef-episode-ep-test-0001 diagnose ok"}},
    ]
    monkeypatch.setattr(evolution, "_final_output_sha", lambda episode_id: None)
    assert evolution.evaluate("[haze_00] restore it", _result(trajectory)) == 0.0
