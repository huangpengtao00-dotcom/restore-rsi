"""同一台机器并行跑多个消融臂:每个臂的状态必须互不干扰。

reef 的 evaluate 串行跑 episode(`tuple(... for task in self._tasks)`),改不动
也不该改。所以并行的粒度放在**配置之间**:消融那几个臂互不相干,同时跑,总时间
等于最慢的那个而不是它们的和。实测一次完整跑 42 分钟(其中 evolve 段 37 分钟),
五个臂串行是三个多小时,并行就还是四十分钟。

前提是状态真的隔离。共用 work 目录会让 trace、缓存、task_refs 互相覆盖 ——
**而那种污染事后完全看不出来**:数字都在,只是来自别的臂。所以这里钉住隔离。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reef_example"))


def _work_dir_with(value: str | None) -> Path:
    """在给定 RESTORE_WORK 下,harness 认为状态目录在哪。"""
    import importlib

    import harness.evolution as ev

    old = os.environ.get("RESTORE_WORK")
    try:
        if value is None:
            os.environ.pop("RESTORE_WORK", None)
        else:
            os.environ["RESTORE_WORK"] = value
        importlib.reload(ev)
        return ev._work_dir()
    finally:
        if old is None:
            os.environ.pop("RESTORE_WORK", None)
        else:
            os.environ["RESTORE_WORK"] = old
        importlib.reload(ev)


def test_default_work_dir_is_unchanged():
    """没设变量时行为必须和以前完全一样,否则这个改动会悄悄搬走所有既有状态。"""
    assert _work_dir_with(None).name == "work"


def test_two_arms_get_disjoint_state():
    a = _work_dir_with("work/abl-mf-zero")
    b = _work_dir_with("work/abl-gate")
    assert a != b
    # 三样都必须分开:trace 记录发生了什么、cache 是内容寻址的产物、
    # task_refs 是判分用的参考图。任何一样共用都会串味。
    for sub in ("results/trace.jsonl", "cache", "task_refs.json"):
        assert (a / sub) != (b / sub)
    assert not str(a).startswith(str(b)) and not str(b).startswith(str(a)), "两个臂的目录不能互相嵌套"


def test_service_url_follows_the_port():
    """端口不分开,第二个臂根本起不来(或者更糟:连到第一个臂的服务上)。"""
    src = (ROOT / "reef_example" / "run.py").read_text()
    assert "REEF_PORT" in src, "run.py 必须从 REEF_PORT 取端口"
    assert '"http://127.0.0.1:8900"' not in src, "端口不能再是硬编码字面量"


@pytest.fixture
def scratch_arm(request):
    """一个临时的臂目录,测完删掉。

    materialize 把产物写在 `reef_example/<RESTORE_WORK>` 下(那是它的设计:臂目录
    要和 serve.yaml 同级才能被 reef 解析),所以用不了 tmp_path,得自己收拾。
    """
    import shutil

    arm = request.param
    target = ROOT / "reef_example" / arm
    shutil.rmtree(target, ignore_errors=True)
    try:
        yield arm
    finally:
        shutil.rmtree(target, ignore_errors=True)


@pytest.mark.parametrize("scratch_arm", ["work/t-arm-a", "work/t-arm-b"], indirect=True)
def test_materialize_writes_into_its_own_arm(scratch_arm):
    arm = scratch_arm
    """真跑一次 materialize,确认产物落在该臂目录里,而不是共用的 work/。"""
    if not (ROOT / "tasks" / "data" / "manifest.json").exists():
        pytest.skip("tasks/data 未生成:先跑 tasks/make_tasks.py")
    env = {**os.environ, "RESTORE_WORK": arm}
    env.pop("RESTORE_ABLATION", None)
    proc = subprocess.run(
        [sys.executable, "materialize.py"],
        cwd=ROOT / "reef_example", env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    produced = ROOT / "reef_example" / arm
    for name in ("tasks.json", "task_refs.json", "recipes/harness_evolve.yaml"):
        assert (produced / name).exists(), f"{arm} 缺 {name};stdout={proc.stdout[-300:]}"
