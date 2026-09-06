"""工具注册表的契约:声明 ≠ 可用,而两者的差必须是显式的。

来源是 JarvisIR 里一个具体的 bug:`ALL_TOOLS` 有 13 项,模型被告知都能选,
但 `restormer` 既没有 `process_image_with_models` 里的执行分支,也不在
`all_model_paths` 里。选中它会走 `print(...)` + `continue` —— 链继续往下,
最后返回一个"处理过"的路径,reward 照常计算。结果是训练信号里

    "这个工具没用"   和   "这个工具没接上"

完全不可区分,而模型只会学到前者。

这里把这件事拆成三条可测的性质:
  1. 启用的工具必须真的可执行(仓库级不变量,防的就是上面那种)
  2. 未启用的工具必须被显式拒绝,且与"不存在的工具"分开报
  3. docker 后端不把宿主机文件名带进容器(路径泄漏的另一半)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from toolbox import cli  # noqa: E402

RESTORE = [sys.executable, "-m", "toolbox.cli"]


# ------------------------------------------------- 1) 启用的必须真能跑

def test_every_enabled_tool_is_actually_runnable():
    """仓库级不变量。这条红了就说明注册表里躺着一个 restormer。"""
    broken = [
        (name, cli.check_tool(name, spec)[1])
        for name, spec in cli._enabled_registry().items()
        if not cli.check_tool(name, spec)[0]
    ]
    assert not broken, f"声明为启用却跑不了:{broken}"


def test_disabled_tools_are_declared_not_silently_absent():
    """未启用的工具仍要留在注册表里,带 note 说明它是什么。

    删掉比标 false 更糟:那样"这个工具还没接"这件事就没有任何地方记着了。
    """
    all_tools, enabled = cli._registry(), cli._enabled_registry()
    disabled = {n: s for n, s in all_tools.items() if n not in enabled}
    assert disabled, "12 个 JarvisIR 专家模型应当在册但未启用"
    for name, spec in disabled.items():
        assert spec.get("note"), f"{name} 未启用但没写它是什么"
        assert spec.get("backend"), f"{name} 未启用但没写将来怎么跑"


# ------------------------------------------------- 2) 拒绝必须显式且可区分

def test_catalog_lists_only_runnable_tools():
    proc = subprocess.run([*RESTORE, "catalog"], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    listed = {t["tool"] for group in json.loads(proc.stdout).values() for t in group}
    assert listed == set(cli._enabled_registry()), "catalog 列出了跑不了的工具"
    assert "ridcp" not in listed, "未启用的专家模型不该出现在 agent 看到的目录里"


@pytest.mark.parametrize(
    "tool,expect_status",
    [("ridcp", "disabled_tool"), ("no_such_tool_at_all", "unknown_tool")],
    ids=["在册但未启用", "根本不存在"],
)
def test_refusals_are_distinct_and_nonzero(tool, expect_status, tmp_path):
    """两种拒绝要人做的事不同,所以不能报成同一种;而且都必须非零退出。

    非零这一点是关键:JarvisIR 那边是 `continue`,返回码始终是成功。
    """
    src = ROOT / "tasks" / "data" / "haze_00" / "input_000.png"
    if not src.exists():
        pytest.skip("tasks/data 未生成:先跑 tasks/make_tasks.py")
    proc = subprocess.run(
        [*RESTORE, "run", tool, str(src), str(tmp_path / "out.png")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode != 0, "被拒绝的工具调用必须非零退出,否则链会当它成功了"
    assert json.loads(proc.stderr)["status"] == expect_status
    assert not (tmp_path / "out.png").exists(), "被拒绝时不该产出任何文件"


# ------------------------------------------------- 3) docker 后端的契约

def test_docker_argv_hides_the_host_filename(tmp_path):
    """容器只看得到 input.png / output.png。

    JarvisIR 的 process_image 在输入路径里 grep "fog"/"night"/"snow" 来限制
    可用工具,数据集按场景命名目录时那就是一条从路径泄漏答案的通道。这个仓
    已经为此把任务 id 换成不可读哈希,工具层不能再开一条。
    """
    argv = cli.docker_argv("docker:jarvisir-tools ridcp", tmp_path / "in", tmp_path / "out")
    joined = " ".join(argv)
    assert argv[:3] == ["docker", "run", "--rm"]
    assert argv[-2:] == [cli.DOCKER_IN, cli.DOCKER_OUT]
    assert "haze" not in joined and "fog" not in joined
    assert "input.png" in joined and "output.png" in joined


def test_docker_argv_rejects_a_malformed_spec():
    with pytest.raises(ValueError, match="bad docker backend spec"):
        cli.docker_argv("docker:", Path("/tmp/a"), Path("/tmp/b"))


def test_docker_backend_raises_when_docker_is_missing(monkeypatch, tmp_path):
    """没有 docker 就说没有 —— 不能让这一步看起来跑过了。"""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="需要 docker"):
        cli.run_docker_tool("docker:jarvisir-tools ridcp", tmp_path / "a.png", tmp_path / "b.png")


def test_doctor_exits_zero_while_every_enabled_tool_works():
    proc = subprocess.run([*RESTORE, "doctor"], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    assert "跑不了 0" in proc.stdout
