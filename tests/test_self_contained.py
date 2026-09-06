"""这个仓必须自包含:任何一份源码都不许伸到仓外去取东西。

2026-09-06 CI 第一次跑就红了,原因是三处各自 `sys.path.insert` 到
`~/research/judge-lab` 再 import 同一个函数。在有那个目录的机器上一切正常 ——
所以本地永远是绿的,而在**任何别人的机器上**这个仓造不出数据、跑不了那几项测试。

这条判据在本地就能抓到它,不用等 CI 跑一轮。它必须能区分两种 path 注入:

  合法    指向仓内(tests/ 把 reef_example/ 加进 sys.path 才能 import harness)
  非法    指向 Path.home() / 绝对家目录 / 任何仓外位置
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", "work", "data", "node_modules"}

#: 仓外路径注入的形态。`Path.home()` 和 `~` 展开都算;`Path(__file__)` 起头的不算。
OUTSIDE_PATH = re.compile(r"Path\.home\(\)|os\.path\.expanduser|['\"]~/")


def _sources() -> list[Path]:
    return [
        p for p in ROOT.rglob("*.py")
        if not (set(p.relative_to(ROOT).parts) & SKIP_DIRS)
    ]


def test_there_are_sources_to_check():
    """判据本身要先证明它在看东西 —— 一个扫不到文件的扫描器永远是绿的。"""
    assert len(_sources()) >= 15, f"只扫到 {len(_sources())} 个 .py,SKIP_DIRS 是不是把仓吃掉了"


@pytest.mark.parametrize("path", _sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_import_from_outside_the_repo(path: Path):
    """不许 import 任何不在这个仓里、也不在依赖声明里的包。"""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            root_pkg = name.split(".")[0]
            assert root_pkg != "judgelab", (
                f"{path.relative_to(ROOT)}:{node.lineno} imports {name!r} —— judge-lab 不在这个仓里,"
                f"也不在 pyproject 的依赖里。底图合成的唯一实现是 restore_rsi.synth。"
            )


@pytest.mark.parametrize("path", _sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_sys_path_injection_outside_the_repo(path: Path):
    """sys.path 只能指向仓内。指向 home 的那种在别人机器上必然失效。"""
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if "sys.path" not in line:
            continue
        assert not OUTSIDE_PATH.search(line), (
            f"{path.relative_to(ROOT)}:{lineno} 把仓外路径塞进 sys.path:\n    {line.strip()}\n"
            f"仓内相对路径(Path(__file__).parent / ...)可以,家目录不行 —— "
            f"它在你的机器上有,在 CI 和别人那里没有,而症状是测试静默 skip 而不是变红。"
        )
