"""两条来自真实部署的判据:文件编码,和专家模型的挂载。

都来自 2026-09-06 在一台中文 Windows(GBK locale)上接第一个真专家模型时撞到的问题。
它们在 macOS 上永远不会出现 —— 又一次"本地绿不算绿"。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCAN_DIRS = ("toolbox", "restore_rsi", "tasks", "scripts", "reef_example")
SKIP_PARTS = {"__pycache__", ".venv", "work", "evidence"}


def _sources() -> list[Path]:
    out = []
    for root in SCAN_DIRS:
        for p in (ROOT / root).rglob("*.py"):
            if not (set(p.relative_to(ROOT).parts) & SKIP_PARTS):
                out.append(p)
    return out


def test_there_are_sources_to_scan():
    """扫不到文件的扫描器永远是绿的。"""
    assert len(_sources()) >= 10


@pytest.mark.parametrize("path", _sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_text_io_declares_utf8(path):
    """`read_text()` / `write_text()` 必须显式写 encoding。

    不写就跟着系统 locale 走。在中文 Windows 上那是 GBK,读一个 UTF-8 的
    registry.yaml 直接 `UnicodeDecodeError`,服务连启动都启动不了 —— 而在 macOS
    和 Linux 上(locale 是 UTF-8)永远看不到这件事。

    这个仓的注释和 note 字段大量是中文,所以这不是理论风险:2026-09-06 在 4070 上
    起工具服务时就是这么崩的,当时只能靠 `PYTHONUTF8=1` 绕过去。
    """
    src = path.read_text(encoding="utf-8")
    offenders = []
    for match in re.finditer(r"\.(read_text|write_text)\(", src):
        # 从括号开始做括号匹配,取出参数区间再看有没有 encoding
        i, depth, in_str, quote = match.end() - 1, 0, False, ""
        while i < len(src):
            c = src[i]
            if in_str:
                if c == "\\":
                    i += 2
                    continue
                if c == quote:
                    in_str = False
            elif c in "\"'":
                in_str, quote = True, c
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if "encoding" not in src[match.end():i]:
            offenders.append(src[:match.start()].count("\n") + 1)
    assert not offenders, (
        f"{path.relative_to(ROOT)} 第 {offenders} 行的 read_text/write_text 没写 encoding;"
        f"在 GBK locale 的机器上会崩,而在这里永远看不到"
    )


# ------------------------------------------------- 专家模型挂载

def test_expert_modules_hook_exists():
    """挂一个专家模型不该需要改这个仓。

    模型环境是各机器各不同的(RIDCP 的依赖只在那台 Windows 上装得起来),
    所以 adapter 不进这个仓,靠 RESTORE_EXPERT_MODULES 从外部挂。
    """
    from toolbox import server

    assert callable(server.load_expert_modules)
    assert server.load_expert_modules("") == [], "空配置应当挂 0 个,而不是报错"


def test_a_module_that_cannot_load_stops_the_server():
    """挂不上就该起不来,不能"少一个工具照常启动"。

    静默跳过的后果是:服务正常起来、/tools 少一个、循环那侧报 unknown tool,
    而真正的原因(权重路径错了之类)埋在启动日志里没人看。和 JarvisIR 里
    `restormer` 那个坑是同一件事:声明了却不能跑,而且不响。
    """
    from toolbox import server

    with pytest.raises(ModuleNotFoundError):
        server.load_expert_modules("definitely_not_a_real_module_zzz")


def test_an_expert_may_not_shadow_a_builtin(monkeypatch):
    """同名会让分数不知道是谁跑出来的。"""
    import types

    from toolbox import server

    mod = types.ModuleType("fake_expert_mod")
    mod.EXPERTS = {"dehaze_dcp": lambda img: img}
    monkeypatch.setitem(sys.modules, "fake_expert_mod", mod)
    with pytest.raises(RuntimeError, match="与 builtin 同名"):
        server.load_expert_modules("fake_expert_mod")


def test_a_module_without_experts_is_refused(monkeypatch):
    import types

    from toolbox import server

    mod = types.ModuleType("empty_expert_mod")
    monkeypatch.setitem(sys.modules, "empty_expert_mod", mod)
    with pytest.raises(RuntimeError, match="没有提供非空的 EXPERTS"):
        server.load_expert_modules("empty_expert_mod")


def test_a_good_module_registers_and_shows_up(monkeypatch):
    import types

    import numpy as np

    from toolbox import server

    mod = types.ModuleType("good_expert_mod")
    mod.EXPERTS = {"fake_ridcp": lambda img: np.asarray(img)}
    monkeypatch.setitem(sys.modules, "good_expert_mod", mod)
    monkeypatch.setattr(server, "EXPERT_ADAPTERS", {})
    assert server.load_expert_modules("good_expert_mod") == ["fake_ridcp"]
    assert "fake_ridcp" in server.available_tools()
