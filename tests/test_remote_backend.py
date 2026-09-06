"""工具服务与 remote 后端的契约:换了执行位置,结果不能变;失败不能变成分数。

为什么要有这条链路:循环跑在哪台机器、卡在哪台机器,是两件独立的事。
真专家模型要 GPU,而 GPU 未必和跑循环的机器是同一台(实测 macbook -> 4070
走 tailnet DERP 中继:132ms,有丢包)。JarvisArt 也是为此做了 Agent-to-Lightroom
的 server-client 协议 —— Lightroom 只能跑在特定机器上。

三条性质:
  1. 经 HTTP 跑一遍和本机直接跑,**像素完全相同**
  2. 工具名不对 -> 4xx(客户端不该重试);工具崩了 -> 5xx 且客户端抛错
  3. 服务端报的是它此刻真能跑的,不是注册表里打算接的
"""

from __future__ import annotations

import socket
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from toolbox import builtin, server  # noqa: E402
from toolbox.cli import remote_tool_url, run_remote_tool  # noqa: E402

SRC = ROOT / "tasks" / "data" / "haze_00" / "input_000.png"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def tool_server():
    """在随机端口起一个真的工具服务。真跑 HTTP,不 mock —— mock 掉的正是要测的东西。"""
    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


# ------------------------------------------------- 1) 换位置不改结果

@pytest.mark.skipif(not SRC.exists(), reason="tasks/data 未生成:先跑 tasks/make_tasks.py")
@pytest.mark.parametrize("tool", ["dehaze_dcp", "lowlight_gamma", "denoise_median"])
def test_remote_result_is_pixel_identical_to_local(tool, tool_server, tmp_path, monkeypatch):
    """这条是整个后端存在的前提:执行位置不是一个自由变量。

    它一旦红,之前所有跨机器跑出来的数字都不能和本机的比 —— 那正是
    "拿两个数比大小却没先确认可比性" 那类错误的又一个入口。
    """
    monkeypatch.setattr("toolbox.cli.TOOL_SERVER", tool_server)
    local = getattr(builtin, tool)(np.asarray(Image.open(SRC).convert("RGB")))
    out = tmp_path / "remote.png"
    run_remote_tool(f"remote:{tool}", SRC, out)
    remote = np.asarray(Image.open(out).convert("RGB"))
    assert np.array_equal(local, remote), f"{tool}: 经 HTTP 与本机结果不同,max|diff|={np.abs(local.astype(int)-remote.astype(int)).max()}"


# ------------------------------------------------- 2) 失败必须是失败

def test_unknown_tool_is_rejected_without_retrying(tool_server, tmp_path, monkeypatch):
    """4xx:工具名不对,重试多少次都一样。错误里要带「有什么」,免得对面猜。"""
    monkeypatch.setattr("toolbox.cli.TOOL_SERVER", tool_server)
    if not SRC.exists():
        pytest.skip("tasks/data 未生成")
    with pytest.raises(RuntimeError, match="rejected: HTTP 400"):
        run_remote_tool("remote:no_such_tool", SRC, tmp_path / "o.png")
    assert not (tmp_path / "o.png").exists()


def test_a_crashing_tool_raises_rather_than_returning_the_input(tool_server, tmp_path, monkeypatch):
    """工具崩了必须抛错。

    绝不能返回 200 加原图:那会让客户端把"工具崩了"记成"这个工具没改善画质",
    而这两件事在训练信号里完全不可区分 —— JarvisIR 的 `continue` 就是这么来的。
    """
    def explode(_img):
        raise ZeroDivisionError("模型炸了")

    monkeypatch.setitem(server.EXPERT_ADAPTERS, "always_broken", explode)
    monkeypatch.setattr("toolbox.cli.TOOL_SERVER", tool_server)
    monkeypatch.setattr("toolbox.cli.REMOTE_RETRIES", 1)  # 5xx 会重试,测试里不等
    if not SRC.exists():
        pytest.skip("tasks/data 未生成")
    with pytest.raises(RuntimeError, match="always_broken"):
        run_remote_tool("remote:always_broken", SRC, tmp_path / "o.png")
    assert not (tmp_path / "o.png").exists(), "失败时不该留下半个产物"


def test_unreachable_server_raises(tmp_path, monkeypatch):
    """服务不可达 != 这个工具没用。"""
    monkeypatch.setattr("toolbox.cli.TOOL_SERVER", f"http://127.0.0.1:{_free_port()}")
    monkeypatch.setattr("toolbox.cli.REMOTE_RETRIES", 1)
    if not SRC.exists():
        pytest.skip("tasks/data 未生成")
    with pytest.raises(RuntimeError, match="failed 1x"):
        run_remote_tool("remote:dehaze_dcp", SRC, tmp_path / "o.png")


# ------------------------------------------------- 3) 报的是真能跑的

def test_server_reports_what_it_can_actually_run(tool_server, monkeypatch):
    from toolbox.cli import remote_tools

    monkeypatch.setattr("toolbox.cli.TOOL_SERVER", tool_server)
    reported = remote_tools()
    assert reported == set(server.available_tools()), "服务端报的工具集与它真能跑的不一致"
    assert "ridcp" not in reported, "注册表里 enabled:false 的专家模型不该被服务端报成可用"


def test_expert_adapters_are_reported_separately(tool_server, monkeypatch):
    """`experts` 和 `builtin` 分开报:全是 builtin 时链路是通的,
    但结论只对经典算子成立,不能当成真专家模型上的结论。"""
    import json
    import urllib.request

    monkeypatch.setitem(server.EXPERT_ADAPTERS, "fake_expert", lambda img: img)
    with urllib.request.urlopen(f"{tool_server}/tools", timeout=10) as resp:  # noqa: S310
        payload = json.loads(resp.read().decode())
    assert payload["experts"] == ["fake_expert"]
    assert "fake_expert" not in payload["builtin"]
    assert set(payload["tools"]) == set(payload["experts"]) | set(payload["builtin"])


def test_remote_spec_must_be_a_bare_tool_name():
    assert remote_tool_url("remote:ridcp", "http://x:1/")[1] == "http://x:1/run/ridcp"
    for bad in ("remote:", "remote:a/b"):
        with pytest.raises(ValueError, match="bad remote backend spec"):
            remote_tool_url(bad)
