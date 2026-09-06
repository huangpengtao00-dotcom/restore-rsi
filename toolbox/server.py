"""工具服务:在有卡的机器上跑,把复原工具暴露成 HTTP。

    uv run python -m toolbox.server --port 8710

循环那侧用 `remote:<tool>` 后端调它,`RESTORE_TOOL_SERVER` 指到这里。
工具在哪于是变成一个配置:循环和卡可以在同一台机器(零网络),也可以隔着
tailnet(实测 macbook -> 4070 走 DERP:132ms,有丢包,所以客户端带重试)。

**为什么不用 FastAPI**:这个仓的依赖是 numpy/pillow/pyyaml 三个。工具服务是
基础设施,不该给它加一层框架 —— 标准库的 http.server 完全够,单请求串行正好
匹配"一张卡一次跑一个模型"。

**协议**(两个端点,故意小):

    GET  /tools           -> {"tools": [...], "device": "..."}  服务端此刻真能跑什么
    POST /run/<tool>      body = 图片字节 -> 200 + 结果图片字节
                          4xx = 工具名不对(重试无意义)
                          5xx = 执行失败(客户端会重试)

只收发字节,**不传文件名**。JarvisIR 的 `process_image` 在输入路径里 grep
"fog"/"night"/"snow" 来决定哪些工具可用,而数据集按场景命名目录 —— 那是一条从
文件系统泄漏答案的通道。这里不留。

**接 JarvisIR 的 13 个专家模型**:实现一个 adapter 加进 `EXPERT_ADAPTERS`,
签名和 builtin 一样是 `np.ndarray -> np.ndarray`。它们各自的依赖互相冲突
(13 篇论文各带一套 basicsr/torch),所以每个 adapter 要么在自己的 conda 环境里
以子进程跑,要么在自己的容器里 —— 服务端只负责转发,不负责调和版本。
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

log = logging.getLogger("toolbox.server")

#: 真专家模型的挂载点。空的时候服务端只提供 builtin 工具 —— 这不是占位,
#: 是让整条链路在没有卡的机器上也能端到端自测(客户端、协议、缓存、trace 全都真跑)。
EXPERT_ADAPTERS: dict[str, object] = {}

MAX_BODY = 64 * 1024 * 1024  # 64MB,一张图绰绰有余;防止一个坏请求吃光内存


def _builtin_tools() -> dict:
    """注册表里 backend 是 builtin 的那些,直接从 toolbox.builtin 取函数。"""
    import yaml

    from . import builtin
    from .cli import HERE

    registry = yaml.safe_load((HERE / "registry.yaml").read_text())["tools"]
    out = {}
    for name, spec in registry.items():
        backend = spec.get("backend", "")
        if backend.startswith("builtin:"):
            fn = getattr(builtin, backend.split(":", 1)[1], None)
            if fn is not None:
                out[name] = fn
    return out


def available_tools() -> dict:
    """服务端此刻真能跑的工具。**先算出来再报**,不报注册表里的意图。"""
    return {**_builtin_tools(), **EXPERT_ADAPTERS}


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
        return "cpu (torch 在,但没有可用的 cuda)"
    except ImportError:
        return "cpu (无 torch)"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        log.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/tools":
            tools = available_tools()
            self._send_json(200, {
                "tools": sorted(tools),
                "device": _device(),
                # 分开报,因为"能跑"和"是真模型"是两件事:全是 builtin 时
                # 链路是通的,但结论不能当成真专家模型上的结论。
                "experts": sorted(EXPERT_ADAPTERS),
                "builtin": sorted(_builtin_tools()),
            })
            return
        self._send_json(404, {"error": "only GET /tools"})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/run/"):
            self._send_json(404, {"error": "only POST /run/<tool>"})
            return
        tool = self.path[len("/run/"):].strip("/")
        tools = available_tools()
        if tool not in tools:
            # 4xx:客户端重试没有意义,而且要把"有什么"说出来,免得对面猜
            self._send_json(400, {"error": f"unknown tool {tool!r}", "known": sorted(tools)})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            self._send_json(400, {"error": f"bad Content-Length {length}"})
            return
        payload = self.rfile.read(length)
        try:
            image = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
            result = tools[tool](image)
            if not isinstance(result, np.ndarray):
                raise TypeError(f"{tool} 返回了 {type(result).__name__},要 np.ndarray")
            buf = io.BytesIO()
            Image.fromarray(result).save(buf, format="PNG")
        except Exception as exc:  # noqa: BLE001
            # 5xx + 真实原因。绝不返回 200 加原图 —— 那会让客户端把
            # "工具崩了"记成"这个工具没改善画质",而两者在训练信号里不可区分。
            log.error("run %s failed: %s", tool, traceback.format_exc())
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}", "tool": tool})
            return
        self._send(200, buf.getvalue(), "image/png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=os.environ.get("RESTORE_SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("RESTORE_SERVER_PORT", "8710")))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    tools = available_tools()
    log.info("device: %s", _device())
    log.info("tools: %d 个(builtin %d,expert %d)", len(tools), len(_builtin_tools()), len(EXPERT_ADAPTERS))
    if not EXPERT_ADAPTERS:
        log.warning("EXPERT_ADAPTERS 为空:链路可测,但跑出来的结论只对经典算子成立,不是真专家模型上的结论")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
