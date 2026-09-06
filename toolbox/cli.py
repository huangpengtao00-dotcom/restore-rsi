"""`restore` —— agent 在 episode 里唯一能碰的复原接口。四个子命令:

  restore catalog                       列出**可用**工具(registry.yaml 里 enabled 的)
  restore doctor                        逐个验证声明的工具是否真能跑(声明 != 可用)
  restore diagnose IN                   退化诊断 JSON(现在是启发式;以后换 VLM/IQA,接口不变)
  restore run TOOL IN OUT               跑一个工具;按 (tool, 输入内容哈希) 缓存,同序列不重跑
  restore submit OUT                    显式交卷:判这一张。不交则退回最后一步(并记下"没交")
  restore score IN OUT [--ref REF]      打分并写 verifier 记录;stdout 最后一行 `REEF_SCORE=<float>`

设计约束(与 reef episode 机制对齐):
- episode 目录跑完即删,evaluate() 只能拿到 stdout/trajectory —— 所以 `score` 把结果、证据、
  工具链写到 RESTORE_RESULTS_DIR(外部持久目录),并把分数打到 stdout 最后一行。
- 每一步都落 JSONL 轨迹(工具、输入/输出哈希、耗时、状态),这是可观测与失败归因的原料。
- 不静默降级:工具崩 → 非零退出 + 显式 status,绝不悄悄把输入当输出。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).parent
RESULTS_DIR = Path(os.environ.get("RESTORE_RESULTS_DIR", HERE.parent / "work" / "results"))
CACHE_DIR = Path(os.environ.get("RESTORE_CACHE_DIR", HERE.parent / "work" / "cache"))
TRACE_PATH = Path(os.environ.get("RESTORE_TRACE", RESULTS_DIR / "trace.jsonl"))
EPISODE_ID = os.environ.get("RESTORE_EPISODE_ID", "adhoc")


def _registry() -> dict:
    """全部工具,含未启用的。只有 `doctor` 和错误信息该用这个。"""
    return yaml.safe_load((HERE / "registry.yaml").read_text(encoding="utf-8"))["tools"]


def _enabled_registry() -> dict:
    """agent 真正能用的工具。`enabled: false` 的声明在案但不可执行。

    分开这两个,是因为"这个工具没用"和"这个工具没接上"必须可区分 ——
    JarvisIR 把 restormer 放进 ALL_TOOLS 却没有它的执行分支,选中它只会
    `print` 一句然后 `continue`,链照常走完、reward 照常算,两种情况在
    训练信号里长得一模一样。

    registry 里的 `enabled` 是**默认值**,`RESTORE_ENABLE_TOOLS` 在运行时覆盖它:

        RESTORE_TOOL_SERVER=http://<host>:8710 RESTORE_ENABLE_TOOLS=ridcp ./run.sh

    为什么要分开:专家模型的可用性是**这台机器此刻**的属性,不是仓库的属性。
    RIDCP 装在一台特定的机器上,那台机器关机服务就没了 —— 把 `enabled: true`
    提交进仓,等于宣称所有人随时都能跑它,于是 CI 永远红,而"启用即可跑"那条
    不变量就只能被删掉。真正该做的是让仓库说默认情况,让环境说此刻的情况。
    """
    override = {t.strip() for t in os.environ.get("RESTORE_ENABLE_TOOLS", "").split(",") if t.strip()}
    registry = _registry()
    unknown = override - set(registry)
    if unknown:
        # 拼错一个工具名就静默少启用一个,而症状只是 agent "好像没用那个工具"
        raise ValueError(
            f"RESTORE_ENABLE_TOOLS 里有注册表中不存在的工具:{sorted(unknown)};"
            f"已知:{sorted(registry)}"
        )
    return {
        name: spec for name, spec in registry.items()
        if name in override or spec.get("enabled", True)
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _trace(event: dict) -> None:
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": time.time(), "episode": EPISODE_ID, **event}
    with TRACE_PATH.open("a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


# ---------------- catalog ----------------

def cmd_catalog() -> int:
    reg = _enabled_registry()
    by_task: dict[str, list] = {}
    for name, spec in reg.items():
        by_task.setdefault(spec["task"], []).append({"tool": name, "cost": spec.get("cost", 1), "note": spec.get("note", "")})
    print(json.dumps(by_task, ensure_ascii=False, indent=2))
    return 0


# ---------------- diagnose ----------------

#: diagnose 的标定常数。来源写在 diagnose_array 的 docstring 里,
#: 由 tests/test_diagnose.py 的契约测试守住(干净图四项须 ~0、同底图须单调、模糊须能测出)。
_DIAG = {
    "low_light_zero": 0.32,   # 去雾后亮度均值高于此视为不暗(12 题实测:不暗 [0.324, 0.499])
    "low_light_span": 0.30,
    "haze_zero": 0.30,        # 亮度归一后的暗通道地板/大气光,干净图 0.281、无雾最高 0.281
    "haze_span": 0.45,
    "noise_span": 0.20,       # 强噪声实测中值残差 0.107–0.179,除以 0.20 不饱和
    # 模糊:二阶导能量 / 一阶导能量。零点取 12 张干净图里最低的那个(0.743),
    # 于是干净图与本任务集的退化图全读 0(它们都没加过模糊);高斯模糊 r=1 起开始读到。
    # 注意:grad/std 这类"自归一梯度能量"测不出模糊 —— 模糊把一阶导和 std 同比缩小,
    # 比值几乎不变(实测 r=0→3 只从 0.0548 降到 0.0473)。必须用二阶/一阶之比。
    "blur_zero": 0.74,
    "blur_span": 0.45,
}


def _luminance(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float64) / 255.0 @ np.array([0.299, 0.587, 0.114])


def diagnose_array(img: np.ndarray) -> dict:
    """退化诊断。按消融配置分发:默认 v1(解耦),`diagnosis=coupled_v0` 走修复前那版。

    保留 v0 不是为了兼容 —— 是因为"观测互相污染值多少可达分数"这个问题,只有把
    坏的那版留在手边才答得出来。两版输出同一个 JSON 形状,`diagnoser` 字段区分。
    """
    import os

    if os.environ.get("RESTORE_ABLATION", "").find("diagnosis=coupled_v0") >= 0:
        return _diagnose_v0_coupled(img)
    return _diagnose_v1_decoupled(img)


def _diagnose_v0_coupled(img: np.ndarray) -> dict:
    """修复前的诊断。**消融臂,不要拿它跑基线。**

    四个读数在原图上各自算统计量,于是互相污染(2026-09-03 实测):同一张图同样的
    雾,只是加了暗光,雾读数就从 0.725 掉到 0.065(掉 91%);而 `blur` 在**干净参考
    图上就读 0.912** —— 它测的其实是"梯度能量低",在任何不带噪声的图上都成立,所以
    它是个反噪声读数,不是模糊读数。

    行为侧后果也测过:agent 唯一的观测是这个,于是把从不出现在任何最优链里的锐化
    用成了第二常用工具(23.7%),而含雾题最优链的第一步去雾只占 9.2%。

    数值与 3847ed5^ 完全一致,原样搬过来,不做任何"顺手的改进" —— 一改就不是对照了。
    """
    from PIL import ImageFilter

    x = img.astype(np.float64) / 255.0
    lum = x @ np.array([0.299, 0.587, 0.114])
    dark = x.min(axis=2)
    # 雾:暗通道整体偏高 + 全局对比度低
    haze = float(np.clip((dark.mean() - 0.15) / 0.5, 0, 1)) * float(np.clip(1.0 - lum.std() / 0.25, 0, 1))
    # 低光:亮度均值低
    low_light = float(np.clip((0.35 - lum.mean()) / 0.35, 0, 1))
    # 噪声:中值残差
    med = np.asarray(Image.fromarray(img).filter(ImageFilter.MedianFilter(3))).astype(np.float64) / 255.0
    noise = float(np.clip(np.abs(x - med).mean() / 0.04, 0, 1))
    # 模糊:梯度能量低
    gy, gx = np.gradient(lum)
    blur = float(np.clip(1.0 - np.sqrt(gx**2 + gy**2).mean() / 0.05, 0, 1))
    return {
        "degradations": {"haze": round(haze, 3), "low_light": round(low_light, 3), "noise": round(noise, 3), "blur": round(blur, 3)},
        "stats": {"lum_mean": round(float(lum.mean()), 3), "lum_std": round(float(lum.std()), 3), "dark_channel_mean": round(float(dark.mean()), 3)},
        "diagnoser": "heuristic-v0-coupled",
    }


def _diagnose_v1_decoupled(img: np.ndarray) -> dict:
    """退化诊断:四个 0–1 强度,不做硬分类——让 agent 自己决策。
    以后换 VLM/专用诊断模型时保持同一 JSON 形状。

    **每个读数都在"其他退化已被抵消"的图上测**,这是 v1 相对 v0 的全部区别。
    v0 直接在原图上测四个统计量,结果三个读数互相污染(2026-09-03 实测,同一底图、
    同一雾强度 t=0.7,只是加了暗光,雾读数就从 0.725 掉到 0.065,掉 91%);而 `blur`
    在**干净参考图上就读 0.912** —— 它测的其实是"梯度能量低",在任何不带噪声的图上
    都成立,所以它是个反噪声读数,不是模糊读数。agent 唯一的观测是这个,于是把从不
    出现在最优链里的锐化用成了第二常用工具。

    v1 的解耦:
    - `low_light` 在**去雾后**的图上测亮度均值 —— 雾的大气光会把原图提亮;
    - `haze` 在**亮度归一后**的图上测「暗通道 5% 分位 / 大气光」 —— 暗光会压缩暗通道,
      且除以 A 抵消整体明暗。用暗通道地板而不是均值,是因为均值被彩色高饱和区抬高;
    - `noise` / `blur` 在**去雾且亮度归一后**的图上测,`blur` 用 grad/std 自归一,
      因此不再随明暗与对比度漂移。

    残余掩盖:同底图同雾强度加暗光,雾读数仍掉 22%(v0 是 91%)。已知未消除。

    标定常数在 `_DIAG`,由 `tests/test_diagnose.py` 守住。
    """
    from PIL import ImageFilter

    from .builtin import dcp_estimate, dehaze_dcp, lowlight_gamma

    x = img.astype(np.float64) / 255.0

    dehazed = dehaze_dcp(img)                       # 抵消雾,再量暗
    lum_dehazed = float(_luminance(dehazed).mean())
    low_light = float(np.clip((_DIAG["low_light_zero"] - lum_dehazed) / _DIAG["low_light_span"], 0, 1))

    brightened = lowlight_gamma(img)                # 抵消暗,再量雾(雾仍在)
    airlight, _ = dcp_estimate(brightened)
    dark_floor = float(np.quantile((brightened.astype(np.float64) / 255.0).min(axis=2), 0.05))
    haze_feat = dark_floor / max(float(np.mean(airlight)), 1e-3)
    haze = float(np.clip((haze_feat - _DIAG["haze_zero"]) / _DIAG["haze_span"], 0, 1))

    normalised = lowlight_gamma(dehazed)            # 雾与暗都抵消,再量噪声与模糊
    n = normalised.astype(np.float64) / 255.0
    smoothed = np.asarray(Image.fromarray(normalised).filter(ImageFilter.MedianFilter(3)))
    med = smoothed.astype(np.float64) / 255.0
    noise = float(np.clip(np.abs(n - med).mean() / _DIAG["noise_span"], 0, 1))

    # 模糊在**中值滤波后**的图上量:噪声的高频会假装"锐利",不去掉就测不出模糊。
    lum_s = _luminance(smoothed)
    gy, gx = np.gradient(lum_s)
    grad = float(np.sqrt(gx**2 + gy**2).mean())
    lap = float(np.abs(np.gradient(gx, axis=1) + np.gradient(gy, axis=0)).mean())
    hf_ratio = lap / max(grad, 1e-9)
    blur = float(np.clip((_DIAG["blur_zero"] - hf_ratio) / _DIAG["blur_span"], 0, 1))

    return {
        "degradations": {"haze": round(haze, 3), "low_light": round(low_light, 3), "noise": round(noise, 3), "blur": round(blur, 3)},
        "stats": {
            "lum_mean_dehazed": round(lum_dehazed, 3),
            "dark_floor_over_airlight": round(haze_feat, 3),
            "high_freq_ratio": round(hf_ratio, 4),
        },
        "diagnoser": "heuristic-v1-decoupled",
    }


def cmd_diagnose(inp: Path) -> int:
    d = diagnose_array(_load(inp))
    d["input_sha"] = _sha(inp)
    _trace({"kind": "diagnose", "input_sha": d["input_sha"], "result": d["degradations"]})
    print(json.dumps(d, ensure_ascii=False, indent=2))
    return 0


# ---------------- remote backend ----------------

#: 工具服务的地址。**工具在哪是一个配置,不是一处代码**:
#:
#:   循环跑在有卡的机器上   RESTORE_TOOL_SERVER=http://127.0.0.1:8710   零网络
#:   循环跑在别的机器上     RESTORE_TOOL_SERVER=http://<tailnet-ip>:8710 走 tailnet
#:
#: 这个形状不是凭空来的:JarvisArt 也是因为 Lightroom 只能跑在特定机器上,
#: 才做了 Agent-to-Lightroom 的 server-client 协议来支持多机多卡。
TOOL_SERVER = os.environ.get("RESTORE_TOOL_SERVER", "http://127.0.0.1:8710")

#: 单次调用的墙钟上限。真专家模型第一次调用要加载权重,给足。
REMOTE_TIMEOUT_S = float(os.environ.get("RESTORE_REMOTE_TIMEOUT", "600"))

#: 重试次数。实测 tailnet 走 DERP 中继时会丢包(macbook -> 4070:132ms,50% loss),
#: 而一次网络抖动不该被记成"这个工具把图弄坏了"。重试穷尽后**抛错**,不返回原图。
REMOTE_RETRIES = int(os.environ.get("RESTORE_REMOTE_RETRIES", "3"))


def remote_tool_url(backend: str, server: str | None = None) -> tuple[str, str]:
    """`remote:<tool>` -> (工具名, 完整 URL)。纯函数,便于测试。"""
    tool = backend.split(":", 1)[1].strip()
    if not tool or "/" in tool:
        raise ValueError(f"bad remote backend spec: {backend!r} (want 'remote:<tool>')")
    base = (server or TOOL_SERVER).rstrip("/")
    return tool, f"{base}/run/{tool}"


def remote_tools(server: str | None = None, timeout: float = 10.0) -> set[str]:
    """服务端此刻真的能跑哪些工具。`doctor` 用它,所以它必须问服务端,
    而不是相信注册表 —— 注册表说的是「打算接」,服务端说的是「接上了」。"""
    import json as _json
    import urllib.request

    base = (server or TOOL_SERVER).rstrip("/")
    with urllib.request.urlopen(f"{base}/tools", timeout=timeout) as resp:  # noqa: S310
        return set(_json.loads(resp.read().decode())["tools"])


def run_remote_tool(backend: str, inp: Path, out: Path) -> None:
    """把图 POST 给工具服务,把结果写到 out。任何失败都抛错。

    只发字节,不发文件名 —— 与 docker 后端同样的理由:JarvisIR 的
    `process_image` 在输入路径里 grep "fog"/"night"/"snow" 来限制可用工具,
    而数据集按场景命名目录,那就是一条从路径泄漏答案的通道。
    """
    import time as _time
    import urllib.error
    import urllib.request

    tool, url = remote_tool_url(backend)
    payload = inp.read_bytes()
    last = None
    for attempt in range(1, REMOTE_RETRIES + 1):
        request = urllib.request.Request(  # noqa: S310
            url, data=payload, method="POST",
            headers={"Content-Type": "image/png", "Content-Length": str(len(payload))},
        )
        try:
            with urllib.request.urlopen(request, timeout=REMOTE_TIMEOUT_S) as resp:  # noqa: S310
                body = resp.read()
            if not body:
                raise RuntimeError(f"remote tool {tool!r} returned 200 with an empty body")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(body)
            return
        except urllib.error.HTTPError as exc:
            # 4xx 是我们自己的错(工具名不对、服务端没这个工具),重试无意义
            detail = exc.read()[:300].decode("utf-8", "replace")
            if exc.code < 500:
                raise RuntimeError(f"remote tool {tool!r} rejected: HTTP {exc.code} {detail}") from exc
            last = f"HTTP {exc.code} {detail}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = repr(exc)
        if attempt < REMOTE_RETRIES:
            _time.sleep(2.0 * attempt)
    raise RuntimeError(
        f"remote tool {tool!r} failed {REMOTE_RETRIES}x at {url}: {last}. "
        f"不把输入当输出传下去 —— 一次网络失败不是「这个工具没用」。"
    )


# ---------------- docker backend ----------------

#: 容器内的固定路径。宿主机的文件名**不进容器** —— 这是有意的,不是洁癖。
#: JarvisIR 的 `restoration_toolkit.process_image` 在输入路径字符串里 grep
#: "fog" / "night" / "snow" 来决定哪些工具可用(`tool_dict` 那段);数据集按
#: 场景命名目录是常规做法,于是"该用哪个工具"有一部分是路径给的,不是模型学的。
#: 这个仓自己踩过同一个坑的另一半(任务 id 泄漏退化组合,已改成不可读哈希),
#: 所以工具层这里不留第二条泄漏通道:容器只看得到 input.png 和 output.png。
DOCKER_IN = "/work/in/input.png"
DOCKER_OUT = "/work/out/output.png"

#: 单个工具的墙钟上限。真专家模型在 CPU 上会很慢,给足;但不能没有 ——
#: 一个挂住的容器会让整个 evolve step 无限期停在那里,而 reef 那侧看起来只是"还在跑"。
DOCKER_TIMEOUT_S = int(os.environ.get("RESTORE_DOCKER_TIMEOUT", "900"))

#: 有卡时置 1。默认关,因为"悄悄跑在 CPU 上"比"报错说没有卡"更难发现。
DOCKER_GPUS = os.environ.get("RESTORE_DOCKER_GPUS", "")


def docker_argv(backend: str, in_dir: Path, out_dir: Path) -> list[str]:
    """`docker:<image> [cmd...]` -> 完整 argv。纯函数,不跑 docker,便于测试。

    镜像的契约只有一条:读 DOCKER_IN,把结果写到 DOCKER_OUT。接口与 builtin
    后端一致(一进一出),所以换后端不动循环、不动缓存、不动 trace。
    """
    spec = backend.split(":", 1)[1].strip()
    if not spec:
        raise ValueError(f"bad docker backend spec: {backend!r} (want 'docker:<image> [cmd...]')")
    image, *cmd = spec.split()
    argv = ["docker", "run", "--rm"]
    if DOCKER_GPUS:
        argv += ["--gpus", DOCKER_GPUS]
    argv += [
        "-v", f"{in_dir.resolve()}:{Path(DOCKER_IN).parent}:ro",
        "-v", f"{out_dir.resolve()}:{Path(DOCKER_OUT).parent}",
        image, *cmd, DOCKER_IN, DOCKER_OUT,
    ]
    return argv


def run_docker_tool(backend: str, inp: Path, out: Path) -> None:
    """在容器里跑一个工具。失败一律抛异常 —— 绝不把输入当输出传下去。

    JarvisIR 那边模型不存在时是 `print(...)` 加 `continue`:链继续往下走,最后
    返回一个"处理过"的路径,而 reward 照常计算。于是训练信号里"这个工具没用"
    和"这个工具没实现"完全不可区分 —— `restormer` 就在 ALL_TOOLS 里,却既没有
    执行分支也不在 all_model_paths 里。这里不留这条路。
    """
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        raise RuntimeError(
            f"docker backend {backend!r} 需要 docker,但 PATH 上没有。"
            f"没有 docker 就该说没有,而不是让这一步看起来跑过了。"
        )
    with tempfile.TemporaryDirectory(prefix="restore-docker-") as scratch:
        in_dir, out_dir = Path(scratch) / "in", Path(scratch) / "out"
        in_dir.mkdir(), out_dir.mkdir()
        (in_dir / Path(DOCKER_IN).name).write_bytes(inp.read_bytes())
        argv = docker_argv(backend, in_dir, out_dir)
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=DOCKER_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"docker tool timed out after {DOCKER_TIMEOUT_S}s: {' '.join(argv)}") from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"docker tool exited {proc.returncode}: {' '.join(argv)}\n"
                f"stderr: {(proc.stderr or proc.stdout)[:500]}"
            )
        produced = out_dir / Path(DOCKER_OUT).name
        if not produced.exists():
            raise RuntimeError(
                f"docker tool exited 0 but wrote no {DOCKER_OUT}: {' '.join(argv)}\n"
                f"stdout: {proc.stdout[:300]}"
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(produced.read_bytes())


# ---------------- run ----------------

def cmd_run(tool: str, inp: Path, out: Path) -> int:
    reg = _enabled_registry()
    if tool not in reg:
        known_but_off = tool in _registry()
        print(json.dumps({
            # 两种拒绝分开报,因为它们要人做的事不同:一个是打错名字,
            # 一个是这个工具还没接上(去 registry 里把 enabled 打开并跑 doctor)。
            "status": "disabled_tool" if known_but_off else "unknown_tool",
            "tool": tool,
            "known": sorted(reg),
            **({"hint": "declared in registry.yaml but enabled: false - it is not runnable, "
                        "so this step did not happen"} if known_but_off else {}),
        }), file=sys.stderr)
        return 2
    in_sha = _sha(inp)
    cache_key = f"{tool}-{in_sha}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{cache_key}.png"
    t0 = time.time()
    if cached.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(cached.read_bytes())
        _trace({"kind": "run", "tool": tool, "input_sha": in_sha, "output_sha": _sha(out), "status": "cache_hit", "ms": 0})
        print(json.dumps({"status": "cache_hit", "tool": tool, "output": str(out)}))
        return 0
    backend = reg[tool]["backend"]
    try:
        if backend.startswith("builtin:"):
            from . import builtin  # type: ignore

            fn = getattr(builtin, backend.split(":", 1)[1])
            result = fn(_load(inp))
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result).save(out)
        elif backend.startswith("docker:"):
            run_docker_tool(backend, inp, out)
        elif backend.startswith("remote:"):
            run_remote_tool(backend, inp, out)
        else:
            raise ValueError(f"bad backend spec: {backend}")
        if not out.exists():
            raise RuntimeError(f"{tool}: backend returned without writing {out}")
        cached.write_bytes(out.read_bytes())
    except Exception as e:
        _trace({"kind": "run", "tool": tool, "input_sha": in_sha, "status": "failed", "error": repr(e), "ms": int((time.time() - t0) * 1000)})
        print(json.dumps({"status": "failed", "tool": tool, "error": repr(e)}), file=sys.stderr)
        return 1
    ms = int((time.time() - t0) * 1000)
    _trace({"kind": "run", "tool": tool, "input_sha": in_sha, "output_sha": _sha(out), "status": "ok", "ms": ms})
    print(json.dumps({"status": "ok", "tool": tool, "output": str(out), "ms": ms}))
    return 0


# ---------------- submit ----------------

def cmd_submit(path: Path) -> int:
    """`restore submit OUT` —— agent 显式声明"我做完了,判这一张"。

    在此之前判分取的是**最后一次成功的 `restore run`**,也就是把"停手"当成
    "不再发命令"的副作用。那让两件事无法分开:

      * 它认为这一步是最好的,所以停了
      * 它还想继续,但轮次用完了 / 它跑偏了 / 它把图弄坏了还没发现

    而 2026-09-04 穷举 3096 个决策点得到的结论正是:**不知道该在哪停,值 23% 的
    可达分数**(12 题贪心合计 5.6092,同一轨迹上出现过的最高分合计 7.2625;
    `noise_01` 走到过 0.8276 最后停在 0.0568)。一个只能被动观察的行为没法要求
    agent 改进它,所以这里把它变成一个动作:agent 必须说出"停在这一张"。

    不提交也是数据 —— judge 会退回最后一步并把这件事记下来,于是"主动提交率"
    和"提交的那一步是不是轨迹上最好的一步"都变成可以直接量的东西。
    """
    if not path.exists():
        print(json.dumps({"status": "no_such_output", "path": str(path)}), file=sys.stderr)
        return 1
    sha = _sha(path)
    _trace({"kind": "submit", "output_sha": sha, "path": str(path)})
    print(json.dumps({"status": "submitted", "output_sha": sha, "path": str(path)}, ensure_ascii=False))
    return 0


# ---------------- doctor ----------------

def check_tool(name: str, spec: dict) -> tuple[bool, str]:
    """这个工具此刻真的能跑吗?(不跑它,只验可执行性)"""
    backend = spec.get("backend", "")
    if backend.startswith("builtin:"):
        from . import builtin

        fn = backend.split(":", 1)[1]
        return (True, "builtin ok") if hasattr(builtin, fn) else (False, f"builtin.{fn} 不存在")
    if backend.startswith("remote:"):
        tool, url = remote_tool_url(backend)
        try:
            available = remote_tools()
        except Exception as exc:  # noqa: BLE001 - doctor 的职责就是把不可达如实报出来
            return False, f"工具服务 {TOOL_SERVER} 问不到:{exc!r}"
        return (True, f"服务端有 {tool}") if tool in available else (False, f"服务端没有 {tool}(有:{sorted(available)})")
    if backend.startswith("docker:"):
        import shutil

        if shutil.which("docker") is None:
            return False, "PATH 上没有 docker"
        image = backend.split(":", 1)[1].strip().split()[0]
        proc = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True)
        return (True, f"镜像 {image} 在本地") if proc.returncode == 0 else (False, f"镜像 {image} 不在本地")
    return False, f"无法识别的 backend: {backend!r}"


def cmd_doctor() -> int:
    """逐个验证注册表里声明的工具是否真的可执行。

    这条命令存在的理由是一个具体的 bug:JarvisIR 的 ALL_TOOLS 里有 13 个工具,
    其中 `restormer` 既没有执行分支也没有权重路径,选中它就是静默跳过。没有
    任何东西会说出这件事,而模型会把它学成"restormer 没用"。

    退出码:0 = 所有**已启用**的工具都能跑;1 = 有启用的工具跑不了。
    未启用的工具单独列出,不影响退出码 —— 它们本来就说了自己不可用。
    """
    reg = _registry()
    broken, ok, disabled = [], [], []
    for name, spec in sorted(reg.items()):
        is_on = spec.get("enabled", True)
        good, why = check_tool(name, spec)
        (ok if good else broken).append((name, why)) if is_on else disabled.append((name, why, good))

    for name, why in ok:
        print(f"  ok       {name:22s} {why}")
    for name, why in broken:
        print(f"  BROKEN   {name:22s} {why}", file=sys.stderr)
    for name, why, good in disabled:
        print(f"  disabled {name:22s} {why}{'  (可用,打开 enabled 即可)' if good else ''}")

    print(f"\n启用 {len(ok) + len(broken)} 个:可跑 {len(ok)},跑不了 {len(broken)};未启用 {len(disabled)} 个")
    if broken:
        print("有工具声明为启用却跑不了 —— 这正是 agent 会当成「这个工具没用」的那种失败。", file=sys.stderr)
        return 1
    return 0


# ---------------- score ----------------

def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 99.0 if mse == 0 else float(10 * np.log10(255.0**2 / mse))


#: 有参考时的分数刻度:相对退化输入的 PSNR 改善量,除以 IMPROVEMENT_SPAN 归一。
#:
#: v0 用的是绝对刻度 `(PSNR-15)/20`,下限 15 dB **高于任务起点** —— 12 个任务里 8 个
#: 的退化图本身就低于 15 dB(三个评测任务是 13.19 / 12.76 / 8.59 dB)。于是 8.6 dB 和
#: 14.9 dB 都记 0.0:一整段真实差异对闸门不可见。实测代价:18 个 episode 分数里 7 个
#: (39%)钉死在 0,9 次逐题比较里 5 次至少一边是 0 —— 那些"平局"不是平局。而且单个
#: 工具用一次几乎全得 0.0000,agent 的第一步永远拿不到反馈。
#:
#: 改成改善量之后:identity(什么都不做)恒等于 0.0,而任何真实改善都有梯度。
#: 逐题看这只是减掉一个常数(单调变换),所以**不改变闸门的逐题胜负**,只是把被地板
#: 压平的那段还原出来。SPAN=15 dB 的依据:各步数预算下最优链的实测改善量最大约 14 dB。
IMPROVEMENT_SPAN_DB = 15.0


def reference_score(psnr_out: float, psnr_in: float) -> float:
    """改善量归一到 0–1。判据只在这里定义,oracle 与 judge 都调它,不各自重推。"""
    return float(np.clip((psnr_out - psnr_in) / IMPROVEMENT_SPAN_DB, 0, 1))


def cmd_score(inp: Path, out: Path, ref: Path | None) -> int:
    """有参考(合成任务,答案已知)→ 相对输入的 PSNR 改善量归一到 0-1;无参考 → 诊断分改善量。
    两者都记进 verifier 记录;无参考分数是已知不可靠的(见 judge-lab 结论),标 `reliability: low`。"""
    if not out.exists():
        _trace({"kind": "score", "status": "no_output"})
        print(json.dumps({"status": "no_output"}), file=sys.stderr)
        print("REEF_SCORE=0.0")
        return 1
    o = _load(out)
    record: dict = {"episode": EPISODE_ID, "input_sha": _sha(inp), "output_sha": _sha(out)}
    if ref is not None and ref.exists():
        r = _load(ref)
        if r.shape != o.shape:
            o_img = Image.fromarray(o).resize((r.shape[1], r.shape[0]))
            o = np.asarray(o_img)
        i = _load(inp)
        if r.shape != i.shape:
            i = np.asarray(Image.fromarray(i).resize((r.shape[1], r.shape[0])))
        psnr, psnr_in = _psnr(o, r), _psnr(i, r)
        score = reference_score(psnr, psnr_in)
        record.update({
            "mode": "reference",
            "psnr": round(psnr, 3),
            "psnr_input": round(psnr_in, 3),
            "gain_db": round(psnr - psnr_in, 3),
            "score": round(score, 4),
            "reliability": "high",
        })
    else:
        before, after = diagnose_array(_load(inp))["degradations"], diagnose_array(o)["degradations"]
        improve = float(np.mean([before[k] - after[k] for k in before]))
        score = float(np.clip(0.5 + improve, 0, 1))
        record.update({"mode": "no_reference", "before": before, "after": after, "score": round(score, 4), "reliability": "low"})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{EPISODE_ID}.verdict.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / f"{EPISODE_ID}.output.png").write_bytes(out.read_bytes())
    _trace({"kind": "score", **{k: v for k, v in record.items() if k in ("mode", "score", "psnr")}})
    print(json.dumps(record, ensure_ascii=False))
    print(f"REEF_SCORE={score:.4f}")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd, args = argv[0], argv[1:]
    if cmd == "catalog":
        return cmd_catalog()
    if cmd == "doctor":
        return cmd_doctor()
    if cmd == "submit" and len(args) == 1:
        return cmd_submit(Path(args[0]))
    if cmd == "diagnose" and len(args) == 1:
        return cmd_diagnose(Path(args[0]))
    if cmd == "run" and len(args) == 3:
        return cmd_run(args[0], Path(args[1]), Path(args[2]))
    if cmd == "score" and len(args) >= 2:
        ref = Path(args[args.index("--ref") + 1]) if "--ref" in args else None
        return cmd_score(Path(args[0]), Path(args[1]), ref)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


def _entry() -> None:  # console_scripts 入口
    sys.exit(main(sys.argv[1:]))
