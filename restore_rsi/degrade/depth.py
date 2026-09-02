"""深度来源:Depth-Anything-V2-Small(真深度,优先)或可复现的合成深度(替身)。

约定输出 `dist`:float64 HxW,∈[0,1],**0=近 1=远**(距离感,不是 disparity)。
- Depth-Anything 输出的是相对逆深度(近大远小),min-max 归一后取 1-x 得到 dist —— 与 JarvisIR
  fog_simulator 里 `t = exp(-(1-depth)·2β)` 的用法一致。
- 合成深度 = 竖直线性梯度(上远下近)+ 低频余弦噪声,只依赖传入的 rng,同 seed 位级同图。

不静默降级:`resolve_backend("auto")` 加载模型失败时返回 source="synthetic" 并把失败原因原文放进
info["fallback_reason"],由调用方写进 manifest;`resolve_backend("depth_anything")` 失败直接抛。
模型权重走 HF cache(~/.cache/huggingface),不进仓;国内网络可设 HF_ENDPOINT=https://hf-mirror.com。
"""

from __future__ import annotations

import sys

import numpy as np

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
SOURCE_DA = "depth_anything_v2_small"
SOURCE_SYN = "synthetic"
MODES = ("auto", "synthetic", "depth_anything")


class DepthUnavailable(RuntimeError):
    pass


def synthetic_depth(shape: tuple[int, int], rng: np.random.Generator, n_waves: int = 6) -> np.ndarray:
    """线性梯度 + 低频噪声的合成 dist 图(0 近 1 远),归一到 [0,1]。"""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    yy, xx = yy / max(h - 1, 1), xx / max(w - 1, 1)
    field = 1.0 - yy  # 顶部(天空/远景)远,底部近
    for _ in range(n_waves):
        fx, fy = rng.uniform(0.3, 2.0, 2)  # 每边 0.3~2 个周期 → 低频
        phase = rng.uniform(0.0, 2 * np.pi)
        amp = rng.uniform(0.03, 0.12)
        field = field + amp * np.cos(2 * np.pi * (fx * xx + fy * yy) + phase)
    field = field - field.min()
    return field / max(field.max(), 1e-6)


class DepthAnythingV2:
    source = SOURCE_DA

    def __init__(self, model_id: str = DEPTH_MODEL_ID):
        try:
            import torch  # noqa: F401
            from transformers import pipeline
        except Exception as e:  # ImportError 或 transformers 内部导入失败
            raise DepthUnavailable(f"import failed: {type(e).__name__}: {e}") from e
        try:
            self.pipe = pipeline(task="depth-estimation", model=model_id, device="cpu")
        except Exception as e:
            raise DepthUnavailable(f"pipeline load failed for {model_id}: {type(e).__name__}: {e}") from e
        self.model_id = model_id

    def __call__(self, img_u8: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        h, w = img_u8.shape[:2]
        with torch.no_grad():
            out = self.pipe(Image.fromarray(img_u8))
        pred = out["predicted_depth"]
        if pred.ndim == 2:
            pred = pred[None]
        pred = torch.nn.functional.interpolate(pred[None].float(), size=(h, w), mode="bicubic", align_corners=False)[0, 0]
        disp = pred.cpu().numpy().astype(np.float64)
        disp = (disp - disp.min()) / max(disp.max() - disp.min(), 1e-6)
        return 1.0 - disp


def resolve_backend(mode: str) -> tuple[DepthAnythingV2 | None, str, dict]:
    """返回 (estimator|None, source, info)。source ∈ {depth_anything_v2_small, synthetic}。"""
    if mode not in MODES:
        raise ValueError(f"depth mode must be one of {MODES}, got {mode!r}")
    if mode == "synthetic":
        return None, SOURCE_SYN, {"model": None, "reason": "requested synthetic"}
    try:
        est = DepthAnythingV2()
        return est, SOURCE_DA, {"model": est.model_id}
    except DepthUnavailable as e:
        if mode == "depth_anything":
            raise
        print(f"[degrade] Depth-Anything unavailable, falling back to synthetic depth: {e}", file=sys.stderr)
        return None, SOURCE_SYN, {"model": None, "fallback_reason": str(e)}


def depth_to_u16(dist: np.ndarray) -> np.ndarray:
    return (np.clip(dist, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def depth_from_u16(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float64) / 65535.0
