"""退化族的公共契约与像素工具。

每个退化族 = 一个模块,暴露:
  NAME            族名(manifest 里 applied_order 用)
  SURROGATE       True 表示是"替身"简化模型(非物理),manifest 会标出
  MODEL           一句话说明物理模型出处
  NEEDS_DEPTH     是否需要深度图(fog)
  params_for_severity(severity, rng) -> dict    severity∈{0..4} → 全部数值参数(含 rng 抖动;
                                                 **同一 rng 状态下,不同 severity 抽到同一组抖动因子**,
                                                 这样按 seed 固定后 severity 扫描是单调的)
  apply(img_u8, params, rng[, depth]) -> img_u8  纯函数;severity 0 的参数下位级恒等

约定:
- 图像 uint8 HxWx3 sRGB;内部 float64 [0,1];回 uint8 用 to_u8(四舍五入,不做抖动)。
- apply 内所有随机性只来自传入的 rng(np.random.Generator);不读全局随机状态。
"""

from __future__ import annotations

import numpy as np

SEVERITIES = (0, 1, 2, 3, 4)


def to_float(img_u8: np.ndarray) -> np.ndarray:
    if img_u8.dtype != np.uint8 or img_u8.ndim != 3 or img_u8.shape[2] != 3:
        raise ValueError(f"expect uint8 HxWx3, got {img_u8.dtype} {img_u8.shape}")
    return img_u8.astype(np.float64) / 255.0


def to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def check_severity(severity: int) -> int:
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be in {SEVERITIES}, got {severity!r}")
    return int(severity)


def jitter(rng: np.random.Generator, spread: float) -> float:
    """对数域均匀抖动因子 exp(U(-spread, spread)),用于给同一档位的样本拉开差异。"""
    return float(np.exp(rng.uniform(-spread, spread)))


def as_jsonable(params: dict) -> dict:
    """把 numpy 标量/数组变成纯 Python,保证 manifest 可 json 且重放时读回同值。"""
    out = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            out[k] = [float(a) for a in v.ravel()]
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [float(a) if isinstance(a, (float, np.floating)) else a for a in v]
        else:
            out[k] = v
    return out
