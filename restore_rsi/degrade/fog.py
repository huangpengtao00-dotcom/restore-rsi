"""雾(fog)退化:Koschmieder 大气散射 I = J·t + A·(1−t),t = exp(−β·s·d)。

重实现自 JarvisIR `degradation_synthesis/fog/fog_simulator.py` 的模型结构(MIT 许可),不复用其代码:
- d 是归一化距离 ∈[0,1](0 近 1 远),来源见 depth.py(Depth-Anything-V2 或合成),manifest 标 depth_source;
- s = depth_scale 是 JarvisIR 里写死的 2.0,这里显式记录;
- A = 灰体大气光 A_gray + 逐通道色偏(JarvisIR:U[0.25,1.0] + U[-0.025,0.025]);为了像真雾,这里
  A_gray 取 U[0.72,0.95]——低 A 是"黑雾",不是探针想要的;
- JarvisIR 雾合成器内部还随机叠 gamma/噪声/JPEG,这里**不**做:叠加交给 pipeline 显式组合。

severity 映射:β 0 / 0.35 / 0.70 / 1.20 / 2.00(JarvisIR 原范围 U[0.3,1.5],4 档略超以覆盖浓雾),
按 seed 乘 exp(U(−0.15,0.15)) 抖动;severity 0 → β=0 → t≡1 → 位级恒等。
"""

from __future__ import annotations

import numpy as np

from .base import as_jsonable, check_severity, jitter, to_float, to_u8
from .depth import synthetic_depth

NAME = "fog"
SURROGATE = False
NEEDS_DEPTH = True
MODEL = "Koschmieder atmospheric scattering with depth (JarvisIR fog_simulator structure, reimplemented)"

_BETA = {1: 0.35, 2: 0.70, 3: 1.20, 4: 2.00}
DEPTH_SCALE = 2.0


def params_for_severity(severity: int, rng: np.random.Generator) -> dict:
    severity = check_severity(severity)
    j_beta = jitter(rng, 0.15)
    a_gray = float(rng.uniform(0.72, 0.95))
    a_shift = rng.uniform(-0.025, 0.025, 3)
    beta = 0.0 if severity == 0 else _BETA[severity] * j_beta
    A = np.clip(a_gray + a_shift, 0.0, 1.0)
    return as_jsonable({"beta": beta, "A": A, "depth_scale": DEPTH_SCALE})


def transmission(dist: np.ndarray, params: dict) -> np.ndarray:
    return np.exp(-float(params["beta"]) * float(params["depth_scale"]) * dist)


def apply(img_u8: np.ndarray, params: dict, rng: np.random.Generator, depth: np.ndarray | None = None) -> np.ndarray:
    """depth=None 时用 rng 生成合成深度(自包含用法);pipeline 总是显式传 depth 并记录来源。"""
    if float(params["beta"]) == 0.0:
        return img_u8.copy()
    x = to_float(img_u8)
    if depth is None:
        depth = synthetic_depth(x.shape[:2], rng)
    if depth.shape != x.shape[:2]:
        raise ValueError(f"depth shape {depth.shape} != image {x.shape[:2]}")
    t = transmission(depth, params)[..., None]
    A = np.asarray(params["A"], dtype=np.float64).reshape(1, 1, 3)
    return to_u8(x * t + A * (1.0 - t))
