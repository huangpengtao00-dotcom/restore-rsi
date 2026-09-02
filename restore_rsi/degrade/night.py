"""低光(night)退化:反向 ISP 到 RAW 域 → 光子衰减 + shot/read 噪声 → 正向 ISP 回 sRGB。

重实现自 JarvisIR `degradation_synthesis/night/night_simulator.py` 的**模型结构**(源头是 MAET, Cui et al.
ICCV'21 的 unprocess/process 链;JarvisIR 仓 MIT 许可),不复用其代码。链路逐步对应:

  sRGB ─inverse tone map (smoothstep⁻¹)─▶ inverse gamma (x^γ) ─▶ sRGB→camRGB (CCM) ─▶ inverse WB
       ─▶ × darkness(光子数下降) ─▶ + N(0, x·shot + read)(RAW 噪声) ─▶ + U(±quant_amp)(量化噪声)
       ─▶ WB ─▶ camRGB→sRGB (CCM⁻¹) ─▶ gamma (x^(1/γ)) ─▶ sRGB

与原实现的有意差别(都记在 params 里,不是隐含行为):
- 全部随机量(γ、CCM 序号、三组增益、darkness、shot/read、量化幅度)由 params 显式给出,apply 内只剩噪声抽样;
- shot/read 用"有效方差系数"表示(原实现是 shot·0.1 / read·0.1,这里把 0.1 折进数值);
- 原实现反了 tone map 但**没有**正向 tone map 回去(MAET 同),这里照原样保留该结构,不补;
- darkness==1 且噪声为 0 时(severity 0)apply 直接返回输入副本:没有光子损失也没有传感器噪声就没有退化。

severity 映射(1→4 逐档更暗更噪;3–4 落在 JarvisIR 原配置 darkness∈[0.07,0.15] 的区间):
  darkness  0.35 / 0.18 / 0.11 / 0.07      shot=read(有效方差系数) 5e-6 / 2e-5 / 8e-5 / 2.5e-4
其余量按 JarvisIR 原范围抽样:γ∈U[2,3.5],red∈U[1.9,2.4],blue∈U[1.5,1.9],rgb_gain∈U[0.75,0.95],量化 bits∈{2,4,6}。
"""

from __future__ import annotations

import numpy as np

from .base import as_jsonable, check_severity, jitter, to_float, to_u8

NAME = "night"
SURROGATE = False
NEEDS_DEPTH = False
MODEL = "inverse-ISP low-light (MAET/JarvisIR night_simulator structure, reimplemented)"

# 四组相机 XYZ→camRGB 矩阵(MAET/Unprocessing 里的常用相机矩阵)与 sRGB→XYZ(D65)。
XYZ2CAMS = np.array(
    [
        [[1.0234, -0.2969, -0.2266], [-0.5625, 1.6328, -0.0469], [-0.0703, 0.2188, 0.6406]],
        [[0.4913, -0.0541, -0.0202], [-0.613, 1.3513, 0.2906], [-0.1564, 0.2151, 0.7183]],
        [[0.838, -0.263, -0.0639], [-0.2887, 1.0725, 0.2496], [-0.0627, 0.1427, 0.5438]],
        [[0.6596, -0.2079, -0.0562], [-0.4782, 1.3016, 0.1933], [-0.097, 0.1581, 0.5181]],
    ],
    dtype=np.float64,
)
RGB2XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
    dtype=np.float64,
)

# severity → (darkness, shot, read);severity 0 单独处理
_SEV = {
    1: (0.35, 5e-6, 5e-6),
    2: (0.18, 2e-5, 2e-5),
    3: (0.11, 8e-5, 8e-5),
    4: (0.07, 2.5e-4, 2.5e-4),
}
_EPS = 1e-8


def rgb2cam_matrix(ccm_idx: int) -> np.ndarray:
    m = XYZ2CAMS[ccm_idx] @ RGB2XYZ
    return m / m.sum(axis=1, keepdims=True)  # 行归一,白色保持白色


def params_for_severity(severity: int, rng: np.random.Generator) -> dict:
    """先抽全部抖动/相机量,再查档位表 —— 抽样顺序与 severity 无关,同 rng 状态下各档共用同一组抖动。"""
    severity = check_severity(severity)
    j_dark = jitter(rng, 0.12)
    j_noise = jitter(rng, 0.25)
    gamma = float(rng.uniform(2.0, 3.5))
    ccm_idx = int(rng.integers(0, len(XYZ2CAMS)))
    red_gain = float(rng.uniform(1.9, 2.4))
    blue_gain = float(rng.uniform(1.5, 1.9))
    rgb_gain = float(rng.uniform(0.75, 0.95))
    bits = int(rng.choice([2, 4, 6]))
    if severity == 0:
        darkness, shot, read, quant_amp = 1.0, 0.0, 0.0, 0.0
    else:
        d, s, r = _SEV[severity]
        darkness, shot, read = d * j_dark, s * j_noise, r * j_noise
        quant_amp = 1.0 / (255.0 * bits)
    return as_jsonable(
        {
            "gamma": gamma,
            "ccm_idx": ccm_idx,
            "rgb_gain": rgb_gain,
            "red_gain": red_gain,
            "blue_gain": blue_gain,
            "darkness": darkness,
            "shot_noise": shot,
            "read_noise": read,
            "quant_amp": quant_amp,
            "quant_bits": bits,
        }
    )


def is_identity(params: dict) -> bool:
    return params["darkness"] == 1.0 and params["shot_noise"] == 0.0 and params["read_noise"] == 0.0 and params["quant_amp"] == 0.0


def apply(img_u8: np.ndarray, params: dict, rng: np.random.Generator) -> np.ndarray:
    if is_identity(params):
        return img_u8.copy()
    x = to_float(img_u8)
    gamma = float(params["gamma"])
    rgb2cam = rgb2cam_matrix(int(params["ccm_idx"]))
    cam2rgb = np.linalg.inv(rgb2cam)
    red, blue, rgb_gain = float(params["red_gain"]), float(params["blue_gain"]), float(params["rgb_gain"])

    # ---- unprocess: sRGB → RAW ----
    x = 0.5 - np.sin(np.arcsin(1.0 - 2.0 * np.clip(x, 0.0, 1.0)) / 3.0)  # smoothstep⁻¹
    x = np.maximum(x, _EPS) ** gamma  # inverse gamma → 线性
    x = x @ rgb2cam.T  # sRGB → camRGB
    x = x * (np.array([1.0 / red, 1.0, 1.0 / blue]) * rgb_gain)  # inverse white balance
    # ---- RAW 域退化 ----
    x = x * float(params["darkness"])  # 光子数下降
    var = np.maximum(x * float(params["shot_noise"]) + float(params["read_noise"]), _EPS)
    x = x + rng.normal(0.0, 1.0, x.shape) * np.sqrt(var)  # shot + read 噪声
    qa = float(params["quant_amp"])
    if qa > 0:
        x = x + rng.uniform(-qa, qa, x.shape)  # 量化噪声
    # ---- process: RAW → sRGB ----
    x = x * np.array([red, 1.0, blue])  # white balance
    x = x @ cam2rgb.T  # camRGB → sRGB
    x = np.maximum(x, _EPS) ** (1.0 / gamma)  # gamma
    return to_u8(x)
