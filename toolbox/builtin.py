"""无 GPU 的经典复原算子。目的不是效果最好,而是让整条自进化循环在笔记本上跑通;
真 JarvisIR 专家模型以后按同一接口(np.uint8 HxWx3 -> 同形)从 docker 后端接入。

每个算子都是纯函数、确定性(同输入同输出),这是缓存与可复现的前提。"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _to_pil(a: np.ndarray) -> Image.Image:
    return Image.fromarray(a)


def dehaze_dcp(img: np.ndarray, omega: float = 0.9, t_min: float = 0.15, win: int = 15) -> np.ndarray:
    """暗通道先验去雾(He et al. 2009 的简化版:最小滤波求暗通道,估 A,估 t,恢复)。"""
    x = img.astype(np.float64) / 255.0
    dark = x.min(axis=2)
    dark_min = np.asarray(_to_pil((dark * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(win))) / 255.0
    n = max(1, int(dark_min.size * 0.001))
    idx = np.argsort(dark_min.ravel())[-n:]
    A = x.reshape(-1, 3)[idx].mean(axis=0)
    t = 1.0 - omega * (x / np.maximum(A, 1e-3)).min(axis=2)
    t = np.asarray(_to_pil((np.clip(t, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(win // 2))) / 255.0
    t = np.clip(t, t_min, 1.0)[..., None]
    J = (x - A) / t + A
    return (np.clip(J, 0, 1) * 255 + 0.5).astype(np.uint8)


def lowlight_gamma(img: np.ndarray, target_mean: float = 0.45) -> np.ndarray:
    """自适应 gamma:把亮度均值拉到 target_mean。"""
    x = img.astype(np.float64) / 255.0
    lum = float((x @ np.array([0.299, 0.587, 0.114])).mean())
    lum = max(lum, 1e-3)
    gamma = np.log(target_mean) / np.log(lum) if lum < target_mean else 1.0
    gamma = float(np.clip(gamma, 0.3, 1.0))
    return (np.clip(x**gamma, 0, 1) * 255 + 0.5).astype(np.uint8)


def lowlight_clahe(img: np.ndarray, tiles: int = 8, clip: float = 3.0) -> np.ndarray:
    """分块直方图均衡(CLAHE 的朴素实现,作用在亮度通道)。"""
    x = img.astype(np.float64)
    lum = x @ np.array([0.299, 0.587, 0.114])
    h, w = lum.shape
    out = np.zeros_like(lum)
    th, tw = max(1, h // tiles), max(1, w // tiles)
    for i in range(0, h, th):
        for j in range(0, w, tw):
            blk = lum[i : i + th, j : j + tw]
            hist, edges = np.histogram(blk, bins=256, range=(0, 255))
            limit = clip * blk.size / 256
            excess = np.clip(hist - limit, 0, None).sum()
            hist = np.minimum(hist, limit) + excess / 256
            cdf = np.cumsum(hist)
            cdf = 255 * cdf / cdf[-1]
            out[i : i + th, j : j + tw] = np.interp(blk, edges[:-1], cdf)
    ratio = (out + 1) / (lum + 1)
    return (np.clip(x * ratio[..., None], 0, 255) + 0.5).astype(np.uint8)


def denoise_bilateral(img: np.ndarray, radius: int = 2, sigma_s: float = 2.0, sigma_r: float = 25.0) -> np.ndarray:
    """双边滤波(纯 numpy,窄窗口)。"""
    x = img.astype(np.float64)
    pad = np.pad(x, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    h, w, _ = x.shape
    acc = np.zeros_like(x)
    wsum = np.zeros((h, w, 1))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = pad[radius + dy : radius + dy + h, radius + dx : radius + dx + w]
            gs = np.exp(-(dx * dx + dy * dy) / (2 * sigma_s**2))
            gr = np.exp(-((shifted - x) ** 2).sum(axis=2, keepdims=True) / (2 * sigma_r**2))
            wgt = gs * gr
            acc += shifted * wgt
            wsum += wgt
    return (np.clip(acc / wsum, 0, 255) + 0.5).astype(np.uint8)


def denoise_median(img: np.ndarray, size: int = 3) -> np.ndarray:
    return np.asarray(_to_pil(img).filter(ImageFilter.MedianFilter(size)))


def sharpen_unsharp(img: np.ndarray, radius: int = 2, percent: int = 150) -> np.ndarray:
    return np.asarray(_to_pil(img).filter(ImageFilter.UnsharpMask(radius=radius, percent=percent)))
