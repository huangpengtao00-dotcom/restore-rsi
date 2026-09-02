"""`simple` 族:tasks/make_tasks.py 里的三个简化退化(雾 / 低光 / 噪声),**替身**,非物理模型。

保留它们是为了和旧任务集对得上、以及做快速烟雾测试;manifest 里 SURROGATE=True 会标出。
真探针请用 night.py / fog.py。severity 0 全部位级恒等。

  haze_simple      I = J·τ + A(1−τ),τ = 1 − 0.75·s(无深度,全图均匀)
  low_light_simple I = J^γ · g,γ = 1 + 2.2·s,g = 1 − 0.4·s
  noise_simple     I = J + N(0, 0.12·s)
其中 s = severity/4 再乘一点 seed 抖动。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .base import as_jsonable, check_severity, to_float, to_u8


def _strength(severity: int, rng: np.random.Generator) -> float:
    severity = check_severity(severity)
    j = float(rng.uniform(0.9, 1.1))
    return 0.0 if severity == 0 else min(1.0, severity / 4.0 * j)


# ---- haze_simple ----
def _haze_params(severity, rng):
    s = _strength(severity, rng)
    A = float(rng.uniform(0.75, 0.95))
    return as_jsonable({"trans": 1.0 - 0.75 * s, "A": A})


def _haze_apply(img_u8, params, rng):
    if float(params["trans"]) == 1.0:
        return img_u8.copy()
    x = to_float(img_u8)
    tr = float(params["trans"])
    return to_u8(x * tr + float(params["A"]) * (1.0 - tr))


# ---- low_light_simple ----
def _ll_params(severity, rng):
    s = _strength(severity, rng)
    return as_jsonable({"gamma": 1.0 + 2.2 * s, "gain": 1.0 - 0.4 * s})


def _ll_apply(img_u8, params, rng):
    if float(params["gamma"]) == 1.0 and float(params["gain"]) == 1.0:
        return img_u8.copy()
    x = to_float(img_u8)
    return to_u8(x ** float(params["gamma"]) * float(params["gain"]))


# ---- noise_simple ----
def _noise_params(severity, rng):
    s = _strength(severity, rng)
    return as_jsonable({"sigma": 0.12 * s})


def _noise_apply(img_u8, params, rng):
    if float(params["sigma"]) == 0.0:
        return img_u8.copy()
    x = to_float(img_u8)
    return to_u8(x + rng.normal(0.0, float(params["sigma"]), x.shape))


def _family(name, params_fn, apply_fn):
    return SimpleNamespace(
        NAME=name,
        SURROGATE=True,
        NEEDS_DEPTH=False,
        MODEL=f"simplified surrogate from tasks/make_tasks.py ({name})",
        params_for_severity=params_fn,
        apply=apply_fn,
    )


haze_simple = _family("haze_simple", _haze_params, _haze_apply)
low_light_simple = _family("low_light_simple", _ll_params, _ll_apply)
noise_simple = _family("noise_simple", _noise_params, _noise_apply)
