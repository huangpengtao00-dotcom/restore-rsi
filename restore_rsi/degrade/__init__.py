"""记录参数、固定 seed、可叠加的退化探针生成器。

族注册表 FAMILIES:name → 模块(见 base.py 的契约)。链式叠加用 "night+fog" 这种写法(见 pipeline.parse_chain)。
"""

from __future__ import annotations

from . import fog, night, simple

FAMILIES = {
    night.NAME: night,
    fog.NAME: fog,
    simple.haze_simple.NAME: simple.haze_simple,
    simple.low_light_simple.NAME: simple.low_light_simple,
    simple.noise_simple.NAME: simple.noise_simple,
}

__all__ = ["FAMILIES", "night", "fog", "simple"]
