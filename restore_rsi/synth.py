"""干净底图的合成:这个仓里**唯一**一份实现。

2026-09-06 从 judge-lab 搬进来。原先有三处各自 `sys.path.insert` 到
`~/research/judge-lab` 再 import 同一个函数(tasks/make_tasks.py、
restore_rsi/degrade/pipeline.py、tests/test_diagnose.py),于是这个仓在别人的机器上
造不出数据、跑不了那几项测试 —— 而在有那个目录的机器上一切正常,所以本地永远是绿的。
第一次跑 CI 就红了三处中的两处(第三处已先修),这正是"本地绿不算绿"。

现在只有一份实现,三处都从这里 import。真数据(UIEB / CleanBench)接进来时也只换这里。
"""

from __future__ import annotations

import numpy as np


def gen_test_image(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """梯度背景 + 8 个随机色块:直方图够丰富,四个诊断读数都测得动。

    与 judge-lab 的原实现**逐字节一致**已验证:6 组 (seed x size) 加同一个 rng 上
    连续 5 次调用,`np.array_equal` 全部 True;端到端上 make_tasks.py 写出的 25 个
    文件 sha256 前后相同。所以 2026-09-03~09-04 那批实测数字与新生成的数据仍可比。
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) / size
    base = np.stack(
        [0.25 + 0.5 * xx, 0.25 + 0.5 * yy, 0.35 + 0.3 * (xx + yy) / 2], axis=-1
    )
    for _ in range(8):
        cx, cy = rng.uniform(0, size, 2)
        r = rng.uniform(size * 0.05, size * 0.22)
        color = rng.uniform(0.05, 0.95, 3)
        mask = (yy * size - cy) ** 2 + (xx * size - cx) ** 2 < r**2
        base[mask] = color
    return (base * 255).astype(np.uint8)
