"""报分口径:「失败」必须是相对这道题能达到什么来定义的。

放在 harness 而不是 run.py,因为这是一条**判据**,不是中继的实现细节 ——
判据要能被单独测试,而 run.py 顶层要 import reef_client。
"""

from __future__ import annotations


def attained(score: float, frame: dict) -> tuple[float, str]:
    """(报给 reef 的分, 写进 feedback 的口径说明)。

    报的是**达成率**(相对这道题穷举出来的上界),不是绝对分。`data.max_score`
    是一把绝对的尺子,而这 12 道题的上界差 46.5 倍(0.0215 ~ 1.0000),实测 4/12
    判反:

      low_light_00   上界 1.000,拿到 0.656(达成 66%)-> 绝对阈判"成功",
                     于是**永不进 batch**,propose 从没见过这道有巨大改进空间的题
      haze_noise_00  上界 0.202,拿到 0.228(达成 113%)-> 绝对阈判"失败",
                     于是进 batch,让 propose 去为一道已经做到头的题找不存在的改进

    batch 的组成因此系统性地偏了:该学的进不来,已到极限的混进来。挪阈值救不了 ——
    一把尺子量不了 46 倍的量程。

    绝对分和上界都留在返回的说明里,所以换算随时可还原,历史数字也还对得上。
    没有上界时退回绝对分,但**必须在说明里写出来**:同一批数字里混着两种口径而
    没人看得出来,比用错口径更糟。
    """
    ceiling = frame.get("ceiling")
    if ceiling and ceiling > 0:
        reported = score / ceiling
        return reported, f" score={score:.4f} ceiling={ceiling:.4f}(d{frame.get('ceiling_depth')}) attained={reported:.3f}"
    return score, f" score={score:.4f} ceiling=none(reported as absolute)"
