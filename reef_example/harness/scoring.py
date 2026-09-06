"""报分口径:「失败」必须是相对这道题能达到什么来定义的。

放在 harness 而不是 run.py,因为这是一条**判据**,不是中继的实现细节 ——
判据要能被单独测试,而 run.py 顶层要 import reef_client。
"""

from __future__ import annotations

import os
from pathlib import Path


def work_dir() -> Path:
    """本次配置的状态目录。并行跑多个消融臂时每个臂一份,`RESTORE_WORK` 指定。"""
    return Path(__file__).resolve().parent.parent / os.environ.get("RESTORE_WORK", "work")


def batch_threshold() -> float:
    """报到多少分才算「失败」、于是进 batch。**从 materialize 生成的 recipe 读。**

    run.py 原来硬编码 `score <= 0.5`,而真正决定 reef 是否 batch 的是
    `data.max_score`(serve.yaml,可被 RESTORE_MAX_SCORE 覆盖)。两处各自写一份
    判据的后果,2026-09-06 实测到了:阈值调到 0.9 之后三道题报 0.727/0.783/0.741,
    reef 那侧全部会 batch,而 run.py 按自己的 0.5 认为「every task passed」,打印
    一句就退出,根本不去等 evolve。整轮四十分钟白跑,日志看起来一切正常。

    和同一天的 `tool_catalog` 是同一个坑:判据在两处各自实现,迟早分叉。

    放在 harness 而不是 run.py,理由和 `attained` 一样:判据要能被单独测试,
    而 run.py 顶层要 import reef_client。
    """
    import yaml

    recipe = work_dir() / "recipes" / "harness_evolve.yaml"
    try:
        data = yaml.safe_load(recipe.read_text(encoding="utf-8")).get("data") or {}
    except (OSError, AttributeError, yaml.YAMLError) as exc:
        raise RuntimeError(
            f"读不到 {recipe} 里的 data.max_score({exc!r})。run.sh 会先跑 materialize.py "
            f"生成它;直接跑 run.py 时请先 materialize。不猜一个默认值 —— 猜错了整轮的"
            f"「有没有失败」都是错的,而日志看起来完全正常。"
        ) from exc
    if "max_score" not in data:
        raise RuntimeError(f"{recipe} 的 data 段没有 max_score:{sorted(data)}")
    return float(data["max_score"])


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
    from .ablation import current

    # batch_criterion=absolute 复刻改动前的行为:一把绝对尺子量 46.5 倍的量程。
    # 保留它不是为了兼容,是因为"这个缺陷值多少可达分数"要靠它跑对照。
    if current().batch_criterion == "absolute":
        return score, f" score={score:.4f} ceiling=ignored(ablation:absolute)"
    ceiling = frame.get("ceiling")
    if ceiling and ceiling > 0:
        reported = score / ceiling
        return reported, f" score={score:.4f} ceiling={ceiling:.4f}(d{frame.get('ceiling_depth')}) attained={reported:.3f}"
    return score, f" score={score:.4f} ceiling=none(reported as absolute)"


# --------------------------------------------------------------- 测量失败的编码

class MeasurementFailure(RuntimeError):
    """测量失败:judge 这一侧坏了,不是 agent 的表现。

    默认(`measurement_failure=raise`)抛出去,让这一步响亮地中止。其余三种编码
    存在**只为了做对照实验** —— 它们各自复刻一篇已发表工作公开代码里的真实做法。
    """


def encode_measurement_failure(reason: str, task_id: str, seed_key: str = "") -> float:
    """把一次测量失败变成一个分数(或抛出)。返回值直接进闸门。

    四种编码,三个不同的偏差方向。这不是设计空间的枚举,是从三篇顶会工作的公开
    代码里读出来的实际做法:

      raise    默认。测量失败不是分数,抛错中止这一步
      zero     记 0.0 —— JarvisArt 的 reward 路径上 28 处这么写
               (grpo_lr.py 9 处、lrc_tools_reward.py 19 处)。**悲观**偏差,而且
               在 GRPO 里污染组内优势归一化:一组候选里混进几个因工具崩掉而得 0 的,
               整组均值和标准差都被拉偏,梯度于是奖励"别让工具崩"而不是"把图修好"
      exclude  记 NaN —— JarvisEvo 的 `if result: group_list.append(result)`,失败样本
               从分母消失。**乐观**偏差(与 zero 方向相反),而且失败数根本不报告。
               NaN 在闸门里既不算胜也不算负(比较恒为 False),正好复刻"从分母消失"
      guess    猜一个分 —— VIEScore 的 `guess_if_cannot_parse`:解析不出来就猜。
               纯噪声,方向不定

    `guess` 用按 (task_id, seed_key, reason) 派生的确定性 RNG,所以同一次实验可复现 ——
    一个不可复现的对照组等于没有对照组。
    """
    from .ablation import current

    mode = current().measurement_failure
    if mode == "raise":
        raise MeasurementFailure(f"[{task_id}] {reason}")
    if mode == "zero":
        return 0.0
    if mode == "exclude":
        return float("nan")
    if mode == "guess":
        import hashlib
        import random

        digest = hashlib.sha256(f"{task_id}|{seed_key}|{reason}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big")).random()
    raise AssertionError(f"unreachable measurement_failure mode {mode!r}")
