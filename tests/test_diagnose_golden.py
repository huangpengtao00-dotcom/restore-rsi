"""诊断口径不许悄悄漂移。

`diagnose_array` 内部调用 `dcp_estimate` / `dehaze_dcp` / `lowlight_gamma` ——
它们是工具箱里**会被改进**的算子。而 test_diagnose.py 里的五条判据全是相对性质
(干净图近零 / 在场与不在场可分 / 对强度单调 / 真模糊测得出),换一个更好的去雾
算法它们照样全绿,但:

  * agent 唯一的观测变了
  * 2026-09-03/04 记录的行为侧数字(锐化 5.8%、去雾 17.3%、提亮 40.4%)不再可比
  * 没有任何东西会报警

这正是"加字段 / 改口径前先查所有消费方"那条:旧结论是在旧口径的世界里写下的,
会静默地把新数据当旧数据算。所以这里钉住**绝对读数**。

改动算子导致这里变红时,不要调容差。要做的是:

    uv run python tests/test_diagnose_golden.py --regen

然后在 commit 里写明**哪一批历史数字因此不可比**。红本身就是那句话的触发器。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from restore_rsi.synth import gen_test_image  # noqa: E402
from toolbox.cli import diagnose_array  # noqa: E402

GOLDEN = Path(__file__).parent / "data" / "diagnose_golden.json"

#: 平台噪声的实测上限。第一版把容差定在 0.002,理由是"算子确定性、底图 seed 固定,
#: 差异只可能来自浮点累积" —— 在这台 mac 上完全成立,推上去 CI 立刻红:
#: macOS arm64 与 Linux x86_64 之间 blur 差 0.0130、haze 差 0.0030、low_light 差 0.0050。
#: 又一次"本地绿不算绿",这回栽的是我自己定的判据。
PLATFORM_NOISE_CEILING = 0.0130

#: 一次真实口径变更的信号强度,实测:把 lowlight_gamma 的 target_mean 从 0.45 改成 0.50,
#: 七个读数发生变化,最大的一项(haze_low_light.haze)差 0.1100。
KNOWN_DRIFT_SIGNAL = 0.1100

#: 容差落在噪声与信号之间。**这个数是量出来的,不是拍的**:0.02 之上是平台差异,
#: 之下是真实漂移。那次变异在 0.02 下仍有 3 项变红,报警足够。
#:
#: 逐 case 的判别力并不均匀,得说清楚:haze_low_light.haze 是 0.003 噪声对 0.110 信号
#: (37 倍,很稳),而 blur_060.blur 是 0.013 噪声对 0.025 信号(不到 2 倍,勉强)。
#: 所以这条判据靠的是**多个 case 一起看**,不是任何单独一项。
TOL = 0.02


def _haze(x, t, rng):
    A = 0.85
    trans = 1.0 - 0.75 * t
    return x * trans + A * (1.0 - trans)


def _lowlight(x, t, rng):
    return np.clip(x, 0, 1) ** (1.0 + 2.2 * t) * (1.0 - 0.4 * t)


def _noise(x, t, rng):
    return x + rng.normal(0, 0.12 * t, x.shape)


def _blur(x, t, rng):
    u8 = (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
    return np.asarray(Image.fromarray(u8).filter(ImageFilter.GaussianBlur(radius=2.0 * t))).astype(np.float64) / 255.0


DEGRADERS = {"haze": _haze, "low_light": _lowlight, "noise": _noise, "blur": _blur}

#: 八种情形:干净、四个单退化、三个双退化组合。双退化在这里不是凑数 ——
#: 读数互相掩盖正是 v0 的病根,golden 把残余掩盖的量也一起钉住了
#: (haze 单独 0.732,叠暗光 0.387 = 掉 47%,叠噪声 0.538 = 掉 26%)。
CASES = [
    ("clean", []),
    ("haze_060", [("haze", 0.6)]),
    ("low_light_060", [("low_light", 0.6)]),
    ("noise_060", [("noise", 0.6)]),
    ("blur_060", [("blur", 0.6)]),
    ("haze_low_light", [("haze", 0.6), ("low_light", 0.6)]),
    ("low_light_noise", [("low_light", 0.6), ("noise", 0.6)]),
    ("haze_noise", [("haze", 0.6), ("noise", 0.6)]),
]


def build(chain) -> np.ndarray:
    """固定 seed:同一份输入,每次跑都是同样的字节。"""
    rng = np.random.default_rng(0)
    x = gen_test_image(np.random.default_rng(0), size=256).astype(np.float64) / 255.0
    for family, strength in chain:
        x = DEGRADERS[family](x, strength, rng)
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def _golden() -> dict:
    return json.loads(GOLDEN.read_text())


def test_tolerance_still_discriminates():
    """元判据:容差必须落在实测的噪声与信号之间。

    这条测试存在的唯一理由,是**挡住将来的自己**。上面那条判据一旦在别的机器上
    偶发地红,最省事的做法就是把 TOL 调大到不红为止 —— 而 TOL 一旦越过
    KNOWN_DRIFT_SIGNAL,整个文件就变成了一个永远绿的摆设,比没有更糟:它看起来
    在守着口径,实际上什么都不守。

    两个边界都是实测值(来源见上面的注释),改它们之前先重新量。
    """
    assert PLATFORM_NOISE_CEILING < TOL, (
        f"TOL={TOL} 不高于实测平台噪声 {PLATFORM_NOISE_CEILING},换台机器就会假红"
    )
    assert TOL < KNOWN_DRIFT_SIGNAL, (
        f"TOL={TOL} 已经不低于已知的真实漂移 {KNOWN_DRIFT_SIGNAL} —— "
        f"这条判据再也抓不到它本来要抓的东西了。不要为了变绿走到这一步:"
        f"先确认是不是真的口径变了,是就 --regen 并写明历史数字作废。"
    )


def test_golden_file_covers_every_case():
    """判据先要证明自己在看东西:少一个 case 就等于少一份保护,而那不会让别的测试变红。"""
    recorded = set(_golden()["readings"])
    assert recorded == {name for name, _ in CASES}, (
        f"golden 文件与 CASES 不一致:只在文件里 {recorded - {n for n, _ in CASES}},"
        f"只在 CASES 里 {[n for n, _ in CASES if n not in recorded]}"
    )


@pytest.mark.parametrize("name,chain", CASES, ids=[n for n, _ in CASES])
def test_readings_match_the_recorded_calibration(name, chain):
    golden = _golden()["readings"][name]
    actual = diagnose_array(build(chain))["degradations"]
    for key, expected in golden.items():
        assert abs(actual[key] - expected) <= TOL, (
            f"{name}.{key}: 现在读 {actual[key]},记录的是 {expected}(差 {actual[key] - expected:+.4f})。\n"
            f"如果你刚改了 dcp_estimate / dehaze_dcp / lowlight_gamma 或 diagnose 本身,这就是口径变更:\n"
            f"  1. `uv run python tests/test_diagnose_golden.py --regen` 重新记录\n"
            f"  2. 在 commit 里写明哪一批历史数字因此不可比\n"
            f"如果是换了机器/架构导致的偶发小差(实测跨平台上限 {PLATFORM_NOISE_CEILING}),"
            f"那也不要直接调大 TOL —— 先重新量平台噪声和漂移信号,确认容差仍落在两者之间。"
        )


def _regen() -> None:
    readings = {name: diagnose_array(build(chain))["degradations"] for name, chain in CASES}
    payload = _golden()
    payload["readings"] = readings
    payload["diagnoser"] = diagnose_array(build([]))["diagnoser"]
    GOLDEN.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    for name, values in readings.items():
        print(f"  {name:16s} {values}")
    print(f"-> {GOLDEN}")


if __name__ == "__main__":
    if "--regen" in sys.argv:
        _regen()
    else:
        print(__doc__)
