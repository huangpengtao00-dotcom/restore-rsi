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

#: 容差。算子确定性、底图由固定 seed 生成,所以差异只可能来自浮点累积;
#: 定在 0.002 而不是更松,是因为**松容差会把真实的口径变更放过去**,那就白钉了。
TOL = 0.002


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
            f"不要靠调大 TOL 让它变绿 —— 那正好把这条判据要抓的东西放过去。"
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
