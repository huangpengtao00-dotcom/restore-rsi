"""退化探针生成器的契约测试:恒等 / 确定性 / 单调性 / manifest 可重放。

判据都从模块自己的口径来:亮度 = 输出均值;噪声方差 = 平坦灰图输出的像素方差(图本身无纹理,方差全是噪声);
雾的暗通道 = 逐像素三通道最小值的均值。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from restore_rsi.degrade import FAMILIES, fog, night
from restore_rsi.degrade.base import SEVERITIES
from restore_rsi.degrade.depth import synthetic_depth
from restore_rsi.degrade.pipeline import degrade_frame, load_manifest, make_dataset, replay_task, sample_params

SEEDS = [0, 1, 7]


def _img(seed: int = 0, size: int = 96) -> np.ndarray:
    """有纹理、有明暗的测试底图(不依赖 judge-lab)。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    base = np.stack([0.2 + 0.6 * xx, 0.3 + 0.5 * yy, 0.5 + 0.3 * np.sin(6 * xx)], -1)
    for _ in range(5):
        cx, cy, r = rng.uniform(0, size, 2).tolist() + [rng.uniform(8, 25)]
        base[((yy * size - cy) ** 2 + (xx * size - cx) ** 2) < r**2] = rng.uniform(0.05, 0.95, 3)
    return (np.clip(base, 0, 1) * 255 + 0.5).astype(np.uint8)


def _apply(fam, img, params, seed, depth=None):
    rng = np.random.default_rng(seed)
    return fam.apply(img, params, rng, depth=depth) if fam.NEEDS_DEPTH else fam.apply(img, params, rng)


# ---------------- severity 0 恒等 ----------------

@pytest.mark.parametrize("name", sorted(FAMILIES))
@pytest.mark.parametrize("seed", SEEDS)
def test_severity0_identity(name, seed):
    fam = FAMILIES[name]
    img = _img(seed)
    params = fam.params_for_severity(0, np.random.default_rng(seed))
    depth = synthetic_depth(img.shape[:2], np.random.default_rng(seed)) if fam.NEEDS_DEPTH else None
    out = _apply(fam, img, params, seed, depth)
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert np.array_equal(out, img)


# ---------------- 同 seed 位级同 ----------------

@pytest.mark.parametrize("name", sorted(FAMILIES))
@pytest.mark.parametrize("severity", [1, 3, 4])
def test_deterministic_same_seed(name, severity):
    fam = FAMILIES[name]
    img = _img(3)
    p1 = fam.params_for_severity(severity, np.random.default_rng(11))
    p2 = fam.params_for_severity(severity, np.random.default_rng(11))
    assert p1 == p2
    depth = synthetic_depth(img.shape[:2], np.random.default_rng(5)) if fam.NEEDS_DEPTH else None
    a = _apply(fam, img, p1, 99, depth)
    b = _apply(fam, img, p2, 99, depth)
    assert np.array_equal(a, b)
    # 换噪声 seed 应该不同(有噪声的族),证明 rng 真的被用了
    if name in ("night", "noise_simple"):
        c = _apply(fam, img, p1, 100, depth)
        assert not np.array_equal(a, c)


def test_params_jitter_independent_of_severity():
    """同 rng 状态下,不同 severity 抽到同一组相机/抖动量(单调扫描的前提)。"""
    keep = ("gamma", "ccm_idx", "rgb_gain", "red_gain", "blue_gain", "quant_bits")
    ps = [night.params_for_severity(s, np.random.default_rng(42)) for s in SEVERITIES]
    for k in keep:
        assert len({p[k] for p in ps}) == 1, k
    pf = [fog.params_for_severity(s, np.random.default_rng(42)) for s in SEVERITIES]
    assert all(p["A"] == pf[0]["A"] for p in pf)


# ---------------- night 单调性 ----------------

@pytest.mark.parametrize("seed", SEEDS)
def test_night_brightness_monotone_decreasing(seed):
    img = _img(seed)
    means = []
    for s in SEVERITIES:
        params = night.params_for_severity(s, np.random.default_rng(seed))
        means.append(_apply(night, img, params, seed).mean())
    assert all(a > b for a, b in zip(means, means[1:])), means
    assert all(0 < p["darkness"] <= 1 for p in [night.params_for_severity(s, np.random.default_rng(seed)) for s in SEVERITIES])


@pytest.mark.parametrize("seed", SEEDS)
def test_night_noise_variance_monotone_increasing(seed):
    flat = np.full((64, 64, 3), 128, np.uint8)  # 平坦灰:输出像素方差全部来自噪声
    variances = []
    for s in SEVERITIES:
        params = night.params_for_severity(s, np.random.default_rng(seed))
        variances.append(_apply(night, flat, params, seed).astype(np.float64).var())
    assert variances[0] == 0.0
    assert all(a < b for a, b in zip(variances, variances[1:])), variances
    shots = [night.params_for_severity(s, np.random.default_rng(seed))["shot_noise"] for s in SEVERITIES]
    assert shots == sorted(shots)


def test_night_severity_ramp_pinned():
    """档位表的数值本身,不只是它们的顺序。

    单调性测试挡不住一次小幅重调:把 severity 1 的 darkness 从 0.35 改成 0.30,
    顺序仍然成立,48 项照样全绿(2026-09-03 变异实测)。而这张表是对外声明过的
    口径——docstring 写明 3/4 档落在 JarvisIR 原配置 darkness∈[0.07,0.15] 里,
    README 的天花板数字也是在这张表下测的。所以把它钉死:要改,先改这里。
    """
    assert night._SEV == {
        1: (0.35, 5e-6, 5e-6),
        2: (0.18, 2e-5, 2e-5),
        3: (0.11, 8e-5, 8e-5),
        4: (0.07, 2.5e-4, 2.5e-4),
    }
    # docstring 的那句"3–4 档落在 JarvisIR 原区间"必须为真
    assert all(0.07 <= night._SEV[s][0] <= 0.15 for s in (3, 4))
    # shot 与 read 同值是有意的(见模块 docstring 第二条),不是复制粘贴
    assert all(shot == read for _, shot, read in night._SEV.values())


# ---------------- fog 单调性 ----------------

def _dark_channel_mean(img_u8: np.ndarray) -> float:
    return float(img_u8.min(axis=2).mean())


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("real_like_depth", [False, True])
def test_fog_dark_channel_monotone_increasing(seed, real_like_depth):
    img = _img(seed)
    if real_like_depth:  # 模拟"真深度":随机块状距离图,不是梯度
        rng = np.random.default_rng(seed + 100)
        depth = np.kron(rng.uniform(0, 1, (6, 6)), np.ones((16, 16)))
    else:
        depth = synthetic_depth(img.shape[:2], np.random.default_rng(seed))
    dcs = []
    for s in SEVERITIES:
        params = fog.params_for_severity(s, np.random.default_rng(seed))
        dcs.append(_dark_channel_mean(_apply(fog, img, params, seed, depth)))
    assert all(a < b for a, b in zip(dcs, dcs[1:])), dcs
    betas = [fog.params_for_severity(s, np.random.default_rng(seed))["beta"] for s in SEVERITIES]
    assert betas[0] == 0.0 and betas == sorted(betas)


def test_fog_transmission_respects_depth():
    """近处 t 大(几乎不变),远处 t 小(逼近 A):这是"有深度的雾"和 haze_simple 的本质区别。"""
    img = np.full((32, 32, 3), 40, np.uint8)
    depth = np.zeros((32, 32))
    depth[16:] = 1.0  # 下半远
    params = fog.params_for_severity(4, np.random.default_rng(0))
    out = _apply(fog, img, params, 0, depth).astype(float)
    assert out[:16].mean() < out[16:].mean()
    assert abs(out[:16].mean() - 40) < 1.0  # 近处 d=0 → t=1 → 原样


# ---------------- 叠加与 manifest 重放 ----------------

def test_stack_order_matters_and_is_recorded(tmp_path):
    img = _img(1)
    params = sample_params(["night", "fog"], 3, 5)
    seeds = {"night": 1, "fog": 2}
    depth = synthetic_depth(img.shape[:2], np.random.default_rng(9))
    a = degrade_frame(img, ["night", "fog"], params, seeds, depth)
    b = degrade_frame(img, ["fog", "night"], params, seeds, depth)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("depth_mode", ["synthetic"])
def test_manifest_complete_and_replayable(tmp_path, depth_mode):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        Image.fromarray(_img(i)).save(src / f"im{i}.png")
    mpath = make_dataset(
        out=tmp_path / "out",
        chains=["night", "fog", "night+fog", "haze_simple+noise_simple"],
        severities=[0, 2, 4],
        n=2,
        seed=123,
        src=src,
        depth_mode=depth_mode,
        n_frames=2,
    )
    tasks = load_manifest(mpath)
    assert len(tasks) == 4 * 3 * 2
    for t in tasks:
        gt = t["gt"]
        assert set(gt) >= {"applied_order", "severity", "params", "seeds", "depth_source", "families", "source"}
        assert len(t["frames"]) == 2 and len(gt["seeds"]) == 2
        for name in gt["applied_order"]:
            assert name in gt["params"] and name in gt["severity"]
            assert all(isinstance(v, (int, float, list)) for v in gt["params"][name].values())
            assert gt["families"][name]["surrogate"] == FAMILIES[name].SURROGATE
            assert all(name in s for s in gt["seeds"])
        needs_depth = any(FAMILIES[n].NEEDS_DEPTH for n in gt["applied_order"])
        assert (gt["depth_source"] == "synthetic") if needs_depth else (gt["depth_source"] is None)
        if needs_depth:
            assert all(Path(f["depth"]).exists() for f in t["frames"])
        # 序列两帧:同参数、不同噪声 seed
        assert gt["seeds"][0] != gt["seeds"][1]
        # 重放位级一致(只从 reference + params + seeds 重建,不看已存 input)
        for frame, regen in zip(t["frames"], replay_task(t), strict=True):
            stored = np.asarray(Image.open(frame["input"]).convert("RGB"))
            assert np.array_equal(stored, regen), t["task_id"]
    # json 往返后仍可重放(manifest 是唯一真相源)
    reloaded = json.loads(Path(mpath).read_text())["tasks"]
    t = next(x for x in reloaded if x["task_id"].startswith("night+fog_s4"))
    stored = np.asarray(Image.open(t["frames"][0]["input"]).convert("RGB"))
    assert np.array_equal(stored, replay_task(t)[0])


def test_make_dataset_deterministic_across_runs(tmp_path):
    kw = dict(chains=["night+fog"], severities=[3], n=1, seed=7, depth_mode="synthetic")
    m1 = make_dataset(out=tmp_path / "a", **kw)
    m2 = make_dataset(out=tmp_path / "b", **kw)
    t1, t2 = load_manifest(m1)[0], load_manifest(m2)[0]
    assert t1["gt"]["params"] == t2["gt"]["params"] and t1["gt"]["seeds"] == t2["gt"]["seeds"]
    a = np.asarray(Image.open(t1["frames"][0]["input"]))
    b = np.asarray(Image.open(t2["frames"][0]["input"]))
    assert np.array_equal(a, b)


def test_depth_anything_strict_mode_does_not_silently_fallback(monkeypatch):
    """--depth depth_anything 加载失败必须抛,不许悄悄换合成深度。"""
    from restore_rsi.degrade import depth as d

    def boom(self, model_id=d.DEPTH_MODEL_ID):
        raise d.DepthUnavailable("forced")

    monkeypatch.setattr(d.DepthAnythingV2, "__init__", boom)
    with pytest.raises(d.DepthUnavailable):
        d.resolve_backend("depth_anything")
    est, source, info = d.resolve_backend("auto")
    assert est is None and source == "synthetic" and "forced" in info["fallback_reason"]
