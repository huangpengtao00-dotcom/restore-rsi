"""测试默认跑基线配置。

消融开关从环境变量读,所以一个残留的 `RESTORE_ABLATION` 会让整套测试在别的配置下
跑 —— 而症状是若干条判据莫名其妙地红(或者更糟:golden 值对上了但对的是另一版
诊断)。测试环境不该被外部环境影响,要测消融的自己 setenv。
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _baseline_ablation(monkeypatch):
    monkeypatch.delenv("RESTORE_ABLATION", raising=False)
