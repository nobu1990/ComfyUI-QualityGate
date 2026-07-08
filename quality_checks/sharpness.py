"""
quality_checks/sharpness.py

ボケ/低ディテール検出。ラプラシアン分散（focus measure）を使う古典手法。
拡散モデルの生成物では、破綻・溶けた領域はディテールが失われて分散が下がる。

score は「分散が min_var 以下なら 0、good_var 以上なら 1」の線形マップ。
デフォルトは 512px 前後の生成画像を想定した経験値。実データで調整する前提。
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import QAResult, clamp01, register


@register("sharpness")
def check_sharpness(
    img_bgr: np.ndarray,
    min_var: float = 30.0,
    good_var: float = 150.0,
    pass_var: float = 60.0,
) -> QAResult:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    score = clamp01((var - min_var) / max(good_var - min_var, 1e-6))
    passed = var >= pass_var
    return QAResult(
        name="sharpness",
        score=score,
        passed=passed,
        detail=f"lap_var={var:.1f} (pass>={pass_var:.0f})",
        metrics={"laplacian_var": var},
    )
