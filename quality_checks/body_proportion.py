"""
quality_checks/body_proportion.py

顔写真→全身生成での「頭の大きさ（対 肩幅）」を計測する。アイデア②の中核。

【確定した計測指標（実データ検証済み, 2026-07-06）】
  R = 耳〜耳の距離(頭幅) / 肩幅        ※微小クロップ K 回の平均(TTA)で算出

  - 頭幅: MediaPipe Pose の左右耳(7,8)の距離。髪が下に伸びても不変(＝髪型に強い)。
          髪で耳が隠れても Pose が位置推定するので、耳可視/不可視で挙動が変わらない
          （実データで確認、分岐不要）。
  - 肩幅: 左右肩(11,12)の距離。body スケールの基準。
  - TTA: 単発測定は耳/肩キーポイントの揺れ(std≈0.006)が信号(std≈0.012)の半分を占める。
         微小クロップ K 枚の平均でノイズを 1/√K に落とし、0.01 の差を識別可能にする。

以前の「髪込み頭面積 / 胴体長^2」は (1)長い髪で過大 (2)顔検出が全身画像の小さい顔で失敗
という理由で不採用。耳〜耳は顔検出もセグメントも顎ラインも要らず最も頑健。

スコアは参照画像の R をターゲットにした「絶対ズレ」で採点（相対%でなく絶対値。R の
実スケールが狭いため）。

【依存】mediapipe(Pose)。未導入/未検出時は skipped=True でカスケードを壊さない。
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import QAResult, clamp01, register

_POSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
_MODELS = Path(__file__).resolve().parent.parent / "models"

# TTA用の決定的クロップ (top,bottom,left,right の割合)。先頭は無クロップ。
_TTA_CROPS = [
    (0.0, 0.0, 0.0, 0.0),
    (0.02, 0.0, 0.0, 0.0), (0.0, 0.02, 0.0, 0.0),
    (0.0, 0.0, 0.02, 0.0), (0.0, 0.0, 0.0, 0.02),
    (0.015, 0.015, 0.0, 0.0), (0.0, 0.0, 0.015, 0.015),
    (0.01, 0.01, 0.01, 0.01),
]
TTA_K = 6                 # 平均に使うクロップ数（多いほど安定・重い）
DEFAULT_TOL = 0.02        # 採点の絶対トレランス（Rズレ0.02で0点、0.01で0.5点）

_pose = None
_init_error: Optional[str] = None


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310
    tmp.replace(dest)
    return dest


def _init():
    global _pose, _init_error
    if _pose is not None or _init_error is not None:
        return
    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker, PoseLandmarkerOptions, RunningMode,
        )
        model = _download(_POSE_URL, _MODELS / "pose_landmarker_lite.task")
        _pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=RunningMode.IMAGE, num_poses=1))
    except Exception as e:
        _init_error = f"{type(e).__name__}: {e}"


def _ear_over_shoulder(img_bgr) -> Optional[float]:
    """単発: 耳幅/肩幅。人物が取れなければ None。"""
    import mediapipe as mp
    h, w = img_bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pr = _pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not pr.pose_landmarks:
        return None
    lm = pr.pose_landmarks[0]

    def P(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    shw = float(np.linalg.norm(P(11) - P(12)))
    if shw <= 1:
        return None
    ear = float(np.linalg.norm(P(7) - P(8)))
    return ear / shw


def measure(img_bgr: np.ndarray, k: Optional[int] = None) -> Optional[dict]:
    """R = 耳幅/肩幅 を TTA(K回クロップ平均)で算出。人物が取れなければ None。要 _init()。"""
    k = TTA_K if k is None else max(1, k)
    h, w = img_bgr.shape[:2]
    vals = []
    for (t, b, l, r) in _TTA_CROPS[:k]:
        crop = img_bgr[int(t * h):h - int(b * h), int(l * w):w - int(r * w)]
        if crop.size == 0:
            continue
        v = _ear_over_shoulder(crop)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return {"ratio": float(np.mean(vals)), "n_tta": len(vals)}


def ratio_score(ratio: float, target: float, tol: float = DEFAULT_TOL) -> float:
    """target からの絶対ズレで採点。指数減衰: ズレ0→1.0、ズレ tol→0.37、2tol→0.14。

    ハードクランプ(線形で0)だと、参照がバッチから離れると全画像が0点に飽和して
    順位がつかない。指数減衰なら遠くても単調に下がり、常に近い順にランキングできる。
    tol は e-folding スケール（小さいほど近さに敏感）。
    """
    if tol <= 0:
        return 0.0
    return float(np.exp(-abs(ratio - target) / tol))


@register("body_proportion")
def check_body_proportion(
    img_bgr: np.ndarray,
    target_ratio: Optional[float] = None,
    tol: float = DEFAULT_TOL,
    pass_threshold: float = 0.5,
) -> QAResult:
    """単体チェック。target_ratio（参照画像の R）が無ければ skip。"""
    _init()
    if _pose is None:
        return QAResult(name="body_proportion", score=1.0, passed=True, skipped=True,
                        detail=f"SKIPPED (mediapipe unavailable: {_init_error})")
    if target_ratio is None:
        return QAResult(name="body_proportion", score=1.0, passed=True, skipped=True,
                        detail="SKIPPED (no target_ratio; use the batch ranker)")

    m = measure(img_bgr)
    if m is None:
        return QAResult(name="body_proportion", score=1.0, passed=True, skipped=True,
                        detail="no person detected (not evaluated)")

    score = ratio_score(m["ratio"], target_ratio, tol)
    return QAResult(
        name="body_proportion",
        score=score,
        passed=bool(score >= pass_threshold),
        detail=f"ear/shoulder={m['ratio']:.3f} (target={target_ratio:.3f}, tol={tol})",
        metrics={"ratio": m["ratio"], "n_tta": float(m["n_tta"])},
    )
