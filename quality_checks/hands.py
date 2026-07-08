"""
quality_checks/hands.py  ─  手の破綻検出（Phase 1: MediaPipe ランドマーク＋幾何ヒューリスティック）

⚠️ EXPERIMENTAL / 棚上げ（2026-07-06）:
  実データ検証で、握り拳・物を握る手・角度で指が隠れる手（いずれも自然な良品）を
  「破綻」と誤検出することが判明（fusion/proportion が自然なポーズでも下がるため）。
  幾何ヒューリスティックでは「壊れた手」と「自然なポーズ」を原理的に切り分けられない。
  → 汎用カスケードのデフォルトからは外してある。実用にはPhase 2（自前学習の分類器）が必要。
  プロジェクトの主軸は body_proportion（アイデア②）へ移行済み。

生成画像で最も金になる破綻＝「壊れた手・指」を弾く。

【設計と、正直な限界】
MediaPipe Hands は *実写の手* で学習されており、21点ランドマークを返す。
AI生成の壊れた手に対する挙動は次の2通りで、どちらもシグナルになる:
  (a) そもそも手として検出できない（低信頼）      → detection シグナル
  (b) 無理に21点を当て込み、幾何が歪む            → geometry シグナル
ただし MediaPipe は常に「5本指ぶんの21点」を出すため、
"6本目の指"を直接カウントすることはできない。壊れは (a)(b) に間接的に現れる。
→ ここは粗い一次フィルタ。確度の高い指本数判定は Phase 2（自前学習モデル / ONNX）で置換する。

【依存】
mediapipe（ComfyUI env に導入済み前提）。未導入環境では skipped=True を返して
カスケードを壊さない（集約から除外される）。

ランドマーク index:
  0=wrist / 1-4=thumb / 5-8=index / 9-12=middle / 13-16=ring / 17-20=pinky
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .base import QAResult, clamp01, register

# --- モデル（初回のみDL、リポジトリにはコミットしない） ---
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

# 指ごとの [先端, 全4関節] index
_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
_FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

_landmarker = None          # 遅延生成のシングルトン
_init_error: Optional[str] = None


def _ensure_model() -> Path:
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _MODEL_PATH.with_suffix(".task.part")
    urllib.request.urlretrieve(_MODEL_URL, tmp)  # noqa: S310 (公式Googleストレージ)
    tmp.replace(_MODEL_PATH)
    return _MODEL_PATH


def _get_landmarker():
    """HandLandmarker を1度だけ構築。mediapipe 無しなら _init_error に理由を残す。"""
    global _landmarker, _init_error
    if _landmarker is not None or _init_error is not None:
        return _landmarker
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        model = _ensure_model()
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=RunningMode.IMAGE,
            num_hands=4,
            min_hand_detection_confidence=0.3,
        )
        _landmarker = HandLandmarker.create_from_options(opts)
    except Exception as e:  # ImportError / DL失敗 / モデル不整合
        _init_error = f"{type(e).__name__}: {e}"
    return _landmarker


def _xy(landmarks) -> np.ndarray:
    """21点の正規化(x,y)を (21,2) 配列で返す。"""
    return np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float64)


def _finger_length(pts: np.ndarray, idxs: List[int]) -> float:
    seg = np.diff(pts[idxs], axis=0)
    return float(np.linalg.norm(seg, axis=1).sum())


def _hand_plausibility(pts: np.ndarray) -> tuple[float, dict]:
    """1つの手の幾何がどれだけ自然か [0,1]。fusion（指の癒着）と proportion（長さ異常）を見る。"""
    # 手のスケール = 手首→中指付け根。0除算ガード。
    scale = float(np.linalg.norm(pts[9] - pts[0])) or 1e-6

    # (1) 指先の癒着: 隣り合う指先が近すぎる = 融合の疑い
    adj = [("index", "middle"), ("middle", "ring"), ("ring", "pinky")]
    gaps = [np.linalg.norm(pts[_TIPS[a]] - pts[_TIPS[b]]) / scale for a, b in adj]
    min_gap = float(min(gaps))
    # gap>=0.25 で健全(1.0)、<=0.08 で癒着(0.0)
    fusion_score = clamp01((min_gap - 0.08) / (0.25 - 0.08))

    # (2) 指の長さ比が人体的にありえるか。scale正規化して妥当帯 [0.5, 1.9] を外れた指を減点。
    lengths = {f: _finger_length(pts, idxs) / scale for f, idxs in _FINGERS.items()}
    outliers = sum(1 for f, L in lengths.items() if not (0.5 <= L <= 1.9))
    proportion_score = clamp01(1.0 - 0.25 * outliers)

    # 保守的合成: どれか1つでも強い破綻(0付近)があれば全体を落とす。
    # 平均だと「片方だけ完全破綻」を見逃す(0.5*0+0.5*1=0.5)ため min を主成分にする。
    worst = min(fusion_score, proportion_score)
    mean = 0.5 * (fusion_score + proportion_score)
    score = 0.7 * worst + 0.3 * mean
    return clamp01(score), {
        "min_tip_gap": round(min_gap, 3),
        "fusion": round(fusion_score, 3),
        "proportion": round(proportion_score, 3),
        "length_outliers": float(outliers),
    }


@register("hands")
def check_hands(
    img_bgr: np.ndarray,
    expected_hands: Optional[int] = None,
    pass_threshold: float = 0.5,
) -> QAResult:
    lm = _get_landmarker()
    if lm is None:
        return QAResult(
            name="hands", score=1.0, passed=True, skipped=True,
            detail=f"SKIPPED (mediapipe unavailable: {_init_error})",
        )

    import mediapipe as mp

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = lm.detect(mp_img)
    hands = result.hand_landmarks or []
    n = len(hands)

    # 手が1つも検出されない場合
    if n == 0:
        if expected_hands:  # 手があるはずなのに無い = 破綻/欠損
            return QAResult(name="hands", score=0.0, passed=False,
                            detail=f"hands=0 (expected={expected_hands})",
                            metrics={"hand_count": 0.0})
        # 手を期待していない画像（風景など）は評価対象外
        return QAResult(name="hands", score=1.0, passed=True, skipped=True,
                        detail="no hands detected (not evaluated)")

    # 検出された各手の幾何プラウジビリティ。最悪の手でそのバッチを代表させる。
    per_hand = [_hand_plausibility(_xy(h)) for h in hands]
    geom = min(s for s, _ in per_hand)
    worst = min(per_hand, key=lambda t: t[0])[1]

    if expected_hands is not None:
        count_score = clamp01(1.0 - 0.5 * abs(n - expected_hands))
        score = clamp01(0.5 * geom + 0.5 * count_score)
        count_ok = n == expected_hands
    else:
        score = geom
        count_ok = True

    passed = bool(score >= pass_threshold and count_ok)
    return QAResult(
        name="hands",
        score=score,
        passed=passed,
        detail=(f"hands={n} geom={geom:.2f} "
                f"tip_gap={worst['min_tip_gap']} out={int(worst['length_outliers'])}"),
        metrics={"hand_count": float(n), "geometry": geom, **worst},
    )
