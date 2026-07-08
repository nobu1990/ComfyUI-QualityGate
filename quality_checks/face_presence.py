"""
quality_checks/face_presence.py

顔の有無・個数チェック。OpenCV 同梱の Haar cascade を使うので追加DL不要。
「ポートレートを頼んだのに顔が生成されない/複数写り込む」破綻を弾く最小版。

将来の拡張ポイント（README のロードマップ参照）:
  - Haar -> YOLO-face / RetinaFace に差し替えて再現率を上げる
  - landmark を取り、目・口の破綻や left/right 反転を検出する
  - ArcFace 埋め込みを出力し、キャラ一貫性スコアラー(サンプル②)へ橋渡し

このカスケード構造自体が SeedCount の「検出→再検出」の転用。
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import QAResult, clamp01, register

# Haar cascade は cv2 に同梱。パスは実行時に解決する。
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


@register("face_presence")
def check_face_presence(
    img_bgr: np.ndarray,
    expected: int = 1,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size_frac: float = 0.08,
) -> QAResult:
    if _cascade.empty():
        return QAResult(
            name="face_presence", score=0.0, passed=False,
            detail="haar cascade failed to load",
        )

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    min_side = int(min(h, w) * min_size_frac)

    faces = _cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_side, min_side),
    )
    n = len(faces)

    # expected と一致で 1.0、ズレ 1 個ごとに 0.5 減点。
    score = clamp01(1.0 - 0.5 * abs(n - expected))
    passed = n == expected
    return QAResult(
        name="face_presence",
        score=score,
        passed=passed,
        detail=f"faces={n} (expected={expected})",
        metrics={"face_count": float(n)},
    )
