"""
quality_checks/identity.py

参照顔との同一性（ArcFace cosine類似度）を計測する。複合ランキングの identity 軸。
「顔写真→全身生成」で顔がどれだけ参照本人に似ているかを 512次元埋め込みで測る。

【実装メモ】
env の insightface が古い(0.2.1)で高レベル FaceAnalysis(name=...) は壊れているが、
SCRFD 検出器と ArcFaceONNX 認識器の個別クラス＋face_align.norm_crop は生きているので
onnx を直接ロードして使う。モデルは buffalo_l パック（det_10g + w600k_r50）。

モデル探索順: env INSIGHTFACE_MODEL_DIR → ~/.insightface/models/buffalo_l →
ComfyUI の models/insightface 配下（folder_paths で解決）。見つからなければ skip 相当（None）。

【依存】insightface（onnx直ロード）, onnxruntime。未導入/未検出時は None を返す。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import clamp01

_det = None
_rec = None
_init_error: Optional[str] = None

# 同一人物(生成バリエーション)の cosine は概ね 0.5〜0.95、別人は <0.3。
# この帯を [0,1] に伸ばして他軸と重み付けできるようにする。
_COS_LOW, _COS_HIGH = 0.30, 0.90


def _find_pack() -> Optional[Path]:
    cands = []
    env = os.environ.get("INSIGHTFACE_MODEL_DIR")
    if env:
        cands.append(Path(env))
    cands.append(Path.home() / ".insightface" / "models" / "buffalo_l")
    # ComfyUI の models/insightface 配下（folder_paths で環境非依存に解決）。
    try:
        import folder_paths
        base = Path(folder_paths.models_dir) / "insightface"
        cands.append(base / "models" / "buffalo_l")
        cands.append(base / "buffalo_l")
    except Exception:
        pass
    for c in cands:
        if c and (c / "w600k_r50.onnx").exists() and (c / "det_10g.onnx").exists():
            return c
    return None


def _init():
    global _det, _rec, _init_error
    if (_det is not None and _rec is not None) or _init_error is not None:
        return
    try:
        import insightface.model_zoo as mz
        pack = _find_pack()
        if pack is None:
            _init_error = "buffalo_l pack not found (set INSIGHTFACE_MODEL_DIR)"
            return
        ctx = int(os.environ.get("INSIGHTFACE_CTX", "0"))
        _det = mz.get_model(str(pack / "det_10g.onnx"))
        _det.prepare(ctx_id=ctx, input_size=(640, 640))
        _rec = mz.get_model(str(pack / "w600k_r50.onnx"))
        _rec.prepare(ctx_id=ctx)
    except Exception as e:
        _init_error = f"{type(e).__name__}: {e}"


def embed(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """最大の顔の L2正規化 512次元埋め込み。顔が無ければ None。要 _init()。"""
    from insightface.utils import face_align
    bboxes, kpss = _det.detect(img_bgr, max_num=0, metric="default")
    if bboxes.shape[0] == 0:
        return None
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    i = int(np.argmax(areas))
    aligned = face_align.norm_crop(img_bgr, kpss[i])
    feat = _rec.get_feat(aligned).flatten()
    n = np.linalg.norm(feat)
    return feat / n if n > 0 else None


def identity_score(cosine: float) -> float:
    """cosine類似度 → [0,1]。_COS_LOW以下で0、_COS_HIGH以上で1。"""
    return clamp01((cosine - _COS_LOW) / (_COS_HIGH - _COS_LOW))


def available() -> bool:
    _init()
    return _det is not None and _rec is not None
