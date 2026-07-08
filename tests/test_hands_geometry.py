"""
手の幾何プラウジビリティ関数の決定的テスト。

_hand_plausibility は純 numpy なので mediapipe 無しでも検証できる。
「自然な開いた手」は高スコア、「指先が癒着した手」は低スコアになることを確認する。
（MediaPipe 実推論の end-to-end は tests/test_hands_live.py を wan22 env で回す）
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quality_checks.hands import _hand_plausibility  # noqa: E402


def _good_hand() -> np.ndarray:
    return np.array([
        [0.50, 0.90],                               # 0 wrist
        [0.40, 0.85], [0.34, 0.80], [0.30, 0.75], [0.27, 0.70],   # thumb 1-4
        [0.44, 0.62], [0.43, 0.52], [0.42, 0.45], [0.42, 0.40],   # index 5-8
        [0.50, 0.60], [0.50, 0.49], [0.50, 0.41], [0.50, 0.35],   # middle 9-12
        [0.56, 0.62], [0.57, 0.52], [0.58, 0.45], [0.58, 0.40],   # ring 13-16
        [0.62, 0.66], [0.64, 0.58], [0.65, 0.52], [0.66, 0.48],   # pinky 17-20
    ], dtype=np.float64)


def _fused_hand() -> np.ndarray:
    """指先(8,12,16,20)をほぼ同一点に寄せた「癒着」ケース。"""
    pts = _good_hand().copy()
    for tip in (8, 12, 16, 20):
        pts[tip] = [0.50, 0.38]
    return pts


def test_good_hand_scores_high():
    score, m = _hand_plausibility(_good_hand())
    assert score > 0.8, (score, m)


def test_fused_hand_scores_low():
    score, m = _hand_plausibility(_fused_hand())
    assert score < 0.5, (score, m)
    assert m["min_tip_gap"] < 0.08


def test_good_beats_fused():
    g, _ = _hand_plausibility(_good_hand())
    f, _ = _hand_plausibility(_fused_hand())
    assert g > f


def _run_all():
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  ok  {name}"); n += 1
    print(f"\n{n} tests passed")


if __name__ == "__main__":
    _run_all()
