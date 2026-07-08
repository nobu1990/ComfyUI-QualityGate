"""
ratio_score（参照Rへの指数減衰採点）の決定的テスト。mediapipe不要・純関数。
MediaPipe実推論の end-to-end は wan22 env で tools/diagnose_proportion4.py 等で確認。
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quality_checks.body_proportion import DEFAULT_TOL, ratio_score  # noqa: E402


def test_exact_match_is_full():
    assert ratio_score(0.385, 0.385) == 1.0


def test_exponential_decay():
    # ズレ tol で exp(-1)=0.368、2tol で exp(-2)=0.135
    assert abs(ratio_score(0.385 + 0.02, 0.385, tol=0.02) - math.exp(-1)) < 1e-9
    assert abs(ratio_score(0.385 + 0.04, 0.385, tol=0.02) - math.exp(-2)) < 1e-9


def test_never_fully_zero():
    # 遠くても0にならず順位がつく（飽和しない）
    assert ratio_score(0.385 + 0.2, 0.385) > 0.0


def test_symmetry():
    t = 0.385
    assert abs(ratio_score(t + 0.008, t) - ratio_score(t - 0.008, t)) < 1e-9


def test_closer_scores_higher():
    t = 0.385
    assert ratio_score(t + 0.005, t) > ratio_score(t + 0.015, t)


def test_default_tol_value():
    assert DEFAULT_TOL == 0.02


def _run_all():
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  ok  {name}"); n += 1
    print(f"\n{n} tests passed")


if __name__ == "__main__":
    _run_all()
