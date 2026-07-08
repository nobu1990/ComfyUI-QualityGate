"""
identity_score（cosine→[0,1]正規化）の決定的テスト。insightface不要・純関数。
実埋め込みの end-to-end は wan22 env で検証する。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quality_checks.identity import identity_score  # noqa: E402


def test_low_and_high_clamp():
    assert identity_score(0.30) == 0.0     # 別人域
    assert identity_score(0.10) == 0.0
    assert identity_score(0.90) == 1.0     # 高一致
    assert identity_score(0.99) == 1.0


def test_monotonic():
    assert identity_score(0.90) > identity_score(0.70) > identity_score(0.50)


def test_midpoint():
    # 帯の中央 0.60 は概ね 0.5
    assert abs(identity_score(0.60) - 0.5) < 1e-6


def _run_all():
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  ok  {name}"); n += 1
    print(f"\n{n} tests passed")


if __name__ == "__main__":
    _run_all()
