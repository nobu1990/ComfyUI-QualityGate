"""
ComfyUI/torch なしで検出カスケードを検証するスモークテスト。

    python -m pytest tests/            # pytest があれば
    python tests/test_cascade.py       # 素の実行でもOK（下の __main__ を使う）

合成画像で「ボケ画像は sharpness で落ちる」「顔なし画像は face_presence で落ちる」
という期待挙動だけ確認する。実写での精度は README のロードマップどおり後で測る。
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quality_checks import available_checks, run_cascade  # noqa: E402


def _sharp_texture(size=256):
    """高周波ノイズ入りの鮮鋭な画像。"""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def _blurred(size=256):
    img = _sharp_texture(size)
    return cv2.GaussianBlur(img, (0, 0), 8.0)


def test_registry_has_checks():
    assert "sharpness" in available_checks()
    assert "face_presence" in available_checks()


def test_sharp_beats_blurred():
    sharp_score, _ = run_cascade(_sharp_texture(), ["sharpness"])
    blur_score, _ = run_cascade(_blurred(), ["sharpness"])
    assert sharp_score > blur_score
    assert blur_score < 0.5  # ボケは低スコア


def test_no_face_fails_face_check():
    # ランダムノイズには顔がない → face_presence は passed=False になるはず
    _, results = run_cascade(_sharp_texture(), ["face_presence"])
    face = next(r for r in results if r.name == "face_presence")
    assert face.passed is False


def _run_all():
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} tests passed")


if __name__ == "__main__":
    _run_all()
