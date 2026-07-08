"""
ノード配線の統合スモークテスト（ComfyUI/torch なしで実行）。

ハイフン入りフォルダ名でも import できるよう、パッケージとして明示ロードする。
IMAGE テンソルの代わりに numpy [B,H,W,C] float(0..1) を流し、
QualityFilterBatch が合格/不合格を分けて空にならず返すことを確認する。
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_pkg():
    spec = importlib.util.spec_from_file_location(
        "cqg", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cqg"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_batch():
    """3枚: ノイズ(鮮鋭・顔なし) / ボケ / 一様グレー。"""
    rng = np.random.default_rng(1)
    noise = rng.random((128, 128, 3), dtype=np.float32)
    import cv2
    blurred = cv2.GaussianBlur((noise * 255).astype(np.uint8), (0, 0), 8.0)
    blurred = blurred.astype(np.float32) / 255.0
    gray = np.full((128, 128, 3), 0.5, dtype=np.float32)
    return np.stack([noise, blurred, gray], axis=0)


def test_filter_and_gate():
    pkg = _load_pkg()
    Gate = pkg.NODE_CLASS_MAPPINGS["QualityGate"]
    Filt = pkg.NODE_CLASS_MAPPINGS["QualityFilterBatch"]

    batch = _fake_batch()

    images, score, all_passed, report = Gate().evaluate(batch, threshold=0.6, expected_faces=1)
    assert 0.0 <= score <= 1.0
    assert isinstance(report, str) and "QualityGate report" in report

    passed_t, rejected_t, passed_count, report2 = Filt().filter(batch, threshold=0.6, expected_faces=1)
    # 顔がある画像は無いので、expected_faces=1 だと全滅 → passed_count=0 のはず
    assert passed_count == 0
    # フォールバックで空テンソルにならないこと
    assert np.asarray(passed_t).shape[0] >= 1
    assert np.asarray(rejected_t).shape[0] == 3
    print(f"  gate score={score:.2f} all_passed={all_passed} passed_count={passed_count}")


if __name__ == "__main__":
    test_filter_and_gate()
    print("\nintegration test passed")
