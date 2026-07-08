"""
quality_checks/base.py

生成物QAの「検出カスケード」を組み立てるための土台。

各チェックは 1 枚の BGR 画像(np.uint8, HxWx3) を受け取り、
QAResult を返す純粋関数として実装する。ComfyUI にも torch にも依存しないので
単体でテスト・調整できる（SeedCount の seed_prelabel.py と同じ設計思想）。

スコアは [0.0, 1.0] に正規化する。1.0 が「破綻なし」、0.0 が「破綻」。
ノード側で各チェックのスコアを重み付き集約し、閾値で合否を決める。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np


@dataclass
class QAResult:
    """1 つのチェックの結果。"""

    name: str
    score: float            # [0,1] 高いほど良品
    passed: bool            # このチェック単体の合否
    detail: str = ""        # 人間が読む一言（"faces=1, sharpness=142.3" など）
    metrics: Dict[str, float] = field(default_factory=dict)  # 数値の生ログ
    skipped: bool = False   # 依存欠如などで評価できなかった（集約から除外する）


# チェック関数のシグネチャ: (img_bgr, **params) -> QAResult
CheckFn = Callable[..., QAResult]

# 名前 -> チェック関数 のレジストリ。@register で自動登録される。
REGISTRY: Dict[str, CheckFn] = {}


def register(name: str) -> Callable[[CheckFn], CheckFn]:
    """チェック関数をレジストリに登録するデコレータ。"""

    def _wrap(fn: CheckFn) -> CheckFn:
        if name in REGISTRY:
            raise ValueError(f"check '{name}' is already registered")
        REGISTRY[name] = fn
        return fn

    return _wrap


def available_checks() -> List[str]:
    return sorted(REGISTRY.keys())


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def run_cascade(
    img_bgr: np.ndarray,
    checks: List[str],
    weights: Dict[str, float] | None = None,
    params: Dict[str, dict] | None = None,
) -> tuple[float, List[QAResult]]:
    """
    指定した順にチェックを実行し、(集約スコア, 各結果) を返す。

    集約スコアは重み付き平均。weights 未指定なら等重み。
    どれか 1 つでも passed=False なら「破綻あり」と扱いたい場合は
    呼び出し側で all(r.passed for r in results if not r.skipped) を見ればよい。

    skipped=True の結果（依存欠如で評価不能など）は集約スコアから除外する。
    """
    weights = weights or {}
    params = params or {}
    results: List[QAResult] = []
    for name in checks:
        fn = REGISTRY.get(name)
        if fn is None:
            results.append(
                QAResult(name=name, score=0.0, passed=False,
                         detail=f"unknown check '{name}'")
            )
            continue
        results.append(fn(img_bgr, **params.get(name, {})))

    scored = [r for r in results if not r.skipped]
    if not scored:
        return 0.0, results

    total_w = sum(weights.get(r.name, 1.0) for r in scored)
    agg = sum(r.score * weights.get(r.name, 1.0) for r in scored) / total_w
    return clamp01(agg), results
