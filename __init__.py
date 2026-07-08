"""
ComfyUI-QualityGate

生成物のQA/破綻検出をノード化するカスタムノードパック。
ComfyUI はこの __init__.py の NODE_CLASS_MAPPINGS を読み込む。
"""

from .nodes import (
    CompositeRank,
    CompositeRankFolder,
    ProportionMatchRank,
    QualityFilterBatch,
    QualityGate,
    SaveToFolder,
)

NODE_CLASS_MAPPINGS = {
    "QualityGate": QualityGate,
    "QualityFilterBatch": QualityFilterBatch,
    "ProportionMatchRank": ProportionMatchRank,
    "CompositeRank": CompositeRank,
    "CompositeRankFolder": CompositeRankFolder,
    "SaveToFolder": SaveToFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QualityGate": "Quality Gate (score & pass)",
    "QualityFilterBatch": "Quality Filter Batch (deliver only passing)",
    "ProportionMatchRank": "Proportion Match Rank (head/body vs reference)",
    "CompositeRank": "Composite Rank (identity × proportion × sharpness)",
    "CompositeRankFolder": "Composite Rank Folder (streaming, memory-safe)",
    "SaveToFolder": "Save To Folder (create/specify path)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
