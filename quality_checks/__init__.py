"""生成物QAの検出チェック群。import した時点で各チェックがレジストリに登録される。"""

from . import body_proportion, face_presence, hands, sharpness  # noqa: F401  (登録の副作用)
from .base import (  # noqa: F401
    QAResult,
    available_checks,
    clamp01,
    register,
    run_cascade,
)
