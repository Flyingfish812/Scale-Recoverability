"""
VCNN sweep runner — thin wrapper bridging to original training/sweep.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.sweep import (  # noqa: E402
    run_vcnn_sweep,
    load_field_array,
    encode_value,
    encode_mask_tag,
)

__all__ = [
    "run_vcnn_sweep",
    "load_field_array",
    "encode_value",
    "encode_mask_tag",
]
