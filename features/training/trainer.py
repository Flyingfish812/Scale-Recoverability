"""
VCNN training loop — thin wrapper that bridges luna.* common layer
with the existing training infrastructure.

For the full training implementation, see the original training/trainer.py.
This module provides the clean public API surface.
"""

from __future__ import annotations

# Re-export key functions from the existing training module during migration.
# After full migration, these will be implemented directly here using luna.*.
import sys
from pathlib import Path

# Bridge to original training code during migration
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.trainer import (  # noqa: E402
    train_vcnn,
    set_global_seed,
    resolve_torch_device,
    adjust_learning_rate,
    ObservationFeatureDataset,
    save_checkpoint,
    save_test_raw_artifacts,
)

__all__ = [
    "train_vcnn",
    "set_global_seed",
    "resolve_torch_device",
    "adjust_learning_rate",
    "ObservationFeatureDataset",
    "save_checkpoint",
    "save_test_raw_artifacts",
]
