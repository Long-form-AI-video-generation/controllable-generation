"""Pose control utilities. Importing this package loads no models."""

from __future__ import annotations

from typing import Any

__all__ = [
    "PoseRecordValidationError",
    "load_pose_record",
    "validate_pose_record",
]


def __getattr__(name: str) -> Any:
    # Imported lazily so ``python -m src.pose_models.record_validation`` does
    # not load the module twice.
    if name in __all__:
        from . import record_validation

        return getattr(record_validation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
