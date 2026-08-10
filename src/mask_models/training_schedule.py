"""Pure helpers for the mask training schedule."""

from __future__ import annotations

import math


def optimizer_steps_per_epoch(
    num_batches: int,
    gradient_accumulation_steps: int,
) -> int:
    if num_batches < 0:
        raise ValueError("num_batches must be non-negative")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    return math.ceil(num_batches / gradient_accumulation_steps)


def expected_optimizer_steps(
    num_batches: int,
    gradient_accumulation_steps: int,
    epochs: int,
) -> int:
    if epochs < 0:
        raise ValueError("epochs must be non-negative")
    return optimizer_steps_per_epoch(
        num_batches,
        gradient_accumulation_steps,
    ) * epochs

