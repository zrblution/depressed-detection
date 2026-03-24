"""Scheduler helpers."""

from __future__ import annotations

import math


def cosine_decay(start: float, end: float, step: int, total_steps: int) -> float:
    progress = step / max(total_steps, 1)
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * progress))

