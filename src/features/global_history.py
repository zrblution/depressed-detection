"""Global history sampling."""

from __future__ import annotations

import math
from datetime import datetime
from typing import List

import numpy as np

from src.utils.schemas import DEFAULT_GLOBAL_STATS


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_global_history(
    all_posts: List[dict],
    *,
    segments: int = 8,
    coverage: float = 0.6,
    max_per_segment: int = 128,
) -> List[List[dict]]:
    if not all_posts:
        return [[] for _ in range(segments)]

    ordered = sorted(all_posts, key=lambda item: item["posting_time"])
    total = len(ordered)
    k_seg = min(math.ceil(coverage * total / segments), max_per_segment)
    segment_size = math.ceil(total / segments)
    sampled_segments: List[List[dict]] = []
    for segment_idx in range(segments):
        start = segment_idx * segment_size
        end = min(start + segment_size, total)
        segment_posts = ordered[start:end]
        if len(segment_posts) <= k_seg:
            sampled_segments.append(segment_posts)
            continue
        stride = len(segment_posts) / max(k_seg, 1)
        indices = [min(int(idx * stride), len(segment_posts) - 1) for idx in range(k_seg)]
        sampled_segments.append([segment_posts[idx] for idx in indices])
    return sampled_segments


def compute_global_stats(all_posts: List[dict], eligible_evidence_count: int) -> dict:
    if not all_posts:
        return dict(DEFAULT_GLOBAL_STATS)
    ordered = sorted(all_posts, key=lambda item: item["posting_time"])
    times = [_parse_time(post["posting_time"]) for post in ordered]
    span_days = (max(times) - min(times)).days
    posting_freq = len(ordered) / max(span_days, 1)
    if len(times) > 1:
        intervals = np.array(
            [(times[i + 1] - times[i]).total_seconds() / 86400.0 for i in range(len(times) - 1)],
            dtype=np.float32,
        )
        burstiness = float(intervals.std() / max(intervals.mean(), 1e-6))
    else:
        burstiness = 0.0
    return {
        "total_posts": len(ordered),
        "eligible_evidence_posts": int(eligible_evidence_count),
        "posting_freq": round(float(posting_freq), 4),
        "active_span_days": int(span_days),
        "temporal_burstiness": round(float(burstiness), 4),
    }

