"""Evidence block construction."""

from __future__ import annotations

from datetime import datetime
from typing import List


ELIGIBLE_THRESHOLD = 0.3


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def filter_eligible_posts(scored_posts: List[dict]) -> List[dict]:
    eligible = [
        post
        for post in scored_posts
        if post.get("first_person", False)
        and post.get("literal_self_evidence", False)
        and float(post.get("composite_evidence_score", 0.0)) >= ELIGIBLE_THRESHOLD
    ]
    eligible.sort(key=lambda item: item["posting_time"])
    return eligible


def _compute_block_features(block_posts: List[dict], block_id: int) -> dict:
    post_ids = [post["post_id"] for post in block_posts]
    times = [_parse_time(post["posting_time"]) for post in block_posts]
    span_days = (max(times) - min(times)).days if len(times) > 1 else 0
    symptom_dims = set()
    anchors = set()
    for post in block_posts:
        for dim, value in post.get("symptom_vector", {}).items():
            if int(value) > 0:
                symptom_dims.add(dim)
        anchors.update(post.get("clinical_context", {}).get("anchor_types", []))
    repeated_days = len({time.date() for time in times})
    duration_support = any(post.get("duration", {}).get("has_hint", False) for post in block_posts)
    functional_impairment_max = max(int(post.get("functional_impairment", 0)) for post in block_posts)
    crisis_max = max(int(post.get("crisis_level", 0)) for post in block_posts)
    avg_confidence = sum(float(post.get("confidence", 0.0)) for post in block_posts) / max(len(block_posts), 1)
    block_score = (
        0.25 * min(len(block_posts) / 5.0, 1.0)
        + 0.20 * min(span_days / 14.0, 1.0)
        + 0.20 * min(len(symptom_dims) / 5.0, 1.0)
        + 0.15 * (1.0 if duration_support else 0.0)
        + 0.10 * min(functional_impairment_max / 3.0, 1.0)
        + 0.10 * min(avg_confidence, 1.0)
    )
    representative_posts = sorted(
        block_posts,
        key=lambda item: item.get("composite_evidence_score", 0.0),
        reverse=True,
    )[:3]
    return {
        "block_id": block_id,
        "post_ids": post_ids,
        "block_post_count": len(block_posts),
        "block_span_days": span_days,
        "symptom_category_count": len(symptom_dims),
        "repeated_days": repeated_days,
        "duration_support": duration_support,
        "functional_impairment_max": functional_impairment_max,
        "crisis_max": crisis_max,
        "clinical_anchor_count": len(anchors),
        "avg_confidence": round(avg_confidence, 4),
        "block_score": round(block_score, 4),
        "representative_posts": representative_posts,
    }


def build_evidence_blocks(eligible_posts: List[dict], *, max_gap_days: int = 7) -> List[dict]:
    if not eligible_posts:
        return []
    blocks = []
    current = [eligible_posts[0]]
    for post in eligible_posts[1:]:
        previous_time = _parse_time(current[-1]["posting_time"])
        current_time = _parse_time(post["posting_time"])
        if (current_time - previous_time).days <= max_gap_days:
            current.append(post)
        else:
            blocks.append(_compute_block_features(current, len(blocks)))
            current = [post]
    blocks.append(_compute_block_features(current, len(blocks)))
    return blocks

