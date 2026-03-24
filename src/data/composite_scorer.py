"""Composite evidence scoring for path-A risk posts."""

from __future__ import annotations

import math
from typing import Iterable, List

from src.utils.schemas import SYMPTOM_DIMENSIONS


def compute_dynamic_k(total_posts: int) -> int:
    if total_posts >= 160:
        return math.ceil(total_posts * 0.125)
    if total_posts >= 20:
        return 20
    return total_posts


def compute_composite_evidence_score(post: dict) -> float:
    symptom_vector = post.get("symptom_vector", {})
    values = [int(symptom_vector.get(dim, 0)) for dim in SYMPTOM_DIMENSIONS]
    if not values:
        return 0.0

    max_symptom = max(values) / 3.0
    symptom_coverage = sum(value > 0 for value in values) / len(SYMPTOM_DIMENSIONS)
    symptom_strength = 0.6 * max_symptom + 0.4 * symptom_coverage
    crisis_norm = min(max(float(post.get("crisis_level", 0)) / 3.0, 0.0), 1.0)
    has_anchor = 1.0 if post.get("clinical_context", {}).get("anchor_types") else 0.0
    duration_support = 1.0 if post.get("duration", {}).get("has_hint", False) else 0.0
    confidence = min(max(float(post.get("confidence", 0.0)), 0.0), 1.0)
    self_disclosure = 1.0 if (post.get("first_person") and post.get("literal_self_evidence")) else 0.0
    score = (
        0.35 * symptom_strength
        + 0.20 * crisis_norm
        + 0.20 * has_anchor
        + 0.10 * duration_support
        + 0.10 * confidence
        + 0.05 * self_disclosure
    )
    return round(min(max(score, 0.0), 1.0), 4)


def add_composite_scores(posts: Iterable[dict]) -> List[dict]:
    scored_posts: List[dict] = []
    for post in posts:
        enriched = dict(post)
        enriched["composite_evidence_score"] = compute_composite_evidence_score(post)
        scored_posts.append(enriched)
    return scored_posts


def get_risk_posts_a(user_scored_posts: List[dict]) -> List[dict]:
    sorted_posts = sorted(
        user_scored_posts,
        key=lambda item: item.get("composite_evidence_score", 0.0),
        reverse=True,
    )
    return sorted_posts[: compute_dynamic_k(len(sorted_posts))]

