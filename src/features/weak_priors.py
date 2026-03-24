"""Weak-prior computation."""

from __future__ import annotations

from typing import Dict, List


def compute_p_sd(user_evidence_posts: List[dict]) -> float:
    if not user_evidence_posts:
        return 0.0
    score = 0.0
    for post in user_evidence_posts:
        mention_type = post.get("clinical_context", {}).get("disease_mention_type")
        anchors = post.get("clinical_context", {}).get("anchor_types", [])
        if mention_type == "current_self_claim":
            score += 0.4
        score += 0.2 * min(len(anchors), 2)
        if post.get("literal_self_evidence", False) and post.get("temporality") == "current":
            score += 0.2
        score += 0.1 * float(post.get("confidence", 0.0))
    return round(min(score / max(len(user_evidence_posts), 1), 1.0), 4)


def compute_p_ep(user_blocks: List[dict]) -> float:
    if not user_blocks:
        return 0.0
    best_block = max(user_blocks, key=lambda item: item["block_score"])
    score = (
        0.30 * min(best_block["block_post_count"] / 5.0, 1.0)
        + 0.20 * min(best_block["block_span_days"] / 14.0, 1.0)
        + 0.20 * min(best_block["symptom_category_count"] / 5.0, 1.0)
        + 0.15 * (1.0 if best_block["duration_support"] else 0.0)
        + 0.15 * min(best_block["functional_impairment_max"] / 3.0, 1.0)
    )
    return round(min(score, 1.0), 4)


def compute_p_sp(user_evidence_posts: List[dict], p_sd: float, p_ep: float) -> float:
    if p_sd >= 0.5 or p_ep >= 0.5:
        return 0.0
    if not user_evidence_posts or len(user_evidence_posts) > 3:
        return 0.0
    scores = sorted(
        (float(post.get("composite_evidence_score", 0.0)) for post in user_evidence_posts),
        reverse=True,
    )
    avg_confidence = sum(float(post.get("confidence", 0.0)) for post in user_evidence_posts) / len(user_evidence_posts)
    score = 0.5 * scores[0]
    if len(scores) > 1:
        score += 0.3 * scores[1]
    score += 0.2 * avg_confidence
    return round(min(score, 1.0), 4)


def compute_crisis_score(scored_posts: List[dict]) -> int:
    if not scored_posts:
        return 0
    return max(int(post.get("crisis_level", 0)) for post in scored_posts)


def compute_all_priors(user_evidence_posts: List[dict], user_blocks: List[dict], all_scored_posts: List[dict]) -> Dict[str, float]:
    p_sd = compute_p_sd(user_evidence_posts)
    p_ep = compute_p_ep(user_blocks)
    p_sp = compute_p_sp(user_evidence_posts, p_sd, p_ep)
    return {
        "self_disclosure": p_sd,
        "episode_supported": p_ep,
        "sparse_evidence": p_sp,
        "crisis_score": compute_crisis_score(all_scored_posts),
    }

