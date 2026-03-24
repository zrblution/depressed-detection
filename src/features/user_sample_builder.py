"""Assemble user-level samples."""

from __future__ import annotations

from typing import List

from src.features.evidence_block import build_evidence_blocks, filter_eligible_posts
from src.features.global_history import build_global_history, compute_global_stats
from src.features.weak_priors import compute_all_priors
from src.utils.schemas import DEFAULT_PRIORS


def _sort_llm_risk_posts(posts: List[dict]) -> List[dict]:
    return sorted(posts, key=lambda item: item.get("composite_evidence_score", 0.0), reverse=True)


def _sort_template_risk_posts(posts: List[dict]) -> List[dict]:
    return sorted(posts, key=lambda item: item.get("risk_score", 0.0), reverse=True)


def build_depressed_user_sample(
    user_id: str,
    all_scored_posts: List[dict],
    risk_posts_a: List[dict],
    risk_posts_b: List[dict],
    all_standardized_posts: List[dict],
    *,
    max_gap_days: int = 7,
    max_blocks: int = 3,
    global_history_coverage: float = 0.6,
    global_history_max_per_segment: int = 128,
) -> dict:
    eligible_posts = filter_eligible_posts(all_scored_posts)
    blocks = build_evidence_blocks(eligible_posts, max_gap_days=max_gap_days)
    priors = compute_all_priors(eligible_posts, blocks, all_scored_posts)
    global_segments = build_global_history(
        all_standardized_posts,
        coverage=global_history_coverage,
        max_per_segment=global_history_max_per_segment,
    )
    global_stats = compute_global_stats(all_standardized_posts, len(eligible_posts))
    top_blocks = sorted(blocks, key=lambda item: item["block_score"], reverse=True)[:max_blocks]
    return {
        "user_id": user_id,
        "label": 1,
        "priors": {
            "self_disclosure": priors["self_disclosure"],
            "episode_supported": priors["episode_supported"],
            "sparse_evidence": priors["sparse_evidence"],
        },
        "crisis_score": priors["crisis_score"],
        "risk_posts_llm": [
            {
                "post_id": post["post_id"],
                "text": post["text"],
                "composite_evidence_score": post.get("composite_evidence_score", 0.0),
                "crisis_level": post.get("crisis_level", 0),
                "temporality": post.get("temporality", "unclear"),
            }
            for post in _sort_llm_risk_posts(risk_posts_a)
        ],
        "risk_posts_template": [
            {
                "post_id": post["post_id"],
                "text": post["text"],
                "risk_score": post.get("risk_score", 0.0),
                "matched_dimensions": post.get("matched_dimensions", []),
            }
            for post in _sort_template_risk_posts(risk_posts_b)
        ],
        "episode_blocks": top_blocks,
        "global_history_posts": [
            [{"post_id": post["post_id"], "text": post["text"]} for post in segment]
            for segment in global_segments
        ],
        "global_stats": global_stats,
    }


def build_template_only_user_sample(
    user_id: str,
    label: int,
    risk_posts_b: List[dict],
    all_standardized_posts: List[dict],
    *,
    global_history_coverage: float = 0.6,
    global_history_max_per_segment: int = 128,
) -> dict:
    global_segments = build_global_history(
        all_standardized_posts,
        coverage=global_history_coverage,
        max_per_segment=global_history_max_per_segment,
    )
    global_stats = compute_global_stats(all_standardized_posts, 0)
    return {
        "user_id": user_id,
        "label": int(label),
        "priors": dict(DEFAULT_PRIORS),
        "crisis_score": 0,
        "risk_posts_llm": [],
        "risk_posts_template": [
            {
                "post_id": post["post_id"],
                "text": post["text"],
                "risk_score": post.get("risk_score", 0.0),
                "matched_dimensions": post.get("matched_dimensions", []),
            }
            for post in _sort_template_risk_posts(risk_posts_b)
        ],
        "episode_blocks": [],
        "global_history_posts": [
            [{"post_id": post["post_id"], "text": post["text"]} for post in segment]
            for segment in global_segments
        ],
        "global_stats": global_stats,
    }
