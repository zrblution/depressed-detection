"""Shared constants and light schema utilities."""

from __future__ import annotations

from typing import Dict, Optional


DATASET_IDS = ("swdd", "twitter", "erisk")
SYMPTOM_DIMENSIONS = (
    "depressed_mood",
    "anhedonia",
    "sleep",
    "fatigue",
    "appetite_or_weight",
    "worthlessness_or_guilt",
    "concentration",
    "psychomotor",
    "suicidal_ideation",
)
CHANNEL_NAMES = {
    0: "self_disclosure",
    1: "episode_supported",
    2: "sparse_evidence",
    3: "mixed",
    4: "general",
}
DEFAULT_PRIORS = {
    "self_disclosure": 0.0,
    "episode_supported": 0.0,
    "sparse_evidence": 0.0,
}
DEFAULT_GLOBAL_STATS = {
    "total_posts": 0,
    "eligible_evidence_posts": 0,
    "posting_freq": 0.0,
    "active_span_days": 0,
    "temporal_burstiness": 0.0,
}
POST_MARKER_WIDTH = 4
META_TOKEN = "[META]"

_DISEASE_MENTION_MAP = {
    "none": "none",
    "current_self_claim": "current_self_claim",
    "self_history": "past_self_claim",
    "past_self_claim": "past_self_claim",
    "generic_topic": "general",
    "general": "general",
    "other_person": "other_person",
}


def empty_symptom_vector() -> Dict[str, int]:
    return {key: 0 for key in SYMPTOM_DIMENSIONS}


def make_post_marker(index: int) -> str:
    if index < 1:
        raise ValueError("post marker index must be >= 1")
    return f"[POST_{index:0{POST_MARKER_WIDTH}d}]"


def normalize_disease_mention_type(value: Optional[str]) -> str:
    text = (value or "none").strip()
    return _DISEASE_MENTION_MAP.get(text, "general")
