"""User-level dataset and formatting."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.utils.schemas import META_TOKEN, make_post_marker


def _sorted_risk_posts(posts: List[dict], *, llm: bool) -> List[dict]:
    key = "composite_evidence_score" if llm else "risk_score"
    return sorted(posts, key=lambda item: item.get(key, 0.0), reverse=True)


def _ensure_non_empty_risk_posts(sample: dict, risk_posts: List[dict]) -> List[dict]:
    if risk_posts:
        return risk_posts
    for segment in sample.get("global_history_posts", []):
        if segment:
            first = segment[0]
            return [{"post_id": first["post_id"], "text": first["text"], "risk_score": 0.0, "matched_dimensions": []}]
    return [{"post_id": f"{sample['user_id']}__fallback", "text": "[NO_CONTENT]", "risk_score": 0.0, "matched_dimensions": []}]


def _resolve_evidence_target_score(post: dict) -> float:
    if "composite_evidence_score" in post:
        return min(max(float(post.get("composite_evidence_score", 0.0)), 0.0), 1.0)
    if "risk_score" in post:
        return min(max(float(post.get("risk_score", 0.0)), 0.0), 1.0)
    return 0.0


def format_user_sample(
    sample: dict,
    *,
    is_training: bool,
    p_risk_swap: float = 0.5,
    p_meta_drop: float = 0.5,
    p_block_drop: float = 0.4,
    p_prior_drop: float = 0.3,
    p_post_drop: float = 0.3,
    max_risk_posts: Optional[int] = None,
    max_global_posts_per_segment: Optional[int] = None,
    force_risk_source: Optional[str] = None,
) -> dict:
    label = int(sample["label"])
    is_depressed = label == 1
    risk_posts_llm = _sorted_risk_posts(sample.get("risk_posts_llm", []), llm=True)
    risk_posts_template = _sorted_risk_posts(sample.get("risk_posts_template", []), llm=False)

    if force_risk_source == "llm" and risk_posts_llm:
        risk_posts = risk_posts_llm
        has_meta = True
    elif force_risk_source == "template":
        risk_posts = risk_posts_template
        has_meta = False
    elif is_training and is_depressed and risk_posts_llm:
        if random.random() < p_risk_swap:
            risk_posts = risk_posts_template
            has_meta = False
        else:
            risk_posts = risk_posts_llm
            has_meta = True
    else:
        risk_posts = risk_posts_template
        has_meta = False

    risk_posts = _ensure_non_empty_risk_posts(sample, risk_posts)
    if has_meta and is_training and random.random() < p_meta_drop:
        has_meta = False

    episode_blocks = list(sample.get("episode_blocks", []))
    if is_training and is_depressed and random.random() < p_block_drop:
        episode_blocks = []

    priors = dict(sample.get("priors", {}))
    priors.setdefault("self_disclosure", 0.0)
    priors.setdefault("episode_supported", 0.0)
    priors.setdefault("sparse_evidence", 0.0)
    crisis_score = int(sample.get("crisis_score", 0))
    if is_training and is_depressed and random.random() < p_prior_drop:
        priors = {"self_disclosure": 0.0, "episode_supported": 0.0, "sparse_evidence": 0.0}
        crisis_score = 0

    if is_training and len(risk_posts) > 3 and random.random() < p_post_drop:
        keep_count = max(int(len(risk_posts) * 0.7), 3)
        risk_posts = random.sample(risk_posts, keep_count)
        risk_posts = _sorted_risk_posts(risk_posts, llm=has_meta)

    if max_risk_posts is not None:
        risk_posts = risk_posts[:max_risk_posts]

    risk_texts: List[str] = []
    risk_markers: List[str] = []
    risk_post_ids: List[str] = []
    evidence_target_scores: List[float] = []
    for idx, post in enumerate(risk_posts):
        marker = make_post_marker(idx + 1)
        risk_markers.append(marker)
        risk_post_ids.append(post["post_id"])
        if has_meta and "composite_evidence_score" in post:
            meta_info = {
                "symptom_strength": f"{post.get('composite_evidence_score', 0.0):.2f}",
                "crisis": int(post.get("crisis_level", 0)),
                "temporality": post.get("temporality", "unclear"),
            }
            meta_str = " ".join(f"{key}={value}" for key, value in meta_info.items())
            risk_texts.append(f"{marker} {post['text']} {META_TOKEN} {meta_str}")
        else:
            risk_texts.append(f"{marker} {post['text']}")
        evidence_target_scores.append(_resolve_evidence_target_score(post))

    block_texts: List[str] = []
    block_markers: List[str] = []
    next_index = len(risk_markers) + 1
    max_block_score = 0.0
    for block in episode_blocks[:3]:
        max_block_score = max(max_block_score, float(block.get("block_score", 0.0)))
        for post in block.get("representative_posts", [])[:3]:
            marker = make_post_marker(next_index)
            next_index += 1
            block_markers.append(marker)
            block_texts.append(f"{marker} {post['text']}")

    global_segment_texts: List[List[str]] = []
    global_segment_markers: List[List[str]] = []
    for segment in sample.get("global_history_posts", []):
        if max_global_posts_per_segment is not None:
            segment = segment[:max_global_posts_per_segment]
        seg_texts: List[str] = []
        seg_markers: List[str] = []
        for post in segment:
            marker = make_post_marker(next_index)
            next_index += 1
            seg_markers.append(marker)
            seg_texts.append(f"{marker} {post['text']}")
        global_segment_markers.append(seg_markers)
        global_segment_texts.append(seg_texts)

    global_stats = sample.get("global_stats", {})
    eligible_norm = min(float(global_stats.get("eligible_evidence_posts", 0)) / 20.0, 1.0)
    total_posts_norm = min(float(global_stats.get("total_posts", 0)) / 1000.0, 1.0)
    active_span_norm = min(float(global_stats.get("active_span_days", 0)) / 365.0, 1.0)
    stats_tensor = torch.tensor(
        [
            float(global_stats.get("posting_freq", 0.0)),
            float(global_stats.get("temporal_burstiness", 0.0)),
            eligible_norm,
            total_posts_norm,
            active_span_norm,
        ],
        dtype=torch.float32,
    )
    pi_u = torch.tensor(
        [priors["self_disclosure"], priors["episode_supported"], priors["sparse_evidence"]],
        dtype=torch.float32,
    )
    crisis_tensor = torch.tensor([crisis_score / 3.0], dtype=torch.float32)
    meta_vector = torch.tensor(
        [
            priors["self_disclosure"],
            priors["episode_supported"],
            priors["sparse_evidence"],
            crisis_score / 3.0,
            float(global_stats.get("posting_freq", 0.0)),
            float(global_stats.get("temporal_burstiness", 0.0)),
            total_posts_norm,
            active_span_norm,
            eligible_norm,
            max_block_score,
        ],
        dtype=torch.float32,
    )
    return {
        "user_id": sample["user_id"],
        "label": torch.tensor([label], dtype=torch.float32),
        "risk_texts": risk_texts,
        "risk_markers": risk_markers,
        "risk_post_ids": risk_post_ids,
        "evidence_target_scores": torch.tensor(evidence_target_scores, dtype=torch.float32),
        "block_texts": block_texts,
        "block_markers": block_markers,
        "global_segment_texts": global_segment_texts,
        "global_segment_markers": global_segment_markers,
        "pi_u": pi_u,
        "crisis": crisis_tensor,
        "stats": stats_tensor,
        "meta_vector": meta_vector,
        "is_depressed": is_depressed,
    }


class UserDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        *,
        is_training: bool = True,
        p_risk_swap: float = 0.5,
        p_meta_drop: float = 0.5,
        p_block_drop: float = 0.4,
        p_prior_drop: float = 0.3,
        p_post_drop: float = 0.3,
        max_risk_posts: Optional[int] = None,
        max_global_posts_per_segment: Optional[int] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.samples = []
        with Path(data_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    self.samples.append(json.loads(text))
        if max_samples is not None and len(self.samples) > max_samples:
            rng = random.Random(seed)
            self.samples = rng.sample(self.samples, k=int(max_samples))
        self.is_training = is_training
        self.p_risk_swap = p_risk_swap
        self.p_meta_drop = p_meta_drop
        self.p_block_drop = p_block_drop
        self.p_prior_drop = p_prior_drop
        self.p_post_drop = p_post_drop
        self.max_risk_posts = max_risk_posts
        self.max_global_posts_per_segment = max_global_posts_per_segment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return format_user_sample(
            self.samples[index],
            is_training=self.is_training,
            p_risk_swap=self.p_risk_swap,
            p_meta_drop=self.p_meta_drop,
            p_block_drop=self.p_block_drop,
            p_prior_drop=self.p_prior_drop,
            p_post_drop=self.p_post_drop,
            max_risk_posts=self.max_risk_posts,
            max_global_posts_per_segment=self.max_global_posts_per_segment,
        )


def single_user_collate(batch: List[dict]) -> dict:
    return batch[0]
