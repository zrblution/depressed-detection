"""Inference pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

from src.data.raw_loader import normalize_bool_like
from src.data.template_screener import PHQ9TemplateScreener
from src.features.global_history import build_global_history, compute_global_stats
from src.model.full_model import WPGMoEModel
from src.utils.schemas import CHANNEL_NAMES, make_post_marker


def _strip_state_dict_wrappers(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _validate_checkpoint_compatibility(model: WPGMoEModel, state_dict: dict[str, torch.Tensor], *, model_path: str) -> dict[str, torch.Tensor]:
    normalized_state_dict = _strip_state_dict_wrappers(state_dict)
    model_state = model.state_dict()

    if any("lora" in key.lower() for key in normalized_state_dict):
        raise RuntimeError(
            f"Checkpoint at {model_path} contains LoRA parameters, which are incompatible with the encoder-only mainline."
        )

    matching = []
    shape_mismatches = []
    for key, tensor in normalized_state_dict.items():
        expected = model_state.get(key)
        if expected is None:
            continue
        if tuple(expected.shape) != tuple(tensor.shape):
            shape_mismatches.append((key, tuple(tensor.shape), tuple(expected.shape)))
        else:
            matching.append(key)

    if not matching:
        raise RuntimeError(
            f"Checkpoint/backbone mismatch for {model_path}: no parameter names overlap with the current mDeBERTa model."
        )

    if shape_mismatches:
        sample = "; ".join(
            f"{key}: ckpt{ckpt_shape} != model{model_shape}"
            for key, ckpt_shape, model_shape in shape_mismatches[:5]
        )
        raise RuntimeError(
            f"Checkpoint/backbone mismatch for {model_path}: tensor shape differences detected ({sample})."
        )

    return normalized_state_dict


def raw_user_to_standardized_posts(raw_user: dict) -> tuple[str, List[dict]]:
    user_id = str(raw_user["nickname"])
    posts = []
    for idx, tweet in enumerate(raw_user.get("tweets", [])):
        posts.append(
            {
                "user_id": user_id,
                "post_id": f"{user_id}__{idx}",
                "text": str(tweet.get("tweet_content", "")),
                "posting_time": str(tweet.get("posting_time", "")),
                "tweet_is_original": normalize_bool_like(tweet.get("tweet_is_original", "True")),
            }
        )
    posts.sort(key=lambda item: item["posting_time"])
    return user_id, posts


class InferencePipeline:
    def __init__(
        self,
        *,
        model_path: str,
        model_config: dict,
        screener_model: str = "/home/tos_lx/basemodel/gte-small-zh",
        language: str = "zh",
        selection_cfg: dict | None = None,
        device: str = "cuda",
        screener_device: str | None = None,
        screener_batch_size: int = 256,
        screener_encode_chunk_size: int | None = None,
        screener_multi_gpu: bool = False,
        screener_target_devices: list[str] | None = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.screener = PHQ9TemplateScreener(
            model_name=screener_model,
            language=language,
            selection_cfg=selection_cfg,
            device=screener_device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
            batch_size=screener_batch_size,
            encode_chunk_size=screener_encode_chunk_size,
            multi_gpu=screener_multi_gpu,
            target_devices=screener_target_devices,
        )
        self.model = WPGMoEModel(model_config).to(self.device)
        checkpoint_path = str(Path(model_path))
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        state_dict = _validate_checkpoint_compatibility(self.model, state_dict, model_path=checkpoint_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def predict_from_raw_user(self, raw_user: dict) -> dict:
        user_id, posts = raw_user_to_standardized_posts(raw_user)
        return self.predict(user_id, posts)

    @torch.no_grad()
    def predict(self, user_id: str, posts: List[dict]) -> dict:
        if not posts:
            return self._empty_result(user_id)
        risk_posts = self.screener.screen_user(posts)
        if not risk_posts:
            return self._empty_result(user_id, total_posts=len(posts))
        global_segments = build_global_history(posts)
        global_stats = compute_global_stats(posts, 0)

        risk_texts = []
        risk_markers = []
        risk_post_ids = []
        for idx, post in enumerate(risk_posts):
            marker = make_post_marker(idx + 1)
            risk_markers.append(marker)
            risk_post_ids.append(post["post_id"])
            risk_texts.append(f"{marker} {post['text']}")

        next_index = len(risk_markers) + 1
        global_segment_texts = []
        global_segment_markers = []
        for segment in global_segments:
            seg_texts = []
            seg_markers = []
            for post in segment:
                marker = make_post_marker(next_index)
                next_index += 1
                seg_markers.append(marker)
                seg_texts.append(f"{marker} {post['text']}")
            global_segment_texts.append(seg_texts)
            global_segment_markers.append(seg_markers)

        total_posts_norm = min(float(global_stats.get("total_posts", 0)) / 1000.0, 1.0)
        active_span_norm = min(float(global_stats.get("active_span_days", 0)) / 365.0, 1.0)
        stats = torch.tensor(
            [
                float(global_stats.get("posting_freq", 0.0)),
                float(global_stats.get("temporal_burstiness", 0.0)),
                0.0,
                total_posts_norm,
                active_span_norm,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        pi_u = torch.zeros(3, dtype=torch.float32, device=self.device)
        crisis = torch.zeros(1, dtype=torch.float32, device=self.device)
        meta_vector = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, float(global_stats.get("posting_freq", 0.0)), float(global_stats.get("temporal_burstiness", 0.0)), total_posts_norm, active_span_norm, 0.0, 0.0],
            dtype=torch.float32,
            device=self.device,
        )
        output = self.model(
            risk_post_texts=risk_texts,
            risk_post_markers=risk_markers,
            risk_post_ids=risk_post_ids,
            block_post_texts=[],
            block_post_markers=[],
            global_segment_texts=global_segment_texts,
            global_segment_markers=global_segment_markers,
            pi_u=pi_u,
            crisis=crisis,
            stats=stats,
            meta_vector=meta_vector,
        )
        depressed_prob = float(torch.sigmoid(output["logit"]).item())
        label = 1 if depressed_prob > 0.5 else 0
        top_indices = output["evidence_top_indices"].detach().cpu().tolist()
        top_scores = output["top_evidence_scores"].detach().cpu().tolist()
        evidence_post_ids = [risk_post_ids[idx] for idx in top_indices]
        evidence_texts = [risk_posts[idx]["text"] for idx in top_indices]
        gate_weights = output["gate_weights"].detach().cpu().tolist()
        dominant_channel = CHANNEL_NAMES[int(output["gate_weights"].argmax().item())]
        return {
            "user_id": user_id,
            "label": label,
            "depressed_logit": round(depressed_prob, 4),
            "crisis_score": 0,
            "gate_weights": [round(weight, 4) for weight in gate_weights],
            "dominant_channel": dominant_channel,
            "evidence_post_ids": evidence_post_ids,
            "evidence_scores": [round(float(score), 4) for score in top_scores],
            "evidence_texts": evidence_texts,
            "total_posts": len(posts),
            "risk_posts_count": len(risk_posts),
        }

    @torch.no_grad()
    def predict_batch(self, users: Dict[str, List[dict]]) -> List[dict]:
        from tqdm import tqdm

        return [self.predict(user_id, posts) for user_id, posts in tqdm(users.items(), desc="推理中")]

    def _empty_result(self, user_id: str, *, total_posts: int = 0) -> dict:
        return {
            "user_id": user_id,
            "label": 0,
            "depressed_logit": 0.0,
            "crisis_score": 0,
            "gate_weights": [0.2] * 5,
            "dominant_channel": "general",
            "evidence_post_ids": [],
            "evidence_scores": [],
            "evidence_texts": [],
            "total_posts": total_posts,
            "risk_posts_count": 0,
        }
