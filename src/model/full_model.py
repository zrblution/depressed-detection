"""Full WPG-MoE model wrapper."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .evidence_head import EvidenceHead
from .expert_network import ExpertGroup
from .gate_network import GateNetwork
from .moe_head import MoEHead
from .post_encoder import PostEncoder
from .user_representation import UserRepresentationModule


class WPGMoEModel(nn.Module):
    def __init__(self, config: Dict[str, object]) -> None:
        super().__init__()
        self.encoder = PostEncoder(
            model_name=str(config.get("model_name", "/home/tos_lx/basemodel/mdeberta-v3-base")),
            max_post_tokens=int(config.get("max_post_tokens", 512)),
            num_post_markers=int(config.get("num_post_markers", 2048)),
            pooling_strategy=str(config.get("pooling_strategy", "auto")),
            torch_dtype=config.get("torch_dtype", "auto"),
            gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
        )
        hidden_dim = self.encoder.hidden_dim
        expert_output = int(config.get("expert_output", 256))
        num_experts = int(config.get("num_experts", 5))
        stats_dim = int(config.get("stats_dim", 5))
        meta_dim = int(config.get("meta_dim", 10))
        self.user_representation = UserRepresentationModule(hidden_dim, stats_dim=stats_dim)
        self.gate = GateNetwork(
            hidden_dim,
            num_experts=num_experts,
            gate_hidden=int(config.get("gate_hidden", 256)),
            stats_dim=stats_dim,
            prior_dim=4,
        )
        self.experts = ExpertGroup(
            hidden_dim,
            meta_dim=meta_dim,
            expert_hidden=int(config.get("expert_hidden", 512)),
            expert_output=expert_output,
            num_experts=num_experts,
        )
        self.moe_head = MoEHead(expert_output_dim=expert_output)
        self.evidence_head = EvidenceHead(post_dim=hidden_dim, user_dim=expert_output, gate_dim=num_experts)
        self.evidence_top_k = config.get("evidence_top_k")
        self.evidence_ratio = config.get("evidence_ratio", 0.12)
        self.evidence_min_k = int(config.get("evidence_min_k", 4))
        self.evidence_max_k = config.get("evidence_max_k", 10)
        self.evidence_min_score = config.get("evidence_score_threshold", 0.55)

    def forward(
        self,
        *,
        risk_post_texts: List[str],
        risk_post_markers: List[str],
        risk_post_ids: Optional[List[str]] = None,
        block_post_texts: Optional[List[str]] = None,
        block_post_markers: Optional[List[str]] = None,
        global_segment_texts: Optional[List[List[str]]] = None,
        global_segment_markers: Optional[List[List[str]]] = None,
        pi_u: torch.Tensor,
        crisis: torch.Tensor,
        stats: torch.Tensor,
        meta_vector: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if not risk_post_texts:
            raise ValueError("risk_post_texts must not be empty")
        block_post_texts = block_post_texts or []
        block_post_markers = block_post_markers or []
        global_segment_texts = global_segment_texts or [[] for _ in range(8)]
        global_segment_markers = global_segment_markers or [[] for _ in range(len(global_segment_texts))]

        risk_post_reps = self.encoder.encode_posts(risk_post_texts, risk_post_markers)
        block_post_reps = self.encoder.encode_posts(block_post_texts, block_post_markers) if block_post_texts else None
        target_dtype = self.user_representation.attn_sd.attention.weight.dtype
        risk_post_reps = risk_post_reps.to(dtype=target_dtype)
        if block_post_reps is not None:
            block_post_reps = block_post_reps.to(dtype=target_dtype)
        pi_u = pi_u.to(dtype=target_dtype)
        crisis = crisis.to(dtype=target_dtype)
        stats = stats.to(dtype=target_dtype)
        if meta_vector is not None:
            meta_vector = meta_vector.to(dtype=target_dtype)
        segment_reps = []
        for texts, markers in zip(global_segment_texts, global_segment_markers):
            if texts:
                encoded = self.encoder.encode_posts(texts, markers)
                segment_reps.append(encoded.mean(dim=0))
            else:
                segment_reps.append(torch.zeros(self.encoder.hidden_dim, device=self.encoder.device))
        stacked_segments = torch.stack(segment_reps, dim=0).to(dtype=target_dtype)
        z_dict = self.user_representation(
            risk_post_reps=risk_post_reps,
            block_post_reps=block_post_reps,
            segment_reps=stacked_segments,
            global_stats=stats,
        )
        gate_weights = self.gate(z_dict, pi_u, crisis, stats)
        if meta_vector is None:
            zero_pad = torch.zeros(1, device=self.encoder.device, dtype=stats.dtype)
            meta_vector = torch.cat([pi_u, crisis.view(-1)[:1], stats, zero_pad], dim=0)
        expert_outputs = self.experts(
            [z_dict["z_sd"], z_dict["z_ep"], z_dict["z_sp"], z_dict["z_mix"], z_dict["z_g"]],
            meta_vector,
        )
        logit, h_u = self.moe_head(gate_weights, expert_outputs)
        evidence_scores = self.evidence_head(risk_post_reps, h_u, gate_weights)
        top_scores, top_indices = self.evidence_head.select_top_evidence(
            evidence_scores,
            top_k=int(self.evidence_top_k) if self.evidence_top_k is not None else None,
            ratio=float(self.evidence_ratio) if self.evidence_ratio is not None else None,
            min_k=self.evidence_min_k,
            max_k=int(self.evidence_max_k) if self.evidence_max_k is not None else None,
            min_score=float(self.evidence_min_score) if self.evidence_min_score is not None else None,
        )
        return {
            "logit": logit,
            "gate_weights": gate_weights,
            "evidence_scores": evidence_scores,
            "evidence_top_indices": top_indices,
            "expert_outputs": torch.stack(expert_outputs, dim=0),
            "h_u": h_u,
            "z_dict": z_dict,
            "risk_post_ids": risk_post_ids or [],
            "top_evidence_scores": top_scores,
        }
