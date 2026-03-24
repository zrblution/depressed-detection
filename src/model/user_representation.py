"""User-level representation builders."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 2:
            raise ValueError("AttentionPooling expects [N, D]")
        scores = self.attention(hidden_states).squeeze(-1)
        weights = F.softmax(scores, dim=0)
        return (weights.unsqueeze(-1) * hidden_states).sum(dim=0)


class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, segment_representations: torch.Tensor) -> torch.Tensor:
        q_states = self.q_proj(segment_representations)
        k_states = self.k_proj(segment_representations)
        v_states = self.v_proj(segment_representations)
        weights = torch.softmax(torch.matmul(q_states, k_states.transpose(0, 1)) / self.scale, dim=-1)
        return torch.matmul(weights, v_states).mean(dim=0)


class UserRepresentationModule(nn.Module):
    def __init__(self, hidden_dim: int, *, stats_dim: int = 5) -> None:
        super().__init__()
        self.attn_sd = AttentionPooling(hidden_dim)
        self.attn_ep = AttentionPooling(hidden_dim)
        self.attn_sp = AttentionPooling(hidden_dim)
        self.temporal_attn = TemporalSelfAttention(hidden_dim)
        self.stats_proj = nn.Linear(stats_dim, hidden_dim)

    def forward(
        self,
        risk_post_reps: torch.Tensor,
        block_post_reps: torch.Tensor | None,
        segment_reps: torch.Tensor,
        global_stats: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_sd = self.attn_sd(risk_post_reps)
        z_ep = self.attn_ep(block_post_reps) if block_post_reps is not None and block_post_reps.numel() else self.attn_ep(risk_post_reps)
        z_sp = self.attn_sp(risk_post_reps[: min(3, risk_post_reps.shape[0])])
        z_mix = risk_post_reps.mean(dim=0)
        z_g = self.temporal_attn(segment_reps) + self.stats_proj(global_stats)
        return {"z_sd": z_sd, "z_ep": z_ep, "z_sp": z_sp, "z_mix": z_mix, "z_g": z_g}

