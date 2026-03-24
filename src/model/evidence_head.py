"""Evidence selection head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class EvidenceHead(nn.Module):
    def __init__(self, post_dim: int, *, user_dim: int = 256, gate_dim: int = 5, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(post_dim + user_dim + gate_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, post_reps: torch.Tensor, h_u: torch.Tensor, gate_weights: torch.Tensor) -> torch.Tensor:
        count = post_reps.shape[0]
        user_states = h_u.unsqueeze(0).expand(count, -1)
        gate_states = gate_weights.unsqueeze(0).expand(count, -1)
        combined = torch.cat([post_reps, user_states, gate_states], dim=-1)
        return torch.sigmoid(self.network(combined).squeeze(-1))

    def select_top_evidence(
        self,
        scores: torch.Tensor,
        *,
        top_k: int | None = None,
        ratio: float | None = None,
        min_k: int = 1,
        max_k: int | None = None,
        min_score: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = int(scores.shape[0])
        if count == 0:
            empty = scores.new_empty((0,))
            return empty, torch.empty((0,), dtype=torch.long, device=scores.device)

        if ratio is not None:
            if ratio <= 0:
                raise ValueError("ratio must be > 0 when provided")
            target_k = math.ceil(count * ratio)
        elif top_k is not None:
            target_k = int(top_k)
        else:
            target_k = int(min_k)

        target_k = max(int(min_k), target_k)
        if max_k is not None:
            target_k = min(target_k, int(max_k))
        target_k = min(target_k, count)

        ranked_scores, ranked_indices = torch.sort(scores, descending=True)
        if min_score is not None:
            keep_mask = ranked_scores >= float(min_score)
            filtered_scores = ranked_scores[keep_mask]
            filtered_indices = ranked_indices[keep_mask]
            if filtered_scores.numel() >= int(min_k):
                return filtered_scores[: min(target_k, filtered_scores.numel())], filtered_indices[
                    : min(target_k, filtered_indices.numel())
                ]
        return ranked_scores[:target_k], ranked_indices[:target_k]
