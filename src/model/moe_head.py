"""Dense MoE fusion head."""

from __future__ import annotations

import torch
import torch.nn as nn


class MoEHead(nn.Module):
    def __init__(self, *, expert_output_dim: int = 256) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(expert_output_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, gate_weights: torch.Tensor, expert_outputs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(expert_outputs, dim=0)
        h_u = (gate_weights.unsqueeze(-1) * stacked).sum(dim=0)
        logit = self.classifier(h_u)
        return logit, h_u

