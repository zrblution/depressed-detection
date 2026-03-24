"""Expert networks for dense MoE."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class SingleExpert(nn.Module):
    def __init__(self, input_dim: int, *, hidden_dim: int = 512, output_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_state)


class ExpertGroup(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        meta_dim: int = 10,
        expert_hidden: int = 512,
        expert_output: int = 256,
        num_experts: int = 5,
    ) -> None:
        super().__init__()
        self.meta_dim = meta_dim
        self.meta_proj = nn.Linear(meta_dim, meta_dim)
        self.experts = nn.ModuleList(
            [
                SingleExpert(hidden_dim + meta_dim, hidden_dim=expert_hidden, output_dim=expert_output)
                for _ in range(num_experts)
            ]
        )

    def forward(self, z_list: List[torch.Tensor], meta_vector: torch.Tensor) -> List[torch.Tensor]:
        projected_meta = self.meta_proj(meta_vector)
        return [expert(torch.cat([z_list[idx], projected_meta], dim=0)) for idx, expert in enumerate(self.experts)]

