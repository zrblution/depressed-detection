"""Gate network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GateNetwork(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        num_experts: int = 5,
        gate_hidden: int = 256,
        stats_dim: int = 5,
        prior_dim: int = 4,
    ) -> None:
        super().__init__()
        input_dim = 5 * hidden_dim + stats_dim + prior_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gate_hidden, gate_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gate_hidden, num_experts),
        )

    def forward(
        self,
        z_dict: dict,
        pi_u: torch.Tensor,
        crisis: torch.Tensor,
        stats: torch.Tensor,
    ) -> torch.Tensor:
        crisis_tensor = crisis.view(-1)[:1]
        features = torch.cat(
            [
                z_dict["z_sd"],
                z_dict["z_ep"],
                z_dict["z_sp"],
                z_dict["z_mix"],
                z_dict["z_g"],
                pi_u,
                crisis_tensor,
                stats,
            ],
            dim=0,
        )
        return F.softmax(self.network(features), dim=0)

