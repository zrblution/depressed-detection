"""Loss functions for WPG-MoE training."""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1), pos_weight=self.pos_weight.to(logits.device))


class RoutingLoss(nn.Module):
    def __init__(self, *, min_confidence: float = 0.6, min_gap: float = 0.1) -> None:
        super().__init__()
        self.min_confidence = min_confidence
        self.min_gap = min_gap

    def forward(self, gate_weights: torch.Tensor, pi_u: torch.Tensor) -> torch.Tensor:
        sorted_values = torch.sort(pi_u, descending=True).values
        top_gap = sorted_values[0] - sorted_values[1] if sorted_values.numel() > 1 else sorted_values[0]
        if float(sorted_values[0]) < self.min_confidence or float(top_gap) < self.min_gap:
            return torch.zeros((), device=gate_weights.device)
        gate_prior = gate_weights[:3]
        gate_prior = gate_prior / (gate_prior.sum() + 1e-8)
        target = pi_u / (pi_u.sum() + 1e-8)
        return F.kl_div(torch.log(gate_prior + 1e-8), target, reduction="sum")


class EvidenceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, evidence_scores: torch.Tensor, evidence_targets: torch.Tensor, *, is_depressed: bool) -> torch.Tensor:
        if not is_depressed or evidence_targets.numel() == 0:
            return torch.zeros((), device=evidence_scores.device)
        silver_labels = evidence_targets.to(device=evidence_scores.device, dtype=evidence_scores.dtype).clamp_(0.0, 1.0)
        return self.loss(evidence_scores, silver_labels)


class BalanceLoss(nn.Module):
    def __init__(self, *, num_experts: int = 5, tau: float = 0.1) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.tau = tau

    def forward(self, gate_weights_batch: torch.Tensor) -> torch.Tensor:
        importance = gate_weights_batch.mean(dim=0)
        l_importance = self.num_experts * (importance**2).sum()
        sharpened = F.softmax(gate_weights_batch / self.tau, dim=-1)
        load = sharpened.mean(dim=0)
        l_load = self.num_experts * (importance * load).sum()
        return l_importance + l_load


class EntropyLoss(nn.Module):
    def __init__(self, *, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, gate_weights_batch: torch.Tensor) -> torch.Tensor:
        entropy = -(gate_weights_batch * torch.log(gate_weights_batch + self.eps)).sum(dim=-1)
        return -entropy.mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        *,
        alpha: float = 0.3,
        beta: float = 0.2,
        gamma: float = 0.15,
        delta_init: float = 0.1,
        delta_min: float = 0.02,
        pos_weight: float = 1.0,
        num_experts: int = 5,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_init = delta_init
        self.delta_min = delta_min
        self.cls_loss = ClassificationLoss(pos_weight=pos_weight)
        self.route_loss = RoutingLoss()
        self.evidence_loss = EvidenceLoss()
        self.balance_loss = BalanceLoss(num_experts=num_experts)
        self.entropy_loss = EntropyLoss()

    def get_delta(self, current_epoch: int, total_epochs: int) -> float:
        progress = current_epoch / max(total_epochs, 1)
        return self.delta_min + 0.5 * (self.delta_init - self.delta_min) * (1 + math.cos(math.pi * progress))

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        gate_weights_batch: torch.Tensor,
        pi_u_batch: torch.Tensor,
        evidence_scores_batch: List[torch.Tensor],
        evidence_targets_batch: List[torch.Tensor],
        is_depressed_batch: List[bool],
        *,
        current_epoch: int,
        total_epochs: int,
    ) -> Dict[str, torch.Tensor | float]:
        l_cls = self.cls_loss(logits, labels)
        l_route = torch.zeros((), device=logits.device)
        route_count = 0
        for gate_weights, pi_u in zip(gate_weights_batch, pi_u_batch):
            route_loss = self.route_loss(gate_weights, pi_u)
            if route_loss.item() > 0:
                l_route = l_route + route_loss
                route_count += 1
        if route_count:
            l_route = l_route / route_count

        l_evidence = torch.zeros((), device=logits.device)
        evidence_count = 0
        for evidence_scores, evidence_targets, is_depressed in zip(
            evidence_scores_batch, evidence_targets_batch, is_depressed_batch
        ):
            evidence_loss = self.evidence_loss(evidence_scores, evidence_targets, is_depressed=is_depressed)
            if evidence_loss.item() > 0:
                l_evidence = l_evidence + evidence_loss
                evidence_count += 1
        if evidence_count:
            l_evidence = l_evidence / evidence_count

        l_balance = self.balance_loss(gate_weights_batch)
        l_entropy = self.entropy_loss(gate_weights_batch)
        delta = self.get_delta(current_epoch, total_epochs)
        total = l_cls + self.alpha * l_route + self.beta * l_evidence + self.gamma * l_balance + delta * l_entropy
        return {
            "total": total,
            "cls": l_cls.detach(),
            "route": l_route.detach(),
            "evidence": l_evidence.detach(),
            "balance": l_balance.detach(),
            "entropy": l_entropy.detach(),
            "delta": delta,
        }
