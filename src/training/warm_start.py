"""Stage-D expert warm start."""

from __future__ import annotations

import random
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from src.training.dataset import format_user_sample
from src.training.distributed import (
    all_reduce_scalar,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
)


class _WarmStartDataset(Dataset):
    def __init__(self, samples: List[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def _single_item_collate(batch: List[dict]) -> dict:
    return batch[0]


def _sync_gradients(parameters: List[torch.nn.Parameter]) -> None:
    if not is_distributed():
        return
    world_size = float(get_world_size())
    for parameter in parameters:
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def _expert_subsets(train_samples: List[dict]) -> Dict[int, List[dict]]:
    depressed = [sample for sample in train_samples if int(sample["label"]) == 1]
    top_ratio = 0.30
    top_count = max(int(len(depressed) * top_ratio), 1) if depressed else 0
    return {
        0: sorted(depressed, key=lambda item: item["priors"]["self_disclosure"], reverse=True)[:top_count],
        1: sorted(depressed, key=lambda item: item["priors"]["episode_supported"], reverse=True)[:top_count],
        2: sorted(depressed, key=lambda item: item["priors"]["sparse_evidence"], reverse=True)[:top_count],
        3: depressed,
        4: train_samples,
    }


def warm_start_experts(model, train_samples: List[dict], config: dict) -> Dict[str, List[dict]]:
    device = next(model.parameters()).device
    base_model = model
    base_model.train()
    history: Dict[str, List[dict]] = {"experts": []}
    subsets = _expert_subsets(train_samples)
    max_samples_per_expert = config.get("warm_start_max_samples_per_expert")
    seed = int(config.get("seed", 42))
    classifier = base_model.moe_head.classifier
    criterion = torch.nn.BCEWithLogitsLoss()
    epochs = int(config.get("warm_start_epochs", 3))

    for expert_idx, subset in subsets.items():
        if max_samples_per_expert is not None and len(subset) > int(max_samples_per_expert):
            rng = random.Random(seed + expert_idx)
            subset = rng.sample(subset, k=int(max_samples_per_expert))
        if not subset:
            history["experts"].append({"expert_idx": expert_idx, "steps": 0, "loss": 0.0})
            continue
        if is_main_process():
            print(
                f"[Stage D] expert={expert_idx} subset_size={len(subset)} epochs={epochs}",
                flush=True,
            )
        for param in base_model.parameters():
            param.requires_grad = False
        for param in base_model.experts.experts[expert_idx].parameters():
            param.requires_grad = True
        for param in classifier.parameters():
            param.requires_grad = True
        trainable_params = list(base_model.experts.experts[expert_idx].parameters()) + list(classifier.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(config.get("warm_start_lr", 1e-4)),
        )
        running_loss = 0.0
        steps = 0
        dataset = _WarmStartDataset(subset)
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=False,
            )
            if is_distributed()
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=sampler,
            collate_fn=_single_item_collate,
        )
        for epoch in range(epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            progress = tqdm(
                loader,
                desc=f"Stage D Expert {expert_idx} Epoch {epoch + 1}",
                leave=False,
                disable=not is_main_process(),
            )
            for raw_sample in progress:
                sample = format_user_sample(
                    raw_sample,
                    is_training=False,
                    max_risk_posts=config.get("max_risk_posts"),
                    max_global_posts_per_segment=config.get("global_history_max_per_segment"),
                    force_risk_source="llm",
                )
                output = base_model(
                    risk_post_texts=sample["risk_texts"],
                    risk_post_markers=sample["risk_markers"],
                    risk_post_ids=sample["risk_post_ids"],
                    block_post_texts=sample["block_texts"],
                    block_post_markers=sample["block_markers"],
                    global_segment_texts=sample["global_segment_texts"],
                    global_segment_markers=sample["global_segment_markers"],
                    pi_u=sample["pi_u"].to(device),
                    crisis=sample["crisis"].to(device),
                    stats=sample["stats"].to(device),
                    meta_vector=sample["meta_vector"].to(device),
                )
                expert_output = output["expert_outputs"][expert_idx]
                forced_logit = classifier(expert_output)
                label = sample["label"].to(device)
                loss = criterion(forced_logit.view(-1), label.view(-1))
                optimizer.zero_grad()
                loss.backward()
                _sync_gradients(trainable_params)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                running_loss += float(loss.item())
                steps += 1
                progress.set_postfix(loss=f"{running_loss / max(steps, 1):.4f}")
            logged_loss = running_loss
            logged_steps = float(steps)
            if is_distributed():
                logged_loss = all_reduce_scalar(running_loss, device=str(device))
                logged_steps = all_reduce_scalar(float(steps), device=str(device))
            if is_main_process():
                print(
                    f"[Stage D] expert={expert_idx} epoch={epoch + 1} avg_loss={logged_loss / max(logged_steps, 1.0):.4f}",
                    flush=True,
                )
        history["experts"].append(
            {
                "expert_idx": expert_idx,
                "subset_size": len(subset),
                "steps": int(logged_steps) if is_distributed() else steps,
                "loss": logged_loss / max(logged_steps, 1.0),
            }
        )
    for param in base_model.parameters():
        param.requires_grad = True
    return history
