"""Stage-C encoder pretraining."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from src.training.distributed import all_reduce_scalar, barrier, get_rank, is_distributed, is_main_process
from src.utils.schemas import META_TOKEN, make_post_marker


POST_MARKER = make_post_marker(1)


class PostScoringDataset(Dataset):
    def __init__(
        self,
        scored_posts_path: str | Path,
        *,
        allowed_user_ids: Optional[Sequence[str]] = None,
        p_meta_drop: float = 0.5,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        allowed = set(allowed_user_ids) if allowed_user_ids is not None else None
        self.rows = []
        with Path(scored_posts_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if allowed is not None and row.get("user_id") not in allowed:
                    continue
                self.rows.append(row)
        if max_samples is not None and len(self.rows) > max_samples:
            rng = random.Random(seed)
            self.rows = rng.sample(self.rows, k=int(max_samples))
        self.p_meta_drop = p_meta_drop

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        score = float(row.get("composite_evidence_score", 0.0))
        if random.random() >= self.p_meta_drop:
            meta = f"symptom_strength={score:.2f} crisis={int(row.get('crisis_level', 0))} temporality={row.get('temporality', 'unclear')}"
            text = f"{POST_MARKER} {row['text']} {META_TOKEN} {meta}"
        else:
            text = f"{POST_MARKER} {row['text']}"
        return {"text": text, "score": torch.tensor(score, dtype=torch.float32)}


class PostScoreRegressor(nn.Module):
    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.regressor = nn.Linear(encoder.hidden_dim, 1)

    def forward(self, texts: list[str], scores: torch.Tensor | None = None):
        representations = self.encoder.encode_posts(list(texts), [POST_MARKER] * len(texts))
        target_dtype = self.regressor.weight.dtype
        representations = representations.to(dtype=target_dtype)
        predictions = self.regressor(representations).squeeze(-1)
        if scores is None:
            return predictions
        target = scores.to(device=predictions.device, dtype=target_dtype)
        loss = F.mse_loss(predictions, target)
        return loss, predictions


def _evaluate_regression(encoder, regressor, dataloader) -> float:
    if dataloader is None:
        return 0.0
    device = encoder.device
    target_dtype = regressor.weight.dtype
    loss_fn = nn.MSELoss()
    total = 0.0
    steps = 0
    encoder.eval()
    regressor.eval()
    with torch.no_grad():
        for batch in dataloader:
            reps = encoder.encode_posts(list(batch["text"]), [POST_MARKER] * len(batch["text"])).to(dtype=target_dtype)
            scores = batch["score"].to(device=device, dtype=target_dtype)
            prediction = regressor(reps).squeeze(-1)
            total += float(loss_fn(prediction, scores).item())
            steps += 1
    encoder.train()
    regressor.train()
    return total / max(steps, 1)


def _evaluate_regression_deepspeed(engine, dataloader) -> float:
    if dataloader is None:
        return 0.0
    engine.eval()
    total = 0.0
    steps = 0.0
    with torch.no_grad():
        for batch in dataloader:
            predictions = engine.module(list(batch["text"]))
            scores = batch["score"].to(device=predictions.device, dtype=predictions.dtype)
            total += float(F.mse_loss(predictions, scores).item())
            steps += 1.0
    total = all_reduce_scalar(total, device=str(engine.device))
    steps = all_reduce_scalar(steps, device=str(engine.device))
    engine.train()
    return total / max(steps, 1.0)


def _make_progress(iterable, *, desc: str):
    return tqdm(iterable, desc=desc, leave=False, disable=not is_main_process())


def pretrain_encoder(encoder, train_dataset, val_dataset, config: dict) -> dict:
    if config.get("deepspeed_enabled"):
        return _pretrain_encoder_deepspeed(encoder, train_dataset, val_dataset, config)

    device = encoder.device
    regressor = nn.Linear(encoder.hidden_dim, 1).to(device)
    target_dtype = regressor.weight.dtype
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(regressor.parameters()),
        lr=float(config.get("encoder_pretrain_lr", config.get("lr", 2e-5))),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=int(config.get("encoder_pretrain_batch_size", 16)), shuffle=True)
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=int(config.get("encoder_pretrain_batch_size", 16)), shuffle=False)

    history = {"train_mse": [], "val_mse": []}
    best_state = None
    best_val = float("inf")
    for epoch in range(int(config.get("encoder_pretrain_epochs", 1))):
        running = 0.0
        steps = 0
        progress = _make_progress(train_loader, desc=f"Stage C Epoch {epoch + 1}")
        for batch in progress:
            reps = encoder.encode_posts(list(batch["text"]), [POST_MARKER] * len(batch["text"])).to(dtype=target_dtype)
            scores = batch["score"].to(device=device, dtype=target_dtype)
            prediction = regressor(reps).squeeze(-1)
            loss = loss_fn(prediction, scores)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(regressor.parameters()), 1.0)
            optimizer.step()
            running += float(loss.item())
            steps += 1
            progress.set_postfix(train_mse=f"{running / max(steps, 1):.4f}")
        train_mse = running / max(steps, 1)
        val_mse = _evaluate_regression(encoder, regressor, val_loader)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        if is_main_process():
            print(
                f"[Stage C] epoch={epoch + 1} train_mse={train_mse:.4f} val_mse={val_mse:.4f}",
                flush=True,
            )
        if val_loader is None or val_mse <= best_val:
            best_val = val_mse
            best_state = {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}
    if best_state is not None:
        encoder.load_state_dict(best_state)
    return history


def _pretrain_encoder_deepspeed(encoder, train_dataset, val_dataset, config: dict) -> dict:
    import deepspeed

    stage_model = PostScoreRegressor(encoder).to(encoder.device)
    optimizer = torch.optim.AdamW(
        list(stage_model.parameters()),
        lr=float(config.get("encoder_pretrain_lr", config.get("lr", 2e-5))),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=config.get("world_size"),
        rank=get_rank(),
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("encoder_pretrain_batch_size", 16)),
        sampler=train_sampler,
        shuffle=False,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=config.get("world_size"),
            rank=get_rank(),
            shuffle=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config.get("encoder_pretrain_batch_size", 16)),
            sampler=val_sampler,
            shuffle=False,
        )

    ds_config = dict(config["deepspeed_config_dict"])
    ds_config["train_micro_batch_size_per_gpu"] = int(config.get("encoder_pretrain_batch_size", 16))
    ds_config["gradient_accumulation_steps"] = 1
    ds_config["gradient_clipping"] = float(config.get("gradient_clipping", 1.0))
    engine, _, _, _ = deepspeed.initialize(
        model=stage_model,
        model_parameters=[param for param in stage_model.parameters() if param.requires_grad],
        optimizer=optimizer,
        config=ds_config,
    )

    history = {"train_mse": [], "val_mse": []}
    best_val = float("inf")
    best_state = None
    for epoch in range(int(config.get("encoder_pretrain_epochs", 1))):
        train_sampler.set_epoch(epoch)
        running = 0.0
        steps = 0.0
        progress = _make_progress(train_loader, desc=f"Stage C Epoch {epoch + 1}")
        for batch in progress:
            scores = batch["score"].to(device=engine.device)
            loss, _ = engine(list(batch["text"]), scores)
            engine.backward(loss)
            engine.step()
            running += float(loss.item())
            steps += 1.0
            progress.set_postfix(train_mse=f"{running / max(steps, 1.0):.4f}")
        train_mse = all_reduce_scalar(running, device=str(engine.device)) / max(
            all_reduce_scalar(steps, device=str(engine.device)),
            1.0,
        )
        val_mse = _evaluate_regression_deepspeed(engine, val_loader)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        if is_main_process():
            print(
                f"[Stage C][DeepSpeed] epoch={epoch + 1} train_mse={train_mse:.4f} val_mse={val_mse:.4f}",
                flush=True,
            )
        if is_main_process() and (val_loader is None or val_mse <= best_val):
            best_val = val_mse
            best_state = {
                key: value.detach().cpu().clone() for key, value in engine.module.encoder.state_dict().items()
            }
    if is_main_process() and best_state is not None:
        engine.module.encoder.load_state_dict(best_state)
    barrier()
    return history
