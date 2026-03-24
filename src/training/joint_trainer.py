"""Stage-E joint training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from src.training.dataset import UserDataset, single_user_collate
from src.training.distributed import (
    all_reduce_scalar,
    all_gather_objects,
    barrier,
    broadcast_object,
    flatten_gathered,
    is_distributed,
    is_main_process,
)
from src.training.losses import CombinedLoss
from src.utils.io_utils import ensure_dir, write_json


def _forward_single_sample(model, sample: dict, device: torch.device) -> Dict[str, torch.Tensor]:
    return model(
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


def evaluate(model, dataloader, device: torch.device) -> Dict[str, float]:
    active_model = model.module if hasattr(model, "module") else model
    active_model.eval()
    probs: List[float] = []
    preds: List[int] = []
    golds: List[int] = []
    gate_weights: List[List[float]] = []
    with torch.no_grad():
        for sample in dataloader:
            output = _forward_single_sample(active_model, sample, device)
            prob = float(torch.sigmoid(output["logit"]).item())
            pred = 1 if prob > 0.5 else 0
            gold = int(sample["label"].item())
            probs.append(prob)
            preds.append(pred)
            golds.append(gold)
            gate_weights.append(output["gate_weights"].detach().cpu().tolist())
    if is_distributed():
        gathered = all_gather_objects(
            {
                "probs": probs,
                "preds": preds,
                "golds": golds,
                "gate_weights": gate_weights,
            }
        )
        if is_main_process():
            probs = flatten_gathered([item["probs"] for item in gathered])
            preds = flatten_gathered([item["preds"] for item in gathered])
            golds = flatten_gathered([item["golds"] for item in gathered])
            gate_weights = flatten_gathered([item["gate_weights"] for item in gathered])
        else:
            probs = []
            preds = []
            golds = []
            gate_weights = []
    if not golds:
        metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0, "avg_gate_weights": [0.0] * 5}
    elif is_main_process() or not is_distributed():
        precision, recall, f1, _ = precision_recall_fscore_support(golds, preds, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(golds, probs)
        except ValueError:
            auc = 0.0
        avg_gate = [sum(weights[idx] for weights in gate_weights) / max(len(gate_weights), 1) for idx in range(5)]
        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
            "avg_gate_weights": avg_gate,
        }
    else:
        metrics = None
    return broadcast_object(metrics, src=0)


def train_joint(model, train_path: str, val_path: str, config: dict) -> Dict[str, object]:
    device = next(model.parameters()).device
    train_dataset = UserDataset(
        train_path,
        is_training=True,
        p_risk_swap=float(config.get("p_risk_swap", 0.5)),
        p_meta_drop=float(config.get("p_meta_drop", 0.5)),
        p_block_drop=float(config.get("p_block_drop", 0.4)),
        p_prior_drop=float(config.get("p_prior_drop", 0.3)),
        p_post_drop=float(config.get("p_post_drop", 0.3)),
        max_risk_posts=config.get("max_risk_posts"),
        max_global_posts_per_segment=config.get("global_history_max_per_segment"),
        max_samples=config.get("joint_max_train_samples"),
        seed=int(config.get("seed", 42)),
    )
    val_dataset = UserDataset(
        val_path,
        is_training=False,
        max_risk_posts=config.get("max_risk_posts"),
        max_global_posts_per_segment=config.get("global_history_max_per_segment"),
        max_samples=config.get("joint_max_val_samples"),
        seed=int(config.get("seed", 42)) + 1,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed() else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=single_user_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=single_user_collate,
    )

    encoder_params = []
    non_encoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            non_encoder_params.append(param)
    param_groups = []
    if non_encoder_params:
        param_groups.append({"params": non_encoder_params, "lr": float(config.get("lr_head", 1e-4))})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": float(config.get("lr_encoder", 2e-5))})
    if not param_groups:
        raise ValueError("No trainable parameters found for joint training")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(config.get("weight_decay", 0.01)))
    use_deepspeed = bool(config.get("deepspeed_enabled"))
    active_model = model
    active_optimizer = optimizer
    if use_deepspeed:
        import deepspeed

        ds_config = dict(config["deepspeed_config_dict"])
        ds_config["train_micro_batch_size_per_gpu"] = 1
        ds_config["gradient_accumulation_steps"] = 1
        ds_config["gradient_clipping"] = float(config.get("gradient_clipping", 1.0))
        active_model, active_optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            model_parameters=[param for param in model.parameters() if param.requires_grad],
            config=ds_config,
        )
    criterion = CombinedLoss(
        alpha=float(config.get("alpha", 0.3)),
        beta=float(config.get("beta", 0.2)),
        gamma=float(config.get("gamma", 0.15)),
        delta_init=float(config.get("delta_init", 0.1)),
        delta_min=float(config.get("delta_min", 0.02)),
        pos_weight=float(config.get("pos_weight", 1.0)),
        num_experts=int(config.get("num_experts", 5)),
    )
    max_epochs = int(config.get("max_epochs", 1))
    accumulation_size = int(config.get("batch_size", 16))
    freeze_encoder_epochs = int(config.get("freeze_encoder_epochs", 0))
    patience = int(config.get("patience", 5))
    save_path = Path(config["save_path"])
    log_path = Path(config["log_path"])
    ensure_dir(save_path.parent)
    ensure_dir(log_path.parent)

    training_log: Dict[str, List[dict]] = {"epochs": []}
    best_f1 = -1.0
    wait = 0
    encoder_trainable_flags = {name: param.requires_grad for name, param in model.encoder.named_parameters()}

    def _set_encoder_trainable(trainable: bool) -> None:
        for name, param in model.encoder.named_parameters():
            param.requires_grad = encoder_trainable_flags[name] if trainable else False

    for epoch in range(max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        _set_encoder_trainable(epoch >= freeze_encoder_epochs)
        active_model.train()
        if not use_deepspeed:
            active_optimizer.zero_grad()
        accum_logits: List[torch.Tensor] = []
        accum_labels: List[torch.Tensor] = []
        accum_gates: List[torch.Tensor] = []
        accum_priors: List[torch.Tensor] = []
        accum_evidence_scores: List[torch.Tensor] = []
        accum_evidence_targets: List[torch.Tensor] = []
        accum_is_depressed: List[bool] = []
        epoch_total = 0.0
        optimizer_steps = 0
        progress = tqdm(
            train_loader,
            desc=f"Stage E Epoch {epoch + 1}",
            leave=False,
            disable=not is_main_process(),
        )

        for step, sample in enumerate(progress, start=1):
            forward_model = active_model.module if hasattr(active_model, "module") else active_model
            output = _forward_single_sample(forward_model, sample, device)
            accum_logits.append(output["logit"].view(1))
            accum_labels.append(sample["label"].to(device))
            accum_gates.append(output["gate_weights"])
            accum_priors.append(sample["pi_u"].to(device))
            accum_evidence_scores.append(output["evidence_scores"])
            accum_evidence_targets.append(sample["evidence_target_scores"].to(device))
            accum_is_depressed.append(bool(sample["is_depressed"]))

            if len(accum_logits) == accumulation_size or step == len(train_loader):
                current_batch = len(accum_logits)
                loss_dict = criterion(
                    torch.cat(accum_logits, dim=0),
                    torch.cat(accum_labels, dim=0),
                    torch.stack(accum_gates, dim=0),
                    torch.stack(accum_priors, dim=0),
                    accum_evidence_scores,
                    accum_evidence_targets,
                    accum_is_depressed,
                    current_epoch=epoch,
                    total_epochs=max_epochs,
                )
                loss = loss_dict["total"] / current_batch
                if use_deepspeed:
                    active_model.backward(loss)
                    active_model.step()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    active_optimizer.step()
                    active_optimizer.zero_grad()
                epoch_total += float(loss_dict["total"].item())
                optimizer_steps += 1
                if is_main_process():
                    progress.set_postfix(
                        train_loss=f"{epoch_total / max(optimizer_steps, 1):.4f}",
                        cls=f"{float(loss_dict['cls'].item()):.4f}",
                    )
                accum_logits = []
                accum_labels = []
                accum_gates = []
                accum_priors = []
                accum_evidence_scores = []
                accum_evidence_targets = []
                accum_is_depressed = []

        logged_epoch_total = epoch_total
        logged_optimizer_steps = float(optimizer_steps)
        if is_distributed():
            logged_epoch_total = all_reduce_scalar(epoch_total, device=str(device))
            logged_optimizer_steps = all_reduce_scalar(float(optimizer_steps), device=str(device))
        metrics = evaluate(active_model, val_loader, device)
        epoch_record = {
            "epoch": epoch + 1,
            "train_total_loss": logged_epoch_total / max(logged_optimizer_steps, 1.0),
            "val_f1": metrics["f1"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_auc": metrics["auc"],
            "avg_gate_weights": metrics["avg_gate_weights"],
            "delta": criterion.get_delta(epoch, max_epochs),
            "encoder_frozen": bool(epoch < freeze_encoder_epochs),
        }
        if is_main_process():
            training_log["epochs"].append(epoch_record)
            print(
                "[Stage E] "
                f"epoch={epoch + 1} "
                f"train_total_loss={epoch_record['train_total_loss']:.4f} "
                f"val_f1={epoch_record['val_f1']:.4f} "
                f"val_precision={epoch_record['val_precision']:.4f} "
                f"val_recall={epoch_record['val_recall']:.4f} "
                f"val_auc={epoch_record['val_auc']:.4f} "
                f"encoder_frozen={epoch_record['encoder_frozen']}",
                flush=True,
            )
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                wait = 0
                model_to_save = active_model.module if hasattr(active_model, "module") else active_model
                torch.save(model_to_save.state_dict(), save_path)
            else:
                wait += 1
        best_f1 = float(broadcast_object(best_f1, src=0))
        wait = int(broadcast_object(wait, src=0))
        stop_training = wait >= patience
        stop_training = bool(broadcast_object(stop_training, src=0))
        if stop_training:
            break

    if is_main_process():
        write_json(log_path, training_log)
    barrier()
    return {"best_f1": best_f1, "log": training_log}
