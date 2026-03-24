#!/usr/bin/env python3
"""Train WPG-MoE."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.full_model import WPGMoEModel
from src.training.distributed import barrier, broadcast_object, is_distributed, is_main_process
from src.training.encoder_pretrain import PostScoringDataset, pretrain_encoder
from src.training.joint_trainer import train_joint
from src.training.warm_start import warm_start_experts
from src.utils.config import load_yaml_config, resolve_path
from src.utils.io_utils import ensure_dir, write_json


def _load_train_samples(train_path: Path) -> list[dict]:
    rows = []
    with train_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _collect_positive_user_ids(sample_path: Path) -> list[str]:
    user_ids = []
    for row in _load_train_samples(sample_path):
        if int(row["label"]) == 1:
            user_ids.append(row["user_id"])
    return user_ids


def _resolve_requested_device(device_value: str | None) -> str:
    if not device_value:
        return "cuda" if torch.cuda.is_available() else "cpu"
    normalized = device_value.strip().lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        if is_main_process():
            print(f"[Train] requested device '{device_value}' is unavailable; falling back to cpu", flush=True)
        return "cpu"
    return device_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip_stage_c", action="store_true")
    parser.add_argument("--skip_stage_d", action="store_true")
    parser.add_argument("--stop_after_stage_c", action="store_true")
    parser.add_argument("--train_path")
    parser.add_argument("--val_path")
    parser.add_argument("--scored_posts_path")
    parser.add_argument("--encoder_save_path")
    parser.add_argument("--device")
    parser.add_argument("--encoder_pretrain_max_train_samples", type=int)
    parser.add_argument("--encoder_pretrain_max_val_samples", type=int)
    parser.add_argument("--encoder_pretrain_p_meta_drop", type=float)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--deepspeed_config")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.train_path:
        config["train_path"] = args.train_path
    if args.val_path:
        config["val_path"] = args.val_path
    if args.scored_posts_path:
        config["scored_posts_path"] = args.scored_posts_path
    if args.encoder_save_path:
        config["encoder_save_path"] = args.encoder_save_path
    if args.device:
        config["device"] = args.device
    if args.encoder_pretrain_max_train_samples is not None:
        config["encoder_pretrain_max_train_samples"] = int(args.encoder_pretrain_max_train_samples)
    if args.encoder_pretrain_max_val_samples is not None:
        config["encoder_pretrain_max_val_samples"] = int(args.encoder_pretrain_max_val_samples)
    if args.encoder_pretrain_p_meta_drop is not None:
        config["p_meta_drop"] = float(args.encoder_pretrain_p_meta_drop)
    use_deepspeed = bool(args.deepspeed or args.local_rank >= 0 or int(os.environ.get("WORLD_SIZE", "1")) > 1)
    if use_deepspeed:
        import deepspeed

        deepspeed.init_distributed(dist_backend="nccl")
        local_rank = args.local_rank if args.local_rank >= 0 else int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        config["local_rank"] = local_rank
        config["world_size"] = int(os.environ.get("WORLD_SIZE", "1"))
        ds_config_path = args.deepspeed_config or str(ROOT / "deepspeed" / "zero2.json")
        with open(ds_config_path, "r", encoding="utf-8") as handle:
            config["deepspeed_config_dict"] = json.load(handle)
        config["deepspeed_enabled"] = True
    else:
        config["deepspeed_enabled"] = False
    train_path = resolve_path(config["train_path"])
    val_path = resolve_path(config["val_path"])
    scored_posts_path = resolve_path(config["scored_posts_path"])
    encoder_save_path = resolve_path(config["encoder_save_path"])
    warmstart_save_path = resolve_path(config["warmstart_save_path"])
    final_save_path = resolve_path(config["save_path"])
    log_path = resolve_path(config["log_path"])

    ensure_dir(encoder_save_path.parent)
    ensure_dir(warmstart_save_path.parent)
    ensure_dir(final_save_path.parent)
    ensure_dir(log_path.parent)

    if use_deepspeed:
        device = torch.device(f"cuda:{config['local_rank']}")
    else:
        resolved_device = _resolve_requested_device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
        config["device"] = resolved_device
        device = torch.device(resolved_device)
    model = WPGMoEModel(config).to(device)

    train_positive_ids = _collect_positive_user_ids(train_path)
    val_positive_ids = _collect_positive_user_ids(val_path)
    seed = int(config.get("seed", 42))
    pretrain_history: dict = {}
    if args.skip_stage_c:
        if is_main_process():
            print("[Train] skip Stage C", flush=True)
        if encoder_save_path.exists():
            state_dict = torch.load(encoder_save_path, map_location=device)
            model.encoder.load_state_dict(state_dict)
            pretrain_history = {"skipped": True, "loaded_from": str(encoder_save_path)}
        else:
            pretrain_history = {
                "skipped": True,
                "loaded_from": None,
                "warning": "encoder checkpoint not found; continuing with current initialization",
            }
    else:
        if is_main_process():
            print(
                f"[Train] Stage C start: train_positive_users={len(train_positive_ids)} val_positive_users={len(val_positive_ids)}",
                flush=True,
            )
        train_dataset = PostScoringDataset(
            scored_posts_path,
            allowed_user_ids=train_positive_ids,
            p_meta_drop=float(config.get("p_meta_drop", 0.5)),
            max_samples=config.get("encoder_pretrain_max_train_samples"),
            seed=seed,
        )
        val_dataset = PostScoringDataset(
            scored_posts_path,
            allowed_user_ids=val_positive_ids,
            p_meta_drop=float(config.get("p_meta_drop", 0.5)),
            max_samples=config.get("encoder_pretrain_max_val_samples"),
            seed=seed + 1,
        )
        pretrain_history = pretrain_encoder(model.encoder, train_dataset, val_dataset, config)
        if is_main_process():
            torch.save(model.encoder.state_dict(), encoder_save_path)
            print(f"[Train] Stage C complete: saved {encoder_save_path}", flush=True)
            write_json(
                log_path,
                {
                    "stage_c": pretrain_history,
                    "stopped_after_stage_c": bool(args.stop_after_stage_c),
                    "train_path": str(train_path),
                    "val_path": str(val_path),
                    "scored_posts_path": str(scored_posts_path),
                    "encoder_save_path": str(encoder_save_path),
                },
            )
        barrier()
        if is_distributed() and not is_main_process():
            state_dict = torch.load(encoder_save_path, map_location=device)
            model.encoder.load_state_dict(state_dict)
        barrier()

    if args.stop_after_stage_c:
        if is_main_process():
            print("[Train] stop_after_stage_c set; exiting after Stage C.", flush=True)
        return

    train_samples = _load_train_samples(train_path)
    warm_start_history: dict = {}
    if args.skip_stage_d:
        if is_main_process():
            print("[Train] skip Stage D", flush=True)
        if warmstart_save_path.exists():
            state_dict = torch.load(warmstart_save_path, map_location=device)
            model.load_state_dict(state_dict)
            warm_start_history = {"skipped": True, "loaded_from": str(warmstart_save_path)}
        else:
            warm_start_history = {
                "skipped": True,
                "loaded_from": None,
                "warning": "warm-start checkpoint not found; continuing with current initialization",
            }
    else:
        if is_main_process():
            print(f"[Train] Stage D start: train_samples={len(train_samples)}", flush=True)
        warm_start_history = warm_start_experts(model, train_samples, config)
        if is_main_process():
            torch.save(model.state_dict(), warmstart_save_path)
            print(f"[Train] Stage D complete: saved {warmstart_save_path}", flush=True)
        barrier()
        warm_start_history = broadcast_object(warm_start_history if is_main_process() else None, src=0)
        if is_distributed() and not is_main_process():
            state_dict = torch.load(warmstart_save_path, map_location=device)
            model.load_state_dict(state_dict)
        barrier()

    train_config = dict(config)
    train_config["save_path"] = str(final_save_path)
    train_config["log_path"] = str(log_path)
    if is_main_process():
        print(f"[Train] Stage E start: train={train_path} val={val_path}", flush=True)
    joint_history = train_joint(model, str(train_path), str(val_path), train_config)
    if is_main_process():
        print(f"[Train] Stage E complete: best_f1={joint_history['best_f1']:.4f}", flush=True)
        print(json.dumps({"stage_c": pretrain_history, "stage_d": warm_start_history, "stage_e": joint_history}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
