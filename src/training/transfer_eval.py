"""Cross-dataset transfer evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.model.full_model import WPGMoEModel
from src.training.dataset import UserDataset, single_user_collate
from src.training.joint_trainer import evaluate
from src.utils.io_utils import ensure_dir, write_json


def run_transfer_eval(model_path: str, sample_path: str, config: dict, output_path: str | Path) -> Dict[str, float]:
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = WPGMoEModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    dataset = UserDataset(
        sample_path,
        is_training=False,
        max_risk_posts=config.get("max_risk_posts"),
        max_global_posts_per_segment=config.get("global_history_max_per_segment"),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=single_user_collate)
    metrics = evaluate(model, dataloader, device)
    write_json(output_path, metrics)
    return metrics
