#!/usr/bin/env python3
"""Build user-level samples from stage-01 artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.processed_loader import load_grouped_scored_posts, load_risk_posts, load_standardized_users
from src.features.user_sample_builder import build_depressed_user_sample, build_template_only_user_sample
from src.utils.io_utils import ensure_dir, read_json, write_jsonl


def _build_split_rows(
    users,
    scored_posts,
    risk_posts_a,
    risk_posts_b,
    split_user_ids,
    *,
    split_name: str,
    max_gap_days: int,
    global_history_coverage: float,
    global_history_max_per_segment: int,
) -> list[dict]:
    rows = []
    for user_id in split_user_ids:
        user = users[user_id]
        label = int(user["label"])
        if label == 1 and split_name == "train":
            sample = build_depressed_user_sample(
                user_id,
                scored_posts.get(user_id, []),
                risk_posts_a.get(user_id, []),
                risk_posts_b.get(user_id, []),
                user["posts"],
                max_gap_days=max_gap_days,
                global_history_coverage=global_history_coverage,
                global_history_max_per_segment=global_history_max_per_segment,
            )
        else:
            sample = build_template_only_user_sample(
                user_id,
                label,
                risk_posts_b.get(user_id, []),
                user["posts"],
                global_history_coverage=global_history_coverage,
                global_history_max_per_segment=global_history_max_per_segment,
            )
        rows.append(sample)
    return rows


def _write_standard_splits(
    output_dir: Path,
    dataset: str,
    users,
    scored_posts,
    risk_posts_a,
    risk_posts_b,
    splits,
    *,
    max_gap_days: int,
    global_history_coverage: float,
    global_history_max_per_segment: int,
) -> None:
    for split_name in ("train", "val", "test"):
        rows = _build_split_rows(
            users,
            scored_posts,
            risk_posts_a,
            risk_posts_b,
            splits.get(split_name, []),
            split_name=split_name,
            max_gap_days=max_gap_days,
            global_history_coverage=global_history_coverage,
            global_history_max_per_segment=global_history_max_per_segment,
        )
        write_jsonl(output_dir / f"{dataset}_{split_name}.jsonl", rows)


def _write_cv_splits(
    output_dir: Path,
    users,
    scored_posts,
    risk_posts_a,
    risk_posts_b,
    folds,
    *,
    max_gap_days: int,
    global_history_coverage: float,
    global_history_max_per_segment: int,
) -> None:
    erisk_root = ensure_dir(output_dir / "erisk")
    for fold_idx, split_map in enumerate(folds):
        fold_dir = ensure_dir(erisk_root / f"fold_{fold_idx}")
        for split_name in ("train", "val", "test"):
            rows = _build_split_rows(
                users,
                scored_posts,
                risk_posts_a,
                risk_posts_b,
                split_map.get(split_name, []),
                split_name=split_name,
                max_gap_days=max_gap_days,
                global_history_coverage=global_history_coverage,
                global_history_max_per_segment=global_history_max_per_segment,
            )
            write_jsonl(fold_dir / f"{split_name}.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--standardized_file", required=True)
    parser.add_argument("--scored_file", required=True)
    parser.add_argument("--risk_a_file", required=True)
    parser.add_argument("--risk_b_file", required=True)
    parser.add_argument("--splits_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_gap_days", type=int, default=7)
    parser.add_argument("--global_history_coverage", type=float, default=0.6)
    parser.add_argument("--global_history_max_per_segment", type=int, default=128)
    args = parser.parse_args()

    users = load_standardized_users(args.standardized_file)
    scored_posts = load_grouped_scored_posts(args.scored_file)
    risk_posts_a = load_risk_posts(args.risk_a_file)
    risk_posts_b = load_risk_posts(args.risk_b_file)
    splits = read_json(args.splits_file)
    output_dir = ensure_dir(args.output_dir)

    if "folds" in splits:
        _write_cv_splits(
            output_dir,
            users,
            scored_posts,
            risk_posts_a,
            risk_posts_b,
            splits["folds"],
            max_gap_days=args.max_gap_days,
            global_history_coverage=args.global_history_coverage,
            global_history_max_per_segment=args.global_history_max_per_segment,
        )
    else:
        _write_standard_splits(
            output_dir,
            args.dataset,
            users,
            scored_posts,
            risk_posts_a,
            risk_posts_b,
            splits,
            max_gap_days=args.max_gap_days,
            global_history_coverage=args.global_history_coverage,
            global_history_max_per_segment=args.global_history_max_per_segment,
        )
    print(f"用户样本构建完成: {output_dir}")


if __name__ == "__main__":
    main()
