#!/usr/bin/env python3
"""Run stage-01 data processing and template screening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.composite_scorer import add_composite_scores, get_risk_posts_a
from src.data.raw_loader import generate_cv_folds, generate_splits, load_dataset
from src.data.template_screener import PHQ9TemplateScreener
from src.utils.io_utils import ensure_dir, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--scored_path", required=True)
    parser.add_argument("--cleaned_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--encoder_model", default="/home/tos_lx/basemodel/gte-small-zh")
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    parser.add_argument("--cv_folds", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--encode_chunk_size", type=int, default=4096)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--target_devices", nargs="*", default=None)
    args = parser.parse_args()

    depressed_users, control_users = load_dataset(args.scored_path, args.cleaned_path)
    all_users = depressed_users + control_users
    output_dir = ensure_dir(args.output_dir)
    if args.cv_folds > 0:
        splits = {"folds": generate_cv_folds(all_users, n_folds=args.cv_folds)}
    else:
        splits = generate_splits(all_users)
    write_json(output_dir / "splits.json", splits)
    write_jsonl(output_dir / "all_users_standardized.jsonl", all_users)

    scored_posts_rows = []
    risk_posts_a = {}
    for user in depressed_users:
        scored_posts = add_composite_scores(user["posts"])
        for post in scored_posts:
            scored_posts_rows.append(post)
        risk_posts_a[user["user_id"]] = [
            {
                "post_id": post["post_id"],
                "text": post["text"],
                "composite_evidence_score": post["composite_evidence_score"],
                "crisis_level": post.get("crisis_level", 0),
                "temporality": post.get("temporality", "unclear"),
            }
            for post in get_risk_posts_a(scored_posts)
        ]
    write_jsonl(output_dir / "depressed_scored_posts.jsonl", scored_posts_rows)
    write_json(output_dir / "risk_posts_a.json", risk_posts_a)

    all_user_posts = {user["user_id"]: user["posts"] for user in all_users}
    with PHQ9TemplateScreener(
        model_name=args.encoder_model,
        language=args.language,
        device=args.device,
        batch_size=args.batch_size,
        encode_chunk_size=args.encode_chunk_size,
        multi_gpu=args.multi_gpu,
        target_devices=args.target_devices,
    ) as screener:
        risk_posts_b = screener.screen_all_users(all_user_posts)
    write_json(output_dir / "risk_posts_b.json", risk_posts_b)
    print(f"完成 {args.dataset_name}: 输出目录 {output_dir}")


if __name__ == "__main__":
    main()
