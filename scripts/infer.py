#!/usr/bin/env python3
"""Inference entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.raw_loader import load_user_file
from src.inference.explanation import generate_explanation
from src.inference.pipeline import InferencePipeline
from src.utils.config import load_yaml_config, resolve_path
from src.utils.io_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--screener_device", default=None)
    parser.add_argument("--screener_batch_size", type=int, default=None)
    parser.add_argument("--screener_encode_chunk_size", type=int, default=None)
    parser.add_argument("--screener_multi_gpu", action="store_true")
    parser.add_argument("--screener_target_devices", nargs="*", default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    model_path = resolve_path(args.model_path)
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    pipeline = InferencePipeline(
        model_path=str(model_path),
        model_config=config,
        screener_model=str(config.get("screener_model", "/home/tos_lx/basemodel/gte-small-zh")),
        language=str(config.get("language", "zh")),
        selection_cfg=config.get("risk_selection"),
        device=args.device,
        screener_device=args.screener_device or config.get("screener_device"),
        screener_batch_size=int(args.screener_batch_size or config.get("screener_batch_size", 256)),
        screener_encode_chunk_size=args.screener_encode_chunk_size or config.get("screener_encode_chunk_size"),
        screener_multi_gpu=bool(args.screener_multi_gpu or config.get("screener_multi_gpu", False)),
        screener_target_devices=args.screener_target_devices or config.get("screener_target_devices"),
    )
    raw_users = load_user_file(resolve_path(args.input_file), has_score=False)
    user_posts = {user["user_id"]: user["posts"] for user in raw_users}
    results = pipeline.predict_batch(user_posts) if args.batch else [pipeline.predict(uid, posts) for uid, posts in user_posts.items()]

    llm_client = None
    model_name = None
    if args.explain:
        try:
            from openai import OpenAI

            llm_client = OpenAI(api_key=config.get("explanation_api_key"))
            model_name = config.get("explanation_model")
        except Exception:
            llm_client = None
        for result in results:
            result["explanation"] = generate_explanation(result, llm_client=llm_client, model_name=model_name)
    write_json(resolve_path(args.output_file), results)
    print(json.dumps({"users": len(results), "depressed": sum(result["label"] for result in results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
