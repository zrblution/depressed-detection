#!/usr/bin/env python3
"""Run cross-dataset transfer evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.transfer_eval import run_transfer_eval
from src.utils.config import load_yaml_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--sample_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    metrics = run_transfer_eval(
        model_path=str(resolve_path(args.model_path)),
        sample_path=str(resolve_path(args.sample_path)),
        config=config,
        output_path=resolve_path(args.output_path),
    )
    print(metrics)


if __name__ == "__main__":
    main()

