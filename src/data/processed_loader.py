"""Load processed artifacts emitted by the data pipeline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List

from src.utils.io_utils import read_json, read_jsonl


def load_standardized_users(path: str | Path) -> Dict[str, dict]:
    return {row["user_id"]: row for row in read_jsonl(path)}


def load_grouped_scored_posts(path: str | Path) -> Dict[str, List[dict]]:
    grouped: DefaultDict[str, List[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        grouped[row["user_id"]].append(row)
    for posts in grouped.values():
        posts.sort(key=lambda item: item.get("composite_evidence_score", 0.0), reverse=True)
    return dict(grouped)


def load_risk_posts(path: str | Path) -> Dict[str, List[dict]]:
    payload = read_json(path)
    return {user_id: list(posts) for user_id, posts in payload.items()}

