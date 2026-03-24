"""File IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, List


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)
    return target


def iter_json_records(path: str | Path) -> Iterator[Any]:
    """Yield records from JSONL or a JSON array file."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"Expected JSON array in {source}")
            for item in payload:
                yield item
            return

        for line in handle:
            text = line.strip()
            if not text:
                continue
            yield json.loads(text)


def read_jsonl(path: str | Path) -> List[Any]:
    return list(iter_json_records(path))


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target

