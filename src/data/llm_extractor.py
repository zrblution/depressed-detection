"""Reference-only wrapper for the offline LLM scoring stage."""

from __future__ import annotations

from pathlib import Path


REFERENCE_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "Data Processing and User Aggregation"
    / "03-post-scoring"
    / "score_all_vllm_local.py"
)


def get_reference_command() -> str:
    return f"python '{REFERENCE_SCRIPT}' --help"


def run_llm_extractor(*args, **kwargs) -> None:
    raise RuntimeError(
        "LLM post scoring is not part of the online pipeline. "
        f"Use the offline reference script instead: {get_reference_command()}"
    )
