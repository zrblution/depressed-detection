"""Raw user loading and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sklearn.model_selection import StratifiedKFold, train_test_split

from src.utils.io_utils import iter_json_records
from src.utils.schemas import empty_symptom_vector, normalize_disease_mention_type


@dataclass
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42


def normalize_label(value: object) -> int:
    return int(str(value).strip())


def normalize_bool_like(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def normalize_gender(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "None", "null", "NULL"}:
        return None
    return text


def _normalize_clinical_context(payload: object) -> Dict[str, object]:
    if not isinstance(payload, dict):
        payload = {}
    anchor_types = payload.get("anchor_types", [])
    if not isinstance(anchor_types, list):
        anchor_types = []
    return {
        "disease_mention_type": normalize_disease_mention_type(payload.get("disease_mention_type")),
        "anchor_types": list(anchor_types),
    }


def _normalize_duration(payload: object) -> Dict[str, object]:
    if not isinstance(payload, dict):
        payload = {}
    hint_span = payload.get("hint_span_days")
    if hint_span in {"", "None", "null"}:
        hint_span = None
    return {
        "has_hint": normalize_bool_like(payload.get("has_hint", False)),
        "hint_span_days": int(hint_span) if hint_span is not None else None,
    }


def _normalize_symptom_vector(payload: object) -> Dict[str, int]:
    normalized = empty_symptom_vector()
    if not isinstance(payload, dict):
        return normalized
    for key in normalized:
        try:
            normalized[key] = int(payload.get(key, 0))
        except Exception:
            normalized[key] = 0
    return normalized


def normalize_raw_user(raw: dict, *, has_score: bool) -> dict:
    user_id = str(raw["nickname"])
    posts = []
    for idx, tweet in enumerate(raw.get("tweets", [])):
        post = {
            "post_id": f"{user_id}__{idx}",
            "user_id": user_id,
            "text": str(tweet.get("tweet_content", "")),
            "posting_time": str(tweet.get("posting_time", "")),
            "tweet_is_original": normalize_bool_like(tweet.get("tweet_is_original", "True")),
        }
        if has_score and isinstance(tweet.get("score"), dict):
            score = tweet["score"]
            post.update(
                {
                    "symptom_vector": _normalize_symptom_vector(score.get("symptom_vector")),
                    "first_person": normalize_bool_like(score.get("first_person", False)),
                    "literal_self_evidence": normalize_bool_like(score.get("literal_self_evidence", False)),
                    "confidence": float(score.get("confidence", 0.0) or 0.0),
                    "crisis_level": int(score.get("crisis_level", 0) or 0),
                    "duration": _normalize_duration(score.get("duration")),
                    "functional_impairment": int(score.get("functional_impairment", 0) or 0),
                    "clinical_context": _normalize_clinical_context(score.get("clinical_context")),
                    "temporality": str(score.get("temporality", "unclear") or "unclear"),
                }
            )
        posts.append(post)

    posts.sort(key=lambda item: item["posting_time"])
    return {
        "user_id": user_id,
        "label": normalize_label(raw.get("label", 0)),
        "gender": normalize_gender(raw.get("gender")),
        "posts": posts,
    }


def load_user_file(path: str | Path, *, has_score: bool) -> List[dict]:
    return [normalize_raw_user(record, has_score=has_score) for record in iter_json_records(path)]


def load_dataset(scored_path: str | Path, cleaned_path: str | Path) -> Tuple[List[dict], List[dict]]:
    depressed_users = load_user_file(scored_path, has_score=True)
    control_users = load_user_file(cleaned_path, has_score=False)
    return depressed_users, control_users


def _stable_train_val_test_split(uids: Sequence[str], config: SplitConfig) -> Tuple[List[str], List[str], List[str]]:
    if len(uids) < 3:
        train = list(uids[:1])
        val = list(uids[1:2])
        test = list(uids[2:])
        return train, val, test

    test_ratio = max(1.0 - config.train_ratio - config.val_ratio, 0.0)
    train, rest = train_test_split(
        list(uids),
        train_size=config.train_ratio,
        random_state=config.seed,
        shuffle=True,
    )
    if not rest:
        return list(train), [], []
    val_ratio_of_rest = config.val_ratio / max(config.val_ratio + test_ratio, 1e-8)
    if len(rest) == 1:
        return list(train), list(rest), []
    val, test = train_test_split(
        list(rest),
        train_size=val_ratio_of_rest,
        random_state=config.seed,
        shuffle=True,
    )
    return list(train), list(val), list(test)


def generate_splits(
    users: Sequence[dict],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[str]]:
    config = SplitConfig(train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    depressed = [user["user_id"] for user in users if int(user["label"]) == 1]
    control = [user["user_id"] for user in users if int(user["label"]) == 0]
    d_train, d_val, d_test = _stable_train_val_test_split(depressed, config)
    c_train, c_val, c_test = _stable_train_val_test_split(control, config)
    return {
        "train": sorted(d_train + c_train),
        "val": sorted(d_val + c_val),
        "test": sorted(d_test + c_test),
    }


def _split_fold_train_val(train_ids: Sequence[str], labels: Sequence[int], seed: int) -> Tuple[List[str], List[str]]:
    if len(train_ids) <= 1:
        return list(train_ids), []
    try:
        tr_ids, val_ids = train_test_split(
            list(train_ids),
            test_size=max(1, int(round(len(train_ids) * 0.1))),
            random_state=seed,
            shuffle=True,
            stratify=list(labels) if len(set(labels)) > 1 else None,
        )
    except ValueError:
        tr_ids, val_ids = train_test_split(list(train_ids), test_size=1, random_state=seed, shuffle=True)
    return sorted(tr_ids), sorted(val_ids)


def generate_cv_folds(users: Sequence[dict], *, n_folds: int = 5, seed: int = 42) -> List[Dict[str, List[str]]]:
    user_ids = [user["user_id"] for user in users]
    labels = [int(user["label"]) for user in users]
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: List[Dict[str, List[str]]] = []
    for fold_id, (train_index, test_index) in enumerate(splitter.split(user_ids, labels)):
        train_ids = [user_ids[idx] for idx in train_index]
        train_labels = [labels[idx] for idx in train_index]
        test_ids = [user_ids[idx] for idx in test_index]
        final_train, val_ids = _split_fold_train_val(train_ids, train_labels, seed + fold_id)
        folds.append({"train": final_train, "val": val_ids, "test": sorted(test_ids)})
    return folds
