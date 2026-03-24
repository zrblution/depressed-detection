"""PHQ-9 template screening (path B)."""

from __future__ import annotations

import heapq
from collections import defaultdict
from itertools import count
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from src.data.composite_scorer import compute_dynamic_k


PHQ9_TEMPLATES_ZH = {
    "depressed_mood": [
        "我感到心情低落、沮丧、绝望",
        "我每天都很难过，觉得生活没有意义",
        "心里特别压抑，什么都不想做",
    ],
    "anhedonia": [
        "我对什么事情都提不起兴趣",
        "以前喜欢的事情现在完全不想做了",
        "对任何事都没有快乐的感觉",
    ],
    "sleep": [
        "我失眠了很久都睡不着",
        "每天晚上翻来覆去睡不着觉",
        "睡眠质量很差经常半夜醒来",
    ],
    "fatigue": [
        "我整天都觉得很累没有力气",
        "做什么事都提不起精神来",
        "感觉身体被掏空了一样疲惫",
    ],
    "appetite_or_weight": [
        "我最近完全没有胃口吃不下东西",
        "体重变化很大不是暴食就是不吃",
        "食欲很差看到食物就想吐",
    ],
    "worthlessness_or_guilt": [
        "我觉得自己一无是处是个废物",
        "总觉得自己是个失败者对不起所有人",
        "强烈的自责感觉自己什么都做不好",
    ],
    "concentration": [
        "我无法集中注意力做任何事情",
        "脑子里一片空白什么都想不起来",
        "注意力完全无法集中",
    ],
    "psychomotor": [
        "我变得很迟钝反应也慢了",
        "说话和做事都变得非常缓慢",
        "坐立不安总是很烦躁",
    ],
    "suicidal_ideation": [
        "我不想活了觉得死了算了",
        "反复想到死亡或者伤害自己",
        "活着太痛苦了不如死了好",
    ],
}

PHQ9_TEMPLATES_EN = {
    "depressed_mood": [
        "Feeling down, depressed, or hopeless",
        "I feel so sad and empty every day",
        "Life feels meaningless and I cannot stop feeling depressed",
    ],
    "anhedonia": [
        "Little interest or pleasure in doing things",
        "I do not enjoy anything anymore",
        "Lost all motivation and interest in activities I used to love",
    ],
    "sleep": [
        "Trouble falling asleep, staying asleep, or sleeping too much",
        "I cannot sleep at night",
        "Waking up in the middle of the night and cannot go back to sleep",
    ],
    "fatigue": [
        "Feeling tired or having little energy",
        "I am exhausted all the time",
        "So drained I can barely get out of bed",
    ],
    "appetite_or_weight": [
        "Poor appetite or overeating",
        "I have lost my appetite completely",
        "Weight changing drastically because I cannot regulate eating",
    ],
    "worthlessness_or_guilt": [
        "Feeling bad about yourself or that you are a failure",
        "I am worthless and everything is my fault",
        "Overwhelming guilt and self-hatred",
    ],
    "concentration": [
        "Trouble concentrating on things",
        "Cannot focus on anything",
        "Unable to concentrate or make simple decisions",
    ],
    "psychomotor": [
        "Moving or speaking slowly, or being fidgety and restless",
        "I have become very slow",
        "Cannot sit still and constantly feel agitated",
    ],
    "suicidal_ideation": [
        "Thoughts that you would be better off dead or hurting yourself",
        "I do not want to live anymore",
        "Recurring thoughts of death and self-harm",
    ],
}


class PHQ9TemplateScreener:
    """Template-based risk screener."""

    def __init__(
        self,
        model_name: str = "/home/tos_lx/basemodel/gte-small-zh",
        *,
        language: str = "zh",
        selection_cfg: dict | None = None,
        device: str | None = None,
        batch_size: int = 256,
        encode_chunk_size: int | None = None,
        multi_gpu: bool = False,
        target_devices: Sequence[str] | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "sentence-transformers is required for strict document mode. "
                "Install dependencies from code/requirements.txt."
            ) from exc

        self.selection_cfg = selection_cfg or {"mode": "dynamic_k"}
        self.device = self._resolve_device(device)
        self.batch_size = max(int(batch_size), 1)
        default_chunk_size = max(self.batch_size * 16, 4096)
        self.encode_chunk_size = max(int(encode_chunk_size or default_chunk_size), self.batch_size)
        self.target_devices = self._resolve_target_devices(target_devices, multi_gpu)
        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.pool = None
        if self.target_devices:
            self.pool = self.encoder.start_multi_process_pool(target_devices=list(self.target_devices))
        self.templates = PHQ9_TEMPLATES_ZH if language == "zh" else PHQ9_TEMPLATES_EN
        self.template_dimensions = list(self.templates)
        template_texts: list[str] = []
        self.template_slices: dict[str, tuple[int, int]] = {}
        start = 0
        for dim, texts in self.templates.items():
            template_texts.extend(texts)
            end = start + len(texts)
            self.template_slices[dim] = (start, end)
            start = end
        self.template_embeddings = self._encode_texts(template_texts, use_pool=False)

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if device is None or device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and torch.cuda.is_available():
            return "cuda:0"
        return device

    @staticmethod
    def _resolve_target_devices(target_devices: Sequence[str] | None, multi_gpu: bool) -> list[str]:
        if target_devices:
            return [str(device) for device in target_devices]
        if not multi_gpu:
            return []
        if not torch.cuda.is_available():
            return []
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]

    def close(self) -> None:
        if self.pool is not None:
            self.encoder.stop_multi_process_pool(self.pool)
            self.pool = None

    def __enter__(self) -> "PHQ9TemplateScreener":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _encode_texts(self, texts: Sequence[str], *, use_pool: bool | None = None) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        should_use_pool = self.pool is not None if use_pool is None else use_pool
        if should_use_pool and self.pool is not None:
            return self.encoder.encode_multi_process(
                list(texts),
                pool=self.pool,
                batch_size=self.batch_size,
                chunk_size=self.encode_chunk_size,
                normalize_embeddings=True,
            ).astype(np.float32)
        return self.encoder.encode(
            list(texts),
            batch_size=self.batch_size,
            device=self.device,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def _compute_scores(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if embeddings.size == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0, len(self.template_dimensions)), dtype=np.float32)
        similarity_matrix = embeddings @ self.template_embeddings.T
        dim_scores = np.column_stack(
            [
                similarity_matrix[:, start:end].max(axis=1)
                for start, end in (self.template_slices[dim] for dim in self.template_dimensions)
            ]
        ).astype(np.float32)
        sorted_scores = np.sort(dim_scores, axis=1)
        top2_avg = sorted_scores[:, -2:].mean(axis=1)
        risk_scores = 0.6 * sorted_scores[:, -1] + 0.4 * top2_avg
        return risk_scores.astype(np.float32), dim_scores

    def _build_results(self, posts: Sequence[dict], risk_scores: np.ndarray, dim_scores: np.ndarray) -> list[dict]:
        results = []
        for idx, post in enumerate(posts):
            per_dim = {
                dim: round(float(dim_scores[idx, dim_idx]), 4)
                for dim_idx, dim in enumerate(self.template_dimensions)
            }
            results.append(
                {
                    "post_id": post["post_id"],
                    "text": post["text"],
                    "posting_time": post.get("posting_time"),
                    "risk_score": round(float(risk_scores[idx]), 4),
                    "matched_dimensions": [dim for dim, score in per_dim.items() if score >= 0.5],
                    "dim_scores": per_dim,
                }
            )
        return results

    def screen_user(self, user_posts: List[dict]) -> List[dict]:
        if not user_posts:
            return []
        texts = [post["text"] for post in user_posts]
        post_embeddings = self._encode_texts(texts)
        risk_scores, dim_scores = self._compute_scores(post_embeddings)
        results = self._build_results(user_posts, risk_scores, dim_scores)
        results.sort(key=lambda item: item["risk_score"], reverse=True)
        return results[: compute_dynamic_k(len(results))]

    def screen_all_users(self, all_user_posts: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
        from tqdm import tqdm

        user_limits = {user_id: compute_dynamic_k(len(posts)) for user_id, posts in all_user_posts.items()}
        heaps: dict[str, list[tuple[float, int, dict]]] = defaultdict(list)
        tie_breaker = count()
        total_posts = sum(len(posts) for posts in all_user_posts.values())
        progress = tqdm(total=total_posts, desc="PHQ-9 模板筛选", unit="posts")

        batch_entries: list[tuple[str, dict]] = []
        batch_texts: list[str] = []
        for user_id, posts in all_user_posts.items():
            for post in posts:
                batch_entries.append((user_id, post))
                batch_texts.append(post["text"])
                if len(batch_texts) >= self.encode_chunk_size:
                    self._consume_batch(batch_entries, batch_texts, user_limits, heaps, tie_breaker)
                    progress.update(len(batch_texts))
                    batch_entries.clear()
                    batch_texts.clear()

        if batch_texts:
            self._consume_batch(batch_entries, batch_texts, user_limits, heaps, tie_breaker)
            progress.update(len(batch_texts))

        progress.close()
        results: dict[str, list[dict]] = {}
        for user_id in all_user_posts:
            ranked = sorted(heaps.get(user_id, []), key=lambda item: (item[0], item[1]), reverse=True)
            results[user_id] = [item[2] for item in ranked]
        return results

    def _consume_batch(
        self,
        batch_entries: Sequence[tuple[str, dict]],
        batch_texts: Sequence[str],
        user_limits: Dict[str, int],
        heaps: dict[str, list[tuple[float, int, dict]]],
        tie_breaker: Iterable[int],
    ) -> None:
        embeddings = self._encode_texts(batch_texts)
        risk_scores, dim_scores = self._compute_scores(embeddings)
        posts = [post for _, post in batch_entries]
        rows = self._build_results(posts, risk_scores, dim_scores)
        for (user_id, _), row in zip(batch_entries, rows):
            limit = user_limits.get(user_id, 0)
            if limit <= 0:
                continue
            heap = heaps[user_id]
            entry = (row["risk_score"], next(tie_breaker), row)
            if len(heap) < limit:
                heapq.heappush(heap, entry)
            elif entry[0] > heap[0][0]:
                heapq.heapreplace(heap, entry)
