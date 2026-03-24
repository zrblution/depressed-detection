"""Distributed training helpers."""

from __future__ import annotations

from typing import Any, Iterable

import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce_scalar(value: float, *, device: str | None = None) -> float:
    if not is_distributed():
        return float(value)
    import torch

    scalar = torch.tensor(float(value), device=device or "cuda")
    dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
    return float(scalar.item())


def broadcast_object(obj: Any, *, src: int = 0) -> Any:
    if not is_distributed():
        return obj
    objects = [obj if get_rank() == src else None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def all_gather_objects(local_obj: Any) -> list[Any]:
    if not is_distributed():
        return [local_obj]
    gathered = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, local_obj)
    return gathered


def flatten_gathered(items: Iterable[list[Any]]) -> list[Any]:
    merged: list[Any] = []
    for item in items:
        merged.extend(item)
    return merged
