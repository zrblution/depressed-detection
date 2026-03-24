"""Shared post encoder."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.utils.schemas import META_TOKEN, make_post_marker


class PostEncoder(nn.Module):
    """Post encoder that supports both bidirectional and decoder-style backbones."""

    def __init__(
        self,
        model_name: str = "/home/tos_lx/basemodel/mdeberta-v3-base",
        *,
        max_post_tokens: int = 512,
        num_post_markers: int = 2048,
        pooling_strategy: str = "auto",
        torch_dtype: Union[str, torch.dtype, None] = "auto",
        gradient_checkpointing: bool = False,
        use_lora: bool = False,
        trust_remote_code: bool = False,
        device_map: str | None = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_post_tokens = max_post_tokens
        self.num_post_markers = num_post_markers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.cls_token is not None:
                self.tokenizer.pad_token = self.tokenizer.cls_token
            elif self.tokenizer.sep_token is not None:
                self.tokenizer.pad_token = self.tokenizer.sep_token
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"output_hidden_states": True}
        resolved_dtype = self._resolve_torch_dtype(torch_dtype)
        if resolved_dtype is None:
            # `auto` can resolve to fp16 checkpoint weights on GPU, which is unstable
            # for the plain AdamW training paths used in Stage C / Stage E.
            resolved_dtype = torch.float32
        model_kwargs["torch_dtype"] = resolved_dtype
        loading_report_logger = logging.getLogger("transformers.utils.loading_report")
        previous_loading_report_level = loading_report_logger.level
        loading_report_logger.setLevel(logging.ERROR)
        try:
            backbone = AutoModel.from_pretrained(model_name, **model_kwargs)
        finally:
            loading_report_logger.setLevel(previous_loading_report_level)
        post_tokens = [make_post_marker(idx) for idx in range(1, num_post_markers + 1)]
        special_tokens = {"additional_special_tokens": post_tokens + [META_TOKEN]}
        added_tokens = self.tokenizer.add_special_tokens(special_tokens)
        if added_tokens:
            backbone.resize_token_embeddings(len(self.tokenizer))
        self.post_token_ids = {
            token: self.tokenizer.convert_tokens_to_ids(token)
            for token in post_tokens
        }
        if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
            if hasattr(backbone, "enable_input_require_grads"):
                backbone.enable_input_require_grads()
        self.backbone = backbone
        self.cls_token_id = self.tokenizer.cls_token_id
        self.meta_token_id = self.tokenizer.convert_tokens_to_ids(META_TOKEN)
        self.hidden_dim = self._resolve_hidden_size(backbone)
        self.pooling_strategy = self._resolve_pooling_strategy(backbone, pooling_strategy)
        if self.pooling_strategy == "last_token":
            self.tokenizer.padding_side = "left"

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def format_post_text(self, post_marker: str, text: str, meta_info: Optional[Dict[str, object]] = None) -> str:
        formatted = f"{post_marker} {text}"
        if meta_info:
            meta_text = " ".join(f"{key}={value}" for key, value in meta_info.items())
            formatted = f"{formatted} {META_TOKEN} {meta_text}"
        return formatted

    def encode_posts(self, formatted_texts: List[str], post_markers: List[str]) -> torch.Tensor:
        if len(formatted_texts) != len(post_markers):
            raise ValueError("formatted_texts and post_markers must be aligned")
        encodings = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=self.max_post_tokens,
            return_tensors="pt",
        )
        encodings = {key: value.to(self.device) for key, value in encodings.items()}
        last_hidden = self._forward_last_hidden(encodings)
        representations = []
        for batch_idx, marker in enumerate(post_markers):
            if self.pooling_strategy == "last_token":
                position = self._resolve_last_token_position(encodings["attention_mask"][batch_idx])
            else:
                marker_id = self.post_token_ids.get(marker)
                token_ids = encodings["input_ids"][batch_idx]
                position = self._resolve_marker_position(token_ids, marker_id)
            representations.append(last_hidden[batch_idx, position, :])
        return torch.stack(representations, dim=0)

    def _forward_last_hidden(self, encodings: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.backbone(**encodings, return_dict=True)
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is not None:
            return last_hidden
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            return hidden_states[-1]
        raise RuntimeError("backbone did not return hidden states")

    def _resolve_marker_position(self, token_ids: torch.Tensor, marker_id: Optional[int]) -> int:
        if marker_id is not None:
            positions = (token_ids == marker_id).nonzero(as_tuple=True)[0]
            if len(positions):
                return int(positions[0].item())
        if self.cls_token_id is not None:
            positions = (token_ids == self.cls_token_id).nonzero(as_tuple=True)[0]
            if len(positions):
                return int(positions[0].item())
        return 0

    @staticmethod
    def _resolve_last_token_position(attention_mask: torch.Tensor) -> int:
        positions = attention_mask.nonzero(as_tuple=True)[0]
        if len(positions):
            return int(positions[-1].item())
        return 0

    @staticmethod
    def _resolve_torch_dtype(torch_dtype: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
        if torch_dtype is None or torch_dtype == "auto":
            return None
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        if isinstance(torch_dtype, str):
            candidate = torch_dtype.replace("torch.", "")
            resolved = getattr(torch, candidate, None)
            if isinstance(resolved, torch.dtype):
                return resolved
        raise ValueError(f"Unsupported torch_dtype value: {torch_dtype!r}")

    @staticmethod
    def _resolve_hidden_size(backbone: nn.Module) -> int:
        hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(backbone.config, "d_model", None)
        if hidden_size is None and hasattr(backbone.config, "text_config"):
            hidden_size = getattr(backbone.config.text_config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(backbone.config.text_config, "d_model", None)
        if hidden_size is None:
            raise ValueError(f"Unable to determine hidden size for backbone {backbone.__class__.__name__}")
        return int(hidden_size)

    @staticmethod
    def _resolve_pooling_strategy(backbone: nn.Module, requested: str) -> str:
        if requested in {"marker", "last_token"}:
            return requested
        if requested != "auto":
            raise ValueError(f"Unsupported pooling_strategy value: {requested!r}")

        model_type = str(getattr(backbone.config, "model_type", "")).lower()
        architectures = " ".join(getattr(backbone.config, "architectures", []) or []).lower()
        class_name = backbone.__class__.__name__.lower()
        causal_like_signature = " ".join([model_type, architectures, class_name])
        causal_like_tokens = ("qwen", "llama", "mistral", "gpt", "gemma", "phi")
        if bool(getattr(backbone.config, "is_decoder", False)) or any(
            token in causal_like_signature for token in causal_like_tokens
        ):
            return "last_token"
        return "marker"
