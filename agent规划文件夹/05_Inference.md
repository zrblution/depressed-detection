# 05 — 推理部署 Agent：端到端推理 Pipeline + 解释生成

---

## Agent 目标

实现 WPG-MoE 系统的完整推理 Pipeline：

1. **PHQ-9 模板筛选**：为新用户产出 B 套 risk_posts（~毫秒/用户）
2. **全局历史采样**：分段采样 global_history_summary
3. **模型推理**：编码 → 用户表示 → MoE → 分类 + 证据选择
4. **解释生成**：证据帖子 → LLM 生成自然语言解释（可选后处理）

> **核心保证**：推理时**完全不依赖 LLM 打分**，仅需预训练 embedding 模型（模板筛选）+ 训练好的模型权重。

> **Backbone 选择说明**：对当前帖子级判别任务，推理主线推荐使用统一多语言 encoder `microsoft/mdeberta-v3-base`。`Qwen3.5-2B` 可保留为对照实验或后处理解释模型，但不再建议作为默认帖子编码器。

---

## 输入与输出

### 输入
| 文件/资源 | 说明 |
|---|---|
| 原始用户 JSONL | 外部输入契约：每行一个用户对象 `{nickname, label, gender, tweets:[...]}` |
| 标准化帖子列表 | 内部输入契约：`List[{"user_id", "post_id", "text", "posting_time", "tweet_is_original"}]`，由推理入口从原始 JSONL 展平得到 |
| 模型权重 | `data/models/{dataset}_final_model.pt` |
| 模板编码器 | `thenlper/gte-small-zh` 或 `all-MiniLM-L6-v2` |

### 输出
| 字段 | 类型 | 说明 |
|---|---|---|
| `user_id` | str | 用户 ID |
| `label` | int | 0/1 预测标签 |
| `depressed_logit` | float | 抑郁概率 |
| `crisis_score` | int | 为保持训练 / 推理输出合同一致，推理阶段固定返回 0 |
| `gate_weights` | List[float] | 5 维专家权重 |
| `dominant_channel` | str | 主导通道名称 |
| `evidence_post_ids` | List[str] | top-3 证据帖子真实 `post_id`（如 `user_001__3`） |
| `evidence_scores` | List[float] | 证据分数 |
| `explanation` | str | 可选，LLM 生成的自然语言解释 |

---

## 推荐技术栈

| 组件 | 推荐库 |
|---|---|
| 模板筛选 | `sentence-transformers` |
| 模型推理 | `torch`, `transformers` |
| 解释生成 | `openai` / `vllm`（可选） |

---

## Step-by-Step 实现

### Step 1：推理 Pipeline (`src/inference/pipeline.py`)

```python
"""
pipeline.py — 端到端推理 Pipeline

完整流程:
  原始用户 JSONL → 标准化 / 展平（nickname→user_id, tweet_content→text, tweets[]→posts, 生成 post_id）
                → PHQ-9 模板筛选(B 套 risk_posts, 动态 K)
                → 全局历史采样(8 段 × K_seg)
             → 帖子编码([POST_id] text, 无 META)
             → 用户表示构造(5 路 attention pooling)
             → 门控 + MoE 融合(零先验)
             → 分类 + 证据选择
             → 结构化输出

推理时不可用信息（全部置零/空）:
  - risk_posts_llm (A 套) → 不存在
  - [META] 标签 → 不附加
  - episode_blocks → 空列表
  - user_meta_priors (π_u) → 零向量
  - crisis_score → 0
"""
import torch
import json
from typing import List, Dict
from src.data.template_screener import PHQ9TemplateScreener
from src.features.global_history import build_global_history, compute_global_stats
from src.model.full_model import WPGMoEModel


CHANNEL_NAMES = {
    0: "self_disclosure",
    1: "episode_supported",
    2: "sparse_evidence",
    3: "mixed",
    4: "general"
}


def normalize_bool_like(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def raw_user_to_standardized_posts(raw_user: dict) -> tuple[str, List[dict]]:
    """
    将原始用户级 JSONL 记录转换为内部帖子列表。
    
    字段映射：
    - nickname -> user_id
    - tweets[] -> 帖子列表
    - tweet_content -> text
    - 合成 post_id = f"{user_id}__{idx}"
    """
    user_id = str(raw_user["nickname"])
    posts = []
    for idx, tweet in enumerate(raw_user.get("tweets", [])):
        posts.append({
            "user_id": user_id,
            "post_id": f"{user_id}__{idx}",
            "text": tweet["tweet_content"],
            "posting_time": tweet["posting_time"],
            "tweet_is_original": normalize_bool_like(tweet.get("tweet_is_original", "True"))
        })
    posts.sort(key=lambda x: x["posting_time"])
    return user_id, posts


class InferencePipeline:
    """
    WPG-MoE 推理 Pipeline。
    
    初始化时加载:
    1. PHQ-9 模板筛选器 (gte-small / MiniLM)
    2. 训练好的 WPGMoE 模型权重
    """
    
    def __init__(self,
                 model_path: str,
                 model_config: dict,
                 screener_model: str = "thenlper/gte-small-zh",
                 language: str = "zh",
                 selection_cfg: dict | None = None,
                 device: str = "cuda"):
        """
        Args:
            model_path: 模型权重路径
            model_config: 模型配置字典（与训练一致）
            screener_model: 模板筛选编码器名称
            language: "zh" 或 "en"
            selection_cfg: risk_posts 动态 K 策略配置
            device: 推理设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.selection_cfg = selection_cfg or {"mode": "dynamic_k"}
        
        # 1. 加载模板筛选器
        self.screener = PHQ9TemplateScreener(
            model_name=screener_model,
            language=language,
            selection_cfg=self.selection_cfg
        )
        
        # 2. 加载模型
        self.model = WPGMoEModel(model_config).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    @torch.no_grad()
    def predict_from_raw_user(self, raw_user: dict) -> dict:
        user_id, posts = raw_user_to_standardized_posts(raw_user)
        return self.predict(user_id, posts)
    
    @torch.no_grad()
    def predict(self, user_id: str, posts: List[dict]) -> dict:
        """
        对单个用户执行推理。
        
        Args:
            user_id: 用户 ID
            posts: 标准化后的帖子列表，每条含 {"user_id", "post_id", "text", "posting_time"}
        
        Returns:
            预测结果字典
        """
        if not posts:
            return self._empty_result(user_id)
        
        posts_sorted = sorted(posts, key=lambda x: x["posting_time"])
        
        # ===== Step 1: PHQ-9 模板筛选 =====
        risk_posts = self.screener.screen_user(posts_sorted)  # dynamic-K
        
        # ===== Step 2: 全局历史采样 =====
        global_segments = build_global_history(posts_sorted, S=8)
        global_stats = compute_global_stats(posts_sorted, 0)
        
        # ===== Step 3: 构造模型输入 =====
        # risk_posts 文本（无 META 标签）
        risk_texts = []
        risk_markers = []
        for i, rp in enumerate(risk_posts):
            marker = f"[POST_{i+1:02d}]"
            risk_texts.append(f"{marker} {rp['text']}")
            risk_markers.append(marker)
        
        # Episode blocks: 推理时始终为空
        block_texts = []
        block_markers = []
        
        # Global history
        global_segment_texts = []
        global_segment_markers = []
        marker_idx = len(risk_markers) + 1
        for seg in global_segments:
            seg_texts = []
            seg_markers = []
            for gp in seg:
                marker = f"[POST_{marker_idx:02d}]"
                seg_texts.append(f"{marker} {gp['text']}")
                seg_markers.append(marker)
                marker_idx += 1
            global_segment_texts.append(seg_texts)
            global_segment_markers.append(seg_markers)
        
        # 先验: 推理时全零
        pi_u = torch.zeros(3, dtype=torch.float32, device=self.device)
        crisis = torch.zeros(1, dtype=torch.float32, device=self.device)
        
        # 统计特征
        stats = torch.tensor([
            global_stats.get("posting_freq", 0),
            global_stats.get("temporal_burstiness", 0),
            0.0,  # avg_sentiment_trend placeholder
            min(global_stats.get("total_posts", 0) / 1000.0, 1.0),
            min(global_stats.get("active_span_days", 0) / 365.0, 1.0)
        ], dtype=torch.float32, device=self.device)
        
        # ===== Step 4: 模型前向推理 =====
        output = self.model(
            risk_post_texts=risk_texts,
            risk_post_markers=risk_markers,
            block_post_texts=block_texts,
            block_post_markers=block_markers,
            global_segment_texts=global_segment_texts,
            global_segment_markers=global_segment_markers,
            pi_u=pi_u,
            crisis=crisis,
            stats=stats
        )
        
        # ===== Step 5: 解析输出 =====
        logit = torch.sigmoid(output["logit"]).item()
        label = 1 if logit > 0.5 else 0
        gate_weights = output["gate_weights"].cpu().tolist()
        
        # 主导通道
        dominant_idx = output["gate_weights"].argmax().item()
        dominant_channel = CHANNEL_NAMES[dominant_idx]
        
        # 证据帖子选择 (top-3)
        evidence_scores = output["evidence_scores"]
        top_indices, top_scores = self.model.evidence_head.select_top_evidence(
            evidence_scores, top_k=3
        )
        
        evidence_post_ids = [risk_posts[idx.item()]["post_id"] for idx in top_indices]
        evidence_texts = [risk_posts[idx.item()]["text"] for idx in top_indices]
        
        return {
            "user_id": user_id,
            "label": label,
            "depressed_logit": round(logit, 4),
            "crisis_score": 0,
            "gate_weights": [round(w, 4) for w in gate_weights],
            "dominant_channel": dominant_channel,
            "evidence_post_ids": evidence_post_ids,
            "evidence_scores": [round(s.item(), 4) for s in top_scores],
            "evidence_texts": evidence_texts,
            "total_posts": len(posts),
            "risk_posts_count": len(risk_posts)
        }
    
    def predict_batch(self, users: Dict[str, List[dict]]) -> List[dict]:
        """
        批量推理。
        
        Args:
            users: { user_id: [post_dict, ...] }
        
        Returns:
            List[dict], 每个用户一条预测结果
        """
        from tqdm import tqdm
        results = []
        for uid, posts in tqdm(users.items(), desc="推理中"):
            result = self.predict(uid, posts)
            results.append(result)
        return results
    
    def _empty_result(self, user_id: str) -> dict:
        return {
            "user_id": user_id,
            "label": 0,
            "depressed_logit": 0.0,
            "crisis_score": 0,
            "gate_weights": [0.2] * 5,
            "dominant_channel": "general",
            "evidence_post_ids": [],
            "evidence_scores": [],
            "evidence_texts": [],
            "total_posts": 0,
            "risk_posts_count": 0
        }
```

---

### Step 2：解释生成（后处理）(`src/inference/explanation.py`)

```python
"""
explanation.py — 证据帖子解释生成

流程:
1. 主模型输出 evidence_post_ids + evidence_scores
2. 映射回原始帖子文本
3. 构造 prompt 输入 LLM 生成自然语言解释

这是可选的后处理步骤，不影响分类结果。
"""
from typing import List, Dict


EXPLANATION_PROMPT = """你是一个心理健康评估助手。以下是系统从用户社交媒体帖子中识别出的抑郁证据。

用户预测结果: {prediction_label}
预测置信度: {confidence:.2f}
主导模式: {dominant_channel}

证据帖子（按重要性排序）:
{evidence_list}

请基于以上证据，生成一段简洁的临床解释：
1. 概述该用户呈现的主要抑郁信号特征
2. 解释各条证据帖子为何被选为关键证据
3. 评估证据的可靠性和局限性
4. 如果是 depressed，建议关注的方向

请用客观、专业的语气，避免过度诊断。控制在 200 字以内。
"""


def generate_explanation(prediction: dict,
                          llm_client=None) -> str:
    """
    为单个用户的预测结果生成自然语言解释。
    
    Args:
        prediction: predict() 返回的结果字典
        llm_client: LLM API 客户端（可选，无则返回模板解释）
    
    Returns:
        explanation: str
    """
    # 构造证据列表文本
    evidence_list = ""
    for i, (pid, text, score) in enumerate(zip(
        prediction["evidence_post_ids"],
        prediction["evidence_texts"],
        prediction["evidence_scores"]
    )):
        evidence_list += f"  {i+1}. [{pid}] (证据分数: {score})\n     \"{text[:100]}...\"\n"
    
    label_text = "depressed" if prediction["label"] == 1 else "non_depressed"
    
    prompt = EXPLANATION_PROMPT.format(
        prediction_label=label_text,
        confidence=prediction["depressed_logit"],
        dominant_channel=prediction["dominant_channel"],
        evidence_list=evidence_list
    )
    
    if llm_client:
        response = llm_client.chat.completions.create(
            model="your-model-name",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content
    else:
        # 无 LLM 时返回结构化模板解释
        return _template_explanation(prediction)


def _template_explanation(prediction: dict) -> str:
    """
    模板化解释（无需 LLM）。
    """
    if prediction["label"] == 0:
        return (f"用户 {prediction['user_id']} 未检测到显著抑郁信号。"
                f"模型置信度: {1 - prediction['depressed_logit']:.1%}。"
                f"通用专家权重最高 ({prediction['gate_weights'][4]:.2f})，"
                f"表明帖子内容未匹配抑郁证据模式。")
    
    channel_desc = {
        "self_disclosure": "自述披露模式（用户直接描述自身抑郁状态或诊断经历）",
        "episode_supported": "时间支持模式（多条帖子在时间上呈现持续性抑郁症状）",
        "sparse_evidence": "稀疏证据模式（少量但高强度的抑郁信号）",
        "mixed": "混合模式（跨多种抑郁表达方式）",
        "general": "通用模式"
    }
    
    desc = channel_desc.get(prediction["dominant_channel"], "未知模式")
    
    evidence_summary = "; ".join(
        f"[{pid}](分数={score})"
        for pid, score in zip(prediction["evidence_post_ids"],
                               prediction["evidence_scores"])
    )
    
    return (f"用户 {prediction['user_id']} 被检测为 depressed "
            f"(置信度: {prediction['depressed_logit']:.1%})。\n"
            f"主导模式: {desc}\n"
            f"关键证据: {evidence_summary}")
```

---

### Step 3：推理入口脚本 (`scripts/infer.py`)

```python
"""
infer.py — 推理入口

用法:
  # 单用户推理
  python scripts/infer.py \
    --model_path data/models/weibo_final_model.pt \
    --config configs/weibo.yaml \
    --input_file data/raw/new_users.jsonl \
    --output_file results/predictions.json

  # 批量推理
  python scripts/infer.py \
    --model_path data/models/weibo_final_model.pt \
    --config configs/weibo.yaml \
    --input_file data/raw/test_users.jsonl \
    --output_file results/test_predictions.json \
    --batch
"""
import argparse, json, yaml
from src.data.raw_loader import load_jsonl_users
from src.inference.pipeline import InferencePipeline
from src.inference.explanation import generate_explanation


def main(args):
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化推理 pipeline
    pipeline = InferencePipeline(
        model_path=args.model_path,
        model_config=config,
        screener_model=config.get("screener_model", "thenlper/gte-small-zh"),
        language=config.get("language", "zh"),
        selection_cfg=config.get("risk_selection", {"mode": "dynamic_k"}),
        device=args.device
    )
    
    # 加载原始用户级 JSONL，并标准化为内部帖子列表
    raw_users = load_jsonl_users(args.input_file, has_score=False)
    user_posts = {u["user_id"]: u["posts"] for u in raw_users}
    print(f"加载 {len(user_posts)} 个用户，共 {sum(len(v) for v in user_posts.values())} 条帖子")
    
    # 推理
    if args.batch:
        results = pipeline.predict_batch(user_posts)
    else:
        results = []
        for uid, posts in user_posts.items():
            result = pipeline.predict(uid, posts)
            results.append(result)
    
    # 可选：生成解释
    if args.explain:
        try:
            from openai import OpenAI
            llm_client = OpenAI(api_key=config.get("explanation_api_key"))
        except Exception:
            llm_client = None
        
        for result in results:
            result["explanation"] = generate_explanation(result, llm_client)
    
    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印摘要
    depressed_count = sum(1 for r in results if r["label"] == 1)
    print(f"\n推理完成！")
    print(f"总用户数: {len(results)}")
    print(f"Depressed: {depressed_count} ({depressed_count/max(len(results),1)*100:.1f}%)")
    print(f"Non-depressed: {len(results) - depressed_count}")
    print(f"结果保存至: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="模型权重路径")
    parser.add_argument("--config", required=True, help="YAML 配置文件")
    parser.add_argument("--input_file", required=True, help="输入用户级 JSONL")
    parser.add_argument("--output_file", default="results/predictions.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", action="store_true", help="批量推理模式")
    parser.add_argument("--explain", action="store_true", help="生成自然语言解释")
    args = parser.parse_args()
    main(args)
```

---

## 推理延迟预估

| 步骤 | 耗时 | 说明 |
|---|---|---|
| PHQ-9 模板筛选 | ~5ms/用户 | gte-small 编码 + 余弦相似度 |
| 全局历史采样 | ~1ms/用户 | 纯 Python 采样 |
| 帖子编码 | ~60-120ms/用户 | BERT-family encoder，约 `K_dyn + ΣK_seg` 条帖子；实际随 batch / GPU 波动 |
| MoE 前向 | ~10ms/用户 | 用户级轻量 MLP |
| Evidence Head | ~1ms/用户 | `K_dyn` 条帖子打分 |
| **总计** | **~75-135ms/用户** | 通常显著低于 2B causal LM 编码方案 |

---

## 推理 vs 训练输入一致性

| 输入组件 | 训练 (depressed) | 推理 | 桥接机制 |
|---|---|---|---|
| risk_posts | 50% A 套 / 50% B 套（数量由 01 动态 K 产物决定） | **始终 B 套（动态 K）** | Risk Source Swap |
| META 标签 | 25% 有 META | **始终无** | META Dropout |
| episode_blocks | 60% 有 / 40% 空 | **始终空** | Block Dropout |
| 先验 π_u | 70% 非零 / 30% 零 | **始终零** | Prior Dropout |
| global_history | 始终有 | **始终有** | 完全一致 |

---

## 注意事项

1. **推理时完全不需要 LLM**，仅需模板筛选编码器 + 训练好的模型
2. **zero 先验是正常的**：模型通过 Prior Dropout 训练已学会在零先验下仅依赖表示向量路由
3. **解释生成是可选后处理**，不影响分类结果，高延迟场景可跳过
4. **POST marker 需与训练一致**：`[POST_01]`~`[POST_XX]` 的 token ID 必须与训练时相同
5. **global_history 段数和覆盖率必须与训练一致**：S=8, coverage=60%
6. **批量推理可通过并行帖子编码加速**（多用户的帖子合并为大 batch 编码）
7. **若采用 BERT-family 主干**，建议直接复用训练时的 special tokens 与 tokenizer 词表扩展方式，避免证据对齐漂移
