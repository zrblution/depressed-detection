# 02 — 特征工程 Agent：Evidence Block + 弱先验 + 用户级样本构建

---

## Agent 目标

将 `01_Data_Pipeline` 产出的帖子级打分结果，聚合为**用户级特征**：
1. 构建 Evidence Block（基于时间邻近性）
2. 计算三通道弱先验 `π_u = (p_sd, p_ep, p_sp)` + `crisis_score`
3. 构建 Global History Summary（分段采样）
4. 组装用户级训练样本（训练集 depressed 用户双套 risk_posts；其余用户使用推理兼容格式）

> **Backbone 兼容性说明**：本模块构造的用户级样本本质上是 `risk_posts + priors + blocks + global_history` 的统一合同，对下游编码器类型不敏感。当前主线切换到 `microsoft/mdeberta-v3-base` 后，`[META]` 仍可作为普通 special token / 文本标签保留，因此这里的 schema 无需重构。

---

## 输入与输出

### 输入
| 文件 | 来源 | 说明 |
|---|---|---|
| `data/processed/{dataset}/all_users_standardized.jsonl` | 01 标准化输出 | 所有用户的标准化用户级数据（`user_id`, `label`, `gender`, `posts`） |
| `data/processed/{dataset}/depressed_scored_posts.jsonl` | 01 通路 A | depressed 训练用户的帖子级全量打分结果，已展平并带 `composite_evidence_score` |
| `data/processed/{dataset}/risk_posts_a.json` | 01 通路 A | depressed 训练用户的 A 套 risk_posts（动态 K） |
| `data/processed/{dataset}/risk_posts_b.json` | 01 通路 B | 所有用户的 B 套 risk_posts（动态 K） |
| `data/processed/{dataset}/splits.json` | 01 划分输出 | 训练 / 验证 / 测试划分 |

### 输出
| 文件 | 格式 | 说明 |
|---|---|---|
| `data/user_samples/{dataset}_train.jsonl` | JSONL | 用户级训练样本（每行一个用户） |
| `data/user_samples/{dataset}_val.jsonl` | JSONL | 用户级验证样本 |
| `data/user_samples/{dataset}_test.jsonl` | JSONL | 用户级测试样本 |

---

## 推荐技术栈

| 组件 | 推荐库 |
|---|---|
| 数据处理 | `pandas`, `numpy` |
| 日期计算 | `datetime` |
| 序列化 | `json`, `jsonlines` |

---

## Step-by-Step 实现

### Step 1：合格证据帖过滤 (`src/features/evidence_block.py` — 第一部分)

```python
"""
evidence_block.py — 合格证据帖过滤 + Evidence Block 构建

合格条件：
  1. first_person == True
  2. literal_self_evidence == True
  3. composite_evidence_score >= 0.3
"""
from datetime import datetime, timedelta
from typing import List, Dict
from collections import defaultdict


ELIGIBLE_THRESHOLD = 0.3
HIGH_RISK_THRESHOLD = 0.5


def filter_eligible_posts(scored_posts: List[dict]) -> List[dict]:
    """
    从全量打分帖子中筛选合格证据帖。
    
    条件：
    - first_person == True
    - literal_self_evidence == True  
    - composite_evidence_score >= 0.3
    
    Args:
        scored_posts: 已计算 composite_evidence_score 的帖子列表
    
    Returns:
        eligible_posts: 合格证据帖列表（按时间排序）
    """
    eligible = []
    for p in scored_posts:
        if (p.get("first_person", False) and
            p.get("literal_self_evidence", False) and
            p.get("composite_evidence_score", 0) >= ELIGIBLE_THRESHOLD):
            eligible.append(p)
    
    eligible.sort(key=lambda x: x["posting_time"])
    return eligible
```

---

### Step 2：Evidence Block 构建 (`src/features/evidence_block.py` — 第二部分)

```python
def build_evidence_blocks(eligible_posts: List[dict],
                          max_gap_days: int = 7) -> List[dict]:
    """
    将合格证据帖按时间邻近性分组为 evidence blocks。
    
    规则：
    - 按时间排序后，相邻两条证据帖间隔 <= max_gap_days 天 → 同一 block
    - 间隔超过 max_gap_days → 开启新 block
    
    Args:
        eligible_posts: 合格证据帖（已按时间排序）
        max_gap_days: 同一 block 内允许的最大间隔天数
    
    Returns:
        blocks: List[dict], 每个 block 包含完整统计特征
    """
    if not eligible_posts:
        return []
    
    # 解析日期
    def parse_time(t):
        if isinstance(t, str):
            return datetime.fromisoformat(t.replace('Z', '+00:00'))
        return t
    
    blocks = []
    current_block_posts = [eligible_posts[0]]
    
    for i in range(1, len(eligible_posts)):
        prev_time = parse_time(eligible_posts[i - 1]["posting_time"])
        curr_time = parse_time(eligible_posts[i]["posting_time"])
        gap = (curr_time - prev_time).days
        
        if gap <= max_gap_days:
            current_block_posts.append(eligible_posts[i])
        else:
            blocks.append(_compute_block_features(current_block_posts, len(blocks)))
            current_block_posts = [eligible_posts[i]]
    
    # 最后一个 block
    blocks.append(_compute_block_features(current_block_posts, len(blocks)))
    
    return blocks


def _compute_block_features(block_posts: List[dict], block_id: int) -> dict:
    """
    计算单个 block 的统计特征。
    
    特征列表：
    | 特征                       | 计算方式                                    |
    |---------------------------|---------------------------------------------|
    | block_post_count          | 直接计数                                    |
    | block_span_days           | 末帖日期 - 首帖日期                          |
    | symptom_category_count    | 合并所有帖子 symptom_vector，有多少维 > 0     |
    | repeated_days             | 帖子日期去重后计数                           |
    | duration_support          | any(post.duration.has_hint)                 |
    | functional_impairment_max | max(post.functional_impairment)             |
    | crisis_max                | max(post.crisis_level)                      |
    | clinical_anchor_count     | anchor_types 取并集的元素数                  |
    | avg_confidence            | mean(post.confidence)                       |
    | block_score               | 加权组合                                    |
    """
    from datetime import datetime
    
    def parse_time(t):
        if isinstance(t, str):
            return datetime.fromisoformat(t.replace('Z', '+00:00'))
        return t
    
    post_ids = [p["post_id"] for p in block_posts]
    
    # block_span_days
    times = [parse_time(p["posting_time"]) for p in block_posts]
    span_days = (max(times) - min(times)).days if len(times) > 1 else 0
    
    # symptom_category_count
    all_dims = set()
    for p in block_posts:
        sv = p.get("symptom_vector", {})
        for dim, val in sv.items():
            if val > 0:
                all_dims.add(dim)
    symptom_category_count = len(all_dims)
    
    # repeated_days
    unique_dates = set(parse_time(p["posting_time"]).date() for p in block_posts)
    repeated_days = len(unique_dates)
    
    # duration_support
    duration_support = any(p.get("duration", {}).get("has_hint", False) for p in block_posts)
    
    # functional_impairment_max
    fi_max = max((p.get("functional_impairment", 0) for p in block_posts), default=0)
    
    # crisis_max
    crisis_max = max((p.get("crisis_level", 0) for p in block_posts), default=0)
    
    # clinical_anchor_count
    all_anchors = set()
    for p in block_posts:
        anchors = p.get("clinical_context", {}).get("anchor_types", [])
        all_anchors.update(anchors)
    clinical_anchor_count = len(all_anchors)
    
    # avg_confidence
    confidences = [p.get("confidence", 0) for p in block_posts]
    avg_confidence = sum(confidences) / max(len(confidences), 1)
    
    # block_score（加权组合）
    block_score = (
        0.25 * min(len(block_posts) / 5, 1.0) +  # post count
        0.20 * min(span_days / 14, 1.0) +          # span
        0.20 * min(symptom_category_count / 5, 1.0) +  # symptom diversity
        0.15 * (1.0 if duration_support else 0.0) +
        0.10 * min(fi_max / 3, 1.0) +
        0.10 * avg_confidence
    )
    
    return {
        "block_id": block_id,
        "post_ids": post_ids,
        "block_post_count": len(block_posts),
        "block_span_days": span_days,
        "symptom_category_count": symptom_category_count,
        "repeated_days": repeated_days,
        "duration_support": duration_support,
        "functional_impairment_max": fi_max,
        "crisis_max": crisis_max,
        "clinical_anchor_count": clinical_anchor_count,
        "avg_confidence": round(avg_confidence, 4),
        "block_score": round(block_score, 4),
        # 保留代表帖子文本（取 composite_evidence_score 最高的 3 条）
        "representative_posts": sorted(
            block_posts, key=lambda x: x.get("composite_evidence_score", 0), reverse=True
        )[:3]
    }
```

---

### Step 3：三通道弱先验计算 (`src/features/weak_priors.py`)

```python
"""
weak_priors.py — 三通道弱先验 (p_sd, p_ep, p_sp) + crisis_score 计算

核心：
- 这三个值 不是 one-hot 标签，不要求和为 1
- 它们是连续的弱语义先验，用于辅助 MoE 路由训练
- 仅对 depressed 用户计算（normal 用户全零）
"""
from typing import List, Dict


def compute_p_sd(user_evidence_posts: List[dict]) -> float:
    """
    自述披露先验 (Self-Disclosure Prior)。
    
    核心特征权重：
    - current_self_claim: +0.4
    - anchor_types 非空: +0.2 × len(anchor_types)
    - literal_self_evidence + current temporality: +0.2
    - confidence: +0.1 × conf
    
    最终归一化到 [0, 1]
    """
    if not user_evidence_posts:
        return 0.0
    
    score = 0.0
    for p in user_evidence_posts:
        # 当前自述声明
        if p.get("clinical_context", {}).get("disease_mention_type") == "current_self_claim":
            score += 0.4
        # 临床锚点
        anchors = p.get("clinical_context", {}).get("anchor_types", [])
        if len(anchors) > 0:
            score += 0.2 * len(anchors)
        # 自述 + 当前时态
        if p.get("literal_self_evidence", False) and p.get("temporality") == "current":
            score += 0.2
        # 置信度
        score += 0.1 * p.get("confidence", 0)
    
    return min(score / max(len(user_evidence_posts), 1), 1.0)


def compute_p_ep(user_blocks: List[dict]) -> float:
    """
    时间支持先验 (Episode-Supported Prior)。
    
    基于最佳 evidence block 的统计特征：
    - block_post_count: 0.3 × min(count/5, 1.0)
    - block_span_days: 0.2 × min(span/14, 1.0) 
    - symptom_category_count: 0.2 × min(cats/5, 1.0)
    - duration_support: 0.15 × bool
    - functional_impairment_max: 0.15 × min(fi/3, 1.0)
    """
    if not user_blocks:
        return 0.0
    
    best_block = max(user_blocks, key=lambda b: b["block_score"])
    
    score = 0.0
    score += 0.30 * min(best_block["block_post_count"] / 5, 1.0)
    score += 0.20 * min(best_block["block_span_days"] / 14, 1.0)
    score += 0.20 * min(best_block["symptom_category_count"] / 5, 1.0)
    score += 0.15 * (1.0 if best_block["duration_support"] else 0.0)
    score += 0.15 * min(best_block["functional_impairment_max"] / 3, 1.0)
    
    return min(score, 1.0)


def compute_p_sp(user_evidence_posts: List[dict],
                 p_sd: float, p_ep: float) -> float:
    """
    稀疏证据先验 (Sparse-Evidence Prior)。
    
    触发条件：
    1. p_sd < 0.5 且 p_ep < 0.5（不属于前两种主导模式）
    2. 合格证据帖数量 <= 3
    3. 至少存在 1 条帖子（排除零证据用户）
    
    计算：
    - 0.5 × top-1 composite_score
    - 0.3 × top-2 composite_score （如有）
    - 0.2 × avg_confidence
    """
    # 排除已被其他通道覆盖的用户
    if p_sd >= 0.5 or p_ep >= 0.5:
        return 0.0
    
    # 排除证据帖过多的用户
    if len(user_evidence_posts) > 3:
        return 0.0
    
    if not user_evidence_posts:
        return 0.0
    
    top_scores = sorted(
        [p.get("composite_evidence_score", 0) for p in user_evidence_posts],
        reverse=True
    )[:3]
    
    score = 0.5 * top_scores[0]
    if len(top_scores) > 1:
        score += 0.3 * top_scores[1]
    
    avg_conf = sum(p.get("confidence", 0) for p in user_evidence_posts) / len(user_evidence_posts)
    score += 0.2 * avg_conf
    
    return min(score, 1.0)


def compute_crisis_score(scored_posts: List[dict]) -> int:
    """
    危机分数：取用户全部帖子中 crisis_level 的最大值。
    
    Returns:
        crisis_score ∈ {0, 1, 2, 3}
    """
    if not scored_posts:
        return 0
    return max(p.get("crisis_level", 0) for p in scored_posts)


def compute_all_priors(user_evidence_posts: List[dict],
                       user_blocks: List[dict],
                       all_scored_posts: List[dict]) -> Dict:
    """
    计算用户的完整弱先验。
    
    Returns:
        {
            "self_disclosure": float,
            "episode_supported": float,
            "sparse_evidence": float,
            "crisis_score": int
        }
    """
    p_sd = compute_p_sd(user_evidence_posts)
    p_ep = compute_p_ep(user_blocks)
    p_sp = compute_p_sp(user_evidence_posts, p_sd, p_ep)
    crisis = compute_crisis_score(all_scored_posts)
    
    return {
        "self_disclosure": round(p_sd, 4),
        "episode_supported": round(p_ep, 4),
        "sparse_evidence": round(p_sp, 4),
        "crisis_score": crisis
    }
```

---

### Step 4：Global History Summary 构建 (`src/features/global_history.py`)

```python
"""
global_history.py — 全局历史摘要构建

将用户全部帖子按时间分为 S=8 段，每段采样 K_seg 条。
K_seg = ceil(0.6 × N / S)，cap at 128，保证覆盖率 >= 60%。
"""
import math
from typing import List


def build_global_history(all_posts: List[dict],
                         S: int = 8,
                         coverage: float = 0.6,
                         k_seg_max: int = 128) -> List[List[dict]]:
    """
    构建全局历史摘要。
    
    Args:
        all_posts: 用户全部帖子（已按时间排序）
        S: 分段数，默认 8
        coverage: 目标覆盖率，默认 0.6（60%）
        k_seg_max: 每段最大采样数，默认 128
    
    Returns:
        segments: List[List[dict]], S 段，每段含采样帖子列表
    """
    N = len(all_posts)
    if N == 0:
        return [[] for _ in range(S)]
    
    # 动态 K_seg 计算
    k_seg = min(math.ceil(coverage * N / S), k_seg_max)
    
    # 等分为 S 段
    segment_size = math.ceil(N / S)
    segments = []
    
    for s_idx in range(S):
        start = s_idx * segment_size
        end = min(start + segment_size, N)
        seg_posts = all_posts[start:end]
        
        if len(seg_posts) <= k_seg:
            # 段内帖子不足 K_seg，全部保留
            sampled = seg_posts
        else:
            # 均匀采样 k_seg 条
            step = len(seg_posts) / k_seg
            indices = [int(i * step) for i in range(k_seg)]
            sampled = [seg_posts[idx] for idx in indices]
        
        segments.append(sampled)
    
    return segments


def compute_global_stats(all_posts: List[dict],
                         eligible_evidence_count: int) -> dict:
    """
    计算全局统计特征。
    
    Returns:
        {
            "total_posts": int,
            "eligible_evidence_posts": int,
            "posting_freq": float,     # 帖子/天
            "active_span_days": int,
            "temporal_burstiness": float
        }
    """
    from datetime import datetime
    
    if not all_posts:
        return {
            "total_posts": 0,
            "eligible_evidence_posts": 0,
            "posting_freq": 0.0,
            "active_span_days": 0,
            "temporal_burstiness": 0.0
        }
    
    def parse_time(t):
        if isinstance(t, str):
            return datetime.fromisoformat(t.replace('Z', '+00:00'))
        return t
    
    times = [parse_time(p["posting_time"]) for p in all_posts]
    span = (max(times) - min(times)).days
    
    posting_freq = len(all_posts) / max(span, 1)
    
    # temporal_burstiness: 帖子间隔的变异系数
    if len(times) > 1:
        import numpy as np
        intervals = [(times[i+1] - times[i]).total_seconds() / 86400
                      for i in range(len(times) - 1)]
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        burstiness = std_interval / max(mean_interval, 1e-6)
    else:
        burstiness = 0.0
    
    return {
        "total_posts": len(all_posts),
        "eligible_evidence_posts": eligible_evidence_count,
        "posting_freq": round(posting_freq, 4),
        "active_span_days": span,
        "temporal_burstiness": round(burstiness, 4)
    }
```

---

### Step 5：用户级训练样本组装 (`src/features/user_sample_builder.py`)

```python
"""
user_sample_builder.py — 组装用户级训练样本

训练集 depressed 用户结构：
{
    user_id, label=1,
    priors: {p_sd, p_ep, p_sp},
    crisis_score,
    risk_posts_llm: A 套 risk_posts,
    risk_posts_template: B 套 risk_posts,
    episode_blocks: top-m blocks,
    global_history_posts: 8 段采样帖子,
    global_stats: 统计特征
}

Template-only 用户结构（normal 用户 + val/test depressed 用户）：
{
    user_id, label in {0, 1},
    priors: {0, 0, 0},
    crisis_score: 0,
    risk_posts_llm: [],
    risk_posts_template: B 套 risk_posts,  (无 risk_posts_llm)
    episode_blocks: [],
    global_history_posts: 8 段采样帖子,
    global_stats: 统计特征
}
"""
import json
from typing import Dict, List
from src.features.evidence_block import filter_eligible_posts, build_evidence_blocks
from src.features.weak_priors import compute_all_priors
from src.features.global_history import build_global_history, compute_global_stats


def build_depressed_user_sample(
    user_id: str,
    all_scored_posts: List[dict],     # LLM 全量打分帖子
    risk_posts_a: List[dict],          # A 套 risk_posts
    risk_posts_b: List[dict],          # B 套 risk_posts
    all_raw_posts: List[dict],         # 全部原始帖子（用于 global_history）
    max_gap_days: int = 7,
    max_blocks: int = 3
) -> dict:
    """
    构建 depressed 用户的训练样本。
    
    步骤：
    1. 从全量打分帖子中过滤合格证据帖（score >= 0.3）
    2. 构建 evidence blocks
    3. 计算三通道弱先验 + crisis_score
    4. 取 top-m blocks
    5. 构建 global_history_summary
    6. 组装完整样本
    """
    # 1. 合格证据帖
    eligible_posts = filter_eligible_posts(all_scored_posts)
    
    # 2. Evidence blocks
    blocks = build_evidence_blocks(eligible_posts, max_gap_days)
    top_blocks = sorted(blocks, key=lambda b: b["block_score"], reverse=True)[:max_blocks]
    
    # 3. 弱先验
    priors = compute_all_priors(eligible_posts, blocks, all_scored_posts)
    
    # 4. Global history
    all_raw_posts_sorted = sorted(all_raw_posts, key=lambda x: x["posting_time"])
    global_segments = build_global_history(all_raw_posts_sorted)
    global_stats = compute_global_stats(all_raw_posts_sorted, len(eligible_posts))
    
    # 5. 组装
    sample = {
        "user_id": user_id,
        "label": 1,
        "priors": {
            "self_disclosure": priors["self_disclosure"],
            "episode_supported": priors["episode_supported"],
            "sparse_evidence": priors["sparse_evidence"]
        },
        "crisis_score": priors["crisis_score"],
        "risk_posts_llm": [
            {
                "post_id": p["post_id"],
                "text": p["text"],
                "composite_evidence_score": p["composite_evidence_score"],
                "crisis_level": p.get("crisis_level", 0),
                "temporality": p.get("temporality", "unclear")
            }
            for p in risk_posts_a
        ],
        "risk_posts_template": [
            {"post_id": p["post_id"], "text": p["text"],
             "risk_score": p["risk_score"],
             "matched_dimensions": p.get("matched_dimensions", [])}
            for p in risk_posts_b
        ],
        "episode_blocks": top_blocks,
        "global_history_posts": [
            [{"post_id": p["post_id"], "text": p["text"]} for p in seg]
            for seg in global_segments
        ],
        "global_stats": global_stats
    }
    
    return sample


def build_template_only_user_sample(
    user_id: str,
    label: int,
    risk_posts_b: List[dict],          # B 套 risk_posts（仅此一套）
    all_standardized_posts: List[dict] # 标准化后的全部帖子
) -> dict:
    """
    构建推理兼容的用户样本。
    
    适用对象：
    - normal 用户
    - 验证/测试 split 中的 depressed 用户
    
    设计要求：
    - 保留真实监督标签（held-out depressed 仍为 label=1）
    - priors 全零
    - 无 risk_posts_llm
    - episode_blocks 为空
    """
    all_raw_sorted = sorted(all_standardized_posts, key=lambda x: x["posting_time"])
    global_segments = build_global_history(all_raw_sorted)
    global_stats = compute_global_stats(all_raw_sorted, 0)
    
    sample = {
        "user_id": user_id,
        "label": int(label),
        "priors": {
            "self_disclosure": 0.0,
            "episode_supported": 0.0,
            "sparse_evidence": 0.0
        },
        "crisis_score": 0,
        "risk_posts_llm": [],
        "risk_posts_template": [
            {"post_id": p["post_id"], "text": p["text"],
             "risk_score": p["risk_score"],
             "matched_dimensions": p.get("matched_dimensions", [])}
            for p in risk_posts_b
        ],
        "episode_blocks": [],
        "global_history_posts": [
            [{"post_id": p["post_id"], "text": p["text"]} for p in seg]
            for seg in global_segments
        ],
        "global_stats": global_stats
    }
    
    return sample
```

---

### Step 6：数据划分与完整构建脚本 (`scripts/build_user_samples.py`)

```python
"""
构建全部用户级训练样本。

用法：
  python scripts/build_user_samples.py \
    --dataset weibo \
    --standardized_file data/processed/weibo/all_users_standardized.jsonl \
    --scored_file data/processed/weibo/depressed_scored_posts.jsonl \
    --risk_a_file data/processed/weibo/risk_posts_a.json \
    --risk_b_file data/processed/weibo/risk_posts_b.json \
    --splits_file data/processed/weibo/splits.json \
    --output_dir data/user_samples
"""
import argparse, json
from tqdm import tqdm
from src.data.processed_loader import load_standardized_users, load_grouped_scored_posts
from src.features.user_sample_builder import (
    build_depressed_user_sample,
    build_template_only_user_sample,
)


def main(args):
    # 加载 01 的正式产物
    users = load_standardized_users(args.standardized_file)
    scored_posts = load_grouped_scored_posts(args.scored_file)
    with open(args.risk_a_file, 'r', encoding='utf-8') as f:
        risk_posts_a = json.load(f)
    with open(args.risk_b_file, 'r', encoding='utf-8') as f:
        risk_posts_b = json.load(f)
    with open(args.splits_file, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    # 按 split 分组
    for split in ["train", "val", "test"]:
        output_path = f"{args.output_dir}/{args.dataset}_{split}.jsonl"
        users_in_split = splits.get(split, [])
        
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for uid in tqdm(users_in_split, desc=f"构建 {split} 样本"):
                user = users[uid]
                label = int(user["label"])
                raw_posts = user["posts"]
                template_rp = risk_posts_b.get(uid, [])
                
                if label == 1 and split == "train":
                    # 训练集 depressed 用户：消费 01 固化的 A/B 套正式产物
                    scored = scored_posts.get(uid, [])
                    risk_a = risk_posts_a.get(uid, [])
                    if not scored or not risk_a:
                        print(f"[WARN] {uid} 缺少 A 套产物，降级为 template-only depressed 格式")
                        sample = build_template_only_user_sample(
                            uid, label=1, risk_posts_b=template_rp, all_standardized_posts=raw_posts
                        )
                        out_f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                        continue

                    sample = build_depressed_user_sample(
                        uid, scored, risk_a, template_rp, raw_posts
                    )
                else:
                    # normal 用户 / val-test depressed 用户：统一使用推理兼容格式
                    sample = build_template_only_user_sample(
                        uid, label=label, risk_posts_b=template_rp, all_standardized_posts=raw_posts
                    )
                
                out_f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"[{split}] 完成: {len(users_in_split)} 用户 → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--standardized_file", required=True)
    parser.add_argument("--scored_file", required=True)
    parser.add_argument("--risk_a_file", required=True)
    parser.add_argument("--risk_b_file", required=True)
    parser.add_argument("--splits_file", required=True)
    parser.add_argument("--output_dir", default="data/user_samples")
    args = parser.parse_args()
    main(args)
```

---

## 注意事项

1. **Evidence Block 构建基于全量打分帖子**（不受 top-K 限制），保证 block 结构的完整性
2. **弱先验三个值独立计算**，不要求和为 1，都在 [0, 1] 范围
3. **p_sp 有排他条件**：当 p_sd ≥ 0.5 或 p_ep ≥ 0.5 时，p_sp 自动为 0
4. **K_seg 动态计算**保证覆盖率 ≥ 60%，但 cap 在 128 防止极端用户（eRisk 3385 条）内存爆炸
5. **Template-only 样本保留真实监督标签**：normal 用户是 `label=0`，val/test depressed 用户仍是 `label=1`，只是输入退化为 B 套 + 零先验
6. **数据划分**：Weibo/Twitter 使用 80/10/10 划分，eRisk 使用 5-fold CV
