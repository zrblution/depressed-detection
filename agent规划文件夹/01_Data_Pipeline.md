# 01 — 数据管线 Agent：原始数据加载 + composite 计算 + PHQ-9 模板筛选

---

## Agent 目标

本模块负责将**原始用户级数据文件**转化为两套结构化数据：

1. **原始数据加载**：读取实际 JSONL/JSON 文件，执行字段映射 + 数据划分（train/val/test）
2. **通路 A**（composite 规则计算）：基于已有 LLM 打分结果计算 `composite_evidence_score` + A 套 risk_posts
3. **通路 B**（PHQ-9 模板筛选）：对所有用户执行轻量级 embedding 相似度筛选，产出 B 套 risk_posts

> **注意**：LLM 全量结构化抽取（通路 A 的 LLM 打分部分）已**离线完成**，`depressed.scored.jsonl` 文件已包含 `score` 嵌套字段。本模块直接加载已有结果。

> **Backbone 兼容性说明**：本模块输出的 `risk_posts / blocks / splits` 对下游主干模型是**schema 无关**的。当前主线已统一为 `microsoft/mdeberta-v3-base`，但这里的数据合同对 `Qwen`、单语 `BERT` 或多语 `BERT` 系列都无需重构。

---

## 输入与输出

### 输入
| 文件 | 格式 | 说明 |
|---|---|---|
| `--scored_path` | JSONL，每行 = 一个用户 | 当前仓库实际形态：`dataset/.../score_data/depressed.scored.jsonl`，顶层 `{nickname, label, gender, tweets}`，其中 `label` 是字符串 `"1"` |
| `--cleaned_path` | JSONL，每行 = 一个用户 | 当前仓库实际形态：`dataset/.../cleaned_data/control.cleaned.jsonl` 或 `cleaned-data/normal.cleaned.jsonl`，顶层 `{nickname, label, gender, tweets}`，其中 `label` 是字符串 `"0"` |

### 输出
| 文件 | 格式 | 说明 |
|---|---|---|
| `data/processed/{dataset}/all_users_standardized.jsonl` | JSONL | 所有用户的标准化用户级数据（`user_id`, `label`, `gender`, `posts`） |
| `data/processed/{dataset}/depressed_scored_posts.jsonl` | JSONL | depressed 用户帖子级数据 + `composite_evidence_score`，已展平 |
| `data/processed/{dataset}/risk_posts_a.json` | JSON | 每用户 top-K (动态) 的 A 套 risk_posts，仅 depressed 用户 |
| `data/processed/{dataset}/risk_posts_b.json` | JSON | 每用户 top-K (动态) 的 B 套 risk_posts，所有用户 |
| `data/processed/{dataset}/splits.json` | JSON | 数据划分结果 `{train: [uid, ...], val: [...], test: [...]}` |

---

## 推荐技术栈

| 组件 | 推荐库 | 版本 |
|---|---|---|
| 轻量编码器 | `sentence-transformers` | ≥ 2.2 |
| 中文编码器 | `thenlper/gte-small-zh` 或 `BAAI/bge-small-zh` | — |
| 英文编码器 | `sentence-transformers/all-MiniLM-L6-v2` | — |
| 数据处理 | `pandas`, `tqdm`, `json` | — |
| 余弦相似度 | `sklearn.metrics.pairwise.cosine_similarity` 或 `torch.nn.functional.cosine_similarity` | — |
| 数据划分 | `sklearn.model_selection.train_test_split` | — |

---

## 动态 K 计算公式

> **所有出现 top-K 选取的地方统一使用此公式**：

```python
def compute_dynamic_K(total_posts: int) -> int:
    """
    动态计算 risk_posts 的 K 值。
    
    规则：
    - total_posts >= 160: K = ceil(total_posts * 0.125)  → 12.5%
    - 20 <= total_posts < 160: K = 20                    → 保底 20
    - total_posts < 20: K = total_posts                  → 全部保留
    """
    import math
    if total_posts >= 160:
        return math.ceil(total_posts * 0.125)
    elif total_posts >= 20:
        return 20
    else:
        return total_posts
```

---

## Step-by-Step 实现

### Step 1：原始数据加载器 (`src/data/raw_loader.py`)

```python
"""
raw_loader.py — 加载原始用户级数据，执行字段映射 + 数据划分

当前仓库的实际数据格式：
  - depressed.scored.jsonl: JSONL, 每行 {nickname, label, gender, tweets: [{tweet_content, posting_time, tweet_is_original, score}]}
  - control.cleaned.jsonl / normal.cleaned.jsonl: JSONL, 每行 {nickname, label, gender, tweets: [{tweet_content, posting_time, tweet_is_original}]}

真实磁盘类型（必须先标准化）：
  - label: "1" / "0"（字符串）
  - gender: "男" / "女" / "None"（字符串）
  - tweet_is_original: "True" / "False"（字符串）

字段映射：
  - nickname → user_id
  - tweet_content → text
  - 合成 post_id = f"{user_id}__{idx}"
  - score.* → 展平到帖子顶层
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def normalize_label(value) -> int:
    """原始 label 统一转成 int。"""
    return int(str(value).strip())


def normalize_bool_like(value) -> bool:
    """兼容 'True' / 'False' / bool / 0/1。"""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def normalize_gender(value) -> Optional[str]:
    """把 'None' / '' / null 统一成 None。"""
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "None", "null", "NULL"}:
        return None
    return text


def load_jsonl_users(filepath: str, has_score: bool) -> List[dict]:
    """加载当前仓库实际使用的用户级 JSONL。"""
    users = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            raw = json.loads(line.strip())
            users.append(normalize_raw_user(raw, has_score=has_score))
    return users


def normalize_raw_user(raw: dict, has_score: bool) -> dict:
    """
    将磁盘上的原始用户对象转换为内部统一格式。
    
    字段映射:
      nickname → user_id
      label(str) → label(int)
      gender("None") → None
      tweet_content → text
      tweet_is_original(str/bool) → bool
      合成 post_id = f"{user_id}__{idx}"
      score.* → 展平到帖子顶层
    """
    user_id = str(raw["nickname"])
    posts = []
    
    for idx, tweet in enumerate(raw.get("tweets", [])):
        post = {
            "post_id": f"{user_id}__{idx}",
            "user_id": user_id,
            "text": tweet["tweet_content"],
            "posting_time": tweet["posting_time"],
            "tweet_is_original": normalize_bool_like(tweet.get("tweet_is_original", "True")),
        }
        
        if has_score and "score" in tweet:
            score = tweet["score"]
            # 展平 score 嵌套字段到帖子顶层
            post["symptom_vector"] = score.get("symptom_vector", {})
            post["first_person"] = score.get("first_person", False)
            post["literal_self_evidence"] = score.get("literal_self_evidence", False)
            post["confidence"] = score.get("confidence", 0.0)
            post["crisis_level"] = score.get("crisis_level", 0)
            post["duration"] = score.get("duration", {"has_hint": False, "hint_span_days": None})
            post["functional_impairment"] = score.get("functional_impairment", 0)
            post["clinical_context"] = score.get("clinical_context", {"disease_mention_type": "none", "anchor_types": []})
            post["temporality"] = score.get("temporality", "unclear")
        
        posts.append(post)
    
    # 按时间排序
    posts.sort(key=lambda x: x["posting_time"])
    
    return {
        "user_id": user_id,
        "label": normalize_label(raw["label"]),
        "gender": normalize_gender(raw.get("gender")),
        "posts": posts
    }


def load_dataset(scored_path: str, cleaned_path: str) -> Tuple[List[dict], List[dict]]:
    """
    加载单个数据集的全部用户。
    
    Args:
        scored_path: scored JSONL 路径
        cleaned_path: cleaned JSONL 路径
    
    Returns:
        (depressed_users, control_users)
    """
    depressed_users = load_jsonl_users(scored_path, has_score=True)
    control_users = load_jsonl_users(cleaned_path, has_score=False)
    return depressed_users, control_users


def generate_splits(users: List[dict], 
                    train_ratio: float = 0.8,
                    val_ratio: float = 0.1,
                    seed: int = 42) -> Dict[str, List[str]]:
    """
    生成用户级分层 train/val/test 划分。
    
    按 label 分层，确保每个 split 中的类别比例一致。
    
    Args:
        users: 全部用户列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        splits: {"train": [uid, ...], "val": [...], "test": [...]}
    """
    from sklearn.model_selection import train_test_split
    
    # 按规范化后的 int label 分组
    depressed = [u["user_id"] for u in users if u["label"] == 1]
    control = [u["user_id"] for u in users if u["label"] == 0]
    
    test_ratio = 1.0 - train_ratio - val_ratio
    val_of_remaining = val_ratio / (val_ratio + test_ratio)
    
    def split_group(uids):
        train, rest = train_test_split(uids, train_size=train_ratio, random_state=seed)
        val, test = train_test_split(rest, train_size=val_of_remaining, random_state=seed)
        return train, val, test
    
    d_train, d_val, d_test = split_group(depressed)
    c_train, c_val, c_test = split_group(control)
    
    return {
        "train": list(d_train) + list(c_train),
        "val": list(d_val) + list(c_val),
        "test": list(d_test) + list(c_test)
    }


def generate_cv_folds(users: List[dict], n_folds: int = 5, 
                      seed: int = 42) -> List[Dict[str, List[str]]]:
    """
    生成 n-fold 交叉验证划分（用于 eRisk）。
    
    Returns:
        folds: List[{"train": [...], "val": [...], "test": [...]}]
    """
    from sklearn.model_selection import StratifiedKFold
    
    uids = [u["user_id"] for u in users]
    labels = [u["label"] for u in users]
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    
    for train_idx, test_idx in skf.split(uids, labels):
        train_uids = [uids[i] for i in train_idx]
        test_uids = [uids[i] for i in test_idx]
        
        # 从 train 中再划出 10% 作为 val
        val_size = max(int(len(train_uids) * 0.1), 1)
        val_uids = train_uids[:val_size]
        train_uids = train_uids[val_size:]
        
        folds.append({
            "train": train_uids,
            "val": val_uids,
            "test": test_uids
        })
    
    return folds
```

---

### Step 2：LLM 结构化抽取器（`src/data/llm_extractor.py`）

> **⚠️ 此步骤已离线完成**，`depressed.scored.jsonl` 已包含 LLM 打分的 `score` 嵌套字段。  
> `llm_extractor.py` **仅保留作为参考工具**，不在主执行流程中调用。  
> 如需对新数据重新执行 LLM 打分，可使用 `数据集/代码/score_all_vllm_local.py`。

原始代码保持不变，此处省略。

---

### Step 3：`composite_evidence_score` 规则计算 (`src/data/composite_scorer.py`)

```python
"""
composite_scorer.py — 规则计算 composite_evidence_score

公式：
  score = 0.35 * symptom_strength
        + 0.20 * crisis_norm
        + 0.20 * has_anchor
        + 0.10 * duration_support
        + 0.10 * confidence
        + 0.05 * self_disclosure

其中：
  symptom_strength = 0.6 * max(sv)/3 + 0.4 * count(sv>0)/9

阈值：
  >= 0.3 → 合格证据帖
  >= 0.5 → 高风险候选帖
"""
import math


def compute_composite_evidence_score(post: dict) -> float:
    """
    对单条已打分帖子计算 composite_evidence_score。
    
    Args:
        post: LLM 打分后的帖子 dict（已展平 score 字段）
    
    Returns:
        score: float, 范围 [0, 1]
    """
    sv = post.get("symptom_vector", {})
    vals = list(sv.values())
    if not vals:
        return 0.0
    
    max_symptom = max(vals) / 3.0
    symptom_coverage = sum(1 for v in vals if v > 0) / 9.0
    symptom_strength = 0.6 * max_symptom + 0.4 * symptom_coverage

    crisis_norm = post.get("crisis_level", 0) / 3.0
    has_anchor = 1.0 if len(post.get("clinical_context", {}).get("anchor_types", [])) > 0 else 0.0
    duration_support = 1.0 if post.get("duration", {}).get("has_hint", False) else 0.0
    confidence = post.get("confidence", 0.0)

    # 自述加分：first_person + literal_self_evidence 同时为 True
    self_disclosure = 1.0 if (post.get("first_person", False) and
                               post.get("literal_self_evidence", False)) else 0.0

    score = (0.35 * symptom_strength
             + 0.20 * crisis_norm
             + 0.20 * has_anchor
             + 0.10 * duration_support
             + 0.10 * confidence
             + 0.05 * self_disclosure)
    
    return round(score, 4)


def add_composite_scores(scored_posts: list) -> list:
    """
    为全部已打分帖子添加 composite_evidence_score 字段。
    """
    for post in scored_posts:
        post["composite_evidence_score"] = compute_composite_evidence_score(post)
    return scored_posts


def compute_dynamic_K(total_posts: int) -> int:
    """
    动态计算 risk_posts 的 K 值。
    
    规则：
    - total_posts >= 160: K = ceil(total_posts * 0.125)  → 12.5%
    - 20 <= total_posts < 160: K = 20                    → 保底 20
    - total_posts < 20: K = total_posts                  → 全部保留
    """
    if total_posts >= 160:
        return math.ceil(total_posts * 0.125)
    elif total_posts >= 20:
        return 20
    else:
        return total_posts


def get_risk_posts_a(user_scored_posts: list) -> list:
    """
    A 套 risk_posts：按 composite_evidence_score 降序取 top-K（动态）。
    
    Args:
        user_scored_posts: 该用户的全部已打分帖子
    
    Returns:
        risk_posts_a: List[dict], 每条含 post_id / text / composite_evidence_score
    """
    K = compute_dynamic_K(len(user_scored_posts))
    sorted_posts = sorted(user_scored_posts,
                          key=lambda p: p.get("composite_evidence_score", 0),
                          reverse=True)
    return sorted_posts[:K]
```

---

### Step 4：PHQ-9 模板筛选器 (`src/data/template_screener.py`)

> **执行范围**：所有用户，训练 + 推理均执行

```python
"""
template_screener.py — 通路 B：PHQ-9 模板筛选

使用 gte-small / all-MiniLM-L6-v2 编码，计算帖子与 PHQ-9 模板的余弦相似度，
取 risk_score top-K（动态）帖子作为 B 套 risk_posts。
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List
from src.data.composite_scorer import compute_dynamic_K


# ====== PHQ-9 九维症状描述模板 ======
PHQ9_TEMPLATES_ZH = {
    "depressed_mood": [
        "我感到心情低落、沮丧、绝望",
        "我每天都很难过，觉得生活没有意义",
        "心里特别压抑，什么都不想做"
    ],
    "anhedonia": [
        "我对什么事情都提不起兴趣",
        "以前喜欢的事情现在完全不想做了",
        "对任何事都没有快乐的感觉"
    ],
    "sleep": [
        "我失眠了很久都睡不着",
        "每天晚上翻来覆去睡不着觉",
        "睡眠质量很差经常半夜醒来"
    ],
    "fatigue": [
        "我整天都觉得很累没有力气",
        "做什么事都提不起精神来",
        "感觉身体被掏空了一样疲惫"
    ],
    "appetite_or_weight": [
        "我最近完全没有胃口吃不下东西",
        "体重变化很大不是暴食就是不吃",
        "食欲很差看到食物就想吐"
    ],
    "worthlessness_or_guilt": [
        "我觉得自己一无是处是个废物",
        "总觉得自己是个失败者对不起所有人",
        "强烈的自责感觉自己什么都做不好"
    ],
    "concentration": [
        "我无法集中注意力做任何事情",
        "脑子里一片空白什么都想不起来",
        "注意力完全无法集中"
    ],
    "psychomotor": [
        "我变得很迟钝反应也慢了",
        "说话和做事都变得非常缓慢",
        "坐立不安总是很烦躁"
    ],
    "suicidal_ideation": [
        "我不想活了觉得死了算了",
        "反复想到死亡或者伤害自己",
        "活着太痛苦了不如死了好"
    ]
}

PHQ9_TEMPLATES_EN = {
    "depressed_mood": [
        "Feeling down, depressed, or hopeless",
        "I feel so sad and empty every day",
        "Life feels meaningless and I can't stop feeling depressed"
    ],
    "anhedonia": [
        "Little interest or pleasure in doing things",
        "I don't enjoy anything anymore, nothing makes me happy",
        "Lost all motivation and interest in activities I used to love"
    ],
    "sleep": [
        "Trouble falling asleep, staying asleep, or sleeping too much",
        "I can't sleep at night, insomnia is ruining my life",
        "Waking up at 3am every night and can't go back to sleep"
    ],
    "fatigue": [
        "Feeling tired or having little energy",
        "I'm exhausted all the time, no energy to do anything",
        "So drained and fatigued I can barely get out of bed"
    ],
    "appetite_or_weight": [
        "Poor appetite or overeating",
        "I've lost my appetite completely, can't eat anything",
        "Stress eating or not eating at all, weight changing drastically"
    ],
    "worthlessness_or_guilt": [
        "Feeling bad about yourself, or that you are a failure",
        "I'm worthless and useless, everything is my fault",
        "Overwhelming guilt and self-hatred"
    ],
    "concentration": [
        "Trouble concentrating on things",
        "Can't focus on anything, my mind goes blank",
        "Unable to concentrate or make simple decisions"
    ],
    "psychomotor": [
        "Moving or speaking slowly, or being fidgety and restless",
        "I've become so slow, everything takes forever",
        "Can't sit still, constantly agitated and restless"
    ],
    "suicidal_ideation": [
        "Thoughts that you would be better off dead or hurting yourself",
        "I don't want to live anymore, thinking about ending it",
        "Recurring thoughts of death and self-harm"
    ]
}


class PHQ9TemplateScreener:
    """
    PHQ-9 模板筛选器。
    
    初始化时对 PHQ-9 模板进行编码（一次性），
    之后对每个用户的帖子执行筛选。
    """
    
    def __init__(self, model_name: str = "thenlper/gte-small-zh",
                 language: str = "zh"):
        """
        Args:
            model_name: Sentence-BERT 模型名
                中文: "thenlper/gte-small-zh" 或 "BAAI/bge-small-zh"
                英文: "sentence-transformers/all-MiniLM-L6-v2"
            language: "zh" 或 "en"
        """
        self.encoder = SentenceTransformer(model_name)
        self.templates = PHQ9_TEMPLATES_ZH if language == "zh" else PHQ9_TEMPLATES_EN
        
        # 预编码所有模板（一次性）
        self.template_embeddings = {}
        for dim, texts in self.templates.items():
            self.template_embeddings[dim] = self.encoder.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            )  # shape: [n_variants, embed_dim]
    
    def screen_user(self, user_posts: List[dict]) -> List[dict]:
        """
        对单个用户的所有帖子执行 PHQ-9 模板筛选。
        K 值动态计算。
        
        Args:
            user_posts: List[dict], 每条含 "post_id", "text"
        
        Returns:
            risk_posts: List[dict], top-K 条, 每条含:
                - post_id, text, risk_score, matched_dimensions, dim_scores
        """
        if not user_posts:
            return []
        
        K = compute_dynamic_K(len(user_posts))
        
        texts = [p["text"] for p in user_posts]
        post_embs = self.encoder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )  # shape: [N, embed_dim]
        
        results = []
        for i, p_emb in enumerate(post_embs):
            p_emb_2d = p_emb.reshape(1, -1)  # [1, d]
            
            dim_scores = {}
            for dim, t_embs in self.template_embeddings.items():
                # [1, d] vs [n_variants, d] → [1, n_variants] → max
                sims = cosine_similarity(p_emb_2d, t_embs)[0]
                dim_scores[dim] = float(np.max(sims))
            
            # risk_score = 0.6 * max_dim_score + 0.4 * mean(top-2)
            sorted_scores = sorted(dim_scores.values(), reverse=True)
            max_dim_score = sorted_scores[0]
            top2_avg = np.mean(sorted_scores[:2])
            risk_score = 0.6 * max_dim_score + 0.4 * top2_avg
            
            # 匹配的维度（相似度 >= 0.5）
            matched_dims = [d for d, s in dim_scores.items() if s >= 0.5]
            
            results.append({
                "post_id": user_posts[i]["post_id"],
                "text": user_posts[i]["text"],
                "posting_time": user_posts[i].get("posting_time"),
                "risk_score": round(float(risk_score), 4),
                "matched_dimensions": matched_dims,
                "dim_scores": {d: round(s, 4) for d, s in dim_scores.items()}
            })
        
        # 按 risk_score 降序排序，取 top-K
        results.sort(key=lambda x: x["risk_score"], reverse=True)
        return results[:K]
    
    def screen_all_users(self, all_user_posts: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
        """
        对所有用户执行筛选。
        
        Args:
            all_user_posts: { user_id: [post_dict, ...] }
        
        Returns:
            all_risk_posts: { user_id: [risk_post_dict, ...] }
        """
        from tqdm import tqdm
        all_risk_posts = {}
        for uid, posts in tqdm(all_user_posts.items(), desc="PHQ-9 模板筛选"):
            all_risk_posts[uid] = self.screen_user(posts)
        return all_risk_posts
```

---

### Step 5：脚本入口

#### `scripts/run_template_screening.py`

```python
"""
执行 PHQ-9 模板筛选并写出统一的 processed 产物。

用法：
  python scripts/run_template_screening.py \
    --dataset_name swdd \
    --scored_path dataset/weibo/SWDD/score_data/depressed.scored.jsonl \
    --cleaned_path dataset/weibo/SWDD/cleaned_data/control.cleaned.jsonl \
    --output_dir data/processed/swdd \
    --encoder_model thenlper/gte-small-zh \
    --language zh
"""
import argparse, json
from src.data.raw_loader import load_dataset, generate_splits
from src.data.composite_scorer import add_composite_scores, get_risk_posts_a
from src.data.template_screener import PHQ9TemplateScreener
from pathlib import Path


def main(args):
    # 1. 加载原始数据
    depressed_users, control_users = load_dataset(args.scored_path, args.cleaned_path)
    all_users = depressed_users + control_users
    print(f"Depressed 用户: {len(depressed_users)}, Control 用户: {len(control_users)}")
    print(f"总帖子数: {sum(len(u['posts']) for u in all_users)}")
    
    # 2. 数据划分
    if args.cv_folds > 0:
        from src.data.raw_loader import generate_cv_folds
        folds = generate_cv_folds(all_users, n_folds=args.cv_folds, seed=42)
        splits_data = {"folds": folds}
    else:
        splits = generate_splits(all_users, seed=42)
        splits_data = splits
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2a. 写出统一的标准化用户文件（02 的唯一用户级输入）
    with open(output_dir / "all_users_standardized.jsonl", 'w', encoding='utf-8') as f:
        for user in all_users:
            f.write(json.dumps(user, ensure_ascii=False) + '\n')
    print(f"标准化用户文件已保存: {output_dir / 'all_users_standardized.jsonl'}")

    with open(output_dir / "splits.json", 'w', encoding='utf-8') as f:
        json.dump(splits_data, f, ensure_ascii=False, indent=2)
    print(f"数据划分已保存: {output_dir / 'splits.json'}")
    
    # 3. 对 depressed 用户计算 composite_evidence_score
    all_risk_a = {}
    scored_output = output_dir / "depressed_scored_posts.jsonl"
    with open(scored_output, 'w', encoding='utf-8') as f:
        for user in depressed_users:
            posts = user["posts"]
            posts = add_composite_scores(posts)
            
            # A 套 risk_posts（动态 K）
            risk_a = get_risk_posts_a(posts)
            all_risk_a[user["user_id"]] = [
                {"post_id": p["post_id"],
                 "text": p["text"],
                 "composite_evidence_score": p["composite_evidence_score"],
                 "crisis_level": p.get("crisis_level", 0),
                 "temporality": p.get("temporality", "unclear")}
                for p in risk_a
            ]
            
            # 写出每个帖子
            for p in posts:
                f.write(json.dumps(p, ensure_ascii=False) + '\n')
    
    with open(output_dir / "risk_posts_a.json", 'w', encoding='utf-8') as f:
        json.dump(all_risk_a, f, ensure_ascii=False, indent=2)
    print(f"A 套 risk_posts 已保存")
    
    # 4. PHQ-9 模板筛选（所有用户）
    screener = PHQ9TemplateScreener(
        model_name=args.encoder_model,
        language=args.language
    )
    
    all_user_posts = {u["user_id"]: u["posts"] for u in all_users}
    all_risk_b = screener.screen_all_users(all_user_posts)
    
    with open(output_dir / "risk_posts_b.json", 'w', encoding='utf-8') as f:
        json.dump(all_risk_b, f, ensure_ascii=False, indent=2)
    
    print(f"B 套 risk_posts 已保存，完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, help="数据集名称，如 swdd / twitter / erisk")
    parser.add_argument("--scored_path", required=True, help="原始 scored JSONL 路径")
    parser.add_argument("--cleaned_path", required=True, help="原始 cleaned JSONL 路径")
    parser.add_argument("--output_dir", required=True, help="输出目录，如 data/processed/swdd")
    parser.add_argument("--encoder_model", default="thenlper/gte-small-zh")
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    parser.add_argument("--cv_folds", type=int, default=0, help=">0 时使用 k-fold CV (eRisk)")
    args = parser.parse_args()
    main(args)
```

---

## 注意事项

1. **LLM 打分已离线完成**，`depressed.scored.jsonl` 已包含 `score` 嵌套字段，本模块直接加载
2. **字段标准化在 `raw_loader.py` 中统一执行**：`label(str)→int`，`gender("None")→None`，`tweet_is_original(str/bool)→bool`
3. **01 的正式交付产物统一落在 `data/processed/{dataset}/...`**，供 `02` 和 `04` 直接消费
3. **K 值动态计算**：≥160帖取12.5%，20~159帖取20，<20帖全部保留
4. **PHQ-9 模板编码仅执行一次**（初始化时），后续仅计算余弦相似度
5. 推理时**仅需通路 B**（模板筛选），通路 A 不参与
6. `composite_evidence_score` 是纯规则计算，不经 LLM
7. **数据划分**：SWDD/Twitter 使用 80/10/10 用户级分层采样（seed=42），eRisk 使用 5-fold CV
