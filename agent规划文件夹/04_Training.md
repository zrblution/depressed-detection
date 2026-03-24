# 04 — 训练流程 Agent：多阶段训练 + 多层 Dropout + 损失函数

---

## Agent 目标

实现 WPG-MoE 系统的完整多阶段训练流程：

1. **阶段 C**：帖子编码器预训练（默认 backbone = `microsoft/mdeberta-v3-base`, MSE on composite_score）
2. **阶段 D**：专家 Warm-Start（按弱先验分组初始化）
3. **阶段 E**：联合训练（全损失函数 + 多层 Dropout 数据增强）

以及训练所需的所有支撑组件：Dataset、损失函数、学习率调度。

> **主线 Backbone 说明**：当前训练主线默认使用统一多语言编码器 `microsoft/mdeberta-v3-base`。若做单语上界实验，可额外配置 `MacBERT` 或 `DeBERTa-v3-base` 作为对照。

---

## 输入与输出

### 输入
| 文件 | 来源 | 说明 |
|---|---|---|
| `data/user_samples/{dataset}_train.jsonl` | 02 输出 | 用户级训练样本 |
| `data/user_samples/{dataset}_val.jsonl` | 02 输出 | 用户级验证样本 |
| `data/processed/{dataset}/depressed_scored_posts.jsonl` | 01 通路 A | depressed 全量打分帖子（阶段 C 用） |

### 输出
| 文件 | 说明 |
|---|---|
| `data/models/{dataset}_encoder_pretrained.pt` | 阶段 C 预训练编码器权重 |
| `data/models/{dataset}_warmstart_experts.pt` | 阶段 D warm-start 专家权重 |
| `data/models/{dataset}_final_model.pt` | 阶段 E 最终训练模型权重 |
| `data/models/{dataset}_training_log.json` | 训练日志（loss 曲线、F1 等） |

---

## 推荐技术栈

| 组件 | 推荐库 |
|---|---|
| 训练循环 | `torch`, `torch.utils.data` |
| 参数高效微调（可选） | `peft` |
| 优化器 | `torch.optim.AdamW` |
| 学习率调度 | `torch.optim.lr_scheduler` |
| 混合精度 | `torch.amp` |
| 日志 | `wandb` 或 `tensorboard` |

---

## Step-by-Step 实现

### Step 1：用户级 Dataset（含多层 Dropout）(`src/training/dataset.py`)

> **关键**：多层 Dropout 在 Dataset 的 `__getitem__` 中执行，属于数据增强层面。

```python
"""
dataset.py — 用户级 Dataset

核心特性：
1. 每个样本 = 一个用户
2. 训练时对 depressed 用户执行四层 Dropout 数据增强
3. 动态切换 A/B 套 risk_posts（Risk Source Swap）
"""
import json
import random
import torch
from torch.utils.data import Dataset
from typing import List, Dict


class UserDataset(Dataset):
    """
    用户级训练 Dataset。
    
    每个 __getitem__ 返回一个用户的完整输入，
    包含多层 Dropout 数据增强。
    """
    
    def __init__(self,
                 data_path: str,
                 is_training: bool = True,
                 p_risk_swap: float = 0.5,     # Risk Source Swap 概率
                 p_meta_drop: float = 0.5,     # META Dropout 概率（仅 A 套）
                 p_block_drop: float = 0.4,    # Episode Block Dropout 概率
                 p_prior_drop: float = 0.3,    # Prior Dropout 概率
                 p_post_drop: float = 0.3,     # 随机丢弃 30% risk_posts
                 max_risk_posts: int | None = None):
        """
        Args:
            data_path: 用户级样本 JSONL 文件路径
            is_training: 是否训练模式（控制 Dropout 是否启用）
            p_risk_swap: Risk Source Swap 概率
            p_meta_drop: META Dropout 概率
            p_block_drop: Episode Block Dropout 概率
            p_prior_drop: Prior Dropout 概率
            p_post_drop: 随机丢弃 risk_posts 的概率
            max_risk_posts: 可选上限；默认保持 01 动态 K 的完整输出
        """
        self.is_training = is_training
        self.p_risk_swap = p_risk_swap
        self.p_meta_drop = p_meta_drop
        self.p_block_drop = p_block_drop
        self.p_prior_drop = p_prior_drop
        self.p_post_drop = p_post_drop
        self.max_risk_posts = max_risk_posts
        
        # 加载数据
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> dict:
        sample = self.samples[idx]
        is_depressed = int(sample["label"]) == 1
        
        # ======== 多层 Dropout 数据增强 ========
        
        # --- Layer 1: Risk Source Swap ---
        # 决定使用 A 套还是 B 套 risk_posts
        has_meta = False
        if self.is_training and is_depressed and sample.get("risk_posts_llm"):
            if random.random() < self.p_risk_swap:
                # 使用 B 套（模板筛选，推理质量）
                risk_posts = sample["risk_posts_template"]
                has_meta = False
            else:
                # 使用 A 套（LLM 高质量）
                risk_posts = sample["risk_posts_llm"]
                has_meta = True
        else:
            # Template-only 用户 / 推理 / 无 LLM 数据：始终 B 套
            risk_posts = sample.get("risk_posts_template", [])
            has_meta = False
        
        # --- Layer 2: META Dropout ---
        # 仅在使用 A 套时生效
        if has_meta and self.is_training and random.random() < self.p_meta_drop:
            has_meta = False  # 去掉 META 标签
        
        # --- Layer 3: Episode Block Dropout ---
        episode_blocks = sample.get("episode_blocks", [])
        if self.is_training and is_depressed and random.random() < self.p_block_drop:
            episode_blocks = []  # 整体清空
        
        # --- Layer 4: Prior Dropout ---
        priors = sample.get("priors", {"self_disclosure": 0, "episode_supported": 0, "sparse_evidence": 0})
        crisis_score = sample.get("crisis_score", 0)
        if self.is_training and is_depressed and random.random() < self.p_prior_drop:
            priors = {"self_disclosure": 0.0, "episode_supported": 0.0, "sparse_evidence": 0.0}
            crisis_score = 0
        
        # --- 随机帖子丢弃（仅训练时） ---
        if self.is_training and random.random() < self.p_post_drop:
            keep_count = max(int(len(risk_posts) * 0.7), 3)  # 至少保留 3 条
            risk_posts = random.sample(risk_posts, min(keep_count, len(risk_posts)))
        
        # ======== 构造帖子文本 ========
        
        risk_texts = []
        risk_markers = []
        risk_composite_scores = []  # 用于 L_evidence 计算
        
        selected_risk_posts = (
            risk_posts if self.max_risk_posts is None else risk_posts[:self.max_risk_posts]
        )
        
        for i, rp in enumerate(selected_risk_posts):
            marker = f"[POST_{i+1:02d}]"
            text = rp["text"]
            
            if has_meta and "composite_evidence_score" in rp:
                # A 套 + 有 META：附加结构化标签
                meta_fields = {
                    "symptom_strength": f"{rp.get('composite_evidence_score', 0.0):.2f}",
                    "crisis": rp.get("crisis_level", 0),
                    "temporality": rp.get("temporality", "unclear")
                }
                meta_str = " ".join(f"{k}={v}" for k, v in meta_fields.items())
                formatted = f"{marker} {text} [META] {meta_str}"
            else:
                formatted = f"{marker} {text}"
            
            risk_texts.append(formatted)
            risk_markers.append(marker)
            
            # composite_score 作为 evidence silver label
            cs = rp.get("composite_evidence_score", 0.0)
            risk_composite_scores.append(cs)
        
        # Episode block 帖子
        block_texts = []
        block_markers = []
        for block in episode_blocks[:3]:
            for j, bp in enumerate(block.get("representative_posts", [])[:3]):
                marker = f"[POST_{len(risk_markers) + len(block_markers) + 1:02d}]"
                block_texts.append(f"{marker} {bp['text']}")
                block_markers.append(marker)
        
        # Global history
        global_segment_texts = []
        global_segment_markers = []
        for seg in sample.get("global_history_posts", []):
            seg_texts = []
            seg_markers = []
            for k, gp in enumerate(seg):
                marker = f"[POST_{len(risk_markers) + len(block_markers) + len(global_segment_markers)*4 + k + 1:02d}]"
                seg_texts.append(f"{marker} {gp['text']}")
                seg_markers.append(marker)
            global_segment_texts.append(seg_texts)
            global_segment_markers.append(seg_markers)
        
        # ======== 构造返回值 ========
        
        pi_u = torch.tensor([
            priors["self_disclosure"],
            priors["episode_supported"],
            priors["sparse_evidence"]
        ], dtype=torch.float32)
        
        crisis_tensor = torch.tensor([crisis_score / 3.0], dtype=torch.float32)
        
        stats = sample.get("global_stats", {})
        stats_tensor = torch.tensor([
            stats.get("posting_freq", 0),
            stats.get("temporal_burstiness", 0),
            0.0,  # avg_sentiment_trend (placeholder)
            min(stats.get("total_posts", 0) / 1000.0, 1.0),
            min(stats.get("active_span_days", 0) / 365.0, 1.0)
        ], dtype=torch.float32)
        
        return {
            "user_id": sample["user_id"],
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "risk_texts": risk_texts,
            "risk_markers": risk_markers,
            "risk_composite_scores": torch.tensor(risk_composite_scores, dtype=torch.float32),
            "block_texts": block_texts,
            "block_markers": block_markers,
            "global_segment_texts": global_segment_texts,
            "global_segment_markers": global_segment_markers,
            "pi_u": pi_u,
            "crisis": crisis_tensor,
            "stats": stats_tensor,
            "is_depressed": is_depressed
        }
```

---

### Step 2：损失函数 (`src/training/losses.py`)

```python
"""
losses.py — 全部损失函数实现

L = L_cls + α·L_route + β·L_evidence + γ·L_balance + δ·L_entropy

默认权重: α=0.3, β=0.2, γ=0.15, δ=0.1→0.02(cosine decay)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ClassificationLoss(nn.Module):
    """
    L_cls: 用户级二分类 BCE 损失。
    支持 class-weighted loss 解决不平衡问题。
    """
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
    
    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.bce(logit.squeeze(), label)


class RoutingLoss(nn.Module):
    """
    L_route: 弱先验引导的路由损失。
    
    KL(normalize(g_{1:3}) || normalize([p_sd, p_ep, p_sp]))
    
    仅在高置信用户上启用：
    - max(p) >= 0.6
    - max(p) - second_max(p) >= 0.1
    """
    def __init__(self, min_confidence: float = 0.6, min_gap: float = 0.1):
        super().__init__()
        self.min_confidence = min_confidence
        self.min_gap = min_gap
    
    def forward(self,
                gate_weights: torch.Tensor,   # [5] (batch 中单个样本)
                pi_u: torch.Tensor            # [3] (p_sd, p_ep, p_sp)
                ) -> torch.Tensor:
        """
        仅约束 gate 前 3 维。
        
        Returns:
            loss: scalar, 如果不满足启用条件则返回 0
        """
        # 检查启用条件
        p_max = pi_u.max()
        sorted_p = pi_u.sort(descending=True).values
        gap = sorted_p[0] - sorted_p[1] if len(sorted_p) > 1 else sorted_p[0]
        
        if p_max < self.min_confidence or gap < self.min_gap:
            return torch.tensor(0.0, device=gate_weights.device, requires_grad=True)
        
        # 提取 gate 前 3 维并归一化
        g_prior = gate_weights[:3]
        g_prior_norm = g_prior / (g_prior.sum() + 1e-8)
        
        # 归一化弱先验
        pi_norm = pi_u / (pi_u.sum() + 1e-8)
        
        # KL 散度（pi_norm 作为 target 分布）
        loss = F.kl_div(
            torch.log(g_prior_norm + 1e-8),
            pi_norm,
            reduction='sum'
        )
        
        return loss


class EvidenceLoss(nn.Module):
    """
    L_evidence: 证据帖子选择损失。
    
    L_evidence = Σ_i BCE(ŝ_i, σ(composite_evidence_score_i))
    
    仅对 depressed 用户计算。
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction='mean')
    
    def forward(self,
                evidence_scores: torch.Tensor,     # [K], 模型预测的证据分数（已 sigmoid）
                composite_scores: torch.Tensor,     # [K], composite_evidence_score
                is_depressed: bool
                ) -> torch.Tensor:
        if not is_depressed:
            return torch.tensor(0.0, device=evidence_scores.device, requires_grad=True)
        
        # Silver label: sigmoid(composite_evidence_score)
        silver_labels = torch.sigmoid(composite_scores)
        
        return self.bce(evidence_scores, silver_labels)


class BalanceLoss(nn.Module):
    """
    L_balance: 双项负载均衡损失 (Importance + Load)。
    
    L_importance = K · Σ f_k² (f_k = batch 内第 k 个专家的平均权重)
    
    L_load = K · Σ f_k · P_k
    其中 P_k = mean_batch(softmax(g_k / τ))，τ=0.1
    
    L_balance = L_importance + L_load
    
    注意: 此损失在 batch 级别计算，forward 接收整个 batch 的 gate 权重。
    """
    def __init__(self, num_experts: int = 5, tau: float = 0.1):
        super().__init__()
        self.K = num_experts
        self.tau = tau
    
    def forward(self, gate_weights_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate_weights_batch: [B, K], batch 内所有用户的 gate 权重
        
        Returns:
            L_balance: scalar
        """
        B, K = gate_weights_batch.shape
        
        # Importance loss: f_k = mean(g_k), L_imp = K · Σ f_k²
        f = gate_weights_batch.mean(dim=0)  # [K]
        L_importance = K * (f ** 2).sum()
        
        # Load loss: P_k = mean(softmax(g_k / τ))
        sharpened = F.softmax(gate_weights_batch / self.tau, dim=-1)  # [B, K]
        P = sharpened.mean(dim=0)  # [K]
        L_load = K * (f * P).sum()
        
        return L_importance + L_load


class EntropyLoss(nn.Module):
    """
    L_entropy: 单样本 gate 熵正则化。
    
    L_entropy = -mean_batch(Σ g_k · log(g_k + ε))
    
    防止 Dense MoE 退化为 Sparse MoE（gate 变成 one-hot）。
    δ 使用 cosine schedule 从 0.1 衰减至 0.02。
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, gate_weights_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate_weights_batch: [B, K]
        
        Returns:
            L_entropy: scalar (负熵，最小化此值 = 最大化熵)
        """
        entropy = -(gate_weights_batch * torch.log(gate_weights_batch + self.eps)).sum(dim=-1)
        return -entropy.mean()  # 负号：最小化 L_entropy = 最大化 gate 熵


class CombinedLoss(nn.Module):
    """
    总损失: L = L_cls + α·L_route + β·L_evidence + γ·L_balance + δ·L_entropy
    """
    def __init__(self,
                 alpha: float = 0.3,
                 beta: float = 0.2,
                 gamma: float = 0.15,
                 delta_init: float = 0.1,
                 delta_min: float = 0.02,
                 pos_weight: float = 1.0,
                 num_experts: int = 5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_init = delta_init
        self.delta_min = delta_min
        
        self.cls_loss = ClassificationLoss(pos_weight)
        self.route_loss = RoutingLoss()
        self.evidence_loss = EvidenceLoss()
        self.balance_loss = BalanceLoss(num_experts)
        self.entropy_loss = EntropyLoss()
    
    def get_delta(self, current_epoch: int, total_epochs: int) -> float:
        """
        δ 的 cosine decay schedule。
        
        δ = δ_min + 0.5 * (δ_init - δ_min) * (1 + cos(π * epoch / total_epochs))
        """
        progress = current_epoch / max(total_epochs, 1)
        delta = self.delta_min + 0.5 * (self.delta_init - self.delta_min) * \
                (1 + math.cos(math.pi * progress))
        return delta
    
    def forward(self,
                logits: torch.Tensor,              # [B]
                labels: torch.Tensor,              # [B]
                gate_weights_batch: torch.Tensor,  # [B, K]
                pi_u_batch: torch.Tensor,          # [B, 3]
                evidence_scores_batch: list,       # List[Tensor], 每个 [K_i]
                composite_scores_batch: list,      # List[Tensor], 每个 [K_i]
                is_depressed_batch: list,          # List[bool]
                current_epoch: int = 0,
                total_epochs: int = 30
                ) -> dict:
        """
        Returns:
            {
                "total": scalar,
                "cls": scalar,
                "route": scalar,
                "evidence": scalar,
                "balance": scalar,
                "entropy": scalar
            }
        """
        B = logits.shape[0]
        
        # L_cls (batch-level)
        l_cls = self.cls_loss(logits, labels)
        
        # L_route (sample-level, 求平均)
        l_route = torch.tensor(0.0, device=logits.device, requires_grad=True)
        route_count = 0
        for i in range(B):
            r = self.route_loss(gate_weights_batch[i], pi_u_batch[i])
            if r.item() > 0:
                l_route = l_route + r
                route_count += 1
        if route_count > 0:
            l_route = l_route / route_count
        
        # L_evidence (sample-level, 求平均)
        l_evidence = torch.tensor(0.0, device=logits.device, requires_grad=True)
        evidence_count = 0
        for i in range(B):
            e = self.evidence_loss(
                evidence_scores_batch[i],
                composite_scores_batch[i],
                is_depressed_batch[i]
            )
            if e.item() > 0:
                l_evidence = l_evidence + e
                evidence_count += 1
        if evidence_count > 0:
            l_evidence = l_evidence / evidence_count
        
        # L_balance (batch-level)
        l_balance = self.balance_loss(gate_weights_batch)
        
        # L_entropy (batch-level)
        l_entropy = self.entropy_loss(gate_weights_batch)
        
        # δ cosine decay
        delta = self.get_delta(current_epoch, total_epochs)
        
        # 总损失
        total = (l_cls
                 + self.alpha * l_route
                 + self.beta * l_evidence
                 + self.gamma * l_balance
                 + delta * l_entropy)
        
        return {
            "total": total,
            "cls": l_cls.detach(),
            "route": l_route.detach(),
            "evidence": l_evidence.detach(),
            "balance": l_balance.detach(),
            "entropy": l_entropy.detach(),
            "delta": delta
        }
```

---

### Step 3：阶段 C — 编码器预训练 (`src/training/encoder_pretrain.py`)

```python
"""
encoder_pretrain.py — 阶段 C：帖子编码器预训练

任务: 给定帖子文本（含或不含完整 META），预测 composite_evidence_score (MSE 回归)
数据: depressed 用户全量打分帖子（~84 万条）
META Dropout: 50% 概率去掉 [META] 标签
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json, random


class PostScoringDataset(Dataset):
    """帖子级评分预测 Dataset"""
    
    def __init__(self, scored_posts_path: str, p_meta_drop: float = 0.5):
        self.posts = []
        with open(scored_posts_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.posts.append(json.loads(line.strip()))
        self.p_meta_drop = p_meta_drop
    
    def __len__(self):
        return len(self.posts)
    
    def __getitem__(self, idx):
        post = self.posts[idx]
        text = post["text"]
        score = post.get("composite_evidence_score", 0.0)
        
        # META Dropout: 50% 概率附加 META 标签
        if random.random() >= self.p_meta_drop:
            # 附加 META
            meta_fields = {
                "symptom_strength": f"{score:.2f}",
                "crisis": post.get("crisis_level", 0),
                "temporality": post.get("temporality", "unclear")
            }
            meta_str = " ".join(f"{k}={v}" for k, v in meta_fields.items())
            text = f"[POST_01] {text} [META] {meta_str}"
        else:
            text = f"[POST_01] {text}"
        
        return {"text": text, "score": torch.tensor(score, dtype=torch.float32)}


def pretrain_encoder(encoder, train_dataset, val_dataset, config):
    """
    阶段 C 训练循环。
    
    Args:
        encoder: PostEncoder 实例
        train_dataset: PostScoringDataset
        val_dataset: PostScoringDataset
        config: {
            "lr": 2e-5,
            "epochs": 5,
            "batch_size": 32,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01
        }
    """
    # 在 encoder 上添加回归头
    regressor = nn.Linear(encoder.hidden_dim, 1).to(encoder.device)
    
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(regressor.parameters()),
        lr=config.get("lr", 2e-5),
        weight_decay=config.get("weight_decay", 0.01)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 32),
                              shuffle=True, num_workers=4)
    
    total_steps = len(train_loader) * config.get("epochs", 5)
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.get("lr", 2e-5),
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps
    )
    
    mse_loss = nn.MSELoss()
    
    encoder.train()
    for epoch in range(config.get("epochs", 5)):
        total_loss = 0
        for batch in train_loader:
            texts = batch["text"]
            scores = batch["score"].to(encoder.device)
            
            # 编码
            markers = ["[POST_01]"] * len(texts)
            reps = encoder.encode_posts(list(texts), markers)  # [B, d]
            
            # 回归预测
            predicted = regressor(reps).squeeze(-1)  # [B]
            
            loss = mse_loss(predicted, scores)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"[阶段C] Epoch {epoch+1}: MSE Loss = {avg_loss:.4f}")
    
    return encoder  # 返回预训练后的编码器（不返回 regressor，它只是辅助任务）
```

---

### Step 4：阶段 D — 专家 Warm-Start (`src/training/warm_start.py`)

```python
"""
warm_start.py — 阶段 D：专家 Warm-Start

用通道对应的高置信用户子集分别预训练每个专家：
  Expert_SD: p_sd top-30% depressed 用户
  Expert_EP: p_ep top-30% depressed 用户
  Expert_SP: p_sp top-30% depressed 用户
  Expert_MIX: 全部 depressed 用户
  Expert_G:  全部用户（含 normal）
"""
import torch


def warm_start_experts(model, train_samples: list, config: dict):
    """
    对每个专家执行独立的短训练（3-5 epochs）。
    
    Args:
        model: WPGMoEModel 实例
        train_samples: 用户级训练样本列表
        config: 超参数
    """
    depressed_samples = [s for s in train_samples if int(s["label"]) == 1]
    
    # 按各通道先验排序，取 top-30%
    top_ratio = 0.30
    n_top = max(int(len(depressed_samples) * top_ratio), 1)
    
    expert_subsets = {
        0: sorted(depressed_samples,                               # Expert_SD
                   key=lambda s: s["priors"]["self_disclosure"],
                   reverse=True)[:n_top],
        1: sorted(depressed_samples,                               # Expert_EP
                   key=lambda s: s["priors"]["episode_supported"],
                   reverse=True)[:n_top],
        2: sorted(depressed_samples,                               # Expert_SP
                   key=lambda s: s["priors"]["sparse_evidence"],
                   reverse=True)[:n_top],
        3: depressed_samples,                                      # Expert_MIX
        4: train_samples                                           # Expert_G
    }
    
    for expert_idx, subset in expert_subsets.items():
        expert = model.experts.experts[expert_idx]
        print(f"[阶段D] Warm-starting Expert_{expert_idx} on {len(subset)} users...")
        
        # 冻结其他组件，仅训练当前专家
        for param in model.parameters():
            param.requires_grad = False
        for param in expert.parameters():
            param.requires_grad = True
        
        optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-4)
        
        # 短训练（3 epochs，使用简单的分类任务）
        for epoch in range(3):
            # ... 前向 → 仅算 L_cls → 反向 → 优化 expert 参数
            pass  # 具体实现参考 joint_trainer 的前向逻辑
        
        print(f"  Expert_{expert_idx} warm-start 完成")
    
    # 恢复所有参数可训练状态
    for param in model.parameters():
        param.requires_grad = True
```

---

### Step 5：阶段 E — 联合训练主循环 (`src/training/joint_trainer.py`)

```python
"""
joint_trainer.py — 阶段 E：联合训练

总损失: L = L_cls + 0.3·L_route + 0.2·L_evidence + 0.15·L_balance + 0.1→0.02·L_entropy

Optimizer: 
  - MoE 头部: AdamW, lr=1e-4
  - Backbone encoder: AdamW, lr=2e-5

数据增强: 多层 Dropout（在 Dataset 中执行）
"""
import torch
from torch.utils.data import DataLoader
from src.training.losses import CombinedLoss
from src.training.dataset import UserDataset


def train_joint(model, train_path, val_path, config):
    """
    联合训练主循环。
    
    Args:
        model: WPGMoEModel（阶段 C 预训练编码器 + 阶段 D warm-start 专家）
        train_path: 训练集 JSONL 路径
        val_path: 验证集 JSONL 路径
        config: 超参数字典
    """
    device = next(model.parameters()).device
    
    # Dataset
    train_dataset = UserDataset(train_path, is_training=True,
                                 p_risk_swap=config.get("p_risk_swap", 0.5),
                                 p_meta_drop=config.get("p_meta_drop", 0.5),
                                 p_block_drop=config.get("p_block_drop", 0.4),
                                 p_prior_drop=config.get("p_prior_drop", 0.3))
    val_dataset = UserDataset(val_path, is_training=False)
    
    # 注意: 用户级样本不适合标准 batch collate，使用 batch_size=1 逐用户处理
    # 或自定义 collate_fn 处理变长输入
    train_loader = DataLoader(train_dataset, batch_size=1,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=0)
    
    # --- 差异化学习率 ---
    # Backbone encoder: lr=2e-5
    # 其他参数 (MoE head, gate, evidence head): lr=1e-4
    encoder_params = list(model.encoder.parameters())
    head_params = [
        param for name, param in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": config.get("lr_head", 1e-4)},
        {"params": encoder_params, "lr": config.get("lr_encoder", 2e-5)}
    ], weight_decay=0.01)
    
    # 损失函数
    criterion = CombinedLoss(
        alpha=config.get("alpha", 0.3),
        beta=config.get("beta", 0.2),
        gamma=config.get("gamma", 0.15),
        delta_init=config.get("delta_init", 0.1),
        delta_min=config.get("delta_min", 0.02),
        pos_weight=config.get("pos_weight", 1.0),
        num_experts=config.get("num_experts", 5)
    )
    
    max_epochs = config.get("max_epochs", 30)
    patience = config.get("patience", 5)
    best_f1 = 0.0
    wait = 0
    
    for epoch in range(max_epochs):
        model.train()
        epoch_losses = {"total": 0, "cls": 0, "route": 0, "evidence": 0, "balance": 0, "entropy": 0}
        
        # --- Batch 累积（模拟 batch_size=16）---
        # 因为每个用户的输入是变长的，逐用户前向，每 16 个用户更新一次
        batch_size = config.get("batch_size", 16)
        accumulation_steps = batch_size
        
        all_logits = []
        all_labels = []
        all_gate_weights = []
        all_pi_u = []
        all_evidence_scores = []
        all_composite_scores = []
        all_is_depressed = []
        
        optimizer.zero_grad()
        
        for step, sample in enumerate(train_loader):
            # sample 是 batch_size=1 的 dict，解包
            output = model(
                risk_post_texts=sample["risk_texts"][0],       # 去掉 batch 维
                risk_post_markers=sample["risk_markers"][0],
                block_post_texts=sample["block_texts"][0],
                block_post_markers=sample["block_markers"][0],
                global_segment_texts=sample["global_segment_texts"][0],
                global_segment_markers=sample["global_segment_markers"][0],
                pi_u=sample["pi_u"].squeeze(0).to(device),
                crisis=sample["crisis"].squeeze(0).to(device),
                stats=sample["stats"].squeeze(0).to(device)
            )
            
            all_logits.append(output["logit"])
            all_labels.append(sample["label"].to(device))
            all_gate_weights.append(output["gate_weights"])
            all_pi_u.append(sample["pi_u"].squeeze(0).to(device))
            all_evidence_scores.append(output["evidence_scores"])
            all_composite_scores.append(sample["risk_composite_scores"].squeeze(0).to(device))
            all_is_depressed.append(sample["is_depressed"][0])
            
            # 每 accumulation_steps 计算一次 loss 并更新
            if (step + 1) % accumulation_steps == 0 or step == len(train_loader) - 1:
                batch_logits = torch.cat(all_logits, dim=0)
                batch_labels = torch.cat(all_labels, dim=0)
                batch_gate = torch.stack(all_gate_weights, dim=0)
                batch_pi = torch.stack(all_pi_u, dim=0)
                
                loss_dict = criterion(
                    batch_logits, batch_labels,
                    batch_gate, batch_pi,
                    all_evidence_scores, all_composite_scores,
                    all_is_depressed,
                    current_epoch=epoch, total_epochs=max_epochs
                )
                
                loss = loss_dict["total"] / accumulation_steps
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                for k in epoch_losses:
                    epoch_losses[k] += loss_dict.get(k, torch.tensor(0)).item()
                
                # 清空累积
                all_logits, all_labels = [], []
                all_gate_weights, all_pi_u = [], []
                all_evidence_scores, all_composite_scores = [], []
                all_is_depressed = []
        
        # --- 验证 ---
        val_f1 = evaluate(model, val_loader, device)
        
        print(f"[阶段E] Epoch {epoch+1}/{max_epochs} | "
              f"Loss={epoch_losses['total']:.4f} | "
              f"Val F1={val_f1:.4f} | "
              f"δ={criterion.get_delta(epoch, max_epochs):.4f}")
        
        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            torch.save(model.state_dict(), config["save_path"])
            print(f"  ✓ 最佳模型已保存 (F1={best_f1:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break


def evaluate(model, val_loader, device):
    """验证集评估，返回 F1 score"""
    model.eval()
    from sklearn.metrics import f1_score
    
    preds, golds = [], []
    with torch.no_grad():
        for sample in val_loader:
            output = model(
                risk_post_texts=sample["risk_texts"][0],
                risk_post_markers=sample["risk_markers"][0],
                block_post_texts=sample["block_texts"][0],
                block_post_markers=sample["block_markers"][0],
                global_segment_texts=sample["global_segment_texts"][0],
                global_segment_markers=sample["global_segment_markers"][0],
                pi_u=sample["pi_u"].squeeze(0).to(device),
                crisis=sample["crisis"].squeeze(0).to(device),
                stats=sample["stats"].squeeze(0).to(device)
            )
            pred = (torch.sigmoid(output["logit"]) > 0.5).int().item()
            preds.append(pred)
            golds.append(sample["label"].int().item())
    
    return f1_score(golds, preds, zero_division=0)
```

---

### Step 6：训练入口脚本 (`scripts/train.py`)

```python
"""
train.py — 完整训练入口

用法:
  python scripts/train.py --config configs/weibo.yaml
"""
import argparse, yaml, torch
from src.model.full_model import WPGMoEModel
from src.training.encoder_pretrain import pretrain_encoder, PostScoringDataset
from src.training.warm_start import warm_start_experts
from src.training.joint_trainer import train_joint


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = WPGMoEModel(config).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # --- 阶段 C: 编码器预训练 ---
    print("\n" + "="*50)
    print("阶段 C: 帖子编码器预训练")
    print("="*50)
    train_scored = PostScoringDataset(config["scored_posts_path"])
    pretrain_encoder(model.encoder, train_scored, train_scored, config)
    torch.save(model.encoder.state_dict(), config["encoder_save_path"])
    
    # --- 阶段 D: 专家 Warm-Start ---
    print("\n" + "="*50)
    print("阶段 D: 专家 Warm-Start")
    print("="*50)
    import json
    train_samples = []
    with open(config["train_path"], 'r') as f:
        for line in f:
            train_samples.append(json.loads(line))
    warm_start_experts(model, train_samples, config)
    
    # --- 阶段 E: 联合训练 ---
    print("\n" + "="*50)
    print("阶段 E: 联合训练")
    print("="*50)
    train_joint(model, config["train_path"], config["val_path"], config)
    
    print("\n训练完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    args = parser.parse_args()
    main(args)
```

---

## 多层 Dropout 位置总结

| Dropout 层 | 执行位置 | 代码文件 | 具体位置 |
|---|---|---|---|
| Risk Source Swap (p=0.5) | `Dataset.__getitem__` | `dataset.py` | Layer 1: 选择 A/B 套 |
| META Dropout (p=0.5) | `Dataset.__getitem__` | `dataset.py` | Layer 2: 去掉 `[META]` |
| Episode Block Dropout (p=0.4) | `Dataset.__getitem__` | `dataset.py` | Layer 3: `blocks=[]` |
| Prior Dropout (p=0.3) | `Dataset.__getitem__` | `dataset.py` | Layer 4: `pi_u=0, crisis=0` |
| Post Drop (p=0.3) | `Dataset.__getitem__` | `dataset.py` | 随机丢弃 30% risk_posts |
| META Dropout (阶段C) | `PostScoringDataset.__getitem__` | `encoder_pretrain.py` | 50% 概率去掉 META |

---

## 注意事项

1. **差异化学习率**：encoder 参数用 2e-5，MoE 头部用 1e-4；若后续改为 adapter / LoRA，再单独设更高 adapter lr
2. **δ cosine decay**：L_entropy 权重从 0.1 衰减到 0.02，训练后期允许 gate 更尖锐
3. **L_route 仅高置信用户**：max(p)≥0.6 且 gap≥0.1 时才启用，避免噪声先验误导路由
4. **梯度累积**模拟 batch_size=16（因变长输入逐用户前向）
5. **Early stopping on val F1**，patience=5
6. **数据划分**：Weibo/Twitter 80/10/10，eRisk 5-fold CV
