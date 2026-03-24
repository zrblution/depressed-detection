# 03 — 模型定义 Agent：编码器 + MoE + 门控 + Evidence Head

---

## Agent 目标

定义 WPG-MoE 系统的全部 PyTorch 模型组件：

1. **`PostEncoder`**：共享帖子编码器（BERT-family encoder）
2. **`UserRepresentationModule`**：5 路用户级表示构造（attention pooling）
3. **`GateNetwork`**：门控网络（2 层 MLP → softmax → 5 维权重）
4. **`ExpertNetwork`**：5 个专家 MLP（hidden=512, 2 层）
5. **`MoEHead`**：Dense MoE 融合 + 分类头
6. **`EvidenceHead`**：证据选择头
7. **`FullModel`**：完整模型封装

---

## 输入与输出

### 模型输入（forward 参数 + 旁路元数据）
| 参数 | 类型 | shape | 说明 |
|---|---|---|---|
| `risk_post_texts` | `List[str]` | `[K_dyn]` | 动态 K 条 risk_posts 文本 |
| `risk_post_markers` | `List[str]` | `[K_dyn]` | 编码器内部 marker，如 `[POST_01]` |
| `risk_post_ids` | `List[str]` | `[K_dyn]` | 与 `risk_post_texts` 对齐的标准化 `post_id`，如 `user_001__3`；用于输出映射，不进入编码器 |
| `episode_block_texts` | `List[List[str]]` | `[m, 3]` | m=3 个 block，各含 ≤3 条帖子 |
| `global_history_texts` | `List[List[str]]` | `[S, K_seg]` | S=8 段，各含 K_seg 条帖子 |
| `pi_u` | `Tensor` | `[3]` | `(p_sd, p_ep, p_sp)` |
| `crisis_score` | `int` | `scalar` | 危机分数 0-3 |
| `global_stats` | `Tensor` | `[5]` | 统计特征向量 |
| `has_meta` | `bool` | — | 是否附带 META 标签 |

### 模型输出
| 输出 | 类型 | shape | 说明 |
|---|---|---|---|
| `logit` | `Tensor` | `[1]` | 未经 sigmoid 的 raw logit |
| `gate_weights` | `Tensor` | `[5]` | 5 个专家的权重 |
| `evidence_scores` | `Tensor` | `[K_dyn]` | 每条 risk_post 的证据分数 |
| `expert_outputs` | `Tensor` | `[5, d_expert]` | 各专家输出（用于分析） |

---

## 推荐技术栈

| 组件 | 推荐库 | 说明 |
|---|---|---|
| Backbone | `transformers` (`AutoModel`) | 主线推荐 `microsoft/mdeberta-v3-base`；`MacBERT` / `DeBERTa-v3-base` 仅作单语对照 |
| 参数高效微调 | `peft`（可选） | 显存受限时再启用 adapter / LoRA，不作为主线默认 |
| 模型构建 | `torch.nn` | MLP, Attention, Sigmoid |
| Tokenizer | `transformers` (`AutoTokenizer`) | 对应 BERT-family tokenizer |

---

## Step-by-Step 实现

### Step 1：帖子编码器 (`src/model/post_encoder.py`)

```python
"""
post_encoder.py — 共享帖子编码器

Backbone: BERT-family encoder
推荐默认:
  - unified: microsoft/mdeberta-v3-base
  - optional baselines: hfl/chinese-macbert-base / microsoft/deberta-v3-base
输入: 帖子文本（可选附带 [META] 标签）
输出: 帖子级表示 h_i ∈ R^d

编码方式：
- 对帖子文本前缀 [POST_{id}] 标记
- 取最后一层 [POST_{id}] token 的隐状态作为帖子表示
- 若 marker 未命中，则 fallback 到 `[CLS]`
- 手动添加 `[POST_XX]` 和 `[META]` special tokens 以保持训练 / 推理合同一致
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class PostEncoder(nn.Module):
    """
    共享帖子编码器。
    
    初始化流程:
    1. 加载 BERT-family 预训练 encoder
    2. 添加 [POST_01]~[POST_XX] special tokens
    3. 添加 [META] special token
    4. 默认全量微调 encoder 参数
    """
    
    def __init__(self,
                 model_name: str = "microsoft/mdeberta-v3-base",
                 max_post_tokens: int = 256,
                 num_post_markers: int = 2048):
        super().__init__()
        
        self.max_post_tokens = max_post_tokens
        
        # 1. 加载 tokenizer + 模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # 2. 添加 POST marker special tokens
        post_tokens = [f"[POST_{i:02d}]" for i in range(1, num_post_markers + 1)]
        meta_tokens = ["[META]"]
        special_tokens = {"additional_special_tokens": post_tokens + meta_tokens}
        self.tokenizer.add_special_tokens(special_tokens)
        self.backbone.resize_token_embeddings(len(self.tokenizer))
        
        # 存储 POST token ids 用于后续提取隐状态
        self.post_token_ids = {
            tok: self.tokenizer.convert_tokens_to_ids(tok)
            for tok in post_tokens
        }
        
        # 隐藏维度
        self.hidden_dim = self.backbone.config.hidden_size
    
    @property
    def device(self):
        return next(self.backbone.parameters()).device
    
    def format_post_text(self, post_id_marker: str, text: str,
                         meta_info: dict = None) -> str:
        """
        格式化帖子输入文本。
        
        有 META 时: "[POST_01] 我好难过 [META] symptom_strength=0.7 crisis=2 temporality=current"
        无 META 时: "[POST_01] 我好难过"
        
        Args:
            post_id_marker: 如 "[POST_01]"
            text: 原始帖子文本
            meta_info: 可选, {"symptom_strength", "crisis", "temporality"}
        """
        formatted = f"{post_id_marker} {text}"
        if meta_info:
            meta_str = " ".join(f"{k}={v}" for k, v in meta_info.items())
            formatted += f" [META] {meta_str}"
        return formatted
    
    def encode_posts(self, formatted_texts: list, post_markers: list) -> torch.Tensor:
        """
        批量编码帖子，提取 POST marker token 的隐状态。
        
        Args:
            formatted_texts: 已格式化的帖子文本列表
            post_markers: 对应的 POST marker token 列表（如 ["[POST_01]", "[POST_02]", ...]）
        
        Returns:
            post_representations: Tensor [num_posts, hidden_dim]
        """
        # Tokenize（截断到 max_post_tokens）
        encodings = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=self.max_post_tokens,
            return_tensors="pt"
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        # Forward（encoder-only，无需 lm_head）
        with torch.set_grad_enabled(self.training):
            outputs = self.backbone(**encodings, return_dict=True)
        
        last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # 提取每条帖子对应 POST marker 位置的隐状态
        representations = []
        for i, marker in enumerate(post_markers):
            marker_id = self.post_token_ids[marker]
            # 找到 marker token 在序列中的位置
            token_ids = encodings["input_ids"][i]
            positions = (token_ids == marker_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                pos = positions[0].item()
                representations.append(last_hidden[i, pos, :])
            else:
                # fallback: 取 [CLS]
                representations.append(last_hidden[i, 0, :])
        
        return torch.stack(representations, dim=0)  # [num_posts, hidden_dim]
```

---

### Step 2：用户级表示构造 (`src/model/user_representation.py`)

```python
"""
user_representation.py — 五路用户级表示构造

z_sd: 自述披露流 — attention pooling（学到关注自述类帖子）
z_ep: 时间支持流 — episode_blocks 或 fallback 到 risk_posts
z_sp: 稀疏证据流 — top-3 risk_posts attention pooling
z_mix: 混合流     — 全部 risk_posts mean pooling
z_g:  全局历史   — 分段编码 + 时序注意力池化 + 统计特征拼接
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionPooling(nn.Module):
    """
    Learned attention pooling: z = Σ softmax(w^T · h_i) · h_i
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_weight = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, h: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            h: [num_items, hidden_dim]
            mask: [num_items], 1=有效, 0=padding
        Returns:
            z: [hidden_dim]
        """
        scores = self.attention_weight(h).squeeze(-1)  # [num_items]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=0)  # [num_items]
        z = (weights.unsqueeze(-1) * h).sum(dim=0)  # [hidden_dim]
        return z


class TemporalSelfAttention(nn.Module):
    """
    段间时序自注意力：用于 global_history 的 S 个段表示之间进行注意力融合。
    
    α_s = softmax(Q(c_s)^T · K([c_1,...,c_S]) / √d)
    z_g = Σ α_s · V(c_s)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)
    
    def forward(self, segment_reps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            segment_reps: [S, hidden_dim], S 段表示
        Returns:
            z_g: [hidden_dim], 融合后的全局历史表示
        """
        Q = self.Q(segment_reps)  # [S, d]
        K = self.K(segment_reps)  # [S, d]
        V = self.V(segment_reps)  # [S, d]
        
        attn_scores = torch.matmul(Q, K.T) / self.scale  # [S, S]
        attn_weights = F.softmax(attn_scores, dim=-1)     # [S, S]
        z_g = torch.matmul(attn_weights, V).mean(dim=0)   # [d]
        
        return z_g


class UserRepresentationModule(nn.Module):
    """
    构造 5 路用户级表示。
    
    初始化 3 个独立的 AttentionPooling head（sd, ep, sp）
    + 1 个 TemporalSelfAttention（global_history）
    + 1 个 stats 投影层
    """
    def __init__(self, hidden_dim: int, stats_dim: int = 5):
        """
        Args:
            hidden_dim: 编码器隐藏维度
            stats_dim: 全局统计特征维度
                (posting_freq, temporal_burstiness,
                 avg_sentiment_trend, total_post_count, active_span_days)
        """
        super().__init__()
        self.attn_sd = AttentionPooling(hidden_dim)
        self.attn_ep = AttentionPooling(hidden_dim)
        self.attn_sp = AttentionPooling(hidden_dim)
        self.temporal_attn = TemporalSelfAttention(hidden_dim)
        self.stats_proj = nn.Linear(stats_dim, hidden_dim)
    
    def forward(self,
                risk_post_reps: torch.Tensor,   # [K, d]
                block_post_reps: torch.Tensor,   # [M, d] 或 None（空 blocks）
                segment_reps: torch.Tensor,       # [S, d]
                global_stats: torch.Tensor        # [stats_dim]
                ) -> dict:
        """
        Returns:
            {
                "z_sd": [d], "z_ep": [d], "z_sp": [d],
                "z_mix": [d], "z_g": [d]
            }
        """
        # z_sd: 自述披露流 — attention pooling on risk_posts
        z_sd = self.attn_sd(risk_post_reps)
        
        # z_ep: 时间支持流
        if block_post_reps is not None and block_post_reps.shape[0] > 0:
            z_ep = self.attn_ep(block_post_reps)
        else:
            # fallback: 无 episode_blocks 时用 risk_posts
            z_ep = self.attn_ep(risk_post_reps)
        
        # z_sp: 稀疏证据流 — top-3 risk_posts attention pooling
        top3_reps = risk_post_reps[:3]  # 已按得分排序，取前 3
        z_sp = self.attn_sp(top3_reps)
        
        # z_mix: 混合流 — mean pooling
        z_mix = risk_post_reps.mean(dim=0)
        
        # z_g: 全局历史 — 时序注意力 + 统计特征拼接
        z_temporal = self.temporal_attn(segment_reps)
        z_stats = self.stats_proj(global_stats)
        z_g = z_temporal + z_stats  # 简单相加（也可用 concat + linear）
        
        return {
            "z_sd": z_sd, "z_ep": z_ep, "z_sp": z_sp,
            "z_mix": z_mix, "z_g": z_g
        }
```

---

### Step 3：门控网络 (`src/model/gate_network.py`)

```python
"""
gate_network.py — 门控网络

输入: [z_sd, z_ep, z_sp, z_mix, z_g, p_sd, p_ep, p_sp, crisis_score, stats...]
输出: g ∈ R^5, softmax 归一化的专家权重

结构: 两层 MLP, hidden=256
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GateNetwork(nn.Module):
    """
    门控网络：基于用户级证据状态向量输出 5 维专家权重。
    
    输入维度 = 5 * hidden_dim (五路表示) + 3 (弱先验) + 1 (crisis) + stats_dim
    输出维度 = 5 (专家数)
    """
    def __init__(self,
                 hidden_dim: int,
                 num_experts: int = 5,
                 gate_hidden: int = 256,
                 stats_dim: int = 5,
                 prior_dim: int = 4):
        """
        Args:
            hidden_dim: 编码器隐藏维度
            num_experts: 专家数量
            gate_hidden: 门控 MLP 隐藏层维度
            stats_dim: 统计特征维度
            prior_dim: 先验维度 (p_sd + p_ep + p_sp + crisis_score = 4)
        """
        super().__init__()
        
        input_dim = 5 * hidden_dim + prior_dim + stats_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gate_hidden, gate_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gate_hidden, num_experts)
        )
    
    def forward(self,
                z_dict: dict,            # {"z_sd", "z_ep", "z_sp", "z_mix", "z_g"}
                pi_u: torch.Tensor,      # [3]: (p_sd, p_ep, p_sp)
                crisis: torch.Tensor,    # [1]: crisis_score
                stats: torch.Tensor      # [stats_dim]
                ) -> torch.Tensor:
        """
        Returns:
            gate_weights: [num_experts], softmax 归一化
        """
        # 拼接所有输入
        gate_input = torch.cat([
            z_dict["z_sd"],
            z_dict["z_ep"],
            z_dict["z_sp"],
            z_dict["z_mix"],
            z_dict["z_g"],
            pi_u,                    # [3]
            crisis.unsqueeze(0) if crisis.dim() == 0 else crisis,  # [1]
            stats                    # [stats_dim]
        ], dim=0)
        
        logits = self.mlp(gate_input)          # [num_experts]
        gate_weights = F.softmax(logits, dim=0)  # [num_experts]
        
        return gate_weights
```

---

### Step 4：专家网络 (`src/model/expert_network.py`)

```python
"""
expert_network.py — 5 个专家 MLP

每个专家结构相同（2 层 MLP, hidden=512），但各自关注不同的输入:
  Expert_SD:  z_sd + 临床 meta
  Expert_EP:  z_ep + block 统计
  Expert_SP:  z_sp + top-k 单帖证据
  Expert_MIX: z_mix + 全部 meta
  Expert_G:   z_g + 全部 meta
"""
import torch
import torch.nn as nn


class SingleExpert(nn.Module):
    """
    单个专家：2 层 MLP。
    input_dim → hidden → output_dim
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ExpertGroup(nn.Module):
    """
    5 个专家组。
    
    每个专家的输入拼接规则:
    - Expert_SD:  z_sd + pi_u (sd相关)
    - Expert_EP:  z_ep + block_feats
    - Expert_SP:  z_sp + top-k scores
    - Expert_MIX: z_mix + pi_u + crisis
    - Expert_G:   z_g + pi_u + crisis + stats
    
    简化实现: 所有专家接收相同维度输入（z_k + meta_vector），
    通过训练自动学习关注不同方面。
    """
    def __init__(self,
                 hidden_dim: int,
                 meta_dim: int = 10,
                 expert_hidden: int = 512,
                 expert_output: int = 256,
                 num_experts: int = 5):
        """
        Args:
            hidden_dim: 编码器隐藏维度（每个 z 的维度）
            meta_dim: 附加 meta 特征维度
            expert_hidden: 专家 MLP 隐藏层
            expert_output: 专家输出维度
        """
        super().__init__()
        
        # 每个专家输入 = z_k (hidden_dim) + meta (meta_dim)
        input_dim = hidden_dim + meta_dim
        
        self.experts = nn.ModuleList([
            SingleExpert(input_dim, expert_hidden, expert_output)
            for _ in range(num_experts)
        ])
        
        # Meta 投影层（将不同长度的 meta 映射到统一 meta_dim）
        self.meta_proj = nn.Linear(meta_dim, meta_dim)
    
    def forward(self,
                z_list: list,         # [z_sd, z_ep, z_sp, z_mix, z_g], 各 [d]
                meta: torch.Tensor    # [meta_dim]
                ) -> list:
        """
        Returns:
            expert_outputs: List[Tensor], 每个 [expert_output_dim]
        """
        meta_proj = self.meta_proj(meta)
        outputs = []
        for i, expert in enumerate(self.experts):
            expert_input = torch.cat([z_list[i], meta_proj], dim=0)
            outputs.append(expert(expert_input))
        return outputs
```

---

### Step 5：MoE 融合 + 分类头 (`src/model/moe_head.py`)

```python
"""
moe_head.py — Dense MoE 融合 + 分类头

h_u = Σ_{k=1}^{5} g_k · E_k(input_k)
ŷ_u = σ(W_cls · h_u + b_cls)
"""
import torch
import torch.nn as nn


class MoEHead(nn.Module):
    """
    Dense MoE 融合头。
    
    将 gate_weights 与 expert_outputs 加权求和，
    再经分类头输出用户级预测。
    """
    def __init__(self, expert_output_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(expert_output_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(self,
                gate_weights: torch.Tensor,     # [num_experts]
                expert_outputs: list             # List[Tensor], 各 [expert_output_dim]
                ) -> torch.Tensor:
        """
        Dense 融合：h_u = Σ g_k · E_k
        
        Returns:
            logit: [1], 未经 sigmoid 的 raw logit
        """
        # 加权融合
        stacked = torch.stack(expert_outputs, dim=0)  # [num_experts, output_dim]
        h_u = (gate_weights.unsqueeze(-1) * stacked).sum(dim=0)  # [output_dim]
        
        # 分类
        logit = self.classifier(h_u)  # [1]
        
        return logit, h_u
```

---

### Step 6：Evidence Head (`src/model/evidence_head.py`)

```python
"""
evidence_head.py — 证据选择头

ŝ_i = σ(MLP([h_i; h_u; g]))

每条 risk_post 的证据分数 = 帖子表示 + 用户融合表示 + gate 权重 三者拼接后经 MLP。
取 top-3 作为最终证据帖子。
"""
import torch
import torch.nn as nn


class EvidenceHead(nn.Module):
    """
    证据选择头。
    
    对 risk_posts 中的每条帖子打证据分数，
    训练时用 composite_evidence_score 的 sigmoid 变换作为 silver label。
    """
    def __init__(self,
                 post_dim: int,          # 帖子表示维度 (= encoder hidden_dim)
                 user_dim: int = 256,    # MoE 融合后的用户表示维度
                 gate_dim: int = 5,      # gate 权重维度
                 hidden_dim: int = 128):
        super().__init__()
        
        input_dim = post_dim + user_dim + gate_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self,
                post_reps: torch.Tensor,     # [K, post_dim]
                h_u: torch.Tensor,            # [user_dim]
                gate_weights: torch.Tensor    # [gate_dim]
                ) -> torch.Tensor:
        """
        对每条 risk_post 计算证据分数。
        
        Returns:
            evidence_scores: [K], 经 sigmoid 的证据分数
        """
        K = post_reps.shape[0]
        
        # 将 h_u 和 gate 扩展到 K 条帖子
        h_u_expanded = h_u.unsqueeze(0).expand(K, -1)           # [K, user_dim]
        gate_expanded = gate_weights.unsqueeze(0).expand(K, -1) # [K, gate_dim]
        
        # 拼接
        combined = torch.cat([post_reps, h_u_expanded, gate_expanded], dim=-1)  # [K, input_dim]
        
        # MLP + sigmoid
        scores = torch.sigmoid(self.mlp(combined).squeeze(-1))  # [K]
        
        return scores
    
    def select_top_evidence(self, scores: torch.Tensor, top_k: int = 3):
        """
        选择 top-k 证据帖子。
        
        Returns:
            top_indices: [top_k], 帖子在 risk_posts 中的索引
            top_scores: [top_k], 对应的证据分数
        """
        k = min(top_k, scores.shape[0])
        top_scores, top_indices = torch.topk(scores, k)
        return top_indices, top_scores
```

---

### Step 7：完整模型封装 (`src/model/full_model.py`)

```python
"""
full_model.py — WPG-MoE 完整模型封装

将所有组件组合为单一 nn.Module，暴露统一的 forward 接口。
"""
import torch
import torch.nn as nn
from .post_encoder import PostEncoder
from .user_representation import UserRepresentationModule
from .gate_network import GateNetwork
from .expert_network import ExpertGroup
from .moe_head import MoEHead
from .evidence_head import EvidenceHead


class WPGMoEModel(nn.Module):
    """
    WPG-MoE: Weak-Prior-Guided Dense Mixture-of-Experts
    
    完整前向流程:
    1. PostEncoder 编码 risk_posts / blocks / global_history
    2. UserRepresentationModule 构造 5 路用户表示
    3. GateNetwork 输出 5 维 gate 权重
    4. ExpertGroup 各专家产生判别表示
    5. MoEHead Dense 融合 + 分类
    6. EvidenceHead 选 top-3 证据帖
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: 超参数字典，必须包含:
                - model_name: str, backbone 模型名
                - max_post_tokens: int
                - num_post_markers: int
                - num_experts: int (default 5)
                - expert_hidden: int (default 512)
                - expert_output: int (default 256)
                - gate_hidden: int (default 256)
                - stats_dim: int (default 5)
                - meta_dim: int (default 10)
        """
        super().__init__()
        
        # 组件初始化
        self.encoder = PostEncoder(
            model_name=config.get("model_name", "microsoft/mdeberta-v3-base"),
            max_post_tokens=config.get("max_post_tokens", 256),
            num_post_markers=config.get("num_post_markers", 2048)
        )
        
        hidden_dim = self.encoder.hidden_dim
        expert_output = config.get("expert_output", 256)
        num_experts = config.get("num_experts", 5)
        stats_dim = config.get("stats_dim", 5)
        meta_dim = config.get("meta_dim", 10)
        
        self.user_rep = UserRepresentationModule(
            hidden_dim=hidden_dim,
            stats_dim=stats_dim
        )
        
        self.gate = GateNetwork(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            gate_hidden=config.get("gate_hidden", 256),
            stats_dim=stats_dim,
            prior_dim=4  # p_sd + p_ep + p_sp + crisis
        )
        
        self.experts = ExpertGroup(
            hidden_dim=hidden_dim,
            meta_dim=meta_dim,
            expert_hidden=config.get("expert_hidden", 512),
            expert_output=expert_output,
            num_experts=num_experts
        )
        
        self.moe_head = MoEHead(expert_output_dim=expert_output)
        
        self.evidence_head = EvidenceHead(
            post_dim=hidden_dim,
            user_dim=expert_output,
            gate_dim=num_experts
        )
    
    def forward(self,
                risk_post_texts: list,          # List[str], K 条
                risk_post_markers: list,        # List[str], K 个标记
                block_post_texts: list,         # List[str], 可为空
                block_post_markers: list,       # List[str], 可为空
                global_segment_texts: list,     # List[List[str]], S 段
                global_segment_markers: list,   # List[List[str]], S 段
                pi_u: torch.Tensor,             # [3]
                crisis: torch.Tensor,           # [1]
                stats: torch.Tensor,            # [stats_dim]
                meta_vector: torch.Tensor = None  # [meta_dim]
                ) -> dict:
        """
        完整前向传播。
        
        Returns:
            {
                "logit": [1],
                "gate_weights": [5],
                "evidence_scores": [K],
                "h_u": [expert_output],
                "z_dict": dict of user representations
            }
        """
        # 1. 编码 risk_posts
        risk_reps = self.encoder.encode_posts(risk_post_texts, risk_post_markers)
        
        # 2. 编码 episode_blocks（如有）
        if block_post_texts and len(block_post_texts) > 0:
            block_reps = self.encoder.encode_posts(block_post_texts, block_post_markers)
        else:
            block_reps = None
        
        # 3. 编码 global_history（分段编码 + 段内 mean pooling）
        segment_reps = []
        for seg_texts, seg_markers in zip(global_segment_texts, global_segment_markers):
            if seg_texts:
                seg_rep = self.encoder.encode_posts(seg_texts, seg_markers)
                segment_reps.append(seg_rep.mean(dim=0))
            else:
                segment_reps.append(torch.zeros(self.encoder.hidden_dim,
                                                 device=risk_reps.device))
        segment_reps = torch.stack(segment_reps, dim=0)  # [S, d]
        
        # 4. 用户表示构造
        z_dict = self.user_rep(risk_reps, block_reps, segment_reps, stats)
        
        # 5. 门控
        gate_weights = self.gate(z_dict, pi_u, crisis, stats)
        
        # 6. 专家
        z_list = [z_dict["z_sd"], z_dict["z_ep"], z_dict["z_sp"],
                  z_dict["z_mix"], z_dict["z_g"]]
        if meta_vector is None:
            meta_vector = torch.zeros(self.experts.meta_proj.in_features,
                                       device=risk_reps.device)
        expert_outputs = self.experts(z_list, meta_vector)
        
        # 7. MoE 融合 + 分类
        logit, h_u = self.moe_head(gate_weights, expert_outputs)
        
        # 8. Evidence Head
        evidence_scores = self.evidence_head(risk_reps, h_u, gate_weights)
        
        return {
            "logit": logit,
            "gate_weights": gate_weights,
            "evidence_scores": evidence_scores,
            "h_u": h_u,
            "z_dict": z_dict
        }
```

---

## 模型架构维度速查

| 组件 | 输入维度 | 输出维度 |
|---|---|---|
| PostEncoder | 文本 → tokenize → [seq_len, d] | h_i ∈ R^d（d=backbone hidden_size）|
| AttentionPooling | [K, d] | [d] |
| TemporalSelfAttention | [S, d] | [d] |
| GateNetwork | `5d + 4 + stats_dim` | [5] |
| SingleExpert | `d + meta_dim` | [expert_output=256] |
| MoEHead | [5, 256] + [5] → [256] | logit [1] |
| EvidenceHead | `[K, d + 256 + 5]` | [K] |

---

## 注意事项

1. **默认优先 full fine-tune BERT-family encoder**；只有显存明显不足时才退回 adapter / LoRA
2. **POST marker tokens 需手动添加**到 tokenizer，否则会被拆成 sub-tokens
3. **Attention pooling 是 learned 的**（带可训练权重），不是简单 mean pooling
4. **z_ep 有 fallback 机制**：episode_blocks 为空时退化为 attention pooling on risk_posts（推理时总是 fallback）
5. **gate 的 softmax 输出确保权重和为 1**，所有专家始终参与（Dense MoE）
6. **Evidence Head 的 silver label 来自 composite_evidence_score 的 sigmoid 变换**（见 04_Training 损失函数）
