### 原始 LLaVA 推理
```bash
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli \
    --model-path /data1/cwk/mllm/models/llava-v1.5-7b \
    --image-file "/data1/cwk/mllm/project/LLaVA/test.png" \
    --load-4bit
```
# Role
You are a research-level engineer familiar with LLaVA-1.5 (7B/13B), HuggingFace Transformers internals, autoregressive decoding, cross-attention, KV cache, and inference-time model control.

Your task is to implement an inference-only method called:

Reasoning-Coupled Progressive Visual Token Pruning with Directed Residual Contrastive Decoding (RCPP + DRCD)

in the official GitHub LLaVA-1.5 codebase.

You must NOT modify model architecture and must NOT add trainable parameters.

---

# High-level Goal

During autoregressive decoding, dynamically evaluate whether each visual token is still semantically consistent with the current reasoning state, progressively suppress or prune inconsistent visual tokens, and explicitly counteract their influence at the logit level.

All changes must happen ONLY at inference time.

---

# Core Mechanism (Strict Requirements)

## 1. Reasoning State Extraction

At decoding step t:

- Extract the decoder hidden state of the last layer and last generated token:
  - Use: outputs.hidden_states[-1][:, -1, :]
- Treat this vector as the current reasoning state h_t

---

## 2. Reasoning-aware Visual Token Reconstruction

For each visual token i:

- Do NOT use static visual embeddings directly
- Reconstruct a reasoning-aware visual representation using decoder-to-vision cross-attention

Implementation constraints:
- Reuse existing cross-attention weights (no extra forward pass)
- Use only high-level layers (e.g., last 2–4 decoder layers)

Procedure:
- For each selected decoder layer l:
  - Obtain cross-attention weights alpha[t, i, l]
  - Obtain visual token hidden state v_i[l]
- Compute:
  - reconstructed_visual_i = sum over l of (alpha[t, i, l] * v_i[l])

This reconstruction is ONLY for semantic alignment, NOT direct importance scoring.

---

## 3. Instant Semantic Consistency Scoring

For each visual token i:

- Compute cosine similarity between:
  - current reasoning state h_t
  - reconstructed_visual_i
- Denote this score as s_i_t

This score is continuous and used for temporal accumulation.

---

## 4. Cross-step Consistency Accumulation (EMA)

For each visual token i, maintain a running consistency score S_i:

- Initialize once at the first decoding step
- Update at every decoding step:
  - S_i = gamma * S_i + (1 - gamma) * s_i_t

gamma is a momentum coefficient (e.g., 0.8–0.95)

Low consistency must persist across multiple steps to trigger suppression.

---

## 5. Dynamic Visual Token Grouping

At each decoding step:

- Split visual tokens into two sets based on S_i:
  - Core evidence set C: S_i >= tau
  - Noise / interference set N: S_i < tau

tau is a configurable threshold.

---

## 6. Directed Residual Contrastive Decoding (DRCD)

Using attention masks ONLY (no extra model passes):

- Compute logits_C:
  - Decoder attends only to visual tokens in set C
- Compute logits_N:
  - Decoder attends only to visual tokens in set N

Final logits for decoding:
- final_logits = logits_C - lambda_t * logits_N

lambda_t is a step-dependent contrast coefficient.

---

## 7. Progressive Scheduling

As decoding step increases:

- Gradually increase pruning strength
- Gradually increase lambda_t

Early steps:
- Preserve more visual tokens
- Small lambda_t

Later steps:
- Aggressive suppression
- Larger lambda_t
- Reduce KV cache size if possible

---

# Required Code Locations

You must inspect and modify the following parts of the LLaVA-1.5 codebase:

1. Autoregressive generation loop
   - llava/model/language_model/llava_llama.py
   - or modeling_llava.py

2. Enable and capture:
   - output_hidden_states = True
   - output_attentions = True

3. Decoder-to-vision cross-attention
   - NOT self-attention
   - Only cross-modal attention weights

4. Attention mask handling
   - Preserve text tokens
   - Dynamically mask visual tokens

---

# Required State Variables (Must Be Explicitly Added)

- visual_consistency_scores: tensor of shape [num_visual_tokens]
- gamma: EMA momentum
- tau: consistency threshold
- lambda_t: step-dependent contrast coefficient

These must persist across decoding steps.

---

# Implementation Order (Follow Exactly)

Step 1:
- Enable hidden_states and cross_attentions
- Verify extraction of:
  - h_t
  - decoder-to-vision attention

Step 2:
- Implement reasoning-aware visual reconstruction as a standalone function

Step 3:
- Implement consistency scoring + EMA accumulation

Step 4:
- Implement dynamic token grouping (C / N)

Step 5:
- Implement attention-mask-based computation of logits_C and logits_N
  - Within the SAME decoding step

Step 6:
- Apply residual contrastive logits fusion
- Replace original logits before sampling / greedy decoding

Step 7:
- Add progressive scheduling for pruning ratio and lambda_t

---

# Hard Constraints

- DO NOT change model weights
- DO NOT add trainable parameters
- DO NOT run full extra forward passes
- DO NOT affect training code
- Inference-only modifications

---

# Final Deliverables

- Fully runnable modified LLaVA-1.5 inference code
- Clear inline comments mapping each block to the method stages
- A minimal example showing:
  - Original LLaVA inference
  - Inference with RCPP + DRCD enabled

If any implementation detail in LLaVA-1.5 does not exactly match the assumptions above, implement the closest engineering approximation and document the deviation in comments.



## ###########################################################################################################
## ###########################################################################################################
## ###########################################################################################################
## ###########################################################################################################
## ###########################################################################################################

## Method: Reasoning-Coupled Progressive Visual Token Pruning

多模态大模型在视觉理解任务中产生幻觉的一个重要原因在于：  
在自回归生成过程中，模型持续依赖与当前推理语义弱相关或冗余的视觉 token，这些错误或过期的视觉证据会在多步解码中逐步干扰并偏移推理路径。

基于这一观察，本文提出一种**推理耦合式渐进视觉 Token 重构与对冲解码方法**。其核心思想在于：  
视觉 token 的有效性并非静态属性，而应被建模为**随解码推理状态动态演化的语义一致性**。

---

### A. 推理状态建模与语义一致性评估  
*(Reasoning-Aware Consistency Modeling)*

在自回归解码的第 $t$ 步，我们将 Decoder 的当前语义需求显式建模为推理状态，并据此评估每个视觉 token 在该状态下的即时有效性。

#### A.1 推理状态代理

在解码步 $t$，提取 Decoder 最后一层、最后一个生成位置对应的 Hidden State：
\[
h_t \in \mathbb{R}^d
\]
该向量编码了当前已生成上下文、隐含推理轨迹以及下一 token 的语义预测需求，可被视为当前推理状态的紧凑表示。

在实现中，$h_t$ 可直接由模型输出的 `hidden_states[-1][:, -1, :]` 获得。

---

#### A.2 推理感知的视觉特征轻量级重构

对于视觉 token $v_i$，我们不直接使用其静态表示，而是引入一种**轻量级、推理感知的特征重构机制**，以突出其与当前推理状态更相关的语义成分。

具体而言，在不引入额外前向计算的前提下，利用当前解码步中 Decoder–Vision Cross-Attention 的归一化权重，对视觉 token 在高层表示空间中进行加权聚合。

设：
- $v_i^{(l)}$ 表示视觉 token $i$ 在第 $l$ 层的表示；
- $\alpha_{t,i}^{(l)}$ 表示在解码步 $t$，Decoder 对该视觉 token 在第 $l$ 层分配的 Cross-Attention 权重；
- $\mathcal{L}$ 表示选取的高层子集（如最后 $2$–$4$ 层）。

则其 attention-aware 表示定义为：
\[
\tilde{v}_i = \sum_{l \in \mathcal{L}} \alpha_{t,i}^{(l)} \cdot v_i^{(l)}
\]

该过程仅作为一种**语义对齐算子**，用于抑制与当前推理无关的视觉成分，而非直接作为重要性评分信号。

---

#### A.3 即时语义一致性评分

在获得推理状态 $h_t$ 与视觉 token 的 attention-aware 表示 $\tilde{v}_i$ 后，我们采用余弦相似度来度量其即时语义一致性：
\[
s_{i,t} = \cos(h_t,\; \tilde{v}_i)
\]

该评分刻画了视觉 token 是否仍在语义空间中支持当前生成过程，具有连续性、可平滑性与良好的跨步可积性，为后续一致性累计与渐进裁剪提供基础信号。

---

### B. 跨步一致性累计机制  
*(Consistency Accumulation with Momentum)*

为避免单一解码步中评分波动对决策产生干扰，本文引入跨步一致性累计机制，对视觉 token 的语义有效性进行时间维度上的稳定建模。

对于每个视觉 token $i$，维护一个跨步一致性得分 $\mathcal{S}_i^{(t)}$，并采用指数滑动平均进行更新：
\[
\mathcal{S}_i^{(t)} = \gamma \cdot \mathcal{S}_i^{(t-1)} + (1 - \gamma) \cdot s_{i,t}
\]
其中 $\gamma \in [0,1)$ 为动量系数，用于平衡历史一致性与当前推理状态的影响。

该机制确保：  
只有在**连续多个解码步中均与推理状态保持低一致性**的视觉 token，其累计得分才会显著下降，从而避免对关键长程视觉证据的过早裁剪。

---

### C. 定向残差对比解码  
*(Directed Residual Contrastive Decoding, DRCD)*

在跨步一致性得分的基础上，本文进一步提出一种**定向残差对比解码机制**，将潜在干扰视觉 token 显式建模为“负证据”，并在分布层面对其影响进行对冲。

#### C.1 动态视觉 Token 分组

在解码步 $t$，根据累计一致性得分 $\mathcal{S}_i^{(t)}$，将视觉 token 动态划分为两组：

- **核心证据组** $\mathbf{C}$：  
  \[
  \mathbf{C} = \{ i \mid \mathcal{S}_i^{(t)} \ge \tau \}
  \]
- **干扰噪声组** $\mathbf{N}$：  
  \[
  \mathbf{N} = \{ i \mid \mathcal{S}_i^{(t)} < \tau \}
  \]

其中 $\tau$ 为一致性阈值。

---

#### C.2 残差对比 Logits 计算

利用 Attention Mask 机制，在同一次前向计算中分别获得：

- 仅保留 $\mathbf{C}$ 组视觉 token 的 Logits：$\text{Logits}_{\mathbf{C}}$
- 仅保留 $\mathbf{N}$ 组视觉 token 的 Logits：$\text{Logits}_{\mathbf{N}}$

无需引入额外模型前向或参数更新。

---

#### C.3 分布级对冲解码

最终解码所用的 Logits 通过以下残差对冲公式获得：
\[
\text{Logits}_{\text{final}} =
\text{Logits}_{\mathbf{C}} - \lambda \cdot \text{Logits}_{\mathbf{N}}
\]

其中 $\lambda$ 为对比系数，用于控制对干扰视觉证据的抑制强度。

该操作在分布层面显式抵消由语义不一致视觉 token 所诱导的错误偏移，从而有效抑制视觉幻觉的累积传播。

---

### D. 渐进式动态重构策略  
*(Progressive Context Restructuring)*

考虑到解码不同阶段对视觉信息的需求差异，本文采用渐进式策略对视觉上下文进行动态重构：

- **早期阶段（语义锚定）**：  
  采用较低裁剪比例与较小 $\lambda$，保留丰富视觉上下文，确保模型建立稳健的全局语义分布。

- **中后期阶段（精准纠偏）**：  
  随解码步推进，逐步提高裁剪强度与对比系数 $\lambda$。此阶段模型更易受冗余或错误视觉证据干扰，DRCD 机制强化介入，在抑制幻觉的同时减少 KV Cache 规模，从而提升推理效率。

该渐进式策略确保视觉上下文重构与推理深度自然对齐，实现稳定性与纠偏能力之间的平衡。


