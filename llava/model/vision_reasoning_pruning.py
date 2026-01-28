"""
Reasoning-Coupled Progressive Visual Token Pruning Module
用于解决多模态大模型视觉幻觉问题的推理感知视觉Token剪枝方案
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import numpy as np


class ReasoningStateModeling(nn.Module):
    """
    推理状态建模与语义一致性评估模块
    Module A: Reasoning-Aware Consistency Modeling
    """

    def __init__(self, hidden_dim: int = 4096, num_layers_for_attention: int = 3):
        """
        Args:
            hidden_dim: 隐层维度 (e.g., 4096 for 7B/13B LLaVA)
            num_layers_for_attention: 用于attention聚合的高层层数
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers_for_attention = num_layers_for_attention

    def get_reasoning_state(
        self,
        decoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        A.1 推理状态代理 - 从 Decoder 最后一层、最后一个生成位置提取隐状态
        
        Args:
            decoder_hidden_states: shape (batch_size, seq_len, hidden_dim)
                Decoder 最后一层的隐状态
        
        Returns:
            h_t: shape (batch_size, hidden_dim) - 当前推理状态
        """
        # 提取最后一个位置（当前生成步的上下文表示）
        h_t = decoder_hidden_states[:, -1, :]  # (batch_size, hidden_dim)
        return h_t

    def compute_attention_aware_vision_features(
        self,
        vision_tokens: torch.Tensor,
        cross_attention_weights: List[torch.Tensor],
        num_layers: Optional[int] = None,
    ) -> torch.Tensor:
        """
        A.2 推理感知的视觉特征轻量级重构
        利用 Cross-Attention 权重对视觉 token 进行加权聚合
        
        Args:
            vision_tokens: shape (batch_size, num_vision_tokens, hidden_dim)
                视觉token的静态表示
            cross_attention_weights: List of tensors with shape 
                (batch_size, num_heads, query_len, num_vision_tokens)
                不同层的Cross-Attention权重
            num_layers: 用于聚合的高层层数 (default: self.num_layers_for_attention)
        
        Returns:
            attention_aware_features: shape (batch_size, num_vision_tokens, hidden_dim)
                注意力感知的视觉特征
        """
        if num_layers is None:
            num_layers = self.num_layers_for_attention

        batch_size, num_vision_tokens, hidden_dim = vision_tokens.shape
        device = vision_tokens.device

        # 选择最后num_layers层的attention权重
        selected_attention_weights = cross_attention_weights[-num_layers:]
        
        # 聚合attention权重：对heads和query维度进行平均
        aggregated_weights = torch.zeros(
            batch_size, num_vision_tokens, device=device
        )
        
        for attn_weight in selected_attention_weights:
            # attn_weight: (batch_size, num_heads, query_len, num_vision_tokens)
            # 对heads和query维度平均，得到 (batch_size, num_vision_tokens)
            layer_weight = attn_weight.mean(dim=(1, 2))  # 平均所有head和query位置
            aggregated_weights += layer_weight
        
        # 归一化
        aggregated_weights = aggregated_weights / num_layers
        aggregated_weights = aggregated_weights / (aggregated_weights.sum(dim=1, keepdim=True) + 1e-6)
        
        # 加权聚合：v_tilde_i = sum(alpha * v_i)
        attention_aware_features = (
            aggregated_weights.unsqueeze(-1) * vision_tokens  # (batch_size, num_vision_tokens, 1) * (batch_size, num_vision_tokens, hidden_dim)
        )
        
        return attention_aware_features

    def compute_consistency_score(
        self,
        reasoning_state: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        A.3 即时语义一致性评分
        使用余弦相似度评估视觉token与当前推理状态的一致性
        
        Args:
            reasoning_state: shape (batch_size, hidden_dim)
                推理状态 h_t
            vision_features: shape (batch_size, num_vision_tokens, hidden_dim)
                视觉特征（注意力感知或原始）
        
        Returns:
            consistency_scores: shape (batch_size, num_vision_tokens)
                每个视觉token的即时一致性评分 [0, 1]
        """
        # 归一化
        reasoning_state_norm = F.normalize(reasoning_state, p=2, dim=-1)  # (batch_size, hidden_dim)
        vision_features_norm = F.normalize(vision_features, p=2, dim=-1)  # (batch_size, num_vision_tokens, hidden_dim)
        
        # 余弦相似度：cos(h_t, v_i) = (h_t · v_i) / (||h_t|| * ||v_i||)
        # (batch_size, 1, hidden_dim) @ (batch_size, hidden_dim, num_vision_tokens) -> (batch_size, 1, num_vision_tokens)
        consistency_scores = torch.bmm(
            reasoning_state_norm.unsqueeze(1),
            vision_features_norm.transpose(1, 2)
        )  # (batch_size, 1, num_vision_tokens)
        
        consistency_scores = consistency_scores.squeeze(1)  # (batch_size, num_vision_tokens)
        
        # 将余弦相似度 [-1, 1] 映射到 [0, 1]
        consistency_scores = (consistency_scores + 1.0) / 2.0
        
        return consistency_scores


class ConsistencyAccumulation(nn.Module):
    """
    跨步一致性累计机制
    Module B: Consistency Accumulation with Momentum
    """

    def __init__(self, momentum: float = 0.9):
        """
        Args:
            momentum: 指数滑动平均的动量系数 γ ∈ [0, 1)
        """
        super().__init__()
        self.momentum = momentum
        self.accumulated_scores = None

    def update_accumulated_score(
        self,
        instant_scores: torch.Tensor,
        reset: bool = False,
    ) -> torch.Tensor:
        """
        B.1 跨步一致性累计 - 指数滑动平均更新
        S_i^(t) = γ * S_i^(t-1) + (1 - γ) * s_i^(t)
        
        Args:
            instant_scores: shape (batch_size, num_vision_tokens)
                当前解码步的即时一致性评分
            reset: 是否重置累计得分（用于新样本或新生成过程）
        
        Returns:
            accumulated_scores: shape (batch_size, num_vision_tokens)
                更新后的累计一致性得分
        """
        batch_size, num_vision_tokens = instant_scores.shape
        device = instant_scores.device

        if reset or self.accumulated_scores is None:
            # 初始化或重置累计得分
            self.accumulated_scores = instant_scores.clone()
        else:
            # 确保维度匹配
            if self.accumulated_scores.shape != instant_scores.shape:
                self.accumulated_scores = instant_scores.clone()
            else:
                # 指数滑动平均更新
                self.accumulated_scores = (
                    self.momentum * self.accumulated_scores +
                    (1 - self.momentum) * instant_scores
                )

        return self.accumulated_scores.clone()

    def reset(self):
        """重置累计得分"""
        self.accumulated_scores = None


class DirectedResidualContrastiveDecoding(nn.Module):
    """
    定向残差对比解码机制
    Module C: Directed Residual Contrastive Decoding (DRCD)
    """

    def __init__(self, consistency_threshold: float = 0.5, contrast_coefficient: float = 1.0):
        """
        Args:
            consistency_threshold: 一致性阈值 τ
            contrast_coefficient: 对比系数 λ
        """
        super().__init__()
        self.consistency_threshold = consistency_threshold
        self.contrast_coefficient = contrast_coefficient

    def group_vision_tokens(
        self,
        accumulated_scores: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        C.1 动态视觉Token分组
        根据累计一致性得分将视觉token分为核心证据组C和干扰噪声组N
        
        Args:
            accumulated_scores: shape (batch_size, num_vision_tokens)
                累计一致性得分
            threshold: 一致性阈值（可覆盖默认值）
        
        Returns:
            core_mask: shape (batch_size, num_vision_tokens) - bool
                核心证据组mask (True表示在C组)
            noise_mask: shape (batch_size, num_vision_tokens) - bool
                干扰噪声组mask (True表示在N组)
        """
        if threshold is None:
            threshold = self.consistency_threshold

        # C = {i | S_i^(t) >= τ}
        core_mask = accumulated_scores >= threshold  # (batch_size, num_vision_tokens)
        
        # N = {i | S_i^(t) < τ}
        noise_mask = ~core_mask

        return core_mask, noise_mask

    def compute_residual_contrastive_logits(
        self,
        logits_core: torch.Tensor,
        logits_noise: torch.Tensor,
        lambda_coeff: Optional[float] = None,
    ) -> torch.Tensor:
        """
        C.2 & C.3 残差对比Logits计算与分布级对冲
        Logits_final = Logits_C - λ * Logits_N
        
        Args:
            logits_core: shape (vocab_size,) or (batch_size, vocab_size)
                仅保留C组视觉token的Logits
            logits_noise: shape (vocab_size,) or (batch_size, vocab_size)
                仅保留N组视觉token的Logits
            lambda_coeff: 对比系数λ（可覆盖默认值）
        
        Returns:
            logits_final: shape 与输入相同
                最终对冲后的Logits
        """
        if lambda_coeff is None:
            lambda_coeff = self.contrast_coefficient

        # 分布级对冲：从核心证据的logits中减去噪声的影响
        logits_final = logits_core - lambda_coeff * logits_noise

        return logits_final


class ProgressiveRestructuringScheduler(nn.Module):
    """
    渐进式动态重构策略
    Module D: Progressive Context Restructuring
    """

    def __init__(
        self,
        max_decoding_steps: int = 128,
        early_phase_threshold: float = 0.3,
        late_phase_threshold: float = 0.7,
    ):
        """
        Args:
            max_decoding_steps: 最大解码步数
            early_phase_threshold: 早期阶段（语义锚定）的进度阈值
            late_phase_threshold: 中后期阶段（精准纠偏）的进度阈值
        """
        super().__init__()
        self.max_decoding_steps = max_decoding_steps
        self.early_phase_threshold = early_phase_threshold
        self.late_phase_threshold = late_phase_threshold

    def get_phase_config(
        self,
        current_step: int,
    ) -> Dict[str, float]:
        """
        根据当前解码步返回该阶段的配置参数
        
        Args:
            current_step: 当前解码步 (从0开始)
        
        Returns:
            config: 包含该阶段推荐参数的字典
                - pruning_ratio: 视觉token裁剪比例 [0, 1]
                - lambda_coeff: 对比系数 λ
                - consistency_threshold: 一致性阈值 τ
                - use_drcd: 是否使用DRCD
        """
        progress = current_step / max(self.max_decoding_steps, 1.0)

        if progress < self.early_phase_threshold:
            # 早期阶段：语义锚定
            # 保留丰富视觉上下文，较低裁剪比例，较小λ
            config = {
                "pruning_ratio": 0.1,  # 仅裁剪10%冗余token
                "lambda_coeff": 0.3,   # 较小的对比系数
                "consistency_threshold": 0.2,  # 较低阈值
                "use_drcd": False,  # 早期不使用DRCD
                "phase": "early_anchoring",
            }
        elif progress < self.late_phase_threshold:
            # 中期阶段：逐步纠偏
            # 线性增加裁剪强度
            transition_progress = (progress - self.early_phase_threshold) / (
                self.late_phase_threshold - self.early_phase_threshold
            )
            config = {
                "pruning_ratio": 0.1 + transition_progress * 0.3,  # 10% -> 40%
                "lambda_coeff": 0.3 + transition_progress * 0.4,   # 0.3 -> 0.7
                "consistency_threshold": 0.2 + transition_progress * 0.2,  # 0.2 -> 0.4
                "use_drcd": True,
                "phase": "middle_transition",
            }
        else:
            # 后期阶段：精准纠偏
            # 最大化裁剪强度和对冲能力
            config = {
                "pruning_ratio": 0.4,  # 裁剪40%冗余token
                "lambda_coeff": 0.7,   # 较大的对比系数
                "consistency_threshold": 0.4,  # 较高阈值
                "use_drcd": True,  # 强化DRCD
                "phase": "late_refinement",
            }

        return config


class VisionTokenPruningPipeline(nn.Module):
    """
    整合上述所有模块的完整视觉Token剪枝管道
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_layers_for_attention: int = 3,
        momentum: float = 0.9,
        consistency_threshold: float = 0.5,
        contrast_coefficient: float = 1.0,
        max_decoding_steps: int = 128,
    ):
        super().__init__()

        self.reasoning_state_modeling = ReasoningStateModeling(
            hidden_dim=hidden_dim,
            num_layers_for_attention=num_layers_for_attention,
        )
        self.consistency_accumulation = ConsistencyAccumulation(momentum=momentum)
        self.drcd = DirectedResidualContrastiveDecoding(
            consistency_threshold=consistency_threshold,
            contrast_coefficient=contrast_coefficient,
        )
        self.progressive_scheduler = ProgressiveRestructuringScheduler(
            max_decoding_steps=max_decoding_steps,
        )

    def forward(
        self,
        vision_tokens: torch.Tensor,
        decoder_hidden_states: torch.Tensor,
        cross_attention_weights: List[torch.Tensor],
        current_step: int = 0,
        use_progressive: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        完整的推理感知视觉Token剪枝流程
        
        Args:
            vision_tokens: shape (batch_size, num_vision_tokens, hidden_dim)
                视觉token表示
            decoder_hidden_states: shape (batch_size, seq_len, hidden_dim)
                Decoder最后一层隐状态
            cross_attention_weights: List of attention tensors
                Cross-Attention权重列表
            current_step: 当前解码步
            use_progressive: 是否使用渐进式策略
        
        Returns:
            output: 包含以下键的字典
                - consistency_scores: 即时一致性得分
                - accumulated_scores: 累计一致性得分
                - core_mask: 核心证据组mask
                - noise_mask: 干扰噪声组mask
                - phase_config: 当前阶段配置
        """
        # A. 推理状态建模
        reasoning_state = self.reasoning_state_modeling.get_reasoning_state(
            decoder_hidden_states
        )
        
        attention_aware_features = self.reasoning_state_modeling.compute_attention_aware_vision_features(
            vision_tokens, cross_attention_weights
        )
        
        # A.3 计算即时一致性评分
        consistency_scores = self.reasoning_state_modeling.compute_consistency_score(
            reasoning_state, attention_aware_features
        )
        
        # B. 跨步一致性累计
        accumulated_scores = self.consistency_accumulation.update_accumulated_score(
            consistency_scores
        )
        
        # D. 获取当前阶段配置
        phase_config = self.progressive_scheduler.get_phase_config(current_step)
        
        # C. DRCD：根据配置选择是否应用
        if use_progressive and phase_config["use_drcd"]:
            core_mask, noise_mask = self.drcd.group_vision_tokens(
                accumulated_scores,
                threshold=phase_config["consistency_threshold"],
            )
        else:
            core_mask, noise_mask = self.drcd.group_vision_tokens(
                accumulated_scores
            )
        
        output = {
            "consistency_scores": consistency_scores,
            "accumulated_scores": accumulated_scores,
            "core_mask": core_mask,
            "noise_mask": noise_mask,
            "phase_config": phase_config,
            "reasoning_state": reasoning_state,
            "attention_aware_features": attention_aware_features,
        }
        
        return output
