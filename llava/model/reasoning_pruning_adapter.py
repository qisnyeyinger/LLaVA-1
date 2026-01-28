"""
Adapter module for integrating Vision Reasoning Pruning with LLaVA's generation process
将推理感知的视觉Token剪枝机制集成到LLaVA生成管道中
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
from .vision_reasoning_pruning import VisionTokenPruningPipeline


class LLaVAReasoningPruningAdapter(nn.Module):
    """
    LLaVA 推理感知视觉Token剪枝适配器
    
    这个适配器在 LLaVA 的生成过程中集成推理感知的视觉Token剪枝机制：
    1. 在每个解码步提取推理状态和注意力权重
    2. 评估视觉token的语义一致性
    3. 动态剪枝冗余或干扰的视觉token
    4. 应用定向残差对比解码DRCD
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_layers_for_attention: int = 3,
        momentum: float = 0.9,
        consistency_threshold: float = 0.5,
        contrast_coefficient: float = 1.0,
        max_decoding_steps: int = 128,
        enable_progressive: bool = True,
        enable_drcd: bool = True,
    ):
        """
        Args:
            hidden_dim: 模型隐层维度
            num_layers_for_attention: 用于attention聚合的高层层数
            momentum: 一致性累计的动量系数
            consistency_threshold: 初始一致性阈值
            contrast_coefficient: 初始对比系数
            max_decoding_steps: 最大解码步数
            enable_progressive: 是否启用渐进式策略
            enable_drcd: 是否启用DRCD机制
        """
        super().__init__()
        
        self.pruning_pipeline = VisionTokenPruningPipeline(
            hidden_dim=hidden_dim,
            num_layers_for_attention=num_layers_for_attention,
            momentum=momentum,
            consistency_threshold=consistency_threshold,
            contrast_coefficient=contrast_coefficient,
            max_decoding_steps=max_decoding_steps,
        )
        
        self.enable_progressive = enable_progressive
        self.enable_drcd = enable_drcd
        
        # 记录当前解码步
        self.current_step = 0
        self.max_decoding_steps = max_decoding_steps
        
        # 存储用于调试和分析的统计信息
        self.pruning_stats = {}

    def reset(self):
        """重置解码过程中的累计状态"""
        self.pruning_pipeline.consistency_accumulation.reset()
        self.current_step = 0
        self.pruning_stats = {}

    def extract_vision_tokens_from_embeds(
        self,
        input_embeds: torch.Tensor,
        image_token_mask: torch.Tensor,
        num_vision_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        从模型输入embedding中提取视觉token
        
        Args:
            input_embeds: shape (batch_size, seq_len, hidden_dim)
                模型输入embedding
            image_token_mask: shape (batch_size, seq_len)
                标记视觉token位置的mask
            num_vision_tokens: 视觉token数量（如果不指定则从mask推断）
        
        Returns:
            vision_tokens: shape (batch_size, num_vision_tokens, hidden_dim)
            num_vision_tokens: 提取的视觉token数量
        """
        # 获取视觉token所在的位置
        vision_token_positions = torch.where(image_token_mask)
        
        if num_vision_tokens is None:
            num_vision_tokens = (image_token_mask.sum(dim=1).max()).item()
        
        batch_size = input_embeds.shape[0]
        hidden_dim = input_embeds.shape[-1]
        
        # 提取视觉token embedding
        vision_tokens = input_embeds[image_token_mask]  # (num_total_vision_tokens, hidden_dim)
        vision_tokens = vision_tokens.unsqueeze(0)  # 简化处理，假设单张图像
        
        return vision_tokens, num_vision_tokens

    def create_pruned_attention_mask(
        self,
        original_attention_mask: torch.Tensor,
        vision_token_positions: torch.Tensor,
        pruning_indices: torch.Tensor,
        keep_text_tokens: bool = True,
    ) -> torch.Tensor:
        """
        根据剪枝结果创建新的attention mask
        
        Args:
            original_attention_mask: shape (batch_size, seq_len)
                原始attention mask
            vision_token_positions: shape (batch_size, seq_len)
                视觉token位置标记
            pruning_indices: 要移除的视觉token索引
            keep_text_tokens: 是否保留文本token
        
        Returns:
            new_attention_mask: 修改后的attention mask
        """
        new_mask = original_attention_mask.clone()
        
        # 标记要被剪枝的视觉token位置
        vision_positions = torch.where(vision_token_positions)
        
        # 这里的实现取决于如何在attention mask中表示被剪枝的token
        # 简单方案：将被剪枝token的mask设为False
        for idx_to_remove in pruning_indices:
            new_mask[vision_positions[0][idx_to_remove], vision_positions[1][idx_to_remove]] = False
        
        return new_mask

    def forward(
        self,
        input_embeds: torch.Tensor,
        decoder_hidden_states: torch.Tensor,
        cross_attention_weights: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_token_mask: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        noise_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        推理感知视觉Token剪枝的完整流程
        
        Args:
            input_embeds: shape (batch_size, seq_len, hidden_dim)
                当前输入embedding
            decoder_hidden_states: shape (batch_size, seq_len, hidden_dim)
                Decoder最后一层的隐状态
            cross_attention_weights: List of attention tensors
                各层的Cross-Attention权重（可选）
            attention_mask: shape (batch_size, seq_len)
                注意力掩码（可选）
            image_token_mask: shape (batch_size, seq_len)
                视觉token标记（可选）
            logits: shape (vocab_size,) or (batch_size, vocab_size)
                核心证据组logits（用于DRCD）
            noise_logits: shape (vocab_size,) or (batch_size, vocab_size)
                噪声组logits（用于DRCD）
        
        Returns:
            output: 包含以下键的字典
                - consistency_scores: 即时一致性评分
                - accumulated_scores: 累计一致性评分
                - core_mask: 核心证据组mask
                - noise_mask: 干扰噪声组mask
                - phase_config: 当前阶段配置
                - pruning_mask: 建议的剪枝mask
                - final_logits: DRCD处理后的logits（如果提供）
                - step: 当前解码步
        """
        
        # 如果未提供cross_attention_weights，创建placeholder
        if cross_attention_weights is None:
            cross_attention_weights = [
                torch.ones(
                    input_embeds.shape[0], 8, input_embeds.shape[1], 
                    input_embeds.shape[1],
                    device=input_embeds.device,
                    dtype=input_embeds.dtype,
                )
                for _ in range(3)
            ]
        
        # 执行推理感知的视觉Token剪枝流程
        pruning_output = self.pruning_pipeline(
            vision_tokens=input_embeds,
            decoder_hidden_states=decoder_hidden_states,
            cross_attention_weights=cross_attention_weights,
            current_step=self.current_step,
            use_progressive=self.enable_progressive,
        )
        
        # 提取输出
        consistency_scores = pruning_output["consistency_scores"]
        accumulated_scores = pruning_output["accumulated_scores"]
        core_mask = pruning_output["core_mask"]
        noise_mask = pruning_output["noise_mask"]
        phase_config = pruning_output["phase_config"]
        
        # 应用DRCD（如果启用）
        final_logits = None
        if self.enable_drcd and logits is not None and noise_logits is not None:
            final_logits = self.pruning_pipeline.drcd.compute_residual_contrastive_logits(
                logits_core=logits,
                logits_noise=noise_logits,
                lambda_coeff=phase_config.get("lambda_coeff", 1.0),
            )
        
        # 根据累计得分创建剪枝建议
        pruning_mask = core_mask  # 保留核心证据，移除干扰
        
        # 记录统计信息
        self.pruning_stats[self.current_step] = {
            "consistency_scores": consistency_scores.detach().cpu().mean().item(),
            "accumulated_scores": accumulated_scores.detach().cpu().mean().item(),
            "core_token_ratio": core_mask.float().mean().item(),
            "phase": phase_config.get("phase", "unknown"),
            "lambda_coeff": phase_config.get("lambda_coeff", 0.0),
        }
        
        # 更新解码步
        self.current_step += 1
        
        return {
            "consistency_scores": consistency_scores,
            "accumulated_scores": accumulated_scores,
            "core_mask": core_mask,
            "noise_mask": noise_mask,
            "phase_config": phase_config,
            "pruning_mask": pruning_mask,
            "final_logits": final_logits,
            "step": self.current_step - 1,
        }

    def get_pruning_stats(self) -> Dict[int, Dict[str, float]]:
        """获取剪枝统计信息"""
        return self.pruning_stats.copy()

    def print_pruning_summary(self):
        """打印剪枝摘要"""
        if not self.pruning_stats:
            print("No pruning statistics available.")
            return
        
        print("\n" + "="*60)
        print("Vision Token Pruning Summary")
        print("="*60)
        print(f"{'Step':<6} {'Consistency':<15} {'Core Ratio':<15} {'Phase':<20} {'Lambda':<10}")
        print("-"*60)
        
        for step in sorted(self.pruning_stats.keys()):
            stats = self.pruning_stats[step]
            print(
                f"{step:<6} "
                f"{stats['consistency_scores']:<15.4f} "
                f"{stats['core_token_ratio']:<15.2%} "
                f"{stats['phase']:<20} "
                f"{stats['lambda_coeff']:<10.3f}"
            )
        
        print("="*60 + "\n")


def integrate_pruning_into_model(model, adapter_config: Optional[Dict[str, Any]] = None):
    """
    将剪枝适配器集成到LLaVA模型中
    
    Args:
        model: LLaVA模型实例
        adapter_config: 适配器配置字典
    
    Returns:
        model: 集成了剪枝功能的模型
        adapter: LLaVAReasoningPruningAdapter实例
    """
    if adapter_config is None:
        adapter_config = {}
    
    # 获取模型隐层维度
    hidden_dim = getattr(model.config, 'hidden_size', 4096)
    
    # 创建适配器
    adapter = LLaVAReasoningPruningAdapter(
        hidden_dim=hidden_dim,
        **{k: v for k, v in adapter_config.items() if k in [
            'num_layers_for_attention', 'momentum', 'consistency_threshold',
            'contrast_coefficient', 'max_decoding_steps', 'enable_progressive',
            'enable_drcd'
        ]}
    )
    
    # 将适配器绑定到模型（作为属性）
    model.pruning_adapter = adapter
    
    return model, adapter
