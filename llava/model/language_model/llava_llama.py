#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[CausalLMOutputWithPast, torch.LongTensor]:
        # 检查是否启用了剪枝功能
        enable_pruning = kwargs.pop("enable_pruning", False)
        
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # 如果启用了剪枝，我们需要修改生成过程
        if enable_pruning and hasattr(self, 'pruning_adapter'):
            # 重置剪枝适配器的状态
            self.pruning_adapter.reset()
            
            # 返回自定义的生成结果
            return self.custom_generate_with_pruning(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        else:
            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

    def custom_generate_with_pruning(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # 为带剪枝的生成设置特殊参数
        max_new_tokens = kwargs.get('max_new_tokens', 512)
        
        # 临时存储生成结果
        generated_tokens = []
        current_input_embeds = inputs_embeds
        current_attention_mask = attention_mask
        current_position_ids = position_ids
        
        # 获取初始的past_key_values
        past_key_values = None
        
        # 逐token生成，应用剪枝
        for i in range(max_new_tokens):
            # 获取模型输出
            outputs = self.forward(
                inputs_embeds=current_input_embeds,
                attention_mask=current_attention_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True  # 为了获取注意力权重
            )
            
            # 获取下一个token的概率
            next_token_logits = outputs.logits[:, -1, :]
            
            # 应用剪枝逻辑（如果存在剪枝适配器）
            if hasattr(self, 'pruning_adapter'):
                # 获取交叉注意力权重
                cross_attention_weights = []
                if outputs.attentions is not None:
                    # 从注意力权重中提取交叉注意力部分
                    for layer_attn in outputs.attentions:
                        # 这里我们简化处理，实际实现可能需要更复杂的逻辑
                        cross_attention_weights.append(layer_attn)
                
                # 应用剪枝适配器
                pruning_result = self.pruning_adapter(
                    input_embeds=current_input_embeds,
                    decoder_hidden_states=outputs.last_hidden_state,
                    cross_attention_weights=cross_attention_weights if cross_attention_weights else None,
                    logits=next_token_logits
                )
                
                # 如果剪枝适配器返回了最终logits，使用它们
                if 'final_logits' in pruning_result and pruning_result['final_logits'] is not None:
                    next_token_logits = pruning_result['final_logits']
            
            # 选择下一个token
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)  # (batch_size, 1)
            
            # 将token ID转换为embeddings
            next_token_embeds = self.model.embed_tokens(next_token_id)  # (batch_size, 1, hidden_dim)
            
            # 更新生成的嵌入
            current_input_embeds = torch.cat([current_input_embeds, next_token_embeds], dim=1)
            
            # 更新attention mask
            if current_attention_mask is not None:
                new_attention_mask = torch.cat([
                    current_attention_mask,
                    torch.ones((current_attention_mask.shape[0], 1), 
                              dtype=current_attention_mask.dtype, 
                              device=current_attention_mask.device)
                ], dim=1)
                current_attention_mask = new_attention_mask
            
            # 更新position ids
            if current_position_ids is not None:
                new_position_ids = torch.cat([
                    current_position_ids,
                    current_position_ids[:, -1:] + 1
                ], dim=1)
                current_position_ids = new_position_ids
            
            # 更新past_key_values为outputs.past_key_values
            past_key_values = outputs.past_key_values
            
            # 记录生成的token
            generated_tokens.append(next_token_id)
            
            # 检查是否达到结束条件
            eos_token_id = self.config.eos_token_id
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
                
        # 将生成的token拼接起来
        generated_tokens_tensor = torch.cat(generated_tokens, dim=1)
        return generated_tokens_tensor

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
