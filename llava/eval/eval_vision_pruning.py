"""
Testing and Evaluation Script for Vision Token Pruning
视觉Token剪枝的测试与评估脚本

用于：
1. 对比原始LLaVA与剪枝版本的幻觉减少情况
2. 测试推理性能和速度
3. 可视化剪枝效果
"""

import os
import sys
import torch
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

# LLaVA imports
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.model.reasoning_pruning_adapter import integrate_pruning_into_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
import requests
from io import BytesIO


def load_image(image_file: str) -> Image.Image:
    """加载图像文件或URL"""
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


class VisionTokenPruningEvaluator:
    """视觉Token剪枝评估器"""

    def __init__(
        self,
        model_path: str,
        model_base: Optional[str] = None,
        device: str = 'cuda',
        load_8bit: bool = False,
        load_4bit: bool = False,
    ):
        """
        初始化评估器
        
        Args:
            model_path: 模型路径
            model_base: 基础模型路径
            device: 设备
            load_8bit: 使用8-bit量化
            load_4bit: 使用4-bit量化
        """
        disable_torch_init()
        
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = \
            load_pretrained_model(model_path, model_base, self.model_name, 
                                load_8bit, load_4bit, device=device)
        
        self.device = device
        self.model_path = model_path
        
        # 设置会话模式
        if "llama-2" in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"
        
        print(f"\n✓ 模型加载完成")
        print(f"  模型: {self.model_name}")
        print(f"  会话模式: {self.conv_mode}")
        print(f"  设备: {device}\n")

    def setup_pruning(self, adapter_config: Optional[Dict] = None) -> None:
        """为模型集成剪枝适配器"""
        if adapter_config is None:
            adapter_config = {
                'num_layers_for_attention': 3,
                'momentum': 0.9,
                'consistency_threshold': 0.5,
                'contrast_coefficient': 1.0,
                'max_decoding_steps': 128,
                'enable_progressive': True,
                'enable_drcd': True,
            }
        
        self.model, self.pruning_adapter = integrate_pruning_into_model(
            self.model, adapter_config
        )
        
        print(f"✓ 已启用视觉Token剪枝")
        print(f"  配置: {adapter_config}\n")

    def generate_response(
        self,
        image: Image.Image,
        prompt: str,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        use_pruning: bool = True,
    ) -> Tuple[str, Dict, float]:
        """
        生成响应
        
        Args:
            image: 输入图像
            prompt: 输入提示
            temperature: 生成温度
            max_new_tokens: 最大新token数
            use_pruning: 是否使用剪枝
        
        Returns:
            response: 生成的文本
            stats: 统计信息字典
            elapsed_time: 耗时（秒）
        """
        # 准备输入
        image_size = image.size
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if type(image_tensor) is list:
            image_tensor = [img.to(self.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=torch.float16)

        # 重置剪枝状态
        if use_pruning and hasattr(self.model, 'pruning_adapter'):
            self.model.pruning_adapter.reset()

        # 构建对话
        conv = conv_templates[self.conv_mode].copy()
        if DEFAULT_IMAGE_TOKEN not in prompt:
            if self.model.config.mm_use_im_start_end:
                prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
            else:
                prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self.device)

        # 计时生成
        start_time = time.time()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=temperature > 0,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        elapsed_time = time.time() - start_time

        # 解码
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        # 收集统计信息
        stats = {
            'elapsed_time': elapsed_time,
            'output_length': len(response),
        }
        
        if use_pruning and hasattr(self.model, 'pruning_adapter'):
            stats['pruning_stats'] = self.model.pruning_adapter.get_pruning_stats()

        return response, stats, elapsed_time

    def evaluate_hallucination_reduction(
        self,
        image_file: str,
        questions: List[str],
        output_dir: str = "./evaluation_results",
    ) -> Dict:
        """
        评估幻觉减少情况
        
        Args:
            image_file: 测试图像文件
            questions: 测试问题列表
            output_dir: 输出目录
        
        Returns:
            results: 包含对比结果的字典
        """
        os.makedirs(output_dir, exist_ok=True)
        
        image = load_image(image_file)
        results = {
            'image_file': image_file,
            'model': self.model_name,
            'questions': [],
        }

        print(f"\n{'='*70}")
        print(f"开始评估幻觉减少 - 使用图像: {image_file}")
        print(f"{'='*70}\n")

        # 原始LLaVA的响应
        print("Step 1: 运行原始LLaVA（无剪枝）")
        print("-" * 70)
        
        original_responses = []
        for i, question in enumerate(tqdm(questions, desc="原始LLaVA")):
            try:
                response, stats, elapsed_time = self.generate_response(
                    image, question, use_pruning=False
                )
                original_responses.append({
                    'question': question,
                    'response': response,
                    'stats': stats,
                })
            except Exception as e:
                print(f"  错误处理问题 {i}: {e}")
                original_responses.append({
                    'question': question,
                    'response': f"[Error: {str(e)}]",
                    'stats': {'error': str(e)},
                })

        # 带剪枝的LLaVA响应
        print("\nStep 2: 运行带剪枝的LLaVA")
        print("-" * 70)
        
        self.setup_pruning()
        
        pruned_responses = []
        for i, question in enumerate(tqdm(questions, desc="剪枝LLaVA")):
            try:
                response, stats, elapsed_time = self.generate_response(
                    image, question, use_pruning=True
                )
                pruned_responses.append({
                    'question': question,
                    'response': response,
                    'stats': stats,
                })
            except Exception as e:
                print(f"  错误处理问题 {i}: {e}")
                pruned_responses.append({
                    'question': question,
                    'response': f"[Error: {str(e)}]",
                    'stats': {'error': str(e)},
                })

        # 对比分析
        print("\nStep 3: 对比分析")
        print("-" * 70)
        
        comparison_results = []
        for orig, pruned in zip(original_responses, pruned_responses):
            comparison = {
                'question': orig['question'],
                'original_response': orig['response'],
                'pruned_response': pruned['response'],
                'response_length_original': len(orig['response']),
                'response_length_pruned': len(pruned['response']),
                'time_original': orig['stats'].get('elapsed_time', 0),
                'time_pruned': pruned['stats'].get('elapsed_time', 0),
            }
            comparison_results.append(comparison)
            
            results['questions'].append(comparison)

        # 计算统计信息
        avg_time_original = np.mean([q['time_original'] for q in comparison_results])
        avg_time_pruned = np.mean([q['time_pruned'] for q in comparison_results])
        
        results['statistics'] = {
            'num_questions': len(questions),
            'avg_response_length_original': np.mean([q['response_length_original'] for q in comparison_results]),
            'avg_response_length_pruned': np.mean([q['response_length_pruned'] for q in comparison_results]),
            'avg_time_original': avg_time_original,
            'avg_time_pruned': avg_time_pruned,
            'speedup': avg_time_original / (avg_time_pruned + 1e-6),
        }

        # 保存结果
        output_file = os.path.join(output_dir, f"evaluation_{int(time.time())}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 评估完成")
        print(f"结果已保存到: {output_file}\n")
        
        # 打印摘要
        print("\n" + "="*70)
        print("评估摘要")
        print("="*70)
        print(f"问题数量: {results['statistics']['num_questions']}")
        print(f"原始LLaVA平均响应长度: {results['statistics']['avg_response_length_original']:.0f} 字符")
        print(f"剪枝LLaVA平均响应长度: {results['statistics']['avg_response_length_pruned']:.0f} 字符")
        print(f"原始LLaVA平均推理时间: {results['statistics']['avg_time_original']:.3f} 秒")
        print(f"剪枝LLaVA平均推理时间: {results['statistics']['avg_time_pruned']:.3f} 秒")
        print(f"推理加速比: {results['statistics']['speedup']:.2f}x")
        print("="*70 + "\n")

        return results


def create_argument_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='LLaVA视觉Token剪枝评估工具'
    )

    # 模型参数
    parser.add_argument("--model-path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--model-base", type=str, default=None,
                        help="基础模型路径")
    parser.add_argument("--device", type=str, default='cuda',
                        help="设备 (cuda 或 cpu)")
    parser.add_argument("--load-8bit", action="store_true",
                        help="使用8-bit量化")
    parser.add_argument("--load-4bit", action="store_true",
                        help="使用4-bit量化")

    # 评估参数
    parser.add_argument("--image-file", type=str, required=True,
                        help="测试图像文件")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                        help="输出目录")

    # 剪枝参数
    parser.add_argument("--consistency-threshold", type=float, default=0.5,
                        help="一致性阈值")
    parser.add_argument("--contrast-coefficient", type=float, default=1.0,
                        help="对比系数")
    parser.add_argument("--enable-progressive", action="store_true", default=True,
                        help="启用渐进式策略")
    parser.add_argument("--enable-drcd", action="store_true", default=True,
                        help="启用DRCD机制")

    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()

    # 创建评估器
    evaluator = VisionTokenPruningEvaluator(
        model_path=args.model_path,
        model_base=args.model_base,
        device=args.device,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
    )

    # 默认测试问题
    test_questions = [
        "What is in this picture?",
        "Please describe the main object in the picture in detail.",
        "How many people are in the picture? Count them carefully.",
        "What scene is this? Describe the overall environment.",
        "What is the background of the picture? Is it indoor or outdoor?",
        "What colors are the main objects in the image?",
        "Is there any text visible in the picture? If yes, what does it say?",
    ]
    # 执行评估
    adapter_config = {
        'consistency_threshold': args.consistency_threshold,
        'contrast_coefficient': args.contrast_coefficient,
        'enable_progressive': args.enable_progressive,
        'enable_drcd': args.enable_drcd,
    }

    results = evaluator.evaluate_hallucination_reduction(
        image_file=args.image_file,
        questions=test_questions,
        output_dir=args.output_dir,
    )
