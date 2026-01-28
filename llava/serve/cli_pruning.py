"""
LLaVA CLI with Vision Reasoning-Coupled Token Pruning
带有推理耦合式视觉Token剪枝的LLaVA命令行推理工具
"""

import argparse
import torch
import os
import sys

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.model.reasoning_pruning_adapter import LLaVAReasoningPruningAdapter, integrate_pruning_into_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    """加载图像文件或URL"""
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # 模型加载
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, 
        args.load_8bit, args.load_4bit, device=args.device
    )

    # 集成推理感知的视觉Token剪枝
    if args.enable_pruning:
        adapter_config = {
            'hidden_dim': model.config.hidden_size,
            'num_layers_for_attention': args.num_attention_layers,
            'momentum': args.momentum,
            'consistency_threshold': args.consistency_threshold,
            'contrast_coefficient': args.contrast_coefficient,
            'max_decoding_steps': args.max_steps,
            'enable_progressive': args.enable_progressive,
            'enable_drcd': args.enable_drcd,
        }
        model, pruning_adapter = integrate_pruning_into_model(model, adapter_config)
        print(f"\n✓ 已启用视觉Token剪枝适配器")
        print(f"  - 一致性阈值: {args.consistency_threshold}")
        print(f"  - 对比系数: {args.contrast_coefficient}")
        print(f"  - 渐进式策略: {args.enable_progressive}")
        print(f"  - DRCD机制: {args.enable_drcd}\n")
    else:
        pruning_adapter = None
        print("\n⚠ 使用原始LLaVA（未启用Token剪枝）\n")

    # 推理模式选择
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(f'[WARNING] 推断的会话模式是 {conv_mode}，而 --conv-mode 是 {args.conv_mode}，使用 {args.conv_mode}')
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    # 图像加载和处理
    image = load_image(args.image_file)
    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    print(f"图像已加载: {args.image_file}")
    print(f"模型: {model_name}")
    print(f"推理模式: {args.conv_mode}")
    print("\n输入您的问题 (输入 'exit' 退出，'stats' 显示剪枝统计信息):\n")

    # 交互循环
    question_count = 0
    while True:
        try:
            inp = input(f"{roles[0]}: ").strip()
        except EOFError:
            inp = ""
        
        if not inp:
            continue
        
        if inp.lower() == 'exit':
            print("再见！")
            break
        
        if inp.lower() == 'stats':
            if pruning_adapter is not None:
                pruning_adapter.print_pruning_summary()
            else:
                print("未启用Token剪枝，无统计信息")
            continue

        print(f"{roles[1]}: ", end="", flush=True)

        # 重置剪枝适配器状态（新生成过程）
        if pruning_adapter is not None:
            pruning_adapter.reset()

        if image is not None:
            # 第一条消息，包含图像
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            image = None
        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(model.device)
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # 生成
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                streamer=streamer,
            )

        # 解码输出
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        print()

        # 准备下一轮
        question_count += 1
        conv.messages[-1][-1] = outputs[0]


def create_argument_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='LLaVA推理工具 - 支持视觉Token剪枝'
    )

    # 模型相关参数
    parser.add_argument("--model-path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--model-base", type=str, default=None,
                        help="基础模型路径")
    parser.add_argument("--image-file", type=str, required=True,
                        help="输入图像文件路径")
    parser.add_argument("--device", type=str, default='cuda',
                        help="设备 (cuda 或 cpu)")
    parser.add_argument("--conv-mode", type=str, default=None,
                        help="会话模式")
    
    # 量化相关参数
    parser.add_argument("--load-8bit", action="store_true",
                        help="使用8-bit量化")
    parser.add_argument("--load-4bit", action="store_true",
                        help="使用4-bit量化")

    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="生成温度")
    parser.add_argument("--top-p", type=float, default=None,
                        help="nucleus采样的top-p")
    parser.add_argument("--num-beams", type=int, default=1,
                        help="beam搜索的beam数")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="最多生成的新token数")

    # 剪枝相关参数
    parser.add_argument("--enable-pruning", action="store_true", default=True,
                        help="启用视觉Token剪枝")
    parser.add_argument("--disable-pruning", dest="enable_pruning", action="store_false",
                        help="禁用视觉Token剪枝（原始LLaVA）")
    parser.add_argument("--num-attention-layers", type=int, default=3,
                        help="用于attention聚合的高层层数")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="一致性累计的动量系数")
    parser.add_argument("--consistency-threshold", type=float, default=0.5,
                        help="初始一致性阈值")
    parser.add_argument("--contrast-coefficient", type=float, default=1.0,
                        help="对比系数λ")
    parser.add_argument("--enable-progressive", action="store_true", default=True,
                        help="启用渐进式动态重构")
    parser.add_argument("--disable-progressive", dest="enable_progressive", action="store_false",
                        help="禁用渐进式策略")
    parser.add_argument("--enable-drcd", action="store_true", default=True,
                        help="启用DRCD机制")
    parser.add_argument("--disable-drcd", dest="enable_drcd", action="store_false",
                        help="禁用DRCD")
    parser.add_argument("--max-steps", type=int, default=128,
                        help="最大解码步数")

    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    main(args)
