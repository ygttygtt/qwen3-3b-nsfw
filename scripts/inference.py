#!/usr/bin/env python3
"""
Qwen3-3B NSFW 推理脚本
支持交互式对话和批量推理
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(
    model_path: str,
    lora_path: Optional[str] = None,
    use_quantization: bool = False,
    device: str = "auto",
):
    """加载模型和 tokenizer"""
    logger.info(f"Loading model from {model_path}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 模型加载参数
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": device,
    }

    # 量化配置
    if use_quantization:
        logger.info("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    # 如果有 LoRA 权重，加载它们
    if lora_path and os.path.exists(lora_path):
        logger.info(f"Loading LoRA weights from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # 合并 LoRA 权重

    model.eval()
    logger.info("Model loaded successfully")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> str:
    """生成回复"""
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 解码生成的文本
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response


def interactive_mode(
    model,
    tokenizer,
    system_prompt: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
):
    """交互式对话模式"""
    print("\n" + "="*50)
    print("Qwen3-3B NSFW 交互式对话")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清空对话历史")
    print("="*50 + "\n")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        print(f"系统提示: {system_prompt}\n")

    while True:
        try:
            # 获取用户输入
            user_input = input("你: ").strip()

            # 检查退出命令
            if user_input.lower() in ["quit", "exit", "q"]:
                print("再见！")
                break

            # 检查清空命令
            if user_input.lower() == "clear":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                print("对话历史已清空\n")
                continue

            # 跳过空输入
            if not user_input:
                continue

            # 添加用户消息
            messages.append({"role": "user", "content": user_input})

            # 生成回复
            print("AI: ", end="", flush=True)
            response = generate_response(
                model,
                tokenizer,
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            print(response)

            # 添加助手回复到历史
            messages.append({"role": "assistant", "content": response})
            print()

        except KeyboardInterrupt:
            print("\n\n对话被中断。再见！")
            break
        except Exception as e:
            logger.error(f"生成回复时出错: {e}")
            print("抱歉，生成回复时出现错误。请重试。")


def batch_inference(
    model,
    tokenizer,
    input_file: str,
    output_file: str,
    system_prompt: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
):
    """批量推理"""
    import json

    logger.info(f"Processing batch inference from {input_file}")

    # 读取输入文件
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []

    for i, line in enumerate(lines):
        try:
            data = json.loads(line.strip())
            instruction = data.get("instruction", "")
            input_text = data.get("input", "")

            # 构建消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            if input_text:
                messages.append({"role": "user", "content": f"{instruction}\n{input_text}"})
            else:
                messages.append({"role": "user", "content": instruction})

            # 生成回复
            response = generate_response(
                model,
                tokenizer,
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            # 保存结果
            result = {
                "instruction": instruction,
                "input": input_text,
                "generated_output": response,
                "original_output": data.get("output", ""),
            }
            results.append(result)

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(lines)} examples")

        except Exception as e:
            logger.error(f"Error processing line {i}: {e}")
            continue

    # 保存结果到文件
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info(f"Batch inference completed. Results saved to {output_file}")
    logger.info(f"Processed {len(results)} examples successfully")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-3B NSFW 推理脚本")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型路径",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="LoRA 权重路径",
    )
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        help="使用 4-bit 量化",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互式对话模式",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="批量推理的输入文件",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="inference_results.jsonl",
        help="批量推理的输出文件",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="",
        help="系统提示",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成温度",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备",
    )

    args = parser.parse_args()

    # 加载模型
    model, tokenizer = load_model(
        args.model_path,
        args.lora_path,
        args.use_quantization,
        args.device,
    )

    # 运行推理
    if args.interactive:
        interactive_mode(
            model,
            tokenizer,
            args.system_prompt,
            args.max_new_tokens,
            args.temperature,
        )
    elif args.input_file:
        batch_inference(
            model,
            tokenizer,
            args.input_file,
            args.output_file,
            args.system_prompt,
            args.max_new_tokens,
            args.temperature,
        )
    else:
        print("请指定 --interactive 或 --input_file 参数")
        sys.exit(1)


if __name__ == "__main__":
    main()