#!/usr/bin/env python3
"""
Qwen3-3B NSFW 模型评估脚本
评估微调后模型的性能
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any

import torch
from datasets import load_dataset
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
    lora_path: str = None,
    use_quantization: bool = False,
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
        "device_map": "auto",
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
        model = model.merge_and_unload()

    model.eval()
    logger.info("Model loaded successfully")

    return model, tokenizer


def evaluate_model(
    model,
    tokenizer,
    eval_data: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """评估模型性能"""
    logger.info(f"Evaluating model on {len(eval_data)} examples")

    results = []
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for i, example in enumerate(eval_data):
        try:
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            expected_output = example.get("output", "")
            system = example.get("system", "")

            # 构建消息
            messages = []
            if system:
                messages.append({"role": "system", "content": system})

            if input_text:
                messages.append({"role": "user", "content": f"{instruction}\n{input_text}"})
            else:
                messages.append({"role": "user", "content": instruction})

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
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # 解码生成的文本
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            generated_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # 计算相似度（简单的字符串匹配）
            is_correct = generated_output.strip() == expected_output.strip()
            if is_correct:
                correct_predictions += 1
            total_predictions += 1

            # 保存结果
            result = {
                "instruction": instruction,
                "input": input_text,
                "expected_output": expected_output,
                "generated_output": generated_output,
                "is_correct": is_correct,
            }
            results.append(result)

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(eval_data)} examples")

        except Exception as e:
            logger.error(f"Error processing example {i}: {e}")
            continue

    # 计算指标
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    metrics = {
        "total_examples": len(eval_data),
        "processed_examples": total_predictions,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
    }

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")

    return {
        "metrics": metrics,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen3-3B NSFW 模型评估")
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
        "--eval_data",
        type=str,
        required=True,
        help="评估数据文件路径（JSONL 格式）",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="评估结果输出文件",
    )
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        help="使用 4-bit 量化",
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

    args = parser.parse_args()

    # 加载模型
    model, tokenizer = load_model(
        args.model_path,
        args.lora_path,
        args.use_quantization,
    )

    # 加载评估数据
    logger.info(f"Loading evaluation data from {args.eval_data}")
    eval_dataset = load_dataset("json", data_files=args.eval_data, split="train")
    eval_data = [example for example in eval_dataset]

    # 评估模型
    evaluation_results = evaluate_model(
        model,
        tokenizer,
        eval_data,
        args.max_new_tokens,
        args.temperature,
    )

    # 保存结果
    logger.info(f"Saving evaluation results to {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    # 打印摘要
    metrics = evaluation_results["metrics"]
    print("\n" + "="*50)
    print("评估结果摘要")
    print("="*50)
    print(f"总样本数: {metrics['total_examples']}")
    print(f"处理样本数: {metrics['processed_examples']}")
    print(f"正确预测数: {metrics['correct_predictions']}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()