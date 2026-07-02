#!/usr/bin/env python3
"""
数据预处理脚本
用于清洗、验证和转换数据集
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any
from collections import Counter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                if line.strip():
                    data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """保存数据到 JSONL 文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def validate_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """验证数据质量"""
    logger.info("Validating data quality...")

    stats = {
        "total_examples": len(data),
        "missing_fields": Counter(),
        "empty_fields": Counter(),
        "length_stats": {
            "instruction": [],
            "output": [],
        },
    }

    required_fields = ["instruction", "output"]

    for i, item in enumerate(data):
        # 检查必需字段
        for field in required_fields:
            if field not in item:
                stats["missing_fields"][field] += 1
            elif not item[field]:
                stats["empty_fields"][field] += 1
            else:
                # 统计长度
                if field in stats["length_stats"]:
                    stats["length_stats"][field].append(len(str(item[field])))

    # 计算长度统计
    for field in stats["length_stats"]:
        lengths = stats["length_stats"][field]
        if lengths:
            stats["length_stats"][field] = {
                "min": min(lengths),
                "max": max(lengths),
                "avg": sum(lengths) / len(lengths),
                "median": sorted(lengths)[len(lengths) // 2],
            }
        else:
            stats["length_stats"][field] = {}

    return stats


def clean_data(
    data: List[Dict[str, Any]],
    min_instruction_length: int = 5,
    min_output_length: int = 10,
    max_instruction_length: int = 2000,
    max_output_length: int = 5000,
) -> List[Dict[str, Any]]:
    """清洗数据"""
    logger.info("Cleaning data...")

    cleaned_data = []
    removed_count = 0

    for item in data:
        # 检查必需字段
        if "instruction" not in item or "output" not in item:
            removed_count += 1
            continue

        instruction = str(item["instruction"]).strip()
        output = str(item["output"]).strip()

        # 检查长度
        if len(instruction) < min_instruction_length:
            removed_count += 1
            continue

        if len(instruction) > max_instruction_length:
            removed_count += 1
            continue

        if len(output) < min_output_length:
            removed_count += 1
            continue

        if len(output) > max_output_length:
            removed_count += 1
            continue

        # 清洗字段
        cleaned_item = {
            "instruction": instruction,
            "input": str(item.get("input", "")).strip(),
            "output": output,
            "system": str(item.get("system", "")).strip(),
            "history": item.get("history", []),
        }

        cleaned_data.append(cleaned_item)

    logger.info(f"Removed {removed_count} examples during cleaning")
    logger.info(f"Remaining examples: {len(cleaned_data)}")

    return cleaned_data


def split_data(
    data: List[Dict[str, Any]],
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    """划分训练集和验证集"""
    import random

    random.seed(seed)
    random.shuffle(data)

    split_index = int(len(data) * (1 - eval_ratio))
    train_data = data[:split_index]
    eval_data = data[split_index:]

    logger.info(f"Split data: {len(train_data)} train, {len(eval_data)} eval")

    return train_data, eval_data


def analyze_content(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析数据集内容"""
    logger.info("Analyzing dataset content...")

    analysis = {
        "system_prompts": Counter(),
        "instruction_patterns": Counter(),
        "output_lengths": [],
    }

    for item in data:
        # 统计系统提示
        system = item.get("system", "")
        if system:
            analysis["system_prompts"][system] += 1

        # 分析指令模式
        instruction = item.get("instruction", "")
        if "请创作" in instruction or "生成" in instruction:
            analysis["instruction_patterns"]["创作类"] += 1
        elif "对话" in instruction or "聊天" in instruction:
            analysis["instruction_patterns"]["对话类"] += 1
        else:
            analysis["instruction_patterns"]["其他"] += 1

        # 输出长度
        output_length = len(str(item.get("output", "")))
        analysis["output_lengths"].append(output_length)

    # 计算输出长度统计
    if analysis["output_lengths"]:
        lengths = analysis["output_lengths"]
        analysis["output_length_stats"] = {
            "min": min(lengths),
            "max": max(lengths),
            "avg": sum(lengths) / len(lengths),
            "median": sorted(lengths)[len(lengths) // 2],
        }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="数据预处理脚本")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 JSONL 文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录路径",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="清洗数据",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="划分训练集和验证集",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.1,
        help="验证集比例",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="分析数据集内容",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="验证数据质量",
    )

    args = parser.parse_args()

    # 加载数据
    logger.info(f"Loading data from {args.input}")
    data = load_jsonl(args.input)
    logger.info(f"Loaded {len(data)} examples")

    # 验证数据
    if args.validate:
        stats = validate_data(data)
        print("\n" + "="*50)
        print("数据质量报告")
        print("="*50)
        print(f"总样本数: {stats['total_examples']}")
        print(f"缺失字段: {dict(stats['missing_fields'])}")
        print(f"空字段: {dict(stats['empty_fields'])}")
        print(f"长度统计: {stats['length_stats']}")
        print("="*50 + "\n")

    # 清洗数据
    if args.clean:
        data = clean_data(data)

    # 分析内容
    if args.analyze:
        analysis = analyze_content(data)
        print("\n" + "="*50)
        print("数据分析报告")
        print("="*50)
        print(f"系统提示分布: {dict(analysis['system_prompts'].most_common(5))}")
        print(f"指令模式分布: {dict(analysis['instruction_patterns'])}")
        if "output_length_stats" in analysis:
            print(f"输出长度统计: {analysis['output_length_stats']}")
        print("="*50 + "\n")

    # 划分数据
    if args.split:
        train_data, eval_data = split_data(data, args.eval_ratio)

        # 保存数据
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            train_path = os.path.join(args.output, "train.jsonl")
            eval_path = os.path.join(args.output, "eval.jsonl")

            save_jsonl(train_data, train_path)
            save_jsonl(eval_data, eval_path)

            logger.info(f"Saved train data to {train_path}")
            logger.info(f"Saved eval data to {eval_path}")
        else:
            # 保存到当前目录
            save_jsonl(train_data, "train.jsonl")
            save_jsonl(eval_data, "eval.jsonl")

            logger.info("Saved train data to train.jsonl")
            logger.info("Saved eval data to eval.jsonl")
    elif args.output:
        # 保存清洗后的数据
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, "processed.jsonl")
        save_jsonl(data, output_path)
        logger.info(f"Saved processed data to {output_path}")

    logger.info("Data preprocessing completed!")


if __name__ == "__main__":
    main()