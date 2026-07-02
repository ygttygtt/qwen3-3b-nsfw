#!/usr/bin/env python3
"""
Qwen3-3B NSFW 微调训练脚本
支持 LoRA、QLoRA 和全参数微调
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_dataset_from_jsonl(data_path: str) -> Dataset:
    """从 JSONL 文件加载数据集"""
    logger.info(f"Loading dataset from {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")
    logger.info(f"Loaded {len(dataset)} examples")
    return dataset


def preprocess_data(
    dataset: Dataset,
    tokenizer,
    max_length: int = 2048,
) -> Dataset:
    """预处理数据集，转换为模型输入格式"""

    def tokenize_function(examples):
        # 构建对话格式
        texts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples["input"][i] if examples["input"][i] else ""
            output_text = examples["output"][i]
            system = examples["system"][i] if examples["system"][i] else ""

            # 构建 Qwen 对话格式
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            if input_text:
                messages.append({"role": "user", "content": f"{instruction}\n{input_text}"})
            else:
                messages.append({"role": "user", "content": instruction})
            messages.append({"role": "assistant", "content": output_text})

            # 应用聊天模板
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        # 设置 labels（与 input_ids 相同，用于因果语言建模）
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # 应用 tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    return tokenized_dataset


def create_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """创建 LoRA 配置"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.get("r", 8),
        lora_alpha=config.get("alpha", 16),
        lora_dropout=config.get("dropout", 0.05),
        target_modules=config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        bias="none",
    )
    return lora_config


def create_quantization_config(config: Dict[str, Any]) -> BitsAndBytesConfig:
    """创建量化配置（用于 QLoRA）"""
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    return quant_config


def main():
    parser = argparse.ArgumentParser(description="Qwen3-3B NSFW 微调训练")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="训练配置文件路径",
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    logger.info(f"Loaded config: {config}")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 加载 tokenizer
    model_name = config.get("model_name", "Qwen/Qwen3-3B")
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    # 确保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    logger.info(f"Loading model: {model_name}")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    # 如果使用量化（QLoRA）
    if config.get("use_quantization", False):
        logger.info("Using 4-bit quantization (QLoRA)")
        model_kwargs["quantization_config"] = create_quantization_config(config)

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # 如果使用量化，准备模型
    if config.get("use_quantization", False):
        model = prepare_model_for_kbit_training(model)

    # 如果使用 LoRA
    if config.get("use_lora", True):
        logger.info("Applying LoRA")
        lora_config = create_lora_config(config.get("lora", {}))
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 加载数据集
    data_path = config.get("data_path", "data/train.jsonl")
    dataset = load_dataset_from_jsonl(data_path)

    # 预处理数据集
    max_length = config.get("max_length", 2048)
    tokenized_dataset = preprocess_data(dataset, tokenizer, max_length)

    # 划分训练集和验证集
    split_ratio = config.get("eval_split", 0.1)
    if split_ratio > 0:
        split_dataset = tokenized_dataset.train_test_split(test_size=split_ratio, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None

    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

    # 设置训练参数
    output_dir = config.get("output_dir", "outputs")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 2),
        per_device_eval_batch_size=config.get("eval_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 2e-4),
        weight_decay=config.get("weight_decay", 0.01),
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        lr_scheduler_type=config.get("scheduler", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 100),
        save_total_limit=config.get("save_total_limit", 3),
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=config.get("eval_steps", 100) if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        bf16=True,
        tf32=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=config.get("report_to", "tensorboard"),
        run_name=config.get("run_name", "qwen3-3b-nsfw"),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 开始训练
    logger.info("Starting training...")
    trainer.train()

    # 保存最终模型
    final_model_path = os.path.join(output_dir, "final_model")
    logger.info(f"Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # 如果使用 LoRA，保存 LoRA 权重
    if config.get("use_lora", True):
        lora_path = os.path.join(output_dir, "lora_weights")
        logger.info(f"Saving LoRA weights to {lora_path}")
        model.save_pretrained(lora_path)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()