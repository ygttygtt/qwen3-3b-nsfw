# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

本项目用于在 Auto DL 云平台上微调 Qwen3-3B 模型，目标是将其训练为支持成人内容（NSFW）的对话模型。

**核心文档：**
- `CLAUDE.md` - 项目指导文件
- `README.md` - 项目说明
- `TRAINING_GUIDE.md` - 详细操作手册（必读）

数据集来源于 `E:\YGTT_Project\Create_datasets\output\train_qwen3_3b_nsfw.jsonl`。

## 数据集

- **格式**: JSONL，每行一个 JSON 对象
- **字段**: `instruction` (用户指令), `input` (输入上下文，通常为空), `output` (模型回复), `system` (系统提示), `history` (对话历史，通常为空数组)
- **内容**: 中文成人向对话数据，包含显性词汇
- **位置**: 原始数据集位于 `E:\YGTT_Project\Create_datasets\output\train_qwen3_3b_nsfw.jsonl`，已复制到本项目的 `data/train.jsonl`

## 环境要求

- **平台**: Auto DL 云 GPU 实例（推荐 RTX 3090 或更高）
- **Python**: 3.10+
- **PyTorch**: 2.0+ (CUDA 11.8)
- **核心库**:
  - `transformers` (Hugging Face)
  - `peft` (参数高效微调)
  - `datasets` (数据集加载)
  - `accelerate` (分布式训练)
  - `bitsandbytes` (量化训练，可选)

## 常用命令

### 环境设置
```bash
# 安装依赖
pip install -r requirements.txt

# 登录 Hugging Face (如需下载模型)
huggingface-cli login
```

### 一键启动脚本
```bash
# 赋予执行权限
chmod +x scripts/run.sh quick_start.sh

# 查看帮助
./scripts/run.sh help

# 测试环境是否满足要求
./scripts/run.sh test

# 执行完整流程 (预处理 -> LoRA训练 -> 评估)
./scripts/run.sh all

# 快速启动（推荐新手使用）
./quick_start.sh all

# 单独执行各个步骤
./scripts/run.sh preprocess
./scripts/run.sh train-lora
./scripts/run.sh evaluate
./scripts/run.sh inference

# 训练监控
python scripts/monitor.py --output_dir outputs --interval 5
```

### 数据预处理
```bash
# 将 JSONL 转换为训练格式
python scripts/preprocess_data.py --input data/train.jsonl --output data --clean --split --eval_ratio 0.1
```

### 训练
```bash
# 使用 LoRA 微调
python scripts/train.py --config configs/lora_config.yaml

# 使用全参数微调
python scripts/train.py --config configs/full_finetune_config.yaml
```

### 评估
```bash
# 评估模型性能
python scripts/evaluate.py --model_path outputs/checkpoint-XXX --eval_data data/eval.jsonl
```

### 推理
```bash
# 交互式测试
python scripts/inference.py --model_path outputs/final_model --interactive
```

## 项目结构

```
Qwen3_3B_FineTuningNSFW/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── configs/                 # 训练配置文件
│   ├── lora_config.yaml
│   └── full_finetune_config.yaml
├── data/                   # 数据集目录
│   ├── train.jsonl
│   └── eval.jsonl
├── scripts/                # 核心脚本
│   ├── preprocess_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── run.sh             # 一键启动脚本
├── outputs/                # 模型输出目录
└── utils/                  # 工具函数
    ├── __init__.py
    ├── data_collator.py
    └── metrics.py
```

## 微调策略

### 推荐方法: LoRA (Low-Rank Adaptation)
- **优势**: 显存占用低，训练速度快，适合单 GPU
- **配置**: rank=8, alpha=16, dropout=0.05
- **目标模块**: `q_proj`, `v_proj`, `k_proj`, `o_proj`

### 备选方法: QLoRA (4-bit 量化 + LoRA)
- **适用场景**: 显存不足时
- **配置**: 4-bit NF4 量化，双重量化

### 全参数微调
- **适用场景**: 有足够显存（≥24GB）且追求最佳效果
- **注意**: 学习率需调低（1e-5 ~ 5e-6）

## 训练超参数建议

- **学习率**: LoRA: 2e-4, 全参数: 1e-5
- **批大小**: 根据显存调整，建议 gradient accumulation 步数为 4
- **轮数**: 3-5 epochs
- **序列长度**: 2048 tokens（根据数据集调整）
- **优化器**: AdamW (weight_decay=0.01)
- **调度器**: Cosine annealing with warmup

## 注意事项

1. **NSFW 内容**: 数据集包含成人内容，确保在合规环境下使用
2. **显存管理**: 使用 gradient checkpointing 和混合精度训练
3. **模型保存**: 定期保存 checkpoint，避免训练中断丢失进度
4. **评估指标**: 关注 loss 下降趋势和生成质量
5. **安全**: 微调后的模型仅用于合法用途，遵守当地法律法规