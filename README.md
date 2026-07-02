# Qwen3-3B NSFW Fine-tuning

基于 Qwen3-3B 的成人内容对话模型微调项目

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 项目简介

本项目提供 Qwen3-3B 模型的完整微调方案，支持 LoRA、QLoRA 和全参数微调。包含数据预处理、模型训练、评估推理的全流程工具链。

## 环境要求

| 项目 | 推荐配置 |
|------|----------|
| GPU | RTX 3090 (24GB) 或更高 |
| Python | 3.10+ |
| PyTorch | 2.0+ |
| CUDA | 11.8+ |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 测试环境
python scripts/test_environment.py

# 数据预处理
python scripts/preprocess_data.py --input data/train.jsonl --output data --clean --split

# 开始训练（LoRA）
python scripts/train.py --config configs/lora_config.yaml

# 交互式测试
python scripts/inference.py --model_path outputs/final_model --interactive
```

## 训练方式

| 方式 | 显存需求 | 适用场景 |
|------|----------|----------|
| LoRA | 8-12GB | 快速迭代，显存有限 |
| QLoRA | 6-8GB | 显存紧张 |
| 全参数 | 20GB+ | 追求最佳效果 |

## 项目结构

```
qwen3-3b-nsfw/
├── configs/              # 训练配置
│   ├── lora_config.yaml
│   └── full_finetune_config.yaml
├── data/                 # 数据集
│   └── train.jsonl
├── scripts/              # 核心脚本
│   ├── train.py          # 训练
│   ├── evaluate.py       # 评估
│   ├── inference.py      # 推理
│   ├── preprocess_data.py # 数据预处理
│   └── monitor.py        # 训练监控
├── utils/                # 工具函数
├── outputs/              # 模型输出
├── TRAINING_GUIDE.md     # 详细操作手册
└── README.md
```

## 核心功能

- **数据预处理**: 数据清洗、格式转换、训练集/验证集划分
- **模型训练**: 支持 LoRA/QLoRA/全参数微调，梯度检查点，混合精度训练
- **评估推理**: 多指标评估，交互式对话，批量推理
- **训练监控**: TensorBoard 集成，实时 GPU 监控

## 配置说明

LoRA 配置示例 (`configs/lora_config.yaml`):

```yaml
model_name: "Qwen/Qwen3-3B"
use_lora: true

lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

epochs: 3
batch_size: 2
learning_rate: 0.0002
max_length: 2048
```

## 数据格式

JSONL 格式，每行包含:

```json
{
  "instruction": "用户指令",
  "input": "输入上下文（可选）",
  "output": "期望输出",
  "system": "系统提示（可选）",
  "history": []
}
```

## 常见问题

**显存不足**: 减小 `batch_size`，启用 `gradient_checkpointing`，或使用 QLoRA

**训练 loss 不下降**: 调整学习率，检查数据质量，增加训练轮数

**依赖安装失败**: 使用国内镜像 `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

## 文档

- [详细操作手册](TRAINING_GUIDE.md) - 完整的训练流程指南
- [贡献指南](CONTRIBUTING.md) - 如何参与贡献
- [安全政策](SECURITY.md) - 安全漏洞报告

## 许可证

[MIT License](LICENSE)