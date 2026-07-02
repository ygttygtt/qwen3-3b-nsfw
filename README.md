<div align="center">

# 🎭 Qwen3-3B NSFW Fine-tuning

**基于 Qwen3-3B 的成人内容对话模型微调项目**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/your-username/qwen3-3b-nsfw.svg)](https://github.com/your-username/qwen3-3b-nsfw)
[![GitHub Forks](https://img.shields.io/github/forks/your-username/qwen3-3b-nsfw.svg)](https://github.com/your-username/qwen3-3b-nsfw)

[English](#english) | [中文](#中文)

</div>

---

## 📖 项目简介

本项目提供了一套完整的 **Qwen3-3B 模型微调方案**，专注于成人内容对话生成。项目包含从数据预处理、模型训练到评估推理的全流程工具链，支持 LoRA、QLoRA 和全参数微调等多种训练方式。

### ✨ 核心特性

- 🚀 **多种微调方式**：支持 LoRA、QLoRA、全参数微调
- 📊 **完整工具链**：数据预处理 → 训练 → 评估 → 推理
- 🎯 **一键启动**：提供快速启动脚本，5 分钟上手
- 📈 **实时监控**：TensorBoard 集成，实时查看训练状态
- 🐳 **Docker 支持**：提供 Dockerfile，快速部署
- 📝 **详细文档**：完整的操作手册和代码注释

---

## 🎬 效果展示

<div align="center">

### 训练过程监控

![Training Monitor](https://via.placeholder.com/800x400?text=Training+Loss+Curve)

### 交互式对话示例

```
用户: 你好，能聊聊吗？
助手: 当然可以啊，想聊什么呢？今天过得怎么样？
用户: 有点无聊，能说点有意思的吗？
助手: 那我给你讲个笑话吧，从前有个人...
```

</div>

---

## 🚀 快速开始

### 环境要求

| 项目 | 推荐配置 |
|------|----------|
| GPU | RTX 3090 (24GB) 或更高 |
| Python | 3.10+ |
| PyTorch | 2.0+ |
| CUDA | 11.8+ |

### 安装

```bash
# 克隆项目
git clone https://github.com/your-username/qwen3-3b-nsfw.git
cd qwen3-3b-nsfw

# 安装依赖
pip install -r requirements.txt

# 测试环境
python scripts/test_environment.py
```

### 一键训练

```bash
# 赋予执行权限
chmod +x quick_start.sh

# 执行完整流程（预处理 → 训练 → 评估）
./quick_start.sh all
```

### 分步执行

```bash
# 1. 数据预处理
python scripts/preprocess_data.py --input data/train.jsonl --output data --clean --split

# 2. 开始训练（LoRA）
python scripts/train.py --config configs/lora_config.yaml

# 3. 评估模型
python scripts/evaluate.py --model_path outputs/final_model --eval_data data/eval.jsonl

# 4. 交互式测试
python scripts/inference.py --model_path outputs/final_model --interactive
```

---

## 📁 项目结构

```
qwen3-3b-nsfw/
├── 📄 README.md                # 项目说明
├── 📄 TRAINING_GUIDE.md        # 详细操作手册
├── 📄 requirements.txt         # Python 依赖
├── 📄 quick_start.sh           # 快速启动脚本
├── 📁 configs/                 # 训练配置
│   ├── lora_config.yaml        # LoRA 配置
│   └── full_finetune_config.yaml
├── 📁 data/                    # 数据集
│   └── train.jsonl
├── 📁 scripts/                 # 核心脚本
│   ├── train.py                # 训练脚本
│   ├── evaluate.py             # 评估脚本
│   ├── inference.py            # 推理脚本
│   ├── preprocess_data.py      # 数据预处理
│   ├── monitor.py              # 训练监控
│   └── test_environment.py     # 环境测试
├── 📁 utils/                   # 工具函数
│   ├── data_collator.py        # 数据整理器
│   └── metrics.py              # 评估指标
└── 📁 outputs/                 # 模型输出
```

---

## 📊 训练方式对比

| 方式 | 显存需求 | 训练速度 | 效果 | 推荐场景 |
|:----:|:--------:|:--------:|:----:|----------|
| **LoRA** | 8-12GB | ⚡ 快 | ⭐⭐⭐⭐ | 显存有限，快速迭代 |
| **QLoRA** | 6-8GB | 🚶 中 | ⭐⭐⭐⭐ | 显存紧张 |
| **全参数** | 20GB+ | 🐢 慢 | ⭐⭐⭐⭐⭐ | 追求最佳效果 |

---

## ⚙️ 配置说明

### LoRA 配置示例

```yaml
# configs/lora_config.yaml
model_name: "Qwen/Qwen3-3B"
use_lora: true

lora:
  r: 8                    # LoRA 秩
  alpha: 16               # 缩放因子
  dropout: 0.05           # Dropout 比例
  target_modules:         # 应用 LoRA 的模块
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"

# 训练参数
epochs: 3
batch_size: 2
learning_rate: 0.0002
max_length: 2048
```

---

## 📈 训练监控

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir outputs --bind_all --port 6006

# 访问 http://your-ip:6006
```

### 实时监控脚本

```bash
python scripts/monitor.py --output_dir outputs --interval 5
```

---

## 🐳 Docker 部署

```bash
# 构建镜像
docker build -t qwen3-3b-nsfw .

# 运行容器
docker run --gpus all -v ./data:/app/data -v ./outputs:/app/outputs qwen3-3b-nsfw
```

---

## 📋 数据集格式

项目使用 JSONL 格式，每行包含：

```json
{
  "instruction": "用户指令",
  "input": "输入上下文（可选）",
  "output": "期望输出",
  "system": "系统提示（可选）",
  "history": []
}
```

---

## 🔧 常见问题

<details>
<summary><b>显存不足 (OOM) 怎么办？</b></summary>

1. 减小批大小：`batch_size: 1`
2. 启用梯度检查点：`gradient_checkpointing: true`
3. 使用 QLoRA：`use_quantization: true`
4. 减小序列长度：`max_length: 1024`

</details>

<details>
<summary><b>训练 loss 不下降怎么办？</b></summary>

1. 调整学习率（尝试 1e-4 或 5e-5）
2. 检查数据质量
3. 增加训练轮数
4. 减小批大小

</details>

<details>
<summary><b>依赖安装失败怎么办？</b></summary>

```bash
# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 升级 pip
pip install --upgrade pip
```

</details>

---

## 📚 文档

- 📖 [详细操作手册](TRAINING_GUIDE.md) - 完整的训练流程指南
- 📝 [项目技术文档](CLAUDE.md) - 代码架构和设计说明
- 🤝 [贡献指南](CONTRIBUTING.md) - 如何参与项目贡献
- 🔒 [安全政策](SECURITY.md) - 安全漏洞报告

---

## 🛠️ 技术栈

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/TransformERS-4.35+-yellow?style=for-the-badge&logo=huggingface&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

</div

---

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

## 🙏 致谢

- [Qwen](https://github.com/QwenLM/Qwen) - 基础模型
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - 模型框架
- [PEFT](https://github.com/huggingface/peft) - 参数高效微调库

---

## 📮 联系方式

- GitHub: [@your-username](https://github.com/your-username)
- Email: your-email@example.com

---

<div align="center">

**如果觉得有用，请给个 ⭐ Star 支持一下！**

</div>

---

<a name="english"></a>

## 🇬🇧 English

### Introduction

This project provides a complete **Qwen3-3B model fine-tuning solution** for adult content dialogue generation. It includes the entire toolchain from data preprocessing, model training to evaluation and inference, supporting multiple training methods such as LoRA, QLoRA, and full parameter fine-tuning.

### Key Features

- 🚀 **Multiple Fine-tuning Methods**: LoRA, QLoRA, Full Parameter
- 📊 **Complete Toolchain**: Preprocessing → Training → Evaluation → Inference
- 🎯 **Quick Start**: Get started in 5 minutes
- 📈 **Real-time Monitoring**: TensorBoard integration
- 🐳 **Docker Support**: Easy deployment
- 📝 **Detailed Documentation**: Complete operation manual

### Quick Start

```bash
# Clone project
git clone https://github.com/your-username/qwen3-3b-nsfw.git
cd qwen3-3b-nsfw

# Install dependencies
pip install -r requirements.txt

# One-click training
chmod +x quick_start.sh
./quick_start.sh all
```

### Documentation

- [Training Guide](TRAINING_GUIDE.md) - Complete training workflow
- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [Security Policy](SECURITY.md) - Report security vulnerabilities

---

<div align="center">

**⭐ Star this repo if you find it useful!**

</div>