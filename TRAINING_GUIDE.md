# Qwen3-3B NSFW 微调完整操作手册

## 📋 目录

1. [环境准备](#1-环境准备)
2. [项目上传](#2-项目上传)
3. [环境配置](#3-环境配置)
4. [数据准备](#4-数据准备)
5. [训练流程](#5-训练流程)
6. [评估与推理](#6-评估与推理)
7. [常见问题](#7-常见问题)
8. [高级配置](#8-高级配置)

---

## 1. 环境准备

### 1.1 Auto DL 实例配置

**推荐配置：**
- **GPU**: RTX 3090 (24GB) 或 RTX 4090 (24GB)
- **内存**: 32GB+
- **磁盘**: 50GB+
- **镜像**: PyTorch 2.0+ / CUDA 11.8

**创建实例步骤：**
1. 登录 Auto DL 官网
2. 选择「容器实例」→「创建」
3. 选择 GPU 型号（推荐 RTX 3090）
4. 选择镜像：`PyTorch 2.0.1` → `CUDA 11.8` → `Python 3.10`
5. 点击「创建并开机」

### 1.2 获取实例信息

创建完成后，记录以下信息：
- **登录指令**: `ssh -p 端口号 root@区域地址`
- **密码**: 实例密码
- **Jupyter 地址**: 用于文件管理

---

## 2. 项目上传

### 2.1 方法一：Jupyter 上传（推荐）

1. 打开 Auto DL 的 Jupyter 界面
2. 进入 `/root` 目录
3. 上传整个 `Qwen3_3B_FineTuningNSFW` 文件夹
4. 等待上传完成

### 2.2 方法二：SCP 上传

```bash
# 在本地执行
scp -P 端口号 -r Qwen3_3B_FineTuningNSFW root@区域地址:/root/
```

### 2.3 方法三：Git 克隆

```bash
# 在 Auto DL 实例上执行
cd /root
git clone https://github.com/your-username/qwen3-3b-nsfw.git
mv qwen3-3b-nsfw Qwen3_3B_FineTuningNSFW
```

---

## 3. 环境配置

### 3.1 连接到实例

```bash
# 使用 SSH 连接
ssh -p 端口号 root@区域地址

# 或使用 Auto DL 的 Web Terminal
```

### 3.2 进入项目目录

```bash
cd /root/Qwen3_3B_FineTuningNSFW
```

### 3.3 测试环境

```bash
# 检查 Python 环境
python --version

# 检查 CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 运行环境测试脚本
python scripts/test_environment.py
```

### 3.4 安装依赖

```bash
# 升级 pip
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt

# 如果遇到网络问题，使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.5 登录 Hugging Face（可选）

如果需要下载模型：

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 登录（需要访问令牌）
huggingface-cli login
# 输入你的 HF token
```

---

## 4. 数据准备

### 4.1 数据集位置

数据集已位于：`data/train.jsonl`

**数据格式：**
```json
{
  "instruction": "用户指令",
  "input": "输入上下文（可选）",
  "output": "期望输出",
  "system": "系统提示（可选）",
  "history": []
}
```

### 4.2 数据预处理

```bash
# 方法一：使用预处理脚本
python scripts/preprocess_data.py \
    --input data/train.jsonl \
    --output data \
    --clean \
    --split \
    --eval_ratio 0.1

# 方法二：使用 Makefile
make preprocess

# 方法三：使用一键脚本
./scripts/run.sh preprocess
```

**预处理说明：**
- `--clean`: 清洗数据（移除空指令、过短/过长文本）
- `--split`: 划分训练集和验证集
- `--eval_ratio 0.1`: 10% 数据作为验证集

### 4.3 验证数据

```bash
# 检查数据文件
ls -la data/

# 查看数据样本
head -n 5 data/train.jsonl

# 统计数据条数
wc -l data/train.jsonl
```

---

## 5. 训练流程

### 5.1 训练方式选择

| 方式 | 显存需求 | 训练速度 | 效果 | 推荐场景 |
|------|----------|----------|------|----------|
| **LoRA** | 8-12GB | 快 | 好 | 显存有限，快速迭代 |
| **QLoRA** | 6-8GB | 中 | 好 | 显存紧张 |
| **全参数** | 20GB+ | 慢 | 最好 | 追求最佳效果 |

### 5.2 LoRA 微调（推荐）

```bash
# 方法一：直接运行
python scripts/train.py --config configs/lora_config.yaml

# 方法二：使用 Makefile
make train-lora

# 方法三：使用一键脚本
./scripts/run.sh train-lora
```

**LoRA 配置说明：**
```yaml
# configs/lora_config.yaml
lora:
  r: 8              # LoRA 秩（越大效果越好，显存占用越高）
  alpha: 16         # 缩放因子（通常为 r 的 2 倍）
  dropout: 0.05     # Dropout 比例
  target_modules:   # 应用 LoRA 的模块
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
```

### 5.3 全参数微调

```bash
# 方法一：直接运行
python scripts/train.py --config configs/full_finetune_config.yaml

# 方法二：使用 Makefile
make train-full

# 方法三：使用一键脚本
./scripts/run.sh train-full
```

**全参数微调注意事项：**
- 需要 24GB+ 显存
- 学习率调低至 1e-5 ~ 5e-6
- 训练时间较长

### 5.4 训练过程监控

**TensorBoard 监控：**

```bash
# 启动 TensorBoard
tensorboard --logdir outputs --bind_all --port 6006

# 在浏览器中访问
# http://实例IP:6006
```

**实时查看日志：**

```bash
# 查看训练日志
tail -f outputs/train.log

# 查看 GPU 使用情况
watch -n 1 nvidia-smi
```

### 5.5 训练参数调优

**学习率调整：**
```yaml
# LoRA 微调
learning_rate: 0.0002  # 2e-4

# 全参数微调
learning_rate: 0.00001  # 1e-5
```

**批大小调整：**
```yaml
# 根据显存调整
batch_size: 2  # 每个 GPU 的批大小
gradient_accumulation_steps: 4  # 梯度累积步数
# 有效批大小 = batch_size * gradient_accumulation_steps
```

**训练轮数：**
```yaml
epochs: 3  # 通常 3-5 轮即可
```

### 5.6 训练完成

训练完成后，模型保存在：
- **最终模型**: `outputs/final_model/`
- **LoRA 权重**: `outputs/lora_weights/`
- **检查点**: `outputs/checkpoint-XXXX/`

---

## 6. 评估与推理

### 6.1 评估模型

```bash
# 方法一：直接运行
python scripts/evaluate.py \
    --model_path outputs/final_model \
    --eval_data data/eval.jsonl \
    --output_file evaluation_results.json

# 方法二：使用 Makefile
make evaluate

# 方法三：使用一键脚本
./scripts/run.sh evaluate
```

**评估指标：**
- `exact_match`: 精确匹配率
- `f1`: F1 分数
- `rouge_l`: ROUGE-L 分数
- `bleu`: BLEU 分数

### 6.2 交互式推理

```bash
# 方法一：直接运行
python scripts/inference.py \
    --model_path outputs/final_model \
    --interactive

# 方法二：使用 Makefile
make inference

# 方法三：使用一键脚本
./scripts/run.sh inference
```

**交互式命令：**
- 输入文本进行对话
- 输入 `clear` 清空对话历史
- 输入 `quit` 或 `exit` 退出

### 6.3 批量推理

```bash
python scripts/inference.py \
    --model_path outputs/final_model \
    --input_file data/eval.jsonl \
    --output_file inference_results.jsonl
```

### 6.4 使用 LoRA 权重推理

如果只保存了 LoRA 权重：

```bash
python scripts/inference.py \
    --model_path Qwen/Qwen3-3B \
    --lora_path outputs/lora_weights \
    --interactive
```

---

## 7. 常见问题

### 7.1 显存不足 (OOM)

**症状：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**

1. **减小批大小：**
```yaml
batch_size: 1  # 从 2 改为 1
gradient_accumulation_steps: 8  # 相应增加
```

2. **启用梯度检查点：**
```yaml
gradient_checkpointing: true
```

3. **使用 QLoRA（4-bit 量化）：**
```yaml
use_quantization: true
```

4. **减小序列长度：**
```yaml
max_length: 1024  # 从 2048 改为 1024
```

### 7.2 训练 loss 不下降

**可能原因：**
- 学习率过高/过低
- 数据质量问题
- 批大小过大

**解决方案：**
```yaml
# 调整学习率
learning_rate: 0.0001  # 尝试更小的值

# 增加数据清洗
# 在 preprocess_data.py 中调整参数

# 减小批大小
batch_size: 1
```

### 7.3 模型生成质量差

**可能原因：**
- 训练轮数不足
- 数据量太少
- 过拟合

**解决方案：**
```yaml
# 增加训练轮数
epochs: 5

# 增加数据量（使用抽取脚本生成更多数据）

# 添加正则化
lora:
  dropout: 0.1  # 增加 dropout
weight_decay: 0.01
```

### 7.4 依赖安装失败

**解决方案：**
```bash
# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 升级 pip
pip install --upgrade pip

# 如果特定包失败，单独安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 7.5 数据格式错误

**症状：**
```
JSONDecodeError: Expecting value
```

**解决方案：**
```bash
# 检查 JSONL 格式
python -c "
import json
with open('data/train.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f'Line {i}: {e}')
"
```

---

## 8. 高级配置

### 8.1 自定义训练配置

复制并修改配置文件：

```bash
cp configs/lora_config.yaml configs/my_config.yaml
```

**配置文件说明：**

```yaml
# 模型配置
model_name: "Qwen/Qwen3-3B"  # 模型名称或本地路径
use_lora: true                 # 是否使用 LoRA
use_quantization: false        # 是否使用量化

# 训练配置
epochs: 3                      # 训练轮数
batch_size: 2                  # 批大小
learning_rate: 0.0002          # 学习率
max_length: 2048               # 最大序列长度

# 保存配置
output_dir: "outputs"          # 输出目录
save_steps: 100                # 保存间隔
logging_steps: 10              # 日志间隔
```

### 8.2 使用自定义模型

如果模型已下载到本地：

```yaml
# 修改 model_name 为本地路径
model_name: "/root/models/Qwen3-3B"
```

### 8.3 多 GPU 训练

Auto DL 多 GPU 实例：

```bash
# 使用 accelerate 启动
accelerate launch --multi_gpu scripts/train.py --config configs/lora_config.yaml
```

### 8.4 断点续训

从检查点继续训练：

```bash
python scripts/train.py \
    --config configs/lora_config.yaml \
    --resume_from_checkpoint outputs/checkpoint-100
```

### 8.5 自定义系统提示

在配置文件中添加：

```yaml
# 推理时使用
system_prompt: "你是一个支持成人内容的对话助手。"
```

---

## 📊 训练流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    训练流程概览                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 环境准备                                                │
│     ├─ 创建 Auto DL 实例                                    │
│     ├─ 安装依赖                                             │
│     └─ 测试环境                                             │
│                                                             │
│  2. 数据准备                                                │
│     ├─ 检查数据集                                           │
│     ├─ 数据预处理                                           │
│     └─ 划分训练/验证集                                      │
│                                                             │
│  3. 模型训练                                                │
│     ├─ 选择训练方式 (LoRA/全参数)                           │
│     ├─ 配置训练参数                                         │
│     ├─ 启动训练                                             │
│     └─ 监控训练过程                                         │
│                                                             │
│  4. 评估推理                                                │
│     ├─ 评估模型性能                                         │
│     ├─ 交互式测试                                           │
│     └─ 批量推理                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 快速开始（5 分钟上手）

```bash
# 1. 进入项目目录
cd /root/Qwen3_3B_FineTuningNSFW

# 2. 安装依赖
pip install -r requirements.txt

# 3. 测试环境
python scripts/test_environment.py

# 4. 数据预处理
python scripts/preprocess_data.py --input data/train.jsonl --output data --clean --split

# 5. 开始训练（LoRA）
python scripts/train.py --config configs/lora_config.yaml

# 6. 交互式测试
python scripts/inference.py --model_path outputs/final_model --interactive
```

---

## 📝 注意事项

1. **显存管理**: 监控 GPU 显存使用，避免 OOM
2. **数据备份**: 重要数据及时备份，实例可能被回收
3. **模型保存**: 定期保存检查点，避免训练中断丢失进度
4. **合规使用**: 确保在合规环境下使用，遵守当地法律法规

---

## 🔗 相关资源

- [Qwen3 官方文档](https://qwen.readthedocs.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PEFT 文档](https://huggingface.co/docs/peft/)
- [Auto DL 文档](https://www.autodl.com/docs/)

---

**最后更新**: 2026-07-02
**版本**: 1.0