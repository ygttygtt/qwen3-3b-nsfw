# AutoDL 部署与微调经验总结

## 一、环境配置

### 1.1 镜像选择（重要！）

| 错误选择 | 正确选择 |
|----------|----------|
| PyTorch 2.3.0 | **PyTorch >= 2.4.0** |
| CUDA 12.1 | CUDA >= 12.4 |

**教训：** LLaMA-Factory 0.9.6 需要 PyTorch >= 2.4.0，旧版本会导致依赖地狱。

### 1.2 推荐镜像配置

```
PyTorch 2.8.0+ / CUDA 12.8 / Python 3.12
GPU: RTX 3090 或 4090 (24GB+)
```

### 1.3 系统盘 vs 数据盘

| 目录 | 说明 | 注意事项 |
|------|------|----------|
| `/root/` | 系统盘（30GB） | 关机不丢失，但空间小 |
| `/root/autodl-tmp/` | 数据盘（50GB+） | 关机不丢失，空间大 |

**教训：** 模型和数据集必须放数据盘！系统盘只有 30GB，装完依赖就没空间了。

### 1.4 端口映射

AutoDL 只映射 **6006** 和 **6008** 端口。LLaMA-Factory Web UI 默认用 7860，需要手动改：

```bash
# 启动时指定端口
llamafactory-cli webui --port 6006 --server 0.0.0.0
```

---

## 二、模型选择

### 2.1 Base vs Instruct

| 模型类型 | 优点 | 缺点 |
|----------|------|------|
| **Base** | 训练自由度高 | 对话能力差，只会续写，角色混乱 |
| **Instruct** | 对话能力强，角色理解好 | 训练自由度稍低 |

**结论：对话场景必须用 Instruct 模型！**

### 2.2 推荐模型

- **Qwen3.5-4B-Instruct**（推荐）
- Qwen3-8B-Instruct（效果更好，显存要求高）
- Qwen2.5-7B-Instruct（稳定成熟）

### 2.3 Base 模型的问题

- 不理解对话格式，容易角色混乱
- 用户说一句话，模型以为是自己的内容然后续写
- 缺乏对话意识，输出像文章而不是对话
- 需要极其高质量的数据集才能训练好

---

## 三、数据集

### 3.1 数据清洗（重要！）

训练前必须清洗数据，移除：

1. **格式标记**：`（字数约280）`、`（超过280字）` 等
2. **脏话过多的数据**：影响模型输出质量
3. **格式不一致的数据**：统一格式
4. **过短或过长的数据**：筛选合理长度

### 3.2 数据格式

LLaMA-Factory 支持的格式：

```json
{
  "instruction": "用户指令",
  "input": "输入上下文（可选）",
  "output": "期望输出",
  "system": "系统提示（可选）"
}
```

### 3.3 数据量建议

| 数据量 | 效果 |
|--------|------|
| 2K | 能用，但效果有限 |
| 5K-10K | 比较理想 |
| 10K+ | 更好 |

### 3.4 数据质量 > 数据数量

- 1000 条高质量数据 > 5000 条低质量数据
- 对话数据需要自然、多样
- 避免重复模式和模板化输出

---

## 四、LLaMA-Factory 配置

### 4.1 安装

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e '.[torch,metrics]'
```

### 4.2 加速优化

```bash
pip install causal-conv1d flash-attn
```

安装后训练速度提升 30-50%。

### 4.3 推荐训练参数（LoRA）

```yaml
# 模型
model_name_or_path: /path/to/model
template: qwen3_5

# LoRA
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: all

# 训练
learning_rate: 2e-4
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
lr_scheduler_type: cosine
warmup_steps: 50

# 优化
bf16: true
flash_attn: auto
enable_thinking: false  # NSFW 对话不需要思考模式
```

### 4.4 数据集注册

1. 将数据文件复制到 `LLaMA-Factory/data/` 目录
2. 编辑 `data/dataset_info.json` 添加配置：

```json
{
  "my_dataset": {
    "file_name": "my_data.jsonl",
    "columns": {
      "prompt": "instruction",
      "response": "output",
      "system": "system"
    }
  }
}
```

---

## 五、训练与测试

### 5.1 训练监控

```bash
# 查看 GPU 使用
nvidia-smi

# 查看训练日志
tail -f saves/model/lora/train_xxx/trainer_log.jsonl
```

### 5.2 测试方法

1. Web UI Chat 页面
2. 加载基础模型 + LoRA 权重
3. 对比加载 LoRA 前后的效果

### 5.3 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 输出有括号格式 | 训练数据有 | 清洗数据 |
| 对话能力差 | 用了 Base 模型 | 换 Instruct 模型 |
| 脏话太多 | 数据质量问题 | 清洗数据 |
| 角色混乱 | Base 模型 + 数据问题 | 换模型 + 清洗数据 |
| 训练慢 | 没装 flash-attn | 安装加速库 |

---

## 六、下次改进计划

### 6.1 模型

- [ ] 下载 Qwen3.5-4B-Instruct
- [ ] 测试 Instruct 模型效果

### 6.2 数据

- [ ] 清洗"字数约xxx"等格式标记
- [ ] 减少脏话数据比例
- [ ] 增加对话多样性
- [ ] 目标：5K-10K 高质量数据

### 6.3 环境

- [ ] 安装 flash-attn 加速
- [ ] 使用 PyTorch 2.8+ 镜像
- [ ] 项目放数据盘

### 6.4 训练

- [ ] 关闭 enable_thinking
- [ ] 使用 bf16 精度
- [ ] 调整学习率和 epoch

---

## 七、快速部署清单

```bash
# 1. 克隆项目（数据盘）
cd /root/autodl-tmp
git clone https://github.com/ygttygtt/qwen3-3b-nsfw.git
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git

# 2. 安装 LLaMA-Factory
cd LLaMA-Factory
pip install -e '.[torch,metrics]'
pip install causal-conv1d flash-attn

# 3. 复制数据
cp /root/autodl-tmp/qwen3-3b-nsfw/data/train.jsonl data/nsfw_train.jsonl

# 4. 注册数据集（编辑 data/dataset_info.json）

# 5. 启动 Web UI（端口 6006）
llamafactory-cli webui --port 6006 --server 0.0.0.0

# 6. 访问
# https://xxx.autodl.pro
```

---

**最后更新：2026-07-03**
**版本：1.0**
