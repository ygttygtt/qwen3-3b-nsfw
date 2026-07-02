.PHONY: help install preprocess train-lora train-full evaluate inference test all clean

# 默认目标
help:
	@echo "Qwen3-3B NSFW 微调项目"
	@echo ""
	@echo "可用命令:"
	@echo "  make install      - 安装依赖"
	@echo "  make preprocess   - 数据预处理"
	@echo "  make train-lora   - 使用 LoRA 微调"
	@echo "  make train-full   - 全参数微调"
	@echo "  make evaluate     - 评估模型"
	@echo "  make inference    - 交互式推理"
	@echo "  make test         - 测试环境"
	@echo "  make all          - 执行完整流程"
	@echo "  make clean        - 清理输出文件"

# 安装依赖
install:
	pip install -r requirements.txt

# 数据预处理
preprocess:
	python scripts/preprocess_data.py --input data/train.jsonl --output data --clean --split --eval_ratio 0.1

# LoRA 微调
train-lora:
	python scripts/train.py --config configs/lora_config.yaml

# 全参数微调
train-full:
	python scripts/train.py --config configs/full_finetune_config.yaml

# 评估模型
evaluate:
	python scripts/evaluate.py --model_path outputs/final_model --eval_data data/eval.jsonl --output_file evaluation_results.json

# 交互式推理
inference:
	python scripts/inference.py --model_path outputs/final_model --interactive

# 测试环境
test:
	python scripts/test_environment.py

# 执行完整流程
all: preprocess train-lora evaluate

# 清理输出文件
clean:
	rm -rf outputs/*
	rm -f evaluation_results.json
	rm -f inference_results.jsonl
	rm -f *.log