#!/bin/bash

# Qwen3-3B NSFW 微调快速启动脚本
# 用法: ./quick_start.sh [选项]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_header() {
    echo -e "${CYAN}"
    echo "=========================================="
    echo "   Qwen3-3B NSFW 微调快速启动"
    echo "=========================================="
    echo -e "${NC}"
}

# 检查环境
check_environment() {
    print_step "检查环境..."

    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        exit 1
    fi

    # 检查 CUDA
    if ! python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        print_warn "CUDA 不可用，训练将使用 CPU（非常慢）"
    else
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        print_info "GPU: $GPU_NAME"
    fi

    print_info "环境检查完成"
}

# 安装依赖
install_dependencies() {
    print_step "安装依赖..."

    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt -q
        print_info "依赖安装完成"
    else
        print_error "requirements.txt 不存在"
        exit 1
    fi
}

# 数据预处理
preprocess_data() {
    print_step "数据预处理..."

    if [ ! -f "data/train.jsonl" ]; then
        print_error "数据集 data/train.jsonl 不存在"
        exit 1
    fi

    python3 scripts/preprocess_data.py \
        --input data/train.jsonl \
        --output data \
        --clean \
        --split \
        --eval_ratio 0.1

    print_info "数据预处理完成"
    print_info "训练集: data/train.jsonl"
    print_info "验证集: data/eval.jsonl"
}

# 训练模型
train_model() {
    local mode=${1:-lora}

    print_step "开始训练 (模式: $mode)..."

    case $mode in
        lora)
            python3 scripts/train.py --config configs/lora_config.yaml
            ;;
        full)
            python3 scripts/train.py --config configs/full_finetune_config.yaml
            ;;
        qlora)
            # 创建 QLoRA 配置
            cat > configs/qlora_config.yaml << 'EOF'
model_name: "Qwen/Qwen3-3B"
use_lora: true
use_quantization: true

lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"

data_path: "data/train.jsonl"
max_length: 2048
eval_split: 0.1

epochs: 3
batch_size: 2
eval_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 0.0002
weight_decay: 0.01
scheduler: "cosine"
warmup_ratio: 0.1

output_dir: "outputs"
logging_steps: 10
save_steps: 100
save_total_limit: 3
eval_steps: 100
report_to: "tensorboard"
run_name: "qwen3-3b-nsfw-qlora"
gradient_checkpointing: true
EOF
            python3 scripts/train.py --config configs/qlora_config.yaml
            ;;
        *)
            print_error "未知的训练模式: $mode"
            print_info "可用模式: lora, full, qlora"
            exit 1
            ;;
    esac

    print_info "训练完成！"
    print_info "模型保存在: outputs/final_model/"
}

# 评估模型
evaluate_model() {
    print_step "评估模型..."

    if [ ! -d "outputs/final_model" ]; then
        print_error "模型目录不存在，请先训练"
        exit 1
    fi

    python3 scripts/evaluate.py \
        --model_path outputs/final_model \
        --eval_data data/eval.jsonl \
        --output_file evaluation_results.json

    print_info "评估完成！"
    print_info "结果保存在: evaluation_results.json"
}

# 交互式推理
interactive_inference() {
    print_step "启动交互式推理..."

    if [ ! -d "outputs/final_model" ]; then
        print_error "模型目录不存在，请先训练"
        exit 1
    fi

    python3 scripts/inference.py \
        --model_path outputs/final_model \
        --interactive
}

# 显示帮助
show_help() {
    echo "用法: ./quick_start.sh [选项]"
    echo ""
    echo "选项:"
    echo "  check       - 检查环境"
    echo "  install     - 安装依赖"
    echo "  preprocess  - 数据预处理"
    echo "  train       - 训练模型 (默认 LoRA)"
    echo "  train-lora  - 使用 LoRA 训练"
    echo "  train-full  - 全参数训练"
    echo "  train-qlora - 使用 QLoRA 训练"
    echo "  evaluate    - 评估模型"
    echo "  inference   - 交互式推理"
    echo "  all         - 执行完整流程"
    echo "  help        - 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  ./quick_start.sh all          # 执行完整流程"
    echo "  ./quick_start.sh train-lora   # 使用 LoRA 训练"
    echo "  ./quick_start.sh inference    # 交互式推理"
}

# 主函数
main() {
    print_header

    case "${1:-help}" in
        check)
            check_environment
            ;;
        install)
            check_environment
            install_dependencies
            ;;
        preprocess)
            check_environment
            preprocess_data
            ;;
        train)
            check_environment
            train_model lora
            ;;
        train-lora)
            check_environment
            train_model lora
            ;;
        train-full)
            check_environment
            train_model full
            ;;
        train-qlora)
            check_environment
            train_model qlora
            ;;
        evaluate)
            evaluate_model
            ;;
        inference)
            interactive_inference
            ;;
        all)
            check_environment
            install_dependencies
            preprocess_data
            train_model lora
            evaluate_model
            print_info "完整流程执行完成！"
            print_info "运行 './quick_start.sh inference' 开始交互式测试"
            ;;
        help|*)
            show_help
            ;;
    esac
}

# 执行主函数
main "$@"