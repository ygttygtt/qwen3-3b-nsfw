#!/bin/bash

# Qwen3-3B NSFW 微调启动脚本
# 用法: ./scripts/run.sh [command]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# 检查 Python 环境
check_environment() {
    print_info "检查 Python 环境..."

    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        exit 1
    fi

    python3 --version
    print_info "Python 环境检查完成"
}

# 安装依赖
install_dependencies() {
    print_info "安装依赖..."
    pip install -r requirements.txt
    print_info "依赖安装完成"
}

# 数据预处理
preprocess_data() {
    print_info "数据预处理..."
    python3 scripts/preprocess_data.py \
        --input data/train.jsonl \
        --output data \
        --clean \
        --split \
        --eval_ratio 0.1
    print_info "数据预处理完成"
}

# 训练模型 (LoRA)
train_lora() {
    print_info "开始 LoRA 微调..."
    python3 scripts/train.py --config configs/lora_config.yaml
    print_info "LoRA 微调完成"
}

# 训练模型 (全参数)
train_full() {
    print_info "开始全参数微调..."
    python3 scripts/train.py --config configs/full_finetune_config.yaml
    print_info "全参数微调完成"
}

# 评估模型
evaluate_model() {
    print_info "评估模型..."
    python3 scripts/evaluate.py \
        --model_path outputs/final_model \
        --eval_data data/eval.jsonl \
        --output_file evaluation_results.json
    print_info "模型评估完成"
}

# 交互式推理
inference_interactive() {
    print_info "启动交互式推理..."
    python3 scripts/inference.py \
        --model_path outputs/final_model \
        --interactive
}

# 批量推理
inference_batch() {
    print_info "批量推理..."
    python3 scripts/inference.py \
        --model_path outputs/final_model \
        --input_file data/eval.jsonl \
        --output_file inference_results.jsonl
    print_info "批量推理完成"
}

# 测试环境
test_environment() {
    print_info "测试环境..."
    python3 scripts/test_environment.py
}

# 显示帮助
show_help() {
    echo "Qwen3-3B NSFW 微调启动脚本"
    echo ""
    echo "用法: ./scripts/run.sh [command]"
    echo ""
    echo "可用命令:"
    echo "  check       - 检查 Python 环境"
    echo "  install     - 安装依赖"
    echo "  preprocess  - 数据预处理"
    echo "  train-lora  - 使用 LoRA 微调"
    echo "  train-full  - 全参数微调"
    echo "  evaluate    - 评估模型"
    echo "  inference   - 交互式推理"
    echo "  batch       - 批量推理"
    echo "  test        - 测试环境是否满足要求"
    echo "  all         - 执行完整流程 (preprocess -> train-lora -> evaluate)"
    echo "  help        - 显示此帮助信息"
}

# 主函数
main() {
    check_environment

    case "${1:-help}" in
        check)
            ;;
        install)
            install_dependencies
            ;;
        preprocess)
            preprocess_data
            ;;
        train-lora)
            train_lora
            ;;
        train-full)
            train_full
            ;;
        evaluate)
            evaluate_model
            ;;
        inference)
            inference_interactive
            ;;
        batch)
            inference_batch
            ;;
        test)
            test_environment
            ;;
        all)
            preprocess_data
            train_lora
            evaluate_model
            ;;
        help|*)
            show_help
            ;;
    esac
}

# 执行主函数
main "$@"