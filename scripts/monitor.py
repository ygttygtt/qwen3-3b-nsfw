#!/usr/bin/env python3
"""
训练监控脚本
实时监控训练进度和 GPU 使用情况
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import psutil


def get_gpu_info():
    """获取 GPU 信息"""
    if not torch.cuda.is_available():
        return None

    gpu_info = []
    for i in range(torch.cuda.device_count()):
        info = {
            "device": i,
            "name": torch.cuda.get_device_name(i),
            "memory_total": torch.cuda.get_device_properties(i).total_memory / 1024**3,
            "memory_allocated": torch.cuda.memory_allocated(i) / 1024**3,
            "memory_reserved": torch.cuda.memory_reserved(i) / 1024**3,
            "utilization": torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0,
        }
        info["memory_free"] = info["memory_total"] - info["memory_allocated"]
        info["memory_usage_percent"] = (info["memory_allocated"] / info["memory_total"]) * 100
        gpu_info.append(info)

    return gpu_info


def get_system_info():
    """获取系统信息"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_total": psutil.virtual_memory().total / 1024**3,
        "memory_available": psutil.virtual_memory().available / 1024**3,
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
    }


def get_training_status(output_dir):
    """获取训练状态"""
    status = {
        "is_training": False,
        "current_epoch": 0,
        "current_step": 0,
        "total_steps": 0,
        "loss": None,
        "learning_rate": None,
    }

    # 检查是否有训练日志
    log_file = Path(output_dir) / "train.log"
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    # 尝试解析 JSON 格式的日志
                    try:
                        log_data = json.loads(last_line)
                        status["current_epoch"] = log_data.get("epoch", 0)
                        status["current_step"] = log_data.get("step", 0)
                        status["loss"] = log_data.get("loss")
                        status["learning_rate"] = log_data.get("learning_rate")
                        status["is_training"] = True
                    except json.JSONDecodeError:
                        # 尝试解析文本格式的日志
                        if "loss" in last_line.lower():
                            status["is_training"] = True
        except Exception:
            pass

    # 检查最新的 checkpoint
    checkpoints = list(Path(output_dir).glob("checkpoint-*"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
        checkpoint_step = int(latest_checkpoint.name.split("-")[1])
        status["current_step"] = checkpoint_step

    return status


def display_monitor(output_dir, refresh_interval=5):
    """显示监控界面"""
    print("\033[2J\033[H")  # 清屏
    print("=" * 60)
    print("           Qwen3-3B 训练监控")
    print("=" * 60)

    try:
        while True:
            # 获取信息
            gpu_info = get_gpu_info()
            system_info = get_system_info()
            training_status = get_training_status(output_dir)

            # 移动光标到开头
            print("\033[H")

            # 显示时间
            print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 60)

            # 显示 GPU 信息
            if gpu_info:
                print("GPU 状态:")
                for gpu in gpu_info:
                    print(f"  GPU {gpu['device']}: {gpu['name']}")
                    print(f"    显存: {gpu['memory_allocated']:.1f}GB / {gpu['memory_total']:.1f}GB "
                          f"({gpu['memory_usage_percent']:.1f}%)")
                    print(f"    利用率: {gpu['utilization']}%")
            else:
                print("GPU: 不可用")

            print("-" * 60)

            # 显示系统信息
            print("系统状态:")
            print(f"  CPU: {system_info['cpu_percent']}%")
            print(f"  内存: {system_info['memory_available']:.1f}GB / {system_info['memory_total']:.1f}GB "
                  f"({system_info['memory_percent']}%)")
            print(f"  磁盘: {system_info['disk_usage']}%")

            print("-" * 60)

            # 显示训练状态
            print("训练状态:")
            if training_status["is_training"]:
                print(f"  状态: 训练中")
                print(f"  Epoch: {training_status['current_epoch']}")
                print(f"  Step: {training_status['current_step']}")
                if training_status["total_steps"] > 0:
                    progress = (training_status["current_step"] / training_status["total_steps"]) * 100
                    print(f"  进度: {progress:.1f}%")
                if training_status["loss"]:
                    print(f"  Loss: {training_status['loss']:.4f}")
                if training_status["learning_rate"]:
                    print(f"  学习率: {training_status['learning_rate']:.6f}")
            else:
                print(f"  状态: 未检测到训练")

            print("-" * 60)
            print("按 Ctrl+C 退出监控")

            # 等待刷新
            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n监控已停止")


def main():
    parser = argparse.ArgumentParser(description="训练监控脚本")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="训练输出目录"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="刷新间隔（秒）"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f"错误: 输出目录 {args.output_dir} 不存在")
        sys.exit(1)

    display_monitor(args.output_dir, args.interval)


if __name__ == "__main__":
    main()