#!/usr/bin/env python3
"""
环境测试脚本
检查是否满足微调要求
"""

import sys
import importlib
import subprocess
from typing import List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """检查 Python 版本"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (需要 3.10+)"


def check_package(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """检查包是否安装"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        if min_version and version != "unknown":
            # 版本比较
            from packaging.version import Version
            try:
                if Version(version) < Version(min_version):
                    return False, f"{package_name} {version} (需要 {min_version}+)"
            except Exception:
                # fallback: 简单字符串比较
                if version < min_version:
                    return False, f"{package_name} {version} (需要 {min_version}+)"
        return True, f"{package_name} {version}"
    except ImportError:
        return False, f"{package_name} 未安装"


def check_cuda() -> Tuple[bool, str]:
    """检查 CUDA 是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            return True, f"CUDA {cuda_version} - {device_name}"
        else:
            return False, "CUDA 不可用"
    except ImportError:
        return False, "PyTorch 未安装"


def check_gpu_memory() -> Tuple[bool, str]:
    """检查 GPU 显存"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024 ** 3)
            if gpu_memory_gb >= 20:
                return True, f"GPU 显存: {gpu_memory_gb:.1f} GB"
            else:
                return False, f"GPU 显存不足: {gpu_memory_gb:.1f} GB (建议 20GB+)"
        else:
            return False, "CUDA 不可用"
    except ImportError:
        return False, "PyTorch 未安装"


def check_disk_space() -> Tuple[bool, str]:
    """检查磁盘空间"""
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024 ** 3)
    if free_gb >= 50:
        return True, f"可用磁盘空间: {free_gb:.1f} GB"
    else:
        return False, f"磁盘空间不足: {free_gb:.1f} GB (建议 50GB+)"


def main():
    print("=" * 60)
    print("Qwen3-3B NSFW 微调环境检查")
    print("=" * 60)

    checks = []

    # 检查 Python 版本
    checks.append(("Python 版本", check_python_version()))

    # 检查必需包
    required_packages = [
        ("torch", "2.0.0"),
        ("transformers", "4.35.0"),
        ("peft", "0.6.0"),
        ("datasets", "2.14.0"),
        ("accelerate", "0.24.0"),
        ("bitsandbytes", "0.41.0"),
        ("scipy", "1.11.0"),
        ("sentencepiece", "0.1.99"),
        ("protobuf", "4.24.0"),
        ("tensorboard", "2.14.0"),
        ("scikit-learn", "1.3.0"),
        ("numpy", "1.24.0"),
        ("pandas", "2.0.0"),
        ("tqdm", "4.65.0"),
        ("pyyaml", "6.0.0"),
        ("einops", "0.7.0"),
    ]

    for package_name, min_version in required_packages:
        checks.append((f"{package_name}", check_package(package_name, min_version)))

    # 检查 CUDA
    checks.append(("CUDA", check_cuda()))

    # 检查 GPU 显存
    checks.append(("GPU 显存", check_gpu_memory()))

    # 检查磁盘空间
    checks.append(("磁盘空间", check_disk_space()))

    # 打印结果
    print("\n检查结果:")
    print("-" * 60)

    all_passed = True
    for name, (passed, message) in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {name}: {message}")
        if not passed:
            all_passed = False

    print("-" * 60)

    if all_passed:
        print("\n✓ 所有检查通过！环境满足微调要求。")
        return 0
    else:
        print("\n✗ 部分检查未通过，请根据上述信息修复环境问题。")
        return 1


if __name__ == "__main__":
    sys.exit(main())