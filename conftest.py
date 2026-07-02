"""
Pytest 配置文件
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """pytest 配置"""
    # 添加自定义标记
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require GPU",
    )


def pytest_collection_modifyitems(config, items):
    """修改测试项"""
    # 自动跳过需要 GPU 的测试（如果没有 GPU）
    import torch
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="No GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)