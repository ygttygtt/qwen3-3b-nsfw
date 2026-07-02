"""
数据预处理脚本测试
"""

import os
import json
import tempfile
import pytest
from pathlib import Path

# 添加项目根目录到 Python 路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.preprocess_data import (
    load_jsonl,
    save_jsonl,
    validate_data,
    clean_data,
    split_data,
)


@pytest.fixture
def sample_data():
    """示例数据"""
    return [
        {
            "instruction": "请创作一段成人小说",
            "input": "",
            "output": "这是一个测试输出",
            "system": "你是一个内容创作专家",
            "history": [],
        },
        {
            "instruction": "生成一段对话",
            "input": "场景：办公室",
            "output": "这是一个更长的测试输出，用于测试数据清洗功能",
            "system": "",
            "history": [],
        },
        {
            "instruction": "",  # 空指令
            "input": "",
            "output": "这个条目应该被过滤掉",
            "system": "",
            "history": [],
        },
    ]


@pytest.fixture
def temp_dir():
    """临时目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_load_jsonl(sample_data, temp_dir):
    """测试加载 JSONL 文件"""
    # 保存示例数据
    file_path = os.path.join(temp_dir, "test.jsonl")
    save_jsonl(sample_data, file_path)

    # 加载数据
    loaded_data = load_jsonl(file_path)

    assert len(loaded_data) == len(sample_data)
    assert loaded_data[0]["instruction"] == sample_data[0]["instruction"]


def test_save_jsonl(sample_data, temp_dir):
    """测试保存 JSONL 文件"""
    file_path = os.path.join(temp_dir, "test.jsonl")
    save_jsonl(sample_data, file_path)

    assert os.path.exists(file_path)

    # 验证文件内容
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == len(sample_data)


def test_validate_data(sample_data):
    """测试数据验证"""
    stats = validate_data(sample_data)

    assert stats["total_examples"] == 3
    assert stats["missing_fields"]["instruction"] == 1
    assert stats["empty_fields"]["instruction"] == 1


def test_clean_data(sample_data):
    """测试数据清洗"""
    cleaned = clean_data(sample_data, min_instruction_length=5)

    # 应该过滤掉空指令的条目
    assert len(cleaned) == 2

    # 验证清洗后的数据
    for item in cleaned:
        assert len(item["instruction"]) >= 5


def test_split_data(sample_data):
    """测试数据划分"""
    train_data, eval_data = split_data(sample_data, eval_ratio=0.33, seed=42)

    # 验证划分比例
    total = len(train_data) + len(eval_data)
    assert total == len(sample_data)

    # 验证数据完整性
    all_data = train_data + eval_data
    assert len(all_data) == len(sample_data)


def test_end_to_end(sample_data, temp_dir):
    """端到端测试"""
    # 保存原始数据
    input_path = os.path.join(temp_dir, "input.jsonl")
    save_jsonl(sample_data, input_path)

    # 加载数据
    data = load_jsonl(input_path)

    # 清洗数据
    cleaned = clean_data(data, min_instruction_length=5)

    # 划分数据
    train_data, eval_data = split_data(cleaned, eval_ratio=0.5, seed=42)

    # 保存结果
    train_path = os.path.join(temp_dir, "train.jsonl")
    eval_path = os.path.join(temp_dir, "eval.jsonl")
    save_jsonl(train_data, train_path)
    save_jsonl(eval_data, eval_path)

    # 验证文件存在
    assert os.path.exists(train_path)
    assert os.path.exists(eval_path)

    # 验证数据完整性
    total = len(train_data) + len(eval_data)
    assert total == len(cleaned)