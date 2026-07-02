"""
自定义数据整理器
用于处理对话数据的特殊需求
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import torch
from transformers import PreTrainedTokenizer


@dataclass
class ConversationDataCollator:
    """
    对话数据整理器
    处理多轮对话数据，支持动态 padding
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    padding: bool = True
    truncation: bool = True
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        整理对话数据

        Args:
            features: 包含对话数据的字典列表

        Returns:
            整理后的张量字典
        """
        # 提取 input_ids 和 labels
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            attention_mask = feature.get("attention_mask", [1] * len(input_ids))

            # 截断
            if self.truncation and len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]
                attention_mask = attention_mask[:self.max_length]

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)

        # 动态 padding
        if self.padding:
            max_len = max(len(ids) for ids in input_ids_list)

            padded_input_ids = []
            padded_labels = []
            padded_attention_mask = []

            for input_ids, labels, attention_mask in zip(
                input_ids_list, labels_list, attention_mask_list
            ):
                # 计算需要填充的长度
                padding_length = max_len - len(input_ids)

                # 填充 input_ids
                padded_input_ids.append(
                    input_ids + [self.tokenizer.pad_token_id] * padding_length
                )

                # 填充 labels（使用 -100 忽略填充位置）
                padded_labels.append(labels + [-100] * padding_length)

                # 填充 attention_mask
                padded_attention_mask.append(
                    attention_mask + [0] * padding_length
                )

            return {
                "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
                "labels": torch.tensor(padded_labels, dtype=torch.long),
                "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            }
        else:
            return {
                "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
                "labels": torch.tensor(labels_list, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            }


@dataclass
class QwenConversationDataCollator:
    """
    Qwen 模型专用对话数据整理器
    支持 Qwen 的对话格式和特殊 token
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    padding: bool = True
    truncation: bool = True
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        整理 Qwen 对话数据

        Args:
            features: 包含对话数据的字典列表

        Returns:
            整理后的张量字典
        """
        # 提取特征
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            attention_mask = feature.get("attention_mask", [1] * len(input_ids))

            # 截断
            if self.truncation and len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]
                attention_mask = attention_mask[:self.max_length]

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)

        # 动态 padding
        if self.padding:
            max_len = max(len(ids) for ids in input_ids_list)

            padded_input_ids = []
            padded_labels = []
            padded_attention_mask = []

            for input_ids, labels, attention_mask in zip(
                input_ids_list, labels_list, attention_mask_list
            ):
                # 计算需要填充的长度
                padding_length = max_len - len(input_ids)

                # 填充 input_ids
                padded_input_ids.append(
                    input_ids + [self.tokenizer.pad_token_id] * padding_length
                )

                # 填充 labels（使用 -100 忽略填充位置）
                padded_labels.append(labels + [-100] * padding_length)

                # 填充 attention_mask
                padded_attention_mask.append(
                    attention_mask + [0] * padding_length
                )

            return {
                "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
                "labels": torch.tensor(padded_labels, dtype=torch.long),
                "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            }
        else:
            return {
                "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
                "labels": torch.tensor(labels_list, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            }