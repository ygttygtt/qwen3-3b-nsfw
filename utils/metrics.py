"""
评估指标工具
用于计算模型评估的各种指标
"""

import re
import string
from collections import Counter
from typing import List, Dict, Any, Optional


def normalize_answer(s: str) -> str:
    """标准化答案文本"""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """计算精确匹配分数"""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """计算 F1 分数"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    """计算 BLEU 分数（简化版本）"""

    def get_ngrams(text: str, n: int) -> Counter:
        words = text.split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i:i + n]))
        return Counter(ngrams)

    def compute_bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
        ref_ngrams = [get_ngrams(reference, n) for n in range(1, max_n + 1)]
        hyp_ngrams = [get_ngrams(hypothesis, n) for n in range(1, max_n + 1)]

        precisions = []
        for ref_ngram, hyp_ngram in zip(ref_ngrams, hyp_ngrams):
            if len(hyp_ngram) == 0:
                precisions.append(0.0)
            else:
                common = ref_ngram & hyp_ngram
                precisions.append(sum(common.values()) / sum(hyp_ngram.values()))

        # 几何平均
        if any(p == 0 for p in precisions):
            return 0.0

        log_avg = sum(1 / max_n * p for p in precisions)
        bleu = log_avg

        # 简洁惩罚
        ref_len = len(reference.split())
        hyp_len = len(hypothesis.split())
        if hyp_len < ref_len:
            bleu *= min(1, (1 - ref_len / hyp_len))

        return bleu

    scores = []
    for ref, hyp in zip(references, hypotheses):
        scores.append(compute_bleu_score(ref, hyp))

    return sum(scores) / len(scores) if scores else 0.0


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """计算 ROUGE-L 分数"""

    def lcs_length(x: List[str], y: List[str]) -> int:
        """计算最长公共子序列长度"""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    lcs = lcs_length(ref_words, hyp_words)

    if len(ref_words) == 0 or len(hyp_words) == 0:
        return 0.0

    precision = lcs / len(hyp_words)
    recall = lcs / len(ref_words)

    if precision + recall == 0:
        return 0.0

    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_metrics(
    predictions: List[str],
    references: List[str],
    metrics: List[str] = None,
) -> Dict[str, float]:
    """
    计算多个评估指标

    Args:
        predictions: 预测结果列表
        references: 参考答案列表
        metrics: 要计算的指标列表，默认为所有指标

    Returns:
        指标名称到分数的映射
    """
    if metrics is None:
        metrics = ["exact_match", "f1", "rouge_l", "bleu"]

    results = {}

    if "exact_match" in metrics:
        exact_scores = [
            compute_exact_match(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        results["exact_match"] = sum(exact_scores) / len(exact_scores)

    if "f1" in metrics:
        f1_scores = [
            compute_f1(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        results["f1"] = sum(f1_scores) / len(f1_scores)

    if "rouge_l" in metrics:
        rouge_scores = [
            compute_rouge_l(ref, pred)
            for pred, ref in zip(predictions, references)
        ]
        results["rouge_l"] = sum(rouge_scores) / len(rouge_scores)

    if "bleu" in metrics:
        results["bleu"] = compute_bleu(references, predictions)

    return results


def compute_perplexity(loss: float) -> float:
    """计算困惑度"""
    import math
    return math.exp(loss)


def compute_response_quality(response: str) -> Dict[str, Any]:
    """
    评估回复质量

    Args:
        response: 模型回复

    Returns:
        质量评估结果
    """
    quality_metrics = {
        "length": len(response),
        "word_count": len(response.split()),
        "sentence_count": len(re.split(r'[。！？.!?]+', response)),
        "has_chinese": bool(re.search(r'[一-鿿]', response)),
        "has_english": bool(re.search(r'[a-zA-Z]', response)),
        "is_empty": len(response.strip()) == 0,
        "is_too_short": len(response.strip()) < 10,
        "is_too_long": len(response.strip()) > 1000,
    }

    return quality_metrics