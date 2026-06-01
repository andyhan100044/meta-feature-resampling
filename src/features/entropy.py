"""
复杂度特征提取
包括: 样本熵、排列熵
"""
import numpy as np
import math

def sample_entropy(signal, m=2, r=0.2):
    """样本熵 (Sample Entropy)

    Args:
        signal: 输入信号
        m: 模式长度
        r: 相似度阈值 (std的倍数)

    Returns:
        float: 样本熵值
    """
    n = len(signal)
    if n < m + 1:
        return 0.0

    r_threshold = r * np.std(signal)
    if r_threshold == 0:
        return 0.0

    # 构建m维模式
    patterns_m = np.array([signal[i:i+m] for i in range(n-m)])
    patterns_m1 = np.array([signal[i:i+m+1] for i in range(n-m)])

    def _max_dist(xi, xj):
        return np.max(np.abs(xi - xj))

    def _count_matches(patterns):
        count = 0
        for i in range(len(patterns)):
            for j in range(len(patterns)):
                if i != j and _max_dist(patterns[i], patterns[j]) <= r_threshold:
                    count += 1
        return count

    # 计算A和B
    A = _count_matches(patterns_m1) / ((n - m - 1) * (n - m - 2) + 1e-10)
    B = _count_matches(patterns_m) / ((n - m - 1) * (n - m - 1) + 1e-10)

    if A == 0 or B == 0:
        return 0.0

    return -np.log(A / B + 1e-10)

def permutation_entropy(signal, m=3):
    """排列熵 (Permutation Entropy)

    Args:
        signal: 输入信号
        m: 模式长度

    Returns:
        float: 排列熵值 (归一化)
    """
    n = len(signal)
    if n < m + 1:
        return 0.0

    # 构建排列模式
    patterns = [np.argsort(signal[i:i+m]) for i in range(n-m)]

    # 统计各模式出现次数
    counts = {}
    for p in patterns:
        key = tuple(p)
        counts[key] = counts.get(key, 0) + 1

    # 计算熵
    total = len(patterns)
    pe = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            pe -= p * np.log(p)

    # 归一化
    max_pe = np.log(math.factorial(m))
    if max_pe == 0:
        return 0.0

    return pe / max_pe