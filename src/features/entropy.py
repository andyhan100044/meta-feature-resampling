"""
复杂度特征提取 - 优化版本
包括: 样本熵、排列熵
"""
import numpy as np
import math
from numpy.lib.stride_tricks import as_strided


def permutation_entropy(signal, m=3):
    """排列熵 (Permutation Entropy) - 完全向量化实现

    Args:
        signal: 输入信号
        m: 模式长度

    Returns:
        float: 排列熵值 (归一化)
    """
    n = len(signal)
    if n < m + 1:
        return 0.0

    # 用stride_tricks避免Python循环，提取所有m维窗口：(n-m+1, m)
    windows = as_strided(signal, shape=(n - m + 1, m), strides=(signal.strides[0], signal.strides[0]))
    windows = windows.copy()  # 确保连续内存

    # 计算每个窗口的排列编码：排序索引的整数编码
    order = np.argsort(windows, axis=1)
    ranks = np.argsort(order, axis=1)
    codes = np.dot(ranks, (m ** np.arange(m))).astype(np.int64)

    # 统计各排列出现次数
    unique, counts = np.unique(codes, return_counts=True)
    total = len(codes)

    # 计算熵
    probs = counts / total
    pe = -np.sum(probs * np.log(probs + 1e-10))

    # 归一化
    max_pe = np.log(math.factorial(m))
    if max_pe == 0:
        return 0.0

    return pe / max_pe


def sample_entropy(signal, m=2, r=0.2):
    """样本熵 (Sample Entropy) - 向量化实现

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

    def _count_matches(patterns):
        """统计匹配对数 - 向量化"""
        n_p = len(patterns)
        diff = patterns[:, np.newaxis, :] - patterns[np.newaxis, :, :]
        max_dists = np.max(np.abs(diff), axis=2)
        np.fill_diagonal(max_dists, np.inf)
        matches = np.sum(max_dists <= r_threshold, axis=1)
        return np.sum(matches)

    # 构建模式矩阵
    patterns_m = as_strided(signal, shape=(n - m, m), strides=(signal.strides[0], signal.strides[0])).copy()
    patterns_m1 = as_strided(signal, shape=(n - m - 1, m + 1), strides=(signal.strides[0], signal.strides[0])).copy()

    A = _count_matches(patterns_m1)
    B = _count_matches(patterns_m)

    n_m1 = n - m - 1
    n_m = n - m
    A = A / (n_m1 * (n_m1 - 1) + 1e-10)
    B = B / (n_m * (n_m - 1) + 1e-10)

    if A == 0 or B == 0:
        return 0.0

    return -np.log(A / B + 1e-10)
