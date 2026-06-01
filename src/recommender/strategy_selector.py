"""
策略选择器 - 基于规则的记忆推理
用于训练阶段生成推荐器的训练数据
"""
import numpy as np
from collections import Counter

class StrategySelector:
    """
    基于信号特征的策略选择器

    使用启发式规则为训练数据生成策略标签
    这些规则基于领域知识设计:
    - IR高 → 需要激进过采样 (ADASYN, BorderlineSMOTE)
    - 熵低 → 信号规律性强 → BorderlineSMOTE有效
    - 熵高 → 信号复杂 → ADASYN自适应
    """

    def __init__(self):
        self.strategies = ['SMOTE', 'BorderlineSMOTE', 'ADASYN', 'RUS', 'SMOTETomek']

    def select_by_features(self, ir, permutation_entropy, spectral_entropy,
                          main_freq_ratio=None, rms_freq=None):
        """基于特征选择策略

        Args:
            ir: 不平衡率
            permutation_entropy: 排列熵
            spectral_entropy: 谱熵
            main_freq_ratio: 主频能量比 (可选)
            rms_freq: 均方根频率 (可选)

        Returns:
            int: 策略索引 (0-4)
        """
        # 高IR + 低熵 → BorderlineSMOTE (边界精准造样)
        if ir > 20 and permutation_entropy < 0.8:
            return 1  # BorderlineSMOTE

        # 高谱熵 → 信号复杂 → ADASYN自适应
        if spectral_entropy > 2.5:
            return 2  # ADASYN

        # 非常高IR → SMOTETomek (混合策略)
        if ir > 30:
            return 4  # SMOTETomek

        # 中等IR → SMOTE
        if ir > 10:
            return 0  # SMOTE

        # 低IR → RUS足够
        return 3  # RUS

    def select_batch(self, X, ir_values):
        """批量选择策略

        Args:
            X: 特征矩阵 (n_samples, n_features)
            ir_values: 不平衡率列表

        Returns:
            np.ndarray: 策略索引数组
        """
        labels = []
        for i, ir in enumerate(ir_values):
            # 特征顺序: rms, variance, peak_to_peak, shape_factor,
            #         impulse_factor, clearance_factor, main_freq_ratio,
            #         spectral_entropy, centroid_freq, rms_freq,
            #         sample_entropy, permutation_entropy
            perm_ent = X[i, 11] if X.shape[1] > 11 else 0.5
            spec_ent = X[i, 7] if X.shape[1] > 7 else 1.0
            main_freq = X[i, 6] if X.shape[1] > 6 else 0.5

            label = self.select_by_features(ir, perm_ent, spec_ent, main_freq)
            labels.append(label)

        return np.array(labels)