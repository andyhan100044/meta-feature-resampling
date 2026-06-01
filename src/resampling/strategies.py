"""
重采样策略实现
包括: SMOTE, BorderlineSMOTE, ADASYN, RandomUnderSampler, SMOTETomek
"""
import numpy as np
from sklearn.utils import check_X_y
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE

STRATEGIES = ['SMOTE', 'BorderlineSMOTE', 'ADASYN', 'RUS', 'SMOTETomek']

def get_resampler(strategy, k_neighbors=5, random_state=42):
    """获取重采样器实例

    Args:
        strategy: 策略名称
        k_neighbors: k近邻参数
        random_state: 随机种子

    Returns:
        imblearn重采样器实例
    """
    resamplers = {
        'SMOTE': SMOTE(k_neighbors=k_neighbors, random_state=random_state),
        'BorderlineSMOTE': BorderlineSMOTE(k_neighbors=k_neighbors, random_state=random_state),
        'ADASYN': ADASYN(n_neighbors=k_neighbors, random_state=random_state),
        'RUS': RandomUnderSampler(random_state=random_state),
        'SMOTETomek': SMOTETomek(random_state=random_state),
    }

    if strategy not in resamplers:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(resamplers.keys())}")

    return resamplers[strategy]

def resample(X, y, strategy, k_neighbors=5, random_state=42):
    """执行重采样

    Args:
        X: 特征矩阵
        y: 标签
        strategy: 策略名称
        k_neighbors: k近邻参数
        random_state: 随机种子

    Returns:
        X_res, y_res: 重采样后的数据
    """
    resampler = get_resampler(strategy, k_neighbors, random_state)
    return resampler.fit_resample(X, y)

def get_available_strategies():
    """获取可用策略列表"""
    return STRATEGIES.copy()

class ResamplingPipeline:
    """重采样流水线"""

    def __init__(self, strategy='SMOTE', k_neighbors=5, random_state=42):
        self.strategy = strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.resampler = get_resampler(strategy, k_neighbors, random_state)

    def fit_resample(self, X, y):
        """执行重采样"""
        return self.resampler.fit_resample(X, y)

    def __repr__(self):
        return f"ResamplingPipeline(strategy='{self.strategy}', k_neighbors={self.k_neighbors})"