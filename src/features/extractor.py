"""
元特征提取器 - 统一接口
"""
import numpy as np
from .time_domain import extract_time_features
from .freq_domain import extract_freq_features
from .entropy import sample_entropy, permutation_entropy

class MetaFeatureExtractor:
    """元特征提取器

    提取信号元特征：时域(7) + 频域(4) + 熵(2) = 13维特征
    """

    def __init__(self, sample_rate=12000):
        self.sample_rate = sample_rate

    def extract(self, signal):
        """提取完整元特征向量

        Args:
            signal: 振动信号数组

        Returns:
            dict: 完整特征字典
        """
        time_feats = extract_time_features(signal)
        freq_feats = extract_freq_features(signal, self.sample_rate)

        # 熵特征
        samp_entropy = sample_entropy(signal)
        perm_entropy = permutation_entropy(signal)

        return {
            **time_feats,
            **freq_feats,
            'sample_entropy': float(samp_entropy),
            'permutation_entropy': float(perm_entropy)
        }

    def to_vector(self, features):
        """将特征字典转换为向量

        用于推荐器输入

        Returns:
            np.ndarray: 特征向量 (13维)
        """
        return np.array([
            features['rms'],
            features['variance'],
            features['peak_to_peak'],
            features['shape_factor'],
            features['impulse_factor'],
            features['clearance_factor'],
            features['main_freq_ratio'],
            features['spectral_entropy'],
            features['centroid_freq'],
            features['rms_freq'],
            features['sample_entropy'],
            features['permutation_entropy']
        ])

    def get_feature_names(self):
        """获取特征名称列表"""
        return [
            'rms', 'variance', 'peak_to_peak', 'shape_factor',
            'impulse_factor', 'clearance_factor', 'main_freq_ratio',
            'spectral_entropy', 'centroid_freq', 'rms_freq',
            'sample_entropy', 'permutation_entropy'
        ]