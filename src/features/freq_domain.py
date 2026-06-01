"""
频域特征提取
包括: 主频能量比、谱熵、重心频率、均方根频率
"""
import numpy as np
from scipy.signal import welch

def extract_freq_features(signal, sample_rate=12000):
    """提取频域特征

    Args:
        signal: 振动信号数组
        sample_rate: 采样率

    Returns:
        dict: 频域特征字典
    """
    # 计算功率谱密度
    freqs, psd = welch(signal, fs=sample_rate, nperseg=min(256, len(signal)))

    total_power = np.sum(psd) + 1e-10

    # 主频能量比
    main_freq_power = np.max(psd)
    main_freq_ratio = main_freq_power / total_power

    # 谱熵 (Spectral Entropy)
    psd_norm = psd / total_power
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

    # 重心频率 (Centroid Frequency)
    centroid_freq = np.sum(freqs * psd) / total_power

    # 均方根频率 (RMS Frequency)
    rms_freq = np.sqrt(np.sum((freqs**2) * psd) / total_power)

    return {
        'main_freq_ratio': float(main_freq_ratio),
        'spectral_entropy': float(spectral_entropy),
        'centroid_freq': float(centroid_freq),
        'rms_freq': float(rms_freq)
    }