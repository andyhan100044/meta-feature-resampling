"""
时域特征提取
包括: 均值、方差、RMS、峰峰值、波形因子、脉冲因子、裕度因子
"""
import numpy as np

def extract_time_features(signal):
    """提取时域特征

    Args:
        signal: 振动信号数组

    Returns:
        dict: 时域特征字典
    """
    n = len(signal)
    mean = np.mean(signal)
    var = np.var(signal)
    rms = np.sqrt(np.mean(signal**2))
    peak_to_peak = np.max(signal) - np.min(signal)

    # 波形因子 (Shape Factor)
    shape_factor = rms / (np.mean(np.abs(signal)) + 1e-10)

    # 脉冲因子 (Impulse Factor)
    impulse_factor = np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-10)

    # 裕度因子 (Clearance Factor)
    sqrt_abs_mean = np.mean(np.sqrt(np.abs(signal)))**2
    clearance_factor = np.max(np.abs(signal)) / (sqrt_abs_mean + 1e-10)

    return {
        'mean': float(mean),
        'variance': float(var),
        'rms': float(rms),
        'peak_to_peak': float(peak_to_peak),
        'shape_factor': float(shape_factor),
        'impulse_factor': float(impulse_factor),
        'clearance_factor': float(clearance_factor)
    }