"""
特征提取模块测试
"""
import numpy as np
import sys
sys.path.insert(0, 'E:/000001research')

from src.features import (
    extract_time_features,
    extract_freq_features,
    sample_entropy,
    permutation_entropy,
    MetaFeatureExtractor
)

def test_time_features():
    """测试时域特征提取"""
    # 生成测试信号: 正弦波 + 噪声
    t = np.linspace(0, 1, 1024)
    signal = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(1024)

    features = extract_time_features(signal)

    assert 'rms' in features
    assert 'mean' in features
    assert features['rms'] > 0
    print(f"[PASS] Time domain features: RMS={features['rms']:.4f}")

def test_freq_features():
    """测试频域特征提取"""
    t = np.linspace(0, 1, 1024)
    signal = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(1024)

    features = extract_freq_features(signal, sample_rate=1024)

    assert 'spectral_entropy' in features
    assert 'centroid_freq' in features
    print(f"[PASS] Frequency domain features: spectral_entropy={features['spectral_entropy']:.4f}")

def test_entropy():
    """测试熵特征"""
    signal = np.random.randn(1024)

    samp_en = sample_entropy(signal)
    perm_en = permutation_entropy(signal)

    assert 0 <= samp_en <= 10
    assert 0 <= perm_en <= 1
    print(f"[PASS] Entropy features: sample_entropy={samp_en:.4f}, permutation_entropy={perm_en:.4f}")

def test_extractor():
    """测试元特征提取器"""
    extractor = MetaFeatureExtractor(sample_rate=12000)

    signal = np.random.randn(1024)
    features = extractor.extract(signal)
    vector = extractor.to_vector(features)

    assert len(vector) == 12
    assert len(extractor.get_feature_names()) == 12
    print(f"[PASS] Extractor: feature vector dimension={len(vector)}")

if __name__ == "__main__":
    test_time_features()
    test_freq_features()
    test_entropy()
    test_extractor()
    print("\n所有测试通过!")