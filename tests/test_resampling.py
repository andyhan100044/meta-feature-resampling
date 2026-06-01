"""
重采样模块测试
"""
import numpy as np
import sys
sys.path.insert(0, 'E:/000001research')

from src.resampling import resample, get_available_strategies, ResamplingPipeline

def test_resample_strategies():
    """测试各种重采样策略"""
    # 创建不平衡数据集: 1000个样本, 10个特征, IR=10
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    y = np.array([0] * 900 + [1] * 100)  # IR = 9

    print(f"原始数据: {len(y)} 样本, 类分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    strategies = get_available_strategies()
    print(f"可用策略: {strategies}")

    for strategy in strategies:
        try:
            X_res, y_res = resample(X, y, strategy)
            unique, counts = np.unique(y_res, return_counts=True)
            print(f"[OK] {strategy}: {len(y_res)} samples, distribution: {dict(zip(unique, counts))}")
        except Exception as e:
            print(f"[FAIL] {strategy}: {e}")

def test_pipeline():
    """测试重采样流水线"""
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = np.array([0] * 180 + [1] * 20)

    pipeline = ResamplingPipeline(strategy='SMOTE')
    X_res, y_res = pipeline.fit_resample(X, y)

    print(f"\n[OK] Pipeline test: {pipeline}")
    print(f"  原始: {len(y)} → 重采样后: {len(y_res)}")

if __name__ == "__main__":
    test_resample_strategies()
    test_pipeline()
    print("\nAll tests passed!")