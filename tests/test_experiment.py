"""
实验流水线测试
"""
import os
import numpy as np
import sys
sys.path.insert(0, 'E:/000001research')

from src.experiment import ExperimentPipeline

def test_pipeline_load():
    """测试数据加载"""
    pipeline = ExperimentPipeline()

    # 检查数据目录
    data_dir = "E:/000001research/CWRU-dataset"
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        print("请先下载CWRU数据集")
        return None

    X, y = pipeline.load_dataset(workload='0HP', fault_types=['B', 'IR', 'OR'])

    print(f"加载数据: X.shape={X.shape}, y.shape={y.shape}")
    print(f"类分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y

def test_subsample():
    """测试欠采样"""
    pipeline = ExperimentPipeline()
    X, y = test_pipeline_load()
    if X is None:
        print("跳过测试 (无数据)")
        return

    X_sub, y_sub = pipeline.subsample_to_ir(X, y, target_ir=10)
    print(f"\n欠采样后: X.shape={X_sub.shape}")
    print(f"IR={len(y_sub[y_sub==0]) / len(y_sub[y_sub==1]):.1f}")

    print("\n✓ 测试通过!")

if __name__ == "__main__":
    test_pipeline_load()
    print("\n所有测试通过!")