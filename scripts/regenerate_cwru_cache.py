"""重新生成CWRU特征缓存

使用方法:
    python scripts/regenerate_cwru_cache.py

输出:
    cache/cwru_multiclass_features.npz (12维特征, 4类)
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment.pipeline import ExperimentPipeline

CACHE_DIR = 'E:/000001research/cache'
DATA_DIR = 'E:/000001research/CWRU-dataset'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    print("=" * 60)
    print("重新生成CWRU特征缓存")
    print("=" * 60)

    ensure_dir(CACHE_DIR)

    # 初始化实验管道
    pipeline = ExperimentPipeline(
        data_dir=DATA_DIR,
        sample_rate=12000,
        frame_size=1024,
        overlap_ratio=0.5
    )

    # 加载数据：0HP工况，4类（Normal, B, IR, OR）
    print("\n[Step 1] 加载CWRU数据...")
    X, y = pipeline.load_dataset(
        workload='0HP',
        fault_types=['B', 'IR', 'OR'],
        normal_data=True
    )

    print(f"  数据形状: X={X.shape}, y={y.shape}")
    print(f"  类别分布:")
    for cls, name in enumerate(['Normal', 'B', 'IR', 'OR']):
        count = sum(y == cls)
        print(f"    {cls} ({name}): {count}")

    # 保存缓存
    print("\n[Step 2] 保存缓存...")
    cache_file = os.path.join(CACHE_DIR, 'cwru_multiclass_features.npz')
    np.savez_compressed(cache_file, X=X, y=y)
    print(f"  已保存: {cache_file}")
    print(f"  文件大小: {os.path.getsize(cache_file) / 1024:.1f} KB")

    print("\n完成!")

if __name__ == "__main__":
    main()
