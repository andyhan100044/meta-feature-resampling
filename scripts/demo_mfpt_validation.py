"""
MFPT跨数据集验证演示
工作流程:
1. 在CWRU上训练推荐器并保存
2. 加载MFPT数据
3. 加载推荐器并验证

Usage:
    python scripts/demo_mfpt_validation.py
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment.pipeline import ExperimentPipeline


def quick_load_cwru_subset(pipeline, max_files_per_type=1, max_frames_per_file=50):
    """快速加载CWRU子集用于演示"""
    from src.data.parse_cwru import parse_cwru_mat, create_frames

    fault_dir = os.path.join(pipeline.data_dir, "12k_Drive_End_Bearing_Fault_Data")
    X_list, y_list = [], []

    for fault_type in ['B', 'IR', 'OR']:
        for size in ['007']:
            size_dir = os.path.join(fault_dir, fault_type, size)
            if not os.path.exists(size_dir):
                continue
            files = sorted([f for f in os.listdir(size_dir) if f.endswith('.mat')])[:max_files_per_type]
            for filename in files:
                if int(filename.replace('.mat', '').split('_')[-1]) != 0:
                    continue
                signal = parse_cwru_mat(os.path.join(size_dir, filename))
                frames = create_frames(signal, pipeline.frame_size, pipeline.overlap_ratio)
                for frame in frames[:max_frames_per_file]:
                    vec = pipeline.extractor.to_vector(pipeline.extractor.extract(frame))
                    X_list.append(vec)
                    y_list.append(1)

    # 正常数据
    normal_dir = os.path.join(pipeline.data_dir, "Normal")
    for filename in sorted(os.listdir(normal_dir)):
        if not filename.endswith('.mat'):
            continue
        if int(filename.replace('.mat', '').split('_')[-1]) != 0:
            continue
        signal = parse_cwru_mat(os.path.join(normal_dir, filename))
        frames = create_frames(signal, pipeline.frame_size, pipeline.overlap_ratio)
        for frame in frames[:30]:
            vec = pipeline.extractor.to_vector(pipeline.extractor.extract(frame))
            X_list.append(vec)
            y_list.append(0)
        break

    return np.array(X_list), np.array(y_list)


def main():
    print("=" * 60)
    print("MFPT Cross-Dataset Validation Demo")
    print("=" * 60)

    # Step 1: 创建流水线
    print("\n[Step 1] 创建实验流水线...")
    pipeline = ExperimentPipeline(
        data_dir='E:/000001research/CWRU-dataset',
        sample_rate=12000,
        frame_size=1024,
        overlap_ratio=0.5
    )
    print("  Pipeline created")

    # Step 2: 加载CWRU子集并训练
    print("\n[Step 2] 加载CWRU子集并训练推荐器...")
    t0 = time.time()
    X, y = quick_load_cwru_subset(pipeline, max_files_per_type=1, max_frames_per_file=50)
    print(f"  CWRU loaded: X.shape={X.shape}, 0={sum(y==0)}, 1={sum(y==1)}, took {time.time()-t0:.1f}s")

    # IR=3 (minor=100, major=300)
    X_sub, y_sub = pipeline.subsample_to_ir(X, y, target_ir=3)
    print(f"  Subsampled IR=3: X.shape={X_sub.shape}, 0={sum(y_sub==0)}, 1={sum(y_sub==1)}")

    ir = np.full(len(y_sub), 3.0)
    pipeline.train_recommender(X_sub, y_sub, ir)
    print("  Recommender trained")

    # Step 3: 保存推荐器
    print("\n[Step 3] 保存推荐器...")
    model_dir = 'E:/000001research/models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'recommender_cwru.pkl')
    pipeline.save_recommender(model_path)
    print(f"  Saved to: {model_path}")

    # Step 4: 在MFPT上验证
    print("\n[Step 4] 加载MFPT数据并验证...")
    print("  (加载预转换的.mat文件)")

    t0 = time.time()
    X_mfpt, y_mfpt = pipeline.load_mfpt_dataset(
        'E:/000001research/MFPT_data/mat',
        fault_types=['inner', 'outer'],
        normal_data=True,
        max_frames_per_file=50,
        frame_size=1024,
        overlap_ratio=0.5
    )
    print(f"  MFPT loaded: X.shape={X_mfpt.shape}, 0={sum(y_mfpt==0)}, 1={sum(y_mfpt==1)}, took {time.time()-t0:.1f}s")

    # 构造IR=3
    X_mfpt_sub, y_mfpt_sub = pipeline.subsample_to_ir(X_mfpt, y_mfpt, target_ir=3)
    print(f"  Subsampled IR=3: X.shape={X_mfpt_sub.shape}, 0={sum(y_mfpt_sub==0)}, 1={sum(y_mfpt_sub==1)}")

    # 加载保存的推荐器
    pipeline2 = ExperimentPipeline(sample_rate=12000)
    pipeline2.load_recommender(model_path)
    print("  Recommender loaded from disk")

    # 运行验证
    print("\n[Step 5] 运行对比实验...")
    strategies = ['SMOTE', 'BorderlineSMOTE', 'RUS', 'SMOTETomek']
    print(f"  Testing {len(strategies)} fixed strategies + Ours on MFPT (IR=3, CV=3)...")

    for strategy in strategies:
        f1 = pipeline.run_single_experiment(X_mfpt_sub, y_mfpt_sub, strategy, cv_folds=3)
        print(f"    Fixed-{strategy}: F1-macro={f1:.4f}")

    f1_ours = pipeline.run_single_experiment(X_mfpt_sub, y_mfpt_sub, pipeline2.recommender, cv_folds=3)
    print(f"    Ours (loaded): F1-macro={f1_ours:.4f}")

    print("\n" + "=" * 60)
    print("Done! Recommender trained on CWRU, validated on MFPT.")
    print("=" * 60)


if __name__ == "__main__":
    main()
