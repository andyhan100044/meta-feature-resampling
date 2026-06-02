"""
主程序入口
Meta-Feature Driven Adaptive Resampling 实验系统

Usage:
    python main.py --stage all              # 运行全部阶段
    python main.py --stage data             # 仅数据下载
    python main.py --stage experiment      # 运行实验
    python main.py --stage paper           # 生成论文
    python main.py --config config.yaml    # 指定配置文件
"""
import argparse
import yaml
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def stage_data(config):
    """数据准备阶段"""
    print("\n" + "="*60)
    print("Stage 1: 数据准备")
    print("="*60)
    print("\n[CWRU数据集]")
    print("CWRU数据集已本地存储在 CWRU-dataset/ 目录")
    print("\n数据准备完成!")

def stage_features(config):
    """特征提取阶段"""
    print("\n" + "="*60)
    print("Stage 2: 元特征提取")
    print("="*60)

    from src.features.extractor import MetaFeatureExtractor
    from src.data.parse_cwru import parse_cwru_mat, create_frames
    import numpy as np

    extractor = MetaFeatureExtractor(sample_rate=config['signal']['sample_rates'][0])
    data_dir = config['dataset']['data_dir']

    print(f"特征提取器初始化完成")
    print(f"帧大小: {config['signal']['frame_size']}")
    print(f"重叠率: {config['signal']['overlap_ratio']}")

    # 演示特征提取
    demo_signal = np.random.randn(config['signal']['frame_size'])
    features = extractor.extract(demo_signal)
    print(f"\n演示信号特征维度: {len(extractor.to_vector(features))}")
    print("元特征提取阶段完成!")

def stage_experiment(config):
    """实验阶段"""
    print("\n" + "="*60)
    print("Stage 3: 对比实验")
    print("="*60)

    from src.experiment import ExperimentPipeline, ResultComparator
    import numpy as np

    print("加载CWRU数据集...")

    pipeline = ExperimentPipeline(
        data_dir="E:/000001research/CWRU-dataset",
        sample_rate=config['signal']['sample_rates'][0],
        frame_size=config['signal']['frame_size'],
        overlap_ratio=config['signal']['overlap_ratio']
    )

    # 加载数据
    try:
        X, y = pipeline.load_dataset(
            workload='0HP',
            fault_types=['B', 'IR', 'OR'],
            normal_data=True
        )
        print(f"数据加载完成: X.shape={X.shape}, y.shape={y.shape}")

        # 构造不平衡场景
        ir_values = config['experiment']['imbalance_ratios']
        print(f"\n测试不平衡率: {ir_values}")

        # 运行对比实验(演示用IR=10)
        X_sub, y_sub = pipeline.subsample_to_ir(X, y, target_ir=10)
        results = pipeline.run_comparison(X_sub, y_sub, ir_values=10, cv_folds=3)

        print("\n实验结果 (IR=10):")
        for method, f1 in results.items():
            print(f"  {method}: F1-macro={f1:.4f}")

    except FileNotFoundError as e:
        print(f"警告: 数据文件未找到 - {e}")
        print("请确保CWRU数据集已下载到指定目录")

    print("\n实验阶段完成!")

def stage_paper(config):
    """论文生成阶段"""
    print("\n" + "="*60)
    print("Stage 4: 论文生成")
    print("="*60)

    from src.paper import PaperWriter

    writer = PaperWriter(output_dir="docs/paper")
    path = writer.save_paper("manuscript.tex")

    print(f"论文已生成: {path}")

    # 列出论文结构
    print("\n论文结构:")
    print("  - Abstract")
    print("  - Introduction")
    print("  - Related Work")
    print("  - Methodology")
    print("  - Experimental Results")
    print("  - Discussion")
    print("  - Conclusion")

    print("\n论文生成阶段完成!")

def stage_all(config):
    """运行全部阶段"""
    print("\n" + "="*60)
    print("Meta-Feature Driven Adaptive Resampling")
    print("完整实验流程")
    print("="*60)

    stage_features(config)
    stage_experiment(config)
    stage_paper(config)

    print("\n" + "="*60)
    print("全部阶段完成!")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Meta-Feature Driven Adaptive Resampling 实验系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --stage all              运行完整流程
  python main.py --stage experiment       仅运行实验
  python main.py --config my_config.yaml 使用自定义配置
        """
    )

    parser.add_argument(
        '--config',
        default='config.yaml',
        help='配置文件路径 (默认: config.yaml)'
    )
    parser.add_argument(
        '--stage',
        default='all',
        choices=['data', 'features', 'experiment', 'paper', 'all'],
        help='运行阶段 (默认: all)'
    )

    args = parser.parse_args()

    # 加载配置
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"错误: 配置文件不存在 - {config_path}")
        print("使用默认配置...")
        config = {
            'dataset': {'data_dir': 'data/raw'},
            'signal': {'sample_rates': [12000], 'frame_size': 1024, 'overlap_ratio': 0.5},
            'experiment': {'imbalance_ratios': [5, 10, 20, 50]}
        }
    else:
        config = load_config(config_path)

    print(f"配置加载: {args.config}")
    print(f"运行阶段: {args.stage}")

    # 执行对应阶段
    if args.stage == 'data':
        stage_data(config)
    elif args.stage == 'features':
        stage_features(config)
    elif args.stage == 'experiment':
        stage_experiment(config)
    elif args.stage == 'paper':
        stage_paper(config)
    elif args.stage == 'all':
        stage_all(config)

if __name__ == "__main__":
    main()