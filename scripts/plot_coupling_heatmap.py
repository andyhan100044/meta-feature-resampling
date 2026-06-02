"""绘制策略-特征耦合矩阵热力图"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.resampling import get_available_strategies

CACHE_DIR = 'E:/000001research/cache'
RESULTS_DIR = 'E:/000001research/results'

FEATURE_GROUPS = {
    'time': [0, 1, 2, 3, 4, 5],
    'freq': [6, 7, 8, 9],
    'entropy': [10, 11],
    'time+freq': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'time+entropy': [0, 1, 2, 3, 4, 5, 10, 11],
    'freq+entropy': [6, 7, 8, 9, 10, 11],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}

FEATURE_NAMES_DISPLAY = ['time', 'freq', 'entropy', 'time+freq', 'time+entropy', 'freq+entropy', 'all']
STRATEGIES = get_available_strategies()  # SMOTE, BorderlineSMOTE, ADASYN, RUS, SMOTETomek

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_cwru():
    cache_file = os.path.join(CACHE_DIR, 'cwru_multiclass_features.npz')
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data['X'], data['y']
    raise FileNotFoundError(f"CWRU特征缓存不存在")

def resample_safe(X, y, strategy):
    from src.resampling import resample
    try:
        return resample(X, y, strategy)
    except:
        if strategy == 'SMOTE':
            try:
                return resample(X, y, 'BorderlineSMOTE')
            except:
                pass
        if strategy in ('SMOTE', 'BorderlineSMOTE'):
            try:
                return resample(X, y, 'RUS')
            except:
                pass
        if strategy != 'RUS':
            return resample(X, y, 'RUS')
        raise

def subsample_imbalanced(X, y, ir_target):
    normal_count = sum(y == 0)
    target_fault = int(normal_count / ir_target)
    X_list, y_list = [], []
    for cls in range(4):
        X_cls = X[y == cls]
        n = normal_count if cls == 0 else min(len(X_cls), target_fault)
        np.random.seed(42 + cls)
        idx = np.random.choice(len(X_cls), n, replace=False)
        X_list.append(X_cls[idx])
        y_list.append(np.full(n, cls))
    return np.vstack(X_list), np.concatenate(y_list)

def build_matrix(X, y, strategies, feature_groups, cv_folds=5):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score

    min_count = min(sum(y == c) for c in range(4))
    cv_actual = max(2, min(cv_folds, min_count))
    skf = StratifiedKFold(n_splits=cv_actual, shuffle=True, random_state=42)
    folds = list(skf.split(X, y))

    matrix = {}
    for strategy in strategies:
        matrix[strategy] = {}
        for feat_name, feat_idx in feature_groups.items():
            f1_scores = []
            for train_idx, test_idx in folds:
                X_tr, X_te = X[train_idx][:, feat_idx], X[test_idx][:, feat_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                try:
                    X_r, y_r = resample_safe(X_tr, y_tr, strategy)
                    clf = SVC(kernel='rbf', random_state=42)
                    clf.fit(X_r, y_r)
                    f1 = f1_score(y_te, clf.predict(X_te), average='macro')
                    f1_scores.append(f1)
                except:
                    f1_scores.append(0.0)
            matrix[strategy][feat_name] = np.mean(f1_scores)
    return matrix

def plot_heatmap(matrix, strategies, feature_names, title, save_path):
    """绘制单个热力图"""
    n_strategies = len(strategies)
    n_features = len(feature_names)

    data = np.zeros((n_strategies, n_features))
    for i, s in enumerate(strategies):
        for j, f in enumerate(feature_names):
            data[i, j] = matrix[s][f]

    fig, ax = plt.subplots(figsize=(10, 6))

    # 使用YlOrRd colormap，突出显示高值
    cmap = 'YlOrRd'
    vmin, vmax = 0.4, 0.9

    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    # 添加数值标签
    for i in range(n_strategies):
        for j in range(n_features):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=9)

    # 设置标签
    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_strategies))
    ax.set_xticklabels(feature_names, fontsize=10)
    ax.set_yticklabels(strategies, fontsize=10)

    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')

    ax.set_xlabel('Feature Set', fontsize=12)
    ax.set_ylabel('Resampling Strategy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('F1-macro', rotation=270, labelpad=15, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("=" * 50)
    print("绘制策略-特征耦合矩阵热力图")
    print("=" * 50)

    # 加载数据
    print("\n[Step 1] 加载CWRU数据...")
    X, y = load_cwru()
    print(f"  数据: X.shape={X.shape}")

    # 对三个IR值构建矩阵
    ir_configs = [3, 5, 10]
    matrices = {}
    feature_names = list(FEATURE_GROUPS.keys())  # 7个特征集

    for target_ir in ir_configs:
        print(f"\n[Step 2] 构建IR={target_ir}的耦合矩阵...")
        X_sub, y_sub = subsample_imbalanced(X, y, target_ir)
        counts = [sum(y_sub == c) for c in range(4)]
        actual_ir = counts[0] / min(counts[1:])
        print(f"  各类: {counts}, 实际IR: {actual_ir:.1f}")

        matrix = build_matrix(X_sub, y_sub, STRATEGIES, FEATURE_GROUPS, cv_folds=5)
        matrices[f'IR={target_ir}'] = matrix
        print(f"  矩阵构建完成")

    # 保存矩阵数据
    ensure_dir(RESULTS_DIR)
    matrix_data_file = os.path.join(RESULTS_DIR, 'strategy_feature_matrix.pkl')
    with open(matrix_data_file, 'wb') as f:
        pickle.dump(matrices, f)
    print(f"\n  矩阵数据已保存: {matrix_data_file}")

    # 绘制三张热力图
    ensure_dir(os.path.join(RESULTS_DIR, 'figures'))

    for target_ir in ir_configs:
        matrix = matrices[f'IR={target_ir}']
        title = f'Strategy-Feature Coupling Matrix (IR={target_ir})'
        save_path = os.path.join(RESULTS_DIR, 'figures', f'coupling_matrix_ir{target_ir}.png')
        plot_heatmap(matrix, STRATEGIES, feature_names, title, save_path)

    # 绘制汇总对比图（三个矩阵并排）
    print(f"\n[Step 3] 绘制汇总对比图...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, target_ir in enumerate(ir_configs):
        ax = axes[idx]
        matrix = matrices[f'IR={target_ir}']

        data = np.zeros((len(STRATEGIES), len(feature_names)))
        for i, s in enumerate(STRATEGIES):
            for j, f in enumerate(feature_names):
                data[i, j] = matrix[s][f]

        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0.4, vmax=0.9)

        for i in range(len(STRATEGIES)):
            for j in range(len(feature_names)):
                ax.text(j, i, f'{data[i, j]:.2f}',
                       ha='center', va='center', color='black', fontsize=8)

        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_yticks(np.arange(len(STRATEGIES)))
        ax.set_xticklabels(feature_names, fontsize=9)
        ax.set_yticklabels(STRATEGIES, fontsize=9)
        ax.set_title(f'IR={target_ir}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Set', fontsize=10)

        # 在第一张图添加ylabel
        if idx == 0:
            ax.set_ylabel('Resampling Strategy', fontsize=10)

    # 添加总标题
    fig.suptitle('Strategy-Feature Coupling Analysis Across IR Ratios', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    summary_path = os.path.join(RESULTS_DIR, 'figures', 'coupling_matrix_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {summary_path}")

    # 打印矩阵数值
    print("\n" + "=" * 60)
    print("策略-特征耦合矩阵数值")
    print("=" * 60)

    for target_ir in ir_configs:
        matrix = matrices[f'IR={target_ir}']
        print(f"\nIR={target_ir}:")
        header = f"{'Strategy':<20} |" + "".join([f" {f:>10} |" for f in feature_names])
        print(header)
        print("-" * (24 + 13 * len(feature_names)))
        for strategy in STRATEGIES:
            row = f"{strategy:<20} |"
            for feat_name in feature_names:
                val = matrix[strategy][feat_name]
                row += f" {val:>10.4f} |"
            print(row)

    print("\n" + "=" * 60)
    print("热力图已保存到:")
    for target_ir in ir_configs:
        print(f"  figures/coupling_matrix_ir{target_ir}.png")
    print(f"  figures/coupling_matrix_summary.png")
    print("=" * 60)


if __name__ == "__main__":
    main()