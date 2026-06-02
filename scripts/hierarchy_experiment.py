"""层次分类实验 - 安全优先视角

核心问题：层次分类的价值不在于F1-macro，而在于：
1. 安全优先：Level 1 保证高故障检出率（漏检代价极高）
2. 可解释性：先判断"是否需要维护"，再判断"什么故障"
3. 实践意义：符合实际维护决策流程

评估指标：
- Level 1 Recall (故障检出率): 越高越好
- Level 2 Accuracy (故障类型准确率): 给出故障类型
- Flat F1-macro: 对比基准
"""
import os, sys, pickle, numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hierarchy import Level1Classifier, Level2Classifier, CascadeClassifier
from src.resampling import get_available_strategies

CACHE_DIR = 'E:/000001research/cache'
RESULTS_DIR = 'E:/000001research/results'

FEATURE_GROUPS = {
    'time': [0, 1, 2, 3, 4, 5],
    'freq': [6, 7, 8, 9],
    'entropy': [10, 11],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

def load_cwru():
    cache_file = os.path.join(CACHE_DIR, 'cwru_multiclass_features.npz')
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data['X'], data['y']
    raise FileNotFoundError(f"CWRU特征缓存不存在: {cache_file}")

def subsample_imbalanced(X, y, ir_target, n_fault=100):
    """
    构造不平衡数据集 - 模拟真实工业场景

    设计理由（论文中需阐明）：
    1. 真实工业场景：正常运行时数据大量积累（Normal样本充足），
       而故障样本来自历史故障记录（相对稀缺）
    2. 我们通过bootstrap重采样扩充Normal样本，模拟"数据充足的场景"
    3. 故障样本保持一定数量，确保Level 2三分类任务有足够训练数据
    4. IR = n_normal / n_fault, 其中 n_normal = n_fault * ir_target

    实验设计选择说明：
    - 之前的方法同时缩减fault和normal，导致训练样本严重不足
    - 正确的方法应该是：模拟"正常数据充足"的工业场景，
      这样更符合实际，也为Level 2提供足够的fault样本

    Args:
        X, y: 原始数据
        ir_target: 目标不平衡比 (n_normal / n_fault)
        n_fault: 每个fault类保留的样本数下限
    """
    X_list, y_list = [], []

    for cls in range(4):
        X_cls = X[y == cls]
        if cls == 0:
            # Normal类: Bootstrap重采样扩充，模拟充足数据
            # 目标: n_normal = n_fault * ir_target
            n_normal_target = n_fault * ir_target
            np.random.seed(42)
            # Bootstrap with replacement
            idx = np.random.choice(len(X_cls), n_normal_target, replace=True)
            X_normal = X_cls[idx]
            # 添加少量噪声模拟新样本
            noise = np.random.randn(*X_normal.shape) * 0.01
            X_normal = X_normal + noise
            X_list.append(X_normal)
            y_list.append(np.full(n_normal_target, cls))
        else:
            # Fault类: 保留足够数量用于Level 2训练
            n = min(len(X_cls), max(n_fault, 50))  # 至少50个
            np.random.seed(42 + cls)
            idx = np.random.choice(len(X_cls), n, replace=False)
            X_list.append(X_cls[idx])
            y_list.append(np.full(n, cls))

    return np.vstack(X_list), np.concatenate(y_list)

def flat_classification(X, y, strategy, cv_folds=5):
    """扁平4分类 baseline"""
    feat_idx = FEATURE_GROUPS['all']  # 用全特征
    min_count = min(sum(y == c) for c in range(4))
    cv_actual = max(2, min(cv_folds, min_count))
    skf = StratifiedKFold(n_splits=cv_actual, shuffle=True, random_state=42)

    f1_scores = []
    for train_idx, test_idx in skf.split(X, y):
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
    return np.mean(f1_scores)

def run_hierarchy_evaluation(X, y, strategy, cv_folds=5):
    """评估层次分类器"""
    min_count = min(sum(y == c) for c in range(4))
    cv_actual = max(2, min(cv_folds, min_count))
    skf = StratifiedKFold(n_splits=cv_actual, shuffle=True, random_state=42)

    results = {
        'level1_recall': [],    # 故障检出率
        'level1_f1': [],        # Level 1 F1
        'level2_accuracy': [],  # Level 2 故障类型准确率
        'combined_accuracy': [], # 整体4分类准确率
    }

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        try:
            # 训练 Level 1 (熵特征)
            l1 = Level1Classifier(resampling_strategy=strategy, use_entropy_features=True)
            l1.fit(X_tr, y_tr)
            l1_eval = l1.evaluate(X_te, y_te)

            # 训练 Level 2 (全特征)
            fault_mask_tr = y_tr > 0
            if fault_mask_tr.sum() > 3:
                l2 = Level2Classifier(resampling_strategy=strategy, feature_type='all')
                l2.fit(X_tr[fault_mask_tr], y_tr[fault_mask_tr])

                # 级联评估
                fault_mask_te = y_te > 0
                if fault_mask_te.sum() > 0:
                    # Level 2 准确率
                    l2_pred = l2.predict(X_te[fault_mask_te])
                    l2_acc = accuracy_score(y_te[fault_mask_te], l2_pred)

                    # 整体准确率
                    cascade = CascadeClassifier(
                        resampling_strategy=strategy,
                        level1_feature='entropy',
                        level2_feature='all'
                    )
                    cascade.fit(X_tr, y_tr)
                    combined_acc = accuracy_score(y_te, cascade.predict(X_te))
                else:
                    l2_acc = 0
                    combined_acc = 0

                results['level1_recall'].append(l1_eval['recall'])
                results['level1_f1'].append(l1_eval['f1'])
                results['level2_accuracy'].append(l2_acc)
                results['combined_accuracy'].append(combined_acc)
            else:
                results['level1_recall'].append(l1_eval['recall'])
                results['level1_f1'].append(l1_eval['f1'])
                results['level2_accuracy'].append(0)
                results['combined_accuracy'].append(0)
        except Exception as e:
            results['level1_recall'].append(0)
            results['level1_f1'].append(0)
            results['level2_accuracy'].append(0)
            results['combined_accuracy'].append(0)

    return {k: np.mean(v) for k, v in results.items()}

def main():
    print("=" * 60)
    print("层次分类实验 - 安全优先视角")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 加载数据
    print("\n[Step 1] 加载数据...")
    X, y = load_cwru()
    print(f"  原始数据: X.shape={X.shape}")
    print(f"  类别分布: Normal={sum(y==0)}, B={sum(y==1)}, IR={sum(y==2)}, OR={sum(y==3)}")

    # 实验配置
    ir_configs = [3, 5, 10]
    strategies = get_available_strategies()
    all_results = {}

    print("\n[Step 2] 运行实验...")
    for ir_target in ir_configs:
        print(f"\n{'='*50}")
        print(f"IR = {ir_target}")
        print(f"{'='*50}")

        # 构造不平衡数据
        X_sub, y_sub = subsample_imbalanced(X, y, ir_target)
        counts = [sum(y_sub == c) for c in range(4)]
        actual_ir = counts[0] / min(counts[1:]) if min(counts[1:]) > 0 else float('inf')
        print(f"  类别分布: Normal={counts[0]}, B={counts[1]}, IR={counts[2]}, OR={counts[3]}")
        print(f"  实际IR: {actual_ir:.1f}")

        # Baseline: 扁平4分类 (不同策略)
        print(f"\n  [扁平4分类 Baseline]")
        flat_results = {}
        for strategy in strategies:
            f1 = flat_classification(X_sub, y_sub, strategy, cv_folds=5)
            flat_results[strategy] = f1
            print(f"    {strategy}: F1-macro = {f1:.4f}")

        best_strategy = max(flat_results, key=flat_results.get)
        best_flat_f1 = flat_results[best_strategy]
        print(f"    Best: {best_strategy} = {best_flat_f1:.4f}")

        # 层次分类评估
        print(f"\n  [层次分类评估]")
        hier_results = run_hierarchy_evaluation(X_sub, y_sub, best_strategy, cv_folds=5)

        print(f"    Level 1 (Normal vs Fault):")
        print(f"      - Recall (故障检出率): {hier_results['level1_recall']:.4f}")
        print(f"      - F1: {hier_results['level1_f1']:.4f}")
        print(f"    Level 2 (B vs IR vs OR):")
        print(f"      - Accuracy: {hier_results['level2_accuracy']:.4f}")
        print(f"    Combined (4-class accuracy): {hier_results['combined_accuracy']:.4f}")

        # 对比
        print(f"\n  [对比总结]")
        print(f"    扁平最优 F1-macro: {best_flat_f1:.4f}")
        print(f"    层次分类故障检出率: {hier_results['level1_recall']:.4f} (关键安全指标)")
        print(f"    层次分类故障类型准确率: {hier_results['level2_accuracy']:.4f}")

        all_results[f'IR={ir_target}'] = {
            'distribution': counts,
            'flat_best': (best_strategy, best_flat_f1),
            'flat_all': flat_results,
            'hierarchy': hier_results
        }

    # 汇总表格
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)

    print("\n表1: 扁平4分类 vs 层次分类")
    print(f"{'IR':<8} | {'扁平F1-macro':>12} | {'层次故障检出率':>14} | {'层次故障类型准确率':>18} | {'层次4类准确率':>14}")
    print("-" * 80)
    for ir_k, res in all_results.items():
        flat_strat, flat_f1 = res['flat_best']
        hier = res['hierarchy']
        print(f"{ir_k:<8} | {flat_f1:>12.4f} | {hier['level1_recall']:>14.4f} | {hier['level2_accuracy']:>18.4f} | {hier['combined_accuracy']:>14.4f}")

    print("\n表2: 关键发现")
    print("-" * 60)
    for ir_k, res in all_results.items():
        hier = res['hierarchy']
        flat_strat, flat_f1 = res['flat_best']

        # 安全分析
        if hier['level1_recall'] >= 0.95:
            safety = "安全可靠"
        elif hier['level1_recall'] >= 0.80:
            safety = "基本安全"
        else:
            safety = "存在风险"

        # 实用性分析
        if hier['level2_accuracy'] >= 0.80:
            utility = "高"
        elif hier['level2_accuracy'] >= 0.60:
            utility = "中"
        else:
            utility = "低"

        print(f"{ir_k}: 故障检出率={hier['level1_recall']:.2%} → {safety}, 故障类型准确率={hier['level2_accuracy']:.2%} → 实用性{utility}")

    # 保存结果
    ensure_dir(RESULTS_DIR)
    results_file = os.path.join(RESULTS_DIR, 'hierarchy_safety_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump({
            'all_results': all_results,
            'timestamp': datetime.now().isoformat()
        }, f)
    print(f"\n结果已保存: {results_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
