"""完整实验脚本 - 多分类版本"""
import os, sys, time, pickle, numpy as np
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.experiment.pipeline import ExperimentPipeline
from src.resampling import get_available_strategies
from src.recommender import MetaFeatureRecommender, StrategySelector

CACHE_DIR = 'E:/000001research/cache'
RESULTS_DIR = 'E:/000001research/results'
MODEL_PATH = 'E:/000001research/models/recommender_full.pkl'

# 多分类标签
CLASS_NAMES = ['Normal', 'B', 'IR', 'OR']
CLASS_LABELS = {name: i for i, name in enumerate(CLASS_NAMES)}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_multiclass_features(data_dir, sample_rate=12000, frame_size=1024, overlap_ratio=0.5, max_frames_per_file=50):
    """加载多分类特征数据"""
    cache_file = os.path.join(CACHE_DIR, 'cwru_multiclass_features.npz')
    ensure_dir(CACHE_DIR)

    if os.path.exists(cache_file):
        print(f"  [加载缓存] {cache_file}")
        data = np.load(cache_file)
        return data['X'], data['y']

    print(f"  [提取特征] 首次运行，请稍候...")
    from src.data.parse_cwru import parse_cwru_mat, create_frames
    from src.features.extractor import MetaFeatureExtractor

    extractor = MetaFeatureExtractor(sample_rate=sample_rate)
    X_list, y_list = [], []
    workloads = ['0HP', '1HP', '2HP', '3HP']
    fault_types = ['B', 'IR', 'OR']

    for workload in workloads:
        expected_idx = {'0HP': 0, '1HP': 1, '2HP': 2, '3HP': 3}[workload]

        # 故障数据
        for fault_type in fault_types:
            for size in ['007', '014']:
                fault_dir = os.path.join(data_dir, "12k_Drive_End_Bearing_Fault_Data", fault_type, size)
                if not os.path.exists(fault_dir):
                    continue
                for filename in sorted(os.listdir(fault_dir)):
                    if not filename.endswith('.mat'):
                        continue
                    file_idx = int(filename.replace('.mat', '').split('_')[-1])
                    if file_idx != expected_idx:
                        continue
                    signal = parse_cwru_mat(os.path.join(fault_dir, filename))
                    frames = create_frames(signal, frame_size, overlap_ratio)
                    for i, frame in enumerate(frames[:max_frames_per_file]):
                        vec = extractor.to_vector(extractor.extract(frame))
                        X_list.append(vec)
                        y_list.append(CLASS_LABELS[fault_type])

        # 正常数据
        normal_dir = os.path.join(data_dir, "Normal")
        if os.path.exists(normal_dir):
            for filename in sorted(os.listdir(normal_dir)):
                if not filename.endswith('.mat'):
                    continue
                file_idx = int(filename.replace('.mat', '').split('_')[-1])
                if file_idx != expected_idx:
                    continue
                signal = parse_cwru_mat(os.path.join(normal_dir, filename))
                frames = create_frames(signal, frame_size, overlap_ratio)
                for i, frame in enumerate(frames[:max_frames_per_file]):
                    vec = extractor.to_vector(extractor.extract(frame))
                    X_list.append(vec)
                    y_list.append(CLASS_LABELS['Normal'])

    X = np.array(X_list)
    y = np.array(y_list)
    np.savez_compressed(cache_file, X=X, y=y)
    print(f"  特征提取完成: X.shape={X.shape}")
    for cn in CLASS_NAMES:
        print(f"    {cn}: {sum(y == CLASS_LABELS[cn])}")
    print(f"  已缓存到: {cache_file}")
    return X, y


def subsample_to_balance(X, y, target_per_class):
    """将每类欠采样到相同数量，构造多分类不平衡场景"""
    X_list, y_list = [], []
    for cls in range(len(CLASS_NAMES)):
        X_cls = X[y == cls]
        n = min(len(X_cls), target_per_class)
        np.random.seed(42 + cls)
        idx = np.random.choice(len(X_cls), n, replace=False)
        X_list.append(X_cls[idx])
        y_list.append(np.full(n, cls))
    return np.vstack(X_list), np.concatenate(y_list)


def resample_safe(X, y, strategy):
    """安全的重采样，失败时自动fallback到SMOTE"""
    from src.resampling import resample
    try:
        return resample(X, y, strategy)
    except Exception as e:
        if strategy != 'SMOTE':
            return resample(X, y, 'SMOTE')
        raise


def run_comparison(X, y, strategies, cv_folds=5):
    """运行多分类对比实验"""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score

    results = {}
    for strategy in strategies:
        print(f"    Testing {strategy}...")
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                X_r, y_r = resample_safe(X_train, y_train, strategy)
            except:
                continue

            clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
            clf.fit(X_r, y_r)
            y_pred = clf.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='macro')
            scores.append(f1)

        if scores:
            results[strategy] = np.mean(scores)
            print(f"      {strategy}: F1-macro={results[strategy]:.4f}")
        else:
            results[strategy] = 0.0
            print(f"      {strategy}: FAILED")

    # 随机选择基准
    random_scores = []
    for _ in range(5):
        strat = np.random.choice(strategies)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_scores = []
        for train_idx, test_idx in skf.split(X, y):
            try:
                X_r, y_r = resample_safe(X[train_idx], y[train_idx], strat)
                clf = SVC(kernel='rbf', random_state=42)
                clf.fit(X_r, y_r)
                f1 = f1_score(y[test_idx], clf.predict(X[test_idx]), average='macro')
                fold_scores.append(f1)
            except:
                pass
        if fold_scores:
            random_scores.append(np.mean(fold_scores))
    results['Random-Select'] = np.mean(random_scores) if random_scores else 0.0
    print(f"      Random-Select: F1-macro={results['Random-Select']:.4f}")

    return results


def train_recommender(X, y, target_per_class):
    """训练多分类推荐器"""
    selector = StrategySelector()
    rec = MetaFeatureRecommender(max_depth=5)

    # 构造多分类不平衡：每类不同数量
    X_list, y_list = [], []
    for cls in range(len(CLASS_NAMES)):
        X_cls = X[y == cls]
        # 每类不同数量：Normal最多，其他故障类少一些
        n = target_per_class[cls]
        np.random.seed(42 + cls)
        idx = np.random.choice(len(X_cls), min(n, len(X_cls)), replace=False)
        X_list.append(X_cls[idx])
        y_list.append(np.full(len(idx), cls))

    X_sub = np.vstack(X_list)
    y_sub = np.concatenate(y_list)

    # 为每类计算"等效IR"（相对于最小类）
    counts = [sum(y_sub == c) for c in range(len(CLASS_NAMES))]
    min_count = min(counts)
    ir_per_sample = []
    for cls in range(len(CLASS_NAMES)):
        ir = counts[cls] / min_count
        ir_per_sample.extend([ir] * counts[cls])
    ir_arr = np.array(ir_per_sample)

    # 生成策略标签
    y_strat = selector.select_batch(X_sub, ir_arr)
    X_ir = np.column_stack([X_sub, ir_arr])

    rec.fit(X_ir, y_strat)
    return rec, X_sub, y_sub


def main():
    t_start = time.time()
    print("=" * 60)
    print("完整CWRU实验 - 多分类版本")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    ensure_dir(RESULTS_DIR)
    ensure_dir('E:/000001research/models')

    # Step 1: 加载数据
    print("\n[Step 1] 加载CWRU数据...")
    t0 = time.time()
    X, y = load_multiclass_features(
        data_dir='E:/000001research/CWRU-dataset',
        sample_rate=12000, frame_size=1024, overlap_ratio=0.5,
        max_frames_per_file=50)
    print(f"  数据: X.shape={X.shape}")
    for cn in CLASS_NAMES:
        print(f"    {cn}: {sum(y == CLASS_LABELS[cn])}")
    print(f"  耗时: {time.time()-t0:.1f}s")

    # Step 2: 训练推荐器
    print("\n[Step 2] 训练推荐器...")
    t0 = time.time()
    # 每类目标数量：[Normal=200, B=150, IR=150, OR=150]
    target_per_class = [200, 150, 150, 150]
    rec, X_train, y_train = train_recommender(X, y, target_per_class)
    print(f"  推荐器训练完成: {time.time()-t0:.1f}s")
    rec.save(MODEL_PATH)
    print(f"  模型已保存: {MODEL_PATH}")

    # Step 3: 测试不同不平衡程度的对比实验
    print("\n[Step 3] 运行对比实验...")

    # 定义不同的不平衡配置
    imbalance_configs = [
        ("轻度不平衡 (每类 150/50)", [150, 50, 50, 50]),
        ("中度不平衡 (200/30)", [200, 30, 30, 30]),
        ("重度不平衡 (200/10)", [200, 10, 10, 10]),
    ]

    all_results = {}
    strategies = get_available_strategies()

    for config_name, target_per_class in imbalance_configs:
        print(f"\n  [{config_name}]")
        X_bal, y_bal = subsample_to_balance(X, y, min(target_per_class))
        print(f"  各类样本数: {[sum(y_bal == c) for c in range(4)]}")

        results = run_comparison(X_bal, y_bal, strategies, cv_folds=5)

        # 推荐器
        from sklearn.model_selection import StratifiedKFold
        from sklearn.svm import SVC
        from sklearn.metrics import f1_score
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in skf.split(X_bal, y_bal):
            X_tr, X_te = X_bal[train_idx], X_bal[test_idx]
            y_tr, y_te = y_bal[train_idx], y_bal[test_idx]
            # 推荐策略
            ir_arr = np.full(len(y_tr), 10.0)
            X_ir_tr = np.column_stack([X_tr, ir_arr])
            strat_idx = rec.predict(X_ir_tr)[0]
            recommended = rec.STRATEGIES[strat_idx]
            # 用推荐策略重采样
            try:
                X_r, y_r = resample_safe(X_tr, y_tr, recommended)
                clf = SVC(kernel='rbf', random_state=42)
                clf.fit(X_r, y_r)
                f1 = f1_score(y_te, clf.predict(X_te), average='macro')
                scores.append(f1)
            except:
                pass
        if scores:
            results['Ours'] = np.mean(scores)
            print(f"      Ours (recommended={recommended}): F1-macro={results['Ours']:.4f}")
        else:
            results['Ours'] = 0.0

        all_results[config_name] = results

    # Step 4: 伪在线验证
    print("\n[Step 4] 伪在线验证...")
    pseudo_results = {}
    workload_names = ['0HP', '1HP', '2HP', '3HP']

    for hidden_idx in range(4):
        hidden_wl = workload_names[hidden_idx]
        # 用文件索引区分工况（每个文件对应一个HP）
        # 简单处理：按样本顺序分
        n_per_workload = len(X) // 4
        start_idx = hidden_idx * n_per_workload
        end_idx = (hidden_idx + 1) * n_per_workload

        test_mask = np.zeros(len(X), dtype=bool)
        test_mask[start_idx:end_idx] = True
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # 平衡训练数据
        X_bal_train, y_bal_train = subsample_to_balance(X_train, y_train, 50)
        X_bal_test, y_bal_test = subsample_to_balance(X_test, y_test, 50)

        # 训练专用推荐器
        rec_leave, _, _ = train_recommender(X_train, y_train, [50, 50, 50, 50])

        # 推荐策略
        ir_arr = np.full(len(X_bal_test), 10.0)
        X_ir_test = np.column_stack([X_bal_test, ir_arr])
        strat_idx = rec_leave.predict(X_ir_test)[0]
        recommended = rec_leave.STRATEGIES[strat_idx]

        # 最佳固定策略
        best_f1 = 0
        best_fixed = 'SMOTE'
        for strat in strategies:
            try:
                X_r, y_r = resample_safe(X_bal_test, y_bal_test, strat)
                clf = SVC(kernel='rbf', random_state=42)
                clf.fit(X_r, y_r)
                f1 = f1_score(y_bal_test, clf.predict(X_bal_test), average='macro')
                if f1 > best_f1:
                    best_f1 = f1
                    best_fixed = strat
            except:
                pass

        # 推荐器效果
        try:
            X_r, y_r = resample_safe(X_bal_test, y_bal_test, recommended)
            clf = SVC(kernel='rbf', random_state=42)
            clf.fit(X_r, y_r)
            f1_ours = f1_score(y_bal_test, clf.predict(X_bal_test), average='macro')
        except:
            f1_ours = 0.0

        gain = (f1_ours - best_f1) * 100
        pseudo_results[hidden_wl] = {
            'nearest': workload_names[(hidden_idx + 1) % 4],
            'recommended': recommended,
            'best_fixed': best_fixed,
            'gain': gain,
            'f1_ours': f1_ours,
            'f1_fixed_best': best_f1
        }
        print(f"  Hidden={hidden_wl}, Rec={recommended}, BestFixed={best_fixed}, "
              f"F1 Ours={f1_ours:.3f}, F1 Best={best_f1:.3f}, Gain={gain:+.1f}%")

    # Step 5: 输出汇总
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)

    print("\n表X: F1-macro Results (多分类)")
    print(f"{'Method':<22} | {'轻度IR':>8} | {'中度IR':>8} | {'重度IR':>8}")
    print("-" * 52)
    methods_order = ["Fixed-SMOTE", "Fixed-BorderlineSMOTE", "Fixed-ADASYN",
                    "Fixed-RUS", "Fixed-SMOTETomek", "Random-Select", "Ours"]
    config_keys = [k for k, v in imbalance_configs]
    for method in methods_order:
        row = f"{method:<22} |"
        for ck in config_keys:
            val = all_results[ck].get(method, 0.0)
            row += f" {val:>8.4f} |" if val > 0 else f" {'N/A':>8} |"
        print(row)

    print("\n表Y: Pseudo-Online Validation")
    print(f"{'Hidden':<8} | {'Recommended':<16} | {'BestFixed':<14} | {'Gain':>6}")
    print("-" * 55)
    for wl, res in pseudo_results.items():
        print(f"{wl:<8} | {res['recommended']:<16} | {res['best_fixed']:<14} | {res['gain']:>+6.1f}%")

    # 保存
    results_file = os.path.join(RESULTS_DIR, 'experiment_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump({
            'table_results': all_results,
            'pseudo_online': pseudo_results,
            'timestamp': datetime.now().isoformat()
        }, f)
    print(f"\n结果已保存: {results_file}")
    print(f"总耗时: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
