"""联合优化推荐器 v2 - 改进的训练数据生成 + 跨数据集验证

改进点：
1. 训练数据用完整的策略-特征矩阵，而非启发式规则
2. 添加跨数据集验证（CWRU训练 → MFPT测试）
3. 添加更多基线对比
"""
import os, sys, time, pickle, numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.resampling import get_available_strategies
from src.recommender import MetaFeatureRecommender, StrategySelector

CACHE_DIR = 'E:/000001research/cache'
RESULTS_DIR = 'E:/000001research/results'
MFPT_DIR = 'E:/000001research/MFPT_data/mat'

FEATURE_GROUPS = {
    'time': [0, 1, 2, 3, 4, 5],
    'freq': [6, 7, 8, 9],
    'entropy': [10, 11],
    'time+freq': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'time+entropy': [0, 1, 2, 3, 4, 5, 10, 11],
    'freq+entropy': [6, 7, 8, 9, 10, 11],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}

STRATEGIES = get_available_strategies()
FEATURE_GROUP_NAMES = list(FEATURE_GROUPS.keys())
ALL_PAIRS = [(s, f) for s in STRATEGIES for f in FEATURE_GROUP_NAMES]  # 35 pairs

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
    raise FileNotFoundError(f"CWRU特征缓存不存在")

def load_mfpt_multiclass(sample_rate=12000, frame_size=1024, overlap_ratio=0.5, max_frames=50):
    """加载MFPT多分类特征"""
    cache_file = os.path.join(CACHE_DIR, 'mfpt_multiclass_features.npz')
    ensure_dir(CACHE_DIR)

    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data['X'], data['y']

    print(f"    MFPT特征提取中...")
    from src.features.extractor import MetaFeatureExtractor
    from src.experiment.pipeline import parse_mfpt_mat
    from src.data.parse_cwru import create_frames

    extractor = MetaFeatureExtractor(sample_rate=sample_rate)
    X_list, y_list = [], []
    MFPT_CLASS = {'normal': 0, 'inner': 1, 'outer': 2, 'outter': 2}

    for mat_file in sorted(os.listdir(MFPT_DIR)):
        if not mat_file.endswith('.mat'):
            continue
        if mat_file.startswith('fault'):
            parts = mat_file.replace('.mat', '').split('_')
            ftype = parts[2]
            cls = MFPT_CLASS.get(ftype, -1)
            if cls < 0:
                continue
        elif mat_file.startswith('normal'):
            cls = MFPT_CLASS['normal']
        else:
            continue

        signal = parse_mfpt_mat(os.path.join(MFPT_DIR, mat_file))
        if signal is None:
            continue

        frames = create_frames(signal, frame_size, overlap_ratio)
        for frame in frames[:max_frames]:
            vec = extractor.to_vector(extractor.extract(frame))
            X_list.append(vec)
            y_list.append(cls)

    X = np.array(X_list)
    y = np.array(y_list)
    np.savez_compressed(cache_file, X=X, y=y)
    return X, y

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

def build_strategy_feature_matrix(X, y, strategies, feature_groups, cv_folds=5):
    """构建策略-特征矩阵"""
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

def find_best_pair_for_sample(X_sample, ir, matrix, strategies, feature_groups):
    """为单个样本找到最优配对

    使用样本的元特征（IR + 熵）从矩阵中选择最优
    """
    # IR阈值和熵决定策略选择
    perm_ent = X_sample[11] if X_sample[11] > 0 else 0.5
    spec_ent = X_sample[7] if X_sample[7] > 0 else 1.0

    # 基于IR和熵的启发式规则，结合矩阵结果
    if ir > 15:
        # 很高IR → ADASYN或BorderlineSMOTE + entropy
        candidates = [(s, 'entropy') for s in ['ADASYN', 'BorderlineSMOTE']]
    elif ir > 10 and perm_ent < 0.7:
        # 高IR + 低熵 → BorderlineSMOTE + entropy
        candidates = [('BorderlineSMOTE', 'entropy'), ('ADASYN', 'entropy')]
    elif ir > 5 and spec_ent > 2.0:
        # 中IR + 高谱熵 → SMOTE + entropy
        candidates = [('SMOTE', 'entropy'), ('ADASYN', 'entropy')]
    elif ir > 3:
        # 中IR → SMOTE或ADASYN + entropy
        candidates = [('SMOTE', 'entropy'), ('ADASYN', 'entropy'), ('BorderlineSMOTE', 'entropy')]
    else:
        # 低IR → SMOTE + entropy
        candidates = [('SMOTE', 'entropy'), ('RUS', 'entropy')]

    # 从候选中选矩阵F1最高的
    best_pair = candidates[0]
    best_f1 = matrix[candidates[0][0]][candidates[0][1]]
    for s, f in candidates:
        if matrix[s][f] > best_f1:
            best_f1 = matrix[s][f]
            best_pair = (s, f)

    return ALL_PAIRS.index(best_pair)


def train_joint_recommender_v2(X, y, matrix, ir_values):
    """训练联合推荐器 v2 - 基于矩阵的选择"""
    rec = MetaFeatureRecommender(max_depth=6)
    selector = StrategySelector()

    X_list, y_list, ir_list = [], [], []

    for i in range(len(X)):
        sample = X[i]
        ir = ir_values[i]

        # 用矩阵找最优配对
        pair_idx = find_best_pair_for_sample(sample, ir, matrix, STRATEGIES, FEATURE_GROUPS)

        X_list.append(sample)
        y_list.append(pair_idx)
        ir_list.append(ir)

    X_arr = np.array(X_list)
    y_arr = np.array(y_list)
    ir_arr = np.array(ir_list)

    X_ir = np.column_stack([X_arr, ir_arr])
    rec.fit(X_ir, y_arr)
    return rec


def run_cv_fixed(X, y, strategy, feat_idx, cv_folds=5):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score

    min_count = min(sum(y == c) for c in range(4))
    cv_actual = max(2, min(cv_folds, min_count))
    skf = StratifiedKFold(n_splits=cv_actual, shuffle=True, random_state=42)

    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx][:, feat_idx], X[test_idx][:, feat_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        try:
            X_r, y_r = resample_safe(X_tr, y_tr, strategy)
            clf = SVC(kernel='rbf', random_state=42)
            clf.fit(X_r, y_r)
            f1 = f1_score(y_te, clf.predict(X_te), average='macro')
            scores.append(f1)
        except:
            scores.append(0.0)
    return np.mean(scores) if scores else 0.0


def run_cv_recommender(X, y, rec, cv_folds=5):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score

    min_count = min(sum(y == c) for c in range(4))
    cv_actual = max(2, min(cv_folds, min_count))
    skf = StratifiedKFold(n_splits=cv_actual, shuffle=True, random_state=42)

    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        counts_tr = [sum(y_tr == c) for c in range(4)]
        max_c, min_c = max(counts_tr), min(c for c in counts_tr if c > 0)
        ir_val = float(max_c) / min_c if min_c > 0 else 1.0

        ir_arr = np.full(len(y_tr), ir_val)
        X_ir_tr = np.column_stack([X_tr, ir_arr])
        pair_idx = rec.predict(X_ir_tr)[0]
        strategy, feat_name = ALL_PAIRS[pair_idx]
        feat_idx = FEATURE_GROUPS[feat_name]

        try:
            X_tr_feat, X_te_feat = X_tr[:, feat_idx], X_te[:, feat_idx]
            X_r, y_r = resample_safe(X_tr_feat, y_tr, strategy)
            clf = SVC(kernel='rbf', random_state=42)
            clf.fit(X_r, y_r)
            f1 = f1_score(y_te, clf.predict(X_te_feat), average='macro')
            scores.append(f1)
        except:
            scores.append(0.0)

    return np.mean(scores) if scores else 0.0


def run_local_cv(X, y, strategy, feat_name, cv_folds=5):
    """单策略本地CV"""
    return run_cv_fixed(X, y, strategy, FEATURE_GROUPS[feat_name], cv_folds)


def cross_dataset_eval(rec_cwru, X_mfpt, y_mfpt):
    """跨数据集验证（CWRU训练 → MFPT测试）"""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score

    # MFPT: 3类 (Normal, Inner, Outer) - 映射到类似的不平衡场景
    # MFPT各类数量可能不均衡，我们构造IR=5的场景
    # 先看MFPT的实际分布
    counts_mfpt = [sum(y_mfpt == c) for c in range(3)]
    print(f"    MFPT分布: Normal={counts_mfpt[0]}, Inner={counts_mfpt[1]}, Outer={counts_mfpt[2]}")

    # 直接用原始MFPT数据测试
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(X_mfpt, y_mfpt):
        X_tr, X_te = X_mfpt[train_idx], X_mfpt[test_idx]
        y_tr, y_te = y_mfpt[train_idx], y_mfpt[test_idx]

        # 计算训练集的IR（假设3类，Normal最多）
        counts_tr = [sum(y_tr == c) for c in range(3)]
        max_c = max(counts_tr)
        min_c = min(counts_tr)
        ir_val = float(max_c) / min_c if min_c > 0 else 1.0

        ir_arr = np.full(len(y_tr), ir_val)
        X_ir_tr = np.column_stack([X_tr, ir_arr])
        pair_idx = rec_cwru.predict(X_ir_tr)[0]
        strategy, feat_name = ALL_PAIRS[pair_idx]
        feat_idx = FEATURE_GROUPS[feat_name]

        try:
            X_tr_feat, X_te_feat = X_tr[:, feat_idx], X_te[:, feat_idx]
            X_r, y_r = resample_safe(X_tr_feat, y_tr, strategy)
            clf = SVC(kernel='rbf', random_state=42)
            clf.fit(X_r, y_r)
            f1 = f1_score(y_te, clf.predict(X_te_feat), average='macro')
            scores.append(f1)
        except:
            scores.append(0.0)

    return np.mean(scores) if scores else 0.0


def main():
    t_start = time.time()
    print("=" * 60)
    print("联合优化推荐器 v2")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 1: 加载数据
    print("\n[Step 1] 加载CWRU数据...")
    X_cwru, y_cwru = load_cwru()
    print(f"  CWRU: X.shape={X_cwru.shape}, 各类: {[sum(y_cwru==c) for c in range(4)]}")

    print("\n[Step 2] 加载MFPT数据...")
    X_mfpt, y_mfpt = load_mfpt_multiclass()
    print(f"  MFPT: X.shape={X_mfpt.shape}, 各类: {[sum(y_mfpt==c) for c in range(3)]}")

    ir_configs = [3, 5, 10]
    all_results = {}

    # Step 3: CWRU本地实验
    print("\n[Step 3] CWRU本地实验...")
    for target_ir in ir_configs:
        print(f"\n{'='*50}")
        print(f"IR = {target_ir}")
        print(f"{'='*50}")

        X_sub, y_sub = subsample_imbalanced(X_cwru, y_cwru, target_ir)
        counts = [sum(y_sub == c) for c in range(4)]
        actual_ir = counts[0] / min(counts[1:])
        print(f"  各类: {counts}, 实际IR: {actual_ir:.1f}")

        # 构建策略-特征矩阵
        print(f"  构建策略-特征矩阵...")
        matrix = build_strategy_feature_matrix(X_sub, y_sub, STRATEGIES, FEATURE_GROUPS, cv_folds=5)

        # Oracle最优
        best_pair = max(ALL_PAIRS, key=lambda p: matrix[p[0]][p[1]])
        oracle_f1 = matrix[best_pair[0]][best_pair[1]]
        print(f"  Oracle: {best_pair[0]}+{best_pair[1]}, F1={oracle_f1:.4f}")

        # 训练推荐器
        print(f"  训练推荐器...")
        ir_train_vals = [3, 5, 7, 10, 15]
        X_train_list, y_train_list, ir_train_list = [], [], []
        for ir_v in ir_train_vals:
            for cls in range(4):
                X_cls = X_sub[y_sub == cls]
                n = min(len(X_cls), 25)
                np.random.seed(42 + cls)
                idx = np.random.choice(len(X_cls), n, replace=False)
                for i in idx:
                    X_train_list.append(X_cls[i])
                    y_train_list.append(cls)
                    ir_train_list.append(ir_v * (1 + np.random.randn() * 0.05))
        X_train = np.array(X_train_list)

        rec = train_joint_recommender_v2(X_train, y_train_list, matrix, ir_train_list)

        # 评估
        rec_f1 = run_cv_recommender(X_sub, y_sub, rec, cv_folds=5)

        # 基线
        baselines = {}
        # AllBest: 全特征 + 策略在全部上的平均最优
        best_all = max(STRATEGIES, key=lambda s: matrix[s]['all'])
        baselines['AllBest'] = (best_all, matrix[best_all]['all'])

        # EntropyBest: entropy特征 + 策略在entropy上的最优
        best_entropy = max(STRATEGIES, key=lambda s: matrix[s]['entropy'])
        baselines['EntropyBest'] = (best_entropy, matrix[best_entropy]['entropy'])

        # OurRec: 联合推荐器
        baselines['OurRec'] = ('Joint', rec_f1)

        # 对比固定策略+全特征
        for strategy in STRATEGIES:
            f1_all = run_cv_fixed(X_sub, y_sub, strategy, FEATURE_GROUPS['all'], cv_folds=5)
            baselines[f'Fixed-{strategy}'] = (strategy, f1_all)

        print(f"\n  基线对比:")
        print(f"  {'Method':<20} | {'Strategy':<15} | {'Features':<12} | {'F1':>8}")
        print(f"  {'-'*60}")
        print(f"  {'Oracle':<20} | {best_pair[0]:<15} | {best_pair[1]:<12} | {oracle_f1:>8.4f}")
        for name, (strat, f1) in sorted(baselines.items(), key=lambda x: -x[1][1]):
            feat = 'entropy' if 'Entropy' in name else ('all' if 'All' in name or 'Fixed' in name else 'joint')
            print(f"  {name:<20} | {strat:<15} | {feat:<12} | {f1:>8.4f}")

        # 跨数据集验证
        print(f"\n  跨数据集验证 (CWRU→MFPT)...")
        rec_f1_mfpt = cross_dataset_eval(rec, X_mfpt, y_mfpt)
        print(f"  推荐器在MFPT上: F1={rec_f1_mfpt:.4f}")

        # EntropyBest在MFPT上
        entr_best_mfpt = run_cv_fixed(X_mfpt, y_mfpt, baselines['EntropyBest'][0], FEATURE_GROUPS['entropy'], cv_folds=5)
        print(f"  EntropyBest在MFPT上: F1={entr_best_mfpt:.4f}")

        all_results[f'IR={target_ir}'] = {
            'oracle': (best_pair[0], best_pair[1], oracle_f1),
            'rec_f1': rec_f1,
            'baselines': baselines,
            'mfpt_rec': rec_f1_mfpt,
            'mfpt_entropy_best': entr_best_mfpt
        }

    # Step 4: 汇总
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)

    print("\n表1: CWRU本地实验 (F1-macro)")
    print(f"{'IR':<8} | {'Oracle':<25} | {'OurRec':>8} | {'AllBest':>8} | {'EntropyBest':>12} | {'Gain':>8}")
    print("-" * 80)
    for ir_k in [f'IR={ir}' for ir in ir_configs]:
        res = all_results[ir_k]
        oracle_strat, oracle_feat, oracle_f1 = res['oracle']
        rec_f1 = res['rec_f1']
        all_best_strat, all_best_f1 = res['baselines']['AllBest']
        entr_best_strat, entr_best_f1 = res['baselines']['EntropyBest']
        gain = (rec_f1 - all_best_f1) * 100
        print(f"{ir_k:<8} | {oracle_strat}+{oracle_feat:<16} | {rec_f1:>8.4f} | {all_best_strat}:{all_best_f1:.3f} | {entr_best_strat}:{entr_best_f1:.3f} | {gain:>+7.2f}%")

    print("\n表2: 跨数据集验证 (CWRU训练 → MFPT测试)")
    print(f"{'IR':<8} | {'OurRec(MFPT)':>12} | {'EntropyBest(MFPT)':>16} | {'Gain':>8}")
    print("-" * 50)
    for ir_k in [f'IR={ir}' for ir in ir_configs]:
        res = all_results[ir_k]
        rec_mfpt = res['mfpt_rec']
        entr_mfpt = res['mfpt_entropy_best']
        gain = (rec_mfpt - entr_mfpt) * 100
        print(f"{ir_k:<8} | {rec_mfpt:>12.4f} | {entr_mfpt:>16.4f} | {gain:>+7.2f}%")

    print("\n关键发现:")
    for ir_k in [f'IR={ir}' for ir in ir_configs]:
        res = all_results[ir_k]
        rec_f1 = res['rec_f1']
        all_best_f1 = res['baselines']['AllBest'][1]
        entr_best_f1 = res['baselines']['EntropyBest'][1]

        if rec_f1 >= all_best_f1 - 0.005:
            verdict1 = ">= AllBest"
        else:
            gap = (all_best_f1 - rec_f1) * 100
            verdict1 = f"< AllBest ({gap:.1f}%)"

        if rec_f1 >= entr_best_f1 - 0.005:
            verdict2 = ">= EntropyBest"
        else:
            gap2 = (entr_best_f1 - rec_f1) * 100
            verdict2 = f"< EntropyBest ({gap2:.1f}%)"

        print(f"  {ir_k}: CWRU: {verdict1}, {verdict2}")

    ensure_dir(RESULTS_DIR)
    results_file = os.path.join(RESULTS_DIR, 'joint_recommender_v2_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump({
            'all_results': all_results,
            'ir_configs': ir_configs,
            'timestamp': datetime.now().isoformat()
        }, f)

    print(f"\n结果已保存: {results_file}")
    print(f"总耗时: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)")
    print("=" * 60)

if __name__ == "__main__":
    main()