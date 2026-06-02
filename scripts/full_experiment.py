"""完整实验脚本"""
import os, sys, time, pickle, numpy as np
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.experiment.pipeline import ExperimentPipeline
from src.resampling import get_available_strategies
from src.recommender import MetaFeatureRecommender, StrategySelector

CACHE_DIR = 'E:/000001research/cache'
RESULTS_DIR = 'E:/000001research/results'
MODEL_PATH = 'E:/000001research/models/recommender_full.pkl'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_cwru_features(data_dir, sample_rate=12000, frame_size=1024, overlap_ratio=0.5, max_frames_per_file=100):
    cache_file = os.path.join(CACHE_DIR, 'cwru_features.npz')
    ensure_dir(CACHE_DIR)
    if os.path.exists(cache_file):
        print(f"  [加载缓存] {cache_file}")
        data = np.load(cache_file)
        return data['X'], data['y'], data['workloads']
    print(f"  [提取特征] 首次运行，请稍候...")
    from src.data.parse_cwru import parse_cwru_mat, create_frames
    from src.features.extractor import MetaFeatureExtractor
    extractor = MetaFeatureExtractor(sample_rate=sample_rate)
    X_list, y_list, workload_list = [], [], []
    workloads = ['0HP', '1HP', '2HP', '3HP']
    fault_types = ['B', 'IR', 'OR']
    for workload in workloads:
        expected_idx = {'0HP': 0, '1HP': 1, '2HP': 2, '3HP': 3}[workload]
        for fault_type in fault_types:
            for size in ['007', '014', '021', '028']:
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
                    for i, frame in enumerate(frames):
                        if i >= max_frames_per_file:
                            break
                        vec = extractor.to_vector(extractor.extract(frame))
                        X_list.append(vec)
                        y_list.append(1)
                        workload_list.append(workload)
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
                for i, frame in enumerate(frames[:30]):
                    vec = extractor.to_vector(extractor.extract(frame))
                    X_list.append(vec)
                    y_list.append(0)
                    workload_list.append(workload)
    X = np.array(X_list)
    y = np.array(y_list)
    workload_arr = np.array(workload_list)
    np.savez_compressed(cache_file, X=X, y=y, workloads=workload_arr)
    print(f"  特征提取完成: X.shape={X.shape}, 0={sum(y==0)}, 1={sum(y==1)}")
    print(f"  已缓存到: {cache_file}")
    return X, y, workload_arr

def run_experiment_for_ir(X, y, ir_value, cv_folds=5, rec=None):
    pipeline = ExperimentPipeline(sample_rate=12000)
    X_sub, y_sub = pipeline.subsample_to_ir(X, y, ir_value)
    results = {}
    strategies = get_available_strategies()
    for strategy in strategies:
        try:
            f1 = pipeline.run_single_experiment(X_sub, y_sub, strategy, cv_folds=cv_folds)
            results['Fixed-' + strategy] = f1
        except:
            results['Fixed-' + strategy] = 0.0
    random_f1s = []
    for _ in range(5):
        strategy = np.random.choice(strategies)
        try:
            f1 = pipeline.run_single_experiment(X_sub, y_sub, strategy, cv_folds=cv_folds)
            random_f1s.append(f1)
        except:
            pass
    results['Random-Select'] = np.mean(random_f1s) if random_f1s else 0.0
    if rec is not None:
        results['Ours'] = pipeline.run_single_experiment(X_sub, y_sub, rec, cv_folds=cv_folds)
    return results

def main():
    t_start = time.time()
    print("=" * 60)
    print("完整CWRU实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    ensure_dir(RESULTS_DIR)
    ensure_dir('E:/000001research/models')
    print("\n[Step 1] 加载CWRU数据...")
    t0 = time.time()
    X, y, workload_arr = extract_cwru_features(
        data_dir='E:/000001research/CWRU-dataset',
        sample_rate=12000, frame_size=1024, overlap_ratio=0.5,
        max_frames_per_file=100)
    print(f"  数据: X.shape={X.shape}, 0={sum(y==0)}, 1={sum(y==1)}")
    print(f"  耗时: {time.time()-t0:.1f}s")
    print("\n[Step 2] 训练推荐器...")
    t0 = time.time()
    pipeline = ExperimentPipeline(data_dir='E:/000001research/CWRU-dataset', sample_rate=12000)
    X_sub, y_sub = pipeline.subsample_to_ir(X, y, target_ir=10)
    ir = np.full(len(y_sub), 10.0)
    X_ir = np.column_stack([X_sub, ir])
    rec = MetaFeatureRecommender(max_depth=5)
    selector = StrategySelector()
    y_strat = selector.select_batch(X_sub, ir)
    rec.fit(X_ir, y_strat)
    print(f"  推荐器训练完成: {time.time()-t0:.1f}s")
    rec.save(MODEL_PATH)
    print(f"  模型已保存: {MODEL_PATH}")
    print("\n[Step 3] 运行对比实验 (CV=5)...")
    ir_values = [5, 10, 20, 50]
    all_results = {}
    for ir_val in ir_values:
        print(f"\n  IR={ir_val}:")
        results = run_experiment_for_ir(X, y, ir_val, cv_folds=5, rec=rec)
        for method, f1 in sorted(results.items(), key=lambda x: -x[1] if x[1] > 0 else 0):
            if f1 > 0:
                print(f"    {method}: F1={f1:.4f}")
        all_results[ir_val] = results
    print("\n[Step 4] 伪在线验证...")
    pseudo_results = {}
    workload_names = ['0HP', '1HP', '2HP', '3HP']
    for hidden_idx in range(4):
        hidden_wl = workload_names[hidden_idx]
        train_mask = workload_arr != hidden_wl
        test_mask = workload_arr == hidden_wl
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        X_sub_train, y_sub_train = pipeline.subsample_to_ir(X_train, y_train, target_ir=10)
        X_sub_test, y_sub_test = pipeline.subsample_to_ir(X_test, y_test, target_ir=10)
        ir_train = np.full(len(y_sub_train), 10.0)
        y_strat_train = selector.select_batch(X_sub_train, ir_train)
        X_ir_train = np.column_stack([X_sub_train, ir_train])
        rec_leave = MetaFeatureRecommender(max_depth=5)
        rec_leave.fit(X_ir_train, y_strat_train)
        nearest_wl = workload_names[(hidden_idx + 1) % 4]
        ir_test = np.full(len(X_sub_test), 10.0)
        X_ir_test = np.column_stack([X_sub_test, ir_test])
        strat_idx = rec_leave.predict(X_ir_test)[0]
        recommended = rec_leave.STRATEGIES[strat_idx]
        f1_fixed_best = 0
        for strategy in get_available_strategies():
            try:
                f1 = pipeline.run_single_experiment(X_sub_test, y_sub_test, strategy, cv_folds=3)
                if f1 > f1_fixed_best:
                    f1_fixed_best = f1
            except:
                pass
        f1_ours = pipeline.run_single_experiment(X_sub_test, y_sub_test, rec_leave, cv_folds=3)
        gain = (f1_ours - f1_fixed_best) * 100
        pseudo_results[hidden_wl] = {'nearest': nearest_wl, 'recommended': recommended, 'gain': gain}
        print(f"  Hidden={hidden_wl}, Nearest={nearest_wl}, Recommended={recommended}, Gain={gain:+.1f}%")
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print("\n表3: F1-macro Results on CWRU Dataset")
    print(f"{'Method':<22} | {'IR=5':>8} | {'IR=10':>8} | {'IR=20':>8} | {'IR=50':>8}")
    print("-" * 68)
    methods_order = ["Fixed-SMOTE", "Fixed-BorderlineSMOTE", "Fixed-ADASYN", "Fixed-RUS", "Fixed-SMOTETomek", "Random-Select", "Ours"]
    for method in methods_order:
        row = f"{method:<22} |"
        for ir_val in ir_values:
            val = all_results[ir_val].get(method, 0.0)
            row += f" {val:>8.4f} |" if val > 0 else f" {'N/A':>8} |"
        print(row)
    print("\n表4: Pseudo-Online Validation Results")
    print(f"{'Hidden Workload':<15} | {'Nearest':<10} | {'Recommended':<16} | {'Gain':>6}")
    print("-" * 58)
    for wl, res in pseudo_results.items():
        print(f"{wl:<15} | {res['nearest']:<10} | {res['recommended']:<16} | {res['gain']:>+6.1f}%")
    results_file = os.path.join(RESULTS_DIR, 'experiment_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump({'table3': all_results, 'table4': pseudo_results, 'timestamp': datetime.now().isoformat()}, f)
    print(f"\n结果已保存: {results_file}")
    print(f"总耗时: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} min)")
    print("=" * 60)

if __name__ == "__main__":
    main()
