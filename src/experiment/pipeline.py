"""
工业信号分类实验流水线
"""
import numpy as np
import os
from pathlib import Path
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from src.features.extractor import MetaFeatureExtractor
from src.resampling import resample
from src.recommender import MetaFeatureRecommender, StrategySelector
from src.data.parse_cwru import parse_cwru_mat, create_frames


def parse_mfpt_mat(mat_path):
    """解析MFPT .mat文件

    Args:
        mat_path: .mat文件路径

    Returns:
        signal: 振动信号数组
    """
    data = loadmat(mat_path)
    # MFPT数据是纯文本转的，只有一个数据列
    for key in data.keys():
        if not key.startswith('_'):
            signal = data[key].flatten()
            return signal
    return None


class ExperimentPipeline:
    """实验流水线"""

    def __init__(self, data_dir="E:/000001research/CWRU-dataset",
                 sample_rate=12000, frame_size=1024, overlap_ratio=0.5):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.overlap_ratio = overlap_ratio
        self.extractor = MetaFeatureExtractor(sample_rate=sample_rate)
        self.selector = StrategySelector()
        self.recommender = None

    def load_dataset(self, workload='0HP', fault_types=None, normal_data=True):
        """加载CWRU数据集

        Args:
            workload: 工况 ('0HP', '1HP', '2HP', '3HP')
            fault_types: 故障类型列表, 如 ['B', 'IR', 'OR']
            normal_data: 是否包含正常数据

        Returns:
            X: 特征矩阵
            y: 标签 (0=正常, 1=故障)
        """
        X_list = []
        y_list = []

        fault_dir = os.path.join(self.data_dir, "12k_Drive_End_Bearing_Fault_Data")

        # 加载故障数据
        if fault_types:
            for fault_type in fault_types:
                for size in ['007', '014', '021', '028']:
                    size_dir = os.path.join(fault_dir, fault_type, size)
                    if not os.path.exists(size_dir):
                        continue

                    for filename in os.listdir(size_dir):
                        if not filename.endswith('.mat'):
                            continue
                        # 检查工况
                        parts = filename.replace('.mat', '').split('_')
                        workload_idx = int(parts[-1])
                        expected_idx = {'0HP': 0, '1HP': 1, '2HP': 2, '3HP': 3}[workload]
                        if workload_idx != expected_idx:
                            continue

                        mat_path = os.path.join(size_dir, filename)
                        signal = parse_cwru_mat(mat_path)
                        if signal is None:
                            continue

                        frames = create_frames(signal, self.frame_size, self.overlap_ratio)
                        for frame in frames:
                            features = self.extractor.extract(frame)
                            X_list.append(self.extractor.to_vector(features))
                            y_list.append(1)  # 故障

        # 加载正常数据
        if normal_data:
            normal_dir = os.path.join(self.data_dir, "Normal")
            if os.path.exists(normal_dir):
                for filename in os.listdir(normal_dir):
                    if not filename.endswith('.mat'):
                        continue
                    # 检查工况
                    parts = filename.replace('.mat', '').split('_')
                    workload_idx = int(parts[-1])
                    expected_idx = {'0HP': 0, '1HP': 1, '2HP': 2, '3HP': 3}[workload]
                    if workload_idx != expected_idx:
                        continue

                    mat_path = os.path.join(normal_dir, filename)
                    signal = parse_cwru_mat(mat_path)
                    if signal is None:
                        continue

                    frames = create_frames(signal, self.frame_size, self.overlap_ratio)
                    for frame in frames[:30]:  # 限制正常样本数
                        features = self.extractor.extract(frame)
                        X_list.append(self.extractor.to_vector(features))
                        y_list.append(0)  # 正常

        return np.array(X_list), np.array(y_list)

    def load_mfpt_dataset(self, mfpt_dir, fault_types=None, normal_data=True,
                          max_frames_per_file=100, frame_size=1024, overlap_ratio=0.5):
        """加载MFPT数据集

        Args:
            mfpt_dir: MFPT mat文件目录 (如 'MFPT_data/mat')
            fault_types: 故障类型 ['inner', 'outer'] 或 None
            normal_data: 是否包含正常数据
            max_frames_per_file: 每个文件最多帧数
            frame_size: 帧大小
            overlap_ratio: 重叠率

        Returns:
            X: 特征矩阵
            y: 标签 (0=正常, 1=故障)
        """
        X_list = []
        y_list = []

        # 故障数据：fault_{load}lbs_{type}.mat
        if fault_types:
            for load in ['050', '100', '150', '200', '250', '300']:
                for ftype in fault_types:
                    for mat_file in os.listdir(mfpt_dir):
                        if not mat_file.startswith('fault'):
                            continue
                        if load not in mat_file or ftype not in mat_file:
                            continue
                        if not mat_file.endswith('.mat'):
                            continue

                        mat_path = os.path.join(mfpt_dir, mat_file)
                        signal = parse_mfpt_mat(mat_path)
                        if signal is None:
                            continue

                        frames = create_frames(signal, frame_size, overlap_ratio)
                        for frame in frames[:max_frames_per_file]:
                            features = self.extractor.extract(frame)
                            X_list.append(self.extractor.to_vector(features))
                            y_list.append(1)  # 故障

        # 正常数据：normal_{load}lbs_{idx}.mat
        if normal_data:
            for mat_file in os.listdir(mfpt_dir):
                if not mat_file.startswith('normal'):
                    continue
                if not mat_file.endswith('.mat'):
                    continue

                mat_path = os.path.join(mfpt_dir, mat_file)
                signal = parse_mfpt_mat(mat_path)
                if signal is None:
                    continue

                frames = create_frames(signal, frame_size, overlap_ratio)
                # MFPT正常数据更长，限制帧数
                for frame in frames[:max_frames_per_file]:
                    features = self.extractor.extract(frame)
                    X_list.append(self.extractor.to_vector(features))
                    y_list.append(0)  # 正常

        return np.array(X_list), np.array(y_list)

    def subsample_to_ir(self, X, y, target_ir):
        """通过欠采样制造目标不平衡率"""
        unique_classes, counts = np.unique(y, return_counts=True)
        major_class = unique_classes[np.argmax(counts)]
        minor_class = unique_classes[np.argmin(counts)]

        n_minor = counts.min()
        n_major_target = int(n_minor * target_ir)

        X_major = X[y == major_class]
        X_minor = X[y == minor_class]

        np.random.seed(42)
        indices = np.random.choice(len(X_major), size=min(n_major_target, len(X_major)), replace=False)
        X_major_sub = X_major[indices]

        X_sub = np.vstack([X_major_sub, X_minor])
        y_sub = np.array([major_class] * len(X_major_sub) + [minor_class] * len(X_minor))

        return X_sub, y_sub

    def train_recommender(self, X, y, ir_values):
        """训练推荐器"""
        if isinstance(ir_values, (int, float)):
            ir_values = np.full(len(y), ir_values)

        y_strategy = self.selector.select_batch(X, ir_values)
        X_with_ir = np.column_stack([X, ir_values])

        self.recommender = MetaFeatureRecommender(max_depth=5)
        self.recommender.fit(X_with_ir, y_strategy)

        return self.recommender

    def save_recommender(self, path):
        """保存训练好的推荐器

        Args:
            path: 保存路径 (如 'models/recommender.pkl')
        """
        if self.recommender is None:
            raise ValueError("Recommender not trained yet")
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.recommender.save(path)

    def load_recommender(self, path):
        """加载推荐器

        Args:
            path: 模型文件路径
        """
        self.recommender = MetaFeatureRecommender.load(path)

    def run_single_experiment(self, X, y, strategy_or_recommender, cv_folds=5):
        """运行单次实验

        Args:
            X: 特征矩阵
            y: 标签
            strategy_or_recommender: 策略名称或推荐器
            cv_folds: 交叉验证折数

        Returns:
            f1_macro: F1-macro分数
        """
        if isinstance(strategy_or_recommender, MetaFeatureRecommender):
            ir = len(y[y==0]) / len(y[y==1]) if len(y[y==1]) > 0 else 10
            meta_features = np.column_stack([X, np.full(len(y), ir)])
            strategy_idx = strategy_or_recommender.predict(meta_features)[0]
            strategy = strategy_or_recommender.STRATEGIES[strategy_idx]
        else:
            strategy = strategy_or_recommender

        try:
            X_res, y_res = resample(X, y, strategy)
        except Exception:
            return 0.0

        clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in skf.split(X_res, y_res):
            X_train, X_test = X_res[train_idx], X_res[test_idx]
            y_train, y_test = y_res[train_idx], y_res[test_idx]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            from sklearn.metrics import f1_score
            f1 = f1_score(y_test, y_pred, average='macro')
            scores.append(f1)

        return np.mean(scores)

    def run_comparison(self, X, y, ir_values, cv_folds=5):
        """运行对比实验

        Returns:
            dict: 各方法的F1-macro分数
        """
        results = {}

        self.train_recommender(X, y, ir_values)

        strategies = ['SMOTE', 'BorderlineSMOTE', 'ADASYN', 'RUS', 'SMOTETomek']
        for strategy in strategies:
            results[f'Fixed-{strategy}'] = self.run_single_experiment(X, y, strategy, cv_folds)

        random_f1 = []
        for _ in range(5):
            strategy = np.random.choice(strategies)
            random_f1.append(self.run_single_experiment(X, y, strategy, cv_folds))
        results['Random-Select'] = np.mean(random_f1)

        results['Ours'] = self.run_single_experiment(X, y, self.recommender, cv_folds)

        return results

    def validate_on_mfpt(self, mfpt_dir, ir_values=10, cv_folds=5,
                        fault_types=None, max_frames_per_file=100):
        """在MFPT数据集上验证推荐器

        需要先调用 train_recommender() 或 load_recommender()

        Args:
            mfpt_dir: MFPT mat文件目录
            ir_values: 目标不平衡率
            cv_folds: 交叉验证折数
            fault_types: 故障类型 ['inner', 'outer']
            max_frames_per_file: 每个文件最多帧数

        Returns:
            dict: 各方法的F1-macro分数
        """
        if self.recommender is None:
            raise ValueError("Recommender not loaded or trained. "
                             "Call train_recommender() or load_recommender() first.")

        if fault_types is None:
            fault_types = ['inner', 'outer']

        X, y = self.load_mfpt_dataset(
            mfpt_dir,
            fault_types=fault_types,
            normal_data=True,
            max_frames_per_file=max_frames_per_file,
            frame_size=self.frame_size,
            overlap_ratio=self.overlap_ratio
        )

        if len(X) == 0:
            raise ValueError(f"No MFPT data loaded from {mfpt_dir}")

        # 构造不平衡
        X_sub, y_sub = self.subsample_to_ir(X, y, ir_values)

        results = {}

        # 固定策略
        strategies = ['SMOTE', 'BorderlineSMOTE', 'ADASYN', 'RUS', 'SMOTETomek']
        for strategy in strategies:
            results[f'Fixed-{strategy}'] = self.run_single_experiment(X_sub, y_sub, strategy, cv_folds)

        # 随机选择
        random_f1 = []
        for _ in range(5):
            strategy = np.random.choice(strategies)
            random_f1.append(self.run_single_experiment(X_sub, y_sub, strategy, cv_folds))
        results['Random-Select'] = np.mean(random_f1)

        # 推荐器
        results['Ours'] = self.run_single_experiment(X_sub, y_sub, self.recommender, cv_folds)

        return results
