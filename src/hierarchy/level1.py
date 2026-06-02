"""
Level 1 分类器：Normal vs Fault

判断轴承是否处于故障状态，这是维护决策的关键第一步。
漏检（把故障判为正常）的代价远高于误报（把正常判为故障）。
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


class Level1Classifier:
    """Level 1: Normal vs Fault 二分类器"""

    def __init__(self, resampling_strategy='SMOTE', use_entropy_features=True):
        """
        Args:
            resampling_strategy: 重采样策略 ('SMOTE', 'ADASYN', 'SMOTETomek', etc.)
            use_entropy_features: 是否只用熵特征 (更鲁棒)
        """
        self.resampling_strategy = resampling_strategy
        self.use_entropy_features = use_entropy_features
        self.clf = SVC(kernel='rbf', random_state=42)
        self.feature_indices = None  # [10, 11] for entropy only

    def _get_feature_indices(self, n_features):
        """根据设置返回特征索引"""
        if self.use_entropy_features:
            # 只使用熵特征 (sample entropy, permutation entropy)
            return [10, 11] if n_features == 12 else [6, 7]  # 12维或6维的熵特征位置
        else:
            # 使用所有特征
            return list(range(n_features))

    def fit(self, X, y):
        """
        训练 Level 1 分类器

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签 (0=Normal, 1=Fault)
        """
        n_features = X.shape[1]
        self.feature_indices = self._get_feature_indices(n_features)

        # 只使用选定的特征
        X_feat = X[:, self.feature_indices]

        # 转换为二分类标签: Normal(0) vs Fault(1,2,3)
        y_binary = (y > 0).astype(int)

        # 重采样处理不平衡
        from src.resampling import resample
        X_resampled, y_resampled = resample(X_feat, y_binary, self.resampling_strategy)

        # 训练分类器
        self.clf.fit(X_resampled, y_resampled)
        return self

    def predict(self, X):
        """预测: 返回二分类标签 (0=Normal, 1=Fault)"""
        X_feat = X[:, self.feature_indices]
        return self.clf.predict(X_feat)

    def predict_proba(self, X):
        """预测概率 (如果SVC支持的话)"""
        # SVC默认不支持predict_proba，这里用decision_function作为置信度
        X_feat = X[:, self.feature_indices]
        decision = self.decision_function(X_feat)
        # 转换为概率形式 (近似)
        proba = 1 / (1 + np.exp(-decision))
        return np.column_stack([1 - proba, proba])

    def decision_function(self, X):
        """决策函数值"""
        X_feat = X[:, self.feature_indices]
        return self.clf.decision_function(X_feat)

    def evaluate(self, X, y):
        """
        评估 Level 1 分类器性能

        Args:
            X: 特征矩阵
            y: 真实标签 (0=Normal, 1/2/3=Fault)

        Returns:
            dict: 包含各种评估指标
        """
        y_binary = (y > 0).astype(int)
        y_pred = self.predict(X)

        cm = confusion_matrix(y_binary, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        return {
            'f1': f1_score(y_binary, y_pred),
            'precision': precision_score(y_binary, y_pred, zero_division=0),
            'recall': recall_score(y_binary, y_pred, zero_division=0),  # 召回率 = 故障检出率
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,  # 特异性
            'fault_detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,  # 故障检出率 = Recall
            'false_alarm_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,  # 误报率
            'confusion_matrix': cm.tolist()
        }
