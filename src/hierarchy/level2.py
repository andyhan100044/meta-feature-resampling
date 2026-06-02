"""
Level 2 分类器：B vs IR vs OR

在 Level 1 判断为 Fault 的基础上，确定具体故障类型。
这是三分类问题。
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


class Level2Classifier:
    """Level 2: B vs IR vs OR 三分类器"""

    def __init__(self, resampling_strategy='SMOTE', feature_type='all'):
        """
        Args:
            resampling_strategy: 重采样策略
            feature_type: 'entropy' (熵特征) / 'freq' (频域特征) / 'all' (全特征)
        """
        self.resampling_strategy = resampling_strategy
        self.feature_type = feature_type
        self.clf = SVC(kernel='rbf', random_state=42)
        self.feature_indices = None

    def _get_feature_indices(self, n_features):
        """根据设置返回特征索引"""
        if self.feature_type == 'entropy':
            # 熵特征: indices 10, 11 (12维) 或 6, 7 (6维)
            return [10, 11] if n_features == 12 else [6, 7]
        elif self.feature_type == 'freq':
            # 频域特征: indices 6-9 (12维) 或 2-5 (6维)
            return [6, 7, 8, 9] if n_features == 12 else [2, 3, 4, 5]
        else:  # 'all'
            return list(range(n_features))

    def fit(self, X, y):
        """
        训练 Level 2 分类器

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签 (0=Normal, 1=B, 2=IR, 3=OR)
                   注意: 训练时只用 y > 0 的样本 (排除 Normal)
        """
        n_features = X.shape[1]
        self.feature_indices = self._get_feature_indices(n_features)

        # 只使用选定的特征
        X_feat = X[:, self.feature_indices]

        # 只使用 Fault 样本 (y = 1, 2, 3)
        mask = y > 0
        X_fault = X_feat[mask]
        y_fault = y[mask]

        # 重映射标签: B=0, IR=1, OR=2 (方便三分类)
        y_remapped = y_fault - 1  # 1→0, 2→1, 3→2

        # 重采样处理三分类的不平衡
        from src.resampling import resample
        X_resampled, y_resampled = resample(X_fault, y_remapped, self.resampling_strategy)

        # 训练分类器
        self.clf.fit(X_resampled, y_resampled)
        return self

    def predict(self, X):
        """
        预测: 返回故障类型标签 (1=B, 2=IR, 3=OR)
        注意: 假设输入X已经是Level 1判断为Fault的样本
        """
        X_feat = X[:, self.feature_indices]
        # 预测结果是 0, 1, 2，需要转换回 1, 2, 3
        y_pred_remapped = self.clf.predict(X_feat)
        return y_pred_remapped + 1

    def predict_proba(self, X):
        """预测概率 (近似)"""
        X_feat = X[:, self.feature_indices]
        decision = self.clf.decision_function(X_feat)
        # Softmax近似
        exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
        proba = exp_decision / exp_decision.sum(axis=1, keepdims=True)
        return proba

    def evaluate(self, X, y):
        """
        评估 Level 2 分类器性能

        Args:
            X: 特征矩阵 (只包含Fault样本)
            y: 真实标签 (1=B, 2=IR, 3=OR)

        Returns:
            dict: 包含各种评估指标
        """
        y_remapped = y - 1
        y_pred = self.predict(X) - 1  # 转换回来比较

        # 处理预测结果可能超出范围的情况
        y_pred = np.clip(y_pred, 0, 2)

        return {
            'f1_macro': f1_score(y_remapped, y_pred, average='macro'),
            'f1_weighted': f1_score(y_remapped, y_pred, average='weighted'),
            'f1_per_class': f1_score(y_remapped, y_pred, average=None).tolist(),
            'confusion_matrix': confusion_matrix(y_remapped, y_pred).tolist()
        }
