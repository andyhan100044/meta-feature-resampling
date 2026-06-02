"""
级联分类器：Level 1 → Level 2

完整的两层分类流程：
1. Level 1: Normal vs Fault
2. Level 2: (如果Fault) B vs IR vs OR

最终输出: 4类标签 (0=Normal, 1=B, 2=IR, 3=OR)
"""
import numpy as np
from .level1 import Level1Classifier
from .level2 import Level2Classifier


class CascadeClassifier:
    """级联分类器：先判断是否故障，再确定故障类型"""

    def __init__(self, resampling_strategy='SMOTE', level1_feature='entropy', level2_feature='all'):
        """
        Args:
            resampling_strategy: 重采样策略
            level1_feature: Level 1特征类型 ('entropy' / 'freq' / 'all')
            level2_feature: Level 2特征类型 ('entropy' / 'freq' / 'all')
        """
        self.level1 = Level1Classifier(
            resampling_strategy=resampling_strategy,
            use_entropy_features=(level1_feature == 'entropy')
        )
        self.level2 = Level2Classifier(
            resampling_strategy=resampling_strategy,
            feature_type=level2_feature
        )

    def fit(self, X, y):
        """
        训练级联分类器

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签 (0=Normal, 1=B, 2=IR, 3=OR)
        """
        # 训练 Level 1
        self.level1.fit(X, y)

        # 训练 Level 2 (只用 Fault 样本)
        mask = y > 0
        if mask.sum() > 0:
            self.level2.fit(X[mask], y[mask])

        return self

    def predict(self, X):
        """
        预测: 返回4类标签 (0=Normal, 1=B, 2=IR, 3=OR)

        Args:
            X: 特征矩阵 (n_samples, n_features)

        Returns:
            y_pred: 预测标签 (n_samples,)
        """
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=int)

        # Level 1 预测
        level1_pred = self.level1.predict(X)  # 0=Normal, 1=Fault

        # Normal 样本直接赋值
        normal_mask = level1_pred == 0
        y_pred[normal_mask] = 0

        # Fault 样本用 Level 2 判断
        fault_mask = level1_pred == 1
        if fault_mask.sum() > 0:
            y_pred[fault_mask] = self.level2.predict(X[fault_mask])

        return y_pred

    def predict_proba(self, X):
        """
        预测概率

        Returns:
            proba: (n_samples, 4) 每类概率
        """
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, 4))

        # Level 1 概率
        level1_proba = self.level1.predict_proba(X)  # (n_samples, 2)

        # Level 1 的 Normal 概率 → proba[:, 0]
        proba[:, 0] = level1_proba[:, 0]

        # Level 1 的 Fault 概率需要分配给 1, 2, 3
        # 假设在 Fault 情况下，B/IR/OR 概率相等 (后续可以改进)
        fault_proba = level1_proba[:, 1].reshape(-1, 1)

        # Level 2 概率
        fault_mask = (level1_pred := self.level1.predict(X)) == 1
        if fault_mask.sum() > 0:
            level2_proba = self.level2.predict_proba(X[fault_mask])
            proba[fault_mask, 1:] = level2_proba * fault_proba[fault_mask]

        return proba

    def evaluate(self, X, y):
        """
        评估级联分类器性能

        Returns:
            dict: 包含整体指标和分层指标
        """
        y_pred = self.predict(X)

        # Level 1 评估
        level1_metrics = self.level1.evaluate(X, y)

        # Level 2 评估 (只用 Fault 样本)
        fault_mask = y > 0
        if fault_mask.sum() > 0:
            level2_metrics = self.level2.evaluate(X[fault_mask], y[fault_mask])
        else:
            level2_metrics = None

        # 整体4分类评估
        from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
        from sklearn.metrics import classification_report

        # 计算整体指标 (忽略Normal，只看Fault分类)
        fault_true = y > 0
        fault_pred = y_pred > 0

        # Fault检出率 (Level 1 recall)
        fault_detection_rate = (fault_true & fault_pred).sum() / fault_true.sum()

        # 故障类型准确率 (Level 2准确率，只看Fault样本)
        if fault_mask.sum() > 0:
            fault_type_acc = (y_pred[fault_mask] == y[fault_mask]).mean()
        else:
            fault_type_acc = 0

        # 整体4分类F1-macro
        f1_macro = f1_score(y, y_pred, average='macro')

        return {
            'f1_macro': f1_macro,
            'accuracy': accuracy_score(y, y_pred),
            'fault_detection_rate': level1_metrics['recall'],  # Level 1 recall
            'false_alarm_rate': level1_metrics['false_alarm_rate'],
            'fault_type_accuracy': fault_type_acc,
            'level1_metrics': level1_metrics,
            'level2_metrics': level2_metrics,
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
