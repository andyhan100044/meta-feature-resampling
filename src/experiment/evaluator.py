"""
实验评估器
"""
import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score

class ExperimentEvaluator:
    """实验评估器"""

    @staticmethod
    def f1_macro(y_true, y_pred):
        """计算F1-macro"""
        return f1_score(y_true, y_pred, average='macro')

    @staticmethod
    def g_mean(y_true, y_pred):
        """计算G-mean"""
        recalls = recall_score(y_true, y_pred, average=None)
        return np.prod(recalls) ** (1 / len(recalls))

    @staticmethod
    def auc_roc(y_true, y_proba):
        """计算AUC-ROC"""
        return roc_auc_score(y_true, y_proba)

    @staticmethod
    def evaluate_all(y_true, y_pred, y_proba=None):
        """综合评估"""
        results = {
            'f1_macro': ExperimentEvaluator.f1_macro(y_true, y_pred),
            'g_mean': ExperimentEvaluator.g_mean(y_true, y_pred)
        }
        if y_proba is not None:
            results['auc_roc'] = ExperimentEvaluator.auc_roc(y_true, y_proba)
        return results