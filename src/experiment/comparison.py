"""
对比实验模块
"""
import numpy as np
import pandas as pd
from pathlib import Path

class ResultComparator:
    """结果对比器"""

    def __init__(self):
        self.results = []

    def add_result(self, method, ir, f1_macro, g_mean=None, auc_roc=None):
        """添加结果"""
        self.results.append({
            'Method': method,
            'IR': ir,
            'F1-macro': f1_macro,
            'G-mean': g_mean,
            'AUC-ROC': auc_roc
        })

    def to_dataframe(self):
        """转换为DataFrame"""
        return pd.DataFrame(self.results)

    def save(self, path):
        """保存结果"""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        return df

    def summary_table(self):
        """生成汇总表格"""
        df = self.to_dataframe()
        summary = df.pivot_table(index='Method', columns='IR', values='F1-macro')
        return summary