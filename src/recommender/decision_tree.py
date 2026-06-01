"""
轻量级决策树推荐器
核心设计原则: 可解释、毫秒级推理、无GPU依赖
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle

class MetaFeatureRecommender:
    """
    基于信号元特征的轻量级决策树推荐器

    核心逻辑:
    1. 提取信号元特征 (时域+频域+熵+IR)
    2. 预训练决策树输出推荐策略
    3. 可解释规则遍历
    """

    STRATEGIES = ['SMOTE', 'BorderlineSMOTE', 'ADASYN', 'RUS', 'SMOTETomek']

    def __init__(self, max_depth=5, criterion='gini', random_state=42):
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state
        self.tree = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_state
        )
        self._is_fitted = False

    def fit(self, X, y):
        """训练推荐器

        Args:
            X: 元特征矩阵 (n_samples, n_features)
            y: 策略标签 (n_samples,),值为0-4对应STRATEGIES
        """
        self.tree.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X):
        """预测推荐策略索引

        Args:
            X: 元特征矩阵

        Returns:
            np.ndarray: 策略索引 (0-4)
        """
        return self.tree.predict(X)

    def predict_strategy(self, X):
        """预测推荐策略名称

        Args:
            X: 元特征矩阵

        Returns:
            list: 策略名称列表
        """
        indices = self.predict(X)
        return [self.STRATEGIES[i] for i in indices]

    def predict_proba(self, X):
        """预测概率

        Returns:
            np.ndarray: (n_samples, n_strategies)
        """
        return self.tree.predict_proba(X)

    def get_rules(self):
        """获取可解释规则

        Returns:
            list: 规则列表,每条规则包含path和strategy
        """
        if not self._is_fitted:
            return []

        tree = self.tree.tree_
        rules = []

        def traverse(node, path):
            if tree.children_left[node] == -1:  # 叶子节点
                class_idx = np.argmax(tree.value[node])
                rules.append({
                    'path': path,
                    'strategy': self.STRATEGIES[class_idx],
                    'samples': int(tree.value[node][0][class_idx])
                })
            else:
                feature = tree.feature[node]
                threshold = tree.threshold[node]
                path_left = path + [(feature, threshold, '<=')]
                path_right = path + [(feature, threshold, '>')]
                traverse(tree.children_left[node], path_left)
                traverse(tree.children_right[node], path_right)

        traverse(0, [])
        return rules

    def print_rules(self, feature_names=None):
        """打印可解释规则"""
        rules = self.get_rules()
        for i, rule in enumerate(rules):
            print(f"\n规则 {i+1}:")
            print(f"  推荐策略: {rule['strategy']}")
            print(f"  样本数: {rule['samples']}")
            conditions = []
            for feat_idx, threshold, op in rule['path']:
                feat_name = feature_names[feat_idx] if feature_names else f"f{feat_idx}"
                conditions.append(f"{feat_name} {op} {threshold:.4f}")
            print(f"  条件: {' AND '.join(conditions)}")

    def save(self, path):
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """加载模型"""
        with open(path, 'rb') as f:
            return pickle.load(f)