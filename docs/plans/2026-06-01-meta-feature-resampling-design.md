# Meta-Feature Driven Adaptive Resampling - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现完整的论文实验框架：包括CWRU/MFPT数据集下载、信号特征提取、决策树推荐器、以及对比实验和论文撰写

**Architecture:** 流水线架构：数据下载 → 信号预处理 → 元特征提取 → 决策树推荐器训练 → 对比实验 → 结果可视化 → 英文论文生成

**Tech Stack:** Python 3.9+, numpy, scipy, sklearn, pandas, matplotlib, PyYAML

---

## Task 1: 项目初始化与依赖

**Files:**
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `README.md`

**Step 1: Write requirements.txt**

```txt
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
pandas>=2.0.0
matplotlib>=3.7.0
PyYAML>=6.0
tabulate>=0.9.0
```

**Step 2: Write config.yaml**

```yaml
dataset:
  cwru_url: "https://engineering.case.edu/-bearing/databases/12k Drive End Bearing Fault Data"
  mfpt_url: "https://mfpt.org/data-sets/"
  data_dir: "data"
  save_dir: "data/raw"

signal:
  frame_size: 1024
  overlap_ratio: 0.5
  window: "hanning"
  sample_rates: [12000, 48000]

resampling:
  strategies:
    - SMOTE
    - BorderlineSMOTE
    - ADASYN
    - RandomUnderSampler
    - SMOTETomek
  k_neighbors: 5

classifier:
  type: SVM
  kernel: rbf
  C: 1.0
  gamma: scale

recommender:
  max_depth: 5
  criterion: gini

experiment:
  imbalance_ratios: [5, 10, 20, 50]
  cv_folds: 5
  random_seed: 42
```

**Step 3: Commit**

```bash
git add requirements.txt config.yaml README.md
git commit -m "chore: project initialization"
```

---

## Task 2: CWRU数据集下载与解析

**Files:**
- Create: `src/data/download_cwru.py`
- Create: `src/data/parse_cwru.py`
- Modify: `src/data/__init__.py`

**Step 1: Write download script**

```python
"""
CWRU轴承数据集下载脚本
数据集来源: Case Western Reserve University Bearing Database
"""
import os
import requests
from pathlib import Path

def download_cwru(data_dir="data/raw"):
    # CWRU数据需要手动下载，这里提供下载链接和校验
    base_url = "https://engineering.case.edu/bearing/databases/"
    print(f"请访问 {base_url} 手动下载数据集")
    print("需要下载:")
    print("  - 12k Drive End Bearing Fault Data")
    print("  - 48k Drive End Bearing Fault Data")
    print("  - Normal Baseline Data")
```

**Step 2: Write parse script**

```python
"""
CWRU数据集解析模块
处理.mat格式的轴承数据，构造不平衡分类数据集
"""
import os
import numpy as np
from scipy.io import loadmat

def parse_cwru_mat(mat_path, sample_rate=12000):
    """解析CWRU .mat文件"""
    data = loadmat(mat_path)
    # 根据文件结构提取信号
    # 典型结构: data['DEkey'] = drive end data
    signal = data['DEkey'].flatten()
    return signal

def create_frames(signal, frame_size=1024, overlap_ratio=0.5):
    """分帧处理"""
    step = int(frame_size * (1 - overlap_ratio))
    frames = []
    for i in range(0, len(signal) - frame_size, step):
        frames.append(signal[i:i+frame_size])
    return np.array(frames)
```

---

## Task 3: 特征提取模块

**Files:**
- Create: `src/features/time_domain.py`
- Create: `src/features/freq_domain.py`
- Create: `src/features/entropy.py`
- Create: `src/features/extractor.py`
- Create: `tests/test_features.py`

**Step 1: Write time_domain.py**

```python
"""
时域特征提取
包括: 均值、方差、RMS、峰峰值、波形因子、脉冲因子、裕度因子
"""
import numpy as np

def extract_time_features(signal):
    """提取时域特征"""
    n = len(signal)
    mean = np.mean(signal)
    var = np.var(signal)
    rms = np.sqrt(np.mean(signal**2))
    peak_to_peak = np.max(signal) - np.min(signal)

    # 波形因子
    shape_factor = rms / (np.mean(np.abs(signal)) + 1e-10)

    # 脉冲因子
    impulse_factor = np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-10)

    # 裕度因子
    clearance_factor = np.max(np.abs(signal)) / (np.mean(np.sqrt(np.abs(signal)))**2 + 1e-10)

    return {
        'mean': mean,
        'variance': var,
        'rms': rms,
        'peak_to_peak': peak_to_peak,
        'shape_factor': shape_factor,
        'impulse_factor': impulse_factor,
        'clearance_factor': clearance_factor
    }
```

**Step 2: Write freq_domain.py**

```python
"""
频域特征提取
包括: 主频能量比、谱熵、重心频率、均方根频率
"""
import numpy as np
from scipy.fft import fft
from scipy.signal import welch

def extract_freq_features(signal, sample_rate=12000):
    """提取频域特征"""
    # 计算功率谱
    freqs, psd = welch(signal, fs=sample_rate, nperseg=min(256, len(signal)))

    # 主频能量比
    total_power = np.sum(psd)
    main_freq_power = np.max(psd)
    main_freq_ratio = main_freq_power / (total_power + 1e-10)

    # 谱熵
    psd_norm = psd / (total_power + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

    # 重心频率
   重心_freq = np.sum(freqs * psd) / (total_power + 1e-10)

    # 均方根频率
    rms_freq = np.sqrt(np.sum((freqs**2) * psd) / (total_power + 1e-10))

    return {
        'main_freq_ratio': main_freq_ratio,
        'spectral_entropy': spectral_entropy,
        'centroid_freq': centroid_freq,
        'rms_freq': rms_freq
    }
```

**Step 3: Write entropy.py**

```python
"""
复杂度特征提取
包括: 样本熵、排列熵
"""
import numpy as np

def sample_entropy(signal, m=2, r=0.2):
    """样本熵"""
    n = len(signal)
    r_threshold = r * np.std(signal)

    def _max_dist(xi, xj):
        return np.max(np.abs(xi - xj))

    def _phi(m):
        patterns = np.array([signal[i:i+m] for i in range(n-m)])
        res = 0
        for i in range(len(patterns)):
            count = sum(_max_dist(patterns[i], patterns[j]) <= r_threshold
                       for j in range(len(patterns)) if j != i)
            if count > 0:
                res += np.log(count / (len(patterns) - 1))
        return res / (n - m)

    return np.abs(_phi(m+1) - _phi(m))

def permutation_entropy(signal, m=3):
    """排列熵"""
    n = len(signal)
    patterns = [np.argsort(signal[i:i+m]) for i in range(n-m)]
    counts = {}
    for p in patterns:
        key = tuple(p)
        counts[key] = counts.get(key, 0) + 1

    total = sum(counts.values())
    pe = -sum((count/total) * np.log(count/total) for count in counts.values())
    return pe / np.log(np.math.factorial(m))
```

**Step 4: Write extractor.py**

```python
"""
元特征提取器 - 统一接口
"""
from src.features.time_domain import extract_time_features
from src.features.freq_domain import extract_freq_features
from src.features.entropy import sample_entropy, permutation_entropy

class MetaFeatureExtractor:
    def __init__(self, sample_rate=12000):
        self.sample_rate = sample_rate

    def extract(self, signal):
        """提取完整元特征向量"""
        time_feats = extract_time_features(signal)
        freq_feats = extract_freq_features(signal, self.sample_rate)

        samp_entropy = sample_entropy(signal)
        perm_entropy = permutation_entropy(signal)

        return {
            **time_feats,
            **freq_feats,
            'sample_entropy': samp_entropy,
            'permutation_entropy': perm_entropy
        }

    def to_vector(self, features):
        """转换为特征向量"""
        return np.array([
            features['rms'],
            features['variance'],
            features['peak_to_peak'],
            features['shape_factor'],
            features['impulse_factor'],
            features['clearance_factor'],
            features['main_freq_ratio'],
            features['spectral_entropy'],
            features['centroid_freq'],
            features['rms_freq'],
            features['sample_entropy'],
            features['permutation_entropy']
        ])
```

---

## Task 4: 决策树推荐器

**Files:**
- Create: `src/recommender/decision_tree.py`
- Create: `src/recommender/strategy_selector.py`
- Create: `tests/test_recommender.py`

**Step 1: Write decision_tree.py**

```python
"""
轻量级决策树推荐器
核心设计原则: 可解释、毫秒级推理、无GPU依赖
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class MetaFeatureRecommender:
    """
    基于信号元特征的轻量级决策树推荐器

    核心逻辑:
    1. 提取信号元特征 (时域+频域+熵+IR)
    2. 预训练决策树输出推荐策略
    3. 可解释规则遍历
    """

    def __init__(self, max_depth=5, criterion='gini'):
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            random_state=42
        )
        self.strategies = ['SMOTE', 'BorderlineSMOTE', 'ADASYN', 'RUS', 'SMOTETomek']

    def fit(self, X, y):
        """训练推荐器"""
        self.tree.fit(X, y)
        return self

    def predict(self, X):
        """预测推荐策略"""
        return self.tree.predict(X)

    def predict_proba(self, X):
        """预测概率"""
        return self.tree.predict_proba(X)

    def get_rules(self):
        """获取可解释规则"""
        tree = self.tree.tree_
        rules = []

        def traverse(node, path):
            if tree.children_left[node] == -1:  # 叶子节点
                class_idx = np.argmax(tree.value[node])
                rules.append({
                    'path': path,
                    'strategy': self.strategies[class_idx],
                    'samples': tree.value[node][0][class_idx]
                })
            else:
                feature = tree.feature[node]
                threshold = tree.threshold[node]
                path_left = path + [(feature, threshold, 'left')]
                path_right = path + [(feature, threshold, 'right')]
                traverse(tree.children_left[node], path_left)
                traverse(tree.children_right[node], path_right)

        traverse(0, [])
        return rules
```

---

## Task 5: 重采样策略实现

**Files:**
- Create: `src/resampling/strategies.py`
- Create: `tests/test_resampling.py`

**Step 1: Write strategies.py**

```python
"""
重采样策略实现
包括: SMOTE, BorderlineSMOTE, ADASYN, RandomUnderSampler, SMOTETomek
"""
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE

def get_resampler(strategy, k_neighbors=5):
    """获取重采样器实例"""
    resamplers = {
        'SMOTE': SMOTE(k_neighbors=k_neighbors, random_state=42),
        'BorderlineSMOTE': BorderlineSMOTE(k_neighbors=k_neighbors, random_state=42),
        'ADASYN': ADASYN(k_neighbors=k_neighbors, random_state=42),
        'RUS': RandomUnderSampler(random_state=42),
        'SMOTETomek': SMOTETomek(k_neighbors=k_neighbors, random_state=42),
    }
    return resamplers.get(strategy)

def resample(X, y, strategy, k_neighbors=5):
    """执行重采样"""
    resampler = get_resampler(strategy, k_neighbors)
    if resampler is None:
        raise ValueError(f"Unknown strategy: {strategy}")
    return resampler.fit_resample(X, y)
```

---

## Task 6: 实验流水线

**Files:**
- Create: `src/experiment/pipeline.py`
- Create: `src/experiment/evaluator.py`
- Create: `src/experiment/comparison.py`

**Step 1: Write pipeline.py**

```python
"""
工业信号分类实验流水线
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from src.features.extractor import MetaFeatureExtractor
from src.resampling.strategies import resample
from src.recommender.decision_tree import MetaFeatureRecommender

def run_experiment(X, y, recommender=None, strategy=None, cv_folds=5):
    """
    运行单次实验

    Args:
        X: 特征矩阵
        y: 标签
        recommender: 预训练推荐器（可选）
        strategy: 重采样策略（可选，与recommender二选一）
        cv_folds: 交叉验证折数

    Returns:
        F1-macro分数
    """
    # 如果有推荐器，获取推荐策略
    if recommender is not None:
        meta_features = extract_meta_features_for_recommender(X)
        strategy_idx = recommender.predict(meta_features)[0]
        strategy = recommender.strategies[strategy_idx]

    # 执行重采样
    if strategy:
        X_res, y_res = resample(X, y, strategy)
    else:
        X_res, y_res = X, y

    # 训练分类器
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

    # 交叉验证评估
    scores = cross_val_score(clf, X_res, y_res, cv=cv_folds, scoring='f1_macro')
    return np.mean(scores)

def extract_meta_features_for_recommender(X):
    """为推荐器提取元特征"""
    extractor = MetaFeatureExtractor()
    features_list = []
    for signal in X:
        features = extractor.extract(signal)
        features_list.append(extractor.to_vector(features))
    return np.array(features_list)
```

---

## Task 7: 结果可视化

**Files:**
- Create: `src/visualization/plots.py`
- Create: `results/figures/`

**Step 1: Write plots.py**

```python
"""
实验结果可视化
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_f1_vs_ir(results_dict, save_path="results/figures/f1_vs_ir.png"):
    """绘制F1-macro随IR变化曲线"""
    plt.figure(figsize=(10, 6))

    ir_values = results_dict['IR']
    methods = ['Fixed-SMOTE', 'Fixed-ADASYN', 'Fixed-RUS', 'Auto-sklearn', 'Ours']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for method, color in zip(methods, colors):
        plt.plot(ir_values, results_dict[method], marker='o', label=method, color=color)

    plt.xscale('log')
    plt.xlabel('Imbalance Ratio (IR)', fontsize=12)
    plt.ylabel('F1-macro', fontsize=12)
    plt.title('Classification Performance vs. Imbalance Ratio', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_comparison_table(table_data, save_path="results/figures/comparison.png"):
    """绘制对比表格图"""
    # 实现表格可视化
    pass
```

---

## Task 8: 论文生成

**Files:**
- Create: `src/paper/writer.py`
- Create: `docs/paper/manuscript.tex`

**Step 1: Write writer.py**

```python
"""
英文论文自动生成模块
基于论文大纲结构生成完整英文论文
"""
import os
from datetime import datetime

class PaperWriter:
    """论文写作器"""

    def __init__(self, output_dir="docs/paper"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_section(self, section_name, content):
        """写入章节"""
        pass

    def generate_full_paper(self, experimental_results):
        """生成完整论文"""
        paper = f"""
\\usepackage{{graphicx}}
\\usepackage{{amsmath}}
\\usepackage{{algorithm}}
\\usepackage{{algorithmic}}

\\title{{Meta-Feature Driven Adaptive Resampling for Industrial Signal Classification with Severely Imbalanced Fault Samples}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
...
\\end{{abstract}}

\\section{{Introduction}}
...

\\section{{Related Work}}
...

\\section{{Methodology}}
...

\\section{{Experiments}}
...

\\section{{Discussion}}
...

\\section{{Conclusion}}
...

\\end{{document}}
"""
        return paper
```

---

## Task 9: 主程序入口

**Files:**
- Create: `main.py`

**Step 1: Write main.py**

```python
"""
主程序入口
Meta-Feature Driven Adaptive Resampling 实验系统
"""
import argparse
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Meta-Feature Driven Adaptive Resampling")
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--stage', default='all', choices=['data', 'features', 'train', 'experiment', 'paper', 'all'])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Meta-Feature Driven Adaptive Resampling System")
    print("=" * 60)

    if args.stage in ['data', 'all']:
        print("\\n[Stage 1] 数据下载与解析...")

    if args.stage in ['features', 'all']:
        print("\\n[Stage 2] 元特征提取...")

    if args.stage in ['train', 'all']:
        print("\\n[Stage 3] 推荐器训练...")

    if args.stage in ['experiment', 'all']:
        print("\\n[Stage 4] 对比实验...")

    if args.stage in ['paper', 'all']:
        print("\\n[Stage 5] 论文生成...")

    print("\\n完成!")

if __name__ == "__main__":
    main()
```

---

## 执行选项

**Plan complete and saved to `docs/plans/2026-06-01-meta-feature-resampling-design.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**