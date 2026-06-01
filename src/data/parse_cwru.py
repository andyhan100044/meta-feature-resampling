"""
CWRU数据集解析模块
处理.mat格式的轴承数据，构造不平衡分类数据集
"""
import os
import numpy as np
from scipy.io import loadmat
from pathlib import Path

# 工况定义 (马力 -> RPM)
WORKLOADS = {
    "0HP": 1797,
    "1HP": 1772,
    "2HP": 1750,
    "3HP": 1730
}

# 故障类型
FAULT_TYPES = {
    "N": "Normal",
    "IR": "InnerRace",
    "OR": "OuterRace",
    "B": "Ball"
}

# 故障尺寸 (英寸)
FAULT_SIZES = ["007", "014", "021", "028"]

def parse_cwru_mat(mat_path):
    """
    解析CWRU .mat文件

    Args:
        mat_path: .mat文件路径

    Returns:
        signal: 振动信号数组
    """
    data = loadmat(mat_path)
    # 获取第一个非meta键的值
    for key in data.keys():
        if not key.startswith('_'):
            signal = data[key].flatten()
            return signal
    return None

def create_frames(signal, frame_size=1024, overlap_ratio=0.5):
    """
    分帧处理

    Args:
        signal: 输入信号
        frame_size: 帧大小
        overlap_ratio: 重叠率

    Returns:
        frames: 分帧后的数组
    """
    step = int(frame_size * (1 - overlap_ratio))
    frames = []
    for i in range(0, len(signal) - frame_size, step):
        frames.append(signal[i:i+frame_size])
    return np.array(frames)

def load_cwru_data(data_dir="data/raw/cwru"):
    """
    加载完整的CWRU数据集

    Returns:
        data_dict: {workload: {fault_type: {size: signal}}}
    """
    data_dict = {}

    for hp, rpm in WORKLOADS.items():
        data_dict[hp] = {}

        # 正常数据
        normal_dir = os.path.join(data_dir, "Normal_Baseline_Data")
        if os.path.exists(normal_dir):
            normal_files = [f for f in os.listdir(normal_dir) if hp in f and f.endswith('.mat')]
            if normal_files:
                path = os.path.join(normal_dir, normal_files[0])
                data_dict[hp]["N"] = {"000": parse_cwru_mat(path)}

    return data_dict

def subsample_to_imbalance_ratio(X, y, target_ir):
    """
    通过欠采样多数类来制造目标不平衡率

    Args:
        X: 特征矩阵
        y: 标签
        target_ir: 目标不平衡率 (major/minor)

    Returns:
        X_sub, y_sub: 欠采样后的数据
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    major_class = unique_classes[np.argmax(counts)]
    minor_class = unique_classes[np.argmin(counts)]

    n_minor = counts.min()
    n_major_target = n_minor * target_ir

    # 分离类别
    X_major = X[y == major_class]
    X_minor = X[y == minor_class]

    # 随机欠采样多数类
    np.random.seed(42)
    indices = np.random.choice(len(X_major), size=int(n_major_target), replace=False)
    X_major_sub = X_major[indices]

    X_sub = np.vstack([X_major_sub, X_minor])
    y_sub = np.array([major_class] * len(X_major_sub) + [minor_class] * len(X_minor))

    return X_sub, y_sub

if __name__ == "__main__":
    print("CWRU数据解析模块")
    print("请先将数据下载到 data/raw/cwru/ 目录")