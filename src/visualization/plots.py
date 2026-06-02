"""
实验结果可视化
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非GUI后端
import numpy as np
import pandas as pd
from pathlib import Path

# 设置中文字体和全局样式
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def plot_f1_vs_ir(results_dict, save_path="results/figures/f1_vs_ir.png"):
    """
    绘制F1-macro随IR变化曲线

    Args:
        results_dict: 包含 'IR' 和各方法F1分数的字典
        save_path: 保存路径
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    ir_values = results_dict['IR']

    methods = ['Fixed-SMOTE', 'Fixed-ADASYN', 'Fixed-RUS', 'Fixed-SMOTETomek',
               'Random-Select', 'Auto-sklearn', 'Ours']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#7f7f7f', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    plt.figure(figsize=(10, 6))

    for method, color, marker in zip(methods, colors, markers):
        if method in results_dict:
            plt.plot(ir_values, results_dict[method],
                     marker=marker, label=method, color=color, linewidth=2, markersize=8)

    plt.xscale('log')
    plt.xlabel('Imbalance Ratio (IR)', fontsize=14)
    plt.ylabel('F1-macro', fontsize=14)
    plt.title('Classification Performance vs. Imbalance Ratio', fontsize=16)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Figure saved: {save_path}")

def plot_comparison_bar(results_df, save_path="results/figures/comparison_bar.png"):
    """
    绘制对比柱状图

    Args:
        results_df: DataFrame包含Method, IR, F1-macro列
        save_path: 保存路径
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # 选取特定IR的结果
    df_selected = results_df[results_df['IR'].isin([5, 20, 50])]

    plt.figure(figsize=(12, 6))

    methods = df_selected['Method'].unique()
    ir_values = df_selected['IR'].unique()
    x = np.arange(len(methods))
    width = 0.25

    for i, ir in enumerate(ir_values):
        df_ir = df_selected[df_selected['IR'] == ir]
        values = [df_ir[df_ir['Method'] == m]['F1-macro'].values[0]
                  if len(df_ir[df_ir['Method'] == m]) > 0 else 0 for m in methods]
        plt.bar(x + i * width, values, width, label=f'IR={ir}')

    plt.xlabel('Method', fontsize=14)
    plt.ylabel('F1-macro', fontsize=14)
    plt.title('Method Comparison at Different Imbalance Ratios', fontsize=16)
    plt.xticks(x + width, methods, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Figure saved: {save_path}")

def plot_ablation_study(ablation_dict, save_path="results/figures/ablation.png"):
    """
    绘制消融分析图

    Args:
        ablation_dict: 消融结果字典
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    configs = list(ablation_dict.keys())
    f1_values = list(ablation_dict.values())

    # 按F1值排序
    sorted_pairs = sorted(zip(configs, f1_values), key=lambda x: x[1], reverse=True)
    configs, f1_values = zip(*sorted_pairs)

    colors = ['#2ecc71' if '完整' in c else '#e74c3c' if '仅' in c else '#3498db'
              for c in configs]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(configs, f1_values, color=colors)

    plt.xlabel('F1-macro', fontsize=14)
    plt.title('Feature Ablation Study (CWRU, IR=20)', fontsize=16)
    plt.xlim(0.5, 0.75)

    # 添加数值标签
    for bar, val in zip(bars, f1_values):
        plt.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Figure saved: {save_path}")

def plot_pseudo_online_results(results_df, save_path="results/figures/pseudo_online.png"):
    """
    绘制伪在线验证结果

    Args:
        results_df: DataFrame包含藏起工况、最近邻工况、推荐策略、增益列
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    workloads = results_df['Hidden Workload'].values
    gains = results_df['Gain (%)'].values

    colors = ['#27ae60' if g > 10 else '#f39c12' if g > 5 else '#e74c3c' for g in gains]

    plt.bar(workloads, gains, color=colors)
    plt.xlabel('Hidden Workload', fontsize=14)
    plt.ylabel('Gain over Fixed-SMOTE (%)', fontsize=14)
    plt.title('Pseudo-Online Validation: Generalization to Unseen Workloads', fontsize=16)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    for i, (w, g) in enumerate(zip(workloads, gains)):
        plt.text(i, g + 0.5, f'+{g:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Figure saved: {save_path}")

def plot_time_accuracy_tradeoff(save_path="results/figures/time_accuracy.png"):
    """
    绘制时间-精度权衡图

    论文中Figure 3的概念实现
    """
    # 数据点 (基于论文描述)
    data = {
        'Ours': {'time': 0.003, 'f1': 0.71, 'color': '#27ae60', 'marker': '*'},
        'Auto-sklearn': {'time': 300, 'f1': 0.66, 'color': '#e74c3c', 'marker': 'o'},
        'Fixed-SMOTE': {'time': 0.1, 'f1': 0.62, 'color': '#3498db', 'marker': 's'},
    }

    plt.figure(figsize=(10, 6))

    for name, info in data.items():
        plt.scatter(info['time'], info['f1'],
                   s=200, c=info['color'], marker=info['marker'],
                   label=name, linewidth=2, edgecolors='black')

    plt.xscale('log')
    plt.xlabel('Recommendation Time (seconds, log scale)', fontsize=14)
    plt.ylabel('F1-macro', fontsize=14)
    plt.title('Time-Accuracy Trade-off', fontsize=16)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # 添加标注
    plt.annotate('Fast & Accurate', xy=(0.01, 0.72), fontsize=10, color='green')
    plt.annotate('Slow but Accurate', xy=(100, 0.68), fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Figure saved: {save_path}")

def generate_latex_table(results_df, caption="Experimental Results", label="tab:results"):
    """
    生成LaTeX表格代码

    Returns:
        str: LaTeX表格代码
    """
    # 透视表
    pivot = results_df.pivot_table(index='Method', columns='IR', values='F1-macro')

    latex = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{'l' + 'c' * len(pivot.columns)}}}
\\toprule
\\multirow{{2}}{{*}}{{Method}} & \\multicolumn{{{len(pivot.columns)}}}{{c}}{{IR}} \\\\
\\cmidrule{{2-{len(pivot.columns)+1}}}
"""

    # 表头
    ir_cols = ' & '.join([str(ir) for ir in pivot.columns])
    latex += f"& {ir_cols} \\\\ \\midrule\n"

    # 数据行
    for method in pivot.index:
        values = []
        for ir in pivot.columns:
            val = pivot.loc[method, ir]
            values.append(f"{val:.2f}")
        latex += f"{method} & {' & '.join(values)} \\\\ \n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex