# Meta-Feature Driven Adaptive Resampling

工业振动信号分类中的不平衡故障样本增强策略自适应推荐系统

## 项目结构
- `data/` - 数据集目录
- `src/` - 源代码
  - `data/` - 数据下载与解析
  - `features/` - 特征提取
  - `resampling/` - 重采样策略
  - `recommender/` - 推荐器
  - `experiment/` - 实验流水线
  - `visualization/` - 可视化
  - `paper/` - 论文生成
- `results/` - 实验结果
- `docs/` - 文档

## 安装
pip install -r requirements.txt

## 使用
python main.py --stage all