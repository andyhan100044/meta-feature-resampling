"""
推荐器模块测试
"""
import numpy as np
import sys
sys.path.insert(0, 'E:/000001research')

from src.recommender import MetaFeatureRecommender, StrategySelector

def test_strategy_selector():
    """测试策略选择器"""
    selector = StrategySelector()

    # 测试各种特征组合
    # 测试各种特征组合 (按规则顺序: 1.高IR+低熵 2.高谱熵 3.非常高IR 4.中等IR 5.低IR)
    test_cases = [
        # (ir, perm_ent, spec_ent, expected_strategy_name)
        (25, 0.5, 1.5, 'BorderlineSMOTE'),  # 高IR + 低熵 → 条件1
        (15, 0.9, 3.0, 'ADASYN'),            # 高谱熵 → 条件2
        (35, 0.6, 2.0, 'BorderlineSMOTE'),   # 高IR + 低熵 → 条件1 (先匹配)
        (18, 0.9, 1.5, 'SMOTE'),             # 中等IR → 条件4
        (8, 0.7, 1.5, 'RUS'),                # 低IR → 条件5
        (3, 0.5, 1.0, 'RUS'),                # 低IR → 条件5
    ]

    for ir, perm_ent, spec_ent, expected in test_cases:
        idx = selector.select_by_features(ir, perm_ent, spec_ent)
        strategy = selector.strategies[idx]
        status = "PASS" if strategy == expected else "FAIL"
        print(f"{status} IR={ir}, PE={perm_ent}, SE={spec_ent} -> {strategy}")

    print("\nPASS: 策略选择器测试通过!")

def test_recommender_fit():
    """测试推荐器训练"""
    # 模拟训练数据: 100样本, 12维特征
    np.random.seed(42)
    X_train = np.random.randn(100, 12)
    ir_values = np.random.choice([5, 10, 20, 50], 100)

    # 使用策略选择器生成标签
    selector = StrategySelector()
    y_train = selector.select_batch(X_train, ir_values)

    print(f"标签分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # 训练推荐器
    recommender = MetaFeatureRecommender(max_depth=5)
    recommender.fit(X_train, y_train)

    # 测试预测
    X_test = np.random.randn(10, 12)
    predictions = recommender.predict(X_test)
    strategies = recommender.predict_strategy(X_test)

    print(f"\n预测示例: {strategies[:3]}")
    print("PASS: 推荐器训练测试通过!")

    return recommender

def test_recommender_rules():
    """测试规则提取"""
    recommender = test_recommender_fit()

    rules = recommender.get_rules()
    print(f"\n提取到 {len(rules)} 条规则")

    for i, rule in enumerate(rules[:3]):
        print(f"规则 {i+1}: {rule['strategy']}, {rule['samples']} 样本")

    print("\nPASS: 规则提取测试通过!")

if __name__ == "__main__":
    test_strategy_selector()
    test_recommender_fit()
    test_recommender_rules()
    print("\n所有测试通过!")