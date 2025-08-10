import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


method = 'RF'
data = 'tg'

#1.加载训练数据
train_path = '{}_train.csv'.format(data)
train_data = pd.read_csv(train_path,sep=',',header=0).values
train_feature = train_data[:,0:-1]
train_label = train_data[:,-1]



# 2.加载模型
if method == 'KNN':
    model = KNeighborsRegressor(n_neighbors=2,          # 增加邻居数
        weights='distance',      # 按距离倒数加权
        p=1,                     # 曼哈顿距离（高维效果更好）
        algorithm='kd_tree',     # 明确使用KD树加速
        leaf_size=2,            # 优化树结构大小
        metric='minkowski'       # 保持闵可夫斯基距离)
)
elif method == 'MLP':
    model = MLPRegressor( hidden_layer_sizes=(64,32),  # 两层隐藏层
        activation='relu',            # 保持ReLU激活
        solver='adam',                # Adam优化器
        alpha=0.01,                   # 更强的L2正则化
        batch_size=32,                # 明确批次大小
        learning_rate='adaptive',     # 自适应学习率
        learning_rate_init=0.01,
        max_iter=1000,                # 增加最大迭代次数
        early_stopping=True,          # 启用早停
        validation_fraction=0.2,      # 更大的验证集比例
        n_iter_no_change=8,          # 50次无改善则停止
        random_state=42,             # 固定随机种子
        tol=1e-5                     # 更严格的停止阈值
)
elif method == 'Tree':
    model = DecisionTreeRegressor(
    max_depth=50,            # 树的最大深度
    criterion='squared_error',  # 分裂标准 (MSE)
    splitter='best',        # 选择最佳分裂点
    min_samples_split=10,    # 节点分裂最小样本数
    min_samples_leaf=1,     # 叶节点最小样本数
    min_weight_fraction_leaf=0.0,
    max_features=None,      # 考虑所有特征
    random_state=None,      # 随机种子
    max_leaf_nodes=None,    # 不限制叶节点数
    min_impurity_decrease=0.0,
    ccp_alpha=0.0      # 最小剪枝复杂度
)
elif method == 'RF':
    model = RandomForestRegressor(
        n_estimators=100,  # 增加树数量提升稳定性
        max_depth=150,  # 限制树深度防止过拟合
        min_samples_split=3,  # 提高分裂门槛
        min_samples_leaf=1,  # 确保叶节点足够样本
        max_features='log2',  # 每棵树考虑 √49≈7 个特征
        criterion='squared_error',
)


# 3.模型训练
model.fit(train_feature,train_label)

# 4.加载测试集
test_path = '{}_test.csv'.format(data)
test_data = pd.read_csv(test_path,sep=',',header=0).values
test_feature = test_data[:,0:-1]
y_true = test_data[:,-1]


#5.把测试集输入模型，得到测试样本的预测标签
y_pred = model.predict(test_feature)


from scipy import stats

# 示例数据（替换为您的实际数据）

# 1. 计算均方误差 (MSE)
mse = np.mean((y_true - y_pred) ** 2)


# 3. 计算均方根误差 (RMSE)
rmse = np.sqrt(mse)

# 4. 计算相对均方根误差 (RRMSE)
rrmse = rmse / np.mean(y_true) * 100

# 5. 计算相对偏差 (RD)
rd = np.mean((y_pred - y_true) / y_true) * 100

map = np.mean(np.abs(y_true - y_pred))
# 6. 计算线性拟合的调整R²
# 执行线性回归
slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred, y_true)
r_squared = r_value ** 2

# 计算调整R²
n = len(y_true)  # 样本数量
p = 1            # 预测变量数量（一元线性回归）
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
print([float(mse), float(rmse),float(rrmse),float(map),float(adjusted_r_squared)])

