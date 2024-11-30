import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import f1_score, precision_score, recall_score
# 加载数据集
train = pd.read_csv('./processed_train.csv', parse_dates=['date'])

# 特征工程
train['codetime'] = (train['date'] - train['date'].min()).dt.days
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['day_of_week'] = train['date'].dt.dayofweek

# 去除销售为零的数据，避免对模型产生噪声
train = train[train['sales'] > 0]

# 将分类特征转为哑变量（One-Hot Encoding）
train_dummy = pd.get_dummies(train, columns=['family'])

# 准备训练数据
X = train_dummy.drop(columns=['sales', 'date'])
y = train_dummy['sales']

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 随机森林回归模型
# 这里暂时使用较小的参数，实际可进行调参优化
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)

# 预测与评估
y_pred = rf_model.predict(X_test)

# 模型评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# RMSLE计算
log_actual = np.log1p(y_test)
log_pred = np.log1p(y_pred)
rmsle = np.sqrt(np.mean((log_pred - log_actual) ** 2))

# 输出模型评估结果
print("随机森林模型性能：")
print(f"R^2 得分: {r2}")
print(f"均方误差 (MSE): {mse}")
print(f"均方根误差 (RMSE): {rmse}")
print(f"平均绝对误差 (MAE): {mae}")
print(f"均方根对数误差 (RMSLE): {rmsle}")

# 可选：超参数调优
# 使用GridSearchCV或RandomizedSearchCV进行参数优化
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=3,
                           n_jobs=-1,
                           verbose=2,
                           scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 使用最佳参数重新评估模型
best_y_pred = best_model.predict(X_test)
best_mse = mean_squared_error(y_test, best_y_pred)
best_rmse = np.sqrt(best_mse)
best_r2 = r2_score(y_test, best_y_pred)

print("\n调优后的随机森林模型性能：")
print(f"最佳 R^2 得分: {best_r2}")
print(f"最佳均方误差 (MSE): {best_mse}")
print(f"最佳均方根误差 (RMSE): {best_rmse}")

# 如果需要，可以保存调优后的模型
import joblib
joblib.dump(best_model, 'optimized_rf_model.pkl')
print("模型已保存为 'optimized_rf_model.pkl'")
