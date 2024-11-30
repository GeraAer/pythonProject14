import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 加载预处理后的数据
train = pd.read_csv('processed_train.csv')
test = pd.read_csv('processed_test.csv')

# 定义特征和目标变量
X = train.drop(columns=['sales', 'date'])  # 特征，不包含目标列 'sales' 和时间列 'date'
y = train['sales']  # 目标变量

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # 训练模型
rf_val_preds = rf_model.predict(X_val)  # 在验证集上进行预测
rf_mse = mean_squared_error(y_val, rf_val_preds)  # 计算均方误差
print(f"随机森林回归模型的验证集均方误差（MSE）：{rf_mse}")

# 在测试集上进行预测
X_test = test.drop(columns=['date'])
rf_test_preds = rf_model.predict(X_test)
test['sales_pred_rf'] = rf_test_preds  # 将预测结果保存到测试集

# 线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)  # 训练模型
lr_val_preds = lr_model.predict(X_val)  # 在验证集上进行预测
lr_mse = mean_squared_error(y_val, lr_val_preds)  # 计算均方误差
print(f"线性回归模型的验证集均方误差（MSE）：{lr_mse}")

# 在测试集上进行预测
lr_test_preds = lr_model.predict(X_test)
test['sales_pred_lr'] = lr_test_preds  # 将预测结果保存到测试集

# 保存测试集预测结果
test[['date', 'store_nbr', 'family', 'sales_pred_rf', 'sales_pred_lr']].to_csv('test_predictions.csv', index=False)
print("测试集预测结果已保存至 'test_predictions.csv'")
