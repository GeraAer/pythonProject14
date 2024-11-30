import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import gc

# 加载数据集
train = pd.read_csv(
    './train.csv',
    usecols=['id', 'date', 'store_nbr', 'family', 'sales', 'onpromotion'],
    parse_dates=["date"]
)

test = pd.read_csv(
    './test.csv',
    usecols=['id', 'date', 'store_nbr', 'family', 'onpromotion'],
    parse_dates=["date"]
)

stores = pd.read_csv('./stores.csv')
transactions = pd.read_csv('./transactions.csv')
oil = pd.read_csv('./oil.csv', parse_dates=['date'])
holidays_events = pd.read_csv('./holidays_events.csv', parse_dates=['date'])

# 确保所有数据集的 date 列格式一致
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])
oil['date'] = pd.to_datetime(oil['date'])
transactions['date'] = pd.to_datetime(transactions['date'])
holidays_events['date'] = pd.to_datetime(holidays_events['date'])

# 1. 预处理 onpromotion 列，转换为布尔值（0/1）
train['onpromotion'] = train['onpromotion'].apply(lambda x: 1 if x != 0 else 0)
test['onpromotion'] = test['onpromotion'].apply(lambda x: 1 if x != 0 else 0)

# 2. 编码类别数据
le_family = LabelEncoder()
train['family'] = le_family.fit_transform(train['family'])
test['family'] = le_family.transform(test['family'])

# 3. 处理 stores 数据集中的类别变量
le_city = LabelEncoder()
le_state = LabelEncoder()
le_type = LabelEncoder()
stores['city'] = le_city.fit_transform(stores['city'])
stores['state'] = le_state.fit_transform(stores['state'])
stores['type'] = le_type.fit_transform(stores['type'])

# 4. 将油价数据合并到 train 和 test 数据集中
train = train.merge(oil, on='date', how='left')
test = test.merge(oil, on='date', how='left')
train['dcoilwtico'] = train['dcoilwtico'].ffill()
test['dcoilwtico'] = test['dcoilwtico'].ffill()

# 5. 将交易数据按日期和商店编号合并到 train 和 test 数据集中
train = train.merge(transactions, on=['date', 'store_nbr'], how='left')
test = test.merge(transactions, on=['date', 'store_nbr'], how='left')
train['transactions'] = train['transactions'].fillna(0)
test['transactions'] = test['transactions'].fillna(0)

# 6. 合并 stores 数据集，添加城市、州、类型和集群信息
train = train.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')

# 7. 提取日期特征
train['day_of_week'] = train['date'].dt.dayofweek
train['month'] = train['date'].dt.month
train['day_of_month'] = train['date'].dt.day
test['day_of_week'] = test['date'].dt.dayofweek
test['month'] = test['date'].dt.month
test['day_of_month'] = test['date'].dt.day

# 8. 处理假期数据，标记是否为节假日或特别活动日
holidays_events['is_holiday'] = holidays_events['transferred'].apply(lambda x: 0 if x else 1)
holidays_events = holidays_events[['date', 'is_holiday']]
train = train.merge(holidays_events, on='date', how='left')
test = test.merge(holidays_events, on='date', how='left')
train['is_holiday'] = train['is_holiday'].fillna(0)
test['is_holiday'] = test['is_holiday'].fillna(0)

# 9. 特征工程：生成滞后特征
def create_lag_features(df, date_col='date', target_col='sales', lags=[3, 7, 14, 30]):
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby(['store_nbr', 'family'])[target_col].shift(lag)
    return df

# 为训练集创建滞后特征
train = train.sort_values(by=['store_nbr', 'family', 'date'])
train = create_lag_features(train)

# 填充滞后特征的空值为 0
train = train.fillna(0)

# 10. 删除不再需要的列
train.drop(['id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

# 11. 将处理后的 train 和 test 保存为新的 CSV 文件，以便后续使用
train.to_csv('processed_train.csv', index=False)
test.to_csv('processed_test.csv', index=False)

print("数据预处理完成，保存为 'processed_train.csv' 和 'processed_test.csv'")
