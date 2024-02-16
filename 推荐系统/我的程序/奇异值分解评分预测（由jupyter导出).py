import numpy as np
from tqdm import tqdm
import pandas as pd

file_path = '../dataset/ml-latest-small/ratings.csv'
total_lines = sum(1 for line in open(file_path, encoding='utf-8'))
with tqdm(total=total_lines, desc=f'加载 {file_path}') as pbar:
    # 三千万个数据太大了，内存放不下，改成5000
    # 但是如果取5000还用大矩阵的话就太过稀疏了，所以改成small数据集
    data = pd.read_csv(file_path, usecols=(0, 1, 2), dtype=float, encoding='utf-8', converters=None).head(20000)
    pbar.update(len(data))
ratings_data = data.to_numpy()

file_path = '../dataset/ml-latest-small/movies.csv'
total_lines = sum(1 for line in open(file_path, encoding='utf-8'))
with tqdm(total=total_lines, desc=f'加载 {file_path}') as pbar:
    data = pd.read_csv(file_path, usecols=(0, 1), dtype=float, encoding='utf-8', converters={1: lambda x: x})
    pbar.update(len(data))
movies_data = data.to_numpy()

# 假设用户和电影的ID是连续的整数，可以使用unique来获取唯一的用户和电影ID
# 这个是评分矩阵维度的参数
user_ids = np.unique(ratings_data[:, 0])
movie_ids = np.unique(movies_data[:, 0])
users_num = user_ids.shape[0]
movies_num = movie_ids.shape[0]
iter_list = []
loss_list = []

# 打乱评分矩阵
np.random.seed(42)  # 设置随机种子以保持可重复性
np.random.shuffle(ratings_data)

# 划分数据集
train_size = int(0.8 * len(ratings_data))
val_size = test_size = (len(ratings_data) - train_size) // 2

train_data = ratings_data[:train_size]
val_data = ratings_data[train_size:train_size + val_size]
test_data = ratings_data[train_size + val_size:]

# 打印数据集大小
print("训练集大小:", len(train_data))
print("验证集大小:", len(val_data))
print("测试集大小:", len(test_data))

# # 进行奇异值分解并训练模型

# 构建评分矩阵，users_num行movies_num列，先全部填上NaN，把有评分的部分再做调整

def create_R(data_input):
    R = np.full((users_num, movies_num), np.nan)

    for rating in data_input:
        user_id, movie_id, rating_value = rating[0], rating[1], rating[2]
        row = user_id - 1
        column = np.where(movies_data[:, 0] == movie_id)[0]  # 找到电影的索引
        if len(column) > 0:
            column = column[0]
            R[int(row), int(column)] = rating_value

    return R

# 进行矩阵分解
def matrix_depart_bias(matrix_input, k=5, epochs=1000, lr=0.002, reg=0.02):
    # matrix_input 是输入的用户-物品评分矩阵
    # k是隐藏特征数量
    # epochs是训练轮数
    # lr是学习率
    # reg是正则化参数
    
    # 初始化参数
    num_users, num_items = matrix_input.shape
    epoch_loss = 0
    num_rate = 0

    # 随机初始化用户矩阵 P 和物品矩阵 Q
    P = np.random.rand(num_users, k)
    Q = np.random.rand(num_items, k)
    
    # 初始化用户和物品的偏置项
    user_bias = np.zeros(num_users)
    item_bias = np.zeros(num_items)
    global_bias = np.nanmean(matrix_input)   # 需要忽略NaN值

    # 训练模型
    for epoch in range(epochs):
        for i in range(num_users):
            for j in range(num_items):
                if not np.isnan(matrix_input[i][j]):
                    loss = matrix_input[i][j] - user_bias[i] - item_bias[j] - global_bias - np.dot(P[i], Q[j])
                    epoch_loss += loss ** 2
                    num_rate += 1
                    P[i] += lr * (2 * loss * Q[j] - 2 * reg * P[i])
                    Q[j] += lr * (2 * loss * P[i] - 2 * reg * Q[j])
                    user_bias[i] += lr * (2 * loss - 2 * reg * user_bias[i])
                    item_bias[j] += lr * (2 * loss - 2 * reg * item_bias[j])
        epoch_loss /= num_rate
        iter_list.append(epoch+1)
        loss_list.append(epoch_loss)
        
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print('Epoch:', epoch+1, 'Loss:', epoch_loss)
        epoch_loss = 0
        num_rate = 0
        
    return P, Q, user_bias, item_bias, global_bias

# 根据训练好的参数完成预测矩阵
def predict(P, Q, user_bias, item_bias, global_bias):
    # 初始化参数
    num_users = P.shape[0]
    num_items = Q.shape[0]
    
    # 初始化一个与评分矩阵相同形状的矩阵，用于存储预测值
    predictions = np.zeros((num_users, num_items))

    # 遍历所有用户和物品的组合
    for i in range(num_users):
        for j in range(num_items):
            # 计算预测评分
            predictions[i, j] = user_bias[i] + item_bias[j] + global_bias + np.dot(P[i], Q[j])

    return predictions

# 用均方根误差进行评估
def evaluate(R, R_pred):
    # 初始化参数
    num_users, num_items = R.shape
    rmse = 0.0
    
    # 计算 RMSE
    num_ratings = 0
    for i in range(num_users):
        for j in range(num_items):
            if not np.isnan(R[i, j]):
                rmse += (R[i, j] - R_pred[i, j]) ** 2
                num_ratings += 1
    rmse = np.sqrt(rmse / num_ratings)

    return rmse

# 训练
matrix_train = create_R(train_data)
print('开始训练')
train_P, train_Q, train_user_bias, train_item_bias, train_global_bias = matrix_depart_bias(matrix_train)
print('训练结束')
P_pred = predict(train_P, train_Q, train_user_bias, train_item_bias, train_global_bias)

# 验证集上进行验证
R_val = create_R(val_data)
rmse_val = evaluate(R_val, P_pred)
print("验证集上的均方误差为：", rmse_val)

# 测试集上测试
R_test = create_R(test_data)
rmse_test = evaluate(R_test, P_pred)
print("测试集上的均方误差为：", rmse_test)

import matplotlib.pyplot as plt

plt.plot(iter_list, loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()



