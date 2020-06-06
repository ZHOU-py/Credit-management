
import numpy as np
'''
作用：
    运用Knn方式计算离训练集一个点最近的K_num个点的平均距离
输入：
    X_train: 输入的训练集，仅包含正样本或负样本
    X：X_train 中一个点
    index：X在 X_train 中的索引
    K_num: 最近的K_num个点的数量
输出：
    distance_avg: X点最近的K_num个点的平均距离
'''
def calcul_one_knn_distance(X, X_train,index, K_num):

    distance = []
    for i in range(len(X_train)):
        if i == index:
            continue
        dist = 0
        for (x,y) in zip(X,X_train[i]):
            dist += (x-y)**2

        distance.append(dist**0.5)
    sortedDistance = sorted(distance)[0:K_num]

    distance_avg = np.mean(sortedDistance)
    return distance_avg
'''
作用：
    运用Knn方式计算离训练集各个点最近的K_num个点的平均距离
输入：
    X_train: 输入的训练集，仅包含正样本或负样本
    K_num: 最近的K_num个点的数量
输出：
    X_distance: list，训练集中每个点到离他最近的 K_num 个点的平均距离
    X_max: X_distance 中最大距离
    X_min： X_distance 中最小距离
'''
def calcul_knn_distance(X_train,K_num):

    X_train_list = X_train.tolist()
    X_distance = []
    for i in range(len(X_train_list)):
        X_distance.append(calcul_one_knn_distance(X_train[i],X_train_list,i ,K_num))
    X_max = max(X_distance)
    X_min = min(X_distance)
    return X_distance, X_max, X_min

'''
作用：
    将训练集里的正负样本按照正负排序
输入： 
    X_train - 分割后的训练集，
    y_train - 分割后的训练集标签
输出： 
    X_train - 按照正负排序后的先正后负的训练集， 
    y_train - 按照正负排序后的先正后负的训练集标签
    count - 负样本数量
'''
def sort_good_bad(X_train, y_train):
    X_train_list = X_train.tolist()
    y_train_list = y_train.tolist()
    X_list_bad = []
    X_list_good = []
    y_list_bad = []
    y_list_good = []
    count = 0
    for i in range(len(X_train_list)):
        if y_train_list[i] == -1:
            X_list_bad.append(X_train_list[i])
            y_list_bad.append(y_train_list[i])
            count += 1
        else:
            X_list_good.append(X_train_list[i])
            y_list_good.append(y_train_list[i])
    X_list_bad.extend(X_list_good)
    y_list_bad.extend(y_list_good)

    X_train = np.array(X_list_bad)
    y_train = np.array(y_list_bad)
    return X_train, y_train, count

'''
FSVM 中用来计算点到该分类中心的距离
输入：
    X_train: 仅包含正样本或负样本的训练集
输出：
    distance：list，各个点到该分类中心的距离
'''
def calcul_avg_distance(X_train):

    X_sum = np.zeros(X_train.shape[1])

    for i in range(0,len(X_train)):
        X_sum += X_train[i]

    X_mean = X_sum / len(X_train)
    distance = []
    for i in range(0, len(X_train)):
        dist = 0
        for (x, y) in zip(X_mean, X_train[i]):
            dist += (x - y) ** 2
        distance.append(dist ** 0.5)
    return distance
