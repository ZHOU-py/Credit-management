import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from bagging import Bagging
from sklearn import svm
from sklearn import preprocessing
import random
from keras.utils import to_categorical
from opts import DLOption
from dbn_tf import DBN
from nn_tf import NN
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

def RepetitionRandomSampling(data, number, rate):  # 有放回采样，number为抽样的个数
    sample = []
    for i in range(int(rate * number)):
        sample.append(data[random.randint(0, len(data) - 1)])
    return sample



def Voting(data):  # 投票法
    term = np.transpose(data)  # 转置
    result = list()  # 存储结果

    for target in term:
        one = list(target).count(1)
        zero = list(target).count(0)
        if one > zero:
            result.append(1)
        else:
            result.append(0)

    return result


def lowSamoling(df, percent=3/3):
    data1 = df[df[0] == 1]  # 将多数
    data0 = df[df[0] == 0]  # 将少数类别的样本放在data0

    index = np.random.randint(
        len(data1), size=int(percent * (len(df) - len(data1))))  # 随机给定下采样取出样本的序号
    lower_data1 = data1.iloc[list(index)]  # 下采样
    return(pd.concat([lower_data1, data0]))

if __name__ == '__main__':
    #读取数据
    train = pd.read_csv("../data.csv", header=0)
    #将数据都变为int型
    for col in train.columns:
        for i in range(1000):
            train[col][i] = int(train[col][i])


    #归一化处理
    min_max_scaler = preprocessing.MinMaxScaler()
    train = min_max_scaler.fit_transform(train)

    #分割为train和test两个数据集
    train, test = train_test_split(train, test_size=0.2)

    print(len(train))
    train = pd.DataFrame(train)

    train = np.array(lowSamoling(train))

    print(len(train))