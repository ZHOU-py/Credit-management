import random
from sklearn.metrics import precision_score, recall_score
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class Bagging(object):

    def __init__(self, n_estimators, estimator, rate=1.0, methode='svm'):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.rate = rate
        self.methode = methode

    def Voting(self, data):  # 投票法
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

    # 随机欠采样函数
    def UnderSampling(self, data):
        # np.random.seed(np.random.randint(0,1000))
        data = np.array(data)
        np.random.shuffle(data)  # 打乱data
        newdata = data[0:int(data.shape[0] * self.rate), :]  # 切片，取总数*rata的个数，删去（1-rate）%的样本
        return newdata

    def TrainPredict(self, train, test):  # 训练基础模型，并返回模型预测结果
        clf = self.estimator.fit(train[:, 0:-1], train[:, -1])
        result = clf.predict(test[:, 0:-1])
        return result

    # 简单有放回采样
    def RepetitionRandomSampling(self, data, number):  # 有放回采样，number为抽样的个数
        sample = []
        for i in range(int(self.rate * number)):
            sample.append(data[random.randint(0, len(data) - 1)])
        return sample

    def Metrics(self, predict_data, test):  # 评价函数
        score = predict_data
        recall = recall_score(test[:, 0], score, average=None)  # 召回率
        precision = precision_score(test[:, 0], score, average=None)  # 查准率
        return recall, precision

    def MutModel_clf(self, train, test, sample_type="RepetitionRandomSampling"):
        result = list()
        num_estimators = len(self.estimator)  # 使用基础模型的数量

        if sample_type == "RepetitionRandomSampling":
            print("选择的采样方法：", sample_type)
            sample_function = self.RepetitionRandomSampling
        elif sample_type == "UnderSampling":
            print("选择的采样方法：", sample_type)
            sample_function = self.UnderSampling
            print("采样率", self.rate)
        elif sample_type == "IF_SubSample":
            print("选择的采样方法：", sample_type)
            sample_function = self.IF_SubSample
            print("采样率", (1.0 - self.rate))


        #普通svm
        if self.methode == 'svm':
            for estimator in self.estimator:
                sample = np.array(sample_function(train, len(train)))  # 构建数据集
                clf = estimator.fit(sample[:,1:],sample[:,0])

                #这里result存储每一个训练好的svm
                result.append(clf)



        if self.methode == 'fsvm':
            for estimator in self.estimator:
                sample = np.array(sample_function(train, len(train)))  # 构建数据集


                estimator.m_func(sample[:, 1:], test[:,1:], sample[:, 0])
                estimator.fit(sample[:, 1:], test[:,1:], sample[:, 0])
                # #
                # print(estimator.predict(test[:,1:]))

                result.append(estimator)      #训练模型 返回每个模型的输出


        return result
