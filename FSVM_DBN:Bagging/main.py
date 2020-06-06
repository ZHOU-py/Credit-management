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
#from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from fsvm import HYP_SVM
#from fsvmClass_f import HYP_SVM

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





#下采样
    # Parce que le bon clients est la majority class, on le fait en appelant function random.randint dans le numpy
    # Échantillonnage aléatoire jusqu'à le nombre de bon clients égal mauvais clients

def lowSampling(df, percent=3/3):
    data1 = df[df[0] == 1]  # 将多数
    data0 = df[df[0] == 0]  # 将少数类别的样本放在data0
    
    # len(data1): Plage d'échantillonnage aléatoire est tous les bons clients, 
    # size=int(percent * (len(df) - len(data1))): Taille d'échantillon est le nombre de bad clients
    index = np.random.randint(
        len(data1), size=int(percent * (len(df) - len(data1))))  # 随机给定下采样取出样本的序号
    lower_data1 = data1.iloc[list(index)]  # Samplying 
    return(pd.concat([lower_data1, data0]))

#上采样
def upSampling(train):
#    X_train, y_train = SMOTE(kind='svm', ratio=1).fit_sample(train[:, 1:], train[:, 0])
    X_train, y_train = SVMSMOTE().fit_sample(train[:, 1:], train[:, 0])
    return X_train, y_train


def svmBagging(SamplingMethode):
    # 读取数据
    train = pd.read_csv("../data/data.csv", header=0)
    # 将数据都变为int型
    for col in train.columns:
        for i in range(1000):
            train[col][i] = int(train[col][i])

    # 归一化处理
    min_max_scaler = preprocessing.MinMaxScaler()
    train = min_max_scaler.fit_transform(train)

    # 分割为train和test两个数据集
    train, test = train_test_split(train, test_size=0.2)
    
    #À travers Upsampling and LowSampling, Équilibrer les données,
    # C'est à dire, le nombre de bons clients soit le même que le nombre de mauvais clients
    
    # Le différence entre eux, c'est UpSampling élève le nombre de mauvais clients just qu'aux égal bon clients
    if SamplingMethode == 'upSampling':
        # 这里做上采样
        X_train, y_train = upSampling(train)
        y_train = y_train.reshape(len(y_train), 1)
        train = np.append(y_train, X_train, axis=1)
        print("Apres UpSampling, la quantité des données équal: ", len(train))
        
    # LowSampling réduit le nombre de bon clients just qu'aux égal mauvais clients
    elif SamplingMethode == 'lowSamoling':
        train = pd.DataFrame(train)
        train = np.array(lowSampling(train))
        print("Apres LowSampling, la quantité des données équal: ", len(train))


    # 切割出EI数据集
    #
    EI = np.array(RepetitionRandomSampling(train, len(train),0.5))
    EI_train = EI[:, 1:]
    EI_test = EI[:, 0]

    clf_svm = [svm.SVC(kernel='rbf', gamma= 'scale', C=1.75) for _ in range(40)]
    #clf_svm = [fsvmClass.HYP_SVM(kernel='polynomial', C=1.5, P=1.5) for _ in range(40)]

    # clf_svm = [HYP_SVM(C=1.5) for _ in range(2)]
    bag = Bagging(40, clf_svm, 0.5)
    svms = bag.MutModel_clf(np.array(train), np.array(test))

    result = list()
    for each in svms:
        result.append(each.predict(EI_train))
    #chaque colonne est le résultat de chaque svms

    result = np.array(result)
    trX = np.transpose(result)   #transpose pour chaque ligne est le résultat
    trX = trX.astype(np.float32)
    trY = to_categorical(EI_test) # devenir une binary class
    trY = trY.astype(np.float32)

    # DLOption为一个用于存储模型超参数的类。
    # 按顺序来是: epoches, learning_rate, batchsize, momentum, penaltyL2,dropoutProb
    opts = DLOption(300, 0.01, 64, 0.01, 0., 0.2)

    # DBN类代表DBN网络类，参数分别为sizes, opts, X
    # 这里的[400,200,100]表示有三层RBM，每一层输出为400，200和100
    dbn = DBN([400, 100], opts, trX)
    # DBN训练
    dbn.train()

    # 这里初始化三层全联接层，前两层全联接层使用已训练好的RBM参数填入进去，最后一层进行fine-turn训练
    # 输入参数分别为sizes, opts, X, Y
    # nn = NN([100], opts, trX, trY)
    # 这里创建的三层，前两层的输出与上面保持一致
    nn = NN([400, 100], opts, trX, trY)
    # 这里加载已经训练好的dbn参数
    nn.load_from_dbn(dbn)
    # 训练最后一层输出层，达到分类效果
    nn.train()
    

    
    testX = test[:, 1:]
    testY = test[:, 0]
    test_result = list()
    for each in svms:
        test_result.append(each.predict(testX))
    test_result = np.array(test_result)
    teX = np.transpose(test_result)
    teX = teX.astype(np.float32)
    teY = testY.astype(np.float32)

    # score = Voting(result)

    cm = confusion_matrix(teY, nn.predict(teX))
    sns.heatmap(cm, annot=True, fmt='')
    plt.title('DBN')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    acc = len(teY[teY == nn.predict(teX)])/len(teY)
    sp = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    se = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    
    print("accuracy for DBN: ", round(acc, 3))
    print("specifity for DBN: ",round(sp, 3))
    print("Sensitivity for DBN: ",round(se, 3))

    score = Voting(test_result)

    cm = confusion_matrix(teY, score)
    sns.heatmap(cm, annot=True, fmt='')
    plt.title('Voting')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    acc = len(teY[teY == score])/len(teY)
    sp = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    se = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    
    print("accuracy for Voting: ", round(acc, 3))
    print("specifity for Voting: ",round(sp, 3))
    print("Sensitivity for Voting: ",round(se, 3))




def fsvmBagging(SamplingMethode):
    # 读取数据
    train = pd.read_csv("../data/data.csv", header=0)
    # 将数据都变为int型
    for col in train.columns:
        for i in range(1000):
            train[col][i] = int(train[col][i])

    features = train.columns[1:21]
    X = train[features]
    y = train['Creditability']
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    y_train = np.array(y_train)
    y_train = y_train.reshape(len(y_train), 1)
    train = np.append(y_train, np.array(X_train), axis=1)
    if SamplingMethode == 'upSampling':
        X_train, y_train = upSampling(train)
    elif SamplingMethode == 'lowSampling':
        train = pd.DataFrame(train)
        train = np.array(lowSampling(train))
        X_train = train[:,1:]
        y_train = train[:,0]



    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    for i in range(len(y_train)):
        if y_train[i] == 0:
            y_train[i] = -1
    y_test = np.array(y_test)
    for i in range(len(y_test)):
        if y_test[i] == 0:
            y_test[i] = -1
    y_train = np.array(y_train)
    y_train = y_train.reshape(len(y_train), 1)
    train = np.append(y_train, np.array(X_train), axis=1)

    y_test = np.array(y_test)
    y_test = y_test.reshape(len(y_test), 1)
    test = np.append(y_test, np.array(X_test), axis=1)

    # 切割出EI数据集
    # le numbre de sous ensembles de donnée est 0.5*len(train)
    EI = np.array(RepetitionRandomSampling(train, len(train), 0.5))
    EI_train = EI[:, 1:]
    EI_test = EI[:, 0]



    clf_svm = [HYP_SVM(kernel='polynomial', C=1.75, P=0.1) for _ in range(20)]

    # 20 sous ensembles de données, estimator est fsvm, rate est 0.5
    bag = Bagging(20, clf_svm, 0.5,'fsvm')
    svms = bag.MutModel_clf(np.array(train), np.array(test))

    result = list()
    for each in svms:
        result.append(each.predict(EI_train))



    result = np.array(result)
    trX = np.transpose(result)
    trX = trX.astype(np.float32)

    for i in range(len(EI_test)):
        if EI_test[i] == -1:
            EI_test[i] = 0
    trY = to_categorical(EI_test)
    trY = trY.astype(np.float32)



    # DLOption为一个用于存储模型超参数的类。
    # 按顺序来是: epoches, learning_rate, batchsize, momentum, penaltyL2,dropoutProb
    opts = DLOption(10, 1, 64, 0.01, 0., 0.2)

    # DBN类代表DBN网络类，参数分别为sizes, opts, X
    # 这里的[400,100]表示有两层RBM，每一层输出为400和100 [100,50,10]
    dbn = DBN([100,50,10], opts, trX)
    # DBN训练
    dbn.train()

    # 这里初始化三层全联接层，前两层全联接层使用已训练好的RBM参数填入进去，最后一层进行fine-turn训练
    # 输入参数分别为sizes, opts, X, Y
    # nn = NN([100], opts, trX, trY)
    # 这里创建的三层，前两层的输出与上面保持一致  
    nn = NN([100,50,10], opts, trX, trY)
    # 这里加载已经训练好的dbn参数
    nn.load_from_dbn(dbn)
    # 训练最后一层输出层，达到分类效果
    nn.train()

    testX = test[:, 1:]
    testY = test[:, 0]
    test_result = list()
    for each in svms:
        test_result.append(each.predict(testX))


    test_result = np.array(test_result)
    teX = np.transpose(test_result)
    teX = teX.astype(np.float32)
    for i in range(len(testY)):
        if testY[i] == -1:
            testY[i] = 0
    teY = testY.astype(np.float32)

    # print('Fianl acc: ',teY == nn.predict(teX))

    # score = Voting(result)

    cm = confusion_matrix(teY, nn.predict(teX))
    sns.heatmap(cm, annot=True, fmt='')
    plt.title('DBN')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    acc = len(teY[teY == nn.predict(teX)])/len(teY)
    sp = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    se = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    
    print("accuracy for DBN: ", round(acc, 3))
    print("specifity for DBN: ",round(sp, 3))
    print("Sensitivity for DBN: ",round(se, 3))

    score = Voting(test_result)

    cm = confusion_matrix(teY, score)
    sns.heatmap(cm, annot=True, fmt='')
    plt.title('Voting')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    acc = len(teY[teY == score])/len(teY)
    sp = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    se = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    
    print("accuracy for Voting: ", round(acc, 3))
    print("specifity for Voting: ",round(sp, 3))
    print("Sensitivity for Voting: ",round(se, 3))



if __name__ == '__main__':
    #可选择lowSampling，upSampling和origine
#    svmBagging('lowSampling')
    fsvmBagging('upSampling')