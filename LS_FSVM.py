import DataDeal
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import math
import numpy as np
from numpy import linalg as LA
import Kernel_origine
import cvxopt
from cvxopt import matrix
from sklearn.model_selection import train_test_split
import Precision

"""
  membershape value basé sur le class center

"""

def memership_value(data):
    x_1 = data[data[:,-1]==1][:,:-1]
    x_0 = data[data[:,-1]==-1][:,:-1]
    x_centre_1 = np.mean(x_1, axis=0)
    x_centre_0 = np.mean(x_0, axis=0)
    max_distance_1 = 0
    max_distance_0 = 0
    for i in range(len(x_1)):
        distance = LA.norm(x_centre_1 - x_1[i,:])
        if max_distance_1 < distance:
            max_distance_1 = distance
    for i in range(len(x_0)):
        distance = LA.norm(x_centre_0 - x_0[i,:])
        if max_distance_0 < distance:
            max_distance_0 = distance

    memership = []
    for i in range(len(data)):
        if data[i,-1] == 1:
            memership.append(1 - LA.norm(data[i,:-1]-x_centre_1)/(max_distance_1+0.0001))
            #memership.append((1 - LA.norm(data[i,:-1]-x_centre_1)/(max_distance_1+0.0001))*3/4)
            #memership.append((2/(1+np.exp(LA.norm(data[i,:-1]-x_centre_1)))))
        if data[i, -1] == -1:
            memership.append(1 - LA.norm(data[i,:-1]-x_centre_0)/(max_distance_0+0.0001))
            #memership.append(2/(1+np.exp(LA.norm(data[i,:-1]-x_centre_0))))
    return np.array(memership)

"""
  membershape value basé sur actuale hyper-plane

"""
'''
def memership_value(data):
    X = data[:,:-1]
    Y = data[:,-1:].ravel()
    m = Y.shape[0]
    C = 3
    gamma = 1
    # Kernel

    K = Kernel_origine.RBF(m, gamma)
    K.calculate(X)


    H = np.multiply(np.dot(np.matrix(Y).T, np.matrix(Y)), K.kernelMat)
    M_BR = H + np.eye(m) / C
    # Concatenate
    L_L = np.concatenate((np.matrix(0), np.matrix(Y).T), axis=0)
    L_R = np.concatenate((np.matrix(Y), M_BR), axis=0)
    L = np.concatenate((L_L, L_R), axis=1)
    R = np.ones(m + 1)
    R[0] = 0
    # solve
    b_a = LA.solve(L, R)
    b = b_a[0]
    alpha = b_a[1:]
    
    K.expand(X)
    A = np.multiply(alpha, Y)

    f = b + np.dot(K.testMat, A)
    
    d_hyp = abs(f*Y)

    memership = []
#    memership=1 - d_hyp/(max(d_hyp)+0.0001)
    memership=2/(1+ np.exp(d_hyp))
    
    return np.array(memership)
'''
'''
_LSSVMtrain function without CV
kernel_dict = {'type':'RBF', 'gamma' : 1}
(alpha,b,K)=_LSSVMtrain(X,Y,kernel_dict,regulator)
 Y np.array
'''


def _LSSVMtrain(Train_data, kernel_dict, C):
    X = Train_data[:,:-1]
    Y = Train_data[:,-1:].ravel()
    m = Y.shape[0]
  
    # Kernel
    if kernel_dict['type'] == 'RBF':
        K = Kernel_origine.RBF(m, kernel_dict['gamma'])
        K.calculate(X)
    elif kernel_dict['type'] == 'LINEAR':
        K = Kernel_origine.LINEAR(m)
        K.calculate(X)
    elif kernel_dict['type'] == 'POLY':
        K = Kernel_origine.POLY(m, kernel_dict['d'])
        K.calculate(X)

    H = np.multiply(np.dot(np.matrix(Y).T, np.matrix(Y)), K.kernelMat)
    M_BR = H + np.eye(m) / C / memership_value(Train_data)[:,None]
    # Concatenate
    L_L = np.concatenate((np.matrix(0), np.matrix(Y).T), axis=0)
    L_R = np.concatenate((np.matrix(Y), M_BR), axis=0)
    L = np.concatenate((L_L, L_R), axis=1)
    R = np.ones(m + 1)
    R[0] = 0
    # solve
    b_a = LA.solve(L, R)
    b = b_a[0]
    alpha = b_a[1:]
#    print('LS-FSVM b:', b)
    # return
    return (alpha, b, K)


def _LSSVMpredict(Xtest, K, alpha, b, Y):
    K.expand(Xtest)
    A = np.multiply(alpha, Y)

    # f = b + np.dot(K.testMat,alpha)
    f = b + np.dot(K.testMat, A)
    # f = b + np.dot(K.testMat,np.multiply(alpha,Y))
    Y_predict = f
    Y_predict[Y_predict >= 0] = 1
    Y_predict[Y_predict < 0] = -1

    return Y_predict



# Test Code for _LSSVMtrain

if __name__ == '__main__':


    data = DataDeal.get_data()
    Train_data,test = train_test_split(data, test_size=0.2)
    test_length = test.shape[0]
    Y = data[:,-1]
    x_test = test[:,:-1]
    y_test = test[:,-1]
    y_train = Train_data[:,-1]
    
#FSVM    
    scale = 1
    kernel_dict = {'type': 'POLY','d':3}
    (alpha, b, K) = _LSSVMtrain(Train_data, kernel_dict, C=3)
    Y_predict = _LSSVMpredict(x_test, K, alpha, b, y_train)
    print(b)
    Precision.precision(Y_predict,y_test)
    
#SVM
    
    # LSSVM_CV(X,Y,'RBF',[0.1,1,10],[0.1,1,10],arg2 = None)

    #print(Y_predict)
    #print(memership_value(data))
