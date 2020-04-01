import DataDeal
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import math
import numpy as np
from numpy import linalg as LA
import Kernel
import cvxopt
from cvxopt import matrix
from sklearn.model_selection import train_test_split
import Precision


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
'''
def memership_value(data):
    X = data[:,:-1]
    Y = data[:,-1:].ravel()
    m = Y.shape[0]
    
    C = 3
    gamma = 1 / (X.shape[0] * X.var())
        # Kernel

    K = Kernel.RBF(m, gamma)
    K.calculate(X)

    
    P = cvxopt.matrix(np.outer(Y, Y) * K.kernelMat1)
    q = cvxopt.matrix(np.ones(m) * -1)
    A = cvxopt.matrix(Y, (1, m))
    A = matrix(A, (1, m), 'd')
    b = cvxopt.matrix(0.0)
    
    tmp1 = np.diag(np.ones(m) * -1)
    tmp2 = np.identity(m)
    G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
    
    tmp1 = np.zeros(m)
    tmp2 = np.ones(m) * C
    h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
    
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    alpha = np.ravel(solution['x'])
    
    b = 0
    sum_y = sum(Y)
    A = np.multiply(alpha, Y)
    b = (sum_y - np.sum(K.kernelMat1 * A.reshape(len(A),1)))/len(alpha)
    

    K = Kernel.RBF(m,gamma,m)
    K.expand(X,X)


    f = b + np.sum(K.testMat1 * A.reshape(len(A),1),axis=0)

    d_hyp = abs(f*Y)
    memership = []
#    memership= 1 - d_hyp/(max(d_hyp)+0.0001)
    memership=2/(1+ np.exp(d_hyp))
    
    return np.array(memership)
'''

def _FSVMtrain(Train_data,kernel_dict,C,membership):
    X = Train_data[:,:-1]
    Y = Train_data[:,-1:].ravel()
    m = Y.shape[0]
    
        # Kernel
    if kernel_dict['type'] == 'RBF':
        K = Kernel.RBF(m, kernel_dict['gamma'])
        K.calculate(X)
    elif kernel_dict['type'] == 'LINEAR':
        K = Kernel.LINEAR(m)
        K.calculate(X)
    elif kernel_dict['type'] == 'POLY':
        K = Kernel.POLY(m, kernel_dict['d'])
        K.calculate(X)
    
    P = cvxopt.matrix(np.outer(Y, Y) * K.kernelMat1)
    q = cvxopt.matrix(np.ones(m) * -1)
    A = cvxopt.matrix(Y, (1, m))
    A = matrix(A, (1, m), 'd')
    b = cvxopt.matrix(0.0)
    
    tmp1 = np.diag(np.ones(m) * -1)
    tmp2 = np.identity(m)
    G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
    
    tmp1 = np.zeros(m)
    tmp2 = np.ones(m) * membership * C

    h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
    # solve QP problem
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    
   # Lagrange multipliers 
    alpha = np.ravel(solution['x'])
#    print(alpha < memership_value(Train_data))
#    print(alpha > 1e-5)
#    print('alpha:',alpha.shape)
    for i in range(m):
        # 这里加入self.m的限制条件，加入fuzzy思想
        sv = np.logical_and(alpha < membership, alpha > 1e-5)
#    print(sv)
#    ind = np.arange(len(alpha))[sv]
#    print('alpha[sv]:',alpha[sv].shape)
    alpha_sv = alpha[sv]
    X_sv = X[sv]
    Y_sv = Y[sv]
#    print('Y_sv:',Y_sv)
#    print('Y:',Y)

    b = 0
    sum_y = sum(Y)
#    print(sum_y)
    A = np.multiply(alpha, Y)
    b = (sum_y - np.sum(K.kernelMat1 * A.reshape(len(A),1)))/len(alpha)
#    print('B',b)
    
    
    return (alpha,b,K,X,Y)


def _FSVMpredict(Xtest, kernel_dict, alpha, b, X, Y):
    m = X.shape[0]
    m_test = Xtest.shape[0]
    
    if kernel_dict['type'] == 'RBF':
        K = Kernel.RBF(m,kernel_dict['gamma'],m_test)
        K.expand(Xtest,X)
    elif kernel_dict['type'] == 'LINEAR':
        K = Kernel.LINEAR(m,m_test)
        K.expand(Xtest,X)
    elif kernel_dict['type'] == 'POLY':
        K = Kernel.POLY(m, kernel_dict['d'],m_test)
        K.expand(Xtest,X)
    
#    print('K.testMat.shape',K.testMat1.shape)
    A = np.multiply(alpha, Y)

    # f = b + np.dot(K.testMat,alpha)
    f = b + np.sum(K.testMat1 * A.reshape(len(A),1),axis=0)
#    print( np.sum(K.testMat1 * A.reshape(len(A),1),axis=0))
    # f = b + np.dot(K.testMat,np.multiply(alpha,Y))
    Y_predict = f
    Y_predict[Y_predict >= 0] = 1
    Y_predict[Y_predict < 0] = -1

    return Y_predict



# Test Code for _LSSVMtrain

if __name__ == '__main__':


    data = DataDeal.get_data()[:500,:]
    Train_data,test = train_test_split(data, test_size=0.2)
    test_length = test.shape[0]
    Y = data[:,-1]
    x_test = test[:,:-1]
    y_test = test[:,-1]
    y_train = Train_data[:,-1]
    
#FSVM    
    scale = 1 / (Train_data[:,:-1].shape[0] * Train_data[:,:-1].var())
    kernel_dict = {'type': 'POLY', 'd': 3}
    membership = memership_value(Train_data)
    (alpha, b,K,X,Y) = _FSVMtrain(Train_data, kernel_dict,3,membership)
    Y_predict = _FSVMpredict(x_test, kernel_dict, alpha, b, X, Y)
    print(y_test)
    Precision.precision(Y_predict,y_test)
    
#SVM
    
    # LSSVM_CV(X,Y,'RBF',[0.1,1,10],[0.1,1,10],arg2 = None)

    #print(Y_predict)
    #print(memership_value(data))
