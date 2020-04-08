#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:45:41 2020

@author: nelson
"""

import DataDeal

import numpy as np
from numpy import linalg as LA
import Kernel
import cvxopt
from cvxopt import matrix
import Precision



"""
  Fuzzy SVM
  Convex function optimization problem  Package: CVXOPT

  C: penalty
  kernel_dict : 
      'type': 'LINEAR' / 'RBF' 'sigma' / 'POLY' 'd'
      
  fuzzyvalue:
      membershape value based on the class of center
      'type': 'Cen' 
      'function' : 'Lin' / 'Exp'
      
      membershape value based on the actuale hyper-plane
      'type': 'Hyp' 
      'function' : 'Lin' / 'Exp'
      
      r_max : radio between 0 and 1
      r_min : radio between 0 and 1    for balancing data
      
      usually for the majority class r = len(y_minority)/len(y_majority) 
          and for the minority class r = 1  
       

"""


class FSVM():
    
    def __init__(self, C=3, kernel_dict={'type': 'LINEAR'}, \
                 fuzzyvalue={'type':'Cen','function':'Lin'}, r_max = 1, r_min = 1):
        
        self.C = C
        self.kernel_dict = kernel_dict
        self.fuzzyvalue = fuzzyvalue
        self.r_max = r_max 
        self.r_min = r_min
        
        self.m_value = None
        self.alpha = None
        self.alpha_sv = None
        self.X_sv = None
        self.Y_sv = None
        self.b = None
        self.K = None

    def _mvalue(self, X, y):
#        print('fuzzy value:', self.fuzzyvalue )
        
        if self.fuzzyvalue['type'] == 'Cen':
            
            x_1 = X[y==1]
            x_0 = X[y==-1]
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
            if self.fuzzyvalue['function'] == 'Lin':
                for i in range(len(y)):
                    if y[i]  == 1:
                        memership.append((1 - LA.norm(X[i]-x_centre_1)/(max_distance_1+0.0001))* self.r_max)
                    if y[i]  == -1:
                        memership.append((1 - LA.norm(X[i]-x_centre_0)/(max_distance_0+0.0001))*self.r_min)
                        
            elif self.fuzzyvalue['function'] == 'Exp':
                for i in range(len(y)):
                    if y[i] == 1:
                        memership.append((2/(1+np.exp(LA.norm(X[i]-x_centre_1))))* self.r_max)
                    if y[i] == -1:
                        memership.append((2/(1+np.exp(LA.norm(X[i]-x_centre_0))))*self.r_min)
                        
        elif self.fuzzyvalue['type'] == 'Hyp':
            m = y.shape[0]
            C = 3
            gamma = 1
            # Kernel
        
            K = Kernel.RBF(m, gamma)
            K.calculate(X)
        
        
            P = cvxopt.matrix(np.outer(y, y) * K.kernelMat)
            q = cvxopt.matrix(np.ones(m) * -1)
            A = cvxopt.matrix(y, (1, m))
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
            sum_y = sum(y)
            A = np.multiply(alpha, y)
            b = (sum_y - np.sum(K.kernelMat * A.reshape(len(A),1)))/len(alpha)
                
            K.expand(X)
            A = np.multiply(alpha, y)
        
            f = b + np.sum(K.testMat * A.reshape(len(A),1),axis=0)
            
            d_hyp = abs(f*y)
        
            memership = []
            if self.fuzzyvalue['function'] == 'Lin':
                for i in range(len(y)):
                    if y[i]  == 1:
                        memership.append((1 - d_hyp[i]/(max(d_hyp)+0.0001))*self.r_max)
                    if y[i]  == -1:
                        memership.append((1 - d_hyp[i]/(max(d_hyp)+0.0001))*self.r_min)
                        
            elif self.fuzzyvalue['function'] == 'Exp':
                for i in range(len(y)):
                    if y[i] == 1:
                        memership.append((2/(1+ np.exp(d_hyp[i])))* self.r_max)
                    if y[i] == -1:
                        memership.append((2/(1+ np.exp(d_hyp[i])))*self.r_min)
                
            
                        
        self.m_value = np.array(memership)
        return self.m_value


    def fit(self, X, Y):
#        print('Kernel:', kernel_dict)
        self.Y = Y
        
        m = Y.shape[0]
      
        # Kernel
        if self.kernel_dict['type'] == 'RBF':
            K = Kernel.RBF(m, self.kernel_dict['sigma'])
            K.calculate(X)
        elif self.kernel_dict['type'] == 'LINEAR':
            K = Kernel.LINEAR(m)
            K.calculate(X)
        elif self.kernel_dict['type'] == 'POLY':
            K = Kernel.POLY(m, self.kernel_dict['d'])
            K.calculate(X)
            
        
        P = cvxopt.matrix(np.outer(Y, Y) * K.kernelMat)
        q = cvxopt.matrix(np.ones(m) * -1)
        A = cvxopt.matrix(Y, (1, m))
        A = matrix(A, (1, m), 'd')
        b = cvxopt.matrix(0.0)
        
        tmp1 = np.diag(np.ones(m) * -1)
        tmp2 = np.identity(m)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        
        tmp1 = np.zeros(m)
        tmp2 = np.ones(m) * self.m_value * self.C
        
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
           # Lagrange multipliers 
        alpha = np.ravel(solution['x'])
 
        for i in range(m):
            sv = np.logical_and(alpha < self.m_value, alpha > 1e-5)
            
        
        alpha_sv = alpha[sv]
        X_sv = X[sv]
        Y_sv = Y[sv]

        b = 0
        sum_y = sum(Y)
        A = np.multiply(alpha, Y)
        b = (sum_y - np.sum(K.kernelMat * A.reshape(len(A),1)))/len(alpha)
        
        self.alpha = alpha
        self.alpha_sv = alpha_sv
        self.X_sv = X_sv
        self.Y_sv = Y_sv
        self.b = b
        self.K = K
        
    def predict(self, X):
    
        self.K.expand(X)
        A = np.multiply(self.alpha, self.Y)
        y_pred = self.b + np.dot(self.K.testMat, A)
                
        y_pred[y_pred >= 0] = 1
        y_pred[y_pred < 0] = -1
        
        return y_pred
    
    def predict_prob(self, X):
        self.K.expand(X)
        A = np.multiply(self.alpha, self.Y)
        y_pred = self.b + np.dot(self.K.testMat, A)

        scale_min = max(y_pred[y_pred<0]) - min(y_pred[y_pred<0])
        scale_max = max(y_pred[y_pred>=0]) -min(y_pred[y_pred>=0])
        
        y_prob = np.zeros(len(y_pred))
        for i in range(len(y_pred)):
            if y_pred[i]<=0:
                y_prob[i] = 0.5*(y_pred[i]-min(y_pred[y_pred<=0]))/scale_min
                y_prob[i] = round(y_prob[i],3)
            else:
                y_prob[i] = 0.5*(y_pred[i]- min(y_pred[y_pred>0]))/scale_max +0.5 
                y_prob[i] = round(y_prob[i],3)
                
        return y_prob




if __name__ == '__main__':
    
    x_train,y_train,x_test,y_test = DataDeal.get_data()
#    Train_data,test = train_test_split(data, test_size=0.2)
    
    
 #   x_test = test[:,:-1]
 #   y_test = test[:,-1]
 #   x_train = Train_data[:,:-1]
 #   y_train = Train_data[:,-1]
    
#FSVM    
    

    kernel_dict = {'type': 'RBF','sigma':0.717}
    fuzzyvalue = {'type':'Cen','function':'Lin'}
    

    clf = FSVM(3,kernel_dict,fuzzyvalue,r_max=3/4)
    clf._mvalue(x_train, y_train) 
    clf.fit(x_train, y_train)
    Y_predict = clf.predict(x_test)
    y_prob = clf.predict_prob(x_test)
    print('y_prob',y_prob)
    Precision.precision(Y_predict,y_test)

        
