#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 22:46:20 2020

@author: nelson
"""

import DataDeal

import numpy as np
from numpy import linalg as LA
import Kernel
from sklearn.model_selection import train_test_split
import Precision
from sklearn import preprocessing

"""

  Least Square Fuzzy SVM
  linear equation problem  Package: NUMPY.LINALG

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

 
class LSFSVM():
    
    def __init__(self, C=3, kernel_dict={'type': 'LINEAR'}, \
                 fuzzyvalue={'type':'Cen','function':'Lin'}, r_max = 1, r_min = 1):

        self.C = C
        self.kernel_dict = kernel_dict
        self.fuzzyvalue = fuzzyvalue
        self.r_max = r_max
        self.r_min = r_min
        
#        self.m_value = None
#        self.alpha = None
#        self.b = None
#        self.K = None
        
        

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
        
        
            H = np.multiply(np.dot(np.matrix(y).T, np.matrix(y)), K.kernelMat)
            M_BR = H + np.eye(m) / C
            # Concatenate
            L_L = np.concatenate((np.matrix(0), np.matrix(y).T), axis=0)
            L_R = np.concatenate((np.matrix(y), M_BR), axis=0)
            L = np.concatenate((L_L, L_R), axis=1)
            R = np.ones(m + 1)
            R[0] = 0
            # solve
            b_a = LA.solve(L, R)
            b = b_a[0]
            alpha = b_a[1:]
            
            K.expand(X)
            A = np.multiply(alpha, y)
        
            f = b + np.dot(K.testMat, A)
            
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
#        print('Kernel:', self.kernel_dict)
        self.Y = Y
        m = len(Y)
      
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
    
        H = np.multiply(np.dot(np.matrix(Y).T, np.matrix(Y)), K.kernelMat)
        M_BR = H + np.eye(m) / (self.C * (self.m_value[:,None]))
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
        
        self.alpha = alpha
        self.b = b
        self.K = K
        
        return self
      
        
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
        



# Test Code for _LSSVMtrain

if __name__ == '__main__':
    
    x_train,y_train,x_test,y_test = DataDeal.get_data()

    
    kernel_dict = {'type': 'RBF','sigma':0.717}
    fuzzyvalue = {'type':'Cen','function':'Lin'}
    
    clf = LSFSVM(10,kernel_dict, fuzzyvalue,3/4)
    m = clf._mvalue(x_train, y_train)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_prob = clf.predict_prob(x_test)
    print('y_prob',y_prob)
    
    Precision.precision(y_pred,y_test)

