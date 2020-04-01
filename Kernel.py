# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 10:42:19 2017
Maybe not used for now.
@author: Dyt
"""

# linear kernel 有没有用到gammal参数？

import math
import numpy as np
from numpy import linalg as LA

'''
def kernel_cal(x1,x2,k_type,gammaVal):
    #x1,x2 numpy.array
	
	num = x1.shape[0]
	for i in range(num):
        diff = x1[i, :] - x2
    K = exp(numpy.dot(diff,diff) / (-gammaVal)) 
    
	if k_type == 'rbf':        
        K = numpy.dot(x1,x2)
		
    return K
'''


class kernel:
    # samples is the number of samples
    def __init__(self, samples, samples_test=0):

        self.samples = samples
        self.kernelMat = np.zeros((samples, samples))
        self.kernelMat1 = np.zeros((samples, samples))
        self.testMat = None
        self.testMat1 = np.zeros((samples, samples_test))
    def call(self, i, j):
        return self.kernelMat[i][j]

    def _call_test(self, idx_train_sv,idx_test):
        return self.testMat[idx_train_sv][idx_test]

def gaussian_kernel(X1,X2,gamma):
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        return np.exp((-LA.norm(X1 - X2) ** 2)  * gamma)
    
def polynomial_kernel(x, y, p=1.5):
    return (1 + np.dot(x, y)) ** p

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

class RBF(kernel):
    def __init__(self, samples, gamma, samples_test=0):
        kernel.__init__(self, samples, samples_test)
        self.gamma = gamma;
        
    

    def calculate(self, X):
        X2 = np.sum(np.multiply(X, X), 1)  # sum colums of the matrix
        K0 = np.matrix(X2) + np.matrix(X2).T - 2 * np.dot(X, X.T)
        self.kernelMat = np.array(np.power(np.exp(-1.0 / self.gamma ** 2), K0))
        
        for i in range(len(X)):
            for j in range(len(X)):
                self.kernelMat1[i, j] = gaussian_kernel(X[i], X[j],self.gamma)
                

        self.X = X

    def expand(self, Xtest, X):
        
        for i in range(X.shape[0]):
            for j in range(Xtest.shape[0]):
                self.testMat1[i, j] = gaussian_kernel(X[i], Xtest[j],self.gamma)
        


class LINEAR(kernel):
    def __init__(self, samples, samples_test=0):
        kernel.__init__(self, samples, samples_test)

    def calculate(self, X):
        self.kernelMat = np.dot(X, X.T)
        self.X = X
        for i in range(len(X)):
            for j in range(len(X)):
                self.kernelMat1[i, j] = linear_kernel(X[i], X[j])

    def expand(self, Xtest, X):
        for i in range(X.shape[0]):
            for j in range(Xtest.shape[0]):
                self.testMat1[i, j] = linear_kernel(X[i], Xtest[j])


class POLY(kernel):
    # c>=0    d in N+
    def __init__(self, samples, d=2, samples_test=0):
        kernel.__init__(self, samples,samples_test)
        self.d = d

    def calculate(self, X):
        self.kernelMat = np.power((np.dot(X, X.T) + 1), self.d)
        self.X = X
        
        for i in range(len(X)):
            for j in range(len(X)):
                self.kernelMat1[i, j] = polynomial_kernel(X[i], X[j], self.d)

    def expand(self, Xtest, X):
        for i in range(X.shape[0]):
            for j in range(Xtest.shape[0]):
                self.testMat1[i, j] = polynomial_kernel(X[i], Xtest[j],self.d)
