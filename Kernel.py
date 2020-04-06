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
    def __init__(self, samples):
        '''
        Two Mat must be converted into np.array
        '''
        self.samples = samples
        self.kernelMat = np.zeros((samples, samples))
        self.testMat = None

    def call(self, i, j):
        return self.kernelMat[i][j]

    def _call_test(self, idx_test, idx_train):
        return self.testMat[idx_test][idx_train]


class RBF(kernel):
    def __init__(self, samples, sigma):
        kernel.__init__(self, samples)
        self.sigma = sigma;

    def calculate(self, X):
        X2 = np.sum(np.multiply(X, X), 1)  # sum colums of the matrix
        K0 = np.matrix(X2) + np.matrix(X2).T - 2 * np.dot(X, X.T)
        self.kernelMat = np.array(np.power(np.exp(-1.0 / (2*self.sigma ** 2)), K0))
        self.X = X
    '''
    Calculate the kernel for test data
    '''
    def expand(self, Xtest):
        X2_train = np.sum(np.multiply(self.X, self.X), 1)
        X2_test = np.sum(np.multiply(Xtest, Xtest), 1)
        tmp = np.matrix(X2_train) + np.matrix(X2_test).T
        if tmp.shape[0] != X2_test.shape[0]:
            tmp = tmp.T
        K0 = tmp - 2 * np.dot(Xtest, self.X.T)
        # K0 = np.matrix(X2_train).T + np.matrix(X2_test) -2 * np.dot(Xtest,self.X.T)
        self.testMat = np.array(np.power(np.exp(-1.0 / (2*self.sigma ** 2)), K0))


class LINEAR(kernel):
    def __init__(self, samples):
        kernel.__init__(self, samples)

    def calculate(self, X):
        self.kernelMat = np.dot(X, X.T)
        self.X = X

    def expand(self, Xtest):
        self.testMat = np.dot(Xtest, self.X.T)


class POLY(kernel):
    # c>=0    d in N+
    def __init__(self, samples, d=2):
        kernel.__init__(self, samples)
        self.d = d

    def calculate(self, X):
        self.kernelMat = np.power((np.dot(X, X.T) + 1), self.d)
        self.X = X

    def expand(self, Xtest):
        self.testMat = np.power((np.dot(Xtest, self.X.T) + 1), self.d)
