#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:21:54 2020

@author: Zinan
"""

import DataDeal
import pandas as pd
import numpy as np
from numpy import linalg as LA
import Kernel
import cvxopt
from cvxopt import matrix
import Precision
from imblearn.over_sampling import SVMSMOTE
import math
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn.model_selection import StratifiedShuffleSplit


"""
  Bilateral-Weighted Fuzzy SVM
  Convex function optimization problem  Package: CVXOPT

Parameters

  C: penalty
  kernel_dict : 
      'type': 'LINEAR' / 'RBF' 'sigma' / 'POLY' 'd'
      
  fuzzyvalue:
      
      membershape value based on the actuale hyper-plane
      'type': 'Hyp' 
      'function' : 'Linear' / 'Bridge' / 'Logistic' / 'Probit'
      


Methods

    _mvalue(self, X, y)
        Calculate fuzzy membership value
        
    fit(self, X, Y)
        Fit the model according to the given training data.
    
    predict(self, X)
        Predict class labels for samples in X.
        
    Platt_Probabilistic(self,deci,label,prior1,prior0)
        For posterior class probability Pr(y = 1|x) = 1/(1+exp(Af+B)) calculate 
        Position parameter (B) and scale parameter (A)
    
    predict_prob(self,X)
        Posterior class probability Pr(y = 1|x)
    
    
    decision_function(self, X)
        Predict confidence scores for samples.
        The confidence score for a sample is the signed distance of that sample to the hyperplane.
       

"""


class BFSVM():
    
    def __init__(self, C=3, kernel_dict={'type': 'LINEAR'}, \
                 fuzzyvalue={'type':'Hyp','function':'Linear'},databalance='origine'):
        
        self.C = C
        self.kernel_dict = kernel_dict
        self.fuzzyvalue = fuzzyvalue
        self.databalance = databalance

        
        self.m_value = None
        self.gamme = None
        self.beta = None
        self.b = None
        self.K = None
        self.y_predict = None

    def _mvalue(self, X, y):
#        print('fuzzy value:', self.fuzzyvalue )
        train_data = np.append(X,y.reshape(len(y),1),axis=1)
        
        if self.databalance =='LowSampling':
            data_maj = train_data[y == 1]  # 将多数
            data_min =  train_data[y != 1] 
            index = np.random.randint(len(data_maj), size=len(data_min)) 
            lower_data_maj = data_maj[list(index)]
            train_data = np.append(lower_data_maj,data_min,axis=0)
            X = train_data[:,:-1]
            y = train_data[:,-1]
        
        elif self.databalance =='UpSampling':
            X, y = SVMSMOTE(random_state=42).fit_sample(train_data[:, :-1],\
                                       np.asarray(train_data[:, -1]))
            
        else:
            X = X
            y = y
        
                        
        if self.fuzzyvalue['type'] == 'Hyp':
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
#            print(f)
            memership = []
            if self.fuzzyvalue['function'] == 'Linear':
                memership = (f - min(f))/ (max(f) - min(f))
                        
            elif self.fuzzyvalue['function'] == 'Bridge':
                s_up = np.percentile(f,75)
                s_down = np.percentile(f,25)
                memership = np.zeros((len(f)))
                for i in range(len(f)):
                    if f[i]>s_up:
                        memership[i] = 1
                    elif f[i]<=s_down:
                        memership[i] = 0
                    else:
                        memership[i] = (f[i]-s_down)/(s_up-s_down)
                        
            elif self.fuzzyvalue['function'] == 'Logistic':
                a = 1
                N_pos = len(y[y==1])
                #the average of the N+ th highest primary score and the (N+ +1)th highest primary score.
                b = (f[N_pos]+f[N_pos+1])/2   
                memership = np.exp(a*f+b)/(np.exp(a*f+b)+1)
            
            elif self.fuzzyvalue['function'] == 'Probit':
                memership = norm.cdf(f)
                
                        
        self.m_value = np.array(memership)
        return self.m_value


    def fit(self, X, Y):
#        print('Kernel:', kernel_dict)
        train_data = np.append(X,Y.reshape(len(Y),1),axis=1)
        
        if self.databalance =='LowSampling':
            data_maj = train_data[Y == 1]  # 将多数
            data_min =  train_data[Y != 1] 
            index = np.random.randint(len(data_maj), size=len(data_min)) 
            lower_data_maj = data_maj[list(index)]
            train_data = np.append(lower_data_maj,data_min,axis=0)
            X = train_data[:,:-1]
            Y = train_data[:,-1]
            self.Y =  Y
        
        elif self.databalance =='UpSampling':
            X, Y = SVMSMOTE(random_state=42).fit_sample(train_data[:, :-1],\
                                       np.asarray(train_data[:, -1]))
            self.Y =  Y
            
        else:
            X = X
            Y = Y
            self.Y =  Y
        
        m = Y.shape[0]
      
        # Kernel
        if self.kernel_dict['type'] == 'RBF':
            K = Kernel.RBF(m, self.kernel_dict['sigma'])
        elif self.kernel_dict['type'] == 'LINEAR':
            K = Kernel.LINEAR(m)
        elif self.kernel_dict['type'] == 'POLY':
            K = Kernel.POLY(m, self.kernel_dict['d'])
            
        K.calculate(X)
            
        kernel = np.zeros((2*m, 2*m))
        kernel[:m,:m] = K.kernelMat 
        P = cvxopt.matrix(kernel)
        q = cvxopt.matrix(np.hstack((np.ones(m)*-1,np.ones(m)*-2)))
        
        A = cvxopt.matrix(np.hstack((np.ones(m),np.zeros(m))))
        A = matrix(A, (1, 2*m), 'd')
        b = cvxopt.matrix(0.0)
        
        tmp1 = np.hstack((np.identity(m),np.identity(m)))
        tmp2 = np.hstack((np.diag(np.ones(m) * -1),np.diag(np.ones(m) * -1)))
        tmp3 = np.hstack((np.zeros((m,m)),np.identity(m)))
        tmp4 = np.hstack((np.zeros((m,m)),np.diag(np.ones(m) * -1)))
        G = cvxopt.matrix(np.vstack((tmp1, tmp2, tmp3, tmp4)))
        
        tmp1 = np.zeros(m)
        tmp2 = np.ones(m) * self.m_value * self.C
        tmp3 = np.ones(m) * (1-self.m_value) * self.C
        h = cvxopt.matrix(np.hstack((tmp2, tmp1,tmp3,tmp1)))
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        sol = np.ravel(solution['x'])
        gamma = sol[:m]
        beta = sol[m:]
        alpha = gamma + beta
        w_phi = np.multiply(np.sum(K.kernelMat,axis=1),gamma)

        b = 0
        b = np.sum(Y-w_phi)/len(Y)       
        print('b',b)
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.b = b
        self.K = K
        self.kernelMat = K.kernelMat
        
    def predict(self, X):
    
        self.K.expand(X)
        y_predict = self.b + np.dot(self.K.testMat, self.gamma)
        y_pred = y_predict.copy()
        y_pred[y_pred >= 0] = 1
        y_pred[y_pred < 0] = -1
        
        self.y_pred = y_pred
        self.y_predict = y_predict
        
        return y_pred
    
    def Platt_Probabilistic(self,deci,label,prior1,prior0):
        maxiter=100
        minstep=1e-10
        sigma=1e-12
        
        hiTarget=(prior1+1.0)/(prior1+2.0)
        loTarget=1/(prior0+2.0)
        leng=prior1+prior0
        t = np.zeros(leng)
        for i in range(leng):
            if label[i] > 0:
                t[i]=hiTarget
            else:
                t[i]=loTarget
        
        A=0.0
        B=math.log((prior0+1.0)/(prior1+1.0))
        fval=0.0
        
        for i in range(leng):
            fApB=deci[i]*A+B
            if fApB >= 0:
                fval += t[i]*fApB+math.log(1+np.exp(-fApB))
            else:
                fval += (t[i]-1)*fApB+math.log(1+np.exp(fApB))
       
        for it in range(maxiter): 
    #Update Gradient and Hessian (use H’ = H + sigma I)
            h11=h22=sigma
            h21=g1=g2=0.0
            
            for i  in range(leng):
                fApB=deci[i]*A+B
                if fApB >= 0:
                    p=np.exp(-fApB)/(1.0+np.exp(-fApB))
                    q=1.0/(1.0+np.exp(-fApB))
                else:
                    p=1.0/(1.0+np.exp(fApB))
                    q=np.exp(fApB)/(1.0+np.exp(fApB))
                    
                d2=p*q
                h11 += deci[i]*deci[i]*d2
                h22 += d2
                h21 += deci[i]*d2
    
                d1=t[i]-p
                g1 += deci[i]*d1
                g2 += d1
        
            if (abs(g1)<1e-5 and abs(g2)<1e-5): #Stopping criteria
                break
    #Compute modified Newton directions
        
            det=h11*h22-h21*h21
            dA=-(h22*g1-h21*g2)/det
            dB=-(-h21*g1+h11*g2)/det
            gd=g1*dA+g2*dB
            stepsize=1
            
            while (stepsize >= minstep):
                #Line search
                newA=A+stepsize*dA
                newB=B+stepsize*dB
                newf=0.0
                for i in range(leng):
                    fApB=deci[i]*newA+newB
                    if (fApB >= 0):
                        newf += t[i]*fApB+math.log(1+np.exp(-fApB))
                    else:
                        newf += (t[i]-1)*fApB+math.log(1+np.exp(fApB))
    
                if (newf<fval+0.0001*stepsize*gd):
                    A=newA
                    B=newB
                    fval=newf
                    break #Sufficient decrease satisfied
                else:
                    stepsize /= 2.0
    
            if (stepsize < minstep):
                print('Line search fails')
                break
                
        if (it >= maxiter):
            print('Reaching maximum iterations')
      
        return A,B

    def predict_prob(self,X):
        y_hat = self.b + np.dot(self.kernelMat, self.gamma)

        deci = y_hat
        label = self.Y
        prior1 = len(self.Y[self.Y==1])
        prior0 = len(self.Y[self.Y==-1])
        A,B = self.Platt_Probabilistic(deci,label,prior1,prior0)
        
        y_prob = 1/(1+np.exp( A * self.y_predict+B)) 
        for i in range(len(y_prob)):
            y_prob[i] = round(y_prob[i],3)
            
        return y_prob
    
    
    def decision_function(self, X):
        return self.y_predict
        



# Test Code for _LSSVMtrain

if __name__ == '__main__':
    
    data = pd.read_csv('DF4.csv')
    X = data.drop(['default'],axis = 1)
    label = data['default']
    data = DataDeal.get_data(X,label,'normaliser',scaler='True')
    x = data[:,:-1]
    y = data[:,-1]
    
    Train_data,test = train_test_split(data, test_size=0.2,random_state = 42)
   
    x_test = test[:,:-1]
    y_test = test[:,-1]
    x_train = Train_data[:,:-1]
    y_train = Train_data[:,-1]
    
#    ss=StratifiedShuffleSplit(n_splits=3,test_size=0.2,train_size=0.8, random_state=0)
#    for train_index, test_index in ss.split(x, y):
#       x_train, x_test = x[train_index,:], x[test_index,:]#训练集对应的值
#       y_train, y_test = y[train_index], y[test_index]#类别集对应的值

    
    kernel_dict = {'type': 'RBF','sigma':0.717}
    fuzzyvalue = {'type':'Hyp','function':'Probit'}
    
    clf = BFSVM(10,kernel_dict, fuzzyvalue,'origine')
    m = clf._mvalue(x_train, y_train)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_prob = clf.predict_prob(x_test)
    decision_function = clf.decision_function(x_test)

#    print('y_prob',y_prob)   
#    print(y_pred[y_prob<0.5])
#    print(y_pred[y_prob>0.5])
    print(decision_function[:10])
    
    print('y_prob',y_prob[:10]) 
    print('y_pred',y_pred[:10]) 
    print('y_test',y_test[:10]) 
    
    Precision.precision(y_pred,y_test)