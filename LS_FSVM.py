#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 22:46:20 2020

@author: zinan
"""

import DataDeal

import numpy as np
from numpy import linalg as LA
import Kernel
import Precision
from imblearn.over_sampling import SVMSMOTE
import math
from sklearn.model_selection import train_test_split

"""

  Least Square Fuzzy SVM
  linear equation problem  Package: NUMPY.LINALG

Parameters

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

 
class LSFSVM():
    
    def __init__(self, C=3, kernel_dict={'type': 'LINEAR'}, \
                 fuzzyvalue={'type':'Cen','function':'Lin'}, databalance='origine',r_max = 1, r_min = 1):

        self.C = C
        self.kernel_dict = kernel_dict
        self.fuzzyvalue = fuzzyvalue
        self.r_max = r_max
        self.r_min = r_min
        self.databalance = databalance
        
#        self.m_value = None
#        self.alpha = None
#        self.b = None
#        self.K = None
        
        

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
        self.kernelMat = K.kernelMat
        
    def predict(self, X):
    
        self.K.expand(X)
        A = np.multiply(self.alpha, self.Y)
        y_predict = self.b + np.dot(self.K.testMat, A)
        self.y_predict = y_predict
        y_pred = y_predict.copy()
        y_pred[y_pred >= 0] = 1
        y_pred[y_pred < 0] = -1
        self.y_pred = y_pred
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
        A = np.multiply(self.alpha, self.Y)
        y_hat = self.b + np.dot(self.kernelMat, A)

        deci = y_hat
        label = self.Y
        prior1 = len(self.Y[self.Y==1])
        prior0 = len(self.Y[self.Y==-1])
        A,B = self.Platt_Probabilistic(deci,label,prior1,prior0)
        
        y_prob = 1/(1+np.exp( A*self.y_predict+B)) 
        for i in range(len(y_prob)):
            y_prob[i] = round(y_prob[i],3)
            
        return y_prob
    
    
    def decision_function(self, X):
        return self.y_predict
        



# Test Code for _LSSVMtrain

if __name__ == '__main__':
    
    data = DataDeal.get_data('german_numerical.csv')
    Train_data,test = train_test_split(data, test_size=0.2)
    
    x_test = test[:,:-1]
    y_test = test[:,-1]
    x_train = Train_data[:,:-1]
    y_train = Train_data[:,-1]

    
    kernel_dict = {'type': 'RBF','sigma':0.717}
    fuzzyvalue = {'type':'Cen','function':'Lin'}
    
    clf = LSFSVM(10,kernel_dict, fuzzyvalue,'o',3/4)
    m = clf._mvalue(x_train, y_train)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_prob = clf.predict_prob(x_test)
    decision_function = clf.decision_function(x_test)

    print('y_prob',y_prob)   
    print(y_pred[y_prob<0.5])
    print(y_pred[y_prob>0.5])
    print(decision_function)
    
    Precision.precision(y_pred,y_test)


