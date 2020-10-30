#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:55:23 2020

@author: nelson
"""

'''
 Improved_BRF_low:
     Keep the bad clients, put the predicted probability [threshold_low,1 ] to the next classifier
     
 Improved_BRF_up:
     Keep the good clients, put the predicted probability [0, threshold_up] to the next classifier
     
 Improved_BRF
     Keep the certain bad and good clients,
     put the predicted probability [threshold_low, threshold_up] to the next classifier

'''


from imblearn.ensemble import BalancedRandomForestClassifier
import numpy as np
import pickle
from collections import Counter
import Plot_Prob_Distribution




def Improved_BRF_low(x_train,y_train,x_test,y_test,threshold1_low,threshold2_low,threshold3_low):
    
    clf1 = BalancedRandomForestClassifier(max_leaf_nodes=20,\
            n_estimators = 60,criterion = 'entropy',min_samples_leaf=20,min_samples_split=50,\
            max_depth=7, oob_score = True,random_state=10)
    
    clf2 = BalancedRandomForestClassifier(max_leaf_nodes=20,max_features = 10,\
            n_estimators = 60,criterion = 'entropy',min_samples_leaf=10,min_samples_split=30,\
            max_depth=9, oob_score = True,random_state=10)
        
    clf3 = BalancedRandomForestClassifier(max_leaf_nodes=20,max_features = 14,\
            n_estimators = 40,criterion = 'entropy',min_samples_leaf=10,min_samples_split=50,\
            max_depth=7, oob_score = True,random_state=10)
    
    ################################################## Data frist Classifier
    print('################################################## Data frist Classifier')
    print('Train Clients %s'%Counter(y_train))
    print('Test Clients %s'%Counter(y_test)) 
     
    clf1.fit(x_train,y_train)
    
    with open('BRF_clf1_low.pkl', 'wb') as f:
        pickle.dump(clf1, f, pickle.HIGHEST_PROTOCOL)
        
        
    y_pred1 = clf1.predict(x_test)
    
    y_prob1 = clf1.predict_proba(x_test)[:,1]
    y_prob1_train = clf1.predict_proba(x_train)[:,1]
           
    Plot_Prob_Distribution.Plot_probability(y_test,y_prob1,threshold1_low,threshold1_low)
    
    Prediction = np.zeros(y_test.shape)
    for i in range(len(y_test)):
        if y_prob1[i] <= threshold1_low:
            Prediction[i] = -1
        else:            
            Prediction[i] = clf1.predict(x_test[i,:].reshape(1, -1))
     

    ################################################## Data second Classifier
    print('################################################## Data second Classifier')


    train_choix_bool = (y_prob1_train > threshold1_low)
    test_choix_bool = (y_prob1 > threshold1_low)
    print('Train Clients %s'%Counter(y_train[train_choix_bool]))
    print('Test Clients %s'%Counter(y_test[test_choix_bool]))
    

    clf2.fit(x_train[train_choix_bool],y_train[train_choix_bool])
    with open('BRF_clf2_low.pkl', 'wb') as f:
        pickle.dump(clf2, f, pickle.HIGHEST_PROTOCOL)
    
    y_prob2 = clf2.predict_proba(x_test[test_choix_bool])[:,1]


    y_prob2_train = np.zeros(len(x_train))
    for i in range(len(x_train)):
        if (y_prob1_train[i] > threshold1_low):
            y_prob2_train[i] = clf2.predict_proba(x_train[i,:].reshape(1,-1))[:,1]
    
    
    
    Plot_Prob_Distribution.Plot_probability(y_test[test_choix_bool],y_prob2,threshold2_low,threshold2_low)
    
    y_prob2 = np.zeros(len(x_test))
    for i in range(len(x_test)):
        if  (y_prob1[i] > threshold1_low):

            y_prob2[i] = clf2.predict_proba(x_test[i,:].reshape(1,-1))[:,1]
    
            
    for i in range(len(y_test)):
        if (y_prob1[i]+y_prob2[i])/2 <= threshold2_low:
            Prediction[i] = -1
        else:
            Prediction[i] = clf2.predict(x_test[i,:].reshape(1, -1))
            
    
    ################################################## Data third Classifier
    print('################################################## Data third Classifier')

    train_choix_bool = (y_prob1_train>threshold1_low) & (y_prob2_train>threshold2_low) 
    test_choix_bool = (y_prob1>threshold1_low) & (y_prob2>threshold2_low) 
            
    print('Train Clients %s'%Counter(y_train[train_choix_bool]))
    print('Test Clients %s'%Counter(y_test[test_choix_bool]))
    
    clf3.fit(x_train[train_choix_bool],y_train[train_choix_bool])
    
    with open('BRF_clf3_low.pkl', 'wb') as f:
        pickle.dump(clf3, f, pickle.HIGHEST_PROTOCOL)
    
    y_prob3 = clf3.predict_proba(x_test[test_choix_bool])[:,1]
    
    y_prob3_train = np.zeros(len(x_train))
    for i in range(len(x_train)):
        if  (y_prob1_train[i]>threshold1_low) & (y_prob2_train[i]>threshold2_low) :
            y_prob3_train[i] = clf3.predict_proba(x_train[i,:].reshape(1,-1))[:,1]
    
    
    Plot_Prob_Distribution.Plot_probability(y_test[test_choix_bool],y_prob3,threshold3_low,threshold3_low)
    
    y_prob3 = np.zeros(len(x_test))
    for i in range(len(x_test)):
        if  (y_prob1[i]>threshold1_low) & (y_prob2[i]>threshold2_low) :

            y_prob3[i] = clf3.predict_proba(x_test[i,:].reshape(1,-1))[:,1]
    
    ##########  Model 1        
    for i in range(len(y_test)):
        if y_prob3[i] <= threshold3_low:
            Prediction[i] = -1
        else:
            Prediction[i] = clf3.predict(x_test[i,:].reshape(1, -1))

    ##########  Model 2        
    y_Prob = np.zeros(len(x_test))
    for i in range(len(x_test)):
        if  (y_prob1[i]<threshold1_low) :
            y_Prob[i] = -1
        else:
            if (y_prob1[i]+y_prob2[i])/2 < threshold2_low:
                y_Prob[i] = -1
            else:
                y_Prob[i] = (y_prob1[i]+y_prob2[i]+y_prob3[i])/3
            
    y_Pred = np.sign(y_Prob-0.5)
    
    return y_pred1, y_Pred



def Improved_BRF(x_train,y_train,x_test,y_test,threshold1_up,threshold2_up,threshold3_up,\
                 threshold1_low,threshold2_low,threshold3_low):
    
    clf2 = BalancedRandomForestClassifier(max_leaf_nodes=20,\
                                      n_estimators = 60,criterion = 'entropy',min_samples_leaf=20,min_samples_split=50,\
                                      max_depth=7, oob_score = True,random_state=10)

    clf3 = BalancedRandomForestClassifier(max_leaf_nodes=20,max_features = 20,\
        n_estimators = 160,criterion = 'entropy',min_samples_leaf=10,min_samples_split=10,\
        max_depth=13, oob_score = True,random_state=10)
    
    clf4 = BalancedRandomForestClassifier(max_leaf_nodes=20,max_features = 14,\
        n_estimators = 80,criterion = 'entropy',min_samples_leaf=10,min_samples_split=10,\
        max_depth=7, oob_score = True,random_state=10)
    
    ################################################## Data frist Classifier
    print('################################################## Data frist Classifier')
    print('Train Clients %s'%Counter(y_train))
    print('Test Clients %s'%Counter(y_test)) 
     
    clf2.fit(x_train,y_train)
    
    with open('BRF_clf1.pkl', 'wb') as f:
        pickle.dump(clf2, f, pickle.HIGHEST_PROTOCOL)
        
        
    y_pred2 = clf2.predict(x_test)

    y_prob2 = clf2.predict_proba(x_test)[:,1]
    y_prob2_train = clf2.predict_proba(x_train)[:,1]
    

    Plot_Prob_Distribution.Plot_probability(y_test,y_prob2,threshold1_low,threshold1_up)
    
    Prediction = np.zeros(y_test.shape)
    for i in range(len(y_test)):
        if y_prob2[i] >= threshold1_up:
            Prediction[i] = 1
        else:            
            Prediction[i] = clf2.predict(x_test[i,:].reshape(1, -1))
    ################################################## Data second Classifier
    print('################################################## Data second Classifier')
    train_choix_bool = (y_prob2_train<threshold1_up) & (y_prob2_train>threshold1_low)
    test_choix_bool = (y_prob2<threshold1_up)& (y_prob2>threshold1_low)

    print('Train Clients %s'%Counter(y_train[train_choix_bool]))
    print('Test Clients %s'%Counter(y_test[test_choix_bool]))
    

    clf3.fit(x_train[train_choix_bool],y_train[train_choix_bool])
    with open('BRF_clf2.pkl', 'wb') as f:
        pickle.dump(clf3, f, pickle.HIGHEST_PROTOCOL)
    
    y_prob3 = clf3.predict_proba(x_test[test_choix_bool])[:,1]


    y_prob3_train = np.zeros(len(x_train))
    for i in range(len(x_train)):
        if (y_prob2_train[i] < threshold1_up) & (y_prob2_train[i] > threshold1_low):
            y_prob3_train[i] = clf3.predict_proba(x_train[i,:].reshape(1,-1))[:,1]
    
    Plot_Prob_Distribution.Plot_probability(y_test[test_choix_bool],y_prob3,threshold2_low,threshold2_up)
    
    y_prob3 = np.zeros(len(x_test))
    for i in range(len(x_test)):
        if  (y_prob2[i] < threshold1_up)& (y_prob2[i] > threshold1_low):
            y_prob3[i] = clf3.predict_proba(x_test[i,:].reshape(1,-1))[:,1]
    
            
    for i in range(len(y_test)):
        if y_prob3[i] >= threshold2_up:
            Prediction[i] = 1
        else:
            Prediction[i] = clf3.predict(x_test[i,:].reshape(1, -1))
            
    
    ################################################## Data third Classifier
    print('################################################## Data third Classifier')
    train_choix_bool = (y_prob2_train<threshold1_up) & (y_prob2_train>threshold1_low) &\
                    (y_prob3_train<threshold2_up) & (y_prob3_train>threshold2_low)
    test_choix_bool = (y_prob2<threshold1_up) & (y_prob2>threshold1_low) &\
                    (y_prob3<threshold2_up) & (y_prob3>threshold2_low)

            
    print('Train Clients %s'%Counter(y_train[train_choix_bool]))
    print('Test Clients %s'%Counter(y_test[test_choix_bool]))
    
    clf4.fit(x_train[train_choix_bool],y_train[train_choix_bool])
    with open('BRF_clf3.pkl', 'wb') as f:
        pickle.dump(clf4, f, pickle.HIGHEST_PROTOCOL)
    
    y_prob4 = clf4.predict_proba(x_test[test_choix_bool])[:,1]
    
    y_prob4_train = np.zeros(len(x_train))
    for i in range(len(x_train)):
        if  (y_prob2_train[i]<threshold1_up) & (y_prob2_train[i]>threshold1_low) &\
                    (y_prob3_train[i]<threshold2_up) & (y_prob3_train[i]>threshold2_low) :
            y_prob4_train[i] = clf4.predict_proba(x_train[i,:].reshape(1,-1))[:,1]
            
    Plot_Prob_Distribution.Plot_probability(y_test[test_choix_bool],y_prob4,threshold3_low,threshold3_up)
    
    y_prob4 = np.zeros(len(x_test))
    for i in range(len(x_test)):
        if (y_prob2[i]<threshold1_up) & (y_prob2[i]>threshold1_low) &\
                    (y_prob3[i]<threshold2_up) & (y_prob3[i]>threshold2_low)  :
            y_prob4[i] = clf4.predict_proba(x_test[i,:].reshape(1,-1))[:,1]
    
    ##########  Model 1        
    for i in range(len(y_test)):
        if y_prob4[i] >= threshold3_up:
            Prediction[i] = 1
        else:
            Prediction[i] = clf4.predict(x_test[i,:].reshape(1, -1))

    ##########  Model 2        
    y_Prob = np.zeros(len(x_test))
    for i in range(len(x_test)):
        if  (y_prob2[i]>threshold1_up) :
            y_Prob[i] = 1
        elif (y_prob2[i]<threshold1_low):
            y_Prob[i] = -1
        else:
            if (y_prob2[i]+y_prob3[i])/2 >threshold2_up:
                y_Prob[i] = 1
                
            elif (y_prob2[i]+y_prob3[i])/2 <threshold2_low:
                y_Prob[i] = -1
                
            else:
                y_Prob[i] = (y_prob2[i]+y_prob3[i]+y_prob4[i])/3
            
    y_Pred = np.sign(y_Prob-0.5)
    
    return y_pred2,y_Pred





def Improved_BRF_up(x_train,y_train,x_test,y_test,threshold1_up,threshold2_up,threshold3_up):
    
    clf2 = BalancedRandomForestClassifier(max_leaf_nodes=20,\
                                      n_estimators = 60,criterion = 'entropy',min_samples_leaf=20,min_samples_split=50,\
                                      max_depth=7, oob_score = True,random_state=10)

    clf3 = BalancedRandomForestClassifier(max_leaf_nodes=20,max_features = 20,\
        n_estimators = 160,criterion = 'entropy',min_samples_leaf=10,min_samples_split=10,\
        max_depth=13, oob_score = True,random_state=10)
    
    clf4 = BalancedRandomForestClassifier(max_leaf_nodes=20,max_features = 14,\
        n_estimators = 80,criterion = 'entropy',min_samples_leaf=10,min_samples_split=10,\
        max_depth=7, oob_score = True,random_state=10)
    
    ################################################## Data frist Classifier
    print('################################################## Data frist Classifier')
    print('Train Clients %s'%Counter(y_train))
    print('Test Clients %s'%Counter(y_test)) 
     
    clf2.fit(x_train,y_train)
    
    with open('BRF_clf1_up.pkl', 'wb') as f:
        pickle.dump(clf2, f, pickle.HIGHEST_PROTOCOL)
        
        
    y_pred2 = clf2.predict(x_test)

    
    y_prob2 = clf2.predict_proba(x_test)[:,1]
    y_prob2_train = clf2.predict_proba(x_train)[:,1]
    

    Plot_Prob_Distribution.Plot_probability(y_test,y_prob2,threshold1_up,threshold1_up)
    
    Prediction = np.zeros(y_test.shape)
    for i in range(len(y_test)):
        if y_prob2[i] >= threshold1_up:
            Prediction[i] = 1
        else:            
            Prediction[i] = clf2.predict(x_test[i,:].reshape(1, -1))
    ################################################## Data second Classifier
    print('################################################## Data second Classifier')


    train_choix_bool = (y_prob2_train < threshold1_up)
    test_choix_bool = (y_prob2 < threshold1_up)
    print('Train Clients %s'%Counter(y_train[train_choix_bool]))
    print('Test Clients %s'%Counter(y_test[test_choix_bool]))
    

    clf3.fit(x_train[train_choix_bool],y_train[train_choix_bool])
    with open('BRF_clf2_up.pkl', 'wb') as f:
        pickle.dump(clf3, f, pickle.HIGHEST_PROTOCOL)
    
    y_prob3 = clf3.predict_proba(x_test[test_choix_bool])[:,1]


    y_prob3_train = np.zeros(len(x_train))
    for i in range(len(x_train)):
        if (y_prob2_train[i] < threshold1_up):
            y_prob3_train[i] = clf3.predict_proba(x_train[i,:].reshape(1,-1))[:,1]
    
    Plot_Prob_Distribution.Plot_probability(y_test[test_choix_bool],y_prob3,threshold2_up,threshold2_up)
    
    y_prob3 = np.zeros(len(x_test))
    for i in range(len(x_test)):
        if  (y_prob2[i] < threshold1_up):
            y_prob3[i] = clf3.predict_proba(x_test[i,:].reshape(1,-1))[:,1]
    
            
    for i in range(len(y_test)):
        if y_prob3[i] >= threshold2_up:
            Prediction[i] = 1
        else:
            Prediction[i] = clf3.predict(x_test[i,:].reshape(1, -1))
            
    
    ################################################## Data third Classifier
    print('################################################## Data third Classifier')

    train_choix_bool = (y_prob2_train<threshold1_up) & (y_prob3_train<threshold2_up) 
    test_choix_bool = (y_prob2<threshold1_up) & (y_prob3<threshold2_up) 
            
    print('Train Clients %s'%Counter(y_train[train_choix_bool]))
    print('Test Clients %s'%Counter(y_test[test_choix_bool]))
    
    clf4.fit(x_train[train_choix_bool],y_train[train_choix_bool])
    with open('BRF_clf3_up.pkl', 'wb') as f:
        pickle.dump(clf4, f, pickle.HIGHEST_PROTOCOL)
    
    y_prob4 = clf4.predict_proba(x_test[test_choix_bool])[:,1]
    
    y_prob4_train = np.zeros(len(x_train))
    for i in range(len(x_train)):
        if  (y_prob2_train[i]<threshold1_up) & (y_prob3_train[i]<threshold2_up) :
            y_prob4_train[i] = clf4.predict_proba(x_train[i,:].reshape(1,-1))[:,1]
            
    Plot_Prob_Distribution.Plot_probability(y_test[test_choix_bool],y_prob4,threshold3_up,threshold3_up)
    
    y_prob4 = np.zeros(len(x_test))
    for i in range(len(x_test)):
        if  (y_prob2[i]<threshold1_up) & (y_prob3[i]<threshold2_up) :
            y_prob4[i] = clf4.predict_proba(x_test[i,:].reshape(1,-1))[:,1]
    
    ##########  Model 1        
    for i in range(len(y_test)):
        if y_prob4[i] >= threshold3_up:
            Prediction[i] = 1
        else:
            Prediction[i] = clf4.predict(x_test[i,:].reshape(1, -1))

    ##########  Model 2        
    y_Prob = np.zeros(len(x_test))
    for i in range(len(x_test)):
        if  (y_prob2[i]>threshold1_up) :
            y_Prob[i] = 1
        else:
            if y_prob3[i]  > threshold2_up:
                y_Prob[i] = 1
            else:
                y_Prob[i] = (y_prob2[i]+y_prob3[i]+y_prob4[i])/3
            
    y_Pred = np.sign(y_Prob-0.5)
    
    return y_pred2,y_Pred

