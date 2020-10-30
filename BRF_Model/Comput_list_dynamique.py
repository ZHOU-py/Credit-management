#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:51:48 2020

@author: nelson
"""

import numpy as np

name_list = ['0%','5%','10%','15%','20%','25%','30%','35%','40%','45%','50%','55%','60%','65%',\
             '70%','75%','80%','85%','90%','95%','100%']

def Comput_threshold(y_test,y_prob,classifier_threshold_low,classifier_threshold_up):
    num_list_bad = [0]
    for i in np.arange(0,1,0.05):
        num_list_bad.append(sum(y_test[(y_prob>=i)&(y_prob<i+0.05)]==-1))        
        
    num_list_good = [0]
    for i in np.arange(0,1,0.05):
        num_list_good.append(sum(y_test[(y_prob>=i)&(y_prob<i+0.05)]==1))
    
    ratio_good_bad = np.zeros((np.array(num_list_good).shape))
    for i in range(len(num_list_good)):
        if num_list_good[i]==0:
            ratio_good_bad[i] = 0
        else:
            if i >= name_list.index('50%'):
                if num_list_bad[i] == 0:
                    ratio_good_bad[i] = classifier_threshold_up
                elif num_list_bad[i] != 0:
                    ratio_good_bad[i] = np.array(num_list_good[i])/np.array(num_list_bad[i])
                    
            elif i < name_list.index('50%'):
                if num_list_bad[i] == 0:
                    ratio_good_bad[i] = classifier_threshold_low
                elif num_list_bad[i] != 0:
                    ratio_good_bad[i] = np.array(num_list_good[i])/np.array(num_list_bad[i])
                
    #good clients / bad clients upper than 100, we set this as threshold.
    threshold_low = 0.5            
    for i in range(name_list.index('50%'),0,-1):
        print('ratio_good_bad{}:{}'.format(name_list[i],ratio_good_bad[i]))
        if (ratio_good_bad[i] <= classifier_threshold_low):   
            threshold_low = i*0.05+0.05
            if threshold_low <= 0.5:
                print(i)
                break
            else:
                threshold_low = 0.5
                break
        else:
            continue
        
    threshold_up = 0.5            
    for i in range(name_list.index('50%'),len(ratio_good_bad),1):
        print('ratio_good_bad{}:{}'.format(name_list[i],ratio_good_bad[i]))
        if (ratio_good_bad[i] >= classifier_threshold_up):   
            threshold_up = i*0.05-0.05
            if threshold_up >= 0.5:
                print(i)
                break
            else:
                threshold_up = 0.5
                break
        else:
            continue
        
    return threshold_low,threshold_up