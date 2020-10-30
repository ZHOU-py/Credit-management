#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:53:18 2020

@author: nelson
"""
import matplotlib.pyplot as plt
import numpy as np

name_list = ['0%','5%','10%','15%','20%','25%','30%','35%','40%','45%','50%','55%','60%','65%',\
             '70%','75%','80%','85%','90%','95%','100%']

def Plot_probability(y_test,y_prob,threshold_low,threshold_up):
    plt.figure(figsize=(10,5))
    num_list_bad = [0]
    for i in np.arange(0,1,0.05):
        num_list_bad.append(sum(y_test[(y_prob>=i)&(y_prob<i+0.05)]==-1))
        
        
    num_list_good = [0]
    for i in np.arange(0,1,0.05):
        num_list_good.append(sum(y_test[(y_prob>=i)&(y_prob<i+0.05)]==1))

            
    x =list(range(len(num_list_bad)))
    total_width, n = 0.8, 2
    width = total_width / n
     
    
    plt.bar(x, num_list_bad, width=width, label='Actual Bad',fc = 'goldenrod')
    for a,b in zip(x,num_list_bad):
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=7)
        
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list_good, width=width, label='Actual Good',tick_label = name_list,fc = 'cornflowerblue')
    
    for a,b in zip(x,num_list_good):
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=7)
     
    plt.vlines(x[int(threshold_up/0.05)], 0, 150, colors = "r", linestyles = "dashed")
    plt.vlines(x[int(threshold_low/0.05)], 0, 150, colors = "r", linestyles = "dashed")
    plt.xlabel('Predicted probability of being good')
    plt.ylabel('Number of the clients')
    plt.title(" Distribution of actual good clients and bad clients across the predicted probabilities of being good")
    plt.legend()
    plt.show()