#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:37:14 2020

@author: nelson
"""

import Data_Numeric
import pandas as pd
import DataDeal
import numpy as np
import FSVM
import seaborn as sns
import matplotlib.pyplot as plt

import pickle


def IV_plot(model):
    data = pd.read_csv('german_credit.csv')
        #    print(data.describe())
    X = data.drop(['default'],axis = 1)
    
    if model=='Origine':
    
        Y = data['default'].copy()
        Y=Y-1
        Y[Y==-1]=1
    
    
    elif model=='FSVM':
    
        lable = data['default']
        df = Data_Numeric.Data_numerique(X)
        data = DataDeal.get_data(df,lable)
        x = data[:,:-1]
        with open('save/FSVM_Cen_Lin_RBF_Origine.pickle', 'rb') as f:
            clf = pickle.load(f)
        
        y_pred = clf.predict(x)
        y = y_pred.copy()
        y[y==-1]=0
        y = y.astype('int64')
        y_df = pd.DataFrame({"Yp":y})
        Y = y_df['Yp']
    
    elif model=='LSFSVM':
        lable = data['default']
        df = Data_Numeric.Data_numerique(X)
        data = DataDeal.get_data(df,lable)
        x = data[:,:-1]
        with open('save/LSFSVM_Cen_Lin_RBF_Origine.pickle', 'rb') as f:
            clf = pickle.load(f)
        
        y_pred = clf.predict(x)
        y = y_pred.copy()
        y[y==-1]=0
        y = y.astype('int64')
        y_df = pd.DataFrame({"Yp":y})
        Y = y_df['Yp']
        
    elif model=='LSFSVM_bagging':
        lable = data['default']
        df = Data_Numeric.Data_numerique(X)
        data = DataDeal.get_data(df,lable)
        x = data[:,:-1]
        with open('save/LSFSVMbag_Cen_Lin_RBF_Origine.pickle', 'rb') as f:
            clf = pickle.load(f)
        
        y_pred = clf.predict(x)
        y = y_pred.copy()
        y[y==-1]=0
        y = y.astype('int64')
        y_df = pd.DataFrame({"Yp":y})
        Y = y_df['Yp']
        
    elif model=='FSVM_bagging':
        lable = data['default']
        df = Data_Numeric.Data_numerique(X)
        data = DataDeal.get_data(df,lable)
        x = data[:,:-1]
        with open('save/FSVMbag_Cen_Lin_RBF_Origine.pickle', 'rb') as f:
            clf = pickle.load(f)
        
        y_pred = clf.predict(x)
        y = y_pred.copy()
        y[y==-1]=0
        y = y.astype('int64')
        y_df = pd.DataFrame({"Yp":y})
        Y = y_df['Yp']
    
    badnum=len(Y[Y==0])    # amount of bad clients
    goodnum=Y.count()-badnum    # amount of good clients
    
    
    
    def self_bin_object(X):
        d1 = pd.DataFrame({"X": X, "Y": Y,"Bucket": X})#create a DateFrame X-- attribut ， Y--label ， Bucket--each binning    
        d2 = d1.groupby('Bucket', as_index = True)# Group and aggregate according to binning results
        d3 = pd.DataFrame(d2.count(), columns = ['good'])      
        d3['good'] = d2.sum().Y    
        d3['total'] = d2.count().Y   
        d3['bad'] =  d3['total'] - d3['good']
        d3['rate'] = d2.mean().Y
        d3['woe']=np.log((d3['bad']/badnum)/(d3['good']/goodnum))# calcuate WOE of each binning
        d3['badattr'] = d3['bad']/badnum  # distribution of bad clients in each binning
        d3['goodattr'] = d3['good']/goodnum  # distribution of good clients in each binning
        iv = ((d3['badattr']-d3['goodattr'])*d3['woe']).sum()  # calculate Information VAlue
        d4 = (d3.sort_index(by ='good')).reset_index(drop=True)   # ranking 
        woe=list(d4['woe'].round(3))
        return iv,d3,woe
    
    
    def self_bin_numeric(X,cut):
        d1 = pd.DataFrame({"X": X, "Y": Y,"Bucket": pd.cut(X, cut)})    
        d2 = d1.groupby('Bucket', as_index = True)
        d3 = pd.DataFrame(d2.count(), columns = ['good'])      
        d3['good'] = d2.sum().Y    
        d3['total'] = d2.count().Y   
        d3['bad'] =  d3['total']- d3['good']
        d3['rate'] = d2.mean().Y
        d3['woe']=np.log((d3['bad']/badnum)/(d3['good']/goodnum))
        d3['badattr'] = d3['bad']/badnum  
        d3['goodattr'] = d3['good']/goodnum  
        iv = ((d3['badattr']-d3['goodattr'])*d3['woe']).sum()  
#        d4 = (d3.sort_index(by ='good')).reset_index(drop=True)  
        d4 = (d3.sort_index(axis=1)).reset_index(drop=True) 
        woe=list(d4['woe'].round(3))
        return iv,d3,woe
        
        
    iv_fw =  self_bin_object(X['foreign_worker'])[0] 
    iv_acs =  self_bin_object(X['account_check_status'])[0]     
    iv_ch =  self_bin_object(X['credit_history'])[0]
    iv_pur =  self_bin_object(X['purpose'])[0]
    iv_sav =  self_bin_object(X['savings'])[0]
    iv_pes =  self_bin_object(X['present_emp_since'])[0]
    iv_pss =  self_bin_object(X['personal_status_sex'])[0]
    iv_od =  self_bin_object(X['other_debtors'])[0]   
    iv_pro =  self_bin_object(X['property'])[0] 
    iv_oip =  self_bin_object(X['other_installment_plans'])[0] 
    iv_hous =  self_bin_object(X['housing'])[0] 
    iv_job =  self_bin_object(X['job'])[0] 
    iv_tele =  self_bin_object(X['telephone'])[0] 
     
        
    iv_iaip =  self_bin_object(X['installment_as_income_perc'])[0]   
    iv_prs =  self_bin_object(X['present_res_since'])[0] 
    iv_ctb =  self_bin_object(X['credits_this_bank'])[0] 
    iv_pum =  self_bin_object(X['people_under_maintenance'])[0]    
    
    
    iv_dim =  self_bin_numeric(X['duration_in_month'],4)[0]  
    iv_ca =  self_bin_numeric(X['credit_amount'],5)[0]
    iv_age =  self_bin_numeric(X['age'],5)[0] 
    
#    print(self_bin_object(X['credits_this_bank']))
#    print(self_bin_numeric(X['duration_in_month'],4))
     
    IV = [iv_acs,iv_dim,iv_ch,iv_pur,iv_ca,iv_sav,iv_pes,iv_iaip,iv_pss,iv_od,iv_prs,\
          iv_pro,iv_age,iv_oip,iv_hous,iv_ctb,iv_job,iv_pum,iv_tele,iv_fw]
    
            
    indices = np.argsort(IV)[::-1]
    featurerank=[]
    for f in range(len(indices)):
        featurerank.append(X.columns[indices[f]])
                
    
    plt.figure(figsize=(10,8))
    feature_imp = pd.Series(IV,index=X.columns).sort_values(ascending=False)
    
    ivlist = pd.Series(IV).sort_values(ascending=False)
    ivlist.values
    for a,b in zip(ivlist.values,np.arange(0.2,20.2,1)):
        plt.text(a, b, round(a,4))
    
    #plt.text(ivlist.values[0]-1, 2.2, '1.148')
    sns.barplot(x=feature_imp,y=feature_imp.index)
    
    #plt.vlines(auc_complete,feature_imp.index[19], feature_imp.index[0])
    plt.xlim((0, 1))
    #plt.ylim((0, 1))
    plt.xlabel('Information Value')
    plt.ylabel('Attribut')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()


if __name__ == '__main__':
#    IV_plot('Origine')
#    IV_plot('LSFSVM')
    IV_plot('FSVM')
#    IV_plot('FSVM_bagging')
#    IV_plot('LSFSVM_bagging')








