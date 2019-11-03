# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:41:15 2019

@author: Asus
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dist_g_fn = lambda x,G_total: 1.0*(x.sum()+0.5) / G_total
dist_b_fn = lambda x,B_total: 1.0*(x.sum()+0.5) / B_total
n_fn = lambda x: x.sum()

def cal_iv_for_testset(df_test,iv_tab_train,y_colname,x_colname,p_good_ind):
    #start
    if p_good_ind == 0:
        #convert Y
        df_test = pd.concat([1- df_test[y_colname], df_test[x_colname]], axis=1, keys=['Y', 'X'])
    else:
        df_test = pd.concat([df_test[y_colname], df_test[x_colname]], axis=1, keys=['Y', 'X'])
    
    df_test = df_test.loc[df_test['X'] != "."]
    #copy the first 3 cols from the iv_tab_train
    iv_tab_test = iv_tab_train[['RANGE','X_MIN','X_MAX']].copy()
    Y = df_test['Y'].values.astype(np.float) 
    X = df_test['X'].values.astype(np.float)
    
    N_list = []
    Prec_N_list = []
    G_list = []
    B_list = []
    woe_list = []
    iv_list = []
    N_total = len(X)
    G_total = Y.sum()
    B_total = N_total - G_total
    for i in range(0,iv_tab_test.shape[0]):
        #print(i)
        row = iv_tab_test.iloc[i]
        if i < iv_tab_test.shape[0] - 1:
            #not last row
            next_row = iv_tab_test.iloc[i+1]
        else:
            next_row = None
        #print(X)
        #print(row)
        if i == 0:
            #first row
            N = len(X[X < next_row.X_MIN]) #N
            G = (Y[X < next_row.X_MIN]).sum()
        elif i == iv_tab_test.shape[0] - 1:
            #last row
            N = len(X[X >= row.X_MIN]) #N
            G = (Y[X >= row.X_MIN]).sum()
        else:
            N = len(X[(X >= row.X_MIN) & (X < next_row.X_MAX)]) #N
            G = (Y[(X >= row.X_MIN) & (X < next_row.X_MAX)]).sum()
            
        perc_N = round(N/N_total,2) #N
        
        B = N - G
        
        dist_G = dist_g_fn(G,G_total)
        dist_B = dist_b_fn(B,B_total)
        
        woe = np.log(dist_G/dist_B)
        iv = (1.0*(dist_G - dist_B)*np.log(dist_G/dist_B))
        
        N_list.append(N)
        Prec_N_list.append(perc_N)
        G_list.append(G)
        B_list.append(B)
        woe_list.append(woe)
        iv_list.append(iv)
        
    iv_tab_test['N'] =  N_list
    iv_tab_test['%N'] =  Prec_N_list
    iv_tab_test['#GOOD'] =  G_list
    iv_tab_test['#BAD'] =  B_list
    iv_tab_test['WOE'] =  woe_list
    iv_tab_test['IV'] =  iv_list
    
    return iv_tab_test,iv_tab_test['IV'].sum()
        
def save_woe_test(sel_iv_tab,x_colname,p_cost_fn_name,folder_name='woe_pic'):
    r = 'darkred'
    g = 'darkgreen'
    ax = sel_iv_tab.plot(kind='bar',x='RANGE',y='WOE',ylim = (-2.5,2.5), width=0.9
                     , color=sel_iv_tab["WOE"].apply(lambda x: g if x>0 else r))
    ax.set_xlabel("Ranges")
    ax.set_ylabel("WoE")
    plt.tight_layout()
#    #plt.savefig(x_colname+p_cost_fn_name+'.png', transparent=True)
    plt.savefig('./{0}/'.format(folder_name)+x_colname+p_cost_fn_name+'.png')
    plt.clf()

