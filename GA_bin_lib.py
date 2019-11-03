# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:25:23 2019

@author: Asus
"""

import numpy as np

#load data
import pandas as pd
import matplotlib.pyplot as plt

def rand_max20(len_x):
    len_x = len_x - 1 # n unique values, n-1 cutpoints
    if len_x > 20:
        a = np.random.randint(low=0,high=2, size=20) # max 1 is 20
        b = np.zeros(len_x - 20) # 118 - 20 are all 0
        c = np.hstack([a,b])
        np.random.shuffle(c) #shuffle
    else:
        c = np.random.randint(low=0,high=2, size=len_x)
    
    return c

dist_g_fn = lambda x,G_total: 1.0*(x.sum()+0.5) / G_total
dist_b_fn = lambda x,B_total: 1.0*(x.sum()+0.5) / B_total
n_fn = lambda x: x.sum()

def save_woe(individual,X,G,B,N,G_total,B_total,N_total,x_colname,p_cost_fn_name,folder_name='woe_pic'):
    cutpoints = np.where(individual == 1)[0] + 1
    
    G_splits = np.split(G, cutpoints)
    G_splits = [i for i in G_splits if i.size > 0] #remove empty
    
    B_splits = np.split(B, cutpoints)
    B_splits = [i for i in B_splits if i.size > 0] #remove empty
    
    N_splits = np.split(N, cutpoints)
    N_splits = [i for i in N_splits if i.size > 0] #remove empty
    
    X_splits = np.split(X, cutpoints)
    X_splits = [i.astype(np.float) for i in X_splits if i.size > 0] #remove empty
    
    x_range = [ "{0}-{1}".format(arr.min(),arr.max()) for arr in X_splits]
    x_min = [ arr.min() for arr in X_splits]
    x_max = [ arr.max() for arr in X_splits]
    

    dist_G = np.array([dist_g_fn(i,G_total) for i in G_splits])
    dist_B = np.array([dist_b_fn(i,B_total) for i in B_splits])
    
    woe = np.log(dist_G/dist_B)
    iv = (1.0*(dist_G - dist_B)*np.log(dist_G/dist_B))
    
    n = np.array(list(map(n_fn,N_splits)))
    n_perc = (n/N_total).round(2)
    n_good = np.array(list(map(n_fn,G_splits)))
    n_bad = np.array(list(map(n_fn,B_splits)))
    
    
    r = 'darkred'
    g = 'darkgreen'
    sel_iv_tab = pd.DataFrame({"RANGE":x_range
                               ,"X_MIN":x_min
                               ,"X_MAX":x_max
                               ,"N":n
                               ,"%N":n_perc
                               ,"#GOOD":n_good
                               ,"#BAD":n_bad
                               , "WOE":woe
                               , "IV":iv})
    ax = sel_iv_tab.plot(kind='bar',x='RANGE',y='WOE',ylim = (-2.5,2.5), width=0.9
                     , color=sel_iv_tab["WOE"].apply(lambda x: g if x>0 else r))
    ax.set_xlabel("Ranges")
    ax.set_ylabel("WoE")
    plt.tight_layout()
#    #plt.savefig(x_colname+p_cost_fn_name+'.png', transparent=True)
    plt.savefig('./{0}/'.format(folder_name)+x_colname+p_cost_fn_name+'.png')
    plt.clf()
    #print("IV",iv.sum())
    return sel_iv_tab

def display_woe(individual,X,G,B,N,G_total,B_total,N_total,x_colname,p_cost_fn_name):
    cutpoints = np.where(individual == 1)[0] + 1
    
    G_splits = np.split(G, cutpoints)
    G_splits = [i for i in G_splits if i.size > 0] #remove empty
    
    B_splits = np.split(B, cutpoints)
    B_splits = [i for i in B_splits if i.size > 0] #remove empty
    
    N_splits = np.split(N, cutpoints)
    N_splits = [i for i in N_splits if i.size > 0] #remove empty
    
    X_splits = np.split(X, cutpoints)
    X_splits = [i for i in X_splits if i.size > 0] #remove empty
    
    x_range = [ "{0}-{1}".format(arr.min(),arr.max()) for arr in X_splits]
    x_min = [ arr.min() for arr in X_splits]
    x_max = [ arr.max() for arr in X_splits]
    
    #dist_G = np.array(list(map(dist_g_fn,G_splits)))
    dist_G = np.array([dist_g_fn(i,G_total) for i in G_splits])
    #dist_B = np.array(list(map(dist_b_fn,B_splits)))
    dist_B = np.array([dist_b_fn(i,B_total) for i in B_splits])
    
    woe = np.log(dist_G/dist_B)
    iv = (1.0*(dist_G - dist_B)*np.log(dist_G/dist_B))
    
    n = np.array(list(map(n_fn,N_splits)))
    n_good = np.array(list(map(n_fn,G_splits)))
    n_bad = np.array(list(map(n_fn,B_splits)))
    
    
    r = 'darkred'
    g = 'darkgreen'
    sel_iv_tab = pd.DataFrame({"RANGE":x_range
                               ,"X_MIN":x_min
                               ,"X_MAX":x_max
                               ,"N":n
                               ,"#GOOD":n_good
                               ,"#BAD":n_bad
                               , "WOE":woe
                               , "IV":iv})
    ax = sel_iv_tab.plot(kind='bar',x='RANGE',y='WOE',ylim = (-2.5,2.5), width=0.9
                     , color=sel_iv_tab["WOE"].apply(lambda x: g if x>0 else r))
    ax.set_xlabel("Ranges")
    ax.set_ylabel("WoE")
    plt.tight_layout()
    
    return sel_iv_tab
def IV_noconst(individual,X,G,B,N,G_total,B_total,N_total
            ,p_min_woe_diff=0.05
            ,p_min_perc=0.05
            ,p_max_perc=0.45
            ,p_penalty_weight=0.5):
    #force 4 bins, 3 cutpoints or max 20 bins, 19 cut points (Death penalty)
    if sum(individual) > len(X)-1-1 or sum(individual) > 19 or sum(individual) == 0:
        return -100000,
    cutpoints = np.where(individual == 1)[0] + 1
    
    G_splits = np.split(G, cutpoints)
    G_splits = [i for i in G_splits if i.size > 0] #remove empty
    B_splits = np.split(B, cutpoints)
    B_splits = [i for i in B_splits if i.size > 0] #remove empty
    N_splits = np.split(N, cutpoints)
    N_splits = [i for i in N_splits if i.size > 0] #remove empty
    
    dist_G = np.array([dist_g_fn(i,G_total) for i in G_splits])
    dist_B = np.array([dist_b_fn(i,B_total) for i in B_splits])
    
    iv = (1.0*(dist_G - dist_B)*np.log(dist_G/dist_B))    
    
    n = np.array(list(map(n_fn,N_splits)))
    pect_n = n/N_total
    
    pos_iv = iv[(pect_n >= p_min_perc)]
    neg_iv = iv[ ~(pect_n >= p_min_perc) ]
    
    IV = pos_iv.sum() - p_penalty_weight*neg_iv.sum()
    
    IV = round(IV,6)
    return IV,

def IV_mono(individual,X,G,B,N,G_total,B_total,N_total,penalty_weight1=1):
    #force 4 bins, 3 cutpoints or max 20 bins, 19 cut points (Death penalty)
    if individual.sum() > len(X)-1-1 or individual.sum() > 19 or individual.sum() == 0:
        return -10000,

    cutpoints = np.where(individual == 1)[0] + 1
    
    G_splits = np.split(G, cutpoints)
    G_splits = [i for i in G_splits if i.size > 0] #remove empty
    B_splits = np.split(B, cutpoints)
    B_splits = [i for i in B_splits if i.size > 0] #remove empty
    N_splits = np.split(N, cutpoints)
    N_splits = [i for i in N_splits if i.size > 0] #remove empty
    dist_G = np.array([dist_g_fn(i,G_total) for i in G_splits])
    dist_B = np.array([dist_b_fn(i,B_total) for i in B_splits])
    woe = np.log(dist_G/dist_B)
    
    is_incr = ( np.diff(woe) >= 0.05*np.max(woe) )
    is_decr = ( np.diff(woe) <= -0.05*np.max(woe) )
    
    if is_incr.sum() >= is_decr.sum():
        mono_chk = np.hstack(([True],is_incr)) # trend is more toward increasing
    else:
        mono_chk = np.hstack(([True],is_decr))
        
    iv = (1.0*(dist_G - dist_B)*np.log(dist_G/dist_B))    
    
    n = np.array(list(map(n_fn,N_splits)))
    
    n_from_min = n - np.ceil(N_total*0.05)
    n_from_max = np.ceil(2*N_total/iv.shape[0]) - n #2 times of avg
    
    
    f_x = iv.sum() \
    + penalty_weight1*(n_from_min[ n_from_min < 0].sum() \
                      + n_from_max[ n_from_max < 0].sum()) \
    - penalty_weight1*n[~mono_chk].sum() # penalty minus values
    
    f_x = round(f_x,6)
    return f_x,

def IV_peaks(individual,X,G,B,N,G_total,B_total,N_total
            ,p_min_woe_diff=0.05
            ,p_min_perc=0.05
            ,p_max_perc=0.45
            ,p_penalty_weight1=1
            ,p_penalty_weight2=2
            ,p_n_peak=1):
    #force 4 bins, 3 cutpoints or max 20 bins, 19 cut points (Death penalty)
    if individual.sum() > len(X)-1-1 or individual.sum() > 19 or individual.sum() == 0:
        return -100000,
    cutpoints = np.where(individual == 1)[0] + 1
    
    G_splits = np.split(G, cutpoints)
    G_splits = [i for i in G_splits if i.size > 0] #remove empty
    
    B_splits = np.split(B, cutpoints)
    B_splits = [i for i in B_splits if i.size > 0] #remove empty
    
    N_splits = np.split(N, cutpoints)
    N_splits = [i for i in N_splits if i.size > 0] #remove empty
    
    dist_G = np.array([dist_g_fn(i,G_total) for i in G_splits])
    dist_B = np.array([dist_b_fn(i,B_total) for i in B_splits])
    woe = np.log(dist_G/dist_B)
    iv = (1.0*(dist_G - dist_B)*np.log(dist_G/dist_B)) 
    
    n = np.array(list(map(n_fn,N_splits)))
    n_from_min = n - np.ceil(N_total*p_min_perc)
    n_from_max = np.ceil(2*N_total/iv.shape[0]) - n #2 times of avg
    
    local_min = np.r_[False, woe[1:] < woe[:-1] ] \
                            & np.r_[woe[:-1] < woe[1:] , False]
    local_max = np.r_[False, woe[1:] > woe[:-1] ] \
                            & np.r_[woe[:-1] > woe[1:], False]
                            
    is_incr = ( np.diff(woe) >= p_min_woe_diff*np.max(woe) )
    is_decr = ( np.diff(woe) <= -p_min_woe_diff*np.max(woe) )
    diff_5perc = np.hstack(([True],is_incr | is_decr))                            
    
    if local_min.sum() + local_max.sum() <= p_n_peak:
        f_x = iv.sum() \
        + p_penalty_weight1*(n_from_min[ n_from_min < 0].sum() \
                          + n_from_max[ n_from_max < 0].sum())\
        - p_penalty_weight1*n[~diff_5perc].sum()                         
    else:
        f_x = iv.sum() \
        + p_penalty_weight1*(n_from_min[ n_from_min < 0].sum() \
                          + n_from_max[ n_from_max < 0].sum()) \
        - p_penalty_weight1*n[~diff_5perc].sum() \
        - p_penalty_weight1*n[local_min | local_max].sum() # penalty minus values
    
    f_x = round(f_x,6)
    return f_x,

def IV_1peak(individual,X,G,B,N,G_total,B_total,N_total
            ,p_min_woe_diff=0.05
            ,p_min_perc=0.05
            ,p_max_perc=0.45
            ,p_penalty_weight1=1
            ,p_penalty_weight2=2
            ):
    return IV_peaks(individual,X,G,B,N,G_total,B_total,N_total
            ,p_min_woe_diff=p_min_woe_diff
            ,p_min_perc=p_min_perc
            ,p_max_perc=p_max_perc
            ,p_penalty_weight1=p_penalty_weight1
            ,p_penalty_weight2=p_penalty_weight2
            ,p_n_peak=1)

def IV_2peak(individual,X,G,B,N,G_total,B_total,N_total
            ,p_min_woe_diff=0.05
            ,p_min_perc=0.05
            ,p_max_perc=0.45
            ,p_penalty_weight1=1
            ,p_penalty_weight2=2
            ):
    return IV_peaks(individual,X,G,B,N,G_total,B_total,N_total
            ,p_min_woe_diff=p_min_woe_diff
            ,p_min_perc=p_min_perc
            ,p_max_perc=p_max_perc
            ,p_penalty_weight1=p_penalty_weight1
            ,p_penalty_weight2=p_penalty_weight2
            ,p_n_peak=2)    
