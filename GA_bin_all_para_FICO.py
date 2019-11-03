# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:59:45 2019

@author: Asus
"""

import numpy as np

#load data
import pandas as pd
import random
from deap import base
from deap import creator
from deap import tools
import timeit
import tqdm
import multiprocessing  as mp
import itertools

#import from GA_bin_lib.py
from GA_bin_lib import *

def ga_bin(zip_params):
    df,x_colname,y_colname,p_good_ind,p_cost_fn,p_cost_fn_name = zip_params
    p_inv_gen_fn=rand_max20
    p_n_population = 6000
    p_inv_mutate=0.05
    p_tournsize=3
    p_CXPB=0.5
    p_MUTPB=0.1
    p_no_develop_times=20
    p_n_max_gens=300
    """
    INPUT:
    df,x_colname,y_colname,good_ind
    p_cost_fn :     type=function; GA fitness
    X :             type=np.array; bin IDs
    p_inv_mutate :  type=float; Chance of mutate of an individual bit
    p_tournsize :   type=int; select p_tournsize to breed the next gen
    p_CXPB :        type=float; the probability with which two individuals are crossed
    p_MUTPB :       type=float; the probability for mutating an individual
    """
    
    #start
    if p_good_ind == 0:
        #convert Y
        df_bin = pd.concat([1- df[y_colname], df[x_colname]], axis=1, keys=['Y', 'X'])
    else:
        df_bin = pd.concat([df[y_colname], df[x_colname]], axis=1, keys=['Y', 'X'])
    
    #fix NA for FICO it's "."
    df_bin = df_bin.loc[df_bin['X'] != "."]
    #df_bin['X'] = round(df_bin['X']) #added round
    #df_bin['X'].fillna(-np.inf)
    group_obj = df_bin.groupby('X')
    X = np.array([*group_obj.groups.keys()])
    G = np.array(group_obj['Y'].sum())
    N = np.array(group_obj['X'].count())
    B = N - G
    
    G_total = G.sum()
    B_total = B.sum()
    N_total = N.sum()
        
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray , fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Attribute generator 
    #                      define 'attr_bool' to be an attribute ('gene')
    #                      which corresponds to integers sampled uniformly
    #                      from the range [0,1] (i.e. 0 or 1 with equal
    #                      probability)
    #toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("attr_bool", rand_max20, len(X))
    
    # Structure initializers
    #                         define 'individual' to be an individual
    #                         consisting of 100 'attr_bool' elements ('genes')
    #toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(X))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_bool)
    
    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    #----------
    # Operator registration
    #----------
    # register the goal / fitness function
    #toolbox.register("evaluate", IV_mono)
    #toolbox.register("evaluate", IV_fluc)
    #toolbox.register("evaluate", IV_noconst)
    toolbox.register("evaluate", p_cost_fn)
    #toolbox.register("evaluate", IV_2peak)
    
    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)
    
    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutFlipBit, indpb=p_inv_mutate)
    
    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=p_tournsize)
    
    pop = toolbox.population(n=p_n_population)

    CXPB, MUTPB = p_CXPB, p_MUTPB
    
    # Evaluate the entire population
    fitnesses = [ toolbox.evaluate(i,X,G,B,N,G_total,B_total,N_total) for i in pop ]
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    no_develop_times = 0
    old_max = 0
    # Begin the evolution
    while no_develop_times < p_no_develop_times and g < p_n_max_gens:
        # A new generation
        g = g + 1
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        #list [start:end:step]
        for child1, child2 in zip(offspring[0::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [ toolbox.evaluate(i,X,G,B,N,G_total,B_total,N_total) for i in invalid_ind ]
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = np.array([ind.fitness.values[0] for ind in pop])
        
        if old_max >= fits.max():
            no_develop_times = no_develop_times + 1
        else:
            old_max = fits.max()
            no_develop_times = 0
    
    best_ind = tools.selBest(pop, 1)[0]
    #print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    sel_iv_tab = save_woe(best_ind,X,G,B,N,G_total,B_total,N_total,x_colname,p_cost_fn_name,'FICO_woe_pic_all')
    
    return x_colname,p_cost_fn,sel_iv_tab

def bin_all(p_df,p_good_ind,y_colname):
    sol_rows = []
    iv_tab_list = []
    
    #SETUP your desired cost function (fitness function) here
    #all_cost_fn = [IV_mono,IV_1peak,IV_2peak,IV_noconst]
    all_cost_fn = [IV_2peak]
    #all_cost_fn_name = ['mono','1peak','2peak','noconst']
    all_cost_fn_name = ['2peak']
    
    pool = mp.Pool(mp.cpu_count())
    
    results = pool.map(ga_bin, tqdm.tqdm([ (p_df
                                            ,i[1] #x_colname
                                            ,y_colname
                                            ,p_good_ind
                                            ,i[0][0] #p_cost_fn
                                            ,i[0][1] #p_cost_fn_name
                                            ) \
                                          for i \
                                          in itertools.product(zip(all_cost_fn,all_cost_fn_name)\
                                                               ,p_df.columns[p_df.columns != y_colname])]) ) #parallel call
    
    pool.close()
    pool.join()
        
    #save
    
    for count,r in enumerate(results):
        x_colname,p_cost_fn,sel_iv_tab = r[0],r[1],r[2]
        row = dict()
        row['feature'] = x_colname
        row['cost_fn'] = p_cost_fn
        row['iv_tab_index'] = count
        row['no_bins'] = sel_iv_tab.shape[0]
        row['IV'] = sel_iv_tab['IV'].sum()
        
        
        iv_tab_list.append(sel_iv_tab) # add the iv_tab to sol
        sol_rows.append(row)
    sol_df = pd.DataFrame(sol_rows)
    return sol_df,iv_tab_list 


if __name__ == "__main__":
    #__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
  
    df_train = pd.read_csv("FICO_RiskData_train.csv")
    df_test = pd.read_csv("FICO_RiskData_test.csv")
    y_colname = "Risk_Flag"
    p_good_ind = 1 # 0 or 1 if good
    
    #x_colnames = [i for i in df_train.columns.values if i not in [y_colname]]
    #train
    df_cut_train = df_train[[y_colname,"Loan_Amt"
,"Loan_Amt_Req"
,"LTV"
,"FICO_Score"
,"Prior_Custom_Score"
,"Current_Custom_Score"
,"CB_Age_Oldest_TL"
,"CB_Age_Newest_TL"
,"CB_Avg_Mos_File"
,"CB_Nb_Sat_TL"
,"CB_Nb_60_Plus_TL"
,"CB_Nb_90_Plus_TL"
,"CB_Pct_Sat_TL"
,"CB_Mos_Since_Dlq"
,"CB_Max_Dlq_12_Mos"
,"CB_Max_Dlq_Ever"
,"CB_Nb_Total_TL"
,"CB_Nb_TL_Open_12"
,"CB_Pct_IL_TL"
,"CB_Nb_Inq_6_Mos"
,"CB_Nb_Inq_6_Mos_excl_7_Days"
,"CB_Rev_Util"
,"CB_IL_Util"
,"CB_Nb_Rev_TL_w_Bal"
,"CB_Nb_IL_TL_w_Bal"
,"CB_Nb_Rev_Tl_75_Pct_Limit"
,"CB_Pct_TL_w_Bal"
]]
    #df_cut_train = df_train[[y_colname,"Age"]]
    #df_cut_train = df_train
    sol_df_train,iv_tab_list_train = bin_all(df_cut_train,p_good_ind,y_colname)
    
    #test
    df_cut_test = df_test
    #df_cut_test = pd.DataFrame({'Risk_Flag':df_test['Risk_Flag'],'LTV':df_test['LTV']})
    import cal_iv_tab_for_testset as iv_test_lib
    #copy sol_df
    sol_df_test = sol_df_train[['cost_fn','feature','iv_tab_index']].copy()
    n_bin_list = []
    iv_list = []
    iv_tab_list_test = []
    for i in range(0,sol_df_train.shape[0]):
        iv_tab_train = iv_tab_list_train[sol_df_train['iv_tab_index'][i]]
        feature_name = sol_df_train.iloc[i]["feature"]
        cost_fn_name = str(sol_df_train.iloc[i]["cost_fn"]).split(" ")[1]
        iv_tab_test,iv = iv_test_lib.cal_iv_for_testset(df_cut_test,iv_tab_train,y_colname,feature_name,p_good_ind)
        n_bin = iv_tab_test.shape[0]
        
        n_bin_list.append(n_bin)
        iv_list.append(iv)
        iv_tab_list_test.append(iv_tab_test)
        
        iv_test_lib.save_woe_test(iv_tab_test,feature_name+"_test",cost_fn_name,folder_name='FICO_woe_pic_all')
        
    sol_df_test['IV'] = iv_list
    sol_df_test['no_bins'] = n_bin_list
    
    print(sol_df_train)
    print(iv_tab_list_train)
    print(sol_df_test)
    print(iv_tab_list_test)
    
    #summary_dict_tmp = {"dataset":0,"feature":0,"cost_fn":0,"IV":0,"no_bins":0}
    #save summary
    summary_rows = []
    for i in range(0,sol_df_train.shape[0]):
        iv_tab_train = iv_tab_list_train[sol_df_train['iv_tab_index'][i]]
        iv_tab_test = iv_tab_list_test[sol_df_test['iv_tab_index'][i]]
        feature_name = sol_df_train.iloc[i]["feature"]
        cost_fn_name = str(sol_df_train.iloc[i]["cost_fn"]).split(" ")[1]
        n_bin = iv_tab_train.shape[0]
        
        #summary
        for d in zip(["train","test"],[iv_tab_train,iv_tab_test]):
            row = dict()
            row["dataset"] = d[0]
            row["feature"] = feature_name
            row["cost_fn"] = cost_fn_name
            row["IV"] = d[1].IV.sum()
            row["no_bins"] = n_bin
            
            summary_rows.append(row)
        
        iv_train_mod = iv_tab_train.copy()
        iv_test_mod = iv_tab_test.copy()
        iv_train_mod["dataset"] = "train"
        iv_train_mod["feature"] = feature_name
        iv_train_mod["cost_fn"] = cost_fn_name
        
        iv_test_mod["dataset"] = "test"
        iv_test_mod["feature"] = feature_name
        iv_test_mod["cost_fn"] = cost_fn_name
        
        if i == 0:
            df_bins = iv_train_mod.copy()
        else:
            df_bins = df_bins.append(pd.Series(), ignore_index = True)
            df_bins = df_bins.append(iv_train_mod)
            
        df_bins = df_bins.append(pd.Series(), ignore_index = True)
        df_bins = df_bins.append(iv_test_mod)
        
    df_summary = pd.DataFrame(summary_rows)
    
    df_bins.to_csv("df_bins.csv",index=False)
    df_summary.to_csv("df_summary.csv",index=False)