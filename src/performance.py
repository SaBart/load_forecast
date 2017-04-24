'''PERFORMANCE MEASURES'''

import os as os
import re as re
import numpy as np
import pandas as pd
import patsy
import dataprep as dp
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.metrics import r2_score
from functools import partial


# removes all negative values from dataframe
def no_neg(data):
	data[data<0]=0
	return data

# root mean square error (RMSE)
def rmse(pred,true):
	pred=dp.d2s(pred) # DataFrame to Series
	true=dp.d2s(true) # DataFrame to Series
	return np.sqrt(((pred-true) ** 2).mean())

# scaled root mean square error (SRMSE)
def srmse(pred,true):
	pred=dp.d2s(pred) # DataFrame to Series
	true=dp.d2s(true) # DataFrame to Series
	return np.sqrt(((pred-true) ** 2).mean())/true.mean()

# symmetric mean absolute percentage error (SMAPE)
def smape(pred,true):
	pred=dp.d2s(pred) # DataFrame to Series
	true=dp.d2s(true) # DataFrame to Series
	return ((pred-true).abs()/(true.abs()+pred.abs())).mean()

# mean absolute scaled error (MASE)
def mase(pred,true,shift=7*48):
	pred=dp.d2s(pred) # DataFrame to Series
	true=dp.d2s(true) # DataFrame to Series
	return (pred-true).abs().mean()/(true.shift(shift)-true).dropna().abs().mean()

# scaled maximum absolute error (MAE)
def smae(pred,true):
	pred=dp.d2s(pred) # DataFrame to Series
	true=dp.d2s(true) # DataFrame to Series
	return ((pred-true).abs().max())/true.mean()
	
# finds the best shift to use for naive method and MASE
def opt_shift(data, shifts=[48,48*7]):
	data=dp.d2s(data) # DataFrame to Series
	results=pd.DataFrame() # empty dataframe for scores
	for s1,s2 in [(s1,s2) for s1 in shifts for s2 in shifts]: # for each shift to consider
		if s1==s2: continue # skip if shifts are equal
		measures={'SMAE':smae,'SRMSE':srmse,'SMAPE':smape,'MASE':partial(mase,shift=s2)} # measures to consider	
		score=ev(pred=data.shift(s1), true=data,label='pred:{},true:{}'.format(s1,s2),measures=measures) # compute accuracy measures
		results=pd.concat([results,score]) # append computed measures
	return results

# returns various measures/metrics/scores of goodness of fit 
def ev(pred,true,label='test',measures={'SMAE':smae,'RMSE':rmse,'SRMSE':srmse,'SMAPE':smape,'MASE':partial(mase,shift=7*48)}):
	pred=no_neg(dp.d2s(pred)) # DataFrame to Series & replace negative values with zeros
	true=dp.d2s(true) # DataFrame to Series
	score={name:ms(pred=pred,true=true) for name,ms in measures.items()}
	results=pd.DataFrame(data=score,index=[label]) # convert dictionary into a dataframe
	return results

# return performance measures for all experiments in a directory
def ev_dir(pred_dir,true,measures={'SMAE':smae,'RMSE':rmse,'SRMSE':srmse,'SMAPE':smape,'MASE':partial(mase,shift=7*48)}):
	preds=[re.match(r'^[^0-9]*$', name).group(0) for name in os.listdir(pred_dir) if re.match(r'^[^0-9]*$', name)] # list of files for performance evaluation
	merge_bases={re.split(r'[0-9]',name)[0] for name in os.listdir(pred_dir) if len(re.split(r'[0-9]',name))>1} # find all bases to merge
	for base in merge_bases: # for each base to merge
		name =re.sub(r'_$', 'w', base)+'.csv' # create name for merged predictions
		if name in preds: continue
		paths=[pred_dir +base +str(i) +'.csv' for i in range(7)] # make relevant paths
		pred=dp.load_merge(paths, idx='date', dates=True) # load and merge partitions
		dp.save(data=pred, path=pred_dir+name, idx='date') # save merged predictions
		preds+=[name] # add name to the list of files for acc
	result=pd.concat([ev(pred=dp.load(path=pred_dir+name, idx='date', dates=True),true=true,label=re.sub(r'.csv', '', name),measures=measures) for name in preds])
	return result

def ev_day(pred,true,measures={'SMAE':smae,'RMSE':rmse,'SRMSE':srmse,'SMAPE':smape,'MASE':partial(mase,shift=7*48)}):
	result=pd.DataFrame() # empty dataframe for performance
	for i in range(1,len(pred)+1): # for each day
		if i-8<0: # i too small for MASE
			l=0 # lower bound
			s=i-1 # shift
		else: # MASE possible
			l=i-8 # include whole week
			s=7 # default shift
		pred_day=pred.iloc[l:i,].shift(-s).shift(s) # we only want last day
		true_day=true.iloc[l:i,] # whole week + one day because of shifts in MASE
		pfm=ev(pred=pred_day,true=true_day,label=pred.index[i-1],measures=measures) # evaluate performance
		result=pd.concat([result,pfm]) # add results for this day	
	return result
		
# computes min & mean rank according to performance measures
def rank(data):
	sum_perf=data.sum(axis='columns') # sum performance measures
	data['rank']=data.rank(method='dense',ascending=True).min(axis='columns') # add column with mean rank
	data['mean_rank']=data.rank(method='dense',ascending=True).mean(axis='columns') # add column with mean rank
	data['sum']=sum_perf # add sum column
	data.sort_values(by=['rank','mean_rank','sum'],inplace=True) # sort by rank
	data=data.drop(['mean_rank','sum'],axis='columns') # remove unnecessary columns
	return data