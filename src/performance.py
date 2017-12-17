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
from itertools import combinations



# removes all negative values from dataframe
def no_neg(data):
	data[data<0]=0
	return data

# root mean square error (RMSE)
def rmse(true,pred):
	pred=dp.d2s(pred) # DataFrame to Series
	true=dp.d2s(true) # DataFrame to Series
	return np.sqrt(((true-pred) ** 2).mean())

# scaled root mean square error (SRMSE)
def srmse(true,pred,mean=0): # mean parameter added for day-wise performance evaluation
	pred=dp.d2s(pred) # DataFrame to Series
	true=dp.d2s(true) # DataFrame to Series
	if mean==0:mean=true.mean()
	return np.sqrt(((true-pred) ** 2).mean())/mean

# symmetric mean absolute percentage error (SMAPE)
def smape(true,pred):
	pred=dp.d2s(pred) # DataFrame to Series
	true=dp.d2s(true) # DataFrame to Series
	return ((true-pred).abs()/(true.abs()+pred.abs())).mean()

# mean absolute scaled error (MASE)
def mase(true,pred,shift=7*48):
	pred=dp.d2s(pred) # DataFrame to Series
	true=dp.d2s(true) # DataFrame to Series
	return (true-pred).abs().mean()/(true.shift(shift)-true).dropna().abs().mean()

# s maximum absolute error (MAE)
def mae(true,pred):
	pred=dp.d2s(pred) # DataFrame to Series
	true=dp.d2s(true) # DataFrame to Series
	return ((true-pred).abs().max())

# scaled maximum absolute error (MAE)
def smae(true,pred):
	pred=dp.d2s(pred) # DataFrame to Series
	true=dp.d2s(true) # DataFrame to Series
	return ((true-pred).abs().max())/true.mean()
	
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

# returns various measures/metrics/scores of accuracy
def ev(true,pred,label='test',parse_label=False,measures={'SMAE':smae,'SRMSE':srmse,'SMAPE':smape,'MASE':partial(mase,shift=7*48)}):
	pred=no_neg(pred) # replace negative values with zeros
	score={name:ms(pred=pred,true=true) for name,ms in measures.items()} # compute performance
	if parse_label: # convert label to columns
		for par in re.split(r',',label):
			if '=' in par: # parameter = value
				par,value=(re.split(r'=',par)[0],re.split(r'=',par)[1]) # parse parameters and values
			else: value=True
			if par: score[par]=value # for each parameter in the label create column and mark it True 
	results=pd.DataFrame(data=score,index=[label]) # convert dictionary into a dataframe
	return results

# return performance measures for all experiments in a directory
def ev_dir(pred_dir,true,measures={'SMAE':smae,'SRMSE':srmse,'SMAPE':smape,'MASE':partial(mase,shift=7*48)}):
	if os.listdir(true):
		files=[os.path.splitext(file)[0] for file in os.listdir(pred_dir) if os.path.isfile(true+os.path.splitext(file)[0]+'/test.csv')] # files in both directories 
		result=pd.concat([ev(pred=dp.load(path=pred_dir+file+'.csv', idx='date', dates=True),true=dp.load(path=true+file+'/test.csv',idx='date',dates=True),label=file,parse_label=False,measures=measures) for file in files],axis=0,join='outer') # merge results
	else: result=pd.concat([ev(pred=dp.load(path=pred_dir+name, idx='date', dates=True),true=true,label=re.sub(r',?[^,]*.csv', '', name),parse_label=True,measures=measures) for name in os.listdir(pred_dir)],axis=0,join='outer') # merge results
	result=result.fillna(value=False) # replace nans with False
	return result

# evaluate ensembles, construct all ensembles from files in directory
def ev_ens(pred_dir,true,comb_size=5,measures={'SMAE':smae,'SRMSE':srmse,'SMAPE':smape,'MASE':partial(mase,shift=7*48)}):
	res=pd.DataFrame()
	for size in range(1,comb_size+1): # for each combination size
		for comb in combinations(os.listdir(pred_dir),size): # for each combination of said size
			comb=list(comb) # combination to list
			data=dp.load_merge(fol=pred_dir, paths=list(comb), idx='date', dates=True, axis=1) # merge ensemble members' forecasts into one dataframe
			new_res=ev(pred=data.mean(axis=1),true=true,label='+'.join(data.columns),measures=measures) # average and evaluate forecasts
			print(new_res) # report progress
			res=pd.concat([res,new_res])
	return res

# evaluate performance day-wise
def ev_day(true,pred,measures={'SRMSE':partial(srmse,mean=0)}):
	result=pd.DataFrame() # empty dataframe for performance
	for i in range(0,len(pred)): # for each day
		pred_day=pd.DataFrame(pred.iloc[[i]])
		true_day=pd.DataFrame(true.iloc[[i]])
		pf=ev(pred=pred_day,true=true_day,label=pred.index[i],measures=measures) # evaluate performance
		result=pd.concat([result,pf]) # add results for this day	
	return result


