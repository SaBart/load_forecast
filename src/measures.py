'''MEASURES'''

import os as os
import re as re
import numpy as np
import matplotlib.pyplot as plt
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
def mase(pred,true,shift=7*24):
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
		measures={'SMAE':smae,'RMSE':rmse,'SRMSE':srmse,'SMAPE':smape,'MASE':partial(mase,shift=s2)} # measures to consider	
		score=acc(pred=data.shift(s1), true=data,label='pred:{},true:{}'.format(s1,s2),measures=measures) # compute accuracy measures
		results=pd.concat([results,score]) # append computed measures
	return results

# returns various measures/metrics/scores of goodness of fit 
def acc(pred,true,label='test',measures={'SMAE':smae,'RMSE':rmse,'SRMSE':srmse,'SMAPE':smape,'MASE':partial(mase,shift=7*48)}):
	pred=no_neg(dp.d2s(pred)) # DataFrame to Series & replace negative values with zeros
	true=dp.d2s(true) # DataFrame to Series
	score={name:ms(pred=pred,true=true) for name,ms in measures.items()}
	results=pd.DataFrame(data=score,index=[label]) # convert dictionary into a dataframe
	return results

# return acc measures for all experiments in a directory
def accs(pred_dir,true,measures={'SMAE':smae,'RMSE':rmse,'SRMSE':srmse,'SMAPE':smape,'MASE':partial(mase,shift=7*48)}):
	preds=[re.match(r'^[^0-9]*$', name).group(0) for name in os.listdir(pred_dir) if re.match(r'^[^0-9]*$', name)] # list of files for acc
	merge_bases={re.split(r'[0-9]',name)[0] for name in os.listdir(pred_dir) if len(re.split(r'[0-9]',name))>1} # find all bases to merge
	for base in merge_bases: # for each base to merge
		paths=[pred_dir +base +str(i) +'.csv' for i in range(7)] # make relevant paths
		pred=dp.load_merge(paths, idx='date', dates=True) # load and merge partitions
		name =re.sub(r'_$', 'w', base)+'.csv' # create name for merged predictions
		dp.save(data=pred, path=pred_dir+name, idx='date') # save merged predictions
		preds+=[name] # add name to the list of files for acc
	result=pd.concat([acc(pred=dp.load(path=pred_dir+name, idx='date', dates=True),true=true,label=re.sub(r'.csv', '', name),measures=measures) for name in preds])
	return result

# computes mean rank according to accuracy measures
def rank(data):
	data['rank']=data.rank(method='dense',ascending=True).min(axis='columns') # add column with mean rank
	data.sort_values(by='rank',inplace=True) # sort by rank
	return data