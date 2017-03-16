'''MEASURES'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import patsy
import dataprep as dp
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.metrics import r2_score

# root mean square error (RMSE)
def rmse(pred,true):
	pred=dp.d2s(pred) # DataFrame to Series
	true=dp.d2s(true) # DataFrame to Series
	return np.sqrt(((pred-true) ** 2).mean())

# symmetric mean absolute percentage error
def smape(pred,true):
	pred=dp.d2s(pred) # DataFrame to Series
	true=dp.d2s(true) # DataFrame to Series
	return ((true-pred).abs()/(true.abs()+pred.abs())).mean()

def mase(pred,true,shift=7*24):
	pred=dp.d2s(pred) # DataFrame to Series
	true=dp.d2s(true) # DataFrame to Series
	return ((pred-true)/(true.shift(shift)-true).dropna().abs().mean()).abs().mean()

# finds the best shift to use for naive method and MASE
def opt_shift(data, shifts=[60*24,60*24*7]):
	shift=shifts[0] # initial shift
	M=float('inf') # initial MASE
	for s1,s2 in [(s1,s2) for s1 in shifts for s2 in shifts]: # fore ach each shift to consider
		m=mase(pred=data.shift(s1).dropna(), true=data.shift(-s1).dropna(), shift=s2) # compute new MASE with respect to best shift so far
		print(s1,s2,m)
		if m<M: # if new MASE is the best so far
			shift=s1 # save shift
			M=m # save MASE
	return shift

# returns various measures/metrics/scores of goodness of fit 
def acc(pred,true,label='test'):
	score={} # initialize dictionary
	score['RMSE']=rmse(pred=pred,true=true) # compute RMSE
	score['SMAPE']=smape(pred=pred,true=true) # compute SMAPE
	score['MASE']=mase(pred=pred,true=true)
	results=pd.DataFrame(data=score,index=[label]) # convert dictionary into a dataframe
	return results