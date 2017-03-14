'''DATA IMPUTATION'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dataprep as dp
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from copy import deepcopy


# impute missing values using R
def impute(data,method=''):
	data=dp.d2s(data) # flatten dataframe inot series
	pandas2ri.activate() # activate connection
	impts=importr('imputeTS') # package for time series imputation
	if method=='seadec':
		result=pandas2ri.ri2py(impts.na_seadec(ro.FloatVector(data.values),algorithm='interpolation')) # get results of imputation from R
		data=pd.DataFrame(index=data.index,columns=data.columns,data=np.reshape(result,newshape=(len(data.index),len(data.columns)), order='C')) # construct DataFrame using original index and columns
	if method=='kalman':
		result=pandas2ri.ri2py(impts.na_kalman(ro.FloatVector(data.values),model='auto.arima')) # get results of imputation from R
		data=pd.DataFrame(index=data.index,columns=data.columns,data=np.reshape(result,newshape=(len(data.index),len(data.columns)), order='C')) # construct DataFrame using original index and columns
	return data

# returns the longest no outage (LNO)== longest continuous subset with no nan values
def lno(data):
	data=dp.d2s(data) # flatten data into a Series	
	u=l=U=L=0 # initial assignment to local & global bounds
	while u<len(data) and l<len(data): # while bounds are within dataframe
		l=u# set the lower bound to the same position as the uper bound
		while (l<len(data) and data.iloc[l]!=data.iloc[l]):l+=1 # while there is nan shift lower bound
		u=l # set the upper bound to the same position as the lower bound
		while (u<len(data) and data.iloc[u]==data.iloc[u]):u+=1 # while there is not a nan shift upper bound
		print(l,u)
		if (u-l>=U-L):U,L=u,l # if the interval is the longest so far save the corresponding bounds
	return data[L:U] # return LNO
	
# introduce outages to data according to distribution	
def add_out(data,dist):
	data=dp.d2s(data) # flatten dataframe
	prob=np.random.choice(list(dist.keys()),len(data),p=list(dist.values())) # generate lengths of outages
	i=0 # start position
	while i<len(data):
		l=dp.round_u(prob[i]*len(data),base=1) # length of outage
		if l>0: # outage occurred
			print('i:{},l:{}'.format(i,l))
			data[i:i+l]=np.nan # introduce new outage of length l
			i+=l # shift current position to the end of outage interval
		else: i+=1 # no outage, next position
	return data
	
# returns the distribution outage (consecutive nans) lengths
def out_dist(data):
	data=dp.d2s(data) # flatten dataframe
	out_cnts={} # dictionry of outage counts
	out=0 # length of outage
	for i in range(len(data)):
		if data.iloc[i]!=data.iloc[i]: # if nan
			out+=1 # increment current number of consecutive nans
		else: 
			if out in out_cnts: out_cnts[out] += 1 # increment dictionary entry
			else: out_cnts[out] = 1 # new entry in dictionary
			out=0 # reset the number of consecutive nans
	if out in out_cnts: out_cnts[out] += 1 # increment dictionary entry
	else: out_cnts[out] = 1 # new entry in dictionary
	out_cnt=sum(out_cnts.values()) # total number of outages (zero length included)
	out_dist={} # dictionary for outage distribution
	for olen,ocnt in out_cnts.items(): # for each entry in outage counts
		out_dist[olen/len(data)]=ocnt/out_cnt # transform key and value into fractions
	return out_dist

# returns various metrics/scores of goodness of fit 
def scores(true,pred,label='test'):
	score={} # initialize dictionary
	score['mse']=
	score['r2']=
	results=pd.DataFrame(data=score,index=[label]) # convert dictionary into a dataframe
	return results

# returns data imputed with the best method
def auto_impute(data,n_iter=10,methods=['seadec','kalman']):
	data_c=deepcopy(data) # copy dataframe
	data_c=dp.d2s(data_c) # flatten dataframe
	out_dist=out_dist(data_c) # get the distribution of outage lengths
	data_lno=lno(data_c) # get the longest no outage (LNO)
	data_lno=dp.cut(data_lno) # # remove incomplete first and last days
	results=list() # initialize empty list for results
	for i in range(n_iter):
		data_out=add_out(data=data_lno,dist=out_dist) # add outages
		while data_out.isnull().sum().sum()>0:data_out=add_out(data=data_lno,dist=out_dist) # add outages until there is at least one
		result=pd.DataFrame() # empty dataframe for scores
		for method in methods: # for each method under consideration	
			data_imp=impute(data=data_out,method=method) # impute data with said methods
			score=scores(true=data_lno,pred=data_imp,label=method)
			result=pd.concat([result,score])
			# score data_imp vs data_lno # check goodness of fit
			# pack results into dataframe
		results.append(result) # add to results
	return sum(results)/n_iter




matrixA={}
matrixA['a']=[0]
matrixA['b']=[0]

pd.DataFrame(matrixA,index=['test'])




