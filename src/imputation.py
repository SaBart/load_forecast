'''DATA IMPUTATION'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dataprep as dp
import measures as ms
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from copy import deepcopy


# impute missing values using R
def imp(data,method=''):
	pandas2ri.activate() # activate connection
	impts=importr('imputeTS') # package for time series imputation
	if method=='seadec':
		result=pandas2ri.ri2py(impts.na_seadec(ro.FloatVector(data.values),algorithm='interpolation')) # get results of imputation from R
		data=pd.DataFrame(index=data.index,data=np.reshape(result,newshape=data.shape, order='C')) # construct DataFrame using original index and columns
	if method=='kalman':
		result=pandas2ri.ri2py(impts.na_kalman(ro.FloatVector(data.values),model='auto.arima')) # get results of imputation from R
		data=pd.DataFrame(index=data.index,data=np.reshape(result,newshape=data.shape, order='C')) # construct DataFrame using original index and columns
	return data

# returns the longest no outage (LNO)== longest continuous subset with no nan values
def lno(data):	
	u=l=U=L=0 # initial assignment to local & global bounds
	while u<len(data) and l<len(data): # while bounds are within dataframe
		l=u# set the lower bound to the same position as the uper bound
		while (l<len(data) and data.iloc[l]!=data.iloc[l]):l+=1 # while there is nan shift lower bound
		u=l # set the upper bound to the same position as the lower bound
		while (u<len(data) and data.iloc[u]==data.iloc[u]):u+=1 # while there is not a nan shift upper bound
		if (u-l>=U-L):U,L=u,l # if the interval is the longest so far save the corresponding bounds
	return data[L:U] # return LNO
	
# introduce outages to data according to distribution	
def add_out(data,dist):
	prob=np.random.choice(list(dist.keys()),len(data),p=list(dist.values())) # generate lengths of outages
	while True: # while there is no outage
		data_out=deepcopy(data) # copy dataframe to preserve original values
		i=0 # reset start position
		while i<len(data_out): # iterate and add outages
			l=dp.round_u(prob[i]*len(data_out),base=1) # length of outage
			if l>0: # outage occurred
				data_out[i:i+l]=np.nan # introduce new outage of length l
				i+=l # shift current position to the end of outage interval
			else: i+=1 # no outage, next position
		if data_out.isnull().sum()>0: break
	return data_out
	
# returns the distribution outage (consecutive nans) lengths
def out_dist(data):
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

# returns data imputed with the best method
def opt_imp(data,n_iter=10,methods=['seadec','kalman']):
	data=dp.d2s(data) # flatten dataframe
	dist=out_dist(data) # get the distribution of outage lengths
	data_lno=lno(data) # get the longest no outage (LNO)
	#data_lno=dp.cut(data_lno) # # remove incomplete first and last days
	results=list() # initialize empty list for results
	for i in range(n_iter):
		data_out=add_out(data=data_lno,dist=dist) # add outages
		result=pd.DataFrame() # empty dataframe for scores
		for method in methods: # for each method under consideration	
			data_imp=imp(data=data_out,method=method) # impute data with said methods
			score=ms.acc(true=data_lno,pred=data_imp,label=method) # compute accuracy measures
			result=pd.concat([result,score]) # append computed measures
		results.append(result) # add to results
	return sum(results)/n_iter




