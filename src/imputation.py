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
from functools import partial


# impute missing values using R
def imp(data,method,**kwargs):
	pandas2ri.activate() # activate connection
	impts=importr('imputeTS') # package for time series imputation
	result=pandas2ri.ri2py(method(ro.FloatVector(data.values),kwargs)) # get results of imputation from R
	data=pd.Series(index=data.index,data=np.reshape(result,newshape=data.shape, order='C')) # construct DataFrame using original index and columns
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
def opt_imp(data,n_iter=10,methods=['locf','nocb','interpol_lin','interpol_spline','interpol_stine','seadec_iterpol','seadec_locf','seadec_mean','seadec_random','seadec_kalman','seadec_ma','kalman_arima','kalman_structTS'],measures={'MAE':ms.mae,'RMSE':ms.rmse,'SRMSE':ms.srmse,'SMAPE':ms.smape,'MASE':partial(ms.mase,shift=60*24*7)}):
	#data=dp.d2s(data) # flatten dataframe
	dist=out_dist(data) # get the distribution of outage lengths
	data_lno=lno(data) # get the longest no outage (LNO)
	#data_lno=dp.cut(data_lno) # # remove incomplete first and last days
	results=list() # initialize empty list for results
	for i in range(n_iter):
		data_out=add_out(data=data_lno,dist=dist) # add outages
		result=pd.DataFrame() # empty dataframe for scores
		for method in methods: # for each method under consideration	
			data_imp=imp(data=data_out,method=method) # impute data with said methods
			score=ms.acc(pred=data_imp,true=data_lno,label=method,measures=measures) # compute accuracy measures
			result=pd.concat([result,score]) # append computed measures
		results.append(result) # add to results
	return sum(results)/n_iter


impts=importr('imputeTS') # package for time series imputation
params={'random':{'method':impts.na_random},
		'mean':{'method':impts.na_mean},
		'ma':{'method':impts.na_ma},
		'locf':{'method':impts.na_locf},
		'interpol':{'method':impts.na_interpolation},
		'seadec':{'method':impts.na_seadec},
		'seasplit':{'method':impts.na_seasplit},
		'kalman':{'method':impts.na_kalman}
	}	


mean={'option':['mean','median','mode']} # params for mean
ma={'weighting':['simple','linear','exponential'],'k':np.arange(2,11)} # params for moving average
locf={'option':['locf','nocb'],'na_remaining':'rev'} # params for last observation carry forward
interpol={'option':['linear','spline','stine']} # params for interpolation
kalman={'model':['auto.arima','structTS']}

methods={impts.na_random:{},
		impts.na_mean:mean,
		impts.na_ma:ma,
		impts.na_locf:locf,
		impts.na_interpolation:interpol,
		impts.na_kalman:kalman,
		impts.na_seadec:{'algorithm':['random',{'mean':mean},{'ma':ma},{'locf':locf},{'interpolation':interpol},{'kalman':kalman}]},
		impts.na_seasplit:{'algorithm':['random',{'mean':mean},{'ma':ma},{'locf':locf},{'interpolation':interpol},{'kalman':kalman}]},
	}

for method,params in methods.items():
	# TODO: make [**kwargs] for all combination in params
	# if element of params is dictionary, extract & combine contents
	method(params)


if method=='random':
		result=pandas2ri.ri2py(impts.na_random(ro.FloatVector(data.values)) # get results of imputation from R
	if method=='mean':
		result=pandas2ri.ri2py(impts.na_mean(ro.FloatVector(data.values),option='mean')) # get results of imputation from R
	if method=='median':
		result=pandas2ri.ri2py(impts.na_mean(ro.FloatVector(data.values),option='median')) # get results of imputation from R
	if method=='mode':
		result=pandas2ri.ri2py(impts.na_mean(ro.FloatVector(data.values),option='mode')) # get results of imputation from R
	if method=='ma_simple':
		result=pandas2ri.ri2py(impts.na_ma(ro.FloatVector(data.values),weighting='simple')) # get results of imputation from R
	if method=='ma_lin':
		result=pandas2ri.ri2py(impts.na_ma(ro.FloatVector(data.values),weighting='linear')) # get results of imputation from R
	if method=='ma_exp':
		result=pandas2ri.ri2py(impts.na_ma(ro.FloatVector(data.values),weighting='exponential')) # get results of imputation from R
	if method=='locf':
		result=pandas2ri.ri2py(impts.na_locf(ro.FloatVector(data.values),option='locf',na_remaining='rev')) # get results of imputation from R
	if method=='nocb':
		result=pandas2ri.ri2py(impts.na_locf(ro.FloatVector(data.values),option='nocb',na_remaining='rev')) # get results of imputation from R
	if method=='interpol_lin':
		result=pandas2ri.ri2py(impts.na_interpolation(ro.FloatVector(data.values),option='linear') # get results of imputation from R
	if method=='interpol_spline':
		result=pandas2ri.ri2py(impts.na_interpolation(ro.FloatVector(data.values),option='spline') # get results of imputation from R
	if method=='interpol_stine':
		result=pandas2ri.ri2py(impts.na_interpolation(ro.FloatVector(data.values),option='stine') # get results of imputation from R
	if method=='seadec_interpol':
		result=pandas2ri.ri2py(impts.na_seadec(ro.FloatVector(data.values),algorithm='interpolation')) # get results of imputation from R
	if method=='seadec_locf':
		result=pandas2ri.ri2py(impts.na_seadec(ro.FloatVector(data.values),algorithm='locf')) # get results of imputation from R
	if method=='seadec_mean':
		result=pandas2ri.ri2py(impts.na_seadec(ro.FloatVector(data.values),algorithm='mean',option='mean')) # get results of imputation from R
	if method=='seadec_median':
		result=pandas2ri.ri2py(impts.na_seadec(ro.FloatVector(data.values),algorithm='mean',option='median')) # get results of imputation from R
	if method=='seadec_mode':
		result=pandas2ri.ri2py(impts.na_seadec(ro.FloatVector(data.values),algorithm='mean',option='mode')) # get results of imputation from R
	if method=='seadec_random':
		result=pandas2ri.ri2py(impts.na_seadec(ro.FloatVector(data.values),algorithm='random')) # get results of imputation from R
	if method=='seadec_kalman':
		result=pandas2ri.ri2py(impts.na_seadec(ro.FloatVector(data.values),algorithm='kalman')) # get results of imputation from R
	if method=='seadec_ma':
		result=pandas2ri.ri2py(impts.na_seadec(ro.FloatVector(data.values),algorithm='ma')) # get results of imputation from R
	if method=='kalman_arima':
		result=pandas2ri.ri2py(impts.na_kalman(ro.FloatVector(data.values),model='auto.arima')) # get results of imputation from R
	if method=='kalman_structTS':
		result=pandas2ri.ri2py(impts.na_kalman(ro.FloatVector(data.values),model='structTS')) # get results of imputation from R
	if method=='seasplit_interpol':
		result=pandas2ri.ri2py(impts.na_seasplit(ro.FloatVector(data.values),algorithm='interpolation')) # get results of imputation from R
	if method=='seasplit_locf':
		result=pandas2ri.ri2py(impts.na_seasplit(ro.FloatVector(data.values),algorithm='locf')) # get results of imputation from R
	if method=='seasplit_mean':
		result=pandas2ri.ri2py(impts.na_seasplit(ro.FloatVector(data.values),algorithm='mean',option='mean')) # get results of imputation from R
	if method=='seasplit_median':
		result=pandas2ri.ri2py(impts.na_seasplit(ro.FloatVector(data.values),algorithm='mean',option='median')) # get results of imputation from R
	if method=='seasplit_mode':
		result=pandas2ri.ri2py(impts.na_seasplit(ro.FloatVector(data.values),algorithm='mean',option='mode')) # get results of imputation from R
	if method=='seasplit_random':
		result=pandas2ri.ri2py(impts.na_seasplit(ro.FloatVector(data.values),algorithm='random')) # get results of imputation from R
	if method=='seasplit_kalman':
		result=pandas2ri.ri2py(impts.na_seasplit(ro.FloatVector(data.values),algorithm='kalman')) # get results of imputation from R
	if method=='seasplit_ma':
		result=pandas2ri.ri2py(impts.na_seasplit(ro.FloatVector(data.values),algorithm='ma')) # get results of imputation from R
	
	
	
	data=pd.Series(index=data.index,data=np.reshape(result,newshape=data.shape, order='C')) # construct DataFrame using original index and columns
	return data



