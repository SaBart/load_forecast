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
from itertools import product


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
def opt_imp(data,n_iter=10,methods,measures={'MAE':ms.mae,'RMSE':ms.rmse,'SRMSE':ms.srmse,'SMAPE':ms.smape,'MASE':partial(ms.mase,shift=60*24*7)}):
	dist=out_dist(data) # get the distribution of outage lengths
	data_lno=lno(data) # get the longest no outage (LNO)
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

# converts dictionary to tuples
def d2t(d):
	result=[] # 
	for key,args in d.items():
		if not isinstance(args,dict): result+=[[(key,arg) for arg in args]]
		else:
			for k,a in args.items():
				result+=[[[(key,k)]]+d2t(a)]
	return result
	
# converts tuples to kwargs
def t2k(methods):	
	return [[[{kw:arg for kw,arg in c} for c in comb] for comb in product(*t)] for t in d2t(methods)]
	
	[c for c in [comb for comb in [product(*t) for t in d2t(methods)]]]
	
	[c for c in [comb for comb in product(*t)]]
	
	for t in d2t(methods):
		comb=product(*t)
		for c in comb:print(c)
	
	
# converts dictionary of lists to list of dictionaries (list of all combinations of kwargs)
def dl2ld(dictionary):
	return [{kw:arg for kw,arg in comb} for comb in product(*[[(kw,arg) for arg in args] for kw,args in dictionary.items()])] # all combinations of kwargs

	


random={}
mean={'option':['mean','median','mode']} # params for mean
ma={'weighting':['simple','linear','exponential'],'k':np.arange(2,11)} # params for moving average
locf={'option':['locf','nocb'],'na_remaining':['rev']} # params for last observation carry forward
interpol={'option':['linear','spline','stine']} # params for interpolation
kalman={'model':['auto.arima','structTS']}

methods={impts.na_random:random,
		impts.na_mean:mean,
		impts.na_ma:ma,
		impts.na_locf:locf,
		impts.na_interpolation:interpol,
		impts.na_kalman:kalman,
		impts.na_seadec:{'algorithm':{'random':random,'mean':mean,'ma':ma,'locf':locf,'interpolation':interpol,'kalman':kalman}},
		impts.na_seasplit:{'algorithm':{'random':random,'mean':mean,'ma':ma,'locf':locf,'interpolation':interpol,'kalman':kalman}},
	}

for method,params in methods.items():
	for kwargs in [{kw:arg for kw,arg in comb} for comb in product(*[[(kw,arg) for arg in args] for kw,args in params.items()])]: # for all combinations of kwargs
		print(kwargs)
		#data_imp=imp(data=data_out,method=method,**kwargs)
	




for kwargs in [{kw:arg for kw,arg in comb}for comb in product(*[[(kw,arg) for arg in args] for kw,args in params.items()])]: print(kwargs)
	
	
[[[(kw,k) for k,a in dict.items()] for dict in args] for kw,args in params.items()]
	
	# TODO: make [**kwargs] for all combination in params
	kwargs=[{kw:arg for kw,arg in params.items()}]
	{ for in product(params.values())}
	print(kwargs)
	# if element of params is dictionary, extract & combine contents
	# method(params)

for tuple in product(*[[(kw,arg) for arg in args] for kw,args in params.items()]):
		d={k:a for k,a in tuple}
		print(d)
	
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


a = ('2',)
b = 'z'
(b,)+a


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

	if isinstance(obj,dict):
		result+=[[(key,)+a for a in o2t(args)] for key,args in obj.items()]
	for el in obj:
		if not isinstance(el, dict): result+=[(el,)]
		else: result+=[[(key,)+a for a in o2t(args)] for key,args in obj.items()]
	return result
	
	if not isinstance(obj,dict): return [(obj,)]
	else: return [[[(kw,)+a for a in o2t(arg)] for arg in args] for kw,args in obj.items()]

