'''HOLT-WINTER'S EXPONENTIAL SMOOTHING'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dataprep as dp
import importlib
import patsy
import gc
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.metrics import r2_score
from tqdm import tqdm
import time

# calls libraries from R to find the best arima model
def tbats(train,test,hor=24,batch=28,freq=[24,7*24,365.25*24]):
	pandas2ri.activate() # activate connection
	forecast=importr('forecast') # forecast package
	msts=forecast.msts # forecast package multiple seasonality time series
	test_pred=pd.DataFrame(data=None,index=test.index,columns=test.columns) # template structure for dataframe for predictions
	for i in tqdm(range(len(test))): # for each sample in test set
		test_ts=msts(dp.flatten(pd.concat([train,test[:i]])),seasonal_periods=ro.IntVector(freq)) # add a new day from test set to the current train set
		if i%batch==0: # # if its time to retrain
			gc.collect() # python does not have direct access to R objects, thus garbage collection does not trigger often enough
			model=forecast.tbats(test_ts) # find best model on the current train set
		else: # it is not the time to retrain
			model=forecast.Arima(test_ts,model=model) # do not train, use current model with new observations
		test_pred.iloc[i,:]=pandas2ri.ri2py(forecast.forecast(model,h=hor).rx2('mean')) # predict new values
	pandas2ri.deactivate() # close connection
	return test_pred

# searches for the best arima model for horizontal predictions
def arima_h(train,test,batch=28,freq=24):
	return arima(train,test,hor=24,batch=batch,freq=freq) # predict all days with correct horizon

# searches for the best arima model for horizontal predictions for each day of the week separately
def arima_hw(train,test,batch=28,freq=24):
	test_pred=pd.DataFrame(data=None,index=test.index,columns=test.columns) # prepare dataframe template for out of sample prediction on test set	
	for (i,train_day,test_day) in [(i, dp.split(train,nsplits=7)[i], dp.split(test,nsplits=7)[i]) for i in dp.split(train,nsplits=7)]: # for each day
		test_day_pred=arima(train_day,test_day,hor=24,batch=batch,freq=freq) # predict for all hours of the respective day
		test_pred.iloc[i::7,:]=test_day_pred # fill corresponding rows with out of sample predictions
	return test_pred

# searches for the best arima model for vertical predictions for each hour separately
def arima_v(train,test,batch=28,freq=7):
	test_pred=pd.DataFrame(data=None,index=test.index,columns=test.columns) # prepare dataframe template for out of sample prediction on test set
	for col in train: # for each hour
		print(col)
		test_pred[col]=arima(train[col].to_frame(),test[col].to_frame(),hor=1,batch=batch,freq=freq) # fill corresponding column with predictions
	return test_pred

# searches for the best arima model for vertical predictions for each hour & day of the week 
def arima_vw(train,test,batch=28,freq=52): # assume yearly seasonality, i.e. 52 weeks
	test_pred=pd.DataFrame(data=None,index=test.index,columns=test.columns) # prepare dataframe template for out of sample prediction on test set	
	for (i,train_day,test_day) in [(i, dp.split(train,nsplits=7)[i], dp.split(test,nsplits=7)[i]) for i in dp.split(train,nsplits=7)]: # for each day
		test_day_pred=arima_v(train_day,test_day,batch=batch,freq=freq) # predict for all hours of the respective day
		test_pred.iloc[i::7,:]=test_day_pred # fill corresponding rows with out of sample predictions
	return test_pred
	
def arima_all(train,test,nsplits,arima,**kwargs):
	test_pred=pd.DataFrame(index=test.index,columns=test.columns)
	chunk_size=dp.flr(len(test)/nsplits,base=7) # compute size of one chunk, round for complete weeks
	for i,test_chunk in test.groupby(np.arange(len(test))//chunk_size): # for each partition of test set
		test_pred=pd.concat([test_pred,arima(train,test_chunk,**kwargs)]) # append new predictions
		train=pd.concat([train,test_chunk]) # add current test partition to train set
	return test_pred	
	

np.random.seed(0) # fix seed for reprodicibility
path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv' # data path
load_raw=dp.load(path) # load data
load_raw=dp.cut(load_raw) # remove leading & trailing Nans
targets=dp.m2h(load_raw,nan='keep') # minutes to hours, preserving nans
targets.fillna(method='bfill',inplace=True) # fill nans withprevious values

train,test=dp.split_train_test(data=targets, test_size=0.25, base=7)
dp.save(data=train,path='C:/Users/SABA/Google Drive/mtsg/data/train.csv')
dp.save(data=test,path='C:/Users/SABA/Google Drive/mtsg/data/test.csv')


# vertical
test_pred=arima_v(train,test,batch=28,freq=7)
r2_score(y_true=test,y_pred=test_pred,multioutput='uniform_average')
dp.save(data=test_pred,path='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_v.csv')
# vertical week
test_pred=arima_vw(train,test,batch=28,freq=52)
r2_score(y_true=test,y_pred=test_pred,multioutput='uniform_average')
dp.save(data=test_pred,path='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_vw.csv')
# horizontal
test_pred=arima(train,test,hor=24,batch=28,freq=24)
r2_score(y_true=test,y_pred=test_pred,multioutput='uniform_average')
dp.save(data=test_pred,path='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_h.csv')
# horizontal week
test_pred=arima_hw(train,test,batch=28,freq=52)
r2_score(y_true=test,y_pred=test_pred,multioutput='uniform_average')
dp.save(data=test_pred,path='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_hw.csv')


