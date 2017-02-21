'''ARIMA MODELS'''

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
def arima(train,test,hor=24,batch=28,freq=24):
	pandas2ri.activate() # activate connection
	forecast=importr('forecast') # forecast package
	ts=ro.r.ts # R time series
	test_pred=pd.DataFrame(data=None,index=test.index,columns=test.columns) # template structure for dataframe for predictions
	for i in tqdm(range(len(test))): # for each sample in test set
		test_ts=ts(dp.flatten(pd.concat([train,test[:i]])),frequency=freq) # add a new day from test set to the current train set
		if i%batch==0: # # if its time to retrain
			gc.collect() # python does not have direct access to R objects, thus garbage collection does not trigger often enough
			model=forecast.auto_arima(test_ts) # find best model on the current train set
		else: # it is not the time to retrain
			model=forecast.Arima(test_ts,model=model) # do not train, use current model with new observations
		test_pred.iloc[i,:]=pandas2ri.ri2py(forecast.forecast(model,h=hor).rx2('mean')) # predict new values
		tqdm.write('\r')
	pandas2ri.deactivate() # deactivate connection
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
data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data
load_raw=dp.load(data_dir+'household_power_consumption.csv') # load data
load_raw=dp.cut(load_raw) # remove leading & trailing Nans
targets=dp.m2h(load_raw,nan='keep') # minutes to hours, preserving nans
targets.fillna(method='bfill',inplace=True) # fill nans withprevious values

train,test=dp.split_train_test(data=targets, test_size=0.25, base=7)
dp.save(data=train,directory=data_dir,name='train')
dp.save(data=test,directory=data_dir,name='test')

train=pd.read_csv(data_dir+'train.csv',index_col=0)
test=pd.read_csv(data_dir +'test.csv',index_col=0)


dp.split_save(train, nsplits=7, directory=data_dir, name='train')
dp.split_save(test, nsplits=7, directory=data_dir, name='test')

# vertical
test_pred=arima_v(train,test,batch=28,freq=7)
r2_score(y_true=test,y_pred=test_pred,multioutput='uniform_average')
dp.save(data=test_pred,directory=data_dir,name='arima_v')
# vertical week
test_pred=arima_vw(train,test,batch=4,freq=52)
r2_score(y_true=test,y_pred=test_pred,multioutput='uniform_average')
dp.save(data=test_pred,directory=data_dir,name='arima_vw')
# horizontal
test_pred=arima_h(train,test,batch=28,freq=24)
r2_score(y_true=test,y_pred=test_pred,multioutput='uniform_average')
dp.save(data=test_pred,directory=data_dir,name='arima_h')
# horizontal week
test_pred=arima_hw(train,test,batch=4,freq=52)
r2_score(y_true=test,y_pred=test_pred,multioutput='uniform_average')
dp.save(data=test_pred,directory=data_dir,name='arima_hw')





train=dp.split(train,nsplits=7)[1]
test=dp.split(test,nsplits=7)[1]






train.to_csv(path_or_buf='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/load_train.csv',header=True,sep=',',decimal='.')
test.to_csv(path_or_buf='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/load_test.csv',header=True,sep=',',decimal='.')
targets.to_csv(path_or_buf='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/load_proc.csv',header=True,sep=',',decimal='.')

# calls libraries from R to find the best arima model 
def arima_1(train,test,freq=7):
	pandas2ri.activate() # activate interface between python and R 
	forecast=importr('forecast') # forecast package
	ts=ro.r.ts # R time series
	fitted=ro.r('fitted') # function exporting forecasts used while fitting model
	r_train_ts=ts(train,frequency=freq) # construct R's ts object
	r_test_ts=ts(test,frequency=freq) # construct R's ts object
	fit_train=forecast.auto_arima(r_train_ts) # find best model on train test
	fit_test=forecast.Arima(r_test_ts,model=fit_train) # get predictions on test set
	train_pred=pd.Series(pandas2ri.ri2py(fitted(fit_train)),index=train.index) # reconstruct pandas DataFrame from R float vector
	test_pred=pd.Series(pandas2ri.ri2py(fitted(fit_test)),index=test.index) # reconstruct pandas DataFrame from R float vector
	return train_pred,test_pred


def test(n=1000):
	for i in tqdm(range(n)):
		time.sleep(0.1)



def arima_2(train,test,hor=24,batch=7,freq=24):
	pandas2ri.activate()
	forecast=importr('forecast') # forecast package
	ts=ro.r.ts # R time series
	test_pred=pd.DataFrame(data=None,index=test.index,columns=test.columns) # template structure for dataframe for predictions
	for i in tqdm(range(len(test))): # for each sample in test set
		tqdm.write('Step: %i' % i)
		test_ts=ts(dp.flatten(pd.concat([train,test[:i]])),frequency=freq) # add a new day from test set to the current train set
		if i==0: # if  in the first iteration
			model=forecast.auto_arima(test_ts) # find best model on train test (i.e. original model)
			sign=forecast.arimaorder(model) # retrieve the order of this best model
			order=sign[0:3] # AR, differencing & MA orders
			seasonal=sign[3:6] if len(sign)>=6 else ro.IntVector((0,0,0))# seasonal components
		else: # not the first iteration
			if i%batch==0: # if its time to retrain
				gc.collect() # python does not have direct access to R objects, thus garbage collection does not trigger often enough 
				model=forecast.Arima(test_ts,order=order,seasonal=seasonal,method="CSS-ML") # train Arima model on current (expanded) train set using orders found on the original train set
			else: # it is not the time to retrain
				model=forecast.Arima(test_ts,model=model) # do not train, use current model with new observations
		test_pred.iloc[i,:]=pandas2ri.ri2py(forecast.forecast(model,h=hor).rx2('mean')) # predict new values
	return test_pred

