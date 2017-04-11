'''WORKSPACE'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dataprep as dp
import datavis as dv
import imputation as imp
import measures as ms
import patsy
import gc
import rpy2.robjects as ro
import time
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.metrics import r2_score
from tqdm import tqdm
from importlib import reload
from functools import partial


impts=importr('imputeTS') # package for time series imputation

# CONSTANTS
data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data 
exp_dir='C:/Users/SABA/Google Drive/mtsg/data/experiments/' # directory containing results of experiments
wip_dir='C:/Users/SABA/Google Drive/mtsg/data/wip/' # work in progress directory

# LOAD DATA
#data=dp.load_lp(data_dir+'household_power_consumption.csv') # load data
#dp.save(data,path=data_dir+'data.csv',idx='datetime') # save processed data
data=dp.load(path=data_dir+'data.csv', idx='datetime',cols='load',dates=True)
# data=dp.cut(data) # remove incomplete first and last days

# VISUALIZE DATA
#dv.nan_hist(data) # histogram of nans
#dv.nan_bar(data) # bar chart of nans
#dv.nan_heat(data) # heatmap of nans

# FILLING MISSING VALUES
#ms.opt_shift(data,shifts=[60*24,60*24*7]) # find the best shift for naive predictor for MASE
shift=60*24*7# the shift that performed best
measures={'SMAE':ms.smae,'SRMSE':ms.srmse,'SMAPE':ms.smape,'MASE':partial(ms.mase,shift=shift)} # measures to consider

random={} # params for random
mean={'option':['mean','median','mode']} # params for mean
ma={'weighting':['simple','linear','exponential'],'k':np.arange(2,15)} # params for moving average
locf={'option':['locf','nocb'],'na.remaining':['rev']} # params for last observation carry forward
interpol={'option':['linear','spline','stine']} # params for interpolation
kalman={'model':['auto.arima','StructTS']}
dec_split=[{**{'algorithm':['random']},**random},
		{**{'algorithm':['mean']},**mean},
		{**{'algorithm':['ma']},**ma},
		{**{'algorithm':['locf']},**locf},
		{**{'algorithm':['interpolation']},**interpol},
		{**{'algorithm':['kalman']},**kalman}]



# simple methods to use for imputation
methods=[{'name':'random','alg':impts.na_random,'opt':random},
		{'name':'mean','alg':impts.na_mean,'opt':mean},
		{'name':'ma','alg':impts.na_ma,'opt':ma},
		{'name':'locf','alg':impts.na_locf,'opt':locf},
		{'name':'interpol','alg':impts.na_interpolation,'opt':interpol},
		{'name':'kalman','alg':impts.na_kalman,'opt':kalman}]
# add more complex methods
methods+=[{'name':'seadec','alg':impts.na_seadec,'opt':opt} for opt in dec_split]+[{'name':'seasplit','alg':impts.na_seasplit,'opt':opt} for opt in dec_split]

np.random.seed(0) # fix seed for reprodicibility
imp_res=imp.opt_imp(data,methods=methods, n_iter=10,measures=measures)
dp.save(imp_res,path=data_dir+'imp.csv',idx='datetime') # save results


imp_res=dp.load(path=data_dir+'imp.csv',idx='method')
imp_res=ms.rank(imp_res) # rank data

# impute the whole dataset using three best methods of imputation
data_nocb=imp.imp(data, alg=impts.na_locf, freq=1440, **{'option':'nocb','na.remaining':'rev'})
data_seadec=imp.imp(data, alg=impts.na_seadec, freq=1440, **{'algorithm':'ma','weighting':'simple','k':2})
data_arima=imp.imp(data, alg=impts.na_kalman, freq=1440, **{'model':'auto.arima'}) # arima(5,1,0) is the best model

# save imputed data
dp.save(data_nocb, path=data_dir+'data_nocb.csv', idx='datetime')
dp.save(data_seadec, path=data_dir+'data_seadec.csv', idx='datetime')
dp.save(data_arima, path=data_dir+'data_arima.csv', idx='datetime')


# AGGREGATE DATA & CREATE TRAIN & TEST SETS
temp_dir=data_dir+'nocb/arima/data/' # where to save 	
data=dp.load(path=data_dir+'data_nocb.csv',idx='datetime',cols='load',dates=True) # load imputed data

data=dp.resample(data) # aggregate minutes to half-hours
train,test=dp.train_test(data=data, test_size=0.255, base=7) # split into train & test sets
train=train.tail(364)
dp.save(data=train,path=temp_dir+'/train.csv',idx='date') # save train set
dp.save(data=test,path=temp_dir+'/test.csv',idx='date') # save test set
dp.save_dict(dic=dp.split(train,nsplits=7),path=temp_dir+'/train_',idx='date') # split train set according to weekdays and save each into a separate file
dp.save_dict(dic=dp.split(test,nsplits=7),path=temp_dir+'/test_',idx='date') # split test set according to weekdays and save each into a separate file
	

# WEATHER DATA

# downloading weather in parts due to the limit on API requests (only 500 per day) 
dates=pd.DatetimeIndex(data.index).strftime('%Y%m%d')[:400] # first part of dates
dp.dl_save_w(dates, data_dir+'weather_1.csv') # save first part
dates=pd.DatetimeIndex(data.index).strftime('%Y%m%d')[400:800] # second part of dates
dp.dl_save_w(dates, data_dir+'weather_2.csv') # save second part
dates=pd.DatetimeIndex(data.index).strftime('%Y%m%d')[800:1200] # third part of dates
dp.dl_save_w(dates, data_dir+'weather_3.csv') # save third part
dates=pd.DatetimeIndex(data.index).strftime('%Y%m%d')[1200:] # fourth part of dates
dp.dl_save_w(dates, data_dir+'weather_4.csv') # save fourth part

# formatting weather data
paths=['weather_1.csv','weather_2.csv','weather_3.csv','weather_4.csv'] # files to be concatenated
weather=dp.load_concat_w([data_dir+path for path in paths],idx='timestamp',cols=['tempm','hum','pressurem'],dates=True) # join all parts of weather data
weather.fillna(method='bfill',inplace=True) # fill missiong values (for column with maximum missin it is still only 0.045% of all)

# splitting, aggregating & saving weather datas
temp_dir=data_dir+'nocb/arima/data/' # where to save
for col in weather: # for each column=weather parameter
	data_w=dp.resample(data=weather[col], freq=48) # reshape to have time of day as columns
	train_w,test_w=dp.train_test(data=data_w, test_size=0.255, base=7) # split into train & test sets
	train_w=train_w.tail(364)
	dp.save(data=train_w,path=temp_dir+col+'_train.csv',idx='date') # save train set
	dp.save(data=test_w,path=temp_dir+col+'_test.csv',idx='date') # save test set
	dp.save_dict(dic=dp.split(train_w,nsplits=7),path=temp_dir+col+'_train_',idx='date') # split train set according to weekdays and save each into a separate file
	dp.save_dict(dic=dp.split(test_w,nsplits=7),path=temp_dir+col+'_test_',idx='date') # split test set according to weekdays and save each into a separate file


# EXPERIMENTAL RESULTS
shift=48*7# the shift that performed best
measures={'SMAE':ms.smae,'SRMSE':ms.srmse,'SMAPE':ms.smape,'MASE':partial(ms.mase,shift=shift)} # measures to consider

exp_dir='C:/Users/SABA/Google Drive/mtsg/data/nocb/ets/results/' # directory containing results of experiments
true=dp.load(data_dir+'nocb/ets/data/test.csv',idx='date',dates=True)
results=ms.accs(exp_dir, true, measures=measures)







paths=[wip_dir + path for path in ['test_0.csv','test_1.csv','test_2.csv','test_3.csv','test_4.csv','test_5.csv','test_6.csv']]
data=dp.load_merge(paths,index='date')
test=dp.load(wip_dir+'test.csv',index='date')
