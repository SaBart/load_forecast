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

# CONSTANTS
data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data 
exp_dir='C:/Users/SABA/Google Drive/mtsg/data/experiments/' # directory containing results of experiments
wip_dir='C:/Users/SABA/Google Drive/mtsg/data/wip/' # work in progress directory

# load data
data=dp.load_lp(data_dir+'household_power_consumption.csv') # load data
dp.save(data,path=data_dir+'data.csv') # save processed data
data=dp.load(path=data_dir+'data.csv', index='date')
# data=dp.cut(data) # remove incomplete first and last days

# visualize data
dv.nan_hist(data) # histogram of nans
dv.nan_bar(data) # bar chart of nans
dv.nan_heat(data) # heatmap of nans

# fill missiong values
ms.opt_shift(data,shifts=[60*24,60*24*7]) # find the best shift for naive predictor for MASE
shift=60*24*7# the shift that performed best
measures={'MAE':ms.mae,'RMSE':ms.rmse,'SRMSE':ms.srmse,'SMAPE':ms.smape,'MASE':partial(ms.mase,shift=shift)} # measures to consider

random={} # params for random
mean={'option':['mean','median','mode']} # params for mean
ma={'weighting':['simple','linear','exponential'],'k':np.arange(2,11)} # params for moving average
locf={'option':['locf','nocb'],'na_remaining':['rev']} # params for last observation carry forward
interpol={'option':['linear','spline','stine']} # params for interpolation
kalman={'model':['auto.arima','StructTS']}
dec_split=[{**{'algorithm':['random']},**random},
		{**{'algorithm':['mean']},**mean},
		{**{'algorithm':['ma']},**ma},
		{**{'algorithm':['locf']},**locf},
		{**{'algorithm':['interpolation']},**interpol},
		{**{'algorithm':['kalman']},**kalman}]

impts=importr('imputeTS') # package for time series imputation

# simple methods to use for imputation
methods=[{'name':'random','alg':impts.na_random,'opt':random},
		{'name':'mean','alg':impts.na_mean,'opt':mean},
		{'name':'ma','alg':impts.na_ma,'opt':ma},
		{'name':'locf','alg':impts.na_locf,'opt':locf},
		{'name':'interpol','alg':impts.na_interpolation,'opt':interpol},
		{'name':'kalman','alg':impts.na_kalman,'opt':kalman}]
# add more complex methods
methods+=[{'name':'seadec','alg':impts.na_seadec,'opt':opt} for opt in dec_split]+[{'name':'seasplit','alg':impts.na_seasplit,'opt':opt} for opt in dec_split]
		
imp_res=imp.opt_imp(data,methods=methods, n_iter=10,measures=measures) # test for best imputation method




data=dp.m2h(data) # minutes to hours

# prepate train & test sets
train,test=dp.train_test(data=data, test_size=0.25, base=7) # split into train & test sets
dp.save(data=train,path=wip_dir+'train.csv') # save train set
dp.save(data=test,path=wip_dir+'test.csv') # save test set
dp.save_dict(dic=dp.split(train,nsplits=7),path=wip_dir+'train_') # split train set according to weekdays and save each into a separate file
dp.save_dict(dic=dp.split(test,nsplits=7),path=wip_dir+'test_') # split test set according to weekdays and save each into a separate file

# downloading weather in parts due to the limit on API requests (only 500 per day) 
dates=pd.DatetimeIndex(targets.index).strftime('%Y%m%d')[:400] # first part of dates
dp.dl_save_w(dates, data_dir+'weather_1.csv') # save first part
dates=pd.DatetimeIndex(targets.index).strftime('%Y%m%d')[400:800] # second part of dates
dp.dl_save_w(dates, data_dir+'weather_2.csv') # save second part
dates=pd.DatetimeIndex(targets.index).strftime('%Y%m%d')[800:1200] # third part of dates
dp.dl_save_w(dates, data_dir+'weather_3.csv') # save third part
dates=pd.DatetimeIndex(targets.index).strftime('%Y%m%d')[1200:] # fourth part of dates
dp.dl_save_w(dates, data_dir+'weather_4.csv') # save fourth part

# formatting. splitting and saving weather data
paths=['weather_1.csv','weather_2.csv','weather_3.csv','weather_4.csv'] # files to be concatenated
weather=dp.load_concat_w([data_dir+path for path in paths],index='timestamp',cols=['tempm','hum','pressurem','wspdm']) # join all parts of weather data
weather.fillna(method='bfill',inplace=True) # fill missiong values
weather_split=dp.split_cols(weather) # dictionary of dataframes each containing only one weather attribute values
for name,data in weather_split.items():
	train_w,test_w=dp.train_test(data=data, test_size=0.25, base=7) # split into train & test sets
	dp.save(data=train_w,path=wip_dir+'train_'+name+'.csv') # save train set
	dp.save(data=test_w,path=wip_dir+'test_'+name+'.csv') # save test set
	dp.save_dict(dic=dp.split(train_w,nsplits=7), path=wip_dir+'train_'+name+'_') # split train set according to weekdays and save each into a separate file
	dp.save_dict(dic=dp.split(test_w,nsplits=7), path=wip_dir+'test_'+name+'_') # split test set according to weekdays and save each into a separate file


paths=[wip_dir + path for path in ['test_0.csv','test_1.csv','test_2.csv','test_3.csv','test_4.csv','test_5.csv','test_6.csv']]
data=dp.load_merge(paths,index='date')
test=dp.load(wip_dir+'test.csv',index='date')






