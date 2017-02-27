'''WORKSPACE'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dataprep as dp
import patsy
import gc
import rpy2.robjects as ro
import time
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.metrics import r2_score
from tqdm import tqdm
from importlib import reload

data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data

# load and format energy consumption
load_raw=dp.load(data_dir+'household_power_consumption.csv') # load data
load_raw=dp.cut(load_raw) # remove leading & trailing Nans
targets=dp.m2h(load_raw,nan='keep') # minutes to hours, preserving nans
targets.fillna(method='bfill',inplace=True) # fill nans withprevious values

train,test=dp.split_train_test(data=targets, test_size=0.25, base=7) # split into train & test sets
dp.save(data=train,path=data_dir+'train.csv') # save train set
dp.save(data=test,path=data_dir+'test.csv') # save test set

# downloading weather in parts due to the limit on API requests
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
atts_w=dp.split_cols(weather) # dictionary of dataframes each containing only one weather attribute values

for name,att in atts_w.items():
	train_att,test_att=dp.split_train_test(data=att, test_size=0.25, base=7) # split into train & test sets
	dp.save(data=train_att,path=data_dir+'train_'+name+'.csv') # save train set
	dp.save(data=test_att,path=data_dir+'test_'+name+'.csv') # save test set



