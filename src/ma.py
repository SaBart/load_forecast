'''MOVING AVERAGE'''

import os
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import dataprep as dp
import patsy
import importlib
from sklearn.metrics import r2_score
from unittest.mock import inplace
from sklearn import multioutput

np.random.seed(0) # fix seed for reprodicibility
path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv' # data path
load_raw=dp.load(path) # load data
targets=load_raw.apply(axis=1,func=(lambda x: np.nan if (x.isnull().sum()>0) else x.mean())).unstack() # custom sum function where any Nan in arguments gives Nan as result

# moving average
for i in range(1,50):
	pred=targets.rolling(window=i).mean().shift(1)
	load=pd.concat({'pred':pred,'targets':targets},axis=1)
	load.dropna(inplace=True)
	print(r2_score(y_pred=load['pred'],y_true=load['targets'],multioutput='uniform_average'))

# moving average for separated days
for w in range(1,50): # optimise window size
	pred=pd.DataFrame(data=None,index=targets.index,columns=targets.columns) # initialize predictions to Nans
	for (i,day) in dp.split(data=targets, nsplits=2).items(): # for each day
		pred.iloc[i::2,:]=day.rolling(window=w).mean().shift(1) # assign predictions to corresponding rows, shift to exclude current day
	load=pd.concat({'pred':pred,'targets':targets},axis=1) # join targets and predictions into one dataset
	load.dropna(inplace=True) # drop a couple of rows with Nans at the beginning produced by moving average
	print(r2_score(y_pred=load['pred'],y_true=load['targets'],multioutput='uniform_average'))


