'''SUPPORT VECTOR REGRESSION'''

import os
import importlib
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import dataprep as dp
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data 
exp_dir='C:/Users/SABA/Google Drive/mtsg/data/experiments/' # directory containing results of experiments
wip_dir='C:/Users/SABA/Google Drive/mtsg/data/wip/' # work in progress directory

data=dp.load_lp(data_dir+'household_power_consumption.csv') # load data
data=dp.cut(data) # remove incomplete first and last days
data=dp.m2h(data) # minutes to hours, preserving nans
data.fillna(method='bfill',inplace=True) # fill nans withprevious values


# TODO: correct grid
grid_space={'n_hidden':[10,20,30],
			'nb_epoch':[500,1000,1500,2000],
			'batch_size':[1,5,10,20]
		}

for i in range(1,8): # optimize for number of time steps
	X,Y=dp.split_X_Y(dp.shift(data,n_shifts=i,shift=1,target_label='targets').dropna(),target_label='targets') # create patterns & targets in the correct format
	grid_space['n_in']=[X.shape[1]] # workaround for enabling varying pattern lengths corresponding to the number of time steps
	model=SVR() # create model template
	grid_setup = GridSearchCV(estimator=model, param_grid=grid_space, cv=dp.tscv(total=len(X),base=7, test_size=0.25, batch=28),n_jobs=1, scoring=make_scorer(r2_score,multioutput='uniform_average'), verbose=10) # set up the grid search
	# TODO: fit best model using time series cross validation
	grid_result = grid_setup.fit(X.as_matrix(), Y.as_matrix()) # fit best parameters
	# summarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) # print best parameters
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):	print("%f (%f) with: %r" % (mean, stdev, param)) # print all sets of parameters
