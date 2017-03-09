'''SUPPORT VECTOR REGRESSION'''

import os
import importlib
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import dataprep as dp
from tkinter import *
from tkinter import ttk
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

max_shifts=8
grid_space={'C':[1.0],
			'epsilon':[0.1],
			'kernel':['linear','poly','rbf'],
			'degree':[5],
			'coef0':[0.0],
			'shrinking':[True],
			'max_iter':[-1],
		}

rt = Tk()
rt.geometry('{}x{}'.format(400, 100))
#progress_var = DoubleVar() #here you have ints but when calc. %'s usually floats
theLabel = Label(rt, text="Sample text to show")
theLabel.pack()
pb = ttk.Progressbar(rt, maximum=max_shifts*len(data.columns))
pb.pack(fill=X, expand=1)
	
for n_shifts in range(1,max_shifts): # optimize for number of time steps
	for col in data: # for each hour of the day
		X,Y=dp.split_X_Y(dp.shift(data[col],n_shifts=n_shifts,shift=1,target_label='targets').dropna(),target_label='targets') # create patterns & targets in the correct format
		model=SVR() # create model template
		grid_setup = GridSearchCV(estimator=model, param_grid=grid_space, cv=dp.tscv(total=len(X),base=7, test_size=0.25, batch=28),n_jobs=4, scoring=make_scorer(r2_score,multioutput='uniform_average'), verbose=10) # set up the grid search
		grid_result = grid_setup.fit(X.as_matrix(), Y.as_matrix()) # fit best parameters
		print("BEST: params: %s, score: %f" % (grid_result.best_params_,grid_result.best_score_)) # print best parameters
		means = grid_result.cv_results_['mean_test_score'] # means
		stds = grid_result.cv_results_['std_test_score'] # standard deviations
		params = grid_result.cv_results_['params'] # parameters
		for mean, stdev, param in zip(means, stds, params):	print("params: %s, meas: %f, stdev: %f" % (param, mean, stdev)) # print all results
		pb.step() # update progress bar

# TODO: fit best model using time series cross validation
