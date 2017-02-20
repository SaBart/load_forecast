'''LONG SHORT-TERM MEMORY'''
import os
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import dataprep as dp
import importlib

# create & train basic NN model
def create_model(n_in=24, n_out=24, n_hidden=10, batch_size=10, activation='relu', loss='mean_squared_error', optimizer='adam'):
	from keras.models import Sequential
	from keras.layers.core import Dense
	from keras.layers import LSTM
	model = Sequential() # FFN
	model.add(LSTM(input_dim=n_in,output_dim=n_hidden,activation=activation)) # input & hidden recurrent layers
	model.add(Dense(n_out)) # output layer
	model.compile(loss=loss, optimizer=optimizer) # assemble network	
	return model


from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

np.random.seed(0) # fix seed for reprodicibility
path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv' # data path
load=dp.load(path) # load data
load_with_nans=load.apply(axis=1,func=(lambda x: np.nan if (x.isnull().sum()>0) else x.mean())).unstack() # custom sum function where any Nan in arguments gives Nan as result
# set grid search parameters and ranges
grid_space={'n_hidden':[10,20,30]}

for i in range(1,2): # optimize for number of time steps
	X,Y=dp.split_X_Y(dp.shift(load_with_nans,n_shifts=i,shift=1).dropna()) # create patterns & targets in a suitable format
	X=dp.order(X) # put timesteps in the correct order starting from the oldest
	X=np.reshape(X.as_matrix(), (X.shape[0],i,X.shape[1]//i)) # reshape patterns to match specific requirements of LSTM
	grid_space['n_in']=[X.shape[2]] # workaround for enabling varying pattern lengths corresponding to the number of time steps
	model=KerasRegressor(build_fn=create_model,verbose=0) # create model template
	grid_setup = GridSearchCV(estimator=model, param_grid=grid_space, cv=TimeSeriesSplit(n_splits=3),n_jobs=1, scoring=make_scorer(r2_score,multioutput='uniform_average'), verbose=10) # set up the grid search	
	grid_result = grid_setup.fit(X, Y.as_matrix()) # find best parameters
	# summarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) # print best parameters
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):	print("%f (%f) with: %r" % (mean, stdev, param)) # print all sets of parameters

plt.plot(grid_result.best_estimator_.predict(X)[0])











