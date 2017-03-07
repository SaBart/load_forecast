'''SIMPLE NEURAL NETWORK'''

import os
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import dataprep as dp
import importlib
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

# DATA PROCESSING METHODS

# create & train basic NN model
def create_model(n_in=24, n_out=24, n_hidden=10, activation='relu', loss='mean_squared_error', optimizer='adam'):
	from keras.models import Sequential
	from keras.layers.core import Dense
	model = Sequential() # FFN
	model.add(Dense(n_hidden, input_dim=n_in,activation=activation)) # input & hidden layers
	#model.add(Dropout({{uniform(0, 1)}})) # randomly set a number of inputs to 0 to prevent overfitting
	model.add(Dense(n_out)) # output layer
	model.compile(loss=loss, optimizer=optimizer) # assemble network	
	return model


np.random.seed(0) # fix seed for reprodicibility
path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv' # data path
load=dp.load(path) # load data
load_with_nans=load.apply(axis=1,func=(lambda x: np.nan if (x.isnull().sum()>0) else x.mean())).unstack() # custom sum function where any Nan in arguments gives Nan as result
# set grid search parameters and ranges
grid_space={'n_hidden':[10,20,30],
			'nb_epoch':[500,1000,1500,2000],
			'batch_size':[1,5,10,20]
		}

for i in range(1,6): # optimize for number of time steps
	X,Y=dp.split_X_Y(dp.shift(load_with_nans,n_shifts=i,shift=1).dropna()) # create patterns & targets in the correct format
	X=dp.order(X) # put timesteps in the correct order starting from the oldest
	grid_space['n_in']=[X.shape[1]] # workaround for enabling varying pattern lengths corresponding to the number of time steps
	model=KerasRegressor(build_fn=create_model,verbose=0) # create model template
	grid_setup = GridSearchCV(estimator=model, param_grid=grid_space, cv=TimeSeriesSplit(n_splits=3),n_jobs=1, scoring=make_scorer(r2_score,multioutput='uniform_average'), verbose=10) # set up the grid search
	grid_result = grid_setup.fit(X.as_matrix(), Y.as_matrix()) # fit best parameters
	# summarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) # print best parameters	means = grid_result.cv_results_['mean_test_score']	stds = grid_result.cv_results_['std_test_score']	params = grid_result.cv_results_['params']	for mean, stdev, param in zip(means, stds, params):	print("%f (%f) with: %r" % (mean, stdev, param)) # print all sets of parameters

plt.plot(grid_result.best_estimator_.predict(X.as_matrix())[0])
