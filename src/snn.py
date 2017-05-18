'''NEURAL NETWORKS'''

import os
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import dataprep as dp
import performance as pf
import importlib
import keras as kr
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from functools import partial
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout

# DATA PROCESSING METHODS

# create & train basic NN model
def nn(n_in=48, n_out=48, n_hid=30,dropout=0.1,hid_act='sigmoid',out_act='softmax',opt='rmsprop'):
	model = Sequential() # FFN
	model.add(Dense(n_hid,input_dim=n_in,activation=hid_act)) # input & hidden layers
	model.add(Dropout(p=dropout))
	model.add(Dense(n_out,activation=out_act)) # output layer
	model.compile(loss='mse', optimizer=opt) # assemble network	
	return model

def ev(train,test,model,prep=None,postp=None):
	pred=pd.DataFrame() # dataframe for predictions
	loss=pd.DataFrame() # dataframe for loss function
	T,args=prep(train)
	V,_=prep(test,args)
	TV=pd.concat([T,V])
	TV=dp.add_lags(data=TV, lags=[1,2,3,4,5,6,7], nolag='targets')
	T=TV[:len(TV)-len(test)]
	V=TV[len(TV)-len(test):]
	T_X,T_Y=dp.X_Y(data=T,Y_lab='targets')
	V_X,V_Y=dp.X_Y(data=V,Y_lab='targets')
	#model.fit(T_X.as_matrix(), T_Y.as_matrix(), nb_epoch=100, batch_size=28,verbose=2,validation_data=(V_X.as_matrix(),V_Y.as_matrix()),callbacks=[stop]) # train neural network
	hist=model.fit(T_X.as_matrix(), T_Y.as_matrix(), nb_epoch=100, batch_size=1,verbose=2,validation_data=(V_X.as_matrix(),V_Y.as_matrix())) # train neural network
	V_pred=pd.DataFrame(model.predict(V_X.as_matrix()),index=V_Y.index,columns=V_Y.columns) # forecasts for the next batch
	#V_pred=dp.z_inv(data=V_pred, mean=mean, std=std) # de-standardize data
	V_pred=postp(V_pred,args) 
	pred=pd.concat([pred,V_pred]) # add new predictions
	new_loss=pd.DataFrame(hist.history) # new loss
	loss=pd.concat([loss,new_loss],axis=0,ignore_index=True) # append to old loss
	perf=pf.ev(pred=pred, true=true, label='nn', measures=measures) # evaluate performance
	return perf,loss

np.random.seed(0) # fix seed for reprodicibility
data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data 
exp_dir='C:/Users/SABA/Google Drive/mtsg/data/experiments/nn/' # directory containing results of experiments

true=dp.load(path=data_dir+'experiments/data/test.csv',idx='date',dates=True) # observations to forecast
measures={'SRMSE':pf.srmse,'MASE':partial(pf.mase,shift=48*7),'SMAPE':pf.smape,'SMAE':pf.smae,} # performance to consider
data=dp.load(path=data_dir+'data_imp.csv',idx='datetime',cols='load',dates=True) # load data
data=dp.resample(data=data, freq=1440) # make columns represent time intervals
train,test=dp.train_test(data=data, base=7, test_size=0.255) # split into train and test sets

batch=28 # batch size for cross validation
stop=kr.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto') # end training when loss on validation starts decreasing
model=nn(n_in=7*48, n_out=48, n_hid=100,dropout=0.5,hid_act='sigmoid',out_act='softmax',opt='rmsprop') # compile neural network

perf,loss=train(train,test,model=model,prep=dp.de_seas,postp=dp.re_seas) # evaluate network


	





batch=28 # batch size for cross validation
stop=kr.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto') # end training when loss on validation starts decreasing
model=nn(n_in=48, n_out=48, n_hid=100,dropout=0.1,hid_act='sigmoid',out_act='softmax',opt='rmsprop') # compile neural network
pred=pd.DataFrame() # dataframe for predictions
loss=pd.DataFrame() # dataframe for loss function
for i in range(0,len(test),batch): # for each batch
	if (len(test)-i)<batch: val_size=len(test)-i # not enough observation for complete batch
	else: val_size=batch # smaller batch 
	T=pd.concat([train,test[:i+val_size]]) # add new batch to train test
	mean,std=dp.mean_std(T[:len(T)-val_size]) # get mean and std only from train set
	T=dp.z_val(data=T,mean=mean,std=std) # standardize data
	T=dp.add_lags(data=T, lags=[7], nolag='targets') # add lags
	V=T[len(T)-val_size:] # build validation set
	T=T[:len(T)-val_size] # build train set
	T_X,T_Y=dp.X_Y(data=T,Y_lab='targets') # create patterns & targets in the correct format
	V_X,V_Y=dp.X_Y(data=V,Y_lab='targets') # patterns for forecasting
	#model.fit(T_X.as_matrix(), T_Y.as_matrix(), nb_epoch=100, batch_size=28,verbose=2,validation_data=(V_X.as_matrix(),V_Y.as_matrix()),callbacks=[stop]) # train neural network
	hist=model.fit(T_X.as_matrix(), T_Y.as_matrix(), nb_epoch=100, batch_size=10,verbose=2,validation_data=(V_X.as_matrix(),V_Y.as_matrix())) # train neural network
	V_pred=pd.DataFrame(model.predict(V_X.as_matrix()),index=V_Y.index,columns=V_Y.columns) # forecasts for the next batch
	V_pred=dp.z_inv(data=V_pred, mean=mean, std=std) # de-standardize data 
	pred=pd.concat([pred,V_pred]) # add new predictions
	new_loss=pd.DataFrame(hist.history) # new loss
	loss=pd.concat([loss,new_loss],axis=0,ignore_index=True) # append to old loss
pf.ev(pred=pred, true=true, label='nn', measures=measures) # evaluate performance
loss.plot()






dp.save(data=pred, path=exp_dir+'nn.csv', idx='date')







data=dp.add_lags(data=data, lags=[1], nolag='targets') # add lagged observations



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



