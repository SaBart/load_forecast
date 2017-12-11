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
from functools import partial
from keras.models import Sequential
from keras.layers.core import Dense
from _operator import concat


# DATA PROCESSING METHODS

# create & train basic NN model
def sln(n_in=48, n_out=48, n_hid=30,hid_act='sigmoid',out_act='linear',opt='adam'):
	model = Sequential() # MLN
	model.add(Dense(n_hid,input_dim=n_in,activation=hid_act,init='glorot_uniform')) # input & hidden layers
	model.add(Dense(n_out,activation=out_act,init='glorot_uniform')) # output layer
	model.compile(loss='mse', optimizer=opt) # assemble network	
	return model

# create & train NN model with 2 hidden layers
def mln(n_in=48, n_out=48, n_hid1=30,n_hid2=30,hid1_act='sigmoid',hid2_act='',out_act='linear',opt='adam'):
	model = Sequential() # FFN
	model.add(Dense(n_hid1,input_dim=n_in,activation=hid1_act,init='glorot_uniform')) # input & hidden layers
	model.add(Dense(n_hid2,activation=hid2_act,init='glorot_uniform')) # input & hidden layers
	model.add(Dense(n_out,activation=out_act,init='glorot_uniform')) # output layer
	model.compile(loss='mse', optimizer=opt) # assemble network	
	return model

# executes one step of cross-validation
def ev(train,test,model,epochs=100,restart=10,batch=50,f_lags=[i for i in range(48)],p_lags=[i+1 for i in range(48)],weather_train=None,weather_test=None,prep=None,postp=None):
	if prep and postp: # whether to do additional preprocessing (and reverse preprocessing) methods
		train,args=prep(train)  # preprocess train set and extract arduments
		test,_=prep(test,args) # preprocess test set using extracted arguments
	T,args_2=dp.de_std(train) # standardise train set
	V,_=dp.de_std(test,args_2) # standardise test set using mean, std from train set
	TV=pd.concat([T,V],axis=0) # concat for adding lagged values
	TV=dp.add_lags(data=TV, f_lags=f_lags,p_lags=p_lags,f_lab='Y',p_lab='X') # add lagged values
	if weather_train and weather_test: # whether to use weather data 
		W_TV={key:pd.concat([weather_train[key],weather_test[key]],axis=0) for key in weather_train.keys()} # dictionary with entry for each weather characteristic		
		for key in W_TV: W_TV[key]=dp.add_lags(data=W_TV[key],f_lags=f_lags,p_lags=[],f_lab=key) # lagg weather characteristics
		TV=pd.concat([W_TV[key] for key in W_TV]+[TV],axis=1).dropna() # merge weather characteristics
	TV=dp.idx2dmy(TV) # add dummies for dates
	T=TV[:test.index.min()] # everything before first date in test is train set
	V=TV[test.index.min():] # everything after first date in test is test set
	V=V[(V.index.hour==0) & (V.index.minute==0)] # remove entries with Y other than 00:00-23:00
	T_X,T_Y=dp.X_Y(data=T,Y_lab='Y') # split train set into samples (input, output)
	V_X,V_Y=dp.X_Y(data=V,Y_lab='Y') # split test set into samples (input, output)
	stop=kr.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto') # end training when loss on validation starts decreasing
	best_m=None # best model
	best_loss=float('inf') # loss of best model
	for i in range(restart): # restart network a number of times
		m=model(n_in=T_X.shape[1], n_out=T_Y.shape[1]) # create model
		hist=m.fit(T_X.as_matrix(), T_Y.as_matrix(), nb_epoch=epochs, batch_size=batch,shuffle=True,validation_split=0.2,callbacks=[stop],verbose=0) # train neural network
		if hist.history['val_loss'][-1]<best_loss: # better that current best
			best_m=m # save model as best
			train_loss=pd.DataFrame(data=hist.history['loss']) # save loss on train set
			val_loss=pd.DataFrame(data=hist.history['val_loss']) # save loss on test set
			best_loss=hist.history['val_loss'][-1] # save loss as best
	V_pred=pd.DataFrame(best_m.predict(V_X.as_matrix()),index=test.index,columns=test.columns) # forecasts for the next batch
	V_pred=dp.re_std(V_pred,args_2) # re-standardize data
	if prep and postp: # if data was preprocessed
		V_pred=postp(V_pred,args) # revert other preprocessing
	return V_pred,train_loss,val_loss


def cv(train,test,model,epochs=100,restart=10,batch=28,mini_batch=50,f_lags=[i for i in range(48)],p_lags=[i+1 for i in range(48)],weather_train=None,weather_test=None,prep=None,postp=None):
	W_T=None
	W_V=None
	pred=pd.DataFrame() # dataframe for predictions
	train_loss=pd.DataFrame() # dataframe for loss on training set
	val_loss=pd.DataFrame() # dataframe for loss on validation set
	for i in range(0,len(test),batch): # for each batch
		print(i//batch) # report progress
		if (len(test)-i)<batch: val_size=len(test)-i # not enough observation for complete batch (at the end of dataset)
		else: val_size=batch # smaller batch 
		T=pd.concat([train,test[:i]]) # build train set
		V=test[i:i+val_size]  # build validation set
		if weather_train and weather_test: # if using weather data
			W_T={}
			W_V={}
			for key in weather_train.keys(): # for each weather characteristic
				W_T[key]=pd.concat([weather_train[key],weather_test[key][:i]]) # build weather train set
				W_V[key]=weather_test[key][i:]  # build weather validation set
		new_pred,tl,l=ev(train=T,test=V,model=model,epochs=100,batch=mini_batch,restart=restart,f_lags=[i for i in range(48)],p_lags=kwargs['lags'],weather_train=W_T,weather_test=W_V,prep=prep,postp=postp) # evaluate network
		pred=pd.concat([pred,new_pred],axis=0) # add new predictions
		train_loss=pd.concat([train_loss,tl],axis=0,ignore_index=True) # append to old loss
		val_loss=pd.concat([val_loss,l],axis=0,ignore_index=True) # append to old loss
	return pred,train_loss,val_loss


np.random.seed(0) # fix seed for reprodicibility
data_dir='C:/Users/SABA/Google Drive/mtsg/data/train_test/' # directory containing data 
exp_dir='C:/Users/SABA/Google Drive/mtsg/data/sln/' # directory containing results of experiments

true=dp.load(path=data_dir+'test.csv',idx='date',dates=True) # observations to forecast
measures={'SRMSE':pf.srmse,'MASE':partial(pf.mase,shift=48*7),'SMAPE':pf.smape,'SMAE':pf.smae,} # performance to consider
train=dp.load(path=data_dir+'train.csv', idx='date', dates=True) # load train set
test=dp.load(path=data_dir+'test.csv', idx='date', dates=True) # load test set
weather_train={name:dp.load(path=data_dir+name+'_train.csv', idx='date', dates=True) for name in ['temp','hum','wind']} # load weather characteristics for train set
weather_test={name:dp.load(path=data_dir+name+'_test.csv', idx='date', dates=True) for name in ['temp','hum','wind']} # load weather characteristics for test set


batch=28 # batch size for cross validation
# parameters for cross-validation
params={
	'hidden':[50,100,200],
	'lags':[[i+1 for i in range(72)],[i+1 for i in range(72)]+[i+1 for i in range(48*6,48*7+24)]],
	'prep':['dec','mean','none'],
	'weather':[1,0],
	'act':['sigmoid','tanh']
	}

i=0
# SLNs
for kwargs in dp.dol2lod(params): # for each combination of parameters
	i+=1
	name=''
	for key in sorted(kwargs): # format label for model results
		if key=='lags': name+=key+'='+str(max(kwargs[key]))+','
		else: name+=key+'='+str(kwargs[key])+','
	name+='sln.csv'
	print(str(i)+'/'+str(len(dp.dol2lod(params))),name) # report progress
	model=partial(sln, n_hid=kwargs['hidden'],hid_act=kwargs['act'],out_act='linear',opt='adam') # construct model
	# construct preprocessing functions
	if kwargs['prep']=='dec':
		prep=dp.de_seas
		postp=dp.re_seas
	if kwargs['prep']=='mean':
		prep=dp.de_mean
		postp=dp.re_mean
	if kwargs['prep']=='none':
		prep=None
		postp=None
	if kwargs['weather']:
		W_T=weather_train
		W_V=weather_test
	else:
		W_T=None
		W_V=None
		# start cross-validation
	pred,train_loss,val_loss=cv(train=train,test=test,model=model,epochs=100,restart=10,batch=batch,mini_batch=64,f_lags=[i for i in range(48)],p_lags=kwargs['lags'],weather_train=W_T,weather_test=W_V,prep=prep,postp=postp)
	dp.save(data=pred, path=exp_dir+name, idx='date') # save results


np.random.seed(0) # fix seed for reprodicibility
data_dir='C:/Users/SABA/Google Drive/mtsg/data/train_test/' # directory containing data 
exp_dir='C:/Users/SABA/Google Drive/mtsg/data//mln/' # directory containing results of experiments

true=dp.load(path=data_dir+'test.csv',idx='date',dates=True) # observations to forecast
measures={'SRMSE':pf.srmse,'MASE':partial(pf.mase,shift=48*7),'SMAPE':pf.smape,'SMAE':pf.smae,} # performance to consider
train=dp.load(path=data_dir+'train.csv', idx='date', dates=True) # load train set
test=dp.load(path=data_dir+'test.csv', idx='date', dates=True) # load test set
weather_train={name:dp.load(path=data_dir+name+'_train.csv', idx='date', dates=True) for name in ['temp','hum','wind']} # load weather characteristics for train set
weather_test={name:dp.load(path=data_dir+name+'_test.csv', idx='date', dates=True) for name in ['temp','hum','wind']} # load weather characteristics for test set

batch=28 # batch size for cross validation
# parameters for cross-validation
params={
	'hid1':[50,100,200],
	'hid2':[50,100,200],
	'lags':[[i+1 for i in range(72)],[i+1 for i in range(72)]+[i+1 for i in range(48*6,48*7+24)]],
	'prep':['dec','mean','none'],
	'weather':[1,0],
	'act1':['sigmoid','tanh'],
	'act2':['sigmoid','tanh']
	}
		
i=0
# MLN
for kwargs in dp.dol2lod(params):
	i+=1
	name=''
	for key in sorted(kwargs): # format label for model results
		if key=='lags': name+=key+'='+str(max(kwargs[key]))+','
		else: name+=key+'='+str(kwargs[key])+','
	name+='mln.csv'
	if name in os.listdir(exp_dir):	continue
	print(str(i)+'/'+str(len(dp.dol2lod(params))),name)
	model=partial(mln, n_hid1=kwargs['hid1'],n_hid2=kwargs['hid2'],hid1_act=kwargs['act1'],hid2_act=kwargs['act2'],out_act='linear',opt='adam')
	# construct preprocessing functions
	if kwargs['prep']=='dec':
		prep=dp.de_seas
		postp=dp.re_seas
	if kwargs['prep']=='mean':
		prep=dp.de_mean
		postp=dp.re_mean
	if kwargs['prep']=='none':
		prep=None
		postp=None
	if kwargs['weather']:
		W_T=weather_train
		W_V=weather_test
	else:
		W_T=None
		W_V=None
	# start cross-validation
	pred,train_loss,val_loss=cv(train=train,test=test,model=model,epochs=100,restart=3,batch=batch,mini_batch=64,f_lags=[i for i in range(48)],p_lags=kwargs['lags'],weather_train=W_T,weather_test=W_V,prep=prep,postp=postp)
	dp.save(data=pred, path=exp_dir+name, idx='date')
	
#pf.ev(pred=pred, true=true, label='mln', measures=measures) # evaluate performance of individual network



