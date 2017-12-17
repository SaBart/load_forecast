'''WORKSPACE'''

import os as os
import numpy as np
import pandas as pd
import dataprep as dp
import imputation as imp
import performance as pf
import datetime
from rpy2.robjects.packages import importr
from importlib import reload
from functools import partial
from dataprep import resample
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# LOAD DATA
data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data
data=dp.load_lp(data_dir+'raw/loads.csv') # load raw data and process it
#dp.save(data,path=data_dir+'data.csv',idx='datetime') # save processed data
#data=dp.load(path=data_dir+'data.csv', idx='datetime',cols='load',dates=True) # load processed data

# OPTIMISE SHIFT FOR NAIVE FORECASTING METHOD
data_res=resample(data,freq=1440) # aggregate minute intervals into half-hour intervals
results=pf.opt_shift(data_res,shifts=[48,7*48]) # find the best shift for naive predictor for MASE

# IMPUTATION
impts=importr('imputeTS') # package for time series imputation
shift=7*48# the shift that performed the best
measures={'SMAE':pf.smae,'SRMSE':pf.srmse,'SMAPE':pf.smape,'MASE':partial(pf.mase,shift=shift)} # performance to consider

random={} # params for random
mean={'option':['mean','median','mode']} # params for mean
ma={'weighting':['simple','linear','exponential'],'k':np.arange(1,11)} # params for moving average
locf={'option':['locf','nocb'],'na.remaining':['rev']} # params for last observation carry forward
interpol={'option':['linear','spline','stine']} # params for interpolation
kalman={'model':['auto.arima','StructTS']} # params for state-space models
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

methods+=[{'name':'seadec','alg':impts.na_seadec,'opt':opt} for opt in dec_split]+[{'name':'seasplit','alg':impts.na_seasplit,'opt':opt} for opt in dec_split] # add more complex methods

np.random.seed(0) # fix seed for reproducibility
imp_res=imp.opt_imp(data,methods=methods, n_iter=10,measures=measures) # try all imputation methods
dp.save(imp_res,path=data_dir+'imp/imp.csv',idx='method') # save results

data=imp.imp(data, alg=impts.na_seadec, freq=1440, **{'algorithm':'ma','weighting':'linear','k':2}) # impute the whole dataset using three best methods of imputation
dp.save(data, path=data_dir+'data_imp.csv', idx='datetime') # save imputed data


# AGGREGATE DATA & CREATE TRAIN & TEST SETS
exp_dir='C:/Users/SABA/Google Drive/mtsg/data/train_test/' # directory for the results
data=dp.load(path=data_dir+'data_imp.csv',idx='datetime',cols='load',dates=True) # load imputed data

data=dp.resample(data,freq=1440) # aggregate minutes to half-hours
train,test=dp.train_test(data=data, test_size=0.255, base=7) # split into train & test sets
dp.save(data=train,path=exp_dir+'train.csv',idx='date') # save train set
dp.save(data=test,path=exp_dir+'test.csv',idx='date') # save test set
dp.save_dict(dic=dp.split(train,nsplits=7),path=exp_dir+'train_',idx='date') # split train set according to weekdays and save each into a separate file
dp.save_dict(dic=dp.split(test,nsplits=7),path=exp_dir+'test_',idx='date') # split test set according to weekdays and save each into a separate file
	

# WEATHER DATA

data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data 

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
weather=dp.load_concat_w([data_dir+path for path in paths],idx='timestamp',cols=['tempm','hum','wspdm'],dates=True) # join all parts of weather data
weather.fillna(method='bfill',inplace=True) # fill missiong values (for column with maximum missin it is still only 0.045% of all)

# save selected weather parameters
temp=weather['tempm'] # extract temperature
hum=weather['hum'] # extract humidity
wind=weather['wspdm'] # extract wind speed
dp.save(data=temp,path=data_dir+'temp.csv',idx='datetime') # save temperature
dp.save(data=hum,path=data_dir+'hum.csv',idx='datetime') # save humidity
dp.save(data=wind,path=data_dir+'wind.csv',idx='datetime') # save wind speed

# splitting, aggregating & saving weather data
temp_dir=data_dir+'experiments/data/' # where to save
for name in ['temp','hum','wind']: # for each column=weather parameter
	data_w=dp.load(path=data_dir+name+'.csv', idx='datetime', dates=True)
	data_w,_=dp.de_std(data_w) # standardize data
	data_w=dp.resample(data=data_w, freq=48) # reshape to have time of day as columns
	train_w,test_w=dp.train_test(data=data_w, test_size=0.255, base=7) # split into train & test sets
	dp.save(data=train_w,path=temp_dir+name+'_train.csv',idx='date') # save train set
	dp.save(data=test_w,path=temp_dir+name+'_test.csv',idx='date') # save test set
	dp.save_dict(dic=dp.split(train_w,nsplits=7),path=temp_dir+name+'_train_',idx='date') # split train set according to weekdays and save each into a separate file
	dp.save_dict(dic=dp.split(test_w,nsplits=7),path=temp_dir+name+'_test_',idx='date') # split test set according to weekdays and save each into a separate file


# IMPUTATION RESULTS

exp_dir='C:/Users/SABA/Google Drive/mtsg/data/imp/' # directory containing results of experiments
results=dp.load(exp_dir + 'imp.csv',idx='method')
results=results.sort_values(by=['SRMSE','MASE','SMAPE','SMAE']) # sort in ascending order
results=results[['SRMSE','MASE','SMAPE','SMAE']] # rearrange columns
print(results.to_latex(float_format='%.4f')) # make table for latex


# DATA ANALYSIS

# BC decomposition graph
bc=dp.load(path=data_dir+'train_bc.csv', idx='date', dates=True)['2007-03-24':'2007-04-02']
bc=bc.reset_index()
f,ax=plt.subplots(nrows=2,ncols=1,sharex=True)

bc['load'].plot(ax=ax[0])
bc['Box-Cox'].plot(ax=ax[1])
ax[1].minorticks_off()
ax[1].set_xticks([i*48 for i in range(11)])
ax[1].set_xticklabels(['Fri','Sat','Sun','Mon','Tue','Wen','Thu','Fri','Sat','Sun','Mon'],fontsize=16)
for a in ax:
	a.tick_params(labelsize=16)
	a.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0].set_ylabel('Load (kW)',fontsize=18)
ax[1].set_ylabel('Box-Cox',fontsize=18)
plt.xlabel('Day of week',fontsize=18)


# STL decomposition graph
dec=dp.load(path=data_dir+'train_dec.csv', idx='date', dates=True)['2007-03-24':'2007-04-02']
dec=dec.reset_index()

f,ax=plt.subplots(nrows=4,ncols=1,sharex=True)
dec['load'].plot(ax=ax[0])
dec['seasonal'].plot(ax=ax[1])
dec['trend'].plot(ax=ax[2])
dec['remainder'].plot(ax=ax[3])
ax[3].minorticks_off()
ax[3].set_xticks([i*48 for i in range(11)])
ax[3].set_xticklabels(['Fri','Sat','Sun','Mon','Tue','Wen','Thu','Fri','Sat','Sun','Mon'],fontsize=16)
for a in ax:
	a.tick_params(labelsize=16)
	a.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0].set_ylabel('Load (kW)',fontsize=18)
ax[1].set_ylabel('Seasonality (kW)',fontsize=18)
ax[2].set_ylabel('Trend (kW)',fontsize=18)
ax[3].set_ylabel('Remainder (kW)',fontsize=18)
#for a in ax:a.set_ylabel('Load (kW)')
plt.xlabel('Day of week',fontsize=18)


# plot outages for time intervals
data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data 
data=dp.load(path=data_dir+'data.csv', idx='datetime',cols='load',dates=True) # load data
data=dp.cut(data=data,freq=1440) # remove incomplete first and last days
data=data.isnull().resample(rule='30Min',closed='left',label='left').sum() # count nans for each half-hour
data=data.value_counts()
data.index=data.index.astype(int)
data=data.sort_index()

f,ax=plt.subplots() # get axis handle
ax.set_xticks(data.index)
ax.set_xticklabels(data.index,fontsize=16)
ax.set_yscale('log') # set logarithmic scale for y-values
ax.set_yticks(data)
ax.set_yticklabels(data,fontsize=16)
ax.grid(linestyle=':',axis='y',zorder=0)
ax.grid(axis='x',visible=False)
ax.set_xlabel('Outage length (min)',fontsize=18)
ax.set_ylabel('Number of half-hour intervals',fontsize=18)
data.plot(ax=ax,kind='bar',align='center',edgecolor='k',zorder=3,width=1.0,grid=True)


# plot outage lengths
data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data 
data=dp.load(path=data_dir+'data.csv', idx='datetime',cols='load',dates=True) # load data
data=dp.cut(data=data,freq=1440) # remove incomplete first and last days
out_len=imp.out_len(data)
out_len=pd.Series(out_len)

f,ax=plt.subplots() # get axis handle
ax.set_xticks(out_len.index)
ax.set_xticklabels(out_len.index,fontsize=16)
ax.set_yscale('log') # set logarithmic scale for y-values
ax.set_yticks(out_len)
ax.set_yticklabels(out_len,fontsize=16)
ax.grid(linestyle=':',axis='y',zorder=0)
ax.grid(axis='x',visible=False)
ax.set_xlabel('Outage length (min)',fontsize=18)
ax.set_ylabel('Count',fontsize=18)
out_len.plot(ax=ax,kind='bar',align='center',edgecolor='k',zorder=3,width=1.0,grid=True)


# average load profile
data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data 
data=dp.load(path=data_dir+'data_imp.csv', idx='datetime',cols='load',dates=True) # load data
data=dp.resample(data,freq=1440) # aggregate minutes to half-hours

f,ax=plt.subplots() # get axis handle
data.mean().plot(ax=ax,kind='bar',align='edge',edgecolor='k',width=1.0)
ax.tick_params(labelsize=16)
ax.set_xticks(range(49))
ax.set_xticklabels([pd.to_datetime(time,format='%H%M').strftime('%H:%M') for time in data.columns] + ['24:00'])
ax.set_xlabel('Time',fontsize=18)
ax.set_ylabel('Load (kW)',fontsize=18)


# average Mon, Tue,... load profiles
data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data 
data=dp.load(path=data_dir+'data_imp.csv', idx='datetime',cols='load',dates=True) # load data
data=dp.resample(data,freq=1440) # aggregate minutes to half-hours
days=dp.split(data,nsplits=7)
labels=['Sun','Mon','Tue','Wen','Thu','Fri','Sat']

f,ax=plt.subplots(nrows=7,ncols=1,sharex=True)
for i,load in days.items():
	load.mean().plot(ax=ax[i],kind='bar',align='edge',edgecolor='k',width=1.0)
	ax[i].tick_params(labelsize=16)
	ax[i].set_yticks([0.0,1.0,2.0])
	ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	ax[i].set_ylabel(labels[i],fontsize=18)
ax[6].set_xticks(range(49))
ax[6].set_xticklabels([pd.to_datetime(time,format='%H%M').strftime('%H:%M') for time in data.columns] + ['24:00'])
ax[6].set_xlabel('Time',fontsize=18,labelpad=8)
f.text(0.07, 0.5, 'Load (kW)', ha='center', va='center', rotation='vertical',fontsize=18)

# plot acf for half-hourly lags
data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data 
data=dp.load(path=data_dir+'data_all.csv', idx='datetime',cols='load',dates=True) # load data

f,ax=plt.subplots()
plot_acf(data,lags=range(9*48+9),use_vlines=True,alpha=None,ax=ax)
ax.tick_params(labelsize=16)
ax.set_xticks([i*48 for i in range(0,10)])
ax.set_xticklabels(range(0,10))
ax.set_xlabel('Lag (days)',fontsize=18)
ax.set_ylabel('Autocorrelation',fontsize=18)
ax.set_title('')

# acf for weekly lags
data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data 
data=dp.load(path=data_dir+'data_all.csv', idx='datetime',cols='load',dates=True) # load data

f,ax=plt.subplots()
plot_acf(data,lags=[i*48*7 for i in range(0,205+1)],use_vlines=True,alpha=None,ax=ax)
ax.tick_params(labelsize=16)
ax.set_xticks([i*48*7*52 for i in range(0,5)])
ax.set_xticklabels(range(0,5))
ax.set_xlabel('Lags (years)',fontsize=18)
ax.set_ylabel('Autocorrelation',fontsize=18)
ax.set_title('')

# histogram
data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data 
data=dp.load(path=data_dir+'data_all.csv', idx='datetime',dates=True) # load data

f,ax=plt.subplots()
bins= [i/10 for i in range(0,52)]
data.hist(ax=ax,bins=500,grid=False,edgecolor='k',normed=True)
data.plot(ax=ax,kind='kde',lw=3,style='--',color='red')
ax.set_xlim(0,5.1)
ax.tick_params(labelsize=16)
ax.set_xticks(bins)
ax.set_xticklabels(labels=bins,rotation='vertical')
ax.set_xlabel('Load (kW)',fontsize=18)
ax.set_ylabel('Normalized number of half-hour intervals',fontsize=18)

# histogram for Box-Cox transformation
data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data 
data=dp.load(path=data_dir+'data_bc.csv', idx='datetime',dates=True) # load data

f,ax=plt.subplots()
bins=[i/10 for i in range(-26, 21)]
data.hist(ax=ax,bins=500,grid=False,edgecolor='k',normed=True)
data.plot(ax=ax,kind='kde',lw=3,style='--',color='red')
ax.set_xlim(-2.6,2.0)
ax.tick_params(labelsize=16)
ax.set_xticks(bins)
ax.set_xticklabels(labels=bins,rotation='vertical')
ax.set_xlabel('Load (Box-Cox transformed)',fontsize=18)
ax.set_ylabel('Normalized number of half-hour intervals',fontsize=18)


# plot weather data
data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data 
temp=dp.load(path=data_dir+'temp.csv', idx='datetime', cols='tempm', dates=True)
hum=dp.load(path=data_dir+'hum.csv', idx='datetime', cols='hum', dates=True)
wind=dp.load(path=data_dir+'wind.csv', idx='datetime', cols='wspdm', dates=True)

f,ax=plt.subplots(nrows=3,ncols=1,sharex=True)
temp.plot(ax=ax[0])
hum.plot(ax=ax[1])
wind.plot(ax=ax[2])
ax[2].minorticks_off()
for a in ax:a.tick_params(labelsize=16)
ax[0].set_ylabel('Temperature ($^\circ$C)',fontsize=18)
ax[1].set_ylabel('Humidity (%)',fontsize=18)
ax[2].set_ylabel('Wind speed (km/h)',fontsize=18)
tick_locs = [datetime.date(year=y,month=1,day=1) for y in [2007,2008,2009,2010]] +[ datetime.date(year=y,month=7,day=1) for y in [2007,2008,2009,2010]]
tick_labels = map(lambda x: x.strftime('%b'), tick_locs)
plt.xticks(tick_locs, tick_labels)
plt.xlabel('Month',fontsize=18)

# IMPUTATION RESULTS
exp_dir='C:/Users/SABA/Google Drive/mtsg/data/experiments/imp/' # directory containing results of experiments
ma=dp.load(path=exp_dir+'ma.csv', idx='method',dates=False) # load moving average results
interpol=dp.load(path=exp_dir+'interpol.csv', idx='method',dates=False) # load moving average results
other=dp.load(path=exp_dir+'other.csv', idx='method',dates=False) # load moving average results
print(ma.to_latex(float_format='%.4f',index=False))
print(interpol.to_latex(float_format='%.4f',index=False))
print(other.to_latex(float_format='%.4f',index=False))


# EXPERIMENTAL RESULTS
data_dir='C:/Users/SABA/Google Drive/mtsg/data/experiments/data/' # directory containing data
true=dp.load(data_dir+'test.csv',idx='date',dates=True) # load true values for test set
measures={'SRMSE':pf.srmse,'MASE':partial(pf.mase,shift=48*7),'SMAPE':pf.smape,'SMAE':pf.smae} # performance to consider

# table for latex
exp_dir='C:/Users/SABA/Google Drive/mtsg/data/experiments/es_week/' # directory containing results of experiments
results=pf.ev_dir(exp_dir, true, measures=measures) # evaluate performance of all results in directory
results=results.sort_values(by=['SRMSE','MASE','SMAPE','SMAE'])
results=results[['wa','ha','dec','bc','SRMSE','MASE','SMAPE','SMAE']]
#results=results[['wa','ha','dec','bc','SRMSE','MASE','SMAPE','SMAE']]
print((1*results).to_latex(float_format='%.4f',index=False)) # make table for latex

# ensemble averaging
exp_dir='C:/Users/SABA/Google Drive/mtsg/data/experiments/ensemble/' # directory containing results of experiments
results=pf.ev_ens(pred_dir=exp_dir, true=true, comb_size=5, measures=measures)
results=results.sort_values(by=['SRMSE','MASE','SMAPE','SMAE'])
results=results[['SRMSE','MASE','SMAPE','SMAE']]
print((1*results).to_latex(float_format='%.4f')) # make table for latex

# best & worst day graphs for best method
measures={'SRMSE':partial(pf.srmse,mean=dp.d2s(true).mean())} # performance measures to consider
exp_dir='C:/Users/SABA/Google Drive/mtsg/data/experiments/mln/' # directory containing results of experiments
pred=dp.load(exp_dir+'act1=sigmoid,act2=sigmoid,hid1=50,hid2=200,lags=360,prep=mean,weather=0,nn.csv',idx='date',dates=True) # load best predictions
results=pf.ev_day(pred=pred,true=true,measures=measures) # evaluate performance for each day separately
results=results.sort_values(by=['SRMSE'])

f,ax=plt.subplots(nrows=2,ncols=1,sharex=True)
true.loc['2010-06-10'].plot(ax=ax[0],marker='o')
pred.loc['2010-06-10'].plot(ax=ax[0],marker='s')
true.loc['2009-12-24'].plot(ax=ax[1],marker='o')
pred.loc['2009-12-24'].plot(ax=ax[1],marker='s')
ax[1].minorticks_off()
ax[1].set_xticks([i*8 for i in range(7)])
ax[1].set_xticklabels(['00:00','04:00','08:00','12:00','16:00','20:00','24:00'],fontsize=16)
for a in ax:
	a.tick_params(labelsize=16)
	a.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0].set_ylabel('Best day (kW)',fontsize=18)
ax[1].set_ylabel('Worst day (kW)',fontsize=18)
ax[1].legend(loc=0,labels=['true','forecast'],fontsize=16)
plt.xlabel('Hour',fontsize=18)

# best and worst days for the best ensemble
measures={'SRMSE':partial(pf.srmse,mean=dp.d2s(true).mean())} # performance measures to consider
exp_dir='C:/Users/SABA/Google Drive/mtsg/data/experiments/ensemble/' # directory containing results of experiments
paths=['ha,es.csv','des,arma.csv','ha,des,arma.csv','none,mln.csv','mean,mln.csv'] # best ensemble
pred=dp.load_merge(fol=exp_dir,paths=paths,idx='date',dates=True,axis=1).mean(axis=1) # combine the output of ensemble members
pred=dp.s2d(pred) # convert from setries to dataframe
results=pf.ev_day(pred=pred,true=true,measures=measures) # evaluate performance for each day separately
results=results.sort_values(by=['SRMSE'])

f,ax=plt.subplots(nrows=2,ncols=1,sharex=True)
true.loc['2010-05-04'].plot(ax=ax[0],marker='o')
pred.loc['2010-05-04'].plot(ax=ax[0],marker='s')
true.loc['2009-12-24'].plot(ax=ax[1],marker='o')
pred.loc['2009-12-24'].plot(ax=ax[1],marker='s')
ax[1].minorticks_off()
ax[1].set_xticks([i*8 for i in range(7)])
ax[1].set_xticklabels(['00:00','04:00','08:00','12:00','16:00','20:00','24:00'],fontsize=16)
for a in ax:
	a.tick_params(labelsize=16)
	a.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0].set_ylabel('Best day (kW)',fontsize=18)
ax[1].set_ylabel('Worst day (kW)',fontsize=18)
ax[1].legend(loc=0,labels=['true','forecast'],fontsize=16)
plt.xlabel('Hour',fontsize=18)

# dendrogram
exp_dir='C:/Users/SABA/Google Drive/mtsg/data/experiments/corr/' # directory containing results of experiments
res=dp.load_merge(fol=exp_dir,paths=os.listdir(exp_dir), idx='date', dates=True, axis=1)
res=res.fillna(value=0)
for col in res:	res[col]=dp.d2s(true)-res[col]
corr=res.corr(method='pearson')
dp.save(data=corr, path=data_dir+'corr.csv')
# dendrogram using http://biit.cs.ut.ee/clustvis/


		

import csv
import operator
from itertools import groupby
import random

header=''

path='C:/Users/SABA/tmp/15min_100267.csv'

for key, rows in groupby(csv.reader(open(path)),
						 operator.itemgetter(0)):
	with open("C:/Users/SABA/tmp/15min/%s.csv" % key, "w") as output:
		if header:output.write(",".join(header) + "\n")
		for row in rows:
			if (row[0]=='dataid'): header=row
			else:
				row = [s.replace('\\N', '') for s in row] 
				output.write(",".join(row) + "\n")


source='C:/Users/SABA/tmp/15min/raw/'
target='C:/Users/SABA/tmp/15min/data/sample/'
hids=[os.path.splitext(filename)[0] for filename in os.listdir(source) if len(dp.load_dp(source+filename))>=15*24*28*13] # sufficiently long datasets

random.shuffle(hids) # random sample

for hid in hids[:10]: # for each house id
	print(hid)
	data=dp.load_dp(source+hid+'.csv') # load data
	data[data<=0]=np.nan # replace negative values with nans
	if not dp.full(data):
		data=data.fillna(data.ewm(alpha=0.5,ignore_na=True).mean()) # impute data
		print(len(data[data<=0]))
		#data=imp.imp(data, alg=impts.na_seadec, freq=4*24, **{'algorithm':'ma','weighting':'exponential','k':2}) # impute the whole dataset using three best methods of imputation
	data=dp.resample(data,freq=4*24) # aggregate 15min to half-hours
	print(dp.full(data))
	if data.empty: continue
	new_dir=target+hid+'/'
	if not os.path.exists(new_dir):
		os.makedirs(new_dir)
	train,test=dp.train_test(data=data, test_size=28*13, base=7) # split into train & test sets
	dp.save(data=train,path=new_dir+'train.csv',idx='date') # save train set
	dp.save(data=test,path=new_dir+'test.csv',idx='date') # save test set
	dp.save_dict(dic=dp.split(train,nsplits=7),path=new_dir+'train_',idx='date') # split train set according to weekdays and save each into a separate file
	dp.save_dict(dic=dp.split(test,nsplits=7),path=new_dir+'test_',idx='date') # split test set according to weekdays and save each into a separate file




pred_dir='C:/Users/SABA/tmp/15min/results/arma/'
true_dir='C:/Users/SABA/tmp/15min/data/sample/'

pf.ev_dir(pred_dir=pred_dir, true=true_dir)