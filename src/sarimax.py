'''SARIMAX'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dataprep as dp
from sklearn.metrics import r2_score,mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX



def data_stats(data,window=7):
	from statsmodels.tsa.stattools import adfuller
	#Determing rolling statistics
	mean_rol=data.rolling(window=window).mean()
	std_rol=data.rolling(window=window).std()
	#Plot rolling statistics:
	orig = plt.plot(data, color='blue',label='load')
	mean = plt.plot(mean_rol, color='red', label='rolling mean')
	std = plt.plot(std_rol, color='green', label = 'rolling std')
	plt.legend(loc='best')
	plt.title('rolling Mean & standard Deviation')
	plt.show(block=False)
	#Perform Dickey-Fuller test:
	print('Results of Dickey-Fuller Test:')
	dftest = adfuller(data, autolag='AIC')
	dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
	for key,value in dftest[4].items():
		dfoutput['Critical Value (%s)'%key] = value
	print(dfoutput)
	
# perform Dickey-Fuller test to check data stacionarity:	
def df_test(data):
	from statsmodels.tsa.stattools import adfuller	
	print('Results of Dickey-Fuller Test:')
	df_test = adfuller(data, autolag='AIC')
	dfoutput = pd.Series(df_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
	for key,value in df_test[4].items():
		dfoutput['Critical Value (%s)'%key] = value
	print(dfoutput)

# ACF & PACF plots to determine p,q
def plot_cfs(data,lag_acf=10,lag_pacf=10):
	fig = plt.figure(figsize=(12,8))
	ax1 = fig.add_subplot(2,1,1)
	fig = plot_acf(data,lags=lag_acf,ax=ax1) # plot ACF
	ax2 = fig.add_subplot(2,1,2)
	fig=plot_pacf(data,lags=lag_pacf,ax=ax2) # plot PACF
	
np.random.seed(0) # fix seed for reprodicibility
path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv' # data path
load=dp.load(path) # load data
dp.cut(load,inplace=True) # remove leading & trailing Nans
load_with_nans=load.apply(axis=1,func=(lambda x: np.nan if (x.isnull().sum()>0) else x.mean())).unstack() # custom sum function where any Nan in arguments gives Nan as result		
load_filled_nans=pd.DataFrame(load_with_nans.fillna(method='bfill')) # placeholder, explore also custom predictions

# perform Dickey-Fuller test for each hour
for i in range(0,24):
	print('{}:'.format(i))
	df_test(load_train[i])
	print()


train,test=dp.split_train_test(load_filled_nans, test_size=0.25) # split data into train & test samples

def sarimax(train,test):
	train_pred=pd.DataFrame(data=None,index=train.index,columns=train.columns) # in sample predictions on train set
	test_pred=pd.DataFrame(data=None,index=test.index,columns=test.columns) # out of sample prediction on test set	
	for (i,train_day,test_day) in [(i, dp.split(train,nsplits=7)[i], dp.split(test,nsplits=7)[i]) for i in dp.split(train,nsplits=7)]: # for each day
		train_pred_day=pd.DataFrame(data=None,index=train_day.index,columns=train_day.columns) # in sample predictions on train set
		test_pred_day=pd.DataFrame(data=None,index=test_day.index,columns=test_day.columns) # out of sample prediction on test set
		for hour in train_day: # for each hour in a day
			train_day_hour=train_day[hour] # train samples for particular hour
			test_day_hour=test_day[hour] # test samples for particular hour
			model_train = SARIMAX(train_day_hour, order=(0,1,1),seasonal_order=(0,1,1,7),trend='c',measurement_error=True).fit() # train model
			model_test=SARIMAX(pd.concat([train_day_hour,test_day_hour]), order=(0,1,1),seasonal_order=(0,1,1,7),trend='c',measurement_error=True).filter(model_train.params) # workaround for rolling day ahead forecast
			train_pred_day[hour]=model_test.predict(start=0,end=len(train_day)-1) # predict in sample on train set
			test_pred_day[hour]=model_test.predict(start=len(train_day)) # predict out of sample on test set
		train_pred.iloc[i::7,:]=train_pred_day # fill corresponding rows with in sample predictions
		test_pred.iloc[i::7,:]=test_pred_day # fill corresponding rows with out of sample predictions
	return train_pred,test_pred

r2_score(y_pred=pred_test,y_true=test,multioutput='uniform_average')



load=load_train[11]
load_ma=load.rolling(window=28).mean()
load_ewm=load.ewm(halflife=7).mean()
load_1diff=load.shift(1)-load
load_7diff=load.shift(7)-load
load_364diff=load.shift(364)-load


model_0diff=ARIMA(load,order=(0,0,0)).fit(trend='c')
model_1diff=ARIMA(load.values,order=(0,1,0)).fit(trend='c')
model_2diff=ARIMA(load.values,order=(0,2,0)).fit(trend='c')


model_0diff.predict(start=0,end=len(load)-1)
model_1diff.predict(start=1,end=len(load)-1)
model_2diff.predict(start=2,end=len(load)-1)

mean_squared_error(y_true=load, y_pred=model_0diff.predict(start=0,end=len(load)-1))
mean_squared_error(y_true=load.shift(1).dropna(), y_pred=model_1diff.predict(start=1,end=len(load)-1))
mean_squared_error(y_true=load.shift(2).dropna(), y_pred=model_2diff.predict(start=2,end=len(load)-1))



plt.plot(load,color='black')
plt.plot(load_ma,color='red')
plt.plot(load_ewm,color='green')
plt.plot(load_ewm,color='blue')
plt.plot(load_1diff,color='orange')



	
load_dec=seasonal_decompose(load.values,freq=7) # decompose presuming weeks to correspond to load cycles, seasonal_decompose uses a symmetric moving average by default => head & tail nans
load_dec.plot()

week_period=load_dec.seasonal
load_noweek=load-week_period

plt.plot(load.index,load,'b',load_noweek,'r')


fig=plt.figure(figsize=(12,8))
fig.add_subplot(2,1,1)
fig=plt.plot(load.index,load)
fig.add_subplot(2,2,1)
fig=plt.plot(load_noweek.index,load_noweek)
plt.show()

load_noweek_dec=seasonal_decompose(load_noweek.values,freq=28)
load_noweek_dec.plot()

load_resid=pd.Series(load_dec.resid,index=load_filled_nans.index).dropna()
	



	
acf,ci,Q,p_value = acf(load_resid, nlags=50,alpha=0.05,  qstat=True, unbiased=True)

res = arma_order_select_ic(y, max_ar=7, max_ma=7, ic=['aic', 'bic', 'hqic'], trend='c', fit_kw=dict(method='css'))

# ARMA model

model.

results_AR = model.fit(disp=-1)  
plt.plot(data)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-data)**2))

# MA model
model = ARIMA(ts_log, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.plot(data)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-data)**2))

# AR+MA model
model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(data)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-data)**2))






#Plot ACF: 
plt.subplot(1,2,1) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(1,2,2)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


lag_acf = acf(Load_resid, nlags=50)
lag_pacf = pacf(load_resid, nlags=50, method='ols')



fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)
ax1.plot(load.index,load)
ax1.set_title('Removed week seasionality')
ax2.plot(load_noweek.index,load_noweek)
