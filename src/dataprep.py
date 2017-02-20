# DATA PROCESSING METHODS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from json import loads
from urllib.request import urlopen

# loads data
def load(path='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/household_power_consumption.csv'):
	data=pd.read_csv(path,header=0,sep=";",usecols=[0,1,2], names=['date','time','load'],dtype={'load': np.float64},na_values=['?'], parse_dates=['date'], date_parser=(lambda x:pd.to_datetime(x,format='%d/%m/%Y'))) # read csv
	data['date']=pd.DatetimeIndex(data['date']).strftime('%Y%m%d') # reformat dates
	data['hour']=pd.DatetimeIndex(data['time']).hour # new column for hours
	data['minute']=pd.DatetimeIndex(data['time']).minute # new column for minutes
	data=pd.pivot_table(data,index=['date','hour'], columns='minute', values='load') # pivot so that minutes are columns, date & hour multi-index and load is value
	data=data.applymap(lambda x:(x*1000)/60) # convert kW to Wh 
	return data

def download_day(date):
	request='http://api.wunderground.com/api/8cc2ef7d44313e70/history_'+date+'/q/ORY.json' # construct request
	stream=urlopen(request) # open stream
	data=stream.read().decode('utf-8') # read data from stream
	stream.close() # close stream
	return loads(data) # parse json data

date='20061216'

def download_weather(dates):
	data=pd.DataFrame(data=None,index=pd.MultiIndex.from_product([dates,range(24)], names=['date','hour']),columns=['temp','psr','hum','prc','wspd','wdir'])
	for date in dates: # for each date
		obs=[o for o in download_day(date)['history']['observations'] if o['metar'].startswith('METAR')] # get all METAR format observations
		for o in obs: # for each observation
			year = o['date']['year'] # year
			month = o['date']['mon'] # month
			day = o['date']['mday'] # day
			hour = o['date']['hour'] # hour
			min = o['date']['min'] # minute
			temp = o['tempm'] # temperature [C]
			psr = o['pressurem'] # pressure [mBar]=10^2[Pa]
			hum = o['hum'] # humidity [%]
			prc=o['precipm'] # precipitation [mm]
			wspd = o['wspdm'] # wind speed [kmph]
			wdir = o['wdird'] # wind direction [deg]
			

# saves data to csv
def save(data,directory,name):
	data.to_csv(directory+name+'.csv',header=True)
	
# splits data according to weekdays and saves them 
def split_save(data,nsplits,directory,name):
	for i,chunk in split(data,nsplits=nsplits).items(): # for each day
		save(chunk,directory,name+'_'+str(i)) # save under special name
	
# combines minute time intervals into hours
def m2h(data,nan='keep'):
	if nan=='keep': # if we want to keep Nans
		data= data.apply(axis=1,func=(lambda x: np.nan if (x.isnull().sum()>0) else x.mean())).unstack() # custom sum function where any Nan in minute time interval results in Nan for hour time interval
	data.index=pd.to_datetime(data.index) # convert string representation of dates in index to datetime index
	return data

# flattens data, converts columns into a multiindex level
def flatten(data):
	if not isinstance(data, pd.Series): data=data.stack() # if not series (already flat) then flatten
	return data
	
# remove incomplete first and last days
def cut(data):
	f,_=data.index.min() # first day
	l,_=data.index.max() # last day
	if len(data.loc[f])<24: # if first day is incomplete
		data=data.drop(f,level=0) # drop the whole day
	if len(data.loc[l])<24: # if last day is incomplete
		data=data.drop(l,level=0) # drop the whole day
	return data

# shifts data for time series forcasting
def shift(data,n_shifts=1,shift=1):
	data_shifted={} # lagged dataframes for merging
	for i in range(0,n_shifts+1): # for each time step
		label='targets' # label for target values
		if i!=n_shifts:label='t-{}'.format(n_shifts-i) # labels for patterns
		data_shifted[label]=data.shift(-i*shift) # add lagged dataframe
	res=pd.concat(data_shifted.values(),axis=1,join='inner',keys=data_shifted.keys()) # merge lagged dataframes
	return res.dropna() # TODO: handling missing values

# order timesteps from the oldest
def order(data):
	data=data[sorted(data.columns,reverse=True,key=(lambda x:x[0]))] # sort first level of column multiindex in descending order
	return data
	
# split data into patterns & targets
def split_X_Y(data,target_label='targets'):
	X=data.select(lambda x:x[0] not in [target_label], axis=1) # everything not labelled "target" is a pattern, [0] refers to the level of multi-index
	Y=data[target_label] # targets
	return X, Y

# split data into train & test sets
def split_train_test(data, base=7,test_size=0.25): # in time series analysis order of samples usually matters, so no shuffling of samples
	idx=flr((1-test_size)*len(data),base) if test_size>0 else len(data) # calculate number of samples in train set 
	train,test =data[:idx],data[idx:] # split data into train & test sets
	return train,test

# split data into n datasets (according to weekdays)
def split(data,nsplits=7): 
	return {i:data.iloc[i::nsplits] for i in range(nsplits)} # return as a dictionary {offset:data}
	
# rounds down to the nearest multiple of base
def flr(x,base=7):
	return base*int(x/base)