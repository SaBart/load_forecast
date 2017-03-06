# DATA PROCESSING METHODS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from json import loads
from urllib.request import urlopen

# loads load profiles
def load_lp(path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv'):
	data=pd.read_csv(path,header=0,sep=";",usecols=[0,1,2], names=['date','time','load'],dtype={'load': np.float64},na_values=['?'], parse_dates=['date'], date_parser=(lambda x:pd.to_datetime(x,format='%d/%m/%Y'))) # read csv
	data['hour']=pd.DatetimeIndex(data['time']).hour # new column for hours
	data['minute']=pd.DatetimeIndex(data['time']).minute # new column for minutes
	data=pd.pivot_table(data,index=['date','hour'], columns='minute', values='load') # pivot so that minutes are columns, date & hour multi-index and load is value
	if not data.index.is_monotonic_increasing: data.sort_index(inplace=True) # sort dates if necessary
	data=data.applymap(lambda x:(x*1000)/60) # convert kW to Wh
	return data

# loads file
def load(path,index='date'):
	data=pd.read_csv(path,header=0,sep=",", parse_dates=[index],index_col=index)
	return data

# saves data to csv
def save(data,path):
	data.to_csv(path,header=True)
	return

# saves dictionary containing {key:dataframe}
def save_dict(dic,path):
	for key,value in dic.items():
		save(data=value,path=path+str(key)+'.csv') # save data
	return

# downloads one day worth of weather data
def dl_day(date):
	request='http://api.wunderground.com/api/8cc2ef7d44313e70/history_'+date+'/q/ORY.json' # construct request
	stream=urlopen(request) # open stream
	data=stream.read().decode('utf-8') # read data from stream
	stream.close() # close stream
	return loads(data) # parse json data

# download & save raw weather data for specified dates
def dl_save_w(dates,path):
	with open(path, 'w', newline='') as csv_file: # open stream
		obs=[o for o in dl_day(dates[0])['history']['observations'] if o['metar'].startswith('METAR')] # get all METAR format observations
		obs[0].pop('utcdate',None) # remove utcdate entry
		obs[0].pop('date',None) # remove date entry
		obs[0]['timestamp']='timestamp' # add entry for timestamps
		writer=csv.DictWriter(csv_file,fieldnames=obs[0].keys()) # set up dictionary writer with proper fields 
		writer.writeheader() # write header
		for date in dates: # for each date
			obs=[o for o in dl_day(date)['history']['observations'] if o['metar'].startswith('METAR')] # get all METAR format observations
			for o in obs: # for each observation
				o.pop('utcdate',None) # remove utcdate entry
				date=o.pop('date',None) # get date entry (it is a dictionary, extracting values will follow)
				if not date is None:
					o['timestamp']='{}-{}-{} {}:{}'.format(date['year'],date['mon'],date['mday'],date['hour'],date['min'])
				writer.writerow(o)
	return			

# loads & formats raw weather data  
def load_w(path='C:/Users/SABA/Google Drive/mtsg/data/weather_3.csv',index='timestamp',cols=['tempm','hum','pressurem','wspdm']):
	data=pd.read_csv(path,header=0,sep=",",usecols=[index]+cols, parse_dates=[index],index_col=index) # read csv
	data=data.resample('H').mean() # average values across hours
	data['date']=pd.DatetimeIndex(data.index).normalize() # new column for dates
	data['hour']=pd.DatetimeIndex(data.index).hour # new column for hours
	data=pd.pivot_table(data,index=['date','hour']) # pivot so that minutes are columns, date & hour multi-index and load is value
	return data

# splits weather data by columns & formats each part and outputs a dictionary with {keys}=={column names} 
def split_cols(data):
	return {col:data[col].unstack() for col in data.columns} # return a dictionary of dataframes each with values from only one column of original dataframe and key equal to column name	 

def split_cols_save(data,paths):
	for (col,path) in zip(data.columns,paths): # for each pair of a column and a path 
		save(data[col].unstack(),path) # save formatted column under the path
	return

# loads and concats multiple weather files into one dataframe
def load_concat_w(paths,index='timestamp',cols=['tempm','hum','pressurem','wspdm']):
	data=pd.concat([load_w(path,index,cols) for path in paths])
	return data

# combines minute time intervals into hours
def m2h(data):
	data=data.mean(axis=1,skipna=False).unstack()  # average load across hours, preserving nans
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
	
def load_merge(paths,index='date'):
	data=pd.concat([pd.read_csv(path,header=0,sep=",", parse_dates=[index],index_col=index) for path in paths], axis=0) # load all dataframes into one
	if not data.index.is_monotonic_increasing:
		data.sort_index(inplace=True) # order index
	return data	
	
# rounds down to the nearest multiple of base
def flr(x,base=7):
	return base*int(x/base)