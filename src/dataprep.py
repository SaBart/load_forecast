# DATA PROCESSING METHODS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from json import loads
from urllib.request import urlopen
from copy import deepcopy

# loads load profiles
def load_lp(path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv'):
	data=pd.read_csv(path,header=0,sep=";",usecols=[0,1,2], names=['date','time','load'],dtype={'load': np.float64},na_values=['?'], parse_dates=['date'], date_parser=(lambda x:pd.to_datetime(x,format='%d/%m/%Y'))) # read csv
	data['hour']=pd.DatetimeIndex(data['time']).hour # new column for hours
	data['minute']=pd.DatetimeIndex(data['time']).minute # new column for minutes
	data=pd.pivot_table(data,index=['date','hour'], columns='minute', values='load') # pivot so that minutes are columns, date & hour multi-index and load is value
	data=order(data) # order data if necessary
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
def d2s(data):
	if not isinstance(data, pd.Series): data=data.stack(dropna=False) # if not Series (already flat) then flatten
	return data
	
# invert d2s operation
def s2d(data):
	if not isinstance(data, pd.DataFrame): data=data.unstack() # if not DataFrame (not flat) then unflatten
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
def shift(data,n_shifts=1,shift=1,target_label='targets'):
	data_shifted={} # lagged dataframes for merging
	for i in range(n_shifts+1): # for each time step
		label=target_label # label for target values
		if i!=0:label='t-{}'.format(i) # labels for patterns
		data_shifted[label]=data.shift(i*shift) # add lagged dataframe
	res=pd.concat(data_shifted.values(),axis='columns',keys=data_shifted.keys()) # merge lagged dataframes
	return res.dropna() # TODO: handling missing values

# order timesteps from the oldest
def order(data):
	if not data.index.is_monotonic_increasing: data=data.sort_index() # sort dates if necessary
	return data
	
# split data into patterns & targets
def X_Y(data,target_label='targets'):
	X=data.select(lambda x:x[0] not in [target_label], axis=1) # everything not labelled "target" is a pattern, [0] refers to the level of multi-index
	Y=data[target_label] # targets
	return X, Y

# split data into train & test sets
def train_test(data, base=7,test_size=0.25): # in time series analysis order of samples usually matters, so no shuffling of samples
	split_idx=round_rem(total=len(data),base=base,test_size=test_size) # calculate the index that splits dataset into train, test
	train,test =data[:split_idx],data[split_idx:] # split data into train & test sets
	return train,test

# returns number n that the (total-n) is rounded to base and (total-n)/total>test_size  
def round_rem(total,base=7,test_size=0.25):
	return total-round_u(total-(1-test_size)*total,base) if test_size>0 else total # calculate the index that splits dataset into train, test

# split data into n datasets (according to weekdays)
def split(data,nsplits=7): 
	return {i:data.iloc[i::nsplits] for i in range(nsplits)} # return as a dictionary {offset:data}
	
def load_merge(paths,index='date'):
	data=pd.concat([pd.read_csv(path,header=0,sep=",", parse_dates=[index],index_col=index) for path in paths], axis=0) # load all dataframes into one
	if not data.index.is_monotonic_increasing:
		data.sort_index(inplace=True) # order index
	return data	
	
# rounds down to the nearest multiple of base
def round_d(x,base=7):
	return base*int(x/base)

# rounds up to the nearest multiple of base
def round_u(x,base=7):
	if x%base==0: result=x
	else: result=base*int((x+base)/base)
	return int(result)

# construct training & testing sets for time series cross validation
def tscv(total,base=7,test_size=0.25,batch=28):
	len_train=round_rem(total=total,base=base,test_size=test_size) # calculate number of training samples
	tscv_iter=[(np.arange(i),i+np.arange(min(batch,total-i))) for i in range(len_train,total,batch)] # construct the iterator, a list of tuples, each containing train & test indices
	return tscv_iter

# standardise data
def z_val(data):
	data_flat=d2s(data) # flatten DataFrame into a Series
	mean=data_flat.mean() # get mean
	std=data_flat.std() # get std
	return (data-mean)/std,mean,std

# invert standardisation
def z_inv(data,mean,std):
	return (data*std)+mean
