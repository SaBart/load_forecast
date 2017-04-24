# DATA PROCESSING METHODS

import numpy as np
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
	data=pd.read_csv(path,header=0,sep=";",usecols=[0,1,2], names=['date','time','load'],dtype={'load': np.float64},na_values=['?'], parse_dates={'datetime':['date','time']},date_parser=(lambda x:pd.to_datetime(x,format='%d/%m/%Y %H:%M:%S'))) # read csv
	data.set_index(keys='datetime',inplace=True) # set date as an index
	data=data['load'] # convert the only important column to series
	data=data.reindex(pd.date_range(start=data.index.min(), end=data.index.max(),freq='1min'),fill_value=np.NaN) # add nans in case of missing entries
	data.index=data.index-pd.Timedelta(minutes=1) # subtract one minute to preserve consistency of dates
	if not data.index.is_monotonic_increasing: data.sort_index(inplace=True) # sort dates if necessary
	return data

# loads file
def load(path,idx='date',cols=[],dates=False):
	#if date_idx:data=pd.read_csv(path,header=0,sep=",", parse_dates=[idx],index_col=idx) # timestamp index
	#else: data=pd.read_csv(path,header=0,sep=",",index_col=idx) # non timestamp index
	data=pd.read_csv(path,header=0,sep=",",index_col=idx,parse_dates=dates) # non timestamp index
	if cols:data=data[cols] # extract only wanted columns
	return data

# saves data to csv
def save(data,path,idx=None):
	data.to_csv(path,header=True,index_label=idx)
	return

# saves dictionary containing {key:dataframe}
def save_dict(dic,path,idx=None):
	for key,value in dic.items():
		save(data=value,path=path+str(key)+'.csv',idx=idx) # save data
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

# splits weather data by columns & formats each part and outputs a dictionary with {keys}=={column names} 
def split_cols(data):
	return {col:data[col].unstack() for col in data.columns} # return a dictionary of dataframes each with values from only one column of original dataframe and key equal to column name	 

def split_cols_save(data,paths):
	for (col,path) in zip(data.columns,paths): # for each pair of a column and a path 
		save(data[col].unstack(),path) # save formatted column under the path
	return

# loads, concats & formats multiple weather files into one dataframe
def load_concat_w(paths,idx='timestamp',cols=['tempm','hum','pressurem'],dates=False):
	data=pd.concat([load(path=path,idx=idx,cols=cols,dates=dates) for path in paths]) # load and concat data
	data=data.where(data>-100,np.nan) # replace nonsensical values with nans
	data=data.resample(rule='30Min').mean() # insert nans for missing days
	return data

# combines minute time intervals into half-hour time intervals
def resample(data,freq=1440):
	data=cut(data,freq=freq) # remove incomplete first and last days
	data=data.resample(rule='30Min',closed='left',label='left').mean() # aggregate into 30min intervals
	values=data.name # get the series name
	data=data.to_frame() # convert to dataframe
	data['date']=pd.to_datetime(data.index.date) # create date column from index
	data['time']=data.index.strftime('%H%M') # create time column from index
	data=pd.pivot_table(data=data,index='date',columns='time',values=values) # pivot dataframe so that dates are index and times are columns
	return data

# flattens data, converts columns into a multiindex level
def d2s(data):
	if not isinstance(data, pd.Series):
			data=data.stack(dropna=False) # if not Series (already flat) then flatten
			data.index=pd.to_datetime(data.index.get_level_values(0).astype(str)+' '+ data.index.get_level_values(1),format='%Y-%m-%d %H%M')
	return data
	
# invert d2s operation
def s2d(data):
	if not isinstance(data, pd.DataFrame): data=data.unstack() # if not DataFrame (not flat) then unflatten
	return data
	
# remove incomplete first and last days
def cut(data,freq=1440):
	counts=data.fillna(value=0).resample('1D').count() # first replace nans to include in count then count
	days=counts[counts>=freq].index # complete days
	data=data[days.min().strftime('%Y-%m-%d'):days.max().strftime('%Y-%m-%d')] # preserve only complete days
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
def train_test(data, base=7,test_size=0.255): # in time series analysis order of samples usually matters, so no shuffling of samples
	split_idx=round_d((1-test_size)*len(data)) # calculate the index that splits dataset into train, test
	train,test =data[:split_idx],data[split_idx:] # split data into train & test sets
	return train,test

# returns number n that the (total-n) is rounded to base and (total-n)/total>test_size  
def round_rem(total,base=7,test_size=0.25):
	return total-round_u(total-(1-test_size)*total,base) if test_size>0 else total # calculate the index that splits dataset into train, test

# split data into n datasets (according to weekdays)
def split(data,nsplits=7): 
	return {i:data.iloc[i::nsplits] for i in range(nsplits)} # return as a dictionary {offset:data}
	
def load_merge(paths,idx='date',cols=[],dates=True):
	data=pd.concat([load(path, idx=idx,cols=cols, dates=dates) for path in paths], axis=0) # load all dataframes into one
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
