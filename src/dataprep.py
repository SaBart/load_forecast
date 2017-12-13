# DATA PROCESSING METHODS

import os as os
import re as re
import numpy as np
import pandas as pd
import csv
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from json import loads
from urllib.request import urlopen
from itertools import product

# loads load profiles
def load_lp(path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv'):
	data=pd.read_csv(path,header=0,sep=";",usecols=[0,1,2], names=['date','time','load'],dtype={'load': np.float64},na_values=['?'], parse_dates={'datetime':['date','time']},date_parser=(lambda x:pd.to_datetime(x,format='%d/%m/%Y %H:%M:%S'))) # read csv
	data.set_index(keys='datetime',inplace=True) # set date as an index
	data=data['load'] # convert the only important column to series
	data=data.reindex(pd.date_range(start=data.index.min(), end=data.index.max(),freq='1min'),fill_value=np.NaN) # add nans in case of missing entries
	data.index=data.index-pd.Timedelta(minutes=1) # subtract one minute to preserve consistency of dates
	if not data.index.is_monotonic_increasing: data.sort_index(inplace=True) # sort dates if necessary
	return data

# laod dataport loads
def load_dp(path):
	data=pd.read_csv(path,header=0,sep=",", index_col='local_15min', usecols=['local_15min','use'],na_values=['?'], parse_dates=True) # read csv
	data.columns=['load'] # rename columns
	data.index.name='local' # rename index
	data=data['load'] # convert the only important column to series
	data=data.reindex(pd.date_range(start=data.index.min(), end=data.index.max(),freq='15min'),fill_value=np.NaN) # add nans in case of missing entries
	data.index=data.index-pd.Timedelta(minutes=15) # subtract one minute to preserve consistency of dates
	if not data.index.is_monotonic_increasing: data.sort_index(inplace=True) # sort dates if necessary
	return data


# loads file
def load(path,idx='date',cols=[],dates=False):
	data=pd.read_csv(path,header=0,sep=",",index_col=idx,parse_dates=dates) # non timestamp index
	if cols:data=data[cols] # extract only wanted columns
	if len(data.columns)==1:data=data.squeeze() # transform dataframe to series
	return data

# load multiple files and merge them
def load_merge(fol,paths,idx='date',cols=[],dates=False,axis=0):
	if axis==0: # merge on rows
		data=pd.concat([load(fol+path, idx=idx,cols=cols, dates=dates) for path in paths], axis=axis) # load all dataframes into one
	else: # merge on columns
		data=pd.concat([d2s(load(fol+path, idx=idx,cols=cols, dates=dates)) for path in paths], axis=axis,keys=[re.sub(pattern=r'\.csv', repl='',string=path) for path in paths]) # load all dataframes into one
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

def full(data):
	return d2s(data).isnull().sum()==0

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
				if not date is None: # format date
					o['timestamp']='{}-{}-{} {}:{}'.format(date['year'],date['mon'],date['mday'],date['hour'],date['min'])
				writer.writerow(o)
	return			

# splits weather data by columns & formats each part and outputs a dictionary with {keys}=={column names} 
def split_cols(data):
	return {col:data[col].unstack() for col in data.columns} # return a dictionary of dataframes each with values from only one column of original dataframe and key equal to column name	 

# splid dataframe and save each column as separate file
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
	data=cut(data=data,freq=freq) # remove incomplete first and last days
	if not data.empty: data=data.resample(rule='30Min',closed='left',label='left').mean() # aggregate into 30min intervals
	return s2d(data)

# flattens dataframes into series
def d2s(data):
	if not isinstance(data, pd.Series): # if not Series (already flat)
			if len(data.columns)==1: data=data.squeeze() # already flat
			else:
				data=data.stack(dropna=False).squeeze() # dataframe to series
				data.index=pd.to_datetime(data.index.get_level_values(0).astype(str)+' '+ data.index.get_level_values(1),format='%Y-%m-%d %H%M') # format datetime index
	return data
	
# invert d2s operation
def s2d(data):
	if not isinstance(data, pd.DataFrame): # if not DataFrame (not flat) then unflatten
		values=data.name # get the series name
		data=data.to_frame() # convert series to dataframe
		data['date']=pd.to_datetime(data.index.date) # create date column from index
		data['time']=data.index.strftime('%H%M') # create time column from index
		data=pd.pivot_table(data=data,index='date',columns='time',values=values) # pivot dataframe so that dates are index and times are columns
	return data
	
# remove incomplete first and last days
def cut(data,freq=1440):
	counts=data.groupby(data.index.date).count() # count non-nan values
	#counts=data.isnull().resample(rule='1D',closed='left',label='right').count() 
	days=counts[counts>=freq].index # complete days
	if len(days)>0:data=data[days.min().strftime('%Y-%m-%d'):days.max().strftime('%Y-%m-%d')] # preserve only complete days
	else: data=pd.DataFrame()
	return data

# merge forecasts for week-adjusted data
def merge_dir_files(pred_dir):
	merge_bases={re.split(r'_[0-9]',name)[0] for name in os.listdir(pred_dir) if len(re.split(r'_[0-9]',name))>1} # find all bases to merge
	for base in merge_bases: # for each base to merge
		paths=[base + '_' +str(i) +'.csv' for i in range(7)] # make relevant paths
		pred=load_merge(fol=pred_dir,paths=paths, idx='date', dates=True, axis=0) # load and merge partitions
		name ='wa,'+ base+'.csv' # create name for merged predictions
		save(data=pred, path=pred_dir+name, idx='date') # save merged predictions
		for path in paths: os.remove(path)
	return

# shifts data day-wise for time series forecasting
def add_day_lags(data,lags=[1],nolag='targets'):
	data_shifted={} # lagged dataframes for merging
	lags=[0]+lags # zero lag for target values
	for i in lags: # for each lag
		label=nolag # label for target values
		if i!=0:label='t-{}'.format(i) # labels for patterns
		data_shifted[label]=data.shift(i) # add lagged dataframe
	res=pd.concat(data_shifted.values(),axis=1,keys=data_shifted.keys()) # merge lagged dataframes
	return res.dropna() # TODO: handling missing values

# shifts data for time series forcasting
def add_lags(data,f_lags=[0],p_lags=[1],f_lab='Y',p_lab='X'):
	fut=None
	past=None
	data=d2s(data) # dataframe to series
	shifted={} # lagged dataframes for merging
	for i in f_lags: shifted[i]=data.shift(-i) # add lagged dataframe
	if shifted:
		fut=pd.concat(shifted.values(),axis=1,keys=shifted.keys()) # merge lagged dataframes
		fut.columns=pd.MultiIndex.from_product([[f_lab],fut.columns])
	shifted={} # lagged dataframes for merging
	for i in p_lags: shifted[-i]=data.shift(i) # add lagged dataframe
	if shifted: 
		past=pd.concat(shifted.values(),axis=1,keys=shifted.keys()) # merge lagged dataframes
		past.columns=pd.MultiIndex.from_product([[p_lab],past.columns])
	data=pd.concat([past,fut], axis=1).dropna()
	data=data.reindex_axis(sorted(data.columns), axis=1)
	return data
	
# construct dummy variables for days of the week and months
def idx2dmy(data): 
	# dummies for day of the week
	days=pd.get_dummies(data.index.dayofweek) #get dummies
	days.index=data.index # copy index
	days.columns=pd.MultiIndex.from_product([['day'],days.columns]) # construct columns
	# dummies for months
	months=pd.get_dummies(data.index.month) #get dummies
	months.index=data.index # copy index
	months.columns=pd.MultiIndex.from_product([['month'],months.columns])  # construct columns	 
	data=pd.concat([days,months,data],axis=1) # merge dataframes
	data=data.reindex_axis(sorted(data.columns), axis=1) # sort columns
	return data

# order timesteps from the oldest
def order(data):
	if not data.index.is_monotonic_increasing: data=data.sort_index() # sort dates if necessary
	return data
	
# split data into patterns & targets
def X_Y(data,Y_lab='Y'):
	Y=data[Y_lab] # targets
	X=data.drop(Y_lab,axis=1,level=0) # drop Ys from inputs
	X=X.reindex_axis(sorted(X.columns), axis=1) # order columns
	Y=Y.reindex_axis(sorted(Y.columns), axis=1) # order columns
	return X, Y

# split data into train & test sets
def train_test(data, base=7,test_size=0.255): # in time series analysis order of samples usually matters, so no shuffling of samples
	if test_size<1:
		split_idx=round_d((1-test_size)*len(data)) # calculate the index that splits dataset into train, test
	else:
		split_idx=round_d(len(data)-test_size) 	
	train,test =data[:split_idx],data[split_idx:] # split data into train & test sets
	return train,test

# returns number n that the (total-n) is rounded to base and (total-n)/total>test_size  
def round_rem(total,base=7,test_size=0.255):
	return total-round_u(total-(1-test_size)*total,base) if test_size>0 else total # calculate the index that splits dataset into train, test

# split data into n datasets (according to weekdays)
def split(data,nsplits=7): 
	return {i:data.iloc[i::nsplits] for i in range(nsplits)} # return as a dictionary {offset:data}
		
# rounds down to the nearest multiple of base
def round_d(x,base=7):
	return base*int(x/base)

# rounds up to the nearest multiple of base
def round_u(x,base=7):
	if x%base==0: result=x
	else: result=base*int((x+base)/base)
	return int(result)

# construct training & testing sets for time series cross validation
def tscv(data,test_size=369,batch=28):
	len_train=len(data)-test_size # initial train size
	tscv_iter=[(np.arange(i),i+np.arange(min(batch,len(data)-i))) for i in range(len_train,len(data),batch)] # construct the iterator, a list of tuples, each containing train & test indices
	tscv=[(data[:max(train)+1],data[min(test):max(test)+1]) for train,test in tscv_iter] # construct train and test sets according to iterator
	return tscv

# grid search parameter generator
def dol2lod(dol):
	return [{kw:arg for kw,arg in comb}for comb in product(*[[(kw,arg) for arg in args] for kw,args in dol.items()])] #dictionary of lists to list of dictionaries


# standardise data
def de_std(data,args=None):
	if args is None:
		data_flat=d2s(data) # flatten DataFrame into a Series
		mean=data_flat.mean() # get mean
		std=data_flat.std() # get std
	else:
		mean,std=args
	return (data-mean)/std,(mean,std)

# invert standardisation
def re_std(data,args):
	mean,std=args
	return (data*std)+mean

# subtract average
def de_mean(data,avg_days=None):
	if avg_days is None:
		avg_days=data.groupby(by=data.index.weekday).mean() # average across weekdays
	#data=data.groupby(by=data.index.weekday).transform(lambda x: x-x.mean()) # subtract average days
	data=data.groupby(by=data.index.weekday).apply(lambda x: x-avg_days.loc[x.name])
	return data,avg_days
	
# add average
def re_mean(data,avg_days):	
	return data.groupby(by=data.index.weekday).apply(lambda x: x+avg_days.loc[x.name])
	
# seseasonalisation
def de_seas(data,seas=None,window=7):
	if seas is None:
		pandas2ri.activate() # activate connection
		stats=importr('stats') # forecast package
		ts=ro.r.ts # R time series
		data_ts=ts(d2s(data),frequency=48) # add a new day from test set to the current train set
		dec=stats.stl(data_ts,s_window=7,robust=True).rx2('time.series') # decompose time series
		#trend=pandas2ri.ri2py(dec.rx(True,'trend')) # convert R object to pandas dataframe
		seas=pd.Series(pandas2ri.ri2py(dec.rx(True,'seasonal')),index=d2s(data).index) # convert R object to pandas series
		seas.name='' # remove name
		seas=s2d(seas) # convert series do dataframe
		#rem=pandas2ri.ri2py(dec.rx(True,'remainder')) # convert R object to pandas dataframe
		data=data-seas # de-seasonalize data
		pandas2ri.deactivate() # deactivate connection
	else:
		s=seas.tail(len(data)) # get seasonality of previous batch
		s.index=data.index # change index to alogn dataframes
		data=data-s # de-seasonalize data
	return data,seas

# reseasonalisation
def re_seas(data,seas):
	s=seas.tail(len(data)) # get seasonality of previous batch
	s.index=data.index	# change index to align dataframes
	return data+s # re-seasonalize data

	