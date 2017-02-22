'''WEATHER DATA DOWNLOADING'''

import pandas as pd
import dataprep as dp
import csv
from json import loads
from urllib.request import urlopen

# downloads one day worth of weather data
def download_day(date):
	request='http://api.wunderground.com/api/8cc2ef7d44313e70/history_'+date+'/q/ORY.json' # construct request
	stream=urlopen(request) # open stream
	data=stream.read().decode('utf-8') # read data from stream
	stream.close() # close stream
	return loads(data) # parse json data

# download & save raw weather data for specified dates
def download_weather(dates,directory,name):
	with open(directory+name, 'w', newline='') as csv_file: # open stream
		obs=[o for o in download_day(dates[0])['history']['observations'] if o['metar'].startswith('METAR')] # get all METAR format observations
		obs[0].pop('utcdate',None) # remove utcdate entry
		obs[0].pop('date',None) # remove date entry
		writer=csv.DictWriter(csv_file,fieldnames=obs[0].keys()) # set up dictionary writer with proper fields 
		writer.writeheader() # write header
		for date in dates: # for each date
			obs=[o for o in download_day(date)['history']['observations'] if o['metar'].startswith('METAR')] # get all METAR format observations
			for o in obs: # for each observation
				o.pop('utcdate',None) # remove utcdate entry
				date=o.pop('date',None) # get date entry (it is a dictionary, extracting values will follow)
				if not date is None:
					o['timestamp']='{}-{}-{} {}:{}'.format(date['year'],date['mon'],date['mday'],date['hour'],date['min'])
				writer.writerow(o)
	return			

# loads & formats raw weather data  
def load_weather(path='C:/Users/SABA/Google Drive/mtsg/data/weather_2.csv',index='timestamp',cols=['tempm','hum','pressurem','wspdm']):
	data=pd.read_csv(path,header=0,sep=",",usecols=[index]+cols, parse_dates=[index],index_col=index) # read csv
	data=data.resample('H').mean() # average values across hours
	data['date']=pd.DatetimeIndex(data.index).normalize() # new column for dates
	data['hour']=pd.DatetimeIndex(data.index).hour # new column for hours
	data=pd.pivot_table(data,index=['date','hour']) # pivot so that minutes are columns, date & hour multi-index and load is value
	return data

# splits weather data by columns, formats and saves each part 
def split_save(data,paths):
	for (col,path) in zip(data.columns,paths): # for each pair of a column and a path 
		dp.save(data[col].unstack(),path) # save formatted column under the path
	return	 



data_dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data
load_raw=dp.load(data_dir+'household_power_consumption.csv') # load data
load_raw=dp.cut(load_raw) # remove leading & trailing Nans
targets=dp.m2h(load_raw,nan='keep') # minutes to hours, preserving nans

dates=pd.DatetimeIndex(targets.index).strftime('%Y%m%d')[:400] # reformat dates
download_weather(dates, data_dir, 'weather_1.csv')
dates=pd.DatetimeIndex(targets.index).strftime('%Y%m%d')[400:800] # reformat dates
download_weather(dates, data_dir, 'weather_2.csv')
dates=pd.DatetimeIndex(targets.index).strftime('%Y%m%d')[800:1200] # reformat dates
download_weather(dates, data_dir, 'weather_3.csv')
dates=pd.DatetimeIndex(targets.index).strftime('%Y%m%d')[1200:] # reformat dates
download_weather(dates, data_dir, 'weather_4.csv')

data=load_weather(data_dir+'weather_2.csv',cols=['tempm','hum','pressurem','wspdm'])
split_save(data,[data_dir+col+'.csv' for col in data])

date='20061216'
data=pd.DataFrame(data=None,index=pd.MultiIndex.from_product([dates,range(24)], names=['date','hour']),columns=['temp','psr','hum','prc','wspd','wdir'])
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
				wdir = o['wdird'] # wind direction [deg]s