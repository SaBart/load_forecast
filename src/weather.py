'''WEATHER DATA DOWNLOADING'''

import pandas as pd
import dataprep as dp
import csv
from json import loads
from urllib.request import urlopen

def download_day(date):
	request='http://api.wunderground.com/api/8cc2ef7d44313e70/history_'+date+'/q/ORY.json' # construct request
	stream=urlopen(request) # open stream
	data=stream.read().decode('utf-8') # read data from stream
	stream.close() # close stream
	return loads(data) # parse json data

def download_weather(dates,directory,name):
	with open(directory+name, 'w', newline='') as csv_file: # open stream
		obs=[o for o in download_day(dates[0])['history']['observations'] if o['metar'].startswith('METAR')] # get all METAR format observations
		writer=csv.DictWriter(csv_file,fieldnames=obs[0].keys()) # set up dictionary writer with proper fields 
		writer.writeheader() # write header
		for date in dates: # for each date
			obs=[o for o in download_day(date)['history']['observations'] if o['metar'].startswith('METAR')] # get all METAR format observations
			for o in obs: # for each observation
				writer.writerow(o)
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