'''WEATHER DATA DOWNLOADING'''

import pandas as pd
import dataprep as dp








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

data=load_weather(data_dir+'weather_3.csv',cols=['tempm','hum','pressurem','wspdm'])
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