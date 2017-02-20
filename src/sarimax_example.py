
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose


# In[2]:

df = pd.read_csv('C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/portland-oregon-average-monthly-.csv', index_col=0)
df.index.name=None
df.reset_index(inplace=True)
df.drop(df.index[114], inplace=True)


# In[3]:

start = datetime.datetime.strptime("1973-01-01", "%Y-%m-%d")
date_list = [start + relativedelta(months=x) for x in range(0,114)]
df['index'] =date_list
df.set_index(['index'], inplace=True)
df.index.name=None


# In[4]:

df.columns= ['riders']
df['riders'] = df.riders.apply(lambda x: int(x)*100)


# In[5]:

df.riders.plot(figsize=(12,8), title= 'Monthly Ridership', fontsize=14)
plt.savefig('month_ridership.png', bbox_inches='tight')


# In[6]:

decomposition = seasonal_decompose(df.riders, freq=12)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)


# In[7]:

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# In[8]:

test_stationarity(df.riders)


# In[9]:

df.riders_log= df.riders.apply(lambda x: np.log(x))  
test_stationarity(df.riders_log)


# In[10]:

df['first_difference'] = df.riders - df.riders.shift(1)  
test_stationarity(df.first_difference.dropna(inplace=False))


# In[11]:

df['log_first_difference'] = df.riders_log - df.riders_log.shift(1)  
test_stationarity(df.log_first_difference.dropna(inplace=False))


# In[12]:

df['seasonal_difference'] = df.riders - df.riders.shift(12)  
test_stationarity(df.seasonal_difference.dropna(inplace=False))


# In[13]:

df['log_seasonal_difference'] = df.riders_log - df.riders_log.shift(12)  
test_stationarity(df.log_seasonal_difference.dropna(inplace=False))


# In[14]:

df['seasonal_first_difference'] = df.first_difference - df.first_difference.shift(12)  
test_stationarity(df.seasonal_first_difference.dropna(inplace=False))


# In[15]:

df['log_seasonal_first_difference'] = df.log_first_difference - df.log_first_difference.shift(12)  
test_stationarity(df.log_seasonal_first_difference.dropna(inplace=False))


# In[16]:

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df.seasonal_first_difference.iloc[13:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df.seasonal_first_difference.iloc[13:], lags=40, ax=ax2)


# In[17]:

mod = sm.tsa.statespace.SARIMAX(df.riders, trend='n', order=(0,1,0), seasonal_order=(0,1,1,12))
results = mod.fit()
print(results.summary())


# In[27]:

mod = sm.tsa.statespace.SARIMAX(df.riders, trend='n', order=(0,1,0), seasonal_order=(1,1,1,12))
results = mod.fit()
print(results.summary())


# In[19]:

df['forecast'] = results.predict(end= 114, dynamic= True)  
df[['riders', 'forecast']].plot(figsize=(12, 8)) 
plt.savefig('ts_df_predict.png', bbox_inches='tight')


# In[20]:

npredict =df.riders['1982'].shape[0]
fig, ax = plt.subplots(figsize=(12,6))
npre = 12
ax.set(title='Ridership', xlabel='Date', ylabel='Riders')
ax.plot(df.index[-npredict-npre+1:], df.ix[-npredict-npre+1:, 'riders'], 'o', label='Observed')
ax.plot(df.index[-npredict-npre+1:], df.ix[-npredict-npre+1:, 'forecast'], 'g', label='Dynamic forecast')
legend = ax.legend(loc='lower right')
legend.get_frame().set_facecolor('w')
plt.savefig('ts_predict_compare.png', bbox_inches='tight')


# In[21]:

start = datetime.datetime.strptime("1982-07-01", "%Y-%m-%d")
date_list = [start + relativedelta(months=x) for x in range(0,12)]
future = pd.DataFrame(index=date_list, columns= df.columns)
df = pd.concat([df, future])


# In[30]:

df['forecast'] = results.predict(start = 114, end = 125, dynamic= True)  
df[['riders', 'forecast']].ix[-24:].plot(figsize=(12, 8)) 
plt.savefig('ts_predict_future.png', bbox_inches='tight')


# In[25]:

## A little extra I was doing to try and include an exogenous variable for number of weekdays in each month.
## Thinking that people take public transportation more during the week, a count of weekdays in each month could help explain some of the variance.  

start = datetime.datetime.strptime("1973-01-01", "%Y-%m-%d")
moving = start
d = {}
year =0
month =0
while moving < datetime.datetime(1982,7,1):
#     print moving
    if moving.year == year:
        if moving.month == month:
            if moving.weekday() < 5:
                d[str(moving.year)+"-"+ str(moving.month)] += 1
        else:
            d[str(moving.year)+"-"+ str(moving.month)]=0
            if moving.weekday() < 5:
                d[str(moving.year)+"-"+ str(moving.month)] += 1
    else:
#         d[moving.year] = {}
        d[str(moving.year)+"-"+ str(moving.month)]=0
        if moving.weekday() < 5:
            d[str(moving.year)+"-"+ str(moving.month)] += 1


    year = moving.year
    month = moving.month
    moving += datetime.timedelta(days=1)
df_dow = pd.DataFrame(d.items(), columns=['Month', 'DateValue'])
df_dow.Month = pd.to_datetime(df_dow.Month)
df_dow.sort('Month', inplace=True)

def holiday_adj(x):
    if x['Month'].month==1:
        x['DateValue'] -=1
        return x['DateValue'] 
    elif x['Month'].month==2:
        x['DateValue'] -=1
        return x['DateValue']
    elif x['Month'].month==5:
        x['DateValue'] -=1
        return x['DateValue']
    elif x['Month'].month==7:
        x['DateValue'] -=1
        return x['DateValue']
    elif x['Month'].month==9:
        x['DateValue'] -=1
        return x['DateValue']
    elif x['Month'].month==10:
        x['DateValue'] -=1
        return x['DateValue']    
    elif x['Month'].month==11:
        x['DateValue'] -=3   
        return x['DateValue']
    elif x['Month'].month==12:
        x['DateValue'] -=2
        return x['DateValue']

    else:
        return x['DateValue']
    
df_dow['days'] = df_dow.apply(holiday_adj, axis=1)
df_dow.set_index('Month', inplace=True)
df_dow.index.name = None

