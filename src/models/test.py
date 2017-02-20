'''
Created on 4 Dec 2016

@author: SABA
'''
import os
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd


#from numpy.distutils.conv_template import file
#from pandas.io.tests.parser import parse_dates


# PARAMETERS

time_ints=24 # number of time intervals
cols=np.arange(1,1+time_ints).tolist() # column names


# LOADING ENVIRONMENTAL DATA
dir_data='C:/Users/SABA/Google Drive/mtsg/data/homeB-all/homeB-environmental/'

temp_in = pd.DataFrame(columns=cols) # inside temperature
temp_out = pd.DataFrame(columns=cols) # outside temperature
hum_in = pd.DataFrame(columns=cols) # inside humidity
hum_out = pd.DataFrame(columns=cols) # outside humidity

for file in os.listdir(dir_data):
	if file.endswith(".csv"):
		env_raw =pd.read_csv(r'C:\Users\SABA\Google Drive\mtsg\data\homeB-all\homeB-power\2012-Apr-15.csv',header=None,sep=",",usecols=[0,1,2,3,4], names=['timestamp','temp_in','temp_out','hum_in','hum_out'],index_col=[0]) # load loads
		# add previous values for missing timestamps
		# maybe predict them from a couple of previus values?
		start=env_raw.index.min() # first value of new index
		idx=pd.Index(np.arange(start=start,stop=start+60*60*24,step=300)) # timestamps for the whole day, end at 23:59:59
		env_full=env_raw.reindex(idx, method='nearest') # fill missing with nearest values (predict them?)
		temp_in_hrs=load_full.as_matrix(columns=['temp_in']).reshape(-1,load_full.shape[0]//time_ints).average(axis=1) # averaging inside temperature
		temp_out_hrs=load_full.as_matrix(columns=['temp_out']).reshape(-1,load_full.shape[0]//time_ints).average(axis=1) # averaging outside temperature
		hum_in_hrs=load_full.as_matrix(columns=['hum_in']).reshape(-1,load_full.shape[0]//time_ints).average(axis=1) # averaging inside humidity
		hum_out_hrs=load_full.as_matrix(columns=['hum_out']).reshape(-1,load_full.shape[0]//time_ints).average(axis=1) # averaging outside humidity
		temp_in.loc[temp_in.shape[0]]=temp_in_hrs
		temp_out.loc[temp_out.shape[0]]=temp_out_hrs
		hum_in.loc[hum_in.shape[0]]=hum_in_hrs
		hum_out.loc[hum_out.shape[0]]=hum_out_hrs



plt.plot(loads.loc[80])
plt.show()

# MISSING DATA HISTOGRAM 

dir_data='C:/Users/SABA/Google Drive/mtsg/data/homeB-all/homeB-power/'
#loads_all = pd.DataFrame(columns=['time','load'])
loads_list=[]

for file in os.listdir(dir_data):
	if file.endswith(".csv"):
		load_raw =pd.read_csv(dir_data+file,header=None,sep=",",usecols=[0,1], parse_dates=True, names=['time','load'],index_col=[0]) # load loads
		loads_list.append(load_raw)
		
loads_all=pd.concat(loads_list)		


# LOADING LOAD DATA
dir_data='C:/Users/SABA/Google Drive/mtsg/data/homeB-all/homeB-power/'
loads = pd.DataFrame()

for file in os.listdir(dir_data):
	if file.endswith(".csv"):
		load_raw =pd.read_csv(dir_data+file,header=None,sep=",",usecols=[0,1], parse_dates=True, date_parser=(lambda x:pd.to_datetime(x,unit='s')), names=['time','load'],index_col=[0]) # load loads
		if load_raw.shape[0]/(60*60*24)<0.90: # discard file with more than 5% missing data
			continue
		# add previous values for missing timestamps
		# maybe predict them from a couple of previus values?
		start=load_raw.index.min() # first value of new index
		idx=pd.date_range(start=start,periods=86400,freq='1S') # timestamps for the whole day, end at 23:59:59
		load_full=load_raw.reindex(idx, method='nearest') # fill missing with nearest values (predict them?)
		load_hrs=load_full.resample('H').sum() # hourly aggregation of loads
		# build new entry for current day
		load_hrs['date']=pd.DatetimeIndex(load_hrs.index).date # new index
		load_hrs['time']=pd.DatetimeIndex(load_hrs.index).time # new columns
		loads=loads.append(load_hrs.pivot(index='date', columns='time', values='load')) # pivot & append new entry

loads.ix['2012-06-04'].plot(y='load')

plt.plot(np.arange(24),loads.ix[19].values[1:])
plt.plot(np.arange(24),load_hrs.tolist())
plt.show() 

# DATA PROCESSING METHODS

# loads data
def load_data(path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv'):
	load=pd.read_csv(path,header=0,sep=";",usecols=[0,1,2], names=['date','time','load'],dtype={'load': np.float64},na_values=['?'], parse_dates=['date'], date_parser=(lambda x:pd.to_datetime(x,format='%d/%m/%Y'))) # read csv
	load['hour']=pd.DatetimeIndex(load['time']).hour # new culumn for hours
	load['minute']=pd.DatetimeIndex(load['time']).minute # new column for minutes
	load=pd.pivot_table(load,index=['date','hour'], columns='minute', values='load') # pivot so that minutes are columns, date & hour multi-index and load is value
	load=load.applymap(lambda x:(x*1000)/60) # convert kW to Wh 
	load.sort_index(inplace=True) # sort entries (just in case)
	return load

# remove incomplete first and last days
def cut_data(data_temp,inplace=False):
	if (inplace):data=data_temp
	else: data=data_temp.copy()
	f,_=data.index.min() # first day
	l,_=data.index.max() # last day
	if (len(data.loc[f])<24): # if first day is incomplete
		data.drop(f,level=0,inplace=True) # drop the whole day
	if (len(data.loc[l])<24): # if last day is incomplete
		data.drop(l,level=0,inplace=True) # drop the whole day
	return data

# loads generator data
def load_gen_data(path='C:/Users/SABA/Google Drive/mtsg/code/generator/out/Electricity_Profile.csv'):
	load_raw =pd.read_csv(path,header=None,sep=",",usecols=[0], names=['load'],dtype={'load': np.float64}) # load loads
	load=load_raw.groupby(load_raw.index//60).sum() # hourly aggregation
	nb_days=load.shape[0]//24 # number of days
	load['hour']=pd.Series(np.concatenate([np.arange(1,25)]*nb_days)) # new column for pivoting
	load['day']=pd.Series(np.repeat(np.arange(1,nb_days+1), repeats=24)) # new column for pivoting
	load=load.pivot(index='day',columns='hour',values='load') # pivoting
	return load

# shifts data for time series forcasting
def shift_data(data,nb_shifts=1,shift=7):
	data_lagged={} # lagged dataframes for merging
	for i in range(0,nb_shifts+1): # for each time step
		data_lagged[i-nb_shifts]=data.shift(-i*shift) # add lagged dataframe
	res=pd.concat(data_lagged.values(),axis=1,join='inner',keys=data_lagged.keys()) # merge lagged dataframes	
	return res.dropna()

# separates data into training & testing sets & converts dataframes to numpy matrices 
def format_data(path='C:/Users/SABA/Google Drive/mtsg/code/generator/out/Electricity_Profile.csv', lag=1, test_size=0.2):
	data=shift_data(load_data(path),nb_shifts=lag)
	train, test =split_train_test(data, test_size)
	X_train,Y_train=split_X_Y(train)
	X_test,Y_test=split_X_Y(test)
	return X_train.as_matrix(), Y_train.as_matrix(), X_test.as_matrix(), Y_test.as_matrix()

# split data into X & Y
def split_X_Y(data):
	X=data.select(lambda x:x[0] not in [0], axis=1)
	Y=data[0]
	return X, Y

# split data into train & test sets
def split_train_test(data, test_size=0.2):
	from sklearn.model_selection import train_test_split
	train, test =train_test_split(data, test_size=test_size)
	return train,test

# split data into 7 datasets according to weekdays
def split_week_days(data):
	Sun=data.iloc[::7, :] # simulation starts on Sunday 1 of January
	Mon=data.iloc[1::7, :]
	Tue=data.iloc[2::7, :]
	Wen=data.iloc[3::7, :]
	Thu=data.iloc[4::7, :]
	Fri=data.iloc[5::7, :]
	Sat=data.iloc[6::7, :]
	return Sun, Mon, Tue, Wen, Thu, Fri, Sat
	
# create & train basic NN model
def create_model(nb_in=24, nb_out=24, nb_hidden=50, nb_epoch=200, batch_size=1, activation='relu', loss='mean_squared_error', optimizer='adam'):
	from keras.models import Sequential
	from keras.layers.core import Dense
	model = Sequential() # FFN
	model.add(Dense(nb_hidden, input_dim=nb_in,activation=activation)) # input & hidden layers
	#model.add(Dropout({{uniform(0, 1)}})) # randomly set a number of inputs to 0 to prevent overfitting
	model.add(Dense(nb_out)) # output layer
	model.compile(loss=loss, optimizer=optimizer) # assemble network	
	return model

# missing data statistics

# plot histogram of missing data
def nan_hist(data):
	import matplotlib.pyplot as plt
	nans=data.isnull().sum(axis=1) # count NaNs row-wise
	_,ax = plt.subplots() # get axis handle
	ax.set_yscale('log') # set logarithmic scale for y-values
	nans.hist(ax=ax,bins=60,bottom=1) # plot histogram of missing values, 
	plt.show()

# plot heatmap of missing data
def nan_heat(data):
	import matplotlib.pyplot as plt
	import seaborn as sns
	nans=data.isnull().sum(axis=1).unstack(fill_value=60) # count NaNs for each hour & 
	sns.heatmap(nans) # produce heatmap

# plot bars  for missing data
def nan_bar(data):
	import matplotlib.pyplot as plt
	nans=data.isnull().sum(axis=1) # count NaNs row-wise
	nans.plot(kind='bar') # plot histogram of missing values,

# generator data
# mlp optimisation
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

seed=0 # fix seed for reprodicibility
np.random.seed(seed)
path='C:/Users/SABA/Google Drive/mtsg/code/generator/out/Electricity_Profile.csv' # data path
model=MLPRegressor(solver='adam') # configure model
# grid parameter space
param_grid={'hidden_layer_sizes': [(10,), (25,), (50,), (75,),(100,),(125,),(150,)],
		'max_iter': [1000],
		'batch_size':[1,10,20,50,100,200]
		}

for i in range(1,6): # optimize number of time steps
	Sun,Mon,Tue,Wen,Thu,Fri,Sat=split_week_days(load_data(path))
	X,Y=split_X_Y(shift_data(Wen,nb_shifts=i,shift=1)) # prepare data
	best_model = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1) # configure grid search
	search_result = best_model.fit(X.as_matrix(), Y.as_matrix())
	print("Best: %f using %s" % (search_result.best_score_, search_result.best_params_))
	means = search_result.cv_results_['mean_test_score']
	stds = search_result.cv_results_['std_test_score']
	params = search_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))
	print()


# French data optimisation
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

seed=0 # fix seed for reprodicibility
np.random.seed(seed)
path='C:/Users/SABA/Google Drive/mtsg/data/household_power_consumption.csv' # data path

load=load_data(path) # load data
nan_hist(load)
nan_bar(load)
nan_heat(load)

# keep NANs
load_with_nans=load.apply(axis=1,func=(lambda x: np.nan if (x.isnull().sum()>0) else x.sum())).unstack() # custom sum function where any Nan in arguments gives Nan as result
#load_with_nans.isnull().equals(load.isnull().any(axis=1)) # check correctness of lambda function

model=MLPRegressor(solver='adam') # configure model
# grid parameter space
param_grid={'hidden_layer_sizes': [(10,), (25,), (50,), (75,),(100,),(125,),(150,)],
		'max_iter': [1000],
		'batch_size':[1,10,20,50,100,200]
		}

for i in range(1,6): # optimize number of time steps
	X,Y=split_X_Y(shift_data(load_with_nans,nb_shifts=i,shift=1).dropna()) # prepare data
	best_model = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1) # setting up grid search
	search_result = best_model.fit(X.as_matrix(), Y.as_matrix()) #  find best parameters
	print("Best: %f using %s" % (search_result.best_score_, search_result.best_params_))
	means = search_result.cv_results_['mean_test_score']
	stds = search_result.cv_results_['std_test_score']
	params = search_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))
	print()


# recurrent network optimisation

from tensorflow.contrib.learn import

import tensorflow.contrib.learn.python.learn as learn
from sklearn import datasets, metrics
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
feature_columns = learn.infer_real_valued_columns_from_input(iris.data)
classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3, feature_columns=feature_columns)

param_grid={'steps': [1000],
		'batch_size':[1,10,20,50,100,200]
		}

best_model = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1) # setting up grid search
search_result = best_model.fit(X.as_matrix(), Y.as_matrix()) #  find best parameters



classifier.fit(iris.data, iris.target, steps=200, batch_size=32)
iris_predictions = list(classifier.predict(iris.data, as_iterable=True))
score = metrics.accuracy_score(iris.target, iris_predictions)
print("Accuracy: %f" % score)








classifier.fit(training_set.data,training_set.target)


# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
									 y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples.
new_samples = np.array(
	[[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))





























from sklearn.metrics import mean_squared_error, make_scorer
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=create_model)
mse = make_scorer(mean_squared_error, multioutput='uniform_average')

#nb_hidden=[10,20,30,40,50,60,70,80,90,100] # domain for number of hidden neurons
nb_in=[X.shape[1]]
nb_out=[Y.shape[1]]
nb_hidden=[10,20,30]
param_grid={'nb_hidden':nb_hidden} # grid parameter space
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, scoring='r2')
grid_result = grid.fit(X.as_matrix(), Y.as_matrix())
# summarize results




model=create_model(X_train.shape[1],Y_train.shape[1])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2)
data_pred=model.predict(X_test)

plt.plot(data_pred[0])
plt.plot(Y_test[0])
plt.show()



# Use scikit-learn to grid search the number of neurons
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
# Function to create model, required for KerasClassifier
def create_model(neurons=1):
	# create model
	model = Sequential()
	model.add(Dense(neurons, input_dim=8, init='uniform', activation='linear', W_constraint=maxnorm(4)))
	model.add(Dropout(0.2))
	model.add(Dense(1, init='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load dataset
dataset = np.loadtxt('C:/Users/SABA/Google Drive/mtsg/code/load_forecast/src/models/pima-indians-diabetes.csv', delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=10, verbose=0)
# define the grid search parameters
neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))



from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim


for l in range(1,3):
	best_run, best_model = optim.minimize(model=model,
										  data=data(l),
										  algo=tpe.suggest,
										  max_evals=5,
										  trials=Trials())
	X_train, Y_train, X_test, Y_test = data()
	print("Evalutation of best performing model:")
	print(best_model.evaluate(X_test, Y_test))







	









def cut_data(data,steps=1):
	new_dict={}
	tuples=[] # multiIndex
	for col in data:
		new_dict[col]=data[col] # add old column
		for i in range(1,steps+1):
			new_dict['%s_lag%d' %(col,i)]=data[col].shift(i) # add shifted column
			tuples.append((i,col)) # new multiIndex entry 
	res=pd.DataFrame(new_dict,index=data.index) 
	return res.dropna(axis=0)
	
	
	
	
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3'],'C': ['C0', 'C1', 'C2', 'C3'],'D': ['D0', 'D1', 'D2', 'D3']}, index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],'B': ['B4', 'B5', 'B6', 'B7'],'C': ['C4', 'C5', 'C6', 'C7'],'D': ['D4', 'D5', 'D6', 'D7']},index=[4, 5, 6, 7])

df = pd.DataFrame({'A': [0,1,2,3],'B': [4,5,6,7],'C': [8,9,10,11],'D': [12,13,14,15]})



	
	for col in data:
		new_dict[col]=data[col] # add old column
		for i in range(1,steps+1):
			new_dict['%s_lag%d' %(col,i)]=data[col].shift(i) # add shifted column 
	res=pd.DataFrame(new_dict,index=data.index) 
	return res.dropna(axis=0)




load.plot(kind='bar')




dir_data='C:/Users/SABA/Google Drive/mtsg/data/homeB-all/homeB-power/'
loads = pd.DataFrame(columns=['day'] + cols)

for file in os.listdir(dir_data):
	if file.endswith(".csv"):
		load_raw =pd.read_csv(dir_data+file,header=None,sep=",",usecols=[0,1], parse_dates=True, date_parser=(lambda x:pd.to_datetime(x,unit='s')), names=['time','load'],index_col=[0]) # load loads
		if load_raw.shape[0]/(60*60*24)<0.90: # discard file with more than 5% missing data
			continue
		# add previous values for missing timestamps
		# maybe predict them from a couple of previus values?
		start=load_raw.index.min() # first value of new index
		idx=pd.date_range(start=start,periods=86400,freq='1S') # timestamps for the whole day, end at 23:59:59
		#idx=pd.Index(np.arange(start,start+60*60*24)) # timestamps for the whole day, end at 23:59:59
		load_full=load_raw.reindex(idx, method='nearest') # fill missing with nearest values (predict them?)
		load_full.resample('H').sum() # aggregate hours
		#load_hrs=load_full.as_matrix(columns=['load']).reshape(-1,load_full.shape[0]//time_ints).sum(axis=1) # aggregation for time intervals
		pd.concat(loads,)
		loads.loc[loads.shape[0]]=[start]+load_hrs.tolist()

loads.set_index('day', inplace=True)

 
 
 
 
 
load_dh=np.reshape(load_h,(365,-1),order='C')

X=np.arange(0,np.size(load_dh, axis=0))


seed=0
np.random.seed(seed)


#create model
print('creating model')

model = Sequential()
model.add(Dense(100, input_dim=1, init='uniform', activation='relu'))
model.add(Dense(24, init='uniform', activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# training
print('Training')

model.fit(X, load_dh, batch_size=10, nb_epoch=10000, verbose=2, validation_split=0.3, shuffle=True)

scores = model.evaluate(X, load_dh)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))



# Multilayer Perceptron to Predict International Airline Passengers (t+1, given t, t-1, t-2)
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = pd.read_csv('C:/Users/SABA/Google Drive/mtsg/code/load_forecast/src/models/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
# reshape dataset
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=2)
# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()







# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("C:/Users/SABA/Google Drive/mtsg/code/load_forecast/src/models/pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
