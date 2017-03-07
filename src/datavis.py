'''DATA VISUALIZATION'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# plot histogram of missing values
def nan_hist(data):
	nans=data.isnull().sum(axis=1) # count NaNs row-wise
	_,ax = plt.subplots() # get axis handle
	ax.set_yscale('log') # set logarithmic scale for y-values
	ax.set_xlim(xmin=-0.5, xmax=60.5)
	ax.set_ylim(ymin=0.5,ymax=nans.unique().max())
	ax.set_yticks(nans.value_counts())
	ax.set_yticklabels(nans.value_counts())
	nans.hist(ax=ax,bins=np.arange(62)-0.5,bottom=0.01) # plot histogram of missing values, 
	plt.show()
	return

# plot heatmap of missing values
def nan_heat(data):
	nans=data.isnull().sum(axis=1).unstack(fill_value=60) # count NaNs for each hour 
	sns.heatmap(nans) # produce heatmap
	return

# plot bars  for missing values
def nan_bar(data):
	nans=data.isnull().sum(axis=1) # count NaNs row-wise
	nans.plot(kind='bar') # plot histogram of missing values,
	return