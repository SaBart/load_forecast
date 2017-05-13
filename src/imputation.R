library('imputeTS')
source('dataprep.R')

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data

data<-load(paste(data_dir,'data.csv', sep=''),idx='datetime') # load data set
data<-na.ma(x=data,k=2,weighting='linear')
save(data=data,path=paste(data_dir,'data_imp.csv', sep=''))
