library('imputeTS')
source('dataprep.R')

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data
wip_dir<-'C:/Users/SABA/Google Drive/mtsg/data/wip/' # directory containing data


data<-load(paste(data_dir,'data.csv', sep='')) # load data set
data_ts<-ts(data,frequency=1440) # build time series object

data<-na.kalman(x=data_ts,model='auto.arima',trace=TRUE)

save(data=data,path=paste(data_dir,'data_arima.csv', sep=''))



for (i in 0:5)
  data<-load(paste(wip_dir,'out_',i,'.csv', sep='')) # load train set
  data<-ts(data,frequency=1440)
  print(paste(i,'structTs',sep=''))
  structTs<-na.kalman(x=data,model='StructTS')
  print(paste(i,'arima',sep=''))
  arima<-na.kalman(x=data,model='auto.arima',trace=TRUE)
  print(paste(i,'seadec_structTS',sep=''))
  seadec_structTS<-na.seadec(x=data,algorithm='kalman',model='StructTS')
  print(paste(i,'seadec_arima',sep=''))
  seadec_arima<-na.seadec(x=data,algorithm='kalman',model='auto.arima',trace=TRUE)
  print(paste(i,'seasplit_structTS',sep=''))
  seasplit_structTS=na.seaaplit(x=data,algorithm='kalman',model='StructTS')
  print(paste(i,'seasplit_arima',sep=''))
  seasplit_arima<-na.seasplit(x=data,algorithm='kalman',model='auto.arima',trace=TRUE)
  save(data=structTS,path=paste(exp_dir,i,'_strucTS.csv',sep='')) # write results
  save(data=arima,path=paste(exp_dir,i,'_arima.csv',sep='')) # write results
  save(data=seadec_structTS,path=paste(exp_dir,i,'_seadec_strucTS.csv',sep='')) # write results
  save(data=seadec_arima,path=paste(exp_dir,i,'_seadec_arima.csv',sep='')) # write results
  save(data=seasplit_structTS,path=paste(exp_dir,i,'_seasplit_strucTS.csv',sep='')) # write results
  save(data=seasplit_arima,path=paste(exp_dir,i,'_seasplit_arima.csv',sep='')) # write results