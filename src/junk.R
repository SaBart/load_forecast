library(forecast)
library(imputeTS)

ets=function(train,test,hor=1,batch=7,freq=7){
  
}














train=read.csv('C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/train.csv',header=TRUE,sep=',',dec='.')
test=read.csv('C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/test.csv',header=TRUE,sep=',',dec='.')


batch=7
hor=24
train_ts=ts(train,frequency=24)
test_ts=ts(test,frequency=24)





train_ts=ts(train[[2]],frequency=findfrequency(train[[2]]))
test_ts=ts(test[[2]],frequency=findfrequency(test[[2]]))
fit_train_ts=ets(train_ts)
fit_test_ts=ets(test_ts,model=fit_train_ts)
train_ts_pred=fitted(fit_train_ts)
test_ts_pred=fitted(fit_test_ts)
ts.plot(train_ts,train_ts_pred,col=c('black','red'),lty=c(5,1))
ts.plot(test_ts,test_ts_pred,col=c('black','red'),lty=c(5,1))


train_xts=xts(train[[10]],as.POSIXct(train[["date"]]))
test_xts=xts(test[[10]],order.by=as.POSIXct(test$date))
fit_train_xts=ets(train_xts)
fit_test_xts=ets(test_xts,model=fit_train_xts)
train_xts_pred=fitted(fit_train_xts)
test_xts_pred=fitted(fit_test_xts)
ts.plot(train_xts,train_xts_pred,col=c('black','red'),lty=c(5,1))
ts.plot(test_xts,test_xts_pred,col=c('black','red'),lty=c(5,1))

# Multi-step, re-estimation
h <- 5
train <- window(hsales,end=1989.99)
test <- window(hsales,start=1990)
n <- length(test) - h + 1
fit <- auto.arima(train)
order <- arimaorder(fit)
fcmat <- matrix(0, nrow=n, ncol=h)
for(i in 1:n)
{  
  x <- window(hsales, end=1989.99 + (i-1)/12)
  refit <- Arima(x, order=order[1:3], seasonal=order[4:6])
  fcmat[i,] <- forecast(refit, h=h)$mean
}



train_ts=ts(train[[2]],frequency=findfrequency(train[[2]]))
test_ts=ts(test[[2]],frequency=findfrequency(test[[2]]))
fit_train_ts=ets(train_ts)
fit_test_ts=ets(test_ts,model=fit_train_ts)
train_ts_pred=fitted(fit_train_ts)
test_ts_pred=fitted(fit_test_ts)
ts.plot(train_ts,train_ts_pred,col=c('black','red'),lty=c(5,1))
ts.plot(test_ts,test_ts_pred,col=c('black','red'),lty=c(5,1))


train_xts=xts(train[[10]],as.POSIXct(train[["date"]]))
test_xts=xts(test[[10]],order.by=as.POSIXct(test$date))
fit_train_xts=ets(train_xts)
fit_test_xts=ets(test_xts,model=fit_train_xts)
train_xts_pred=fitted(fit_train_xts)
test_xts_pred=fitted(fit_test_xts)
ts.plot(train_xts,train_xts_pred,col=c('black','red'),lty=c(5,1))
ts.plot(test_xts,test_xts_pred,col=c('black','red'),lty=c(5,1))

# Multi-step, re-estimation
h <- 5
train <- window(hsales,end=1989.99)
test <- window(hsales,start=1990)
n <- length(test) - h + 1
fit <- auto.arima(train)
order <- arimaorder(fit)
fcmat <- matrix(0, nrow=n, ncol=h)
for(i in 1:n)
{  
  x <- window(hsales, end=1989.99 + (i-1)/12)
  refit <- Arima(x, order=order[1:3], seasonal=order[4:6])
  fcmat[i,] <- forecast(refit, h=h)$mean
}





set.seed(1)
library(lubridate)
index <- ISOdatetime(2010,1,1,0,0,0)+1:8759*60*60
month <- month(index)
hour <- hour(index)
usage <- 1000+10*rnorm(length(index))-25*(month-6)^2-(hour-12)^2
usage <- ts(usage,frequency=24)

#Create monthly dummies.  Add other xvars to this matrix
xreg <- model.matrix(~as.factor(month))[,2:12]
colnames(xreg) <- c('Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')




date_train<-train$date # extract date column from train set
date_test<-test$date # extract date column from test set
train<-train[ , !names(train) %in% c('date')] # drop date column from train set
test<-test[ , !names(test) %in% c('date')] # drop date column from test set


date_train<-train$date # extract date column from train set
date_test<-test$date # extract date column from test set
train<-train[ , !names(train) %in% c('date')] # drop date column from train set
test<-test[ , !names(test) %in% c('date')] # drop date column from test set

rownames(test_pred_hw)<-date_test # set "index"


date_train<-train$date # extract date column from train set
date_test<-test$date # extract date column from test set
train<-train[ , !names(train) %in% c('date')] # drop date column from train set
test<-test[ , !names(test) %in% c('date')] # drop date column from test set

rownames(test_pred_vw)<-date_test # set "index"



pop_col=function(data,col){ # removes and returns column from dataframe
  poped_col<-data$col # extract column from dataframe
  data<<-data[ , !names(data) %in% c(col)] # drop column from dataframe
  return(poped_col)
}















# NO EXTERNAL REGRESSORS

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/results/' # directory for the results of experiments

train<-load(paste(data_dir,'train_full.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set

# horizontal predictions
test_pred_h<-arima_h(train,test,batch=28,freq=48) # predict values
save(data=test_pred_h,path=paste(exp_dir,'arima_h','.csv',sep='')) # write results

# vertical predictions
test_pred_v<-arima_v(train,test,batch=28,freq=7) # predict values
save(data=test_pred_v,path=paste(exp_dir,'arima_v','.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hwi<-arima_h(train,test,batch=4,freq=48) # horizontal predictions for this day
  save(data=test_pred_hwi,path=paste(exp_dir,'arima_h_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vwi<-arima_v(train,test,batch=4,freq=52) # vertical predictions for this day
  save(data=test_pred_vwi,path=paste(exp_dir,'arima_v_',i,'.csv',sep='')) # write results
}

# FOURIER EXTERNAL REGRESSORS

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set

# horizontal predictions
K<-f_ords(train,freq=365.25*24,freqs=c(24,7*24),max_order=12) # find best fourier coefficients
# K=c(10,6)
test_pred_hf<-arima_h(train,test,batch=28,freq=24,f_K=K) # horizontal prediction
save(data=test_pred_hf,path=paste(exp_dir,'arima_hf.csv',sep='')) # write results

# vertical predictions
K<-f_ords(train,freq=365.25,freqs=c(7),max_order=3) # find best fourier coefficients
# K=c(2)
test_pred_vf<-arima_v(train,test,batch=28,freq=7,f_K=K) # horizontal prediction
save(data=test_pred_vf,path=paste(exp_dir,'arima_vf.csv',sep='')) # write results


# WEATHER EXTERNAL REGRESSORS

train<-load(paste(dir,'train.csv', sep='')) # load train set
test<-load(paste(dir,'test.csv', sep='')) # load test set
wxregs_train<-lapply(list('train_tempm.csv','train_hum.csv','train_wspdm.csv','train_pressurem.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for train set
wxregs_test<-lapply(list('test_tempm.csv','test_hum.csv','test_wspdm.csv','test_pressurem.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for test set

# horizontal predictions
test_pred_hw<-arima_h(train,test,batch=28,freq=24,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # horizontal prediction
save(data=test_pred_hw,path=paste(exp_dir,'arima_hw.csv',sep='')) # write results

# vertical predictions
test_pred_vw<-arima_h(train,test,batch=28,freq=7,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # vertical prediction
save(data=test_pred_vw,path=paste(exp_dir,'arima_vw.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  wxregs_train<-lapply(list('train_tempm_','train_hum_','train_wspdm_','train_pressurem_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for train set
  wxregs_test<-lapply(list('test_tempm_','test_hum_','test_wspdm_','test_pressurem_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for test set
  test_pred_hwi<-arima_h(train,test,batch=4,freq=24,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # horizontal predictions for this day
  save(data=test_pred_hwi,path=paste(exp_dir,'arima_hw_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  wxregs_train<-lapply(list('train_tempm_','train_hum_','train_wspdm_','train_pressurem_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for train set
  wxregs_test<-lapply(list('test_tempm_','test_hum_','test_wspdm_','test_pressurem_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for test set
  test_pred_vwi<-arima_v(train,test,batch=4,freq=4,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # horizontal predictions for this day
  save(data=test_pred_hwi,path=paste(exp_dir,'arima_vw_',i,'.csv',sep='')) # write results
}


# FOURIER & WEATHER EXTERNAL REGRESSORS

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set
wxregs_train<-lapply(list('train_tempm.csv','train_hum.csv','train_wspdm.csv','train_pressurem.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for train set
wxregs_test<-lapply(list('test_tempm.csv','test_hum.csv','test_wspdm.csv','test_pressurem.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for test set

# horizontal predictions
# K<-f_ords(train,freq=365.25*24,freqs=c(24,7*24),max_order=10) # find best fourier coefficients
K=c(10,6)
test_pred_hfw<-arima_h(train,test,batch=28,freq=24,f_K=K,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # horizontal prediction
save(data=test_pred_hfw,path=paste(exp_dir,'arima_hfw.csv',sep='')) # write results

# vertical predictions
K<-f_ords(train,freq=365.25,freqs=c(7),max_order=3) # find best fourier coefficients
K=c(10)
test_pred_vfw<-arima_h(train,test,batch=28,freq=7,f_K=K,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # vertical prediction
save(data=test_pred_vfw,path=paste(exp_dir,'arima_vfw.csv',sep='')) # write results





# DEC

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/results/' # directory for the results of experiments

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set

# horizontal predictions
test_pred_h<-arima_h(train,test,batch=28,freq=48,dec=TRUE) # predict values
save(data=test_pred_h,path=paste(exp_dir,'dec,arima_h','.csv',sep='')) # write results

# vertical predictions
test_pred_v<-arima_v(train,test,batch=28,freq=7,dec=TRUE) # predict values
save(data=test_pred_v,path=paste(exp_dir,'dec,arima_v','.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hwi<-arima_h(train,test,batch=4,freq=48,dec=TRUE) # horizontal predictions for this day
  save(data=test_pred_hwi,path=paste(exp_dir,'dec,arima_h_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vwi<-arima_v(train,test,batch=4,freq=52,dec=TRUE) # vertical predictions for this day
  save(data=test_pred_vwi,path=paste(exp_dir,'dec,arima_v_',i,'.csv',sep='')) # write results
}

# DEC & FOURIER EXTERNAL REGRESSORS

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/results/' # directory for the results of experiments

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set

# horizontal predictions
K<-f_ords(train,freq=365.25*48,freqs=c(48,7*48),max_order=24) # find best fourier coefficients
# K=c(10,6)
test_pred_hf<-arima_h(train,test,batch=28,freq=365.25*48,freqs=c(48,7*48),f_K=dec_K_h,dec=TRUE) # horizontal prediction
save(data=test_pred_hf,path=paste(exp_dir,'dec,freg,arima_h.csv',sep='')) # write results

# vertical predictions
test_pred_vf<-arima_v(train,test,batch=28,freq=365.25,freqs=c(48,7*48),f_K=dec_K_v,dec=TRUE) # horizontal prediction
save(data=test_pred_vf,path=paste(exp_dir,'dec,freg,arima_v.csv',sep='')) # write results


# DEC & WEATHER EXTERNAL REGRESSORS

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/results/' # directory for the results of experiments

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set
wxregs_train<-lapply(list('tempm_train.csv','hum_train.csv','pressurem_train.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for train set
wxregs_test<-lapply(list('tempm_test.csv','hum_test.csv','pressurem_test.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for test set

# horizontal predictions
test_pred_hw<-arima_h(train,test,batch=28,freq=48,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE) # horizontal prediction
save(data=test_pred_hw,path=paste(exp_dir,'dec,wreg,arima_h.csv',sep='')) # write results

# vertical predictions
test_pred_vw<-arima_v(train,test,batch=28,freq=7,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE) # vertical prediction
save(data=test_pred_vw,path=paste(exp_dir,'dec,wreg,arima_v.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  wxregs_train<-lapply(list('tempm_train_','hum_train_','pressurem_train_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for train set
  wxregs_test<-lapply(list('tempm_test_','hum_test_','pressurem_test_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for test set
  test_pred_hwi<-arima_h(train,test,batch=4,freq=48,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE,box_cox = TRUE) # horizontal predictions for this day
  save(data=test_pred_hwi,path=paste(exp_dir,'dec,bc,wreg,arima_h_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  wxregs_train<-lapply(list('tempm_train_','hum_train_','pressurem_train_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for train set
  wxregs_test<-lapply(list('tempm_test_','hum_test_','pressurem_test_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for test set
  test_pred_vwi<-arima_v(train,test,batch=4,freq=52,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE,box_cox = TRUE) # horizontal predictions for this day
  save(data=test_pred_vwi,path=paste(exp_dir,'dec,wreg,arima_v_',i,'.csv',sep='')) # write results
}


# DEC & FOURIER & WEATHER EXTERNAL REGRESSORS

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/results/' # directory for the results of experiments

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set
wxregs_train<-lapply(list('tempm_train.csv','hum_train.csv','pressurem_train.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for train set
wxregs_test<-lapply(list('tempm_test.csv','hum_test.csv','pressurem_test.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for test set

# horizontal predictions
test_pred_hfw<-arima_h(train,test,batch=28,freq=365.25*48,freqs=c(48,7*48),f_K=dec_K_h,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE) # horizontal prediction
save(data=test_pred_hfw,path=paste(exp_dir,'dec,fwreg,arima_h.csv',sep='')) # write results

# vertical predictions
test_pred_vfw<-arima_v(train,test,batch=28,freq=365.25,freqs=c(7),f_K=dec_K_v,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE) # vertical prediction
save(data=test_pred_vfw,path=paste(exp_dir,'dec,fwreg,arima_v.csv',sep='')) # write results





# DEC & BOX-COX

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/results/' # directory for the results of experiments

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set

# horizontal predictions
test_pred_h<-arima_h(train,test,batch=28,freq=48,dec=TRUE,box_cox = TRUE) # predict values
save(data=test_pred_h,path=paste(exp_dir,'dec,bc,arima_h','.csv',sep='')) # write results

# vertical predictions
test_pred_v<-arima_v(train,test,batch=28,freq=7,dec=TRUE,box_cox = TRUE) # predict values
save(data=test_pred_v,path=paste(exp_dir,'dec,bc,arima_v','.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hwi<-arima_h(train,test,batch=4,freq=48,dec=TRUE,box_cox = TRUE) # horizontal predictions for this day
  save(data=test_pred_hwi,path=paste(exp_dir,'dec,bc,arima_h_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vwi<-arima_v(train,test,batch=4,freq=52,dec=TRUE,box_cox = TRUE) # vertical predictions for this day
  save(data=test_pred_vwi,path=paste(exp_dir,'dec,bc,arima_v_',i,'.csv',sep='')) # write results
}

# DEC & BOX-COX & FOURIER EXTERNAL REGRESSORS

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/results/' # directory for the results of experiments

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set

# horizontal predictions
K<-f_ords(train,freq=365.25*48,freqs=c(48,7*48),max_order=24) # find best fourier coefficients
# K=c(10,6)
test_pred_hf<-arima_h(train,test,batch=28,freq=365.25*48,freqs=c(48,7*48),f_K=dec_bc_K_h,dec=TRUE,box_cox=TRUE) # horizontal prediction
save(data=test_pred_hf,path=paste(exp_dir,'dec,bc,freg,arima_h.csv',sep='')) # write results

# vertical predictions
test_pred_vf<-arima_v(train,test,batch=28,freq=365.25,freqs=c(7),f_K=dec_bc_K_v,dec=TRUE,box_cox=TRUE) # horizontal prediction
save(data=test_pred_vf,path=paste(exp_dir,'dec,bc,freg,arima_v.csv',sep='')) # write results


# DEC & BOX-COX & WEATHER EXTERNAL REGRESSORS

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/results/' # directory for the results of experiments

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set
wxregs_train<-lapply(list('tempm_train.csv','hum_train.csv','pressurem_train.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for train set
wxregs_test<-lapply(list('tempm_test.csv','hum_test.csv','pressurem_test.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for test set

# horizontal predictions
test_pred_hw<-arima_h(train,test,batch=28,freq=48,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE,box_cox = TRUE) # horizontal prediction
save(data=test_pred_hw,path=paste(exp_dir,'dec,bc,wreg,arima_h.csv',sep='')) # write results

# vertical predictions
test_pred_vw<-arima_v(train,test,batch=28,freq=7,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE,box_cox = TRUE) # vertical prediction
save(data=test_pred_vw,path=paste(exp_dir,'dec,bc,wreg,arima_v.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  wxregs_train<-lapply(list('tempm_train_','hum_train_','pressurem_train_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for train set
  wxregs_test<-lapply(list('tempm_test_','hum_test_','pressurem_test_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for test set
  test_pred_hwi<-arima_h(train,test,batch=4,freq=48,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE,box_cox = TRUE) # horizontal predictions for this day
  save(data=test_pred_hwi,path=paste(exp_dir,'dec,bc,wreg,arima_h_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  wxregs_train<-lapply(list('tempm_train_','hum_train_','pressurem_train_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for train set
  wxregs_test<-lapply(list('tempm_test_','hum_test_','pressurem_test_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for test set
  test_pred_vwi<-arima_v(train,test,batch=4,freq=52,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE,box_cox = TRUE) # horizontal predictions for this day
  save(data=test_pred_vwi,path=paste(exp_dir,'dec,bc,wreg,arima_v_',i,'.csv',sep='')) # write results
}


# DEC & BOX-COX & FOURIER & WEATHER EXTERNAL REGRESSORS

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/results/' # directory for the results of experiments

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set
wxregs_train<-lapply(list('tempm_train.csv','hum_train.csv','pressurem_train.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for train set
wxregs_test<-lapply(list('tempm_test.csv','hum_test.csv','pressurem_test.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for test set

# horizontal predictions
test_pred_hfw<-arima_h(train,test,batch=28,freq=365.25*48,freqs=c(48,7*48),f_K=dec_bc_K_h,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE,box_cox = TRUE) # horizontal prediction
save(data=test_pred_hfw,path=paste(exp_dir,'dec,bc,fwreg,arima_h.csv',sep='')) # write results

# vertical predictions
test_pred_vfw<-arima_v(train,test,batch=28,freq=365.25,freqs=c(7),f_K=dec_bc_K_v,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE,box_cox = TRUE) # vertical prediction
save(data=test_pred_vfw,path=paste(exp_dir,'dec,bc,fwreg,arima_v.csv',sep='')) # write results