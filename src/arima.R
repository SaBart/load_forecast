library(forecast)
library(lubridate)

load<-function(path){
  data<-read.csv(path,header=TRUE,row.names='date',sep=',',dec='.') # load data
  return(data)
}

pop_col=function(data,col){ # removes and returns column from dataframe
  poped_col<-data$col # extract column from dataframe
  data<<-data[ , !names(data) %in% c(col)] # drop column from dataframe
  return(poped_col)
}

f_ords<-function(train,freq=24,freqs,max_order){
  train<-c(t(train)) # flatten train set
  params<-expand.grid(lapply(freqs,function(x) seq(max_order))) # all combinations of fourier orders
  aicc_best<-Inf # best aicc statistic
  param_best<-NULL # best parameters
  for (i in 1:nrow(params)){ # for each combination of orders
    param<-unlist(params[i,]) # combination of orders
    xreg_train<-fourier(msts(train,seasonal.periods=freqs),K=param) # fourier terms for particular multi-seasonal time series
    fit=auto.arima(ts(train,frequency = freq),xreg=xreg_train,seasonal=FALSE,parallel = TRUE,stepwise=FALSE) # find best arima model
    if (fit$aicc<aicc_best){ # if there is an improvement in aicc statistic
      param_best<-param # save these orders
      aicc_best<-fit$aicc # save new best aicc value
    }
    print(param)
    print(fit$aicc)
  }
  return(param_best)
}

arima<-function(train,test,hor=1,batch=7,freq=24,f_K=NULL,wxreg_train=NULL,wxreg_test=NULL){
  pb<-txtProgressBar(min = 0, max = nrow(test), style = 3) # initialize progress bar
  test_pred<-data.frame(matrix(data=NA,nrow=nrow(test),ncol=ncol(test),dimnames=list(rownames(test),colnames(test)))) # initialize matrix for predictions
  train<-c(t(train)) # flatten train set
  test<-c(t(test)) # flatten test set
  if (is.null(f_K)){ # not considering multiple seasonalities
    fxreg_train<-NULL
    fxreg_test<-NULL
  }
  else { # considering multiple seasonalities
    fxreg_train<-fourier(msts(train,seasonal.periods=freqs),K=f_K)
    fxreg_test<-fourier(msts(test,seasonal.periods=freqs),K=f_K)
  }
  if (is.null(wxreg_train)|is.null(wxreg_test)) # not considering weather regressors
  {
    wxreg_train<-NULL
    wxreg_test<-NULL
  }
  else{ # considering weather regressors
    wxreg_train<-do.call(cbind,lapply(wxreg_train,function(x) c(t(x)))) # flatten and combine weather regressors for train set
    wxreg_test<-do.call(cbind,lapply(wxreg_test,function(x) c(t(x)))) # flatten and combine weather regressors for test set
  }
  xreg_train<-cbind(fxreg_train,wxreg_train) # combine fourier & weather into one matrix for train set
  xreg_test<-cbind(fxreg_test,wxreg_test) # combine fourier & weather into one matrix for test set
  xreg=NULL # default covariances
  for (i in seq(0,length(test)-hor,hor)){ # for each window of observations in test set
    train_ts<-ts(c(train,test[seq_len(i)]),frequency=freq) # add new observations from test set to the current train set
    if (!is.null(xreg_train)&!is.null(xreg_test)){ # if considering external regressors
      xreg<-rbind(xreg_train,xreg_test[seq_len(i),]) # add covariates corresponding to new observations
      xreg_pred<-xreg_test[i+seq_len(hor),] # add covariates for predictions
    }
    if (i%%batch==0){ # if its time to retrain
      model<-auto.arima(train_ts,xreg=xreg,seasonal=FALSE,parallel = TRUE,stepwise=FALSE) # find best model on the current train set
    }
    else{ # it is not the time to retrain
      model<-Arima(train_ts,model=model,xreg=xreg) # do not train, use current model with new observations
    }
    test_pred[i+1,]<-forecast(model,h=hor,xreg=xreg_pred)$mean # predict new values
    setTxtProgressBar(pb, i) # update progress
  }
  close(pb) # close progress bar
  return(data.frame(test_pred))
}

arima_h<-function(train,test,batch=7,freq=24,f_K=NULL,wxreg_train=NULL,wxreg_test=NULL){
  return(arima(train,test,hor=24,batch=batch,freq=freq,f_K=f_K,wxreg_train=wxreg_train,wxreg_test=wxreg_test))
}

arima_v<-function(train,test,batch=7,freq=7,f_K=NULL,wxreg_train=NULL,wxreg_test=NULL){
  test_pred<-as.data.frame(lapply(test, function(x) rep.int(NA, length(x)))) # template dataframe for predictions
  for (col in names(train)){
    train_day<-as.data.frame(train[[col]]) # convert dataframe column to dataframe
    test_day<-as.data.frame(test[[col]]) # convert dataframe column to dataframe
    colnames(train_day)<-c(col) # set column name to match
    colnames(test_day)<-c(col) # set column name to match
    wxreg_train<-lapply(wxreg_train,function(x) as.data.frame(`[[`(x, col))) # extract a particular column from each member of list of covariates
    wreg_test<-lapply(wxreg_test,function(x) as.data.frame(`[[`(x, col))) # extract a particular column from each member of list of covariates  
    test_pred[[col]]<-arima(train_day,test_day,hor=1,batch=batch,freq=freq,f_K=f_K,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # predictions
  }
  return(test_pred)
}


dir<-'C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data

# NO EXTERNAL REGRESSORS

train<-load(paste(dir,'train.csv', sep='')) # load train set
test<-load(paste(dir,'test.csv', sep='')) # load test set

# horizontal predictions
test_pred_h<-arima_h(train,test,batch=28,freq=24) # horizontal prediction
write.csv(test_pred_h,file<-'C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_h.csv',quote = FALSE) # write predictions

# vertical predictions
test_pred_v<-arima_v(train,test,batch=28,freq=7) # vertical predictions
write.csv(test_pred_v,file='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_v.csv') # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hw<-arima_h(train,test,batch=4,freq=24) # horizontal predictions for this day
  write.csv(test_pred_hw,file=paste(dir,'arima_h_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vw<-arima_v(train,test,batch=4,freq=4) # horizontal predictions for this day
  write.csv(test_pred_vw,file=paste(dir,'arima_v_',i,'.csv',sep='')) # write results
}

# FOURIER EXTERNAL REGRESSORS

train<-load(paste(dir,'train.csv', sep='')) # load train set
test<-load(paste(dir,'test.csv', sep='')) # load test set

# horizontal predictions
K<-f_ords(train,freq=24,freqs=c(24*7,365.25*7),max_order=10) # find best fourier coefficients
# K=c(10,6)
test_pred_hf<-arima_h(train,test,batch=28,freq=24,f_K=K) # horizontal prediction
write.csv(test_pred_h,file<-'C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_hf.csv',quote = FALSE) # write predictions

# vertical predictions
K<-f_ords(train,freq=7,freqs=c(365.25),max_order=20) # find best fourier coefficients
# K=c(10)
test_pred_vf<-arima_v(train,test,batch=28,freq=7,f_K=K) # horizontal prediction
write.csv(test_pred_h,file<-'C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_vf.csv',quote = FALSE) # write predictions


# WEATHER EXTERNAL REGRESSORS

# horizontal predictions

train<-load(paste(dir,'train.csv', sep='')) # load train set
test<-load(paste(dir,'test.csv', sep='')) # load test set
wxregs_train<-lapply(list('train_tempm.csv','train_hum.csv','train_wspdm.csv','train_pressurem.csv'),function(x) load(paste(dir,x,sep=''))) # load weather covariates for train set
wxregs_test<-lapply(list('test_tempm.csv','test_hum.csv','test_wspdm.csv','test_pressurem.csv'),function(x) load(paste(dir,x,sep=''))) # load weather covariates for test set

# horizontal predictions
# K<-f_ords(train,freq=24,freqs=c(24*7,365.25*7),max_order=10) # find best fourier coefficients
K=c(10,6)
test_pred_hw<-arima_h(train,test,batch=28,freq=24,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # horizontal prediction
write.csv(test_pred_hfw,file<-'C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_hw.csv',quote = FALSE) # write predictions

# vertical predictions
# K<-f_ords(train,freq=7,freqs=c(365.25),max_order=20) # find best fourier coefficients
K=c(10)
test_pred_vw<-arima_h(train,test,batch=28,freq=7,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # vertical prediction
write.csv(test_pred_vfw,file<-'C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_vw.csv',quote = FALSE) # write predictions

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(dir,'test_',i,'.csv', sep='')) # load test set
  wxregs_train<-lapply(list('train_tempm_','train_hum_','train_wspdm_','train_pressurem_'),function(x) load(paste(dir,x,i,'.csv',sep=''))) # load weather covariates for train set
  wxregs_test<-lapply(list('test_tempm_','test_hum_','test_wspdm_','test_pressurem_'),function(x) load(paste(dir,x,i,'.csv',sep=''))) # load weather covariates for test set
  test_pred_hw<-arima_h(train,test,batch=4,freq=24,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # horizontal predictions for this day
  write.csv(test_pred_hw,file=paste(dir,'arima_hw_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(dir,'test_',i,'.csv', sep='')) # load test set
  wxregs_train<-lapply(list('train_tempm_','train_hum_','train_wspdm_','train_pressurem_'),function(x) load(paste(dir,x,i,'.csv',sep=''))) # load weather covariates for train set
  wxregs_test<-lapply(list('test_tempm_','test_hum_','test_wspdm_','test_pressurem_'),function(x) load(paste(dir,x,i,'.csv',sep=''))) # load weather covariates for test set
  test_pred_vw<-arima_v(train,test,batch=4,freq=4,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # horizontal predictions for this day
  write.csv(test_pred_vw,file=paste(dir,'arima_vw_',i,'.csv',sep='')) # write results
}


# FOURIER & WEATHER EXTERNAL REGRESSORS

train<-load(paste(dir,'train.csv', sep='')) # load train set
test<-load(paste(dir,'test.csv', sep='')) # load test set
wxregs_train<-lapply(list('train_tempm.csv','train_hum.csv','train_wspdm.csv','train_pressurem.csv'),function(x) load(paste(dir,x,sep=''))) # load weather covariates for train set
wxregs_test<-lapply(list('test_tempm.csv','test_hum.csv','test_wspdm.csv','test_pressurem.csv'),function(x) load(paste(dir,x,sep=''))) # load weather covariates for test set

# horizontal predictions
# K<-f_ords(train,freq=24,freqs=c(24*7,365.25*7),max_order=10) # find best fourier coefficients
K=c(10,6)
test_pred_hfw<-arima_h(train,test,batch=28,freq=24,f_K=K,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # horizontal prediction
write.csv(test_pred_hfw,file<-'C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_hfw.csv',quote = FALSE) # write predictions

# vertical predictions
# K<-f_ords(train,freq=7,freqs=c(365.25),max_order=20) # find best fourier coefficients
K=c(10)
test_pred_vfw<-arima_h(train,test,batch=28,freq=7,f_K=K,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # vertical prediction
write.csv(test_pred_vfw,file<-'C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_vfw.csv',quote = FALSE) # write predictions






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