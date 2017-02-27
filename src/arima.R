library(forecast)
library(lubridate)

pop_col=function(data,col){ # removes and returns column from dataframe
  poped_col<-data$col # extract column from dataframe
  data<<-data[ , !names(data) %in% c(col)] # drop column from dataframe
  return(poped_col)
}

f_ords<-function(train,freq=24,freqs,max_order){
  params<-expand.grid(lapply(freqs,function(x) seq(max_order))) # all combinations of fourier orders
  aicc_best<-Inf # best aicc statistic
  param_best<-NULL # best parameters
  for (i in 1:nrow(params)){ # for each combination of orders
    param<-unlist(params[i,]) # combination of orders
    xreg_train<-fourier(msts(train,seasonal.periods=freqs),K=param) # fourier terms for particular multi-seasonal time series
    fit=auto.arima(ts(c(t(train)),frequency = freq),xreg=xreg_train,seasonal=FALSE,parallel = TRUE,stepwise=FALSE) # find best arima model
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
    wxreg_train<-cbind(lapply(wxreg_train,function(x) c(t(x)))) # format and combine weather regressors for train set
    wxreg_test<-cbind(lapply(wxreg_test,function(x) c(t(x)))) # format and combine weather regressors for test set
  }
  xreg_train<-cbind(fxreg_train,wxreg_train) # combine fourier & weather into one matrix for train set
  xreg_test<-cbind(fxreg_test,wxreg_test) # combine fourier & weather into one matrix for test set
  test_pred<-data.frame(matrix(data=NA,nrow=nrow(test),ncol=ncol(test),dimnames=list(rownames(test),colnames(test)))) # initialize matrix for predictions
  for (i in 0:nrow(test)-1){ # for each sample in test set
    train_ts<-ts(c(t(rbind(train,head(test,i)))),frequency=freq) # add a new day from test set to the current train set
    if (is.null(xreg_train)|is.null(xreg_test)){ # not considering external regressors
      xreg_train<-NULL # preserve NULL
      xreg_test<-NULL # preserve NULL
    }
    else{ # considering external regressors
      xreg_train<-rbind(xreg_train,xreg_test[,])
    }
    if (i%%batch==0){ # if its time to retrain
      model<-auto.arima(train_ts,xreg=cbind(fxreg_train,wxreg_train)) # find best model on the current train set
    }
    else{ # it is not the time to retrain
      model<-Arima(train_ts,model=model,xreg=cbind(fxreg_train,wxreg_train)) # do not train, use current model with new observations
    }
    test_pred[i+1,]<-forecast(model,h=hor,xreg=)$mean # predict new values
    setTxtProgressBar(pb, i) # update progress
  }
  close(pb) # close progress bar
  return(data.frame(test_pred))
}

arima_h<-function(train,test,batch=7,freq=24,freqs=NULL,fourier=NULL,xreg_train=NULL,xreg_test=NULL){
  return(arima(train,test,hor=24,batch=batch,freq=freq))
}

arima_v<-function(train,test,batch=7,freq=7,freqs=NULL,fourier=NULL,xreg_train=NULL,xreg_test=NULL){
  test_pred<-as.data.frame(lapply(test, function(x) rep.int(NA, length(x)))) # template dataframe for predictions
  for (col in names(train)){
    train_day<-as.data.frame(train[[col]]) # convert dataframe column to dataframe
    test_day<-as.data.frame(test[[col]]) # convert dataframe column to dataframe
    colnames(train_day)<-c(col) # set column name to match
    colnames(test_day)<-c(col) # set column name to match
    test_pred[[col]]<-arima(train_day,test_day,hor=1,batch=batch,freq=freq) # predictions
  }
  return(test_pred)
}

dir<-'C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data

# NO EXTERNAL REGRESSORS

# horizontal predictions
test_pred_h<-arima_h(train,test,batch=28,freq=24) # horizontal prediction
rownames(test_pred_h)<-date_test # set "index"
write.csv(test_pred_h,file<-'C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_rh.csv',quote = FALSE) # write predictions

# vertical predictions
test_pred_v<-arima_v(train,test,batch=28,freq=7) # vertical predictions
rownames(test_pred_v)<-date_test # set "index"
write.csv(test_pred_v,file='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_rv.csv') # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-read.csv(paste(dir,'train_',i,'.csv', sep=''),header=TRUE,sep=',',dec='.') # load train set
  test<-read.csv(paste(dir,'test_',i,'.csv', sep=''),header=TRUE,sep=',',dec='.') # load test set
  test_pred_hw<-arima_h(train,test,batch=4,freq=24) # horizontal predictions for this day
  write.csv(test_pred_hw,file=paste(dir,'arima_rh_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-read.csv(paste(dir,'train_',i,'.csv', sep=''),header=TRUE,sep=',',dec='.') # load train set
  test<-read.csv(paste(dir,'test_',i,'.csv', sep=''),header=TRUE,sep=',',dec='.') # load test set
  test_pred_vw<-arima_v(train,test,batch=4,freq=4) # horizontal predictions for this day
  write.csv(test_pred_vw,file=paste(dir,'arima_rv_',i,'.csv',sep='')) # write results
}

# FOURIER EXTERNAL REGRESSORS

train<-read.csv(paste(dir,'train.csv', sep=''),header=TRUE,row.names='date',sep=',',dec='.') # load train set
test<-read.csv(paste(dir,'test.csv', sep=''),header=TRUE,row.names='date',sep=',',dec='.') # load test set

# horizontal predictions
K<-f_ords(train,freq=24,freqs=c(24*7,365.25*7),max_order=10) # find best fourier coefficients
# K=c(10,6)
test_pred_h<-arima_h(train,test,batch=28,freq=24,freqs=c(24*7,365.25*7),fourier=K) # horizontal prediction
rownames(test_pred_h)<-date_test # set "index"
write.csv(test_pred_h,file<-'C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_rh.csv',quote = FALSE) # write predictions


# WEATHER EXTERNAL REGRESSORS

# horizontal predictions
train<-read.csv(paste(dir,'train.csv', sep=''),header=TRUE,row.names='date',sep=',',dec='.') # load train set
test<-read.csv(paste(dir,'test.csv', sep=''),header=TRUE,row.names='date',sep=',',dec='.') # load test set
temp_train<-read.csv(paste(dir,'train_tempm.csv', sep=''),header=TRUE,row.names='date',sep=',',dec='.') # load temperatures for train set
temp_test<-read.csv(paste(dir,'test_tempm.csv', sep=''),header=TRUE,row.names='date',sep=',',dec='.') # load temperatures for test set
hum_train<-read.csv(paste(dir,'train_hum.csv', sep=''),header=TRUE,row.names='date',sep=',',dec='.') # load humidities for train set
hum_test<-read.csv(paste(dir,'test_hum.csv', sep=''),header=TRUE,row.names='date',sep=',',dec='.') # load humidities for test set
wspd_train<-read.csv(paste(dir,'train_wspdm.csv', sep=''),header=TRUE,row.names='date',sep=',',dec='.') # load wind speeds for train set
wspd_test<-read.csv(paste(dir,'test_wspdm.csv', sep=''),header=TRUE,row.names='date',sep=',',dec='.') # load wind speeds for test set
prsr_train<-read.csv(paste(dir,'train_pressurem.csv', sep=''),header=TRUE,row.names='date',sep=',',dec='.') # load pressures for train set
prsr_test<-read.csv(paste(dir,'test_pressurem.csv', sep=''),header=TRUE,row.names='date',sep=',',dec='.') # load pressures for test set


xregs=list(temp_train,hum_train,wspd_train,prsr_train)


# FOURIER & WEATHER EXTERNAL REGRESSORS




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