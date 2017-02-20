library(forecast)


arima=function(train,test,hor=1,batch=7,freq=7){
  test_pred=matrix(data=NA,nrow=nrow(test),ncol=ncol(test)) # initialize matrix for predictions
  pb=txtProgressBar(min = 0, max = nrow(test), style = 3) # initialize progress bar
  for (i in 0:nrow(test)){ # for each sample in test set
    test_ts=ts(c(t(rbind(train,head(test,i)))),frequency=freq) # add a new day from test set to the current train set
    if (i%%batch==0){ # # if its time to retrain
      model=auto.arima(test_ts) # find best model on the current train set
    }
    else{ # it is not the time to retrain
      model=Arima(test_ts,model=model) # do not train, use current model with new observations
    }
    test_pred[i,]=forecast(model,h=hor)$mean # predict new values
    setTxtProgressBar(pb, i) # update progress
  }
  close(pb) # close progress bar
  return(data.frame(test_pred))
}

arima_h=function(train,test,batch=7,freq=24){
  return(arima(train,test,hor=24,batch=batch,freq=freq))
}

arima_v=function(train,test,batch=7,freq=7){
  test_pred=as.data.frame(lapply(test, function(x) rep.int(NA, length(x)))) # template dataframe for predictions
  for (col in names(train)){
    train_day=as.data.frame(train[[col]]) # convert dataframe column to dataframe
    test_day=as.data.frame(test[[col]]) # convert dataframe column to dataframe
    colnames(train_day)=c(col) # set column name to match
    colnames(test_day)=c(col) # set column name to match
    test_pred[[col]]=arima(train_day,test_day,hor=1,batch=batch,freq=freq) # predictions
  }
  return(test_pred)
}

dir='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/' # directory containing data
train=read.csv(paste(dir,'train.csv', sep=''),header=TRUE,sep=',',dec='.') # load train set
test=read.csv(paste(dir,'test.csv', sep=''),header=TRUE,sep=',',dec='.') # load test set
date_train=train$date # extract date column from train set
date_test=test$date # extract date column from test set
train=train[ , !names(train) %in% c('date')] # drop date column from train set
test=test[ , !names(test) %in% c('date')] # drop date column from test set

test_pred_h=arima_h(train,test,batch=28,freq=24) # horizontal prediction
rownames(test_pred_h)=date_test # set "index"
write.csv(test_pred_h,file='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_rh.csv',quote = FALSE) # write predictions

test_pred_v=arima_v(train,test,batch=28,freq=24) # vertical predictions
rownames(test_pred_v)=date_test # set "index"
write.csv(test_pred_v,file='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_rv.csv') # write results

for (i in 0:6){ # for each day
  train=read.csv(paste(dir,'train_',i,'.csv', sep=''),header=TRUE,sep=',',dec='.') # load train set
  test=read.csv(paste(dir,'test_',i,'.csv', sep=''),header=TRUE,sep=',',dec='.') # load test set
  date_train=train$date # extract date column from train set
  date_test=test$date # extract date column from test set
  train=train[ , !names(train) %in% c('date')] # drop date column from train set
  test=test[ , !names(test) %in% c('date')] # drop date column from test set
  test_pred_hw=arima_h(train,test,batch=28,freq=24) # horizontal predictions for this day
  rownames(test_pred_hw)=date_test # set "index"
  write.csv(test_pred_hw,file=paste(dir,'arima_rh_',i,'.csv',sep='')) # write results
}

for (i in 0:6){ # for each day
  train=read.csv(paste(dir,'train_',i,'.csv', sep=''),header=TRUE,sep=',',dec='.') # load train set
  test=read.csv(paste(dir,'test_',i,'.csv', sep=''),header=TRUE,sep=',',dec='.') # load test set
  date_train=train$date # extract date column from train set
  date_test=test$date # extract date column from test set
  train=train[ , !names(train) %in% c('date')] # drop date column from train set
  test=test[ , !names(test) %in% c('date')] # drop date column from test set
  test_pred_vw=arima_v(train,test,batch=28,freq=24) # horizontal predictions for this day
  rownames(test_pred_vw)=date_test # set "index"
  write.csv(test_pred_vw,file=paste(dir,'arima_rv_',i,'.csv',sep='')) # write results
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