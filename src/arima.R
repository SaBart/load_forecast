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

dir='C:/Users/SABA/Google Drive/mtsg/data/' # directory containing data

train=read.csv(paste(dir,'train.csv', sep=''),header=TRUE,sep=',',dec='.') # load train set
test=read.csv(paste(dir,'test.csv', sep=''),header=TRUE,sep=',',dec='.') # load test set
date_train=train$date # extract date column from train set
date_test=test$date # extract date column from test set
train=train[ , !names(train) %in% c('date')] # drop date column from train set
test=test[ , !names(test) %in% c('date')] # drop date column from test set

# horizontal predictions
test_pred_h=arima_h(train,test,batch=28,freq=24) # horizontal prediction
rownames(test_pred_h)=date_test # set "index"
write.csv(test_pred_h,file='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_rh.csv',quote = FALSE) # write predictions

# vertical predictions
test_pred_v=arima_v(train,test,batch=28,freq=7) # vertical predictions
rownames(test_pred_v)=date_test # set "index"
write.csv(test_pred_v,file='C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/arima_rv.csv') # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train=read.csv(paste(dir,'train_',i,'.csv', sep=''),header=TRUE,sep=',',dec='.') # load train set
  test=read.csv(paste(dir,'test_',i,'.csv', sep=''),header=TRUE,sep=',',dec='.') # load test set
  date_train=train$date # extract date column from train set
  date_test=test$date # extract date column from test set
  train=train[ , !names(train) %in% c('date')] # drop date column from train set
  test=test[ , !names(test) %in% c('date')] # drop date column from test set
  test_pred_hw=arima_h(train,test,batch=4,freq=24) # horizontal predictions for this day
  rownames(test_pred_hw)=date_test # set "index"
  write.csv(test_pred_hw,file=paste(dir,'arima_rh_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train=read.csv(paste(dir,'train_',i,'.csv', sep=''),header=TRUE,sep=',',dec='.') # load train set
  test=read.csv(paste(dir,'test_',i,'.csv', sep=''),header=TRUE,sep=',',dec='.') # load test set
  date_train=train$date # extract date column from train set
  date_test=test$date # extract date column from test set
  train=train[ , !names(train) %in% c('date')] # drop date column from train set
  test=test[ , !names(test) %in% c('date')] # drop date column from test set
  test_pred_vw=arima_v(train,test,batch=4,freq=4) # horizontal predictions for this day
  rownames(test_pred_vw)=date_test # set "index"
  write.csv(test_pred_vw,file=paste(dir,'arima_rv_',i,'.csv',sep='')) # write results
}






