library(forecast)
library(lubridate)

pop_col=function(data,col){ # removes and returns column from dataframe
  poped_col<-data$col # extract column from dataframe
  data<<-data[ , !names(data) %in% c(col)] # drop column from dataframe
  return(poped_col)
}


arima<-function(train,test,hor=1,batch=7,freq=7,xreg_train=NULL,xreg_test=NULL,freqs=NULL){
  if (is.null(freqs)){ # not considering multiple seasonalities
    dummy_year_train=model.matrix(~as.factor(year(rownames(train))))[,1:11]
    
    dummy_year=model.matrix(~as.factor(year(c(rownames(train),rownames(test)))))
    dummy_year_test=model.matrix(~as.factor(year(rownames(test))))[,1:11]
  }
  else { # considering multiple seasonalities
    params=expand.grid(lapply(K,function(x) seq(0,x%/%2))) # all combinations of fourier orders
    aicc_best=Inf
    param_best
    for (i in 1:nrow(params)){ # for each combination of orders
      param=unlist(params[i,] # order combination
      xreg_train=fourier(msts(train,seasonal.periods=freqs),K=param) # 
      fit=auto.arima(ts(c(t(train)),frequency = freq),xreg=xreg_train)
      if (fit$aicc<aicc_best){
        param_best=param
        aicc_best=fit$aicc
        }
    }
    
  }
  test_pred<-data.frame(matrix(data=NA,nrow=nrow(test),ncol=ncol(test),dimnames=list(rownames(test),colnames(test)))) # initialize matrix for predictions
  pb<-txtProgressBar(min = 0, max = nrow(test), style = 3) # initialize progress bar
  for (i in 0:nrow(test)){ # for each sample in test set
    test_ts<-ts(c(t(rbind(train,head(test,i)))),frequency=freq) # add a new day from test set to the current train set
    if (is.null(xreg)){ # not considering external regressors
      xreg_ts<-xreg # preserve NULL
    }
    else{ # considering external regressors
      xreg_ts<-ts.union(lapply(xreg,function(x,freq) ts(c(t(x)),frequency=freq),freq=freq)) # format and combine external regressors
    }
    if (i%%batch==0){ # # if its time to retrain
      model<-auto.arima(test_ts,xreg=xreg_test) # find best model on the current train set
    }
    else{ # it is not the time to retrain
      model<-Arima(test_ts,model=model,xreg=xreg_test) # do not train, use current model with new observations
    }
    test_pred[i+1,]<-forecast(model,h=hor)$mean # predict new values
    setTxtProgressBar(pb, i) # update progress
  }
  close(pb) # close progress bar
  return(data.frame(test_pred))
}

arima_h<-function(train,test,batch=7,freq=24){
  return(arima(train,test,hor=24,batch=batch,freq=freq))
}

arima_v<-function(train,test,batch=7,freq=7){
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

train<-read.csv(paste(dir,'train.csv', sep=''),header=TRUE,row.names='date',sep=',',dec='.') # load train set
test<-read.csv(paste(dir,'test.csv', sep=''),header=TRUE,row.names='date',sep=',',dec='.') # load test set

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