library(forecast)
source('dataprep.R')

bats_w=function(train,test,hor=1,batch=7,freqs=c(24,7*24,365.25*24)){
  pb=txtProgressBar(min = 0, max = nrow(test), style = 3) # initialize progress bar
  test_pred<-data.frame(matrix(data=NA,nrow=nrow(test),ncol=ncol(test),dimnames=list(rownames(test),colnames(test)))) # initialize matrix for predictions
  train<-c(t(train)) # flatten train set
  test<-c(t(test)) # flatten test set
  for (i in seq(0,length(test)-hor,hor)){ # for each sample in test set
    train_ts<-msts(c(train,test[seq_len(i)]),seasonal.periods=freq) # add new observations from test set to the current train set
    if (i%%batch==0){ # # if its time to retrain
      model=bats(test_ts,use.parallel=TRUE) # find best model on the current train set
    }
    else{ # it is not the time to retrain
      model=bats(test_ts,model=model) # do not train, use current model with new observations
    }
    test_pred[(i%/%hor)+1,]=forecast(model,h=hor)$mean # predict new values
    setTxtProgressBar(pb, i) # update progress
  }
  close(pb) # close progress bar
  return(data.frame(test_pred))
}

bats_h=function(train,test,batch=7,freqs=c(24,7*24,365.25*24)){
  return(bats_w(train,test,hor=24,batch=batch,freq=freq))
}

bats_v=function(train,test,batch=7,freqs=c(7*24,365.25*24)){
  test_pred=as.data.frame(lapply(test, function(x) rep.int(NA, length(x)))) # template dataframe for predictions
  for (col in names(train)){
    train_day=as.data.frame(train[[col]]) # convert dataframe column to dataframe
    test_day=as.data.frame(test[[col]]) # convert dataframe column to dataframe
    colnames(train_day)=c(col) # set column name to match
    colnames(test_day)=c(col) # set column name to match
    test_pred[[col]]=bats_w(train_day,test_day,hor=1,batch=batch,freq=freq) # predictions
  }
  return(test_pred)
}

wip_dir='C:/Users/SABA/Google Drive/mtsg/data/wip/' # directory containing data
exp_dir='C:/Users/SABA/Google Drive/mtsg/data/experiments/' # directory for the results of experiments

train<-load(paste(wip_dir,'train.csv', sep='')) # load train set
test<-load(paste(wip_dir,'test.csv', sep='')) # load test set

# horizontal prediction
test_pred_h=bats_h(train,test,batch=28,freqs=c(24,7*24,365.25*24)) # predict values
write.csv(data.frame('date'=rownames(test_pred_h),test_pred_h),file=paste(exp_dir,'bats_h','.csv',sep=''),quote = FALSE) # write predictions

# vertical predictions
test_pred_v=bats_v(train,test,batch=28,freqs=c(7*24,365.25*24)) # predict values
write.csv(data.frame('date'=rownames(test_pred_v),test_pred_v),file=paste(exp_dir,'bats_v','.csv',sep=''),quote = FALSE) # write results







