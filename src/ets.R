library(forecast)
source('dataprep.R')

ets_w=function(train,test,hor=1,batch=7,freq=7){
  pb=txtProgressBar(min = 0, max = nrow(test), style = 3) # initialize progress bar
  test_pred<-data.frame(matrix(data=NA,nrow=nrow(test),ncol=ncol(test),dimnames=list(rownames(test),colnames(test)))) # initialize matrix for predictions
  train<-c(t(train)) # flatten train set
  test<-c(t(test)) # flatten test set
  for (i in seq(0,length(test)-hor,hor)){ # for each sample in test set
    train_ts<-ts(c(train,test[seq_len(i)]),frequency=freq) # add new observations from test set to the current train set
    if (i%%batch==0){ # # if its time to retrain
      model=ets(test_ts) # find best model on the current train set
    }
    else{ # it is not the time to retrain
      model=ets(test_ts,model=model) # do not train, use current model with new observations
    }
    test_pred[(i%/%hor)+1,]=forecast(model,h=hor)$mean # predict new values
    setTxtProgressBar(pb, i) # update progress
  }
  close(pb) # close progress bar
  return(data.frame(test_pred))
}

ets_h=function(train,test,batch=7,freq=24){
  return(ets_w(train,test,hor=24,batch=batch,freq=freq))
}

ets_v=function(train,test,batch=7,freq=7){
  test_pred=as.data.frame(lapply(test, function(x) rep.int(NA, length(x)))) # template dataframe for predictions
  for (col in names(train)){
    train_day=as.data.frame(train[[col]]) # convert dataframe column to dataframe
    test_day=as.data.frame(test[[col]]) # convert dataframe column to dataframe
    colnames(train_day)=c(col) # set column name to match
    colnames(test_day)=c(col) # set column name to match
    test_pred[[col]]=ets_w(train_day,test_day,hor=1,batch=batch,freq=freq) # predictions
  }
  return(test_pred)
}

wip_dir='C:/Users/SABA/Google Drive/mtsg/data/wip/' # directory containing data
exp_dir='C:/Users/SABA/Google Drive/mtsg/data/experiments/' # directory for the results of experiments

train<-load(paste(wip_dir,'train.csv', sep='')) # load train set
test<-load(paste(wip_dir,'test.csv', sep='')) # load test set

# horizontal prediction
test_pred_h=ets_h(train,test,batch=28,freq=24) # predict values
write.csv(data.frame('date'=rownames(test_pred_h),test_pred_h),file=paste(exp_dir,'ets_h.csv'),quote = FALSE) # write predictions

# vertical predictions
test_pred_v=ets_v(train,test,batch=28,freq=7) # predict values
write.csv(data.frame('date'=rownames(test_pred_v),test_pred_v),file=paste(exp_dir,'ets_v.csv'),quote = FALSE) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(wip_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(wip_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hw=ets_h(train,test,batch=4,freq=4) # horizontal predictions for this day
  write.csv(data.frame('date'=rownames(test_pred_hw),test_pred_hw),file=paste(exp_dir,'ets_h_',i,'.csv',sep=''),quote = FALSE) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(wip_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(wip_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vw=ets_v(train,test,batch=4,freq=4) # horizontal predictions for this day
  write.csv(data.frame('date'=rownames(test_pred_vw),test_pred_vw),file=paste(exp_dir,'ets_v_',i,'.csv',sep=''),quote = FALSE) # write results
}






