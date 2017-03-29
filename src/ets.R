library('forecast')
library('tcltk')
source('dataprep.R')


ets_w<-function(train,test,hor=48,batch=7,freq=48,box_cox=FALSE){
  test_pred<-data.frame(matrix(data=NA,nrow=nrow(test),ncol=ncol(test),dimnames=list(rownames(test),colnames(test)))) # initialize matrix for predictions
  train<-c(t(train)) # flatten train set
  test<-c(t(test)) # flatten test set
  for (i in seq(0,length(test)-hor,hor)){ # for each sample in test set
    train_ts<-ts(c(train,test[seq_len(i)]),frequency=freq) # add new observations from test set to the current train set
    if (i%%(batch*hor)==0){ # # if its time to retrain
      bc_lambda<-if (box_cox) BoxCox.lambda(train,method='') else NULL # estimate lambda for Box-Cox transformation
      model<-ets(test_ts,lambda = bc_lambda) # find best model on the current train set
      cat(i%/%(batch*hor),model$components,'\n') # print number of retrainings and the type of model
    }
    else{ # it is not the time to retrain
      model<-ets(test_ts,model=model,lambda=bc_lambda) # do not train, use current model with new observations
    }
    test_pred[(i%/%hor)+1,]<-forecast(model,h=hor,lambda=bc_lambda)$mean # predict new values
  }
  return(test_pred)
}

ets_h<-function(train,test,batch=7,freq=48,box_cox=FALSE){
  return(ets_w(train,test,hor=48,batch=batch,freq=freq,box_cox=box_cox))
}

ets_v<-function(train,test,batch=7,freq=7,box_cox=FALSE){
  test_pred<-as.data.frame(lapply(test, function(x) rep.int(NA, length(x)))) # template dataframe for predictions
  for (col in names(train)){
    train_day<-as.data.frame(train[[col]]) # convert dataframe column to dataframe
    test_day<-as.data.frame(test[[col]]) # convert dataframe column to dataframe
    colnames(train_day)<-c(col) # set column name to match
    colnames(test_day)<-c(col) # set column name to match
    test_pred[[col]]<-ets_w(train_day,test_day,hor=1,batch=batch,freq=freq,box_cox=box_cox) # predictions
  }
  return(test_pred)
}

wip_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/exp/' # directory for the results of experiments

# NO PREPROCESSING

train<-load(path=paste(wip_dir,'train.csv', sep=''),index='date') # load train set
test<-load(path=paste(wip_dir,'test.csv', sep=''),index='date') # load test set

# horizontal prediction
test_pred_h<-ets_h(train,test,batch=7,freq=48) # predict values
save(data=test_pred_h,path=paste(exp_dir,'ets_h.csv',sep='')) # write results

# vertical predictions
test_pred_v<-ets_v(train,test,batch=7,freq=7) # predict values
save(data=test_pred_v,path=paste(exp_dir,'ets_v.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(wip_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(wip_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hw<-ets_h(train,test,batch=1,freq=48) # horizontal predictions for this day
  save(data=test_pred_hw,path=paste(exp_dir,'ets_h_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(wip_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(wip_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vw<-ets_v(train,test,batch=1,freq=4) # horizontal predictions for this day
  save(data=test_pred_vw,path=paste(exp_dir,'ets_v_',i,'.csv',sep='')) # write results
}


# BOX_COX TRANSFORMATION

train<-load(paste(wip_dir,'train.csv', sep='')) # load train set
test<-load(paste(wip_dir,'test.csv', sep='')) # load test set

# horizontal prediction
test_pred_h<-ets_h(train,test,batch=7,freq=48,box_cox = TRUE) # predict values
save(data=test_pred_h,path=paste(exp_dir,'ets_h.csv',sep='')) # write results

# vertical predictions
test_pred_v<-ets_v(train,test,batch=7,freq=7,box_cox = TRUE) # predict values
save(data=test_pred_v,path=paste(exp_dir,'ets_v.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(wip_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(wip_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hw<-ets_h(train,test,batch=1,freq=48,box_cox = TRUE) # horizontal predictions for this day
  save(data=test_pred_hw,path=paste(exp_dir,'ets_h_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(wip_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(wip_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vw<-ets_v(train,test,batch=1,freq=4,box_cox = TRUE) # horizontal predictions for this day
  save(data=test_pred_vw,path=paste(exp_dir,'ets_v_',i,'.csv',sep='')) # write results
}

