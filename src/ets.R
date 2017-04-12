library('forecast')
source('dataprep.R')


ets_w<-function(train,test,hor=48,batch=7,freq=48,box_cox=FALSE, dec=FALSE){
  test_pred<-data.frame(matrix(data=NA,nrow=nrow(test),ncol=ncol(test),dimnames=list(rownames(test),colnames(test))),check.names = FALSE) # initialize dataframe for predictions
  train<-c(t(train)) # flatten train set
  test<-c(t(test)) # flatten test set
  for (i in seq(0,length(test)-hor,hor)){ # for each sample in test set
    train_ts<-ts(c(train,test[seq_len(i)]),frequency=freq) # add new observations from test set to the current train set
    if (i%%(batch*hor)==0){ # # if its time to retrain
      bc_lambda<-if (box_cox) BoxCox.lambda(train,method='guerrero') else NULL # estimate lambda for Box-Cox transformation
      if (dec){
        model<-stlm(train_ts,method='ets',allow.multiplicative.trend = TRUE,s.window='periodic',robust=TRUE,lambda=bc_lambda,biasadj=FALSE)
        cat(i%/%(batch*hor),model$model$components,'\n') # print number of retrainings and the type of model
      }
      else{
        model<-ets(train_ts,lambda=bc_lambda,biasadj=FALSE)
        cat(i%/%(batch*hor),model$components,'\n') # print number of retrainings and the type of model
      }
    }
    else{ # it is not the time to retrain
      if (dec){
        model<-stlm(train_ts,model=model,s.window='periodic',robust=TRUE,lambda=bc_lambda,biasadj=FALSE) # do not train, use current model with new observations
      }
      else{
        model<-ets(train_ts,model=model,lambda=bc_lambda,biasadj=FALSE)
      }
        
    }
    test_pred[(i%/%hor)+1,]<-forecast(model,h=hor,lambda=bc_lambda,biasadj=FALSE)$mean # predict new values
  }
  return(test_pred)
}

ets_h<-function(train,test,batch=7,freq=48,box_cox=FALSE,dec=FALSE){
  return(ets_w(train,test,hor=48,batch=batch,freq=freq,box_cox=box_cox,dec=dec))
}

ets_v<-function(train,test,batch=7,freq=7,box_cox=FALSE,dec=FALSE){
  test_pred<-data.frame(matrix(data=NA,nrow=nrow(test),ncol=ncol(test),dimnames=list(rownames(test),colnames(test))),check.names = FALSE) # initialize dataframe for predictions
  for (col in names(train)){
    train_day<-as.data.frame(train[[col]],row.names=rownames(train)) # convert dataframe column to dataframe
    test_day<-as.data.frame(test[[col]],row.names=rownames(test)) # convert dataframe column to dataframe
    colnames(train_day)<-c(col) # set column name to match
    colnames(test_day)<-c(col) # set column name to match
    test_pred[[col]]<-ets_w(train_day,test_day,hor=1,batch=batch,freq=freq,box_cox=box_cox,dec=dec)[[col]] # predictions
  }
  return(test_pred)
}

# NO PREPROCESSING

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/ets/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/ets/exp/' # directory for the results of experiments

train<-load(path=paste(data_dir,'train.csv', sep=''),index='date') # load train set
test<-load(path=paste(data_dir,'test.csv', sep=''),index='date') # load test set

# horizontal prediction
test_pred_h<-ets_h(train,test,batch=7,freq=48) # predict values
save(data=test_pred_h,path=paste(exp_dir,'ets_h.csv',sep='')) # write results

# vertical predictions
test_pred_v<-ets_v(train,test,batch=7,freq=7) # predict values
save(data=test_pred_v,path=paste(exp_dir,'ets_v.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hw<-ets_h(train,test,batch=1,freq=48) # horizontal predictions for this day
  save(data=test_pred_hw,path=paste(exp_dir,'ets_h_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vw<-ets_v(train,test,batch=1,freq=52) # horizontal predictions for this day
  save(data=test_pred_vw,path=paste(exp_dir,'ets_v_',i,'.csv',sep='')) # write results
}


# BOX_COX TRANSFORMATION

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/ets/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/ets/results/' # directory for the results of experiments

train<-load(path=paste(data_dir,'train.csv', sep=''),index='date') # load train set
test<-load(path=paste(data_dir,'test.csv', sep=''),index='date') # load test set

# horizontal prediction
test_pred_h<-ets_h(train,test,batch=7,freq=48,box_cox = TRUE) # predict values
save(data=test_pred_h,path=paste(exp_dir,'bc,ets_h.csv',sep='')) # write results

# vertical predictions
test_pred_v<-ets_v(train,test,batch=7,freq=7,box_cox = TRUE) # predict values
save(data=test_pred_v,path=paste(exp_dir,'bc,ets_v.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hw<-ets_h(train,test,batch=1,freq=48,box_cox = TRUE) # horizontal predictions for this day
  save(data=test_pred_hw,path=paste(exp_dir,'bc,ets_h_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vw<-ets_v(train,test,batch=1,freq=52,box_cox = TRUE) # horizontal predictions for this day
  save(data=test_pred_vw,path=paste(exp_dir,'bc,ets_v_',i,'.csv',sep='')) # write results
}


# DECOMPOSITION

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/ets/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/ets/results/' # directory for the results of experiments

train<-load(path=paste(data_dir,'train.csv', sep=''),index='date') # load train set
test<-load(path=paste(data_dir,'test.csv', sep=''),index='date') # load test set

# horizontal prediction
test_pred_h<-ets_h(train,test,batch=7,freq=48,dec=TRUE) # predict values
save(data=test_pred_h,path=paste(exp_dir,'dec,ets_h.csv',sep='')) # write results

# vertical predictions
test_pred_v<-ets_v(train,test,batch=7,freq=7,,dec=TRUE) # predict values
save(data=test_pred_v,path=paste(exp_dir,'dec,ets_v.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hw<-ets_h(train,test,batch=1,freq=48,dec=TRUE) # horizontal predictions for this day
  save(data=test_pred_hw,path=paste(exp_dir,'dec,ets_h_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vw<-ets_v(train,test,batch=1,freq=52,dec=TRUE) # horizontal predictions for this day
  save(data=test_pred_vw,path=paste(exp_dir,'dec,ets_v_',i,'.csv',sep='')) # write results
}


# DECOPMPOSITION + BOX COX

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/ets/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/ets/results/' # directory for the results of experiments

train<-load(path=paste(data_dir,'train.csv', sep=''),index='date') # load train set
test<-load(path=paste(data_dir,'test.csv', sep=''),index='date') # load test set

# horizontal prediction
test_pred_h<-ets_h(train,test,batch=7,freq=48,box_cox = TRUE,dec=TRUE) # predict values
save(data=test_pred_h,path=paste(exp_dir,'dec,bc,ets_h.csv',sep='')) # write results

# vertical predictions
test_pred_v<-ets_v(train,test,batch=7,freq=7,box_cox = TRUE,dec=TRUE) # predict values
save(data=test_pred_v,path=paste(exp_dir,'dec,bc,ets_v.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hw<-ets_h(train,test,batch=1,freq=48,box_cox = TRUE,dec=TRUE) # horizontal predictions for this day
  save(data=test_pred_hw,path=paste(exp_dir,'dec,bc,ets_h_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vw<-ets_v(train,test,batch=1,freq=52,box_cox = TRUE,dec=TRUE) # horizontal predictions for this day
  save(data=test_pred_vw,path=paste(exp_dir,'dec,bc,ets_v_',i,'.csv',sep='')) # write results
}