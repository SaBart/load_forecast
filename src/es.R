library('forecast')
source('dataprep.R')

# cross-validation for ES models
es<-function(train,test,hor=48,batch=7,freq=48,box_cox=FALSE, dec=FALSE){
  test_pred<-data.frame(matrix(data=NA,nrow=nrow(test),ncol=ncol(test),dimnames=list(rownames(test),colnames(test))),check.names = FALSE) # initialize dataframe for predictions
  train<-c(t(train)) # flatten train set
  test<-c(t(test)) # flatten test set
  for (i in seq(0,length(test)-hor,hor)){ # for each sample in test set
    train_ts<-ts(c(train,test[seq_len(i)]),frequency=freq) # add new observations from test set to the current train set
    if (i%%(batch*hor)==0){ # # if its time to retrain
      bc_lambda<-if (box_cox) BoxCox.lambda(train,method='loglik') else NULL # estimate lambda for Box-Cox transformation
      if (dec){
        model<-stlm(train_ts,method='ets',allow.multiplicative.trend = TRUE,s.window=7,robust=TRUE,lambda=bc_lambda,biasadj=FALSE) # de-seasonalise and estimate ES model
        cat(i%/%(batch*hor),model$model$components,'\n') # print number of retrainings and the type of model
      }
      else{
        model<-ets(train_ts,lambda=bc_lambda,biasadj=FALSE) # estimate ES model
        cat(i%/%(batch*hor),model$components,'\n') # print number of retrainings and the type of model
      }
    }
    else{ # it is not the time to retrain
      if (dec){
        model<-stlm(train_ts,model=model,s.window=7,robust=TRUE,lambda=bc_lambda,biasadj=FALSE) # do not train, use current model with new observations + de-seasonalise
      }
      else{
        model<-ets(train_ts,model=model,lambda=bc_lambda,biasadj=FALSE) # do not train, use current model with new observations
      }
    }
    test_pred[(i%/%hor)+1,]<-forecast(model,h=hor,lambda=bc_lambda,biasadj=FALSE)$mean # forecast new values
  }
  return(test_pred)
}

# no adjustment
es_h<-function(train,test,batch=7,freq=48,box_cox=FALSE,dec=FALSE){
  return(es(train,test,hor=48,batch=batch,freq=freq,box_cox=box_cox,dec=dec))
}

# hour adjustment for ES models
es_v<-function(train,test,batch=7,freq=7,box_cox=FALSE,dec=FALSE){
  test_pred<-data.frame(matrix(data=NA,nrow=nrow(test),ncol=ncol(test),dimnames=list(rownames(test),colnames(test))),check.names = FALSE) # initialize dataframe for predictions
  for (col in names(train)){ # for each half-hour interval
    train_day<-as.data.frame(train[[col]],row.names=rownames(train)) # convert dataframe column to dataframe
    test_day<-as.data.frame(test[[col]],row.names=rownames(test)) # convert dataframe column to dataframe
    colnames(train_day)<-c(col) # set column name to match
    colnames(test_day)<-c(col) # set column name to match
    test_pred[[col]]<-es(train_day,test_day,hor=1,batch=batch,freq=freq,box_cox=box_cox,dec=dec)[[col]] # forecast
  }
  return(test_pred)
}

# NO PREPROCESSING

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/experiments/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/experiments/es_week/' # directory for the results of experiments

# no adjustment
train<-load(path=paste(data_dir,'train.csv', sep=''),idx='date') # load train set
test<-load(path=paste(data_dir,'test.csv', sep=''),idx='date') # load test set
test_pred_h<-es_h(train,test,batch=7,freq=48) # predict values
save(data=test_pred_h,path=paste(exp_dir,'es.csv',sep='')) # write results

# hour adjustment
train<-load(path=paste(data_dir,'train.csv', sep=''),idx='date') # load train set
test<-load(path=paste(data_dir,'test.csv', sep=''),idx='date') # load test set
test_pred_v<-es_v(train,test,batch=7,freq=7) # predict values
save(data=test_pred_v,path=paste(exp_dir,'ha,es.csv',sep='')) # write results

# week adjustment
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hw<-es_h(train,test,batch=1,freq=48) # horizontal predictions for this day
  save(data=test_pred_hw,path=paste(exp_dir,'es_',i,'.csv',sep='')) # write results
}

# week + hour adjustment
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vw<-es_v(train,test,batch=1,freq=52) # horizontal predictions for this day
  save(data=test_pred_vw,path=paste(exp_dir,'ha,es_',i,'.csv',sep='')) # write results
}


# BOX_COX TRANSFORMATION

# no adjustment
train<-load(path=paste(data_dir,'train.csv', sep=''),idx='date') # load train set
test<-load(path=paste(data_dir,'test.csv', sep=''),idx='date') # load test set
test_pred_h<-es_h(train,test,batch=7,freq=48,box_cox = TRUE) # predict values
save(data=test_pred_h,path=paste(exp_dir,'bc,es.csv',sep='')) # write results

# hour adjustment
train<-load(path=paste(data_dir,'train.csv', sep=''),idx='date') # load train set
test<-load(path=paste(data_dir,'test.csv', sep=''),idx='date') # load test set
test_pred_v<-es_v(train,test,batch=7,freq=7,box_cox = TRUE) # predict values
save(data=test_pred_v,path=paste(exp_dir,'ha,bc,es.csv',sep='')) # write results

# week adjustment
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hw<-es_h(train,test,batch=1,freq=48,box_cox = TRUE) # horizontal predictions for this day
  save(data=test_pred_hw,path=paste(exp_dir,'bc,es_',i,'.csv',sep='')) # write results
}

# week + hour adjustment
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vw<-es_v(train,test,batch=1,freq=52,box_cox = TRUE) # horizontal predictions for this day
  save(data=test_pred_vw,path=paste(exp_dir,'ha,bc,es_',i,'.csv',sep='')) # write results
}


# DE-SEASONALISATION

# no adjustment
train<-load(path=paste(data_dir,'train.csv', sep=''),idx='date') # load train set
test<-load(path=paste(data_dir,'test.csv', sep=''),idx='date') # load test set
test_pred_h<-es_h(train,test,batch=7,freq=48,dec=TRUE) # predict values
save(data=test_pred_h,path=paste(exp_dir,'dec,es.csv',sep='')) # write results

# hour adjustment
train<-load(path=paste(data_dir,'train.csv', sep=''),idx='date') # load train set
test<-load(path=paste(data_dir,'test.csv', sep=''),idx='date') # load test set
test_pred_v<-es_v(train,test,batch=7,freq=7,dec=TRUE) # predict values
save(data=test_pred_v,path=paste(exp_dir,'ha,dec,es.csv',sep='')) # write results

# week adjustment
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hw<-es_h(train,test,batch=1,freq=48,dec=TRUE) # horizontal predictions for this day
  save(data=test_pred_hw,path=paste(exp_dir,'dec,es_',i,'.csv',sep='')) # write results
}

# week + hour adjustment
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vw<-es_v(train,test,batch=1,freq=52,dec=TRUE) # horizontal predictions for this day
  save(data=test_pred_vw,path=paste(exp_dir,'ha,dec,es_',i,'.csv',sep='')) # write results
}


# DE-SEASONALISATION + BOX COX

# no adjustment
train<-load(path=paste(data_dir,'train.csv', sep=''),idx='date') # load train set
test<-load(path=paste(data_dir,'test.csv', sep=''),idx='date') # load test set
test_pred_h<-es_h(train,test,batch=7,freq=48,box_cox = TRUE,dec=TRUE) # predict values
save(data=test_pred_h,path=paste(exp_dir,'dec,bc,es.csv',sep='')) # write results

# hour adjustment
train<-load(path=paste(data_dir,'train.csv', sep=''),idx='date') # load train set
test<-load(path=paste(data_dir,'test.csv', sep=''),idx='date') # load test set
test_pred_v<-es_v(train,test,batch=7,freq=7,box_cox = TRUE,dec=TRUE) # predict values
save(data=test_pred_v,path=paste(exp_dir,'ha,dec,bc,es.csv',sep='')) # write results

# week adjustment
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hw<-es_h(train,test,batch=1,freq=48,box_cox = TRUE,dec=TRUE) # horizontal predictions for this day
  save(data=test_pred_hw,path=paste(exp_dir,'dec,bc,es_',i,'.csv',sep='')) # write results
}

# week + hour adjustment
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vw<-es_v(train,test,batch=1,freq=52,box_cox = TRUE,dec=TRUE) # horizontal predictions for this day
  save(data=test_pred_vw,path=paste(exp_dir,'ha,dec,bc,es_',i,'.csv',sep='')) # write results
}




data_dir<-'C:/Users/SABA/tmp/15min/data/' # directory containing data
exp_dir<-'C:/Users/SABA/tmp/15min/results/es/' # directory for the results of experiments

total=length(dir(path = data_dir, full.names = FALSE, no..=TRUE))
i=0

for (d in dir(path = data_dir, full.names = FALSE, no..=TRUE)){
  i=i+1
  cat(d,':',i/total*100,'%\n',sep='')
  train<-load(path=paste(data_dir,d,'/train.csv',sep=''),idx='date') # load train set
  test<-load(path=paste(data_dir,d,'/test.csv',sep=''),idx='date') # load test set
  test_pred_v<-es_v(train,test,batch=28,freq=7) # predict values
  save(data=test_pred_v,path=paste(exp_dir,d,'.csv',sep='')) # write results
}
