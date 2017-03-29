library(forecast)
library(lubridate)
source('dataprep.R')

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
  total=nrow(test) # number of days to predict
  pb <- tkProgressBar(title = "ETS", min = 0, max = total, width = 500) # initialize progress bar
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
  xreg=NULL # default covariates
  xreg_pred=NULL # default covariates for predictions
  for (i in seq(0,length(test)-hor,hor)){ # for each window of observations in test set
    train_ts<-ts(c(train,test[seq_len(i)]),frequency=freq) # add new observations from test set to the current train set
    if (!is.null(xreg_train)&!is.null(xreg_test)){ # if considering external regressors
      xreg<-rbind(xreg_train,xreg_test[seq_len(i),]) # add covariates corresponding to new observations
      xreg_pred<-xreg_test[i+seq_len(hor),] # add covariates for predictions
    }
    if (i%%(batch*hor)==0){ # if its time to retrain
      bc_lambda<-if (box_cox) BoxCox.lambda(train,method='') else NULL # estimate lambda for Box-Cox transformation
      model<-auto.arima(train_ts,xreg=xreg,seasonal=FALSE,parallel = TRUE,stepwise=FALSE,lambda=bc_lambda) # find best model on the current train set
      print(arimaorder(model)) # print the type of model
    }
    else{ # it is not the time to retrain
      model<-Arima(train_ts,model=model,xreg=xreg,lambda=bc_lambda) # do not train, use current model with new observations
    }
    d=(i%/%hor)+1
    test_pred[d,]<-forecast(model,h=hor,xreg=xreg_pred,lambda=bc_lambda)$mean # predict new values
      setTkProgressBar(pb, d,label=paste( d,'/',total)) # update progress
  }
  close(pb) # close progress bar
  return(test_pred)
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


wip_dir='C:/Users/SABA/Google Drive/mtsg/data/wip/' # directory containing data
exp_dir='C:/Users/SABA/Google Drive/mtsg/data/experiments/' # directory for the results of experiments

# NO EXTERNAL REGRESSORS

train<-load(paste(wip_dir,'train.csv', sep='')) # load train set
test<-load(paste(wip_dir,'test.csv', sep='')) # load test set

# horizontal predictions
test_pred_h<-arima_h(train,test,batch=28,freq=24) # predict values
save(data=test_pred_h,path=paste(exp_dir,'arima_h','.csv',sep='')) # write results

# vertical predictions
test_pred_v<-arima_v(train,test,batch=28,freq=7) # predict values
save(data=test_pred_v,path=paste(exp_dir,'arima_v','.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(wip_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(wip_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hwi<-arima_h(train,test,batch=4,freq=24) # horizontal predictions for this day
  save(data=test_pred_hwi,path=paste(exp_dir,'arima_h_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(wip_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(wip_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vwi<-arima_v(train,test,batch=4,freq=4) # vertical predictions for this day
  save(data=test_pred_vwi,path=paste(exp_dir,'arima_v_',i,'.csv',sep='')) # write results
}

# FOURIER EXTERNAL REGRESSORS

train<-load(paste(wip_dir,'train.csv', sep='')) # load train set
test<-load(paste(wip_dir,'test.csv', sep='')) # load test set

# horizontal predictions
K<-f_ords(train,freq=365.25*24,freqs=c(24,7*24),max_order=12) # find best fourier coefficients
# K=c(10,6)
test_pred_hf<-arima_h(train,test,batch=28,freq=24,f_K=K) # horizontal prediction
save(data=test_pred_hf,path=paste(exp_dir,'arima_hf.csv',sep='')) # write results

# vertical predictions
K<-f_ords(train,freq=365.25,freqs=c(7),max_order=3) # find best fourier coefficients
# K=c(2)
test_pred_vf<-arima_v(train,test,batch=28,freq=7,f_K=K) # horizontal prediction
save(data=test_pred_vf,path=paste(exp_dir,'arima_vf.csv',sep='')) # write results


# WEATHER EXTERNAL REGRESSORS

train<-load(paste(dir,'train.csv', sep='')) # load train set
test<-load(paste(dir,'test.csv', sep='')) # load test set
wxregs_train<-lapply(list('train_tempm.csv','train_hum.csv','train_wspdm.csv','train_pressurem.csv'),function(x) load(paste(wip_dir,x,sep=''))) # load weather covariates for train set
wxregs_test<-lapply(list('test_tempm.csv','test_hum.csv','test_wspdm.csv','test_pressurem.csv'),function(x) load(paste(wip_dir,x,sep=''))) # load weather covariates for test set

# horizontal predictions
test_pred_hw<-arima_h(train,test,batch=28,freq=24,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # horizontal prediction
save(data=test_pred_hw,path=paste(exp_dir,'arima_hw.csv',sep='')) # write results

# vertical predictions
test_pred_vw<-arima_h(train,test,batch=28,freq=7,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # vertical prediction
save(data=test_pred_vw,path=paste(exp_dir,'arima_vw.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(wip_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(wip_dir,'test_',i,'.csv', sep='')) # load test set
  wxregs_train<-lapply(list('train_tempm_','train_hum_','train_wspdm_','train_pressurem_'),function(x) load(paste(wip_dir,x,i,'.csv',sep=''))) # load weather covariates for train set
  wxregs_test<-lapply(list('test_tempm_','test_hum_','test_wspdm_','test_pressurem_'),function(x) load(paste(wip_dir,x,i,'.csv',sep=''))) # load weather covariates for test set
  test_pred_hwi<-arima_h(train,test,batch=4,freq=24,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # horizontal predictions for this day
  save(data=test_pred_hwi,path=paste(exp_dir,'arima_hw_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(wip_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(wip_dir,'test_',i,'.csv', sep='')) # load test set
  wxregs_train<-lapply(list('train_tempm_','train_hum_','train_wspdm_','train_pressurem_'),function(x) load(paste(wip_dir,x,i,'.csv',sep=''))) # load weather covariates for train set
  wxregs_test<-lapply(list('test_tempm_','test_hum_','test_wspdm_','test_pressurem_'),function(x) load(paste(wip_dir,x,i,'.csv',sep=''))) # load weather covariates for test set
  test_pred_vwi<-arima_v(train,test,batch=4,freq=4,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # horizontal predictions for this day
  save(data=test_pred_hwi,path=paste(exp_dir,'arima_vw_',i,'.csv',sep='')) # write results
}


# FOURIER & WEATHER EXTERNAL REGRESSORS

train<-load(paste(wip_dir,'train.csv', sep='')) # load train set
test<-load(paste(wip_dir,'test.csv', sep='')) # load test set
wxregs_train<-lapply(list('train_tempm.csv','train_hum.csv','train_wspdm.csv','train_pressurem.csv'),function(x) load(paste(wip_dir,x,sep=''))) # load weather covariates for train set
wxregs_test<-lapply(list('test_tempm.csv','test_hum.csv','test_wspdm.csv','test_pressurem.csv'),function(x) load(paste(wip_dir,x,sep=''))) # load weather covariates for test set

# horizontal predictions
# K<-f_ords(train,freq=365.25*24,freqs=c(24,7*24),max_order=10) # find best fourier coefficients
K=c(10,6)
test_pred_hfw<-arima_h(train,test,batch=28,freq=24,f_K=K,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # horizontal prediction
save(data=test_pred_hfw,path=paste(exp_dir,'arima_hfw.csv',sep='')) # write results

# vertical predictions
K<-f_ords(train,freq=365.25,freqs=c(7),max_order=3) # find best fourier coefficients
K=c(10)
test_pred_vfw<-arima_h(train,test,batch=28,freq=7,f_K=K,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # vertical prediction
save(data=test_pred_vfw,path=paste(exp_dir,'arima_vfw.csv',sep='')) # write results










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