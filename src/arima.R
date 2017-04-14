library(forecast)
source('dataprep.R')

pop_col=function(data,col){ # removes and returns column from dataframe
  poped_col<-data$col # extract column from dataframe
  data<<-data[ , !names(data) %in% c(col)] # drop column from dataframe
  return(poped_col)
}

f_ords<-function(train,freq=48,freqs,ords,dec=FALSE,box_cox=FALSE){
  train<-c(t(train)) # flatten train set
  aicc_best<-Inf # best aicc statistic
  param_best<-NULL # best parameters
  bc_lambda<-if (box_cox) BoxCox.lambda(train,method='guerrero') else NULL # estimate lambda for Box-Cox transformation
  for (i in 1:nrow(ords)){ # for each combination of orders
    ord<-unlist(ords[i,]) # combination of orders
    xreg_train<-fourier(msts(train,seasonal.periods=freqs),K=ord) # fourier terms for particular multi-seasonal time series
    if (dec) # if decompose first
    {
      fit=stlm(ts(train,frequency = freq),method='arima',xreg=xreg_train,s.window='periodic',robust=TRUE,trace=TRUE,lambda = bc_lambda)$model  # find best arima model after decomposition
    }
    else{ # dont decompose
      fit=auto.arima(ts(train,frequency = freq),xreg=xreg_train,seasonal=FALSE,trace=TRUE,lambda = bc_lambda) # find best arima model  
    }
    if (fit$aicc<aicc_best){ # if there is an improvement in aicc statistic
      ord_best<-ord # save these orders
      aicc_best<-fit$aicc # save new best aicc value
    }
  }
  return(ord_best)
}

arima<-function(train,test,hor=1,batch=7,freq=48,freqs=NULL,f_K=NULL,wxregs_train=NULL,wxregs_test=NULL,box_cox=FALSE,dec=FALSE){
  test_pred<-data.frame(matrix(data=NA,nrow=nrow(test),ncol=ncol(test),dimnames=list(rownames(test),colnames(test))),check.names = FALSE) # initialize matrix for predictions
  train<-c(t(train)) # flatten train set
  test<-c(t(test)) # flatten test set
  if (is.null(f_K)){ # not considering multiple seasonalities
    fxregs_train<-NULL
    fxregs_test<-NULL
  }
  else { # considering multiple seasonalities
    fxregs_train<-fourier(msts(train,seasonal.periods=freqs),K=f_K)
    fxregs_test<-fourier(msts(test,seasonal.periods=freqs),K=f_K)
  }
  if (is.null(wxregs_train)|is.null(wxregs_test)) # not considering weather regressors
  {
    wxregs_train<-NULL
    wxregs_test<-NULL
  }
  else{ # considering weather regressors
    wxregs_train<-do.call(cbind,lapply(wxregs_train,function(x) c(t(x)))) # flatten and combine weather regressors for train set
    wxregs_test<-do.call(cbind,lapply(wxregs_test,function(x) c(t(x)))) # flatten and combine weather regressors for test set
  }
  xregs_train<-cbind(fxregs_train,wxregs_train) # combine fourier & weather into one matrix for train set
  xregs_test<-cbind(fxregs_test,wxregs_test) # combine fourier & weather into one matrix for test set
  xregs=NULL # default covariates
  xregs_pred=NULL # default covariates for predictions
  for (i in seq(0,length(test)-hor,hor)){ # for each window of observations in test set
    train_ts<-ts(c(train,test[seq_len(i)]),frequency=freq) # add new observations from test set to the current train set
    if (!is.null(xregs_train)&!is.null(xregs_test)){ # if considering external regressors
      xregs<-rbind(xregs_train,xregs_test[seq_len(i),]) # add covariates corresponding to new observations
      xregs_pred<-xregs_test[i+seq_len(hor),] # add covariates for predictions
    }
    if (i%%(batch*hor)==0){ # if its time to retrain
      model<-NULL
      bc_lambda<-if (box_cox) BoxCox.lambda(train,method='guerrero') else NULL # estimate lambda for Box-Cox transformation
      if (dec){ # if decomposition is to be applied
       model<-stlm(train_ts,method='arima',xreg=xregs,s.window='periodic',robust=TRUE,lambda=bc_lambda,biasadj = FALSE,trace=TRUE) # find best model on the current train set
      }
      else { # no decomposition
        model<-auto.arima(train_ts,xreg=xregs,lambda=bc_lambda,biasadj = FALSE,trace = TRUE) # find best model on the current train set
      }
      cat(i%/%(batch*hor),'\n') # print number of retrainings and the type of model
    }
    else{ # it is not the time to retrain
      if (dec){
        model<-stlm(train_ts,model=model$model,modelfunction=function(x, ...) {Arima(x, xreg=xregs, ...)},s.window='periodic',robust=TRUE,lambda=bc_lambda,biasadj = FALSE) # do not train, use current model with new observations
      }
      else
      {
        model<-Arima(train_ts,model=model,xreg=xregs,lambda=bc_lambda,biasadj=FALSE) # do not train, use current model with new observations  
      }
    }
    test_pred[(i%/%hor)+1,]<-forecast(model,h=hor,xreg=xregs_pred,lambda=bc_lambda,biasadj=FALSE)$mean # predict new values
  }
  return(test_pred)
}

arima_h<-function(train,test,batch=7,freq=48,f_K=NULL,wxregs_train=NULL,wxregs_test=NULL,box_cox=FALSE,dec=FALSE){
  return(arima(train,test,hor=48,batch=batch,freq=freq,f_K=f_K,wxregs_train=wxregs_train,wxregs_test=wxregs_test,box_cox = box_cox,dec = dec))
}

arima_v<-function(train,test,f_K=NULL,wxregs_train=NULL,wxregs_test=NULL,...){
  test_pred<-data.frame(matrix(data=NA,nrow=nrow(test),ncol=ncol(test),dimnames=list(rownames(test),colnames(test))),check.names = FALSE) # initialize dataframe for predictions
  for (col in names(train)){
    train_col<-as.data.frame(train[[col]],row.names=rownames(train)) # convert dataframe column to dataframe
    test_col<-as.data.frame(test[[col]],row.names=rownames(test)) # convert dataframe column to dataframe
    colnames(train_col)<-c(col) # set column name to match
    colnames(test_col)<-c(col) # set column name to match
    f_K_col= if (!is.null(f_K)) f_K[[col]] else f_K
    if (is.null(wxregs_train)|is.null(wxregs_test)) # no weather regressors
    {
      wxregs_train_col<-NULL
      wxregs_test_col<-NULL
    }
    else # consider weather regressors
    {
      wxregs_train_col<-lapply(wxregs_train,function(x) as.data.frame(`[[`(x, col))) # extract a particular column from each member of list of covariates
      wxregs_test_col<-lapply(wxregs_test,function(x) as.data.frame(`[[`(x, col))) # extract a particular column from each member of list of covariates  
    }
    test_pred[[col]]<-arima(train_col,test_col,hor=1,f_K=f_K_col,wxregs_train=wxregs_train_col,wxregs_test=wxregs_test_col,...)[[col]] # predictions
  }
  return(test_pred)
}

# ESTIMATE FOURIER ORDERS

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/data/' # directory containing data

train<-load(paste(data_dir,'train_full.csv', sep='')) # load train set

# horizontal predictions

ords=params<-expand.grid(seq(from=5,to=20,by=5),seq(from=50,to=150,by=50)) # all combinations of fourier orders to try
K_h<-f_ords(train,freq=365.25*48,freqs=c(48,7*48),ords=ords) # find best fourier coefficients
K_h=c(10,6)

# vertical predictions

ords<-params<-expand.grid(seq(3)) # all combinations of fourier orders to try
K_v <- sapply(names(train),function(x) NULL) # initialize empty list for orders
for (col in names(train)){
  train_col<-as.data.frame(train[[col]],row.names=rownames(train)) # convert dataframe column to dataframe
  colnames(train_col)<-c(col) # set column name to match
  K_v[[col]]<-f_ords(train_col,freq=365.25,freqs=c(7),ords=ords) # find best fourier coefficients  
}

# vertical predictions & BC

ords<-params<-expand.grid(seq(3)) # all combinations of fourier orders to try
bc_K_v <- sapply(names(train),function(x) NULL) # initialize empty list for orders
for (col in names(train)){
  train_col<-as.data.frame(train[[col]],row.names=rownames(train)) # convert dataframe column to dataframe
  colnames(train_col)<-c(col) # set column name to match
  bc_K_v[[col]]<-f_ords(train_col,freq=365.25,freqs=c(7),ords=ords,box_cox=TRUE) # find best fourier coefficients  
}

# vertical predictions & DEC & BC

ords<-params<-expand.grid(seq(3)) # all combinations of fourier orders to try
dec_bc_K_v <- sapply(names(train),function(x) NULL) # initialize empty list for orders
for (col in names(train)){
  train_col<-as.data.frame(train[[col]],row.names=rownames(train)) # convert dataframe column to dataframe
  colnames(train_col)<-c(col) # set column name to match
  dec_bc_K_v[[col]]<-f_ords(train_col,freq=365.25,freqs=c(7),ords=ords,dec=TRUE,box_cox=TRUE) # find best fourier coefficients  
}



# NO EXTERNAL REGRESSORS

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/results/' # directory for the results of experiments

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set

# horizontal predictions
test_pred_h<-arima_h(train,test,batch=28,freq=48) # predict values
save(data=test_pred_h,path=paste(exp_dir,'arima_h','.csv',sep='')) # write results

# vertical predictions
test_pred_v<-arima_v(train,test,batch=28,freq=7) # predict values
save(data=test_pred_v,path=paste(exp_dir,'arima_v','.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hwi<-arima_h(train,test,batch=4,freq=48) # horizontal predictions for this day
  save(data=test_pred_hwi,path=paste(exp_dir,'arima_h_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vwi<-arima_v(train,test,batch=4,freq=52) # vertical predictions for this day
  save(data=test_pred_vwi,path=paste(exp_dir,'arima_v_',i,'.csv',sep='')) # write results
}

# FOURIER EXTERNAL REGRESSORS

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set

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
wxregs_train<-lapply(list('train_tempm.csv','train_hum.csv','train_wspdm.csv','train_pressurem.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for train set
wxregs_test<-lapply(list('test_tempm.csv','test_hum.csv','test_wspdm.csv','test_pressurem.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for test set

# horizontal predictions
test_pred_hw<-arima_h(train,test,batch=28,freq=24,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # horizontal prediction
save(data=test_pred_hw,path=paste(exp_dir,'arima_hw.csv',sep='')) # write results

# vertical predictions
test_pred_vw<-arima_h(train,test,batch=28,freq=7,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # vertical prediction
save(data=test_pred_vw,path=paste(exp_dir,'arima_vw.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  wxregs_train<-lapply(list('train_tempm_','train_hum_','train_wspdm_','train_pressurem_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for train set
  wxregs_test<-lapply(list('test_tempm_','test_hum_','test_wspdm_','test_pressurem_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for test set
  test_pred_hwi<-arima_h(train,test,batch=4,freq=24,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # horizontal predictions for this day
  save(data=test_pred_hwi,path=paste(exp_dir,'arima_hw_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  wxregs_train<-lapply(list('train_tempm_','train_hum_','train_wspdm_','train_pressurem_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for train set
  wxregs_test<-lapply(list('test_tempm_','test_hum_','test_wspdm_','test_pressurem_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for test set
  test_pred_vwi<-arima_v(train,test,batch=4,freq=4,wxreg_train=wxreg_train,wxreg_test=wxreg_test) # horizontal predictions for this day
  save(data=test_pred_hwi,path=paste(exp_dir,'arima_vw_',i,'.csv',sep='')) # write results
}


# FOURIER & WEATHER EXTERNAL REGRESSORS

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set
wxregs_train<-lapply(list('train_tempm.csv','train_hum.csv','train_wspdm.csv','train_pressurem.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for train set
wxregs_test<-lapply(list('test_tempm.csv','test_hum.csv','test_wspdm.csv','test_pressurem.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for test set

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



# DEC & BOX-COX

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/results/' # directory for the results of experiments

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set

# horizontal predictions
test_pred_h<-arima_h(train,test,batch=28,freq=48,dec=TRUE,box_cox = TRUE) # predict values
save(data=test_pred_h,path=paste(exp_dir,'dec,bc,arima_h','.csv',sep='')) # write results

# vertical predictions
test_pred_v<-arima_v(train,test,batch=28,freq=7,dec=TRUE,box_cox = TRUE) # predict values
save(data=test_pred_v,path=paste(exp_dir,'dec,bc,arima_v','.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_hwi<-arima_h(train,test,batch=4,freq=48,dec=TRUE,box_cox = TRUE) # horizontal predictions for this day
  save(data=test_pred_hwi,path=paste(exp_dir,'dec,bc,arima_h_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  test_pred_vwi<-arima_v(train,test,batch=4,freq=52,dec=TRUE,box_cox = TRUE) # vertical predictions for this day
  save(data=test_pred_vwi,path=paste(exp_dir,'dec,bc,arima_v_',i,'.csv',sep='')) # write results
}

# DEC & BOX-COX & FOURIER EXTERNAL REGRESSORS

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/results/' # directory for the results of experiments

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set

# horizontal predictions
K<-f_ords(train,freq=365.25*48,freqs=c(48,7*48),max_order=24) # find best fourier coefficients
# K=c(10,6)
test_pred_hf<-arima_h(train,test,batch=28,freq=48,f_K=dec_bc_K_h) # horizontal prediction
save(data=test_pred_hf,path=paste(exp_dir,'dec,bc,freg,arima_h.csv',sep='')) # write results

# vertical predictions
test_pred_vf<-arima_v(train,test,batch=28,freq=7,f_K=dec_bc_K_v) # horizontal prediction
save(data=test_pred_vf,path=paste(exp_dir,'dec,bc,freg,arima_v.csv',sep='')) # write results

# DEC & BOX-COX & WEATHER EXTERNAL REGRESSORS

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/results/' # directory for the results of experiments

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set
wxregs_train<-lapply(list('tempm_train.csv','hum_train.csv','pressurem_train.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for train set
wxregs_test<-lapply(list('tempm_test.csv','hum_test.csv','pressurem_test.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for test set

# horizontal predictions
test_pred_hw<-arima_h(train,test,batch=28,freq=48,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE,box_cox = TRUE) # horizontal prediction
save(data=test_pred_hw,path=paste(exp_dir,'dec,bc,wreg,arima_h.csv',sep='')) # write results

# vertical predictions
test_pred_vw<-arima_h(train,test,batch=28,freq=7,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE,box_cox = TRUE) # vertical prediction
save(data=test_pred_vw,path=paste(exp_dir,'dec,bc,wreg,arima_v.csv',sep='')) # write results

# horizontal predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  wxregs_train<-lapply(list('tempm_train_','hum_train_','pressurem_train_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for train set
  wxregs_test<-lapply(list('tempm_test_','hum_test_','pressurem_test_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for test set
  test_pred_hwi<-arima_h(train,test,batch=4,freq=48,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE,box_cox = TRUE) # horizontal predictions for this day
  save(data=test_pred_hwi,path=paste(exp_dir,'dec,bc,wreg,arima_h_',i,'.csv',sep='')) # write results
}

# vertical predictions for each day separately
for (i in 0:6){ # for each day
  train<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train set
  test<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test set
  wxregs_train<-lapply(list('tempm_train_','hum_train_','pressurem_train_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for train set
  wxregs_test<-lapply(list('tempm_test_','hum_test_','pressurem_test_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather covariates for test set
  test_pred_vwi<-arima_v(train,test,batch=4,freq=52,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE,box_cox = TRUE) # horizontal predictions for this day
  save(data=test_pred_vwi,path=paste(exp_dir,'dec,bc,wreg,arima_v_',i,'.csv',sep='')) # write results
}

# DEC & BOX-COX & FOURIER & WEATHER EXTERNAL REGRESSORS

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/results/' # directory for the results of experiments

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set
wxregs_train<-lapply(list('tempm_train.csv','hum_train.csv','pressurem_train.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for train set
wxregs_test<-lapply(list('tempm_test.csv','hum_test.csv','pressurem_test.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for test set

# horizontal predictions
test_pred_hfw<-arima_h(train,test,batch=28,freq=365.25*48,freqs=c(48,7*48),f_K=dec_bc_K_h,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE,box_cox = TRUE) # horizontal prediction
save(data=test_pred_hfw,path=paste(exp_dir,'dec,bc,fwreg,arima_h.csv',sep='')) # write results

# vertical predictions
test_pred_vfw<-arima_h(train,test,batch=28,freq=365.25*48,freqs=c(7),f_K=dec_bc_K_v,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=TRUE,box_cox = TRUE) # vertical prediction
save(data=test_pred_vfw,path=paste(exp_dir,'dec,bc,fwreg,arima_v.csv',sep='')) # write results
