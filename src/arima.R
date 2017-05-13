library(forecast)
source('dataprep.R')


f_ords<-function(train,freq=48,freqs,ords,dec=FALSE,bc=FALSE){
  train<-c(t(train)) # flatten train set
  aicc_best<-Inf # best aicc statistic
  param_best<-NULL # best parameters
  bc_lambda<-if (bc) BoxCox.lambda(train,method='loglik') else NULL # estimate lambda for Box-Cox transformation
  for (i in 1:nrow(ords)){ # for each combination of orders
    ord<-unlist(ords[i,]) # combination of orders
    xregs_train<-fourier(msts(train,seasonal.periods=freqs),K=ord) # fourier terms for particular multi-seasonal time series
    if (dec) { # if decompose first
      model<-stlm(ts(train,frequency = freq),method='arima',xreg=xregs_train,s.window=7,robust=TRUE,trace=TRUE,lambda = bc_lambda)$model  # find best arima model after decomposition      
    }
    else{ # dont decompose
      model<-auto.arima(ts(train,frequency = freq),xreg=xregs_train,seasonal=FALSE,trace=TRUE,lambda = bc_lambda) # find best arima model 
    }
    if (model$aicc<aicc_best){ # if there is an improvement in aicc statistic
      ord_best<-ord # save these orders
      aicc_best<-model$aicc # save new best aicc value
    }
  }
  return(ord_best)
}

arima<-function(train,test,hor=1,batch=7,freq=48,freqs=NULL,ord=NULL,wxregs_train=NULL,wxregs_test=NULL,bc=FALSE,dec=FALSE){
  test_pred<-data.frame(matrix(data=NA,nrow=nrow(test),ncol=ncol(test),dimnames=list(rownames(test),colnames(test))),check.names = FALSE) # initialize matrix for predictions
  train<-c(t(train)) # flatten train set
  test<-c(t(test)) # flatten test set
  if (is.null(ord)){ # not considering multiple seasonalities
    fxregs_train<-NULL
    fxregs_test<-NULL
    seasonal<-TRUE
  }
  else { # considering multiple seasonalities
    fxregs<-fourier(msts(c(train,test),seasonal.periods=freqs),K=ord)
    fxregs_train<-fxregs[1:length(train),]
    fxregs_test<-fxregs[1:length(test),]
    seasonal<-FALSE
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
  #train<-c(t(train))[1:(48*28*2)]
  #wxregs_train<-wxregs_train[1:(48*28*2),]
  xregs_train<-cbind(fxregs_train,wxregs_train) # combine fourier & weather into one matrix for train set
  xregs_test<-cbind(fxregs_test,wxregs_test) # combine fourier & weather into one matrix for test set
  xregs=NULL # default covariates
  xregs_pred=NULL # default covariates for predictions
  model<-NULL
  for (i in seq(0,length(test)-hor,hor)){ # for each window of observations in test set
    train_ts<-ts(c(train,test[seq_len(i)]),frequency=freq) # add new observations from test set to the current train set
    if (!is.null(xregs_train)&!is.null(xregs_test)){ # if considering external regressors
      xregs<-rbind(xregs_train,xregs_test[seq_len(i),]) # add covariates corresponding to new observations
      xregs_pred<-matrix(xregs_test[i+seq_len(hor),],ncol=ncol(xregs_test)) # add covariates for predictions
    }
    if (i%%(batch*hor)==0){ # if its time to retrain
      bc_lambda<-if (bc) BoxCox.lambda(train,method='guerrero') else NULL # estimate lambda for Box-Cox transformation
      if (dec){ # if decomposition is to be applied
        model<-stlm(train_ts,method='arima',xreg=xregs,s.window=7,robust=TRUE,lambda=bc_lambda,biasadj = FALSE,trace=TRUE) # find best model on the current train set
      }
      else { # no decomposition
        model<-auto.arima(train_ts,xreg=xregs,seasonal=FALSE,lambda=bc_lambda,biasadj = FALSE,trace = TRUE) # find best model on the current train set
      }
      cat('training: ',i%/%(batch*hor),'\n') # print number of retrainings and the type of model
    }
    else{ # it is not the time to retrain
      if (dec){
        if (!is.null(xregs))
        {
          model<-stlm(train_ts,model=model$model,modelfunction=function(x, ...) {Arima(x, xreg=xregs, ...)},s.window='periodic',robust=TRUE,lambda=bc_lambda,biasadj = FALSE) # do not train, use current model with new observations  
        }
        else {
          model<-stlm(train_ts,model=model,s.window=7,robust=TRUE,lambda=bc_lambda,biasadj = FALSE) # do not train, use current model with new observations  
        }
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

arima_h<-function(train,test,batch=7,freq=48,freqs=NULL,ord=NULL,wxregs_train=NULL,wxregs_test=NULL,bc=FALSE,dec=FALSE){
  return(arima(train,test,hor=48,batch=batch,freq=freq,freqs=freqs,ord=ord,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec = dec,bc = bc))
}

arima_v<-function(train,test,batch=7,freq=7,ord=NULL,freqs=NULL,wxregs_train=NULL,wxregs_test=NULL,bc=FALSE,dec=FALSE){
  test_pred<-data.frame(matrix(data=NA,nrow=nrow(test),ncol=ncol(test),dimnames=list(rownames(test),colnames(test))),check.names = FALSE) # initialize dataframe for predictions
  for (col in names(train)){
    train_col<-as.data.frame(train[[col]],row.names=rownames(train)) # convert dataframe column to dataframe
    test_col<-as.data.frame(test[[col]],row.names=rownames(test)) # convert dataframe column to dataframe
    colnames(train_col)<-c(col) # set column name to match
    colnames(test_col)<-c(col) # set column name to match
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
    test_pred[[col]]<-arima(train_col,test_col,hor=1,batch=batch,freq=freq,freqs=freqs,ord=ord,wxregs_train=wxregs_train_col,wxregs_test=wxregs_test_col,dec=dec,bc=bc)[[col]] # predictions
    cat('time: ',col,'\n')
  }
  return(test_pred)
}


# ESTIMATE FOURIER ORDERS

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/nocb/arima/data/' # directory containing data

train<-load(paste(data_dir,'train_full.csv', sep='')) # load train set

# horizontal predictions

ords=params<-expand.grid(c(5,10,15,20),c(10,25,50)) # all combinations of fourier orders to try
ord_h<-f_ords(train,freq=365.25*48,freqs=c(48,7*48),ords=ords) # find best fourier coefficients

# vertical predictions

ords<-params<-expand.grid(seq(3)) # all combinations of fourier orders to try
ord_v <- sapply(names(train),function(x) NULL) # initialize empty list for orders
for (col in names(train)){
  train_col<-as.data.frame(train[[col]],row.names=rownames(train)) # convert dataframe column to dataframe
  colnames(train_col)<-c(col) # set column name to match
  ord_v[[col]]<-f_ords(train_col,freq=365.25,freqs=c(7),ords=ords) # find best fourier coefficients  
}


# EXPERIMENTS

data_dir<-'C:/Users/SABA/Google Drive/mtsg/data/experiments/data/' # directory containing data
exp_dir<-'C:/Users/SABA/Google Drive/mtsg/data/experiments/' # directory for the results of experiments

train<-load(paste(data_dir,'train.csv', sep='')) # load train set
test<-load(paste(data_dir,'test.csv', sep='')) # load test set
wxregs_train<-lapply(list('tempm_train.csv','hum_train.csv','pressurem_train.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for train set
wxregs_test<-lapply(list('tempm_test.csv','hum_test.csv','pressurem_test.csv'),function(x) load(paste(data_dir,x,sep=''))) # load weather covariates for test set

train_day<-sapply(paste('D',seq(0,6),sep=''),function(x) NULL) # initialize empty list for train days
test_day<-sapply(paste('D',seq(0,6),sep=''),function(x) NULL) # initialize empty list for test days
wxregs_train_day<-sapply(paste('D',seq(0,6),sep=''),function(x) NULL) # initialize empty list for train weather day
wxregs_test_day<-sapply(paste('D',seq(0,6),sep=''),function(x) NULL) # initialize empty list for test weather day

for (i in 0:6){
  day<-paste('D',i,sep='') # index name
  train_day[[day]]<-load(paste(data_dir,'train_',i,'.csv', sep='')) # load train day
  test_day[[day]]<-load(paste(data_dir,'test_',i,'.csv', sep='')) # load test day
  wxregs_train_day[[day]]<-lapply(list('tempm_train_','hum_train_','pressurem_train_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather for train day
  wxregs_test_day[[day]]<-lapply(list('tempm_test_','hum_test_','pressurem_test_'),function(x) load(paste(data_dir,x,i,'.csv',sep=''))) # load weather for test day
}

params<-data.frame(row.names=c('np,','bc,','dec,','dec,bc,'),'bc'=c(FALSE,TRUE,FALSE,TRUE),'dec'=c(FALSE,FALSE,TRUE,TRUE))

for (name in rownames(params)){
  
  name<-''
  bc<-params[name,]$bc
  dec<-params[name,]$dec

  # NO EXTERNAL REGRESSORS
  
  # 
  test_pred_h<-arima_h(train,test,batch=28,freq=48,dec=dec,bc = bc) # predict values
  save(data=test_pred_h,path=paste(exp_dir,'arima/',name,'arima','.csv',sep='')) # write results
  
  # ha
  test_pred_v<-arima_v(train,test,batch=28,freq=7,dec=dec,bc = bc) # predict values
  save(data=test_pred_v,path=paste(exp_dir,'arima/','ha,',name,'arima','.csv',sep='')) # write results
  
  # wa
  for (i in 0:6){ # for each day
    day<-paste('D',i,sep='') # index name
    test_pred_hwi<-arima_h(train_day[[day]],test_day[[day]],batch=4,freq=48,dec=dec,bc = bc) # horizontal predictions for this day
    save(data=test_pred_hwi,path=paste(exp_dir,'arima/',name,'arima_',i,'.csv',sep='')) # write results
  }
  
  # wa & ha
  for (i in 0:6){ # for each day
    day<-paste('D',i,sep='') # index name
    test_pred_vwi<-arima_v(train_day[[day]],test_day[[day]],batch=4,freq=52,dec=dec,bc = bc) # vertical predictions for this day
    save(data=test_pred_vwi,path=paste(exp_dir,'arima/','ha,',name,'arima_',i,'.csv',sep='')) # write results
  }
  
  #  FOURIER EXTERNAL REGRESSORS

  # 
  test_pred_hf<-arima_h(train,test,batch=28,freq=365.25*48,freqs=c(48,7*48),ord=c(3,5),dec=dec,bc = bc) # horizontal prediction
  save(data=test_pred_hf,path=paste(exp_dir,'arimax/',name,'fregs,arimax.csv',sep='')) # write results
  
  # ha
  test_pred_vf<-arima_v(train,test,batch=28,freq=365.25,freqs=c(7),ord=3,dec=dec,bc = bc) # horizontal prediction
  save(data=test_pred_vf,path=paste(exp_dir,'arimax/','ha,',name,'fregs,arimax.csv',sep='')) # write results
  
  
  # WEATHER EXTERNAL REGRESSORS
  
  # 
  test_pred_hw<-arima_h(train,test,batch=28,freq=48,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=dec,bc = bc) # horizontal prediction
  save(data=test_pred_hw,path=paste(exp_dir,'arimax/',name,'wregs,arimax.csv',sep='')) # write results
  
  # ha
  test_pred_vw<-arima_v(train,test,batch=28,freq=7,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=dec,bc = bc) # vertical prediction
  save(data=test_pred_vw,path=paste(exp_dir,'arimax/','ha,',name,'wregs,arimax.csv',sep='')) # write results
  
  # wa
  for (i in 0:6){ # for each day
    day<-paste('D',i,sep='') # index name
    test_pred_hwi<-arima_h(train_day[[day]],test_day[[day]],batch=4,freq=48,wxregs_train=wxregs_train_day[[day]],wxregs_test=wxregs_test_day[[day]],dec=dec,bc = bc) # horizontal predictions for this day
    save(data=test_pred_hwi,path=paste(exp_dir,'arimax/',name,'wregs,arimax_',i,'.csv',sep='')) # write results
  }
  
  # wa & ha
  for (i in 0:6){ # for each day
    day<-paste('D',i,sep='') # index name
    test_pred_vwi<-arima_v(train_day[[day]],test_day[[day]],batch=4,freq=52,wxregs_train=wxregs_train_day[[day]],wxregs_test=wxregs_test_day[[day]],dec=dec,bc = bc) # horizontal predictions for this day
    save(data=test_pred_vwi,path=paste(exp_dir,'arimax/','ha,',name,'wregs,arimax_',i,'.csv',sep='')) # write results
  }
  
  
  # FOURIER & WEATHER EXTERNAL REGRESSORS
  
  # 
  test_pred_hfw<-arima_h(train,test,batch=28,freq=365.25*48,freqs=c(48,7*48),ord=c(3,5),wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=dec,bc = bc) # horizontal prediction
  save(data=test_pred_hfw,path=paste(exp_dir,'arimax/',name,'fregs,wregs,arimax.csv',sep='')) # write results
  
  # ha
  test_pred_vfw<-arima_v(train,test,batch=28,freq=365.25,freqs=c(7),ord=3,wxregs_train=wxregs_train,wxregs_test=wxregs_test,dec=dec,bc = bc) # vertical prediction
  save(data=test_pred_vfw,path=paste(exp_dir,'arimax/','ha,',name,'fregs,wregs,arimax.csv',sep='')) # write results
}

