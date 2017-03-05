library(forecast)


ets=function(train,test,hor=1,batch=7,freq=7){
  
}














train=read.csv('C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/train.csv',header=TRUE,sep=',',dec='.')
test=read.csv('C:/Users/SABA/Google Drive/mtsg/code/load_forecast/data/test.csv',header=TRUE,sep=',',dec='.')


batch=7
hor=24
train_ts=ts(train,frequency=24)
test_ts=ts(test,frequency=24)





train_ts=ts(train[[2]],frequency=findfrequency(train[[2]]))
test_ts=ts(test[[2]],frequency=findfrequency(test[[2]]))
fit_train_ts=ets(train_ts)
fit_test_ts=ets(test_ts,model=fit_train_ts)
train_ts_pred=fitted(fit_train_ts)
test_ts_pred=fitted(fit_test_ts)
ts.plot(train_ts,train_ts_pred,col=c('black','red'),lty=c(5,1))
ts.plot(test_ts,test_ts_pred,col=c('black','red'),lty=c(5,1))


train_xts=xts(train[[10]],as.POSIXct(train[["date"]]))
test_xts=xts(test[[10]],order.by=as.POSIXct(test$date))
fit_train_xts=ets(train_xts)
fit_test_xts=ets(test_xts,model=fit_train_xts)
train_xts_pred=fitted(fit_train_xts)
test_xts_pred=fitted(fit_test_xts)
ts.plot(train_xts,train_xts_pred,col=c('black','red'),lty=c(5,1))
ts.plot(test_xts,test_xts_pred,col=c('black','red'),lty=c(5,1))

# Multi-step, re-estimation
h <- 5
train <- window(hsales,end=1989.99)
test <- window(hsales,start=1990)
n <- length(test) - h + 1
fit <- auto.arima(train)
order <- arimaorder(fit)
fcmat <- matrix(0, nrow=n, ncol=h)
for(i in 1:n)
{  
  x <- window(hsales, end=1989.99 + (i-1)/12)
  refit <- Arima(x, order=order[1:3], seasonal=order[4:6])
  fcmat[i,] <- forecast(refit, h=h)$mean
}



train_ts=ts(train[[2]],frequency=findfrequency(train[[2]]))
test_ts=ts(test[[2]],frequency=findfrequency(test[[2]]))
fit_train_ts=ets(train_ts)
fit_test_ts=ets(test_ts,model=fit_train_ts)
train_ts_pred=fitted(fit_train_ts)
test_ts_pred=fitted(fit_test_ts)
ts.plot(train_ts,train_ts_pred,col=c('black','red'),lty=c(5,1))
ts.plot(test_ts,test_ts_pred,col=c('black','red'),lty=c(5,1))


train_xts=xts(train[[10]],as.POSIXct(train[["date"]]))
test_xts=xts(test[[10]],order.by=as.POSIXct(test$date))
fit_train_xts=ets(train_xts)
fit_test_xts=ets(test_xts,model=fit_train_xts)
train_xts_pred=fitted(fit_train_xts)
test_xts_pred=fitted(fit_test_xts)
ts.plot(train_xts,train_xts_pred,col=c('black','red'),lty=c(5,1))
ts.plot(test_xts,test_xts_pred,col=c('black','red'),lty=c(5,1))

# Multi-step, re-estimation
h <- 5
train <- window(hsales,end=1989.99)
test <- window(hsales,start=1990)
n <- length(test) - h + 1
fit <- auto.arima(train)
order <- arimaorder(fit)
fcmat <- matrix(0, nrow=n, ncol=h)
for(i in 1:n)
{  
  x <- window(hsales, end=1989.99 + (i-1)/12)
  refit <- Arima(x, order=order[1:3], seasonal=order[4:6])
  fcmat[i,] <- forecast(refit, h=h)$mean
}





set.seed(1)
library(lubridate)
index <- ISOdatetime(2010,1,1,0,0,0)+1:8759*60*60
month <- month(index)
hour <- hour(index)
usage <- 1000+10*rnorm(length(index))-25*(month-6)^2-(hour-12)^2
usage <- ts(usage,frequency=24)

#Create monthly dummies.  Add other xvars to this matrix
xreg <- model.matrix(~as.factor(month))[,2:12]
colnames(xreg) <- c('Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')



