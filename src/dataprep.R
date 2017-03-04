
load<-function(path){
  data<-read.csv(path,header=TRUE,row.names='date',sep=',',dec='.') # load data
  return(data)
}

save<-function(data,path,index='date'){
  data=cbind(date=rownames(data),data)
  write.csv(data,file=path,quote = FALSE,row.names=FALSE) # write data
}