
load<-function(path,index='date'){
  data<-read.csv(path,header=TRUE,row.names=index,sep=',',dec='.',check.names=FALSE) # load data
  return(data)
}

save<-function(data,path){
  data=cbind(date=rownames(data),data) # add rownames as a first column
  write.csv(data,file=path,quote = FALSE,row.names=FALSE) # write data
}