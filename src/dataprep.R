
load<-function(path){
  data<-read.csv(path,header=TRUE,row.names='date',sep=',',dec='.') # load data
  return(data)
}