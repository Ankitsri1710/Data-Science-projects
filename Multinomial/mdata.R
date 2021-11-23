# Loading dataset
mdata<-read.csv(file.choose(),header = T)
head(mdata)
mdata<-mdata[,c(-1,-2)]
tail(mdata)
# EDA
str(mdata)
attach(mdata)
prop.table(table(female))
prop.table(table(ses))
prop.table(table(schtyp))
prop.table(table(prog))
prop.table(table(honors))
library(Hmisc)
describe(mdata)
par(mfrow=c(2,2))
hist(read,col=rainbow(10))
boxplot(read,col="red",horizontal = T,main='Boxplot of read')
hist(write,col = rainbow(10))
boxplot(write,col = "red",horizontal =T,main='Boxplot of write')
par(mfrow=c(2,2))
hist(math,col=rainbow(10))
boxplot(math,col="red",horizontal = T,main='Boxplot of math')
hist(science,col = rainbow(10))
boxplot(science,col = "red",horizontal =T,main='Boxplot of science')
library(ggplot2)
v1<-ggplot(mdata,aes(x=read,fill=prog,color=prog))+geom_histogram(binwidth = 1)+
  labs(title = 'Distribution reading section by different programs')
v1+theme_bw()
v2<-ggplot(mdata,aes(x=write,fill=prog,color=prog))+geom_histogram(binwidth = 1)+
  labs(title = 'Distribution of writing section by diferent programs')
v2+theme_bw()
v3<-ggplot(mdata,aes(x=math,fill=prog,color=prog))+geom_histogram(bindwidth=1)+
  labs(title = 'Distribution maths in different programs')
v3+theme_bw()
v4<-ggplot(mdata,aes(x=science,fill=prog,color=prog))+geom_histogram(bindwidth=1)+
  labs(title='Distribution of science in different programs')
v4+theme_bw()
mdata$female<-as.factor(mdata$female)
mdata$ses<-as.factor(mdata$ses)
mdata$schtyp<-as.factor(mdata$schtyp)
mdata$prog<-as.factor(mdata$prog)
mdata$honors<-as.factor(mdata$honors)
table(mdata$prog)

n<-nrow(mdata)
n1<-n*0.8
n2<-n-n1
train_index<-sample(1:n,n1)
train<-mdata[train_index,]
test<-mdata[-train_index,]
attach(mdata)
library(nnet)
train$prog<-relevel(train$prog,ref = 'general')
program<-multinom(prog~.,data = train)
summary(program)

x<-summary(program)$coefficients / summary(program)$standard.errors
x

p_value<-(1-pnorm(abs(x),0,1))*2
p_value
summary(program)$coefficients

#odds ratio
exp(coef(program))

Prob<-fitted(program)

test_pred<-predict(program,newdata = test,type="probs")
test_pred

class(test_pred)
test_pred<-data.frame(test_pred)
test_pred["Prediction"]<-NULL

getnames<-function(i){
  return(names(which.max(i)))
}
testpred_namea<-apply(test_pred,1,getnames)
test_pred$Prediction<-testpred_namea
View(test_pred)

#confusion matrix
table(test_pred$Prediction,test$prog)

mean(test_pred$Prediction==test$prog)

barplot(table(test_pred$Prediction,test$prog),beside = T,col = c('orange','Dodgerblue4','lightgreen'),main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")
#train
train_pred<-predict(program,newdata = train,type="probs")
train_pred

class(train_pred)
train_pred<-data.frame(train_pred)
train_pred["Prediction"]<-NULL
trainpred_namea<-apply(train_pred,1,getnames)
train_pred$Prediction<-trainpred_namea
View(train_pred)

#confusion matrix
table(train_pred$Prediction,train$prog)

mean(train_pred$Prediction==train$prog)

barplot(table(train_pred$Prediction,train$prog),beside = T,col = c('orange','Dodgerblue4','lightgreen'),main = "Predicted(X-axis) - Legends(Actual)", ylab ="count")

