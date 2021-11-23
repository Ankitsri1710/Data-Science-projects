#loading data
fraud_check<-read.csv(file.choose(),header = T)
attach(fraud_check)
str(fraud_check)
fraud_check$Undergrad<-as.factor(fraud_check$Undergrad)
fraud_check$Marital.Status<-as.factor(fraud_check$Marital.Status)
fraud_check$Urban<-as.factor(fraud_check$Urban)
summary(fraud_check)
library(Hmisc)
describe(fraud_check)
mean(is.na(fraud_check))
boxplot(fraud_check[,c(3,4)],col = rainbow(2))$out
boxplot(fraud_check[,5],col = "dodgerblue4",xlab="work Experience")$out

fraud_check$Taxable.Income<-ifelse(Taxable.Income<=30000,"Risky","Good")
str(fraud_check)
fraud_check$Taxable.Income<-as.factor(fraud_check$Taxable.Income)
table(fraud_check$Taxable.Income)

norm<-function(x){
  return((x-min(x))/(max(x)-min(x)))
}
fraud_check$City.Population<-norm(fraud_check$City.Population)
fraud_check$Work.Experience<-norm(fraud_check$Work.Experience)

library(caTools)
set.seed(0)
split<-sample.split(fraud_check$Taxable.Income,SplitRatio = 0.7)
train<-subset(fraud_check,split==TRUE)
test<-subset(fraud_check,split==FALSE)

library(C50)
Decision_tree<-C5.0(Taxable.Income~.,train)
summary(Decision_tree)
windows()
plot(Decision_tree)
dt_test_pred<-predict(Decision_tree,test)
table(test$Taxable.Income,dt_test_pred)
dt_test_acc<-mean(test$Taxable.Income==dt_test_pred)
dt_test_acc
dt_train_pred<-predict(Decision_tree,train)
table(train$Taxable.Income,dt_train_pred)
dt_train_acc<-mean(train$Taxable.Income==dt_train_pred)
dt_train_acc

library(randomForest)
Randomforest<-randomForest(Taxable.Income~.,train,importance=TRUE)
print(Randomforest)
plot(Randomforest)
varImpPlot(Randomforest)
test_pred<-predict(Randomforest,test)
test_acc<-mean(test$Taxable.Income==test_pred)
test_acc

train_pred<-predict(Randomforest,train)
train_acc<-mean(train$Taxable.Income==train_pred)
train_acc
library(gmodels)
CrossTable(test$Taxable.Income,test_pred,
           dnn = c("Actual","Predicted"),prop.chisq = F,prop.c = F,prop.r = F)
CrossTable(train$Taxable.Income,train_pred,
           dnn = c("Actual","Predicted"),prop.r = F,prop.c = F,prop.chisq = F)
