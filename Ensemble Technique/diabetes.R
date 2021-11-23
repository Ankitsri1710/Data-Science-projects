#loading dataset
diabetes<-read.csv(file.choose(),header=T)
# Exploratory data analysis
library(Hmisc)
describe(diabetes)
boxplot(diabetes[,-9],col = rainbow(9))$out
sum(is.na(diabetes))
str(diabetes)
diabetes$Class.variable<-as.factor(diabetes$Class.variable)
attach(diabetes)
library(GGally)
ggpairs(diabetes)
ggcorr(diabetes)
library(ggplot2)
attach(diabetes)
round(prop.table(table(diabetes$Class.variable))*100,1)
v1<-ggplot(diabetes,aes(x=Body.mass.index,fill=Outcome,color=Outcome))+geom_histogram(binwidth = 1)+
  labs(title = 'Distribution of BMI by Diabetic Class')
v1+theme_bw()
v2<-ggplot(diabetes,aes(x=Number.of.times.pregnant,fill=Outcome,color=Outcome))+
  geom_histogram(binwidth = 1)+labs(title = 'Distribution of Pregnancy cases by Diabetic Class')
v2+theme_bw()
v3<-ggplot(diabetes,aes(x=Age,fill=Outcome,color=Outcome))+geom_histogram()+
  labs(title = 'Distribution of Age by Diabetic class')
v3+theme_bw()
v4<-ggplot(diabetes,aes(x=Triceps.skin.fold.thickness,fill=Outcome,color=Outcome))+geom_histogram()+
  labs(title = 'Distribution of Triceps skin fold thickness by Diabetic class')
v4+theme_bw()
v5<-ggplot(diabetes,aes(x=Diastolic.blood.pressure,fill=Outcome,color=Outcome))+geom_histogram()+
  labs(title = 'Distribution of Diastolic Blood Pressure by Diabetic class')
v5+theme_bw()
v6<-ggplot(diabetes,aes(x=Diabetes.pedigree.function,fill=Outcome,color=Outcome))+geom_histogram()+
  labs(title = 'Distribution of Diabetes pedigree function by Diabetic class')
v6+theme_bw()
v7<-ggplot(diabetes,aes(x=Plasma.glucose.concentration,fill=Outcome,color=Outcome))+geom_histogram()+
  labs(title = 'Distribution of Plasma Glucose Concentration by Diabetic class')
v7+theme_bw()
# Spliting of data
library(caTools)
set.seed(0)
SPLIT<-sample.split(diabetes$Class.variable,SplitRatio = 0.85)
diabetes_train<-subset(diabetes,SPLIT==TRUE)
diabetes_test<-subset(diabetes,SPLIT==FALSE)
# Model building
library(adabag)
adaboost_diabetes<-boosting(Class.variable~.,diabetes_train,boos = TRUE)
adaboost_diabetes$importance
importanceplot(adaboost_diabetes)
print(adaboost_diabetes$trees[1])
print(adaboost_diabetes$prob)
# Model evaluation on test data
test_predict<-predict(adaboost_diabetes,diabetes_test)
table(diabetes_test$Class.variable,test_predict$class)
mean(diabetes_test$Class.variable==test_predict$class)
head(data.frame(diabetes_test$Class.variable,test_predict$prob,test_predict$error,test_predict$class))
# Model evaluation on train data
train_predict<-predict(adaboost_diabetes,diabetes_train)
table(diabetes_train$Class.variable,train_predict$class)
mean(diabetes_train$Class.variable==train_predict$class)
head(data.frame(diabetes_train$Class.variable,train_predict$prob,
                train_predict$error,train_predict$class))
# Gradient boosting
library(gbm)
#model building
GBM<-gbm(Class.variable~.,data = diabetes_train,distribution = "multinomial"
         ,n.trees = 1000,interaction.depth = 1,shrinkage = 0.1)
print(GBM)
pred_GBM_test<-predict(GBM,newdata = diabetes_test,n.trees = 1000
                           ,type = 'response')
labels<-colnames(pred_GBM_test)[apply(pred_GBM_test,1,which.max)]
results<-data.frame(diabetes_test$Class.variable,labels)
# Model evaluation on test data
library(gmodels)
CM<-CrossTable(diabetes_test$Class.variable,labels,data = diabetes_test,prop.r = F,prop.c = F
               ,prop.t = F,prop.chisq = F,dnn = c("AcTUAL","PREDICTED"))
barplot(table(diabetes_test$Class.variable,labels),beside = T,col = c("Orange",'Lightgreen'))
test_acc<-mean(diabetes_test$Class.variable==labels)
test_acc
# Model evaluation in train data
pred_GBM_train<-predict(GBM,newdata = diabetes_train,n.trees = 1000
                        ,type = 'response')
response<-colnames(pred_GBM_train)[apply(pred_GBM_train,1,which.max)]
train_result<-data.frame(diabetes_train$Class.variable,response)
CMTrain<-CrossTable(diabetes_train$Class.variable,response,data=diabetes_train,prop.r = F
                    ,prop.c = F,prop.t = F,prop.chisq = F,dnn = c('ACTUAL','PREDICTED'))
barplot(table(diabetes_train$Class.variable,response),beside = T,col = c('Dodgerblue4','lightgreen'))
acc_train<-mean(diabetes_train$Class.variable==response)
acc_train

# Extreme gradient boosting
str(diabetes)
mydata<-diabetes
mydata$Class.variable<-ifelse(mydata$Class.variable=="YES",1,0)
str(mydata)
set.seed(1)
split<-sample.split(mydata,SplitRatio = 0.85)
train<-subset(mydata,split==T)
test<-subset(mydata,split==F)

train_y<-train[,9]
train_x<-data.matrix(train[,-9])

test_y<-test[,9]
test_x<-data.matrix(test[,-9])

library(xgboost)
xgbtrain<-xgb.DMatrix(data = train_x,label=train_y)
xgbtest<-xgb.DMatrix(data = test_x,label=test_y)
# Model building
xgboosting<-xgboost(data = xgbtrain,nrounds = 500,max.depth=12
                    ,objective="multi:softmax",num_class=2,verbose = 2)
# model evaluation on test
xgbtest_pred<-predict(xgboosting,xgbtest)
table(test_y,xgbtest_pred)
xgbtest_acc<-mean(test_y==xgbtest_pred)
xgbtest_acc
# model evaluation on train
xgbtrain_pred<-predict(xgboosting,xgbtrain)
table(train_y,xgbtrain_pred)
xgbtrain_acc<-mean(train_y==xgbtrain_pred)
xgbtrain_acc
