# loading dataset
salarydata_train<-read.csv(file.choose(),header = T)
salarydata_test<-read.csv(file.choose(),header = T)
# Exploratory data analysis
library(Hmisc)
describe(salarydata_train)
describe(salarydata_test)
salarydata_train$Salary<-as.factor(salarydata_train$Salary)
salarydata_test$Salary<-as.factor(salarydata_test$Salary)
attach(salarydata_train)
library(ggplot2)
 # Visualization
v1<-ggplot(salarydata_train,aes(x=hoursperweek,fill=Salary,color=Salary))+
  geom_histogram(bindwidth=1)+labs(title = 'Hours per week distribution by Salary')
v1+theme_bw()

v2<-ggplot(salarydata_train,aes(x=capitalgain,fill=Salary,color=Salary))+
  geom_histogram(bindwidth=1)+labs(title = 'Capital gain by Salary')
v2+theme_bw()

v3<-ggplot(salarydata_train,aes(x=age,fill=Salary,color=Salary))+
  geom_freqpoly()+labs(title = 'Age by Salary')
v3+theme_bw()

prop.table(table(salarydata_train$Salary))*100

prop.table(table(salarydata_test$Salary))*100

x<-salarydata_train[,-14]
y<-salarydata_train$Salary
# Model building
library(e1071)
model<-naiveBayes(x,y)
model

# Model evaluation
Predict_test<-predict(model,newdata = salarydata_test)
Predict_test
table(salarydata_test$Salary,Predict_test)
library(gmodels)
CrossTable(salarydata_test$Salary,Predict_test,dnn = c('Actual','Predicted'))
Test_acc<-mean(salarydata_test$Salary==Predict_test)
Test_acc

Predict_train<-predict(model,newdata = salarydata_train)
table(salarydata_train$Salary,Predict_train)
CrossTable(salarydata_train$Salary,Predict_train,dnn = c('Actual','Predicted'))
Train_acc<-mean(salarydata_train$Salary==Predict_train)
Train_acc

Predicted_Response<-Predict_test
updated_test<-cbind(salarydata_test,Predicted_Response)

Predicted_response<-Predict_train
updated_train<-cbind(salarydata_train,Predicted_response)
