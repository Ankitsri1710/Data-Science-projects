# Loading dataset
salarydata_train<-read.csv(file.choose(),header = T)
salarydata_test<-read.csv(file.choose(),header = T)
# Data Preprocessing
sum(is.na(salarydata_test))
sum(is.na(salarydata_train))
str(salarydata_train)
salarydata_train$workclass<-as.factor(salarydata_train$workclass)
salarydata_train$maritalstatus<-as.factor(salarydata_train$maritalstatus)
salarydata_train$occupation<-as.factor(salarydata_train$occupation)
salarydata_train$relationship<-as.factor(salarydata_train$relationship)
salarydata_train$race<-as.factor(salarydata_train$race)
salarydata_train$sex<-as.factor(salarydata_train$sex)
salarydata_train$native<-as.factor(salarydata_train$native)
salarydata_train$Salary<-as.factor(salarydata_train$Salary)
salarydata_train$education<-as.factor(salarydata_train$education)
# Exploratory data analysis
round(prop.table(table(salarydata_train$Salary))*100,digits=2)
round(prop.table(table(salarydata_test$Salary))*100,digits = 2)
# Visualization
library(ggplot2)
v1<-ggplot(salarydata_train,aes(x=hoursperweek,fill=Salary,color=Salary))+
  geom_histogram(bindwidth=1)+labs(title = 'Hours per week distribution by Salary')
v1+theme_bw()

v2<-ggplot(salarydata_train,aes(x=capitalgain,fill=Salary,color=Salary))+
  geom_histogram(bindwidth=1)+labs(title = 'Capital gain by Salary')
v2+theme_bw()

v3<-ggplot(salarydata_train,aes(x=age,fill=Salary,color=Salary))+
  geom_freqpoly()+labs(title = 'Age by Salary')
v3+theme_bw()
#Scaling of training data. 
salarydata_train$age<-scale(salarydata_train$age)
salarydata_train$educationno<-scale(salarydata_train$educationno)
salarydata_train$capitalgain<-scale(salarydata_train$capitalgain)
salarydata_train$hoursperweek<-scale(salarydata_train$hoursperweek)
salarydata_train$capitalloss<-scale(salarydata_train$capitalloss)
# Scaling of testing data.
salarydata_test$age<-scale(salarydata_test$age)
salarydata_test$educationno<-scale(salarydata_test$educationno)
salarydata_test$capitalgain<-scale(salarydata_test$capitalgain)
salarydata_test$hoursperweek<-scale(salarydata_test$hoursperweek)
salarydata_test$capitalloss<-scale(salarydata_test$capitalloss)
attach(salarydata_test)
attach(salarydata_train)
# Model Building
library(kernlab)
salary_classifier<-ksvm(Salary~.,data=salarydata_train,kernel="rbfdot")
salary_classifier
pred_test<-predict(salary_classifier,salarydata_test)
pred_test
table(pred_test)
# Model evaluation on test data
library(gmodels)
CrossTable(salarydata_test$Salary,pred_test,dnn = c('Actual','Predicted'))
mean(salarydata_test$Salary==pred_test)

# Model evaluation on train data
pred_train<-predict(salary_classifier,salarydata_train)
CrossTable(salarydata_train$Salary,pred_train)
mean(salarydata_train$Salary==pred_train)

Predicted_Response<-pred_test
Newdata_test<-cbind(salarydata_test,Predicted_Response)

PREDICTED_RESPONSE<-pred_train
Newdata_train<-cbind(salarydata_train,PREDICTED_RESPONSE)


