#library(ISLR)
#data(package="ISLR")
#Carseats<-Carseats

company_data<-read.csv(file.choose(),header = T)
summary(company_data)
hist(company_data$Sales,probability = T)
lines(density(company_data$Sales))
boxplot(company_data$Sales,horizontal = T,col="Dodgerblue4",main='Boxplot of sales')$out
boxplot(company_data[,c(-1,-7,-10,-11)],horizontal = F,col = rainbow(11))$out
hist(company_data[,c(-1,-7,-10,-11)])
attach(company_data)

sum(is.na(company_data))
library(Hmisc)
describe(company_data)

company_data$Sales<-ifelse(Sales>=10,"HIGH","LOW")
table(company_data$Sales)
str(company_data)
company_data$Sales<-as.factor(company_data$Sales)
company_data$ShelveLoc<-as.factor(company_data$ShelveLoc)
company_data$Urban<-as.factor(company_data$Urban)
company_data$US<-as.factor(company_data$US)
library(ggplot2)
v1<-ggplot(company_data,aes(x=Income,fill=Sales,color=Sales))+geom_histogram(bindwidth=1)+
  labs(title = 'Distribution of Income by sales')
v1+theme_bw()
v2<-ggplot(company_data,aes(x=CompPrice,fill=Sales,color=Sales))+geom_histogram(bindwidth=1)+
  labs(title = 'Distribution of CompPrice by sales')
v2+theme_bw()
v3<-ggplot(company_data,aes(x=Advertising,fill=Sales,color=Sales))+geom_histogram(bindwidth=1)+
  labs(title = 'Distribution of Advertising by sales')
v3+theme_bw()
v4<-ggplot(company_data,aes(x=Population,fill=Sales,color=Sales))+geom_histogram(bindwidth=1)+
  labs(title = 'Distribution of Population by sales')
v4+theme_bw()
v5<-ggplot(company_data,aes(x=Price,fill=Sales,color=Sales))+geom_histogram(bindwidth=1)+
  labs(title = 'Distribution of Price by sales')
v5+theme_bw()
v6<-ggplot(company_data,aes(x=Age,fill=Sales,color=Sales))+geom_histogram(bindwidth=1)+
  labs(title = 'Distribution of Age by sales')
v6+theme_bw()
v7<-ggplot(company_data,aes(x=Education,fill=Sales,color=Sales))+geom_histogram(bindwidth=1)+
  labs(title = 'Distribution of Education by sales')
v7+theme_bw()
# Model Buiding
library(C50)
decision_tree<-C5.0(Sales~.,data = company_data)
summary(decision_tree)
windows()
plot(decision_tree)
acc<-((70+314)/(70+9+7+314))
acc
company_data_rand<-company_data[order(runif(400)),]

train<-company_data_rand[1:320,]
test<-company_data_rand[321:400,]

prop.table(table(company_data_rand$Sales))
prop.table(table(train$Sales))
prop.table(table(test$Sales))

library(C50)
model<-C5.0(train[,-1],train$Sales)
plot(model)

summary(model)

test_result<-predict(model,test)
test_result
test_acc<-mean(test$Sales==test_result)
test_acc

train_result<-predict(model,train)
train_result
train_acc<-mean(train$Sales==train_result)
train_acc

table(train$Sales,train_result)
table(test$Sales,test_result)

library(gmodels)
CrossTable(test$Sales,test_result,dnn = c("Actual","Predicted"))
CrossTable(train$Sales,train_result,dnn = c("Actual","Predicted"))


# Randomforest
library(randomForest)
randomforest<-randomForest(Sales~.,data = company_data,importance=TRUE)
randomforest
plot(randomforest)
varImpPlot(randomforest)

train_rfmodel<-randomForest(Sales~.,data = train,importance=TRUE)
train_rfmodel
plot(train_rfmodel)
varImpPlot(train_rfmodel)
result_train<-predict(train_rfmodel,train)
acc_train<-mean(train$Sales==result_train)
acc_train
table(train$Sales,result_train)
result_test<-predict(train_rfmodel,test)
acc_test<-mean(test$Sales==result_test)
acc_test
table(test$Sales,result_test)

Predicted_train<-result_train
Actual_train<-train$Sales
updated_train<-cbind(Actual_train,Predicted_train)

Predicted_test<-result_test
Actual_test<-test$Sales
Updated_test<-cbind(Actual_test,Predicted_test)
