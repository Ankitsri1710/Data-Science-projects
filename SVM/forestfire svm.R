# Loading dataset
forestfires<-read.csv(file.choose(),header = T)
attach(forestfires)
str(forestfires)
# Data preprocessing
forestfires$month<-as.factor(forestfires$month)
forestfires$day<-as.factor(forestfires$day)
forestfires$size_category<-as.factor(forestfires$size_category)
# install.packages(Hmisc)
library(Hmisc) # Harrell miscellaneous library for statistical analysis.
describe(forestfires$area)
prop.table(table(forestfires$size_category))
library(ggplot2)
v<-ggplot(forestfires,aes(x=area,fill=size_category,color=size_category))+
  geom_histogram(bindwidth=1)+labs(title = 'Distribution area by category')
v+theme_bw()
# Spliting dataset
library(caTools) # Library for splitting data
split<-sample.split(forestfires$size_category,SplitRatio = 0.8)
train<-subset(forestfires,split==T)
test<-subset(forestfires,split==F)

# Model building
library(kernlab)# library for SVM classifier
size_classification<-ksvm(size_category~.,train,kernel="vanilladot")
size_classification
pred_test<-predict(size_classification,test)
# Model evaluation on test
library(gmodels)
CrossTable(test$size_category,pred_test)
mean(test$size_category==pred_test)

# Model evaluation on train
pred_train<-predict(size_classification,train)
CrossTable(train$size_category,pred_train)
mean(train$size_category==pred_train)

Predicted_Response<-pred_test
newdata<-cbind(test,Predicted_Response)

Predicted_response<-pred_train
Newdata<-cbind(train,Predicted_response)
