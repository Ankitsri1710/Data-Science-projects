# Loading dataset
zoo<-read.csv(file.choose(),header = T)
summary(zoo)
prop.table(table(zoo$type))
str(zoo)
attach(zoo)
# Spliting of data
zoo$type<-as.factor(zoo$type)
train_zoo<-zoo[1:80,2:17]
test_zoo<-zoo[80:101,2:17]

train_label<-zoo[1:80,18]
test_label<-zoo[80:101,18]
# model building and model evaluation
library(class)
pred_train1<-knn(train = train_zoo,test = train_zoo,cl=train_label,k=5)
pred_train1
table(train_label,pred_train1)
library(gmodels)
CrossTable(train_label,pred_train1,prop.c = F,prop.chisq = F,prop.r = F,prop.t = F)
# model evaluation on train
mean(train_label==pred_train1)
pred_train2<-knn(train = train_zoo,test = train_zoo,cl=train_label,k=8)
table(train_label,pred_train2)
mean(train_label==pred_train2)
pred_train3<-knn(train = train_zoo,test = train_zoo,cl=train_label,k=3)
table(train_label,pred_train3)
mean(train_label==pred_train3)
# model evaluation on test
pred_test1<-knn(train=test_zoo,test = test_zoo,cl=test_label,k=5)
table(test_label,pred_test1)                 
mean(test_label==pred_test1)
pred_test2<-knn(train = test_zoo,test = test_zoo,cl=test_label,k=8)
table(test_label,pred_test2)
mean(test_label==pred_test2)
pred_test3<-knn(train = test_zoo,test = test_zoo,cl=test_label,k=3)
table(test_label,pred_test3)
mean(test_label==pred_test3)
