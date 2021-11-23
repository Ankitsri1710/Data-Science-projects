#loading dataset
wbcd<-read.csv(file.choose(),header = T)
# Exploratory data analysis
library(Hmisc)
describe(wbcd)
mydata<-wbcd[,-1]
str(mydata)
mydata$diagnosis<-as.factor(mydata$diagnosis)
summary(mydata)

library(GGally)
ggcorr(mydata[-1])
corr_matrix<-cor(mydata[,2:ncol(mydata)])
corrplot::corrplot(corr_matrix,order = "hclust",t1.cex=1,addrect = 5)
attach(mydata)
#Splitting data
library(caTools)
set.seed(0)
split<-sample.split(mydata$diagnosis,SplitRatio = 0.8)
train<-subset(mydata,split==T)
test<-subset(mydata,split==F)

# Adaboost
library(adabag)
adaboost<-boosting(diagnosis~.,data = train,boos = TRUE)
print(adaboost)
adaboost$importance
importanceplot(adaboost)
# Prediction on test data
test_pred<-predict(adaboost,test)
test_pred$confusion
test_acc<-mean(test$diagnosis==test_pred$class)
test_acc
data.frame(test$diagnosis,test_pred$class,test_pred$prob)
# Model evaluation on training data
train_pred<-predict(adaboost,train)
train_pred$confusion
train_acc<-mean(train$diagnosis==train_pred$class)
train_acc
data.frame(train$diagnosis,train_pred$class,train_pred$prob)

wbcd<-wbcd[,-1]
wbcd$diagnosis<-ifelse(wbcd$diagnosis=="B",1,0)
set.seed(0)
split2<-sample.split(wbcd$diagnosis,SplitRatio = 0.8)
train_wbcd<-subset(wbcd,split2==T)
test_wbcd<-subset(wbcd,split2==F)
# Extreme gradiant boosting
library(xgboost)
train_y<-train_wbcd[,1]

train_x<-data.matrix(train_wbcd[,-1])

test_y<-test_wbcd[,1]
test_x<-data.matrix(test_wbcd[,-1])

xgb_train<-xgb.DMatrix(data=train_x,label=train_y)
xgb_test<-xgb.DMatrix(data=test_x,label=test_y)
# Buiding model
xgboosting<-xgboost(data = xgb_train,nrounds = 100,max.depth=8
                    ,objective = "multi:softmax", eta = 0.3,num_class = 2,verbose = T)
print(xgboosting)
# Model evaluation on test
xgbPredict_test<-predict(xgboosting,xgb_test)
table(test_y,xgbPredict_test)
Xgtest_acc<-mean(test_y==xgbPredict_test)
Xgtest_acc
# Model Evaluation on train
xgbPredict_train<-predict(xgboosting,xgb_train)
table(train_y,xgbPredict_train)
Xgtrain_acc<-mean(train_y==xgbPredict_train)
Xgtrain_acc
