# Loading dataset 
toyota_corolla<-read.csv(file.choose(),header = T)
View(toyota_corolla)
# data preprocesing
mydata<-toyota_corolla[,c(3,4,7,9,13,14,16,17,18)]
View(mydata)
# Exploratory data analysis
summary(mydata)
library(Hmisc)
describe(mydata)
library(moments)
skewness(mydata)
kurtosis(mydata)
library(robustHD)
PRICE<-mydata$Price
Win_PRICE<-winsorize(mydata$Price)
KM<-mydata$KM
Win_KM<-winsorize(mydata$KM)
WEIGHT<-mydata$Weight
Win_WEIGHT<-winsorize(mydata$Weight)
par(mfrow=c(3,3))
hist(PRICE,main = "Histogram of Price",col = "Dodgerblue4")
boxplot(PRICE,main = "Boxplot before outlier treatment",col = "Dodgerblue4",horizontal = T)
boxplot(Win_PRICE,main = "Boxplot after outlier treatment",col = "Yellow",horizontal = T)
hist(KM,main = "Histogram of KM",col = "Red")
boxplot(KM,main="Boxplot before outlier treatment",col = "red",horizontal = T)
boxplot(Win_KM,main="Boxplot after outlier treatment",col = "lightgreen",horizontal = T)
hist(WEIGHT,main = "Histogram of WEIGHT")
boxplot(WEIGHT,main="Boxplot before outlier treatment",horizontal = T)
boxplot(Win_WEIGHT,main="Boxplot after outlier treatment",col = "blue",horizontal = T)


par(mfrow=c(1,1))
plot(Price,KM)
plot(mydata)
library(GGally)
ggpairs(mydata)
ggcorr(mydata)

reg_lin<-lm(Price~.,data=mydata)
summary(reg_lin)

library(car)
influencePlot(reg_lin,id.n=3,data=mydata)
influenceIndexPlot(reg_lin,id.n=3,data=mydata)

reg_lin1<-lm(Price~.,data=mydata[-c(81,222,961),])
summary(reg_lin1)
rms_lin1<-sqrt(mean(reg_lin1$residuals^2))
rms_lin1
plot(reg_lin1)
influencePlot(reg_lin1,id.n=3,data=mydata[-c(81,222,961),])
influenceIndexPlot(reg_lin1,id.n=3,data=mydata[-c(81,222,961),])
reg_lin2<-lm(Price~.,data=mydata[-c(81,222,961,602,148),])
summary(reg_lin2)
rms_lin2<-sqrt(mean(reg_lin2$residuals^2))
rms_lin2
plot(reg_lin2)
influencePlot(reg_lin2,id.n=3,data=mydata[-c(81,222,961,602,148),])
influenceIndexPlot(reg_lin2,id.n=3,data=mydata[-c(81,222,961,602,148),])
reg_lin3<-lm(Price~.,data = mydata[-c(81,222,961,602,148,655,524),])
summary(reg_lin3)
rms_lin3<-sqrt(mean(reg_lin3$residuals^2))
rms_lin3
pred_lin3<-data.frame(predict(reg_lin3,interval = 'predict'))
pred_lin3
plot(reg_lin3)
# Splitting of data
new<-mydata[-c(81,222,961),]
# Spliting data
set.seed(5)
n<-nrow(new)
n1<-n*0.7
n2<-n-n1
train_ind<-sample(1:n,n1)
train<-new[train_ind,]
test<-new[-train_ind,]
 # Training model
reg_train<-lm(Price~.,data = train)
summary(reg_train)
rms_train<-sqrt(mean(reg_train$residuals**2))
rms_train

reg_test<-lm(Price~.,data = test)
summary(reg_test)
rms_test<-sqrt(mean(reg_test$residuals**2))
rms_test

new1<-mydata[-c(81,222,961,602,148),]
# Spliting data
set.seed(5)
n<-nrow(new)
n1<-n*0.7
n2<-n-n1
train_ind<-sample(1:n,n1)
train<-new[train_ind,]
test<-new[-train_ind,]
# Training model
reg_train<-lm(Price~.,data = train)
summary(reg_train)
rms_train<-sqrt(mean(reg_train$residuals**2))
rms_train

reg_test<-lm(Price~.,data = test)
summary(reg_test)
rms_test<-sqrt(mean(reg_test$residuals**2))
rms_test

new2<-mydata[-c(81,222,961,602,148,655,524),]
# Spliting data
set.seed(5)
n<-nrow(new)
n1<-n*0.7
n2<-n-n1
train_ind<-sample(1:n,n1)
train<-new[train_ind,]
test<-new[-train_ind,]
# Training model
reg_train<-lm(Price~.,data = train)
summary(reg_train)
rms_train<-sqrt(mean(reg_train$residuals**2))
rms_train

reg_test<-lm(Price~.,data = test)
summary(reg_test)
rms_test<-sqrt(mean(reg_test$residuals**2))
rms_test
library(corpcor)
cor2pcor(cor(mydata))
library(car)
vif(reg_lin)
# Transformation
reg<-lm(Price~.,data = mydata[,-6])
summary(reg)
influencePlot(reg,id.n=3,data=mydata[,-6])
influenceIndexPlot(reg,id.n=3,data=mydata[,-6])
reg1<-lm(Price~.,data = mydata[-81,-6])
summary(reg1)
rms1<-sqrt(mean(reg1$residuals**2))
rms1
influencePlot(reg1,id.n=3,data=mydata[-81,-6])
reg2<-lm(Price~.,data = mydata[-c(81,222),-6])
summary(reg2)
rms2<-sqrt(mean(reg2$residuals**2))
rms2
influencePlot(reg2,id.n=3,data=mydata[-c(81,222),-6])
reg3<-lm(Price~.,data = mydata[-c(81,222,961),-6])
summary(reg3)
rms3<-sqrt(mean(reg3$residuals**2))
rms3
reg4<-lm(Price~.,data = mydata[-c(81,222,961,602),-6])
summary(reg4)
rms4<-sqrt(mean(reg4$residuals**2))
rms4
plot(reg4)
# Spliting data 
my_data1<-mydata[-81,-6]
set.seed(13)
p<-nrow(my_data)
p1<-p*0.7
p2<-p-p1
train_ind<-sample(1:p,p1)
train<-my_data[train_ind,]
test<-my_data[-train_ind,]
# Training model
reg_train<-lm(Price~.,data = train)
summary(reg_train)
rms_train<-sqrt(mean(reg_train$residuals**2))
rms_train

reg_test<-lm(Price~.,data = test)
summary(reg_test)
rms_test<-sqrt(mean(reg_test$residuals**2))
rms_test

my_data2<-mydata[-c(81,222),-6]
set.seed(13)
p<-nrow(my_data)
p1<-p*0.7
p2<-p-p1
train_ind<-sample(1:p,p1)
train<-my_data[train_ind,]
test<-my_data[-train_ind,]
# Training model
reg_train<-lm(Price~.,data = train)
summary(reg_train)
rms_train<-sqrt(mean(reg_train$residuals**2))
rms_train

reg_test<-lm(Price~.,data = test)
summary(reg_test)
rms_test<-sqrt(mean(reg_test$residuals**2))
rms_test

my_data3<-mydata[-c(81,222,961),-6]
set.seed(13)
p<-nrow(my_data)
p1<-p*0.7
p2<-p-p1
train_ind<-sample(1:p,p1)
train<-my_data[train_ind,]
test<-my_data[-train_ind,]
# Training model
reg_train<-lm(Price~.,data = train)
summary(reg_train)
rms_train<-sqrt(mean(reg_train$residuals**2))
rms_train

reg_test<-lm(Price~.,data = test)
summary(reg_test)
rms_test<-sqrt(mean(reg_test$residuals**2))
rms_test

my_data4<-mydata[-c(81,222,961,602),-6]
set.seed(13)
p<-nrow(my_data)
p1<-p*0.7
p2<-p-p1
train_ind<-sample(1:p,p1)
train<-my_data[train_ind,]
test<-my_data[-train_ind,]
# Training model
reg_train<-lm(Price~.,data = train)
summary(reg_train)
rms_train<-sqrt(mean(reg_train$residuals**2))
rms_train

reg_test<-lm(Price~.,data = test)
summary(reg_test)
rms_test<-sqrt(mean(reg_test$residuals**2))
rms_test
# Subset selection technique
library(leaps)
reg_forw<-regsubsets(Price~.,data = mydata,nvmax=20,method = 'forward')
summary(reg_forw)
which.max(summary(reg_forw)$adjr2)cor(mydata)
coef(reg_forw,7)
summary(reg)
