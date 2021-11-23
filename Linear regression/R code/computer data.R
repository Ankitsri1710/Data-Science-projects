# Loading dataset
computer_Data<-read.csv(file.choose(),header = T)
View(computer_Data)
# Data preprocessing
computer_Data['cd']<-ifelse(computer_Data$cd=='yes',1,0)
computer_Data['multi']<-ifelse(computer_Data$multi=='yes',1,0)
computer_Data['premium']<-ifelse(computer_Data$premium=='yes',1,0)
computer_Data<-computer_Data[-1]
# Exploratory data analysis
attach(computer_Data)
summary(computer_Data)
library(Hmisc)
describe(computer_Data)
boxplot(computer_Data[,-c(6,7,8)],col=rainbow(10),horizontal = F,
        main="Boxplot before outlier treatment")$out
library(robustHD)
computer_Data$price<-winsorize(price)
computer_Data$hd<-winsorize(hd)
boxplot(computer_Data[,-c(6,7,8)],col=rainbow(10),main="Boxplot after outlier treatment")

dotplot(price,main='Dotplot of price')
dotplot(speed,main='Dotplot of speed')
dotplot(ram,main='Dotplot of ram')
dotplot(hd,main='Dotplot of hd')

hist(computer_Data[,-c(6,7,8)])
# Bivariate analysis
library(ggplot2)
v1<-ggplot(computer_Data,aes(x=speed,fill=price,color=price))+
  geom_histogram(bindwidth=5)+labs(title = "speed of computer by price constraint")
v1+theme_bw()
library(GGally)
ggpairs(computer_Data)
ggcorr(computer_Data)
# subset selection technique.
library(leaps)
# Forward subset selection
reg_forw<-regsubsets(price~.,data = computer_Data,nvmax = 10,method = 'forward')
summary(reg_forw)
which.max(summary(reg_forw)$adjr2)
# Model building
reg<-lm(price~.,data = computer_Data)
summary(reg)

pred<-data.frame(predict(reg,interval = 'predict'))
pred
rms<-sqrt(mean(reg$residuals^2))
rms
# Model evaluation
plot(reg)
cor(pred$fit,price)
# Transformation
# Log transformation
reg_log<-lm(price~log(speed)+log(screen)+log(hd)+log(ram)+cd+multi+premium+ads+trend,data = computer_Data)
summary(reg_log)
pred_log<-data.frame(predict(reg_log,interval = 'predict'))
rms_log<-sqrt(mean(reg_log$residuals^2))
rms_log
# Evaluation
plot(reg_log)
cor(pred_log$fit,price)
cor(computer_Data)
library(car)
influenceIndexPlot(reg,id.n=5)
influencePlot(reg,id.n=5,col='blue')
reg1<-lm(price~.,data = computer_Data[-c(80,1507),])
summary(reg1)
pred1<-data.frame(predict(reg1,interval = 'predict'))
pred1
rms1<-sqrt(mean(reg1$residuals^2))
rms1
influenceIndexPlot(reg1,id.n=3)
influencePlot(reg1,id.n=3)

reg2<-lm(price~.,data = computer_Data[-c(80,1507,1806,1992,6186),])
summary(reg2)
pred2<-data.frame((predict(reg2,interval = 'predict')))
pred2
rms2<-sqrt(mean(reg2$residuals^2))
rms2
plot(reg2)
influencePlot(reg2,id.n=2)
reg3<-lm(price~.,data = computer_Data[-c(80,1507,1806,1992,6186,2097,4),] )
summary(reg3)
pred3<-data.frame(predict(reg3,interval = 'predict'))
pred3
plot(reg3)
rms3<-sqrt(mean(reg3$residuals^2))
rms3
# So our best model is reg2 as it has least error
mydata<-computer_Data[-c(80,1507,1806,1992,6186,2097,4),]
# spliting of data 
n<-nrow(mydata)
n1<-n*0.7
n2<-(n-n1)
train_ind<-sample(1:n,n1)
train<-mydata[train_ind,]
test<-mydata[-train_ind,]
# train model building
reg_train<-lm(price~.,data = train)
summary(reg_train)
rms_train<-sqrt(mean(reg_train$residuals^2))
rms_train
# test model building
reg_test<-lm(price~.,data = test)

summary(reg_test)
rms_test<-sqrt(mean(reg_test$residuals^2))
rms_test

New_dataset<-cbind(mydata,pred3)
