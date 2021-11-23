# Loading dataset startup
startup<-read.csv(file.choose(),header = T)
View(startup)
summary(startup)
library(Hmisc)
describe(startup)
# Boxplot
boxplot(startup[-4],col=rainbow(4))
# Dot Plot
dotplot(startup$R.D.Spend,xlab ='R.D Spend',main='Dotplot R.D Spend')
dotplot(startup$Administration,xlab='Administration',main='Dotplot Administration')
dotplot(startup$Marketing.Spend,xlab = 'Marketing Spend',main='Dotplot Marketing Spend')
dotplot(startup$Profit,xlab = 'Profit',main='Dotplot of Profit')
# Quartile to Quartile Plot
qqnorm(startup$R.D.Spend,col='red',main = 'Q-Q Plot of R.D Spend',pch=20)
qqline(startup$R.D.Spend,col='blue')
qqnorm(startup$Administration,col='red',main = 'Q-Q Plot of Administration Spend',pch=20)
qqline(startup$Administration,col='blue')
qqnorm(startup$Marketing.Spend,main='Q-Q Plot of Marketing Spend',pch=20,col='red')
qqline(startup$Marketing.Spend,col='blue')
qqnorm(startup$Profit,col='Dodgerblue4',pch=20,main = "Q-Q Plot of Profit")
qqline(startup$Profit,col='red')
# Histogram
attach(startup)
hist(R.D.Spend)
hist(Administration)
hist(Marketing.Spend)
hist(Profit,col='Dodgerblue4')
library(moments)
skewness(startup[-4])
kurtosis(startup[-4])
# Bivariate Analysis
#scatter plot
plot(startup[-4])

library(GGally)
ggpairs(startup[-4])
ggcorr(startup[-4])
# Model Building
reg<-lm(Profit~R.D.Spend+Administration+Marketing.Spend,data=startup[-4])
summary(reg)
# As model shows that Administrative and Marketing spend is not statistically significant.
# Model Evaluation
regadmin<-lm(Profit~Administration)
summary(regadmin)

reg_marketing<-lm(Profit~Marketing.Spend)
summary(reg_marketing)

reg_marketing_admin<-lm(Profit~Administration + Marketing.Spend)
summary(reg_marketing_admin)

reg_marketing_RD<-lm(Profit~R.D.Spend+Marketing.Spend)
summary(reg_marketing_RD)
# Diagnostic Plots
library(car)
plot(reg)

qqPlot(reg,id.n=5)

influenceIndexPlot(reg,id.n=3)
influencePlot(reg,id.n=3)
avPlots(reg,id.n=2,cex=0.8,col='red')
reg1<-lm(Profit~Administration+Marketing.Spend+R.D.Spend,data = startup[-4][-50,])
summary(reg1)
plot(reg1$fitted.values,reg1$residuals)
plot(reg1)

reg_final<-lm(Profit~R.D.Spend+Marketing.Spend,data = startup[,-c(2,4)][-50,])
summary(reg_final)
# Evaluation model assumption
plot(reg_final$fitted.values,reg_final$residuals)
plot(reg_final)

qqnorm(reg_final$residuals)
qqline(reg_final$residuals)

# Subset selection technique
library(leaps)
reg_best<-regsubsets(Profit~R.D.Spend+Administration+Marketing.Spend,data = startup[-4])
summary(reg_best)
summary(reg_best)$adjr2
which.max(summary(reg_best)$adjr2)# It shows second model is best model.

# forward selection
reg_forw<-regsubsets(Profit~R.D.Spend+Administration+Marketing.Spend,data = startup[-4],nvmax = 10,method = 'forward')
summary(reg_forw)
summary(reg_forw)$adjr2
which.max(summary(reg_forw)$adjr2)# It shows second model is best model.

# backward selection 
reg_back<-regsubsets(Profit~R.D.Spend+Administration+Marketing.Spend,data = startup[-4],
                     nvmax = 10,method = 'backward')
summary(reg_back)
summary(reg_back)$adjr2
which.max(summary(reg_back)$adjr2)# It shows second model is best model.
startupnew<-startup[-4]
# Data Partition
set.seed(0)
n<-nrow(startup)
n1<-n*0.6
n2<-n-n1
train_ind<-sample(1:n,n1)
train<-startup[train_ind,]
test<-startup[-train_ind,]
# Train model
train_model<-lm(Profit~R.D.Spend+Marketing.Spend,data = train[,-c(2,4)][-50,])
summary(train_model)
pred<-data.frame(predict(train_model,interval = 'predict'))
pred
rms_train<-sqrt(mean(train_model$residuals**2))
rms_train

rms_test<-sqrt(mean((pred$fit)-(test$Profit))**2)
rms_test

New_data<-cbind(train$Profit,pred)
New_data
