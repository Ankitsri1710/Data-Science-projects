# Loading dataset
startup<-read.csv(file.choose(),header = T)
startup<-startup[,c(5,1,2,3)]
attach(startup)
library(glmnet)

x<-model.matrix(Profit~.,data = startup)[,-1]
y<-Profit

grid<-10^seq(10,-2,length=100)
grid

#Ridge regression
model_ridge<-glmnet(x,y,alpha = 0,lambda = grid)
summary(model_ridge)
cv_fit<-cv.glmnet(x,y,alpha = 0,lambda = grid)
summary(cv_fit)
plot(cv_fit)

optimumlambda<-cv_fit$lambda.min
optimumlambda
model_predict<-predict(model_ridge,s=optimumlambda,newx = x)
ridge_predict<-model_predict
ridge_predict
sse<-sum((y-ridge_predict)^2)
sse
sst<-sum((y-mean(y))**2)
sst
R_squred<-(1-sse/sst)
R_squred
ridge_result<-cbind(startup,ridge_predict)
ridge_result
predict(model_ridge,s=optimumlambda,type = 'coefficient',newx = x)

# Lasso regression
model_lasso<-glmnet(x,y,alpha = 1,lambda = grid)
cv1_fit<-cv.glmnet(x,y,alpha = 1,lambda = grid)
cv1_fit
summary(cv1_fit)
plot(cv1_fit)

optimumlambda1<-cv1_fit$lambda.min
optimumlambda1
model_predict1<-predict(model_lasso,s=optimumlambda1,newx = x)
lasso_predicted<-model_predict1
sse<-sum((y-lasso_predicted)**2)
sst<-sum((y-mean(y))**2)
R_squred1<-(1-sse/sst)
R_squred1

predict(model_lasso,s=optimumlambda1,type = 'coefficient',newx = x)
lasso_Result<-cbind(startup,lasso_predicted)
lasso_Result
