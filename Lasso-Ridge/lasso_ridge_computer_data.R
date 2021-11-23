#loading dataset
computer_data<-read.csv(file.choose(),header = T)
computer_data<-computer_data[,-1]
attach(computer_data)
library(glmnet)
x<-model.matrix(price~.,data = computer_data)[,-1]
y<-price

grid<-10^seq(10,-4,length=100)
grid
# Ridge model
model_ridge<-glmnet(x,y,alpha=0,lambda = grid)
cv_fit_ridge<-cv.glmnet(x,y,alpha=0,lambda = grid)
cv_fit_ridge
plot(cv_fit_ridge)
optimumlambda_ridge<-cv_fit_ridge$lambda.min
optimumlambda_ridge
ridge_predict<-predict(model_ridge,s=optimumlambda_ridge,newx = x)
head(ridge_predict)
y_pred<-ridge_predict
y_act<-y
sse<-sum((y_act-y_pred)**2)
sst<-sum((y_act-mean(y_act))**2)
R_squared<-(1-(sse/sst))
R_squared
predict(model_ridge,s=optimumlambda_ridge,type = 'coefficient',newx = x)
ridge_result<-cbind(computer_data,ridge_predict)
ridge_result
# Lasso model
model_lasso<-glmnet(x,y,alpha=1,lambda = grid)
fit_lasso<-cv.glmnet(x,y,alpha=1,lambda = grid)
fit_lasso
plot(fit_lasso)
optimumlambda_lasso<-fit_lasso$lambda.min
optimumlambda_lasso
lasso_predict<-predict(model_lasso,s=optimumlambda_lasso,newx = x)
lasso_predict
yp<-lasso_predict
sse_lasso<-sum((yp-y)**2)
sst_lasso<-sum((y-mean(y))**2)
R_squared_lasso<-(1-(sse_lasso/sst_lasso))
R_squared_lasso
predict(model_lasso,s=optimumlambda_ridge,type = 'coefficient',newx = x)
lasso_result<-cbind(computer_data,lasso_predict)
lasso_result
