# Loading dataset
toyota_corolla<-read.csv(file.choose(),header = T)
View(toyota_corolla)
mydata<-toyota_corolla[,c(3,4,7,9,13,14,16,17,18)]
DATA<-mydata[-c(81,222,961,602),-6]
View(mydata)
attach(DATA)
library(glmnet)
x<-model.matrix(Price~.,data = DATA)[,-1]
y<-Price

grid<-10^seq(10,-4,length=100)
grid
 # Lasso model
lasso<-glmnet(x,y,alpha=1,lambda=grid)
cv<-cv.glmnet(x,y,alpha=1,lambda = grid)
cv
plot(cv)
optimumlambda<-cv$lambda.min
optimumlambda
lasso_predict<-predict(lasso,s=optimumlambda,newx = x)
lasso_predict
y_pred<-lasso_predict
sse<-sum((y_pred-y)**2)
sst<-sum((y-mean(y))**2)
R_squared<-(1-(sse/sst))
R_squared
predict(lasso,s=optimumlambda,type = 'coefficient',newx = x)
lasso_result<-cbind(DATA,lasso_predict)
lasso_result
# Ridge model
ridge<-glmnet(x,y,alpha = 0,lambda = grid)
cv_R<-cv.glmnet(x,y,alpha=0,lambda = grid)
cv_R
plot(cv_R)
optimumlambda_R<-cv_R$lambda.min
optimumlambda_R
Ridge_predict<-predict(ridge,s=optimumlambda_R,newx = x)
Ridge_predict
y_predR<-Ridge_predict
sseR<-sum((y_predR-y)**2)
R_squaredR<-(1-(sseR/sst))
R_squaredR
predict(ridge,s=optimumlambda,type = 'coefficient',newx = x)
Ridge_result<-cbind(DATA,Ridge_predict)
Ridge_result
