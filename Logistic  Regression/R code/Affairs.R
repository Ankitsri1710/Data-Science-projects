#Loading dataset
library(AER)
data(Affairs,package = "AER")
# Data Preprocessing
# Exploratory data analysis
summary(Affairs)
library(Hmisc)
describe(Affairs)
sum(is.na(Affairs))
#Visualization
#Univariate analysis
attach(Affairs)
par(mfrow=c(4,2))
hist(occupation,col=rainbow(7))
boxplot(occupation,col = rainbow(7),horizontal = T,main='Boxplot of occupation')
hist(age,col=rainbow(9))
boxplot(age,col = rainbow(9),horizontal = T,main='Boxplot of age')
hist(yearsmarried,col=rainbow(8))
boxplot(yearsmarried,col = rainbow(8),horizontal = T,main='Boxplot of Years Married')
hist(religiousness,col=rainbow(5))
boxplot(religiousness,col = rainbow(5),horizontal = T,main='Boxplot of Religiousness')
par(mfrow=c(2,2))
hist(affairs,col=rainbow(6),main='Histogram of number of Affairs')
boxplot(affairs,col = rainbow(6),horizontal = T,main='Boxplot of number of Affairs')
hist(education,col=rainbow(7))
boxplot(education,col = rainbow(7),horizontal = T,main='Boxplot of Education')
attach(Affairs)
# Bivariate analysis
Affairs$affairs<-as.factor(Affairs$affairs)
library(ggplot2)
v<-ggplot(Affairs,aes(x=age,fill=affairs,color=affairs))+geom_histogram(binwidth = 1)+
  labs(title = "Disrtibution of age by Affairs")
v+theme_bw()
v1<-ggplot(Affairs,aes(x=yearsmarried,fill=affairs,color=affairs))+geom_histogram(binwidth=1)+
  labs(title = 'Distribution of yearsmarried by affairs')
v1+theme()
v2<-ggplot(Affairs,aes(x=education,fill=affairs,color=affairs))+geom_histogram(binwidth=1)+
  labs(title = 'Distribution of education by affairs')
v2+theme_bw()
v3<-ggplot(Affairs,aes(x=religiousness,fill=affairs,color=affairs))+geom_histogram(binwidth=1)+
  labs(title = 'Distribution of Religiousness by affairs')
v3+theme_bw()
Affairs$gender<-ifelse(Affairs$gender== 'female' ,0,1)
Affairs$children<-ifelse(Affairs$children=='no',0,1)
Affairs$affairs<-ifelse(Affairs$affairs=='0',0,1)

Affairs$affairs<-as.numeric(Affairs$affairs)
model<-lm(affairs~.,data = Affairs)
summary(model)
pred1<-predict(model,Affairs)
pred1

log_model<-glm(affairs~.,data = Affairs,family = "binomial")
summary(log_model)

exp(coef(log_model))

prob<-plogis(predict(log_model,Affairs))

Confusion_matrix<-table(prob>0.5,Affairs$affairs)
Confusion_matrix

# Model accuracy
acc<-sum(diag(Confusion_matrix)/sum(Confusion_matrix))
acc
pred_val<-ifelse(prob > 0.5 ,1,0)

library(caret)
confusionMatrix(factor(Affairs$affairs,levels =c(0,1)),factor(pred_val,levels = c(0,1)))

library(InformationValue)
optimumcutoff<-optimalCutoff(Affairs$affairs,prob)
optimumcutoff

library(car)
vif(log_model)

misClassError(Affairs$affairs,prob,threshold = optimumcutoff)

plotROC(Affairs$affairs,prob)

# Confusion matrix
pred_values<-ifelse(prob > optimumcutoff, 1,0)
pred_values

result<-table(pred_values,Affairs$affairs)
result

sensitivity(pred_values,Affairs$affairs)

confusionMatrix(actuals = Affairs$affairs, predictedScores = pred_values)
ACC<-sum(diag(result)/sum(result))
ACC

###################
# Data Partitioning
n <- nrow(Affairs)
n1 <- n * 0.85
n2 <- n - n1
train_index <- sample(1:n, n1)
train <- Affairs[train_index, ]
test <- Affairs[-train_index, ]

# Train the model using Training data
finalmodel <- glm(affairs ~ ., data = train, family = "binomial")
summary(finalmodel)

# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test

# Confusion matrix 
confusion <- table(prob_test > optimumcutoff, test$affairs)
confusion

# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 

# Creating empty vectors to store predicted classes based on threshold value
predvalues <- NULL
predvalues <- ifelse(prob_test > optimumcutoff, 1, 0)

# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"predvalues"] <- predvalues

test_table<-table(test$affairs, test$predvalues)
acc_test<-sum(diag(test_table)/sum(test_table))
acc_test
# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train

# Confusion matrix
confusion_train <- table(prob_train > optimumcutoff, train$affairs)
confusion_train

# Model Accuracy 
Acc_train <- sum(diag(confusion_train)/sum(confusion_train))
Acc_train

