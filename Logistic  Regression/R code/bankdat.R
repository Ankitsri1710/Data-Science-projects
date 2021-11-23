# Loading dataset
bank_data<-read.csv(file.choose(),header=T)
View(bank_data)
#Exploratory data analysis
summary(bank_data)
sum(is.na(bank_data))
attach(bank_data)
prop.table(table(y))*100

#Visualization
bank_data$y<-as.factor(bank_data$y)
library(ggplot2)
v1<-ggplot(bank_data,aes(x=default,fill=y,color=y))+geom_histogram(bindwidth=1)+
  labs(title = 'Proportion of defaulted customers subscribed term deposit')
v1+theme_bw()
v2<-ggplot(bank_data,aes(x=duration,fill=y,color=y))+geom_histogram(bindwidth=1)+
  labs(title = 'Distribution of duration by term deposit')
v2+theme_bw()
v3<-ggplot(bank_data,aes(x=age,fill=y,color=y))+geom_histogram(bindwidth=1)+
  labs(title = 'Distribution of age by term deposit')
v3+theme_bw()
v4<-ggplot(bank_data,aes(x=balance,fill=y,color=y))+geom_histogram(bindwidth=1)+
  labs(title = "Distribtion of balance by term deposit")
v4+theme_bw()
# Model building
bank_data$y<-as.numeric(bank_data$y)
bank_data$y<-ifelse(bank_data$y==1,0,1)
model<-lm(y~.,data = bank_data)
summary(model)

pred1<-predict(model,bank_data)
pred1

model1<-glm(y~.,data = bank_data,family  = 'binomial')
summary(model1)

exp(coef(model1))
PROB<-predict(model1,bank_data,type = "response")
PROB

CONFUSION<-table(PROB>0.5,bank_data$y)
CONFUSION

acc<-sum(diag(CONFUSION)/sum(CONFUSION))
acc

library(InformationValue)
optcutoff<-optimalCutoff(bank_data$y,PROB)
optcutoff

misClassError(bank_data$y,PROB,threshold = optcutoff)

plotROC(bank_data$y,PROB)

#Confusion matrix
predvalues<-ifelse(PROB > optcutoff, 1,0)

result<-table(predvalues,bank_data$y)
result
Acc<-sum(diag(result)/sum(result))
Acc

sensitivity(predvalues,bank_data$y)

###################
# Data Partitioning
n <- nrow(bank_data)
n1 <- n * 0.85
n2 <- n - n1
train_index <- sample(1:n, n1)
train <- bank_data[train_index, ]
test <- bank_data[-train_index, ]

# Train the model using Training data
finalmodel <- glm(y ~ ., data = train, family = "binomial")
summary(finalmodel)

# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test

# Confusion matrix 
confusion <- table(prob_test > optcutoff, test$y)
confusion

# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
pred_values <- ifelse(prob_test > optcutoff, 1, 0)

# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"pred_values"] <- pred_values

table(test$y, test$pred_values)


# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train

# Confusion matrix
confusion_train <- table(prob_train > optcutoff, train$y)
confusion_train

# Model Accuracy 
Acc_train <- sum(diag(confusion_train)/sum(confusion_train))
Acc_train

