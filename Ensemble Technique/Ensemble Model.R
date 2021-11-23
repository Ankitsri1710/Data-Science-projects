###############Start of code##########
data("iris")

View(iris)

#Accuracy with single model
library(caret)
inTraininglocal <- createDataPartition(iris$Species, p = 0.75, list = F)
training <- iris[inTraininglocal, ]
testing <- iris[-inTraininglocal, ]

#install.packages("C50")
library(C50)
model <- C5.0(training$Species ~ ., data = training[, -5])
plot(model)
pred <- predict.C5.0(model, testing[, -5])
a <- table(testing$Species, pred)

sum(diag(a))/sum(a)

#*****************************************************************
 ########Bagging
acc <- c()
for(i in 1:11)
{
  inTraininglocal <- createDataPartition(iris$Species, p = 0.75, list = F)
  training1 <- iris[inTraininglocal, ]
  testing <-iris[-inTraininglocal, ]
  fittree <- C5.0(training1$Species ~ ., data = training1[, -5])
  pred <- predict.C5.0(fittree,testing[ , -5])
  a <- table(testing$Species, pred)
  acc <- c(acc,sum(diag(a))/sum(a))
}
acc
mean(acc)

#**************************************************************
############## Boosting

# Accuracy with single model with Boosting

inTraininglocal <- createDataPartition(iris$Species, p = 0.75, list = F)
training <- iris[inTraininglocal, ]
testing <- iris[-inTraininglocal, ]

model <- C5.0(training$Species ~ ., data = training[, -5], trials = 10)
pred <- predict.C5.0(model, testing[, -5])
a <- table(testing$Species, pred)

sum(diag(a))/sum(a)

#***************************************************************
######### Bagging and Boosting
 acc <- c()
for(i in 1:11)
{
  
  inTraininglocal <- createDataPartition(iris$Species, p = 0.75, list = F)
  training1 <- iris[inTraininglocal, ]
  testing <- iris[-inTraininglocal, ]
  
  fittree <- C5.0(training1$Species ~ ., data = training1, trials = 10)
  pred <- predict.C5.0(fittree, testing[, -5])
  a <- table(testing$Species, pred)
  
  acc <- c(acc, sum(diag(a))/sum(a))
  
}

acc
mean(acc)
