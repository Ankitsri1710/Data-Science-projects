# Load the Data
# movies.csv
movies = read.csv(file.choose())

##Exploring and preparing the data ----
str(movies)

library(caTools)
set.seed(0)
split <- sample.split(movies$Collection, SplitRatio = 0.8)
movies_train <- subset(movies, split == TRUE)
movies_test <- subset(movies, split == FALSE)


# install.packages("randomForest")
library(randomForest)

bagging = randomForest(movies_train$Collection ~ ., data = movies_train, mtry = 17)
# bagging will take all the columns ---> mtry = all the attributes

test_pred = predict(bagging, movies_test)

rmse_bagging <- sqrt(mean(movies_test$Collection - test_pred)^2)
rmse_bagging

# Prediction for trained data result
train_pred <- predict(bagging, movies_train)

# RMSE on Train Data
train_rmse <- sqrt(mean(movies_train$Collection - train_pred)^2)
train_rmse
