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

rf = randomForest(movies_train$Collection ~ ., data = movies_train)
# Default mtry value will be equal p/3
# 17/3 = 5.66 = 6 (rounded)

test_rf_pred = predict(rf, movies_test)

rmse_rf <- sqrt(mean(movies_test$Collection - test_rf_pred)^2)
rmse_rf

# Prediction for trained data result
train_rf_pred <- predict(rf, movies_train)

# RMSE on Train Data
train_rmse_rf <- sqrt(mean(movies_train$Collection - train_rf_pred)^2)
train_rmse_rf
