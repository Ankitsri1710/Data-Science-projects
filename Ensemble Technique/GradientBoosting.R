# Load the Data
# movies.csv
movies = read.csv(file.choose())
movies<-movies[,-c(12,15)]
##Exploring and preparing the data ----
str(movies)

library(caTools)
set.seed(0)
split <- sample.split(movies$Collection, SplitRatio = 0.8)
movies_train <- subset(movies, split == TRUE)
movies_test <- subset(movies, split == FALSE)
movies$X3D_available<-as.factor(movies$X3D_available)
movies$Genre<-as.factor(movies$Genre)

#install.packages("gbm")
library(gbm)

boosting = gbm(movies_train$Collection ~ ., data = movies_train, distribution = 'gaussian',
               n.trees = 5000, interaction.depth = 4, shrinkage = 0.2, verbose = F)
# distribution = Gaussian for regression and Bernouli for classification

# Prediction for test data result
boost_test = predict(boosting, movies_test, n.trees = 5000)

rmse_boosting <- sqrt(mean(movies_test$Collection - boost_test)^2)
rmse_boosting

# Prediction for train data result
boost_train <- predict(boosting, movies_train, n.trees = 5000)

# RMSE on Train Data
rmse_train <- sqrt(mean(movies_train$Collection - boost_train)^2)
rmse_train
