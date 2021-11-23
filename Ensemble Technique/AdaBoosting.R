# Load the Data
# Note: Adaboost can only be run for classification trees. 
# Regression trees cannot be run in R

# movies_classification.csv
movies = read.csv(file.choose())

##Exploring and preparing the data ----
str(movies)

library(caTools)
set.seed(0)
split <- sample.split(movies$Start_Tech_Oscar, SplitRatio = 0.8)
movies_train <- subset(movies, split == TRUE)
movies_test <- subset(movies, split == FALSE)

summary(movies_train)

#install.packages("adabag")
library(adabag)

movies_train$Start_Tech_Oscar <- as.factor(movies_train$Start_Tech_Oscar)

adaboost <- boosting(Start_Tech_Oscar ~ ., data = movies_train, boos = TRUE)

# Test data
adaboost_test = predict(adaboost, movies_test)

table(adaboost_test$class, movies_test$Start_Tech_Oscar)
mean(adaboost_test$class == movies_test$Start_Tech_Oscar)


# Train data
adaboost_train = predict(adaboost, movies_train)

table(adaboost_train$class, movies_train$Start_Tech_Oscar)
mean(adaboost_train$class == movies_train$Start_Tech_Oscar)

