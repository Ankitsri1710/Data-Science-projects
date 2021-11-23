# Load the Data
# Note: XGboosting data should be convereted into Dmatrix 

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
attach(movies_train)

#install.packages("xgboost")
library(xgboost)

train_y <- movies_train$Start_Tech_Oscar == "1"

str(movies_train)

# create dummy variables on attributes
train_x <- model.matrix(movies_train$Start_Tech_Oscar ~ . -1, data = movies_train)

train_x <- train_x[, -12]
# 'n-1' dummy variables are required, hence deleting the additional variables
     
test_y <- movies_test$Start_Tech_Oscar == "1"

# create dummy variables on attributes
test_x <- model.matrix(movies_test$Start_Tech_Oscar ~ .-1, data = movies_test)
test_x <- test_x[, -12]

# DMatrix on train
Xmatrix_train <- xgb.DMatrix(data = train_x, label = train_y)
# DMatrix on test 
Xmatrix_test <- xgb.DMatrix(data = test_x, label = test_y)


# Max number of boosting iterations - nround
xg_boosting <- xgboost(data = Xmatrix_train, nround = 50,
                       objective = "multi:softmax", eta = 0.3, 
                       num_class = 2, max_depth = 100)

# Prediction for test data
xgbpred_test <- predict(xg_boosting, Xmatrix_test)
table(test_y, xgbpred_test)
mean(test_y == xgbpred_test)

# Prediction for train data
xgbpred_train <- predict(xg_boosting, Xmatrix_train)
table(train_y, xgbpred_train)
mean(train_y == xgbpred_train)
