# Voting for Classification
# load the dataset

cc <- read.csv(file.choose())

set.seed(12345)
Train_Test <- sample(c("Train", "Test"), nrow(cc), replace = TRUE, prob = c(0.7, 0.3))
cc_Train <- cc[Train_Test == "Train",]
cc_TestX <- within(cc[Train_Test == "Test", ], rm(good_bad))
cc_TestY <- cc[Train_Test == "Test", "good_bad"]

library(randomForest)
# RANDOM FOREST ANALYSIS
cc_RF <- randomForest(good_bad ~ ., data = cc_Train, keep.inbag = TRUE, ntree = 500)

# New data voting
# Overall class prediction
cc_RF_Test_Margin <- predict(cc_RF, newdata = cc_TestX, type = "class")

# Prediction 
cc_RF_Test_Predict <- predict(cc_RF, newdata = cc_TestX, type = "class", predict.all = TRUE)

sum(cc_RF_Test_Margin == cc_RF_Test_Predict$aggregate)

head(cc_RF_Test_Margin == cc_RF_Test_Predict$aggregate)

# Majority Voting
dim(cc_RF_Test_Predict$individual)

# View(cc_RF_Test_Predict$individual) # Prediction at each tree
Row_Count_Max <- function(x) names(which.max(table(x)))

Voting_Predict <- apply(cc_RF_Test_Predict$individual, 1, Row_Count_Max)

head(Voting_Predict)
tail(Voting_Predict)

all(Voting_Predict == cc_RF_Test_Predict$aggregate)
all(Voting_Predict == cc_RF_Test_Margin)

mean(Voting_Predict == cc_TestY)
