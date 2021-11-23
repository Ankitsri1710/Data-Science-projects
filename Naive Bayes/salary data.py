import pandas as pd
import numpy as np


salarydata_Train=pd.read_csv("C:\\Users\\sriva\\Desktop\\naive bayes\\assignment\\SalaryData_Train.csv")
salarydata_Test=pd.read_csv("C:\\Users\\sriva\\Desktop\\naive bayes\\assignment\\SalaryData_Test.csv")
train_x=salarydata_Train.iloc[:,[0,3,9,10,11]]
train_y=salarydata_Train.Salary

test_x=salarydata_Test.iloc[:,[0,3,9,10,11]]
test_y=salarydata_Test.Salary

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb = MB()
classifier_mb.fit(train_x, train_y)

test_pred_nb=classifier_mb.predict(test_x)
np.mean(test_pred_nb==test_y)

train_pred_nb=classifier_mb.predict(train_x)
np.mean(train_pred_nb==train_y)

pd.crosstab(test_y, test_pred_nb, rownames = ['Actual'], colnames= ['Predictions'])
pd.crosstab(train_y, train_pred_nb, rownames = ['Actual'], colnames= ['Predictions'])
