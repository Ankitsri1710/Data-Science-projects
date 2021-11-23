import pandas as pd
import numpy as np

salarydata_Train=pd.read_csv("C:\\Users\\sriva\\Desktop\\SVM\\Assignment\\SalaryData_Train.csv")
salarydata_Test=pd.read_csv("C:\\Users\\sriva\\Desktop\\SVM\\Assignment\\SalaryData_Test.csv")
train_x=salarydata_Train.iloc[:,[0,3,9,10,11]]
train_y=salarydata_Train.Salary

test_x=salarydata_Test.iloc[:,[0,3,9,10,11]]
test_y=salarydata_Test.Salary

from sklearn.svm import SVC
# kernel="linear"
model_lin=SVC(kernel="linear")
model_lin.fit(train_x, train_y)
pred_train_lin=model_lin.predict(train_x)
pred_test_lin=model_lin.predict(test_x)
np.mean(pred_train_lin==train_y)
np.mean(pred_test_lin==test_y)
# kernel="rbf"
model_rbf=SVC(kernel="rbf")
model_rbf.fit(train_x,train_y)
pred_train_rbf=model_rbf.predict(train_x)
pred_test_rbf=model_rbf.predict(test_x)
np.mean(pred_train_rbf==train_y)
np.mean(pred_test_rbf==test_y)
