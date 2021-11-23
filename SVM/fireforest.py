# Importing libraries
import pandas as pd
import numpy as np
# Importing dataset
forestfire=pd.read_csv("C:\\Users\\sriva\\Desktop\\SVM\\Assignment\\forestfires.csv")
forestfire=forestfire.iloc[:,2:31]
forestfire.describe()
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# Splitting data
train,test=train_test_split(forestfire, test_size=0.20)
train_x=train.iloc[:,0:28]
train_y=train.iloc[:,28]
test_x=test.iloc[:,0:28]
test_y=test.iloc[:,28]
# model building
# kernel="linear"
model_linear=SVC(kernel="linear")
model_linear.fit(train_x,train_y)
pred_train_lin=model_linear.predict(train_x)
pred_test_lin=(model_linear.predict(test_x))
np.mean(pred_test_lin == test_y)
np.mean(pred_train_lin==train_y)

# kernel="rbf"
model_rbf=SVC(kernel="rbf")
model_rbf.fit(train_x,train_y)
pred_train_rbf=model_rbf.predict(train_x)
pred_test_rbf=model_rbf.predict(test_x)
np.mean(pred_test_rbf==test_y)
np.mean(pred_train_rbf==train_y)
