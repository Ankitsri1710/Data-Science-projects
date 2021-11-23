import pandas as pd
import numpy as np

Fraud_check=pd.read_csv("C:\\Users\\sriva\\Desktop\\Decision tree\\assignment\\Fraud_check.csv")
Fraud_check.isnull().sum()
Fraud_check.columns
# Data Preprocessing
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
Fraud_check["Undergrad"]=lb.fit_transform(Fraud_check["Undergrad"])
Fraud_check["Marital.Status"]=lb.fit_transform(Fraud_check["Marital.Status"])
Fraud_check["Urban"]=lb.fit_transform(Fraud_check["Urban"])

Formula=(lambda x:"Risky" if x<=30000 else "Non-Risky")
Fraud_check['Sales']=Fraud_check['Taxable.Income'].apply(Formula)

colnames = list(Fraud_check.columns)

predictors = colnames[:6]
target = colnames[6]
#Splitting data
from sklearn.model_selection import train_test_split
train, test = train_test_split(Fraud_check, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])

# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

##### Random Forest ##########
# Train Test partition of the data
PREDICTORS= Fraud_check.iloc[:,0:6]
type(PREDICTORS)

TARGET = Fraud_check.iloc[:,6:]
type(TARGET)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(PREDICTORS,TARGET, test_size = 0.3, random_state=0)


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

rf_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, rf_clf.predict(x_test))
accuracy_score(y_test, rf_clf.predict(x_test))
accuracy_score(y_train,rf_clf.predict(x_train))
