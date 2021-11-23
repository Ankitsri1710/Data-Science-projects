import pandas as pd
import numpy as np

companydata=pd.read_csv("C:\\Users\\sriva\\Desktop\\Decision tree\\assignment\\Company_Data.csv")
companydata.isnull().sum()
companydata.columns
# Data Preprocessing
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
companydata["ShelveLoc"]=lb.fit_transform(companydata["ShelveLoc"])
companydata["Urban"]=lb.fit_transform(companydata["Urban"])
companydata["US"]=lb.fit_transform(companydata["US"])

Formula=(lambda x:"Low" if x<=10 else "High")
companydata['Sales_Response']=companydata['Sales'].apply(Formula)

colnames = list(companydata.columns)

predictors = colnames[:11]
target = colnames[11]
#Splitting data
from sklearn.model_selection import train_test_split
train, test = train_test_split(companydata, test_size = 0.3)

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
PREDICTORS= companydata.iloc[:,0:11]
type(PREDICTORS)

TARGET = companydata.iloc[:,11:]
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
