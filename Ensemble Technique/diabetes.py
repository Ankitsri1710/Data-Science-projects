import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("C:\\Users\\sriva\\Desktop\\Ensemble Techniques\\assignment\\Diabetes_RF.csv")

# Data pre-processing
df.head()
df.info()
lb=LabelEncoder()
df.iloc[:,8]=lb.fit_transform(df.iloc[:,8])
# Input and Output Split
predictors = df.loc[:, df.columns!="Class variable"]
type(predictors)

target = df.iloc[:,8]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)


from sklearn import tree
clftree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier


bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 500,
                            bootstrap = True, n_jobs = 1, random_state = 42)

bag_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, bag_clf.predict(x_test))
accuracy_score(y_test, bag_clf.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, bag_clf.predict(x_train))
accuracy_score(y_train, bag_clf.predict(x_train))

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

ada_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test))

# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(x_train))

######### Gradient boosting ###############
from sklearn.ensemble import GradientBoostingClassifier

grad_clf = GradientBoostingClassifier()

grad_clf.fit(x_train, y_train)

confusion_matrix(y_test, grad_clf.predict(x_test))
accuracy_score(y_test, grad_clf.predict(x_test))


# Hyperparameters
grad_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 500, max_depth = 1)
grad_clf2.fit(x_train, y_train)

# Evaluation on Testing Data
confusion_matrix(y_test, grad_clf2.predict(x_test))
accuracy_score(y_test, grad_clf2.predict(x_test))

# Evaluation on Training Data
accuracy_score(y_train, grad_clf2.predict(x_train))
####### Extreme gradient boosting ###########
import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 500, learning_rate = 0.3, n_jobs = -1)

xgb_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, xgb_clf.predict(x_test))
accuracy_score(y_test, xgb_clf.predict(x_test))

xgb.plot_importance(xgb_clf)

xgb_clf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)

param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0,9],
               'rag_alpha': [1e-2, 0.1, 1]}