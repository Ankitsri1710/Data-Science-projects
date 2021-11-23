
import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
bankdata = pd.read_csv("C:/Users/sriva/Desktop/logistic Regression/Assignment/bank_data.csv", sep = ",")
bankdata.describe()
bankdata.isna().sum()
bankdata.columns
# Model building 
import statsmodels.formula.api as sm
logit_model = sm.logit('y~ age+default+balance+housing+loan+duration+campaign+pdays', data = bankdata).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(bankdata.iloc[ :, 0: 32])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(bankdata.y, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
bankdata["pred"] = np.zeros(45211)
# taking threshold value and above the prob value will be treated as correct value 
bankdata.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(bankdata["pred"], bankdata["y"])
classification


### Splitting the data into train and test data 
mydata=bankdata.iloc[:,0:32]
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(bankdata, test_size = 0.3) # 30% test data

# Model building 

model = sm.logit('y~ age+default+balance+housing+loan+duration+campaign+pdays', data = train_data).fit()
#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(13564)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['y'])
confusion_matrix

accuracy_test = (9293+1273)/(9298+324+2674+1273) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["y"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["y"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 0:32 ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(31647)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['y'])
confusion_matrx

accuracy_train = (21618 + 2956)/(21618+736+6337+2956)
print(accuracy_train)
