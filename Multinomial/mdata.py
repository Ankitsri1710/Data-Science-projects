# Imporing libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Importing datset
mdata=pd.read_csv("C:\\Users\\sriva\\Desktop\\Multinomial\\Assignment\\mdata.csv")
mdata.head(15)
mdata=mdata.iloc[:,2:]
mdata.describe()

mdata.prog.value_counts()
# Visualization
# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "prog", y = "read", data = mdata)
sns.boxplot(x = "prog", y = "write", data = mdata)
sns.boxplot(x = "prog", y = "math", data = mdata)
sns.boxplot(x = "prog", y = "science", data = mdata)

# Scatter plot for each categorical choice of car
sns.stripplot(x = "prog", y = "read", jitter = True, data = mdata)
sns.stripplot(x = "prog", y = "write", jitter = True, data = mdata)
sns.stripplot(x = "prog", y = "math", jitter = True, data = mdata)
sns.stripplot(x = "prog", y = "science", jitter = True, data = mdata)
# Splitting of data
mydata=mdata.iloc[:,3:8]
train,test=train_test_split(mydata,test_size=0.2)
multinom_model=LogisticRegression(multi_class="multinomial",solver="newton-cg").fit(train.iloc[:,1:],train.iloc[:,0])
print(multinom_model)

test_predict = multinom_model.predict(test.iloc[:, 1:]) # Test predictions
test_predict
pd.crosstab(test_predict, test.prog)
# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)

train_predict =multinom_model.predict(train.iloc[:, 1:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict) 
pd.crosstab(train_predict, train.prog)
