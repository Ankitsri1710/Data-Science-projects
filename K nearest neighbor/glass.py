import pandas as pd
import numpy as np

glass=pd.read_csv("C:\\Users\\sriva\\Desktop\\K nearest neighbour\\assignment\\glass.csv")

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)
glass_n=norm_func(glass.iloc[:,0:9])
glass_n.describe()

x=np.array(glass_n)
y=np.array(glass['Type'])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
pred

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))
pd.crosstab(y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(2,60,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(x_train, y_train)
    train_acc = np.mean(neigh.predict(x_train) == y_train)
    test_acc = np.mean(neigh.predict(x_test) == y_test)
    acc.append([train_acc, test_acc])

import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(2,60,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(2,60,2),[i[1] for i in acc],"bo-")
