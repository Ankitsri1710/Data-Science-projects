# Importing libraries
import pandas as pd
import numpy as np

zoo=pd.read_csv("C:\\Users\\sriva\\Desktop\\K nearest neighbour\\assignment\\zoo.csv")
zoo.describe()

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

zoo_n=norm_func(zoo.iloc[:,1:17])
zoo_n.describe()

x=np.array(zoo_n)
y=np.array(zoo['type'])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=15)
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

for i in range(3,55,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(x_train, y_train)
    train_acc = np.mean(neigh.predict(x_train) == y_train)
    test_acc = np.mean(neigh.predict(x_test) == y_test)
    acc.append([train_acc, test_acc])

import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,55,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,55,2),[i[1] for i in acc],"bo-")

