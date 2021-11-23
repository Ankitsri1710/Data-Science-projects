
import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier


data = pd.read_csv("C:\\Datasets_BA\\360DigiTMG\\DS_India\\360DigiTMG DS India Module wise PPTs\\Module 20 AdaBoost_Extreme Gradient Boosting\\Data\\iris.csv") #importing the dataset 
data.head() #seeting the head of the data

df_x = data.iloc[:,:4] #dividing the i/p and o/p variebale
df_y = data.iloc[:,4]

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.25, random_state=4) #dividing the data randomly
y_test.head()

#decision tree
dt = DecisionTreeClassifier() #storing the classifer in dt

dt.fit(x_train, y_train) #fitting te model 

dt.score(x_test, y_test) #checking the score like accuracy

dt.score(x_train, y_train)
#so our model is overfitting 

#Random Forest clssifer: it is a ensemble of Decision tree 
rf = RandomForestClassifier(n_estimators = 10) # n_estimator number of tree in the forest 
rf.fit(x_train,y_train) #fitting the random forest model 

rf.score(x_test, y_test) #doing the accuracy of the test model 

rf.score(x_train, y_train) #doing the accuracy of the train model 

#Bagging - Gradient 
bg = BaggingClassifier(DecisionTreeClassifier(), max_samples = 0.5, max_features = 1.0, n_estimators = 20)
bg.fit(x_train,y_train) #fitting the model 

bg.score(x_test, y_test) #test accuracy

bg.score(x_train, y_train) #train accuracy 


# Ada boosting 
ada = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=10, learning_rate=1)
ada.fit(x_train,y_train)

ada.score(x_test,y_test)

ada.score(x_train,y_train)

# Voting Classifier 
from sklearn.linear_model import LogisticRegression # importing logistc regression
from sklearn.svm import SVC # importing Svm 

lr = LogisticRegression() 
dt = DecisionTreeClassifier()
svm = SVC(kernel= 'poly', degree=2)

evc = VotingClassifier(estimators=[('lr', lr),('dt', dt),('svm', svm)], voting='hard')

evc.fit(x_train, y_train)

evc.score(x_test, y_test)

evc.score(x_train, y_train)
 