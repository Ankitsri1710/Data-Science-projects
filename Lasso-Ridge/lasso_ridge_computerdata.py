# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
computerdata = pd.read_csv("C:\\Users\\sriva\\Desktop\\Multiple linear regression\\Assignment\\Computer_Data.csv")
computerdata=computerdata.iloc[:,1:]
computerdata.describe()
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
computerdata['cd']=lb.fit_transform(computerdata['cd'])
computerdata['multi']=lb.fit_transform(computerdata['multi'])
computerdata['premium']=lb.fit_transform(computerdata['premium'])
# Jointplot
sns.jointplot(x=computerdata['speed'], y=computerdata['price'])
sns.jointplot(x=computerdata['ram'], y=computerdata['price'])
sns.jointplot(x=computerdata['hd'], y=computerdata['price'])
sns.jointplot(x=computerdata['screen'], y=computerdata['price'])

#Pair plot
sns.pairplot(computerdata.iloc[:, :])
# Correlation matrix 
computerdata.corr()

# Model building
ml1 = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend ', data = computerdata).fit() # regression model

# Summary
ml1.summary()
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(computerdata.iloc[:, 1:], computerdata.price)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(computerdata.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(computerdata.iloc[:, 1:])

# Adjusted r-square
lasso.score(computerdata.iloc[:, 1:], computerdata.price)

# RMSE
np.sqrt(np.mean((pred_lasso - computerdata.price)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(computerdata.iloc[:, 1:], computerdata.price)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(computerdata.columns[1:]))

rm.alpha

pred_rm = rm.predict(computerdata.iloc[:, 1:])

# Adjusted r-square
rm.score(computerdata.iloc[:, 1:], computerdata.price)

# RMSE
np.sqrt(np.mean((pred_rm - computerdata.price)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(computerdata.iloc[:, 1:], computerdata.price) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(computerdata.columns[1:]))

enet.alpha

pred_enet = enet.predict(computerdata.iloc[:, 1:])

# Adjusted r-square
enet.score(computerdata.iloc[:, 1:], computerdata.price)

# RMSE
np.sqrt(np.mean((pred_enet - computerdata.price)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-20, 1e-15, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 40]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 10)
lasso_reg.fit(computerdata.iloc[:, 1:], computerdata.price)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(computerdata.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(computerdata.iloc[:, 1:], computerdata.price)

# RMSE
np.sqrt(np.mean((lasso_pred - computerdata.price)**2))


# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-20, 1e-15, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 15]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 10)
ridge_reg.fit(computerdata.iloc[:, 1:], computerdata.price)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(computerdata.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(computerdata.iloc[:, 1:], computerdata.price)

# RMSE
np.sqrt(np.mean((ridge_pred - computerdata.price)**2))

# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 25]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 10)
enet_reg.fit(computerdata.iloc[:, 1:], computerdata.price)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(computerdata.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(computerdata.iloc[:, 1:], computerdata.price)

# RMSE
np.sqrt(np.mean((enet_pred - computerdata.price)**2))
