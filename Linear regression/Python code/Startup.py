# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
startup= pd.read_csv("F:\\My Assignment\\Multiple linear regression\\Assignment\\50_Startups.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)
startup=startup.drop(["State"],axis=1)
startup.describe()
# Third moment decision
import scipy.stats as stats
stats.skew(startup)
# Fourth moment decision
stats.kurtosis(startup)
# analysing null values
startup.isnull().sum()
import seaborn as sns
sns.heatmap(startup.isnull())
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
# Jointplot
import seaborn as sns
sns.jointplot(x=startup['R&D Spend'], y=startup['Profit'])
sns.jointplot(x=startup['Administration'], y=startup['Profit'])
sns.jointplot(x=startup['Marketing Spend'], y=startup['Profit'])

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(startup.iloc[:, :])
                             
# Correlation matrix 
startup.corr()
startup.columns=['RD_Spend','Administration','Marketing_Spend','Profit']
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Profit ~ RD_Spend + Administration + Marketing_Spend', data = startup).fit() # regression model

# Summary
ml1.summary()


# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals

startup1 = startup.drop(startup.index[[49]])

# Preparing model                  
ml2 = smf.ols('Profit ~ RD_Spend + Administration + Marketing_Spend', data = startup1).fit() # regression model

# Summary
ml2.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_ad = smf.ols('Administration ~ RD_Spend  + Marketing_Spend', data = startup).fit().rsquared # regression model

vif_ad = 1/(1 - rsq_ad) 

rsq_rd = smf.ols('RD_Spend~ Administration  + Marketing_Spend', data = startup).fit().rsquared # regression model
vif_rd=1/(1-rsq_rd)

rsq_ms = smf.ols('Marketing_Spend ~ RD_Spend + Administration', data = startup).fit().rsquared # regression model
vif_ms=1/(1-rsq_ms)



# Storing vif values in a data frame
d1 = {'Variables':['Marketing_Spend', 'RD_Spend', 'Administration'], 'VIF':[vif_ms, vif_rd, vif_ad]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As hd is having highest VIF value, but less than 10.
startup2=startup1.drop(startup.index[48])
# Final model
ml3 = smf.ols('Profit ~ RD_Spend + Administration + Marketing_Spend', data = startup2).fit() # regression model


# Prediction
pred = ml3.predict(startup)

# Q-Q plot
res = ml3.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
from scipy import stats
import pylab
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = startup.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(ml3)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
startup_train, startup_test = train_test_split(startup2, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train= smf.ols('Profit ~ RD_Spend + Administration + Marketing_Spend', data = startup2).fit() # regression model

# prediction on test data set 
test_pred = model_train.predict(startup_test)

# test residual values 
test_resid = test_pred - startup_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(startup_train)

# train residual values 
train_resid  = train_pred - startup_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
