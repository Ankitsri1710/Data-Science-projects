# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
computerdata = pd.read_csv("C:\\Users\\sriva\\Desktop\\Multiple linear regression\\Assignment\\Computer_Data.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)
computerdata=computerdata.iloc[:,1:]
computerdata.describe()
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
computerdata['cd']=lb.fit_transform(computerdata['cd'])
computerdata['multi']=lb.fit_transform(computerdata['multi'])
computerdata['premium']=lb.fit_transform(computerdata['premium'])
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
# Jointplot
import seaborn as sns
sns.jointplot(x=computerdata['speed'], y=computerdata['price'])
sns.jointplot(x=computerdata['ram'], y=computerdata['price'])
sns.jointplot(x=computerdata['hd'], y=computerdata['price'])
sns.jointplot(x=computerdata['screen'], y=computerdata['price'])
# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(computerdata['speed'])

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(computerdata.iloc[:, :])
                             
# Correlation matrix 
computerdata.corr()

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend ', data = computerdata).fit() # regression model

# Summary
ml1.summary()


# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals

computerdata_new = computerdata.drop(computerdata.index[[1700]])

# Preparing model                  
ml_new= smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend ', data = computerdata).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_hd = smf.ols('hd~ speed + ram + screen + cd + multi + premium + ads + trend ', data = computerdata).fit().rsquared  
vif_hd = 1/(1 - rsq_hd) 

rsq_ram = smf.ols('ram~ speed + hd + screen + cd + multi + premium + ads + trend ', data = computerdata).fit().rsquared  
vif_ram = 1/(1 - rsq_ram) 

rsq_speed = smf.ols('speed~ hd + ram + screen + cd + multi + premium + ads + trend ', data = computerdata).fit().rsquared  
vif_speed = 1/(1 - rsq_speed) 

rsq_screen = smf.ols('screen~ speed + ram + hd + cd + multi + premium + ads + trend ', data = computerdata).fit().rsquared  
vif_screen = 1/(1 - rsq_screen) 

rsq_cd = smf.ols('cd~ speed + ram + screen + hd + multi + premium + ads + trend ', data = computerdata).fit().rsquared  
vif_cd = 1/(1 - rsq_cd) 

rsq_multi = smf.ols('multi~ speed + ram + screen + cd + hd + premium + ads + trend ', data = computerdata).fit().rsquared  
vif_multi = 1/(1 - rsq_multi) 

rsq_premium = smf.ols('premium~ speed + ram + screen + cd + multi + hd + ads + trend ', data = computerdata).fit().rsquared  
vif_premium = 1/(1 - rsq_premium) 

rsq_ads = smf.ols('ads~ speed + ram + screen + cd + multi + premium + hd + trend ', data = computerdata).fit().rsquared  
vif_ads = 1/(1 - rsq_ads) 

rsq_trend= smf.ols('trend~ speed + ram + screen + cd + multi + premium + ads + hd ', data = computerdata).fit().rsquared  
vif_trend = 1/(1 - rsq_trend) 

# Storing vif values in a data frame
d1 = {'Variables':['hd', 'speed', 'cd', 'ads','multi','premium','ram','screen'], 'VIF':[vif_hd, vif_speed, vif_cd, vif_ads,vif_multi,vif_premium,vif_ram,vif_screen]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As hd is having highest VIF value, but less than 10.
computerdata=computerdata.drop(computerdata.index[1700])
# Final model
final_ml = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend ', data = computerdata).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(computerdata)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
from scipy import stats
import pylab
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = computerdata.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
computerdata_train, computerdata_test = train_test_split(computerdata, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train=smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend ', data = computerdata_train).fit()

# prediction on test data set 
test_pred = model_train.predict(computerdata_test)

# test residual values 
test_resid = test_pred - computerdata_test.price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(computerdata_train)

# train residual values 
train_resid  = train_pred - computerdata_train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
