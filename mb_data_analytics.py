# -*- coding: utf-8 -*-
"""
Created on Thu May  4 22:02:15 2017

@author: abhin067
"""
#%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as st

dataframe =  pd.read_csv("C:\\Users\\abhin067\\Documents\\Data\\machine learning\\Melbourne_housing_extra_data_500.csv")
dataframe.head()
dataframe["Date"] = pd.to_datetime(dataframe["Date"],dayfirst=True)
#len(dataframe["Date"].unique())/4
var = dataframe[dataframe["Type"]=="h"].sort_values("Date", ascending=False).groupby("Date").std()
count = dataframe[dataframe["Type"]=="h"].sort_values("Date", ascending=False).groupby("Date").count()
mean = dataframe[dataframe["Type"]=="h"].sort_values("Date", ascending=False).groupby("Date").mean()
#mean["Price"].plot(yerr=var["Price"],ylim=(400000,1500000))
means = dataframe[(dataframe["Type"]=="h") & (dataframe["Distance"]<13)].dropna().sort_values("Date", ascending=False).groupby("Date").mean()
errors = dataframe[(dataframe["Type"]=="h") & (dataframe["Distance"]<13)].dropna().sort_values("Date", ascending=False).groupby("Date").std()
sns.heatmap(dataframe[dataframe["Type"] == "h"].corr(), annot=True)

from sklearn.cross_validation import train_test_split
dataframe_dr = dataframe.dropna().sort_values("Date")
from datetime import date
all_Data = []
days_since_start = [(x - dataframe_dr["Date"].min()).days for x in dataframe_dr["Date"]]
dataframe_dr["Days"] = days_since_start
dummies = pd.get_dummies(dataframe_dr[["Type", "Method", "Suburb", "SellerG", "CouncilArea"]])
dummies_other = dummies.drop(["Type_h"],axis=1)
#Type_dummies = pd.get_dummies(dataframe_dr[["Type", "Method"]])
#all_Data = dataframe_dr.drop(["Type", "Method", "Suburb", "SellerG", "CouncilArea", "Date", "Price"],axis=1).join(dummies_other)
all_Data = dataframe_dr.drop(["Type", "Method", "Suburb", "SellerG", "CouncilArea", "Date", "Price","Lattitude","Longtitude","Landsize","Bedroom2","YearBuilt","Car","Distance"],axis=1)
all_Data.head().to_csv("C:\\Users\\abhin067\\Documents\\Data\\machine learning\\out.csv")
X = all_Data
y = dataframe_dr["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
X.columns
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
ranked_factors = coeff_df.sort_values("Coefficient", ascending = False)
ranked_factors
predictions = lm.predict(X_test)
percent_diff = ((y_test - predictions)/(y_test))*100
plt.scatter(y_test, predictions)
plt.ylim([200000,1000000])
plt.xlim([200000,1000000])
sns.distplot((y_test-predictions),bins=50)
from sklearn import metrics
print("MAE:", metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100,random_state=0)
reg.fit(X_train,y_train)
pred = reg.predict(X_test)
percent_diff_other = ((y_test - pred)/(y_test))*100
                     
regressor_ols = st.OLS(endog=y_train,exog=X_train).fit()
regressor_ols.summary()