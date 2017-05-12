# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:13:58 2017

@author: abhin067
"""
#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as st
import seaborn as sns
import bisect
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from datetime import date
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from time import time

def GetOptimizedFeaturedDataSetLR(X, y):
    adj_rsquared_arr = []
    iterator = 1
    #Get dataframe column Count
    col_count = X.shape[1]
    #calculates adjusted R-squared value 
    while(iterator<=col_count):
        X_train_temp=SelectKBest(f_regression,k=iterator).fit_transform(X,y)
        regressor_ols_temp = st.OLS(endog=y,exog=X_train_temp).fit()
        adj_rsquared_arr.append(regressor_ols_temp.rsquared_adj)
        iterator = iterator + 1
    
    #return dataset with best features
    x_best_fit_obj = SelectKBest(f_regression,k=col_count - adj_rsquared_arr.index(max(adj_rsquared_arr)))
    #x_best_fit_obj = SelectKBest(f_regression,k=293)
    x_best = x_best_fit_obj.fit_transform(X,y)
    indices = x_best_fit_obj.fit(X,y).get_support(indices=True)
    return x_best,indices

#MLRE Model
def GenerateLinearRegressionModel(X_train,Y_train):
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    #prediction=lm.predict(X_test)
    return lm 

#Random Forest Model
def GenerateRandomForestModel(X_train, Y_train):
    reg = RandomForestRegressor(n_estimators=500,random_state=0)
    reg.fit(X_train,Y_train)
    return reg

def GetDataPostSanitization(_filePath):
    dataframe =  pd.read_csv(_filePath)
    dataframe.head()
    
    #Get Y value
    y = pd.DataFrame(dataframe.iloc[:,-1])
    #Drop Y value column
    df_y_dropped = dataframe.drop(list(y.columns.values),axis=1)
    #df_y_dropped = dataframe
    
    #Date Cloumn Formatting
    #Get all String Columns name list
    _str_Cols = list(df_y_dropped.select_dtypes(include=['object']).columns.values)
    _str_Date_Col_Header = ""
    for cat in _str_Cols:
        try:
            df_y_dropped[cat] = pd.to_datetime(df_y_dropped[cat],dayfirst=True)
            #Assign column Name
            if _str_Date_Col_Header =="":
                _str_Date_Col_Header = cat
        except:
            #do nothing
            pass
            #print("Error")
    
    all_Data = []
    #converting date in days in start
    days_since_start = [(x - df_y_dropped[_str_Date_Col_Header].min()).days for x in df_y_dropped[_str_Date_Col_Header]]
    df_y_dropped["Days"] = days_since_start
    #drop date column
    df_other=df_y_dropped.drop(_str_Date_Col_Header,axis=1)   
    
    #Converting categorical data into numbers
    _str_Cols = list(df_y_dropped.select_dtypes(include=['object']).columns.values)    
    dummies = pd.get_dummies(df_other,columns=_str_Cols)
    
    #Drop last column for statistical data consistency
    #dummies_other = dummies.drop(dummies.columns[len(dummies.columns)-1],axis=1)
    all_Data = df_other.drop(df_other,axis=1).join(dummies)
    #all_Data = df_other.drop(df_other,axis=1).join(dummies_other)
    #return all_Data
    return all_Data,y

def SplitDataSetTrainTest(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
    return X_train, X_test, y_train, y_test

def GetLinearRegressorModelAndScore(X, y):
    #append ones to the train set
    X = np.append(arr=np.ones((X.shape[0],1)).astype(int),values=X,axis=1)    
    X_train, X_test, y_train, y_test = SplitDataSetTrainTest(X,y)    
    #Get best Features
    X_best_fit,indices = GetOptimizedFeaturedDataSetLR(X_train,y_train)
    
    #Get MLRE model
    lm = GenerateLinearRegressionModel(X_best_fit,y_train)
    score = GetModelScore(lm,X_best_fit,y_train,100)
    score_in_range = list(filter(lambda x:((x>=-1.0)&(x<=1.0)),score))
    X_test_feature_transformed = X_test[0]
    X_test_feature_transformed = (pd.DataFrame(X_test))[indices.tolist()]
    pred = lm.predict(X_test_feature_transformed)
    return lm,np.mean(score_in_range),indices,pred,y_test

def GetModelScore(model, X_train, y_train, CV):
    return cross_val_score(estimator=model,X=X_train,y=y_train,cv=CV)

def GetRandomForestModelAndScore(X, y):
    #Split test data set
    X_train, X_test, y_train, y_test = SplitDataSetTrainTest(X,y)
    reg = RandomForestRegressor(n_estimators=500,random_state=0)
    reg.fit(X_train,y_train)
    score = GetModelScore(reg,X_train,y_train,10)
    preds = reg.predict(X_test)
    return reg,score.mean(),preds,y_test

def plotLMvsRF(y_test_LM, pred_LM, y_test_RF, pred_RF):
    plt.ylim([0,1000000])
    plt.xlim([0,1000000])
    plt.scatter(y_test_LM, pred_LM,color="red")
    plt.scatter(y_test_RF, pred_RF,color="blue")

_filePath = "Melbourne_housing_data_blank_removed.csv"
X,y = GetDataPostSanitization(_filePath)
linearModel,score_LM,feature_indices,pred_LM,y_test_LM = GetLinearRegressorModelAndScore(X,y)
RFModel,score_RF,predictions,y_test_RF = GetRandomForestModelAndScore(X,y)
plotLMvsRF(y_test_LM,pred_LM,y_test_RF,predictions)

# use a full grid over all parameters
param_grid = {"max_depth": [None,10,100],
              "max_features": [10,100,200,300],
              "min_samples_split": [10,100,1000],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              #"criterion": ["gini", "entropy"]
              }
grid_search = GridSearchCV(RFModel, param_grid=param_grid,cv=2)
start = time()
grid_search.fit(X, y)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)

from sklearn.ensemble import GradientBoostingRegressor
gradient_boost_reg = GradientBoostingRegressor(n_estimators=300,learning_rate=0.05,random_state=0)
grad_boost_model = gradient_boost_reg.fit(X,y)
score_grad_boost = GetModelScore(grad_boost_model,X_train,y_train,10)
print(score_grad_boost.mean())
X_train, X_test, y_train, y_test = SplitDataSetTrainTest(X,y)


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
rf_features=[]
rf_features=clf.feature_importances_
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_new.shape(150, 2)

#reg = RandomForestRegressor(n_estimators=500,random_state=0)
#reg.fit(X_train,y_train)

percent_diff_other = ((y_test - pred)/(y_test))*100
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
dataset = datasets.load_iris()
model = ExtraTreesClassifier()
model.fit()
from sklearn.metrics import accuracy_score
feature_importance = reg.feature_importances_
np_arr = np.array(y_test, dtype=pd.Float64Index)
score = accuracy_score(np_arr[0], predictions[0])
accuracy_score(y_test, pred)




from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
poly = PolynomialFeatures(degree=2)
X_train_ = poly.fit_transform(X_train)
y_train_ = poly.fit_transform(y_train)
X_test_ = poly.fit_transform(X_test)
#X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2,random_state=0)
preds = GenerateLinearRegressionModel(X_train, y_train, X_test)

from sklearn.model_selection import cross_val_score
scores=cross_val_score(estimator=reg,X=X_train,y=y_train,cv=10)
