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
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import configparser
from sklearn.ensemble import GradientBoostingRegressor
import shutil
import os
import math

config = configparser.RawConfigParser()
config.read("C:/Users/animeshl913/Desktop/Hack-A-Thon 2017/MachineLearnngWebAPI/restapp/config.ini")
_filepath = config.get("File","filepath")

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
    x_best = x_best_fit_obj.fit_transform(X,y)
    indices = x_best_fit_obj.fit(X,y).get_support(indices=True)
    return x_best,indices

#MLRE Model
def GenerateLinearRegressionModel(X_train,Y_train):
    lm = LinearRegression()
    lm.fit(X_train,Y_train)
    return lm 

#Random Forest Model
def GenerateRandomForestModel(X_train, Y_train):
    reg = RandomForestRegressor(n_estimators=500,random_state=0)
    reg.fit(X_train,Y_train)
    return reg

def GenerateGradientBoostModel(X_train, Y_train):
    gradient_boost_reg = GradientBoostingRegressor(n_estimators=300,learning_rate=0.05,random_state=0)
    grad_boost_model = gradient_boost_reg.fit(X_train,Y_train)
    return grad_boost_model

def GetDataFrameFromCSV(_filePath):
    dataframe =  pd.read_csv(_filePath)
    dataframe_blank_removed = dataframe.dropna(axis=0)
    return dataframe_blank_removed


def ConvertDateFieldsToDays(dataframe):
    #Date Cloumn Formatting
    #Get all String Columns name list
    _str_Cols = list(dataframe.select_dtypes(include=['object']).columns.values)
    _str_Date_Col_Header = ""
    for cat in _str_Cols:
        try:
            dataframe[cat] = pd.to_datetime(dataframe[cat],dayfirst=True)
            #Assign column Name
            if _str_Date_Col_Header =="":
                _str_Date_Col_Header = cat
        except:
            #do nothing
            pass
            #print("Error")
    
    #converting date in days in start
    days_since_start = [(x - dataframe[_str_Date_Col_Header].min()).days for x in dataframe[_str_Date_Col_Header]]
    dataframe["Days"] = days_since_start
    #drop date column
    df_other=dataframe.drop(_str_Date_Col_Header,axis=1)
    return df_other

def ConvertCategoricalDataInDummies(dataframe):
    #Converting categorical data into numbers
    _str_Cols = list(dataframe.select_dtypes(include=['object']).columns.values)    
    dummies = pd.get_dummies(dataframe,columns=_str_Cols)    
    #Drop last column for statistical data consistency
    return dataframe.drop(dataframe,axis=1).join(dummies)

def GetDataPostSanitization(_filePath):
    dataframe= GetDataFrameFromCSV(_filePath)
    #Get Y value
    y = pd.DataFrame(dataframe.iloc[:,-1])
    #Drop Y value column
    df_y_dropped = dataframe.drop(list(y.columns.values),axis=1)
    
    df_y_dropped_date_converted = ConvertDateFieldsToDays(df_y_dropped)
    all_Data = ConvertCategoricalDataInDummies(df_y_dropped_date_converted)  
    
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

def GetGradientBoostModelAndScore(X,y):
    X_train, X_test, y_train, y_test = SplitDataSetTrainTest(X,y)
    grad_boost_model = GenerateGradientBoostModel(X_train,y_train)
    score_grad_boost = GetModelScore(grad_boost_model,X_train,y_train,10)
    preds = grad_boost_model.predict(X_test)
    return grad_boost_model,score_grad_boost.mean(),preds,y_test

def plotModelScatter(y_test_LM, pred_LM,plotName, x_low_lim=0,x_high_lim=1000000,y_low_lim=0,y_high_lim=1000000,color='red'):
    plt.ylim([y_low_lim,y_high_lim])
    plt.xlim([x_low_lim,x_high_lim])
    scatter = plt.scatter(y_test_LM, pred_LM,color=color)
    plt.savefig(plotName+'.png')
    scatter.remove()
    
def GetHyperParametersTuned(model, X, y):
    # use a full grid over all parameters
    param_grid = {"max_depth": [None,10,100],
              "max_features": [10,100,200,300],
              "min_samples_split": [10,100,1000],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              #"criterion": ["gini", "entropy"]
              }
    grid_search = GridSearchCV(model, param_grid=param_grid,cv=2)
    grid_search.fit(X, y)
    return grid_search.best_params_

def ExportClassifier(model, model_Name):
    joblib.dump(model, model_Name+'.pk1')
    return True

def ImportClassifier(model_Name):
    try:
        model = joblib.load(model_Name)
        return model
    except:
        return None

def createDirectory(_filePath,_problemStatement):
    if os.path.exists(_problemStatement):
        shutil.rmtree(_problemStatement)
    os.makedirs(_problemStatement)
    os.makedirs(os.path.join(_problemStatement, 'Data'))
    os.makedirs(os.path.join(_problemStatement, 'Plots'))
    os.makedirs(os.path.join(_problemStatement, 'Models'))

    #Copying data files
    shutil.copy2(_filePath, os.path.join(".\\"+_problemStatement, 'Data'))
    
def logdataOnConfigINI(_problemStatement,selectedModel,accuracyScore):
    configpath='C:/Users/animeshl913/Desktop/Hack-A-Thon 2017/MachineLearnngWebAPI/restapp/config.ini'
    config.set('Outputs',_problemStatement,selectedModel+"|"+accuracyScore)
    with open(configpath,'w') as configfile:
        config.write(configfile)
        
#_param_list='Suburb=Abbotsford,Rooms=2,Type=h,Method=S,SellerG=Biggin,Date=2016-02-04 00:00:00.000,Distance=2.5,Postcode=3067,Bedroom2=2,Bathroom=1,Car=0,Landsize=156,BuildingArea=79,YearBuilt=1900,CouncilArea=Yarra'
def Predict(_param_list):
    #get model data
    dataframe = GetDataFrameFromCSV(".\Melbourne Housing Prediction\Data\Melbourne_housing_data_blank_removed.csv")
    dataframe_y_dropped = dataframe.drop("Price",axis=1)
    _model='.\\Melbourne Housing Prediction\\Models\\Gradient_Boost_Model.pk1'
    model = ImportClassifier(_model)
    _params = _param_list.split(",")
    index=[]
    value=[]
    for param in _params:
        index.append(param.split("=")[0])
        try:
            if "." in param.split("=")[1]:
                value.append(float(param.split("=")[1]))
            else:
                value.append(int(param.split("=")[1]))
        except:
                value.append(param.split("=")[1])
    new_dict = dict()
    for val,key in zip(value,index):
        new_dict[key]=val
    
    df=pd.DataFrame(new_dict,index=[0])
    dataframes = [dataframe_y_dropped,df]
    merged_dataframe = pd.concat(dataframes,ignore_index=True,axis=0)
    merged_dataframe_date_converted = ConvertDateFieldsToDays(merged_dataframe)
    merged_dataframe_date_converted_categories=ConvertCategoricalDataInDummies(merged_dataframe_date_converted)
    #Get Last Row in Dataframe
    pred_data = merged_dataframe_date_converted_categories.iloc[[-1]]
    pred = model.predict(pred_data)
    return str(math.floor(pred*0.9)) + "-" + str(math.floor(pred*1.1))
    
    
def train(_filePath,_problemStatement):
    createDirectory(_filePath,_problemStatement)
    #_filePath = 'fundamentals.csv'
    X,y = GetDataPostSanitization(_filePath)
    linearModel,score_LM,feature_indices,pred_LM,y_test_LM = GetLinearRegressorModelAndScore(X,y)
    ExportClassifier(linearModel,'.\\'+_problemStatement+'\\Models\\'+"Linear_Model")
    RFModel,score_RF,predictions,y_test_RF = GetRandomForestModelAndScore(X,y)
    ExportClassifier(RFModel,'.\\'+_problemStatement+'\\Models\\'+"Random_Forest_Model")
    GBModel,score_GB,pred_GB,y_test_GB = GetGradientBoostModelAndScore(X,y)
    ExportClassifier(GBModel,'.\\'+_problemStatement+'\\Models\\'+"Gradient_Boost_Model")
    plotModelScatter(y_test_LM,pred_LM,'.\\'+_problemStatement+'\\Plots\\'+'linearModel',0,pred_LM.max(),0,pred_LM.max(),'blue')
    plotModelScatter(y_test_RF,predictions,'.\\'+_problemStatement+'\\Plots\\'+'RandomForestModel',0,predictions.max(),0,predictions.max(),'Green')
    plotModelScatter(y_test_GB,pred_GB,'.\\'+_problemStatement+'\\Plots\\'+'GradientBoostModel',0,pred_GB.max(),0,pred_GB.max(),'red')
    new_dict={'LM':score_LM,'RF':score_RF,'GB':score_GB}    
    selectedMOdel=max(new_dict, key=lambda i:new_dict[i])
    logdataOnConfigINI(_problemStatement,str(selectedMOdel),str(new_dict.get(selectedMOdel)))
    return new_dict,selectedMOdel
