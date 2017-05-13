import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

# Importing the dataset
dataset = pd.read_csv('xAPI-Edu-Data.csv')

X = pd.DataFrame(dataset.iloc[:, 0:-1])
y = pd.DataFrame(dataset.iloc[:, -1])

df_y_dropped = dataset.drop(list(y.columns.values),axis=1)
# Encoding categorical data
_str_Cols = list(dataset.select_dtypes(include=['object']).columns.values)


def ConvertCategoricalDataInDummies(dataframe):
    #Converting categorical data into numbers
    _str_Cols = list(dataframe.select_dtypes(include=['object']).columns.values)    
    dummies = pd.get_dummies(dataframe,columns=_str_Cols)
    
    #Drop last column for statistical data consistency
    return dataframe.drop(dataframe,axis=1).join(dummies)

_categoryEnodedFeatures=ConvertCategoricalDataInDummies(X)
_categoryEnccodedVariable=ConvertCategoricalDataInDummies(y)
X=(_categoryEnodedFeatures.iloc[:,:].values)
y=(_categoryEnccodedVariable.iloc[:,:].values)
X=X[:,0:-1]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

#input layer and first hidden layer
classifier.add(Dense(output_dim=35,activation='relu',init='uniform',input_dim=71))

#second hidden layer
classifier.add(Dense(output_dim=35,activation='relu',init='uniform'))

#output layer
classifier.add(Dense(output_dim=3,activation='softmax',init='uniform'))

#ccomple
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(X_test)


ymax=np.amax(y_pred,axis=1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
