# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 16:43:51 2020

@author: user
"""

#DATASET: ADULT
#ALGORITHM: LOGISTIC REGRESSION
#OBJECTIVE: TO FIND INCOME CATEGORY OF AN INDIVIDUAL(LESS THAN 50K OR MORE THAN 50K)

import pandas as pd
import numpy as np

#data load:
adult_df= pd.read_csv(r'D:\DATA SCIENCE DOCS\Python docs\4 logistic regression python\10 adult_data.csv', 
                      header=None,
                      delimiter=" *, *", engine="python")


#showing 1st 5 elements of dataframe
adult_df.head()
#to display all variable:
pd.set_option("display.max_columns", None)
adult_df.head()

#to know shape of data in matrix form:
adult_df.shape

#to provide column names as data does not have column names:

adult_df.columns=['age', 'workclass','fnlwgt', 'education', 'education-num', 
                 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                 'capital-gain','capital-loss', 'hours-per-week', 'native-country',
                 'income']

adult_df.head()


#EDA
# missing values finding
adult_df.isnull().sum() #showing zero as na is not written anywhere as missing values

adult_df=adult_df.replace(["?"],np.nan) #wherever ? is there will be replaced by nan

adult_df.isnull().sum()


#creating copy of dataframe
adult_df_1= pd.DataFrame.copy(adult_df)

#missing values treatment with mode of the column as it is categorical value
for value in ["workclass", "occupation", "native-country"]:
 adult_df_1[value].fillna(adult_df_1[value].mode()[0], inplace= True)     

adult_df_1.isnull().sum()    



#labelencoding

colname= ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country","income"]
colname

 
from sklearn import preprocessing
le={}
le=preprocessing.LabelEncoder()

for x in colname:
    adult_df_1[x]= le.fit_transform(adult_df_1[x]) 

  
    
adult_df_1.head()    
    

#data splitting to X and Y ie input variable and output variable 
adult_df.dtypes
X=adult_df_1.values[:,:-1]
Y=adult_df_1.values[:,-1]   
    
    
#for normalisation of data using standardscaler fro sklearn package
from  sklearn.preprocessing import StandardScaler

scaler= StandardScaler()

scaler.fit(X)
X=scaler.transform(X)
print(X)


Y=Y.astype(int)
Y.dtype

#splitting to train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.3, random_state=10)    



#appying algorithm logisticregression
from sklearn.linear_model import LogisticRegression 
classifier=LogisticRegression()
classifier.fit(X_train, Y_train)

Y_pred=classifier.predict(X_test)   
    
print(list(zip(Y_test, Y_pred)))
print(classifier.coef_)
print(classifier.intercept_)
    
#confusion matrix building 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report  
cfm= confusion_matrix(Y_test, Y_pred)
print(cfm)

print("classification_report:")
print(classification_report(Y_test, Y_pred))    

acc=accuracy_score(Y_pred, Y_test)
print("accuracy score:", acc)    
    
    
    
    
    
    
    