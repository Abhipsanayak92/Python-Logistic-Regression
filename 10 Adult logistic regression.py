# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:16:59 2019

@author: user
"""

#%% logistic regression
#objective- to predict income category of an indivudual
import pandas as pd
import numpy as np

#%%
adult_df= pd.read_csv(r'D:\DATA SCIENCE DOCS\Python docs\4 logistic regression python\10 adult_data.csv', 
                      header=None,
                      delimiter=" *, *", engine="python")

#bydeafult header = 0, means we are not having any header,
#so it should not take 1st obs as 1st header
#delimiter is to ignore space in missing values
#(means space after ? and spaces before ? and spaces at both side of ?)
adult_df.head()

#%% to display all variable
pd.set_option("display.max_columns", None)
adult_df.head()

#%%
adult_df.shape
#we are writing those variables which we need for building model
adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']

adult_df.head()

#%%preprocessing the data

adult_df.isnull().sum()  #isnull() will take those missing values where na is written

adult_df= adult_df.replace(["?"], np.nan)

adult_df.isnull().sum()

#%%create a copy of dataframe

adult_df_rev= pd.DataFrame.copy(adult_df)

#%%replacing the missing values with valus in the top row of each column
for value in ["workclass", "occupation", "native_country"]:
    adult_df_rev[value].fillna(adult_df_rev[value].mode()[0], inplace = True)
 
adult_df_rev.isnull().sum()

#%% when you have missing values in mixed values(eg int,float,categorical , objects)
"""
for x in adult_df_rev.columns[:]:
if adult_df_rev[x].dtype=='object':
adult_df_rev[x].fillna(adult_df_rev[x].mode()[0],inplace=True)
elif adult_df_rev[x].dtype=='int64':
adult_df_rev[x].fillna(adult_df_rev[x].mean(),inplace=True)
"""

#%%
adult_df_rev.workclass.value_counts()
#####################why only these 8 variables among 15 variables bczwewill do flagging in these 8 variables by labelencoding
colname= ['workclass', 'education','marital_status', 'occupation', 'relationship',
'race', 'sex', 'native_country', 'income']
colname

#%% for preprocessing the data
# everything converted to numbers
#labelenoding always done in alphabetical order]
from  sklearn import preprocessing

le={}

le= preprocessing.LabelEncoder()   

for x in colname:
    adult_df_rev[x]= le.fit_transform(adult_df_rev[x])
    
#%%
adult_df_rev.head()
#############doubt where we made this flagging
#0 for <= 50k
#1 for >50k

#%%
adult_df_rev.dtypes

X=adult_df_rev.values[:,:-1] #weare selecing all variables except last one as x
Y=adult_df_rev.values[:,-1] #we are selecting only last variable as y


#%% 
from sklearn.preprocessing import StandardScaler
#to scale the data to unit variance (means normalizing the data)
scaler = StandardScaler()

scaler.fit(X)

X=scaler.transform(X)

print(X)

#%%

Y=Y.astype(int)

#%%running a  basic model
from sklearn.model_selection import train_test_split
#splitting data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.3,random_state=10)
#########################doubt random state =10 and wht is x y
#%%

from sklearn.linear_model import LogisticRegression

#create model

classifier =LogisticRegression()
#fitting training data to the model
classifier.fit(X_train, Y_train)

Y_pred= classifier.predict(X_test)

print(list(zip(Y_test, Y_pred)))  #to merge 2 lists, zip() used
 
print(classifier.coef_)

print(classifier.intercept_)

#%%
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cfm= confusion_matrix(Y_test, Y_pred)
print(cfm)

print("Classication report: ")
print(classification_report(Y_test, Y_pred))

acc = accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)

#%%