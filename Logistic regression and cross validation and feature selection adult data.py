
# coding: utf-8

# In[54]:



import pandas as pd
import numpy as np


# In[55]:


adult_df = pd.read_csv(r'D:\DATA SCIENCE DOCS\Python docs\4 logistic regression python\10 adult_data.csv',header = None, delimiter=' *, *',engine='python')

adult_df.head()


# In[56]:
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# In[57]:
adult_df.shape


# In[58]:#as there is no header in data, so we are assigning this header manually
adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 
                    'education_num','marital_status', 'occupation',
                    'relationship','race', 'sex', 'capital_gain',
                    'capital_loss','hours_per_week', 'native_country',
                    'income']

adult_df.head()


# **Pre processing the data**

# In[59]:


adult_df.isnull().sum()


# In[60]:

#here we replace nan wherver ? is present in our data so that we can find no of missing values
adult_df=adult_df.replace(['?'], np.nan)


# In[61]:
adult_df.isnull().sum()
#no of missing values

# In[62]:


#create a copy of the dataframe
adult_df_rev = pd.DataFrame.copy(adult_df)

#adult_df_rev.describe(include= 'all')


# In[63]:


#replace the missing values with values in the top row of each column
for value in ['workclass', 'occupation',
              'native_country']:
    adult_df_rev[value].fillna(adult_df_rev[value].mode()[0],inplace=True)


# In[64]:


adult_df_rev.workclass.mode()


# In[65]:


"""
for x in adult_df_rev.columns[:]:
    if adult_df_rev[x].dtype=='object':
        adult_df_rev[x].fillna(adult_df_rev[x].mode()[0],inplace=True)
    elif adult_df_rev[x].dtype=='int64':
        adult_df_rev[x].fillna(adult_df_rev[x].mean(),inplace=True)
"""


# In[66]:


adult_df_rev.isnull().sum()
#adult_df_rev.head()


# In[67]:
"""converting categorical var to numeric can be done by
 a) creating dummy variable  
 b) creating levels
 c) manual encoding
#to create dummy var- get_dummy() in pandas
# in sklearn, OneHotEncoder() also used for creating dummy var
"""

adult_df_rev.workclass.value_counts()


# In[68]:

#these are the column name which are having categorical variable
colname = ['workclass', 'education',
          'marital_status', 'occupation',
          'relationship','race', 'sex',
          'native_country', 'income']
colname


# In[69]:
"""label encoder will create a dictionary in background and all unique variables will be taken as keys
this labeling will happen in alphabetical order only
eg private , self emp, local govt, state govt- it will take 0,1,2,3 etc acc to alphabetical order
"""

# For preprocessing the data
from sklearn import preprocessing #syntax - from <parent package> import <sub package>


le={}

le=preprocessing.LabelEncoder()
#fit wil create levels and poplulate the dictionary
#transform will map those levels in to variables

for x in colname:
     adult_df_rev[x]=le.fit_transform(adult_df_rev[x])


# In[70]:

#just to ensure that every categorical var has transformed to numbers by printing headers
adult_df_rev.head()


#here it is important to note down dependent var leveling (y variable)
#bcz it will helpful in interpreting confusion matrix
#0--> <=50K
#1--> >50K


# In[71]:
adult_df_rev.dtypes
#we are printing data types of each variable to cross chk the leveling is done or not


# In[72]:
"""
creating array X and Y 
X being array of dependent var
Y being array of independent var

always try creating y first as it contains only one var
"""
X = adult_df_rev.values[:,0:-1]
# we are taking all rows and only columns except income ie last column
#so [:,0:-1] means all rows but column from 0th index to -2 index ie excluding income var
#here in in column place(ie after comma, [inclusive:exclusive] format works)

Y = adult_df_rev.values[:,-1] #before comma is rows, after comma is column
#here y variable is income var, so we can either pass +ve ibdexing or _ve indexing
#here income being last column we are writing -1



# In[73]:
#scaling is not mandatory but it will increase overall accuracy

"""
1, normalization



"""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)  #it will find the ranges which is away from  means
#we are scaling X variable here
#no y variable as there is only 1 variable

X = scaler.transform(X) # every thing is in range of -3 to +3
print(X)


# In[74]:


#np.set_printoptions(threshold=np.inf)


# In[75]:

#converting y var to numbers
Y=Y.astype(int)


#%%
# **Running a basic model**

# In[76]:

from sklearn.model_selection import train_test_split

#Split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                    random_state=10)  
#sklearn bydefault takes ratio of 75% and 25%
#we used to take 80-20(less than 1000 obs)  or 70-30 ratio (>1000 obs)
#random_state= 10 means your output will become same as others
#it is helpful in grp project as everyone's output will be same
#even if i am not writing random_state then also it will not affect much
#random_state range can be anything

# In[77]:


from sklearn.linear_model import LogisticRegression
#create a model
classifier=LogisticRegression()
#fitting training data to the model
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))

print(classifier.coef_)
print(classifier.intercept_)


# In[78]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)



# **Adjusting the threshold**

# In[28]:


# store the predicted probabilities
y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)


# In[29]:


y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value > 0.45:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)

"""
we are taking 0.46 here as we found in 0.46 threshold we are gettig lowest error and that too lowest type 2 model
we have done this by hit and trial method

"""
y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value > 0.46:  
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)

# In[30]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,y_pred_class)
print(cfm)
acc=accuracy_score(Y_test, y_pred_class)
print("Accuracy of the model: ",acc)
print(classification_report(Y_test, y_pred_class))






# In[31]:


#arange()- the range should be in between 0 to 1 and in interval of 0.05(threshold of 0.05)
for a in np.arange(0,1,0.05):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    #type 1 error =cfm[0,1] and type 2 error = cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
          cfm[1,0]," , type 1 error:", cfm[0,1])


#in interval of 0.01 i.e threshold = 0.01
for a in np.arange(0,1,0.01):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    #type 1 error =cfm[0,1] and type 2 error = cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
          cfm[1,0]," , type 1 error:", cfm[0,1])

#we are finding here in which threshold we are getting lowest error and that too in that lowest type 2 error


"""
#we are taking 0.46 here as we found in 0.46 threshold we are gettig lowest error and that too lowest type 2 model
"""
y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value > 0.46:  
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)

# In[30]:
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,y_pred_class)
print(cfm)
acc=accuracy_score(Y_test, y_pred_class)
print("Accuracy of the model: ",acc)
print(classification_report(Y_test, y_pred_class))


# **Running model using cross validation**

# In[32]:


#Using cross validation

classifier=(LogisticRegression())

#performing kfold_cross_validation
from sklearn.model_selection import KFold
kfold_cv=KFold(n_splits=10)
print(kfold_cv)

from sklearn.model_selection import cross_val_score
#cross_val_score will give accuracy of all folds
#running the model using scoring metric as accuracy
kfold_cv_result=cross_val_score(estimator=classifier,X=X_train,
                                                 y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean()) #taking mean of all accuracy of folds

#here we are getting mean accuracy as 82.43
#and before cross validation our model accuracy was 82.27
#both accuracy are almost identical so we dont need to use cross validation here 


for train_value, test_value in kfold_cv.split(X_train):
    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])

#fit() will work on 1st 9 fold of each iterations in training data 
#predict() will work on 10th fold in testing data on each iteration
 

Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))

# In[33]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusion_matrix=confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)
print()


print("Classification report: ")

print(classification_report(Y_test,Y_pred))

accuracy_score=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",accuracy_score)

#here we did not need cross validated model
#still we ran the model with cross validation model
#thats why it did not give any significant difference in accuracy




#%%
# **Feature selection using Recursive Feature Elimination**

colname=adult_df_rev.columns[:]


# In[35]:


from sklearn.feature_selection import RFE
rfe = RFE(classifier, 8) #we need 8 most imp variable
model_rfe = rfe.fit(X_train, Y_train)
print("Num Features: ",model_rfe.n_features_)
print("Selected Features: ") #where true is written means that variable is kept in model
#false means that variable was eliminated

print(list(zip(colname, model_rfe.support_)))
print("Feature Ranking: ", model_rfe.ranking_) 
#highest number was eliminated 1st followed by subsequent variables
#variables which are written as 1 are kept in model
#here we can see that many important variables are eliminated by RFE like education and occupation
# so we should not use this at the very 1st go

# In[36]:


Y_pred=model_rfe.predict(X_test)
#print(list(zip(Y_test,Y_pred)))


# In[37]:

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusion_matrix=confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)
print()


print("Classification report: ")

print(classification_report(Y_test,Y_pred))

accuracy_score=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",accuracy_score)

#output we are getting 82.23 after RFE method execution with only 8 variables
#base model was giving accuracy as 82.27 with 15 variables
# In[38]:
#here we are subsetting whole data variable which are important as per RFE and those variabkes which I feel important but which was eliminated by RFE
#also i have included Y variable
#we built new model according to RFE and our business knowledge and logical sense

"""new_data=adult_df_rev[['age','workclass','occupation','race','sex','income']]
new_data.head()
new_X=new_data.values[:,:-1]
new_Y=new_data.values[:,-1]
print(new_X)
print(new_Y)
"""


#%%
# **Feature selection using Univariate Selection**

X = adult_df_rev.values[:,:-1]
Y = adult_df_rev.values[:,-1]

#this step is to remove non negative values as selectkbest will not work on negative value
# In[40]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


test = SelectKBest(score_func=chi2, k=11) #11 most imp variables i want to keep
fit1 = test.fit(X, Y)

print(fit1.scores_)
print(list(zip(colname,fit1.get_support())))
#here also we are finding some imp variales are not getting selected
X = fit1.transform(X)

print(X)


# In[41]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)


# In[42]:


from sklearn.model_selection import train_test_split

#Split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=10)  


# In[43]:


from sklearn.linear_model import LogisticRegression
#create a model
classifier=(LogisticRegression())
#fitting training data to the model
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))


# In[44]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusion_matrix=confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)
print()


print("Classification report: ")

print(classification_report(Y_test,Y_pred))

accuracy_score=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",accuracy_score)

#accuracy = 82.24 after using chi square using 10 variables


#%%
# ** Variance Threshold**

X = adult_df_rev.values[:,:-1]
Y = adult_df_rev.values[:,-1]


# In[46]:

from sklearn.feature_selection import VarianceThreshold

#scaling required
vt = VarianceThreshold(0.0)
fit1 = vt.fit(X, Y)
print(fit1.variances_)

X = fit1.transform(X)
print(X)
print(X.shape[1])
print(list(zip(colname,fit1.get_support())))

#no variables eliminated

#%%
from sklearn.feature_selection import VarianceThreshold

#scaling required
vt = VarianceThreshold(0.3)
fit1 = vt.fit(X, Y)
print(fit1.variances_)

X = fit1.transform(X)
print(X)
print(X.shape[1])
print(list(zip(colname,fit1.get_support())))
