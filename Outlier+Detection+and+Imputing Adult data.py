
# coding: utf-8

# In[9]:


# Required Python Machine learning Packages
import pandas as pd
import numpy as np


# In[10]:


adult_df = pd.read_csv(r'D:\DATA SCIENCE DOCS\Python docs\4 logistic regression python\.csv',
                       header = None, delimiter=' *, *',engine='python')

adult_df.head()


# In[11]:


adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'capital_gain', 'capital_loss',
                    'hours_per_week', 'native_country', 'income']

adult_df.head()


# In[12]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
adult_df.boxplot() #for plotting boxplots for all the numerical columns in the df
plt.show()


# In[13]:


adult_df.boxplot(column='fnlwgt')
plt.show()


# In[14]:


adult_df.boxplot(column='capital_gain')
plt.show()


# In[15]:


adult_df.boxplot(column='capital_loss')
plt.show()


# In[16]:


adult_df.boxplot(column='hours_per_week')
plt.show()


# In[17]:


adult_df.boxplot(column='age') 
plt.show()


# In[18]:


#for value in colname:
q1 = adult_df['age'].quantile(0.25) #first quartile value
q3 = adult_df['age'].quantile(0.75) # third quartile value
iqr = q3-q1 #Interquartile range
low  = q1-1.5*iqr #acceptable range
high = q3+1.5*iqr #acceptable range


# In[19]:


adult_df_include = adult_df.loc[(adult_df['age'] >= low) &(adult_df['age'] <= high)] # meeting the acceptable range
adult_df_exclude = adult_df.loc[(adult_df['age'] < low) |(adult_df['age'] > high)] #not meeting the acceptable range


# In[20]:


print(adult_df_include.shape)


# In[21]:


print(adult_df_exclude.shape)


# In[22]:


print(low)


# In[23]:


age_mean=int(adult_df_include.age.mean()) #finding the mean of the acceptable range
print(age_mean)


# In[24]:


#imputing outlier values with mean value
adult_df_exclude.age=age_mean


# In[25]:


#getting back the original shape of df
adult_df_rev=pd.concat([adult_df_include,adult_df_exclude]) #concatenating both dfs to get the original shape
adult_df_rev.shape


# In[ ]:


#capping approach

adult_df_exclude.loc[adult_df_exclude["age"] <low, "age"] = low
adult_df_exclude.loc[adult_df_exclude["age"] >high, "age"] = high


