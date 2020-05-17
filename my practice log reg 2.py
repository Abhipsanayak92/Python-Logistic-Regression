# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:34:09 2020

@author: user
"""

import pandas as pd
import numpy as np

#loading data
adult_df= pd.read_csv(r'D:\DATA SCIENCE DOCS\Python docs\4 logistic regression python\10 adult_data.csv', 
                      header=None,
                      delimiter=" *, *", engine="python")

adult_df.head()

#providing header name 
pd.set_option("display.max_columns", None)
adult_df.head()

adult_df.shape
adult_df.columns= []