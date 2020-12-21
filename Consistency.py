#!/usr/bin/env python
# coding: utf-8

# # Does the training score approach the OLS equivalent when we increase the training sample. Let's try.

# In[1]:


# Importing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[2]:


#For this project we are going to use data on housing prices in Argentina.
#importing the data.
df=pd.read_csv(r'C:\Users\Rodgers\Desktop\Machine learning\Argentina1.csv')
print(df.head())
print(df.info())


# In[4]:


#Simple OLS model.
y=df['price']
x=df[['rooms', 'bathrooms','surface_total','surface_covered','bedrooms']]
x = sm.add_constant(x)
model=sm.OLS(y,x).fit()
model_prediction=model.predict(x)
model_details=model.summary()
print(model_details)


# In[20]:


#train test split sklearn, starting with a training size of 0.9
X_train,X_test,y_train,y_test=train_test_split(x,y, test_size=0.1, random_state=25)


# In[22]:


lr = LinearRegression()
lr.fit(X_train,y_train)
y_predict=lr.predict(X_test)
print(lr.coef_)


# In[46]:


test_size_=[0.1,0.2,0.3,0.4,0.5]


# In[50]:


def Learn(x,y,list_):
    #for each test_size, train a model and return the coefficients
    for i in list_:
        X_train,X_test,y_train,y_test=train_test_split(x,y, test_size=i, random_state=25)
        lr = LinearRegression()
        lr.fit(X_train,y_train)
        return Learn (x,y,list_)


# In[ ]:


Learn(x,y,test_size_)


# In[26]:


#train test split sklearn, starting with a training size of 0.8
X_train,X_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=25)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_predict=lr.predict(X_test)
print(lr.coef_)


# In[27]:


#train test split sklearn, starting with a training size of 0.7
X_train,X_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=25)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_predict=lr.predict(X_test)
print(lr.coef_)


# In[28]:


#train test split sklearn, starting with a training size of 0.6
X_train,X_test,y_train,y_test=train_test_split(x,y, test_size=0.4, random_state=25)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_predict=lr.predict(X_test)
print(lr.coef_)


# In[29]:


#train test split sklearn, starting with a training size of 0.5
X_train,X_test,y_train,y_test=train_test_split(x,y, test_size=0.5, random_state=25)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_predict=lr.predict(X_test)
print(lr.coef_)


# In[30]:


#train test split sklearn, starting with a training size of 0.95
X_train,X_test,y_train,y_test=train_test_split(x,y, test_size=0.05, random_state=25)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_predict=lr.predict(X_test)
print(lr.coef_)


# #### As the sample size that we use to train our ML models grows the coeffcients converge to their true values. If the sample is big enough, then we need not worry if the model is learning the correct coefficients. However if the sample is small, most Ml methods typically do not learn the correct coefficients.

# #### As the sample size goes to the true size of the data set, the ML model fits converge to the model estimates that we get with simple OLS.

# In[ ]:




