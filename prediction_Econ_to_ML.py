#!/usr/bin/env python
# coding: utf-8

# # Predicting House Prices, Econometrics(OLS) & ML

# The data is from https://www.properati.com.ar/publish and is used only for puposes of demonstrating model fitting as in machine learning and Model estimation in Econometrics. The datasets contain real estate listings in Argentina, Colombia, Ecuador, Perú, and Uruguay. With information on number of rooms, districts, prices, etc.They include houses, apartments, commercial lots, and more.We want to model the price of the house ('the label' in ML, the dependent variable in Economics) as a 
# function of all the relevant attributes(features as in ML and Variables in Econ). We will fit the model( as in ML) and Estimate the model(As in Econ)

# In[1]:


#import modules:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[28]:


#read in the raw data set.
df=pd.read_csv(r'C:\Users\Rodgers\Desktop\Machine learning\Kaggle Datasets\ar_properties.csv')
print(df.head())
print(df.info())


# In[3]:


# in this cell we drop the columns that we will not be using.Most of them do not bias in any way the estimation we are trying
#achieve e.g. for the decsription, all examples(observations) are properties. So we don't need that column.
df=data.drop(['id','title','description','ad_type','created_on','l5','l6','start_date','end_date','lat','lon','l2','l4'], axis=1)


# In[4]:


#print(df.head())
print(df.shape) #after dropping unneccesary columns, we are left with 12, down from 25


# In[4]:


#Here, we just play with the code. This function takes in a data frame and tells us how many unique values, there are
#per feature
def unique_values(dataframe):
    list_=[]
    for i in dataframe.columns:
        list_.append(dataframe[i].nunique())
    return list_


# In[5]:


#This one tells us what are the categories per variable:
def unique_types(dataframe):
    for i in dataframe.columns:
        print(dataframe[i].unique())


# In[6]:


#Testing our functions...
print(unique_values(df))
#print(unique_types(df)) #not quite useful when the data is not categorical


# In[7]:


#Here we take a look at first five columns of the reduced data set.
print(df.head())


# In[8]:


#df.apply(lambda x: x.fillna(x.mean()))


# In[9]:


#converting all the currencies to USD, based on Google exchange rates on June 12th 2020. Done for internal consistency 
#in estimation.
df.loc[df.currency=='ARS','price']=df['price']*0.014
df.loc[df.currency=='UYU','price']=df['price']*0.023
df.loc[df.currency=='PEN','price']=df['price']*0.29


# In[10]:


#selecting examples per country, it would be prudent to only consider houses within cities, within the same country
Argentina=df.loc[df['l1'] == 'Argentina']
Uruguay=df.loc[df['l1'] == 'Uruguay']
Estados_Unidos=df.loc[df['l1'] == 'Estados Unidos']
Brasil=df.loc[df['l1'] == 'Brasil']


# In[11]:


#Argentina dispropoortinately has the largest share of the data then Uruguay, the US and Brasil in that order
print(Argentina.shape,Uruguay.shape,Estados_Unidos.shape,Brasil.shape)


# In[12]:


#let's clean up argentina
#Argentina.isnull().sum()


# In[13]:


#def summary(dataframe):
 #   for i in dataframe.columns:
  #      if dataframe[i] != int:
    
    #return dataframe[i].mean()


# In[14]:


#time spent on the listing.
#data['start']=pd.to_datetime(data['start_date'])
#data['end']=pd.to_datetime(data['end_date'],errors='coerce')
#data['d']=data['end']-data['start']


# In[15]:


#data.d.mean()


# In[16]:


#select per country:
#data['Argentina']=data.loc[data['l1'] == 'Argentina']
#data['Argentina']=data.loc[data['l1'] == 'Argentina']
#data['l1'].to_string()


# In[17]:


#below we examine all the columns and clean them up ready for estimation
#df['l1'].unique() # There are four countries and 1267 cities in the data set. We'll have to trim down the data set into only the
#major cities in each country


# In[18]:


#df.isnull().sum() # This code tells us what are the number of NaN values in every column of the data set.


# In[19]:


#df.currency.unique()


# In[20]:


# out of the variables left in our reduced data set, we only select the numerical ones for use in the estimation procedure
Argentina_=Argentina[['rooms','bedrooms','bathrooms','surface_total','surface_covered','price']]
Braz=Brasil[['rooms','bedrooms','bathrooms','surface_total','surface_covered','price']]
Uru=Uruguay[['rooms','bedrooms','bathrooms','surface_total','surface_covered','price']]
US_=Estados_Unidos[['rooms','bedrooms','bathrooms','surface_total','surface_covered','price']]


# In[32]:


#Brasil.mean()


# In[21]:


#Fill in the nans with the means of the columns
#df.apply(lambda x: x.fillna(x.mean()))

#Brazil.apply(lambda x: x.fillna(x.mean(axis=0)))
#Uruguay.apply(lambda x: x.fillna(x.mean(axis=0)))
#US=US_.apply(lambda x: x.fillna(x.mean(axis=0)))
Argentina1=Argentina_.fillna(Argentina_.mean(axis=0))
Brazil=Braz.fillna(Braz.mean(axis=0))
Uruguay1=Uru.fillna(Uru.mean(axis=0))
US=US_.fillna(US_.mean(axis=0))


# In[22]:


#print(Brazil)
#print(US.head())
print(Uruguay1.head())


# In[23]:


#Argentina.fillna(Argentina.mean())


# In[27]:


Brazil.to_csv(r'C:\Users\Rodgers\Desktop\Machine learning\Brazil.csv', index = False)
Uruguay1.to_csv(r'C:\Users\Rodgers\Desktop\Machine learning\Uruguay.csv', index = False)
US.to_csv(r'C:\Users\Rodgers\Desktop\Machine learning\US.csv', index = False)
Argentina1.to_csv(r'C:\Users\Rodgers\Desktop\Machine learning\Argentina1.csv', index = False)


# In[25]:


#Argentina['l3'].nunique()


# In[26]:


#Argentina.tail(10)


# In[72]:


#OLS model for Brazil.
y=Brazil['price']
x=Brazil[['rooms', 'bathrooms','surface_total','surface_covered','bedrooms']]
x = sm.add_constant(x)
model=sm.OLS(y,x).fit()
model_prediction=model.predict(x)
model_details=model.summary()
print(model_details)


# In[62]:


#Linear regression for SK learn
reg = LinearRegression().fit(x, y) #pretty much the same coefficients
reg.score(x, y)
print(reg.coef_)
print(reg.intercept_)


# In[63]:


#from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
#coef_, which contains the coefficients #coef._[0] is the intercept
#intercept_, which contains the intercept
#sklearn‘s linear_model.LinearRegression comes with a .score() 
#method that returns the coefficient of determination R² of the prediction.


# In[64]:


#train test split sklearn
X_train,X_test,y_train,y_test=train_test_split(x,y, test_size=0.5, random_state=25)


# In[65]:


lr = LinearRegression()
lr.fit(X_train,y_train)
y_predict=lr.predict(X_test) # tells us the values that we could get in case in case we pass in new x values
lr_train_score=lr.score(X_train,y_train)
lr_test_score=lr.score(X_test,y_test)
print(lr_train_score)
print(lr_test_score)
print(lr.coef_)


# In[66]:


lr.fit(x,y)
print(lr.coef_)


# In[67]:


Brazil.corr()


# In[73]:


#OLS model for US
y1=US['price']
x1=US[['rooms', 'bathrooms','surface_total','surface_covered','bedrooms']]
x1 = sm.add_constant(x1)
model1=sm.OLS(y1,x1).fit()
model_prediction1=model1.predict(x1)
model_details1=model1.summary()
print(model_details1)


# In[ ]:




