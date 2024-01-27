#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


from sklearn.ensemble import RandomForestRegressor


# In[3]:


from sklearn import metrics


# In[5]:


df = pd.read_csv("gld_price_data.csv")
df.head()


# In[12]:


df.tail()


# In[14]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# In[11]:


df.dtypes


# In[15]:


#we can use the .info() to get the the dtypes and if there are nullable or not
df.info()


# In[16]:


#let's see if their is a positive or negative correlation beween the columns of the dataset 
#we can say that a positive correlation => is directly proportional
#we can say that a negative correlation => is indirectly proportional
#to see this correlation we construct a heatmap 


# In[22]:


corrl = df.corr()


# In[27]:


plt.figure(figsize=(8,8))
sns.heatmap(corrl, cbar =True, square =True, fmt='.1f', annot=True, annot_kws ={'size' : 8}, cmap="icefire_r")


# In[28]:


#we can conclude from this heatmap above thte GLD column is positevely corelated to the SLV column with +0.9
# also USO with EUR/USD at +0.8 , 
# spx and eur/usd has a negative correlation with -0.7
# and uso and spx have also a negative correlation at -0.6 


# In[29]:


print(corrl['GLD'])


# In[41]:


#let's see the distribution of the GLD 
sns.distplot(df['GLD'], color = 'yellow')


# In[44]:


#splitting the data
X = df.drop(['Date', 'GLD'], axis =1)
Y = df['GLD']


# In[45]:


print(X)


# In[46]:


print(Y)


# In[54]:


#Split and Test 
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size =0.2, random_state =2)


# In[55]:


reg = RandomForestRegressor(n_estimators =100)


# In[56]:


reg.fit(x_train,y_train)


# In[57]:


test_pred = reg.predict(x_test)


# In[58]:


print(test_pred)


# In[60]:


#model Evaluation

#R squared error 
s1 = metrics.r2_score(y_test, test_pred)
print("R squared error",s1)


# In[61]:


#let us compare the actual values vs the predicted values 


# In[62]:


y_test = list(y_test)


# In[64]:


plt.plot(y_test, color = 'blue', label ='Actual values')
plt.plot(test_pred, color = 'yellow', label ='predicted values')
plt.title('Actual values vs Predicted values')
plt.xlabel('Numbers of Actual Values')
plt.ylabel('Gold prices')
plt.legend()
plt.show()


# In[65]:



from sklearn.linear_model import LinearRegression


# In[66]:


model = LinearRegression()


# In[67]:


model.fit(x_train,y_train)


# In[69]:


y_pred = model.predict(x_test)


# In[70]:


#R squared error 
sl = metrics.r2_score(y_test, y_pred)
print("R squared error",sl)


# In[71]:


def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

lr = MAPE(y_test,y_pred)
rfr=  MAPE(y_test,test_pred)
print("Linear Regression: ", lr, "%")
print("Random Forrest Regressor: ", rfr, "%")


# In[72]:


# After testing out the Mean Absolute Percantage error of both models Linear Regression , Random Forest Regression
# we got really good results for testing errors in both models especially in Random Forrest Regressor 
#where we got really good results


# In[73]:


from xgboost import XGBRegressor


# In[74]:


m = XGBRegressor()


# In[75]:


m.fit(x_train,y_train)


# In[76]:


pred = m.predict(x_test)


# In[77]:


sx = metrics.r2_score(y_test,pred)
print("R squared error: ",sx)


# In[79]:


fx = MAPE(y_test,pred)
print("Mean Absolute Percantage Error: ", fx, "%")


# In[80]:


# after trying a boosting ensemble technique the MAPE is 1.187% these are great results
# but the bagging ensemble technique which is a random forest regressor algortithm performed better with a MAPE 1.065%


# In[81]:





# In[85]:





# In[ ]:




