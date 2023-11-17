#!/usr/bin/env python
# coding: utf-8

# In[28]:


pip install pmdarima


# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[30]:


df = pd.read_csv(r'C:\Users\User\Downloads\hdfc.csv')
print('shape of data',df.shape)
df.head()


# In[31]:


# Convert the 'date' column to datetime format
df['Date'] = pd.to_datetime(data['Date'])

# Set the 'date' column as the index
df.set_index('Date', inplace=True)


# In[32]:


df['Price'].plot(figsize=(12,6))


# In[33]:


from statsmodels.tsa.stattools import adfuller
def ad_test(dataset):
    dftest = adfuller (dataset, autolag = 'AIC')
    print("ADF",dftest[0])
    print("p value",dftest[1])
    print("lags",dftest[2])
    print("no of observations used for adf regression and critical value calculation",dftest[3])
    print("critical values")
    for key,val in dftest[4].items():
        print("\t",key,":",val)


# In[34]:


ad_test(df['Price'])


# In[35]:


from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')


# In[36]:


stepwise_fit=auto_arima(df['Price'],trace=True,suppress_warnings=True)
stepwise_fit.summary()


# In[37]:


from statsmodels.tsa.arima.model import ARIMA



# In[38]:


print(df.shape)
train=df.iloc[:-30]
test=df.iloc[-30:]
print(train.shape,test.shape)


# In[39]:


model = ARIMA(train['Price'], order=(2,1,2))
model = model.fit()
model.summary()


# In[40]:


start=len(train)
end=len(train)+len(test)-1
pred=model.predict(start=start,end=end,type='levels')
pred.index=df.index[start:end+1]
print(pred)


# In[41]:


pred.plot(legend=True)
test['Price'].plot(legend=True)


# In[42]:


test['Price'].mean()


# In[43]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(pred,test['Price']))
print(rmse)


# In[44]:


model2 = ARIMA(train['Price'], order=(2,1,2))
model2 = model2.fit()
df.tail()


# In[45]:


index_future_date=pd.date_range(start='2024-12-01',end='2024-12-31')
pred=model2.predict(start=len(df),end=len(df)+30,type='levels').rename('arima prediction')
pred.index=index_future_date
print(pred)


# In[46]:


pred.plot(figsize=(12,5),legend=True)


# In[47]:


print(len(test['Price']), len(pred))


# In[48]:


# Adjust prediction index to match test data length
index_future_date = pd.date_range(start=test.index[0], periods=len(test), freq='B')
pred = pred[:len(test)]  # Trim the predictions to match the test data length
pred.index = index_future_date



# In[49]:


# Calculate evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score




# In[50]:


# Mean Absolute Error (MAE)
mae = mean_absolute_error(test['Price'], pred)
print(f"Mean Absolute Error (MAE): {mae}")



# In[51]:


# Root Mean Squared Error (RMSE)
rmse = sqrt(mean_squared_error(test['Price'], pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")



# In[52]:


# R-squared (coefficient of determination)
r2 = r2_score(test['Price'], pred)
print(f"R-squared (R2): {r2}")


# In[ ]:





# In[ ]:




