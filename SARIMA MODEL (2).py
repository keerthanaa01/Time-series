#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv(r'C:\Users\User\Downloads\hd.csv')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.columns=["Date","Price"]
df.head()


# In[7]:


df['Date']=pd.to_datetime(df['Date'])


# In[8]:


df.head()


# In[9]:


df.set_index('Date',inplace=True)


# In[10]:


df.head()


# In[11]:


df.describe()


# In[12]:


a=df.plot()
a


# In[12]:


from statsmodels.tsa.stattools import adfuller


# In[13]:


test_result=adfuller(df['Price'])


# In[14]:


def adfuller_test(Price):
    result=adfuller(Price)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary")


# In[15]:


adfuller_test(df['Price'])


# In[16]:


df['Price First Difference'] = df['Price'] - df['Price'].shift(1)


# In[17]:


df['Price'].shift(1)


# In[18]:


df['Price First Difference']=df['Price']-df['Price'].shift(12)


# In[19]:


df.head(15)


# In[20]:


adfuller_test(df['Price First Difference'].dropna())


# In[21]:


df['Price First Difference'].plot()


# In[22]:


import pandas


# In[23]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a pandas DataFrame called df with a column 'Price'
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(df['Price'])
plt.show()


# In[24]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[25]:


import matplotlib.pyplot as plt
import statsmodels.api as sm

# Assuming df['Seasonal First Difference'] is your time series data
fig = plt.figure(figsize=(12, 8))

# Autocorrelation Plot
ax1 = fig.add_subplot(211)
plot_acf(df['Price First Difference'].iloc[13:], lags=20, ax=ax1)

# Partial Autocorrelation Plot
ax2 = fig.add_subplot(212)
plot_pacf(df['Price First Difference'].iloc[13:], lags=20, ax=ax2)

plt.show()


# In[26]:


from statsmodels.tsa.arima.model import ARIMA


# In[27]:


model=ARIMA(df['Price'],order=(2,1,2))
model_fit=model.fit()


# In[28]:


model_fit.summary()


# In[29]:


df['forecast']=model_fit.predict(start=900,end=1200,dynamic=True)
df[['Price','forecast']].plot(figsize=(12,8))


# In[48]:


import statsmodels.api as sm


# In[49]:


model=sm.tsa.statespace.SARIMAX(df['Price'],order=(2, 1, 2),seasonal_order=(2,1,2,12))
results=model.fit()


# In[50]:


df['forecast']=results.predict(start=900,end=1200,dynamic=True)
df[['Price','forecast']].plot(figsize=(12,8))


# In[51]:


from pandas.tseries.offsets import DateOffset


# In[52]:


import pandas as pd

# Assuming df.index[-1] is the last date in your DataFrame index
last_date = df.index[-1]

# Generate future dates with an offset of 0 to 23 months from the last date
future_dates = pd.date_range(start=last_date, periods=24, freq='MS')

# If you want to create a DataFrame with the future dates, you can do:
future_df = pd.DataFrame(index=future_dates, columns=df.columns)


# In[53]:


future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)


# In[54]:


future_datest_df.tail()


# In[55]:


future_df=pd.concat([df,future_datest_df])


# In[67]:


future_df['forecast'] = results.predict(start = 900, end = 2700, dynamic= True)  
future_df[['Price', 'forecast']].plot(figsize=(12, 8)) 


# In[ ]:




