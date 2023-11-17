#!/usr/bin/env python
# coding: utf-8

# In[1]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from tqdm import tqdm_notebook
from itertools import product

import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv(r'C:\Users\User\Downloads\hdfc.csv')
print(df.shape)  # (123, 8)
df.head()
     


# In[3]:


import matplotlib.pyplot as plt

# Assuming your DataFrame is named stock_data and columns are Date, Price, Open, High, Low, Volume, Change(%)
fig, axes = plt.subplots(nrows=4, ncols=2, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    if i == 0:
        data = df['Date']
        ax.plot(data, color='red', linewidth=1)
        ax.set_title('Date')
    elif i == 1:
        data = df['Price']
        ax.plot(data, color='green', linewidth=1)
        ax.set_title('Price')
    elif i == 2:
        data = df['Open']
        ax.plot(data, color='blue', linewidth=1)
        ax.set_title('Open')
    elif i == 3:
        data = df['High']
        ax.plot(data, color='orange', linewidth=1)
        ax.set_title('High')
    elif i == 4:
        data = df['Low']
        ax.plot(data, color='purple', linewidth=1)
        ax.set_title('Low')
    elif i == 5:
        data = df['Volume']
        ax.plot(data, color='brown', linewidth=1)
        ax.set_title('Volume')
    elif i == 6:
        data = df['Change(%)']
        ax.plot(data, color='pink', linewidth=1)
        ax.set_title('Change(%)')
    
    # Decorations
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()
plt.show()


# In[4]:


ad_fuller_result_1 = adfuller(df['High'].diff()[1:])

print('realgdp')
print(f'ADF Statistic: {ad_fuller_result_1[0]}')
print(f'p-value: {ad_fuller_result_1[1]}')

print('\n---------------------\n')

ad_fuller_result_2 = adfuller(df['Low'].diff()[1:])

print('realcons')
print(f'ADF Statistic: {ad_fuller_result_2[0]}')
print(f'p-value: {ad_fuller_result_2[1]}')


# In[5]:


print('High causes Low?\n')
print('------------------')
granger_1 = grangercausalitytests(df[['Low', 'High']], 4)

print('\Low causes High?\n')
print('------------------')
granger_2 = grangercausalitytests(df[['High', 'Low']], 4)
     


# In[6]:


df = df[['Low','High']]
print(df.shape)
     


# In[7]:


train_df=df[:-1000]
test_df=df[-1000:]


# In[8]:


print(train_df.shape)


# In[9]:


print(test_df.shape)


# In[10]:


model = VAR(train_df.diff()[1:])


# In[11]:


sorted_order=model.select_order(maxlags=20)
print(sorted_order.summary())


# In[12]:


var_model = VARMAX(train_df, order=(8,0),enforce_stationarity= True)
fitted_model = var_model.fit(disp=False)
print(fitted_model.summary())
     


# In[13]:


n_forecast = 1000
predict = fitted_model.get_prediction(start=len(train_df),end=len(train_df) + n_forecast-1)
predictions=predict.predicted_mean
     


# In[14]:


predictions.columns=['Low_predicted','High_predicted']
predictions
     


# In[15]:


test_vs_pred=pd.concat([test_df,predictions],axis=1)



# In[16]:


test_vs_pred.plot(figsize=(12,5))


# In[17]:


from sklearn.metrics import mean_squared_error
import math 
from statistics import mean

rmse_low = math.sqrt(mean_squared_error(predictions['Low_predicted'], test_df['Low']))
print('Mean value of Low is: {}. Root Mean Squared Error is: {}'.format(mean(test_df['Low']), rmse_low))

rmse_high = math.sqrt(mean_squared_error(predictions['High_predicted'], test_df['High']))
print('Mean value of High is: {}. Root Mean Squared Error is: {}'.format(mean(test_df['High']), rmse_high))

