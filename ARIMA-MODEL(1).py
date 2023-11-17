#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA


# In[2]:


# Load your dataset
data = pd.read_csv(r'C:\Users\User\Downloads\hdfc.csv')




# In[3]:


# Convert the 'date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set the 'date' column as the index
data.set_index('Date', inplace=True)



# In[4]:


data = data['Price']
data.describe()


# In[5]:


# Select the 'price' column for analysis
#price_data = data['Price']


# In[6]:


# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title('Stock Price Time Series')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[7]:


data.info()


# In[8]:


# Calculate rolling mean and rolling standard deviation
rolling_mean = data.rolling(window=12).mean()
rolling_std = data.rolling(window=12).std()

# Plot original data, rolling mean, and rolling standard deviation
plt.figure(figsize=(12, 6))
plt.plot(data, label='Original Data')
plt.plot(rolling_mean, label='Rolling Mean')
plt.plot(rolling_std, label='Rolling Std')
plt.title('Rolling Mean and Standard Deviation')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[9]:


from statsmodels.tsa.stattools import adfuller

# Perform Augmented Dickey-Fuller test
result = adfuller(data, autolag='AIC')

# Print the test statistic and p-value
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Check the p-value
if result[1] <= 0.05:
    print('The time series is stationary.')
else:
    print('The time series is non-stationary.')


# In[10]:


import numpy as np


# In[11]:


# Log transformation
stationary_data = np.log(data)


# In[12]:


plt.subplot(2, 1, 2)
plt.plot(stationary_data)
plt.title('Log tansform data')
plt.xlabel('Index')
plt.ylabel('Log(Price)')


# In[13]:


plt.subplot(2, 1, 1)
plt.plot(data)
plt.title('Original Data')
plt.xlabel('Index')
plt.ylabel('Price')


# In[14]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[15]:


# Decompose the time series into its components
result = seasonal_decompose(stationary_data, model='multiplicative', period=160)

# Plot the original time series, trend, seasonal, and residual components
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(data, label='Original')
plt.legend(loc='upper left')
plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(4, 1, 4)
plt.plot(result.resid, label='Residual')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# In[16]:


differenced_data = data.diff(periods=1).dropna()


# In[17]:


plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(data)
plt.title('Original Time Series')
plt.xlabel('Date')
plt.ylabel('Price')

plt.subplot(2, 1, 2)
plt.plot(differenced_data)
plt.title('Differenced Time Series')
plt.xlabel('Date')
plt.ylabel('Differenced Price')

plt.tight_layout()
plt.show()


# In[18]:


# Calculate rolling mean and rolling standard deviation
rolling_mean = differenced_data.rolling(window=12).mean()
rolling_std = differenced_data.rolling(window=12).std()

# Plot original data, rolling mean, and rolling standard deviation
plt.figure(figsize=(12, 6))
plt.plot(differenced_data, label='Original Data')
plt.plot(rolling_mean, label='Rolling Mean')
plt.plot(rolling_std, label='Rolling Std')
plt.title('Rolling Mean and Standard Deviation')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[19]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Plot autocorrelation
plt.figure(figsize=(12, 6))
plot_acf(differenced_data, lags=20, title='Autocorrelation')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

# Plot partial autocorrelation
plt.figure(figsize=(12, 6))
plot_pacf(differenced_data, lags=20, title='Partial Autocorrelation')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()


# In[20]:


from statsmodels.tsa.arima.model import ARIMA


# In[21]:


# Fit an ARIMA model to the training data
# You may need to tune the order parameter for better results
order = (1, 1, 2)  # Example order, you can adjust it
model = ARIMA(differenced_data, order=order)
result_ARIMA = model.fit()
plt.figure(figsize=(16,8))
plt.plot(differenced_data)
plt.plot(result_ARIMA.fittedvalues,color='red')



# In[22]:


ARIMA_diff_predictions = pd.Series(result_ARIMA.fittedvalues,copy=True)


# In[23]:


print(ARIMA_diff_predictions.head())


# In[24]:


ARIMA_diff_predictions_cumsum = ARIMA_diff_predictions.cumsum()
print(ARIMA_diff_predictions_cumsum.head())


# In[25]:


ARIMA_log_prediction = pd.Series(stationary_data.iloc[0],index=stationary_data.index)
ARIMA_log_prediction = ARIMA_log_prediction.add(ARIMA_diff_predictions_cumsum,fill_value=0)
ARIMA_log_prediction.head()


# In[26]:


# Inverse the cumulative sum to obtain differenced values
ARIMA_diff_predictions = ARIMA_diff_predictions_cumsum - ARIMA_diff_predictions_cumsum.shift(1)
ARIMA_diff_predictions = ARIMA_diff_predictions.fillna(0)

# Inverse the differencing to obtain stationary log-transformed values
ARIMA_log_prediction = stationary_data.shift(1)
ARIMA_log_prediction[0] = stationary_data[0]  # Fill the first NaN value with the original value
ARIMA_log_prediction = ARIMA_log_prediction.add(ARIMA_diff_predictions, fill_value=0)



# Print the first few rows of the final predictions
print(ARIMA_log_prediction.head())


# In[27]:


# Convert the log-transformed values back to the original scale
ARIMA_final_prediction = np.exp(ARIMA_log_prediction)
print(ARIMA_final_prediction.head())


# In[ ]:





# In[28]:


pip install pmdarima


# In[29]:


import pmdarima as pm
def arimamodel(timeseries):
    automodel=pm.auto_arima(timeseries,
                          start_p=3,
                            start_q=3,
                            max_p=5,
                            max_q=5,
                            test="adf",
                            seasonal =True,
                            trace=True )
    return automodel


# In[30]:


arimamodel(stationary_data)

