import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as st
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt

# Import data
data = pd.read_csv("Treasury_Portfolio_No_Commas.csv")

# Converting Business.date to a numeric date
data['Timestamp'] = pd.to_datetime(data['Business_Date'])

# Seperate the first 64 objects as a training set and a testing set
train = data[0:63]
test = data[63:]

# Plotting data
'''
train.total_value.plot(label='Training Total')
test.total_value.plot(label='Testing Total')
data.Fixed_Income.plot(label='Fixed Income')
data.Deposits.plot(label='Deposits')
data.Other_liabilities.plot(label='Other liabilities')
data.Secured_Funding.plot(label='Secured Funding')
data.Equities.plot(label='Equities')
data.Unsecured_Funding.plot(label='Unsecured Funding')
plt.legend(loc='upper left')
plt.title("Initial plot from Treasury Portfolio")
plt.show()
'''

# Naive Approach
'''
dd = np.asarray(train.total_value)
y_hat = test.copy()
y_hat['naive'] = dd[len(dd) - 1]
train.total_value.plot(label='Train')
test.total_value.plot(label='Test')
y_hat.naive.plot(label='Naive Forecast')
plt.legend(loc='best')
plt.title('Naive method')
plt.show()
'''


# Holts Linear Trend model
'''
y_hat_avg = test.copy()
fit1 = st.tsa.api.Holt(np.asarray(train['total_value'])).fit(
    smoothing_level=0.3, smoothing_slope=0.1)
y_hat_avg['Holt_Linear'] = fit1.forecast(len(test))
train.total_value.plot(label='Training')
test.total_value.plot(label='Testing')
plt.plot(y_hat_avg['Holt_Linear'], label='Holt Linear')
y_hat.naive.plot(label='Naive Forecast')
plt.legend(loc='best')
plt.show()

rms_Holts = sqrt(mean_squared_error(test.total_value, y_hat_avg.Holt_Linear))
rms_naive = sqrt(mean_squared_error(test.total_value, y_hat.naive))
print("Holts Linear Trend ", rms_Holts)
print("Naive Approach ", rms_naive)
'''
# Moving Average
'''
y_hat_move_avg = test.copy()
y_hat_move_avg['moving_avg_forecast'] = train['total_value'].rolling(
    60).mean().iloc[-1]
plt.plot(train['total_value'], label='Train')
plt.plot(test['total_value'], label='Test')
plt.plot(y_hat_move_avg['moving_avg_forecast'],
         label='Moving Average Forecast')
plt.legend(loc='best')
plt.show()
rms_move_avg = sqrt(mean_squared_error(
    test.total_value, y_hat_move_avg.moving_avg_forecast))
print("Moving average RMS ", rms_move_avg)
'''

# ARIMA
y_hat_avg_ARIMA = test.copy()
fit2 = sm.tsa.statespace.SARIMAX(train.total_value).fit()
y_hat_avg_ARIMA['SARIMA'] = fit2.predict(start=64, end=80, dynamic=True)
train.total_value.plot(label='Train')
test.total_value.plot(label='Test')
plt.plot(y_hat_avg_ARIMA['SARIMA'], label='SARIMA')
plt.plot(y_hat_avg['Holt_Linear'], label='Holt Linear')
plt.plot(y_hat_move_avg['moving_avg_forecast'],
         label='Moving Average Forecast')
y_hat.naive.plot(label='Naive Forecast')
plt.legend(loc='best')
plt.show()

rms_move_avg = sqrt(mean_squared_error(
    test.total_value, y_hat_move_avg.moving_avg_forecast))

#rms_ARIMA = sqrt(mean_squared_error(test.total_value, y_hat_avg_ARIMA.SARIMA))
rms_Holts = sqrt(mean_squared_error(test.total_value, y_hat_avg.Holt_Linear))
rms_naive = sqrt(mean_squared_error(test.total_value, y_hat.naive))
print("Holts Linear Trend ", rms_Holts)
print("Naive Approach ", rms_naive)
print("Moving average RMS ", rms_move_avg)
#print("ARIMA Model RMS ", rms_ARIMA)
