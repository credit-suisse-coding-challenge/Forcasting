import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as st
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt, floor

data = pd.read_csv("Treasury_Portfolio_No_Commas.csv")


def standard_tests(column, train_percent):
    '''
    This runs a standard set of time series predictive analysis
    on a column of data. It uses the train_percent to determine how
    much data should be used to train, with the rest being used to
    test the validity of the models. It creates a plot of the data 
    as well as the predicted values. It then prints the RMSA values for 
    the different tests.
    '''
    train_num = floor(train_percent * len(column))
    train = column[0:train_num]
    test = column[train_num:]

    plt.plot(train, label='Train')
    plt.plot(test, label='Test')

    rmsa_naive = naive_approach(train, test)
    rmsa_holts = holts_linear_model(train, test)
    rmsa_moving = moving_avg_model(train, test)
    rmsa_ARIMA = ARIMA_model(train, test)
    plt.title("Assorted Models and Their Predictions")
    plt.legend(loc='best')
    plt.show()

    print("Naive Model RMSA: %.4g" % rmsa_naive)
    print("Moving Average RMSA: %.4g" % rmsa_moving)
    print("Holts Linear Trend RMSA: %.4g" % rmsa_holts)
    print("Autoregressive Integrated Moving Average(ARIMA): %.4g" % rmsa_ARIMA)


def ARIMA_model(train, test):
    '''
    This uses the Autoregressive Integrated Moving Average model 
    to predict the next five values. It returns the RMSA associated 
    with this model on the given data.
    '''
    y_hat = test.copy()
    fit = sm.tsa.statespace.SARIMAX(train).fit()
    y_hat = fit.predict(start=len(train) + 1, end=len(
        train) + len(test), dynamic=True)
    plt.plot(range(len(train), len(train) + len(test)), y_hat, label="ARIMA")
    return compute_RMSA(test, y_hat)


def moving_avg_model(train, test):
    '''
    This function plots the moving average model given a train and test set to work
    on. 
    Returns: The RMSA value assocated with the given model on the given data
    '''
    y_hat = test.copy()
    moving_avg = train.rolling(60).mean().iloc[-1]
    y_hat = [moving_avg] * len(test)
    plt.plot(range(len(train), len(train) + len(test)),
             y_hat, label='Moving Average Forecast')
    return compute_RMSA(test, y_hat)


def naive_approach(train, test):
    '''
    This function computes the naive approach model. It plots the 
    predicted values and returns the RMSA value associated with this 
    model
    Returns: The RMSA value assocated with the given model and data
    '''
    dd = np.asarray(train)
    y_hat = test.copy()
    avg_naive = dd[len(dd) - 1]
    y_hat = [avg_naive] * len(test)
    plt.plot(range(len(train), len(train) + len(test)),
             y_hat, label='Naive Forecast')
    return compute_RMSA(test, y_hat)


def holts_linear_model(train, test):
    '''
    This uses the Holts Linear model to predict the next 5 days. 
    Returns: The RMSA of the model with the given data
    '''
    y_hat = test.copy()
    fit1 = st.tsa.api.Holt(np.asarray(train)).fit(
        smoothing_level=0.3, smoothing_slope=0.1)
    y_hat = fit1.forecast(len(test))
    plt.plot(range(len(train), len(train) + len(test)),
             y_hat, label='Holt Linear')
    return compute_RMSA(test, y_hat)


def compute_RMSA(test, computed):
    '''
    This function computs a the RMSA value given a test column
    and the predicted output
    '''
    return sqrt(mean_squared_error(test, computed))


standard_tests(data.total_value, 0.8)
