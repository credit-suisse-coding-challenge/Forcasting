import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as st
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt, floor


data = pd.read_csv("Treasury_Portfolio_No_Commas.csv")


def standard_tests(column, train_percent, file_name):
    '''
    This runs a standard set of time series predictive analysis
    on a column of data. It uses the train_percent to determine how
    much data should be used to train, with the rest being used to
    test the validity of the models. It creates a plot of the data
    as well as the predicted values. It then prints the RMSA values for
    the different tests. The function also takes in a file name which 
    it then uses to create a csv with the predicted data
    '''
    train_num = floor(train_percent * len(column))
    train = column[0:train_num]
    test = column[train_num:]
    predicted = pd.DataFrame()

    plt.plot(train, label='Train')
    plt.plot(test, label='Test')

    rmse_naive, naive_predictions = naive_approach(train, test)
    rmse_holts, holts_predictions = holts_linear_model(train, test)
    rmse_moving, moving_predictions = moving_avg_model(train, test)
    rmse_ARIMA, ARIMA_predictions = ARIMA_model(train, test)

    predicted = pd.DataFrame(
        {'Naive': naive_predictions,
         'Holts': holts_predictions,
         'Moving_Average': moving_predictions,
         'ARIMA': ARIMA_predictions},
        columns=['Naive', 'Holts', 'Moving_Average', 'ARIMA']
    )

    plt.title("Assorted Models and Their Predictions")
    plt.legend(loc='best')
    plt.show()

    print("Naive Model RMSE: %.4g" % rmse_naive)
    print("Moving Average RMSE: %.4g" % rmse_moving)
    print("Holts Linear Trend RMSE: %.4g" % rmse_holts)
    print("Autoregressive Integrated Moving Average(ARIMA) RMSE: %.4g" % rmse_ARIMA)
    predicted.to_csv("Predictions/" + file_name + ".csv")
    return predicted


def ARIMA_model(train, test):
    '''
    This uses the Autoregressive Integrated Moving Average model
    to predict the next five values. 
    Returns: the RMSE and the predicted values for the next 5 days.
    '''
    y_hat = test.copy()
    fit = sm.tsa.statespace.SARIMAX(train).fit()
    y_hat = fit.predict(start=len(train) + 1, end=len(
        train) + len(test) + 5, dynamic=True)
    plt.plot(range(len(train), len(train) + len(test) + 5),
             y_hat, label="ARIMA")
    return compute_RMSE(test, y_hat[:-5]), y_hat[:-5]


def moving_avg_model(train, test):
    '''
    This function plots the moving average model given a train and test set to work
    on.
    Returns: The RMSE value assocated with the given model on the given data
    '''
    y_hat = test.copy()
    moving_avg = train.rolling(60).mean().iloc[-1]
    y_hat = [moving_avg] * (len(test) + 5)
    plt.plot(range(len(train), len(train) + len(test) + 5),
             y_hat, label='Moving Average Forecast')
    return compute_RMSE(test, y_hat[:-5]), y_hat[:-5]


def naive_approach(train, test):
    '''
    This function computes the naive approach model. It plots the
    predicted values and returns the RMSA value associated with this
    model.
    Returns: The RMSA value assocated with the given model and data
    '''
    dd = np.asarray(train)
    y_hat = test.copy()
    avg_naive = dd[len(dd) - 1]
    y_hat = [avg_naive] * (len(test) + 5)
    plt.plot(range(len(train), len(train) + len(test) + 5),
             y_hat, label='Naive Forecast')
    return compute_RMSE(test, y_hat[:-5]), y_hat[:-5]


def holts_linear_model(train, test):
    '''
    This uses the Holts Linear model to predict the next 5 days.
    Returns: The RMSA of the model with the given data
    '''
    y_hat = test.copy()
    fit1 = st.tsa.api.Holt(np.asarray(train)).fit(
        smoothing_level=0.3, smoothing_slope=0.1)
    y_hat = fit1.forecast(len(test) + 5)
    plt.plot(range(len(train), len(train) + len(test) + 5),
             y_hat, label='Holt Linear')
    return compute_RMSE(test, y_hat[:-5]), y_hat[:-5]


def compute_RMSE(test, computed):
    '''
    This function computs a the Root Mean Squared Error (RMSE) value
    given a test column and the predicted output
    '''
    return sqrt(mean_squared_error(test, computed))


standard_tests(data.total_value, 0.8, "predicted_total_value")
