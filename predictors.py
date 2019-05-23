from utilities import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def SMA_predict(train,test,col_name,n):
    add_SMA(train,col_name,n)
    col = np.asarray(train[col_name])
    for i in range(len(train),len(train) + len(test)):
        col = np.append(col,np.mean(col[i-n:i]))

    return col

def SMA_predict_future(data,col_name,n):
    col = np.asarray(data[col_name])
    for i in range(len(data),len(data) + 5):
        col = np.append(col,np.mean(col[i-n:i]))

    return col

def EMA_predict(train,test,col_name,n,alpha):
    add_EMA(train,col_name,n,alpha)
    col = np.asarray(train[col_name])

    for i in range(len(train),len(train) + len(test)):
        wsum=0
        for j in range(1,n+1):
            if j == n:
                wsum+= ((1-alpha)**j)*col[i-j]
            else:
                wsum+= alpha*((1-alpha)**(j-1))*col[i-j]
        col = np.append(col,wsum)

    return col

def EMA_predict_future(data,col_name,n,alpha):
    col = np.asarray(data[col_name])

    for i in range(len(train),len(train) + 5):
        wsum=0
        for j in range(1,n+1):
            if j == n:
                wsum+= ((1-alpha)**j)*col[i-j]
            else:
                wsum+= alpha*((1-alpha)**(j-1))*col[i-j]
        col = np.append(col,wsum)

    return col






if __name__ == '__main__':
    # Import data
    data = pd.read_csv("Treasury_Portfolio_No_Commas.csv")

    # Converting Business.date to a numeric date
    data['Timestamp'] = pd.to_datetime(data['Business_Date'])

    # Seperate the first 64 objects as a training set and a testing set

    name = 'total_value'
    test = data[63:].copy()

    train = data[0:63].copy()
    sma = SMA_predict(train,test,name,5)
    data['predict_SMA'] = sma
    print(RMSE(data['predict_SMA'][63:],test[name]))

    train = data[0:63].copy()
    ema = EMA_predict(train,test,name,5,0.4)
    data['predict_EMA'] = ema
    print(RMSE(data['predict_EMA'][63:],test[name]))

    future = data.copy()
    ema = EMA_predict_future(future,name,5,0.75)
    print(ema[80:])

    future = data.copy()
    sma = SMA_predict_future(future,name,5)
    print(sma[80:])


    train[name].plot(label='Training Total')
    test[name].plot(label='Testing Total')
    data.predict_SMA.plot(label='predict_SMA')
    data.predict_EMA.plot(label='predict_EMA')
    plt.legend(loc='upper left')
    plt.title("SMA Predict")
    plt.show()


    # headers = ['total_value','Fixed_Income','Deposits','Other_liabilities','Secured_Funding','Unsecured_Funding','Equities']
    # from scipy import stats
    # lag = 1
    # for lag in range(1,5):
    #     print("LAG: " + str(lag))
    #     for header in headers:
    #         lagged = header + "_lag_" + str(lag)
    #         data[lagged] = data[header].shift(lag)
    #         slope, intercept, r_value, p_value, std_err = stats.linregress(np.asarray(data[lagged])[lag:],np.asarray(data[header])[lag:])
    #         # print(header,slope, intercept, r_value, p_value, std_err,sep = " , ")
    #         # print(header,r_value**2,sep = " : ")
    #         print(header + " --- slope: " + str(slope) + "  intercept: " + str(intercept))
    #     print()
