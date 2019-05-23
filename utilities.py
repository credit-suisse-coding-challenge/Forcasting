import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Function to add Asset Allocation fields to data frame and plot if desired
def add_asset_allocation(data, show):
    TV = data['total_value']

    FI = np.asarray(data['Fixed_Income'])
    D = np.asarray(data['Deposits'])
    OL = np.asarray(data['Other_liabilities'])
    SF = np.asarray(data['Secured_Funding'])
    UF = np.asarray(data['Unsecured_Funding'])
    E = np.asarray(data['Equities'])

    data['portion_FI'] = FI/TV
    data['portion_D'] = D/TV
    data['portion_OL'] = OL/TV
    data['portion_SF'] = SF/TV
    data['portion_UF'] = UF/TV
    data['portion_E'] = E/TV

    if show:
        data.portion_FI.plot(label='Fixed Income')
        data.portion_D.plot(label='Deposits')
        data.portion_OL.plot(label='Other liabilities')
        data.portion_SF.plot(label='Secured Funding')
        data.portion_E.plot(label='Equities')
        data.portion_UF.plot(label='Unsecured Funding')
        plt.legend(loc='upper left')
        plt.title("Treasury Portfolio Asset Allocation")
        plt.show()

#Function for adding SMA column to dataframe, n is length of SMA
def add_SMA(data,col_name,n):
    col = np.asarray(data[col_name])
    label = 'SMA_' + str(n)
    sma = np.empty([len(col),1])

    for i in range(len(col)):
        if i <= n:
            sma[i] = np.NaN
        else:
            sma[i] = np.mean(col[i-n:i])
    data[label] = sma

#Function for adding EMA column to dataframe, n is length of EMA, alpha is smoothing factor
def add_EMA(data,col_name,n,alpha):
    col = np.asarray(data[col_name])
    label = 'EMA_' + str(n)
    ema = np.empty([len(col),1])

    for i in range(len(col)):
        if i < n:
            ema[i] = np.NaN
        else:
            wsum=0
            for j in range(1,n+1):
                if j == n:
                    wsum+= ((1-alpha)**j)*col[i-j]
                else:
                    wsum+= alpha*((1-alpha)**(j-1))*col[i-j]
            ema[i] = wsum
    data[label] = ema

#Mean Squared Error between observed and predicted
def MSE(pred, obs):
    if len(obs) != len(pred):
        print("Error - inputs must be same length")
        return 0
    pred = np.asarray(pred)
    obs = np.asarray(obs)
    ErrorSum = 0
    for i in range(len(pred)):
        error = (obs[i] - pred[i])**2
        ErrorSum += error
    return ErrorSum/len(pred)

#Root Mean Squared Error between observed and predicted
def RMSE(pred, obs):
    from math import sqrt
    return sqrt(MSE(pred,obs))

def add_SMA_bands(data,col_name,n):
    add_SMA(data,col_name,n)

    col = np.asarray(data[col_name])
    sma = np.asarray(data['SMA_' + str(n)])
    label1 = 'SMA_Upper_' + str(n)
    label2 = 'SMA_Lower_' + str(n)

    smaU = np.empty([len(col),1])
    smaL = np.empty([len(col),1])

    for i in range(len(col)):
        if i <= n:
            smaU[i] = np.NaN
            smaL[i] = np.NaN
        else:
            sd = np.std(col[i-n:i])
            smaU[i] = sma[i] + 2*sd
            smaL[i] = sma[i] - 2*sd

    data[label1] = smaU
    data[label2] = smaL

def add_EMA_bands(data,col_name,n,alpha):
    add_EMA(data,col_name,n,alpha)

    col = np.asarray(data[col_name])
    ema = np.asarray(data['EMA_' + str(n)])
    label1 = 'EMA_Upper_' + str(n)
    label2 = 'EMA_Lower_' + str(n)

    emaU = np.empty([len(col),1])
    emaL = np.empty([len(col),1])

    for i in range(len(col)):
        if i <= n:
            emaU[i] = np.NaN
            emaL[i] = np.NaN
        else:
            sd = np.std(col[i-n:i])
            emaU[i] = ema[i] + 2*sd
            emaL[i] = ema[i] - 2*sd

    data[label1] = emaU
    data[label2] = emaL

def generate_auto_corr(data,show):
    headers = ['total_value','Fixed_Income','Deposits','Other_liabilities','Secured_Funding','Unsecured_Funding','Equities']

    nlags = 10
    corr = np.zeros([nlags,len(headers)])

    for i in range(0,nlags):
        j=0
        for col in headers:
            corr[i][j] = data[col].autocorr(lag = i)
            j+=1

    corr = pd.DataFrame(data=corr,columns=headers)

    if show:
        corr.total_value.plot(label = 'Total Value')
        corr.Fixed_Income.plot(label='Fixed Income')
        corr.Deposits.plot(label='Deposits')
        corr.Other_liabilities.plot(label='Other liabilities')
        corr.Secured_Funding.plot(label='Secured Funding')
        corr.Equities.plot(label='Equities')
        corr.Unsecured_Funding.plot(label='Unsecured Funding')

        plt.legend(loc='upper left')
        plt.title("Treasury Portfolio AutoCorrelations")
        plt.xlabel("Lag")
        plt.ylabel("ACF")
        plt.show()
    return corr

if __name__ == '__main__':
    # Import data
    data = pd.read_csv("Treasury_Portfolio_No_Commas.csv")

    # Converting Business.date to a numeric date
    data['Timestamp'] = pd.to_datetime(data['Business_Date'])

    # Seperate the first 64 objects as a training set and a testing set
    train = data[0:63]
    test = data[63:]

    generate_auto_corr(data,True)

    # Plotting data
    # train.total_value.plot(label='Training Total')
    # test.total_value.plot(label='Testing Total')
    # data.total_value.plot(label = 'Total Value')
    # data.Fixed_Income.plot(label='Fixed Income')
    # data.Deposits.plot(label='Deposits')
    # data.Other_liabilities.plot(label='Other liabilities')
    # data.Secured_Funding.plot(label='Secured Funding')
    # data.Equities.plot(label='Equities')
    # data.Unsecured_Funding.plot(label='Unsecured Funding')
    # plt.legend(loc='upper left')
    # plt.title("Treasury Portfolio Value Plot")
    # plt.show()

    # add_asset_allocation(data,True)
    # add_EMA(data,'Equities',5,0.7)
    # add_EMA_bands(data,'Equities',5,0.6)
    # add_EMA(data,'total_value',5,0.55)
    # add_SMA(data,'total_value',10)
    # add_SMA(data,'total_value',15)
    # train.total_value.plot(label='Training Total')
    # test.total_value.plot(label='Testing Total')
    # data.total_value.plot(label='total_value')
    # data.Equities.plot(label='Equities')
    # data.EMA_5.plot(label='EMA5')
    # data.EMA_Upper_5.plot(label='EMA_Upper_5')
    # data.EMA_Lower_5.plot(label='EMA_Lower_5')
    # data.EMA_5.plot(label='EMA5')
    # data.SMA_10.plot(label='SMA10')
    # data.SMA_15.plot(label='SMA15')
    # plt.legend(loc='upper left')
    # plt.title("Treasury Portfolio")
    # plt.show()
