import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data
data = pd.read_csv("Treasury_Portfolio_No_Commas.csv")

# Converting Business.date to a numeric date
data['Timestamp'] = pd.to_datetime(data['Business_Date'])

# Seperate the first 64 objects as a training set and a testing set
train = data[0:63]
test = data[63:]

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
    col = data[col_name]
    label = 'SMA_' + str(n)
    sma = np.empty([len(col),1])

    for i in range(len(col)):
        if i <= n:
            sma[i] = np.NaN
        else:
            sum=0
            for j in range(1,n+1):
                sum+=col[i-j]
            sma[i] = sum/n
    data[label] = sma

#Function for adding EMA column to dataframe, n is length of EMA, alpha is smoothing factor
def add_EMA(data,col_name,n,alpha):
    col = data[col_name]
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
    ErrorSum = 0
    for i in range(len(pred)):
        error = (obs[i] - pred[i])**2
        ErrorSum += error
    return ErrorSum/len(pred)





#testing
# add_asset_allocation(data,True)
# add_SMA(data,'total_value',5)
# add_EMA(data,'total_value',5,0.55)
# add_SMA(data,'total_value',10)
# add_SMA(data,'total_value',15)
# train.total_value.plot(label='Training Total')
# test.total_value.plot(label='Testing Total')
# data.total_value.plot(label='total_value')
# data.SMA_5.plot(label='SMA5')
# data.EMA_5.plot(label='EMA5')
# data.SMA_10.plot(label='SMA10')
# data.SMA_15.plot(label='SMA15')
# plt.legend(loc='upper left')
# plt.title("Treasury Portfolio")
# plt.show()
