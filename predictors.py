from utilities import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    # Import data
    data = pd.read_csv("Treasury_Portfolio_No_Commas.csv")

    # Converting Business.date to a numeric date
    data['Timestamp'] = pd.to_datetime(data['Business_Date'])

    # Seperate the first 64 objects as a training set and a testing set
    train = data[0:63]
    test = data[63:]


    headers = ['total_value','Fixed_Income','Deposits','Other_liabilities','Secured_Funding','Unsecured_Funding','Equities']
    from scipy import stats
    lag = 1
    for lag in range(1,5):
        print("LAG: " + str(lag))
        for header in headers:
            lagged = header + "_lag_" + str(lag)
            data[lagged] = data[header].shift(lag)
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.asarray(data[lagged])[lag:],np.asarray(data[header])[lag:])
            # print(header,slope, intercept, r_value, p_value, std_err,sep = " , ")
            # print(header,r_value**2,sep = " : ")
            print(header + " --- slope: " + str(slope) + "  intercept: " + str(intercept))
        print()
