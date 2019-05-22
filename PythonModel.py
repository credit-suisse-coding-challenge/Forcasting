import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as st

# Import data
data = pd.read_csv("Treasury_Portfolio_No_Commas.csv")

# Converting Business.date to a numeric date
data['Timestamp'] = pd.to_datetime(data['Business_Date'])

# Seperate the first 64 objects as a training set and a testing set
train = data[0:63]
test = data[63:]

# Plotting data
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
