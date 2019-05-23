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
