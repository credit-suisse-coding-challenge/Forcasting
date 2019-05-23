# Forecasting the Market Value of a Treasury Portfolio

This project demonstrates viable methods for forecasting the future value (t+5) of a given portfolio.

Our repository has four different python files: ModelFunction, PythonModel, predictors, and utilities. PythonModel, predictors, and utitlities were all used primarily for data exploration and intitial model testing. These files are not formatted to be viewed as finished products. This is especially true for the PythonModel file which was just used to initially create different models without much forethought. It was then recreated in ModelFunction which we consider to be our final code product. 

ModelFunction contains different functions for each of the models we used and a function allows the user to input in a Pandas column, a training percent, and a file name. It then plots the original data split into training in testing data based on the training percent, as well as the predicted values given by each model. It then saves the predicted values into a csv file and prints the RMSE values for each different model. 

