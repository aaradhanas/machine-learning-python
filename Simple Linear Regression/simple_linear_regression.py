# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 09:52:35 2018

@author: AAS
"""

# Mathematical tools
import numpy as np
# Plotting data
import matplotlib.pyplot as plt
# Import and manage datasets
import pandas as pd

# IMPORTING DATASET

dataset = pd.read_csv('Salary_Data.csv')
# Read data from all columns except last one
X =  dataset.iloc[:, :-1].values
# Read data from the last column
y = dataset.iloc[:, 1].values

# SPLITTING THE DATASET INTO TRAINING SET AND TEST SET

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature scaling is taken care by the simple linear regression alogrithm

# Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test results
y_pred = regressor.predict(X_test)