# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:21:46 2018

@author: AAS
"""

# Multiple Linear Regression

import pandas as pd
import numpy as np

# IMPORTING DATASET

dataset = pd.read_csv('50_Startups.csv')
# Read data from all columns except last one
X =  dataset.iloc[:, :-1].values
# Read data from the last column
y = dataset.iloc[:, 4].values


# ENCODING CATEGORICAL DATA

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

# Replace the categories with integer categorical values
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder( categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding dummy variable trap by removing the first column (This is automatically done by the python regression library)
X = X[:, 1:]

# SPLITTING THE DATASET INTO TRAINING SET AND TEST SET

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling is taken care by the linear regression algorithm

# FITTING MULTIPLE LINEAR REGRESSION TO THE TRAINING SET

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# PREDICTING TEST SET RESULTS
y_pred = regressor.predict(X_test)

# BUILDING THE OPTIMAL MODEL USING BACKWARD ELIMINATION
import statsmodels.formula.api as sm
# Add the intercept (x0) column
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# PERFORM BACKWARD ELIMINATION
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

# Step 1
significance_level = 0.05
# Step 2
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Step 3
regressor_OLS.summary()
# Step 4 - x2 has the highest P-Value and is greater than SL
X_opt = X[:, [0, 1, 3, 4, 5]]
# Step 5 - Refit the model with the updated  X_opt
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()# Step 4 - x1 has the highest P-Value and is greater than SL
X_opt = X[:, [0, 3, 4, 5]]
# Step 5 - Refit the model with the updated  X_opt
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# Step 4 - x4 has the highest P-Value and is greater than SL
X_opt = X[:, [0, 3, 5]]
# Step 5 - Refit the model with the updated  X_opt
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Step 4 - x5 has the highest P-Value and is greater than SL
X_opt = X[:, [0, 3]]
# Step 5 - Refit the model with the updated  X_opt
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Predict test results using the optimal set of features
regressor.fit(X_train[:,[2]], y_train)
y_pred_new = regressor.predict(X_test[:,[2]])