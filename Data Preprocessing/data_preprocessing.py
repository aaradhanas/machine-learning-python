# -*- coding: utf-8 -*-
"""
Data Preprocessing

@author: AAS
"""

# IMPORTING LIBRARIES

# Mathematical tools
import numpy as np
# Plotting data
import matplotlib.pyplot as plt
# Import and manage datasets
import pandas as pd

# IMPORTING DATASET

dataset = pd.read_csv('Data.csv')
# Read data from all columns except last one
X =  dataset.iloc[:, :-1].values
# Read data from the last column
y = dataset.iloc[:, 3].values

# HANDLE MISSING DATA

# Imputation transformer for completing missing values
from sklearn.preprocessing import Imputer
imputer = Imputer( missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
# Replaces the NaN values with valid values using the mentioned strategy (here, it is mean)
X[:, 1:3] =  imputer.transform(X[:, 1:3])

"""from sklearn.preprocessing import CategoricalImputer
imputer = Imputer( missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, 0])
X[:, 0] =  imputer.transform(X[:, 0])"""

# ENCODING CATEGORICAL DATA

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

# Replace the categories with integer categorical values
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# France = 1 0 0
# Germany = 0 1 0
# Spain = 0 0 1
onehotencoder = OneHotEncoder( categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)