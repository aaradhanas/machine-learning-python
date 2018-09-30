# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 10:37:46 2018

@author: AAS
"""

#https://www.kaggle.com/anandakshay44/linear-regression-analysis-of-imdb-dataset

import pandas as pd
import seaborn as sns

data = pd.read_csv('movie_metadata.csv')

# To check for nan values
data.isnull().any()
data.fillna(value=0, axis=1, inplace=True)

corr_value = data.corr()
sns.heatmap(data.corr(), vmax = 1, square=True)


#Defining features and target for this dataset based on co-relation
features = ['actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross',
       'num_voted_users', 'cast_total_facebook_likes', 'facenumber_in_poster',
       'num_user_for_reviews', 'budget', 'title_year',
       'actor_2_facebook_likes', 'aspect_ratio',
       'movie_facebook_likes']
target = ['imdb_score']

# Split data into training and test set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

#X_train = train[features].dropna()
X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

reg_score_train = regressor.score(X_train, y_train)
reg_score_test = regressor.score(X_test, y_test)

y_pred = regressor.predict(X_test);
