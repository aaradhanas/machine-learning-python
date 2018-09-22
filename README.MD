# Machine learning using Python

Commonly used machine learning algorithms have been implemented in Python.

## Data Preprocessing

This is an important step to be done before training your data. Data preprocessing ensures your data is in the proper state to be fed into any machine learning algorithm. The following preprocessing steps need to be followed:

- Handling missing data
- Categorization of data
- Splitting the dataset into training set and test set
- Feature scaling

## Simple Linear Regression

Simple Linear Regression is an algorithm where there is one dependent variable and one independent variable. The machine learns the correlation between these variables and then predicts the value of the independent variable given the dependent variable. The mathematical equation goes like this:

<code> y = b0 + b1 * x1 </code>

where,  
    b0 is the constant,  
    b1 is the coefficient,  
    x1 is the independent variable &  
    y is the dependent variable

### Ordinary least squares

This is used to measure the amount of error between the real and predicted values. This error can be measured by taking the difference between the two values and squaring them. This is summed up for all the data samples.

<code>SUM (y - y^) ^ 2</code>

The goal of the alogrithm is to minimize this error.

### Assumptions of linear regression

- Linearity
- Homoscedasticity
- Multivariate normality
- Independence of errors
- Lack of multicollinearity