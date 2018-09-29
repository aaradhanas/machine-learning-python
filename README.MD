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

## Multiple Linear Regression

Multiple Linear Regression is an algorithm where there is one dependent variable and multiple independent variables. The machine learns the correlation between these variables and then predicts the value of the independent variable given the dependent variables. The mathematical equation goes like this:

<code> y = b0 + b1 * x1 + b2 * x2 + b3 * x3 + ...... + bn * xn </code>

where,  
    b0 is the constant,  
    b1, b2, b3,..., bn are the coefficients,  
    x1, x2, x3,...,xn are independent variables &  
    y is the dependent variable

### Dummy variable trap

Categorical data refers to data values which represent categories - data values with a fixed and unordered number of values, for instance gender (male/female) or season (summer/winder/spring/fall. In a regression model, these values can be represented by dummy variables - variables containing values such as 1 or 0 representing the presence or absence of the categorical value.

By including dummy variable in a regression model however, one should be careful of the Dummy Variable Trap. It is a scenario in which the independent variables are multicollinear - a scenario in which two or more variables are highly correlated; in simple terms one variable can be predicted from the others.

References:  
http://www.algosome.com/articles/dummy-variable-trap-regression.html  
https://marryingpython.blogspot.com/2018/07/dummy-trap-explained.html  
https://www.youtube.com/watch?v=qrWx3OjZL3o&t=321s  


### P-Value

P value is a statistical measure that helps scientists determine whether or not their hypotheses are correct. P values are used to determine whether the results of their experiment are within the normal range of values for the events being observed.

References:    
https://www.mathbootcamps.com/what-is-a-p-value/  
https://www.wikihow.com/Calculate-P-Value  


### Building a Model

While bulding a model, it is necessary to choose the right features to be included rather than including all of them. There are 5 methods of how to come up with the correct set of features and  build a multiple linear regression model:

1. **All-in**  
You include all the features. This happens when you have prior domain knowlege or when you are preparing for backward elimination.

2. **Backward Elimination**  

Step 1: Select a significance level to stay in the model (eg. SL = 0.05)  
Step 2: Fit the full model with all possible predictors  
Step 3: Consider the predictor with the highest P-value. If P > SL, go to Step 4, otherwise finish (model is ready)  
Step 4: Remove the predictor  
Step 5: Refit the model with the new set of predictors. Go to Step 3.  

3. **Forward Selection**

Step 1: Select a significance level to enter the model (eg. SL = 0.05)  
Step 2: Fit all simple regression models (y with x1, y with x2,y with x3,...., y with xn). Select the one with the lowest P-value  
Step 3: Keep this variable and fit all possible models with one extra predictor along with this selected variable. For example, if the selected variable is x1, fit all possible models (x1 & x2, x1 & x3, x1 & x4, and so on)  
Step 4: Consider the predictor with the lowest P-value. If P < SL, go to Step 3, otherwise finish (keep the previous model)