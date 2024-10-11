import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('./data/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Simple Linear Regression model on the Training set
#
# Build and train the Simple Linear Regression model
# Regression: It is used to predict a continuous real value
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
#
# Using the predict method
y_pred = regressor.predict(X_test)

# Visualising the Training set results
# 
# Plot a chart with matplotlib
# Y axis is salary
# x axis is years of experience
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

# Plot the regression line
# Is the line of the predictions coming as close as possible
# to the real results
# - first argument is the train set value
# - second arguments is the predicted Train set results of the training values
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

plt.show()

# Making a single prediction
#
# Print the salary of an employee with 12 years of experience
# the value of the feature (12 years) was input in a double pair of square brackets. 
# That's because the "predict" method always expects a 2D array as the format 
# of its inputs
# 12 -> scalar
# [12] -> 1D array
# [[12]] -> 2D array
print(regressor.predict([[12]]))

# Getting the final linear regression equation with the values of the coefficients
r_sq = regressor.score(X, y)
print('coefficient of determination:', r_sq)

# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)