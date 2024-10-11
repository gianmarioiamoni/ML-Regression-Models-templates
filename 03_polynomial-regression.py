import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./data/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the PR model on the whole dataset
#
# 1. We create the matrix of features at different powers 
#    using PolynomialFeatures class from preprocessing module 
#    of sklearn library
# 2. We integrate that into a linear regression model, 
#    as PRM is a linear combination of the powered features
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# create the matrix of features; degree = n
poly_reg = PolynomialFeatures(degree = 4)
# transform the matrix of single feature in a new matrix containing powered features
X_poly = poly_reg.fit_transform(X)

# Integrate the new fatures matrix into a linear regression model
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# Visualising the Polynomial Regression results
#
# When we use lin_reg2 regressor, It must be applied to the tranformed
# matrix of features X into the matrix of features at the different powers
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Polynomial Regression
#
# We predict the same salary with PRM
# We use the polynomial regressor
# the input will be the matrix of several features at different powers
# the input value is a 2D array containing 6.5
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))

