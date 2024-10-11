import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./data/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
#
# We don't apply feature regression on DTR model: the predictions
# from the decision tree are resulting from successive splits of the data,
# throught the nodes of the tree and therefore there are not
# some equiations like with other models. the split can be done 
# with the orignal values, even if they are in different scales
#
# The only difference with RTM is that the predictions are made
# by using a whole set of trees, specified by the n_estimators parameter
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 10, random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])

print(y_pred)