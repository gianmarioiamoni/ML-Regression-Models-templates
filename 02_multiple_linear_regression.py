import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('./data/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# NO NEED to apply FEATURES SCALING in linear Regression,
# as we have coefficients of the equation which compensate
# the weights of the values

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
#
# The sklearn library takes care of dummy variable trap 
# and of selection of the best fetures wiht the minimal P-value
# The class used for MLR is the same as for SLR
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# RFE implementation
from sklearn.feature_selection import RFE

# create an object of the class RFE
# and pass the regressor object
# and the number of features we want to select
selector = RFE(estimator=regressor, n_features_to_select=5)

# regressor.fit(X_train, y_train)

# apply RFE to the Training set
selector.fit(X_train, y_train)

# get the list of the selected features 
print(selector.support_)

# get the ranking of the selected features
print(selector.ranking_)

# Predicting the test set results
#
# In MLR we have several features instead of one.
# We cannot plot a graph like in SLR with features in X axis
# and dependant valiable in the Y axis.
# Instead, we diplay 2 vectors:
#   - the vector of the real dependant variable values (profit in the example) 
#     in the Test set
#   - the vector of the predicted dependant variable values 
#     of the same Test set
# So we can compare for each line of the Test set to see if 
# the predicted dependant variable is close to the real dependant variable value

# get the prediction vector
# y_pred = regressor.predict(X_test)
y_pred = selector.predict(X_test)
np.set_printoptions(precision=2)

# display the 2 vectors of the predicted values together
# we use numPy concatenate function
# vectors has to have the same shape and printed vertically 
# multiple_linear_regression.py(1 as the last argument of np.concatenate)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))