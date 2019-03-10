#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from scipy.io import arff
from sklearn.linear_model import Ridge,LinearRegression

# Load data
data 	= arff.loadarff('data/house/house.arff')
data 	= pd.DataFrame(data[0])
X			= data.ix[:,['houseSize','lotSize','bedrooms','granite','bathroom']]
Y			= data.ix[:,'sellingPrice']
print(data)

# Get Model
ridge = Ridge()
ridge.fit(X, Y)
model = ridge.coef_
print('\nModel :\n')
print(model)
#Model :
#[-4.93635922e+00  7.99491521e+00  2.27757080e+04 -2.79564039e+02
#  2.17451778e+04]

# Predict house price
prediction_data = [[3198, 9669, 5, 1, 1]]
predicted_price = ridge.predict(prediction_data)
print("Real price : 217894$\nPre. price : {:6.0f}$".format(predicted_price[0]))
#Real price : 217894$
#Pre. price : 212742$