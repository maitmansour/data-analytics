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

ridge = Ridge()
ridge.fit(X, Y)

coeff = ridge.coef_
print('\nModel :\n')
print(coeff)


prediction_data = [[3198, 9669, 5, 1, 1]]
predicted_price = ridge.predict(prediction_data)
print("Real price : 217894$\nPre. price : {:6.0f}$".format(predicted_price[0]))