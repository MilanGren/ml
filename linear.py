#!/usr/bin/env python3.5

from sklearn import datasets, linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

diabetes = datasets.load_diabetes()

#print(diabetes.keys())

X_train = diabetes.data[:-20][:,np.newaxis,2]
X_test  = diabetes.data[-20:][:,np.newaxis,2]

y_train = diabetes.target[:-20]
y_test  = diabetes.target[-20:]


regr = linear_model.LinearRegression()

regr.fit(X_train,y_train)

y_pred = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))


#OBRAZKY
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

