#!/usr/bin/env python3.5


from sklearn import datasets, linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class X:
  def __init__(self,model_name,y_pred,X_test,y_test):
    self.model_name  = model_name
    self.y_pred = y_pred
    self.X_test = X_test
    self.mean_squared_error = mean_squared_error(y_test, y_pred)

def get(model_name,X_train,y_train,X_test,y_test):
  regr = linear_model.LinearRegression()  

  if model_name == 'lin':
    regr = linear_model.LinearRegression()
  elif model_name == 'rid':
    regr = linear_model.Ridge(alpha = 0.5)
  
  regr.fit(X_train,y_train)
#  print('Coefficients: \n', regr.coef_)  
  y_pred = regr.predict(X_test)
  return X(model_name,y_pred,X_test,y_test)
   
diabetes = datasets.load_diabetes()

#print(diabetes.keys())

X_train = diabetes.data[:-20][:,np.newaxis,2] #.tolist()
X_test  = diabetes.data[-20:][:,np.newaxis,2]

y_train = diabetes.target[:-20]
y_test  = diabetes.target[-20:] #.tolist()

lin = get('lin',X_train,y_train,X_test,y_test)
rid = get('rid',X_train,y_train,X_test,y_test)

#print('Variance score: %.2f' % r2_score(y_test, y_pred))

#OBRAZKY
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

f = plt.figure()
f.suptitle('different linear models')
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, lin.y_pred, color='blue', linewidth=3, label="%-s: %-f" % (lin.model_name,lin.mean_squared_error))
plt.plot(X_test, rid.y_pred, color='red',  linewidth=3, label="%-s: %-f" % (rid.model_name,rid.mean_squared_error))
plt.legend(loc='upper left')
f.savefig('plot.pdf')

