from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

Data = fetch_california_housing(as_frame=True)

X = Data['data']
y = Data['target']


model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# fit model, predict values and get mean squared error
model.fit(X_train, y_train)
pred = model.predict(X_test)
mean_squared_error(y_test, pred) ** 0.5

# adding polynomial features with 2 degree
pf = PolynomialFeatures(degree=2)
pf.fit(X_train)
X_train_new = pf.transform(X_train)
X_test_new = pf.transform(X_test)

# fit model after adding polynomial features, predict values and get mean squared error
model.fit(X_train_new, y_train)
pred2 = model.predict(X_test_new)
mean_squared_error(y_test, pred2) ** 0.5
