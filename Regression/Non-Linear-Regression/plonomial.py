import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('cars.csv')
df = df[['Dimensions.Height', 'Fuel Information.Highway mpg', 'Fuel Information.City mpg']]

ra = np.random.rand(len(df)) < 0.8
train = df[ra]
test = df[~ra]

x_train = train[['Fuel Information.City mpg']].values
y_train = train[['Fuel Information.Highway mpg']].values
polynomial = PolynomialFeatures(degree=2)
x_poly_train = polynomial.fit_transform(x_train)

model = LinearRegression()
model.fit(x_poly_train, y_train)
print(model.coef_)
print(model.intercept_)

plt.plot(x_train , y_train, '.b')
plt.plot(x_train, model.predict(x_poly_train), '.r')
