import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def sigmoid(x, beta_1, beat_2):
    return 1 / (1 + np.exp(beta_1*(x - beat_2)))


x = np.arange(0, 100)
y = np.power(x, 2)

x_data = x / x.max()
y_data = y / y.max()

b, r = curve_fit(sigmoid, x_data, y_data)
y_train = sigmoid(x_data, *b)

plt.plot(x_data, y_train)
plt.plot(x_data, y_data, 'r')

dd = 10 / x.max()
dd = sigmoid(dd, *b)
dd * np.max(y)
