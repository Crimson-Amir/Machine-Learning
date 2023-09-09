import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn import linear_model, preprocessing

cars = pd.read_csv('cars.csv')
data = cars[['fuel', 'doors', 'body', 'drive', 'weight', 'engine-size', 'bhp', 'mpg', 'price']].dropna()
p = 'price'

re = preprocessing.LabelEncoder()
list_update = ['fuel', 'doors', 'body', 'drive']
data[list_update] = data[list_update].apply(lambda s: re.fit_transform(s))

x = data.drop('price', axis=1).values.tolist()
y = data[p].values

best = 0
for _ in range(100000):
     
    liner = linear_model.LinearRegression()
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    liner.fit(x_train, y_train)
    acc = liner.score(x_test, y_test)

    if acc > best:
        best = acc
        with open('best_score.pickle', 'wb') as f:
            pickle.dump(liner, f)

print(best)
model = pickle.load(open('best_score.pickle', 'rb'))
result = model.predict(x_test)

for x in range(len(result)):
    print(f'{result[x]}  {x_test[x]}  {y_test[x]}')

