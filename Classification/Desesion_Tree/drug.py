import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pydotplus
from sklearn.metrics import classification_report

df = pd.read_csv('drug200.csv')

le = LabelEncoder()
col = ['Sex', 'BP', 'Cholesterol', 'Drug']
df[col] = df[col].apply(lambda x: le.fit_transform(x))

X = df.drop('Drug', axis=1).values
Y = df['Drug'].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

tree = DecisionTreeClassifier(criterion='entropy')
tree.fit(x_train, y_train)

a = tree.predict(x_test)
for _ in a[:5]:
    print(x_test[_], a[_], y_test[_])

print(tree.score(x_test, y_test))

print(classification_report(y_test, a))