import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv('churn.csv')
df = df.drop(['State', 'Account Length', 'Phone'], axis=1)

le = LabelEncoder()
lists = ["Int'l Plan",'VMail Plan']
df[lists] = df[lists].apply(lambda x: le.fit_transform(x))

le2 = LabelEncoder()
le2.fit(['True.', 'False'])
df['Churn?'] = le2.transform(df['Churn?'])

X = df[['Area Code', "Int'l Plan", "VMail Plan", "Intl Calls", "Intl Charge", "CustServ Calls"]].astype(int)
Y = df['Churn?']

X = StandardScaler().fit(X).transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
model = LogisticRegression()
model.fit(x_train, y_train)

# predict = model.predict_proba(x_test)
# print(predict[:10])
predict = model.predict(x_test)
for _ in range(len(predict[:5])):
    print(x_test[_], predict[_], y_test.values[_])

print(model.score(x_test, y_test))
print(accuracy_score(y_test, predict))
print(confusion_matrix(y_test, predict, labels=[1, 0]))
print(classification_report(y_test, predict))