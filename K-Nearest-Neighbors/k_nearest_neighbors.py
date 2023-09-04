import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import sklearn
import pickle

df = pd.read_csv('dog.csv')

re = preprocessing.LabelEncoder()
df = df.apply(lambda s: re.fit_transform(s))

x = df.drop('Breed', axis=1).values
y = df['Breed'].values

# score = 0
# for _ in range(10000):
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.01)
#
#     model = KNeighborsClassifier()
#     model.fit(x_train, y_train)
#
#     acc = model.score(x_test, y_test)
#
#     if acc > score:
#         score = acc
#         with open('knn.pickle', 'wb') as f:
#             pickle.dump(model, f)
#
# print(score)

model = pickle.load(open('knn.pickle', 'rb'))
predict = model.predict(x_test)

for a in range(len(predict)):
    print(predict[a], x_test[a], y_test[a])
