from sklearn import svm, datasets, metrics
import sklearn

data = datasets.load_breast_cancer()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)

predict = model.predict(x_test)
acc = metrics.accuracy_score(y_test, predict)

print(acc)
