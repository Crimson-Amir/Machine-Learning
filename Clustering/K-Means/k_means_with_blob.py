import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

centers = np.array([[2, 5], [6, 3], [7, 9]])
X, Y = make_blobs(5000, centers=centers)
plt.plot(X[:, 0], X[:, 1], '.r')
plt.plot(centers[:, 0], centers[:, 1], 'og')

model = KMeans(3, n_init=12)
model.fit(X)

colors = plt.cm.Spectral(np.linspace(0, 1, len(set(model.labels_))))  # len(centers) also work and better
for k, color in zip(range(len(centers)), colors):
    dol = model.labels_ == k
    # print(model.cluster_centers_, k, X[dol, 0])
    plt.plot(X[dol, 0], X[dol, 1], '.', color)
    plt.plot(model.cluster_centers_[k, 0], model.cluster_centers_[k, 1], 'or')

a = {
    'car': ['camaro', 'mustang', 'chalenger'],
    'silander': [8, 6, 10]
}

le = LabelEncoder()
a['car'] = le.fit_transform(a['car'])

df = pd.DataFrame(a)
sk = StandardScaler().fit(df)
sk.transform(df)
