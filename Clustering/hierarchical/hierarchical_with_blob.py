from sklearn.datasets import make_blobs
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

centers = [[1, 5], [7, 4], [5, 9]]
X, Y = make_blobs(500, 2, centers=centers, cluster_std=.9)
plt.plot(X[:, 0], X[:, 1], '.')

minmax = MinMaxScaler()
minmax.fit(X)
new = minmax.transform(X)

model = AgglomerativeClustering(3)
model.fit(X)

model.n_clusters_
model.labels_
model.n_leaves_