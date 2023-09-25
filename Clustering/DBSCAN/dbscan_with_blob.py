import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

centers = [[4, 7], [5, 3], [9, 12]]
X, Y = make_blobs(500, centers=centers, cluster_std=.9)

model = DBSCAN(eps=.8, min_samples=6)
model.fit(X)
