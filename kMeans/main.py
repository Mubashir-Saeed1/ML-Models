import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

mean1 = [np.random.randint(50), np.random.randint(50)]
mean2 = [np.random.randint(50), np.random.randint(50)]
mean3 = [np.random.randint(50), np.random.randint(50)]

cova = [[100, 0], [0, 100]]

x1, y1 = np.random.multivariate_normal(mean1, cova, 100).T
x2, y2 = np.random.multivariate_normal(mean2, cova, 100).T
x3, y3 = np.random.multivariate_normal(mean3, cova, 100).T

x = np.append(x1, x2)
y = np.append(y1, y2)
z = np.append(x3, y3)

plt.plot(x, y, 'x')
plt.plot(x, z, 'x')
plt.axis('equal')
plt.show()

zipped = zip(x, y, z)
listedZip = list(zipped)

X = np.array(listedZip)

kMeans = KMeans(n_clusters=3)
fittedModel = kMeans.fit(X)

centroids = kMeans.cluster_centers_
labels = kMeans.labels_

print('Centroids: ', centroids)
print('Labels', labels)

colors = ['g.', 'r.', 'b.']
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]])
    plt.plot(X[i][0], X[i][2], colors[labels[i]])

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x',  s=150, zorder = 10)
plt.scatter(centroids[:, 0],  centroids[:, 2], marker='x', s=150, zorder = 10)
plt.show()

print('Centroids: ', centroids)
print('Means: ', mean1, mean2, mean3)
