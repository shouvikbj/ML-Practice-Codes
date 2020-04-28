import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use('ggplot')


centers = [[1,1,1],[5,5,5],[3,10,10]]
x,_ = make_blobs(n_samples=200, centers=centers, cluster_std=1)

ms = MeanShift()
ms.fit(x)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters :",n_clusters_)

colors = 10*['r','y','b','c','k','g','m']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(x)):
    ax.scatter(x[i][0], x[i][1], x[i][2], c=colors[labels[i]], marker='o', s=10)

ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],
            marker='x',color='k',s=150,linewidths=5,zorder=10)

plt.show()


