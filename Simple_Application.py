from Partie_1 import *
from sklearn.cluster import KMeans

""" Partie2 :Creatiion de base de données Artificielle"""

import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
#2
X,y=make_blobs(n_samples=600,n_features=2,centers=3)
#3
plt.scatter(X[:,0],X[:,1])
plt.show()

#5
VX=Validity_Index(X)
c_opt=max(VX)

#6
kmeans = KMeans(n_clusters=VX.index(c_opt)+2)
kmeans.fit(X)

#7
Cluster_centers = kmeans.cluster_centers_
pred_X = kmeans.fit_predict(X)


#8
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
plt.show()
