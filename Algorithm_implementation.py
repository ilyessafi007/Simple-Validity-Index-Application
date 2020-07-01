""" Partie1 :Développement de l’algorithme de Validity Index"""
import numpy as np
from sklearn.cluster import KMeans


#a Between Scatter matrix
def BSX(i,X,Cluster_centers):
    mean_1 = np.mean(X[:,0])
    mean_2 = np.mean(X[:,1])
    mean_X= np.array([[mean_1],[mean_2]])
    bsx=np.zeros((2,2))
    n=X.shape[0]/i
    for j in range(i):
        bsx += n*(Cluster_centers[j].reshape(2,1)-mean_X).dot((Cluster_centers[j].reshape(2,1)-mean_X).T)
    return bsx

#b Within Scatter matrix

def WSX(i,X,Cluster_centers,pred_X):
    Cluster_center = Cluster_centers[i]
    wsx=np.zeros((2,2))
    c=0
    for j in X:
        if pred_X[c] == i:
            wsx+= (j.reshape(2,1)-Cluster_center).dot((j.reshape(2,1)-Cluster_center).T)
        c+=1
    return wsx




#c Algorithme de Validity Index
def Validity_Index(X):
    vsc=[]
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X)
        Cluster_centers = kmeans.cluster_centers_
        pred_X = kmeans.fit_predict(X)
        bsx = BSX(i, X,Cluster_centers)
        Sep_i = np.trace(bsx)
        Comp_i = 0
        for j in range(i):
            wsx = WSX(j, X, Cluster_centers,pred_X)
            trc = np.trace(wsx)
            Comp_i += trc
        Vsc = Sep_i / Comp_i
        if i ==3:
            print(Sep_i,Comp_i)
        vsc.append(Vsc)
    return vsc
