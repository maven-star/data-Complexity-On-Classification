import numpy as np
from fcmeans import FCM
import random, math
import pandas as pd


def f_cm(X, nc):                # fitting the fuzzy-c-means
    X=np.array(X)
    fcm = FCM(n_clusters=nc)
    fcm.fit(X)
    return fcm.predict(X)


def callamin(Data,Target,cluster_size):
    print("Data shape :",np.shape(Data))
    Cluster = f_cm(Data, cluster_size)  # seperating  Data with cluster_size
    ind = [i for i in range(cluster_size)]  # comparing the Data  and  seperate cluster_value
    clster_g = [[] for i in range(cluster_size)]
    Targ_g = [[] for i in range(cluster_size)]  # splitting of target value to the clustersize
    for i in range(len(Cluster)):
        for j in range(len(ind)):
            if (Cluster[i] == ind[j]):
                clster_g[ind[j]].append(Data[i])  # Appending the compared value to its appropriate array
                Targ_g[ind[j]].append(Target[i])  # Appending the compared value to its appropriate array




    return clster_g,Targ_g
