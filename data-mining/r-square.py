from mpl_toolkits.mplot3d import Axes3D
from sklearn. cluster import KMeans
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from importlib.machinery import SourceFileLoader
R_square_clustering = SourceFileLoader("r_square", "scripts/R_square_clustering.py").load_module()

X = pd.read_csv('data/fruit_data_with_colors.txt', sep='\t')
X.drop(["fruit_name","fruit_subtype","fruit_label"], axis = 1, inplace = True) 

scaler = MinMaxScaler()
X_norm=scaler.fit_transform(X)


# metrique r2
lst_k=range(2,10)
lst_rsq = []
for k in lst_k:
	est=KMeans(n_clusters=k)
	est . fit (X_norm)
	lst_rsq.append(R_square_clustering.r_square(X_norm, est.cluster_centers_,est.labels_,k))

fig = plt. figure ()
plt . plot(lst_k, lst_rsq, 'bx-' )
plt . xlabel('k')
plt . ylabel('RSQ')
plt . title ('The Elbow Method showing the optimal k')
plt . savefig ('plot/k-Elbow-Method')
plt . close ( fig )