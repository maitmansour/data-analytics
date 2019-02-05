from mpl_toolkits.mplot3d import Axes3D
from sklearn. cluster import KMeans
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from importlib.machinery import SourceFileLoader
purity = SourceFileLoader("purity_score", "scripts/purity.py").load_module()
R_square_clustering = SourceFileLoader("r_square", "scripts/R_square_clustering.py").load_module()

X = pd.read_csv('data/fruit_data_with_colors.txt', sep='\t')
X.drop(["fruit_name","fruit_subtype"], axis = 1, inplace = True) 

scaler = MinMaxScaler()
X_norm=scaler.fit_transform(X)

y_true=X['fruit_label']
putiry_scores=[]
lst_k=range(2,11)
lst_rsq = []
for k in lst_k:
	est=KMeans(n_clusters=k)
	est . fit (X_norm)
	lst_rsq.append(R_square_clustering.r_square(X_norm, est.cluster_centers_,est.labels_,k))
	putiry_scores.append(purity.purity_score(y_true,est.labels_))

fig = plt. figure ()
plt . plot(lst_k, putiry_scores, 'bx-' )
plt . xlabel('Clusters')
plt . ylabel('Purity Score')
plt . title ('Purity Scores')
plt . savefig ('plot/purity-scores')
plt . close ( fig )
