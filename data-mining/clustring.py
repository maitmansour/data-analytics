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

# read input text and put data inside a data frame
fruits = pd.read_csv('data/fruit_data_with_colors.txt', sep='\t')
X = pd.read_csv('data/fruit_data_with_colors.txt', sep='\t')
X.drop(["fruit_name","fruit_subtype","fruit_label"], axis = 1, inplace = True) 

scaler = MinMaxScaler()
X_norm=scaler.fit_transform(X)

# TODO : enlever label, et appliquer un filtre de normalisation pour obtenir x_norm

# Plot clusters
lst_kmeans = [KMeans(n_clusters=n) for n in range(3,6)]
titles = [str(x)+'clusters 'for x in range(3,6)]
fignum = 1
for kmeans in lst_kmeans:
	fig = plt. figure (fignum, figsize =(8, 6))
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	kmeans.fit(X_norm)
	labels = kmeans.labels_
	ax. scatter (X['mass'], X['width'], X['color_score'],c=labels.astype(np.float), edgecolor='k')

	ax.w_xaxis.set_ticklabels ([])
	ax.w_yaxis.set_ticklabels ([])
	ax.w_zaxis.set_ticklabels ([])
	ax.set_xlabel('mass')
	ax.set_ylabel('width')
	ax.set_zlabel('color_score')
	ax. set_title ( titles [fignum - 1])
	ax. dist = 12
	plt . savefig ('plot/k-means_'+str(2+fignum)+'_clusters')
	fignum = fignum + 1
	plt . close ( fig )

# Plot the ground truth
fig = plt. figure (fignum, figsize =(8, 6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
for label in fruits [ 'fruit_name'].unique():
	ax.text3D(fruits .loc[ fruits [ 'fruit_name']==label].mass.mean(),
	fruits .loc[ fruits [ 'fruit_name']==label].width.mean(),
	fruits .loc[ fruits [ 'fruit_name']==label].color_score.mean(),
	label ,
	horizontalalignment='center',
	bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))

y=fruits['fruit_label']
ax. scatter (X['mass'], X['width'], X['color_score'], c=y, edgecolor='k')
ax.w_xaxis.set_ticklabels ([])
ax.w_yaxis.set_ticklabels ([])
ax.w_zaxis.set_ticklabels ([])
ax.set_xlabel('mass')
ax.set_ylabel('width')
ax.set_zlabel('color_score')
ax. set_title ('Ground Truth')
ax. dist = 12
plt . savefig ('plot/k-means_ground_truth')
plt . close ( fig )

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