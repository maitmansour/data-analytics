#!/usr/bin/env python
# -*- coding: utf-8 -*-

# librairies import
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage

# Correlation_circle
def correlation_circle(df,nb_var,x_axis,y_axis):
    fig, axes = plt.subplots(figsize=(8,8))
    axes.set_xlim(-1,1)
    axes.set_ylim(-1,1)
    # label with variable names
    for j in range(nb_var):
        # ignore two first columns of df: Nom and Code^Z
        plt.annotate(df.columns[j+2],(corvar[j,x_axis],corvar[j,y_axis]))
    # axes
    plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
    # add a circle
    cercle = plt.Circle((0,0),1,color='blue',fill=False)
    axes.add_artist(cercle)
    plt.savefig('plot/acp_correlation_circle_axes_'+str(x_axis)+'_'+str(y_axis))
    plt.close(fig)

# K-means clustering
def kmeans_clustering(data,max_classes):
    del data['Nom']
    del data['Code']
    sum_of_se = {}
    for k in range (1, max_classes):
      kmeans_model = KMeans(n_clusters=k, random_state=1).fit(data.iloc[:, :])
      labels = kmeans_model.labels_
      sum_of_se[k] = kmeans_model.inertia_
    plt.figure()
    plt.plot(list(sum_of_se.keys()), list(sum_of_se.values()))
    plt.xlabel("Number of clusters")
    plt.ylabel("Sum of Squared errors")
    plt.savefig('plot/k-means_clusters.png')

def hierarchical_clustering(data):
    Z = linkage(data, 'ward')
    c, coph_dists = cophenet(Z, pdist(Df))
    plt.figure(figsize=(8, 8))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Country number')
    plt.ylabel('Distances')
    dendrogram(Z,leaf_rotation=90.,leaf_font_size=8.,)
    plt.savefig('plot/hierarchical_clustering_dendogram.png')


# read input text and put data inside a data frame
data = pd.read_csv('data/eurostat/eurostat-2013.csv')

# data head
print("\nEurostat Head \n")
print( data .head())

# data/population
data["tet00002 (2013)"]=data["tet00002 (2013)"]/data["tps00001 (2013)"]
data["tsc00004 (2012)"]=data["tsc00004 (2012)"]/data["tps00001 (2013)"]

# remove population data
del data["tps00001 (2013)"]

print("\nEurostat Head \n")
print( data .head())

# StandarScaller Standardize X_norm by removing the mean and scaling to unit variance (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)
# We have to standrize our data, because there is a very large defirence between data (eg. 0.3 and -5 at tec00115 AT and CY)

scaler = StandardScaler()
scaled_X_norm = data.copy()

col_names = ["tec00115 (2013)","teilm (F dec 2013)","teilm (M dec 2013)","tec00118 (2013)","teimf00118 (dec 2013)","tsdsc260(2013)","tet00002 (2013)","tsc00001 (2011)","tsc00004 (2012)"]
X_norm = scaled_X_norm[col_names]
scaler = StandardScaler().fit(X_norm.values)
X_norm = scaler.transform(X_norm.values)

print("\nEurostat Head Standrized\n")
scaled_X_norm[col_names] = X_norm
print(scaled_X_norm)

# Get Principal Components
acp = PCA(n_components=4)
principal_components=acp.fit_transform(X_norm)
print(principal_components)

y=['Principal Component 1', 'Principal Component 2', 'Principal Component 3', 'Principal Component 4']
acpDf = pd.DataFrame(data = principal_components, columns =y )
finalDf = pd.concat([acpDf, data[['Code']]], axis = 1)
Df=acpDf.astype(float)

# Save Principal Components
g=sns.lmplot("Principal Component 1","Principal Component 2",hue='Code',data=finalDf,fit_reg=False,scatter=True,height=7)
plt.savefig('plot/principal_component_1_and2.png')

g=sns.lmplot("Principal Component 3","Principal Component 4",hue='Code',data=finalDf,fit_reg=False,scatter=True,height=7)
plt.savefig('plot/principal_component_3_and4.png')

# Get variance
variance= acp.explained_variance_ratio_

# Get Correlation between variables 
eigenvalues = 4/(4*acp.explained_variance_ratio_)
sqrt_eigenvalues = np.sqrt(eigenvalues)
corvar = np.zeros((9,9))
for k in range(4):
    corvar[:,k] = acp.components_[k,:] * sqrt_eigenvalues[k]
print("\nCorrelation between variables\n")
print(corvar)

# Draw correlation circle
correlation_circle(nb_var=4,df=data,x_axis=0,y_axis=1)

# K-means clustering
kmeans_clustering(data=data,max_classes=15)

# Hierarchical Clustering Dendrogram
hierarchical_clustering(data=data)