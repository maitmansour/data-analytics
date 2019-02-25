#!/usr/bin/env python
# -*- coding: utf-8 -*-

# librairies import
import pandas as pd
from sklearn.preprocessing import StandardScaler

#correlation_circle
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
    plt.savefig('acp_correlation_circle_axes_'+str(x_axis)+'_'+str(y_axis))
    plt.close(fig)



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


# StandarScaller Standardize features by removing the mean and scaling to unit variance (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)
# We have to standrize our data, because there is a very large defirence between data (eg. 0.3 and -5 at tec00115 AT and CY)

scaler = StandardScaler()
columns_to_standrize=['','','','','','','']
scaled_features = data.copy()

col_names = ["tec00115 (2013)","teilm (F dec 2013)","teilm (M dec 2013)","tec00118 (2013)","teimf00118 (dec 2013)","tsdsc260(2013)","tet00002 (2013)","tsc00001 (2011)","tsc00004 (2012)"]
features = scaled_features[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)

print("\nEurostat Head Standrized\n")
scaled_features[col_names] = features
print(scaled_features)
# print centroids associated with several countries
#lst_countries=['EL','FR','DE','US']
## centroid of the entire dataset
## est: KMeans model fit to the dataset
#print est.cluster_centers_
#for name in lst_countries:
#    num_cluster = est.labels_[y.loc[y==name].index][0]
#    print 'Num cluster for '+name+': '+str(num_cluster)
#    print '\tlist of countries: '+', '.join(y.iloc[np.where(est.labels_==num_cluster)].values)
#    print '\tcentroid: '+str(est.cluster_centers_[num_cluster])

