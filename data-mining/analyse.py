##############################################################
# Nom : Mohamed AIT MANSOUR									 #
# Source : M2 ILSEN - Avignon University					 #
# Données créé par : Iain Murray de l’université d’Édimbourg #
##############################################################

# librairies import
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

# read input text and put data inside a data frame
fruits = pd.read_csv('data/fruit_data_with_colors.txt', sep='\t')

# data head
print("\nFruits Head \n")
print( fruits .head())

# print nb of instances and features
print("\nNumber of instances and Features \n")
print( fruits .shape)

# print feature types
print("\nFeatures Types \n")
print( fruits .dtypes)

# print balance between classes
print("\nBalance between classes \n")
print( fruits .groupby('fruit_name').size())

# Declaration
cmap = cm.get_cmap('gnuplot')
fruits_name=['mass','width','height','color_score']
X=fruits[fruits_name]
y=fruits['fruit_label']

# Scatter Matrix
pd.plotting.scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9),cmap=cmap)

# Generate Hists
for attr in fruits_name:
	pd.DataFrame({k: v for k, v in fruits.groupby('fruit_name')[attr]}). plot. hist (stacked=True)
	plt . suptitle (attr)
	plt . savefig ('plot/fruits_histogram_'+attr)