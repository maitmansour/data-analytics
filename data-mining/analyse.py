##############################################################
# Nom : Mohamed AIT MANSOUR									 #
# Source : M2 ILSEN - Avignon University					 #
# Données créé par : Iain Murray de l’université d’Édimbourg #
##############################################################

# librairies import
import pandas as pd

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