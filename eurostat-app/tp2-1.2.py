#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from scipy.io import arff
import pandas as pd

# read data from file
data = arff.loadarff('data/pull/vote.arff')
vote = pd.DataFrame(data[0])


# convert categorical values into one-hot vectors and ignore ? values
# corresponding to missing values
# ex: handicapped-infants=y -> [1,0], handicapped-infants=n -> [0,1], handicapped-infants=? -> [0,0]

# Que proposez-vous pour tenir compte de cette absence de valeurs pour ne pas voir apparaitre des règles non pertinentes ?
# Afin d'éviter l'apparence des régles non pertinantes, on va supprimer les données dont la classe est absente
vote_one_hot = pd.get_dummies(vote)
vote_one_hot.drop(vote_one_hot.filter(regex='_\?$',axis=1).columns,axis=1,inplace=True)

# Affichage des données
#print(vote_one_hot)

# item_sets contienne l'ensemble des items set, avec le numero du ligne qui donne cet itemset, 
# le nombre des item sets qu'on a obtenu est : 118
#Donnez leur statistique en fonction du nombre d'items qu'ils contiennent 


frequent_itemsets=apriori(vote_one_hot, min_support=0.4)

print(frequent_itemsets)


# ici, on affiche plus de détails sur les item sets
frequent_itemsets_details = apriori(vote_one_hot, min_support=0.4, use_colnames=True)
frequent_itemsets_details['length'] = frequent_itemsets_details['itemsets'].apply(lambda x: len(x))



#calculer les itemsets avec apriori
#calculer les itemset frequet freequent_item_sets = apriori(vote one hot)




# check that there is no rule implying Republicans
#filter(lambda x: "Class_'republican'" in x,rules['antecedents'])
#filter(lambda x: "Class_'republican'" in x,rules['consequents'])

