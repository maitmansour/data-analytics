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
vote_one_hot = pd.get_dummies(vote)
vote_one_hot.drop(vote_one_hot.filter(regex='_\?$',axis=1).columns,axis=1,inplace=True)
print(vote_one_hot)


#calculer les itemsets avec apriori
#calculer les itemset frequet freequent_item_sets = apriori(vote one hot)




# check that there is no rule implying Republicans
#filter(lambda x: "Class_'republican'" in x,rules['antecedents'])
#filter(lambda x: "Class_'republican'" in x,rules['consequents'])

