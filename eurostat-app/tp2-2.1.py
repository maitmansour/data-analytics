#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

dummycl = DummyClassifier(strategy="most_frequent")
gmb = GaussianNB()
dectree = tree.DecisionTreeClassifier()
logreg = LogisticRegression(solver="liblinear")
svc = svm.SVC(gamma='scale')

lst_classif = [dummycl, gmb, dectree, logreg, svc]
lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Logistic regression', 'SVM']

def accuracy_score(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        scores = cross_val_score(clf, X, y, cv=5)
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def confusion_matrix(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        predicted = cross_val_predict(clf, X, y, cv=5) 
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f" % metrics.accuracy_score(y, predicted))
        print(metrics.confusion_matrix(y, predicted))
        
# Load data
data = arff.loadarff('data/eurostat/crx.arff')
crx = pd.DataFrame(data[0])

# Replace missing values by mean and scale numeric values
data_num = crx.select_dtypes(include='float64')

# Replace missing values by mean and discretize categorical values
data_cat = crx.select_dtypes(exclude='float64').drop('class',axis=1)

# Get data Shape
print("Shape : ",crx.shape)
# Shape :  (690, 16)
# Attribute to predict : class
# Types : numeric, categorial

# Equilibrate class (negative,positive) data
occurences_classes=crx['class'].value_counts()
print("\n Frequency : \n",occurences_classes)
#'negative'    383
#'positive'    307

# Normlize numerique data
scaler = StandardScaler();
scaler.fit(data_num) 
data_num = scaler.transform(data_num)

# Replace nan data (MEAN)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(data_num)
data_num = imputer.transform(data_num)

# Transform categorial data
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(crx["class"])

# Calcul et affichage des scores en appelant la methode accuracy_score fournit dans le tp
print('\n Accuracy Score  : \n')
accuracy_score(lst_classif,lst_classif_names,data_num,Y)

#Accuracy of Dummy classifier on cross-validation: 0.56 (+/- 0.00)
#Accuracy of Naive Bayes classifier on cross-validation: 0.71 (+/- 0.08)
#Accuracy of Decision tree classifier on cross-validation: 0.68 (+/- 0.10)
#Accuracy of Logistic regression classifier on cross-validation: 0.75 (+/- 0.09)
#Accuracy of SVM classifier on cross-validation: 0.74 (+/- 0.10)

# Replace none values with most frequent values
imputer= SimpleImputer(missing_values="'none'", strategy='most_frequent')
data_cat = imputer.fit_transform(data_cat)

# Replace ? values with most frequent values
imputer = SimpleImputer(missing_values='?', strategy='most_frequent')
data_cat = imputer.fit_transform(data_cat)

# Data descritization
categorical_columns=["A1","A4","A5","A6","A7","A9","A10","A12","A13"]
data_cat = pd.DataFrame(data_cat, columns=categorical_columns);
data_cat = pd.get_dummies(data_cat)

#Calcul des scores pour les donnees categorical
print('\n Accuracy Score  : \n')
accuracy_score(lst_classif,lst_classif_names,data_cat,Y)
#Accuracy of Dummy classifier on cross-validation: 0.56 (+/- 0.00)
#Accuracy of Naive Bayes classifier on cross-validation: 0.64 (+/- 0.11)
#Accuracy of Decision tree classifier on cross-validation: 0.81 (+/- 0.22)
#Accuracy of Logistic regression classifier on cross-validation: 0.84 (+/- 0.26)
#Accuracy of SVM classifier on cross-validation: 0.85 (+/- 0.28