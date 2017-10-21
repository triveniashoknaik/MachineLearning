#MachineLearning
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:41:24 2017

@author: TriveniAshok
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

#training sets for genuine and imposter 
X_train1 = pd.read_csv("tr1gLA.csv" , header=None)
X_train2 = pd.read_csv("tr1iLA.csv" , header=None)

#concatenating the two training sets into one
frames = [X_train1,X_train2]
X_train = pd.concat(frames)

#generating the label for genuine and imposter
df = pd.read_csv('tr1gLA.csv' , header=None)
df['Label']='1'
df1 = pd.read_csv('tr1iLA.csv' , header=None)
df1['Label']='0'

#concatenating the labels
frames1 = [df,df1]
ddf = pd.concat(frames1)

#represents y_train with labels
y_train = ddf.iloc[: , 40].values

#testing sets for genuine and imposter 
X_test1 = pd.read_csv("ts1gLA.csv" , header=None)
X_test2 = pd.read_csv("ts1iLA.csv" , header=None)

#concatenating the two testing sets into one
frames = [X_test1,X_test2]
X_test = pd.concat(frames)

#generating the label for genuine and imposter
df = pd.read_csv('ts1gLA.csv' , header=None)
df['Label']='1'
df1 = pd.read_csv('ts1iLA.csv' , header=None)
df1['Label']='0'

#concatenating the labels
frames1 = [df,df1]
ddf = pd.concat(frames1)

#represents y_train with labels
y_test = ddf.iloc[: , 40].values

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred_prob = classifier.predict_proba(X_test)

#ConfusionMatrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#predicting the area under the curve
roc_auc_score(np.int_(y_test), y_pred_prob[:,1])
# y_t = np.array([0, 0, 1, 1])
