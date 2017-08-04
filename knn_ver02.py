from sklearn.neighbors import KNeighborsClassifier
import pandas as pd  
import os
import sys
import numpy as np 
from scipy.spatial import distance
from sklearn import preprocessing
from collections import Counter
sys.path.insert(0, '../titanic/')
neigh = KNeighborsClassifier(n_neighbors=9,algorithm ='kd_tree', weights='distance')
sys.path.insert(0, '../titanic/')
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)
X_train = train_data[train_data.columns.difference(['Survived', 'Name', 'Cabin','Ticket', 'Embarked'])]

y_train = train_data['Survived']

X_test = test_data[test_data.columns.difference(['Survived', 'Name', 'Cabin', 'Ticket', 'Embarked'])]
X_train.loc[X_train['Sex'] == 'female', 'Sex'] = 1
X_train.loc[X_train['Sex'] == 'male', 'Sex'] = 0
X_test.loc[X_test['Sex'] == 'female', 'Sex'] = 1
X_test.loc[X_test['Sex'] == 'male', 'Sex'] = 0
min_max_scaler = preprocessing.MinMaxScaler()
norm_X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train.values))
norm_X_test = pd.DataFrame(min_max_scaler.fit_transform(X_test.values))	

neigh.fit(norm_X_train, y_train,)
y_test = neigh.predict(X_test)
data = pd.DataFrame({'PassengerId':X_test['PassengerId'] , 'Survived': y_test}, columns=['PassengerId', 'Survived'])
data.set_index(['PassengerId'], inplace = True)
print(data.head())
data.to_csv('Submission_new.csv')
