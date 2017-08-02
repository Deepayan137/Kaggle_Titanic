import pandas as pd  
import os
import sys
import numpy as np 
from scipy.spatial import distance
from sklearn import preprocessing
from collections import Counter
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
print("Shape of train data: %s %s"%(X_train.shape[0], X_train.shape[1]))
print("Shape of test data: %s %s"%(X_test.shape[0], X_test.shape[1]))

print(X_train.head())

def distance_matrix(X_train, X_test):
	dist_mat = np.zeros((X_train.shape[0], X_test.shape[0]))
	#print(dist_mat)
	print('Computing Distances')
	for i in range(X_train.shape[0]):
		for j in range(X_test.shape[0]):
			dist_mat[i,j] = np.sqrt(np.sum(X_train.iloc[i].values - X_test.iloc[j].values)**2)
	return dist_mat
def vote(X_s):
	pos =[]
	
	one = 0
	zero = 0
	for j in range(len(X_s)):
		if y_train[X_s[j]] == 1:
			one+=1
		else:
			zero+=1
	if one>zero:
		return 1
	else:
		return 0


def prediction(k):
	print('making predictions')
	dist_mat = distance_matrix(norm_X_train, norm_X_test)
	X_smallest=[]
	for j in range(dist_mat.shape[1]):
		X_smallest.append(np.argsort(dist_mat[:,j])[:k])
	pred = [vote(X_smallest[i]) for i in range(len(X_smallest))]
	
	return(pred)

y_test = prediction(7)
print('Done.....')
data = pd.DataFrame({'PassengerId':X_test['PassengerId'] , 'Survived': y_test}, columns=['PassengerId', 'Survived'])
data.set_index(['PassengerId'])
data.to_csv('Submission.csv')